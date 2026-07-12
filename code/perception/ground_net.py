"""
code/perception/ground_net.py — GROUND_NET learned-grounding backend (RF-1
split of code/grounding.py; docs/nx6_judge.md, docs/nx6_train_heatmap.md,
docs/nx7_adoption.md, docs/vf1_showpiece.md).

NX-6 INTEGRATION: the query-conditioned heatmap detector NX-6 JUDGE selected
(`runs/nx6_heatmap_B/model_best.pt`), swapped in for the classical HSV+depth
pipeline (code/perception/hsv_pipeline.py) when GROUND_NET=1 (default ON,
docs/nx9_avoid.md). Same (dist,bearing)+confidence contract
(code.perception.types.GroundingResult) as the classical path, fed the SAME
active-camera RGBD frame any given call site already rendered this cycle, and
the query built from the SAME (target_color, target_shape) instruction-target
spec every call site already threads through `ground()`'s signature.

Module-state ownership (docs/refactor_plan.md invariant 5): every function
here is parameterized by an explicit `GroundNetState` instance rather than
touching bare module globals, so the SINGLE singleton instance constructed by
code/perception/grounding.py's dispatch module is the one and only place this
backend's mutable state (detector cache, NX-7 track state, one-shot notice
flags, latency log, VF-1 render cache) actually lives -- coherent by
construction across the old `code.grounding` alias and the new
`code.perception.grounding` path alike (both resolve to the exact same
module object, which owns the exact same `GroundNetState` instance).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from code.perception.lock_gate import (M3_GATE_BEARING_DEG, M3_GATE_BEARING_NEAR_MULT,
                                       M3_GATE_DIST_CLOSING_MULT, M3_GATE_DIST_FLOOR_M,
                                       M3_EXPECTED_CLOSING_M_PER_CYCLE, M3_NEAR_RANGE_M,
                                       _ang_diff_rad)
from code.perception.types import GroundingResult


@dataclass
class GroundNetState:
    """Mutable state for one process's GROUND_NET backend -- constructed exactly
    once (as a module-level singleton) by code/perception/grounding.py."""

    detector:    Any        = None     # lazy-loaded singleton (checkpoint loads once per process)
    load_failed: bool       = False    # sticky: don't retry a load every cycle after a failure
    class_names: list | None = None    # populated on first successful load (CLASS_NAMES)
    color_names: list | None = None    # populated on first successful load (COLOR_NAMES)
    widefov_warned:   bool  = False    # one-shot warning: widefov cam_type is untested/unsupported
    fallback_warned:  bool  = False    # NX-9: one-shot notice for the ckpt-missing classical fallback
    optout_notified:  bool  = False    # VR-1: one-shot notice when GROUND_NET=0 explicitly disables the detector
    lat_ms: list = field(default_factory=list)   # per-cycle inference latency (ms), module-level log

    # NX-7 FIX B track state (dist_m, bearing_rad) of the last ACCEPTED detection
    # (via either tau_acquire or tau_track+continuity) -- None when no track is
    # live. Must be reset at the start of every episode (see reset_track()) to
    # avoid one episode's last detection spuriously validating an unrelated
    # low-confidence blob at the start of the next episode in the same
    # long-lived eval process.
    track_dist_m:      float | None = None
    track_bearing_rad: float | None = None

    # VF-1 (docs/vf1_showpiece.md): render-side-only cache of the last GROUND_NET
    # grounding cycle's confidence heatmap + this-cycle decision, so a caller can
    # display it (fancy_demo.py's detector-heatmap overlay) with ZERO extra
    # inference (reuses the SAME forward pass this backend already ran). Never
    # read by any control-flow code path -- see get_last_heatmap().
    last_heatmap: dict | None = None


def get_last_heatmap(state: GroundNetState) -> dict | None:
    """
    VF-1 pure-read accessor: the last GROUND_NET grounding cycle's cached
    confidence heatmap, keyed to the query it was computed for.

    Render-side only -- never read by any control-flow code path.

    Returns:
        None if GROUND_NET was never invoked (or is off) in this process,
        or the detector failed to load. Otherwise a dict:
          prob:       (H,W) float32 sigmoid confidence map in [0,1], or None
          confidence: float, the model's raw peak confidence this cycle
          accepted:   bool, whether this cycle's detection was accepted (>= tau,
                      or track-hysteresis continuation)
          color/shape: the (target_color, target_shape) query this cache is for
          cam_type:   'grounding' | 'proximity'
    """
    return state.last_heatmap


def reset_track(state: GroundNetState) -> None:
    """Clear NX-7 FIX B's hysteresis track state. Callers should invoke this
    once at the start of every episode (before the first ground() call) --
    same pattern as constructing a fresh code.perception.lock_gate.LockGate()
    per episode. A no-op (cheap) when GROUND_NET_HYSTERESIS is off."""
    state.track_dist_m      = None
    state.track_bearing_rad = None


def latency_stats(state: GroundNetState) -> dict:
    """Diagnostic helper for smoke tests / gate runs to report GROUND_NET
    latency alongside the closed-loop success metrics.

    Returns:
        Summary stats (ms) over every GROUND_NET inference call made so far
        in this process: n, mean_ms, p50_ms, p95_ms, p99_ms, max_ms. Empty
        dict if GROUND_NET was never invoked.
    """
    if not state.lat_ms:
        return {}
    xs = sorted(state.lat_ms)
    n = len(xs)

    def _pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return xs[idx]

    return dict(n=n, mean_ms=sum(xs) / n, p50_ms=_pct(0.50), p95_ms=_pct(0.95),
                p99_ms=_pct(0.99), max_ms=xs[-1])


def load_detector(state: GroundNetState, ckpt_path: str, device: str, tau: float) -> Any:
    """Lazy global load of the NX-6 heatmap detector checkpoint. Loaded once per
    process (not per call/episode); subsequent calls return the cached instance
    (or None, sticky, if the first load failed -- avoids retry-storming a bad
    checkpoint path every grounding cycle). Import of
    code.perception.detector.model is deliberately deferred to inside this
    function (rather than a module-level import here) to avoid pulling in
    torch/cuda for every caller of code.perception.grounding, matching the
    original module's lazy-import behaviour exactly.

    Returns:
        The cached code.perception.detector.model.HeatmapDetector instance
        (typed Any here since importing that type at module level is
        deliberately avoided), or None if loading failed.
    """
    if state.detector is not None or state.load_failed:
        return state.detector
    try:
        import torch
        from code.perception.detector.model import CLASS_NAMES, COLOR_NAMES, HeatmapDetector
        state.class_names = CLASS_NAMES
        state.color_names = COLOR_NAMES
        actual_device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        state.detector = HeatmapDetector.load(ckpt_path, device=actual_device)
        print(f"[grounding] GROUND_NET=1: loaded detector {ckpt_path!r} "
              f"on device={actual_device!r} (conf_thresh={tau})", flush=True)
    except Exception as e:
        state.load_failed = True
        print(f"[grounding] GROUND_NET=1: FAILED to load detector from "
              f"{ckpt_path!r} ({e!r}) -- ground() will fall back to the "
              f"classical HSV+depth pipeline for the rest of this process "
              f"(NX-9 graceful fallback, docs/nx9_avoid.md).", flush=True)
    return state.detector


def infer(state: GroundNetState, ego_rgb: np.ndarray, ego_depth: np.ndarray,
         target_color: str, target_shape: str, intrinsics: dict, *,
         tau: float, hysteresis: bool, tau_track: float) -> GroundingResult:
    """GROUND_NET=1 backend: the NX-6 JUDGE-selected query-conditioned heatmap
    detector, in place of the classical HSV+depth pipeline. Same
    (dist,bearing)+confidence contract as the classical path. Fed whichever
    camera's frame the caller already rendered this cycle (grounding-cam-far
    vs proximity-cam-near, selected via intrinsics['is_proximity'] -- the
    SAME flag the classical path reads).

    `state.detector` must already be loaded (see `load_detector`) -- this
    function assumes the caller has already checked it is not None.
    """
    det = state.detector
    if det is None:
        return GroundingResult(0, 1, 0, 0.0, True)

    shape_key = str(target_shape).lower().strip()
    color_key = str(target_color).lower().strip()
    if (state.class_names is None or shape_key not in state.class_names
            or color_key not in state.color_names):
        # Query outside the detector's trained vocabulary -- fail safe to
        # not_visible rather than raising or silently guessing. Never actually
        # observed in this codebase: target_shape/target_color always come
        # straight from arena.SHAPES/COLORS, the SAME ordering/name-set
        # dataset/det_v1's labels (and this detector's CLASS_NAMES/COLOR_NAMES)
        # were built from -- see code/perception/detector/model.py's module comment.
        return GroundingResult(0, 1, 0, 0.0, True)

    is_proximity = bool(intrinsics.get('is_proximity', False))
    is_widefov   = bool(intrinsics.get('is_widefov', False))
    if is_widefov and not state.widefov_warned:
        state.widefov_warned = True
        print("[grounding] GROUND_NET=1: widefov camera requested but the "
              "detector was only trained/validated on {grounding,proximity} "
              "cam_type geometry (docs/nx6_data.md) -- falling back to "
              "'grounding' pitch geometry. CAMERA_MODE=widefov combined with "
              "GROUND_NET=1 is an untested combination (out of scope for the "
              "NX-6 integration gate, which ran with no CAMERA_MODE set).",
              flush=True)
    cam_type = "proximity" if is_proximity else "grounding"

    t0 = time.perf_counter()
    try:
        # NX-7 FIX B: always decode at conf_thresh=0.0 so `out['confidence']` is
        # the model's TRUE raw peak sigmoid probability every cycle (present or
        # not) -- decode_single already computes `confidence` unconditionally
        # and only thresholds `present` separately, so this is free (no extra
        # forward pass); we just do our own thresholding below instead of
        # trusting det.infer()'s built-in `present` when hysteresis is
        # enabled, so the tau_track path can see it.
        out = det.infer(ego_rgb, ego_depth, class_name=shape_key, color_name=color_key,
                        cam_type=cam_type,
                        conf_thresh=(0.0 if hysteresis else tau))
    except Exception as e:
        print(f"[grounding] GROUND_NET=1: infer() raised {e!r} this cycle -- "
              f"treating as not_visible.", flush=True)
        return GroundingResult(0, 1, 0, 0.0, True)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    state.lat_ms.append(dt_ms)

    conf    = float(out['confidence'])
    dist    = max(0.0, float(out['dist_m']))
    yaw_err = math.radians(float(out['bearing_deg']))

    accepted = conf >= tau   # tau_acquire -- unchanged, always the fast path

    if not accepted and hysteresis and conf >= tau_track:
        # tau_track continuation: only valid while a track is live AND the new
        # (dist,bearing) is spatially continuous with it, gated by the SAME
        # M3 innovation-gate constants code/perception/lock_gate.py uses
        # downstream (kept in sync by import, not duplicated).
        if state.track_dist_m is not None:
            near = state.track_dist_m < M3_NEAR_RANGE_M
            bearing_gate_rad = math.radians(
                M3_GATE_BEARING_DEG * (M3_GATE_BEARING_NEAR_MULT if near else 1.0))
            dist_gate_m = max(M3_GATE_DIST_FLOOR_M,
                              M3_EXPECTED_CLOSING_M_PER_CYCLE * M3_GATE_DIST_CLOSING_MULT)
            d_bearing = abs(_ang_diff_rad(yaw_err, state.track_bearing_rad))
            d_dist    = abs(dist - state.track_dist_m)
            if d_bearing <= bearing_gate_rad and d_dist <= dist_gate_m:
                accepted = True

    # VF-1 (docs/vf1_showpiece.md): cache this cycle's confidence heatmap +
    # decision for render-side display (fancy_demo.py's detector-heatmap
    # overlay). Reuses the forward pass det.infer() already ran above (the
    # detector caches its own last sigmoid map as an attribute, set with zero
    # extra inference -- see code/perception/detector/model.py's
    # HeatmapDetector.infer()). Pure additive side effect: does not change
    # `accepted`/`conf`/`dist`/`yaw_err` or either return path below.
    _heat_prob = getattr(det, 'last_heat_prob', None)
    state.last_heatmap = dict(
        prob=(_heat_prob.copy() if _heat_prob is not None else None),
        confidence=conf,
        accepted=accepted,
        color=color_key,
        shape=shape_key,
        cam_type=cam_type,
    )

    if not accepted:
        return GroundingResult(0, 1, 0, 0.0, True)

    if hysteresis:
        state.track_dist_m      = dist
        state.track_bearing_rad = yaw_err

    return GroundingResult(
        dist        = dist,
        cos_th      = math.cos(yaw_err),
        sin_th      = math.sin(yaw_err),
        confidence  = conf,
        not_visible = False,
        # best_area / phys_w / phys_h / n_raw_components / ... intentionally
        # left None: those are classical-pipeline-specific diagnostics with no
        # analogue for a learned heatmap detection. See gate_detection()'s
        # `area is not None` check -- this makes LOCK_M1 a provable no-op for
        # GROUND_NET detections specifically (by design: the network's own
        # conf_thresh already serves the "is this detection trustworthy"
        # role M1's area floor serves for the classical pipeline), while
        # LOCK_M3's bearing/distance innovation gate is unaffected.
    )
