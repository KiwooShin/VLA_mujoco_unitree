"""
code.control.avoid.core — NX-9 LOCAL OBSTACLE AVOIDANCE (docs/nx9_avoid.md).

The system as of NX-8 (docs/nx8_stall.md) walks straight lines at the goal —
no local obstacle awareness at all. Three independent, geometry-confirmed
failures are path-obstacle collisions, not grounding or locomotion-stability
problems:
  1. demo ep1 under GROUND_NET=1: a physical wedge against the scene's own
     orange-cone distractor, ~0.25m off the straight-line path (docs/nx8_stall.md
     §2.3 — definitive qpos+geometry trace).
  2. demo ep4: compound failure where even the privileged GT-goal rollout
     fails (docs/fa1_failures.md §1) — geometry-bound, honest partial target.
  3. search ep12: a fall caused by a distractor 0.92m along the approach path
     (docs/nx1_scan.md).

This module is a shared, backend-agnostic, call-site-agnostic helper (reused
by code/inferencer.py, code/eval_search.py, code/fancy_demo.py — the "reuse a
shared helper" precedent set by code/scan_sched.py and code/lock_mgmt.py) that:

  1. At the existing grounding cadence (5-10Hz, GROUNDING_PERIOD in
     inferencer.py) — ZERO extra renders — reads the depth frame the caller
     ALREADY rendered this cycle for classical/GROUND_NET grounding, and
     builds a near-field obstacle signal via full-frame back-projection
     (reusing the exact camera model code/grounding.py's cam_to_egocentric
     uses, so bearings/distances are apples-to-apples with cached_goal_vec).
  2. Excludes the floor via a height-above-ground back-projection cut, and
     excludes the target's own pixels via a bearing-window mask around the
     current goal bearing (only when goal dist < AVOID_TARGET_EXEMPT_DIST_M —
     don't dodge the thing we're approaching).
  3. Converts the remaining near-field, in-corridor obstacle mass into a
     bounded, hysteresis-smoothed YAW-RATE bias (never lateral — steer.py's
     own VX_YAW_DAMP=0.0 / "G1 walks straight" comment confirms the BC
     teacher never strafed, so a lateral bias would be off-distribution;
     yaw is the in-distribution steering DOF).
  4. Exposes `biased_vel_cmd()`, a drop-in replacement for steer.py's control
     law (same MAX_VX/MAX_WZ/YAW_KP/FACE_THR_RAD/DECEL_DIST constants,
     imported not duplicated) plus the bias term, clipped back to steer.py's
     own MAX_WZ bound — so the combined command is provably never outside
     the velocity range steer.py itself (the BC teacher) ever produced.

Toggle: AVOID env var — default ON since NX-9 adoption (docs/nx9_avoid.md
§5; opt out with AVOID=0). Every function below is cheap (numpy-vectorized,
no torch, no extra renders) and a no-op by construction when AVOID is
disabled — callers gate every call site on `AVOID and ...`.

Carve-outs (enforced by the CALLER structurally, mirroring STALL_BREAK's own
"reached only on the guaranteed non-scan path" precedent in inferencer.py):
  - off during scan/rescan/dwell: callers gate the computation on
    `not _scan_active`, and the bias injection site is additionally only on
    the normal (post-scan) student-step code path, exactly like STALL_BREAK.
  - off when goal dist < AVOID_MIN_GOAL_DIST_M (1.2m): passed as
    `carved_out=True` — proximity endgame, the target IS the close object.
  - off while the goal is STALE (coasting on hold-last-known-goal for more
    than AVOID_STALE_MAX_MISSED_CYCLES grounding cycles): the existing bias
    only decays via `decay_bias()` — see that constant's comment for the
    search-ep14 fall trace that motivated this (a stale cached goal makes
    the target exemption and the proximity cut both point at the wrong
    place, letting AVOID attack its own target).
  - off in maneuver-difficulty scenes: `is_maneuver_scene()` helper, checked
    by the caller before ever invoking `compute_obstacle_bias` (same
    `scene_cfg.get('difficulty')` pattern STALL_BREAK's `_stall_is_maneuver`
    uses) — maneuver scenes can legitimately involve close, deliberate
    contact with course geometry, out of this mechanism's intended scope.
  - decays to zero within ~1s of the corridor clearing: `AVOID_DECAY_FACTOR`
    per zero-obstacle grounding cycle (0.5/cycle @ 5Hz reaches <5% in 5
    cycles = 1.0s), snapped to exactly 0.0 below `AVOID_DEADBAND` — no
    permanent path offset (the underlying goal EMA / cached_goal_vec is
    NEVER touched by this module; only the injected *velocity* is biased).

RF-1: split out of code/avoid.py (see code/avoid.py, the old-path compat
alias, and docs/refactor_plan.md) — this is the "core" half of the
core/helpers split; code.control.avoid.geometry holds the back-projection
math, code.control.avoid._selftest holds the synthetic-frame smoke test.
"""

from __future__ import annotations

import math
import os

import numpy as np

from code.control.avoid.geometry import backproject_frame


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() == "1"


# ---------------------------------------------------------------------------
# Toggle + constants
# ---------------------------------------------------------------------------
# NX-9 ADOPTION (docs/nx9_avoid.md §5): default ON. Cleared all three of the
# adoption matrix's gates on the final code state (seed 999, n=15 each):
# demo/classical 10/15 held exactly (baseline fail set, zero regressions),
# easy 15/15 exact, search 15/15 (the documented ep12 distractor-collision
# fall FIXED). Backend-agnostic (classical + GROUND_NET). Opt out with
# AVOID=0.
AVOID = _env_flag("AVOID", default="1")   # ADOPTED default ON (docs/nx9_avoid.md)

# --- Near-field obstacle window ---
# NX-9 CONSTANTS REVISION (one pass, per the task's bounded-ladder rule —
# docs/nx9_avoid.md §mechanism-replay): the FIRST pass (AVOID_NEAR_M=1.5,
# AVOID_MIN_DEPTH_FOR_WEIGHT_M=0.30) correctly DETECTED demo ep1's
# obstacle-collision cone from step ~70 onward (instrumented replay,
# n_obstacle_px growing to 4500+ by step ~120) but the severity ramp only
# reaches meaningful magnitude within ~0.3-0.5m — a range the depth camera
# geometry cannot reliably resolve at all once the robot is that close
# (self-occlusion / near-field cutoff), so the bias stayed tiny (<0.02 rad/s)
# through the whole detectable window and vanished (n_obstacle_px->0) right
# as the robot's forward progress stalled (~step 160), i.e. the mechanism was
# "detected but too weak, too late." Revised BOTH constants together as one
# bundled pass targeting that exact diagnosis: AVOID_NEAR_M widened (more
# lead time before the obstacle is close enough to matter) and
# AVOID_MIN_DEPTH_FOR_WEIGHT_M raised to a range the sensor can actually
# still see cleanly (severity now saturates at a physically-achievable
# distance, not a near-contact one).
AVOID_NEAR_M              = 2.0    # m — pixels farther than this are ignored entirely (was 1.5)
AVOID_MIN_VALID_DEPTH_M   = 0.15   # m — below this, treat as sensor noise/self-body, ignore
AVOID_MIN_DEPTH_FOR_WEIGHT_M = 1.0   # m — severity saturates to 1.0 at/below this range (was 0.30)
AVOID_CORRIDOR_HALF_DEG   = 25.0   # deg — forward corridor half-width (bearing)
AVOID_N_BEARING_BINS      = 25     # angular bins across the corridor (~2 deg/bin) —
                                    # closest-obstacle-per-bin, not diluted by frame area

# --- Carve-outs ---
# NX-13 (docs/nx13_avoid_hygiene.md): RE-APPLIED the 1.2 -> 1.6 revision on
# its own merits (self-body hygiene, not an ep4-flip bet). History: NX-11
# (docs/nx11_ep4.md) first tried this exact change, aligning this cutoff
# with inferencer.py's CAM_D_HI=1.6 PROXIMITY->GROUNDING Schmitt threshold,
# after root-causing a real bug: AVOID's depth back-projection has no
# self-body exclusion (only a floor-height cut and a target-bearing
# exemption), and the PROXIMITY camera's steep 58 deg pitch, mounted close
# to the robot's own chest, captures the robot's OWN raised/swinging arms
# during locomotion at very close range (0.24-0.43m, ~0.8m above the ground
# -- nowhere near the floor cut or the target's exemption bearing) --
# misclassified as a real obstacle (visually confirmed via saved RGB+depth
# frames on demo ep4, docs/nx11_ep4.md §3). Widening the carve-out to 1.6
# measurably fixed THIS (avoid_bias_active_frac=0.000, confirmed 2/2), but
# NX-11 reverted it anyway because ep4 didn't FLIP 2/2 (ep4's dominant
# failure is an unrelated late-episode balance-loss mechanism, per
# docs/nx11_ep4.md §5) -- an ep4-flip bar the fix was never meant to clear.
# NX-13 re-evaluated the fix on its own merits (mechanism-confirmed bug,
# zero known regression risk under NX-9's mid-path avoidance wins -- see
# docs/nx13_avoid_hygiene.md for full mechanism replays + five full-gate
# lines) and ADOPTED it.
AVOID_MIN_GOAL_DIST_M     = 1.6    # m — proximity endgame: never avoid below this goal dist
AVOID_TARGET_EXEMPT_DIST_M = 2.0   # m — only mask the target's own bearing window this close
AVOID_TARGET_EXEMPT_MIN_DEG = 8.0  # deg — floor on the target-exemption half-width
AVOID_TARGET_EXEMPT_MAX_DEG = 30.0 # deg — ceiling on the target-exemption half-width
AVOID_TARGET_EXEMPT_SIZE_M  = 0.35 # m — nominal target radius used for the atan() half-width
# Goal-freshness carve-out (spec compliance, found via search ep14's
# instrumented fall trace — docs/nx9_avoid.md §3.3): every AVOID carve-out
# that protects the target from being treated as an obstacle (the
# bearing-window exemption, the <1.2m proximity endgame cut) is keyed to
# `cached_goal_vec` — which FREEZES at the last-known value during a
# hold-last-known-goal coast (HOLD_GOAL_HORIZON). In ep14's fall, the goal
# froze at 1.73m/+23.5° for ~38 consecutive missed cycles while the robot
# circled its own target at <1m true range; the stale exemption window
# missed the target's true bearing and the stale 1.73m distance kept the
# proximity cut from ever firing, so AVOID saturated its bias against the
# very object it exists to protect ("don't dodge the thing we're
# approaching") and drove the circling into a fall. Fix: only COMPUTE a new
# bias while the goal is fresh (detected within the last
# AVOID_STALE_MAX_MISSED_CYCLES grounding cycles — the tolerance covers
# 1-2-cycle detection blinks, which the hysteresis was always designed to
# ride through); during a longer stale coast the existing bias simply
# DECAYS via `decay_bias()` (same schedule as a cleared corridor: zero
# within ~1s).
AVOID_STALE_MAX_MISSED_CYCLES = 2

# --- Floor exclusion (back-projection height cut) ---
AVOID_FLOOR_MARGIN_M      = 0.10   # m — points within this of "floor height" are excluded

# --- Bias shaping / bounds ---
AVOID_MAX_WZ_BIAS         = 0.30   # rad/s — cap on the ADDED bias (steer.py's own MAX_WZ=0.80
                                    # is the ceiling on the TOTAL command after clipping, so the
                                    # combined command is always within the BC teacher's own
                                    # observed range; VX_YAW_DAMP=0.0 in steer.py confirms the
                                    # teacher never strafed, so lateral bias is never used here)
AVOID_TIE_BREAK_IMBALANCE = 0.20   # deterministic right-turn preference when an obstacle sits
                                    # exactly on the corridor centerline (L≈R, would otherwise
                                    # produce a zero net bias in front of a dead-ahead obstacle)
AVOID_TIE_BREAK_EPS       = 0.05   # |imbalance| below this is treated as "centered"

# --- Hysteresis / decay (~0.5s persistence, decay to 0 within ~1s @ 5Hz grounding cadence) ---
AVOID_EMA_ALPHA           = 0.6    # blend weight for a fresh nonzero raw bias
AVOID_DECAY_FACTOR        = 0.5    # multiply prev bias by this each zero-obstacle cycle
AVOID_DEADBAND            = 0.01   # rad/s — snap-to-zero threshold


def is_maneuver_scene(scene_cfg: dict) -> bool:
    """Same `scene_cfg.get('difficulty')` carve-out check STALL_BREAK's
    `_stall_is_maneuver` uses in code/inferencer.py — kept here so every AVOID
    call site (inferencer.py / eval_search.py / fancy_demo.py) checks it the
    same way instead of re-deriving it three times.

    Args:
        scene_cfg: Scene config dict (checked for a 'difficulty' key).

    Returns:
        True iff `scene_cfg['difficulty']` is (case-insensitively) 'maneuver'.
    """
    return str(scene_cfg.get('difficulty', '')).lower() == 'maneuver'


def decay_bias(prev_bias_wz: float) -> float:
    """One decay step of the hysteresis schedule, WITHOUT computing a new
    bias — used by call sites on grounding cycles where AVOID must not
    produce a fresh bias (stale-goal coast per
    AVOID_STALE_MAX_MISSED_CYCLES; see that constant's comment) but an
    existing bias should still bleed off on the same ~1s schedule a cleared
    corridor uses, rather than freezing or snapping.

    Args:
        prev_bias_wz: Previous cycle's bias (rad/s).

    Returns:
        The decayed bias (rad/s), snapped to 0.0 below AVOID_DEADBAND and
        clipped to +/-AVOID_MAX_WZ_BIAS.
    """
    b = prev_bias_wz * AVOID_DECAY_FACTOR
    if abs(b) < AVOID_DEADBAND:
        b = 0.0
    return float(np.clip(b, -AVOID_MAX_WZ_BIAS, AVOID_MAX_WZ_BIAS))


# ---------------------------------------------------------------------------
# Main entry point: obstacle -> bounded yaw-rate bias
# ---------------------------------------------------------------------------
def compute_obstacle_bias(
    depth_m:          np.ndarray,
    intr:              dict,
    cam_height_m:      float,
    goal_dist_m:       float,
    goal_bearing_rad:  float,
    prev_bias_wz:      float,
    carved_out:        bool = False,
) -> tuple[float, dict]:
    """
    Compute this cycle's obstacle-avoidance yaw-rate bias (rad/s).

    Args:
        depth_m: (H,W) float32 depth frame ALREADY rendered by the
            caller this grounding cycle (zero extra renders).
        intr: Intrinsics dict from the SAME render call
            (fx,fy,cx,cy[,pitch_deg,is_proximity]).
        cam_height_m: Current camera height above the (flat) ground plane
            (pelvis z + CAM_HEAD_Z) — used for the floor cut.
        goal_dist_m: Current cached goal distance (m) — used both for the
            target-exemption window and to report carve-out.
        goal_bearing_rad: Current cached goal bearing (rad, positive=LEFT) —
            center of the target-exemption window.
        prev_bias_wz: Previous cycle's returned bias (for hysteresis/decay).
        carved_out: True when the CALLER has already determined AVOID
            should not apply this cycle (goal_dist < 1.2m or
            maneuver scene) — hard-zeros immediately, no decay
            (matches "off when goal dist < 1.2m" / "off in
            maneuver mode", which are not "corridor cleared"
            events and should not linger).

    Returns:
        Tuple (bias_wz, info): bias_wz is the SMOOTHED/hysteresis-applied
        bias to add to the commanded yaw rate this cycle; info is a debug
        dict (raw_bias, left, right, severity, imbalance, n_obstacle_px,
        carved_out) for logging/tests.
    """
    if carved_out:
        return 0.0, dict(raw_bias=0.0, left=0.0, right=0.0, severity=0.0,
                          imbalance=0.0, n_obstacle_px=0, carved_out=True)

    dist, bearing, y_vert = backproject_frame(depth_m, intr)
    bearing_deg = np.degrees(bearing)

    valid_mask = np.isfinite(dist) & (dist >= AVOID_MIN_VALID_DEPTH_M) & (dist <= AVOID_NEAR_M)
    corridor_mask = np.abs(bearing_deg) <= AVOID_CORRIDOR_HALF_DEG
    height_above_ground = cam_height_m - y_vert
    floor_mask = height_above_ground < AVOID_FLOOR_MARGIN_M

    target_exempt_mask = np.zeros_like(valid_mask)
    if goal_dist_m < AVOID_TARGET_EXEMPT_DIST_M:
        halfwidth_deg = math.degrees(math.atan(
            AVOID_TARGET_EXEMPT_SIZE_M / max(goal_dist_m, 0.4)))
        halfwidth_deg = float(np.clip(halfwidth_deg,
                                       AVOID_TARGET_EXEMPT_MIN_DEG,
                                       AVOID_TARGET_EXEMPT_MAX_DEG))
        goal_bearing_deg = math.degrees(goal_bearing_rad)
        d_bearing = np.abs(np.mod(bearing_deg - goal_bearing_deg + 180.0, 360.0) - 180.0)
        target_exempt_mask = d_bearing <= halfwidth_deg

    obstacle_mask = valid_mask & corridor_mask & (~floor_mask) & (~target_exempt_mask)
    n_obstacle_px = int(obstacle_mask.sum())

    # NX-9 design note: a naive per-pixel average over the FULL corridor area
    # (most of which, for any real frame, is far/empty/floor) heavily dilutes
    # a real obstacle's signal, since a close object usually only fills a
    # fraction of the corridor's vertical extent. Instead, bin the corridor
    # into AVOID_N_BEARING_BINS angular bins and take the CLOSEST obstacle
    # return per bin (the physically meaningful "is something in this
    # direction, and how close" signal — the same principle a 1-D range-scan
    # obstacle detector uses) — this is what "repulsion proportional to
    # 1/depth weighted by proximity to corridor center" resolves to per bin.
    bin_edges_deg = np.linspace(-AVOID_CORRIDOR_HALF_DEG, AVOID_CORRIDOR_HALF_DEG,
                                 AVOID_N_BEARING_BINS + 1)
    bin_centers_deg = 0.5 * (bin_edges_deg[:-1] + bin_edges_deg[1:])
    min_dist_per_bin = np.full(AVOID_N_BEARING_BINS, AVOID_NEAR_M, dtype=np.float32)

    if n_obstacle_px > 0:
        obs_bearing_deg = bearing_deg[obstacle_mask]
        obs_dist        = dist[obstacle_mask]
        bin_idx = np.clip(
            ((obs_bearing_deg + AVOID_CORRIDOR_HALF_DEG)
             / (2.0 * AVOID_CORRIDOR_HALF_DEG) * AVOID_N_BEARING_BINS).astype(np.int64),
            0, AVOID_N_BEARING_BINS - 1)
        np.minimum.at(min_dist_per_bin, bin_idx, obs_dist)

    # Per-bin severity ("repulsion proportional to 1/depth"): 0 at
    # AVOID_NEAR_M, 1 at/inside AVOID_MIN_DEPTH_FOR_WEIGHT_M.
    inv_lo = 1.0 / AVOID_NEAR_M
    inv_hi = 1.0 / AVOID_MIN_DEPTH_FOR_WEIGHT_M
    inv_d = 1.0 / np.clip(min_dist_per_bin, AVOID_MIN_DEPTH_FOR_WEIGHT_M, None)
    severity_bin = np.clip((inv_d - inv_lo) / (inv_hi - inv_lo), 0.0, 1.0)
    # Weight by proximity to corridor center (1 at bearing=0, 0 at the edge).
    center_bin = np.clip(1.0 - np.abs(bin_centers_deg) / AVOID_CORRIDOR_HALF_DEG, 0.0, 1.0)
    weight_bin = severity_bin * center_bin

    left_bins  = bin_centers_deg > 0.0
    right_bins = ~left_bins

    # Worst-bin (max), not mean-over-all-bins: a single close, centered
    # object should trigger a decisive response even when most of the
    # corridor is clear (a "how occupied is this whole half" mean dilutes a
    # small, localized obstacle — exactly the ep1-cone-collision shape this
    # module targets — down toward the deadband; the worst bin per side is
    # the standard "react to the nearest/most-salient threat" reading of
    # "repulsion proportional to 1/depth weighted by proximity to corridor
    # center", and it makes AVOID_MAX_WZ_BIAS's ceiling interpretable
    # exactly: reached only when the single worst obstruction is BOTH
    # at/inside AVOID_MIN_DEPTH_FOR_WEIGHT_M AND dead-center).
    L = float(weight_bin[left_bins].max())  if left_bins.any()  else 0.0
    R = float(weight_bin[right_bins].max()) if right_bins.any() else 0.0

    if n_obstacle_px == 0:
        raw_bias = 0.0
        imbalance = 0.0
        overall = 0.0
    else:
        overall = float(np.clip(L + R, 0.0, 1.0))
        denom = L + R
        imbalance = (L - R) / denom if denom > 1e-9 else 0.0
        if overall > (AVOID_DEADBAND / AVOID_MAX_WZ_BIAS) and abs(imbalance) < AVOID_TIE_BREAK_EPS:
            # Obstacle centered in the corridor (L≈R) — deterministic
            # right-hand tie-break so a dead-ahead obstacle still produces
            # a decisive turn instead of a canceled zero bias.
            imbalance = AVOID_TIE_BREAK_IMBALANCE
        # L > R (more obstacle mass on the LEFT, positive bearing) -> steer
        # RIGHT (negative wz), matching steer.py's "positive yaw_err = LEFT"
        # convention used throughout this codebase.
        raw_bias = -AVOID_MAX_WZ_BIAS * overall * float(np.clip(imbalance, -1.0, 1.0))

    # Hysteresis: blend a fresh nonzero raw bias in quickly, but decay a
    # stale bias slowly (geometrically) toward zero once the corridor clears
    # — persists ~0.5s, reaches <5% of its prior value within ~5 cycles
    # (~1.0s @ 5Hz grounding cadence), then snaps to exactly 0.0.
    if abs(raw_bias) > 1e-9:
        bias_wz = AVOID_EMA_ALPHA * raw_bias + (1.0 - AVOID_EMA_ALPHA) * prev_bias_wz
    else:
        bias_wz = prev_bias_wz * AVOID_DECAY_FACTOR
    if abs(bias_wz) < AVOID_DEADBAND:
        bias_wz = 0.0
    bias_wz = float(np.clip(bias_wz, -AVOID_MAX_WZ_BIAS, AVOID_MAX_WZ_BIAS))

    info = dict(raw_bias=raw_bias, left=L, right=R, severity=overall,
                imbalance=imbalance, n_obstacle_px=n_obstacle_px, carved_out=False)
    return bias_wz, info


# ---------------------------------------------------------------------------
# Shared steer.py + bias helper (reused by every rollout loop)
# ---------------------------------------------------------------------------
def biased_vel_cmd(goal_dist: float, cos_th: float, sin_th: float,
                    bias_wz: float, stop_r: float) -> np.ndarray:
    """
    steer.py's own control law, evaluated from an already-known (dist,
    cos_th, sin_th) goal (e.g. cached_goal_vec — works identically whether
    that goal came from classical HSV+depth or GROUND_NET, since both
    populate the same GroundingResult contract), PLUS the AVOID yaw-rate
    bias — clipped back to steer.py's own MAX_WZ bound so the combined
    command is provably within the range the BC teacher (steer.py) itself
    ever produced during data collection.

    Mirrors the existing "learned-grounding velocity injection" replica
    block in code/inferencer.py (same constants, same formula) — this is
    that same math, factored out once so all three rollout loops
    (inferencer.py / eval_search.py / fancy_demo.py) call one function
    instead of three near-duplicate copies.

    Args:
        goal_dist: Current goal distance (m).
        cos_th: Cosine of the goal-heading error.
        sin_th: Sine of the goal-heading error.
        bias_wz: AVOID yaw-rate bias (rad/s) to add on top of the steering
            law's own yaw command.
        stop_r: Stop radius (m) — zero velocity is returned inside this.

    Returns:
        np.float32[3] = [vx, vy=0, wz] (vy always 0 — steer.py's own
        VX_YAW_DAMP=0.0 / "G1 walks straight" comment: the BC teacher never
        strafed, so AVOID never injects a lateral command either).
    """
    from code.control.steer import MAX_VX, MAX_WZ, YAW_KP, FACE_THR_RAD, DECEL_DIST

    if goal_dist < stop_r:
        return np.zeros(3, dtype=np.float32)

    yaw_err = math.atan2(sin_th, cos_th)
    wz = float(np.clip(YAW_KP * yaw_err + bias_wz, -MAX_WZ, MAX_WZ))
    yaw_align = max(0.0, math.cos(yaw_err))
    if abs(yaw_err) > FACE_THR_RAD:
        vx = 0.0
    else:
        decel = min(1.0, max(0.0, (goal_dist - stop_r) / max(DECEL_DIST - stop_r, 0.1)))
        vx = float(np.clip(MAX_VX * yaw_align * decel, 0.0, MAX_VX))
    return np.array([vx, 0.0, wz], dtype=np.float32)
