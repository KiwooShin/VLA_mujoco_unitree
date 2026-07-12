"""
code.runtime.goal_pipeline — grounding-cycle/goal/EMA/hold/handoff/scan state
for the closed-loop Inferencer.

RF-1 split of code/inferencer.py (docs/refactor_plan.md): `GoalPipeline`
below owns everything the original monolithic `Inferencer.rollout()` tracked
as local/`nonlocal` variables across a rollout episode for goal smoothing,
CAM-2 camera handoff, the NX-1/NX-10 scan-and-acquire schedule, NX-2/NX-5
lock management, and NX-9 AVOID's per-cycle obstacle bias — i.e. everything
EXCEPT the actual `classical_ground(...)` call sites, which stay in
`code/runtime/inferencer.py` itself (via `Inferencer._ground`) so that
diagnostic scripts monkeypatching `code.inferencer.classical_ground`
(code/gen_det_failcases.py, eval/nx7_ep1_diag/*, eval/nx8_stall/*) keep
observing every grounding call, exactly as when this logic lived inline in
one file. `code/runtime/rollout_step.py` calls `inf._ground(...)` to get a
`GroundingResult`, then hands it to the methods below.

This is a mechanical extraction: the control flow and numeric logic are
unchanged from the pre-RF-1 monolithic function, only *where* the code lives
and *how* state is threaded (object attributes instead of closure cells).
"""

from __future__ import annotations

import math

import numpy as np

from code.sim.arena import CAMERA_MODE, CAM_HEAD_Z
from code.perception.lock_mgmt import LockGate, ReacquisitionScan
from code.control.scan_sched import BidirectionalScanSchedule, SCAN_DWELL_STEPS as _H3_DWELL_STEPS
from code.control import avoid as _avoid
from code.runtime.constants import AVOID, GROUNDING_PERIOD
from code.runtime.goal_config import (
    GOAL_EMA_ALPHA, HOLD_GOAL_HORIZON,
    H3_LEG_DEG, SCAN_TIMEOUT, SCAN_RATE, SCAN_DT, SCAN_ALIGNED_THR_DEG,
    CAM_D_LO, CAM_D_HI, CAM_PROXIMITY_D_FAR,
)


class GoalPipeline:
    """Per-episode grounding-cycle/goal/EMA/hold/handoff/scan state machine.

    Owns everything a rollout episode's goal-tracking needs across steps:
    the EMA'd/held goal vector, the CAM-2 active-camera Schmitt trigger, the
    H3/ReacquisitionScan scan-and-acquire schedule, the NX-2/NX-5 LockGate
    wiring, and the NX-9 AVOID per-cycle obstacle-bias state. Does NOT call
    `classical_ground` itself — see module docstring.
    """

    def __init__(self, need_classical_render: bool, need_learned_render: bool,
                 avoid_is_maneuver: bool, verbose: bool = False) -> None:
        """Initializes fresh per-episode state (mirrors the original
        `Inferencer.rollout()` preamble, docs/refactor_plan.md).

        Args:
            need_classical_render: True when this episode uses classical
                HSV+depth grounding (arch A, goal_source='classical').
            need_learned_render: True when this episode uses the trained
                grounding head (arch A, goal_source='learned', grounding_trained).
            avoid_is_maneuver: True if NX-9 AVOID should stay carved out for
                this episode's difficulty (maneuver scenes).
            verbose: Per-step diagnostic prints.
        """
        self.need_classical_render = need_classical_render
        self.need_learned_render   = need_learned_render
        self.verbose = verbose

        # Grounding cache (Arch A)
        self.cached_goal_vec     = np.array([2.0, 1.0, 0.0], dtype=np.float32)
        self.last_grounding_step = -999

        # Scan-and-acquire state
        self.scan_active    = True      # True until target centered in frame
        self.scan_yaw_delta = 0.0       # cumulative yaw scanned (rad) -- diagnostic only,
                                         # no longer drives the schedule (see module comment)
        self.h3_scan_sched  = BidirectionalScanSchedule(
            scan_rate=SCAN_RATE, leg_deg=H3_LEG_DEG, dwell_steps=_H3_DWELL_STEPS)

        # Goal smoothing: exponential moving average of detected goals (E6)
        self.goal_ema             = None      # set on first detection
        # Last-known-good goal: hold when briefly lost (E6)
        self.last_known_goal      = None      # set on first detection; held through gaps
        self.frames_since_detection = 0       # steps since last valid detection

        # CAM-2 (Phase 1): Schmitt-trigger handoff state
        self.active_cam       = 'GROUNDING'   # default at episode start (targets start 1.5-9m away)
        self.cam_miss_count   = 0             # consecutive misses on the active camera
        self.video_frame_cache = None         # demo-viz: last labeled active-cam frame (video only)

        # NX-2/NX-5 (docs/rs1_lock_mgmt.md, docs/nx5_coherence.md): shared
        # lock-management gate (LOCK_M1..M5, LOCK_M7, each independently
        # toggled via env var; M1/M3 default ON (opt-out), M2/M4/M5/M7
        # default OFF (opt-in) per docs/nx2_final.md / docs/nx5_coherence.md
        # -- see code/lock_mgmt.py). With all of LOCK_M1..M5=0 (M7 is
        # already off by default), every LockGate method call below is a
        # provable no-op pass-through, so that legacy configuration is
        # byte-identical-behavior by construction.
        self.lock_gate          = LockGate()
        self.using_rescan_sched = False   # True only while a M4/M5/M7-triggered bounded
                                           # rescan (ReacquisitionScan) is driving scan_active,
                                           # as opposed to the original H3 scan.
        self.rescan_sched       = None
        # M7 odometry-coherence watchdog: robot's own world-frame XY at the
        # previous classical grounding cycle (privileged sim state, but only
        # the ROBOT's own pose -- not the target's -- exactly what a real
        # state-estimator/leg-odometry stack would give on hardware). None
        # until the first grounding cycle. Always maintained (cheap: one
        # qpos copy + a handful of flops per grounding cycle) regardless of
        # LOCK_M7 -- `LockGate.end_of_cycle()` is itself a no-op when off,
        # matching every other mechanism's "always call, no-op internally"
        # contract.
        self.m7_prev_xy = None

        # NX-9 AVOID (docs/nx9_avoid.md): per-episode obstacle-bias state.
        # `avoid_bias_wz` persists across steps between grounding cycles
        # (same pattern as `cached_goal_vec`); only ever updated/read on the
        # guaranteed-non-scan "normal mode" path (structural carve-out,
        # mirroring STALL_BREAK), and reset to 0 whenever a scan/rescan
        # begins (in `_lock_drop_and_rescan`) so a stale bias from before a
        # scan never silently reapplies once normal mode resumes.
        self.avoid_bias_wz      = 0.0
        self.avoid_is_maneuver  = avoid_is_maneuver
        self.avoid_cycles_total  = 0     # diagnostic: grounding cycles AVOID evaluated
        self.avoid_cycles_active = 0     # diagnostic: cycles with |bias| > 0 after this cycle

    # ------------------------------------------------------------------
    # Grounding-cycle cadence
    # ------------------------------------------------------------------
    def due_for_classical_grounding(self, step: int) -> bool:
        """True iff a classical-grounding cycle should run this step."""
        return (self.need_classical_render and
                (step - self.last_grounding_step) >= GROUNDING_PERIOD)

    def due_for_learned_grounding(self, step: int) -> bool:
        """True iff a learned-grounding cycle should run this step."""
        return (self.need_learned_render and
                (step - self.last_grounding_step) >= GROUNDING_PERIOD)

    def mark_grounding_step(self, step: int) -> None:
        """Records that a grounding cycle (classical or learned) ran this step."""
        self.last_grounding_step = step

    # ------------------------------------------------------------------
    # CAM-2 bounded fallback probe (docs/cam_opt2_multicam.md handoff rule)
    # ------------------------------------------------------------------
    def register_detection_outcome(self, not_visible: bool) -> None:
        """Updates the active-camera miss streak from the PRIMARY (pre-probe)
        detection outcome this cycle."""
        if not_visible:
            self.cam_miss_count += 1
        else:
            self.cam_miss_count = 0

    def maybe_probe_camera(self) -> str | None:
        """Returns the OTHER camera name to probe this cycle, or None.

        After 2 consecutive misses on the active camera, try the OTHER
        camera once this cycle. CAM-1 (Phase 2, toggle): no probe/handoff in
        widefov mode — gate is a no-op for cam2 (default). PLAUSIBILITY GATE
        (docs/cam_p1.md / docs/cam_p4_gate.md): only probe PROXIMITY when the
        last-known EMA distance says the target could actually be inside its
        ~0.22-1.81m band; probing GROUNDING from PROXIMITY is always safe.
        """
        if CAMERA_MODE != 'widefov' and self.cam_miss_count >= 2:
            other_cam = 'GROUNDING' if self.active_cam == 'PROXIMITY' else 'PROXIMITY'
            probe_ok = (other_cam == 'GROUNDING' or
                        (self.last_known_goal is not None and
                         float(self.last_known_goal[0]) <= CAM_PROXIMITY_D_FAR))
            if probe_ok:
                return other_cam
        return None

    def on_probe_adopted(self, other_cam: str) -> None:
        """Called when the fallback probe on `other_cam` detected the target.

        NX-2 mandatory carve-out (docs/rs1_lock_mgmt.md risk #2): the
        fallback probe-adopt is a legitimate (dist,bearing) discontinuity,
        not a track anomaly -- bypass M3/M4 for it.
        """
        self.active_cam     = other_cam
        self.cam_miss_count = 0
        self.lock_gate.mark_discontinuity()

    # ------------------------------------------------------------------
    # Classical detection ingestion (EMA / hold / CAM-2 handoff / scan-exit)
    # ------------------------------------------------------------------
    def process_classical_detection(self, gr, step: int) -> None:
        """Ingests this cycle's (post-probe) `GroundingResult`.

        Mirrors the original inline `if not gr.not_visible: ... else: ...`
        block exactly (EMA update, NX-2 detection gating, CAM-2 Schmitt
        handoff, scan-exit-on-alignment, or -- on a miss/gate-reject -- hold
        the last-known goal / bounded coast -> rescan).
        """
        if not gr.not_visible:
            raw_goal = gr.goal_vec.copy()
            # NX-2 (LOCK_M1/M2/M3, docs/rs1_lock_mgmt.md): gate the raw detection
            # BEFORE it's allowed to feed the EMA/last-known-goal. With all three
            # toggles off this is a provable pass-through (always True) -- see
            # code/lock_mgmt.py's LockGate.gate_detection docstring.
            accept_hit = self.lock_gate.gate_detection(
                float(raw_goal[0]), math.atan2(raw_goal[2], raw_goal[1]), gr.best_area)
            if accept_hit:
                self.frames_since_detection = 0
                if self.goal_ema is None:
                    self.goal_ema        = raw_goal.copy()
                    self.last_known_goal = raw_goal.copy()
                else:
                    self.goal_ema = GOAL_EMA_ALPHA * raw_goal + (1.0 - GOAL_EMA_ALPHA) * self.goal_ema
                    # Re-normalize cos/sin
                    th = math.atan2(self.goal_ema[2], self.goal_ema[1])
                    self.goal_ema[1] = math.cos(th)
                    self.goal_ema[2] = math.sin(th)
                    self.last_known_goal = self.goal_ema.copy()
                self.cached_goal_vec = self.goal_ema.copy()
                # CAM-2 (Phase 1): Schmitt-trigger handoff on the EMA'd distance —
                # D_LO/D_HI straddle the 0.92-1.81m dual-visible band (docs/cam_p1.md),
                # so this only flips once per approach/retreat, not every cycle.
                # CAM-1 (Phase 2, toggle): no handoff in widefov mode (single camera,
                # active_cam stays at its unused initial value) — gate is a no-op
                # for cam2 (default).
                if CAMERA_MODE != 'widefov':
                    ema_dist = float(self.goal_ema[0])
                    if self.active_cam == 'GROUNDING' and ema_dist < CAM_D_LO:
                        self.active_cam = 'PROXIMITY'
                        # NX-2 carve-out: Schmitt flip is a legitimate discontinuity.
                        self.lock_gate.mark_discontinuity()
                    elif self.active_cam == 'PROXIMITY' and ema_dist > CAM_D_HI:
                        self.active_cam = 'GROUNDING'
                        self.lock_gate.mark_discontinuity()
                # Exit scan mode when target is aligned (bearing < threshold).
                # Partial detections (bearing still large) keep scanning so the robot
                # continues rotating to better center the target in the image frame.
                if self.scan_active:
                    det_bearing_deg = abs(math.degrees(math.atan2(self.goal_ema[2], self.goal_ema[1])))
                    if det_bearing_deg < SCAN_ALIGNED_THR_DEG:
                        self.scan_active = False
                        if self.verbose:
                            print(f"  [scan] ALIGNED at step={step}  "
                                  f"yaw_err={math.degrees(math.atan2(self.goal_ema[2],self.goal_ema[1])):+.1f}°",
                                  flush=True)
                    elif self.verbose:
                        print(f"  [scan] Partial det step={step}  "
                              f"bearing={math.degrees(math.atan2(self.goal_ema[2],self.goal_ema[1])):+.1f}° "
                              f"(still scanning, thr={SCAN_ALIGNED_THR_DEG}°)",
                              flush=True)
            else:
                # NX-2: gate rejected this detection -- treat this cycle like a miss.
                self.frames_since_detection += 1
                if self.last_known_goal is not None and self.frames_since_detection <= HOLD_GOAL_HORIZON:
                    self.cached_goal_vec = self.last_known_goal.copy()
                elif self.lock_gate.coast_expired(self.frames_since_detection, HOLD_GOAL_HORIZON):
                    if self.verbose:
                        print(f"  [lock] M5 coast expired (gate-rejected) -> "
                              f"drop+rescan at step={step}", flush=True)
                    self._lock_drop_and_rescan()
        else:
            self.frames_since_detection += 1
            # Hold last-known goal if recently seen, else keep cached (straight-ahead initially)
            if self.last_known_goal is not None and self.frames_since_detection <= HOLD_GOAL_HORIZON:
                self.cached_goal_vec = self.last_known_goal.copy()
            # V2 (unchanged when LOCK_M5 off): keep whatever cached_goal_vec was
            # (straight-ahead default or last ema) -- silent freeze forever.
            elif self.lock_gate.coast_expired(self.frames_since_detection, HOLD_GOAL_HORIZON):
                # NX-2 (LOCK_M5): bounded coast -> reroute to rescan instead of an
                # unbounded silent freeze.
                if self.verbose:
                    print(f"  [lock] M5 coast expired -> drop+rescan at step={step}",
                          flush=True)
                self._lock_drop_and_rescan()

    def process_learned_detection(self, raw_gr: np.ndarray) -> None:
        """Ingests one learned-grounding-head prediction (EMA only — no
        frames_since_detection/scan-exit bookkeeping, matching the original
        inline learned-grounding block exactly)."""
        if self.goal_ema is None:
            self.goal_ema        = raw_gr.copy()
            self.last_known_goal = raw_gr.copy()
        else:
            self.goal_ema = GOAL_EMA_ALPHA * raw_gr + (1.0 - GOAL_EMA_ALPHA) * self.goal_ema
            th = math.atan2(self.goal_ema[2], self.goal_ema[1])
            self.goal_ema[1] = math.cos(th)
            self.goal_ema[2] = math.sin(th)
            self.last_known_goal = self.goal_ema.copy()
        self.cached_goal_vec = self.goal_ema.copy()

    # ------------------------------------------------------------------
    # NX-2/NX-5 end-of-cycle divergence + odometry-coherence watchdog
    # ------------------------------------------------------------------
    def end_of_cycle_lock_check(self, data_mj, yaw: float, stop_r: float, step: int) -> None:
        """Runs the M4 (divergence) / M7 (odometry-coherence) watchdog once
        per classical grounding cycle, using the resolved best-estimate
        distance for this cycle. Provable no-op when LOCK_M4/LOCK_M7 are off.

        NX-5 (LOCK_M7, docs/nx5_coherence.md): projects the robot's own
        measured world-frame displacement SINCE THE LAST GROUNDING CYCLE
        onto the current goal bearing (robot body frame: cached_goal_vec's
        cos_th/sin_th are egocentric, so rotate the world displacement by
        -yaw before dotting with them). This is "measured odometric
        displacement" per the design brief -- the robot's own pose, not the
        target's, exactly what a real state-estimator would give.
        """
        walking_toward_goal = (not self.scan_active) and (float(self.cached_goal_vec[0]) > stop_r)
        m7_proj_disp_m = 0.0
        cur_xy = data_mj.qpos[0:2].copy()
        if self.m7_prev_xy is not None:
            dxw = float(cur_xy[0] - self.m7_prev_xy[0])
            dyw = float(cur_xy[1] - self.m7_prev_xy[1])
            cy, sy = math.cos(yaw), math.sin(yaw)
            d_body_x =  dxw * cy + dyw * sy
            d_body_y = -dxw * sy + dyw * cy
            m7_proj_disp_m = (d_body_x * float(self.cached_goal_vec[1])
                               + d_body_y * float(self.cached_goal_vec[2]))
        self.m7_prev_xy = cur_xy
        if self.lock_gate.end_of_cycle(float(self.cached_goal_vec[0]), walking_toward_goal,
                                        m7_proj_disp_m):
            if self.verbose:
                print(f"  [lock] {self.lock_gate.last_trigger} "
                      f"{'divergence' if self.lock_gate.last_trigger == 'M4' else 'coherence'} "
                      f"-> drop+rescan at step={step}", flush=True)
            self._lock_drop_and_rescan()

    # ------------------------------------------------------------------
    # NX-9 AVOID: per-cycle obstacle-bias update
    # ------------------------------------------------------------------
    def update_avoid_bias(self, depth, intr_active, data_mj, step: int) -> None:
        """Updates the local-obstacle-avoidance yaw-rate bias for this cycle.

        Reuses THIS cycle's already-rendered `depth`/`intr_active` (zero
        extra renders) -- runs at the same grounding cadence, AFTER
        cached_goal_vec is finalized for this cycle (so the target-exemption
        window and the "goal dist < 1.2m" carve-out both see the up-to-date
        goal). Carve-outs (docs/nx9_avoid.md §1.3): never while `scan_active`
        (the caller only invokes this on the non-scan path); a fresh bias is
        only computed while the goal is FRESH (`frames_since_detection` <=
        AVOID_STALE_MAX_MISSED_CYCLES) -- on a longer stale coast the
        existing bias only decays.
        """
        if AVOID and not self.avoid_is_maneuver and not self.scan_active:
            self.avoid_cycles_total += 1
            if self.frames_since_detection > _avoid.AVOID_STALE_MAX_MISSED_CYCLES:
                self.avoid_bias_wz = _avoid.decay_bias(self.avoid_bias_wz)
            else:
                avoid_goal_dist_now = float(self.cached_goal_vec[0])
                avoid_goal_bearing_now = math.atan2(float(self.cached_goal_vec[2]),
                                                     float(self.cached_goal_vec[1]))
                avoid_carved = (avoid_goal_dist_now < _avoid.AVOID_MIN_GOAL_DIST_M)
                avoid_cam_h = float(data_mj.qpos[2]) + CAM_HEAD_Z
                self.avoid_bias_wz, avoid_dbg = _avoid.compute_obstacle_bias(
                    depth, intr_active, cam_height_m=avoid_cam_h,
                    goal_dist_m=avoid_goal_dist_now,
                    goal_bearing_rad=avoid_goal_bearing_now,
                    prev_bias_wz=self.avoid_bias_wz, carved_out=avoid_carved)
                if self.verbose and abs(self.avoid_bias_wz) > 1e-9:
                    print(f"  [avoid] bias_wz={self.avoid_bias_wz:+.3f} "
                          f"L={avoid_dbg['left']:.2f} R={avoid_dbg['right']:.2f} "
                          f"n_px={avoid_dbg['n_obstacle_px']} step={step}", flush=True)
            if abs(self.avoid_bias_wz) > 1e-9:
                self.avoid_cycles_active += 1

    # ------------------------------------------------------------------
    # H3 scan-and-acquire — STUDENT-DRIVEN (WBC-free)
    # ------------------------------------------------------------------
    def try_scan_step(self, yaw: float, step: int) -> float | None:
        """Advances the scan state machine by one step; returns the commanded
        turning wz if the caller should execute a scan-mode student forward
        pass this step (and then skip the normal-mode pass), or None if
        normal-mode should run instead (scan inactive, or it just ended).

        NX-10 (docs/nx10_scan_fix.md): pattern is NX-1's bidirectional bounded
        triangle-wave (`h3_scan_sched`, code/scan_sched.py's
        `BidirectionalScanSchedule` class, H3-local `H3_LEG_DEG=90` amplitude)
        -- 0->+90° (CCW), dwell, +90->0° (CW), dwell, 0->-90° (CW), dwell,
        repeat -- tracking the robot's ACTUAL accumulated yaw each step
        rather than assuming the commanded SCAN_RATE is fully realized.
        Timeout: after SCAN_TIMEOUT (absolute episode step) elapses, exit
        scan and use default/last cached_goal_vec, same fallback as before.

        NX-2 (LOCK_M4/M5): a lock-drop-triggered rescan uses a FRESH
        ReacquisitionScan (its own LOCAL step counter) instead of the H3 scan,
        because H3's SCAN_TIMEOUT check is keyed on the EPISODE's absolute
        `step` -- re-arming it mid-episode would immediately time out (step
        already >> SCAN_TIMEOUT). This branch is ONLY ever taken after a
        M4/M5 trigger (both individually toggled, default off); with those
        off, `using_rescan_sched` is never True and the H3 scan always runs.
        """
        if not (self.scan_active and self.need_classical_render):
            return None
        if self.using_rescan_sched:
            scan_wz = self.rescan_sched.step(yaw)
            if scan_wz is None:
                self.scan_active        = False
                self.using_rescan_sched = False
                if self.verbose:
                    print(f"  [lock][rescan] TIMEOUT at step={step}, "
                          f"falling back to default goal", flush=True)
                return None
            return scan_wz
        elif step >= SCAN_TIMEOUT:
            self.scan_active = False
            if self.verbose:
                print(f"  [scan] TIMEOUT at step={step}, falling back to default goal", flush=True)
            return None
        else:
            # NX-10: bounded bidirectional schedule, driven by REALIZED yaw (see
            # module docstring for the diagnosis). Dwell legs return wz=0.0
            # (in-distribution stand-still, matches the ReacquisitionScan /
            # eval_search dwell behavior).
            scan_wz = self.h3_scan_sched.step(yaw)
            self.scan_yaw_delta += scan_wz * SCAN_DT
            return scan_wz

    # ------------------------------------------------------------------
    def _lock_drop_and_rescan(self) -> None:
        """M4 (divergence) / M5 (coast-expiry) shared action: drop the lock,
        clear EMA/last-known-goal, and re-enter scan via NX-1's bounded
        BidirectionalScanSchedule (never unbounded rotation -- see
        code/lock_mgmt.py's ReacquisitionScan docstring for why this can't
        just reuse the H3 scan's own absolute-step timeout)."""
        self.lock_gate.force_drop()
        self.goal_ema             = None
        self.last_known_goal      = None
        self.frames_since_detection = 0
        self.scan_active          = True
        self.using_rescan_sched   = True
        self.rescan_sched         = ReacquisitionScan(scan_rate=SCAN_RATE)
        self.cached_goal_vec      = np.array([2.0, 1.0, 0.0], dtype=np.float32)
        self.avoid_bias_wz        = 0.0   # NX-9: fresh depth read once normal mode resumes
