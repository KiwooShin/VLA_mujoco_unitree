"""
code/perception/lock_gate.py — NX-2/NX-5 target-lock state machine (RF-1
split of code/lock_mgmt.py; docs/refactor_plan.md).

See code/perception/lock_mgmt.py's module docstring for the full design
rationale (docs/rs1_lock_mgmt.md, docs/nx2_final.md, docs/nx5_coherence.md) --
this file holds the toggle env-vars, calibrated constants, and the
`LockGate` state machine itself; code/perception/lock_rescan.py holds the
bounded-rescan wrapper `ReacquisitionScan`.

    LOCK_M1=0   disables the area-quality floor (raw contour px^2, GroundingResult.best_area) -- ON by default
    LOCK_M2=1   enables N-of-M (2-of-3) tentative -> confirmed lock initiation -- OFF by default (REJECT, docs/nx2_iso.md)
    LOCK_M3=0   disables the innovation gate + incumbent inertia (association gating) -- ON by default
    LOCK_M4=1   enables the divergence watchdog (drop + rescan on a monotonic dist trend) -- OFF by default (REJECT, docs/nx2_iso.md)
    LOCK_M5=1   enables bounded coast -> reroute to rescan after hold-goal-horizon expiry -- OFF by default (bundled w/ REJECTed M2, docs/nx2_iso.md)
    LOCK_M7=1   enables the odometry-coherence watchdog (drop + rescan when walked
                displacement toward the goal bearing isn't matched by a commensurate
                goal-distance shrink) -- OFF by default (NX-5, docs/nx5_coherence.md,
                pending gate verdict)
"""

from __future__ import annotations

import math
import os
from collections import deque


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() == "1"


# ---------------------------------------------------------------------------
# Toggles (independent; M1/M3 default ON (opt-out), M2/M4/M5 default OFF
# (opt-in) -- see docs/nx2_final.md for the KEEP/REJECT verdicts behind this)
# ---------------------------------------------------------------------------
# M1/M3: KEEP-verdicted (docs/nx2_iso.md, docs/nx2_final.md) -> default ON,
# opt-out via LOCK_M1=0 / LOCK_M3=0.
LOCK_M1 = _env_flag("LOCK_M1", default="1")   # area-quality floor
LOCK_M3 = _env_flag("LOCK_M3", default="1")   # innovation gate + incumbent inertia
# M2/M4/M5: REJECT-verdicted (docs/nx2_iso.md) -> stay default OFF, opt-in.
LOCK_M2 = _env_flag("LOCK_M2")   # N-of-M tentative->confirmed
LOCK_M4 = _env_flag("LOCK_M4")   # divergence watchdog
LOCK_M5 = _env_flag("LOCK_M5")   # bounded coast -> reroute to rescan
# M7: NX-5 (docs/nx5_coherence.md), pending gate verdict -> default OFF, opt-in.
LOCK_M7 = _env_flag("LOCK_M7")   # odometry-coherence watchdog


# ---------------------------------------------------------------------------
# Constants (docs/rs1_lock_mgmt.md §2, tuned/validated per docs/nx2_impl.md)
# ---------------------------------------------------------------------------

# --- M1: area-quality floor ---
# Raw contour area (px^2, GroundingResult.best_area -- the SAME quantity that
# feeds conf_area in grounding.py's confidence formula), not the bbox w*h
# proxy the design brief floated as a lower-cost alternative: an empirical
# instrumented check (docs/nx2_impl.md) showed bbox w*h is NOT a reliable
# stand-in here -- ep0/ep5's false-positive blobs are thin/irregular slivers
# with a large bounding box but small true contour area (exactly why their
# conf_area is near-zero in the first place), so a bbox-area floor could not
# cleanly separate them from legitimate small-but-compact far-range blobs.
# Value: instrumented ground() across demo eps 1/3/6/9 (all currently-PASSING
# long-range episodes, eval/p4_gate_demo) recorded a global minimum accepted
# contour area of 123.5 px^2 (ep3, red cube, dist=1.57m during final
# approach). 100.0 sits ~19% below that floor -- zero risk of rejecting any
# detection observed in those 4 episodes' full rollouts -- while still well
# above the raw MIN_BLOB_AREA=40px detection floor (2.5x) and rejecting the
# most degenerate near-noise blobs (e.g. ep2's 44/60 px^2 false positives).
# NOTE (documented limitation): this floor gives PARTIAL protection against
# ep0/ep5's specific failure -- their false lock's STEADY-STATE accepted area
# (median 522-9194 px^2 across the instrumented reruns) is well above any
# floor that doesn't also reject legitimate far detections, so M1 mainly
# trims the worst transient slivers rather than eliminating the false lock
# outright. This matches the design brief's own MEDIUM-confidence framing for
# M1 (docs/rs1_lock_mgmt.md §2 ranking #3) -- it is defense-in-depth alongside
# M2/M4, not a standalone fix.
M1_AREA_FLOOR_PX2 = 100.0

# --- M2: N-of-M tentative -> confirmed ---
M2_CONFIRM_M       = 2      # of the last N cycles must be mutually consistent
M2_CONFIRM_N       = 3
M2_TOL_DIST_M      = 0.6    # metres
M2_TOL_BEARING_DEG = 12.0   # degrees

# --- M3: innovation gate + incumbent inertia ---
M3_GATE_BEARING_DEG              = 25.0   # base bearing innovation gate (deg)
M3_GATE_BEARING_NEAR_MULT        = 1.5    # multiplier below M3_NEAR_RANGE_M
M3_NEAR_RANGE_M                  = 2.0    # metres (proximity-handoff band)
M3_GATE_DIST_FLOOR_M             = 0.8    # metres, absolute floor
M3_GATE_DIST_CLOSING_MULT        = 1.5
M3_EXPECTED_CLOSING_M_PER_CYCLE  = 0.16   # ~0.8 m/s * 0.2 s/cycle (upper end
                                          # of the task brief's 0.5-0.8 m/s
                                          # walk-speed range) -- in practice
                                          # 0.16*1.5=0.24 < the 0.8m floor, so
                                          # the floor is what actually binds;
                                          # kept symbolic for documentation
                                          # fidelity / future re-tuning.
M3_INCUMBENT_MARGIN              = 1.3    # challenger area >= 1.3x incumbent
M3_INCUMBENT_K                   = 2      # sustained cycles required to replace

# --- M4: divergence watchdog ---
M4_WINDOW_N                     = 15      # grounding cycles (~3s @ 5Hz)
M4_TREND_MARGIN_M               = 0.5     # metres
M4_EXEMPT_CYCLES_AFTER_CONFIRM  = 15       # cycles after any (re)confirmation
M4_EXEMPT_CYCLES_AROUND_HANDOFF = 2        # +/- cycles around a legit discontinuity

# --- M7: odometry-coherence watchdog (docs/nx5_coherence.md) ---
# Sliding DISTANCE window (not a cycle/step count -- M4's documented fragility,
# docs/nx2_final.md: a burst of heading-correction transients tripped M4's
# 15-cycle window on ep13 without the robot actually diverging). Once the
# CONFIRMED lock's commanded/measured displacement projected toward the goal
# bearing accumulates to M7_X_WALK_M while walking, the goal distance must
# have shrunk by at least M7_K_MIN_FRAC * M7_X_WALK_M over the same window --
# a real static target satisfies this by construction (the robot is walking
# toward it); a fixed wrong-depth pseudo-point (ep0/ep5's ~5.45m sliver blob
# vs a true 3.3-3.5m target, docs/fa1_failures.md) does not.
M7_X_WALK_M               = 1.75   # metres -- task brief's 1.5-2.0m band, midpoint
M7_K_MIN_FRAC             = 0.4    # required shrink = K_MIN_FRAC * X_WALK_M
M7_MIN_GOAL_DIST_M        = 1.5    # endgame carve-out: suspend below this (proximity-cam territory)
# Short-term (not hard-block) re-lock penalty after an M7 trigger -- prevents
# an instant same-blob re-confirm (the failure that killed M4, docs/nx2_final.md).
M7_PENALTY_CYCLES         = 50     # cycles the penalty stays active
M7_PENALTY_BEARING_DEG    = 10.0   # +/- degrees around the dropped lock's bearing
M7_PENALTY_DIST_TOL_M     = 0.75   # metres tolerance around the dropped lock's distance
M7_PENALTY_CONFIRM_M      = 2      # of the last N cycles must be mutually consistent...
M7_PENALTY_CONFIRM_N      = 2      # ...to re-confirm while inside the penalty zone
M7_PENALTY_TOL_DIST_M     = 0.6    # metres (corroboration tolerance, reuses M2's values)
M7_PENALTY_TOL_BEARING_DEG = 12.0  # degrees (corroboration tolerance, reuses M2's values)


def _ang_diff_rad(a: float, b: float) -> float:
    """Signed angular difference a-b, wrapped to (-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


# ---------------------------------------------------------------------------
# LockGate — the shared per-episode state machine
# ---------------------------------------------------------------------------
class LockGate:
    """
    One instance per rollout. State: 'NONE' | 'CONFIRMED' (the brief's
    'TENTATIVE' and 'COASTING' states are folded in here: TENTATIVE is just
    'NONE' plus the M2 ring buffer; COASTING is the caller's own existing
    hold-goal/`_frames_since_detection` bookkeeping -- LockGate doesn't need
    a separate flag for it, `coast_expired()` below is the only thing that
    cares).

    Call sequence per grounding cycle (both callers follow this):
      1. If the CAM-2 fallback probe adopts the other camera's reading, or
         the `_active_cam` Schmitt trigger flips, call `mark_discontinuity()`
         (inferencer.py only -- eval_search.py has no second camera).
      2. If a raw detection exists this cycle, call
         `gate_detection(dist, bearing_rad, area)` BEFORE updating the
         caller's own EMA/last-known-goal -- only update them if it returns
         True; otherwise treat the cycle as a miss (matches the "gate
         rejected -> don't feed EMA" design).
      3. If no raw detection (or the gate rejected it) and the caller's own
         hold-goal-horizon has just been exceeded, call
         `coast_expired(frames_since_detection, horizon)`; if True, drop the
         lock (force_drop()) and re-enter scan via a fresh
         `ReacquisitionScan` instead of freezing forever.
      4. Exactly once per grounding cycle (regardless of hit/miss/reject),
         call `end_of_cycle(current_best_dist_estimate, walking, proj_disp_m)`
         (the `proj_disp_m` arg is M7-only -- commanded/measured base
         displacement since the last call, projected onto the current goal
         bearing; omit/None if not computing it). If True, M4 or M7 has
         fired (`self.last_trigger` says which) -- drop the lock and
         re-enter scan, same action as (3).
    """

    def __init__(self) -> None:
        """Initialize a fresh, unlocked ('NONE') state machine."""
        self.state              = 'NONE'
        self._m2_hist           = deque(maxlen=M2_CONFIRM_N)
        self._incumbent         = None    # dict(dist, bearing, area)
        self._challenger_streak = 0
        self._dist_hist         = deque(maxlen=M4_WINDOW_N)
        self._confirm_cycle     = None
        self._cycle_count       = 0
        self._discontinuity_cooldown = 0
        self.last_trigger       = None    # 'M4' | 'M7' | None -- diagnostics only

        # M7 accumulation window state.
        self._m7_accum          = 0.0
        self._m7_window_dist0   = None
        # M7 short-term re-lock penalty state (persists across force_drop() --
        # it is deliberately keyed on the DROPPED lock, not the current one).
        self._m7_penalty_bearing       = None
        self._m7_penalty_dist          = None
        self._m7_penalty_expire_cycle  = None
        self._m7_penalty_hist          = deque(maxlen=M7_PENALTY_CONFIRM_N)

    # -- carve-out signal (docs/rs1_lock_mgmt.md risk #2) --
    def mark_discontinuity(self, cooldown_cycles: int = M4_EXEMPT_CYCLES_AROUND_HANDOFF) -> None:
        """
        Call at the exact cycle a legitimate (dist,bearing) discontinuity
        occurs: the CAM-2 fallback probe-adopt event (`gr = gr2`) or an
        `_active_cam` Schmitt-trigger flip. Bypasses M3's innovation gate for
        this cycle's `gate_detection()` call and suppresses M4's trigger for
        the next `cooldown_cycles` cycles. Also clears the M4 distance
        window -- pre-event samples are from a different camera's geometry
        and would otherwise corrupt the post-event window-min comparison
        once the cooldown lapses. Safe/no-op cost-wise regardless of toggle
        state (pure bookkeeping; only read when LOCK_M3/LOCK_M4 are on).
        """
        self._discontinuity_cooldown = max(self._discontinuity_cooldown, cooldown_cycles)
        self._dist_hist.clear()
        # M7: pre/post-handoff distances are from different camera geometries
        # (docs/nx5_coherence.md carve-out) -- reset the accumulation window so
        # it restarts cleanly from the first post-handoff cycle instead of
        # comparing across the discontinuity.
        self._m7_accum        = 0.0
        self._m7_window_dist0 = None

    def _confirm(self, entry: dict) -> None:
        self.state              = 'CONFIRMED'
        self._incumbent         = entry
        self._confirm_cycle     = self._cycle_count
        self._challenger_streak = 0
        self._m2_hist.clear()
        # M7: (re)start the coherence-accumulation window fresh from this
        # confirm's own distance -- never carry a stale reference across a
        # fresh/re acquisition.
        self._m7_accum          = 0.0
        self._m7_window_dist0   = entry['dist']

    def _m7_in_penalty_zone(self, dist: float, bearing_rad: float) -> bool:
        """True iff an active M7 re-lock penalty exists and (dist, bearing)
        falls within its bearing/distance tolerance of the dropped lock."""
        if self._m7_penalty_expire_cycle is None:
            return False
        if self._cycle_count >= self._m7_penalty_expire_cycle:
            return False
        d_bearing = abs(_ang_diff_rad(bearing_rad, self._m7_penalty_bearing))
        d_dist    = abs(dist - self._m7_penalty_dist)
        return (d_bearing <= math.radians(M7_PENALTY_BEARING_DEG) and
                d_dist    <= M7_PENALTY_DIST_TOL_M)

    def gate_detection(self, dist: float, bearing_rad: float, area: float | None) -> bool:
        """
        Decide whether a raw detection this cycle should be treated as an
        accepted "hit" (caller should feed it to the EMA / last-known-goal)
        or a rejection (caller should treat this cycle like a miss).

        With LOCK_M1/M2/M3 all explicitly set to 0 this is a provable
        pass-through (always True), matching the pre-NX-2 legacy behavior
        documented in docs/nx2_impl.md: M1's check is skipped; M2's branch
        immediately confirms on the very first call from 'NONE' (matching the
        old "first detection above the raw floor seeds the lock" behavior);
        M3's branch unconditionally accepts and refreshes the incumbent every
        call once CONFIRMED (matching the old "every accepted detection
        updates the EMA" behavior, no gating). Under the shipped defaults
        (M1/M3 ON, M2 OFF), M1's area floor and M3's innovation gate are both
        live -- see docs/nx2_final.md for the validated combined-gate result.
        """
        if LOCK_M1 and area is not None and area < M1_AREA_FLOOR_PX2:
            return False

        entry = dict(dist=dist, bearing=bearing_rad, area=area)

        if self.state != 'CONFIRMED':
            # M7 (docs/nx5_coherence.md): short-term, NOT hard-block, re-lock
            # penalty. A fresh candidate landing near the (bearing, dist) of a
            # lock M7 JUST dropped needs M7_PENALTY_CONFIRM_M-of-N corroborating
            # cycles instead of the usual instant single-frame confirm -- this
            # is specifically what M4 lacked (docs/nx2_final.md) and let a
            # dropped false lock re-seed itself on the very next frame.
            # Outside the penalty zone (different bearing/dist, or the penalty
            # has expired), behaviour is completely unchanged.
            if LOCK_M7 and self._m7_in_penalty_zone(dist, bearing_rad):
                self._m7_penalty_hist.append(entry)
                latest = self._m7_penalty_hist[-1]
                tol_bearing_rad = math.radians(M7_PENALTY_TOL_BEARING_DEG)
                n_consistent = sum(
                    1 for h in self._m7_penalty_hist
                    if abs(h['dist'] - latest['dist']) < M7_PENALTY_TOL_DIST_M
                    and abs(_ang_diff_rad(h['bearing'], latest['bearing'])) < tol_bearing_rad
                )
                if n_consistent >= M7_PENALTY_CONFIRM_M:
                    self._confirm(entry)
                    return True
                return False
            if not LOCK_M2:
                self._confirm(entry)
                return True
            self._m2_hist.append(entry)
            latest = self._m2_hist[-1]
            tol_bearing_rad = math.radians(M2_TOL_BEARING_DEG)
            n_consistent = sum(
                1 for h in self._m2_hist
                if abs(h['dist'] - latest['dist']) < M2_TOL_DIST_M
                and abs(_ang_diff_rad(h['bearing'], latest['bearing'])) < tol_bearing_rad
            )
            if n_consistent >= M2_CONFIRM_M:
                self._confirm(entry)
                return True
            return False

        # state == 'CONFIRMED'
        bypass = self._discontinuity_cooldown > 0
        if not LOCK_M3 or bypass:
            self._incumbent         = entry
            self._challenger_streak = 0
            if bypass:
                # Legit discontinuity also counts as a fresh (re)confirmation
                # for M4's post-confirm exemption window.
                self._confirm_cycle = self._cycle_count
            return True

        inc = self._incumbent
        near = inc['dist'] < M3_NEAR_RANGE_M
        bearing_gate_rad = math.radians(
            M3_GATE_BEARING_DEG * (M3_GATE_BEARING_NEAR_MULT if near else 1.0))
        dist_gate_m = max(M3_GATE_DIST_FLOOR_M,
                          M3_EXPECTED_CLOSING_M_PER_CYCLE * M3_GATE_DIST_CLOSING_MULT)
        d_bearing = abs(_ang_diff_rad(bearing_rad, inc['bearing']))
        d_dist    = abs(dist - inc['dist'])

        if d_bearing <= bearing_gate_rad and d_dist <= dist_gate_m:
            # Within the innovation gate -- normal track continuation.
            self._incumbent         = entry
            self._challenger_streak = 0
            return True

        # Outside the gate: only accept as a challenger if it sustains a
        # real quality margin over the incumbent for K consecutive cycles.
        inc_area = inc['area'] or 0.0
        beats_incumbent = (inc_area <= 0.0) or (
            area is not None and area >= M3_INCUMBENT_MARGIN * inc_area)
        if beats_incumbent:
            self._challenger_streak += 1
            if self._challenger_streak >= M3_INCUMBENT_K:
                self._incumbent         = entry
                self._challenger_streak = 0
                return True
            return False
        self._challenger_streak = 0
        return False

    def end_of_cycle(self, best_dist_estimate: float, walking: bool,
                      proj_disp_m: float | None = None) -> bool:
        """
        Call exactly once per grounding cycle, regardless of hit/miss/gate
        outcome, with the caller's current best distance estimate for this
        cycle (its EMA'd/held goal distance) and (for M7) the commanded/
        measured base displacement since the last call, PROJECTED onto the
        current goal bearing (metres, may be negative; None/0 if unavailable
        or not applicable, e.g. the very first cycle). Returns True iff M4
        or M7 has detected a bad lock and the caller should force_drop() +
        re-enter scan; `self.last_trigger` records which one ('M4'|'M7') for
        logging/diagnostics.

        With LOCK_M4/LOCK_M7 both off this always returns False (window
        bookkeeping still runs harmlessly for internal state consistency but
        has no externally observable effect).

        M7 (docs/nx5_coherence.md): sliding DISTANCE window, not a cycle/step
        count (M4's documented fragility -- a burst of heading-correction
        transients can trip a fixed-cycle window without the robot actually
        diverging, docs/nx2_final.md). Accumulates `proj_disp_m` while
        CONFIRMED + walking + no discontinuity cooldown + goal distance >=
        M7_MIN_GOAL_DIST_M (endgame/proximity-cam carve-out); once the
        accumulation reaches M7_X_WALK_M, the goal distance must have shrunk
        by >= M7_K_MIN_FRAC * M7_X_WALK_M over that same window or M7 fires.
        A passed check SLIDES the window forward (reset accum/reference to
        the current cycle) rather than freezing a one-shot check.
        """
        self._cycle_count += 1
        triggered = False

        if self.state == 'CONFIRMED':
            self._dist_hist.append(best_dist_estimate)
            if LOCK_M4 and walking and self._discontinuity_cooldown == 0:
                recently_confirmed = (
                    self._confirm_cycle is not None and
                    (self._cycle_count - self._confirm_cycle) < M4_EXEMPT_CYCLES_AFTER_CONFIRM)
                if not recently_confirmed and len(self._dist_hist) >= M4_WINDOW_N:
                    window_min = min(self._dist_hist)
                    if (best_dist_estimate - window_min) > M4_TREND_MARGIN_M:
                        triggered = True
                        self.last_trigger = 'M4'

            if (LOCK_M7 and walking and self._discontinuity_cooldown == 0
                    and best_dist_estimate >= M7_MIN_GOAL_DIST_M):
                if self._m7_window_dist0 is None:
                    # First eligible cycle after a confirm/discontinuity
                    # reset/endgame suspension -- (re)start the window here.
                    self._m7_window_dist0 = best_dist_estimate
                    self._m7_accum        = 0.0
                else:
                    self._m7_accum += (proj_disp_m or 0.0)
                    if self._m7_accum >= M7_X_WALK_M:
                        shrink   = self._m7_window_dist0 - best_dist_estimate
                        required = M7_K_MIN_FRAC * M7_X_WALK_M
                        if shrink >= required:
                            # Coherent -- slide the window forward.
                            self._m7_window_dist0 = best_dist_estimate
                            self._m7_accum        = 0.0
                        else:
                            # Incoherent: walked >= X_WALK toward the bearing
                            # without the distance shrinking commensurately --
                            # a static wrong-depth pseudo-point, not a real
                            # approach. Snapshot the dropped lock's
                            # (bearing, dist) for the short-term re-lock
                            # penalty BEFORE the caller's force_drop() clears
                            # `_incumbent`.
                            if self._incumbent is not None:
                                self._m7_penalty_bearing = self._incumbent['bearing']
                                self._m7_penalty_dist    = self._incumbent['dist']
                                self._m7_penalty_expire_cycle = (
                                    self._cycle_count + M7_PENALTY_CYCLES)
                                self._m7_penalty_hist.clear()
                            triggered = True
                            self.last_trigger = 'M7'
        else:
            self._dist_hist.clear()
            # M7's window is only meaningful while CONFIRMED and is properly
            # (re)started in `_confirm()`; clear defensively so a stale
            # window can't leak into a future confirm via a code-path we
            # haven't anticipated.
            self._m7_accum        = 0.0
            self._m7_window_dist0 = None

        if self._discontinuity_cooldown > 0:
            self._discontinuity_cooldown -= 1
        return triggered

    def coast_expired(self, frames_since_detection: int, hold_goal_horizon: int) -> bool:
        """
        M5: returns True iff the caller's hold-goal horizon has just been
        exceeded AND LOCK_M5 is on -- caller should force_drop() + re-enter
        scan instead of silently freezing `cached_goal_vec` forever. With
        LOCK_M5 off, always False (matches the pre-NX-2 behavior of an
        unbounded freeze).
        """
        return LOCK_M5 and frames_since_detection > hold_goal_horizon

    def force_drop(self) -> None:
        """Reset to 'NONE' -- caller must separately clear its own
        _goal_ema/_last_known_goal and re-enter scan (see ReacquisitionScan).

        Deliberately does NOT touch the M7 penalty state
        (_m7_penalty_bearing/_dist/_expire_cycle/_hist): that state is keyed
        on the just-DROPPED lock and must survive exactly this call so the
        short-term re-lock penalty (docs/nx5_coherence.md) can apply to
        whatever the caller re-locks onto next."""
        self.state              = 'NONE'
        self._m2_hist.clear()
        self._incumbent         = None
        self._challenger_streak = 0
        self._dist_hist.clear()
        self._confirm_cycle     = None
        self._m7_accum          = 0.0
        self._m7_window_dist0   = None
