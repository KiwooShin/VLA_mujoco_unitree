"""
lock_mgmt.py — NX-2 target-lock management (docs/rs1_lock_mgmt.md).

Shared helper module implementing the 5 lock-stability mechanisms from the
design brief, imported identically by `code/inferencer.py` (easy/demo/maneuver)
and `code/eval_search.py` (search) -- the EMA/hold-goal/scan machinery in those
two files was already flagged as duplicated-and-drifting (see NX-1's
`code/scan_sched.py`, which fixed the same class of problem for scan logic);
this module follows the same pattern for lock management instead of adding a
third copy-paste.

Each mechanism is gated by its OWN independent env var. Per the combined/
cross-skill re-gate (docs/nx2_final.md), M1 and M3 are KEEP-verdicted and now
DEFAULT ON (opt-out); M2/M4/M5 were REJECT-verdicted (regressed previously-
passing episodes without fixing their target episodes) and remain default OFF
(opt-in):

    LOCK_M1=0   disables the area-quality floor (raw contour px^2, GroundingResult.best_area) -- ON by default
    LOCK_M2=1   enables N-of-M (2-of-3) tentative -> confirmed lock initiation -- OFF by default (REJECT, docs/nx2_iso.md)
    LOCK_M3=0   disables the innovation gate + incumbent inertia (association gating) -- ON by default
    LOCK_M4=1   enables the divergence watchdog (drop + rescan on a monotonic dist trend) -- OFF by default (REJECT, docs/nx2_iso.md)
    LOCK_M5=1   enables bounded coast -> reroute to rescan after hold-goal-horizon expiry -- OFF by default (bundled w/ REJECTed M2, docs/nx2_iso.md)

With M1/M3 at their new ON defaults and M2/M4/M5 left at their OFF defaults,
every public method below matches the fully-validated `eval/nx2_combined_*`
gate results (demo 10/15, easy 15/15, search 14/15 -- see docs/nx2_final.md).
Setting all five of LOCK_M1..LOCK_M5=0 reproduces the pre-NX-2 byte-identical
pass-through behavior (see per-method docstrings) documented in
docs/nx2_impl.md. This is deliberate: it means the two call sites only need
ONE extra function call per decision point, not a maze of conditionals, and
both the "all-off legacy" and "shipped-defaults" properties are enforced
structurally rather than by convention.

Mandatory carve-outs (docs/rs1_lock_mgmt.md risk #2): M3's innovation gate and
M4's divergence watchdog must NOT fire on the two legitimate (dist,bearing)
discontinuities this codebase already knows about -- the CAM-2 fallback
probe-adopt event and an `_active_cam` Schmitt-trigger flip (both only exist
in inferencer.py; eval_search.py has no second camera and never calls
`mark_discontinuity()`). Callers signal these via `LockGate.mark_discontinuity()`
at the point the event happens; the resulting cooldown window bypasses M3's
gate for new detections and suppresses M4's trigger, and clears the M4 dist
window so stale pre-event samples cannot corrupt the post-event trend check.

Rescans triggered by M4/M5 reuse NX-1's `BidirectionalScanSchedule`
(`code/scan_sched.py`) via `ReacquisitionScan` below -- never an unbounded
spin. This is a SEPARATE small wrapper from the callers' own initial-scan
mechanisms (inferencer.py's absolute-step H3 sweep, eval_search.py's own
`BidirectionalScanSchedule` instance) because both of those gate their
timeout off the EPISODE's absolute step counter; re-arming either mid-episode
would immediately "time out" since the absolute step is already well past
their timeout constant by the time a mid-episode rescan can trigger.
`ReacquisitionScan` tracks its own LOCAL step counter from the moment it is
constructed, so it is safe to instantiate fresh at any point in an episode.

See docs/nx2_impl.md for the full write-up, chosen constants, and the
empirical evidence behind the M1 floor value.
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Optional

from code.scan_sched import (BidirectionalScanSchedule, SCAN_LEG_DEG,
                              SCAN_DWELL_STEPS, SCAN_TIMEOUT as _RESCAN_TIMEOUT_STEPS)


# ---------------------------------------------------------------------------
# Toggles (independent; M1/M3 default ON (opt-out), M2/M4/M5 default OFF
# (opt-in) -- see docs/nx2_final.md for the KEEP/REJECT verdicts behind this)
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() == "1"


# M1/M3: KEEP-verdicted (docs/nx2_iso.md, docs/nx2_final.md) -> default ON,
# opt-out via LOCK_M1=0 / LOCK_M3=0.
LOCK_M1 = _env_flag("LOCK_M1", default="1")   # area-quality floor
LOCK_M3 = _env_flag("LOCK_M3", default="1")   # innovation gate + incumbent inertia
# M2/M4/M5: REJECT-verdicted (docs/nx2_iso.md) -> stay default OFF, opt-in.
LOCK_M2 = _env_flag("LOCK_M2")   # N-of-M tentative->confirmed
LOCK_M4 = _env_flag("LOCK_M4")   # divergence watchdog
LOCK_M5 = _env_flag("LOCK_M5")   # bounded coast -> reroute to rescan


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
         call `end_of_cycle(current_best_dist_estimate, walking)`; if True,
         M4 has fired -- drop the lock and re-enter scan, same action as (3).
    """

    def __init__(self):
        self.state              = 'NONE'
        self._m2_hist           = deque(maxlen=M2_CONFIRM_N)
        self._incumbent         = None    # dict(dist, bearing, area)
        self._challenger_streak = 0
        self._dist_hist         = deque(maxlen=M4_WINDOW_N)
        self._confirm_cycle     = None
        self._cycle_count       = 0
        self._discontinuity_cooldown = 0

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

    def _confirm(self, entry: dict) -> None:
        self.state              = 'CONFIRMED'
        self._incumbent         = entry
        self._confirm_cycle     = self._cycle_count
        self._challenger_streak = 0
        self._m2_hist.clear()

    def gate_detection(self, dist: float, bearing_rad: float, area: Optional[float]) -> bool:
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

    def end_of_cycle(self, best_dist_estimate: float, walking: bool) -> bool:
        """
        Call exactly once per grounding cycle, regardless of hit/miss/gate
        outcome, with the caller's current best distance estimate for this
        cycle (its EMA'd/held goal distance). Returns True iff M4 has
        detected a divergent lock and the caller should force_drop() +
        re-enter scan.

        With LOCK_M4 off this always returns False (dist-window bookkeeping
        still runs harmlessly for internal state consistency but has no
        externally observable effect).
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
        else:
            self._dist_hist.clear()

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
        _goal_ema/_last_known_goal and re-enter scan (see ReacquisitionScan)."""
        self.state              = 'NONE'
        self._m2_hist.clear()
        self._incumbent         = None
        self._challenger_streak = 0
        self._dist_hist.clear()
        self._confirm_cycle     = None


# ---------------------------------------------------------------------------
# ReacquisitionScan — bounded rescan for M4/M5-triggered re-acquisition
# ---------------------------------------------------------------------------
class ReacquisitionScan:
    """
    Thin wrapper around NX-1's `BidirectionalScanSchedule` for a rescan
    triggered mid-episode by M4/M5. Deliberately NOT the same schedule
    instance/mechanism the caller uses for its OWN initial scan:
      - inferencer.py's initial "H3" scan gates its SCAN_TIMEOUT off the
        episode's ABSOLUTE step count (`if step >= SCAN_TIMEOUT`) -- re-arming
        it mid-episode (step already >> SCAN_TIMEOUT=200) would immediately
        "time out" without ever actually rotating.
      - eval_search.py's initial scan uses `BidirectionalScanSchedule`
        already, but its OWN outer timeout check is likewise gated off the
        absolute episode step, with the same bug for a mid-episode re-arm.
    `ReacquisitionScan` tracks its own LOCAL step counter starting from 0 at
    construction time, so it is always safe to instantiate fresh whenever
    M4/M5 drops a lock, regardless of how far into the episode that happens.
    Still fully bounded/never-unbounded-rotation, per the design brief's
    mandatory carve-out: it's the exact same `BidirectionalScanSchedule`
    class and constants (`SCAN_LEG_DEG`, `SCAN_DWELL_STEPS`) NX-1 validated.
    """

    def __init__(self, scan_rate: float = 0.6):
        self._sched = BidirectionalScanSchedule(
            scan_rate=scan_rate, leg_deg=SCAN_LEG_DEG, dwell_steps=SCAN_DWELL_STEPS)
        self._local_step = 0

    def step(self, current_yaw_rad: float) -> Optional[float]:
        """Advance by one control step. Returns commanded wz (rad/s), or
        None if the bounded rescan has timed out (caller should exit rescan
        mode and fall back to the default/cached goal, same as the existing
        scan-timeout fallback)."""
        if self._local_step >= _RESCAN_TIMEOUT_STEPS:
            return None
        self._local_step += 1
        return self._sched.step(current_yaw_rad)
