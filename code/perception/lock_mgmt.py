"""
lock_mgmt.py — NX-2 target-lock management (docs/rs1_lock_mgmt.md).

RF-1 note: this module is now a thin aggregator (docs/refactor_plan.md) —
the constants/toggles and the `LockGate` state machine live in
code/perception/lock_gate.py, and the bounded-rescan wrapper
`ReacquisitionScan` lives in code/perception/lock_rescan.py. This file
re-exports both under the original flat namespace so every existing caller
(code/inferencer.py, code/eval_search.py, code/perception/ground_net.py,
code/fancy_demo.py) keeps working unchanged, and is what the old top-level
`code/lock_mgmt.py` path aliases to (sys.modules pattern, docs/refactor_plan.md).

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
    LOCK_M7=1   enables the odometry-coherence watchdog (drop + rescan when walked
                displacement toward the goal bearing isn't matched by a commensurate
                goal-distance shrink) -- OFF by default (NX-5, docs/nx5_coherence.md,
                pending gate verdict)

With M1/M3 at their new ON defaults and M2/M4/M5/M7 left at their OFF defaults,
every public method below matches the fully-validated `eval/nx2_combined_*`
gate results (demo 10/15, easy 15/15, search 14/15 -- see docs/nx2_final.md).
Setting LOCK_M1=0 LOCK_M2=0 LOCK_M3=0 LOCK_M4=0 LOCK_M5=0 (LOCK_M7 is already
off by default and structurally independent of the other four) reproduces the
pre-NX-2 byte-identical pass-through behavior (see per-method docstrings)
documented in docs/nx2_impl.md. This is deliberate: it means the two call
sites only need ONE extra function call per decision point, not a maze of
conditionals, and both the "all-off legacy" and "shipped-defaults" properties
are enforced structurally rather than by convention.

Mandatory carve-outs (docs/rs1_lock_mgmt.md risk #2): M3's innovation gate,
M4's divergence watchdog, and M7's odometry-coherence watchdog must NOT fire
on the two legitimate (dist,bearing) discontinuities this codebase already
knows about -- the CAM-2 fallback probe-adopt event and an `_active_cam`
Schmitt-trigger flip (both only exist in inferencer.py; eval_search.py has no
second camera and never calls `mark_discontinuity()`). Callers signal these
via `LockGate.mark_discontinuity()` at the point the event happens; the
resulting cooldown window bypasses M3's gate for new detections, suppresses
M4's trigger, clears the M4 dist window so stale pre-event samples cannot
corrupt the post-event trend check, AND resets M7's accumulation window
(pre/post-handoff distances come from different camera geometries and are
not comparable).

Rescans triggered by M4/M5/M7 reuse NX-1's `BidirectionalScanSchedule`
(`code/scan_sched.py`) via `ReacquisitionScan` below -- never an unbounded
spin. This is a SEPARATE small wrapper from the callers' own initial-scan
mechanisms (inferencer.py's absolute-step H3 sweep, eval_search.py's own
`BidirectionalScanSchedule` instance) because both of those gate their
timeout off the EPISODE's absolute step counter; re-arming either mid-episode
would immediately "time out" since the absolute step is already well past
their timeout constant by the time a mid-episode rescan can trigger.
`ReacquisitionScan` tracks its own LOCAL step counter from the moment it is
constructed, so it is safe to instantiate fresh at any point in an episode.

M7 additionally applies a short-term (not hard-block) RE-LOCK PENALTY after
it fires: a fresh detection landing within `M7_PENALTY_BEARING_DEG` /
`M7_PENALTY_DIST_TOL_M` of the just-dropped lock's (bearing, dist) needs
`M7_PENALTY_CONFIRM_M`-of-`M7_PENALTY_CONFIRM_N` mutually-consistent hits
(instead of the usual single-frame confirm) for `M7_PENALTY_CYCLES` cycles --
this exists specifically so a dropped false lock can't instantly re-seed
itself on the very next frame (the mechanism that killed M4, docs/nx2_final.md
M4 section), while a real target sitting nearby can still relock within a
couple of frames rather than being hard-blocked.

See docs/nx2_impl.md for the full write-up, chosen constants, and the
empirical evidence behind the M1 floor value.
"""

from __future__ import annotations

from code.perception.lock_gate import (LOCK_M1, LOCK_M2, LOCK_M3, LOCK_M4, LOCK_M5, LOCK_M7,
                                       M1_AREA_FLOOR_PX2, M2_CONFIRM_M, M2_CONFIRM_N,
                                       M2_TOL_BEARING_DEG, M2_TOL_DIST_M,
                                       M3_EXPECTED_CLOSING_M_PER_CYCLE, M3_GATE_BEARING_DEG,
                                       M3_GATE_BEARING_NEAR_MULT, M3_GATE_DIST_CLOSING_MULT,
                                       M3_GATE_DIST_FLOOR_M, M3_INCUMBENT_K,
                                       M3_INCUMBENT_MARGIN, M3_NEAR_RANGE_M,
                                       M4_EXEMPT_CYCLES_AFTER_CONFIRM,
                                       M4_EXEMPT_CYCLES_AROUND_HANDOFF, M4_TREND_MARGIN_M,
                                       M4_WINDOW_N, M7_K_MIN_FRAC, M7_MIN_GOAL_DIST_M,
                                       M7_PENALTY_BEARING_DEG, M7_PENALTY_CONFIRM_M,
                                       M7_PENALTY_CONFIRM_N, M7_PENALTY_CYCLES,
                                       M7_PENALTY_DIST_TOL_M, M7_PENALTY_TOL_BEARING_DEG,
                                       M7_PENALTY_TOL_DIST_M, M7_X_WALK_M, LockGate,
                                       _ang_diff_rad, _env_flag)
from code.perception.lock_rescan import ReacquisitionScan

__all__ = [
    "LOCK_M1", "LOCK_M2", "LOCK_M3", "LOCK_M4", "LOCK_M5", "LOCK_M7",
    "M1_AREA_FLOOR_PX2", "M2_CONFIRM_M", "M2_CONFIRM_N", "M2_TOL_BEARING_DEG", "M2_TOL_DIST_M",
    "M3_EXPECTED_CLOSING_M_PER_CYCLE", "M3_GATE_BEARING_DEG", "M3_GATE_BEARING_NEAR_MULT",
    "M3_GATE_DIST_CLOSING_MULT", "M3_GATE_DIST_FLOOR_M", "M3_INCUMBENT_K", "M3_INCUMBENT_MARGIN",
    "M3_NEAR_RANGE_M", "M4_EXEMPT_CYCLES_AFTER_CONFIRM", "M4_EXEMPT_CYCLES_AROUND_HANDOFF",
    "M4_TREND_MARGIN_M", "M4_WINDOW_N", "M7_K_MIN_FRAC", "M7_MIN_GOAL_DIST_M",
    "M7_PENALTY_BEARING_DEG", "M7_PENALTY_CONFIRM_M", "M7_PENALTY_CONFIRM_N", "M7_PENALTY_CYCLES",
    "M7_PENALTY_DIST_TOL_M", "M7_PENALTY_TOL_BEARING_DEG", "M7_PENALTY_TOL_DIST_M", "M7_X_WALK_M",
    "LockGate", "ReacquisitionScan", "_ang_diff_rad", "_env_flag",
]
