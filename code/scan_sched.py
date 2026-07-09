"""
scan_sched.py — NX-1 bidirectional bounded-rotation scan schedule.

Context (docs/fa1_failures.md, docs/rot_dart.md): the search skill's CCW-only
continuous scan (`eval_search.py`'s `_run_search_rollout`, duplicated in
`fancy_demo.py`'s `run_fancy_rollout`) is the #1-ranked root cause of all 3
search falls (ep5/7/8 in `eval/p4_gate_search_rerun`): a "wrong-side" target
(one that the fixed-CCW direction reaches the long way around) forces
550-600 continuous steps (~378-413 degrees) of uninterrupted in-place
rotation before the target enters the aligned cone -- well past the
~470-step (~323 degree) in-distribution bound the shared policy tolerates
(the next-longest *succeeding* scan is 470 steps). `docs/rot_dart.md`
independently confirmed prolonged continuous rotation is OOD for this
policy and that RETRAINING to fix it regresses other skills -- so this is a
deploy-side-only scan-schedule fix, not a model change.

Fix: replace the fixed single-direction sweep with a bounded, direction-
alternating "triangle wave" in yaw around the scan-start heading:

    0 -> +LEG_DEG -> 0 -> -LEG_DEG -> 0 -> +LEG_DEG -> ...

Every leg is capped at LEG_DEG degrees of *continuous* rotation (tracked via
actual accumulated yaw, not assumed step counts, so it self-corrects if the
realized rotation rate ever drifts from the commanded rate) and is followed
by a brief stand-still DWELL (wz=0, in-distribution behaviour) before the
next leg begins -- so no uninterrupted rotation segment ever approaches the
diagnosed OOD range, regardless of how many legs a full scan needs.

Why LEG_DEG=165 (revised from an initial 150 -- see docs/nx1_scan.md "attempt
1" for the failed first cut): a CCW leg of amplitude A only ever brings a
*left-side* (positive-bearing) target within the acquisition cone once BOTH
(a) it is within SCAN_ALIGNED_THR_DEG=40 degrees AND (b) it is within the
GROUNDING camera's own visibility half-angle (not just the aligned
threshold) -- classical grounding can't detect a target the camera can't
see at all. The grounding render is FOVY=45 (vertical) at 480x360, i.e.
~28.9-degree HORIZONTAL half-FOV (2*atan(tan(22.5)*480/360)/2) -- narrower
than the 40-degree aligned check, so *visibility*, not alignment, is the
binding constraint for a leg amplitude A: need A >= theta_t - 28.9, worst
sampled bearing 180 -> A >= 151.1. LEG_DEG=150 (the first attempt) left
essentially zero margin against this -- combined with GROUNDING_PERIOD=10
grounding cadence (the ~10-15-degree catchment window near the leg
boundary is only ~15-22 steps wide, a couple of grounding cycles at most)
it produced 2 genuine coverage misses (bearing 179.8 and 167.6, both
SCAN_TIMEOUT with zero detections, in the "attempt 1" full re-gate; see
docs/nx1_scan.md). LEG_DEG=165 gives a ~14-degree margin -- still far
inside the ~470-step / ~323-degree in-distribution ceiling (165 degrees
~= 240 steps at SCAN_RATE=0.6 rad/s). The full first coverage pass
(CCW 0->+165, dwell, CW +165->0, dwell, CW 0->-165 -- see docstring on
BidirectionalScanSchedule for why the return sweep is itself split into
two 165-degree legs) takes ~3*240 + 2*DWELL nominal steps; SCAN_TIMEOUT
and MAXSTEPS_SEARCH in the callers are sized with margin above that
(including margin for real-world realized-rotation lag relative to the
nominal SCAN_RATE, observed empirically at up to ~1.2-1.3x nominal steps
in the "attempt 1" re-gate).

See docs/nx1_scan.md for the full design writeup, both gate-eval attempts,
and the final per-episode results.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Shared constants (importable by eval_search.py / fancy_demo.py / demo.py)
# ---------------------------------------------------------------------------
SCAN_LEG_DEG      = 165.0   # max continuous same-direction rotation per leg (deg)
SCAN_DWELL_STEPS  = 45      # stand-still steps between legs (in-distribution dwell;
                             # upper end of the suggested 30-50 range, for extra
                             # re-stabilization margin at the scan-exit handoff)
SCAN_TIMEOUT      = 1150    # hard step cap (safety net; nominal full coverage pass
                             # completes in ~810 steps at SCAN_RATE=0.6 -- see above --
                             # with margin for observed real-world rotation lag)

# Leg direction pattern, repeating every 4 legs: CCW, CW, CW, CCW, CCW, CW, CW, ...
# leg 0: 0 -> +LEG_DEG        (CCW)
# leg 1: +LEG_DEG -> 0        (CW)   \_ together, a full CW sweep from
# leg 2: 0 -> -LEG_DEG        (CW)   /  +LEG_DEG down to -LEG_DEG, split into
#                                       two capped legs by the leg1/leg2 dwell
# leg 3: -LEG_DEG -> 0        (CCW)  (repeat, extra passes if not yet found)
_LEG_SIGNS = (+1, -1, -1, +1)


class BidirectionalScanSchedule:
    """
    Stateful bounded-rotation scan schedule. Call `.step(current_yaw_rad)`
    once per scan step (with the robot's current world yaw in radians) to
    get the wz (rad/s) to command this step.

    Tracks *actual* accumulated yaw (unwrapped, integrated from consecutive
    yaw readings) rather than assuming steps * nominal_rate * dt, per the
    diagnosis that this should bound real rotation, not just elapsed time --
    self-corrects if the policy doesn't perfectly track the commanded wz.
    """

    def __init__(self, scan_rate: float = 0.6, leg_deg: float = SCAN_LEG_DEG,
                 dwell_steps: int = SCAN_DWELL_STEPS):
        self.scan_rate   = scan_rate
        self.leg_deg     = leg_deg
        self.dwell_steps = dwell_steps

        self._leg_idx        = 0
        self._leg_yaw_start  = 0.0   # accumulated-yaw value at start of current leg
        self._accum_yaw_deg  = 0.0   # accumulated yaw (deg) since first call, signed
        self._dwell_counter  = 0
        self._dwelling       = False
        self._prev_yaw       = None  # last raw yaw (rad) seen, for delta integration

    def step(self, current_yaw_rad: float) -> float:
        """Advance the schedule by one step; return commanded wz (rad/s)."""
        if self._prev_yaw is not None:
            dyaw = current_yaw_rad - self._prev_yaw
            dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))  # wrap to (-pi, pi]
            self._accum_yaw_deg += math.degrees(dyaw)
        self._prev_yaw = current_yaw_rad

        if self._dwelling:
            self._dwell_counter += 1
            if self._dwell_counter >= self.dwell_steps:
                self._dwelling      = False
                self._dwell_counter = 0
                self._leg_idx      += 1
                self._leg_yaw_start = self._accum_yaw_deg
            return 0.0

        sign     = _LEG_SIGNS[self._leg_idx % 4]
        traveled = (self._accum_yaw_deg - self._leg_yaw_start) * sign
        if traveled >= self.leg_deg:
            self._dwelling      = True
            self._dwell_counter = 0
            return 0.0
        return sign * self.scan_rate

    @property
    def leg_idx(self) -> int:
        return self._leg_idx

    @property
    def is_dwelling(self) -> bool:
        return self._dwelling
