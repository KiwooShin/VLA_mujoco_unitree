"""
code.runtime.constants — module-level constants + env toggles for the
closed-loop deploy Inferencer (RF-1 split of code/inferencer.py; docs/refactor_plan.md).

RF-1 note: this is a pure "moving" split — every constant, comment, and env
toggle below is verbatim from the original flat code/inferencer.py (only the
file it lives in changed). code/runtime/inferencer.py imports all of these
by name so `code.inferencer.<NAME>` (old path, via the sys.modules alias)
keeps resolving exactly as before for every external caller
(code/eval_search.py-family, code/apps/fancy/rollout.py, code/verify_settle.py,
code/render_showcase_videos.py, code/bench_widefov_visibility.py, etc.).
"""

from __future__ import annotations

import os
from pathlib import Path

from code.sim.teacher import DEFAULT_ANGLES
from code.control import avoid as _avoid

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FALL_HEIGHT        = 0.50     # pelvis z below this → fall (matches gen_dataset.py)
GROUNDING_PERIOD   = 10       # run grounding every N steps → 50Hz/10 = 5 Hz
KEYFRAME_PATH      = str(_REPO / "checkpoint" / "stand_keyframe.npz")
PROPRIO_K          = 6        # proprio history length
PROPRIO_DIM        = 55
PROPRIO_DIM_PHASE  = 57       # proprio_dim when gait phase [sin,cos] appended (Fix 4)
IMG_SIZE           = 128
HOLD_STEPS_REQUIRED = 5       # consecutive in-range steps to succeed
ACTION_SCALE       = 0.25     # from teacher.py: target_dof = action * 0.25 + default_angles
SETTLE_STEPS       = 80       # warm-up steps using teacher (zero cmd)


def _env_flag(name: str, default: str = "0") -> bool:
    """Returns True iff environment variable `name` is set to the string "1"."""
    return os.environ.get(name, default).strip() == "1"


# ---------------------------------------------------------------------------
# NX-8 STALL_BREAK (docs/nx8_stall.md): steering-level physical-stall watchdog.
# ---------------------------------------------------------------------------
# Root cause this targets (docs/nx7_adoption.md §2): under GROUND_NET=1, demo
# ep1's robot goes PHYSICALLY STATIC (world-frame (x,y) pinned to a ~0.1m box,
# `qpos` instrumentation) around step ~250-300 -- ~1000 steps before any
# hold-goal-horizon-keyed recovery mechanism (M4/M5/M7, all gated on
# accumulated odometry or elapsed detection-loss time) is even ELIGIBLE to
# fire, because those mechanisms all require either motion (M7's accumulator)
# or a long elapsed coasting window (M5) to trigger -- neither is available
# when the robot isn't moving in the first place. NX-7 confirmed this is a
# locomotion/policy-level phenomenon (the classical grounder's detections are
# STILL accurate for the ~200 steps right up to the freeze), consistent with
# this policy's known BC-history stall/static-collapse attractor
# (`docs/CAMPAIGN.md` T2/E3 era: "mean-regression on absolute joint targets"
# gait collapse, later fixed for the general case via residual actions + DART
# + gait-phase input, but evidently not fully retired for every out-of-
# distribution goal-signal combination).
#
# Detects "commanding forward motion but the robot isn't actually moving"
# directly at the steering level (independent of WHICH grounding backend is
# in use -- classical or GROUND_NET -- since this is a locomotion phenomenon,
# not a grounding one) and injects a bounded stop -> resume recovery: force
# `gt_vel=0` (the "stand" command -- in-distribution, matches the
# stand-keyframe episode-start init and the scan-dwell segments the policy
# was trained on) for STALL_RECOVERY_STEPS, then resume the normal
# goal-directed velocity flow ("episode-start-like" = in-distribution).
#
# Opt-in, default OFF. Carve-outs (checked at the trigger site below): never
# while `_scan_active` (covers the initial H3 scan, any M4/M5/M7-triggered
# ReacquisitionScan, AND its dwell segments -- the watchdog's own
# window-update/trigger-check code only executes on the guaranteed-non-scan
# "normal mode" path each step, so scan/rescan/dwell steps structurally never
# feed the window or fire the trigger, not just a value check); never below
# STALL_MIN_GOAL_DIST_M (final-approach creep intentionally involves low
# commanded vx / fine positioning, not a stall); never during a
# `difficulty == 'maneuver'` scene (maneuver scenes can legitimately involve
# sustained forward-vx-commanded-but-not-translating segments, e.g. pushing
# against/around an obstacle by design -- out of this mechanism's intended
# scope).
STALL_BREAK            = _env_flag("STALL_BREAK")   # opt-in, default OFF
STALL_VX_THR_MPS        = 0.2    # m/s -- commanded |v_fwd| considered "trying to walk"
STALL_WINDOW_STEPS      = 100    # steps of sustained high-vx-cmd + low displacement
STALL_DISP_THR_M        = 0.15   # metres -- odometric displacement over the window
STALL_MIN_GOAL_DIST_M   = 2.0    # metres -- never trigger during final-approach creep
STALL_RECOVERY_STEPS    = 50     # steps of forced full-stop (gt_vel=0) during recovery
STALL_COOLDOWN_STEPS    = 100    # steps after a recovery completes before re-arming
                                  # (give the robot a chance to actually move again
                                  # before the watchdog is eligible to re-trigger)

# ---------------------------------------------------------------------------
# NX-9 AVOID (docs/nx9_avoid.md): local obstacle avoidance.
# ---------------------------------------------------------------------------
# The system has no local obstacle awareness at all -- it walks straight lines
# at the goal (dist,bearing), which is exactly why demo ep1's collision with a
# scene distractor (docs/nx8_stall.md), demo ep4's compound failure
# (docs/fa1_failures.md), and search ep12's distractor-caused fall
# (docs/nx1_scan.md) are all path-obstacle collisions, not grounding or
# locomotion-stability problems. See code/avoid.py for the full mechanism
# (back-projected near-field depth -> floor/target-exempt corridor mask ->
# bounded, hysteresis-smoothed yaw-rate bias). DEFAULT ON since NX-9 adoption
# (docs/nx9_avoid.md §5; opt out with AVOID=0); wired only
# for `goal_source='classical'` (covers BOTH the classical HSV+depth backend
# AND GROUND_NET, since GROUND_NET is dispatched INSIDE `ground()` under the
# same call site -- see code/grounding.py) because that is the only backend
# the validation ladder (docs/nx9_avoid.md) exercises; zero extra renders
# (reuses the depth frame already rendered for grounding this cycle).
AVOID = _avoid.AVOID   # ADOPTED default ON (docs/nx9_avoid.md); opt out with AVOID=0

# Default joint angles (imported from teacher.py, also available in action_stats from ckpt)
_DEFAULT_ANGLES_NP = DEFAULT_ANGLES.copy()  # shape (15,)

# Gait phase constants (Fix 4)
_LEFT_ANKLE_PITCH_IDX = 4     # in lower-body joint positions (qpos[7:22])
_LEFT_ANKLE_DEFAULT   = -0.2  # default angle (rad)
