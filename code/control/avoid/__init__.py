"""
code.control.avoid — NX-9 local obstacle avoidance (docs/nx9_avoid.md).

RF-1 package split of the old flat code/avoid.py (695 lines) into:
  - core.py      : constants, carve-out helpers, compute_obstacle_bias,
                   biased_vel_cmd — the "core" bias-computation logic.
  - geometry.py  : backproject_frame — the "helpers" back-projection math
                   (mirrors code/grounding.py's cam_to_egocentric).
  - _selftest.py : the synthetic-depth-frame smoke test that used to live
                   under `if __name__ == "__main__":` in code/avoid.py
                   (still runnable via `python code/avoid.py`; also covered,
                   case by case, by tests/control/test_avoid.py).

This __init__ re-exports the full flat public surface of the old module so
`from code.avoid import <name>` / `code.avoid.<name>` (via the code/avoid.py
old-path alias) keep working unchanged for every caller (code/inferencer.py,
code/eval_search.py, code/fancy_demo.py) — see docs/refactor_plan.md.
"""

from code.control.avoid.core import (
    AVOID,
    AVOID_CORRIDOR_HALF_DEG,
    AVOID_DEADBAND,
    AVOID_DECAY_FACTOR,
    AVOID_EMA_ALPHA,
    AVOID_FLOOR_MARGIN_M,
    AVOID_MAX_WZ_BIAS,
    AVOID_MIN_DEPTH_FOR_WEIGHT_M,
    AVOID_MIN_GOAL_DIST_M,
    AVOID_MIN_VALID_DEPTH_M,
    AVOID_N_BEARING_BINS,
    AVOID_NEAR_M,
    AVOID_STALE_MAX_MISSED_CYCLES,
    AVOID_TARGET_EXEMPT_DIST_M,
    AVOID_TARGET_EXEMPT_MAX_DEG,
    AVOID_TARGET_EXEMPT_MIN_DEG,
    AVOID_TARGET_EXEMPT_SIZE_M,
    AVOID_TIE_BREAK_EPS,
    AVOID_TIE_BREAK_IMBALANCE,
    biased_vel_cmd,
    compute_obstacle_bias,
    decay_bias,
    is_maneuver_scene,
)
from code.control.avoid.geometry import backproject_frame
from code.control.avoid.geometry import backproject_frame as _backproject_frame

__all__ = [
    "AVOID",
    "AVOID_CORRIDOR_HALF_DEG",
    "AVOID_DEADBAND",
    "AVOID_DECAY_FACTOR",
    "AVOID_EMA_ALPHA",
    "AVOID_FLOOR_MARGIN_M",
    "AVOID_MAX_WZ_BIAS",
    "AVOID_MIN_DEPTH_FOR_WEIGHT_M",
    "AVOID_MIN_GOAL_DIST_M",
    "AVOID_MIN_VALID_DEPTH_M",
    "AVOID_N_BEARING_BINS",
    "AVOID_NEAR_M",
    "AVOID_STALE_MAX_MISSED_CYCLES",
    "AVOID_TARGET_EXEMPT_DIST_M",
    "AVOID_TARGET_EXEMPT_MAX_DEG",
    "AVOID_TARGET_EXEMPT_MIN_DEG",
    "AVOID_TARGET_EXEMPT_SIZE_M",
    "AVOID_TIE_BREAK_EPS",
    "AVOID_TIE_BREAK_IMBALANCE",
    "backproject_frame",
    "biased_vel_cmd",
    "compute_obstacle_bias",
    "decay_bias",
    "is_maneuver_scene",
]
