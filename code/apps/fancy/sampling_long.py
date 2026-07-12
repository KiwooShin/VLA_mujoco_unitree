"""FD2 long-distance (4-7m) + multi-goal scene samplers for the fancy demo
(code/fancy_demo.py, RF-1 split). Both build on
code/apps/fancy/sampling.py's VF-5 placement helpers.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from code.apps.fancy.constants import ARENA_HALF_LONG, DIST_MAX_LONG, DIST_MIN_LONG, MAXSTEPS_FANCY, RELIABLE_COLORS
from code.apps.fancy.sampling import FANCY_MIN_OBJECTS, _place_fancy_object_xy, _select_fancy_distractor_combos


# ---------------------------------------------------------------------------
# FD2: Long-distance scene sampler (4-7 m, reliable colors, large arena)
# VF-5: >=7 objects (target + >=6 distractors), see the placement-rules
# comment block in code/apps/fancy/sampling.py.
# ---------------------------------------------------------------------------

def sample_fancy_scene_long(rng: np.random.Generator, ep_idx: int,
                             dist_min: float = DIST_MIN_LONG,
                             dist_max: float = DIST_MAX_LONG) -> dict:
    """Sample a long-distance search scene:
    - Target OUTSIDE initial FOV (bearing > 45° from robot_yaw=0)
    - RELIABLE color for the target: red, orange, yellow, purple
      (grounding_dist.md: 78% success at 4-9m for non-cyan/blue)
    - Distance 4–7m (impressive walk; arena_size=8m to fit)
    - VF-5: >=7 objects total (target + >=6 distractors, mixed shapes/colors
      from the full palette), non-overlapping, >=1.2m apart, >=0.8m from
      walls, >=1.0m from the robot spawn -- see the placement-rules comment
      block above sample_fancy_scene_long. At least one same-color/
      different-shape pair is guaranteed (nice demo content).
    - Robot near origin (robot_xy ≈ 0)

    FD2: biases toward MEDIUM-LONG distances to make the reel impressive.

    Args:
        rng: NumPy random generator used for all sampling.
        ep_idx: Episode index (unused directly here; kept for interface
            parity with the other sample_fancy_* functions).
        dist_min: Minimum target distance in meters.
        dist_max: Maximum target distance in meters.

    Returns:
        scene_cfg dict compatible with run_fancy_rollout.
    """
    from code.arena import COLORS, SHAPES
    from code.eval_search import SEARCH_FOV_HALF_DEG

    arena_half   = ARENA_HALF_LONG  # 8m — room for 7m targets
    fov_half_rad = math.radians(SEARCH_FOV_HALF_DEG)

    # Robot slightly off-center
    rx = float(rng.uniform(-0.5, 0.5))
    ry = float(rng.uniform(-0.5, 0.5))
    robot_yaw = 0.0

    # Only reliable colors for the target (red, orange, yellow, purple)
    reliable_idxs = [i for i, (cname, _) in enumerate(COLORS)
                     if cname in RELIABLE_COLORS]
    shape_idxs = list(range(len(SHAPES)))

    tgt_ci = int(rng.choice(reliable_idxs))
    tgt_si = int(rng.choice(shape_idxs))

    # VF-5: >=6 distractors (any color/shape from the full palette), one of
    # which is guaranteed to share the target's color with a different shape.
    n_distractors = FANCY_MIN_OBJECTS - 1
    distractor_combos = _select_fancy_distractor_combos(
        rng, [(tgt_ci, tgt_si)], n_distractors, len(COLORS), len(SHAPES))
    chosen_combos = [(tgt_ci, tgt_si)] + distractor_combos

    objects: List[dict] = []
    for local_i, (ci, si) in enumerate(chosen_combos):
        color_name, color_rgb = COLORS[ci]
        shape_name, size      = SHAPES[si]
        size_val = float(size)
        is_target = (local_i == 0)

        if is_target:
            # Long distance: 4–7m, out-of-FOV
            ox, oy = _place_fancy_object_xy(
                rng, rx, ry, robot_yaw, arena_half, size_val, objects,
                dist_bounds=(dist_min, dist_max),
                out_of_fov=True, fov_half_rad=fov_half_rad)
        else:
            # Distractors: 2–5m from robot, anywhere
            ox, oy = _place_fancy_object_xy(
                rng, rx, ry, robot_yaw, arena_half, size_val, objects,
                dist_bounds=(2.0, 5.0))

        objects.append({
            "color_name": color_name,
            "color_rgb":  color_rgb,
            "shape_name": shape_name,
            "size":       size_val,
            "x":          float(ox),
            "y":          float(oy),
            "dist_from_robot": float(math.hypot(ox - rx, oy - ry)),
        })

    tgt = objects[0]
    dx, dy = tgt["x"] - rx, tgt["y"] - ry
    init_bearing_deg = abs(math.degrees(
        math.atan2(math.sin(math.atan2(dy, dx) - robot_yaw),
                   math.cos(math.atan2(dy, dx) - robot_yaw))
    ))

    return {
        "arena_size":       arena_half,
        "robot_xy":         (rx, ry),
        "robot_yaw":        robot_yaw,
        "objects":          objects,
        "target_index":     0,
        "stop_r":           0.5,
        "horizon":          MAXSTEPS_FANCY,
        "lighting":         {"ambient": 0.45},
        "difficulty":       "search_long",
        "init_bearing_deg": init_bearing_deg,
    }


def sample_fancy_multi_goal_scene(rng: np.random.Generator, n_goals: int = 2) -> dict:
    """Sample a scene with n_goals sub-goal objects at varied distances
    (2–6.5m), each a distinct reliable color+shape, PLUS (VF-5) >=5
    additional distractor objects (mixed shapes/colors from the full
    palette) so every multi-goal scene has >=7 objects total. Robot at
    origin, yaw=0.

    All objects (goals AND distractors) honor the VF-5 placement rules
    (>=1.2m apart, >=0.8m from walls, >=1.0m from the robot spawn -- see the
    comment block above sample_fancy_scene_long); at least one same-color/
    different-shape pair is guaranteed.

    Args:
        rng: NumPy random generator used for all sampling.
        n_goals: Number of distinct (color, shape) sub-goal objects to place.

    Returns:
        scene_cfg dict; target_index=0 (first object is 1st sub-goal).
        Multi-goal rollout iterates target_index across 0..n_goals-1.
    """
    from code.arena import COLORS, SHAPES
    from code.eval_search import SEARCH_FOV_HALF_DEG

    arena_half   = ARENA_HALF_LONG
    fov_half_rad = math.radians(SEARCH_FOV_HALF_DEG)

    rx, ry    = 0.0, 0.0
    robot_yaw = 0.0

    reliable_idxs = [i for i, (cname, _) in enumerate(COLORS)
                     if cname in RELIABLE_COLORS]
    shape_idxs = list(range(len(SHAPES)))

    # Pick n_goals distinct (color, shape) combos from reliable colors
    chosen = []
    all_reliable = [(ci, si) for ci in reliable_idxs for si in shape_idxs]
    rng.shuffle(all_reliable)
    for combo in all_reliable:
        if len(chosen) >= n_goals:
            break
        if combo not in chosen:
            chosen.append(combo)
    # Fill with extras if needed
    while len(chosen) < n_goals:
        chosen.append(all_reliable[len(chosen) % len(all_reliable)])

    # VF-5: >=5 additional distractors beyond the n_goals sub-goals (any
    # color/shape from the full palette), one guaranteed to share the first
    # sub-goal's color with a different shape.
    n_others = 5
    other_combos = _select_fancy_distractor_combos(
        rng, chosen, n_others, len(COLORS), len(SHAPES))
    all_combos = chosen + other_combos   # first n_goals entries ARE the sub-goals

    objects: List[dict] = []
    for local_i, (ci, si) in enumerate(all_combos):
        color_name, color_rgb = COLORS[ci]
        shape_name, size      = SHAPES[si]
        size_val = float(size)
        is_first_goal = (local_i == 0)
        is_goal = local_i < n_goals

        # Vary distance: first sub-goal further, later sub-goals medium,
        # extra distractors spread across the same medium band.
        if is_first_goal:
            d_lo, d_hi = 4.5, 6.5   # first goal: long
        else:
            d_lo, d_hi = 2.5, 5.0   # subsequent sub-goals + distractors: medium

        ox, oy = _place_fancy_object_xy(
            rng, rx, ry, robot_yaw, arena_half, size_val, objects,
            dist_bounds=(d_lo, d_hi),
            out_of_fov=is_first_goal, fov_half_rad=fov_half_rad)

        objects.append({
            "color_name": color_name,
            "color_rgb":  color_rgb,
            "shape_name": shape_name,
            "size":       size_val,
            "x":          float(ox),
            "y":          float(oy),
            "dist_from_robot": float(math.hypot(ox - rx, oy - ry)),
            "is_goal":    is_goal,
        })

    tgt = objects[0]
    dx, dy = tgt["x"] - rx, tgt["y"] - ry
    init_bearing_deg = abs(math.degrees(
        math.atan2(math.sin(math.atan2(dy, dx) - robot_yaw),
                   math.cos(math.atan2(dy, dx) - robot_yaw))
    ))

    return {
        "arena_size":       arena_half,
        "robot_xy":         (rx, ry),
        "robot_yaw":        robot_yaw,
        "objects":          objects,
        "target_index":     0,   # multi-goal: executor overrides this per sub-goal
        "stop_r":           0.5,
        "horizon":          MAXSTEPS_FANCY,
        "lighting":         {"ambient": 0.45},
        "difficulty":       "multi_goal",
        "init_bearing_deg": init_bearing_deg,
        "n_goals":          n_goals,
    }
