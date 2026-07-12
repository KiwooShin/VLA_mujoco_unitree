"""Short-range search scene sampler + VF-5 object-placement helpers for the
fancy demo (code/fancy_demo.py, RF-1 split).

Owns FIRST_SCENE_SEED (the curated deterministic seed used for the very
first --web/terminal scene draw) since it's fundamentally a scene-sampling
curation concern, even though it's consumed by code/apps/fancy/live.py's
FancySceneManager.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from code.apps.fancy.constants import MAXSTEPS_FANCY, RELIABLE_COLORS


# FS-1 (2026-07-10): curated deterministic FIRST scene for --web/terminal launch.
# The old always-[1234,0] first draw reproducibly landed on `red cone,
# dist=4.35m, bearing=77.6°` -- the documented walking-instability residual
# (robot SPOTTED the target at step 20 then walked steadily *away*,
# dist 4.24m->9.4m; docs/vr1_rehearsal.md friction #3), i.e. a recruiter's
# very first impression was a known-bad draw. Subsequent new_scene() calls
# were never the problem (they auto-resample after every rollout), so only
# this one fixed draw needed curating.
#
# FS-2 (2026-07-11, docs/vf5_cam_objects.md's flag / docs/fs2_first_scene_
# resample.md): VF-5 rewrote sample_fancy_scene_long's placement loop to
# guarantee >=7 objects per scene (was 3). Same FIRST_SCENE_SEED=1259 draw
# under the NEW sampler produces a DIFFERENT 7-object scene (yellow cube,
# dist=4.95m, bearing=54.6° -- still fine geometrically, but never
# re-verified end-to-end against the new sampler/camera), so the seed was
# re-picked from scratch under the actual current sampler rather than
# trusting the old pick to still apply.
# Picked by: geometry pre-filter (color in RELIABLE_COLORS, dist 4-7m,
# bearing in [60,110] AND positive-signed -- matches BidirectionalScan-
# Schedule's _LEG_SIGNS=(+1,-1,-1,+1) "positive leg0 first" so no leg0->leg1
# reversal is needed, sidestepping the rotation-order-instability class
# documented in docs/gen1_multiseed.md §3.1 / docs/nx12_turn_dwell.md; no
# same-color/SAME-shape distractor -- docs/gen1_multiseed.md §3.3's
# false-lock risk (now also guaranteed by construction for every VF-5
# scene, via _select_fancy_distractor_combos' pairwise-distinct combos);
# no distractor within 0.5m of the straight robot->target path; target
# shape != cone -- docs/nx16_cone_stall.md's cone-specific confidence-decay
# risk), then verified by actually running the full rollout headlessly 2x:
#   seed=3461 -> target=yellow cube, dist=4.97m, bearing=84.9° (out-of-FOV),
#   7 objects incl. a same-color/diff-shape yellow ball + yellow cylinder,
#   nearest distractor 2.15m off the straight path.
#   Both runs: success=True, fell=False, steps=714/711, final_dist=0.468m/
#   0.468m, wall~308-309s each (small step-count delta is the documented
#   EGL/physics jitter, docs/gen1_multiseed.md -- not a concern). See
#   docs/vf5_cam_objects.md (first-scene re-curation notes).
FIRST_SCENE_SEED = 3461


# ---------------------------------------------------------------------------
# Reliable-color search scene sampler
# ---------------------------------------------------------------------------

def sample_fancy_scene(rng: np.random.Generator, ep_idx: int) -> dict:
    """Sample a search scene with:
    - Target OUTSIDE initial FOV (bearing > 45° from robot_yaw=0)
    - RELIABLE colors biased: orange, red, yellow, purple (avoid cyan/blue HSV wall collision)
    - Distance 2-4m (easy enough to reach but shows search phase)
    - Multiple objects placed non-overlapping

    Args:
        rng: NumPy random generator used for all sampling.
        ep_idx: Episode index (unused directly here; kept for interface
            parity with the other sample_fancy_* functions).

    Returns:
        scene_cfg dict compatible with run_fancy_rollout.
    """
    from code.arena import COLORS, SHAPES
    from code.eval_search import SEARCH_FOV_HALF_DEG, SEARCH_DIST_MIN, SEARCH_DIST_MAX

    arena_half = 4.0
    margin     = 0.55

    rx = float(rng.uniform(-0.3, 0.3))
    ry = float(rng.uniform(-0.3, 0.3))
    robot_yaw  = 0.0
    fov_half_rad = math.radians(SEARCH_FOV_HALF_DEG)

    # Build color lookup
    color_name_to_rgb = {name: rgb for name, rgb in COLORS}

    # Bias target toward reliable colors
    reliable_idxs = [i for i, (cname, _) in enumerate(COLORS)
                     if cname in RELIABLE_COLORS]
    shape_idxs    = list(range(len(SHAPES)))

    # Choose 3 (color, shape) combos; first is target, rest are distractors
    # Target: reliable color
    tgt_ci = int(rng.choice(reliable_idxs))
    tgt_si = int(rng.choice(shape_idxs))

    # Distractors: any color/shape but avoid same as target
    combos = [(ci, si) for ci in range(len(COLORS)) for si in range(len(SHAPES))
              if (ci, si) != (tgt_ci, tgt_si)]
    d_indices = rng.choice(len(combos), size=2, replace=False)
    chosen_combos = [(tgt_ci, tgt_si)] + [combos[k] for k in d_indices]

    objects = []
    for local_i, (ci, si) in enumerate(chosen_combos):
        color_name, color_rgb = COLORS[ci]
        shape_name, size      = SHAPES[si]
        size_val = float(size)
        is_target = (local_i == 0)

        placed = False
        for _ in range(5000):
            if is_target:
                d     = float(rng.uniform(2.0, 4.0))
                side  = rng.integers(2)
                if side == 0:
                    angle = float(rng.uniform(robot_yaw + fov_half_rad, robot_yaw + math.pi))
                else:
                    angle = float(rng.uniform(robot_yaw - math.pi, robot_yaw - fov_half_rad))
            else:
                d     = float(rng.uniform(1.0, 3.5))
                angle = float(rng.uniform(-math.pi, math.pi))

            ox = rx + d * math.cos(angle)
            oy = ry + d * math.sin(angle)

            if abs(ox) + size_val / 2 + margin >= arena_half:
                continue
            if abs(oy) + size_val / 2 + margin >= arena_half:
                continue
            if any(math.hypot(ox - o["x"], oy - o["y"]) < 0.8 for o in objects):
                continue

            if is_target:
                dx, dy   = ox - rx, oy - ry
                obj_angle = math.atan2(dy, dx)
                err = math.atan2(math.sin(obj_angle - robot_yaw), math.cos(obj_angle - robot_yaw))
                if abs(err) <= fov_half_rad:
                    continue

            objects.append({
                "color_name": color_name,
                "color_rgb":  color_rgb,
                "shape_name": shape_name,
                "size":       size_val,
                "x":          float(ox),
                "y":          float(oy),
                "dist_from_robot": float(math.hypot(ox - rx, oy - ry)),
            })
            placed = True
            break

        if not placed:
            # Fallback
            for _ in range(10000):
                if is_target:
                    d = float(rng.uniform(2.0, 4.0))
                    side = rng.integers(2)
                    angle = (float(rng.uniform(robot_yaw + fov_half_rad, robot_yaw + math.pi))
                             if side == 0
                             else float(rng.uniform(robot_yaw - math.pi, robot_yaw - fov_half_rad)))
                    ox = rx + d * math.cos(angle)
                    oy = ry + d * math.sin(angle)
                else:
                    ox = float(rng.uniform(-(arena_half - margin), arena_half - margin))
                    oy = float(rng.uniform(-(arena_half - margin), arena_half - margin))

                if abs(ox) + 0.5 + margin < arena_half and abs(oy) + 0.5 + margin < arena_half:
                    if not any(math.hypot(ox - o["x"], oy - o["y"]) < 0.5 for o in objects):
                        objects.append({
                            "color_name": color_name,
                            "color_rgb":  color_rgb,
                            "shape_name": shape_name,
                            "size":       size_val,
                            "x":          float(ox),
                            "y":          float(oy),
                            "dist_from_robot": float(math.hypot(ox - rx, oy - ry)),
                        })
                        break

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
        "difficulty":       "search",
        "init_bearing_deg": init_bearing_deg,
    }


# ---------------------------------------------------------------------------
# VF-5 (docs/vf5_cam_objects.md, user feedback #2 "add more objects in the
# fields... at least 7 objects"): the fancy-demo scenes (single-goal AND
# multi-goal) now place >=7 objects total instead of the old 3 -- richer,
# more populated demo scenes. This section ONLY affects the fancy-demo
# samplers below (sample_fancy_scene_long, sample_fancy_multi_goal_scene) --
# code/scene.py (the FROZEN eval-benchmark sampler used by eval_closedloop/
# eval_search/eval_maneuver) is untouched.
#
# Placement rules applied to EVERY object (target/goals AND distractors):
#   - >=FANCY_OBJ_MIN_SEP_M between any two object centers
#   - >=FANCY_OBJ_WALL_MARGIN_M clearance from an object's edge to the wall
#   - >=FANCY_OBJ_MIN_ROBOT_M from the robot spawn point
#   - the target/each goal keeps its EXISTING distance-band + out-of-FOV logic
#     (unchanged -- this VF-5 pass only touches distractor count/placement and
#     tightens the shared wall/separation margins slightly)
# At least one same-color/different-shape pair is guaranteed in every sampled
# scene (nice demo content -- shows the detector discriminates by shape, not
# just color; grounding always queries a specific color+shape combo, so a
# same-color/different-shape distractor does not by itself create the
# same-color/SAME-shape "false lock" risk documented in
# docs/gen1_multiseed.md §3.3 -- that risk requires an exact combo match,
# which the guarantee below explicitly avoids by construction).
# Determinism is preserved: both helpers below are pure functions of the
# `rng` passed in (same np.random.Generator state -> same output), matching
# every other sampler in this file.
FANCY_MIN_OBJECTS       = 7      # target/goal(s) + distractors, total >= this
FANCY_OBJ_MIN_SEP_M     = 1.2    # min center-to-center distance between any two objects
FANCY_OBJ_WALL_MARGIN_M = 0.8    # min clearance from an object's edge to the wall
FANCY_OBJ_MIN_ROBOT_M   = 1.0    # min distance from any object to the robot spawn point


def _place_fancy_object_xy(
    rng: np.random.Generator,
    rx: float, ry: float, robot_yaw: float,
    arena_half: float,
    size_val: float,
    existing: List[dict],
    dist_bounds: Tuple[float, float],
    out_of_fov: bool = False,
    fov_half_rad: float = 0.0,
    min_sep: float = FANCY_OBJ_MIN_SEP_M,
    wall_margin: float = FANCY_OBJ_WALL_MARGIN_M,
    min_robot_dist: float = FANCY_OBJ_MIN_ROBOT_M,
    n_tries: int = 12000,
) -> Tuple[float, float]:
    """Sample one object's (x, y) honoring the VF-5 wall/separation/robot-
    distance rules, within `dist_bounds` of the robot spawn and (optionally)
    outside the robot's initial FOV.

    Three progressively-relaxed passes (mirrors the pre-existing fallback
    pattern in this file's samplers) guarantee placement always succeeds for
    the object counts/arena sizes this demo uses -- the strict rule is tried
    first (`n_tries` draws), then `min_sep`/`wall_margin` are loosened, so a
    late-placed object in a crowded scene never blocks the whole sampler.

    Args:
        rng: NumPy random generator used for all sampling.
        rx, ry: Robot spawn position.
        robot_yaw: Robot spawn yaw, in radians (used only when out_of_fov).
        arena_half: Arena half-size in meters.
        size_val: This object's placement size (diameter-ish), used for wall
            clearance.
        existing: Already-placed object dicts (with 'x'/'y' keys) to avoid.
        dist_bounds: (min, max) distance from the robot spawn point.
        out_of_fov: Whether this object must be outside the robot's initial
            FOV half-angle.
        fov_half_rad: FOV half-angle in radians (used only when out_of_fov).
        min_sep, wall_margin, min_robot_dist: VF-5 placement rules (see
            module-level constants above for the defaults).
        n_tries: Rejection-sampling attempts per relaxation stage.

    Returns:
        (x, y) world position for the new object.
    """
    d_lo, d_hi = dist_bounds
    ox, oy = rx, ry
    for relax in range(3):  # 0 = strict, 1 = relaxed separation, 2 = relaxed wall too
        cur_sep  = min_sep     if relax == 0 else (0.7 if relax == 1 else 0.5)
        cur_wall = wall_margin if relax < 2  else 0.5
        for _ in range(n_tries):
            d = float(rng.uniform(d_lo, d_hi))
            if out_of_fov:
                side = rng.integers(2)
                if side == 0:
                    angle = float(rng.uniform(robot_yaw + fov_half_rad, robot_yaw + math.pi))
                else:
                    angle = float(rng.uniform(robot_yaw - math.pi, robot_yaw - fov_half_rad))
            else:
                angle = float(rng.uniform(-math.pi, math.pi))
            ox = rx + d * math.cos(angle)
            oy = ry + d * math.sin(angle)

            if abs(ox) + size_val / 2 + cur_wall >= arena_half:
                continue
            if abs(oy) + size_val / 2 + cur_wall >= arena_half:
                continue
            if math.hypot(ox - rx, oy - ry) < min_robot_dist:
                continue
            if any(math.hypot(ox - o["x"], oy - o["y"]) < cur_sep for o in existing):
                continue
            return ox, oy
    # Should not happen for this demo's object counts/arena sizes, but never
    # crash the sampler -- fall through with the last candidate tried.
    return ox, oy


def _select_fancy_distractor_combos(
    rng: np.random.Generator,
    primary_combos: List[Tuple[int, int]],
    n_distractors: int,
    n_colors: int,
    n_shapes: int,
) -> List[Tuple[int, int]]:
    """Select `n_distractors` distinct (color_idx, shape_idx) combos, distinct
    from `primary_combos` and from each other, guaranteeing at least one
    selected combo shares its color with `primary_combos[0]` (the main
    target/first goal) while using a different shape -- VF-5's "at least one
    same-color/different-shape pair somewhere in the scene".

    Deterministic given `rng`'s state (same seed -> same combos).

    Args:
        rng: NumPy random generator used for all sampling.
        primary_combos: (color_idx, shape_idx) combos already used by the
            target/goal object(s), to be excluded from the distractor pool.
        n_distractors: Number of distractor combos to select.
        n_colors: Number of colors in the palette (len(COLORS)).
        n_shapes: Number of shapes in the palette (len(SHAPES)).

    Returns:
        List of `n_distractors` distinct (color_idx, shape_idx) combos.
    """
    used = set(primary_combos)
    all_combos = [(ci, si) for ci in range(n_colors) for si in range(n_shapes)]
    remaining = [c for c in all_combos if c not in used]

    chosen: List[Tuple[int, int]] = []
    if n_distractors > 0 and primary_combos:
        base_ci, base_si = primary_combos[0]
        partner_opts = [(ci, si) for (ci, si) in remaining if ci == base_ci and si != base_si]
        if partner_opts:
            k = int(rng.integers(len(partner_opts)))
            partner = partner_opts[k]
            chosen.append(partner)
            remaining.remove(partner)

    n_more = n_distractors - len(chosen)
    if n_more > 0:
        n_more = min(n_more, len(remaining))
        idxs = rng.choice(len(remaining), size=n_more, replace=False)
        chosen.extend(remaining[int(k)] for k in idxs)
    return chosen
