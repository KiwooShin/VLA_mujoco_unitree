"""code.eval.search_types — search-skill scene sampler + result schema + constants.

Split out of the original ``eval_search.py`` (RF-1): pure-logic scene sampling
(``sample_search_scene``) and the per-episode result schema (``SearchResult``),
plus every module-level constant shared by ``code.eval.search_rollout`` (the
standalone rollout loop) and ``code.eval.search`` (aggregation/reporting + CLI).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_SEED       = 999
# NX-1: bumped from 1400 (the old "same as demo preset" value) -- the bidirectional
# bounded scan (code/scan_sched.py) caps every continuous rotation segment safely,
# but can spend more TOTAL steps finding an unfavorable-side target than the old
# fixed-CCW scan did in its common case (it now always visits up to ~3*SCAN_LEG_DEG
# of yaw before giving up, vs. sometimes finding a favorable-side target almost
# immediately). SCAN_TIMEOUT=1150 (scan_sched.py) alone left too little of the
# 1400 budget for the approach phase on several previously-passing episodes
# (e.g. ep0: spotted at step 890, only 510 left, final_dist=0.51 -- one hair
# outside STOP_R_SEARCH). 2000 gives ~850 steps of approach headroom even in the
# worst observed case. See docs/nx1_scan.md.
MAXSTEPS_SEARCH = 2000           # hard cap (was 1400 pre-NX-1)
STOP_R_SEARCH   = 0.5            # slightly lenient stop radius
N_RENDER        = 3              # max videos to render
GOTO_CKPT       = str(_REPO / "checkpoint" / "goto_best.pt")

# FOV constraint for search scenes: target MUST start outside this cone
SEARCH_FOV_HALF_DEG = 45.0       # target angle from robot heading must exceed this
SEARCH_DIST_MIN     = 2.0        # target distance (easy case, no obstacles)
SEARCH_DIST_MAX     = 4.5        # keep it reachable after scan

# Scan threshold from inferencer.py (target_bearing < this to exit scan mode)
SCAN_ALIGNED_THR_DEG = 40.0


# ---------------------------------------------------------------------------
# Out-of-FOV scene sampler
# ---------------------------------------------------------------------------

def sample_search_scene(rng: np.random.Generator, episode_idx: int) -> dict:
    """Sample a search scene where the target is OUTSIDE the initial FOV.

    Strategy:
      - Robot near centre, facing +X (yaw=0)
      - Target placed at bearing > SEARCH_FOV_HALF_DEG from robot heading
      - Easy case: no obstacles, small arena (4m half), target at 2-4.5m
      - 3 objects total (1 target + 2 distractors)

    The FOV constraint ensures scan is REQUIRED to find the target.

    Args:
        rng: NumPy random Generator for deterministic sampling.
        episode_idx: Episode index (unused by the sampling logic itself;
            kept for caller-side bookkeeping/logging).

    Returns:
        Scene config dict with arena_size, robot_xy, robot_yaw, objects,
        target_index, instruction, stop_r, horizon, lighting, difficulty,
        and init_bearing_deg.
    """
    from code.arena import COLORS, SHAPES
    from code.scene import _make_instruction

    arena_half = 4.0
    margin = 0.55

    # Robot: near centre, fixed yaw=0 (face +X)
    rx = float(rng.uniform(-0.3, 0.3))
    ry = float(rng.uniform(-0.3, 0.3))
    robot_yaw = 0.0

    # Choose 3 unique (color, shape) combos
    all_combos = [(ci, si) for ci in range(len(COLORS)) for si in range(len(SHAPES))]
    chosen_indices = rng.choice(len(all_combos), size=3, replace=False)
    chosen_combos  = [all_combos[k] for k in chosen_indices]
    target_local   = 0   # first combo is always the target

    objects = []
    fov_half_rad = math.radians(SEARCH_FOV_HALF_DEG)

    for local_i, (ci, si) in enumerate(chosen_combos):
        color_name, color_rgb = COLORS[ci]
        shape_name, size      = SHAPES[si]
        size_val = float(size)
        is_target = (local_i == target_local)

        placed = False
        for _ in range(5000):
            if is_target:
                # Force OUT of FOV: bearing must be > SEARCH_FOV_HALF_DEG from robot yaw
                d = float(rng.uniform(SEARCH_DIST_MIN, SEARCH_DIST_MAX))
                # Sample bearing in the "behind" arc: outside ±fov_half_rad from robot_yaw
                # Use two side arcs: [robot_yaw+fov_half_rad, robot_yaw+pi] and
                # [robot_yaw-pi, robot_yaw-fov_half_rad]
                side = rng.integers(2)
                if side == 0:
                    angle = float(rng.uniform(robot_yaw + fov_half_rad,
                                              robot_yaw + math.pi))
                else:
                    angle = float(rng.uniform(robot_yaw - math.pi,
                                              robot_yaw - fov_half_rad))
            else:
                # Distractors: anywhere, at least 0.8m from robot
                d = float(rng.uniform(0.8, 3.5))
                angle = float(rng.uniform(-math.pi, math.pi))

            ox = rx + d * math.cos(angle)
            oy = ry + d * math.sin(angle)

            # Bounds check
            if abs(ox) + size_val / 2 + margin >= arena_half:
                continue
            if abs(oy) + size_val / 2 + margin >= arena_half:
                continue

            # No overlap with already-placed objects (min 0.8m)
            if any(math.hypot(ox - o["x"], oy - o["y"]) < 0.8 for o in objects):
                continue

            # Verify target bearing constraint
            if is_target:
                dx, dy = ox - rx, oy - ry
                obj_angle = math.atan2(dy, dx)
                err = math.atan2(math.sin(obj_angle - robot_yaw),
                                 math.cos(obj_angle - robot_yaw))
                if abs(err) <= fov_half_rad:
                    continue   # accidentally inside FOV, resample

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
            # Fallback: place anywhere in arena (relaxed constraints)
            for _ in range(10000):
                if is_target:
                    # Must still be out of FOV
                    side = rng.integers(2)
                    d    = float(rng.uniform(SEARCH_DIST_MIN, SEARCH_DIST_MAX))
                    if side == 0:
                        angle = float(rng.uniform(robot_yaw + fov_half_rad,
                                                   robot_yaw + math.pi))
                    else:
                        angle = float(rng.uniform(robot_yaw - math.pi,
                                                   robot_yaw - fov_half_rad))
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

    tgt = objects[target_local]
    instruction = _make_instruction(rng, tgt["color_name"], tgt["shape_name"])

    # Compute initial bearing for verification
    dx, dy = tgt["x"] - rx, tgt["y"] - ry
    init_angle = math.atan2(dy, dx)
    init_bearing_deg = abs(math.degrees(
        math.atan2(math.sin(init_angle - robot_yaw), math.cos(init_angle - robot_yaw))
    ))

    return {
        "arena_size":       arena_half,
        "robot_xy":         (rx, ry),
        "robot_yaw":        robot_yaw,
        "objects":          objects,
        "target_index":     target_local,
        "instruction":      instruction,
        "stop_r":           STOP_R_SEARCH,
        "horizon":          MAXSTEPS_SEARCH,
        "lighting":         {"ambient": 0.4},
        "difficulty":       "search",
        "init_bearing_deg": init_bearing_deg,   # diagnostic: how far out of FOV
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    ep_idx:          int
    instruction:     str
    target_color:    str
    target_shape:    str
    target_dist:     float
    init_bearing_deg: float       # initial bearing to target (must be > SEARCH_FOV_HALF_DEG)
    spotted:         bool          # target entered FOV during scan (scan_active became False)
    reached:         bool          # final_dist < stop_r AND upright
    success:         bool          # spotted AND reached
    failure_tag:     str
    steps:           int
    scan_steps:      int           # steps spent in scan mode (until spotted or timeout)
    final_dist:      float
    fell:            bool
    ms_per_step:     float
    video_path:      str | None = None
    avoid_bias_active_frac: float = 0.0   # NX-9: fraction of grounding cycles with |bias|>0
