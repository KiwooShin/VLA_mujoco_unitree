"""
maneuver_scene.py — Scene sampler for the MANEUVER skill.

Task: "go straight, turn {left/right} after passing the {color}{shape}"

Layout:
  - Robot starts near one edge of the arena facing inward (+X direction).
  - Landmark object placed 3-5m ahead (in front direction) as the trigger.
  - Additional 2-4 distractors placed elsewhere.
  - After passing the landmark (robot_x > landmark_x + PASS_MARGIN), execute the turn.
  - Turn target: robot faces left (90 deg CCW) or right (90 deg CW).

Coordinate system:
  - Robot always starts facing +X direction (yaw=0).
  - Landmark is placed at distance LANDMARK_DIST ahead (+X) with small Y jitter.
  - Turn direction is left or right, encoded as target_yaw = +90° or -90°.

scene_cfg additions (beyond normal scene_cfg):
  - 'task': 'maneuver'
  - 'landmark_index': int  (index into objects)
  - 'turn_direction': 'left' | 'right'  (which way to turn after landmark)
  - 'target_heading': float  (rad, final heading after turn: +pi/2 = left, -pi/2 = right)
  - 'landmark_xy': (float, float)
  - 'pass_margin': float  (how far past landmark before turn is triggered)
"""

import math

import numpy as np

from code.sim.arena_build import COLORS, SHAPES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARENA_HALF       = 5.0    # 10m arena
ROBOT_OFFSET     = 0.75   # fraction of arena: robot starts near edge
LANDMARK_DIST_MIN = 3.0   # min distance to landmark (m)
LANDMARK_DIST_MAX = 5.5   # max distance to landmark (m)
LANDMARK_Y_JITTER = 0.5   # lateral jitter on landmark placement (m)
PASS_MARGIN      = 0.6    # how far robot must go past landmark_x before turn triggers
MIN_PAIRWISE     = 1.0    # min spacing between objects
STOP_R           = 0.4    # not used for maneuver, but kept for schema compatibility
HORIZON          = 1400   # max steps (same as demo)
SETTLE_STEPS     = 80

# Instruction templates
_DIRECTIONS = ['left', 'right']
_VERB_STRAIGHT = ["go straight", "walk straight", "head forward", "proceed straight"]
_VERB_TURN     = ["turn {d}", "make a {d} turn", "bear {d}", "swing {d}"]
_TEMPLATES = [
    "go straight then turn {d} after passing the {c} {s}",
    "walk forward and turn {d} after you pass the {c} {s}",
    "head straight, then turn {d} when you pass the {c} {s}",
    "pass the {c} {s} and turn {d}",
    "proceed past the {c} {s} then turn {d}",
    "when you pass the {c} {s}, turn {d}",
]


def _make_instruction(rng: np.random.Generator, color: str, shape: str, direction: str) -> str:
    """Render a random maneuver instruction template for the given landmark/turn."""
    tpl = _TEMPLATES[int(rng.integers(len(_TEMPLATES)))]
    return tpl.format(c=color, s=shape, d=direction)


def derive_rng(base_seed: int, episode_idx: int) -> np.random.Generator:
    """Create a per-episode RNG derived from base_seed and episode_idx."""
    ss = np.random.SeedSequence([base_seed, episode_idx])
    return np.random.default_rng(ss)


def sample_maneuver_scene(rng: np.random.Generator) -> dict:
    """Sample a deterministic maneuver scene configuration.

    Robot at (-ARENA_HALF * ROBOT_OFFSET, jitter) facing +X.
    Landmark at (rx + LANDMARK_DIST, small_y_jitter).
    Turn direction: left or right (random).
    2-4 distractor objects elsewhere.

    Args:
        rng: Caller-owned RNG; advances state.

    Returns:
        dict compatible with build_arena() plus maneuver-specific keys.
    """
    arena_half = ARENA_HALF
    margin = 0.6

    # Robot start position (left edge, facing +X)
    rx = -arena_half * ROBOT_OFFSET + float(rng.uniform(-0.3, 0.3))
    ry = float(rng.uniform(-0.8, 0.8))
    robot_yaw = 0.0  # always face +X

    # Turn direction
    turn_dir = _DIRECTIONS[int(rng.integers(2))]
    target_heading = math.pi / 2.0 if turn_dir == 'left' else -math.pi / 2.0

    # Landmark placement: ahead and roughly aligned with robot
    landmark_dist = float(rng.uniform(LANDMARK_DIST_MIN, LANDMARK_DIST_MAX))
    landmark_x = rx + landmark_dist
    landmark_y = ry + float(rng.uniform(-LANDMARK_Y_JITTER, LANDMARK_Y_JITTER))

    # Clamp to arena
    landmark_x = max(-(arena_half - margin), min(arena_half - margin, landmark_x))
    landmark_y = max(-(arena_half - margin), min(arena_half - margin, landmark_y))

    # Choose landmark color/shape (unique)
    all_combos = [(ci, si) for ci in range(len(COLORS)) for si in range(len(SHAPES))]
    landmark_combo_idx = int(rng.integers(len(all_combos)))
    lci, lsi = all_combos[landmark_combo_idx]
    l_color_name, l_color_rgb = COLORS[lci]
    l_shape_name, l_size = SHAPES[lsi]

    landmark_obj = {
        "color_name": l_color_name,
        "color_rgb":  l_color_rgb,
        "shape_name": l_shape_name,
        "size":       float(l_size),
        "x":          float(landmark_x),
        "y":          float(landmark_y),
        "dist_from_robot": float(landmark_dist),
    }

    # Distractor objects
    used_combos = {landmark_combo_idx}
    n_distractors = int(rng.integers(2, 5))  # 2-4 distractors

    objects = [landmark_obj]
    for _ in range(n_distractors):
        # Pick unique combo
        for attempt in range(200):
            ci2 = int(rng.integers(len(all_combos)))
            if ci2 not in used_combos:
                used_combos.add(ci2)
                break
        else:
            continue  # give up on this distractor

        d_ci, d_si = all_combos[ci2]
        d_color_name, d_color_rgb = COLORS[d_ci]
        d_shape_name, d_size = SHAPES[d_si]
        d_size_v = float(d_size)

        # Place distractor anywhere (not too close to landmark or robot)
        for _ in range(500):
            dx = float(rng.uniform(-(arena_half - margin), arena_half - margin))
            dy = float(rng.uniform(-(arena_half - margin), arena_half - margin))

            # Not too close to other objects
            too_close = False
            for obj in objects:
                if math.hypot(dx - obj["x"], dy - obj["y"]) < MIN_PAIRWISE:
                    too_close = True
                    break
            # Not too close to robot start
            if math.hypot(dx - rx, dy - ry) < 0.8:
                too_close = True

            if not too_close:
                objects.append({
                    "color_name": d_color_name,
                    "color_rgb":  d_color_rgb,
                    "shape_name": d_shape_name,
                    "size":       d_size_v,
                    "x":          float(dx),
                    "y":          float(dy),
                    "dist_from_robot": float(math.hypot(dx - rx, dy - ry)),
                })
                break

    # Landmark is always index 0
    landmark_index = 0
    instruction = _make_instruction(rng, l_color_name, l_shape_name, turn_dir)

    return {
        # Standard scene_cfg keys (for build_arena compatibility)
        "arena_size":    arena_half,
        "robot_xy":      (rx, ry),
        "robot_yaw":     robot_yaw,
        "objects":       objects,
        "target_index":  landmark_index,   # landmark is the "target"
        "instruction":   instruction,
        "stop_r":        STOP_R,
        "horizon":       HORIZON,
        "lighting":      {"ambient": float(rng.uniform(0.35, 0.60))},
        "difficulty":    "maneuver",

        # Maneuver-specific keys
        "task":          "maneuver",
        "landmark_index": landmark_index,
        "landmark_xy":   (float(landmark_x), float(landmark_y)),
        "turn_direction": turn_dir,
        "target_heading": float(target_heading),
        "pass_margin":   PASS_MARGIN,
    }


if __name__ == "__main__":
    rng = derive_rng(42, 0)
    sc = sample_maneuver_scene(rng)
    lm = sc["objects"][sc["landmark_index"]]
    print(f"Instruction: {sc['instruction']}")
    print(f"Robot start: ({sc['robot_xy'][0]:.2f}, {sc['robot_xy'][1]:.2f}) yaw={sc['robot_yaw']:.2f}")
    print(f"Landmark: {lm['color_name']} {lm['shape_name']} at ({lm['x']:.2f}, {lm['y']:.2f})  dist={lm['dist_from_robot']:.2f}m")
    print(f"Turn: {sc['turn_direction']}  target_heading={math.degrees(sc['target_heading']):.1f} deg")
    print(f"Pass margin: {sc['pass_margin']} m")
    print(f"Objects: {len(sc['objects'])} total")
    print("maneuver_scene.py smoke PASS")
