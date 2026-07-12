"""
scene.py — Deterministic scene sampler for G1Nav.

API
---
sample_scene(rng, difficulty) -> scene_cfg dict

The returned dict is consumed by arena.build_arena() and gen_dataset.py.

Difficulty presets
------------------
easy:
  arena ~4m half-size, 3 objects, target IN FOV (within +/-FOV_HALF_DEG of robot heading),
  dist ~1.5-2.5 m, STOP_R=0.6, horizon ~600 steps

demo:
  arena ~5-6m half-size (10-12m total), 5-7 objects FAR apart (min pairwise ~2.5m),
  robot at edge, target dist 4-9 m, target often OUT of initial FOV, STOP_R=0.4,
  horizon ~1400 steps, varied lighting

Phase-1 constraint: every (color, shape) pair UNIQUE per scene.
Instruction template: "{go to|walk to|approach|head to} the {color} {shape}"
"""

import math
import os as _os
import sys

import numpy as np

# Defensive sys.path bootstrap so this module (and its __main__ smoke block)
# still resolves `code.*` imports when run directly without PYTHONPATH set.
# RF-1 note: this file moved from code/scene.py to code/sim/scene.py (one
# level deeper), so the walk-up to the repo root now needs 3 dirname() hops
# instead of the original 2 — same destination (repo root), same behavior.
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from code.sim.arena_build import COLORS, SHAPES

# ---------------------------------------------------------------------------
# Instruction vocabulary
# ---------------------------------------------------------------------------
_VERBS: list[str] = [
    "go to", "walk to", "approach", "head to",
    "head over to", "move to", "navigate to",
    "make your way to", "get to", "proceed to",
]
_TEMPLATES: list[str] = [
    "{v} the {c} {s}",
    "{v} the {s} that is {c}",
    "{v} the {c}-colored {s}",
    "please {v} the {c} {s}",
    "your goal is the {c} {s}",
    "find the {c} {s} and {v} it",
    "{v} the {c} {s} over there",
]

def _make_instruction(rng: np.random.Generator, color: str, shape: str) -> str:
    """Render a random instruction template for the given target color/shape."""
    tpl = _TEMPLATES[int(rng.integers(len(_TEMPLATES)))]
    verb = _VERBS[int(rng.integers(len(_VERBS)))]
    return tpl.format(v=verb, c=color, s=shape)


# ---------------------------------------------------------------------------
# Per-difficulty defaults
# ---------------------------------------------------------------------------
DIFFICULTY_PRESETS: dict[str, dict] = {
    "easy": dict(
        arena_half    = 4.0,      # half-side of square arena (m)
        n_objects     = 3,
        target_in_fov = True,     # target placed within +/-FOV_HALF_DEG of initial heading
        fov_half_deg  = 35.0,     # robot starts facing +X; target must be within this cone
        dist_min      = 1.5,
        dist_max      = 2.5,
        min_pairwise  = 0.70,     # minimum object-object spacing (m)
        robot_offset  = 0.0,      # fraction of arena: 0=centre, 1=edge
        stop_r        = 0.6,
        horizon       = 600,
        lighting_min  = 0.4,
        lighting_max  = 0.4,
    ),
    "demo": dict(
        arena_half    = 5.5,      # -> ~11m arena total
        n_objects_min = 5,
        n_objects_max = 7,
        target_in_fov = False,    # target often out of FOV — triggers turn-in-place
        fov_half_deg  = 30.0,
        dist_min      = 4.0,
        dist_max      = 9.0,
        min_pairwise  = 2.5,
        robot_offset  = 0.75,     # robot starts near edge
        stop_r        = 0.4,
        horizon       = 1400,
        lighting_min  = 0.25,
        lighting_max  = 0.65,
    ),
}


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------
def sample_scene(rng: np.random.Generator, difficulty: str = "easy") -> dict:
    """Sample a deterministic scene configuration.

    Args:
        rng: Caller-owned RNG; advances state.
        difficulty: One of "easy" | "demo".

    Returns:
        dict with keys:
            arena_size      : float  (half-side, m)
            robot_xy        : (float, float)
            robot_yaw       : float  (rad)
            objects         : list of object dicts
            target_index    : int    (index into objects)
            instruction     : str
            stop_r          : float
            horizon         : int
            lighting        : dict  {"ambient": float}
            difficulty      : str

    Raises:
        ValueError: If `difficulty` is not a known preset.
    """
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"Unknown difficulty {difficulty!r}. Choose from {list(DIFFICULTY_PRESETS)}")

    p = DIFFICULTY_PRESETS[difficulty]

    # ---- Arena ----
    arena_half = float(p["arena_half"])
    margin = 0.55  # keep objects away from walls

    # ---- Robot start ----
    if difficulty == "easy":
        # Robot near centre (slight random jitter)
        rx = float(rng.uniform(-0.5, 0.5))
        ry = float(rng.uniform(-0.5, 0.5))
        robot_yaw = 0.0   # faces +X
    else:  # demo
        # Robot near one edge, facing inward
        side = int(rng.integers(4))
        offset = float(p["robot_offset"]) * arena_half
        if side == 0:   # near -X wall, face +X
            rx, ry = -offset, float(rng.uniform(-offset * 0.5, offset * 0.5))
            robot_yaw = 0.0
        elif side == 1: # near +X wall, face -X
            rx, ry = offset, float(rng.uniform(-offset * 0.5, offset * 0.5))
            robot_yaw = math.pi
        elif side == 2: # near -Y wall, face +Y
            rx, ry = float(rng.uniform(-offset * 0.5, offset * 0.5)), -offset
            robot_yaw = math.pi / 2.0
        else:           # near +Y wall, face -Y
            rx, ry = float(rng.uniform(-offset * 0.5, offset * 0.5)), offset
            robot_yaw = -math.pi / 2.0

    robot_xy = (rx, ry)

    # ---- Choose N unique (color, shape) combos ----
    if difficulty == "easy":
        n_objects = int(p["n_objects"])
    else:
        n_objects = int(rng.integers(p["n_objects_min"], p["n_objects_max"] + 1))

    all_combos = [(ci, si) for ci in range(len(COLORS)) for si in range(len(SHAPES))]
    chosen_indices = rng.choice(len(all_combos), size=n_objects, replace=False)
    chosen_combos  = [all_combos[k] for k in chosen_indices]

    # ---- Choose target first (so we can enforce FOV constraint) ----
    target_local_idx = int(rng.integers(n_objects))  # which object is the target

    # ---- Place objects ----
    objects = []

    def _in_bounds(x: float, y: float, size: float) -> bool:
        hs = size / 2.0
        return (abs(x) + hs + margin < arena_half and
                abs(y) + hs + margin < arena_half)

    def _no_overlap(x: float, y: float, placed: list[dict], min_d: float) -> bool:
        return all(math.hypot(x - o["x"], y - o["y"]) >= min_d for o in placed)

    def _in_fov(x: float, y: float, robot_x: float, robot_y: float,
                robot_yaw_rad: float, half_deg: float) -> bool:
        dx, dy = x - robot_x, y - robot_y
        angle  = math.atan2(dy, dx)
        err    = math.atan2(math.sin(angle - robot_yaw_rad),
                            math.cos(angle - robot_yaw_rad))
        return abs(err) <= math.radians(half_deg)

    min_pair = float(p["min_pairwise"])
    dist_min = float(p["dist_min"])
    dist_max = float(p["dist_max"])

    for local_i, (ci, si) in enumerate(chosen_combos):
        color_name, color_rgb = COLORS[ci]
        shape_name, size      = SHAPES[si]
        is_target = (local_i == target_local_idx)
        size_val  = float(size)

        placed = False
        for _ in range(2000):
            # Sample in polar coords around the robot
            if is_target:
                d = float(rng.uniform(dist_min, dist_max))
                if difficulty == "easy" and p["target_in_fov"]:
                    # Force inside FOV cone
                    half_rad = math.radians(float(p["fov_half_deg"]))
                    angle    = float(rng.uniform(robot_yaw - half_rad,
                                                 robot_yaw + half_rad))
                else:
                    angle = float(rng.uniform(-math.pi, math.pi))
            else:
                # Distractors: anywhere in bounds, at least 0.7m from robot
                d     = float(rng.uniform(0.7, arena_half * 1.3))
                angle = float(rng.uniform(-math.pi, math.pi))

            ox = rx + d * math.cos(angle)
            oy = ry + d * math.sin(angle)

            if not _in_bounds(ox, oy, size_val):
                continue
            if not _no_overlap(ox, oy, objects, min_pair):
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
            # Fallback: place near centre with reduced constraints
            for _ in range(5000):
                ox = float(rng.uniform(-(arena_half - margin), arena_half - margin))
                oy = float(rng.uniform(-(arena_half - margin), arena_half - margin))
                if _no_overlap(ox, oy, objects, 0.5):
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

    # ---- Instruction ----
    tgt = objects[target_local_idx]
    instruction = _make_instruction(rng, tgt["color_name"], tgt["shape_name"])

    # ---- Lighting ----
    ambient = float(rng.uniform(p["lighting_min"], p["lighting_max"]))

    return {
        "arena_size":   arena_half,
        "robot_xy":     robot_xy,
        "robot_yaw":    float(robot_yaw),
        "objects":      objects,
        "target_index": target_local_idx,
        "instruction":  instruction,
        "stop_r":       float(p["stop_r"]),
        "horizon":      int(p["horizon"]),
        "lighting":     {"ambient": ambient},
        "difficulty":   difficulty,
    }


# ---------------------------------------------------------------------------
# Seed derivation helper (used by gen_dataset.py)
# ---------------------------------------------------------------------------
def derive_rng(base_seed: int, episode_idx: int) -> np.random.Generator:
    """Create a per-episode RNG derived from base_seed and episode_idx."""
    ss = np.random.SeedSequence([base_seed, episode_idx])
    return np.random.default_rng(ss)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for diff in ("easy", "demo"):
        rng = derive_rng(42, 0)
        sc  = sample_scene(rng, diff)
        tgt = sc["objects"][sc["target_index"]]
        print(f"[{diff}]  arena={sc['arena_size']*2:.1f}m  "
              f"n_obj={len(sc['objects'])}  "
              f"target='{sc['instruction']}'  "
              f"dist={tgt['dist_from_robot']:.2f}m  "
              f"stop_r={sc['stop_r']}  horizon={sc['horizon']}  "
              f"ambient={sc['lighting']['ambient']:.2f}")
    print("scene.py smoke PASS")
