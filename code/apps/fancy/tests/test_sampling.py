"""Unit tests for code.apps.fancy.sampling: sample_fancy_scene + the VF-5
object-placement helpers (_place_fancy_object_xy, _select_fancy_distractor_
combos), and the FIRST_SCENE_SEED curation constant.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from code.apps.fancy.constants import RELIABLE_COLORS
from code.apps.fancy.sampling import (
    FANCY_MIN_OBJECTS, FANCY_OBJ_MIN_ROBOT_M, FANCY_OBJ_MIN_SEP_M,
    FANCY_OBJ_WALL_MARGIN_M, FIRST_SCENE_SEED,
    _place_fancy_object_xy, _select_fancy_distractor_combos, sample_fancy_scene,
)


class SampleFancySceneTest(unittest.TestCase):
    def test_deterministic_given_same_seed(self) -> None:
        rng1 = np.random.default_rng(np.random.SeedSequence([1, 0]))
        rng2 = np.random.default_rng(np.random.SeedSequence([1, 0]))
        sc1 = sample_fancy_scene(rng1, 0)
        sc2 = sample_fancy_scene(rng2, 0)
        self.assertEqual(sc1["objects"], sc2["objects"])
        self.assertEqual(sc1["robot_xy"], sc2["robot_xy"])

    def test_target_color_is_reliable(self) -> None:
        for seed in range(10):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
            sc = sample_fancy_scene(rng, 0)
            tgt = sc["objects"][sc["target_index"]]
            self.assertIn(tgt["color_name"], RELIABLE_COLORS)

    def test_target_outside_initial_fov(self) -> None:
        for seed in range(20):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            sc = sample_fancy_scene(rng, 0)
            self.assertGreater(sc["init_bearing_deg"], 45.0 - 1e-6)

    def test_target_within_distance_band(self) -> None:
        for seed in range(20):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
            sc = sample_fancy_scene(rng, 0)
            tgt = sc["objects"][0]
            self.assertGreaterEqual(tgt["dist_from_robot"], 2.0 - 1e-6)
            self.assertLessEqual(tgt["dist_from_robot"], 4.0 + 1e-6)

    def test_returns_three_objects(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([7, 0]))
        sc = sample_fancy_scene(rng, 0)
        self.assertEqual(len(sc["objects"]), 3)

    def test_scene_cfg_has_required_keys(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([7, 0]))
        sc = sample_fancy_scene(rng, 0)
        for key in ("arena_size", "robot_xy", "robot_yaw", "objects",
                    "target_index", "stop_r", "horizon", "difficulty"):
            self.assertIn(key, sc)
        self.assertEqual(sc["target_index"], 0)
        self.assertEqual(sc["difficulty"], "search")


class PlaceFancyObjectXyTest(unittest.TestCase):
    def test_respects_distance_bounds(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(30):
            ox, oy = _place_fancy_object_xy(
                rng, rx=0.0, ry=0.0, robot_yaw=0.0, arena_half=8.0, size_val=0.3,
                existing=[], dist_bounds=(2.0, 4.0),
            )
            d = math.hypot(ox, oy)
            self.assertGreaterEqual(d, 2.0 - 0.05)
            self.assertLessEqual(d, 4.0 + 0.05)

    def test_out_of_fov_respected(self) -> None:
        rng = np.random.default_rng(1)
        fov_half = math.radians(45.0)
        for _ in range(30):
            ox, oy = _place_fancy_object_xy(
                rng, rx=0.0, ry=0.0, robot_yaw=0.0, arena_half=8.0, size_val=0.3,
                existing=[], dist_bounds=(2.0, 4.0),
                out_of_fov=True, fov_half_rad=fov_half,
            )
            bearing = abs(math.atan2(oy, ox))
            self.assertGreaterEqual(bearing, fov_half - 1e-6)

    def test_min_separation_from_existing_objects(self) -> None:
        rng = np.random.default_rng(2)
        existing = [{"x": 2.0, "y": 0.0}]
        ox, oy = _place_fancy_object_xy(
            rng, rx=0.0, ry=0.0, robot_yaw=0.0, arena_half=8.0, size_val=0.3,
            existing=existing, dist_bounds=(1.0, 3.0), min_sep=1.2,
        )
        d = math.hypot(ox - 2.0, oy - 0.0)
        self.assertGreaterEqual(d, 1.2 - 0.05)

    def test_min_robot_distance_respected(self) -> None:
        rng = np.random.default_rng(3)
        ox, oy = _place_fancy_object_xy(
            rng, rx=0.0, ry=0.0, robot_yaw=0.0, arena_half=8.0, size_val=0.3,
            existing=[], dist_bounds=(0.0, 5.0), min_robot_dist=1.0,
        )
        self.assertGreaterEqual(math.hypot(ox, oy), 1.0 - 0.05)

    def test_deterministic_given_same_rng_state(self) -> None:
        r1 = np.random.default_rng(np.random.SeedSequence([9]))
        r2 = np.random.default_rng(np.random.SeedSequence([9]))
        a = _place_fancy_object_xy(r1, 0.0, 0.0, 0.0, 8.0, 0.3, [], (2.0, 4.0))
        b = _place_fancy_object_xy(r2, 0.0, 0.0, 0.0, 8.0, 0.3, [], (2.0, 4.0))
        self.assertEqual(a, b)


class SelectFancyDistractorCombosTest(unittest.TestCase):
    def test_correct_count_and_distinctness(self) -> None:
        rng = np.random.default_rng(0)
        combos = _select_fancy_distractor_combos(
            rng, primary_combos=[(0, 0)], n_distractors=6, n_colors=7, n_shapes=4,
        )
        self.assertEqual(len(combos), 6)
        self.assertEqual(len(set(combos)), 6)
        self.assertNotIn((0, 0), combos)

    def test_guarantees_same_color_different_shape_partner(self) -> None:
        rng = np.random.default_rng(1)
        combos = _select_fancy_distractor_combos(
            rng, primary_combos=[(2, 1)], n_distractors=3, n_colors=7, n_shapes=4,
        )
        same_color_diff_shape = [c for c in combos if c[0] == 2 and c[1] != 1]
        self.assertGreaterEqual(len(same_color_diff_shape), 1)

    def test_zero_distractors_returns_empty(self) -> None:
        rng = np.random.default_rng(2)
        combos = _select_fancy_distractor_combos(
            rng, primary_combos=[(0, 0)], n_distractors=0, n_colors=7, n_shapes=4,
        )
        self.assertEqual(combos, [])

    def test_saturating_when_pool_too_small(self) -> None:
        # Only 2x2=4 combos total; excluding 1 primary leaves 3 available —
        # asking for 10 distractors must not crash, and returns <= remaining.
        rng = np.random.default_rng(3)
        combos = _select_fancy_distractor_combos(
            rng, primary_combos=[(0, 0)], n_distractors=10, n_colors=2, n_shapes=2,
        )
        self.assertLessEqual(len(combos), 3)
        self.assertEqual(len(set(combos)), len(combos))


class FirstSceneSeedTest(unittest.TestCase):
    def test_seed_is_a_positive_int(self) -> None:
        self.assertIsInstance(FIRST_SCENE_SEED, int)
        self.assertGreater(FIRST_SCENE_SEED, 0)

    def test_placement_constants_are_positive(self) -> None:
        self.assertGreater(FANCY_MIN_OBJECTS, 0)
        self.assertGreater(FANCY_OBJ_MIN_SEP_M, 0)
        self.assertGreater(FANCY_OBJ_WALL_MARGIN_M, 0)
        self.assertGreater(FANCY_OBJ_MIN_ROBOT_M, 0)


if __name__ == "__main__":
    unittest.main()
