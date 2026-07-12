"""Unit tests for code.apps.fancy.sampling_long: the FD2 long-distance +
multi-goal scene samplers, incl. the documented FIRST_SCENE_SEED regression
(seed=3461 -> yellow cube, 4.97m, bearing=84.9 deg, 7 objects — see
code/apps/fancy/sampling.py's FIRST_SCENE_SEED comment).
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from code.apps.fancy.constants import ARENA_HALF_LONG, DIST_MAX_LONG, DIST_MIN_LONG, RELIABLE_COLORS
from code.apps.fancy.sampling import FANCY_MIN_OBJECTS, FIRST_SCENE_SEED
from code.apps.fancy.sampling_long import sample_fancy_multi_goal_scene, sample_fancy_scene_long


class SampleFancySceneLongTest(unittest.TestCase):
    def test_first_scene_seed_regression(self) -> None:
        """Pins the exact curated first-launch draw documented in
        code/apps/fancy/sampling.py's FIRST_SCENE_SEED comment."""
        rng = np.random.default_rng(np.random.SeedSequence([FIRST_SCENE_SEED, 0]))
        sc = sample_fancy_scene_long(rng, 0)
        tgt = sc["objects"][sc["target_index"]]
        self.assertEqual(tgt["color_name"], "yellow")
        self.assertEqual(tgt["shape_name"], "cube")
        self.assertAlmostEqual(tgt["dist_from_robot"], 4.97, places=2)
        self.assertAlmostEqual(sc["init_bearing_deg"], 84.9, places=1)
        self.assertEqual(len(sc["objects"]), 7)

    def test_deterministic_given_same_seed(self) -> None:
        rng1 = np.random.default_rng(np.random.SeedSequence([5, 1]))
        rng2 = np.random.default_rng(np.random.SeedSequence([5, 1]))
        sc1 = sample_fancy_scene_long(rng1, 0)
        sc2 = sample_fancy_scene_long(rng2, 0)
        self.assertEqual(sc1["objects"], sc2["objects"])

    def test_at_least_min_objects(self) -> None:
        for seed in range(10):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
            sc = sample_fancy_scene_long(rng, 0)
            self.assertGreaterEqual(len(sc["objects"]), FANCY_MIN_OBJECTS)

    def test_target_within_custom_distance_band(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([11, 0]))
        sc = sample_fancy_scene_long(rng, 0, dist_min=1.0, dist_max=2.0)
        tgt = sc["objects"][0]
        self.assertGreaterEqual(tgt["dist_from_robot"], 1.0 - 0.05)
        self.assertLessEqual(tgt["dist_from_robot"], 2.0 + 0.05)

    def test_target_color_reliable(self) -> None:
        for seed in range(10):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
            sc = sample_fancy_scene_long(rng, 0)
            self.assertIn(sc["objects"][0]["color_name"], RELIABLE_COLORS)

    def test_arena_half_matches_constant(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([1, 0]))
        sc = sample_fancy_scene_long(rng, 0)
        self.assertEqual(sc["arena_size"], ARENA_HALF_LONG)

    def test_same_color_different_shape_pair_present(self) -> None:
        # VF-5 guarantee: at least one same-color/different-shape pair.
        found_any = False
        for seed in range(15):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 4]))
            sc = sample_fancy_scene_long(rng, 0)
            by_color: dict[str, set[str]] = {}
            for o in sc["objects"]:
                by_color.setdefault(o["color_name"], set()).add(o["shape_name"])
            if any(len(shapes) > 1 for shapes in by_color.values()):
                found_any = True
                break
        self.assertTrue(found_any)


class SampleFancyMultiGoalSceneTest(unittest.TestCase):
    def test_deterministic_given_same_seed(self) -> None:
        rng1 = np.random.default_rng(np.random.SeedSequence([2, 0]))
        rng2 = np.random.default_rng(np.random.SeedSequence([2, 0]))
        sc1 = sample_fancy_multi_goal_scene(rng1, n_goals=2)
        sc2 = sample_fancy_multi_goal_scene(rng2, n_goals=2)
        self.assertEqual(sc1["objects"], sc2["objects"])

    def test_first_n_goals_objects_marked_is_goal(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([3, 0]))
        sc = sample_fancy_multi_goal_scene(rng, n_goals=3)
        goal_flags = [o["is_goal"] for o in sc["objects"]]
        self.assertEqual(goal_flags[:3], [True, True, True])
        self.assertTrue(all(not f for f in goal_flags[3:]))

    def test_n_goals_field_matches_request(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([4, 0]))
        sc = sample_fancy_multi_goal_scene(rng, n_goals=2)
        self.assertEqual(sc["n_goals"], 2)

    def test_total_object_count_at_least_goals_plus_five(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([6, 0]))
        sc = sample_fancy_multi_goal_scene(rng, n_goals=2)
        self.assertGreaterEqual(len(sc["objects"]), 2 + 5)

    def test_robot_at_origin(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([8, 0]))
        sc = sample_fancy_multi_goal_scene(rng, n_goals=2)
        self.assertEqual(sc["robot_xy"], (0.0, 0.0))
        self.assertEqual(sc["robot_yaw"], 0.0)

    def test_goals_use_distinct_color_shape_combos(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([10, 0]))
        sc = sample_fancy_multi_goal_scene(rng, n_goals=3)
        combos = [(o["color_name"], o["shape_name"]) for o in sc["objects"][:3]]
        self.assertEqual(len(combos), len(set(combos)))


if __name__ == "__main__":
    unittest.main()
