"""Unit tests for code.sim.maneuver_scene (MANEUVER skill scene sampler)."""

import math
import unittest

from code.sim.maneuver_scene import (
    ARENA_HALF,
    HORIZON,
    LANDMARK_DIST_MAX,
    LANDMARK_DIST_MIN,
    ROBOT_OFFSET,
    STOP_R,
    _make_instruction,
    derive_rng,
    sample_maneuver_scene,
)


class TestDeriveRng(unittest.TestCase):
    def test_deterministic(self) -> None:
        r1 = derive_rng(1, 2)
        r2 = derive_rng(1, 2)
        self.assertEqual(list(r1.integers(0, 999, size=5)), list(r2.integers(0, 999, size=5)))


class TestSampleManeuverSceneRegressionPin(unittest.TestCase):
    """Pin against the module's own pre-RF-1 documented smoke output (seed=42, ep=0)."""

    def test_seed42_ep0_matches_pre_refactor_smoke(self) -> None:
        rng = derive_rng(42, 0)
        sc = sample_maneuver_scene(rng)
        self.assertEqual(sc["instruction"],
                         "head straight, then turn left when you pass the cyan ball")
        self.assertAlmostEqual(sc["robot_xy"][0], -3.59, places=2)
        self.assertAlmostEqual(sc["robot_xy"][1], -0.10, places=2)
        self.assertEqual(sc["robot_yaw"], 0.0)
        lm = sc["objects"][sc["landmark_index"]]
        self.assertAlmostEqual(lm["x"], 1.16, places=2)
        self.assertAlmostEqual(lm["y"], -0.50, places=2)
        self.assertEqual(sc["turn_direction"], "left")
        self.assertAlmostEqual(math.degrees(sc["target_heading"]), 90.0, places=1)
        self.assertEqual(sc["pass_margin"], 0.6)
        self.assertEqual(len(sc["objects"]), 4)

    def test_repeat_call_is_deterministic(self) -> None:
        sc1 = sample_maneuver_scene(derive_rng(11, 5))
        sc2 = sample_maneuver_scene(derive_rng(11, 5))
        self.assertEqual(sc1, sc2)


class TestSampleManeuverSceneStructure(unittest.TestCase):
    """Structural invariants across many seeds."""

    def _scenes(self, n: int = 20):
        for ep in range(n):
            yield sample_maneuver_scene(derive_rng(999, ep))

    def test_task_is_maneuver(self) -> None:
        for sc in self._scenes():
            self.assertEqual(sc["task"], "maneuver")
            self.assertEqual(sc["difficulty"], "maneuver")

    def test_landmark_always_index_zero(self) -> None:
        for sc in self._scenes():
            self.assertEqual(sc["landmark_index"], 0)
            self.assertEqual(sc["target_index"], 0)
            self.assertIs(sc["objects"][0], sc["objects"][sc["landmark_index"]])

    def test_turn_direction_is_left_or_right(self) -> None:
        for sc in self._scenes():
            self.assertIn(sc["turn_direction"], ("left", "right"))

    def test_target_heading_matches_turn_direction(self) -> None:
        for sc in self._scenes():
            if sc["turn_direction"] == "left":
                self.assertAlmostEqual(sc["target_heading"], math.pi / 2.0)
            else:
                self.assertAlmostEqual(sc["target_heading"], -math.pi / 2.0)

    def test_robot_starts_facing_plus_x(self) -> None:
        for sc in self._scenes():
            self.assertEqual(sc["robot_yaw"], 0.0)

    def test_robot_starts_near_left_edge(self) -> None:
        for sc in self._scenes():
            rx, _ = sc["robot_xy"]
            # ROBOT_OFFSET*ARENA_HALF +/- 0.3 jitter
            expected = -ARENA_HALF * ROBOT_OFFSET
            self.assertAlmostEqual(rx, expected, delta=0.31)

    def test_landmark_distance_in_range_before_clamp(self) -> None:
        for sc in self._scenes():
            rx, _ = sc["robot_xy"]
            lm = sc["objects"][sc["landmark_index"]]
            dist = lm["dist_from_robot"]
            self.assertGreaterEqual(dist, LANDMARK_DIST_MIN - 1e-6)
            self.assertLessEqual(dist, LANDMARK_DIST_MAX + 1e-6)

    def test_landmark_within_arena_bounds(self) -> None:
        for sc in self._scenes():
            lm = sc["objects"][sc["landmark_index"]]
            self.assertLessEqual(abs(lm["x"]), ARENA_HALF)
            self.assertLessEqual(abs(lm["y"]), ARENA_HALF)

    def test_distractor_count_in_range(self) -> None:
        for sc in self._scenes():
            n_distractors = len(sc["objects"]) - 1
            self.assertGreaterEqual(n_distractors, 2)
            self.assertLessEqual(n_distractors, 4)

    def test_unique_color_shape_pairs(self) -> None:
        for sc in self._scenes():
            pairs = [(o["color_name"], o["shape_name"]) for o in sc["objects"]]
            self.assertEqual(len(pairs), len(set(pairs)))

    def test_schema_compatible_with_goto_scene_cfg(self) -> None:
        """build_arena() only needs arena_size/objects/lighting — verify those keys exist
        with the right types regardless of the maneuver-specific extras."""
        for sc in self._scenes(n=3):
            self.assertIsInstance(sc["arena_size"], float)
            self.assertIsInstance(sc["objects"], list)
            self.assertIn("ambient", sc["lighting"])
            self.assertEqual(sc["stop_r"], STOP_R)
            self.assertEqual(sc["horizon"], HORIZON)


class TestMakeInstruction(unittest.TestCase):
    def test_contains_color_shape_and_direction(self) -> None:
        rng = derive_rng(3, 0)
        text = _make_instruction(rng, "green", "cylinder", "right")
        self.assertIn("green", text)
        self.assertIn("cylinder", text)
        self.assertIn("right", text)


if __name__ == "__main__":
    unittest.main()
