"""Unit tests for code.sim.scene (goto/search scene sampler).

Covers: difficulty presets (easy/demo), seeded determinism (including a
hard-coded regression pin against the pre-RF-1 module's own documented
smoke output), placement constraints (bounds, uniqueness, FOV cone for
'easy'), and the derive_rng seed-derivation helper.
"""

import math
import unittest

from code.sim.scene import (
    DIFFICULTY_PRESETS,
    _make_instruction,
    derive_rng,
    sample_scene,
)


class TestDeriveRng(unittest.TestCase):
    """derive_rng seed derivation."""

    def test_same_args_give_identical_stream(self) -> None:
        r1 = derive_rng(42, 0)
        r2 = derive_rng(42, 0)
        self.assertEqual(list(r1.integers(0, 1000, size=10)),
                          list(r2.integers(0, 1000, size=10)))

    def test_different_episode_gives_different_stream(self) -> None:
        r1 = derive_rng(42, 0)
        r2 = derive_rng(42, 1)
        self.assertNotEqual(list(r1.integers(0, 1000, size=10)),
                             list(r2.integers(0, 1000, size=10)))

    def test_different_base_seed_gives_different_stream(self) -> None:
        r1 = derive_rng(42, 0)
        r2 = derive_rng(43, 0)
        self.assertNotEqual(list(r1.integers(0, 1000, size=10)),
                             list(r2.integers(0, 1000, size=10)))


class TestSampleSceneErrors(unittest.TestCase):
    """Input validation."""

    def test_unknown_difficulty_raises(self) -> None:
        rng = derive_rng(1, 0)
        with self.assertRaises(ValueError):
            sample_scene(rng, "impossible")

    def test_error_message_lists_presets(self) -> None:
        rng = derive_rng(1, 0)
        try:
            sample_scene(rng, "nope")
            self.fail("expected ValueError")
        except ValueError as e:
            for key in DIFFICULTY_PRESETS:
                self.assertIn(key, str(e))


class TestSampleSceneRegressionPin(unittest.TestCase):
    """Hard-coded regression pin against the module's own pre-RF-1 smoke output.

    docs/refactor_plan.md invariant #1 requires zero behavior change; these
    exact values were captured by running the ORIGINAL (pre-split) scene.py
    smoke test at seed=(42, 0), so any drift here means sampling logic moved
    during the split, not just the file.
    """

    def test_easy_seed42_ep0_matches_pre_refactor_smoke(self) -> None:
        rng = derive_rng(42, 0)
        sc = sample_scene(rng, "easy")
        self.assertEqual(sc["arena_size"], 4.0)
        self.assertEqual(len(sc["objects"]), 3)
        self.assertEqual(sc["instruction"], "proceed to the cone that is purple")
        tgt = sc["objects"][sc["target_index"]]
        self.assertAlmostEqual(tgt["dist_from_robot"], 2.48, places=2)
        self.assertEqual(sc["stop_r"], 0.6)
        self.assertEqual(sc["horizon"], 600)
        self.assertAlmostEqual(sc["lighting"]["ambient"], 0.40, places=2)

    def test_demo_seed42_ep0_matches_pre_refactor_smoke(self) -> None:
        rng = derive_rng(42, 0)
        sc = sample_scene(rng, "demo")
        self.assertEqual(sc["arena_size"], 5.5)
        self.assertEqual(len(sc["objects"]), 7)
        self.assertEqual(sc["instruction"], "please move to the blue cube")
        tgt = sc["objects"][sc["target_index"]]
        self.assertAlmostEqual(tgt["dist_from_robot"], 6.25, places=2)
        self.assertEqual(sc["stop_r"], 0.4)
        self.assertEqual(sc["horizon"], 1400)

    def test_full_repeat_call_is_byte_identical(self) -> None:
        """Re-deriving the same (seed, episode) twice must give an identical dict."""
        sc1 = sample_scene(derive_rng(7, 3), "demo")
        sc2 = sample_scene(derive_rng(7, 3), "demo")
        self.assertEqual(sc1, sc2)


class TestSampleSceneEasyConstraints(unittest.TestCase):
    """Structural invariants of the 'easy' preset across several seeds."""

    def _scenes(self, n: int = 12):
        for ep in range(n):
            yield sample_scene(derive_rng(123, ep), "easy")

    def test_object_count(self) -> None:
        for sc in self._scenes():
            self.assertEqual(len(sc["objects"]), 3)

    def test_unique_color_shape_pairs(self) -> None:
        for sc in self._scenes():
            pairs = [(o["color_name"], o["shape_name"]) for o in sc["objects"]]
            self.assertEqual(len(pairs), len(set(pairs)))

    def test_robot_starts_facing_plus_x(self) -> None:
        for sc in self._scenes():
            self.assertEqual(sc["robot_yaw"], 0.0)

    def test_objects_within_arena_bounds(self) -> None:
        for sc in self._scenes():
            half = sc["arena_size"]
            for o in sc["objects"]:
                self.assertLess(abs(o["x"]), half)
                self.assertLess(abs(o["y"]), half)

    def test_target_within_fov_cone(self) -> None:
        """easy's target_in_fov=True must place the target within the FOV half-angle."""
        preset = DIFFICULTY_PRESETS["easy"]
        half_rad = math.radians(preset["fov_half_deg"])
        for sc in self._scenes():
            tgt = sc["objects"][sc["target_index"]]
            rx, ry = sc["robot_xy"]
            angle = math.atan2(tgt["y"] - ry, tgt["x"] - rx)
            err = math.atan2(math.sin(angle - sc["robot_yaw"]), math.cos(angle - sc["robot_yaw"]))
            self.assertLessEqual(abs(err), half_rad + 1e-9)

    def test_target_distance_range(self) -> None:
        preset = DIFFICULTY_PRESETS["easy"]
        for sc in self._scenes():
            tgt = sc["objects"][sc["target_index"]]
            self.assertGreaterEqual(tgt["dist_from_robot"], preset["dist_min"] - 1e-6)
            self.assertLessEqual(tgt["dist_from_robot"], preset["dist_max"] + 1e-6)

    def test_lighting_fixed_for_easy(self) -> None:
        preset = DIFFICULTY_PRESETS["easy"]
        for sc in self._scenes():
            self.assertAlmostEqual(sc["lighting"]["ambient"], preset["lighting_min"])


class TestSampleSceneDemoConstraints(unittest.TestCase):
    """Structural invariants of the 'demo' preset across several seeds."""

    def _scenes(self, n: int = 12):
        for ep in range(n):
            yield sample_scene(derive_rng(321, ep), "demo")

    def test_object_count_in_range(self) -> None:
        preset = DIFFICULTY_PRESETS["demo"]
        for sc in self._scenes():
            self.assertGreaterEqual(len(sc["objects"]), preset["n_objects_min"])
            self.assertLessEqual(len(sc["objects"]), preset["n_objects_max"])

    def test_unique_color_shape_pairs(self) -> None:
        for sc in self._scenes():
            pairs = [(o["color_name"], o["shape_name"]) for o in sc["objects"]]
            self.assertEqual(len(pairs), len(set(pairs)))

    def test_lighting_within_range(self) -> None:
        preset = DIFFICULTY_PRESETS["demo"]
        for sc in self._scenes():
            amb = sc["lighting"]["ambient"]
            self.assertGreaterEqual(amb, preset["lighting_min"] - 1e-9)
            self.assertLessEqual(amb, preset["lighting_max"] + 1e-9)

    def test_robot_start_side_yaw_matches_offset_direction(self) -> None:
        """Robot always starts facing roughly toward arena centre (docs/nx12_turn_dwell.md)."""
        for sc in self._scenes():
            rx, ry = sc["robot_xy"]
            yaw = sc["robot_yaw"]
            self.assertIn(round(yaw, 3), [0.0, round(math.pi, 3), round(math.pi / 2, 3),
                                          round(-math.pi / 2, 3)])


class TestMakeInstruction(unittest.TestCase):
    """_make_instruction template rendering (RF-1 keeps this private helper importable —
    eval_search.py imports it directly from the old code.scene path)."""

    def test_contains_color_and_shape(self) -> None:
        rng = derive_rng(9, 0)
        text = _make_instruction(rng, "red", "cube")
        self.assertIn("red", text)
        self.assertIn("cube", text)

    def test_deterministic_for_fixed_rng_state(self) -> None:
        text1 = _make_instruction(derive_rng(5, 0), "blue", "ball")
        text2 = _make_instruction(derive_rng(5, 0), "blue", "ball")
        self.assertEqual(text1, text2)


if __name__ == "__main__":
    unittest.main()
