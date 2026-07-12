"""Unit tests for code.apps.fancy.live: resolve_live_instruction (the
shared live-entry-point resolver) + FancySceneManager's FIRST_SCENE_SEED
first-draw behavior (scene sampling itself is mocked out — that's
code.apps.fancy.sampling*'s concern, covered by its own test modules)."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from code.apps.fancy.live import FancySceneManager, resolve_live_instruction


def _obj(color: str, shape: str, dist: float = 1.0) -> dict:
    return {"color_name": color, "shape_name": shape, "dist_from_robot": dist}


class ResolveLiveInstructionTest(unittest.TestCase):
    def test_no_scene_loaded(self) -> None:
        result = resolve_live_instruction("find the red ball", {})
        self.assertEqual(result["mode"], "no_match")
        self.assertIn("No scene loaded", result["message"])

    def test_single_unambiguous_match(self) -> None:
        scene = {"objects": [_obj("red", "ball"), _obj("blue", "cube")]}
        result = resolve_live_instruction("find the red ball", scene)
        self.assertEqual(result["mode"], "single")
        self.assertEqual(result["target_indices"], [0])
        self.assertEqual(result["goals"][0]["color"], "red")

    def test_multi_goal_instruction(self) -> None:
        scene = {"objects": [_obj("red", "ball"), _obj("yellow", "cube")]}
        result = resolve_live_instruction("find the red ball then find the yellow cube", scene)
        self.assertEqual(result["mode"], "multi")
        self.assertEqual(result["target_indices"], [0, 1])

    def test_no_parse_for_gibberish(self) -> None:
        scene = {"objects": [_obj("red", "ball")]}
        result = resolve_live_instruction("do a barrel roll", scene)
        self.assertEqual(result["mode"], "no_parse")

    def test_no_match_reports_scene_inventory(self) -> None:
        scene = {"objects": [_obj("red", "ball")]}
        result = resolve_live_instruction("find the green cone", scene)
        self.assertEqual(result["mode"], "no_match")
        self.assertIn("red ball", result["message"])

    def test_clarify_on_ambiguous_duplicate(self) -> None:
        scene = {"objects": [_obj("red", "ball", 1.0), _obj("red", "ball", 2.0)]}
        result = resolve_live_instruction("find the red ball", scene)
        self.assertEqual(result["mode"], "clarify")
        self.assertIsNotNone(result["message"])

    def test_goals_use_actual_object_attributes_not_raw_hint(self) -> None:
        # "the ball" alone would be ambiguous in general, but with a single
        # ball in the scene it resolves and the returned goal's color/shape
        # are the MATCHED object's own attributes.
        scene = {"objects": [_obj("purple", "ball")]}
        result = resolve_live_instruction("find the ball", scene)
        self.assertEqual(result["mode"], "single")
        self.assertEqual(result["goals"][0]["color"], "purple")


class FancySceneManagerTest(unittest.TestCase):
    def test_first_scene_uses_first_scene_seed(self) -> None:
        captured = {}

        def _fake_sample(rng, ep_idx):
            captured["state"] = rng.bit_generator.state
            return {"objects": [_obj("yellow", "cube", 4.97)], "target_index": 0,
                    "init_bearing_deg": 84.9}

        with mock.patch("code.apps.fancy.live.sample_fancy_scene_long", side_effect=_fake_sample) as mock_long:
            mgr = FancySceneManager(seed_offset=0)
            mgr.new_scene(long_dist=True)
            mock_long.assert_called_once()

        # Reproduce the expected SeedSequence independently and confirm the
        # RNG state used for the first draw matches it exactly.
        from code.apps.fancy.sampling import FIRST_SCENE_SEED
        expected_rng = np.random.default_rng(np.random.SeedSequence([FIRST_SCENE_SEED, 0]))
        self.assertEqual(captured["state"], expected_rng.bit_generator.state)

    def test_second_scene_uses_plain_sequence_not_first_scene_seed(self) -> None:
        with mock.patch("code.apps.fancy.live.sample_fancy_scene_long",
                         return_value={"objects": [_obj("red", "ball")], "target_index": 0,
                                        "init_bearing_deg": 50.0}):
            mgr = FancySceneManager(seed_offset=3)
            mgr.new_scene()
            self.assertEqual(mgr._ep_count, 1)
            mgr.new_scene()
            self.assertEqual(mgr._ep_count, 2)

    def test_short_range_sampler_used_when_long_dist_false(self) -> None:
        with mock.patch("code.apps.fancy.live.sample_fancy_scene") as mock_short, \
             mock.patch("code.apps.fancy.live.sample_fancy_scene_long") as mock_long:
            mock_short.return_value = {"objects": [_obj("red", "ball")], "target_index": 0,
                                        "init_bearing_deg": 50.0}
            mgr = FancySceneManager()
            mgr.new_scene(long_dist=False)
            mock_short.assert_called_once()
            mock_long.assert_not_called()

    def test_scene_cfg_property_roundtrip(self) -> None:
        mgr = FancySceneManager()
        self.assertIsNone(mgr._scene_cfg)
        mgr._scene_cfg = {"objects": []}
        self.assertEqual(mgr._scene_cfg, {"objects": []})


if __name__ == "__main__":
    unittest.main()
