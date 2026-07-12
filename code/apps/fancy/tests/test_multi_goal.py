"""Unit tests for code.apps.fancy.multi_goal: instruction-clause splitting,
goal-hint extraction, ambiguity resolution, and run_fancy_rollout_multi's
sub-goal sequencing (with run_fancy_rollout mocked out — the closed-loop
rollout itself is code.apps.fancy.rollout's concern)."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from code.apps.fancy.multi_goal import (
    _extract_goal_hint, _parse_multi_goal_fancy, _resolve_goal_to_index,
    _split_multi_goal_parts, run_fancy_rollout_multi,
)


def _obj(color: str, shape: str, dist: float = 1.0, x: float = 0.0, y: float = 0.0) -> dict:
    return {"color_name": color, "shape_name": shape, "dist_from_robot": dist, "x": x, "y": y}


class SplitMultiGoalPartsTest(unittest.TestCase):
    def test_single_clause_no_split(self) -> None:
        self.assertEqual(_split_multi_goal_parts("find the red ball"), ["find the red ball"])

    def test_then_separator(self) -> None:
        self.assertEqual(_split_multi_goal_parts("a then b"), ["a", "b"])

    def test_and_then_separator(self) -> None:
        self.assertEqual(_split_multi_goal_parts("a and then b"), ["a", "b"])

    def test_after_that_and_afterwards_and_next(self) -> None:
        for sep in ["after that", "afterwards", "next"]:
            self.assertEqual(_split_multi_goal_parts(f"a {sep} b"), ["a", "b"], msg=sep)

    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(_split_multi_goal_parts(""), [])

    def test_whitespace_only_parts_dropped(self) -> None:
        self.assertEqual(_split_multi_goal_parts("a then   then b"), ["a", "b"])


class ExtractGoalHintTest(unittest.TestCase):
    def test_color_and_shape_both_found(self) -> None:
        hint = _extract_goal_hint("find the red ball")
        self.assertEqual(hint["color"], "red")
        self.assertEqual(hint["shape"], "ball")

    def test_order_independent(self) -> None:
        hint = _extract_goal_hint("the ball that is red")
        self.assertEqual(hint["color"], "red")
        self.assertEqual(hint["shape"], "ball")

    def test_word_boundary_prevents_false_match(self) -> None:
        # "reddish" must not false-match "red" (word-boundary regex).
        hint = _extract_goal_hint("the reddish ball")
        self.assertIsNone(hint["color"])
        self.assertEqual(hint["shape"], "ball")

    def test_multiple_colors_mentioned_leaves_color_none(self) -> None:
        hint = _extract_goal_hint("red or blue ball")
        self.assertIsNone(hint["color"])
        self.assertEqual(hint["colors_mentioned"], {"red", "blue"})

    def test_no_color_or_shape_found(self) -> None:
        hint = _extract_goal_hint("do something")
        self.assertIsNone(hint["color"])
        self.assertIsNone(hint["shape"])
        self.assertEqual(hint["colors_mentioned"], set())

    def test_prompt_part_is_stripped_input(self) -> None:
        hint = _extract_goal_hint("  find the red ball  ")
        self.assertEqual(hint["prompt_part"], "find the red ball")


class ParseMultiGoalFancyTest(unittest.TestCase):
    def test_two_clauses_parsed(self) -> None:
        goals = _parse_multi_goal_fancy("find the red ball then find the yellow cube")
        self.assertEqual(len(goals), 2)
        self.assertEqual(goals[0]["color"], "red")
        self.assertEqual(goals[1]["shape"], "cube")

    def test_clause_with_no_hint_is_dropped(self) -> None:
        goals = _parse_multi_goal_fancy("do a barrel roll then find the red ball")
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["color"], "red")


class ResolveGoalToIndexTest(unittest.TestCase):
    def test_unambiguous_match(self) -> None:
        objs = [_obj("red", "ball"), _obj("blue", "cube")]
        hint = _extract_goal_hint("find the red ball")
        idx, clarify = _resolve_goal_to_index(hint, objs)
        self.assertEqual(idx, 0)
        self.assertIsNone(clarify)

    def test_no_color_or_shape_returns_none_none(self) -> None:
        objs = [_obj("red", "ball")]
        hint = _extract_goal_hint("do something")
        idx, clarify = _resolve_goal_to_index(hint, objs)
        self.assertIsNone(idx)
        self.assertIsNone(clarify)

    def test_no_matching_object_returns_none_none(self) -> None:
        objs = [_obj("red", "ball")]
        hint = _extract_goal_hint("find the green cone")
        idx, clarify = _resolve_goal_to_index(hint, objs)
        self.assertIsNone(idx)
        self.assertIsNone(clarify)

    def test_ambiguous_tie_asks_clarification(self) -> None:
        objs = [_obj("red", "ball", dist=1.0), _obj("red", "ball", dist=2.0)]
        hint = _extract_goal_hint("find the red ball")
        idx, clarify = _resolve_goal_to_index(hint, objs)
        self.assertIsNone(idx)
        self.assertIsNotNone(clarify)

    def test_ambiguous_resolved_by_extra_attribute_score(self) -> None:
        # "the ball" alone is ambiguous between a red ball and blue ball, but
        # mentioning "red" in the same clause breaks the tie via the
        # attribute-overlap scoring (not needing an exact regex match).
        objs = [_obj("red", "ball"), _obj("blue", "ball")]
        hint = _extract_goal_hint("the red ball that I like")
        idx, clarify = _resolve_goal_to_index(hint, objs)
        self.assertEqual(idx, 0)
        self.assertIsNone(clarify)


class RunFancyRolloutMultiTest(unittest.TestCase):
    def _scene(self) -> dict:
        return {"objects": [
            _obj("red", "ball", dist=2.0, x=2.0, y=0.0),
            _obj("yellow", "cube", dist=3.0, x=0.0, y=3.0),
        ]}

    def test_skips_goal_not_in_scene(self) -> None:
        with mock.patch("code.apps.fancy.multi_goal.run_fancy_rollout") as mock_rollout:
            goals = [{"color": "green", "shape": "cone", "prompt_part": "find the green cone"}]
            result = run_fancy_rollout_multi(inf=mock.Mock(), goals=goals, scene_cfg=self._scene(),
                                              render_video=False)
            mock_rollout.assert_not_called()
        self.assertFalse(result["success"])
        self.assertEqual(result["goal_results"][0]["failure_tag"], "not_in_scene")

    def test_sequential_success_marks_completed_targets(self) -> None:
        fake_result = {"success": True, "failure_tag": "success", "steps": 10,
                        "final_dist": 0.1, "frames_sbs": [], "path_trail_out": [],
                        "live_ctx": None}
        with mock.patch("code.apps.fancy.multi_goal.run_fancy_rollout", return_value=fake_result) as mock_rollout:
            goals = [{"color": "red", "shape": "ball", "prompt_part": "find the red ball"}]
            result = run_fancy_rollout_multi(inf=mock.Mock(), goals=goals, scene_cfg=self._scene(),
                                              render_video=False)
        self.assertTrue(result["success"])
        self.assertEqual(mock_rollout.call_count, 1)
        call_kwargs = mock_rollout.call_args.kwargs
        self.assertEqual(call_kwargs["scene_cfg"]["target_index"], 0)
        self.assertTrue(call_kwargs["keep_alive"] is False)  # last (only) goal

    def test_non_last_goal_failure_without_live_ctx_stops_sequence(self) -> None:
        fail_result = {"success": False, "failure_tag": "fall", "steps": 3,
                        "final_dist": 5.0, "frames_sbs": [], "path_trail_out": [],
                        "fell": True, "live_ctx": None}
        with mock.patch("code.apps.fancy.multi_goal.run_fancy_rollout", return_value=fail_result) as mock_rollout:
            goals = [
                {"color": "red", "shape": "ball", "prompt_part": "find the red ball"},
                {"color": "yellow", "shape": "cube", "prompt_part": "find the yellow cube"},
            ]
            result = run_fancy_rollout_multi(inf=mock.Mock(), goals=goals, scene_cfg=self._scene(),
                                              render_video=False)
        self.assertFalse(result["success"])
        # Sequence must stop after the first goal (no live_ctx to resume from).
        self.assertEqual(mock_rollout.call_count, 1)
        self.assertEqual(len(result["goal_results"]), 1)

    def test_keep_alive_true_for_all_but_last_goal(self) -> None:
        ok_with_ctx = {"success": True, "failure_tag": "success", "steps": 1,
                        "final_dist": 0.1, "frames_sbs": [], "path_trail_out": [],
                        "live_ctx": {"renderer": mock.Mock()}}
        ok_no_ctx = {"success": True, "failure_tag": "success", "steps": 1,
                     "final_dist": 0.1, "frames_sbs": [], "path_trail_out": [],
                     "live_ctx": None}
        with mock.patch("code.apps.fancy.multi_goal.run_fancy_rollout",
                         side_effect=[ok_with_ctx, ok_no_ctx]) as mock_rollout:
            goals = [
                {"color": "red", "shape": "ball", "prompt_part": "find the red ball"},
                {"color": "yellow", "shape": "cube", "prompt_part": "find the yellow cube"},
            ]
            result = run_fancy_rollout_multi(inf=mock.Mock(), goals=goals, scene_cfg=self._scene(),
                                              render_video=False)
        self.assertTrue(result["success"])
        first_kwargs = mock_rollout.call_args_list[0].kwargs
        second_kwargs = mock_rollout.call_args_list[1].kwargs
        self.assertTrue(first_kwargs["keep_alive"])
        self.assertFalse(second_kwargs["keep_alive"])
        self.assertIsNone(first_kwargs["resume_ctx"])
        self.assertEqual(second_kwargs["resume_ctx"], ok_with_ctx["live_ctx"])

    def test_total_steps_summed_across_goals(self) -> None:
        r1 = {"success": True, "failure_tag": "success", "steps": 5, "final_dist": 0.1,
              "frames_sbs": [], "path_trail_out": [], "live_ctx": {"renderer": mock.Mock()}}
        r2 = {"success": True, "failure_tag": "success", "steps": 7, "final_dist": 0.1,
              "frames_sbs": [], "path_trail_out": [], "live_ctx": None}
        with mock.patch("code.apps.fancy.multi_goal.run_fancy_rollout", side_effect=[r1, r2]):
            goals = [
                {"color": "red", "shape": "ball", "prompt_part": "find the red ball"},
                {"color": "yellow", "shape": "cube", "prompt_part": "find the yellow cube"},
            ]
            result = run_fancy_rollout_multi(inf=mock.Mock(), goals=goals, scene_cfg=self._scene(),
                                              render_video=False)
        self.assertEqual(result["total_steps"], 12)
        self.assertEqual(result["n_goals"], 2)


if __name__ == "__main__":
    unittest.main()
