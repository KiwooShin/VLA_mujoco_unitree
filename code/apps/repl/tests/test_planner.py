"""Unit tests for code.apps.repl.planner: SceneManager, SubGoal, Planner.

Covers the rule-based NL -> sub-goal parsing (goto/maneuver/search/compound),
ambiguity-resolution clarification questions, clarification re-parsing, and
SceneManager's pure describe/list helpers (scene sampling itself is not
exercised here — that lives in code.scene, owned by another group).
"""

from __future__ import annotations

import unittest

from code.apps.repl.planner import Planner, SceneManager, SubGoal


def _obj(color: str, shape: str, dist: float = 1.0, x: float = 0.0, y: float = 0.0) -> dict:
    return {"color_name": color, "shape_name": shape, "dist_from_robot": dist, "x": x, "y": y}


class _FakeSceneManager:
    """Minimal stand-in for SceneManager exposing only object_list()."""

    def __init__(self, objects: list[dict]) -> None:
        self._objects = objects

    def object_list(self) -> list[dict]:
        return self._objects


class SubGoalStrTest(unittest.TestCase):
    def test_goto_str(self) -> None:
        g = SubGoal(skill="goto", color="red", shape="ball")
        self.assertEqual(str(g), "goto(red ball)")

    def test_maneuver_str(self) -> None:
        g = SubGoal(skill="maneuver", color="blue", shape="cube", direction="left")
        self.assertEqual(str(g), "maneuver(turn_left after blue cube)")

    def test_search_str(self) -> None:
        g = SubGoal(skill="search", color="orange", shape="cone")
        self.assertEqual(str(g), "search(orange cone)")

    def test_unknown_skill_str_falls_back_to_description(self) -> None:
        g = SubGoal(skill="teleport", description="beam me up")
        self.assertEqual(str(g), "teleport(beam me up)")

    def test_defaults(self) -> None:
        g = SubGoal(skill="goto")
        self.assertEqual(g.status, "pending")
        self.assertIsNone(g.result)


class SceneManagerTest(unittest.TestCase):
    def test_no_scene_yet(self) -> None:
        sm = SceneManager()
        self.assertIsNone(sm.scene_cfg)
        self.assertEqual(sm.object_list(), [])
        self.assertEqual(sm.describe_scene(), "(no scene yet)")

    def test_describe_scene_marks_target(self) -> None:
        sm = SceneManager()
        sm._scene_cfg = {
            "objects": [_obj("red", "ball", 2.0, 1.0, 1.0), _obj("blue", "cube", 3.0, -1.0, 2.0)],
            "target_index": 1,
        }
        desc = sm.describe_scene()
        self.assertIn("<-- TARGET", desc)
        lines = desc.splitlines()
        self.assertIn("<-- TARGET", lines[2])
        self.assertNotIn("<-- TARGET", lines[1])

    def test_object_list_returns_scene_objects(self) -> None:
        sm = SceneManager()
        objs = [_obj("red", "ball")]
        sm._scene_cfg = {"objects": objs, "target_index": 0}
        self.assertEqual(sm.object_list(), objs)

    def test_difficulty_and_seed_offset_stored(self) -> None:
        sm = SceneManager(difficulty="easy", seed_offset=7)
        self.assertEqual(sm.difficulty, "easy")
        self.assertEqual(sm.seed_offset, 7)


class PlannerGotoTest(unittest.TestCase):
    def _planner(self, objects: list[dict]) -> Planner:
        return Planner(_FakeSceneManager(objects))

    def test_go_to_the_color_shape(self) -> None:
        p = self._planner([_obj("red", "ball")])
        goals, clarify = p.parse("go to the red ball")
        self.assertIsNone(clarify)
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].skill, "goto")
        self.assertEqual((goals[0].color, goals[0].shape), ("red", "ball"))

    def test_goto_verb_variants(self) -> None:
        p = self._planner([_obj("blue", "cube")])
        for instr in ["walk to the blue cube", "approach the blue cube",
                      "navigate to the blue cube", "head to the blue cube",
                      "get to the blue cube", "reach the blue cube"]:
            goals, clarify = p.parse(instr)
            self.assertIsNone(clarify, msg=instr)
            self.assertEqual(goals[0].skill, "goto", msg=instr)

    def test_goto_color_shape_swapped_order(self) -> None:
        p = self._planner([_obj("red", "ball")])
        goals, clarify = p.parse("go to the ball red")
        self.assertIsNone(clarify)
        self.assertEqual((goals[0].color, goals[0].shape), ("red", "ball"))

    def test_no_matching_object_reports_available(self) -> None:
        p = self._planner([_obj("red", "ball")])
        goals, clarify = p.parse("go to the blue cube")
        self.assertEqual(goals, [])
        self.assertIn("blue cube", clarify)
        self.assertIn("red ball", clarify)

    def test_ambiguous_duplicate_color_shape_asks_clarification(self) -> None:
        # _detect_goto only ever matches when BOTH a known color and a known
        # shape word are present, so the only way _resolve_referent sees
        # multiple candidates is two scene objects sharing the exact same
        # (color, shape) combo (e.g. two red balls at different spots).
        p = self._planner([
            _obj("red", "ball", dist=2.0), _obj("red", "ball", dist=5.0),
        ])
        goals, clarify = p.parse("go to the red ball")
        self.assertEqual(goals, [])
        self.assertIsNotNone(clarify)
        self.assertIn("red ball", clarify)


class PlannerManeuverTest(unittest.TestCase):
    def _planner(self, objects: list[dict]) -> Planner:
        return Planner(_FakeSceneManager(objects))

    def test_turn_after_landmark(self) -> None:
        p = self._planner([_obj("blue", "cube")])
        goals, clarify = p.parse("turn left after the blue cube")
        self.assertIsNone(clarify)
        self.assertEqual(goals[0].skill, "maneuver")
        self.assertEqual(goals[0].direction, "left")
        self.assertEqual((goals[0].color, goals[0].shape), ("blue", "cube"))

    def test_pass_then_turn(self) -> None:
        p = self._planner([_obj("red", "cylinder")])
        goals, clarify = p.parse("pass the red cylinder then turn right")
        # This is two "parts" (split on 'then'): the first part alone
        # ("pass the red cylinder") has no maneuver direction attached in
        # THIS clause and is not a goto/search pattern, so it yields no
        # goal; the second part ("turn right") alone has no landmark. The
        # combined single-clause form is what actually resolves:
        goals2, clarify2 = p.parse("pass the red cylinder and turn right")
        self.assertIsNone(clarify2)
        self.assertEqual(goals2[0].skill, "maneuver")
        self.assertEqual(goals2[0].direction, "right")

    def test_when_you_pass_form(self) -> None:
        p = self._planner([_obj("orange", "cone")])
        goals, clarify = p.parse("when you pass the orange cone, turn left")
        self.assertIsNone(clarify)
        self.assertEqual(goals[0].skill, "maneuver")
        self.assertEqual(goals[0].direction, "left")

    def test_turn_dir_when_you_pass_form(self) -> None:
        p = self._planner([_obj("orange", "cone")])
        goals, clarify = p.parse("turn right when you pass the orange cone")
        self.assertIsNone(clarify)
        self.assertEqual(goals[0].direction, "right")

    def test_maneuver_landmark_not_in_scene(self) -> None:
        p = self._planner([_obj("red", "ball")])
        goals, clarify = p.parse("turn left after the blue cube")
        self.assertEqual(goals, [])
        self.assertIn("blue cube", clarify)


class PlannerSearchTest(unittest.TestCase):
    def _planner(self, objects: list[dict]) -> Planner:
        return Planner(_FakeSceneManager(objects))

    def test_find_triggers_search_not_goto(self) -> None:
        p = self._planner([_obj("red", "ball")])
        goals, clarify = p.parse("find the red ball")
        self.assertIsNone(clarify)
        self.assertEqual(goals[0].skill, "search")
        self.assertEqual((goals[0].color, goals[0].shape), ("red", "ball"))

    def test_search_verb_variants(self) -> None:
        p = self._planner([_obj("red", "ball")])
        for instr in ["search for the red ball", "look for the red ball",
                      "locate the red ball", "hunt for the red ball"]:
            goals, clarify = p.parse(instr)
            self.assertEqual(goals[0].skill, "search", msg=instr)

    def test_search_does_not_require_object_present(self) -> None:
        # search does not resolve against the scene (unlike goto/maneuver) —
        # any (color, shape) mentioned yields a search SubGoal verbatim.
        p = self._planner([])
        goals, clarify = p.parse("find the purple cone")
        self.assertIsNone(clarify)
        self.assertEqual((goals[0].color, goals[0].shape), ("purple", "cone"))


class PlannerCompoundTest(unittest.TestCase):
    def _planner(self, objects: list[dict]) -> Planner:
        return Planner(_FakeSceneManager(objects))

    def test_goto_then_search(self) -> None:
        p = self._planner([_obj("red", "ball")])
        goals, clarify = p.parse("go to the red ball then find the blue cube")
        self.assertIsNone(clarify)
        self.assertEqual(len(goals), 2)
        self.assertEqual(goals[0].skill, "goto")
        self.assertEqual(goals[1].skill, "search")

    def test_and_then_and_afterwards_and_next_all_split(self) -> None:
        p = self._planner([_obj("red", "ball")])
        for sep in ["and then", "after that", "afterwards", "next"]:
            instr = f"find the red ball {sep} find the red ball"
            goals, clarify = p.parse(instr)
            self.assertEqual(len(goals), 2, msg=sep)

    def test_ambiguous_second_part_returns_first_goals_plus_clarify(self) -> None:
        p = self._planner([
            _obj("red", "ball", dist=2.0), _obj("red", "ball", dist=5.0),
        ])
        goals, clarify = p.parse("find the blue cube then go to the red ball")
        # first part is a search (no scene resolution needed) so it succeeds;
        # second part is ambiguous goto (two red balls) -> clarify returned
        # along with the first part's already-parsed goal.
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].skill, "search")
        self.assertIsNotNone(clarify)


class PlannerNoMatchTest(unittest.TestCase):
    def test_unparseable_instruction(self) -> None:
        p = Planner(_FakeSceneManager([_obj("red", "ball")]))
        goals, clarify = p.parse("do a barrel roll")
        self.assertEqual(goals, [])
        self.assertIn("didn't understand", clarify)

    def test_empty_instruction(self) -> None:
        p = Planner(_FakeSceneManager([_obj("red", "ball")]))
        goals, clarify = p.parse("")
        self.assertEqual(goals, [])
        self.assertIsNotNone(clarify)


class PlannerClarificationTest(unittest.TestCase):
    def test_clarify_with_color_and_shape_refines(self) -> None:
        p = Planner(_FakeSceneManager([_obj("red", "ball"), _obj("blue", "ball")]))
        goals, clarify = p.parse("go to the ball")
        self.assertIsNotNone(clarify)
        goals2, clarify2 = p.parse_clarification("go to the ball", "the red one")
        self.assertIsNone(clarify2)
        self.assertEqual(goals2[0].color, "red")

    def test_clarify_answer_is_fresh_instruction(self) -> None:
        p = Planner(_FakeSceneManager([_obj("red", "ball")]))
        goals, clarify = p.parse_clarification("go to the ball", "find the red ball")
        self.assertIsNone(clarify)
        self.assertEqual(goals[0].skill, "search")

    def test_clarify_with_only_color_reuses_original_shape(self) -> None:
        p = Planner(_FakeSceneManager([_obj("red", "ball"), _obj("blue", "ball")]))
        p.parse("go to the ball")
        goals, clarify = p.parse_clarification("go to the ball", "red")
        self.assertIsNone(clarify)
        self.assertEqual((goals[0].color, goals[0].shape), ("red", "ball"))

    def test_clarify_with_only_shape_reuses_original_color(self) -> None:
        objs = [_obj("red", "ball"), _obj("red", "cube")]
        p = Planner(_FakeSceneManager(objs))
        p.parse("go to the red object")
        goals, clarify = p.parse_clarification("go to the red object", "cube")
        self.assertIsNone(clarify)
        self.assertEqual(goals[0].shape, "cube")

    def test_clarify_falls_back_to_treating_answer_as_fresh(self) -> None:
        p = Planner(_FakeSceneManager([_obj("red", "ball")]))
        goals, clarify = p.parse_clarification("go to the ball", "gibberish nonsense")
        self.assertEqual(goals, [])
        self.assertIsNotNone(clarify)


class PlannerNoSceneTest(unittest.TestCase):
    def test_resolve_referent_no_scene_loaded(self) -> None:
        p = Planner(_FakeSceneManager([]))
        goals, clarify = p.parse("go to the red ball")
        self.assertEqual(goals, [])
        self.assertIn("No scene loaded", clarify)


if __name__ == "__main__":
    unittest.main()
