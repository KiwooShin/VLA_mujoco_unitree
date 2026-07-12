"""Unit tests for code.apps.repl.executor: EventBus + Executor's sub-goal
dispatch/target-resolution logic (heavy policy rollouts are mocked out —
those closed-loop numerics are gated by the eval_* suites, not this file).
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

from code.apps.repl.executor import EventBus, Executor
from code.apps.repl.planner import SceneManager, SubGoal


def _obj(color: str, shape: str, dist: float = 1.0, x: float = 0.0, y: float = 0.0) -> dict:
    return {"color_name": color, "shape_name": shape, "dist_from_robot": dist, "x": x, "y": y}


class EventBusTest(unittest.TestCase):
    def test_emit_and_get_events(self) -> None:
        bus = EventBus()
        bus.emit({"type": "a"})
        bus.emit({"type": "b"})
        events = bus.get_events()
        self.assertEqual([e["type"] for e in events], ["a", "b"])

    def test_get_events_since_ts_filters(self) -> None:
        bus = EventBus()
        bus.emit({"type": "a"})
        t_mid = time.time()
        time.sleep(0.001)
        bus.emit({"type": "b"})
        events = bus.get_events(since_ts=t_mid)
        self.assertEqual([e["type"] for e in events], ["b"])

    def test_get_state_reflects_last_values_per_key(self) -> None:
        bus = EventBus()
        bus.emit({"type": "a", "x": 1})
        bus.emit({"type": "b", "x": 2, "y": 3})
        state = bus.get_state()
        self.assertEqual(state["type"], "b")
        self.assertEqual(state["x"], 2)
        self.assertEqual(state["y"], 3)

    def test_deque_maxlen_bounds_history(self) -> None:
        bus = EventBus()
        for i in range(250):
            bus.emit({"type": "e", "i": i})
        events = bus.get_events()
        self.assertEqual(len(events), 200)
        # Oldest 50 dropped -> first surviving event has i == 50.
        self.assertEqual(events[0]["i"], 50)

    def test_events_are_independent_snapshots(self) -> None:
        bus = EventBus()
        bus.emit({"type": "a"})
        snapshot = bus.get_events()
        bus.emit({"type": "b"})
        self.assertEqual(len(snapshot), 1)


class ExecutorGotoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = SceneManager()
        self.bus = EventBus()
        self.ex = Executor(scene_manager=self.scene, bus=self.bus, render_video=False)

    def test_goto_target_not_in_scene(self) -> None:
        self.scene._scene_cfg = {"objects": [_obj("red", "ball")], "target_index": 0}
        goal = SubGoal(skill="goto", color="blue", shape="cube", description="navigate to blue cube")
        result = self.ex._run_goto(goal, ep_id=1, gi=0)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_tag"], "target_not_in_scene")

    def test_goto_no_scene_loaded(self) -> None:
        goal = SubGoal(skill="goto", color="red", shape="ball")
        result = self.ex._run_goto(goal, ep_id=1, gi=0)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_tag"], "no_scene")

    def test_goto_success_path_calls_inferencer_with_correct_target(self) -> None:
        self.scene._scene_cfg = {
            "objects": [_obj("red", "ball"), _obj("blue", "cube")],
            "target_index": 0,
        }
        goal = SubGoal(skill="goto", color="blue", shape="cube", description="navigate to blue cube")

        fake_result = mock.Mock(success=True, failure_tag="success", steps=42,
                                 final_dist=0.3, forward_disp=2.0, video_path=None)
        fake_inf = mock.Mock()
        fake_inf.rollout.return_value = fake_result
        self.ex._goto_inferencer = fake_inf

        result = self.ex._run_goto(goal, ep_id=1, gi=0)

        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 42)
        # The scene_cfg passed to rollout() must have target_index overridden
        # to the BLUE CUBE (index 1), not the sampler's own target_index=0.
        call_kwargs = fake_inf.rollout.call_args.kwargs
        self.assertEqual(call_kwargs["scene_cfg"]["target_index"], 1)

    def test_goto_rollout_exception_is_caught(self) -> None:
        self.scene._scene_cfg = {"objects": [_obj("red", "ball")], "target_index": 0}
        goal = SubGoal(skill="goto", color="red", shape="ball")
        fake_inf = mock.Mock()
        fake_inf.rollout.side_effect = RuntimeError("boom")
        self.ex._goto_inferencer = fake_inf

        result = self.ex._run_goto(goal, ep_id=1, gi=0)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_tag"], "error")


class ExecutorSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = SceneManager()
        self.bus = EventBus()
        self.ex = Executor(scene_manager=self.scene, bus=self.bus, render_video=False)

    def test_search_target_not_in_scene_emits_info_event(self) -> None:
        self.scene._scene_cfg = {"objects": [_obj("red", "ball")], "target_index": 0}
        goal = SubGoal(skill="search", color="green", shape="cone")
        result = self.ex._run_search_stub(goal, ep_id=1, gi=0)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_tag"], "target_not_in_scene")
        types = [e["type"] for e in self.bus.get_events()]
        self.assertIn("search_info", types)

    def test_search_no_scene(self) -> None:
        goal = SubGoal(skill="search", color="red", shape="ball")
        result = self.ex._run_search_stub(goal, ep_id=1, gi=0)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_tag"], "no_scene")


class ExecutorExecuteDispatchTest(unittest.TestCase):
    """Covers Executor.execute()'s goal-type dispatch + event emission,
    with each skill runner mocked out (dispatch logic itself is what's
    under test here, not the rollouts)."""

    def setUp(self) -> None:
        self.scene = SceneManager()
        self.scene._scene_cfg = {"objects": [_obj("red", "ball")], "target_index": 0}
        self.bus = EventBus()
        self.ex = Executor(scene_manager=self.scene, bus=self.bus, render_video=False)

    def test_unknown_skill_reports_unknown_skill(self) -> None:
        goal = SubGoal(skill="teleport")
        results = self.ex.execute([goal])
        self.assertEqual(results[0]["failure_tag"], "unknown_skill")
        self.assertEqual(goal.status, "failed")

    def test_execute_emits_episode_start_and_done(self) -> None:
        with mock.patch.object(Executor, "_run_goto", return_value={"success": True, "failure_tag": "success"}):
            goal = SubGoal(skill="goto", color="red", shape="ball")
            self.ex.execute([goal])
        types = [e["type"] for e in self.bus.get_events()]
        self.assertEqual(types[0], "episode_start")
        self.assertIn("goal_start", types)
        self.assertIn("goal_done", types)
        self.assertEqual(types[-1], "episode_done")

    def test_execute_marks_goal_status_done_or_failed(self) -> None:
        with mock.patch.object(Executor, "_run_maneuver", return_value={"success": False, "failure_tag": "fall"}):
            goal = SubGoal(skill="maneuver", color="red", shape="ball", direction="left")
            self.ex.execute([goal])
        self.assertEqual(goal.status, "failed")
        self.assertEqual(goal.result["failure_tag"], "fall")

    def test_execute_increments_episode_id_across_calls(self) -> None:
        with mock.patch.object(Executor, "_run_goto", return_value={"success": True}):
            self.ex.execute([SubGoal(skill="goto", color="red", shape="ball")])
            self.ex.execute([SubGoal(skill="goto", color="red", shape="ball")])
        eps = [e["ep"] for e in self.bus.get_events() if e["type"] == "episode_start"]
        self.assertEqual(eps, [1, 2])

    def test_execute_multiple_goals_reports_n_success(self) -> None:
        with mock.patch.object(Executor, "_run_goto", return_value={"success": True}), \
             mock.patch.object(Executor, "_run_search_stub", return_value={"success": False, "failure_tag": "x"}):
            goals = [SubGoal(skill="goto", color="red", shape="ball"),
                     SubGoal(skill="search", color="red", shape="ball")]
            self.ex.execute(goals)
        done_events = [e for e in self.bus.get_events() if e["type"] == "episode_done"]
        self.assertEqual(done_events[-1]["n_success"], 1)
        self.assertEqual(done_events[-1]["n_total"], 2)


class ExecutorLazyInferencerCacheTest(unittest.TestCase):
    def test_maneuver_inferencer_is_cached_across_calls(self) -> None:
        scene = SceneManager()
        ex = Executor(scene_manager=scene, bus=EventBus(), render_video=False)
        with mock.patch("code.apps.repl.executor.ManeuverInferencer") as MockMI:
            first = ex._get_maneuver_inferencer()
            second = ex._get_maneuver_inferencer()
            self.assertIs(first, second)
            self.assertEqual(MockMI.call_count, 1)


if __name__ == "__main__":
    unittest.main()
