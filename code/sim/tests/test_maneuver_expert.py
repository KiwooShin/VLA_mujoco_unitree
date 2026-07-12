"""Unit tests for code.sim.maneuver_expert (scripted FSM expert)."""

import math
import unittest

import numpy as np

from code.sim.maneuver_expert import (
    FORWARD_VX,
    HEADING_DONE_THR,
    MAX_WZ,
    ManeuverExpert,
    State,
    _angle_diff,
)


def _scene(landmark_xy=(3.0, 0.0), target_heading=math.pi / 2.0,
           turn_direction="left", pass_margin=0.6) -> dict:
    """Build a minimal maneuver scene_cfg for expert-only tests."""
    return {
        "landmark_xy": landmark_xy,
        "target_heading": target_heading,
        "turn_direction": turn_direction,
        "pass_margin": pass_margin,
    }


class TestAngleDiff(unittest.TestCase):
    def test_zero(self) -> None:
        self.assertEqual(_angle_diff(0.0, 0.0), 0.0)

    def test_simple_positive(self) -> None:
        self.assertAlmostEqual(_angle_diff(math.pi / 2, 0.0), math.pi / 2)

    def test_wraps_past_pi(self) -> None:
        # a - b = 3pi/2, should wrap to -pi/2
        result = _angle_diff(3 * math.pi / 2, 0.0)
        self.assertAlmostEqual(result, -math.pi / 2)

    def test_wraps_past_negative_pi(self) -> None:
        result = _angle_diff(-3 * math.pi / 2, 0.0)
        self.assertAlmostEqual(result, math.pi / 2)

    def test_result_within_bounds(self) -> None:
        for a in np.linspace(-10, 10, 41):
            for b in np.linspace(-10, 10, 5):
                d = _angle_diff(float(a), float(b))
                self.assertGreater(d, -math.pi - 1e-9)
                self.assertLessEqual(d, math.pi + 1e-9)


class TestManeuverExpertInit(unittest.TestCase):
    def test_starts_in_straight_state(self) -> None:
        expert = ManeuverExpert(_scene())
        self.assertEqual(expert.state, State.STRAIGHT)
        self.assertFalse(expert.landmark_passed)

    def test_reset_returns_to_straight(self) -> None:
        expert = ManeuverExpert(_scene(landmark_xy=(0.0, 0.0), pass_margin=0.0))
        expert.step((1.0, 0.0), 0.0)  # already past landmark -> TURN_PHASE
        self.assertEqual(expert.state, State.TURN_PHASE)
        expert.reset()
        self.assertEqual(expert.state, State.STRAIGHT)
        self.assertFalse(expert.landmark_passed)

    def test_default_pass_margin_when_absent(self) -> None:
        sc = _scene()
        sc.pop("pass_margin")
        expert = ManeuverExpert(sc)  # should default to 0.6, not raise
        self.assertEqual(expert.state, State.STRAIGHT)


class TestManeuverExpertFSMTransitions(unittest.TestCase):
    def test_straight_before_landmark(self) -> None:
        expert = ManeuverExpert(_scene(landmark_xy=(3.0, 0.0), pass_margin=0.6))
        vel, priv = expert.step((0.0, 0.0), 0.0)
        self.assertEqual(priv["fsm_state"], State.STRAIGHT)
        self.assertFalse(priv["landmark_passed"])
        self.assertEqual(vel[0], FORWARD_VX)

    def test_transitions_to_turn_phase_exactly_at_threshold(self) -> None:
        expert = ManeuverExpert(_scene(landmark_xy=(3.0, 0.0), pass_margin=0.6))
        # Not yet past: rx < 3.6
        _, priv = expert.step((3.59, 0.0), 0.0)
        self.assertEqual(priv["fsm_state"], State.STRAIGHT)
        # Exactly at threshold: rx == landmark_x + pass_margin -> passed (>=)
        _, priv = expert.step((3.60, 0.0), 0.0)
        self.assertEqual(priv["fsm_state"], State.TURN_PHASE)
        self.assertTrue(priv["landmark_passed"])

    def test_landmark_passed_is_sticky(self) -> None:
        """Once triggered, landmark_passed stays True even if robot regresses in x."""
        expert = ManeuverExpert(_scene(landmark_xy=(0.0, 0.0), pass_margin=0.0))
        expert.step((1.0, 0.0), 0.0)
        self.assertTrue(expert.landmark_passed)
        _, priv = expert.step((-5.0, 0.0), 0.0)  # regress far behind
        self.assertTrue(priv["landmark_passed"])
        self.assertEqual(priv["fsm_state"], State.TURN_PHASE)

    def test_turn_phase_holds_zero_forward_velocity(self) -> None:
        expert = ManeuverExpert(_scene(landmark_xy=(0.0, 0.0), pass_margin=0.0,
                                       target_heading=math.pi / 2))
        expert.step((1.0, 0.0), 0.0)  # trigger TURN_PHASE
        vel, priv = expert.step((1.0, 0.0), 0.0)
        self.assertEqual(priv["fsm_state"], State.TURN_PHASE)
        self.assertEqual(vel[0], 0.0)
        self.assertEqual(vel[1], 0.0)

    def test_turn_phase_transitions_to_straight2_within_threshold(self) -> None:
        expert = ManeuverExpert(_scene(landmark_xy=(0.0, 0.0), pass_margin=0.0,
                                       target_heading=math.pi / 2))
        expert.step((1.0, 0.0), 0.0)  # trigger TURN_PHASE
        # heading within HEADING_DONE_THR of target -> should exit to STRAIGHT2
        almost_there = math.pi / 2 - HEADING_DONE_THR * 0.5
        _, priv = expert.step((1.0, 0.0), almost_there)
        self.assertEqual(priv["fsm_state"], State.STRAIGHT2)

    def test_turn_phase_does_not_exit_outside_threshold(self) -> None:
        expert = ManeuverExpert(_scene(landmark_xy=(0.0, 0.0), pass_margin=0.0,
                                       target_heading=math.pi / 2))
        expert.step((1.0, 0.0), 0.0)  # trigger TURN_PHASE
        far_from_target = math.pi / 2 - HEADING_DONE_THR * 3.0
        _, priv = expert.step((1.0, 0.0), far_from_target)
        self.assertEqual(priv["fsm_state"], State.TURN_PHASE)

    def test_straight2_resumes_forward_motion(self) -> None:
        expert = ManeuverExpert(_scene(landmark_xy=(0.0, 0.0), pass_margin=0.0,
                                       target_heading=math.pi / 2))
        expert.step((1.0, 0.0), 0.0)                       # -> TURN_PHASE
        expert.step((1.0, 0.0), math.pi / 2 - 1e-4)         # -> STRAIGHT2
        vel, priv = expert.step((1.0, 0.0), math.pi / 2)
        self.assertEqual(priv["fsm_state"], State.STRAIGHT2)
        self.assertEqual(vel[0], FORWARD_VX)


class TestManeuverExpertCommandBounds(unittest.TestCase):
    def test_turn_command_clipped_to_max_wz(self) -> None:
        expert = ManeuverExpert(_scene(landmark_xy=(0.0, 0.0), pass_margin=0.0,
                                       target_heading=math.pi))
        expert.step((1.0, 0.0), 0.0)  # -> TURN_PHASE, large heading_err (~pi)
        vel, priv = expert.step((1.0, 0.0), 0.0)
        self.assertLessEqual(abs(vel[2]), MAX_WZ + 1e-4)

    def test_straight_command_clipped_to_max_wz(self) -> None:
        expert = ManeuverExpert(_scene(landmark_xy=(10.0, 0.0), pass_margin=0.6))
        # Large heading error while still in STRAIGHT: robot facing away from 0
        vel, _ = expert.step((0.0, 100.0), math.pi)
        self.assertLessEqual(abs(vel[2]), MAX_WZ + 1e-4)

    def test_vel_cmd_is_length_3_float32_array(self) -> None:
        expert = ManeuverExpert(_scene())
        vel, _ = expert.step((0.0, 0.0), 0.0)
        self.assertIsInstance(vel, np.ndarray)
        self.assertEqual(vel.shape, (3,))
        self.assertEqual(vel.dtype, np.float32)


class TestManeuverExpertPrivState(unittest.TestCase):
    def test_priv_state_keys(self) -> None:
        expert = ManeuverExpert(_scene())
        _, priv = expert.step((0.0, 0.0), 0.0)
        expected_keys = {
            "subgoal_index", "target_heading", "heading_err",
            "cos_target", "sin_target", "landmark_passed", "fsm_state",
        }
        self.assertEqual(set(priv.keys()), expected_keys)

    def test_cos_sin_target_match_target_heading(self) -> None:
        target = math.pi / 3
        expert = ManeuverExpert(_scene(target_heading=target))
        _, priv = expert.step((0.0, 0.0), 0.0)
        self.assertAlmostEqual(priv["cos_target"], math.cos(target))
        self.assertAlmostEqual(priv["sin_target"], math.sin(target))

    def test_subgoal_index_matches_fsm_state(self) -> None:
        expert = ManeuverExpert(_scene())
        _, priv = expert.step((0.0, 0.0), 0.0)
        self.assertEqual(priv["subgoal_index"], priv["fsm_state"])


class TestStateEnum(unittest.TestCase):
    def test_ordering_matches_documented_subgoal_indices(self) -> None:
        self.assertEqual(int(State.STRAIGHT), 0)
        self.assertEqual(int(State.TURN_PHASE), 1)
        self.assertEqual(int(State.STRAIGHT2), 2)


if __name__ == "__main__":
    unittest.main()
