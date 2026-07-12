"""Unit tests for code.runtime.rollout_state.RolloutState's dataclass
invariants (zero-arg defaults, mutable-default independence)."""

from __future__ import annotations

import unittest

from code.runtime.rollout_state import RolloutState
from code.runtime.constants import PROPRIO_DIM


class TestRolloutStateDefaults(unittest.TestCase):
    def test_defaults_construct_without_args(self):
        s = RolloutState()
        self.assertIsNone(s.early_result)
        self.assertIsNone(s.target_xy)
        self.assertEqual(s.target_color, "")
        self.assertEqual(s.target_shape, "")
        self.assertEqual(s.stop_r, 0.6)
        self.assertEqual(s.nj, 0)
        self.assertEqual(s.frames_ego, [])
        self.assertEqual(s.frames_tp, [])
        self.assertFalse(s.use_phase)
        self.assertEqual(s.eff_proprio_dim, PROPRIO_DIM)
        self.assertFalse(s.use_residual)
        self.assertFalse(s.inject_gt_vel)
        self.assertFalse(s.inject_cached)
        self.assertFalse(s.use_learned_goal)
        self.assertFalse(s.use_gt_goal)
        self.assertFalse(s.need_learned_render)
        self.assertEqual(s.te_buffer, [])
        self.assertIsNone(s.goal_pipeline)
        self.assertEqual(s.all_target_dofs, [])
        self.assertEqual(s.step_times, [])
        self.assertEqual(s.hold_counter, 0)
        self.assertFalse(s.fell)
        self.assertEqual(s.steps_done, 0)
        self.assertEqual(s.stall_recovery_remaining, 0)
        self.assertEqual(s.stall_cooldown_remaining, 0)
        self.assertEqual(s.stall_trigger_count, 0)
        self.assertFalse(s.stall_is_maneuver)
        self.assertEqual(s.cur_vx_cmd, 0.0)

    def test_mutable_default_fields_are_independent_per_instance(self):
        a = RolloutState()
        b = RolloutState()
        a.frames_ego.append("frame")
        a.all_target_dofs.append([1, 2, 3])
        a.te_buffer.append((0, None, None))
        self.assertEqual(b.frames_ego, [])
        self.assertEqual(b.all_target_dofs, [])
        self.assertEqual(b.te_buffer, [])


if __name__ == "__main__":
    unittest.main()
