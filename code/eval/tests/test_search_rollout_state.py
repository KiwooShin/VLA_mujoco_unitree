"""Unit tests for code.eval.search_rollout_state / search_rollout_step.

Covers: ``_RolloutSetup``'s field defaults (the bundle the split rollout
loop threads state through), and ``_lock_drop_and_rescan`` -- the one
pure-logic state-mutation helper extracted from the per-step loop -- using a
plain ``_RolloutSetup`` instance as the stand-in "mutable state" object
(no MuJoCo/GPU needed for either).
"""

from __future__ import annotations

import unittest

import numpy as np

from code.eval.search_rollout_state import _RolloutSetup
from code.eval.search_rollout_step import _lock_drop_and_rescan


class TestRolloutSetupDefaults(unittest.TestCase):
    """Sanity on the dataclass's zero-arg defaults (used before _setup_search_rollout
    populates it, and directly by tests that stand in for a real rollout)."""

    def test_defaults_construct_without_args(self):
        s = _RolloutSetup()
        self.assertIsNone(s.early_result)
        self.assertEqual(s.objects, [])
        self.assertEqual(s.target_idx, 0)
        self.assertTrue(s._scan_active)
        self.assertEqual(s.last_grounding_step, -999)
        self.assertEqual(s.scan_steps, 0)
        self.assertFalse(s.spotted)
        self.assertFalse(s.fell)
        self.assertEqual(s.steps_done, 0)
        self.assertEqual(s.step_times, [])
        self.assertEqual(s._all_target_dofs, [])
        self.assertEqual(s.frames_ego, [])
        self.assertEqual(s.frames_tp, [])

    def test_mutable_default_fields_are_independent_per_instance(self):
        """dataclass field(default_factory=list) must not alias across instances."""
        a = _RolloutSetup()
        b = _RolloutSetup()
        a.objects.append({'x': 1})
        self.assertEqual(a.objects, [{'x': 1}])
        self.assertEqual(b.objects, [])

    def test_scan_constants_defaults(self):
        s = _RolloutSetup()
        self.assertEqual(s.SCAN_RATE, 0.6)
        self.assertEqual(s._GOAL_EMA_ALPHA, 0.4)
        self.assertEqual(s.HOLD_GOAL_HORIZON, 100)


class _FakeLockGate:
    """Stand-in for code.lock_mgmt.LockGate: records force_drop() calls."""
    def __init__(self):
        self.force_drop_calls = 0

    def force_drop(self):
        self.force_drop_calls += 1


class TestLockDropAndRescan(unittest.TestCase):
    """_lock_drop_and_rescan mutates a _RolloutSetup in place; no MuJoCo needed."""

    def _make_setup(self) -> _RolloutSetup:
        s = _RolloutSetup()
        s._lock_gate = _FakeLockGate()
        s.SCAN_RATE = 0.6
        s._goal_ema = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        s._last_known_goal = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        s._frames_since_det = 42
        s._scan_active = False
        s._using_rescan_sched = False
        s._rescan_sched = None
        s.cached_goal_vec = np.array([0.1, 0.9, 0.1], dtype=np.float32)
        s._avoid_bias_wz = 0.37
        return s

    def test_resets_goal_and_lock_state(self):
        s = self._make_setup()
        _lock_drop_and_rescan(s)

        self.assertEqual(s._lock_gate.force_drop_calls, 1)
        self.assertIsNone(s._goal_ema)
        self.assertIsNone(s._last_known_goal)
        self.assertEqual(s._frames_since_det, 0)
        self.assertTrue(s._scan_active)
        self.assertTrue(s._using_rescan_sched)
        self.assertEqual(s._avoid_bias_wz, 0.0)

    def test_rearms_a_fresh_rescan_schedule(self):
        s = self._make_setup()
        old_sched = s._rescan_sched
        _lock_drop_and_rescan(s)
        self.assertIsNotNone(s._rescan_sched)
        self.assertIsNot(s._rescan_sched, old_sched)

    def test_resets_cached_goal_vec_to_default(self):
        s = self._make_setup()
        _lock_drop_and_rescan(s)
        np.testing.assert_array_equal(s.cached_goal_vec, np.array([2.0, 1.0, 0.0], dtype=np.float32))

    def test_idempotent_call_pattern(self):
        """Calling twice in a row (e.g. two watchdogs firing back-to-back)
        must not raise and must leave the same reset state."""
        s = self._make_setup()
        _lock_drop_and_rescan(s)
        first_sched = s._rescan_sched
        _lock_drop_and_rescan(s)
        self.assertIsNot(s._rescan_sched, first_sched)
        self.assertEqual(s._lock_gate.force_drop_calls, 2)


if __name__ == '__main__':
    unittest.main()
