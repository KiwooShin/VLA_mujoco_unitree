"""Unit tests for code.runtime.constants (env toggles + module constants)."""

from __future__ import annotations

import os
import unittest

from code.runtime.constants import _env_flag, FALL_HEIGHT, GROUNDING_PERIOD, PROPRIO_DIM, \
    PROPRIO_DIM_PHASE, PROPRIO_K, ACTION_SCALE, SETTLE_STEPS, HOLD_STEPS_REQUIRED, \
    STALL_RECOVERY_STEPS, STALL_COOLDOWN_STEPS, KEYFRAME_PATH


class TestEnvFlag(unittest.TestCase):
    """`_env_flag` reads an env var and returns True iff it is exactly "1"."""

    def setUp(self):
        self._sentinel = "RUNTIME_TEST_ENV_FLAG_XYZ"
        os.environ.pop(self._sentinel, None)

    def tearDown(self):
        os.environ.pop(self._sentinel, None)

    def test_unset_defaults_false(self):
        self.assertFalse(_env_flag(self._sentinel))

    def test_unset_with_custom_default_true(self):
        # default param is compared as a STRING against the raw env value;
        # with the var unset, os.environ.get returns the default itself.
        self.assertTrue(_env_flag(self._sentinel, default="1"))

    def test_set_to_1_is_true(self):
        os.environ[self._sentinel] = "1"
        self.assertTrue(_env_flag(self._sentinel))

    def test_set_to_0_is_false(self):
        os.environ[self._sentinel] = "0"
        self.assertFalse(_env_flag(self._sentinel))

    def test_set_to_arbitrary_string_is_false(self):
        os.environ[self._sentinel] = "true"
        self.assertFalse(_env_flag(self._sentinel))

    def test_whitespace_is_stripped(self):
        os.environ[self._sentinel] = " 1 "
        self.assertTrue(_env_flag(self._sentinel))


class TestConstantValues(unittest.TestCase):
    """Pins the exact numeric values moved verbatim from the pre-RF-1 file —
    a regression net against an accidental typo during the split."""

    def test_physics_and_proprio_constants(self):
        self.assertEqual(FALL_HEIGHT, 0.50)
        self.assertEqual(GROUNDING_PERIOD, 10)
        self.assertEqual(PROPRIO_K, 6)
        self.assertEqual(PROPRIO_DIM, 55)
        self.assertEqual(PROPRIO_DIM_PHASE, 57)
        self.assertEqual(ACTION_SCALE, 0.25)
        self.assertEqual(SETTLE_STEPS, 80)
        self.assertEqual(HOLD_STEPS_REQUIRED, 5)

    def test_stall_break_recovery_constants(self):
        self.assertEqual(STALL_RECOVERY_STEPS, 50)
        self.assertEqual(STALL_COOLDOWN_STEPS, 100)

    def test_keyframe_path_points_at_checkpoint_dir(self):
        self.assertTrue(KEYFRAME_PATH.endswith("checkpoint/stand_keyframe.npz"))


if __name__ == "__main__":
    unittest.main()
