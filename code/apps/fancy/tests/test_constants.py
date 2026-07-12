"""Unit tests for code.apps.fancy.constants: VF-1 env-var toggle logic
(`_fancy_env_flag` / `_fancy_feat` / FANCY_PLAIN override) + basic constant
invariants.
"""

from __future__ import annotations

import importlib
import os
import unittest
from unittest import mock

from code.apps.fancy import constants as C


class FancyEnvFlagTest(unittest.TestCase):
    def test_default_true_when_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SOME_UNSET_FLAG", None)
            self.assertTrue(C._fancy_env_flag("SOME_UNSET_FLAG", "1"))

    def test_default_false_when_unset_and_default_0(self) -> None:
        os.environ.pop("SOME_UNSET_FLAG2", None)
        self.assertFalse(C._fancy_env_flag("SOME_UNSET_FLAG2", "0"))

    def test_explicit_1_is_true(self) -> None:
        with mock.patch.dict(os.environ, {"MY_FLAG": "1"}):
            self.assertTrue(C._fancy_env_flag("MY_FLAG", "0"))

    def test_explicit_0_is_false(self) -> None:
        with mock.patch.dict(os.environ, {"MY_FLAG": "0"}):
            self.assertFalse(C._fancy_env_flag("MY_FLAG", "1"))

    def test_whitespace_is_stripped(self) -> None:
        with mock.patch.dict(os.environ, {"MY_FLAG": "  1  "}):
            self.assertTrue(C._fancy_env_flag("MY_FLAG", "0"))

    def test_arbitrary_string_is_false(self) -> None:
        with mock.patch.dict(os.environ, {"MY_FLAG": "yes"}):
            self.assertFalse(C._fancy_env_flag("MY_FLAG", "1"))


class FancyFeatTest(unittest.TestCase):
    def test_feat_default_on(self) -> None:
        with mock.patch.object(C, "FANCY_PLAIN", False):
            os.environ.pop("FANCY_SOME_FEATURE", None)
            self.assertTrue(C._fancy_feat("SOME_FEATURE"))

    def test_feat_can_be_individually_disabled(self) -> None:
        with mock.patch.object(C, "FANCY_PLAIN", False), \
             mock.patch.dict(os.environ, {"FANCY_SOME_FEATURE": "0"}):
            self.assertFalse(C._fancy_feat("SOME_FEATURE"))

    def test_fancy_plain_forces_off_regardless_of_own_flag(self) -> None:
        with mock.patch.object(C, "FANCY_PLAIN", True), \
             mock.patch.dict(os.environ, {"FANCY_SOME_FEATURE": "1"}):
            self.assertFalse(C._fancy_feat("SOME_FEATURE"))


class ConstantInvariantsTest(unittest.TestCase):
    def test_reliable_colors_subset_of_full_palette(self) -> None:
        full_colors = {"red", "yellow", "blue", "green", "orange", "purple", "cyan"}
        self.assertTrue(set(C.RELIABLE_COLORS).issubset(full_colors))
        self.assertNotIn("cyan", C.RELIABLE_COLORS)
        self.assertNotIn("blue", C.RELIABLE_COLORS)

    def test_state_constants_all_distinct(self) -> None:
        states = [C.STATE_IDLE, C.STATE_SEARCHING, C.STATE_LOCATED,
                  C.STATE_MOVING, C.STATE_REACHED, C.STATE_FAILED]
        self.assertEqual(len(states), len(set(states)))

    def test_stream_width_is_bev_plus_ego(self) -> None:
        self.assertEqual(C.STREAM_W, C.BEV_W + C.EGO_W)

    def test_dist_bounds_ordered(self) -> None:
        self.assertLess(C.DIST_MIN_LONG, C.DIST_MAX_LONG)

    def test_skill_stages_five_items(self) -> None:
        self.assertEqual(len(C.SKILL_STAGES), 5)
        self.assertEqual(C.SKILL_STAGES[0], "SCAN")
        self.assertEqual(C.SKILL_STAGES[-1], "REACH")


if __name__ == "__main__":
    unittest.main()
