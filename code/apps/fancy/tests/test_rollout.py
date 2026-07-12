"""Structural regression tests for code.apps.fancy.rollout.run_fancy_rollout.

The function itself drives a full MuJoCo closed-loop simulation (checkpoint
load, physics stepping, classical grounding) — that behavior is exercised
by the fancy-demo smoke test / eval gates, not stdlib unit tests. This file
only pins the public call contract (signature/defaults) that the rest of
this package (multi_goal.py, live.py, cli.py) depends on, so an accidental
signature change is caught cheaply.
"""

from __future__ import annotations

import inspect
import unittest

from code.apps.fancy.rollout import run_fancy_rollout


class RunFancyRolloutSignatureTest(unittest.TestCase):
    def test_required_positional_params_present(self) -> None:
        sig = inspect.signature(run_fancy_rollout)
        for name in ("inf", "scene_cfg", "prompt"):
            self.assertIn(name, sig.parameters)
            self.assertEqual(sig.parameters[name].default, inspect.Parameter.empty)

    def test_optional_params_have_expected_defaults(self) -> None:
        sig = inspect.signature(run_fancy_rollout)
        p = sig.parameters
        self.assertTrue(p["render_video"].default)
        self.assertIsNone(p["video_path"].default)
        self.assertEqual(p["goal_idx"].default, 0)
        self.assertEqual(p["n_goals"].default, 1)
        self.assertIsNone(p["path_trail_in"].default)
        self.assertIsNone(p["completed_targets"].default)
        self.assertIsNone(p["resume_ctx"].default)
        self.assertFalse(p["keep_alive"].default)

    def test_maxsteps_default_matches_constant(self) -> None:
        from code.apps.fancy.constants import MAXSTEPS_FANCY
        sig = inspect.signature(run_fancy_rollout)
        self.assertEqual(sig.parameters["maxsteps"].default, MAXSTEPS_FANCY)


if __name__ == "__main__":
    unittest.main()
