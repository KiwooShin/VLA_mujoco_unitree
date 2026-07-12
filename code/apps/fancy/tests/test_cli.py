"""Unit tests for code.apps.fancy.cli.main()'s argument-driven dispatch
(smoke / web / terminal). Heavy pieces (`run_smoke`, `_start_fancy_web_ui`,
`_terminal_loop`, real Inferencer construction) are mocked out — this file
covers the CLI wiring/dispatch logic only."""

from __future__ import annotations

import sys
import unittest
from unittest import mock

from code.apps.fancy import cli


class MainDispatchTest(unittest.TestCase):
    def _run_main_with_argv(self, argv: list[str]) -> None:
        with mock.patch.object(sys, "argv", ["prog"] + argv):
            cli.main()

    def test_smoke_flag_calls_run_smoke_and_returns(self) -> None:
        with mock.patch.object(cli, "run_smoke") as mock_smoke, \
             mock.patch.object(cli, "FancySceneManager") as mock_sm:
            self._run_main_with_argv(["--smoke", "--no-render"])
            mock_smoke.assert_called_once()
            mock_sm.assert_not_called()

    def test_smoke_forwards_args(self) -> None:
        with mock.patch.object(cli, "run_smoke") as mock_smoke:
            self._run_main_with_argv([
                "--smoke", "--no-render", "--device", "cpu",
                "--maxsteps", "555", "--n-smoke", "3",
                "--scenario-title", "Custom Title",
            ])
            kwargs = mock_smoke.call_args.kwargs
            self.assertEqual(kwargs["device"], "cpu")
            self.assertEqual(kwargs["maxsteps"], 555)
            self.assertEqual(kwargs["n_episodes"], 3)
            self.assertEqual(kwargs["scenario_title"], "Custom Title")
            self.assertFalse(kwargs["render_video"])

    def test_web_flag_dispatches_to_start_fancy_web_ui(self) -> None:
        with mock.patch("code.inferencer.Inferencer") as mock_inf, \
             mock.patch.object(cli, "FancySceneManager") as mock_sm_cls, \
             mock.patch.object(cli, "_start_fancy_web_ui") as mock_web, \
             mock.patch.object(cli, "_terminal_loop") as mock_term, \
             mock.patch("time.sleep", side_effect=KeyboardInterrupt):
            mock_sm_cls.return_value.new_scene.return_value = None
            self._run_main_with_argv(["--web", "--no-render"])
            mock_web.assert_called_once()
            mock_term.assert_not_called()

    def test_default_mode_dispatches_to_terminal_loop(self) -> None:
        with mock.patch("code.inferencer.Inferencer") as mock_inf, \
             mock.patch.object(cli, "FancySceneManager") as mock_sm_cls, \
             mock.patch.object(cli, "_start_fancy_web_ui") as mock_web, \
             mock.patch.object(cli, "_terminal_loop") as mock_term:
            mock_sm_cls.return_value.new_scene.return_value = None
            self._run_main_with_argv(["--no-render"])
            mock_term.assert_called_once()
            mock_web.assert_not_called()

    def test_invalid_int_arg_exits(self) -> None:
        with self.assertRaises(SystemExit):
            self._run_main_with_argv(["--maxsteps", "not-an-int"])


if __name__ == "__main__":
    unittest.main()
