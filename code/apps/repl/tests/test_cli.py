"""Unit tests for code.apps.repl.cli.main()'s argument-driven dispatch
(smoke / web / terminal). Heavy pieces (`_smoke_test`, `_start_web_ui`,
`_terminal_repl`, real Inferencer construction) are mocked out — this file
covers the CLI wiring/dispatch logic only.
"""

from __future__ import annotations

import sys
import unittest
from unittest import mock

from code.apps.repl import cli


class MainDispatchTest(unittest.TestCase):
    def _run_main_with_argv(self, argv: list[str]) -> None:
        with mock.patch.object(sys, "argv", ["prog"] + argv):
            cli.main()

    def test_smoke_flag_calls_smoke_test_and_returns(self) -> None:
        with mock.patch.object(cli, "_smoke_test") as mock_smoke, \
             mock.patch.object(cli, "SceneManager") as mock_sm:
            self._run_main_with_argv(["--smoke", "--no-render"])
            mock_smoke.assert_called_once()
            # Smoke mode must return before touching the REPL/Web scaffolding.
            mock_sm.assert_not_called()

    def test_smoke_forwards_maxsteps_and_device(self) -> None:
        with mock.patch.object(cli, "_smoke_test") as mock_smoke:
            self._run_main_with_argv([
                "--smoke", "--no-render", "--device", "cpu",
                "--maxsteps-goto", "123", "--maxsteps-maneuver", "45",
            ])
            kwargs = mock_smoke.call_args.kwargs
            self.assertEqual(kwargs["device"], "cpu")
            self.assertEqual(kwargs["maxsteps_goto"], 123)
            self.assertEqual(kwargs["maxsteps_maneuver"], 45)
            self.assertFalse(kwargs["render_video"])

    def test_web_flag_starts_web_ui_not_terminal_repl(self) -> None:
        with mock.patch.object(cli, "_start_web_ui") as mock_web, \
             mock.patch.object(cli, "_terminal_repl") as mock_term, \
             mock.patch("time.sleep", side_effect=KeyboardInterrupt):
            self._run_main_with_argv(["--web", "--no-render"])
            mock_web.assert_called_once()
            mock_term.assert_not_called()

    def test_default_mode_runs_terminal_repl(self) -> None:
        with mock.patch.object(cli, "_start_web_ui") as mock_web, \
             mock.patch.object(cli, "_terminal_repl") as mock_term:
            self._run_main_with_argv(["--no-render"])
            mock_term.assert_called_once()
            mock_web.assert_not_called()

    def test_difficulty_choice_validated(self) -> None:
        with self.assertRaises(SystemExit):
            self._run_main_with_argv(["--difficulty", "impossible"])


if __name__ == "__main__":
    unittest.main()
