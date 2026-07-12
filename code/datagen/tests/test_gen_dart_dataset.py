"""Unit tests for code.datagen.gen_dart_dataset (RF-1): the CLI aggregator.

Confirms constants, the re-exported (GaitPhaseTracker, build_proprio)
old-path-compat surface, and that the CLI shim's argparse subcommands
(generate/add-phase/combine) still work end to end.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from code.datagen.gen_dart_dataset import (
    GaitPhaseTracker,
    PROPRIO_DIM,
    add_phase_to_clean_dataset,
    build_proprio,
    combine_datasets,
    run_dart_episode,
)
from code.datagen.gen_dart_rollout import FPS

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _subprocess_env() -> dict:
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    existing_pp = env.get("PYTHONPATH", "")
    root = str(_REPO_ROOT)
    env["PYTHONPATH"] = root if not existing_pp else f"{root}:{existing_pp}"
    return env


class ReExportsTest(unittest.TestCase):
    def test_gait_phase_tracker_reexported(self) -> None:
        tracker = GaitPhaseTracker()
        self.assertTrue(hasattr(tracker, "update"))

    def test_build_proprio_reexported(self) -> None:
        self.assertTrue(callable(build_proprio))

    def test_run_dart_episode_reexported(self) -> None:
        self.assertTrue(callable(run_dart_episode))

    def test_combine_helpers_reexported(self) -> None:
        self.assertTrue(callable(add_phase_to_clean_dataset))
        self.assertTrue(callable(combine_datasets))

    def test_constants(self) -> None:
        self.assertEqual(PROPRIO_DIM, 55)
        self.assertEqual(FPS, 50)


class CliSubcommandsTest(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "code/gen_dart_dataset.py", *args],
            capture_output=True, text=True, timeout=60,
            cwd=str(_REPO_ROOT), env=_subprocess_env(),
        )

    def test_bare_invocation_prints_help_and_exits_nonzero(self) -> None:
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage", (result.stdout + result.stderr).lower())

    def test_top_level_help_lists_subcommands(self) -> None:
        result = self._run("--help")
        self.assertEqual(result.returncode, 0)
        for sub in ("generate", "add-phase", "combine"):
            self.assertIn(sub, result.stdout)

    def test_generate_help(self) -> None:
        result = self._run("generate", "--help")
        self.assertEqual(result.returncode, 0)
        for flag in ("--difficulty", "--seed", "--num-episodes", "--noise", "--maxsteps", "--out"):
            self.assertIn(flag, result.stdout)

    def test_add_phase_help(self) -> None:
        result = self._run("add-phase", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--in-dir", result.stdout)
        self.assertIn("--out-dir", result.stdout)

    def test_combine_help(self) -> None:
        result = self._run("combine", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--clean-dir", result.stdout)
        self.assertIn("--dart-dir", result.stdout)
        self.assertIn("--out", result.stdout)

    def test_generate_missing_required_args_fails(self) -> None:
        result = self._run("generate")
        self.assertNotEqual(result.returncode, 0)


class GenerateEndToEndTest(unittest.TestCase):
    """One tiny real `generate` run through the CLI shim (writes to a tempdir)."""

    def test_generate_writes_expected_layout(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "dart_out")
            result = subprocess.run(
                [sys.executable, "code/gen_dart_dataset.py", "generate",
                 "--difficulty", "easy", "--seed", "5", "--num-episodes", "1",
                 "--noise", "0.07", "--maxsteps", "20", "--out", out],
                capture_output=True, text=True, timeout=120,
                cwd=str(_REPO_ROOT), env=_subprocess_env(),
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertTrue((Path(out) / "meta" / "info.json").exists())
            self.assertTrue((Path(out) / "data" / "chunk-000" / "episode_000000.parquet").exists())


if __name__ == "__main__":
    unittest.main()
