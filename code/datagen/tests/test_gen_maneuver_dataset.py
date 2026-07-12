"""Unit tests for code.datagen.gen_maneuver_dataset (RF-1).

Covers the DART maneuver rollout (run_maneuver_episode, real mujoco but
tiny hard_maxsteps), constants, and the CLI shim.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from code.datagen.gen_maneuver_dataset import (
    FALL_HEIGHT,
    FPS,
    HOLD_STEPS,
    PROPRIO_DIM,
    run_maneuver_episode,
)
from code.maneuver_scene import derive_rng, sample_maneuver_scene
from code.teacher import WBCTeacher

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _subprocess_env() -> dict:
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    existing_pp = env.get("PYTHONPATH", "")
    root = str(_REPO_ROOT)
    env["PYTHONPATH"] = root if not existing_pp else f"{root}:{existing_pp}"
    return env


class ConstantsTest(unittest.TestCase):
    def test_values(self) -> None:
        self.assertEqual(FPS, 50)
        self.assertEqual(FALL_HEIGHT, 0.50)
        self.assertEqual(HOLD_STEPS, 5)
        self.assertEqual(PROPRIO_DIM, 55)


class RunManeuverEpisodeTest(unittest.TestCase):
    def test_short_episode_rows_well_formed(self) -> None:
        teacher = WBCTeacher()
        rng = derive_rng(base_seed=3, episode_idx=0)
        scene_cfg = sample_maneuver_scene(rng)

        result = run_maneuver_episode(
            teacher=teacher, scene_cfg=scene_cfg, episode_idx=0,
            global_frame_offset=0, noise_sigma=0.07, hard_maxsteps=25,
            rng_noise=np.random.default_rng(11),
        )

        self.assertIsNotNone(result)
        rows = result["rows"]
        self.assertGreater(len(rows), 0)
        first = rows[0]
        self.assertEqual(len(first["proprio"]), 55)
        self.assertEqual(len(first["action"]), 15)
        self.assertEqual(len(first["phase"]), 2)
        for key in ("subgoal_index", "target_heading", "heading_err",
                    "cos_target", "sin_target", "landmark_passed"):
            self.assertIn(key, first)
        self.assertIn("landmark_passed", result)
        self.assertIn("final_state", result)
        self.assertIn("success", result)

    def test_global_frame_offset_and_episode_idx(self) -> None:
        teacher = WBCTeacher()
        rng = derive_rng(base_seed=4, episode_idx=1)
        scene_cfg = sample_maneuver_scene(rng)
        result = run_maneuver_episode(
            teacher=teacher, scene_cfg=scene_cfg, episode_idx=9,
            global_frame_offset=500, noise_sigma=0.0, hard_maxsteps=10,
            rng_noise=np.random.default_rng(0),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["rows"][0]["index"], 500)
        self.assertEqual(result["rows"][0]["episode_index"], 9)

    def test_default_rng_when_none(self) -> None:
        teacher = WBCTeacher()
        rng = derive_rng(base_seed=5, episode_idx=0)
        scene_cfg = sample_maneuver_scene(rng)
        result = run_maneuver_episode(
            teacher=teacher, scene_cfg=scene_cfg, episode_idx=0,
            global_frame_offset=0, noise_sigma=0.05, hard_maxsteps=10,
            rng_noise=None,
        )
        self.assertIsNotNone(result)


class CliShimTest(unittest.TestCase):
    def test_generate_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "code/gen_maneuver_dataset.py", "generate", "--help"],
            capture_output=True, text=True, timeout=60,
            cwd=str(_REPO_ROOT), env=_subprocess_env(),
        )
        self.assertEqual(result.returncode, 0)
        for flag in ("--seed", "--num-episodes", "--noise", "--maxsteps", "--out"):
            self.assertIn(flag, result.stdout)

    def test_bare_invocation_exits_nonzero(self) -> None:
        result = subprocess.run(
            [sys.executable, "code/gen_maneuver_dataset.py"],
            capture_output=True, text=True, timeout=60,
            cwd=str(_REPO_ROOT), env=_subprocess_env(),
        )
        self.assertNotEqual(result.returncode, 0)

    def test_generate_end_to_end_tiny(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "maneuver_out")
            result = subprocess.run(
                [sys.executable, "code/gen_maneuver_dataset.py", "generate",
                 "--seed", "0", "--num-episodes", "1", "--noise", "0.07",
                 "--maxsteps", "20", "--out", out],
                capture_output=True, text=True, timeout=120,
                cwd=str(_REPO_ROOT), env=_subprocess_env(),
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertTrue((Path(out) / "meta" / "info.json").exists())


if __name__ == "__main__":
    unittest.main()
