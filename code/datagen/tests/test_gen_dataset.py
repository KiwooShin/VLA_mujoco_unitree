"""Unit tests for code.datagen.gen_dataset + gen_dataset_rollout (RF-1).

build_proprio is tested against a small synthetic MuJoCo model (no arena.py
dependency needed -- just enough qpos/qvel to exercise the exact slices).
check_determinism is exercised as a small real-mujoco integration smoke
(n_check=1, difficulty='easy', no rendering -- a few hundred ms).
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _subprocess_env() -> dict:
    """Environment for CLI-shim subprocess tests (repo root importable)."""
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    existing_pp = env.get("PYTHONPATH", "")
    root = str(_REPO_ROOT)
    env["PYTHONPATH"] = root if not existing_pp else f"{root}:{existing_pp}"
    return env

from code.datagen.gen_dataset_rollout import (
    FALL_HEIGHT,
    FPS,
    HOLD_STEPS,
    MAXSTEPS,
    PROPRIO_DIM,
    SETTLE_STEPS,
    build_proprio,
    check_determinism,
)


def _make_test_model(n_hinges: int = 15) -> mujoco.MjModel:
    """Builds a minimal free-joint + N-hinge-chain MuJoCo model.

    Gives exactly nq=22 (7 free + 15 hinge) and nv=21 (6 free + 15 hinge),
    matching build_proprio's qpos[7:22]/qvel[6:21] slices without needing
    the real arena.
    """
    body_open, body_close = "", ""
    for i in range(n_hinges):
        body_open += (
            f'<body name="l{i}" pos="0 0 -0.05">'
            f'<joint name="j{i}" type="hinge" axis="0 1 0"/>'
            f'<geom type="capsule" fromto="0 0 0 0 0 -0.05" size="0.02"/>'
        )
        body_close += "</body>"
    xml = (
        '<mujoco><worldbody><body name="base" pos="0 0 1">'
        '<freejoint/><geom name="pelvis" type="sphere" size="0.1"/>'
        f"{body_open}{body_close}</body></worldbody></mujoco>"
    )
    return mujoco.MjModel.from_xml_string(xml)


class BuildProprioTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _make_test_model()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def test_shape_and_dtype(self) -> None:
        prev_action = np.zeros(15, dtype=np.float32)
        p = build_proprio(self.data, prev_action)
        self.assertEqual(p.shape, (55,))
        self.assertEqual(p.dtype, np.float32)

    def test_layout_matches_qpos_qvel_slices(self) -> None:
        self.data.qpos[7:22] = np.arange(15, dtype=np.float64) * 0.1
        self.data.qvel[6:21] = np.arange(15, dtype=np.float64) * -0.1
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[3:6] = [0.01, 0.02, 0.03]
        self.data.qvel[0:3] = [1.0, 2.0, 3.0]
        prev_action = np.arange(15, dtype=np.float32) + 100.0
        p = build_proprio(self.data, prev_action)
        np.testing.assert_allclose(p[0:15], self.data.qpos[7:22], atol=1e-6)
        np.testing.assert_allclose(p[15:30], self.data.qvel[6:21], atol=1e-6)
        np.testing.assert_allclose(p[30:34], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(p[34:37], [0.01, 0.02, 0.03], atol=1e-6)
        np.testing.assert_allclose(p[37:40], [1.0, 2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(p[40:55], prev_action, atol=1e-6)

    def test_does_not_mutate_inputs(self) -> None:
        prev_action = np.ones(15, dtype=np.float32) * 7.0
        before = prev_action.copy()
        build_proprio(self.data, prev_action)
        np.testing.assert_array_equal(prev_action, before)

    def test_prev_action_dtype_cast(self) -> None:
        # int input must still produce a float32 output vector.
        prev_action = np.zeros(15, dtype=np.int64)
        p = build_proprio(self.data, prev_action)
        self.assertEqual(p.dtype, np.float32)


class ConstantsTest(unittest.TestCase):
    def test_fps_is_fifty(self) -> None:
        self.assertEqual(FPS, 50)

    def test_maxsteps_dict(self) -> None:
        self.assertEqual(MAXSTEPS["easy"], 600)
        self.assertEqual(MAXSTEPS["demo"], 1400)

    def test_other_constants(self) -> None:
        self.assertEqual(SETTLE_STEPS, 80)
        self.assertEqual(FALL_HEIGHT, 0.50)
        self.assertEqual(HOLD_STEPS, 5)
        self.assertEqual(PROPRIO_DIM, 55)


class CheckDeterminismSmokeTest(unittest.TestCase):
    """Real-mujoco integration smoke: cheap (no rendering, 1 short episode)."""

    def test_determinism_easy_seed(self) -> None:
        ok = check_determinism("easy", seed=321, n_check=1)
        self.assertTrue(ok)


class CliShimTest(unittest.TestCase):
    """The old-path entry shim must still expose the same argparse CLI."""

    def test_help_runs_and_lists_expected_flags(self) -> None:
        result = subprocess.run(
            [sys.executable, "code/gen_dataset.py", "--help"],
            capture_output=True, text=True, timeout=60,
            cwd=str(_REPO_ROOT), env=_subprocess_env(),
        )
        self.assertEqual(result.returncode, 0)
        for flag in ("--difficulty", "--seed", "--num-episodes", "--noise", "--out"):
            self.assertIn(flag, result.stdout)

    def test_missing_required_args_exits_nonzero(self) -> None:
        result = subprocess.run(
            [sys.executable, "code/gen_dataset.py"],
            capture_output=True, text=True, timeout=60,
            cwd=str(_REPO_ROOT), env=_subprocess_env(),
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
