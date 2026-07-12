"""Cheap real-render integration smoke for the closed-loop Inferencer.

Exercises the *actual* split (constants/helpers/gt_goal/io/goal_pipeline/
rollout_state/rollout_step/inferencer files talking to real MuJoCo + a real
checkpoint) end to end, at a tiny step budget with gt goals (no render), to
catch wiring bugs pure unit tests (with stub state) can't see -- e.g. an
import cycle, a wrong attribute name on the shared state object, or a
mismatched function signature across the split files.

Skips gracefully (not a failure) if EGL/MuJoCo or the checkpoint isn't
available in this environment, per docs/refactor_plan.md's "skip gracefully
if EGL unavailable" guidance (mirrors code/eval/tests/test_rollout_integration_smoke.py).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_GOTO_CKPT = _REPO / 'checkpoint' / 'goto_best.pt'


def _mujoco_egl_available() -> bool:
    try:
        import mujoco
        mujoco.GLContext(64, 64)
        return True
    except Exception:
        return False


_EGL_OK = _mujoco_egl_available()


@unittest.skipUnless(_EGL_OK, "EGL/MuJoCo GL context unavailable in this environment")
@unittest.skipUnless(_GOTO_CKPT.exists(), f"checkpoint not found: {_GOTO_CKPT}")
class TestInferencerRolloutSmoke(unittest.TestCase):
    """A handful of GT-goal control steps of the real closed-loop Inferencer."""

    def test_short_gt_goal_rollout_runs_without_raising(self):
        from code.runtime.inferencer import Inferencer
        from code.sim.scene import sample_scene, derive_rng

        inf = Inferencer(
            checkpoint_path=str(_GOTO_CKPT), arch='A', device='cpu',
            goal_source='gt', vel_source='predicted', verbose=False,
        )
        rng = derive_rng(999, 0)
        scene_cfg = sample_scene(rng, difficulty='easy')

        result = inf.rollout(
            scene_cfg=scene_cfg, instruction=scene_cfg['instruction'],
            maxsteps=8, render_video=False,
        )

        self.assertLessEqual(result.steps, 8)
        self.assertIsInstance(result.fell, bool)
        self.assertIsInstance(result.upright, bool)
        self.assertIn(result.failure_tag,
                       ('success', 'fall', 'didnt-reach', 'lost-target', 'wrong-object'))
        self.assertEqual(result.goal_source, 'gt')

    def test_old_path_alias_produces_an_equivalent_inferencer(self):
        """`code.inferencer.Inferencer` (the sys.modules-aliased old path)
        must be usable identically to `code.runtime.inferencer.Inferencer`."""
        from code.inferencer import Inferencer as OldPathInferencer
        from code.sim.scene import sample_scene, derive_rng

        inf = OldPathInferencer(
            checkpoint_path=str(_GOTO_CKPT), arch='A', device='cpu',
            goal_source='gt', vel_source='predicted', verbose=False,
        )
        rng = derive_rng(999, 0)
        scene_cfg = sample_scene(rng, difficulty='easy')
        result = inf.rollout(scene_cfg=scene_cfg, instruction=scene_cfg['instruction'],
                             maxsteps=5, render_video=False)
        self.assertLessEqual(result.steps, 5)


if __name__ == "__main__":
    unittest.main()
