"""Cheap real-render integration smokes for the search and maneuver rollouts.

These exercise the *actual* split (state/step/rollout files talking to real
MuJoCo + a real checkpoint) end to end, at a tiny step budget, to catch
wiring bugs that pure unit tests (with stub state) can't see -- e.g. an
import cycle, a wrong attribute name on the shared state object, or a
mismatched function signature across the split files.

Skips gracefully (not a failure) if EGL/MuJoCo or a checkpoint isn't
available in this environment, per docs/refactor_plan.md's "skip gracefully
if EGL unavailable" guidance.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

_REPO = Path(__file__).resolve().parent.parent.parent
_GOTO_CKPT = _REPO / 'checkpoint' / 'goto_best.pt'
_MANEUVER_CKPT = _REPO / 'checkpoint' / 'maneuver_best.pt'


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
class TestSearchRolloutSmoke(unittest.TestCase):
    """A handful of control steps of the real search rollout (scan phase only)."""

    def test_short_rollout_runs_without_raising(self):
        import numpy as np

        from code.inferencer import Inferencer
        from code.eval.search_types import sample_search_scene
        from code.eval.search_rollout import _run_search_rollout

        inf = Inferencer(
            checkpoint_path=str(_GOTO_CKPT), arch='A', device='cpu',
            goal_source='classical', verbose=False,
        )
        rng = np.random.default_rng(999)
        scene_cfg = sample_search_scene(rng, 0)

        result = _run_search_rollout(
            inf=inf, scene_cfg=scene_cfg, instruction=scene_cfg['instruction'],
            maxsteps=8, render_video=False, video_path=None,
        )

        for key in ('success', 'spotted', 'scan_steps', 'failure_tag', 'steps',
                    'final_dist', 'fell', 'ms_per_step', 'avoid_bias_active_frac'):
            self.assertIn(key, result)
        self.assertLessEqual(result['steps'], 8)
        self.assertIsInstance(result['fell'], bool)


@unittest.skipUnless(_EGL_OK, "EGL/MuJoCo GL context unavailable in this environment")
@unittest.skipUnless(_MANEUVER_CKPT.exists(), f"checkpoint not found: {_MANEUVER_CKPT}")
class TestManeuverRolloutSmoke(unittest.TestCase):
    """A handful of control steps of the real maneuver rollout."""

    def test_short_rollout_runs_without_raising(self):
        import torch

        from code.small_vla import GroundedNav
        from code.maneuver_scene import sample_maneuver_scene, derive_rng
        from code.eval.maneuver_rollout import run_maneuver_rollout

        ckpt = torch.load(str(_MANEUVER_CKPT), map_location='cpu', weights_only=False)
        arch = ckpt.get('arch', 'A')
        proprio_dim = ckpt.get('proprio_dim', 62)
        model = GroundedNav(arch=arch, teacher_forcing=True, chunk_H=1, proprio_dim=proprio_dim)
        model.load_state_dict(ckpt.get('model_state', ckpt), strict=False)
        model.eval()

        as_raw = ckpt.get('action_stats')
        action_stats = None
        if as_raw:
            import numpy as np
            action_stats = {
                'mean': np.array(as_raw['mean'], dtype=np.float32),
                'std': np.array(as_raw['std'], dtype=np.float32),
                'default_angles': np.array(as_raw['default_angles'], dtype=np.float32),
            }

        rng = derive_rng(999, 0)
        scene_cfg = sample_maneuver_scene(rng)

        result = run_maneuver_rollout(
            model=model, action_stats=action_stats, device=torch.device('cpu'),
            scene_cfg=scene_cfg, maxsteps=8, render_video=False, video_path=None,
        )

        self.assertLessEqual(result.steps, 8)
        self.assertIn(result.failure_tag, ('success', 'fall', 'no_landmark', 'wrong_heading'))
        self.assertIsInstance(result.fell, bool)


if __name__ == '__main__':
    unittest.main()
