"""Unit tests for code.apps.repl.maneuver_inferencer.ManeuverInferencer's
construction-time keyframe-loading branch (the closed-loop rollout() itself
is a full MuJoCo simulation — exercised by the eval_maneuver.py gates, not
unit tests here).
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from code.apps.repl.maneuver_inferencer import ManeuverInferencer


class KeyframeInitTest(unittest.TestCase):
    def test_use_keyframe_false_skips_loading(self) -> None:
        mi = ManeuverInferencer(checkpoint_path="/nonexistent/ckpt.pt", use_keyframe=False)
        self.assertIsNone(mi._keyframe)

    def test_missing_keyframe_file_leaves_none(self) -> None:
        with mock.patch("os.path.isfile", return_value=False):
            mi = ManeuverInferencer(checkpoint_path="/nonexistent/ckpt.pt", use_keyframe=True)
        self.assertIsNone(mi._keyframe)

    def test_present_keyframe_file_is_loaded(self) -> None:
        fake_npz = {
            "qpos_local": np.zeros(29, dtype=np.float32),
            "qvel_local": np.zeros(28, dtype=np.float32),
            "target_dof": np.zeros(15, dtype=np.float32),
            "height": np.array(0.78),
        }
        with mock.patch("os.path.isfile", return_value=True), \
             mock.patch("numpy.load", return_value=fake_npz):
            mi = ManeuverInferencer(checkpoint_path="/nonexistent/ckpt.pt", use_keyframe=True)
        self.assertIsNotNone(mi._keyframe)
        self.assertAlmostEqual(mi._keyframe["height"], 0.78)
        self.assertEqual(mi._keyframe["qpos_local"].shape, (29,))

    def test_constructor_stores_device_and_path(self) -> None:
        mi = ManeuverInferencer(checkpoint_path="/some/ckpt.pt", device="cpu", use_keyframe=False)
        self.assertEqual(mi.checkpoint_path, "/some/ckpt.pt")
        self.assertEqual(mi.device_str, "cpu")
        self.assertFalse(mi._loaded)
        self.assertIsNone(mi._model)
        self.assertIsNone(mi._action_stats)


if __name__ == "__main__":
    unittest.main()
