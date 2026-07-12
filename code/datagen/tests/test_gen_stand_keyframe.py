"""Unit test for code.datagen.gen_stand_keyframe (RF-1).

Real WBC settle (cheap, ~0.5s, no rendering); writes to a tempdir so the
real checkpoint/stand_keyframe.npz used by the live system is never touched.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from code.datagen.gen_stand_keyframe import SETTLE_STEPS, gen_keyframe


class GenKeyframeTest(unittest.TestCase):
    def test_writes_expected_keys_to_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "stand_keyframe_test.npz")
            written = gen_keyframe(out_path=out_path)
            self.assertEqual(written, out_path)
            self.assertTrue(Path(out_path).exists())

            data = np.load(out_path)
            for key in ("qpos_local", "qvel_local", "target_dof", "height", "settle_steps"):
                self.assertIn(key, data.files)
            self.assertEqual(int(data["settle_steps"]), SETTLE_STEPS)
            self.assertEqual(data["target_dof"].shape, (15,))
            # Settled height should be a plausible standing height, and the
            # xy translation must be stripped (robot-local frame).
            self.assertGreater(float(data["height"]), 0.5)
            self.assertEqual(float(data["qpos_local"][0]), 0.0)
            self.assertEqual(float(data["qpos_local"][1]), 0.0)


if __name__ == "__main__":
    unittest.main()
