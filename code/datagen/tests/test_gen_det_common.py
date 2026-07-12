"""Unit tests for code.datagen.gen_det_common (RF-1)."""

from __future__ import annotations

import unittest

from code.datagen.gen_det_common import (
    CAM_SWITCH_DIST_M,
    COLOR2I,
    COLOR_NAMES,
    SHAPE2I,
    SHAPE_NAMES,
    SIZE_M,
    _env_note,
    pick_cam,
)


class PickCamTest(unittest.TestCase):
    def test_below_switch_dist_is_proximity(self) -> None:
        self.assertEqual(pick_cam(0.5), "proximity")

    def test_at_switch_dist_is_proximity(self) -> None:
        self.assertEqual(pick_cam(CAM_SWITCH_DIST_M), "proximity")

    def test_above_switch_dist_is_grounding(self) -> None:
        self.assertEqual(pick_cam(CAM_SWITCH_DIST_M + 0.01), "grounding")

    def test_far_distance_is_grounding(self) -> None:
        self.assertEqual(pick_cam(9.0), "grounding")

    def test_zero_distance_is_proximity(self) -> None:
        self.assertEqual(pick_cam(0.0), "proximity")


class ClassTablesTest(unittest.TestCase):
    def test_color_and_shape_counts(self) -> None:
        self.assertEqual(len(COLOR_NAMES), 7)
        self.assertEqual(len(SHAPE_NAMES), 4)

    def test_index_tables_are_bijective(self) -> None:
        self.assertEqual(sorted(COLOR2I.values()), list(range(len(COLOR_NAMES))))
        self.assertEqual(sorted(SHAPE2I.values()), list(range(len(SHAPE_NAMES))))
        for name, idx in COLOR2I.items():
            self.assertEqual(COLOR_NAMES[idx], name)
        for name, idx in SHAPE2I.items():
            self.assertEqual(SHAPE_NAMES[idx], name)

    def test_size_m_covers_every_shape(self) -> None:
        for shape in SHAPE_NAMES:
            self.assertIn(shape, SIZE_M)
            self.assertGreater(SIZE_M[shape], 0.0)


class EnvNoteTest(unittest.TestCase):
    def test_env_note_does_not_raise(self) -> None:
        _env_note()  # just needs to not crash; prints to stdout.


if __name__ == "__main__":
    unittest.main()
