"""Unit tests for code/perception/hsv_config.py: constants + toggles."""
from __future__ import annotations

import os
import unittest

import numpy as np

from code.perception.hsv_config import (GROUND_SPLIT_SIZE_HI, GROUND_SPLIT_SIZE_LO,
                                        HSV_BOUNDS, M6_NEAR_DEPTH_M, M6_SIZE_BAND_HI,
                                        M6_SIZE_BAND_LO, MAX_DEPTH_M, MIN_BLOB_AREA,
                                        MIN_DEPTH_M, MIN_DEPTH_PROXIMITY_M,
                                        MIN_DEPTH_WIDEFOV_M, NOMINAL_DIMS_M, _env_flag)


class TestEnvFlag(unittest.TestCase):

    def setUp(self):
        self._saved = os.environ.pop("G2_TEST_FLAG", None)

    def tearDown(self):
        os.environ.pop("G2_TEST_FLAG", None)
        if self._saved is not None:
            os.environ["G2_TEST_FLAG"] = self._saved

    def test_default_used_when_unset(self):
        self.assertFalse(_env_flag("G2_TEST_FLAG", default="0"))
        self.assertTrue(_env_flag("G2_TEST_FLAG", default="1"))

    def test_exact_string_one_is_true(self):
        os.environ["G2_TEST_FLAG"] = "1"
        self.assertTrue(_env_flag("G2_TEST_FLAG"))

    def test_anything_else_is_false(self):
        for val in ("0", "true", "yes", "TRUE", "2", ""):
            os.environ["G2_TEST_FLAG"] = val
            self.assertFalse(_env_flag("G2_TEST_FLAG"), msg=f"val={val!r}")

    def test_whitespace_around_one_is_stripped(self):
        # _env_flag strips before comparing, so surrounding whitespace around
        # "1" is still accepted as true.
        for val in (" 1", "1 ", " 1 "):
            os.environ["G2_TEST_FLAG"] = val
            self.assertTrue(_env_flag("G2_TEST_FLAG"), msg=f"val={val!r}")


class TestHsvBounds(unittest.TestCase):

    EXPECTED_COLORS = {"red", "yellow", "blue", "green", "orange", "purple", "cyan"}

    def test_all_expected_colors_present(self):
        self.assertEqual(set(HSV_BOUNDS.keys()), self.EXPECTED_COLORS)

    def test_red_has_two_ranges_others_have_one(self):
        self.assertEqual(len(HSV_BOUNDS["red"]), 2)
        for color in self.EXPECTED_COLORS - {"red"}:
            self.assertEqual(len(HSV_BOUNDS[color]), 1)

    def test_bounds_are_valid_hsv_arrays(self):
        for color, ranges in HSV_BOUNDS.items():
            for lo, hi in ranges:
                self.assertEqual(lo.shape, (3,))
                self.assertEqual(hi.shape, (3,))
                self.assertTrue(np.all(lo <= hi), msg=f"{color}: lo>hi")
                self.assertTrue(np.all((lo >= 0) & (hi <= 255)))
                # Hue channel is OpenCV convention [0,179].
                self.assertLessEqual(lo[0], 179)
                self.assertLessEqual(hi[0], 179)


class TestNominalDims(unittest.TestCase):

    def test_ball_and_sphere_are_symmetric_and_equal(self):
        self.assertEqual(NOMINAL_DIMS_M["ball"], NOMINAL_DIMS_M["sphere"])
        w, h = NOMINAL_DIMS_M["ball"]
        self.assertEqual(w, h)

    def test_cube_and_box_are_symmetric_and_equal(self):
        self.assertEqual(NOMINAL_DIMS_M["cube"], NOMINAL_DIMS_M["box"])
        w, h = NOMINAL_DIMS_M["cube"]
        self.assertEqual(w, h)

    def test_cylinder_taller_than_wide(self):
        w, h = NOMINAL_DIMS_M["cylinder"]
        self.assertAlmostEqual(h, w * 1.6, places=9)
        self.assertGreater(h, w)

    def test_cone_taller_than_wide(self):
        w, h = NOMINAL_DIMS_M["cone"]
        self.assertAlmostEqual(h, w * 2.09, places=9)
        self.assertGreater(h, w)

    def test_all_dims_positive(self):
        for shape, (w, h) in NOMINAL_DIMS_M.items():
            self.assertGreater(w, 0, msg=shape)
            self.assertGreater(h, 0, msg=shape)


class TestDepthAndSizeConstantsOrdering(unittest.TestCase):
    """Pins down relative orderings the pipeline logic depends on."""

    def test_proximity_and_widefov_floors_below_standard_floor(self):
        self.assertLess(MIN_DEPTH_PROXIMITY_M, MIN_DEPTH_M)
        self.assertLess(MIN_DEPTH_WIDEFOV_M, MIN_DEPTH_M)

    def test_min_depth_below_max_depth(self):
        self.assertLess(MIN_DEPTH_M, MAX_DEPTH_M)

    def test_size_band_lo_below_hi(self):
        self.assertLess(M6_SIZE_BAND_LO, M6_SIZE_BAND_HI)
        self.assertLess(GROUND_SPLIT_SIZE_LO, GROUND_SPLIT_SIZE_HI)

    def test_near_depth_positive(self):
        self.assertGreater(M6_NEAR_DEPTH_M, 0.0)

    def test_min_blob_area_positive(self):
        self.assertGreater(MIN_BLOB_AREA, 0)


if __name__ == "__main__":
    unittest.main()
