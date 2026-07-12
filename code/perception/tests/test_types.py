"""Unit tests for code/perception/types.py: GroundingResult dataclass."""
from __future__ import annotations

import unittest

import numpy as np

from code.perception.types import GroundingResult


class TestGroundingResult(unittest.TestCase):

    def test_required_fields_only(self):
        r = GroundingResult(1.5, 0.9, 0.1, 0.8, False)
        self.assertEqual(r.dist, 1.5)
        self.assertEqual(r.cos_th, 0.9)
        self.assertEqual(r.sin_th, 0.1)
        self.assertEqual(r.confidence, 0.8)
        self.assertFalse(r.not_visible)

    def test_optional_fields_default_none(self):
        r = GroundingResult(0.0, 1.0, 0.0, 0.0, True)
        self.assertIsNone(r.mask)
        self.assertIsNone(r.bbox)
        self.assertIsNone(r.best_area)
        self.assertIsNone(r.phys_w)
        self.assertIsNone(r.phys_h)
        self.assertIsNone(r.n_raw_components)
        self.assertIsNone(r.n_candidates)
        self.assertIsNone(r.split_reselected)
        self.assertIsNone(r.size_plausible)

    def test_goal_vec_shape_and_values(self):
        r = GroundingResult(3.0, 0.5, 0.25, 0.9, False)
        v = r.goal_vec
        self.assertEqual(v.shape, (3,))
        self.assertEqual(v.dtype, np.float32)
        np.testing.assert_allclose(v, np.array([3.0, 0.5, 0.25], dtype=np.float32))

    def test_goal_vec_not_visible_default_zero_dist(self):
        r = GroundingResult(0, 1, 0, 0.0, True)
        v = r.goal_vec
        np.testing.assert_allclose(v, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    def test_diagnostic_fields_settable_by_keyword(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        r = GroundingResult(2.0, 1.0, 0.0, 0.5, False, mask=mask, bbox=(1, 2, 3, 4),
                           best_area=123.5, phys_w=0.2, phys_h=0.3,
                           n_raw_components=2, n_candidates=3,
                           split_reselected=True, size_plausible=True)
        self.assertIs(r.mask, mask)
        self.assertEqual(r.bbox, (1, 2, 3, 4))
        self.assertEqual(r.best_area, 123.5)
        self.assertEqual(r.phys_w, 0.2)
        self.assertEqual(r.phys_h, 0.3)
        self.assertEqual(r.n_raw_components, 2)
        self.assertEqual(r.n_candidates, 3)
        self.assertTrue(r.split_reselected)
        self.assertTrue(r.size_plausible)

    def test_two_instances_are_independent(self):
        # Dataclass without shared mutable default state -- mutating one
        # instance's optional field must not affect a sibling instance.
        r1 = GroundingResult(1, 1, 0, 1.0, False, best_area=10.0)
        r2 = GroundingResult(2, 1, 0, 1.0, False)
        self.assertIsNone(r2.best_area)
        r1.best_area = 99.0
        self.assertIsNone(r2.best_area)


if __name__ == "__main__":
    unittest.main()
