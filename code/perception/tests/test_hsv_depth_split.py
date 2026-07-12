"""Unit tests for code/perception/hsv_depth_split.py: NX-4 depth-guided
component splitting + CAM-2 depth-outlier rejection."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from code.perception.hsv_depth_split import (_circle_fill_score, _histogram_depth_clusters,
                                             _quick_candidate_depth, _reject_depth_outliers,
                                             _split_component_by_depth, _split_contours_by_depth)


class TestHistogramDepthClusters(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(_histogram_depth_clusters(np.array([])), [])

    def test_single_tight_cluster(self):
        vals = np.array([1.0, 1.01, 1.02, 0.99, 1.0], dtype=np.float32)
        clusters = _histogram_depth_clusters(vals, bin_m=0.15, gap_m=0.5)
        self.assertEqual(len(clusters), 1)

    def test_two_well_separated_clusters(self):
        near = np.linspace(1.0, 1.2, 20)
        far = np.linspace(3.0, 3.2, 20)
        vals = np.concatenate([near, far])
        clusters = _histogram_depth_clusters(vals, bin_m=0.1, gap_m=0.5)
        self.assertEqual(len(clusters), 2)
        # sorted ascending
        self.assertLess(clusters[0][1], clusters[1][0])

    def test_gap_smaller_than_threshold_merges(self):
        # A small gap (< gap_m) between two sub-populations should NOT split.
        near = np.linspace(1.0, 1.2, 10)
        far = np.linspace(1.35, 1.5, 10)   # gap of ~0.15m, below gap_m=0.5
        vals = np.concatenate([near, far])
        clusters = _histogram_depth_clusters(vals, bin_m=0.05, gap_m=0.5)
        self.assertEqual(len(clusters), 1)

    def test_coverage_is_exhaustive(self):
        vals = np.array([1.0, 1.5, 2.0, 5.0, 5.5])
        clusters = _histogram_depth_clusters(vals, bin_m=0.2, gap_m=0.5)
        lo = min(v[0] for v in clusters)
        hi = max(v[1] for v in clusters)
        self.assertLessEqual(lo, vals.min())
        self.assertGreaterEqual(hi, vals.max())


def _rect_mask(shape, y0, y1, x0, x1):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class TestSplitComponentByDepth(unittest.TestCase):

    def test_too_few_valid_samples_no_split(self):
        mask = _rect_mask((50, 50), 10, 15, 10, 15)  # 25px, but only ~2 valid depths
        depth = np.full((50, 50), np.nan, dtype=np.float32)
        depth[10:12, 10:12] = 1.0   # only 4 valid pixels < GROUND_SPLIT_MIN_SAMPLES(12)
        pieces = _split_component_by_depth(mask, depth, min_depth=0.1, max_depth=12.0)
        self.assertEqual(len(pieces), 1)
        np.testing.assert_array_equal(pieces[0], mask)

    def test_single_depth_population_no_split(self):
        mask = _rect_mask((50, 50), 10, 30, 10, 30)   # 400px
        depth = np.full((50, 50), 2.0, dtype=np.float32)
        pieces = _split_component_by_depth(mask, depth, min_depth=0.1, max_depth=12.0)
        self.assertEqual(len(pieces), 1)

    def test_two_depth_populations_split(self):
        mask = _rect_mask((50, 50), 10, 30, 10, 50)   # wide rectangle, 800px
        depth = np.full((50, 50), np.nan, dtype=np.float32)
        depth[10:30, 10:30] = 2.0    # near half
        depth[10:30, 30:50] = 6.0    # far half, well-separated gap > 0.5m
        pieces = _split_component_by_depth(mask, depth, min_depth=0.1, max_depth=12.0)
        self.assertEqual(len(pieces), 2)
        # each piece is a subset of the original mask, disjoint from the other
        self.assertTrue(np.all(pieces[0] <= mask))
        self.assertTrue(np.all(pieces[1] <= mask))
        overlap = pieces[0] & pieces[1]
        self.assertEqual(overlap.sum(), 0)


class TestSplitContoursByDepth(unittest.TestCase):

    def test_no_op_when_split_disabled_conceptually(self):
        # A single compact blob with uniform depth should pass through unchanged
        # (still one contour after the call).
        mask = np.zeros((60, 60), dtype=np.uint8)
        cv2.circle(mask, (30, 30), 15, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        depth = np.full((60, 60), 2.0, dtype=np.float32)
        out = _split_contours_by_depth(list(contours), mask.shape, depth, 0.1, 12.0)
        self.assertEqual(len(out), 1)

    def test_merged_blob_splits_into_two(self):
        # Simulate a target+wall merged blob: one wide rectangle, two depth populations.
        mask = np.zeros((60, 100), dtype=np.uint8)
        mask[20:40, 10:90] = 255
        depth = np.full((60, 100), np.nan, dtype=np.float32)
        depth[20:40, 10:50] = 2.0
        depth[20:40, 50:90] = 8.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = _split_contours_by_depth(list(contours), mask.shape, depth, 0.1, 12.0)
        self.assertGreaterEqual(len(out), 2)

    def test_tiny_contour_dropped(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        cv2.rectangle(mask, (5, 5), (7, 7), 255, -1)   # ~4px^2, below GROUND_SPLIT_MIN_PIECE_PX
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        depth = np.full((30, 30), 2.0, dtype=np.float32)
        out = _split_contours_by_depth(list(contours), mask.shape, depth, 0.1, 12.0)
        self.assertEqual(out, [])


class TestQuickCandidateDepth(unittest.TestCase):

    def test_returns_median_of_valid_pixels(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        cv2.circle(mask, (20, 20), 10, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        depth = np.full((40, 40), 3.5, dtype=np.float32)
        d, blob_mask = _quick_candidate_depth(contours[0], mask.shape, depth, 0.1, 12.0)
        self.assertAlmostEqual(d, 3.5, places=3)
        self.assertEqual(blob_mask.shape, mask.shape)

    def test_none_when_too_few_valid_pixels(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        cv2.rectangle(mask, (5, 5), (25, 25), 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        depth = np.full((40, 40), 100.0, dtype=np.float32)  # all out of [min,max] range
        d, _ = _quick_candidate_depth(contours[0], mask.shape, depth, 0.1, 12.0)
        self.assertIsNone(d)


class TestCircleFillScore(unittest.TestCase):

    def test_circle_scores_near_one(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        score = _circle_fill_score(contours[0])
        self.assertGreater(score, 0.9)

    def test_square_scores_below_circle(self):
        mask_sq = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask_sq, (20, 20), (80, 80), 255, -1)
        contours_sq, _ = cv2.findContours(mask_sq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        score_sq = _circle_fill_score(contours_sq[0])
        # A square inscribed in its own bounding circle covers ~2/pi=0.6366 of it.
        self.assertLess(score_sq, 0.9)
        self.assertGreater(score_sq, 0.5)

    def test_degenerate_zero_radius_returns_zero(self):
        # A single-point "contour" has near-zero enclosing circle area.
        cnt = np.array([[[5, 5]]], dtype=np.int32)
        score = _circle_fill_score(cnt)
        self.assertEqual(score, 0.0)


class TestRejectDepthOutliers(unittest.TestCase):

    def test_too_few_samples_no_op(self):
        vals = np.array([1.0, 2.0, 3.0])
        out = _reject_depth_outliers(vals)
        np.testing.assert_array_equal(out, vals)

    def test_all_within_one_bin_no_op(self):
        vals = np.array([1.0, 1.01, 1.02, 1.03, 1.04])
        out = _reject_depth_outliers(vals, bin_m=0.05)
        np.testing.assert_array_equal(out, vals)

    def test_disjoint_minority_cluster_dropped(self):
        # Majority cluster around 2.0m, small disjoint cluster (self-body) around 0.3m.
        rng = np.random.default_rng(0)
        majority = rng.normal(2.0, 0.02, size=50)
        minority = rng.normal(0.3, 0.01, size=5)
        vals = np.concatenate([majority, minority]).astype(np.float32)
        out = _reject_depth_outliers(vals, bin_m=0.05)
        self.assertGreater(out.min(), 1.0)
        self.assertLess(len(out), len(vals))

    def test_output_never_empty_when_input_nonempty(self):
        vals = np.array([1.0, 1.0, 1.0, 1.0])
        out = _reject_depth_outliers(vals)
        self.assertGreater(out.size, 0)


if __name__ == "__main__":
    unittest.main()
