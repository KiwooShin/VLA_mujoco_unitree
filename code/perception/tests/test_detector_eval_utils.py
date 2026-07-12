"""Unit tests for code/perception/detector/eval_utils.py: threshold scoring
and selection utilities (pure numpy, no model/dataset needed)."""
from __future__ import annotations

import unittest

import numpy as np

from code.perception.detector.eval_utils import (InferenceResult, _angle_diff_deg,
                                                 presence_only_pr, score_at_threshold,
                                                 select_threshold)


def _make_result(confidence, dist_pred, bearing_pred, has_target, dist_gt, bearing_gt):
    n = len(confidence)
    return InferenceResult(
        confidence=np.array(confidence, dtype=np.float64),
        dist_pred=np.array(dist_pred, dtype=np.float64),
        bearing_pred=np.array(bearing_pred, dtype=np.float64),
        has_target=np.array(has_target, dtype=np.float64),
        dist_gt=np.array(dist_gt, dtype=np.float64),
        bearing_gt=np.array(bearing_gt, dtype=np.float64),
        cam_type=["grounding"] * n,
        class_id=np.zeros(n, dtype=np.int64),
        color_id=np.zeros(n, dtype=np.int64),
        row_i=np.arange(n),
    )


class TestAngleDiffDeg(unittest.TestCase):

    def test_zero_difference(self):
        d = _angle_diff_deg(np.array([10.0]), np.array([10.0]))
        np.testing.assert_allclose(d, [0.0])

    def test_wraparound_positive_side(self):
        d = _angle_diff_deg(np.array([179.0]), np.array([-179.0]))
        np.testing.assert_allclose(d, [-2.0])

    def test_wraparound_negative_side(self):
        d = _angle_diff_deg(np.array([-179.0]), np.array([179.0]))
        np.testing.assert_allclose(d, [2.0])

    def test_vectorized_multiple_values(self):
        a = np.array([0.0, 90.0, -90.0])
        b = np.array([10.0, 80.0, -100.0])
        d = _angle_diff_deg(a, b)
        np.testing.assert_allclose(d, [-10.0, 10.0, 10.0])


class TestScoreAtThreshold(unittest.TestCase):

    def test_perfect_detection_precision_recall_one(self):
        res = _make_result(confidence=[0.9, 0.8], dist_pred=[3.0, 5.0], bearing_pred=[0.0, 0.0],
                           has_target=[1, 1], dist_gt=[3.0, 5.0], bearing_gt=[0.0, 0.0])
        score = score_at_threshold(res, tau=0.5)
        self.assertAlmostEqual(score["precision"], 1.0)
        self.assertAlmostEqual(score["recall"], 1.0)
        self.assertEqual(score["tp"], 2)
        self.assertEqual(score["fp"], 0)

    def test_false_positive_on_negative_frame(self):
        res = _make_result(confidence=[0.9], dist_pred=[3.0], bearing_pred=[0.0],
                           has_target=[0], dist_gt=[float("nan")], bearing_gt=[float("nan")])
        score = score_at_threshold(res, tau=0.5)
        self.assertEqual(score["fp"], 1)
        self.assertEqual(score["tp"], 0)
        self.assertAlmostEqual(score["precision"], 0.0)

    def test_mislocalized_positive_counts_as_false_positive(self):
        res = _make_result(confidence=[0.9], dist_pred=[10.0], bearing_pred=[0.0],
                           has_target=[1], dist_gt=[3.0], bearing_gt=[0.0])
        score = score_at_threshold(res, tau=0.5, dist_tol=0.5)
        self.assertEqual(score["fp"], 1)
        self.assertEqual(score["tp"], 0)

    def test_no_positives_recall_is_nan(self):
        res = _make_result(confidence=[0.1], dist_pred=[3.0], bearing_pred=[0.0],
                           has_target=[0], dist_gt=[float("nan")], bearing_gt=[float("nan")])
        score = score_at_threshold(res, tau=0.5)
        self.assertTrue(np.isnan(score["recall"]))

    def test_no_detections_precision_defaults_to_one(self):
        res = _make_result(confidence=[0.1], dist_pred=[3.0], bearing_pred=[0.0],
                           has_target=[1], dist_gt=[3.0], bearing_gt=[0.0])
        score = score_at_threshold(res, tau=0.5)
        self.assertEqual(score["n_detected"], 0)
        self.assertEqual(score["precision"], 1.0)

    def test_bearing_error_beyond_tolerance_is_false_positive(self):
        res = _make_result(confidence=[0.9], dist_pred=[3.0], bearing_pred=[10.0],
                           has_target=[1], dist_gt=[3.0], bearing_gt=[0.0])
        score = score_at_threshold(res, tau=0.5, bearing_tol=2.0)
        self.assertEqual(score["fp"], 1)


class TestSelectThreshold(unittest.TestCase):

    def test_feasible_threshold_maximizes_recall_at_min_precision(self):
        # 10 positives all detected perfectly at conf=0.9, 10 negatives all
        # confidently (wrongly) fire only below conf=0.3.
        conf = [0.9] * 10 + [0.2] * 10
        dist_pred = [3.0] * 10 + [3.0] * 10
        bearing_pred = [0.0] * 20
        has_target = [1] * 10 + [0] * 10
        dist_gt = [3.0] * 10 + [float("nan")] * 10
        bearing_gt = [0.0] * 10 + [float("nan")] * 10
        res = _make_result(conf, dist_pred, bearing_pred, has_target, dist_gt, bearing_gt)
        best, curve = select_threshold(res, min_precision=0.9)
        self.assertTrue(best["met_precision_gate"])
        self.assertAlmostEqual(best["recall"], 1.0)
        self.assertGreaterEqual(len(curve), 1)

    def test_infeasible_falls_back_to_best_precision(self):
        # Every "detection" mislocalizes -> precision can never reach 0.9.
        res = _make_result(confidence=[0.9, 0.8], dist_pred=[100.0, 100.0], bearing_pred=[0.0, 0.0],
                           has_target=[1, 1], dist_gt=[3.0, 3.0], bearing_gt=[0.0, 0.0])
        best, curve = select_threshold(res, min_precision=0.9)
        self.assertFalse(best["met_precision_gate"])

    def test_custom_taus_respected(self):
        res = _make_result(confidence=[0.5], dist_pred=[3.0], bearing_pred=[0.0],
                           has_target=[1], dist_gt=[3.0], bearing_gt=[0.0])
        taus = np.array([0.1, 0.6])
        best, curve = select_threshold(res, taus=taus)
        self.assertEqual(len(curve), 2)


class TestPresenceOnlyPr(unittest.TestCase):

    def test_ignores_localization_only_presence(self):
        # A wildly mislocalized but present detection still counts as TP here
        # (unlike score_at_threshold) -- that's the whole point of this metric.
        res = _make_result(confidence=[0.9], dist_pred=[100.0], bearing_pred=[90.0],
                           has_target=[1], dist_gt=[3.0], bearing_gt=[0.0])
        out = presence_only_pr(res, tau=0.5)
        self.assertEqual(out["tp"], 1)
        self.assertEqual(out["fp"], 0)
        self.assertAlmostEqual(out["precision"], 1.0)
        self.assertAlmostEqual(out["recall"], 1.0)

    def test_false_negative_counted(self):
        res = _make_result(confidence=[0.1], dist_pred=[3.0], bearing_pred=[0.0],
                           has_target=[1], dist_gt=[3.0], bearing_gt=[0.0])
        out = presence_only_pr(res, tau=0.5)
        self.assertEqual(out["fn"], 1)
        self.assertEqual(out["tp"], 0)
        self.assertAlmostEqual(out["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
