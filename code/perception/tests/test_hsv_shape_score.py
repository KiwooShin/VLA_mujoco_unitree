"""Unit tests for code/perception/hsv_shape_score.py: V5 shape-discrimination
blob scoring."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from code.perception.hsv_shape_score import (_blob_composite_score, _score_all_contours,
                                             _shape_match_score)


def _circle_contour(r=30, canvas=100):
    mask = np.zeros((canvas, canvas), dtype=np.uint8)
    cv2.circle(mask, (canvas // 2, canvas // 2), r, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def _square_contour(side=50, canvas=100):
    mask = np.zeros((canvas, canvas), dtype=np.uint8)
    x0 = (canvas - side) // 2
    cv2.rectangle(mask, (x0, x0), (x0 + side, x0 + side), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def _tall_rect_contour(w=20, h=70, canvas=100):
    mask = np.zeros((canvas, canvas), dtype=np.uint8)
    x0 = (canvas - w) // 2
    y0 = (canvas - h) // 2
    cv2.rectangle(mask, (x0, y0), (x0 + w, y0 + h), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def _triangle_contour(canvas=100):
    mask = np.zeros((canvas, canvas), dtype=np.uint8)
    pts = np.array([[canvas // 2, 15], [15, 85], [85, 85]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


class TestShapeMatchScore(unittest.TestCase):

    def test_tiny_contour_scores_zero(self):
        cnt = np.array([[[5, 5]]], dtype=np.int32)
        self.assertEqual(_shape_match_score(cnt, "ball"), 0.0)

    def test_circle_favors_ball_over_cube(self):
        circle = _circle_contour()
        ball_score = _shape_match_score(circle, "ball")
        cube_score = _shape_match_score(circle, "cube")
        self.assertGreater(ball_score, cube_score)

    def test_square_favors_cube_over_ball(self):
        square = _square_contour()
        cube_score = _shape_match_score(square, "cube")
        ball_score = _shape_match_score(square, "ball")
        self.assertGreater(cube_score, ball_score)

    def test_tall_rect_favors_cylinder_over_cube(self):
        tall = _tall_rect_contour()
        cyl_score = _shape_match_score(tall, "cylinder")
        cube_score = _shape_match_score(tall, "cube")
        self.assertGreater(cyl_score, cube_score)

    def test_triangle_favors_cone_over_ball(self):
        tri = _triangle_contour()
        cone_score = _shape_match_score(tri, "cone")
        ball_score = _shape_match_score(tri, "ball")
        self.assertGreater(cone_score, ball_score)

    def test_unknown_shape_neutral_score(self):
        circle = _circle_contour()
        self.assertEqual(_shape_match_score(circle, "teapot"), 0.5)

    def test_score_bounded_zero_one(self):
        for cnt in (_circle_contour(), _square_contour(), _tall_rect_contour(), _triangle_contour()):
            for shape in ("ball", "sphere", "cube", "box", "cylinder", "cone"):
                s = _shape_match_score(cnt, shape)
                self.assertGreaterEqual(s, 0.0)
                self.assertLessEqual(s, 1.0)

    def test_sphere_alias_matches_ball(self):
        circle = _circle_contour()
        self.assertAlmostEqual(_shape_match_score(circle, "ball"),
                               _shape_match_score(circle, "sphere"))

    def test_box_alias_matches_cube(self):
        square = _square_contour()
        self.assertAlmostEqual(_shape_match_score(square, "cube"),
                               _shape_match_score(square, "box"))


class TestBlobCompositeScore(unittest.TestCase):

    def test_pure_area_when_shape_weight_zero(self):
        circle = _circle_contour(r=10)
        area = float(cv2.contourArea(circle))
        score = _blob_composite_score(circle, "teapot", shape_weight=0.0, max_area=area)
        # norm_area = sqrt(area/max_area) = 1.0 when max_area==area
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_pure_shape_when_shape_weight_one(self):
        circle = _circle_contour()
        score = _blob_composite_score(circle, "ball", shape_weight=1.0, max_area=1e9)
        self.assertAlmostEqual(score, _shape_match_score(circle, "ball"), places=6)


class TestScoreAllContours(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(_score_all_contours([], "ball"), [])

    def test_all_below_min_area_returns_empty(self):
        tiny = np.array([[[1, 1]], [[1, 2]], [[2, 2]], [[2, 1]]], dtype=np.int32)
        self.assertEqual(_score_all_contours([tiny], "ball"), [])

    def test_single_candidate_survives_regardless_of_shape(self):
        square = _square_contour()
        scored = _score_all_contours([square], "ball")
        self.assertEqual(len(scored), 1)

    def test_sorted_descending_by_score(self):
        circle = _circle_contour(r=30)
        square = _square_contour(side=50)
        scored = _score_all_contours([circle, square], "ball")
        scores = [s for s, _ in scored]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_shape_can_beat_larger_wrong_shape_area(self):
        # A big square distractor (larger area) vs a smaller true-shaped circle:
        # with SHAPE_WEIGHT=0.75 (default) the circle should win the "ball" query
        # even though the square has ~2x the area.
        small_circle = _circle_contour(r=20, canvas=200)
        big_square = _square_contour(side=57, canvas=200)  # ~area-matched-ish but bigger
        scored = _score_all_contours([small_circle, big_square], "ball")
        best_score, best_cnt = scored[0]
        # Identify which one won by comparing contour areas.
        self.assertAlmostEqual(float(cv2.contourArea(best_cnt)),
                               float(cv2.contourArea(small_circle)))

    def test_stage1_soft_fallback_when_all_below_cutoff(self):
        # Two candidates that are both poor matches for "cone" (a circle and a
        # square) should still return a ranking (soft fallback), not an empty list.
        circle = _circle_contour()
        square = _square_contour()
        scored = _score_all_contours([circle, square], "cone")
        self.assertEqual(len(scored), 2)


if __name__ == "__main__":
    unittest.main()
