"""Unit tests for code/perception/hsv_size_gate.py: NX-3 M6 physical-size
plausibility gate."""
from __future__ import annotations

import unittest

from code.perception.hsv_size_gate import _estimate_physical_size, _physical_size_plausible


class TestEstimatePhysicalSize(unittest.TestCase):

    def test_pinhole_formula(self):
        # real_size = pixel_size * depth / focal
        w, h = _estimate_physical_size(bbox_w_px=100.0, bbox_h_px=50.0, depth_m=2.0,
                                       fx=200.0, fy=200.0)
        self.assertAlmostEqual(w, 100.0 * 2.0 / 200.0)
        self.assertAlmostEqual(h, 50.0 * 2.0 / 200.0)

    def test_degenerate_fx_returns_zero(self):
        w, h = _estimate_physical_size(10, 10, 2.0, fx=0.0, fy=100.0)
        self.assertEqual((w, h), (0.0, 0.0))

    def test_degenerate_fy_returns_zero(self):
        w, h = _estimate_physical_size(10, 10, 2.0, fx=100.0, fy=-1.0)
        self.assertEqual((w, h), (0.0, 0.0))

    def test_degenerate_depth_returns_zero(self):
        w, h = _estimate_physical_size(10, 10, 0.0, fx=100.0, fy=100.0)
        self.assertEqual((w, h), (0.0, 0.0))
        w, h = _estimate_physical_size(10, 10, -1.0, fx=100.0, fy=100.0)
        self.assertEqual((w, h), (0.0, 0.0))


class TestPhysicalSizePlausible(unittest.TestCase):
    """Uses an intrinsics dict + image geometry chosen so the back-projected
    physical size can be worked out by hand: fx=fy=100, depth=2m -> a bbox of
    (w_px, h_px) yields phys=(w_px*2/100, h_px*2/100)."""

    IMG_W, IMG_H = 200, 200
    INTR = dict(fx=100.0, fy=100.0)

    def test_unknown_shape_fails_open(self):
        plausible, pw, ph = _physical_size_plausible(
            (10, 10, 50, 50), 2.0, "teapot", self.INTR, self.IMG_W, self.IMG_H, 0, 0, 0)
        self.assertTrue(plausible)

    def test_nominal_ball_size_passes_unclipped(self):
        # ball nominal = 0.24m. bbox 12px @ depth 2m, fx=100 -> phys = 12*2/100=0.24 exact match.
        plausible, pw, ph = _physical_size_plausible(
            (50, 50, 12, 12), 2.0, "ball", self.INTR, self.IMG_W, self.IMG_H, 0, 0, 0)
        self.assertTrue(plausible)
        self.assertAlmostEqual(pw, 0.24)
        self.assertAlmostEqual(ph, 0.24)

    def test_far_too_large_unclipped_fails(self):
        # bbox 500px wide @ depth 2m, fx=100 -> phys_w = 10.0m, way above ball HI band.
        plausible, pw, ph = _physical_size_plausible(
            (50, 50, 500, 12), 2.0, "ball", self.INTR, self.IMG_W, self.IMG_H, 0, 0, 0)
        self.assertFalse(plausible)

    def test_far_too_small_unclipped_fails(self):
        # bbox 1px wide @ depth 2m, fx=100 -> phys_w = 0.02m, below ball LO band (0.08*0.24=0.019
        # actually just at the edge) -- use an even smaller bbox to be unambiguous.
        plausible, pw, ph = _physical_size_plausible(
            (50, 50, 0.5, 12), 2.0, "ball", self.INTR, self.IMG_W, self.IMG_H, 0, 0, 0)
        self.assertFalse(plausible)

    def test_clipped_left_right_skips_lower_bound(self):
        # bbox touches left margin (x<=margin_l_px+1): width axis should be
        # exempt from the LOWER bound (a tiny clipped width must not fail low).
        plausible, pw, ph = _physical_size_plausible(
            (0, 50, 1, 12), 2.0, "ball", self.INTR, self.IMG_W, self.IMG_H,
            margin_l_px=0, margin_r_px=0, margin_b_px=0)
        # height (12px unclipped) is plausible (0.24m exact); width tiny+clipped is exempt low.
        self.assertTrue(plausible)

    def test_clipped_but_far_beyond_hi_still_fails(self):
        # Clipping only ever TRUNCATES the true extent, so a clipped width that
        # STILL measures far beyond the HI bound is definitive evidence of an
        # implausibly large object (when depth >= M6_NEAR_DEPTH_M).
        plausible, pw, ph = _physical_size_plausible(
            (0, 50, 500, 12), 2.0, "ball", self.INTR, self.IMG_W, self.IMG_H,
            margin_l_px=0, margin_r_px=0, margin_b_px=0)
        self.assertFalse(plausible)

    def test_clipped_near_depth_fully_exempt(self):
        # Below M6_NEAR_DEPTH_M (1.2m), a clipped axis is fully exempt even if
        # its measured extent is enormous.
        plausible, pw, ph = _physical_size_plausible(
            (0, 50, 500, 12), 1.0, "ball", self.INTR, self.IMG_W, self.IMG_H,
            margin_l_px=0, margin_r_px=0, margin_b_px=0)
        self.assertTrue(plausible)

    def test_clipped_top_and_bottom_both_exempt_low_for_height(self):
        # touches_top (y<=1) OR touches_bottom triggers the height clip path.
        plausible, pw, ph = _physical_size_plausible(
            (50, 0, 12, 1), 2.0, "ball", self.INTR, self.IMG_W, self.IMG_H,
            margin_l_px=0, margin_r_px=0, margin_b_px=0)
        self.assertTrue(plausible)

    def test_custom_lo_hi_band_overrides_default(self):
        # A bbox that fails the default M6 band should pass a much wider custom band.
        bbox = (50, 50, 500, 12)
        default_plausible, _, _ = _physical_size_plausible(
            bbox, 2.0, "ball", self.INTR, self.IMG_W, self.IMG_H, 0, 0, 0)
        wide_plausible, _, _ = _physical_size_plausible(
            bbox, 2.0, "ball", self.INTR, self.IMG_W, self.IMG_H, 0, 0, 0, lo=0.0, hi=1000.0)
        self.assertFalse(default_plausible)
        self.assertTrue(wide_plausible)

    def test_cube_and_box_share_nominal_dims(self):
        p1, w1, h1 = _physical_size_plausible(
            (50, 50, 12, 12), 2.0, "cube", self.INTR, self.IMG_W, self.IMG_H, 0, 0, 0)
        p2, w2, h2 = _physical_size_plausible(
            (50, 50, 12, 12), 2.0, "box", self.INTR, self.IMG_W, self.IMG_H, 0, 0, 0)
        self.assertEqual(p1, p2)
        self.assertAlmostEqual(w1, w2)
        self.assertAlmostEqual(h1, h2)

    def test_shape_name_case_and_whitespace_insensitive(self):
        p1, _, _ = _physical_size_plausible(
            (50, 50, 12, 12), 2.0, "Ball", self.INTR, self.IMG_W, self.IMG_H, 0, 0, 0)
        p2, _, _ = _physical_size_plausible(
            (50, 50, 12, 12), 2.0, "  ball  ", self.INTR, self.IMG_W, self.IMG_H, 0, 0, 0)
        self.assertTrue(p1)
        self.assertTrue(p2)


if __name__ == "__main__":
    unittest.main()
