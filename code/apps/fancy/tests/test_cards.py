"""Unit tests for code.apps.fancy.cards: _final_canvas_dims + the title/outro
VF-1 stats cards."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from code.apps.fancy import cards as C


class FinalCanvasDimsTest(unittest.TestCase):
    def test_matches_original_low_res_no_hud_formula(self) -> None:
        with mock.patch.object(C, "FEAT_HIRES", False), mock.patch.object(C, "FEAT_HUD", False):
            h, w = C._final_canvas_dims()
        expected_w = int(C.EGO_W * (C.BEV_H / C.EGO_H)) + 3 + C.BEV_W
        self.assertEqual((h, w), (C.BEV_H, expected_w))

    def test_hud_adds_strip_height(self) -> None:
        with mock.patch.object(C, "FEAT_HIRES", False), mock.patch.object(C, "FEAT_HUD", True):
            h, _ = C._final_canvas_dims()
        with mock.patch.object(C, "FEAT_HIRES", False), mock.patch.object(C, "FEAT_HUD", False):
            h_no_hud, _ = C._final_canvas_dims()
        self.assertEqual(h - h_no_hud, C.HUD_BAR_H)

    def test_hires_uses_panel_display_dims(self) -> None:
        with mock.patch.object(C, "FEAT_HIRES", True), mock.patch.object(C, "FEAT_HUD", False):
            h, w = C._final_canvas_dims()
        self.assertEqual(h, C.PANEL_DISPLAY_H)
        self.assertEqual(w, C.PANEL_DISPLAY_W * 2 + 3)


class MakeTitleCardTest(unittest.TestCase):
    def test_shape_matches_final_canvas_dims(self) -> None:
        h, w = C._final_canvas_dims()
        img = C.make_title_card("go to the red ball", "Demo Scenario", 5, 38)
        self.assertEqual(img.shape, (h, w, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_fade_in_increases_intensity(self) -> None:
        dim = C.make_title_card("instr", "Title", 0, 38)
        bright = C.make_title_card("instr", "Title", 10, 38)
        # frame 0 is fully faded-out (all colors scaled by 0) so text pixels
        # should be darker/less prevalent than the fully faded-in frame.
        self.assertLessEqual(int(dim.sum()), int(bright.sum()))

    def test_handles_empty_instruction(self) -> None:
        img = C.make_title_card("", "Title", 0, 38)
        self.assertIsNotNone(img)


class MakeOutroCardTest(unittest.TestCase):
    def test_same_shape_as_input_frame(self) -> None:
        frame = np.zeros((240, 640, 3), dtype=np.uint8)
        out = C.make_outro_card(frame, sim_time_s=12.3, dist_traveled_m=4.5,
                                 final_dist_m=0.4, steps=500)
        self.assertEqual(out.shape, frame.shape)

    def test_does_not_mutate_input_frame(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame_copy = frame.copy()
        C.make_outro_card(frame, sim_time_s=1.0, dist_traveled_m=1.0,
                           final_dist_m=0.1, steps=10)
        self.assertTrue(np.array_equal(frame, frame_copy))

    def test_narrow_frame_panel_is_clamped(self) -> None:
        frame = np.zeros((100, 50, 3), dtype=np.uint8)
        out = C.make_outro_card(frame, sim_time_s=1.0, dist_traveled_m=1.0,
                                 final_dist_m=0.1, steps=10)
        self.assertEqual(out.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
