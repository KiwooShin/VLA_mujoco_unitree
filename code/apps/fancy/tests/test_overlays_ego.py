"""Unit tests for code.apps.fancy.overlays_ego: detector-heatmap blend +
ego|BEV SBS frame composition."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from code.apps.fancy.constants import BEV_H, BEV_W, EGO_H, EGO_W, STATE_SEARCHING
from code.apps.fancy.overlays_ego import compose_sbs_frame, draw_detector_heatmap_overlay


class DrawDetectorHeatmapOverlayTest(unittest.TestCase):
    def _ego(self) -> np.ndarray:
        return np.full((60, 80, 3), 30, dtype=np.uint8)

    def test_none_cache_returns_unchanged(self) -> None:
        ego = self._ego()
        out, conf = draw_detector_heatmap_overlay(ego, None, "red", "ball")
        self.assertIs(out, ego)
        self.assertIsNone(conf)

    def test_mismatched_color_shape_returns_unchanged(self) -> None:
        ego = self._ego()
        cache = {"prob": np.ones((10, 10)), "color": "blue", "shape": "cube",
                  "accepted": True, "confidence": 0.9}
        out, conf = draw_detector_heatmap_overlay(ego, cache, "red", "ball")
        self.assertIs(out, ego)
        self.assertIsNone(conf)

    def test_not_accepted_returns_unchanged(self) -> None:
        ego = self._ego()
        cache = {"prob": np.ones((10, 10)), "color": "red", "shape": "ball",
                  "accepted": False, "confidence": 0.9}
        out, conf = draw_detector_heatmap_overlay(ego, cache, "red", "ball")
        self.assertIs(out, ego)
        self.assertIsNone(conf)

    def test_matching_accepted_cache_blends_and_returns_confidence(self) -> None:
        ego = self._ego()
        prob = np.zeros((10, 10), dtype=np.float32)
        prob[5, 5] = 1.0
        cache = {"prob": prob, "color": "red", "shape": "ball",
                  "accepted": True, "confidence": 0.77}
        out, conf = draw_detector_heatmap_overlay(ego, cache, "red", "ball")
        self.assertEqual(out.shape, ego.shape)
        self.assertAlmostEqual(conf, 0.77)
        self.assertFalse(np.array_equal(out, ego))

    def test_case_insensitive_query_match(self) -> None:
        ego = self._ego()
        cache = {"prob": np.ones((10, 10)) * 0.5, "color": "red", "shape": "ball",
                  "accepted": True, "confidence": 0.5}
        out, conf = draw_detector_heatmap_overlay(ego, cache, "RED", "  Ball ")
        self.assertIsNotNone(conf)


class ComposeSbsFrameTest(unittest.TestCase):
    def _ego_rgb(self) -> np.ndarray:
        return np.full((EGO_H, EGO_W, 3), 60, dtype=np.uint8)

    def _bev_bgr(self) -> np.ndarray:
        return np.full((BEV_H, BEV_W, 3), 20, dtype=np.uint8)

    def test_low_res_output_shape(self) -> None:
        with mock.patch("code.apps.fancy.overlays_ego.FEAT_HIRES", False), \
             mock.patch("code.apps.fancy.overlays_ego.FEAT_HUD", False):
            sbs = compose_sbs_frame(self._ego_rgb(), self._bev_bgr(), STATE_SEARCHING, "prompt", 1.0)
        expected_ego_w = int(EGO_W * (BEV_H / EGO_H))
        self.assertEqual(sbs.shape, (BEV_H, expected_ego_w + 3 + BEV_W, 3))

    def test_hud_bar_adds_strip(self) -> None:
        with mock.patch("code.apps.fancy.overlays_ego.FEAT_HIRES", False), \
             mock.patch("code.apps.fancy.overlays_ego.FEAT_HUD", True):
            sbs = compose_sbs_frame(
                self._ego_rgb(), self._bev_bgr(), STATE_SEARCHING, "prompt", 1.0,
                hud_ctx={"prompt": "p", "stage_idx": 0, "dist": 1.0, "bearing_deg": 0.0,
                         "step": 1, "walk_speed_mps": 0.1, "active_cam": "GROUNDING", "cam_flash": False},
            )
        from code.apps.fancy.constants import HUD_BAR_H
        self.assertEqual(sbs.shape[0], BEV_H + HUD_BAR_H)

    def test_hud_ctx_none_skips_hud_even_if_feat_on(self) -> None:
        with mock.patch("code.apps.fancy.overlays_ego.FEAT_HIRES", False), \
             mock.patch("code.apps.fancy.overlays_ego.FEAT_HUD", True):
            sbs = compose_sbs_frame(self._ego_rgb(), self._bev_bgr(), STATE_SEARCHING, "p", 1.0, hud_ctx=None)
        self.assertEqual(sbs.shape[0], BEV_H)

    def test_hires_resizes_to_panel_display_dims(self) -> None:
        with mock.patch("code.apps.fancy.overlays_ego.FEAT_HIRES", True), \
             mock.patch("code.apps.fancy.overlays_ego.FEAT_HUD", False):
            sbs = compose_sbs_frame(self._ego_rgb(), self._bev_bgr(), STATE_SEARCHING, "p", 1.0)
        from code.apps.fancy.constants import PANEL_DISPLAY_H, PANEL_DISPLAY_W
        self.assertEqual(sbs.shape, (PANEL_DISPLAY_H, PANEL_DISPLAY_W * 2 + 3, 3))

    def test_proximity_vs_grounding_labels_differ(self) -> None:
        with mock.patch("code.apps.fancy.overlays_ego.FEAT_HIRES", False), \
             mock.patch("code.apps.fancy.overlays_ego.FEAT_HUD", False):
            sbs_g = compose_sbs_frame(self._ego_rgb(), self._bev_bgr(), STATE_SEARCHING, "p", 1.0,
                                       active_cam="GROUNDING")
            sbs_p = compose_sbs_frame(self._ego_rgb(), self._bev_bgr(), STATE_SEARCHING, "p", 1.0,
                                       active_cam="PROXIMITY")
        self.assertFalse(np.array_equal(sbs_g, sbs_p))


if __name__ == "__main__":
    unittest.main()
