"""Unit tests for code.apps.fancy.hud.draw_hud_bar: shape/no-crash + a few
content-sensitive checks (pure drawing, no state reads beyond `ctx`)."""

from __future__ import annotations

import unittest

import numpy as np

from code.apps.fancy.constants import HUD_BAR_H, SKILL_STAGES
from code.apps.fancy.hud import draw_hud_bar


class DrawHudBarTest(unittest.TestCase):
    def _ctx(self, **overrides) -> dict:
        base = dict(prompt="find the red ball", stage_idx=2, dist=1.23,
                    bearing_deg=15.0, step=100, walk_speed_mps=0.5,
                    active_cam="GROUNDING", cam_flash=False)
        base.update(overrides)
        return base

    def test_output_shape(self) -> None:
        img = draw_hud_bar(960, self._ctx())
        self.assertEqual(img.shape, (HUD_BAR_H, 960, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_does_not_crash_with_missing_optional_fields(self) -> None:
        img = draw_hud_bar(800, {"prompt": "", "stage_idx": -1})
        self.assertEqual(img.shape, (HUD_BAR_H, 800, 3))

    def test_none_dist_and_bearing_do_not_crash(self) -> None:
        img = draw_hud_bar(800, self._ctx(dist=None, bearing_deg=None))
        self.assertEqual(img.shape[1], 800)

    def test_wide_and_narrow_widths_both_work(self) -> None:
        for w in (200, 640, 1600):
            img = draw_hud_bar(w, self._ctx())
            self.assertEqual(img.shape, (HUD_BAR_H, w, 3))

    def test_proximity_camera_flash_produces_different_pixels_than_head(self) -> None:
        img_head = draw_hud_bar(960, self._ctx(active_cam="GROUNDING", cam_flash=False))
        img_prox_flash = draw_hud_bar(960, self._ctx(active_cam="PROXIMITY", cam_flash=True))
        self.assertFalse(np.array_equal(img_head, img_prox_flash))


if __name__ == "__main__":
    unittest.main()
