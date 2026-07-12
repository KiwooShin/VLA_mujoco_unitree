"""Unit/integration tests for code/perception/hsv_pipeline.py: ground_classical().

Pure numpy/OpenCV synthetic inputs -- no MuJoCo rendering required (cheap,
<0.1s total)."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from code.arena import EGO_FOVY, EGO_H, EGO_W, get_ego_intrinsics
import code.perception.hsv_pipeline as hsv_pipeline
from code.perception.hsv_pipeline import ground_classical


def _blank_scene(h=EGO_H, w=EGO_W, bg=(30, 30, 30)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bg
    return img


class TestGroundClassicalBasic(unittest.TestCase):

    def setUp(self):
        self.intr = get_ego_intrinsics(EGO_W, EGO_H, EGO_FOVY)

    def test_unknown_color_not_visible(self):
        img = _blank_scene()
        depth = np.full((EGO_H, EGO_W), 3.0, dtype=np.float32)
        r = ground_classical(img, depth, "chartreuse", "ball", self.intr)
        self.assertTrue(r.not_visible)
        self.assertEqual(r.confidence, 0.0)

    def test_no_matching_pixels_not_visible(self):
        img = _blank_scene()   # dark grey only, no target color present
        depth = np.full((EGO_H, EGO_W), 3.0, dtype=np.float32)
        r = ground_classical(img, depth, "red", "ball", self.intr)
        self.assertTrue(r.not_visible)

    def test_red_ball_detected(self):
        img = _blank_scene()
        cx_pix, cy_pix = EGO_W * 3 // 4, EGO_H // 3
        cv2.circle(img, (cx_pix, cy_pix), 35, (220, 40, 40), -1)
        depth = np.full((EGO_H, EGO_W), 3.0, dtype=np.float32)
        r = ground_classical(img, depth, "red", "ball", self.intr, return_mask=True)
        self.assertFalse(r.not_visible)
        self.assertGreater(r.dist, 0.0)
        self.assertGreater(r.confidence, 0.0)
        self.assertIsNotNone(r.mask)
        self.assertIsNotNone(r.bbox)
        self.assertIsNotNone(r.best_area)

    def test_return_mask_false_gives_none_mask(self):
        img = _blank_scene()
        cv2.circle(img, (EGO_W // 2, EGO_H // 2), 30, (220, 40, 40), -1)
        depth = np.full((EGO_H, EGO_W), 3.0, dtype=np.float32)
        r = ground_classical(img, depth, "red", "ball", self.intr, return_mask=False)
        self.assertFalse(r.not_visible)
        self.assertIsNone(r.mask)

    def test_too_small_blob_rejected(self):
        img = _blank_scene()
        # A 4x4 red patch is far below MIN_BLOB_AREA=40 after morphological open.
        img[10:14, 10:14] = (220, 40, 40)
        depth = np.full((EGO_H, EGO_W), 3.0, dtype=np.float32)
        r = ground_classical(img, depth, "red", "ball", self.intr)
        self.assertTrue(r.not_visible)

    def test_all_depth_out_of_range_not_visible(self):
        img = _blank_scene()
        cv2.circle(img, (EGO_W // 2, EGO_H // 2), 30, (220, 40, 40), -1)
        # Depth far beyond MAX_DEPTH_M (12.0) everywhere -> no valid depth samples.
        depth = np.full((EGO_H, EGO_W), 50.0, dtype=np.float32)
        r = ground_classical(img, depth, "red", "ball", self.intr)
        self.assertTrue(r.not_visible)

    def test_shape_discrimination_prefers_correct_shape(self):
        # Larger red square (distractor) + smaller red circle (true ball target)
        # -- SHAPE_WEIGHT=0.75 should make the circle win the "ball" query.
        img = _blank_scene(h=240, w=320)
        cv2.rectangle(img, (200, 40), (300, 140), (220, 40, 40), -1)   # big square, ~100x100
        cv2.circle(img, (60, 180), 20, (220, 40, 40), -1)              # smaller true ball
        depth = np.full((240, 320), 3.0, dtype=np.float32)
        r = ground_classical(img, depth, "red", "ball", self.intr, return_mask=True)
        self.assertFalse(r.not_visible)
        # The circle is centred near x=60 -- bbox x should be well left of the square's.
        x, y, w, h = r.bbox
        self.assertLess(x + w / 2.0, 160)


class TestProximityAndWidefovDepthFloors(unittest.TestCase):

    def setUp(self):
        self.intr_base = get_ego_intrinsics(EGO_W, EGO_H, EGO_FOVY)

    def test_near_depth_rejected_without_proximity_flag(self):
        img = _blank_scene()
        cv2.circle(img, (EGO_W // 2, EGO_H // 2), 30, (220, 40, 40), -1)
        # 0.3m is below the standard MIN_DEPTH_M=0.60 floor.
        depth = np.full((EGO_H, EGO_W), 0.3, dtype=np.float32)
        r = ground_classical(img, depth, "red", "ball", self.intr_base)
        self.assertTrue(r.not_visible)

    def test_near_depth_accepted_with_proximity_flag(self):
        img = _blank_scene()
        cv2.circle(img, (EGO_W // 2, EGO_H // 2), 30, (220, 40, 40), -1)
        depth = np.full((EGO_H, EGO_W), 0.3, dtype=np.float32)
        intr = dict(self.intr_base)
        intr["is_proximity"] = True
        r = ground_classical(img, depth, "red", "ball", intr)
        self.assertFalse(r.not_visible)

    def test_near_depth_accepted_with_widefov_flag(self):
        img = _blank_scene()
        cv2.circle(img, (EGO_W // 2, EGO_H // 2), 30, (220, 40, 40), -1)
        depth = np.full((EGO_H, EGO_W), 0.3, dtype=np.float32)
        intr = dict(self.intr_base)
        intr["is_widefov"] = True
        r = ground_classical(img, depth, "red", "ball", intr)
        self.assertFalse(r.not_visible)


class TestLockM6Gate(unittest.TestCase):
    """LOCK_M6 is read as a bare module global inside ground_classical(), so
    monkeypatching hsv_pipeline.LOCK_M6 directly toggles the gate for the test
    without needing to reimport the module (matches how the real env-toggle
    is read once at import, but the *effect* of the resulting bool is a plain
    runtime branch we can flip)."""

    def setUp(self):
        self._orig = hsv_pipeline.LOCK_M6
        self.intr = get_ego_intrinsics(EGO_W, EGO_H, EGO_FOVY)

    def tearDown(self):
        hsv_pipeline.LOCK_M6 = self._orig

    def test_implausibly_huge_blob_passes_when_gate_off(self):
        hsv_pipeline.LOCK_M6 = False
        img = _blank_scene(h=240, w=320)
        # A near-full-frame red blob at 3m implies a nominal-ball-busting physical size.
        cv2.rectangle(img, (5, 5), (315, 235), (220, 40, 40), -1)
        depth = np.full((240, 320), 3.0, dtype=np.float32)
        r = ground_classical(img, depth, "red", "ball", self.intr)
        self.assertFalse(r.not_visible)

    def test_implausibly_huge_blob_rejected_when_gate_on(self):
        hsv_pipeline.LOCK_M6 = True
        img = _blank_scene(h=240, w=320)
        cv2.rectangle(img, (5, 5), (315, 235), (220, 40, 40), -1)
        depth = np.full((240, 320), 3.0, dtype=np.float32)
        r = ground_classical(img, depth, "red", "ball", self.intr)
        self.assertTrue(r.not_visible)


class TestGroundSplitToggle(unittest.TestCase):
    """GROUND_SPLIT is likewise a bare module global read at call time."""

    def setUp(self):
        self._orig_split = hsv_pipeline.GROUND_SPLIT
        self.intr = get_ego_intrinsics(EGO_W, EGO_H, EGO_FOVY)

    def tearDown(self):
        hsv_pipeline.GROUND_SPLIT = self._orig_split

    def test_split_populates_diagnostic_fields(self):
        hsv_pipeline.GROUND_SPLIT = True
        img = _blank_scene()
        cv2.circle(img, (EGO_W // 2, EGO_H // 2), 30, (220, 40, 40), -1)
        depth = np.full((EGO_H, EGO_W), 3.0, dtype=np.float32)
        r = ground_classical(img, depth, "red", "ball", self.intr)
        self.assertFalse(r.not_visible)
        self.assertIsNotNone(r.n_raw_components)
        self.assertIsNotNone(r.n_candidates)
        self.assertIsNotNone(r.split_reselected)
        self.assertIsNotNone(r.size_plausible)

    def test_split_off_diagnostic_fields_stay_none(self):
        hsv_pipeline.GROUND_SPLIT = False
        img = _blank_scene()
        cv2.circle(img, (EGO_W // 2, EGO_H // 2), 30, (220, 40, 40), -1)
        depth = np.full((EGO_H, EGO_W), 3.0, dtype=np.float32)
        r = ground_classical(img, depth, "red", "ball", self.intr)
        self.assertFalse(r.not_visible)
        self.assertIsNone(r.n_raw_components)
        self.assertIsNone(r.n_candidates)
        self.assertIsNone(r.split_reselected)
        self.assertIsNone(r.size_plausible)


class TestBackgroundForegroundRescue(unittest.TestCase):
    """E6 fix v3 / V3 depth-FG rescue: a blob covering most of the frame with
    a wide depth range is rejected as background, UNLESS a compact foreground
    (significantly closer than its local neighbourhood) sub-region can be
    rescued."""

    def setUp(self):
        self.intr = get_ego_intrinsics(EGO_W, EGO_H, EGO_FOVY)

    def test_pure_background_wall_rejected(self):
        h, w = EGO_H, EGO_W
        img = _blank_scene(h, w)
        img[:, :] = (100, 210, 220)   # cyan-ish fill across the whole frame
        # Wide depth gradient (range > 0.7m) with NO compact closer foreground patch.
        grad = np.linspace(4.0, 9.0, w, dtype=np.float32)
        depth = np.tile(grad, (h, 1))
        r = ground_classical(img, depth, "cyan", "ball", self.intr)
        self.assertTrue(r.not_visible)

    def test_foreground_patch_rescued_from_background(self):
        h, w = EGO_H, EGO_W
        img = _blank_scene(h, w)
        img[:, :] = (100, 210, 220)   # cyan-ish fill across the whole frame
        grad = np.linspace(4.0, 9.0, w, dtype=np.float32)
        depth = np.tile(grad, (h, 1)).copy()
        # Compact circular patch, well inside the frame, pulled much closer
        # than its local (blurred) neighbourhood -- the FG-rescue trigger.
        cy, cx = h // 2, w // 2
        cv2.circle(depth, (cx, cy), 22, 1.5, -1)
        r = ground_classical(img, depth, "cyan", "ball", self.intr)
        self.assertFalse(r.not_visible)
        self.assertLess(r.dist, 4.0)   # rescued detection should read near the patch's depth


if __name__ == "__main__":
    unittest.main()
