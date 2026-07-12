"""Unit tests for code/perception/detector/data.py: dataset loading helpers,
target construction, and augmentation.

Builds tiny synthetic SplitCache-like fixtures directly (bypassing parquet/
npz I/O) via SplitCache.__new__, the same technique load_failcase_cache uses
for its own non-standard layout."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from code.perception.detector.data import (SplitCache, HeatmapDataset, _crop_resize,
                                            _depth_dropout, _photometric, build_example_index,
                                            collate, gaussian_heatmap, oversample_far_or_wide,
                                            residual_target)
from code.perception.detector.model import TARGET_H, TARGET_INTR, TARGET_W
from code.perception.geometry import cam_to_egocentric
from code.arena import backproject_pixel


def _make_cache(row_labels, cam_types=None):
    n = len(row_labels)
    cache = SplitCache.__new__(SplitCache)
    cache.frames = list(range(n))   # only used via len(cache) -> len(self.frames)
    cache.rgb = np.zeros((n, TARGET_H, TARGET_W, 3), dtype=np.uint8)
    cache.depth = np.full((n, TARGET_H, TARGET_W), 3.0, dtype=np.float32)
    cache.cam_type = np.array(cam_types or ["grounding"] * n)
    cache.row_labels = row_labels
    return cache


class TestGaussianHeatmap(unittest.TestCase):

    def test_peak_at_target_location(self):
        hm = gaussian_heatmap(20, 30, cx=15.0, cy=10.0, sigma=2.0)
        py, px = np.unravel_index(np.argmax(hm), hm.shape)
        self.assertEqual((py, px), (10, 15))

    def test_peak_value_is_one(self):
        hm = gaussian_heatmap(10, 10, 5.0, 5.0, sigma=1.5)
        self.assertAlmostEqual(float(hm.max()), 1.0, places=5)

    def test_shape_matches_args(self):
        hm = gaussian_heatmap(7, 11, 3.0, 3.0)
        self.assertEqual(hm.shape, (7, 11))
        self.assertEqual(hm.dtype, np.float32)


class TestResidualTarget(unittest.TestCase):

    def test_invalid_depth_returns_zero(self):
        self.assertEqual(residual_target(10, 10, 0.0, 3.0, 0, "grounding", TARGET_INTR), 0.0)
        self.assertEqual(residual_target(10, 10, -1.0, 3.0, 0, "grounding", TARGET_INTR), 0.0)
        self.assertEqual(residual_target(10, 10, float("nan"), 3.0, 0, "grounding", TARGET_INTR), 0.0)

    def test_residual_matches_manual_geometry(self):
        from code.perception.detector.model import PITCH_BY_CAM, SIZE_M, CLASS_NAMES
        cx, cy, depth_at_px, dist_gt, class_id, cam_type = 96.0, 72.0, 3.0, 5.0, 0, "grounding"
        got = residual_target(cx, cy, depth_at_px, dist_gt, class_id, cam_type, TARGET_INTR)
        radius = SIZE_M.get(CLASS_NAMES[class_id], 0.24) / 2.0
        x_cam, y_cam, z_cam = backproject_pixel(cx, cy, depth_at_px, TARGET_INTR)
        pitch = PITCH_BY_CAM[cam_type]
        dist_bp, _ = cam_to_egocentric(x_cam, y_cam, z_cam + radius, pitch_deg=pitch,
                                       use_corrected_unpitch=True)
        self.assertAlmostEqual(got, dist_gt - dist_bp, places=6)


class TestBuildExampleIndex(unittest.TestCase):

    def test_positive_labels_always_included(self):
        labs = [dict(class_id=0, color_id=0, cx=50.0, cy=50.0, dist_gt=3.0, bearing_gt=0.0,
                    clipped=False, area_px=100)]
        cache = _make_cache([labs])
        rng = np.random.default_rng(0)
        examples = build_example_index(cache, rng, neg_per_object_frame=0, neg_per_empty_frame=0)
        positives = [e for e in examples if e[3] is not None]
        self.assertEqual(len(positives), 1)
        self.assertEqual(positives[0][:3], (0, 0, 0))

    def test_negatives_sampled_for_empty_frame(self):
        cache = _make_cache([[]])   # one frame, no labels
        rng = np.random.default_rng(0)
        examples = build_example_index(cache, rng, neg_per_object_frame=1, neg_per_empty_frame=2)
        self.assertEqual(len(examples), 2)
        for e in examples:
            self.assertIsNone(e[3])

    def test_hard_negatives_drawn_from_correct_pools(self):
        labs = [dict(class_id=0, color_id=0, cx=1.0, cy=1.0, dist_gt=1.0, bearing_gt=0.0,
                    clipped=False, area_px=50)]
        cache = _make_cache([labs])
        rng = np.random.default_rng(1)
        examples = build_example_index(cache, rng, neg_per_object_frame=0, neg_per_empty_frame=0,
                                       hard_color_negs=1, hard_shape_negs=1)
        negs = [e for e in examples if e[3] is None]
        self.assertEqual(len(negs), 2)
        # hard_color: same color(0) different class; hard_shape: same class(0) different color.
        classes_colors = {(e[1], e[2]) for e in negs}
        self.assertTrue(any(co == 0 and ci != 0 for ci, co in classes_colors))
        self.assertTrue(any(ci == 0 and co != 0 for ci, co in classes_colors))


class TestOversampleFarOrWide(unittest.TestCase):

    def test_no_op_when_extra_copies_zero(self):
        examples = [(0, 0, 0, dict(dist_gt=10.0, bearing_gt=0.0))]
        out = oversample_far_or_wide(examples, extra_copies=0)
        self.assertEqual(out, examples)

    def test_far_positive_duplicated(self):
        far = (0, 0, 0, dict(dist_gt=10.0, bearing_gt=0.0))
        near = (1, 0, 0, dict(dist_gt=2.0, bearing_gt=0.0))
        neg = (2, 0, 0, None)
        out = oversample_far_or_wide([far, near, neg], extra_copies=2, dist_thresh_m=6.0)
        self.assertEqual(out.count(far), 3)   # original + 2 copies
        self.assertEqual(out.count(near), 1)
        self.assertEqual(out.count(neg), 1)

    def test_wide_bearing_positive_duplicated(self):
        wide = (0, 0, 0, dict(dist_gt=2.0, bearing_gt=25.0))
        out = oversample_far_or_wide([wide], extra_copies=1, bearing_thresh_deg=20.0)
        self.assertEqual(out.count(wide), 2)

    def test_negatives_never_duplicated(self):
        neg = (0, 0, 0, None)
        out = oversample_far_or_wide([neg], extra_copies=3)
        self.assertEqual(out, [neg])


class TestPhotometric(unittest.TestCase):

    def test_output_shape_dtype_and_range(self):
        rgb = np.full((20, 20, 3), 128, dtype=np.uint8)
        rng = np.random.default_rng(0)
        out = _photometric(rgb, rng)
        self.assertEqual(out.shape, rgb.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.all(out >= 0) and np.all(out <= 255))

    def test_deterministic_given_seeded_rng(self):
        rgb = np.full((10, 10, 3), 100, dtype=np.uint8)
        out1 = _photometric(rgb.copy(), np.random.default_rng(42))
        out2 = _photometric(rgb.copy(), np.random.default_rng(42))
        np.testing.assert_array_equal(out1, out2)


class TestCropResize(unittest.TestCase):

    def test_output_shape_matches_input(self):
        rgb = np.zeros((100, 120, 3), dtype=np.uint8)
        depth = np.zeros((100, 120), dtype=np.float32)
        rng = np.random.default_rng(0)
        rgb_r, depth_r, cx, cy, intr = _crop_resize(rgb, depth, 60.0, 50.0, True, rng, TARGET_INTR)
        self.assertEqual(rgb_r.shape, rgb.shape)
        self.assertEqual(depth_r.shape, depth.shape)

    def test_positive_target_stays_in_frame_with_margin(self):
        rgb = np.zeros((100, 120, 3), dtype=np.uint8)
        depth = np.zeros((100, 120), dtype=np.float32)
        for seed in range(10):
            rng = np.random.default_rng(seed)
            rgb_r, depth_r, cx, cy, intr = _crop_resize(rgb, depth, 60.0, 50.0, True, rng, TARGET_INTR)
            if cx is not None:
                self.assertTrue(0 <= cx < 120)
                self.assertTrue(0 <= cy < 100)

    def test_negative_example_ignores_target_coords(self):
        rgb = np.zeros((100, 120, 3), dtype=np.uint8)
        depth = np.zeros((100, 120), dtype=np.float32)
        rng = np.random.default_rng(0)
        rgb_r, depth_r, cx, cy, intr = _crop_resize(rgb, depth, None, None, False, rng, TARGET_INTR)
        self.assertIsNone(cx)
        self.assertIsNone(cy)

    def test_intrinsics_rescaled(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.zeros((100, 100), dtype=np.float32)
        rng = np.random.default_rng(3)
        _, _, _, _, new_intr = _crop_resize(rgb, depth, 50.0, 50.0, True, rng, TARGET_INTR)
        self.assertIn("fx", new_intr)
        self.assertIn("fy", new_intr)


class _FixedRng:
    """Duck-typed stand-in for np.random.Generator exposing only what
    _depth_dropout calls, so both of its branches can be hit deterministically."""

    def __init__(self, random_value):
        self._random_value = random_value

    def random(self):
        return self._random_value

    def normal(self, loc, scale, size):
        return np.full(size, loc, dtype=np.float32)


class TestDepthDropout(unittest.TestCase):

    def test_full_dropout_branch(self):
        depth_in = np.full((10, 10), 0.5, dtype=np.float32)
        out = _depth_dropout(depth_in, _FixedRng(0.05))   # r < 0.15
        np.testing.assert_array_equal(out, np.zeros_like(depth_in))

    def test_near_field_noise_branch(self):
        depth_in = np.full((10, 10), 0.05, dtype=np.float32)   # near-field (<1.2/12)
        out = _depth_dropout(depth_in, _FixedRng(0.20))   # 0.15 <= r < 0.40
        # noise multiplier is 1.0 (fixed rng), so near-field values pass through unchanged.
        np.testing.assert_allclose(out, depth_in, atol=1e-5)

    def test_no_op_branch(self):
        depth_in = np.full((10, 10), 0.5, dtype=np.float32)
        out = _depth_dropout(depth_in, _FixedRng(0.9))   # r >= 0.40
        np.testing.assert_array_equal(out, depth_in)


class TestHeatmapDataset(unittest.TestCase):

    def _cache_with_one_positive_one_negative(self):
        pos_lab = dict(class_id=0, color_id=0, cx=96.0, cy=72.0, dist_gt=3.0, bearing_gt=2.0,
                       clipped=False, area_px=100)
        return _make_cache([[pos_lab], []])

    def test_len_matches_examples(self):
        cache = self._cache_with_one_positive_one_negative()
        examples = [(0, 0, 0, cache.row_labels[0][0]), (1, 1, 1, None)]
        ds = HeatmapDataset(cache, examples, train=False)
        self.assertEqual(len(ds), 2)

    def test_positive_example_has_target_and_valid_resid(self):
        cache = self._cache_with_one_positive_one_negative()
        examples = [(0, 0, 0, cache.row_labels[0][0])]
        ds = HeatmapDataset(cache, examples, train=False)
        item = ds[0]
        self.assertEqual(float(item["has_target"]), 1.0)
        self.assertEqual(item["x"].shape, (4, TARGET_H, TARGET_W))
        self.assertEqual(item["heat"].shape, (TARGET_H, TARGET_W))
        self.assertFalse(torch.isnan(item["dist_gt"]))

    def test_negative_example_zero_heat_and_nan_gt(self):
        cache = self._cache_with_one_positive_one_negative()
        examples = [(1, 2, 3, None)]
        ds = HeatmapDataset(cache, examples, train=False)
        item = ds[0]
        self.assertEqual(float(item["has_target"]), 0.0)
        self.assertTrue(torch.isnan(item["dist_gt"]))
        self.assertTrue(torch.isnan(item["bearing_gt"]))
        self.assertAlmostEqual(float(item["heat"].max()), 0.0)

    def test_eval_mode_deterministic_across_calls(self):
        cache = self._cache_with_one_positive_one_negative()
        examples = [(0, 0, 0, cache.row_labels[0][0])]
        ds = HeatmapDataset(cache, examples, train=False, seed=7)
        item1 = ds[0]
        item2 = ds[0]
        torch.testing.assert_close(item1["x"], item2["x"])
        torch.testing.assert_close(item1["heat"], item2["heat"])

    def test_collate_stacks_batch(self):
        cache = self._cache_with_one_positive_one_negative()
        examples = [(0, 0, 0, cache.row_labels[0][0]), (1, 1, 1, None)]
        ds = HeatmapDataset(cache, examples, train=False)
        batch = collate([ds[0], ds[1]])
        self.assertEqual(batch["x"].shape, (2, 4, TARGET_H, TARGET_W))
        self.assertEqual(len(batch["cam_type"]), 2)
        self.assertEqual(len(batch["class_id"]), 2)


if __name__ == "__main__":
    unittest.main()
