"""Unit tests for code.train.nx6_heatmap: focal_heatmap_loss and compute_loss.

Pure tensor math (CenterNet-style penalty-reduced focal loss + a
peak-pixel-only smooth-L1 distance residual) -- verified against a
hand-computed reference for small, fully-controlled inputs.
"""

from __future__ import annotations

import math
import unittest

import torch

from code.train.nx6_heatmap import focal_heatmap_loss, compute_loss


class TestFocalHeatmapLoss(unittest.TestCase):
    def test_all_negative_example_has_zero_pos_loss(self):
        B, H, W = 2, 4, 4
        heat_logit = torch.zeros(B, H, W)
        heat_target = torch.zeros(B, H, W)     # no gaussian bump anywhere
        peak_mask = torch.zeros(B, H, W)       # negative examples: all-zero mask
        total_neg, pos_loss = focal_heatmap_loss(heat_logit, heat_target, peak_mask)
        torch.testing.assert_close(pos_loss, torch.zeros(B))
        self.assertTrue(torch.all(total_neg >= 0))

    def test_matches_manual_formula_single_pixel(self):
        # 1x1x1 "image" for a fully hand-computable case.
        heat_logit = torch.tensor([[[0.5]]])
        heat_target = torch.tensor([[[0.2]]])
        peak_mask = torch.tensor([[[1.0]]])
        alpha, beta, eps = 2.0, 4.0, 1e-6

        total_neg, pos_loss = focal_heatmap_loss(
            heat_logit, heat_target, peak_mask, alpha=alpha, beta=beta, eps=eps)

        p = torch.sigmoid(torch.tensor(0.5)).item()
        neg_w = (1.0 - 0.2) ** beta
        expected_neg = -neg_w * (p ** alpha) * math.log(1.0 - p + eps)
        expected_pos = -((1.0 - p) ** alpha) * math.log(p + eps) * 1.0

        self.assertAlmostEqual(total_neg.item(), expected_neg, places=5)
        self.assertAlmostEqual(pos_loss.item(), expected_pos, places=5)

    def test_output_shapes_are_batch_only(self):
        B, H, W = 3, 6, 6
        heat_logit = torch.randn(B, H, W)
        heat_target = torch.rand(B, H, W)
        peak_mask = torch.zeros(B, H, W)
        peak_mask[:, 0, 0] = 1.0
        total_neg, pos_loss = focal_heatmap_loss(heat_logit, heat_target, peak_mask)
        self.assertEqual(total_neg.shape, (B,))
        self.assertEqual(pos_loss.shape, (B,))

    def test_higher_confidence_correct_prediction_lowers_pos_loss(self):
        """A more-confident correct peak (higher p at the GT-peak pixel)
        should yield lower positive loss."""
        heat_target = torch.tensor([[[1.0]]])
        peak_mask = torch.tensor([[[1.0]]])
        low_conf_logit = torch.tensor([[[-2.0]]])
        high_conf_logit = torch.tensor([[[2.0]]])
        _, pos_loss_low = focal_heatmap_loss(low_conf_logit, heat_target, peak_mask)
        _, pos_loss_high = focal_heatmap_loss(high_conf_logit, heat_target, peak_mask)
        self.assertLess(pos_loss_high.item(), pos_loss_low.item())


def _batch(B=2, H=4, W=4, has_target_val=1.0, resid_val=0.5):
    heat = torch.zeros(B, H, W)
    peak_mask = torch.zeros(B, H, W)
    peak_mask[:, 0, 0] = 1.0
    heat[:, 0, 0] = 1.0
    has_target = torch.full((B,), has_target_val)
    resid = torch.full((B,), resid_val)
    return {"heat": heat, "peak_mask": peak_mask, "has_target": has_target, "resid": resid}


class TestComputeLoss(unittest.TestCase):
    def test_returns_loss_and_parts_dict(self):
        B, H, W = 2, 4, 4
        heat_logit = torch.randn(B, H, W)
        dist_resid = torch.zeros(B, H, W)
        batch = _batch(B, H, W)
        loss, parts = compute_loss(heat_logit, dist_resid, batch)
        self.assertIn('heatmap_loss', parts)
        self.assertIn('dist_loss', parts)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_dist_loss_zero_when_residual_matches_gt_at_peak(self):
        B, H, W = 2, 4, 4
        heat_logit = torch.zeros(B, H, W)
        dist_resid = torch.zeros(B, H, W)
        dist_resid[:, 0, 0] = 0.5   # matches resid_val below
        batch = _batch(B, H, W, resid_val=0.5)
        _, parts = compute_loss(heat_logit, dist_resid, batch)
        self.assertAlmostEqual(parts['dist_loss'], 0.0, places=5)

    def test_dist_loss_only_counts_positive_examples(self):
        """has_target=0 examples must not contribute to dist_loss even with a
        large residual mismatch (they're masked out by has_target)."""
        B, H, W = 1, 4, 4
        heat_logit = torch.zeros(B, H, W)
        dist_resid = torch.zeros(B, H, W)
        dist_resid[:, 0, 0] = 99.0   # would be huge error if counted
        batch = _batch(B, H, W, has_target_val=0.0, resid_val=0.0)
        # n_pos is clamped to >=1 so heatmap_loss stays finite even with 0 positives
        _, parts = compute_loss(heat_logit, dist_resid, batch)
        self.assertAlmostEqual(parts['dist_loss'], 0.0, places=5)

    def test_lambda_dist_scales_distance_term(self):
        B, H, W = 1, 4, 4
        heat_logit = torch.zeros(B, H, W)
        dist_resid = torch.zeros(B, H, W)
        dist_resid[:, 0, 0] = 1.0
        batch = _batch(B, H, W, resid_val=0.0)   # smooth_l1(1.0, 0.0) = 0.5 (beta=1 default)
        loss_1x, parts_1x = compute_loss(heat_logit, dist_resid, batch, lambda_dist=1.0)
        loss_2x, parts_2x = compute_loss(heat_logit, dist_resid, batch, lambda_dist=2.0)
        self.assertAlmostEqual(parts_1x['dist_loss'], parts_2x['dist_loss'], places=5)
        expected_diff = parts_1x['dist_loss']   # extra 1.0x lambda adds one more dist_loss unit
        self.assertAlmostEqual(loss_2x.item() - loss_1x.item(), expected_diff, places=4)


if __name__ == '__main__':
    unittest.main()
