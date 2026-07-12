"""Unit tests for code.train.gaitfix_epoch: _run_epoch and audit_velocity_head.

Uses a tiny hand-written nn.Module standing in for GroundedNav (the real
model isn't needed -- _run_epoch only requires something callable that takes
(ego_rgb, lang_emb, proprio_h, gt_goal=, gt_vel=) and returns a preds dict,
plus a `.train(bool)` method, which nn.Module already provides). This keeps
the tests fast and decoupled from code.policy.small_vla's own evolution.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from code.train.gaitfix_epoch import _run_epoch, audit_velocity_head
from code.train.gaitfix_loss import GaitFixLoss


class _TinyModel(nn.Module):
    """Minimal stand-in for GroundedNav: linear action head off proprio_h,
    plus optional goal/vel heads, so gradients actually flow end to end."""

    def __init__(self, proprio_dim=5, has_vel_head=True):
        super().__init__()
        self.action_fc = nn.Linear(proprio_dim, 15)
        self.done_fc = nn.Linear(proprio_dim, 1)
        self.goal_fc = nn.Linear(proprio_dim, 3)
        self.has_vel_head = has_vel_head
        if has_vel_head:
            self.vel_fc = nn.Linear(proprio_dim, 3)

    def forward(self, ego_rgb, lang_emb, proprio_h, gt_goal=None, gt_vel=None):
        # proprio_h: (B, K, proprio_dim) -> use last frame
        x = proprio_h[:, -1, :]
        out = {
            'action': self.action_fc(x).unsqueeze(1),   # (B, 1, 15)
            'done': self.done_fc(x).squeeze(-1),         # (B,)
            'goal': self.goal_fc(x),                     # (B, 3)
        }
        if self.has_vel_head:
            out['vel'] = self.vel_fc(x)
        return out


def _make_batch(B=4, K=6, proprio_dim=5):
    return {
        'ego_rgb': torch.zeros(B, 3, 8, 8),
        'lang_emb': torch.zeros(B, 2048),
        'proprio_h': torch.randn(B, K, proprio_dim),
        'action': torch.zeros(B, 1, 15),
        'goal': torch.zeros(B, 3),
        'vel_cmd': torch.zeros(B, 3),
        'done': torch.zeros(B),
    }


def _action_stats(n=15):
    import numpy as np
    return {
        'mean': np.zeros(n, dtype=np.float32),
        'std': np.ones(n, dtype=np.float32),
        'default_angles': np.zeros(n, dtype=np.float32),
    }


class TestRunEpoch(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.loss_fn = GaitFixLoss(action_stats=_action_stats(), device='cpu')

    def test_returns_expected_metric_keys(self):
        model = _TinyModel()
        loader = [_make_batch()]
        metrics = _run_epoch(model, loader, self.loss_fn, None, self.device, train=False)
        for key in ('total', 'action', 'done', 'goal', 'vel', 'done_acc'):
            self.assertIn(key, metrics)

    def test_eval_mode_does_not_update_parameters(self):
        model = _TinyModel()
        before = model.action_fc.weight.detach().clone()
        loader = [_make_batch()]
        _run_epoch(model, loader, self.loss_fn, None, self.device, train=False)
        after = model.action_fc.weight.detach().clone()
        torch.testing.assert_close(before, after)

    def test_train_mode_with_optimizer_updates_parameters(self):
        model = _TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.action_fc.weight.detach().clone()
        loader = [_make_batch()]
        _run_epoch(model, loader, self.loss_fn, opt, self.device, train=True)
        after = model.action_fc.weight.detach().clone()
        self.assertFalse(torch.allclose(before, after))

    def test_averages_over_multiple_batches(self):
        model = _TinyModel()
        loader = [_make_batch(), _make_batch(), _make_batch()]
        metrics = _run_epoch(model, loader, self.loss_fn, None, self.device, train=False)
        # Sanity: averaged metrics are finite floats, not accumulated sums.
        self.assertIsInstance(metrics['action'], float)
        self.assertTrue(metrics['action'] == metrics['action'])  # not NaN

    def test_done_acc_reflects_threshold_at_zero_logit(self):
        """done_pred = (logits > 0); craft a model whose done head is a fixed
        positive/negative bias to pin the accuracy computation."""
        model = _TinyModel()
        with torch.no_grad():
            model.done_fc.weight.zero_()
            model.done_fc.bias.fill_(5.0)   # always predicts logit>0 -> pred=1
        batch = _make_batch(B=4)
        batch['done'] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        metrics = _run_epoch(model, [batch], self.loss_fn, None, self.device, train=False)
        self.assertAlmostEqual(metrics['done_acc'], 0.5, places=6)

    def test_empty_loader_returns_safe_defaults(self):
        model = _TinyModel()
        metrics = _run_epoch(model, [], self.loss_fn, None, self.device, train=False)
        self.assertEqual(metrics['total'], 0.0)
        self.assertEqual(metrics['done_acc'], 0.0)


class TestAuditVelocityHead(unittest.TestCase):
    def test_reports_error_when_model_has_no_vel_head(self):
        model = _TinyModel(has_vel_head=False)
        loader = [_make_batch()]
        stats = audit_velocity_head(model, loader, torch.device('cpu'))
        self.assertIn('error', stats)

    def test_reports_stats_when_vel_head_present(self):
        model = _TinyModel(has_vel_head=True)
        loader = [_make_batch(), _make_batch()]
        stats = audit_velocity_head(model, loader, torch.device('cpu'), n_batches=2)
        for key in ('pred_mean_vx', 'gt_mean_vx', 'mae_vx', 'n_samples', 'vel_head_near_zero'):
            self.assertIn(key, stats)
        self.assertEqual(stats['n_samples'], 8)   # 2 batches * B=4

    def test_vel_head_near_zero_flag_true_for_near_zero_predictions(self):
        model = _TinyModel(has_vel_head=True)
        with torch.no_grad():
            model.vel_fc.weight.zero_()
            model.vel_fc.bias.zero_()
        stats = audit_velocity_head(model, [_make_batch()], torch.device('cpu'), n_batches=1)
        self.assertTrue(stats['vel_head_near_zero'])

    def test_vel_head_near_zero_flag_false_for_large_predictions(self):
        model = _TinyModel(has_vel_head=True)
        with torch.no_grad():
            model.vel_fc.weight.zero_()
            model.vel_fc.bias[0] = 2.0
        stats = audit_velocity_head(model, [_make_batch()], torch.device('cpu'), n_batches=1)
        self.assertFalse(stats['vel_head_near_zero'])

    def test_respects_n_batches_cap(self):
        model = _TinyModel()
        loader = [_make_batch() for _ in range(5)]
        stats = audit_velocity_head(model, loader, torch.device('cpu'), n_batches=2)
        self.assertEqual(stats['n_samples'], 8)


if __name__ == '__main__':
    unittest.main()
