"""Unit tests for code.train.gaitfix_loss: GaitFixLoss and JOINT_NAMES.

Extensive coverage of the residual/standardized action loss (Fix 1): the
normalize/denormalize round trip, swing-joint upweighting, the multi-task
forward pass with/without optional goal/vel heads, and 2D-vs-3D action-target
broadcasting.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from code.train.gaitfix_loss import GaitFixLoss, JOINT_NAMES


def _action_stats(n=15, mean=0.0, std=1.0, default=0.0):
    return {
        'mean': np.full(n, mean, dtype=np.float32),
        'std': np.full(n, std, dtype=np.float32),
        'default_angles': np.full(n, default, dtype=np.float32),
    }


class TestJointNames(unittest.TestCase):
    def test_length_matches_15_dof(self):
        self.assertEqual(len(JOINT_NAMES), 15)

    def test_all_strings_and_unique(self):
        self.assertTrue(all(isinstance(n, str) for n in JOINT_NAMES))
        self.assertEqual(len(set(JOINT_NAMES)), len(JOINT_NAMES))


class TestNormalizeDenormalize(unittest.TestCase):
    def test_round_trip_identity_with_trivial_stats(self):
        stats = _action_stats(mean=0.0, std=1.0, default=0.0)
        loss_fn = GaitFixLoss(action_stats=stats)
        action = torch.randn(4, 1, 15)
        normed = loss_fn.normalize_action(action)
        back = loss_fn.denormalize_action(normed)
        torch.testing.assert_close(back, action)

    def test_round_trip_with_nontrivial_stats(self):
        mean = np.linspace(-0.1, 0.1, 15).astype(np.float32)
        std = np.linspace(0.05, 0.3, 15).astype(np.float32)
        default = np.linspace(-0.2, 0.2, 15).astype(np.float32)
        stats = {'mean': mean, 'std': std, 'default_angles': default}
        loss_fn = GaitFixLoss(action_stats=stats)
        action = torch.randn(2, 3, 15)
        normed = loss_fn.normalize_action(action)
        back = loss_fn.denormalize_action(normed)
        torch.testing.assert_close(back, action, atol=1e-5, rtol=1e-5)

    def test_normalize_matches_manual_formula(self):
        stats = _action_stats(mean=0.1, std=2.0, default=0.5)
        loss_fn = GaitFixLoss(action_stats=stats)
        action = torch.full((1, 1, 15), 1.5)
        normed = loss_fn.normalize_action(action)
        # delta = 1.5 - 0.5 = 1.0; (1.0 - 0.1) / 2.0 = 0.45
        torch.testing.assert_close(normed, torch.full((1, 1, 15), 0.45))


class TestSwingWeighting(unittest.TestCase):
    def test_no_upweighting_when_swing_weight_is_one(self):
        stats = _action_stats()
        loss_fn = GaitFixLoss(action_stats=stats, swing_weight=1.0)
        torch.testing.assert_close(loss_fn._joint_w, torch.ones(15))

    def test_top5_highest_variance_joints_upweighted(self):
        std = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0.1, 0.2, 0.3, 0.4, 0.5],
                        dtype=np.float32)
        stats = {'mean': np.zeros(15, dtype=np.float32), 'std': std,
                  'default_angles': np.zeros(15, dtype=np.float32)}
        loss_fn = GaitFixLoss(action_stats=stats, swing_weight=3.0)
        top5_idx = set(np.argsort(std)[::-1][:5].tolist())
        for i in range(15):
            expected = 3.0 if i in top5_idx else 1.0
            self.assertAlmostEqual(float(loss_fn._joint_w[i]), expected, places=6)


class TestForward(unittest.TestCase):
    def _preds(self, B=2, H=1, with_goal=True, with_vel=True):
        preds = {
            'action': torch.zeros(B, H, 15),
            'done': torch.zeros(B),
        }
        if with_goal:
            preds['goal'] = torch.zeros(B, 3)
        if with_vel:
            preds['vel'] = torch.zeros(B, 3)
        return preds

    def test_zero_error_gives_zero_action_and_goal_vel_loss(self):
        stats = _action_stats()
        loss_fn = GaitFixLoss(action_stats=stats)
        preds = self._preds()
        B = 2
        total, parts = loss_fn(
            preds,
            action_gt_abs=torch.zeros(B, 1, 15),
            done_gt=torch.zeros(B),
            goal_gt=torch.zeros(B, 3),
            vel_gt=torch.zeros(B, 3),
        )
        self.assertAlmostEqual(parts['action'], 0.0, places=6)
        self.assertAlmostEqual(parts['goal'], 0.0, places=6)
        self.assertAlmostEqual(parts['vel'], 0.0, places=6)
        # done: logits=0 vs gt=0 -> BCE(sigmoid(0)=0.5, 0) = -log(0.5) ~ 0.693
        self.assertAlmostEqual(parts['done'], np.log(2.0), places=4)
        self.assertGreater(total.item(), 0.0)

    def test_missing_goal_and_vel_heads_score_zero(self):
        stats = _action_stats()
        loss_fn = GaitFixLoss(action_stats=stats)
        preds = self._preds(with_goal=False, with_vel=False)
        total, parts = loss_fn(
            preds, action_gt_abs=torch.zeros(2, 1, 15), done_gt=torch.zeros(2),
        )
        self.assertEqual(parts['goal'], 0.0)
        self.assertEqual(parts['vel'], 0.0)

    def test_2d_action_gt_is_promoted_to_3d(self):
        """action_gt_abs given as (B, 15) (no horizon dim) must still work."""
        stats = _action_stats()
        loss_fn = GaitFixLoss(action_stats=stats)
        preds = self._preds(H=1)
        total_2d, parts_2d = loss_fn(
            preds, action_gt_abs=torch.zeros(2, 15), done_gt=torch.zeros(2),
        )
        total_3d, parts_3d = loss_fn(
            preds, action_gt_abs=torch.zeros(2, 1, 15), done_gt=torch.zeros(2),
        )
        self.assertAlmostEqual(parts_2d['action'], parts_3d['action'], places=6)

    def test_total_is_weighted_sum_of_parts(self):
        stats = _action_stats()
        w_action, w_goal, w_vel, w_done = 5.0, 1.0, 1.0, 1.0
        loss_fn = GaitFixLoss(action_stats=stats, w_action=w_action, w_goal=w_goal,
                               w_vel=w_vel, w_done=w_done)
        preds = self._preds()
        preds['action'] = torch.full((2, 1, 15), 0.3)
        preds['goal'] = torch.full((2, 3), 0.2)
        preds['vel'] = torch.full((2, 3), -0.1)
        total, parts = loss_fn(
            preds, action_gt_abs=torch.zeros(2, 1, 15), done_gt=torch.ones(2),
            goal_gt=torch.zeros(2, 3), vel_gt=torch.zeros(2, 3),
        )
        expected = (w_action * parts['action'] + w_goal * parts['goal']
                    + w_vel * parts['vel'] + w_done * parts['done'])
        self.assertAlmostEqual(total.item(), expected, places=4)
        self.assertAlmostEqual(parts['total'], total.item(), places=6)

    def test_deterministic_given_same_inputs(self):
        stats = _action_stats()
        loss_fn = GaitFixLoss(action_stats=stats)
        preds = self._preds()
        preds['action'] = torch.full((2, 1, 15), 0.5)
        kwargs = dict(action_gt_abs=torch.zeros(2, 1, 15), done_gt=torch.zeros(2),
                      goal_gt=torch.zeros(2, 3), vel_gt=torch.zeros(2, 3))
        total_a, _ = loss_fn(preds, **kwargs)
        total_b, _ = loss_fn(preds, **kwargs)
        self.assertEqual(total_a.item(), total_b.item())


if __name__ == '__main__':
    unittest.main()
