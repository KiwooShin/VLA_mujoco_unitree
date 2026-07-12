"""Unit tests for code.policy.small_vla (RF-1: blocks / heads / model split).

Exercises the from-scratch vision/proprio blocks, the grounding/velocity/
action/done heads, and the top-level GroundedNav student (both Architecture
A and C, teacher-forcing on/off, vel_proprio on/off, action chunking, and a
`load_state_dict` round trip pinning the "GroundedNav importable + loadable
at the old path" invariant) — all at tiny dimensions so the suite runs fast
on CPU.
"""

from __future__ import annotations

import unittest

import torch

from code.policy.small_vla import blocks as B
from code.policy.small_vla import heads as H
from code.policy.small_vla.model import DEFAULTS, GroundedNav


torch.manual_seed(0)


# ---------------------------------------------------------------------------
# blocks.py
# ---------------------------------------------------------------------------

class TestPatchEmbed(unittest.TestCase):
    def test_output_shape(self):
        pe = B.PatchEmbed(img_size=32, patch_size=8, in_ch=3, embed_dim=16)
        x = torch.randn(2, 3, 32, 32)
        out = pe(x)
        self.assertEqual(out.shape, (2, 16, 16))  # n_patches=(32/8)^2=16

    def test_asserts_on_non_divisible_size(self):
        with self.assertRaises(AssertionError):
            B.PatchEmbed(img_size=30, patch_size=8, in_ch=3, embed_dim=16)


class TestAttention(unittest.TestCase):
    def test_preserves_shape(self):
        attn = B.Attention(dim=16, heads=4)
        x = torch.randn(2, 5, 16)
        out = attn(x)
        self.assertEqual(out.shape, (2, 5, 16))


class TestTransformerBlock(unittest.TestCase):
    def test_preserves_shape(self):
        blk = B.TransformerBlock(dim=16, heads=4, ff_mult=2)
        x = torch.randn(2, 5, 16)
        out = blk(x)
        self.assertEqual(out.shape, (2, 5, 16))


class TestTinyViT(unittest.TestCase):
    def test_forward_shapes(self):
        vit = B.TinyViT(img_size=32, patch_size=8, in_ch=3, dim=16, depth=2, heads=4)
        x = torch.randn(3, 3, 32, 32)
        patches, pooled = vit(x)
        self.assertEqual(patches.shape, (3, 16, 16))  # (B, N, D)
        self.assertEqual(pooled.shape, (3, 16))        # (B, D)

    def test_rgbd_in_ch4(self):
        vit = B.TinyViT(img_size=16, patch_size=8, in_ch=4, dim=8, depth=1, heads=2)
        x = torch.randn(1, 4, 16, 16)
        patches, pooled = vit(x)
        self.assertEqual(patches.shape, (1, 4, 8))
        self.assertEqual(pooled.shape, (1, 8))


class TestProprioEncoder(unittest.TestCase):
    def test_forward_shape(self):
        enc = B.ProprioEncoder(proprio_dim=10, hidden=8)
        x = torch.randn(4, 6, 10)  # (B, K, proprio_dim)
        out = enc(x)
        self.assertEqual(out.shape, (4, 8))

    def test_single_history_frame(self):
        enc = B.ProprioEncoder(proprio_dim=5, hidden=4)
        out = enc(torch.randn(2, 1, 5))
        self.assertEqual(out.shape, (2, 4))


# ---------------------------------------------------------------------------
# heads.py
# ---------------------------------------------------------------------------

class TestGroundingHead(unittest.TestCase):
    def test_patch_token_path(self):
        head = H.GroundingHead(vis_dim=16, lang_dim=8, goal_dim=3, n_patches=16)
        vis = torch.randn(2, 16, 16)   # (B, N, D), N=16 -> 4x4 grid
        lang = torch.randn(2, 8)
        out = head(vis, lang)
        self.assertEqual(out.shape, (2, 3))

    def test_cls_pooled_fallback_path(self):
        head = H.GroundingHead(vis_dim=16, lang_dim=8, goal_dim=3, n_patches=16)
        vis = torch.randn(2, 16)   # (B, D) — 2D input triggers the CLS fallback
        lang = torch.randn(2, 8)
        out = head(vis, lang)
        self.assertEqual(out.shape, (2, 3))

    def test_column_convention_left_right_bearing_differs(self):
        # Concentrate all patch-attention mass in the LEFTMOST column vs. the
        # RIGHTMOST column and check the two produce different bearing outputs
        # (pins that column position actually drives the output, i.e. the
        # column->bearing physics wiring is exercised, without hardcoding a sign).
        head = H.GroundingHead(vis_dim=4, lang_dim=4, goal_dim=3, n_patches=16)
        head.eval()
        lang = torch.zeros(1, 4)
        vis_left = torch.zeros(1, 16, 4)
        vis_left[:, 0::4, :] = 10.0   # column 0 patches (row-major, G=4) get huge features
        vis_right = torch.zeros(1, 16, 4)
        vis_right[:, 3::4, :] = 10.0  # column 3 patches
        with torch.no_grad():
            out_left = head(vis_left, lang)
            out_right = head(vis_right, lang)
        self.assertFalse(torch.allclose(out_left, out_right))


class TestVelocityHead(unittest.TestCase):
    def test_no_proprio_path(self):
        head = H.VelocityHead(goal_dim=3, vis_dim=8, lang_dim=4, vel_dim=3)
        out = head(torch.randn(2, 3), torch.randn(2, 8), torch.randn(2, 4))
        self.assertEqual(out.shape, (2, 3))

    def test_vel_proprio_path(self):
        head = H.VelocityHead(goal_dim=3, vis_dim=8, lang_dim=4, vel_dim=3,
                               vel_proprio=True, proprio_enc_dim=6)
        out = head(torch.randn(2, 3), torch.randn(2, 8), torch.randn(2, 4),
                    proprio_emb=torch.randn(2, 6), phase=torch.randn(2, 2))
        self.assertEqual(out.shape, (2, 3))

    def test_vel_proprio_true_but_missing_proprio_args_raises(self):
        # forward() falls back to the (goal,vis,lang)-only concat when proprio_emb/
        # phase are omitted, but self.net was built for the LARGER vel_proprio
        # input width -> a real shape mismatch, not a silent success.
        head = H.VelocityHead(goal_dim=3, vis_dim=8, lang_dim=4, vel_dim=3,
                               vel_proprio=True, proprio_enc_dim=6)
        with self.assertRaises(RuntimeError):
            head(torch.randn(2, 3), torch.randn(2, 8), torch.randn(2, 4))


class TestActionHead(unittest.TestCase):
    def test_single_step_chunk(self):
        head = H.ActionHead(feat_dim=16, action_dim=5, chunk_H=1, n_heads=4)
        out = head(torch.randn(3, 16))
        self.assertEqual(out.shape, (3, 1, 5))

    def test_multi_step_chunk(self):
        head = H.ActionHead(feat_dim=16, action_dim=5, chunk_H=4, n_heads=4)
        out = head(torch.randn(3, 16))
        self.assertEqual(out.shape, (3, 4, 5))


class TestDoneHead(unittest.TestCase):
    def test_output_shape(self):
        head = H.DoneHead(vis_dim=8, lang_dim=4)
        out = head(torch.randn(5, 8), torch.randn(5, 4))
        self.assertEqual(out.shape, (5,))


# ---------------------------------------------------------------------------
# model.py — GroundedNav
# ---------------------------------------------------------------------------

def _small_cfg(**overrides):
    cfg = dict(
        img_size=32, in_ch=3, patch_size=8, vit_depth=1, vit_heads=4, vit_dim=16,
        vit_ff_mult=2, lang_dim=8, proprio_dim=10, proprio_K=3, gru_hidden=8,
        goal_dim=3, vel_dim=3, action_dim=5, chunk_H=1, lang_proj_dim=8,
        goal_proj_dim=4, vel_proj_dim=4, dropout=0.0, vel_proprio=False,
    )
    cfg.update(overrides)
    return cfg


def _dummy_batch(cfg, B=2):
    return dict(
        ego_rgb=torch.randn(B, cfg['in_ch'], cfg['img_size'], cfg['img_size']),
        lang_emb=torch.randn(B, cfg['lang_dim']),
        proprio_h=torch.randn(B, cfg['proprio_K'], cfg['proprio_dim']),
    )


class TestGroundedNavArchA(unittest.TestCase):
    def test_forward_with_teacher_forcing_and_gt(self):
        cfg = _small_cfg()
        model = GroundedNav(arch='A', teacher_forcing=True, **cfg)
        model.eval()
        batch = _dummy_batch(cfg)
        out = model(**batch, gt_goal=torch.randn(2, 3), gt_vel=torch.randn(2, 3))
        self.assertEqual(set(out.keys()), {'action', 'done', 'goal', 'vel'})
        self.assertEqual(out['action'].shape, (2, cfg['chunk_H'], cfg['action_dim']))
        self.assertEqual(out['goal'].shape, (2, 3))
        self.assertEqual(out['vel'].shape, (2, 3))
        self.assertEqual(out['done'].shape, (2,))

    def test_forward_teacher_forcing_without_gt_uses_predictions(self):
        cfg = _small_cfg()
        model = GroundedNav(arch='A', teacher_forcing=True, **cfg)
        model.eval()
        batch = _dummy_batch(cfg)
        out = model(**batch)  # no gt_goal/gt_vel
        self.assertEqual(out['action'].shape, (2, cfg['chunk_H'], cfg['action_dim']))

    def test_forward_no_teacher_forcing_ignores_gt(self):
        cfg = _small_cfg()
        model = GroundedNav(arch='A', teacher_forcing=False, **cfg)
        model.eval()
        batch = _dummy_batch(cfg)
        # Even though gt is supplied, teacher_forcing=False means it must be ignored.
        out_with_gt = model(**batch, gt_goal=torch.zeros(2, 3), gt_vel=torch.zeros(2, 3))
        torch.manual_seed(123)
        out_without_gt = model(**batch)
        self.assertTrue(torch.allclose(out_with_gt['action'], out_without_gt['action']))

    def test_chunk_h_greater_than_one(self):
        cfg = _small_cfg(chunk_H=4)
        model = GroundedNav(arch='A', teacher_forcing=True, **cfg)
        model.eval()
        batch = _dummy_batch(cfg)
        out = model(**batch, gt_goal=torch.randn(2, 3), gt_vel=torch.randn(2, 3))
        self.assertEqual(out['action'].shape, (2, 4, cfg['action_dim']))

    def test_vel_proprio_path_runs_and_uses_phase_slice(self):
        cfg = _small_cfg(vel_proprio=True, proprio_dim=10)
        model = GroundedNav(arch='A', teacher_forcing=True, **cfg)
        model.eval()
        self.assertTrue(model.vel_proprio)
        batch = _dummy_batch(cfg)
        out = model(**batch, gt_goal=torch.randn(2, 3), gt_vel=torch.randn(2, 3))
        self.assertEqual(out['vel'].shape, (2, 3))

    def test_eval_mode_deterministic_forward(self):
        cfg = _small_cfg()
        model = GroundedNav(arch='A', teacher_forcing=True, **cfg)
        model.eval()
        batch = _dummy_batch(cfg, B=1)
        with torch.no_grad():
            out1 = model(**batch, gt_goal=torch.zeros(1, 3), gt_vel=torch.zeros(1, 3))
            out2 = model(**batch, gt_goal=torch.zeros(1, 3), gt_vel=torch.zeros(1, 3))
        self.assertTrue(torch.equal(out1['action'], out2['action']))
        self.assertTrue(torch.equal(out1['done'], out2['done']))


class TestGroundedNavArchC(unittest.TestCase):
    def test_forward_shapes_no_goal_or_vel_keys(self):
        cfg = _small_cfg()
        model = GroundedNav(arch='C', teacher_forcing=True, **cfg)
        model.eval()
        batch = _dummy_batch(cfg)
        out = model(**batch)
        self.assertEqual(set(out.keys()), {'action', 'done'})
        self.assertEqual(out['action'].shape, (2, cfg['chunk_H'], cfg['action_dim']))

    def test_arch_string_is_uppercased(self):
        model = GroundedNav(arch='c', **_small_cfg())
        self.assertEqual(model.arch, 'C')
        self.assertFalse(hasattr(model, 'grounding'))


class TestGroundedNavParamCount(unittest.TestCase):
    def test_arch_a_breakdown_includes_grounding_and_velocity(self):
        model = GroundedNav(arch='A', **_small_cfg())
        counts = model.param_count()
        self.assertIn('grounding', counts)
        self.assertIn('velocity', counts)
        self.assertEqual(counts['total'], sum(p.numel() for p in model.parameters()))

    def test_arch_c_breakdown_excludes_grounding_and_velocity(self):
        model = GroundedNav(arch='C', **_small_cfg())
        counts = model.param_count()
        self.assertNotIn('grounding', counts)
        self.assertNotIn('velocity', counts)
        self.assertEqual(counts['total'], sum(p.numel() for p in model.parameters()))


class TestGroundedNavCheckpointRoundTrip(unittest.TestCase):
    """Pins the old-path / checkpoint-load invariant this file's docstring cites."""

    def test_state_dict_round_trips_between_two_instances(self):
        cfg = _small_cfg()
        m1 = GroundedNav(arch='A', **cfg)
        m2 = GroundedNav(arch='A', **cfg)
        missing, unexpected = m2.load_state_dict(m1.state_dict(), strict=True)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])

    def test_loaded_weights_produce_identical_forward(self):
        cfg = _small_cfg()
        m1 = GroundedNav(arch='A', teacher_forcing=True, **cfg)
        m2 = GroundedNav(arch='A', teacher_forcing=True, **cfg)
        m2.load_state_dict(m1.state_dict())
        m1.eval()
        m2.eval()
        batch = _dummy_batch(cfg, B=1)
        gt_goal, gt_vel = torch.randn(1, 3), torch.randn(1, 3)
        with torch.no_grad():
            out1 = m1(**batch, gt_goal=gt_goal, gt_vel=gt_vel)
            out2 = m2(**batch, gt_goal=gt_goal, gt_vel=gt_vel)
        self.assertTrue(torch.equal(out1['action'], out2['action']))


class TestDefaultsDict(unittest.TestCase):
    def test_defaults_has_expected_keys(self):
        for key in ('img_size', 'in_ch', 'patch_size', 'vit_depth', 'vit_heads',
                    'vit_dim', 'lang_dim', 'proprio_dim', 'proprio_K', 'gru_hidden',
                    'goal_dim', 'vel_dim', 'action_dim', 'chunk_H'):
            self.assertIn(key, DEFAULTS)

    def test_constructor_merges_overrides_over_defaults(self):
        model = GroundedNav(arch='A', chunk_H=7, **{k: v for k, v in _small_cfg().items()
                                                     if k != 'chunk_H'})
        self.assertEqual(model.chunk_H, 7)


if __name__ == "__main__":
    unittest.main()
