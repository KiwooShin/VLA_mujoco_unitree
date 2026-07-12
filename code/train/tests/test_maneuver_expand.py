"""Unit tests for code.train.maneuver: _expand_proprio_enc and load_loco_checkpoint.

_expand_proprio_enc is exercised against a lightweight duck-typed stand-in
for GroundedNav (it only ever touches `model.proprio_enc.gru`), keeping this
test decoupled from code.policy.small_vla's own internals. load_loco_checkpoint
is exercised against a real (tiny) GroundedNav + a real checkpoint file on
disk, since it constructs a real GroundedNav internally.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from code.train.maneuver import _expand_proprio_enc, load_loco_checkpoint, PROPRIO_DIM


class _FakeProprioEnc:
    def __init__(self, input_dim, hidden=8):
        self.gru = nn.GRU(input_dim, hidden, batch_first=True, num_layers=1)


class _FakeModel:
    def __init__(self, input_dim, hidden=8):
        self.proprio_enc = _FakeProprioEnc(input_dim, hidden)


class TestExpandProprioEnc(unittest.TestCase):
    def test_new_gru_has_expanded_input_size(self):
        model = _FakeModel(57, hidden=8)
        _expand_proprio_enc(model, old_dim=57, new_dim=62)
        self.assertEqual(model.proprio_enc.gru.input_size, 62)
        self.assertEqual(model.proprio_enc.gru.hidden_size, 8)

    def test_old_weight_columns_preserved_exactly(self):
        model = _FakeModel(57, hidden=8)
        old_weight = model.proprio_enc.gru.weight_ih_l0.detach().clone()
        old_hh = model.proprio_enc.gru.weight_hh_l0.detach().clone()
        old_bias_ih = model.proprio_enc.gru.bias_ih_l0.detach().clone()
        old_bias_hh = model.proprio_enc.gru.bias_hh_l0.detach().clone()

        _expand_proprio_enc(model, old_dim=57, new_dim=62)

        new_gru = model.proprio_enc.gru
        torch.testing.assert_close(new_gru.weight_ih_l0[:, :57], old_weight)
        torch.testing.assert_close(new_gru.weight_hh_l0, old_hh)
        torch.testing.assert_close(new_gru.bias_ih_l0, old_bias_ih)
        torch.testing.assert_close(new_gru.bias_hh_l0, old_bias_hh)

    def test_new_columns_are_not_simply_zero(self):
        """New input columns are orthogonal-init, not left at PyTorch's default
        (nonzero, small) init or zeroed -- just sanity-check they're not
        identically the old GRU's (out-of-range) values and have some spread."""
        model = _FakeModel(57, hidden=8)
        _expand_proprio_enc(model, old_dim=57, new_dim=62)
        new_cols = model.proprio_enc.gru.weight_ih_l0[:, 57:62]
        self.assertEqual(new_cols.shape, (24, 5))   # 3*hidden=24 rows
        self.assertGreater(new_cols.abs().sum().item(), 0.0)

    def test_forward_pass_works_after_expansion(self):
        model = _FakeModel(57, hidden=8)
        _expand_proprio_enc(model, old_dim=57, new_dim=62)
        x = torch.randn(2, 6, 62)
        _, h = model.proprio_enc.gru(x)
        self.assertEqual(h.shape, (1, 2, 8))


class TestLoadLocoCheckpoint(unittest.TestCase):
    def _make_loco_checkpoint(self, tmp_dir: str, proprio_dim=57) -> str:
        from code.small_vla import GroundedNav
        model = GroundedNav(arch='A', teacher_forcing=True, chunk_H=1, proprio_dim=proprio_dim)
        ckpt = {
            'epoch': 3,
            'arch': 'A',
            'chunk_H': 1,
            'proprio_dim': proprio_dim,
            'model_state': model.state_dict(),
        }
        path = str(Path(tmp_dir) / 'loco.pt')
        torch.save(ckpt, path)
        return path

    def test_expands_57d_checkpoint_to_maneuver_dim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_loco_checkpoint(tmp, proprio_dim=57)
            model, ckpt = load_loco_checkpoint(path, torch.device('cpu'))
            self.assertEqual(model.proprio_enc.gru.input_size, PROPRIO_DIM)
            self.assertEqual(ckpt['proprio_dim'], 57)

    def test_forward_pass_works_on_expanded_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_loco_checkpoint(tmp, proprio_dim=57)
            model, _ = load_loco_checkpoint(path, torch.device('cpu'))
            model.eval()
            with torch.no_grad():
                out = model(
                    ego_rgb=torch.zeros(1, 3, 128, 128),
                    lang_emb=torch.zeros(1, 2048),
                    proprio_h=torch.zeros(1, 6, PROPRIO_DIM),
                )
            self.assertIn('action', out)

    def test_no_expansion_needed_when_already_maneuver_dim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_loco_checkpoint(tmp, proprio_dim=PROPRIO_DIM)
            model, ckpt = load_loco_checkpoint(path, torch.device('cpu'))
            self.assertEqual(model.proprio_enc.gru.input_size, PROPRIO_DIM)


if __name__ == '__main__':
    unittest.main()
