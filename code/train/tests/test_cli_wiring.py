"""CLI flag-wiring tests for code.train.{dart_phase,maneuver,gaitfix} main(),
plus a cheap --help smoke for nx6_heatmap / nx6_heatmap_eval (whose main()
bodies aren't cleanly separable into a mockable "driver" function).

Per docs/refactor_plan.md invariant 2, every README command must keep working
verbatim. compute_action_stats/run_overfit_gate/train_full are monkeypatched
so no dataset I/O or real training happens; only the argv -> kwargs wiring
is under test.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import code.train.dart_phase as dart_phase_mod
import code.train.maneuver as maneuver_mod
import code.train.gaitfix as gaitfix_mod

_FAKE_STATS = {
    'mean': [0.0] * 15, 'std': [1.0] * 15,
    'default_angles': [0.0] * 15, 'n_frames': 100,
}


class _ArgvGuard:
    def __init__(self, argv):
        self._new = argv
        self._old = None

    def __enter__(self):
        self._old = sys.argv
        sys.argv = self._new
        return self

    def __exit__(self, *exc):
        sys.argv = self._old


class TestDartPhaseCli(unittest.TestCase):
    def test_flags_reach_train_full(self):
        captured = {}

        def fake_train_full(**kwargs):
            captured.update(kwargs)
            return []

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                'train_dart_phase.py', '--data', 'dataset/dart_combined_v2',
                '--out', tmp, '--epochs', '3', '--batch', '16',
                '--lr', '1e-4', '--swing-weight', '1.5', '--device', 'cpu',
            ]
            with mock.patch.object(dart_phase_mod, 'compute_action_stats', return_value=_FAKE_STATS), \
                 mock.patch.object(dart_phase_mod, 'train_full', fake_train_full), \
                 _ArgvGuard(argv):
                dart_phase_mod.main()

        self.assertEqual(captured['repo_path'], 'dataset/dart_combined_v2')
        self.assertEqual(captured['n_epochs'], 3)
        self.assertEqual(captured['batch_size'], 16)
        self.assertEqual(captured['lr'], 1e-4)
        self.assertEqual(captured['swing_weight'], 1.5)
        self.assertEqual(captured['arch'], 'A')
        self.assertFalse(captured['reset_epoch'])

    def test_reset_epoch_flag(self):
        captured = {}

        def fake_train_full(**kwargs):
            captured.update(kwargs)
            return []

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                'train_dart_phase.py', '--data', 'x', '--out', tmp,
                '--resume-ckpt', 'runs/x/best.pt', '--reset-epoch',
            ]
            with mock.patch.object(dart_phase_mod, 'compute_action_stats', return_value=_FAKE_STATS), \
                 mock.patch.object(dart_phase_mod, 'train_full', fake_train_full), \
                 _ArgvGuard(argv):
                dart_phase_mod.main()

        self.assertTrue(captured['reset_epoch'])
        self.assertEqual(captured['resume_ckpt'], 'runs/x/best.pt')

    def test_overfit_only_skips_train_full(self):
        def fake_train_full(**kwargs):
            self.fail("train_full must not be called with --overfit-only")

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(dart_phase_mod, 'compute_action_stats', return_value=_FAKE_STATS), \
             mock.patch.object(dart_phase_mod, 'run_overfit_gate',
                               return_value={'status': 'PASS', 'epoch': 1,
                                              'action_loss': 0.01, 'elapsed': 0.1}), \
             mock.patch.object(dart_phase_mod, 'train_full', fake_train_full), \
             _ArgvGuard(['train_dart_phase.py', '--data', 'x', '--out', tmp,
                         '--overfit', '--overfit-only']):
            dart_phase_mod.main()   # must not raise (train_full never called)


class TestManeuverTrainCli(unittest.TestCase):
    def test_multi_repo_data_flag_and_defaults(self):
        captured = {}

        def fake_train_full(**kwargs):
            captured.update(kwargs)
            return []

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                'train_maneuver.py', '--data', 'dataset/maneuver', 'dataset/dart_combined_v2',
                '--out', tmp, '--epochs', '2', '--lr', '5e-5',
            ]
            with mock.patch.object(maneuver_mod, 'compute_action_stats', return_value=_FAKE_STATS), \
                 mock.patch.object(maneuver_mod, 'train_full', fake_train_full), \
                 _ArgvGuard(argv):
                maneuver_mod.main()

        self.assertEqual(captured['repo_path'], ['dataset/maneuver', 'dataset/dart_combined_v2'])
        self.assertEqual(captured['n_epochs'], 2)
        self.assertEqual(captured['lr'], 5e-5)


class TestGaitfixCli(unittest.TestCase):
    def test_flags_reach_train_full(self):
        captured = {}

        def fake_train_full(**kwargs):
            captured.update(kwargs)
            return []

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                'train_gaitfix.py', '--data', 'dataset/easy_train80',
                '--out', tmp, '--epochs', '5', '--batch', '32',
                '--swing-weight', '2.5',
            ]
            with mock.patch.object(gaitfix_mod, 'compute_action_stats', return_value=_FAKE_STATS), \
                 mock.patch.object(gaitfix_mod, 'train_full', fake_train_full), \
                 _ArgvGuard(argv):
                gaitfix_mod.main()

        self.assertEqual(captured['repo_path'], 'dataset/easy_train80')
        self.assertEqual(captured['n_epochs'], 5)
        self.assertEqual(captured['batch_size'], 32)
        self.assertEqual(captured['swing_weight'], 2.5)

    def test_overfit_gate_warns_but_continues_on_fail(self):
        def fake_train_full(**kwargs):
            return []

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(gaitfix_mod, 'compute_action_stats', return_value=_FAKE_STATS), \
             mock.patch.object(gaitfix_mod, 'run_overfit_gate',
                               return_value={'status': 'FAIL', 'epoch': 300,
                                              'action_loss': 5.0, 'elapsed': 1.0}), \
             mock.patch.object(gaitfix_mod, 'train_full', fake_train_full), \
             _ArgvGuard(['train_gaitfix.py', '--data', 'x', '--out', tmp,
                         '--overfit']):
            gaitfix_mod.main()   # must not raise even though gate FAILed


class TestNx6HeatmapHelpSmoke(unittest.TestCase):
    """Cheap argparse-config smoke: --help must exit 0 and not crash, for both
    the training entry and the standalone-eval entry (nx6_heatmap_eval's main
    isn't separable into a mockable driver -- see code/train/nx6_heatmap_eval.py)."""

    def _help_exit_code(self, module_path: str) -> int:
        result = subprocess.run(
            [sys.executable, '-c', f'import {module_path} as m; m.main()', '--help'],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode

    def test_train_nx6_heatmap_help(self):
        code = self._help_exit_code('code.train.nx6_heatmap')
        self.assertEqual(code, 0)

    def test_nx6_heatmap_eval_help(self):
        code = self._help_exit_code('code.train.nx6_heatmap_eval')
        self.assertEqual(code, 0)


if __name__ == '__main__':
    unittest.main()
