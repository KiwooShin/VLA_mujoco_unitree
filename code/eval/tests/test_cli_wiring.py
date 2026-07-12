"""CLI flag-wiring tests for code.eval.{closedloop,search,maneuver}.

Per docs/refactor_plan.md invariant 2, every README command must keep working
verbatim. These tests monkeypatch each module's heavy driver function
(evaluate / evaluate_search / evaluate_maneuver) so no simulation/GPU work
actually runs, then drive main() with a crafted argv and assert every CLI
flag reached the driver with the expected value -- this pins the flag <->
kwarg mapping without needing a real dataset or checkpoint.
"""

from __future__ import annotations

import sys
import unittest
from unittest import mock

import code.eval.closedloop as closedloop_mod
import code.eval.search as search_mod
import code.eval.maneuver as maneuver_mod


class _ArgvGuard:
    """Context manager that restores sys.argv on exit."""
    def __init__(self, argv):
        self._new = argv
        self._old = None

    def __enter__(self):
        self._old = sys.argv
        sys.argv = self._new
        return self

    def __exit__(self, *exc):
        sys.argv = self._old


class TestClosedloopCli(unittest.TestCase):
    def test_default_flags(self):
        captured = {}

        def fake_evaluate(**kwargs):
            captured.update(kwargs)
            return {}

        with mock.patch.object(closedloop_mod, 'evaluate', fake_evaluate), \
             mock.patch.object(sys, 'exit'), \
             _ArgvGuard(['eval_closedloop.py', '--checkpoint', 'runs/x/best.pt']):
            closedloop_mod.main()

        self.assertEqual(captured['checkpoint_path'], 'runs/x/best.pt')
        self.assertEqual(captured['arch'], 'A')
        self.assertEqual(captured['difficulty'], 'easy')
        self.assertEqual(captured['n_scenes'], 15)
        self.assertEqual(captured['device'], 'cpu')
        self.assertEqual(captured['render'], True)
        self.assertEqual(captured['goal_source'], 'classical')
        self.assertEqual(captured['vel_source'], 'predicted')
        self.assertEqual(captured['seed'], 999)

    def test_non_default_flags(self):
        captured = {}

        def fake_evaluate(**kwargs):
            captured.update(kwargs)
            return {}

        argv = [
            'eval_closedloop.py',
            '--checkpoint', 'runs/demo_dart_A/epoch_0003.pt',
            '--arch', 'C',
            '--difficulty', 'demo',
            '--n', '7',
            '--device', 'cuda',
            '--out', 'eval/custom',
            '--no-render',
            '--goal-source', 'gt',
            '--vel-source', 'gt',
            '--seed', '123',
        ]
        with mock.patch.object(closedloop_mod, 'evaluate', fake_evaluate), \
             mock.patch.object(sys, 'exit'), \
             _ArgvGuard(argv):
            closedloop_mod.main()

        self.assertEqual(captured['arch'], 'C')
        self.assertEqual(captured['difficulty'], 'demo')
        self.assertEqual(captured['n_scenes'], 7)
        self.assertEqual(captured['device'], 'cuda')
        self.assertEqual(captured['out_dir'], 'eval/custom')
        self.assertEqual(captured['render'], False)
        self.assertEqual(captured['goal_source'], 'gt')
        self.assertEqual(captured['vel_source'], 'gt')
        self.assertEqual(captured['seed'], 123)

    def test_exits_zero_after_eval(self):
        with mock.patch.object(closedloop_mod, 'evaluate', lambda **kw: {}), \
             mock.patch.object(sys, 'exit') as mock_exit, \
             _ArgvGuard(['eval_closedloop.py', '--smoke']):
            closedloop_mod.main()
        mock_exit.assert_called_once_with(0)


class TestSearchCli(unittest.TestCase):
    def test_default_flags(self):
        captured = {}

        def fake_evaluate_search(**kwargs):
            captured.update(kwargs)
            return {}

        with mock.patch.object(search_mod, 'evaluate_search', fake_evaluate_search), \
             _ArgvGuard(['eval_search.py']):
            search_mod.main()

        self.assertIsNone(captured['checkpoint_path'])
        self.assertEqual(captured['n_scenes'], 15)
        self.assertEqual(captured['device'], 'cpu')
        self.assertEqual(captured['out_dir'], 'eval/search')
        self.assertEqual(captured['render_video'], True)
        self.assertEqual(captured['smoke'], False)
        self.assertEqual(captured['seed'], 999)

    def test_non_default_flags(self):
        captured = {}

        def fake_evaluate_search(**kwargs):
            captured.update(kwargs)
            return {}

        argv = [
            'eval_search.py',
            '--checkpoint', 'checkpoint/goto_best.pt',
            '--n', '3',
            '--device', 'cuda',
            '--out', 'eval/search_custom',
            '--smoke',
            '--no-video',
            '--seed', '42',
        ]
        with mock.patch.object(search_mod, 'evaluate_search', fake_evaluate_search), \
             _ArgvGuard(argv):
            search_mod.main()

        self.assertEqual(captured['checkpoint_path'], 'checkpoint/goto_best.pt')
        self.assertEqual(captured['n_scenes'], 3)
        self.assertEqual(captured['device'], 'cuda')
        self.assertEqual(captured['out_dir'], 'eval/search_custom')
        self.assertEqual(captured['render_video'], False)
        self.assertEqual(captured['smoke'], True)
        self.assertEqual(captured['seed'], 42)


class TestManeuverCli(unittest.TestCase):
    def test_default_flags_hybrid_vel_on_by_default(self):
        captured = {}

        def fake_evaluate_maneuver(**kwargs):
            captured.update(kwargs)
            return {}

        with mock.patch.object(maneuver_mod, 'evaluate_maneuver', fake_evaluate_maneuver), \
             mock.patch.object(sys, 'exit'), \
             _ArgvGuard(['eval_maneuver.py', '--checkpoint', 'runs/maneuver_A/epoch_0002.pt']):
            maneuver_mod.main()

        self.assertEqual(captured['checkpoint_path'], 'runs/maneuver_A/epoch_0002.pt')
        self.assertEqual(captured['n_scenes'], 15)
        self.assertEqual(captured['seed'], 999)
        self.assertEqual(captured['device_str'], 'cpu')
        self.assertEqual(captured['render_n'], 3)
        self.assertEqual(captured['smoke'], False)
        self.assertEqual(captured['smoke_steps'], 150)
        self.assertEqual(captured['free_vel'], False)
        # hybrid_vel is ON by default per the CLI docstring/README.
        self.assertEqual(captured['hybrid_vel'], True)

    def test_free_vel_disables_hybrid(self):
        captured = {}

        def fake_evaluate_maneuver(**kwargs):
            captured.update(kwargs)
            return {}

        argv = ['eval_maneuver.py', '--checkpoint', 'x.pt', '--free-vel']
        with mock.patch.object(maneuver_mod, 'evaluate_maneuver', fake_evaluate_maneuver), \
             mock.patch.object(sys, 'exit'), \
             _ArgvGuard(argv):
            maneuver_mod.main()

        self.assertEqual(captured['free_vel'], True)
        self.assertEqual(captured['hybrid_vel'], False)

    def test_no_hybrid_vel_flag_disables_hybrid(self):
        captured = {}

        def fake_evaluate_maneuver(**kwargs):
            captured.update(kwargs)
            return {}

        argv = ['eval_maneuver.py', '--checkpoint', 'x.pt', '--no-hybrid-vel']
        with mock.patch.object(maneuver_mod, 'evaluate_maneuver', fake_evaluate_maneuver), \
             mock.patch.object(sys, 'exit'), \
             _ArgvGuard(argv):
            maneuver_mod.main()

        self.assertEqual(captured['free_vel'], False)
        self.assertEqual(captured['hybrid_vel'], False)

    def test_missing_required_checkpoint_raises_systemexit(self):
        with _ArgvGuard(['eval_maneuver.py']):
            with self.assertRaises(SystemExit):
                maneuver_mod.main()


if __name__ == '__main__':
    unittest.main()
