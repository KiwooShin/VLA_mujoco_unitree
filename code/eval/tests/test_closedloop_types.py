"""Unit tests for code.eval.closedloop: EpisodeResult, MAXSTEPS, and the
pure-logic helpers (_print_table, _write_log) that don't need a live sim.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path

from code.eval.closedloop import (
    EpisodeResult, MAXSTEPS, EVAL_SEED, N_RENDER_EPS,
    _print_table, _write_log,
)


class TestMaxstepsAndConstants(unittest.TestCase):
    def test_maxsteps_presets(self):
        self.assertEqual(MAXSTEPS['easy'], 600)
        self.assertEqual(MAXSTEPS['demo'], 1700)

    def test_eval_seed_is_held_out_999(self):
        self.assertEqual(EVAL_SEED, 999)

    def test_n_render_eps(self):
        self.assertEqual(N_RENDER_EPS, 3)


class TestEpisodeResult(unittest.TestCase):
    def _make(self, **overrides) -> EpisodeResult:
        base = dict(
            ep_idx=0, instruction='go to the red ball', target_color='red',
            target_shape='ball', target_dist=2.0, success=True,
            failure_tag='success', steps=300, final_dist=0.1, fell=False,
            ms_per_step=3.4, goal_source='classical',
        )
        base.update(overrides)
        return EpisodeResult(**base)

    def test_defaults(self):
        r = self._make()
        self.assertEqual(r.vel_source, 'predicted')
        self.assertEqual(r.action_osc_std, 0.0)
        self.assertEqual(r.forward_disp, 0.0)
        self.assertIsNone(r.video_path)

    def test_asdict_is_json_serializable_plain_dict(self):
        r = self._make(video_path='eval/ep0000_archA.mp4')
        d = dataclasses.asdict(r)
        self.assertIsInstance(d, dict)
        json.dumps(d)   # must not raise
        self.assertEqual(d['video_path'], 'eval/ep0000_archA.mp4')


class TestPrintTable(unittest.TestCase):
    def test_prints_one_row_per_result_without_raising(self):
        results = [
            EpisodeResult(
                ep_idx=i, instruction=f'go to the target number {i} over there please' * 2,
                target_color='red',
                target_shape='ball', target_dist=1.0 + i, success=(i % 2 == 0),
                failure_tag=('success' if i % 2 == 0 else 'fall'), steps=100 * i,
                final_dist=0.05 * i, fell=(i % 2 != 0), ms_per_step=2.0,
                goal_source='classical',
            )
            for i in range(3)
        ]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_table(results)
        out = buf.getvalue()
        self.assertIn('SUCCESS', out)
        self.assertIn('FAIL[fall]', out)
        # Long instructions get truncated with '..'
        self.assertIn('..', out)

    def test_empty_results_does_not_raise(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_table([])
        self.assertIn('outcome', buf.getvalue())   # header still printed


class TestWriteLog(unittest.TestCase):
    def test_writes_one_json_line_per_result(self):
        results = [
            EpisodeResult(
                ep_idx=0, instruction='go to the blue cube', target_color='blue',
                target_shape='cube', target_dist=1.5, success=True,
                failure_tag='success', steps=200, final_dist=0.2, fell=False,
                ms_per_step=3.0, goal_source='classical',
            ),
            EpisodeResult(
                ep_idx=1, instruction='go to the green cone', target_color='green',
                target_shape='cone', target_dist=2.5, success=False,
                failure_tag='fall', steps=50, final_dist=1.9, fell=True,
                ms_per_step=3.0, goal_source='classical',
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _write_log(results, tmp, arch='A', goal_source='classical', difficulty='easy')
            log_path = Path(tmp) / 'eval_log_archA_classical_easy.jsonl'
            self.assertTrue(log_path.exists())
            lines = log_path.read_text().strip().split('\n')
            self.assertEqual(len(lines), 2)
            row0 = json.loads(lines[0])
            self.assertEqual(row0['ep_idx'], 0)
            self.assertEqual(row0['target_color'], 'blue')

    def test_overwrites_on_each_call(self):
        """_write_log is called every episode with the growing results list --
        it must overwrite (not append to) the file each time."""
        r = EpisodeResult(
            ep_idx=0, instruction='x', target_color='red', target_shape='ball',
            target_dist=1.0, success=True, failure_tag='success', steps=1,
            final_dist=0.0, fell=False, ms_per_step=1.0, goal_source='classical',
        )
        with tempfile.TemporaryDirectory() as tmp:
            _write_log([r], tmp, 'A', 'classical', 'easy')
            _write_log([r, r], tmp, 'A', 'classical', 'easy')
            log_path = Path(tmp) / 'eval_log_archA_classical_easy.jsonl'
            lines = log_path.read_text().strip().split('\n')
            self.assertEqual(len(lines), 2)


if __name__ == '__main__':
    unittest.main()
