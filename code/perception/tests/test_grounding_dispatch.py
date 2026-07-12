"""Unit tests for code/perception/grounding.py: ground() dispatch, instruction
parsing, and the module-owned GROUND_NET state singleton.

Uses monkeypatching of the module's own globals (GROUND_NET, _STATE) rather
than process re-exec/reload, since those are read as plain module-level names
at call time -- the same technique env-toggle-dependent code paths get
exercised with throughout this test package."""
from __future__ import annotations

import unittest

import numpy as np

import code.perception.grounding as G


class TestParseInstruction(unittest.TestCase):

    def test_basic_color_and_shape(self):
        color, shape = G._parse_instruction("go to the red ball")
        self.assertEqual(color, "red")
        self.assertEqual(shape, "ball")

    def test_whole_word_matching_avoids_substring_false_positive(self):
        # "red" must not match inside a longer word like "bored" or "credible".
        color, shape = G._parse_instruction("walk to the credible blue cube")
        self.assertEqual(color, "blue")
        self.assertEqual(shape, "cube")

    def test_no_match_returns_none_none(self):
        color, shape = G._parse_instruction("do a little dance")
        self.assertIsNone(color)
        self.assertIsNone(shape)

    def test_case_insensitive(self):
        color, shape = G._parse_instruction("GO TO THE GREEN CYLINDER")
        self.assertEqual(color, "green")
        self.assertEqual(shape, "cylinder")

    def test_first_matching_color_and_shape_win(self):
        # COLORS/SHAPES are scanned in their declared order; the instruction
        # text may contain only one of each in practice, but ensure a
        # single clean match still resolves both fields together.
        color, shape = G._parse_instruction("the orange cone is the goal")
        self.assertEqual(color, "orange")
        self.assertEqual(shape, "cone")


class TestGroundDispatch(unittest.TestCase):
    """Exercises ground()'s routing logic directly, with GROUND_NET forced via
    monkeypatched module globals (avoids depending on a real checkpoint file
    or subprocess re-exec to flip the env-toggle-derived constant)."""

    def setUp(self):
        self._orig_ground_net = G.GROUND_NET
        self._orig_state = G._STATE
        G._STATE = G._gn.GroundNetState()

    def tearDown(self):
        G.GROUND_NET = self._orig_ground_net
        G._STATE = self._orig_state

    def test_ground_net_off_routes_to_classical_and_notifies_once(self):
        G.GROUND_NET = False
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.full((8, 8), 3.0, dtype=np.float32)
        self.assertFalse(G._STATE.optout_notified)
        r = G.ground(rgb, depth, "chartreuse", "ball", {})   # unknown color -> not_visible, cheap
        self.assertTrue(r.not_visible)
        self.assertTrue(G._STATE.optout_notified)
        # Second call must not re-notify (one-shot) -- no observable error either way,
        # but the flag should stay True and not toggle back.
        G.ground(rgb, depth, "chartreuse", "ball", {})
        self.assertTrue(G._STATE.optout_notified)

    def test_ground_net_on_bad_checkpoint_sets_fallback_warned_and_uses_classical(self):
        G.GROUND_NET = True
        orig_ckpt = G.GROUND_NET_CKPT
        G.GROUND_NET_CKPT = "/nonexistent/checkpoint/model_best.pt"
        try:
            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            depth = np.full((8, 8), 3.0, dtype=np.float32)
            self.assertFalse(G._STATE.fallback_warned)
            r = G.ground(rgb, depth, "chartreuse", "ball", {})
            self.assertTrue(r.not_visible)   # unknown color, but must have gone through classical
            self.assertTrue(G._STATE.fallback_warned)
            self.assertTrue(G._STATE.load_failed)
        finally:
            G.GROUND_NET_CKPT = orig_ckpt

    def test_accessor_functions_delegate_to_state(self):
        self.assertIsNone(G.get_ground_net_last_heatmap())
        G._STATE.last_heatmap = dict(confidence=0.5)
        self.assertEqual(G.get_ground_net_last_heatmap(), dict(confidence=0.5))

        G._STATE.track_dist_m = 1.0
        G._STATE.track_bearing_rad = 0.5
        G.reset_ground_net_track()
        self.assertIsNone(G._STATE.track_dist_m)
        self.assertIsNone(G._STATE.track_bearing_rad)

        self.assertEqual(G.ground_net_latency_stats(), {})
        G._STATE.lat_ms.append(12.0)
        stats = G.ground_net_latency_stats()
        self.assertEqual(stats["n"], 1)


if __name__ == "__main__":
    unittest.main()
