"""Unit tests for code/perception/lock_mgmt.py: the aggregator re-export
surface (the module the old top-level code/lock_mgmt.py aliases to)."""
from __future__ import annotations

import unittest

import code.perception.lock_gate as lock_gate
import code.perception.lock_mgmt as lock_mgmt
import code.perception.lock_rescan as lock_rescan


class TestReExportIdentity(unittest.TestCase):
    """Every re-exported name must be the SAME object as its source-of-truth
    definition (not a copy/subclass) -- coherence across the split."""

    def test_lockgate_is_the_real_class(self):
        self.assertIs(lock_mgmt.LockGate, lock_gate.LockGate)

    def test_reacquisitionscan_is_the_real_class(self):
        self.assertIs(lock_mgmt.ReacquisitionScan, lock_rescan.ReacquisitionScan)

    def test_ang_diff_rad_is_the_real_function(self):
        self.assertIs(lock_mgmt._ang_diff_rad, lock_gate._ang_diff_rad)

    def test_toggle_constants_match(self):
        for name in ("LOCK_M1", "LOCK_M2", "LOCK_M3", "LOCK_M4", "LOCK_M5", "LOCK_M7"):
            self.assertEqual(getattr(lock_mgmt, name), getattr(lock_gate, name), msg=name)

    def test_m_constants_match(self):
        names = ["M1_AREA_FLOOR_PX2", "M2_CONFIRM_M", "M2_CONFIRM_N", "M2_TOL_DIST_M",
                "M2_TOL_BEARING_DEG", "M3_GATE_BEARING_DEG", "M3_GATE_BEARING_NEAR_MULT",
                "M3_NEAR_RANGE_M", "M3_GATE_DIST_FLOOR_M", "M3_GATE_DIST_CLOSING_MULT",
                "M3_EXPECTED_CLOSING_M_PER_CYCLE", "M3_INCUMBENT_MARGIN", "M3_INCUMBENT_K",
                "M4_WINDOW_N", "M4_TREND_MARGIN_M", "M4_EXEMPT_CYCLES_AFTER_CONFIRM",
                "M4_EXEMPT_CYCLES_AROUND_HANDOFF", "M7_X_WALK_M", "M7_K_MIN_FRAC",
                "M7_MIN_GOAL_DIST_M", "M7_PENALTY_CYCLES", "M7_PENALTY_BEARING_DEG",
                "M7_PENALTY_DIST_TOL_M", "M7_PENALTY_CONFIRM_M", "M7_PENALTY_CONFIRM_N",
                "M7_PENALTY_TOL_DIST_M", "M7_PENALTY_TOL_BEARING_DEG"]
        for name in names:
            self.assertEqual(getattr(lock_mgmt, name), getattr(lock_gate, name), msg=name)


class TestFunctionalUsageThroughAggregator(unittest.TestCase):
    """Sanity check that the aggregator's re-exported LockGate/ReacquisitionScan
    are fully functional when used exactly as the original callers do
    (`from code.lock_mgmt import LockGate, ReacquisitionScan`)."""

    def test_lockgate_usable_via_aggregator(self):
        g = lock_mgmt.LockGate()
        self.assertEqual(g.state, "NONE")
        g.gate_detection(dist=2.0, bearing_rad=0.0, area=None)
        # (area=None bypasses M1 regardless of its toggle state)

    def test_reacquisitionscan_usable_via_aggregator(self):
        r = lock_mgmt.ReacquisitionScan()
        wz = r.step(0.0)
        self.assertIsInstance(wz, float)

    def test_all_dunder_all_names_resolve(self):
        for name in lock_mgmt.__all__:
            self.assertTrue(hasattr(lock_mgmt, name), msg=f"missing {name}")


if __name__ == "__main__":
    unittest.main()
