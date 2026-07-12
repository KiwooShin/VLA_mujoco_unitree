"""Unit tests for code/perception/lock_rescan.py: ReacquisitionScan bounded
rescan wrapper."""
from __future__ import annotations

import unittest

from code.perception.lock_rescan import ReacquisitionScan, _RESCAN_TIMEOUT_STEPS


class TestReacquisitionScan(unittest.TestCase):

    def test_returns_float_wz_while_within_timeout(self):
        r = ReacquisitionScan(scan_rate=0.6)
        wz = r.step(current_yaw_rad=0.0)
        self.assertIsInstance(wz, float)

    def test_times_out_after_local_step_budget(self):
        r = ReacquisitionScan(scan_rate=0.6)
        last = None
        for _ in range(_RESCAN_TIMEOUT_STEPS):
            last = r.step(current_yaw_rad=0.0)
        # Exactly at the timeout boundary, still fine; the NEXT call must be None.
        timed_out = r.step(current_yaw_rad=0.0)
        self.assertIsNone(timed_out)

    def test_fresh_instance_has_its_own_local_counter(self):
        # A brand new ReacquisitionScan must be usable even if constructed
        # "late" in an episode -- i.e. its budget is always fresh (local step
        # counter starts at 0, independent of any global/episode step count).
        r1 = ReacquisitionScan()
        for _ in range(_RESCAN_TIMEOUT_STEPS):
            r1.step(0.0)
        self.assertIsNone(r1.step(0.0))
        r2 = ReacquisitionScan()
        # r2 must NOT be considered already-timed-out just because r1 is.
        self.assertIsNotNone(r2.step(0.0))

    def test_default_scan_rate_is_positive(self):
        r = ReacquisitionScan()
        wz = r.step(0.0)
        self.assertIsNotNone(wz)


if __name__ == "__main__":
    unittest.main()
