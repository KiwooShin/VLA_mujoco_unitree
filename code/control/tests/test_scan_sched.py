"""Unit tests for code.control.scan_sched (RF-1).

Pins the NX-1 bidirectional bounded-rotation scan schedule: leg progression,
dwell insertion, the CCW/CW/CW/CCW sign pattern, self-correcting accumulated-
yaw tracking (vs. assumed step counts), and yaw-wraparound robustness.
"""

from __future__ import annotations

import math
import unittest

from code.control import scan_sched as SS


def _drive(sched: SS.BidirectionalScanSchedule, n_steps: int, dt: float = 1.0,
           start_yaw: float = 0.0) -> tuple[list[float], list[int], list[bool]]:
    """Simulate `n_steps` of perfect-tracking rotation (realized wz == commanded wz).

    Returns (wz_trace, leg_idx_trace, dwelling_trace).
    """
    yaw = start_yaw
    wz_trace, leg_trace, dwell_trace = [], [], []
    for _ in range(n_steps):
        wz = sched.step(yaw)
        wz_trace.append(wz)
        leg_trace.append(sched.leg_idx)
        dwell_trace.append(sched.is_dwelling)
        yaw = math.atan2(math.sin(yaw + wz * dt), math.cos(yaw + wz * dt))
    return wz_trace, leg_trace, dwell_trace


class TestBidirectionalScanScheduleBasics(unittest.TestCase):
    """Constructor defaults and the leg_idx/is_dwelling read-only properties."""

    def test_defaults(self):
        sched = SS.BidirectionalScanSchedule()
        self.assertEqual(sched.leg_idx, 0)
        self.assertFalse(sched.is_dwelling)
        self.assertEqual(sched.scan_rate, 0.6)
        self.assertEqual(sched.leg_deg, SS.SCAN_LEG_DEG)
        self.assertEqual(sched.dwell_steps, SS.SCAN_DWELL_STEPS)

    def test_custom_params(self):
        sched = SS.BidirectionalScanSchedule(scan_rate=1.0, leg_deg=90.0, dwell_steps=5)
        self.assertEqual(sched.scan_rate, 1.0)
        self.assertEqual(sched.leg_deg, 90.0)
        self.assertEqual(sched.dwell_steps, 5)

    def test_first_call_does_not_crash_with_no_prior_yaw(self):
        sched = SS.BidirectionalScanSchedule()
        wz = sched.step(0.0)
        self.assertEqual(wz, sched.scan_rate)  # leg 0 sign is +1 (CCW)


class TestLegProgressionAndSigns(unittest.TestCase):
    """Leg index advances through the CCW, CW, CW, CCW, ... pattern."""

    def test_first_leg_is_ccw_positive(self):
        sched = SS.BidirectionalScanSchedule(scan_rate=0.6, leg_deg=30.0, dwell_steps=3)
        wz = sched.step(0.0)
        self.assertAlmostEqual(wz, 0.6)

    def test_leg_completes_and_dwells(self):
        # scan_rate rad/s at dt=1s per step => leg_deg reached quickly with a small leg.
        # Note: accumulated yaw only starts incrementing from the 2nd call onward
        # (the 1st call has no prior yaw reading to diff against), so reaching
        # 30 deg at 10 deg/step takes 4 calls, not 3.
        sched = SS.BidirectionalScanSchedule(scan_rate=math.radians(10), leg_deg=30.0, dwell_steps=4)
        wz_trace, leg_trace, dwell_trace = _drive(sched, 5)
        self.assertTrue(any(dwell_trace), f"never entered dwell: {dwell_trace}")

    def test_dwell_holds_zero_wz_for_dwell_steps(self):
        sched = SS.BidirectionalScanSchedule(scan_rate=math.radians(20), leg_deg=20.0, dwell_steps=5)
        wz_trace, leg_trace, dwell_trace = _drive(sched, 20)
        dwell_wz = [w for w, d in zip(wz_trace, dwell_trace) if d]
        self.assertTrue(dwell_wz, "expected at least one dwelling step")
        self.assertTrue(all(w == 0.0 for w in dwell_wz))

    def test_sign_pattern_ccw_cw_cw_ccw(self):
        # Tiny leg_deg + short dwell so many legs complete within a short trace.
        sched = SS.BidirectionalScanSchedule(scan_rate=math.radians(30), leg_deg=15.0, dwell_steps=2)
        wz_trace, leg_trace, dwell_trace = _drive(sched, 60)
        # Collect the sign of the first genuinely-moving (wz != 0) sample seen for
        # each leg index, in order. wz == 0 is ambiguous: it is either a true dwell
        # step OR the one-step-lagged "just exited dwell" transition step (which
        # already reports the new leg_idx / is_dwelling=False but still returns
        # 0.0 for that single step) -- so filter on wz, not on is_dwelling.
        seen_signs = {}
        for wz, leg in zip(wz_trace, leg_trace):
            if wz != 0.0 and leg not in seen_signs:
                seen_signs[leg] = 1 if wz > 0 else -1
        expected = {0: 1, 1: -1, 2: -1, 3: 1}
        self.assertGreaterEqual(len(seen_signs), 3, f"too few legs observed: {seen_signs}")
        for leg, sign in expected.items():
            if leg in seen_signs:
                self.assertEqual(seen_signs[leg], sign, f"leg {leg} sign mismatch")

    def test_leg_idx_monotonically_nondecreasing(self):
        sched = SS.BidirectionalScanSchedule(scan_rate=math.radians(25), leg_deg=15.0, dwell_steps=3)
        _, leg_trace, _ = _drive(sched, 80)
        self.assertEqual(leg_trace, sorted(leg_trace))


class TestAccumulatedYawTracking(unittest.TestCase):
    """Self-correction: tracks REALIZED yaw deltas, not assumed step counts."""

    def test_slower_than_nominal_realized_rate_delays_leg_completion(self):
        # Command a small per-step rotation relative to leg_deg (fine resolution,
        # so the +/-1-step quantization noise is negligible), but only a fraction
        # of it is REALIZED each step (simulate policy tracking lag) -> the leg
        # should take proportionally longer to complete, since the schedule
        # integrates actual yaw deltas rather than assuming steps * nominal_rate.
        commanded_rate = math.radians(3.0)  # ~3 deg/step at dt=1s
        leg_deg = 30.0
        sched_full = SS.BidirectionalScanSchedule(scan_rate=commanded_rate, leg_deg=leg_deg, dwell_steps=1000)
        sched_half = SS.BidirectionalScanSchedule(scan_rate=commanded_rate, leg_deg=leg_deg, dwell_steps=1000)

        def steps_to_dwell(sched, realized_fraction):
            yaw = 0.0
            for i in range(1, 2000):
                wz = sched.step(yaw)
                if sched.is_dwelling:
                    return i
                yaw = yaw + wz * realized_fraction
            raise AssertionError("never reached dwell")

        n_full = steps_to_dwell(sched_full, realized_fraction=1.0)
        n_half = steps_to_dwell(sched_half, realized_fraction=0.5)
        self.assertGreater(n_half, n_full * 1.5)

    def test_yaw_wraparound_does_not_break_accumulation(self):
        # Start right near the +pi/-pi seam; the schedule integrates deltas via
        # atan2(sin(dyaw), cos(dyaw)) so it must not see a spurious ~2pi jump.
        sched = SS.BidirectionalScanSchedule(scan_rate=math.radians(15), leg_deg=20.0, dwell_steps=3)
        start_yaw = math.pi - math.radians(5)  # a couple of steps will cross the seam
        wz_trace, leg_trace, dwell_trace = _drive(sched, 10, start_yaw=start_yaw)
        # No assertion error / exception is itself the main pin; also sanity-check
        # that we didn't immediately jump straight to a late leg (which would
        # indicate the wraparound was misread as ~360 degrees of rotation).
        self.assertLessEqual(max(leg_trace), 2)


class TestModuleConstants(unittest.TestCase):
    """Pin the documented default constants (used by eval_search/fancy_demo/demo)."""

    def test_leg_deg_is_165(self):
        self.assertEqual(SS.SCAN_LEG_DEG, 165.0)

    def test_dwell_steps_is_45(self):
        self.assertEqual(SS.SCAN_DWELL_STEPS, 45)

    def test_timeout_is_1150(self):
        self.assertEqual(SS.SCAN_TIMEOUT, 1150)

    def test_leg_signs_pattern(self):
        self.assertEqual(SS._LEG_SIGNS, (1, -1, -1, 1))


if __name__ == "__main__":
    unittest.main()
