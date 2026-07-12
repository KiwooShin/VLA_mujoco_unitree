"""Unit tests for code/perception/lock_gate.py: NX-2/NX-5 LockGate state
machine + M1-M7 toggles.

Each mechanism's toggle (LOCK_M1..LOCK_M7) is read as a bare module-level
name at call time inside gate_detection()/end_of_cycle()/coast_expired(), so
tests monkeypatch `lock_gate.LOCK_Mx` directly (save/restore per test) to
exercise both the on and off code paths deterministically, independent of
whatever env vars happen to be set in the process."""
from __future__ import annotations

import math
import unittest

import code.perception.lock_gate as lock_gate
from code.perception.lock_gate import LockGate, _ang_diff_rad


class _ToggleGuard(unittest.TestCase):
    """Base class saving/restoring every LOCK_Mx toggle around each test."""

    TOGGLE_NAMES = ("LOCK_M1", "LOCK_M2", "LOCK_M3", "LOCK_M4", "LOCK_M5", "LOCK_M7")

    def setUp(self):
        self._saved = {name: getattr(lock_gate, name) for name in self.TOGGLE_NAMES}

    def tearDown(self):
        for name, val in self._saved.items():
            setattr(lock_gate, name, val)

    def _set(self, **kwargs):
        for name, val in kwargs.items():
            setattr(lock_gate, name, val)


class TestAngDiffRad(unittest.TestCase):

    def test_zero_difference(self):
        self.assertAlmostEqual(_ang_diff_rad(1.0, 1.0), 0.0)

    def test_wraps_to_pi_range(self):
        d = _ang_diff_rad(math.pi - 0.1, -math.pi + 0.1)
        self.assertLessEqual(abs(d), math.pi)
        self.assertAlmostEqual(d, -0.2, places=6)

    def test_sign_convention(self):
        self.assertGreater(_ang_diff_rad(0.5, 0.1), 0.0)
        self.assertLess(_ang_diff_rad(0.1, 0.5), 0.0)


class TestDefaults(unittest.TestCase):
    """Pins the shipped defaults documented in the module docstring, given no
    LOCK_Mx env vars are set in this process."""

    def test_m1_m3_default_on_others_default_off(self):
        # NOTE: reflects the toggles AS READ AT IMPORT TIME for this process.
        self.assertTrue(lock_gate.LOCK_M1)
        self.assertTrue(lock_gate.LOCK_M3)
        self.assertFalse(lock_gate.LOCK_M2)
        self.assertFalse(lock_gate.LOCK_M4)
        self.assertFalse(lock_gate.LOCK_M5)
        self.assertFalse(lock_gate.LOCK_M7)


class TestAllOffLegacyPassthrough(_ToggleGuard):
    """With M1/M2/M3/M4/M5/M7 all off, gate_detection is a provable
    pass-through and end_of_cycle never triggers (docs/nx2_impl.md)."""

    def setUp(self):
        super().setUp()
        self._set(LOCK_M1=False, LOCK_M2=False, LOCK_M3=False, LOCK_M4=False,
                 LOCK_M5=False, LOCK_M7=False)

    def test_first_detection_confirms_immediately(self):
        g = LockGate()
        self.assertEqual(g.state, "NONE")
        accepted = g.gate_detection(dist=3.0, bearing_rad=0.1, area=1.0)
        self.assertTrue(accepted)
        self.assertEqual(g.state, "CONFIRMED")

    def test_every_subsequent_detection_accepted_regardless_of_jump(self):
        g = LockGate()
        g.gate_detection(1.0, 0.0, 10.0)
        # A huge bearing/distance jump would normally trip M3 -- must still pass with M3 off.
        accepted = g.gate_detection(dist=20.0, bearing_rad=3.0, area=0.001)
        self.assertTrue(accepted)

    def test_end_of_cycle_never_triggers(self):
        g = LockGate()
        g.gate_detection(5.0, 0.0, 10.0)
        for _ in range(50):
            triggered = g.end_of_cycle(best_dist_estimate=5.0 + 1.0, walking=True, proj_disp_m=0.0)
            self.assertFalse(triggered)

    def test_coast_expired_always_false(self):
        g = LockGate()
        self.assertFalse(g.coast_expired(frames_since_detection=1000, hold_goal_horizon=1))


class TestM1AreaFloor(_ToggleGuard):

    def setUp(self):
        super().setUp()
        self._set(LOCK_M1=True, LOCK_M2=False, LOCK_M3=False)

    def test_area_below_floor_rejected(self):
        g = LockGate()
        accepted = g.gate_detection(dist=2.0, bearing_rad=0.0, area=50.0)  # < M1_AREA_FLOOR_PX2=100
        self.assertFalse(accepted)
        self.assertEqual(g.state, "NONE")

    def test_area_at_or_above_floor_accepted(self):
        g = LockGate()
        accepted = g.gate_detection(dist=2.0, bearing_rad=0.0, area=100.0)
        self.assertTrue(accepted)

    def test_none_area_bypasses_floor(self):
        # GROUND_NET detections carry area=None -- M1 must be a no-op for them.
        g = LockGate()
        accepted = g.gate_detection(dist=2.0, bearing_rad=0.0, area=None)
        self.assertTrue(accepted)


class TestM2NofM(_ToggleGuard):

    def setUp(self):
        super().setUp()
        self._set(LOCK_M1=False, LOCK_M2=True, LOCK_M3=False)

    def test_single_detection_not_yet_confirmed(self):
        g = LockGate()
        accepted = g.gate_detection(2.0, 0.0, 10.0)
        self.assertFalse(accepted)
        self.assertEqual(g.state, "NONE")

    def test_two_consistent_of_three_confirms(self):
        g = LockGate()
        g.gate_detection(2.0, 0.0, 10.0)
        accepted = g.gate_detection(2.05, 0.01, 10.0)   # consistent with the first
        self.assertTrue(accepted)
        self.assertEqual(g.state, "CONFIRMED")

    def test_inconsistent_detections_never_confirm(self):
        g = LockGate()
        g.gate_detection(2.0, 0.0, 10.0)
        g.gate_detection(8.0, 2.0, 10.0)   # wildly different -- not consistent
        accepted = g.gate_detection(15.0, -2.0, 10.0)
        self.assertFalse(accepted)
        self.assertEqual(g.state, "NONE")


class TestM3InnovationGate(_ToggleGuard):

    def setUp(self):
        super().setUp()
        self._set(LOCK_M1=False, LOCK_M2=False, LOCK_M3=True)

    def test_small_innovation_within_gate_accepted(self):
        g = LockGate()
        g.gate_detection(5.0, 0.0, 100.0)   # confirms immediately (M2 off)
        accepted = g.gate_detection(5.1, 0.02, 100.0)   # small change
        self.assertTrue(accepted)

    def test_large_innovation_rejected_without_sustained_quality_margin(self):
        g = LockGate()
        g.gate_detection(5.0, 0.0, 100.0)
        # Big bearing jump, and challenger area does NOT beat incumbent margin.
        accepted = g.gate_detection(5.0, math.radians(80.0), 100.0)
        self.assertFalse(accepted)

    def test_large_innovation_accepted_after_k_consecutive_quality_challenges(self):
        g = LockGate()
        g.gate_detection(5.0, 0.0, 100.0)
        big_bearing = math.radians(80.0)
        strong_area = 100.0 * 1.3 + 1.0   # >= M3_INCUMBENT_MARGIN * incumbent
        accepted1 = g.gate_detection(5.0, big_bearing, strong_area)
        self.assertFalse(accepted1)   # first challenger cycle (K=2 required)
        accepted2 = g.gate_detection(5.0, big_bearing, strong_area)
        self.assertTrue(accepted2)    # second consecutive -> replaces incumbent

    def test_challenger_streak_resets_on_non_qualifying_cycle(self):
        g = LockGate()
        g.gate_detection(5.0, 0.0, 100.0)
        big_bearing = math.radians(80.0)
        strong_area = 100.0 * 1.3 + 1.0
        g.gate_detection(5.0, big_bearing, strong_area)   # streak=1
        g.gate_detection(5.0, big_bearing, 1.0)            # weak area -> resets streak
        accepted = g.gate_detection(5.0, big_bearing, strong_area)  # streak=1 again, not enough
        self.assertFalse(accepted)

    def test_discontinuity_bypass_accepts_unconditionally(self):
        g = LockGate()
        g.gate_detection(5.0, 0.0, 100.0)
        g.mark_discontinuity()
        # Even a wild jump with weak area must be accepted during the cooldown.
        accepted = g.gate_detection(50.0, math.radians(170.0), 0.001)
        self.assertTrue(accepted)


class TestM4DivergenceWatchdog(_ToggleGuard):

    def setUp(self):
        super().setUp()
        self._set(LOCK_M1=False, LOCK_M2=False, LOCK_M3=False, LOCK_M4=True)

    def test_monotonic_growth_beyond_margin_triggers(self):
        g = LockGate()
        g.gate_detection(3.0, 0.0, 10.0)
        # Burn through the post-confirm exemption window first.
        for _ in range(lock_gate.M4_EXEMPT_CYCLES_AFTER_CONFIRM + 1):
            g.end_of_cycle(best_dist_estimate=3.0, walking=True, proj_disp_m=0.0)
        triggered = False
        dist = 3.0
        for _ in range(lock_gate.M4_WINDOW_N + 2):
            dist += 0.2
            triggered = g.end_of_cycle(best_dist_estimate=dist, walking=True, proj_disp_m=0.0)
            if triggered:
                break
        self.assertTrue(triggered)
        self.assertEqual(g.last_trigger, "M4")

    def test_recently_confirmed_exempt(self):
        g = LockGate()
        g.gate_detection(3.0, 0.0, 10.0)
        # Immediately spike the distance within the post-confirm exemption window.
        triggered = g.end_of_cycle(best_dist_estimate=30.0, walking=True, proj_disp_m=0.0)
        self.assertFalse(triggered)

    def test_not_walking_never_triggers(self):
        g = LockGate()
        g.gate_detection(3.0, 0.0, 10.0)
        for _ in range(lock_gate.M4_WINDOW_N + lock_gate.M4_EXEMPT_CYCLES_AFTER_CONFIRM + 5):
            triggered = g.end_of_cycle(best_dist_estimate=100.0, walking=False, proj_disp_m=0.0)
            self.assertFalse(triggered)

    def test_discontinuity_cooldown_suppresses_trigger(self):
        g = LockGate()
        g.gate_detection(3.0, 0.0, 10.0)
        for _ in range(lock_gate.M4_EXEMPT_CYCLES_AFTER_CONFIRM + 1):
            g.end_of_cycle(best_dist_estimate=3.0, walking=True, proj_disp_m=0.0)
        g.mark_discontinuity(cooldown_cycles=1000)
        triggered = g.end_of_cycle(best_dist_estimate=100.0, walking=True, proj_disp_m=0.0)
        self.assertFalse(triggered)


class TestM5CoastExpiry(_ToggleGuard):

    def test_off_never_expires(self):
        self._set(LOCK_M5=False)
        g = LockGate()
        self.assertFalse(g.coast_expired(1000, 1))

    def test_on_expires_past_horizon(self):
        self._set(LOCK_M5=True)
        g = LockGate()
        self.assertFalse(g.coast_expired(5, 10))
        self.assertTrue(g.coast_expired(11, 10))


class TestM7OdometryCoherence(_ToggleGuard):

    def setUp(self):
        super().setUp()
        self._set(LOCK_M1=False, LOCK_M2=False, LOCK_M3=False, LOCK_M4=False, LOCK_M7=True)

    def test_coherent_approach_never_triggers(self):
        g = LockGate()
        g.gate_detection(10.0, 0.0, 10.0)
        dist = 10.0
        for _ in range(20):
            # Walk 0.2m/cycle toward the target with a commensurate distance shrink.
            dist -= 0.2
            triggered = g.end_of_cycle(best_dist_estimate=dist, walking=True, proj_disp_m=0.2)
            self.assertFalse(triggered)

    def test_incoherent_walk_without_shrink_triggers(self):
        g = LockGate()
        g.gate_detection(10.0, 0.0, 10.0)
        triggered = False
        for _ in range(20):
            # Walked displacement accumulates but the reported distance never shrinks.
            triggered = g.end_of_cycle(best_dist_estimate=10.0, walking=True, proj_disp_m=0.3)
            if triggered:
                break
        self.assertTrue(triggered)
        self.assertEqual(g.last_trigger, "M7")

    def test_endgame_carve_out_suspends_below_min_goal_dist(self):
        g = LockGate()
        g.gate_detection(1.0, 0.0, 10.0)   # below M7_MIN_GOAL_DIST_M=1.5
        for _ in range(20):
            triggered = g.end_of_cycle(best_dist_estimate=1.0, walking=True, proj_disp_m=0.3)
            self.assertFalse(triggered)

    def test_penalty_zone_requires_corroboration_after_trigger(self):
        g = LockGate()
        g.gate_detection(10.0, 0.0, 10.0)
        for _ in range(20):
            if g.end_of_cycle(best_dist_estimate=10.0, walking=True, proj_disp_m=0.3):
                break
        g.force_drop()
        self.assertEqual(g.state, "NONE")
        # A fresh detection landing right where the dropped lock was needs
        # M7_PENALTY_CONFIRM_M-of-N corroborating cycles, not an instant confirm.
        accepted1 = g.gate_detection(10.0, 0.0, 10.0)
        self.assertFalse(accepted1)
        accepted2 = g.gate_detection(10.05, 0.01, 10.0)
        self.assertTrue(accepted2)

    def test_penalty_zone_does_not_apply_to_unrelated_detection(self):
        g = LockGate()
        g.gate_detection(10.0, 0.0, 10.0)
        for _ in range(20):
            if g.end_of_cycle(best_dist_estimate=10.0, walking=True, proj_disp_m=0.3):
                break
        g.force_drop()
        # A detection FAR from the dropped lock's (bearing,dist) is outside the
        # penalty zone and falls back to ordinary M2-off instant-confirm behavior.
        accepted = g.gate_detection(2.0, math.radians(170.0), 10.0)
        self.assertTrue(accepted)


class TestForceDropAndMiscState(_ToggleGuard):

    def setUp(self):
        super().setUp()
        self._set(LOCK_M1=False, LOCK_M2=False, LOCK_M3=False)

    def test_force_drop_resets_to_none(self):
        g = LockGate()
        g.gate_detection(5.0, 0.0, 10.0)
        self.assertEqual(g.state, "CONFIRMED")
        g.force_drop()
        self.assertEqual(g.state, "NONE")

    def test_fresh_gate_state_is_none(self):
        g = LockGate()
        self.assertEqual(g.state, "NONE")
        self.assertIsNone(g.last_trigger)


if __name__ == "__main__":
    unittest.main()
