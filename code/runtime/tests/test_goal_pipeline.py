"""Unit tests for code.runtime.goal_pipeline.GoalPipeline — the grounding-
cycle/goal-EMA/hold/CAM-2-handoff/scan state machine.

These are pure-logic tests: `GoalPipeline` never calls `classical_ground`
itself (see its module docstring), so it can be driven entirely with
synthetic `GroundingResult`-shaped inputs and a fake MjData, no MuJoCo/EGL
needed. LockGate/ReacquisitionScan/BidirectionalScanSchedule are exercised
via their real (default-toggle) implementations, matching how
`code/perception/tests/` already covers LockGate's own internal state
machine in isolation — these tests instead pin how GoalPipeline *wires*
them.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from code.runtime.goal_pipeline import GoalPipeline
from code.runtime.goal_config import (
    GOAL_EMA_ALPHA, HOLD_GOAL_HORIZON, SCAN_ALIGNED_THR_DEG, SCAN_TIMEOUT,
    CAM_D_LO, CAM_D_HI, CAM_PROXIMITY_D_FAR,
)
from code.perception.types import GroundingResult


def _gr(dist: float, bearing_deg: float, not_visible: bool = False,
        confidence: float = 0.9, best_area: float = 1e5) -> GroundingResult:
    """Builds a synthetic GroundingResult at the given (dist, bearing)."""
    th = math.radians(bearing_deg)
    return GroundingResult(dist=dist, cos_th=math.cos(th), sin_th=math.sin(th),
                            confidence=confidence, not_visible=not_visible,
                            best_area=best_area)


class _FakeMjData:
    def __init__(self, x: float = 0.0, y: float = 0.0, height: float = 0.75):
        self.qpos = np.zeros(3, dtype=np.float64)
        self.qpos[0] = x
        self.qpos[1] = y
        self.qpos[2] = height


def _gp(classical: bool = True, learned: bool = False, maneuver: bool = False) -> GoalPipeline:
    return GoalPipeline(need_classical_render=classical, need_learned_render=learned,
                        avoid_is_maneuver=maneuver, verbose=False)


class TestConstruction(unittest.TestCase):
    def test_defaults(self):
        gp = _gp()
        np.testing.assert_allclose(gp.cached_goal_vec, [2.0, 1.0, 0.0])
        self.assertEqual(gp.last_grounding_step, -999)
        self.assertTrue(gp.scan_active)
        self.assertEqual(gp.active_cam, 'GROUNDING')
        self.assertEqual(gp.cam_miss_count, 0)
        self.assertIsNone(gp.goal_ema)
        self.assertIsNone(gp.last_known_goal)
        self.assertEqual(gp.frames_since_detection, 0)
        self.assertFalse(gp.using_rescan_sched)
        self.assertEqual(gp.avoid_bias_wz, 0.0)


class TestGroundingCadence(unittest.TestCase):
    def test_due_immediately_at_start(self):
        gp = _gp()
        self.assertTrue(gp.due_for_classical_grounding(0))

    def test_not_due_again_until_period_elapses(self):
        gp = _gp()
        gp.mark_grounding_step(5)
        self.assertFalse(gp.due_for_classical_grounding(6))
        self.assertFalse(gp.due_for_classical_grounding(14))
        self.assertTrue(gp.due_for_classical_grounding(15))

    def test_learned_cadence_independent_flag(self):
        gp = _gp(classical=False, learned=True)
        self.assertFalse(gp.due_for_classical_grounding(0))
        self.assertTrue(gp.due_for_learned_grounding(0))


class TestCamProbe(unittest.TestCase):
    def test_no_probe_below_two_misses(self):
        gp = _gp()
        gp.register_detection_outcome(not_visible=True)
        self.assertIsNone(gp.maybe_probe_camera())

    def test_probe_after_two_consecutive_misses(self):
        gp = _gp()
        gp.last_known_goal = np.array([1.0, 1.0, 0.0], dtype=np.float32)  # within probe-gate range
        gp.register_detection_outcome(not_visible=True)
        gp.register_detection_outcome(not_visible=True)
        self.assertEqual(gp.maybe_probe_camera(), 'PROXIMITY')

    def test_hit_resets_miss_count(self):
        gp = _gp()
        gp.register_detection_outcome(not_visible=True)
        gp.register_detection_outcome(not_visible=False)
        self.assertEqual(gp.cam_miss_count, 0)
        gp.register_detection_outcome(not_visible=True)
        self.assertIsNone(gp.maybe_probe_camera())

    def test_probing_from_proximity_to_grounding_always_ok(self):
        gp = _gp()
        gp.active_cam = 'PROXIMITY'
        gp.register_detection_outcome(not_visible=True)
        gp.register_detection_outcome(not_visible=True)
        self.assertEqual(gp.maybe_probe_camera(), 'GROUNDING')

    def test_probing_proximity_gated_by_last_known_distance(self):
        gp = _gp()
        gp.last_known_goal = np.array([5.0, 1.0, 0.0], dtype=np.float32)  # far, > CAM_PROXIMITY_D_FAR
        gp.register_detection_outcome(not_visible=True)
        gp.register_detection_outcome(not_visible=True)
        self.assertIsNone(gp.maybe_probe_camera())   # gated out

    def test_probing_proximity_allowed_within_far_limit(self):
        gp = _gp()
        gp.last_known_goal = np.array([CAM_PROXIMITY_D_FAR - 0.1, 1.0, 0.0], dtype=np.float32)
        gp.register_detection_outcome(not_visible=True)
        gp.register_detection_outcome(not_visible=True)
        self.assertEqual(gp.maybe_probe_camera(), 'PROXIMITY')

    def test_on_probe_adopted_flips_camera_and_resets_miss_count(self):
        gp = _gp()
        gp.cam_miss_count = 2
        gp.on_probe_adopted('PROXIMITY')
        self.assertEqual(gp.active_cam, 'PROXIMITY')
        self.assertEqual(gp.cam_miss_count, 0)


class TestClassicalDetectionEma(unittest.TestCase):
    def test_first_hit_sets_ema_and_last_known(self):
        gp = _gp()
        gp.process_classical_detection(_gr(3.0, 0.0), step=0)
        np.testing.assert_allclose(gp.cached_goal_vec, [3.0, 1.0, 0.0], atol=1e-5)
        self.assertEqual(gp.frames_since_detection, 0)

    def test_ema_blends_toward_new_detection(self):
        # Delta kept within LOCK_M3's default-on innovation gate (dist_gate_m
        # = max(0.8, 0.16*1.5) = 0.8m) so the second call isn't rejected as
        # an implausible frame-to-frame jump once state is CONFIRMED.
        gp = _gp()
        gp.process_classical_detection(_gr(3.0, 0.0), step=0)
        gp.process_classical_detection(_gr(2.5, 0.0), step=10)
        expected_dist = GOAL_EMA_ALPHA * 2.5 + (1.0 - GOAL_EMA_ALPHA) * 3.0
        self.assertAlmostEqual(float(gp.cached_goal_vec[0]), expected_dist, places=5)

    def test_ema_converges_toward_repeated_detections(self):
        # Gradual small steps (each within the M3 innovation gate) walking the
        # EMA down toward a steady 1.0m, mirroring the robot physically
        # closing distance a bit each grounding cycle rather than an
        # implausible single-frame teleport.
        gp = _gp()
        gp.process_classical_detection(_gr(5.0, 0.0), step=0)
        dist = 5.0
        i = 1
        while dist > 1.0:
            dist = max(1.0, dist - 0.5)
            gp.process_classical_detection(_gr(dist, 0.0), step=i * 10)
            i += 1
        for _ in range(30):
            gp.process_classical_detection(_gr(1.0, 0.0), step=i * 10)
            i += 1
        self.assertAlmostEqual(float(gp.cached_goal_vec[0]), 1.0, places=3)

    def test_miss_holds_last_known_goal_within_horizon(self):
        gp = _gp()
        gp.process_classical_detection(_gr(2.0, 0.0), step=0)
        held = gp.cached_goal_vec.copy()
        gp.process_classical_detection(_gr(0.0, 0.0, not_visible=True), step=10)
        np.testing.assert_allclose(gp.cached_goal_vec, held, atol=1e-6)
        self.assertEqual(gp.frames_since_detection, 1)

    def test_miss_streak_within_hold_horizon_keeps_holding(self):
        gp = _gp()
        gp.process_classical_detection(_gr(2.0, 30.0), step=0)
        held = gp.cached_goal_vec.copy()
        for i in range(1, HOLD_GOAL_HORIZON):
            gp.process_classical_detection(_gr(0.0, 0.0, not_visible=True), step=i)
        np.testing.assert_allclose(gp.cached_goal_vec, held, atol=1e-6)
        self.assertEqual(gp.frames_since_detection, HOLD_GOAL_HORIZON - 1)

    def test_scan_exits_when_bearing_within_aligned_threshold(self):
        gp = _gp()
        self.assertTrue(gp.scan_active)
        gp.process_classical_detection(_gr(3.0, SCAN_ALIGNED_THR_DEG - 5.0), step=0)
        self.assertFalse(gp.scan_active)

    def test_scan_stays_active_on_partial_detection(self):
        gp = _gp()
        gp.process_classical_detection(_gr(3.0, SCAN_ALIGNED_THR_DEG + 20.0), step=0)
        self.assertTrue(gp.scan_active)

    def test_scan_exit_is_sticky_once_aligned(self):
        gp = _gp()
        gp.process_classical_detection(_gr(3.0, 0.0), step=0)  # exits scan
        self.assertFalse(gp.scan_active)
        # A later, wide-bearing detection should NOT re-enter scan mode
        # (the original only checked `if self.scan_active:` on the way in).
        gp.process_classical_detection(_gr(3.0, 150.0), step=10)
        self.assertFalse(gp.scan_active)


class TestCam2SchmittHandoff(unittest.TestCase):
    def test_switches_to_proximity_below_lo_threshold(self):
        gp = _gp()
        gp.process_classical_detection(_gr(CAM_D_LO - 0.05, 0.0), step=0)
        self.assertEqual(gp.active_cam, 'PROXIMITY')

    def test_stays_on_grounding_above_lo_threshold(self):
        gp = _gp()
        gp.process_classical_detection(_gr(CAM_D_LO + 0.2, 0.0), step=0)
        self.assertEqual(gp.active_cam, 'GROUNDING')

    def test_switches_back_to_grounding_above_hi_threshold(self):
        gp = _gp()
        gp.active_cam = 'PROXIMITY'
        gp.goal_ema = np.array([1.4, 1.0, 0.0], dtype=np.float32)  # seed EMA in dual-visible band
        # A raw detection far enough that even after EMA damping (alpha=0.4)
        # the blended distance still clears CAM_D_HI.
        gp.process_classical_detection(_gr(3.0, 0.0), step=0)
        blended = GOAL_EMA_ALPHA * 3.0 + (1 - GOAL_EMA_ALPHA) * 1.4
        self.assertGreater(blended, CAM_D_HI)   # sanity on the test's own arithmetic
        self.assertEqual(gp.active_cam, 'GROUNDING')

    def test_hysteresis_band_does_not_flip(self):
        """Between CAM_D_LO and CAM_D_HI, whichever camera is active stays
        active (no chatter)."""
        gp = _gp()
        gp.active_cam = 'PROXIMITY'
        gp.goal_ema = np.array([1.4, 1.0, 0.0], dtype=np.float32)
        mid = (CAM_D_LO + CAM_D_HI) / 2.0
        gp.process_classical_detection(_gr(mid, 0.0), step=0)
        self.assertEqual(gp.active_cam, 'PROXIMITY')


class TestLockDropAndRescan(unittest.TestCase):
    def test_coast_expired_miss_streak_triggers_rescan_when_m5_on(self):
        gp = _gp()
        gp.process_classical_detection(_gr(2.0, 0.0), step=0)
        gp.scan_active = False
        if not gp.lock_gate.coast_expired(HOLD_GOAL_HORIZON + 1, HOLD_GOAL_HORIZON):
            self.skipTest("LOCK_M5 is off in this environment (pre-NX-2 default) — "
                          "coast_expired() is a provable no-op, matching the "
                          "unbounded-freeze legacy behavior this test would otherwise pin.")
        for i in range(1, HOLD_GOAL_HORIZON + 2):
            gp.process_classical_detection(_gr(0.0, 0.0, not_visible=True), step=i)
        self.assertTrue(gp.scan_active)
        self.assertTrue(gp.using_rescan_sched)
        self.assertIsNotNone(gp.rescan_sched)
        self.assertIsNone(gp.goal_ema)
        self.assertIsNone(gp.last_known_goal)
        self.assertEqual(gp.frames_since_detection, 0)

    def test_manual_lock_drop_and_rescan_resets_state(self):
        gp = _gp()
        gp.process_classical_detection(_gr(2.0, 0.0), step=0)
        gp.avoid_bias_wz = 0.3
        gp._lock_drop_and_rescan()
        self.assertIsNone(gp.goal_ema)
        self.assertIsNone(gp.last_known_goal)
        self.assertEqual(gp.frames_since_detection, 0)
        self.assertTrue(gp.scan_active)
        self.assertTrue(gp.using_rescan_sched)
        self.assertIsNotNone(gp.rescan_sched)
        np.testing.assert_allclose(gp.cached_goal_vec, [2.0, 1.0, 0.0])
        self.assertEqual(gp.avoid_bias_wz, 0.0)


class TestScanStep(unittest.TestCase):
    def test_inactive_when_scan_not_active(self):
        gp = _gp()
        gp.scan_active = False
        self.assertIsNone(gp.try_scan_step(yaw=0.0, step=0))

    def test_inactive_when_not_classical_render_mode(self):
        gp = _gp(classical=False, learned=True)
        self.assertIsNone(gp.try_scan_step(yaw=0.0, step=0))

    def test_h3_scan_returns_nonzero_wz_before_timeout(self):
        gp = _gp()
        wz = gp.try_scan_step(yaw=0.0, step=0)
        self.assertIsNotNone(wz)
        self.assertIsInstance(wz, float)

    def test_h3_scan_times_out_and_deactivates(self):
        gp = _gp()
        result = gp.try_scan_step(yaw=0.0, step=SCAN_TIMEOUT)
        self.assertIsNone(result)
        self.assertFalse(gp.scan_active)

    def test_rescan_schedule_used_after_lock_drop(self):
        gp = _gp()
        gp._lock_drop_and_rescan()
        self.assertTrue(gp.using_rescan_sched)
        wz = gp.try_scan_step(yaw=0.0, step=5000)  # far past H3's absolute SCAN_TIMEOUT
        # ReacquisitionScan uses its OWN local step counter, so it must not
        # immediately time out just because the episode step is huge.
        self.assertIsNotNone(wz)


class TestAvoidBias(unittest.TestCase):
    def test_no_op_when_scan_active(self):
        gp = _gp()
        self.assertTrue(gp.scan_active)
        gp.update_avoid_bias(depth=None, intr_active=None, data_mj=_FakeMjData(), step=0)
        self.assertEqual(gp.avoid_cycles_total, 0)

    def test_no_op_when_maneuver_carved_out(self):
        gp = _gp(maneuver=True)
        gp.scan_active = False
        gp.update_avoid_bias(depth=None, intr_active=None, data_mj=_FakeMjData(), step=0)
        self.assertEqual(gp.avoid_cycles_total, 0)

    def test_decays_when_detection_stale(self):
        from code.control import avoid as _avoid
        gp = _gp()
        gp.scan_active = False
        gp.avoid_bias_wz = 0.2
        gp.frames_since_detection = _avoid.AVOID_STALE_MAX_MISSED_CYCLES + 1
        gp.update_avoid_bias(depth=None, intr_active=None, data_mj=_FakeMjData(), step=0)
        self.assertEqual(gp.avoid_cycles_total, 1)
        self.assertLessEqual(abs(gp.avoid_bias_wz), 0.2)


class TestLearnedDetection(unittest.TestCase):
    def test_does_not_touch_frames_since_detection_or_scan(self):
        gp = _gp(classical=False, learned=True)
        gp.frames_since_detection = 7
        gp.scan_active = True
        raw = np.array([2.0, 0.9, 0.1], dtype=np.float32)
        gp.process_learned_detection(raw)
        self.assertEqual(gp.frames_since_detection, 7)   # untouched (matches original)
        self.assertTrue(gp.scan_active)                   # untouched (matches original)
        np.testing.assert_allclose(gp.cached_goal_vec, raw, atol=1e-5)

    def test_ema_blends_across_calls(self):
        gp = _gp(classical=False, learned=True)
        gp.process_learned_detection(np.array([4.0, 1.0, 0.0], dtype=np.float32))
        gp.process_learned_detection(np.array([2.0, 1.0, 0.0], dtype=np.float32))
        expected = GOAL_EMA_ALPHA * 2.0 + (1 - GOAL_EMA_ALPHA) * 4.0
        self.assertAlmostEqual(float(gp.cached_goal_vec[0]), expected, places=5)


if __name__ == "__main__":
    unittest.main()
