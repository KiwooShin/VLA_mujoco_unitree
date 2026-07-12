"""Unit tests for code/perception/ground_net.py: GROUND_NET backend, state-
parameterized so it can be exercised without a real trained checkpoint (a
fake duck-typed detector stands in for code.perception.detector.model.HeatmapDetector)."""
from __future__ import annotations

import math
import unittest

import numpy as np

from code.perception import ground_net as gn


class _FakeDetector:
    """Duck-types code.perception.detector.model.HeatmapDetector's public
    surface used by ground_net.infer(): .infer(...) and .last_heat_prob."""

    def __init__(self, confidence: float, dist_m: float, bearing_deg: float,
                raise_on_infer: bool = False):
        self.confidence = confidence
        self.dist_m = dist_m
        self.bearing_deg = bearing_deg
        self.raise_on_infer = raise_on_infer
        self.last_heat_prob = np.full((4, 4), confidence, dtype=np.float32)
        self.calls = []

    def infer(self, rgb, depth, class_name, color_name, cam_type, conf_thresh=0.5):
        self.calls.append(dict(class_name=class_name, color_name=color_name,
                               cam_type=cam_type, conf_thresh=conf_thresh))
        if self.raise_on_infer:
            raise RuntimeError("boom")
        return dict(present=self.confidence >= conf_thresh, confidence=self.confidence,
                    dist_m=self.dist_m, bearing_deg=self.bearing_deg, peak_px=(1.0, 1.0))


def _rgbd(h=8, w=8):
    return np.zeros((h, w, 3), dtype=np.uint8), np.full((h, w), 2.0, dtype=np.float32)


class TestGroundNetStateDefaults(unittest.TestCase):

    def test_fresh_state_defaults(self):
        s = gn.GroundNetState()
        self.assertIsNone(s.detector)
        self.assertFalse(s.load_failed)
        self.assertIsNone(s.class_names)
        self.assertIsNone(s.color_names)
        self.assertFalse(s.widefov_warned)
        self.assertFalse(s.fallback_warned)
        self.assertFalse(s.optout_notified)
        self.assertEqual(s.lat_ms, [])
        self.assertIsNone(s.track_dist_m)
        self.assertIsNone(s.track_bearing_rad)
        self.assertIsNone(s.last_heatmap)

    def test_two_instances_independent(self):
        s1 = gn.GroundNetState()
        s2 = gn.GroundNetState()
        s1.lat_ms.append(5.0)
        self.assertEqual(s2.lat_ms, [])
        s1.detector = object()
        self.assertIsNone(s2.detector)


class TestAccessors(unittest.TestCase):

    def test_get_last_heatmap_reads_state(self):
        s = gn.GroundNetState()
        self.assertIsNone(gn.get_last_heatmap(s))
        s.last_heatmap = dict(confidence=0.9)
        self.assertEqual(gn.get_last_heatmap(s), dict(confidence=0.9))

    def test_reset_track_clears_both_fields(self):
        s = gn.GroundNetState(track_dist_m=1.2, track_bearing_rad=0.3)
        gn.reset_track(s)
        self.assertIsNone(s.track_dist_m)
        self.assertIsNone(s.track_bearing_rad)

    def test_latency_stats_empty(self):
        s = gn.GroundNetState()
        self.assertEqual(gn.latency_stats(s), {})

    def test_latency_stats_summary(self):
        s = gn.GroundNetState(lat_ms=[10.0, 20.0, 30.0, 40.0, 50.0])
        stats = gn.latency_stats(s)
        self.assertEqual(stats["n"], 5)
        self.assertAlmostEqual(stats["mean_ms"], 30.0)
        self.assertEqual(stats["max_ms"], 50.0)
        self.assertIn("p50_ms", stats)
        self.assertIn("p95_ms", stats)
        self.assertIn("p99_ms", stats)


class TestLoadDetectorFailure(unittest.TestCase):

    def test_missing_checkpoint_sticky_failure(self):
        s = gn.GroundNetState()
        det1 = gn.load_detector(s, "/nonexistent/path/model_best.pt", "cpu", 0.5)
        self.assertIsNone(det1)
        self.assertTrue(s.load_failed)
        # Second call must NOT retry (sticky) -- still None, no exception.
        det2 = gn.load_detector(s, "/nonexistent/path/model_best.pt", "cpu", 0.5)
        self.assertIsNone(det2)

    def test_already_loaded_short_circuits(self):
        s = gn.GroundNetState()
        fake = _FakeDetector(0.9, 2.0, 0.0)
        s.detector = fake
        # load_detector should just return the cached instance without
        # touching load_failed/class_names.
        out = gn.load_detector(s, "/irrelevant/path.pt", "cpu", 0.5)
        self.assertIs(out, fake)
        self.assertFalse(s.load_failed)


class TestInfer(unittest.TestCase):

    def _state(self, detector):
        return gn.GroundNetState(detector=detector, class_names=["ball", "cube", "cylinder", "cone"],
                                 color_names=["red", "yellow", "blue", "green", "orange",
                                             "purple", "cyan"])

    def test_no_detector_not_visible(self):
        s = gn.GroundNetState()   # detector is None
        rgb, depth = _rgbd()
        r = gn.infer(s, rgb, depth, "red", "ball", {}, tau=0.5, hysteresis=False, tau_track=0.3)
        self.assertTrue(r.not_visible)

    def test_unknown_shape_not_visible(self):
        fake = _FakeDetector(0.9, 2.0, 5.0)
        s = self._state(fake)
        rgb, depth = _rgbd()
        r = gn.infer(s, rgb, depth, "red", "teapot", {}, tau=0.5, hysteresis=False, tau_track=0.3)
        self.assertTrue(r.not_visible)
        self.assertEqual(len(fake.calls), 0)   # never even called infer()

    def test_unknown_color_not_visible(self):
        fake = _FakeDetector(0.9, 2.0, 5.0)
        s = self._state(fake)
        rgb, depth = _rgbd()
        r = gn.infer(s, rgb, depth, "chartreuse", "ball", {}, tau=0.5, hysteresis=False, tau_track=0.3)
        self.assertTrue(r.not_visible)

    def test_above_tau_accepted(self):
        fake = _FakeDetector(confidence=0.9, dist_m=2.5, bearing_deg=10.0)
        s = self._state(fake)
        rgb, depth = _rgbd()
        r = gn.infer(s, rgb, depth, "red", "ball", {}, tau=0.64, hysteresis=False, tau_track=0.3)
        self.assertFalse(r.not_visible)
        self.assertAlmostEqual(r.dist, 2.5)
        self.assertAlmostEqual(r.confidence, 0.9)
        self.assertAlmostEqual(math.atan2(r.sin_th, r.cos_th), math.radians(10.0), places=6)

    def test_below_tau_no_hysteresis_rejected(self):
        fake = _FakeDetector(confidence=0.5, dist_m=2.5, bearing_deg=0.0)
        s = self._state(fake)
        rgb, depth = _rgbd()
        r = gn.infer(s, rgb, depth, "red", "ball", {}, tau=0.64, hysteresis=False, tau_track=0.3)
        self.assertTrue(r.not_visible)

    def test_negative_dist_clamped_to_zero(self):
        fake = _FakeDetector(confidence=0.9, dist_m=-1.0, bearing_deg=0.0)
        s = self._state(fake)
        rgb, depth = _rgbd()
        r = gn.infer(s, rgb, depth, "red", "ball", {}, tau=0.5, hysteresis=False, tau_track=0.3)
        self.assertEqual(r.dist, 0.0)

    def test_infer_exception_is_not_visible(self):
        fake = _FakeDetector(confidence=0.9, dist_m=2.0, bearing_deg=0.0, raise_on_infer=True)
        s = self._state(fake)
        rgb, depth = _rgbd()
        r = gn.infer(s, rgb, depth, "red", "ball", {}, tau=0.5, hysteresis=False, tau_track=0.3)
        self.assertTrue(r.not_visible)

    def test_cam_type_selected_from_intrinsics(self):
        fake = _FakeDetector(confidence=0.9, dist_m=1.0, bearing_deg=0.0)
        s = self._state(fake)
        rgb, depth = _rgbd()
        gn.infer(s, rgb, depth, "red", "ball", {"is_proximity": True}, tau=0.5,
                hysteresis=False, tau_track=0.3)
        self.assertEqual(fake.calls[-1]["cam_type"], "proximity")
        gn.infer(s, rgb, depth, "red", "ball", {}, tau=0.5, hysteresis=False, tau_track=0.3)
        self.assertEqual(fake.calls[-1]["cam_type"], "grounding")

    def test_widefov_warns_once(self):
        fake = _FakeDetector(confidence=0.9, dist_m=1.0, bearing_deg=0.0)
        s = self._state(fake)
        rgb, depth = _rgbd()
        self.assertFalse(s.widefov_warned)
        gn.infer(s, rgb, depth, "red", "ball", {"is_widefov": True}, tau=0.5,
                hysteresis=False, tau_track=0.3)
        self.assertTrue(s.widefov_warned)
        # Calling again should not raise/loop -- flag stays True, no crash.
        gn.infer(s, rgb, depth, "red", "ball", {"is_widefov": True}, tau=0.5,
                hysteresis=False, tau_track=0.3)
        self.assertTrue(s.widefov_warned)

    def test_hysteresis_track_continuation_accepts_marginal_detection(self):
        fake = _FakeDetector(confidence=0.5, dist_m=2.0, bearing_deg=0.0)
        s = self._state(fake)
        s.track_dist_m = 2.0
        s.track_bearing_rad = 0.0
        rgb, depth = _rgbd()
        r = gn.infer(s, rgb, depth, "red", "ball", {}, tau=0.64, hysteresis=True, tau_track=0.4)
        self.assertFalse(r.not_visible)

    def test_hysteresis_rejects_when_far_from_track(self):
        fake = _FakeDetector(confidence=0.5, dist_m=8.0, bearing_deg=90.0)
        s = self._state(fake)
        s.track_dist_m = 2.0
        s.track_bearing_rad = 0.0
        rgb, depth = _rgbd()
        r = gn.infer(s, rgb, depth, "red", "ball", {}, tau=0.64, hysteresis=True, tau_track=0.4)
        self.assertTrue(r.not_visible)

    def test_hysteresis_no_track_state_rejects_marginal(self):
        fake = _FakeDetector(confidence=0.5, dist_m=2.0, bearing_deg=0.0)
        s = self._state(fake)
        # track_dist_m is None -- no track yet, cannot accept via hysteresis.
        rgb, depth = _rgbd()
        r = gn.infer(s, rgb, depth, "red", "ball", {}, tau=0.64, hysteresis=True, tau_track=0.4)
        self.assertTrue(r.not_visible)

    def test_accepted_detection_updates_track_when_hysteresis_on(self):
        fake = _FakeDetector(confidence=0.9, dist_m=3.0, bearing_deg=15.0)
        s = self._state(fake)
        rgb, depth = _rgbd()
        gn.infer(s, rgb, depth, "red", "ball", {}, tau=0.5, hysteresis=True, tau_track=0.3)
        self.assertAlmostEqual(s.track_dist_m, 3.0)
        self.assertAlmostEqual(s.track_bearing_rad, math.radians(15.0))

    def test_rejected_detection_does_not_touch_track_when_hysteresis_off(self):
        fake = _FakeDetector(confidence=0.9, dist_m=3.0, bearing_deg=15.0)
        s = self._state(fake)
        gn.infer(s, *_rgbd(), "red", "ball", {}, tau=0.5, hysteresis=False, tau_track=0.3)
        self.assertIsNone(s.track_dist_m)

    def test_heatmap_cache_populated_after_infer(self):
        fake = _FakeDetector(confidence=0.9, dist_m=3.0, bearing_deg=15.0)
        s = self._state(fake)
        self.assertIsNone(s.last_heatmap)
        gn.infer(s, *_rgbd(), "red", "ball", {}, tau=0.5, hysteresis=False, tau_track=0.3)
        self.assertIsNotNone(s.last_heatmap)
        self.assertEqual(s.last_heatmap["color"], "red")
        self.assertEqual(s.last_heatmap["shape"], "ball")
        self.assertTrue(s.last_heatmap["accepted"])

    def test_latency_appended_on_success(self):
        fake = _FakeDetector(confidence=0.9, dist_m=3.0, bearing_deg=0.0)
        s = self._state(fake)
        self.assertEqual(s.lat_ms, [])
        gn.infer(s, *_rgbd(), "red", "ball", {}, tau=0.5, hysteresis=False, tau_track=0.3)
        self.assertEqual(len(s.lat_ms), 1)
        self.assertGreaterEqual(s.lat_ms[0], 0.0)


if __name__ == "__main__":
    unittest.main()
