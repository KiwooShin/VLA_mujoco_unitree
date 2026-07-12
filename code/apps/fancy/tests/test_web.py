"""Unit tests for code.apps.fancy.web: the Flask-missing fallback path, the
`/status`-payload JSON-sanitization used by `_do_rollout`, and a cheap
integration smoke that `_start_fancy_web_ui` actually starts a server
thread (route wiring itself is exercised by the eval/rollout gates, not
here)."""

from __future__ import annotations

import builtins
import threading
import time
import unittest
from unittest import mock

import numpy as np

from code.apps.fancy.live import FancySceneManager
from code.apps.fancy.web import _start_fancy_web_ui


class FlaskMissingTest(unittest.TestCase):
    def test_returns_none_when_flask_unavailable(self) -> None:
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "flask":
                raise ImportError("simulated: no flask")
            return real_import(name, *args, **kwargs)

        scene_mgr = FancySceneManager()
        builtins.__import__ = _fake_import
        try:
            result = _start_fancy_web_ui(inf=mock.Mock(), scene_manager=scene_mgr, out_dir="/tmp", port=1)
        finally:
            builtins.__import__ = real_import
        self.assertIsNone(result)


class FlaskServerSmokeTest(unittest.TestCase):
    def test_start_fancy_web_ui_returns_running_daemon_thread(self) -> None:
        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("flask not installed")

        scene_mgr = FancySceneManager()
        fake_scene = {
            "objects": [{"color_name": "red", "shape_name": "ball", "dist_from_robot": 1.0}],
            "target_index": 0, "init_bearing_deg": 50.0,
        }
        with mock.patch("code.apps.fancy.live.sample_fancy_scene_long", return_value=fake_scene):
            scene_mgr.new_scene()

        thread = _start_fancy_web_ui(inf=mock.Mock(), scene_manager=scene_mgr,
                                      out_dir="/tmp/fancy_test_out", port=17644)
        self.assertIsInstance(thread, threading.Thread)
        self.assertTrue(thread.daemon)
        time.sleep(0.3)
        self.assertTrue(thread.is_alive())


class StatusResultSanitizationTest(unittest.TestCase):
    """Mirrors the JSON-serializable-only filter `_do_rollout` applies to
    the raw rollout result before storing it in `_status_state` (guards
    against a leaked np.ndarray/np.float32 crashing the `/status` route's
    jsonify() call)."""

    def _sanitize(self, result: dict) -> dict:
        return {
            k: (v.item() if hasattr(v, 'item') and getattr(v, 'ndim', 1) == 0 else v)
            for k, v in result.items()
            if isinstance(v, (bool, int, float, str, type(None)))
            or (hasattr(v, 'item') and getattr(v, 'ndim', 1) == 0)
        }

    def test_scalar_numpy_values_converted_to_python(self) -> None:
        result = {"success": True, "final_dist": np.float32(0.42), "steps": 10}
        out = self._sanitize(result)
        self.assertIsInstance(out["final_dist"], float)
        self.assertEqual(out["steps"], 10)

    def test_ndarray_fields_are_dropped(self) -> None:
        result = {"success": True, "frames_sbs": [np.zeros((4, 4, 3))], "path_trail_out": [np.array([1.0, 2.0])]}
        out = self._sanitize(result)
        self.assertNotIn("frames_sbs", out)
        self.assertNotIn("path_trail_out", out)
        self.assertIn("success", out)

    def test_none_values_kept(self) -> None:
        result = {"video_path": None}
        out = self._sanitize(result)
        self.assertIn("video_path", out)
        self.assertIsNone(out["video_path"])


if __name__ == "__main__":
    unittest.main()
