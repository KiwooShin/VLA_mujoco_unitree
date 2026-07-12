"""Unit tests for code.apps.repl.web._start_web_ui: the Flask-missing
fallback path, and a cheap integration smoke that the server thread actually
starts (route wiring/rollouts are exercised by the other test modules +
the standing eval gates, not here).
"""

from __future__ import annotations

import builtins
import threading
import time
import unittest

from code.apps.repl.executor import EventBus, Executor
from code.apps.repl.planner import Planner, SceneManager
from code.apps.repl.web import _start_web_ui


class FlaskMissingTest(unittest.TestCase):
    def test_returns_none_when_flask_unavailable(self) -> None:
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "flask":
                raise ImportError("simulated: no flask")
            return real_import(name, *args, **kwargs)

        scene = SceneManager()
        bus = EventBus()
        planner = Planner(scene)
        executor = Executor(scene_manager=scene, bus=bus, render_video=False)

        builtins.__import__ = _fake_import
        try:
            result = _start_web_ui(bus, executor, planner, scene, port=1)
        finally:
            builtins.__import__ = real_import
        self.assertIsNone(result)


class FlaskServerSmokeTest(unittest.TestCase):
    def test_start_web_ui_returns_a_running_daemon_thread(self) -> None:
        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("flask not installed")

        scene = SceneManager()
        scene._scene_cfg = {"objects": [], "target_index": 0}
        bus = EventBus()
        planner = Planner(scene)
        executor = Executor(scene_manager=scene, bus=bus, render_video=False)

        thread = _start_web_ui(bus, executor, planner, scene, port=17643)
        self.assertIsInstance(thread, threading.Thread)
        self.assertTrue(thread.daemon)
        time.sleep(0.3)
        self.assertTrue(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
