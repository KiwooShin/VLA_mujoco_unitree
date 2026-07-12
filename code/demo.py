"""
demo.py — Interactive REPL Demo for G1Nav

RF-1: split into code/apps/repl/{constants,planner,maneuver_inferencer,
executor,web,cli}.py. This file is now a thin CLI entry shim + compat
re-export so `python code/demo.py ...` and `from code.demo import ...`
(old import path — used by code/record_showcase.py) both keep working
verbatim. See code/apps/repl/cli.py for the real docstring/usage notes and
code/apps/repl/*.py module docstrings for the per-file architecture.

Usage (unchanged):
  # Terminal REPL (always works):
  MUJOCO_GL=egl python code/demo.py

  # Web UI on port 5000:
  MUJOCO_GL=egl python code/demo.py --web

  # Canned smoke test (3 instructions, saves videos):
  MUJOCO_GL=egl python code/demo.py --smoke --out eval/demo --device cuda

  # Save video to specific dir:
  MUJOCO_GL=egl python code/demo.py --out eval/demo --difficulty demo

  # Easy mode (shorter walks, 93% success):
  MUJOCO_GL=egl python code/demo.py --difficulty easy
"""

from __future__ import annotations

# Re-export the old flat-module public surface (record_showcase.py imports
# SceneManager, Planner, Executor, EventBus, COLORS, SHAPES from `code.demo`).
from code.apps.repl.constants import (
    COLORS, SHAPES, MANEUVER_DIRECTIONS,
    GOTO_CKPT, MANEUVER_CKPT, MAXSTEPS_GOTO, MAXSTEPS_MANEUVER,
    DEMO_OUT_DIR, WEB_PORT, LANG_CACHE,
    _check_cuda, _load_lang_cache, _get_lang_emb,
)
from code.apps.repl.planner import SceneManager, SubGoal, Planner
from code.apps.repl.maneuver_inferencer import ManeuverInferencer
from code.apps.repl.executor import EventBus, Executor
from code.apps.repl.web import _start_web_ui
from code.apps.repl.cli import _terminal_repl, _smoke_test, main

if __name__ == "__main__":
    main()
