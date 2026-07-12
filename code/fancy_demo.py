"""
fancy_demo.py — FANCY Interactive Demo for G1Nav  (FD2 — enhanced)

RF-1: split into code/apps/fancy/{constants,sampling,sampling_long,
overlays_projection,overlays_bev,overlays_ego,hud,cards,rollout,multi_goal,
video,live,web_html,web,cli}.py. This file is now a thin CLI entry shim +
compat re-export so `python code/fancy_demo.py ...` and
`from code.fancy_demo import run_fancy_rollout, _concat_reel` (old import
path — used by code/render_showcase_reel.py) both keep working verbatim.
See code/apps/fancy/cli.py for the real docstring/usage notes and
code/apps/fancy/*.py module docstrings for the per-file architecture.

Polished visualization + interaction layer on top of demo.py's planner/executor/inferencer.
Reuses: Planner, Executor, SceneManager, Inferencer, _run_search_rollout from demo.py / eval_search.py.

View: ego camera (RGB robot view) | ELEVATED 3D DIAGONAL BEV FOLLOW-CAM side-by-side.

Usage (unchanged):
    # Headless smoke test — 5-6 long episodes + multi-goal:
    MUJOCO_GL=egl python code/fancy_demo.py --smoke --out eval/fancy_demo --device cuda

    # Showcase reel (6 episodes, auto concat):
    MUJOCO_GL=egl python code/fancy_demo.py --smoke --out eval/fancy_demo --device cuda --n-smoke 6

    # Web UI mode:
    MUJOCO_GL=egl python code/fancy_demo.py --web --out eval/fancy_demo --device cuda

    # Quick smoke (no render, validation only):
    MUJOCO_GL=egl python code/fancy_demo.py --smoke --out eval/fancy_demo --no-render
"""

from __future__ import annotations

# Re-export the old flat-module public surface. code/render_showcase_reel.py
# imports `run_fancy_rollout` + `_concat_reel` from `code.fancy_demo`.
from code.apps.fancy.constants import (
    ARENA_HALF_LONG, BEV_AZIMUTH, BEV_DISTANCE, BEV_ELEVATION, BEV_H, BEV_LOOKAT_Z, BEV_W,
    DIST_MAX_LONG, DIST_MIN_LONG, EGO_H, EGO_W, FANCY_OUT_DIR,
    FANCY_PLAIN, FEAT_AVOID_VIZ, FEAT_HEATMAP, FEAT_HIRES, FEAT_HUD, FEAT_TITLECARD, FEAT_TRAIL,
    GOTO_CKPT_DEFAULT, HEATMAP_ALPHA, HUD_BAR_H, KEYFRAME_PATH, MAXSTEPS_FANCY,
    PANEL_DISPLAY_H, PANEL_DISPLAY_W, RELIABLE_COLORS, RELIABLE_SHAPES, SKILL_STAGES,
    STATE_FAILED, STATE_IDLE, STATE_LOCATED, STATE_MOVING, STATE_REACHED, STATE_SEARCHING,
    STREAM_W, WEB_PORT,
)
from code.apps.fancy.overlays_projection import (
    TRAIL_COOL_BGR, TRAIL_WARM_BGR, _dashed_line, _lerp_color_bgr, world_to_bev_pixel,
)
from code.apps.fancy.overlays_bev import _STATE_COLOR_MAP, draw_avoid_overlay, draw_bev_overlays
from code.apps.fancy.overlays_ego import compose_sbs_frame, draw_detector_heatmap_overlay
from code.apps.fancy.hud import draw_hud_bar
from code.apps.fancy.cards import _final_canvas_dims, make_outro_card, make_title_card
from code.apps.fancy.video import _concat_reel, _write_fancy_video
from code.apps.fancy.sampling import (
    FANCY_MIN_OBJECTS, FANCY_OBJ_MIN_ROBOT_M, FANCY_OBJ_MIN_SEP_M, FANCY_OBJ_WALL_MARGIN_M,
    FIRST_SCENE_SEED, _place_fancy_object_xy, _select_fancy_distractor_combos, sample_fancy_scene,
)
from code.apps.fancy.sampling_long import sample_fancy_multi_goal_scene, sample_fancy_scene_long
from code.apps.fancy.rollout import run_fancy_rollout
from code.apps.fancy.multi_goal import (
    _extract_goal_hint, _parse_multi_goal_fancy, _resolve_goal_to_index, _split_multi_goal_parts,
    run_fancy_rollout_multi,
)
from code.apps.fancy.live import FancySceneManager, _terminal_loop, resolve_live_instruction
from code.apps.fancy.web_html import _HTML_FANCY
from code.apps.fancy.web import (
    _placeholder_frame, _set_stream_frame, _start_fancy_web_ui,
    _status_lock, _status_state, _stream_frame, _stream_lock,
)
from code.apps.fancy.cli import _has_cuda, main, run_smoke

if __name__ == "__main__":
    main()
