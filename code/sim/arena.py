"""
arena.py — Reassembled public API for the G1Nav arena (RF-1).

The original flat ``code/arena.py`` (695 lines) was split by responsibility
into three files, each independently importable and each <500 lines:

  - ``arena_build.py``   — scene construction (``build_arena``), the fixed
                           color/shape palettes, and every camera geometry
                           constant.
  - ``arena_cameras.py`` — pure camera math: positioning
                           (``_set_ego_cam``), pinhole intrinsics
                           (``get_ego_intrinsics``), back-projection
                           (``backproject_pixel``).
  - ``arena_render.py``  — ``ArenaRenderer``, the EGL renderer lifecycle
                           (ego/grounding/proximity/widefov/third-person).

This module re-exports the union of all three so the ~25 existing importers
of ``code.arena`` (and its compat alias at ``code/arena.py``) see an
unchanged flat namespace — nothing here is new logic, only re-exports.
"""

# noqa: F401 throughout this block — these are re-exports, not local uses.
# _add_geom/_rgb255_to_rgba1/_set_ego_cam are private helpers that some old
# importers reach for directly (e.g. gen_det_failcases.py's
# `from code.arena import _set_ego_cam`), so they must stay attributes of
# this reassembled module even though nothing below calls them locally.
from code.sim.arena_build import (  # noqa: F401
    _add_geom,
    _rgb255_to_rgba1,
    build_arena,
    CAMERA_MODE,
    CAM_FWD,
    CAM_HEAD_Z,
    CAM_PITCH,
    COLORS,
    EGO_FOVY,
    EGO_H,
    EGO_W,
    G1_XML,
    GROUNDING_H,
    GROUNDING_PITCH,
    GROUNDING_W,
    PROXIMITY_H,
    PROXIMITY_PITCH,
    PROXIMITY_W,
    SHAPES,
    TP_H,
    TP_W,
    WIDEFOV_FOVY,
    WIDEFOV_H,
    WIDEFOV_PITCH,
    WIDEFOV_W,
)
from code.sim.arena_cameras import _set_ego_cam, backproject_pixel, get_ego_intrinsics  # noqa: F401
from code.sim.arena_render import ArenaRenderer

__all__ = [
    "ArenaRenderer",
    "CAMERA_MODE",
    "CAM_FWD",
    "CAM_HEAD_Z",
    "CAM_PITCH",
    "COLORS",
    "EGO_FOVY",
    "EGO_H",
    "EGO_W",
    "G1_XML",
    "GROUNDING_H",
    "GROUNDING_PITCH",
    "GROUNDING_W",
    "PROXIMITY_H",
    "PROXIMITY_PITCH",
    "PROXIMITY_W",
    "SHAPES",
    "TP_H",
    "TP_W",
    "WIDEFOV_FOVY",
    "WIDEFOV_H",
    "WIDEFOV_PITCH",
    "WIDEFOV_W",
    "backproject_pixel",
    "build_arena",
    "get_ego_intrinsics",
]


if __name__ == "__main__":
    # Quick smoke test: build a minimal scene and render one frame
    import sys

    import mujoco

    scene_cfg = {
        "arena_size": 4.0,
        "objects": [
            {"color_name": "red",  "color_rgb": (220, 40, 40),
             "shape_name": "ball", "size": 0.24, "x": 1.5, "y": 0.0},
            {"color_name": "blue", "color_rgb": (50, 90, 220),
             "shape_name": "cube", "size": 0.24, "x": 0.0, "y": 1.8},
        ],
        "lighting": {"ambient": 0.4},
    }
    model = build_arena(scene_cfg)
    data  = mujoco.MjData(model)
    model.opt.timestep = 0.005
    mujoco.mj_resetData(model, data)
    data.qpos[0:2] = [-1.5, 0.0]
    data.qpos[2]   = 0.79
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    renderer = ArenaRenderer(model)
    rgb, depth, intr = renderer.render_ego(data, yaw=0.0)
    print(f"arena.py smoke: rgb={rgb.shape}, depth={depth.shape}, intr={intr}")
    tp_cam = renderer.make_tp_cam()
    renderer.update_tp_cam(tp_cam, data)
    tp_rgb = renderer.render_tp(data, tp_cam)
    print(f"  tp={tp_rgb.shape}")
    renderer.close()
    print("arena.py smoke PASS")
    sys.exit(0)
