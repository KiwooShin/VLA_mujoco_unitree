"""
arena_build.py — Parametric MuJoCo arena construction for G1Nav.

Role: this file owns everything needed to turn a scene-sampler config dict
into a compiled ``mujoco.MjModel`` — the fixed color/shape palettes, every
camera geometry constant (ego/grounding/proximity/widefov/third-person), the
canonical resource paths, and ``build_arena()`` itself. Camera *math*
(positioning, intrinsics, back-projection) lives in ``arena_cameras.py``;
the runtime ``ArenaRenderer`` (EGL renderer lifecycle) lives in
``arena_render.py``. See ``code/sim/arena.py`` for the reassembled public
API (RF-1 split of the original ``code/arena.py``).

Builds a MjSpec (or compiled MjModel) from:
  - Flat floor + low perimeter walls
  - G1 robot via g1_gear_wbc.xml
  - N colored primitive objects (ball/cube/cylinder/cone) from a fixed palette
  - Non-overlapping placement
  - Ego RGBD camera (robot-head-attached, tilted ~30-35 deg down)
  - Third-person camera (for videos only; never a model input)

Public API
----------
build_arena(scene_cfg) -> mujoco.MjModel
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# GPU-rendering fix (2026-07-11): on glvnd systems where Mesa wins the EGL
# vendor race (symptom: "libEGL warning: egl: failed to create dri2 screen"),
# MuJoCo silently binds llvmpipe SOFTWARE rendering — measured 409 ms/frame
# for a 640x480 offscreen render vs 1.3 ms on the actual GPU (315x). Steer
# glvnd to the NVIDIA ICD when it exists and the user hasn't chosen one.
_NVIDIA_EGL_ICD = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if os.path.exists(_NVIDIA_EGL_ICD):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NVIDIA_EGL_ICD)

import mujoco

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------
_HERE: str = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT: str = os.path.dirname(os.path.dirname(_HERE))
_WBC_ROOT: str = os.path.join(
    _REPO_ROOT,
    "third_party/Isaac-GR00T/external_dependencies/"
    "GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1",
)
G1_XML: str = os.path.join(_WBC_ROOT, "g1_gear_wbc.xml")

# ---------------------------------------------------------------------------
# Fixed color palette  (name, RGB-uint8)
# ---------------------------------------------------------------------------
COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("red",     (220,  40,  40)),
    ("yellow",  (235, 205,  40)),
    ("blue",    ( 50,  90, 220)),
    ("green",   ( 40, 180,  70)),
    ("orange",  (240, 140,  30)),
    ("purple",  (150,  60, 200)),
    ("cyan",    ( 40, 200, 210)),
]

# Shape definitions  (name, half-size for placement radius)
SHAPES: list[tuple[str, float]] = [
    ("ball",     0.24),   # sphere
    ("cube",     0.24),   # box
    ("cylinder", 0.22),   # cylinder
    ("cone",     0.26),   # approximated as cone-shaped box/cylinder stack
]

# ---------------------------------------------------------------------------
# Ego-camera parameters (head camera, tilted downward)
# ---------------------------------------------------------------------------
EGO_W, EGO_H = 320, 240          # native render resolution
EGO_FOVY     = 90.0              # vertical FOV (degrees)
CAM_HEAD_Z   = 0.55              # height above pelvis z
CAM_FWD      = 0.10             # forward offset from pelvis xy
CAM_PITCH    = 32.0              # downward tilt (degrees) — actual robot ego camera
# V2: higher-resolution grounding render (480x360) with REDUCED camera tilt.
#
# CRITICAL FINDING (V2): With 32 deg camera downward tilt, targets at 7m+ fall BELOW
# the image entirely (pixel_row > image height). At 6m they appear at row 233/240
# (97%) — just outside the 5% bottom margin crop. This is why demo/classical
# detection rate is ~0% at 6-9m.
#
# Fix: Use a shallower 20 deg tilt for the grounding render. This keeps demo targets
# (4-9m) in the image at rows 236-279/360 (65-78%), well within detection bounds.
# The depth values from MuJoCo remain accurate. Bearing accuracy maintained via
# cam_to_egocentric() with GROUNDING_PITCH parameter.
#
# Resolution: 480x360 (1.5x) so targets are 2.25x larger in pixel area.
# Together: pitch fix + resolution increase enables 4-9m detection.
GROUNDING_W, GROUNDING_H = 480, 360   # grounding render resolution (V2)
GROUNDING_PITCH = 26.0                # grounding camera downward tilt (V2); shallower than 32 deg
# Choice: 26 deg covers 1.5-9m targets in frame (row 86-329/360 = 24%-91% of image).
# 20 deg: too shallow — close targets (1.5m) fall off the top.
# 32 deg: too steep — far targets (7m+) fall off the bottom.
# 26 deg is the sweet spot that covers both easy (1.5-2.5m) and demo (4-9m) ranges.

# ---------------------------------------------------------------------------
# CAM-2 (Phase 1, docs/cam_opt2_multicam.md / docs/cam_p1.md): PROXIMITY camera.
#
# Same head mount as the ego/grounding cams (CAM_HEAD_Z, CAM_FWD unchanged — with the
# P0 cam.distance=1.0 fix, the rendered eye sits EXACTLY at that mount point for ANY
# pitch, so no new forward-offset constant is needed; grounding.CAM_ROBOT_FORWARD_OFFSET_M
# =0.10 applies unchanged to this camera too), just steeper tilt so the last ~0.2-1.8m
# of approach stays in-frame after the head/grounding cameras' near-cutoff (~0.7-0.9m).
#
# Geometry (docs/cam_opt2_multicam.md section 2, H=CAM_HEAD_Z+pelvis~1.29m, FOVY=45 deg):
#   58 deg pitch -> d_near~0.22m, d_far~1.81m  (overlaps grounding/ego cams' own near-cutoff
#   by a wide ~0.9m band, giving a safe hysteresis zone for the handoff in inferencer.py).
PROXIMITY_W, PROXIMITY_H = 320, 240   # cheapest of the three renders — close targets are large
PROXIMITY_PITCH = 58.0                # steep downward tilt — covers ~0.22-1.81m

# ---------------------------------------------------------------------------
# CAM-1 (Phase 2, docs/cam_opt1_widefov.md / docs/cam_p2.md): WIDE-FOV single-camera
# mode, an A/B alternative to the adopted CAM-2 champion above.
#
# HARD RULE: this is a config TOGGLE, never a replacement of the CAM-2 code paths.
# CAMERA_MODE defaults to 'cam2' (the champion, docs/cam_p1.md) — with the toggle at
# its default, every code path below that reads CAMERA_MODE is a no-op and behaviour
# is byte-identical to CAM-2. Set env var CAMERA_MODE=widefov to activate CAM-1 instead
# (single camera, same head mount, no proximity cam, no Schmitt handoff).
CAMERA_MODE: str = os.environ.get("CAMERA_MODE", "cam2").strip().lower()
if CAMERA_MODE not in ("cam2", "widefov"):
    raise ValueError(f"Unknown CAMERA_MODE={CAMERA_MODE!r} (expected 'cam2' or 'widefov')")

# Single wide-FOV renderer, same head mount (CAM_HEAD_Z, CAM_FWD unchanged) as the
# cam2 cameras. Geometry (docs/cam_opt1_widefov.md section 2, H=CAM_HEAD_Z+pelvis~1.34m):
# solving d_near=H/tan(theta+phi)=0.30m for FOVY=70 deg (phi=35 deg) gives theta~42.4 deg;
# resulting d_far=H/tan(theta-phi)~10.3m (theta>phi, so a real far cutoff exists,
# comfortably beyond the 8-9m demo range). 640x480 matches the existing TP_W/TP_H so
# the offscreen buffer sizing above (max(...,TP_W/TP_H)) already covers it with no
# change needed there.
WIDEFOV_W, WIDEFOV_H = 640, 480        # single renderer resolution
WIDEFOV_FOVY  = 70.0                    # vertical FOV (degrees) — probe value (task brief)
WIDEFOV_PITCH = 42.0                    # downward tilt solved for d_near~0.3m at FOVY=70

# Third-person camera defaults (for video only)
TP_W, TP_H   = 640, 480


# ---------------------------------------------------------------------------
# Helper: add a geom to a worldbody spec
# ---------------------------------------------------------------------------
def _add_geom(wb: mujoco.MjsBody, gtype: mujoco.mjtGeom,
              size: list[float] | tuple[float, ...],
              pos: list[float] | tuple[float, ...],
              rgba: list[float] | tuple[float, ...],
              name: str | None = None) -> mujoco.MjsGeom:
    """Add a geom to a worldbody spec.

    Args:
        wb: Worldbody spec to add the geom to.
        gtype: MuJoCo geom type (e.g. mujoco.mjtGeom.mjGEOM_BOX).
        size: Geom size parameters (meaning depends on gtype).
        pos: (x, y, z) position of the geom.
        rgba: (r, g, b, a) color, each channel in [0, 1].
        name: Optional geom name.

    Returns:
        The newly created geom spec.
    """
    g = wb.add_geom()
    g.type  = gtype
    g.size  = list(size)
    g.pos   = list(pos)
    g.rgba  = list(rgba)
    if name:
        g.name = name
    return g


def _rgb255_to_rgba1(rgb: tuple[int, int, int], alpha: float = 1.0) -> list[float]:
    """Convert a 0-255 RGB tuple to a 0-1 RGBA list."""
    return [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, alpha]


# ---------------------------------------------------------------------------
# Build the MjModel from a scene config produced by scene.py
# ---------------------------------------------------------------------------
def build_arena(scene_cfg: dict) -> mujoco.MjModel:
    """Build a compiled MjModel from a scene config.

    Args:
        scene_cfg: dict produced by scene.sample_scene(), must contain:
            arena_size  : float  — half-width of the square arena (m)
            objects     : list of dicts with keys:
                            color_name, color_rgb, shape_name, size, x, y
            lighting    : optional dict with 'ambient' float

    Returns:
        Compiled mujoco.MjModel, ready for mujoco.MjData.
    """
    arena_size = float(scene_cfg["arena_size"])
    objects    = scene_cfg["objects"]
    lighting   = scene_cfg.get("lighting", {})

    # ---- Load robot spec ----
    spec = mujoco.MjSpec.from_file(G1_XML)

    # Set offscreen buffer large enough for all cameras (ego, grounding, proximity, TP)
    try:
        spec.visual.global_.offwidth  = max(EGO_W, GROUNDING_W, PROXIMITY_W, TP_W)
        spec.visual.global_.offheight = max(EGO_H, GROUNDING_H, PROXIMITY_H, TP_H)
    except Exception:
        pass  # older mujoco — ignore

    wb = spec.worldbody

    # ---- Floor ----
    # g1_gear_wbc.xml already contains a 'floor' geom; skip adding one.
    # We update its colour after compile instead.

    # ---- Perimeter walls (thin boxes) ----
    half = arena_size
    wall_h   = 0.35
    wall_t   = 0.05
    wall_rgba = [0.80, 0.80, 0.82, 1.0]
    # +X wall
    _add_geom(wb, mujoco.mjtGeom.mjGEOM_BOX,
              [wall_t, half, wall_h],
              [ half, 0, wall_h], wall_rgba, "wall_px")
    # -X wall
    _add_geom(wb, mujoco.mjtGeom.mjGEOM_BOX,
              [wall_t, half, wall_h],
              [-half, 0, wall_h], wall_rgba, "wall_nx")
    # +Y wall
    _add_geom(wb, mujoco.mjtGeom.mjGEOM_BOX,
              [half, wall_t, wall_h],
              [0,  half, wall_h], wall_rgba, "wall_py")
    # -Y wall
    _add_geom(wb, mujoco.mjtGeom.mjGEOM_BOX,
              [half, wall_t, wall_h],
              [0, -half, wall_h], wall_rgba, "wall_ny")

    # ---- Objects ----
    for i, obj in enumerate(objects):
        rgba = _rgb255_to_rgba1(obj["color_rgb"])
        hs   = obj["size"] / 2.0
        ox, oy = float(obj["x"]), float(obj["y"])
        shape = obj["shape_name"]
        oname = f"obj_{i}"

        if shape == "ball":
            _add_geom(wb, mujoco.mjtGeom.mjGEOM_SPHERE,
                      [hs, hs, hs], [ox, oy, hs], rgba, oname)

        elif shape == "cube":
            _add_geom(wb, mujoco.mjtGeom.mjGEOM_BOX,
                      [hs, hs, hs], [ox, oy, hs], rgba, oname)

        elif shape == "cylinder":
            # MuJoCo cylinder size = [radius, half-height, 0]
            _add_geom(wb, mujoco.mjtGeom.mjGEOM_CYLINDER,
                      [hs, hs * 1.6, hs], [ox, oy, hs * 1.6], rgba, oname)

        elif shape == "cone":
            # No native cone in MuJoCo — approximate with a cylinder for
            # physics + a stacked narrower box for visual distinctiveness.
            # (A tapered capsule-like appearance)
            cone_h = hs * 2.2
            # Base cylinder (wide, short)
            _add_geom(wb, mujoco.mjtGeom.mjGEOM_CYLINDER,
                      [hs, cone_h * 0.5, hs],
                      [ox, oy, cone_h * 0.5], rgba, oname)
            # Top cap (narrow, shorter) — visual only marker
            top_rgba = [rgba[0] * 0.85, rgba[1] * 0.85, rgba[2] * 0.85, 1.0]
            _add_geom(wb, mujoco.mjtGeom.mjGEOM_BOX,
                      [hs * 0.35, hs * 0.35, cone_h * 0.45],
                      [ox, oy, cone_h + cone_h * 0.45], top_rgba, f"obj_{i}_tip")
        else:
            # Fallback: sphere
            _add_geom(wb, mujoco.mjtGeom.mjGEOM_SPHERE,
                      [hs, hs, hs], [ox, oy, hs], rgba, oname)

    # ---- Lighting ----
    ambient = float(lighting.get("ambient", 0.4))
    try:
        spec.visual.headlight.ambient = [ambient, ambient, ambient]
        spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        spec.visual.headlight.specular = [0.3, 0.3, 0.3]
    except Exception:
        pass

    # ---- CAM-1 (Phase 2, toggle): widen the ACTUAL rendered FOVY ----
    # cam_opt1_widefov.md Finding #1: MuJoCo's rendered FOVY for an mjCAMERA_FREE camera
    # comes from model.vis.global_.fovy (model-wide), which cam2 mode never sets (stays
    # at MuJoCo's compiled-in 45 deg default — exactly what grounding.EGO_FOVY_RENDERED
    # already assumes). Only in widefov mode do we override it here, so cam2's rendered
    # FOVY (and every cam2 intrinsics computation downstream) is completely untouched.
    if CAMERA_MODE == 'widefov':
        try:
            spec.visual.global_.fovy = WIDEFOV_FOVY
        except Exception:
            pass

    # ---- Compile ----
    model = spec.compile()

    # Make floor white/light grey (set rgba after compile)
    try:
        fid = model.geom("floor").id
        model.geom_rgba[fid] = [0.92, 0.92, 0.90, 1.0]
    except Exception:
        pass

    return model
