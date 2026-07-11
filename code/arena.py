"""
arena.py — Parametric MuJoCo arena for G1Nav.

Builds a MjSpec (or compiled MjModel) from:
  - Flat floor + low perimeter walls
  - G1 robot via g1_gear_wbc.xml
  - N colored primitive objects (ball/cube/cylinder/cone) from a fixed palette
  - Non-overlapping placement
  - Ego RGBD camera (robot-head-attached, tilted ~30–35° down)
  - Third-person camera (for videos only; never a model input)

Public API
----------
build_arena(scene_cfg) -> mujoco.MjModel
render_ego(model, data) -> (rgb, depth, intrinsics)
render_tp(renderer, model, data, tp_cam) -> rgb
"""

import os
import math

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

import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------
_HERE: str = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT: str = os.path.dirname(_HERE)
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
# CRITICAL FINDING (V2): With 32° camera downward tilt, targets at 7m+ fall BELOW
# the image entirely (pixel_row > image height). At 6m they appear at row 233/240
# (97%) — just outside the 5% bottom margin crop. This is why demo/classical
# detection rate is ~0% at 6-9m.
#
# Fix: Use a shallower 20° tilt for the grounding render. This keeps demo targets
# (4-9m) in the image at rows 236-279/360 (65-78%), well within detection bounds.
# The depth values from MuJoCo remain accurate. Bearing accuracy maintained via
# cam_to_egocentric() with GROUNDING_PITCH parameter.
#
# Resolution: 480x360 (1.5x) so targets are 2.25x larger in pixel area.
# Together: pitch fix + resolution increase enables 4-9m detection.
GROUNDING_W, GROUNDING_H = 480, 360   # grounding render resolution (V2)
GROUNDING_PITCH = 26.0                # grounding camera downward tilt (V2); shallower than 32°
# Choice: 26° covers 1.5-9m targets in frame (row 86-329/360 = 24%-91% of image).
# 20°: too shallow — close targets (1.5m) fall off the top.
# 32°: too steep — far targets (7m+) fall off the bottom.
# 26° is the sweet spot that covers both easy (1.5-2.5m) and demo (4-9m) ranges.

# ---------------------------------------------------------------------------
# CAM-2 (Phase 1, docs/cam_opt2_multicam.md / docs/cam_p1.md): PROXIMITY camera.
#
# Same head mount as the ego/grounding cams (CAM_HEAD_Z, CAM_FWD unchanged — with the
# P0 cam.distance=1.0 fix, the rendered eye sits EXACTLY at that mount point for ANY
# pitch, so no new forward-offset constant is needed; grounding.CAM_ROBOT_FORWARD_OFFSET_M
# =0.10 applies unchanged to this camera too), just steeper tilt so the last ~0.2-1.8m
# of approach stays in-frame after the head/grounding cameras' near-cutoff (~0.7-0.9m).
#
# Geometry (docs/cam_opt2_multicam.md §2, H=CAM_HEAD_Z+pelvis≈1.29m, FOVY=45°):
#   58° pitch -> d_near≈0.22m, d_far≈1.81m  (overlaps grounding/ego cams' own near-cutoff
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
# cam2 cameras. Geometry (docs/cam_opt1_widefov.md §2, H=CAM_HEAD_Z+pelvis≈1.34m):
# solving d_near=H/tan(theta+phi)=0.30m for FOVY=70° (phi=35°) gives theta≈42.4°;
# resulting d_far=H/tan(theta-phi)≈10.3m (theta>phi, so a real far cutoff exists,
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
    # at MuJoCo's compiled-in 45° default — exactly what grounding.EGO_FOVY_RENDERED
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


# ---------------------------------------------------------------------------
# Ego-camera helpers
# ---------------------------------------------------------------------------
def _set_ego_cam(cam: mujoco.MjvCamera, qpos: np.ndarray, yaw: float,
                 pitch_deg: float = CAM_PITCH) -> None:
    """Position the free camera to simulate a robot-head ego camera.

    Args:
        cam: MjvCamera to position in-place.
        qpos: Robot qpos array; qpos[0:3] is the pelvis (x, y, z) position.
        yaw: Robot heading (rad).
        pitch_deg: Downward camera tilt in degrees (default=CAM_PITCH=32°).
            For grounding renders, use GROUNDING_PITCH=20° so distant targets
            (4-9m) appear in the image rather than falling below the bottom edge.
    """
    px, py, pz = qpos[0], qpos[1], qpos[2]
    # Camera origin: slightly forward + at head height
    cx = px + CAM_FWD * math.cos(yaw)
    cy = py + CAM_FWD * math.sin(yaw)
    cz = pz + CAM_HEAD_Z

    pitch_rad = math.radians(pitch_deg)
    # Forward direction of camera (tilted down)
    dx = math.cos(pitch_rad) * math.cos(yaw)
    dy = math.cos(pitch_rad) * math.sin(yaw)
    dz = -math.sin(pitch_rad)

    cam.lookat[:] = [cx + dx, cy + dy, cz + dz]
    # P0 fix (2026-07-08): cam.distance was 0.001 ("nearly pinhole"), but MuJoCo's
    # free camera places the eye at `lookat - distance*forward`. With lookat set to
    # `origin + 1.0*forward_dir` above, the true eye was `origin + (1-distance)*forward_dir`
    # -- i.e. it silently DRIFTED with pitch (drifted ~0.947m fwd / ~0.53m low at 32 deg).
    # This is exactly why grounding.py needed the empirical, pitch-specific
    # CAM_ROBOT_FORWARD_OFFSET_M=0.947 hack (valid only at 32 deg).
    # Setting distance=1.0 makes (1-distance)=0, so the eye sits EXACTLY at
    # `origin` (cx,cy,cz) regardless of pitch -- decoupling camera position from
    # tilt. This generalizes to any pitch (needed for multi-cam / dynamic-tilt
    # options) and lets CAM_ROBOT_FORWARD_OFFSET_M collapse to the constant
    # CAM_FWD (recalibrated in grounding.py).
    cam.distance   = 1.0
    cam.azimuth    = math.degrees(yaw)
    cam.elevation  = -pitch_deg


def get_ego_intrinsics(w: int = EGO_W, h: int = EGO_H,
                       fovy_deg: float = EGO_FOVY) -> dict:
    """Return pinhole camera intrinsics for the ego camera.

    Args:
        w: Image width in pixels.
        h: Image height in pixels.
        fovy_deg: Vertical field of view in degrees.

    Returns:
        dict with keys: fx, fy, cx, cy (all in pixels), width, height, fovy_deg.
    """
    fovy_rad = math.radians(fovy_deg)
    fy = (h / 2.0) / math.tan(fovy_rad / 2.0)
    fovx_rad = 2.0 * math.atan(math.tan(fovy_rad / 2.0) * w / h)
    fx = (w / 2.0) / math.tan(fovx_rad / 2.0)
    return dict(fx=fx, fy=fy, cx=w / 2.0 - 0.5, cy=h / 2.0 - 0.5,
                width=w, height=h, fovy_deg=fovy_deg)


class ArenaRenderer:
    """
    Manages ego + third-person renderers for an arena model.

    V2: Added separate high-resolution grounding renderer (480x360) for improved
    detection at 4-9m demo distances. The grounding renderer uses FOVY=45° (same as
    ego camera) but at 1.5x resolution so distant targets occupy more pixels.

    Usage
    -----
    renderer = ArenaRenderer(model)
    rgb, depth, intr = renderer.render_ego(data, yaw)
    rgb_gr, depth_gr, intr_gr = renderer.render_grounding(data, yaw)
    tp_rgb            = renderer.render_tp(data, tp_cam)
    renderer.close()
    """

    def __init__(self, model: mujoco.MjModel,
                 ego_w: int = EGO_W, ego_h: int = EGO_H,
                 grounding_w: int = GROUNDING_W, grounding_h: int = GROUNDING_H,
                 proximity_w: int = PROXIMITY_W, proximity_h: int = PROXIMITY_H,
                 tp_w: int = TP_W, tp_h: int = TP_H,
                 widefov_w: int = WIDEFOV_W, widefov_h: int = WIDEFOV_H) -> None:
        """Allocate the ego/grounding/proximity/TP renderers and cameras.

        Args:
            model: Compiled arena MjModel to render.
            ego_w: Ego camera render width.
            ego_h: Ego camera render height.
            grounding_w: Grounding camera render width.
            grounding_h: Grounding camera render height.
            proximity_w: Proximity camera render width.
            proximity_h: Proximity camera render height.
            tp_w: Third-person camera render width.
            tp_h: Third-person camera render height.
            widefov_w: Wide-FOV camera render width (only used when
                CAMERA_MODE=='widefov').
            widefov_h: Wide-FOV camera render height (only used when
                CAMERA_MODE=='widefov').
        """
        self._model    = model
        self._ego_w    = ego_w
        self._ego_h    = ego_h
        self._grounding_w = grounding_w
        self._grounding_h = grounding_h
        self._proximity_w = proximity_w
        self._proximity_h = proximity_h
        self._ego_rend = mujoco.Renderer(model, ego_h, ego_w)
        # V2: dedicated grounding renderer at higher resolution
        # Reuse a single renderer (not created per-call) to avoid EGL context exhaustion
        self._gr_rend  = mujoco.Renderer(model, grounding_h, grounding_w)
        # CAM-2 (Phase 1): dedicated proximity renderer, same pre-allocate-once pattern
        # (no EGL context exhaustion — precedent from _gr_rend). Only ever rendered when
        # the Schmitt-trigger handoff in inferencer.py has selected the proximity camera,
        # so steady-state per-cycle cost is unchanged (one render per grounding cycle).
        self._prox_rend = mujoco.Renderer(model, proximity_h, proximity_w)
        self._tp_rend  = mujoco.Renderer(model, tp_h, tp_w)
        self._egc      = mujoco.MjvCamera()
        self._egc.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._gr_cam   = mujoco.MjvCamera()
        self._gr_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._prox_cam = mujoco.MjvCamera()
        self._prox_cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        # CAM-1 (Phase 2, toggle): additional wide-FOV renderer, built ONLY when
        # CAMERA_MODE=='widefov'. Purely additive — every line above is constructed
        # exactly as before regardless of mode, so cam2's renderer set (and EGL context
        # count) is completely unaffected when the toggle is at its 'cam2' default.
        self._widefov_w    = widefov_w
        self._widefov_h    = widefov_h
        self._widefov_rend = None
        self._widefov_cam  = None
        if CAMERA_MODE == 'widefov':
            self._widefov_rend = mujoco.Renderer(model, widefov_h, widefov_w)
            self._widefov_cam  = mujoco.MjvCamera()
            self._widefov_cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def render_ego(self, data: mujoco.MjData, yaw: float,
                   render_depth: bool = True) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """Render the ego (head) camera at native 320x240 resolution.

        Args:
            data: MuJoCo simulation data (provides qpos for camera placement).
            yaw: Robot heading (rad) used to orient the ego camera.
            render_depth: If False, skip depth rendering (saves ~15ms/frame on EGL).

        Returns:
            rgb: np.ndarray  shape (H,W,3)  uint8
            depth: np.ndarray  shape (H,W)  float32  (metres) or None
            intr: dict  {fx,fy,cx,cy,width,height,fovy_deg}
        """
        _set_ego_cam(self._egc, data.qpos, yaw)

        self._ego_rend.update_scene(data, self._egc)
        rgb = self._ego_rend.render().copy()

        depth = None
        if render_depth:
            self._ego_rend.enable_depth_rendering()
            self._ego_rend.update_scene(data, self._egc)
            depth = self._ego_rend.render().copy().astype(np.float32)
            self._ego_rend.disable_depth_rendering()

        intr = get_ego_intrinsics(self._ego_w, self._ego_h, EGO_FOVY)
        return rgb, depth, intr

    def render_grounding(self, data: mujoco.MjData, yaw: float,
                         render_depth: bool = True) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """Render at higher-resolution (480x360) with shallower pitch for demo-distance grounding.

        V2 key changes vs render_ego:
        1. Resolution: 480x360 (1.5x) → targets are 2.25x larger in pixel area.
        2. Camera pitch: 20° (vs 32° for ego) → targets at 4-9m stay in frame.
           At 32°: target at 7m falls below image. At 20°: target at 9m at row 279/360.
        3. Same EGL context (single renderer object) → no context exhaustion.

        Bearing calculation: cam_to_egocentric() must use GROUNDING_PITCH (not CAM_PITCH)
        to correctly un-pitch the camera frame. This is handled in grounding.py.

        Args:
            data: MuJoCo simulation data (provides qpos for camera placement).
            yaw: Robot heading (rad) used to orient the grounding camera.
            render_depth: If False, skip depth rendering.

        Returns:
            rgb: np.ndarray  shape (360,480,3)  uint8
            depth: np.ndarray  shape (360,480)  float32  (metres) or None
            intr: dict  {fx,fy,cx,cy,width,height,fovy_deg,pitch_deg}
        """
        _set_ego_cam(self._gr_cam, data.qpos, yaw, pitch_deg=GROUNDING_PITCH)

        self._gr_rend.update_scene(data, self._gr_cam)
        rgb = self._gr_rend.render().copy()

        depth = None
        if render_depth:
            self._gr_rend.enable_depth_rendering()
            self._gr_rend.update_scene(data, self._gr_cam)
            depth = self._gr_rend.render().copy().astype(np.float32)
            self._gr_rend.disable_depth_rendering()

        # Intrinsics for grounding resolution (FOVY=45° same as ego camera, scaled for 480x360)
        from code.grounding import get_ego_intrinsics_rendered
        intr = get_ego_intrinsics_rendered(self._grounding_w, self._grounding_h)
        # Include pitch so cam_to_egocentric can use the correct tilt
        intr['pitch_deg'] = GROUNDING_PITCH
        return rgb, depth, intr

    def render_proximity(self, data: mujoco.MjData, yaw: float,
                         render_depth: bool = True) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """CAM-2 (Phase 1): render the steep-pitch proximity camera, same head mount as
        render_ego/render_grounding (only PROXIMITY_PITCH differs — no XML change, no
        new offset calibration needed post-P0, see arena.py constants comment).

        Covers ~0.22-1.81m (docs/cam_opt2_multicam.md geometry table), i.e. the final
        close-range approach where the shallower ego/grounding cameras' target has
        already fallen below the frame. Selected by the Schmitt-trigger handoff in
        inferencer.py — NOT rendered every cycle, only when active_cam==PROXIMITY.

        Args:
            data: MuJoCo simulation data (provides qpos for camera placement).
            yaw: Robot heading (rad) used to orient the proximity camera.
            render_depth: If False, skip depth rendering.

        Returns:
            rgb: np.ndarray  shape (PROXIMITY_H,PROXIMITY_W,3)  uint8
            depth: np.ndarray  shape (PROXIMITY_H,PROXIMITY_W)  float32  (metres) or None
            intr: dict  {fx,fy,cx,cy,width,height,fovy_deg,pitch_deg,is_proximity}
        """
        _set_ego_cam(self._prox_cam, data.qpos, yaw, pitch_deg=PROXIMITY_PITCH)

        self._prox_rend.update_scene(data, self._prox_cam)
        rgb = self._prox_rend.render().copy()

        depth = None
        if render_depth:
            self._prox_rend.enable_depth_rendering()
            self._prox_rend.update_scene(data, self._prox_cam)
            depth = self._prox_rend.render().copy().astype(np.float32)
            self._prox_rend.disable_depth_rendering()

        from code.grounding import get_ego_intrinsics_rendered
        intr = get_ego_intrinsics_rendered(self._proximity_w, self._proximity_h)
        intr['pitch_deg']    = PROXIMITY_PITCH
        # Flag consumed by grounding.ground() to activate the stricter self-body-rejection
        # depth floor/geometry check (steeper pitch -> optical center closer to the robot's
        # own chest, see docs/cam_opt2_multicam.md "Real risk" + docs/cam_p0.md ep14 finding).
        intr['is_proximity'] = True
        return rgb, depth, intr

    def render_widefov(self, data: mujoco.MjData, yaw: float,
                       render_depth: bool = True) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """CAM-1 (Phase 2, toggle, docs/cam_opt1_widefov.md / docs/cam_p2.md): render the
        single wide-FOV camera, same head mount as render_ego/render_grounding
        (CAM_HEAD_Z, CAM_FWD unchanged), pitch=WIDEFOV_PITCH, FOVY=WIDEFOV_FOVY (the
        actual rendered FOVY, set at build time via spec.visual.global_.fovy in
        build_arena() — only when CAMERA_MODE=='widefov', so this method is only
        meaningful/called in that mode; the renderer is None otherwise, see __init__).

        No proximity camera, no handoff — this single render is the entire grounding
        camera for CAM-1.

        Args:
            data: MuJoCo simulation data (provides qpos for camera placement).
            yaw: Robot heading (rad) used to orient the wide-FOV camera.
            render_depth: If False, skip depth rendering.

        Returns:
            rgb: np.ndarray  shape (WIDEFOV_H,WIDEFOV_W,3)  uint8
            depth: np.ndarray  shape (WIDEFOV_H,WIDEFOV_W)  float32  (metres) or None
            intr: dict  {fx,fy,cx,cy,width,height,fovy_deg,pitch_deg,is_widefov}
        """
        _set_ego_cam(self._widefov_cam, data.qpos, yaw, pitch_deg=WIDEFOV_PITCH)

        self._widefov_rend.update_scene(data, self._widefov_cam)
        rgb = self._widefov_rend.render().copy()

        depth = None
        if render_depth:
            self._widefov_rend.enable_depth_rendering()
            self._widefov_rend.update_scene(data, self._widefov_cam)
            depth = self._widefov_rend.render().copy().astype(np.float32)
            self._widefov_rend.disable_depth_rendering()

        # Actual rendered FOVY (WIDEFOV_FOVY) fed straight into the standard pinhole
        # intrinsics formula — this is exactly cam_opt1_widefov.md Finding #1's fix,
        # scoped to the widefov path only (cam2's get_ego_intrinsics_rendered() hardcode
        # of 45 deg is untouched).
        intr = get_ego_intrinsics(self._widefov_w, self._widefov_h, WIDEFOV_FOVY)
        intr['pitch_deg']  = WIDEFOV_PITCH
        intr['is_widefov'] = True
        return rgb, depth, intr

    def render_tp(self, data: mujoco.MjData, tp_cam: mujoco.MjvCamera) -> np.ndarray:
        """Render third-person view (for video only)."""
        self._tp_rend.update_scene(data, tp_cam)
        return self._tp_rend.render().copy()

    def make_tp_cam(self) -> mujoco.MjvCamera:
        """Create a default third-person MjvCamera (not yet tracking any body)."""
        cam        = mujoco.MjvCamera()
        cam.type   = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance  = 5.0
        cam.azimuth   = 135.0
        cam.elevation = -20.0
        return cam

    def update_tp_cam(self, tp_cam: mujoco.MjvCamera, data: mujoco.MjData,
                      distance: float = 5.0) -> None:
        """Track the robot with the third-person camera."""
        bxy = data.qpos[0:2]
        tp_cam.lookat[:] = [bxy[0], bxy[1], 0.5]
        tp_cam.distance   = distance
        tp_cam.azimuth    = 135.0
        tp_cam.elevation  = -20.0

    def close(self) -> None:
        """Close all underlying EGL renderers."""
        self._ego_rend.close()
        self._gr_rend.close()
        self._prox_rend.close()
        self._tp_rend.close()
        if self._widefov_rend is not None:
            self._widefov_rend.close()


# ---------------------------------------------------------------------------
# Back-projection helper (used by grounding.py)
# ---------------------------------------------------------------------------
def backproject_pixel(u: float, v: float, depth_m: float, intr: dict) -> np.ndarray:
    """Back-project image pixel (u,v) at given depth (metres) to camera-frame 3D point.

    Camera frame: x=right, y=down, z=forward (OpenCV convention).

    Args:
        u: Pixel column (x, right).
        v: Pixel row (y, down).
        depth_m: Depth at (u, v) in metres.
        intr: Camera intrinsics dict with keys fx, fy, cx, cy.

    Returns:
        np.ndarray shape (3,)  [x_cam, y_cam, z_cam]
    """
    x = (u - intr["cx"]) * depth_m / intr["fx"]
    y = (v - intr["cy"]) * depth_m / intr["fy"]
    z = depth_m
    return np.array([x, y, z], dtype=np.float32)


if __name__ == "__main__":
    # Quick smoke test: build a minimal scene and render one frame
    import sys

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
