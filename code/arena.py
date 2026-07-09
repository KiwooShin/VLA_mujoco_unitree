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

import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_WBC_ROOT = os.path.join(
    _REPO_ROOT,
    "third_party/Isaac-GR00T/external_dependencies/"
    "GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1",
)
G1_XML = os.path.join(_WBC_ROOT, "g1_gear_wbc.xml")

# ---------------------------------------------------------------------------
# Fixed color palette  (name, RGB-uint8)
# ---------------------------------------------------------------------------
COLORS = [
    ("red",     (220,  40,  40)),
    ("yellow",  (235, 205,  40)),
    ("blue",    ( 50,  90, 220)),
    ("green",   ( 40, 180,  70)),
    ("orange",  (240, 140,  30)),
    ("purple",  (150,  60, 200)),
    ("cyan",    ( 40, 200, 210)),
]

# Shape definitions  (name, half-size for placement radius)
SHAPES = [
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

# Third-person camera defaults (for video only)
TP_W, TP_H   = 640, 480


# ---------------------------------------------------------------------------
# Helper: add a geom to a worldbody spec
# ---------------------------------------------------------------------------
def _add_geom(wb, gtype, size, pos, rgba, name=None):
    g = wb.add_geom()
    g.type  = gtype
    g.size  = list(size)
    g.pos   = list(pos)
    g.rgba  = list(rgba)
    if name:
        g.name = name
    return g


def _rgb255_to_rgba1(rgb, alpha=1.0):
    return [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, alpha]


# ---------------------------------------------------------------------------
# Build the MjModel from a scene config produced by scene.py
# ---------------------------------------------------------------------------
def build_arena(scene_cfg: dict) -> mujoco.MjModel:
    """
    Parameters
    ----------
    scene_cfg : dict produced by scene.sample_scene(), must contain:
        arena_size  : float  — half-width of the square arena (m)
        objects     : list of dicts with keys:
                        color_name, color_rgb, shape_name, size, x, y
        lighting    : optional dict with 'ambient' float

    Returns
    -------
    mujoco.MjModel  (compiled, ready for mujoco.MjData)
    """
    arena_size = float(scene_cfg["arena_size"])
    objects    = scene_cfg["objects"]
    lighting   = scene_cfg.get("lighting", {})

    # ---- Load robot spec ----
    spec = mujoco.MjSpec.from_file(G1_XML)

    # Set offscreen buffer large enough for all cameras (ego, grounding, TP)
    try:
        spec.visual.global_.offwidth  = max(EGO_W, GROUNDING_W, TP_W)
        spec.visual.global_.offheight = max(EGO_H, GROUNDING_H, TP_H)
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

    Parameters
    ----------
    pitch_deg : downward camera tilt in degrees (default=CAM_PITCH=32°).
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
    cam.distance   = 0.001          # very close (nearly pinhole)
    cam.azimuth    = math.degrees(yaw)
    cam.elevation  = -pitch_deg


def get_ego_intrinsics(w: int = EGO_W, h: int = EGO_H,
                       fovy_deg: float = EGO_FOVY) -> dict:
    """
    Return pinhole camera intrinsics for the ego camera.

    Returns
    -------
    dict with keys: fx, fy, cx, cy (all in pixels)
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
                 tp_w: int = TP_W, tp_h: int = TP_H):
        self._model    = model
        self._ego_w    = ego_w
        self._ego_h    = ego_h
        self._grounding_w = grounding_w
        self._grounding_h = grounding_h
        self._ego_rend = mujoco.Renderer(model, ego_h, ego_w)
        # V2: dedicated grounding renderer at higher resolution
        # Reuse a single renderer (not created per-call) to avoid EGL context exhaustion
        self._gr_rend  = mujoco.Renderer(model, grounding_h, grounding_w)
        self._tp_rend  = mujoco.Renderer(model, tp_h, tp_w)
        self._egc      = mujoco.MjvCamera()
        self._egc.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._gr_cam   = mujoco.MjvCamera()
        self._gr_cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def render_ego(self, data: mujoco.MjData, yaw: float,
                   render_depth: bool = True):
        """
        Render the ego (head) camera at native 320x240 resolution.

        Parameters
        ----------
        render_depth : if False, skip depth rendering (saves ~15ms/frame on EGL).

        Returns
        -------
        rgb   : np.ndarray  shape (H,W,3)  uint8
        depth : np.ndarray  shape (H,W)    float32   (metres) or None
        intr  : dict        {fx,fy,cx,cy,width,height,fovy_deg}
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
                         render_depth: bool = True):
        """
        Render at higher-resolution (480x360) with shallower pitch for demo-distance grounding.

        V2 key changes vs render_ego:
        1. Resolution: 480x360 (1.5x) → targets are 2.25x larger in pixel area.
        2. Camera pitch: 20° (vs 32° for ego) → targets at 4-9m stay in frame.
           At 32°: target at 7m falls below image. At 20°: target at 9m at row 279/360.
        3. Same EGL context (single renderer object) → no context exhaustion.

        Bearing calculation: cam_to_egocentric() must use GROUNDING_PITCH (not CAM_PITCH)
        to correctly un-pitch the camera frame. This is handled in grounding.py.

        Returns
        -------
        rgb   : np.ndarray  shape (360,480,3)  uint8
        depth : np.ndarray  shape (360,480)    float32   (metres) or None
        intr  : dict        {fx,fy,cx,cy,width,height,fovy_deg,pitch_deg}
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

    def render_tp(self, data: mujoco.MjData, tp_cam: mujoco.MjvCamera):
        """Render third-person view (for video only)."""
        self._tp_rend.update_scene(data, tp_cam)
        return self._tp_rend.render().copy()

    def make_tp_cam(self) -> mujoco.MjvCamera:
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

    def close(self):
        self._ego_rend.close()
        self._gr_rend.close()
        self._tp_rend.close()


# ---------------------------------------------------------------------------
# Back-projection helper (used by grounding.py)
# ---------------------------------------------------------------------------
def backproject_pixel(u: float, v: float, depth_m: float, intr: dict) -> np.ndarray:
    """
    Back-project image pixel (u,v) at given depth (metres) to camera-frame 3D point.

    Camera frame: x=right, y=down, z=forward (OpenCV convention).

    Returns
    -------
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
