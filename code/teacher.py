"""
WBCTeacher: wraps the NVIDIA GR00T WBC Walk ONNX policy for the Unitree G1 humanoid.

The teacher maps a velocity command (vx, vy, omega_z) + proprio history -> 15 lower-body
joint position targets at 50 Hz (control_dt = 0.02 s = 4 physics substeps of 0.005 s each).

Observation layout (single frame, 86-d, stacked 6x -> 516-d input to the ONNX):
  [0:7]           command: [vx*2, vy*2, wz*0.5, height_cmd, 0, rpy[0], rpy[1], rpy[2]]
                  (cmd_scale=[2,2,0.5]; element [4] is freq_cmd but hardcoded 0 in non-gait policy)
  [7:10]          pelvis angular velocity (rad/s) * 0.5
  [10:13]         gravity orientation vector (rotate [0,0,-1] by inverse pelvis quat)
  [13:13+nj]      joint positions - default_angles, scaled by 1.0
  [13+nj:13+2nj]  joint velocities * 0.05
  [13+2nj:13+2nj+15]  previous action (raw network output, not target_dof)

  Where nj = total DOF of G1 (29 for the gear_wbc model).
  Single-frame dim = 86 (hardcoded; remaining slots stay zero).
  History length = 6; full obs = 6 * 86 = 516.

Joint order (15 lower-body, indices 0..14 in qpos[7:22], ctrl[0:15]):
  0: left_hip_pitch
  1: left_hip_roll
  2: left_hip_yaw
  3: left_knee
  4: left_ankle_pitch
  5: left_ankle_roll
  6: right_hip_pitch
  7: right_hip_roll
  8: right_hip_yaw
  9: right_knee
  10: right_ankle_pitch
  11: right_ankle_roll
  12: waist_yaw
  13: waist_roll
  14: waist_pitch   (or torso depending on XML ordering)

Default joint angles (rad):
  legs L: [-0.1, 0, 0, 0.3, -0.2, 0]
  legs R: [-0.1, 0, 0, 0.3, -0.2, 0]
  waist:   [0, 0, 0]

PD gains (applied on 15 lower-body joints):
  kp: [150, 150, 150, 200, 40, 40,  150, 150, 150, 200, 40, 40,  250, 250, 250]
  kd: [2,   2,   2,   4,   2,  2,   2,   2,   2,   4,   2,  2,   5,   5,   5]

Arms/waist upper DOF (indices 15..28): held at zero with kp=100, kd=0.5.

Control rate: 50 Hz (control_decimation=4, sim_dt=0.005 s).
Physics substeps: 4 per control step.
"""

import os
import collections
import math
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import mujoco
import onnxruntime as ort
import yaml

# ---- Canonical paths -----------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_WBC_ROOT = os.path.join(
    _REPO_ROOT,
    "third_party/Isaac-GR00T/external_dependencies/"
    "GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1",
)

G1_XML = os.path.join(_WBC_ROOT, "g1_gear_wbc.xml")
CFG_YAML = os.path.join(_WBC_ROOT, "g1_gear_wbc.yaml")
WALK_ONNX = os.path.join(_WBC_ROOT, "policy/GR00T-WholeBodyControl-Walk.onnx")

# ---- Simulation constants (from g1_gear_wbc.yaml) ------------------------------
SIM_DT = 0.005          # physics timestep (s)
CONTROL_DECIMATION = 4  # physics steps per policy step
CONTROL_DT = SIM_DT * CONTROL_DECIMATION   # 0.02 s -> 50 Hz

# ---- Policy constants ----------------------------------------------------------
NUM_ACTIONS = 15
NUM_OBS = 516          # 86 * 6
OBS_HISTORY_LEN = 6
SINGLE_OBS_DIM = 86

DEFAULT_ANGLES = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # right leg
     0.0, 0.0, 0.0,                     # waist
], dtype=np.float32)

KPS = np.array([150, 150, 150, 200, 40, 40,
                150, 150, 150, 200, 40, 40,
                250, 250, 250], dtype=np.float32)

KDS = np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
                2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
                5.0, 5.0, 5.0], dtype=np.float32)

ANG_VEL_SCALE = 0.5
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)  # vx, vy, omega_z scales
HEIGHT_CMD = 0.74
RPY_CMD = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# Standing reset height (pelvis z from g1_gear_wbc.xml body pos="0 0 0.793")
RESET_HEIGHT = 0.79


# ---- Helpers -------------------------------------------------------------------

def _grav_orient(quat: np.ndarray) -> np.ndarray:
    """Rotate gravity vector [0, 0, -1] into the body frame via inverse quaternion.

    Args:
        quat: Orientation quaternion [w, x, y, z] (MuJoCo convention).

    Returns:
        np.ndarray: The gravity vector expressed in the body frame.
    """
    w, x, y, z = quat
    # conjugate (inverse for unit quaternion): [w, -x, -y, -z]
    qc = np.array([w, -x, -y, -z], dtype=np.float64)
    v = np.array([0.0, 0.0, -1.0])
    # rotate v by qc
    return np.array([
        v[0]*(qc[0]**2+qc[1]**2-qc[2]**2-qc[3]**2)
          + v[1]*2*(qc[1]*qc[2]-qc[0]*qc[3])
          + v[2]*2*(qc[1]*qc[3]+qc[0]*qc[2]),
        v[0]*2*(qc[1]*qc[2]+qc[0]*qc[3])
          + v[1]*(qc[0]**2-qc[1]**2+qc[2]**2-qc[3]**2)
          + v[2]*2*(qc[2]*qc[3]-qc[0]*qc[1]),
        v[0]*2*(qc[1]*qc[3]-qc[0]*qc[2])
          + v[1]*2*(qc[2]*qc[3]+qc[0]*qc[1])
          + v[2]*(qc[0]**2-qc[1]**2-qc[2]**2+qc[3]**2),
    ], dtype=np.float32)


def _yaw_of(quat: np.ndarray) -> float:
    """Extract yaw (rotation around world Z) from MuJoCo quaternion [w,x,y,z].

    Args:
        quat: Orientation quaternion [w, x, y, z] (MuJoCo convention).

    Returns:
        Yaw angle in radians.
    """
    w, x, y, z = quat
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


# ---- WBCTeacher ---------------------------------------------------------------

class WBCTeacher:
    """
    Wraps the NVIDIA GR00T WBC Walk ONNX for headless MuJoCo simulation.

    Usage
    -----
    teacher = WBCTeacher()
    teacher.reset()
    for _ in range(100):
        targets = teacher.step(vel_cmd=(0.5, 0.0, 0.0))   # returns (15,) array

    The teacher internally handles physics substeps and history buffering.
    Call teacher.step() at 50 Hz; it advances physics 4x per call.
    """

    def __init__(
        self,
        xml_path: str = G1_XML,
        onnx_path: str = WALK_ONNX,
        use_gpu: bool = True,
    ) -> None:
        """Load the ONNX Walk policy and the MuJoCo model, and init state.

        Args:
            xml_path: Path to the MuJoCo G1 model XML.
            onnx_path: Path to the ONNX Walk policy.
            use_gpu: If True, prefer CUDAExecutionProvider (falls back to
                CPU if unavailable).
        """
        # ---- Load ONNX ---
        providers = []
        if use_gpu:
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.device_str = "GPU (CUDA)"
            else:
                providers = ["CPUExecutionProvider"]
                self.device_str = "CPU"
        else:
            providers = ["CPUExecutionProvider"]
            self.device_str = "CPU"

        self._sess = ort.InferenceSession(onnx_path, providers=providers)
        self._iname = self._sess.get_inputs()[0].name

        # ---- Load MuJoCo model ---
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)

        # DOF count (excludes 7 freejoint DOFs: 3 pos + 4 quat)
        self._nj = self.model.nq - 7  # typically 29 for g1_gear_wbc

        # Body indices for readout
        self._pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        # Policy state
        self._action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self._target_dof = DEFAULT_ANGLES.copy()
        self._obs_history = collections.deque(
            [np.zeros(SINGLE_OBS_DIM, dtype=np.float32)] * OBS_HISTORY_LEN,
            maxlen=OBS_HISTORY_LEN,
        )
        self._obs_buf = np.zeros(NUM_OBS, dtype=np.float32)

        # Counter for decimation
        self._substep_counter = 0

    # ---- Public API ---

    def reset(
        self,
        pos_xy: tuple = (0.0, 0.0),
        yaw: float = 0.0,
    ) -> None:
        """Reset the robot to a standing pose.

        Args:
            pos_xy: (x, y) world position to reset the pelvis to.
            yaw: World yaw (radians) to reset the pelvis orientation to.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = pos_xy[0]
        self.data.qpos[1] = pos_xy[1]
        self.data.qpos[2] = RESET_HEIGHT
        # Quaternion for given yaw: [cos(yaw/2), 0, 0, sin(yaw/2)]
        self.data.qpos[3] = math.cos(yaw / 2)
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = math.sin(yaw / 2)
        # Set default joint angles
        self.data.qpos[7:7 + NUM_ACTIONS] = DEFAULT_ANGLES
        # Zero velocities
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Reset policy state
        self._action[:] = 0.0
        self._target_dof = DEFAULT_ANGLES.copy()
        self._obs_history = collections.deque(
            [np.zeros(SINGLE_OBS_DIM, dtype=np.float32)] * OBS_HISTORY_LEN,
            maxlen=OBS_HISTORY_LEN,
        )
        self._obs_buf[:] = 0.0
        self._substep_counter = 0

    def step(
        self,
        vel_cmd: tuple = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """
        Run one control step (50 Hz):
          1. Build observation from current state + history.
          2. Run the ONNX Walk policy.
          3. Apply joint targets via PD control for CONTROL_DECIMATION substeps.
          4. Return the 15 joint targets (rad, in DEFAULT_ANGLES frame).

        Args:
            vel_cmd: (vx, vy, omega_z) in m/s and rad/s.

        Returns:
            np.ndarray of shape (15,): absolute joint position targets (rad).
        """
        # Build obs
        single_obs = self._build_single_obs(vel_cmd)
        self._obs_history.append(single_obs)

        for i, h in enumerate(self._obs_history):
            self._obs_buf[i * SINGLE_OBS_DIM:(i + 1) * SINGLE_OBS_DIM] = h

        # Run policy
        obs_in = self._obs_buf[None].astype(np.float32)  # (1, 516)
        raw_action = self._sess.run(None, {self._iname: obs_in})[0].squeeze()  # (15,)
        self._action = raw_action.astype(np.float32)
        self._target_dof = self._action * ACTION_SCALE + DEFAULT_ANGLES

        # Physics substeps
        for _ in range(CONTROL_DECIMATION):
            self._apply_pd()
            mujoco.mj_step(self.model, self.data)

        return self._target_dof.copy()

    # ---- Read-only state accessors ---

    @property
    def base_pos(self) -> np.ndarray:
        """Pelvis position [x, y, z]."""
        return self.data.qpos[0:3].copy()

    @property
    def base_height(self) -> float:
        """Pelvis height (z, metres)."""
        return float(self.data.qpos[2])

    @property
    def base_yaw(self) -> float:
        """Pelvis yaw (radians)."""
        return _yaw_of(self.data.qpos[3:7])

    @property
    def sim_time(self) -> float:
        """Elapsed simulation time (seconds)."""
        return float(self.data.time)

    # ---- Internal helpers ---

    def _build_single_obs(self, vel_cmd: tuple) -> np.ndarray:
        """Build a single 86-d observation frame.

        Args:
            vel_cmd: (vx, vy, omega_z) in m/s and rad/s.

        Returns:
            np.ndarray of shape (86,): the single-frame observation.
        """
        vx, vy, wz = vel_cmd

        # Command vector (7 elements):
        # [0:3] loco_cmd * CMD_SCALE
        # [3]   height_cmd
        # [4]   0  (freq_cmd slot; the Walk policy uses the non-gait variant -> zero)
        # [5:8] rpy_cmd (zeroed)
        command = np.zeros(7, dtype=np.float32)
        command[0] = vx * CMD_SCALE[0]
        command[1] = vy * CMD_SCALE[1]
        command[2] = wz * CMD_SCALE[2]
        command[3] = HEIGHT_CMD
        # command[4] is freq_cmd -> leave 0 (run_mujoco_gear_wbc.py style, not gait)
        command[4:7] = RPY_CMD

        # Proprio
        nj = self._nj
        qj = self.data.qpos[7:7 + nj].copy()
        dqj = self.data.qvel[6:6 + nj].copy()
        quat = self.data.qpos[3:7].copy()    # [w, x, y, z]
        omega = self.data.qvel[3:6].copy()   # pelvis angular velocity (world frame)

        # Defaults padded to nj length
        pad = np.zeros(nj, dtype=np.float32)
        pad[:NUM_ACTIONS] = DEFAULT_ANGLES[:NUM_ACTIONS]

        qj_scaled = (qj - pad) * DOF_POS_SCALE
        dqj_scaled = dqj * DOF_VEL_SCALE
        grav = _grav_orient(quat)
        omega_scaled = omega * ANG_VEL_SCALE

        obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
        obs[0:7] = command
        obs[7:10] = omega_scaled
        obs[10:13] = grav
        obs[13:13 + nj] = qj_scaled
        obs[13 + nj:13 + 2 * nj] = dqj_scaled
        obs[13 + 2 * nj:13 + 2 * nj + 15] = self._action  # previous action
        return obs

    def _apply_pd(self) -> None:
        """Apply PD torques for the current target_dof to ctrl."""
        nj = self._nj
        # Lower body (15 joints): PD toward target_dof
        leg_tau = (
            (self._target_dof - self.data.qpos[7:7 + NUM_ACTIONS]) * KPS
            + (0.0 - self.data.qvel[6:6 + NUM_ACTIONS]) * KDS
        )
        self.data.ctrl[:NUM_ACTIONS] = leg_tau

        # Upper body (remaining joints): hold at zero with stiff PD
        if nj > NUM_ACTIONS:
            n_upper = nj - NUM_ACTIONS
            arm_tau = (
                (0.0 - self.data.qpos[7 + NUM_ACTIONS:7 + nj]) * 100.0
                + (0.0 - self.data.qvel[6 + NUM_ACTIONS:6 + nj]) * 0.5
            )
            self.data.ctrl[NUM_ACTIONS:nj] = arm_tau


# ---- Smoke test ----------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import imageio.v2 as imageio

    os.makedirs("eval/teacher_smoke", exist_ok=True)

    # ---- EGL env vars for headless rendering ---
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                          "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

    print("=" * 60)
    print("WBCTeacher Smoke Test")
    print("=" * 60)

    teacher = WBCTeacher()
    print(f"ONNX provider: {teacher.device_str}")
    print(f"Model DOFs (nj): {teacher._nj}")
    print(f"Single obs dim: {SINGLE_OBS_DIM}, total obs: {NUM_OBS}")

    # ---- Renderer for video ---
    RENDER_W, RENDER_H = 640, 480
    renderer = mujoco.Renderer(teacher.model, RENDER_H, RENDER_W)
    tp_cam = mujoco.MjvCamera()
    tp_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    frames = []

    def render_frame(teacher: WBCTeacher, tp_cam: mujoco.MjvCamera) -> np.ndarray:
        """Render a third-person frame tracking the robot.

        Args:
            teacher: Active WBCTeacher instance being tracked.
            tp_cam: Third-person MjvCamera to position on the robot.

        Returns:
            np.ndarray: The rendered RGB frame.
        """
        # Third-person: track the robot
        bxy = teacher.data.qpos[0:2]
        tp_cam.lookat[:] = [bxy[0], bxy[1], 0.5]
        tp_cam.distance = 3.5
        tp_cam.azimuth = 140.0
        tp_cam.elevation = -20.0
        renderer.update_scene(teacher.data, tp_cam)
        return renderer.render().copy()

    # ---- Phase 0: settle for 1s (zero cmd) before we start timing ---
    teacher.reset()
    for _ in range(50):  # 50 * 0.02s = 1s settle
        teacher.step(vel_cmd=(0.0, 0.0, 0.0))

    # ---- Phase 1: Stand 2 s (zero vel cmd) ---
    print("\n--- Phase 1: Stand (zero vel cmd, 2 s) ---")
    teacher.reset()
    MIN_HEIGHT = 0.55  # pelvis height threshold for "upright"
    stand_ok = True
    t0 = time.perf_counter()

    step_times = []
    for i in range(100):  # 100 * 0.02 s = 2 s
        ts = time.perf_counter()
        teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        step_times.append((time.perf_counter() - ts) * 1000)
        h = teacher.base_height
        frames.append(render_frame(teacher, tp_cam))
        if i % 25 == 0:
            print(f"  t={teacher.sim_time:.2f}s  pos={teacher.base_pos}  yaw={teacher.base_yaw:.3f} rad  h={h:.3f}")
        if h < MIN_HEIGHT:
            stand_ok = False
            print(f"  FALL at t={teacher.sim_time:.2f}s, height={h:.3f}")
            break

    elapsed_1 = time.perf_counter() - t0
    p1 = "PASS" if stand_ok else "FAIL"
    print(f"Phase 1: {p1}  (min_height_threshold={MIN_HEIGHT}, actual_height={teacher.base_height:.3f})")

    # ---- Phase 2: Walk forward 3 s (vx=0.5) ---
    print("\n--- Phase 2: Walk forward (vx=0.5, 3 s) ---")
    teacher.reset()
    x0 = teacher.base_pos[0]
    walk_ok = True
    for i in range(150):  # 150 * 0.02 s = 3 s
        ts = time.perf_counter()
        teacher.step(vel_cmd=(0.5, 0.0, 0.0))
        step_times.append((time.perf_counter() - ts) * 1000)
        h = teacher.base_height
        frames.append(render_frame(teacher, tp_cam))
        if i % 37 == 0:
            print(f"  t={teacher.sim_time:.2f}s  pos={teacher.base_pos}  yaw={teacher.base_yaw:.3f} rad  h={h:.3f}")
        if h < MIN_HEIGHT:
            walk_ok = False
            print(f"  FALL at t={teacher.sim_time:.2f}s, height={h:.3f}")
            break

    dx = teacher.base_pos[0] - x0
    walk_ok = walk_ok and (dx > 0.2)
    p2 = "PASS" if walk_ok else "FAIL"
    print(f"Phase 2: {p2}  (dx={dx:.3f} m, required >0.2 m, height={teacher.base_height:.3f})")

    # ---- Phase 3: Turn in place (wz=0.8, 2 s) ---
    print("\n--- Phase 3: Turn in place (wz=0.8, 2 s) ---")
    teacher.reset()
    yaw0 = teacher.base_yaw
    turn_ok = True
    for i in range(100):  # 100 * 0.02 s = 2 s
        ts = time.perf_counter()
        teacher.step(vel_cmd=(0.0, 0.0, 0.8))
        step_times.append((time.perf_counter() - ts) * 1000)
        h = teacher.base_height
        frames.append(render_frame(teacher, tp_cam))
        if i % 25 == 0:
            print(f"  t={teacher.sim_time:.2f}s  pos={teacher.base_pos}  yaw={teacher.base_yaw:.3f} rad  h={h:.3f}")
        if h < MIN_HEIGHT:
            turn_ok = False
            print(f"  FALL at t={teacher.sim_time:.2f}s, height={h:.3f}")
            break

    dyaw = abs(teacher.base_yaw - yaw0)
    # unwrap in case it crosses ±pi
    if dyaw > math.pi:
        dyaw = 2 * math.pi - dyaw
    turn_ok = turn_ok and (dyaw > 0.1)
    p3 = "PASS" if turn_ok else "FAIL"
    print(f"Phase 3: {p3}  (dyaw={dyaw:.3f} rad, required >0.1 rad, height={teacher.base_height:.3f})")

    # ---- Timing ---
    ms_mean = float(np.mean(step_times))
    ms_p95 = float(np.percentile(step_times, 95))
    print(f"\n--- Timing ---")
    print(f"  ms/step (mean): {ms_mean:.2f} ms")
    print(f"  ms/step (p95):  {ms_p95:.2f} ms")
    print(f"  Control Hz: 50 (20 ms budget)")
    print(f"  ONNX device: {teacher.device_str}")

    # ---- Save mp4 ---
    video_path = "eval/teacher_smoke/teacher_smoke.mp4"
    imageio.mimwrite(video_path, frames, fps=50, macro_block_size=1)
    print(f"\nVideo saved: {video_path}  ({len(frames)} frames)")

    # ---- Summary ---
    all_pass = stand_ok and walk_ok and turn_ok
    print("\n" + "=" * 60)
    print(f"Phase 1 (stand 2s):         {p1}")
    print(f"Phase 2 (walk forward 3s):  {p2}")
    print(f"Phase 3 (turn in place 2s): {p3}")
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    print(f"ONNX: {teacher.device_str}")
    print(f"ms/step (mean): {ms_mean:.2f}")
    print("=" * 60)

    # Write results to a quick log for the teacher.md doc
    results = {
        "phase1_stand": p1,
        "phase2_walk": p2,
        "phase3_turn": p3,
        "overall": "PASS" if all_pass else "FAIL",
        "ms_step_mean": round(ms_mean, 2),
        "ms_step_p95": round(ms_p95, 2),
        "onnx_device": teacher.device_str,
        "height_final_phase1": round(float(teacher.base_height), 3) if not stand_ok else "ok",
        "dx_phase2": round(float(dx), 3),
        "dyaw_phase3": round(float(dyaw), 3),
    }
    import json
    with open("eval/teacher_smoke/smoke_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results written to docs/smoke_results.json")

    sys.exit(0 if all_pass else 1)
