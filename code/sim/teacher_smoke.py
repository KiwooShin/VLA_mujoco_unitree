"""
teacher_smoke.py — WBCTeacher CLI smoke test (RF-1 split of code/teacher.py's
``__main__`` block).

Runs three phases (stand / walk-forward / turn-in-place) against a live
``WBCTeacher``, renders a third-person mp4, and writes
``eval/teacher_smoke/smoke_results.json``. Invoke via the compat entry shim
at the old path (``python code/teacher.py``) or directly:

    MUJOCO_GL=egl python -m code.sim.teacher_smoke
"""

import json
import math
import os
import sys
import time

import imageio.v2 as imageio
import mujoco
import numpy as np

from code.sim.teacher import NUM_OBS, SINGLE_OBS_DIM, WBCTeacher


def render_frame(teacher: WBCTeacher, tp_cam: mujoco.MjvCamera, renderer: mujoco.Renderer) -> np.ndarray:
    """Render a third-person frame tracking the robot.

    Args:
        teacher: Active WBCTeacher instance being tracked.
        tp_cam: Third-person MjvCamera to position on the robot.
        renderer: Renderer bound to teacher.model, used to draw the frame.

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


def main() -> int:
    """Run the 3-phase WBCTeacher smoke test. Returns the process exit code."""
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
        frames.append(render_frame(teacher, tp_cam, renderer))
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
        frames.append(render_frame(teacher, tp_cam, renderer))
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
        frames.append(render_frame(teacher, tp_cam, renderer))
        if i % 25 == 0:
            print(f"  t={teacher.sim_time:.2f}s  pos={teacher.base_pos}  yaw={teacher.base_yaw:.3f} rad  h={h:.3f}")
        if h < MIN_HEIGHT:
            turn_ok = False
            print(f"  FALL at t={teacher.sim_time:.2f}s, height={h:.3f}")
            break

    dyaw = abs(teacher.base_yaw - yaw0)
    # unwrap in case it crosses +/-pi
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
    with open("eval/teacher_smoke/smoke_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results written to docs/smoke_results.json")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
