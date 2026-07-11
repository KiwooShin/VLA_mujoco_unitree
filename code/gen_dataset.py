"""
gen_dataset.py — Deterministic dataset generator for G1Nav.

CLI
---
MUJOCO_GL=egl python code/gen_dataset.py \\
    --difficulty easy \\
    --seed 0 \\
    --num-episodes 20 \\
    [--noise 0.05] \\
    --out /tmp/g1nav_easy

Output layout (LeRobot V2-ish)
-------------------------------
<out>/
  videos/                      (ego mp4 per episode  + side-by-side tp mp4)
  data/chunk-000/
    episode_NNNNNN.parquet
  meta/
    modality.json
    episodes.jsonl
    tasks.jsonl
    stats.json
    manifest.jsonl

Parquet schema per row (one control step):
  frame_index       int
  episode_index     int
  index             int   (global)
  task_index        int
  timestamp         float (s)
  proprio           list[float] (55-d, see below)
  action            list[float] (15)
  goal              list[float] (3)   [dist, cosθ, sinθ]
  vel_cmd           list[float] (3)   [vx, vy, ωz]
  done              int
  task_description  str

proprio (55-d):
  [0:15]   lower-body joint positions
  [15:30]  lower-body joint velocities
  [30:34]  base IMU quaternion [w,x,y,z]
  [34:37]  base angular velocity (rad/s)
  [37:40]  base linear acceleration (m/s²)  ← approx from qvel
  [40:55]  prev_action (15 lower-body joint targets)

Videos: ego_rgb mp4 (per-episode, 50 fps) + side-by-side (ego|tp).
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
# GPU-rendering fix (2026-07-11): steer glvnd to the NVIDIA EGL ICD when
# present, BEFORE mujoco initializes EGL — otherwise Mesa can win the vendor
# race and MuJoCo silently renders on llvmpipe (CPU) at ~400 ms/frame vs
# ~1.3 ms on the GPU. Idempotent; no-op when the ICD file is absent or the
# user already chose a vendor. See code/arena.py for the measured numbers.
import os as _os
_NVIDIA_EGL_ICD = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if _os.path.exists(_NVIDIA_EGL_ICD):
    _os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NVIDIA_EGL_ICD)
import mujoco
import numpy as np
import pandas as pd

# Local modules
_HERE: str      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT: str = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from code.arena import ArenaRenderer, EGO_H, EGO_W, build_arena
from code.scene import derive_rng, sample_scene
from code.steer import goal_vec, steer as steer_cmd
from code.teacher import CONTROL_DECIMATION, DEFAULT_ANGLES, SIM_DT, WBCTeacher, _yaw_of

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FPS: int          = int(round(1.0 / (SIM_DT * CONTROL_DECIMATION)))  # 50
SETTLE_STEPS: int = 80       # steps to settle before logging begins (zero vel cmd)
FALL_HEIGHT: float = 0.50     # pelvis z below this → fallen → discard
HOLD_STEPS: int   = 5        # consecutive steps within stop_r → success (short: 5 steps = 0.1s)
MAXSTEPS: dict[str, int] = {"easy": 600, "demo": 1400}  # HARD per-episode step cap (never exceeded)

PROPRIO_DIM: int  = 55       # full proprio vector length

# ---------------------------------------------------------------------------
# Proprio builder
# ---------------------------------------------------------------------------
def build_proprio(data: mujoco.MjData, prev_action: np.ndarray) -> np.ndarray:
    """Builds the 55-d proprio vector from MuJoCo data.

    Layout:
      [0:15]   lower-body joint positions
      [15:30]  lower-body joint velocities
      [30:34]  base IMU quaternion [w,x,y,z]
      [34:37]  base angular velocity (rad/s)
      [37:40]  base linear velocity (proxy for accel) [vx,vy,vz] in world frame
      [40:55]  prev_action (15 joint targets)

    Args:
        data: MuJoCo data holding the current physics state.
        prev_action: Previous joint-target action (15-d), appended as part
            of the observation.

    Returns:
        A (55,) float32 array with the layout described above.
    """
    q_lb   = data.qpos[7:22].copy()      # 15 lower-body joint positions
    dq_lb  = data.qvel[6:21].copy()      # 15 lower-body joint velocities
    quat   = data.qpos[3:7].copy()       # [w, x, y, z]
    ang_v  = data.qvel[3:6].copy()       # angular velocity
    lin_v  = data.qvel[0:3].copy()       # linear velocity (proxy for accel)
    return np.concatenate([
        q_lb.astype(np.float32),
        dq_lb.astype(np.float32),
        quat.astype(np.float32),
        ang_v.astype(np.float32),
        lin_v.astype(np.float32),
        prev_action.astype(np.float32),
    ])  # shape (55,)


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------
def run_episode(
    teacher: WBCTeacher,
    scene_cfg: dict,
    episode_idx: int,
    global_frame_offset: int,
    noise_sigma: float = 0.0,
    render_tp: bool = False,
    render_depth: bool = True,
    verbose: bool = False,
) -> dict | None:
    """Drives the WBC teacher to the target object and collects data.

    Speed notes:
      - render_tp=False (default): skips third-person rendering (~530ms/step
        saved). Set True only for the demo side-by-side video.
      - render_depth=True (default): renders depth for grounding. Set False
        to further speed up (pure-RGB collection).

    Args:
        teacher: WBCTeacher instance to step; its model/data are replaced
            with the arena built from `scene_cfg`.
        scene_cfg: Scene configuration dict from `sample_scene`.
        episode_idx: Episode index recorded in the `episode_index` column.
        global_frame_offset: Running frame count offset added to the
            per-frame `index` column.
        noise_sigma: DART-style exec-noise std (rad) added to the velocity
            command. 0.0 disables noise.
        render_tp: If True, also renders third-person frames.
        render_depth: If True, also renders depth for grounding.
        verbose: If True, prints periodic progress lines every 50 steps.

    Returns:
        A dict with keys:
            rows: list of row dicts (one per control step).
            ego_rgb_seq: list of np.uint8 (H,W,3) ego RGB frames.
            ego_dep_seq: list of np.float32 (H,W) depth frames, or an empty
                list if `render_depth` is False.
            tp_rgb_seq: list of np.uint8 (H,W,3) third-person frames, or an
                empty list if `render_tp` is False.
            reached: bool, whether the episode reached the goal.
            n_steps: int, number of logged steps.
        Returns None if the episode fell (discard) or exceeded the
        MAXSTEPS cap (discard).
    """
    difficulty   = scene_cfg.get("difficulty", "easy")
    hard_maxsteps = MAXSTEPS.get(difficulty, 600)

    model = build_arena(scene_cfg)
    data  = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    renderer = ArenaRenderer(model, EGO_W, EGO_H)
    tp_cam   = renderer.make_tp_cam() if render_tp else None

    # Inject the compiled model into the teacher (replace model/data in-place)
    teacher.model = model
    teacher.data  = data
    teacher._nj   = model.nq - 7
    teacher._pelvis_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )

    # Reset to scene's robot start pose
    rx, ry   = scene_cfg["robot_xy"]
    ryaw     = scene_cfg["robot_yaw"]
    teacher.reset(pos_xy=(rx, ry), yaw=ryaw)

    stop_r  = scene_cfg["stop_r"]
    horizon = min(scene_cfg["horizon"], hard_maxsteps)  # HARD CAP — never infinite
    tgt     = scene_cfg["objects"][scene_cfg["target_index"]]
    txy     = (tgt["x"], tgt["y"])
    task_description = scene_cfg["instruction"]

    prev_action = DEFAULT_ANGLES.copy()

    ego_rgb_seq: list = []
    ego_dep_seq: list = []
    tp_rgb_seq: list  = []
    rows: list        = []

    reached     = False
    hold_count  = 0
    fallen      = False

    # ---- Settle (zero vel cmd, not logged) ----
    for settle_i in range(SETTLE_STEPS):
        teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        if teacher.base_height < FALL_HEIGHT:
            renderer.close()
            return None   # fell during settle → discard

    # ---- Main rollout ----
    step_t0 = time.time()
    for t in range(horizon):
        bpos = teacher.base_pos         # [x, y, z]
        bxy  = bpos[:2]
        byaw = teacher.base_yaw

        # Steering
        vel, dist, yaw_err = steer_cmd(bxy, byaw, txy, stop_r)

        # Teacher step (DART: perturb execution, log clean target)
        if noise_sigma > 0.0:
            # Perturb the vel_cmd slightly to induce state diversity (DART-style)
            noisy_vel = vel + noise_sigma * np.random.randn(3).astype(np.float32)
            clean_targets = teacher.step(vel_cmd=tuple(noisy_vel))
        else:
            clean_targets = teacher.step(vel_cmd=tuple(vel))

        # Check fall
        if teacher.base_height < FALL_HEIGHT:
            fallen = True
            break

        # Render ego RGB (always)
        rgb, depth, intr = renderer.render_ego(data, byaw,
                                               render_depth=render_depth)

        # Render third-person (only if requested)
        tp_rgb = None
        if render_tp and tp_cam is not None:
            renderer.update_tp_cam(tp_cam, data)
            tp_rgb = renderer.render_tp(data, tp_cam)

        # Build proprio
        proprio = build_proprio(data, prev_action)

        # Goal vector
        gv = goal_vec(dist, yaw_err)

        # Log row
        global_idx = global_frame_offset + len(rows)
        done_flag  = int(dist < stop_r)

        rows.append({
            "frame_index":      t,
            "episode_index":    episode_idx,
            "index":            global_idx,
            "task_index":       0,  # filled by caller
            "timestamp":        float(t) / FPS,
            "proprio":          proprio.tolist(),
            "action":           clean_targets.tolist(),
            "goal":             gv.tolist(),
            "vel_cmd":          vel.tolist(),
            "done":             done_flag,
            "task_description": task_description,
        })

        ego_rgb_seq.append(rgb)
        if render_depth:
            ego_dep_seq.append(depth)
        if render_tp and tp_rgb is not None:
            tp_rgb_seq.append(tp_rgb)

        prev_action = clean_targets.copy()

        # Verbose per-step progress
        if verbose and (t % 50 == 0 or dist < stop_r):
            elapsed = time.time() - step_t0
            sps = (t + 1) / max(elapsed, 0.001)
            print(f"    [ep{episode_idx}] t={t:4d}/{horizon}  dist={dist:.2f}m  "
                  f"h={teacher.base_height:.3f}  reached={reached}  "
                  f"{sps:.1f}stp/s", flush=True)

        # Termination: reached goal
        if dist < stop_r:
            reached    = True
            hold_count += 1
        else:
            # Don't reset hold_count — allow non-consecutive approach
            pass
        if reached and hold_count >= HOLD_STEPS:
            break

    renderer.close()

    if fallen:
        return None

    return {
        "rows":        rows,
        "ego_rgb_seq": ego_rgb_seq,
        "ego_dep_seq": ego_dep_seq,
        "tp_rgb_seq":  tp_rgb_seq,
        "reached":     reached,
        "n_steps":     len(rows),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Parses CLI arguments and runs deterministic dataset generation."""
    ap = argparse.ArgumentParser(description="G1Nav deterministic dataset generator")
    ap.add_argument("--difficulty",    choices=["easy", "demo"], required=True)
    ap.add_argument("--seed",          type=int, required=True)
    ap.add_argument("--num-episodes",  type=int, required=True)
    ap.add_argument("--noise",         type=float, default=0.0,
                    help="DART exec-noise std (rad) on vel_cmd. 0=off.")
    ap.add_argument("--out",           required=True)
    ap.add_argument("--render-tp",     action="store_true",
                    help="Render third-person video (slow, ~530ms/frame extra). "
                         "Default: ego only. Set for demo SBS video.")
    ap.add_argument("--no-depth",      action="store_true",
                    help="Skip depth rendering (faster, but grounding validation "
                         "will use GT distance). Default: render depth.")
    ap.add_argument("--verbose",       action="store_true",
                    help="Print per-step progress (useful for debugging).")
    args = ap.parse_args()

    render_tp    = args.render_tp
    render_depth = not args.no_depth

    t0 = time.time()

    # Speed estimate
    import sys as _sys
    est_sps = 5.5 if not render_tp else 1.3
    difficulty = args.difficulty
    horizon    = MAXSTEPS[difficulty]
    est_total  = args.num_episodes * horizon / est_sps
    print(f"[gen_dataset] difficulty={difficulty}  episodes={args.num_episodes}  "
          f"render_tp={render_tp}  render_depth={render_depth}")
    print(f"[gen_dataset] Estimated time: ~{est_total/60:.0f} min "
          f"({est_sps:.1f} stp/s, {horizon} steps/ep max)")
    print(f"[gen_dataset] MAXSTEPS cap: {horizon}  HOLD_STEPS: {HOLD_STEPS}", flush=True)

    # ---- Output directories ----
    out_data    = os.path.join(args.out, "data", "chunk-000")
    out_videos  = os.path.join(args.out, "videos")
    out_meta    = os.path.join(args.out, "meta")
    for d in [out_data, out_videos, out_meta]:
        os.makedirs(d, exist_ok=True)

    # ---- Create teacher (singleton — model replaced per episode) ----
    print("[gen_dataset] Loading WBCTeacher...", flush=True)
    teacher = WBCTeacher()
    print("[gen_dataset] Teacher loaded.", flush=True)

    tasks_map: dict[str, int] = {}
    episodes_meta: list       = []
    all_proprio: list         = []
    all_actions: list         = []
    global_frame_idx          = 0
    n_success                 = 0
    n_attempted               = 0
    n_fallen                  = 0

    ep_written = 0  # number of written episodes

    import cv2 as _cv2

    for ep_i in range(args.num_episodes):
        rng       = derive_rng(args.seed, ep_i)
        scene_cfg = sample_scene(rng, args.difficulty)
        instr     = scene_cfg["instruction"]
        tgt_obj   = scene_cfg["objects"][scene_cfg["target_index"]]
        print(f"\n[ep {ep_i:03d}/{args.num_episodes}]  '{instr}'  "
              f"dist={tgt_obj['dist_from_robot']:.2f}m", flush=True)

        if instr not in tasks_map:
            tasks_map[instr] = len(tasks_map)
        task_idx = tasks_map[instr]

        ep_t0 = time.time()
        n_attempted += 1
        result = run_episode(
            teacher             = teacher,
            scene_cfg           = scene_cfg,
            episode_idx         = ep_written,
            global_frame_offset = global_frame_idx,
            noise_sigma         = args.noise,
            render_tp           = render_tp,
            render_depth        = render_depth,
            verbose             = args.verbose,
        )
        ep_elapsed = time.time() - ep_t0

        if result is None:
            n_fallen += 1
            print(f"  ep {ep_i:03d}  FALLEN/MAXSTEPS (discarded)  t={ep_elapsed:.0f}s",
                  flush=True)
            continue

        rows        = result["rows"]
        ego_rgb_seq = result["ego_rgb_seq"]
        ego_dep_seq = result["ego_dep_seq"]
        tp_rgb_seq  = result["tp_rgb_seq"]
        reached     = result["reached"]

        # Update task_index in rows
        for r in rows:
            r["task_index"] = task_idx

        # ---- Write parquet ----
        ep_name = f"episode_{ep_written:06d}"
        df      = pd.DataFrame(rows)
        df.to_parquet(os.path.join(out_data, f"{ep_name}.parquet"), index=False)

        # ---- Write ego RGB mp4 ----
        ego_path = os.path.join(out_videos, f"{ep_name}_ego.mp4")
        imageio.mimwrite(ego_path, ego_rgb_seq, fps=FPS, macro_block_size=1)

        # ---- Write side-by-side (ego | tp) mp4 (only if tp was rendered) ----
        if render_tp and len(tp_rgb_seq) > 0:
            sbs_frames = []
            for eg, tp in zip(ego_rgb_seq, tp_rgb_seq):
                tp_r = _cv2.resize(tp, (EGO_W, EGO_H))
                sbs  = np.concatenate([eg, tp_r], axis=1)
                sbs_frames.append(sbs)
            sbs_path = os.path.join(out_videos, f"{ep_name}_sbs.mp4")
            imageio.mimwrite(sbs_path, sbs_frames, fps=FPS, macro_block_size=1)

        tgt_meta   = scene_cfg["objects"][scene_cfg["target_index"]]
        final_dist = float(rows[-1]["goal"][0]) if rows else 99.0

        episodes_meta.append({
            "episode_index":    ep_written,
            "task_index":       task_idx,
            "length":           len(rows),
            "success":          bool(reached),
            "final_goal_dist":  round(final_dist, 3),
            "difficulty":       args.difficulty,
            "seed":             args.seed,
            "ep_seed_index":    ep_i,
            "target_color":     tgt_meta["color_name"],
            "target_shape":     tgt_meta["shape_name"],
            "tasks":            [instr],
        })

        all_proprio.extend([r["proprio"] for r in rows])
        all_actions.extend([r["action"]  for r in rows])
        global_frame_idx += len(rows)

        if reached:
            n_success += 1

        elapsed = time.time() - t0
        ep_sps = len(rows) / max(ep_elapsed, 0.001)
        succ_rate_so_far = n_success / max(1, ep_written + 1)
        print(f"  ep {ep_i:03d} -> {ep_name}  steps={len(rows):4d}  "
              f"reached={reached}  dist_final={final_dist:.2f}m  "
              f"succ_so_far={succ_rate_so_far:.2f}  "
              f"ep_t={ep_elapsed:.0f}s ({ep_sps:.1f}stp/s)  total_t={elapsed:.0f}s",
              flush=True)

        ep_written += 1

    # ---- Stats ----
    arr_p = np.array(all_proprio, dtype=np.float32)
    arr_a = np.array(all_actions, dtype=np.float32)

    def _stat(a: np.ndarray) -> dict[str, list[float]]:
        """Computes per-dimension mean/std/min/max for stats.json."""
        return {
            "mean": a.mean(0).tolist(),
            "std":  (a.std(0) + 1e-6).tolist(),
            "min":  a.min(0).tolist(),
            "max":  a.max(0).tolist(),
        }

    stats = {
        "proprio": _stat(arr_p) if len(arr_p) > 0 else {},
        "action":  _stat(arr_a) if len(arr_a) > 0 else {},
    }

    # ---- Write meta files ----
    tasks_meta = [
        {"task_index": v, "task": k}
        for k, v in sorted(tasks_map.items(), key=lambda x: x[1])
    ]

    n_total_frames   = global_frame_idx
    success_rate     = n_success / max(1, ep_written)

    modality = {
        "proprio": {
            "joint_pos":    {"start": 0,  "end": 15},
            "joint_vel":    {"start": 15, "end": 30},
            "base_quat":    {"start": 30, "end": 34},
            "base_angvel":  {"start": 34, "end": 37},
            "base_linacc":  {"start": 37, "end": 40},
            "prev_action":  {"start": 40, "end": 55},
        },
        "action": {
            "lower_body": {"start": 0, "end": 15},
        },
        "goal": {
            "dist":   {"index": 0},
            "cos_th": {"index": 1},
            "sin_th": {"index": 2},
        },
        "vel_cmd": {
            "vx": {"index": 0},
            "vy": {"index": 1},
            "wz": {"index": 2},
        },
        "label": {
            "done": {},
        },
        "annotation": {
            "task_description": {},
        },
    }

    elapsed_total = time.time() - t0
    info = {
        "codebase_version": "v2.0",
        "fps":              FPS,
        "robot":            "unitree_g1_lowerbody",
        "difficulty":       args.difficulty,
        "seed":             args.seed,
        "noise_sigma":      args.noise,
        "proprio_dim":      PROPRIO_DIM,
        "action_dim":       15,
        "total_episodes":   ep_written,
        "total_frames":     n_total_frames,
        "n_fallen":         n_fallen,
        "n_attempted":      n_attempted,
        "success_rate":     round(success_rate, 3),
        "render_tp":        render_tp,
        "render_depth":     render_depth,
        "total_time_s":     round(elapsed_total, 1),
        "eps_per_min":      round(ep_written / max(elapsed_total / 60, 0.01), 2),
    }

    manifest = {
        "files": [
            f"data/chunk-000/episode_{i:06d}.parquet"
            for i in range(ep_written)
        ]
    }

    json.dump(modality, open(os.path.join(out_meta, "modality.json"), "w"), indent=2)
    json.dump(info,     open(os.path.join(out_meta, "info.json"),     "w"), indent=2)
    json.dump(stats,    open(os.path.join(out_meta, "stats.json"),    "w"), indent=2)

    with open(os.path.join(out_meta, "episodes.jsonl"), "w") as f:
        for em in episodes_meta:
            f.write(json.dumps(em) + "\n")

    with open(os.path.join(out_meta, "tasks.jsonl"), "w") as f:
        for tm in tasks_meta:
            f.write(json.dumps(tm) + "\n")

    with open(os.path.join(out_meta, "manifest.jsonl"), "w") as f:
        for fp in manifest["files"]:
            f.write(json.dumps({"path": fp}) + "\n")

    print(f"\n{'='*60}")
    print(f"DONE: {ep_written} episodes, {n_total_frames} frames")
    print(f"  success_rate: {success_rate:.3f}  ({n_success}/{ep_written})")
    print(f"  fallen (discarded): {n_fallen}")
    print(f"  total time: {elapsed_total:.0f}s")
    print(f"  eps/min: {info['eps_per_min']}")
    print(f"  output: {args.out}/")
    print(f"{'='*60}", flush=True)


# ---------------------------------------------------------------------------
# Determinism check helper (importable by test scripts)
# ---------------------------------------------------------------------------
def check_determinism(difficulty: str, seed: int, n_check: int = 3) -> bool:
    """Runs two independent passes and verifies identical parquet outputs.

    Args:
        difficulty: Scene difficulty, forwarded to `sample_scene`.
        seed: Base seed forwarded to `derive_rng`.
        n_check: Number of episodes to check per pass.

    Returns:
        True if deterministic (identical actions across both passes),
        False otherwise.
    """
    import shutil
    import tempfile

    results = []
    for run in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            teacher = WBCTeacher()
            rows_all = []
            for ep_i in range(n_check):
                rng       = derive_rng(seed, ep_i)
                scene_cfg = sample_scene(rng, difficulty)
                result    = run_episode(
                    teacher             = teacher,
                    scene_cfg           = scene_cfg,
                    episode_idx         = ep_i,
                    global_frame_offset = 0,
                    noise_sigma         = 0.0,
                    render_tp           = False,  # skip tp for speed
                    render_depth        = False,  # skip depth for speed
                )
                if result is not None:
                    rows_all.append(result["rows"])
            results.append(rows_all)

    # Compare
    if len(results[0]) != len(results[1]):
        print(f"[determinism] FAIL: different episode counts {len(results[0])} vs {len(results[1])}")
        return False
    for ep_i, (r0, r1) in enumerate(zip(results[0], results[1])):
        if len(r0) != len(r1):
            print(f"[determinism] FAIL ep{ep_i}: different step counts {len(r0)} vs {len(r1)}")
            return False
        for step_i, (s0, s1) in enumerate(zip(r0, r1)):
            a0 = np.array(s0["action"])
            a1 = np.array(s1["action"])
            if not np.allclose(a0, a1, atol=1e-5):
                print(f"[determinism] FAIL ep{ep_i} step{step_i}: action mismatch")
                return False
    print(f"[determinism] PASS: {n_check} episodes identical across 2 runs")
    return True


if __name__ == "__main__":
    main()
