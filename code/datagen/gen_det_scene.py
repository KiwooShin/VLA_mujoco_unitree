"""
code/datagen/gen_det_scene.py — Scene construction + one-scene rollout capture.

Role: split out of gen_det_dataset.py (RF-1) — dispatches to the right scene
sampler (easy/demo/search) and runs one scene end-to-end: builds the arena,
then captures trajectory + teleport frames. See gen_det_dataset.py's module
docstring for the two frame families (TRAJECTORY, TELEPORT).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from code.arena import ArenaRenderer, build_arena
from code.datagen.gen_det_capture import SegRenderer, capture_frame
from code.datagen.gen_det_common import (
    DUAL_RENDER_PROB, FALL_HEIGHT, MAX_GAIT_SNAPSHOTS, MAXSTEPS_TRAJ,
    N_TELEPORT_FOCUS, N_TELEPORT_RANDOM, N_TRAJ_TARGET, SETTLE_STEPS, pick_cam,
)
from code.datagen.gen_det_labels import build_id_to_obj
from code.eval_search import sample_search_scene
from code.scene import derive_rng, sample_scene
from code.steer import steer as steer_cmd
from code.teacher import WBCTeacher, _yaw_of


# ---------------------------------------------------------------------------
# Scene construction (unifies scene.py 'easy'/'demo' + eval_search.py 'search')
# ---------------------------------------------------------------------------
def make_scene_cfg(rng: np.random.Generator, style: str, ep_i: int) -> dict:
    """Builds a scene config for the given style, dispatching to the right sampler.

    Args:
        rng: Random generator used for sampling.
        style: "easy" or "demo" (routed to `code.scene.sample_scene`), or
            "search" (routed to `code.eval_search.sample_search_scene`).
        ep_i: Episode index, forwarded to the search-style sampler.

    Returns:
        The sampled scene configuration dict.
    """
    if style == "search":
        sc = sample_search_scene(rng, ep_i)
    else:
        sc = sample_scene(rng, style)
    return sc


# ---------------------------------------------------------------------------
# One scene -> list of frame records
# ---------------------------------------------------------------------------
def run_scene(scene_id: int, style: str, base_seed: int, ep_i: int,
             rng_sample: np.random.Generator) -> tuple[dict, list]:
    """Runs one scene: builds the arena, then captures trajectory + teleport frames.

    Builds one arena for the scene and captures three families of frames:
    (1) a walking trajectory toward the scene's nominal target, subsampled
    along the approach; (2) teleported "focus" frames at controlled
    (distance, bearing) offsets from a random object; and (3) fully random
    "confusion" teleport poses. See the module docstring for details.

    Args:
        scene_id: Scene index, used to derive the scene's RNG seed.
        style: Scene family: "easy", "demo", or "search".
        base_seed: Base seed forwarded to `derive_rng`.
        ep_i: Episode index forwarded to the search-style sampler.
        rng_sample: Random generator used for all in-scene sampling
            (teleport poses, dual-camera decisions, etc.).

    Returns:
        A tuple (scene_cfg, frame_records). `frame_records` is empty if
        the robot fell during the initial settle.
    """
    rng = derive_rng(base_seed, scene_id)
    scene_cfg = make_scene_cfg(rng, style, ep_i)
    objects = scene_cfg["objects"]
    n_obj = len(objects)
    target_idx = scene_cfg["target_index"]
    target_obj = objects[target_idx]
    target_xy = np.array([target_obj["x"], target_obj["y"]], dtype=np.float64)

    arena_model = build_arena(scene_cfg)
    arena_model.opt.timestep = 0.005
    id_to_obj = build_id_to_obj(arena_model, n_obj)

    teacher = WBCTeacher(use_gpu=False)
    teacher.model = arena_model
    teacher.data = mujoco.MjData(arena_model)
    teacher._nj = arena_model.nq - 7
    teacher._pelvis_id = mujoco.mj_name2id(arena_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    rx0, ry0 = scene_cfg["robot_xy"]
    robot_yaw0 = float(scene_cfg.get("robot_yaw", 0.0))
    teacher.reset(pos_xy=(rx0, ry0), yaw=robot_yaw0)

    data_mj = teacher.data
    for _ in range(SETTLE_STEPS):
        teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        if teacher.base_height < FALL_HEIGHT:
            return scene_cfg, []   # fell during settle — skip scene

    renderer = ArenaRenderer(arena_model)
    seg_rend = SegRenderer(arena_model)

    frame_records = []
    gait_snapshots = [data_mj.qpos.copy()]   # snapshot 0: settled standing pose

    def _dual_maybe(cam_type: str, rgb_cam_dist: float) -> list[str]:
        """Returns the camera list for this frame, occasionally rendering both."""
        cams = [cam_type]
        other = "proximity" if cam_type == "grounding" else "grounding"
        if rng_sample.random() < DUAL_RENDER_PROB:
            cams.append(other)
        return cams

    # ---- 1) TRAJECTORY: walk toward the scene's nominal target ----
    maxsteps = MAXSTEPS_TRAJ.get(style, 600)
    sample_every = max(3, maxsteps // N_TRAJ_TARGET)
    step = 0
    last_sample = -999
    fell = False
    while step < maxsteps:
        robot_xy = data_mj.qpos[0:2].copy()
        robot_yaw = _yaw_of(data_mj.qpos[3:7])
        dist_to_target = float(np.linalg.norm(robot_xy - target_xy))

        if step - last_sample >= sample_every:
            yaw_now = robot_yaw
            for cam_type in _dual_maybe(pick_cam(dist_to_target), dist_to_target):
                rec = capture_frame(renderer, seg_rend, data_mj, yaw_now, cam_type,
                                    objects, id_to_obj)
                rec["source"] = "trajectory"
                frame_records.append(rec)
            last_sample = step
            if len(gait_snapshots) < MAX_GAIT_SNAPSHOTS and step > 0:
                gait_snapshots.append(data_mj.qpos.copy())
            if dist_to_target < 0.30:
                break

        vel_cmd, _, _ = steer_cmd(robot_xy, robot_yaw, target_xy, stop_r=0.22)
        teacher.step(vel_cmd=tuple(float(v) for v in vel_cmd))
        if teacher.base_height < FALL_HEIGHT:
            fell = True
            break
        step += 1

    # ---- 2) TELEPORT: focus (controlled dist/bearing to a random object) ----
    arena_half = float(scene_cfg["arena_size"])
    margin = 0.35
    log_lo, log_hi = math.log(0.3), math.log(10.0)

    def _teleport_pose(
        focus_xy: tuple[float, float], dist: float, bearing_off_deg: float,
    ) -> tuple[float, float, float]:
        """Computes a teleport (x, y, yaw) at `dist`/`bearing_off_deg` from `focus_xy`."""
        approach_ang = float(rng_sample.uniform(-math.pi, math.pi))
        rx = focus_xy[0] + dist * math.cos(approach_ang)
        ry = focus_xy[1] + dist * math.sin(approach_ang)
        bearing_to_focus = approach_ang + math.pi
        yaw = bearing_to_focus - math.radians(bearing_off_deg)
        return rx, ry, yaw

    def _in_bounds(rx: float, ry: float) -> bool:
        """Returns True if (rx, ry) is within the arena bounds minus margin."""
        return abs(rx) < arena_half - margin and abs(ry) < arena_half - margin

    def _apply_pose(rx: float, ry: float, yaw: float) -> None:
        """Teleports the robot to (rx, ry, yaw) using a cached gait qpos snapshot."""
        snap = gait_snapshots[int(rng_sample.integers(len(gait_snapshots)))]
        data_mj.qpos[:] = snap
        data_mj.qpos[0] = rx
        data_mj.qpos[1] = ry
        data_mj.qpos[3] = math.cos(yaw / 2.0)
        data_mj.qpos[4] = 0.0
        data_mj.qpos[5] = 0.0
        data_mj.qpos[6] = math.sin(yaw / 2.0)
        mujoco.mj_forward(arena_model, data_mj)

    n_focus_ok = 0
    for _ in range(N_TELEPORT_FOCUS * 3):
        if n_focus_ok >= N_TELEPORT_FOCUS:
            break
        oi = int(rng_sample.integers(n_obj))
        focus = objects[oi]
        focus_xy = (focus["x"], focus["y"])
        dist = math.exp(rng_sample.uniform(log_lo, log_hi))
        wide = rng_sample.random() < 0.25
        bearing_off = float(rng_sample.uniform(-80, 80) if wide else rng_sample.uniform(-45, 45))
        rx, ry, yaw = _teleport_pose(focus_xy, dist, bearing_off)
        if not _in_bounds(rx, ry):
            continue
        _apply_pose(rx, ry, yaw)
        cam_type = pick_cam(dist)
        for ct in _dual_maybe(cam_type, dist):
            rec = capture_frame(renderer, seg_rend, data_mj, yaw, ct, objects, id_to_obj)
            rec["source"] = "teleport_focus"
            frame_records.append(rec)
        n_focus_ok += 1

    # ---- 3) TELEPORT: fully random confusion poses ----
    n_rand_ok = 0
    for _ in range(N_TELEPORT_RANDOM * 3):
        if n_rand_ok >= N_TELEPORT_RANDOM:
            break
        rx = float(rng_sample.uniform(-(arena_half - margin), arena_half - margin))
        ry = float(rng_sample.uniform(-(arena_half - margin), arena_half - margin))
        yaw = float(rng_sample.uniform(-math.pi, math.pi))
        if not _in_bounds(rx, ry):
            continue
        _apply_pose(rx, ry, yaw)
        cam_type = "proximity" if rng_sample.random() < 0.4 else "grounding"
        rec = capture_frame(renderer, seg_rend, data_mj, yaw, cam_type, objects, id_to_obj)
        rec["source"] = "teleport_random"
        frame_records.append(rec)
        n_rand_ok += 1

    renderer.close()
    seg_rend.close()
    return scene_cfg, frame_records
