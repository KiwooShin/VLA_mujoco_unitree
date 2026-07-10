"""
gen_det_dataset.py — NX-6: labeled dataset generator for a learned object detector
that replaces the classical HSV grounder (docs/nx5_coherence.md §CLOSURE).

MuJoCo gives PERFECT labels: we render RGB + depth + INSTANCE SEGMENTATION from the
SAME camera pose (GROUNDING cam, 26° pitch, 480x360; PROXIMITY cam, 58° pitch,
320x240 — the two cameras actually used at deploy, code/arena.py). Segmentation geom
IDs map directly to object instances (built in code/arena.py's build_arena()), so
every visible object's class/color/bbox/centroid/GT-(dist,bearing) is exact — no HSV
heuristics, no classical-grounder failure modes.

Two families of frames per scene (one arena build, many frames — amortizes the
~200-500ms MjSpec compile cost):
  1. TRAJECTORY: walk the robot toward the scene's nominal target via steer.py's
     privileged controller (WBCTeacher physics), subsampling frames along the
     approach. Gives natural standing + mid-gait joint poses and a natural far->near
     distance sweep. A handful of qpos snapshots are cached along the way (joint
     angles only) for reuse by teleport frames below ("mid-gait perturbations").
  2. TELEPORT: directly place the robot (teleport qpos, mj_forward, no physics) at
     controlled (distance, bearing) offsets from a randomly chosen object in the
     scene — log-uniform distance in [0.3, 10]m, bearing offset wide enough to
     produce partial/clipped and fully out-of-frame negatives — plus fully random
     (x, y, yaw) "confusion" poses that exercise multi-object / same-color-distractor
     scenes exactly as scene.py naturally samples them (same-hue-different-shape
     objects are common because only (color,shape) PAIRS are forced unique).
     Joint angles for teleport frames are drawn from the trajectory-phase snapshot
     cache (or the settled stand pose) so teleported frames still look like a
     standing/mid-gait robot rather than a T-pose.

Scene families (reusing the actual samplers, not reinvented):
  - code/scene.py sample_scene(..., 'easy')   — close range, 3 objects, target in FOV
  - code/scene.py sample_scene(..., 'demo')   — far range, 5-7 objects, target often
    out of FOV, the exact regime docs/fa1_failures.md's ep0/2/4/5/12 come from
  - code/eval_search.py sample_search_scene() — target forced OUTSIDE ±45° FOV,
    the exact regime the search skill's scan-then-approach depends on

Labels (per visible object per frame — from the segmentation mask, GT world pose):
  class (ball/cube/cylinder/cone), color, pixel bbox, pixel centroid, area,
  clipped (touches an image edge), depth_median_m (from GT mask, not HSV mask),
  GT egocentric (dist_gt_m, bearing_gt_deg) from the true robot/object world poses,
  PLUS a back-projected (dist_bp_m, bearing_bp_deg) computed the same way the
  deployed grounder does (arena.backproject_pixel -> grounding.cam_to_egocentric,
  with a nominal-radius correction since depth is measured to the object's near
  SURFACE, not its center) — the gap between these two is the label-geometry
  sanity check (§4 of the NX-6 brief).

Storage (compact, <4GB target):
  {split}/images_{cam}.npz   — rgb: (N,H,W,3) uint8, depth: (N,H,W) float16
  {split}/frames.parquet     — one row per frame (pose, qpos, scene_id, camera, source)
  {split}/labels.parquet     — one row per visible-object-in-frame detection
  scenes.json                — one row per scene (objects list, for exact replay)
  preview/*.png              — 12 random samples: RGB + mask overlay + bbox + text
  meta.json                  — dataset summary + label-geometry error stats

ANTI-HANG: --smoke runs 2 scenes end-to-end and reports throughput/estimate before
any full run. Progress printed + flushed every scene.

Usage
-----
  MUJOCO_GL=egl python code/gen_det_dataset.py --smoke
  MUJOCO_GL=egl python code/gen_det_dataset.py --n-easy 90 --n-demo 180 --n-search 80 \
      --seed 7001 --out dataset/det_v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import mujoco
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from code.arena import (
    build_arena, ArenaRenderer, backproject_pixel, _set_ego_cam,
    COLORS, SHAPES, GROUNDING_W, GROUNDING_H, GROUNDING_PITCH,
    PROXIMITY_W, PROXIMITY_H, PROXIMITY_PITCH,
)
from code.scene import sample_scene, derive_rng
from code.eval_search import sample_search_scene
from code.steer import steer as steer_cmd, egocentric_goal, _angle_diff
from code.teacher import WBCTeacher, _yaw_of
from code.grounding import get_ego_intrinsics_rendered, cam_to_egocentric

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLOR_NAMES = [c for c, _ in COLORS]                 # 7 colors
SHAPE_NAMES = [s for s, _ in SHAPES]                 # 4 shapes: ball,cube,cylinder,cone
SIZE_M      = dict(SHAPES)                           # nominal diameter/edge per shape
COLOR2I     = {c: i for i, c in enumerate(COLOR_NAMES)}
SHAPE2I     = {s: i for i, s in enumerate(SHAPE_NAMES)}

FALL_HEIGHT      = 0.50
SETTLE_STEPS     = 80
MIN_PIXELS       = 6          # minimum mask pixels to keep a detection
GEOM_RE          = re.compile(r"^obj_(\d+)(?:_tip)?$")

CAM_SWITCH_DIST_M = 1.8        # proximity below this true distance, else grounding
DUAL_RENDER_PROB   = 0.20      # occasionally render BOTH cams at the same pose

MAXSTEPS_TRAJ = {"easy": 260, "demo": 900, "search": 550}
N_TRAJ_TARGET = 12             # aim for ~this many trajectory samples per scene
N_TELEPORT_FOCUS  = 10
N_TELEPORT_RANDOM = 6
MAX_GAIT_SNAPSHOTS = 8

DIFF_FOR_SEARCH = "search"     # pseudo-difficulty label for search-style scenes


def _env_note():
    print(f"[gen_det_dataset] MUJOCO_GL={os.environ.get('MUJOCO_GL')}  "
          f"mujoco={mujoco.__version__}", flush=True)


# ---------------------------------------------------------------------------
# Geom-id -> object-index map (handles the cone's extra "_tip" geom)
# ---------------------------------------------------------------------------
def build_id_to_obj(model: mujoco.MjModel, n_objects: int) -> np.ndarray:
    id_to_obj = -np.ones(model.ngeom, dtype=np.int32)
    for gi in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
        if not name:
            continue
        m = GEOM_RE.match(name)
        if m:
            oi = int(m.group(1))
            if oi < n_objects:
                id_to_obj[gi] = oi
    return id_to_obj


def seg_to_objmap(seg: np.ndarray, id_to_obj: np.ndarray) -> np.ndarray:
    """seg: (H,W,2) int32 from enable_segmentation_rendering(); channel0=geom id."""
    inst = seg[..., 0]
    valid = inst >= 0
    idx = np.where(valid, inst, 0)
    idx = np.clip(idx, 0, id_to_obj.shape[0] - 1)
    obj_map = np.where(valid, id_to_obj[idx], -1)
    return obj_map


# ---------------------------------------------------------------------------
# Persistent segmentation renderers (avoid EGL context exhaustion — same
# pre-allocate-once pattern as ArenaRenderer's own renderers)
# ---------------------------------------------------------------------------
class SegRenderer:
    def __init__(self, model: mujoco.MjModel):
        self._gr_rend = mujoco.Renderer(model, GROUNDING_H, GROUNDING_W)
        self._pr_rend = mujoco.Renderer(model, PROXIMITY_H, PROXIMITY_W)
        self._gr_cam = mujoco.MjvCamera(); self._gr_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._pr_cam = mujoco.MjvCamera(); self._pr_cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def render(self, data: mujoco.MjData, yaw: float, cam_type: str) -> np.ndarray:
        if cam_type == "proximity":
            rend, cam, pitch = self._pr_rend, self._pr_cam, PROXIMITY_PITCH
        else:
            rend, cam, pitch = self._gr_rend, self._gr_cam, GROUNDING_PITCH
        _set_ego_cam(cam, data.qpos, yaw, pitch_deg=pitch)
        rend.update_scene(data, cam)
        rend.enable_segmentation_rendering()
        seg = rend.render().copy()
        rend.disable_segmentation_rendering()
        return seg

    def close(self):
        self._gr_rend.close()
        self._pr_rend.close()


# ---------------------------------------------------------------------------
# Per-object label derivation from a segmentation-derived object-index map.
# Standalone (no renderer dependency) so both gen_det_dataset.py's synthetic
# generator AND gen_det_failcases.py's live-replay instrumentation can share
# the exact same labeling logic.
# ---------------------------------------------------------------------------
def derive_object_labels(rgb: np.ndarray, depth: np.ndarray, obj_map: np.ndarray,
                         objects: list, robot_xy: np.ndarray, robot_yaw: float,
                         intr: dict) -> list:
    h_img, w_img = rgb.shape[0], rgb.shape[1]
    labels = []
    for oi, obj in enumerate(objects):
        mask = (obj_map == oi)
        area = int(mask.sum())
        if area < MIN_PIXELS:
            continue
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        cx_px, cy_px = float(xs.mean()), float(ys.mean())
        clipped = (x0 <= 0) or (x1 >= w_img - 1) or (y0 <= 0) or (y1 >= h_img - 1)

        dvals = depth[mask]
        dvals = dvals[np.isfinite(dvals) & (dvals > 0.0) & (dvals < 30.0)]
        depth_med = float(np.median(dvals)) if dvals.size > 0 else -1.0

        obj_xy = np.array([obj["x"], obj["y"]], dtype=np.float64)
        dist_gt, yaw_err_gt, _ = egocentric_goal(robot_xy, robot_yaw, obj_xy)
        bearing_gt_deg = math.degrees(yaw_err_gt)

        dist_bp, bearing_bp_deg, err_dist, err_bearing = (np.nan,) * 4
        if depth_med > 0:
            x_cam, y_cam, z_cam = backproject_pixel(cx_px, cy_px, depth_med, intr)
            radius = float(SIZE_M.get(obj["shape_name"], 0.24)) / 2.0
            # NOTE: always use the geometrically-CORRECT un-pitch formula here (both
            # cameras), not production's per-camera legacy toggle (docs/cam_p1.md:
            # production only applies the fix to the 58° proximity cam and knowingly
            # leaves the 26° grounding cam on the old formula so as not to shift the
            # distribution the deployed policy/EMA was tuned against — a DEPLOYMENT
            # concern, irrelevant here). This label-geometry check validates our OWN
            # backprojection pipeline (arena intrinsics + offsets) against analytic
            # GT, so it must use the actually-correct transform for both cameras.
            # .get(..., GROUNDING_PITCH) not intr["pitch_deg"]: eval_search.py's cam2
            # grounding-camera call site reuses a loop-invariant intrinsics dict that
            # never gets 'pitch_deg' merged in (a pre-existing quirk of that file, see
            # code/gen_det_failcases.py's instrumentation notes) even though the frame
            # was actually rendered at GROUNDING_PITCH — so the fallback here is the
            # physically-correct render pitch, not an arbitrary default.
            dist_bp_raw, yerr_bp = cam_to_egocentric(
                x_cam, y_cam, z_cam + radius,
                pitch_deg=float(intr.get("pitch_deg", GROUNDING_PITCH)),
                use_corrected_unpitch=True,
            )
            dist_bp = float(dist_bp_raw)
            bearing_bp_deg = math.degrees(yerr_bp)
            err_dist = abs(dist_bp - dist_gt)
            err_bearing = abs(math.degrees(_angle_diff(yerr_bp, yaw_err_gt)))

        labels.append(dict(
            obj_idx=oi, class_name=obj["shape_name"], class_id=SHAPE2I.get(obj["shape_name"], -1),
            color_name=obj["color_name"], color_id=COLOR2I.get(obj["color_name"], -1),
            bbox_x=x0, bbox_y=y0, bbox_w=bw, bbox_h=bh,
            centroid_px_x=cx_px, centroid_px_y=cy_px, area_px=area, clipped=bool(clipped),
            depth_median_m=depth_med,
            dist_gt_m=float(dist_gt), bearing_gt_deg=float(bearing_gt_deg),
            dist_bp_m=dist_bp, bearing_bp_deg=bearing_bp_deg,
            err_dist_m=err_dist, err_bearing_deg=err_bearing,
        ))
    return labels


# ---------------------------------------------------------------------------
# Frame capture: RGB + depth + segmentation -> per-object label rows
# ---------------------------------------------------------------------------
def capture_frame(renderer: ArenaRenderer, seg_rend: SegRenderer,
                  data_mj: mujoco.MjData, yaw: float, cam_type: str,
                  objects: list, id_to_obj: np.ndarray) -> dict:
    if cam_type == "proximity":
        rgb, depth, intr = renderer.render_proximity(data_mj, yaw, render_depth=True)
    else:
        rgb, depth, intr = renderer.render_grounding(data_mj, yaw, render_depth=True)
    seg = seg_rend.render(data_mj, yaw, cam_type)
    obj_map = seg_to_objmap(seg, id_to_obj)

    robot_xy = data_mj.qpos[0:2].copy()
    robot_yaw = _yaw_of(data_mj.qpos[3:7])
    labels = derive_object_labels(rgb, depth, obj_map, objects, robot_xy, robot_yaw, intr)

    return dict(
        rgb=rgb, depth=depth.astype(np.float16), cam_type=cam_type,
        robot_x=float(robot_xy[0]), robot_y=float(robot_xy[1]), robot_yaw=float(robot_yaw),
        qpos=data_mj.qpos.copy().astype(np.float32),
        n_objects_visible=len(labels), labels=labels,
    )


def pick_cam(dist_m: float) -> str:
    return "proximity" if dist_m <= CAM_SWITCH_DIST_M else "grounding"


# ---------------------------------------------------------------------------
# Scene construction (unifies scene.py 'easy'/'demo' + eval_search.py 'search')
# ---------------------------------------------------------------------------
def make_scene_cfg(rng: np.random.Generator, style: str, ep_i: int) -> dict:
    if style == "search":
        sc = sample_search_scene(rng, ep_i)
    else:
        sc = sample_scene(rng, style)
    return sc


# ---------------------------------------------------------------------------
# One scene -> list of frame records
# ---------------------------------------------------------------------------
def run_scene(scene_id: int, style: str, base_seed: int, ep_i: int,
             rng_sample: np.random.Generator) -> tuple:
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

    def _dual_maybe(cam_type, rgb_cam_dist):
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

    def _teleport_pose(focus_xy, dist, bearing_off_deg):
        approach_ang = float(rng_sample.uniform(-math.pi, math.pi))
        rx = focus_xy[0] + dist * math.cos(approach_ang)
        ry = focus_xy[1] + dist * math.sin(approach_ang)
        bearing_to_focus = approach_ang + math.pi
        yaw = bearing_to_focus - math.radians(bearing_off_deg)
        return rx, ry, yaw

    def _in_bounds(rx, ry):
        return abs(rx) < arena_half - margin and abs(ry) < arena_half - margin

    def _apply_pose(rx, ry, yaw):
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


# ---------------------------------------------------------------------------
# Driver: generate scenes, split by scene, write per-split artifacts
# ---------------------------------------------------------------------------
def generate(args):
    _env_note()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = (["easy"] * args.n_easy + ["demo"] * args.n_demo + ["search"] * args.n_search)
    rng_split = np.random.default_rng(np.random.SeedSequence([args.seed, 0xD5]))
    order = rng_split.permutation(len(plan))
    n_train = int(round(0.8 * len(plan)))
    n_val = int(round(0.1 * len(plan)))
    split_of_scene = {}
    for rank, idx in enumerate(order):
        if rank < n_train:
            split_of_scene[idx] = "train"
        elif rank < n_train + n_val:
            split_of_scene[idx] = "val"
        else:
            split_of_scene[idx] = "test"

    scenes_meta = {}
    per_split = {"train": [], "val": [], "test": []}  # each entry: (scene_id, style, frame_records)

    t_start = time.perf_counter()
    n_ok, n_fell = 0, 0
    for scene_id, style in enumerate(plan):
        rng_sample = np.random.default_rng(np.random.SeedSequence([args.seed, 0xA11CE, scene_id]))
        ep_i = scene_id
        scene_cfg, frame_records = run_scene(scene_id, style, args.seed, ep_i, rng_sample)
        split = split_of_scene[scene_id]
        if not frame_records:
            n_fell += 1
            print(f"  [scene {scene_id:4d}] {style:6s} SKIP (fell during settle)", flush=True)
            continue
        n_ok += 1
        scenes_meta[scene_id] = dict(
            style=style, split=split, arena_size=scene_cfg["arena_size"],
            objects=scene_cfg["objects"], target_index=scene_cfg["target_index"],
            instruction=scene_cfg["instruction"],
            lighting=scene_cfg.get("lighting", {}),
        )
        per_split[split].append((scene_id, style, scene_cfg, frame_records))
        n_frames = len(frame_records)
        elapsed = time.perf_counter() - t_start
        print(f"  [scene {scene_id:4d}] {style:6s} split={split:5s} "
              f"n_obj={len(scene_cfg['objects'])} frames={n_frames:3d}  "
              f"elapsed={elapsed:6.1f}s", flush=True)

        if args.smoke and n_ok >= args.smoke_scenes:
            break

    print(f"\n[gen] scenes ok={n_ok} fell={n_fell}  total_wall={time.perf_counter()-t_start:.1f}s",
          flush=True)

    # ---- Write per-split artifacts ----
    all_frame_rows = []
    all_label_rows = []
    frame_uid = 0
    geom_err_dist = []
    geom_err_bearing = []
    classes_counts = {s: 0 for s in SHAPE_NAMES}

    for split in ("train", "val", "test"):
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        buf = {"grounding": {"rgb": [], "depth": []}, "proximity": {"rgb": [], "depth": []}}
        idx_counter = {"grounding": 0, "proximity": 0}

        for scene_id, style, scene_cfg, frame_records in per_split[split]:
            target_idx = scene_cfg["target_index"]
            for rec in frame_records:
                ct = rec["cam_type"]
                arr_idx = idx_counter[ct]
                idx_counter[ct] += 1
                buf[ct]["rgb"].append(rec["rgb"])
                buf[ct]["depth"].append(rec["depth"])

                all_frame_rows.append(dict(
                    frame_uid=frame_uid, scene_id=scene_id, split=split, difficulty=style,
                    cam_type=ct, array_idx=arr_idx, source=rec["source"],
                    robot_x=rec["robot_x"], robot_y=rec["robot_y"], robot_yaw=rec["robot_yaw"],
                    qpos=rec["qpos"].tolist(),
                    instruction=scene_cfg["instruction"], target_index=int(target_idx),
                    n_objects_visible=rec["n_objects_visible"],
                    lighting_ambient=float(scene_cfg.get("lighting", {}).get("ambient", 0.4)),
                ))

                for lb in rec["labels"]:
                    all_label_rows.append(dict(frame_uid=frame_uid, scene_id=scene_id, split=split,
                                               cam_type=ct, is_instructed_target=(lb["obj_idx"] == target_idx),
                                               **lb))
                    classes_counts[lb["class_name"]] = classes_counts.get(lb["class_name"], 0) + 1
                    good = (not lb["clipped"]) and lb["area_px"] >= 200 and not math.isnan(lb["err_dist_m"] or np.nan)
                    if good:
                        geom_err_dist.append(lb["err_dist_m"])
                        geom_err_bearing.append(lb["err_bearing_deg"])

                frame_uid += 1

        for ct in ("grounding", "proximity"):
            if not buf[ct]["rgb"]:
                continue
            rgb_arr = np.stack(buf[ct]["rgb"], axis=0).astype(np.uint8)
            depth_arr = np.stack(buf[ct]["depth"], axis=0).astype(np.float16)
            np.savez_compressed(split_dir / f"images_{ct}.npz", rgb=rgb_arr, depth=depth_arr)
            print(f"[gen] {split}/images_{ct}.npz  rgb={rgb_arr.shape}  "
                  f"file={ (split_dir / f'images_{ct}.npz').stat().st_size/1e6:.1f}MB", flush=True)

    frames_df = pd.DataFrame(all_frame_rows)
    labels_df = pd.DataFrame(all_label_rows)

    for split in ("train", "val", "test"):
        split_dir = out_dir / split
        fdf = frames_df[frames_df["split"] == split]
        ldf = labels_df[labels_df["split"] == split]
        fdf.to_parquet(split_dir / "frames.parquet", index=False)
        ldf.to_parquet(split_dir / "labels.parquet", index=False)
        print(f"[gen] {split}: frames={len(fdf)} labels={len(ldf)}", flush=True)

    with open(out_dir / "scenes.json", "w") as f:
        json.dump(scenes_meta, f, indent=1)

    geom_err_dist = np.array(geom_err_dist, dtype=np.float64) if geom_err_dist else np.array([0.0])
    geom_err_bearing = np.array(geom_err_bearing, dtype=np.float64) if geom_err_bearing else np.array([0.0])
    p95_dist = float(np.percentile(geom_err_dist, 95))
    p95_bearing = float(np.percentile(geom_err_bearing, 95))

    n_grounding = int((frames_df["cam_type"] == "grounding").sum())
    n_proximity = int((frames_df["cam_type"] == "proximity").sum())

    meta = dict(
        frames_total=int(len(frames_df)),
        frames_grounding_cam=n_grounding,
        frames_proximity_cam=n_proximity,
        scenes=int(n_ok),
        scenes_fell=int(n_fell),
        classes=SHAPE_NAMES, colors=COLOR_NAMES,
        classes_counts=classes_counts,
        n_labels_total=int(len(labels_df)),
        label_geometry_err_m_p95=p95_dist,
        label_geometry_err_deg_p95=p95_bearing,
        label_geometry_n_checked=int(geom_err_dist.size),
        seed=args.seed,
        camera=dict(grounding=dict(w=GROUNDING_W, h=GROUNDING_H, pitch_deg=GROUNDING_PITCH),
                    proximity=dict(w=PROXIMITY_W, h=PROXIMITY_H, pitch_deg=PROXIMITY_PITCH)),
        split_scene_counts={s: len(per_split[s]) for s in ("train", "val", "test")},
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[gen] META: {json.dumps(meta, indent=2)}", flush=True)

    return meta, out_dir


# ---------------------------------------------------------------------------
# Preview: 12 random samples -> RGB + mask overlay + bbox + label text
# ---------------------------------------------------------------------------
def make_preview(out_dir: Path, n_samples: int = 12, seed: int = 12345):
    scenes_meta = json.load(open(out_dir / "scenes.json"))
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(exist_ok=True)

    frames_all = []
    for split in ("train", "val", "test"):
        fp = out_dir / split / "frames.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            df["split_dir"] = split
            frames_all.append(df)
    frames_df = pd.concat(frames_all, ignore_index=True)
    labels_all = []
    for split in ("train", "val", "test"):
        lp = out_dir / split / "labels.parquet"
        if lp.exists():
            labels_all.append(pd.read_parquet(lp))
    labels_df = pd.concat(labels_all, ignore_index=True)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(frames_df), size=min(n_samples, len(frames_df)), replace=False)

    for k, ridx in enumerate(chosen):
        row = frames_df.iloc[ridx]
        scene_id = str(int(row["scene_id"]))
        sc = scenes_meta[scene_id]
        objects = sc["objects"]
        style = sc["style"]
        scene_cfg = dict(arena_size=sc["arena_size"], objects=objects,
                         lighting=sc.get("lighting", {}))
        model = build_arena(scene_cfg)
        model.opt.timestep = 0.005
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        qpos = np.array(row["qpos"], dtype=np.float64)
        data.qpos[:len(qpos)] = qpos
        mujoco.mj_forward(model, data)

        id_to_obj = build_id_to_obj(model, len(objects))
        renderer = ArenaRenderer(model)
        seg_rend = SegRenderer(model)
        yaw = float(row["robot_yaw"])
        ct = row["cam_type"]
        if ct == "proximity":
            rgb, depth, intr = renderer.render_proximity(data, yaw, render_depth=True)
        else:
            rgb, depth, intr = renderer.render_grounding(data, yaw, render_depth=True)
        seg = seg_rend.render(data, yaw, ct)
        obj_map = seg_to_objmap(seg, id_to_obj)
        renderer.close(); seg_rend.close()

        vis = rgb.copy()
        overlay = vis.copy()
        rng_color = np.random.default_rng(0)
        palette = (rng_color.integers(60, 255, size=(len(objects), 3))).astype(np.uint8)
        for oi in range(len(objects)):
            m = obj_map == oi
            if m.sum() < MIN_PIXELS:
                continue
            overlay[m] = palette[oi]
        vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

        flab = labels_df[(labels_df["frame_uid"] == row["frame_uid"])]
        for _, lb in flab.iterrows():
            x, y, w, h = int(lb["bbox_x"]), int(lb["bbox_y"]), int(lb["bbox_w"]), int(lb["bbox_h"])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
            txt = f"{lb['color_name']} {lb['class_name']} d={lb['dist_gt_m']:.1f}m"
            cv2.putText(vis, txt, (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                       (255, 255, 0), 1, cv2.LINE_AA)

        header = f"scene={scene_id} style={style} cam={ct} src={row['source']} n_obj_vis={len(flab)}"
        canvas = np.zeros((vis.shape[0] + 18, vis.shape[1], 3), dtype=np.uint8)
        canvas[18:, :, :] = vis
        cv2.putText(canvas, header, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                   cv2.LINE_AA)
        out_path = preview_dir / f"sample_{k:02d}_scene{scene_id}_{ct}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"[preview] wrote {len(chosen)} images to {preview_dir}", flush=True)
    return len(chosen)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-easy", type=int, default=90)
    ap.add_argument("--n-demo", type=int, default=180)
    ap.add_argument("--n-search", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--out", default="dataset/det_v1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-scenes", type=int, default=2)
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_easy = min(args.n_easy, 2)
        args.n_demo = min(args.n_demo, 2)
        args.n_search = min(args.n_search, 2)

    t0 = time.perf_counter()
    meta, out_dir = generate(args)
    if not args.no_preview and meta["frames_total"] > 0:
        make_preview(out_dir, n_samples=12)
    dt = time.perf_counter() - t0
    print(f"\n[gen_det_dataset] DONE in {dt/60:.1f} min  "
          f"({meta['frames_total']} frames, {meta['scenes']} scenes)", flush=True)
    if args.smoke:
        per_scene = dt / max(1, meta["scenes"])
        per_frame = dt / max(1, meta["frames_total"])
        print(f"[smoke] {per_scene:.1f}s/scene  {per_frame*1000:.0f}ms/frame", flush=True)


if __name__ == "__main__":
    main()
