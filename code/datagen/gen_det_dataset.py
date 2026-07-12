"""
code/datagen/gen_det_dataset.py — NX-6: labeled dataset generator for a learned object detector
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

Role: CLI entry point (argparse + `generate` driver + `make_preview`).
Split out (RF-1) from the physics/label pieces, which live in sibling modules:
  - code/datagen/gen_det_common.py   — shared constants, _env_note, pick_cam
  - code/datagen/gen_det_labels.py   — build_id_to_obj, seg_to_objmap,
                                        derive_object_labels
  - code/datagen/gen_det_capture.py  — SegRenderer, capture_frame
  - code/datagen/gen_det_scene.py    — make_scene_cfg, run_scene
build_id_to_obj, SegRenderer, seg_to_objmap, derive_object_labels, and
pick_cam are re-exported here for old-path compat (gen_det_failcases.py and
eval/nx14_gen1_confusion/capture.py import them from `code.gen_det_dataset`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
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

_HERE: Path = Path(__file__).resolve().parent
_REPO: Path = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from code.arena import (
    ArenaRenderer, GROUNDING_H, GROUNDING_PITCH, GROUNDING_W, PROXIMITY_H,
    PROXIMITY_PITCH, PROXIMITY_W, build_arena,
)
from code.datagen.gen_det_capture import SegRenderer
from code.datagen.gen_det_common import (
    COLOR_NAMES, MIN_PIXELS, SHAPE_NAMES, _env_note, pick_cam,
)
from code.datagen.gen_det_labels import build_id_to_obj, derive_object_labels, seg_to_objmap
from code.datagen.gen_det_scene import run_scene

__all__ = [
    "build_id_to_obj", "SegRenderer", "seg_to_objmap", "derive_object_labels", "pick_cam",
    "generate", "make_preview", "main",
]


# ---------------------------------------------------------------------------
# Driver: generate scenes, split by scene, write per-split artifacts
# ---------------------------------------------------------------------------
def generate(args: argparse.Namespace) -> tuple[dict, Path]:
    """Generates scenes, captures frames, and writes per-split dataset artifacts.

    Args:
        args: Parsed command-line arguments (see `main`), with fields
            `n_easy`, `n_demo`, `n_search`, `seed`, `out`, `smoke`, and
            `smoke_scenes`.

    Returns:
        A tuple (meta, out_dir): the dataset summary dict (also written to
        `meta.json`) and the output directory path.
    """
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
def make_preview(out_dir: Path, n_samples: int = 12, seed: int = 12345) -> int:
    """Writes preview images (RGB + mask overlay + bboxes + text) to disk.

    Args:
        out_dir: Dataset output directory containing per-split
            frames/labels parquet files and `scenes.json`.
        n_samples: Number of random frames to sample for previews.
        seed: Random seed for sample selection and the label color palette.

    Returns:
        The number of preview images written.
    """
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
def main() -> None:
    """Parses CLI arguments and runs dataset generation (plus previews)."""
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
