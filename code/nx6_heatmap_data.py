"""
nx6_heatmap_data.py — NX-6 TRAIN (heatmap variant): dataset loading, resizing,
target-heatmap construction, and augmentation for dataset/det_v1 and
dataset/det_failcases.

Shared by code/train_nx6_heatmap.py and code/eval_nx6_heatmap.py.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from code.nx6_heatmap_model import (
    CLASS_NAMES, COLOR_NAMES, N_CLASS, N_COLOR, SIZE_M, TARGET_W, TARGET_H,
    TARGET_INTR, PITCH_BY_CAM, MAX_DEPTH_CLIP_M, encode_query,
)
from code.arena import (GROUNDING_W, GROUNDING_H, PROXIMITY_W, PROXIMITY_H,
                        backproject_pixel)
from code.grounding import cam_to_egocentric

ORIG_WH = {"grounding": (GROUNDING_W, GROUNDING_H), "proximity": (PROXIMITY_W, PROXIMITY_H)}
ALL_COMBOS = [(ci, co) for ci in range(N_CLASS) for co in range(N_COLOR)]  # 28


# ---------------------------------------------------------------------------
# Cache: resize every frame once to (TARGET_H, TARGET_W); attach per-frame label list
# with pixel coords already rescaled to the target canvas.
# ---------------------------------------------------------------------------
class SplitCache:
    """Holds resized RGB/depth arrays + per-frame label lists for one dataset split
    (or the failcases set, which uses the same {images_*.npz, frames.parquet,
    labels.parquet} layout, see docs/nx6_data.md §6/§7)."""

    def __init__(self, root: str, split: str | None, verbose: bool = True):
        self.root = root
        self.split = split
        base = os.path.join(root, split) if split else root
        self.frames = pd.read_parquet(os.path.join(base, "frames.parquet"))
        self.labels = pd.read_parquet(os.path.join(base, "labels.parquet"))
        self.frames = self.frames.sort_values("frame_uid").reset_index(drop=True)

        # frame_uid -> row index in self.frames
        self.uid2row = {int(u): i for i, u in enumerate(self.frames["frame_uid"].values)}

        # group labels by frame_uid
        self.labels_by_frame: dict[int, list[dict]] = {}
        for r in self.labels.itertuples():
            self.labels_by_frame.setdefault(int(r.frame_uid), []).append(dict(
                class_id=int(r.class_id), color_id=int(r.color_id),
                cx=float(r.centroid_px_x), cy=float(r.centroid_px_y),
                dist_gt=float(r.dist_gt_m), bearing_gt=float(r.bearing_gt_deg),
                clipped=bool(r.clipped), area_px=int(r.area_px),
            ))

        # IMPORTANT: NpzFile.__getitem__ decompresses the WHOLE array from the zip on
        # every call (no caching) -- must extract each array ONCE, not per-row, or
        # this becomes O(n_rows) full-array decompressions (catastrophically slow).
        npz_cache = {}
        for cam in ("grounding", "proximity"):
            p = os.path.join(base, f"images_{cam}.npz")
            if os.path.exists(p):
                with np.load(p) as z:
                    npz_cache[cam] = dict(rgb=z["rgb"], depth=z["depth"])

        n = len(self.frames)
        self.rgb = np.zeros((n, TARGET_H, TARGET_W, 3), dtype=np.uint8)
        self.depth = np.zeros((n, TARGET_H, TARGET_W), dtype=np.float32)
        self.cam_type = self.frames["cam_type"].values
        self.frame_uid = self.frames["frame_uid"].values.astype(np.int64)

        # per-row rescaled label list (pixel coords already in TARGET canvas)
        self.row_labels: list[list[dict]] = [[] for _ in range(n)]

        for i, row in enumerate(self.frames.itertuples()):
            cam = row.cam_type
            arr = npz_cache[cam]
            idx = int(row.array_idx)
            rgb_o = arr["rgb"][idx]
            depth_o = arr["depth"][idx].astype(np.float32)
            self.rgb[i] = cv2.resize(rgb_o, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            self.depth[i] = cv2.resize(depth_o, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

            ow, oh = ORIG_WH[cam]
            sx, sy = TARGET_W / ow, TARGET_H / oh
            for lb in self.labels_by_frame.get(int(row.frame_uid), []):
                self.row_labels[i].append(dict(
                    class_id=lb["class_id"], color_id=lb["color_id"],
                    cx=lb["cx"] * sx, cy=lb["cy"] * sy,
                    dist_gt=lb["dist_gt"], bearing_gt=lb["bearing_gt"],
                    clipped=lb["clipped"], area_px=lb["area_px"],
                ))
            if verbose and (i + 1) % 3000 == 0:
                print(f"  [SplitCache {split or root}] resized {i+1}/{n}", flush=True)

        if verbose:
            print(f"[SplitCache {split or root}] {n} frames ready "
                  f"({(self.rgb.nbytes+self.depth.nbytes)/1e6:.0f}MB)", flush=True)

    def __len__(self):
        return len(self.frames)


def load_failcase_cache(root: str = "dataset/det_failcases", verbose: bool = True) -> SplitCache:
    """dataset/det_failcases uses a different npz naming convention than dataset/det_v1
    (`images_{ep_tag}_{cam_type}.npz` instead of `images_{cam_type}.npz`, since it's a
    handful of live-replay episodes rather than a train/val/test split, docs/nx6_data.md
    §6). Builds a SplitCache-compatible object (same attributes: .frames, .rgb, .depth,
    .cam_type, .row_labels) without going through SplitCache.__init__'s split-dir /
    fixed-npz-name assumptions."""
    cache = SplitCache.__new__(SplitCache)
    cache.root = root
    cache.split = None
    cache.frames = pd.read_parquet(os.path.join(root, "frames.parquet")).sort_values(
        "frame_uid").reset_index(drop=True)
    cache.labels = pd.read_parquet(os.path.join(root, "labels.parquet"))
    cache.uid2row = {int(u): i for i, u in enumerate(cache.frames["frame_uid"].values)}

    cache.labels_by_frame = {}
    for r in cache.labels.itertuples():
        cache.labels_by_frame.setdefault(int(r.frame_uid), []).append(dict(
            class_id=int(r.class_id), color_id=int(r.color_id),
            cx=float(r.centroid_px_x), cy=float(r.centroid_px_y),
            dist_gt=float(r.dist_gt_m), bearing_gt=float(r.bearing_gt_deg),
            clipped=bool(r.clipped), area_px=int(r.area_px),
            is_instructed_target=bool(r.is_instructed_target),
        ))

    npz_cache = {}
    for ep_tag, cam in cache.frames[["ep_tag", "cam_type"]].drop_duplicates().itertuples(index=False):
        p = os.path.join(root, f"images_{ep_tag}_{cam}.npz")
        with np.load(p) as z:
            npz_cache[(ep_tag, cam)] = dict(rgb=z["rgb"], depth=z["depth"])

    n = len(cache.frames)
    cache.rgb = np.zeros((n, TARGET_H, TARGET_W, 3), dtype=np.uint8)
    cache.depth = np.zeros((n, TARGET_H, TARGET_W), dtype=np.float32)
    cache.cam_type = cache.frames["cam_type"].values
    cache.frame_uid = cache.frames["frame_uid"].values.astype(np.int64)
    cache.row_labels = [[] for _ in range(n)]
    cache.ep_tag = cache.frames["ep_tag"].values
    cache.step = cache.frames["step"].values
    cache.target_color = cache.frames["target_color"].values
    cache.target_shape = cache.frames["target_shape"].values
    cache.gt_dist_true_target_m = cache.frames["gt_dist_true_target_m"].values
    cache.gt_bearing_true_target_deg = cache.frames["gt_bearing_true_target_deg"].values
    cache.classical_dist_m = cache.frames["classical_dist_m"].values
    cache.classical_not_visible = cache.frames["classical_not_visible"].values
    cache.episode_failure_tag = cache.frames["episode_failure_tag"].values
    cache.episode_final_dist = cache.frames["episode_final_dist"].values

    for i, row in enumerate(cache.frames.itertuples()):
        arr = npz_cache[(row.ep_tag, row.cam_type)]
        idx = int(row.array_idx)
        rgb_o = arr["rgb"][idx]
        depth_o = arr["depth"][idx].astype(np.float32)
        cache.rgb[i] = cv2.resize(rgb_o, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        cache.depth[i] = cv2.resize(depth_o, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

        ow, oh = ORIG_WH[row.cam_type]
        sx, sy = TARGET_W / ow, TARGET_H / oh
        for lb in cache.labels_by_frame.get(int(row.frame_uid), []):
            d = dict(class_id=lb["class_id"], color_id=lb["color_id"],
                     cx=lb["cx"] * sx, cy=lb["cy"] * sy,
                     dist_gt=lb["dist_gt"], bearing_gt=lb["bearing_gt"],
                     clipped=lb["clipped"], area_px=lb["area_px"])
            d["is_instructed_target"] = lb["is_instructed_target"]
            cache.row_labels[i].append(d)

    if verbose:
        print(f"[load_failcase_cache] {n} frames, {len(cache.labels)} labels, "
              f"{len(npz_cache)} episode/cam npz files", flush=True)
    return cache


# ---------------------------------------------------------------------------
# Geometry helper: residual target for a positive example (same pipeline as
# code/gen_det_dataset.py:derive_object_labels, but evaluated at a single
# (possibly resized/augmented) pixel + intrinsics rather than the mask median).
# ---------------------------------------------------------------------------
def residual_target(cx: float, cy: float, depth_at_px: float, dist_gt: float,
                    class_id: int, cam_type: str, intr: dict) -> float:
    if depth_at_px <= 0 or not np.isfinite(depth_at_px):
        return 0.0
    radius = SIZE_M.get(CLASS_NAMES[class_id], 0.24) / 2.0
    x_cam, y_cam, z_cam = backproject_pixel(cx, cy, depth_at_px, intr)
    pitch = PITCH_BY_CAM[cam_type]
    dist_bp, _ = cam_to_egocentric(x_cam, y_cam, z_cam + radius, pitch_deg=pitch,
                                    use_corrected_unpitch=True)
    return float(dist_gt - dist_bp)


def gaussian_heatmap(h, w, cx, cy, sigma=2.5):
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    hm = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma))
    return hm.astype(np.float32)


# ---------------------------------------------------------------------------
# Example index: (row_idx, class_id, color_id, target_label_or_None)
# ---------------------------------------------------------------------------
def build_example_index(cache: SplitCache, rng: np.random.Generator,
                        neg_per_object_frame: int = 1, neg_per_empty_frame: int = 2):
    """One entry per positive label + sampled negative queries per frame."""
    examples = []
    n = len(cache)
    for i in range(n):
        labs = cache.row_labels[i]
        present = {(l["class_id"], l["color_id"]) for l in labs}
        for l in labs:
            examples.append((i, l["class_id"], l["color_id"], l))
        n_neg = neg_per_empty_frame if not labs else neg_per_object_frame
        complement = [c for c in ALL_COMBOS if c not in present]
        if complement and n_neg > 0:
            pick = rng.choice(len(complement), size=min(n_neg, len(complement)), replace=False)
            for k in pick:
                ci, co = complement[k]
                examples.append((i, ci, co, None))
    return examples


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def _photometric(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = rgb.astype(np.float32)
    if rng.random() < 0.7:
        out *= rng.uniform(0.75, 1.3)
    if rng.random() < 0.5:
        out += rng.uniform(-20, 20)
    if rng.random() < 0.5:
        out += rng.normal(0, 6.0, size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _crop_resize(rgb, depth, cx, cy, has_target, rng, intr):
    """Random resized crop in [0.8,1.0] scale; keeps target center inside crop (with
    margin) for positive examples. Returns new rgb, depth, new_cx, new_cy, new_intr."""
    H, W = rgb.shape[:2]
    scale = rng.uniform(0.8, 1.0)
    cw, ch = int(W * scale), int(H * scale)
    if has_target and cx is not None:
        margin = 4
        x0_lo = max(0, int(cx) - cw + margin)
        x0_hi = min(W - cw, int(cx) - margin)
        y0_lo = max(0, int(cy) - ch + margin)
        y0_hi = min(H - ch, int(cy) - margin)
        x0 = rng.integers(x0_lo, x0_hi + 1) if x0_hi >= x0_lo else rng.integers(0, W - cw + 1)
        y0 = rng.integers(y0_lo, y0_hi + 1) if y0_hi >= y0_lo else rng.integers(0, H - ch + 1)
    else:
        x0 = rng.integers(0, W - cw + 1)
        y0 = rng.integers(0, H - ch + 1)

    rgb_c = rgb[y0:y0 + ch, x0:x0 + cw]
    depth_c = depth[y0:y0 + ch, x0:x0 + cw]
    rgb_r = cv2.resize(rgb_c, (W, H), interpolation=cv2.INTER_AREA)
    depth_r = cv2.resize(depth_c, (W, H), interpolation=cv2.INTER_NEAREST)

    sx, sy = W / cw, H / ch
    new_intr = dict(intr)
    new_intr["fx"] = intr["fx"] * sx
    new_intr["fy"] = intr["fy"] * sy
    new_intr["cx"] = (intr["cx"] - x0) * sx
    new_intr["cy"] = (intr["cy"] - y0) * sy

    new_cx = new_cy = None
    if cx is not None:
        new_cx = (cx - x0) * sx
        new_cy = (cy - y0) * sy
        if not (0 <= new_cx < W and 0 <= new_cy < H):
            new_cx = new_cy = None  # cropped out -> becomes a negative example
    return rgb_r, depth_r, new_cx, new_cy, new_intr


def _depth_dropout(depth_in: np.ndarray, rng: np.random.Generator):
    """depth_in: normalized [0,1] depth channel (post-clip/scale). Returns possibly
    corrupted copy + a flag for which mode was applied (for the *input* only --
    the residual target always uses the true un-corrupted depth, see build_batch)."""
    r = rng.random()
    if r < 0.15:
        return np.zeros_like(depth_in)               # full channel dropout
    elif r < 0.40:
        out = depth_in.copy()
        near_mask = out < (1.2 / MAX_DEPTH_CLIP_M)     # near-field ~<1.2m is noisiest
        noise = rng.normal(1.0, 0.15, size=out.shape).astype(np.float32)
        out = np.where(near_mask, np.clip(out * noise, 0, 1), out)
        return out
    return depth_in


class HeatmapDataset(Dataset):
    """Randomized per-epoch example sampling + augmentation. Set `train=False` to
    disable augmentation and reuse a fixed example list (for val/test)."""

    def __init__(self, cache: SplitCache, examples: list, train: bool, seed: int = 0,
                sigma: float = 2.5):
        self.cache = cache
        self.examples = examples
        self.train = train
        self.sigma = sigma
        self.seed = seed

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        row_i, class_id, color_id, lab = self.examples[idx]
        rng = np.random.default_rng((self.seed * 1_000_003 + idx) & 0xFFFFFFFF)

        rgb = self.cache.rgb[row_i].copy()
        depth_m = self.cache.depth[row_i].copy()
        cam_type = str(self.cache.cam_type[row_i])
        intr = dict(TARGET_INTR)

        cx = cy = None
        dist_gt = None
        if lab is not None:
            cx, cy, dist_gt = lab["cx"], lab["cy"], lab["dist_gt"]

        if self.train:
            rgb, depth_m, cx, cy, intr = _crop_resize(
                rgb, depth_m, cx, cy, lab is not None, rng, intr)
            rgb = _photometric(rgb, rng)

        H, W = depth_m.shape
        has_target = cx is not None
        peak_mask = np.zeros((H, W), dtype=np.float32)
        if has_target:
            heat = gaussian_heatmap(H, W, cx, cy, sigma=self.sigma)
            px, py = int(round(cx)), int(round(cy))
            px = min(max(px, 0), W - 1)
            py = min(max(py, 0), H - 1)
            peak_mask[py, px] = 1.0
            depth_at_px = float(depth_m[py, px])
            resid = residual_target(float(px), float(py), depth_at_px, dist_gt,
                                    class_id, cam_type, intr)
        else:
            heat = np.zeros((H, W), dtype=np.float32)
            px = py = 0
            resid = 0.0

        depth_in = np.clip(depth_m, 0.0, MAX_DEPTH_CLIP_M) / MAX_DEPTH_CLIP_M
        if self.train:
            depth_in = _depth_dropout(depth_in, rng)

        x = np.concatenate([rgb.astype(np.float32) / 255.0, depth_in[..., None]], axis=-1)
        x_t = torch.from_numpy(x.transpose(2, 0, 1)).float()
        q_t = torch.from_numpy(encode_query(class_id, color_id)).float()
        heat_t = torch.from_numpy(heat).float()

        peak_mask_t = torch.from_numpy(peak_mask).float()
        return dict(x=x_t, q=q_t, heat=heat_t, has_target=torch.tensor(float(has_target)),
                    resid=torch.tensor(float(resid)), py=torch.tensor(py), px=torch.tensor(px),
                    peak_mask=peak_mask_t,
                    dist_gt=torch.tensor(float(dist_gt) if dist_gt is not None else float("nan")),
                    bearing_gt=torch.tensor(float(lab["bearing_gt"]) if lab is not None else float("nan")),
                    cam_type=cam_type, class_id=class_id, color_id=color_id, row_i=row_i)


def collate(batch):
    out = {}
    for k in ("x", "q", "heat", "has_target", "resid", "py", "px", "dist_gt", "bearing_gt",
             "peak_mask"):
        out[k] = torch.stack([b[k] for b in batch])
    out["cam_type"] = [b["cam_type"] for b in batch]
    out["class_id"] = [b["class_id"] for b in batch]
    out["color_id"] = [b["color_id"] for b in batch]
    out["row_i"] = [b["row_i"] for b in batch]
    return out
