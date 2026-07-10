"""
train_nx6_heatmap.py — NX-6 TRAIN (heatmap variant): trains TinyHeatmapUNet
(code/nx6_heatmap_model.py) on dataset/det_v1 (code/nx6_heatmap_data.py).

Loss: CenterNet-style penalty-reduced pixel focal loss on the presence heatmap +
Smooth-L1 on the distance residual (only supervised at the GT peak pixel, only for
positive examples).

Selection metric (periodic val eval, code/nx6_heatmap_eval_utils.py): recall @
(bearing err < 2deg AND dist err < 0.5m) subject to precision >= 0.9.

ANTI-HANG: --smoke runs a tiny subset (1 epoch, first N rows) end-to-end including
one val-metric pass, before any full run. All prints flushed.

Usage
-----
  python code/train_nx6_heatmap.py --smoke
  python code/train_nx6_heatmap.py --epochs 60 --batch 256 --out runs/nx6_heatmap_A
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("MUJOCO_GL", "egl")

from code.nx6_heatmap_model import TinyHeatmapUNet, N_CLASS, N_COLOR, TARGET_W, TARGET_H
from code.nx6_heatmap_data import (SplitCache, build_example_index, HeatmapDataset, collate,
                                   oversample_far_or_wide)
from code.nx6_heatmap_eval_utils import run_inference, select_threshold, presence_only_pr


def focal_heatmap_loss(heat_logit, heat_target, peak_mask, alpha=2.0, beta=4.0, eps=1e-6):
    """CenterNet-style penalty-reduced focal loss. Uses an elementwise mask-multiply
    (not advanced indexing / gather) to pick out the per-example GT-peak pixel --
    measured ~10x cheaper backward than `tensor[idx_b, py, px]` gather on this GPU
    (see docs/nx6_train_heatmap.md perf note)."""
    p = torch.sigmoid(heat_logit)
    neg_weight = (1.0 - heat_target).pow(beta)
    neg_loss_map = -neg_weight * p.pow(alpha) * torch.log(1.0 - p + eps)
    total_neg = neg_loss_map.sum(dim=(1, 2))  # (B,)

    pos_loss_map = -(1.0 - p).pow(alpha) * torch.log(p + eps) * peak_mask
    pos_loss = pos_loss_map.sum(dim=(1, 2))  # (B,) -- 0 for negative examples (mask all-zero)
    return total_neg, pos_loss  # caller normalizes by n_pos


def compute_loss(heat_logit, dist_resid, batch, lambda_dist=1.0):
    device = heat_logit.device
    heat_target = batch["heat"].to(device)
    peak_mask = batch["peak_mask"].to(device)
    has_target = batch["has_target"].to(device)

    total_neg, pos_loss = focal_heatmap_loss(heat_logit, heat_target, peak_mask)
    n_pos = has_target.sum().clamp(min=1.0)
    heatmap_loss = (total_neg.sum() + pos_loss.sum()) / n_pos

    resid_pred = (dist_resid * peak_mask).sum(dim=(1, 2))  # value at peak pixel (0 if negative)
    resid_gt = batch["resid"].to(device)
    dist_loss_per = F.smooth_l1_loss(resid_pred, resid_gt, reduction="none")
    dist_loss = (dist_loss_per * has_target).sum() / n_pos

    loss = heatmap_loss + lambda_dist * dist_loss
    return loss, dict(heatmap_loss=float(heatmap_loss.item()), dist_loss=float(dist_loss.item()))


def evaluate_val(model, val_cache, val_examples, device, tag=""):
    res = run_inference(model, val_cache, val_examples, device, batch_size=256, num_workers=2)
    best, curve = select_threshold(res, min_precision=0.9, bearing_tol=2.0, dist_tol=0.5)
    presence = presence_only_pr(res, tau=best["tau"])
    print(f"  [val{tag}] tau={best['tau']:.3f} precision={best['precision']:.3f} "
          f"recall={best['recall']:.3f} gate_met={best['met_precision_gate']} "
          f"(tp={best['tp']} fp={best['fp']} n_pos={best['n_pos']}) "
          f"presence_P={presence['precision']:.3f} presence_R={presence['recall']:.3f}", flush=True)
    return best, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/det_v1")
    ap.add_argument("--out", default="runs/nx6_heatmap_A")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-dist", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--neg-per-object-frame", type=int, default=1)
    ap.add_argument("--neg-per-empty-frame", type=int, default=2)
    ap.add_argument("--val-neg-per-object-frame", type=int, default=3)
    ap.add_argument("--val-neg-per-empty-frame", type=int, default=6)
    # NX-14 detector v2 (docs/nx14_detector_v2.md): additive, default-0 (byte-for-byte
    # v1-reproducing) train-set-only strengthening -- val/test sampling is untouched
    # so v1-vs-v2 offline comparisons stay apples-to-apples on the same protocol.
    ap.add_argument("--hard-color-negs", type=int, default=0,
                    help="extra same-color/different-shape (twin-distractor) negatives "
                         "per labeled train frame, on top of --neg-per-object-frame")
    ap.add_argument("--hard-shape-negs", type=int, default=0,
                    help="extra same-shape/different-color negatives per labeled train frame")
    ap.add_argument("--far-oversample", type=int, default=0,
                    help="extra duplicate copies of positive train examples beyond "
                         "--far-dist-thresh or --far-bearing-thresh (0 = off)")
    ap.add_argument("--far-dist-thresh", type=float, default=6.0)
    ap.add_argument("--far-bearing-thresh", type=float, default=20.0)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    print(f"[train_nx6_heatmap] loading caches from {args.data} ...", flush=True)
    train_cache = SplitCache(args.data, "train")
    val_cache = SplitCache(args.data, "val")
    print(f"[train_nx6_heatmap] caches loaded in {time.time()-t0:.1f}s", flush=True)

    if args.smoke:
        keep = 300
        train_cache.frames = train_cache.frames.iloc[:keep]
        train_cache.rgb = train_cache.rgb[:keep]
        train_cache.depth = train_cache.depth[:keep]
        train_cache.cam_type = train_cache.cam_type[:keep]
        train_cache.row_labels = train_cache.row_labels[:keep]
        vkeep = 150
        val_cache.frames = val_cache.frames.iloc[:vkeep]
        val_cache.rgb = val_cache.rgb[:vkeep]
        val_cache.depth = val_cache.depth[:vkeep]
        val_cache.cam_type = val_cache.cam_type[:vkeep]
        val_cache.row_labels = val_cache.row_labels[:vkeep]
        args.epochs = 2
        args.eval_every = 1
        args.save_every = 1
        args.batch = 8
        args.num_workers = 0
        print("[train_nx6_heatmap] SMOKE MODE: subset + 2 epochs", flush=True)

    val_rng = np.random.default_rng(12345)
    val_examples = build_example_index(val_cache, val_rng,
                                       neg_per_object_frame=args.val_neg_per_object_frame,
                                       neg_per_empty_frame=args.val_neg_per_empty_frame)
    n_val_pos = sum(1 for e in val_examples if e[3] is not None)
    print(f"[train_nx6_heatmap] val examples: {len(val_examples)} "
          f"({n_val_pos} positive, {len(val_examples)-n_val_pos} negative)", flush=True)

    model = TinyHeatmapUNet().to(args.device)
    print(f"[train_nx6_heatmap] model params: {model.num_params()} "
          f"({model.num_params()/1e6:.3f}M)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    curves = {"train_loss": [], "val": []}
    best_score = -1e9
    best_epoch = -1

    model_cfg = dict(in_ch=4, base=32, embed_dim=64, query_dim=N_CLASS + N_COLOR)

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        epoch_rng = np.random.default_rng(args.seed * 7919 + epoch)
        train_examples = build_example_index(
            train_cache, epoch_rng, neg_per_object_frame=args.neg_per_object_frame,
            neg_per_empty_frame=args.neg_per_empty_frame,
            hard_color_negs=args.hard_color_negs, hard_shape_negs=args.hard_shape_negs)
        train_examples = oversample_far_or_wide(
            train_examples, extra_copies=args.far_oversample,
            dist_thresh_m=args.far_dist_thresh, bearing_thresh_deg=args.far_bearing_thresh)
        train_ds = HeatmapDataset(train_cache, train_examples, train=True,
                                  seed=args.seed * 1000 + epoch, sigma=args.sigma)
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers,
                              drop_last=True, persistent_workers=False)

        model.train()
        running = {"loss": 0.0, "heatmap_loss": 0.0, "dist_loss": 0.0, "n": 0}
        for step, batch in enumerate(train_dl):
            x = batch["x"].to(args.device, non_blocking=True)
            q = batch["q"].to(args.device, non_blocking=True)
            heat_logit, dist_resid = model(x, q)
            loss, parts = compute_loss(heat_logit, dist_resid, batch, lambda_dist=args.lambda_dist)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running["loss"] += float(loss.item())
            running["heatmap_loss"] += parts["heatmap_loss"]
            running["dist_loss"] += parts["dist_loss"]
            running["n"] += 1

        sched.step()
        n = max(1, running["n"])
        avg_loss = running["loss"] / n
        curves["train_loss"].append(dict(epoch=epoch, loss=avg_loss,
                                         heatmap_loss=running["heatmap_loss"] / n,
                                         dist_loss=running["dist_loss"] / n,
                                         n_examples=len(train_examples),
                                         lr=sched.get_last_lr()[0]))
        print(f"[epoch {epoch}/{args.epochs}] n_ex={len(train_examples)} steps={n} "
              f"loss={avg_loss:.4f} heat={running['heatmap_loss']/n:.4f} "
              f"dist={running['dist_loss']/n:.4f} lr={sched.get_last_lr()[0]:.2e} "
              f"time={time.time()-ep_t0:.1f}s", flush=True)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            best, val_curve = evaluate_val(model, val_cache, val_examples, args.device,
                                           tag=f" ep{epoch}")
            curves["val"].append(dict(epoch=epoch, **{k: v for k, v in best.items()}))
            score = best["recall"] if best["met_precision_gate"] else (best["precision"] - 10.0)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(dict(model_state=model.state_dict(), model_cfg=model_cfg,
                               epoch=epoch, val_metric=best), os.path.join(args.out, "model_best.pt"))
                print(f"  -> new best (score={score:.4f}), saved model_best.pt", flush=True)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(dict(model_state=model.state_dict(), model_cfg=model_cfg, epoch=epoch),
                      os.path.join(args.out, f"epoch_{epoch:04d}.pt"))

        with open(os.path.join(args.out, "curves.json"), "w") as f:
            json.dump(curves, f, indent=2)

    print(f"[train_nx6_heatmap] DONE. best_epoch={best_epoch} best_score={best_score:.4f} "
          f"total_time={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
