"""
nx6_heatmap_eval_utils.py — shared inference-and-score utilities for
code/train_nx6_heatmap.py (periodic val metric) and code/eval_nx6_heatmap.py
(final val/test/failcase evaluation).

Metric (per the NX-6 TRAIN brief): recall @ (bearing err < 2deg AND dist err < 0.5m)
subject to precision >= 0.9, where a "positive" detection counts as a true positive
only if the query object is actually present in the frame AND both error bars are
met; a confident detection on a negative-query frame, or a confident-but-wrong-
location detection on a positive frame, both count as false positives (standard
detection-metric convention).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from code.nx6_heatmap_data import HeatmapDataset, SplitCache, collate
from code.nx6_heatmap_model import MAX_DEPTH_CLIP_M, TARGET_INTR, decode_single


@dataclass
class InferenceResult:
    """Per-example inference outputs and ground truth, aligned 1:1 with an example list."""

    confidence: np.ndarray
    dist_pred: np.ndarray
    bearing_pred: np.ndarray
    has_target: np.ndarray   # GT presence, 0/1
    dist_gt: np.ndarray
    bearing_gt: np.ndarray
    cam_type: list
    class_id: np.ndarray
    color_id: np.ndarray
    row_i: np.ndarray


@torch.no_grad()
def run_inference(model: torch.nn.Module, cache: SplitCache, examples: list, device: str,
                  batch_size: int = 128, num_workers: int = 2) -> InferenceResult:
    """Runs the model (eval mode, no augmentation) over a fixed example list and
    decodes each prediction.

    Args:
        model: Heatmap model (callable as `model(x, q) -> (heat_logit, dist_resid)`).
        cache: SplitCache backing `examples`.
        examples: Example index as built by `build_example_index` (and optionally
            `oversample_far_or_wide`).
        device: Torch device to run inference on.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.

    Returns:
        InferenceResult with arrays aligned 1:1 with `examples`.
    """
    model.eval()
    ds = HeatmapDataset(cache, examples, train=False, seed=0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                    num_workers=num_workers)

    conf_l, dist_l, bear_l = [], [], []
    has_l, dgt_l, bgt_l = [], [], []
    cam_l, cls_l, col_l, row_l = [], [], [], []

    for batch in dl:
        x = batch["x"].to(device)
        q = batch["q"].to(device)
        heat_logit, dist_resid = model(x, q)
        heat_logit = heat_logit.cpu().numpy()
        dist_resid = dist_resid.cpu().numpy()
        depth_in = (x[:, 3].cpu().numpy()) * MAX_DEPTH_CLIP_M  # undo input normalization

        B = heat_logit.shape[0]
        for b in range(B):
            dec = decode_single(heat_logit[b], dist_resid[b], depth_in[b],
                                class_id=batch["class_id"][b], cam_type=batch["cam_type"][b],
                                conf_thresh=0.0)  # thresh applied later during sweep
            conf_l.append(dec["confidence"])
            dist_l.append(dec["dist_m"])
            bear_l.append(dec["bearing_deg"])
        has_l.append(batch["has_target"].numpy())
        dgt_l.append(batch["dist_gt"].numpy())
        bgt_l.append(batch["bearing_gt"].numpy())
        cam_l.extend(batch["cam_type"])
        cls_l.extend(batch["class_id"])
        col_l.extend(batch["color_id"])
        row_l.extend(batch["row_i"])

    return InferenceResult(
        confidence=np.array(conf_l), dist_pred=np.array(dist_l), bearing_pred=np.array(bear_l),
        has_target=np.concatenate(has_l), dist_gt=np.concatenate(dgt_l),
        bearing_gt=np.concatenate(bgt_l), cam_type=cam_l,
        class_id=np.array(cls_l), color_id=np.array(col_l), row_i=np.array(row_l),
    )


def score_at_threshold(res: InferenceResult, tau: float, bearing_tol: float = 2.0,
                       dist_tol: float = 0.5) -> dict:
    """Scores one confidence threshold: detection precision/recall with localization gating.

    Args:
        res: Inference outputs from `run_inference`.
        tau: Confidence threshold; a detection fires where confidence >= tau.
        bearing_tol: Max bearing error (deg) for a detection to count as localized.
        dist_tol: Max distance error (m) for a detection to count as localized.

    Returns:
        dict with tau, precision, recall, tp, fp, n_pos, n_detected.
    """
    detected = res.confidence >= tau
    is_pos = res.has_target > 0.5

    bearing_err = np.abs(_angle_diff_deg(res.bearing_pred, res.bearing_gt))
    dist_err = np.abs(res.dist_pred - res.dist_gt)
    localized_ok = is_pos & (bearing_err < bearing_tol) & (dist_err < dist_tol)

    tp = detected & localized_ok
    # negative-frame detections OR mislocalized positive detections
    fp = detected & (~localized_ok)
    n_pos = int(is_pos.sum())
    n_detected = int(detected.sum())

    precision = float(tp.sum()) / n_detected if n_detected > 0 else 1.0
    recall = float(tp.sum()) / n_pos if n_pos > 0 else float("nan")
    return dict(tau=float(tau), precision=precision, recall=recall,
                tp=int(tp.sum()), fp=int(fp.sum()), n_pos=n_pos, n_detected=n_detected)


def _angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest signed difference a-b wrapped to (-180,180], vectorized, degrees."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def select_threshold(res: InferenceResult, min_precision: float = 0.9,
                     bearing_tol: float = 2.0, dist_tol: float = 0.5,
                     taus: np.ndarray | None = None) -> tuple[dict, list[dict]]:
    """Sweep confidence threshold; return the operating point with the highest recall
    subject to precision >= min_precision. If none clears the bar, return the point
    with the highest precision achieved anywhere (honest fallback) with a flag.

    Args:
        res: Inference outputs from `run_inference`.
        min_precision: Minimum precision required for a threshold to be feasible.
        bearing_tol: Max bearing error (deg) for a detection to count as localized.
        dist_tol: Max distance error (m) for a detection to count as localized.
        taus: Thresholds to sweep; defaults to `linspace(0.01, 0.99, 99)`.

    Returns:
        Tuple of (best score dict, full list of per-threshold score dicts). The
        best dict has an added 'met_precision_gate' bool flag.
    """
    if taus is None:
        taus = np.concatenate([np.linspace(0.01, 0.99, 99)])
    curve = [score_at_threshold(res, t, bearing_tol, dist_tol) for t in taus]
    feasible = [c for c in curve if c["precision"] >= min_precision and c["n_detected"] > 0]
    if feasible:
        best = max(feasible, key=lambda c: c["recall"])
        best["met_precision_gate"] = True
    else:
        best = max(curve, key=lambda c: c["precision"])
        best["met_precision_gate"] = False
    return best, curve


def presence_only_pr(res: InferenceResult, tau: float) -> dict:
    """Simple presence detection precision/recall (ignores localization accuracy) --
    supplementary diagnostic, not the selection metric.

    Args:
        res: Inference outputs from `run_inference`.
        tau: Confidence threshold; a detection fires where confidence >= tau.

    Returns:
        dict with precision, recall, tp, fp, fn.
    """
    detected = res.confidence >= tau
    is_pos = res.has_target > 0.5
    tp = detected & is_pos
    fp = detected & (~is_pos)
    fn = (~detected) & is_pos
    precision = float(tp.sum()) / max(1, int(detected.sum()))
    recall = float(tp.sum()) / max(1, int(is_pos.sum()))
    return dict(precision=precision, recall=recall, tp=int(tp.sum()), fp=int(fp.sum()),
                fn=int(fn.sum()))
