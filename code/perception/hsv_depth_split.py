"""
code/perception/hsv_depth_split.py — NX-4 depth-guided component splitting +
CAM-2 depth-outlier (self-body) rejection for the classical HSV+depth
grounding pipeline (RF-1 split of code/grounding.py; docs/nx4_depth_split.md,
docs/cam_p1.md).

Two independent mechanisms live here:
  - Depth-histogram clustering + component splitting (`_split_component_by_depth`,
    `_split_contours_by_depth`, `_histogram_depth_clusters`) that separates a
    connected HSV-mask blob into depth-consistent sub-blobs BEFORE size/shape
    scoring, so a target fused with an adjacent same-hue wall/distractor region
    can be told apart from it.
  - Depth-outlier rejection (`_reject_depth_outliers`) that keeps only the
    single largest contiguous depth cluster within an already-selected blob's
    mask, discarding a disjoint contaminating population (robot self-body /
    anti-aliased edge bleed) -- used only for the proximity/widefov cameras.

`_quick_candidate_depth` and `_circle_fill_score` are small helpers used by
the NX-4 re-selection pass in code/perception/hsv_pipeline.py.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from code.perception.hsv_config import (DEPTH_CLUSTER_BIN_M, EROSION_ITER,
                                        GROUND_SPLIT_BIN_M, GROUND_SPLIT_GAP_M,
                                        GROUND_SPLIT_MIN_PIECE_PX,
                                        GROUND_SPLIT_MIN_SAMPLES, MIN_VALID_DEPTH_PX)


def _histogram_depth_clusters(depth_vals: np.ndarray, bin_m: float = GROUND_SPLIT_BIN_M,
                              gap_m: float = GROUND_SPLIT_GAP_M) -> list:
    """
    Cluster a 1-D array of depth samples into contiguous-occupancy runs in a
    fine histogram, merging runs separated by a zero-bin gap SHORTER than
    gap_m (i.e. only a genuine >= gap_m physical gap in the data counts as a
    split point). Returns a list of (lo_m, hi_m) tuples, sorted ascending,
    covering every input sample exactly once (no overlaps, no gaps skipped).
    A single-element result means "one depth-consistent population" (no
    split warranted).
    """
    if depth_vals.size == 0:
        return []
    lo, hi = float(depth_vals.min()), float(depth_vals.max())
    if hi - lo < gap_m:
        return [(lo, hi)]
    nbins = max(1, int(math.ceil((hi - lo) / bin_m)))
    hist, edges = np.histogram(depth_vals, bins=nbins, range=(lo, hi))
    gap_bins = max(1, int(math.ceil(gap_m / bin_m)))

    occupied = hist > 0
    runs = []  # [start_bin, end_bin_inclusive]
    i = 0
    while i < nbins:
        if not occupied[i]:
            i += 1
            continue
        j = i
        while j + 1 < nbins and occupied[j + 1]:
            j += 1
        runs.append([i, j])
        i = j + 1
    if not runs:
        return []

    merged = [runs[0]]
    for r in runs[1:]:
        prev = merged[-1]
        zero_gap = r[0] - prev[1] - 1
        if zero_gap < gap_bins:
            prev[1] = r[1]
        else:
            merged.append(r)

    return [(float(edges[s]), float(edges[e + 1])) for s, e in merged]


def _split_component_by_depth(comp_mask: np.ndarray, depth_map: np.ndarray,
                              min_depth: float, max_depth: float,
                              bin_m: float = GROUND_SPLIT_BIN_M,
                              gap_m: float = GROUND_SPLIT_GAP_M,
                              min_samples: int = GROUND_SPLIT_MIN_SAMPLES,
                              min_piece_px: int = GROUND_SPLIT_MIN_PIECE_PX) -> list:
    """
    Split a single connected HSV-mask component into depth-consistent
    sub-masks. Returns a list of boolean masks (same shape as comp_mask); a
    length-1 list `[comp_mask]` means "no split" (either < 2 depth clusters
    found, or too few valid-depth samples to trust clustering at all --
    conservative no-op in every ambiguous case).

    Pixels with a valid depth reading are assigned to the nearest depth
    cluster unambiguously (clusters partition the observed depth range with
    no overlap, by construction of `_histogram_depth_clusters`). Pixels
    WITHOUT a valid depth reading (occluded / out of [min_depth, max_depth])
    are assigned to the cluster of their nearest assigned neighbour in 2-D
    image space (task brief: "keep them with their 2D neighbours' mode"),
    via `scipy.ndimage.distance_transform_edt`'s nearest-index return; if
    scipy is unavailable those pixels are simply left out of every sub-mask
    (safe no-op -- they contribute to no candidate's area/bbox either way).
    """
    ys, xs = np.where(comp_mask)
    if ys.size == 0:
        return [comp_mask]
    depths = depth_map[ys, xs]
    valid = np.isfinite(depths) & (depths > min_depth) & (depths < max_depth)
    if int(valid.sum()) < min_samples:
        return [comp_mask]

    clusters = _histogram_depth_clusters(depths[valid], bin_m, gap_m)
    if len(clusters) < 2:
        return [comp_mask]

    edges_hi = np.array([c[1] for c in clusters])

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    local_mask  = comp_mask[y0:y1, x0:x1]
    local_depth = depth_map[y0:y1, x0:x1]
    ly, lx = np.where(local_mask)
    ld = local_depth[ly, lx]
    lvalid = np.isfinite(ld) & (ld > min_depth) & (ld < max_depth)

    label = np.full(local_mask.shape, -1, dtype=np.int32)
    vld = ld[lvalid]
    cluster_idx = np.clip(np.searchsorted(edges_hi, vld, side='left'), 0, len(clusters) - 1)
    label[ly[lvalid], lx[lvalid]] = cluster_idx

    unassigned = local_mask & (label == -1)
    if unassigned.any() and (label >= 0).any():
        try:
            from scipy import ndimage
            _, indices = ndimage.distance_transform_edt(
                label == -1, return_distances=True, return_indices=True)
            propagated = label[tuple(indices)]
            label = np.where(unassigned, propagated, label)
        except Exception:
            pass   # leave those pixels unassigned (dropped from every sub-mask)

    sub_masks = []
    for c in range(len(clusters)):
        piece = (label == c) & local_mask
        if int(piece.sum()) < min_piece_px:
            continue
        full = np.zeros_like(comp_mask)
        full[y0:y1, x0:x1] = piece
        sub_masks.append(full)

    if len(sub_masks) < 2:
        return [comp_mask]
    return sub_masks


def _split_contours_by_depth(contours: list, mask_shape: tuple, ego_depth: np.ndarray,
                             min_depth: float, max_depth: float) -> list:
    """
    Expand each input HSV-mask contour into one or more depth-consistent
    sub-contours (docs/nx4_depth_split.md §1). Components that are already
    depth-pure, or too small to safely judge, pass through unchanged as a
    single-element result. Purely a candidate-set EXPANSION -- scoring and
    selection happen downstream exactly as before, now over the (possibly
    larger) candidate list.
    """
    out = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < GROUND_SPLIT_MIN_PIECE_PX:
            continue   # never a viable candidate either way; skip cheaply
        comp_mask = np.zeros(mask_shape, dtype=np.uint8)
        cv2.drawContours(comp_mask, [cnt], -1, 255, cv2.FILLED)
        pieces = _split_component_by_depth(comp_mask > 0, ego_depth, min_depth, max_depth)
        if len(pieces) == 1:
            out.append(cnt)
            continue
        for piece in pieces:
            piece_u8 = piece.astype(np.uint8) * 255
            sub_contours, _ = cv2.findContours(piece_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for sc in sub_contours:
                if float(cv2.contourArea(sc)) >= GROUND_SPLIT_MIN_PIECE_PX:
                    out.append(sc)
    return out


def _quick_candidate_depth(cnt: np.ndarray, mask_shape: tuple, ego_depth: np.ndarray,
                           min_depth_eff: float, max_depth_m: float,
                           erosion_iter: int = EROSION_ITER) -> tuple:
    """
    Cheap per-candidate median depth for the NX-4 re-selection pass (a
    lighter-weight version of the main pipeline's erode-then-median step,
    run once per split candidate rather than once per accepted detection).
    Returns (depth_m_or_None, blob_mask_uint8). None depth means "too few
    valid-depth pixels to trust" -- caller treats that candidate as
    NOT size-plausible (conservative; never lets an unverifiable candidate
    win over a verified-plausible one).
    """
    blob_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(blob_mask, [cnt], -1, 255, cv2.FILLED)
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded  = cv2.erode(blob_mask, erode_k, iterations=erosion_iter)
    depth_vals = ego_depth[eroded > 0]
    valid = depth_vals[(depth_vals > min_depth_eff) & (depth_vals < max_depth_m)]
    if len(valid) < MIN_VALID_DEPTH_PX:
        return None, blob_mask
    return float(np.median(valid)), blob_mask


def _circle_fill_score(cnt: np.ndarray) -> float:
    """
    GROUND_SHAPE (docs/nx4_depth_split.md §3): contour area / minimum-
    enclosing-circle area. A true ball's silhouette IS (approximately) a
    circle, so this ratio sits close to 1.0; a cube's silhouette is a
    quadrilateral inscribed in its own bounding circle, so the ratio sits
    well below 1.0 (a square inscribed in a circle covers exactly 2/pi =
    0.6366 of it). Used only to arbitrate a same-color ball-vs-cube twin
    (e.g. ep12's cyan ball distractor vs the cyan cube target) when BOTH
    candidates already passed the size-plausibility band -- see the
    `GROUND_SHAPE` arbitration block in `ground()`.
    """
    area = float(cv2.contourArea(cnt))
    (_, _), r = cv2.minEnclosingCircle(cnt)
    circ_area = math.pi * r * r
    if circ_area < 1e-6:
        return 0.0
    return float(min(1.0, area / circ_area))


def _reject_depth_outliers(depth_vals: np.ndarray,
                           bin_m: float = DEPTH_CLUSTER_BIN_M) -> np.ndarray:
    """
    Return the subset of depth_vals belonging to the single largest contiguous
    histogram cluster (the modal object surface), discarding a disjoint minority
    population (self-body / edge-bleed contamination -- see module comment
    above CAM-2's depth-outlier rejection in code/perception/hsv_config.py).

    No-op (returns depth_vals unchanged) when there are too few samples to cluster
    meaningfully, or when all samples already sit within one bin.
    """
    if depth_vals.size < 4:
        return depth_vals
    lo, hi = float(depth_vals.min()), float(depth_vals.max())
    if hi - lo < bin_m:
        return depth_vals
    nbins = max(1, int(math.ceil((hi - lo) / bin_m)))
    hist, edges = np.histogram(depth_vals, bins=nbins, range=(lo, hi))
    peak = int(np.argmax(hist))
    lo_bin, hi_bin = peak, peak
    while lo_bin > 0 and hist[lo_bin - 1] > 0:
        lo_bin -= 1
    while hi_bin < len(hist) - 1 and hist[hi_bin + 1] > 0:
        hi_bin += 1
    lo_val, hi_val = float(edges[lo_bin]), float(edges[hi_bin + 1])
    cluster = depth_vals[(depth_vals >= lo_val) & (depth_vals <= hi_val)]
    return cluster if cluster.size > 0 else depth_vals
