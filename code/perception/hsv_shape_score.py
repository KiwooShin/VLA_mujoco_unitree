"""
code/perception/hsv_shape_score.py — V5 shape-discrimination blob scoring for
the classical HSV+depth grounding pipeline (RF-1 split of code/grounding.py).

`_score_all_contours` is the entry point used by
code/perception/hsv_pipeline.py: a two-stage shape-discriminating selection
over the candidate contour set (shape-match filtering, then a composite
area+shape score).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from code.perception.hsv_config import (MIN_BLOB_AREA, SHAPE_MIN_THRESHOLD,
                                        SHAPE_REL_THRESHOLD, SHAPE_WEIGHT)


def _shape_match_score(cnt: np.ndarray, target_shape: str) -> float:
    """
    Compute a [0, 1] score measuring how well contour `cnt` matches `target_shape`.

    Features used per shape:
      ball/sphere  → high circularity (4π·A/P²)
      cube/box     → high convex solidity + aspect ratio near 1 (square bbox)
      cylinder     → high convex solidity + aspect ratio > 1 (taller than wide)
      cone         → triangular profile: centroid y < bbox center y (tapered top)
                     + low circularity (triangle ≠ circle)

    Returns 0.0 if contour too small to score reliably.
    """
    area = float(cv2.contourArea(cnt))
    if area < 1.0:
        return 0.0

    peri = cv2.arcLength(cnt, True)
    circ = 4.0 * math.pi * area / (peri * peri + 1e-6)  # 1=circle, 0=line

    # Convex hull solidity: area / convex_hull_area (1=convex, <1=concave)
    hull     = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / max(1.0, hull_area)

    # Bounding box aspect: h/w (>1 means taller than wide)
    bx, by, bw, bh = cv2.boundingRect(cnt)
    aspect_hw = bh / max(1, bw)   # height/width ratio

    # Centroid position relative to bbox (for cone taper detection)
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cy_cnt = M["m01"] / M["m00"]  # centroid y in image (row, increases downward)
    else:
        cy_cnt = by + bh / 2.0
    bbox_center_y = by + bh / 2.0
    # Cone: wider at base (bottom), narrow at top.
    # In image coords, top = small row, bottom = large row.
    # For a cone the centroid is BELOW the bbox center (more area in the base).
    # centroid_above = cy_cnt < bbox_center_y  → top-heavy (anti-cone)
    # centroid_below = cy_cnt > bbox_center_y  → bottom-heavy (cone-like)
    centroid_frac = (cy_cnt - bbox_center_y) / max(1, bh)  # -0.5..+0.5; +ve = below center

    shape_key = target_shape.lower()

    # Bounding-box fill ratio: area / (bbox_w * bbox_h)
    # Circle: fill≈0.74; Square: fill≈0.97; Cylinder/Rectangle: fill≈0.95; Triangle≈0.47
    bbox_fill = area / max(1.0, bw * bh)

    if shape_key in ("ball", "sphere"):
        # Ball: high circularity AND low bbox_fill (circle doesn't fill its bbox).
        # circ^3 amplifies the gap: circle 0.88→0.68, square 0.79→0.49.
        # bbox_fill: circle≈0.74, square≈0.97 → complement: circle=0.26, square=0.03.
        # Combined score emphasizes both features.
        low_fill_score = max(0.0, 1.0 - bbox_fill) * 3.0  # 0.78 for circle, 0.09 for square; cap at 1
        low_fill_score = min(1.0, low_fill_score)
        score = 0.5 * (circ ** 3) + 0.5 * low_fill_score

    elif shape_key in ("cube", "box"):
        # Cube: high bbox_fill (square fills bbox ~97%) + aspect ≈ 1.
        # Key discriminator: bbox_fill^3 ≈ 0.91 for cube vs 0.40 for ball.
        # The ^3 exponent dramatically amplifies the fill-ratio gap.
        # Cylinder also has high fill but tall aspect (hw>1.5), which is penalized.
        fill_score  = bbox_fill ** 3   # 0.91 for square, 0.40 for circle, 0.86 for rect
        aspect_score = 1.0 - abs(aspect_hw - 1.0) / 2.0  # 1 at aspect=1, 0 at aspect=3
        aspect_score = max(0.0, aspect_score)
        score = 0.65 * fill_score + 0.35 * aspect_score

    elif shape_key in ("cylinder",):
        # Cylinder: convex + taller than wide (aspect_hw > 1 when viewed from side).
        # Solidity should be high (rectangular cross-section = no concavities).
        # Key discriminator vs cube: aspect_hw >> 1 (cylinder ~2-4, cube ~1).
        # We strongly reward aspect > 1.5 (clearly taller than wide).
        # Note: from directly above a cylinder looks circular, but demo scenes are side-on.
        taller_score = min(1.0, max(0.0, (aspect_hw - 1.0) / 2.0))  # 0 at aspect=1, 1 at aspect=3
        score = 0.4 * solidity + 0.6 * taller_score

    elif shape_key in ("cone",):
        # Cone: triangular profile — tapered top, wider base.
        # Centroid below bbox center (centroid_frac > 0, ideally +0.1 to +0.2 for triangle).
        # Circularity low (triangle << 1).
        taper_score = max(0.0, min(1.0, centroid_frac * 5.0 + 0.5))  # 0 if top-heavy, 1 if bottom-heavy
        non_circle  = 1.0 - circ   # high for triangle/square, 0 for perfect circle
        score = 0.5 * taper_score + 0.5 * non_circle

    else:
        # Unknown shape — neutral score
        score = 0.5

    return float(max(0.0, min(1.0, score)))


def _blob_composite_score(cnt: np.ndarray, target_shape: str,
                          shape_weight: float = SHAPE_WEIGHT,
                          max_area: float = 1.0) -> float:
    """
    Composite score combining relative area and shape match.

    score = shape_weight * shape_match + (1-shape_weight) * norm_area

    norm_area = sqrt(area / max_area) in [0,1] — normalised across the candidate set.
    Call _score_all_contours() instead of this directly for proper normalisation.
    """
    area        = float(cv2.contourArea(cnt))
    norm_area   = math.sqrt(area / max(1.0, max_area))   # [0,1]
    shape_score = _shape_match_score(cnt, target_shape)
    return shape_weight * shape_score + (1.0 - shape_weight) * norm_area


def _score_all_contours(contours: list, target_shape: str,
                        shape_weight: float = SHAPE_WEIGHT,
                        shape_min_thresh: float = SHAPE_MIN_THRESHOLD,
                        shape_rel_thresh: float = SHAPE_REL_THRESHOLD) -> list:
    """
    Two-stage shape-discriminating blob selection.

    Stage 1 (multi-blob filtering): when >1 valid candidate exists:
      - Compute shape_score for all candidates.
      - Find best_shape_score (max across all).
      - Keep only candidates with shape_score >= max(shape_min_thresh,
          best_shape_score * shape_rel_thresh).
      - This filters distractors with clearly wrong shape even if they are
        much larger (closer) than the target.
      - Soft fallback: if ALL candidates fail the filter, keep all (return
        composite-score ranking, no hard veto).

    Stage 2: rank survivors by composite score (normalised area + shape match).

    Returns list of (score, cnt) sorted descending by score.
    """
    if not contours:
        return []
    # Filter by min area
    valid = [(float(cv2.contourArea(c)), c) for c in contours
             if float(cv2.contourArea(c)) >= MIN_BLOB_AREA]
    if not valid:
        return []

    # Compute shape scores for all
    shape_scores = [(_shape_match_score(c, target_shape), a, c) for a, c in valid]

    # Stage 1: multi-blob relative-threshold filtering
    if len(shape_scores) > 1:
        best_ss = max(ss for ss, _, _ in shape_scores)
        # Dynamic threshold: keep if within rel_thresh of the best candidate's shape
        cutoff  = max(shape_min_thresh, best_ss * shape_rel_thresh)
        filtered = [(ss, a, c) for ss, a, c in shape_scores if ss >= cutoff]
        if filtered:
            shape_scores = filtered
        # else: all below cutoff → keep all (soft fallback)

    # Stage 2: composite score with normalised area
    max_a = max(a for _, a, _ in shape_scores)
    scored = []
    for ss, area, cnt in shape_scores:
        norm_a = math.sqrt(area / max(1.0, max_a))
        comp = shape_weight * ss + (1.0 - shape_weight) * norm_a
        scored.append((comp, cnt))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
