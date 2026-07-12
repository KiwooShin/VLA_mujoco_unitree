"""
code/perception/hsv_pipeline.py — classical HSV colour + depth grounding
pipeline (RF-1 split of code/grounding.py; docs/refactor_plan.md).

`ground_classical()` is the byte-for-byte-preserved body of the original
`ground()`'s classical (non-GROUND_NET) code path: HSV colour threshold ->
shape/blob filter -> (optional NX-4 depth-guided split + re-selection) ->
median-depth back-projection -> egocentric (dist, yaw_err) + confidence.
Called by code/perception/grounding.py's `ground()` dispatch whenever
GROUND_NET is off, or as a graceful fallback when the learned detector's
checkpoint is unavailable.

Pipeline
--------
1. HSV colour threshold with per-colour bounds → binary mask
2. Shape/blob filter (min area, circularity) → best candidate blob
3. Morphological erosion to remove noisy border pixels
4. Median depth over eroded mask → back-project to 3-D camera-frame point
5. Rotate from camera frame to egocentric robot frame → (dist, yaw_err)
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from code.arena import backproject_pixel
from code.perception.geometry import CAM_PITCH_RAD, cam_to_egocentric
from code.perception.hsv_config import (EROSION_ITER, GROUND_SHAPE, GROUND_SPLIT,
                                        GROUND_SPLIT_SIZE_HI, GROUND_SPLIT_SIZE_LO,
                                        HSV_BOUNDS, IMG_MARGIN_BOTTOM, IMG_MARGIN_LEFT,
                                        IMG_MARGIN_RIGHT, LOCK_M6, MAX_DEPTH_M,
                                        MIN_BLOB_AREA, MIN_DEPTH_M, MIN_DEPTH_PROXIMITY_M,
                                        MIN_DEPTH_WIDEFOV_M, MIN_VALID_DEPTH_PX)
from code.perception.hsv_depth_split import (_circle_fill_score, _quick_candidate_depth,
                                             _reject_depth_outliers, _split_contours_by_depth)
from code.perception.hsv_shape_score import _score_all_contours
from code.perception.hsv_size_gate import _physical_size_plausible
from code.perception.types import GroundingResult


def ground_classical(
    ego_rgb:      np.ndarray,   # (H,W,3) uint8
    ego_depth:    np.ndarray,   # (H,W) float32, metres
    target_color: str,          # e.g. "red"
    target_shape: str,          # e.g. "ball"
    intrinsics:   dict,         # {fx,fy,cx,cy,width,height[,pitch_deg]}
    *,
    return_mask: bool = False,
) -> GroundingResult:
    """
    Detect target object and compute egocentric goal via the classical
    HSV+depth pipeline (no learned detector involved).

    Args:
        ego_rgb: BGR or RGB uint8 image (H,W,3).
        ego_depth: Depth map in metres (H,W).
        target_color: Colour name (must be in HSV_BOUNDS).
        target_shape: Shape name -- used for blob filter heuristics.
        intrinsics: Camera intrinsics dict.
        return_mask: If True, attach the binary mask to the result.

    Returns:
        GroundingResult with the detected (dist, cos_th, sin_th, confidence,
        not_visible) and, when applicable, diagnostic fields.
    """
    color_key = target_color.lower().strip()
    if color_key not in HSV_BOUNDS:
        return GroundingResult(0, 1, 0, 0.0, True)

    # CAM-2 (Phase 1): the proximity camera needs its own (lower) depth floor and its
    # own corrected un-pitch sign, and its detections get depth-outlier/self-body
    # clustering. Flag threaded through by ArenaRenderer.render_proximity()'s
    # intrinsics dict -- absent (False) for the existing grounding/ego cameras, so
    # their behaviour is completely unchanged.
    is_proximity   = bool(intrinsics.get('is_proximity', False))
    # CAM-1 (Phase 2, toggle): the wide-FOV camera gets the same near-field treatment
    # (lower depth floor + self-body depth-outlier rejection + corrected un-pitch sign)
    # as the proximity camera, for the same reason -- see ArenaRenderer.render_widefov().
    # Absent (False) whenever cam2's cameras are used, so their behaviour is unchanged.
    is_widefov     = bool(intrinsics.get('is_widefov', False))
    if is_proximity:
        min_depth_eff = MIN_DEPTH_PROXIMITY_M
    elif is_widefov:
        min_depth_eff = MIN_DEPTH_WIDEFOV_M
    else:
        min_depth_eff = MIN_DEPTH_M

    # ---- Colour segmentation ----
    # Convert to HSV (OpenCV expects BGR input)
    if ego_rgb.shape[2] == 3:
        bgr = cv2.cvtColor(ego_rgb, cv2.COLOR_RGB2BGR)
    else:
        bgr = ego_rgb
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_BOUNDS[color_key]:
        mask |= cv2.inRange(hsv, lo, hi)

    # ---- E6 fix: mask out robot-body self-occlusion region ----
    # The ego camera sees the robot's arms/torso in the lower portion and side edges.
    # Zero out those image regions to prevent false self-detections.
    h_img, w_img = mask.shape[:2]
    l_px = int(IMG_MARGIN_LEFT   * w_img)
    r_px = int(IMG_MARGIN_RIGHT  * w_img)
    b_px = int(IMG_MARGIN_BOTTOM * h_img)
    if l_px > 0:
        mask[:, :l_px] = 0
    if r_px > 0:
        mask[:, w_img - r_px:] = 0
    if b_px > 0:
        mask[h_img - b_px:, :] = 0

    # ---- Shape / blob filter ----
    # V2: use smaller kernel at higher resolutions to avoid erasing tiny distant blobs.
    # At 480x360, a 5x5 kernel covers the same angular area as a ~3x3 at 320x240.
    # Scale kernel with image area relative to native 320x240.
    _scale = math.sqrt((h_img * w_img) / (240.0 * 320.0))
    _k = max(3, int(round(5 * _scale)) | 1)  # ensure odd; min 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_k, _k))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return GroundingResult(0, 1, 0, 0.0, True,
                               mask=(mask if return_mask else None))

    # ---- NX-4 (docs/nx4_depth_split.md): depth-guided blob splitting ----
    # Expand each connected HSV component into depth-consistent sub-components
    # BEFORE scoring/selection, so a merged true-target+wall stripe becomes two
    # separate candidates instead of one implausibly-shaped/sized blob. No-op
    # (contours unchanged) for every component that is already depth-pure, and
    # entirely inert when the toggle is off.
    n_raw_components = len(contours)
    if GROUND_SPLIT:
        contours = _split_contours_by_depth(contours, mask.shape, ego_depth,
                                            min_depth_eff, MAX_DEPTH_M)
        if not contours:
            return GroundingResult(0, 1, 0, 0.0, True,
                                   mask=(mask if return_mask else None))

    # ---- Select best blob (V5: shape-discriminating composite score) ----
    # When multiple same-color blobs exist (distractors), we pick the one whose
    # contour geometry best matches the instructed shape (ball/cube/cylinder/cone).
    # We normalise area across all candidates so a 2-3× larger distractor cannot
    # overwhelm the shape signal (SHAPE_WEIGHT=0.75 gives shape the majority vote).
    scored_contours = _score_all_contours(contours, target_shape)

    if not scored_contours:
        return GroundingResult(0, 1, 0, 0.0, True,
                               mask=(mask if return_mask else None))

    # ---- NX-4 §2: component re-selection on the (possibly split) candidate
    # set -- prefer size-plausible candidates over implausible ones; among
    # plausible candidates (or when nothing is plausible / toggle is off),
    # keep the existing composite-score ranking unchanged.
    chosen_idx        = 0
    _winning_blob_mask = None
    split_reselected  = False
    n_candidates      = len(scored_contours)
    winner_plausible  = None
    if GROUND_SPLIT:
        evaluated = []
        for idx, (cscore, ccnt) in enumerate(scored_contours):
            carea = float(cv2.contourArea(ccnt))
            if carea < MIN_BLOB_AREA:
                continue
            cx0, cy0, cw0, ch0 = cv2.boundingRect(ccnt)
            cdepth, cmask = _quick_candidate_depth(ccnt, mask.shape, ego_depth,
                                                    min_depth_eff, MAX_DEPTH_M)
            if cdepth is None:
                cplausible, cphys_w, cphys_h = False, 0.0, 0.0
            else:
                cplausible, cphys_w, cphys_h = _physical_size_plausible(
                    (cx0, cy0, cw0, ch0), cdepth, target_shape, intrinsics, w_img, h_img,
                    l_px, r_px, b_px, lo=GROUND_SPLIT_SIZE_LO, hi=GROUND_SPLIT_SIZE_HI)
            evaluated.append(dict(idx=idx, score=cscore, cnt=ccnt, bbox=(cx0, cy0, cw0, ch0),
                                  plausible=cplausible, blob_mask=cmask))
        if evaluated:
            # GROUND_SHAPE (§3): arbitrate a same-color ball-vs-cube twin ONLY
            # when >=2 candidates already passed the size-plausibility band --
            # never used to reject the sole candidate on shape alone.
            if GROUND_SHAPE:
                plausible_run = [e for e in evaluated if e['plausible']]
                shape_key = str(target_shape).lower().strip()
                if len(plausible_run) >= 2 and shape_key in ("ball", "sphere", "cube", "box"):
                    target_fill = 0.90 if shape_key in ("ball", "sphere") else 0.68
                    def _fill_dist(e: dict) -> float:
                        return abs(_circle_fill_score(e['cnt']) - target_fill)
                    best_shape_pick = min(plausible_run, key=lambda e: (_fill_dist(e), -e['score']))
                    evaluated = [best_shape_pick] + [e for e in evaluated if e is not best_shape_pick]

            evaluated.sort(key=lambda e: (not e['plausible'], -e['score']))
            winner = evaluated[0]
            chosen_idx         = winner['idx']
            _winning_blob_mask = winner['blob_mask']
            winner_plausible   = winner['plausible']
            split_reselected   = (chosen_idx != 0)

    best_score, best_cnt = scored_contours[chosen_idx]
    best_area = float(cv2.contourArea(best_cnt))

    if best_cnt is None or best_area < MIN_BLOB_AREA:
        return GroundingResult(0, 1, 0, 0.0, True,
                               mask=(mask if return_mask else None))

    # ---- Blob mask and bbox ----
    if _winning_blob_mask is not None:
        blob_mask = _winning_blob_mask
    else:
        blob_mask = np.zeros_like(mask)
        cv2.drawContours(blob_mask, [best_cnt], -1, 255, cv2.FILLED)

    x, y, w, h = cv2.boundingRect(best_cnt)
    cx_px = x + w / 2.0
    cy_px = y + h / 2.0

    # ---- Erode mask before depth sampling ----
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded  = cv2.erode(blob_mask, erode_k, iterations=EROSION_ITER)

    # ---- Depth estimation ----
    depth_vals = ego_depth[eroded > 0]
    valid      = depth_vals[(depth_vals > min_depth_eff) & (depth_vals < MAX_DEPTH_M)]
    if is_proximity or is_widefov:
        # CAM-2/CAM-1 self-body / near-field-artifact rejection (see module comment
        # above _reject_depth_outliers): drop a disjoint contaminating depth population
        # (robot's own arm/hand, or an anti-aliased edge-bleed pixel) before the
        # median is computed, rather than hoping the median alone is robust to it.
        valid = _reject_depth_outliers(valid)

    if len(valid) < MIN_VALID_DEPTH_PX:
        # E6 fix: if depth is insufficient (all pixels within robot-body range or
        # too sparse), treat as NOT visible — a spurious color blob at robot-body
        # depth should not be used as a goal direction.
        # V2: lowered from 5 to MIN_VALID_DEPTH_PX=3 for distant tiny blobs.
        return GroundingResult(0, 1, 0, 0.0, True,
                               mask=(blob_mask if return_mask else None),
                               bbox=None)

    # ---- Foreground cluster filter ----
    # E6 fix v3: detect and reject background wall false positives.
    # A blob that is very large (>30% of valid image area) AND spans a wide depth
    # range (>0.8m) is likely the arena wall/floor being mistakenly detected, not
    # the actual target object. Real objects are compact and at a consistent depth.
    valid_image_area = (h_img - int(IMG_MARGIN_BOTTOM * h_img)) * (w_img - int((IMG_MARGIN_LEFT + IMG_MARGIN_RIGHT) * w_img))
    blob_fill_ratio = best_area / max(1, valid_image_area)
    depth_range = float(np.percentile(valid, 90)) - float(np.percentile(valid, 10))

    # Check if blob is a foreground object (compact & consistent depth) or background
    is_background_blob = (blob_fill_ratio > 0.30 and depth_range > 0.7)
    if is_background_blob:
        # V3 DEPTH-FG RESCUE: when the whole image is cyan/blue-tinted (e.g. arena
        # walls render at H≈104-105 which overlaps cyan/blue HSV bounds), the HSV
        # mask covers 80%+ of the frame and gets correctly rejected by the background
        # filter above.  But the ACTUAL object is still present — it's just a compact
        # foreground cluster sitting in front of the cyan-tinted background.
        #
        # Fix: re-run blob detection using ONLY pixels that are significantly CLOSER
        # than their local neighbourhood (depth-gradient foreground mask).  This
        # separates the target object (at its true depth) from the arena surfaces
        # (at the peripheral depth).  We use a uniform filter as a fast local-mean
        # estimate (scipy not required — we use cv2.blur for speed).
        #
        # Threshold: pixel is foreground if depth < local_mean - FG_DEPTH_MARGIN.
        # FG_DEPTH_MARGIN=0.15m: objects 15 cm closer than background are selected.
        # This safely handles targets at 4-9m against walls at 5-11m.
        FG_DEPTH_MARGIN = 0.15   # metres — foreground/background separation
        FG_BLUR_K = 51           # pixels — neighbourhood size for local mean (must be odd)
        depth_f = ego_depth.copy()
        depth_f[depth_f > MAX_DEPTH_M] = MAX_DEPTH_M  # cap far values
        local_mean = cv2.blur(depth_f, (FG_BLUR_K, FG_BLUR_K))
        fg_depth_mask = (ego_depth < local_mean - FG_DEPTH_MARGIN)

        # Intersect HSV mask (pre-morphology) with foreground depth mask
        # Rebuild the raw HSV mask (before morphology) intersected with depth-fg
        raw_hsv = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo_b, hi_b in HSV_BOUNDS[color_key]:
            raw_hsv |= cv2.inRange(hsv, lo_b, hi_b)
        # Apply margins
        if l_px > 0: raw_hsv[:, :l_px] = 0
        if r_px > 0: raw_hsv[:, w_img - r_px:] = 0
        if b_px > 0: raw_hsv[h_img - b_px:, :] = 0

        fg_mask_u8 = (fg_depth_mask & (raw_hsv > 0)).astype(np.uint8) * 255

        # Morphological cleanup on the FG mask
        fg_mask_u8 = cv2.morphologyEx(fg_mask_u8, cv2.MORPH_OPEN,  kernel, iterations=1)
        fg_mask_u8 = cv2.morphologyEx(fg_mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

        fg_contours, _ = cv2.findContours(fg_mask_u8, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

        # NX-4 (docs/nx4_depth_split.md): split each FG-rescue candidate by depth
        # BEFORE the aspect-ratio pre-filter. The FG-rescue mask (pixels closer
        # than their local blurred-depth neighbourhood) is exactly where NX-3's
        # merged true-target+wall-fragment stripes come from -- a "relatively
        # foreground" pixel population can span both the real object AND an
        # adjacent wall seam/corner that also happens to read locally-closer.
        # Splitting first lets a genuinely compact true-target sub-piece survive
        # even when the raw FG contour was fused with a wider neighbour.
        if GROUND_SPLIT and fg_contours:
            fg_contours = _split_contours_by_depth(list(fg_contours), mask.shape, ego_depth,
                                                    min_depth_eff, MAX_DEPTH_M)

        # Pre-filter: reject wall-stripe artifacts (wide+shallow → aspect > 8.0).
        valid_fg = []
        for cnt in fg_contours:
            a = float(cv2.contourArea(cnt))
            if a < MIN_BLOB_AREA:
                continue
            _bx, _by, _bw, _bh = cv2.boundingRect(cnt)
            if _bw / max(1, _bh) > 8.0:
                continue
            valid_fg.append(cnt)
        # V5 NOTE: FG rescue uses AREA-ONLY selection (no shape discrimination).
        # Reason: in FG rescue scenarios (cyan/blue wall overlap), the target may be
        # MUCH farther than the distractor. Filtering distractors by shape in FG rescue
        # often causes the ONLY detected FG blob to be discarded (the close distractor
        # is filtered but the far target is not in the FG mask at all), resulting in
        # not_visible and robot drift. The primary path (above) handles shape discrimination
        # for non-wall-overlap multi-blob cases. FG rescue falls back to largest-area
        # (V4 behavior) to maximize detection reliability.
        # NX-4: when GROUND_SPLIT is on and there is an actual choice to make
        # (>1 survivor), prefer a size-plausible candidate over an implausible
        # one before falling back to the original area-only tie-break -- still
        # pure area-only (byte-identical to legacy) whenever the toggle is off
        # or there is only one candidate (never risks discarding the sole
        # detection, matching the brief's conservative-arbitration rule).
        fg_best_cnt  = None
        fg_best_area = 0.0
        fg_best_score = 0.0
        if GROUND_SPLIT and len(valid_fg) > 1:
            fg_evaluated = []
            for cnt in valid_fg:
                a = float(cv2.contourArea(cnt))
                fx0, fy0, fw0, fh0 = cv2.boundingRect(cnt)
                fdepth, _fmask = _quick_candidate_depth(cnt, mask.shape, ego_depth,
                                                        min_depth_eff, MAX_DEPTH_M)
                if fdepth is None:
                    fplausible = False
                else:
                    fplausible, _, _ = _physical_size_plausible(
                        (fx0, fy0, fw0, fh0), fdepth, target_shape, intrinsics, w_img, h_img,
                        l_px, r_px, b_px, lo=GROUND_SPLIT_SIZE_LO, hi=GROUND_SPLIT_SIZE_HI)
                fg_evaluated.append(dict(cnt=cnt, area=a, plausible=fplausible))
            fg_evaluated.sort(key=lambda e: (not e['plausible'], -e['area']))
            fg_best_cnt  = fg_evaluated[0]['cnt']
            fg_best_area = fg_evaluated[0]['area']
        else:
            for cnt in valid_fg:
                a = float(cv2.contourArea(cnt))
                if a > fg_best_score:
                    fg_best_score = a
                    fg_best_area = a
                    fg_best_cnt = cnt

        if fg_best_cnt is None:
            # Depth-FG rescue also failed — truly not visible
            return GroundingResult(0, 1, 0, 0.0, True,
                                   mask=(blob_mask if return_mask else None),
                                   bbox=None)

        # Use the depth-FG blob as the detection
        blob_mask = np.zeros_like(mask)
        cv2.drawContours(blob_mask, [fg_best_cnt], -1, 255, cv2.FILLED)
        x, y, w, h = cv2.boundingRect(fg_best_cnt)
        cx_px    = x + w / 2.0
        cy_px    = y + h / 2.0
        best_area = fg_best_area

        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded  = cv2.erode(blob_mask, erode_k, iterations=EROSION_ITER)
        depth_vals = ego_depth[eroded > 0]
        valid      = depth_vals[(depth_vals > min_depth_eff) & (depth_vals < MAX_DEPTH_M)]
        if is_proximity or is_widefov:
            valid = _reject_depth_outliers(valid)

        if len(valid) < MIN_VALID_DEPTH_PX:
            return GroundingResult(0, 1, 0, 0.0, True,
                                   mask=(blob_mask if return_mask else None),
                                   bbox=None)

        # FG rescue background check: the wall-edge artifact appears as a wide, shallow
        # stripe (large width, small height → large aspect ratio).
        # Real target objects are compact: balls ≈ circular, cubes ≈ square, cones/cylinders
        # are tall. Maximum expected aspect ratio for a real object: ~3-4.
        # Wall-edge artifacts: width >> height (aspect ratio 10-30).
        # Also apply the fill-ratio AND depth-range filter from the main loop.
        fg_fill_ratio  = fg_best_area / max(1, valid_image_area)
        fg_depth_range = (float(np.percentile(valid, 90)) - float(np.percentile(valid, 10))
                          if len(valid) > 10 else 0.0)
        # Aspect ratio: w = bbox width, h = bbox height (from cv2.boundingRect(fg_best_cnt))
        # Note: 'w' and 'h' were set by the cv2.boundingRect(fg_best_cnt) call above.
        fg_bbox_aspect = w / max(1, h)   # width / height of FG blob bounding box
        # Reject if: fill-ratio+depth-range fail OR blob is extremely wide+shallow (wall stripe)
        fg_is_background = ((fg_fill_ratio > 0.30 and fg_depth_range > 0.7) or
                            fg_bbox_aspect > 8.0)
        if fg_is_background:
            return GroundingResult(0, 1, 0, 0.0, True,
                                   mask=(blob_mask if return_mask else None),
                                   bbox=None)

    depth_m = float(np.median(valid))

    # ---- NX-3 M6: physical-size plausibility gate (docs/nx3_size_gate.md) ----
    # Computed regardless of LOCK_M6 (see GroundingResult.phys_w/phys_h docstring
    # above -- lets calibration/diagnostic callers read the estimate off any
    # ordinary ground() call). Only actually REJECTS the detection when LOCK_M6=1.
    _m6_plausible, _phys_w, _phys_h = _physical_size_plausible(
        (x, y, w, h), depth_m, target_shape, intrinsics, w_img, h_img,
        l_px, r_px, b_px)
    if LOCK_M6 and not _m6_plausible:
        return GroundingResult(0, 1, 0, 0.0, True,
                               mask=(blob_mask if return_mask else None),
                               bbox=(x, y, w, h),
                               best_area=best_area,
                               phys_w=_phys_w, phys_h=_phys_h)

    # ---- Back-project centroid to 3-D ----
    pt3d    = backproject_pixel(cx_px, cy_px, depth_m, intrinsics)
    # V2: if intrinsics contains 'pitch_deg' (from grounding render), use it.
    # The grounding render uses GROUNDING_PITCH=20° (not 32°), so cam_to_egocentric
    # must un-rotate with the same pitch to get correct bearing and distance.
    pitch_deg_for_unrot = float(intrinsics.get('pitch_deg', CAM_PITCH_RAD * 180.0 / math.pi))
    # CAM-1 (Phase 2, toggle): the widefov camera's pitch (WIDEFOV_PITCH=42 deg) is
    # steep enough that the pre-existing un-pitch sign bug (docs/cam_p1.md) is a real
    # risk (it was fatal at PROXIMITY_PITCH=58 deg, small-but-present at 26/32 deg) --
    # empirically verified monotonic/accurate via a distance-sweep bench before this
    # was locked in (see docs/cam_p2.md, bench_widefov_dist.py).
    dist, yaw_err = cam_to_egocentric(pt3d[0], pt3d[1], pt3d[2], pitch_deg=pitch_deg_for_unrot,
                                      use_corrected_unpitch=(is_proximity or is_widefov))

    dist = max(0.0, dist)

    # ---- Confidence ----
    max_area     = float(ego_rgb.shape[0] * ego_rgb.shape[1])
    conf_area    = min(1.0, best_area / (max_area * 0.5))   # saturates at 50% fill
    conf_depth   = 1.0 if len(valid) > 20 else len(valid) / 20.0
    confidence   = float(0.6 * conf_area + 0.4 * conf_depth)
    confidence   = min(1.0, confidence)

    return GroundingResult(
        dist        = dist,
        cos_th      = math.cos(yaw_err),
        sin_th      = math.sin(yaw_err),
        confidence  = confidence,
        not_visible = False,
        mask        = blob_mask if return_mask else None,
        bbox        = (x, y, w, h),
        best_area   = best_area,
        phys_w      = _phys_w,
        phys_h      = _phys_h,
        n_raw_components = n_raw_components if GROUND_SPLIT else None,
        n_candidates      = n_candidates if GROUND_SPLIT else None,
        split_reselected  = split_reselected if GROUND_SPLIT else None,
        size_plausible    = winner_plausible if GROUND_SPLIT else None,
    )
