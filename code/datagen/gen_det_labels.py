"""
code/datagen/gen_det_labels.py — Segmentation -> per-object detection labels.

Role: split out of gen_det_dataset.py (RF-1) — the label-derivation logic
(pure functions of arrays/dicts, no mujoco rendering), shared by
gen_det_dataset.py's synthetic generator and gen_det_failcases.py's
live-replay instrumentation (via the old-path alias).

Contents:
  - build_id_to_obj      — geom-id -> object-index lookup array.
  - seg_to_objmap        — segmentation map -> per-pixel object-index map.
  - derive_object_labels — per-object label dicts from an object-index map.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from code.arena import GROUNDING_PITCH, backproject_pixel
from code.datagen.gen_det_common import GEOM_RE, MIN_PIXELS, SHAPE2I, COLOR2I, SIZE_M
from code.grounding import cam_to_egocentric
from code.steer import _angle_diff, egocentric_goal


# ---------------------------------------------------------------------------
# Geom-id -> object-index map (handles the cone's extra "_tip" geom)
# ---------------------------------------------------------------------------
def build_id_to_obj(model: mujoco.MjModel, n_objects: int) -> np.ndarray:
    """Builds a geom-id -> object-index lookup array.

    Handles the cone's extra "_tip" geom by mapping it to the same object
    index as its base geom.

    Args:
        model: Compiled MuJoCo model (from `build_arena`).
        n_objects: Number of scene objects to map (geoms named "obj_{i}"
            with i >= n_objects are ignored).

    Returns:
        (model.ngeom,) int32 array mapping geom id to object index, or -1
        for geoms that are not scene objects.
    """
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
    """Converts a segmentation map to a per-pixel object-index map.

    Args:
        seg: (H,W,2) int32 array from enable_segmentation_rendering();
            channel 0 holds the geom id.
        id_to_obj: Geom-id -> object-index lookup array, from
            `build_id_to_obj`.

    Returns:
        (H,W) int32 array of object indices, or -1 where no known scene
        object is present.
    """
    inst = seg[..., 0]
    valid = inst >= 0
    idx = np.where(valid, inst, 0)
    idx = np.clip(idx, 0, id_to_obj.shape[0] - 1)
    obj_map = np.where(valid, id_to_obj[idx], -1)
    return obj_map


# ---------------------------------------------------------------------------
# Per-object label derivation from a segmentation-derived object-index map.
# Standalone (no renderer dependency) so both gen_det_dataset.py's synthetic
# generator AND gen_det_failcases.py's live-replay instrumentation can share
# the exact same labeling logic.
# ---------------------------------------------------------------------------
def derive_object_labels(rgb: np.ndarray, depth: np.ndarray, obj_map: np.ndarray,
                         objects: list, robot_xy: np.ndarray, robot_yaw: float,
                         intr: dict) -> list:
    """Derives per-object detection labels from a segmentation-based object map.

    For each object with a foreground mask, computes the pixel bounding
    box/centroid/area, whether the mask is clipped by an image edge, the
    median depth, the ground-truth egocentric (distance, bearing), and the
    back-projected (distance, bearing) computed the same way the deployed
    grounder does — the gap between the two is the label-geometry sanity
    check.

    Args:
        rgb: (H,W,3) uint8 rendered RGB image; only its (H, W) shape is
            used here.
        depth: (H,W) float depth image, in meters.
        obj_map: (H,W) int32 object-index map from `seg_to_objmap` (-1
            where no object is present).
        objects: List of object dicts from the scene config (x, y,
            shape_name, color_name, ...).
        robot_xy: (2,) robot world xy position.
        robot_yaw: Robot yaw, in radians.
        intr: Camera intrinsics dict used for back-projection.

    Returns:
        A list of per-object label dicts (one per visible object with
        enough mask pixels), with keys obj_idx, class_name, class_id,
        color_name, color_id, bbox_x/y/w/h, centroid_px_x/y, area_px,
        clipped, depth_median_m, dist_gt_m, bearing_gt_deg, dist_bp_m,
        bearing_bp_deg, err_dist_m, err_bearing_deg.
    """
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
