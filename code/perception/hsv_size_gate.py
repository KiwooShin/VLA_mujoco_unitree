"""
code/perception/hsv_size_gate.py — NX-3 physical-size plausibility gate (M6)
for the classical HSV+depth grounding pipeline (RF-1 split of
code/grounding.py; docs/nx3_size_gate.md).

Back-projects a candidate blob's pixel bbox extent to a physical width/height
(pinhole: real_size = pixel_size * depth / focal) and checks it against the
known nominal size for the instructed shape (code/perception/hsv_config.py's
NOMINAL_DIMS_M). Used both by the whole-blob M6 gate in
code/perception/hsv_pipeline.py and by the NX-4 split-candidate re-selection
pass in code/perception/hsv_depth_split.py (via its own independently-tuned
lo/hi band).
"""
from __future__ import annotations

from code.perception.hsv_config import M6_SIZE_BAND_HI, M6_SIZE_BAND_LO, M6_NEAR_DEPTH_M, NOMINAL_DIMS_M


def _estimate_physical_size(bbox_w_px: float, bbox_h_px: float, depth_m: float,
                            fx: float, fy: float) -> tuple:
    """
    Back-project a pixel bbox extent to physical (width_m, height_m) at the
    blob's own median depth, via the pinhole relation real_size = pixel_size *
    depth / focal_length. Exact for a fronto-parallel object centred on the
    principal axis; a reasonable plausibility-banding approximation otherwise
    (off-axis/oblique viewing shrinks the apparent size somewhat -- exactly why
    the band below is generous, not a tight tolerance). Returns (0.0, 0.0) for
    degenerate inputs (never used to silently pass a bad detection -- callers
    treat (0,0) as "could not evaluate", see _physical_size_plausible's fail-open).
    """
    if fx <= 0 or fy <= 0 or depth_m <= 0:
        return 0.0, 0.0
    return bbox_w_px * depth_m / fx, bbox_h_px * depth_m / fy


def _physical_size_plausible(bbox: tuple, depth_m: float, target_shape: str,
                             intrinsics: dict, img_w: int, img_h: int,
                             margin_l_px: int, margin_r_px: int, margin_b_px: int,
                             *, lo: float = M6_SIZE_BAND_LO, hi: float = M6_SIZE_BAND_HI
                             ) -> tuple:
    """
    Returns (plausible: bool, phys_w_m: float, phys_h_m: float).

    `lo`/`hi` default to M6's calibrated band but are overridable so the
    NX-4 split-candidate re-selection path (docs/nx4_depth_split.md) can use
    its own, independently-calibrated GROUND_SPLIT_SIZE_LO/HI band on
    depth-pure split pieces without disturbing LOCK_M6's whole-blob gate.

    Fails OPEN (plausible=True) for unknown shapes or degenerate inputs -- this
    gate should never be the reason a legitimate, recognised-shape detection is
    rejected due to a missing lookup or a zero/negative depth, only due to an
    actual measured size mismatch.

    Per-axis decision rule (Config F, calibrated in docs/nx3_size_gate.md):
      - UNCLIPPED axis: both bounds enforced (LO <= ratio <= HI).
      - CLIPPED axis (bbox touches the usable-region edge on that axis --
        left/right for width; bottom OR top for height): the LOWER bound is
        always skipped (partial visibility truncates the measured extent, e.g.
        a bottom-clipped tall cone on the proximity cam reads too SHORT -- that
        must not count against it; required by the task brief). The UPPER bound
        is still enforced when depth_m >= M6_NEAR_DEPTH_M: clipping can only
        make a blob's measured extent SMALLER than its true extent, so
        "measured > HI" is definitive evidence of an implausibly large object
        regardless of clipping. Below M6_NEAR_DEPTH_M the clipped axis is fully
        exempt (near-field depth estimates are unreliable -- see the
        M6_NEAR_DEPTH_M constant comment for the measured evidence).
    """
    x, y, w, h = bbox
    phys_w, phys_h = _estimate_physical_size(
        w, h, depth_m, intrinsics.get('fx', 0.0), intrinsics.get('fy', 0.0))

    nominal = NOMINAL_DIMS_M.get(str(target_shape).lower().strip())
    if nominal is None or phys_w <= 0.0 or phys_h <= 0.0:
        return True, phys_w, phys_h

    nominal_w, nominal_h = nominal
    touches_bottom = (y + h) >= (img_h - margin_b_px - 1)
    touches_top    = y <= 1
    touches_lr     = (x <= margin_l_px + 1) or ((x + w) >= (img_w - margin_r_px - 1))
    far            = depth_m >= M6_NEAR_DEPTH_M

    def _axis_ok(phys: float, nominal_sz: float, clipped: bool) -> bool:
        if clipped:
            return (phys <= nominal_sz * hi) if far else True
        return nominal_sz * lo <= phys <= nominal_sz * hi

    width_ok  = _axis_ok(phys_w, nominal_w, touches_lr)
    height_ok = _axis_ok(phys_h, nominal_h, touches_bottom or touches_top)
    return (width_ok and height_ok), phys_w, phys_h
