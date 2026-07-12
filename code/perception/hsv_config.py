"""
code/perception/hsv_config.py — HSV colour bounds, blob/depth thresholds, and
the NX-3/NX-4 gate toggles + calibrated constants for the classical grounding
pipeline (RF-1 split of code/grounding.py; docs/refactor_plan.md).

Pure constants + one tiny helper (`_env_flag`) -- no functions with real
logic live here (see code/perception/hsv_size_gate.py and
code/perception/hsv_depth_split.py for the gates/splitting that consume
these numbers). Every evidence comment/citation is preserved verbatim from
the original module.
"""
from __future__ import annotations

import os

import numpy as np

from code.arena import SHAPES

# ---------------------------------------------------------------------------
# HSV colour bounds  (H in [0,179], S/V in [0,255], OpenCV convention)
# ---------------------------------------------------------------------------
# Each entry: list of (lower, upper) tuples — some colours need 2 ranges (red wraps).
#
# TUNING NOTE (E6 grounding-fix):
# MuJoCo renders with ambient=0.4 lighting which desaturates colors.  A 0.4-ambient
# scene raises all channels by ~102 (0.4*255), reducing saturation significantly.
# Measured: red (220,40,40) renders as ~(234,126,126) → HSV S≈118.
#           orange (240,140,30) renders as ~(246,186,120) → HSV S≈131.
# Old S thresholds (120 for red, 140 for orange) excluded these.
# Fix: lower S floor to 60 for all colors so ambient-lit simulated objects are detected.
# The H range is kept tight to avoid cross-color confusion.
HSV_BOUNDS: dict[str, list] = {
    "red":    [
        (np.array([0,   60,  60]), np.array([12,  255, 255])),
        (np.array([165, 60,  60]), np.array([179, 255, 255])),
    ],
    "yellow": [(np.array([18,  60,  80]), np.array([40, 255, 255]))],
    "blue":   [(np.array([100,  60,  60]), np.array([135, 255, 255]))],
    "green":  [(np.array([45,   50,  50]), np.array([88,  255, 255]))],
    "orange": [(np.array([10,   60,  80]), np.array([22, 255, 255]))],
    "purple": [(np.array([125,  40,  40]), np.array([160, 255, 255]))],
    "cyan":   [(np.array([85,   60,  50]), np.array([108, 255, 255]))],
}

# ---------------------------------------------------------------------------
# Shape / blob thresholds
# ---------------------------------------------------------------------------
# V2: MIN_BLOB_AREA lowered to 40px to detect 0.48m targets at 9m at 480x360 resolution.
# At 480x360, a 0.48m ball at 9m covers ~422px² → centroid is reliable even with small mask.
# At 320x240 (native), 9m → 187px². At 480x360, 9m → 422px² (well above 40px threshold).
# After morphological open (removes < 5px radius blobs), clean small blobs remain detectable.
MIN_BLOB_AREA    = 40    # pixels — V2: lowered from 60 to handle distant targets at 9m
EROSION_ITER     = 1     # V2: reduced from 2 (distant tiny blobs erode away at iter=2)

# V5: Shape-discrimination weight — how strongly shape features influence blob selection.
# 0.0 = pure area (old behaviour); 1.0 = pure shape score.
# Set to 0.75: shape features strongly influence and will override moderate area differences.
# A 2× area blob with wrong shape (circ=0.79 vs 0.88) will still lose to correct shape.
SHAPE_WEIGHT     = 0.75  # V5: weight of shape-match score relative to area score

# V5: Two-stage shape selection parameters.
# Stage 1: when >1 blob exists, filter out blobs whose shape_score is significantly below
#   the best shape score among all candidates.
# SHAPE_REL_THRESHOLD: discard blobs with shape_score < best_shape_score * ratio.
# Set to 0.75: if the best ball score is 0.65 (real ball), discard anything < 0.49.
#   A same-color cube (ball score ≈ 0.48) will be filtered; the real ball (0.65) survives.
#   With a single blob, no filtering occurs (no best to compare against).
SHAPE_MIN_THRESHOLD = 0.45   # V5: absolute floor (used as fallback)
SHAPE_REL_THRESHOLD = 0.78   # V5: relative threshold — keep if score >= best*ratio
MAX_DEPTH_M      = 12.0  # discard depth readings beyond this (likely sky/wall)
# E6 fix v3: The MuJoCo ego camera is placed ~0.947m FORWARD of the robot origin
# (due to cam.distance=0.001 + lookat 1m ahead setup in arena._set_ego_cam).
# This means the ROBOT BODY is BEHIND the camera and does NOT appear in frame.
# Therefore, the large 50% bottom crop and 12% side margins are unnecessary and
# were actively BLOCKING valid target detections (targets at 1.5-2.5m appear in
# the lower portion of the image).
# New approach: small 5% bottom crop (for noisy edge pixels) + 3% side margins.
#
# P0 GATE FINDING (2026-07-08, docs/cam_p0.md): lowering this floor (tried 0.18m
# and 0.35m) was ALSO proposed as a P0 prerequisite fix, on the theory that 0.60m
# discarded valid near-field depth. Isolated A/B testing (full 3-skill re-eval,
# n=15 seed=999, plus single-episode causal isolation) showed:
#   - The cam.distance/CAM_ROBOT_FORWARD_OFFSET_M fix ALONE (this floor unchanged
#     at 0.60m) already gives easy 93.3%->100%, demo 60.0%->66.7%, search
#     80.0%->80.0% (zero regression) -- because recalibrating the offset from
#     0.947m->0.10m alone collapses the *effective* near-cutoff from ~1.55m to
#     ~0.7m from the robot origin, without touching this constant at all.
#   - Lowering MIN_DEPTH_M further (0.18 or 0.35, combined with the cam.distance
#     fix) added NO measurable improvement on easy/demo (identical results) but
#     caused a real regression on search (80.0%->73.3%, isolated to one episode:
#     the robot got within 0.53m then overshot/circled instead of stopping).
#     Root cause: trusting depth that close, combined with the now-correct
#     (undrifted, closer-to-body) eye position, opens a detection window into
#     the robot's own self-occlusion zone (legs/feet in frame) -- exactly the
#     risk flagged as unresolved future-work in docs/cam_opt1_widefov.md /
#     docs/cam_opt2_multicam.md ("depth-based self-body rejection" needed).
# Verdict: KEEP this constant at its original 0.60m value; the near-field win
# comes entirely from the geometry fix below, not from relaxing this floor.
MIN_DEPTH_M      = 0.60  # discard depth < 0.6m (sensor noise / very close floor)

# CAM-2 (Phase 1): the PROXIMITY camera's whole purpose is detecting targets down to
# ~0.22-0.3m (docs/cam_opt2_multicam.md geometry), so it cannot use the 0.60m floor above
# (that would blind it over its entire useful range). Used only when
# intrinsics['is_proximity'] is set (see ground() below) -- the grounding/ego cameras keep
# MIN_DEPTH_M=0.60 exactly as the P0 gate validated it.
MIN_DEPTH_PROXIMITY_M = 0.15  # just under the geometric d_near~0.22m; a loose safety net
                              # against degenerate near-zero readings, not the primary
                              # defense (see _reject_depth_outliers below for that).

# CAM-1 (Phase 2, toggle, docs/cam_opt1_widefov.md / docs/cam_p2.md): the wide-FOV
# camera's stated near-field goal is ~0.3m (task brief), same rationale as the
# proximity camera above -- used only when intrinsics['is_widefov'] is set (see
# ground() below). cam2's grounding/ego/proximity paths are unaffected (this key is
# never present in their intrinsics dicts).
MIN_DEPTH_WIDEFOV_M = 0.15    # loose safety net; same value as MIN_DEPTH_PROXIMITY_M
MIN_CONFIDENCE   = 0.05  # V2: lowered from 0.10 (distant targets produce small blobs)

# Reduced margins: robot body is NOT in camera frame (camera is 0.947m ahead of robot).
# Small margins only to handle lens distortion at extreme edges.
IMG_MARGIN_LEFT   = 0.03   # fraction of width to ignore on left/right edges
IMG_MARGIN_RIGHT  = 0.03
IMG_MARGIN_BOTTOM = 0.05   # fraction of height to ignore at bottom (noisy edge pixels)

# V2: minimum valid depth pixels (lowered for distant targets where eroded mask is small)
MIN_VALID_DEPTH_PX = 3   # was 5, reduced for 480x360 high-res render of distant blobs


# ---------------------------------------------------------------------------
# NX-3 (docs/nx3_size_gate.md): physical-size plausibility gate (M6).
# ---------------------------------------------------------------------------
# Targets are known primitives with KNOWN physical dimensions (see arena.py's
# build_arena(): every per-shape branch sets hs = obj["size"]/2 and uses hs as the
# geom's radius (ball/cylinder/cone base) or half-edge (cube) -- so SHAPES' "size"
# value IS the object's full diameter/edge-length, not a "half-size" despite the
# module docstring's naming). From a candidate blob's pixel bbox extent + its OWN
# median depth + the camera's focal length, we can back out the blob's PHYSICAL
# width/height in metres (pinhole: real_size = pixel_size * depth / focal) and
# compare against the known nominal size for the instructed shape. A false-positive
# blob at the wrong depth (ep0/ep5's sliver locked ~2m too far; ep2's confident
# wall/distractor collision; ep12's hijack) generically produces an implausible
# physical size for ITS OWN reported depth, even when its pixel-space area/fill-
# ratio/depth-sample-count look individually fine -- exactly the failure mode M1's
# conf_area-based floor cannot see, since M1 never looks at depth at all (see
# docs/rs1_lock_mgmt.md's M1 section and docs/nx2_iso.md's M1 isolation report).
#
# NOMINAL_DIMS_M: (width_m, height_m) per shape, derived EXACTLY from
# arena.build_arena()'s geom-size formulas (not just the raw SHAPES table):
#   ball/cube : symmetric, width==height==size (sphere radius=size/2 -> diameter=
#               size; cube half-edge=size/2 -> edge=size)
#   cylinder  : radius=size/2 -> diameter(width)=size; half-height=(size/2)*1.6 ->
#               height=size*1.6
#   cone      : base radius=size/2 -> diameter(width)=size; total height (base
#               cylinder + narrow tip box, see build_arena()'s "cone" branch) =
#               ((size/2)*2.2)*1.9 = size*2.09
_SHAPE_SIZE_M: dict[str, float] = dict(SHAPES)   # {"ball":0.24, "cube":0.24, "cylinder":0.22, "cone":0.26}
NOMINAL_DIMS_M: dict[str, tuple[float, float]] = {
    "ball":     (_SHAPE_SIZE_M["ball"],     _SHAPE_SIZE_M["ball"]),
    "sphere":   (_SHAPE_SIZE_M["ball"],     _SHAPE_SIZE_M["ball"]),
    "cube":     (_SHAPE_SIZE_M["cube"],     _SHAPE_SIZE_M["cube"]),
    "box":      (_SHAPE_SIZE_M["cube"],     _SHAPE_SIZE_M["cube"]),
    "cylinder": (_SHAPE_SIZE_M["cylinder"], _SHAPE_SIZE_M["cylinder"] * 1.6),
    "cone":     (_SHAPE_SIZE_M["cone"],     _SHAPE_SIZE_M["cone"] * 2.09),
}

# Plausibility band multipliers (of nominal size). CALIBRATED (docs/nx3_size_gate.md)
# against 781 instrumented accepted detections across 15 real episodes (demo passing
# eps 1/3/6/9/13 + demo failing eps 0/2/5/12 + easy eps 0-5, seed 999), classified
# true/false by |reported dist - GT dist to true target|:
#   - LO=0.08: the smallest TRUE detection observed anywhere was ratio 0.105 (easy
#     ep5's far/eroded early blobs); the initially-proposed 0.4 floor would have
#     rejected 9/15 of easy ep2's TRUE hits (a currently-100% episode). The low side
#     carries almost no discriminative power anyway (the ep0/ep2/ep5 false blobs are
#     all caught on the HIGH side) -- so it is set permissively, purely as a
#     degenerate-sliver backstop.
#   - HI=2.5: TRUE unclipped detections' width/height ratios never exceeded 1.9
#     (aggregate p95=1.4); the ep0/ep2/ep5 FALSE populations sit at 3.3-47.7x --
#     a clean >1.7x margin on both sides of 2.5.
M6_SIZE_BAND_LO = 0.08
M6_SIZE_BAND_HI = 2.5

# Near-depth stand-down for CLIPPED axes (calibration finding, docs/nx3_size_gate.md):
# during a legitimate final approach the PROXIMITY camera produces a full-frame blob
# (clipped on ALL four edges, area ~60000 px^2) whose median depth saturates ~0.97m
# while the true range keeps closing -- its back-projected "physical size" reads
# 3.5-7.6x nominal despite being the REAL target (depth corruption at extreme close
# range: surrounding-floor pixels dominating the median + the sensor depth floor).
# Observed in PASSING demo ep1 and easy eps 2/4 -- enforcing the upper bound on
# clipped axes there would break currently-100% easy episodes. Below this depth a
# clipped axis is fully exempt; at/above it, a clipped axis still gets the UPPER
# bound (clipping can only truncate the measured extent, so measured > HI implies
# true > HI regardless of clipping -- this is what catches ep2's 12.4m-wide
# LR-clipped wall blob at 8.8m and ep5's 3.3-5.1x LR-clipped blobs at 6.1m, both of
# which sail through a naive "skip clipped axes entirely" rule).
M6_NEAR_DEPTH_M = 1.2   # metres; matches inferencer.py's CAM_D_LO handoff bound


def _env_flag(name: str, default: str = "0") -> bool:
    """Return True iff env var `name` equals "1" (falling back to `default` if unset)."""
    return os.environ.get(name, default).strip() == "1"


# NX-3 M6: default OFF (opt-in via LOCK_M6=1), gated the same way as
# lock_mgmt.py's LOCK_M1..M5 toggles -- but this one lives in grounding.py, not
# lock_mgmt.py, because it is a GROUNDING-level rejection (a blob that fails this
# check never becomes a `not_visible=False` detection at all, for EITHER call site
# -- inferencer.py AND eval_search.py both get it automatically since both call
# ground()) rather than a lock-STATE-dependent mechanism. This is exactly why it
# can reach ep12's hijack: that failure enters via the mandatory CAM-2 handoff
# discontinuity carve-out in lock_mgmt.LockGate.gate_detection() (`bypass=True`),
# which no lock-state mechanism (M1/M3/M4) can veto because it fires unconditionally
# on ANY detection at that cycle -- but the hijacking blob still has to pass THIS
# check first, upstream of lock_mgmt entirely, before it ever reaches that carve-out.
LOCK_M6 = _env_flag("LOCK_M6")


# ---------------------------------------------------------------------------
# NX-4 (docs/nx4_depth_split.md): depth-guided blob splitting + component
# re-selection, and a secondary shape-check arbitration toggle.
# ---------------------------------------------------------------------------
# Root problem this targets (docs/nx3_size_gate.md §4): the HSV connected-
# component mask often MERGES the true target with an adjacent same-hue
# distractor/wall region into one 2-D-contiguous blob (e.g. ep1's cyan cube
# fused with the cyan-tinted wall into one 74x10px stripe; ep13's target ball
# fused with a wall region behind it) -- NX-3's physical-size gate could not
# separate these merged blobs from ep0/ep2/ep5's pure-false stripes because,
# once merged, they occupy the exact same implausible-size range. Splitting
# the component BEFORE size-checking (by depth continuity -- a real object's
# surface is one continuous depth cluster; a merged target+wall region is
# two, separated by the physical gap between them) fixes the false premise:
# after splitting, the true target's own sub-blob should read a plausible
# size while the merged-in wall fragment does not.
GROUND_SPLIT = _env_flag("GROUND_SPLIT")   # depth-guided blob splitting + re-selection
GROUND_SHAPE = _env_flag("GROUND_SHAPE")   # secondary: circle-fill shape arbitration (needs GROUND_SPLIT)

# Depth-histogram clustering constants for the split (docs/nx4_depth_split.md
# §1): a component's valid-depth pixels are histogrammed at GROUND_SPLIT_BIN_M
# resolution; a run of empty bins spanning >= GROUND_SPLIT_GAP_M is treated as
# a genuine physical gap between two distinct objects (task brief: 0.4-0.6m).
GROUND_SPLIT_BIN_M       = 0.15   # metres, histogram bin width
GROUND_SPLIT_GAP_M       = 0.5    # metres, minimum empty-bin run to call a split
GROUND_SPLIT_MIN_SAMPLES = 12     # minimum valid-depth pixels before attempting a split
                                   # (too few samples -> clustering is unreliable noise;
                                   # conservative no-op, matches _reject_depth_outliers'
                                   # own size<4 no-op precedent just above)
GROUND_SPLIT_MIN_PIECE_PX = 25    # minimum pixel count to keep a split-off sub-blob
                                   # (below MIN_BLOB_AREA=40 deliberately -- a genuine
                                   # small/distant true-target fragment surviving a split
                                   # should not be thrown away by a floor tuned for
                                   # whole, unsplit blobs; _score_all_contours' own
                                   # MIN_BLOB_AREA filter still applies afterwards to
                                   # each split piece's re-extracted contour)

# Re-selection size-plausibility band (docs/nx4_depth_split.md §2): applied to
# the POST-SPLIT candidate set, independently tunable from LOCK_M6's
# M6_SIZE_BAND_LO/HI (M6 gates un-split whole blobs; this band screens
# depth-pure split pieces, which -- per the design brief's own hypothesis --
# should cluster much more tightly around 1x nominal once merged-in
# distractor/wall fragments are physically separated out). Calibrated in
# docs/nx4_depth_split.md against split-candidate diagnostics on demo eps
# 0/1/2/3/5/6/9/13 + easy (same protocol as NX-3's M6 calibration).
GROUND_SPLIT_SIZE_LO = 0.08
GROUND_SPLIT_SIZE_HI = 2.5


# ---------------------------------------------------------------------------
# CAM-2 (Phase 1): depth-based self-body / near-field-artifact rejection.
# ---------------------------------------------------------------------------
# Concrete failure mode this addresses (empirically confirmed, docs/cam_p1.md): at the
# PROXIMITY camera's steep 58° pitch, once the robot is within ~0.5-0.7m of the target,
# the robot's OWN arms/hands enter the frame flanking the target (visually confirmed by
# rendering a real close-approach walk). Because the robot body is grey/low-saturation
# (g1_gear_wbc.xml rgba 0.2/0.7), it does not itself match the saturated HSV target
# palette -- but color-bleed at object silhouette edges (antialiasing blending the
# target's color with whatever is directly behind/around it, including a nearer
# occluder) can pull a MINORITY of contaminating depth samples into the target's color
# mask, well outside the plausible depth range for a single compact object. This is
# exactly the "self-occlusion" risk flagged as unresolved in docs/cam_opt1_widefov.md /
# docs/cam_opt2_multicam.md and pinned down as the P0 ep14 overshoot mechanism
# (docs/cam_p0.md): trusting depth that close, without this rejection, corrupts the
# median-depth estimate and produces jittery (dist,bearing) that can make the frozen
# policy overshoot/circle instead of stopping.
#
# Mechanism: cluster the valid depth samples into a 1-D histogram (fine bins) and keep
# only the samples in the single largest CONTIGUOUS-nonzero cluster around the modal
# bin. A real object's own surface (even a tall cone/cylinder up to ~0.7m) produces a
# continuous run of depths with no internal gaps, so it always survives as one cluster;
# a disjoint contaminating population (a hand/arm at a very different depth, or floor
# bleeding in through an anti-aliased edge) shows up as a separate cluster across an
# empty gap and gets dropped. This is a much more targeted defense than a blanket depth
# floor (which would either blind the camera over its whole useful near-field range, or
# fail to catch contamination sitting inside the "trusted" range).
DEPTH_CLUSTER_BIN_M = 0.05   # histogram bin width for the contiguous-cluster search
