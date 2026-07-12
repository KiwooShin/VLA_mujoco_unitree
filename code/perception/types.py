"""
code/perception/types.py — shared result type for the grounding pipeline (RF-1).

Split out of the original code/grounding.py (see docs/refactor_plan.md) with
zero behavior change: `GroundingResult` is the single contract shared by the
classical HSV+depth backend (code/perception/hsv_pipeline.py) and the learned
GROUND_NET backend (code/perception/ground_net.py), and returned by
code/perception/grounding.py's `ground()` dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GroundingResult:
    """Egocentric grounding result: goal geometry plus optional diagnostics.

    `dist`/`cos_th`/`sin_th`/`confidence`/`not_visible` are always populated
    (see `ground()`'s docstring for the contract). Every other field is an
    optional diagnostic, default None, populated only by the specific code
    paths documented in each field's own inline comment below.
    """

    dist:        float         # metres to target (0 if not visible)
    cos_th:      float         # cos(yaw_err)
    sin_th:      float         # sin(yaw_err)
    confidence:  float         # [0, 1]
    not_visible: bool          # True when target not detected
    mask:        np.ndarray | None = field(default=None, repr=False)
    bbox:        tuple | None      = field(default=None, repr=False)  # (x,y,w,h)
    # NX-2 (docs/rs1_lock_mgmt.md M1): raw contour area (px^2) of the accepted blob,
    # i.e. the SAME `best_area` that feeds `conf_area` in the confidence formula
    # below. Purely additive field (default None, always passed by keyword at every
    # call site below) -- zero change to any other returned value or to any existing
    # caller that doesn't read it. Exposed because an empirical check (NX-2
    # instrumentation, see docs/nx2_impl.md) showed the bbox w*h proxy suggested in
    # the design brief is NOT a reliable stand-in for true contour area: a thin/
    # irregular sliver (the exact ep0/ep5 failure mode) can have a large bounding
    # box (low fill-ratio) while its true contour area stays tiny -- which is
    # precisely why conf_area saturates low for it in the first place. Gating on
    # this field directly (LOCK_M1) targets the root cause exactly; gating on bbox
    # w*h would not have reliably distinguished the two populations (verified
    # numerically, see docs/nx2_impl.md).
    best_area:   float | None         = field(default=None, repr=False)
    # NX-3 (docs/nx3_size_gate.md M6): the accepted blob's back-projected physical
    # width/height in metres (pinhole: pixel_extent * depth / focal -- see
    # _estimate_physical_size). Purely additive fields (default None, always passed
    # by keyword at the call sites below) -- populated whenever a blob reaches the
    # point in ground() where depth_m/bbox are known, REGARDLESS of whether LOCK_M6
    # is on (so calibration/diagnostic callers can read result.phys_w/phys_h off any
    # ordinary ground() call without needing to enable the gate or monkeypatch
    # internals -- this is what docs/nx3_size_gate.md's calibration step relies on).
    phys_w:      float | None         = field(default=None, repr=False)
    phys_h:      float | None         = field(default=None, repr=False)
    # NX-4 (docs/nx4_depth_split.md): split/re-selection diagnostics. Purely
    # additive (default None); only populated when GROUND_SPLIT=1 -- with the
    # toggle off, `ground()` never runs the split/re-selection code path at
    # all, so these stay None and there is zero added cost (same "toggle OFF
    # is provably inert" property as LOCK_M6's phys_w/phys_h fields, which
    # ARE always populated because that computation is unconditionally cheap;
    # this one is gated because the split pass itself is the thing being
    # toggled).
    n_raw_components: int | None   = field(default=None, repr=False)  # contours before split
    n_candidates:      int | None  = field(default=None, repr=False)  # candidates after split
    split_reselected:  bool | None = field(default=None, repr=False)  # winner != naive top-score pick
    size_plausible:    bool | None = field(default=None, repr=False)  # winner passed GROUND_SPLIT_SIZE band

    @property
    def goal_vec(self) -> np.ndarray:
        """Return [dist, cos_th, sin_th] as a float32 array."""
        return np.array([self.dist, self.cos_th, self.sin_th], dtype=np.float32)
