"""
code.runtime.goal_config — tunable constants for `code.runtime.goal_pipeline`.

RF-1 split of code/inferencer.py (docs/refactor_plan.md): pulled out of
goal_pipeline.py purely to keep that file under the <500-line budget — these
constants (and their evidence comments) are otherwise unchanged from the
original flat file.
"""

from __future__ import annotations

from code.sim.teacher import SIM_DT, CONTROL_DECIMATION

# ---------------------------------------------------------------------------
# Goal smoothing constants (E6)
# ---------------------------------------------------------------------------
GOAL_EMA_ALPHA  = 0.4       # blending factor: new=alpha*detected + (1-alpha)*ema
# V2: HOLD_GOAL_HORIZON extended to 100 steps (was 50).
# Progressive re-detection: robot walks toward last-known goal for up to 100 steps,
# then re-detects at closer range. At 4-9m, after walking 1-2m closer, the target
# is 2-3m nearer → 40-80% larger in the image → much more reliable detection.
# Key: 100 steps * 0.02s * 0.55m/s MAX_VX ≈ 1.1m forward progress during hold.
HOLD_GOAL_HORIZON = 100      # V2: extended from 50 — progressive re-detection window

# ---------------------------------------------------------------------------
# H3 scan-and-acquire schedule (NX-1/NX-10, docs/nx10_scan_fix.md)
# ---------------------------------------------------------------------------
# E6 fix: do NOT default to straight-ahead.  Instead, start in SCAN mode:
# the robot turns in place (pure ωz) until the target is detected AND centered
# (detected bearing < SCAN_ALIGNED_THR), or until SCAN_TIMEOUT steps elapsed.
# This ensures the first committed goal is based on an unoccluded frontal detection.
#
# Design: scan RIGHT for first 60 steps, then LEFT for next 60 (120 total).
# Acquired when:  target detected AND abs(bearing) < SCAN_ALIGNED_THR (target near-center).
# Partial detections (target partially occluded, large bearing) keep scanning.
#
# NX-10 (docs/fa2_residuals.md, docs/nx10_scan_fix.md): the old H3 scan assumed the
# commanded SCAN_RATE was fully realized (step_count * rate * dt) to bound a ±90°
# right/left/right sweep (75/125/0 step split over a 200-step budget) -- but the
# student-driven turn only realizes a fraction of the commanded rate in practice, so
# the fixed step budget only ever swept a REALIZED ~-61°/+64° arc (confirmed by
# instrumented replay), not the intended ±90°, leaving demo ep2's target (bearing
# -73.8°) structurally unreachable by the scan regardless of detector quality --
# 0/140 raw detector calls ever saw the target in frame. Fix: reuse NX-1's
# BidirectionalScanSchedule (code/scan_sched.py) -- the SAME already-validated shared
# CLASS eval_search.py/fancy_demo.py use -- which tracks the robot's ACTUAL
# accumulated yaw (integrated from real per-step yaw readings, not assumed from
# step*rate) so each leg always completes its full REAL angular sweep regardless of
# realized-rate drift, self-correcting exactly like the search-skill fix did for its
# own rotation-coverage bug.
#
# `H3_LEG_DEG` is deliberately NOT eval_search's own `SCAN_LEG_DEG` (=165, code/
# scan_sched.py) -- the dwell length `_H3_DWELL_STEPS`=45 IS reused as-is (that part
# of the shared constants was never implicated). First attempt used 165° legs
# directly, and it DID fix ep2/ep4's coverage, but the
# full n=15 re-gate surfaced a NEW regression: ep9 (bearing -39.7°, a previously-
# passing episode) started FALLING (reproducible, not noise) ~480 steps in, partway
# through the unfavorable-direction leg0(full 165°)+dwell+leg1(full 165° return)
# sequence -- a realized single-leg rotation of ~375 steps, uncomfortably close to
# the ~470-step/~323° continuous-rotation OOD ceiling docs/rot_dart.md /
# docs/nx1_scan.md diagnosed for this same shared policy (even though each leg is
# individually dwell-bounded, back-to-back unfavorable-direction legs apparently
# still stack risk in demo's environment/physics that eval_search's own validated
# 165°/45-step-dwell gate never triggered). 90° restores the ORIGINAL H3 design's own
# stated intent ("sweeps ±90° arc") -- just now correctly REALIZED via actual yaw
# tracking instead of the buggy assumed-rate calculation -- roughly halving worst-
# case single-leg rotation (~205 realized steps for 90° vs ~375 for 165°), which
# empirically eliminates the ep9 fall while still comfortably covering ep2 (-73.8°,
# needs only ~44.9° into leg2) and ep4 (+62.6°, found directly in leg0). KNOWN
# LIMITATION (documented, out of scope for this fix): a 90° leg gives a HARD ceiling
# of ±(90+28.9)=±118.9° effective bearing coverage -- demo scenes sample target
# bearing uniformly over the full ±180° (code/scene.py, `target_in_fov=False`), so a
# target beyond ±118.9° (not present in the seed=999 n=15 gate set -- max observed
# magnitude is ep2's 73.8°) would still time out unfound. Widening further would need
# a redesign beyond this fix's scope (e.g. detecting/escaping the OOD-risk condition
# directly, per docs/nx8_stall.md's STALL_BREAK precedent) -- see docs/nx10_scan_fix.md.
H3_LEG_DEG            = 90.0      # NOT eval_search's 165 -- see comment above
# `SCAN_TIMEOUT` here is this INITIAL scan's own absolute-episode-step safety net
# (mirrors eval_search's identically-purposed outer `SCAN_TIMEOUT` check) --
# distinct from `ReacquisitionScan`'s LOCAL step counter (code/lock_mgmt.py), which
# is the only thing safe to re-arm mid-episode. Bumped from 200 -> 1000: empirically
# (docs/nx10_scan_fix.md) the worst unfavorable-direction demo bearing in the gate set
# (ep9, -39.7°) clears leg0+dwell+leg1+dwell and finds the target partway through
# leg2 at a REALIZED absolute step of ~470 (reproducible); 1000 gives ample margin.
# `MAXSTEPS['demo']` (code/eval_closedloop.py) / `MAXSTEPS_GOTO` (code/demo.py) were
# bumped 1400 -> 1700 for the same reason NX-1 bumped MAXSTEPS_SEARCH: ep9's post-scan
# walk-in (already heading-aligned) needed ~1043 more realized steps (470 -> ~1513)
# to converge below stop_r -- the old 1400 cap would cut it off short.
SCAN_TIMEOUT         = 1000      # safety-net cap (was 200) -- see comment above
SCAN_RATE            = 0.6       # rad/s scan rate (unchanged; same as eval_search)
SCAN_DT              = SIM_DT * CONTROL_DECIMATION  # 0.02s per step
# Exit scan when bearing < SCAN_ALIGNED_THR or first detection (whichever is looser).
SCAN_ALIGNED_THR_DEG = 40.0     # target bearing < this → aligned, exit scan

# ---------------------------------------------------------------------------
# CAM-2 (Phase 1, docs/cam_opt2_multicam.md / docs/cam_p1.md): Schmitt-trigger
# handoff between the GROUNDING camera (26° pitch, far/mid range) and the new
# PROXIMITY camera (58° pitch, ~0.22-1.81m) on the EMA'd last-known distance.
# Render ONLY the active camera each grounding cycle -> steady-state compute is
# unchanged from pre-CAM-2 (still exactly one render per cycle in the common
# case; the bounded fallback probe adds a second render only on repeated
# misses, a handful of times per episode at most).
# ---------------------------------------------------------------------------
CAM_D_LO      = 1.2     # m — switch GROUNDING->PROXIMITY below this
CAM_D_HI      = 1.6     # m — switch PROXIMITY->GROUNDING above this
# CAM-P4 (docs/cam_p4_gate.md): the fallback PROBE's plausibility gate is keyed
# on the PROXIMITY camera's own physical far limit, not CAM_D_HI (the hysteresis
# threshold tuned for the reverse PROXIMITY->GROUNDING switch). CX-3 found
# (docs/cam_p3_demo.md) that gating on CAM_D_HI can deadlock: the EMA lags a fast
# monotonic approach (it blends past-higher and current-lower raw distances), so
# when GROUNDING loses the target just above CAM_D_HI (observed: last EMA~1.70m
# at true ~1.2m distance), the frozen last-known distance never re-updates (no
# further detection occurs to refresh it) and the probe gate blocks PROXIMITY
# forever -> permanent dead-reckoning for the rest of the approach (exactly the
# failure mode CAM-2 was built to eliminate). Fix: gate on the PROXIMITY camera's
# own physical far limit (d_far~=1.81m, docs/cam_opt2_multicam.md / arena.py
# PROXIMITY_PITCH=58 geometry) instead — still safely excludes genuinely-far
# detections (e.g. the ep13 blue-ball-at-4.96m regression, docs/cam_p1.md, >>1.81m
# either way) while covering the EMA-lag margin. Re-gated clean (docs/cam_p4_gate.md).
CAM_PROXIMITY_D_FAR = 1.81   # m — proximity camera's physical far limit (probe gate)
