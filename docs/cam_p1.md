# CAM-P1 — CAM-2 Proximity Camera + Schmitt-Trigger Handoff (Gate Report)

**Date:** 2026-07-08
**Agent:** CX-1 (Phase 1 of the camera-visibility experiment)
**Design brief implemented:** docs/cam_opt2_multicam.md (proximity cam at same head mount,
58° pitch, Schmitt-trigger handoff, same (dist,bearing) goal → no policy retrain).
**Checkpoint under test:** `checkpoint/goto_best.pt` (unchanged). **Eval protocol:** seed=999,
n=15, same as docs/cam_p0.md (`eval_closedloop.py --no-render`, `eval_search.py --no-video`).

## TL;DR

| Skill | P0 baseline | CAM-2 final | Δ |
|---|---|---|---|
| easy/classical | 100.0% (15/15) | **100.0% (15/15)** | 0 |
| demo/classical | 66.7% (10/15) | **66.7% (10/15)** | 0 |
| search | 80.0% (12/15) | **80.0% (12/15)** | 0 |

**Zero regression** on all three gated skills (per-episode outcomes match the P0 runs
episode-for-episode, including the same 3 search falls at eps 5/7/8 and the same 5 demo
cyan/blue wall-HSV failures). **Close-range visibility is transformed:** the target now
stays detected continuously from the handoff (~1.2 m) down to ~0.26 m true distance —
through every skill's stop radius (easy 0.6 / search 0.5 / demo 0.4 m) — where the
P0-only system went blind at ~0.7 m and dead-reckoned the entire final approach.

**VERDICT: ADOPT CAM-2.**

---

## 1. What was implemented

### 1a. Proximity camera (`code/arena.py`)
- `PROXIMITY_PITCH=58.0`, `PROXIMITY_W,H=320,240` (FOVY=45° via the shared rendered
  intrinsics) at the **same head mount** (`CAM_HEAD_Z=0.55`, `CAM_FWD=0.10`).
- Because of the P0 `cam.distance=1.0` fix, the rendered eye sits exactly at the mount
  point at any pitch → **no new forward-offset constant needed**; the recalibrated
  `CAM_ROBOT_FORWARD_OFFSET_M=0.10` applies unchanged (this is precisely why P0 was a
  prerequisite).
- `ArenaRenderer.render_proximity()` with a **pre-allocated** `mujoco.Renderer`
  (same anti-EGL-exhaustion pattern as `_gr_rend`). Its intrinsics dict carries
  `pitch_deg=58` and `is_proximity=True` (the flag that activates the proximity-only
  code paths in grounding.py — the 26°/32° camera paths are untouched).
- Offscreen buffer sizing extended to include the proximity dims.

### 1b. NEW BUG FOUND & FIXED (proximity-path only): un-pitch sign error
Validating the new camera against ground-truth distance sweeps exposed a real,
pre-existing geometry bug in `grounding.cam_to_egocentric()`: the forward-distance
term used `z_robot = y_cam*sin(pitch) + z_cam*cos(pitch)`; the geometrically correct
form for MuJoCo's look-at camera is `z_cam*cos(pitch) − y_cam*sin(pitch)`.

- At the shallow existing pitches (26°/32°) the error is modest and *hides inside* the
  already-documented "MuJoCo z-buffer distance underestimation" (docs/grounding_dist.md
  reported −28% at range; part of that is THIS sign error). The P0-gated behaviour is
  built on top of it, so it is left byte-for-byte unchanged there.
- At 58° the bug is fatal: reported distance **decreases** as true distance increases
  (true 0.3→1.8 m reported 1.01→0.49 m — an inverted, unusable mapping).
- Fix: `use_corrected_unpitch` parameter, threaded from `intrinsics['is_proximity']`
  only. Post-fix proximity sweep (true → reported): 0.30→0.36, 0.60→0.56, 1.00→0.94,
  1.40→1.33, 1.80→1.69 m — monotonic, accurate to within the object's own near-surface
  offset. Bearing sign convention verified unaffected (±0.3 m lateral targets report
  correctly-signed ~±16° bearings).
- **Follow-on flagged (out of scope):** applying the corrected formula to the 26°
  grounding camera would remove a systematic distance bias, but it shifts the
  distribution the deployed policy/EMA was tuned against, so it needs its own gated
  experiment.

### 1c. Depth-based self-body rejection (`code/grounding.py`)
Rendering a real close-approach walk (WBC teacher, orange cube, 1.27→0.15 m) confirmed
the concrete self-occlusion mechanism at 58°: **the robot's own arms/hands enter the
frame flanking the target inside ~0.7 m** (visually confirmed in saved frames), and the
target's color-mask depth histogram transiently goes bimodal (spread jumps 0.24→0.89 m
at true dist 0.68 m — contaminating samples at ~0.48 m vs object surface at ~1.2 m
camera depth). This is the same mechanism behind the P0 ep14 search regression.

Mitigation implemented (proximity path only):
- `MIN_DEPTH_PROXIMITY_M=0.15` replaces the 0.60 m floor **only when
  `is_proximity=True`** (the 0.60 m floor would blind the proximity cam over its whole
  useful range; grounding/ego cams keep 0.60 exactly as P0 shipped it).
- `_reject_depth_outliers()`: 1-D histogram clustering (5 cm bins) keeps only the
  largest contiguous depth cluster around the mode before the median — a disjoint
  contaminating population (arm/hand/edge-bleed at a different depth) is dropped; a
  real object's continuous surface always survives intact.
- Post-fix closed-loop check: reported distance stays monotonic and smooth through the
  arm-occlusion zone (no glitch at 0.65-0.68 m where the raw bimodality occurs),
  detection continuous from 1.27 m down to 0.256 m true distance; clean not-visible
  below ~0.24 m (inside every stop radius, and physically past the d_near≈0.22 m
  geometric limit).

### 1d. Schmitt-trigger handoff (`code/inferencer.py`)
- `CAM_D_LO=1.2 / CAM_D_HI=1.6` on the existing EMA'd distance; default GROUNDING at
  episode start. Only the **active** camera is rendered per grounding cycle →
  steady-state per-cycle cost unchanged (the 320×240 proximity render is cheaper than
  the 480×360 grounding render, so the close-range phase is actually cheaper).
- Bounded fallback probe: after 2 consecutive misses on the active camera, probe the
  other camera once and adopt its result if it detects.
- **PLAUSIBILITY GATE (added after first gate run):** the GROUNDING→PROXIMITY probe
  only fires when the last-known distance ≤ `CAM_D_HI`. See §2.
- All cameras funnel into the same `ground()` → (dist, cosθ, sinθ) EMA — the frozen
  policy never sees which camera produced the goal. No policy change.
- Demo viz: video frames now show the ACTIVE camera's view with a
  `CAM: GROUNDING|PROXIMITY d=X.XXm` label, so the handoff is visible in clips.

## 2. Gate history — the one regression found and fixed

**Round 1** (probe ungated): easy 100%, demo **60.0% (9/15 — regression)**, search not
yet run. Per-episode diff vs P0 isolated the flip to **ep13 (blue ball, 4.96 m)**:
P0 SUCCESS (609 steps, fd=0.36) → FAIL (fd=5.44). Diagnosis: camera-attributable, not
noise. Blue/cyan targets miss frequently at range (wall-HSV collision), so the 2-miss
probe kept rendering the proximity camera at far range — where it stares at the
blue-ish checkered floor (rendered H≈105, inside blue/cyan HSV bounds). A floor
false-positive at a bogus close distance flips `active_cam` to PROXIMITY and the
Schmitt trigger (EMA now < D_HI) locks it there — a self-reinforcing trap.

**Fix:** the plausibility gate in §1d — the proximity camera physically cannot see a
target beyond ~1.8 m, so probing it is only meaningful when the last-known distance is
≤ D_HI. This makes far-range behaviour **identical to P0 by construction**.

**Round 2 (final, shipped):** easy 100%, demo 66.7% (ep13 recovered: SUCCESS, 631
steps, fd=0.38), search 80.0% — per-episode identical to P0 across all 45 episodes.

## 3. Close-range visibility — the actual point of CAM-2

- P0-only: effective near-cutoff ~0.7 m (0.60 m depth floor + 0.10 m offset); the last
  ~0.7 m of every approach ran open-loop on `HOLD_GOAL_HORIZON` dead-reckoning.
- CAM-2: continuous detection through the handoff band down to **0.256 m true
  distance** (measured closed-loop against ground truth), i.e. through the stop radius
  of every skill (demo's fd≈0.36 m stops included). The blind window shrinks from
  ~0.7 m to below the success threshold — vision-based stopping.
- Search ep14 (P0's documented self-occlusion/overshoot risk case, orange cube 2.02 m):
  SUCCESS, fd=0.48, tracked visibly to dist=0.53 m — no overshoot.

## 4. Demo clips (handoff visible end-to-end)

- `videos/cam_p1_handoff_easy_ep14.mp4` — orange cube 1.51 m: near-immediate handoff,
  proximity-tracked stop (success, fd=0.58).
- `videos/cam_p1_handoff_easy_ep0.mp4` — orange cone 2.40 m: full GROUNDING far phase
  → labeled PROXIMITY handoff at ~1.2 m → target in frame to the stop (success,
  fd=0.56). Frame checks: mid-clip `CAM: GROUNDING d=1.87m`, final frames
  `CAM: PROXIMITY d=0.81m` with the target visible between the robot's own in-frame
  arms (the self-body-rejection scenario, live).

## 5. Files changed

- `code/arena.py` — PROXIMITY_* constants, `_prox_rend`/`_prox_cam`, `render_proximity()`,
  offscreen buffer sizing, `close()`.
- `code/grounding.py` — `MIN_DEPTH_PROXIMITY_M`, `_reject_depth_outliers()`,
  `cam_to_egocentric(use_corrected_unpitch=...)`, `is_proximity` handling in `ground()`
  (both the main and FG-rescue depth paths).
- `code/inferencer.py` — Schmitt-trigger state + handoff, plausibility-gated fallback
  probe, active-camera video labeling (`_label_active_cam`).

## 6. Eval artifacts

- `eval/p1_easy_cam2_v2/`, `eval/p1_demo_cam2_v2/`, `eval/p1_search_cam2/` — final
  shipped-configuration results (100.0 / 66.7 / 80.0).
- `eval/p1_easy_cam2/`, `eval/p1_demo_cam2/` — Round 1 (ungated probe): 100.0 / 60.0
  (ep13 regression, since fixed by the plausibility gate).
