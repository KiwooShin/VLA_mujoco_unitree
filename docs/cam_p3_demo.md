# CAM-P3 — Fancy Demo Refresh with CAM-2 Active-Camera Ego Panel

**Date:** 2026-07-09
**Agent:** CX-3 (final camera-experiment phase)
**Goal:** fix the user's original complaint about `demo.gif` — the ego-view panel lost
the target object during the final approach — by making `code/fancy_demo.py`'s ego
panel show the CAM-2 **active camera** (head/GROUNDING far, PROXIMITY near) with the
Schmitt-trigger handoff (docs/cam_p1.md) visible, and record a clean success clip.

**Checkpoint:** `checkpoint/goto_best.pt` (unchanged, frozen). **CAMERA_MODE:** not set
(default `cam2` champion, per instructions).

## TL;DR

- Ego panel now renders the **active camera** (HEAD CAM / PROXIMITY CAM label, top
  right of the ego panel) instead of a fixed third "native ego" camera that was never
  actually driving detection.
- Found and fixed a real deadlock in the CAM-2 handoff **as ported into fancy_demo.py**
  (see §2) — without the fix, the demo reproduced the exact bug the user complained
  about (object lost near the stop, robot dead-reckoning blind).
- Clean success recorded: orange cube, 3.06 m, out-of-FOV (96° bearing) →
  search scan (150 steps) → located → long walk → PROXIMITY handoff at ~1.2–1.4 m
  (true dist) → REACHED at 0.48 m, **object visible in the ego panel at every frame
  through the stop**, including flanked by the robot's own arms (the documented
  self-body-occlusion scenario) — visually confirmed frame-by-frame.
- `videos/fancy_demo_cam2.mp4` (499 frames @ 25 fps ≈ 20.0 s), `videos/demo_cam2.gif`
  (400px, 7 fps, 4.1 MB), `videos/frames_cam2/` (3 stills for README).

## 1. What was implemented (`code/fancy_demo.py`)

`run_fancy_rollout()` previously had its **own** simplified search/goto loop, separate
from `code/inferencer.py`'s `Inferencer.rollout()`. It never used the proximity camera
or the Schmitt-trigger handoff at all — classical grounding always rendered via
`renderer.render_grounding()` (the far/head camera), and the ego video panel was a
**third, unrelated camera** (`renderer.render_ego()`, the native/default head cam) that
was never involved in detection. That's why the old ego panel lost the object close-up:
it was showing a camera that was never the one tracking the target, and had no
close-range coverage at all.

Changes:

1. **Ported the CAM-2 Schmitt-trigger handoff** from `code/inferencer.py`'s main
   rollout loop into `run_fancy_rollout()`, faithfully mirroring: `CAM_D_LO=1.2` /
   `CAM_D_HI=1.6` switch thresholds, `_active_cam` state (`GROUNDING`|`PROXIMITY`),
   per-cycle single-camera rendering (`renderer.render_grounding()` /
   `renderer.render_proximity()`, whichever is active), the bounded fallback probe
   (2 consecutive misses → probe the other camera once), and the plausibility gate.
   `intr_active` (the render call's own returned intrinsics dict, carrying
   `is_proximity`/`pitch_deg`) is now threaded into `classical_ground()` instead of a
   stale fixed grounding-only intrinsics dict computed once outside the loop.
2. **Ego panel now shows the active camera.** `_video_frame_cache` holds the last
   labeled active-camera frame (`_label_active_cam()`, reused from `inferencer.py`:
   draws `CAM: GROUNDING|PROXIMITY d=X.XXm` top-left), refreshed every grounding cycle
   (10 steps) and reused on in-between steps so video stays at full step-rate without
   extra renders. `_render_sbs_frame()` uses this cached frame as the ego panel instead
   of calling `renderer.render_ego()`.
3. **New "HEAD CAM" / "PROXIMITY CAM" label** added to `compose_sbs_frame()` (top-right
   of the ego panel, distinct from the smaller `_label_active_cam` overlay) — the
   explicit, easy-to-read indicator the task asked for so viewers see the handoff
   happen. Function signature gained an `active_cam` parameter (default `"GROUNDING"`,
   backward compatible).
4. **BEV follow-cam panel unchanged** (still the elevated diagonal third-person view
   with path trail / target ring / FOV cone / status banner overlays).
5. Web UI (`--web`) automatically inherits the same ego-panel behavior since it calls
   the same `run_fancy_rollout()` — verified still functional (see §3).
6. Multi-goal rollout (`run_fancy_rollout_multi`) needed no changes — it calls
   `run_fancy_rollout()` per sub-goal, so it's covered for free.

## 2. Bug found & fixed: plausibility-gate deadlock (fancy_demo.py-local)

While generating the demo clip, the **first** attempt (orange cube, 3.06 m) reproduced
the user's exact original complaint even with the ported CAM-2 logic: the ego panel
got stuck on "HEAD CAM" showing empty floor from ~1.2 m true distance all the way to
the stop (0.48 m) — i.e. the handoff never fired, and the robot dead-reckoned the last
~0.7 m blind, exactly the old failure mode CAM-2 was built to eliminate.

Root cause (confirmed via temporary instrumentation, `FANCY_CAM_DEBUG=1` env var, still
present in the code gated behind that flag): the GROUNDING camera lost the target
(`not_visible=True`) at a moment when its own EMA'd distance estimate was **1.696 m**
(true physical distance ≈1.2 m at that point — the EMA lags a fast monotonic approach,
since it blends past-higher and current-lower raw detections). The bounded fallback
probe's plausibility gate — ported faithfully from `code/inferencer.py` — requires
`last_known_distance <= CAM_D_HI (1.6)` before probing PROXIMITY. Since 1.696 > 1.6, the
probe was blocked, and because `_last_known_goal` only updates on a **successful**
detection, the stale 1.696 m value never got a chance to update: the gate was
permanently closed for the rest of the episode.

**Fix (scoped to `code/fancy_demo.py` only — `code/inferencer.py`'s champion numbers,
easy 100 / demo 66.7 / search 80, are untouched):** the probe's plausibility gate now
uses the PROXIMITY camera's own physical far limit, `CAM_PROXIMITY_D_FAR = 1.81 m`
(`docs/cam_opt2_multicam.md` geometry: 58° pitch → d_near≈0.22 m, d_far≈1.81 m), instead
of the hysteresis threshold `CAM_D_HI (1.6 m)`. This still safely excludes genuinely
far detections (the ep13 blue-ball-at-4.96 m regression documented in `docs/cam_p1.md`
is nowhere close to 1.81 m either way) while covering the EMA-lag margin that caused the
deadlock. Re-running the same seed with the fix: the probe fired at the very next
grounding cycle, correctly switched to PROXIMITY, and the target stayed detected
continuously (true dist 1.19 m → 0.48 m) all the way to REACHED.

This fix is local to the demo file and does not touch the gated eval path
(`code/inferencer.py`, `eval_closedloop.py`, `eval_search.py`), so the adopted CAM-2
champion's gated numbers are unaffected. It's flagged here as a genuine (if apparently
rare — not observed in the 45 gated eval episodes) edge case in the shared Schmitt-
trigger design, in case a future agent revisits `inferencer.py`'s own gate.

## 3. Verification

- **Web UI**: started `--web --port 5099`, confirmed `/`, `/scene_info`, `/status`,
  and `/stream` (MJPEG) all respond (200, valid JPEG bytes) — unchanged/working.
- **Clean success episode**: seed-derived scene, orange cube @ 3.06 m, bearing 96.1°
  (well outside initial FOV, guarantees the search phase). Result:
  `success=True, spotted=True, steps=499, final_dist=0.479 m`.
  - Search: 150 steps of CCW scan → SPOTTED (bearing 27.9° at detection).
  - Long walk: HEAD CAM tracks the cube continuously from ~2.9 m down to ~1.2 m true
    distance (visually confirmed frames at n=150, 300, 415).
  - Handoff: GROUNDING lost the target at true dist ≈1.2 m (frame n=415, "HEAD CAM",
    empty floor) → PROXIMITY probe fires next cycle → frame n=425 shows "PROXIMITY CAM"
    with the orange cube visible, flanked by the robot's own hands (self-body-occlusion
    scenario from `docs/cam_p1.md` §1c, visually reproduced).
  - Final approach/stop: frames n=475/485/495/498 (state MOVING→REACHED) all show
    "PROXIMITY CAM" with the cube clearly in frame at true dist 0.63/0.53/0.49/0.48 m —
    **confirmed visible at the stop.**

## 4. Deliverables

| File | Description |
|---|---|
| `videos/fancy_demo_cam2.mp4` | Full clean-success clip, 499 frames @ 25 fps ≈ 20.0 s, 1282×480 (ego \| BEV side-by-side), 8.3 MB |
| `videos/demo_cam2.gif` | GitHub-ready GIF, 400×150, 7 fps, 140 frames, loop, 4.1 MB (ffmpeg palettegen/paletteuse) |
| `videos/frames_cam2/01_search_scan.png` | SEARCHING state, target out of FOV, BEV shows direction arrow |
| `videos/frames_cam2/02_mid_walk_headcam.png` | MOVING, HEAD CAM tracking the cube at ~1.9 m |
| `videos/frames_cam2/03_final_approach_proximity_reached.png` | REACHED, PROXIMITY CAM, cube visible between the robot's hands at 0.48 m |

## 5. Files changed

- `code/fancy_demo.py`:
  - `run_fancy_rollout()` — ported CAM-2 Schmitt-trigger handoff + bounded fallback
    probe (with the `CAM_PROXIMITY_D_FAR` gate fix, §2) into the grounding block;
    replaced the fixed-camera ego render with the cached active-camera labeled frame.
  - `compose_sbs_frame()` — new `active_cam` parameter; ego panel now labeled
    "HEAD CAM" / "PROXIMITY CAM" instead of a static "EGO CAM".
  - `_render_sbs_frame()` (nested in `run_fancy_rollout`) — uses `_video_frame_cache`
    instead of `renderer.render_ego()`.
  - Web UI HTML — cosmetic label update ("ACTIVE CAM ... CAM-2 handoff" instead of
    "EGO CAM").
  - Optional `FANCY_CAM_DEBUG=1` env-gated debug prints left in place (no-op by
    default) for any future diagnosis of the handoff state machine.

## 6. v2 re-record (CX-6, 2026-07-09) — NX-1 bidirectional scan now shown

The shipped GIF from §4 predated NX-1's bounded bidirectional scan
(`docs/nx1_scan.md`, `code/scan_sched.py`) and NX-2's lock hardening
(`docs/nx2_final.md`), both since landed in `code/fancy_demo.py` — the old GIF
showed the retired fixed-CCW sweep. Re-recorded with the same recipe (no code
changes; `CAMERA_MODE` unset, `checkpoint/goto_best.pt`).

**Episode** (seed `SeedSequence([1, 3])` on `sample_fancy_scene_long`):
"find the orange cube", 4.29 m, signed bearing **-112.1°** (right side —
deliberately unreachable by the CCW-first leg, forcing a visible direction
reversal). Result: `success=True, spotted@step 900, steps=1474,
final_dist=0.471 m`, no fall.

- Scan: leg0 CCW 0→+165° (~step 0-300), dwell, **reversal**, CW sweep back
  through the start heading and on toward the target's side (legs 1+2),
  SPOTTED at step 900 (bearing 30.7° at detection) — the bidirectional
  triangle-wave sweep and its reversal are clearly visible in the BEV FOV cone.
- Long walk: HEAD CAM tracks the cube 4.5 m → ~1.2 m.
- Handoff: HEAD CAM → **PROXIMITY CAM** between frames ~1380-1420 (true dist
  ~1.2-0.9 m); cube visible between the robot's hands from there on.
- Stop: REACHED at 0.47 m, cube in the ego panel through the final frame;
  labels legible (verified frame-by-frame, final ~3 s).

**Deliverables:**

| File | Description |
|---|---|
| `videos/fancy_demo_v2.mp4` | Full clip, 1474 frames @ 25 fps ≈ 59 s, 1282×480, 24.1 MB |
| `videos/demo_v2.gif` | 400×150, 7 fps, 220 frames, 2× playback speed + 2 s end-frame hold, palettegen/paletteuse, **7.5 MB** |
| `videos/frames_v2/01_scan_leg0_ccw.png` | SEARCHING, CCW leg0 in progress |
| `videos/frames_v2/02_scan_cw_after_reversal.png` | SEARCHING, CW leg after the direction reversal |
| `videos/frames_v2/03_reached_proximity_cam.png` | REACHED, PROXIMITY CAM, cube between the hands at 0.47 m |

Staging asset `VLA_mujoco_unitree/assets/demo.gif` byte-copied
from `videos/demo_v2.gif` (not committed at the time of writing).

## 7. v3 re-record (CX-7, 2026-07-09) — FINAL adopted system: GROUND_NET + AVOID default-on

The v2 GIF (§6) predated the NX-9 adoption pass (`docs/nx9_avoid.md`): learned
grounding (`GROUND_NET`, the NX-6 heatmap detector) and local obstacle
avoidance (`AVOID`) are now both **default ON**. Re-recorded with the same
recipe and **zero env toggles** (pure defaults — log confirms
`GROUND_NET=1: loaded detector runs/nx6_heatmap_A/model_best.pt` and
`AVOID=True`, no classical-fallback lines). No code changes; only
`run_fancy_rollout(maxsteps=2800)` (up from the 2000 default) because a
~-96° scan + 6.8 m walk needs ~1,800 steps and attempt logs showed 2000
truncating still-closing episodes.

**Episode** (seed `SeedSequence([7007, 1701])` on
`sample_fancy_scene_long(dist_min=4.5, dist_max=7.0)`): "find the red cube",
**6.83 m** (vs v2's 4.29 m), signed bearing **-96.5°** (right side — CCW leg0
misses, forcing the visible NX-1 scan reversal). Scene prefiltered (pure-numpy
seed scan, `docs/showcase.md` §3 trick, extended with path-geometry checks) so
a **red ball distractor sits 0.37 m off the straight robot→target line at 54%
of the path** — inside AVOID's ±25° / 2.0 m corridor during the walk, and a
same-color decoy for the learned grounder. Result: `success=True,
spotted@step 880, steps=1775, final_dist=0.471 m`, no fall,
`avoid_bias_active_frac=0.044`.

- Scan: leg0 CCW 0→+165°, dwell, **reversal**, CW sweep to the right side,
  SPOTTED at step 880 (~7.1 m across the arena) — reversal clearly visible in
  the BEV FOV cone.
- Long walk (6.83 m): HEAD CAM tracks the cube continuously; mid-walk frames
  show BOTH red objects in the ego panel (ball near, cube far) with the robot
  holding lock on the cube — the GROUND_NET query-conditioned (shape-aware)
  detector never flips to the same-color ball decoy.
- Decoy pass: robot skirts LEFT of the red ball (~0.4 m clearance) with AVOID
  biasing yaw during the pass window (bias active on ~4% of grounding cycles —
  honest note: at this perpendicular offset the corrective bend is a gentle
  bow in the trail, not a dramatic S-weave; the engagement is confirmed by the
  `avoid_bias_active_frac` metric and the close pass is plainly visible).
- Handoff: HEAD CAM → **PROXIMITY CAM** between frames ~1560–1690 (true dist
  ~1.6–1.1 m); cube visible between the robot's hands from there on.
- Stop: REACHED at 0.47 m, cube in the ego panel through the final frame;
  HEAD CAM / PROXIMITY CAM / state / dist labels all legible at 400 px
  (verified on rendered GIF frames, final ~3 s hold included).

**Attempt history (bounded, 3 attempts):** seed 497 (red cone 6.59 m, -149.6°,
blue cylinder 0.34 m off-path) was a near-miss at `maxsteps=2000`
(fd=0.777 m still closing, `avoid_bias_active_frac=0.233`) and veered off in
the endgame on the 2800-step retry (fd=2.94 m — flaky seed, dropped); seed
1701 succeeded first try.

**Deliverables:**

| File | Description |
|---|---|
| `videos/fancy_demo_v3.mp4` | Full clip, 1775 frames @ 25 fps = 71 s, 1282×480, 30.3 MB |
| `videos/demo_v3.gif` | 400×150, 7 fps, 195 frames, 2.75× playback speed + 2 s end-frame hold, palettegen/paletteuse, **7.4 MB** |
| `videos/frames_v3/01_scan_leg0_ccw.png` | SEARCHING, CCW leg0 in progress |
| `videos/frames_v3/02_scan_cw_after_reversal.png` | SEARCHING, CW leg after the direction reversal |
| `videos/frames_v3/03_decoy_redball_pass_avoid.png` | MOVING, red-ball decoy at the robot's side mid-pass, trail + target ring in BEV |
| `videos/frames_v3/04_reached_proximity_cam.png` | REACHED, PROXIMITY CAM, red cube between the hands at 0.47 m |

Staging asset `VLA_mujoco_unitree/assets/demo.gif` byte-copied
from `videos/demo_v3.gif` (`cmp`-verified; not committed at the time of writing).
