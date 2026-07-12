# VF-5 — BEV camera closer to ground + >=7-object demo scenes

**Date:** 2026-07-11
**Scope:** `code/fancy_demo.py` only. Two user-reported demo issues:

1. "Move camera a little bit closer to the ground. I cannot distinguish
   objects in current setting well" — the VF-3 whole-arena framing
   (`BEV_DISTANCE=28.0` / `BEV_ELEVATION=-67.0`) is near-top-down and far
   enough that every object renders as a few-pixel dot.
2. "Add more objects in the fields... at least 7 objects" — the fancy-demo
   scenes only ever placed 3 objects (target + 2 distractors).

Both fixed. (1) is render-only (two constants). (2) touches only the two
fancy-demo scene samplers (`sample_fancy_scene_long`,
`sample_fancy_multi_goal_scene`); `code/scene.py` — the FROZEN
eval-benchmark sampler used by `eval_closedloop`/`eval_search`/
`eval_maneuver` — is untouched (§4 invariance check).

---

## 1. Camera retune (`BEV_DISTANCE` 28.0→17.0, `BEV_ELEVATION` −67.0→−43.5)

Same analytic-footprint methodology as VF-3 (`world_to_bev_pixel()`
ground-plane projection of every reachable object position), plus real
renders of a 7-object scene at each candidate, viewed both full-size and at
480px gallery scale.

Candidates swept: (14,−38), (16,−42), (18,−45), (20,−50), then fine
interpolation. Frame-bounds check: project objects at every (angle, dist)
combination the samplers can actually produce — distractors at 2–5 m any
bearing, targets at 4–7 m outside the initial ±45° FOV — through the
candidate camera (robot at lookat, the follow-cam's invariant):

| candidate | out-of-frame probe points (of 120) | object legibility (renders) |
|---|---|---|
| d=14, e=−38 | 12 — real target spawns at 6–7 m / 45–90° land up to 63 px below frame | best (largest objects) |
| d=16, e=−42 | 3 — only the 7.0 m ring at 30–60°, ≤18 px over | very good |
| **d=17, e=−43.5** | **1 — only the exact-45°/7.0 m point, 0.1 px over; excluded by construction (sampler requires bearing strictly > FOV half-angle)** | **very good** |
| d=18, e=−45 | 0 | good |
| d=20, e=−50 | 0 | noticeably smaller |
| d=28, e=−67 (old) | 0 | few-pixel dots — the complaint |

Chose **d=17.0 / e=−43.5**: shapes are legible (cone vs. cylinder vs. cube
vs. ball distinguishable in the rendered comparison, including after a
480px downscale) and the only theoretically croppable point is one the
target sampler can never produce. Distractors ≤5 m are in frame with wide
margin at every bearing (worst-case v=396/480 at d=5 m). Azimuth/lookat
logic unchanged; the BEV cam still tracks the robot every frame, so these
bounds hold wherever the robot walks (objects are checked relative to the
robot; only the outer walls leave frame during long excursions, as before).

Comparison renders from the sweep (scratchpad, session dir
`vf5_cam/cam_d{14,16,17,18,20}_e*.png` + `_480.png` variants) were viewed
frame-by-frame; the verified-episode frames in §3 are the durable record.

## 2. >=7-object scenes (fancy-demo samplers only)

New module constants + two helpers in `code/fancy_demo.py`:

- `FANCY_MIN_OBJECTS = 7`, `FANCY_OBJ_MIN_SEP_M = 1.2`,
  `FANCY_OBJ_WALL_MARGIN_M = 0.8`, `FANCY_OBJ_MIN_ROBOT_M = 1.0`.
- `_place_fancy_object_xy()` — rejection-samples one (x,y) honoring
  separation/wall/robot-distance rules with the target's existing
  distance-band + out-of-FOV logic; three progressively-relaxed passes
  (mirrors the samplers' pre-existing fallback pattern) so placement never
  hard-fails.
- `_select_fancy_distractor_combos()` — picks distinct (color,shape)
  distractor combos from the FULL palette, guaranteeing at least one
  same-color/different-shape partner for the main target. Same-color/
  SAME-shape (the `docs/gen1_multiseed.md` §3.3 false-lock combo) is
  excluded by construction — all combos are pairwise distinct.

`sample_fancy_scene_long`: target (reliable color, 4–7 m, out-of-FOV,
unchanged logic) + 6 distractors (2–5 m, any bearing) = 7 objects.
`sample_fancy_multi_goal_scene`: n_goals sub-goals (reliable colors,
unchanged distance bands: first 4.5–6.5 m out-of-FOV, later 2.5–5.0 m) +
5 extra distractors = 7 objects for the default n_goals=2. Objects gained
an `is_goal` field (additive; no reader existed before).

Determinism preserved: both samplers remain pure functions of the passed
`rng`. Validation sweep (scratchpad `vf5_check_samplers.py`): 30 seeds of
`sample_fancy_scene_long` + 20 seeds of `sample_fancy_multi_goal_scene` —
**all 50 scenes**: exactly 7 objects, all pairwise separations >=1.2 m, all
wall clearances >=0.8 m, all robot-spawn distances >=1.0 m, same-color/
diff-shape pair present, target out-of-FOV; same-seed re-sample
byte-identical (both samplers).

## 3. Verified episode (new camera + rich scene, full pipeline)

Prefiltered seed (geometry filter over `SeedSequence([9001, ep])`,
ep 0..399, then picked for a distractor sitting almost exactly on the
robot→target straight path): **`np.random.SeedSequence([9001, 200])` →
`sample_fancy_scene_long(rng, 200)`** — target **orange cylinder,
dist=5.95 m, bearing=+149.1°** (far out-of-FOV, positive-signed to match
the scan schedule's positive-first leg order), 7 objects incl. an
orange cone + orange cube (same-color/diff-shape partners) and a yellow
cone at perp 0.04 m / t=0.55 of the walk path.

Result (device=cuda, defaults, `run_fancy_rollout`, MAXSTEPS=2000):
**SUCCESS** — `spotted=True` at step 270 (scan), `steps=948`,
`final_dist=0.464 m`, `fell=False`, wall 397 s, 1036 frames
(38 title + 948 sim + 50 outro ✓).

Artifacts (`videos/vf5_verify/`):
- `verify_ep.mp4` (41.4 s), `verify_ep_result.json`, `verify_ep_framelog.json`
- 5 representative frames, each viewed full-size AND at 480px gallery scale
  (`frame_*_480.png`): `frame_title.png` (title card),
  `frame_scan.png` (step 135 SEARCHING — all 7 objects legible in BEV,
  target ring on the orange cylinder, scan cone visible),
  `frame_mid_walk.png` (step 641 MOVING, dist 2.98 m — NEURAL DETECTOR
  conf=0.99 on the ego cylinder, gradient trail + dashed goal line),
  `frame_close.png` (step 880 MOVING, dist 1.00 m — PROXIMITY CAM handoff,
  HANDOFF breadcrumb lit), `frame_reached.png` (REACHED + outro stats card:
  19.0 s, 6.38 m traveled, 948 steps).

At 480px the object shapes/colors in the BEV panel remain distinguishable —
the specific complaint the old d=28/e=−67 framing failed.

## 4. Invariance spot-check

- **Files touched: `code/fancy_demo.py` only.** Verified two ways:
  `find code -newermt "2026-07-11 10:00"` returns only `fancy_demo.py`, and
  `cmp` against the published repo copy shows `scene.py`,
  `eval_closedloop.py`, `eval_search.py`, `grounding.py`, `avoid.py`,
  `steer.py`, `inferencer.py`, `arena.py`, `scan_sched.py`, `lock_mgmt.py`
  all byte-identical (only `fancy_demo.py` differs, as expected pre-sync).
- **Frozen-eval smoke:** `eval_closedloop.py --checkpoint
  checkpoint/goto_best.pt --difficulty easy --n 1 --device cuda
  --no-render` → **1/1 SUCCESS**, steps=253, final_dist=0.56 m, 0 falls,
  ~3.4 ms/step NN+physics — harness loads, runs, and scores normally.

## 5. Files changed

- `code/fancy_demo.py`:
  - `BEV_DISTANCE` 28.0→17.0, `BEV_ELEVATION` −67.0→−43.5 (+ derivation
    comment block, §1).
  - New: `FANCY_MIN_OBJECTS`/`FANCY_OBJ_MIN_SEP_M`/`FANCY_OBJ_WALL_MARGIN_M`/
    `FANCY_OBJ_MIN_ROBOT_M`, `_place_fancy_object_xy()`,
    `_select_fancy_distractor_combos()` (§2).
  - `sample_fancy_scene_long()` / `sample_fancy_multi_goal_scene()` —
    rewritten placement loops using the helpers; target/goal distance-band +
    out-of-FOV logic unchanged (§2).
- `videos/vf5_verify/` — new (§3 artifacts).
- `docs/vf5_cam_objects.md` — this file.

Note: `sample_fancy_scene` (the short-range 3-object sampler) is now
effectively legacy — `FancySceneManager.new_scene()` defaults to
`long_dist=True` and the smoke/web/terminal paths all draw from the two
updated samplers. Left as-is (still referenced by the `long_dist=False`
branch). The FS-1 curated `FIRST_SCENE_SEED=1259` first draw now produces a
7-object scene (different from the one documented in
`docs/fs1_first_scene.md`); if the first-launch experience matters for the
next recording pass, re-verify that seed's rollout the same way FS-1 did.

**Update (2026-07-11, FS-2):** Done — see `docs/fs2_first_scene.md`.
Re-ran FS-1's geometry pre-filter against the current (7-object)
`sample_fancy_scene_long`, picked a new top candidate (`seed=3461`: yellow
cube, dist=4.97m, bearing=84.9°, 7 objects, no same-color/same-shape
confusion, nearest distractor 2.15m clear of the straight path), verified
it headlessly 2x (both SUCCESS, no fall, `final_dist`~0.468m both runs,
frame counts independently confirmed via `ffprobe` against `38+steps+50`,
md5s confirmed unique against every existing file under `videos/` and
`demo_videos/`), and updated `FIRST_SCENE_SEED` in `code/fancy_demo.py`
1259 -> 3461.
