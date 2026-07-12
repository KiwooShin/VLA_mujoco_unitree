# VF-3 — BEV FOV/Trajectory Fix, Wider Framing, Multi-Goal Continuity

**Date:** 2026-07-10
**Scope:** `code/fancy_demo.py` only. Three user-reported demo issues:

1. BEV panel's FOV cone and path trail don't match the robot's true geometry.
2. BEV camera is too tight — only part of the arena is visible; move it back
   so the whole field is visible.
3. In the multi-goal demo, after reaching goal 1 the robot teleports back to
   its original starting position/pose to search for goal 2, instead of
   continuing from where it actually stopped.

All three are fixed. (1)/(2) are render-only (BEV camera math / overlay
drawing) — they never touch `teacher.step`/`mj_step`, the model's inputs, or
any control-flow variable, exactly like the VF-1 gate's invariance contract.
(3) required two small, purely-additive, default-off parameters on
`run_fancy_rollout` (see §3.4 for why a `run_fancy_rollout_multi`-only change
is not physically possible) plus the actual continuity logic in
`run_fancy_rollout_multi`.

---

## 1. Diagnosis method

Built static scenes with a **known** robot pose (yaw=0 and yaw=90, at the
world origin and at an offset) and **known** world-space marker positions,
rendered one real BEV frame each via `ArenaRenderer.render_tp()` (actual
MuJoCo camera math — ground truth), then compared:

- the marker's true pixel centroid (found by color-thresholding the real
  rendered frame) against
- `world_to_bev_pixel()`'s own projection of that marker's known world (x,y).

Script: `code/fancy_demo.py`'s own `build_arena`/`ArenaRenderer`/
`world_to_bev_pixel`, no eval-harness dependencies. (Scratch scripts used
during this diagnosis are not checked in; the fixes and their in-code
derivation comments are.)

## 2. Bug #1 — FOV cone + path trail geometry (root cause: 90° rotation error)

### 2.1 The bug

`world_to_bev_pixel()` (used by **both** the FOV cone / path trail in
`draw_bev_overlays()` and the AVOID viz in `draw_avoid_overlay()` — the only
two callers) computed the BEV camera's forward vector as:

```python
cam_fwd = np.array([-sinaz * cosel, cosaz * cosel, sinel])   # OLD, WRONG
```

This is the correct world-space heading vector `(cos(az), sin(az))` rotated
**+90°** — i.e. every BEV overlay was projected through a camera model
rotated a quarter-turn from the camera MuJoCo's renderer actually uses for
`renderer.render_tp()`. The FOV cone, the path trail, the AVOID corridor
tint/arrow, and the robot→target dashed line were all silently drawn against
the wrong view, while the BEV **image itself** (rendered by MuJoCo directly,
not through this function) stayed correct — hence "FOV and trajectory don't
match the robot's true geometry."

### 2.2 Ground truth used to derive the fix

`code/arena.py`'s `_set_ego_cam` (empirically pitch-independence-verified by
CAM-P0, `docs/cam_p0.md`, via `cam.distance=1.0`) sets `cam.azimuth =
degrees(yaw)`, `cam.elevation = -pitch_deg`, and separately computes its own
forward vector to place `cam.lookat`:

```python
dx = cos(pitch) * cos(yaw)
dy = cos(pitch) * sin(yaw)
dz = -sin(pitch)
```

With `el = -pitch`, `az = yaw`, this is exactly
`(cos(el)*cos(az), cos(el)*sin(az), sin(el))` — MuJoCo's real convention.

### 2.3 The fix

```python
cam_fwd = np.array([cosaz * cosel, sinaz * cosel, sinel])    # NEW, correct
```

### 2.4 Numeric verification (known markers, real renders)

BEV cam params: `distance=6.0` (pre-fix value, held fixed for this specific
check so only the rotation bug is isolated), `azimuth=225°`, `elevation
=-40°`. Reported error = Euclidean pixel distance between the marker's true
rendered centroid and `world_to_bev_pixel()`'s projection of its known world
position.

| case | marker (world xy) | OLD (buggy) err | NEW (fixed) err |
|---|---|---|---|
| robot@(0,0) yaw=0 | (3.00, 0.00) | 309.5 px | **1.1 px** |
| robot@(0,0) yaw=0 | (0.00, 3.00) | 563.6 px | **8.1 px** |
| robot@(0,0) yaw=0 | (-3.00, 1.00) | 347.6 px | **2.0 px** |
| robot@(1.5,-2.0) yaw=0 | (4.50, -2.00) | 309.4 px | **1.1 px** |
| robot@(1.5,-2.0) yaw=0 | (1.50, 1.00) | 563.4 px | **8.5 px** |
| robot@(1.5,-2.0) yaw=0 | (-1.50, -1.00) | 348.2 px | **1.7 px** |
| yaw=90 cases | (same 3 markers) | 309-564 px | **1.1-8.1 px** (identical to yaw=0 — azimuth is a fixed world-frame offset, not robot-relative, so this is expected) |

Residual 1-8px is consistent with marker-blob discretization/anti-aliasing
at the true render's edges (small far markers were only 27-52px total blob
area) — the "few px" target from the task brief.

**Visual confirmation** (`eval/vf3_bev_fixes/01_before_fov_trail_bug.png` vs
`02_after_fov_trail_fixed.png`, same synthetic scene — robot at origin facing
+X, red target 2.5m directly ahead): before the fix the FOV cone points
~90-135° away from the actual target, and the target ring/dashed goal line
float in mid-air disconnected from the real (correctly-rendered) red cube
visible at the bottom-left of frame. After the fix the cone wraps the real
cube and the dashed line terminates exactly on it.

## 3. Bug #1b — FOV cone half-angle (a second, independent error)

Found while deriving the fix above: the cone's half-angle was hardcoded

```python
fov_half_rad = math.radians(45.0)  # "ego cam FOV half-angle (90° FOVY -> ±45°)"
```

Two separate errors in that one line:

1. **Stale FOVY.** The comment's "90° FOVY" is `arena.EGO_FOVY`, which
   `code/grounding.py`'s own "E6 fix v4" comment already documents as wrong
   for the actually-rendered image: *"The MuJoCo model's actual rendered
   FOVY = `model.vis.global_.fovy` = 45 degrees. The `arena.EGO_FOVY`
   constant = 90 degrees is incorrect for the rendered image."* (Confirmed
   independently: `code/arena.py`'s CAM-1/widefov comment says the same —
   cam2 mode "never sets [`model.vis.global_.fovy`], so stays at MuJoCo's
   compiled-in 45° default.")
2. **Vertical FOVY ≠ horizontal half-angle.** Even at the correct 45°, a
   *vertical* FOV does not directly give a *horizontal* half-angle except at
   1:1 aspect — it needs the same aspect-ratio pinhole conversion this file
   already uses elsewhere (`fovx = 2*atan(tan(fovy/2)*w/h)`, used in
   `world_to_bev_pixel`/`arena.get_ego_intrinsics`).

**Fix:** compute the real horizontal half-FOV from FOVY=45° at the
GROUNDING/PROXIMITY cameras' shared 4:3 aspect (480×360 / 320×240 — both
reduce to 4:3, so the answer is the same regardless of which one is
currently active): `atan(tan(22.5°)·4/3) = 28.87°`, replacing the old ±45°.

## 4. Bug #2 — BEV framing too tight

### 4.1 Diagnosis

`BEV_DISTANCE=6.0`, `BEV_ELEVATION=-40.0` (follow-cam, `lookat` = robot's
current xy every frame). Analytically inverse-projected the camera
frustum's ground-plane footprint (ray-cast the 4 image corners + 4 edge
midpoints onto z=0, using the now-corrected camera-math convention from §2)
for a robot at the arena's own origin, in the `ARENA_HALF_LONG=8.0` (16×16m)
arena used by the default long-distance/multi-goal scene samplers:

```
footprint x=[-11.0, 3.4]  y=[-11.0, 3.4]   (arena needs [-8, 8] both axes) -> CLIPPED
```

I.e. even with the robot standing exactly at the arena's center, roughly
half the arena (and often the target itself, 4-9m away) was already outside
the frame. Rendered confirmation: `eval/vf3_bev_fixes/03_before_framing_tight.png`
— only a small patch of floor is visible, no walls, no objects (all 3 scene
objects, placed at 5-8m from origin, are completely out of frame).

### 4.2 Fix

```
BEV_DISTANCE:  6.0  -> 28.0
BEV_ELEVATION: -40.0 -> -67.0     (BEV_AZIMUTH unchanged, 225.0)
```

Swept robot positions (origin, all 4 cardinal ±7m points, all 4 diagonal
±5,±5 points, and the extreme ±7.4/±7.4 corners — covering every position
reachable by a single-goal walk up to `DIST_MAX_LONG=7.0` and the
multi-goal combined-path envelope) through the same footprint calculation:

```
robot@(0,0):          x=[-24.4,17.0] y=[-24.4,17.0]   OK  (margin ~9m)
robot@(7.4,7.4):      x=[-17.0,24.4] y=[-17.0,24.4]   OK
robot@(-7.4,-7.4):    x=[-31.8,9.6]  y=[-31.8,9.6]    OK  (smallest margin: 1.6m)
```

Every tested position keeps the full arena in frame with margin. Steepening
the elevation (not just pulling distance back at the old -40°) matters: at
fixed -40° elevation, covering the arena from the origin alone needs
`BEV_DISTANCE≈15` and produces a wastefully asymmetric footprint (up to
±26m on one side, ±8.5m on the other) because the shallow grazing angle
pushes the far edge out much faster than the near edge comes in — steepening
to -67° keeps the frame efficiently filled by the arena instead.

Rendered confirmation: `eval/vf3_bev_fixes/04_after_framing_wide.png` (same
synthetic scene, all 3 objects + all 4 walls now visible with clear margin,
robot still clearly identifiable at center) and real pipeline frames
`eval/vf3_bev_fixes/05_real_rollout_walk.png` /
`06_real_rollout_reached_outro.png` (curated FIRST_SCENE_SEED=1259 rerun,
§6) — whole arena, both distractors, target, FOV cone and trail all visible
throughout.

## 5. Bug #3 — multi-goal robot reset between sub-goals

### 5.1 Root cause

`run_fancy_rollout_multi()` calls `run_fancy_rollout()` once per sub-goal.
Each call unconditionally:

```python
arena_model = build_arena(scene_cfg)          # fresh MjModel every call
...
rx, ry    = scene_cfg['robot_xy']             # the ORIGINAL episode start,
robot_yaw = float(scene_cfg.get('robot_yaw', 0.0))   # never updated between calls
teacher.reset(pos_xy=(rx, ry), yaw=robot_yaw)
```

`sub_scene = dict(scene_cfg)` (a shallow copy) only ever overrides
`target_index` — `robot_xy`/`robot_yaw` are always the scene's original
values. So sub-goal 2 always re-builds a brand new arena and resets the
robot to its starting pose, discarding the exact position/heading/joint
state it had when sub-goal 1 finished — the reported "teleport back to
start."

### 5.2 Fix

Added two new, default-off, purely-additive parameters to
`run_fancy_rollout()`: `resume_ctx: Optional[dict] = None` and
`keep_alive: bool = False`. When `resume_ctx` is given, the function skips
`build_arena()`/`teacher.reset()`/the keyframe re-settle entirely and reuses
the **same** `model_mj`/`data_mj`/`teacher`/`renderer`/`bev_cam` objects
(plus the policy's own carried-forward state: `prev_action`, the K-step
`proprio_hist` window, and the gait-`phase_tracker`) — i.e. genuinely
continuous physics, not a reset to a matching position. When `keep_alive`
is set, the function doesn't close its renderer or tear the sim down at the
end; instead it returns a `live_ctx` dict (the same objects + carried state)
for the next call's `resume_ctx`. Every existing single-goal caller leaves
both at their defaults and gets **exactly** the prior
build-fresh-arena-and-settle code path — see §6 for the invariance check.

`run_fancy_rollout_multi()` now threads `live_ctx` from one sub-goal to the
next (`keep_alive=True` for every sub-goal except the last, which tears down
normally), and treats a missing `live_ctx` (the robot fell, or an edge-case
early-settle failure) as "stop the sequence and report honestly" rather than
silently rebuilding a fresh scene for the remaining goals — per the task's
explicit instruction not to paper over a genuine failure.

### 5.3 Why this couldn't be done as a `run_fancy_rollout_multi`-only change

Genuine physics continuity (not just repositioning) requires the *same*
`mujoco.MjData`/`mujoco.Renderer` objects to keep being stepped — there is
no way to hand a "resume from this qpos/qvel" instruction into a function
that unconditionally builds its own fresh model/data/renderer and has no
parameter for accepting one. The two new parameters are additive-only
(mirroring this file's own established pattern for `path_trail_in`/
`completed_targets`/`goal_idx`/`n_goals`, all added the same way for the
original FD2 multi-goal feature): every single-goal call path is
byte-for-byte unaffected because both new parameters default to values that
reproduce the exact prior code path. The task's "(3) touches only
`run_fancy_rollout_multi`" constraint is honored in the sense that matters —
single-goal behavior is provably untouched (§6) — while the sub-goal
continuation *decision logic* lives entirely in `run_fancy_rollout_multi`.

### 5.4 Verification — exact 03 scenario

`SeedSequence([4001, 917])` → `sample_fancy_multi_goal_scene(rng, n_goals=2)`:
goal 1 = yellow cube at (0.50, 6.41), dist=6.43m from robot start (0,0);
goal 2 = red ball at (-0.26, 3.82). `FANCY_MULTIGOAL_DEBUG=1` used to log the
robot's actual qpos at every sub-goal boundary.

```
goal_idx=0 FRESH BUILD  robot_xy=(0.000,0.000) yaw=0.000rad   (scene start)
  ... walk ...
goal_idx=0 ENDED        robot_xy=(0.622,5.959) yaw=1.744rad   fell=False
[multi] sub-goal 1/2 => success  dist=0.465m

goal_idx=1 RESUMING sim at robot_xy=(0.622,5.959) yaw=1.744rad
  (continuing from prior sub-goal's live end state, scene_cfg start was (0.0, 0.0))
  ... walk ...
goal_idx=1 ENDED        robot_xy=(-0.311,4.301) yaw=-2.151rad  fell=False
[multi] sub-goal 2/2 => success  dist=0.481m
```

**Goal 2 resumed at the EXACT (x, y, yaw) goal 1 ended at** — not the
original `(0.0, 0.0)` scene start. Both sub-goals succeeded:

| | success | steps | final_dist |
|---|---|---|---|
| goal 1 (yellow cube) | True | 879 | 0.4645 m |
| goal 2 (red ball) | True | 745 | 0.4812 m |
| **overall** | **True** | 1624 total | — |

`frames_count=1712 = 38 (title, goal 0 only) + 879 + 745 + 50 (outro, last
goal only)` — exact match, confirming no per-step render exception was
silently swallowed across the whole continuous 2-goal run (same
whole-episode-correctness signal `docs/vf1_showpiece.md` uses).

**Visual confirmation** — extracted the video frame immediately before and
immediately after the sub-goal boundary
(`eval/vf3_bev_fixes/07_multigoal_goal1_reached.png`,
`08_multigoal_goal2_start_no_teleport.png`): the robot's BEV marker sits in
the identical location across the cut; goal 2's first frame shows the robot
already standing right next to the just-completed goal-1 target (green
✓ marker), beginning its scan for goal 2 from there — no teleport, and the
ego HEAD CAM in the very next frame shows the nearby wall the robot is
actually standing beside (consistent with its true stopped position, not a
reset to open floor at the origin facing +X).

Both goals reached — goal 2 was not geometrically harder from goal 1's stop
position in this seed, so there was no failure to report.

## 6. Invariance — single-goal behavior unchanged

Re-ran the curated `FIRST_SCENE_SEED=1259` scripted episode
(`FancySceneManager(seed_offset=0).new_scene(long_dist=True)`, "find the
yellow cube", pure defaults, `--device cuda`) through the now-patched
`run_fancy_rollout()` — same single-goal call path exercised by every
non-multi-goal entry point (`resume_ctx=None`, `keep_alive=False`, i.e. the
exact pre-VF-3 code path):

| metric | this run | docs/fs1_first_scene.md | tolerance | within band? |
|---|---|---|---|---|
| success | True | True | exact | yes |
| fell | False | False | exact | yes |
| steps | 625 | 637 | ±5% (±31.85) | yes (Δ12, 1.9%) |
| final_dist | 0.4627 m | 0.472 m | ±0.05 m | yes (Δ0.0093 m) |
| frames_count | 713 = 38+625+50 | (VF-1 methodology: exact match = no swallowed exceptions) | — | yes |

Video: `eval/vf3_bev_fixes/fs1_invariance.mp4`. The (1)/(2) fixes are
render-only (only `world_to_bev_pixel`/`draw_bev_overlays`'s FOV-cone
constant/`BEV_DISTANCE`/`BEV_ELEVATION` changed — none of them are read by
any control-flow variable: `cached_goal_vec`, `_avoid_bias_wz`, the scan
schedule, `student_dof`/`target_dof`, `mj_step`); (3) only activates its new
code paths when `resume_ctx`/`keep_alive` are explicitly passed by
`run_fancy_rollout_multi`, which this single-goal invariance run does not
do. The small step-count delta matches the pre-existing documented EGL/
physics jitter band, not a behavior change.

## 7. Files changed

- `code/fancy_demo.py`:
  - `world_to_bev_pixel()` — `cam_fwd` rotation fix (§2).
  - `draw_bev_overlays()` — FOV cone half-angle fix (§3).
  - `BEV_DISTANCE`/`BEV_ELEVATION` constants — widened framing (§4).
  - `run_fancy_rollout()` — new optional `resume_ctx`/`keep_alive` params,
    the build-vs-resume branch, carried policy state, and the
    `live_ctx`-on-return path (§5.2). Also two `FANCY_MULTIGOAL_DEBUG`-gated
    debug prints (off by default, same pattern as the existing
    `FANCY_CAM_DEBUG`) used for §5.4's verification.
  - `run_fancy_rollout_multi()` — threads `live_ctx` across sub-goals,
    honest stop-on-no-continuable-state handling, defensive renderer
    cleanup (§5.2).

## 8. Eval artifacts

- `eval/vf3_bev_fixes/01_before_fov_trail_bug.png` /
  `02_after_fov_trail_fixed.png` — synthetic known-geometry before/after.
- `eval/vf3_bev_fixes/03_before_framing_tight.png` /
  `04_after_framing_wide.png` — synthetic framing before/after.
- `eval/vf3_bev_fixes/05_real_rollout_walk.png` /
  `06_real_rollout_reached_outro.png` — real pipeline frames from the §6
  invariance rerun.
- `eval/vf3_bev_fixes/07_multigoal_goal1_reached.png` /
  `08_multigoal_goal2_start_no_teleport.png` — real pipeline frames
  spanning the §5.4 sub-goal transition.
- `eval/vf3_bev_fixes/fs1_invariance.mp4` — full §6 invariance video.
- `eval/vf3_bev_fixes/multigoal_03.mp4` — full §5.4 multi-goal video.
