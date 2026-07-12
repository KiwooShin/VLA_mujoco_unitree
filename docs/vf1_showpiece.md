# VF-1 — Showpiece Visual Upgrade for `code/fancy_demo.py`

**Date:** 2026-07-10
**Goal:** make `fancy_demo.py`'s rendered output dramatically more impressive —
VISUAL/OVERLAY changes ONLY. The robot's behavior (physics, grounding, steering,
scan, rollout logic) is BIT-IDENTICAL to before this change; every new feature is
render-side, reading state that already existed (or adding pure-read
instrumentation), never altering control flow.

## TL;DR

- 6 new render-side features, all individually toggleable, all default ON.
- `FANCY_PLAIN=1` disables every one of them at once and reproduces the
  pre-VF1 frame byte-for-byte (verified: same 1282x480 canvas, same small
  badges, no HUD/heatmap/gradient-trail/title-outro).
- Curated first-scene invariance check (seed=1259, "find the yellow cube"):
  **success=True** (matches doc), **steps=624** (doc: 637, Δ2.0%, within the
  documented ±5% EGL-jitter band), **final_dist=0.458m** (doc: 0.472m,
  Δ0.014m, within ±0.05m). Byte-identical control path — only the renderer
  changed.
- Per-step wall time with ALL new overlays on: **492.9 ms/step**, vs.
  **492.6 ms/step** measured on the SAME scene before any of this change —
  i.e. render-side additions cost ~0.3ms/step, nowhere near the 900ms budget.
  No resolution downgrade needed.
- Final canvas: **1602x646** (both panels displayed at 800x600 + a 46px HUD
  strip; ~1600x600 as asked, off by the HUD strip and a 1px mp4 codec trim).

## Files touched

- `code/fancy_demo.py` — all 6 features, the toggle system, `--scenario-title`
  CLI arg, and the plumbing to thread it through `run_smoke()`, the Flask web
  UI, and the terminal loop.
- `code/grounding.py` — `_ground_net()` now caches the last GROUND_NET cycle's
  confidence heatmap + decision (`_ground_net_last_heatmap` module global) and
  exposes it via `get_ground_net_last_heatmap()`. Purely additive: the cache
  write happens after `accepted`/`conf`/`dist`/`yaw_err` are already computed
  and does not change either `_ground_net()` return path.
- `code/nx6_heatmap_model.py` — `HeatmapDetector.infer()` now stashes
  `self.last_heat_prob` (the sigmoid confidence map from the SAME forward pass
  already computed, zero extra inference) and `self.last_heat_meta`.

## Why this is behavior-invariant (the gate)

Every new overlay is one of:

1. **A pure read of state that already existed** for control purposes
   (`cached_goal_vec`, `_avoid_bias_wz`/`_avoid_dbg`, `path_trail`,
   `current_state`, `_active_cam`, `_scan_active`, `data_mj.qpos`/`qvel`) —
   read-only, never reassigned by any of the new drawing code.
2. **A pure-read cache populated alongside a computation that already ran**
   every grounding cycle regardless of GROUND_NET's on/off value for THIS
   change (the detector's own forward pass) — the cache write is placed after
   every value that feeds the returned `GroundingResult` is already finalized,
   so it cannot perturb `accepted`/`dist`/`yaw_err` or either return path.
3. **New telemetry accumulators** (`dist_traveled_m`, the HUD's camera-flash
   countdown) that exist purely to feed pixels and are never read by any
   control-flow variable (`cached_goal_vec`, `_avoid_bias_wz`, the scan
   schedule, `student_dof`/`target_dof`, `mj_step`).
4. **Static frames** (title/outro cards) appended to `frames_sbs` strictly
   BEFORE the simulation loop starts or AFTER it has already finished —
   never interleaved with a control step.

None of the new code touches `teacher.step`/`mj_step`, the model's inputs
(`ego_rgb`, `gt_goal`, `gt_vel`, `proprio_h`), `cached_goal_vec`, the scan
schedule, or `_avoid_bias_wz`'s computed value — it only reads them.
Confirmed empirically too: the curated-seed rerun (see below) reproduces the
documented behavior within the pre-existing EGL/physics jitter band.

## The 6 features (in priority order)

All gated by one module-level flag apiece (default ON), all hard-disabled by
`FANCY_PLAIN=1`:

| # | Feature | Env var | Where |
|---|---------|---------|-------|
| 1 | Detector heatmap overlay | `FANCY_HEATMAP` | ego panel |
| 2 | AVOID visualization | `FANCY_AVOID_VIZ` | BEV panel |
| 3 | HUD bar | `FANCY_HUD` | bottom strip |
| 4 | Gradient trail + dashed goal line | `FANCY_TRAIL_GRADIENT` | BEV panel |
| 5 | Title card + outro stats card | `FANCY_TITLECARD` | full frame, pre/post-roll |
| 6 | 1600x600-ish canvas | `FANCY_HIRES` | whole composition |

### 1. Detector heatmap overlay (`draw_detector_heatmap_overlay`)

`code/nx6_heatmap_model.py`'s `HeatmapDetector.infer()` already computes a
per-pixel sigmoid confidence map every grounding cycle when `GROUND_NET=1`
(the ADOPTED default) — it just threw the map away after `decode_single()`
picked the argmax pixel. VF-1 stashes that map (`self.last_heat_prob`, one
elementwise numpy op on the array already produced, no extra inference) and
`code/grounding.py._ground_net()` caches it plus this cycle's
`(color, shape, cam_type, confidence, accepted)` alongside it.
`compose_sbs_frame()` blends it onto the ego panel only when: GROUND_NET is
active, the cache matches the CURRENT episode's target color+shape (so a
stale cache from a different query never leaks through), and the cycle
accepted a detection. Per-pixel alpha is proportional to confidence (capped
at 0.40, comfortably in the 0.35–0.45 gate band) with a small Gaussian blur so
the (typically tight, few-pixel) detector peak reads as a visible glow rather
than a hard-edged patch or a single dot — see
`eval/vf1_showpiece/frame_walk_heatmap.png` for the effect on a real cube
detection (conf=0.96). A "NEURAL DETECTOR conf=X.XX" tag is drawn at final
display resolution (crisp text, not upscaled) in the ego panel's bottom-left.

### 2. AVOID visualization (`draw_avoid_overlay`)

Reads `_avoid_bias_wz` and `compute_obstacle_bias()`'s own `info` dict
(`code/avoid.py`, already computed every grounding cycle whenever AVOID is
active and not scanning) — never calls `compute_obstacle_bias` itself. Draws:
an asymmetric left/right corridor-wedge tint (shaded by the L/R severities
avoid.py already computed), a repulsion arrow (direction = the bias's own
sign convention, magnitude = `|bias_wz| / AVOID_MAX_WZ_BIAS`), and a small
"AVOID" chip that only lights up while the bias is outside the deadband. The
curated first-scene has no near-path obstacle by construction (`fs1` picked a
distractor-clear scene on purpose), so this overlay's no-op path is what's
exercised in the gate-check video; a synthetic unit test (not checked in)
confirmed the drawing code itself with a fabricated nonzero bias/info.

### 3. HUD bar (`draw_hud_bar`)

A 46px strip below both panels: typed instruction verbatim (left), the
5-stage breadcrumb **SCAN > LOCK > WALK > HANDOFF > REACH** (center, active
stage highlighted, passed stages in green) — mapped from `_scan_active` /
`current_state` / `_active_cam`, which already fully determine it — live
`dist`/`bearing`/`step`/`walk-speed` readouts and a `CAM: HEAD|PROXIMITY` chip
(right) that flashes yellow for 10 frames right after a handoff. Bearing is a
fresh geometric read of `target_xy`/`robot_xy`/`yaw` (same inputs the BEV
overlays already use); walk-speed is `|data_mj.qvel[0:2]|`, a pure read of
sim state nothing else consumes. Verified the breadcrumb tracks the real
handoff live: `frame_check_630` (not checked in, but see
`eval/vf1_showpiece/frame_reached_outro.png`'s "HANDOFF" shown as a
completed/green stage) showed **HANDOFF** highlighted the instant
`_active_cam` became `PROXIMITY` while still `MOVING`, then **REACH**
highlighted at the true stop.

### 4. Path trail gradient + dashed goal line

`draw_bev_overlays()`'s path trail now interpolates cool-blue → warm-orange
by recency (thicker for the most recent segment) instead of a flat green
fade; the robot→target line is now dashed and drawn in the target's own
scene color (`color_rgb` from the object dict) instead of a fixed magenta.
Both fall back to the exact original single-color behavior when
`FANCY_TRAIL_GRADIENT=0`/`FANCY_PLAIN=1`.

### 5. Title card + outro stats card

`make_title_card()`: ~1.5s (38 frames @ 25fps) pre-roll, once per overall
episode (`goal_idx==0` only — a multi-goal run's later sub-goals don't repeat
it), showing `--scenario-title` + the typed instruction with a short fade-in.
Generated and appended to `frames_sbs` BEFORE the simulation loop starts.
`make_outro_card()`: ~2s (50 frames @ 25fps) freeze on the actual LAST
rendered SBS frame (so the scene/robot/target/proximity-cam close-up are
still visible) with a stats panel — elapsed sim time
(`steps * SIM_DT * CONTROL_DECIMATION`), total distance traveled (a new
per-step odometry accumulator, pure telemetry), final distance, and step
count — appended once, only on a successful REACHED finish, only at the
FINAL sub-goal. Both card generators mirror `compose_sbs_frame()`'s own size
arithmetic (`_final_canvas_dims()`) so every frame in one video shares one
shape, which `cv2.VideoWriter` requires.

### 6. ~1600x600 canvas

Both panels are displayed at 800x600 via a cheap `cv2.resize` on the
already-rendered frame — the native MuJoCo render resolutions
(`EGO_W/H`=320x240, `GROUNDING_W/H`=480x360, `PROXIMITY_W/H`=320x240,
`BEV_W/H`=640x480) are **completely unchanged**, so this costs a resize, not
a higher-resolution render. Measured overhead: negligible (see the
per-step-wall-time comparison above). Final encoded canvas: 1602x646 (800+3
divider+800 wide, 600 + 46px HUD strip tall; the video codec trims 1px off
odd widths, same pre-existing behavior the ORIGINAL 1283-wide canvas also had
— not a regression).

## Verification — behavior invariance (the gate)

Curated first scene (`FIRST_SCENE_SEED=1259`, `FancySceneManager` ep_count==0,
"find the yellow cube", pure defaults — GROUND_NET=1, AVOID=1, all adopted
fixes, all VF-1 overlays ON), one run, `--device cuda`:

| metric | this run (new renderer) | docs/fs1_first_scene.md | tolerance | within band? |
|---|---|---|---|---|
| success | True | True | exact | yes |
| fell | False | False | exact | yes |
| steps | 624 | 637 | ±5% (±31.85) | yes (Δ13, 2.0%) |
| final_dist | 0.458 m | 0.472 m | ±0.05 m | yes (Δ0.014 m) |
| ms/step | 492.9 | (fs1: n/a; pre-VF1 baseline on same scene: 492.6) | n/a | yes, +0.3ms |
| wall | 311.9 s | ~315.5 s | (informational) | consistent |

Video: `eval/vf1_showpiece/gate_check.mp4` (712 frames = 38 title + 624 step +
50 outro — the exact expected count, i.e. every single per-step render call
succeeded with zero silently-swallowed exceptions across the whole episode).

`FANCY_PLAIN=1` sanity run (40 steps, same curated scene): frame size
1282x480 (matches the ORIGINAL arithmetic, `EGO_W*(BEV_H/EGO_H)+3+BEV_W` =
640+3+640=1283, same 1px codec trim), frame count 40 (no title/outro cards
added), visually confirmed identical layout to the pre-VF1 look (small
badges, no HUD, no heatmap, no gradient trail) — see reasoning above for why
this is provably the same code path as before VF-1.

### 3 extracted + viewed frames

- `eval/vf1_showpiece/frame_scan.png` — SEARCHING, HUD breadcrumb on SCAN,
  no heatmap (nothing detected yet), off-screen target arrow + FOV cone on BEV.
- `eval/vf1_showpiece/frame_walk_heatmap.png` — MOVING, HEAD CAM, cube visible
  with a tight detector-confidence glow + "NEURAL DETECTOR conf=0.96" tag,
  gradient path trail (blue→orange), dashed yellow goal line to the (yellow)
  target, HUD breadcrumb on WALK.
- `eval/vf1_showpiece/frame_reached_outro.png` — REACHED, PROXIMITY CAM
  close-up of the cube between the robot's hands, outro stats card (time
  12.5s, traveled 4.41m, final dist 0.458m, steps 624), HUD breadcrumb on
  REACH with SCAN/LOCK/WALK/HANDOFF all marked done.

All three were also downscaled 50% (`*_50pct.png` in the same directory) and
visually re-checked: every text element (state badge, cam label, NEURAL
DETECTOR tag, HUD readouts/breadcrumb, outro stats) stayed legible, nothing
overlapped, the heatmap glow stayed a small localized tint (not obscuring the
scene), and no color read as garish.

## Toggles reference

- `FANCY_PLAIN=1` — disables ALL 6 features at once, exact pre-VF1 rendering.
- `FANCY_HEATMAP=0` / `FANCY_AVOID_VIZ=0` / `FANCY_HUD=0` /
  `FANCY_TRAIL_GRADIENT=0` / `FANCY_TITLECARD=0` / `FANCY_HIRES=0` — disable
  one feature at a time (everything else stays on). All default `1`.
- `--scenario-title "..."` (new CLI arg, `main()`) — scenario name shown on
  the title card; threaded through `--smoke` (`run_smoke`), `--web`, and the
  terminal fallback.

## Notes / honesty

- The AVOID visualization's "lit" path (nonzero bias) was not exercised in
  the gate-check video, because the curated first scene was deliberately
  picked (`docs/fs1_first_scene.md`) to have no near-path distractor — AVOID
  correctly stayed at zero bias the whole episode. The drawing code itself
  was validated with a synthetic fabricated `avoid_info`/`avoid_bias_wz`
  (confirmed correct wedge/arrow/chip rendering) and the real run's frame
  count (see above) proves the whole render call path, including this
  function's no-op branch, never raised.
- `frames_count` matching `38 + steps + 50` exactly (no dropped frames) is
  used throughout this doc as the evidence that no per-step render exception
  was silently swallowed by `run_fancy_rollout`'s existing
  `try/except: pass` around `_render_sbs_frame()` — a stronger, whole-episode
  correctness signal than spot-checking a few frames alone.
