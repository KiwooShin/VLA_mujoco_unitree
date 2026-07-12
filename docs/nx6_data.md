# NX-6 — Learned-Grounding Detection Dataset

**Date:** 2026-07-09
**Agent:** NX-6 (DATA), follow-on to `docs/nx5_coherence.md` §CLOSURE
**Goal:** a labeled dataset for training a small object detector that replaces the
classical HSV+depth grounder (`code/grounding.py`) — the four-agent NX-2→NX-5 chain
concluded that no further lock-management/heuristic fix on top of the classical
detector's output can separate its bearing-correct/distance-wrong false locks from
its legitimate detections, because that discrimination requires *appearance*
information (texture, edge continuity, shape) that is available in the raw pixels
but structurally inaccessible downstream of the classical pipeline's own collapsed
`(dist, bearing, area)` summary. This dataset supplies that appearance-level
supervision directly from MuJoCo's own instance segmentation — pixel-perfect labels,
no HSV heuristics, no classical-grounder failure modes baked in.

**Output:**
- `dataset/det_v1/` — 11,059 frames, 350 scenes, train/val/test = 280/35/35 scenes
  (80/10/10 **by scene**), 11,116 object-detection labels.
- `dataset/det_failcases/` — 118 frames from live replays of the checkpoint's 5
  documented demo failures (ep0/2/4/5/12) + search ep12, concentrated around the
  actual failure moments (the ep12 twin-hijack captured cleanly, see §4).
- Label-geometry sanity check (§5): back-projected (dist,bearing) from the stored
  bbox centroid + depth matches analytic ground truth to **p95 = 4.2cm / 2.7°**
  over 7,251 unoccluded checked detections.

---

## 1. Why MuJoCo segmentation, not the classical grounder, for labels

`mujoco.Renderer.enable_segmentation_rendering()` returns, per pixel, the exact geom
ID that was rasterized there (`(H,W,2)` int32, channel 0 = geom id, -1 = background),
rendered from the *same* camera pose used for the RGB/depth frame. `code/arena.py`'s
`build_arena()` names every object's geom(s) `obj_{i}` (`obj_{i}_tip` for the cone's
extra tip geom) in scene-object list order, so geom id → object index → (class,
color) is a closed-form lookup, not a heuristic. This sidesteps every failure axis
NX-2→NX-5 hit: no HSV hue collisions (wall/floor render into the cyan/blue band, see
§4), no blob-merging, no depth-outlier contamination — the segmentation mask is
exactly the visible pixels of that object at that camera pose, occlusion-aware for
free (MuJoCo's rasterizer naturally shows the frontmost geom per pixel).

## 2. Cameras, scenes, sampling (code/gen_det_dataset.py)

**Cameras** — both cameras actually used at deploy (`code/arena.py`,
`docs/cam_p1.md`): GROUNDING (26° pitch, 480×360) and PROXIMITY (58° pitch,
320×240, active below ~1.8m true distance, matching the deployed Schmitt-trigger
handoff band `CAM_D_LO=1.2/CAM_D_HI=1.6`).

**Scene families** — the actual samplers, not reinvented:
- `code/scene.py sample_scene(..., 'easy')` — 90 scenes, close range (1.5–2.5m),
  3 objects, target in FOV.
- `code/scene.py sample_scene(..., 'demo')` — 180 scenes, far range (4–9m), 5–7
  objects, target often out of FOV — the exact regime `docs/fa1_failures.md`'s
  ep0/2/4/5/12 come from.
- `code/eval_search.py sample_search_scene()` — 80 scenes, target forced outside
  ±45° FOV — the regime the search skill's scan-then-approach depends on.

Same-color-different-shape distractors and the cyan/blue-hue-collision wall/floor
(confirmed visually in the preview set — the floor renders as a blue-tinted
checkerboard, see `dataset/det_v1/preview/`) arise **naturally** from these samplers
(scene.py only forces (color, shape) *pairs* unique, so same-color multi-object
scenes are common — 350 scenes × 5-7 objects from a 7-color palette guarantees
frequent repeats) — 0 scenes needed to be special-cased.

**Frames per scene** (one arena build serves many frames — amortizes the
~200-500ms MjSpec compile):
1. **Trajectory** (~12 samples/scene): the robot walks toward the scene's nominal
   target via `code/steer.py`'s privileged controller + `WBCTeacher` physics,
   subsampled along the approach. Gives natural standing + mid-gait joint poses and
   a natural far→near distance sweep (breaks at <0.3m). A handful of qpos snapshots
   (joint angles only) are cached along the way.
2. **Teleport-focus** (~10 samples/scene): the robot is placed (qpos teleport +
   `mj_forward`, no physics) at a log-uniform distance in **[0.3, 10]m** and a wide
   bearing offset (±45°, ±80° for 25% of draws) from a randomly chosen object in the
   scene — deliberately including partial-clip and fully-out-of-frame negatives.
   Joint angles reused from the trajectory-phase snapshot cache, so teleported
   frames still look like a standing/mid-gait robot, not a T-pose.
3. **Teleport-random** (~6 samples/scene): fully random `(x, y, yaw)` "confusion"
   poses — generic multi-object / same-color-distractor scenes as the samplers
   naturally produce them.

25% of trajectory/teleport-focus draws additionally render the *other* camera at
the same pose, enriching the ~1.2–1.8m handoff-band overlap.

## 3. Labels

Per visible object per frame (from the segmentation mask + the true world pose —
`code/gen_det_dataset.py:derive_object_labels()`):

| field | source |
|---|---|
| `class_name`/`class_id` | shape (ball/cube/cylinder/cone — 4 classes; the READ-FIRST brief said "cone/ball/cube" but the sim actually samples all 4 `code/scene.py` shapes, so all 4 are labeled) |
| `color_name`/`color_id` | the 7-color `code/arena.py` palette |
| `bbox_x/y/w/h`, `centroid_px_x/y`, `area_px` | from the segmentation mask directly |
| `clipped` | bbox touches an image edge (partial visibility) |
| `depth_median_m` | median of the GT mask's own depth pixels (not an HSV mask) |
| `dist_gt_m`, `bearing_gt_deg` | **analytic** egocentric goal from true robot/object world poses (`steer.egocentric_goal`) — the actual training target |
| `dist_bp_m`, `bearing_bp_deg`, `err_dist_m`, `err_bearing_deg` | back-projected from (centroid, depth) via `arena.backproject_pixel` → `grounding.cam_to_egocentric` (nominal-radius-corrected) — the label-geometry self-check, §5 |
| `is_instructed_target` | whether this object is the one the scene's own NL instruction refers to (kept for reference; the detector itself is class+color-conditioned, not restricted to "the" target) |

Frame-level: `robot_x/y/yaw`, full `qpos` (36-d, enables exact re-render for any
frame — used by the preview step), `scene_id`, `split`, `difficulty`, `cam_type`,
`source`, `instruction`, `n_objects_visible`, `lighting_ambient`.

**Storage** (per split): `images_{grounding,proximity}.npz` (`rgb` uint8, `depth`
float16, `savez_compressed` — ~11.5× compression on these flat-color synthetic
renders), `frames.parquet`, `labels.parquet`. `scenes.json` (shared across splits)
holds each scene's full object list for exact reproducibility. Total dataset size:
**707MB** (well under the 4GB target).

## 4. Results

```
frames_total          = 11,059   (target 8-12k ✓)
frames_grounding_cam  = 5,242 (47.4%)
frames_proximity_cam  = 5,817 (52.6%)   (target ~60/40; actual mix close, driven by
                                          log-uniform distance sampling + natural
                                          trajectory dwell time near the target)
scenes                = 350  (0 fell during settle)
scenes split           = train 280 / val 35 / test 35  (80/10/10 BY SCENE)
classes_counts         = ball 2,581 / cube 2,636 / cylinder 2,897 / cone 3,002
n_labels_total          = 11,116
dist_gt_m range         = 0.15 – 11.8m  (median 1.86m; covers the 0.3-10m target
                                          plus a few natural extremes from large
                                          demo-arena distractors / near-teleports)
clipped fraction        = 33.3%   (partial-visibility coverage)
zero-object frames      = 23.9%   (hard negatives — wall/floor/empty-FOV)
color balance            = 1,416 – 1,736 per color (7 colors, reasonably even)
```

12 preview images (`dataset/det_v1/preview/*.png` — RGB + mask overlay + bbox +
label text, re-rendered from stored `qpos` for exact reproducibility) confirm: tight
per-object bboxes/masks including the cone's two-geom instance, correct multi-object
scenes with clipped partial objects, and correctly-labeled zero-object hard negatives
against the blue-tinted checkered floor — the exact wall/floor hue-collision risk
`docs/fa1_failures.md` documents for the classical grounder.

## 5. Label-geometry verification

For every detection with `depth_median_m` valid, `derive_object_labels()` also
back-projects the stored (centroid pixel, median depth) through
`arena.backproject_pixel` → `grounding.cam_to_egocentric` (nominal-radius-corrected
for the surface-vs-center depth offset; **always using the geometrically-correct
un-pitch formula for both cameras**, not production's per-camera legacy toggle —
`docs/cam_p1.md` documents production intentionally leaving the 26° grounding camera
on the old, slightly-biased formula so as not to shift the distribution the deployed
policy was tuned against; that's a deployment concern, irrelevant to validating this
dataset's own backprojection pipeline) and compares against the analytic GT
`(dist_gt_m, bearing_gt_deg)`.

Filtering to **unoccluded** detections (`not clipped` and `area_px >= 200`,
7,251 / 11,116 detections):

```
label_geometry_err_m_p95    = 0.0419 m   (4.2 cm)
label_geometry_err_deg_p95  = 2.67°
```

This is the expected ~exact result for a purely geometric pipeline (arena
intrinsics + offsets + segmentation-mask centroid/depth) validated against its own
analytic ground truth — the residual is consistent with pixel-centroid discretization
and the nominal-radius approximation (a sphere/cube/etc.'s near-surface depth vs. its
true center), not a bug.

## 6. Failure-case acid test (`code/gen_det_failcases.py` → `dataset/det_failcases/`)

Replays the **current deployed system** (`checkpoint/goto_best.pt`, demo/classical,
seed=999 — the exact held-out seed `docs/fa1_failures.md`/`eval_closedloop.py` use)
on the 5 documented demo failures + search ep12, and labels ~20 frames per episode
concentrated on the failure moments — using the *same* segmentation-based labeling
as the main dataset, captured live via non-invasive monkeypatching of
`code.grounding.ground()` (transparent passthrough: the real function is always
called, behavior is byte-identical to an uninstrumented run; only a labeled side-
buffer is populated), following the same caller-frame-introspection pattern already
established in this codebase (FA-1's `diag_ep0_raw.py`, NX-5's `nx5_mech_check.py`).
Frame selection: the highest `|classical_reported_dist − GT_dist_to_true_target|`
cycles (deduplicated with a minimum cycle spacing) plus evenly-spaced context frames
across the whole episode.

```
n_episodes = 6   n_frames = 118   n_labels = 77
```

| episode | outcome (this replay) | doc'd outcome (`fa1_failures.md`) | mechanism captured |
|---|---|---|---|
| demo ep0 | FAIL fd=3.40 | FAIL fd=3.40 | marginal/flickering blob at wrong depth — classical `dist≈6.1-6.2m` vs GT true target `~4.3m` (err≈1.85m) from step 0; by step ~60 the TRUE cyan cone drops out of the segmented view entirely as the robot walks toward the wrong equilibrium |
| demo ep2 | FAIL fd=10.9 | FAIL fd=10.66 (walks away) | confident false-lock, monotonically diverging |
| demo ep4 | FAIL fd=10.3 | FAIL fd=10.24 (walks away) | total detection miss (purple, no wall-hue overlap) |
| demo ep5 | FAIL fd=3.55 | FAIL fd=3.52 (stalls) | same stall signature as ep0 |
| demo ep12 | FAIL fd=5.65 | FAIL fd=6.67 (round trip) | **twin-hijack, captured cleanly**: frame_uid=79 (step 0) shows BOTH the true cyan cube target (dist_gt=6.18m, `is_instructed_target=True`) and the cyan-ball distractor (dist_gt=2.90m) visible together — classical reports dist≈2.73m (locked near the distractor already); by frame_uid=80 (step 200) the true cube has left the segmented view and the classical lock stabilizes on the nearby cyan ball (`classical_dist≈1.0m` for the remainder, cam handoff to PROXIMITY for 17/19 kept frames) |
| search ep12 | FAIL (fall), fd=1.80 | (not individually documented; included per brief) | acid-test diversity frame, not a targeted reproduction |

Per-episode final distances reproduce the documented failures closely (small
deltas are the documented GPU/EGL physics run-to-run non-determinism,
`docs/cam_p0.md`). The ep12 twin-hijack is the clearest capture: the dataset
contains, side by side in the same frame, the correctly-labeled true target and the
same-color-different-shape distractor that the classical grounder switches to —
exactly the appearance-discrimination case a learned detector needs to see.

**Implementation note:** `derive_object_labels()`'s backprojection originally
indexed `intr["pitch_deg"]` directly; `eval_search.py`'s cam2 grounding-camera call
site reuses a loop-invariant intrinsics dict that never has `pitch_deg` merged in
(a pre-existing quirk of that file — `code/grounding.py`'s own `ground()` already
defaults this to a *different*, also-slightly-wrong 32° fallback when absent,
`code/grounding.py:1340`). Fixed to `.get("pitch_deg", GROUNDING_PITCH)` — the
physically-correct render pitch for that call site regardless of what the passed-in
intrinsics dict happens to say, since the frame was always actually rendered via
`render_grounding()` at 26°. Zero WARN-dropped cycles after the fix (was 11/103 for
search ep12 before). `code/eval_search.py` itself was **not modified** — read-only
instrumentation only, per the DATA agent's read-first scope.

## 7. Files

- `code/gen_det_dataset.py` — main synthetic dataset generator (§2-3, §5).
- `code/gen_det_failcases.py` — failure-case replay + extraction (§6).
- `dataset/det_v1/{train,val,test}/{images_grounding.npz,images_proximity.npz,
  frames.parquet,labels.parquet}`, `dataset/det_v1/scenes.json`,
  `dataset/det_v1/meta.json`, `dataset/det_v1/preview/*.png` (12 samples).
- `dataset/det_failcases/{images_{ep_tag}_{cam}.npz,frames.parquet,
  labels.parquet,meta.json}`.
- Generation logs: `logs/gen_det_dataset.log` (76.8 min, 350/350 scenes, 0 falls),
  `logs/gen_det_failcases.log` (6/6 episodes, 0 instrumentation drops after fix).

**Determinism:** `--seed 7001` (main dataset; scene RNG via
`code.scene.derive_rng(seed, scene_id)`, teleport-pose RNG via a disjoint
`SeedSequence([seed, 0xA11CE, scene_id])` stream, train/val/test split via
`SeedSequence([seed, 0xD5])`); failcase replay uses the project's own held-out
`EVAL_SEED=999` matching the documents it reproduces.

## 8. Suggested next step (not built here — out of DATA-agent scope)

Train a small detector (e.g. a light anchor-free CNN head predicting per-cell
class/color/bbox, or a DETR-lite) on `dataset/det_v1/{train,val}`, evaluate held-out
detection metrics on `test`, THEN gate it end-to-end by swapping it in for
`code/grounding.ground()`'s HSV pipeline behind a toggle (same pattern as every
other mechanism in this codebase — default OFF, full easy/demo/search re-gate before
ADOPT) and checking specifically whether it closes the demo/classical ep0/2/5/12
failures `dataset/det_failcases/` targets, without regressing the passers whose
current success also depends on the classical grounder's incidental behavior (ep1,
ep13 per `docs/nx5_coherence.md` §CLOSURE).
