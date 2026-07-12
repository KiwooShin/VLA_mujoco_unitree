# CAM-P2 — CAM-1 Wide-FOV Single Camera (A/B Gate Report)

**Date:** 2026-07-08
**Agent:** CX-2 (Phase 2 of the camera-visibility experiment)
**Design brief implemented:** docs/cam_opt1_widefov.md (single wide-FOV camera at the
existing head mount, no proximity cam, no handoff).
**Champion under test (baseline):** CAM-2 (docs/cam_p1.md) — proximity cam (58° pitch)
+ Schmitt handoff (D_LO 1.2 / D_HI 1.6) + self-body rejection + plausibility gate.
**Checkpoint:** `checkpoint/goto_best.pt` (unchanged). **Eval protocol:** seed=999, n=15
(`eval_closedloop.py --no-render`, `eval_search.py --no-video`), same as cam_p0/cam_p1.

## TL;DR

| Skill | CAM-2 champion | CAM-1 (widefov) | Δ |
|---|---|---|---|
| easy/classical | 100.0% (15/15) | **100.0% (15/15)** | 0 |
| demo/classical | 66.7% (10/15) | **60.0% (9/15)** | **−6.7pp (1 ep net)** |
| search | 80.0% (12/15) | **73.3% (11/15)** | **−6.7pp (1 ep)** |
| Visible-to-stop (true dist) | ~0.256 m | ~0.40–0.46 m (closed-loop) / 0.30 m (static bench) | worse |
| ms/step (easy / demo) | 24.5 / 18.0 ms | 44.2 / 38.4 ms | **~2× slower** |

**VERDICT: CAM-2 STAYS CHAMPION.** CAM-1 matches on easy but regresses on both demo and
search (one net episode each), has a shallower close-range visibility floor than CAM-2's
proximity camera, and is compute-*heavier* per step (larger single frame), not cheaper —
none of the conditions in the adoption gate are met. **Repo left in `cam2` (default)
mode; CAM-1 is a toggle, off by default.**

---

## 1. What was implemented (toggle, not a replacement)

Per the hard rule, CAM-1 is a `CAMERA_MODE` env-var toggle (`cam2` default / `widefov`),
implemented as additive branches around the existing CAM-2 code, never by editing its
behavior when the toggle is off:

- **`code/arena.py`**: `CAMERA_MODE = os.environ.get("CAMERA_MODE", "cam2")`. New
  constants `WIDEFOV_W,H=640,480`, `WIDEFOV_FOVY=70°`, `WIDEFOV_PITCH=42°` (same head
  mount, `CAM_HEAD_Z`/`CAM_FWD` unchanged). `build_arena()` only sets
  `spec.visual.global_.fovy = WIDEFOV_FOVY` when `CAMERA_MODE=='widefov'` (cam2 mode
  never touches this, so its rendered FOVY stays at MuJoCo's 45° default exactly as
  cam_p1 shipped it). `ArenaRenderer` gets one more renderer, `_widefov_rend`, built
  **only** in widefov mode (`None` otherwise) — cam2's four renderers (ego/grounding/
  proximity/tp) are constructed by the exact same unconditional lines as before.
  `render_widefov()` returns intrinsics via the existing generic
  `get_ego_intrinsics(w,h,fovy_deg)` fed the *actual* WIDEFOV_FOVY (this is exactly the
  Finding-#1 fix from docs/cam_opt1_widefov.md, scoped only to this new path).
- **`code/grounding.py`**: `is_widefov` flag threaded through `intrinsics` exactly like
  `is_proximity` (same pattern CAM-2 established). Widefov detections get their own
  depth floor (`MIN_DEPTH_WIDEFOV_M=0.15`, same rationale as the proximity floor),
  self-body depth-outlier clustering (`_reject_depth_outliers`, reused unchanged), and
  the corrected un-pitch sign (`use_corrected_unpitch=is_proximity or is_widefov`).
- **`code/inferencer.py`**: the render-selection block gets an
  `if CAMERA_MODE=='widefov': render_widefov(...) elif ...` branch ahead of the
  existing `_active_cam`-based cam2 selection; the Schmitt-handoff and bounded-fallback-
  probe blocks are wrapped in `if CAMERA_MODE != 'widefov':` (both true, hence no-ops,
  when the toggle is at its default).
- **Geometry** (`H=CAM_HEAD_Z+pelvis≈1.34m`): solving `d_near=H/tan(θ+φ)=0.30m` for
  FOVY=70° (φ=35°, the task-specified probe value) gives θ≈42.4°, with resulting
  `d_far=H/tan(θ−φ)≈10.3m` — comfortably past the 8–9m demo range, so 70° was never
  escalated to 60° (no need — see §2, far-range held up fine at 70°).

## 2. New bugs found & fixed while implementing/benching (both empirically, not assumed)

**(a) Un-pitch sign bug is real at WIDEFOV_PITCH=42°, same class as cam_p1's finding at
58°.** A direct-distance-sweep bench (`code/bench_widefov_dist.py`, known target
positions 0.15–11 m, both formulas compared) showed the *uncorrected* formula
saturates and is nowhere near monotonic (e.g. true 9m → reports 2.2m), while the
*corrected* formula (`z_cam·cosθ − y_cam·sinθ`) tracks true distance to within a few
percent across the whole 0.3–9.5m range. Locked in as `is_widefov → corrected` from the
start (no regression window shipped).

**(b) `eval_search.py` has its own hand-duplicated rollout loop that bypassed the
toggle entirely — this, not a genuine CAM-1 weakness, caused the first search run's
catastrophic 0/15.** `_run_search_rollout()` (used only by the search-skill evaluator)
predates CAM-2/CAM-1 and reimplements the render/grounding loop independently of
`Inferencer.rollout()`. It always called `renderer.render_grounding()` with intrinsics
precomputed for a fixed 45° FOVY. Because `spec.visual.global_.fovy` is a **model-wide**
MuJoCo setting, activating widefov mode silently widened the FOVY of *every* camera
built from that model, including this one — so the rendered image was actually 70° FOVY
while the bearing/distance math still assumed 45°, corrupting every backprojected
point. Symptom: 100% spot-rate (HSV detection still fires) but 0% reach-rate, with
robots walking monotonically *away* from the target after "spotting" it (e.g. ep1:
2.53m → 6.49m over 1200 steps, plateauing at an arena wall). Fixed the same way as the
production toggle: `_run_search_rollout` now calls `renderer.render_widefov()` and uses
its own per-cycle intrinsics when `CAMERA_MODE=='widefov'`; the `else` branch is
byte-for-byte the original cam2 call. Re-verified cam2 mode unaffected (n=2 check
reproduced champion's episodes 0/1 almost exactly: fd 0.46/0.49 vs champion's
0.476/0.492, scan_steps 400/470 matching exactly). This fix took search from 0.0%→73.3%
in widefov mode — the *real* CAM-1 number is 73.3%, not 0%.

## 3. Far/near-range bench (before the full eval, per the task's gate)

Static direct-facing bench (`code/bench_widefov_dist.py`; target on-axis at known world
distance, `WIDEFOV_FOVY=70°`, `WIDEFOV_PITCH=42°`, using the real `arena.py`/
`grounding.py` code):

| d_true (m) | 0.15 | 0.20 | 0.25 | **0.30** | 0.5 | 1 | 2 | 4 | 6 | 8 | 9 | 9.5 | **10.0** | 10.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blob area (px²) | 0 | 0 | 1879 | 3451 | 1568 | 3272 | 2088 | 1266 | 606 | 277 | 132 | 81 | 41 | 0 |
| `ground()` detects | N | N | N | **Y** | Y | Y | Y | Y | Y | Y | Y | Y | N (edge) | N |
| reported dist | — | — | — | 0.32 | 0.46 | 0.87 | 1.94 | 3.95 | 5.87 | 7.85 | 8.85 | 9.35 | — | — |

**Far range: solid.** 70° FOVY at 640×480 clears `MIN_BLOB_AREA=40px²` all the way to
9.5m (3.5m past the 6m demo floor, 0.5–1.5m past the 8–9m demo ceiling) with 2–3×
margin at 8–9m — no need to escalate to 60° FOVY. This matched the analytic
`d_far≈10.3m` prediction closely (usable cutoff ~9.5–10m once erosion/min-valid-pixel
effects are accounted for).
**Near range: meets the 0.3m design target in a static, forward-facing pose** — but see
§4 for why the real closed-loop number is worse.

## 4. Full A/B eval

### easy/classical — MATCH (100.0% both)
All 15 episodes SUCCESS, matching champion episode-for-episode (verified ep0 orange
cone 2.40m: fd=0.56m in both).

### demo/classical — REGRESSION (60.0% vs 66.7%, net −1 episode)
Per-episode diff vs champion (`eval/p1_demo_cam2_v2` vs `eval/p2_demo_widefov`):

| ep | target | dist | CAM-2 | CAM-1 | |
|---|---|---|---|---|---|
| 0 | cyan cone | 4.32m | didnt-reach | **success** | CAM-1 gain |
| 1 | cyan cube | 7.42m | success | **didnt-reach** | CAM-1 loss |
| 13 | blue ball | 4.96m | success | **didnt-reach** | CAM-1 loss |
| 12 others | — | — | same | same | — |

Net: +1 gain, −2 losses = −1 episode (−6.7pp). The two new losses are exactly the
class of failure this experiment's own risk assessment predicted: cyan/blue targets are
already the documented wall-HSV collision case in this codebase (`docs/grounding_dist.md`,
and champion's own 5 demo failures are all cyan/blue); CAM-1's wider FOV shrinks
angular resolution (smaller blob per the FOV-vs-resolution trade-off in
docs/cam_opt1_widefov.md §1), so the already-marginal cyan/blue detections tip over
into failure slightly more often than they do at CAM-2's narrower 45° grounding FOVY.
ep1 also newly failed with `fell=True` (self-body/gait interaction, 1 new fall vs
CAM-2's 0 falls on demo).

### search — REGRESSION (73.3% vs 80.0%, 1 episode)
After the eval_search.py bypass bug was fixed (§2b), 14/15 episodes matched the
champion exactly (same successes, same 3 falls at eps 5/7/8). One new loss:

- **ep1 (yellow cube, 2.53m, init bearing 120.4°)**: spotted at bearing=39.4°, right at
  the `SCAN_ALIGNED_THR_DEG=40°` edge (champion spotted the same scene more centrally).
  Exiting scan on a marginal, near-threshold bearing estimate committed the robot to a
  slightly-off heading that never self-corrected (no re-scan once committed) — true
  distance grew monotonically from 2.53m to 5.89m over the rollout, plateauing at an
  arena wall. Consistent with the same reduced-angular-resolution mechanism as the demo
  regression: a wider FOV's per-pixel angular value is coarser, so a borderline
  detection is more likely to carry a small-but-consequential bearing error.

### Close-range visibility-to-stop
Closed-loop instrumented rollouts (`code/bench_widefov_visibility.py`, real
`Inferencer`-driven approach with gait dynamics, not the static bench) logging true
GT distance at the last cycle where `ground()` still detected the target:

| Scene | stop_r | last-seen true dist |
|---|---|---|
| easy (5 eps) | 0.60m | 0.57–0.99m (min 0.571m) |
| demo (2 successful eps) | 0.40m | 0.40m, 0.46m |

CAM-1 keeps the target visible through each skill's own stop radius (just barely for
demo, 0.40–0.46m against a 0.40m threshold) but its floor is **~0.40–0.46m in real
closed-loop conditions**, not the champion's **0.256m**. The static bench's 0.30m figure
doesn't survive contact with real gait sway/self-body motion during an actual walking
approach — exactly the effect CAM-2's proximity camera was purpose-built (steeper 58°
pitch, dedicated self-body rejection tuned against a real close-approach walk) to
handle, and CAM-1's single 42°-pitch camera does not match it.

### Compute cost — CAM-1 is *slower*, not cheaper
| | CAM-2 (champion) | CAM-1 (widefov) |
|---|---|---|
| easy mean ms/step | 24.5 | **44.2** |
| demo mean ms/step | 18.0 | **38.4** |

The plan's premise ("one renderer replacing two, cheaper overall") doesn't hold: CAM-2
already renders only the *active* camera per cycle (never both), so the real
comparison is one 480×360 (or 320×240 proximity) render vs. one 640×480 render — and
the resolution bump needed to preserve far-field detection at a wider FOV costs
roughly 2× the render time. CAM-1's code-simplicity advantage (no handoff state
machine, one camera) is real but is explicitly only a tiebreaker per the gate, and
doesn't apply here since CAM-1 already loses on the primary criteria.

## 5. Decision

Per the gate ("CAM-1 wins ONLY if it matches the champion on all three skills AND
visible-to-stop, with lower complexity as the tiebreak — any regression → CAM-2 stays
champion"): CAM-1 regresses demo (−6.7pp) and search (−6.7pp), has a shallower
close-range floor (~0.40–0.46m vs 0.256m), and is ~2× slower per step. **CAM-2 remains
the champion.** CAM-1 is retained in the codebase only as an opt-in, off-by-default
toggle (`CAMERA_MODE=widefov`) for future reference/research — not activated for
deploy.

**Repo state confirmed:** `CAMERA_MODE` defaults to `'cam2'` (no env var set); verified
with `code/arena.py`/`code/grounding.py` module smokes, `eval_closedloop.py --smoke`
and a real-checkpoint n=1 run (byte-for-byte reproduction of the champion's ep0:
fd=0.56m), and `eval_search.py` n=2 (champion's ep0/ep1 reproduced almost exactly)
— all run with the toggle unset, all after every code edit in this phase.

## 6. Files changed

- `code/arena.py` — `CAMERA_MODE` toggle constant, `WIDEFOV_W/H/FOVY/PITCH`, FOVY
  override in `build_arena()` (gated), `_widefov_rend`/`_widefov_cam` (additive,
  built only in widefov mode), `render_widefov()`, `close()` extended.
- `code/grounding.py` — `MIN_DEPTH_WIDEFOV_M`, `is_widefov` flag threaded through
  `ground()` (depth floor, outlier rejection, corrected un-pitch), same pattern as
  `is_proximity`.
- `code/inferencer.py` — `CAMERA_MODE` import; render-selection branch; Schmitt-
  handoff and bounded-fallback-probe blocks gated `if CAMERA_MODE != 'widefov'`.
- `code/eval_search.py` — **bug fix**: `_run_search_rollout()`'s render/intrinsics
  selection now also respects `CAMERA_MODE` (previously always used cam2's
  `render_grounding()`, silently broken once the model-wide FOVY changed under it).
- `code/bench_widefov_dist.py` (new) — static far/near-range + sign-convention bench.
- `code/bench_widefov_visibility.py` (new) — closed-loop close-range visibility probe.

## 7. Eval artifacts

- `eval/p2_easy_widefov/`, `eval/p2_demo_widefov/`, `eval/p2_search_widefov/` — CAM-1
  final numbers (100.0 / 60.0 / 73.3), search after the eval_search.py fix.
- `eval/p1_easy_cam2_v2/`, `eval/p1_demo_cam2_v2/`, `eval/p1_search_cam2/` — CAM-2
  champion numbers (unchanged, reused from Phase 1 for the A/B diff).
