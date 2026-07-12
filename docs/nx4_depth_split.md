# NX-4 — Depth-Guided Blob Splitting + Component Re-Selection: REJECT

**Date:** 2026-07-09
**Agent:** NX-4 (grounding segmentation experiment, follow-on to `docs/nx3_size_gate.md`)
**Checkpoint:** `checkpoint/goto_best.pt` (unchanged — deploy-side only)
**CAMERA_MODE:** not set (cam2 champion). Baseline: demo/classical 10/15 = 66.7%
(`eval/nx2_defaults_confirm`, failing eps 0, 2, 4, 5, 12).

## TL;DR — VERDICT: REJECT both toggles (GROUND_SPLIT and GROUND_SHAPE stay default OFF)

- **Idea (from NX-3's §4 root cause):** the merged true-target+wall stripes that
  killed the M6 size gate should be separable by DEPTH — a target at ~4-7m fused
  with a wall at a different depth should show a bimodal depth histogram inside
  one connected HSV component. Split each component at depth gaps >= ~0.5m,
  re-select among the now depth-pure candidates with a physical-size
  plausibility preference, and (secondary toggle) arbitrate ep12's
  size-identical ball-vs-cube twin by contour circle-fill.
- **The central hypothesis is FALSIFIED by direct measurement, not just by the
  closed-loop gate** (§3): instrumenting the split pass on live ep1/ep13/ep0
  frames shows the merged stripes' depth is a CONTINUOUS oblique ramp
  (e.g. 6.41→7.06m, std ≈ 0.20m, ZERO empty bins at 0.15m histogram
  resolution — a single cluster). The wall is viewed at grazing incidence, so
  its depth blends smoothly through the target's depth; there is no 0.4-0.6m
  gap to split on. Across a full instrumented 15-episode run
  (demo 0/1/2/3/5/6/9/12/13 + easy 0-5), **zero raw-path splits fired
  anywhere**.
- **Where the merged stripes actually live:** not the primary HSV-contour path
  but the V3 DEPTH-FG RESCUE path in `code/grounding.py` — in wall-hue
  episodes the full-frame background blob is rejected, then FG rescue finds
  *fragments* (several wall stripes + sometimes the true target) and picks by
  area only. The split + size-plausible re-selection was extended there too
  (that is where it took effect).
- **Closed-loop demo gate `GROUND_SPLIT=1` (seed 999, n=15): 9/15 = 60.0%,
  both runs — a confirmed regression** (baseline 10/15). Zero target episodes
  fixed (0/2/5/12 all still FAIL at essentially baseline final_dists);
  **ep13 newly broken, reproduced 2/2** (`eval/nx4_demo_split` fd=8.36,
  `eval/nx4_demo_split_rerun` fd=8.35, vs baseline SUCCESS fd=0.36).
- **ep13's break is the same structural overlap NX-3 documented, resurfacing
  at the re-selection level** (§5): at step 0 the FG-rescue re-selection
  prefers the size-plausible LEFT wall fragment (36x10px at x=14, bearing
  +31° — AWAY from the target; it passes the band only via the clipped-axis
  carve-out at the image margin) over the size-IMplausible RIGHT fragment
  (79x10px, ratio 5.1x, bearing -26.6° — pointing AT the target, and exactly
  what the passing baseline seeds its lock from). Physical size cannot
  observe which wall fragment's bearing is correct — the discriminating
  variable is still bearing-vs-target, invisible to size/geometry.
- **GROUND_SHAPE probe on ep12** (`GROUND_SPLIT=1 GROUND_SHAPE=1`, standalone
  matched-harness replay): **still FAIL fd=5.65m** (baseline 6.25-6.73). The
  re-selection machinery demonstrably runs in ep12 (27 re-selection events in
  the instrumented rollout — the most of any episode) but never converts the
  outcome; the hijack still enters via the CAM-2 handoff carve-out with a
  full-frame proximity blob for which no second size-plausible candidate
  exists to arbitrate against (the conservative two-plausible-candidates
  precondition, by design, never fires there).
- **Cross-skill gates not run** — the demo gate already fails both KEEP bars
  (no target fixed AND a reproduced regression), same early-stop reasoning as
  NX-3.
- **Default-off code path confirmed safe:** two full no-env demo runs
  (`eval/nx4_default_off_confirm`, `eval/nx4_default_off_rerun`) both scored
  9/15 with **ep1** the sole delta vs baseline — ep1 is the exact episode
  `docs/nx3_size_gate.md` already documented as "marginal, occasionally
  bistable... can flip on its own between identical no-change runs" (NX-3's
  own no-env confirms flipped it too: PASS/FAIL/PASS across three M6-off
  runs). A standalone matched-harness ep1 replay with the shipped default-off
  code passed **3/3** (steps 917-940, fd=0.37, matching the historical
  936-step success), and the off path is also structurally inert: every
  NX-4 branch is guarded by `if GROUND_SPLIT:`; with the toggle unset the
  only executed additions are a `len()` call and four always-None diagnostic
  fields. ep13 and every other episode match baseline in both no-env runs.
- **NOT synced to staging** (`VLA_mujoco_unitree/`): the
  brief conditions sync on adoption; verdict is REJECT.

---

## 1. Design (what was built — all in `code/grounding.py`, additive)

- **`GROUND_SPLIT`** (default OFF): depth-guided splitting + re-selection.
  - `_histogram_depth_clusters(depth_vals)` — 1-D histogram
    (`GROUND_SPLIT_BIN_M=0.15`), contiguous-occupancy runs merged unless
    separated by an empty-bin gap >= `GROUND_SPLIT_GAP_M=0.5m` (task brief's
    0.4-0.6m band). Vectorized numpy.
  - `_split_component_by_depth(comp_mask, depth_map, ...)` — assigns each
    valid-depth pixel to its depth cluster (`np.searchsorted` on cluster
    edges); invalid-depth pixels are propagated from their nearest assigned
    2-D neighbour (`scipy.ndimage.distance_transform_edt` nearest-index; if
    scipy is missing those pixels are simply dropped — safe no-op).
    Conservative no-ops: < `GROUND_SPLIT_MIN_SAMPLES=12` valid samples, or
    < 2 clusters, or < 2 surviving pieces (`GROUND_SPLIT_MIN_PIECE_PX=25`).
  - `_split_contours_by_depth(...)` — expands the candidate contour set;
    applied to BOTH the raw HSV contour list and (second commit within this
    closure, after §3's finding) the FG-rescue contour list.
  - Re-selection: on the post-split scored candidate set, every candidate
    gets a cheap median depth (`_quick_candidate_depth`, the same
    erode-then-median as the main pipeline) and a per-axis size-plausibility
    verdict (`_physical_size_plausible`, reusing NX-3's Config F rules with
    an independently tunable band `GROUND_SPLIT_SIZE_LO/HI = 0.08/2.5`);
    candidates sort by `(plausible DESC, existing composite score DESC)` —
    i.e. size-plausible preferred, existing V5 scoring otherwise unchanged.
    In FG rescue the same preference wraps the legacy area-only pick, only
    when > 1 candidate survives (never rejects a sole candidate).
- **`GROUND_SHAPE`** (default OFF, requires GROUND_SPLIT): ball-vs-cube
  arbitration by `_circle_fill_score` = contour_area /
  min-enclosing-circle area (ball silhouette ≈ 0.97 measured on synthetic;
  cube ≈ 2/π = 0.637 exactly). Fires ONLY when >= 2 size-plausible
  candidates exist and the instructed shape is ball/cube (never rejects the
  only candidate on shape) — the target spec reaches `ground()` as the
  `target_shape` argument from the scene's instruction parse, so no plumbing
  changes were needed.
- Diagnostics: additive `GroundingResult.n_raw_components / n_candidates /
  split_reselected / size_plausible` fields (None unless GROUND_SPLIT=1).
- 23/23 CPU unit tests pass (scratchpad `test_nx4_split_unit.py`): histogram
  clustering (single/dual/triple populations, sub-gap merge), synthetic
  merged-blob split, depth-pure no-op, too-few-samples no-op, invalid-depth
  hole propagation to the nearest cluster, contour expansion end-to-end,
  circle-fill discrimination, and toggle-off inertness spot checks.
  `python code/grounding.py` smoke passes identically with the toggle off,
  on, and on+shape.

## 2. Mechanism-level outcome table (instrumented BEFORE the closed-loop gates, per brief §4)

Instrumented `ground()` (same wrapper pattern as NX-3's `calibrate_m6.py`)
across demo eps 0/1/2/3/5/6/9/12/13 + easy 0-5, `GROUND_SPLIT=1`, seed 999
(scratchpad `nx4_instrument.py` → `nx4_split_calib.json`):

| ep | hypothesis | measured mechanism outcome | closed-loop |
|---|---|---|---|
| 0 (cyan cone) | false stripes split + rejected | **no split fires** — its stripes (e.g. 225x10px at d≈6.4-7.1m) are depth-CONTINUOUS (single cluster); the ~5.4m sliver lock persists unchanged | FAIL fd=3.38 (baseline 3.35) — NOT fixed |
| 2 (blue cone) | wall stripe split + rejected | no split fires; FG-rescue fragments all wall-family | FAIL fd=10.70/10.64 (baseline 10.50) — NOT fixed |
| 5 (cyan ball) | stripes split + rejected | no split fires (71 hits, 0 split events) | FAIL fd=3.20/3.27 (baseline 4.17) — NOT fixed |
| 12 (cyan cube) | twin arbitration after split | re-selection fires 27x (most of any ep) but never at the handoff-hijack moment; shape toggle probed separately (§6) | FAIL fd=5.96/6.74 (baseline 6.25) — NOT fixed |
| 1 (cyan cube, merged-blob passer) | true component selected post-split (better) or lock lost (worse) | no split fires on its merged stripes (depth ramp 6.41→7.06m, std 0.20, single cluster at 0.15m bins); lock diet unchanged | SUCCESS 2/2 (903/925 steps) — not endangered |
| 13 (blue ball, merged-blob passer) | same | **BROKEN by FG-rescue re-selection** (see §5) — the step-0 seed flips from the bearing-correct implausible fragment to a bearing-wrong "plausible" fragment | **FAIL 2/2 (fd 8.36/8.35; baseline SUCCESS 0.36)** |
| easy 0-5 | no regressions | 0 split events, 0 re-selections, all SUCCESS in instrumented run | (cross-skill gate not reached) |

## 3. The measured falsification (why no splits ever fire)

Direct per-component depth histograms on live frames (scratchpad
`nx4_diag_depth_hist.py`), first 120 steps of eps 1/13/0 — every substantial
candidate component (80-20000px), 38+13+21 components examined:

- ep1's merged stripes (74x10px, the exact 4.7-4.8x-ratio objects NX-3
  §4 blamed): depth spans 6.41-7.06m **continuously** — std 0.20m, single
  cluster, zero empty bins at 0.15m resolution. The "cube fused with wall"
  is geometrically a wall seen at grazing incidence whose depth sweeps
  smoothly through the target's depth. No 0.4-0.6m gap exists.
- ep13: same signature (stripes 6.4-7.2m continuous; the separate TRUE ball
  blob at 4.2-4.5m is its own already-separate component, std 0.03).
- ep0: stripes 6.4-7.7m continuous (std 0.20-0.35), single cluster.
- Full 15-episode instrumented run: `n_split_events=0` on every accepted
  detection in every episode.

Consequence: the split machinery is a structural no-op on this arena's real
failure modes. What remained live was only the size-plausibility
RE-SELECTION preference on multi-candidate sets (mostly in FG rescue) — and
that is exactly the piece NX-3 already proved cannot discriminate
bearing-useful from bearing-useless wall fragments. §5 shows it now breaking
ep13 through that exact blind spot.

## 4. Demo/classical gate — before → after

**Run:** `GROUND_SPLIT=1 python code/eval_closedloop.py --checkpoint
checkpoint/goto_best.pt --arch A --difficulty demo --n 15 --device cuda
--out eval/nx4_demo_split --no-render --goal-source classical
--vel-source predicted --seed 999` (1-ep smoke crash-free first; GPU idle).

### Result: 9/15 = 60.0% both runs (baseline 10/15 = 66.7%)

| ep | baseline (`nx2_defaults_confirm`) | SPLIT run 1 (`eval/nx4_demo_split`) | SPLIT rerun (`eval/nx4_demo_split_rerun`) | verdict |
|---|---|---|---|---|
| 0 | FAIL fd=3.35 | FAIL fd=3.38 | FAIL fd=3.38 | target NOT fixed |
| 1 | SUCCESS fd=0.36 | SUCCESS fd=0.37 | SUCCESS fd=0.36 | unchanged |
| 2 | FAIL fd=10.50 | FAIL fd=10.70 | FAIL fd=10.64 | target NOT fixed |
| 3 | SUCCESS | SUCCESS | SUCCESS | unchanged |
| 4 | FAIL fd=10.04 | FAIL fd=10.29 | FAIL fd=10.21 | unchanged (never a target) |
| 5 | FAIL fd=4.17 | FAIL fd=3.20 | FAIL fd=3.27 | target NOT fixed |
| 6-11 | all SUCCESS | all SUCCESS | all SUCCESS | unchanged |
| 12 | FAIL fd=6.25 | FAIL fd=5.96 | FAIL fd=6.74 | target NOT fixed |
| **13** | **SUCCESS fd=0.36, 594 steps** | **FAIL fd=8.36, 1400 steps** | **FAIL fd=8.35, 1400 steps** | **newly broken, reproduced** |
| 14 | SUCCESS | SUCCESS | SUCCESS | unchanged |

Two identical 9/15 outcomes with an identical failing set — a real mechanism
effect, not the single-flip noise class (`docs/cam_p0.md` protocol satisfied
by the rerun).

## 5. ep13 root cause (instrumented, not guessed)

Matched-harness standalone replays of ep13 with per-call detection logging
(scratchpad `nx4_diag_ep13.py` / `nx4_diag_ep13_baseline.py`):

- **Baseline step 0:** FG rescue picks the largest fragment — 79x10px at
  x=373, phys_w ratio 5.1x (size-implausible), **bearing -26.6°, toward the
  target**. Steps 20-90 then lock the TRUE ball (dist ≈ 4.0-4.2m, bearing
  ≈ -30°, phys ratios ≈ 1.0) and the episode converges (fd 1.17m by step
  500; full run SUCCESS at 591-594 steps).
- **GROUND_SPLIT step 0:** the re-selection evaluates the same fragment set
  and prefers the only "size-plausible" one — 36x10px at x=14, phys_w=0.56m.
  It passes the band **only via the clipped-axis carve-out** (x=14 == the 3%
  left margin, so the width lower bound is skipped and the upper bound
  0.56 <= 0.24*2.5*... applies at far depth) — and its **bearing is +31°,
  away from the target**. The EMA seeds on the wrong side; the trajectory
  diverges from the first cycle (gt_dist stalls ≈3.6-3.8m by step 500 vs
  baseline's 1.17m) and never recovers (fd 8.35-8.36 at 1400 steps).
- The true ball IS briefly re-acquired (steps 10-50, dist 4.0-4.2, bearing
  -30°) but the corrupted seed + subsequent wall-top stripe locks (452px-wide
  ceiling stripes at 8.5-8.8m, bearing -0.1°) dominate the rest.

This is NX-3 §4's overlap theorem restated one level up: among wall
fragments, physical size selects a *random* fragment with respect to
bearing-usefulness. ep13's baseline pass depends on the size-IMplausible
fragment winning. Any size-preference rule that helps ep0/ep2/ep5's stripe
rejection necessarily reshuffles ep13's seed lottery. No
`GROUND_SPLIT_SIZE_LO/HI` constant fixes this: the winning wrong fragment
passed through the clipped-axis carve-out (needed for legitimate partial
visibility, per NX-3's calibration), and the bearing-correct fragment is
unambiguously implausible at 5.1x under any band that retains rejection
power. Tuning was therefore not attempted — the same
structurally-falsified reasoning precedent as `docs/nx2_iso.md` M1/M3 and
`docs/nx3_size_gate.md` §4.

## 6. GROUND_SHAPE / ep12 (honest scoping)

`GROUND_SPLIT=1 GROUND_SHAPE=1` standalone matched-harness ep12 replay
(scratchpad `nx4_ep12_shape_check.py`): **FAIL[didnt-reach] fd=5.65m,
1400 steps** — indistinguishable from baseline (5.65-6.74 across runs).
Why it cannot fire at the decisive moment: the hijack is accepted at the
CAM-2 handoff as a full-frame proximity blob (~60000px², d≈0.97m,
docs/nx2_iso.md §M3) — at that cycle there is ONE candidate, and the
conservative precondition (two size-plausible same-color candidates)
correctly refuses to arbitrate. Earlier in the episode, when both the cube
and the ball are visible and plausible, the arbitration does fire (part of
ep12's 27 re-selection events) — but per FA-1's trace the approach is
already correct for the first ~600 steps, so those events change nothing.
The staged gate condition ("if demo passes, add GROUND_SHAPE") was never
reached; the standalone probe is reported for completeness. The circle-fill
scorer itself is validated (0.968 vs 0.637 on synthetic silhouettes) and
remains available opt-in for any future mechanism that can reach the
handoff moment with a real candidate pair.

## 7. Default-off confirmation

- `eval/nx4_default_off_confirm` + `eval/nx4_default_off_rerun` (plain
  command, no env vars): both 9/15, failing sets {0,1,2,4,5,12} — i.e.
  baseline's failing set plus **ep1**, the episode `docs/nx3_size_gate.md`
  TL;DR already documents as bistable across identical no-change runs
  (NX-3's three M6-off runs went PASS/FAIL/PASS on it).
- Standalone matched-harness ep1 replay with the shipped default-off code:
  **3/3 SUCCESS** (steps 940/917/917, fd=0.37 — matching the historical
  936-step success signature).
- Structural inertness: with `GROUND_SPLIT` unset, the only executed NX-4
  code is `n_raw_components = len(contours)` plus four always-None dataclass
  fields; every behavioral branch (`_split_contours_by_depth`, re-selection,
  FG-rescue preference, GROUND_SHAPE) is inside `if GROUND_SPLIT:` guards.
  `_physical_size_plausible`'s new `lo`/`hi` keywords default to the exact
  M6 constants, leaving LOCK_M6 semantics untouched.
- ep13 (the episode NX-4's ON-path breaks) is SUCCESS in both no-env runs
  (591/594 steps, fd=0.36) — the regression is confined to the toggle.

## 8. Files changed / kept / NOT synced

- `code/grounding.py` — additive: `GROUND_SPLIT`/`GROUND_SHAPE` toggles
  (default OFF), `GROUND_SPLIT_*` constants, `_histogram_depth_clusters`,
  `_split_component_by_depth`, `_split_contours_by_depth`,
  `_quick_candidate_depth`, `_circle_fill_score`, the gated split +
  re-selection blocks in `ground()` (main path + FG rescue), four additive
  `GroundingResult` diagnostic fields, and overridable `lo`/`hi` keywords on
  `_physical_size_plausible` (defaults = M6 band; M6 call sites unchanged).
- No changes to `lock_mgmt.py`, `inferencer.py`, `eval_search.py`,
  `scene.py`, `arena.py`.
- **NOT synced** to `VLA_mujoco_unitree/code/` — sync is
  conditioned on adoption; verdict is REJECT. (Staging note carried forward
  from NX-3: whoever next syncs `grounding.py` will carry the default-off
  M6 + NX-4 code along; both are provably inert unless their env vars are
  set.)
- Diagnostic/calibration scripts live in scratchpad only:
  `test_nx4_split_unit.py` (23 tests), `nx4_instrument.py`,
  `nx4_split_calib.json` (15-episode instrumented dataset),
  `nx4_diag_depth_hist.py` (the falsifying depth-continuity measurement),
  `nx4_diag_ep13.py` / `nx4_diag_ep13_baseline.py` (ep13 break trace),
  `nx4_ep12_shape_check.py`, `nx4_ep1_bistable_check.py`.

## 9. Artifacts

- `eval/nx4_demo_split/` — GROUND_SPLIT=1 demo gate run 1 (9/15) + run.log
- `eval/nx4_demo_split_rerun/` — stability rerun, identical condition (9/15)
- `eval/nx4_default_off_confirm/`, `eval/nx4_default_off_rerun/` — no-env
  confirm runs (9/15 each; sole delta = bistable ep1, standalone 3/3 PASS)
- scratchpad `nx4_split_calib.json` — instrumented 15-episode dataset

## 10. What this closes off (guidance for the next attempt)

NX-3 ended with: "any future physically-principled detection filter must
first make the detections themselves physically honest (e.g. splitting
merged blobs...)". NX-4 tested exactly that and found the merged blobs are
**not splittable by depth in this arena** — the merge is an oblique-wall
depth *continuum*, not a bimodal composite. The remaining honest routes to
the 0/2/5/12 family are (a) bearing/temporal corroboration (does a
candidate's bearing stay consistent with the motion-integrated target
estimate?), which is the one variable NX-3/NX-4 both identified as the true
discriminator; and (b) for ep12 specifically, corroborating the post-handoff
detection before the CAM-2 carve-out trusts it unconditionally
(`docs/nx2_iso.md` §M3's flagged future work) — with the caveat documented
there about the ep13/cam_p3/cam_p4 deadlock class.
