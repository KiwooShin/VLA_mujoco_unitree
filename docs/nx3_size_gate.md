# NX-3 — Physical-Size Plausibility Gate (M6): Calibration, Gate, REJECT Verdict

**Date:** 2026-07-09
**Agent:** NX-3 (grounding discrimination experiment)
**Checkpoint:** `checkpoint/goto_best.pt` (unchanged — deploy-side only)
**CAMERA_MODE:** not set (cam2 champion). Baseline: demo/classical 10/15 = 66.7%
(`eval/nx2_defaults_confirm`, failing eps 0, 2, 4, 5, 12; `docs/nx2_final.md`).

## TL;DR — VERDICT: REJECT (LOCK_M6 stays default OFF)

- **Idea:** targets are known primitives with known dimensions (`code/arena.py`
  `SHAPES` + `build_arena()` geom formulas). Back-project each candidate blob's
  pixel bbox extent through its own median depth and the camera focal length
  (pinhole: `real_size = pixel_extent * depth / focal`) and reject blobs whose
  physical width/height is implausible for the instructed shape. Grounding-level
  (inside `ground()`), so it covers `inferencer.py` AND `eval_search.py`
  automatically and sits UPSTREAM of the CAM-2 handoff carve-out that shields
  ep12's hijack from M3.
- **Calibration on 781 real accepted detections (15 episodes) found clean
  STATIC separation for eps 0/2/5** (their false blobs measure 3.3-47.7x nominal
  width at their own reported depths; true unclipped detections never exceed
  1.9x) **and NO separation for ep12** (its hijacker is a real 0.24m cyan ball
  vs a 0.24m cyan cube target — physically plausible in every way size can see).
- **Closed-loop demo gate (LOCK_M6=1, n=15, seed 999): 8/15 = 53.3% —
  a confirmed regression.** Zero target episodes fixed (0/2/5/12 all still
  FAIL); two previously-passing episodes newly broken (**ep1** fd 0.37→6.44m,
  **ep13** fd 0.37→3.18m), both reproduced near-identically on a full
  independent rerun (`eval/nx3_demo_rerun`: 8/15, same failing set, ep1
  fd=5.90, ep13 fd=3.73) — real mechanism effects, not the single-flip noise
  class.
- **Root cause of the regression is a fundamental overlap, not a tunable
  constant** (see §4): passing episodes' locks are partly fed by *merged* blobs
  (true target + wall/distractor fused into one wide stripe by HSV overlap)
  whose physical size is exactly as implausible as the false-lock stripes M6
  targets — ep1's bearing-USEFUL merged stripes measure 4.7-4.8x nominal width,
  inside ep0's bearing-USELESS false-stripe range of 3.4-6.3x. The only
  difference between the two is whether the stripe's centroid bearing happens
  to point at the target — a quantity physical size cannot observe. No size
  band separates them; tuning was therefore not attempted (same
  already-falsified-structurally reasoning as `docs/nx2_iso.md` M1/M3).
- **Code kept as default-OFF opt-in** (`LOCK_M6=1` to enable), same pattern as
  the REJECTed M2/M4/M5 toggles in `code/lock_mgmt.py`. No-env-var demo
  confirm: the first run (`eval/nx3_default_off_confirm`) showed 9/15 with a
  single flip (ep1) — exactly the single-flip EGL-non-determinism noise class
  `docs/cam_p0.md` documents, so it was rerun once per protocol
  (`eval/nx3_default_off_rerun`): **10/15, byte-identical failing set to
  baseline (0, 2, 4, 5, 12), zero flips** — the default-off code path is
  confirmed inert (it is also provably so: the only behavioral branch is
  guarded by `if LOCK_M6`; everything else is pure arithmetic plus two
  additive diagnostic fields). Side-finding worth recording: ep1 can flip on
  its own between identical no-change runs — it is a marginal, occasionally
  bistable episode (consistent with §4: its lock rides on merged marginal
  blobs). This slightly softens the ep1 attribution in §3 (though 2/2
  M6-on failures at fd 5.9-6.4 vs 1-in-many baseline flips still points at
  M6); the ep13 regression is unambiguous (ep13 is rock-stable at SUCCESS
  fd=0.36-0.37 across all three M6-off runs and broke in both M6-on runs).
  The additive `GroundingResult.phys_w`/`phys_h` diagnostic fields remain
  populated on every detection for future work.
- **Not synced to staging** (`VLA_mujoco_unitree/`): the task
  brief conditions sync on adoption (gates passing + >=1 target ep fixed);
  neither held.

---

## 1. Design (what was built)

All in `code/grounding.py` (no changes to `lock_mgmt.py`, `inferencer.py`, or
`eval_search.py` — both call sites inherit the gate through `ground()` itself):

- **`NOMINAL_DIMS_M`** — physical (width, height) per shape, derived from
  `arena.build_arena()`'s actual geom formulas, not just the `SHAPES` table:
  ball/cube 0.24x0.24m; cylinder 0.22m wide x 0.352m tall (half-height =
  `hs*1.6`); cone 0.26m wide x 0.5434m tall (base cylinder `hs*2.2` + tip box
  → total `size*2.09`). Note `SHAPES`' "size" values are full
  diameters/edge-lengths in `build_arena()` (each branch halves them into
  `hs`), despite the "half-size" naming in the module docstring.
- **`_estimate_physical_size(bbox_w_px, bbox_h_px, depth_m, fx, fy)`** —
  pinhole back-projection. Populates the new additive
  `GroundingResult.phys_w`/`phys_h` fields on every accepted detection,
  regardless of toggle state (used for calibration; free diagnostics).
- **`_physical_size_plausible(...)`** — per-axis band check (final "Config F"
  rule, calibrated in §2-§3):
  - Unclipped axis: `0.08 <= ratio <= 2.5` of nominal (both bounds).
  - Clipped axis (bbox touches image edge / usable-region margin on that
    axis): lower bound always skipped (partial visibility truncates measured
    extent — a bottom-clipped near-field cone must not be rejected for reading
    too short; per task brief). Upper bound still enforced when
    `depth >= M6_NEAR_DEPTH_M = 1.2m` (clipping can only SHRINK measured
    extent, so measured > HI is definitive even when clipped); fully exempt
    below 1.2m (near-field depth is corrupted — see §2 finding 3).
  - Fails OPEN on unknown shape / degenerate depth or focal.
- **`LOCK_M6`** env toggle, default OFF. When ON, an implausible best blob is
  returned as `not_visible=True` — it can never seed or refresh a lock, in
  either call site, including at the CAM-2 handoff cycle (the gate runs before
  the detection ever reaches `lock_mgmt`'s carve-out).
- 44/44 CPU unit tests pass (scratchpad `test_m6_unit.py`): nominal dims,
  round-trip math, accept at 3-9m for all 4 shapes, reject
  sliver/oversize/wall-stripe classes, edge-clip skip semantics (bottom, top,
  left/right, full-frame BTLR at near depth), fail-open, and `ground()`-level
  toggle wiring. `python code/grounding.py` smoke passes unchanged with the
  toggle off.

## 2. Calibration (before gating): 781 accepted detections, 15 episodes

Instrumented `ground()` (wrapper around `code.inferencer.classical_ground`,
LOCK_M6 off, shipped M1/M3 defaults on) across demo passing eps 1/3/6/9/13,
demo failing eps 0/2/5/12, easy eps 0-5 (seed 999). Each accepted detection
classified TRUE (|reported dist − GT dist to true target| <= 1.0m), FALSE
(> 1.5m), or ambiguous. Data: scratchpad `m6_calib.json` (+ `calibrate_m6.py`,
`analyze_m6*.py`).

Key numbers (width ratio = `phys_w / nominal_w`):

| Population | n | min | p5 | median | p95 | max |
|---|---|---|---|---|---|---|
| TRUE width-ratios (all eps) | 367 | 0.105 | 0.42 | 1.00 | 6.9* | 41.5* |
| FALSE width-ratios (all eps) | 414 | 0.12 | 0.97 | 4.56 | 47.7 | 49.8 |
| TRUE height-ratios (unclipped) | 281 | 0.21 | 0.31 | 1.00 | 1.41 | 1.89 |

*TRUE tail >2.5 is (a) full-frame proximity blobs at <1.2m (finding 3) and
(b) merged true+wall stripes (finding 4 — the eventual killer).

Findings that shaped the final rule:

1. **The task brief's 0.4x lower bound would break easy.** Easy ep2 (cyan
   cube, currently 100%) has TRUE early detections down to ratio 0.15 (9/15 of
   its true hits fall below 0.4); easy ep5 down to 0.105. → LO=0.08
   (near-vestigial backstop; all discriminative power is on the high side).
2. **Naive "skip clipped axes entirely" leaks the biggest false blobs.** ep2's
   wall blob (12.4m physical width, ratio 47.7) and ep5's false blobs
   (3.3-5.1x) are LR-clipped — a full skip accepts them (ep5: only 4/59
   rejected). Enforcing the upper bound on clipped axes at far depth fixes
   this (clipping only truncates).
3. **But full enforcement breaks final approaches.** In PASSING demo ep1 and
   easy eps 2/4, the proximity camera's legitimate final-approach blob is
   full-frame (clipped all 4 edges, ~60000px², median depth saturated ~0.97m
   while true range keeps closing) and reads 3.5-7.6x nominal. → clipped axes
   fully exempt below `M6_NEAR_DEPTH_M=1.2m`. (This same signature is ep12's
   hijack — see §4.)
4. Final "Config F" static confusion per episode (LO=0.08, HI=2.5,
   near-exempt <1.2m): TRUE rejections 0 in every passing episode except ONE
   borderline blob in ep13 (a wall-merged stripe at err=0.97m from GT — barely
   classified "true"); FALSE rejections 66/66 (ep0), 21/21 (ep2), 58/59 (ep5),
   3/139 (ep12), 35/35 (ep13's bearing-correct wall blobs — flagged
   pre-gate as the main risk).

### Separation summary (static, per target episode)

| Target ep | False-blob signature | Separation |
|---|---|---|
| ep0 (cyan cone) | wide shallow stripes, ratio 3.4-6.3 wide / 0.29-0.38 tall at d=5.5-6.2m | **CLEAN** (66/66 rejected, high side) |
| ep2 (blue cone) | LR-clipped wall stripe, ratio 47.7 at d=8.8m (+ merged FG-rescue blobs 6.5-8.4) | **CLEAN** (21/21 + 13/17 of its "true"-labelled merged blobs) |
| ep5 (cyan ball) | LR-clipped stripes, ratio 3.3-5.1 at d≈6.1m | **CLEAN** (58/59 rejected) |
| ep12 (cyan cube) | pre-handoff: the real cyan-ball distractor at its true depth, ratio ≈1.0; post-handoff: full-frame proximity blob at 0.97m (same signature as legitimate final approaches) | **NONE** (3/139 rejected — the hijacker is a physically plausible real object; size cannot separate it) |

## 3. Demo/classical gate — the static separation does NOT survive the closed loop

**Run:** `LOCK_M6=1 python code/eval_closedloop.py --checkpoint
checkpoint/goto_best.pt --arch A --difficulty demo --n 15 --device cuda --out
eval/nx3_demo --no-render --goal-source classical --vel-source predicted
--seed 999` (1-ep smoke crash-free first; GPU checked idle — CX-6's render job
had finished).

### Result: 8/15 = 53.3% both runs (baseline 10/15 = 66.7%)

| ep | baseline (`nx2_defaults_confirm`) | M6 run 1 (`eval/nx3_demo`) | M6 rerun (`eval/nx3_demo_rerun`) | verdict |
|---|---|---|---|---|
| 0 | FAIL fd=3.35 | FAIL fd=3.01 | FAIL fd=3.05 | target NOT fixed (fd slightly better, still fails) |
| **1** | **SUCCESS fd=0.36** | **FAIL fd=6.44, 1400 steps** | **FAIL fd=5.90** | **newly broken, reproduced** |
| 2 | FAIL fd=10.50 | FAIL fd=7.77 | FAIL fd=7.87 | target NOT fixed (walks away less far, still fails) |
| 3 | SUCCESS | SUCCESS | SUCCESS | unchanged |
| 4 | FAIL fd=10.04 | FAIL fd=10.39 | FAIL fd=10.20 | unchanged (never a target) |
| 5 | FAIL fd=4.17 | FAIL fd=4.51 | FAIL fd=4.04 | target NOT fixed |
| 6-11 | all SUCCESS | all SUCCESS | all SUCCESS | unchanged |
| 12 | FAIL fd=6.25 | FAIL fd=6.39 | FAIL fd=6.02 | target NOT fixed (as predicted by §2: no separation) |
| **13** | **SUCCESS fd=0.37, 602 steps** | **FAIL fd=3.18, 1400 steps** | **FAIL fd=3.73** | **newly broken, reproduced** |
| 14 | SUCCESS | SUCCESS | SUCCESS | unchanged |

Two flips (> the "exactly one" noise threshold of `docs/cam_p0.md`), so the
full 15-episode condition was rerun once per the M4-isolation precedent
(`docs/nx2_iso.md`): identical 8/15, identical failing set, near-identical
final_dists — the regressions are mechanism effects, not jitter.

**Cross-skill gates were not run** — the demo gate already fails both KEEP
bars (no target fixed AND regressions introduced), so there is no adoption
candidate to validate further; burning easy/search GPU time on a
REJECT-verdicted config would not change the outcome.

## 4. Why (root cause, from the calibration records — not guessed)

The closed-loop failure was traceable in the same calibration data that
predicted the static separation:

- **ep1** exits its scan and seeds its early lock from *merged* blobs (cyan
  cube + cyan-tinted wall fused into one 74x10px stripe at step 0-30: ratio
  4.7-4.8 wide, bearing +9.5° — pointing AT the target). M6 rejects them
  (unclipped, far, > 2.5). In the recorded (M6-off) trajectory a clean
  correctly-sized blob appears by step 20 — but with M6 ON the trajectory
  diverges from the very first rejected cycle (delayed scan-exit, sparser EMA
  feed), and the episode never converges (fd 5.9-6.4 at 1400 steps vs 936-step
  success).
- **ep13** passes at baseline while ~80% of its accepted mid-episode
  detections are wall blobs at d≈9.6m (GT ≈3.3m) whose ratio is 19-24x — but
  whose *bearing* tracks the true target (the wall stripe sits behind the
  ball). M6 rejects all 35; the lock starves mid-approach and the episode
  fails at fd 3.2-3.7m.
- **The overlap is fundamental:** ep1's bearing-useful merged stripes
  (4.7-4.8x) sit INSIDE ep0's bearing-useless false-stripe range (3.4-6.3x),
  and ep13's (19-24x) below ep2's (47.7x) but far above any TRUE unclipped
  blob. Every band that rejects ep0/ep5's false population also rejects ep1's
  lock diet; every band loose enough to spare ep1 (HI>=6.3) passes all of
  ep0's false blobs. Aspect-ratio variants fail the same way (ep0's false
  stripe and ep1's merged-true stripe are geometrically the same object
  class). The discriminating variable is whether the stripe's centroid bearing
  points at the target — invisible to any size/geometry feature. **No M6
  constant fixes this; tuning was not attempted** (structurally falsified,
  same reasoning precedent as `docs/nx2_iso.md` §M1/§M3).
- A deeper irony worth recording: this codebase's passing episodes *depend*
  on size-implausible detections (merged stripes, saturated-depth full-frame
  proximity blobs) being accepted. Any future "physically principled"
  detection filter must first make the detections themselves physically
  honest (e.g. splitting merged blobs, fixing near-field median depth) before
  a physical plausibility check can be safe.

## 5. What ep12 taught (honest scoping)

The brief hoped a grounding-level size check would catch ep12's
handoff-carve-out hijack. Measured reality: the hijacker is the scene's real
cyan BALL distractor (0.24m) vs the cyan CUBE target (0.24m) — identical
nominal size, detected at its own true depth with ratio ≈1.0 both before the
handoff and (as a legitimately full-frame near-field blob) after it. Physical
size is the wrong axis for ep12; the correct discriminators remain shape
(V5's shape score already tries; the ball wins anyway once close/large) or
handoff corroboration (flagged as future work in `docs/nx2_iso.md` §M3).

## 6. Files changed / kept / NOT synced

- `code/grounding.py` — additive: `NOMINAL_DIMS_M`, `_estimate_physical_size`,
  `_physical_size_plausible`, `M6_SIZE_BAND_LO/HI`, `M6_NEAR_DEPTH_M`,
  `LOCK_M6` toggle (default OFF), gate call in `ground()` after `depth_m`,
  `GroundingResult.phys_w/phys_h` fields. With `LOCK_M6` unset, the gate
  branch is dead code and the fields are pure diagnostics; the no-env demo
  confirm (first run one ep1 noise-flip, rerun per protocol:
  `eval/nx3_default_off_rerun`) reproduces the pre-NX-3 baseline 10/15 with
  the byte-identical failing set (0, 2, 4, 5, 12).
- NOT synced to `VLA_mujoco_unitree/code/` — the brief
  conditions sync on adoption; verdict is REJECT. (The staging copy therefore
  stays at NX-2 state; whoever next syncs `grounding.py` should be aware it
  will carry this default-off M6 code along.)
- Diagnostic/calibration scripts live in scratchpad only (never committed):
  `test_m6_unit.py` (44 tests), `calibrate_m6.py`, `analyze_m6.py`,
  `analyze_m6_v2.py`, `m6_calib.json` (781-detection dataset).

## 7. Artifacts

- `eval/nx3_demo/` — LOCK_M6=1 demo gate run 1 (8/15) + `run.log`
- `eval/nx3_demo_rerun/` — stability rerun, identical condition (8/15)
- `eval/nx3_default_off_confirm/` — no-env-var confirm run 1 (9/15 — single
  ep1 noise-flip, see TL;DR)
- `eval/nx3_default_off_rerun/` — no-env-var confirm rerun (10/15,
  baseline-identical, zero flips)
- scratchpad `m6_calib.json` — 781 instrumented detections, 15 episodes
