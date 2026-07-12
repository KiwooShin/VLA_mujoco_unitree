# FA-2 — Residual Failure Diagnosis: demo ep2 / ep4 under the adopted default stack

**Date:** 2026-07-09
**Agent:** FA-2 (follow-on to `docs/fa1_failures.md`, `docs/nx6_final.md`, `docs/nx9_avoid.md`)
**Stack under test:** pure defaults (no env vars set) — `GROUND_NET=1`, `AVOID=1`,
checkpoint `checkpoint/goto_best.pt` + `runs/nx6_heatmap_A/model_best.pt`
(`conf_thresh=0.59`), `goal_source='classical'` (which internally dispatches to
GROUND_NET), `vel_source='predicted'`, seed 999, `MAXSTEPS['demo']=1400`
(`code/eval_closedloop.py`), `SCAN_TIMEOUT=200` (`code/inferencer.py`).

**Question:** why do demo ep2/ep4 still fail under the adopted GROUND_NET+AVOID
stack (`docs/nx9_avoid.md` §4.4, the last 2/15), and would a deploy-side
EXPLORATION behavior ("relocate to a new vantage point when the initial scan
finds nothing") fix them?

## TL;DR

**Neither episode is occluded.** Geometric line-of-sight checks (§1) find no
distractor within its own footprint radius of the straight spawn→target
segment in either scene — the target is optically visible from the spawn
**position** in both cases. Both failures are **pure scan angular-coverage
misses**: the H3 initial scan (`code/inferencer.py`, `±90°`-intended,
200-step-budget, right-then-left sweep) empirically only realizes a
**~-61°/+64° camera-coverage arc** (confirmed by instrumented replay, not the
intended ±90°+FOV), and:

- **ep2** (blue cone, bearing -73.8° from spawn heading): closest the scan
  ever brings the target to center is **43.2°** off-axis — **14.3° short** of
  the camera's ±28.9° half-FOV. Never detected at any point in the 1400-step
  episode (0/140 raw detector calls present; raw confidence pinned at
  noise-floor 0.006-0.021 throughout). **Verdict: (b) fixable by more scan
  coverage, same position — no relocation needed. High step-budget margin**
  (GT-goal succeeds in 760/1400 steps per `docs/fa1_failures.md`).
- **ep4** (purple ball, bearing +62.6° from spawn heading): closest approach
  is **30.8°** off-axis — only **1.9° short** of the FOV edge, i.e. a
  near-miss. But even a perfect-detection oracle (GT-goal replay, freshly
  re-run here) **also fails**, closing only to fd=2.10m by step 1400 while
  monotonically, non-plateauing, still converging — a pure locomotion-pace
  deficit (the realized turn-then-walk speed for a large initial heading
  offset is far slower than the design-intent steer.py formula or the
  scan's own injected turn rate), **not an obstacle stall**. **Verdict:
  compound — (b) detection-fixable, but (d) a structurally-hard residual
  remains: fixing detection alone would not flip ep4 to SUCCESS inside the
  1400-step budget.**

**One exploration primitive does NOT cleanly cover both**, and more
importantly, **relocation (translation) is not the operative fix for
either** — both are angular/scan-coverage problems from the *existing* spawn
position. A relocate-to-vantage behavior would incidentally help (any
wider/fuller rescan would eventually sweep past the missed bearing) but is a
more expensive superset of the real fix (widen/rebalance the H3 scan sweep,
or add a bounded second-phase rescan on timeout) — and even that full fix
would only rescue ep2, not ep4 (§4).

---

## 1. Geometry reconstruction (`code/scene.py`, `derive_rng(999, ep)`, difficulty='demo')

Demo scenes have **no walls or geometry between robot and target other than
the sampled distractor objects themselves** (`code/scene.py::sample_scene`) —
occlusion, if any, can only come from another object's own footprint sitting
on the direct line-of-sight segment. Checked via point-to-segment distance
(object center vs. the spawn→target line) against each object's placement
radius (`size/2`):

### ep2 — blue cone, dist=4.86m, bearing=-73.8° from spawn heading (yaw=0°)

Robot spawn (-4.125, -0.125), yaw=0°. Distractors: cyan cone (2.92m,+43.2°),
yellow ball (3.61m,-7.8°), green ball (2.39m,-93.6°), orange cone
(6.69m,-27.7°), purple cube (6.77m,+29.3°), cyan cylinder (0.71m,+134.6°).

**No distractor blocks the spawn→target line** — closest perpendicular
distance from any non-target object to the LOS segment is 0.81m (green
ball), far outside its 0.12m radius. **Target is optically visible from
spawn, in principle** — the target itself subtends only 1.53° half-angle at
4.86m (a small but not sub-pixel target).

### ep4 — purple ball, dist=7.21m, bearing=+62.6° from spawn heading (yaw=180°)

Robot spawn (4.125, 1.883), yaw=180°. Distractors: cyan ball (4.84m,-13.3°),
orange cube (3.29m,-57.1°), blue cylinder (3.09m,+18.6°), red cone
(1.09m,+116.2°).

**No distractor blocks the spawn→target line** either (closest: blue
cylinder, perp-dist 2.14m to the LOS, radius 0.11m). Target subtends 0.95°
half-angle at 7.21m.

**Conclusion: neither episode has a physical-occlusion problem.** See
top-down sketches: `eval/fa2_residuals/ep2_topdown.png`,
`eval/fa2_residuals/ep4_topdown.png` (target, all distractors, spawn pose,
LOS line, realized scan-camera coverage wedge, and the replayed
trajectories).

---

## 2. Instrumented replay under pure defaults (item 2)

Single-episode replays (`checkpoint/goto_best.pt`, arch=A, device=cuda, no
env vars — `GROUND_NET`/`AVOID` both default-ON) with two monkeypatches:
`code.nx6_heatmap_model.HeatmapDetector.infer` (logs the RAW peak sigmoid
confidence every call, regardless of `conf_thresh` — `decode_single` always
computes the true peak, only `present` is thresholded) and
`code.inferencer._build_proprio` (logs `(step, x, y, yaw)` every single
step, since it's called exactly once per step on every code path — scan,
rescan, and normal mode alike).

### ep2 — result: FAIL `didnt-reach`, final_dist=7.321, steps=1400, avoid_bias_active_frac=**0.000**

```
[scan] TIMEOUT at step=200, falling back to default goal
dist: 4.86(step0) -> 3.31(step~650, closest approach) -> 7.32(step1400, diverging)
```

The classical false-lock FA-1 documented for ep2 is **confirmed gone** under
GROUND_NET — there is no false-positive lock at all now. Instead: **0/140
raw detector calls ever cross `present`**, and raw confidence never exceeds
**0.021** (pure noise floor; for reference GROUND_NET_TAU=0.59,
GROUND_NET_TAU_TRACK=0.40 — nowhere close). Cross-referencing every raw call
against the true egocentric bearing (computed from the logged qpos + known
target xy): the **minimum |egocentric bearing| ever achieved, over the
ENTIRE 1400-step episode (scan + post-scan straight walk), is 43.2°** at
step 201 — the camera's own ±28.87° half-FOV (`EGO_FOVY_RENDERED=45°`,
480x360 grounding render) is never entered. The target is **never once
inside the camera frame**, at any point in the episode — confirmed visually
by re-rendering the actual grounding-cam RGB frame at the closest-bearing
qpos (`eval/fa2_residuals/ep2_frame.png` — shows the yellow-ball and
orange-cone distractors but no blue anywhere).

After scan-timeout, `cached_goal_vec` falls back to the "straight ahead"
default `(dist=2.0, bearing=0)`, which — since bearing is pinned at exactly
0 forever (no detection ever updates it) — makes the robot walk in a
**dead-straight line** in whatever direction it happened to be facing at
step 200 (yaw≈-31.7°). This line happens to pass fairly near the target
(closest approach 3.31m around step 650) purely by the geometry of where the
scan stopped, then walks past and diverges to fd=7.32m by the 1400-step cap.
**AVOID never engages** (`avoid_bias_active_frac=0.000`) — not because
there's no obstacle (there genuinely isn't one on this path), but because
AVOID's own goal-freshness carve-out (`docs/nx9_avoid.md` §1.3) requires a
goal detected within the last 2 grounding cycles, which never happens here
(no detection occurs, ever).

### ep4 — result: FAIL `didnt-reach`, final_dist=9.958, steps=1400, avoid_bias_active_frac=**0.000**

```
[scan] TIMEOUT at step=200, falling back to default goal
dist: 7.21(step0) -> 7.09(step200) -> monotonically increasing -> 9.96(step1400)
```

Same structural picture: 0/140 raw calls present. Raw confidence peaks
higher than ep2's pure-noise floor (**max 0.236** at step≈301, still far
below tau=0.59) but cross-referencing against true bearing shows this peak
occurs while the true target sits at **+109° egocentric bearing** (nowhere
near the reported in-frame peak's implied geometry) — a spurious activation
on something else in view, not a weak-but-correct target signal. Re-rendered
the grounding-cam frame at that qpos: shows the **blue cylinder distractor**
prominently in frame (`eval/fa2_residuals/ep4_frame.png`) — the elevated
(still sub-threshold) confidence is very plausibly a query-color-conditioned
partial response near the blue cylinder, not the purple ball. **The true
target is never in any captured frame this episode either.**

Bearing analysis: minimum |egocentric bearing| achieved **anywhere in the
1400-step episode is 30.8° at step 76** (during the scan's right-sweep
peak, just before it reverses to sweep left) — **only 1.9° short of the
±28.87° half-FOV**, the closest of the two episodes. Confirmed visually:
`eval/fa2_residuals/ep4_frame.png` (rendered at exactly this qpos) shows an
empty floor + the blue cylinder, no purple ball, consistent with a ~2°
miss. After scan-timeout the robot again just walks straight (yaw stays
near -147° for a while then the frame drifts) and never gets any closer to
alignment — dist diverges monotonically the entire post-scan episode.
`avoid_bias_active_frac=0.000` for the same structural reason as ep2.

---

## 3. Root mechanism: the H3 scan's realized coverage is much narrower than intended

`code/inferencer.py`'s scan comment claims "sweeps ±90° arc," but the actual
per-step yaw trace (both episodes) shows a **realized yaw excursion of only
about -32° to +35°** relative to spawn heading over the 200-step budget
(right-phase for `_quarter=75` steps, left-phase for the remaining 125),
*not* the ±90° the design comment describes — the commanded
`SCAN_RATE=0.6 rad/s` is not fully realized by the WBC-free student-driven
turn (physically achieves roughly 70-80% of the commanded rate). Adding the
camera's own ±28.87° half-FOV gives an actual **detectable bearing coverage
of roughly [-61°, +64°]** from spawn heading — visualized as the orange
wedge in both top-down sketches. ep2's target (-73.8°) and ep4's target
(+62.6°, technically just inside the nominal wedge boundary but the *scan's
own reversal timing* — the right phase ends at `_quarter=75` steps, cutting
off 1.9° before the target enters frame — means the ACTUAL closest approach
undershoots even the nominal wedge) both fall (or nearly fall) outside this
realized window. This is a **scan-schedule/coverage bug** independent of
detector quality — the raw-confidence noise floor (0.006-0.021 typical, ep2)
confirms the detector never even got a partial look, so a detector
retrain/threshold change cannot fix either episode; the fix has to be in the
scan-schedule geometry (`code/inferencer.py`'s H3 scan, or a
`code/scan_sched.py`/`code/lock_mgmt.py`-style bounded second-phase rescan
on timeout).

---

## 4. ep4's GT-goal re-run (item 3): where does it get stuck?

Re-ran `goal_source='gt'` (privileged goal every step, `vel_source='predicted'`,
the student's own trained velocity head — same methodology as the
`eval/DIAG_deployed_demo_gt` artifact FA-1 cites), with `AVOID=1` explicitly
set, instrumented with the same qpos/yaw logger:

```
=== RESULT ep4 GT: FAIL didnt-reach  final_dist=2.090  steps=1400  avoid_bias_active_frac=0.0000 ===
step=   0  dist=7.206   step= 600  dist=4.483   step=1200  dist=2.430
step= 300  dist=6.053   step= 900  dist=3.347   step=1400  dist=2.096  (closest approach = final step)
```

**Key structural finding: `AVOID` never engages for `goal_source='gt'`
rollouts at all, independent of the `AVOID` env var** —
confirmed both by static read and by the `avoid_bias_active_frac=0.0000`
result. `_scan_active` (`code/inferencer.py:684`) initializes `True` and is
**only ever cleared inside the `need_classical_grounding` code path**
(lines 1000/1159/1207), which is gated on `_need_classical_render` — itself
`False` whenever `goal_source != 'classical'`. For `goal_source='gt'`, the H3
scan block is entered exactly zero times, so `_scan_active` silently stays
`True` for the entire episode, and AVOID's own call site
(`if AVOID and not _avoid_is_maneuver and not _scan_active:`, line 1084) is
therefore **permanently gated off**. **"GT+AVOID" as literally specified is
a no-op combination in the current code** — this is a real (if narrow) gap:
AVOID cannot be validated or used on privileged-goal probe runs at all.

Separately, and more importantly for ep4: **there is no plateau/stall to
avoid in the first place.** Distance decreases **monotonically and
continuously** from 7.21m to 2.10m across the full 1400 steps with no
obstacle-wedge signature (compare to the genuine ep1-cone-wedge trace in
`docs/nx8_stall.md` §2.3, which shows a hard plateau) — closest approach is
literally the final step. The bottleneck is **locomotion pace, not
geometry-blocking**: the robot's realized turn-then-walk behavior (driven by
the student's own trained velocity head, not `steer.py`'s formula directly)
converges heading very slowly for this spawn's large initial offset (spawn
yaw=180°, target bearing +62.6° — even at step 1400 the residual yaw error
is still ~52°, well above `steer.py`'s own `FACE_THR_RAD=25°` "should be
turning in place" threshold) — a wide, slow arcing path
(`eval/fa2_residuals/ep4_topdown.png`, green trajectory) that simply
hasn't covered enough ground by step 1400. **AVOID (a yaw-only corridor-bias
mechanism with no forward-speed component) could not fix this even if the
scan_active gate were removed** — there is no obstacle mass to detect, and
AVOID never touches forward speed by design (`docs/nx9_avoid.md` §1.1.5,
"Yaw-only, never lateral," and it doesn't touch vx either).

**Answer to "would AVOID fix the GT run": No — structurally inert for `gt`
goal-source (scan_active-gated), and even if that gate were lifted, there is
no obstacle-collision mechanism present to fix; the shortfall is pure
locomotion-pace against the step budget.**

---

## 5. Verdicts (item 4)

### ep2 — **(b) fixable by more scan coverage, same position (no relocation needed)**

- Target is 14.3° beyond the realized scan-camera coverage edge; no
  occlusion; GT-goal (perfect detection oracle) already succeeds cleanly in
  760/1400 steps (`docs/fa1_failures.md`), so once acquired, locomotion is
  not a limiting factor.
- **Step-budget feasibility: HIGH margin.** Closing the 14° scan gap needs
  only ~25-30 more steps at the scan's realized turn rate (well inside a
  modestly extended or rebalanced 200-step scan budget), leaving >600 spare
  steps of the 1400-step cap for the walk-in.
- If a "vantage point" is still wanted in the literal relocate-to-new-xy
  sense: **not required** — the target is visible from the spawn **position**
  itself (unobstructed LOS, confirmed geometrically and by the render check);
  only the scan's rotational coverage needs to change, not the robot's xy.

### ep4 — **compound: (b) detection-fixable + (d) structurally-hard locomotion residual**

- Target is only 1.9° beyond the realized scan-camera coverage edge — the
  single cheapest fix of the two (extend the scan's right-phase by ~4-8
  steps, or rebalance the 75/125 right/left split).
- **But fixing detection alone does not flip ep4 to SUCCESS**: the freshly
  re-run GT-goal oracle (perfect detection from step 0, i.e. a strictly
  *better* starting condition than any real fix would provide, since real
  detection would only acquire around step ~80-90 at the earliest) still
  only reaches fd=2.10m by step 1400, short of `stop_r=0.4`. A real fix
  would acquire later than GT's step-0 start, so it would do **no better
  than fd≈2.1-3m** within the existing 1400-step budget — still a FAIL.
- **Step-budget feasibility: NOT feasible within MAXSTEPS=1400 as currently
  configured**, even after the detection gap is closed. Closing the
  remaining gap needs either (i) a larger step budget, or (ii) a locomotion
  fix (faster heading convergence / relaxed `FACE_THR_RAD` forward-walk
  during large yaw error) — out of scope for a deploy-side scan/exploration
  change.
- If a relocate-to-vantage primitive is used anyway (e.g. for the detection
  half): a reachable nearby vantage is not really the mechanism at play here
  either (no occlusion) — the same "extend the existing scan a few more
  steps" fix applies, at position = spawn, no translation required.

### Does ONE exploration primitive cover both?

**No, not cleanly.** A "relocate + rescan on scan-miss" primitive would
plausibly still work for detection on **both** episodes (any sufficiently
wide rescan, wherever it's centered, will eventually sweep across either
missed bearing, since neither target is occluded from anywhere reasonable in
the open arena) — but it is a more expensive, less-targeted fix than simply
widening/rebalancing the existing H3 scan (which needs **zero** translation
for either episode). And even a perfect relocate+rescan exploration
primitive would only fully rescue **ep2** — ep4 would still fail on the
independent locomotion-pace deficit confirmed by the GT re-run (§4),
regardless of how quickly or reliably detection is acquired.

---

## 6. Artifacts

- `eval/fa2_residuals/ep2_topdown.png`, `eval/fa2_residuals/ep4_topdown.png`
  — top-down scene reconstruction: arena bounds, all objects (target
  outlined), robot spawn + heading, realized scan-camera coverage wedge,
  replayed trajectory (+ GT trajectory for ep4), spawn→target LOS.
- `eval/fa2_residuals/ep2_frame.png`, `eval/fa2_residuals/ep4_frame.png` —
  actual re-rendered grounding-camera RGB frame at each episode's
  closest-bearing-approach qpos, visually confirming the target is out of
  frame at the closest point the scan ever reaches.
- Diagnostic scripts (scratchpad, not committed):
  `replay_ep.py` (instrumented classical/GROUND_NET replay, raw detector +
  qpos logging), `replay_gt.py` (GT-goal replay + qpos logging),
  `geom_recon.py` (analytic occlusion/bearing check), `render_check.py`
  (frame re-render sanity check), `sketch.py` (top-down plot generator).
- Raw replay logs/JSON (scratchpad): `ep2.json`, `ep4.json`,
  `ep4_gt_avoid.json` (qpos traces + raw/accepted detector call logs).

No files under `code/` were modified — all instrumentation was via
Python-level monkeypatches in standalone scratchpad scripts, matching the
FA-1/NX-7 precedent.
