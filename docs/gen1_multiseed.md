# GEN-1 — Multi-Seed Generalization Validation of the Final Adopted Stack

**Date:** 2026-07-09/10
**Agent:** GEN-1 (independent validation of `docs/nx9_avoid.md`, `docs/nx10_scan_fix.md`,
`docs/nx11_ep4.md`)
**Question:** the headline numbers (easy 100%, demo 93.3%, search 100%) are all
seed 999. Every adopted fix (AVOID, GROUND_NET, H3 realized-yaw scan) was justified
as mechanism-based, not tuned to seed 999's specific episodes. Do fresh seeds confirm
that, or does the stack quietly depend on seed-999-shaped luck?

## TL;DR — VERDICT: MECHANISMS GENERALIZE CLEANLY; ONE PRE-EXISTING, ALREADY-DOCUMENTED RESIDUAL RISK SHOWS UP MORE OFTEN THAN SEED 999 SUGGESTED; ONE LIKELY-NEW FAILURE MODE FOUND (false lock, n=1)

- **AVOID (collision/obstacle avoidance):** generalizes cleanly. **Zero falls in
  30 fresh-seed search episodes** (the skill AVOID's own ep12 fix targeted) — matches/
  betters the seed-999 gate. No AVOID-caused regression anywhere.
- **GROUND_NET (learned detector) + graceful fallback:** generalizes cleanly. All 6
  runs loaded the detector successfully (`[grounding] GROUND_NET=1: loaded detector
  ... on device='cuda'`), zero fallback events, zero exceptions, zero crashes across
  90 fresh episodes.
- **H3 scan realized-yaw fix (scan-coverage):** generalizes cleanly on its own narrow
  claim. **Spot-rate 15/15 in both fresh search seeds and both fresh demo seeds** (no
  "target never once in frame" misses recurred) — the specific ep2-class bug (buggy
  assumed-rate coverage) is fixed for good.
- **Residual risk that generalizes (not new, but bigger footprint than seed 999
  showed):** the H3 scan's `_LEG_SIGNS=(+1,-1,-1,+1)` "always try positive first"
  design, combined with `spawn_yaw=180°` + a **large-magnitude NEGATIVE** target
  bearing, reproducibly drives the policy into the exact rotation-order instability
  `docs/nx10_scan_fix.md` §2.2 diagnosed for its reverted 165°-leg attempt — except
  now observed at the **shipped 90°-leg configuration**, which seed 999's own demo
  set never happened to sample (its only two yaw=180° episodes had bearings +62.6°
  [ep4, positive, no scan-order risk] and −28.2° [ep13, mild negative, survived]).
  Two of three fresh-seed demo falls are this mechanism, confirmed via **byte-identical
  physics trajectories** for two different targets at two different seeds (§3.1) —
  this is not scene-specific tuning-sensitivity, it is a target-independent scan/
  locomotion interaction, exactly the open risk `docs/nx10_scan_fix.md` §6 predicted
  in writing before this test ran ("a future seed/scene could sample a demo target
  beyond [the realized coverage]... the rotation-OOD/leg-amplitude tension... would
  need to resolve").
- **Likely genuinely new failure mode (n=1, flagged not confirmed):** search seed
  1000 ep7 — a **false lock onto a same-color, different-shape distractor** (cyan
  cube instead of the true cyan ball; final robot position matches the decoy's
  coordinates almost exactly, §3.3). Not previously catalogued in nx9/nx10/nx11's own
  residual lists (their catalogued misses were total-non-detection or self-body
  contamination, never "detected the wrong object confidently"). Consistent with the
  task brief's own hypothesis: a color/shape combination underrepresented in det_v1's
  ~11k training frames.
- **Net effect on headline numbers:** easy holds inside its own pre-existing
  documented multi-seed noise band (no adoption-attributable change); search holds/
  betters baseline on the metric AVOID was built to protect (falls) with one
  grounding-discrimination miss; demo shows a real ~7-13 point drop at both fresh
  seeds, fully attributable to the already-known, already-documented residual
  rotation/stability limitation, not to any defect in the adopted mechanisms
  themselves.

---

## 1. Protocol

- Working dir `unitree_vla` (NOT the lighter deploy mirror at
  `VLA_mujoco_unitree`, which ships no weights/physics XML and
  cannot run closed-loop rollouts — confirmed against a release-QA pass on the deploy repo).
- Interpreter: the `g1nav` conda env's python interpreter, `MUJOCO_GL=egl`,
  `PYTHONPATH=.:$PYTHONPATH`.
- Pure defaults: no env vars set. `GROUND_NET` default ON
  (`GROUND_NET_CKPT=runs/nx6_heatmap_A/model_best.pt`, present), `AVOID` default ON.
  Checkpoint `checkpoint/goto_best.pt` (md5-identical to `runs/demo_dart_A/epoch_0003.pt`).
- **CLI flag verification done before launching anything** (per task instruction):
  the top-level `README.md`'s example commands (`--ckpt`, `--n-episodes`,
  `--gr00t-ckpt`) do **not** match `eval_closedloop.py`'s actual argparse
  (`--checkpoint`, `--n`, no GR00T flag at all — GR00T-LM is not loaded at eval
  time, `lang_emb=None` per an earlier reproduction log, already independently caught by a
  prior experiment). Verified `--goal-source classical` (not `learned`) is the
  correct flag for the headline numbers: `code/inferencer.py`'s own AVOID
  integration comment (line ~158) states AVOID is "wired only for
  `goal_source='classical'` (covers BOTH the classical HSV+depth backend AND
  GROUND_NET, since GROUND_NET is dispatched INSIDE `ground()` under the same call
  site)" — `goal_source='learned'` is a **different, legacy**, `_grounding_trained`-gated
  Arch-A head (`docs/vision_grounding.md`'s V1, a documented negative result),
  unrelated to GROUND_NET/NX-6's heatmap detector. The current `README.md` here
  (source-of-truth repo) only ever documents `--goal-source classical`/`gt` for
  goto — never `learned` — confirming this reading. (A separate deploy-repo QA doc,
  a separate deploy-repo QA pass staged a `--goal-source learned` example for the *other*
  repo's README under a different premise; not applicable here — verified this
  isn't silently a routing change by grepping `inferencer.py` directly rather than
  trusting either README at face value.)

Commands run (six total, `--no-render`/`--no-video` for headless throughput, n=15
each):

```bash
MUJOCO_GL=egl PYTHONPATH=".:$PYTHONPATH" python -m code.eval_closedloop \
    --checkpoint checkpoint/goto_best.pt --arch A --difficulty easy --device cuda \
    --seed {1000,2000} --n 15 --goal-source classical --no-render --out eval/gen1_easy_{seed}

MUJOCO_GL=egl PYTHONPATH=".:$PYTHONPATH" python -m code.eval_closedloop \
    --checkpoint checkpoint/goto_best.pt --arch A --difficulty demo --device cuda \
    --seed {1000,2000} --n 15 --goal-source classical --no-render --out eval/gen1_demo_{seed}

MUJOCO_GL=egl PYTHONPATH=".:$PYTHONPATH" python -m code.eval_search \
    --checkpoint checkpoint/goto_best.pt --device cuda \
    --seed {1000,2000} --n 15 --no-video --out eval/gen1_search_{seed}
```

Executed as two batches of 3 concurrent jobs (seed 1000's three conditions, then
seed 2000's), polled via a foreground chained loop (30s cadence). All 6 completed
cleanly, no hangs, no >3-minute log silences, no exceptions, no GROUND_NET fallback
events (`grep`-verified across all 6 logs). Wall time: seed-1000 batch ≈9 min,
seed-2000 batch ≈9 min (each batch bottlenecked by its demo run at ~280-310s).

Raw outputs: `eval/gen1_{easy,demo,search}_{1000,2000}/`, logs in
`logs/gen1_{easy,demo,search}_{1000,2000}.log`. `--no-render`/`--no-video` disable
MP4 encoding only — the classical/GROUND_NET grounding path still renders RGB-D every
grounding cycle regardless (confirmed by reading `need_render = render_video or
need_grounding` in both eval scripts), so this does not change the pipeline under
test, only harness throughput.

---

## 2. Results — 3×3 table

| condition | seed 999 (baseline) | seed 1000 | seed 2000 | fresh-seed avg |
|---|---|---|---|---|
| **easy** | **100.0%** (15/15), 0 falls | **100.0%** (15/15), 0 falls | **86.7%** (13/15), 0 falls, 2 didnt-reach | 93.3% |
| **demo** | **93.3%** (14/15), fails={4}, 1 fall | **86.7%** (13/15), fails={7,12}, 2 falls | **80.0%** (12/15), fails={2,8,14}, 1 fall + 2 didnt-reach | 83.3% |
| **search** | **100.0%** (15/15), spot 15/15, reach 15/15, 0 falls | **93.3%** (14/15), spot 15/15, reach 14/15, 0 falls | **100.0%** (15/15), spot 15/15, reach 15/15, 0 falls | 96.7% |

Per-episode failure lists (fresh seeds only; baseline fail sets are quoted from
`docs/nx10_scan_fix.md` §4.1/§4.2/§4.3, unchanged this cycle):

**demo, seed 1000** (13/15): ep7 `fall` steps=351 fd=4.79 (bearing −9.3°, spawn_yaw 180°);
ep12 `fall` steps=256 fd=7.14 (bearing −49.4°, spawn_yaw 180°).
**demo, seed 2000** (12/15): ep2 `fall` steps=256 fd=6.07 (bearing −74.0°, spawn_yaw 180°);
ep8 `didnt-reach` steps=1700 fd=1.65 (bearing −56.5°, spawn_yaw −90°); ep14 `didnt-reach`
steps=1700 fd=3.31, fwd_disp=0.86m — anomalously low displacement, replayed (§3.2).
**easy, seed 2000** (13/15): ep1 `didnt-reach` steps=600 fd=0.995 (target_dist 1.78m);
ep9 `didnt-reach` steps=600 fd=1.151 (target_dist 2.07m). Both non-fall, near-miss
(fd ≈1.7-2× `stop_r`=0.6), matching the pre-existing "pace-deficit" residual class
documented since `docs/nx6_final.md`/`docs/nx11_ep4.md` — not investigated further
(unambiguous from the log, low information value for a replay).
**search, seed 1000** (14/15): ep7 `didnt-reach` steps=2000 fd=3.34 despite `spotted=True`
at step 810 — replayed, §3.3 (false lock, not a scan/collision/fall issue).

---

## 3. Mechanism-level replays (4 targeted, per the "handful, only where ambiguous" bound)

All replays used `checkpoint/goto_best.pt`, pure defaults, device=cuda, an
instrumented `code.inferencer._build_proprio` monkeypatch (yaw from quaternion,
x/y/z per step — mirrors `docs/nx11_ep4.md`'s own replay method) plus, for the
search replay, an instrumented `code.grounding.ground` wrapper. Scratchpad-only,
no `code/` changes. Scripts not committed (matching this codebase's own precedent
for diagnostic-only scripts, `docs/nx10_scan_fix.md` §2.2).

### 3.1 demo ep12 (seed 1000) and ep2 (seed 2000) — byte-identical falls, scan/locomotion-order instability

Both episodes: `spawn_yaw=180°`, robot at `x≈4.1` (right wall), large-magnitude
**negative** target bearing (−49.4° and −74.0° respectively — different targets,
different colors, different distances 7.38m vs 6.47m).

Replayed trajectories (yaw, height) are **identical to 5+ significant figures at
every logged step through the fall**, e.g. both show `z: 0.734→0.702` (steps 220-234,
flat/aligned) then a fast secondary rotation `yaw −93.98°→−110.42°` over steps 230-256
**exactly coincident with height collapse `0.706→0.519`** — a genuine topple, not a
scan-mid-rotation stumble (yaw is essentially flat 200-230, i.e. past the point a
raw scan sweep would still be actively rotating). Since two different target
positions/bearings produce **the same motion to 5+ decimal places**, the fall
precedes any target-specific control divergence — it is driven purely by
`spawn_yaw=180°` and the scan schedule's own fixed-direction-first mechanics
(`_LEG_SIGNS=(+1,-1,-1,+1)`, "always try positive first"), not by anything about the
specific target. Both bearings are large-magnitude negative, i.e. **outside what a
single positive-first 90° leg0 can reach** — forcing the schedule into a leg0→leg1
reversal, reproducing (at the reduced 90° amplitude) the exact
"wrong-direction-first-then-reverse" instability `docs/nx10_scan_fix.md` §2.2
diagnosed and partially (not fully) mitigated by cutting leg amplitude from 165°→90°.

**This is the same class of "policy-level locomotion-stability limit under
sustained/large rotation demand" `docs/nx11_ep4.md` §5 characterized for baseline
ep4** — sudden accelerating yaw-rate ramp coincident with a monotonic height
collapse, not gradual fatigue — but manifesting **at the scan-handoff itself**
(step ~230-256) rather than late/close-range (ep4's step ~1470+). Read together:
the underlying instability (sudden-rotation-triggered balance loss) is broader than
"close-range circling near the target" — it can trigger anywhere the policy is asked
for a large or awkward-direction turn, including immediately post-scan. Seed 999's
demo set had exactly two `spawn_yaw=180°` episodes (ep4: bearing +62.6°, positive,
no scan-order risk; ep13: bearing −28.2°, negative but small, survived) — never a
large-magnitude negative one — so this specific trigger combination was structurally
absent from the seed-999 gate, exactly as `docs/nx10_scan_fix.md` §6 flagged as an
open, unquantified risk before this test ran.

### 3.2 demo ep14 (seed 2000) — stall-and-spin, replay diverges to a fall but confirms a "wedge" signature

Original harness run: `didnt-reach`, steps=1700 (timeout), fd=3.31m,
forward_disp=**0.86m** (vs. the run's own mean successful forward_disp of 5.6m — a
striking outlier). Replay (separate process, so subject to the harness's own
documented ±1-2-episode EGL/physics jitter, a known and expected effect): diverged to `fall`
at step 1042, but the pre-fall signature is unambiguous and consistent with the
original run's near-zero net displacement — from step ~300 to step ~1000, the
robot's **position barely changes** (`x: −0.84→−0.94, y: −3.51→−3.69`, ~0.15m drift
over 700+ steps) while **yaw continuously rotates through more than a full turn**
(160°→178°→−179°→166°→163°, i.e. it slowly spins in place rather than walking
toward the goal), before the same sudden-rotation/height-collapse fall signature as
§3.1. This is a **stall/wedge pattern** — same family as `docs/nx8_stall.md`'s
original ep1 (AVOID couldn't fully resolve it back then) and `docs/nx9_avoid.md`'s
ep14 knife-edge bistability (§3.3 of that doc) — not a pace deficit and not a
detection miss. AVOID is active by default here and did not prevent the stall; it
may be a geometry AVOID's ±25° corridor/near-field design doesn't cover (e.g. the
robot orbiting an obstacle mostly outside the forward corridor), or a self-reinforcing
oscillation between AVOID's bias and the goal-tracking control loop. Not root-caused
further (out of scope — measurement task, no fixes).

### 3.3 search ep7 (seed 1000) — false lock on a same-color, different-shape distractor

Scene: target = cyan **ball** at (−0.64, −2.07), dist 2.34m; a **cyan cube** distractor
at (2.69, −1.05) (same color, different shape); a blue cone distractor. Robot spawns
near center. `[search] SPOTTED at step=810 bearing=2.7°` — correctly locked onto the
true target initially (near dead-center bearing). But the subsequent walk trajectory
moves in the **wrong direction** entirely: `x: −0.18→2.5, y: 0.62→−1.15` by step 1250,
settling at **`(2.55, −1.15)` — matching the cyan cube's position `(2.69, −1.05)`
almost exactly**, not the true target's `(−0.64, −2.07)`. The last 30 logged raw
`ground()` calls all return `not_visible=True` (a stale-goal coast, holding position
near the wrong object). **This is a false lock: the grounding pipeline (GROUND_NET,
default backend) latched onto the same-colored, differently-shaped distractor instead
of the true target**, despite `nx6_heatmap_model`'s detector head being explicitly
shape-conditioned (`CLASS_NAMES`/`COLOR_NAMES`, queried jointly per
`code/nx6_heatmap_model.py`). Not root-caused further (which cycle flipped the lock,
whether it's a classical-fallback moment or a genuine detector confusion) — flagged
as the clearest candidate for a **new** failure taxonomy entry not previously
catalogued by nx9/nx10/nx11 (whose own residual lists are exclusively
total-non-detection or self-body contamination, never "detected, high-confidence,
wrong object").

---

## 4. Failure taxonomy summary (fresh seeds, both conditions with failures)

| taxonomy class | episodes | new vs. known? |
|---|---|---|
| scan/locomotion rotation-order instability (fall, at or shortly after scan handoff) | demo 1000-ep12, demo 2000-ep2 | **known risk, not yet exposed** — `docs/nx10_scan_fix.md` §2.2/§6 explicitly predicted this gap; now empirically confirmed at fresh seeds, mechanism-identical (byte-identical trajectories) |
| sudden mid-walk rotation/balance collapse (fall, no scan involvement) | demo 1000-ep7 | **known class, broader trigger set than documented** — same signature as `docs/nx11_ep4.md`'s ep4 (sudden yaw-ramp + height collapse, not fatigue), but earlier in the episode and without ep4's close-range/long-dwell precondition |
| stall/wedge (near-zero net displacement, continuous in-place rotation, eventual fall or timeout) | demo 2000-ep14 | **known family** (`docs/nx8_stall.md` ep1, `docs/nx9_avoid.md` ep14 bistability) recurring at a fresh seed; AVOID active but did not resolve it here |
| pace-deficit / near-miss (upright, timeout, fd within ~2x stop_r) | easy 2000-ep1, ep9; demo 2000-ep8 | **known, pre-existing class** (`docs/nx6_final.md`, `docs/nx11_ep4.md`'s own fd=0.781 reference point) |
| **false lock (same-color/different-shape distractor)** | search 1000-ep7 | **likely genuinely new** — not present in nx9/10/11's own catalogued residuals |
| grounding total-miss (target never detected) | **none observed** | the class NX-10's scan fix specifically targeted; **zero recurrences** across 90 fresh episodes — clean generalization |
| self-body/AVOID contamination | **none observed** | the class NX-11 found-and-reverted; not re-triggered here |

---

## 5. Honest analysis — does the "mechanism-based, not tuned" claim hold?

**Yes, for the adopted mechanisms themselves.** Every fresh-seed failure traces to
either (a) a pre-existing, already-written-down residual limitation whose trigger
conditions are broader than seed 999's fixed 15-scene sample exposed (rotation-order
instability, stall/wedge, pace-deficit — all three have prior citations above), or
(b) one plausibly-new grounding-discrimination gap (false lock) that is orthogonal
to AVOID/scan/fallback logic entirely and sits in the detector's own training-data
coverage, exactly as the task brief's own hypothesis anticipated. **Nothing found
here implicates the adopted fix code being narrowly tuned to seed 999's specific
episodes** — AVOID's corridor math, GROUND_NET's dispatch/fallback, and the H3
realized-yaw tracker all behaved exactly as designed at both fresh seeds; the
byte-identical §3.1 replay is the strongest possible evidence of this (a bug
"tuned to episode 4" could not reproduce identically against two different target
positions at two different seeds — a *mechanism* reproduces identically because it
doesn't look at the target at all until later).

**Numerically:** easy's fresh-seed spread (86.7-100%) sits exactly inside the
pre-existing, pre-adoption multi-seed band already documented in `README.md`
("easy 93.3% [86.7-100%]", `docs/robustness.md`) — no adoption-attributable
change. Search holds/betters baseline on falls (0/30 fresh vs. 0/15 baseline) with
one grounding miss. Demo is the one real, measurable move: 93.3%→86.7%/80.0%, a
6.7-13.3 point drop, at or just past the ~±1-2 episode binomial-noise band a n=15
success rate implies (~±6.5pp/episode at p≈0.93) — **but the replays show this is
not sampling noise**: it is the same identifiable geometric trigger (spawn_yaw=180°
+ large-magnitude negative bearing) recurring across both fresh seeds via a
mechanism nx10 had already flagged as unresolved. That is a more informative,
higher-confidence signal than the raw success-rate delta alone would suggest on its
own — the generalization claim for the *fixes* holds, while the generalization
claim for the *overall demo success rate* does not (it was never claimed to be
saturated; `docs/nx10_scan_fix.md`/`docs/nx11_ep4.md` both explicitly scoped
"ep4 remains" and "coverage ceiling... a future seed/scene could still time out" as
open items, not closed ones).

---

## 6. Ranked follow-up targets (if a future cycle picks this up)

1. **Scan rotation-order instability at spawn_yaw=180° + large negative bearing**
   (§3.1) — highest-confidence, cleanest mechanism (byte-identical repro across
   seeds/targets), directly actionable: `docs/nx10_scan_fix.md` §6 already scoped
   the fix shape ("an OOD-detection-and-escape mechanism analogous to
   `STALL_BREAK`, rather than a blanket leg-amplitude increase" — a
   direction-aware first-leg choice, e.g. trying the leg whose sign matches the
   sign of *some* cheap prior estimate of bearing, would also be worth
   considering, though that changes behavior more than a bounded escape hatch).
2. **False lock on same-color/shape-confusable distractors** (§3.3) — single
   occurrence, but structurally interesting: if det_v1's ~11k training frames
   under-sample same-color multi-shape scenes, this could recur more broadly than
   n=1 suggests. Worth a targeted audit of training-data color/shape co-occurrence
   before assuming it's rare in deployment too.
3. **Stall/wedge with AVOID active** (§3.2) — lowest confidence (single replay,
   diverged in failure timing from the original harness run), but if it recurs,
   it would indicate AVOID's ±25° corridor design has a geometric blind spot
   (obstacles mostly outside the forward corridor that still block progress via
   repeated re-approach) distinct from the wedge classes AVOID was validated
   against (nx9's ep1/ep12).
4. Pace-deficit / near-miss episodes — lowest priority; already well-characterized,
   no new information this cycle, likely a genuine locomotion-speed/converge-time
   ceiling rather than a fixable bug.

---

## 7. Files

- Eval outputs: `eval/gen1_easy_1000/`, `eval/gen1_demo_1000/`, `eval/gen1_search_1000/`,
  `eval/gen1_easy_2000/`, `eval/gen1_demo_2000/`, `eval/gen1_search_2000/`
- Logs: `logs/gen1_easy_1000.log`, `logs/gen1_demo_1000.log`, `logs/gen1_search_1000.log`,
  `logs/gen1_easy_2000.log`, `logs/gen1_demo_2000.log`, `logs/gen1_search_2000.log`
- No `code/` changes (measurement-only task, per the task brief). No sync.
