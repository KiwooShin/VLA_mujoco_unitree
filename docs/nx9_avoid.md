# NX-9 AVOID — local obstacle avoidance

**Date:** 2026-07-09
**Agent:** NX-9 (follow-on to `docs/nx8_stall.md`, `docs/fa1_failures.md`, `docs/nx1_scan.md`)
**Starting state:** the system had NO local obstacle awareness at all — it walked
straight lines at the (dist, bearing) goal. Three independent, geometry-confirmed
failures are path-obstacle collisions, not grounding or locomotion-stability
problems:

1. **demo ep1 under `GROUND_NET=1`**: a physical wedge against the scene's own
   orange-cone distractor, ~0.25m off the straight-line path (`docs/nx8_stall.md`
   §2.3 — definitive qpos+geometry trace; `STALL_BREAK` detects the stall but
   cannot escape it, since resuming just walks back into the same obstacle).
2. **demo ep4**: a compound failure where even the privileged GT-goal rollout
   fails (`docs/fa1_failures.md` §1, fd=2.01m — the worst GT miss) — genuinely
   geometry-bound, on top of a total grounding miss (target never seen).
3. **search ep12**: a fall caused by a distractor 0.92m along the direct
   approach path (`docs/nx1_scan.md` §3.3 — "robot blocked by a distractor
   object placed directly between robot and target... not a policy failure").

## TL;DR — VERDICT: **AVOID ADOPTED (default ON)** and, with ep1 finally fixed, **GROUND_NET ADOPTED (default ON)** — the learned-grounding adoption bar that NX-6/NX-7/NX-8 could not clear is now met.

Final full gates (final code state, seed 999, n=15 each):

| gate | config | bar | result | per-episode fails |
|---|---|---|---|---|
| (a) | demo / classical, AVOID=1 | hold 10/15 | **10/15 HELD** (one noise flip, settled per protocol — §4.1) | 0, 2, 4, 5, 12 — identical to baseline |
| (b) | easy, AVOID=1 | 15/15 exact | **15/15** | none |
| (c) | search, AVOID=1 | ≥14/15 | **15/15** (ep12 fall FIXED) | none |
| (d) | demo, GROUND_NET=1 AVOID=1 | ≥13/15 (adoption bar) | **13/15** (ep1 FIXED, zero passer breaks) | 2, 4 — the two honest documented fails |

Mechanism-level (rung 2, all 2/2 reproducible): demo ep1 GROUND_NET+AVOID
**FIXED** (was fd≈5.8 wedge, now SUCCESS fd≈0.36-0.37); search ep12 AVOID
**FIXED** (was fall, now SUCCESS fd≈0.47); demo ep4 **honest unchanged FAIL**
(compound grounding-total-miss ⨯ geometry — AVOID has no target signal to
correct toward); 5 passer spot-checks (demo 0,3,5,6,12, each under its own
winning backend) all HELD with unchanged final_dist.

Two mid-ladder findings materially shaped the final mechanism: a **back-
projection sign bug** caught by validating the synthetic unit frames against a
real rendered floor (§1.2), and a **stale-goal self-attack failure mode**
caught by instrumenting a search-ep14 fall (AVOID treating its own target as
an obstacle while coasting on a frozen goal cache — fixed with a goal-
freshness carve-out, §3.3). One bounded constants revision was used, as the
ladder permits (§1.4).

---

## 1. Design (`code/avoid.py`)

A new shared module, imported by all three rollout loops
(`code/inferencer.py`, `code/eval_search.py`, `code/fancy_demo.py` — the
"reuse a shared helper" precedent `code/scan_sched.py`/`code/lock_mgmt.py`
set). `AVOID` env toggle — **default ON post-adoption** (opt out `AVOID=0`);
every call site gates on `AVOID and not <carve-outs>`.

### 1.1 Mechanism

At the SAME grounding cadence classical/GROUND_NET grounding already runs at
(`GROUNDING_PERIOD`, 5-10Hz) — **zero extra renders**, reusing the depth frame
+ intrinsics the caller already rendered this cycle:

1. **Full-frame back-projection** (`_backproject_frame`): every depth pixel →
   robot-egocentric `(dist, bearing, height_above_ground)`, reusing
   `code/grounding.py`'s exact camera model (`CAM_ROBOT_FORWARD_OFFSET_M`,
   the corrected/uncorrected pitch branches keyed on `is_proximity`) so
   bearings/distances are directly comparable to `cached_goal_vec`.
2. **Masks**: near-field only (`AVOID_MIN_VALID_DEPTH_M=0.15` < dist ≤
   `AVOID_NEAR_M=2.0`), forward corridor only (|bearing| ≤
   `AVOID_CORRIDOR_HALF_DEG=25°`), floor excluded (height-above-ground <
   `AVOID_FLOOR_MARGIN_M=0.10` — §1.2), target's own pixels excluded (a
   bearing window around the current goal bearing, only when goal dist <
   `AVOID_TARGET_EXEMPT_DIST_M=2.0m` — don't dodge the thing being
   approached).
3. **Angular-bin aggregation**: `AVOID_N_BEARING_BINS=25` bins across the
   corridor (~2°/bin); each bin scored by its CLOSEST obstacle return
   (1/depth-shaped severity: 0 at 2.0m, saturating 1.0 at ≤1.0m) weighted by
   proximity to corridor center. Each half-corridor's score is its **worst
   (max) bin**, not a mean — a mean over the mostly-empty corridor dilutes a
   small localized obstacle (exactly ep1's cone) below the deadband; the
   worst-bin reading is the standard nearest-threat interpretation of
   "repulsion proportional to 1/depth weighted by proximity to corridor
   center", and makes the cap exactly interpretable (reached only for a
   very close, dead-center obstruction).
4. **Bias**: `raw = -AVOID_MAX_WZ_BIAS(0.30) * overall * imbalance` (more
   obstacle mass LEFT → turn RIGHT, matching steer.py's positive-yaw=LEFT
   convention); deterministic right-hand tie-break for dead-center
   obstacles. **Hysteresis**: fresh bias blends in via EMA (α=0.6); once the
   corridor clears the bias decays geometrically (×0.5/cycle @5Hz → <5%
   within ~1s), snapping to exactly 0 below the 0.01 rad/s deadband. **No
   permanent path offset**: the goal EMA / `cached_goal_vec` /
   lock-management state is never touched — only the injected velocity is
   biased.
5. **`biased_vel_cmd()`**: steer.py's own control law evaluated from
   `cached_goal_vec` (backend-agnostic — classical and GROUND_NET populate
   the same `GroundingResult` contract) plus the yaw bias, clipped back to
   steer.py's own `MAX_WZ=0.80` — the combined command is provably inside
   the BC teacher's own output range. **Yaw-only, never lateral**
   (steer.py's `VX_YAW_DAMP=0.0` / "G1 walks straight": the teacher never
   strafed, so a lateral bias would be off-distribution).

Injection point: only when the bias is nonzero, the model's velocity-head
input is replaced via the existing `gt_vel` teacher-forcing path (the same
plumbing the learned-grounding velocity-replica injection already uses) —
provably untouched on clear-corridor cycles and whenever AVOID is off.

### 1.2 Floor exclusion — back-projection sign bug found and fixed (rung 1)

The vertical term needed for the floor cut is NOT "the other half" of
`cam_to_egocentric`'s forward-distance rotation — that function's
"uncorrected" branch (grounding/ego cams) is a documented approximation, not
a true rotation. Pairing a naive vertical term with it produced heights
~1.4m off. Caught by validating against a REAL
`ArenaRenderer.render_grounding()` frame (known-floor checkered row): the fix
(`y_vert = z_cam·sin(pitch) + y_cam·cos(pitch)`, always the proper rotation
pairing regardless of branch) reproduces height_above_ground≈0 to <1cm on
both the grounding (26°) and proximity (58°) cameras. The synthetic unit
frames were then rebuilt on the verified-correct inverse.

### 1.3 Carve-outs (final)

- **Scan/rescan/dwell**: computation gated on `not _scan_active` at every
  call site AND the injection site is only on the normal student-step path
  (scan steps `continue` earlier, mirroring STALL_BREAK); the bias is also
  reset to 0 by `_lock_drop_and_rescan()` so nothing stale survives a rescan.
- **Goal dist < `AVOID_MIN_GOAL_DIST_M=1.2m`**: hard-zero (proximity endgame
  — the target IS the close object).
- **Maneuver scenes**: `is_maneuver_scene(scene_cfg)`, same
  `scene_cfg['difficulty']` pattern as STALL_BREAK's carve-out.
- **Goal freshness (added at §3.3)**: a new bias is only computed while the
  goal was detected within the last `AVOID_STALE_MAX_MISSED_CYCLES=2`
  grounding cycles (rides through 1-2-cycle blinks); during a longer
  hold-last-known-goal coast, the existing bias only decays
  (`decay_bias()`, same ~1s schedule) — because every target-protection
  carve-out is keyed to `cached_goal_vec`, which is stale during a coast
  (§3.3's ep14 fall trace: stale exemption window + stale distance ⇒ AVOID
  attacks its own target).
- **Decay to zero within ~1s of the corridor clearing** — verified in unit
  tests (trace 0.28→0.14→0.07→0.035→0.0175→0.0 across 5 cycles).

### 1.4 The one constants revision (permitted by the ladder)

First pass (`AVOID_NEAR_M=1.5`, `AVOID_MIN_DEPTH_FOR_WEIGHT_M=0.30`) on the
ep1 replay: the cone was DETECTED from step ~70 (n_obstacle_px up to 4524)
with the correct sign, but the severity ramp needed near-contact range
(≤0.3-0.5m) — which the camera geometry cannot resolve once that close
(self-occlusion) — so the bias never exceeded 0.023 rad/s and detection
vanished right as progress stalled: "detected, but too weak, too late".
Revised as one bundled pass: `AVOID_NEAR_M` 1.5→2.0 (lead time),
`AVOID_MIN_DEPTH_FOR_WEIGHT_M` 0.30→1.0 (severity saturates at a
physically-resolvable range). Unit rung re-run clean (15/15) before any
replay was re-attempted.

---

## 2. Validation rung 1 — unit/static (synthetic depth frames)

`code/avoid.py` self-test (`python code/avoid.py`): **15/15 PASS** (final
code state) — floor-only frame → zero bias / zero obstacle pixels; clear
frame → zero; wall-left → right-turn bias; wall-right → left-turn bias;
dead-center wall → nonzero decisive bias within cap; target-bearing blob
with close goal → fully exempted (zero); same blob with far goal → NOT
exempted (nonzero — exemption is proximity-conditional); `carved_out=True` →
hard zero; hysteresis decays gradually and reaches 0 within 5 cycles;
`biased_vel_cmd` reflects the bias, clips to steer.py's `MAX_WZ`, zeros
inside stop_r, never outputs lateral velocity; maneuver-scene helper.

---

## 3. Validation rung 2 — mechanism-level replays

### 3.1 demo ep1 (`GROUND_NET=1 AVOID=1`) — weave around the cone: FIXED 2/2

Baseline reproduction (`AVOID=0`): FAIL `didnt-reach` fd=5.782 (the exact
5.8-5.9m band of every NX-7/NX-8 attempt).

| run (final code state) | result | final_dist | bias-active frac | steps |
|---|---|---|---|---|
| 1 | **SUCCESS** | 0.374 | 0.202 | 990 |
| 2 | **SUCCESS** | 0.365 | 0.204 | 922 |

(Post-constants-revision, pre-freshness-fix runs were also 2/2: 0.362/0.366.)

### 3.2 search ep12 (`AVOID=1`) — ball distractor 0.92m along path: FIXED 2/2

Baseline reproduction (`AVOID=0`): FAIL `fall`, spotted@960 fell@1196
(matches `docs/nx1_scan.md`'s spotted@960 fell@1194 within physics jitter).

| run (final code state) | result | final_dist | bias-active frac |
|---|---|---|---|
| 1 | **SUCCESS** | 0.473 | 0.176 |
| 2 | **SUCCESS** | 0.467 | 0.176 |

### 3.3 search ep14 — the stale-goal self-attack finding (and its fix)

The FIRST full search gate (pre-freshness-fix code) scored 14/15 with the
fail set flipped: ep12 fixed but **ep14 fell** (a documented knife-edge
passer — baseline stops at fd≈0.48-0.50 vs stop_r=0.50, flagged in
`docs/cam_p4_gate.md` as "a known close-call, not a clean-margin success").
Standalone: 2/3 falls under AVOID=1 (bias-active 0.21-0.25) vs 0 falls ever
observed at baseline. The instrumented fall trace found the mechanism: after
the (pre-existing, AVOID-independent) marginal-stop overshoot, the target
goes `not_visible` and `cached_goal_vec` FREEZES at 1.73m/+23.5° for ~38
missed cycles while the robot circles its own target at <1m true range — the
stale exemption window misses the target's true bearing, the stale 1.73m
distance keeps the 1.2m proximity cut from ever firing, and AVOID saturates
its bias against the very object it exists to protect, driving the circling
into a fall.

**Fix (spec compliance, not tuning)**: the goal-freshness carve-out (§1.3)
plus gating the computation on `not _scan_active` (the spec's own "off
during scan" carve-out, previously only enforced at the injection site).
After the fix: ep14 under AVOID=1 never engages the bias at all (3/3 runs,
bias-active 0.000, zero falls) — behaviorally identical to AVOID=0, whose
own outcome is genuinely bistable at baseline (observed AVOID=0 runs:
success 0.479 / `didnt-reach` 2.52 / success 0.478 — the knife-edge stop
flips run-to-run with no AVOID involvement). Both targeted fixes (ep1, ep12)
were re-verified 2/2 after this change (§3.1, §3.2), and the FINAL full
search gate scored 15/15 (§4.3).

### 3.4 demo ep4 (`AVOID=1`) — honest: unchanged FAIL, as expected

FAIL `didnt-reach` fd=9.943 (baseline 10.24 — noise-level shift, same
outcome). AVOID engaged substantially (bias-active 0.221) but the target is
never DETECTED at any point in this episode (`docs/fa1_failures.md`: total
grounding miss; even GT-goal fails at fd=2.01) — there is no goal signal for
an avoidance bias to correct toward. Honest partial-scope result: AVOID
addresses the path-obstacle half of ep4's compound failure class, not the
grounding-total-miss half.

### 3.5 Passer spot-checks (each under its own winning backend): 5/5 HELD

| ep (demo) | backend | result | final_dist | bias-active frac |
|---|---|---|---|---|
| 0  | GROUND_NET=1 | SUCCESS | 0.363 | 0.138 |
| 3  | classical    | SUCCESS | 0.368 | 0.198 |
| 5  | GROUND_NET=1 | SUCCESS | 0.366 | 0.058 |
| 6  | classical    | SUCCESS | 0.363 | 0.040 |
| 12 | GROUND_NET=1 | SUCCESS | 0.366 | 0.012 |

final_dist unchanged within the harness's documented jitter band despite
nonzero engagement — no course corruption. (All five also hold inside the
final full gates below.)

---

## 4. Validation rung 3 — full gates (final code state, seed 999, n=15)

### 4.1 (a) demo/classical AVOID=1 — bar: hold 10/15 → **HELD (10/15)**

First run: 9/15 — a single flip on ep7 (cyan cube, baseline passer; every
other episode matched baseline exactly). Per the noise protocol (one rerun):
ep7 standalone replays 2/2 SUCCESS under AVOID=1 (fd 0.377/0.381, bias-active
0.17) and AVOID=0 SUCCESS — and the full protocol rerun scored **10/15 with
the exact baseline fail set**:

| ep | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| result | F | S | F | S | F | F | S | S | S | S | S | S | F | S | S |
| final_dist | 7.27 | 0.39 | 10.56 | 0.37 | 10.23 | 3.75 | 0.37 | 0.37 | 0.37 | 0.37 | 0.36 | 0.37 | 5.71 | 0.35 | 0.37 |

Fails {0,2,4,5,12} = the documented classical baseline fail set
(`docs/nx6_final.md`). Zero regressions; gains none expected (classical's
five fails are grounding-bound, not path-bound).

### 4.2 (b) easy AVOID=1 — bar: 15/15 exact → **15/15**

All 15 SUCCESS, final_dist 0.56-0.59m (the documented easy band). Zero
change from baseline.

### 4.3 (c) search AVOID=1 — bar: ≥14/15 → **15/15**

SPOT 15/15, REACH 15/15, falls 0/15. **ep12 (the target fix) succeeds**
(fd=0.47) — the "ep12-fall fix would give 15/15" outcome the brief hoped
for. ep14 (the §3.3 knife-edge) passed this gate run; its bistability is
pre-existing and AVOID-independent after the freshness fix (§3.3).

### 4.4 (d) demo GROUND_NET=1 AVOID=1 — bar: ≥13/15 → **13/15, ADOPTION BAR MET**

| ep | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| result | S | **S** | F | S | F | S | S | S | S | S | S | S | S | S | S |
| final_dist | 0.37 | **0.37** | 7.08 | 0.36 | 10.35 | 0.37 | 0.37 | 0.37 | 0.36 | 0.37 | 0.37 | 0.37 | 0.37 | 0.39 | 0.37 |

**ep1 — the episode that blocked GROUND_NET adoption across NX-6, NX-7, and
NX-8 — passes.** The only fails are ep2 and ep4, the two honest documented
failures (`docs/fa1_failures.md`: targets never visible in any captured
frame on these episodes' trajectories; ep4 fails even with GT goals). The
fail set {2,4} is a strict subset of classical's {0,2,4,5,12}: **no passer
(of either backend) breaks.** 13/15 = 86.7%, vs classical baseline 66.7%.

---

## 5. Adoption matrix — outcome

| mechanism | condition | outcome | verdict |
|---|---|---|---|
| **AVOID** | (a),(b),(c) all clean | (a) 10/15 held / (b) 15/15 / (c) 15/15 | **ADOPT — default ON** (`code/avoid.py`: `AVOID = _env_flag("AVOID", default="1")`; opt out `AVOID=0`) |
| **GROUND_NET** | (d) ≥13/15 AND no other passer break (with AVOID resolved) | (d) 13/15, fails only honest {2,4} | **ADOPT — default ON** (`code/grounding.py`: `GROUND_NET = _env_flag("GROUND_NET", default="1")`; opt out `GROUND_NET=0`) |

GROUND_NET adoption obligations, both implemented and confirmed:

- **Graceful classical fallback when the checkpoint is missing** (the deploy
  repo ships no weights): `ground()`'s dispatch now falls through to the
  classical HSV+depth pipeline (with a clear, one-shot log line) instead of
  the old opt-in-era sticky `not_visible`-forever behavior. Confirmed: (1)
  in-repo full rollout with `GROUND_NET_CKPT=/nonexistent` → fallback lines
  printed once, classical path used, episode SUCCESS; (2) in the deploy repo
  (no `runs/` at all), `ground()` called with no env → default ON, fallback
  lines, classical result, no crash.
- **No-env confirm runs** (all with NO env vars set, i.e. pure new
  defaults): easy ep0 SUCCESS (detector loaded, AVOID active 0.15); demo ep1
  SUCCESS fd=0.362 (the flagship fix, under pure defaults);
  `fancy_demo.py --smoke` (its own separate loop) SUCCESS fd=0.475, no
  crash.

**Final defaults-state scoreboard:** demo 13/15 = 86.7% (was 10/15 = 66.7%
classical), easy 15/15 = 100%, search 15/15 = 100% (was 14/15 = 93.3%).

---

## 6. Files changed / synced

Changed in `unitree_vla`:

- **`code/avoid.py`** — NEW shared module (mechanism + carve-outs + 15-test
  synthetic self-test in `__main__`). Default ON post-adoption.
- **`code/inferencer.py`** — additive: `_avoid` import, `AVOID` constant
  block, per-episode state (+ reset in `_lock_drop_and_rescan`), the
  grounding-cycle bias computation (scan- and freshness-gated), the
  `gt_vel` injection (lowest precedence, only when bias ≠ 0), new
  `RolloutResult.avoid_bias_active_frac` field. STALL_BREAK untouched.
- **`code/eval_search.py`** — same wiring pattern for its duplicated rollout
  loop; `SearchResult.avoid_bias_active_frac`.
- **`code/fancy_demo.py`** — same wiring pattern for its duplicated rollout
  loop; `avoid_bias_active_frac` in its result dict.
- **`code/grounding.py`** — `GROUND_NET` default flipped ON; graceful
  classical fallback at `ground()`'s dispatch + one-shot fallback notices;
  updated load-failure message. Classical pipeline itself byte-unchanged.

Synced (byte-copied and `cmp`-verified) to
`VLA_mujoco_unitree/code/` — NO checkpoints, NO datasets,
NO git operations:

- Deploy files: `avoid.py`, `inferencer.py`, `eval_search.py`,
  `fancy_demo.py`, `grounding.py`
- Detector pipeline (GROUND_NET adoption): `nx6_heatmap_model.py`,
  `gen_det_dataset.py`, `nx6_heatmap_data.py`, `train_nx6_heatmap.py`,
  `eval_nx6_heatmap.py`, plus `nx6_heatmap_eval_utils.py` (a required import
  of the train/eval pipeline files — caught by an import sanity test in the
  deploy repo; without it `train_nx6_heatmap.py`/`eval_nx6_heatmap.py` fail
  at import).

All 11 synced modules import cleanly in the deploy repo. (The deploy repo
cannot run full physics rollouts — its `third_party/` robot XML was never
shipped there, a pre-existing gap unrelated to this pass; the `ground()`
-level fallback confirm above is the deploy-side behavioral check.)

Diagnostic artifacts (scratchpad, not committed/synced): unit-test runs,
ep1/ep12/ep14/ep7/ep4 replay logs, instrumented ep1/ep14 bias traces, all
four final gate logs + the gate-(a) protocol rerun, no-env confirm logs.

---

## 7. What remains / follow-on

- **demo ep2/ep4** are now the only failures anywhere across the three
  gated skills, and both are grounding-recall failures (target never
  detected on these episodes' trajectories), NOT path/obstacle failures —
  `docs/nx6_final.md` §8's longer-heatmap-training recommendation (training
  was cut short at epoch 41/60 with val loss still decreasing) remains the
  concrete next lever for both.
- **search ep14** remains a documented knife-edge episode (baseline stop
  margin ~7mm vs stop_r) whose outcome is bistable run-to-run independent of
  any NX-9 mechanism (§3.3). A stop-radius hysteresis or a final-approach
  creep extension would be the targeted fix if it ever needs one.
- **AVOID is a structural no-op for `goal_source='gt'` rollouts**: `_scan_active`
  is only ever cleared inside the `need_classical_grounding` path, which never
  executes when `goal_source != 'classical'` — so on a `gt`-goal probe run
  `_scan_active` silently stays `True` all episode and AVOID's own call-site
  gate (`AVOID and not _avoid_is_maneuver and not _scan_active`) is
  permanently closed, independent of the `AVOID` env var. Found and fully
  traced (ep4 GT re-run) by `docs/fa2_residuals.md` §4 — not caught here
  because NX-9's own validation never ran AVOID against a `gt`-goal rollout.
- **AVOID's freshness carve-out** means the mechanism is inert while
  coasting on a stale goal — safe (that's the point), but it also means
  AVOID cannot help against obstacles encountered during a blind coast.
  Pairing it with `STALL_BREAK` (still opt-in, default OFF) for
  collision-during-coast recovery is the natural composition if that gap
  ever shows up in practice.
- The **worst-bin-per-side aggregation** and the **verified back-projection
  helper** (`_backproject_frame` — full-frame egocentric dist/bearing/height
  from any of this codebase's cameras) are reusable beyond avoidance (e.g.
  free-space estimation, floor-anomaly detection).

---

## 8. Maneuver re-gate under the adopted defaults (MV-1, 2026-07-09)

**Question:** did adopting GROUND_NET+AVOID default-ON regress the maneuver
skill? **Answer: NO — 10/15 = 66.7%, inside the documented 66.7–73.3% band
(docs/repro_maneuver.md §2d), with the exact documented failure fingerprint
and no new failure modes.**

**Mechanism finding (why no regression is even possible):** the adopted stack
is *structurally inert* for this eval. `code/eval_maneuver.py` has its own
duplicated rollout loop (the recurring codebase pattern) that never imports
`code/grounding.py`, `code/avoid.py`, or `code/inferencer.py` — verified both
by static grep and by importing `code.eval_maneuver` and inspecting
`sys.modules` (only `code.steer` appears, via a `gen_dart_dataset` utility
import; steer.py was untouched by NX-9). Its rollout never calls `ground()`
(vision = cached zero-image TinyViT embedding; goal = GT-privileged teacher
forcing; vel = expert TF during TURN_PHASE under hybrid-vel), so GROUND_NET
never engages. AVOID's maneuver carve-out (`is_maneuver_scene`) is wired in
the three loops that share scene configs (`inferencer.py`, `eval_search.py`,
`fancy_demo.py`) — it is correct there but simply irrelevant to
eval_maneuver, which the new stack cannot reach at all. eval_maneuver's
entire import chain is byte-identical to CX-4's re-gate state (NX-9 changed
only avoid/inferencer/eval_search/fancy_demo/grounding).

**Run (pure defaults, no env toggles):** `checkpoint/maneuver_best.pt`,
seed 999, n=15, hybrid-vel (default), `--render-n 0`, device=cpu →
`eval/mv1_regate/`: **10/15 = 66.7%** (2 falls, 2 no_landmark,
1 wrong_heading). Per-episode vs the docs/repro_maneuver.md §2b fingerprint:

- ep9 (+46.1°) and ep14 (−82.7°) — the two consistent-fail episodes, at
  their documented error bands (+46..+51° / −81..−88°).
- ep1 FAIL at +25.1° and ep0 PASS at +23.2° — the two documented borderline
  coin-flips straddling the 25° threshold, both inside their documented
  flip ranges.
- ep11 and ep13 falls — the two documented marginal-stability episodes
  (baseline falls 4/6 and 3/6 runs respectively, code-uncorrelated).
- All eight documented solid-success episodes (2,4,5,6,7,8,10,12) plus ep3
  succeeded.

Verdict: **no regression** — in-band result, failure set is a pure draw from
the documented noise structure. Maneuver needed no carve-out to survive
adoption because nothing adopted touches its eval path; the carve-out
matters only for maneuver scenes run through the shared goto-style loops.

(Operational note: a first attempt with the default `--render-n 3` died
silently in the EGL success-video render while another agent was recording
videos on the same GPU — use `--render-n 0`, the documented re-gate
protocol, for headless gating runs.)
