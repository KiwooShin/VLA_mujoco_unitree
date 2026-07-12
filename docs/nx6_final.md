# NX-6 INTEGRATION — learned-grounding detector wired behind GROUND_NET

**Date:** 2026-07-09
**Agent:** NX-6 INTEGRATION (follow-on to `docs/nx6_data.md`, `docs/nx6_train_heatmap.md`,
`docs/nx6_train_centernet.md`, `docs/nx6_judge.md`)
**Checkpoint under test:** `runs/nx6_heatmap_A/model_best.pt` (the heatmap variant
NX-6 JUDGE selected, `conf_thresh=0.59`), `checkpoint/goto_best.pt` (locomotion,
frozen, unchanged).
**Baselines protected (per-episode refs, seed 999 n=15, classical grounder,
`LOCK_M1=1 LOCK_M3=1` shipped defaults):** demo 10/15 (`eval/p4_gate_demo`, fails
0,2,4,5,12), easy 15/15 (`eval/nx2_combined_easy`), search 14/15
(`eval/nx2_combined_search`, fall ep12-scene-fragility).

## TL;DR — VERDICT: net-positive on demo, but REJECT for default-on

`GROUND_NET=1` stays **default OFF**. Code is complete, tested, working, and kept
in the repo as an opt-in backend — but it does not cleanly clear the demo gate's
own ACCEPT bar, so per the task's ADOPT/REJECT protocol it is not flipped
default-on and not synced to the deploy repo.

- **Demo/classical, GROUND_NET=1, seed 999, n=15: 12/15 (reproduced identically
  across 2 full runs).** Fixes 3 of the 4 target episodes (**0, 5, 12** —
  exactly the episodes FA-1 diagnosed as grounding-bound marginal-blob-at-
  wrong-depth / mid-approach distractor-hijack failures, `docs/fa1_failures.md`
  §1) — clears the brief's "fix >=2 of eps 0/2/5/12" sub-bar with a full point
  to spare. **But it reproducibly breaks ep1** (previously a rock-solid
  passer), via a *different, new* mechanism (§2.2) — so the ACCEPT bar's
  "without breaking passers" clause is violated. Net raw count is +2 (10→12),
  but the qualitative bar is not cleanly met.
- **Easy holds 15/15, exactly, zero regressions** (§3).
- **Search holds 14/15, exactly, same single failure** (ep12, fall,
  scene-fragility, unrelated to grounding — §4).
- **Latency is a non-issue**: p95 3.16ms per inference call, integrated
  end-to-end (resize + tensor prep + forward + decode), against a 100-200ms
  (5-10Hz) grounding-cycle budget — >>30x headroom, alongside the 50Hz policy
  (§5).
- **No crashes anywhere**: smoke (1 easy + 1 demo), both full 15-episode
  gates (x2 for demo), and a 1-episode spot-check of the `fancy_demo.py` path
  (which has its own rollout loop, no `lock_mgmt` at all) all ran clean.
- **Why REJECT-for-default despite a net-positive raw count:** this codebase's
  own established discipline across NX-2 through NX-5 (`docs/nx2_final.md`,
  `docs/nx3_size_gate.md`, `docs/nx4_depth_split.md`, `docs/nx5_coherence.md`)
  consistently treats *any reproducible break of a previously-passing episode*
  as disqualifying for default-on adoption, independent of net counts
  elsewhere — e.g. NX-5's M7 was REJECTed specifically for destabilizing one
  passer (ep13) even though it correctly fired on the intended target
  episodes. ep1's break here is the same class of event: reproducible (not
  noise — confirmed via a full-gate rerun, §2.1), mechanistically understood
  (§2.2), and not one of the mechanisms this integration was scoped to fix.
- **The fix mechanism is exactly what NX-5's closure predicted would be
  needed** (`docs/nx5_coherence.md` §CLOSURE: "a learned grounding head...
  could learn the *appearance* features that separate 'this stripe is the
  wall' from 'this stripe is the target'... structurally inaccessible to any
  [lock-management] discriminator tried across NX-2 through NX-5") — and it
  delivered on exactly that promise for eps 0/5/12. The regression is a
  *different* axis (a recall/coverage boundary interacting with the existing
  freeze-forever-on-loss lock behavior, not an appearance-discrimination
  failure), so this is a genuinely promising direction that a follow-on pass
  (§7) could plausibly complete.

---

## 1. What was built (`code/grounding.py`)

Per the brief, wired as an alternative backend **inside `ground()` itself**,
dispatched at the top of the function on `GROUND_NET=1` (default OFF, env
flag). Because every caller in this codebase (`code/inferencer.py`,
`code/eval_search.py`, `code/fancy_demo.py`) already calls `ground()` (aliased
`classical_ground`) with the exact same
`(ego_rgb, ego_depth, target_color, target_shape, intrinsics)` signature, this
required **zero call-site changes** — all three callers pick up the new
backend automatically.

- **Same contract:** returns a `GroundingResult` (dist, cos_th, sin_th,
  confidence, not_visible) — bit-identical shape to the classical path.
  `best_area`/`phys_w`/`phys_h`/split-diagnostics fields are left `None`
  (classical-pipeline-specific; no analogue for a learned heatmap detection —
  see §1.1 for what this means for M1).
- **Same active-camera input:** `intrinsics['is_proximity']` (the same flag
  the classical path already reads) selects `cam_type='proximity'` vs
  `'grounding'` for the detector — the grounding-cam-far / proximity-cam-near
  Schmitt handoff (CAM-2) is completely unchanged and drives which camera's
  frame reaches `ground()` either way. `is_widefov` is handled with a
  one-shot warning + fallback to grounding-pitch geometry (untested
  combination — this integration ran with no `CAMERA_MODE` set, i.e. the
  cam2 champion, per the brief).
- **Query from the existing instruction-target spec:** `target_color`/
  `target_shape` — no new plumbing; verified these are always exactly
  `arena.SHAPES`/`arena.COLORS` names, the SAME vocabulary
  `nx6_heatmap_model.CLASS_NAMES`/`COLOR_NAMES` were trained against (both
  literally derived from the same `arena.py` tables) — a defensive
  vocabulary check still fails safe to `not_visible` if this were ever
  violated.
- **Checkpoint loads once:** `_get_ground_net_detector()` is a lazy,
  sticky-on-failure module-level singleton — one `HeatmapDetector.load()` per
  process, not per episode/cycle. Import of `code.nx6_heatmap_model` is
  deferred to inside this function specifically to sidestep a circular
  import (`nx6_heatmap_model.py` itself imports
  `get_ego_intrinsics_rendered`/`cam_to_egocentric` from `code.grounding`).
- **Inference at the existing grounding cadence** (`GROUNDING_PERIOD`,
  ~5-10Hz) — no new render or cadence logic; `ground()` is called at exactly
  the same points classical grounding already was.
- **Per-cycle latency logged**: `_GROUND_NET_LAT_MS` (module-level list) +
  `ground_net_latency_stats()` helper (n/mean/p50/p95/p99/max) — see §5.
- **Deploy operating point:** `conf_thresh=0.59` (`GROUND_NET_TAU` env var,
  overridable), the val-selected threshold NX-6 JUDGE identified as the one
  that should ship (`docs/nx6_judge.md` §1.2/§1.3/§6), not the looser
  failcase-only-swept 0.29.

### 1.1 M1/M3 lock hygiene — backend-agnostic, confirmed

`code/lock_mgmt.py`'s `LockGate` sits entirely downstream of `ground()`'s
return value and was **not modified**. With `GroundingResult.best_area=None`
for every GROUND_NET detection:
- **M1** (`gate_detection`'s `if LOCK_M1 and area is not None and area <
  M1_AREA_FLOOR_PX2`) becomes a provable no-op for this backend specifically
  — by design, not oversight: a pixel-blob-area floor has no meaning for a
  heatmap detector, whose own `conf_thresh` already serves the "is this
  detection trustworthy" role M1's floor serves for the classical pipeline.
- **M3** (innovation gate + incumbent inertia) is unaffected in its primary
  bearing/distance gating; its area-margin override degenerates to
  "any challenger outside the gate that sustains 2 consecutive cycles
  replaces the incumbent" (since `inc_area <= 0.0` is always true when area
  is always `None`) — still meaningfully hysteretic, just without the
  area-margin refinement.
Both stayed at their shipped ON defaults (opt-out `LOCK_M1=0`/`LOCK_M3=0`,
neither set) for every gate run below.

---

## 2. Demo/classical gate (seed 999, n=15)

### 2.1 Per-episode result table (both full runs, reproducibility check)

| ep | target (dist) | baseline (classical) | run 1 | run 2 | verdict |
|----|---|---|---|---|---|
| 0  | cyan cone (4.32m)    | FAIL fd=3.39 | **SUCCESS** fd=0.37 | **SUCCESS** fd=0.38 | **FIXED**, reproducible |
| 1  | cyan cube (7.42m)    | SUCCESS fd=0.37 | **FAIL** fd=5.88 | **FAIL** fd=5.83 | **BROKEN**, reproducible |
| 2  | blue cone (4.86m)    | FAIL fd=10.66 | FAIL fd=6.21 | FAIL fd=7.17 | unchanged (still fails; final_dist moved but outcome didn't) |
| 3  | red cube (7.00m)     | SUCCESS fd=0.38 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | unchanged (passer holds) |
| 4  | purple ball (7.21m)  | FAIL fd=10.24 | FAIL fd=10.14 | FAIL fd=10.37 | unchanged — **see §2.3, expected** |
| 5  | cyan ball (8.85m)    | FAIL fd=3.52 | **SUCCESS** fd=0.37 | **SUCCESS** fd=0.36 | **FIXED**, reproducible |
| 6  | red cone (8.17m)     | SUCCESS fd=0.36 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | unchanged |
| 7  | cyan cube (5.41m)    | SUCCESS fd=0.37 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | unchanged |
| 8  | red cone (7.86m)     | SUCCESS fd=0.36 | SUCCESS fd=0.36 | SUCCESS fd=0.36 | unchanged |
| 9  | orange cylinder (8.61m) | SUCCESS fd=0.37 | SUCCESS fd=0.36 | SUCCESS fd=0.37 | unchanged |
| 10 | orange ball (6.24m)  | SUCCESS fd=0.37 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | unchanged |
| 11 | yellow cube (6.24m)  | SUCCESS fd=0.36 | SUCCESS fd=0.38 | SUCCESS fd=0.37 | unchanged |
| 12 | cyan cube (6.18m)    | FAIL fd=6.67 | **SUCCESS** fd=0.37 | **SUCCESS** fd=0.36 | **FIXED**, reproducible |
| 13 | blue ball (4.96m)    | SUCCESS fd=0.37 | SUCCESS fd=0.39 | SUCCESS fd=0.39 | unchanged |
| 14 | orange cylinder (6.43m) | SUCCESS fd=0.36 | SUCCESS fd=0.36 | SUCCESS fd=0.36 | unchanged |

**Run 1: 12/15. Run 2: 12/15. Identical pass/fail pattern both times** (only
run-to-run final_dist jitter on the still-failing episodes, consistent with
this harness's documented EGL/physics non-determinism, `docs/cam_p0.md`) —
confirms the ep1 break and the eps 0/5/12 fixes are real, not noise. Per the
brief's protocol ("single-flip noise -> one full rerun"), one full rerun was
run and settled the question: not noise.

- `demo_final_frac = 12/15 = 0.80`
- `demo_fixed_eps = [0, 5, 12]`
- `demo_broken_eps = [1]`

### 2.2 Why ep1 broke — instrumented single-episode replay

Non-invasive monkeypatch of `code.inferencer.classical_ground` (same pattern
FA-1 used for ep0, `docs/fa1_failures.md`), logging every raw `ground()` call
(dist, bearing, confidence, not_visible) alongside the GT distance to the true
target computed from `data_mj.qpos[0:2]`/`target_xy`:

```
step   cam         visible   dist    bearing  conf    gt_dist
   0  GROUNDING    True     7.406    -1.20   0.619    7.423
  90  GROUNDING    True     6.924     1.69   0.731    6.931
 140  GROUNDING    True     6.505     3.07   0.709    6.511
 200  GROUNDING    True     6.004     2.99   0.829    6.001
 230  GROUNDING    True     5.816    15.09   0.681    5.832
 250  GROUNDING    True     5.857    26.63   0.702    5.855
 280  GROUNDING    True     5.863    23.53   0.852    5.880
 300  GROUNDING    False    0.000     0.00   0.000    5.830
 340  GROUNDING    False    0.000     0.00   0.000    5.822
 ... (not_visible every single cycle, step 300 through 1380, 1400-step cap)
```

**Every accepted detection before step ~300 is highly accurate** (dist error
<0.02m vs GT throughout) — this is not a mislocalization problem. What
happens: the target's bearing climbs from ~1-3° (centered) to 15-27° as the
robot's approach angle drifts outward, and around step ~300 the detector's
peak confidence permanently drops below `tau=0.59` and **never recovers for
the remaining ~1100 steps** of the episode (present=False every single
cycle through the 1400-step cap). Downstream: `HOLD_GOAL_HORIZON=100`
expires, and because `LOCK_M5` (bounded coast → rescan) is **default OFF**
(REJECT-verdicted for the classical grounder, `docs/nx2_final.md` — this
integration was scoped to keep it off, per the brief's "M1/M3... on top"),
`cached_goal_vec` **freezes at the last-known goal forever** — the robot
walks confidently toward a stale ~5.8m point and stalls there
(`final_dist≈5.83-5.88m`, matching the frozen last-known distance almost
exactly).

**Diagnosis:** this is a *recall-ceiling × permanent-freeze-on-loss*
interaction, not an appearance-discrimination failure. It is a fundamentally
different mechanism from any of the 5 documented demo failures
(`docs/fa1_failures.md` §1) — the detector isn't confused about *what* the
target is (every detection while visible is accurate); it simply stops
detecting a real, unoccluded, moderate-bearing target past a certain point,
consistent with `docs/nx6_judge.md`/`docs/nx6_train_heatmap.md`'s own honest
val/test recall ceiling (~0.71-0.76 at precision≥0.90 — genuinely not 100%,
training was cut short at epoch 41/60). The classical grounder's much wider
effective recall footprint (full-frame HSV threshold, only 3% side margins,
no bearing-conditioned confidence falloff) evidently kept detecting
(intermittently or continuously) past this point on its own frozen
trajectory and reached the target; GROUND_NET's narrower recall footprint
combined with the existing freeze-forever-on-loss lock behavior (a
pre-existing, unrelated design choice — LOCK_M5 stays REJECTed) turns "one
recall gap on one episode" into a full-episode stall.

One further honest note: `docs/nx6_train_heatmap.md` §3's failcase table
reports demo_ep1 was not even one of the 6 replayed failcase episodes (that
set only covers ep0/2/4/5/12 + search_ep12 — the documented classical
*failures*), so this specific recall gap on a *previously-passing* episode
was never in the training/eval loop's own diagnostic view at all — nothing
in the NX-6 TRAIN/JUDGE pipeline had a chance to catch it, because it isn't
a failcase in the classical grounder's own trajectory distribution. It only
surfaces once the front-end swap changes the robot's approach trajectory
enough to expose a different bearing regime.

### 2.3 ep4 (purple ball) — honest, as instructed: unchanged, and expected to be

FA-1 (`docs/fa1_failures.md` §1) classified ep4 as **compound**: a genuine
total detection miss (target never seen at any bearing during a full scan)
*plus* a likely locomotion/geometry obstruction — and crucially, ep4 is the
**one** demo failure where even the privileged GT-goal rollout also fails
(`fd=2.01m`, the worst of the 3 GT misses). A better front-end detector
cannot fix a failure that persists with a perfect goal signal. Consistent
with this, `docs/nx6_train_heatmap.md` §3's own failcase table independently
found demo_ep4's target is **not visible in any of the 20 captured replay
frames** for this episode at all. GROUND_NET leaves ep4 exactly as failing
as before (fd=10.14/10.37 both runs, close to baseline's fd=10.24) — this is
the expected, honest outcome, not a missed opportunity.

### 2.4 ep2 (blue cone) — unfixed, brief note

Also unchanged (still fails both runs). `docs/nx6_train_heatmap.md` §3's
failcase capture likewise found demo_ep2's target **never visible in any of
its 19 captured frames** (0/19) — under the *classical* grounder's own
frozen trajectory. Under GROUND_NET's different resulting trajectory the
final_dist shifts somewhat run-to-run (6.21m / 7.17m) but the outcome does
not — consistent with the target genuinely not coming into a detectable
view during this episode's approach either way. Not independently
re-instrumented beyond this (out of the brief's required scope, which only
calls for honesty about ep4 specifically).

---

## 3. Easy gate (seed 999, n=15) — holds cleanly

**15/15 = 100.0%, zero regressions, zero change in per-episode outcome**
(all 15 unchanged from the `eval/nx2_combined_easy` baseline — every episode
still succeeds, final_dist ~0.56-0.59m throughout, consistent with easy's
close-range/in-FOV-at-start scene distribution giving the detector
comfortably-visible, well-centered targets the whole episode).

`easy_ok = true`

---

## 4. Search gate (seed 999, n=15) — holds cleanly, per-episode

**14/15 = 93.3%, identical pass/fail pattern to the `eval/nx2_combined_search`
baseline** — the single failure is **ep12** (red cube), tag=`fall`,
`final_dist≈1.99m`, matching the documented "ep12-scene-fragility" baseline
failure exactly (a physical fall shortly after the scan→goto transition,
`docs/fa1_failures.md` §2 — a locomotion/scan-schedule issue, structurally
unrelated to grounding-backend choice). All other 14 episodes succeed
(spot-rate 15/15 = 100%, i.e. GROUND_NET never failed to eventually spot a
target once it entered the search FOV either).

`search_ok = true`

---

## 5. Latency

Dedicated single-process capture (easy n=5 + demo n=5, seed 999, GPU) via
`code.grounding.ground_net_latency_stats()`, covering both the grounding-cam
(far) and proximity-cam (near, via the CAM-2 Schmitt handoff on successful
close approaches) regimes, n=698 inference calls total:

| stat | value (ms) |
|---|---|
| mean | 2.93 |
| p50  | 2.63 |
| **p95** | **3.16** |
| p99  | 4.38 |
| max  | 170.2 (single cold-start CUDA/cuDNN warm-up outlier, first call of the process) |

Budget is the existing 5-10Hz grounding cadence (100-200ms/cycle); p95=3.16ms
leaves **>>30x headroom**, and even the one 170ms warm-up outlier (first
inference call in the process, before CUDA kernels are autotuned) is inside
the loosest (10Hz→200ms is the tighter of the two, but 5Hz→100ms already
covers it once) budget and occurs exactly once per process, not per episode.
This matches (and is slightly higher than, due to added cv2 resize + Python
dict-construction overhead not present in the isolated microbenchmark) NX-6
JUDGE's own isolated GPU benchmark of 1.51ms steady-state
(`docs/nx6_judge.md` §3). The policy's own NN+physics budget (3.44ms/step at
50Hz = 20ms budget) is unaffected — grounding runs at 5-10Hz on a separate
cadence, sharing the same GPU without contention (confirmed via
`nvidia-smi`/`ps` before every launch; no other heavy job was running
concurrently during any of this integration's GPU work).

`latency_ms_p95 = 3.16`

---

## 6. Smoke tests / crash checks

- 1 easy episode (seed 999, ep0, orange cone, 2.40m): SUCCESS, no crash,
  `[grounding] GROUND_NET=1: loaded detector...` printed once.
- 1 demo episode (seed 999, ep0, cyan cone, 4.32m — one of the documented
  classical failures): **SUCCESS** (fd=0.38m) with GROUND_NET=1, no crash.
- `code/fancy_demo.py --smoke --n-smoke 1` (its own separate rollout loop,
  no `lock_mgmt` at all): 1 episode (purple ball, 5.26m, out-of-FOV start),
  **SUCCESS** (fd=0.477m, 1673 steps), no crash. Confirms the "wire it too
  if trivial" path — trivial, because `fancy_demo.py` also just imports
  `ground` from `code.grounding` with the identical call signature, so the
  `GROUND_NET` dispatch inside `ground()` covers it automatically with zero
  changes to `fancy_demo.py` itself.

---

## 7. Files changed / kept / NOT synced

- `code/grounding.py` — additive only: `GROUND_NET`/`GROUND_NET_CKPT`/
  `GROUND_NET_TAU`/`GROUND_NET_DEVICE` env-driven constants, the lazy
  `_get_ground_net_detector()` loader, `ground_net_latency_stats()`,
  `_ground_net()` (the backend itself), and a 6-line dispatch block at the
  top of `ground()`. **Zero lines of the existing classical pipeline changed**
  — confirmed behaviorally (a fresh, no-env `ground()` call after import
  still runs the classical HSV+depth path; `GROUND_NET` defaults to `False`
  on a clean import).
- No changes to `code/lock_mgmt.py`, `code/inferencer.py`,
  `code/eval_search.py`, `code/fancy_demo.py`, `code/nx6_heatmap_model.py`,
  or any other file — this was a pure grounding.py-internal wiring task, per
  the brief's "(or a new code/grounding_net.py imported there)" — folded
  directly into `grounding.py` instead since the backend logic is short
  (~110 lines) and keeping it in one file avoids an extra import hop for
  what is, structurally, one more branch of `ground()`'s existing dispatch
  pattern (it already branches on `is_proximity`/`is_widefov`/`GROUND_SPLIT`
  internally).
- **`GROUND_NET` stays default OFF.** Per the REJECT branch of the task's
  ADOPT/REJECT protocol: no confirm run needed (default is unchanged from
  before this integration), **nothing synced** to
  `VLA_mujoco_unitree/code/` (sync is conditioned on
  ADOPT only).
- Diagnostic/gate scripts used for this integration (scratchpad, not
  committed): `nx6_int_latency.py` (dedicated latency capture),
  `nx6_diag_ep1.py` (ep1 instrumented replay, §2.2).
- Gate artifacts: `eval/nx6_int_smoke/` (1-easy + 1-demo smoke),
  `eval/nx6_int_demo_gate/` + `eval/nx6_int_demo_gate_rerun/` (both full
  demo runs, §2.1), `eval/nx6_int_easy_gate/` (§3),
  `eval/nx6_int_search_gate/` (§4), `eval/nx6_int_latency/` (§5),
  `eval/nx6_int_fancy_smoke/` (§6).

---

## 8. What remains / follow-on

- **The concrete next step this analysis points to**: gate `GROUND_NET=1`
  together with `LOCK_M5=1` (bounded coast → rescan on hold-goal-horizon
  expiry, currently REJECT-verdicted **for the classical grounder**,
  `docs/nx2_final.md`, for unrelated reasons — it was rejected there because
  it triggered *unnecessary* rescans on a detector that already recovers on
  its own via re-detection; that specific failure mode may not transfer to
  GROUND_NET's very different miss profile). If a bounded rescan can
  re-acquire ep1's target once GROUND_NET's confidence permanently drops
  below tau, this would directly fix the §2.2 mechanism without touching
  the detector itself. Out of scope for this integration pass (would need
  its own standalone mechanism-level check + full 3-skill re-gate, matching
  this codebase's own established discipline — not just flipped on
  speculatively).
- **A longer heatmap training run** (`docs/nx6_train_heatmap.md` §5: training
  was stopped early at epoch 41/60, val recall plateaued
  ~0.73-0.76 but loss was still decreasing) could plausibly narrow the
  recall gap behind ep1's break directly, independent of any lock-management
  change.
- **This result should be read as a genuine confirmation of NX-5's closure
  thesis**, not a failure of it: the learned front-end fixed exactly the
  episodes (0, 5, 12) that four independent lock-management mechanisms
  (NX-2 through NX-5) proved were structurally unreachable from
  (size, depth, distance, odometry)-only signals — the appearance-based
  discrimination worked. What remains is an orthogonal, addressable
  infrastructure gap (recall ceiling × freeze-forever lock behavior), not a
  refutation of the approach.
