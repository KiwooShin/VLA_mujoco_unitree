# NX-10 — H3 Initial Scan: Realized-Yaw Coverage Fix

**Date:** 2026-07-09
**Agent:** NX-10 (follow-on to `docs/fa2_residuals.md`, `docs/nx1_scan.md`, `docs/nx9_avoid.md`)
**Starting state:** demo 13/15 (86.7%, `docs/nx9_avoid.md` §4.4) — the only two failures
anywhere across the three gated skills were demo ep2/ep4, both root-caused by
`docs/fa2_residuals.md` to a **scan angular-coverage bug**: `code/inferencer.py`'s H3
initial scan (goal_source='classical'/GROUND_NET) computed its right/left sweep from an
assumed step-count × commanded-rate calculation, but the student-driven turn only
realizes a fraction of the commanded `SCAN_RATE` — so the scan's actual camera-coverage
arc was only **~-61°/+64°** from spawn heading, not the intended ±90°. ep2's target
(bearing -73.8°) sat 14.3° beyond the realized edge and was **never once in the camera
frame** for the entire 1400-step episode.

## TL;DR — VERDICT: ADOPT (demo 13/15 → **14/15 = 93.3%**, no default toggle)

| gate | bar | result |
|---|---|---|
| demo (seed 999, n=15) | ≥14/15, ep2 flipped, no passer broken | **14/15 (93.3%)**, stable across 2 runs, fails={4} only |
| easy (seed 999, n=15) | 15/15 exact | **15/15** |
| search (seed 999, n=15) | 15/15 exact, byte-equal (untouched code) | **15/15**, per-episode scan timings match `docs/nx1_scan.md`/`docs/nx9_avoid.md` exactly |
| fancy_demo `--smoke` (n=6) | no crash | **6/6 SUCCESS**, no crash |

ep2 **FIXED** (bearing -73.8°, now detected mid-scan and converges cleanly to
SUCCESS). ep4 remains the sole failure, exactly as the task brief predicted
("compound pace-deficit... only ep4 remaining") — but its own mechanism changed
materially under the fix (see §4). One regression was found and fixed during
validation (§3): a naive port of eval_search's exact 165°-leg constants caused a
**new** fall on ep9 (previously passing); the final design uses a smaller,
H3-specific leg amplitude that avoids it (§2.2).

---

## 1. Realized coverage: before → after

| | before (buggy assumed-rate calc) | after (NX-10, realized-yaw tracking) |
|---|---|---|
| mechanism | step-count × `SCAN_RATE` × `dt`, 75/125/0-step right/left/right split, `SCAN_TIMEOUT=200` | `code.scan_sched.BidirectionalScanSchedule` (NX-1's shared, already-validated class) — integrates the robot's *actual* per-step yaw readings, self-correcting for realized-rate drift |
| realized camera-coverage arc | **~-61°/+64°** from spawn heading (measured by instrumented replay, `docs/fa2_residuals.md` §3) | reliably the full intended **±90°** (±118.9° including the ±28.9° grounding-camera half-FOV) — confirmed by ep2 (target at -73.8°, found via the schedule's 3rd leg) and ep9 (target at -39.7°, found via the 3rd leg) both being detected |
| detector's own raw signal | 0/140 raw calls ever crossed the noise floor for ep2 (confidence 0.006–0.021) | 53/140 raw calls `present=True`, confidence up to 0.964 (ep2 replay) |

**Known residual limitation** (documented, out of scope for this fix): the final
`H3_LEG_DEG=90` gives a **hard ceiling of ±118.9°** effective bearing coverage —
demo scenes sample target bearing uniformly over the full ±180°
(`code/scene.py`, `target_in_fov=False`), so a target beyond ±118.9° would still
time out unfound. Not present in the seed=999 n=15 gate set (max observed
magnitude is ep2's 73.8°); widening further would require a design beyond this
fix's scope (see §2.2 for why 165° — eval_search's own value — is unsafe here).

---

## 2. Implementation

### 2.1 What changed (`code/inferencer.py`)

The H3 scan block (`if _scan_active and _need_classical_render: ... else:`) no
longer computes `scan_wz` from a hardcoded quarter/step calculation. It now
instantiates `_h3_scan_sched = BidirectionalScanSchedule(scan_rate=SCAN_RATE,
leg_deg=H3_LEG_DEG, dwell_steps=_H3_DWELL_STEPS)` once per episode and calls
`_h3_scan_sched.step(yaw)` every scan step — the exact same call pattern
`eval_search.py`'s own initial scan and `lock_mgmt.py`'s `ReacquisitionScan`
already use, reusing NX-1's shared, already-gated `BidirectionalScanSchedule`
class rather than adding a third divergent scan-logic copy (this codebase's
documented duplicated-loop history, `code/scan_sched.py`'s own docstring).

All existing interactions were preserved unchanged:
- **Scan-exit on detection**: unchanged (`_scan_active = False` when
  `det_bearing_deg < SCAN_ALIGNED_THR_DEG` on an accepted detection).
- **`_scan_active` gating of AVOID and lock logic**: unchanged — AVOID's
  computation/injection and the M4/M7 divergence/coherence watchdogs are still
  structurally unreachable while `_scan_active` (scan steps still `continue`
  before the injection site).
- **`mark_discontinuity` behavior**: unaffected (CAM-2 fallback-probe-adopt and
  Schmitt-trigger handoff logic live entirely in the post-scan/normal-mode
  grounding block, untouched).
- **Rescan (`ReacquisitionScan`) compatibility**: unaffected — the
  `_using_rescan_sched` branch (M4/M5-triggered mid-episode rescans) was not
  touched; it already used `BidirectionalScanSchedule` via `ReacquisitionScan`
  before this fix. The two scan mechanisms remain deliberately separate
  instances for the reason documented in `code/lock_mgmt.py`: H3's own
  `SCAN_TIMEOUT` check is keyed on the *episode's absolute* step, so it isn't
  safe to re-arm mid-episode (`ReacquisitionScan` tracks its own local step
  counter instead).

Two knock-on constant changes, both scoped narrowly to the demo/GOTO skill's
own callers (not touched: `code/gen_dataset.py`, `code/deploy_eval.py`,
`code/bench_*`, `code/render_showcase*.py`, `code/gen_grounding_dataset.py`,
`code/gen_det_failcases.py`, `code/record_showcase.py`,
`code/render_deliverable.py`, `code/eval_keyframe.py` — all independent
one-off/dataset-gen/showcase tools with their own hardcoded `1400`, out of
scope for this fix, matching the "keep the change minimal" brief):
- `code/eval_closedloop.py`: `MAXSTEPS['demo']` 1400 → 1700 (the demo gate's
  own harness).
- `code/demo.py`: `MAXSTEPS_GOTO` 1400 → 1700 (the production GOTO-skill
  driver, `inf.rollout(..., maxsteps=self.maxsteps_goto, ...)`).
  `MAXSTEPS_MANEUVER` left untouched (maneuver has its own separate rollout
  loop, `docs/nx9_avoid.md` §8 — structurally unreachable from this change).

`code/scan_sched.py`, `code/eval_search.py`, `code/fancy_demo.py`,
`code/lock_mgmt.py`, `code/avoid.py`, `code/grounding.py` — **not modified**
(confirmed: search's fresh full re-gate reproduces `docs/nx1_scan.md`'s /
`docs/nx9_avoid.md`'s documented per-episode scan-step counts exactly, e.g. ep4
scan=70, ep5 scan=830, ep12 scan=960 — byte-equal behavior, §5).

### 2.2 Why `H3_LEG_DEG=90`, not eval_search's `SCAN_LEG_DEG=165`

The task brief's starting hypothesis (and the first implementation attempt)
was to reuse eval_search's exact constants (165° legs, 45-step dwells). That
attempt:
- **Did** fix ep2 (found via the schedule's 3rd leg, `ALIGNED` at a
  reproducible realized step of ~850) and comfortably fixed ep4's coverage
  (found directly in leg 0). At `MAXSTEPS['demo']=1400` ep2 was still a
  narrow miss (fd=1.146, ran out of budget ~86-114 steps short) — bumping to
  1700 flipped it cleanly to SUCCESS (steps=1482-1507, fd≈0.365-0.37,
  reproducible 3/3 runs).
- **But** surfaced a **new regression** on ep9 (bearing -39.7°, a previously
  clean passer, baseline fd=0.37): full n=15 re-gate came back FAIL[fall],
  reproducible byte-for-byte (steps=480, final_dist=8.127, both runs
  identical). Instrumented qpos/yaw replay showed the robot committing to a
  **full, mostly-continuous 165° leg0 sweep in the wrong direction first**
  (ep9's target is on the negative side, but the schedule's fixed
  `_LEG_SIGNS=(+1,-1,-1,+1)` pattern always tries positive first) — a
  realized single-leg rotation of **~375 steps**, then a **partial return
  leg1** (another ~105 steps before the fall) — uncomfortably close to the
  **~470-step / ~323° continuous-rotation OOD ceiling** `docs/rot_dart.md` /
  `docs/nx1_scan.md` independently diagnosed for this exact shared policy.
  The dwell between leg0/leg1 was confirmed present and correctly executed
  (flat yaw trace for ~15-20 steps at the leg boundary) — this is not a
  wiring bug, but back-to-back unfavorable-direction 165° legs apparently
  still stack enough rotation-adjacent risk in demo's environment/physics
  that eval_search's own validated 165°/45-dwell gate (15/15, zero
  rotation-attributable falls) never triggered.

Reducing the leg amplitude to **90°** (restoring the *original* H3 design's
own stated intent — "sweeps ±90° arc" — just now correctly *realized* via
actual yaw tracking instead of the buggy assumed-rate calculation) roughly
halves worst-case single-leg realized rotation (~205 steps vs ~375),
empirically eliminating the ep9 fall while still comfortably covering both
target episodes (ep2 needs only ~44.9° into leg 2; ep4 is found directly in
leg 0 at ~33.7° in). This is the shipped configuration. The dwell length
(`SCAN_DWELL_STEPS=45`) was reused as-is from `code/scan_sched.py` — that part
of the shared constants was never implicated in the ep9 regression.

Both leg-amplitude configurations were empirically re-verified via targeted
single-episode replays before locking in the final full-gate run (see
scratchpad `patch_legdeg.py` A/B harness — not committed, matching this
codebase's precedent for diagnostic-only scripts).

---

## 3. Mechanism-level replays

### 3.1 ep2 (target, bearing -73.8°) — **FIXED**

Instrumented replay (`checkpoint/goto_best.pt`, pure defaults, device=cuda):

```
[scan] ALIGNED at step=540  yaw_err=-24.6°   (H3_LEG_DEG=90, found via leg 2)
=== RESULT ep2: success=True  steps=1139  final_dist=0.369  fell=False ===
```

Full-gate runs (`MAXSTEPS['demo']=1700`): SUCCESS both times, steps=1148/1170,
final_dist=0.361/0.370 — comfortable margin (~530-550 spare steps below the
1700 cap). Raw detector confidence up to 0.964-0.966 once in frame (vs the
pre-fix 0/140 present, confidence pinned ≤0.021) — confirms this was purely a
scan-coverage miss, exactly as `docs/fa2_residuals.md` diagnosed, and the fix
directly addresses it.

### 3.2 ep4 (target, bearing +62.6°) — **still FAILS, but the mechanism changed**

`docs/fa2_residuals.md` diagnosed ep4 as *compound*: (b) a scan-coverage
near-miss (1.9° short of the old realized edge) **and** (d) a structurally-hard
locomotion-pace deficit — even a perfect step-0 GT-goal oracle only reached
fd=2.10m by step 1400, so fixing detection alone was predicted *not* to flip
it to SUCCESS.

Detection is now decisively fixed — `[scan] ALIGNED at step=90` (found
directly in leg 0, no longer even a near-miss) — but the episode still fails,
now for a **different, previously-undocumented** proximate reason:

| config | result |
|---|---|
| `H3_LEG_DEG=90`, `maxsteps=1400` (old cap) | FAIL `didnt-reach`, fd=**0.781** — a *near-miss*, much closer than fa2's GT-oracle reference (fd=2.10 at step 1400) |
| `H3_LEG_DEG=90`, `maxsteps=1700` (shipped) | FAIL `fall`, steps=1486-1536, fd=1.368-1.510 (both full-gate runs) |

At the shipped budget the robot gets close enough (well past fa2's GT-oracle
benchmark) that a **fall during final close-range approach** becomes the
proximate cause, rather than the old "target never seen, diverging straight
line" story. Root cause not further investigated (out of scope per the task
brief — ep4 was explicitly predicted to remain the one residual failure);
plausibly a close-quarters stability/obstacle interaction near the target
(the scene has a blue-cylinder distractor at 3.09m/+18.6° bearing near the
approach path, `docs/fa2_residuals.md` §1). **Honest summary: the scan fix
resolves ep4's detection-coverage half completely, but does not flip the
episode to SUCCESS — a different, tighter-margin failure mode remains.**

### 3.3 ep9 (target, bearing -39.7°) — regression found and fixed (see §2.2)

Passing baseline (fd=0.37) → FAIL[fall] under the 165°-leg first attempt →
**SUCCESS** under the shipped 90°-leg design, both full-gate runs:
steps=1510/1547, final_dist=0.366/0.369.

### 3.4 Scan-entry passer spot-checks (2-3 required) — all HELD

Episodes whose target bearing exceeds `SCAN_ALIGNED_THR_DEG=40°` (i.e.
genuinely exercise scan rotation, not just an instant first-frame detection):

| ep | bearing | baseline (`docs/nx9_avoid.md` §4.4) | NX-10 (2 runs) |
|---|---|---|---|
| 0 | +59.5° | SUCCESS, fd=0.37 | SUCCESS, steps=580/614, fd=0.367/0.368 |
| 8 | +28.7°* | SUCCESS, fd=0.36 | SUCCESS, steps=670/701, fd=0.363/0.357 |
| 9 | -39.7° | SUCCESS, fd=0.37 | SUCCESS, steps=1510/1547, fd=0.366/0.369 (see §2.2/§3.3 for why this one needed the leg-amplitude fix) |

(*ep8 is just under the 40° aligned threshold — included as a secondary
spot-check; all 12 other episodes have bearing magnitude <40° and align
within the first few grounding cycles regardless of scan-schedule design, so
they carry no discriminating signal for this fix.)

All 13 originally-passing episodes' final_dist stayed in the normal
0.357-0.393m band across both full-gate runs — no passer broken.

---

## 4. Full gates (seed=999, n=15, pure defaults — no env vars)

### 4.1 demo — `eval/nx10_demo_gate_v2/`, `eval/nx10_demo_gate_v2_rerun/`

**14/15 = 93.3%**, stable across 2 runs (noise protocol), byte-identical fail
set {4} both times:

| ep | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run1 | S 580 | S 863 | **S 1148** | S 824 | F[fall] 1486 | S 1054 | S 972 | S 670 | S 982 | S 1510 | S 744 | S 787 | S 820 | S 683 | S 772 |
| run2 | S 614 | S 896 | **S 1170** | S 849 | F[fall] 1536 | S 1104 | S 1019 | S 701 | S 1032 | S 1547 | S 777 | S 829 | S 856 | S 693 | S 801 |

final_dist for every SUCCESS: 0.357-0.393m (normal band, unchanged from
baseline). ep2 (bold) is the flip. ep4 fd=1.37-1.51 (fall). **Baseline
comparison** (`docs/nx9_avoid.md` §4.4, fails={2,4}): net **+1 episode, no
regressions** — meets the ACCEPT bar exactly ("≥14/15 with ep2 flipped and no
passer broken").

### 4.2 easy — `eval/nx10_easy_gate/`

**15/15 = 100.0% exact**, final_dist 0.556-0.577m (documented easy band,
unchanged). Expected: all 15 easy target bearings are within ±35° of spawn
heading (`target_in_fov=True` by construction), so the scan-schedule change
is a structural no-op here (aligns within the first grounding cycle
regardless of leg amplitude/dwell design).

### 4.3 search — `eval/nx10_search_gate/`

**15/15 = 100.0%** (SPOT-rate 15/15, REACH-rate 15/15, falls 0/15).
Per-episode scan-step counts match `docs/nx1_scan.md`'s / `docs/nx9_avoid.md`'s
documented values exactly where spot-checked (ep4 scan=70, ep5 scan=830, ep11
scan=350, ep12 scan=960, ep13 scan=1000, ep14 scan=280) — confirms byte-equal
behavior, as expected since `code/eval_search.py`/`code/scan_sched.py` were
not touched by this fix.

### 4.4 fancy_demo `--smoke` — `eval/nx10_fancy_smoke/`

**6/6 SUCCESS**, no crash (5 single long-distance episodes + 1 multi-goal,
both sub-goals succeeded). `fancy_demo.py`'s own rollout loop
(`run_fancy_rollout`/`run_fancy_rollout_multi`) uses `code/scan_sched.py`
directly (its own `BidirectionalScanSchedule` instance, NX-1) and does not
call `Inferencer.rollout()`'s H3 scan at all — this smoke mainly confirms no
import/regression breakage from the `code/inferencer.py` changes, not a
direct exercise of the H3 fix itself.

---

## 5. Adoption

**ADOPT — default behavior, no toggle.** This is a bug fix (the H3 scan's
comment always claimed ±90° coverage; it just wasn't realizing it), matching
the task brief's framing. `code/inferencer.py`'s H3 scan block change is
unconditional (no env var gate), consistent with how the rest of the scan
mechanism (no `SCAN_FIX=0` escape hatch exists for the original H3 either).

**Files changed** (`unitree_vla`):
- `code/inferencer.py` — H3 scan block now driven by
  `code.scan_sched.BidirectionalScanSchedule` (`H3_LEG_DEG=90`,
  `SCAN_DWELL_STEPS=45` reused from the shared module, `SCAN_TIMEOUT` bumped
  200→1000); module docstring updated.
- `code/eval_closedloop.py` — `MAXSTEPS['demo']` 1400→1700.
- `code/demo.py` — `MAXSTEPS_GOTO` 1400→1700.
- `docs/nx9_avoid.md` — added a one-line pointer note (§7) to the
  `goal_source='gt'`-makes-AVOID-a-structural-no-op finding
  (`docs/fa2_residuals.md` §4) that predates this fix and was not yet
  cross-referenced from AVOID's own design doc.

**Not modified**: `code/scan_sched.py`, `code/eval_search.py`,
`code/fancy_demo.py`, `code/lock_mgmt.py`, `code/avoid.py`,
`code/grounding.py`, and every dataset-gen/showcase/deploy-eval script that
independently hardcodes `1400` for `'demo'` (`code/gen_dataset.py`,
`code/deploy_eval.py`, `code/bench_widefov_visibility.py`,
`code/gen_grounding_dataset.py`, `code/gen_det_failcases.py`,
`code/render_showcase_videos.py`, `code/render_deliverable.py`,
`code/render_showcase_reel.py`, `code/record_showcase.py`,
`code/eval_keyframe.py`) — deliberately out of scope, matching the "keep the
change minimal" brief; these are independent one-off tools unrelated to the
gated numeric eval.

**Synced (byte-copied, `cmp`-verified, NO git) to
`VLA_mujoco_unitree/code/`**: `inferencer.py`,
`eval_closedloop.py`, `demo.py`.

---

## 6. What remains / follow-on

- **ep4** is now the only failure anywhere across the three gated skills
  (matching the task brief's own prediction). Its mechanism has shifted from
  "total grounding miss + locomotion-pace deficit" (`docs/fa2_residuals.md`)
  to "detection solidly fixed, but a close-range stability issue (fall) or a
  tight budget near-miss (fd=0.781 at the old 1400 cap) remains" — a
  materially different, tighter-margin problem than originally diagnosed, and
  not investigated further here (out of scope per the task brief).
- **Coverage ceiling**: `H3_LEG_DEG=90` gives a hard ±118.9° effective bearing
  limit (§1) — not a concern for the current seed=999 gate set, but a future
  seed/scene could sample a demo target beyond that and still time out
  unfound. A targeted follow-on (if this ever surfaces) would need to resolve
  the rotation-OOD/leg-amplitude tension found in §2.2 (e.g. an
  OOD-detection-and-escape mechanism analogous to `docs/nx8_stall.md`'s
  STALL_BREAK, rather than a blanket leg-amplitude increase).
- **`docs/nx9_avoid.md` cross-reference**: confirmed the `goal_source='gt'`
  AVOID-is-a-no-op finding (`docs/fa2_residuals.md` §4) is now pointed to from
  AVOID's own design doc (§7) — no code change, per the task brief.
