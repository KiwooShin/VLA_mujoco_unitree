# NX-16 — Cone Near-Target Stall: Diagnosis and Fix

**Date:** 2026-07-10
**Agent:** NX-16
**Scope:** `code/fancy_demo.py`'s `run_fancy_rollout()` (and, by extension,
`run_fancy_rollout_multi()`, which calls it per sub-goal) — the same functions
DR-1's reliability sweep (`docs/dr1_demo_reliability.md` §3.2) exercised. Does
**not** touch `code/inferencer.py`, `code/eval_search.py`, `code/lock_mgmt.py`,
or `code/grounding.py` — the gated eval protocols and their default behavior
are unmodified.

---

## TL;DR

- **Mechanism confirmed by direct instrumentation (Suspect "other", not #1/#2/#3
  from the task brief):** the `cone` shape's detection confidence under
  `GROUND_NET=1` (the default learned heatmap detector,
  `runs/nx6_heatmap_B/model_best.pt`) decays steadily as range closes under the
  GROUNDING camera and drops below `GROUND_NET_TAU=0.64` right around 1.6-1.8m
  — just above the `CAM_D_LO=1.2m` GROUNDING→PROXIMITY handoff threshold — so
  the target is lost for good (PROXIMITY also fails to reacquire it at that
  range). `code/fancy_demo.py`'s rollout loop had **no recovery path** for a
  detection lost this way: `cached_goal_vec` freezes forever once
  `_frames_since_det` exceeds `HOLD_GOAL_HORIZON=100`, and the image-blind
  goto policy (it is fed `ego_rgb=zeros`; navigation is driven entirely by the
  injected egocentric `(dist, cosθ, sinθ)` goal vector) keeps consuming that
  stale, never-re-grounded vector — pure open-loop dead reckoning that curves
  past the true target and settles into a stable orbit. This is exactly DR-1's
  "approach to 0.6-0.9m, reverse, rock-stable plateau" signature.
- **Both suspects #1 and #2 are ruled out by the trace evidence** (below).
  Suspect #3 (AVOID) is also ruled out (`avoid_bias_wz` strictly 0.0 throughout
  every stall window on all 3 seeds).
- **This is deploy-side and fixable minimally**: `code/inferencer.py` and
  `code/eval_search.py` already carry the exact mechanism needed
  (`lock_mgmt.py`'s "M5: coast-expiry → drop lock + bounded
  `ReacquisitionScan`"), but `code/fancy_demo.py` never imported `lock_mgmt`
  at all — a straight parity gap, not a new invention. Ported a **locally
  scoped** version of the same recovery into `run_fancy_rollout()` (does not
  read or write `LOCK_M5` / `lock_mgmt.LockGate`, so the gated scripts' default
  behavior — and `LOCK_M5`'s existing REJECT verdict, see §4 below — are both
  completely untouched).
- **Mechanism-test: 3/3 of the previously-still-failing-at-5000-steps cone
  seeds now succeed** (bar was ≥2/3), confirmed across two repeated runs.
  **All 5 of DR-1's originally-capped wild episodes now succeed** (5/5,
  including the 2 that already succeeded before this fix). **Zero falls**
  across every post-fix rerun. Gates hold at their documented baselines
  exactly: demo999 14/15, easy999 15/15, search999 15/15 (unaffected by
  construction — neither gated script imports `fancy_demo.py`).
- **Verdict: ADOPT.** Synced to `VLA_mujoco_unitree/code/`
  (plain file copy, no git).

---

## 1. Reproduction

DR-1's 3 residual failures (`docs/dr1_demo_reliability.md` §3.2) are Sweep-A
episodes 11 (yellow cone, 6.84m, bearing −149.6°), 19 (red cone, 7.64m,
bearing −154.8°), and 27 (yellow cone, 8.64m, bearing −142.8°), built via
`np.random.default_rng(np.random.SeedSequence([424242, ep]))` +
`dr1_sweep.make_scene()` (scratchpad harness). All 3 reproduced cleanly on the
pre-fix code with the exact documented signature: monotonic approach to
0.6-0.9m, reversal, rock-stable plateau at 1000+ steps, no fall.

An instrumented copy of `run_fancy_rollout()` was built (source-level extract
of the real function, `exec`'d inside `code.fancy_demo`'s own module namespace
so every free variable resolves identically — no repo files touched for this
step) that logs, every grounding cycle and every sim step: raw detector
distance, EMA'd distance, ground-truth distance (from `data_mj.qpos[0:2]` vs
the object's true world `(x,y)`), active camera, detector confidence, and the
AVOID bias term.

## 2. Suspects ruled out

- **#3 AVOID:** `avoid_bias_active_frac == 0.0` for all 3 episodes end to end
  (confirmed both via the trace's per-step `avoid_bias_wz` field, which is
  `+0.000` for the entire stall window, and via the returned
  `avoid_bias_active_frac` metric). Ruled out cheaply as instructed.
- **#1 Stop-distance/grounding bias:** **Not the mechanism.** The trace shows
  the detector's raw reported distance tracks ground truth to within a few cm
  the *entire* approach, right up to the moment detection is lost — e.g. ep19
  at step 2030: `raw_dist=1.682m` vs `gt_dist=1.684m`. There is no systematic
  high/low bias in the distance *value*; the detector's confidence simply
  falls below threshold and it stops firing at all (see §3).
- **#2 Success radius vs. physical footprint:** **Not the mechanism.** The
  robot never gets stuck at a fixed collision-limited standoff — GT distance
  actively *oscillates* through the 0.6-0.9m band and beyond (down to 0.52m in
  one case) before reversing, which is inconsistent with a hard physical
  floor. The `stop_r=0.5m` vs. cone-base-radius arithmetic was checked too
  (cone's placement half-size is 0.26m, same order as ball/cube's 0.24m — not
  dramatically larger) and isn't the limiting factor; the real distinguishing
  geometry turned out to be cone *height*, not footprint (see §3).

## 3. Root mechanism

`code/grounding.py` dispatches to `_ground_net()` by default
(`GROUND_NET=1`), the NX-6/NX-14 learned heatmap detector
(`runs/nx6_heatmap_B/model_best.pt`, confirmed loaded at
`conf_thresh=GROUND_NET_TAU=0.64` in every run's log). Instrumented `detect`
events for ep19 (red cone, 7.64m) show confidence decaying smoothly and
monotonically as range closes under the GROUNDING camera (26° pitch):

| step | gt_dist | raw_dist | confidence |
|---|---|---|---|
| 990 | 7.664 | 7.663 | 0.978 |
| 1500 | 4.433 | 4.424 | 0.990 |
| 1900 | 2.418 | 2.355 | 0.906 |
| 1980 | 1.971 | 1.975 | 0.801 |
| 2000 | 1.846 | 1.845 | 0.768 |
| 2020 | 1.752 | 1.767 | 0.740 |
| **2030** | **1.684** | **1.682** | **0.793 (last detection)** |
| 2050-2699 | 0.78 → 1.58 (oscillating) | **frozen at 1.792** | *(no detections at all)* |

After step 2030 the target is never detected again by either camera for the
rest of the (pre-fix) 2700-step run: `active_cam` stays `GROUNDING` forever
(the Schmitt-trigger handoff to `PROXIMITY` only updates on a *successful*
detection, so it can't fire once detection stops), and the CX-3 fallback
probe (`_cam_miss_count >= 2` → try the other camera) does trigger every
cycle (its gate condition, `_last_known_goal[0] <= CAM_PROXIMITY_D_FAR=1.81`,
is satisfied — `1.792 <= 1.81`) but `PROXIMITY` also fails to detect the cone
at this range in every attempt.

Cross-referencing `code/arena.py`'s object mesh (`SHAPES`, `_add_geom`): a
cone is built as a base cylinder (height ≈ `hs*2.2`) plus a stacked narrower
tip box, for a **total object height of ≈0.54m**, vs. 0.24m for ball/cube and
0.35m for cylinder — the cone is **1.5-2.3x taller** than every other shape,
not merely wider. The working hypothesis (consistent with, but not required
to prove, the fix below) is that the cone increasingly clips out of the
GROUNDING camera's frame as range closes, degrading the query-conditioned
detector's confidence below `GROUND_NET_TAU` in a band that straddles the
GROUNDING/PROXIMITY handoff boundary, and that PROXIMITY (58° pitch) doesn't
reliably pick up the slack for this specific shape/range combination.

**Downstream consequence (the actual crash-shaped bug):**
`run_fancy_rollout()`'s goto step feeds the model `ego_rgb=torch.zeros(...)`
— navigation is 100% driven by the injected `cached_goal_vec` (egocentric
`dist, cosθ, sinθ`), never by real vision at that point in the loop. Once
`_frames_since_det > HOLD_GOAL_HORIZON=100`, `cached_goal_vec` simply stops
updating (pre-fix code had no `elif` branch after the hold-goal window — it
just silently freezes). The policy then keeps producing actions conditioned
on a **stale, never-re-grounded egocentric vector** — since it's not
re-anchored to the robot's actual (continuing to move) pose, this is pure
open-loop dead reckoning. A small residual bearing/yaw drift compounds over
hundreds of steps into a curving path that swings past the true target and
settles into a **stable limit-cycle orbit** — exactly DR-1's "reverse, then a
rock-stable plateau" signature, at plateau distances (1.46m, 2.4m, 4.76m)
that match the marginal equilibria of an open-loop circular walk.

## 4. Why this wasn't already fixed by the existing lock-management machinery

`code/lock_mgmt.py` (NX-2, `docs/rs1_lock_mgmt.md`) implements exactly this
recovery as mechanism **M5** ("bounded coast → reroute to rescan after
hold-goal-horizon expiry"), shared by `code/inferencer.py` and
`code/eval_search.py`. Two things kept it from helping here:

1. **`code/fancy_demo.py` never imported `lock_mgmt` at all** — it has its own
   independent (older/simpler) `HOLD_GOAL_HORIZON` + `_frames_since_det`
   bookkeeping that was never wired to the M5 recovery path. Pure parity gap.
2. Even in the two files that *do* import `lock_mgmt`, **`LOCK_M5` defaults
   OFF** (`docs/nx2_final.md`: "M2, M4, M5 ... REJECT"). Re-reading the
   isolation record (`docs/nx2_iso.md` §"M2+M5"), though: M5 was **never
   independently gated at full (15-episode) scale** — it was only ever tested
   bundled with M2 (which broke episode 13), and the one *targeted* bisection
   that isolated M5 alone found it clean (`LOCK_M5=1` only on the broken
   scene: "SUCCESS, steps=604 ... M5 alone showed no adverse effect ... not
   implicated in this regression"). So M5-the-mechanism isn't validated-bad;
   it's just unvalidated at scale, and the REJECT verdict was really about M2.

Given that, the fix below reuses the same building blocks
(`code.lock_mgmt.ReacquisitionScan`, which is just a thin, already-tested
wrapper around NX-1's `BidirectionalScanSchedule` with its own local step
counter — safe to instantiate mid-episode, unlike re-arming the absolute-step
`SCAN_TIMEOUT`) but implements the trigger **entirely locally** inside
`run_fancy_rollout()`, never reading or writing `LOCK_M5` / `LockGate`. This
means: (a) it cannot regress `code/inferencer.py` or `code/eval_search.py` in
any way — they don't import `fancy_demo.py`, and this change doesn't touch
`lock_mgmt.py` — and (b) the fix is unconditionally active in
`fancy_demo.py` rather than being gated behind an opt-in env var, since this
file's specific problem (no recovery mechanism *at all*) is worse than the
general-purpose M5's REJECTed tradeoffs might be at full scale elsewhere.

## 5. Fix

`code/fancy_demo.py`, inside `run_fancy_rollout()`:

- Import `ReacquisitionScan` from `code.lock_mgmt`.
- Add local state (`_using_rescan_sched`, `_rescan_sched`,
  `_rescan_local_steps`) and a `_lock_drop_and_rescan()` closure that resets
  the EMA/last-known-goal, re-enters scan mode via a **fresh**
  `ReacquisitionScan`, resets `cached_goal_vec` to the same default
  `[2.0, 1.0, 0.0]` the original never-spotted-scan-timeout fallback already
  used, and zeroes the AVOID bias for a fresh read once normal mode resumes.
- In the detection-miss branch, once a previously-spotted lock has been
  missing for `> HOLD_GOAL_HORIZON` frames (and we're not already mid-rescan),
  call `_lock_drop_and_rescan()` instead of silently continuing to freeze.
- In the scan-mode step dispatch, branch on `_using_rescan_sched`: drive
  `_rescan_sched.step(yaw)` instead of the original `_scan_sched` (whose
  `SCAN_TIMEOUT` is keyed on the *absolute* episode step and would time out
  on literally the next cycle if reused mid-episode — this is exactly why
  `ReacquisitionScan` exists as a separate local-counter wrapper, per its own
  docstring). On timeout, fall back to the default goal vector (same
  fallback the original scan-timeout path already used) rather than freezing
  forever again.

**Safety refinement found necessary during mechanism-testing (not present in
the first draft):** `ReacquisitionScan`'s built-in bound reuses the shared
`SCAN_TIMEOUT=1150` constant, sized for the *initial* blind scan (unknown
bearing). A coast-expiry rescan starts from a much better prior and in
practice reacquired within ~300-330 steps whenever the target was actually
re-detectable. But on one seed (ep19) where the cone briefly sat in a range
neither camera could detect *at all* (independent of bearing — pure yaw
rotation cannot fix a depth-band blind spot), letting the rescan run for
nearly the full ~1150-step bound before an eventual drift-induced
reacquisition produced an abrupt scan→goto transition that **fell** on 1 of 2
repeated runs (the other repeat instead just timed out safely — consistent
with sitting right at the edge of the walking policy's competence envelope
for an atypically long uninterrupted turn-in-place, not a deterministic bug).
Added a **local** `NX16_RESCAN_MAX_STEPS = 600` cap (~2x the observed
successful-reacquisition time) so this file's own rescan gives up and falls
back to the (already-proven-non-falling) default-goal behavior well before
reaching that instability regime, instead of running the full shared bound.
After this change: **3/3 mechanism-test seeds succeeded with zero falls**,
confirmed across two independent repeated runs (6 episode-runs total, 0
falls).

## 6. Mechanism-test results (3 previously-still-failing-at-5000 seeds)

Real (non-instrumented) `run_fancy_rollout()` call path, 5000-step budget,
same seeds as DR-1's original sweep:

| ep | color | shape | dist | pre-fix @5000 | post-fix (run 1) | post-fix (run 2) |
|---|---|---|---|---|---|---|
| 11 | yellow | cone | 6.84m | FAIL didnt-reach, fd=4.756m | **OK**, fd=0.460-0.470m, no fall | **OK**, fd=0.470m, no fall |
| 19 | red | cone | 7.64m | FAIL didnt-reach, fd=1.455m | **OK**, fd=0.460-0.477m, no fall | **OK**, fd=0.460m, no fall |
| 27 | yellow | cone | 8.64m | FAIL didnt-reach, fd=2.399m | **OK**, fd=0.473-0.486m, no fall | **OK**, fd=0.495m, no fall |

**3/3 flipped** (bar was ≥2/3), **zero falls** across both repeats.

## 7. Regression gates

**5 originally-capped DR-1 wild episodes** (`rerun_capped.py`, 5000-step
budget, same seeds/scenes as DR-1's own extended reruns):

| ep | color | shape | dist | pre-fix | post-fix |
|---|---|---|---|---|---|
| 5 | orange | cube | 7.88m | OK (already succeeded) | **OK**, fd=0.467m |
| 11 | yellow | cone | 6.84m | FAIL | **OK**, fd=0.466m |
| 13 | purple | cube | 8.61m | OK (already succeeded) | **OK**, fd=0.475m |
| 19 | red | cone | 7.64m | FAIL | **OK**, fd=0.460m |
| 27 | yellow | cone | 8.64m | FAIL | **OK**, fd=0.495m |

**5/5 succeed** — both originally-passing episodes still pass; all 3
originally-failing episodes now pass too. Zero falls.

**Gated eval protocols** (pure defaults, seed 999, n=15, `checkpoint/goto_best.pt`,
`--no-render`/`--no-video` for speed only — confirmed not to affect pass/fail,
per DR-1's own headless-vs-video equivalence finding):

| gate | command | result | expected baseline |
|---|---|---|---|
| demo | `eval_closedloop.py --difficulty demo --seed 999 --n 15` | **14/15** (1 pre-existing, unrelated fall: ep4 "purple ball") | hold 14/15 |
| easy | `eval_closedloop.py --difficulty easy --seed 999 --n 15` | **15/15** | 15/15 |
| search | `eval_search.py --seed 999 --n 15` | **15/15** | 15/15 |

All three hold exactly at their documented baselines. This is expected by
construction, not just observed: `grep -n "fancy_demo" code/eval_closedloop.py
code/eval_search.py` returns nothing — neither gated script imports
`code/fancy_demo.py`, so this change cannot affect them.

## 8. Verdict

**ADOPT.** Zero regressions across all required gates; 3/3 mechanism flips;
5/5 wild-episode reruns; zero falls anywhere post-fix. Synced
`code/fancy_demo.py` byte-identical (plain file copy, no git) to
`VLA_mujoco_unitree/code/fancy_demo.py`.

**Not closed as policy-side/scoring-side** — although the underlying detector
recall gap for cones at 1.6-1.8m range is itself a perception-model
limitation out of scope for a deploy-side fix (and is not touched here), the
*crash-shaped* symptom DR-1 found (permanent stall/orbit after losing lock)
was a genuine `fancy_demo.py`-local recovery gap, fixed without touching the
detector, the trained policy, `GROUND_NET_TAU`, camera geometry, or the
success-radius scoring protocol.

**Residual, explicitly out of scope:** the underlying cone-at-1.6-1.8m
detector blind band still exists (this fix only ensures the robot *recovers*
from losing lock, via a bounded rescan, rather than fixing *why* the lock was
lost). If a future pass wants to close that gap directly, it would mean
touching the detector/its confidence threshold or the camera pitch/geometry
— out of this task's "minimal, deploy-side" mandate and riskier (those
constants are already calibrated via a JUDGE-gated precision/recall
trade-off, docs/nx14_detector_v2.md).
