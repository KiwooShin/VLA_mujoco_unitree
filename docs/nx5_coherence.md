# NX-5 — Odometry-Coherence Watchdog (M7): REJECT

**Date:** 2026-07-09
**Agent:** NX-5 (final bounded attempt on the demo/classical grounding line,
follow-on to `docs/nx2_final.md`, `docs/nx3_size_gate.md`, `docs/nx4_depth_split.md`)
**Checkpoint:** `checkpoint/goto_best.pt` (unchanged — deploy-side only)
**CAMERA_MODE:** not set (cam2 champion). Baseline: demo/classical 10/15 = 66.7%
(`eval/nx2_defaults_confirm`, failing eps 0, 2, 4, 5, 12; shipped defaults
LOCK_M1=1, LOCK_M3=1).

## TL;DR — VERDICT: REJECT (LOCK_M7 stays default OFF, never gated)

- **Mechanism implemented and unit-tested (30/30 CPU tests pass):** a
  sliding-DISTANCE-window odometry-coherence watchdog in `code/lock_mgmt.py`
  (`LOCK_M7`, default OFF), wired into both call sites
  (`code/inferencer.py`, `code/eval_search.py`). While CONFIRMED-locked and
  walking, it projects the robot's own measured world-frame displacement
  (from `data_mj.qpos[0:2]`, the robot's own pose — not the target's, exactly
  what a real state-estimator provides) onto the current goal bearing and
  accumulates it. Once the accumulation reaches `M7_X_WALK_M=1.75m`, the
  tracked goal distance must have shrunk by >= `M7_K_MIN_FRAC=0.4 * 1.75 =
  0.70m` over that window, or the lock is dropped and a bounded
  `ReacquisitionScan` (NX-1) re-enters scan, with a short-term (not
  hard-block) 2-of-2-corroboration re-lock penalty (±10°/±0.75m, 50 cycles)
  on the dropped lock's own bearing/distance — designed specifically to avoid
  the two documented fragilities of the REJECTed M4 (`docs/nx2_final.md`):
  a fixed-CYCLE window that trips on legitimate mid-episode transients, and
  an unconditional instant-reconfirm after `force_drop()`.
- **Mechanism-level replay (LOCK_M1=1 LOCK_M3=1 LOCK_M7=1, seed 999, single-episode
  standalone rollouts, §2) falsifies the brief's own working hypothesis before
  any full gate was run.** The brief asked to verify, before gating, that
  "ep13's transient heading corrections are distance-neutral (they don't
  accumulate 1.5m of toward-goal displacement while distance stalls)." They
  are **not** distance-neutral: ep13 reliably (3/3 replays) accumulates
  >=1.75m of projected displacement while its TRACKED goal distance *grows*
  by 2.5-2.8m (not just fails to shrink) — because, exactly as
  `docs/nx3_size_gate.md` §4 already documented, ep13's own **legitimate**
  baseline pass is carried by a bearing-correct, distance-WRONG wall-blob
  lock (tracked ~8.5-8.7m vs GT ~3.0m) for the bulk of the episode. This is
  the same structural signature M7 is designed to catch as a failure.
- **Quantitatively, the incoherence signal is *anti-correlated* with the
  desired classification** (§2.3): using each trigger's own
  `(required_shrink − actual_shrink)` margin as a severity score, ep13 (the
  passer that must stay silent) scores **3.24–3.55** on its first trigger,
  ep5 (a target episode) scores **1.74–3.35**, and ep0 (the primary target
  episode) scores a razor-thin **0.008** when it fires at all (and in 2/3
  replays it does not fire within the 1400-step episode cap at all). No single threshold
  revision to `M7_X_WALK_M`/`M7_K_MIN_FRAC` can reorder this — any bar loose
  enough to stay silent on ep13 is even more silent on ep0; any bar tight
  enough to reliably catch ep0 fires even more reliably (and earlier) on
  ep13. This is the same "overlap theorem" `docs/nx3_size_gate.md` §4 and
  `docs/nx4_depth_split.md` §5 already proved for physical-size and
  depth-split discrimination, now demonstrated numerically for odometric
  coherence: bearing-correctness is invisible to a signal built only from
  commanded/measured displacement and tracked distance, and this codebase's
  passing episodes structurally *depend* on exactly the same
  bearing-correct/distance-wrong detections that its failing episodes rely
  on.
- **Even where the mechanism fires "correctly" on ep0/ep5, it does not help:**
  reacquisition failed after every one of the 7 individual trigger events
  across all replays (0/7 `REACQUIRED`, §2.2) — the bounded rescan never
  re-locked the TRUE target
  within the post-trigger window in any case, most plausibly because the
  same dominant false-but-large detection population (the wall stripe /
  sliver blob) re-corroborates itself within the penalty's 2-of-2 window
  and re-seeds the same wrong lock, exactly the "drop → rescan → relock the
  same wrong blob" failure `docs/nx2_iso.md`'s M4 section already documented
  and root-caused.
- **ep13 (the flagged passer) broke in 2 of 3 mechanism-level replays**
  (FAIL fd=8.23m and fd=8.05m vs baseline SUCCESS fd=0.36-0.37m) and
  recovered by luck in the third (SUCCESS fd=0.36m despite firing) — i.e.
  LOCK_M7 turns a previously "rock-stable" passer (`docs/nx3_size_gate.md`:
  "ep13 is rock-stable... across all three M6-off runs") into a **majority
  break**, not a rare flip. The third replay's second trigger (step 920,
  after the first trigger's rescan already failed to reacquire) shows the
  robot's bearing to the true target has drifted to **-129.9°** — a
  degenerative "drop → rescan → relock something wrong → drift further"
  spiral, the same failure class `docs/nx2_iso.md` §M4 root-caused for the
  REJECTed M4 mechanism.
- **No full n=15 gate was run.** Per the brief's own protocol ("VALIDATE
  MECHANISM-LEVEL FIRST... If the watchdog cannot re-acquire the true target
  after firing... say so and stop — do not tune more than ONE constant set
  revision"): the mechanism-level replay already shows (a) reacquisition
  fails in 100% of firings, (b) the trigger signal is anti-correlated with
  the desired classification by a wide, unambiguous, quantitative margin, and
  (c) a previously rock-stable passer is destabilized. §2.3's quantitative
  analysis shows this is not a threshold-tuning gap — it is the same
  structural non-separability this codebase has now independently rediscovered
  three times (M4's divergence watchdog, M6's physical-size gate, GROUND_SPLIT's
  depth-split re-selection, and now M7's odometric coherence). Per this
  codebase's own established precedent for declining a tune once the
  overlap is structurally demonstrated (`docs/nx2_iso.md` §M4, `docs/nx3_size_gate.md`
  §4, `docs/nx4_depth_split.md` §5), the one permitted constant-set revision
  was not spent — no revision in the brief's own `X_WALK∈[1.5,2.0]`/`K_MIN=0.4·X_WALK`
  family can reorder incoherence margins that already run ep13 > ep5 >> ep0.
- **Code kept, default OFF, not synced.** `LOCK_M7` stays an opt-in env var
  in `code/lock_mgmt.py` (never flipped default-on); structural inertness
  verified (toggle-off smoke tests, `python code/inferencer.py` and
  `eval_search.py --smoke`, both byte-identical to pre-NX-5 behavior). Per
  the brief's ADOPT/REJECT condition (sync only on ADOPT), nothing was
  byte-copied to `VLA_mujoco_unitree/code/`.
- **§CLOSURE (below): the complete NX-2→NX-5 evidence chain that demo/classical
  66.7% is this classical grounder's ceiling**, and what would actually need
  to change.

---

## 1. Design (what was built)

All additive, in `code/lock_mgmt.py` (extending NX-2's `LockGate`/`ReacquisitionScan`
plumbing), plus two small call-site edits in `code/inferencer.py` and
`code/eval_search.py`:

- **`LOCK_M7`** env toggle, default OFF (opt-in), independent of `LOCK_M1..M5`.
- **Accumulation** (`LockGate.end_of_cycle`, extended with an optional
  `proj_disp_m` argument): while `state=='CONFIRMED'`, `walking` (the same
  flag the callers already compute for M4: `not scan_active and dist > stop_r`),
  no discontinuity cooldown active, and the tracked goal distance >=
  `M7_MIN_GOAL_DIST_M=1.5m` (endgame/proximity-cam carve-out), accumulates
  `proj_disp_m` into a running sum. `proj_disp_m` is computed at each call
  site from `data_mj.qpos[0:2]` (the robot's own world-frame position —
  privileged in the sim harness, but analogous to real leg-odometry/state-estimator
  output, never the target's position) diffed against the previous grounding
  cycle's position, rotated into the robot's body frame by the current yaw,
  and dotted with the current goal bearing `(cos_th, sin_th)` from
  `cached_goal_vec`.
- **Sliding-distance-window check:** once the accumulation reaches
  `M7_X_WALK_M` (task brief's 1.5-2.0m band; 1.75m midpoint chosen), the
  tracked goal distance must have shrunk by >= `M7_K_MIN_FRAC * M7_X_WALK_M`
  (0.4 * 1.75 = 0.70m) since the window's reference distance. If it has, the
  window SLIDES forward (reference distance and accumulator both reset to the
  current cycle) rather than freezing a one-shot check — deliberately not a
  fixed-cycle count, per the brief's mandate that this be the axis that fixes
  M4's documented cycle-count fragility (`docs/nx2_final.md`, `docs/nx2_iso.md`
  §M4).
- **Carve-outs:** `mark_discontinuity()` (CAM-2 handoff / `_active_cam`
  Schmitt flip) resets the window outright; the endgame floor
  (`M7_MIN_GOAL_DIST_M`) suspends accumulation without resetting; `walking=False`
  (scan/rescan) suspends both accumulation and the check by construction
  (same flag M4 already uses).
- **On trigger:** `force_drop()` + `ReacquisitionScan` (NX-1's bounded
  `BidirectionalScanSchedule`, the same reacquisition path M4/M5 already use)
  — never an unbounded spin. The dropped lock's own `(bearing, dist)` is
  snapshotted into a short-term (NOT hard-block) re-lock penalty: a fresh
  detection within `M7_PENALTY_BEARING_DEG=±10°` / `M7_PENALTY_DIST_TOL_M=0.75m`
  of the dropped lock needs `M7_PENALTY_CONFIRM_M=2`-of-`N=2` mutually
  consistent cycles (instead of the usual instant single-frame confirm) for
  `M7_PENALTY_CYCLES=50` cycles; detections outside that zone, or after
  expiry, behave exactly as before the drop.
- **30/30 CPU unit tests pass** (scratchpad `test_nx5_m7_unit.py`): static
  wrong-depth lock fires (~cycle 17, matches `X_WALK/disp_per_cycle`); a
  genuinely closing approach stays silent; all three carve-outs (scan,
  discontinuity cooldown, endgame) suspend correctly; the window slides
  forward repeatedly over a long coherent walk without a false trigger; the
  re-lock penalty requires 2-of-2 corroboration in-zone, confirms instantly
  out-of-zone, and expires after `M7_PENALTY_CYCLES`; toggle-off inertness
  (`LOCK_M7=0`) is byte-identical to pre-NX-5 behavior; `LOCK_M1`/`LOCK_M3`
  defaults are untouched.
- `python code/inferencer.py` smoke (arch A+C, random-init, 30 steps) and
  `python code/eval_search.py --smoke` both pass identically with
  `LOCK_M7=0` (unset) and `LOCK_M7=1` — no crash, and the `LOCK_M7=0` output
  is byte-identical to the pre-NX-5 baseline smoke output (structural
  inertness: the only unconditionally-executed addition per grounding cycle
  is one `qpos` copy + a handful of flops to feed `proj_disp_m`, which
  `end_of_cycle` ignores whenever `LOCK_M7` is off, matching every other
  mechanism's "always call, no-op internally" contract).

## 2. Mechanism-level replay (before any full gate, per brief)

**Setup:** standalone single-episode rollouts (not the multi-episode
`eval_closedloop.py` loop) via `code/inferencer.py`'s `Inferencer.rollout()`,
seed 999, demo difficulty, `checkpoint/goto_best.pt`, `goal_source=classical`,
`vel_source=predicted`, `device=cuda`, `LOCK_M1=1 LOCK_M3=1 LOCK_M7=1`
(shipped defaults + M7 under test). Instrumentation: pure Python-level
monkeypatching of `code.inferencer.classical_ground` and
`code.lock_mgmt.LockGate.end_of_cycle` using caller-frame introspection to
read `data_mj`/`target_xy` (same non-invasive pattern as FA-1's
`diag_ep0_raw.py` / NX-3's `calibrate_m6.py` — no repo files touched by the
instrumentation itself). Scratchpad `nx5_mech_check.py`. Episodes replayed
2-3x each (GPU/EGL run-to-run physics non-determinism is a documented
property of this harness, `docs/cam_p0.md`), matching this codebase's
rerun-on-flip protocol.

### 2.1 Fire/silent table

| ep | expected | replay 1 | replay 2 | replay 3 | verdict |
|---|---|---|---|---|---|
| 0 (cyan cone, target) | FIRE | 0 triggers (silent — 1400-step cap reached) | 1 trigger @step=1230 (FAIL fd=3.36, unchanged) | 0 triggers (FAIL fd=3.37, unchanged) | **fires in only 1/3 replays, and too late to matter when it does** |
| 5 (cyan ball, target) | FIRE | 1 trigger @step=280 (FAIL fd=3.28) | -- | 1 trigger @step=310 (FAIL fd=3.98) | fires reliably (2/2), but on a DIFFERENT signature than FA-1's documented late stall (§2.4) |
| 1 (cyan cube, passer, bistable) | SILENT | 0 triggers (SUCCESS fd=0.37) | -- | 0 triggers (SUCCESS fd=0.36) | silent, as required |
| 3 (red cube, passer) | SILENT | 0 triggers (SUCCESS fd=0.37) | -- | 0 triggers (SUCCESS fd=0.37) | silent, as required |
| 13 (blue ball, passer, transient-heavy) | SILENT | 1 trigger @step=290 (**FAIL fd=8.23**, baseline SUCCESS fd=0.36) | 1 trigger @step=310 (SUCCESS fd=0.36, recovered despite firing) | 2 triggers @step=290,920 (**FAIL fd=8.05**) | **fires reliably (3/3), breaks the episode in 2/3 replays** |

### 2.2 Reacquisition after firing

Across the 7 individual trigger events observed (ep5 x2, ep13 x4 across 3
replays, ep0 x1), 0 show the post-trigger bounded rescan re-locking onto
detections consistent with the TRUE target (`|reported_dist - GT_dist| <
1.0m` and `|bearing - GT_bearing| < 15°` for >= 3 consecutive cycles within
300 steps of the trigger): every single firing scores `REACQUIRED=False`.
ep13's one
lucky SUCCESS (replay 2) recovered *despite* this — plausibly the episode's
own baseline trajectory (which does eventually pick up the true ball
mid-approach per `docs/nx3_size_gate.md` §4) reasserted itself after the
rescan interlude wasted some steps, not because M7's rescan targeted the
true ball. Replay 3's SECOND trigger (step 920, gt_bearing=-129.9°) shows
what typically happens instead: the first failed reacquisition leaves the
robot even further from the true target than before the intervention, and a
second trigger fires on the resulting (still-incoherent) track — a
degenerative spiral, not a self-correcting one.

### 2.3 Why tuning cannot fix this — the incoherence-margin ordering

For every firing, define `incoherence_margin = M7_K_MIN_FRAC*M7_X_WALK_M -
(window_dist0 - cached_goal_dist_at_trigger)` — i.e. how far past the trigger
bar the episode actually was (0 = exactly at the bar; larger = the trigger
condition was more emphatically true):

| ep | replay | window_dist0 | cached_goal_dist @ trigger | actual shrink | incoherence margin |
|---|---|---|---|---|---|
| 0 | 2 | 6.177 | 5.486 | +0.692 | **+0.008** (razor-thin) |
| 5 | 1 | 6.120 | 7.157 | -1.037 | +1.737 |
| 5 | 3 | 6.120 | 8.769 | -2.649 | +3.349 |
| 13 | 1 | 6.001 | 8.540 | -2.539 | +3.239 |
| 13 | 2 | 6.001 | 8.708 | -2.707 | +3.407 |
| 13 | 3 (1st trigger) | 6.001 | 8.846 | -2.845 | +3.545 |
| 13 | 3 (2nd trigger, post-spiral) | 8.108 | 7.510 | +0.599 | +0.101 |

**ep13 — the episode that must stay silent — has the LARGEST incoherence
margin of any first-trigger episode replayed (+3.2 to +3.5), 2 orders of
magnitude larger than ep0's (+0.008) and consistently larger than ep5's
(+1.7 to +3.3, itself already an order of magnitude above ep0's).** Any single revision to `M7_X_WALK_M` (brief's
allowed 1.5-2.0m band) or a stricter/looser `M7_K_MIN_FRAC` shifts every
episode's margin by the same additive constant — it cannot reorder them.
A bar loose enough to stop firing on ep13 (margin ~3.2-3.4) would need to
raise the required shrink far below what ep0 achieves (margin ~0.01,
i.e. already borderline-passing); a bar tight enough to make ep0 fire
reliably and early enough to leave recovery time would fire on ep13 even
more certainly and even earlier. This is why the ONE constant-set revision
the brief permits was not spent: it is pre-analyzable to fail, the same
reasoning precedent `docs/nx2_iso.md` §M4, `docs/nx3_size_gate.md` §4, and
`docs/nx4_depth_split.md` §5 already used to decline a tune once a structural
overlap is demonstrated rather than guessed.

**Root cause (read from the replay data, not guessed):** ep13's own baseline
SUCCESS is carried, for a large fraction of its approach, by a bearing-correct,
distance-WRONG wall-blob lock (tracked ~8.5-8.7m vs GT~3.0m — matching
`docs/nx3_size_gate.md` §4's independently-derived figures of "wall blobs at
d≈9.6m, GT≈3.3m... bearing tracks the true target" almost exactly). From
M7's point of view (commanded/measured displacement + tracked distance,
with no access to GT) this is *indistinguishable* from ep0/ep5's failure
signature (a static or growing wrong-depth lock) — both present as "walked
>= X_WALK toward the bearing, tracked distance didn't shrink". The brief's
own working hypothesis — that ep13's risk is *cycle-count* fragility from
transient heading corrections (M4's specific documented failure) — is
falsified by this replay: ep13's incoherence is not a brief transient, it is
the episode's dominant, sustained tracking behavior for a large window of
its baseline-passing trajectory, and a *distance*-based window (immune to
M4's cycle-count problem by construction) still cannot separate it from
ep0/ep5, because the discriminating variable (bearing-correctness of the
locked-onto object) is invisible to any signal built purely from odometry +
tracked distance.

### 2.4 ep0/ep5: even the "intended" firings don't help

Even setting the ep13 problem aside, ep0's firing (when it happens at all —
2/3 replays showed no trigger within the 1400-step cap) occurs at step 1230
of 1400: with ~170 steps (8.5 grounding cycles) left, there is no practical
time for the bounded rescan to matter, and indeed `final_dist` is unchanged
(3.36m vs baseline 3.35m). ep5's firing is reliable (2/2 replays, both
~step 280-310) but does NOT match FA-1's documented failure signature (a
stall at 3.4-3.6m emerging by step ~800-900); instead it catches an earlier,
different phenomenon (the tracked distance growing from ~6.1m to ~7.2-8.8m
in the first ~300 steps — plausibly a legitimate early bearing-alignment
transient, structurally the same class of thing M4 mis-fired on in ep13).
Reacquisition fails in both ep5 replays regardless (§2.2), so even granting
that this early trigger correctly flags *something* wrong, the watchdog
provides no recovery path.

## 3. Gates — not run, and why not

Per the brief's own protocol, the demo/classical closed-loop gate (seed 999,
n=15, ACCEPT bar >=11/15 with eps 0/5 flipped and no reproducible passer
break) is conditioned on the mechanism-level replay "looking right" first.
It does not: reacquisition fails in 100% of observed firings (§2.2), the
firing signal is quantitatively anti-correlated with the desired
classification by a wide, structurally-explained margin (§2.3), and the
flagged passer (ep13) is destabilized into a nondeterministic outcome in
exactly the replay the brief asked to check before gating (§2.1, §2.3). No
full n=15 demo/easy/search gate was run — burning GPU time on a
REJECT-verdicted configuration would not change the outcome, matching the
early-stop precedent already used in `docs/nx3_size_gate.md` §3
("Cross-skill gates were not run — the demo gate already fails both KEEP
bars") and `docs/nx4_depth_split.md` §"Cross-skill gates not run".

## 4. Files changed / kept / NOT synced

- `code/lock_mgmt.py` — additive: `LOCK_M7` toggle (default OFF),
  `M7_X_WALK_M`/`M7_K_MIN_FRAC`/`M7_MIN_GOAL_DIST_M`/`M7_PENALTY_*` constants,
  `LockGate.__init__`'s new M7 state fields, `_confirm()`'s window reset,
  `_m7_in_penalty_zone()`, `gate_detection()`'s penalty branch, `end_of_cycle()`
  extended with an optional `proj_disp_m` argument and the M7 accumulation/check
  block (+ `self.last_trigger` diagnostic), `mark_discontinuity()`'s and
  `force_drop()`'s M7 state resets. All gated behind `if LOCK_M7:` /
  provably inert with it unset (30/30 unit tests include explicit
  toggle-off-inertness checks).
- `code/inferencer.py` — additive: `_m7_prev_xy` odometry tracker + the
  `proj_disp_m` computation at the existing M4 `end_of_cycle()` call site
  (now also passing `proj_disp_m`); verbose log line distinguishes
  `M4 divergence` vs `M7 coherence`. No other logic changed.
- `code/eval_search.py` — same additive pattern at its own `end_of_cycle()`
  call site (search has no CAM-2 handoff, so no `mark_discontinuity()` call
  site changes needed there, matching NX-2's existing note for that file).
- No changes to `code/grounding.py`, `code/scene.py`, `code/arena.py`,
  `code/steer.py`, `code/scan_sched.py`.
- **NOT synced** to `VLA_mujoco_unitree/code/` — sync is
  conditioned on ADOPT; verdict is REJECT. `LOCK_M7` remains default OFF in
  the source-of-truth repo; the existing `LOCK_M1=1 LOCK_M3=1` shipped
  defaults are untouched (confirmed via the toggle-off smoke tests in §1).
- Diagnostic scripts live in scratchpad only (never committed):
  `test_nx5_m7_unit.py` (30 unit tests), `nx5_mech_check.py` (standalone
  single-episode replay harness with caller-frame-introspection
  instrumentation), `nx5_mech_check_out.json` (replay 1: eps 0,5,1,3,13),
  `nx5_debug_full.json` (replay 2, full per-cycle logs: eps 0,13),
  `nx5_mech_check_rerun.json` (replay 3, full per-cycle logs: eps 0,5,1,3,13)
  — raw replay data behind §2's tables.

## §CLOSURE — the NX-2→NX-5 evidence chain: demo/classical 66.7% is this
classical grounder's ceiling

Four independent agents, four independent axes, four structurally-explained
REJECTs, converging on the same underlying fact:

1. **NX-2 (`docs/nx2_final.md`):** lock-management hygiene (area floor +
   innovation gate, M1/M3, ADOPTED as safe no-regression defaults) cannot
   reach ep0/2/4/5/12 — M1's floor sits mathematically below the false
   locks' steady-state area without also cutting legitimate far detections;
   M3's gate is unconditionally bypassed at ep12's hijack moment by the
   mandatory CAM-2 discontinuity carve-out. The higher-impact mechanisms
   (M2 N-of-M, M4 divergence watchdog, M5 bounded coast) were REJECTed:
   M4 in particular broke ep13 by instantly re-confirming the same false
   blob after a forced drop, and separately by mis-firing on a legitimate
   mid-episode transient under its fixed-cycle window.
2. **NX-3 (`docs/nx3_size_gate.md`):** physical-size plausibility (M6) finds
   clean STATIC separation between true and false detections for ep0/2/5 —
   but the closed-loop gate regresses (8/15) because ep1's and ep13's own
   PASSING trajectories are carried by merged/wall blobs that are exactly as
   size-implausible as ep0/2/5's false locks. The discriminating variable is
   whether the blob's bearing points at the target — invisible to size.
3. **NX-4 (`docs/nx4_depth_split.md`):** depth-guided blob splitting, built
   on NX-3's own suggested next step, is falsified even before gating —
   direct depth-histogram measurement shows the "merged" blobs are a
   continuous oblique-wall depth ramp, not a bimodal composite; there is no
   gap to split on. The one place splitting reaches ep13 (FG-rescue
   re-selection) breaks it the same way M6 did — size-plausibility again
   prefers the bearing-wrong fragment.
4. **NX-5 (this document):** odometric coherence, built specifically to be
   immune to M4's fixed-cycle fragility and to add a not-hard-block re-lock
   penalty M4 lacked, still cannot separate ep0/5's failures from ep13's
   pass — because ep13's pass and ep0/5's failures are BOTH carried by
   bearing-correct/distance-wrong detections, and no signal built from
   commanded/measured odometry plus tracked distance (without access to
   ground truth) can tell them apart. Quantitatively, ep13's incoherence
   margin is the LARGEST of any episode tested — the anti-correlation is
   not subtle.

**The common thread across all four:** this arena's classical HSV+depth
grounder produces detections whose *bearing* can be correct while their
*depth/distance* is wrong (grazing-incidence wall stripes, same-hue
distractors, merged blobs) far more often than it produces detections that
are simply absent. Every geometric/temporal/odometric axis tried so far
(static size, depth-splitting, cycle-count divergence, distance-coherence)
is a function of (size, depth, distance) — exactly the axis on which the
true and false populations overlap. None of them can see bearing-correctness
directly, and this codebase's own passing episodes structurally *need*
bearing-correct/distance-wrong detections to succeed (ep1, ep13, and parts
of ep12's approach), so any filter that successfully excludes ep0/2/5's
false locks on a (size, depth, distance)-only basis necessarily also
excludes some of what currently makes ep1/ep13/ep12 pass.

**What a learned detector would change:** a learned grounding head (trained
on paired (image, true-target-bbox) supervision, as opposed to classical
HSV+depth heuristics with post-hoc lock-management gates) could in principle
learn the *appearance* features that separate "this stripe is the wall" from
"this stripe is the target" — texture, edge continuity, object-shape priors
— which are available in the raw pixels but structurally inaccessible to
any of the four (size/depth/cycle/odometry) discriminators tried across
NX-2 through NX-5, all of which operate downstream of the classical
detector's already-collapsed (dist, bearing, area) summary. This reframes
the ceiling correctly: it is not that lock-management or detection-quality
gating is the wrong genre of fix — it is that every fix in that genre this
codebase can build on top of the classical HSV+depth pipeline's OUTPUT is
provably blind to the one variable (bearing-correctness of the underlying
pixels) that actually separates the two populations. This project's own
prior finding that shared-model retraining (rotation-DART, proprio-vel) has
a consistent negative track record for skill-targeted fixes is a different
axis (locomotion/control, not perception) and is not contradicted by this
closure; the recommendation here is narrower and specific to grounding:
future work on the demo/classical failures should target the classical
detector's front end (a trained appearance-based detector/segmenter) rather
than any further downstream lock-management heuristic, which this four-agent
chain has now exhausted.
