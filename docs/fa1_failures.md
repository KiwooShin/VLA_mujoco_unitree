# FA-1 — Failure Analysis of the Residual Demo/Search Failures

**Date:** 2026-07-09
**Agent:** FA-1 (failure analysis to plan the next experiment cycle)
**Champion under analysis:** CAM-2 + probe-gate fix (`docs/cam_p1.md` + `docs/cam_p4_gate.md`),
checkpoint `checkpoint/goto_best.pt` (frozen). Gated numbers: easy/classical 100%,
demo/classical 66.7% (10/15), search 80.0% (12/15, `eval/p4_gate_search_rerun`),
maneuver ~73%.

**Method:** read existing artifacts (`eval/p4_gate_demo`, `eval/p4_gate_search`,
`eval/p4_gate_search_rerun`, `eval/DIAG_deployed_demo_gt`, `eval/p1_demo_cam2_v2`,
docs `cam_p0/p1/p2/p3_demo/p4_gate`, `grounding_dist.md`, `rot_dart.md`)
+ 6 light single-episode GPU reruns of the 5 demo failures (one with an
instrumentation wrapper around `ground()` logging the raw per-cycle detection),
comparing against the existing GT-goal artifact for the same checkpoint/seed. No
sweeps, no retraining, no full n=15 re-evals were run (GPU shared with CX-4).

**Headline correction to prior docs:** `docs/cam_p1.md`/`cam_p4_gate.md` describe the
5 demo failures as uniformly "the documented cyan/blue wall-HSV collisions." This is
**only accurate for 1 of the 5** (ep2). ep4's target is **purple**, whose HSV band
(H:125-160) does not overlap the wall's H≈104-105 — it is a genuine detection
total-miss + a locomotion/geometry problem, unrelated to the wall-hue mechanism.
ep0, ep5, ep12 have **working initial detection** (aligned within 0-40 steps) but fail
downstream via two previously-undocumented mechanisms: a marginal/flickering blob
locked at the wrong depth (ep0, likely ep5), and a mid-approach lock hijack to a
same-color distractor (ep12). See §1 for evidence.

---

## 1. Demo/classical — per-episode failure table (seed 999, n=15)

| ep | target (dist) | classical outcome | GT-goal outcome | mechanism (see detail below) | class |
|----|---|---|---|---|---|
| 0 | cyan cone (4.32m) | FAIL fd=3.40, 1400 steps | FAIL fd=0.68 (close miss) | marginal/flickering blob locked at wrong depth | grounding-bound (locomotion near-sufficient) |
| 2 | blue cone (4.86m) | FAIL fd=10.66 (walks away) | **SUCCESS** fd=0.38 | confident false-positive lock from step 40, never corrects | grounding-bound |
| 4 | purple ball (7.21m) | FAIL fd=10.24 (walks away) | FAIL fd=2.01 (worst GT miss) | total detection miss (never seen in 200-step scan) + path likely obstructed by near distractors | compound: grounding-miss + locomotion/geometry-bound |
| 5 | cyan ball (8.85m) | FAIL fd=3.52 (stalls) | **SUCCESS** fd=0.36 | converges well, then stalls ~3.4-3.6m (same signature as ep0; not independently re-instrumented) | grounding-bound (inferred) |
| 12 | cyan cube (6.18m) | FAIL fd=6.67 (round trip) | **SUCCESS** fd=0.37 | converges to 3.97m by step ~600, then reverses back to 6.66m — mid-approach distractor hijack | grounding-bound |

**GT-vs-classical split: 4/5 grounding-bound (ep0, ep2, ep5, ep12), 1/5 compound
locomotion/geometry-bound (ep4).** (GT source: `eval/DIAG_deployed_demo_gt`, same
checkpoint, same seed — fails only ep0/ep1/ep4, succeeds ep2/ep5/ep12; NB ep1
itself flips the other way, GT-fails but classical-succeeds — flagged as a caveat on
trusting GT as a strict upper bound at long range with predicted-vel, not otherwise
relevant here.)

### Per-episode instrumented detail

**ep0 (cyan cone, 4.32m) — marginal/flickering blob at wrong depth.** Re-run with a
wrapper around `ground()` logging every raw call: scan aligns at **step 0**
(`yaw_err=+29.7°` — detection works immediately, contradicts a "never detects"
story). From step ~100 onward, `ground()` alternates `not_visible` roughly 40/60
(visible on ~60% of grounding cycles) and, whenever visible, reports a **suspiciously
stable dist≈5.4-5.5m, bearing +5° to +13°, confidence pinned at 0.40-0.41** — every
single call, for 90 consecutive grounding cycles (900 steps). Meanwhile the
true GT distance to target sits flat at **3.3-3.5m** the entire time: a persistent
~2m depth bias, not sensor noise. Decoding `confidence = 0.6*conf_area + 0.4*conf_depth`
(`code/grounding.py:797`): confidence≈0.40 with `conf_depth=1.0` implies
`conf_area≈0` — i.e. the accepted blob is barely above `MIN_BLOB_AREA` (40px²), a
sliver, not the target's full silhouette. Combined with `HOLD_GOAL_HORIZON=100` and
EMA smoothing (`_GOAL_EMA_ALPHA=0.4`), this small-but-passing-threshold blob anchors
a stable, wrong goal that the policy dutifully walks toward, producing a stable
**wrong equilibrium at ~3.35m** rather than a locomotion failure. `MIN_CONFIDENCE=0.05`
(code/grounding.py:160) is far below the observed 0.40 — the current threshold does
not reject this class of detection at all.

**ep2 (blue cone, 4.86m) — confident false-positive, no self-correction.** Scan
aligns at step 40 (`yaw_err=+20.0°`). From the very first accepted detection, GT
distance-to-true-target **increases monotonically** every single 50-step checkpoint
for the entire 1400-step episode (4.80 → 9.20+ → final 10.66m) — the robot walks
confidently in a fixed wrong direction and never re-corrects. Scene has a cyan
cylinder distractor at only 0.71m (bearing 134.6°, per `code/scene.py` sampling) —
cyan (H:85-108) and blue (H:100-135) HSV bands overlap at H:100-108, the same band
the arena wall renders in (H≈104-105, `docs/grounding_dist.md`) — any of the wall,
this close cyan distractor, or an edge artifact is a plausible false-positive source
for "blue." GT succeeds cleanly (steps=760, fd=0.38) confirming locomotion is fine
once given a correct goal.

**ep4 (purple ball, 7.21m) — genuine total miss, not a wall-hue case.** Purple's HSV
band (`HSV_BOUNDS["purple"]`, H:125-160) does **not** overlap the wall's H≈104-105 or
either of blue (100-135) / cyan (85-108) at their nominal hues (raw purple H≈139,
raw blue H≈113 — 26° apart) — this episode is **not** the documented wall-collision
mechanism despite being lumped in with it by `docs/cam_p4_gate.md`. Re-run shows
`[scan] TIMEOUT at step=200` with **zero** `ALIGNED`/partial-detection messages the
entire scan — the purple ball is never seen at all, at any bearing, during the full
200-step scan sweep. After timeout the robot walks the default/fallback heading and
drifts monotonically from 7.09m to 10m+. Distinct from the other 4: this is the
**only** demo failure where the **GT-goal run also fails badly** (fd=2.01m, the
worst of the 3 GT misses) — even with a perfect goal signal every step, the robot
does not converge. Scene has 3 distractors clustered close to the direct path (red
cone 1.08m/bearing 116°, blue cylinder 3.09m/bearing 18.6°, orange cube
3.29m/bearing -57.1°) — the leading hypothesis is physical path obstruction/deflection
by the blue cylinder sitting roughly in the travel corridor, but this needs a
GT-goal instrumented rerun (not done here, to stay light on GPU) to confirm.

**ep5 (cyan ball, 8.85m) — same stall signature as ep0 (inferred, not
re-instrumented).** Scan aligns at step 0 (`yaw_err=-29.8°`) and dist closes smoothly
and monotonically from 8.85m down to ~3.4m by step ~800-900, then oscillates
3.37-3.58m for the remaining ~500 steps without further progress — visually
identical to ep0's plateau (before the raw-detection rerun revealed ep0's plateau
was actually a stable *wrong* lock, not a locomotion limit). GT succeeds cleanly
(fd=0.36), so the plateau is not a locomotion ceiling. Not independently
re-instrumented with the raw-`ground()` wrapper (would be a 6th single-episode
rerun); flagged as "same mechanism, high confidence by analogy, not directly
confirmed."

**ep12 (cyan cube, 6.18m) — mid-approach distractor hijack.** Scan aligns at step 0
(`yaw_err=+22.0°`) and dist closes cleanly from 6.18m to **3.97m by step ~600**
(the approach is initially completely correct) — then **reverses**, climbing back
up monotonically to 5.02 (step 1000), 5.76 (step 1150), 6.66m (final). Scene has a
cyan-ball distractor at 2.90m/bearing +21.3° — almost a mirror image of the true
target's bearing (-21.8°) — plausible read: once the robot's approach brings it
close enough that the distractor's blob becomes comparably large/confident, the
EMA-tracked lock flips from the (now also nearby, but still correct) target cube to
the closer distractor ball, sending the robot back the way it came. This is a
distinct failure mode from ep2 (which locks wrong from the very first detection) —
here the *correct* target is tracked correctly for the first ~40% of the episode
before being hijacked.

---

## 2. Search — the 3 falls, cross-referenced against scan duration

`eval/p4_gate_search_rerun/summary.json` (the confirmed 80.0%/12-15 run) + `run.log`
per-step traces. All 3 falls are tagged `fall`, occurring **shortly after the scan→goto
transition**, not during scan itself and not during a long-since-established walk:

| ep | target (dist) | init_bearing | scan_steps (spotted at) | fall step | steps survived post-spot | 
|----|---|---|---|---|---|
| 5 | red cylinder (2.21m) | 72.0° | **570** | 680 | ~110 |
| 7 | orange ball (2.41m) | 60.5° | **600** (scan-timeout boundary) | 662 | ~62 |
| 8 | orange cylinder (2.60m) | 82.8° | **550** | 774 | ~224 |

All other 12 episodes: scan_steps ∈ {70, 140, 150, 180, 280, 350, 370, 400, 400, 410,
450, 470} — **the 3 falls have exactly the 3 longest scan durations of all 15
episodes**, with a clean gap (next-longest non-falling scan is 470 steps, ep1,
succeeds without incident). Scan duration does **not** track `init_bearing_deg`
monotonically (e.g. ep4 at bearing 60.7° scans only 70 steps and succeeds; ep7 at a
nearly-identical 60.5° scans the full 600) — `eval_search.py`'s own scan loop
(`_run_search_rollout`, distinct from `inferencer.py`'s bounded right/left/right H3
demo-scan) rotates **fixed-direction CCW only** up to `SCAN_TIMEOUT=600` steps
(`code/eval_search.py:391-393`), so a target that happens to sit on the "wrong side"
of the one-way sweep requires nearly a full 360° rotation to reach, while a
similar-magnitude bearing on the favorable side is found almost immediately.

**Diagnosis matches existing institutional finding:** `docs/rot_dart.md` (C4,
2026-07-07) independently diagnosed these same 3 episodes ("ep05/ep07/ep08... all 3
falls occur when scan requires >500 steps — the model was trained on
walking-dominated data") and confirmed the mechanism by fine-tuning on
rotation-recovery DART data, which eliminated the falls (3→0) but **catastrophically
regressed demo (60%→20%) and easy (93%→80%)** because the shared model's approach
behavior gets corrupted by rotation-conditioned training data — REJECTED, not
adopted. This rules out a shared-model retrain as the right lever; a deploy-side-only
scan-schedule change (see §3) targets the same root cause without touching the model.

---

## 3. Ranked candidate experiments

### #1 (rank by score) — Cap continuous scan rotation in the search skill (deploy-side only)

**Mechanism:** redesign `eval_search.py`'s scan loop from fixed-direction CCW (up to
600 steps / ~360°+) to a bounded bidirectional sweep — e.g. right up to ~180°, and if
not found, return through center and sweep left up to ~180° — capping the worst-case
*continuous single-direction* rotation at roughly half its current value (~180°,
~260 steps vs up to 600). This directly targets the diagnosed root cause (§2:
perfect separation between falling (550-600 steps) and succeeding (≤470 steps) scan
durations in the current data) without retraining the shared policy, so it carries
none of `rot_dart.md`'s cross-skill regression risk (easy/demo/maneuver never touch
this code path).
**Expected gain:** could plausibly clear all 3 falls → search 80%→~93-100% (n=15).
**Probability of success:** HIGH — strong, clean correlational evidence (3/3 falls
are exactly the 3 longest scans; the H3 demo-scan already uses a similar
bounded-direction pattern successfully) + isolated blast radius.
**Effort:** LOW — deploy-side code change confined to `eval_search.py`'s scan-schedule
block (~20-30 lines); no retraining; verify via a full n=15 search re-eval.
**Risk / watch-outs:** must still guarantee full angular coverage (any bearing in
±180°) — a two-phase right-then-left design does this by construction; needs a check
that legitimate long-bearing targets currently found at 550-600 steps (i.e. genuinely
requiring most of a full rotation) don't flip to "never found" — though the data
suggests these ARE the falling cases already, so even a "spotted but exactly at the
new coverage boundary" outcome is a net improvement over a guaranteed fall.

### #2 — Grounding lock-stability/hysteresis for the demo skill (deploy-side only)

**Mechanism:** 4 of 5 demo failures (ep0, ep2, ep5 [inferred], ep12) are grounding
mis-locks, not locomotion or coverage problems — but they are three distinct
sub-mechanisms (marginal/flickering small-area blob accepted despite conf≈0.40 just
above `MIN_CONFIDENCE=0.05`; immediate false-positive lock with no self-correction;
mid-approach hijack to a closer same-color distractor). A combined fix: (a) require
2 consecutive corroborating detections (similar bearing+dist, within tolerance)
before accepting a *fresh* lock out of a cold/not-visible state — directly targets
ep0's flickering marginal blob; (b) add "lock stickiness" — an established,
larger/higher-conf_area lock should not be displaced by a new detection unless the
new one is comparably large/confident, not just merely above the floor — targets
ep12's mid-approach hijack.
**Expected gain:** plausibly recovers 2-3 of the 5 (ep0, ep5, ep12 share the
"marginal/competing blob" family) → demo 66.7%→~80-87%.
**Probability of success:** MEDIUM — mechanism is directly confirmed for ep0
(instrumented) and ep12 (behaviorally inferred), inferred by analogy for ep5; real
risk of regressing currently-passing episodes that rely on fast single-frame locks
(e.g. easy/classical's near-instant alignments) — this campaign's history
(`rot_dart.md`, `vel_proprio.md`) shows shared-logic changes often have
unexpected cross-episode costs, so a full 3-skill re-gate is mandatory, not optional.
**Effort:** LOW-MEDIUM — confined to `code/grounding.py`/`code/inferencer.py`'s
existing EMA/hold-goal logic (no retraining); needs the standard full re-gate
(easy+demo+search, n=15 each) per this codebase's established discipline.

### #3 — Goal-divergence watchdog to catch confident false-locks (deploy-side only)

**Mechanism:** ep2's specific signature (EMA'd/GT distance to the *true* target
increases monotonically for the entire episode after a confident early lock) is a
physically implausible pattern for a correctly-tracked approach. Add a lightweight
watchdog: if the tracked goal distance has been non-decreasing for N consecutive
grounding cycles (e.g. N≈15-20, ~150-200 steps) while the robot is actively walking
forward, treat the current lock as suspect and force a re-scan / reset
`_last_known_goal`. This is a narrower, more surgical fix than #2 and specifically
targets the "confidently wrong from the start" class.
**Expected gain:** recovers ep2 specifically (~6.7pp on demo: 66.7%→~73%); also a
general robustness net against similar false-locks in scenes not in this n=15 set.
**Probability of success:** MEDIUM — clear, clean mechanism from the ep2 trace, but
the reset threshold needs tuning to avoid false-triggering on legitimate transient
distance increases (e.g. early scan-exit misalignment, or a real detour around an
obstacle).
**Effort:** LOW — isolated addition to `inferencer.py`'s goal-update block; no
retraining.

### Explicitly deprioritized: locomotion fine-tune / more demo-yaw DART data (expensive, low-probability given this evidence)

The task brief's own example lever ("locomotion fine-tune with more demo-yaw DART
data") is **not** supported by this analysis as the next move: only 1 of 5 demo
failures (ep4) is locomotion/geometry-bound, and even that one is compounded by a
genuine grounding total-miss. Retraining the shared goto model has a clean, repeated
negative track record in this codebase for exactly this kind of targeted fix — C4's
rotation-DART fixed search falls but regressed demo 60%→20% and easy 93%→80%
(`docs/rot_dart.md`); V6's proprio-fed vel head similarly regressed goto
(`vel_proprio.md`). Any retraining lever should be held back until the three
deploy-side candidates above are exhausted, and should be scoped as a last resort
given its cost (data-gen + fine-tune + full 3-skill re-gate) and this codebase's
consistent pattern of shared-model changes trading one skill's gain for another's
loss.

---

## 4. Files / artifacts produced by this analysis

- Diagnostic scripts (scratchpad, not committed to the repo):
  `diag_demo_fails.py` (5-episode single-run rerun of ep0/2/4/5/12, verbose),
  `diag_ep0_raw.py` (ep0 rerun with a `ground()`-wrapping instrumentation printing
  raw per-cycle dist/bearing/confidence/not_visible).
- No code in `code/` was modified — all instrumentation was done via a Python-level
  monkeypatch in the standalone diagnostic scripts, to avoid any risk of touching
  files CX-4 might be concurrently using for training/eval.
- Cross-referenced existing artifacts: `eval/DIAG_deployed_demo_gt/`,
  `eval/p4_gate_demo/`, `eval/p4_gate_search/`, `eval/p4_gate_search_rerun/`,
  `docs/rot_dart.md`, `docs/grounding_dist.md`, `docs/cam_p1.md`, `docs/cam_p2.md`,
  `docs/cam_p4_gate.md`.
