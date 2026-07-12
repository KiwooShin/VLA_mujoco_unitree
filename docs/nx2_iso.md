# NX-2 Lock-Management — Isolation Gate Reports

Each mechanism (M1-M5) is gated independently against the demo/classical
baseline (`eval/p4_gate_demo`, 10/15=66.7%, failing eps 0,2,4,5,12). Sections
below are appended by independent per-mechanism isolation-gate agents.

## M1 — area-quality floor (isolation gate)

**Date:** 2026-07-09
**Agent:** M1 isolation gate (LOCK_M1 only, all other LOCK_M2..M5 unset/OFF)
**Run:** `LOCK_M1=1 python code/eval_closedloop.py --checkpoint checkpoint/goto_best.pt --arch A --difficulty demo --goal-source classical --n 15 --seed 999 --device cuda --out eval/nx2_iso_M1`
**Constant used:** `M1_AREA_FLOOR_PX2 = 100.0` (shipped default in `code/lock_mgmt.py`, unchanged)

### Result: 10/15 = 66.7% success — byte-identical pass/fail pattern to baseline

| ep | baseline (`eval/p4_gate_demo`) | M1-only (`eval/nx2_iso_M1`) | changed? |
|---|---|---|---|
| 0 | FAIL[didnt-reach] fd=3.39 | FAIL[didnt-reach] fd=3.40 | no |
| 1 | SUCCESS fd=0.37 | SUCCESS fd=0.36 | no |
| 2 | FAIL[didnt-reach] fd=10.62 | FAIL[didnt-reach] fd=10.74 | no |
| 3 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | no |
| 4 | FAIL[didnt-reach] fd=10.20 | FAIL[didnt-reach] fd=10.34 | no |
| 5 | FAIL[didnt-reach] fd=3.51 | FAIL[didnt-reach] fd=3.39 | no |
| 6 | SUCCESS | SUCCESS | no |
| 7 | SUCCESS | SUCCESS | no |
| 8 | SUCCESS | SUCCESS | no |
| 9 | SUCCESS | SUCCESS | no |
| 10 | SUCCESS | SUCCESS | no |
| 11 | SUCCESS | SUCCESS | no |
| 12 | FAIL[didnt-reach] fd=6.08 | FAIL[didnt-reach] fd=5.89 | no |
| 13 | SUCCESS | SUCCESS | no |
| 14 | SUCCESS | SUCCESS | no |

Same 5 failing episodes as baseline (0, 2, 4, 5, 12), same 10 passing episodes.
Per-episode step counts and final_dist all within normal run-to-run jitter
(e.g. ep1 933→931 steps, ep9 1201→1220 steps) — no episode's success/fail
*tag* moved. Zero flips in either direction, so the cam_p0.md
single-episode-flip non-determinism check does not apply here (nothing flipped
at all, let alone exactly one).

**Target eps 0 and 5: NOT fixed.** Both remain `FAIL[didnt-reach]`, final_dist
essentially unchanged from baseline (ep0 3.39→3.40m, ep5 3.51→3.39m).

**No previously-passing episode newly broken.**

### Why: this is the exact outcome `docs/nx2_impl.md` §2 already predicted

`docs/nx2_impl.md`'s own instrumentation (before this isolation gate ran) found:
- ep0's worst accepted blob measured 124.0 px² — just *above* the 100 px² floor
  → 0 rejections in an instrumented single-episode rerun (69/69 accepted).
- ep5's worst accepted blob measured 188.0 px² — further above the floor still.
- ep3 (currently-passing)'s legitimate minimum accepted blob is 123.5 px² —
  meaning **any** floor raised enough to reject ep0's 124 px² sliver (floor
  ≥125) or ep5's 188 px² sliver (floor ≥189) would sit *above* ep3's own
  legitimate far-range detection and risk truncating a currently-passing
  episode.

This full 15-episode isolation gate empirically confirms that prediction end
to end: at the shipped floor (100 px²), M1 makes **zero** detections gate
calls reject on either target episode, hence zero behavior change anywhere in
the 15-episode suite — a clean, mathematically-forced null result, not a bug
or a noisy non-effect.

### Tuning considered, not attempted

A TUNE re-gate was considered but not run: the floor-vs-ep3 arithmetic above
is a hard, already-documented constraint, not something a re-gate could
resolve. Any floor high enough to catch ep0 (≥125) or ep5 (≥189) provably
exceeds ep3's own legitimate minimum (123.5) — there is no single constant
value in this design that fixes either target episode without risking a
currently-passing regression, so a tune attempt would not be a good-faith
"validate a specific improved value," it would just trade one known failure
for a new one. Burning the ONE allowed re-gate attempt on a change already
proven mathematically unable to help wasn't a good use of it.

### VERDICT: KEEP

Neutral-with-no-regression: M1 fixes neither target episode (0 or 5) but
introduces zero regressions anywhere in the 15-episode demo/classical gate,
matching the design brief's own classification of M1 as a MEDIUM-confidence,
rank-#3 "defense-in-depth hygiene layer" rather than a standalone fix for
ep0/ep5 — `docs/rs1_lock_mgmt.md`'s dedicated fix for the false-lock family is
M4 (divergence watchdog), not M1. Recommend keeping M1 in the composite build
for its documented hygiene value against more degenerate blobs (e.g. ep2's
44/60 px² transients) even though it does not move the needle on this
mechanism's specifically-assigned targets.

**Artifacts:** `eval/nx2_iso_M1/summary_archA_classical_predicted_demo.json`,
`eval/nx2_iso_M1/run.log`. Smoke test (1 ep, ep0 only) at
`eval/nx2_iso_M1_smoke/`. Note on infra: this machine had 3-4 other parallel
isolation-gate jobs (M2/M3/M4) contending for CPU during this run, inflating
wall-clock ms/step 5-25x over baseline (e.g. ep0 508ms/step vs baseline's
21.9ms/step) without affecting the deterministic simulation outcome —
final_dist/steps/success-tag values above are unaffected by this contention,
only wall-clock is.

## M3 — innovation gate + incumbent inertia (isolation gate)

**Date:** 2026-07-09
**Agent:** M3 isolation gate (LOCK_M3 only, all other LOCK_M1/M2/M4/M5 unset/OFF)
**Run:** `LOCK_M3=1 python code/eval_closedloop.py --checkpoint checkpoint/goto_best.pt --arch A --difficulty demo --goal-source classical --n 15 --seed 999 --device cuda --out eval/nx2_iso_M3`
**Constants used:** shipped defaults, unchanged (`M3_GATE_BEARING_DEG=25`, `M3_GATE_BEARING_NEAR_MULT=1.5`, `M3_NEAR_RANGE_M=2.0`, `M3_GATE_DIST_FLOOR_M=0.8`, `M3_INCUMBENT_MARGIN=1.3`, `M3_INCUMBENT_K=2`)

### Result: 10/15 = 66.7% success — byte-identical pass/fail pattern to baseline

| ep | baseline (`eval/p4_gate_demo`) | M3-only (`eval/nx2_iso_M3`) | changed? |
|---|---|---|---|
| 0 | FAIL[didnt-reach] fd=3.39 | FAIL[didnt-reach] fd=3.40 | no |
| 1 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | no |
| 2 | FAIL[didnt-reach] fd=10.62 | FAIL[didnt-reach] fd=10.89 | no |
| 3 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | no |
| 4 | FAIL[didnt-reach] fd=10.20 | FAIL[didnt-reach] fd=10.27 | no |
| 5 | FAIL[didnt-reach] fd=3.51 | FAIL[didnt-reach] fd=3.45 | no |
| 6 | SUCCESS | SUCCESS | no |
| 7 | SUCCESS | SUCCESS | no |
| 8 | SUCCESS | SUCCESS | no |
| 9 | SUCCESS | SUCCESS | no |
| 10 | SUCCESS | SUCCESS | no |
| 11 | SUCCESS | SUCCESS | no |
| 12 | FAIL[didnt-reach] fd=6.08 | FAIL[didnt-reach] fd=6.68 | no |
| 13 | SUCCESS | SUCCESS | no |
| 14 | SUCCESS | SUCCESS | no |

Same 5 failing episodes as baseline (0, 2, 4, 5, 12), same 10 passing episodes.
Zero flips in either direction (nothing moved between SUCCESS/FAIL at all), so
the cam_p0.md single-episode-flip non-determinism check does not apply — there
is nothing ambiguous to re-run for stability.

**Target ep 12: NOT fixed.** Remains `FAIL[didnt-reach]`; final_dist actually
slightly worse (6.08→6.68m, within the kind of run-to-run spread seen
elsewhere in this suite, e.g. ep2's 10.62→10.89m — not a meaningful
degradation, just noise on top of an already-failing trajectory).

**No previously-passing episode newly broken.**

### Root-cause diagnosis: why M3 cannot touch ep12 (traced, not guessed)

Instrumented `LockGate.gate_detection` (full call-by-call trace, scratchpad
`diag_ep12_m3.py`/`diag_ep12_branch.py`) on an isolated ep12 rerun. The
mid-approach hijack is real and pinpoint-able: incumbent at
`dist=1.636m bearing=+10.1° area=229px²` is immediately followed by a
detection at `dist=0.965m bearing=-0.16° area≈60000px²` (a ~260x area jump,
0.67m dist jump, ~10° bearing jump) that gets accepted **instantly**, and
stays locked there (bearing frozen at exactly -0.16°, dist ~0.97m, area
~59700-60250px², essentially unchanging) for the rest of the episode.

Branch-level tracing (added print statements inside a copy of
`gate_detection`, not just the public accept/reject boolean) shows the
accept path taken is:

```
[23] bypass=True LOCK_M3=True -> unconditional accept
[24] bypass=True LOCK_M3=True -> unconditional accept
```

i.e. `self._discontinuity_cooldown > 0` at the exact cycle of the hijack —
this is the **mandatory carve-out** (`docs/rs1_lock_mgmt.md` risk #2,
`LockGate.mark_discontinuity()`), fired by `inferencer.py`'s CAM-2
Schmitt-trigger camera handoff (the incumbent's 1.636-1.676m range straddles
`CAM_D_HI=1.6m` exactly). The handoff swaps the active camera
(GROUNDING↔PROXIMITY), and the *other* camera's grounding call at that exact
cycle returns a large, dead-center false-positive blob. Per design (and
correctly so — this carve-out exists specifically so a legitimate
camera-geometry discontinuity isn't misread as an "innovation violation"),
that post-handoff detection is trusted **unconditionally**, bypassing M3's
bearing/dist gate and the incumbent-margin/K-streak challenger logic
entirely.

**This means no M3 constant can fix ep12 in this scenario** — the failure
never reaches M3's own gated code path at all. Confirmed empirically, not
just reasoned: reran the identical ep12 scenario with M3's gate constants
substantially tightened (`M3_GATE_BEARING_NEAR_MULT 1.5→1.0`,
`M3_GATE_DIST_FLOOR_M 0.8→0.3`, `M3_INCUMBENT_MARGIN 1.3→5.0`,
`M3_INCUMBENT_K 2→3` — none of which are the shipped values, a deliberately
extreme test) and the outcome was unchanged: same hijack, same
`bypass=True` acceptance path, same `FAIL[didnt-reach]` (final_dist 6.08m
untuned vs 5.67m tightened on a 1400-step rerun — still failing, no
meaningful improvement). The tightened constants only affected a handful of
*unrelated* mid-episode reject/accept calls elsewhere in the trace (8
rejections vs 4 at defaults) — none of which touched the pivotal
carve-out-bypassed transition.

### Tuning: attempted as a quick single-episode probe, not the official re-gate

Per the anti-hang/verdict protocol's "you may adjust the constant and
re-gate ONCE," I first cheaply validated on the isolated ep12 scenario alone
(not the full 15-ep suite) whether *any* plausible M3-constant tightening
could help, before spending the one official full re-gate on it. The result
above shows it cannot — the failure path structurally bypasses every one of
M3's tunable constants via the mandatory discontinuity carve-out. Spending
the official one-time full-suite re-gate on the same (already-falsified)
hypothesis would not be a good-faith use of it, so it was not run. This
mirrors the M1 isolation gate's precedent (`docs/nx2_iso.md` §M1) of
declining the tune attempt once the evidence already rules it out
mathematically/structurally, rather than burning it on a change already
shown not to matter.

**Note for whoever owns the camera-handoff carve-out or a future
mechanism:** the actual fix for this specific ep12 failure mode would need
to live in the CAM-2 handoff path itself (e.g. requiring the post-handoff
detection to be corroborated by a second frame before unconditionally
trusting it), not in M3 — but weakening/adding conditions to the mandatory
carve-out is explicitly flagged by the design brief (risk #2) as
reintroducing "the exact ep13/cam_p3/cam_p4 deadlock class this codebase
already fought once," so that is out of scope for this M3-only isolation
gate and is not attempted here.

### VERDICT: KEEP

Neutral-with-no-regression: M3 fixes neither target episode (12) but
introduces zero regressions anywhere in the 15-episode demo/classical gate.
Unlike M1's case, this isn't a floor-vs-passing-episode arithmetic wall —
it's a structural interaction with a *different*, deliberately-protected
mechanism (the CAM-2 Schmitt-handoff carve-out), traced and confirmed at the
branch level. M3's own gate logic is confirmed *not* dead code elsewhere in
the episode (it does gate/reject a handful of transient outliers both at
shipped and tightened constants), just not at the one moment that decides
ep12's outcome. Recommend keeping M3 in the composite build for its general
association-gating hygiene value (protects against genuine gradual-drift
mis-associations elsewhere in the suite) while flagging that ep12's fix, if
pursued, belongs in the CAM-2 handoff path, not M3.

**Artifacts:** `eval/nx2_iso_M3/summary_archA_classical_predicted_demo.json`,
`eval/nx2_iso_M3/run.log`. Smoke test (1 ep, ep0 only) at
`<scratch>/nx2_iso_M3_smoke2/`
(scratchpad, not committed). Diagnostic scripts (scratchpad, not committed):
`diag_ep12_m3.py` (call trace), `diag_ep12_m3_tuned.py` (tightened-constant
probe), `diag_ep12_branch.py` (branch-level trace that found the
`bypass=True` carve-out path). Note on infra: this machine had 3 other
parallel isolation-gate jobs (M1/M2+M5/M4) contending for CPU/GPU during the
full-suite run, inflating wall-clock ms/step up to ~25x over baseline in the
early episodes (e.g. ep0 512ms/step vs baseline's 21.9ms/step) before easing
once those jobs finished partway through — this affected only wall-clock,
not the deterministic simulation outcome (final_dist/steps/success-tag are
unaffected).

## M4 — divergence watchdog (isolation gate)

**Command:** `LOCK_M4=1 python code/eval_closedloop.py --checkpoint
checkpoint/goto_best.pt --arch A --difficulty demo --goal-source classical
--n 15 --seed 999 --device cuda --out eval/nx2_iso_M4` (ONLY `LOCK_M4=1` set;
M1/M2/M3/M5 all default-off). 1-episode-equivalent smoke first (n=3, covers
eps 0-2 including the target episode) confirmed crash-free before committing
to the full run. Full n=15 run repeated twice (`eval/nx2_iso_M4`,
`eval/nx2_iso_M4_rerun`) because the first pass showed two unexpected flips
(more than the "exactly one" noise-class threshold in `docs/cam_p0.md`) —
both flips reproduced identically on rerun, confirming they are real
mechanism effects, not run-to-run jitter.

### Result: 10/15 = 66.7% success both runs — same aggregate rate as
baseline, but the *set* of failing episodes changed

| ep | baseline (`p4_gate_demo`) | M4 run 1 | M4 run 2 (rerun) | verdict |
|---|---|---|---|---|
| 0 | FAIL, fd=3.39 | FAIL, fd=3.39 | FAIL, fd=3.39 | unchanged |
| 1 | SUCCESS | SUCCESS | SUCCESS | unchanged |
| **2** | **FAIL, fd=10.62** (target) | **FAIL, fd=10.98** | **FAIL, fd=11.08** | **target NOT fixed — slightly worse** |
| 3 | SUCCESS | SUCCESS | SUCCESS | unchanged |
| 4 | FAIL, fd=10.20 | FAIL, fd=10.31 | FAIL, fd=10.36 | unchanged |
| **5** | **FAIL, fd=3.51** | **SUCCESS, fd=0.39** | **SUCCESS, fd=0.36** | **unexpected fix (not an M4-claimed target)** |
| 6-11 | all SUCCESS | all SUCCESS | all SUCCESS | unchanged |
| 12 | FAIL, fd=6.08 | FAIL, fd=5.64 | FAIL, fd=5.95 | unchanged (not M4's target) |
| **13** | **SUCCESS, fd=0.37, steps=602** | **FAIL, fd=8.24, steps=1400** | **FAIL, fd=8.27, steps=1400** | **newly broken, reproduced on rerun — confirmed real, not noise** |
| 14 | SUCCESS | SUCCESS | SUCCESS | unchanged |

Both flips (ep5 fixed, ep13 broken) are stable to within normal run-to-run
final_dist jitter (~0.03m) across the two full independent 15-episode runs —
this rules out the single-episode-flip noise class documented in
`docs/cam_p0.md` (that note is specifically about isolated-vs-sequential
non-determinism; here the *same* full-sequential condition was run twice and
gave the same episode-level pattern both times).

### ep2 (the target): mechanism fires but does not convert to success

`docs/nx2_impl.md` §3 already confirmed M4 fires on ep2 in a verbose
instrumented rerun (`[lock] M4 divergence -> drop+rescan at step=410`). This
isolation gate confirms the *aggregate* consequence of that firing: ep2 still
ends the episode as `FAIL[didnt-reach]`, and its final_dist is not improved
(10.98-11.08m with M4 vs 10.62m baseline — actually marginally *worse*, both
comfortably within a "still fails badly" band, not fixed even fractionally).

**Root cause (read from `code/lock_mgmt.py`, not guessed):** with M1/M2/M3
off (this gate's whole point — isolating M4 alone), `force_drop()` resets
`LockGate.state` to `'NONE'`, and `gate_detection()`'s `state != 'CONFIRMED'`
branch immediately re-confirms on the **very next raw detection** when
`LOCK_M2` is off (`code/lock_mgmt.py:226-229`: `if not LOCK_M2:
self._confirm(entry); return True`). Nothing in the isolated-M4 configuration
prevents that next detection from being the *same* false-positive blob that
caused the divergence in the first place — M4's own job is only to *detect*
a diverging track and force a rescan, not to *discriminate* good detections
from bad ones on reacquisition (that's M1's area floor and/or M2's N-of-M
consistency check, both off here by design). So in isolation, M4 produces a
"drop → rescan → immediately relock the same wrong blob → diverge again"
cycle rather than a genuine recovery — consistent with `docs/rs1_lock_mgmt.md`
§2's own composite design (M4 is ranked #1 by impact but is explicitly
written as *one piece* of a state machine that also depends on M2's
confirmation gate to make reacquisition meaningful, not a standalone fix for
ep2). This is a structural/inter-mechanism dependency, not a threshold that
this gate's constants can paper over.

### ep13 (previously 100%-passing): newly broken, reproduced on rerun

Baseline ep13 succeeds cleanly in 602 steps (fd=0.37). With `LOCK_M4=1` only,
it now runs the full 1400 steps and fails (fd=8.24-8.27m across both runs).
This is exactly the interaction risk `docs/rs1_lock_mgmt.md` itself names for
M4 ("must not false-trigger on... the first N cycles right after any
(re)confirmation, where legitimate heading-correction geometry can
transiently increase straight-line distance before the turn completes") —
the shipped exemption (`M4_EXEMPT_CYCLES_AFTER_CONFIRM=15`, only active in
the 15 cycles immediately after (re)confirmation) does not protect against a
legitimate *mid-episode* heading-correction/detour more than 15 cycles after
initial confirmation, which is what appears to be happening here (ep13's
approach evidently has a genuine transient distance increase later in the
walk that exceeds `M4_TREND_MARGIN_M=0.5m` over the `M4_WINDOW_N=15`-cycle
window and gets misread as divergence).

### Tuning considered, not attempted as a full re-gate

The obvious lever is loosening `M4_TREND_MARGIN_M` (0.5m) and/or widening
`M4_WINDOW_N` (15 cycles) to stop the ep13 false-trigger. This is plausible
for ep13 in isolation, but per the root-cause analysis above it would **not**
be expected to convert ep2 to success — ep2's gap is the post-trigger
reacquisition immediately relocking the same false positive (an M1/M2
dependency structurally absent from this M4-only configuration), not the
trigger threshold itself. Loosening the margin only delays *when* M4 fires
on ep2's already-strongly-diverging track (net drift ~6m over the episode,
comfortably clears any plausible margin eventually); it does not address why
reacquisition fails to correct course. Since no single M4 constant is
expected to both (a) stop ep13's false trigger and (b) convert ep2 to an
actual success, and since a full re-gate under current GPU contention costs
~35-40 minutes (two were already spent confirming the flips are real, not
noise), the one-time tune-and-regate allowance was not spent on a change the
structural analysis already predicts would not achieve a net improvement —
same reasoning precedent as M1's and M3's isolation gates in this same file
(declining the tune once the evidence/structure already rules out the
hypothesis, rather than burning it on a change already shown/reasoned not to
matter).

### VERDICT: REJECT

Fails both KEEP bars: the target episode (ep2) is not fixed (still
`FAIL[didnt-reach]`, final_dist not meaningfully improved, arguably slightly
worse), and this is **not** neutral-with-no-regression — ep13, a previously
100%-reliable passing episode, is newly and reproducibly broken (confirmed
identical across two independent full 15-episode reruns, ruling out the
single-flip noise class). The unaffected aggregate success rate (10/15 both
conditions) is coincidental, not evidence of safety: it's one real
regression (ep13) masked by one unclaimed, incidental improvement (ep5 —
not an episode `docs/rs1_lock_mgmt.md` ever attributes to M4). Recommend
against shipping M4 standalone; per the structural root-cause above, M4 most
plausibly only becomes net-positive once paired with M2 (so reacquisition
after a forced drop can't immediately re-confirm the same false positive)
and/or a loosened trigger margin/window (to stop firing on ep13-style
legitimate mid-episode heading corrections) — both are composite-build
questions for a follow-on all-mechanisms-on gate, out of scope for this
single-mechanism isolation pass.

**Artifacts:** `eval/nx2_iso_M4/summary_archA_classical_predicted_demo.json`,
`eval/nx2_iso_M4/run.log` (run 1); `eval/nx2_iso_M4_rerun/summary_archA_classical_predicted_demo.json`,
`eval/nx2_iso_M4_rerun/run.log` (stability-check rerun, identical condition).
Smoke test (n=3, eps 0-2) at
`eval/nx2_iso_M4_smoke/` was removed after confirming crash-free completion
of ep0 (1400 steps, video written, no exception) — not committed. Note on
infra: this run shared the GPU with other parallel isolation-gate jobs
(M1/M2+M5/M3), inflating wall-clock to ~500ms/step vs baseline's ~25ms/step
(~20x) for most of both runs — this affects only wall-clock time (~36-38 min
per full 15-episode run instead of ~7 min), not the deterministic
simulation outcome (final_dist/steps/success-tag reproduced identically in
kind across both runs).

## M2+M5 — N-of-M lock confirmation + bounded coast-to-rescan (isolation gate)

**Date:** 2026-07-09
**Agent:** M2+M5 isolation gate (`LOCK_M2=1 LOCK_M5=1`, all other `LOCK_M1/M3/M4` unset/OFF)
**Run:** `LOCK_M2=1 LOCK_M5=1 python code/eval_closedloop.py --checkpoint checkpoint/goto_best.pt --arch A --difficulty demo --goal-source classical --n 15 --seed 999 --device cuda --out eval/nx2_iso_M2_M5`
**Scope note:** per task brief, this mechanism has no specific target episode — it is general hardening and the bar is "must simply not regress."

### Result: 9/15 = 60.0% success — a confirmed, reproducible regression (baseline 10/15 = 66.7%)

| ep | baseline (`eval/p4_gate_demo`) | M2+M5 run 1 (`eval/nx2_iso_M2_M5`) | M2+M5 run 2 (`eval/nx2_iso_M2_M5_rerun`) | changed? |
|---|---|---|---|---|
| 0 | FAIL[didnt-reach] fd=3.39 | FAIL[didnt-reach] fd=3.39 | FAIL[didnt-reach] fd=3.38 | no |
| 1 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | no |
| 2 | FAIL[didnt-reach] fd=10.62 | FAIL[didnt-reach] fd=10.73 | FAIL[didnt-reach] fd=10.82 | no |
| 3 | SUCCESS fd=0.37 | SUCCESS fd=0.36 | SUCCESS fd=0.36 | no |
| 4 | FAIL[didnt-reach] fd=10.20 | FAIL[**fall**] fd=9.96 | FAIL[didnt-reach] fd=9.64 | tag jitters, still fails both times |
| 5 | FAIL[didnt-reach] fd=3.51 | FAIL[didnt-reach] fd=3.42 | FAIL[didnt-reach] fd=3.27 | no |
| 6 | SUCCESS | SUCCESS | SUCCESS | no |
| 7 | SUCCESS | SUCCESS | SUCCESS | no |
| 8 | SUCCESS | SUCCESS | SUCCESS | no |
| 9 | SUCCESS | SUCCESS | SUCCESS | no |
| 10 | SUCCESS | SUCCESS | SUCCESS | no |
| 11 | SUCCESS | SUCCESS | SUCCESS | no |
| 12 | FAIL[didnt-reach] fd=6.08 | FAIL[didnt-reach] fd=6.73 | FAIL[didnt-reach] fd=6.62 | no |
| **13** | **SUCCESS** steps=602 fd=0.37 | **FAIL[didnt-reach]** fd=9.26 | **FAIL[didnt-reach]** fd=9.21 | **YES — newly broken, both runs** |
| 14 | SUCCESS | SUCCESS | SUCCESS | no |

**Target episodes (0,2,4,5,12): none fixed.** All 5 remain `FAIL[didnt-reach]` (ep4's
tag flickers `fall`↔`didnt-reach` between runs but is a fail either way, matching the
baseline's own fail) — zero measurable benefit from M2 or M5, consistent with this
mechanism's own "no specific target" framing.

**ep13 newly broken — confirmed stable, not the single-flip noise class.** Per the
task's noise-class check (`docs/cam_p0.md`): exactly one unexpected flip was observed,
so the full 15-episode condition was rerun once before judging. Both independent full
runs reproduce `9/15 = 60.0%` and `FAIL[didnt-reach]` on ep13 with near-identical
final_dist (9.26m vs 9.21m) — this is a real, reproducible regression, not run-to-run
jitter.

### Root-cause isolation (single-episode diagnostic, matched harness)

Rather than burn additional full 15-episode reruns to bisect M2 vs M5, ep13's scene is
independent per-episode (`derive_rng(seed, ep_idx)`, `code/scene.py` — episode sampling
does not depend on prior episodes' RNG state), so it can be replayed standalone. This
standalone harness first reproduced the full-gate baseline/M2+M5 results exactly
(SUCCESS/FAIL and failure_tag match), validating it as a faithful proxy for this specific
episode before using it to bisect:

| Condition (ep13 only, standalone) | Result |
|---|---|
| baseline (both off) | SUCCESS, steps=624, fd=0.372 |
| `LOCK_M2=1` only | **FAIL[didnt-reach]**, steps=1400, fd=9.242 |
| `LOCK_M5=1` only | SUCCESS, steps=604, fd=0.380 |
| `LOCK_M2=1 LOCK_M5=1` | FAIL[didnt-reach], steps=1400, fd=9.271 (matches full-gate run) |

**M2 alone reproduces the regression; M5 is clean.** M2's `gate_detection()`
(`code/lock_mgmt.py`) requires `M2_CONFIRM_M=2` of the last `M2_CONFIRM_N=3` buffered
detections to be mutually consistent (within `M2_TOL_DIST_M=0.6m` / `M2_TOL_BEARING_DEG
=12°`) before a fresh lock (state `NONE`→`CONFIRMED`) is accepted; until then every call
returns `False` (treated as a miss by the caller). Structurally, **the very first-ever
detection in any episode can never satisfy this** — with 1 buffered entry, comparing it
to itself trivially gives `n_consistent=1 < M2_CONFIRM_M=2` regardless of tolerance — so
every episode's initial lock (and every re-lock after any lock loss) takes a minimum of
1 extra grounding cycle versus the pre-M2 behavior (which sets `CONFIRMED` immediately on
the first detection). This is normally a small, harmless delay, but in ep13's specific
closed-loop trajectory it was evidently enough to perturb the early approach heading and
send the robot the wrong way for the whole 1400-step episode (`fd=9.2m`, far past the
target) rather than the previously-clean 602-624-step approach.

### TUNE attempt (one, per protocol)

Hypothesis: if the regression were a tolerance mismatch (2nd/3rd buffered detection
falling just outside `M2_TOL_DIST_M`/`M2_TOL_BEARING_DEG` due to early-detection noise)
rather than the structural first-call-always-rejected floor above, loosening tolerance
should let ep13 confirm fast enough to recover its old trajectory. Tested via in-process
monkeypatch of `code.lock_mgmt.M2_TOL_DIST_M`/`M2_TOL_BEARING_DEG` (no on-disk edit to
the shared module — other parallel isolation-gate agents' processes were left untouched):

- **Loosened to `M2_TOL_DIST_M=1.2m` / `M2_TOL_BEARING_DEG=25°` (2x baseline):**
  standalone ep13 replay: still `FAIL[didnt-reach]`, steps=1400, fd=9.265 — **no
  improvement**.
- Confirmed with one official full 15-episode re-gate at this tuned value
  (`eval/nx2_iso_M2_M5_tuned`): **still 9/15 = 60.0%**, ep0/2/4/5/12 all still fail
  (zero target benefit, unchanged from untuned), **ep13 still `FAIL[didnt-reach]`,
  fd=9.25m** — the tune did not fix the regression.

This confirms the root cause is the structural minimum-confirmation-delay of N-of-M
gating with `M2_CONFIRM_M=2` (not a tolerance miscalibration) — tolerance is not "the
constant that's off." The only constant that would eliminate the delay
(`M2_CONFIRM_M=1`) makes M2 accept unconditionally on the first call, i.e. equivalent to
`LOCK_M2=0` (M2 disabled) — not a legitimate tune, it defeats the mechanism's stated
purpose.

### Verdict: REJECT

No target episode benefit (mechanism's own brief: "no specific target — must simply not
regress") and a confirmed, reproducible regression (ep13: SUCCESS→FAIL, stable across two
independent full-gate runs plus a matched single-episode bisection) that survives one
good-faith tune attempt (2x tolerance loosening, itself re-validated with a second full
15-episode gate). Recommend against shipping M2 as currently specified for this
checkpoint/eval configuration; M5 alone showed no adverse effect in the bisection
(`LOCK_M5=1` only: SUCCESS on ep13, matching baseline) and is not implicated in this
regression.

**Artifacts:** `eval/nx2_iso_M2_M5/` (run 1), `eval/nx2_iso_M2_M5_rerun/` (stability-check
rerun, identical condition), `eval/nx2_iso_M2_M5_tuned/` (tune-attempt full re-gate,
`M2_TOL_DIST_M=1.2` / `M2_TOL_BEARING_DEG=25.0`) — each with
`summary_archA_classical_predicted_demo.json` + `run.log`. Note on infra: these runs
shared the GPU/CPU with other parallel isolation-gate jobs (M1/M3/M4), inflating
wall-clock to ~440-510ms/step for episodes 0-2 (~20-25x baseline's ~20ms/step) before
sibling jobs finished and per-step cost dropped back to ~17-40ms/step for the remainder
of each run — this affects only wall-clock time (~30-40 min per full 15-episode run
instead of ~7 min), not the deterministic simulation outcome.
