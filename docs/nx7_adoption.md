# NX-7 ADOPTION — ep1 regression fix attempt (Fix A / Fix B / A+B), full-gate protocol

**Date:** 2026-07-09
**Agent:** NX-7 (follow-on to `docs/nx6_final.md`, `docs/nx6_judge.md`, `docs/nx2_iso.md` §M2+M5)
**Starting state:** `GROUND_NET=1` (`runs/nx6_heatmap_A/model_best.pt`, tau=0.59) scores demo
12/15 (fixes eps 0/5/12, ep4 compound + ep2 never-visible remain, per NX-6's own honest
accounting) but reproducibly **breaks ep1** (previously 100%-passing) — REJECT-verdicted by
NX-6 for exactly this reason (`docs/nx6_final.md` §2.2).

## TL;DR — VERDICT: REJECT (no adoption). `GROUND_NET`, `LOCK_M5`, and the new
`GROUND_NET_HYSTERESIS` all remain default OFF, unchanged.

Neither assigned fix flips ep1 to SUCCESS: **Fix A (LOCK_M5=1) failed 2/2 replays. Fix B
(acquire/track hysteresis, implemented as `GROUND_NET_HYSTERESIS`) was diagnosed as
structurally unable to help before being tested, and A+B combined also failed 2/2
replays** (0/7 SUCCESS across every ep1 replay run in this investigation, all 7 landing on
the same `FAIL[didnt-reach]`, final_dist≈5.83–5.90m). Root-cause instrumentation
(`eval/nx7_ep1_diag/`) found the actual mechanism is **not** the recall-ceiling ×
freeze-forever interaction NX-6 diagnosed — it is a **locomotion/physical stall**: the
robot's true world-frame (x,y) position goes measurably static (~0.1m box) around step
~250–300, ~1000 steps before any hold-goal-horizon-based recovery mechanism (M5, and — a
speculative third option tried here, M7) can even become eligible to fire. Grounding-layer
fixes cannot address a failure that originates and completes before the grounding layer's
recovery window opens. Because the ep1 mechanism-level bar ("must flip to SUCCESS across 2
replays") was never met, **the full demo/easy/search/fancy-demo gates were not run** — they
would not change the verdict (the demo number is already known to be 12/15-with-ep1-broken
from `docs/nx6_final.md`, below the brief's ≥13/15 ACCEPT bar) and running three more
15-episode gates would burn ~40min of GPU time to reconfirm a foregone conclusion. This
matches the brief's own "IF GATES FAIL: revert to default-off, honest doc, no sync" branch.

---

## 1. Fix A — `GROUND_NET=1 LOCK_M5=1` (zero new code)

**Hypothesis:** M5's bounded coast→rescan, after hold-goal-horizon expiry, re-centers the
target and lets the detector re-acquire (loss mechanism = bearing drift toward frame edge).

**Instrumented single-episode replay** (`eval/nx7_ep1_diag/nx7_diag_ep1.py`, non-invasive
monkeypatch of `code.inferencer.classical_ground`, same pattern FA-1/NX-6 used):

| run | M5 trigger | rescan fire | re-acquire (conf) | result |
|---|---|---|---|---|
| 1 | **YES**, step=1290 | **YES** | **YES**, conf=0.761 (≫tau=0.59), bearing recentered to +26.8° | **FAIL[didnt-reach]**, final_dist=5.840 |
| 2 | not needed — detector spontaneously re-acquired at cycle~124 (step~1240) before the 100-cycle hold horizon expired | n/a | **YES**, conf up to 0.897 | **FAIL[didnt-reach]**, final_dist=5.892 |

Mechanism trace confirms every piece NX-6/the brief predicted actually happens (M5 fires,
rescan re-centers, detector re-acquires at high confidence) — **but the episode still
fails both times.** `HOLD_GOAL_HORIZON=100` grounding cycles × ~10 steps/cycle ≈ 1000
steps of coasting before M5 is even eligible to trigger; since the underlying loss begins
at grounding-cycle ~30 (~step 300), M5 cannot fire before step ~1300 — leaving only ~100
of the demo cap's 1400 steps to reorient and walk ~5.8m back to the target. Not enough,
even with an accurate, high-confidence re-detection.

**Fix A: FAILED (0/2 SUCCESS). Per protocol, proceed to Fix B.**

---

## 2. Root-cause re-diagnosis (before touching Fix B) — this is the material finding

Per the brief: *"Check nx6 docs first: on ep1's replay, what does confidence actually
hover at after the drop? If it's <0.3 (true loss, not marginal), hysteresis won't help."*
NX-6's own doc only reports the post-threshold `not_visible` boolean (the raw sub-tau
confidence is discarded by `_ground_net()`'s hardcoded `GroundingResult(0,1,0,0.0,True)`
on a miss) — so this had to be measured directly, one layer deeper
(`eval/nx7_ep1_diag/nx7_diag_ep1_rawconf.py`, wraps the loaded `HeatmapDetector.infer()`
itself, `conf_thresh=0.0`, so `out['confidence']` is always the true raw peak sigmoid
value, present or not — `nx6_heatmap_model.decode_single` already computes this
unconditionally).

**Result:** during the true stuck window (grounding-calls 33–72, i.e. steps ~330–720),
raw confidence is **almost entirely <0.1** (mostly 0.01–0.07 — noise-floor peaks at the
*wrong-sign* bearing and physically-implausible distances, e.g. dist=23.7m, bearing=-26.65°
vs the true target's +25–27°) with exactly **one** marginal blip (call 34: conf=0.514,
accurate position) in ~40 cycles. This matches the brief's own "<0.3 → true loss, hysteresis
won't help" branch almost exactly — **so Fix B alone was expected to fail before it was
even implemented.**

**But a second, unplanned finding is the actually decisive one.** Confidence *does*
eventually recover robustly on its own in this same run — present=True, conf 0.66–0.90,
accurate (dist,bearing), for 60+ consecutive cycles starting ~call 73 (step ~730) — yet the
episode **still ends `FAIL[didnt-reach]`, final_dist=5.903m**, i.e. functionally
identical to every run where detection never recovered at all. Two follow-up checks
pinned down why:

- **`LockGate.gate_detection` (M3) is not the blocker.** Class-level monkeypatch logging
  every call (`nx7_diag_ep1_m3.py`) on a fresh replay shows **28/28 calls accepted**, state
  staying `CONFIRMED` throughout, incumbent tracking the new detections cleanly (near-zero
  bearing/dist innovation) — M3 is doing exactly what it should.
- **The robot's true world-frame position is physically frozen.** `mujoco.mj_step`
  instrumentation (`nx7_diag_ep1_qpos.py`) sampling `data.qpos[0:2]` shows x,y pinned to a
  **~0.1m box** (x≈-1.21 to -1.33, y≈-2.46 to -2.58) from mj_step-call ~1000 (≈env step
  250, `CONTROL_DECIMATION=4`) through episode end at step 1400 — height stays ~0.75m the
  entire time (never trips `FALL_HEIGHT`, i.e. not a fall). **The robot is locomotion-stuck,
  independent of what the grounding/lock layer reports.** Classical grounding succeeds on
  this exact episode (`fd=0.37`, `docs/nx6_final.md` §2.1 baseline column), so GROUND_NET's
  slightly different early-episode (steps 0–250) heading/EMA dynamics evidently steer the
  robot into a pose/position from which it cannot proceed — most plausibly a collision/
  obstacle interaction along a different approach line than classical took, though the
  exact geometry wasn't further isolated (out of this pass's scope — see §5).

This reframes the mechanism entirely: **the detector confidence collapse is a downstream
symptom of the camera view going static once the robot is physically stuck, not the root
cause of the stall.** Every hold-goal-horizon-keyed recovery mechanism (M5, and M7, tried
below) is structurally too late by construction, because the stall itself completes
(step ~250–300) before any of them are even eligible to evaluate (M5/M7 both require ~1000
steps or ~1.75m of accumulated walked distance before they can trigger — neither condition
is available when the robot isn't moving).

---

## 3. Fix B — acquire/track hysteresis (implemented, then A+B combined tested)

Implemented in `code/grounding.py` per the brief's spec: `tau_acquire=GROUND_NET_TAU=0.59`
unchanged; while a **track** is live (state kept in `_ground_net_track_dist_m`/
`_ground_net_track_bearing_rad`, reset per episode via the new
`reset_ground_net_track()`), a continuation detection down to
`GROUND_NET_TAU_TRACK=0.40` is accepted iff spatially continuous with the track, gated by
the *same* M3 innovation-gate constants `code/lock_mgmt.py` uses downstream (imported, not
duplicated — `code/grounding.py` now does `from code.lock_mgmt import M3_GATE_BEARING_DEG,
...`; no circular import, confirmed — `lock_mgmt.py` only imports `code.scan_sched`). Gated
behind `GROUND_NET_HYSTERESIS` (opt-in, default OFF); when off, `det.infer()` is still
called at `conf_thresh=GROUND_NET_TAU` exactly as before — **zero behavior change on the
default path**, confirmed by a regression check (§4).

Given §2's finding (post-drop confidence is <0.3 the overwhelming majority of the time —
true loss, not marginal), the brief's own decision rule says skip straight to **A+B
combined**:

`GROUND_NET=1 LOCK_M5=1 GROUND_NET_HYSTERESIS=1 GROUND_NET_TAU_TRACK=0.40`
(`eval/nx7_ep1_diag/nx7_diag_ep1_fixAB.py`, 2 replays):

| run | visible cycles (of 140) | M5 trigger | result |
|---|---|---|---|
| 1 | 31 (vs. 24 with Fix A alone) | step=1290 | **FAIL[didnt-reach]**, final_dist=5.835 |
| 2 | 33 | (M5 fired; not separately logged this run) | **FAIL[didnt-reach]**, final_dist=5.877 |

Hysteresis modestly extends the visible-cycle count (24→31–33) by rescuing the rare
marginal-confidence cycles, but does not materially change when M5 becomes eligible to
fire (still ~step 1290) nor the outcome — consistent with §2's physical-stall diagnosis:
the bottleneck isn't confidence-threshold timing at all.

**Fix B / A+B combined: FAILED (0/2 SUCCESS, on top of Fix A's own 0/2).**

### 3.1 One more speculative check (outside the assigned A/B menu): LOCK_M7

`LOCK_M7` (NX-5's odometry-coherence watchdog, `docs/nx5_coherence.md`, pending gate
verdict) monitors the robot's own *measured* displacement projected onto the goal bearing
— conceptually the closest existing mechanism to "robot isn't making progress," rather
than "detector lost the target." Tried once (`GROUND_NET=1 LOCK_M7=1 LOCK_M5=0`): **M7
never fires** — no trigger logged, `FAIL[didnt-reach]`, final_dist=5.842. Expected in
hindsight: M7's accumulator only evaluates once `M7_X_WALK_M=1.75m` of *actual* walked
displacement has accumulated (`code/lock_mgmt.py` `end_of_cycle()`), and since the robot's
real displacement is ~0 while stuck, that threshold is never reached — M7 requires motion
to fire, but the failure mode here *is* the absence of motion. Confirms none of the three
existing watchdogs (M4/M5/M7) are the right shape of mechanism for this specific failure.

---

## 4. Regression check — Fix B plumbing does not disturb the NX-6-verdicted path

`code/grounding.py` was edited (additive) even though the ultimate verdict is REJECT, per
this codebase's established practice of keeping tested-but-REJECTed mechanisms in the repo
as opt-in/default-OFF (same treatment as M2/M4/M5 themselves). Confirmed
byte-behavior-safe: with `GROUND_NET_HYSTERESIS` unset, `GROUND_NET=1` alone on demo eps
0/5/12 (`eval/nx7_ep1_diag/nx7_regress_check.py`) still reproduces NX-6's own numbers
exactly:

| ep | target | result | final_dist | NX-6 baseline (`docs/nx6_final.md` §2.1) |
|---|---|---|---|---|
| 0 | cyan cone (4.32m) | SUCCESS | 0.380 | 0.37 / 0.38 |
| 5 | cyan ball (8.85m) | SUCCESS | 0.367 | 0.37 / 0.36 |
| 12 | cyan cube (6.18m) | SUCCESS | 0.369 | 0.37 / 0.36 |

No regression. (ep1 spot-replay is the entire subject of §§1–3 above, so not repeated
here.)

---

## 5. Why full demo/easy/search/fancy-demo gates were not run

The brief's protocol is: validate the ep1 mechanism-level fix (must flip to SUCCESS across
2 replays) **before** the full 15-episode gates. That bar was never met — 0/7 total ep1
replays succeeded across every fix combination tried (Fix A ×2, Fix A+B ×2, plus 3 more
diagnostic-only replays that also happened to be full episode runs: the raw-confidence
run, the M3-gate check, the qpos-trajectory check — all 7 landed on `FAIL[didnt-reach]`,
final_dist in a tight 5.827–5.903m band, strongly reproducible). Per the brief's own
"IF GATES FAIL: revert to default-off, honest doc, no sync" branch, running the full
demo/easy/search/fancy-demo gates at this point would not change the outcome: the demo
number is already known from `docs/nx6_final.md` to be 12/15 with ep1 broken (below the
brief's ≥13/15 ACCEPT bar) whenever `GROUND_NET=1 LOCK_M5=0` and neither tested fix changes
that ep1 stays broken — so the three 15-episode gates (~35–45 GPU-minutes combined, based
on NX-6's own per-run wall-clock) were skipped as a foregone-conclusion re-confirmation,
in favor of spending that time on deeper root-cause instrumentation (§2), which is the
actually load-bearing new information this pass produced.

---

## 6. Verdict and disposition

**REJECT.** No adoption.

- `GROUND_NET` stays default OFF (unchanged from `docs/nx6_final.md`).
- `LOCK_M5` stays default OFF (unchanged from `docs/nx2_final.md`/`docs/nx2_iso.md`).
- New `GROUND_NET_HYSTERESIS`/`GROUND_NET_TAU_TRACK` constants (this pass): default OFF,
  opt-in, kept in the repo (tested working, does not regress eps 0/5/12, does not fix ep1
  either alone or combined with LOCK_M5) — same "keep as opt-in, verdicted REJECT for
  default-on" treatment this codebase has applied to M2/M4/M5 throughout NX-2 through
  NX-5.
- No confirm run needed (no default changed).
- **Nothing synced** to `VLA_mujoco_unitree/code/` — sync is
  ADOPT-conditioned only, and this pass did not adopt.

### Files changed
- `code/grounding.py` — additive only: `GROUND_NET_HYSTERESIS`/`GROUND_NET_TAU_TRACK` env
  constants, `_ground_net_track_dist_m`/`_ground_net_track_bearing_rad` module state,
  `reset_ground_net_track()`, the M3-constant import block, and the hysteresis
  accept/reject branch inside `_ground_net()`. Zero behavior change when
  `GROUND_NET_HYSTERESIS` is unset (confirmed, §4). No other file touched.

### Diagnostic artifacts (kept, not part of the pipeline)
`eval/nx7_ep1_diag/`: `nx7_diag_ep1.py` (Fix A mechanism replay + its 2 run logs/JSON),
`nx7_diag_ep1_rawconf.py` (raw confidence instrumentation), `nx7_diag_ep1_m3.py` (M3-gate
call log), `nx7_diag_ep1_qpos.py` (world-frame position trace), `nx7_diag_ep1_fixAB.py`
(A+B combined, 2 runs), `nx7_regress_check.py` (eps 0/5/12 regression check) — each with
its `.log`/`.json` output alongside.

---

## 7. What remains / follow-on

This pass's actual contribution is narrowing *what kind* of fix ep1 needs, which is not
what NX-6 assumed:

- **Not a grounding-layer problem.** ep1's failure under `GROUND_NET=1` is a locomotion/
  physical stall (robot's world-frame position goes static ~step 250–300), not a detector
  recall gap — the recall gap NX-6 observed is a downstream symptom (static camera view
  once physically stuck), not the cause. No amount of confidence-threshold tuning or
  hold-goal-horizon rescanning can fix a failure that completes before those mechanisms'
  eligibility window opens (~1000 steps / ~1.75m of accumulated walked displacement later).
- **The concrete next step:** instrument *why* the robot stops translating at that specific
  pose/heading under GROUND_NET's (slightly different than classical's) early-episode
  trajectory — likely a collision/obstacle interaction along a divergent approach line,
  possibly combined with a gait/policy dead-zone at large sustained bearing error (~20–27°)
  — this needs scene-geometry + contact-force instrumentation, out of scope for a
  "grounding backend" pass. If it is a genuine collision, the fix belongs in the
  locomotion/steering policy or an obstacle-aware path correction, not in `grounding.py` or
  `lock_mgmt.py`.
- **A watchdog shape that doesn't exist yet:** none of M4 (distance-trend), M5
  (hold-timeout), or M7 (accumulated-displacement-vs-progress) fire on "near-zero
  displacement for N cycles regardless of odometry accumulation" — that's a structurally
  different (and simpler) trigger condition than any of the three, and based on this
  episode's trace would fire far earlier (~step 250–300 instead of never/step-1290) if it
  existed. Worth a standalone isolation-gate pass (own mechanism, own KEEP/REJECT verdict,
  matching this codebase's established discipline) before wiring anything.
- NX-6's own longer-training recommendation (`docs/nx6_final.md` §8: heatmap training was
  cut short at epoch 41/60) remains valid on its own terms for the detector's genuine
  recall ceiling (~0.71–0.76) but is now understood to be orthogonal to ep1 specifically.
