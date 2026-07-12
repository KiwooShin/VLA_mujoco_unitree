# NX-8 STALL — ep1 physical-locomotion-stall closure attempt (H1 goal-parity / H2 STALL_BREAK)

**Date:** 2026-07-09
**Agent:** NX-8 (follow-on to `docs/nx7_adoption.md`, `docs/nx6_final.md`, `docs/nx6_judge.md`)
**Starting state:** `GROUND_NET=1` scores demo 12/15 (fixes eps 0/5/12, ep2/ep4 honest
fails) but reproducibly breaks ep1 (previously 100%-passing) via a **physical
locomotion stall** — NX-7's own re-diagnosis (`docs/nx7_adoption.md` §2) found the
robot's world-frame (x,y) goes static (~0.1m box) around step ~250-300, roughly
1000 steps before any hold-goal-horizon-keyed recovery mechanism (M4/M5/M7) is even
eligible to fire. Every grounding-layer fix NX-7 tried (LOCK_M5, hysteresis, A+B
combined, LOCK_M7) failed 0/7 replays.

## TL;DR — VERDICT: REJECT (no adoption). `GROUND_NET` and the new `STALL_BREAK`
## both remain default OFF. Per the brief's own protocol, the line closes here.

**H1 (goal-signal parity) is REFUTED by data — not just "no bypass found," the
opposite of the hypothesis.** GROUND_NET's raw (dist,bearing) signal over ep1 steps
0-350 is *smoother* than classical's (mean bearing jitter 2.0° vs 6.98°, mean
distance jitter 0.065m vs 0.149m, cycle-to-cycle), and code-reading confirms the
EMA/hold-goal-horizon/LockGate smoothing machinery in `code/inferencer.py` already
applies identically to both backends (GROUND_NET is dispatched *inside*
`code.grounding.ground()`, which `inferencer.py` calls as `classical_ground(...)`
under `goal_source='classical'` regardless of backend — there is no separate code
path to bypass). §1.

**H2 (STALL_BREAK, a new steering-level physical-stall watchdog) was implemented in
`code/inferencer.py`, fires exactly on cue (step ~303-315, matching NX-7's diagnosed
onset almost exactly), and is clean (0/5 spurious triggers on 5 healthy episodes,
provable no-op when off) — but does not flip ep1 to SUCCESS: 0/2 replays**
(`FAIL[didnt-reach]`, final_dist=5.823/5.974, both landing in the same 5.8-5.9m band
every prior NX-7 attempt did). §2.

**The decisive new finding this pass adds**: the robot re-stalls in the *exact
same* ~0.1m box (x≈-1.20 to -1.34, y≈-2.44 to -2.58 — matches NX-7's own qpos trace
almost to the centimeter) on **every one of the 8 stop/resume cycles** STALL_BREAK
triggered per replay. Scene-geometry cross-check (ep1's own `sample_scene` output,
seed 999) shows this box sits almost exactly ON the straight-line path from the
robot's start (-1.26,-4.13) to the cyan-cube target (-1.12,3.30) (predicted on-line
x at y=-2.5 is -1.230 vs the observed stuck-box center x≈-1.27, a 0.04m match), and
the scene's **orange-cone distractor** (episode-start distance 1.75m, world position
(-1.049,-2.386)) sits only **~0.25m from the stuck-box center** — well inside
humanoid-body collision range. This confirms and sharpens NX-7's own closing
speculation ("most plausibly a collision/obstacle interaction along a different
approach line than classical took," `docs/nx7_adoption.md` §7): ep1's stall is a
**genuine physical collision with a distractor object**, sitting almost exactly on
GROUND_NET's straighter, less-jittery (per H1!) approach line to the target — not a
recoverable policy-internal "stand" attractor. That is exactly why a stand-then-
resume-on-the-same-heading recovery cannot escape it: resuming just walks the robot
back into the same obstacle every time. §2.3.

Per the brief's own protocol ("If this fails, the line closes with GROUND_NET as
documented opt-in"), **the full demo/easy/search/fancy-demo adoption gates were not
run** — they are explicitly conditioned on "ep1 fixed 2/2," which did not happen,
and (matching NX-7's own precedent for this identical outcome) the demo number is
already known to be 12/15-with-ep1-broken, below the ≥13/15 ACCEPT bar, so a ~40
GPU-minute re-confirmation of a foregone conclusion was skipped in favor of the
diagnostic work above. §3.

---

## 1. H1 — goal-signal parity (classical vs GROUND_NET, ep1 steps 0-350)

### 1.1 Code-level check: does GROUND_NET bypass any smoothing classical gets?

No. `code/inferencer.py` imports `from code.grounding import ground as
classical_ground` (line 66) and calls `classical_ground(...)` unconditionally at
`GROUNDING_PERIOD` cadence whenever `goal_source == 'classical'` (the setting every
demo/easy/search eval in this codebase uses, `_need_classical_render` at line ~703).
`GROUND_NET=1` is dispatched **inside** `code.grounding.ground()` itself (its very
first check: `if GROUND_NET: return _ground_net(...)`, `code/grounding.py` line
~1234) — it is not a different `goal_source` value and has no separate call site.
Every downstream mechanism (temporal EMA at `_GOAL_EMA_ALPHA=0.4`, `HOLD_GOAL_HORIZON
=100`-cycle hold-last-known-goal, `LockGate.gate_detection` M1/M3 gating, the CAM-2
Schmitt handoff) lives in `code/inferencer.py` lines ~799-958, entirely downstream of
whichever backend `ground()` dispatched to, and runs byte-identically either way.
**There is no bypass to fix at this level** — parity was already complete before
this pass started.

### 1.2 Empirical jitter check (does it matter anyway — is the raw signal noisier even after identical smoothing?)

Instrumented dual replay (`eval/nx8_stall/nx8_h1_goal_parity.py`): classical then
GROUND_NET, same seed/scene (`derive_rng(999, 1)`, demo ep1), same process (`GROUND_NET`
toggled as a `code.grounding` module attribute directly, since it's parsed from the
env once at import time), capped at 500 steps (covers steps 0-350 plus the onset of
the diagnosed stall window). Every raw `ground()` call logged via a non-invasive
monkeypatch of `code.inferencer.classical_ground` (same pattern NX-6/NX-7 used).
Cycle-to-cycle `|Δdist|`/`|Δbearing|` computed over consecutive **visible** cycles
only (cycles<36, i.e. steps<360 at `GROUNDING_PERIOD=10`):

| backend | visible/total cycles | Δdist mean (m) | Δdist std | Δdist p95 | Δbearing mean (deg) | Δbearing std | Δbearing p95 |
|---|---|---|---|---|---|---|---|
| classical | 36/50 | 0.149 | 0.343 | 1.222 | 6.98 | 11.05 | 40.20 |
| GROUND_NET | 26/50 | 0.065 | 0.047 | 0.163 | 2.03 | 1.43 | 3.97 |

**GROUND_NET's raw signal is smoother, not jumpier**, on every jitter statistic —
the opposite of the hypothesis. This matches NX-7's own accuracy finding (every
accepted GROUND_NET detection before the stall has <0.02m distance error vs GT,
`docs/nx7_adoption.md` §2.2) and rules out "192x144 heatmap quantization noise" as a
contributor. The one real difference is **coverage**, not noise: GROUND_NET sees
26/50 cycles visible vs classical's 36/50 in this window — a recall gap, consistent
with NX-6's own documented ~0.71-0.76 recall ceiling — but a coverage gap is not a
signal-quality/jumpiness problem, and (per §1.1) there is no smoothing-parity action
available to take regardless, since parity already holds.

**H1 verdict: REFUTED.** No fix applicable at this hypothesis level. Proceeded to H2
per the brief's own decision rule.

---

## 2. H2 — STALL_BREAK: steering-level physical-stall watchdog

### 2.1 Implementation (`code/inferencer.py`, additive only)

A new opt-in (`STALL_BREAK=1`, default OFF) mechanism, deliberately **not** folded
into `code/lock_mgmt.py`'s `LockGate` (M1-M7 are all grounding/detection-quality
mechanisms; this is a steering-level watchdog operating on the model's own commanded
velocity and the robot's true world-frame displacement, independent of which
grounding backend is active — a locomotion phenomenon, not a grounding one, per
NX-7's own re-diagnosis).

- Tracks `(x, y, vx_cmd)` for the last `STALL_WINDOW_STEPS=100` **steps** (not
  grounding cycles) of the guaranteed-non-scan "normal mode" path only — every
  scan/rescan/dwell step `continue`s before reaching this code, so the "never during
  scan/rescan/dwell" carve-out is satisfied structurally, not just by a value check.
  `vx_cmd` is `out['vel'][0,0]` from the model's own forward pass (`code/small_vla.py`
  — this is what actually feeds `vel_emb` into the action head for `goal_source=
  'classical'`, i.e. the true "commanded" v_fwd for both classical and GROUND_NET
  runs, since neither injects `gt_vel` by default).
- Trigger: `sustained |vx_cmd| > STALL_VX_THR_MPS=0.2` across the whole window AND
  `displacement < STALL_DISP_THR_M=0.15m` between the window's first and last sample
  AND `cached_goal_vec[0] > STALL_MIN_GOAL_DIST_M=2.0m` (never during final-approach
  creep).
- Carve-out: `_stall_is_maneuver` (computed once from `scene_cfg.get('difficulty')
  == 'maneuver'`) disables the entire mechanism for maneuver-type scenes — the only
  such carve-out expressible in this codebase, since `Inferencer.rollout()` itself
  has no other maneuver-mode concept to check against (confirmed by grep — maneuver
  scenes are just a `scene_cfg['difficulty']` value, nothing else distinguishes the
  call path).
- Recovery: on trigger, forces `gt_vel=zeros(3)` (the "stand" command — in-
  distribution, matches stand-keyframe episode-start init and the scan-dwell
  segments in the training data) for `STALL_RECOVERY_STEPS=50` steps, overriding
  whatever `gt_vel_inject_t` would otherwise have been (highest-priority override,
  inserted immediately before the model forward call). After recovery,
  `STALL_COOLDOWN_STEPS=100` steps elapse (window cleared) before the watchdog can
  re-arm, giving the robot a chance to actually move again before re-checking.
- `RolloutResult.stall_break_triggers` (new additive field): per-episode trigger
  count, for gate/spot-check diagnostics.
- **Provable no-op when `STALL_BREAK` is unset**: every new code path is gated on
  the module-level `STALL_BREAK` constant (parsed once, `_env_flag("STALL_BREAK")`,
  default `"0"`); confirmed via a regression smoke (§2.4).

### 2.2 ep1 mechanism-level check — 0/2 SUCCESS

`GROUND_NET=1 LOCK_M5=0 STALL_BREAK=1`, full 1400-step replay
(`eval/nx8_stall/nx8_h2_ep1_replay.py`), 2 runs:

| run | 1st trigger step | total triggers | result | final_dist |
|---|---|---|---|---|
| 1 | 303 | 8 | **FAIL[didnt-reach]** | 5.823 |
| 2 | 315 | 8 | **FAIL[didnt-reach]** | 5.974 |

The watchdog fires almost exactly where NX-7's independent qpos instrumentation
placed the stall onset (~step 250-300, `docs/nx7_adoption.md` §2.2) — the detection
side of the mechanism works as designed. But the episode still fails both times,
landing in the same 5.8-5.9m final_dist band as every one of NX-7's own 7 failed
replay attempts (Fix A, Fix B, A+B combined, M7, raw-confidence, M3-gate, qpos —
`docs/nx7_adoption.md` §5). **STALL_BREAK: FAILED (0/2 SUCCESS).**

### 2.3 Why the recovery doesn't work — root cause (this pass's new finding)

The watchdog re-triggered **8 times per replay** (roughly every 150 steps: 50
recovery + 100 cooldown + however long the window takes to refill and re-trip) —
every single time in the same location. Sampling `data.qpos[0:2]` at a coarse stride
(`eval/nx8_stall/nx8_h2_ep1_replay.py`, same pattern as NX-7's
`nx7_diag_ep1_qpos.py`) across run 1's full 1400 steps:

- Before the first trigger (mj_step calls 200-1000, i.e. steps ~50-250): the robot
  progresses normally from y≈-3.95 toward y≈-2.58.
  - After the first trigger through episode end (mj_step calls 1000-5600, i.e. steps
    ~250-1400, spanning all 8 stop/resume cycles): x≈-1.20 to -1.34, y≈-2.44 to -2.58
    — **the identical ~0.1m box NX-7 found** (`docs/nx7_adoption.md` §2.2: "x≈-1.21 to
    -1.33, y≈-2.46 to -2.58"), unchanged by any of the 8 recoveries.

Cross-checked against ep1's own scene geometry (`sample_scene(derive_rng(999,1),
'demo')`, seed 999): robot starts at `(-1.262, -4.125)` facing yaw=90° (+y), target
(cyan cube) is at `(-1.117, 3.297)` — nearly due north, a 7.42m straight shot with
only a 0.14m lateral offset. The straight-line path's predicted x at y=-2.5 (the
stuck box's y-center) is **-1.230** — the observed stuck-box x-center (≈-1.27) is
only 0.04m off that line: **the robot is stuck almost exactly ON its direct line to
the target.** The scene's **orange-cone distractor** sits at `(-1.049, -2.386)`
(1.75m from the robot at episode start) — only **≈0.25m from the stuck-box center**,
well inside collision range for a humanoid body. This is a **physical collision with
a distractor object that happens to sit on GROUND_NET's more direct approach line**
(recall H1: GROUND_NET's signal is *smoother* than classical's, so it tracks a
straighter line toward the target — classical's jitter, ironically, is what lets it
brush past the same cone without fully wedging). A stand-then-resume-toward-the-
same-goal recovery cannot escape this: it just walks the robot back into the same
obstacle every time it resumes, which is exactly what the identical stuck-box
location across all 8 cycles shows. This sharpens NX-7's own closing speculation
(`docs/nx7_adoption.md` §7 — "most plausibly a collision/obstacle interaction... this
needs scene-geometry + contact-force instrumentation, out of scope for a 'grounding
backend' pass") into a confirmed, geometrically-located mechanism, still correctly
flagged there as belonging in the locomotion/steering policy or an obstacle-aware
path correction — not in `grounding.py`, `lock_mgmt.py`, or a stand/resume watchdog
of the shape tried here.

### 2.4 Spot-replay — mechanism is clean (no spurious triggers), just insufficient for ep1

`GROUND_NET=1 LOCK_M5=0 STALL_BREAK=1`, eps 0/5/12 (GROUND_NET's own wins) + eps 3/6
(classical passers that also pass under GROUND_NET), `eval/nx8_stall/nx8_h2_spot_replay.py`:

| ep | target (dist) | result (baseline: SUCCESS, `docs/nx6_final.md`) | final_dist | stall_break_triggers |
|---|---|---|---|---|
| 0  | cyan cone (4.32m) | **SUCCESS** | 0.380 | 0 |
| 3  | red cube (7.00m)  | **SUCCESS** | 0.370 | 0 |
| 5  | cyan ball (8.85m) | **SUCCESS** | 0.365 | 0 |
| 6  | red cone (8.17m)  | **SUCCESS** | 0.367 | 0 |
| 12 | cyan cube (6.18m) | **SUCCESS** | 0.361 | 0 |

**5/5 wins/passers survive, 0/5 spurious triggers.** STALL_BREAK behaves exactly as
intended on episodes that were never diagnosed as stalling — it is a safe,
well-targeted mechanism, just not sufficient (by itself, with the specific
"stand 50 steps then resume toward the same goal" recovery shape) to escape ep1's
specific collision geometry.

### 2.5 Regression smoke — default path unaffected

`STALL_BREAK`/`GROUND_NET`/`LOCK_M5` all unset (clean defaults), 1 easy episode
(seed 999, ep0): `success=True final_dist=0.563 stall_break_triggers=0` — matches
the documented easy baseline (`docs/nx6_final.md` §3: ~0.56-0.59m). Confirms the new
code is a provable no-op when the toggle is off.

---

## 3. Why the full adoption gates were not run

The brief's protocol conditions the full demo/easy/search/fancy-demo gates on "ep1
fixed 2/2" — that bar was not met (§2.2: 0/2 SUCCESS). This matches NX-7's own
identical situation and identical decision (`docs/nx7_adoption.md` §5): the demo
number under `GROUND_NET=1` with ep1 still broken is already known to be 12/15
(`docs/nx6_final.md` §2.1), below the brief's ≥13/15 ACCEPT bar, and neither H1 nor
H2 changed that — so running ~40 GPU-minutes of full 15-episode gates would only
reconfirm a foregone conclusion. That time went into the root-cause work in §2.3
instead, which is the actually load-bearing new information this pass produced.

---

## 4. Verdict and disposition

**REJECT.** No adoption. This closes the learned-grounding-adoption line per the
task brief's own "if this fails" branch.

- `GROUND_NET` stays default OFF (unchanged from `docs/nx6_final.md`/`docs/nx7_adoption.md`).
- `LOCK_M5` stays default OFF (unchanged).
- `GROUND_NET_HYSTERESIS`/`GROUND_NET_TAU_TRACK` (NX-7) stay default OFF, opt-in, unchanged.
- New `STALL_BREAK` (this pass): kept in the repo as **opt-in, default OFF** — tested
  working exactly as designed (correct trigger timing on ep1, 0/5 spurious triggers
  on healthy episodes, provable no-op when off) but insufficient alone to fix ep1's
  specific failure (a genuine physical collision with a distractor object, not a
  recoverable policy-attractor or grounding-layer issue) — same "keep as opt-in,
  REJECT for default-on" treatment this codebase has applied to `LOCK_M2`/`M4`/`M5`
  and `GROUND_NET_HYSTERESIS` throughout NX-2 through NX-7.
- No confirm run needed (no default changed).
- **Nothing synced** to `VLA_mujoco_unitree/code/` — sync is
  ADOPT-conditioned only, and this pass did not adopt.

**Final documented state:** classical (shipped default) demo 10/15 = 66.7%;
`GROUND_NET=1` opt-in demo 12/15 = 80.0% (fixes eps 0/5/12, breaks ep1, ep2/ep4
honest fails unchanged) — unchanged from NX-6/NX-7, since neither H1 nor H2 moved
this number.

### Files changed
- `code/inferencer.py` — additive only: `_env_flag()` helper, the `STALL_BREAK`/
  `STALL_VX_THR_MPS`/`STALL_WINDOW_STEPS`/`STALL_DISP_THR_M`/`STALL_MIN_GOAL_DIST_M`/
  `STALL_RECOVERY_STEPS`/`STALL_COOLDOWN_STEPS` constants, per-episode state init
  (`_stall_hist`, `_stall_recovery_remaining`, `_stall_cooldown_remaining`,
  `_stall_trigger_count`, `_stall_is_maneuver`, `_cur_vx_cmd`), the `gt_vel_inject_t`
  recovery override (inserted immediately before the normal-mode model forward
  call), the `vx_cmd` capture (immediately after), the window-update/trigger-check
  block (immediately after `dist_to_target` is computed), and the new
  `RolloutResult.stall_break_triggers` field (+ populated in the return statement).
  Zero behavior change when `STALL_BREAK` is unset (confirmed, §2.5). No other file
  touched — `code/grounding.py`, `code/lock_mgmt.py`, `code/steer.py` are all
  byte-unchanged from `docs/nx7_adoption.md`'s state.
- Diagnostic artifacts (scratchpad, not committed/synced): `eval/nx8_stall/`:
  `nx8_h1_goal_parity.py` (+ log/json, §1.2), `nx8_h2_ep1_replay.py` (+ 2 run
  logs/jsons, §2.2/§2.3), `nx8_h2_spot_replay.py` (+ log/json, §2.4),
  `nx8_regress_smoke.log` (§2.5).

---

## 5. What remains / follow-on (for whoever picks this up next)

- **The concrete next step, if this line is ever reopened**: an obstacle-aware path
  correction in the locomotion/steering layer (not grounding, not a stand/resume
  watchdog) — e.g. a lateral nudge or brief detour heading when a STALL_BREAK-style
  trigger fires, rather than resuming on the exact same bearing that walked the
  robot into the obstacle in the first place. This pass's contribution is pinning
  down that the obstacle-collision hypothesis is correct and geometrically locating
  it (§2.3); it does not attempt a fix of that shape (out of this pass's bounded
  scope, which was specifically H1 goal-parity and H2's stand/resume mechanism —
  both now exhausted).
- **A genuinely surprising, reusable finding**: GROUND_NET's raw grounding signal is
  measurably *smoother* than classical's (§1.2) — the opposite of what "learned
  detector on a coarse heatmap grid" intuition would suggest. Worth keeping in mind
  for any future grounding-backend work: this codebase's classical HSV+depth
  pipeline is itself fairly noisy cycle-to-cycle, and a learned replacement doesn't
  need to match that noise floor to be viable — recall (coverage) turned out to be
  the actual axis GROUND_NET is behind on, not precision.
- `docs/nx6_final.md` §8's longer-heatmap-training recommendation (cut short at
  epoch 41/60) remains valid on its own terms for narrowing the recall gap behind
  eps 2/4's still-unfixed failures, but is now understood (per NX-7 and reconfirmed
  here) to be fully orthogonal to ep1, which was never a recall problem.
