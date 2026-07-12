# NX-11 — Final bounded cycle on demo ep4: fall diagnosis, one fix attempt, CLOSE

**Date:** 2026-07-09/10
**Agent:** NX-11 (follow-on to `docs/nx10_scan_fix.md`, `docs/fa2_residuals.md`)
**Starting state:** demo 14/15 (93.3%, `docs/nx10_scan_fix.md` §4.1) — ep4 (purple
ball, 7.21m, spawn yaw 180°, target bearing +62.6°) is the sole remaining failure,
now via a **fall** during the final close-range approach (fd=1.37-1.51 at the
shipped `MAXSTEPS['demo']=1700` cap), a materially different failure mode than the
"never detected" story NX-10 fixed for ep2.

## TL;DR — VERDICT: CLOSE (demo stays at 14/15 = 93.3%, no code changes retained)

Two mechanisms were found by instrumented replay:

1. **A real, previously-undocumented AVOID bug (found, fixed, mechanism-confirmed,
   but NOT ep4's dominant cause — fix reverted).** AVOID's depth-based obstacle
   detector has no self-body exclusion. The PROXIMITY camera (58° steep pitch,
   chest-mounted) captures the robot's own raised/swinging arms during
   locomotion at very close range (0.24-0.43m, ~0.8m above the ground —
   nowhere near the floor cut or the target's exemption bearing) and
   misclassifies them as an external obstacle, injecting a large, fully
   one-sided yaw bias (up to +0.18 rad/s) for ~8 consecutive grounding cycles
   right in the `[CAM_D_LO, CAM_D_HI] = [1.2, 1.6]`m camera-handoff band —
   **visually confirmed** via saved RGB+depth frames (the flagged "obstacle"
   pixels are literally the robot's own hand in frame). This is exactly the
   "AVOID vs. proximity-handoff" interaction the task brief predicted. The
   principled fix (raise `AVOID_MIN_GOAL_DIST_M` 1.2→1.6, aligning the
   carve-out with `CAM_D_HI`) was implemented and **mechanism-confirmed to
   fully suppress the bug** (`avoid_bias_active_frac=0.000` both runs) — but
   ep4 **still failed via a fall, 2/2**, so per the task's "flip 2/2 or
   revert" bound, the fix was reverted; `code/avoid.py` is back to its exact
   NX-9 state (1.2, unit self-test 15/15 unchanged).
2. **ep4's actual dominant failure mechanism, independent of AVOID**: a
   late-episode balance loss after an extended (~150-400 step) period where
   the robot's own bearing-to-target estimate gets "stuck" 20-40° off-center
   while the PROXIMITY camera holds a consistent, high-confidence detection
   and the range oscillates in a tight 0.4-1.0m band — i.e. the robot
   circles/orbits close to the target instead of converging. This state is
   reproduced **identically with `AVOID=0`** (whole mechanism disabled) and
   **with the AVOID fix applied** (bias measurably zero throughout) —
   confirming AVOID is not required to reproduce it. Height/joint-velocity
   traces rule out gradual gait fatigue (pelvis height is flat at
   0.742-0.745m for the first ~1500 steps, zero degrading trend); the fall
   itself is a sudden, ~15-20-step event (an accelerating yaw-rate ramp to
   1.5-2.4 rad/s coincident with a monotonic height collapse through
   `FALL_HEIGHT=0.50`) that always occurs 350-450 steps *after* the camera
   handoff, ruling out "a velocity-command discontinuity at the handoff" as
   the proximate trigger. This is a policy-level locomotion-stability limit
   under sustained close-range misalignment — out of scope (retraining
   banned) and not addressable by the task's permitted handoff-local
   command-smoothing fallback, since the trace does not support "at the
   handoff" as the mechanism (see §5).

No code changes were retained. `docs/nx10_scan_fix.md`'s 14/15 (93.3%) stands as
final. No adoption, no sync (nothing to sync — clean revert).

---

## 1. Setup

Instrumented single-episode replays of demo ep4 (`derive_rng(999, 4)`,
`sample_scene(rng, 'demo')`), `checkpoint/goto_best.pt`, arch=A, device=cuda,
pure defaults (`goal_source='classical'`, `vel_source='predicted'`, `GROUND_NET=1`,
`AVOID=1` both default-on unless noted), `maxsteps=1700` (NX-10's shipped cap).
Monkeypatches (scratchpad-only, no `code/` changes during diagnosis):
`code.inferencer._build_proprio` (once/step on every code path — logs
x,y,yaw,height,joint-qvel-RMS,base-angular-velocity-norm,base-linear-velocity-norm),
`code.inferencer.classical_ground` (per grounding cycle — visible/dist/bearing/conf/
`is_proximity`, i.e. which camera), `code.avoid.compute_obstacle_bias` and
`code.avoid.biased_vel_cmd` (every AVOID call — inputs, carve-out state, returned
bias/velocity), `code.lock_mgmt.LockGate.mark_discontinuity` (Schmitt-flip/rescan
markers). Scripts: scratchpad `nx11_replay_ep4.py` (general instrumented replay),
`nx11_avoid_selfbody_check.py` (targeted per-step obstacle-mask spatial/depth dump +
RGB/depth frame capture, §3).

Scene geometry (`derive_rng(999,4)`, `sample_scene(rng,'demo')`): robot spawn
(4.125, 1.883) yaw=180°; target purple ball (0.808, -4.514), dist=7.21m; 4
distractors — cyan ball (-0.580,2.994), orange cube (2.340,4.641), blue cylinder
(1.199,0.897), red cone (4.604,0.909), all size 0.22-0.26m.

---

## 2. Baseline replay (pure defaults, `AVOID=1`, unmodified NX-10 code)

```
[scan] ALIGNED at step=90  (unchanged from docs/nx10_scan_fix.md)
=== RESULT ep4: success=False tag=fall steps=1470 final_dist=1.332 fell=True
    avoid_bias_active_frac=0.065 ===
```

Camera handoff: `is_proximity` flips False→True at step=1221 (goal_dist≈1.40-1.49,
via the bounded-fallback-probe adoption path — GROUNDING started missing around
this range, not strictly the `_ema_dist<CAM_D_LO=1.2` Schmitt condition, which
widens the effective PROXIMITY-active window somewhat above the nominal 1.2m).

**AVOID bias trace, steps 1231-1311** (goal_dist 1.2-1.6m band, `carved_out=False`
throughout — the old 1.2m cutoff had not yet engaged):

```
step=1231 goal_dist=1.442 bias: 0.000 -> +0.042  n_px=2599  L=0.520 R=0.840  imb=-0.235
step=1251 goal_dist=1.392 bias: -0.019 -> +0.035 n_px=2330  L=0.520 R=0.840  imb=-0.235
step=1261 goal_dist=1.378 bias: +0.035 -> +0.151 n_px=2185  L=0.000 R=0.760  imb=-1.000
step=1271 goal_dist=1.367 bias: +0.151 -> +0.154 n_px= 753  L=0.000 R=0.520  imb=-1.000
step=1281 goal_dist=1.346 bias: +0.154 -> +0.170 n_px= 762  L=0.000 R=0.600  imb=-1.000
step=1291 goal_dist=1.314 bias: +0.170 -> +0.169 n_px=2600  L=0.120 R=0.680  imb=-0.700
step=1301 goal_dist=1.265 bias: +0.169 -> +0.042 n_px=2696  L=0.360 R=0.360  imb=+0.200
step=1311 goal_dist=1.211 bias: +0.042 -> -0.084 n_px=2421  L=0.600 R=0.040  imb=+0.875
step=1321 goal_dist=1.163  carved_out=True -> bias=0.000  (stays 0.000 for the
    remaining ~150 steps up to the fall, confirmed every logged cycle to step 1461)
```

**Geometric cross-check** (robot x,y,yaw from the step log vs. all 4 distractor
world positions, at every AVOID-active step 1171-1470): every distractor's true
egocentric bearing is >100° off the ±25° corridor at every relevant step (e.g. at
step 1261: cyan ball -147.8°, orange cube -171.7°, blue cylinder -166.0°, red cone
+156.4° — all far behind/beside the robot, none in front). **No scene object could
be the source of the L=0.000/R=0.6-0.84 signal.** During this same window the
robot's own bearing to the true target swings from +35° through 0° to -18° while
its yaw itself is turning fast (-126.7°→-80.9°, a 46° swing in 80 steps) — i.e. a
fast, sustained in-progress turn, not a static pose.

Meanwhile bearing (from `classical_ground`, i.e. the target's own tracked
bearing) crosses zero right inside this window and continues past it to -18° to
-28° by step 1341-1461, and never re-centers — the goal_dist trace itself
**reverses** (0.578m at step1401 → 0.980m at step1461, moving *away*) before the
fall. The fall itself (last 60 logged steps): pelvis height flat ~0.717-0.734m
through step 1440, a brief stumble/recovery at 1420-1425 (h dips to 0.706, partial
recovery), then from step ~1450 a monotonic, accelerating collapse (h: 0.716 →
0.509 over 20 steps, base linear-velocity norm climbing to 1.78 m/s, base
angular-velocity norm to 2.45 rad/s) — a genuine topple in progress, ending the
episode via `FALL_HEIGHT=0.50` at step 1470.

---

## 3. Root-causing the AVOID activity: self-body contamination, visually confirmed

`compute_obstacle_bias` was re-run standalone (targeted script,
`nx11_avoid_selfbody_check.py`) at steps 1251/1261/1271/1281/1291 to dump the
obstacle-mask's spatial/depth statistics directly:

```
step=1261 goal_dist=1.300 bias=+0.157 n_obs=2179 is_prox=True
  obstacle px row(v): [104-201] of H=240 (mean=163, 68% down the frame)
  obstacle px dist: min=0.247 max=0.428 mean=0.317 m
  obstacle px height_above_ground: min=0.788 max=0.875 mean=0.848 m
  obstacle px bearing_deg: -25.0 to -5.4
```

At **0.25-0.43m** range and **~0.8m above the ground**, this is nowhere near the
floor (`AVOID_FLOOR_MARGIN_M=0.10` cut correctly excludes it) and nowhere near any
scene object (§2). Saved RGB + depth frames (`nx11_frames/rgb_step1261.png`,
`obsmask_step1261.png`) **visually confirm** the mechanism directly: the
PROXIMITY-camera frame shows the robot's own two arms/hands raised in front of the
camera (a G1 running/turning-gait arm-swing posture) with the purple ball visible
in the upper-left; the green obstacle-mask overlay lands exactly on the robot's own
right-side arm/hand.

**Mechanism**: `code/avoid.py`'s `_backproject_frame`/`compute_obstacle_bias` has
two exclusions — a floor-height cut and a target-bearing exemption — and **no
self-body exclusion**. `code/grounding.py`'s classical detector already has a
documented "stricter self-body-rejection" check gated on `is_proximity`
(`docs/cam_opt2_multicam.md` "Real risk", `docs/cam_p0.md` ep14 finding) — but that
logic lives only in the color+depth blob detector, not in AVOID's independent
depth-back-projection path, which was never given an equivalent check. The
PROXIMITY camera's steep 58° pitch, mounted close to the robot's chest
(`CAM_ROBOT_FORWARD_OFFSET_M=0.10`), puts a swinging arm/hand well inside AVOID's
`AVOID_NEAR_M=2.0` window at depths that immediately saturate severity
(`AVOID_MIN_DEPTH_FOR_WEIGHT_M=1.0`), producing the observed large, fully
one-sided bias.

This is precisely the "AVOID vs. proximity-handoff, two systems fighting"
interaction the task brief hypothesized — refined: it is not the *target* being
misattributed as an obstacle, and not a *real* distractor either, but the robot's
own body, only visible once the PROXIMITY camera activates.

---

## 4. Fix attempt: align `AVOID_MIN_GOAL_DIST_M` with `CAM_D_HI` — mechanism-confirmed, but does not flip ep4 (2/2 FAIL)

**Change** (one constants revision, `code/avoid.py`): `AVOID_MIN_GOAL_DIST_M` 1.2 →
1.6, matching `inferencer.py`'s `CAM_D_HI=1.6` (the Schmitt threshold at which
PROXIMITY is guaranteed to revert to GROUNDING) — carves AVOID out for the entire
band where PROXIMITY could plausibly be active, closing the self-body-contamination
window at its source (endgame carve-out widening, not a geometric self-body-mask
fix — no per-pixel exclusion was added to `_backproject_frame`, out of scope for a
one-constant revision).

**Mechanism-level replay, 2x** (per protocol, before considering full gates):

| run | steps | failure_tag | final_dist | avoid_bias_active_frac |
|---|---|---|---|---|
| fix run 1 | 1589 | fall | 1.024 | **0.000** |
| fix run 2 | 1617 | fall | 1.013 | **0.000** |

The fix works exactly as intended — `avoid_bias_active_frac=0.000` both runs
confirms AVOID never fires anywhere in the episode (the self-body bug is fully
suppressed) — but **ep4 still fails via a fall, 2/2**. A third control run with
`AVOID=0` (the whole mechanism disabled outright, independent confirmation) also
fails via a fall (step 1638, final_dist=0.974) with the same late, sudden,
accelerating-rotation collapse signature. **AVOID — buggy or fixed or fully
disabled — is not ep4's dominant failure mechanism.**

Per the task's bound ("FULL GATES only if ep4 flips 2/2 ... Else revert cleanly,
keep defaults, honest doc"): **reverted**. `code/avoid.py`'s
`AVOID_MIN_GOAL_DIST_M` is back to its exact NX-9/NX-10 value (1.2); the module's
own unit self-test (`python -m code.avoid`) reconfirmed **15/15 PASS**, matching
`docs/nx9_avoid.md` §2 byte-for-byte. No full-gate run was performed (not earned —
the 2/2 flip bar was not met, so per protocol the expensive n=15×3 gate suite was
correctly skipped). The self-body bug and its (reverted) fix are documented here
for any future agent who revisits AVOID/PROXIMITY-camera interactions; the fix
itself is written up in `code/avoid.py`'s own comment at the `AVOID_MIN_GOAL_DIST_M`
constant (marked "tried and REVERTED", with a pointer to this doc) so it is not
silently rediscovered from scratch. Spot-replay protection for NX-9's demo/search
avoidance wins (§6) was therefore not exercised as a live gate (the code is back to
its already-gated NX-9 state), though the geometric case for why they'd have been
safe is noted there for completeness.

---

## 5. The actual residual mechanism: late-episode balance loss, not fatigue, not a handoff discontinuity

With AVOID either fixed-inert or fully off, `classical_ground`'s own bearing trace
in the ~150-400 steps before the fall (both control conditions) shows the same
qualitative shape: the target stays reliably detected (`vis=True`, confidence
0.6-0.95, i.e. this is not a detection problem) but the reported bearing sits
**stuck 20-40° off-center** for 100+ consecutive grounding cycles while range
oscillates in a tight 0.4-1.0m band — the robot is circling/orbiting near the
target rather than converging its heading, well past what a single scan/handoff
event would produce.

**Ruled out — gradual gait fatigue.** Bucketing the fixed-code run's full
step-level joint-velocity-RMS/height trace in 100-step windows (`nx11_ep4_fix_run1.json`):
pelvis height is flat at 0.742-0.745m and joint-qvel-RMS fluctuates in a normal
0.55-0.75 band with **no degrading trend** for the entire first ~1500 steps (15
consecutive 100-step buckets). The fall is not a slow decline — it is a sudden
event confined to the *last* ~100-step bucket (h_min crashes from 0.740 to 0.508
inside that single window), immediately preceded by an accelerating yaw-rate ramp
(ang-vel-norm 0.4→2.4 rad/s) over just ~15-20 steps.

**Ruled out — a velocity-command discontinuity at the handoff.** The camera
handoff (`is_proximity` False→True) occurs around step 1200-1220; the fall occurs
at step 1470-1638 depending on run — **350-450 steps later**, well outside any
plausible reach of a single-cycle command discontinuity. The task's permitted
fallback fix for this branch ("bounded velocity-command slew at the handoff cycle")
is not trace-supported here — there is no temporal proximity between the handoff
event and the fall event to smooth over.

**Best-supported characterization**: a policy-level locomotion-stability limit
under sustained close-range misalignment. ep4 is structurally unusual among the 15
demo episodes (180° spawn-yaw turn + a 7.21m walk + the largest post-scan bearing
of any passing episode), so it plausibly drives the trained policy into a
combination — long PROXIMITY-camera dwell time, persistently large bearing error,
close range, high existing forward momentum — that is underrepresented in
training data. The eventual fast, committed rotation the policy attempts (or
drifts into) while still carrying forward momentum at close range is what
destabilizes it. This is a genuine locomotion-policy limitation, not a scan,
grounding, AVOID, or camera-handoff bug — **retraining is out of scope**, and no
in-scope, trace-supported cheap fix was identified this cycle.

---

## 6. Spot-replay protection (context only — not exercised as a live gate, §4)

The reverted fix would have carved AVOID out below 1.6m instead of 1.2m. For
completeness: NX-9's two documented avoidance wins were demo ep1 (orange-cone
wedge, ~0.25m off a straight-line path at a working distance well above 1.6m —
`docs/nx8_stall.md` §2.3) and search ep12 (a distractor 0.92m along the *approach*
path, i.e. encountered while still closing distance, not at proximity endgame —
`docs/nx1_scan.md`). Both are mid-path collisions at distances the 1.6m carve-out
would not have touched (AVOID remains fully live above 1.6m under either cutoff
value) — geometrically these wins were not at risk from a 1.2→1.6 change. This
was not empirically re-verified via replay since the code was reverted before
reaching the full-gate stage; noted here only so a future revisit of this exact
fix does not need to re-derive the argument from scratch.

---

## 7. Final state

- `code/avoid.py`: `AVOID_MIN_GOAL_DIST_M` unchanged at 1.2 (byte-equivalent
  behavior to `docs/nx10_scan_fix.md`'s shipped state — the constant's value did
  not change; only its comment gained a paragraph documenting the tried-and-reverted
  1.6 revision, per-file diagnostic, no behavior change). Unit self-test 15/15
  PASS reconfirmed.
- No other files touched (`code/inferencer.py`, `code/eval_closedloop.py`,
  `code/demo.py` — all untouched this cycle).
- No full gates run (bound not met — protocol correctly skipped them).
- No sync to `VLA_mujoco_unitree/code/` (nothing changed to sync).
- **demo stays at 14/15 = 93.3%** (`docs/nx10_scan_fix.md` §4.1), easy 15/15,
  search 15/15 — all unchanged, all still the last-gated NX-10 state.
- ep4 is CLOSED for this line of work: two real, now-documented findings (a fixed
  self-body/AVOID interaction bug, reverted for lack of net benefit; and a
  characterized-but-out-of-scope policy-level close-range balance limit) rather
  than an open question. Any future attempt should start from §5, not re-litigate
  §2-4.
