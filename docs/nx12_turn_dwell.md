# NX-12 — Rotation-Order Instability on Large Post-Locate Realignment Turns

**Date:** 2026-07-09/10
**Agent:** NX-12 (follow-on to `docs/gen1_multiseed.md` §3.1, `docs/nx10_scan_fix.md` §6,
`docs/nx1_scan.md`, `docs/nx11_ep4.md`)
**Starting state:** demo 14/15 (93.3%, seed 999, `docs/nx10_scan_fix.md`). GEN-1's
fresh-seed multi-seed validation found the dominant residual: two byte-identical
falls at seed 1000 (ep12) and seed 2000 (ep2), both `spawn_yaw=180°` + a
large-magnitude negative target bearing, a trigger combination seed 999's own
15-episode demo set never happened to sample.

## TL;DR — VERDICT: CLOSE (mechanism NOT fixed; code reverted to byte-identical
prior behavior; no adoption, no sync)

| check | bar | result |
|---|---|---|
| Mechanism-level: demo1000-ep12, demo2000-ep2 falls eliminated? | fall→success on replay | **NO — fall persists at 2 tested parameter points** (delayed 256→314→553 steps, never eliminated) |
| ep4 (seed999) — same family? | honest report either way | **NOT the same family** — independently reconfirmed nx11's own late/close-range characterization; TURN_DWELL is a no-op for ep4 |
| Passer/regression spot-checks | no timing regression | **New margin-erosion risk found**: ep9 (999) drops from ~150-190 spare steps to 47 spare steps under the 1700-step cap |
| Full n=15×5 gates | run only if mechanism bar met | **not run — bar not met** (matches `docs/nx11_ep4.md`'s own "not earned" precedent) |
| Adoption | default-on + sync only on full pass | **not adopted**; `code/scan_sched.py` reverted to its exact prior behavior (re-replay confirms byte-identical fall trace); no sync |

The task's mechanism hypothesis ("after the scan locates the target, one long
continuous realignment turn") does not match what mechanism-level replay actually
shows for these two episodes: **the scan never exits and the target is never
detected in either failing episode** — the fall happens entirely *inside* the H3
scan's own leg0, precisely at the leg0→dwell transition, before any target-specific
signal ever enters the control loop (proven by byte-identical trajectories across
two different targets at two different seeds — see §2). The assigned fix family
(bound continuous turn segments with brief dwells, generalizing
`BidirectionalScanSchedule`'s existing leg/dwell mechanic to a finer grain) was
implemented faithfully and is exactly the right *shape* of fix for a continuous-
rotation-length problem — but the underlying issue here is not continuous-rotation
length (the existing 90°/~208-step leg0 is already far under the documented
~470-step/323° OOD ceiling). Two tested parameterizations (spanning a 4x range of
chunk size) both delayed the fall without preventing it, and evidence points
instead at a spawn/heading-specific **translational-drift-during-in-place-turning**
instability — see §3.

---

## 1. Mechanism-level baseline replay — reproducing GEN-1's byte-identical falls

Instrumented replay (`checkpoint/goto_best.pt`, pure defaults, device=cuda,
`code.inferencer._build_proprio` monkeypatch logging x/y/z/yaw per step, plus a
`code.scan_sched.BidirectionalScanSchedule.step` wrapper exposing leg_idx/dwelling
state — not otherwise observable from outside `rollout()`). Scratchpad-only, no
`code/` changes during this phase.

```
=== demo seed=1000 ep=12: FAIL[fall] steps=256 final_dist=7.143 ===
=== demo seed=2000 ep=2:  FAIL[fall] steps=256 final_dist=6.071 ===
```

Both scenes: `robot_xy=(4.125, y)`, `robot_yaw=180°` (`side==1` spawn in
`code/scene.py` — "near +X wall, face -X"; `x=4.125` and `yaw=π` are **exact** for
every `side==1` sample, 1-of-4 spawn conditions, not a literal wall-collision
setup — arena half-extent is 5.5m, so nominal clearance is 1.375m). Target bearings
-49.4° (ep12) and -74.0° (ep2) — both large-magnitude negative, i.e. "wrong
direction first" for the scan's fixed `_LEG_SIGNS=(+1,-1,-1,+1)` "always try
positive first" leg0.

**x/yaw traces are identical to the reported precision at every logged step**
(only y differs, matching the different target y-coordinates — target position has
zero causal effect on this trajectory):

```
step=237 x=+4.119 yaw=-94.87   step=249 x=+4.207 yaw=-101.74
step=239 x=+4.130 yaw=-95.84   step=251 x=+4.227 yaw=-103.62
step=241 x=+4.142 yaw=-96.88   step=253 x=+4.248 yaw=-105.81
step=243 x=+4.156 yaw=-97.90   step=255 x=+4.268 yaw=-108.70
step=245 x=+4.171 yaw=-98.96
step=247 x=+4.189 yaw=-100.19
```
(both episodes, digit-for-digit)

Scan-schedule internal trace (leg_idx / dwelling / accumulated yaw):

```
scanstep~  1  leg_idx=0 dwelling=False accum_yaw=  +0.00
scanstep~208  leg_idx=0 dwelling=True  accum_yaw= +90.12   <- leg0 (CCW) completes, big dwell begins
scanstep~253  leg_idx=1 dwelling=False accum_yaw= +74.19   <- dwell ends (45 steps), leg1 (CW) begins
scan ended (episode fell) at proprio-step~256
```

**Reading**: leg0 (a single continuous ~90.12° CCW rotation, realized in ~208
steps — already well inside the documented ~470-step/323° in-distribution
ceiling) completes and the 45-step leg-boundary dwell begins (commanded wz=0.0,
i.e. "stand still", exactly the in-distribution behavior `docs/nx1_scan.md`
validated). The height trace is flat/aligned through step ~230 (z≈0.70), then
**gradually collapses (0.703→0.542) while yaw keeps drifting (-94.87°→-108.70°,
an *uncommanded* additional -14° during what should be a zero-wz stand-still)**
— a genuine physical topple in progress, ~30-50 steps into the dwell, i.e. after
active rotation commands have already stopped. This is the qualitative "sudden
accelerating rotation coincident with height collapse" signature `docs/nx11_ep4.md`
§5 characterized for ep4's own fall, but occurring at the scan's own leg0/dwell
boundary rather than late/close-range.

**This corrects the task's framing**: neither episode's scan ever exits
(`_scan_active` never flips False — no detection occurs before the fall), so this
is not a "post-locate turn-toward-goal" event. "Scan-handoff" in
`docs/gen1_multiseed.md`'s own language means the leg0→leg1 (dwell) transition
internal to the scan, not a post-detection handoff to the model's own predicted-
velocity control. The fix therefore correctly targets `code/scan_sched.py`'s
shared `BidirectionalScanSchedule` (per the task brief's own permitted target),
not `code/steer.py` (which is not on the causal path for either failing episode —
confirmed by reading `code/inferencer.py`: with `goal_source='classical'`,
`vel_source='predicted'` [both defaults], the model's own predicted velocity head
drives all post-scan turning; `steer.py`'s explicit control law is only reachable
via `vel_source='gt'`, the `learned`-grounding velocity replica, or AVOID's
`biased_vel_cmd` — none active in either failing episode, which never leave scan
mode at all).

---

## 2. Fix implementation: intra-leg forced sub-dwell

Extended `BidirectionalScanSchedule` (`code/scan_sched.py`), opt-in via
`TURN_DWELL=1` (module `_env_flag`, matching this codebase's `AVOID`/`GROUND_NET`/
`STALL_BREAK` convention): inserts a brief forced stand-still sub-dwell every
`TURN_DWELL_MAX_CONT_DEG` degrees of **continuous** same-direction rotation
(tracked the same way leg completion is — real accumulated yaw, not assumed
step-rate), for `TURN_DWELL_SUB_STEPS` steps, checked *before* the leg-completion
check each step so a leg's own bigger boundary dwell always takes priority at the
exact moment it completes. This does not change a leg's total angular reach (H3's
±118.9° effective coverage is unaffected), only how finely its continuous-rotation
segments are chunked — exactly the assigned fix family, generalized to a finer
grain, and exactly the class of change `docs/nx10_scan_fix.md` §6 and
`docs/gen1_multiseed.md` §6 pre-flagged as the natural next step.

Because `ReacquisitionScan` (`code/lock_mgmt.py`), `eval_search.py`, and
`fancy_demo.py` all construct `BidirectionalScanSchedule(...)` without overriding
the new parameter, the toggle is automatically shared by every consumer (H3's
initial scan, any M4/M5/M7-triggered rescan, and the search skill's own initial
scan) — satisfying "implement in the shared steering/turn path so ALL consumers
get it" without a second divergent copy.

Verified correct in isolation first (a synthetic perfect-tracking simulation,
`leg_deg=90`, `turn_dwell_max_deg=45`): sub-dwell fires at accum≈45.4°, leg
completes at accum≈90.1° as before, leg1 gets its own sub-dwell at accum≈44.7° —
matches the intended chunking exactly before any physics rollout was run.

---

## 3. Mechanism-level validation of the fix — falls NOT eliminated

### 3.1 Default parameters (`TURN_DWELL_MAX_CONT_DEG=45`, `TURN_DWELL_SUB_STEPS=40`)

```
demo seed=1000 ep=12: FAIL[fall] steps=314 (was 256) final_dist=7.497
demo seed=2000 ep=2:  FAIL[fall] steps=314 (was 256) final_dist=6.353
```

Again byte-identical to each other (x/yaw traces match digit-for-digit; only y
differs) — the fix does not break the target-independence property, it just
shifts *when* the same underlying event happens. Scan-schedule trace confirms the
sub-dwells fire correctly:

```
scanstep~104  subdwelling=True  accum_yaw=+45.10   <- 1st sub-dwell (at the 45° cap)
scanstep~144  subdwelling=False accum_yaw=+41.81    (drifted -3.3° DURING the "stand still")
scanstep~246  subdwelling=True  accum_yaw=+86.96   <- 2nd sub-dwell (leg0 nearly complete)
scanstep~286  subdwelling=False accum_yaw=+83.39    (drifted -3.6° DURING the "stand still")
[fall follows almost immediately after the 2nd sub-dwell ends and rotation resumes
 toward leg0's original ~90° completion point]
```

Two findings: (a) the robot is **not perfectly stationary during the sub-dwells**
despite wz=0 commanded — small (~3-4°) backward yaw drift each time, consistent
with an already-destabilizing state rather than a clean recoverable pause; (b) the
fall still lands almost exactly where the *original, unmodified* leg0 would have
completed (~90° cumulative), just later in wall-clock time — i.e. chunking the
same total rotation into smaller dwell-separated pieces did not prevent the
instability, it only delayed reaching the same cumulative-rotation/elapsed-
exposure state.

### 3.2 Aggressive parameters (`TURN_DWELL_MAX_CONT_DEG=20`, `TURN_DWELL_SUB_STEPS=80`, env-override, no code change)

```
demo seed=1000 ep=12: FAIL[fall] steps=553 (2.2x baseline) final_dist=7.701
```

Still falls — now at a **lower** cumulative rotation (~50-60° instead of ~90°),
after 4 sub-dwell cycles. Critically, the robot's x-position had drifted from
4.125 to **5.011** by the time of the fall (arena +X wall face at 5.45m — only
0.44m clearance remaining, down from the spawn's 1.375m). Backward yaw-drift
during each of the 4 sub-dwells is not the only symptom: **net translational
drift accumulates monotonically with elapsed scan time**, and giving the robot
*more* stand-still recovery time (more, longer sub-dwells) gave it more time to
drift toward the wall behind its spawn, not less risk. This is the opposite of
what the fix intended.

### 3.3 Conclusion

Across 2 tested parameterizations spanning a 4x range of chunk size, the fix
**delays but never eliminates** either target fall. The original 90°/~208-step
leg0 is already comfortably inside the documented ~470-step/323° in-distribution
continuous-rotation ceiling, so "a single continuous segment is too long" is not
the operative mechanism here — the evidence instead points at a **spawn/heading-
specific translational-drift-during-in-place-turning instability** (backward
drift toward the wall behind a `side==1` spawn, compounding with elapsed
scan-phase duration) that stand-still dwelling does not arrest and may worsen by
extending exposure time. This is a genuinely different failure shape than the one
the assigned fix family (continuous-rotation-length bounding) was designed to
address, even though it manifests through the same rotation/scan machinery.

---

## 4. ep4 (seed999) — same family? Honest answer: NO

Per the task's explicit request to check honestly either way.

**Baseline (no fix), 3 replays** (pure defaults, same instrumentation):

| run | result | steps | final_dist |
|---|---|---|---|
| 1 | SUCCESS | 1538 | 0.392 |
| 2 | FAIL[fall] | 1501 | 1.400 |
| 3 | FAIL[fall] | 1465 | 1.278 |

Consistent with `docs/nx11_ep4.md`'s own documented ~1470-1650-step fall band and
its explicit characterization of ep4 as a run-to-run-fragile episode (the
codebase's well-documented EGL/physics jitter) — not new
information on its own, but confirms this replay setup reproduces nx11's own
findings faithfully.

**With `TURN_DWELL=1` (default params), 2 replays:**

| run | result | steps | final_dist |
|---|---|---|---|
| 1 | FAIL[fall] | 1527 | 1.147 |
| 2 | FAIL[fall] | 1550 | 0.839 |

Statistically indistinguishable from the baseline noise band — **TURN_DWELL is a
provable no-op for ep4**: its target bearing (+62.6°, positive) is found directly
in leg0 at scanstep~90 (`[scan] ALIGNED at step=90`, matching `docs/nx10_scan_fix.md`
exactly), which is only ~39° of the schedule's own accumulated rotation — well
under the 45° sub-dwell threshold, so the sub-dwell logic never engages at all.

**Verdict: NOT the same family.** ep4's fall (when it occurs) happens 1465-1650
steps in — 1200+ steps after the scan exits and well after the PROXIMITY-camera
handoff (~step 1200-1220) — with the signature `docs/nx11_ep4.md` §5 already
characterized in detail: target reliably detected, bearing stuck 20-40°
off-center, tight-range circling, then a sudden late accelerating-yaw-rate/
height-collapse event. This is entirely disjoint in timing (1465+ vs 256 steps),
trigger (post-detection close-range circling vs pre-detection scan-leg dwell),
and causal chain (no target/detection involvement at all in the ep12/ep2
mechanism, vs. a target-tracking oscillation being central to ep4's) from the
gen1 ep12/ep2 mechanism. This independently reconfirms nx11's own conclusion
(an out-of-scope, policy-level close-range balance limit) rather than uncovering
a shared root cause — an honest negative result, not a new lead.

---

## 5. Passer / regression spot-checks (seed 999) — a new risk found, not just "no regression"

| ep | bearing | scan path | baseline | `TURN_DWELL=1` | note |
|---|---|---|---|---|---|
| 2 | -73.8° | leg0(full)+dwell+leg1(full)+dwell+leg2(partial) — the long-scan control | SUCCESS steps=1154, fd=0.364 | SUCCESS steps=1271, fd=0.370 | holds; +117 steps overhead (2 sub-dwells fired); margin under the 1700 cap narrows 546→429 |
| 0 | +59.5° | fast leg0 find | SUCCESS steps=580-614 | SUCCESS steps=650 | holds; small overhead |
| 9 | -39.7° | leg0(full)+dwell+leg1(full)+dwell+leg2(partial) | SUCCESS steps=1507 (nx10: 1510/1547) | SUCCESS steps=**1653** | **holds, but margin under the 1700 cap collapses from ~150-190 spare steps to 47** |

ep9 is the most informative result here: it is a previously rock-solid passer
(SUCCESS in every prior gate, `docs/nx10_scan_fix.md` §3.4/§4.1) that, under the
fix, still succeeds but now sits within a hair of `MAXSTEPS['demo']=1700`. Given
this codebase's own well-documented run-to-run realized-rotation-rate lag (up to
~1.2-1.3x nominal, `docs/nx1_scan.md`) and general EGL/physics jitter
(per an earlier reproduction log), a 47-step margin is not a safe buffer — a full n=15 gate
would plausibly have flipped this to a new `didnt-reach` regression on top of not
fixing the target mechanism. This is exactly the kind of cost a "bound continuous
rotation via more dwells" fix imposes: every additional intra-leg dwell is pure
step-budget overhead on scenes that already need most of the scan schedule, with
no offsetting benefit demonstrated in §3.

---

## 6. Full n=15×5 gates — not run (bar not met)

Per the task's own protocol ("FULL GATES only if [the mechanism-level check]
flips... Else revert cleanly, keep defaults, honest doc") and this exact
codebase's established precedent for the identical situation
(`docs/nx11_ep4.md` §4: *"per the task's bound... the 2/2 flip bar was not met,
so per protocol the expensive n=15×3 gate suite was correctly skipped"*): the
mechanism-level bar (falls eliminated on the two target GEN-1 episodes) was not
met at either tested parameterization (§3), so the full five-line n=15 gate suite
(demo 999/1000/2000, easy 999, search 999) was **not run**. Running it would not
have changed the verdict (the primary target mechanism remains unfixed) and, per
§5's ep9 finding, would plausibly have introduced a *new* documented regression
on top of that — correctly judged not worth the compute per this task's own
anti-hang/cost-discipline framing and the established codebase precedent.

---

## 7. Adoption

**CLOSE. Not adopted.** `code/scan_sched.py` has been reverted to its exact prior
behavior — re-replay of demo seed1000 ep12 post-revert reproduces the identical
original fall trace (`FAIL[fall] steps=256 final_dist=7.143`, matching §1 digit
for digit), confirming the revert is clean. The module's docstring gained one
paragraph documenting this tried-and-not-adopted investigation (mirroring
`docs/nx11_ep4.md`'s own precedent of leaving a pointer in the code so a future
agent does not silently re-derive this from scratch) — no behavior-affecting line
changed. No other file was touched this cycle.

**Not synced** to `VLA_mujoco_unitree/code/` — nothing changed
to sync (clean revert, matching `docs/nx11_ep4.md`'s own "no sync — nothing
changed" precedent).

**demo stays at 14/15 = 93.3%** (seed 999, `docs/nx10_scan_fix.md` §4.1),
easy 15/15, search 15/15 — all unchanged. The fresh-seed residual
`docs/gen1_multiseed.md` documented (demo 86.7%/80.0% at seed 1000/2000, both
attributable to this exact mechanism) remains open.

---

## 8. What remains / follow-on

- **The actual mechanism is still not fixed.** Evidence in §3 points at a
  spawn/heading-specific translational-drift-during-in-place-turning
  instability (net drift toward the wall behind a `side==1`/`side==3` spawn,
  compounding with elapsed scan-phase duration) rather than a continuous-
  rotation-length OOD problem. A future attempt should start from this framing,
  not re-litigate "bound the segment length further" (already falsified at two
  points spanning a 4x range here).
- **A genuinely different fix shape may be needed**: options not attempted here
  (out of scope for this task's assigned fix family / "one constants revision"
  bound) include (a) a direction-aware first-leg choice (try the leg whose sign
  matches some cheap prior bearing estimate, avoiding the "always wrong-direction-
  first" pattern for negative-bearing targets entirely — explicitly flagged as
  worth considering in `docs/gen1_multiseed.md` §6 item 1), which would let many
  large-negative-bearing targets be found well before ~90° of cumulative rotation
  is ever reached; or (b) directly bounding/damping the observed translational
  drift during scan (e.g. an explicit small counter-bias on vx/vy during scan
  legs) rather than trying to out-wait it with more dwell time, which §3.2 showed
  actively increases wall-proximity exposure.
- **ep9's margin erosion (§5)** is a reusable cautionary data point for any
  future scan-timing change: this episode has near-zero slack left in the
  current 1700-step demo cap once any additional scan-phase overhead is added;
  a future fix that adds scan-phase steps should re-check ep9 specifically.
- **ep4 remains its own, separately-tracked, already-closed residual**
  (`docs/nx11_ep4.md`) — independently reconfirmed here as unrelated to this
  investigation; no new information on it beyond noise-band reconfirmation.

---

## 9. Files

- `code/scan_sched.py` — reverted to exact prior behavior; one added docstring
  paragraph (no behavior change, verified via re-replay).
- No other `code/` files touched.
- No `eval/` gate artifacts produced (full gates not run — bar not met, §6).
- Scratchpad-only diagnostic script (instrumented replay harness), not committed,
  matching this codebase's established precedent for diagnostic-only scripts
  (`docs/nx10_scan_fix.md` §2.2, `docs/gen1_multiseed.md` §3).
