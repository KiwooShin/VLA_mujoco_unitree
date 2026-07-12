# Target-Lock Management — Applied Design Brief

**Date:** 2026-07-09
**Agent:** RS-1 (research-only; no code changes, no GPU)
**Scope:** harden the classical HSV+depth grounder (`code/grounding.py` `ground()`,
consumed by `code/inferencer.py`'s main rollout loop and independently duplicated in
`code/eval_search.py`'s `_run_search_rollout`) against the 3 mechanisms diagnosed in
`docs/fa1_failures.md`: **ep2** (confident false-positive lock, never corrects),
**ep0/ep5** (marginal/flickering blob locked at wrong depth, held stable),
**ep12** (mid-approach hijack to a mirrored-bearing same-color distractor).

This brief is written for the next implementation agent. It assumes familiarity
with `docs/fa1_failures.md` §1 and the current constants below; no further
diagnosis is proposed here, only the fix design.

---

## 0. Ground truth: current system constants (read from code, not assumed)

| Constant | Value | Location |
|---|---|---|
| Control loop | 50 Hz (`SIM_DT * CONTROL_DECIMATION` = 0.02s/step) | `inferencer.py:109` |
| Grounding cadence | 5 Hz (`GROUNDING_PERIOD=10` steps) | `inferencer.py:75` |
| `_GOAL_EMA_ALPHA` | 0.4 | `inferencer.py:618`, `eval_search.py:396` |
| `HOLD_GOAL_HORIZON` | 100 steps = 10 grounding cycles = 2.0s | `inferencer.py:627`, `eval_search.py:399` |
| `MIN_BLOB_AREA` | 40 px² (lowered from 60 in V2 specifically for 9m/0.48m targets) | `grounding.py:96` |
| `MIN_CONFIDENCE` | 0.05 | `grounding.py:160` |
| `confidence` formula | `0.6*conf_area + 0.4*conf_depth`; `conf_depth=1.0` whenever >20 valid depth px, regardless of blob size | `grounding.py:795-797` |
| `SCAN_ALIGNED_THR_DEG` | 40° | `inferencer.py:615` |
| `SCAN_TIMEOUT` | 200 steps (demo, bounded R/L/R) vs 600 steps (search, fixed CCW — separate known issue, FA-1 rank #1, out of scope here) | `inferencer.py:611`, `eval_search.py:391` |
| Schmitt handoff (CAM-2) | `CAM_D_LO=1.2m` (GROUNDING→PROXIMITY), `CAM_D_HI=1.6m` (PROXIMITY→GROUNDING), fallback-probe gate `CAM_PROXIMITY_D_FAR=1.81m` | `inferencer.py:636-652` |
| Walk speed | ~0.5-0.8 m/s → ~0.1-0.16m forward progress per grounding cycle | task brief |

**Critical implementation fact:** the EMA/hold-goal/scan state machine is
**duplicated verbatim** between `inferencer.py` (easy/demo/maneuver) and
`eval_search.py` (search) — not shared. Any lock-management change must be applied
to *both* call sites or it will silently not protect the search skill. See §5.

**Diagnosis of the ep0/ep5 root cause, more precisely than "no corroboration
requirement":** `conf_depth` saturates to 1.0 for any blob with >20 valid depth
px, independent of blob *area*. ep0's accepted blob had `best_area` barely above
the 40px floor (conf_area ≈ 0) but still produced `confidence=0.40` because the
depth term alone contributes up to 0.4 and dominates. This is not a temporal
flicker problem — the wrong blob was detected *consistently* (same dist/bearing,
every call) for 90 straight cycles. An N-of-M temporal-consistency check would
**not** reject it, because it is self-consistent. The actual bug is that the
blended confidence formula lets a decent depth-sample-count compensate for a
near-zero area score. This matters for constant selection in §2.

---

## 1. Candidate mechanisms (tracking-by-detection practice, scaled to our rates)

For each: description → which failure(s) it targets → why → cost → interaction risk.

### M1 — Explicit area-quality floor, independent of the blended confidence score
**What:** Gate lock-eligibility on `conf_area` (or raw `best_area`) directly,
*in addition to* the existing blended `confidence > MIN_CONFIDENCE`, rather than
letting `conf_depth` compensate for it. Equivalent to radar's practice of
gating on raw SNR before it ever reaches a fused track score.
**Targets:** ep0, ep5 — directly, as the primary fix (see root-cause note above).
**Why:** removes the exact compensation path that let a sliver blob pass at
confidence=0.40 while contributing negligible real signal.
**Cost:** very low. `GroundingResult` doesn't currently expose `conf_area`/`best_area`
separately, but it does expose `bbox=(x,y,w,h)` — `w*h` is a usable proxy without
touching `grounding.py`'s return contract at all. ~10 lines in the caller.
**Interaction risk:** `MIN_BLOB_AREA` was deliberately lowered in V2 specifically to
keep detecting legitimately small/far targets (9m, 0.48m objects). A new area
floor for *lock-worthiness* must sit between the raw 40px detection floor and
"clearly a full silhouette" — too high reintroduces total-miss on legitimately
distant small-but-correct blobs (this is the single biggest regression risk in
this whole brief — see §4).

### M2 — N-of-M tentative→confirmed track initiation (SORT/radar track-management pattern)
**What:** classic MOT track lifecycle: a fresh detection opens a `TENTATIVE` track;
promote to `CONFIRMED` only after M of the last N grounding cycles report a
detection passing M1's floor with mutually consistent (dist, bearing) (loose
tolerance to absorb gait-cycle sway sampled at 5Hz — see §4). Standard practice
(SORT's `min_hits`, radar M-of-N / history logic) uses M=2-3 of N=3-5.
**Targets:** general hardening against one-off spurious blobs (edge artifacts,
single-frame HSV noise); secondary/general-safety-net for ep0/ep5, not primary
(M1 is primary there, per the root-cause note).
**Why:** prevents a single anomalous frame from ever seeding a track, at near-zero
cost given we already run at a slow 5Hz cadence (a 2-3 cycle delay is 0.4-0.6s).
**Cost:** low. A length-3 ring buffer of `(dist, bearing, passed_M1)` per call site.
**Interaction risk:** must NOT re-arm on every `CONFIRMED` cycle (only on fresh
(re)initiation from a `NONE`/dropped state) or it adds latency to every single
grounding update, which risks regressing **easy/classical's near-instant single-frame
alignments** (currently 100% — the tightest currently-passing margin in the whole
suite). Confirmation counting should span the scan→walk transition, not reset to
zero the instant `_scan_active` flips false (a target seen consistently for the
last 2-3 scan cycles should confirm immediately on scan-exit, not add extra delay).

### M3 — Association gating with incumbent inertia (Mahalanobis/Euclidean innovation gate)
**What:** once `CONFIRMED`, compute a predicted (dist, bearing) each cycle (constant-
position or light constant-velocity model off the current EMA). Gate incoming raw
detections: reject (don't feed EMA) if `|Δbearing|` or `|Δdist|` exceeds a gate
around the prediction, **unless** the new detection's quality clearly exceeds the
incumbent's by a hysteresis margin, sustained for K≥2 cycles (classic MOT/track-score
"replace only if better by margin for K frames", not "first thing above floor wins").
**Targets:** ep12 — directly, and precisely. This is the standard fix for MOT's
"track ID switch to a nearby distractor of similar appearance" failure mode; ep12's
mirrored-bearing same-color-and-similar-size distractor is a textbook instance.
**Why:** ep12's distractor doesn't beat the true target by a large quality margin —
it just becomes *comparably* large/confident once both are near. A first-past-
the-post accept (current behavior) flips to whichever is fresher; incumbent inertia
requires a real, sustained margin to unseat a working lock.
**Cost:** medium. Needs incumbent's last-accepted quality score (area proxy from
bbox, no `grounding.py` change needed) plus a small predictor state.
**Interaction risk — the sharpest one in this design:** gate thresholds must be
**exempted** for (a) the CAM-2 fallback probe-adopt event (`gr = gr2` in
`inferencer.py:790-798` — this is an intentional trust-the-recovery-camera override
built to escape a miss streak; gating it would defeat its purpose and reintroduce
the exact ep13/`cam_p3`/`cam_p4` deadlock class this codebase already fought once,
`inferencer.py:638-651`), and (b) the grounding cycle at/immediately after an
`_active_cam` Schmitt flip (GROUNDING↔PROXIMITY), since the two cameras have
different FOV/pitch/intrinsics and a legitimate same-target bearing/dist shift at
handoff must not read as an "innovation violation." Also, the bearing gate should
be distance-scaled, not a fixed absolute degree threshold — at close range (near
the 1.2-1.6m handoff band) a given lateral offset maps to a much larger bearing
swing than at 5-9m, so a fixed tight gate risks rejecting the *true* target's own
legitimate bearing motion right as ep12-style close-range approaches are exactly
when this matters most.

### M4 — Divergence watchdog (physically-implausible-trend detector)
**What:** while `CONFIRMED`, walking (commanded vx > 0), and not within an
exemption window (see below), track EMA'd dist over a rolling window of N cycles.
If `dist_now - min(dist_window)` exceeds a margin (net trend, not strict
frame-to-frame monotonicity — robust to single-cycle EMA/gait noise), declare the
lock divergent: drop to `NONE`, clear `_goal_ema`/`_last_known_goal`, re-enter
`_scan_active=True` to force fresh re-acquisition through M2's gate rather than
continuing to trust the same bad detection stream.
**Targets:** ep2 — directly, and it is the *only* mechanism in this set that
does. ep2's blob is not marginal-quality (M1 doesn't catch it — a wall/cyan-
collision blob can be large and confident), not one-off-spurious (M2 doesn't catch
it — it's the same wrong lock, consistently detected), and there's no established
incumbent to protect against a hijack (M3 doesn't apply — it's wrong from
detection #1). It needs its own independent trigger keyed on the one thing that is
anomalous: distance to goal increasing monotonically while the robot is actively
walking toward it is physically implausible for a correct lock.
**Cost:** low. Fixed-size deque of recent EMA dists + a vx-nonzero flag.
**Interaction risk:** exactly the one already flagged in `fa1_failures.md`'s own
candidate #3 write-up — must not false-trigger on (a) the first N cycles right
after any (re)confirmation, where legitimate heading-correction geometry can
transiently increase straight-line distance before the turn completes, or (b) near
the Schmitt handoff band, where a genuine camera-switch dist reading could look
like a jump.

### M5 — Bounded coast, then reroute to re-scan rather than freeze forever
**What:** current `HOLD_GOAL_HORIZON` logic freezes `cached_goal_vec` at its last
value indefinitely once the horizon is exceeded (`inferencer.py:852` and
`eval_search.py`'s equivalent `else` branch — no further action for the rest of the
episode). Convert horizon-exceeded into an explicit state transition: drop to
`NONE`, set `_scan_active=True` (reusing existing scan machinery, not new
infrastructure) instead of silent permanent dead-reckoning.
**Targets:** general hygiene / closes the "coasting vs termination" gap the task
brief called out as currently entirely absent. Not directly evidenced in any of
the 4 named episodes (ep0/ep5 keep getting sporadic re-detections and never
actually exhaust the 100-step coast budget), but cheap, low-risk, and consistent
with M4's drop-to-rescan pattern — worth bundling since it reuses the same new
"drop to NONE → rescan" plumbing M4 needs anyway.
**Cost:** trivial (~5 lines) once M4's rescan-reentry path exists.
**Interaction risk:** must not re-enter a *full* 200/600-step scan sweep mid-episode
after a brief coast-out near the goal (wasteful, could itself look like new
"walking away" behavior to M4 if not handled carefully) — reuse the same bounded
scan, don't add a second scan variant.

### Considered and deprioritized
- **Full constant-velocity Kalman filter replacing the EMA** on (dist,bearing):
more principled association math (real Mahalanobis distance from a covariance
estimate) but meaningfully higher implementation/tuning cost for a CPU-ms budget
that doesn't need it — M3's simple fixed-gate + incumbent-margin approximation
captures ~the same practical benefit for a fraction of the cost and risk. Not
recommended for this pass; worth reconsidering only if M3's fixed gate proves too
coarse in practice.

---

## 2. Recommended minimal composite design

State machine per target-lock, reusing existing `_scan_active` as the `NONE`/
searching state (no new top-level state variable needed beyond a small quality/
history buffer):

```
NONE (scanning) --[M2: M-of-N pass]--> CONFIRMED --[detection, M3 gate]--> CONFIRMED (updated)
                                           |
                                           |--[not_visible]--> COASTING (bounded, M5)
                                           |                        |--[re-detected, M3 gate]--> CONFIRMED
                                           |                        |--[age > HOLD_GOAL_HORIZON]--> NONE (M5)
                                           |
                                           |--[M4: divergence trend]--> NONE (force rescan)
```

**Constants (concrete, scaled to 5Hz grounding / 50Hz control / 0.5-0.8 m/s walk):**

| Param | Value | Rationale |
|---|---|---|
| M1 area-quality floor | `bbox_area >= 0.5-1.0% of valid_image_area` (≈865-1730 px² at 480×360, vs current 40px MIN_BLOB_AREA) | Well above ep0's sliver (~40-60px, conf_area≈0), comfortably below a legitimate 9m/0.48m target's expected silhouette at 480×360 (needs a quick numeric check against `docs/grounding_dist.md`'s known blob-size-vs-distance table before locking in — flagged in §4) |
| M2 confirmation | M=2 of N=3 grounding cycles (0.4-0.6s), tolerance `|Δdist|<0.6m`, `|Δbearing|<12°` between hits | Loose enough to absorb 5Hz-sampled gait sway; tight enough to reject an unrelated spurious blob |
| M3 innovation gate | `|Δbearing| < 25°` (scaled up near proximity-handoff range, e.g. ×1.5 below 2m), `|Δdist| < max(0.8m, expected closing distance this cycle × 1.5)` | Expected closing distance/cycle ≈ 0.1-0.16m at 0.2s/cycle and 0.5-0.8m/s; 0.8m floor covers normal EMA lag + gait noise |
| M3 incumbent-replace margin | challenger area ≥ 1.3× incumbent's, sustained K=2 cycles | Standard "beat by margin for K frames" pattern; prevents first-past-post flips to a comparable-quality distractor |
| M4 divergence window | N=15 grounding cycles (~3s), trigger if `dist_now - min(window) > 0.5m` while vx>0 | Matches FA-1's own suggested N≈15-20; window-min (not frame-to-frame) rejects single-cycle noise |
| M4 exemption | first 15 cycles after any (re)confirmation; ±2 cycles around a Schmitt cam flip | Prevents false-trigger on legitimate post-acquisition heading correction or handoff dist jump |
| M5 coast budget | unchanged, `HOLD_GOAL_HORIZON=100` steps (10 cycles, 2.0s) | No evidence any of the 4 episodes need it changed; only the post-horizon *action* changes (rescan, not freeze) |
| M3/M4 bypass | CAM-2 fallback-probe adopt event (`gr=gr2`); ±1 cycle around `_active_cam` flip | Hard requirement — both events are legitimate, intentional discontinuities in (dist,bearing), not track anomalies |

**Ranked by expected impact on the 4 named episodes:**

1. **M4 (divergence watchdog) → ep2.** Only mechanism that addresses it; HIGH
   confidence (matches FA-1's own independently-derived candidate #3, ep2's
   monotonic-increase signature is unambiguous).
2. **M3 (innovation gate + incumbent inertia) → ep12.** Direct, textbook fix for
   the diagnosed mid-approach hijack; MEDIUM-HIGH confidence, contingent on
   getting the distance-scaled bearing gate right (see risk above).
3. **M1 (area-quality floor) → ep0, ep5.** Primary fix, not M2 — correction to
   the task brief's framing; ep0's blob is self-consistent, not flickering-and-
   inconsistent, so temporal corroboration alone would not reject it. MEDIUM
   confidence pending the numeric floor check in §4.
4. **M2 (N-of-M) + M5 (bounded coast→rescan) → general hardening**, no specific
   documented episode requires them but they close real gaps (one-off spurious
   locks; infinite-freeze-after-coast) at very low marginal cost given M3/M4's
   infrastructure already exists.

---

## 3. Recommended composite, one paragraph

Add a thin lock-management layer above the existing EMA/hold-goal machinery,
shared between `inferencer.py` and `eval_search.py` (see §5): gate every raw
detection on an explicit area-quality floor (M1) before it's eligible to be a
"hit" at all; require 2-of-3 consistent hits to confirm a fresh lock (M2); once
confirmed, gate incoming detections against the current lock with a distance-
scaled innovation gate and require a sustained quality margin to let a challenger
take over (M3); run a rolling divergence check that drops a confirmed lock back
to search if distance-to-goal is trending up while walking forward (M4); and
convert the existing hold-goal horizon's expiry from silent-freeze into an
explicit drop-to-rescan (M5). M3/M4 must both special-case the CAM-2 fallback
probe and the Schmitt handoff boundary as legitimate discontinuities, not
anomalies.

---

## 4. Top regression risks (ranked)

1. **M1's area floor could reintroduce total-misses on legitimately small/distant
   targets.** `MIN_BLOB_AREA` was deliberately lowered in V2 *specifically* to keep
   detecting 9m/0.48m targets — that's exactly the size regime M1 must not
   exclude. Before locking in a number, the implementer should pull actual
   accepted-blob areas for currently-*passing* long-range episodes (e.g. from
   `docs/grounding_dist.md`'s distance/blob-size data or a quick instrumented
   rerun akin to FA-1's `diag_ep0_raw.py`) and set the M1 floor comfortably below
   the smallest of those, not just comfortably above ep0's ~40-60px sliver. This
   is the single most likely way this design regresses a currently-passing
   episode.
2. **M3/M4 must not fight the Schmitt proximity handoff or the CAM-2 fallback
   probe.** Both are legitimate, intentional (dist,bearing) discontinuities by
   design (different camera FOV/pitch; explicit miss-streak recovery). This
   codebase has already been bitten by exactly this class of interaction once
   (the ep13 regression / `cam_p3`→`cam_p4` deadlock fix, `inferencer.py:638-651`)
   — a naively-applied innovation gate or divergence watchdog with no exemption
   for these two events would very plausibly reproduce that failure class, this
   time via the new lock-management layer instead of the old gate constant. Any
   implementation must explicitly bypass M3/M4 for the probe-adopt cycle and the
   ±1-2 cycles around an `_active_cam` flip, and this needs to be gated with a
   real test (search skill and demo-skill close-range approaches both cross the
   1.2-1.6m band routinely).
3. *(secondary, process risk, not a runtime regression)* **Duplicated logic
   across two files.** See §5 — if the fix lands in `inferencer.py` only,
   `eval_search.py`'s search skill gets none of this hardening, and a future
   agent re-discovering "search still hijacks" will waste a cycle rediscovering
   that the two call sites diverged.
4. **M2's confirmation delay interacting with easy/classical's near-instant
   single-frame alignments (currently 100%).** Should be low-risk if confirmation
   window (2 of 3 cycles, 0.4-0.6s) is short relative to episode length, but this
   is exactly the kind of shared-logic change this codebase's own history
   (`docs/rot_dart.md`, `docs/vel_proprio.md`) shows can have unexpected
   cross-episode costs — a full easy+demo+search re-gate (n=15 each) is mandatory
   before considering this done, not optional.

---

## 5. Implementation-cost note (for scoping, not a mandate)

- M1: ~10 lines (bbox-area proxy check at each call site).
- M2: ~20 lines (ring buffer + consistency check), ×2 files.
- M3: ~30-40 lines (predictor state, gate check, incumbent-quality tracking,
  probe/handoff bypass flags), ×2 files (search has no Schmitt handoff, so its
  bypass logic is simpler — only needs the ring-buffer/gate, not the cam-flip
  exemption).
- M4: ~20 lines (rolling deque, vx-flag, exemption window), ×2 files.
- M5: ~5 lines, reuses M4's rescan-reentry path.
- **Strong recommendation:** factor the whole state machine into one shared
  helper (e.g. a small class/module in `code/grounding.py` or a new
  `code/lock_manager.py`) imported by both `inferencer.py` and `eval_search.py`,
  rather than a third copy-paste. The existing duplication is already a known
  maintenance hazard in this codebase (identical `_GOAL_EMA_ALPHA`/
  `HOLD_GOAL_HORIZON` constants independently defined twice); adding 5 more
  stateful mechanisms on top of that duplication compounds the risk of the two
  skills silently drifting.
- Total estimated size: ~150-200 lines including both call sites, well within
  the "LOW-MEDIUM" effort band FA-1's own rank #2 candidate estimated for this
  class of fix.

---

## 6. Verification note (for the implementation agent, not performed here)

Per this codebase's established discipline (`docs/rot_dart.md`):
any change here touches shared grounding/goal-update logic used by all skills, so
a full re-gate (easy + demo + search, n=15 each) is required before claiming a fix,
not just the 4 named failing episodes. Priority order for spot-checks: ep2 (M4),
ep12 (M3), ep0/ep5 (M1), then a full easy/demo/search n=15 to catch regressions
per §4.
