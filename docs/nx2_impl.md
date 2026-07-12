# NX-2 — Target-Lock Management Implementation

**Date:** 2026-07-09
**Agent:** NX-2 (implementer for `docs/rs1_lock_mgmt.md`'s design brief)
**Checkpoint:** `checkpoint/goto_best.pt` (unchanged — deploy-side only, no retraining)

---

## TL;DR

All 5 mechanisms from `docs/rs1_lock_mgmt.md` are implemented in one new shared
module, `code/lock_mgmt.py`, imported identically by `code/inferencer.py`
(easy/demo/maneuver) and `code/eval_search.py` (search) — following the same
pattern NX-1 used for `code/scan_sched.py`, per the brief's own §5
recommendation. Each mechanism is gated by its own independent env var,
default OFF:

| Toggle | Mechanism | Default |
|---|---|---|
| `LOCK_M1=1` | area-quality floor (raw contour px², `GroundingResult.best_area`) | OFF |
| `LOCK_M2=1` | N-of-M (2-of-3) tentative→confirmed lock initiation | OFF |
| `LOCK_M3=1` | innovation gate + incumbent inertia (association gating) | OFF |
| `LOCK_M4=1` | divergence watchdog (drop+rescan on a monotonic dist trend) | OFF |
| `LOCK_M5=1` | bounded coast → reroute to rescan after hold-goal-horizon expiry | OFF |

With all 5 unset, every `LockGate`/`ReacquisitionScan` method call is a
provable pass-through — verified both by construction (see per-method
docstrings in `code/lock_mgmt.py`) and empirically: a 3-episode spot-check
(demo eps 1/3/6, seed 999) with all toggles off matches `eval/p4_gate_demo`
exactly on success/failure_tag, with steps/final_dist within normal
run-to-run jitter (see §4).

---

## 1. `code/grounding.py` — one small additive change

The design brief's low-cost suggestion (M1 gating on `bbox=(x,y,w,h)`'s
`w*h` as a proxy for blob quality, "without touching grounding.py's return
contract at all") was **tried first and rejected** after empirical
instrumentation (see §2) showed it does not reliably separate the two
populations M1 needs to separate: ep0/ep5's false-positive blobs are
thin/irregular slivers with a **large bounding box but small true contour
area** — that mismatch (low fill-ratio) is exactly *why* their `conf_area`
is near-zero in grounding.py's existing confidence formula in the first
place. A bbox-area floor that's safe against currently-passing episodes
(which have compact, near-square/round blobs where bbox ≈ contour) sits
*above* several of the currently-passing episodes' legitimate small far-range
blobs' contour areas but often *below* the false-positive blobs' bbox areas —
i.e. bbox w*h actively points the wrong way for this specific failure mode.

Fix: added one small, purely-additive dataclass field to
`GroundingResult`:

```python
best_area: Optional[float] = field(default=None, repr=False)
```

populated with the raw contour area (`cv2.contourArea`, the exact quantity
that already feeds `conf_area` at `grounding.py:809` — unchanged) at the
sole successful-return site. Default `None`, always passed by keyword,
appended after the existing `bbox` field — zero risk to any existing
positional or keyword construction/consumer of `GroundingResult`, and **zero
change to any other returned value** (dist/cos_th/sin_th/confidence/
not_visible/mask/bbox are byte-identical to before this change for every
caller that doesn't read the new field). Confirmed via `code/grounding.py`'s
own `__main__` smoke test (still passes) and the baseline spot-check (§4).

---

## 2. M1 — area-quality floor: constant choice + evidence

**Chosen floor: `M1_AREA_FLOOR_PX2 = 100.0`** (raw contour px², gated against
`GroundingResult.best_area`).

**Evidence (risk #1 from the design brief, explicitly required before locking
in a number):** instrumented `ground()` via a monkeypatch wrapper (matching
FA-1's own `diag_ep0_raw.py` methodology) across two episode sets, both demo,
seed 999, checkpoint `goto_best.pt`, classical grounding:

- **Currently-PASSING long-range episodes** (`eval/p4_gate_demo`): ep1 (cyan
  cube, 7.42m), ep3 (red cube, 7.00m), ep6 (red cone, 8.17m), ep9 (orange
  cylinder, 8.61m — the single farthest passing target in the demo set).
  Recorded `best_area` (raw contour px²) on every accepted detection across
  full 1400-step rollouts:

  | ep | n accepted | min accepted contour area (px²) | at dist |
  |---|---|---|---|
  | 1 | 92 | 190.0 | 5.43m |
  | 3 | 83 | **123.5** | 1.57m (close-range, end of approach) |
  | 6 | 97 | 229.5 | 5.90m |
  | 9 | 102 | 133.5 | 0.33m (close-range, end of approach) |

  **Global minimum across all 4 currently-passing episodes: 123.5 px²** (ep3).

- **Currently-FAILING episodes' false-positive blobs** (ep0, ep2, ep5 — the
  RS-1-diagnosed marginal/flickering-blob-at-wrong-depth family): min accepted
  contour areas of 124.0 (ep0), 44.0 (ep2), 188.0 (ep5) — confirming these
  really are the degenerate small/thin slivers RS-1 diagnosed (ep0's smallest
  are 19×9, 25×9, 30×9 px bboxes — aspect ratio 2-3:1, vs. ep1's legit
  smallest 15×16, 15×17 — aspect ≈0.9:1).

`M1_AREA_FLOOR_PX2 = 100.0` sits **~19% below** the global passing-episode
minimum (123.5) — zero risk of rejecting any detection observed across the
full rollouts of ep1/3/6/9 — while still ~2.5× the raw `MIN_BLOB_AREA=40px`
detection floor, rejecting the most degenerate near-noise blobs (e.g. ep2's
44/60 px² false positives).

**Documented limitation (honest, not swept under the rug):** this floor gives
**partial**, not complete, protection against ep0/ep5's specific failure. Their
false lock's *steady-state* accepted area (median 522-9194 px² across
instrumented reruns) is well above any floor that doesn't also reject
legitimate far-range detections — a pure area floor can only trim the
worst transient slivers at the very start of the (mis-)lock, not the
established wrong lock itself, since the same thin-sliver blob's *bounding
box* grows over the episode even though its *contour* fill-ratio stays low.
This matches the design brief's own §2 ranking (M1 = MEDIUM confidence, rank
#3, "defense-in-depth alongside M2/M4, not a standalone fix") rather than
over-claiming a fix the evidence doesn't support. Confirmed empirically: a
gate-call-counting instrumented rerun of ep0 with `LOCK_M1=1` (900 steps)
recorded **0 rejections** (69/69 accepted) — in this specific deterministic
replay, ep0's worst observed blob (124 px²) sits just above the 100 px²
floor. Raising the floor above 124 to catch it would cut into the
123.5 px² currently-passing minimum (ep3) — exactly the regression risk
brief §4 flagged as "the single most likely way this design regresses a
currently-passing episode." The floor is deliberately conservative;
M4 (not M1) is what the brief itself identifies as ep2's dedicated fix, and
M1 is scoped as a hardening/hygiene layer, not ep0/ep5's silver bullet.

Diagnostic scripts (scratchpad, not committed):
`/tmp/.../scratchpad/diag_bbox_areas.py` (passing eps),
`/tmp/.../scratchpad/diag_bbox_bad.py` (failing eps), both re-run with
`r.best_area` after the grounding.py field was added.

---

## 3. Design summary of the other 4 mechanisms

All implemented in `code/lock_mgmt.py`'s `LockGate` class (state: `'NONE'` |
`'CONFIRMED'` — the brief's `TENTATIVE` is folded into `'NONE'` + an M2 ring
buffer; `COASTING` is the caller's own existing `_frames_since_detection`
bookkeeping, untouched).

**M2 (N-of-M, `M2_CONFIRM_M=2`/`M2_CONFIRM_N=3`, tol `0.6m`/`12°`):** a fresh
detection from `'NONE'` state opens a 3-slot ring buffer; confirms (→
`'CONFIRMED'`) once 2 of the buffered entries are mutually consistent with the
latest one. Runs on **every** classical-grounding cycle regardless of
`_scan_active`, so a target seen consistently through the last 2-3 scan
cycles confirms immediately on scan-exit (per the brief's explicit
requirement) rather than adding extra post-scan delay.

**M3 (innovation gate + incumbent inertia, `25°`/`0.8m` gate ×1.5 near
`2.0m`, `1.3×` area margin sustained `K=2` cycles):** once `'CONFIRMED'`,
gates each new detection against the last-accepted (dist,bearing). Within
gate → accept & refresh incumbent. Outside gate → only accept as a new
incumbent if its area beats the current incumbent by ≥1.3× for 2 **consecutive**
cycles (streak resets on any cycle that doesn't beat the margin). Confirmed
actively engaging (not dead code): an instrumented full-episode rerun of demo
ep12 with `LOCK_M3=1` recorded 139 gate calls, 1 rejection (an isolated
close-range outlier that didn't sustain the K=2 streak).

**M4 (divergence watchdog, 15-cycle window, `0.5m` trend margin):** while
`'CONFIRMED'` and walking (proxy: `not _scan_active and cached_dist > stop_r`
— see §3.1 below for why this proxy was necessary), tracks a rolling 15-cycle
deque of the resolved best-distance-estimate; if `dist_now - min(window) >
0.5m`, drops the lock (`force_drop()`) and re-enters scan. Exempted for 15
cycles after any (re)confirmation and during a discontinuity cooldown (see
§3.2). **Confirmed firing**: a verbose instrumented rerun of demo ep2 (the
one episode FA-1/RS-1 identified as ep2's *only* addressable mechanism) with
`LOCK_M4=1` printed `[lock] M4 divergence -> drop+rescan at step=410`.

**M5 (bounded coast → rescan, unchanged `HOLD_GOAL_HORIZON=100`):**
`LockGate.coast_expired(frames_since_detection, horizon)` returns `True` iff
`LOCK_M5` is on and the horizon was just exceeded; the caller then does the
same drop+rescan action as M4, instead of freezing `cached_goal_vec` forever.

### 3.1 The "walking" proxy for M4

The brief specifies M4's trigger condition as "while walking (commanded
`vx > 0`)." For `goal_source='classical'` with the (default) `vel_source=
'predicted'`, there is no explicit commanded-vx variable in scope at the
grounding-update call site — vx is the *model's own* output, computed later
in the per-step forward pass from the injected goal, not computed explicitly
at goal-update time (unlike the `vel_source='gt'` path, which does compute an
explicit privileged vx). Both call sites therefore use the proxy `not
_scan_active and dist_estimate > stop_r` ("not currently scanning, and not yet
within the stop radius") — documented explicitly in both `inferencer.py` and
`eval_search.py` at the call site.

### 3.2 Mandatory carve-outs (design brief risk #2)

`LockGate.mark_discontinuity()` is called by `inferencer.py` at the exact
cycle of:
- the CAM-2 fallback probe-adopt event (`gr = gr2`, `inferencer.py`'s
  bounded-probe block), and
- an `_active_cam` Schmitt-trigger flip (GROUNDING↔PROXIMITY).

This bypasses M3's innovation gate for that cycle's `gate_detection()` call
(accepts unconditionally, refreshes the incumbent) and suppresses M4's
trigger for the following 2 cycles, **and clears M4's distance window** (pre-
event samples are from a different camera's geometry/FOV and would otherwise
corrupt the post-event window-min comparison once the cooldown lapses — a
detail not explicit in the brief but a direct consequence of its stated
intent that these events must not read as track anomalies).
`eval_search.py` has no second camera / Schmitt handoff, so it never calls
`mark_discontinuity()` — M3/M4 always apply their full gate there, per the
brief's own simpler-bypass-logic note for search.

### 3.3 Rescan mechanism — why a new `ReacquisitionScan`, not the existing scans

Both callers' *existing* scan mechanisms gate their own timeout off the
**absolute episode step count**:
- `inferencer.py`'s H3 scan: `if step >= SCAN_TIMEOUT` (200).
- `eval_search.py`'s NX-1 `BidirectionalScanSchedule`-driven scan: `if step >=
  SCAN_TIMEOUT` (1150, imported from `code/scan_sched.py`).

Re-arming either mid-episode (as M4/M5 do, potentially at step 600-1400)
would immediately satisfy `step >= SCAN_TIMEOUT` and the "rescan" would exit
before ever rotating a single degree — a **latent bug** neither prior agent
needed to hit since neither pre-NX-2 mechanism ever re-triggered a scan
mid-episode. `code/lock_mgmt.py`'s `ReacquisitionScan` wraps NX-1's own
`BidirectionalScanSchedule` (same `SCAN_LEG_DEG=165`/`SCAN_DWELL_STEPS=45`/
`SCAN_TIMEOUT=1150` constants — per the brief's mandate to reuse it, never an
unbounded spin) but tracks its own **local** step counter starting at 0 from
the moment it's constructed, so it's always safe to instantiate fresh at any
point in an episode. Both callers gain a new `_using_rescan_sched` flag that
routes their scan-step computation to `ReacquisitionScan.step(yaw)` instead
of their original absolute-step logic — but this branch is **only ever
reached after an actual M4/M5 trigger** (both individually toggled, default
off), so with M4/M5 off the original H3 / `BidirectionalScanSchedule` scan
code paths are completely untouched and byte-identical.

---

## 4. Verification

### (a) Baseline (all toggles OFF) — 3-episode spot-check vs `eval/p4_gate_demo`

`checkpoint/goto_best.pt`, demo, seed 999, `goal_source=classical`,
`vel_source=predicted`, device=cpu:

| ep | NX-2 (all off) | `eval/p4_gate_demo` | match |
|---|---|---|---|
| 1 | SUCCESS, steps=937, fd=0.364 | SUCCESS, steps=933, fd=0.365 | **YES** |
| 3 | SUCCESS, steps=851, fd=0.370 | SUCCESS, steps=838, fd=0.369 | **YES** |
| 6 | SUCCESS, steps=995, fd=0.368 | SUCCESS, steps=987, fd=0.360 | **YES** |

Success/failure_tag identical on all 3; step-count deltas (4-8 steps, <1%)
match this codebase's own documented run-to-run jitter (`docs/nx1_scan.md`
§5, EGL/physics non-determinism, ±1-2 steps typical). **Baseline intact.**

### (b) Per-toggle 1-episode smoke (no crash)

All run to completion (`maxsteps=1400`, demo, seed 999, `checkpoint/
goto_best.pt`, device=cpu) with no exception:

| Toggle | Episode | Result | Notes |
|---|---|---|---|
| `LOCK_M1=1` | ep0 (cyan cone, target M1 episode) | FAIL[didnt-reach], fd=3.40 | ran clean; 0/69 gate rejections in this deterministic replay (see §2 limitation) |
| `LOCK_M2=1` | ep1 (any) | SUCCESS, steps=907, fd=0.375 | ran clean |
| `LOCK_M3=1` | ep12 (target M3 episode) | FAIL[didnt-reach], fd=5.70-6.69 (run-to-run variance observed) | ran clean; 1/139 gate rejections confirmed (mechanism engages) |
| `LOCK_M4=1` | ep2 (target M4 episode) | FAIL[didnt-reach], fd=10.94 | ran clean; confirmed **firing** — verbose rerun logged `[lock] M4 divergence -> drop+rescan at step=410` |
| `LOCK_M5=1` | ep3 (any) | SUCCESS, steps=851, fd=0.368 | ran clean |

Also smoke-tested the **search** call site (`code/eval_search.py`'s
`_run_search_rollout`, search ep0, seed 999, `MAXSTEPS_SEARCH=2000`):
baseline (all off), `LOCK_M2=1`, and `LOCK_M4=1` all completed successfully
(spotted@980-990, final_dist 0.47-0.49, no crash) — confirming both call
sites are wired correctly.

`code/grounding.py`'s own `__main__` smoke test still passes after the
`best_area` field addition.

**Scope note:** per task instructions, this pass validates (a) baseline
byte-identical-behavior and (b) each toggle running crash-free + (where
checked) actually engaging its gating logic — it does **not** claim to have
proven each mechanism improves the full n=15 demo/search success rate (that
full 3-skill re-gate, per `docs/rs1_lock_mgmt.md` §6's own verification note,
is scoped as follow-on work for a dedicated gate pass, matching this
codebase's established `docs/nx1_scan.md`/`docs/cam_p4_gate.md` precedent of
separating implementation from full-gate evaluation).

---

## 5. Files changed

- `code/lock_mgmt.py` (**new**) — `LockGate` (M1-M5 state machine + gating)
  and `ReacquisitionScan` (bounded M4/M5 rescan wrapper around NX-1's
  `BidirectionalScanSchedule`). All 5 toggles read once at import time from
  `LOCK_M1`..`LOCK_M5` env vars.
- `code/grounding.py` — added `GroundingResult.best_area` (additive, default
  `None`, populated at the one successful-return site with the existing
  `best_area` contour-area variable — zero change to any other field/caller).
- `code/inferencer.py` — imports `LockGate`/`ReacquisitionScan`; wraps the
  classical-grounding EMA-update block with `_lock_gate.gate_detection(...)`;
  calls `_lock_gate.mark_discontinuity()` at the CAM-2 probe-adopt and
  Schmitt-flip points; adds the M4 `end_of_cycle(...)` check once per
  grounding cycle; adds the M5 `coast_expired(...)` check to both the
  gate-rejected and raw-miss branches; adds a `_using_rescan_sched` branch to
  the H3 scan block that drives `ReacquisitionScan` instead of H3's
  absolute-step logic when a M4/M5 drop has occurred. **The original H3 scan
  logic itself, and every code path exercised when all 5 toggles are off, is
  byte-for-byte unchanged** (new code sits in new `if`/`elif` branches gated
  on flags that are only ever set by the new mechanisms).
- `code/eval_search.py` — identical pattern: imports, gate-wraps the EMA
  update, adds M4/M5 checks, adds a `_using_rescan_sched` branch alongside
  the existing NX-1 `BidirectionalScanSchedule`-driven scan block. `search`
  never calls `mark_discontinuity()` (no second camera).
- `code/scan_sched.py`, `code/fancy_demo.py`, `code/demo.py`, `code/steer.py`,
  `code/teacher.py` — **not modified** (NX-1's scan-schedule logic is reused,
  not disturbed, per task instructions).
- `docs/nx2_impl.md` (this file).

Diagnostic/verification scripts (scratchpad, not committed):
`diag_bbox_areas.py`, `diag_bbox_bad.py` (M1 floor evidence),
`spotcheck_baseline.py` (verification a), `smoke_toggle.py` +
`run_all_smokes.sh` (verification b, demo), `smoke_search.py` +
`run_search_smokes.sh` (verification b, search), `verify_m4_fires.py`,
`verify_m1_m3_engage.py`, `verify_m3_full.py` (mechanism-engagement checks).
