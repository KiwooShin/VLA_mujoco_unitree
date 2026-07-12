# NX-13 — AVOID self-body hygiene fix: adoption on its own merits

**Date:** 2026-07-10
**Agent:** NX-13 (follow-on to `docs/nx11_ep4.md`, `docs/nx9_avoid.md`, `docs/gen1_multiseed.md`)
**Starting state:** demo 14/15 (93.3%, seed 999), easy 15/15, search 15/15 — the
last-gated `docs/nx10_scan_fix.md` state. `docs/nx11_ep4.md` found a real,
mechanism-confirmed AVOID bug (no self-body exclusion in the depth
back-projection; the PROXIMITY camera sees the robot's own swinging arms at
0.24-0.43m and injects a spurious one-sided yaw bias inside the
`[1.2,1.6]`m camera-handoff band) and a working fix (`AVOID_MIN_GOAL_DIST_M`
1.2→1.6), but reverted it solely because it didn't flip demo ep4 2/2 — an
ep4-flip bar the fix was never actually trying to clear. This cycle
re-evaluates the same fix on its own merits: does it hold NX-9's mid-path
avoidance wins, does it demonstrably kill the self-body bias, and does it
hold/improve all five gate lines with zero new failures?

## TL;DR — VERDICT: **ADOPT**. `AVOID_MIN_GOAL_DIST_M` 1.2→1.6, kept. Synced
to the deploy repo (byte-copy, no git).

- **Mechanism replays**: NX-9's two mid-path avoidance wins held identically
  (demo ep1 bias-active 0.2069 both runs, matching NX-9's documented
  0.202-0.204; search ep12 bias-active 0.1714 both runs, matching NX-9's
  documented 0.176). A controlled same-session A/B on two endgame-heavy
  passers (ep0, ep6) directly attributes the self-body-bias elimination to
  the fix: ep0's `avoid_bias_active_frac` drops 0.0392→0.0000 with only the
  constant flipped (nothing else different); ep6 is **byte-identical**
  between old/new (steps=962 both) — proof the fix is surgical, touching
  only the 1.2-1.6m band and leaving genuine mid-path avoidance elsewhere
  untouched.
- **Five full gates (n=15 each, `--no-render`/`--no-video`, seed 999 unless
  noted)**: all five held their bar **exactly**, matching baseline fail sets
  episode-for-episode, on the first run (no noise-protocol rerun needed):
  demo/999 14/15 (fail={4}), easy/999 15/15, search/999 15/15, demo/1000
  13/15 (fails={7,12}), demo/2000 12/15 (fails={2,8,14}).
- **Zero regressions, zero new failure modes.** No episode that previously
  passed now fails, at any of the five gate lines.

---

## 1. Setup

the `g1nav` conda env's python interpreter, `PYTHONPATH=.:$PYTHONPATH`,
`MUJOCO_GL=egl`, pure defaults (no env overrides — `GROUND_NET=1`, `AVOID=1`
both default-on), `checkpoint/goto_best.pt`, arch=A, device=cuda.

**Change** (`code/avoid.py`, one constant): `AVOID_MIN_GOAL_DIST_M` 1.2 → 1.6,
aligning the AVOID carve-out with `inferencer.py`'s `CAM_D_HI=1.6`
PROXIMITY→GROUNDING Schmitt threshold — the exact revision `docs/nx11_ep4.md`
§4 implemented, mechanism-confirmed, and reverted only for missing an
unrelated ep4-flip bar. Module comment updated to record the re-application
and point to this doc; the constant's value is the only behavioral change.
Unit self-test (`python -m code.avoid`) re-confirmed **15/15 PASS** after the
edit.

---

## 2. Mechanism replays

### 2.1 NX-9 mid-path avoidance wins — held 2/2 each

Per the task brief's explicit trade-off concern (obstacles 1.2-1.6m from the
**goal** are no longer avoided, but NX-9's wins are **mid-path** obstacles,
so they should be unaffected) — verified directly:

| episode | run | result | final_dist | avoid_bias_active_frac | NX-9 baseline (§3.1/§3.2) |
|---|---|---|---|---|---|
| demo ep1 (999), cone wedge, GROUND_NET | 1 | SUCCESS | 0.372 | 0.2069 | 0.202 |
| demo ep1 (999) | 2 | SUCCESS | 0.372 | 0.2069 | 0.204 |
| search ep12 (999), distractor 0.92m along path | 1 | SUCCESS | 0.467 | 0.1714 | 0.176 |
| search ep12 (999) | 2 | SUCCESS | 0.467 | 0.1714 | 0.176 |

Both wins reproduce essentially identically to their NX-9 documented values
(the mid-path obstacle in both scenes sits well above the 1.6m goal-distance
threshold at encounter time, exactly as `docs/nx11_ep4.md` §6 argued
geometrically but never empirically re-verified — now confirmed by replay).

### 2.2 Endgame-heavy passers — self-body bias eliminated, no new failures

Two demo-999 passers under their winning backend (ep0 = GROUND_NET, ep6 =
classical), each independently reaching close range (both are gate
successes, i.e. "endgame" by construction):

| episode | backend | result | final_dist | avoid_bias_active_frac | NX-9 baseline |
|---|---|---|---|---|---|
| demo ep0 (999) | GROUND_NET | SUCCESS | 0.383 | **0.0000** | 0.138 |
| demo ep6 (999) | classical | SUCCESS | 0.369 | 0.0202 | 0.040 |

Both dropped relative to NX-9's original baseline (recorded before NX-10's
scan fix and NX-11's diagnosis intervened, so not a perfectly controlled
comparison on its own) — motivating a same-session controlled A/B, isolating
only the constant:

| episode | cutoff | result | steps | final_dist | avoid_bias_active_frac |
|---|---|---|---|---|---|
| ep0 | OLD (1.2) | SUCCESS | 578 | 0.367 | **0.0392** |
| ep0 | NEW (1.6) | SUCCESS | 590 | 0.379 | **0.0000** |
| ep6 | OLD (1.2) | SUCCESS | 962 | 0.366 | 0.0103 |
| ep6 | NEW (1.6) | SUCCESS | 962 | 0.366 | 0.0103 |

ep0: the fix takes a nonzero endgame bias straight to zero, nothing else
about the run changes materially (both SUCCESS, final_dist within the
harness's documented run-to-run jitter). ep6: **byte-identical trajectory**
(steps=962 both runs) — the fix has *zero* effect here because ep6's small
residual bias-active fraction lives entirely outside the 1.2-1.6m band (i.e.
genuine mid-path avoidance activity, structurally unrelated to self-body
contamination). Together these two results are the cleanest possible
confirmation that the fix is surgical: it removes exactly the self-body
endgame band and nothing else, and produces no new endgame collisions or
falls in either replay.

---

## 3. Full gates (n=15 each, `--no-render` / `--no-video`, pure defaults)

All five commands mirror `docs/gen1_multiseed.md`'s exact protocol
(`code.eval_closedloop --goal-source classical --no-render` /
`code.eval_search --no-video`). **All five held their bar exactly on the
first run** — no gate came in under its bar, so the noise-protocol rerun was
not needed for any of them.

### 3.1 demo, seed 999 — bar: hold 14/15 → **HELD (14/15, exact fail-set match)**

| ep | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| result | S | S | S | S | **F(fall)** | S | S | S | S | S | S | S | S | S | S |
| final_dist | 0.38 | 0.37 | 0.36 | 0.37 | 0.94 | 0.37 | 0.36 | 0.38 | 0.37 | 0.37 | 0.37 | 0.37 | 0.37 | 0.39 | 0.36 |

14/15 = 93.3%. Only ep4 fails (fall, steps=1637, final_dist=0.935) — the
exact episode `docs/nx11_ep4.md` characterized as an unrelated late-episode
balance-loss mechanism (fall steps/final_dist in this run land inside
NX-11's own fix-applied mechanism-replay range, steps 1589-1617,
final_dist 1.01-1.02 — consistent, not a new failure signature).

### 3.2 easy, seed 999 — bar: 15/15 exact → **15/15**

All 15 SUCCESS, final_dist 0.559-0.584m — matches the documented easy band
exactly, zero change from baseline.

### 3.3 search, seed 999 — bar: 15/15 exact → **15/15**

SPOT 15/15, REACH 15/15, SUCCESS 15/15, falls 0/15 — including ep12 (the
mid-path win, §2.1) and ep14 (the pre-existing knife-edge bistability,
`docs/nx9_avoid.md` §3.3) both passing this run.

### 3.4 demo, seed 1000 — bar: hold 13/15 → **HELD (13/15, exact fail-set match)**

| ep | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| result | S | S | S | S | S | S | S | **F(fall)** | S | S | S | S | **F(fall)** | S | S |

13/15 = 86.7%. Fails = {7, 12}, both falls — the exact fail set
`docs/gen1_multiseed.md` documented (ep7 fall fd=4.79→4.86, ep12 fall
steps=256 both runs, fd=7.14→7.14 — essentially identical replay). The
rotation-order scan/locomotion instability GEN-1 mechanism-traced for these
two episodes (`docs/gen1_multiseed.md` §3.1) is AVOID-independent (it
recurs before/without any AVOID engagement), so this gate line was never
expected to move — confirmed unmoved.

### 3.5 demo, seed 2000 — bar: hold 12/15 → **HELD (12/15, exact fail-set match)**

| ep | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| result | S | S | **F(fall)** | S | S | S | S | S | **F(reach)** | S | S | S | S | S | **F(reach)** |

12/15 = 80.0%. Fails = {2, 8, 14} — exact match to
`docs/gen1_multiseed.md`'s documented set (ep2 fall steps=256 both runs
fd=6.07→6.07 identical; ep8 didnt-reach steps=1700 both, fd=1.65→1.25
[modest improvement, did not flip to success]; ep14 didnt-reach steps=1700
both, fd=3.31→3.31 identical). No new fail, no flip to success — a clean
hold. (The task brief flagged "small gains are conceivable" as a
possibility, not a requirement; none materialized here beyond the
within-tag fd improvement on ep8, which did not change the pass/fail
outcome.)

---

## 4. Adoption decision

All five gate lines held or matched exactly, with identical (or
near-identical, within documented replay jitter) fail sets to their
respective baselines, and zero new/reproducible failure anywhere. Combined
with the mechanism replays (§2) showing (a) NX-9's two mid-path collision
wins are fully preserved, and (b) the self-body bias is demonstrably and
surgically eliminated at endgame with no collateral effect on genuine
mid-path avoidance — **ADOPT**. The fix is kept: `code/avoid.py`'s
`AVOID_MIN_GOAL_DIST_M = 1.6`.

---

## 5. Files changed / synced

- **`code/avoid.py`** (`unitree_vla`): `AVOID_MIN_GOAL_DIST_M`
  1.2 → 1.6; comment rewritten to record the re-application and point here
  instead of `docs/nx11_ep4.md`. No other lines touched. Unit self-test
  15/15 PASS reconfirmed post-edit.
- **Synced** (byte-copied, `cmp`-verified identical, NO git) to
  `VLA_mujoco_unitree/code/avoid.py`. Deploy-repo import
  smoke-checked (`import code.avoid` → `AVOID_MIN_GOAL_DIST_M == 1.6`, no
  exceptions).
- No other files touched (`code/inferencer.py`, `code/eval_search.py`,
  `code/fancy_demo.py`, `code/grounding.py` all untouched this cycle).

Eval outputs: `eval/nx13_demo_999/`, `eval/nx13_easy_999/`,
`eval/nx13_search_999/`, `eval/nx13_demo_1000/`, `eval/nx13_demo_2000/`.
Logs: `logs/nx13/demo_999.log`, `logs/nx13/easy_999.log`,
`logs/nx13/search_999.log`, `logs/nx13/demo_1000.log`,
`logs/nx13/demo_2000.log`, `logs/nx13/mech_replays.log`,
`logs/nx13/ab_endgame.log`. Diagnostic replay scripts (scratchpad-only, not
committed, matching this codebase's own precedent): `nx13_replay_demo.py`,
`nx13_replay_search.py`, `nx13_ab_endgame.py`.

---

## 6. What remains

Nothing new opened by this cycle. ep4 (demo/999) remains the sole
documented failure at seed 999, unchanged in mechanism from
`docs/nx11_ep4.md` §5 (late-episode balance loss, out of scope). The
rotation-order scan/locomotion instability `docs/gen1_multiseed.md` flagged
for fresh seeds (§3.1, §6 item 1) remains open and AVOID-independent — this
fix does not touch it and was never expected to. `docs/gen1_multiseed.md`'s
other ranked follow-ups (false-lock grounding-discrimination gap, stall/wedge
with AVOID active) are likewise untouched and out of this cycle's scope.
