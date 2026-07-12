# NX-2 Lock-Management — Final Combination & Adoption

**Date:** 2026-07-09
**Agent:** NX-2 combiner (follow-on to `docs/nx2_impl.md` implementation and
`docs/nx2_iso.md` per-mechanism isolation gates)
**Checkpoint:** `checkpoint/goto_best.pt` (unchanged throughout — deploy-side only)
**CAMERA_MODE:** not set (default `cam2` champion)

## TL;DR

- **Adopted mechanisms: M1 (area-quality floor) + M3 (innovation gate +
  incumbent inertia).** Both isolation-verdicted **KEEP** in `docs/nx2_iso.md`
  (zero regressions, no target-episode fix — "defense-in-depth hygiene" per
  the design brief's own MEDIUM-confidence rank-#3/#2 classification).
- **Rejected: M2, M4, M5** (all isolation-verdicted **REJECT** — each broke a
  previously-100%-passing episode, ep13, without fixing its own target
  episode; see `docs/nx2_iso.md` for full root-cause traces).
- **Combined demo/classical gate (`eval/nx2_combined_demo`): 10/15 = 66.7%**,
  byte-identical failing set to baseline (eps 0, 2, 4, 5, 12) — matches the
  best isolation score exactly (M1 alone and M3 alone each also scored
  10/15 with the same failing set), so **no bisection was needed** (combined
  score was not below the best isolation score).
- **Cross-skill re-gate: zero regressions.** easy/classical 15/15 = 100.0%
  (identical to baseline, all 15 SUCCESS), search 14/15 = 93.3% (identical
  per-episode to baseline, same single fall at ep12).
- **Defaults flipped ON.** `LOCK_M1`/`LOCK_M3` are now default-on (opt-out via
  `LOCK_M1=0`/`LOCK_M3=0`) in `code/lock_mgmt.py`; `LOCK_M2`/`LOCK_M4`/`LOCK_M5`
  remain default-off (opt-in). A no-env-var confirm run reproduced the
  combined result exactly (10/15, same failing set) — **defaults-on ==
  combined result, confirmed.**
- **12/15 (80%) aim not reached.** ep0/ep2/ep4/ep5/ep12 all remain failing.
  Per `docs/nx2_iso.md`'s own traced root causes, none of M1/M3's mechanisms
  ever reach the code path that would fix any of these 5 episodes (M1: floor
  is mathematically below the false-positive blobs but can't be raised
  without cutting ep3's legitimate minimum; M3: the ep12 hijack is
  unconditionally accepted via the mandatory CAM-2 discontinuity carve-out,
  bypassing M3's gate entirely; M4, the mechanism actually targeted at
  ep0/ep2/ep5, was REJECTed for breaking ep13). This ceiling was reasoned
  about, not just observed — see `docs/nx2_iso.md`'s M1/M3 sections for the
  traced mechanics.

---

## 1. Combination logic

Per the isolation results (`docs/nx2_iso.md`):

| Mechanism | Verdict | Target ep(s) fixed? | Regression? |
|---|---|---|---|
| M1 (area-quality floor) | **KEEP** | No (0, 5 both unchanged) | None |
| M3 (innovation gate + incumbent inertia) | **KEEP** | No (12 unchanged) | None |
| M4 (divergence watchdog) | REJECT | No (2 unchanged/slightly worse) | **ep13 broken** |
| M2+M5 (N-of-M + bounded coast) | REJECT | No (0,2,4,5,12 all unchanged) | **ep13 broken** (isolated to M2; M5 alone clean, but M2 has no independent KEEP verdict of its own to carry forward) |

Only M1 and M3 carry a KEEP verdict, so the combined build enables
`LOCK_M1=1 LOCK_M3=1` only (M2/M4/M5 left off). No tuned constants were
adopted for M1/M3 — both isolation gates used shipped defaults
(`M1_AREA_FLOOR_PX2=100.0`; `M3_GATE_BEARING_DEG=25`,
`M3_GATE_BEARING_NEAR_MULT=1.5`, `M3_NEAR_RANGE_M=2.0`,
`M3_GATE_DIST_FLOOR_M=0.8`, `M3_INCUMBENT_MARGIN=1.3`, `M3_INCUMBENT_K=2`) and
neither's isolation gate attempted (or needed) a tune — both isolation
reports explicitly declined a tune-and-regate as already-falsified by
structural/mathematical analysis (`docs/nx2_iso.md` §M1 "Tuning considered,
not attempted", §M3 "Tuning: attempted as a quick single-episode probe, not
the official re-gate").

---

## 2. Combined demo/classical gate — `eval/nx2_combined_demo`

**Command** (`PYTHONPATH=.`, `MUJOCO_GL=egl`, `CAMERA_MODE` unset,
the `g1nav` conda env's python interpreter):

```
LOCK_M1=1 LOCK_M3=1 python code/eval_closedloop.py --checkpoint checkpoint/goto_best.pt \
    --arch A --difficulty demo --n 15 --device cuda --out eval/nx2_combined_demo \
    --no-render --goal-source classical --vel-source predicted --seed 999
```

1-episode smoke (ep0, `/tmp/.../scratchpad/nx2_combined_smoke`) ran crash-free
before the full 15-ep gate.

### Result: 10/15 = 66.7% — matches best isolation score exactly, no bisection needed

| ep | baseline (`eval/p4_gate_demo`) | combined M1+M3 (`eval/nx2_combined_demo`) | defaults-on confirm (`eval/nx2_defaults_confirm`) |
|---|---|---|---|
| 0 | FAIL[didnt-reach] fd=3.39 | FAIL[didnt-reach] fd=3.38 | FAIL[didnt-reach] fd=3.35 |
| 1 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | SUCCESS fd=0.36 |
| 2 | FAIL[didnt-reach] fd=10.62 | FAIL[didnt-reach] fd=10.80 | FAIL[didnt-reach] fd=10.50 |
| 3 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | SUCCESS fd=0.36 |
| 4 | FAIL[didnt-reach] fd=10.20 | FAIL[didnt-reach] fd=10.20 | FAIL[didnt-reach] fd=10.04 |
| 5 | FAIL[didnt-reach] fd=3.51 | FAIL[didnt-reach] fd=3.44 | FAIL[didnt-reach] fd=4.17 |
| 6 | SUCCESS fd=0.36 | SUCCESS fd=0.36 | SUCCESS fd=0.36 |
| 7 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | SUCCESS fd=0.37 |
| 8 | SUCCESS fd=0.36 | SUCCESS fd=0.37 | SUCCESS fd=0.37 |
| 9 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | SUCCESS fd=0.36 |
| 10 | SUCCESS fd=0.37 | SUCCESS fd=0.36 | SUCCESS fd=0.37 |
| 11 | SUCCESS fd=0.36 | SUCCESS fd=0.37 | SUCCESS fd=0.38 |
| 12 | FAIL[didnt-reach] fd=6.08 | FAIL[didnt-reach] fd=6.70 | FAIL[didnt-reach] fd=6.25 |
| 13 | SUCCESS fd=0.37 | SUCCESS fd=0.37 | SUCCESS fd=0.37 |
| 14 | SUCCESS fd=0.36 | SUCCESS fd=0.37 | SUCCESS fd=0.36 |

Same 5 failing episodes as baseline and as each individual M1/M3 isolation
run (0, 2, 4, 5, 12); same 10 passing episodes. Zero flips in either
direction, zero interaction effects observed between M1 and M3 (both
independently already null on this suite, and remain null combined — no
"M3 delays M1" or similar surprise). Step/final_dist deltas are all within
this codebase's documented run-to-run jitter (`docs/nx1_scan.md` §5).

**Combined score (10/15) is not below the best individual isolation score
(10/15 for both M1 and M3 alone) — per the task's bisection trigger
condition, no bisection was required.**

---

## 3. Cross-skill re-gate

### 3.1 easy/classical — `eval/nx2_combined_easy`

```
LOCK_M1=1 LOCK_M3=1 python code/eval_closedloop.py --checkpoint checkpoint/goto_best.pt \
    --arch A --difficulty easy --n 15 --device cuda --out eval/nx2_combined_easy \
    --no-render --goal-source classical --vel-source predicted --seed 999
```

1-episode smoke (`/tmp/.../scratchpad/nx2_combined_easy_smoke`) crash-free
before the full run.

**Result: 15/15 = 100.0%** — identical to baseline (`eval/p4_gate_easy`,
15/15). All 15 episodes SUCCESS in both; final_dist deltas ≤0.005m
(e.g. ep0 0.560→0.558m), step-count deltas ≤5 steps — normal jitter, zero
tag flips. **No camera-attributable regression.**

### 3.2 search — `eval/nx2_combined_search`

```
LOCK_M1=1 LOCK_M3=1 python code/eval_search.py --checkpoint checkpoint/goto_best.pt \
    --n 15 --device cuda --out eval/nx2_combined_search --no-video --seed 999
```

1-episode smoke (`/tmp/.../scratchpad/nx2_combined_search_smoke`) crash-free
before the full run.

**Result: 14/15 = 93.3%** — identical to baseline (`eval/nx1_search_gate_v2`,
14/15 per `docs/nx1_scan.md`). Same single fall at **ep12** (red cube,
2.42m — the same marginal-success-turned-fall `docs/nx1_scan.md` §3.3
already root-caused as a target-visibility/final-approach-geometry issue
unrelated to M1/M3's association-gating logic). All 14 other episodes
SUCCESS in both, final_dist within ~0.02m, steps within ~10-90 (normal
render/physics jitter magnitude for this suite, e.g. ep0 1473→1464,
ep13 1526→1495). **No camera-attributable regression; no single-flip
noise-class rerun needed since the per-episode pattern already matched
baseline exactly on the first run** (`docs/cam_p0.md`'s rerun-once protocol
only applies when an unexpected flip is observed — none was).

### Cross-skill summary

| Skill | Baseline | Combined M1+M3 | Δ |
|---|---|---|---|
| demo/classical | 66.7% (10/15) | **66.7% (10/15)** | 0, same failing set |
| easy/classical | 100.0% (15/15) | **100.0% (15/15)** | 0 |
| search | 93.3% (14/15) | **93.3% (14/15)** | 0, same ep12 fall |

---

## 4. Final adoption — defaults flipped ON

`code/lock_mgmt.py`'s toggle block changed from all-5-default-OFF to:

```python
def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() == "1"

# M1/M3: KEEP-verdicted -> default ON, opt-out via LOCK_M1=0 / LOCK_M3=0.
LOCK_M1 = _env_flag("LOCK_M1", default="1")   # area-quality floor
LOCK_M3 = _env_flag("LOCK_M3", default="1")   # innovation gate + incumbent inertia
# M2/M4/M5: REJECT-verdicted -> stay default OFF, opt-in.
LOCK_M2 = _env_flag("LOCK_M2")   # N-of-M tentative->confirmed
LOCK_M4 = _env_flag("LOCK_M4")   # divergence watchdog
LOCK_M5 = _env_flag("LOCK_M5")   # bounded coast -> reroute to rescan
```

Setting all five explicitly to `LOCK_M1=0 LOCK_M2=0 LOCK_M3=0 LOCK_M4=0
LOCK_M5=0` still reproduces the pre-NX-2 byte-identical legacy pass-through
documented in `docs/nx2_impl.md` §4(a) — nothing about the old all-off
behavior was removed, only the *default* changed. Comments in
`code/lock_mgmt.py`, `code/inferencer.py`, and `code/eval_search.py` were
updated to describe the new default state accurately (previously said
"default OFF" for all five; now M1/M3 documented as default-ON/opt-out).

### Defaults-on confirm — `eval/nx2_defaults_confirm`

Plain command, **no env vars set** (relying purely on the new code defaults):

```
python code/eval_closedloop.py --checkpoint checkpoint/goto_best.pt --arch A \
    --difficulty demo --n 15 --device cuda --out eval/nx2_defaults_confirm \
    --no-render --goal-source classical --vel-source predicted --seed 999
```

**Result: 10/15 = 66.7%, same 5 failing episodes (0, 2, 4, 5, 12) as the
explicit `LOCK_M1=1 LOCK_M3=1` combined run** — confirmed at the top of §2's
table above (all three columns — baseline, explicit-toggle combined, and
no-env-var defaults-on — show the identical pass/fail pattern). Sanity-
checked in-process too:

```
$ python -c "import code.lock_mgmt as lm; print(lm.LOCK_M1, lm.LOCK_M2, lm.LOCK_M3, lm.LOCK_M4, lm.LOCK_M5)"
True False True False False
```

**Defaults-on confirmed equal to the combined result.**

---

## 5. Files changed / synced

### Changed in `unitree_vla` (source of truth)

- `code/lock_mgmt.py` — toggle defaults: `LOCK_M1`/`LOCK_M3` now default
  `"1"` (opt-out), `LOCK_M2`/`LOCK_M4`/`LOCK_M5` unchanged at default `"0"`
  (opt-in). Module docstring and two in-method docstrings updated to
  describe the new default state; mechanism logic itself is **byte-for-byte
  unchanged** — only the env-var default values changed.
- `code/inferencer.py`, `code/eval_search.py` — one comment each updated
  ("default OFF" → describes the new M1/M3-on/M2,4,5-off split); no logic
  changes (these files already had the NX-2 gate-wiring from
  `docs/nx2_impl.md`, untouched here).
- `docs/nx2_final.md` (this file).

### Synced to `VLA_mujoco_unitree/code/` (byte-copy, **not
committed/pushed** at the time of writing)

- `code/lock_mgmt.py` (**new file** — staging had no prior copy)
- `code/scan_sched.py` (**new file** — staging had no prior copy; required
  because `lock_mgmt.py` hard-imports `BidirectionalScanSchedule`/
  `SCAN_LEG_DEG`/`SCAN_DWELL_STEPS`/`SCAN_TIMEOUT` from it at module load
  time. Synced as a functional dependency of this closure's own changed
  files, not because NX-1's own work is otherwise being re-synced here —
  without it, importing the synced `lock_mgmt.py`/`eval_search.py` in
  staging would raise `ModuleNotFoundError`.)
- `code/grounding.py` — brought current; the staging copy was missing only
  the additive `best_area` field (verified via diff before syncing: the
  staging copy otherwise already matched, i.e. this is exactly and only
  NX-2's grounding.py change, nothing else riding along).
- `code/inferencer.py`, `code/eval_search.py` — brought current (both
  contain the NX-2 `LockGate`/`ReacquisitionScan` wiring from
  `docs/nx2_impl.md`, now with the updated toggle-default comment from this
  session). **Note:** staging's prior copies of these two files
  pre-dated NX-1's `scan_sched.py` integration as well (a larger diff than
  NX-2 alone) — this sync brings them fully current in one step rather than
  leaving a partially-synced, non-importable intermediate state.
- Not synced (unchanged by NX-2, out of scope): `code/demo.py`,
  `code/fancy_demo.py`, `code/steer.py`, `code/teacher.py` — per
  `docs/nx2_impl.md` §5, these were "not modified" by NX-2 and are left as
  the responsibility of whichever task owns their own sync.
- No one-off bench/debug scripts copied (diagnostic scripts referenced in
  `docs/nx2_impl.md`/`docs/nx2_iso.md` all live in scratchpad, never
  committed to `unitree_vla/code/` in the first place).

---

## 6. Verdict

**ADOPT M1 + M3, default ON.** Both are net-neutral-to-positive by the only
bar this closure could hold them to: zero regressions across all three
gated skills (demo 10/15, easy 15/15, search 14/15 — all identical to
pre-NX-2 baseline), plus the documented hygiene value against more
degenerate false-positive blobs / gradual mis-associations than this
specific 15-episode demo suite happens to exercise. **REJECT M2, M4, M5** —
excluded from the shipped defaults per their isolation verdicts; still
available opt-in via their own env vars for anyone who wants to
experiment further (e.g. pairing a loosened M4 trigger with M2's
confirm-gate, as `docs/nx2_iso.md`'s M4 section speculates could be
net-positive — out of scope for this closure).

**The >=12/15 (80%) demo aim was not reached.** ep0/2/4/5/12 remain failing
under the adopted mechanisms — this is a traced, structural ceiling (see
`docs/nx2_iso.md`'s M1/M3 sections), not a missed tuning opportunity:
M1's floor cannot be raised without cutting ep3's legitimate minimum; M3's
gate is unconditionally bypassed at ep12's exact failure moment by the
mandatory CAM-2 discontinuity carve-out (itself required to prevent a
different, already-fought regression class); and M4 — the mechanism
actually aimed at the ep0/ep2/ep5 false-lock family — was REJECTed because
it broke ep13 without fixing its own targets. Closing that gap would
require either a smarter CAM-2 handoff corroboration step (flagged as
future work in `docs/nx2_iso.md`'s M3 section) or a composite M2+M4 design
(flagged in the same doc's M4 section) — both explicitly out of scope for
this combination/adoption closure.
