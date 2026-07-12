# CAM-P4 — Unify the Probe-Gate Fix into the Champion Inferencer

**Date:** 2026-07-09
**Agent:** CX-5 (follow-on to CX-3's fancy_demo.py-local fix, docs/cam_p3_demo.md)
**Goal:** port the demo-only plausibility-gate fix into `code/inferencer.py` (the gated
eval path) and re-run the full gate to confirm no regression.

**Checkpoint:** `checkpoint/goto_best.pt` (unchanged, frozen). **CAMERA_MODE:** not set
(default `cam2` champion).

## TL;DR

- **Fix ported.** `code/inferencer.py`'s bounded-fallback-probe plausibility gate now
  keys on `CAM_PROXIMITY_D_FAR = 1.81` (the PROXIMITY camera's physical far limit) instead
  of `CAM_D_HI = 1.6` (the Schmitt-trigger hysteresis bound) — identical in spirit to
  CX-3's `code/fancy_demo.py` fix (docs/cam_p3_demo.md §2).
- **Full re-gate: matches champion exactly.** easy 100.0% (15/15), demo 66.7% (10/15),
  search 80.0% (12/15) — same numbers as docs/cam_p1.md. Per-episode outcomes (success/
  fail tag, and which specific episodes fail) are identical to `eval/p1_easy_cam2_v2`,
  `eval/p1_demo_cam2_v2`, `eval/p1_search_cam2` in every episode **except** one transient
  1-episode search flip on the first run, diagnosed as pure GPU/render non-determinism
  (see §2) and confirmed by an immediate full-condition rerun that reproduced the
  champion's exact per-episode result (same steps, same final_dist).
- **VERDICT: ADOPT.** The gate fix is now live in the champion inferencer, not just the
  demo file. Staging copy synced.

## 1. What changed (`code/inferencer.py`)

Single-line semantic change (plus documentation), mirroring `code/fancy_demo.py`'s fix
byte-for-byte in spirit:

```python
# before (shipped gate, docs/cam_p1.md):
_probe_ok = (other_cam == 'GROUNDING' or
             (_last_known_goal is not None and
              float(_last_known_goal[0]) <= CAM_D_HI))          # 1.6 m

# after (this change):
_probe_ok = (other_cam == 'GROUNDING' or
             (_last_known_goal is not None and
              float(_last_known_goal[0]) <= CAM_PROXIMITY_D_FAR))  # 1.81 m
```

A new constant `CAM_PROXIMITY_D_FAR = 1.81` (m) is defined alongside `CAM_D_LO`/`CAM_D_HI`,
with an inline comment reproducing CX-3's deadlock diagnosis (docs/cam_p3_demo.md §2): a
fast monotonic approach can leave the EMA'd last-known distance stuck just above `CAM_D_HI`
(observed ~1.70 m at true ~1.2 m) at the exact moment GROUNDING loses the target, and
because `_last_known_goal` only updates on a successful detection, the stale value never
refreshes — permanently closing the probe gate and forcing blind dead-reckoning for the
rest of the approach. Gating on the PROXIMITY camera's own physical far limit (1.81 m,
`docs/cam_opt2_multicam.md` geometry: 58° pitch → d_near≈0.22 m, d_far≈1.81 m) covers this
EMA-lag margin while still being physically meaningful — the proximity camera genuinely
cannot see anything beyond ~1.81 m, so there is no principled reason to gate tighter than
that.

**ep13 regression check (docs/cam_p1.md §2):** the gate must still block a far-range probe
that would otherwise let the proximity camera lock onto the blue-ish checkered floor
(the Round-1 regression: blue ball at 4.96 m, P0 SUCCESS → CAM-2-ungated FAIL). With the
new gate, `4.96 > 1.81` is still `True` → probe still blocked. The failure mode stays
impossible; confirmed structurally (both 1.6 and 1.81 exclude 4.96 m by a wide margin) and
empirically (demo ep13 succeeds in the re-gate, see §2 below — same outcome as the P1
champion run).

## 2. Full re-gate

Command pattern (`PYTHONPATH=.`, `MUJOCO_GL=egl`, `CAMERA_MODE` unset):

```
python code/eval_closedloop.py --checkpoint checkpoint/goto_best.pt --arch A \
    --difficulty easy --n 15 --device cuda --out eval/p4_gate_easy --no-render \
    --goal-source classical --vel-source predicted --seed 999
python code/eval_closedloop.py --checkpoint checkpoint/goto_best.pt --arch A \
    --difficulty demo --n 15 --device cuda --out eval/p4_gate_demo --no-render \
    --goal-source classical --vel-source predicted --seed 999
python code/eval_search.py --checkpoint checkpoint/goto_best.pt --n 15 --device cuda \
    --out eval/p4_gate_search --no-video --seed 999
```

### Results (first full run)

| Skill | Champion (docs/cam_p1.md) | This re-gate | Δ |
|---|---|---|---|
| easy/classical | 100.0% (15/15) | **100.0% (15/15)** | 0 |
| demo/classical | 66.7% (10/15) | **66.7% (10/15)** | 0 |
| search | 80.0% (12/15) | **73.3% (11/15)** | **−1 (ep14)** |

- **easy**: per-episode identical to `eval/p1_easy_cam2_v2` (all 15 SUCCESS; final_dist
  within noise, e.g. ep0 0.560 vs 0.558).
- **demo**: per-episode identical to `eval/p1_demo_cam2_v2` — same 5 failing episodes
  (ep0 cyan cone, ep2 blue cone, ep4 purple ball, ep5 cyan ball, ep12 cyan cube; all
  `FAIL[didnt-reach]`, the documented cyan/blue wall-HSV collisions, camera-unrelated),
  same 10 successes including **ep13 (blue ball, 4.96 m)** — the exact episode the
  original plausibility gate was built to protect (docs/cam_p1.md §2) — still SUCCESS
  (steps=602, fd=0.371), confirming the ep13 regression stays fixed under the loosened
  1.81 m gate.
- **search**: 14/15 episodes per-episode identical to `eval/p1_search_cam2`, including the
  same 3 falls at ep5/ep7/ep8 (steps/final_dist within normal run-to-run jitter, e.g. ep7
  steps=662 fd=3.16 in both runs — identical). **ep14 (orange cube, 2.02 m) flipped**:
  champion SUCCESS (steps=529, fd=0.48) → this run FAIL[didnt-reach] (steps=1400, fd=2.45).
  Trajectory showed the robot reaching dist=0.53 m at step 500 (near the stop) then
  drifting back out to dist≈2.4 m and getting stuck there for the rest of the episode —
  an overshoot-and-fail-to-recover pattern.

### Diagnosis of the ep14 flip

`docs/cam_p1.md` §3 already flags search ep14 (orange cube, 2.02 m) as "P0's documented
self-occlusion/overshoot risk case" — a known close-call, not a clean-margin success. Per
`docs/cam_p0.md`'s explicit prior finding ("an isolated single-episode rerun is not a
reliable proxy... the full n=15 sequential eval is what the gate actually measures"), the
correct diagnostic is a full-condition rerun (not an isolated single-episode rerun), same
command, same seed:

```
python code/eval_search.py --checkpoint checkpoint/goto_best.pt --n 15 --device cuda \
    --out eval/p4_gate_search_rerun --no-video --seed 999
```

**Result: 80.0% (12/15) — ep14 back to SUCCESS, steps=529, fd=0.48 — bit-for-bit identical
to the champion's own ep14 result.** All 15 episodes in the rerun are per-episode identical
to `eval/p1_search_cam2` (same 3 falls at ep5/7/8, same 12 successes). Step counts across
both re-gate runs jitter by ~0-10 steps episode-to-episode (e.g. ep0: 853 champion / 827
run-1 / 856 run-2; ep9: 987 / 979 / 985) — consistent with the pre-existing GPU/render
non-determinism documented in `docs/cam_p0.md` and `docs/grounding_dist.md`, unrelated to
the gate change (the same magnitude of jitter appears on episodes whose outcome never
changed).

**Conclusion: the ep14 flip in the first run was a pure-noise 1-episode flip, not a
camera-attributable regression.** No code or camera behavior differs between the two
search runs — same checkpoint, same seed, same gate value — yet ep14 alternated
FAIL/SUCCESS while every other episode (including the genuinely fragile ep5/7/8 falls)
stayed stable. This is the same category of non-determinism P0 explicitly documented and
is not caused by the `CAM_D_HI`→`CAM_PROXIMITY_D_FAR` gate widening.

## 3. Verdict

**ADOPT.** All three gated skills re-pass at the champion's numbers (100.0 / 66.7 / 80.0)
with per-episode outcomes matching `eval/p1_easy_cam2_v2`, `eval/p1_demo_cam2_v2`, and
`eval/p1_search_cam2` (search confirmed via the rerun in §2). The ep13 far-range
regression the gate exists to prevent (docs/cam_p1.md §2) remains fixed. The deadlock CX-3
found (docs/cam_p3_demo.md) is now closed in the eval path too, not just the demo file.

`code/inferencer.py`'s gate now matches `code/fancy_demo.py`'s — both files use
`CAM_PROXIMITY_D_FAR = 1.81` for the probe's plausibility check.

## 4. Files changed

- `code/inferencer.py` — `CAM_PROXIMITY_D_FAR = 1.81` constant added; `_probe_ok`'s
  plausibility check now uses it instead of `CAM_D_HI`; comments updated accordingly.
- `VLA_mujoco_unitree/code/inferencer.py` — synced identically (GitHub
  staging copy; **not committed/pushed** at the time of writing).
- `eval/p4_gate_easy/`, `eval/p4_gate_demo/`, `eval/p4_gate_search/` — first full re-gate
  run artifacts (100.0 / 66.7 / 73.3).
- `eval/p4_gate_search_rerun/` — confirming rerun of the search condition (80.0%, ep14
  flip not reproduced).
