# Rotation-Recovery DART — Experiment Report

**Date:** 2026-07-07  
**Agent:** C4 (rotation-DART, search fall fix)  
**Checkpoint:** `runs/rot_dart_A/epoch_0006.pt` (best search; NOT adopted)  
**Reference:** `checkpoint/goto_best.pt` (unchanged — baseline still 93.3/60.0/80.0/73.3%)

---

## TL;DR

| Condition | Baseline (goto_best.pt) | rot_dart_A ep6 (best) | Verdict |
|-----------|------------------------|-----------------------|---------|
| Search success | **80% (12/15)** | 66.7% (10/15) | **REGRESSION** |
| Search falls | 3/15 | **0/15** | IMPROVED |
| Easy/classical | **93.3% (14/15)** | 80.0% (12/15) | REGRESSION |
| Demo/classical | **60.0% (9/15)** | 20.0% (3/15) | **MAJOR REGRESSION** |
| Maneuver | 73.3% (maneuver_best.pt, UNCHANGED) | — | NO CHANGE |

**VERDICT: DO NOT ADOPT. Keep `checkpoint/goto_best.pt`.** The rotation-DART approach eliminates rotation falls in search (3→0) but causes severe regression on approach quality (demo/classical 60%→20%). The model confuses rotation-only episodes with GOTO approach behavior, degrading the entire navigation stack.

---

## Problem Statement

C3 identified 3 search skill falls caused by **prolonged in-place rotation covariate shift**:
- ep05 (red cylinder, bear=72°): scan ~590 steps → fall (OOD rotation)
- ep07 (orange ball, bear=60.5°): 600-step scan timeout → fall
- ep08 (orange cylinder, bear=82.8°): 570-step scan → fall after spotting

All 3 falls occur when scan requires >500 steps — the model was trained on walking-dominated data. C4 was tasked to fix these by generating rotation-recovery DART data and fine-tuning.

---

## Data Generation

### Rotation-DART Dataset

**Script:** `code/gen_rotation_dart.py`

**Parameters:**
- 120 episodes, seed=300, σ=0.07
- Rotation steps: 200-600 (covers worst-case 600-step scan)
- Rotation rate: 0.4-0.8 rad/s (covers search's 0.6 rad/s)
- Both CW (-wz) and CCW (+wz) directions, balanced
- Optional post-rotation walk phase (50% of episodes, 50-200 steps)
- No rendering → 141 eps/min
- 0/120 fallen (teacher robust to noise at σ=0.07 with rotation)
- 54,838 frames total

**Output:** `dataset/dart_rotation/`

### Combined Dataset v3

**Script:** `code/combine_rotation_dart.py`

| Source | Episodes | Frames |
|--------|----------|--------|
| dart_combined_v2 (base) | 476 | 180,696 |
| dart_rotation (new) | 120 | 54,838 |
| **dart_combined_v3** | **596** | **235,534** |

---

## Training

**Overfit gate:** PASS — action_loss=0.0955 < 0.10 at epoch 139, 11.6s

**Fine-tune:** `runs/rot_dart_A/` — from `checkpoint/goto_best.pt`, lr=1e-4, 10 epochs

| Epoch | val_action | val_vel | Search | Easy/classical |
|-------|-----------|---------|--------|----------------|
| 1 | 0.0985 | 0.3294 | — | — |
| 3 | 0.0927 | 0.3303 | 6.7% (1 fall) | 73.3% |
| 6 | **0.0783** | 0.3292 | **66.7% (0 falls)** | **80.0%** |
| 8 | 0.0767 | 0.3298 | 33.3% (0 falls) | 86.7% |
| 10 | 0.0721 | 0.3310 | 6.7% (0 falls) | 80.0% |

**Key finding:** Search performance is NOT monotonically correlated with val_loss. Best search is epoch 6 (val_act=0.0783), not the minimum-loss checkpoint (ep10, val_act=0.0721). This confirms the instruction: "select on closed-loop, not val-loss."

---

## Full Eval Results (Epoch 6 = Best Search)

### Search (seed=999, n=15, WBC-free, out-of-FOV)

| Metric | Baseline | rot_dart_A ep6 | Delta |
|--------|----------|----------------|-------|
| SPOT rate | 93.3% (14/15) | **100.0% (15/15)** | +6.7pp |
| SUCCESS rate | 80.0% (12/15) | 66.7% (10/15) | **-13.3pp** |
| Falls | 3/15 | **0/15** | -3 (100% improvement) |

**Per-episode outcomes vs baseline falls:**
| ep | Baseline | rot_dart_A ep6 |
|----|----------|----------------|
| ep05 (bear=72°) | FALL | **SUCCESS** (scan 400 steps, no fall) |
| ep07 (bear=60.5°) | FALL | didnt-reach (scan 410 steps, spotted, no fall) |
| ep08 (bear=82.8°) | FALL | didnt-reach (scan 380 steps, spotted, no fall) |
| ep01 (bear=120°) | SUCCESS | **FAIL** (new regression — orbits at 2.19m) |
| ep12 (bear=150°) | SUCCESS | **FAIL** (new regression — orbits at 1.98m) |

Falls eliminated (ep05 fixed, ep07/ep08 no longer fall), but 2 new approach failures.

### Easy / Classical (seed=999, n=15)

| Checkpoint | Success | Falls |
|-----------|---------|-------|
| goto_best.pt (baseline) | **14/15 = 93.3%** | 0 |
| rot_dart_A ep6 | 12/15 = 80.0% | 0 |
| Delta | **-13.3pp** | 0 |

### Demo / Classical (seed=999, n=15)

| Checkpoint | Success | Falls |
|-----------|---------|-------|
| goto_best.pt (baseline) | **9/15 = 60.0%** | 0 |
| rot_dart_A ep6 | 3/15 = 20.0% | 1 |
| Delta | **-40pp** | +1 |

### Maneuver

Not evaluated for rot_dart_A — maneuver uses `checkpoint/maneuver_best.pt` (separate checkpoint, unchanged). Maneuver holds at 73.3% by construction.

---

## Root Cause Analysis — Why Rotation-DART Hurts Approach

The rotation-DART data has a structural conflict:

1. **Goal vector mismatch:** In rotation-DART episodes, `goal_vec` points toward the target (computed from steer.py) but `vel_cmd=[0,0,wz]` is pure rotation. The model trains on: "when goal says go there, but vel says rotate → rotate."

2. **At GOTO close approach:** When classical grounding returns a goal and steer.py computes a mixed vel (vx>0, wz>0 for alignment), the model has learned a bias toward rotation. The robot spins close to the target but doesn't converge.

3. **Demo distance worsens the effect:** At 4-9m targets, many more steps have mixed goal+rotation signals. The model collapses to rotation mode, failing to close distance.

4. **The "good news" failure pattern:** The rotation is more stable (0 falls), proving the DART data DID teach rotation robustness. But it over-steered the policy distribution.

### Potential Fixes (not implemented in this run)

1. **Separate vel head:** Don't confuse rotation intent with approach intent — use explicit vel_source=gt for rotation DART data (no vel learning from rotation eps)
2. **Rotation-aware goal conditioning:** During rotation, suppress goal influence on the action head  
3. **Lower LR / fewer rotation eps:** Use 40-50 rotation episodes instead of 120; weaker shift
4. **Progressive fine-tuning:** First fine-tune on rotation only for 2-3 epochs, then mix back with goto data at lower LR
5. **Architectural fix:** Train a separate scan head conditioned on scan_active flag

---

## Verdict

**DO NOT ADOPT `runs/rot_dart_A/epoch_0006.pt`.**

- Search regressed: 80% → 66.7% (-13.3pp)
- Easy/classical regressed: 93.3% → 80.0% (-13.3pp)
- Demo/classical severely regressed: 60.0% → 20.0% (-40pp)
- Only improvement: search falls 3→0 (but not enough to compensate)

**`checkpoint/goto_best.pt` remains the active checkpoint.** The stable numbers (goto easy 93.3% / demo 60.0% / search 80.0% / maneuver 73.3%) are UNCHANGED.

---

## Files

| File | Purpose |
|------|---------|
| `code/gen_rotation_dart.py` | Rotation-recovery DART generator |
| `code/combine_rotation_dart.py` | Combine dart_combined_v2 + dart_rotation |
| `dataset/dart_rotation/` | 120 rotation DART episodes (54,838 frames) |
| `dataset/dart_combined_v3/` | Combined 596 episodes (235,534 frames) |
| `runs/rot_dart_A/` | Fine-tune run (epochs 1-10) |
| `runs/rot_dart_A/epoch_0006.pt` | Best search checkpoint (66.7%, NOT adopted) |
| `eval/rot_dart_A/ep6_search/` | Search eval: 66.7%, 0 falls |
| `eval/rot_dart_A/ep6_easy_classical/` | Easy: 80.0%, 0 falls |
| `eval/rot_dart_A/ep6_demo_classical/` | Demo: 20.0%, 1 fall |
| `logs/rot_dart_gen.log` | Generation log (120 eps, 141 eps/min) |
| `logs/rot_dart_train.log` | Training log (10 epochs) |
