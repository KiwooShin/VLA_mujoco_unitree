# GaitFix Experiment — Full Report

**Date:** 2026-07-06  
**Run:** `runs/gaitfix_A`  
**Baseline:** First bakeoff (docs/bakeoff.md) — all conditions 0% success, gait osc std ~0.002 rad, fwd 0.03 m, ~100 steps

---

## Root Cause (from R6 research)

BC on **absolute joint targets** mode-averages the gait to the mean pose: tiny offline L1 loss, yet zero oscillation in deployment → robot stands but falls quickly. The fix: train in **residual / standardized delta** space so the distribution the model must learn has unit variance per joint.

---

## Fixes Implemented

### Fix 1 — Residual + Standardized Action Target

- Compute per-joint delta from default: `delta = target_dof - default_angles`
- Normalize over training set: `normed = (delta - mean) / std`  
- Model predicts in normalized space; loss = smooth-L1 on normalized delta
- Swing joints (r_ankle_pitch, l_ankle_pitch, r_knee, l_knee, r_hip_pitch) upweighted 2× in loss
- Deploy: `target_dof = default_angles + pred * std + mean` (exact de-normalization)
- Action stats serialized into every checkpoint under `action_stats` key

**Per-joint statistics (14591 training frames):**

| Joint | Std (rad) | Swing upweight |
|-------|-----------|----------------|
| r_ankle_pitch | 0.2896 | 2× |
| l_ankle_pitch | 0.2823 | 2× |
| r_knee | 0.2154 | 2× |
| l_knee | 0.2042 | 2× |
| r_hip_pitch | 0.1242 | 2× |
| (others) | 0.03–0.12 | 1× |

### Fix 2 — Velocity Head Audit + GT-Velocity Eval Mode

- Audit: compare vel head predictions vs GT teacher velocity on val set
- Eval mode B: inject privileged (vx, vy, ωz) from `steer.py` into model forward pass each step

---

## Training

- Arch A, vision OFF (zeros), easy dataset (80 eps, 72 train / 8 val)
- 20 epochs, batch=64, lr=3e-4, swing_weight=2.0
- ~10 min total (29–31 s/epoch on GPU)
- Per-epoch checkpoints: `runs/gaitfix_A/epoch_NNNN.pt`
- Best checkpoint: `runs/gaitfix_A/model_best.pt` = epoch 19/20

**Training curve:**

| Epoch | val_action | Notes |
|-------|-----------|-------|
| 1 | 0.1319 | |
| 4 | 0.0563 | large jump |
| 7 | 0.0381 | |
| 9 | 0.0298 | |
| 12 | 0.0203 | |
| 15 | 0.0128 | |
| 19 | 0.0075 | best |
| 20 | 0.0075 | (tied best) |

Val loss continuously decreasing — no overfitting sign at 20 epochs with ~14k frames.

---

## Fix 2: Velocity Head Audit

Evaluated `model_best.pt` on val set (1416 samples):

| Metric | Prediction | GT |
|--------|-----------|-----|
| vx mean | 0.517 | 0.471 |
| vx std | **0.000** | 0.163 |
| wz mean | 0.024 | 0.071 |
| wz std | **0.000** | 0.106 |
| MAE vx | 0.092 | — |

**Interpretation:** The vel head learned approximately correct mean forward speed (~0.52 m/s, GT ~0.47 m/s) but predicts a **constant** — zero variance. It cannot modulate based on state (distance to target, turning angle, etc.). This means vel prediction is informationally equivalent to injecting a fixed forward command every step.

`vel_head_near_zero: False` — the head is not predicting zero, but it predicts a constant.

---

## Closed-Loop Eval Results

**Protocol:** easy difficulty, seed=999 (held-out), n=15 scenes, no render, maxsteps=600.

### Full Sweep Table

| Condition | Ckpt | Steps | Fwd Disp (m) | Osc Std (rad) | Success |
|-----------|------|-------|--------------|---------------|---------|
| baseline (bakeoff ep15) | — | ~100 | 0.030 | ~0.002 | 0% |
| a (vel=pred) | best | **107.0** | 0.311 | 0.094 | 0% |
| b (vel=gt) | best | 97.4 | 0.378 | 0.118 | 0% |
| a (vel=pred) | ep07 | 79.5 | **0.581** | 0.135 | 0% |
| b (vel=gt) | ep07 | 66.3 | 0.216 | 0.138 | 0% |
| a (vel=pred) | ep12 | 96.1 | 0.283 | 0.115 | 0% |
| b (vel=gt) | ep12 | 82.8 | 0.492 | 0.125 | 0% |
| a (vel=pred) | ep15 | 87.6 | 0.376 | 0.095 | 0% |
| b (vel=gt) | ep15 | 82.9 | 0.471 | 0.110 | 0% |
| a (vel=pred) | ep20 | 107.0 | 0.311 | 0.094 | 0% |
| b (vel=gt) | ep20 | 97.4 | 0.378 | 0.118 | 0% |

Note: ep20 and model_best.pt are identical (ep19 was model_best, ep20 tied and also saved).

### Key Observations

1. **Gait oscillation revived.** Osc std went from ~0.002 rad to 0.094–0.138 rad — a **47–69× increase**. The robot is now commanding alternating joint targets (actual gait cycles), not a static pose.

2. **Forward displacement 10–19× better.** Baseline 0.03 m → best conditions 0.31–0.58 m. The robot moves meaningfully before falling.

3. **Survival steps roughly flat.** Best checkpoint (ep15/best) achieves 107 steps, same order as baseline 100. The robot moves more within those steps, but still falls at roughly the same time.

4. **Non-monotonic across epochs.** Lower val loss (ep20: 0.0075) gives higher survival (107 steps) but less forward displacement than ep07 (0.581 m with 79.5 steps). ep07 has higher oscillation amplitude (0.135 rad) and moves faster but falls sooner — the gait is less stable at ep07, more stable but less aggressive at ep20.

5. **GT velocity (condition b) does not help survival.** With GT-vel injected, osc std increases slightly but survival steps decrease vs condition a. The vel head's constant prediction (~0.52 m/s) is close enough to GT (~0.47 m/s) that injecting GT vel doesn't fix the fundamental problem; in fact it sometimes pushes harder and destabilizes earlier.

6. **All failures are falls.** 15/15 = fall in every condition. The robot falls, never gets close enough to count as "didn't reach" — it moves forward but topples before reaching the target.

---

## What Fixed and What Remains

### Fixed by Fix 1
- Static collapse: eliminated. Robot commands oscillating joint targets.
- Mode-averaging problem: resolved by residual+standardized target.
- Forward progress: restored — robot moves 0.3–0.6 m before falling.

### Remaining Bottleneck: Covariate Shift (BC)
The robot is gait-oscillating and moving forward, but falls at ~100 steps in all conditions. This is **covariate shift** — not mode averaging.

The BC student was trained only on teacher rollouts (on-policy for the teacher). At deployment, small errors in joint targets compound: after 50–100 steps the robot reaches a joint/balance state that was rarely/never in the training distribution, and the model outputs are uncalibrated in that region → topple.

Evidence:
- Higher osc amplitude (ep07) → faster falls (79.5 steps) — more aggressive gait but less stable under compounding error
- Lower osc amplitude (ep20) → longer survival (107 steps) — more conservative gait recovers slightly from drift
- GT vel injection doesn't help — the bottleneck is not velocity magnitude, it's the off-distribution state recovery

---

## Recommended Next Step: DAgger (or RSI)

The next iteration must fix covariate shift. Two options:

**Option 1 — DAgger (preferred)**
Generate on-policy student trajectories, label with teacher actions at each visited state, add to training set, retrain. This directly covers the states where the student makes errors.

**Option 2 — RSI (Reference State Initialization)**
During training, initialize rollouts from random teacher trajectory states (not just episode start), forcing the model to learn recovery from mid-gait states. Cheaper than full DAgger; does not require online rollout labeling.

Both options are explicitly noted as "NEXT iteration" per task brief. Neither requires architecture changes.

---

## Files

| File | Purpose |
|------|---------|
| `code/action_stats.py` | Compute per-joint delta statistics over parquet dataset |
| `code/train_gaitfix.py` | Fix 1+2 training script (residual targets, vel audit) |
| `code/eval_gaitfix.py` | Eval sweep across checkpoints and conditions a/b |
| `code/inferencer.py` | Updated: Fix1 de-norm, Fix2 GT vel injection, osc/disp tracking |
| `code/eval_closedloop.py` | Updated: vel_source param, new metrics in output |
| `runs/gaitfix_A/model_best.pt` | Best checkpoint (val_action=0.0075, ep19) |
| `runs/gaitfix_A/action_stats.json` | Per-joint delta mean/std for de-normalization |
| `eval/gaitfix/gaitfix_results.json` | Full 10-condition eval results |
| `eval/gaitfix/vel_audit.json` | Velocity head audit stats |
