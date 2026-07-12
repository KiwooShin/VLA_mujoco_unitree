# V6 — Proprio-Fed Velocity Head Experiment

**Date:** 2026-07-07
**Experiment:** V6 (exploratory)
**Hypothesis:** Feeding proprio embedding + gait-phase [sin,cos] to the velocity head
enables state-aware velocity (smoother decel/stop, steadier turns) and reduces
maneuver variance (seed-77 46.7% failure).
**Verdict:** NEGATIVE — KEEP v5 baseline (checkpoint/goto_best.pt unchanged).

---

## TL;DR

| Condition | Baseline (goto_best.pt) | V6 ep4 (best CL) | V6 ep7 (best val-act) |
|-----------|------------------------|------------------|-----------------------|
| easy/GT/predicted-vel | **100% (15/15)** | **0% (0/15)** | 0% (0/8) |
| easy/GT/GT-vel | 100% | 100% (5/5) | 75% (6/8) |
| val_action | 0.0877 (from demo_dart_A) | 0.0822 | 0.0782 |

**Key finding:** The proprio-fed vel head produces constant/biased wz (0.05-0.12 rad/s)
that steers the robot away from its target, causing 0% success. The action head is
intact at early epochs (100% with GT vel) but degrades by epoch 7 (75% with GT vel),
confirming that shared-model fine-tuning with the new vel head regresses performance.
**Do NOT adopt; keep checkpoint/goto_best.pt.**

---

## Architecture Change

### What was modified (code/small_vla.py)

`VelocityHead` modified to optionally accept `proprio_emb` (GRU hidden state, 128-d)
+ `phase` (last frame [sin,cos], 2-d) in addition to the original `(goal, vis, lang)`.

```python
# Old: vel = f(goal[3], vis[128], lang[128])  → in_dim=259
# New: vel = f(goal[3], vis[128], lang[128], proprio_emb[128], phase[2])  → in_dim=389
```

Controlled by `vel_proprio=True` flag in DEFAULTS and GroundedNav constructor.
When `vel_proprio=False` (default), old behavior is preserved — no regression for
existing checkpoints.

`GroundedNav.forward()` extracts gait phase from the last frame of `proprio_h[:, -1, -2:]`
(the last 2 dims of the 57-d proprio are [sin(phi), cos(phi)]).

`inferencer.py` updated to detect `vel_proprio=True` in checkpoint metadata and
instantiate the model with the correct architecture.

### What was NOT modified
- Residual action head (Fix 1) — unchanged
- GRU ProprioEncoder — unchanged
- ActionHead — unchanged
- GroundingHead — unchanged
- All other heads — unchanged
- Architecture Arch C — unchanged

---

## Training

**Dataset:** dart_combined_v2 (476 eps, 180,696 frames; same as demo_dart_A)
**Base checkpoint:** checkpoint/goto_best.pt (fine-tune from v5)
**Script:** code/train_velproprio.py
**Run dir:** runs/velproprio_A/

### Overfit gate: PASS

```
Action_loss=0.0974 < 0.10 target at epoch 138 (13.3s)
```

### Weight expansion strategy

The old velocity head input is 259-d; the new one is 389-d.
`train_velproprio.py` loads all non-vel-head weights from goto_best.pt (strict=False),
then expands `velocity.net.0.weight` from (128, 259) → (128, 389):
- Cols 0:259 = copied from goto_best.pt
- Cols 259:389 (proprio_emb + phase) = orthogonal-initialized

### Training curve (7 epochs, stopped on regression detection)

| Epoch | val_act | val_vel | Notes |
|-------|---------|---------|-------|
| 1 | 0.0955 | 0.0364 | |
| 2 | 0.0915 | 0.0328 | |
| 3 | 0.0890 | 0.0312 | |
| 4 | **0.0822** | 0.0303 | **best CL epoch (action head intact)** |
| 5 | 0.0834 | 0.0296 | |
| 6 | 0.0808 | 0.0289 | |
| 7 | **0.0782** | 0.0284 | **best val-act; action head degraded** |

Training stopped at epoch 7 after regression confirmed.

---

## Evaluation Results

### Easy / GT / Predicted Vel (primary metric)

| Checkpoint | n | Success | Failure modes |
|-----------|---|---------|---------------|
| Baseline (goto_best.pt) | 15 | **100%** | — |
| V6 epoch 4 | 15 | **0%** | 15 didnt-reach |
| V6 epoch 7 (smoke) | 8 | **0%** | 8 didnt-reach |

### Easy / GT / GT Vel (action head upper bound, isolates vel head)

| Checkpoint | n | Success | Notes |
|-----------|---|---------|-------|
| Baseline (goto_best.pt) | 15 | **100%** | reference |
| V6 epoch 4 | 5 | **100%** | action head intact, vel head is the failure |
| V6 epoch 7 | 8 | **75% (6/8)** | action head degrading from further training |

---

## Root Cause Analysis

### Finding 1: Vel head produces biased wz regardless of proprio

The vel head at epoch 5 outputs:
```
V6-ep5 (VARIED proprio/phase): vx ∈ [0.50, 0.54], wz ∈ [0.05, 0.12]
Training vel_cmd distribution:  vx mean=0.458 std=0.176, wz mean=0.021 std=0.139
```

The vel head outputs wz > 0 systematically, compared to the training mean of 0.021.
This positive wz bias causes constant leftward turning, steering the robot away from
its GT goal. After 600 steps, the robot has drifted far from the target.

At epoch 7, the vel head shows larger variation (wz ∈ [-0.17, +0.23]) as it learns to
modulate based on proprio — but the action head was not trained on these predicted vel
values (GT vel was teacher-forced during training), so the action head cannot respond
correctly to the vel_emb from the new vel head.

### Finding 2: Action head degrades with continued fine-tuning

With GT vel injected, performance drops:
- ep4: 100% (5/5) — action head intact
- ep7: 75% (6/8) — action head degrading

The reason: fine-tuning modifies the shared GRU and action feat projection using
the gradient signal from both the action loss AND the vel head prediction gradient.
The vel head prediction (with proprio) introduces a different gradient pattern than
the original model, causing the shared backbone weights to shift, degrading navigation.

### Finding 3: The constant-vel pathology persists even with proprio input

The T2/M1 diagnosis was: vel head outputs near-constant velocity (vel std ≈ 0)
because training uses GT vel teacher-forcing → the action head doesn't backpropagate
through vel_pred → the vel head optimizes by learning the mean vel_cmd.

With proprio added:
- The vel head now has MORE information, but the same incentive structure
- It still has the freedom to predict a constant (wz = E[wz] ≈ 0.021)
- Instead, it learns to map proprio features → wz in ways that minimize val_vel loss
  but this mapping is spurious (uses the wrong proprio features or wrong direction)
- The result is a non-constant but incorrect wz that regresses performance

### Why doesn't more proprio information fix the collapse?

The fundamental issue is the **teacher-forcing training objective**:
- At train time: action head sees `vel_proj(GT vel)` → learns correct action for GT vel
- At eval time: action head sees `vel_proj(pred vel)` → pred vel ≠ GT vel → wrong action

Adding proprio to the vel head doesn't break this train/eval mismatch. The vel head still
optimizes to predict the mean vel_cmd (val_vel=0.028 after 7 epochs). The proprio inputs
allow it to be SLIGHTLY non-constant, but the pattern it learns is wrong.

---

## What Would Be Needed to Fix This

1. **Train action head on vel_pred, not GT vel** (change teacher_forcing=False for vel head):
   - Risk: early training unstable (vel head random → action head gets wrong signal)
   - Need: curriculum where vel head is pre-trained first

2. **Decouple vel head from action head** (use vel_pred only at inference):
   - Change: train action head with GT vel (current), deploy with vel_pred
   - This is already the current setup — the issue is vel_pred is constant

3. **Use closed-loop vel supervision** (train vel head on outcomes, not vel_cmd labels):
   - RL-style: reward vel head based on whether robot reaches target
   - Much more complex; out of scope

4. **Keep hybrid-vel for maneuver** (current solution):
   - The maneuver skill already uses hybrid-vel (GT vel during TURN_PHASE)
   - This directly provides the FSM phase signal the vel head needs but can't learn

---

## VERDICT

**STRICT GATE: NEGATIVE. Do NOT adopt V6 checkpoint.**

- V6 regresses easy/GT from 100% → 0% (predicted vel) and to 75% (GT vel)
- No skill improved
- The root cause (teacher-forcing vel head collapse + action head co-degradation)
  is fundamental to the current training architecture
- Consistent with prior campaign pattern: shared-model architecture changes regress
  (BIG scale-up, flow head, frozen-GR00T-vision, learned-grounding, rotation-DART)

**All original checkpoints preserved:**
- checkpoint/goto_best.pt — UNCHANGED
- checkpoint/maneuver_best.pt — UNCHANGED

**Stable baseline numbers (unchanged):**
- goto easy/classical: 93.3% mean [86.7-100%] across 4 seeds
- goto demo/classical: 68.3% mean [60.0-80.0%] across 4 seeds
- search out-of-FOV: 75.0% mean [60.0-86.7%] across 4 seeds
- maneuver: 66.7% mean [46.7-73.3%] across 4 seeds

---

## Files Created / Modified

| File | Action | Purpose |
|------|--------|---------|
| `code/small_vla.py` | **Modified** | VelocityHead vel_proprio flag + GroundedNav wiring |
| `code/inferencer.py` | **Modified** | Detect vel_proprio flag in checkpoint metadata |
| `code/train_velproprio.py` | **Created** | V6 training script |
| `runs/velproprio_A/` | **Created** | V6 training run (epochs 1-7) |
| `eval/velproprio_A/` | **Created** | Diagnostic eval results |
| `docs/vel_proprio.md` | **Created** | This report |
| `checkpoint/goto_best.pt` | **UNCHANGED** | v5 baseline |
| `checkpoint/maneuver_best.pt` | **UNCHANGED** | v5 baseline |

---

## Code Changes Summary (small_vla.py)

The `vel_proprio=False` default means all existing code is backward-compatible.
No existing checkpoint loading breaks (the inferencer checks for `vel_proprio` key).

The change adds ~16K parameters to the vel head when `vel_proprio=True` but this
never reaches production — the original architecture is kept.

Backward compatibility: loading old checkpoints with `vel_proprio=False` (default)
continues to work exactly as before. The DEFAULTS dict now includes `vel_proprio=False`.
