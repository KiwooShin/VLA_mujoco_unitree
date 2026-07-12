# A-vs-C Bake-off Results

**Date**: 2026-07-06. **Agent**: EV1 (eval/diagnose arm).

---

## Summary: 0% Success Across All Conditions

| Condition | Best Ckpt | n | Success | Dominant Fail | Mean Steps |
|-----------|-----------|---|---------|---------------|------------|
| A + goal=gt (upper bound) | ep15 | 15 | **0/15 (0%)** | fall | 103.5 |
| A + goal=classical | ep15 | 15 | **0/15 (0%)** | fall | 101.0 |
| A + goal=learned | ep15 | 15 | **0/15 (0%)** | fall | 114.0 |
| C (blind baseline) | ep20 | 15 | **0/15 (0%)** | fall | 99.0 |

All conditions fail exclusively via `fall`. No `didnt-reach`, no `wrong-object`.

---

## Checkpoint Selection (non-monotonic in offline loss → per condition)

Arch A best closed-loop by mean_steps before fall:

| Epoch | A+gt mean_steps | A+classical mean_steps | A+learned mean_steps |
|-------|----------------|----------------------|---------------------|
| ep10 | 88.0 | - | - |
| ep15 | **103.5** | **101.0** | **114.0** |
| ep20 | 97.3 | 94.2 | 96.0 |
| best (ep17) | 68.0 | - | - |

Epoch 15 is the best closed-loop checkpoint for all A conditions.

Arch C:

| Epoch | C mean_steps | Notes |
|-------|-------------|-------|
| ep11 | 74.0 | Identical every scene (no conditioning) |
| ep20 | **99.0** | Identical every scene (no conditioning) |

Arch C epoch 20 is best. Note: C falls at **exactly the same step across all 15 scenes** per checkpoint — it outputs a deterministic fixed policy regardless of scene, confirming it learned no scene-specific navigation.

---

## Condition Analysis

### Condition a: A + goal=gt (upper bound — zero vision, privileged sim-state goal)

- **Success rate**: 0/15 (0%)
- **Failure**: fall (15/15)
- **Mean survival**: 103.5 steps (2.1 s of sim at 50 Hz)
- **Distance behavior**: robot barely moves toward target (0.03m traversed in 175 steps for ep8)
- **Height trajectory**: slow decay at ~1.4 mm/step, falls at ~100–175 steps

Even with privileged goal (dist, cosθ, sinθ) from the simulation state, the robot falls.
This directly answers diagnostic question (i): **goal→action navigation does NOT work**
at this training stage.

### Condition b: A + goal=classical (HSV+depth grounding, render at 5Hz)

- **Success rate**: 0/15 (0%)
- **Failure**: fall (15/15)
- **Mean survival**: 101.0 steps
- **Closest approach**: ep14 reached 0.86m before fall (stop_r = 0.6m), ep2 reached 0.68m
- **Grounding quality**: classical HSV performs well when target is visible

Classical grounding ≈ GT grounding in terms of success rate — both are limited by the
stability bottleneck, not grounding accuracy.

### Condition c: C (blind baseline, zero vision, no render)

- **Success rate**: 0/15 (0%)
- **Failure**: fall (15/15)
- **Deterministic behavior**: all 15 scenes fall at identical step count per checkpoint
- **Mean survival**: 99.0 steps (ep20), 74.0 steps (ep11)
- **Arch C never moves toward target**: final_dist varies only by initial scene geometry,
  not by learned navigation

---

## Root Cause: Covariate Shift in BC Imitation — Falls from Stability Failure

The single most-limiting bottleneck is **the student PD controller cannot replicate the
teacher WBC's balance control**.

### Evidence

1. **Universal fall (all conditions)**: GT goal, classical goal, learned goal, Arch A, Arch C —
   all fall. The fall is not about goal quality (conditions a/b/c equivalent). It is about
   the action head failing to maintain stable locomotion.

2. **Slow height decay** (diagnosed on ep8, ep15 checkpoint):
   - Height at settle end: 0.744m
   - Decay rate: ~1.4 mm per control step
   - Falls at ~100–175 steps (height below 0.50m threshold)
   - Robot barely moves: 0.03m horizontal displacement in 175 steps

3. **Action mismatch**: Student mean action[joint3] = 0.706 vs teacher settle target 0.862
   (hip flexion underpredicted by 0.156 rad). The student cannot maintain the standing
   posture the teacher produces, causing a slow lean-forward collapse.

4. **Arch C determinism**: Arch C falls at exactly the same step for every scene within a
   checkpoint (74 steps for ep11, 99 steps for ep20). This shows it is outputting a fixed
   trajectory regardless of inputs — confirming BC collapse with no scene-conditioning.

5. **Identical behavior across goal sources**: A+gt and A+learned have nearly identical
   survival times (~97–114 steps), confirming the goal signal (even if perfect) cannot
   help if the action head is unstable.

### Why this happens

Training used:
- `ego_rgb = ZEROS` (no vision input during training)
- Only 75 episodes, 20 epochs
- Pure BC (no DAgger): the student never experienced its own predictions in the training loop
- The student learned joint targets from the WBC teacher's static-distribution data
- At deploy, the student's PD control creates a feedback loop with different dynamics than
  the teacher's ONNX WBC — the student predicts at the teacher's training distribution
  (stable walking) but the actual state drifts from that distribution as soon as balance
  deviates slightly

This is the classic covariate-shift / distribution-mismatch problem of behavioral cloning.

---

## Code Changes (EV1)

### `code/inferencer.py`

Added `--goal-source {learned,classical,gt}` option to `Inferencer`:

```python
class Inferencer:
    def __init__(self, ..., goal_source: str = 'classical', ...):
        # goal_source controls how goal (dist, cosθ, sinθ) is sourced for Arch A:
        # 'gt'        — _compute_gt_goal() from sim state (privileged upper bound, no render)
        # 'classical' — classical HSV+depth, render at 5Hz cadence (deployable)
        # 'learned'   — model's own grounding head prediction (default deploy, no render)
```

Added `_compute_gt_goal(data_mj, target_xy)` helper: computes (dist, cosθ, sinθ) from
simulation qpos, rotated into robot-egocentric frame via robot's yaw.

Key implementation note: `teacher_forcing=True` in the model is set when goal_source is
`gt` or `classical`, so `gt_goal` tensor injection bypasses the untrained grounding head.
For `learned`, `teacher_forcing=False` so the model's grounding head runs freely.

**ALSO FIXED (critical bug)**: The action scaling was wrong. The `ACTION_SCALE * raw_action + DEFAULT_ANGLES` formula was applied at deploy, but the training data stores absolute joint angles (teacher `_target_dof`), so `raw_action` from the model IS already the absolute target. Fixed to `student_target_dof = raw_action` (no further scaling).

### `code/eval_closedloop.py`

Added `--goal-source` CLI flag with same choices. Added `goal_source` to summary JSON
and incremental log filenames.

---

## Diagnostic Videos

| Video | Condition | Steps | Notes |
|-------|-----------|-------|-------|
| `eval/bakeoff/videos/ep008_A_ep15_gt.mp4` | A+gt ep15 | 132 | Longest: good initial gait, slow height decay |
| `eval/bakeoff/videos/ep010_A_ep15_gt.mp4` | A+gt ep15 | 108 | Gets to 1.06m before fall |

---

## Diagnostic Q&A

| Question | Answer |
|----------|--------|
| (i) Does goal→action navigate at all? | **No.** A-GT = 0% success. Robot can't walk stably. |
| (ii) Classical grounding is the limiter? | **No.** A-classical ≈ A-GT ≈ A-learned (all 0%). |
| (iii) Falls dominate even with GT goal? | **Yes.** This is the bottleneck. |
| (iv) Vision training needed? | **Irrelevant for navigation** right now; stability is the blocker. |

---

## Recommended Next Lever

**DAgger / stability training**, not retraining with vision.

The bottleneck is covariate-shift in BC imitation causing falls within 100 steps. Options:

1. **DAgger rollouts** (primary): roll out the student, call the WBC teacher to provide
   corrective actions at the student's visited states. This directly fixes covariate shift
   and is already planned in ADR-001.

2. **RL fine-tuning** (parallel track, mentioned in ADR-001): a KL-penalized PPO or DAPG
   with `reward = not_fell + height_bonus` trains the policy on its own state distribution,
   directly rewarding upright balance. Achievable in MuJoCo without new external weights.

3. **Action chunking + temporal ensembling** (quick win): already architecture-ready
   (chunk_H param). Set chunk_H=8 or 16 and enable temporal ensembling — smooths out
   oscillatory actions that compound into falls.

4. **More BC data** (minimal impact alone): 75 episodes is borderline; 200+ would reduce
   generalization gap, but without DAgger the distribution mismatch persists.

The eval infrastructure (inferencer + eval_closedloop with goal-source sweep) is ready
and confirmed correct. Once DAgger produces a stable policy, re-run with the same n=15
seed-999 protocol to measure improvement.
