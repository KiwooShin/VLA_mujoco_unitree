"""code.runtime — closed-loop deploy rollout harness (RF-1).

RF-1 split of the flat code/inferencer.py (1789 lines, docs/refactor_plan.md,
group RF-2b) into:

  constants        — module-level constants + env toggles (FALL_HEIGHT,
                      GROUNDING_PERIOD, STALL_BREAK/AVOID toggles, ...).
  gait_phase       — `_GaitPhaseTracker` (Fix 4 gait phase input).
  helpers          — pure per-step helpers (`_build_proprio`,
                      `_apply_student_pd`, `_rgb_to_tensor`, `_label_active_cam`).
  gt_goal          — `_compute_gt_goal` (privileged GT goal probe).
  io               — checkpoint/model loading, `RolloutResult`, `_write_video`.
  goal_config      — tunable constants for goal_pipeline.
  goal_pipeline    — `GoalPipeline`: grounding-cycle/goal/EMA/hold/handoff/
                      scan state machine (NX-1/NX-2/NX-5/NX-9/NX-10).
  rollout_state    — per-episode env/settle setup + mutable state bundle.
  rollout_step     — the per-step control loop body.
  inferencer       — `Inferencer` facade re-assembling all of the above;
                      the old top-level `code/inferencer.py` is a
                      sys.modules alias to `code.runtime.inferencer`.

See code/runtime/inferencer.py's module docstring for the full deploy-loop
design (ADR-001, three-rate pipeline, goal_source semantics).
"""
