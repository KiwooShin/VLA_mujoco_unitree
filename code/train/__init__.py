"""code.train — training entry points (RF-1).

Owner: G5. Modules:
  dart_phase        — GroundedNav training with DART data + gait-phase input (Fix 4+5).
  maneuver          — maneuver fine-tune from a locomotion checkpoint.
  gaitfix_loss      — GaitFixLoss (residual/standardized action loss), JOINT_NAMES.
  gaitfix_epoch     — shared train/val epoch runner + velocity-head audit.
  gaitfix           — GaitFix (Fix 1) CLI entry: overfit gate, full training, audit.
  nx6_heatmap       — NX-6 heatmap-detector training entry.
  nx6_heatmap_eval  — NX-6 heatmap-detector final evaluation suite.
"""
