"""code.policy — GroundedNav student model + supporting policy utilities (RF-1).

  - small_vla    : GroundedNav student (Architecture A modular / C baseline),
                   split into blocks / heads / model submodules.
  - action_stats : per-joint action delta statistics (Fix-1 gait-fix
                   normalization), computed over the training set.
  - groot_lang   : GR00T-N1.6 language embedding cache builder.

See docs/refactor_plan.md for the RF-1 package layout this belongs to.
"""
