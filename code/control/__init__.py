"""code.control — locomotion-adjacent control helpers (RF-1).

Shared, backend-agnostic control-law modules used by the various rollout
loops (inferencer.py, eval_search.py, fancy_demo.py, demo.py):

  - steer      : privileged egocentric steering control law (dist/yaw_err
                 -> velocity command).
  - scan_sched : NX-1 bidirectional bounded-rotation scan schedule.
  - avoid      : NX-9 local obstacle avoidance (yaw-rate bias from depth).

See docs/refactor_plan.md for the RF-1 package layout this belongs to.
"""
