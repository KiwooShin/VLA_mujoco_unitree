"""code.eval — closed-loop evaluation entry points (RF-1).

Owner: G5. Modules:
  closedloop            — goto-skill closed-loop evaluator (seed 999).
  search_types          — search-scene sampler + per-episode result schema + constants.
  search_rollout_state  — mutable rollout state + env/settle setup for the search rollout.
  search_rollout        — standalone search-skill rollout loop (scan -> spot -> goto).
  search                — search-skill evaluator entry (aggregation/reporting) + CLI.
  maneuver_types        — maneuver result schema, proprio builder, PD helper, video writer.
  maneuver_rollout      — standalone maneuver-skill rollout loop.
  maneuver              — maneuver-skill evaluator entry (aggregation/reporting) + CLI.
"""
