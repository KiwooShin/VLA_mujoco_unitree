"""code/perception/ — grounding + target-lock + learned-detector package (RF-1).

RF-1 split of the flat code/grounding.py, code/lock_mgmt.py,
code/nx6_heatmap_model.py, code/nx6_heatmap_data.py, and
code/nx6_heatmap_eval_utils.py (docs/refactor_plan.md, group G2-perception).

Submodules:
  types.py           GroundingResult dataclass (shared result contract).
  geometry.py         Camera intrinsics + camera-frame -> egocentric transform.
  hsv_config.py       HSV colour bounds + classical-pipeline threshold/gate constants.
  hsv_size_gate.py    NX-3 physical-size plausibility gate (M6).
  hsv_depth_split.py  NX-4 depth-guided component splitting + CAM-2 depth-outlier rejection.
  hsv_shape_score.py  V5 shape-discrimination blob scoring.
  hsv_pipeline.py     ground_classical(): the classical HSV+depth detection pipeline.
  ground_net.py       GROUND_NET learned-detector backend (state-parameterized).
  grounding.py        ground() dispatch — owns all mutable module state.
  lock_gate.py        LockGate state machine + M1-M7 toggles/constants.
  lock_rescan.py      ReacquisitionScan bounded-rescan wrapper.
  lock_mgmt.py        Aggregator re-exporting lock_gate.py + lock_rescan.py.
  detector/           NX-6 heatmap detector model/data/eval_utils.

Every module here keeps a compat alias at its original flat `code/<name>.py`
path (sys.modules pattern, docs/refactor_plan.md) — old imports keep working
unchanged and observe the exact same module state as the new paths.
"""
