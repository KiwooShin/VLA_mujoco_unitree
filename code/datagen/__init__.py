"""code.datagen — dataset generators (RF-1).

Houses the deterministic/DART simulation-rollout generators that produce the
parquet+mp4 datasets consumed by `code/data/`:

  - `gen_dataset.py`          — clean teacher rollout generator (easy/demo).
  - `gen_dart_dataset.py`     — DART generator CLI (generate/add-phase/
                                combine subcommands); phase tracking, DART
                                rollout, and dataset-merge logic live in
                                sibling `gen_dart_*` modules.
  - `gen_maneuver_dataset.py` — DART generator for the maneuver skill.
  - `gen_det_dataset.py`      — NX-6 labeled object-detection dataset
                                generator CLI; segmentation/label/scene
                                logic lives in sibling `gen_det_*` modules.
  - `gen_stand_keyframe.py`   — one-shot WBC settle -> stand_keyframe.npz.

Moved from the flat `code/*.py` layout (RF-1); old import paths and CLI
entry points (`python code/gen_*.py ...`) keep working via alias + thin
shim files left in their place.
"""
