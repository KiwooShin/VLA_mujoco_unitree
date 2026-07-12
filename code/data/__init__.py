"""code.data — dataset loaders (RF-1).

Houses the PyTorch `Dataset`/`DataLoader` layer consumed by the training
scripts under `code/train/`:

  - `dataset.py`           — SyntheticDataset, LeRobotDataset, ParquetDataset
                             (55-d proprio), plus `make_dataloader`.
  - `dataset_phase.py`     — PhaseParquetDataset (57-d proprio: base + gait
                             phase), plus `make_phase_dataloader`.
  - `dataset_maneuver.py`  — ManeuverParquetDataset (62-d proprio: base +
                             phase + maneuver conditioning), plus
                             `make_maneuver_dataloader`.

Moved from the flat `code/*.py` layout (RF-1); old import paths
(`code.dataset`, `code.dataset_phase`, `code.dataset_maneuver`) keep working
via sys.modules alias shims left in their place.
"""
