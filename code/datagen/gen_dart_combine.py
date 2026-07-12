"""
code/datagen/gen_dart_combine.py — Post-processing: add-phase and combine subcommands.

Role: split out of gen_dart_dataset.py (RF-1) — the two dataset-merge
utilities backing the `add-phase` and `combine` CLI subcommands. Pure
parquet/pandas manipulation; no mujoco physics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from code.datagen.gen_dart_phase import GaitPhaseTracker
from code.datagen.gen_dart_rollout import FPS, PROPRIO_DIM


# ---------------------------------------------------------------------------
# Add gait phase to clean dataset (post-processing existing parquet)
# ---------------------------------------------------------------------------
def add_phase_to_clean_dataset(in_dir: str, out_dir: str) -> int:
    """Adds a gait-phase column to an existing clean dataset.

    Reads each episode parquet from `in_dir`, computes gait phase from the
    proprio column, and writes a new parquet with a `phase` column to
    `out_dir`.

    Args:
        in_dir: Input dataset directory (data/chunk-000/*.parquet plus a
            meta/ directory).
        out_dir: Output dataset directory for the phase-augmented parquet
            files and copied meta files.

    Returns:
        Total number of frames processed.
    """
    import shutil

    in_path  = Path(in_dir)
    out_path = Path(out_dir)
    data_out = out_path / "data" / "chunk-000"
    meta_out = out_path / "meta"
    data_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    # Copy meta files
    for meta_f in (in_path / "meta").glob("*"):
        shutil.copy2(meta_f, meta_out / meta_f.name)
    print(f"[phase] Copied meta from {in_path}/meta → {meta_out}")

    total_frames = 0
    chunk_dir = in_path / "data" / "chunk-000"
    for parq_f in sorted(chunk_dir.glob("episode_*.parquet")):
        df = pd.read_parquet(parq_f)

        tracker = GaitPhaseTracker()
        phases = []
        for _, row in df.iterrows():
            q_lb = np.array(row["proprio"][:15], dtype=np.float32)
            sin_phi, cos_phi = tracker.update(q_lb)
            phases.append([float(sin_phi), float(cos_phi)])

        df["phase"] = phases
        out_parq = data_out / parq_f.name
        df.to_parquet(out_parq, index=False)
        total_frames += len(df)
        print(f"  [phase] {parq_f.name}: {len(df)} frames", flush=True)

    return total_frames


# ---------------------------------------------------------------------------
# Combine DART + clean datasets into a single merged parquet dataset
# ---------------------------------------------------------------------------
def combine_datasets(clean_dir: str, dart_dir: str, out_dir: str) -> dict:
    """Merges clean (with phase) and DART datasets into one combined dataset.

    Re-indexes `episode_index` and the global frame `index` across both
    sources.

    Args:
        clean_dir: Directory of the clean (phase-augmented) dataset.
        dart_dir: Directory of the DART dataset.
        out_dir: Output directory for the combined dataset.

    Returns:
        The `info` dict written to `meta/info.json`, describing dataset
        composition and stats.
    """
    out_path  = Path(out_dir)
    data_out  = out_path / "data" / "chunk-000"
    meta_out  = out_path / "meta"
    data_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    clean_chunk = Path(clean_dir) / "data" / "chunk-000"
    dart_chunk  = Path(dart_dir)  / "data" / "chunk-000"

    all_dfs = []

    # Load clean episodes
    for f in sorted(clean_chunk.glob("episode_*.parquet")):
        df = pd.read_parquet(f)
        # Ensure 'phase' column exists
        if "phase" not in df.columns:
            tracker = GaitPhaseTracker()
            phases = []
            for _, row in df.iterrows():
                q_lb = np.array(row["proprio"][:15], dtype=np.float32)
                sin_phi, cos_phi = tracker.update(q_lb)
                phases.append([float(sin_phi), float(cos_phi)])
            df["phase"] = phases
        all_dfs.append(df)

    n_clean = len(all_dfs)
    print(f"[combine] Loaded {n_clean} clean episodes from {clean_dir}")

    # Load DART episodes
    dart_count = 0
    for f in sorted(dart_chunk.glob("episode_*.parquet")):
        df = pd.read_parquet(f)
        all_dfs.append(df)
        dart_count += 1

    print(f"[combine] Loaded {dart_count} DART episodes from {dart_dir}")

    # Re-index episodes
    global_idx  = 0
    out_files   = []
    ep_meta_list = []
    tasks_map: dict[str, int] = {}
    all_actions = []
    all_proprio = []

    for ep_i, df in enumerate(all_dfs):
        df = df.copy()
        df["episode_index"] = ep_i
        df["index"]         = range(global_idx, global_idx + len(df))
        df["frame_index"]   = range(len(df))

        task_desc = str(df["task_description"].iloc[0])
        if task_desc not in tasks_map:
            tasks_map[task_desc] = len(tasks_map)
        df["task_index"] = tasks_map[task_desc]

        out_parq = data_out / f"episode_{ep_i:06d}.parquet"
        df.to_parquet(out_parq, index=False)
        out_files.append(f"data/chunk-000/episode_{ep_i:06d}.parquet")

        final_dist = float(df["goal"].iloc[-1][0]) if len(df) > 0 else 99.0
        reached    = bool(df["done"].iloc[-1] == 1) if len(df) > 0 else False
        ep_meta_list.append({
            "episode_index":   ep_i,
            "task_index":      tasks_map[task_desc],
            "length":          len(df),
            "success":         reached,
            "final_goal_dist": round(final_dist, 3),
            "is_dart":         ep_i >= n_clean,
            "tasks":           [task_desc],
        })

        all_actions.extend(df["action"].tolist())
        all_proprio.extend(df["proprio"].tolist())
        global_idx += len(df)

    # Stats
    arr_a = np.array(all_actions, dtype=np.float32) if all_actions else np.zeros((1, 15))
    arr_p = np.array(all_proprio, dtype=np.float32) if all_proprio else np.zeros((1, 55))

    def _stat(a: np.ndarray) -> dict[str, list[float]]:
        """Computes per-dimension mean/std/min/max for stats.json."""
        return {"mean": a.mean(0).tolist(), "std": (a.std(0) + 1e-6).tolist(),
                "min": a.min(0).tolist(), "max": a.max(0).tolist()}

    stats = {"proprio": _stat(arr_p), "action": _stat(arr_a)}

    info = {
        "codebase_version":  "dart+phase",
        "fps":               FPS,
        "robot":             "unitree_g1_lowerbody",
        "total_episodes":    len(all_dfs),
        "n_clean_episodes":  n_clean,
        "n_dart_episodes":   dart_count,
        "total_frames":      global_idx,
        "proprio_dim":       PROPRIO_DIM,
        "phase_dim":         2,
        "action_dim":        15,
    }

    tasks_list = [{"task_index": v, "task": k}
                  for k, v in sorted(tasks_map.items(), key=lambda x: x[1])]

    json.dump(info,  open(meta_out / "info.json",   "w"), indent=2)
    json.dump(stats, open(meta_out / "stats.json",  "w"), indent=2)
    with open(meta_out / "episodes.jsonl", "w") as f:
        for em in ep_meta_list:
            f.write(json.dumps(em) + "\n")
    with open(meta_out / "tasks.jsonl", "w") as f:
        for tm in tasks_list:
            f.write(json.dumps(tm) + "\n")
    with open(meta_out / "manifest.jsonl", "w") as f:
        for fp in out_files:
            f.write(json.dumps({"path": fp}) + "\n")

    print(f"[combine] Done: {len(all_dfs)} episodes, {global_idx} frames → {out_dir}")
    return info
