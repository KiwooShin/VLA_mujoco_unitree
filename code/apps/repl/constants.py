"""Shared constants, checkpoint paths, and the GR00T-LM lang-embedding cache
for the REPL demo app (code/demo.py, RF-1 split).

Every other module in code/apps/repl/ imports from here — this is the single
place that:
  - bootstraps sys.path / the MUJOCO_GL env vars exactly as the original
    demo.py did at import time (module state coherence: this must run once,
    from one place),
  - owns the checkpoint-path resolution logic (pinned checkpoint/ dir vs.
    runs/ fallback),
  - owns the lazy-loaded GR00T-LM instruction embedding cache (a
    module-level singleton — see docs/demo.md), and
  - owns the small color/shape vocabularies the rule-based Planner matches
    against.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np


# ---------------------------------------------------------------------------
# CUDA helper (defined before use)
# ---------------------------------------------------------------------------
def _check_cuda() -> bool:
    """Return True if CUDA is available, False otherwise (or if torch import fails)."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOTO_CKPT_EP3  = _REPO / "runs/demo_dart_A/epoch_0003.pt"
GOTO_CKPT_EP5  = _REPO / "runs/demo_dart_A/epoch_0005.pt"
GOTO_CKPT_BEST = _REPO / "runs/demo_dart_A/model_best.pt"
MANEUVER_CKPT  = _REPO / "runs/maneuver_A/epoch_0002.pt"
LANG_CACHE     = _REPO / "dataset/lang_cache.pkl"

# H1 refresh: use pinned stable checkpoints in checkpoint/ dir (demo_dart_A ep3 +
# maneuver_A ep2). Fallback to runs/ paths if checkpoint/ dir not yet populated.
_GOTO_CKPT_PINNED     = _REPO / "checkpoint/goto_best.pt"
_MANEUVER_CKPT_PINNED = _REPO / "checkpoint/maneuver_best.pt"
GOTO_CKPT     = str(_GOTO_CKPT_PINNED)     if _GOTO_CKPT_PINNED.exists()     else (
                str(GOTO_CKPT_EP3)          if GOTO_CKPT_EP3.exists()         else str(GOTO_CKPT_BEST))
MANEUVER_CKPT = str(_MANEUVER_CKPT_PINNED) if _MANEUVER_CKPT_PINNED.exists() else str(MANEUVER_CKPT)

# H1 refresh: demo-distance goto uses 1700 steps (4-9m walks at ~50Hz)
# V2/V3 demo-distance grounding (26° cam, 480x360, depth-FG rescue) wired in inferencer.py.
# Default difficulty = 'demo' to showcase 4-9m LONG walks (key demo deliverable).
# NX-10 (docs/nx10_scan_fix.md): bumped 1400 -> 1700, matching code/eval_closedloop.py's
# MAXSTEPS['demo'] -- the widened H3 scan (BidirectionalScanSchedule, realized-yaw
# tracking) needs more absolute-step budget for unfavorable-direction bearings.
MAXSTEPS_GOTO     = 1700  # demo preset: 1700 steps for 4-9m walks (was 1400 pre-NX-10)
MAXSTEPS_MANEUVER = 1400  # unchanged -- maneuver has its own separate rollout loop (nx9_avoid.md §8)
DEMO_OUT_DIR = _REPO / "eval/demo"
WEB_PORT     = 5000

# Colors and shapes (from arena.py)
COLORS  = ["red", "yellow", "blue", "green", "orange", "purple", "cyan"]
SHAPES  = ["ball", "cube", "cylinder", "cone"]

MANEUVER_DIRECTIONS = ["left", "right"]


# ---------------------------------------------------------------------------
# Language cache (GR00T-LM embeddings)
# ---------------------------------------------------------------------------
_lang_cache: dict[str, np.ndarray] | None = None


def _load_lang_cache() -> dict[str, np.ndarray] | None:
    """Lazily load and cache the GR00T-LM instruction embedding table.

    Returns:
        The cached embedding dict, or None if the cache file is missing or
        fails to load.
    """
    global _lang_cache
    if _lang_cache is not None:
        return _lang_cache
    if LANG_CACHE.exists():
        try:
            with open(LANG_CACHE, "rb") as f:
                _lang_cache = pickle.load(f)
            print(f"[demo] Loaded lang cache: {len(_lang_cache)} instructions", flush=True)
            return _lang_cache
        except Exception as e:
            print(f"[demo] WARN: Failed to load lang cache: {e}", flush=True)
    return None


def _get_lang_emb(instruction: str) -> np.ndarray | None:
    """Look up GR00T-LM embedding from cache; None if not found."""
    cache = _load_lang_cache()
    if cache is None:
        return None
    return cache.get(instruction, None)
