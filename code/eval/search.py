"""code.eval.search — search-skill evaluator: aggregation/reporting + CLI.

Split out of the original ``eval_search.py`` (RF-1): drives
``code.eval.search_rollout._run_search_rollout`` over N held-out scenes,
aggregates spot/reach/success rates, writes the summary JSON, and exposes the
CLI entry point.

Everything the pre-RF-1 ``code.eval_search`` module exposed at import time
(``sample_search_scene``, ``_run_search_rollout``, ``STOP_R_SEARCH``,
``MAXSTEPS_SEARCH``, ``SEARCH_FOV_HALF_DEG``, ``SEARCH_DIST_MIN``,
``SEARCH_DIST_MAX``, ``SCAN_ALIGNED_THR_DEG``, ...) is re-imported here so the
old-path alias (``code/eval_search.py``) keeps every downstream import
(demo.py, fancy_demo.py, gen_det_dataset.py, gen_det_failcases.py,
record_showcase.py, ...) working unchanged.

Eval protocol:
  - Seed: 999 (held-out)
  - n=15 scenes, target OUTSIDE initial FOV (bearing > 45°)
  - Success = spotted (target entered FOV during scan) + reached (final_dist < STOP_R) + upright
  - Reports SPOT-rate + REACH-rate separately
  - Renders 2-3 success videos (ego|third-person: rotate → spot → approach)

ANTI-HANG:
  - Smoke 1 scene first (fast, MAXSTEPS=200)
  - Hard MAXSTEPS cap: 1400 steps
  - Background process + poll every 10s
  - Flush prints throughout

Usage:
    MUJOCO_GL=egl python code/eval_search.py --smoke
    MUJOCO_GL=egl python code/eval_search.py --n 15 --out eval/search --device cuda
    MUJOCO_GL=egl python code/eval_search.py --n 15 --out eval/search --no-video
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

# Re-exported for the pre-RF-1 import path (`from code.eval_search import
# sample_search_scene, _run_search_rollout, STOP_R_SEARCH, ...`) via the
# old-path alias shim.
from code.eval.search_types import (
    EVAL_SEED, MAXSTEPS_SEARCH, STOP_R_SEARCH, N_RENDER, GOTO_CKPT,
    SEARCH_FOV_HALF_DEG, SEARCH_DIST_MIN, SEARCH_DIST_MAX, SCAN_ALIGNED_THR_DEG,
    sample_search_scene, SearchResult,
)
from code.eval.search_rollout import _run_search_rollout

__all__ = [
    'EVAL_SEED', 'MAXSTEPS_SEARCH', 'STOP_R_SEARCH', 'N_RENDER', 'GOTO_CKPT',
    'SEARCH_FOV_HALF_DEG', 'SEARCH_DIST_MIN', 'SEARCH_DIST_MAX', 'SCAN_ALIGNED_THR_DEG',
    'sample_search_scene', 'SearchResult', '_run_search_rollout',
    'evaluate_search', 'main',
]


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate_search(
    checkpoint_path: str | None = None,
    n_scenes:    int   = 15,
    device:      str   = 'cpu',
    out_dir:     str   = 'eval/search',
    render_video: bool = True,
    smoke:       bool  = False,
    seed:        int   = 999,
) -> dict:
    """Run search evaluation: target starts outside initial FOV.

    Args:
        checkpoint_path: Path to a .pt checkpoint, or None to use the
            default GOTO_CKPT.
        n_scenes: Number of held-out scenes to evaluate.
        device: Torch device string for inference ('cpu' or 'cuda').
        out_dir: Output directory for videos and the summary JSON.
        render_video: Whether to render video for the first N_RENDER
            episodes.
        smoke: If True, run a single scene with a 200-step cap.
        seed: Eval seed (default=999, held-out).

    Returns:
        Summary dict with spot_rate, reach_rate, success_rate, per-outcome
        counts, and the full per-episode list (also written to a JSON file
        under out_dir).
    """
    from code.inferencer import Inferencer

    os.makedirs(out_dir, exist_ok=True)

    ckpt = checkpoint_path or GOTO_CKPT
    print(f"[search_eval] Loading inferencer: {ckpt}", flush=True)
    inf = Inferencer(
        checkpoint_path=ckpt,
        arch='A',
        device=device,
        goal_source='classical',
        verbose=False,
    )
    print(f"[search_eval] Inferencer ready", flush=True)

    if smoke:
        n_scenes = 1
        print(f"[search_eval] SMOKE MODE: 1 scene, MAXSTEPS=200", flush=True)

    results: list[SearchResult] = []
    ep_results = []

    for ep_i in range(n_scenes):
        rng = np.random.default_rng(np.random.SeedSequence([seed, ep_i]))
        scene_cfg = sample_search_scene(rng, ep_i)

        tgt    = scene_cfg['objects'][scene_cfg['target_index']]
        init_b = float(scene_cfg.get('init_bearing_deg', 0.0))

        print(f"\n[search_eval] ep={ep_i:02d}  {tgt['color_name']} {tgt['shape_name']}  "
              f"dist={tgt['dist_from_robot']:.2f}m  init_bearing={init_b:.1f}°", flush=True)

        # Render only first N_RENDER episodes (success videos)
        do_render  = render_video and (ep_i < N_RENDER)
        video_path = None
        if do_render:
            video_path = os.path.join(
                out_dir,
                f"search_ep{ep_i:02d}_{tgt['color_name']}_{tgt['shape_name']}.mp4"
            )

        maxsteps = 200 if smoke else MAXSTEPS_SEARCH
        t0 = time.time()

        try:
            raw = _run_search_rollout(
                inf=inf,
                scene_cfg=scene_cfg,
                instruction=scene_cfg['instruction'],
                maxsteps=maxsteps,
                render_video=do_render,
                video_path=video_path,
            )
        except Exception as e:
            import traceback
            print(f"[search_eval] ep={ep_i} EXCEPTION: {e}", flush=True)
            traceback.print_exc()
            raw = dict(success=False, spotted=False, scan_steps=0, failure_tag='error',
                       steps=0, final_dist=999.0, fell=False, ms_per_step=0.0, video_path=None,
                       avoid_bias_active_frac=0.0)

        dt = time.time() - t0
        sr = SearchResult(
            ep_idx           = ep_i,
            instruction      = scene_cfg['instruction'],
            target_color     = tgt['color_name'],
            target_shape     = tgt['shape_name'],
            target_dist      = tgt['dist_from_robot'],
            init_bearing_deg = init_b,
            spotted          = raw['spotted'],
            reached          = (raw['final_dist'] < STOP_R_SEARCH) and not raw['fell'],
            success          = raw['success'],
            failure_tag      = raw['failure_tag'],
            steps            = raw['steps'],
            scan_steps       = raw['scan_steps'],
            final_dist       = raw['final_dist'],
            fell             = raw['fell'],
            ms_per_step      = raw['ms_per_step'],
            video_path       = raw.get('video_path'),
            avoid_bias_active_frac = raw.get('avoid_bias_active_frac', 0.0),
        )
        results.append(sr)
        ep_results.append(asdict(sr))

        print(f"  → spotted={sr.spotted}  reached={sr.reached}  success={sr.success}  "
              f"tag={sr.failure_tag}  steps={sr.steps}  fd={sr.final_dist:.2f}m  "
              f"wall={dt:.1f}s", flush=True)

        # EGL non-determinism fix: force GC between episodes to free EGL renderer objects.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- Summary ----
    n    = len(results)
    spot  = sum(1 for r in results if r.spotted)
    reach = sum(1 for r in results if r.reached)
    succ  = sum(1 for r in results if r.success)
    falls = sum(1 for r in results if r.fell)

    spot_rate  = spot  / n if n else 0.0
    reach_rate = reach / n if n else 0.0
    succ_rate  = succ  / n if n else 0.0

    print(f"\n{'='*60}", flush=True)
    print(f"[search_eval] RESULTS  (n={n}, seed={seed})", flush=True)
    print(f"  SPOT-rate:  {spot}/{n} = {spot_rate:.1%}", flush=True)
    print(f"  REACH-rate: {reach}/{n} = {reach_rate:.1%}", flush=True)
    print(f"  SUCCESS:    {succ}/{n} = {succ_rate:.1%}  (spotted+reached+upright)", flush=True)
    print(f"  Falls:      {falls}/{n}", flush=True)
    print(f"{'='*60}", flush=True)

    summary = {
        "n_scenes":    n,
        "eval_seed":   seed,
        "checkpoint":  ckpt,
        "spot_rate":   spot_rate,
        "reach_rate":  reach_rate,
        "success_rate": succ_rate,
        "n_spot":      spot,
        "n_reach":     reach,
        "n_success":   succ,
        "n_falls":     falls,
        "episodes":    ep_results,
    }

    # Save
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[search_eval] Summary saved → {summary_path}", flush=True)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run the search evaluation."""
    ap = argparse.ArgumentParser(description="Search skill evaluator")
    ap.add_argument("--checkpoint", default=None, help="Path to goto checkpoint")
    ap.add_argument("--n",          type=int, default=15, help="Number of scenes")
    ap.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out",        default="eval/search", help="Output directory")
    ap.add_argument("--smoke",      action="store_true", help="1-scene smoke test")
    ap.add_argument("--no-video",   action="store_true", help="Disable video rendering")
    ap.add_argument("--seed",       type=int, default=999,
                    help="Eval seed (default=999 held-out; use other values for robustness)")
    args = ap.parse_args()

    evaluate_search(
        checkpoint_path=args.checkpoint,
        n_scenes=args.n,
        device=args.device,
        out_dir=args.out,
        render_video=not args.no_video,
        smoke=args.smoke,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
