"""
code.eval.maneuver — maneuver-skill evaluator: aggregation/reporting + CLI.

Split out of the original ``eval_maneuver.py`` (RF-1): loads the checkpoint,
drives ``code.eval.maneuver_rollout.run_maneuver_rollout`` over N held-out
scenes, aggregates success/failure-tag rates, writes the summary JSON, and
exposes the CLI entry point.

Task: "go straight, turn {left/right} after passing the {color}{shape}"

SUCCESS criteria (ALL must pass):
  1. Robot passes the landmark: robot_x > landmark_x + pass_margin
  2. Robot turns correct direction after passing: |final_heading_err| < HEADING_SUCCESS_THR
  3. Robot remains upright (height > 0.50m) throughout
  4. No fall

Eval protocol:
  - seed=999 (held-out)
  - n=15 episodes
  - MAXSTEPS=1400 (same as demo)
  - No render (fast eval)
  - Render first 3 success episodes for video

Everything the pre-RF-1 ``code.eval_maneuver`` module exposed at import time
(``run_maneuver_rollout``, ``evaluate_maneuver``, ``HEADING_SUCCESS_THR``,
``IMG_SIZE``, ...) is re-imported here so the old-path alias
(``code/eval_maneuver.py``) keeps every downstream import (demo.py,
record_showcase.py, ...) working unchanged.

Usage
-----
# Smoke test (1 episode, 150 steps):
MUJOCO_GL=egl python code/eval_maneuver.py \\
    --checkpoint runs/maneuver_A/epoch_0010.pt \\
    --smoke

# Full eval (15 episodes):
MUJOCO_GL=egl python code/eval_maneuver.py \\
    --checkpoint runs/maneuver_A/epoch_0010.pt \\
    --n 15 --device cuda \\
    --out eval/maneuver_A/ep10
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# GPU-rendering fix (2026-07-11): steer glvnd to the NVIDIA EGL ICD when
# present, BEFORE mujoco initializes EGL — otherwise Mesa can win the vendor
# race and MuJoCo silently renders on llvmpipe (CPU) at ~400 ms/frame vs
# ~1.3 ms on the GPU. Idempotent; no-op when the ICD file is absent or the
# user already chose a vendor. See code/arena.py for the measured numbers.
import os as _os
_NVIDIA_EGL_ICD = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if _os.path.exists(_NVIDIA_EGL_ICD):
    _os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NVIDIA_EGL_ICD)
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from code.small_vla import GroundedNav
from code.maneuver_scene import sample_maneuver_scene, derive_rng, HORIZON
from code.dataset_maneuver import PROPRIO_DIM_MANEUVER
from code.train.maneuver import load_loco_checkpoint

# Re-exported for the pre-RF-1 import path (`from code.eval_maneuver import
# run_maneuver_rollout, evaluate_maneuver, HEADING_SUCCESS_THR, IMG_SIZE`) via
# the old-path alias shim.
from code.eval.maneuver_types import ManeuverResult, HEADING_SUCCESS_THR, IMG_SIZE
from code.eval.maneuver_rollout import run_maneuver_rollout

__all__ = [
    'ManeuverResult', 'HEADING_SUCCESS_THR', 'IMG_SIZE',
    'run_maneuver_rollout', 'evaluate_maneuver', 'main',
]


def evaluate_maneuver(
    checkpoint_path: str,
    n_scenes: int = 15,
    seed: int = 999,
    device_str: str = 'cpu',
    out_dir: str = 'eval/maneuver',
    render_n: int = 3,       # render first N successes
    smoke: bool = False,
    smoke_steps: int = 150,
    free_vel: bool = False,   # if True, use model's predicted vel (no teacher forcing)
    hybrid_vel: bool = False, # if True, TF only during TURN_PHASE, free otherwise
) -> dict:
    """Run maneuver evaluation.

    Args:
        checkpoint_path: Path to the .pt checkpoint to evaluate.
        n_scenes: Number of held-out scenes to evaluate.
        seed: Eval seed (default=999, held-out).
        device_str: Torch device string ('cpu' or 'cuda').
        out_dir: Output directory for videos, incremental results, and the
            summary JSON.
        render_n: Number of successful episodes to re-render as video.
        smoke: If True, run a single episode with smoke_steps as the cap.
        smoke_steps: Step cap used when smoke=True.
        free_vel: If True, use the model's predicted velocity (no teacher
            forcing).
        hybrid_vel: If True, teacher-force velocity only during TURN_PHASE,
            free otherwise.

    Returns:
        Summary dict with success_rate, failure tag breakdown, mean steps,
        etc. (also written to out_dir/summary.json).

    Raises:
        FileNotFoundError: If checkpoint_path does not point to an existing
            file.
    """
    device = torch.device(device_str)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[eval_maneuver] Checkpoint: {checkpoint_path}")
    print(f"[eval_maneuver] seed={seed}  n={n_scenes}  device={device_str}")
    print(f"[eval_maneuver] out_dir={out_dir}", flush=True)

    # Load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    ckpt_proprio_dim = ckpt.get('proprio_dim', PROPRIO_DIM_MANEUVER)
    is_maneuver = ckpt.get('maneuver', False)
    arch = ckpt.get('arch', 'A')

    print(f"[eval_maneuver] ckpt proprio_dim={ckpt_proprio_dim}  maneuver={is_maneuver}")

    # Build model at the checkpoint's proprio_dim, then expand if needed
    if ckpt_proprio_dim == PROPRIO_DIM_MANEUVER:
        model = GroundedNav(
            arch=arch, teacher_forcing=True, chunk_H=1,
            proprio_dim=PROPRIO_DIM_MANEUVER,
        ).to(device)
        model_state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
        model.load_state_dict(model_state, strict=False)
    else:
        # Locomotion checkpoint (57-d), expand to 62-d
        model, _ = load_loco_checkpoint(checkpoint_path, device)

    model.eval()

    # Precompute TinyViT output for zero image (cached for all steps — 2x speedup on CPU)
    with torch.no_grad():
        _img_zero = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32, device=device)
        _, vis_pooled_cache = model.vision(_img_zero)
    print(f"[eval_maneuver] Precomputed TinyViT embedding (shape={vis_pooled_cache.shape})", flush=True)

    # Action stats
    _as = ckpt.get('action_stats', None)
    if _as:
        action_stats = {
            'mean':           np.array(_as['mean'],           dtype=np.float32),
            'std':            np.array(_as['std'],            dtype=np.float32),
            'default_angles': np.array(_as['default_angles'], dtype=np.float32),
            'n_frames':       _as.get('n_frames', 0),
        }
        print(f"[eval_maneuver] Loaded action_stats from checkpoint", flush=True)
    else:
        action_stats = None
        print(f"[eval_maneuver] WARNING: No action_stats in checkpoint — using raw action output")

    # Eval loop
    results = []
    n_success = 0
    n_fall = 0
    n_no_lm = 0
    n_wrong_heading = 0
    videos_rendered = 0

    maxsteps = smoke_steps if smoke else HORIZON

    for ep_i in range(n_scenes if not smoke else 1):
        rng       = derive_rng(seed, ep_i)
        scene_cfg = sample_maneuver_scene(rng)

        turn_dir  = scene_cfg["turn_direction"]
        lm_obj    = scene_cfg["objects"][scene_cfg["landmark_index"]]
        print(f"\n[ep {ep_i:02d}/{n_scenes}]  '{scene_cfg['instruction'][:60]}'  "
              f"turn={turn_dir}  lm_dist={lm_obj['dist_from_robot']:.2f}m", flush=True)

        ep_t0 = time.time()
        result = run_maneuver_rollout(
            model             = model,
            action_stats      = action_stats,
            device            = device,
            scene_cfg         = scene_cfg,
            maxsteps          = maxsteps,
            render_video      = False,
            video_path        = None,
            vis_pooled_cache  = vis_pooled_cache,
            free_vel          = free_vel,
            hybrid_vel        = hybrid_vel,
        )
        ep_elapsed = time.time() - ep_t0

        ep_sps = result.steps / max(ep_elapsed, 1e-6)
        print(f"  -> steps={result.steps}  lm_passed={result.landmark_passed}  "
              f"heading_err={math.degrees(result.final_heading_err):.1f}°  "
              f"fell={result.fell}  tag={result.failure_tag}  {ep_sps:.1f}stp/s",
              flush=True)

        if result.success:
            n_success += 1
            # Render video if we haven't done render_n yet
            if videos_rendered < render_n:
                vid_path = os.path.join(out_dir, f"ep{ep_i:03d}_success.mp4")
                print(f"  Rendering success video → {vid_path}")
                result2 = run_maneuver_rollout(
                    model             = model,
                    action_stats      = action_stats,
                    device            = device,
                    scene_cfg         = scene_cfg,
                    maxsteps          = maxsteps,
                    render_video      = True,
                    video_path        = vid_path,
                    vis_pooled_cache  = vis_pooled_cache,
                    free_vel          = free_vel,
                    hybrid_vel        = hybrid_vel,
                )
                result.video_path = result2.video_path
                videos_rendered += 1
        elif result.failure_tag == 'fall':
            n_fall += 1
        elif result.failure_tag == 'no_landmark':
            n_no_lm += 1
        elif result.failure_tag == 'wrong_heading':
            n_wrong_heading += 1

        results.append({
            "ep":              ep_i,
            "turn_direction":  turn_dir,
            "lm_dist":         round(lm_obj['dist_from_robot'], 2),
            "steps":           result.steps,
            "landmark_passed": result.landmark_passed,
            "final_heading_err_deg": round(math.degrees(result.final_heading_err), 1),
            "final_state":     result.final_state,
            "fell":            result.fell,
            "success":         result.success,
            "failure_tag":     result.failure_tag,
            "video_path":      result.video_path,
        })

        # Write incremental results
        with open(os.path.join(out_dir, 'results.jsonl'), 'a') as f:
            f.write(json.dumps(results[-1]) + "\n")

        # EGL non-determinism fix: force GC between episodes.
        gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.synchronize()
                _torch.cuda.empty_cache()
        except Exception:
            pass

    n = len(results)
    success_rate = n_success / max(1, n)
    summary = {
        "checkpoint":          checkpoint_path,
        "seed":                seed,
        "n_episodes":          n,
        "n_success":           n_success,
        "success_rate":        round(success_rate, 3),
        "n_fall":              n_fall,
        "n_no_landmark":       n_no_lm,
        "n_wrong_heading":     n_wrong_heading,
        "pct_fall":            round(n_fall / max(1, n), 3),
        "pct_no_landmark":     round(n_no_lm / max(1, n), 3),
        "pct_wrong_heading":   round(n_wrong_heading / max(1, n), 3),
        "mean_steps":          round(float(np.mean([r["steps"] for r in results])), 1),
        "heading_thr_deg":     round(math.degrees(HEADING_SUCCESS_THR), 1),
    }

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  MANEUVER EVAL RESULTS  (seed={seed}, n={n})")
    print(f"{'='*70}")
    print(f"  {'ep':>4}  {'turn':>6}  {'lm_d':>5}  {'steps':>6}  "
          f"{'lm_pass':>8}  {'head_err':>9}  {'tag'}")
    print("  " + "-"*65)
    for r in results:
        print(f"  {r['ep']:>4}  {r['turn_direction']:>6}  {r['lm_dist']:>5.1f}  "
              f"{r['steps']:>6}  {str(r['landmark_passed']):>8}  "
              f"{r['final_heading_err_deg']:>+8.1f}°  {r['failure_tag']}")
    print()
    print(f"  SUCCESS: {n_success}/{n} = {success_rate:.1%}")
    print(f"  Falls:         {n_fall}/{n}")
    print(f"  No landmark:   {n_no_lm}/{n}")
    print(f"  Wrong heading: {n_wrong_heading}/{n}")
    print(f"  Mean steps:    {summary['mean_steps']:.1f}")
    print(f"{'='*70}", flush=True)

    return summary


def main() -> None:
    """Parse CLI arguments and run the maneuver evaluation."""
    ap = argparse.ArgumentParser(description="Maneuver skill evaluator")
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--n',          type=int,   default=15)
    ap.add_argument('--seed',       type=int,   default=999)
    ap.add_argument('--device',     default='cpu')
    ap.add_argument('--out',        default='eval/maneuver')
    ap.add_argument('--render-n',   type=int,   default=3,
                    help='Render first N success episodes as video')
    ap.add_argument('--smoke',      action='store_true',
                    help='Quick smoke test (1 episode, 150 steps)')
    ap.add_argument('--smoke-steps', type=int, default=150)
    ap.add_argument('--free-vel',   action='store_true',
                    help='Ablation: use model predicted vel (no expert teacher-forcing)')
    ap.add_argument('--no-hybrid-vel', action='store_true',
                    help='Disable hybrid-vel (default: ON — TF vel only during TURN_PHASE)')
    ap.add_argument('--hybrid-vel', action='store_true',
                    help='[deprecated flag, now on by default] TF vel during TURN_PHASE only')
    args = ap.parse_args()

    # hybrid_vel is ON by default; disable with --no-hybrid-vel or --free-vel
    use_hybrid = not args.free_vel and not args.no_hybrid_vel

    summary = evaluate_maneuver(
        checkpoint_path = args.checkpoint,
        n_scenes        = args.n,
        seed            = args.seed,
        device_str      = args.device,
        out_dir         = args.out,
        render_n        = args.render_n,
        smoke           = args.smoke,
        smoke_steps     = args.smoke_steps,
        free_vel        = args.free_vel,
        hybrid_vel      = use_hybrid,
    )
    print(json.dumps(summary, indent=2))
    sys.exit(0)


if __name__ == '__main__':
    main()
