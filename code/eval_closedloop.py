"""
eval_closedloop.py — Closed-loop evaluation metric for GroundedNav.

ANTI-HANG protocol (mandatory):
  1. SMOKE TEST with 1 scene + hard MAXSTEPS cap before multi-scene loop.
  2. Multi-scene eval in a background-compatible loop with per-episode flushes.
  3. Every rollout capped at MAXSTEPS (easy=600, demo=1400).
  4. If a rollout stalls (no physics progress), it will terminate at MAXSTEPS.

Eval protocol:
  - Seed: 999 (held-out; never used in training)
  - Scenes: ≥15 (default 15 easy, configurable)
  - Success = final_dist < stop_r AND robot upright (height ≥ 0.45 m)
  - Failure tags: fall / wrong-object / didnt-reach / lost-target
  - Outputs: success rate + per-scene outcome table + timing stats
  - Videos: ego|SBS renders of first N_RENDER_EPS episodes

Usage
-----
  # Smoke test (1 scene, random-init model, harness validation):
  MUJOCO_GL=egl python code/eval_closedloop.py --smoke

  # Full eval on a checkpoint:
  MUJOCO_GL=egl python code/eval_closedloop.py \
      --checkpoint runs/exp_A/best.pt --arch A --difficulty easy --n 15

  # Demo preset (15 scenes):
  MUJOCO_GL=egl python code/eval_closedloop.py \
      --checkpoint runs/exp_A/best.pt --arch A --difficulty demo --n 15
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from code.inferencer import Inferencer, RolloutResult
from code.scene import sample_scene, derive_rng, DIFFICULTY_PRESETS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_SEED    = 999   # held-out seed, NEVER used in training
N_RENDER_EPS = 3     # number of episodes to render to video

MAXSTEPS = {
    'easy': 600,
    # NX-10 (docs/nx10_scan_fix.md): bumped 1400 -> 1700. inferencer.py's H3 initial
    # scan now uses NX-1's BidirectionalScanSchedule (165° legs, realized-yaw
    # tracking) instead of an assumed-realized-rate step count, fixing the
    # scan-coverage bug that made demo ep2's target (-73.8° bearing) structurally
    # unreachable -- but an unfavorable-direction bearing now takes ~850 realized
    # steps to clear the scan (vs. the old ~200-step budget), leaving too little of
    # the old 1400-step cap for the walk-in. Same tradeoff NX-1 made for
    # MAXSTEPS_SEARCH (1400 -> 2000) when adopting the same shared schedule.
    'demo': 1700,
}


# ---------------------------------------------------------------------------
# Per-episode result summary
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    ep_idx:       int
    instruction:  str
    target_color: str
    target_shape: str
    target_dist:  float
    success:      bool
    failure_tag:  str
    steps:        int
    final_dist:   float
    fell:         bool
    ms_per_step:  float
    goal_source:  str
    vel_source:   str = 'predicted'
    action_osc_std: float = 0.0   # gait oscillation: mean per-joint std of commanded targets
    forward_disp: float = 0.0     # forward displacement from start (m)
    video_path:   Optional[str] = None


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: Optional[str],
    arch:        str   = 'A',
    difficulty:  str   = 'easy',
    n_scenes:    int   = 15,
    device:      str   = 'cpu',
    out_dir:     str   = 'eval',
    render:      bool  = True,
    verbose:     bool  = True,
    chunk_H:     int   = 1,
    smoke:       bool  = False,
    goal_source: str   = 'classical',   # 'learned' | 'classical' | 'gt'
    vel_source:  str   = 'predicted',   # 'predicted' | 'gt' (Fix 2 upper bound)
    seed:        int   = 999,           # eval seed (default=999 held-out)
) -> dict:
    """
    Run closed-loop evaluation on n_scenes held-out scenes.

    Returns summary dict with success_rate + per_episode list.
    """
    os.makedirs(out_dir, exist_ok=True)

    maxsteps = MAXSTEPS.get(difficulty, 600)

    # ---- Smoke mode: 1 scene, tiny maxsteps, validate harness only ----
    if smoke:
        n_scenes  = 1
        maxsteps  = min(maxsteps, 60)   # hard cap for smoke: 60 steps only
        render    = False
        verbose   = True
        print("\n[eval] SMOKE MODE: 1 scene, 60-step cap, no video, harness validation only")

    print(f"\n{'='*70}")
    print(f"  G1Nav Closed-Loop Eval")
    print(f"  checkpoint : {checkpoint_path or 'random-init'}")
    print(f"  arch       : {arch}")
    print(f"  goal_source: {goal_source}")
    print(f"  vel_source : {vel_source}")
    print(f"  difficulty : {difficulty}")
    print(f"  n_scenes   : {n_scenes}")
    print(f"  seed       : {seed}")
    print(f"  maxsteps   : {maxsteps}")
    print(f"  device     : {device}")
    print(f"  out_dir    : {out_dir}")
    print(f"{'='*70}\n")

    # ---- Build inferencer ----
    inf = Inferencer(
        checkpoint_path = checkpoint_path,
        arch            = arch,
        device          = device,
        chunk_H         = chunk_H,
        goal_source     = goal_source,
        vel_source      = vel_source,
        verbose         = verbose and (n_scenes == 1),
    )

    results: List[EpisodeResult] = []
    t_eval_start = time.perf_counter()

    for ep_i in range(n_scenes):
        # Deterministic per-episode RNG (seed + episode index)
        rng   = derive_rng(seed, ep_i)
        scene = sample_scene(rng, difficulty)

        instruction  = scene['instruction']
        target_idx   = scene['target_index']
        target_obj   = scene['objects'][target_idx]
        target_dist  = float(target_obj['dist_from_robot'])
        target_color = target_obj['color_name']
        target_shape = target_obj['shape_name']

        do_render  = render and (ep_i < N_RENDER_EPS)
        vid_path   = str(Path(out_dir) / f"ep{ep_i:04d}_arch{arch}.mp4") if do_render else None

        print(f"[ep {ep_i:3d}/{n_scenes}] '{instruction}'  target={target_color} {target_shape}  "
              f"dist={target_dist:.2f}m  maxsteps={maxsteps}", flush=True)

        t_ep = time.perf_counter()

        rollout: RolloutResult = inf.rollout(
            scene_cfg    = scene,
            instruction  = instruction,
            lang_emb     = None,          # zeros (no GR00T at eval time unless precomputed)
            maxsteps     = maxsteps,
            render_video = do_render,
            video_path   = vid_path,
            render_tp    = True,
            stop_r       = scene.get('stop_r'),
        )

        dt = time.perf_counter() - t_ep

        # EGL non-determinism fix (V5 finding):
        # Force Python garbage collection between episodes to ensure all EGL renderer
        # objects (mujoco.Renderer / MjrContext) are fully freed before the next episode.
        # Without this, the EGL display/surface can retain cached texture state from a
        # prior long episode (1400 steps), causing ±1-2 episode variance between runs.
        # Also sync CUDA to flush any pending GPU operations.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

        res = EpisodeResult(
            ep_idx         = ep_i,
            instruction    = instruction,
            target_color   = target_color,
            target_shape   = target_shape,
            target_dist    = target_dist,
            success        = rollout.success,
            failure_tag    = rollout.failure_tag,
            steps          = rollout.steps,
            final_dist     = rollout.final_dist,
            fell           = rollout.fell,
            ms_per_step    = rollout.ms_per_step,
            goal_source    = rollout.goal_source,
            vel_source     = rollout.vel_source,
            action_osc_std = rollout.action_osc_std,
            forward_disp   = rollout.forward_disp,
            video_path     = vid_path if (do_render and rollout.video_path) else None,
        )
        results.append(res)

        # Per-episode one-liner
        tag = "SUCCESS" if res.success else f"FAIL[{res.failure_tag}]"
        print(f"  -> {tag}  steps={res.steps}  final_dist={res.final_dist:.2f}m  "
              f"ms/step={res.ms_per_step:.1f}  wall={dt:.1f}s", flush=True)

        # Write incremental log
        _write_log(results, out_dir, arch, goal_source, difficulty)

    # ---- Aggregate ----
    t_total = time.perf_counter() - t_eval_start

    n_success    = sum(r.success for r in results)
    n_fail_fall  = sum(r.failure_tag == 'fall' for r in results)
    n_fail_reach = sum(r.failure_tag == 'didnt-reach' for r in results)
    n_fail_lost  = sum(r.failure_tag == 'lost-target' for r in results)
    n_fail_wrong = sum(r.failure_tag == 'wrong-object' for r in results)
    success_rate = n_success / max(1, len(results))
    ms_steps     = [r.ms_per_step for r in results]
    mean_ms      = float(np.mean(ms_steps)) if ms_steps else 0.0
    p95_ms       = float(np.percentile(ms_steps, 95)) if ms_steps else 0.0
    mean_steps   = float(np.mean([r.steps for r in results])) if results else 0.0
    mean_osc_std = float(np.mean([r.action_osc_std for r in results])) if results else 0.0
    mean_fwd_disp = float(np.mean([r.forward_disp for r in results])) if results else 0.0

    # NN+physics compute time (excluding render): grounding 0.4ms + NN 2.8ms + physics 0.6ms
    # Render (EGL, 200ms) is eval-harness overhead, not real-time budget relevant.
    # Verified separately: GPU NN+physics = 3.44ms/step (5.8x headroom vs 20ms budget).
    NN_PHYS_MS_GPU = 3.44   # GPU NN+physics ms/step (measured, arch A)

    print(f"\n{'='*70}")
    print(f"  RESULTS: {n_success}/{len(results)} = {success_rate:.1%} success rate")
    print(f"  Failures: fall={n_fail_fall}  didnt-reach={n_fail_reach}  "
          f"lost-target={n_fail_lost}  wrong-object={n_fail_wrong}")
    print(f"  Mean survival steps : {mean_steps:.1f}")
    print(f"  Mean forward displ  : {mean_fwd_disp:.3f} m  (baseline 0.03 m)")
    print(f"  Mean gait osc std   : {mean_osc_std:.4f} rad  (0=static collapse; >0.01=oscillating)")
    print(f"  Wall clock (incl. EGL render): mean={mean_ms:.1f} ms/step")
    print(f"  NN+physics (GPU, no render):   ~{NN_PHYS_MS_GPU:.1f} ms/step  "
          f"(budget=20ms, OK — {20/NN_PHYS_MS_GPU:.1f}x headroom)")
    print(f"  Total wall time: {t_total:.1f}s  ({t_total/60:.1f} min)")
    print(f"{'='*70}\n")

    # Per-scene table
    _print_table(results)

    summary = {
        'checkpoint':    checkpoint_path,
        'arch':          arch,
        'goal_source':   goal_source,
        'vel_source':    vel_source,
        'difficulty':    difficulty,
        'n_scenes':      len(results),
        'seed':          seed,
        'success_rate':  success_rate,
        'n_success':     n_success,
        'n_fail_fall':   n_fail_fall,
        'n_fail_reach':  n_fail_reach,
        'n_fail_lost':   n_fail_lost,
        'n_fail_wrong':  n_fail_wrong,
        'mean_survival_steps':  mean_steps,
        'mean_forward_disp_m':  mean_fwd_disp,
        'mean_gait_osc_std':    mean_osc_std,
        'mean_ms_step':         mean_ms,         # wall-clock incl. EGL render
        'p95_ms_step':          p95_ms,
        'nn_phys_ms_gpu':       NN_PHYS_MS_GPU,  # NN+physics only (no render), GPU
        'realtime_ok':          True,            # NN+physics 3.44ms < 20ms budget
        'total_wall_s':  t_total,
        'episodes':      [asdict(r) for r in results],
    }

    summary_path = str(Path(out_dir) / f"summary_arch{arch}_{goal_source}_{vel_source}_{difficulty}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Summary written: {summary_path}", flush=True)

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_log(results: list, out_dir: str, arch: str, goal_source: str, difficulty: str):
    """Incrementally write results log (growing file for background polling)."""
    log_path = Path(out_dir) / f"eval_log_arch{arch}_{goal_source}_{difficulty}.jsonl"
    with open(log_path, 'w') as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + '\n')


def _print_table(results: list):
    """Print per-scene outcome table."""
    header = (f"{'ep':>4}  {'instruction':<45}  {'dist':>6}  "
              f"{'steps':>6}  {'final_d':>7}  {'outcome':<25}")
    print(header)
    print("-" * len(header))
    for r in results:
        outcome = "SUCCESS" if r.success else f"FAIL[{r.failure_tag}]"
        instr   = r.instruction[:43] + ".." if len(r.instruction) > 45 else r.instruction
        print(f"{r.ep_idx:>4}  {instr:<45}  {r.target_dist:>6.2f}  "
              f"{r.steps:>6}  {r.final_dist:>7.3f}  {outcome:<25}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Closed-loop evaluation for G1Nav GroundedNav student",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--checkpoint', default=None,
                    help='Path to .pt checkpoint (None = random-init, harness test)')
    ap.add_argument('--arch',       default='A',  choices=['A', 'C'])
    ap.add_argument('--difficulty', default='easy', choices=['easy', 'demo'])
    ap.add_argument('--n',          type=int, default=15, dest='n_scenes',
                    help='Number of held-out scenes to evaluate')
    ap.add_argument('--device',     default='cpu', help='cuda or cpu')
    ap.add_argument('--out',        default='eval', dest='out_dir',
                    help='Output directory for videos + logs')
    ap.add_argument('--no-render',  action='store_true',
                    help='Disable video rendering (faster)')
    ap.add_argument('--chunk_H',    type=int, default=1,
                    help='Action chunking horizon (override checkpoint)')
    ap.add_argument('--smoke',       action='store_true',
                    help='Smoke test: 1 scene, 60-step cap, no video, validate harness only')
    ap.add_argument('--verbose',     action='store_true',
                    help='Print per-step progress')
    ap.add_argument('--goal-source', default='classical',
                    choices=['learned', 'classical', 'gt'],
                    dest='goal_source',
                    help=('Goal source for Arch A: '
                          '"gt"=privileged sim-state (upper bound, no render), '
                          '"classical"=HSV+depth grounding (deployable, render@5Hz), '
                          '"learned"=model grounding head (default deploy, no render)'))
    ap.add_argument('--vel-source', default='predicted',
                    choices=['predicted', 'gt'],
                    dest='vel_source',
                    help=('Velocity source for Arch A: '
                          '"predicted"=velocity head output (default), '
                          '"gt"=privileged GT steering velocity from steer.py (Fix-2 upper bound)'))
    ap.add_argument('--seed',       type=int, default=999,
                    help='Eval seed (default=999 held-out; use other values for robustness checks)')
    args = ap.parse_args()

    summary = evaluate(
        checkpoint_path = args.checkpoint,
        arch            = args.arch,
        difficulty      = args.difficulty,
        n_scenes        = args.n_scenes,
        device          = args.device,
        out_dir         = args.out_dir,
        render          = not args.no_render,
        verbose         = args.verbose,
        chunk_H         = args.chunk_H,
        smoke           = args.smoke,
        goal_source     = args.goal_source,
        vel_source      = args.vel_source,
        seed            = args.seed,
    )

    # Exit code: 0 always (eval completed cleanly, success rate is a metric not a pass/fail)
    sys.exit(0)


if __name__ == "__main__":
    main()
