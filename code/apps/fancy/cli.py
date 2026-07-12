"""Headless smoke test + CLI entry point for the fancy demo (code/
fancy_demo.py, RF-1 split).

`main()` is the single dispatch point: headless smoke test (`--smoke`),
Flask web UI (`--web`), or the interactive terminal loop (default) — CLI
surface identical to the pre-split code/fancy_demo.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np

from code.apps.fancy.constants import (
    DIST_MAX_LONG, DIST_MIN_LONG, FANCY_OUT_DIR, GOTO_CKPT_DEFAULT,
    MAXSTEPS_FANCY, WEB_PORT,
)
from code.apps.fancy.live import FancySceneManager, _terminal_loop
from code.apps.fancy.multi_goal import run_fancy_rollout_multi
from code.apps.fancy.rollout import run_fancy_rollout
from code.apps.fancy.sampling_long import sample_fancy_multi_goal_scene, sample_fancy_scene_long
from code.apps.fancy.video import _concat_reel
from code.apps.fancy.web import _start_fancy_web_ui


def run_smoke(
    out_dir: str,
    ckpt_path: str,
    device: str = "cpu",
    maxsteps: int = MAXSTEPS_FANCY,
    render_video: bool = True,
    n_episodes: int = 6,
    scenario_title: str = "G1Nav Autonomous Fetch",
) -> tuple[list[dict], Optional[str]]:
    """FD2 Headless smoke: LONG-DISTANCE search episodes + multi-goal, saved as MP4s.

    Episode plan (default n=6):
      ep0: long single-goal search (4-7m)
      ep1: long single-goal search (4-7m)
      ep2: long single-goal search (4-7m) — smoke verify 1st episode
      ep3: long single-goal search (4-7m)
      ep4: long single-goal search (4-7m)
      ep5: MULTI-GOAL (2 sub-goals, different reliable colors)

    Only SUCCESS episodes go into the showcase reel (fail-filtered).

    Args:
        out_dir: Output directory for per-episode MP4s + the showcase reel.
        ckpt_path: Goto/search checkpoint path passed to Inferencer.
        device: Torch device string ("cpu" or "cuda").
        maxsteps: Hard step cap forwarded to each episode's rollout.
        render_video: Whether to render ego|BEV SBS frames at all.
        n_episodes: Number of smoke episodes to run (last one is multi-goal
            when n_episodes >= 2).
        scenario_title: Scenario name shown on each episode's VF-1 title card.

    Returns:
        Tuple (summary, reel_path): `summary` is the list of per-episode
        result dicts; `reel_path` is the showcase reel's MP4 path, or None
        if no episode produced a video.
    """
    from code.inferencer import Inferencer

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}", flush=True)
    print(f"G1Nav Fancy Demo — FD2 SMOKE TEST ({n_episodes} episodes)", flush=True)
    print(f"  ckpt:      {ckpt_path}", flush=True)
    print(f"  device:    {device}", flush=True)
    print(f"  maxsteps:  {maxsteps}", flush=True)
    print(f"  render:    {render_video}", flush=True)
    print(f"  dist bias: {DIST_MIN_LONG}–{DIST_MAX_LONG}m (long-range)", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Load inferencer ONCE — anti-EGL exhaustion: don't recreate per episode
    print("[smoke] Loading inferencer...", flush=True)
    inf = Inferencer(
        checkpoint_path=ckpt_path,
        arch='A',
        device=device,
        goal_source='classical',
        verbose=False,
    )
    print("[smoke] Inferencer ready", flush=True)

    summary   = []
    vid_paths = []   # all episode clips
    ok_vids   = []   # SUCCESS only → for reel

    # Determine which episodes are multi-goal
    # Last episode is multi-goal if n_episodes >= 2
    multi_goal_ep = n_episodes - 1 if n_episodes >= 2 else -1

    rng_master = np.random.default_rng(np.random.SeedSequence([42, 2026]))

    for ep_i in range(n_episodes):
        ep_seed = int(rng_master.integers(0, 2**31))
        rng     = np.random.default_rng(ep_seed)

        is_multi = (ep_i == multi_goal_ep)
        print(f"\n{'='*50}", flush=True)
        print(f"--- FD2 Episode {ep_i+1}/{n_episodes}"
              f"  ({'MULTI-GOAL' if is_multi else 'SINGLE long-dist'}) ---", flush=True)

        if is_multi:
            # ── Multi-goal episode ──
            scene_cfg = sample_fancy_multi_goal_scene(rng, n_goals=2)
            objs = scene_cfg["objects"]
            # Sub-goals: first 2 objects (both reliable color+shape)
            goals = []
            for gi in range(min(2, len(objs))):
                o = objs[gi]
                goals.append({
                    "color":       o["color_name"],
                    "shape":       o["shape_name"],
                    "prompt_part": f"find the {o['color_name']} {o['shape_name']}",
                })
            prompt = " then ".join(g["prompt_part"] for g in goals)
            print(f"  Multi-goal: {prompt}", flush=True)
            for g in goals:
                oi = next((i for i, o in enumerate(objs)
                           if o["color_name"] == g["color"] and o["shape_name"] == g["shape"]), None)
                if oi is not None:
                    print(f"    sub-goal: {g['color']} {g['shape']}  dist={objs[oi]['dist_from_robot']:.2f}m", flush=True)

            vid_path = None
            if render_video:
                vid_path = os.path.join(out_dir, f"ep{ep_i:02d}_multi_goal.mp4")

            t0 = time.time()
            try:
                result = run_fancy_rollout_multi(
                    inf=inf,
                    goals=goals,
                    scene_cfg=scene_cfg,
                    maxsteps=maxsteps,
                    render_video=render_video,
                    video_path=vid_path,
                    scenario_title=scenario_title,
                )
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}", flush=True)
                traceback.print_exc()
                result = {"success": False, "n_goals": 2, "goal_results": [],
                          "total_steps": 0, "video_path": None}

            dt = time.time() - t0
            ok_tag = "SUCCESS" if result.get("success") else "FAILED"
            print(f"  {ok_tag}  total_steps={result.get('total_steps',0)}  wall={dt:.1f}s", flush=True)
            vid_out = result.get("video_path")
            if vid_out:
                print(f"  Video: {vid_out}", flush=True)
                vid_paths.append(vid_out)
                if result.get("success"):
                    ok_vids.append(vid_out)

            # Per-subgoal info for summary
            sub_info = []
            for gi2, sr in enumerate(result.get("goal_results", [])):
                sub_info.append({
                    "goal_idx": gi2,
                    "color":    goals[gi2]["color"] if gi2 < len(goals) else "?",
                    "shape":    goals[gi2]["shape"] if gi2 < len(goals) else "?",
                    "success":  sr.get("success", False),
                    "steps":    sr.get("steps", 0),
                    "final_dist": sr.get("final_dist", 0.0),
                    "spotted":  sr.get("spotted", False),
                })
            summary.append({
                "ep": ep_i,
                "type": "multi_goal",
                "n_goals": result.get("n_goals", 2),
                "prompt": prompt,
                "success": result.get("success", False),
                "total_steps": result.get("total_steps", 0),
                "sub_goals": sub_info,
                "wall_time_s": dt,
                "video_path": vid_out,
            })

        else:
            # ── Single long-distance episode ──
            scene_cfg = sample_fancy_scene_long(rng, ep_i)
            tgt       = scene_cfg["objects"][scene_cfg["target_index"]]
            dist_m    = tgt["dist_from_robot"]
            prompt    = f"find the {tgt['color_name']} {tgt['shape_name']}"
            bearing   = scene_cfg["init_bearing_deg"]
            print(f"  Target: {tgt['color_name']} {tgt['shape_name']}  "
                  f"dist={dist_m:.2f}m  bearing={bearing:.1f}° (out-of-FOV)", flush=True)
            print(f"  Prompt: '{prompt}'", flush=True)

            vid_path = None
            if render_video:
                vid_path = os.path.join(
                    out_dir,
                    f"ep{ep_i:02d}_{tgt['color_name']}_{tgt['shape_name']}_{dist_m:.1f}m.mp4"
                )

            t0 = time.time()
            try:
                result = run_fancy_rollout(
                    inf=inf,
                    scene_cfg=scene_cfg,
                    prompt=prompt,
                    maxsteps=maxsteps,
                    render_video=render_video,
                    video_path=vid_path,
                    scenario_title=scenario_title,
                )
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}", flush=True)
                traceback.print_exc()
                result = {"success": False, "failure_tag": "error",
                          "steps": 0, "final_dist": 999.0,
                          "spotted": False, "scan_steps": 0}

            dt = time.time() - t0
            ok_tag = "SUCCESS" if result.get("success") else f"FAILED({result.get('failure_tag','?')})"
            print(f"  {ok_tag}  steps={result.get('steps',0)}  "
                  f"dist={result.get('final_dist',0):.3f}m  wall={dt:.1f}s  "
                  f"spotted={result.get('spotted',False)}  scan_steps={result.get('scan_steps',0)}", flush=True)
            vid_out = result.get("video_path")
            if vid_out:
                print(f"  Video: {vid_out}", flush=True)
                vid_paths.append(vid_out)
                if result.get("success"):
                    ok_vids.append(vid_out)

            summary.append({
                "ep": ep_i,
                "type": "single_long",
                "prompt": prompt,
                "color": tgt["color_name"],
                "shape": tgt["shape_name"],
                "target_dist_m": dist_m,
                "init_bearing_deg": bearing,
                "success": result.get("success", False),
                "failure_tag": result.get("failure_tag", "?"),
                "steps": result.get("steps", 0),
                "final_dist": result.get("final_dist", 0.0),
                "spotted": result.get("spotted", False),
                "scan_steps": result.get("scan_steps", 0),
                "wall_time_s": dt,
                "video_path": vid_out,
            })

    # ── Showcase reel — SUCCESS episodes only ──
    reel_path = None
    reel_src  = ok_vids if ok_vids else vid_paths   # fall back to all if none succeeded
    if reel_src:
        reel_path = os.path.join(out_dir, "fancy_showcase_reel.mp4")
        reel_path = _concat_reel(reel_src, reel_path)
        print(f"\n[FD2] Showcase reel ({len(reel_src)} clips): {reel_path}", flush=True)

    # ── Print summary table ──
    print(f"\n{'='*60}", flush=True)
    print("FD2 FANCY SMOKE SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    n_ok = sum(1 for s in summary if s["success"])
    print(f"  Success: {n_ok}/{len(summary)}", flush=True)
    for s in summary:
        if s["type"] == "multi_goal":
            ok_str = "OK" if s["success"] else "FAIL"
            print(f"  ep{s['ep']} [MULTI-{s['n_goals']}]: {ok_str:4s}  "
                  f"steps={s['total_steps']:5d}  video={s.get('video_path','none')}", flush=True)
            for sg in s.get("sub_goals", []):
                sg_ok = "OK" if sg["success"] else "FAIL"
                print(f"    sub-goal {sg['goal_idx']+1}: {sg['color']} {sg['shape']}  "
                      f"{sg_ok}  spotted={sg['spotted']}  dist={sg['final_dist']:.3f}m", flush=True)
        else:
            ok_str = "OK" if s["success"] else f"FAIL({s.get('failure_tag','?')})"
            print(f"  ep{s['ep']} [SINGLE  {s.get('target_dist_m',0):.1f}m]: "
                  f"{s['color']:7s} {s['shape']:8s}  {ok_str:20s}  "
                  f"steps={s.get('steps',0):5d}  spotted={s.get('spotted','?')}  "
                  f"video={s.get('video_path','none')}", flush=True)
    if reel_path:
        print(f"\n  Showcase reel: {reel_path}", flush=True)

    # Save summary JSON
    summary_path = os.path.join(out_dir, "fancy_showcase_summary_fd2.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON: {summary_path}", flush=True)

    return summary, reel_path



def main() -> None:
    """CLI entry point: dispatches to the headless smoke test, the Flask web
    UI, or the interactive terminal loop, based on the parsed arguments."""
    parser = argparse.ArgumentParser(description="G1Nav Fancy Demo")
    parser.add_argument("--smoke",     action="store_true", help="Headless smoke test")
    parser.add_argument("--web",       action="store_true", help="Flask web UI")
    parser.add_argument("--out",       default=FANCY_OUT_DIR, help="Output dir")
    parser.add_argument("--device",    default="cuda" if _has_cuda() else "cpu")
    parser.add_argument("--ckpt",      default=GOTO_CKPT_DEFAULT,
                        help="Goto/search checkpoint path (default: checkpoint/goto_best.pt)")
    parser.add_argument("--port",      type=int, default=WEB_PORT)
    parser.add_argument("--maxsteps",  type=int, default=MAXSTEPS_FANCY)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--n-smoke",   type=int, default=6,
                        help="Number of smoke episodes (FD2: last ep is multi-goal)")
    parser.add_argument("--scenario-title", default="G1Nav Autonomous Fetch",
                        help="VF-1: scenario name shown on the pre-roll title card")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.smoke:
        run_smoke(
            out_dir=args.out,
            ckpt_path=args.ckpt,
            device=args.device,
            maxsteps=args.maxsteps,
            render_video=not args.no_render,
            n_episodes=args.n_smoke,
            scenario_title=args.scenario_title,
        )
        return

    # Web UI / interactive mode
    from code.inferencer import Inferencer

    print("[fancy_demo] Loading inferencer...", flush=True)
    inf = Inferencer(
        checkpoint_path=args.ckpt,
        arch='A',
        device=args.device,
        goal_source='classical',
        verbose=False,
    )
    print("[fancy_demo] Inferencer ready", flush=True)

    scene_mgr = FancySceneManager(seed_offset=0)
    scene_mgr.new_scene()

    if args.web:
        _start_fancy_web_ui(
            inf=inf,
            scene_manager=scene_mgr,
            out_dir=args.out,
            port=args.port,
            maxsteps=args.maxsteps,
            render_video=not args.no_render,
            scenario_title=args.scenario_title,
        )
        print(f"[fancy_demo] Web UI running at http://localhost:{args.port}", flush=True)
        print("[fancy_demo] Open browser → type 'find the red ball' → watch ego|BEV stream", flush=True)
        print("[fancy_demo] Press Ctrl-C to quit", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[fancy_demo] Shutting down.", flush=True)
    else:
        # Interactive terminal fallback
        _terminal_loop(inf, scene_mgr, args.out, args.maxsteps, not args.no_render,
                       scenario_title=args.scenario_title)


def _has_cuda() -> bool:
    """Return True if a CUDA device is available to torch, else False."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
