"""Terminal REPL, canned smoke test, and CLI entry point for the REPL demo
(code/demo.py, RF-1 split).

`main()` is the single dispatch point: canned smoke test (`--smoke`), Flask
web UI (`--web`), or the interactive terminal REPL (default) — CLI surface
identical to the pre-split code/demo.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np

from code.apps.repl.constants import (
    DEMO_OUT_DIR, MAXSTEPS_GOTO, MAXSTEPS_MANEUVER, WEB_PORT, _check_cuda,
)
from code.apps.repl.executor import EventBus, Executor
from code.apps.repl.planner import Planner, SceneManager
from code.apps.repl.web import _start_web_ui


# ---------------------------------------------------------------------------
# Terminal REPL
# ---------------------------------------------------------------------------
def _terminal_repl(
    scene_manager: SceneManager,
    planner: Planner,
    executor: Executor,
    bus: EventBus,
    out_dir: str,
) -> None:
    """Interactive terminal REPL."""
    print("\n" + "=" * 60, flush=True)
    print("G1Nav Interactive Demo REPL", flush=True)
    print("Commands: <instruction> | 'new' | 'scene' | 'quit'", flush=True)
    print("=" * 60, flush=True)

    # Show initial scene
    scene_manager.new_scene()
    print(f"\n{scene_manager.describe_scene()}\n", flush=True)

    pending_clarify = None   # current clarification question (if any)
    pending_instr   = None   # original instruction that triggered clarification

    while True:
        try:
            if pending_clarify:
                prompt = "clarify> "
            else:
                prompt = "demo> "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[demo] Goodbye!", flush=True)
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("[demo] Goodbye!", flush=True)
            break

        if user_input.lower() in ("new", "reset"):
            scene_manager.new_scene()
            pending_clarify = None
            pending_instr   = None
            print(f"\n{scene_manager.describe_scene()}\n", flush=True)
            continue

        if user_input.lower() in ("scene", "objects", "describe"):
            print(f"\n{scene_manager.describe_scene()}\n", flush=True)
            continue

        if user_input.lower() in ("cancel", "abort"):
            pending_clarify = None
            pending_instr   = None
            print("\nBot: Cancelled. Type a new instruction.\n", flush=True)
            continue

        if user_input.lower() in ("help", "?"):
            print("\nBot: Instructions I understand:", flush=True)
            print("  goto:     'go to the red ball'  |  'navigate to the orange cone'", flush=True)
            print("  maneuver: 'turn left after the blue cube'  |  'pass the red cylinder then turn right'", flush=True)
            print("  search:   'find the red ball'  |  'look for the orange cube'", flush=True)
            print("  multi:    'go to the red ball then find the blue cube'", flush=True)
            print("  cmds:     'new' (new scene)  |  'scene' (show objects)  |  'quit'", flush=True)
            print(flush=True)
            continue

        # If awaiting clarification, use planner.parse_clarification()
        if pending_clarify:
            original  = pending_instr or user_input
            goals, clarify = planner.parse_clarification(original, user_input)
            if clarify:
                print(f"\nBot: {clarify}\n", flush=True)
                pending_clarify = clarify
                pending_instr   = original
                continue
            pending_clarify = None
            pending_instr   = None
        else:
            # Parse fresh instruction
            goals, clarify = planner.parse(user_input)

        if clarify:
            print(f"\nBot: {clarify}\n", flush=True)
            pending_clarify = clarify
            pending_instr   = user_input
            continue

        if not goals:
            print("\nBot: I didn't understand that. Type 'help' for examples.\n", flush=True)
            continue

        # Show plan
        print(f"\nBot: Plan ({len(goals)} step{'s' if len(goals) > 1 else ''}):", flush=True)
        for i, g in enumerate(goals):
            print(f"  [{i+1}] {g}", flush=True)
        print(flush=True)

        # Execute
        print("Executing...", flush=True)
        t0 = time.time()
        results = executor.execute(goals)
        dt = time.time() - t0

        # Summary
        print(f"\n--- Episode Summary ({dt:.1f}s) ---", flush=True)
        for i, (g, r) in enumerate(zip(goals, results)):
            status = "SUCCESS" if r.get("success") else f"FAILED ({r.get('failure_tag', '?')})"
            vid    = r.get("video_path", None)
            vid_msg = f" → video: {vid}" if vid else ""
            print(f"  [{i+1}] {g}: {status}{vid_msg}", flush=True)

        n_success = sum(1 for r in results if r.get("success"))
        print(f"\nTotal: {n_success}/{len(results)} succeeded", flush=True)

        # New scene prompt
        print("\nGenerating new scene for next episode...", flush=True)
        scene_manager.new_scene()
        print(f"\n{scene_manager.describe_scene()}\n", flush=True)


# ---------------------------------------------------------------------------
# Smoke test (canned instructions)
# ---------------------------------------------------------------------------
def _smoke_test(out_dir: str, device: str, maxsteps_goto: int, maxsteps_maneuver: int,
                render_video: bool = True) -> list[dict[str, Any]]:
    """
    Run 4 canned instructions end-to-end headless (S10 polish):
      1. goto (demo-distance, 4-9m)
      2. maneuver (turn after landmark)
      3. search (out-of-FOV find)
      4. multi-goal compound (goto then search)
    Saves videos and prints summary.

    Returns:
        List of per-goal result summaries (label, instruction, skill, success,
        failure_tag, steps, wall_time_s, video_path).
    """
    print("\n" + "=" * 60, flush=True)
    print("G1Nav Demo — SMOKE TEST (4 canned instructions incl. multi-goal)", flush=True)
    print("=" * 60 + "\n", flush=True)

    os.makedirs(out_dir, exist_ok=True)

    bus           = EventBus()
    scene_manager = SceneManager(difficulty="demo", seed_offset=42)
    planner       = Planner(scene_manager)
    executor      = Executor(
        scene_manager=scene_manager,
        bus=bus,
        device=device,
        render_video=render_video,
        out_dir=out_dir,
        maxsteps_goto=maxsteps_goto,
        maxsteps_maneuver=maxsteps_maneuver,
    )

    from code.maneuver_scene import sample_maneuver_scene as _sample_maneuver
    from code.maneuver_scene import derive_rng as _derive_maneuver_rng
    from code.scene import sample_scene as _sample_scene, derive_rng as _derive_rng
    from code.eval_search import sample_search_scene

    # Build test cases
    # Case 1: demo-distance goto
    scene_manager.difficulty = "demo"
    scene_manager.new_scene()
    sc_goto = scene_manager.scene_cfg
    tgt_goto = sc_goto["objects"][sc_goto["target_index"]]
    instr_goto = f"go to the {tgt_goto['color_name']} {tgt_goto['shape_name']}"

    # Case 2: maneuver
    _mrng = _derive_maneuver_rng(999, 0)
    sc_maneuver = _sample_maneuver(_mrng)
    lm = sc_maneuver['objects'][sc_maneuver['landmark_index']]
    instr_maneuver = sc_maneuver['instruction']

    # Case 3: search (out-of-FOV) — use easy difficulty scene + search skill
    _srng = np.random.default_rng(np.random.SeedSequence([999, 0]))
    sc_search = sample_search_scene(_srng, 0)
    tgt_search = sc_search["objects"][sc_search["target_index"]]
    instr_search = f"find the {tgt_search['color_name']} {tgt_search['shape_name']}"

    # Case 4: multi-goal — "go to X then find Y" (uses two separate skills in one episode)
    # Use the goto scene for the first goal; pick a different object for search
    sc_multi = dict(sc_goto)
    objs_multi = sc_goto["objects"]
    # First goal: goto the target
    tgt1 = objs_multi[sc_goto["target_index"]]
    # Second goal: search for another object (non-target)
    tgt2_idx = 1 if sc_goto["target_index"] != 1 else 2
    tgt2_idx = min(tgt2_idx, len(objs_multi) - 1)
    tgt2 = objs_multi[tgt2_idx]
    instr_multi = (f"go to the {tgt1['color_name']} {tgt1['shape_name']} "
                   f"then find the {tgt2['color_name']} {tgt2['shape_name']}")

    test_cases = [
        {"label": "goto_demo_long",   "scene": sc_goto,     "instr": instr_goto,     "difficulty": "demo"},
        {"label": "maneuver",         "scene": sc_maneuver, "instr": instr_maneuver, "difficulty": "demo"},
        {"label": "search_outofFOV",  "scene": sc_search,   "instr": instr_search,   "difficulty": "search"},
        {"label": "multi_goal",       "scene": sc_multi,    "instr": instr_multi,    "difficulty": "demo"},
    ]

    summary = []
    for i, tc in enumerate(test_cases):
        print(f"\n--- Test {i+1}/{len(test_cases)}: {tc['label']} ---", flush=True)
        print(f"Instruction: '{tc['instr']}'", flush=True)

        # Update scene manager to use the prepared scene
        scene_manager._scene_cfg = tc["scene"]
        scene_manager.difficulty = tc["difficulty"]

        goals, clarify = planner.parse(tc["instr"])
        if clarify:
            print(f"Planner needs clarification: {clarify}", flush=True)
            # Try to resolve with scene's target info
            tgt = tc["scene"]["objects"][tc["scene"].get("target_index", 0)]
            fallback_instr = f"go to the {tgt['color_name']} {tgt['shape_name']}"
            goals, clarify = planner.parse(fallback_instr)

        if not goals:
            print(f"No goals parsed — SKIP", flush=True)
            summary.append({"label": tc["label"], "success": False, "reason": "parse_fail",
                             "instruction": tc["instr"]})
            continue

        print(f"Plan ({len(goals)} step{'s' if len(goals) > 1 else ''}): {[str(g) for g in goals]}", flush=True)

        t0 = time.time()
        try:
            results = executor.execute(goals)
        except Exception as e:
            print(f"Executor error: {e}", flush=True)
            results = [{"success": False, "failure_tag": "error", "steps": 0}] * len(goals)
        dt = time.time() - t0

        for g, r in zip(goals, results):
            vid = r.get("video_path")
            print(
                f"  {g}: {'SUCCESS' if r.get('success') else 'FAILED'}  "
                f"steps={r.get('steps', 0)}  time={dt:.1f}s"
                + (f"  video={vid}" if vid else ""),
                flush=True,
            )
            summary.append({
                "label": tc["label"],
                "instruction": tc["instr"],
                "skill": g.skill,
                "success": r.get("success", False),
                "failure_tag": r.get("failure_tag", ""),
                "steps": r.get("steps", 0),
                "wall_time_s": dt,
                "video_path": vid,
            })

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("SMOKE TEST SUMMARY", flush=True)
    print("=" * 60, flush=True)
    n_ok = sum(1 for s in summary if s.get("success"))
    print(f"Success: {n_ok}/{len(summary)}", flush=True)
    for s in summary:
        status = "OK" if s.get("success") else f"FAIL ({s.get('failure_tag', '?')})"
        print(f"  {s['label']:25s} [{s.get('skill','?'):8s}] {status:30s}  video={s.get('video_path')}", flush=True)

    # Save summary JSON
    summary_path = os.path.join(out_dir, "smoke_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}", flush=True)

    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point: parses args and runs smoke test, web UI, or terminal REPL."""
    parser = argparse.ArgumentParser(description="G1Nav Interactive Demo")
    parser.add_argument("--web",    action="store_true", help="Start web UI on port 5000")
    parser.add_argument("--port",   type=int, default=WEB_PORT)
    parser.add_argument("--smoke",  action="store_true", help="Run canned smoke test and exit")
    parser.add_argument("--out",    default=str(DEMO_OUT_DIR), help="Output dir for videos")
    parser.add_argument("--device", default="cuda" if _check_cuda() else "cpu")
    parser.add_argument("--difficulty", default="demo", choices=["easy", "demo"])
    # H1: default changed to 'demo' — showcases 4-9m long walks with V2/V3 grounding
    parser.add_argument("--maxsteps-goto", type=int, default=MAXSTEPS_GOTO)  # H1: default=1400 for demo
    parser.add_argument("--maxsteps-maneuver", type=int, default=MAXSTEPS_MANEUVER)
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering (faster)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Smoke test mode
    if args.smoke:
        _smoke_test(
            out_dir=args.out,
            device=args.device,
            maxsteps_goto=args.maxsteps_goto,
            maxsteps_maneuver=args.maxsteps_maneuver,
            render_video=(not args.no_render),
        )
        return

    # Normal REPL / Web UI mode
    bus           = EventBus()
    scene_manager = SceneManager(difficulty=args.difficulty, seed_offset=0)
    planner       = Planner(scene_manager)
    executor      = Executor(
        scene_manager=scene_manager,
        bus=bus,
        device=args.device,
        render_video=(not args.no_render),
        out_dir=args.out,
        maxsteps_goto=args.maxsteps_goto,
        maxsteps_maneuver=args.maxsteps_maneuver,
    )

    # Pre-load initial scene
    scene_manager.new_scene()

    if args.web:
        _start_web_ui(
            bus=bus,
            executor=executor,
            planner=planner,
            scene_manager=scene_manager,
            port=args.port,
        )
        print(f"[demo] Web UI running at http://localhost:{args.port}", flush=True)
        print("[demo] Press Ctrl-C to quit", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[demo] Shutting down.", flush=True)
    else:
        _terminal_repl(
            scene_manager=scene_manager,
            planner=planner,
            executor=executor,
            bus=bus,
            out_dir=args.out,
        )


if __name__ == "__main__":
    main()
