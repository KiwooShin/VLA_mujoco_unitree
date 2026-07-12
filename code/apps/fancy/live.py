"""Live-instruction resolution + interactive session state for the fancy
demo (code/fancy_demo.py, RF-1 split).

Owns:
  - resolve_live_instruction: THE single shared instruction -> target
    resolver used by both _terminal_loop (this module) and the Flask
    /execute route (code/apps/fancy/web.py).
  - FancySceneManager: current-scene state, incl. the FS-1/FS-2 curated
    FIRST_SCENE_SEED first-launch draw.
  - _terminal_loop: interactive terminal fallback UI.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from code.apps.fancy.multi_goal import (
    _extract_goal_hint, _resolve_goal_to_index, _split_multi_goal_parts,
    run_fancy_rollout_multi,
)
from code.apps.fancy.rollout import run_fancy_rollout
from code.apps.fancy.sampling import FIRST_SCENE_SEED, sample_fancy_scene
from code.apps.fancy.sampling_long import sample_fancy_scene_long
from code.apps.fancy.video import _concat_reel


def resolve_live_instruction(instruction: str, scene_cfg: dict) -> dict:
    """NX-15: THE single shared instruction -> target resolver for both live entry
    points (_terminal_loop, Flask /execute). Never used by the scripted/headless
    entry points (run_smoke(), showcase/recording APIs), which continue to pass
    explicit scene_cfg['target_index'] values untouched -- that default remains
    ONLY the fallback for entry points that explicitly pass an index.

    Args:
        instruction: Raw typed instruction, possibly compound.
        scene_cfg: Current scene config dict (must contain `objects`).

    Returns:
        Dict with keys:
          mode:            "single" | "multi" | "clarify" | "no_match" | "no_parse"
          target_indices:  list[int]   resolved object indices, in goal order
          goals:           list[{"color","shape","prompt_part"}]  resolved goal specs
                           (color/shape are the ACTUAL matched object's attributes,
                           not just the raw parsed hint)
          message:         str or None  (clarify question / no-match / no-parse text)
    """
    objects = (scene_cfg or {}).get("objects", [])
    if not objects:
        return dict(mode="no_match", target_indices=[], goals=[],
                     message="No scene loaded yet.")

    parts = _split_multi_goal_parts(instruction)
    if not parts:
        return dict(mode="no_parse", target_indices=[], goals=[], message=(
            "I didn't understand that instruction. Try things like "
            "'find the red ball' or 'go to the orange cube'."
        ))

    hints = [_extract_goal_hint(p) for p in parts]

    for h in hints:
        if h["color"] is None and h["shape"] is None:
            return dict(mode="no_parse", target_indices=[], goals=[], message=(
                f"I didn't understand '{h['prompt_part']}'. Try things like "
                f"'find the red ball' or 'go to the orange cube'."
            ))

    resolved = [_resolve_goal_to_index(h, objects) for h in hints]

    for idx, clarify in resolved:
        if clarify:
            return dict(mode="clarify", target_indices=[], goals=[], message=clarify)

    for (idx, _clarify), h in zip(resolved, hints):
        if idx is None:
            inv = ", ".join(f"{o['color_name']} {o['shape_name']}" for o in objects)
            c   = h["color"] or "?"
            s   = h["shape"] or "object"
            return dict(mode="no_match", target_indices=[], goals=[], message=(
                f"No {c} {s} in this scene; scene has: {inv}"
            ))

    target_indices = [idx for idx, _ in resolved]
    goals = [
        {"color": objects[idx]["color_name"], "shape": objects[idx]["shape_name"],
         "prompt_part": h["prompt_part"]}
        for (idx, _), h in zip(resolved, hints)
    ]
    mode = "multi" if len(target_indices) > 1 else "single"
    return dict(mode=mode, target_indices=target_indices, goals=goals, message=None)


class FancySceneManager:
    """Manages the current fancy search scene."""

    def __init__(self, seed_offset: int = 0) -> None:
        self.seed_offset = seed_offset
        self._ep_count   = 0
        self._scene_cfg  = None

    def new_scene(self, long_dist: bool = True) -> dict:
        """Sample a new scene. FD2: long_dist=True (4-7m) by default.

        FS-1: the very first scene (self._ep_count == 0) draws from the
        curated FIRST_SCENE_SEED instead of the plain [1234+seed_offset, 0]
        sequence, so a fresh --web/terminal launch always opens on a
        verified-good scene. Every later call (manual "New Scene" button,
        the post-rollout auto-resample, terminal 'new') is untouched and
        keeps drawing from the original random sequence -- only this one
        fixed first draw needed curating.

        Args:
            long_dist: Whether to sample from the long-distance (4-7m)
                scene distribution (sample_fancy_scene_long) instead of the
                shorter-range one (sample_fancy_scene).

        Returns:
            The newly sampled scene_cfg dict (also stored on `self._scene_cfg`).
        """
        if self._ep_count == 0:
            seed_seq = np.random.SeedSequence([FIRST_SCENE_SEED, 0])
        else:
            seed_seq = np.random.SeedSequence([1234 + self.seed_offset, self._ep_count])
        rng = np.random.default_rng(seed_seq)
        if long_dist:
            self._scene_cfg = sample_fancy_scene_long(rng, self._ep_count)
        else:
            self._scene_cfg = sample_fancy_scene(rng, self._ep_count)
        self._ep_count  += 1
        tgt = self._scene_cfg['objects'][self._scene_cfg['target_index']]
        print(f"[fancy] New scene ep={self._ep_count-1}: "
              f"target={tgt['color_name']} {tgt['shape_name']}  "
              f"dist={tgt['dist_from_robot']:.2f}m  "
              f"bearing={self._scene_cfg['init_bearing_deg']:.1f}° (out-of-FOV)",
              flush=True)
        return self._scene_cfg

    @property
    def _scene_cfg(self) -> Optional[dict]: return self.__scene_cfg
    @_scene_cfg.setter
    def _scene_cfg(self, v: Optional[dict]) -> None: self.__scene_cfg = v


def _terminal_loop(
    inf: "Inferencer",
    scene_mgr: "FancySceneManager",
    out_dir: str,
    maxsteps: int,
    render_video: bool,
    scenario_title: str = "G1Nav Autonomous Fetch",
) -> None:
    """Simple terminal loop."""
    print("\n" + "=" * 60, flush=True)
    print("G1Nav Fancy Demo — Terminal Mode", flush=True)
    print("Name the object you want, e.g. 'find the red ball'.", flush=True)
    print("Multi-goal: 'find the red ball then find the yellow cube'.", flush=True)
    print("Type 'new' / 'quit'", flush=True)
    print("=" * 60 + "\n", flush=True)

    ep_num = 0
    vid_paths = []

    while True:
        scene_cfg = scene_mgr._scene_cfg
        # NX-15: no "<TARGET" marker -- which object gets pursued is now decided
        # by what the user types, not by the sampler's default target_index.
        print(f"Scene objects:", flush=True)
        for i, o in enumerate(scene_cfg['objects']):
            print(f"  [{i}] {o['color_name']} {o['shape_name']}  "
                  f"dist={o['dist_from_robot']:.2f}m", flush=True)

        try:
            user = input("\nfancy> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user:
            continue
        if user.lower() in ('quit', 'exit', 'q'):
            break
        if user.lower() in ('new', 'reset'):
            scene_mgr.new_scene()
            continue

        # NX-15: parse instruction -> resolve against the CURRENT scene's objects
        parsed = resolve_live_instruction(user, scene_cfg)
        if parsed["mode"] in ("no_parse", "no_match", "clarify"):
            print(f"\nBot: {parsed['message']}\n", flush=True)
            continue

        ep_num += 1
        vid_path = None
        if render_video:
            os.makedirs(out_dir, exist_ok=True)
            vid_path = os.path.join(out_dir, f"fancy_ep{ep_num:03d}.mp4")

        tgt_desc = ' then '.join(f"{g['color']} {g['shape']}" for g in parsed["goals"])
        print(f"\nExecuting: '{user}' -> target: {tgt_desc}", flush=True)
        t0 = time.time()
        try:
            if parsed["mode"] == "multi":
                result = run_fancy_rollout_multi(
                    inf=inf,
                    goals=parsed["goals"],
                    scene_cfg=scene_cfg,
                    maxsteps=maxsteps,
                    render_video=render_video,
                    video_path=vid_path,
                    scenario_title=scenario_title,
                )
            else:
                # NX-15: target comes from the resolved instruction; scene_cfg
                # itself is left untouched (a copy carries the override) so the
                # scene manager's own state (incl. its default target_index,
                # unused here) is unaffected.
                resolved_scene = dict(scene_cfg)
                resolved_scene["target_index"] = parsed["target_indices"][0]
                result = run_fancy_rollout(
                    inf=inf,
                    scene_cfg=resolved_scene,
                    prompt=user,
                    maxsteps=maxsteps,
                    render_video=render_video,
                    scenario_title=scenario_title,
                    video_path=vid_path,
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {'success': False, 'failure_tag': 'error'}
        dt = time.time() - t0

        if parsed["mode"] == "multi":
            status = "SUCCESS" if result.get('success') else "FAILED"
            print(f"\nResult: {status}  total_steps={result.get('total_steps',0)}  wall={dt:.1f}s", flush=True)
            for gi, sr in enumerate(result.get("goal_results", [])):
                g_ok = "OK" if sr.get("success") else f"FAIL({sr.get('failure_tag','?')})"
                g = parsed["goals"][gi] if gi < len(parsed["goals"]) else {"color": "?", "shape": "?"}
                print(f"  sub-goal {gi+1}: {g['color']} {g['shape']}  "
                      f"{g_ok}  dist={sr.get('final_dist',0):.3f}m", flush=True)
        else:
            status = "SUCCESS" if result.get('success') else f"FAILED ({result.get('failure_tag')})"
            print(f"\nResult: {status}  steps={result.get('steps',0)}  "
                  f"dist={result.get('final_dist',0):.3f}m  wall={dt:.1f}s", flush=True)

        if result.get('video_path'):
            print(f"Video: {result['video_path']}", flush=True)
            vid_paths.append(result['video_path'])

        scene_mgr.new_scene()

    if len(vid_paths) > 1:
        reel = os.path.join(out_dir, "fancy_reel.mp4")
        _concat_reel(vid_paths, reel)


