"""Multi-goal instruction parsing + sequential-sub-goal rollout for the
fancy demo (code/fancy_demo.py, RF-1 split).

`resolve_live_instruction` (the shared live-entry-point resolver) is owned
by code/apps/fancy/live.py instead of here, per the RF-1 file-split plan;
it imports `_split_multi_goal_parts`/`_extract_goal_hint`/
`_resolve_goal_to_index` from this module.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

import numpy as np

from code.apps.fancy.constants import MAXSTEPS_FANCY, RELIABLE_SHAPES
from code.apps.fancy.rollout import run_fancy_rollout
from code.apps.fancy.video import _write_fancy_video

# NX-15: Live instruction parsing + scene resolution — the ONE shared function
# used by BOTH _terminal_loop() and the Flask /execute handler (_do_rollout()).
# Fixes docs/dr1_demo_reliability.md's headline finding: previously neither live
# path parsed the typed instruction at all (target always came from
# scene_cfg['target_index']), and this file's only parser, _parse_multi_goal_fancy()
# (below), was dead code never called from any live entry point.
#
# _parse_multi_goal_fancy()'s clause-splitting regex (then / and-then / after-that /
# next) is reused verbatim and kept under its original name/signature for backward
# compat. Its "does this word belong to the known COLORS/SHAPES set" philosophy is
# generalized from an adjacent-pair regex to a whole-clause word scan
# (_extract_goal_hint) so word order ("the ball that is red"), inserted adjectives
# ("the reddish ball" — doesn't false-match "red" thanks to \b), and "-colored"
# phrasing all resolve instead of silently returning []. Ambiguity handling
# (_resolve_goal_to_index) mirrors demo.py's Planner._resolve_referent(): unique
# (color, shape) match -> go; multiple candidates -> score by how many of the
# OTHER words in the clause match each candidate's attributes, tie -> one-line
# clarification question; zero candidates -> "no <X> in this scene" + inventory.

_ALL_COLORS: list[str] = ["red", "yellow", "blue", "green", "orange", "purple", "cyan"]
_ALL_SHAPES: list[str] = RELIABLE_SHAPES  # ["ball", "cube", "cylinder", "cone"] -- full shape set


def _split_multi_goal_parts(instruction: str) -> List[str]:
    """Split a compound instruction on then/and-then/after-that/next conjunctions.
    Same regex as demo.py's Planner.parse() / the original _parse_multi_goal_fancy()."""
    parts = re.split(
        r'\bthen\b|,\s*then\s*|\band\s+then\b|\band\s+after\s+that\b'
        r'|\bafter\s+that\b|\bafterwards\b|\bnext\b',
        instruction, flags=re.IGNORECASE
    )
    return [p.strip() for p in parts if p.strip()]


def _extract_goal_hint(part: str) -> dict:
    """Extract a best-effort (color, shape) hint from one instruction clause.

    Scans the whole clause for known color/shape words (order-independent --
    handles "red ball", "the ball that is red", "red-colored ball", etc.) rather
    than requiring the two words to be adjacent. `color`/`shape` are set only
    when exactly one candidate word of that kind is present in the clause;
    `colors_mentioned`/`shapes_mentioned` keep the full sets for ambiguity
    scoring (see _resolve_goal_to_index).

    Args:
        part: One instruction clause (already split on then/and-then/etc.).

    Returns:
        Dict with keys `color` (str or None), `shape` (str or None),
        `colors_mentioned` (set[str]), `shapes_mentioned` (set[str]), and
        `prompt_part` (the stripped input clause).
    """
    part_l = part.lower()
    colors_mentioned = {c for c in _ALL_COLORS if re.search(r'\b' + c + r'\b', part_l)}
    shapes_mentioned = {s for s in _ALL_SHAPES if re.search(r'\b' + s + r'\b', part_l)}
    color = next(iter(colors_mentioned)) if len(colors_mentioned) == 1 else None
    shape = next(iter(shapes_mentioned)) if len(shapes_mentioned) == 1 else None
    return {
        "color": color, "shape": shape,
        "colors_mentioned": colors_mentioned, "shapes_mentioned": shapes_mentioned,
        "prompt_part": part.strip(),
    }


def _parse_multi_goal_fancy(instruction: str) -> List[dict]:
    """Rule-based multi-goal parser for fancy_demo (kept under its original name and
    signature for backward compat). Splits on "then" conjunctions, extracts
    (color, shape) per part. Returns list of dicts: [{color, shape, prompt_part}, ...]

    NX-15: now implemented on top of _split_multi_goal_parts()/_extract_goal_hint()
    (the shared internals also used by resolve_live_instruction() below) instead of
    its own standalone regex -- same public contract as before.

    Args:
        instruction: Raw typed instruction, possibly compound (e.g. "find the
            red ball then find the yellow cube").

    Returns:
        List of dicts [{color, shape, prompt_part}, ...], one per clause that
        yielded at least one recognized color/shape word.
    """
    goals = []
    for part in _split_multi_goal_parts(instruction):
        hint = _extract_goal_hint(part)
        if hint["color"] or hint["shape"]:
            goals.append({"color": hint["color"], "shape": hint["shape"],
                           "prompt_part": hint["prompt_part"]})
    return goals


def _resolve_goal_to_index(hint: dict, objects: List[dict]) -> tuple:
    """Resolve one (color, shape) hint against the current scene's object list.

    Args:
        hint: One _extract_goal_hint() result (`color`, `shape`,
            `colors_mentioned`, `shapes_mentioned`, `prompt_part`).
        objects: The current scene's object list (`color_name`/`shape_name`/
            `dist_from_robot` per object).

    Returns:
        Tuple (obj_idx, clarify_question):
          (idx, None)   -- unambiguous match (or unique best-attribute-match winner)
          (None, msg)   -- ambiguous, msg is a one-line clarification question
          (None, None)  -- no matching object in the scene
    """
    color, shape = hint["color"], hint["shape"]
    if color is None and shape is None:
        return None, None

    candidates = [
        i for i, o in enumerate(objects)
        if (color is None or o["color_name"] == color)
        and (shape is None or o["shape_name"] == shape)
    ]
    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, None

    # Ambiguous (e.g. "the ball" with two balls in the scene): pick the
    # candidate matching more of the OTHER words mentioned in the clause;
    # only ask for clarification if that still leaves a tie.
    colors_m, shapes_m = hint["colors_mentioned"], hint["shapes_mentioned"]
    scored = [
        (int(objects[i]["color_name"] in colors_m) + int(objects[i]["shape_name"] in shapes_m), i)
        for i in candidates
    ]
    best = max(sc for sc, _ in scored)
    tied = [i for sc, i in scored if sc == best]
    if len(tied) == 1:
        return tied[0], None

    descs = ", ".join(
        f"{objects[i]['color_name']} {objects[i]['shape_name']} (at {objects[i]['dist_from_robot']:.1f}m)"
        for i in tied
    )
    return None, f"Multiple matching objects found: {descs}. Which one? (say the color and the shape)"

def run_fancy_rollout_multi(
    inf: "Inferencer",
    goals: List[dict],           # [{color, shape, prompt_part}, ...]
    scene_cfg: dict,
    maxsteps: int = MAXSTEPS_FANCY,
    render_video: bool = True,
    video_path: Optional[str] = None,
    frame_callback: "Callable[..., None] | None" = None,
    # VF-1 item 5: title card params, forwarded to the FIRST sub-goal's
    # run_fancy_rollout() call (which is the only one that renders a title card).
    scenario_title: str = "G1Nav Autonomous Fetch",
) -> dict:
    """Execute sequential sub-goals on the SAME scene.

    For each sub-goal:
      - Sets scene target_index to the matching object
      - Runs run_fancy_rollout() with path_trail carried over
      - BEV shows current target ring + completed target dots + goal N/M banner

    Args:
        inf: Inferencer instance (goal_source='classical').
        goals: List of dicts [{color, shape, prompt_part}, ...], in the order
            the sub-goals should be pursued.
        scene_cfg: Scene config dict shared by all sub-goals (only
            `target_index` is overridden per sub-goal).
        maxsteps: Hard step cap forwarded to each sub-goal's run_fancy_rollout().
        render_video: Whether to render ego|BEV SBS frames at all.
        video_path: Output MP4 path for the combined multi-goal video, or None
            to skip writing one.
        frame_callback: Optional callable forwarded to each sub-goal's
            run_fancy_rollout(), invoked each rendered step with
            (sbs_bgr, state, dist, step).
        scenario_title: Scenario name shown on the VF-1 title card, forwarded
            to the FIRST sub-goal's run_fancy_rollout() call.

    Returns:
        Dict with keys: success (overall), n_goals, goal_results (per-goal
        result dicts), total_steps, video_path, frames_count.
    """
    n_goals = len(goals)
    objects = scene_cfg["objects"]

    def _find_obj(color: str, shape: str) -> Optional[int]:
        """Find object index in scene by color+shape (first match)."""
        for i, o in enumerate(objects):
            if o["color_name"] == color and o["shape_name"] == shape:
                return i
        # Fuzzy: color only
        for i, o in enumerate(objects):
            if o["color_name"] == color:
                return i
        return None

    combined_frames = []  # all SBS frames across sub-goals
    path_trail = None
    completed_targets = []
    all_results = []
    overall_success = True
    # VF-3 (docs/vf3_bev_fixes.md, user feedback #3): carries the LIVE MuJoCo
    # sim (+ carried policy state) from one sub-goal's run_fancy_rollout()
    # call to the next, so the robot's actual physical state (position,
    # heading, joint angles/velocities) continues instead of being reset back
    # to the scene's ORIGINAL start for every sub-goal (the bug being fixed
    # here). None on the very first call (nothing to resume yet).
    live_ctx = None

    for gi, goal in enumerate(goals):
        color = goal["color"]
        shape = goal["shape"]
        prompt_part = goal.get("prompt_part", f"find the {color} {shape}")

        # Override scene target_index for this sub-goal
        obj_idx = _find_obj(color, shape)
        if obj_idx is None:
            print(f"  [multi] sub-goal {gi+1}/{n_goals}: '{color} {shape}' NOT in scene — SKIP", flush=True)
            all_results.append({"success": False, "failure_tag": "not_in_scene", "steps": 0})
            overall_success = False
            continue

        sub_scene = dict(scene_cfg)
        sub_scene["target_index"] = obj_idx
        tgt_obj = objects[obj_idx]
        tgt_xy = np.array([tgt_obj["x"], tgt_obj["y"]])

        print(f"\n  [multi] sub-goal {gi+1}/{n_goals}: '{color} {shape}' at "
              f"dist={tgt_obj['dist_from_robot']:.2f}m", flush=True)

        # Video path for this sub-goal clip (no write if part of multi)
        sub_vid_path = None  # we collect frames, write combined video later

        # VF-3: only the LAST sub-goal lets run_fancy_rollout tear down its
        # own sim/renderer as before -- every earlier sub-goal keeps it alive
        # so the NEXT one can resume from it (scene objects are untouched
        # across sub-goals -- `sub_scene` only ever changes `target_index` --
        # so goal-1's target stays exactly where it was; only the robot's
        # physical state and the goal query change).
        is_last_goal = (gi == n_goals - 1)
        result = run_fancy_rollout(
            inf=inf,
            scene_cfg=sub_scene,
            prompt=f"[{gi+1}/{n_goals}] {prompt_part}",
            maxsteps=maxsteps,
            render_video=render_video,
            video_path=None,   # don't save sub-clip yet
            frame_callback=frame_callback,
            goal_idx=gi,
            n_goals=n_goals,
            path_trail_in=path_trail,
            completed_targets=completed_targets,
            scenario_title=scenario_title,
            title_instruction=" then ".join(g.get("prompt_part", f"{g['color']} {g['shape']}") for g in goals),
            resume_ctx=live_ctx,
            keep_alive=(not is_last_goal),
        )

        # Carry trail forward
        path_trail = result.get("path_trail_out", path_trail)

        # Accumulate frames
        if render_video:
            combined_frames.extend(result.get("frames_sbs", []))

        # Mark completed
        if result.get("success"):
            completed_targets.append(tgt_xy.copy())
        else:
            overall_success = False

        all_results.append(result)
        print(f"  [multi] sub-goal {gi+1}/{n_goals} => {result.get('failure_tag')}  "
              f"dist={result.get('final_dist',0):.3f}m", flush=True)

        # VF-3: hand the live sim to the next sub-goal. A missing 'live_ctx'
        # on a non-last goal means run_fancy_rollout couldn't (or the robot
        # fell and there's nothing physically sensible to continue) -- stop
        # the sequence honestly rather than silently rebuilding a fresh scene
        # for the remaining goals (which would reintroduce the exact
        # teleport-back-to-start bug this fix addresses).
        if is_last_goal:
            live_ctx = None
        else:
            live_ctx = result.get("live_ctx")
            if live_ctx is None:
                print(f"  [multi] sub-goal {gi+1}/{n_goals} ended with no continuable "
                      f"sim state (failure_tag={result.get('failure_tag')}, "
                      f"fell={result.get('fell')}) — stopping multi-goal sequence",
                      flush=True)
                overall_success = False
                break

    # VF-3: defensive cleanup -- if the loop ended (break, or the true last
    # goal was skipped via `continue` above) while a live sim was still open,
    # close it here so its EGL renderer doesn't leak.
    if live_ctx is not None:
        try:
            live_ctx['renderer'].close()
        except Exception:
            pass

    # Write combined video
    out_vid = None
    if render_video and video_path and combined_frames:
        out_vid = _write_fancy_video(combined_frames, video_path)

    return dict(
        success=overall_success,
        n_goals=n_goals,
        goal_results=all_results,
        total_steps=sum(r.get("steps", 0) for r in all_results),
        video_path=out_vid,
        frames_count=len(combined_frames),
    )
