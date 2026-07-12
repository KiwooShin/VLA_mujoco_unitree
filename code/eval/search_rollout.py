"""code.eval.search_rollout — standalone search-skill rollout loop (harness).

Split out of the original ``eval_search.py`` (RF-1): this is the outer
per-episode harness — build/settle the env (``code.eval.search_rollout_state``),
run the per-step control logic (``code.eval.search_rollout_step``) in a loop,
then compute the final success/failure verdict and (optionally) write the
video. The setup and per-step logic were extracted into their own modules
purely to fit the <500-line-per-file budget; nothing about the control flow
or numeric logic changed from the pre-RF-1 monolithic function.

This is assembled PURELY behaviorally — the existing Inferencer with goal_source='classical'
already implements a (differently-bounded, ±90°) scan-and-acquire mechanism for the demo
skill's own in-rollout scan (H3); this file's standalone rollout is search-specific:
  1. target starts OUT of initial FOV  → grounding.not_visible=True → scan_active=True
  2. student-driven bounded bidirectional scan (inject wz into action head, WBC-free)
     while checking grounding every cycle
  3. when target detected AND bearing < 40°  → scan_active=False → GOTO begins
  4. classical HSV grounding guides approach → stop within STOP_R

NOTE (no-consolidation invariant): this rollout loop is intentionally NOT
shared with inferencer.py's H3 in-rollout scan or fancy_demo.py's own copy —
they stay independently duplicated per docs/refactor_plan.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from code.inferencer import FALL_HEIGHT, _write_video

from code.eval.search_rollout_state import _setup_search_rollout
from code.eval.search_rollout_step import _search_step


def _run_search_rollout(
    inf,                             # Inferencer instance (goal_source='classical')
    scene_cfg:    dict,
    instruction:  str,
    maxsteps:     int = 2000,
    render_video: bool = False,
    video_path:   str | None = None,
) -> dict:
    """Run one search rollout using the existing Inferencer.

    Tracks when the target is first spotted (scan_active → False).

    Args:
        inf: Inferencer instance (goal_source='classical') providing the
            model and any cached keyframe/action-stats state.
        scene_cfg: Scene configuration from sample_search_scene (robot pose,
            objects, target_index, stop_r, etc.).
        instruction: Natural-language instruction for the episode (unused
            by the rollout logic itself; kept for caller-side logging).
        maxsteps: Maximum number of control steps before terminating.
        render_video: Whether to render ego/third-person frames for video.
        video_path: Output path for the rendered video, or None to skip
            writing.

    Returns:
        Dict with: success, spotted, scan_steps, failure_tag, steps,
        final_dist, fell, ms_per_step, video_path, avoid_bias_active_frac.
    """
    setup = _setup_search_rollout(inf, scene_cfg)
    if setup.early_result is not None:
        return setup.early_result

    for step in range(maxsteps):
        stop = _search_step(setup, inf, step, render_video)
        if stop:
            break

    setup.renderer.close()

    data_mj      = setup.data_mj
    final_height = float(data_mj.qpos[2])
    upright      = final_height >= FALL_HEIGHT and not setup.fell
    final_dist   = float(np.linalg.norm(data_mj.qpos[0:2] - setup.target_xy))
    reached      = (final_dist < setup.stop_r) and upright
    success      = setup.spotted and reached

    if setup.fell:
        failure_tag = 'fall'
    elif not setup.spotted:
        failure_tag = 'scan_timeout'
    elif not reached:
        failure_tag = 'didnt-reach'
    else:
        failure_tag = 'success'

    ms_per_step = float(np.mean(setup.step_times)) if setup.step_times else 0.0

    # Write video
    out_vid = None
    if render_video and video_path and setup.frames_ego:
        _write_video(setup.frames_ego, setup.frames_tp, video_path)
        out_vid = video_path

    return dict(
        success=success,
        spotted=setup.spotted,
        scan_steps=setup.scan_steps,
        failure_tag=failure_tag,
        steps=setup.steps_done,
        final_dist=final_dist,
        fell=setup.fell,
        ms_per_step=ms_per_step,
        video_path=out_vid,
        avoid_bias_active_frac=(setup._avoid_cycles_active / setup._avoid_cycles_total
                                 if setup._avoid_cycles_total > 0 else 0.0),
    )
