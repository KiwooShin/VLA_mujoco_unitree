"""VF-1 item 5: title-card + outro-stats-card frames for the fancy demo
(code/fancy_demo.py, RF-1 split). Static frames appended before/after the
simulation loop — never interleaved with control.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from code.apps.fancy.constants import (
    BEV_H, BEV_W, EGO_H, EGO_W, FEAT_HIRES, FEAT_HUD, HUD_BAR_H,
    PANEL_DISPLAY_H, PANEL_DISPLAY_W, SKILL_STAGES,
)


def _final_canvas_dims() -> Tuple[int, int]:
    """Mirrors compose_sbs_frame's own size arithmetic WITHOUT rendering anything,
    so the title/outro card frames (built before/after the simulation loop, with
    no ego/bev frame at hand) match the exact (H, W) of the per-step SBS frames
    -- required since every frame appended to one video must share one shape.

    Returns:
        (height, width) in pixels, matching compose_sbs_frame()'s output shape
        for the current FEAT_HIRES / FEAT_HUD toggle state.
    """
    if FEAT_HIRES:
        w = PANEL_DISPLAY_W * 2 + 3
        h = PANEL_DISPLAY_H
    else:
        # ORIGINAL sizing: ego (always resized to EGO_W x EGO_H by
        # _label_active_cam) is rescaled to BEV_H tall in compose_sbs_frame,
        # i.e. width = EGO_W * (BEV_H / EGO_H); BEV stays native BEV_W x BEV_H.
        w = int(EGO_W * (BEV_H / EGO_H)) + 3 + BEV_W
        h = BEV_H
    if FEAT_HUD:
        h += HUD_BAR_H
    return h, w


def make_title_card(instruction: str, scenario_title: str, frame_idx: int, n_frames: int) -> np.ndarray:
    """VF-1 item 5: ~1.5s pre-roll title card -- scenario name (large) + the
    typed instruction, with a short fade-in over the first ~10 frames. Static
    content generated BEFORE the simulation loop starts (see its call site in
    run_fancy_rollout) -- purely additive frames, never interleaved with control.

    Args:
        instruction: Typed instruction text (or the combined multi-goal
            instruction) shown under the scenario title.
        scenario_title: Large scenario name shown at the top of the card.
        frame_idx: Zero-based index of this frame within the title-card
            sequence, used to compute the fade-in.
        n_frames: Total number of frames in the title-card sequence (fade-in
            completes at frame_idx >= 10, well before n_frames typically).

    Returns:
        (H, W, 3) uint8 BGR title-card frame, matching _final_canvas_dims().
    """
    import cv2
    h, w = _final_canvas_dims()
    img = np.full((h, w, 3), (24, 18, 14), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (60, 50, 40), 2)

    fade = min(1.0, frame_idx / 10.0)

    def _fade(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
        """Scale a BGR color tuple by the current fade-in level."""
        return tuple(int(c * fade) for c in bgr)

    title = scenario_title
    font_scale_title = min(1.8, w / 700.0)
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, 3)
    cv2.putText(img, title, ((w - tw) // 2, h // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_title, _fade((230, 230, 235)), 3, cv2.LINE_AA)

    instr = f'"{instruction}"'
    font_scale_instr = min(1.1, w / 900.0)
    (iw, ih), _ = cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, font_scale_instr, 2)
    cv2.putText(img, instr, ((w - iw) // 2, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_instr, _fade((120, 220, 255)), 2, cv2.LINE_AA)

    sub = "G1 HUMANOID  -  AUTONOMOUS VISUAL SEARCH & RETRIEVAL"
    (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(img, sub, ((w - sw) // 2, h // 2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                _fade((150, 150, 160)), 1, cv2.LINE_AA)

    pipeline = "   ".join(SKILL_STAGES)
    (pw, ph), _ = cv2.getTextSize(pipeline, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, pipeline, ((w - pw) // 2, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                _fade((90, 200, 140)), 1, cv2.LINE_AA)

    return img


def make_outro_card(last_frame: np.ndarray, sim_time_s: float, dist_traveled_m: float,
                    final_dist_m: float, steps: int) -> np.ndarray:
    """VF-1 item 5: ~2s freeze-frame on REACHED with a stats card overlay (elapsed
    sim time, distance traveled, final distance to target, step count). Built
    from the ACTUAL last rendered SBS frame (scene/robot/target still visible)
    plus a semi-transparent stats panel -- never re-renders anything.

    Args:
        last_frame: The final rendered SBS frame of the episode (copied, not
            mutated), used as the freeze-frame background.
        sim_time_s: Elapsed simulated time in seconds.
        dist_traveled_m: Total odometry distance traveled in meters.
        final_dist_m: Final distance to target in meters.
        steps: Total number of control steps taken.

    Returns:
        (H, W, 3) uint8 BGR outro-card frame, same shape as `last_frame`.
    """
    import cv2
    img = last_frame.copy()
    h, w = img.shape[:2]

    panel_w = min(440, w - 40)
    panel_h = 190
    px0, py0 = (w - panel_w) // 2, (h - panel_h) // 2
    overlay = img.copy()
    cv2.rectangle(overlay, (px0, py0), (px0 + panel_w, py0 + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)
    cv2.rectangle(img, (px0, py0), (px0 + panel_w, py0 + panel_h), (90, 220, 140), 2)

    cv2.putText(img, "REACHED", (px0 + 22, py0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                (90, 220, 140), 2, cv2.LINE_AA)
    lines = [
        f"time:       {sim_time_s:5.1f} s",
        f"traveled:   {dist_traveled_m:5.2f} m",
        f"final dist: {final_dist_m:5.3f} m",
        f"steps:      {steps}",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (px0 + 22, py0 + 74 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return img
