"""VF-1 item 3: bottom HUD bar for the fancy demo's composited SBS frame
(code/fancy_demo.py, RF-1 split).
"""

from __future__ import annotations

import numpy as np

from code.apps.fancy.constants import HUD_BAR_H, SKILL_STAGES


def draw_hud_bar(width: int, ctx: dict) -> np.ndarray:
    """VF-1 item 3: bottom HUD strip spanning the full canvas width --
      - typed instruction, verbatim (left)
      - live distance + bearing, step counter, walk speed (right)
      - 5-stage skill breadcrumb SCAN > LOCK > WALK > HANDOFF > REACH, active
        stage highlighted (center)
      - camera-in-use chip (HEAD/PROXIMITY), flashes for a few frames right
        after a handoff

    Pure render-side function: every field in `ctx` is a read of state that
    already exists in run_fancy_rollout (see its call site in _render_sbs_frame).

    Args:
        width: Full canvas width in pixels (the HUD strip spans this width).
        ctx: Context dict with keys `prompt`, `stage_idx`, `dist`,
            `bearing_deg`, `step`, `walk_speed_mps`, `active_cam`,
            `cam_flash` (see _render_sbs_frame's hud_ctx construction).

    Returns:
        (HUD_BAR_H, width, 3) uint8 BGR HUD strip.
    """
    import cv2
    h = HUD_BAR_H
    img = np.full((h, width, 3), (18, 18, 24), dtype=np.uint8)
    cv2.line(img, (0, 0), (width, 0), (70, 70, 90), 1, cv2.LINE_AA)

    # --- Left: typed instruction, verbatim (truncated only if it can't fit) ---
    prompt = ctx.get('prompt') or ''
    max_chars = max(10, width // 12)
    prompt_disp = prompt if len(prompt) <= max_chars else prompt[:max_chars - 3] + "..."
    cv2.putText(img, f'"{prompt_disp}"', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 235), 1, cv2.LINE_AA)

    # --- Center: skill breadcrumb ---
    stage_idx = ctx.get('stage_idx', -1)
    # Pre-measure total width so the breadcrumb is truly centered.
    seg_font, sep = 0.44, "  >  "
    widths = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, seg_font, 2)[0][0] for s in SKILL_STAGES]
    sep_w  = cv2.getTextSize(sep, cv2.FONT_HERSHEY_SIMPLEX, seg_font, 1)[0][0]
    total_w = sum(widths) + sep_w * (len(SKILL_STAGES) - 1)
    bc_x = max(10, (width - total_w) // 2)
    bc_y = 28
    for i, lab in enumerate(SKILL_STAGES):
        active = (i == stage_idx)
        done = (i < stage_idx)
        color = (255, 255, 255) if active else ((100, 220, 130) if done else (95, 95, 105))
        thick = 2 if active else 1
        if active:
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, seg_font, thick)
            cv2.rectangle(img, (bc_x - 6, bc_y - th - 6), (bc_x + tw + 6, bc_y + 6), (150, 90, 20), -1)
        cv2.putText(img, lab, (bc_x, bc_y), cv2.FONT_HERSHEY_SIMPLEX, seg_font, color, thick, cv2.LINE_AA)
        bc_x += widths[i]
        if i < len(SKILL_STAGES) - 1:
            cv2.putText(img, sep, (bc_x, bc_y), cv2.FONT_HERSHEY_SIMPLEX, seg_font, (90, 90, 100), 1, cv2.LINE_AA)
            bc_x += sep_w

    # --- Right: distance / bearing / step / speed + camera chip ---
    dist        = ctx.get('dist')
    bearing_deg = ctx.get('bearing_deg')
    step        = ctx.get('step', 0)
    speed       = ctx.get('walk_speed_mps', 0.0)
    parts = []
    if dist is not None:
        parts.append(f"dist={dist:.2f}m")
    if bearing_deg is not None:
        parts.append(f"brg={bearing_deg:+.0f}deg")
    parts.append(f"step={step}")
    parts.append(f"v={speed:.2f}m/s")
    txt = "   ".join(parts)

    chip_w = 118
    (txt_w, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    chip_x0 = width - 12 - chip_w
    cv2.putText(img, txt, (chip_x0 - txt_w - 18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (150, 220, 255), 1, cv2.LINE_AA)

    active_cam = ctx.get('active_cam', 'GROUNDING')
    cam_flash  = bool(ctx.get('cam_flash'))
    cam_label  = "PROXIMITY" if active_cam == 'PROXIMITY' else "HEAD"
    cam_color  = (60, 210, 255) if active_cam == 'PROXIMITY' else (255, 200, 150)
    chip_bg    = (0, 230, 255) if cam_flash else (48, 48, 58)
    cv2.rectangle(img, (chip_x0, 8), (chip_x0 + chip_w, h - 8), chip_bg, -1)
    cv2.rectangle(img, (chip_x0, 8), (chip_x0 + chip_w, h - 8), (90, 90, 100), 1)
    cv2.putText(img, f"CAM: {cam_label}", (chip_x0 + 8, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 0, 0) if cam_flash else cam_color, 1, cv2.LINE_AA)

    return img
