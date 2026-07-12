"""Ego-panel overlay + SBS frame composition for the fancy demo (code/
fancy_demo.py, RF-1 split): NX-6 detector confidence heatmap blend + the
final ego|BEV[+HUD] side-by-side compositing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from code.apps.fancy.constants import (
    BEV_H, FEAT_HIRES, FEAT_HUD, FEAT_HEATMAP, HEATMAP_ALPHA,
    PANEL_DISPLAY_W, PANEL_DISPLAY_H, STATE_IDLE,
)
from code.apps.fancy.hud import draw_hud_bar
from code.apps.fancy.overlays_bev import _STATE_COLOR_MAP


def draw_detector_heatmap_overlay(
    ego_bgr: np.ndarray,
    heatmap_cache: Optional[dict],
    target_color: str,
    target_shape: str,
    alpha: float = HEATMAP_ALPHA,
) -> tuple[np.ndarray, Optional[float]]:
    """VF-1 item 1: blend the NX-6 GROUND_NET detector's OWN confidence heatmap
    (cached by code/grounding.py's _ground_net() the same cycle it already ran
    the forward pass for detection -- ZERO extra inference here) onto the ego
    panel as a semi-transparent color map.

    Render-side only: reads a cache, writes only to the returned image copy.
    No-op (returns (ego_bgr, None) unchanged) when GROUND_NET was never
    invoked, the cache is empty/stale (wrong color+shape query -- i.e. the
    cache belongs to a different target than the one THIS episode is
    pursuing), or the cached cycle did not accept a detection.

    Args:
        ego_bgr: (H, W, 3) uint8 BGR ego panel frame to blend onto.
        heatmap_cache: get_ground_net_last_heatmap()'s cache dict (`prob`,
            `color`, `shape`, `accepted`, `confidence`), or None if GROUND_NET
            was never invoked.
        target_color: This episode's target color name, used to check the
            cache matches the currently-pursued target.
        target_shape: This episode's target shape name, used to check the
            cache matches the currently-pursued target.
        alpha: Maximum blend strength (per-pixel alpha is confidence-scaled,
            capped at this value).

    Returns:
        Tuple of (blended_bgr, confidence): `blended_bgr` is a new frame with
        the heatmap blended in, or `ego_bgr` unchanged when there is nothing
        to draw; `confidence` is the cached detection confidence, or None
        when nothing was drawn.
    """
    import cv2
    if heatmap_cache is None or heatmap_cache.get('prob') is None:
        return ego_bgr, None
    if (heatmap_cache.get('color') != target_color.lower().strip() or
            heatmap_cache.get('shape') != target_shape.lower().strip()):
        return ego_bgr, None
    if not heatmap_cache.get('accepted', False):
        return ego_bgr, None

    prob = heatmap_cache['prob']
    h, w = ego_bgr.shape[:2]
    prob_r = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    prob_u8 = np.clip(prob_r * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)

    # Per-pixel alpha proportional to confidence (capped at `alpha`) -- a smooth
    # glow that fades out with confidence rather than a hard-edged patch, and
    # is a provable no-op (alpha~0) wherever the map says "nothing here" (gate
    # check: heatmap must not obscure the scene). A small blur spreads the
    # (typically tight, few-pixel) detector peak into a visible glow radius --
    # purely a display nicety, the underlying confidence values are unchanged.
    alpha_map = np.clip(prob_r, 0.0, 1.0) * alpha
    blur_px = max(3, int(round(0.02 * w))) | 1   # odd kernel size, ~2% of panel width
    alpha_map = cv2.GaussianBlur(alpha_map, (blur_px, blur_px), 0)
    alpha_map = alpha_map[..., None]
    out = (heat_bgr.astype(np.float32) * alpha_map +
           ego_bgr.astype(np.float32) * (1.0 - alpha_map)).astype(np.uint8)
    return out, float(heatmap_cache.get('confidence', 0.0))


def compose_sbs_frame(
    ego_rgb: np.ndarray,   # (EGO_H, EGO_W, 3) uint8 RGB — CAM-2 ACTIVE camera feed
    bev_img: np.ndarray,   # (BEV_H, BEV_W, 3) uint8 BGR
    state: str = STATE_IDLE,
    prompt: str = "",
    dist_to_target: Optional[float] = None,
    goal_idx: int = 0,
    n_goals: int = 1,
    active_cam: str = "GROUNDING",   # CAM-2 (docs/cam_p1.md): 'GROUNDING' (head, far) | 'PROXIMITY' (near)
    # VF-1 item 1: detector heatmap cache + the query it should match.
    heatmap_cache: Optional[dict] = None,
    target_color: str = "",
    target_shape: str = "",
    # VF-1 item 3: HUD bar context dict (see draw_hud_bar) — None disables it
    # regardless of FEAT_HUD.
    hud_ctx: Optional[dict] = None,
) -> np.ndarray:
    """Compose side-by-side frame: ego (left, CAM-2 active-camera feed) | BEV (right)
    [+ VF-1 bottom HUD strip when FEAT_HUD and hud_ctx is given].

    When every VF-1 toggle is off (FANCY_PLAIN=1) this reproduces the pre-VF1
    frame byte-for-byte (same resize target, same badge layout, same divider).

    Args:
        ego_rgb: (EGO_H, EGO_W, 3) uint8 RGB CAM-2 active-camera feed.
        bev_img: (BEV_H, BEV_W, 3) uint8 BGR BEV frame (with its own overlays
            already drawn by draw_bev_overlays()).
        state: Current state-machine state (one of the STATE_* constants).
        prompt: Typed instruction text (unused directly here; forwarded to
            draw_bev_overlays() by the caller).
        dist_to_target: Current distance to target in meters, or None.
        goal_idx: Zero-based index of the current sub-goal (multi-goal runs).
        n_goals: Total number of sub-goals in this episode.
        active_cam: CAM-2 (docs/cam_p1.md) active camera name -- 'GROUNDING'
            (head, far) or 'PROXIMITY' (near).
        heatmap_cache: get_ground_net_last_heatmap()'s cache dict, forwarded
            to draw_detector_heatmap_overlay(); None disables the overlay.
        target_color: This episode's target color name, forwarded to
            draw_detector_heatmap_overlay().
        target_shape: This episode's target shape name, forwarded to
            draw_detector_heatmap_overlay().
        hud_ctx: HUD bar context dict (see draw_hud_bar()); None disables the
            HUD bar regardless of FEAT_HUD.

    Returns:
        (H, W, 3) uint8 BGR composited frame.
    """
    import cv2

    # Convert ego from RGB to BGR
    ego_bgr = cv2.cvtColor(ego_rgb, cv2.COLOR_RGB2BGR)

    # VF-1 item 6: display both panels at PANEL_DISPLAY_W x PANEL_DISPLAY_H
    # (upscaled from the UNCHANGED native render sizes via cv2.resize — no
    # extra MuJoCo render cost). Falls back to the exact original "scale ego
    # to BEV_H, keep BEV native" behavior when FEAT_HIRES is off.
    if FEAT_HIRES:
        target_h = PANEL_DISPLAY_H
        if (ego_bgr.shape[1], ego_bgr.shape[0]) != (PANEL_DISPLAY_W, PANEL_DISPLAY_H):
            ego_bgr = cv2.resize(ego_bgr, (PANEL_DISPLAY_W, PANEL_DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        if (bev_img.shape[1], bev_img.shape[0]) != (PANEL_DISPLAY_W, PANEL_DISPLAY_H):
            bev_img = cv2.resize(bev_img, (PANEL_DISPLAY_W, PANEL_DISPLAY_H), interpolation=cv2.INTER_LINEAR)
    else:
        target_h = BEV_H
        if ego_bgr.shape[0] != target_h:
            scale = target_h / ego_bgr.shape[0]
            ego_bgr = cv2.resize(ego_bgr, (int(ego_bgr.shape[1] * scale), target_h))

    # VF-1 item 1: blend the detector heatmap AFTER the resize (so both the
    # color blend and the text tag below are at final display resolution).
    heatmap_conf = None
    if FEAT_HEATMAP:
        ego_bgr, heatmap_conf = draw_detector_heatmap_overlay(ego_bgr, heatmap_cache,
                                                              target_color, target_shape)

    # Ego overlay: state badge + active-camera label. VF-1: larger badge/font
    # when FEAT_HIRES (kept at the original small size otherwise).
    badge_h = 70 if FEAT_HIRES else 36
    cv2.rectangle(ego_bgr, (0, 0), (ego_bgr.shape[1], badge_h), (20, 20, 20), -1)
    sc = _STATE_COLOR_MAP.get(state, (200, 200, 200))
    if FEAT_HIRES:
        cv2.rectangle(ego_bgr, (10, 8), (240, 58), sc, -1)
        cv2.putText(ego_bgr, state, (20, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 3, cv2.LINE_AA)
    else:
        cv2.rectangle(ego_bgr, (4, 4), (90, 30), sc, -1)
        cv2.putText(ego_bgr, state[:10], (7, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    # CAM-2 handoff label: "HEAD CAM" (GROUNDING, far) / "PROXIMITY CAM" (near) —
    # makes the camera handoff visible to viewers, distinct from the small
    # "CAM: GROUNDING|PROXIMITY d=X.XXm" overlay already baked into ego_rgb by
    # _label_active_cam() in the main rollout loop.
    cam_label = "PROXIMITY CAM" if active_cam == "PROXIMITY" else "HEAD CAM"
    cam_color = (60, 210, 255) if active_cam == "PROXIMITY" else (255, 200, 150)
    cam_font  = 0.9 if FEAT_HIRES else 0.4
    cam_thick = 2 if FEAT_HIRES else 1
    (tw, th_), _ = cv2.getTextSize(cam_label, cv2.FONT_HERSHEY_SIMPLEX, cam_font, cam_thick)
    cam_tx = ego_bgr.shape[1] - tw - (16 if FEAT_HIRES else 8)
    cam_ty = 44 if FEAT_HIRES else 23
    # VF-1 item 3: flash the camera chip's background for a few frames right
    # after a GROUNDING<->PROXIMITY handoff (hud_ctx['cam_flash'], a pure
    # render-side counter maintained by the caller — never read by control).
    if hud_ctx is not None and hud_ctx.get('cam_flash'):
        pad = 6
        cv2.rectangle(ego_bgr, (cam_tx - pad, cam_ty - th_ - pad),
                      (cam_tx + tw + pad, cam_ty + pad), (0, 255, 255), -1)
        cv2.putText(ego_bgr, cam_label, (cam_tx, cam_ty),
                    cv2.FONT_HERSHEY_SIMPLEX, cam_font, (0, 0, 0), cam_thick, cv2.LINE_AA)
    else:
        cv2.putText(ego_bgr, cam_label, (cam_tx, cam_ty),
                    cv2.FONT_HERSHEY_SIMPLEX, cam_font, cam_color, cam_thick, cv2.LINE_AA)

    # VF-1 item 1: "NEURAL DETECTOR" tag + live confidence, bottom-left of the
    # ego panel (drawn at final display resolution for a crisp font).
    if FEAT_HEATMAP and heatmap_conf is not None:
        tag = f"NEURAL DETECTOR  conf={heatmap_conf:.2f}"
        tag_font = 0.62 if FEAT_HIRES else 0.38
        ty = ego_bgr.shape[0] - (14 if FEAT_HIRES else 8)
        cv2.putText(ego_bgr, tag, (10, ty), cv2.FONT_HERSHEY_SIMPLEX, tag_font,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(ego_bgr, tag, (10, ty), cv2.FONT_HERSHEY_SIMPLEX, tag_font,
                    (60, 255, 210), 1, cv2.LINE_AA)

    # Divider line
    divider = np.full((ego_bgr.shape[0], 3, 3), 60, dtype=np.uint8)

    sbs = np.concatenate([ego_bgr, divider, bev_img], axis=1)

    # VF-1 item 3: bottom HUD bar (separate strip, full canvas width).
    if FEAT_HUD and hud_ctx is not None:
        hud_strip = draw_hud_bar(sbs.shape[1], hud_ctx)
        sbs = np.concatenate([sbs, hud_strip], axis=0)

    return sbs
