"""MP4 writer + showcase-reel concatenation for the fancy demo (code/
fancy_demo.py, RF-1 split).

`_concat_reel` is imported directly by code/render_showcase_reel.py via the
old `code.fancy_demo` path (kept working by that module's compat re-export).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


def _write_fancy_video(frames: list[np.ndarray], path: str, fps: int = 25) -> str:
    """Write ego|BEV SBS frames to MP4. Returns path."""
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not frames:
        return path
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"  [fancy] Video saved: {path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)", flush=True)
    return path


def _concat_reel(video_paths: list[str], reel_path: str) -> Optional[str]:
    """Concatenate multiple MP4s into a showcase reel."""
    import cv2
    valid = [p for p in video_paths if p and os.path.isfile(p)]
    if not valid:
        return None

    # Read first frame for size
    cap0 = cv2.VideoCapture(valid[0])
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap0.get(cv2.CAP_PROP_FPS)) or 25
    cap0.release()

    os.makedirs(os.path.dirname(os.path.abspath(reel_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(reel_path, fourcc, fps, (w, h))

    for vp in valid:
        cap = cv2.VideoCapture(vp)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            out.write(frame)
        cap.release()

    out.release()
    print(f"  [fancy] Reel saved: {reel_path}  ({len(valid)} episodes)", flush=True)
    return reel_path
