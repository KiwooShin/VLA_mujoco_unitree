"""
nx6_heatmap_model.py — NX-6 TRAIN (heatmap variant): query-conditioned detection head.

Architecture (per docs/nx6_train_heatmap.md): a small from-scratch U-Net/FCN.
  in:  RGBD (4ch) at (TARGET_H, TARGET_W)
  cond: one-hot(class, 4) + one-hot(color, 7) -> MLP -> broadcast-concat at bottleneck
  out: 2ch map at (TARGET_H, TARGET_W) = [target-presence heatmap logit, distance
       residual (metres)]

Decode: argmax pixel of sigmoid(heatmap) (refined by a local center-of-mass window)
-> back-project via code.arena.backproject_pixel + code.grounding.cam_to_egocentric
(the SAME geometry pipeline dataset/det_v1's own labels were validated against,
docs/nx6_data.md §5) using the query's nominal object radius -> (dist_bp, bearing_deg).
Final distance = dist_bp + predicted residual sampled at the (refined) peak pixel.

No pretrained backbones anywhere in this file (project constraint) — every parameter
here is trained from scratch on dataset/det_v1.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("MUJOCO_GL", "egl")

from code.arena import COLORS, SHAPES, GROUNDING_PITCH, PROXIMITY_PITCH, backproject_pixel
from code.grounding import get_ego_intrinsics_rendered, cam_to_egocentric

# ---------------------------------------------------------------------------
# Constants (must match dataset/det_v1's class_id / color_id ordering, which is
# just the index into code.arena.SHAPES / code.arena.COLORS — verified against
# dataset/det_v1/train/labels.parquet: color_id=6 <-> 'cyan', class_id=1 <-> 'cube').
# ---------------------------------------------------------------------------
CLASS_NAMES = [s[0] for s in SHAPES]          # ['ball','cube','cylinder','cone']
COLOR_NAMES = [c[0] for c in COLORS]          # ['red','yellow','blue','green','orange','purple','cyan']
N_CLASS = len(CLASS_NAMES)
N_COLOR = len(COLOR_NAMES)
SIZE_M = dict(SHAPES)                          # nominal diameter (m) per shape, e.g. ball:0.24

# Both cameras are rendered at FOVY=45deg (code/grounding.py:get_ego_intrinsics_rendered);
# both native resolutions (480x360 grounding, 320x240 proximity) are already 4:3, so a
# uniform-scale resize to TARGET_W x TARGET_H (also 4:3) preserves the pinhole model
# exactly -- intrinsics for the resized canvas are just get_ego_intrinsics_rendered(TARGET_W,
# TARGET_H), independent of which camera the frame came from.
TARGET_W, TARGET_H = 192, 144
TARGET_INTR = get_ego_intrinsics_rendered(TARGET_W, TARGET_H)

PITCH_BY_CAM = {"grounding": GROUNDING_PITCH, "proximity": PROXIMITY_PITCH}

MAX_DEPTH_CLIP_M = 12.0   # depth-input normalization clip (matches grounding.py's MAX_DEPTH_M)


def encode_query(class_id: int, color_id: int) -> np.ndarray:
    """(class_id, color_id) -> 11-d one-hot float32 vector [4 class | 7 color]."""
    v = np.zeros(N_CLASS + N_COLOR, dtype=np.float32)
    v[class_id] = 1.0
    v[N_CLASS + color_id] = 1.0
    return v


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _conv_bn_relu(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TinyHeatmapUNet(nn.Module):
    """Small from-scratch query-conditioned U-Net. Target: <5M params (actual ~0.9M)."""

    def __init__(self, in_ch=4, base=32, embed_dim=64, query_dim=N_CLASS + N_COLOR):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 3, base * 4  # 32,64,96,128

        self.stem  = _conv_bn_relu(in_ch, c1)                 # H, W
        self.down1 = _conv_bn_relu(c1, c2, stride=2)           # H/2
        self.down2 = _conv_bn_relu(c2, c3, stride=2)           # H/4
        self.down3 = _conv_bn_relu(c3, c4, stride=2)           # H/8

        self.query_mlp = nn.Sequential(
            nn.Linear(query_dim, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            _conv_bn_relu(c4 + embed_dim, c4),
            _conv_bn_relu(c4, c4),
        )

        self.up3 = _conv_bn_relu(c4 + c3, c3)   # after upsample to H/4, concat skip c3
        self.up2 = _conv_bn_relu(c3 + c2, c2)   # -> H/2, concat skip c2
        self.up1 = _conv_bn_relu(c2 + c1, c1)   # -> H, concat skip c1

        self.head = nn.Conv2d(c1, 2, kernel_size=1)  # [heatmap_logit, dist_residual_m]
        # Bias the heatmap head toward "no detection" at init (helps early-training
        # precision given negatives dominate pixel count).
        nn.init.constant_(self.head.bias[0], -3.0)
        nn.init.constant_(self.head.bias[1], 0.0)

    def forward(self, x, query):
        """x: (B,4,H,W) float32. query: (B, 11) one-hot float32."""
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        b  = self.down3(s3)

        q = self.query_mlp(query)                              # (B, embed_dim)
        q_map = q[:, :, None, None].expand(-1, -1, b.shape[2], b.shape[3])
        b = self.bottleneck(torch.cat([b, q_map], dim=1))

        u3 = F.interpolate(b, size=s3.shape[2:], mode="bilinear", align_corners=False)
        u3 = self.up3(torch.cat([u3, s3], dim=1))
        u2 = F.interpolate(u3, size=s2.shape[2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, s2], dim=1))
        u1 = F.interpolate(u2, size=s1.shape[2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, s1], dim=1))

        out = self.head(u1)
        return out[:, 0], out[:, 1]   # heatmap_logit (B,H,W), dist_residual (B,H,W)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Decode: prediction -> (present, dist_m, bearing_deg, confidence, peak_uv)
# ---------------------------------------------------------------------------
def _refine_peak(heat_np: np.ndarray, py: int, px: int, win: int = 2):
    """Sub-pixel refinement: intensity-weighted center-of-mass in a (2*win+1) window
    around the hard-argmax pixel. heat_np: (H,W) sigmoid probabilities in [0,1]."""
    H, W = heat_np.shape
    y0, y1 = max(0, py - win), min(H, py + win + 1)
    x0, x1 = max(0, px - win), min(W, px + win + 1)
    patch = heat_np[y0:y1, x0:x1].astype(np.float64)
    s = patch.sum()
    if s <= 1e-8:
        return float(px), float(py)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    cy = float((patch * ys).sum() / s)
    cx = float((patch * xs).sum() / s)
    return cx, cy


def decode_single(heat_logit: np.ndarray, dist_resid: np.ndarray, depth_m: np.ndarray,
                   class_id: int, cam_type: str, conf_thresh: float = 0.5,
                   refine_win: int = 2):
    """
    Decode one example's raw network output to a detection.

    Parameters
    ----------
    heat_logit : (H,W) float32 — raw logits (pre-sigmoid) at TARGET_H x TARGET_W.
    dist_resid : (H,W) float32 — predicted distance residual (metres).
    depth_m    : (H,W) float32 — the (possibly aug/dropout-corrupted at train time,
                 raw metric depth at inference) depth channel *as fed to the network*,
                 resized to TARGET_H x TARGET_W.
    class_id   : query class id (0..3) -- selects the nominal object radius correction.
    cam_type   : 'grounding' or 'proximity' -- selects the un-pitch angle.
    conf_thresh: sigmoid confidence threshold for "present".

    Returns
    -------
    dict(present: bool, confidence: float, dist_m: float, bearing_deg: float,
         peak_px: (x,y))
    """
    prob = 1.0 / (1.0 + np.exp(-heat_logit))
    py, px = np.unravel_index(np.argmax(prob), prob.shape)
    confidence = float(prob[py, px])
    present = confidence >= conf_thresh

    cx, cy = _refine_peak(prob, int(py), int(px), win=refine_win)
    # depth + residual sampled at the *hard* argmax pixel (matches training-time
    # supervision, which is defined at the GT integer pixel -- see nx6_heatmap_data.py).
    z = float(depth_m[py, px])
    radius = SIZE_M.get(CLASS_NAMES[class_id], 0.24) / 2.0
    x_cam, y_cam, z_cam = backproject_pixel(cx, cy, z, TARGET_INTR)
    pitch = PITCH_BY_CAM[cam_type]
    dist_bp, yaw_err_rad = cam_to_egocentric(x_cam, y_cam, z_cam + radius,
                                              pitch_deg=pitch, use_corrected_unpitch=True)
    resid = float(dist_resid[py, px])
    dist_m = float(dist_bp) + resid
    bearing_deg = math.degrees(yaw_err_rad)

    return dict(present=bool(present), confidence=confidence, dist_m=dist_m,
                bearing_deg=bearing_deg, peak_px=(float(cx), float(cy)))


# ---------------------------------------------------------------------------
# Standalone inference wrapper (documented API; see runs/nx6_heatmap_A/README.md)
# ---------------------------------------------------------------------------
class HeatmapDetector:
    """
    Standalone inference wrapper around TinyHeatmapUNet.

    Usage
    -----
        det = HeatmapDetector.load("runs/nx6_heatmap_A/model_best.pt", device="cuda")
        result = det.infer(rgb_uint8, depth_m_f32, class_name="cube", color_name="cyan",
                            cam_type="grounding")
        # result = dict(present, confidence, dist_m, bearing_deg, peak_px)

    `rgb_uint8`: (H,W,3) uint8, any resolution matching the source camera's native
    aspect (4:3) — grounding (480x360) or proximity (320x240) frames both work as-is,
    the wrapper resizes internally.
    `depth_m_f32`: (H,W) float32/float16 metric depth from the same camera pose.
    """

    def __init__(self, model: TinyHeatmapUNet, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device
        # VF-1 (docs/vf1_showpiece.md): render-side-only cache of the last
        # infer() call's confidence map, so a caller can display it (e.g.
        # fancy_demo.py's detector-heatmap overlay) with ZERO extra inference.
        # Pure numpy attribute, never read by any control-flow code path.
        self.last_heat_prob = None   # (TARGET_H, TARGET_W) float32 in [0,1] or None
        self.last_heat_meta = None   # dict(class_name, color_name, cam_type) or None

    @classmethod
    def load(cls, ckpt_path: str, device: str = "cpu"):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("model_cfg", {})
        model = TinyHeatmapUNet(**cfg)
        model.load_state_dict(ckpt["model_state"])
        return cls(model, device=device)

    @torch.no_grad()
    def infer(self, rgb_uint8: np.ndarray, depth_m: np.ndarray, class_name: str,
              color_name: str, cam_type: str, conf_thresh: float = 0.5):
        import cv2
        class_id = CLASS_NAMES.index(class_name)
        color_id = COLOR_NAMES.index(color_name)

        rgb_r = cv2.resize(rgb_uint8, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        depth_r = cv2.resize(np.asarray(depth_m, dtype=np.float32), (TARGET_W, TARGET_H),
                              interpolation=cv2.INTER_NEAREST)
        depth_in = np.clip(depth_r, 0.0, MAX_DEPTH_CLIP_M) / MAX_DEPTH_CLIP_M

        x = np.concatenate([rgb_r.astype(np.float32) / 255.0, depth_in[..., None]], axis=-1)
        x_t = torch.from_numpy(x.transpose(2, 0, 1)[None]).float().to(self.device)
        q_t = torch.from_numpy(encode_query(class_id, color_id)[None]).float().to(self.device)

        heat_logit, dist_resid = self.model(x_t, q_t)
        heat_logit = heat_logit[0].cpu().numpy()
        dist_resid = dist_resid[0].cpu().numpy()

        # VF-1: cache the sigmoid confidence map from THIS forward pass (pure
        # elementwise numpy op on the array already produced above -- no extra
        # inference, no torch/CUDA interaction, no effect on decode_single's own
        # independently-computed prob array below or on the returned detection).
        self.last_heat_prob = 1.0 / (1.0 + np.exp(-heat_logit))
        self.last_heat_meta = dict(class_name=class_name, color_name=color_name, cam_type=cam_type)

        return decode_single(heat_logit, dist_resid, depth_r, class_id, cam_type,
                              conf_thresh=conf_thresh)

    @torch.no_grad()
    def infer_batch_tensor(self, x_t: torch.Tensor, q_t: torch.Tensor):
        """Raw batched forward for eval/benchmark code. Returns (heat_logit, dist_resid)."""
        return self.model(x_t.to(self.device), q_t.to(self.device))


if __name__ == "__main__":
    m = TinyHeatmapUNet()
    print("params:", m.num_params(), f"({m.num_params()/1e6:.3f}M)")
    x = torch.randn(2, 4, TARGET_H, TARGET_W)
    q = torch.zeros(2, N_CLASS + N_COLOR)
    q[:, 1] = 1.0
    q[:, N_CLASS + 6] = 1.0
    h, d = m(x, q)
    print("heat", h.shape, "dist_resid", d.shape)
