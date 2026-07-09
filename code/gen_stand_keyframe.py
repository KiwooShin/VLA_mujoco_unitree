"""
gen_stand_keyframe.py — Run WBC settle ONCE offline and save the resulting
stable standing state as checkpoint/stand_keyframe.npz.

The keyframe captures qpos (minus xy translation, plus relative height) and qvel
after SETTLE_STEPS=80 steps of WBC with zero velocity command. At deploy, the
inferencer can restore this keyframe directly instead of running 80 ONNX steps.

Usage:
    MUJOCO_GL=egl python code/gen_stand_keyframe.py

Output:
    checkpoint/stand_keyframe.npz with keys:
        qpos_local   : (nq,) qpos with x=0, y=0, z=keyframe_z, yaw=0 — robot-local frame
        qvel_local   : (nv,) qvel — all near-zero after settle
        target_dof   : (15,) last WBC joint targets (used as prev_action seed)
        height       : scalar, pelvis z at settle end
        settle_steps : 80 (informational)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import mujoco

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from code.teacher import WBCTeacher, DEFAULT_ANGLES

SETTLE_STEPS = 80   # match inferencer.py constant

def gen_keyframe(out_path: str = "checkpoint/stand_keyframe.npz"):
    out_path = Path(_REPO) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[gen_keyframe] Running WBC settle for {SETTLE_STEPS} steps (zero vel)...")
    teacher = WBCTeacher(use_gpu=False)
    teacher.reset(pos_xy=(0.0, 0.0), yaw=0.0)

    for i in range(SETTLE_STEPS):
        teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        h = teacher.base_height
        if h < 0.50:
            print(f"[gen_keyframe] FELL during settle at step {i}! h={h:.3f}")
            sys.exit(1)
        if i % 20 == 0:
            print(f"  step={i:3d}  height={h:.4f}m")

    h_final = teacher.base_height
    print(f"[gen_keyframe] Settle complete. Final height={h_final:.4f}m")

    # Save robot-local keyframe (strip xy translation, keep relative z and yaw=0)
    qpos = teacher.data.qpos.copy()
    qvel = teacher.data.qvel.copy()

    # Normalize: zero out xy (will be added back per-episode), keep z and orientation
    qpos_local = qpos.copy()
    qpos_local[0] = 0.0   # x
    qpos_local[1] = 0.0   # y
    # yaw is already 0 (we settled from yaw=0)

    # qvel should be near-zero after settle
    print(f"  qvel max |component|: {np.abs(qvel).max():.4f}")

    target_dof = teacher._target_dof.copy()

    np.savez(
        out_path,
        qpos_local=qpos_local,
        qvel_local=qvel,
        target_dof=target_dof,
        height=np.float32(h_final),
        settle_steps=np.int32(SETTLE_STEPS),
    )
    print(f"[gen_keyframe] Saved: {out_path}")
    print(f"  qpos_local shape: {qpos_local.shape}")
    print(f"  qvel_local shape: {qvel.shape}")
    print(f"  target_dof:   {np.array2string(target_dof, precision=4)}")
    print(f"  height:       {h_final:.4f}")
    return str(out_path)


if __name__ == "__main__":
    gen_keyframe()
