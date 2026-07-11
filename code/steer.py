"""
steer.py — Privileged steering controller for G1Nav.

Given the target world pose and current robot base pose, computes an
egocentric (dist, yaw_err) and returns a velocity command (vx, vy, ωz).

Behaviour
---------
1. Turn-in-place first when |yaw_err| > FACE_THR_RAD (used in demo mode
   when the target can start outside the initial FOV).
2. Walk toward the target, decelerating smoothly when close.
3. Stop (zero command) when dist < stop_r.

API
---
steer(robot_xy, robot_yaw, target_xy, stop_r, *,
      max_vx=0.55, max_wz=0.8, decel_dist=0.9) -> (vel_cmd, dist, yaw_err)

vel_cmd  : np.ndarray shape (3,)  [vx, vy, ωz]
dist     : float  distance to target (m)
yaw_err  : float  signed bearing error (rad)
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# Controller parameters
# ---------------------------------------------------------------------------
MAX_VX: float       = 0.55   # max forward speed (m/s)
MAX_WZ: float       = 0.80   # max yaw rate (rad/s)
DECEL_DIST: float   = 0.90   # start decelerating when dist < this (m)
FACE_THR_RAD: float = math.radians(25.0)  # turn-in-place when |yaw_err| > this
YAW_KP: float       = 1.2    # proportional gain on yaw error → ωz
VX_YAW_DAMP: float  = 0.0    # lateral speed (unused for G1 WBC; G1 walks straight)


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference a − b, wrapped to (−π, π]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def egocentric_goal(
    robot_xy: np.ndarray | tuple[float, float],
    robot_yaw: float,
    target_xy: np.ndarray | tuple[float, float],
) -> tuple[float, float, float]:
    """Compute egocentric goal from world-frame poses.

    Args:
        robot_xy: Robot position (x, y) in world frame (m).
        robot_yaw: Robot yaw angle (rad).
        target_xy: Target position (x, y) in world frame (m).

    Returns:
        Tuple of (dist, yaw_err, bearing):
            dist: Euclidean distance (m).
            yaw_err: Signed bearing error in robot frame (rad), in (−π, π].
            bearing: World-frame bearing to target (rad).
    """
    dx = float(target_xy[0]) - float(robot_xy[0])
    dy = float(target_xy[1]) - float(robot_xy[1])
    dist    = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    yaw_err = _angle_diff(bearing, robot_yaw)
    return dist, yaw_err, bearing


def steer(
    robot_xy: np.ndarray | tuple[float, float],
    robot_yaw: float,
    target_xy: np.ndarray | tuple[float, float],
    stop_r: float,
    *,
    max_vx: float      = MAX_VX,
    max_wz: float      = MAX_WZ,
    decel_dist: float  = DECEL_DIST,
) -> tuple[np.ndarray, float, float]:
    """Compute velocity command given robot and target poses.

    Args:
        robot_xy: Robot position (x, y) in world frame (m).
        robot_yaw: Robot yaw angle (rad).
        target_xy: Target position (x, y) in world frame (m).
        stop_r: Success radius (m).
        max_vx: Maximum forward speed (m/s).
        max_wz: Maximum yaw rate (rad/s).
        decel_dist: Deceleration onset distance (m).

    Returns:
        Tuple of (vel_cmd, dist, yaw_err):
            vel_cmd: Velocity command [vx, vy=0, ωz], shape (3,).
            dist: Distance to target (m).
            yaw_err: Signed bearing error (rad).
    """
    dist, yaw_err, _ = egocentric_goal(robot_xy, robot_yaw, target_xy)

    if dist < stop_r:
        return np.zeros(3, dtype=np.float32), dist, yaw_err

    # ---- Yaw command (proportional, clamped) ----
    wz = float(np.clip(YAW_KP * yaw_err, -max_wz, max_wz))

    # ---- Forward speed ----
    # Reduce vx when heading is badly misaligned (turn first, then walk)
    yaw_align = max(0.0, math.cos(yaw_err))  # 1 when aligned, 0 when 90°

    if abs(yaw_err) > FACE_THR_RAD:
        # Turn-in-place mode: very little forward motion
        vx = 0.0
    else:
        # Walk forward, scaled by alignment + deceleration ramp
        decel_factor = min(1.0, max(0.0, (dist - stop_r) / max(decel_dist - stop_r, 0.1)))
        vx = float(np.clip(max_vx * yaw_align * decel_factor, 0.0, max_vx))

    vel_cmd = np.array([vx, 0.0, wz], dtype=np.float32)
    return vel_cmd, dist, yaw_err


# ---------------------------------------------------------------------------
# Goal vector helpers (for dataset logging)
# ---------------------------------------------------------------------------
def goal_vec(dist: float, yaw_err: float) -> np.ndarray:
    """Return normalised egocentric goal vector (dist, cos θ, sin θ).

    This is the privileged label used in the ADR schema.

    Args:
        dist: Distance to target (m).
        yaw_err: Signed bearing error (rad).

    Returns:
        Array [dist, cos(yaw_err), sin(yaw_err)] of shape (3,), dtype float32.
    """
    return np.array([dist, math.cos(yaw_err), math.sin(yaw_err)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Robot at origin, facing +X; target at (3, 1)
    robot_xy  = (0.0, 0.0)
    robot_yaw = 0.0
    target_xy = (3.0, 1.0)
    stop_r    = 0.6

    cmd, dist, yerr = steer(robot_xy, robot_yaw, target_xy, stop_r)
    print(f"dist={dist:.3f}  yaw_err={math.degrees(yerr):.1f}°  "
          f"vel_cmd=[{cmd[0]:.3f}, {cmd[1]:.3f}, {cmd[2]:.3f}]")
    assert dist > stop_r and cmd[0] >= 0.0, "steer sanity failed"

    # Target directly behind (should turn in place)
    cmd2, dist2, yerr2 = steer((0, 0), 0.0, (-3.0, 0.0), stop_r)
    print(f"[behind] dist={dist2:.3f}  yaw_err={math.degrees(yerr2):.1f}°  "
          f"vel_cmd=[{cmd2[0]:.3f}, {cmd2[1]:.3f}, {cmd2[2]:.3f}]")
    assert cmd2[0] == 0.0, "should turn in place when target is behind"

    # Within stop radius
    cmd3, dist3, _ = steer((0, 0), 0.0, (0.3, 0.0), stop_r)
    print(f"[at target] dist={dist3:.3f}  cmd={cmd3}")
    assert np.allclose(cmd3, 0), "should be zero at target"

    print("steer.py smoke PASS")
    sys.exit(0)
