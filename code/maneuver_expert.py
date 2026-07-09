"""
maneuver_expert.py — FSM-based scripted expert for the MANEUVER skill.

Task: "go straight, turn {left/right} after passing the {color}{shape}"

FSM States:
  STRAIGHT  : Walk forward toward/past the landmark.
              Trigger → TURN_PHASE when robot_x > landmark_x + pass_margin
              (robot has passed the landmark in the forward direction).

  TURN_PHASE : Turn in place toward target_heading.
              Uses yaw-rate command (wz).
              Trigger → STRAIGHT2 when |heading_err| < HEADING_DONE_THR.

  STRAIGHT2 : Walk straight in the new heading direction.
              Continues until max steps or episode end.

Velocity commands (match steer.py max values):
  STRAIGHT:   vx=0.50, wz=small_correction_to_stay_aligned_with_landmark_y
  TURN_PHASE: vx=0.0,  wz=±0.80 (toward target_heading)
  STRAIGHT2:  vx=0.50, wz=small_correction_to_maintain_heading

Privileged state (extra obs stored per step):
  - subgoal_index:  int  0=STRAIGHT, 1=TURN_PHASE, 2=STRAIGHT2
  - target_heading: float (rad)
  - heading_err:    float (rad, signed: positive = need to turn CCW)
  - cos(target_heading), sin(target_heading)
  - landmark_passed: bool

NOTE: The expert drives the WBCTeacher directly via vel_cmd, NOT joint-level control.
The teacher handles low-level locomotion; the expert provides high-level velocity commands.
"""

import math
from enum import IntEnum

import numpy as np

# ---------------------------------------------------------------------------
# FSM States
# ---------------------------------------------------------------------------
class State(IntEnum):
    STRAIGHT  = 0
    TURN_PHASE = 1
    STRAIGHT2 = 2

# Controller parameters
FORWARD_VX    = 0.50    # forward walking speed (m/s)
MAX_WZ        = 0.80    # max yaw rate (rad/s)
TURN_KP       = 1.2     # yaw proportional gain
Y_KP          = 0.5     # lateral correction gain (keep landmark in corridor)
HEADING_DONE_THR = math.radians(15.0)  # heading error threshold to exit TURN_PHASE
DECEL_DIST    = 0.8     # not used but kept for interface compatibility

# Lateral corridor: keep robot within this Y-offset of landmark_y while in STRAIGHT
LATERAL_THR   = 0.3    # meters off-center before applying lateral correction yaw


def _angle_diff(a: float, b: float) -> float:
    """Signed a - b in (-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


class ManeuverExpert:
    """
    FSM expert for the maneuver task.

    Usage:
        expert = ManeuverExpert(scene_cfg)
        expert.reset()
        vel_cmd, priv_state = expert.step(robot_xy, robot_yaw)

    priv_state dict contains:
        subgoal_index: int
        target_heading: float (rad)
        heading_err: float (rad)
        cos_target: float
        sin_target: float
        landmark_passed: bool
        fsm_state: int
    """

    def __init__(self, scene_cfg: dict):
        self._scene = scene_cfg
        lm_xy = scene_cfg["landmark_xy"]
        self._landmark_x = float(lm_xy[0])
        self._landmark_y = float(lm_xy[1])
        self._pass_margin = float(scene_cfg.get("pass_margin", 0.6))
        self._target_heading = float(scene_cfg["target_heading"])
        self._turn_direction = scene_cfg["turn_direction"]

        self._state = State.STRAIGHT
        self._pass_triggered = False

    def reset(self):
        self._state = State.STRAIGHT
        self._pass_triggered = False

    def step(self, robot_xy, robot_yaw: float) -> tuple:
        """
        Compute velocity command and privileged state for the current FSM state.

        Parameters
        ----------
        robot_xy : (x, y) robot position
        robot_yaw : float  robot heading (rad)

        Returns
        -------
        vel_cmd : np.ndarray (3,)  [vx, vy, wz]
        priv    : dict with privileged state
        """
        rx, ry = float(robot_xy[0]), float(robot_xy[1])

        # Check landmark pass trigger
        if not self._pass_triggered:
            if rx >= self._landmark_x + self._pass_margin:
                self._pass_triggered = True
                self._state = State.TURN_PHASE

        # Heading error toward target
        heading_err = _angle_diff(self._target_heading, robot_yaw)

        # Check turn completion
        if self._state == State.TURN_PHASE:
            if abs(heading_err) < HEADING_DONE_THR:
                self._state = State.STRAIGHT2

        # Compute velocity command based on state
        if self._state == State.STRAIGHT:
            vel_cmd = self._cmd_straight(rx, ry, robot_yaw)
        elif self._state == State.TURN_PHASE:
            vel_cmd = self._cmd_turn(heading_err)
        else:  # STRAIGHT2
            vel_cmd = self._cmd_straight2(robot_yaw)

        priv = {
            "subgoal_index":  int(self._state),
            "target_heading": self._target_heading,
            "heading_err":    float(heading_err),
            "cos_target":     math.cos(self._target_heading),
            "sin_target":     math.sin(self._target_heading),
            "landmark_passed": self._pass_triggered,
            "fsm_state":      int(self._state),
        }

        return vel_cmd, priv

    def _cmd_straight(self, rx: float, ry: float, robot_yaw: float) -> np.ndarray:
        """Walk forward, with small lateral yaw correction to stay near landmark_y."""
        # Lateral correction: steer slightly to keep robot aligned with landmark_y
        y_err = self._landmark_y - ry
        # Convert lateral error to yaw correction (robot faces +X so lateral is Y)
        wz_corr = float(np.clip(Y_KP * y_err * math.cos(robot_yaw), -0.3, 0.3))
        # Add heading correction back to 0 (forward facing)
        heading_err_to_forward = _angle_diff(0.0, robot_yaw)
        wz_heading = float(np.clip(TURN_KP * heading_err_to_forward, -0.4, 0.4))
        wz = float(np.clip(wz_corr + wz_heading, -MAX_WZ, MAX_WZ))
        return np.array([FORWARD_VX, 0.0, wz], dtype=np.float32)

    def _cmd_turn(self, heading_err: float) -> np.ndarray:
        """Turn in place toward target_heading."""
        wz = float(np.clip(TURN_KP * heading_err, -MAX_WZ, MAX_WZ))
        return np.array([0.0, 0.0, wz], dtype=np.float32)

    def _cmd_straight2(self, robot_yaw: float) -> np.ndarray:
        """Walk in the new heading direction (target_heading), maintaining alignment."""
        heading_err = _angle_diff(self._target_heading, robot_yaw)
        wz = float(np.clip(TURN_KP * heading_err, -MAX_WZ, MAX_WZ))
        return np.array([FORWARD_VX, 0.0, wz], dtype=np.float32)

    @property
    def state(self) -> State:
        return self._state

    @property
    def landmark_passed(self) -> bool:
        return self._pass_triggered


if __name__ == "__main__":
    import sys
    from code.maneuver_scene import sample_maneuver_scene, derive_rng

    rng = derive_rng(42, 0)
    scene = sample_maneuver_scene(rng)
    expert = ManeuverExpert(scene)
    expert.reset()

    rx, ry = scene["robot_xy"]
    robot_yaw = 0.0

    print(f"Scene: {scene['instruction']}")
    print(f"Landmark at x={scene['landmark_xy'][0]:.2f} y={scene['landmark_xy'][1]:.2f}")
    print(f"Turn: {scene['turn_direction']}  target={math.degrees(scene['target_heading']):.1f}°")
    print()

    for t in range(5):
        vel, priv = expert.step((rx, ry), robot_yaw)
        print(f"t={t}  state={State(priv['fsm_state']).name:<12}  vel={vel}  heading_err={math.degrees(priv['heading_err']):.1f}°  passed={priv['landmark_passed']}")
        rx += vel[0] * 0.02 * 10  # simulate 10 steps of 20ms

    print("\nManeuver expert smoke PASS")
