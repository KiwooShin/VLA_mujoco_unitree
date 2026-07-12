"""
code/datagen/gen_dart_phase.py — Gait-phase tracking + proprio builder for DART generation.

Role: split out of gen_dart_dataset.py (RF-1) — the small, pure-logic pieces
shared by the DART rollout (gen_dart_rollout.py) and by other generators
(gen_maneuver_dataset.py imports GaitPhaseTracker/build_proprio from the
old `code.gen_dart_dataset` path, which keeps working via the alias).

Contents:
  - GaitPhaseTracker  — running zero-crossing gait-phase oscillator.
  - build_proprio     — 55-d proprio vector builder (identical layout to
                        code/datagen/gen_dataset_rollout.py's version — kept
                        duplicated per the RF-1 no-consolidation invariant).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from code.teacher import SIM_DT, CONTROL_DECIMATION

# Left ankle pitch index in lower-body joint positions (index 4, per dataset.md)
LEFT_ANKLE_PITCH_IDX: int  = 4    # in qpos[7:22]
LEFT_ANKLE_DEFAULT: float  = -0.2  # from teacher.py default angles


# ---------------------------------------------------------------------------
# Phase extractor — running zero-crossing counter on ankle pitch oscillation
# ---------------------------------------------------------------------------
class GaitPhaseTracker:
    """Tracks gait phase phi in [0, 2pi] using left ankle pitch zero-crossings.

    The ankle pitch oscillates sinusoidally during walking. Positive-going
    zero-crossings of (ankle_pitch - default) mark the start of each cycle.
    Phase advances at a fixed estimated frequency between crossings.

    Output: (sin(phi), cos(phi)) — 2-d unit-circle encoding.
    """

    def __init__(self, freq_hz: float = 1.8) -> None:
        """Initializes the phase tracker.

        Args:
            freq_hz: Estimated walking gait frequency in Hz, used to
                advance phase between zero-crossings.
        """
        self._phi: float = 0.0
        self._prev_q: float = 0.0
        self._initialized: bool = False
        self._freq_hz: float = freq_hz   # typical walking gait frequency
        self._dt: float = SIM_DT * CONTROL_DECIMATION  # 0.02 s

    def update(self, q_lb: np.ndarray) -> tuple[float, float]:
        """Updates phase from lower-body joint positions.

        Args:
            q_lb: (15,) joint positions (same order as dataset).

        Returns:
            A tuple (sin_phi, cos_phi) encoding the current gait phase.
        """
        q_ankle = float(q_lb[LEFT_ANKLE_PITCH_IDX]) - LEFT_ANKLE_DEFAULT

        if not self._initialized:
            self._prev_q = q_ankle
            self._initialized = True
            return (0.0, 1.0)

        # Advance phase by estimated frequency
        self._phi += 2.0 * math.pi * self._freq_hz * self._dt

        # On positive zero-crossing: reset to 0 (start of new cycle)
        if self._prev_q < 0.0 and q_ankle >= 0.0:
            self._phi = 0.0

        self._prev_q = q_ankle
        self._phi = self._phi % (2.0 * math.pi)

        return (math.sin(self._phi), math.cos(self._phi))


# ---------------------------------------------------------------------------
# Proprio builder (identical to gen_dataset_rollout.py)
# ---------------------------------------------------------------------------
def build_proprio(data: mujoco.MjData, prev_action: np.ndarray) -> np.ndarray:
    """Builds the 55-d proprioceptive observation vector from physics state.

    Args:
        data: MuJoCo data holding the current physics state.
        prev_action: Previous joint-target action (15-d), appended as part
            of the observation.

    Returns:
        A (55,) float32 array: [q_lb(15), dq_lb(15), quat(4), ang_v(3),
        lin_v(3), prev_action(15)].
    """
    q_lb   = data.qpos[7:22].copy()
    dq_lb  = data.qvel[6:21].copy()
    quat   = data.qpos[3:7].copy()
    ang_v  = data.qvel[3:6].copy()
    lin_v  = data.qvel[0:3].copy()
    return np.concatenate([
        q_lb.astype(np.float32),
        dq_lb.astype(np.float32),
        quat.astype(np.float32),
        ang_v.astype(np.float32),
        lin_v.astype(np.float32),
        prev_action.astype(np.float32),
    ])   # shape (55,)
