"""
code.runtime.gait_phase — gait phase tracker (Fix 4).

RF-1 split of code/inferencer.py (docs/refactor_plan.md): moved verbatim,
no logic changes.
"""

from __future__ import annotations

import math

import numpy as np

from code.sim.teacher import SIM_DT, CONTROL_DECIMATION
from code.runtime.constants import _LEFT_ANKLE_PITCH_IDX, _LEFT_ANKLE_DEFAULT


class _GaitPhaseTracker:
    """Tracks gait phase phi in [0, 2pi] from left ankle pitch zero-crossings.

    Returns (sin_phi, cos_phi) as a 2-d gait-phase encoding.
    Same implementation as gen_dart_dataset.py.
    """

    def __init__(self, freq_hz: float = 1.8) -> None:
        """Initializes the tracker.

        Args:
            freq_hz: Nominal gait frequency (Hz) used to advance phase between
                zero-crossings.
        """
        self._phi: float = 0.0
        self._prev_q: float = 0.0
        self._initialized: bool = False
        self._freq_hz = freq_hz
        self._dt = SIM_DT * CONTROL_DECIMATION   # 0.02 s

    def update(self, q_lb: np.ndarray) -> np.ndarray:
        """Advances the phase estimate by one control step.

        Args:
            q_lb: (15,) lower-body joint positions.

        Returns:
            np.float32[2] array [sin(phi), cos(phi)].
        """
        q_ankle = float(q_lb[_LEFT_ANKLE_PITCH_IDX]) - _LEFT_ANKLE_DEFAULT
        if not self._initialized:
            self._prev_q = q_ankle
            self._initialized = True
            return np.array([0.0, 1.0], dtype=np.float32)

        self._phi += 2.0 * math.pi * self._freq_hz * self._dt
        if self._prev_q < 0.0 and q_ankle >= 0.0:
            self._phi = 0.0
        self._prev_q = q_ankle
        self._phi = self._phi % (2.0 * math.pi)
        return np.array([math.sin(self._phi), math.cos(self._phi)], dtype=np.float32)
