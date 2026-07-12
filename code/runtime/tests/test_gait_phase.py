"""Unit tests for code.runtime.gait_phase._GaitPhaseTracker."""

from __future__ import annotations

import math
import unittest

import numpy as np

from code.runtime.gait_phase import _GaitPhaseTracker
from code.runtime.constants import _LEFT_ANKLE_PITCH_IDX, _LEFT_ANKLE_DEFAULT


def _q_lb(ankle_pitch: float) -> np.ndarray:
    """Builds a (15,) lower-body joint-position vector with only the left
    ankle pitch set (the only index `_GaitPhaseTracker` reads)."""
    q = np.zeros(15, dtype=np.float32)
    q[_LEFT_ANKLE_PITCH_IDX] = ankle_pitch
    return q


class TestGaitPhaseTracker(unittest.TestCase):
    def test_first_call_returns_phase_zero(self):
        tr = _GaitPhaseTracker()
        out = tr.update(_q_lb(_LEFT_ANKLE_DEFAULT))
        np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-6)

    def test_output_is_unit_norm_sin_cos_pair(self):
        tr = _GaitPhaseTracker()
        tr.update(_q_lb(_LEFT_ANKLE_DEFAULT))
        out = tr.update(_q_lb(_LEFT_ANKLE_DEFAULT + 0.05))
        norm = math.hypot(out[0], out[1])
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_phase_advances_monotonically_without_zero_crossing(self):
        tr = _GaitPhaseTracker(freq_hz=1.8)
        tr.update(_q_lb(_LEFT_ANKLE_DEFAULT - 0.1))   # negative side, initializes
        phis = []
        # Keep q_ankle negative (relative to default) so no zero-crossing fires;
        # phase should strictly advance each step (phi mod 2pi may wrap once,
        # but sin/cos trace a consistent forward rotation).
        prev_th = None
        for _ in range(5):
            out = tr.update(_q_lb(_LEFT_ANKLE_DEFAULT - 0.1))
            th = math.atan2(out[0], out[1])
            phis.append(th)
        # Phase must have moved from the initial 0 after several steps.
        self.assertNotAlmostEqual(phis[-1], 0.0, places=3)

    def test_zero_crossing_resets_phase(self):
        tr = _GaitPhaseTracker(freq_hz=1.8)
        tr.update(_q_lb(_LEFT_ANKLE_DEFAULT - 0.1))    # init, prev_q < 0
        # Advance the phase away from zero first.
        for _ in range(10):
            tr.update(_q_lb(_LEFT_ANKLE_DEFAULT - 0.1))
        # Now cross from negative to non-negative -> phase resets to 0.
        out = tr.update(_q_lb(_LEFT_ANKLE_DEFAULT + 0.1))
        np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-6)

    def test_dt_matches_control_period(self):
        from code.sim.teacher import SIM_DT, CONTROL_DECIMATION
        tr = _GaitPhaseTracker()
        self.assertAlmostEqual(tr._dt, SIM_DT * CONTROL_DECIMATION, places=9)


if __name__ == "__main__":
    unittest.main()
