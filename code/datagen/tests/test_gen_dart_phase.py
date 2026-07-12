"""Unit tests for code.datagen.gen_dart_phase (RF-1): GaitPhaseTracker + build_proprio."""

from __future__ import annotations

import math
import unittest

import mujoco
import numpy as np

from code.datagen.gen_dart_phase import (
    LEFT_ANKLE_DEFAULT,
    LEFT_ANKLE_PITCH_IDX,
    GaitPhaseTracker,
    build_proprio,
)
from code.teacher import CONTROL_DECIMATION, SIM_DT


def _q_lb(ankle_val: float) -> np.ndarray:
    q = np.zeros(15, dtype=np.float32)
    q[LEFT_ANKLE_PITCH_IDX] = ankle_val
    return q


class GaitPhaseTrackerTest(unittest.TestCase):
    def test_first_update_returns_fixed_point(self) -> None:
        tracker = GaitPhaseTracker()
        sin_phi, cos_phi = tracker.update(_q_lb(0.3))
        self.assertEqual((sin_phi, cos_phi), (0.0, 1.0))

    def test_dt_matches_control_period(self) -> None:
        tracker = GaitPhaseTracker()
        self.assertAlmostEqual(tracker._dt, SIM_DT * CONTROL_DECIMATION, places=9)

    def test_phase_is_always_unit_circle(self) -> None:
        tracker = GaitPhaseTracker()
        rng = np.random.default_rng(0)
        for _ in range(200):
            ankle = float(rng.uniform(-1.0, 1.0)) + LEFT_ANKLE_DEFAULT
            sin_phi, cos_phi = tracker.update(_q_lb(ankle))
            self.assertAlmostEqual(sin_phi ** 2 + cos_phi ** 2, 1.0, places=5)

    def test_phase_advances_monotonically_without_crossing(self) -> None:
        tracker = GaitPhaseTracker(freq_hz=1.8)
        tracker.update(_q_lb(LEFT_ANKLE_DEFAULT - 1.0))  # init below zero (relative)
        phis = []
        # Keep feeding negative-relative values so no positive zero-crossing fires.
        for _ in range(5):
            tracker.update(_q_lb(LEFT_ANKLE_DEFAULT - 0.5))
            phis.append(tracker._phi)
        self.assertEqual(phis, sorted(phis))
        # Expected per-step increment.
        step = 2.0 * math.pi * 1.8 * (SIM_DT * CONTROL_DECIMATION)
        self.assertAlmostEqual(phis[1] - phis[0], step, places=6)

    def test_positive_zero_crossing_resets_phase(self) -> None:
        tracker = GaitPhaseTracker()
        tracker.update(_q_lb(LEFT_ANKLE_DEFAULT - 0.5))  # init, prev_q < 0 relative
        # Advance a few steps below zero.
        for _ in range(3):
            tracker.update(_q_lb(LEFT_ANKLE_DEFAULT - 0.3))
        # Now cross to positive relative value -> phase resets to 0 this step.
        sin_phi, cos_phi = tracker.update(_q_lb(LEFT_ANKLE_DEFAULT + 0.3))
        self.assertAlmostEqual(sin_phi, 0.0, places=6)
        self.assertAlmostEqual(cos_phi, 1.0, places=6)

    def test_wraparound_stays_in_0_2pi(self) -> None:
        tracker = GaitPhaseTracker(freq_hz=50.0)  # fast to force wraparound quickly
        tracker.update(_q_lb(LEFT_ANKLE_DEFAULT - 1.0))
        for _ in range(500):
            tracker.update(_q_lb(LEFT_ANKLE_DEFAULT - 1.0))
        self.assertGreaterEqual(tracker._phi, 0.0)
        self.assertLess(tracker._phi, 2.0 * math.pi)

    def test_two_independent_trackers_are_deterministic(self) -> None:
        seq = np.linspace(-1.0, 1.0, 50)
        t1, t2 = GaitPhaseTracker(), GaitPhaseTracker()
        out1 = [t1.update(_q_lb(v)) for v in seq]
        out2 = [t2.update(_q_lb(v)) for v in seq]
        self.assertEqual(out1, out2)


def _make_test_model(n_hinges: int = 15) -> mujoco.MjModel:
    body_open, body_close = "", ""
    for i in range(n_hinges):
        body_open += (
            f'<body name="l{i}" pos="0 0 -0.05">'
            f'<joint name="j{i}" type="hinge" axis="0 1 0"/>'
            f'<geom type="capsule" fromto="0 0 0 0 0 -0.05" size="0.02"/>'
        )
        body_close += "</body>"
    xml = (
        '<mujoco><worldbody><body name="base" pos="0 0 1">'
        '<freejoint/><geom name="pelvis" type="sphere" size="0.1"/>'
        f"{body_open}{body_close}</body></worldbody></mujoco>"
    )
    return mujoco.MjModel.from_xml_string(xml)


class BuildProprioDartTest(unittest.TestCase):
    """Confirms the intentionally-duplicated build_proprio (RF-1
    no-consolidation invariant) behaves identically to gen_dataset_rollout's
    copy for the same inputs."""

    def test_matches_gen_dataset_rollout_copy(self) -> None:
        from code.datagen.gen_dataset_rollout import build_proprio as build_proprio_other

        model = _make_test_model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        data.qpos[7:22] = np.linspace(0, 1, 15)
        data.qvel[6:21] = np.linspace(1, 0, 15)
        prev_action = np.arange(15, dtype=np.float32)

        p1 = build_proprio(data, prev_action)
        p2 = build_proprio_other(data, prev_action)
        np.testing.assert_array_equal(p1, p2)

    def test_shape(self) -> None:
        model = _make_test_model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        p = build_proprio(data, np.zeros(15, dtype=np.float32))
        self.assertEqual(p.shape, (55,))
        self.assertEqual(p.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
