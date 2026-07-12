"""Unit tests for code.apps.fancy.overlays_bev: draw_avoid_overlay +
draw_bev_overlays (shape/no-crash + a handful of content-sensitive checks;
full BEV rendering fidelity is a rendering/gate concern, not a unit-test
concern)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from code.apps.fancy.constants import STATE_IDLE, STATE_MOVING, STATE_REACHED, STATE_SEARCHING
from code.apps.fancy.overlays_bev import _STATE_COLOR_MAP, draw_avoid_overlay, draw_bev_overlays


def _make_cam() -> SimpleNamespace:
    return SimpleNamespace(azimuth=225.0, elevation=-43.5, distance=17.0, lookat=[0.0, 0.0, 0.3])


class DrawAvoidOverlayTest(unittest.TestCase):
    def test_noop_when_info_none(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = draw_avoid_overlay(img, np.array([0.0, 0.0]), 0.0, _make_cam(), None, None,
                                  avoid_bias_wz=0.5, avoid_info=None)
        self.assertIs(out, img)
        self.assertTrue(np.all(out == 0))

    def test_noop_when_bias_below_deadband(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = draw_avoid_overlay(img, np.array([0.0, 0.0]), 0.0, _make_cam(), None, None,
                                  avoid_bias_wz=1e-9, avoid_info={"left": 0.5, "right": 0.0})
        self.assertIs(out, img)

    def test_draws_something_when_active(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = draw_avoid_overlay(img, np.array([0.0, 0.0]), 0.0, _make_cam(), None, None,
                                  avoid_bias_wz=0.3, avoid_info={"left": 0.8, "right": 0.1})
        self.assertGreater(int(out.sum()), 0)


class DrawBevOverlaysTest(unittest.TestCase):
    def _base_kwargs(self, **overrides) -> dict:
        base = dict(
            bev_img=np.zeros((480, 640, 3), dtype=np.uint8),
            path_trail=[np.array([0.0, 0.0]), np.array([0.5, 0.0]), np.array([1.0, 0.2])],
            target_xy=np.array([2.0, 1.0]),
            robot_xy=np.array([0.0, 0.0]),
            robot_yaw=0.0,
            bev_cam=_make_cam(),
            model=None,
            data=None,
            state=STATE_SEARCHING,
            prompt="find the red ball",
            dist_to_target=2.5,
        )
        base.update(overrides)
        return base

    def test_output_shape_matches_input(self) -> None:
        kwargs = self._base_kwargs()
        out = draw_bev_overlays(**kwargs)
        self.assertEqual(out.shape, kwargs["bev_img"].shape)

    def test_does_not_mutate_input_image(self) -> None:
        kwargs = self._base_kwargs()
        original = kwargs["bev_img"].copy()
        draw_bev_overlays(**kwargs)
        self.assertTrue(np.array_equal(kwargs["bev_img"], original))

    def test_no_target_no_crash(self) -> None:
        kwargs = self._base_kwargs(target_xy=None, dist_to_target=None)
        out = draw_bev_overlays(**kwargs)
        self.assertIsNotNone(out)

    def test_empty_path_trail_no_crash(self) -> None:
        kwargs = self._base_kwargs(path_trail=[])
        out = draw_bev_overlays(**kwargs)
        self.assertIsNotNone(out)

    def test_multi_goal_banner_taller(self) -> None:
        kwargs1 = self._base_kwargs(n_goals=1)
        kwargs2 = self._base_kwargs(n_goals=2, goal_idx=1)
        out1 = draw_bev_overlays(**kwargs1)
        out2 = draw_bev_overlays(**kwargs2)
        # Multi-goal banner (68px) is taller than single-goal (56px), so the
        # two outputs differ even outside the target/state-dependent region.
        self.assertFalse(np.array_equal(out1[-56:], out2[-56:]))

    def test_completed_targets_drawn_without_crash(self) -> None:
        kwargs = self._base_kwargs(completed_targets=[np.array([1.0, 1.0])])
        out = draw_bev_overlays(**kwargs)
        self.assertIsNotNone(out)

    def test_state_color_map_covers_all_states(self) -> None:
        for state in (STATE_IDLE, STATE_SEARCHING, STATE_MOVING, STATE_REACHED):
            self.assertIn(state, _STATE_COLOR_MAP)

    def test_avoid_viz_enabled_changes_output(self) -> None:
        kwargs_off = self._base_kwargs(avoid_bias_wz=0.0, avoid_info=None)
        kwargs_on = self._base_kwargs(avoid_bias_wz=0.4, avoid_info={"left": 0.9, "right": 0.0})
        with mock.patch("code.apps.fancy.overlays_bev.FEAT_AVOID_VIZ", True):
            out_off = draw_bev_overlays(**kwargs_off)
            out_on = draw_bev_overlays(**kwargs_on)
        self.assertFalse(np.array_equal(out_off, out_on))


if __name__ == "__main__":
    unittest.main()
