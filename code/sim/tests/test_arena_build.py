"""Unit tests for code.sim.arena_build (scene construction + config constants)."""

import subprocess
import sys
import unittest

import mujoco
import numpy as np

from code.sim.arena_build import (
    CAMERA_MODE,
    COLORS,
    EGO_H,
    EGO_W,
    GROUNDING_H,
    GROUNDING_W,
    PROXIMITY_H,
    PROXIMITY_W,
    SHAPES,
    TP_H,
    TP_W,
    _add_geom,
    _rgb255_to_rgba1,
    build_arena,
)


def _minimal_scene_cfg(objects=None) -> dict:
    if objects is None:
        objects = [
            {"color_name": "red", "color_rgb": (220, 40, 40),
             "shape_name": "ball", "size": 0.24, "x": 1.0, "y": 0.0},
        ]
    return {"arena_size": 4.0, "objects": objects, "lighting": {"ambient": 0.4}}


class TestPalettes(unittest.TestCase):
    def test_colors_are_unique_names(self) -> None:
        names = [c[0] for c in COLORS]
        self.assertEqual(len(names), len(set(names)))

    def test_colors_are_valid_rgb_uint8(self) -> None:
        for name, rgb in COLORS:
            self.assertEqual(len(rgb), 3)
            for ch in rgb:
                self.assertGreaterEqual(ch, 0)
                self.assertLessEqual(ch, 255)

    def test_shapes_are_unique_names(self) -> None:
        names = [s[0] for s in SHAPES]
        self.assertEqual(len(names), len(set(names)))

    def test_shapes_have_positive_size(self) -> None:
        for _, size in SHAPES:
            self.assertGreater(size, 0.0)

    def test_seven_colors_four_shapes(self) -> None:
        """docs reference a 7-color palette and 4 shape classes (code/nx6_data.md)."""
        self.assertEqual(len(COLORS), 7)
        self.assertEqual(len(SHAPES), 4)


class TestRgb255ToRgba1(unittest.TestCase):
    def test_full_white(self) -> None:
        self.assertEqual(_rgb255_to_rgba1((255, 255, 255)), [1.0, 1.0, 1.0, 1.0])

    def test_black_default_alpha(self) -> None:
        self.assertEqual(_rgb255_to_rgba1((0, 0, 0)), [0.0, 0.0, 0.0, 1.0])

    def test_custom_alpha(self) -> None:
        self.assertEqual(_rgb255_to_rgba1((0, 0, 0), alpha=0.5), [0.0, 0.0, 0.0, 0.5])

    def test_midpoint(self) -> None:
        result = _rgb255_to_rgba1((220, 40, 40))
        self.assertAlmostEqual(result[0], 220 / 255.0)
        self.assertAlmostEqual(result[1], 40 / 255.0)
        self.assertAlmostEqual(result[2], 40 / 255.0)


class TestAddGeom(unittest.TestCase):
    def _spec(self) -> mujoco.MjSpec:
        return mujoco.MjSpec()

    def test_sets_type_size_pos_rgba(self) -> None:
        spec = self._spec()
        g = _add_geom(spec.worldbody, mujoco.mjtGeom.mjGEOM_SPHERE,
                      [0.1, 0.1, 0.1], [1.0, 2.0, 0.1], [1, 0, 0, 1], "myball")
        self.assertEqual(g.type, mujoco.mjtGeom.mjGEOM_SPHERE)
        self.assertEqual(list(g.size), [0.1, 0.1, 0.1])
        self.assertEqual(list(g.pos), [1.0, 2.0, 0.1])
        self.assertEqual(list(g.rgba), [1, 0, 0, 1])
        self.assertEqual(g.name, "myball")

    def test_name_optional(self) -> None:
        spec = self._spec()
        g = _add_geom(spec.worldbody, mujoco.mjtGeom.mjGEOM_BOX,
                      [0.1, 0.1, 0.1], [0, 0, 0], [1, 1, 1, 1])
        self.assertEqual(g.name, "")


class TestBuildArena(unittest.TestCase):
    def test_builds_compiled_model(self) -> None:
        model = build_arena(_minimal_scene_cfg())
        self.assertIsInstance(model, mujoco.MjModel)

    def test_offscreen_buffer_covers_all_cameras(self) -> None:
        model = build_arena(_minimal_scene_cfg())
        expected_w = max(EGO_W, GROUNDING_W, PROXIMITY_W, TP_W)
        expected_h = max(EGO_H, GROUNDING_H, PROXIMITY_H, TP_H)
        self.assertGreaterEqual(model.vis.global_.offwidth, expected_w)
        self.assertGreaterEqual(model.vis.global_.offheight, expected_h)

    def test_walls_present_by_name(self) -> None:
        model = build_arena(_minimal_scene_cfg())
        for wname in ("wall_px", "wall_nx", "wall_py", "wall_ny"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, wname)
            self.assertGreaterEqual(gid, 0, f"missing wall geom {wname}")

    def test_object_geom_present_per_shape(self) -> None:
        for shape_name, _ in SHAPES:
            objects = [{"color_name": "red", "color_rgb": (220, 40, 40),
                       "shape_name": shape_name, "size": 0.24, "x": 1.0, "y": 0.0}]
            model = build_arena(_minimal_scene_cfg(objects))
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj_0")
            self.assertGreaterEqual(gid, 0, f"missing obj_0 geom for shape={shape_name}")

    def test_unknown_shape_falls_back_to_sphere(self) -> None:
        objects = [{"color_name": "red", "color_rgb": (220, 40, 40),
                   "shape_name": "dodecahedron", "size": 0.24, "x": 1.0, "y": 0.0}]
        model = build_arena(_minimal_scene_cfg(objects))
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj_0")
        self.assertGreaterEqual(gid, 0)
        self.assertEqual(model.geom_type[gid], mujoco.mjtGeom.mjGEOM_SPHERE)

    def test_cone_adds_tip_geom(self) -> None:
        objects = [{"color_name": "red", "color_rgb": (220, 40, 40),
                   "shape_name": "cone", "size": 0.26, "x": 1.0, "y": 0.0}]
        model = build_arena(_minimal_scene_cfg(objects))
        gid_base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj_0")
        gid_tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj_0_tip")
        self.assertGreaterEqual(gid_base, 0)
        self.assertGreaterEqual(gid_tip, 0)

    def test_floor_color_set_light_grey(self) -> None:
        model = build_arena(_minimal_scene_cfg())
        fid = model.geom("floor").id
        np.testing.assert_allclose(model.geom_rgba[fid], [0.92, 0.92, 0.90, 1.0], atol=1e-6)

    def test_multiple_objects_get_distinct_names(self) -> None:
        objects = [
            {"color_name": "red", "color_rgb": (220, 40, 40),
             "shape_name": "ball", "size": 0.24, "x": 1.0, "y": 0.0},
            {"color_name": "blue", "color_rgb": (50, 90, 220),
             "shape_name": "cube", "size": 0.24, "x": -1.0, "y": 0.5},
        ]
        model = build_arena(_minimal_scene_cfg(objects))
        gid0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj_0")
        gid1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj_1")
        self.assertGreaterEqual(gid0, 0)
        self.assertGreaterEqual(gid1, 0)
        self.assertNotEqual(gid0, gid1)

    def test_empty_object_list_still_builds(self) -> None:
        model = build_arena(_minimal_scene_cfg(objects=[]))
        self.assertIsInstance(model, mujoco.MjModel)

    def test_default_camera_mode_is_cam2(self) -> None:
        """CAMERA_MODE toggle defaults to 'cam2' unless CAMERA_MODE env var is set."""
        self.assertEqual(CAMERA_MODE, "cam2")


class TestCameraModeEnvToggle(unittest.TestCase):
    """CAMERA_MODE is read once at import time from the environment; exercise
    the validation branch via a subprocess so we don't mutate this process's
    already-imported module state."""

    def _run(self, value: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", "import code.sim.arena_build"],
            env={"CAMERA_MODE": value, "PYTHONPATH": ".", "MUJOCO_GL": "egl",
                 "PATH": "/usr/bin:/bin"},
            cwd=".",
            capture_output=True,
            text=True,
        )

    def test_widefov_is_accepted(self) -> None:
        result = self._run("widefov")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_mode_raises_value_error(self) -> None:
        result = self._run("bogus")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown CAMERA_MODE", result.stderr)


if __name__ == "__main__":
    unittest.main()
