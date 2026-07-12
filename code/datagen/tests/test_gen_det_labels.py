"""Unit tests for code.datagen.gen_det_labels (RF-1).

build_id_to_obj uses a small synthetic mujoco model with named geoms (no
real arena needed). seg_to_objmap and derive_object_labels are pure
array/dict logic and are tested with fully synthetic inputs.
"""

from __future__ import annotations

import math
import unittest

import mujoco
import numpy as np

from code.arena import GROUNDING_PITCH, backproject_pixel
from code.datagen.gen_det_common import COLOR2I, MIN_PIXELS, SHAPE2I
from code.datagen.gen_det_labels import build_id_to_obj, derive_object_labels, seg_to_objmap
from code.grounding import cam_to_egocentric
from code.steer import egocentric_goal


def _make_model_with_objects(n_objects: int = 3, include_cone_tip: bool = True) -> mujoco.MjModel:
    bodies = []
    for i in range(n_objects):
        geom = f'<geom name="obj_{i}" type="sphere" size="0.1"/>'
        if include_cone_tip and i == n_objects - 1:
            geom += f'<geom name="obj_{i}_tip" type="sphere" size="0.02" pos="0 0 0.1"/>'
        bodies.append(f'<body name="obj_body_{i}" pos="{i} 0 0.5"><freejoint/>{geom}</body>')
    xml = f'<mujoco><worldbody>{"".join(bodies)}<geom name="floor" type="plane" size="5 5 0.1"/></worldbody></mujoco>'
    return mujoco.MjModel.from_xml_string(xml)


class BuildIdToObjTest(unittest.TestCase):
    def test_maps_each_obj_geom_to_its_index(self) -> None:
        model = _make_model_with_objects(3, include_cone_tip=False)
        id_to_obj = build_id_to_obj(model, n_objects=3)
        self.assertEqual(id_to_obj.shape, (model.ngeom,))
        for i in range(3):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obj_{i}")
            self.assertEqual(id_to_obj[gid], i)

    def test_cone_tip_maps_to_same_object_index(self) -> None:
        model = _make_model_with_objects(3, include_cone_tip=True)
        id_to_obj = build_id_to_obj(model, n_objects=3)
        gid_base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj_2")
        gid_tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj_2_tip")
        self.assertEqual(id_to_obj[gid_base], 2)
        self.assertEqual(id_to_obj[gid_tip], 2)

    def test_unrelated_geoms_are_unmapped(self) -> None:
        model = _make_model_with_objects(3, include_cone_tip=False)
        id_to_obj = build_id_to_obj(model, n_objects=3)
        gid_floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.assertEqual(id_to_obj[gid_floor], -1)

    def test_n_objects_cutoff_ignores_higher_indices(self) -> None:
        model = _make_model_with_objects(3, include_cone_tip=False)
        # Only claim 2 objects are "real" -- obj_2 must be excluded.
        id_to_obj = build_id_to_obj(model, n_objects=2)
        gid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj_2")
        self.assertEqual(id_to_obj[gid2], -1)


class SegToObjmapTest(unittest.TestCase):
    def test_maps_geom_ids_through_lookup(self) -> None:
        id_to_obj = np.array([-1, 5, 7, -1], dtype=np.int32)
        seg = np.zeros((2, 2, 2), dtype=np.int32)
        seg[..., 0] = np.array([[1, 2], [-1, 3]])
        obj_map = seg_to_objmap(seg, id_to_obj)
        expected = np.array([[5, 7], [-1, -1]])
        np.testing.assert_array_equal(obj_map, expected)

    def test_out_of_range_ids_are_clipped_not_crashed(self) -> None:
        id_to_obj = np.array([-1, 0], dtype=np.int32)
        seg = np.zeros((1, 1, 2), dtype=np.int32)
        seg[..., 0] = 999  # way out of range but "valid" (>=0)
        obj_map = seg_to_objmap(seg, id_to_obj)
        self.assertEqual(obj_map[0, 0], id_to_obj[-1])


def _rect_mask(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class DeriveObjectLabelsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.h, self.w = 50, 60
        self.intr = dict(fx=300.0, fy=300.0, cx=30.0, cy=25.0)

    def test_below_min_pixels_is_excluded(self) -> None:
        obj_map = -np.ones((self.h, self.w), dtype=np.int32)
        obj_map[0, 0] = 0  # single pixel < MIN_PIXELS
        objects = [dict(x=1.0, y=1.0, shape_name="cube", color_name="red")]
        rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        depth = np.full((self.h, self.w), 2.0, dtype=np.float32)
        labels = derive_object_labels(rgb, depth, obj_map, objects,
                                      robot_xy=np.zeros(2), robot_yaw=0.0, intr=self.intr)
        self.assertEqual(labels, [])
        self.assertGreater(MIN_PIXELS, 1)

    def test_basic_bbox_area_centroid_not_clipped(self) -> None:
        obj_map = -np.ones((self.h, self.w), dtype=np.int32)
        mask = _rect_mask(self.h, self.w, y0=10, y1=20, x0=20, x1=30)
        obj_map[mask] = 0
        objects = [dict(x=1.0, y=1.0, shape_name="cube", color_name="red")]
        rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        depth = np.full((self.h, self.w), 2.0, dtype=np.float32)

        labels = derive_object_labels(rgb, depth, obj_map, objects,
                                      robot_xy=np.zeros(2), robot_yaw=0.0, intr=self.intr)
        self.assertEqual(len(labels), 1)
        lb = labels[0]
        self.assertEqual(lb["obj_idx"], 0)
        self.assertEqual(lb["bbox_x"], 20)
        self.assertEqual(lb["bbox_y"], 10)
        self.assertEqual(lb["bbox_w"], 10)
        self.assertEqual(lb["bbox_h"], 10)
        self.assertEqual(lb["area_px"], 100)
        self.assertAlmostEqual(lb["centroid_px_x"], 24.5)
        self.assertAlmostEqual(lb["centroid_px_y"], 14.5)
        self.assertFalse(lb["clipped"])
        self.assertEqual(lb["class_name"], "cube")
        self.assertEqual(lb["class_id"], SHAPE2I["cube"])
        self.assertEqual(lb["color_name"], "red")
        self.assertEqual(lb["color_id"], COLOR2I["red"])
        self.assertAlmostEqual(lb["depth_median_m"], 2.0, places=5)

    def test_edge_touching_mask_is_clipped(self) -> None:
        obj_map = -np.ones((self.h, self.w), dtype=np.int32)
        mask = _rect_mask(self.h, self.w, y0=0, y1=10, x0=0, x1=10)  # touches (0,0) edge
        obj_map[mask] = 0
        objects = [dict(x=1.0, y=1.0, shape_name="ball", color_name="blue")]
        rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        depth = np.full((self.h, self.w), 1.5, dtype=np.float32)
        labels = derive_object_labels(rgb, depth, obj_map, objects,
                                      robot_xy=np.zeros(2), robot_yaw=0.0, intr=self.intr)
        self.assertTrue(labels[0]["clipped"])

    def test_ground_truth_matches_egocentric_goal(self) -> None:
        obj_map = -np.ones((self.h, self.w), dtype=np.int32)
        mask = _rect_mask(self.h, self.w, y0=10, y1=20, x0=20, x1=30)
        obj_map[mask] = 0
        objects = [dict(x=1.0, y=1.0, shape_name="cube", color_name="red")]
        rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        depth = np.full((self.h, self.w), 2.0, dtype=np.float32)
        robot_xy = np.array([0.2, -0.3])
        robot_yaw = 0.4

        labels = derive_object_labels(rgb, depth, obj_map, objects,
                                      robot_xy=robot_xy, robot_yaw=robot_yaw, intr=self.intr)
        exp_dist, exp_yaw_err, _ = egocentric_goal(robot_xy, robot_yaw, np.array([1.0, 1.0]))
        self.assertAlmostEqual(labels[0]["dist_gt_m"], exp_dist, places=5)
        self.assertAlmostEqual(labels[0]["bearing_gt_deg"], math.degrees(exp_yaw_err), places=5)

    def test_backprojection_matches_direct_call_and_uses_pitch_fallback(self) -> None:
        obj_map = -np.ones((self.h, self.w), dtype=np.int32)
        mask = _rect_mask(self.h, self.w, y0=10, y1=20, x0=20, x1=30)
        obj_map[mask] = 0
        objects = [dict(x=1.0, y=1.0, shape_name="cube", color_name="red")]
        rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        depth = np.full((self.h, self.w), 2.0, dtype=np.float32)

        # intr has no 'pitch_deg' -> derive_object_labels must fall back to GROUNDING_PITCH.
        labels = derive_object_labels(rgb, depth, obj_map, objects,
                                      robot_xy=np.zeros(2), robot_yaw=0.0, intr=self.intr)
        cx, cy = 24.5, 14.5
        radius = 0.24 / 2.0  # cube not in SIZE_M's explicit shapes falls back? use real value below
        from code.datagen.gen_det_common import SIZE_M
        radius = float(SIZE_M["cube"]) / 2.0
        x_cam, y_cam, z_cam = backproject_pixel(cx, cy, 2.0, self.intr)
        exp_dist, exp_yerr = cam_to_egocentric(x_cam, y_cam, z_cam + radius,
                                               pitch_deg=GROUNDING_PITCH,
                                               use_corrected_unpitch=True)
        self.assertAlmostEqual(labels[0]["dist_bp_m"], float(exp_dist), places=5)
        self.assertAlmostEqual(labels[0]["bearing_bp_deg"], math.degrees(exp_yerr), places=5)

    def test_all_invalid_depth_gives_negative_median_and_nan_bp(self) -> None:
        obj_map = -np.ones((self.h, self.w), dtype=np.int32)
        mask = _rect_mask(self.h, self.w, y0=10, y1=20, x0=20, x1=30)
        obj_map[mask] = 0
        objects = [dict(x=1.0, y=1.0, shape_name="cube", color_name="red")]
        rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        depth = np.full((self.h, self.w), -1.0, dtype=np.float32)  # all invalid
        labels = derive_object_labels(rgb, depth, obj_map, objects,
                                      robot_xy=np.zeros(2), robot_yaw=0.0, intr=self.intr)
        self.assertEqual(labels[0]["depth_median_m"], -1.0)
        self.assertTrue(math.isnan(labels[0]["dist_bp_m"]))

    def test_unknown_shape_and_color_get_minus_one_id(self) -> None:
        obj_map = -np.ones((self.h, self.w), dtype=np.int32)
        mask = _rect_mask(self.h, self.w, y0=10, y1=20, x0=20, x1=30)
        obj_map[mask] = 0
        objects = [dict(x=1.0, y=1.0, shape_name="dodecahedron", color_name="mauve")]
        rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        depth = np.full((self.h, self.w), 2.0, dtype=np.float32)
        labels = derive_object_labels(rgb, depth, obj_map, objects,
                                      robot_xy=np.zeros(2), robot_yaw=0.0, intr=self.intr)
        self.assertEqual(labels[0]["class_id"], -1)
        self.assertEqual(labels[0]["color_id"], -1)

    def test_multiple_objects_only_matching_mask_included(self) -> None:
        obj_map = -np.ones((self.h, self.w), dtype=np.int32)
        obj_map[_rect_mask(self.h, self.w, 0, 10, 0, 10)] = 0
        obj_map[_rect_mask(self.h, self.w, 30, 45, 30, 45)] = 1
        objects = [
            dict(x=1.0, y=0.0, shape_name="ball", color_name="red"),
            dict(x=2.0, y=0.0, shape_name="cone", color_name="green"),
        ]
        rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        depth = np.full((self.h, self.w), 2.0, dtype=np.float32)
        labels = derive_object_labels(rgb, depth, obj_map, objects,
                                      robot_xy=np.zeros(2), robot_yaw=0.0, intr=self.intr)
        self.assertEqual(len(labels), 2)
        idxs = sorted(lb["obj_idx"] for lb in labels)
        self.assertEqual(idxs, [0, 1])


if __name__ == "__main__":
    unittest.main()
