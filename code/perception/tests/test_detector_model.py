"""Unit tests for code/perception/detector/model.py: TinyHeatmapUNet,
decode_single, and the HeatmapDetector inference wrapper."""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from code.perception.detector.model import (CLASS_NAMES, COLOR_NAMES, N_CLASS, N_COLOR,
                                            TARGET_H, TARGET_W, HeatmapDetector,
                                            TinyHeatmapUNet, _refine_peak, decode_single,
                                            encode_query)


class TestEncodeQuery(unittest.TestCase):

    def test_shape_and_onehot_positions(self):
        v = encode_query(class_id=1, color_id=3)
        self.assertEqual(v.shape, (N_CLASS + N_COLOR,))
        self.assertEqual(v.dtype, np.float32)
        self.assertEqual(v[1], 1.0)
        self.assertEqual(v[N_CLASS + 3], 1.0)
        self.assertEqual(v.sum(), 2.0)

    def test_all_class_ids_produce_distinct_vectors(self):
        vecs = [tuple(encode_query(c, 0)) for c in range(N_CLASS)]
        self.assertEqual(len(set(vecs)), N_CLASS)


class TestTinyHeatmapUNet(unittest.TestCase):

    def test_forward_output_shapes(self):
        m = TinyHeatmapUNet()
        x = torch.randn(2, 4, TARGET_H, TARGET_W)
        q = torch.zeros(2, N_CLASS + N_COLOR)
        q[:, 0] = 1.0
        q[:, N_CLASS] = 1.0
        heat, resid = m(x, q)
        self.assertEqual(heat.shape, (2, TARGET_H, TARGET_W))
        self.assertEqual(resid.shape, (2, TARGET_H, TARGET_W))

    def test_num_params_matches_manual_sum(self):
        m = TinyHeatmapUNet()
        expected = sum(p.numel() for p in m.parameters())
        self.assertEqual(m.num_params(), expected)
        self.assertGreater(m.num_params(), 0)

    def test_custom_base_width_changes_param_count(self):
        small = TinyHeatmapUNet(base=8)
        big = TinyHeatmapUNet(base=32)
        self.assertLess(small.num_params(), big.num_params())

    def test_head_bias_initialized_toward_no_detection(self):
        m = TinyHeatmapUNet()
        self.assertAlmostEqual(float(m.head.bias[0].item()), -3.0, places=5)
        self.assertAlmostEqual(float(m.head.bias[1].item()), 0.0, places=5)


class TestRefinePeak(unittest.TestCase):

    def test_degenerate_all_zero_patch_returns_hard_argmax(self):
        heat = np.zeros((10, 10), dtype=np.float32)
        cx, cy = _refine_peak(heat, py=4, px=6, win=2)
        self.assertEqual((cx, cy), (6.0, 4.0))

    def test_symmetric_peak_refines_to_center(self):
        heat = np.zeros((20, 20), dtype=np.float32)
        heat[10, 10] = 1.0
        heat[9, 10] = 0.5
        heat[11, 10] = 0.5
        heat[10, 9] = 0.5
        heat[10, 11] = 0.5
        cx, cy = _refine_peak(heat, py=10, px=10, win=2)
        self.assertAlmostEqual(cx, 10.0, places=5)
        self.assertAlmostEqual(cy, 10.0, places=5)

    def test_asymmetric_peak_shifts_toward_heavier_side(self):
        heat = np.zeros((20, 20), dtype=np.float32)
        heat[10, 10] = 1.0
        heat[10, 11] = 1.0   # extra mass to the right
        cx, cy = _refine_peak(heat, py=10, px=10, win=1)
        self.assertGreater(cx, 10.0)


class TestDecodeSingle(unittest.TestCase):

    def test_confident_peak_present_true(self):
        H, W = TARGET_H, TARGET_W
        heat_logit = np.full((H, W), -10.0, dtype=np.float32)
        py, px = H // 2, W // 2
        heat_logit[py, px] = 10.0   # sigmoid(10) ~ 0.9999
        resid = np.zeros((H, W), dtype=np.float32)
        depth = np.full((H, W), 3.0, dtype=np.float32)
        out = decode_single(heat_logit, resid, depth, class_id=0, cam_type="grounding",
                            conf_thresh=0.5)
        self.assertTrue(out["present"])
        self.assertGreater(out["confidence"], 0.99)
        self.assertGreater(out["dist_m"], 0.0)
        self.assertIn("bearing_deg", out)
        self.assertIn("peak_px", out)

    def test_low_confidence_present_false(self):
        H, W = TARGET_H, TARGET_W
        heat_logit = np.full((H, W), -10.0, dtype=np.float32)
        heat_logit[H // 2, W // 2] = -5.0   # still the argmax but low sigmoid
        resid = np.zeros((H, W), dtype=np.float32)
        depth = np.full((H, W), 3.0, dtype=np.float32)
        out = decode_single(heat_logit, resid, depth, class_id=0, cam_type="grounding",
                            conf_thresh=0.5)
        self.assertFalse(out["present"])

    def test_residual_added_to_backprojected_distance(self):
        H, W = TARGET_H, TARGET_W
        heat_logit = np.full((H, W), -10.0, dtype=np.float32)
        py, px = H // 2, W // 2
        heat_logit[py, px] = 10.0
        depth = np.full((H, W), 3.0, dtype=np.float32)

        resid_zero = np.zeros((H, W), dtype=np.float32)
        out_zero = decode_single(heat_logit, resid_zero, depth, class_id=0, cam_type="grounding")

        resid_plus = np.zeros((H, W), dtype=np.float32)
        resid_plus[py, px] = 1.5
        out_plus = decode_single(heat_logit, resid_plus, depth, class_id=0, cam_type="grounding")

        self.assertAlmostEqual(out_plus["dist_m"] - out_zero["dist_m"], 1.5, places=4)

    def test_proximity_vs_grounding_cam_type_differ(self):
        H, W = TARGET_H, TARGET_W
        heat_logit = np.full((H, W), -10.0, dtype=np.float32)
        heat_logit[H // 2, W // 4] = 10.0   # off-center peak so bearing is nonzero
        resid = np.zeros((H, W), dtype=np.float32)
        depth = np.full((H, W), 3.0, dtype=np.float32)
        out_g = decode_single(heat_logit, resid, depth, class_id=0, cam_type="grounding")
        out_p = decode_single(heat_logit, resid, depth, class_id=0, cam_type="proximity")
        self.assertNotAlmostEqual(out_g["dist_m"], out_p["dist_m"], places=3)


class TestHeatmapDetector(unittest.TestCase):

    def setUp(self):
        self.model = TinyHeatmapUNet(base=8)   # small/fast for test speed
        self.det = HeatmapDetector(self.model, device="cpu")

    def test_construction_sets_eval_mode_and_empty_cache(self):
        self.assertFalse(self.model.training)
        self.assertIsNone(self.det.last_heat_prob)
        self.assertIsNone(self.det.last_heat_meta)

    def test_infer_returns_expected_keys(self):
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        depth = np.full((240, 320), 3.0, dtype=np.float32)
        out = self.det.infer(rgb, depth, class_name=CLASS_NAMES[0], color_name=COLOR_NAMES[0],
                             cam_type="grounding")
        for k in ("present", "confidence", "dist_m", "bearing_deg", "peak_px"):
            self.assertIn(k, out)

    def test_infer_populates_vf1_cache(self):
        rgb = np.zeros((360, 480, 3), dtype=np.uint8)
        depth = np.full((360, 480), 4.0, dtype=np.float32)
        self.det.infer(rgb, depth, class_name=CLASS_NAMES[1], color_name=COLOR_NAMES[2],
                      cam_type="proximity")
        self.assertIsNotNone(self.det.last_heat_prob)
        self.assertEqual(self.det.last_heat_prob.shape, (TARGET_H, TARGET_W))
        self.assertEqual(self.det.last_heat_meta["class_name"], CLASS_NAMES[1])
        self.assertEqual(self.det.last_heat_meta["color_name"], COLOR_NAMES[2])
        self.assertEqual(self.det.last_heat_meta["cam_type"], "proximity")

    def test_infer_accepts_either_native_resolution(self):
        # grounding-cam-native (480x360) and proximity-cam-native (320x240) frames
        # both resize internally to the same TARGET canvas.
        depth_g = np.full((360, 480), 3.0, dtype=np.float32)
        rgb_g = np.zeros((360, 480, 3), dtype=np.uint8)
        out_g = self.det.infer(rgb_g, depth_g, CLASS_NAMES[0], COLOR_NAMES[0], "grounding")

        depth_p = np.full((240, 320), 1.0, dtype=np.float32)
        rgb_p = np.zeros((240, 320, 3), dtype=np.uint8)
        out_p = self.det.infer(rgb_p, depth_p, CLASS_NAMES[0], COLOR_NAMES[0], "proximity")
        self.assertIn("dist_m", out_g)
        self.assertIn("dist_m", out_p)

    def test_load_round_trip(self):
        import tempfile
        from pathlib import Path

        cfg = dict(base=8)
        model = TinyHeatmapUNet(**cfg)
        with tempfile.TemporaryDirectory() as d:
            ckpt_path = Path(d) / "tiny.pt"
            torch.save(dict(model_cfg=cfg, model_state=model.state_dict()), ckpt_path)
            loaded = HeatmapDetector.load(str(ckpt_path), device="cpu")
        self.assertIsInstance(loaded, HeatmapDetector)
        self.assertFalse(loaded.model.training)
        # A forward pass on the freshly loaded model should run without error.
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        depth = np.full((240, 320), 2.0, dtype=np.float32)
        out = loaded.infer(rgb, depth, CLASS_NAMES[0], COLOR_NAMES[0], "grounding")
        self.assertIn("dist_m", out)

    def test_infer_batch_tensor_shapes(self):
        x_t = torch.randn(3, 4, TARGET_H, TARGET_W)
        q_t = torch.zeros(3, N_CLASS + N_COLOR)
        heat, resid = self.det.infer_batch_tensor(x_t, q_t)
        self.assertEqual(heat.shape, (3, TARGET_H, TARGET_W))
        self.assertEqual(resid.shape, (3, TARGET_H, TARGET_W))


if __name__ == "__main__":
    unittest.main()
