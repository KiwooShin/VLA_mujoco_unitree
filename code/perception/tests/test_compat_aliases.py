"""Verifies the RF-1 sys.modules alias pattern for group G2-perception: every
old flat `code/<name>.py` import path resolves to the EXACT SAME module
object as its new `code.perception.*` home, and mutable state written
through one path is observed through the other (docs/refactor_plan.md
invariant 3 + 5)."""
from __future__ import annotations

import unittest

import code.grounding as old_grounding
import code.lock_mgmt as old_lock_mgmt
import code.nx6_heatmap_data as old_nx6_data
import code.nx6_heatmap_eval_utils as old_nx6_eval_utils
import code.nx6_heatmap_model as old_nx6_model
import code.perception.detector.data as new_detector_data
import code.perception.detector.eval_utils as new_detector_eval_utils
import code.perception.detector.model as new_detector_model
import code.perception.grounding as new_grounding
import code.perception.lock_mgmt as new_lock_mgmt


class TestModuleIdentity(unittest.TestCase):
    """sys.modules[old_name] = real_module makes both import paths refer to
    the literal same object -- no copying, so state cannot drift."""

    def test_grounding_alias_identity(self):
        self.assertIs(old_grounding, new_grounding)

    def test_lock_mgmt_alias_identity(self):
        self.assertIs(old_lock_mgmt, new_lock_mgmt)

    def test_nx6_heatmap_model_alias_identity(self):
        self.assertIs(old_nx6_model, new_detector_model)

    def test_nx6_heatmap_data_alias_identity(self):
        self.assertIs(old_nx6_data, new_detector_data)

    def test_nx6_heatmap_eval_utils_alias_identity(self):
        self.assertIs(old_nx6_eval_utils, new_detector_eval_utils)


class TestStateMutationVisibleAcrossPaths(unittest.TestCase):
    """The concrete proof the task asks for: mutate module state through the
    OLD import path and confirm the NEW path observes it (and vice versa)."""

    def setUp(self):
        # Snapshot so we can restore -- other tests in the suite share this
        # process-wide singleton.
        self._orig_optout = old_grounding._STATE.optout_notified
        self._orig_track_dist = old_grounding._STATE.track_dist_m
        self._orig_track_bearing = old_grounding._STATE.track_bearing_rad
        self._orig_ground_fn = old_grounding.ground

    def tearDown(self):
        old_grounding._STATE.optout_notified = self._orig_optout
        old_grounding._STATE.track_dist_m = self._orig_track_dist
        old_grounding._STATE.track_bearing_rad = self._orig_track_bearing
        new_grounding.ground = self._orig_ground_fn

    def test_ground_net_state_mutation_old_to_new(self):
        old_grounding._STATE.optout_notified = True
        self.assertTrue(new_grounding._STATE.optout_notified)

    def test_ground_net_state_mutation_new_to_old(self):
        new_grounding._STATE.track_dist_m = 3.3
        self.assertEqual(old_grounding._STATE.track_dist_m, 3.3)

    def test_reset_ground_net_track_via_old_path_visible_via_new(self):
        new_grounding._STATE.track_dist_m = 1.0
        new_grounding._STATE.track_bearing_rad = 0.2
        old_grounding.reset_ground_net_track()
        self.assertIsNone(new_grounding._STATE.track_dist_m)
        self.assertIsNone(new_grounding._STATE.track_bearing_rad)

    def test_monkeypatch_ground_attribute_via_old_path_seen_by_new_path_callers(self):
        # This is the exact pattern code/gen_det_failcases.py uses:
        # `grounding_mod.ground = _instrumented_ground` where grounding_mod is
        # `import code.grounding as grounding_mod`.
        sentinel = object()

        def _fake(*a, **kw):
            return sentinel

        old_grounding.ground = _fake
        self.assertIs(new_grounding.ground(None, None, None, None, None), sentinel)

    def test_lock_mgmt_toggle_constants_identical_object_via_both_paths(self):
        self.assertEqual(old_lock_mgmt.LOCK_M1, new_lock_mgmt.LOCK_M1)
        self.assertIs(old_lock_mgmt.LockGate, new_lock_mgmt.LockGate)


class TestOldPathPublicSurfaceComplete(unittest.TestCase):
    """Every name external callers import from the OLD flat paths (per the
    grep audit in docs/refactor_plan.md's invariant 3) must resolve."""

    def test_grounding_public_surface(self):
        names = ["ground", "GroundingResult", "HSV_BOUNDS", "MIN_BLOB_AREA", "EROSION_ITER",
                "IMG_MARGIN_LEFT", "IMG_MARGIN_RIGHT", "IMG_MARGIN_BOTTOM", "MIN_DEPTH_M",
                "MAX_DEPTH_M", "MIN_VALID_DEPTH_PX", "get_ego_intrinsics_rendered",
                "cam_to_egocentric", "CAM_ROBOT_FORWARD_OFFSET_M", "_parse_instruction",
                "_score_all_contours", "get_ground_net_last_heatmap", "reset_ground_net_track",
                "ground_net_latency_stats", "validate_grounding", "GROUND_NET"]
        for name in names:
            self.assertTrue(hasattr(old_grounding, name), msg=f"missing {name}")

    def test_lock_mgmt_public_surface(self):
        for name in ("LockGate", "ReacquisitionScan"):
            self.assertTrue(hasattr(old_lock_mgmt, name), msg=f"missing {name}")

    def test_nx6_heatmap_model_public_surface(self):
        for name in ("HeatmapDetector", "TinyHeatmapUNet", "CLASS_NAMES", "COLOR_NAMES",
                    "TARGET_W", "TARGET_H", "encode_query", "decode_single"):
            self.assertTrue(hasattr(old_nx6_model, name), msg=f"missing {name}")

    def test_nx6_heatmap_data_public_surface(self):
        for name in ("SplitCache", "load_failcase_cache", "build_example_index", "ALL_COMBOS",
                    "HeatmapDataset", "collate", "oversample_far_or_wide", "gaussian_heatmap",
                    "residual_target"):
            self.assertTrue(hasattr(old_nx6_data, name), msg=f"missing {name}")

    def test_nx6_heatmap_eval_utils_public_surface(self):
        for name in ("InferenceResult", "run_inference", "select_threshold",
                    "score_at_threshold", "presence_only_pr"):
            self.assertTrue(hasattr(old_nx6_eval_utils, name), msg=f"missing {name}")


if __name__ == "__main__":
    unittest.main()
