"""Unit tests for code.eval.search_types: sample_search_scene + SearchResult.

Covers: determinism (seeded rng), the out-of-FOV placement invariant that is
the entire point of the search skill, distance bounds, non-overlap, arena
bounds, and the SearchResult dataclass's serialization shape (dicts crossing
a boundary must stay dicts per docs/refactor_plan.md invariant 6).
"""

from __future__ import annotations

import dataclasses
import math
import unittest

import numpy as np

from code.eval.search_types import (
    sample_search_scene, SearchResult,
    SEARCH_FOV_HALF_DEG, SEARCH_DIST_MIN, SEARCH_DIST_MAX,
    STOP_R_SEARCH, MAXSTEPS_SEARCH,
)


class TestSampleSearchSceneDeterminism(unittest.TestCase):
    """Seeded determinism: same seed -> byte-identical scene dict."""

    def test_same_seed_same_scene(self):
        for seed in (0, 1, 999, 12345):
            rng_a = np.random.default_rng(seed)
            rng_b = np.random.default_rng(seed)
            scene_a = sample_search_scene(rng_a, episode_idx=0)
            scene_b = sample_search_scene(rng_b, episode_idx=0)
            self.assertEqual(scene_a['robot_xy'], scene_b['robot_xy'])
            self.assertEqual(scene_a['instruction'], scene_b['instruction'])
            self.assertEqual(len(scene_a['objects']), len(scene_b['objects']))
            for oa, ob in zip(scene_a['objects'], scene_b['objects']):
                self.assertEqual(oa['x'], ob['x'])
                self.assertEqual(oa['y'], ob['y'])
                self.assertEqual(oa['color_name'], ob['color_name'])
                self.assertEqual(oa['shape_name'], ob['shape_name'])

    def test_different_episode_idx_does_not_affect_rng_consumption(self):
        """episode_idx is bookkeeping only -- the scene depends solely on rng state."""
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        scene_a = sample_search_scene(rng_a, episode_idx=0)
        scene_b = sample_search_scene(rng_b, episode_idx=999)
        self.assertEqual(scene_a['robot_xy'], scene_b['robot_xy'])
        self.assertEqual(scene_a['objects'], scene_b['objects'])

    def test_different_seed_generally_differs(self):
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        scene_a = sample_search_scene(rng_a, 0)
        scene_b = sample_search_scene(rng_b, 0)
        # Overwhelmingly likely to differ in at least the robot position.
        self.assertNotEqual(scene_a['robot_xy'], scene_b['robot_xy'])


class TestSampleSearchSceneInvariants(unittest.TestCase):
    """The out-of-FOV placement guarantee + geometry sanity, across many seeds."""

    N_SEEDS = 60

    def _scenes(self):
        for seed in range(self.N_SEEDS):
            rng = np.random.default_rng(seed)
            yield seed, sample_search_scene(rng, seed)

    def test_target_outside_fov_cone(self):
        for seed, scene in self._scenes():
            with self.subTest(seed=seed):
                self.assertGreater(scene['init_bearing_deg'], SEARCH_FOV_HALF_DEG)

    def test_target_distance_in_range(self):
        for seed, scene in self._scenes():
            tgt = scene['objects'][scene['target_index']]
            with self.subTest(seed=seed):
                self.assertGreaterEqual(tgt['dist_from_robot'], SEARCH_DIST_MIN - 1e-6)
                self.assertLessEqual(tgt['dist_from_robot'], SEARCH_DIST_MAX + 1e-6)

    def test_three_objects_unique_combos(self):
        for seed, scene in self._scenes():
            with self.subTest(seed=seed):
                objs = scene['objects']
                self.assertEqual(len(objs), 3)
                combos = {(o['color_name'], o['shape_name']) for o in objs}
                self.assertEqual(len(combos), 3)

    def test_objects_within_arena_bounds(self):
        for seed, scene in self._scenes():
            half = scene['arena_size']
            with self.subTest(seed=seed):
                for o in scene['objects']:
                    self.assertLessEqual(abs(o['x']), half)
                    self.assertLessEqual(abs(o['y']), half)

    def test_objects_not_overlapping(self):
        for seed, scene in self._scenes():
            objs = scene['objects']
            with self.subTest(seed=seed):
                for i in range(len(objs)):
                    for j in range(i + 1, len(objs)):
                        d = math.hypot(objs[i]['x'] - objs[j]['x'],
                                       objs[i]['y'] - objs[j]['y'])
                        self.assertGreaterEqual(d, 0.5 - 1e-6)

    def test_instruction_mentions_target_color_and_shape(self):
        for seed, scene in self._scenes():
            tgt = scene['objects'][scene['target_index']]
            with self.subTest(seed=seed):
                self.assertIn(tgt['color_name'], scene['instruction'])
                self.assertIn(tgt['shape_name'], scene['instruction'])

    def test_scene_metadata_fields(self):
        _, scene = next(self._scenes())
        self.assertEqual(scene['stop_r'], STOP_R_SEARCH)
        self.assertEqual(scene['horizon'], MAXSTEPS_SEARCH)
        self.assertEqual(scene['difficulty'], 'search')
        self.assertIn('lighting', scene)
        self.assertEqual(scene['robot_yaw'], 0.0)

    def test_init_bearing_matches_recomputed_bearing(self):
        """init_bearing_deg is a diagnostic; verify it matches an independent
        recomputation from the returned robot/target positions."""
        for seed, scene in self._scenes():
            rx, ry = scene['robot_xy']
            yaw = scene['robot_yaw']
            tgt = scene['objects'][scene['target_index']]
            dx, dy = tgt['x'] - rx, tgt['y'] - ry
            ang = math.atan2(dy, dx)
            expected = abs(math.degrees(math.atan2(math.sin(ang - yaw), math.cos(ang - yaw))))
            with self.subTest(seed=seed):
                self.assertAlmostEqual(expected, scene['init_bearing_deg'], places=6)


class TestSearchResultDataclass(unittest.TestCase):
    """SearchResult is a plain dataclass -- dict-crossing shape must round-trip."""

    def _make(self, **overrides) -> SearchResult:
        base = dict(
            ep_idx=0, instruction='find the red ball', target_color='red',
            target_shape='ball', target_dist=3.0, init_bearing_deg=90.0,
            spotted=True, reached=True, success=True, failure_tag='success',
            steps=500, scan_steps=200, final_dist=0.2, fell=False, ms_per_step=1.5,
        )
        base.update(overrides)
        return SearchResult(**base)

    def test_defaults(self):
        r = self._make()
        self.assertIsNone(r.video_path)
        self.assertEqual(r.avoid_bias_active_frac, 0.0)

    def test_asdict_roundtrip_is_a_plain_dict(self):
        r = self._make(video_path='eval/x.mp4', avoid_bias_active_frac=0.25)
        d = dataclasses.asdict(r)
        self.assertIsInstance(d, dict)
        self.assertEqual(d['video_path'], 'eval/x.mp4')
        self.assertEqual(d['avoid_bias_active_frac'], 0.25)
        self.assertEqual(d['ep_idx'], 0)
        # json-serializable (no numpy scalars/objects leaking through)
        import json
        json.dumps(d)


if __name__ == '__main__':
    unittest.main()
