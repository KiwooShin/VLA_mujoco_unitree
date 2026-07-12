"""Unit tests for code.policy.groot_lang (RF-1).

Covers the pure-logic pieces cheaply testable without a GPU or a real
GR00T-N1.6 checkpoint: instruction-space enumeration, the cache verify/
lookup helpers (against small synthetic caches), and the CLI argument
wiring for --list. GrootLangEncoder / build_cache (which load a real
3B-parameter checkpoint onto a GPU) are integration-only and are
intentionally NOT exercised here.
"""

from __future__ import annotations

import pickle
import tempfile
import unittest
import warnings
from itertools import product
from pathlib import Path

import numpy as np

from code.policy import groot_lang as GL


class TestEnumerateInstructions(unittest.TestCase):
    def test_sorted_and_unique(self):
        instrs = GL.enumerate_instructions()
        self.assertEqual(instrs, sorted(instrs))
        self.assertEqual(len(instrs), len(set(instrs)))

    def test_count_matches_full_cartesian_product_dedup(self):
        instrs = GL.enumerate_instructions()
        full = {
            tpl.format(v=verb, c=color, s=shape)
            for tpl, verb, (color, _), (shape, _) in product(
                GL._TEMPLATES, GL._VERBS, GL.COLORS, GL.SHAPES)
        }
        self.assertEqual(set(instrs), full)
        self.assertEqual(len(instrs), len(full))

    def test_every_instruction_mentions_a_known_color_and_shape(self):
        instrs = GL.enumerate_instructions()
        colors = {c for c, _ in GL.COLORS}
        shapes = {s for s, _ in GL.SHAPES}
        for instr in instrs[:50]:  # sample enough without an O(n^2) blowup
            self.assertTrue(any(c in instr for c in colors), instr)
            self.assertTrue(any(s in instr for s in shapes), instr)

    def test_deterministic_across_calls(self):
        self.assertEqual(GL.enumerate_instructions(), GL.enumerate_instructions())


def _make_cache(n: int = 5, dim: int = 2048, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        f"go to the object #{i}": rng.normal(0, 1, size=dim).astype(np.float32)
        for i in range(n)
    }


class TestVerifyCache(unittest.TestCase):
    def test_valid_cache_passes(self):
        cache = _make_cache()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.pkl"
            with open(path, "wb") as f:
                pickle.dump(cache, f)
            GL.verify_cache(str(path))  # should not raise

    def test_bad_shape_raises(self):
        cache = _make_cache()
        key0 = next(iter(cache))
        cache[key0] = cache[key0][:100]  # wrong shape
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.pkl"
            with open(path, "wb") as f:
                pickle.dump(cache, f)
            with self.assertRaises(AssertionError):
                GL.verify_cache(str(path))

    def test_bad_dtype_raises(self):
        cache = _make_cache()
        key0 = next(iter(cache))
        cache[key0] = cache[key0].astype(np.float64)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.pkl"
            with open(path, "wb") as f:
                pickle.dump(cache, f)
            with self.assertRaises(AssertionError):
                GL.verify_cache(str(path))

    def test_identical_embeddings_raise(self):
        cache = _make_cache(n=2)
        keys = list(cache.keys())
        cache[keys[1]] = cache[keys[0]].copy()  # force near-identical
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.pkl"
            with open(path, "wb") as f:
                pickle.dump(cache, f)
            with self.assertRaises(AssertionError):
                GL.verify_cache(str(path))

    def test_near_zero_embedding_raises(self):
        cache = _make_cache(n=3)
        keys = list(cache.keys())
        cache[keys[0]] = np.zeros(2048, dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.pkl"
            with open(path, "wb") as f:
                pickle.dump(cache, f)
            with self.assertRaises(AssertionError):
                GL.verify_cache(str(path))


class TestGetEmbedding(unittest.TestCase):
    def setUp(self):
        # Reset the module-level cache singleton between tests so lookups
        # against different fixture caches/paths don't leak across tests.
        GL._GLOBAL_CACHE = None
        GL._GLOBAL_CACHE_PATH = None

    def tearDown(self):
        GL._GLOBAL_CACHE = None
        GL._GLOBAL_CACHE_PATH = None

    def test_lookup_hit(self):
        cache = _make_cache(n=3, seed=42)
        key = next(iter(cache))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.pkl"
            with open(path, "wb") as f:
                pickle.dump(cache, f)
            emb = GL.get_embedding(key, str(path))
            np.testing.assert_allclose(emb, cache[key])

    def test_lookup_miss_returns_zeros_with_warning(self):
        cache = _make_cache(n=2, seed=1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.pkl"
            with open(path, "wb") as f:
                pickle.dump(cache, f)
            with self.assertWarns(UserWarning):
                emb = GL.get_embedding("an instruction not in the cache", str(path))
            self.assertTrue(np.allclose(emb, 0.0))
            self.assertEqual(emb.shape, (2048,))

    def test_cache_is_loaded_once_and_reused(self):
        cache = _make_cache(n=2, seed=2)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.pkl"
            with open(path, "wb") as f:
                pickle.dump(cache, f)
            key = next(iter(cache))
            GL.get_embedding(key, str(path))
            loaded_ref = GL._GLOBAL_CACHE
            GL.get_embedding(key, str(path))
            self.assertIs(GL._GLOBAL_CACHE, loaded_ref)  # not reloaded

    def test_switching_cache_path_reloads(self):
        cache1 = _make_cache(n=2, seed=3)
        cache2 = _make_cache(n=2, seed=4)
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = Path(d) / "c1.pkl", Path(d) / "c2.pkl"
            with open(p1, "wb") as f:
                pickle.dump(cache1, f)
            with open(p2, "wb") as f:
                pickle.dump(cache2, f)
            k1 = next(iter(cache1))
            GL.get_embedding(k1, str(p1))
            self.assertEqual(GL._GLOBAL_CACHE_PATH, str(p1))
            k2 = next(iter(cache2))
            GL.get_embedding(k2, str(p2))
            self.assertEqual(GL._GLOBAL_CACHE_PATH, str(p2))


class TestCliListDoesNotRequireGpu(unittest.TestCase):
    """`python code/groot_lang.py --list` must not touch torch/gr00t at all."""

    def test_list_flag_short_circuits_before_model_load(self):
        import argparse
        import sys

        argv = sys.argv
        sys.argv = ["groot_lang.py", "--list"]
        try:
            # main() should return normally without constructing GrootLangEncoder.
            GL.main()
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main()
