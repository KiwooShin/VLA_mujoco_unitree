"""Unit tests for code.apps.repl.constants: CUDA probe + lang-embedding cache.

Covers: _check_cuda's safe-fallback contract, _load_lang_cache's lazy
singleton behavior (hit / miss / corrupt-file paths), and _get_lang_emb's
lookup contract.
"""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from code.apps.repl import constants as C


class CheckCudaTest(unittest.TestCase):
    def test_returns_bool(self) -> None:
        self.assertIsInstance(C._check_cuda(), bool)

    def test_import_failure_returns_false(self) -> None:
        """If `import torch` raises for any reason, the probe must swallow
        it and report False rather than propagating."""
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated: no torch")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = _fake_import
        try:
            self.assertFalse(C._check_cuda())
        finally:
            builtins.__import__ = real_import


class LangCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate the module-level singleton across tests.
        self._orig_cache = C._lang_cache
        self._orig_path = C.LANG_CACHE
        C._lang_cache = None

    def tearDown(self) -> None:
        C._lang_cache = self._orig_cache
        C.LANG_CACHE = self._orig_path

    def test_missing_file_returns_none(self) -> None:
        C.LANG_CACHE = Path("/nonexistent/path/lang_cache.pkl")
        self.assertIsNone(C._load_lang_cache())
        self.assertIsNone(C._get_lang_emb("go to the red ball"))

    def test_loads_and_caches_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lang_cache.pkl"
            payload = {"go to the red ball": [1.0, 2.0, 3.0]}
            with open(p, "wb") as f:
                pickle.dump(payload, f)
            C.LANG_CACHE = p

            cache1 = C._load_lang_cache()
            self.assertEqual(cache1, payload)
            # Second call must return the SAME cached object (singleton),
            # not re-read the file — delete the file and confirm no crash.
            p.unlink()
            cache2 = C._load_lang_cache()
            self.assertIs(cache1, cache2)

    def test_get_lang_emb_hit_and_miss(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lang_cache.pkl"
            payload = {"find the blue cube": [4.0, 5.0]}
            with open(p, "wb") as f:
                pickle.dump(payload, f)
            C.LANG_CACHE = p

            self.assertEqual(C._get_lang_emb("find the blue cube"), [4.0, 5.0])
            self.assertIsNone(C._get_lang_emb("not in cache"))

    def test_corrupt_file_returns_none_and_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lang_cache.pkl"
            p.write_bytes(b"not a pickle")
            C.LANG_CACHE = p
            self.assertIsNone(C._load_lang_cache())


class PathAndVocabTest(unittest.TestCase):
    def test_colors_shapes_nonempty_and_lowercase(self) -> None:
        self.assertTrue(all(c == c.lower() for c in C.COLORS))
        self.assertTrue(all(s == s.lower() for s in C.SHAPES))
        self.assertIn("red", C.COLORS)
        self.assertIn("ball", C.SHAPES)

    def test_maneuver_directions(self) -> None:
        self.assertEqual(set(C.MANEUVER_DIRECTIONS), {"left", "right"})

    def test_maxsteps_positive(self) -> None:
        self.assertGreater(C.MAXSTEPS_GOTO, 0)
        self.assertGreater(C.MAXSTEPS_MANEUVER, 0)


if __name__ == "__main__":
    unittest.main()
