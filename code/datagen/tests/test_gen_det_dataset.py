"""Unit/integration tests for code.datagen.gen_det_dataset (RF-1): the CLI aggregator.

Covers the re-exported old-path-compat surface (gen_det_failcases.py and
eval/nx14_gen1_confusion/capture.py import these from `code.gen_det_dataset`),
the CLI shim, and one small end-to-end generate()+make_preview() smoke.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from code.datagen.gen_det_dataset import (
    SegRenderer,
    build_id_to_obj,
    derive_object_labels,
    generate,
    make_preview,
    pick_cam,
    seg_to_objmap,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _subprocess_env() -> dict:
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    existing_pp = env.get("PYTHONPATH", "")
    root = str(_REPO_ROOT)
    env["PYTHONPATH"] = root if not existing_pp else f"{root}:{existing_pp}"
    return env


class ReExportsTest(unittest.TestCase):
    """gen_det_failcases.py does:
        from code.gen_det_dataset import (build_id_to_obj, SegRenderer,
            seg_to_objmap, derive_object_labels, pick_cam)
    All five names must remain importable from this module after RF-1.
    """

    def test_all_five_names_present_and_callable(self) -> None:
        self.assertTrue(callable(build_id_to_obj))
        self.assertTrue(isinstance(SegRenderer, type))
        self.assertTrue(callable(seg_to_objmap))
        self.assertTrue(callable(derive_object_labels))
        self.assertTrue(callable(pick_cam))


class CliShimTest(unittest.TestCase):
    def test_help_lists_expected_flags(self) -> None:
        result = subprocess.run(
            [sys.executable, "code/gen_det_dataset.py", "--help"],
            capture_output=True, text=True, timeout=60,
            cwd=str(_REPO_ROOT), env=_subprocess_env(),
        )
        self.assertEqual(result.returncode, 0)
        for flag in ("--n-easy", "--n-demo", "--n-search", "--seed", "--out", "--smoke"):
            self.assertIn(flag, result.stdout)

    def test_smoke_end_to_end_via_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "det_out")
            result = subprocess.run(
                [sys.executable, "code/gen_det_dataset.py", "--smoke",
                 "--n-easy", "1", "--n-demo", "0", "--n-search", "0",
                 "--seed", "3", "--out", out, "--no-preview"],
                capture_output=True, text=True, timeout=120,
                cwd=str(_REPO_ROOT), env=_subprocess_env(),
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            meta = json.loads((Path(out) / "meta.json").read_text())
            self.assertGreater(meta["frames_total"], 0)
            self.assertEqual(meta["scenes"], 1)


class GenerateAndPreviewTest(unittest.TestCase):
    """Direct (in-process) end-to-end smoke of generate() + make_preview()."""

    def test_generate_and_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "det_out")
            args = argparse.Namespace(
                n_easy=1, n_demo=0, n_search=0, seed=17, out=out,
                smoke=True, smoke_scenes=1,
            )
            meta, out_dir = generate(args)
            self.assertGreater(meta["frames_total"], 0)
            self.assertEqual(meta["scenes"], 1)
            self.assertTrue((out_dir / "scenes.json").exists())
            self.assertTrue((out_dir / "train" / "frames.parquet").exists())

            n_written = make_preview(out_dir, n_samples=3, seed=1)
            self.assertEqual(n_written, min(3, meta["frames_total"]))
            preview_files = list((out_dir / "preview").glob("*.png"))
            self.assertEqual(len(preview_files), n_written)


if __name__ == "__main__":
    unittest.main()
