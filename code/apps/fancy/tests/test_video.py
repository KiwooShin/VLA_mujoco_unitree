"""Unit tests for code.apps.fancy.video: MP4 writer + showcase-reel
concatenation (cheap real-file I/O; no rendering).
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from code.apps.fancy.video import _concat_reel, _write_fancy_video


class WriteFancyVideoTest(unittest.TestCase):
    def test_writes_file_and_returns_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sub", "out.mp4")
            frames = [np.zeros((16, 24, 3), dtype=np.uint8) for _ in range(5)]
            result = _write_fancy_video(frames, path, fps=10)
            self.assertEqual(result, path)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_empty_frames_returns_path_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "empty.mp4")
            result = _write_fancy_video([], path)
            self.assertEqual(result, path)
            self.assertFalse(os.path.isfile(path))


class ConcatReelTest(unittest.TestCase):
    def test_no_valid_paths_returns_none(self) -> None:
        self.assertIsNone(_concat_reel(["/nonexistent1.mp4", None, ""], "/tmp/out_reel.mp4"))

    def test_concatenates_two_clips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            clip_paths = []
            for i in range(2):
                p = os.path.join(td, f"clip{i}.mp4")
                frames = [np.full((10, 10, 3), i * 50, dtype=np.uint8) for _ in range(3)]
                _write_fancy_video(frames, p, fps=5)
                clip_paths.append(p)

            reel_path = os.path.join(td, "reel.mp4")
            out = _concat_reel(clip_paths, reel_path)
            self.assertEqual(out, reel_path)
            self.assertTrue(os.path.isfile(reel_path))
            self.assertGreater(os.path.getsize(reel_path), 0)

    def test_filters_missing_paths_before_concatenating(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "clip0.mp4")
            _write_fancy_video([np.zeros((8, 8, 3), dtype=np.uint8)], p, fps=5)
            reel_path = os.path.join(td, "reel.mp4")
            out = _concat_reel([p, "/does/not/exist.mp4"], reel_path)
            self.assertEqual(out, reel_path)


if __name__ == "__main__":
    unittest.main()
