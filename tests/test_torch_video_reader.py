"""Проверка последовательных окон TorchVideoReader (без overlap/prefetch)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import importlib.util

import torch

from app.src.utils.torch_video_reader import TorchVideoReader


def _make_short_mp4(path: Path, *, frames: int = 12, fps: int = 25) -> None:
    """Синтетическое видео через ffmpeg lavfi."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={frames / fps}:size=64x48:rate={fps}",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


@unittest.skipUnless(shutil.which("ffmpeg"), "нужен ffmpeg в PATH")
@unittest.skipUnless(torch.cuda.is_available(), "нужна CUDA для avcuda reader")
@unittest.skipUnless(
    importlib.util.find_spec("avcuda") is not None, "нужен установленный avcuda"
)
class TestTorchVideoReaderSequential(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.video_path = Path(self.tmp.name) / "t.mp4"
        _make_short_mp4(self.video_path, frames=12, fps=25)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sequential_window_indices_and_shape(self) -> None:
        device = torch.device("cuda")
        window = 5
        with TorchVideoReader(
            str(self.video_path),
            height=24,
            width=32,
            device=device,
            gpu_resize=False,
        ) as reader:
            f0, s0 = reader.read_next_window(window)
            self.assertEqual(s0, 0)
            self.assertEqual(f0.shape, (5, 3, 24, 32))
            self.assertEqual(f0.dtype, torch.float32)
            self.assertTrue(f0.min() >= 0 and f0.max() <= 1)

            f1, s1 = reader.read_next_window(window)
            self.assertEqual(s1, 5)
            self.assertEqual(f1.shape, (5, 3, 24, 32))

            # Дочитываем до конца потока без падений
            while True:
                f, _s = reader.read_next_window(window)
                if f.shape[0] < 2:
                    break

        # Декодировано 12 кадров; первое окно 0..4, второе начинается с 5.
        self.assertEqual(s0 + f0.shape[0], 5)
        self.assertEqual(s1, 5)

    def test_read_window_requires_sequence(self) -> None:
        device = torch.device("cuda")
        with TorchVideoReader(
            str(self.video_path),
            height=24,
            width=32,
            device=device,
            gpu_resize=False,
        ) as reader:
            reader.read_window(0, 3)
            with self.assertRaises(ValueError):
                reader.read_window(0, 3)


if __name__ == "__main__":
    unittest.main()
