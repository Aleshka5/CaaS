"""
Бенчмарк чтения видео через ``TorchVideoReader``.

Построчный учёт RAM (MiB по строкам) для ``read_next_window`` / декод / ресайз:

.. code-block:: powershell

    $env:CAAS_LINE_PROFILE_VIDEO = "1"
    uv run python -m memory_profiler app/src/utils/file_reading_benchmark.py --video "D:/path/video.mp4"

Или с флагом (выставляет ту же переменную до импорта ридера):

.. code-block:: powershell

    uv run python -m memory_profiler app/src/utils/file_reading_benchmark.py --line-profile-video --video "D:/path/video.mp4"

Кривая RSS процесса во времени (пакет ``memory_profiler``):

.. code-block:: powershell

    uv run mprof run python app/src/utils/file_reading_benchmark.py --video "D:/path/video.mp4"
    uv run mprof plot

Сравнение «всё на GPU» и «decode/resize на CPU, окна в ``gpu_pull`` на CUDA»:

.. code-block:: powershell

    uv run python app/src/utils/file_reading_benchmark.py --video "D:/path/video.mp4" --device cuda --sleep-seconds 0
    uv run python app/src/utils/file_reading_benchmark.py --video "D:/path/video.mp4" --accumulate-device cuda --sleep-seconds 0
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
import torch
from loguru import logger

if TYPE_CHECKING:
    from .torch_video_reader import TorchVideoReader


def _memory_log_suffix() -> str:
    rss = psutil.Process().memory_info().rss
    parts = [f"RAM RSS={rss / (1024**3):.2f} GiB"]
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        parts.append(
            f"VRAM alloc={allocated / (1024**3):.2f} GiB, reserved={reserved / (1024**3):.2f} GiB"
        )
    else:
        parts.append("VRAM=n/a (no CUDA)")
    return " | ".join(parts)


class FileReadingBenchmark:
    def __init__(self, reader: TorchVideoReader):
        self.reader = reader

    def read(
        self,
        window_size: int = 10,
        *,
        sleep_seconds: float = 1.0,
        accumulate_device: torch.device | None = None,
    ) -> None:
        """
        Читает окна через ридер. Если задан ``accumulate_device``, каждое окно
        после ``read_next_window`` переносится на это устройство и кладётся в
        ``gpu_pull`` (имя списка сохранено для экспериментов CPU→GPU).
        """
        gpu_pull: list[torch.Tensor] = []
        with self.reader as reader:
            while True:
                t0 = time.perf_counter()
                frames, _ = reader.read_next_window(window_size)
                read_ms = (time.perf_counter() - t0) * 1000.0
                if accumulate_device is not None:
                    t1 = time.perf_counter()
                    frames = frames.to(accumulate_device, non_blocking=True)
                    h2d_ms = (time.perf_counter() - t1) * 1000.0
                else:
                    h2d_ms = 0.0
                gpu_pull.append(frames)
                if frames.shape[0] < 2:
                    break
                if accumulate_device is not None:
                    logger.info(
                        f"Read {frames.shape[0]} frames: reader={reader.device.type}, "
                        f"stored={frames.device.type} | read+resize {read_ms:.1f} ms, "
                        f"H2D {h2d_ms:.1f} ms | {_memory_log_suffix()}"
                    )
                else:
                    logger.info(
                        f"Read {frames.shape[0]} frames to {frames.device.type} | "
                        f"read+resize {read_ms:.1f} ms | {_memory_log_suffix()}"
                    )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark TorchVideoReader (optional line memory profile)."
    )
    p.add_argument(
        "--video",
        type=Path,
        default=Path("D:/Films/converted/GOT_S01E01.mp4"),
        help="Path to video file",
    )
    p.add_argument("--height", type=int, default=112)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--window-size", type=int, default=200)
    p.add_argument(
        "--device",
        default="cpu",
        help='Device for TorchVideoReader decode/resize output, e.g. "cuda" or "cpu". '
        "Ignored when --accumulate-device is set (reader forced to CPU).",
    )
    p.add_argument(
        "--accumulate-device",
        default=None,
        metavar="DEVICE",
        help='If set (e.g. "cuda"), decode/resize on CPU then .to(this device) each window '
        "into the buffer list (CPU→GPU pull). Compare with --device cuda without this flag.",
    )
    p.add_argument(
        "--line-profile-video",
        action="store_true",
        help="Set CAAS_LINE_PROFILE_VIDEO before loading TorchVideoReader (use with python -m memory_profiler)",
    )
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Pause between windows (log readability)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.line_profile_video:
        os.environ["CAAS_LINE_PROFILE_VIDEO"] = "1"

    from .torch_video_reader import TorchVideoReader

    accumulate_dev: torch.device | None = None
    if args.accumulate_device:
        accumulate_dev = torch.device(args.accumulate_device)
        reader_dev = torch.device("cpu")
    else:
        reader_dev = torch.device(args.device)

    reader = TorchVideoReader(
        str(args.video),
        args.height,
        args.width,
        device=reader_dev,
    )
    benchmark = FileReadingBenchmark(reader)
    benchmark.read(
        window_size=args.window_size,
        sleep_seconds=args.sleep_seconds,
        accumulate_device=accumulate_dev,
    )


if __name__ == "__main__":
    main()
