from __future__ import annotations

from typing import Any, Iterator

import av
import torch
import torch.nn.functional as F

from . import VideoReader


def _metadata_from_open_container(container: av.container.Container) -> tuple[float, int | None]:
    """FPS и оценка числа кадров из уже открытого контейнера (без декодирования)."""
    vs = next(s for s in container.streams if s.type == "video")
    tb = vs.time_base
    rate = vs.average_rate
    if rate is None or rate == 0:
        fps = 30.0
    else:
        fps = float(rate)

    n: int | None = None
    if vs.frames and vs.frames > 0:
        n = int(vs.frames)
    elif vs.duration is not None:
        dur_s = float(vs.duration * tb)
        n = max(1, int(round(dur_s * fps)))
    elif container.duration is not None:
        dur_s = float(container.duration) / float(av.time_base)
        n = max(1, int(round(dur_s * fps)))

    return fps, n


class TorchVideoReader(VideoReader):
    """
    Последовательное чтение видео через PyAV: один проход декодера, скользящее окно.

    Кадры нумеруются в порядке декода (без seek по fps). Выход — float32 [0, 1], NCHW на ``device``.
    """

    def __init__(
        self,
        path: str,
        height: int,
        width: int,
        device: torch.device = torch.device("cpu"),
        *,
        gpu_resize: bool = False,
    ) -> None:
        super().__init__(path=path, device=device, height=height, width=width)
        self._gpu_resize = bool(gpu_resize) and device.type == "cuda"
        self._total_frames: int | None = None
        self._fps: float | None = None
        self._container: av.container.InputContainer | None = None
        self._video_stream: av.video.stream.VideoStream | None = None
        self._decode_iter: Iterator[Any] | None = None
        self._next_expected_start: int = 0

    def __enter__(self) -> "TorchVideoReader":
        self._container = av.open(self.path)
        assert self._container is not None
        self._video_stream = next(s for s in self._container.streams if s.type == "video")
        self._fps, self._total_frames = _metadata_from_open_container(self._container)
        self._decode_iter = self._container.decode(self._video_stream)
        self._next_expected_start = 0
        self.current_frame = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._container is not None:
            self._container.close()
        self._container = None
        self._video_stream = None
        self._decode_iter = None
        self._total_frames = None
        self._fps = None
        self._next_expected_start = 0

    def _ensure_opened(self) -> None:
        if self._container is None or self._decode_iter is None:
            raise RuntimeError("Video not opened. Use context manager: with reader as r: ...")

    def _decode_raw_frames(self, n: int) -> torch.Tensor | None:
        """Декодировать до n кадров; RGB uint8 [T, H, W, 3]. Пустой список → None."""
        if n <= 0:
            return None
        self._ensure_opened()
        frames_list: list[torch.Tensor] = []
        for _ in range(n):
            try:
                frame = next(self._decode_iter)
                # logger.info(f"Decoded frame {frame.pts} | {_memory_log_suffix()}")
            except StopIteration:
                break
            arr = frame.to_ndarray(format="rgb24")
            # Преобразуем numpy array в torch тензор сразу для каждого кадра
            t = torch.from_numpy(arr)  # [H, W, 3], uint8, CPU
            frames_list.append(t)
        if not frames_list:
            return None
        # Stack сразу на CPU в итоговый батч [T, H, W, 3]
        batch = torch.stack(frames_list, dim=0)
        return batch

    def _resize_and_normalize(self, batch_uint8_thwc: torch.Tensor) -> torch.Tensor:
        """[T,H,W,3] uint8 → [T,3,h,w] float32 [0,1] на ``device``."""
        x = batch_uint8_thwc.permute(0, 3, 1, 2).contiguous()
        if x.shape[2] == self.height and x.shape[3] == self.width:
            x = x.float().div_(255.0)
            return x.to(self.device, non_blocking=True)

        if self._gpu_resize:
            x = x.to(self.device, non_blocking=True).float().div_(255.0)
            return F.interpolate(
                x,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )

        x = x.float().div_(255.0)
        x = F.interpolate(
            x,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        return x.to(self.device, non_blocking=True)

    def read_next_window(self, window_size: int) -> tuple[torch.Tensor, int]:
        """
        Следующее окно в порядке декода.

        Каждое окно содержит до ``window_size`` последовательных кадров без prefetch/перекрытия
        между вызовами.

        Returns
        -------
        frames : Tensor
            [T, 3, H, W], float32 на ``device``, значения [0, 1].
        global_start : int
            Индекс первого кадра окна в порядке декода.
        """
        self._ensure_opened()
        if window_size <= 0:
            return (
                torch.zeros(
                    (0, 3, self.height, self.width), dtype=torch.float32, device=self.device
                ),
                self._next_expected_start,
            )

        global_start = self._next_expected_start
        raw = self._decode_raw_frames(window_size)
        if raw is None:
            return (
                torch.zeros(
                    (0, 3, self.height, self.width), dtype=torch.float32, device=self.device
                ),
                global_start,
            )

        try:
            frames = self._resize_and_normalize(raw)
        finally:
            del raw
        t = frames.shape[0]
        self._next_expected_start = global_start + t
        self.current_frame = global_start + t
        return frames, global_start

    def read_window(self, start_frame: int, window_size: int) -> torch.Tensor:
        """
        Последовательное чтение: ``start_frame`` должен совпадать с ожидаемым индексом
        (0 для первого вызова, затем ``previous_start + t``). Иначе — ``ValueError``.
        """
        self._ensure_opened()
        if start_frame != self._next_expected_start:
            raise ValueError(
                f"read_window({start_frame}, ...) нарушает последовательность: "
                f"ожидался start_frame={self._next_expected_start}. "
                "Используйте read_next_window(window_size) или новый контекст ридера."
            )
        frames, _ = self.read_next_window(window_size)
        return frames

    @property
    def total_frames(self) -> int:
        """Оценка по метаданным; 0 если число кадров в контейнере не указано."""
        self._ensure_opened()
        return int(self._total_frames or 0)
