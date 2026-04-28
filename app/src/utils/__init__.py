from __future__ import annotations

from abc import ABC, abstractmethod
import torch


class VideoReader(ABC):
    def __init__(self, path: str, device: torch.device, height: int, width: int):
        self.path = path
        self.device = device
        self.height = height
        self.width = width
        self.current_frame = 0

    @abstractmethod
    def __enter__(self) -> ...:
        """Открывает видео файл."""

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Закрывает видео файл."""

    @abstractmethod
    def read_window(self, start_frame: int, window_size: int) -> torch.Tensor:
        """
        Возвращает окно кадров в строгом последовательном режиме.

        Реализация не поддерживает произвольный random access: ``start_frame`` должен
        совпадать с внутренним ожидаемым индексом (0 для первого вызова, затем
        ``previous_start + previous_window_len``).

        Returns
        -------
        frames : Tensor
            shape = [T, C, H, W]
        """

    @abstractmethod
    def read_next_window(self, window_size: int) -> tuple[torch.Tensor, int]:
        """
        Следующее последовательное окно в порядке видео.

        Returns
        -------
        frames : Tensor
            [T, C, H, W] в формате входа модели (для TorchVideoReader — float32 [0,1]).
        global_start : int
            Индекс первого кадра окна.
        """

    @property
    @abstractmethod
    def total_frames(self) -> int:
        """Число кадров из метаданных, если известно; иначе 0 (см. реализацию)."""
