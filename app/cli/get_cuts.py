from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from loguru import logger
from safetensors.torch import save_file
import torch

from app.config import VideoSettings
from app.src.utils.torch_video_reader import TorchVideoReader


def _collect_frame_ids(markup_payload: Any) -> list[int]:
    """Извлечь индексы кадров склеек из разных форматов markup."""
    if isinstance(markup_payload, dict):
        if "cuts" in markup_payload:
            raw_items = markup_payload["cuts"]
        elif "frames" in markup_payload:
            raw_items = markup_payload["frames"]
        else:
            raw_items = markup_payload
    else:
        raw_items = markup_payload

    ids: list[int] = []

    def _append_if_int(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            ids.append(value)
            return
        if isinstance(value, str) and value.isdigit():
            ids.append(int(value))

    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict):
                if "id" in item:
                    _append_if_int(item["id"])
                elif "frame_id" in item:
                    _append_if_int(item["frame_id"])
            else:
                _append_if_int(item)
    elif isinstance(raw_items, dict):
        for key, value in raw_items.items():
            if isinstance(value, dict) and "id" in value:
                _append_if_int(value["id"])
            else:
                _append_if_int(key)
    else:
        _append_if_int(raw_items)

    ids = sorted(set(frame_id for frame_id in ids if frame_id >= 0))
    return ids


def _load_markup_ids(markup_path: Path) -> list[int]:
    payload = json.loads(markup_path.read_text(encoding="utf-8"))
    ids = _collect_frame_ids(payload)
    if not ids:
        logger.warning("В {} не найдено валидных frame ids", markup_path)
    return ids


def _resolve_markup_path(video_path: Path, markup_dir: Path) -> Path:
    return markup_dir / f"{video_path.stem}_markup.json"


def _gather_cut_frames(
    video_path: Path,
    frame_ids: list[int],
    *,
    target_height: int,
    target_width: int,
    device: torch.device,
    use_gpu_resize: bool,
    window_size: int,
) -> tuple[torch.Tensor, list[int]]:
    if not frame_ids:
        return torch.empty((0, 3, target_height, target_width), dtype=torch.float32), []

    target_set = set(frame_ids)
    collected: list[torch.Tensor] = []
    found_ids: list[int] = []

    with TorchVideoReader(
        str(video_path),
        height=target_height,
        width=target_width,
        device=device,
        gpu_resize=use_gpu_resize,
    ) as reader:
        while True:
            frames, start = reader.read_next_window(window_size)
            count = frames.shape[0]
            if count == 0:
                break

            stop = start + count
            local_hits = [frame_id for frame_id in frame_ids if start <= frame_id < stop]
            if local_hits:
                for frame_id in local_hits:
                    local_idx = frame_id - start
                    # index даёт view; без clone() держим в RAM весь батч окна на каждый кадр
                    collected.append(frames[local_idx].detach().cpu().contiguous().clone())
                    found_ids.append(frame_id)
                    target_set.discard(frame_id)

            del frames

            if not target_set:
                break

    if target_set:
        logger.warning(
            "Для {} не найдены кадры по ids: {}",
            video_path.name,
            sorted(target_set),
        )

    if not collected:
        return torch.empty((0, 3, target_height, target_width), dtype=torch.float32), []

    order = sorted(range(len(found_ids)), key=lambda i: found_ids[i])
    ordered_frames = [collected[idx] for idx in order]
    ordered_ids = [found_ids[idx] for idx in order]
    return torch.stack(ordered_frames, dim=0), ordered_ids


def _process_one_video_job(
    job: tuple[Path, Path, Path, int, int, int],
) -> str:
    """
    Один ролик в отдельном процессе (нужен top-level для multiprocessing / Windows spawn).

    Кортеж: video_path, markup_dir, output_dir, target_height, target_width, window_size.
    Возвращает str(out_path) для логов в родителе.
    """
    torch.set_num_threads(1)
    (
        video_path,
        markup_dir,
        output_dir,
        target_height,
        target_width,
        window_size,
    ) = job

    device = torch.device("cpu")
    use_gpu_resize = False

    markup_path = _resolve_markup_path(video_path, markup_dir)
    if not markup_path.is_file():
        raise FileNotFoundError(f"Markup файл не найден: {markup_path}")

    frame_ids = _load_markup_ids(markup_path)
    logger.info(
        "[{}] Сборка cuts для {}: {} ids, target_size={}x{}",
        os.getpid(),
        video_path.name,
        len(frame_ids),
        target_height,
        target_width,
    )
    frames, found_ids = _gather_cut_frames(
        video_path,
        frame_ids,
        target_height=target_height,
        target_width=target_width,
        device=device,
        use_gpu_resize=use_gpu_resize,
        window_size=window_size,
    )
    ids_tensor = torch.tensor(found_ids, dtype=torch.int64)

    out_path = output_dir / f"{video_path.stem}_cuts.sft"
    save_file(
        {
            "cuts": frames.contiguous(),
            "ids": ids_tensor.contiguous(),
        },
        str(out_path),
    )
    logger.info("[{}] Сохранено {}: cuts shape={}", os.getpid(), out_path, tuple(frames.shape))
    return str(out_path)


def run_get_cuts(
    *,
    video_paths: list[Path],
    markup_dir: Path,
    output_dir: Path,
    target_size: tuple[int, int],
    window_size: int,
    workers: int = 1,
) -> None:
    target_height, target_width = target_size
    output_dir.mkdir(parents=True, exist_ok=True)

    # Этот CLI: только CPU (декод/resize без H2D/D2H; settings.gpu_support не влияет)
    logger.info("Устройство чтения кадров: cpu (get_cuts — принудительно CPU)")

    for video_path in video_paths:
        markup_path = _resolve_markup_path(video_path, markup_dir)
        if not markup_path.is_file():
            raise FileNotFoundError(f"Markup файл не найден: {markup_path}")

    if workers < 1:
        raise ValueError("workers must be >= 1")

    if workers == 1:
        for video_path in video_paths:
            _process_one_video_job(
                (
                    video_path,
                    markup_dir,
                    output_dir,
                    target_height,
                    target_width,
                    window_size,
                )
            )
        return

    jobs = [
        (vp, markup_dir, output_dir, target_height, target_width, window_size) for vp in video_paths
    ]
    logger.info("Параллельно: {} воркеров, {} роликов", workers, len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(_process_one_video_job, jobs):
            logger.info("Готово: {}", out)


def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    parser = argparse.ArgumentParser(
        description="Собрать .sft с кадрами склеек по {movie}_markup.json"
    )
    # parser.add_argument(
    #     "video_paths",
    #     nargs="+",
    #     type=Path,
    #     help="Один или несколько путей к видеофайлам",
    # )
    parser.add_argument(
        "--markup-dir",
        type=Path,
        default=Path("D:/Films/converted_96/markups"),
        help="Каталог с файлами формата {movie}_markup.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("D:/Films/converted_96/sfts"),
        help="Каталог для итоговых {movie}_cuts.sft",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(240, 560),
        help="Целевой размер кадров (height width), по умолчанию TARGET_HEIGHT TARGET_WIDTH",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1024,
        metavar="N",
        help="Размер окна последовательного чтения через TorchVideoReader",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Число параллельных процессов (каждый обрабатывает один ролик; следующий в очереди — когда воркер освободится). ~N×RAM",
    )
    args = parser.parse_args()

    video_paths = [
        Path("D:/Films/converted_96/GOT/GOT4.mp4"),
        # Path("D:/Films/converted_96/GOT/GOT5.mp4"),
        # Path("D:/Films/converted_96/GOT/GOT6.mp4"),
        # Path("D:/Films/converted_96/GOT/GOT7.mp4"),
        # Path("D:/Films/converted_96/GOT/GOT8.mp4"),
        # Path("D:/Films/converted_96/GOT/GOT9.mp4"),
        # Path("D:/Films/converted_96/GOT/GOT10.mp4"),
    ]

    missing = [path for path in video_paths if not path.is_file()]
    if missing:
        raise SystemExit("Видеофайлы не найдены:\n" + "\n".join(missing))
    if args.window_size <= 0:
        raise SystemExit("--window-size должен быть > 0")
    if args.workers < 1:
        raise SystemExit("--workers должен быть >= 1")

    video_settings = VideoSettings()
    target_size = (
        tuple(args.target_size)
        if args.target_size is not None
        else (video_settings.target_height, video_settings.target_width)
    )

    run_get_cuts(
        video_paths=video_paths,
        markup_dir=args.markup_dir,
        output_dir=args.output_dir,
        target_size=target_size,
        window_size=args.window_size,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
