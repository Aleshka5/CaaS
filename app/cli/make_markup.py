"""
CLI: классификация склеек по локальному видео, сохранение markup.json.

Модель загружается через MLFlowClient (Model Registry или runs:/... URI).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tqdm import tqdm
import torch
import torch.nn.functional as F
from loguru import logger

from app.config import Settings, VideoSettings
from app.src.repositories.mlflow.main import MLFlowClient
from app.src.utils.torch_video_reader import TorchVideoReader


def _iter_windows_sync(
    reader: TorchVideoReader, window_size: int
) -> Iterator[tuple[torch.Tensor, int]]:
    while True:
        frames, start = reader.read_next_window(window_size)
        yield frames, start
        if frames.shape[0] < 2:
            break


def _tensorize_model_output(raw: Any) -> torch.Tensor:
    """Привести сырой выход модели к тензору [B, D]."""
    if isinstance(raw, dict):
        for k in ("logits", "logit", "pred", "output"):
            if k in raw:
                raw = raw[k]
                break
        else:
            raw = next(iter(raw.values()))
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    t = raw if torch.is_tensor(raw) else torch.as_tensor(raw, dtype=torch.float32)
    t = t.detach().float()
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


def _batch_cut_probs(logits: torch.Tensor, *, cut_class_index: int) -> torch.Tensor:
    """P(cut) для каждой строки батча; logits: [B, D]."""
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits.squeeze(-1))
    return F.softmax(logits, dim=-1)[:, cut_class_index]


def _output_to_cut_proba(raw: Any, *, cut_class_index: int) -> float:
    t = _tensorize_model_output(raw)
    return float(_batch_cut_probs(t, cut_class_index=cut_class_index)[0])


def _smoke_predict(
    model: torch.nn.Module,
    *,
    height: int,
    width: int,
    device: torch.device,
    cut_class_index: int,
) -> float:
    """Пробное предсказание на шуме; размер [1,3,H,W] как у кадров из TorchVideoReader (MODEL_INPUT_*)."""
    a = torch.rand(1, 3, height, width, device=device)
    b = torch.rand(1, 3, height, width, device=device)
    with torch.inference_mode():
        raw = model(a, b)
    return _output_to_cut_proba(raw, cut_class_index=cut_class_index)


def run_markup(
    *,
    video_path: Path,
    output_path: Path,
    model_id: str,
    settings: Settings,
    video: VideoSettings,
    threshold: float,
    cut_class_index: int,
    window_size: int,
) -> None:
    device = torch.device("cuda" if settings.gpu_support and torch.cuda.is_available() else "cpu")
    if settings.gpu_support and device.type != "cuda":
        logger.warning("GPU_SUPPORT=True, но CUDA недоступна — используется CPU")
    logger.info("Устройство: {}", device)
    client = MLFlowClient(tracking_uri=settings.mlflow_tracking_uri)
    model = client.get_model(model_id, map_location=device)
    model = model.to(device)
    model.eval()

    p_noise = _smoke_predict(
        model,
        height=video.model_input_height,
        width=video.model_input_width,
        device=device,
        cut_class_index=cut_class_index,
    )
    logger.info("Smoke P(cut) на шуме: {:.6f}", p_noise)

    reader = TorchVideoReader(
        str(video_path),
        height=video.model_input_height,
        width=video.model_input_width,
        device=device,
        gpu_resize=settings.gpu_support and device.type == "cuda",
    )
    cuts: list[int] = []

    n_frames_decoded = 0
    with reader:
        window_iter = _iter_windows_sync(reader, window_size)
        with torch.inference_mode():
            for frames, start in window_iter:
                t = frames.shape[0]
                if t > 0:
                    n_frames_decoded = start + t
                if t < 2:
                    if start == 0:
                        logger.warning(
                            "В видео меньше 2 кадров или не удалось прочитать кадры, разметка пустая"
                        )
                    elif n_frames_decoded > 0:
                        logger.info("Конец видео. Всего пройдено кадров: {}", n_frames_decoded)
                    break
                x0, x1 = frames[:-1], frames[1:]
                raw = model(x0, x1)
                probs = _batch_cut_probs(
                    _tensorize_model_output(raw), cut_class_index=cut_class_index
                )
                hi = torch.nonzero(probs >= threshold, as_tuple=True)[0]
                cuts.extend(start + int(j) + 1 for j in hi.tolist())
                logger.info("Пройдено кадров: {}", n_frames_decoded)

    payload = {"cuts": cuts}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Сохранено {} ({} склеек)", output_path, len(cuts))


def _resolve_output_path(output_target: Path, video_path: Path, *, is_multi: bool) -> Path:
    filename = f"{video_path.stem}_markup.json"
    if output_target.exists() and output_target.is_dir():
        return output_target / filename
    if is_multi or output_target.suffix == "":
        return output_target / filename
    return output_target.parent / f"{video_path.stem}_{output_target.name}"


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level:<8}</level> {message}\n")
    p = argparse.ArgumentParser(description="Разметка склеек по видео (MLflow + TorchVideoReader)")
    # p.add_argument(
    #     "video_paths",
    #     nargs="+",
    #     type=Path,
    #     help="Один или несколько путей к видеофайлам",
    # )
    # p.add_argument(
    #     "output_json",
    #     type=Path,
    #     help=(
    #         "Каталог для результатов или файл-шаблон для одного видео. "
    #         "Имена файлов включают название видео."
    #     ),
    # )
    p.add_argument(
        "--model-id",
        default="cnn_scene_classifier",
        help="Имя в Model Registry (подставится версия), либо URI: models:/..., runs:/...",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.61,
        help="Порог P(cut) для пары соседних кадров",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=3000,
        metavar="N",
        help="Размер окна read_next_window; все пары в окне передаются в один forward",
    )
    args = p.parse_args()

    video_paths = [
        "D:/Films/converted_96/GOT/GOT1.mp4",
        "D:/Films/converted_96/GOT/GOT2.mp4",
        "D:/Films/converted_96/GOT/GOT3.mp4",
        "D:/Films/converted_96/GOT/GOT4.mp4",
        "D:/Films/converted_96/GOT/GOT5.mp4",
        "D:/Films/converted_96/GOT/GOT6.mp4",
        "D:/Films/converted_96/GOT/GOT7.mp4",
        "D:/Films/converted_96/GOT/GOT8.mp4",
        "D:/Films/converted_96/GOT/GOT9.mp4",
        "D:/Films/converted_96/GOT/GOT10.mp4",
    ]

    missing = [
        str(Path(video_path)) for video_path in video_paths if not Path(video_path).is_file()
    ]
    if missing:
        raise SystemExit("Файлы не найдены:\n" + "\n".join(missing))
    if args.window_size < 2:
        raise SystemExit("--window-size должен быть >= 2")
    settings = Settings()
    video = VideoSettings()

    is_multi = len(video_paths) > 1
    for video_path in tqdm(video_paths):
        out = _resolve_output_path(
            Path("D:/Films/converted_96/markups"), Path(video_path), is_multi=is_multi
        )
        run_markup(
            video_path=video_path,
            output_path=out,
            model_id=args.model_id,
            settings=settings,
            video=video,
            threshold=args.threshold,
            cut_class_index=1,
            window_size=args.window_size,
        )


if __name__ == "__main__":
    main()
