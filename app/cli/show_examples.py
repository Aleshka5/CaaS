from __future__ import annotations

import argparse
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file
from torchvision.utils import make_grid, save_image


def _open_in_viewer(path: Path) -> None:
    path = path.resolve()
    system = platform.system()
    if system == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
    elif system == "Darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def run_show_examples(
    *,
    sft_path: Path,
    n: int,
    seed: int | None,
    sequential: bool,
    start: int,
    output: Path,
    open_viewer: bool,
) -> None:
    if not sft_path.is_file():
        raise FileNotFoundError(f"Файл не найден: {sft_path}")

    data = load_file(str(sft_path), device="cpu")
    if "cuts" not in data:
        raise KeyError(f"В {sft_path} нет ключа 'cuts' (ожидается датасет из get_cuts: cuts, ids)")
    cuts = data["cuts"]
    ids = data.get("ids")
    if cuts.dim() != 4 or cuts.shape[1] != 3:
        raise ValueError(f"Ожидается cuts формы (N, 3, H, W), получено {tuple(cuts.shape)}")

    n_total = int(cuts.shape[0])
    if n_total == 0:
        raise SystemExit("В файле 0 кадров в cuts")
    if n <= 0:
        raise SystemExit("n должен быть > 0")

    if sequential:
        if start < 0:
            raise SystemExit("--start должен быть >= 0")
        if start >= n_total:
            raise SystemExit(
                f"--start ({start}) вне диапазона: в cuts только {n_total} кадров (индексы 0..{n_total - 1})"
            )
        k = min(n, n_total - start)
        perm = torch.arange(start, start + k, dtype=torch.long)
    else:
        k = min(n, n_total)
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            perm = torch.randperm(n_total, generator=g)[:k]
        else:
            perm = torch.tensor(random.sample(range(n_total), k), dtype=torch.long)

    sample = cuts[perm].detach().float().clamp(0.0, 1.0)

    nrow = max(1, int(math.ceil(math.sqrt(k))))
    grid = make_grid(sample, nrow=nrow, padding=2, pad_value=0.5)

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(output))

    id_str = ""
    if ids is not None and ids.numel() == n_total:
        picked = ids[perm].detach().cpu().tolist()
        id_str = f", frame_ids={picked}"

    mode = f"подряд с индекса {start}" if sequential else "случайная выборка"
    logger.info(
        "{}: {} из {} кадров{} → {}",
        mode,
        k,
        n_total,
        id_str,
        output,
    )
    if open_viewer:
        _open_in_viewer(output)


def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    parser = argparse.ArgumentParser(
        description="Выборка n кадров из .sft (ключи cuts, ids как в get_cuts) и сетка в PNG"
    )
    parser.add_argument(
        "sft_path",
        type=Path,
        help="Путь к .sft (safetensors) с тензором cuts [N,3,H,W]",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=16,
        metavar="N",
        help="Сколько кадров в выборке (по умолчанию 16, не больше числа кадров в файле)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Брать кадры подряд в порядке датасета, а не случайно",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        metavar="I",
        help="С какого индекса в cuts начинать (только вместе с --sequential, по умолчанию 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Сид для случайной выборки (с torch.Generator; с --sequential не используется)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("cuts_sample_grid.png"),
        help="Куда сохранить PNG-сетку (по умолчанию ./cuts_sample_grid.png)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Не открывать просмотрщик ОС после сохранения",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("-n / --count должен быть > 0")
    if not args.sequential and args.start != 0:
        raise SystemExit("--start задан, но нет --sequential: укажите --sequential или уберите --start")

    run_show_examples(
        sft_path=args.sft_path,
        n=args.count,
        seed=None if args.sequential else args.seed,
        sequential=args.sequential,
        start=args.start,
        output=args.output,
        open_viewer=not args.no_open,
    )


if __name__ == "__main__":
    main()
