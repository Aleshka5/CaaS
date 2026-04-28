"""
Загрузка PyTorch-модели из MLflow Model Registry (flavor pytorch).

Примеры model-uri:
  models:/my_model/1
  models:/my_model/Production
"""
from __future__ import annotations

import argparse
import os

import mlflow.pytorch
import torch.nn as nn


def main() -> None:
    p = argparse.ArgumentParser(description="Загрузить модель из MLflow (pytorch flavor)")
    p.add_argument(
        "--model-uri",
        required=True,
        help="MLflow models URI, напр. models:/name/1 или models:/name/Production",
    )
    p.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        help="MLflow tracking URI (или MLFLOW_TRACKING_URI)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="cpu или cuda / cuda:0",
    )
    p.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Вызвать model.eval() после загрузки (по умолчанию: да; отключить: --no-eval)",
    )
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    model: nn.Module = mlflow.pytorch.load_model(args.model_uri, map_location=args.device)
    model = model.to(args.device)
    if args.eval:
        model.eval()

    print(f"Загружено: {args.model_uri}")
    print(f"Класс: {type(model).__name__}, device: {args.device}")


if __name__ == "__main__":
    main()
