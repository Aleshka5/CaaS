"""
Регистрация PyTorch-модели из .pt/.pth в MLflow Model Registry.

Рекомендуемый формат чекпоинта (torch.save):
  {"model_state_dict": ..., "epoch": ..., ...}

Если сохранён целиком nn.Module — класс указывать не нужно.
Если в файле только state_dict или словарь с ключом state_dict — укажите --model-class.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn


def _import_class(module_path: str, class_name: str) -> type:
    mod = importlib.import_module(module_path)
    try:
        return getattr(mod, class_name)
    except AttributeError as e:
        raise ValueError(f"В модуле {module_path!r} нет класса {class_name!r}") from e


def _maybe_strip_dataparallel_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if not state:
        return state
    keys = list(state.keys())
    if not all(k.startswith("module.") for k in keys):
        return state
    return {k[len("module.") :]: v for k, v in state.items()}


def load_checkpoint(path: Path, weights_only_prefer: bool) -> Any:
    """
    Сначала пробуем weights_only=True (безопаснее). Если чекпоинт содержит
    произвольные объекты Python — повторяем с weights_only=False (доверяйте источнику файла).
    """
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if weights_only_prefer:
        try:
            return torch.load(path, **kwargs, weights_only=True)
        except Exception:
            pass
    return torch.load(path, **kwargs, weights_only=False)


def _pick_state_dict(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            v = obj.get(key)
            if isinstance(v, dict) and v and all(isinstance(k, str) for k in v.keys()):
                first = next(iter(v.values()))
                if torch.is_tensor(first):
                    return v
    if isinstance(obj, dict) and obj and all(isinstance(k, str) for k in obj.keys()):
        first = next(iter(obj.values()))
        if torch.is_tensor(first):
            return obj
    return None


def resolve_module(
    checkpoint: Any,
    model_class: type[nn.Module] | None,
    init_kwargs: dict[str, Any],
    strip_dp_prefix: bool,
    strict: bool,
) -> nn.Module:
    if isinstance(checkpoint, nn.Module):
        return checkpoint

    state = _pick_state_dict(checkpoint)
    if state is None:
        raise ValueError(
            "Не удалось извлечь state_dict из чекпоинта и это не nn.Module. "
            "Укажите --model-class MODULE:Class."
        )
    if model_class is None:
        raise ValueError(
            "В чекпоинте нет готового nn.Module. Задайте --model-class path.to.module:ModelClass "
            "и при необходимости --model-init-kwargs."
        )
    if strip_dp_prefix:
        state = _maybe_strip_dataparallel_prefix(state)
    model = model_class(**init_kwargs)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if not strict and (missing or unexpected):
        pass  # ожидаемо при strict=False
    return model


def parse_model_class(spec: str | None) -> type[nn.Module] | None:
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("--model-class должен быть вида package.module:ClassName")
    mod_path, _, cls_name = spec.partition(":")
    cls = _import_class(mod_path.strip(), cls_name.strip())
    if not issubclass(cls, nn.Module):
        raise ValueError(f"{spec} не является подклассом torch.nn.Module")
    return cls


def main() -> None:
    p = argparse.ArgumentParser(description="Залить .pt модель в MLflow Model Registry")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model_best.pt"),
        help="Путь к model_best.pt / checkpoint",
    )
    p.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        help="MLflow tracking URI (или MLFLOW_TRACKING_URI)",
    )
    p.add_argument(
        "--experiment-name",
        default="pytorch_models",
        help="Имя experiment в MLflow",
    )
    p.add_argument("--run-name", default=None, help="Имя run (опционально)")
    p.add_argument(
        "--registered-model-name",
        required=True,
        help="Имя модели в Model Registry",
    )
    p.add_argument(
        "--model-class",
        default=None,
        help="Если в файле state_dict: module.path:ClassName",
    )
    p.add_argument(
        "--model-init-kwargs",
        default="{}",
        help='JSON с аргументами конструктора модели, напр. {"num_classes": 2}',
    )
    p.add_argument(
        "--strip-dataparallel-prefix",
        action="store_true",
        help="Убрать префикс module. из ключей state_dict (обучение через DataParallel)",
    )
    p.add_argument(
        "--no-strict-state-dict",
        action="store_true",
        help="load_state_dict(strict=False)",
    )
    p.add_argument(
        "--no-weights-only-first",
        action="store_true",
        help="Сразу torch.load(..., weights_only=False) без попытки weights_only=True",
    )
    p.add_argument(
        "--artifact-path",
        default="model",
        help="Путь артефакта внутри run (каталог для mlflow.pytorch.log_model)",
    )
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Файл не найден: {args.checkpoint.resolve()}")

    init_kwargs = json.loads(args.model_init_kwargs)
    if not isinstance(init_kwargs, dict):
        raise SystemExit("--model-init-kwargs должен быть JSON-объектом")

    model_cls = parse_model_class(args.model_class)
    raw = load_checkpoint(args.checkpoint, weights_only_prefer=not args.no_weights_only_first)
    model = resolve_module(
        raw,
        model_cls,
        init_kwargs,
        strip_dp_prefix=args.strip_dataparallel_prefix,
        strict=not args.no_strict_state_dict,
    )
    model.eval()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    pip_req = ["torch", "cloudpickle"]

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_param("checkpoint_path", str(args.checkpoint.resolve()))
        if isinstance(raw, dict):
            for meta_key in ("epoch", "best_metric", "config"):
                if meta_key in raw and isinstance(raw[meta_key], (str, int, float, bool)):
                    mlflow.log_param(f"checkpoint_{meta_key}", raw[meta_key])

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=args.artifact_path,
            registered_model_name=args.registered_model_name,
            pip_requirements=pip_req,
        )
        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/{args.artifact_path}"
    print(f"Зарегистрировано: {args.registered_model_name}")
    print(f"Model URI (run): {model_uri}")


if __name__ == "__main__":
    main()
