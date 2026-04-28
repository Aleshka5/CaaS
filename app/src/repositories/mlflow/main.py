from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import torch
from mlflow.exceptions import MlflowException, RestException
from mlflow.tracking import MlflowClient as MlflowTrackingClient

from app.src.repositories import BaseModelRegistry

_log = logging.getLogger(__name__)


class MLFlowClient(BaseModelRegistry):
    """
    Загрузка модели из MLflow Model Registry.

    Поддерживаются:
    - стандартный артефакт `mlflow.pytorch.log_model` (`mlflow.pytorch.load_model`);
    - кастомная раскладка с `log_artifact`: `model/model_best_scripted.pt`, `model/model_best.pt`,
      `config/training_config.json` — для inference используется TorchScript.
    """

    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def _configure(self) -> None:
        mlflow.set_tracking_uri(self.tracking_uri)

    def resolve_model_uri(self, model_ref: str) -> str:
        """
        Преобразует ссылку на модель в URI для load_model / download_artifacts.

        - ``runs:/...`` — без изменений.
        - ``models:/name/1``, ``models:/name/Production``, ``models:/name@alias`` — без изменений.
        - ``models:/name`` (только имя) или ``name`` — ищет версии в Model Registry и
          подставляет ``models:/name/<version>`` (приоритет стадии: Production → Staging → None).
        """
        model_ref = model_ref.strip()
        if model_ref.startswith("runs:/"):
            return model_ref

        if model_ref.startswith("models:/"):
            rest = model_ref.removeprefix("models:/").strip()
            if not rest:
                raise ValueError("Пустой URI после models:/")
            # Версия, стадия (Production), алиас (name@champion) — не подменяем
            if "/" in rest or "@" in rest:
                return model_ref
            registered_name = rest
        else:
            registered_name = model_ref

        return self._registry_uri_with_version(registered_name)

    def _registry_uri_with_version(self, registered_name: str) -> str:
        self._configure()
        client = MlflowTrackingClient(self.tracking_uri)
        safe = registered_name.replace("'", "\\'")
        filter_string = f"name='{safe}'"
        try:
            versions = list(
                client.search_model_versions(filter_string=filter_string, max_results=200)
            )
        except RestException as e:
            raise FileNotFoundError(
                f"Не удалось найти модель {registered_name!r} в Model Registry "
                f"(tracking_uri={self.tracking_uri!r})."
            ) from e

        if not versions:
            try:
                client.get_registered_model(registered_name)
            except RestException as e:
                raise FileNotFoundError(
                    f"Модель {registered_name!r} не зарегистрирована в Model Registry."
                ) from e
            raise FileNotFoundError(
                f"У модели {registered_name!r} нет ни одной версии в Model Registry."
            )

        def _stage_key(v: Any) -> tuple[int, int]:
            stage = (v.current_stage or "None").strip()
            order = {"Production": 0, "Staging": 1, "None": 2, "Archived": 3}.get(stage, 9)
            return (order, -int(v.version))

        best = min(versions, key=_stage_key)
        uri = f"models:/{registered_name}/{best.version}"
        _log.info("Model Registry: %s -> %s (stage=%s)", registered_name, uri, best.current_stage)
        return uri

    @staticmethod
    def _find_file(root: Path, filename: str) -> Path | None:
        for path in (root / filename, root / "model" / filename):
            if path.is_file():
                return path
        for path in root.rglob(filename):
            if path.is_file():
                return path
        return None

    def get_model(
        self,
        model_id: str,
        *,
        map_location: str | torch.device = "cpu",
        try_mlflow_pytorch_flavor: bool = True,
    ) -> Any:
        self._configure()
        uri = self.resolve_model_uri(model_id)

        if try_mlflow_pytorch_flavor:
            try:
                model = mlflow.pytorch.load_model(uri, map_location=map_location)
                model.eval()
                return model
            except (MlflowException, OSError, RuntimeError, ValueError):
                pass

        local_dir = mlflow.artifacts.download_artifacts(artifact_uri=uri)
        root = Path(local_dir)

        scripted_path = self._find_file(root, "model_best_scripted.pt")
        if scripted_path is not None:
            model = torch.jit.load(str(scripted_path), map_location=map_location)
            model.eval()
            return model

        if self._find_file(root, "model_best.pt") is not None:
            raise RuntimeError(
                "В артефактах есть только model_best.pt (state_dict). Для get_model нужен "
                "model_best_scripted.pt или модель, записанная через mlflow.pytorch.log_model."
            )

        raise FileNotFoundError(
            f"После загрузки {uri!r} не найдены model_best_scripted.pt и model_best.pt."
        )
