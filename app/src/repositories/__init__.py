from abc import ABC, abstractmethod
from typing import Any


class BaseModelRegistry(ABC):
    @abstractmethod
    def get_model(self, model_id: str) -> Any:
        pass


class BaseDataRepository(ABC):
    @abstractmethod
    async def get_movie(self, path: str) -> None:
        pass

    @abstractmethod
    async def upload_movie(self, path: str) -> None:
        pass
