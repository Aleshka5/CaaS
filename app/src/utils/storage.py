from pathlib import Path

import aiofiles

from app.src.repositories import BaseDataRepository


class Storage:
    """Локальный кэш + S3: ключ объекта в бакете совпадает с переданным `file_name`."""

    def __init__(self, s3_client: BaseDataRepository) -> None:
        self.s3_client = s3_client

    async def get_movie(self, file_name: str) -> Path:
        """
        Вернуть путь к локальному файлу; если файла нет — скачать из S3 с ключом `file_name`.
        """
        path = Path(file_name)
        if path.is_file():
            return path
        await self.s3_client.get_movie(file_name)
        return path

    async def upload_movie(self, file_name: str, file_content: bytes) -> Path:
        """
        Сохранить байты на диск и загрузить в S3 под тем же ключом, что и `file_name`.
        Если путь уже существует — FileExistsError.
        """
        path = Path(file_name)
        if path.exists():
            raise FileExistsError(str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "wb") as f:
            await f.write(file_content)
        await self.s3_client.upload_movie(file_name)
        return path

    async def delete_movie(self, file_name: str) -> None:
        """
        Удалить только локальный файл. Объект в S3 не трогаем.
        Если файла нет — FileNotFoundError.
        """
        path = Path(file_name)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        path.unlink()
