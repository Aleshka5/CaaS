import asyncio
from pathlib import Path

import aioboto3
import aiofiles

from app.src.repositories import BaseDataRepository


class S3Client(BaseDataRepository):
    """S3: один и тот же строковый идентификатор — ключ объекта и путь к файлу на диске."""

    def __init__(
        self,
        bucket_name: str,
        *,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        chunk_size: int = 8 * 1024 * 1024,
    ) -> None:
        self._bucket = bucket_name
        self._chunk_size = chunk_size
        self._session = aioboto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        self._endpoint_url = endpoint_url

    def _client(self):
        kwargs: dict[str, str] = {}
        if self._endpoint_url:
            kwargs["endpoint_url"] = self._endpoint_url
        return self._session.client("s3", **kwargs)

    async def get_movie(self, path: str) -> None:
        """Скачать объект с ключом `path` в локальный файл по тому же пути `path`."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with self._client() as s3:
            response = await s3.get_object(Bucket=self._bucket, Key=path)
            body = response["Body"]
            async with aiofiles.open(dest, "wb") as f:
                while True:
                    chunk = await body.read(self._chunk_size)
                    if not chunk:
                        break
                    await f.write(chunk)

    async def upload_movie(self, path: str) -> None:
        """Загрузить локальный файл `path` в объект с ключом `path`."""
        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(str(src))
        async with self._client() as s3:
            await s3.upload_file(str(src), self._bucket, path)


if __name__ == "__main__":
    from app.config import Settings
    from app.src.repositories.s3.main import S3Client

    movie = "test_movie"
    s = Settings()
    client = S3Client(
        s.movies_bucket_name,
        aws_access_key_id=s.aws_access_key_id,
        aws_secret_access_key=s.aws_secret_access_key,
        region_name=s.aws_default_region,
        endpoint_url=s.s3_endpoint_url,
    )
    key = f"{movie}/source.mp4"
    asyncio.run(client.get_movie(key))
    asyncio.run(client.upload_movie(key))
