from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class Settings(BaseSettings):
    model_config = _ENV

    gpu_support: bool = Field(default=False, validation_alias="GPU_SUPPORT")
    mlflow_tracking_uri: str = Field(
        default="http://127.0.0.1:5000",
        validation_alias="MLFLOW_TRACKING_URI",
    )
    movies_bucket_name: str = Field(validation_alias="MOVIES_BUCKET_NAME")
    mlflow_bucket_name: str = Field(validation_alias="MLFLOW_BUCKET_NAME")
    aws_access_key_id: str = Field(validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(validation_alias="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = Field(validation_alias="AWS_DEFAULT_REGION")
    s3_endpoint_url: str = Field(validation_alias="S3_ENDPOINT_URL")

    @computed_field
    def mlflow_s3_endpoint_url(self) -> str:
        return self.s3_endpoint_url


class VideoSettings(BaseSettings):
    model_config = _ENV

    target_width: int = Field(default=1120, validation_alias="TARGET_WIDTH")
    target_height: int = Field(default=480, validation_alias="TARGET_HEIGHT")
    target_fps: int = Field(default=30, validation_alias="TARGET_FPS")
    # Вход классификатора (.env MODEL_INPUT_*); в make_markup кадры читаются сразу в этом размере.
    model_input_height: int = Field(default=48, validation_alias="MODEL_INPUT_HEIGHT")
    model_input_width: int = Field(default=112, validation_alias="MODEL_INPUT_WIDTH")
    inference_pair_batch_size: int = Field(
        default=256,
        validation_alias="INFERENCE_PAIR_BATCH_SIZE",
        ge=1,
    )
