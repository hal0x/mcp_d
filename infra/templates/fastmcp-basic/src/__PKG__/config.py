from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", env_file_encoding="utf-8")

    # Example settings
    MODE: str = "dev"


def get_settings() -> Settings:
    return Settings()  # loads from env/.env

