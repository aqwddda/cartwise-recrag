from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str = "http://127.0.0.1:6333"
    healthcheck_timeout_seconds: float = 1.0

    deepseek_api_key: str | None = None
    google_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def llm_is_configured(self) -> bool:
        return any(
            (
                self.deepseek_api_key,
                self.google_api_key,
            )
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
