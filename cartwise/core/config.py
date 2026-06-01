from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str = "http://127.0.0.1:6333"
    healthcheck_timeout_seconds: float = 1.0

    openai_compatible_api_key: str | None = None
    openai_compatible_base_url: str | None = None
    openai_compatible_model: str | None = None
    deepseek_api_key: str | None = None
    google_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    google_base_url: str = "https://ai.google.dev"
    google_model: str = "gemini-2.5-flash"
    llm_timeout_seconds: float = 20.0

    http_proxy: str | None = None
    https_proxy: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def llm_is_configured(self) -> bool:
        return self.llm_api_key is not None

    @property
    def llm_api_key(self) -> str | None:
        return (
            self.openai_compatible_api_key
            or self.deepseek_api_key
            or self.google_api_key
        )

    @property
    def llm_base_url(self) -> str:
        if self.openai_compatible_api_key is not None:
            return self.openai_compatible_base_url or self.deepseek_base_url
        if self.deepseek_api_key is not None:
            return self.deepseek_base_url
        return self.google_base_url

    @property
    def llm_model(self) -> str:
        if self.openai_compatible_api_key is not None:
            return self.openai_compatible_model or self.deepseek_model
        if self.deepseek_api_key is not None:
            return self.deepseek_model
        return self.google_model

    @property
    def external_https_proxy(self) -> str | None:
        return self.https_proxy or self.http_proxy


@lru_cache
def get_settings() -> Settings:
    return Settings()
