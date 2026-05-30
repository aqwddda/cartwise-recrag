import httpx
from fastapi import FastAPI

from cartwise.core.config import Settings, get_settings

app = FastAPI(title="CartWise API")


def check_qdrant(settings: Settings) -> str:
    health_url = f"{settings.qdrant_url.rstrip('/')}/healthz"
    try:
        response = httpx.get(
            health_url,
            timeout=settings.healthcheck_timeout_seconds,
            trust_env=False,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return "unavailable"
    return "ok"


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "api": "ok",
        "qdrant": check_qdrant(settings),
        "recommender": "not_loaded",
        "llm": "configured" if settings.llm_is_configured else "not_configured",
    }
