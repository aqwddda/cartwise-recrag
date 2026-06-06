from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request

from cartwise.api.schemas import (
    LiveHealthResponse,
    ReadyHealthResponse,
    RecommendationResponse,
    RecommendRequest,
    recommendation_response_from_result,
)
from cartwise.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

APPLICATION_SERVICE_STATE_KEY = "application_service"


def create_app(
    *,
    application_service: Any | None = None,
    resource_status: dict[str, str] | None = None,
) -> FastAPI:
    api = FastAPI(title="CartWise API")
    api.state.application_service = application_service
    api.state.resource_status = resource_status or _default_resource_status()

    @api.get("/health", response_model=dict[str, str])
    def health() -> dict[str, str]:
        settings = get_settings()
        return {
            "api": "ok",
            "qdrant": check_qdrant(settings),
            "recommender": (
                "loaded"
                if _get_application_service(api) is not None
                else "not_loaded"
            ),
            "llm": "configured" if settings.llm_is_configured else "not_configured",
        }

    @api.get("/health/live", response_model=LiveHealthResponse)
    def live() -> LiveHealthResponse:
        return LiveHealthResponse(status="ok")

    @api.get("/health/ready", response_model=ReadyHealthResponse)
    def ready() -> ReadyHealthResponse:
        service = _get_application_service(api)
        is_ready = service is not None
        response = ReadyHealthResponse(
            status="ready" if is_ready else "not_ready",
            application_service="ready" if is_ready else "not_initialized",
            resources=dict(api.state.resource_status),
        )
        if is_ready:
            return response
        raise HTTPException(status_code=503, detail=response.model_dump())

    @api.post("/api/v1/recommend", response_model=RecommendationResponse)
    def recommend(
        payload: RecommendRequest,
        service: Any = Depends(get_application_service),
    ) -> RecommendationResponse:
        started = perf_counter()
        try:
            result = service.recommend(payload.to_application_request())
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except (ConnectionError, TimeoutError, RuntimeError) as error:
            logger.exception("Recommendation service is unavailable")
            raise HTTPException(
                status_code=503,
                detail="recommendation service is unavailable",
            ) from error
        except Exception as error:
            logger.exception("Unexpected recommendation API failure")
            raise HTTPException(
                status_code=500,
                detail="internal recommendation error",
            ) from error
        latency_ms = max(0, round((perf_counter() - started) * 1000))
        return recommendation_response_from_result(result, latency_ms=latency_ms)

    return api


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


def get_application_service(request: Request) -> Any:
    service = _get_application_service(request.app)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="recommendation application service is not initialized",
        )
    return service


def build_application_service() -> Any:
    """Composition root placeholder for the real heavy service graph.

    The route layer intentionally does not build retrieval, Qdrant, or LLM
    resources. Stage 9 keeps that construction centralized here for a later
    minimal migration from already verified scripts.
    """

    raise NotImplementedError(
        "Real RecommendationApplicationService construction is not implemented yet"
    )


def _get_application_service(api: FastAPI) -> Any | None:
    return getattr(api.state, APPLICATION_SERVICE_STATE_KEY, None)


def _default_resource_status() -> dict[str, str]:
    settings = get_settings()
    return {
        "qdrant": "not_checked",
        "models": "not_checked",
        "indexes": "not_checked",
        "llm": "configured" if settings.llm_is_configured else "not_configured",
    }


app = create_app()
