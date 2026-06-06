from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from time import perf_counter
from collections.abc import Callable
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
from cartwise.application.factory import (
    ApplicationServiceInitializationError,
    build_application_service as build_real_application_service,
)
from cartwise.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

APPLICATION_SERVICE_STATE_KEY = "application_service"
APPLICATION_SERVICE_STATUS_STATE_KEY = "application_service_status"
APPLICATION_SERVICE_ERROR_STATE_KEY = "application_service_initialization_error"

ApplicationServiceBuilder = Callable[[], Any]


def create_app(
    *,
    application_service: Any | None = None,
    resource_status: dict[str, str] | None = None,
    application_service_builder: ApplicationServiceBuilder | None = None,
    initialize_on_startup: bool = True,
) -> FastAPI:
    builder = application_service_builder or build_application_service

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        if application_service is None and initialize_on_startup:
            _initialize_application_service(api, builder)
        yield

    api = FastAPI(title="CartWise API", lifespan=lifespan)
    api.state.application_service = application_service
    api.state.application_service_status = (
        "ready" if application_service is not None else "not_initialized"
    )
    api.state.application_service_initialization_error = None
    api.state.resource_status = resource_status or _default_resource_status(
        application_service is not None
    )

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
        service_status = _get_application_service_status(api)
        is_ready = service is not None
        response = ReadyHealthResponse(
            status="ready" if is_ready else "not_ready",
            application_service=service_status,
            resources=dict(api.state.resource_status),
            initialization_error=_get_application_service_error(api),
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
        error = _get_application_service_error(request.app)
        detail = "recommendation application service is not initialized"
        if error is not None:
            detail = f"recommendation application service initialization failed: {error}"
        raise HTTPException(
            status_code=503,
            detail=detail,
        )
    return service


def build_application_service() -> Any:
    """Build the real service graph through the application composition root."""

    return build_real_application_service()


def _initialize_application_service(
    api: FastAPI,
    builder: ApplicationServiceBuilder,
) -> None:
    api.state.application_service_status = "initializing"
    try:
        api.state.application_service = builder()
    except ApplicationServiceInitializationError as error:
        _record_initialization_failure(api, error)
    except Exception as error:
        _record_initialization_failure(
            api,
            ApplicationServiceInitializationError(str(error)),
        )
    else:
        api.state.application_service_status = "ready"
        api.state.application_service_initialization_error = None
        api.state.resource_status = _ready_resource_status()


def _record_initialization_failure(
    api: FastAPI,
    error: ApplicationServiceInitializationError,
) -> None:
    logger.exception("Recommendation application service initialization failed")
    api.state.application_service = None
    api.state.application_service_status = "initialization_failed"
    api.state.application_service_initialization_error = str(error)
    api.state.resource_status = _failed_resource_status()


def _get_application_service(api: FastAPI) -> Any | None:
    return getattr(api.state, APPLICATION_SERVICE_STATE_KEY, None)


def _get_application_service_status(api: FastAPI) -> str:
    return getattr(
        api.state,
        APPLICATION_SERVICE_STATUS_STATE_KEY,
        "not_initialized",
    )


def _get_application_service_error(api: FastAPI) -> str | None:
    return getattr(api.state, APPLICATION_SERVICE_ERROR_STATE_KEY, None)


def _default_resource_status(service_is_ready: bool = False) -> dict[str, str]:
    settings = get_settings()
    if service_is_ready:
        return _ready_resource_status()
    return {
        "qdrant": "not_checked",
        "models": "not_checked",
        "indexes": "not_checked",
        "llm": "configured" if settings.llm_is_configured else "not_configured",
    }


def _ready_resource_status() -> dict[str, str]:
    return {
        "qdrant": "ready",
        "models": "ready",
        "indexes": "ready",
        "llm": "configured",
    }


def _failed_resource_status() -> dict[str, str]:
    return {
        "qdrant": "initialization_failed",
        "models": "initialization_failed",
        "indexes": "initialization_failed",
        "llm": "initialization_failed",
    }


app = create_app()
