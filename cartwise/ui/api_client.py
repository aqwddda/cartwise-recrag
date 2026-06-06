"""HTTP client used by the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 90.0


@dataclass(frozen=True)
class ApiError:
    """UI-safe API error shape."""

    message: str
    status_code: int | None = None
    error_type: str = "api_error"
    detail: Any | None = None


@dataclass(frozen=True)
class ApiResult:
    """Structured result returned by the UI API client."""

    ok: bool
    data: dict[str, Any] | None = None
    error: ApiError | None = None
    status_code: int | None = None
    elapsed_ms: int | None = None


class CartWiseApiClient:
    """Small synchronous HTTP client for the Streamlit frontend."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_BASE_URL,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    def check_ready(self) -> ApiResult:
        """Call the backend readiness endpoint."""

        return self._request("GET", "/health/ready")

    def recommend(
        self,
        query: str,
        *,
        user_id: str | None = None,
        top_k: int = 5,
    ) -> ApiResult:
        """Request a single recommendation turn from the backend."""

        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
        }
        if user_id:
            payload["user_id"] = user_id
        return self._request("POST", "/api/v1/recommend", json=payload)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> ApiResult:
        url = f"{self.base_url}{path}"
        started = perf_counter()
        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                trust_env=False,
                transport=self._transport,
            ) as client:
                response = client.request(method, url, json=json)
        except httpx.TimeoutException as error:
            return _error_result(
                ApiError(
                    message="The backend request timed out. Try again or check the backend logs.",
                    error_type="timeout",
                    detail=str(error),
                ),
                started,
            )
        except httpx.ConnectError as error:
            return _error_result(
                ApiError(
                    message="Could not connect to the CartWise backend. Start FastAPI and try again.",
                    error_type="connection_error",
                    detail=str(error),
                ),
                started,
            )
        except httpx.HTTPError as error:
            return _error_result(
                ApiError(
                    message="The backend request failed before a response was returned.",
                    error_type="http_error",
                    detail=str(error),
                ),
                started,
            )

        elapsed_ms = _elapsed_ms(started)
        payload = _response_json(response)
        if 200 <= response.status_code < 300:
            return ApiResult(
                ok=True,
                data=payload if isinstance(payload, dict) else {},
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
            )
        return ApiResult(
            ok=False,
            error=_api_error_from_response(response.status_code, payload),
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
        )


def check_ready(
    base_url: str = DEFAULT_API_BASE_URL,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> ApiResult:
    """Convenience wrapper for readiness checks."""

    return CartWiseApiClient(
        base_url,
        timeout_seconds=timeout_seconds,
    ).check_ready()


def recommend(
    query: str,
    *,
    user_id: str | None = None,
    top_k: int = 5,
    base_url: str = DEFAULT_API_BASE_URL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> ApiResult:
    """Convenience wrapper for recommendation requests."""

    return CartWiseApiClient(
        base_url,
        timeout_seconds=timeout_seconds,
    ).recommend(query, user_id=user_id, top_k=top_k)


def _api_error_from_response(status_code: int, payload: Any) -> ApiError:
    detail = _extract_detail(payload)
    if status_code == 422:
        return ApiError(
            message="Request validation failed. Check the query and top_k values.",
            status_code=status_code,
            error_type="validation_error",
            detail=detail,
        )
    if status_code == 503:
        return ApiError(
            message="Backend is not ready. Check FastAPI, Qdrant, models, indexes, and LLM configuration.",
            status_code=status_code,
            error_type="backend_not_ready",
            detail=detail,
        )
    if status_code >= 500:
        return ApiError(
            message="The backend returned an internal error. Check the FastAPI logs.",
            status_code=status_code,
            error_type="backend_error",
            detail=detail,
        )
    return ApiError(
        message=f"The backend returned HTTP {status_code}.",
        status_code=status_code,
        error_type="api_error",
        detail=detail,
    )


def _extract_detail(payload: Any) -> Any:
    if isinstance(payload, dict) and "detail" in payload:
        return payload["detail"]
    return payload


def _response_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"detail": response.text}


def _error_result(error: ApiError, started: float) -> ApiResult:
    return ApiResult(
        ok=False,
        error=error,
        status_code=error.status_code,
        elapsed_ms=_elapsed_ms(started),
    )


def _elapsed_ms(started: float) -> int:
    return max(0, round((perf_counter() - started) * 1000))
