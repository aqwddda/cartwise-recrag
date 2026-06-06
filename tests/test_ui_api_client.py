from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from cartwise.ui.api_client import CartWiseApiClient


def test_check_ready_success() -> None:
    client = _client(lambda request: httpx.Response(200, json={"status": "ready"}))

    result = client.check_ready()

    assert result.ok is True
    assert result.status_code == 200
    assert result.data == {"status": "ready"}


def test_check_ready_returns_structured_error_for_503() -> None:
    client = _client(
        lambda request: httpx.Response(
            503,
            json={
                "detail": {
                    "status": "not_ready",
                    "initialization_error": "qdrant unavailable",
                }
            },
        )
    )

    result = client.check_ready()

    assert result.ok is False
    assert result.status_code == 503
    assert result.error is not None
    assert result.error.error_type == "backend_not_ready"
    assert result.error.detail["initialization_error"] == "qdrant unavailable"


def test_check_ready_handles_connection_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)

    result = client.check_ready()

    assert result.ok is False
    assert result.error is not None
    assert result.error.error_type == "connection_error"
    assert "connect" in result.error.message.lower()


def test_recommend_success() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(_json_payload(request))
        return httpx.Response(
            200,
            json={
                "query": "guitar tuner",
                "search_query": "guitar tuner",
                "known_user": True,
                "applied_constraints": {},
                "results": [{"rank": 1, "title": "Clip-on tuner"}],
                "diagnostics": [],
                "latency_ms": 1234,
            },
        )

    client = _client(handler)

    result = client.recommend("guitar tuner", user_id="user-1", top_k=3)

    assert result.ok is True
    assert seen_payload == {
        "query": "guitar tuner",
        "user_id": "user-1",
        "top_k": 3,
    }
    assert result.data is not None
    assert result.data["results"][0]["title"] == "Clip-on tuner"


def test_recommend_returns_structured_error_for_422() -> None:
    client = _client(
        lambda request: httpx.Response(
            422,
            json={"detail": [{"loc": ["body", "top_k"], "msg": "less than 50"}]},
        )
    )

    result = client.recommend("guitar tuner", top_k=99)

    assert result.ok is False
    assert result.error is not None
    assert result.error.error_type == "validation_error"
    assert "query and top_k" in result.error.message


def test_recommend_returns_structured_error_for_503() -> None:
    client = _client(
        lambda request: httpx.Response(
            503,
            json={"detail": "recommendation service is unavailable"},
        )
    )

    result = client.recommend("guitar tuner")

    assert result.ok is False
    assert result.error is not None
    assert result.error.error_type == "backend_not_ready"
    assert result.error.detail == "recommendation service is unavailable"


def test_recommend_returns_structured_error_for_500() -> None:
    client = _client(
        lambda request: httpx.Response(
            500,
            json={"detail": "internal recommendation error"},
        )
    )

    result = client.recommend("guitar tuner")

    assert result.ok is False
    assert result.error is not None
    assert result.error.error_type == "backend_error"
    assert "internal error" in result.error.message


def test_recommend_handles_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow backend", request=request)

    client = _client(handler)

    result = client.recommend("guitar tuner")

    assert result.ok is False
    assert result.error is not None
    assert result.error.error_type == "timeout"
    assert "timed out" in result.error.message


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> CartWiseApiClient:
    return CartWiseApiClient(
        "http://api.test",
        transport=httpx.MockTransport(handler),
    )


def _json_payload(request: httpx.Request) -> dict[str, object]:
    return json.loads(request.content.decode("utf-8"))
