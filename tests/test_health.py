from fastapi.testclient import TestClient

from cartwise.api import main
from cartwise.core.config import Settings

client = TestClient(main.app)


def test_health_returns_unavailable_when_qdrant_is_down(monkeypatch) -> None:
    monkeypatch.setattr(main, "check_qdrant", lambda settings: "unavailable")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["api"] == "ok"
    assert response.json()["qdrant"] == "unavailable"


def test_health_returns_ok_when_qdrant_is_up(monkeypatch) -> None:
    monkeypatch.setattr(main, "check_qdrant", lambda settings: "ok")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["qdrant"] == "ok"


def test_qdrant_health_check_bypasses_environment_proxy(monkeypatch) -> None:
    request_options = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    def fake_get(url, **kwargs):
        request_options.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(main.httpx, "get", fake_get)

    assert main.check_qdrant(Settings()) == "ok"
    assert request_options["trust_env"] is False
