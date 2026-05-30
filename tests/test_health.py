from fastapi.testclient import TestClient

from cartwise.api import main

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
