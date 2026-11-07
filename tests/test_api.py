import os

import pytest

# These tests need the real model (mlflow.db) and data, so they only run on your Codespace.
pytestmark = pytest.mark.skipif(not os.path.exists("mlflow.db"),
                                reason="needs the local MLflow model and data")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_next_shift_has_every_borough(client):
    r = client.get("/forecast/next")
    assert r.status_code == 200
    assert len(r.json()["forecasts"]) == 5


def test_rejects_unknown_borough(client):
    r = client.get("/forecast", params={"borough": "NARNIA", "day": "2026-07-02", "shift": "08:00"})
    assert r.status_code == 422                  # FastAPI rejects values not in the dropdown list