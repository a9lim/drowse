from pathlib import Path
import concurrent.futures
import threading
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from drowse.core.surface_topology import SurfaceEvidence
from drowse.server import create_app
from tests.test_drowse_api import _mock_session


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    return TestClient(create_app(_mock_session()))


def sphere():
    i = np.arange(96)
    z = 1 - 2 * (i + 0.5) / 96
    angle = i * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(angle), r * np.sin(angle), z], axis=1).tolist()


def test_inspect_real_surface(client: TestClient):
    pytest.importorskip("ripser")
    response = client.post("/drowse/v1/manifolds/surface-evidence", json={"points": sphere()})
    assert response.status_code == 200
    assert response.json()["candidate"] == "sphere"
    assert response.json()["sample_count"] == 96
    assert "no recovered coordinate chart" in response.json()["reason"]


@pytest.mark.parametrize("points", [[], [[0, 1]] * 385, [[0, 1]] * 32, [[0, 1]] * 31 + [[1, 2, 3]]])
def test_invalid_cloud(client: TestClient, points: list[list[float]]):
    assert client.post("/drowse/v1/manifolds/surface-evidence", json={"points": points}).status_code in (400, 422)


def test_missing_dependency(client: TestClient):
    with patch("drowse.core.surface_topology.detect_surface", side_effect=ImportError):
        response = client.post("/drowse/v1/manifolds/surface-evidence", json={"points": sphere()})
    assert response.status_code == 503
    assert "drowse[topology]" in response.json()["detail"]


def test_busy_inspection_and_lock_release(client: TestClient):
    entered, release = threading.Event(), threading.Event()

    def inspect(points: list[list[float]]):
        entered.set()
        assert release.wait(10)
        return SurfaceEvidence(None, "Unresolved", sample_count=len(points))

    with patch("drowse.core.surface_topology.detect_surface", side_effect=inspect), concurrent.futures.ThreadPoolExecutor() as pool:
        first = pool.submit(client.post, "/drowse/v1/manifolds/surface-evidence", json={"points": sphere()})
        try:
            assert entered.wait(10)
            assert client.post("/drowse/v1/manifolds/surface-evidence", json={"points": sphere()}).status_code == 409
        finally:
            release.set()
        assert first.result().status_code == 200
        assert client.post("/drowse/v1/manifolds/surface-evidence", json={"points": sphere()}).status_code == 200


@pytest.mark.parametrize("domain", [{"type": "klein"}, {"type": "projective", "dim": 2}])
def test_create_quotient_through_api(client: TestClient, domain: dict[str, str | int]):
    payload = {
        "namespace": "local", "name": "surface", "domain": domain,
        "nodes": [{"label": f"node_{i}", "coords": [i * 0.43, i * 1.1], "statements": [f"Example {i}."]} for i in range(8)],
    }
    response = client.post("/drowse/v1/manifolds", json=payload)
    assert response.status_code == 201, response.text
    assert response.json()["domain"] == domain
    assert response.json()["intrinsic_dim"] == 2
    assert "(2D)" in response.json()["domain_label"]
    restored = client.get("/drowse/v1/manifolds/local/surface")
    assert restored.status_code == 200
    assert restored.json()["domain"] == domain
