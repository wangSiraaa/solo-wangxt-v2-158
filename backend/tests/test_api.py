"""API + persistence tests (SQLite in-memory; no PostgreSQL required)."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture()
def client(monkeypatch):
    # Point the ORM at an isolated in-memory SQLite before importing app.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import app.db as db
    eng = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool, future=True)
    db.Base.metadata.create_all(eng)
    TestSession = sessionmaker(bind=eng, future=True)

    from app.main import app, get_session

    def _override():
        s = TestSession()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_session] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["format"] == "PRT-1"
    assert body["control_authority"] == "analysis_only"


def test_datasets_listed(client):
    names = [d["name"] for d in client.get("/api/datasets").json()]
    assert "prt1_scan_rot1p7_240ppi.png" in names


def test_analyze_dataset_and_persist(client):
    r = client.post("/api/analyze",
                    data={"dataset": "prt1_scan_rot1p7_240ppi.png",
                          "declared_ppi": "240"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "candidates")
    assert abs(body["image"]["measured_ppi"] - 240) < 5
    assert "cyan" in body["plates"]
    # Ranges are present and ordered.
    for p in body["plates"].values():
        if p["x_range_um"][0] is not None:
            assert p["x_range_um"][0] <= p["x_range_um"][1]

    run_id = body["run_id"]
    again = client.get(f"/api/runs/{run_id}").json()
    assert again["filename"].endswith(".png")
    assert set(again["plates"]) == {"cyan", "magenta", "yellow", "black"}


def test_overlay_png(client):
    body = client.post("/api/analyze",
                       data={"dataset": "prt1_reference_300ppi.png",
                             "declared_ppi": "300"}).json()
    r = client.get(f"/api/runs/{body['run_id']}/overlay.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 1000


def test_human_confirmation_flow(client):
    body = client.post("/api/analyze",
                       data={"dataset": "prt1_reference_300ppi.png",
                             "declared_ppi": "300"}).json()
    rid = body["run_id"]
    r = client.post(f"/api/runs/{rid}/confirm",
                    json={"operator_id": "qc-07", "decision": "confirmed",
                          "reason": "与离线复核一致"})
    assert r.status_code == 200
    confirms = r.json()["confirmations"]
    assert confirms[-1]["operator_id"] == "qc-07"
    assert confirms[-1]["decision"] == "confirmed"

    # Corrected decision requires values.
    bad = client.post(f"/api/runs/{rid}/confirm",
                      json={"operator_id": "qc-07",
                            "decision": "corrected"})
    assert bad.status_code == 422


def test_offline_review_is_advisory(client):
    body = client.post("/api/analyze",
                       data={"dataset": "prt1_reference_300ppi.png",
                             "declared_ppi": "300"}).json()
    r = client.post(f"/api/runs/{body['run_id']}/review")
    assert r.status_code == 200
    rev = r.json()
    assert rev["mode"] == "offline_advisory_only"
    assert rev["control_authority"].startswith("none")
    # Detector and simple bbox method agree on the clean reference.
    assert rev["verdict"] == "consistent"


def test_unknown_dataset_404(client):
    r = client.post("/api/analyze", data={"dataset": "nope.png"})
    assert r.status_code == 404


def test_garbage_upload_rejected(client):
    r = client.post("/api/analyze",
                    files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 415
