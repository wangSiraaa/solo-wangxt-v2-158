import io
import os

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Confirmation, ModelSuggestion, PlateResult, Scan, init_db, engine

client = TestClient(app)
FIX = os.path.join(os.path.dirname(__file__), "..", "data", "fixtures")


@pytest.fixture(autouse=True)
def _clean_db():
    init_db()
    yield
    for tbl in reversed([Confirmation, ModelSuggestion, PlateResult, Scan]):
        with engine.begin() as c:
            c.execute(tbl.__table__.delete())


def _upload(name="01_baseline", declared_dpi=None):
    path = os.path.join(FIX, name + ".png")
    with open(path, "rb") as f:
        files = {"image": ("t.png", f, "image/png")}
        data = {"name": name}
        if declared_dpi is not None:
            data["declared_dpi"] = declared_dpi
            data["dpi_source"] = "user"
        r = client.post("/api/scans/analyze", files=files, data=data)
    return r


def test_analyze_upload_and_persist():
    r = _upload()
    assert r.status_code == 200
    body = r.json()
    assert body["scan_id"] > 0
    res = body["result"]
    assert res["calibration"]["usable"] is True
    assert set(res["plates"]) == {"C", "M", "Y", "K"}
    assert res["plates"]["K"]["status"] == "reference"


def test_blank_image_is_rejected_not_misregistered():
    import numpy as np
    import cv2
    blank = np.full((200, 200, 3), 255, np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    r = client.post("/api/scans/analyze",
                    files={"image": ("b.png", io.BytesIO(buf.tobytes()), "image/png")},
                    data={"name": "blank"})
    assert r.status_code == 200
    assert r.json()["result"]["status"] == "calibration_failed"


def test_missing_plate_stored_and_suggestion_needs_manual():
    r = _upload("07_missing_c")
    sid = r.json()["scan_id"]
    # 缺 C 版：DB 里 C 状态 missing，offset 为 NULL
    from app.models import SessionLocal
    db = SessionLocal()
    c_res = next(p for p in db.query(PlateResult).filter_by(scan_id=sid) if p.plate == "C")
    assert c_res.status == "missing"
    assert c_res.offset_x_um is None
    # 离线模型建议必须是 needs_manual，不给具体微米值
    sug = client.get(f"/api/scans/{sid}/model-suggestions").json()
    c_sug = next(s for s in sug if s["plate"] == "C")
    assert c_sug["status"] == "needs_manual"
    assert c_sug["suggested_x_um"] is None
    db.close()


def test_model_suggestion_default_pending_not_applied():
    r = _upload("01_baseline")
    sid = r.json()["scan_id"]
    sug = client.get(f"/api/scans/{sid}/model-suggestions").json()
    assert sug
    for s in sug:
        assert s["status"] in ("pending_review", "needs_manual")
    # 算法原始结果不因为模型建议存在而改变
    from app.models import SessionLocal
    db = SessionLocal()
    c = next(p for p in db.query(PlateResult).filter_by(scan_id=sid) if p.plate == "C")
    before = (c.offset_x_um, c.offset_y_um)
    db.close()
    review = client.post(
        f"/api/scans/{sid}/model-suggestions/{sug[0]['id']}/review",
        data={"decision": "accepted_in_part"})
    assert review.status_code == 200
    from app.models import SessionLocal
    db = SessionLocal()
    c = next(p for p in db.query(PlateResult).filter_by(scan_id=sid) if p.plate == "C")
    after = (c.offset_x_um, c.offset_y_um)
    db.close()
    assert before == after  # 人工/模型裁决绝不回写算法值


def test_confirmation_does_not_overwrite_algorithm():
    r = _upload("01_baseline")
    sid = r.json()["scan_id"]
    payload = {"plate": "M", "decision": "corrected",
               "offset_x_um": 12.3, "offset_y_um": -45.6,
               "uncertainty_x_um": 60, "uncertainty_y_um": 60,
               "comment": "人工复核"}
    rc = client.post(f"/api/scans/{sid}/confirmations", json=payload)
    assert rc.status_code == 200
    assert rc.json()["decision"] == "corrected"
    # 算法 M 结果保持原样
    from app.models import SessionLocal
    db = SessionLocal()
    m = next(p for p in db.query(PlateResult).filter_by(scan_id=sid) if p.plate == "M")
    alg_m = (m.offset_x_um, m.offset_y_um)
    db.close()
    res = r.json()["result"]["plates"]["M"]["offset_um"]
    assert list(alg_m) == pytest.approx(res, abs=1e-6)
    assert (12.3, -45.6) != pytest.approx(alg_m, abs=1.0)


def test_dpi_mismatch_warns_but_uses_measured_scale():
    r = _upload("01_baseline", declared_dpi=240)  # 真值约 200
    body = r.json()["result"]
    codes = [w["code"] for w in body["warnings"]]
    assert "dpi_mismatch" in codes
    # 声明 DPI 被记录，但校准尺度以算法测得为准（不静默改用 240）
    assert abs(body["calibration"]["measured_dpi"] - 200) < 10
    assert body["declared_dpi"] == 240


def test_overlay_endpoint_returns_png():
    path = os.path.join(FIX, "01_baseline.png")
    with open(path, "rb") as f:
        r = client.post("/api/scans/overlay",
                        files={"image": ("t.png", f, "image/png")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
