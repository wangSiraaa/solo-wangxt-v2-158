"""FastAPI service: upload PRT-1 scans, read plate offsets, persist + confirm.

Endpoints (all JSON unless noted):

* ``GET  /api/health``
* ``GET  /api/datasets``                 shipped synthetic images
* ``POST /api/analyze``                  multipart upload OR ``dataset=`` name
* ``GET  /api/runs/{run_id}``            stored run + readings + confirmations
* ``GET  /api/runs``                     list recent runs
* ``POST /api/runs/{run_id}/confirm``    human QC decision (never automated)
* ``GET  /api/runs/{run_id}/overlay.png`` evidence overlay image
* ``POST /api/runs/{run_id}/review``     offline advisory model cross-check

The service is analysis-only: no endpoint issues commands to press hardware.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import markspec as spec
from . import model_review, overlay
from .db import (Confirmation, PlateReading, Scan, get_session, init_db)
from .detect import DETECTOR_VERSION, analyze_image
from .synth import DATA_DIR

app = FastAPI(title="PRT-1 registration QA", version="DETECTOR_VERSION")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "format": spec.FORMAT_ID,
            "detector_version": DETECTOR_VERSION,
            "control_authority": "analysis_only"}


@app.get("/api/datasets")
def datasets() -> list[dict]:
    import json
    out = []
    for p in sorted(DATA_DIR.glob("*.png")):
        meta = {}
        sj = Path(str(p) + ".json")
        if sj.exists():
            meta = json.loads(sj.read_text())
        out.append({"name": p.name, "bytes": p.stat().st_size,
                    "truth": meta})
    return out


def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(415, "无法解码为图像（仅支持 PNG/JPEG/TIFF/BMP）")
    return img


def _persist(db: Session, result: dict, filename: str, digest: str) -> Scan:
    scan = Scan(
        run_id=result["run_id"], filename=filename, sha256=digest,
        width_px=result["image"]["width_px"],
        height_px=result["image"]["height_px"],
        declared_scan_ppi=result["image"]["declared_scan_ppi"],
        measured_ppi=result["image"]["measured_ppi"],
        frame_rotation_deg=result["frame"]["rotation_deg"],
        frame_scale=result["frame"]["scale_vs_native"],
        frame_quality=result["frame"]["quality"],
        residual_rotation_deg=result["frame"]["residual_rotation_deg"],
        residual_scale=result["frame"]["residual_scale"],
        status=result["status"],
        detector_version=result["provenance"]["detector_version"],
        provenance=result["provenance"])
    db.add(scan)
    db.flush()
    for plate, pr in result["plates"].items():
        db.add(PlateReading(
            scan_id=scan.id, plate=plate, status=pr["status"],
            dx_um=pr["dx_um"], dy_um=pr["dy_um"],
            x_lo_um=pr["x_range_um"][0], x_hi_um=pr["x_range_um"][1],
            y_lo_um=pr["y_range_um"][0], y_hi_um=pr["y_range_um"][1],
            confidence=pr["confidence"],
            n_x_readings=pr["n_x_readings"], n_y_readings=pr["n_y_readings"],
            notes=pr["notes"]))
    db.commit()
    db.refresh(scan)
    return scan


def _serialize(scan: Scan, db: Session) -> dict:
    readings = db.scalars(
        select(PlateReading).where(PlateReading.scan_id == scan.id)
        .order_by(PlateReading.id)).all()
    confirms = db.scalars(
        select(Confirmation).where(Confirmation.scan_id == scan.id)
        .order_by(Confirmation.created_at)).all()
    return {
        "run_id": scan.run_id, "filename": scan.filename,
        "status": scan.status, "created_at": scan.created_at.isoformat(),
        "image": {"width_px": scan.width_px, "height_px": scan.height_px,
                  "declared_scan_ppi": scan.declared_scan_ppi,
                  "measured_ppi": scan.measured_ppi},
        "frame": {"rotation_deg": scan.frame_rotation_deg,
                  "scale_vs_native": scan.frame_scale,
                  "quality": scan.frame_quality,
                  "residual_rotation_deg": scan.residual_rotation_deg,
                  "residual_scale": scan.residual_scale},
        "plates": {r.plate: {
            "plate": r.plate, "status": r.status, "dx_um": r.dx_um,
            "dy_um": r.dy_um,
            "x_range_um": [r.x_lo_um, r.x_hi_um],
            "y_range_um": [r.y_lo_um, r.y_hi_um],
            "confidence": r.confidence,
            "n_x_readings": r.n_x_readings,
            "n_y_readings": r.n_y_readings, "notes": r.notes}
            for r in readings},
        "confirmations": [
            {"operator_id": c.operator_id, "decision": c.decision,
             "corrected_dx_um": c.corrected_dx_um,
             "corrected_dy_um": c.corrected_dy_um, "reason": c.reason,
             "created_at": c.created_at.isoformat()} for c in confirms],
        "provenance": scan.provenance,
    }


def _run_analysis(data: bytes, filename: str, declared_ppi: float | None,
                  db: Session) -> dict:
    img = _decode(data)
    result = analyze_image(img, declared_ppi)
    result.run_id = uuid.uuid4().hex[:12]
    payload = result.to_dict()
    # Canonical image is recomputed from the upload on demand (overlay), so
    # we don't store image bytes here; the digest ties back to the source.
    digest = hashlib.sha256(data).hexdigest()
    scan = _persist(db, payload, filename, digest)
    out = _serialize(scan, db)
    out["detected"] = payload["detected"]
    out["observations"] = payload["observations"]
    return out


@app.post("/api/analyze")
async def analyze(
    db: Session = Depends(get_session),
    file: UploadFile | None = File(default=None),
    dataset: str | None = Form(default=None),
    declared_ppi: float | None = Form(default=None),
) -> dict:
    if dataset:
        p = DATA_DIR / Path(dataset).name
        if not p.exists():
            raise HTTPException(404, f"未知样例: {dataset}")
        data = p.read_bytes()
        filename = p.name
    elif file is not None:
        if Path(file.filename or "").suffix.lower() not in ALLOWED_EXT:
            raise HTTPException(415, "仅支持 PNG/JPEG/TIFF/BMP")
        data = await file.read()
        if len(data) > 25 * 1024 * 1024:
            raise HTTPException(413, "文件超过 25 MB")
        filename = file.filename or "upload.png"
    else:
        raise HTTPException(400, "需要 file 或 dataset 参数")
    return _run_analysis(data, filename, declared_ppi, db)


@app.get("/api/runs")
def list_runs(limit: int = 50, db: Session = Depends(get_session)) -> dict:
    rows = db.scalars(select(Scan).order_by(Scan.id.desc()).limit(limit)).all()
    return {"runs": [{"run_id": s.run_id, "filename": s.filename,
                      "status": s.status, "measured_ppi": s.measured_ppi,
                      "frame_rotation_deg": s.frame_rotation_deg,
                      "created_at": s.created_at.isoformat()} for s in rows]}


def _get_scan(run_id: str, db: Session) -> Scan:
    scan = db.scalar(select(Scan).where(Scan.run_id == run_id))
    if scan is None:
        raise HTTPException(404, "run 不存在")
    return scan


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_session)) -> dict:
    return _serialize(_get_scan(run_id, db), db)


class ConfirmIn(BaseModel):
    operator_id: str = Field(min_length=1, max_length=64)
    decision: str = Field(pattern="^(confirmed|corrected|rejected)$")
    corrected_dx_um: float | None = None
    corrected_dy_um: float | None = None
    reason: str = Field(default="", max_length=2000)


@app.post("/api/runs/{run_id}/confirm")
def confirm_run(run_id: str, body: ConfirmIn,
                db: Session = Depends(get_session)) -> dict:
    scan = _get_scan(run_id, db)
    if body.decision == "corrected" and (
            body.corrected_dx_um is None or body.corrected_dy_um is None):
        raise HTTPException(422, "corrected 必须给出 corrected_dx_um/dy_um")
    c = Confirmation(scan_id=scan.id, operator_id=body.operator_id,
                     decision=body.decision,
                     corrected_dx_um=body.corrected_dx_um,
                     corrected_dy_um=body.corrected_dy_um, reason=body.reason)
    db.add(c)
    db.commit()
    return _serialize(scan, db)


def _canonical_for(run_id: str, db: Session):
    """Reconstruct the canonical image for a run from the matching shipped
    dataset (analysis is restricted to project-provided synthetic images)."""
    scan = _get_scan(run_id, db)
    p = DATA_DIR / Path(scan.filename).name
    if not p.exists():
        raise HTTPException(404, "源图像不可用（仅限项目样例）")
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    canon, _ = overlay.canonical_warp(img)
    return canon, scan


@app.get("/api/runs/{run_id}/overlay.png")
def run_overlay(run_id: str, db: Session = Depends(get_session)) -> Response:
    canon, scan = _canonical_for(run_id, db)
    payload = _serialize(scan, db)
    vis = overlay.draw_overlay(canon, payload)
    ok, buf = cv2.imencode(".png", vis)
    return Response(content=buf.tobytes(), media_type="image/png")


@app.post("/api/runs/{run_id}/review")
def run_review(run_id: str, db: Session = Depends(get_session)) -> dict:
    canon, _ = _canonical_for(run_id, db)
    return model_review.review(canon, _serialize(_get_scan(run_id, db), db))
