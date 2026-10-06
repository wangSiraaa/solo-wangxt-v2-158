import io

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ..cv.detect_target import analyze, load_image
from ..schemas.reg import ConfirmationIn
from ..services import persist
from ..services.overlay import overlay_png

router = APIRouter(prefix="/api", tags=["registration"])


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "无法解码图像（仅接受项目提供的合成/扫描 PNG/JPG）")
    return img


@router.post("/scans/analyze")
async def analyze_scan(
    image: UploadFile = File(...),
    name: str = Form("unnamed"),
    declared_dpi: float | None = Form(None),
    dpi_source: str = Form("none"),
):
    data = await image.read()
    img = _decode(data)
    result = analyze(img, declared_dpi=declared_dpi, dpi_source=dpi_source)
    scan_id = persist.persist_analysis(
        name, image.filename or "upload.png", result, declared_dpi, dpi_source)
    return {"scan_id": scan_id, "result": result}


@router.post("/scans/{scan_id}/confirmations")
async def add_confirmation(scan_id: int, body: ConfirmationIn):
    try:
        return persist.add_confirmation(scan_id, body)
    except LookupError:
        raise HTTPException(404, "扫描记录不存在")


@router.get("/scans/{scan_id}/model-suggestions")
async def get_suggestions(scan_id: int, status: str | None = None):
    return persist.list_model_suggestions(scan_id, status)


@router.post("/scans/{scan_id}/model-suggestions/{suggestion_id}/review")
async def review_suggestion(scan_id: int, suggestion_id: int,
                            decision: str = Form(...)):
    if decision not in ("accepted_in_part", "rejected", "pending_review"):
        raise HTTPException(400, "decision 必须是 accepted_in_part / rejected")
    try:
        return persist.review_suggestion(scan_id, suggestion_id, decision)
    except LookupError:
        raise HTTPException(404, "建议不存在")


@router.post("/scans/overlay")
async def make_overlay(
    image: UploadFile = File(...),
    declared_dpi: float | None = Form(None),
    dpi_source: str = Form("none"),
):
    """不上库的即时叠层（供前端放大查看时对齐校准尺/候选点）。"""
    data = await image.read()
    img = _decode(data)
    result = analyze(img, declared_dpi=declared_dpi, dpi_source=dpi_source)
    png = overlay_png(img, result)
    return Response(content=png, media_type="image/png",
                    headers={"X-Analysis-Status": result["status"]})


@router.get("/health")
async def health():
    return {"status": "ok", "safety": "offline_review_only_no_device_control"}
