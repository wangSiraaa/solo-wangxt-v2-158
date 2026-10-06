import datetime as dt

import numpy as np

from ..models import (Confirmation, ModelSuggestion, PlateResult, Scan,
                      SessionLocal, init_db)
from .offline_review import MODEL_NAME, MODEL_VERSION, suggest


def persist_analysis(name: str, filename: str, result: dict,
                     declared_dpi: float | None, dpi_source: str) -> int:
    init_db()
    db = SessionLocal()
    try:
        cal = result.get("calibration")
        scan = Scan(
            name=name, filename=filename,
            declared_dpi=declared_dpi, dpi_source=dpi_source,
            status=result.get("status", "new"),
            angle_deg=cal and cal["angle_deg"],
            angle_uncertainty_deg=cal and cal["angle_uncertainty_deg"],
            measured_dpi=cal and cal["measured_dpi"],
            scale_px_per_mm=cal and cal["scale_px_per_mm"],
            scale_uncertainty_percent=cal and cal["scale_uncertainty_percent"],
            calibration_residual_mm=cal and cal["residual_mm"],
            calibration_usable=bool(cal and cal.get("usable")),
            calibration_method=cal and cal.get("method"),
            result_json=result,
        )
        db.add(scan)
        db.flush()
        for plate, b in result.get("plates", {}).items():
            off = b.get("offset_um")
            unc = b.get("uncertainty_um")
            db.add(PlateResult(
                scan_id=scan.id, plate=plate, status=b["status"],
                offset_x_um=off and float(off[0]),
                offset_y_um=off and float(off[1]),
                uncertainty_x_um=unc and float(unc[0]),
                uncertainty_y_um=unc and float(unc[1]),
                confidence_range_um=b.get("confidence_range_um"),
                confidence=b.get("confidence", 0.0),
                candidates_json=b.get("candidates", []),
                notes_json=b.get("notes", []),
            ))
        # 离线模型建议（pending_review，不自动采纳）
        for s in suggest(result):
            db.add(ModelSuggestion(
                scan_id=scan.id, plate=s["plate"],
                model_name=MODEL_NAME, model_version=MODEL_VERSION,
                suggested_x_um=s["x"], suggested_y_um=s["y"],
                suggested_range_um=s["range_um"],
                status="pending_review" if s["x"] is not None else "needs_manual",
                rationale=s["rationale"],
            ))
        db.commit()
        return scan.id
    finally:
        db.close()


def add_confirmation(scan_id: int, payload) -> dict:
    init_db()
    db = SessionLocal()
    try:
        scan = db.get(Scan, scan_id)
        if scan is None:
            raise LookupError("scan_not_found")
        c = Confirmation(
            scan_id=scan_id, plate=payload.plate, decision=payload.decision,
            offset_x_um=payload.offset_x_um, offset_y_um=payload.offset_y_um,
            uncertainty_x_um=payload.uncertainty_x_um,
            uncertainty_y_um=payload.uncertainty_y_um,
            comment=payload.comment, reviewer=payload.reviewer,
        )
        db.add(c)
        # 裁决只更新人工确认状态，绝不覆盖算法原始结果
        scan.status = f"reviewed:{payload.decision}"
        db.commit()
        db.refresh(c)
        return _confirmation_dict(c)
    finally:
        db.close()


def list_model_suggestions(scan_id: int, status: str | None = None):
    init_db()
    db = SessionLocal()
    try:
        q = db.query(ModelSuggestion).filter_by(scan_id=scan_id)
        if status:
            q = q.filter_by(status=status)
        return [
            dict(id=m.id, scan_id=m.scan_id, plate=m.plate,
                 model_name=m.model_name, model_version=m.model_version,
                 suggested_x_um=m.suggested_x_um, suggested_y_um=m.suggested_y_um,
                 suggested_range_um=m.suggested_range_um, status=m.status,
                 rationale=m.rationale)
            for m in q.order_by(ModelSuggestion.id)]
    finally:
        db.close()


def review_suggestion(scan_id: int, suggestion_id: int, decision: str):
    """人工对模型建议裁决：accepted_in_part / rejected；仍然不自动改算法结果。"""
    init_db()
    db = SessionLocal()
    try:
        m = db.get(ModelSuggestion, suggestion_id)
        if m is None or m.scan_id != scan_id:
            raise LookupError("suggestion_not_found")
        m.status = decision
        m.reviewed_at = dt.datetime.utcnow()
        db.commit()
        return dict(id=m.id, status=m.status)
    finally:
        db.close()


def _confirmation_dict(c: Confirmation) -> dict:
    return dict(
        id=c.id, scan_id=c.scan_id, plate=c.plate, decision=c.decision,
        offset_x_um=c.offset_x_um, offset_y_um=c.offset_y_um,
        uncertainty_x_um=c.uncertainty_x_um, uncertainty_y_um=c.uncertainty_y_um,
        comment=c.comment, reviewer=c.reviewer,
        created_at=c.created_at.isoformat(),
    )
