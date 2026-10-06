"""检测主管线：K 校准 → 色版点候选 → 偏移/不确定度/状态/来源。"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import cv2
import numpy as np

from .calibrate import calibrate
from .detect_dots import find_plate_candidates
from .geometry import dpi_uncertainty_to_position_um
from .spec import (
    DOT_NOMINAL_MM,
    K_SELFCHECK_LIMIT_UM,
    MAX_CALIB_RESIDUAL_MM,
    MIN_CONFIDENCE,
    PLATE_KEYS,
    REFERENCE_PLATE,
    SPEC_VERSION,
)

ALGO_VERSION = "regdet-1.0.0"


def load_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解码图像: {path}")
    return img


def _centroid_uncertainty_px(area_px: float, complete: bool) -> float:
    # 亚像素质心经验误差：圆盘越大越稳；残缺点放宽
    base = max(0.15, 0.6 / math.sqrt(max(area_px, 1.0)))
    return base * (1.8 if not complete else 1.0)


def _select_candidates(cands):
    """从同版多个候选中判定主选/备选。返回 (primary|None, ambiguity|None)。"""
    if not cands:
        return None, None
    complete = [c for c in cands if c.complete]
    pool = complete or cands
    # 分越小越好：颜色距离 + 圆度损失
    scored = sorted(
        pool, key=lambda c: c.color_distance + (1 - c.circularity) * 25.0)
    primary = scored[0]
    if len(scored) > 1:
        spread = max(
            math.hypot(c.centroid_px[0] - primary.centroid_px[0],
                       c.centroid_px[1] - primary.centroid_px[1])
            for c in scored[1:])
        return primary, {"spread_px": round(spread, 2), "n_candidates": len(cands)}
    return primary, None


def analyze(image_bgr: np.ndarray, declared_dpi: float | None = None,
            dpi_source: str = "none") -> dict:
    result = {
        "spec_version": SPEC_VERSION,
        "algo_version": ALGO_VERSION,
        "analyzed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": "ok",
        "calibration": None,
        "plates": {},
        "reference_plate": REFERENCE_PLATE,
        "declared_dpi": declared_dpi,
        "dpi_source": dpi_source,
        "warnings": [],
    }

    calib, cmeta, err = calibrate(image_bgr)
    if err is not None:
        result["status"] = "calibration_failed"
        result["error"] = {"code": err.code, "message": err.message}
        return result

    # DPI：优先扫描图自报/人工录入；不静默改写。算法尺度始终单独保留可核对。
    s = calib.scale_px_per_mm
    if declared_dpi:
        declared_s = declared_dpi / 25.4
        dpi_dev = abs(declared_s - s) / s * 100.0
        if dpi_dev > 2.0:
            result["warnings"].append({
                "code": "dpi_mismatch",
                "message": f"声明 DPI 与校准尺度相差 {dpi_dev:.1f}%；微米结果以校准尺度为准",
                "declared_dpi": declared_dpi,
                "measured_dpi": round(calib.dpi_effective, 1),
            })
    result["calibration"] = {
        "angle_deg": round(calib.angle_deg, 3),
        "angle_uncertainty_deg": round(calib.angle_uncertainty_deg, 3),
        "scale_px_per_mm": round(s, 4),
        "measured_dpi": round(calib.dpi_effective, 1),
        "scale_uncertainty_percent": round(calib.scale_uncertainty_percent, 3),
        "residual_mm": round(calib.residual_mm, 4),
        "translation_px": [round(v, 2) for v in calib.translation_px],
        "method": calib.method,
        "meta": {k: (round(v, 3) if isinstance(v, float) else v)
                 for k, v in cmeta.items()},
        "usable": calib.residual_mm <= MAX_CALIB_RESIDUAL_MM,
    }
    if not result["calibration"]["usable"]:
        result["warnings"].append({
            "code": "calibration_poor",
            "message": f"校准残差 {calib.residual_mm:.3f} mm 超出门限 "
                       f"{MAX_CALIB_RESIDUAL_MM} mm，偏移仅作参考",
        })
        result["status"] = "calibration_poor"

    candidates_by_plate = find_plate_candidates(image_bgr, calib.frame_corners_px, s)

    def plate_block(plate: str) -> dict:
        if plate == REFERENCE_PLATE:
            return {"status": "reference", "offset_um": [0.0, 0.0],
                    "uncertainty_um": [0.0, 0.0], "confidence": 1.0,
                    "candidates": [], "provenance": "K 为参照版，偏移恒为 0"}
        cands = candidates_by_plate[plate]
        nominal = np.array(DOT_NOMINAL_MM[plate])
        block = {
            "status": "missing",
            "nominal_mm": list(nominal),
            "offset_um": None,
            "uncertainty_um": None,
            "confidence_range_um": None,
            "confidence": 0.0,
            "candidates": [c.to_dict() for c in cands],
            "selected_index": None,
            "notes": [],
        }
        primary, amb = _select_candidates(cands)
        if primary is None:
            block["status"] = "missing"
            # 找不到点时给出基于标称位置的搜索范围（不是测量值！）
            reach = (np.hypot(*nominal) * calib.scale_uncertainty_percent / 100.0
                     + MAX_CALIB_RESIDUAL_MM)
            block["confidence_range_um"] = round(reach * 1000.0, 1)
            block["notes"].append("未找到合格圆点；range 为搜索半径而非测量精度")
            return block

        idx = block["candidates"].index(primary.to_dict())
        block["selected_index"] = idx
        meas_mm = calib.to_print_mm(np.array([primary.centroid_px]))[0]
        offset_mm = meas_mm - nominal
        block["measured_mm"] = [round(meas_mm[0], 4), round(meas_mm[1], 4)]

        u_px = _centroid_uncertainty_px(primary.area_px, primary.complete)
        u_centroid_um = u_px / s * 1000.0
        u_calib_um = calib.residual_mm * 1000.0
        u_scale_x = dpi_uncertainty_to_position_um(
            meas_mm[0], calib.scale_uncertainty_percent)
        u_scale_y = dpi_uncertainty_to_position_um(
            meas_mm[1], calib.scale_uncertainty_percent)
        ux = math.hypot(u_centroid_um, u_calib_um, u_scale_x)
        uy = math.hypot(u_centroid_um, u_calib_um, u_scale_y)
        if amb:
            spread_mm = amb["spread_px"] / s
            ux = max(ux, spread_mm * 1000.0 / 2.0)
            uy = max(uy, spread_mm * 1000.0 / 2.0)
            block["ambiguity"] = amb

        block["offset_um"] = [round(offset_mm[0] * 1000.0, 1),
                              round(offset_mm[1] * 1000.0, 1)]
        block["uncertainty_um"] = [round(ux, 1), round(uy, 1)]
        block["confidence_range_um"] = round(max(ux, uy), 1)

        conf = (max(0.0, 1 - u_centroid_um / 60.0) * 0.5
                + primary.circularity * 0.25
                + max(0.0, 1 - primary.color_distance / 55.0) * 0.25)
        if calib.residual_mm > MAX_CALIB_RESIDUAL_MM:
            conf *= 0.6
        if not primary.complete:
            conf *= 0.75
            block["notes"].append("圆点不完整（可能被遮挡/残缺），质心可能偏移")
        block["confidence"] = round(min(1.0, conf), 3)

        if amb:
            block["status"] = "ambiguous"
        elif not primary.complete:
            block["status"] = "incomplete"
        elif block["confidence"] < MIN_CONFIDENCE:
            block["status"] = "low_confidence"
        else:
            block["status"] = "measured"
        return block

    for plate in PLATE_KEYS:
        result["plates"][plate] = plate_block(plate)

    # K 自检：dK0 / dK1 应给出相同的参照一致性（由校准残差体现）
    if cmeta["n_kdots"] >= 2 and calib.residual_mm * 1000 > K_SELFCHECK_LIMIT_UM:
        result["warnings"].append({
            "code": "k_selfcheck",
            "message": "两个 K 控制点一致性偏差较大，校准/检测可疑",
            "mismatch_um": round(calib.residual_mm * 1000.0, 1),
        })

    statuses = {p: b["status"] for p, b in result["plates"].items()
                if p != REFERENCE_PLATE}
    if result["status"] == "ok":
        if all(v == "missing" for v in statuses.values()):
            result["status"] = "no_plates"
        elif any(v in ("ambiguous", "incomplete", "missing", "low_confidence")
                 for v in statuses.values()):
            result["status"] = "incomplete"
    return result
