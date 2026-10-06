import cv2
import numpy as np

from ..cv.calibrate import _dark_mask, _find_frame
from ..cv.detect_dots import find_plate_candidates


def build_overlay(image_bgr: np.ndarray, result: dict) -> np.ndarray:
    """生成叠加层（BGRA，透明背景）：校准尺、候选点、不确定范围、坐标轴。

    坐标全部用像素；前端按原图分辨率叠加，再随缩放变换。
    """
    h, w = image_bgr.shape[:2]
    ov = np.zeros((h, w, 4), dtype=np.uint8)
    cal = result.get("calibration")
    if cal is None:
        return ov

    # 重新检测框角点（结果 JSON 没存像素坐标；这里只用于可视化，允许失败）
    try:
        q, side = _find_frame(image_bgr, _dark_mask(image_bgr))
        if q is not None:
            cv2.polylines(ov, [q.astype(np.int32)], True, (0, 220, 255, 220), 1,
                          cv2.LINE_AA)
            for p in q.astype(int):
                cv2.circle(ov, tuple(p), 2, (0, 220, 255, 220), -1)
    except Exception:
        pass

    colors = {"C": (255, 200, 0, 230), "M": (220, 0, 220, 230),
              "Y": (0, 230, 230, 230), "K": (200, 200, 200, 230)}
    for plate, b in result.get("plates", {}).items():
        col = colors.get(plate, (255, 255, 255, 200))
        for c in b.get("candidates", []):
            x, y = int(round(c["centroid_px"][0])), int(round(c["centroid_px"][1]))
            sel = b.get("candidates", []).index(c) == b.get("selected_index")
            r = 6 if sel else 4
            cv2.circle(ov, (x, y), r, col, 1 if not sel else 2, cv2.LINE_AA)
        # 不确定范围（像素）：取 uncertainty 平均换算
        unc = b.get("uncertainty_um")
        if unc and b.get("measured_mm") and b.get("selected_index") is not None:
            c = b["candidates"][b["selected_index"]]
            x, y = int(c["centroid_px"][0]), int(c["centroid_px"][1])
            s = cal["scale_px_per_mm"]
            ru = int(round(max(unc) / 1000.0 * s))
            cv2.circle(ov, (x, y), max(ru, 3), (col[0], col[1], col[2], 90), 1,
                       cv2.LINE_AA)
    return ov


def overlay_png(image_bgr: np.ndarray, result: dict) -> bytes:
    ov = build_overlay(image_bgr, result)
    ok, buf = cv2.imencode(".png", ov)
    return buf.tobytes() if ok else b""
