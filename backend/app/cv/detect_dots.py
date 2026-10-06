"""色版圆点候选检测：Lab 油墨近邻分类 + 连通域几何过滤 + 亚像素加权质心。

只在彩色通道上找 C/M/Y 点；K 点由校准阶段的暗掩膜找到（避免与污点竞争）。
不能确定时返回多个候选，绝不替用户拍板。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .spec import DOT_DIAMETER_MM, INK_SRGB


def _ink_lab_table() -> np.ndarray:
    # 顺序：C, M, Y
    patches = np.zeros((1, 3, 3), dtype=np.uint8)
    for j, key in enumerate(PLATES):
        patches[0, j] = INK_SRGB[key][::-1]  # RGB -> BGR
    lab = cv2.cvtColor(patches, cv2.COLOR_BGR2LAB).astype(np.float64)
    return lab[0]


PLATES = ("C", "M", "Y")
INK_LAB = None  # 延迟初始化（PLATES 顺序 + 末尾 K）


def _ink_table() -> np.ndarray:
    global INK_LAB
    if INK_LAB is None:
        INK_LAB = _ink_lab_table()
    return INK_LAB


@dataclass
class DotCandidate:
    plate: str
    centroid_px: tuple[float, float]
    area_px: float
    circularity: float
    color_distance: float          # 到该版参考色的 Lab 距离
    complete: bool                 # 轮廓是否近似完整圆盘
    bbox: tuple[int, int, int, int]

    def to_dict(self) -> dict:
        return {
            "plate": self.plate,
            "centroid_px": [round(v, 2) for v in self.centroid_px],
            "area_px": round(self.area_px, 1),
            "circularity": round(self.circularity, 3),
            "color_distance": round(self.color_distance, 1),
            "complete": self.complete,
            "bbox": list(self.bbox),
        }


def _soft_membership(lab_roi: np.ndarray, ink_lab: np.ndarray) -> np.ndarray:
    """像素属于某油墨的软隶属度 [0,1]：色度接近 1，白/灰接近 0。"""
    d = np.linalg.norm(lab_roi - ink_lab, axis=2)
    m = np.clip(1.0 - d / 80.0, 0.0, 1.0)
    return m * m  # 平方压低灰底噪声


def find_plate_candidates(
    image_bgr: np.ndarray,
    frame_corners_px: np.ndarray,
    scale_px_per_mm: float,
) -> dict[str, list[DotCandidate]]:
    """在方框内（略放大的 ROI）为 C/M/Y 各找圆点候选。"""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ink_lab = _ink_table()

    hull = cv2.convexHull(np.asarray(frame_corners_px, dtype=np.int32))
    x, y, w, h = cv2.boundingRect(hull)
    pad = int(0.8 * scale_px_per_mm)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(image_bgr.shape[1], x + w + pad), min(image_bgr.shape[0], y + h + pad)

    inside = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(inside, hull, 255)
    roi_mask = np.zeros_like(inside)
    roi_mask[y0:y1, x0:x1] = 255
    inside = cv2.bitwise_and(inside, roi_mask)

    expected_r_px = DOT_DIAMETER_MM / 2.0 * scale_px_per_mm
    area_min = max(6.0, np.pi * (expected_r_px * 0.5) ** 2)
    area_max = np.pi * (expected_r_px * 1.7) ** 2

    # 硬掩膜只用于连通域定位；残缺点（被白纸遮挡 45%）面积下限低，
    # 不能用开运算（会把月牙残片打断/删光），直接在色度掩膜上找连通域。
    a_ch, b_ch = lab[:, :, 1], lab[:, :, 2]
    chroma = cv2.magnitude(a_ch - 128.0, b_ch - 128.0)
    color_mask = (chroma > 20).astype(np.uint8) * 255
    color_mask = cv2.bitwise_and(color_mask, inside)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask)
    out: dict[str, list[DotCandidate]] = {p: [] for p in PLATES}

    H, W = gray.shape
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if not (area_min * 0.7 <= area <= area_max * 1.3):
            continue

        bx = int(stats[i, cv2.CC_STAT_LEFT]); by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH]); bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw > expected_r_px * 4 or bh > expected_r_px * 4:
            continue
        # 软质心：在 bbox 上按各色版隶属度加权，选总权重最大的色版
        ox = max(0, bx - 2); oy = max(0, by - 2)
        ox1 = min(W, bx + bw + 2); oy1 = min(H, by + bh + 2)
        lab_roi = lab[oy:oy1, ox:ox1]
        comp_roi = (labels[oy:oy1, ox:ox1] == i)

        best = None  # (plate, mass, cx, cy, mean_dist)
        for pi, plate in enumerate(PLATES):
            m = _soft_membership(lab_roi, ink_lab[pi]) * comp_roi
            mass = float(m.sum())
            if mass < 0.25:
                continue
            ys, xs = np.mgrid[oy:oy1, ox:ox1]
            cx = float((m * xs).sum() / mass)
            cy = float((m * ys).sum() / mass)
            # 加权平均颜色距离
            dmap = np.linalg.norm(lab_roi - ink_lab[pi], axis=2)
            mean_d = float((m * dmap).sum() / mass)
            if best is None or mass > best[1]:
                best = (plate, mass, cx, cy, mean_d)
        if best is None:
            continue
        plate, soft_area, cx, cy, color_d = best
        # 残缺/被白纸遮挡的点，墨边与白混色后 Lab 距离升高，阈值放宽到 70；
        # 残缺由 complete=False 表达，而不是直接丢弃候选。
        if color_d > 70:
            continue

        # 圆度仍由硬连通域算（形状门限）
        comp_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0
        # 月牙/残片圆度按连通域外包圆归一后可能 >1；夹紧到 [0,1]
        circularity = max(0.0, min(1.0, circularity))
        # 被白纸遮挡一半的残片不再像圆盘，圆度门限放宽（残缺由 complete=False 表达）
        if circularity < 0.25:
            continue

        disc_ratio = soft_area / (np.pi * expected_r_px ** 2)
        complete = bool(0.55 <= disc_ratio <= 1.5 and circularity >= 0.70)

        out[plate].append(DotCandidate(
            plate=plate,
            centroid_px=(cx, cy),
            area_px=soft_area,
            circularity=float(circularity),
            color_distance=color_d,
            complete=complete,
            bbox=(bx, by, bw, bh),
        ))

    return out
