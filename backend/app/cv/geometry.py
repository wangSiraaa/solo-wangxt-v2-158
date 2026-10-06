"""像素坐标 ↔ 印刷坐标的仿射变换工具。

印刷坐标：毫米，y 向上；像素坐标：y 向下。
扫描仿射模型（像素 = 模型(印刷)）：

    p = s · R(θ) · r + t

其中 r 为印刷坐标 (mm)，R 为常规 2x2 旋转矩阵（先印刷坐标 y 翻号到像素朝向，
θ 为扫描图相对样张的旋转），s 单位 px/mm，t 为像素平移。

实现上直接用 OpenCV 的 2x3 仿射矩阵 M（行主序）：
    [x_px]   [m00 m01 m02] [x_mm]
    [y_px] = [m10 m11 m12] [y_mm]
M 已内含 y 轴翻转。其逆矩阵把检测到的像素点变回毫米印刷坐标。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .spec import BASE_DPI, MM_PER_INCH


@dataclass
class Calibration:
    """从黑色校准尺估出的纸张级变换（与色版偏移严格分开）。"""

    M: np.ndarray                  # 印刷(mm) -> 像素
    M_inv: np.ndarray              # 像素 -> 印刷(mm)
    angle_deg: float               # 扫描旋转角（度，绕页面）
    scale_px_per_mm: float         # 扫描尺度
    dpi_effective: float           # s * 25.4
    translation_px: tuple[float, float]
    residual_mm: float             # 校准要素拟合残差（mm）
    angle_uncertainty_deg: float
    scale_uncertainty_percent: float
    method: str                    # provenance: 用了哪些校准要素
    frame_corners_px: np.ndarray   # 方框四角（像素，调试叠层用）
    pip_points_px: dict[str, np.ndarray]

    def to_print_mm(self, points_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.transform(pts, self.M_inv).reshape(-1, 2)
        return out


def build_affine(angle_deg: float, dpi: float, translation_px: tuple[float, float]) -> np.ndarray:
    """生成 印刷(mm, y-up) -> 像素(px, y-down) 的 2x3 仿射矩阵。"""
    s = dpi / MM_PER_INCH
    th = math.radians(angle_deg)
    c, sn = math.cos(th), math.sin(th)
    # 印刷坐标先翻 y（y_up -> y_down），再旋转缩放，最后平移。
    # 对列向量 r_mm:  p = s R(θ) diag(1,-1) r + t
    flip = np.array([[1.0, 0.0], [0.0, -1.0]])
    rot = np.array([[c, -sn], [sn, c]])
    linear = s * rot @ flip
    M = np.hstack([linear, np.array(translation_px, dtype=np.float64).reshape(2, 1)])
    return M


def invert_affine(M: np.ndarray) -> np.ndarray:
    M_inv = cv2.invertAffineTransform(M.astype(np.float64))
    return M_inv


def estimate_scale_dpi(scale_px_per_mm: float) -> float:
    return scale_px_per_mm * MM_PER_INCH


def dpi_uncertainty_to_position_um(
    offset_mm: float, scale_uncertainty_percent: float
) -> float:
    """尺度不确定度经 |offset| 杠杆放大成位置不确定度（µm）。

    中心点附近（offset≈0）尺度误差几乎不贡献偏移；越靠边缘贡献越大。
    """
    return abs(offset_mm) * scale_uncertainty_percent / 100.0 * 1000.0
