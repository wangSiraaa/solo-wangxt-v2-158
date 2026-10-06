"""REG-TARGET/1 规范的唯一事实来源。

夹具生成器 (scripts/generate_fixtures.py) 与检测器 (app/cv/detect.py) 都从这里
导入几何与油墨参数，避免规范文档、生成、检测三方漂移。

坐标约定（见 docs/target-spec.md）：
- 印刷坐标系：毫米，色标方框中心为原点，x 向右、y 向上。
- 基准分辨率：BASE_DPI；扫描像素坐标 x 向右、y 向下。
"""
from __future__ import annotations

from dataclasses import dataclass

SPEC_VERSION = "REG-TARGET/1"
BASE_DPI = 200.0
MM_PER_INCH = 25.4

# ---- 校准要素（全部 K 油墨） ----
FRAME_SIDE_MM = 9.4           # 描边中心线边长（检测器 TLS 目标）
FRAME_OUTER_SIDE_MM = 9.7     # 描边外边边长
FRAME_INNER_SIDE_MM = 9.1     # 描边内边边长
FRAME_LINEWIDTH_MM = 0.30
PIP_SIDE_MM = 0.90            # 角点填充方块边长
PIP_INSET_MM = 0.55           # 方块外边相对方框角的内缩
# 方块中心（印刷坐标 mm）；非对称布置用于定方向。
# 外边相对框外边(±4.85)内缩 0.55 + 方块边长一半 0.45 ⇒ 中心 ±(4.85-0.55-0.45)=±3.85
PIP_CENTERS_MM: dict[str, tuple[float, float]] = {
    "P0": (3.85, 3.85),
    "P1": (-3.85, -3.85),
}

# ---- 色版圆点 ----
DOT_DIAMETER_MM = 0.55

# 标称印刷位置（零偏移时）。所有圆点排在 x 轴上，远离两条 pip 对角线，
# 防止圆点与角点方块连通；K 版带一个冗余点 dK1 做自检。
DOT_NOMINAL_MM: dict[str, tuple[float, float]] = {
    "C": (-3.50, 0.0),
    "M": (-1.75, 0.0),
    "Y": (0.00, 0.00),
    "K": (1.75, 0.00),
    "K1": (3.50, 0.00),
}
# 参与色版偏移上报的点；K1 仅用于一致性自检
PLATE_KEYS = ("C", "M", "Y", "K")
CONTROL_DOT = "K1"
REFERENCE_PLATE = "K"

# ---- 油墨参考色（sRGB, 理想实地；检测在 Lab 空间比较） ----
INK_SRGB: dict[str, tuple[int, int, int]] = {
    "C": (0, 160, 210),
    "M": (210, 0, 140),
    "Y": (245, 215, 0),
    "K": (25, 25, 25),
}

# ---- 质量门限 ----
MIN_CONFIDENCE = 0.45
# 校准残差超过该值（mm）时，校准结果不可用于微米级结论
MAX_CALIB_RESIDUAL_MM = 0.15
# 两个 K 控制点偏移差异超过该值（µm）时，标记校准/检测可疑
K_SELFCHECK_LIMIT_UM = 60.0


def mm_to_px(mm: float, dpi: float = BASE_DPI) -> float:
    return mm * dpi / MM_PER_INCH


def px_to_mm(px: float, dpi: float = BASE_DPI) -> float:
    return px * MM_PER_INCH / dpi


@dataclass(frozen=True)
class TargetSpec:
    spec_version: str = SPEC_VERSION
    base_dpi: float = BASE_DPI


TARGET = TargetSpec()
