"""生成随项目提供的合成色标与扫描图（含 ground truth）。

输出 backend/data/fixtures/ 下：
  <name>.png|jpg  扫描图
  <name>.json     ground truth（真实平移 µm / 扫描旋转角 / DPI / 缺陷）
  manifest.json   索引
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cv.geometry import build_affine
from app.cv.spec import (
    BASE_DPI,
    DOT_DIAMETER_MM,
    DOT_NOMINAL_MM,
    FRAME_LINEWIDTH_MM,
    FRAME_OUTER_SIDE_MM,
    FRAME_SIDE_MM,
    INK_SRGB,
    MM_PER_INCH,
    PIP_CENTERS_MM,
    PIP_INSET_MM,
    PIP_SIDE_MM,
    SPEC_VERSION,
)

RENDER_DPI = 600.0  # 超采样渲染，再仿射到扫描分辨率


def _srgb_to_bgr(c: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = c
    return (b, g, r)


def _mm_to_printpx(points_mm: np.ndarray, shape: tuple[int, int], dpi: float) -> np.ndarray:
    """印刷 mm(y-up) -> 渲染像素(y-down)，原点取画布中心。"""
    pr = dpi / MM_PER_INCH
    h, w = shape[:2]
    cx, cy = w / 2.0, h / 2.0
    out = np.empty_like(points_mm, dtype=np.float64)
    out[:, 0] = points_mm[:, 0] * pr + cx
    out[:, 1] = -points_mm[:, 1] * pr + cy
    return out


def render_print(
    size_mm: float,
    shifts_um: dict[str, tuple[float, float]],
    defects: dict,
) -> np.ndarray:
    """在无旋转、基准尺度的渲染画布上画理想色标 + 色版平移。"""
    n = int(size_mm * RENDER_DPI / MM_PER_INCH)
    img = np.full((n, n, 3), 255, np.uint8)
    pr = RENDER_DPI / MM_PER_INCH

    def shift_of(key: str) -> tuple[float, float]:
        dx, dy = shifts_um.get(key, (0.0, 0.0))
        return dx / 1000.0, dy / 1000.0

    k_color = _srgb_to_bgr(INK_SRGB["K"])
    oh = FRAME_OUTER_SIDE_MM / 2.0   # 外边界 ±4.85mm（与 spec/检测器一致）
    lh = FRAME_LINEWIDTH_MM / 2.0
    # 四条描边带（填充矩形），外边对齐 ±oh；避免 polylines 线宽方向的不确定性
    hseg = np.array([[-oh, lh], [oh, lh], [oh, -lh], [-oh, -lh]])
    vseg = np.array([[-lh, oh], [lh, oh], [lh, -oh], [-lh, -oh]])
    for vy in (oh, -oh):
        seg = hseg.copy(); seg[:, 1] += vy - (lh if vy > 0 else -lh)
        cv2.fillPoly(img, [_mm_to_printpx(seg, img.shape, RENDER_DPI).astype(np.int32)],
                     k_color, lineType=cv2.LINE_AA)
    for vx in (oh, -oh):
        seg = vseg.copy(); seg[:, 0] += vx - (lh if vx > 0 else -lh)
        cv2.fillPoly(img, [_mm_to_printpx(seg, img.shape, RENDER_DPI).astype(np.int32)],
                     k_color, lineType=cv2.LINE_AA)

    # 角点方块（K）。中心 ±4.0mm；方块外边相对框外边内缩 PIP_INSET
    for name, (cx, cy) in PIP_CENTERS_MM.items():
        side_half = PIP_SIDE_MM / 2.0
        if cx > 0:
            x_outer, x_inner = oh - PIP_INSET_MM, oh - PIP_INSET_MM - PIP_SIDE_MM
        else:
            x_outer, x_inner = -oh + PIP_INSET_MM, -oh + PIP_INSET_MM + PIP_SIDE_MM
        if cy > 0:
            y_outer, y_inner = oh - PIP_INSET_MM, oh - PIP_INSET_MM - PIP_SIDE_MM
        else:
            y_outer, y_inner = -oh + PIP_INSET_MM, -oh + PIP_INSET_MM + PIP_SIDE_MM
        poly = np.array(
            [[x_outer, y_outer], [x_inner, y_outer], [x_inner, y_inner], [x_outer, y_inner]]
        )
        # 几何中心（应等于 spec 的 ±4.0mm）
        gcx, gcy = (x_outer + x_inner) / 2, (y_outer + y_inner) / 2
        assert abs(gcx - cx) < 1e-9 and abs(gcy - cy) < 1e-9
        qp = _mm_to_printpx(poly, img.shape, RENDER_DPI).astype(np.int32)
        cv2.fillConvexPoly(img, qp, k_color, lineType=cv2.LINE_AA)

    # 色版圆点（含 K 的两个控制点）
    r_px = DOT_DIAMETER_MM / 2.0 * pr
    for key, (nx, ny) in DOT_NOMINAL_MM.items():
        if "missing_plate" in defects and key == defects["missing_plate"]:
            continue
        # K1 跟随 K 版平移；其它点用自己的版键
        plate = "K" if key == "K1" else key
        sx, sy = shifts_um.get(plate, (0.0, 0.0))
        px, py = _mm_to_printpx(
            np.array([[nx + sx / 1000.0, ny + sy / 1000.0]]), img.shape, RENDER_DPI
        )[0]
        cv2.circle(img, (int(round(px)), int(round(py))), int(round(r_px)),
                   _srgb_to_bgr(INK_SRGB[plate]), -1, lineType=cv2.LINE_AA)

    # 标记残缺：在渲染空间遮挡某点的一部分（白色覆盖）
    if "occlude" in defects:
        occ = defects["occlude"]
        key = occ["dot"]
        plate = "K" if key == "K1" else key
        nx, ny = DOT_NOMINAL_MM[key]
        sx, sy = shifts_um.get(plate, (0.0, 0.0))
        px, py = _mm_to_printpx(
            np.array([[nx + sx / 1000.0, ny + sy / 1000.0]]), img.shape, RENDER_DPI
        )[0]
        # 覆盖圆偏移到点边缘，遮掉约 occ['fraction']
        off = r_px * (1.0 - occ["fraction"] / 2.0)
        a = math.radians(occ.get("angle_deg", 35))
        cv2.circle(
            img,
            (int(round(px + off * math.cos(a))), int(round(py - off * math.sin(a)))),
            int(round(r_px * 1.05)), (255, 255, 255), -1,
        )

    return img


def warp_to_scan(print_img: np.ndarray, angle_deg: float, dpi: float,
                 translation_px: tuple[float, float]) -> np.ndarray:
    size_mm = print_img.shape[0] * MM_PER_INCH / RENDER_DPI
    scan_side = int(round(size_mm * dpi / MM_PER_INCH * 1.18 + abs(translation_px[0]) + 60))
    # 画布留边，避免旋转后裁切
    margin_px = max(scan_side // 5, 40)
    canvas = scan_side + 2 * margin_px
    t = (translation_px[0] + canvas / 2.0 - scan_side / 2.0,
         translation_px[1] + canvas / 2.0 - scan_side / 2.0)
    M = build_affine(angle_deg, dpi, t)  # mm(print) -> scan px

    # printpx(原点中心, y-down) -> mm
    pr = RENDER_DPI / MM_PER_INCH
    h, w = print_img.shape[:2]
    A_print = np.array([[pr, 0, w / 2.0], [0, -pr, h / 2.0], [0, 0, 1.0]])
    W = M @ np.linalg.inv(A_print)
    warped = cv2.warpAffine(
        print_img, W[:2, :], (canvas, canvas),
        flags=cv2.INTER_AREA if dpi < RENDER_DPI else cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    return warped


def add_scan_artifacts(img: np.ndarray, defects: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.copy()

    # 传感器污点：扫描空间的暗斑/彩色斑（与色标无关，应被检测拒绝）
    for spot in defects.get("spots", []):
        x = int(rng.uniform(0.08, 0.92) * out.shape[1])
        y = int(rng.uniform(0.08, 0.92) * out.shape[0])
        r = int(spot.get("r_px", rng.integers(4, 14)))
        if spot.get("color") == "dark":
            col = (int(rng.integers(20, 70)),) * 3
        else:
            col = tuple(int(v) for v in _srgb_to_bgr(
                tuple(int(v) for v in rng.integers(0, 200, size=3))))
        cv2.circle(out, (x, y), r, col, -1, lineType=cv2.LINE_AA)

    # 模糊 + 噪声
    k = defects.get("blur_ksize", 3)
    if k and k > 1:
        out = cv2.GaussianBlur(out, (k, k), k / 3.0)
    sigma = defects.get("noise_sigma", 2.5)
    if sigma:
        noise = rng.normal(0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def build_case(name: str, shifts_um, angle_deg, dpi, translation_px, defects,
               out_dir: Path, seed: int = 0) -> dict:
    print_img = render_print(16.0, shifts_um, defects)
    scan = warp_to_scan(print_img, angle_deg, dpi, translation_px)
    scan = add_scan_artifacts(scan, defects, seed or abs(hash(name)) % (2**31))

    gt = {
        "name": name,
        "spec_version": SPEC_VERSION,
        "base_dpi": BASE_DPI,
        "scan": {
            "angle_deg": angle_deg,
            "dpi": dpi,
            "translation_px": list(translation_px),
        },
        "true_offsets_um": {k: list(shifts_um.get(k, (0.0, 0.0))) for k in ("C", "M", "Y", "K")},
        "defects": (
            {k: v for k, v in defects.items() if k != "missing_plate"}
            | ({"missing_plate": defects["missing_plate"]}
               if "missing_plate" in defects else {})
        ),
    }
    suffix = ".jpg" if defects.get("jpeg_quality") else ".png"
    img_path = out_dir / f"{name}{suffix}"
    if suffix == ".jpg":
        cv2.imwrite(str(img_path), scan,
                    [cv2.IMWRITE_JPEG_QUALITY, defects["jpeg_quality"]])
    else:
        cv2.imwrite(str(img_path), scan)
    gt["image"] = img_path.name
    (out_dir / f"{name}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2))
    return gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "fixtures"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        build_case("01_baseline", {}, 0.0, 200.0, (0, 0), {}, out_dir, 1),
        build_case("02_translations",
                   {"C": (120.0, -80.0), "M": (-60.0, 45.0), "Y": (30.0, 200.0)},
                   0.0, 200.0, (0, 0), {}, out_dir, 2),
        build_case("03_rotated",
                   {"C": (100.0, 50.0), "M": (-40.0, -90.0), "Y": (15.0, 25.0)},
                   2.5, 200.0, (6, -4), {}, out_dir, 3),
        build_case("04_rotated_neg",
                   {"C": (-70.0, 30.0), "Y": (180.0, -120.0)},
                   -3.2, 200.0, (-5, 7), {}, out_dir, 4),
        build_case("05_dpi_300",
                   {"C": (90.0, -60.0), "M": (20.0, 40.0)},
                   1.2, 300.0, (0, 0), {}, out_dir, 5),
        build_case("06_dpi_150",
                   {"C": (140.0, 70.0), "M": (-55.0, -35.0), "Y": (10.0, -15.0)},
                   -1.8, 150.0, (4, 3), {}, out_dir, 6),
        build_case("07_missing_c",
                   {"M": (50.0, 50.0), "Y": (-30.0, 20.0)},
                   0.5, 200.0, (0, 0), {"missing_plate": "C"}, out_dir, 7),
        build_case("08_spots",
                   {"C": (60.0, 60.0), "M": (-25.0, 80.0), "Y": (0.0, 0.0)},
                   0.3, 200.0, (0, 0),
                   {"spots": [{"color": "dark"}, {"color": "color"}, {"color": "dark"}]},
                   out_dir, 8),
        build_case("09_occluded_m",
                   {"C": (40.0, -50.0), "M": (-100.0, 30.0), "Y": (70.0, 70.0)},
                   0.8, 200.0, (0, 0),
                   {"occlude": {"dot": "M", "fraction": 0.45}}, out_dir, 9),
        build_case("10_all_defects",
                   {"C": (110.0, -70.0), "Y": (45.0, 90.0)},
                   2.0, 240.0, (3, -2),
                   {"missing_plate": "M",
                    "spots": [{"color": "dark"}, {"color": "color"}],
                    "occlude": {"dot": "Y", "fraction": 0.3},
                    "jpeg_quality": 70},
                   out_dir, 10),
    ]
    manifest = {
        "spec_version": SPEC_VERSION,
        "description": "随项目提供的合成套准色标扫描图；json 为 ground truth，仅供离线验算",
        "cases": [{k: c[k] for k in ("name", "image", "scan", "true_offsets_um", "defects")}
                  for c in cases],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"wrote {len(cases)} cases -> {out_dir}")


if __name__ == "__main__":
    main()
