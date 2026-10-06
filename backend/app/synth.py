"""Synthetic PRT-1 chart & scanned-image generator.

Two renderers:

* :func:`render_chart` draws the *ideal* vector chart at a chosen PPI, with
  known per-plate translations in micrometres.  Files land in
  ``backend/data/`` and are the only detector inputs the project uses.
* :func:`simulate_scan` resamples the rendered chart with a known rotation,
  scan resolution, Gaussian noise, speckle (printer dirt) and optional
  missing/mutilated marks.  Ground-truth parameters are written to a sidecar
  JSON so calibration checks are auditable.

Run ``python -m app.synth generate`` to regenerate the shipped dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import markspec as spec

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _warp_around(img: np.ndarray, m: np.ndarray, out_wh: tuple[int, int],
                 fill: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    return cv2.warpAffine(img, m, out_wh, flags=cv2.INTER_CUBIC,
                          borderValue=fill)


@dataclass
class PlateShift:
    """Known plate translation, micrometres in chart-local axes."""
    plate: str
    dx_um: float
    dy_um: float


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------

def render_chart(
    ppi: float = spec.NATIVE_PPI,
    plate_shifts: list[PlateShift] | None = None,
) -> np.ndarray:
    """Render an ideal (unscanned) PRT-1 chart as BGR uint8."""
    w, h = spec.chart_size_px(ppi)
    canvas = np.full((h, w, 3), 255, np.uint8)

    ring_r = spec.mm_to_px(spec.RING_RADIUS_MM, ppi)
    ring_t = max(1, int(round(spec.mm_to_px(spec.RING_THICKNESS_MM, ppi))))
    orbit = spec.mm_to_px(spec.PLATE_ORBIT_MM, ppi)
    span = spec.mm_to_px(spec.PLATE_CROSS_SPAN_MM, ppi)
    thick = max(1, int(round(spec.mm_to_px(
        spec.PLATE_CROSS_THICKNESS_MM, ppi))))

    shift_by_plate = {s.plate: s for s in (plate_shifts or [])}

    def shift_px(plate: str) -> tuple[float, float]:
        s = shift_by_plate.get(plate)
        if s is None:
            return 0.0, 0.0
        k = ppi / spec.MM_PER_INCH / 1000.0
        return s.dx_um * k, s.dy_um * k

    # Each plate on its own layer; per-pixel minimum keeps every ink visible.
    layers: dict[str, np.ndarray] = {}
    for plate in spec.PLATES:
        layer = np.full_like(canvas, 255)
        rgb = spec.PLATE_COLORS[plate].render_rgb
        bgr = (rgb[2], rgb[1], rgb[0])
        dx, dy = shift_px(plate)
        if plate == "black":
            for (gx, gy) in spec.grid_points_px(ppi):
                cx, cy = gx + dx, gy + dy
                cv2.circle(layer, (int(round(cx)), int(round(cy))),
                           int(round(ring_r)), bgr, ring_t,
                           lineType=cv2.LINE_AA)
            layers[plate] = layer
            continue
        ang = math.radians(spec.PLATE_ANGLES_DEG[plate])
        ux, uy = math.cos(ang), math.sin(ang)
        for (gx, gy) in spec.grid_points_px(ppi):
            # Mini-cross centred on the shared orbit (plate shift applies).
            px_ = gx + dx + ux * orbit
            py_ = gy + dy + uy * orbit
            ic = (int(round(px_)), int(round(py_)))
            cv2.line(layer,
                     (int(round(px_ - span / 2)), ic[1]),
                     (int(round(px_ + span / 2)), ic[1]),
                     bgr, thick, lineType=cv2.LINE_AA)
            cv2.line(layer,
                     (ic[0], int(round(py_ - span / 2))),
                     (ic[0], int(round(py_ + span / 2))),
                     bgr, thick, lineType=cv2.LINE_AA)
        layers[plate] = layer

    merged = np.full_like(canvas, 255)
    for plate in spec.PLATES:
        merged = np.minimum(merged, layers[plate])

    # Black corner fiducials + ruler anchor the *paper* frame.  They are
    # reference geometry, never shifted with a colour plate.
    pitch = spec.mm_to_px(spec.GRID_PITCH_MM, ppi)
    pad = spec.mm_to_px(spec.RING_RADIUS_MM + 0.6, ppi)
    ox = spec.mm_to_px(spec.GRID_MARGIN_MM, ppi) + pad
    oy = ox
    grid_w = (spec.GRID_COLS - 1) * pitch
    grid_h = (spec.GRID_ROWS - 1) * pitch
    off = spec.mm_to_px(spec.GRID_MARGIN_MM, ppi)
    fid_r = spec.mm_to_px(spec.FIDUCIAL_DIAMETER_MM, ppi) / 2
    for (fx, fy) in ((ox - off, oy - off), (ox + grid_w + off, oy - off),
                     (ox - off, oy + grid_h + off),
                     (ox + grid_w + off, oy + grid_h + off)):
        cv2.circle(merged, (int(round(fx)), int(round(fy))),
                   int(round(fid_r)), (45, 45, 45), -1, lineType=cv2.LINE_AA)

    # Ruler: baseline below the grid, 1 mm minor / 5 mm major ticks,
    # end brackets defining a known 30 mm reference span.
    ry = oy + grid_h + spec.mm_to_px(spec.GRID_MARGIN_MM + 5, ppi)
    x0 = ox
    length = spec.mm_to_px(spec.RULER_LENGTH_MM, ppi)
    tk = max(1, int(round(spec.mm_to_px(spec.RULER_THICKNESS_MM, ppi))))
    cv2.line(merged, (int(x0), int(ry)), (int(x0 + length), int(ry)),
             (30, 30, 30), tk)
    minor = spec.mm_to_px(spec.RULER_TICK_PITCH_MM, ppi)
    for i in range(int(spec.RULER_LENGTH_MM) + 1):
        x = x0 + i * minor
        tick_h = spec.mm_to_px(1.6 if i % 5 == 0 else 0.9, ppi)
        cv2.line(merged, (int(round(x)), int(ry)),
                 (int(round(x)), int(ry - tick_h)),
                 (30, 30, 30), tk)
    bh = spec.mm_to_px(1.2, ppi)
    for x in (x0, x0 + length):
        cv2.line(merged, (int(x), int(ry)), (int(x), int(ry + bh)),
                 (30, 30, 30), tk)
    return merged.astype(np.uint8)


# ---------------------------------------------------------------------------
# Scan simulation
# ---------------------------------------------------------------------------

@dataclass
class ScanParams:
    ppi: float = spec.NATIVE_PPI
    rotation_deg: float = 0.0
    scan_ppi: float = 0.0  # 0 => same as source ppi
    noise_sigma: float = 4.0
    blur_sigma: float = 0.8
    dirt_spots: int = 0
    #: (row, col) grid positions whose dots are partially erased (one plate
    #  dot kept, so the detector returns partial observations).
    damaged_marks: list[tuple[int, int]] = field(default_factory=list)
    #: grid positions erased entirely (ink missing => no candidate there).
    missing_marks: list[tuple[int, int]] = field(default_factory=list)
    plate_shifts: list[PlateShift] = field(default_factory=list)


def _erase_mark(img: np.ndarray, ppi: float, gx: float, gy: float) -> None:
    r = spec.mm_to_px(spec.RING_RADIUS_MM + 0.45, ppi)
    cv2.circle(img, (int(round(gx)), int(round(gy))), int(round(r)),
               (255, 255, 255), -1)


def simulate_scan(params: ScanParams, seed: int = 0) -> np.ndarray:
    """Return a scan *at params.scan_ppi*.  Output pixel size differs from
    the source chart when the scan resolution differs — that is the real
    scale signal the detector must recover."""
    rng = np.random.default_rng(seed)
    chart = render_chart(params.ppi, params.plate_shifts).astype(np.float32)

    # Damage marks at source resolution (real ink loss, not threshold noise).
    points = spec.grid_points_px(params.ppi)
    for (r, c) in params.missing_marks:
        gx, gy = points[r * spec.GRID_COLS + c]
        _erase_mark(chart, params.ppi, gx, gy)
    for (r, c) in params.damaged_marks:
        gx, gy = points[r * spec.GRID_COLS + c]
        _erase_mark(chart, params.ppi, gx, gy)
        # Re-draw only the black ring stub + cyan horizontal arm: partial
        # information on purpose (cyan x readable, y missing; M/Y/K absent).
        ring_r = spec.mm_to_px(spec.RING_RADIUS_MM, params.ppi)
        ring_t = max(1, int(spec.mm_to_px(spec.RING_THICKNESS_MM,
                                          params.ppi)))
        cv2.ellipse(chart, (int(gx), int(gy)), (int(ring_r), int(ring_r)),
                    0, 200, 340, (45, 45, 45), ring_t, lineType=cv2.LINE_AA)
        orbit = spec.mm_to_px(spec.PLATE_ORBIT_MM, params.ppi)
        span = spec.mm_to_px(spec.PLATE_CROSS_SPAN_MM, params.ppi)
        thick = max(1, int(spec.mm_to_px(spec.PLATE_CROSS_THICKNESS_MM,
                                         params.ppi)))
        s = next((p for p in params.plate_shifts
                  if p.plate == "cyan"), None)
        k = params.ppi / spec.MM_PER_INCH / 1000.0
        dxp = s.dx_um * k if s else 0.0
        dyp = s.dy_um * k if s else 0.0
        px_ = gx + dxp + orbit
        py_ = gy + dyp
        cv2.line(chart,
                 (int(px_ - span / 2), int(py_)),
                 (int(px_ + span / 2), int(py_)),
                 (220, 170, 0), thick, lineType=cv2.LINE_AA)

    h0, w0 = chart.shape[:2]
    scan_ppi = params.scan_ppi or params.ppi
    kscale = scan_ppi / params.ppi

    # Rotate in an enlarged white canvas (so corners survive), then resample
    # to scan resolution.  OpenCV positive angle = CCW; we define truth the
    # same way, and the detector reports this same convention.
    pad = int(max(w0, h0) * 0.12)
    canvas = np.full((h0 + 2 * pad, w0 + 2 * pad, 3), 255, np.float32)
    canvas[pad:pad + h0, pad:pad + w0] = chart
    H, W = canvas.shape[:2]
    m = cv2.getRotationMatrix2D((W / 2, H / 2), params.rotation_deg, 1.0)
    rot = _warp_around(canvas, m, (W, H))
    # Metadata rotation uses the visual image-coordinate convention (y-down),
    # which is the negation of cv2's mathematical positive-CCW angle.
    visual_rotation_deg = -params.rotation_deg

    sw, sh = max(16, int(round(W * kscale))), max(16, int(round(H * kscale)))
    interp = cv2.INTER_AREA if kscale < 1 else cv2.INTER_CUBIC
    scan = cv2.resize(rot, (sw, sh), interpolation=interp)

    img = cv2.GaussianBlur(scan, (0, 0), max(0.3, params.blur_sigma
                                             * kscale))
    img += rng.normal(0, params.noise_sigma, img.shape)

    for _ in range(params.dirt_spots):
        x = int(rng.integers(0, sw))
        y = int(rng.integers(0, sh))
        radius = max(1, int(rng.integers(1, max(2, int(5 * kscale) + 1))))
        shade = float(rng.integers(40, 140))
        cv2.circle(img, (x, y), radius, (shade, shade, shade), -1)

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _write_png(path: Path, img: np.ndarray, meta: dict) -> None:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.write_bytes(buf.tobytes())
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))


def generate_dataset(out_dir: Path = DATA_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    ref = render_chart()
    p = out_dir / "prt1_reference_300ppi.png"
    _write_png(p, ref, {"format": spec.FORMAT_ID, "kind": "reference",
                        "ppi": spec.NATIVE_PPI, "plate_shifts_um": {}})
    written.append(p)

    shifts = [PlateShift("cyan", 250, -150), PlateShift("magenta", -180, 90),
              PlateShift("yellow", 0, 300)]
    shifted = render_chart(spec.NATIVE_PPI, shifts)
    p = out_dir / "prt1_shifted_300ppi.png"
    _write_png(p, shifted, {
        "format": spec.FORMAT_ID, "kind": "chart", "ppi": spec.NATIVE_PPI,
        "plate_shifts_um": {s.plate: {"dx": s.dx_um, "dy": s.dy_um}
                            for s in shifts}})
    written.append(p)

    # Rotated 1.7 deg, scanned at 240 PPI (known resolution change).
    sp = ScanParams(ppi=spec.NATIVE_PPI, rotation_deg=1.7, scan_ppi=240.0,
                    noise_sigma=5, blur_sigma=1.0, dirt_spots=12,
                    plate_shifts=shifts)
    img = simulate_scan(sp, seed=42)
    p = out_dir / "prt1_scan_rot1p7_240ppi.png"
    _write_png(p, img, {
        "format": spec.FORMAT_ID, "kind": "scan", "scan_ppi": 240.0,
        "source_ppi": spec.NATIVE_PPI, "rotation_deg": -1.7,
        "plate_shifts_um": {s.plate: {"dx": s.dx_um, "dy": s.dy_um}
                            for s in shifts}})
    written.append(p)

    # Pure paper transform, zero plate shifts: skew/scale must NOT leak into
    # plate offsets.
    sp0 = ScanParams(ppi=spec.NATIVE_PPI, rotation_deg=2.3, scan_ppi=260.0,
                     noise_sigma=4, blur_sigma=0.9, dirt_spots=8)
    img0 = simulate_scan(sp0, seed=11)
    p = out_dir / "prt1_scan_paperonly.png"
    _write_png(p, img0, {
        "format": spec.FORMAT_ID, "kind": "scan", "scan_ppi": 260.0,
        "source_ppi": spec.NATIVE_PPI, "rotation_deg": -2.3,
        "plate_shifts_um": {pl: {"dx": 0, "dy": 0} for pl in spec.PLATES}})
    written.append(p)

    # Defective scan: missing + damaged marks + heavy dirt => candidates with
    # widened ranges, never fake-precise microns.
    sp2 = ScanParams(ppi=spec.NATIVE_PPI, rotation_deg=-0.6, scan_ppi=306.0,
                     noise_sigma=7, blur_sigma=1.2, dirt_spots=40,
                     damaged_marks=[(1, 2), (3, 4), (0, 0)],
                     missing_marks=[(2, 1), (2, 2)],
                     plate_shifts=[PlateShift("cyan", 120, 0),
                                   PlateShift("yellow", -260, 140)])
    img2 = simulate_scan(sp2, seed=7)
    p = out_dir / "prt1_scan_defective.png"
    _write_png(p, img2, {
        "format": spec.FORMAT_ID, "kind": "scan", "scan_ppi": 306.0,
        "source_ppi": spec.NATIVE_PPI, "rotation_deg": 0.6,
        "damaged_marks": sp2.damaged_marks,
        "missing_marks": sp2.missing_marks,
        "plate_shifts_um": {s.plate: {"dx": s.dx_um, "dy": s.dy_um}
                            for s in sp2.plate_shifts}})
    written.append(p)

    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate"])
    args = ap.parse_args()
    if args.cmd == "generate":
        for p in generate_dataset():
            print(p)


if __name__ == "__main__":  # pragma: no cover
    main()
