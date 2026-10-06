"""Draw the detector's evidence overlay on the canonical (paper-corrected)
image.  This is the same geometry the React/Canvas UI uses, rendered server
side for downloads and tests.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import markspec as spec
from .detect import (BLACK_S_MAX, BLACK_V_MAX, Observation, _plate_mask,
                     _ring_centre, _ruler_baseline_px,
                     _similarity_from_fiducials, _find_fiducials)

_OVERLAY_BGR = {
    "cyan": (220, 170, 0), "magenta": (120, 0, 215),
    "yellow": (0, 200, 225), "black": (60, 60, 60),
}
STATE_COLOR = {"ok": (60, 170, 60), "partial": (0, 170, 230),
               "missing": (40, 40, 230)}


def canonical_warp(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    dark = _plate_mask(hsv, "black")
    fids = _find_fiducials(dark)
    cw, ch = spec.chart_size_px(spec.NATIVE_PPI)
    if len(fids) < 3:
        return cv2.resize(img_bgr, (cw, ch)), np.eye(2, 3, dtype=np.float64)
    m, _ = _similarity_from_fiducials(fids)
    return cv2.warpAffine(img_bgr, m, (cw, ch), flags=cv2.INTER_CUBIC,
                          borderValue=(255, 255, 255)), m


def draw_overlay(canon_bgr: np.ndarray, result: dict) -> np.ndarray:
    vis = canon_bgr.copy()
    if len(vis.shape) == 2 or vis.shape[2] == 1:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    points = spec.grid_points_px()
    # Mark cells + state rings.
    for o in result.get("observations", []):
        gx, gy = points[o["row"] * spec.GRID_COLS + o["col"]]
        if o["plate"] != "black":
            ang = math.radians(spec.PLATE_ANGLES_DEG[o["plate"]])
            orbit = spec.mm_to_px(spec.PLATE_ORBIT_MM)
            gx = gx + math.cos(ang) * orbit
            gy = gy + math.sin(ang) * orbit
        color = STATE_COLOR.get(o["state"], (160, 160, 160))
        r = int(spec.mm_to_px(0.55))
        cv2.circle(vis, (int(round(gx)), int(round(gy))), r, color, 1,
                   lineType=cv2.LINE_AA)

    # Plate shift arrows at the first mark of each plate (readable legend).
    for plate, pr in result.get("plates", {}).items():
        gx, gy = points[0]
        if plate != "black":
            ang = math.radians(spec.PLATE_ANGLES_DEG[plate])
            orbit = spec.mm_to_px(spec.PLATE_ORBIT_MM)
            gx += math.cos(ang) * orbit
            gy += math.sin(ang) * orbit
        dx, dy = pr.get("dx_um"), pr.get("dy_um")
        if dx is None or dy is None:
            cv2.putText(vis, f"{plate}:MISSING", (int(gx) + 8, int(gy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 230), 1)
            continue
        px_per_um = spec.NATIVE_PPI / spec.MM_PER_INCH / 1000.0
        ex = gx + dx * px_per_um * 3  # x3 visual exaggeration
        ey = gy + dy * px_per_um * 3
        cv2.arrowedLine(vis, (int(gx), int(gy)), (int(ex), int(ey)),
                        _OVERLAY_BGR[plate], 2, line_type=cv2.LINE_AA,
                        tipLength=0.3)
        cv2.putText(vis, f"{plate} {dx:+.0f},{dy:+.0f}um",
                    (int(ex) + 4, int(ey)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, _OVERLAY_BGR[plate], 1)

    # Ruler baseline check.
    ox, ry = _ruler_baseline_px()
    cv2.line(vis, (int(ox), int(ry)),
             (int(ox + spec.mm_to_px(spec.RULER_LENGTH_MM)), int(ry)),
             (0, 200, 0), 1)
    cv2.putText(vis, "ruler 30mm", (int(ox), int(ry) + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 130, 0), 1)
    return vis
