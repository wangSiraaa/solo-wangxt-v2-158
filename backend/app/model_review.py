"""Offline second-opinion model (never a control signal).

Given the detector's result and the warped canonical image, this module
produces an *advisory* cross-check using a deliberately independent, simple
method: in each expected mark cell, the plate's mini-cross horizontal/vertical
arms are located by ink-weighted column/row profiles (no Hough, no template
fit), and the plate translation is the median cell-wise displacement.

Its role is to catch gross disagreement (swapped axes, scale blunders); it
does not override the OpenCV result, and nothing in this service can send a
command to a press.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import markspec as spec
from .detect import _plate_mask

#: If the advisory median differs from the detector by more than this, the
#: result is flagged for human review (never auto-accepted or corrected).
DISAGREE_UM = 150.0
MODEL_NAME = "offline-arm-profile-review-1.0.0"


def _arm_centres(cell: np.ndarray):
    """Return (x, y) centre of a mini-cross from ink column/row profiles."""
    cols = (cell > 0).sum(axis=0).astype(np.float64)
    rows = (cell > 0).sum(axis=1).astype(np.float64)

    def centre(prof):
        if prof.max() < 2:
            return None
        on = prof > prof.max() * 0.25
        idx = np.where(on)[0]
        # Central run around the profile's midpoint.
        mid = len(prof) // 2
        runs, cur = [], [idx[0]]
        for a, b in zip(idx, idx[1:]):
            if b - a == 1:
                cur.append(b)
            else:
                runs.append(cur); cur = [b]
        runs.append(cur)
        run = min(runs, key=lambda rr: abs(np.mean(rr) - mid))
        r = np.array(run, dtype=np.float64)
        w = prof[run]
        return float((r * w).sum() / w.sum())

    return centre(cols), centre(rows)


def review(canon_bgr: np.ndarray, detector: dict) -> dict:
    hsv = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2HSV)
    masks = {p: _plate_mask(hsv, p) for p in spec.PLATES}
    points = spec.grid_points_px()
    px_per_um = spec.NATIVE_PPI / spec.MM_PER_INCH / 1000.0
    half = int(spec.mm_to_px(spec.RING_RADIUS_MM + 0.7))

    out: dict[str, dict] = {}
    for plate in spec.PLATES:
        dxs, dys = [], []
        for ri in range(spec.GRID_ROWS):
            for ci in range(spec.GRID_COLS):
                gx, gy = points[ri * spec.GRID_COLS + ci]
                if plate == "black":
                    nx, ny = gx, gy
                else:
                    ang = math.radians(spec.PLATE_ANGLES_DEG[plate])
                    orbit = spec.mm_to_px(spec.PLATE_ORBIT_MM)
                    nx, ny = gx + math.cos(ang) * orbit, gy + math.sin(ang) * orbit
                x0, y0 = max(0, int(nx) - half), max(0, int(ny) - half)
                x1 = min(masks[plate].shape[1], int(nx) + half)
                y1 = min(masks[plate].shape[0], int(ny) + half)
                cell = masks[plate][y0:y1, x0:x1]
                cx, cy = _arm_centres(cell)
                if cx is not None:
                    dxs.append((cx + x0 - nx) / px_per_um)
                if cy is not None:
                    dys.append((cy + y0 - ny) / px_per_um)
        rec = {"n": len(dxs), "available": len(dxs) >= 3}
        if dxs:
            rec["dx_um"] = round(float(np.median(dxs)), 1)
            rec["dy_um"] = round(float(np.median(dys)), 1)
        out[plate] = rec

    disagreements: list[dict] = []
    for plate, rec in out.items():
        det = detector.get("plates", {}).get(plate)
        if not det or not rec["available"] or det["status"] == "missing":
            continue
        for axis, det_v, adv_v in (("x", det["dx_um"], rec.get("dx_um")),
                                   ("y", det["dy_um"], rec.get("dy_um"))):
            if det_v is None or adv_v is None:
                continue
            if abs(det_v - adv_v) > DISAGREE_UM:
                disagreements.append({"plate": plate, "axis": axis,
                                      "detector_um": det_v,
                                      "advisory_um": adv_v})
    return {
        "model": MODEL_NAME,
        "mode": "offline_advisory_only",
        "control_authority": "none — cannot drive any press",
        "disagreement_threshold_um": DISAGREE_UM,
        "plates": out,
        "disagreements": disagreements,
        "verdict": "review_required" if disagreements else "consistent",
    }
