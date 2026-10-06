"""PRT-1 detector: locate marks, separate paper frame from plate shifts.

Pipeline (all numbers are auditable in the returned ``provenance``):

1. Classify ink by HSV into per-plate masks (black = low saturation *and*
   low value, so coloured dirt cannot enter CMY masks).
2. Locate the four black corner fiducials.  Their known mm spacing gives the
   *paper* similarity transform (rotation + uniform scale + translation).
3. Warp into a canonical 300 PPI chart frame, then verify scale independently
   against the ruler ticks (1 mm pitch / 30 mm bracket span).
4. Read each plate's arm at every grid intersection.  Horizontal arms give
   the x reading, vertical arms the y reading independently, so a mark with
   one arm erased still yields one good axis instead of a fake centroid.
5. Remove the common residual similarity (rotation/scale shared by *all*
   plates) before estimating per-plate translation.  Global skew therefore
   cancels and can never be reported as a colour-plate error.
6. Return candidates with 95% confidence ranges.  Missing ink / damaged
   marks / excessive dirt widen or void the range instead of producing a
   sharp but unjustified micron value.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from . import markspec as spec

DETECTOR_VERSION = "prt1-detector-1.0.0"

# HSV bands (OpenCV: H 0-179).  Tuned on the shipped synthetic inks, not on
# arbitrary press output.
_HSV_BANDS = {
    "cyan": ((80, 70, 60), (108, 255, 255)),
    "magenta": ((140, 60, 60), (178, 255, 255)),
    "yellow": ((18, 80, 80), (38, 255, 255)),
}
BLACK_V_MAX = 110
BLACK_S_MAX = 90


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    plate: str
    row: int
    col: int
    x_um: float | None
    y_um: float | None = None
    x_sigma_um: float = 0.0
    y_sigma_um: float = 0.0
    arm_h: float = 0.0  # horizontal-arm coverage 0..1 (drives x reading)
    arm_v: float = 0.0  # vertical-arm coverage 0..1 (drives y reading)
    dirt_ratio: float = 0.0
    state: str = "ok"  # ok | partial | missing


@dataclass
class PlateResult:
    plate: str
    status: str  # ok | candidate | missing
    dx_um: float | None = None
    dy_um: float | None = None
    x_lo: float | None = None
    x_hi: float | None = None
    y_lo: float | None = None
    y_hi: float | None = None
    confidence: float = 0.0
    n_x: int = 0
    n_y: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    format_id: str
    detected: bool
    width_px: int
    height_px: int
    declared_scan_ppi: float | None
    measured_ppi: float | None
    frame_rotation_deg: float
    frame_scale: float
    frame_quality: float
    residual_rotation_deg: float
    residual_scale: float
    plates: dict[str, PlateResult]
    observations: list[Observation]
    provenance: dict
    run_id: str = ""
    status: str = "ok"

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "format_id": self.format_id,
            "detected": self.detected,
            "status": self.status,
            "image": {"width_px": self.width_px, "height_px": self.height_px,
                      "declared_scan_ppi": self.declared_scan_ppi,
                      "measured_ppi": _round(self.measured_ppi)},
            "frame": {
                "rotation_deg": round(self.frame_rotation_deg, 3),
                "scale_vs_native": round(self.frame_scale, 5),
                "quality": round(self.frame_quality, 3),
                "residual_rotation_deg": round(self.residual_rotation_deg, 4),
                "residual_scale": round(self.residual_scale, 6),
            },
            "plates": {
                p: {
                    "plate": p, "status": r.status,
                    "dx_um": _round(r.dx_um), "dy_um": _round(r.dy_um),
                    "x_range_um": [_round(r.x_lo), _round(r.x_hi)],
                    "y_range_um": [_round(r.y_lo), _round(r.y_hi)],
                    "confidence": round(r.confidence, 3),
                    "n_x_readings": r.n_x, "n_y_readings": r.n_y,
                    "notes": r.notes,
                }
                for p, r in self.plates.items()
            },
            "observations": [
                {"plate": o.plate, "row": o.row, "col": o.col, "state": o.state,
                 "x_um": _round(o.x_um), "y_um": _round(o.y_um),
                 "x_sigma_um": round(o.x_sigma_um, 1),
                 "y_sigma_um": round(o.y_sigma_um, 1),
                 "arm_h": round(o.arm_h, 2), "arm_v": round(o.arm_v, 2),
                 "dirt_ratio": round(o.dirt_ratio, 2)}
                for o in self.observations
            ],
            "provenance": self.provenance,
        }


def _round(v):
    return None if v is None else round(float(v), 1)


# ---------------------------------------------------------------------------
# Image loading / masks
# ---------------------------------------------------------------------------

def _plate_mask(hsv: np.ndarray, plate: str) -> np.ndarray:
    if plate in _HSV_BANDS:
        lo, hi = _HSV_BANDS[plate]
        return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    s, v = hsv[..., 1], hsv[..., 2]
    return (((s < BLACK_S_MAX) & (v < BLACK_V_MAX))
            .astype(np.uint8) * 255)


# ---------------------------------------------------------------------------
# Step 2: corner fiducials -> paper similarity
# ---------------------------------------------------------------------------

def _find_fiducials(dark: np.ndarray) -> list[tuple[float, float, float]]:
    """Solid dark circles: (x, y, radius_px).  Crosses/ticks fail the
    circularity + fill test, so only the four corner dots should survive."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    solid = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=1)
    cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cand: list[tuple[float, float, float]] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 60:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circ = 4 * math.pi * area / (peri * peri)
        (x, y), r = cv2.minEnclosingCircle(c)
        fill = area / (math.pi * r * r)
        # Contour thresholds are deliberately permissive — blurred corner
        # dots lose circularity; the later corner-geometry tests reject
        # rings/dirt that pass these gates.
        if circ > 0.72 and 6 <= r <= 120 and fill > 0.72:
            cand.append((x, y, r))
    if len(cand) < 4:
        return cand

    # Fiducials are the *largest* solid circles (FIDUCIAL_DIAMETER_MM is well
    # above any mark element), share one radius, and the four chosen points
    # must form a large near-rectangle.  Black rings are annular: a tiny
    # number survive the fill test but are ~20% smaller, so a tight radius
    # cluster plus the rectangle constraint reject them.
    cand.sort(key=lambda t: -t[2])
    rmax = cand[0][2]
    best: list | None = None
    best_score = -1.0
    from itertools import combinations
    # True fiducials match in radius to ~15%; the corner-geometry checks
    # below (aspect + interior-point rejection) keep a smaller ring/blobs
    # out even when they pass the radius gate.
    group = [c for c in cand if abs(c[2] - rmax) <= 0.15 * rmax]
    if len(group) >= 4:
        for quad in combinations(group[:8], 4):
            pts = np.array([(q[0], q[1]) for q in quad], np.float32)
            (rw, rh) = cv2.minAreaRect(pts)[1]
            if min(rw, rh) <= 0:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            if aspect > 1.35:
                continue
            pair_d = [float(np.linalg.norm(pts[a] - pts[b]))
                      for a in range(4) for b in range(a + 1, 4)]
            if min(pair_d) < 0.75 * min(rw, rh):
                continue
            # Every genuine corner lies OUTSIDE the triangle formed by the
            # other three; an interior blob is strictly inside it.
            interior = False
            for k in range(4):
                tri = pts[[j for j in range(4) if j != k]]
                if cv2.pointPolygonTest(tri, (float(pts[k][0]),
                                              float(pts[k][1])),
                                        True) > 0.10 * min(rw, rh):
                    interior = True
                    break
            if interior:
                continue
            score = rw * rh - abs(aspect - 1.18) * rw * rh
            if score > best_score:
                best_score, best = score, list(quad)
    if best is not None:
        return best
    # No clean four-corner configuration: return fewer than four so the
    # caller treats the frame as ambiguous (never substitute a ring/blob).
    return [c for c in group if abs(c[2] - rmax) <= 0.05 * rmax][:3]


def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _similarity_from_fiducials(
        fids: list[tuple[float, float, float]]
) -> tuple[np.ndarray, dict]:
    """2x3 scan->canonical(300 PPI chart) affine + fiducial provenance."""
    src = _order_quad(np.array([(f[0], f[1]) for f in fids], np.float32))
    tl, tr, br, bl = src

    # Canonical fiducial positions, matching the renderer: the first mark
    # centre is at pad_px = (MARGIN + RING_RADIUS + 0.6) mm and each corner
    # dot sits MARGIN mm outside it.
    mark0 = spec.mm_to_px(spec.GRID_MARGIN_MM + spec.RING_RADIUS_MM + 0.6)
    margin = spec.mm_to_px(spec.GRID_MARGIN_MM)
    pitch = spec.mm_to_px(spec.GRID_PITCH_MM)
    span_w = (spec.GRID_COLS - 1) * pitch + 2 * margin
    span_h = (spec.GRID_ROWS - 1) * pitch + 2 * margin
    x0, y0 = mark0 - margin, mark0 - margin
    dst = np.array([[x0, y0], [x0 + span_w, y0],
                    [x0 + span_w, y0 + span_h], [x0, y0 + span_h]],
                   np.float32)

    if len(fids) == 4:
        m, _ = cv2.estimateAffinePartial2D(
            src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2),
            method=cv2.LMEDS)
    else:  # 3 dots: exact affine from the triangle
        m = cv2.getAffineTransform(
            src[:3].reshape(-1, 1, 2), dst[:3].reshape(-1, 1, 2))

    # Scan rotation = visual angle of the paper frame's top edge in image
    # coordinates (y-down).  cv2.getRotationMatrix2D positive = CCW, which
    # appears as a negative visual angle; reporting the visual edge angle is
    # what agrees with on-screen rulers and with the ground-truth metadata.
    edge = tr - tl
    rot = math.degrees(math.atan2(edge[1], edge[0]))
    mm_w = ((spec.GRID_COLS - 1) * spec.GRID_PITCH_MM
            + 2 * spec.GRID_MARGIN_MM)
    mm_h = ((spec.GRID_ROWS - 1) * spec.GRID_PITCH_MM
            + 2 * spec.GRID_MARGIN_MM)
    prov = {
        "fiducials_xy_px": [[round(float(a), 2), round(float(b), 2),
                             round(float(r_), 2)] for a, b, r_ in fids],
        "fiducial_px_per_mm_x": round(float(np.linalg.norm(tr - tl) / mm_w), 4),
        "fiducial_px_per_mm_y": round(float(np.linalg.norm(br - tr) / mm_h), 4),
        "fiducial_rotation_deg": round(rot, 4),
        "fiducial_count": len(fids),
    }
    return m, prov


# ---------------------------------------------------------------------------
# Step 3: ruler verification in canonical space
# ---------------------------------------------------------------------------

def _ruler_baseline_px() -> tuple[float, float]:
    ppi = spec.NATIVE_PPI
    pad = spec.RING_RADIUS_MM + 0.6
    ox = spec.mm_to_px(spec.GRID_MARGIN_MM + pad, ppi)
    ry = (spec.mm_to_px(spec.GRID_MARGIN_MM + pad, ppi)
          + (spec.GRID_ROWS - 1) * spec.mm_to_px(spec.GRID_PITCH_MM, ppi)
          + spec.mm_to_px(spec.GRID_MARGIN_MM + 5, ppi))
    return ox, ry


def _subpixel_groups(prof: np.ndarray, thr: float) -> list[float]:
    """Centres (sub-pixel, ink-weighted) of every above-threshold run."""
    on = prof > thr
    out: list[float] = []
    i = 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j < len(on) and on[j]:
            j += 1
        cols = np.arange(i, j, dtype=np.float64)
        w = prof[i:j].astype(np.float64)
        out.append(float((cols * w).sum() / w.sum()))
        i = j
    return out


def _verify_ruler(canon_dark: np.ndarray) -> dict:
    ppi = spec.NATIVE_PPI
    ox, ry = _ruler_baseline_px()
    x0 = int(ox)
    x1 = int(ox + spec.mm_to_px(spec.RULER_LENGTH_MM, ppi))
    band = int(spec.mm_to_px(2.4, ppi))
    upper = canon_dark[max(0, int(ry) - band):int(ry) - 2, x0:x1]
    if upper.size == 0:
        return {"found": False, "reason": "ruler_region_empty"}

    col = (upper > 0).sum(axis=0).astype(np.float32)
    if col.max() <= 0:
        return {"found": False, "reason": "no_tick_ink", "x0": x0}
    thr = max(2.0, col.max() * 0.35)
    ticks = _subpixel_groups(col, thr)
    if len(ticks) < 5:
        return {"found": False, "reason": "too_few_ticks",
                "ticks": len(ticks)}

    gaps = np.diff(np.array(ticks))
    one_mm = spec.mm_to_px(1.0, ppi)
    near = gaps[(gaps > one_mm * 0.7) & (gaps < one_mm * 1.35)]
    if len(near) < 3:
        return {"found": False, "reason": "tick_pitch_inconsistent",
                "ticks": len(ticks)}
    pitch = float(np.median(near))
    pitch_std = float(np.std(near))
    tick_scale = pitch / one_mm

    # End brackets extend *below* the baseline and mark the auditable 30 mm
    # span.  Their ink-weighted centres give a sub-pixel span — the primary
    # scale reference; the 1 mm tick pitch only cross-checks it.
    below = (canon_dark[int(ry) + 1:int(ry) + int(spec.mm_to_px(1.4, ppi)),
                        x0:x1] > 0).sum(axis=0).astype(np.float32)
    bthr = max(2.0, float(below.max()) * 0.5) if below.max() else 1e9
    brackets = _subpixel_groups(below, bthr)
    bracket_scale = None
    span_px = None
    if len(brackets) >= 2:
        # Leftmost/rightmost groups are the two end brackets.
        bl, br_ = min(brackets), max(brackets)
        span_px = float(br_ - bl)
        bracket_scale = span_px / spec.mm_to_px(spec.RULER_LENGTH_MM, ppi)

    primary_scale = bracket_scale if bracket_scale is not None else tick_scale
    # At 300 PPI a 1 mm tick pitch is only ~11.8 px, so tick grouping carries
    # an unavoidable ~2% quantisation bias; disagreement beyond 3% signals a
    # real scale conflict.
    disagree = (bracket_scale is not None
                and abs(bracket_scale - tick_scale) > 0.03)
    return {
        "found": True,
        "ticks": len(ticks),
        "median_tick_pitch_px": round(pitch, 3),
        "tick_pitch_std_px": round(pitch_std, 3),
        "expected_tick_pitch_px": round(one_mm, 3),
        "tick_pitch_scale": round(tick_scale, 5),
        "bracket_span_px": (None if span_px is None else round(span_px, 2)),
        "bracket_span_scale": (None if bracket_scale is None
                               else round(float(bracket_scale), 5)),
        "primary_scale": round(float(primary_scale), 5),
        "sources_disagree": bool(disagree),
        "ruler_ry_px": round(float(ry), 2),
    }


# ---------------------------------------------------------------------------
# Step 4: per-mark reading (K ring + colour-separated mini-crosses)
# ---------------------------------------------------------------------------

def _ring_centre(mask: np.ndarray, gx: float, gy: float
                 ) -> tuple[float, float, float] | None:
    """Black reference ring centroid via ellipse fit among ring pixels.

    A full ring fits tightly; a damaged arc still gives an ellipse fit with
    lower coverage (which widens uncertainty) instead of a false point.
    """
    ring_r = spec.mm_to_px(spec.RING_RADIUS_MM)
    x0 = int(gx - ring_r * 1.35)
    y0 = int(gy - ring_r * 1.35)
    x1 = int(gx + ring_r * 1.35) + 1
    y1 = int(gy + ring_r * 1.35) + 1
    H, W = mask.shape
    patch = mask[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    ys, xs = np.where(patch > 0)
    if len(xs) < 12:
        return None
    pts = np.stack([xs + max(0, x0), ys + max(0, y0)], axis=1)
    thickness = max(2, int(spec.mm_to_px(spec.RING_THICKNESS_MM) * 2.2))
    cx = cy = None
    if len(pts) >= 5:
        (ecx, ecy), (ax_major, ax_minor), _ = cv2.fitEllipse(
            pts.astype(np.float32))
        if (0.75 * 2 * ring_r < ax_major < 1.35 * 2 * ring_r
                and 0.5 < ax_minor / max(ax_major, 1e-6) <= 1.2):
            cx, cy = float(ecx), float(ecy)
    if cx is None:
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    rr = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    # Expected band thickness in pixels; AA strokes rasterise slightly
    # thinner than the requested integer thickness.
    thickness_px = spec.mm_to_px(spec.RING_THICKNESS_MM) * 0.85
    on_band = ((rr > ring_r - thickness_px)
               & (rr < ring_r + thickness_px)).sum()
    expected_ring = 2 * math.pi * ring_r * thickness_px
    coverage = float(np.clip(on_band / max(expected_ring, 1), 0.0, 1.0))
    dirt = float((np.abs(rr - ring_r) > thickness_px * 1.8).mean())
    return cx, cy, min(coverage, 1.0 - dirt * 0.5)


def _arm_center(prof: np.ndarray, span: float, centre: float
                ) -> tuple[float, float] | None:
    """Sub-pixel arm centre = ink-weighted centroid of the run crossing the
    nominal centre.  Thresholding clips blurred arms at both ends equally, so
    the midpoint stays unbiased; integer column rounding is removed by the
    ink-weighted average.  Returns (center, coverage)."""
    on = prof > max(1, prof.max() * 0.25)
    run = _central_run(on, centre)
    if run is None:
        return None
    lo, hi = run
    cov = (hi - lo + 1) / span
    if cov < 0.30:
        return None
    cols = np.arange(lo, hi + 1)
    w = prof[lo:hi + 1].astype(np.float64)
    center = float((cols * w).sum() / w.sum())
    return center, min(1.0, cov)


def _central_run(on: np.ndarray, centre: float) -> tuple[int, int] | None:
    idx = np.where(on)[0]
    if len(idx) == 0:
        return None
    ic = int(round(centre))
    if not (0 <= ic < len(on)) or not on[ic]:
        ic = int(idx[np.abs(idx - centre).argmin()])
    lo = ic
    while lo - 1 >= 0 and on[lo - 1]:
        lo -= 1
    hi = ic
    while hi + 1 < len(on) and on[hi + 1]:
        hi += 1
    return lo, hi


def _read_cross(mask: np.ndarray, cx_nom: float, cy_nom: float,
                row: int, col: int, plate: str) -> Observation:
    span = spec.mm_to_px(spec.PLATE_CROSS_SPAN_MM)
    band = max(2, int(round(spec.mm_to_px(
        spec.PLATE_CROSS_THICKNESS_MM) * 2.0)))
    half = int(round(span * 0.72))
    H, W = mask.shape
    x0, y0 = max(0, int(cx_nom) - half), max(0, int(cy_nom) - half)
    x1, y1 = min(W, int(cx_nom) + half + 1), min(H, int(cy_nom) + half + 1)
    patch = mask[y0:y1, x0:x1]
    lx = cx_nom - x0
    ly = cy_nom - y0
    hb = patch[max(0, int(ly) - band):int(ly) + band + 1, :]
    vb = patch[:, max(0, int(lx) - band):int(lx) + band + 1]

    px_per_um = spec.NATIVE_PPI / spec.MM_PER_INCH / 1000.0
    obs = Observation(plate=plate, row=row, col=col, x_um=None, y_um=None)
    cov_h = cov_v = 0.0
    if hb.size:
        hm = _arm_center((hb > 0).sum(axis=0), span, lx)
        if hm:
            mid, cov_h = hm
            obs.x_um = (x0 + mid - cx_nom) / px_per_um
            obs.x_sigma_um = max(10.0, (0.35 / max(cov_h, 0.3)
                                        + 0.6 * (1 - cov_h)) / px_per_um)
    if vb.size:
        vm = _arm_center((vb > 0).sum(axis=1), span, ly)
        if vm:
            mid, cov_v = vm
            obs.y_um = (y0 + mid - cy_nom) / px_per_um
            obs.y_sigma_um = max(10.0, (0.35 / max(cov_v, 0.3)
                                        + 0.6 * (1 - cov_v)) / px_per_um)
    obs.arm_h, obs.arm_v = cov_h, cov_v

    ink = int((patch > 0).sum())
    hh = np.zeros(patch.shape[:2], np.uint8)
    vv = np.zeros(patch.shape[:2], np.uint8)
    hh[max(0, int(ly) - band):int(ly) + band + 1, :] = 1
    vv[:, max(0, int(lx) - band):int(lx) + band + 1] = 1
    on_arms = int(((patch > 0) & ((hh | vv) > 0)).sum())
    obs.dirt_ratio = 0.0 if ink == 0 else max(0.0, ink - on_arms) / max(ink, 1)

    if obs.x_um is None and obs.y_um is None:
        obs.state = "missing"
    elif cov_h < 0.55 or cov_v < 0.55:
        obs.state = "partial"
    return obs


def _read_mark(masks: dict[str, np.ndarray], gx: float, gy: float,
               plate: str, row: int, col: int,
               ring_centre: tuple[float, float, float] | None
               ) -> Observation:
    """Read the plate's mini-cross; black reads its ring centre directly."""
    if plate == "black":
        px_per_um = spec.NATIVE_PPI / spec.MM_PER_INCH / 1000.0
        obs = Observation(plate=plate, row=row, col=col,
                          x_um=None, y_um=None)
        if ring_centre is None:
            obs.state = "missing"
            return obs
        cx, cy, cov = ring_centre
        obs.x_um = (cx - gx) / px_per_um
        obs.y_um = (cy - gy) / px_per_um
        sigma = max(10.0, 0.35 / max(cov, 0.25) / px_per_um)
        obs.x_sigma_um = obs.y_sigma_um = sigma
        obs.arm_h = obs.arm_v = cov
        obs.state = "ok" if cov > 0.55 else "partial"
        return obs

    orbit = spec.mm_to_px(spec.PLATE_ORBIT_MM)
    ang = math.radians(spec.PLATE_ANGLES_DEG[plate])
    cx_nom = gx + math.cos(ang) * orbit
    cy_nom = gy + math.sin(ang) * orbit
    return _read_cross(masks[plate], cx_nom, cy_nom, row, col, plate)


# ---------------------------------------------------------------------------
# Step 5: global solve (common similarity vs per-plate translation)
# ---------------------------------------------------------------------------

def _fit_plate_vectors(observations: list[Observation]) -> tuple[dict, dict]:
    """Separate paper residual from plate shifts.

    Model for plate k at nominal position n_mm::

        e_um = F @ n_mm * 1000 + t + d_k,   F = [[a,-b],[b,a]]

    F (small residual rotation/scale) and t are *shared by every plate* and
    describe remaining paper-frame error; d_k is the plate's own translation.
    Black is the reference: d_black is forced to zero.  Weighted, Huber-ish
    iteration keeps partial marks from dominating.
    """
    points = spec.grid_points_px()
    positions = {(ri, ci): points[ri * spec.GRID_COLS + ci]
                 for ri in range(spec.GRID_ROWS)
                 for ci in range(spec.GRID_COLS)}
    um_per_px = spec.MM_PER_INCH / spec.NATIVE_PPI * 1000.0

    by_plate: dict[str, list[Observation]] = {p: [] for p in spec.PLATES}
    for o in observations:
        by_plate[o.plate].append(o)

    def median_vec(plate: str) -> np.ndarray:
        xs = [o.x_um for o in by_plate[plate] if o.x_um is not None]
        ys = [o.y_um for o in by_plate[plate] if o.y_um is not None]
        return np.array([np.median(xs) if xs else 0.0,
                         np.median(ys) if ys else 0.0])

    d = {p: median_vec(p) - median_vec("black") for p in spec.PLATES}

    def n_mm(o: Observation) -> np.ndarray:
        px, py = positions[(o.row, o.col)]
        return np.array([px, py]) * um_per_px / 1000.0

    a = b = tx = ty = 0.0
    for _ in range(4):
        # --- weighted LSQ for [a, b, tx, ty] given current plate vectors ---
        rows_A, rows_y, rows_w = [], [], []
        for o in observations:
            nx, ny = n_mm(o) * 1000.0  # µm-scale coordinates of the position
            if o.x_um is not None:
                rows_A.append([nx, -ny, 1.0, 0.0])
                rows_y.append(o.x_um - d[o.plate][0])
                rows_w.append(1.0 / max(o.x_sigma_um, 5.0) ** 2)
            if o.y_um is not None:
                rows_A.append([ny, nx, 0.0, 1.0])
                rows_y.append(o.y_um - d[o.plate][1])
                rows_w.append(1.0 / max(o.y_sigma_um, 5.0) ** 2)
        if len(rows_A) >= 6:
            sqw = np.sqrt(np.array(rows_w))
            sol, *_ = np.linalg.lstsq(
                np.array(rows_A) * sqw[:, None],
                np.array(rows_y) * sqw, rcond=None)
            a, b, tx, ty = (float(v) for v in sol)

        # --- recompute plate vectors with common similarity removed ---
        F = np.array([[a, -b], [b, a]])
        for p in spec.PLATES:
            vals: dict[str, list[tuple[float, float]]] = {"x": [], "y": []}
            for o in by_plate[p]:
                corr = F @ n_mm(o) * 1000.0 + np.array([tx, ty])
                if o.x_um is not None:
                    vals["x"].append((o.x_um - corr[0], o.x_sigma_um))
                if o.y_um is not None:
                    vals["y"].append((o.y_um - corr[1], o.y_sigma_um))

            def wmean(items):
                if not items:
                    return 0.0
                v = np.array([i[0] for i in items])
                w = np.array([1.0 / max(i[1], 5.0) ** 2 for i in items])
                # One Huber-ish trim of gross outliers (dirt on an arm).
                mu = float(np.average(v, weights=w))
                keep = np.abs(v - np.median(v)) < 250.0
                if keep.sum() >= 2:
                    return float(np.average(v[keep], weights=w[keep]))
                return mu

            d[p] = np.array([wmean(vals["x"]), wmean(vals["y"])])
        d = {p: d[p] - d["black"] for p in spec.PLATES}

    # --- final per-axis statistics --------------------------------------
    F = np.array([[a, -b], [b, a]])
    t = np.array([tx, ty])
    out: dict[str, dict] = {}
    cw_mm = ((spec.GRID_COLS - 1) * spec.GRID_PITCH_MM
             + 2 * spec.GRID_MARGIN_MM + 2 * (spec.RING_RADIUS_MM + 0.6))
    ch_mm = ((spec.GRID_ROWS - 1) * spec.GRID_PITCH_MM
             + 2 * spec.GRID_MARGIN_MM + 2 * (spec.RING_RADIUS_MM + 0.6))
    r_field = 0.5 * math.hypot(cw_mm, ch_mm)
    # Residual-rotation uncertainty leaks ~theta*r into each plate reading;
    # take 10% of the corrected common term as systematic floor.
    frame_sys = 0.1 * math.hypot(a, b) * r_field * 1000.0

    for p in spec.PLATES:
        axis_data: dict[str, list[tuple[float, float]]] = {"x": [], "y": []}
        for o in by_plate[p]:
            corr = F @ n_mm(o) * 1000.0 + t
            if o.x_um is not None:
                axis_data["x"].append((o.x_um - corr[0], o.x_sigma_um))
            if o.y_um is not None:
                axis_data["y"].append((o.y_um - corr[1], o.y_sigma_um))
        rec: dict = {"n_x": len(axis_data["x"]), "n_y": len(axis_data["y"])}
        for axis, items in axis_data.items():
            if not items:
                rec[axis] = None
                continue
            v = np.array([i[0] for i in items])
            sg = np.clip(np.array([i[1] for i in items]), 5, None)
            w = 1.0 / sg ** 2
            mu = float(np.average(v, weights=w))
            mad = float(np.median(np.abs(v - np.median(v))) * 1.4826)
            se = math.sqrt(1.0 / w.sum()
                           + (mad / math.sqrt(len(v))) ** 2
                           + frame_sys ** 2)
            rec[axis] = {"mu": mu, "se": se, "mad": mad}
        out[p] = rec

    common = {"a": a, "b": b, "tx": tx, "ty": ty}
    return out, common


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def analyze_image(img_bgr: np.ndarray,
                  declared_ppi: float | None = None) -> AnalysisResult:
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    dark = _plate_mask(hsv, "black")
    provenance: dict = {
        "detector_version": DETECTOR_VERSION,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "declared_scan_ppi": declared_ppi,
        "steps": [],
    }

    fids = _find_fiducials(dark)
    provenance["steps"].append({"step": "fiducials", "count": len(fids)})
    if len(fids) < 3:
        return AnalysisResult(
            format_id=spec.FORMAT_ID, detected=False, width_px=W, height_px=H,
            declared_scan_ppi=declared_ppi, measured_ppi=None,
            frame_rotation_deg=0.0, frame_scale=1.0, frame_quality=0.0,
            residual_rotation_deg=0.0, residual_scale=0.0,
            plates={p: PlateResult(p, "missing",
                                   notes=["未能定位纸张定位点，色版结果不予估计"])
                    for p in spec.PLATES},
            observations=[], provenance=provenance,
            status="frame_not_found")

    m, fid_prov = _similarity_from_fiducials(fids)
    provenance["steps"].append({"step": "paper_frame", **fid_prov})

    cw, ch = spec.chart_size_px(spec.NATIVE_PPI)
    canon = cv2.warpAffine(img_bgr, m, (cw, ch),
                           flags=cv2.INTER_CUBIC,
                           borderValue=(255, 255, 255))
    canon_hsv = cv2.cvtColor(canon, cv2.COLOR_BGR2HSV)
    canon_dark = _plate_mask(canon_hsv, "black")
    masks = {p: _plate_mask(canon_hsv, p) for p in spec.PLATES}

    # estimateAffinePartial2D maps the *scan* fiducial quad onto the larger
    # canonical(300 PPI) quad, so |linear| = canonical_px / scan_px over the
    # same physical span = NATIVE_PPI / scan_ppi.  Therefore:
    #   scan_ppi      = NATIVE_PPI / |linear|
    #   scan/native px ratio = 1 / |linear|
    canon_per_scan = float(math.hypot(m[0, 0], m[1, 0]))
    scale_vs_native = 1.0 / canon_per_scan
    measured_ppi = spec.NATIVE_PPI / canon_per_scan
    # Rotation is reported as the visual top-edge angle (y-down).
    rot = float(fid_prov["fiducial_rotation_deg"])

    ruler = _verify_ruler(canon_dark)
    provenance["steps"].append({"step": "ruler", **ruler})
    frame_quality = min(1.0, len(fids) / 4) * 0.6
    ruler_mismatch = bool(ruler.get("sources_disagree"))
    if ruler.get("found"):
        corr = float(ruler["primary_scale"])  # canonical px / true px
        frame_quality += 0.4 * min(1.0, ruler["ticks"] / 25)
        if abs(corr - 1.0) < 0.03:
            # The ruler measures the canonical warp: its true-mm scale factor
            # corr means the fiducial factor s should be s/corr.  Blend.
            corrected = canon_per_scan / corr
            blended = 0.6 * canon_per_scan + 0.4 * corrected
            canon_per_scan = blended
            scale_vs_native = 1.0 / blended
            measured_ppi = spec.NATIVE_PPI / blended
        else:
            ruler_mismatch = True
    frame_quality = min(1.0, frame_quality)

    if abs(rot) > spec.MAX_PAPER_ROTATION_DEG:
        provenance["steps"].append(
            {"step": "rotation_guard", "rotation_deg": round(rot, 3),
             "limit_deg": spec.MAX_PAPER_ROTATION_DEG,
             "decision": "rejected"})
        return AnalysisResult(
            format_id=spec.FORMAT_ID, detected=False, width_px=W, height_px=H,
            declared_scan_ppi=declared_ppi, measured_ppi=measured_ppi,
            frame_rotation_deg=rot, frame_scale=scale_vs_native,
            frame_quality=frame_quality,
            residual_rotation_deg=0.0, residual_scale=0.0,
            plates={p: PlateResult(p, "missing",
                                   notes=["纸张旋转超出允许校正范围"])
                    for p in spec.PLATES},
            observations=[], provenance=provenance,
            status="rotation_out_of_range")

    observations: list[Observation] = []
    points = spec.grid_points_px()
    for ri in range(spec.GRID_ROWS):
        for ci in range(spec.GRID_COLS):
            gx, gy = points[ri * spec.GRID_COLS + ci]
            ring = _ring_centre(masks["black"], gx, gy)
            for p in spec.PLATES:
                observations.append(
                    _read_mark(masks, gx, gy, p, ri, ci, ring))

    fit, common = _fit_plate_vectors(observations)
    residual_rot = math.degrees(common["b"])
    residual_scale = common["a"]
    provenance["steps"].append({
        "step": "common_similarity_removed",
        "residual_rotation_deg": round(residual_rot, 5),
        "residual_scale": round(residual_scale, 7),
        "common_translation_um": [round(common["tx"], 2),
                                  round(common["ty"], 2)],
        "note": "该项由各色版共同承担，已在色版偏移估计前扣除",
    })
    if ruler_mismatch:
        provenance["steps"].append(
            {"step": "ruler_guard", "decision": "scale_disagreement_flagged",
             "tick_pitch_scale": ruler.get("tick_pitch_scale"),
             "bracket_span_scale": ruler.get("bracket_span_scale")})

    px_size_um = (spec.MM_PER_INCH / measured_ppi * 1000.0
                  if measured_ppi else 85.0)
    plates_out: dict[str, PlateResult] = {}
    any_candidate = ruler_mismatch
    dirt_by_plate: dict[str, float] = {}
    for p in spec.PLATES:
        rec = fit[p]
        pr = PlateResult(plate=p, status="ok")
        pr.n_x, pr.n_y = rec["n_x"], rec["n_y"]
        for axis in ("x", "y"):
            r = rec[axis]
            if r is None:
                setattr(pr, f"d{axis}_um", None)
                continue
            halfwidth = 1.96 * max(r["se"], 0.5 * px_size_um,
                                   spec.MARK_TOLERANCE_UM * 0.15)
            setattr(pr, f"d{axis}_um", r["mu"])
            setattr(pr, f"{axis}_lo", r["mu"] - halfwidth)
            setattr(pr, f"{axis}_hi", r["mu"] + halfwidth)

        n_min = min(pr.n_x, pr.n_y)
        n_max = max(pr.n_x, pr.n_y)
        worst_hw = max(
            (pr.x_hi - pr.x_lo) / 2 if pr.x_lo is not None else 0.0,
            (pr.y_hi - pr.y_lo) / 2 if pr.y_lo is not None else 0.0)
        plate_obs = [o for o in observations if o.plate == p]
        inked = [o for o in plate_obs if o.state != "missing"]
        dirt = float(np.mean([o.dirt_ratio for o in inked])) if inked else 0.0
        dirt_by_plate[p] = dirt
        partial = sum(1 for o in plate_obs if o.state == "partial")

        if n_max == 0:
            pr.status = "missing"
            pr.notes.append("未检出该色版墨色：疑似缺色，输出空缺而非零偏移")
            pr.x_lo = pr.x_hi = pr.y_lo = pr.y_hi = None
        elif n_max < spec.MIN_MARKS_FOR_CANDIDATE:
            pr.status = "candidate"
            pr.notes.append(
                f"可读标记不足（x轴{pr.n_x}处 / y轴{pr.n_y}处），"
                "仅给候选与可信范围")
        elif (n_min < spec.MIN_MARKS_FOR_CANDIDATE
              or worst_hw > spec.MARK_TOLERANCE_UM
              or partial or dirt > 0.35):
            pr.status = "candidate"
            if n_min < spec.MIN_MARKS_FOR_CANDIDATE:
                pr.notes.append("部分轴可读标记不足，按候选处理")
            if worst_hw > spec.MARK_TOLERANCE_UM:
                pr.notes.append(
                    f"读数离散大（±{worst_hw:.0f} µm），范围已放宽")
            if partial:
                pr.notes.append(f"{partial} 处标记残缺（按单臂读数）")
            if dirt > 0.35:
                pr.notes.append(f"局部污点比例高（{dirt:.0%}）")
        if ruler_mismatch:
            pr.notes.append("校准尺与定位点尺度不一致，尺度存疑")
        any_candidate = any_candidate or pr.status == "candidate"

        pr.confidence = 0.0 if pr.status == "missing" else float(np.clip(
            0.30 + 0.55 * min(n_max, spec.GRID_COLS * spec.GRID_ROWS)
            / (spec.GRID_COLS * spec.GRID_ROWS)
            - worst_hw / 1200.0 - dirt - 0.1 * partial, 0.05, 0.98))
        plates_out[p] = pr

    status = "candidates" if any_candidate else "ok"
    return AnalysisResult(
        format_id=spec.FORMAT_ID, detected=True, width_px=W, height_px=H,
        declared_scan_ppi=declared_ppi, measured_ppi=measured_ppi,
        frame_rotation_deg=rot, frame_scale=scale_vs_native,
        frame_quality=frame_quality,
        residual_rotation_deg=residual_rot,
        residual_scale=residual_scale,
        plates=plates_out, observations=observations,
        provenance=provenance, status=status)


def analyze_file(path: str | Path,
                 declared_ppi: float | None = None) -> AnalysisResult:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解码图像: {path}")
    res = analyze_image(img, declared_ppi)
    res.run_id = uuid.uuid4().hex[:12]
    return res
