"""PRT-1 registration target specification.

This project is deliberately restricted to the *synthetic* chart that ships
with the repository (``backend/data/``) and scans of that same chart.  The
detector therefore recognises one explicit, documented format rather than
guessing at arbitrary print marks.

Chart layout (drawn at :data:`NATIVE_PPI`):

* A regular grid of :data:`GRID_COLS` x :data:`GRID_ROWS` registration marks
  with pitch :data:`GRID_PITCH_MM` millimetres.  Every mark contains five
  *colour-separable* elements (concentric crosses overprint into black and a
  camera cannot separate the inks afterwards):

  - black (K): a full ring at the mark centre — the reference centre;
  - one mini-cross per plate (C/M/Y/K), placed on a common orbit at fixed
    cardinal angles.  Spatial separation keeps each plate's arms readable in
    its own colour band; a missing arm is a missing reading, never a zero.

* Four solid corner fiducials (black) that anchor the *paper* frame.
* A calibration ruler (black/white ticks every millimetre, a long tick every
  five millimetres, a labelled 10 mm span) that anchors scan scale.

Global paper rotation/scale is estimated from the black fiducials, the ruler
and the shared component of all four plates.  Only the *residual* per-colour
component is reported as plate misregistration, so a uniformly skewed scan can
never be mistaken for a colour plate error.
"""

from __future__ import annotations

from dataclasses import dataclass

FORMAT_ID = "PRT-1"

#: PPI the vector chart is authored at.  Physical geometry is defined in mm so
#: the same chart can be rendered at other PPIs for resolution-change tests.
NATIVE_PPI = 300.0

MM_PER_INCH = 25.4

# Mark grid ----------------------------------------------------------------
GRID_COLS = 5
GRID_ROWS = 4
GRID_PITCH_MM = 8.0
GRID_MARGIN_MM = 10.0

#: Per-mark geometry, in mm.  Each plate owns a mini-cross on a common orbit
#: (angle = :data:`PLATE_ANGLES_DEG`); black additionally carries the ring.
RING_RADIUS_MM = 1.10
RING_THICKNESS_MM = 0.20
PLATE_ORBIT_MM = 2.10
PLATE_CROSS_SPAN_MM = 1.50
PLATE_CROSS_THICKNESS_MM = 0.26

#: Cardinal position of each plate's mini-cross around the ring centre.
#: Black owns only the ring (no mini-cross).
PLATE_ANGLES_DEG = {"cyan": 0, "magenta": 90, "yellow": 180}

#: Concentricity tolerance of the synthetic/printed mark itself.  Detection
#: claims finer than this against a damaged mark are meaningless.
MARK_TOLERANCE_UM = 120.0

# Corner fiducials ---------------------------------------------------------
FIDUCIAL_DIAMETER_MM = 3.0

# Ruler --------------------------------------------------------------------
RULER_TICK_PITCH_MM = 1.0
RULER_LENGTH_MM = 30.0
RULER_THICKNESS_MM = 0.18

#: Maximum scan rotation (degrees) the detector will accept before flagging
#: the frame fit as unreliable rather than "correcting" an arbitrary image.
MAX_PAPER_ROTATION_DEG = 8.0

#: Minimum number of marks that must be visible per plate for a reportable
#: (candidate + range) result; below this the plate is reported MISSING.
MIN_MARKS_FOR_CANDIDATE = 3

#: Channel order.  Black is the reference plate by convention (offset 0/0).
PLATES = ("cyan", "magenta", "yellow", "black")


@dataclass(frozen=True)
class PlateColor:
    name: str
    #: RGB used when rendering / matching.  Yellow is darkened slightly in the
    #: crosshair so it remains visible on white; we store pure ink values too.
    render_rgb: tuple[int, int, int]
    ink_cmyk: tuple[float, float, float, float]
    label: str


PLATE_COLORS = {
    "cyan": PlateColor("cyan", (0, 170, 220), (1.0, 0.0, 0.0, 0.0), "C"),
    "magenta": PlateColor("magenta", (215, 0, 120), (0.0, 1.0, 0.0, 0.0), "M"),
    "yellow": PlateColor("yellow", (225, 200, 0), (0.0, 0.0, 1.0, 0.0), "Y"),
    # Near-black but not (0,0,0) so thresholding tolerates scanner lift.
    "black": PlateColor("black", (45, 45, 45), (0.0, 0.0, 0.0, 1.0), "K"),
}


def mm_to_px(mm: float, ppi: float = NATIVE_PPI) -> float:
    return mm * ppi / MM_PER_INCH


def grid_points_px(ppi: float = NATIVE_PPI) -> list[tuple[float, float]]:
    """Nominal mark centres in chart-local pixels at ``ppi``."""
    pitch = mm_to_px(GRID_PITCH_MM, ppi)
    pad = RING_RADIUS_MM + 0.6
    ox = mm_to_px(GRID_MARGIN_MM + pad, ppi)
    oy = ox
    return [
        (ox + c * pitch, oy + r * pitch)
        for r in range(GRID_ROWS)
        for c in range(GRID_COLS)
    ]


def chart_size_px(ppi: float = NATIVE_PPI) -> tuple[int, int]:
    pad = RING_RADIUS_MM + 0.6
    width_mm = 2 * GRID_MARGIN_MM + (GRID_COLS - 1) * GRID_PITCH_MM + 2 * pad
    height_mm = 2 * GRID_MARGIN_MM + (GRID_ROWS - 1) * GRID_PITCH_MM + 2 * pad
    # Ruler sits under the grid; extra room at the bottom.
    height_mm += GRID_MARGIN_MM + 8
    return (int(round(mm_to_px(width_mm, ppi))),
            int(round(mm_to_px(height_mm, ppi))))
