// PRT-1 chart geometry — kept in sync with backend/app/markspec.py.
// The backend returns canonical-space mark coordinates directly, so the UI
// mostly needs constants for drawing cells and the zoomed mark.

export const NATIVE_PPI = 300;
export const MM_PER_INCH = 25.4;
export const GRID_COLS = 5;
export const GRID_ROWS = 4;
export const RING_RADIUS_MM = 1.1;
export const PLATE_ORBIT_MM = 2.1;
export const PLATE_CROSS_SPAN_MM = 1.5;
export const PLATE_ANGLES_DEG = { cyan: 0, magenta: 90, yellow: 180 };

export const PLATE_META = {
  cyan: { label: "青 C", rgb: "rgb(0,170,220)" },
  magenta: { label: "品红 M", rgb: "rgb(215,0,120)" },
  yellow: { label: "黄 Y", rgb: "rgb(205,175,0)" },
  black: { label: "黑 K(基准)", rgb: "rgb(50,50,50)" },
};

export function mmToPx(mm, ppi = NATIVE_PPI) {
  return (mm * ppi) / MM_PER_INCH;
}

export const UM_PER_PX = MM_PER_INCH / NATIVE_PPI * 1000; // canonical px -> um
