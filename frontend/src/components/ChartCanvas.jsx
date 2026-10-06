import React from "react";
import { mmToPx, GRID_COLS, RING_RADIUS_MM, PLATE_ORBIT_MM,
         PLATE_CROSS_SPAN_MM, PLATE_ANGLES_DEG } from "../constants.js";

const STATE_COLOR = { ok: "#3aaa3c", partial: "#e6a500",
                      missing: "#d92b2b" };

/**
 * Canvas overlay on the canonical (paper-rotation/scale-corrected) image.
 *
 * Layers:
 *  1. server-rendered evidence overlay PNG (warped image + arrows/glyphs);
 *  2. per-mark state glyphs drawn from JSON so they stay crisp at any zoom;
 *  3. a measurement readout (µm) following the cursor across a selected mark.
 */
export default function ChartCanvas({ result, overlayUrl, onPickMark }) {
  const [zoom, setZoom] = React.useState(1);
  const [cursor, setCursor] = React.useState(null); // canonical px
  const [selected, setSelected] = React.useState({ row: 0, col: 0 });

  // Canonical grid positions are deterministic (the backend warps to the
  // native 300 PPI frame), so they can be reconstructed client-side.
  function markCenter(row, col) {
    const pad = mmToPx(10 + RING_RADIUS_MM + 0.6);
    const pitch = mmToPx(8);
    return [pad + col * pitch, pad + row * pitch];
  }

  function onMove(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    setCursor({
      x: (e.clientX - rect.left) / zoom,
      y: (e.clientY - rect.top) / zoom,
    });
  }

  function onClick(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / zoom;
    const y = (e.clientY - rect.top) / zoom;
    let best = null, bd = 1e9;
    for (let r = 0; r < 4; r++) for (let c = 0; c < GRID_COLS; c++) {
      const [mx, my] = markCenter(r, c);
      const d = Math.hypot(x - mx, y - my);
      if (d < bd) { bd = d; best = { row: r, col: c }; }
    }
    if (best && bd < mmToPx(4)) {
      setSelected(best);
      onPickMark?.(best);
    }
  }

  const [x0, y0] = markCenter(0, 0);
  const W = x0 * 2 + mmToPx(8) * 4;
  const H = y0 * 2 + mmToPx(8) * 3 + mmToPx(18);

  const obsAt = (row, col) => (result?.observations ?? []).filter(
    o => o.row === row && o.col === col);

  const [sx, sy] = markCenter(selected.row, selected.col);
  const pxPerUm = 25.4 / 300 * 1000;

  return (
    <div className="canvas-wrap">
      <div className="zoom-bar">
        <button onClick={() => setZoom(z => Math.max(1, z / 1.25))}>−</button>
        <span>{Math.round(zoom * 100)}%</span>
        <button onClick={() => setZoom(z => Math.min(8, z * 1.25))}>+</button>
        <button onClick={() => setZoom(1)}>复位</button>
        <span className="hint">点击色标查看局部；滚轮缩放</span>
      </div>
      <div className="canvas-scroll">
        <div
          className="canvas-stage"
          style={{ width: W * zoom, height: H * zoom }}
          onMouseMove={onMove}
          onMouseLeave={() => setCursor(null)}
          onClick={onClick}
          onWheel={(e) => {
            e.preventDefault();
            setZoom(z => e.deltaY < 0 ? Math.min(8, z * 1.1)
                                      : Math.max(1, z / 1.1));
          }}
        >
          {overlayUrl && (
            <img src={overlayUrl} alt="套准证据叠层" draggable={false}
                 style={{ width: W * zoom, height: H * zoom }} />
          )}
          <svg viewBox={`0 0 ${W} ${H}`}
               style={{ width: W * zoom, height: H * zoom,
                        position: "absolute", inset: 0 }}>
            {Array.from({ length: 4 }).map((_, row) =>
              Array.from({ length: GRID_COLS }).map((__, col) => {
                const [cx, cy] = markCenter(row, col);
                const isSel = selected.row === row && selected.col === col;
                return (
                  <g key={`${row}-${col}`}>
                    <circle cx={cx} cy={cy} r={mmToPx(3)} fill="none"
                      stroke={isSel ? "#0050ff" : "rgba(0,0,0,0.15)"}
                      strokeWidth={isSel ? 0.8 : 0.3} />
                    {obsAt(row, col).map(o => {
                      const st = STATE_COLOR[o.state] ?? "#888";
                      const ang = (PLATE_ANGLES_DEG[o.plate] ?? 270)
                        * Math.PI / 180;
                      const ox = o.plate === "black" ? 0
                        : Math.cos(ang) * mmToPx(PLATE_ORBIT_MM);
                      const oy = o.plate === "black" ? 0
                        : Math.sin(ang) * mmToPx(PLATE_ORBIT_MM);
                      const half = mmToPx(PLATE_CROSS_SPAN_MM / 2) + 0.6;
                      return (
                        <circle key={o.plate} cx={cx + ox} cy={cy + oy}
                          r={half} fill="none" stroke={st} strokeWidth={0.25}
                          strokeDasharray={o.state === "partial"
                            ? "0.8 0.6" : undefined} />
                      );
                    })}
                  </g>
                );
              }))}
            {cursor && (
              <g pointerEvents="none">
                <line x1={cursor.x - 4} y1={cursor.y} x2={cursor.x + 4}
                      y2={cursor.y} stroke="#0050ff" strokeWidth={0.2} />
                <line x1={cursor.x} y1={cursor.y - 4} x2={cursor.x}
                      y2={cursor.y + 4} stroke="#0050ff" strokeWidth={0.2} />
              </g>
            )}
          </svg>
        </div>
      </div>
      {cursor && (
        <div className="measure-readout">
          光标相对标记 ({selected.row},{selected.col})：
          <b>Δx {((cursor.x - sx) * pxPerUm) >= 0 ? "+" : ""}
            {((cursor.x - sx) * pxPerUm).toFixed(0)} µm</b>
          <b>Δy {((cursor.y - sy) * pxPerUm) >= 0 ? "+" : ""}
            {((cursor.y - sy) * pxPerUm).toFixed(0)} µm</b>
          <span className="muted">（纸张旋转/尺度已校正后的坐标系）</span>
        </div>
      )}
    </div>
  );
}
