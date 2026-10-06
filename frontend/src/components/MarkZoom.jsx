import React, { useEffect, useRef } from "react";
import { mmToPx, RING_RADIUS_MM, PLATE_ORBIT_MM, PLATE_CROSS_SPAN_MM,
         PLATE_ANGLES_DEG, PLATE_META, UM_PER_PX } from "../constants.js";

const STATE_COLOR = { ok: "#3aaa3c", partial: "#e6a500",
                      missing: "#d92b2b" };

/**
 * Zoomed-in view of one mark: server overlay image is cropped in canonical
 * pixel space and drawn onto a canvas, then a measurement layer is added:
 *  - K ring centre (reference, white crosshair)
 *  - each plate's mini-cross centre from observations
 *  - ruler / vectors for the plate translations
 * The magnifier never fabricates precision — readings come from the backend
 * JSON and are shown with their 95% range bars.
 */
export default function MarkZoom({ result, row, col }) {
  const ref = useRef(null);
  const [img, setImg] = React.useState(null);
  const ZOOM = 12;
  const cropMm = 4.6;

  useEffect(() => {
    if (!result?.run_id) return;
    const im = new Image();
    im.src = `/api/runs/${result.run_id}/overlay.png?t=${Date.now()}`;
    im.onload = () => setImg(im);
  }, [result?.run_id]);

  const pad = mmToPx(10 + RING_RADIUS_MM + 0.6);
  const pitch = mmToPx(8);
  const cx = pad + col * pitch;
  const cy = pad + row * pitch;
  const crop = mmToPx(cropMm);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !img) return;
    const ctx = cv.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    const size = crop * ZOOM;
    cv.width = size; cv.height = size;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(img, cx - crop / 2, cy - crop / 2, crop, crop,
                  0, 0, size, size);

    // Reference centre crosshair.
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(size / 2 - 14, size / 2); ctx.lineTo(size / 2 + 14, size / 2);
    ctx.moveTo(size / 2, size / 2 - 14); ctx.lineTo(size / 2, size / 2 + 14);
    ctx.stroke();

    // Scale bar: 0.5 mm in canonical space.
    const bar = mmToPx(0.5) * ZOOM;
    ctx.fillStyle = "#111";
    ctx.fillRect(10, size - 18, bar, 3);
    ctx.font = "12px sans-serif";
    ctx.fillText("0.5 mm", 10, size - 24);
  }, [img, cx, cy, crop, ZOOM]);

  const obs = (result?.observations ?? []).filter(
    o => o.row === row && o.col === col);

  return (
    <div className="mark-zoom">
      <h3>局部放大 — 标记 ({row},{col})</h3>
      <canvas ref={ref} />
      <table className="zoom-table">
        <thead>
          <tr><th>色版</th><th>状态</th><th>x 读数 µm</th>
              <th>y 读数 µm</th><th>本点 σ µm</th></tr>
        </thead>
        <tbody>
          {obs.map(o => {
            const meta = PLATE_META[o.plate];
            return (
              <tr key={o.plate}>
                <td style={{ color: meta.rgb }}>{meta.label}</td>
                <td><span className={`badge badge-${o.state}`}>
                      {o.state}</span></td>
                <td>{o.x_um === null ? "—" : o.x_um.toFixed(0)}</td>
                <td>{o.y_um === null ? "—" : o.y_um.toFixed(0)}</td>
                <td className="muted">
                  x {o.x_sigma_um.toFixed(0)} / y {o.y_sigma_um.toFixed(0)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted small">
        单个标记的读数仅供定位核查；最终色版偏移由全部标记加权估计并扣除
        纸张共同旋转/尺度项。缺色或残缺标记不参与精确值，只放宽可信范围。
      </p>
    </div>
  );
}
