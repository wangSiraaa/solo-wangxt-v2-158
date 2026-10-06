import React from "react";
import { PLATE_META } from "../constants.js";

function fmt(v, digits = 0) {
  return v === null || v === undefined ? "—" : Number(v).toFixed(digits);
}

/** Per-plate candidates, 95% ranges, confidence and machine/model flags. */
export default function PlateTable({ result, review }) {
  const plates = result?.plates ?? {};
  return (
    <div className="panel">
      <h3>各色版位移（相对黑版，µm）</h3>
      <table className="plate-table">
        <thead>
          <tr>
            <th>色版</th><th>状态</th>
            <th>Δx 候选</th><th>x 95% 范围</th>
            <th>Δy 候选</th><th>y 95% 范围</th>
            <th>读点数 x/y</th><th>置信度</th><th>说明</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(plates).map(([key, p]) => {
            const meta = PLATE_META[key];
            const adv = review?.plates?.[key];
            const disagree = (review?.disagreements ?? [])
              .filter(d => d.plate === key);
            return (
              <tr key={key} className={`row-${p.status}`}>
                <td style={{ color: meta.rgb, fontWeight: 600 }}>
                  {meta.label}</td>
                <td><span className={`badge badge-${p.status}`}>
                      {p.status}</span></td>
                <td className="num">{fmt(p.dx_um)}</td>
                <td className="num range">
                  {p.x_range_um[0] === null ? "—"
                    : `[${fmt(p.x_range_um[0])}, ${fmt(p.x_range_um[1])}]`}
                </td>
                <td className="num">{fmt(p.dy_um)}</td>
                <td className="num range">
                  {p.y_range_um[0] === null ? "—"
                    : `[${fmt(p.y_range_um[0])}, ${fmt(p.y_range_um[1])}]`}
                </td>
                <td className="num muted">
                  {p.n_x_readings}/{p.n_y_readings}</td>
                <td>
                  <ConfBar value={p.confidence} />
                </td>
                <td className="notes">
                  {p.notes?.join("；")}
                  {disagree.length > 0 && (
                    <span className="warn">
                      {" "}离线复核分歧：
                      {disagree.map(d => `${d.axis}轴`).join("、")}
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted small">
        “候选”是模糊点的最可能值，不是精确测量；方括号是综合像素量化、
        读数离散和纸张残差给出的 95% 可信范围。缺色色版显示 “missing”
        且数值为空，不会输出 0 µm。
      </p>
    </div>
  );
}

function ConfBar({ value }) {
  const pct = Math.round((value ?? 0) * 100);
  const color = pct >= 70 ? "#3aaa3c" : pct >= 35 ? "#e6a500" : "#d92b2b";
  return (
    <div className="conf">
      <div className="conf-bar" style={{ width: pct, background: color }} />
      <span>{pct}%</span>
    </div>
  );
}
