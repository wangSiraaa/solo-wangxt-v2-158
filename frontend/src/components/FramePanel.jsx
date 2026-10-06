import React from "react";

/** Paper frame (rotation/scale) block — visually separated from plate
 * shifts so a skewed scan is never mistaken for plate misregistration. */
export default function FramePanel({ result }) {
  if (!result) return null;
  const f = result.frame;
  const img = result.image;
  const prov = result.provenance ?? {};
  const steps = (prov.steps ?? []);
  const fid = steps.find(s => s.step === "paper_frame");
  const ruler = steps.find(s => s.step === "ruler");

  return (
    <div className="panel frame-panel">
      <h3>纸张校正（与色版偏移分开）</h3>
      <div className="frame-grid">
        <Metric label="纸张旋转角" value={`${f.rotation_deg.toFixed(3)}°`}
                hint="由四角黑色定位点估计，已在读取前校正" />
        <Metric label="扫描分辨率"
                value={`${img.measured_ppi?.toFixed(1) ?? "—"} PPI`}
                hint={`申报：${img.declared_scan_ppi ?? "未填写"} PPI`} />
        <Metric label="尺度 scan/native"
                value={f.scale_vs_native.toFixed(4)}
                hint="定位点 70% + 校准尺 30% 融合" />
        <Metric label="定位质量"
                value={`${Math.round(f.quality * 100)}%`} />
        <Metric label="残差共同旋转"
                value={`${f.residual_rotation_deg.toFixed(4)}°`}
                hint="各色版共同项，已扣除，不计入色版" />
        <Metric label="残差共同尺度"
                value={f.residual_scale.toExponential(2)}
                hint="同上" />
      </div>
      <details>
        <summary>校准尺与定位点来源（可审计）</summary>
        <pre className="prov">{JSON.stringify({ fid, ruler }, null, 2)}</pre>
      </details>
      <p className="muted small">
        旋转角超过 ±8° 或定位点少于 3 个时，服务直接拒绝给色版结论；
        校准尺与定位点尺度不一致（&gt;3%）会降级为候选并标记。
      </p>
    </div>
  );
}

function Metric({ label, value, hint }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {hint && <div className="metric-hint muted">{hint}</div>}
    </div>
  );
}
