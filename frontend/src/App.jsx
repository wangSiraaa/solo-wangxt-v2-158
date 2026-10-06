import React, { useEffect, useState } from "react";
import ChartCanvas from "./components/ChartCanvas.jsx";
import MarkZoom from "./components/MarkZoom.jsx";
import PlateTable from "./components/PlateTable.jsx";
import FramePanel from "./components/FramePanel.jsx";
import ConfirmPanel from "./components/ConfirmPanel.jsx";

export default function App() {
  const [datasets, setDatasets] = useState([]);
  const [selected, setSelected] = useState("");
  const [upload, setUpload] = useState(null);
  const [ppi, setPpi] = useState("");
  const [result, setResult] = useState(null);
  const [review, setReview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [mark, setMark] = useState({ row: 0, col: 0 });
  const [err, setErr] = useState("");

  useEffect(() => {
    fetch("/api/datasets").then(r => r.json()).then(ds => {
      setDatasets(ds);
      if (ds.length) setSelected(ds[0].name);
    });
  }, []);

  async function analyze(body) {
    setBusy(true); setErr(""); setReview(null);
    try {
      const res = await fetch("/api/analyze", { method: "POST", body });
      if (!res.ok) throw new Error((await res.json()).detail ?? "分析失败");
      setResult(await res.json());
    } catch (e) {
      setErr(String(e.message ?? e));
      setResult(null);
    } finally { setBusy(false); }
  }

  function runDataset() {
    const fd = new FormData();
    fd.append("dataset", selected);
    if (ppi) fd.append("declared_ppi", ppi);
    analyze(fd);
  }

  function runUpload() {
    if (!upload) return;
    const fd = new FormData();
    fd.append("file", upload);
    if (ppi) fd.append("declared_ppi", ppi);
    analyze(fd);
  }

  async function runReview() {
    const r = await fetch(`/api/runs/${result.run_id}/review`,
                          { method: "POST" });
    setReview(await r.json());
  }

  return (
    <div className="app">
      <header>
        <h1>PRT-1 套准色标质检</h1>
        <span className="subtitle">
          纸张旋转/尺度校正与色版偏移分离 · 模糊点给候选与可信范围 ·
          仅离线复核，不控制设备
        </span>
      </header>

      <section className="panel controls">
        <div className="control-block">
          <label>项目合成样张 / 扫描图</label>
          <select value={selected}
                  onChange={e => setSelected(e.target.value)}>
            {datasets.map(d => (
              <option key={d.name} value={d.name}>
                {d.name}
                {d.truth?.scan_ppi ? `（${d.truth.scan_ppi} PPI 扫描）` : ""}
              </option>
            ))}
          </select>
          <input type="number" placeholder="申报 PPI（可选）" value={ppi}
                 onChange={e => setPpi(e.target.value)} />
          <button disabled={busy || !selected} onClick={runDataset}>
            分析样张
          </button>
        </div>
        <div className="control-block">
          <label>或上传扫描图（限本项目格式 PRT-1）</label>
          <input type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp"
                 onChange={e => setUpload(e.target.files?.[0] ?? null)} />
          <button disabled={busy || !upload} onClick={runUpload}>
            上传并分析
          </button>
        </div>
        {busy && <div className="muted">分析中…</div>}
        {err && <div className="error">{err}</div>}
      </section>

      {result && (
        <>
          <FramePanel result={result} />
          <div className={`status-banner status-${result.status}`}>
            机器结论：<b>{result.status === "ok" ? "可用"
                         : result.status === "candidates"
                           ? "仅候选，需人工判定"
                           : result.status}</b>
            {" "}（run {result.run_id}，
            检测器 {result.provenance.detector_version}）
            <button className="link-btn" onClick={runReview}>
              运行离线模型复核
            </button>
            {review && (
              <span className={review.verdict === "consistent"
                              ? "review-ok" : "review-warn"}>
                {" "}复核结论：{review.verdict === "consistent"
                  ? "一致" : "存在分歧，需人工复核"}
              </span>
            )}
          </div>

          <div className="main-grid">
            <div>
              <ChartCanvas
                result={result}
                overlayUrl={`/api/runs/${result.run_id}/overlay.png`}
                onPickMark={setMark}
              />
            </div>
            <MarkZoom result={result} row={mark.row} col={mark.col} />
          </div>

          <PlateTable result={result} review={review} />
          <ConfirmPanel result={result}
                        onConfirmed={r => setResult({ ...result,
                                                      confirmations:
                                                        r.confirmations })} />
        </>
      )}
    </div>
  );
}
