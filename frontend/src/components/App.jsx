import React, { useRef, useState } from 'react'
import TargetViewer from './TargetViewer.jsx'
import PlatePanel from './PlatePanel.jsx'
import { api } from '../api.js'

export default function App() {
  const [file, setFile] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [data, setData] = useState(null)  // {scan_id, result}
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [declaredDpi, setDeclaredDpi] = useState('')
  const [dpiSource, setDpiSource] = useState('user')
  const [suggestions, setSuggestions] = useState(null)
  const viewerRef = useRef(null)

  async function onPick(f) {
    setFile(f); setError(null); setData(null); setSuggestions(null)
    setImageUrl(URL.createObjectURL(f))
    setBusy(true)
    try {
      const body = await api.postAnalyze(f, {
        declaredDpi: declaredDpi, dpiSource,
      })
      setData(body)
      setSuggestions(await api.getSuggestions(body.scan_id))
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  async function confirm(plate, payload) {
    await api.postConfirmation(data.scan_id, { plate, reviewer: 'operator', ...payload })
  }

  async function reviewSug(id, decision) {
    await api.reviewSuggestion(data.scan_id, id, decision)
    setSuggestions(await api.getSuggestions(data.scan_id))
  }

  const r = data?.result
  const cal = r?.calibration

  return (
    <div className="app">
      <div className="topbar">
        <h1>印刷套准色标 · 离线复核台</h1>
        <span className="boundary">仅限项目合成/扫描图 · 离线建议不控制设备</span>
        <span style={{ flex: 1 }} />
        <label className="upload">
          扫描 DPI（可选）：
          <input style={{ width: 70 }} value={declaredDpi}
                 onChange={e => setDeclaredDpi(e.target.value)} placeholder="如 200" />
          <select value={dpiSource} onChange={e => setDpiSource(e.target.value)}>
            <option value="user">人工录入</option>
            <option value="exif">扫描头自报</option>
          </select>
        </label>
        <input type="file" accept="image/png,image/jpeg"
               onChange={e => e.target.files[0] && onPick(e.target.files[0])} />
        <button className="secondary" onClick={() => viewerRef.current?.reset()}>
          适应窗口
        </button>
      </div>

      <div className="main">
        {imageUrl
          ? <TargetViewer ref={viewerRef} imageUrl={imageUrl} result={r} />
          : <div className="viewer" style={{ display: 'grid', placeItems: 'center' }}>
              <div className="muted" style={{ textAlign: 'center' }}>
                请选择随项目提供的合成色标/扫描图（REG-TARGET/1）<br />
                <span style={{ fontSize: 11 }}>backend/data/fixtures/*.png</span>
              </div>
            </div>}

        <div className="sidebar">
          {busy && <div className="muted">分析中…</div>}
          {error && <div className="errorbox">{error}</div>}

          {r && (
            <div className="card">
              <h3>纸张级校准（与色版偏移分离）</h3>
              {cal ? <>
                <div className="row"><span className="k">扫描旋转角 θ</span>
                  <span>{cal.angle_deg.toFixed(3)}° ±{cal.angle_uncertainty_deg.toFixed(3)}°</span></div>
                <div className="row"><span className="k">测得尺度</span>
                  <span>{cal.scale_px_per_mm.toFixed(3)} px/mm（≈{cal.measured_dpi.toFixed(1)} DPI）</span></div>
                <div className="row"><span className="k">尺度不确定度</span>
                  <span>{cal.scale_uncertainty_percent.toFixed(2)}%</span></div>
                <div className="row"><span className="k">校准残差</span>
                  <span className={cal.usable ? 'status-ok' : 'status-incomplete'}>
                    {cal.residual_mm.toFixed(4)} mm
                    {' '}{cal.usable ? '（可用）' : '（超门限，仅参考）'}</span></div>
                <div className="row"><span className="k">来源</span>
                  <span className="tag">{cal.method}</span></div>
                <div className="row"><span className="k">DPI 来源</span>
                  <span className="tag">{r.dpi_source}{r.declared_dpi ? `（录入 ${r.declared_dpi}）` : ''}</span></div>
                {r.warnings.map((w, i) =>
                  <div key={i} className="warnbox">{w.message}</div>)}
              </> : <div className="errorbox">{r.error?.message}</div>}
            </div>
          )}

          {r && ['C', 'M', 'Y', 'K'].map(p =>
            <PlatePanel key={p} plate={p} b={r.plates[p]} onConfirm={confirm} />)}

          {suggestions?.length > 0 && (
            <div className="card">
              <h3>离线模型复核（仅建议 · 不自动采纳）</h3>
              {suggestions.map(s => (
                <div key={s.id} style={{ padding: '6px 0', borderBottom: '1px solid var(--line)' }}>
                  <div>
                    <strong>{s.plate}</strong>{' '}
                    <span className="pill">{s.status}</span>{' '}
                    <span className="tag">{s.model_name}@{s.model_version}</span>
                  </div>
                  <div className="muted" style={{ fontSize: 11 }}>
                    {s.suggested_x_um != null
                      ? `建议 ${s.suggested_x_um.toFixed(1)}, ${s.suggested_y_um.toFixed(1)} µm（±${s.suggested_range_um}）`
                      : '无微米建议，需人工判读'}
                  </div>
                  <div className="note">{s.rationale}</div>
                  {s.status === 'pending_review' && (
                    <div style={{ marginTop: 4, display: 'flex', gap: 6 }}>
                      <button className="secondary"
                              onClick={() => reviewSug(s.id, 'accepted_in_part')}>部分采纳</button>
                      <button className="secondary"
                              onClick={() => reviewSug(s.id, 'rejected')}>驳回</button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
