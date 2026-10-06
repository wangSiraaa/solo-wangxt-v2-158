import React, { useState } from 'react'

const PLATE_META = {
  C: { name: '青 C', color: '#00a8dc' },
  M: { name: '品红 M', color: '#dc00dc' },
  Y: { name: '黄 Y', color: '#cfcf00' },
  K: { name: '黑 K（参照）', color: '#c8c8c8' },
}

function fmt(v, d = 1) { return v == null ? '—' : Number(v).toFixed(d) }

export default function PlatePanel({ plate, b, onConfirm }) {
  const meta = PLATE_META[plate]
  const [decision, setDecision] = useState('confirmed')
  const [comment, setComment] = useState('')
  const [sent, setSent] = useState(null)

  const statusClass = `status-${b.status}`
  const off = b.offset_um
  const unc = b.uncertainty_um

  async function submit() {
    await onConfirm(plate, {
      decision,
      offset_x_um: off?.[0] ?? null,
      offset_y_um: off?.[1] ?? null,
      uncertainty_x_um: unc?.[0] ?? null,
      uncertainty_y_um: unc?.[1] ?? null,
      comment: comment || null,
    })
    setSent(decision)
  }

  return (
    <div className="card">
      <h3>
        <span className="plate-dot" style={{ background: meta.color }} />
        {meta.name}
        <span className={`pill ${statusClass}`} style={{ marginLeft: 8 }}>{b.status}</span>
      </h3>

      {b.status === 'reference' ? (
        <div className="muted">参照版：偏移恒为 (0, 0) µm，不参与套准偏差。</div>
      ) : (
        <>
          <table>
            <tbody>
              <tr><th>偏移 x/y</th>
                <td>{off ? `${fmt(off[0])} / ${fmt(off[1])} µm` : '无测量值'}</td></tr>
              <tr><th>不确定度 ±</th>
                <td>{unc ? `${fmt(unc[0])} / ${fmt(unc[1])} µm` : '—'}</td></tr>
              <tr><th>可信范围</th>
                <td>{b.confidence_range_um != null
                    ? `±${fmt(b.confidence_range_um)} µm` : '—'}</td></tr>
              <tr><th>候选数</th><td>{b.candidates.length}</td></tr>
              <tr><th>置信度</th><td>{fmt(b.confidence, 2)}</td></tr>
            </tbody>
          </table>

          {b.status === 'missing' && (
            <div className="errorbox">缺色：未找到合格圆点。
              {b.confidence_range_um != null &&
                ` 仅给出搜索半径 ±${fmt(b.confidence_range_um)}µm，不是测量精度。`}
            </div>
          )}
          {(b.status === 'incomplete' || b.status === 'ambiguous') && (
            <div className="warnbox">
              {b.status === 'ambiguous' ? '存在多个候选点，偏移值有歧义。' : '圆点残缺，质心可能偏移。'}
              请结合放大视图人工判读。
            </div>
          )}
          {b.notes?.map((n, i) => <div key={i} className="note">{n}</div>)}

          <div style={{ marginTop: 8, display: 'flex', gap: 6, alignItems: 'center' }}>
            <select value={decision} onChange={e => setDecision(e.target.value)}>
              <option value="confirmed">确认算法值</option>
              <option value="corrected">人工修正</option>
              <option value="rejected">驳回</option>
            </select>
            <input style={{ flex: 1 }} placeholder="备注（可选）"
                   value={comment} onChange={e => setComment(e.target.value)} />
            <button onClick={submit} disabled={sent != null}>提交</button>
          </div>
          {sent && <div className="note" style={{ marginTop: 4 }}>
            已记录人工裁决：{sent}（不回写算法原始结果）</div>}
        </>
      )}
    </div>
  )
}
