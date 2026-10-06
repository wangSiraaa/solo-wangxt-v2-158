import React, { useState } from "react";

/** Human QC confirmation.  Machine results are never auto-confirmed; this is
 * the only way a row gets a decision, and corrections are stored separately. */
export default function ConfirmPanel({ result, onConfirmed }) {
  const [operator, setOperator] = useState("");
  const [decision, setDecision] = useState("confirmed");
  const [dx, setDx] = useState("");
  const [dy, setDy] = useState("");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function submit() {
    setErr("");
    if (!operator.trim()) { setErr("请填写操作员"); return; }
    const body = {
      operator_id: operator.trim(), decision,
      reason: reason.trim(),
      corrected_dx_um: decision === "corrected" ? Number(dx) : null,
      corrected_dy_um: decision === "corrected" ? Number(dy) : null,
    };
    if (decision === "corrected"
        && (!Number.isFinite(body.corrected_dx_um)
            || !Number.isFinite(body.corrected_dy_um))) {
      setErr("修正决定必须填写人工测得的 dx/dy（µm）"); return;
    }
    setBusy(true);
    const res = await fetch(`/api/runs/${result.run_id}/confirm`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    setBusy(false);
    if (!res.ok) { setErr((await res.json()).detail ?? "提交失败"); return; }
    onConfirmed?.(await res.json());
  }

  const prior = result.confirmations ?? [];

  return (
    <div className="panel">
      <h3>人工确认</h3>
      {prior.length > 0 && (
        <ul className="prior-confirms">
          {prior.map((c, i) => (
            <li key={i}>
              <b>{c.operator_id}</b>：{c.decision}
              {c.decision === "corrected"
                && ` → dx ${c.corrected_dx_um}, dy ${c.corrected_dy_um} µm`}
              {c.reason && `（${c.reason}）`}
              <span className="muted"> {c.created_at}</span>
            </li>
          ))}
        </ul>
      )}
      <div className="confirm-row">
        <input placeholder="操作员工号" value={operator}
               onChange={e => setOperator(e.target.value)} />
        <select value={decision} onChange={e => setDecision(e.target.value)}>
          <option value="confirmed">确认机器结果</option>
          <option value="corrected">人工修正</option>
          <option value="rejected">拒绝（重测）</option>
        </select>
        {decision === "corrected" && (
          <>
            <input type="number" placeholder="人工 dx µm" value={dx}
                   onChange={e => setDx(e.target.value)} />
            <input type="number" placeholder="人工 dy µm" value={dy}
                   onChange={e => setDy(e.target.value)} />
          </>
        )}
        <input placeholder="备注（可选）" value={reason}
               onChange={e => setReason(e.target.value)} />
        <button disabled={busy} onClick={submit}>
          {busy ? "提交中…" : "提交确认"}
        </button>
      </div>
      {err && <div className="error">{err}</div>}
      <p className="muted small">
        系统只做离线分析与复核，不向任何印刷设备发送控制指令；
        人工修正不会覆盖机器原始读数，二者在数据库中分别保存。
      </p>
    </div>
  );
}
