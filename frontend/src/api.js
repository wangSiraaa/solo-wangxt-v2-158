const BASE = ''

async function postAnalyze(file, { declaredDpi, dpiSource }) {
  const fd = new FormData()
  fd.append('image', file)
  fd.append('name', file.name)
  if (declaredDpi != null && declaredDpi !== '') {
    fd.append('declared_dpi', declaredDpi)
    fd.append('dpi_source', dpiSource || 'user')
  }
  const r = await fetch(`${BASE}/api/scans/analyze`, { method: 'POST', body: fd })
  if (!r.ok) throw new Error(`analyze ${r.status}`)
  return r.json()
}

async function postConfirmation(scanId, body) {
  const r = await fetch(`${BASE}/api/scans/${scanId}/confirmations`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`confirm ${r.status}`)
  return r.json()
}

async function getSuggestions(scanId) {
  const r = await fetch(`${BASE}/api/scans/${scanId}/model-suggestions`)
  if (!r.ok) throw new Error('suggestions')
  return r.json()
}

async function reviewSuggestion(scanId, id, decision) {
  const fd = new FormData()
  fd.append('decision', decision)
  const r = await fetch(
    `${BASE}/api/scans/${scanId}/model-suggestions/${id}/review`,
    { method: 'POST', body: fd })
  return r.json()
}

export const api = { postAnalyze, postConfirmation, getSuggestions, reviewSuggestion }
