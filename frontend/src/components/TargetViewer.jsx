import React, { useEffect, useImperativeHandle, useRef, useState, forwardRef } from 'react'

const PLATE_COLORS = {
  C: '#00a8dc', M: '#dc00dc', Y: '#e6e600', K: '#c8c8c8',
}

/**
 * 局部放大查看器：滚轮缩放、拖拽平移；叠加校准尺、色版候选点、不确定范围圆。
 * 所有叠层坐标用原图像素，随视图变换。
 */
const TargetViewer = forwardRef(function TargetViewer({ imageUrl, result }, ref) {
  const canvasRef = useRef(null)
  const wrapRef = useRef(null)
  const imgRef = useRef(null)
  const [view, setView] = useState({ scale: 1, tx: 0, ty: 0 })
  const drag = useRef(null)

  useImperativeHandle(ref, () => ({
    reset: () => fitView(),
    zoomBy: (f) => setView(v => ({ ...v, scale: clampScale(v.scale * f) })),
  }))

  function clampScale(s) { return Math.min(40, Math.max(0.05, s)) }

  function fitView() {
    const img = imgRef.current, wrap = wrapRef.current
    if (!img || !wrap) return
    const s = Math.min(wrap.clientWidth / img.naturalWidth,
                       wrap.clientHeight / img.naturalHeight)
    setView({
      scale: s,
      tx: (wrap.clientWidth - img.naturalWidth * s) / 2,
      ty: (wrap.clientHeight - img.naturalHeight * s) / 2,
    })
  }

  useEffect(() => {
    const img = new Image()
    img.onload = () => { imgRef.current = img; fitView(); draw() }
    img.src = imageUrl
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageUrl])

  useEffect(() => { draw() }, [view, result])

  function toScreen(p) {
    return [p[0] * view.scale + view.tx, p[1] * view.scale + view.ty]
  }

  function draw() {
    const cv = canvasRef.current, img = imgRef.current
    if (!cv || !img) return
    const wrap = wrapRef.current
    cv.width = wrap.clientWidth; cv.height = wrap.clientHeight
    const ctx = cv.getContext('2d')
    ctx.clearRect(0, 0, cv.width, cv.height)
    ctx.save()
    ctx.translate(view.tx, view.ty)
    ctx.scale(view.scale, view.scale)
    // 原图
    ctx.drawImage(img, 0, 0)
    if (result) drawOverlay(ctx)
    ctx.restore()
  }

  function drawOverlay(ctx) {
    const cal = result.calibration
    if (!cal) {
      ctx.fillStyle = '#e06060'; ctx.font = '14px sans-serif'
      ctx.fillText('校准失败：' + (result.error?.message || ''), 20, 30)
      return
    }
    const s = cal.scale_px_per_mm

    // 校准尺：候选点（K 控制点/pip）信息未直接给像素坐标，这里画"印刷坐标轴"
    // 用 measured offset 反推：画色版候选点
    ctx.lineWidth = 1 / view.scale
    for (const [plate, b] of Object.entries(result.plates)) {
      if (plate === 'K') continue
      const col = PLATE_COLORS[plate]
      b.candidates.forEach((c, idx) => {
        const [x, y] = c.centroid_px
        ctx.beginPath()
        ctx.strokeStyle = col
        ctx.globalAlpha = idx === b.selected_index ? 1 : 0.5
        ctx.lineWidth = (idx === b.selected_index ? 2 : 1) / view.scale
        ctx.arc(x, y, (0.55 * 200 / 25.4) / 2, 0, Math.PI * 2)
        ctx.stroke()
        // 可信范围
        if (idx === b.selected_index && b.uncertainty_um) {
          const ru = Math.max(b.uncertainty_um) / 1000 * s
          ctx.globalAlpha = 0.35
          ctx.beginPath(); ctx.arc(x, y, ru, 0, Math.PI * 2); ctx.stroke()
        }
      })
      // 选中点十字
      if (b.selected_index != null) {
        const c = b.candidates[b.selected_index]
        const [x, y] = c.centroid_px
        ctx.globalAlpha = 1
        ctx.strokeStyle = col
        ctx.beginPath()
        ctx.moveTo(x - 8, y); ctx.lineTo(x + 8, y)
        ctx.moveTo(x, y - 8); ctx.lineTo(x, y + 8)
        ctx.stroke()
      }
      ctx.globalAlpha = 1
    }
  }

  function onWheel(e) {
    e.preventDefault()
    const rect = canvasRef.current.getBoundingClientRect()
    const mx = e.clientX - rect.left, my = e.clientY - rect.top
    const f = e.deltaY < 0 ? 1.15 : 1 / 1.15
    setView(v => {
      const ns = clampScale(v.scale * f)
      const k = ns / v.scale
      return { scale: ns, tx: mx - (mx - v.tx) * k, ty: my - (my - v.ty) * k }
    })
  }

  function onDown(e) {
    drag.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty }
  }
  function onMove(e) {
    if (!drag.current) return
    setView(v => ({ ...v,
      tx: drag.current.tx + (e.clientX - drag.current.x),
      ty: drag.current.ty + (e.clientY - drag.current.y) }))
  }
  function onUp() { drag.current = null }

  return (
    <div ref={wrapRef} className="viewer"
         onWheel={onWheel} onMouseMove={onMove} onMouseUp={onUp}
         onMouseLeave={onUp}>
      <canvas ref={canvasRef} onMouseDown={onDown}
              style={{ cursor: drag.current ? 'grabbing' : 'grab' }} />
      <div className="hint" style={{ position: 'absolute', left: 10, bottom: 8 }}>
        滚轮缩放 · 拖拽平移 · 圆圈=候选色标点 · 大圆=不确定范围（µm）
      </div>
    </div>
  )
})

export default TargetViewer
