"""K（黑）校准尺检测与纸张级仿射估计。

严格只用黑色要素（方框 + 两个非对称角点 pip + 两个 K 圆点），因此估计出的
旋转角/尺度不包含任何 C/M/Y 色版位移信息。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .spec import (
    DOT_DIAMETER_MM,
    DOT_NOMINAL_MM,
    FRAME_SIDE_MM,
    PIP_CENTERS_MM,
    PIP_SIDE_MM,
)


@dataclass
class CalibError:
    code: str
    message: str


def _dark_mask(image_bgr: np.ndarray, strict: bool = False) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    # strict：只收 K 实地（用于找框，抗暗斑桥接）；否则放宽覆盖抗锯齿边缘
    mask = (l < (60 if strict else 110)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    return mask


def _luminance(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)


def _darkness(lum: np.ndarray) -> np.ndarray:
    # 框线拟合用：实地墨（L 低）权重高，抗锯齿边渐降；纸白为 0
    d = np.clip(1.0 - (lum - 40.0) / 150.0, 0.0, 1.0)
    return (d * d).astype(np.float32)


def _dot_darkness(lum: np.ndarray) -> np.ndarray:
    # 圆点质心用：阈值更宽，覆盖旋转/缩放后灰度抬升到 L~150 的墨像素
    d = np.clip(1.0 - (lum - 60.0) / 170.0, 0.0, 1.0)
    return (d * d).astype(np.float32)


# ---------------------------------------------------------------------------
# 方框定位：Canny + 概率霍夫 → 线段 → 两正交方向聚类 → 等距平行线对成框
# ---------------------------------------------------------------------------

def _merge_collinear(segs, gap_tol=12.0, max_normal_gap=3.5):
    """把共线且端到端间隙小的霍夫线段合并（Canny 常把一条边切成几段）。"""
    items = [tuple(map(float, s)) for s in segs]

    def direction(s):
        d = np.array([s[2] - s[0], s[3] - s[1]])
        return d / max(np.linalg.norm(d), 1e-9)

    merged = True
    cur = items
    while merged:
        merged = False
        used = [False] * len(cur)
        out = []
        for i in range(len(cur)):
            if used[i]:
                continue
            acc = list(cur[i])
            used[i] = True
            changed = True
            while changed:
                changed = False
                d = direction(acc)
                pa = np.array(acc[:2])
                pb = np.array(acc[2:])
                n = np.array([-d[1], d[0]])
                for j in range(len(cur)):
                    if used[j]:
                        continue
                    s = cur[j]
                    dj = direction(s)
                    if abs(np.dot(dj, d)) < 0.9:
                        continue
                    qa, qb = np.array(s[:2]), np.array(s[2:])
                    # 法向对齐
                    if max(abs(np.dot(qa - pa, n)), abs(np.dot(qb - pa, n))) > max_normal_gap:
                        continue
                    # 沿线投影区间重叠或间隙小
                    def proj(pt):
                        return float(np.dot(pt - pa, d))
                    lo, hi = sorted((0.0, np.dot(pb - pa, d)))
                    qa_p, qb_p = proj(qa), proj(qb)
                    qlo, qhi = sorted((qa_p, qb_p))
                    if qlo <= hi + gap_tol and lo - gap_tol <= qhi:
                        used[j] = True
                        all_pts = [pa, pb, pa + d * qlo, pa + d * qhi]
                        pa = min(all_pts, key=lambda pt: np.dot(pt - np.array(acc[:2]), d))
                        pb = max(all_pts, key=lambda pt: np.dot(pt - np.array(acc[:2]), d))
                        acc = [pa[0], pa[1], pb[0], pb[1]]
                        changed = True
            out.append(tuple(acc))
        cur = out
    return np.asarray(cur, dtype=np.float64) if cur else np.zeros((0, 4), np.float64)


def _hough_segments(image_bgr: np.ndarray) -> np.ndarray:
    """返回 (N,4) 线段 x1,y1,x2,y2；多组 Canny 阈值合并 + 共线拼接。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    segs = []
    H, W = gray.shape
    # 框边在低 DPI 图上可能只有 40~50px；先短阈值多召回，矩形装配时再严格筛选
    min_len = max(20, min(H, W) * 0.10)
    for lo, hi in ((30, 90), (50, 140), (20, 70)):
        edges = cv2.Canny(gray, lo, hi, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 720.0,
                                threshold=max(12, int(min_len * 0.4)),
                                minLineLength=min_len, maxLineGap=10)
        if lines is not None:
            segs.append(lines.reshape(-1, 4))
    if not segs:
        return np.zeros((0, 4), np.float64)
    L = np.vstack(segs).astype(np.float64)
    # 共线拼接（不同 Canny 阈值产生的重复/碎片）
    L = _merge_collinear(L)
    # 去重：平行 + 法向间距小 + 沿线投影大部分重叠 → 保留较长者
    keep = []
    for l in sorted(L, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
        d = np.array([l[2] - l[0], l[3] - l[1]])
        lenn = np.linalg.norm(d)
        d /= lenn
        n = np.array([-d[1], d[0]])
        p = (l[:2] + l[2:]) / 2
        dup = False
        for k in keep:
            dk = np.array([k[2] - k[0], k[3] - k[1]])
            lk = np.linalg.norm(dk)
            dk /= lk
            if abs(np.dot(dk, d)) < 0.95:
                continue
            nk = np.array([-dk[1], dk[0]])
            pk = (k[:2] + k[2:]) / 2
            if abs(np.dot(p - pk, nk)) > 3.0:
                continue
            # 沿线投影区间重叠比例
            def span(seg, dref, origin):
                t = [np.dot(seg[:2] - origin, dref), np.dot(seg[2:] - origin, dref)]
                return min(t), max(t)
            a0, a1 = span(l, d, pk)
            b0, b1 = span(k, d, pk)
            ov = max(0.0, min(a1, b1) - max(a0, b0))
            if ov > 0.5 * min(a1 - a0, b1 - b0):
                dup = True
                break
        if not dup:
            keep.append(l)
    return np.asarray(keep)


def _line_from_segment(seg):
    """线段 -> 单位方向 d 与直线 (rho: 法向方程 n·x = rho, 法向 n, 中心 p, 长度)。"""
    a, b = seg[:2], seg[2:]
    d = b - a
    length = float(np.linalg.norm(d))
    d = d / length
    n = np.array([-d[1], d[0]])
    p = (a + b) / 2
    rho = float(np.dot(n, p))
    return d, n, rho, p, length


def _fit_tls(points: np.ndarray, weights: np.ndarray | None = None):
    """加权总最小二乘直线：返回 (单位方向 d, 线上点 p)。退化时返回 None。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return None
    if weights is None:
        w = np.full(len(pts), 1.0 / len(pts))
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = np.nan_to_num(w, nan=0.0)
        if w.sum() <= 1e-12:
            w = np.full(len(pts), 1.0 / len(pts))
        else:
            w = w / w.sum()
    mean = (pts * w[:, None]).sum(axis=0)
    cen = (pts - mean) * np.sqrt(w)[:, None]
    if not np.isfinite(cen).all():
        return None
    try:
        _, sv, vt = np.linalg.svd(cen)
    except np.linalg.LinAlgError:
        return None
    if sv[0] < 1e-6:
        return None
    return vt[0], mean


def _cluster_parallel(segments, d_ref, merge_tol_px: float,
                      edge_thickness_px: float = 4.0):
    """沿固定方向 d_ref（法向 n_ref=perp(d_ref)）把线段聚类成直线。

    segments: [(p_mid, length, seg_direction)]。
    同一条粗墨线的内/外 Canny 边（有符号距离相差 ~线宽）用 edge_thickness_px
    以内的"成对、且中间确实更暗"来识别并合并；孤立线段直接保留为候选。
    返回 [(rho, p_point, support_len, center_line_rho)]，按 rho 排序。
    """
    n_ref = np.array([-d_ref[1], d_ref[0]])
    items = []
    for p, length, d in segments:
        rho = float(np.dot(n_ref, p))
        items.append((rho, p, length))
    items.sort(key=lambda t: t[0])

    used = [False] * len(items)
    clusters = []
    for i, (rho_i, p_i, len_i) in enumerate(items):
        if used[i]:
            continue
        group_idx = [i]
        used[i] = True
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            rho_j = items[j][0]
            if 0 < rho_j - rho_i <= edge_thickness_px:
                group_idx.append(j)
                used[j] = True
        # 合并同组：rho 取内/外边中线（即墨线中心）
        if len(group_idx) >= 2:
            rhos = np.array([items[k][0] for k in group_idx])
            w = np.array([items[k][2] for k in group_idx], dtype=float)
            center_rho = float((rhos.min() + rhos.max()) / 2.0)
        else:
            center_rho = rho_i
        # 记录该直线沿线的覆盖区间（端点投影），用于排除"多个污点碰巧共线"
        t_proj = []
        pts, wts = [], []
        for k in group_idx:
            _, p, length = items[k]
            pts += [p - d_ref * length / 2, p + d_ref * length / 2]
            wts += [length, length]
            t_proj += [float(np.dot(p - d_ref * length / 2, d_ref)),
                       float(np.dot(p + d_ref * length / 2, d_ref))]
        fit = _fit_tls(np.asarray(pts), np.asarray(wts))
        if fit is None:
            p_fit = n_ref * center_rho
        else:
            d_fit, p_fit = fit
            if np.dot(d_fit, d_ref) < 0:
                d_fit = -d_fit
        clusters.append({
            "rho": float(np.dot(n_ref, p_fit)),
            "center_rho": center_rho,
            "p": p_fit,
            "support": float(sum(items[k][2] for k in group_idx)),
            "t_min": min(t_proj), "t_max": max(t_proj),
        })
    clusters.sort(key=lambda c: c["rho"])
    return clusters  # list of dict


def _find_rectangle(segs: np.ndarray):
    """从霍夫线段中找两组正交方向、等距平行线对组成的方框。

    返回 dict(d0, r1, r2, s1, s2, support)，四条墨线中心线方程为 n0·x=r、
    d0·x=s（有符号）；或 None。
    """
    if len(segs) < 4:
        return None
    lines = [_line_from_segment(s) for s in segs]
    lines.sort(key=lambda t: -t[4])

    def try_direction(d0):
        d0 = d0 / np.linalg.norm(d0)
        n0 = np.array([-d0[1], d0[0]])
        seg_v, seg_h = [], []
        for d, n, rho, p, length in lines:
            if abs(np.dot(d, d0)) > 0.90:
                seg_v.append((p, length, d))
            elif abs(np.dot(d, n0)) > 0.90:
                seg_h.append((p, length, d))
        if len(seg_v) < 2 or len(seg_h) < 2:
            return None
        thickness = 7.0
        cv = _cluster_parallel(seg_v, d0, thickness)   # 距离沿 n0
        # 横边：法向 perp(n0)=-d0；聚类距离沿 -d0。后续统一用 (-dv)·x=s 的方程，
        # 因此这里保持 ch 距离为 -d0 方向（不做取反）。
        ch = _cluster_parallel(seg_h, n0, thickness)
        ch.sort(key=lambda c: c["rho"])
        if len(cv) < 2 or len(ch) < 2:
            return None

        def best_pair(clusters, edge_dir, normal_dir):
            """选对边：优先最大的合法间距（外框优先于框内污点线），
            再按支持长度打分。这样"真顶边+真底边"胜过"污点线+真底边"。"""
            best = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    ci, cj = clusters[i], clusters[j]
                    gap = cj["center_rho"] - ci["center_rho"]
                    if gap < 25:
                        continue
                    if abs(np.dot(ci["p"] - cj["p"], edge_dir)) > gap * 0.45:
                        continue
                    span_i = ci["t_max"] - ci["t_min"]
                    span_j = cj["t_max"] - cj["t_min"]
                    if max(span_i, span_j) < gap * 0.7:
                        continue
                    overlap = min(ci["t_max"], cj["t_max"]) - max(ci["t_min"], cj["t_min"])
                    if overlap < gap * 0.4:
                        continue
                    support = ci["support"] + cj["support"]
                    # 外框优先：分数 = 间距（主导）+ 小的支持加成
                    score = gap + support * 0.05
                    cand = (score, gap, ci, cj)
                    if best is None or cand[0] > best[0]:
                        best = cand
            return best

        # 竖边（方向 d0，法向 n0）；横边（方向 n0，法向 -d0）
        pair_v = best_pair(cv, d0, n0)
        pair_h = best_pair(ch, n0, -d0)
        if pair_v is None or pair_h is None:
            return None
        gv1, gv2 = pair_v[2], pair_v[3]
        gh1, gh2 = pair_h[2], pair_h[3]

        def has_inner_parallel(outer_pair, all_clusters, normal, margin_frac=0.15):
            """真框是最外层：两条外边之间不应还有一条长平行线（污点/中线会形成）。"""
            a, b = outer_pair[2], outer_pair[3]
            rlo, rhi = sorted((a["center_rho"], b["center_rho"]))
            margin = (rhi - rlo) * margin_frac
            for c in all_clusters:
                if c is a or c is b:
                    continue
                r = c["center_rho"]
                if rlo + margin < r < rhi - margin and c["support"] > (rhi - rlo) * 0.55:
                    return True
            return False

        # 若配对的"框"内部还夹着另一条长平行线，说明这不是外框（污点连线被误配）。
        # 仅在横边方向检查：案例中污点线是水平的；真框内部只有圆点/方块，无长线。
        if has_inner_parallel(pair_h, ch, n0):
            return None
        gv = abs(gv1["center_rho"] - gv2["center_rho"])
        gh = abs(gh1["center_rho"] - gh2["center_rho"])
        if abs(gv - gh) / max(gv, gh) > 0.15:
            return None
        side = (gv + gh) / 2
        # 角度精修：只取"接近完整边长"的霍夫长边（碎片方向噪声大），
        # 两组长边方向按长度加权平均，横边强制与竖边垂直。
        min_face = side * 0.85
        dv_segs = [l for l in lines
                   if abs(np.dot(l[0], d0)) > 0.95 and l[4] >= min_face]
        dh_segs = [l for l in lines
                   if abs(np.dot(l[0], n0)) > 0.95 and l[4] >= min_face]
        if not dv_segs or not dh_segs:
            d_final = d0
        else:
            def avg_dir(seg_list, ref):
                sx = sy = wsum = 0.0
                for d, n, rho, p, length in seg_list:
                    w = length ** 2  # 长边权重平方：最长的完整边方向最可信
                    dd = d if np.dot(d, ref) > 0 else -d
                    sx += dd[0] * w; sy += dd[1] * w; wsum += w
                v = np.array([sx, sy]) / wsum
                return v / np.linalg.norm(v)
            dv_fit = avg_dir(dv_segs, d0)
            dh_raw = avg_dir(dh_segs, n0)
            perp = np.array([-dv_fit[1], dv_fit[0]])
            dh_fit = perp if np.dot(perp, dh_raw) >= 0 else -perp
            # 以竖边为主（dv_segs 通常更长），横边只做小幅正交校正
            d_final = (dv_fit * 2.0 + np.array([-dh_fit[1], dh_fit[0]])) / 3.0
            d_final /= np.linalg.norm(d_final)
        # 用最终方向重算四条直线的有符号距离（中心沿用聚类点）
        n_final = np.array([-d_final[1], d_final[0]])
        def refit_rho(c, along_normal):
            return float(np.dot(along_normal, c["p"]))
        r1n = refit_rho(gv1, n_final)
        r2n = refit_rho(gv2, n_final)
        s1n = refit_rho(gh1, -d_final)
        s2n = refit_rho(gh2, -d_final)
        if r1n > r2n: r1n, r2n = r2n, r1n
        if s1n > s2n: s1n, s2n = s2n, s1n
        return dict(d0=d_final,
                    r1=r1n, r2=r2n, s1=s1n, s2=s2n,
                    support=pair_v[0] + pair_h[0], side=side)
    # 方向候选：把长线方向归一到 [-90,90) 后取最强种子（去掉 180° 翻转歧义）
    cand = None
    seeds = []
    for d, n, rho, p, length in lines[:8]:
        a = np.arctan2(d[1], d[0])
        if a > np.pi / 2:
            a -= np.pi
        elif a < -np.pi / 2:
            a += np.pi
        seeds.append((length, a))
    seeds.sort(reverse=True)
    tried = set()
    for _, a in seeds:
        for off in (0.0,):
            key = round(float(a + off), 3)
            if key in tried:
                continue
            tried.add(key)
            dd = np.array([np.cos(a + off), np.sin(a + off)])
            r = try_direction(dd)
            if r and (cand is None or r["support"] > cand["support"]):
                cand = r
    # 种子方向 ±7° 细搜
    if cand is not None:
        base = np.arctan2(cand["d0"][1], cand["d0"][0])
        for off in np.deg2rad(np.arange(-7, 7.1, 0.5)):
            if abs(off) < 0.3:
                continue
            a = base + off
            r = try_direction(np.array([np.cos(a), np.sin(a)]))
            if r and r["support"] > cand["support"]:
                cand = r
    return cand


def _line_profile_peaks(dark, d, p, half_len, strip, coverage=0.62):
    """沿方向 d 的墨线，在每个切向位置 t 上找法向暗峰中心。

    返回 (ts, offsets, confidences)。
    """
    h, w = dark.shape
    n = np.array([-d[1], d[0]])
    lim = half_len * coverage
    ts = np.arange(-lim, lim + 0.5, 0.75)
    offsets = np.arange(-strip, strip + 0.01, 0.25)
    T, O = np.meshgrid(ts, offsets, indexing="ij")
    X = p[0] + d[0] * T + n[0] * O
    Y = p[1] + d[1] * T + n[1] * O
    Xi = np.clip(X, 0, w - 1.001)
    Yi = np.clip(Y, 0, h - 1.001)
    x0 = np.floor(Xi).astype(np.int32); y0 = np.floor(Yi).astype(np.int32)
    fx, fy = Xi - x0, Yi - y0
    V = (dark[y0, x0] * (1 - fx) * (1 - fy)
         + dark[y0, x0 + 1] * fx * (1 - fy)
         + dark[y0 + 1, x0] * (1 - fx) * fy
         + dark[y0 + 1, x0 + 1] * fx * fy)
    out_t, out_o, out_c = [], [], []
    for i, t in enumerate(ts):
        v = V[i]
        vv = np.clip(v - v.max() * 0.35, 0, None)
        if vv.sum() < 0.25:
            continue
        out_t.append(t)
        out_o.append(float((vv * offsets).sum() / vv.sum()))
        out_c.append(float(v.max()))
    return np.array(out_t), np.array(out_o), np.array(out_c)


def _refine_rect_on_darkness(dark, d0, side_est, r1, r2, s1, s2,
                             strip=2.0, iterations=10):
    """联合精修矩形（强约束：对边平行、间距同为 side）。

    自由量：θ（旋转）、cr/cs（中心在 nv/-dv 上的位置）、side。
    每条边在每个切向位置采样法向暗峰，用中位残差更新中心/边长，
    用峰位置斜率更新角度。返回 (theta, r1, r2, s1, s2)（中心线方程参数）。
    """
    theta = float(np.arctan2(d0[1], d0[0]))
    cr = (r1 + r2) / 2.0
    cs = (s1 + s2) / 2.0
    side = (r2 - r1 + s2 - s1) / 2.0

    for _ in range(iterations):
        dv = np.array([np.cos(theta), np.sin(theta)])
        nv = np.array([-dv[1], dv[0]])
        anchors = {
            "L": nv * (cr - side / 2),
            "R": nv * (cr + side / 2),
            "T": (-dv) * (cs + side / 2),
            "B": (-dv) * (cs - side / 2),
        }
        med = {}
        for name, anchor in anchors.items():
            ed = dv if name in ("L", "R") else nv
            tt, oo, cc = _line_profile_peaks(dark, ed, anchor, side_est / 2, strip)
            if len(tt) < 6:
                continue
            order = np.argsort(oo)
            med[name] = float(np.interp(
                0.5, np.cumsum(cc[order]) / cc.sum(), oo[order]))
        has_v = "L" in med and "R" in med
        has_h = "T" in med and "B" in med
        dcr = (med["L"] + med["R"]) / 2 if has_v else 0.0
        dcs = (med["T"] + med["B"]) / 2 if has_h else 0.0
        dside = 0.0
        if has_v:
            dside += med["R"] - med["L"]
        if has_h:
            dside += med["T"] - med["B"]
        dside /= (2 if (has_v and has_h) else 1)
        cr += dcr; cs += dcs
        side = max(30.0, side + dside)
        if abs(dcr) < 0.04 and abs(dcs) < 0.04 and abs(dside) < 0.04:
            break

    # 角度不在暗度剖面上精修：低 DPI 下剖面噪声会系统带偏角度。
    # θ 直接采用霍夫矩形装配的方向（其精度由长线决定，优于亚像素剖面斜率）。
    return theta, cr - side / 2, cr + side / 2, cs - side / 2, cs + side / 2


def _edge_pixels(image_bgr: np.ndarray):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = np.zeros_like(gray)
    for lo, hi in ((30, 90), (50, 140), (20, 70)):
        edges = cv2.bitwise_or(edges, cv2.Canny(gray, lo, hi, apertureSize=3))
    ys, xs = np.where(edges > 0)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def _tls_angle_from_pixels(points, ref_dir, band):
    """在 points 中取过原点附近、沿 ref_dir ±band 的窄条，TLS 拟合方向。"""
    if len(points) == 0:
        return None
    n = np.array([-ref_dir[1], ref_dir[0]])
    along = points @ ref_dir
    across = points @ n
    m = np.abs(across) < band
    pts = points[m]
    if len(pts) < 12:
        return None
    mean = pts.mean(axis=0)
    _, sv, vt = np.linalg.svd(pts - mean)
    if sv[0] < 1e-6:
        return None
    d = vt[0]
    return d if np.dot(d, ref_dir) > 0 else -d


def _find_frame(image_bgr: np.ndarray, mask: np.ndarray):
    segs = _hough_segments(image_bgr)
    rect = _find_rectangle(segs)
    if rect is None:
        return None, None
    d0 = rect["d0"]
    side_est = rect["side"]
    dark = _darkness(_luminance(image_bgr))
    strip = max(1.6, side_est * 0.022)
    theta, r1, r2, s1, s2 = _refine_rect_on_darkness(
        dark, d0, side_est, rect["r1"], rect["r2"], rect["s1"], rect["s2"], strip)
    r1, r2 = sorted((r1, r2))
    s1, s2 = sorted((s1, s2))

    dv = np.array([np.cos(theta), np.sin(theta)])
    nv = np.array([-dv[1], dv[0]])
    # 横边方向 = nv，法向 nh = perp(nv) = -dv；上面精修的 s 是沿 -dv 的有符号距离
    nh = -dv

    def intersect_vh(rho_v, rho_h):
        # nv·x = rho_v（竖边），nh·x = rho_h（横边）
        return np.linalg.solve(np.array([nv, nh]), np.array([rho_v, rho_h]))

    q = np.array([
        intersect_vh(r1, s1),
        intersect_vh(r2, s1),
        intersect_vh(r2, s2),
        intersect_vh(r1, s2),
    ])
    sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
    return q, float(sides.mean())


def _components_near(mask: np.ndarray, center: np.ndarray, radius: float,
                     soft_mass: np.ndarray | None = None):
    """在 center 半径内列出连通域；给定时用软质量做亚像素质心。"""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    H, W = mask.shape
    ys_all, xs_all = np.mgrid[0:H, 0:W]
    comps = []
    for i in range(1, n):
        c = cents[i]
        if np.hypot(c[0] - center[0], c[1] - center[1]) > radius:
            continue
        if soft_mass is not None:
            m = (labels == i) & (soft_mass > 0)
            mass = float(soft_mass[m].sum())
            if mass > 1e-6:
                c = np.array([
                    float((soft_mass[m] * xs_all[m]).sum() / mass),
                    float((soft_mass[m] * ys_all[m]).sum() / mass),
                ])
        comps.append((i, c, float(stats[i, cv2.CC_STAT_AREA])))
    return comps, (labels, stats, cents)


def calibrate(image_bgr: np.ndarray):
    """成功返回 (Calibration, meta, None)；失败返回 (None, None, error)。"""
    from .geometry import Calibration

    mask = _dark_mask(image_bgr)
    frame, side_px = _find_frame(image_bgr, mask)
    if frame is None:
        return None, None, CalibError("frame_not_found", "未找到合格的 K 校准方框")

    diag_px = float(np.linalg.norm(frame[0] - frame[2]))
    center = frame.mean(axis=0)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    # 圆点/pip 软暗度：覆盖旋转/缩放后灰度抬升的墨像素
    dark_mass = _dot_darkness(lab[:, :, 0])

    # 小要素提取：框内（含 pip 所在的角部）暗像素里，沿检测到的四条墨线中心线
    # 各涂掉一条带子（去掉框线），这样 pip/圆点与框线断开。
    s_est = side_px / FRAME_SIDE_MM  # px/mm
    roi = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(roi, frame.astype(np.int32), 255)
    small_mask = (lab[:, :, 0] < 150).astype(np.uint8) * 255
    small_mask = cv2.bitwise_and(small_mask, roi)
    # frame 顶点是墨线中心线；pip 外边距墨线中心 0.25mm。只擦中心线附近窄带
    # （半宽 0.15mm+约1px），不要擦到 pip（低 DPI 下尤其重要）。
    s_est = side_px / FRAME_SIDE_MM  # px/mm
    roi = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(roi, frame.astype(np.int32), 255)
    small_mask = (lab[:, :, 0] < 150).astype(np.uint8) * 255
    small_mask = cv2.bitwise_and(small_mask, roi)
    erase_w = min(0.15 * s_est + 1.2, 0.24 * s_est)
    edge_mask = np.zeros_like(small_mask)
    q = frame.astype(np.float64)
    for a, b in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
        cv2.line(edge_mask, tuple(np.round(a).astype(int)),
                 tuple(np.round(b).astype(int)), 255,
                 thickness=max(2, int(erase_w * 2)))
    small_mask = cv2.bitwise_and(small_mask, cv2.bitwise_not(edge_mask))
    comps, cc = _components_near(small_mask, center, diag_px / 2 * 1.02, dark_mass)
    labels, stats, cents = cc
    pip_area = (PIP_SIDE_MM * s_est) ** 2
    dot_area = np.pi * (DOT_DIAMETER_MM / 2 * s_est) ** 2

    # 面积与位置联合分类：pip 在框角（|r|≈5.44mm）、方块；K 圆点在 x 轴、小圆盘
    small = []
    for i, c, area in comps:
        if area > pip_area * 2.5:  # 方框本体/大块
            continue
        d_center_mm = float(np.linalg.norm(c - center) / s_est)
        small.append((i, c, area, d_center_mm))

    # pip：靠近框角（pips 距中心 3.85·√2≈5.44mm；检测误差下 4.3–6.3），
    # 面积在方块量级。下限放宽以容忍被框边裁掉一部分。
    pip_ids = {i for i, c, a, d in small
               if 4.3 < d < 6.3 and pip_area * 0.12 <= a <= pip_area * 2.5}

    # 框方向：q 顶点顺序 (r1,s1)->(r2,s1)->(r2,s2)->(r1,s2)；
    # 边 q1-q0 沿 nv（竖边），边 q3-q0 沿 nh=水平（即印刷 x 轴方向）。
    # 这里只需选与 K 圆点共线的那条（水平边方向），用几何试选由后续假设完成；
    # 先取两条邻边，K 点在框中线附近，用"哪条轴更靠近小圆圆心"判定。
    q = frame.astype(np.float64)
    e_a = q[1] - q[0]
    e_b = q[3] - q[0]
    e_a /= np.linalg.norm(e_a)
    e_b /= np.linalg.norm(e_b)

    def on_x_axis(c, axis, max_off_mm=0.55):
        perp = np.array([-axis[1], axis[0]])
        return abs(float(np.dot(c - center, perp))) / s_est < max_off_mm

    # K 圆点必须真是黑墨：暗度高且近无彩色（a、b 通道靠近灰轴）
    lab_a = lab[:, :, 1]
    lab_b = lab[:, :, 2]

    def component_is_black(comp_id: int) -> bool:
        m = labels == comp_id
        mean_a = float(lab_a[m].mean())
        mean_b = float(lab_b[m].mean())
        chroma = math.hypot(mean_a - 128.0, mean_b - 128.0)
        return chroma < 12.0

    # 先收集圆点大小的小连通域候选，再决定哪条边是 x 轴
    dot_like = [(i, c, a, d) for i, c, a, d in small
                if 1.0 < d < 4.3 and dot_area * 0.35 <= a <= dot_area * 3.2
                and component_is_black(i)]

    # K 点标称在印刷 +x 轴上（1.75、3.5mm）；方向正负由后面的假设枚举处理。
    def kdots_on(axis):
        perp = np.array([-axis[1], axis[0]])
        scored = []
        for i, c, a, d in dot_like:
            if i in pip_ids:
                continue
            rel = c - center
            if abs(float(np.dot(rel, perp))) / s_est > 0.55:
                continue
            t = abs(float(np.dot(rel, axis))) / s_est
            rad_err = min(abs(t - 1.75), abs(t - 3.5))
            if rad_err < 0.7:
                scored.append((rad_err, i, t))
        scored.sort()
        # 半径 1.75 与 3.5 各取一个最佳
        chosen = []
        for target in (1.75, 3.5):
            best_i, best_e = None, None
            for _, i, t in scored:
                if i in chosen:
                    continue
                e = abs(t - target)
                if e < 0.7 and (best_e is None or e < best_e):
                    best_e, best_i = e, i
            if best_i is not None:
                chosen.append(best_i)
        return chosen

    axis_b_count = len(kdots_on(e_b))
    axis_a_count = len(kdots_on(e_a))
    if axis_b_count >= axis_a_count and axis_b_count >= 1:
        axis_x = e_b
        kdot_ids = kdots_on(e_b)
    elif axis_a_count >= 1:
        axis_x = e_a
        kdot_ids = kdots_on(e_a)
    else:
        kdot_ids = []

    def refine_dot_centroid(comp_id: int) -> np.ndarray:
        """在连通域邻域用软暗度做亚像素质心（含抗锯齿半暗像素）。"""
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        pad = 3
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(labels.shape[1], x + w + pad), min(labels.shape[0], y + h + pad)
        comp_m = (labels[y0:y1, x0:x1] == comp_id)
        # 扩张 1px 纳入抗锯齿边
        comp_m = cv2.dilate(comp_m.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
        m = dark_mass[y0:y1, x0:x1] * comp_m
        if m.sum() < 1e-6:
            return cents[comp_id]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        return np.array([float((m * xx).sum() / m.sum()),
                         float((m * yy).sum() / m.sum())])

    pips = [refine_dot_centroid(i) for i in pip_ids]
    kdots = [refine_dot_centroid(i) for i in kdot_ids]
    pip_set = set()

    method = ["K-frame"]
    if len(pips) >= 2 and len(kdots) >= 1:
        method.append("2-pips")
    elif len(pips) < 2:
        return None, None, CalibError("pips_missing", f"K 角点不完整（仅 {len(pips)} 个），无法定方向")
    method.append(f"{len(kdots)}-Kdots")

    # ---- 定向：离散假设穷举，不写脆弱启发式 ----
    pips_arr = np.asarray(pips, dtype=np.float64)
    # 印刷 x 轴在像素中的朝向 u0：
    #   两个 K 点时取点对连线；单个 K 点时取 框中心->该点（身份用假设枚举消歧）。
    if not kdots:
        return None, None, CalibError("kdots_missing", "K 控制点全部缺失，无法消除方向歧义")
    k_identity_options = ("K", "K1")
    if len(kdots) >= 2:
        kd = np.asarray(kdots, dtype=np.float64)
        expected_pair_len = float(
            abs(DOT_NOMINAL_MM["K1"][0] - DOT_NOMINAL_MM["K"][0]) * s_est)
        best_pair, best_err = None, None
        for i in range(len(kd)):
            for j in range(i + 1, len(kd)):
                e = abs(float(np.linalg.norm(kd[j] - kd[i])) - expected_pair_len)
                if best_err is None or e < best_err:
                    best_err, best_pair = e, (kd[i], kd[j])
        u0 = (best_pair[1] - best_pair[0])
        u0 /= np.linalg.norm(u0)
        k_pair = best_pair
        k_identity_options = ("pair",)
    else:
        kp = np.asarray(kdots[0], dtype=np.float64)
        u0 = kp - center
        u0 /= np.linalg.norm(u0)
        k_pair = None

    perp = np.array([-u0[1], u0[0]])
    half = FRAME_SIDE_MM / 2.0
    pip_nom = {n: np.array(v) for n, v in PIP_CENTERS_MM.items()}

    def build_correspondences(a: float, b: float, k_identity: str):
        """a/b ∈ {+1,-1}：印刷 +x/+y 分别映射到 a·u0、b·perp。"""
        u, v = a * u0, b * perp
        rows_px, rows_mm = [], []

        for q in frame:
            d = q - center
            rows_px.append(q)
            rows_mm.append(np.array([
                half if float(np.dot(d, u)) > 0 else -half,
                half if float(np.dot(d, v)) > 0 else -half,
            ]))

        for p in pips_arr:
            su = np.sign(float(np.dot(p - center, u)))
            sv = np.sign(float(np.dot(p - center, v)))
            rows_px.append(p)
            # 只有 (++ )→P0、(--)→P1 合法；落到其它象限时分配给最近标称点（加大残差）
            if su > 0 and sv > 0:
                rows_mm.append(pip_nom["P0"])
            elif su < 0 and sv < 0:
                rows_mm.append(pip_nom["P1"])
            else:
                rows_mm.append(pip_nom["P0"] if su > 0 else pip_nom["P1"])

        if k_identity == "pair":
            # a=+1 时 +u 端为 K1；a=-1 时反之
            order = k_pair if a > 0 else (k_pair[1], k_pair[0])
            for pt, name in zip(order, ("K", "K1")):
                rows_px.append(pt)
                rows_mm.append(np.array(DOT_NOMINAL_MM[name]))
        else:
            rows_px.append(kp)
            rows_mm.append(np.array(DOT_NOMINAL_MM[k_identity]))
        return np.asarray(rows_px, float), np.asarray(rows_mm, float)

    def fit_hypothesis(a: float, b: float, k_identity: str):
        px, mm = build_correspondences(a, b, k_identity)
        M, _, _, _ = np.linalg.lstsq(
            np.hstack([mm, np.ones((len(mm), 1))]), px, rcond=None)
        M = M.T
        pred = M[:, :2] @ mm.T
        pred = pred.T + M[:, 2]
        rms = float(np.sqrt(((pred - px) ** 2).sum(axis=1).mean()))
        return M, rms

    hypotheses = [(a, b, kid)
                  for a in (1.0, -1.0) for b in (1.0, -1.0)
                  for kid in k_identity_options]
    ranked = sorted(
        ((*fit_hypothesis(a, b, kid), a, b, kid) for a, b, kid in hypotheses),
        key=lambda t: t[1])
    M, residual_px, a_best, b_best, kid_best = ranked[0]
    second_rms = ranked[1][1]
    u_best, v_best = a_best * u0, b_best * perp
    res_gap = second_rms - residual_px

    linear = M[:, :2]
    sx = float(np.linalg.norm(linear[:, 0]))
    sy = float(np.linalg.norm(linear[:, 1]))
    scale = (sx + sy) / 2.0
    anisotropy = abs(sx - sy) / scale
    # 印刷 y-up -> 像素 y-down 的仿射 det 为负；去掉翻转后取旋转角
    rot = linear @ np.diag([1.0, -1.0]) / scale
    angle_deg = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))

    residual_mm = residual_px / max(scale, 1e-9)
    n_corr = 4 + len(pips_arr) + (2 if kid_best == "pair" else 1)
    angle_unc = max(0.05, float(np.degrees(np.arctan2(residual_px, side_px))))
    scale_unc = max(0.1, residual_px / side_px * 100.0 * 2.0)
    # 正确/次优假设的残差差不够大 → 定向本身可疑
    if res_gap < max(1.5, 3.0 * residual_px):
        angle_unc = max(angle_unc, 1.5)

    # pip 名称按选定假设回填（叠层/复核用）
    pip_points: dict[str, np.ndarray] = {}
    for p in pips_arr:
        su = np.sign(float(np.dot(p - center, u_best)))
        sv = np.sign(float(np.dot(p - center, v_best)))
        pip_points["P0" if su > 0 and sv > 0 else "P1"] = p

    calib = Calibration(
        M=M.astype(np.float64),
        M_inv=cv2.invertAffineTransform(M),
        angle_deg=angle_deg,
        scale_px_per_mm=scale,
        dpi_effective=scale * 25.4,
        translation_px=(float(M[0, 2]), float(M[1, 2])),
        residual_mm=residual_mm,
        angle_uncertainty_deg=angle_unc,
        scale_uncertainty_percent=scale_unc,
        method="+".join(method),
        frame_corners_px=frame,
        pip_points_px=pip_points,
    )
    meta = {
        "anisotropy_percent": anisotropy * 100.0,
        "n_correspondences": n_corr,
        "n_kdots": len(kdots),
        "orientation_gap_px": res_gap,
    }
    return calib, meta, None
