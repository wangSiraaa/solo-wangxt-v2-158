"""离线模型复核（仅建议，不控制设备、不自动回写）。

实际项目里可替换为本地权重推理；这里给出一个保守、可解释的规则版：
- 只在算法给出 measured/incomplete 时产生建议；
- 建议值默认就是算法值，但置信区间放宽 1.5×（离线模型只做复核，不装精确）；
- 缺色/标记不完整时，只给"需要人工判读"的建议，不猜具体微米值。
"""

MODEL_NAME = "offline_review_rules"
MODEL_VERSION = "offline-v1"


def suggest(result: dict) -> list[dict]:
    """返回 [{plate, x, y, range_um, rationale, status_hint}]。"""
    out = []
    if result.get("status") == "calibration_failed":
        return out
    for plate, b in result.get("plates", {}).items():
        if plate == result.get("reference_plate", "K"):
            continue
        if b["status"] in ("missing", "low_confidence"):
            out.append(dict(
                plate=plate, x=None, y=None,
                range_um=b.get("confidence_range_um"),
                rationale="算法未给定点；仅标记搜索范围，需人工判读，不输出微米值",
                status_hint="needs_manual"))
            continue
        if b.get("offset_um") is None:
            continue
        x, y = b["offset_um"]
        u = b.get("uncertainty_um") or [200.0, 200.0]
        out.append(dict(
            plate=plate, x=float(x), y=float(y),
            range_um=round(max(u) * 1.5, 1),
            rationale="离线复核与算法一致；区间放宽 1.5× 供人工参考",
            status_hint="pending_review"))
    return out
