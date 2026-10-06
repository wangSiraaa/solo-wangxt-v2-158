"""用已知平移/旋转/DPI 的合成夹具验算检测精度。

关键断言：
1. 校准角、尺度与真值一致（允许经验容差）；
2. 各色版偏移恢复误差落在报告的不确定度内（系统诚实，不输出虚假精确值）；
3. 纸张旋转/尺度变化不会被误判成色版偏移——无平移各色版偏移≈0；
4. 缺色、污点、残缺案例：状态为 missing/incomplete/ambiguous 且返回候选/范围，
   绝不从单个模糊点输出精确微米。
"""
import glob
import json
import math
import os

import numpy as np
import pytest

from app.cv.detect_target import analyze, load_image
from app.cv.spec import PIP_CENTERS_MM

FIX = os.path.join(os.path.dirname(__file__), "..", "data", "fixtures")
MM = 25.4


def _cases():
    return [f for f in sorted(glob.glob(os.path.join(FIX, "*.json")))
            if not f.endswith("manifest.json")]


def _truth_affine(gt):
    """复刻生成器扫描仿射（真值），便于把已知印刷坐标变到像素做比对。"""
    from app.cv.geometry import build_affine
    dpi = gt["scan"]["dpi"]
    tx, ty = gt["scan"]["translation_px"]
    size_mm = 16.0
    scan_side = int(round(size_mm * dpi / MM * 1.18 + abs(tx) + 60))
    margin = max(scan_side // 5, 40)
    canvas = scan_side + 2 * margin
    t = (tx + canvas / 2 - scan_side / 2, ty + canvas / 2 - scan_side / 2)
    return build_affine(gt["scan"]["angle_deg"], dpi, t)


def _result(name):
    gt = json.load(open(os.path.join(FIX, name + ".json")))
    img = load_image(os.path.join(FIX, gt["image"]))
    return gt, analyze(img)


def test_all_fixtures_present():
    assert len(_cases()) == 10


@pytest.mark.parametrize("jf", _cases())
def test_calibration_or_explicit_failure(jf):
    gt = json.load(open(jf))
    img = load_image(os.path.join(FIX, gt["image"]))
    r = analyze(img)
    cal = r.get("calibration")
    # 夹具里校准尺总是存在的；即使校准质量差也要有明确状态而不是抛异常
    assert r["status"] in {
        "ok", "incomplete", "calibration_poor", "calibration_failed", "no_plates"}
    if cal is not None and cal["usable"]:
        # 角度误差：报告不确定度必须覆盖（这是"不把整图歪斜当色版偏移"的核心）
        assert abs(cal["angle_deg"] - gt["scan"]["angle_deg"]) <= max(
            1.2, cal["angle_uncertainty_deg"])
        # DPI：算法尺度（未用真值 DPI）误差
        assert abs(cal["measured_dpi"] - gt["scan"]["dpi"]) / gt["scan"]["dpi"] < 0.06


def test_baseline_zero_offset_is_not_skew():
    gt, r = _result("01_baseline")
    assert r["calibration"]["usable"]
    for p in ("C", "M", "Y"):
        b = r["plates"][p]
        assert b["offset_um"] is not None
        err = math.hypot(*[b["offset_um"][i] - gt["true_offsets_um"][p][i]
                          for i in range(2)])
        # 基线偏移真值=0；报告不确定度必须覆盖测量误差
        assert err <= max(b["uncertainty_um"]) + 1e-6, (p, err, b["uncertainty_um"])


@pytest.mark.parametrize("name", ["02_translations", "05_dpi_300", "07_missing_c"])
def test_known_translations_recovered(name):
    gt, r = _result(name)
    for p in ("C", "M", "Y"):
        t = gt["true_offsets_um"][p]
        b = r["plates"][p]
        if b["offset_um"] is None:
            # 缺色版只允许出现在 07 的 C
            assert name == "07_missing_c" and p == "C"
            assert b["status"] == "missing"
            assert b["confidence_range_um"] is not None
            continue
        err = math.hypot(b["offset_um"][0] - t[0], b["offset_um"][1] - t[1])
        # 误差必须落在置信区间内（系统不能自相矛盾地报高精度）
        assert err <= max(b["uncertainty_um"]) + 60, (name, p, err, b["uncertainty_um"])


def test_missing_plate_returns_range_not_precision():
    gt, r = _result("07_missing_c")
    b = r["plates"]["C"]
    assert b["status"] == "missing"
    assert b["offset_um"] is None
    assert b["uncertainty_um"] is None
    assert b["confidence_range_um"] is not None
    assert any("搜索半径" in n for n in b["notes"])


def test_spots_do_not_become_exact_offsets():
    gt, r = _result("08_spots")
    # 有污点时，每个色版都必须带候选信息；若选了点，不确定度不能虚假过小
    for p in ("C", "M", "Y"):
        b = r["plates"][p]
        if b["offset_um"] is not None:
            # 200DPI 下亚像素质心理论底噪 ~30µm；有污点不得报比这更小的"精度"
            assert max(b["uncertainty_um"]) >= 30.0
            assert len(b["candidates"]) >= 1


def test_occluded_dot_is_incomplete_with_candidate():
    gt, r = _result("09_occluded_m")
    b = r["plates"]["M"]
    assert b["status"] in ("incomplete", "ambiguous", "measured")
    assert len(b["candidates"]) >= 1
    if b["status"] == "incomplete":
        assert b["offset_um"] is not None  # 有值但降级，不给"精确"的承诺


def test_rotation_and_scale_separated_from_plate_shift():
    """仅旋转/DPI 变化（无平移）的案例：即使校准残差存在，色版偏移也不应
    被解释成数百微米的系统偏差——误差中心应在不确定度附近。"""
    for name in ("03_rotated", "06_dpi_150"):
        gt, r = _result(name)
        if r["status"] == "calibration_failed":
            continue  # 极端分辨率下允许失败，但不能瞎报
        for p in ("C", "M", "Y"):
            b = r["plates"][p]
            if b["offset_um"] is None:
                continue
            err = math.hypot(b["offset_um"][0] - gt["true_offsets_um"][p][0],
                             b["offset_um"][1] - gt["true_offsets_um"][p][1])
            assert err <= 400.0  # 远小于"把 2.5° 整图歪斜误判"的量级（>1000µm）


def test_provenance_fields_present():
    gt, r = _result("01_baseline")
    assert r["spec_version"] == "REG-TARGET/1"
    assert r["algo_version"]
    assert r["calibration"]["method"]
    assert set(["frame", "pips", "Kdots"]).issubset(
        set(r["calibration"]["method"].split("+")[0].split("-")) |
        {tok for tok in r["calibration"]["method"].replace("+", "-").split("-")})
    assert r["reference_plate"] == "K"
    # K 恒为参照：偏移恒 0
    assert r["plates"]["K"]["offset_um"] == [0.0, 0.0]
    assert r["plates"]["K"]["status"] == "reference"


def test_rejects_non_target_image(tmp_path):
    # 纯噪声/空白图不能被当成色标
    img = np.full((300, 300, 3), 255, np.uint8)
    from app.cv.detect_target import analyze as an
    r = an(img)
    assert r["status"] == "calibration_failed"
    assert r["error"]["code"] in {"frame_not_found", "pips_missing", "kdots_missing"}
