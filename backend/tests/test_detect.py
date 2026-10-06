"""Vision-pipeline tests against the known synthetic ground truth.

These are the calibration checks required by the task: known translations,
scan rotations and resolution changes must be recovered within tolerance,
and a purely skewed scan must NOT be reported as plate misregistration.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import markspec as spec
from app import synth
from app.detect import analyze_image, analyze_file

DATA = Path(__file__).resolve().parent.parent / "data"

TOL_UM_CLEAN = 30.0     # synthetic, no scan degradation
TOL_UM_SCAN = 45.0      # after blur/noise/resampling
TOL_PPI = 4.0
TOL_ROT = 0.1
TOL_ROT_MC = 0.25       # blur + low-angle edge estimation on small dots


def _truth(name):
    return json.loads((DATA / f"{name}.png.json").read_text())


def _plate_cover(result, plate, axis, truth):
    r = result.plates[plate]
    lo = getattr(r, f"{axis}_lo")
    hi = getattr(r, f"{axis}_hi")
    assert lo is not None and hi is not None, f"{plate}.{axis} 缺范围"
    assert lo <= truth <= hi, f"{plate}.{axis} 真值 {truth} 不在 [{lo},{hi}]"


def test_reference_has_zero_plate_offsets():
    r = analyze_file(DATA / "prt1_reference_300ppi.png", 300)
    assert r.detected and r.status == "ok"
    for p in spec.PLATES:
        assert abs(r.plates[p].dx_um) < TOL_UM_CLEAN
        assert abs(r.plates[p].dy_um) < TOL_UM_CLEAN
        assert r.plates[p].status == "ok"


def test_known_translations_recovered():
    r = analyze_file(DATA / "prt1_shifted_300ppi.png", 300)
    truth = _truth("prt1_shifted_300ppi")["plate_shifts_um"]
    for p, t in truth.items():
        assert abs(r.plates[p].dx_um - t["dx"]) < TOL_UM_CLEAN
        assert abs(r.plates[p].dy_um - t["dy"]) < TOL_UM_CLEAN
        _plate_cover(r, p, "x", t["dx"])
        _plate_cover(r, p, "y", t["dy"])
    # Black stays the reference plate.
    assert abs(r.plates["black"].dx_um) < 1e-6
    assert abs(r.plates["black"].dy_um) < 1e-6


def test_rotation_and_resolution_recovered_from_scan():
    r = analyze_file(DATA / "prt1_scan_rot1p7_240ppi.png", 240)
    truth = _truth("prt1_scan_rot1p7_240ppi")
    assert abs(r.frame_rotation_deg - truth["rotation_deg"]) < TOL_ROT
    assert abs(r.measured_ppi - truth["scan_ppi"]) < TOL_PPI
    for p, t in truth["plate_shifts_um"].items():
        assert abs(r.plates[p].dx_um - t["dx"]) < TOL_UM_SCAN
        assert abs(r.plates[p].dy_um - t["dy"]) < TOL_UM_SCAN
        _plate_cover(r, p, "x", t["dx"])
        _plate_cover(r, p, "y", t["dy"])


def test_paper_skew_does_not_become_plate_error():
    """The key separation requirement: 2.3° rotation + 260 PPI, zero plate
    shifts — every plate must still read ~0 inside its confidence range."""
    r = analyze_file(DATA / "prt1_scan_paperonly.png", 260)
    truth = _truth("prt1_scan_paperonly")
    assert abs(r.frame_rotation_deg - truth["rotation_deg"]) < TOL_ROT
    assert abs(r.measured_ppi - 260) < TOL_PPI
    for p in spec.PLATES:
        assert abs(r.plates[p].dx_um) < TOL_UM_SCAN
        assert abs(r.plates[p].dy_um) < TOL_UM_SCAN
        _plate_cover(r, p, "x", 0)
        _plate_cover(r, p, "y", 0)


def test_defective_scan_returns_candidates_with_ranges():
    r = analyze_file(DATA / "prt1_scan_defective.png", 306)
    # Black has damaged rings -> at least one plate is candidate, never an
    # unjustified precise value.
    assert r.status in ("candidates", "ok")
    assert any(pr.status == "candidate" for pr in r.plates.values())
    truth = _truth("prt1_scan_defective")["plate_shifts_um"]
    for p, t in truth.items():
        pr = r.plates[p]
        if pr.x_lo is not None:
            assert pr.x_lo <= t["dx"] <= pr.x_hi
            assert pr.x_hi - pr.x_lo >= 20  # a real range, not a point
        if pr.y_lo is not None:
            assert pr.y_lo <= t["dy"] <= pr.y_hi
            assert pr.y_hi - pr.y_lo >= 20


def test_missing_plate_is_null_not_zero():
    img = synth.render_chart(
        300.0, [synth.PlateShift("cyan", 90, 0)])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([140, 60, 60], np.uint8),
                    np.array([178, 255, 255], np.uint8)) > 0
    img[m] = 255
    r = analyze_image(img, 300)
    assert r.plates["magenta"].status == "missing"
    assert r.plates["magenta"].dx_um is None
    assert r.plates["magenta"].x_lo is None
    assert r.plates["magenta"].confidence == 0.0
    # Other plates still read.
    assert abs(r.plates["cyan"].dx_um - 90) < TOL_UM_CLEAN


def test_excessive_rotation_is_rejected_not_silently_fixed():
    sp = synth.ScanParams(rotation_deg=12.0, scan_ppi=300)
    img = synth.simulate_scan(sp, seed=3)
    r = analyze_image(img, 300)
    assert not r.detected
    assert r.status == "rotation_out_of_range"
    assert all(pr.status == "missing" for pr in r.plates.values())


def test_random_ground_truth_parameter_grid():
    """Monte-Carlo style check across translations / rotations / PPIs."""
    rng = np.random.default_rng(2026)
    for i in range(6):
        shifts = [
            synth.PlateShift("cyan", float(rng.uniform(-300, 300)),
                             float(rng.uniform(-200, 200))),
            synth.PlateShift("magenta", float(rng.uniform(-300, 300)),
                             float(rng.uniform(-200, 200))),
            synth.PlateShift("yellow", float(rng.uniform(-300, 300)),
                             float(rng.uniform(-200, 200)))]
        rot = float(rng.uniform(-3, 3))
        ppi = float(rng.uniform(230, 320))
        sp = synth.ScanParams(rotation_deg=-rot, scan_ppi=ppi,
                              noise_sigma=4, blur_sigma=0.9,
                              dirt_spots=6, plate_shifts=shifts)
        img = synth.simulate_scan(sp, seed=100 + i)
        r = analyze_image(img, ppi)
        assert r.detected, f"case {i} 未检出"
        assert abs(r.frame_rotation_deg - rot) < TOL_ROT_MC
        assert abs(r.measured_ppi - ppi) < TOL_PPI * 1.5
        for s in shifts:
            assert abs(r.plates[s.plate].dx_um - s.dx_um) < 60
            assert abs(r.plates[s.plate].dy_um - s.dy_um) < 60


def test_provenance_contains_version_and_frame_sources():
    r = analyze_file(DATA / "prt1_scan_rot1p7_240ppi.png", 240)
    steps = {s["step"]: s for s in r.provenance["steps"]}
    assert "paper_frame" in steps
    assert "ruler" in steps
    assert steps["ruler"]["found"] is True
    assert r.provenance["detector_version"].startswith("prt1-detector")
