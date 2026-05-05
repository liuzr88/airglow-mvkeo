from datetime import date
import numpy as np
from airglow_mvkeo.config import CalibrationEntry
from airglow_mvkeo.geometry import lookup_calibration, pixel_to_km

CALIB = [
    CalibrationEntry(date(2011, 8, 1),  277, 256, 256, 1.8),
    CalibrationEntry(date(2011, 8, 21), 277, 263, 256, 1.6),
    CalibrationEntry(date(2013, 3, 21), 269, 253, 256, 1.6),
]

def test_calibration_picks_latest_before_date():
    c = lookup_calibration(CALIB, date(2018, 1, 1))
    assert (c.x0, c.y0) == (269, 253)

def test_calibration_boundary_inclusive():
    c = lookup_calibration(CALIB, date(2013, 3, 21))
    assert c.x0 == 269

def test_calibration_before_first_entry_uses_first():
    c = lookup_calibration(CALIB, date(2010, 1, 1))
    assert c.x0 == 277 and c.y0 == 256

def test_pixel_to_km_zero_at_center():
    Xl, Yl = pixel_to_km(size_px=512, x0=256, y0=256, R=256,
                         altitude_km=87, fov_deg=180)
    assert abs(Xl[256]) < 1e-6
    assert abs(Yl[256]) < 1e-6

def test_pixel_to_km_monotonic_outward():
    Xl, _ = pixel_to_km(512, 256, 256, 256, 87, 180)
    assert Xl[260] > Xl[256] > Xl[252]

def test_pixel_to_km_known_value():
    # 0.5*R from center on a 180° FOV → 45° off-zenith → distance = altitude
    Xl, _ = pixel_to_km(512, 256, 256, 256, altitude_km=87, fov_deg=180)
    idx = 256 + 128
    assert abs(Xl[idx] - 87.0) < 1.0
