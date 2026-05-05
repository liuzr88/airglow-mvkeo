"""Geometry: calibration lookup and pixel ↔ horizontal-distance mapping."""
from __future__ import annotations
from datetime import date
import numpy as np
from .config import CalibrationEntry


def lookup_calibration(table: list[CalibrationEntry], d: date) -> CalibrationEntry:
    """Return the latest entry whose `since` <= d. If d is before the earliest
    entry, return the earliest (degrades gracefully on undated test data)."""
    if not table:
        raise ValueError("calibration table is empty")
    chosen = table[0]
    for entry in table:
        if entry.since <= d:
            chosen = entry
        else:
            break
    return chosen


def pixel_to_km(size_px: int, x0: int, y0: int, R: int,
                altitude_km: float, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Map pixel indices to horizontal distance (km) along the WE (Xl) and SN (Yl)
    lines through zenith.

    A pixel at radial distance r from center maps to off-zenith angle
    theta = (|r| / R) * (fov_deg / 2). Horizontal distance d = altitude * tan(theta).
    Beyond ~horizon (theta >= 89.5°) we return NaN.
    """
    half_fov = fov_deg / 2.0
    pix = np.arange(size_px)
    rx = (pix - x0).astype(float)
    ry = (pix - y0).astype(float)

    def _to_km(r):
        theta_deg = np.abs(r) / R * half_fov
        theta_deg = np.where(theta_deg >= 89.5, np.nan, theta_deg)
        d = altitude_km * np.tan(np.deg2rad(theta_deg))
        return np.sign(r) * d

    return _to_km(rx), _to_km(ry)
