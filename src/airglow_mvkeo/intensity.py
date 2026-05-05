"""Intensity-related computations: over-exposure filter, color range, fps choice."""
from __future__ import annotations
import numpy as np

def filter_overexposed(frames: np.ndarray, threshold: float) -> np.ndarray:
    """Return indices of frames whose full-image mean is < threshold.
    `frames` shape: (H, W, N)."""
    means = frames.mean(axis=(0, 1))
    return np.flatnonzero(means < threshold)

def count_low_intensity_frames(frames: np.ndarray, threshold: float) -> int:
    """Number of frames with full-image mean <= threshold.
    Used to gate the 'mostly overexposed' skip (matches MATLAB ReadNcOrig.m:169)."""
    means = frames.mean(axis=(0, 1))
    return int((means <= threshold).sum())

def color_range_zenith(frames: np.ndarray, x0: int, y0: int,
                       low_threshold: float,
                       percentiles: tuple[float, float]) -> tuple[float, float]:
    """Compute (vmin, vmax) from the concatenation of WE-strip (y0+-1, all x) and
    SN-strip (all y, x0+-1) pixels across all frames, masked to <= low_threshold.
    Matches MATLAB ReadNcOrig.m:155-177."""
    we = frames[y0-1:y0+2, :, :].ravel()
    sn = frames[:, x0-1:x0+2, :].ravel()
    pool = np.concatenate([we, sn])
    pool = pool[pool <= low_threshold]
    if pool.size == 0:
        raise ValueError("no usable pixels for color range")
    lo, hi = percentiles
    return float(np.percentile(pool, lo)), float(np.percentile(pool, hi))

def fps_for_cadence(time_seconds: np.ndarray, normal: int, low: int,
                    threshold_minutes: float) -> int:
    """Pick fps based on mean cadence. Matches MATLAB ReadNcOrig.m:190."""
    if len(time_seconds) < 2:
        return normal
    mean_dt_min = np.mean(np.diff(time_seconds)) / 60.0
    return low if mean_dt_min > threshold_minutes else normal
