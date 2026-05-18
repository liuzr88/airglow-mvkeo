"""Frame differencing helpers."""
from __future__ import annotations
import numpy as np


def previous_frame_difference(frames: np.ndarray) -> np.ndarray:
    """Return MATLAB ``*_TD`` frames: current image minus previous image.

    ``CreateMovNC(..., DI=1)`` initializes the first image as history and does
    not write it, so the output has one fewer frame than the raw movie.
    """
    if frames.ndim != 3:
        raise ValueError("frames must be 3D (H, W, N)")
    if frames.shape[2] < 2:
        return np.empty((*frames.shape[:2], 0), dtype=np.float32)
    return np.diff(frames.astype(np.float32), axis=2)

def running_mean_subtract(frames: np.ndarray, time_seconds: np.ndarray,
                          window_minutes: float) -> np.ndarray:
    """Subtract a centered running mean (window_minutes wide) from each frame.

    `frames`: (H, W, N) float
    `time_seconds`: (N,) float, monotonic seconds (UT)
    Returns array of same shape, float32. Edge frames use a partial window.
    """
    if frames.ndim != 3:
        raise ValueError("frames must be 3D (H, W, N)")
    n = frames.shape[2]
    if n == 0:
        return frames.astype(np.float32, copy=True)

    half = window_minutes * 60.0 / 2.0
    out = np.empty_like(frames, dtype=np.float32)
    lo = 0
    hi = 0
    for i in range(n):
        t = time_seconds[i]
        while lo < n and time_seconds[lo] < t - half:
            lo += 1
        while hi < n and time_seconds[hi] <= t + half:
            hi += 1
        window_mean = frames[:, :, lo:hi].mean(axis=2)
        out[:, :, i] = frames[:, :, i].astype(np.float32) - window_mean
    return out
