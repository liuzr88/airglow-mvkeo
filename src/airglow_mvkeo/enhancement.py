"""Wave-enhancement utilities for modern web products."""
from __future__ import annotations
import numpy as np


def running_mean_frames(frames: np.ndarray, time_seconds: np.ndarray,
                        window_minutes: float) -> np.ndarray:
    """Return a centered temporal running mean for ``frames``.

    The implementation streams over the time axis and avoids assumptions about
    uniform cadence, which matters for nights with gaps.
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
        out[:, :, i] = frames[:, :, lo:hi].mean(axis=2)
    return out


def relative_perturbation(frames: np.ndarray, time_seconds: np.ndarray,
                          window_minutes: float, scale: float = 100.0) -> np.ndarray:
    """Return percent perturbation from a slowly varying temporal background."""
    bg = running_mean_frames(frames.astype(np.float32), time_seconds, window_minutes)
    floor = max(float(np.nanpercentile(bg, 5)) * 0.25, 1.0)
    return ((frames.astype(np.float32) - bg) / np.maximum(bg, floor) * scale).astype(np.float32)


def robust_limits(data: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    """Robust finite percentile limits with a safe nonzero span."""
    finite = np.asarray(data, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, percentiles)
    if not float(lo) < float(hi):
        mid = float(lo)
        return mid - 0.5, mid + 0.5
    return float(lo), float(hi)


def symmetric_limits(data: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    """Return symmetric limits around zero from an absolute percentile."""
    finite = np.asarray(data, dtype=np.float32)
    finite = np.abs(finite[np.isfinite(finite)])
    if finite.size == 0:
        return -1.0, 1.0
    lim = float(np.percentile(finite, percentile))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    return -lim, lim


def sigma_clipped_symmetric_limits(
    data: np.ndarray,
    sigma: float = 2.0,
    clip_sigma: float = 3.0,
    max_iter: int = 8,
) -> tuple[float, float]:
    """Return symmetric display limits from the majority Gaussian-like signal.

    Bright stars, moonlit edges, and hot pixels occupy the tails of the TD
    distribution. Iterative clipping estimates the standard deviation of the
    central population, then uses ``sigma * std`` as the display range.
    """
    finite = np.asarray(data, dtype=np.float32)
    work = finite[np.isfinite(finite)]
    if work.size == 0:
        return -1.0, 1.0

    sigma = max(float(sigma), 0.1)
    clip_sigma = max(float(clip_sigma), sigma, 0.1)
    for _ in range(max_iter):
        med = float(np.median(work))
        sd = float(np.std(work))
        if not np.isfinite(sd) or sd <= 0:
            break
        keep = np.abs(work - med) <= clip_sigma * sd
        if keep.all() or keep.sum() < max(32, int(0.05 * work.size)):
            break
        work = work[keep]

    med = float(np.median(work))
    sd = float(np.std(work))
    lim = abs(med) + sigma * sd
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    return -float(lim), float(lim)


def signed_brightness_curve(data: np.ndarray, limit: float, strength: float) -> np.ndarray:
    """Curve signed perturbations to lift subtle waves without clipping peaks.

    ``strength=0`` is linear. Larger values act like an asinh/log stretch while
    preserving sign and the same ``[-limit, limit]`` display range.
    """
    if strength <= 0 or limit <= 0:
        return data.astype(np.float32, copy=True)
    x = np.asarray(data, dtype=np.float32) / float(limit)
    curved = np.arcsinh(strength * x) / np.arcsinh(strength)
    return (np.clip(curved, -1.0, 1.0) * float(limit)).astype(np.float32)


def temporal_smooth(data: np.ndarray, width: int) -> np.ndarray:
    """Lightweight box smoothing along the time axis for cleaner web keograms."""
    if width <= 1 or data.shape[1] == 0:
        return data
    width = int(width)
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(data, ((0, 0), (pad_left, pad_right)), mode="edge")
    csum = np.cumsum(padded, axis=1, dtype=np.float64)
    csum = np.concatenate([np.zeros((data.shape[0], 1), dtype=np.float64), csum], axis=1)
    return ((csum[:, width:] - csum[:, :-width]) / width).astype(np.float32)


def spatial_median3(frames: np.ndarray) -> np.ndarray:
    """Apply a 3x3 spatial median per frame to suppress stars/hot pixels."""
    if frames.ndim != 3:
        raise ValueError("frames must be 3D (H, W, N)")
    out = np.empty_like(frames, dtype=np.float32)
    for i in range(frames.shape[2]):
        img = frames[:, :, i]
        p = np.pad(img, ((1, 1), (1, 1)), mode="edge")
        neighbors = np.stack([
            p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:],
            p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:],
            p[2:, :-2], p[2:, 1:-1], p[2:, 2:],
        ], axis=0)
        out[:, :, i] = np.median(neighbors, axis=0)
    return out
