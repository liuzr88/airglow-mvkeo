import numpy as np
from airglow_mvkeo.intensity import (
    filter_overexposed, count_low_intensity_frames,
    color_range_zenith, fps_for_cadence,
)

def test_filter_drops_high_mean_frames():
    frames = np.zeros((10, 10, 5), dtype=np.float32)
    frames[..., 0] = 5e3
    frames[..., 1] = 1.5e4
    frames[..., 2] = 2e3
    frames[..., 3] = 9e3
    frames[..., 4] = 5e4
    keep_idx = filter_overexposed(frames, threshold=1e4)
    assert keep_idx.tolist() == [0, 2, 3]

def test_count_low_intensity():
    frames = np.zeros((4, 4, 4), dtype=np.float32)
    frames[..., 0] = 1e3
    frames[..., 1] = 6e3
    frames[..., 2] = 4e3
    frames[..., 3] = 8e3
    assert count_low_intensity_frames(frames, threshold=5e3) == 2

def test_color_range_uses_zenith_strip_and_mask():
    rng = np.random.default_rng(0)
    frames = rng.integers(800, 4000, size=(100, 100, 10)).astype(np.float32)
    frames[49:52, :, :] = np.where(rng.random((3, 100, 10)) < 0.05, 1e5, frames[49:52, :, :])
    vmin, vmax = color_range_zenith(frames, x0=50, y0=50, low_threshold=5e3,
                                    percentiles=(10, 80))
    assert 800 < vmin < vmax < 5e3

def test_fps_high_cadence():
    times = np.arange(0, 600, 30).astype(float)
    assert fps_for_cadence(times, normal=30, low=15, threshold_minutes=2.0) == 30

def test_fps_low_cadence():
    times = np.arange(0, 1800, 180).astype(float)
    assert fps_for_cadence(times, normal=30, low=15, threshold_minutes=2.0) == 15
