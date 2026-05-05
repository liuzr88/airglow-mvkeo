import numpy as np
from airglow_mvkeo.difference import running_mean_subtract

def test_running_mean_subtract_constant_is_zero():
    frames = np.full((4, 4, 20), 1000.0, dtype=np.float32)
    times = np.arange(20) * 60.0
    diff = running_mean_subtract(frames, times, window_minutes=5)
    assert np.allclose(diff, 0.0)

def test_running_mean_window_clipped_at_edges():
    rng = np.random.default_rng(0)
    frames = rng.standard_normal((3, 3, 10)).astype(np.float32)
    times = np.arange(10) * 60.0
    diff = running_mean_subtract(frames, times, window_minutes=4)
    assert diff.shape == frames.shape

def test_running_mean_matches_centered_average():
    frames = np.empty((1, 1, 5), dtype=np.float32)
    for i in range(5):
        frames[..., i] = float(i)
    times = np.arange(5) * 60.0
    diff = running_mean_subtract(frames, times, window_minutes=3)
    assert abs(diff[0, 0, 2]) < 1e-6
