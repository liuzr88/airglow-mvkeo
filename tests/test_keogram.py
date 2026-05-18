from datetime import datetime, timezone, timedelta
import numpy as np
from airglow_mvkeo.keogram import (
    extract_slices, wrap_time_hours, insert_gaps, render_keogram,
    scaled_keogram_width,
)

def test_extract_slices_shapes():
    frames = np.arange(100*100*5, dtype=np.float32).reshape(100, 100, 5)
    we, sn = extract_slices(frames, x0=50, y0=50)
    assert we.shape == (100, 5)
    assert sn.shape == (100, 5)
    assert np.array_equal(we, frames[50, :, :])
    assert np.array_equal(sn, frames[:, 50, :])

def test_wrap_time_hours_negative_branch():
    base = datetime(2018, 1, 1, tzinfo=timezone.utc)
    times = np.array([base.replace(hour=22), base.replace(hour=3)])
    h = wrap_time_hours(times)
    assert abs(h[0] - (-2)) < 1e-6
    assert abs(h[1] - 3) < 1e-6

def test_insert_gaps_adds_nan_columns():
    we = np.ones((10, 5), dtype=np.float32)
    times = np.array([0.0, 0.05, 0.10, 1.0, 1.05])  # big gap between idx 2 and 3
    new_we, new_t = insert_gaps(we, times, gap_factor=2.05)
    assert new_we.shape[1] == 7
    assert np.isnan(new_we[:, 3]).all() or np.isnan(new_we[:, 4]).all()
    assert len(new_t) == 7

def test_scaled_keogram_width_maps_duration_to_fixed_full_night():
    assert scaled_keogram_width(np.array([0.0, 10.0]), 10, 1136, 480) == 1136
    assert scaled_keogram_width(np.array([2.0, 7.0]), 10, 1136, 480) == 568
    assert scaled_keogram_width(np.array([2.0, 3.0]), 10, 1136, 480) == 480

def test_render_keogram_writes_jpg(tmp_path):
    we = np.random.default_rng(0).normal(2000, 200, (100, 30)).astype(np.float32)
    sn = np.random.default_rng(1).normal(2000, 200, (100, 30)).astype(np.float32)
    times = np.linspace(-2, 8, 30)
    out = tmp_path / "k.jpg"
    render_keogram(
        we=we, sn=sn, times=times,
        x0=50, y0=50, R=48, altitude_km=87, fov_deg=180, image_size=100,
        vmin=1500, vmax=2500,
        title="OH Airglow @ALO test",
        date_label="2018-01-01",
        out_path=out, dpi=100,
        colormap="gray", colormap_crop=(10, 128),
        distance_ticks_km=(-200, -100, 0, 100, 200),
        distance_tick_labels=("-200","-100","0","+100","200"),
        full_night_hours=10,
        full_night_width_px=640,
        min_width_px=240,
    )
    assert out.exists() and out.stat().st_size > 1000
