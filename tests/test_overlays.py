import numpy as np
from airglow_mvkeo.overlays import (
    apply_colormap, draw_zenith_ring, draw_crosshair, draw_text_corner,
    draw_compass_rose, draw_zenith_angle_labels, compose_movie_frame,
)

def test_apply_colormap_returns_rgb_uint8():
    image = np.linspace(0, 1, 16*16, dtype=np.float32).reshape(16, 16)
    rgb = apply_colormap(image, vmin=0.0, vmax=1.0, name="bone", crop=(4, 124))
    assert rgb.shape == (16, 16, 3) and rgb.dtype == np.uint8

def test_apply_colormap_clips_out_of_range():
    image = np.array([[-1.0, 0.5, 2.0]], dtype=np.float32)
    rgb = apply_colormap(image, vmin=0.0, vmax=1.0, name="bone", crop=(4, 124))
    assert tuple(rgb[0, 0]) != tuple(rgb[0, 2])

def test_draw_zenith_ring_modifies_image_only_on_circle():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_zenith_ring(img.copy(), cx=50, cy=50, radius=40, color=(255,255,255))
    assert tuple(out[50, 50]) == (0, 0, 0)
    assert tuple(out[50, 90]) != (0, 0, 0) or tuple(out[50, 89]) != (0, 0, 0)

def test_draw_crosshair_draws_two_lines():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_crosshair(img.copy(), cx=50, cy=50, half_len=10)
    assert tuple(out[50, 55]) != (0, 0, 0)
    assert tuple(out[55, 50]) != (0, 0, 0)

def test_draw_compass_rose_returns_rgb():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    out = draw_compass_rose(img.copy(), cx=40, cy=160)
    assert out.shape == (200, 200, 3) and out.dtype == np.uint8

def test_draw_zenith_angle_labels_returns_rgb():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    out = draw_zenith_angle_labels(img.copy(), cx=128, cy=128, R=120)
    assert out.shape == (256, 256, 3) and out.dtype == np.uint8

def test_compose_movie_frame_runs_without_error():
    image = np.full((128, 128), 1500.0, dtype=np.float32)
    frame = compose_movie_frame(
        image=image, vmin=1000, vmax=2000,
        x0=64, y0=64, R=60,
        date_str="2018-01-01", time_str="03:14:15 UT",
        site_label="(30.24°S, 70.74°W)",
        site_name="ALO", band="OH",
        markers=[], colormap_name="bone", colormap_crop=(4, 124),
        is_first_or_last=False,
    )
    assert frame.shape == (128, 128, 3) and frame.dtype == np.uint8

def test_compose_movie_frame_first_frame():
    """First/last frame draws extra rings and zenith angle labels."""
    image = np.full((256, 256), 1500.0, dtype=np.float32)
    frame = compose_movie_frame(
        image=image, vmin=1000, vmax=2000,
        x0=128, y0=128, R=120,
        date_str="2023-08-07", time_str="03:14:15 UT",
        site_label="(30.24°S, 70.74°W)",
        site_name="ALO", band="O5",
        markers=[{"name": "GEMINIS", "az_deg": 82, "r_frac": 0.89}],
        colormap_name="bone", colormap_crop=(4, 124),
        is_first_or_last=True,
    )
    assert frame.shape == (256, 256, 3) and frame.dtype == np.uint8

def test_compose_movie_frame_legacy_site_label():
    """Backwards-compat: site_name/band not supplied, falls back to site_label alone."""
    image = np.full((128, 128), 1500.0, dtype=np.float32)
    frame = compose_movie_frame(
        image=image, vmin=1000, vmax=2000,
        x0=64, y0=64, R=60,
        date_str="2018-01-01", time_str="03:14:15 UT",
        site_label="ALO OH (test)",
        markers=[], colormap_name="bone", colormap_crop=(4, 124),
        is_first_or_last=False,
    )
    assert frame.shape == (128, 128, 3) and frame.dtype == np.uint8
