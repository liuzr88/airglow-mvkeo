"""Keogram building & rendering."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from .geometry import pixel_to_km
from .intensity import matlab_get_range

def extract_slices(frames: np.ndarray, x0: int, y0: int) -> tuple[np.ndarray, np.ndarray]:
    """Return centerline keogram samples as simple stacked columns.

    ``frames`` is used as an image cube, so axis 0 is the displayed vertical
    coordinate and axis 1 is the displayed horizontal coordinate. Therefore
    each output column is a direct sample from the image center:
    - W-E varies horizontally at fixed center row: ``frames[y0, :, :]``
    - S-N varies vertically at fixed center column: ``frames[:, x0, :]``
    """
    we = frames[y0, :, :]
    sn = frames[:, x0, :]
    return we, sn

def wrap_time_hours(times: np.ndarray) -> np.ndarray:
    """Convert datetime UT to fractional hours; values > 16 wrap to negative
    (a night spanning 22:00–08:00 plots as −2..+8)."""
    hrs = np.array([t.hour + t.minute/60 + t.second/3600 + t.microsecond/3.6e9
                    for t in times], dtype=float)
    hrs[hrs > 16] -= 24
    return hrs

def insert_gaps(we: np.ndarray, times: np.ndarray, gap_factor: float = 2.05) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN columns into `we` and corresponding interpolated time points
    where Δt > gap_factor × mean(Δt). Returns (we_new, times_new)."""
    if len(times) < 2:
        return we, times
    dt = np.diff(times)
    mean_dt = float(dt.mean())
    threshold = gap_factor * mean_dt
    gap_idx = np.flatnonzero(dt > threshold)
    if not len(gap_idx):
        return we, times
    parts_we = []
    parts_t = []
    last = 0
    H = we.shape[0]
    nan_col = np.full((H, 1), np.nan, dtype=we.dtype)
    for gi in gap_idx:
        parts_we.append(we[:, last:gi+1])
        parts_t.append(times[last:gi+1])
        gap_w = times[gi+1] - times[gi]
        parts_we.append(nan_col)
        parts_t.append(np.array([times[gi] + 0.01 * gap_w]))
        parts_we.append(nan_col)
        parts_t.append(np.array([times[gi+1] - 0.01 * gap_w]))
        last = gi + 1
    parts_we.append(we[:, last:])
    parts_t.append(times[last:])
    return np.concatenate(parts_we, axis=1), np.concatenate(parts_t)

def _cropped_cmap(name: str, crop: tuple[int, int]):
    base = matplotlib.colormaps[name].resampled(128)
    colors = base(np.arange(128))[crop[0]:crop[1]]
    cmap = matplotlib.colors.ListedColormap(colors)
    cmap.set_bad("white")
    return cmap

def scaled_keogram_width(times: np.ndarray, full_night_hours: float,
                         full_night_width_px: int, min_width_px: int) -> int:
    """Return output width so equal pixel width always means equal time span."""
    if len(times) < 2 or full_night_hours <= 0:
        return int(full_night_width_px)
    duration_hours = max(float(np.nanmax(times) - np.nanmin(times)), 0.0)
    frac = min(duration_hours / full_night_hours, 1.0)
    return int(max(min_width_px, round(full_night_width_px * frac)))

def _resize_to_width(path: Path, width_px: int) -> None:
    with Image.open(path) as img:
        if img.width == width_px:
            return
        height_px = max(1, round(img.height * width_px / img.width))
        resized = img.resize((width_px, height_px), Image.Resampling.LANCZOS)
        save_kwargs = {"quality": 92, "optimize": True} if path.suffix.lower() in {".jpg", ".jpeg"} else {}
        resized.save(path, **save_kwargs)

def render_keogram(*, we: np.ndarray, sn: np.ndarray, times: np.ndarray,
                   x0: int, y0: int, R: int, altitude_km: float, fov_deg: float,
                   image_size: int, vmin: float | None, vmax: float | None,
                   title: str, date_label: str,
                   out_path: str | Path, dpi: int,
                   colormap: str, colormap_crop: tuple[int, int],
                   distance_ticks_km: tuple[float, ...],
                   distance_tick_labels: tuple[str, ...],
                   gap_factor: float | None = None,
                   show_colorbar: bool = False,
                   value_label: str = "Intensity",
                   contour: bool = False,
                   contour_levels: int = 64,
                   interpolation: str = "nearest",
                   full_night_hours: float | None = None,
                   full_night_width_px: int | None = None,
                   min_width_px: int = 480) -> None:
    """Render the 2-panel MATLAB-style keogram and save to JPG/PNG."""
    if len(distance_ticks_km) != len(distance_tick_labels):
        raise ValueError("distance_ticks_km and distance_tick_labels must have equal length")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if gap_factor and gap_factor > 0:
        we_g, t_g = insert_gaps(we, times, gap_factor=gap_factor)
        sn_g, _ = insert_gaps(sn, times, gap_factor=gap_factor)
    else:
        we_g, t_g = we, times
        sn_g = sn

    Xl, Yl = pixel_to_km(image_size, x0, y0, R, altitude_km, fov_deg)

    def _ticks_for(line_km):
        good = np.isfinite(line_km)
        positions = np.interp(distance_ticks_km, line_km[good], np.arange(image_size)[good])
        return positions

    cmap = _cropped_cmap(colormap, colormap_crop)

    fig = plt.figure(figsize=(7, 6.5), dpi=dpi)
    ax_top = fig.add_axes([0.10, 0.53, 0.80, 0.40])     # S-N (top, per MATLAB)
    ax_bot = fig.add_axes([0.10, 0.07, 0.80, 0.40])     # W-E (bottom)

    extent = (t_g.min(), t_g.max(), 0, image_size)

    masked_we = np.ma.masked_invalid(we_g)
    masked_sn = np.ma.masked_invalid(sn_g)

    sn_vmin, sn_vmax = matlab_get_range(sn_g) if vmin is None or vmax is None else (vmin, vmax)
    we_vmin, we_vmax = matlab_get_range(we_g) if vmin is None or vmax is None else (vmin, vmax)

    sn_ticks = _ticks_for(Yl)
    we_ticks = _ticks_for(Xl)
    try:
        sn_ylim = tuple(np.interp((-500, 500), distance_ticks_km, sn_ticks))
        we_ylim = tuple(np.interp((-500, 500), distance_ticks_km, we_ticks))
    except ValueError:
        sn_ylim = (0, image_size)
        we_ylim = (0, image_size)

    if contour:
        t_mesh, y_mesh = np.meshgrid(t_g, np.arange(image_size))
        sn_levels = np.linspace(sn_vmin, sn_vmax, contour_levels)
        we_levels = np.linspace(we_vmin, we_vmax, contour_levels)
        im_top = ax_top.contourf(t_mesh, y_mesh, masked_sn, levels=sn_levels,
                                 cmap=cmap, extend="both")
    else:
        im_top = ax_top.imshow(masked_sn, aspect="auto", origin="lower",
                               extent=extent, cmap=cmap, vmin=sn_vmin, vmax=sn_vmax,
                               interpolation=interpolation)
    ax_top.set_ylim(sn_ylim)
    ax_top.set_yticks(sn_ticks)
    ax_top.set_yticklabels(distance_tick_labels, fontsize=9)
    ax_top.set_ylabel("Off-zenith Distance (km)  S→N", fontsize=10)
    ax_top.set_xticklabels([])
    ax_top.set_title(title, fontsize=12)

    if contour:
        im_bot = ax_bot.contourf(t_mesh, y_mesh, masked_we, levels=we_levels,
                                 cmap=cmap, extend="both")
    else:
        im_bot = ax_bot.imshow(masked_we, aspect="auto", origin="lower",
                               extent=extent, cmap=cmap, vmin=we_vmin, vmax=we_vmax,
                               interpolation=interpolation)
    ax_bot.set_ylim(we_ylim)
    ax_bot.set_yticks(we_ticks)
    ax_bot.set_yticklabels(distance_tick_labels, fontsize=9)
    ax_bot.set_ylabel("Off-zenith Distance (km)  W→E", fontsize=10)
    ax_bot.set_xlabel(f"UT hours (date: {date_label})", fontsize=10)

    if show_colorbar:
        cax_top = fig.add_axes([0.92, 0.53, 0.018, 0.40])
        cax_bot = fig.add_axes([0.92, 0.07, 0.018, 0.40])
        cb_top = fig.colorbar(im_top, cax=cax_top)
        cb_bot = fig.colorbar(im_bot, cax=cax_bot)
        cb_top.ax.tick_params(labelsize=8)
        cb_bot.ax.tick_params(labelsize=8)
        cb_top.set_label(value_label, fontsize=8)
        cb_bot.set_label(value_label, fontsize=8)

    fig.savefig(out, format=out.suffix.lstrip("."), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if full_night_hours and full_night_width_px:
        target_width = scaled_keogram_width(
            t_g, full_night_hours, full_night_width_px, min_width_px
        )
        _resize_to_width(out, target_width)
