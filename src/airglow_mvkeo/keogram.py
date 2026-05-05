"""Keogram building & rendering."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .geometry import pixel_to_km

def extract_slices(frames: np.ndarray, x0: int, y0: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (we, sn). we[:, t] is the W-E line at row y0; sn[:, t] is the S-N line at col x0."""
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

def render_keogram(*, we: np.ndarray, sn: np.ndarray, times: np.ndarray,
                   x0: int, y0: int, R: int, altitude_km: float, fov_deg: float,
                   image_size: int, vmin: float, vmax: float,
                   title: str, date_label: str,
                   out_path: str | Path, dpi: int,
                   colormap: str, colormap_crop: tuple[int, int],
                   distance_ticks_km: tuple[float, ...],
                   distance_tick_labels: tuple[str, ...]) -> None:
    """Render the 2-panel keogram (W-E top, S-N bottom) and save to JPG/PNG."""
    if len(distance_ticks_km) != len(distance_tick_labels):
        raise ValueError("distance_ticks_km and distance_tick_labels must have equal length")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    we_g, t_g = insert_gaps(we, times)
    sn_g, _   = insert_gaps(sn, times)

    Xl, Yl = pixel_to_km(image_size, x0, y0, R, altitude_km, fov_deg)

    def _ticks_for(line_km):
        good = np.isfinite(line_km)
        positions = np.interp(distance_ticks_km, line_km[good], np.arange(image_size)[good])
        return positions

    cmap = _cropped_cmap(colormap, colormap_crop)

    fig = plt.figure(figsize=(7, 6.5), dpi=dpi)
    ax_top = fig.add_axes([0.10, 0.53, 0.80, 0.40])     # W-E (top, per MATLAB)
    ax_bot = fig.add_axes([0.10, 0.07, 0.80, 0.40])     # S-N (bottom)

    extent = (t_g.min(), t_g.max(), 0, image_size)

    masked_we = np.ma.masked_invalid(we_g)
    masked_sn = np.ma.masked_invalid(sn_g)

    ax_top.imshow(masked_we, aspect="auto", origin="lower",
                  extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest")
    ax_top.set_yticks(_ticks_for(Xl))
    ax_top.set_yticklabels(distance_tick_labels, fontsize=9)
    ax_top.set_ylabel("Off-zenith Distance (km)  W→E", fontsize=10)
    ax_top.set_xticklabels([])
    ax_top.set_title(title, fontsize=12)

    ax_bot.imshow(masked_sn, aspect="auto", origin="lower",
                  extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest")
    ax_bot.set_yticks(_ticks_for(Yl))
    ax_bot.set_yticklabels(distance_tick_labels, fontsize=9)
    ax_bot.set_ylabel("Off-zenith Distance (km)  S→N", fontsize=10)
    ax_bot.set_xlabel(f"UT hours (date: {date_label})", fontsize=10)

    fig.savefig(out, format=out.suffix.lstrip("."), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
