"""Per-night driver: orchestrates read → filter → render → JSON sidecar."""
from __future__ import annotations
import logging
import time as _time
from datetime import timezone
from pathlib import Path
import numpy as np
from .config import Config
from .io import NightData, OutputPaths, output_paths, read_night, write_json_sidecar
from .geometry import lookup_calibration
from .intensity import (
    filter_overexposed, count_low_intensity_frames,
    color_range_zenith, fps_for_cadence,
)
from .difference import running_mean_subtract
from .keogram import extract_slices, wrap_time_hours, render_keogram
from .movie import write_h264_video
from .overlays import compose_movie_frame
from . import __version__

log = logging.getLogger(__name__)

def process_night(nc_path: str | Path, out_dir: str | Path, cfg: Config, *,
                  overwrite: bool = False, only: set[str] | None = None,
                  dry_run: bool = False) -> dict:
    """Process one night. Always writes a JSON sidecar (even on skip).
    `only` ⊆ {"keogram","raw","diff"}; default = all three."""
    only = only or {"keogram", "raw", "diff"}
    night = read_night(nc_path)
    paths = output_paths(out_dir, night.band, night.date)
    paths.dir.mkdir(parents=True, exist_ok=True)

    base_payload = {
        "date": night.date.isoformat(),
        "band": night.band,
        "site": cfg.site.name,
        "generator_version": __version__,
        "n_frames_total": int(night.intensity.shape[2]),
    }

    if dry_run:
        return {**base_payload, "status": "dry_run", "files_planned": {
            "keogram": paths.keogram.name,
            "movie_raw": paths.movie_raw.name,
            "movie_diff": paths.movie_diff.name,
        }}

    if paths.json.exists() and not overwrite:
        log.info("[%s %s] sidecar exists, skipping (use --overwrite to redo)",
                 night.date, night.band)
        return {**base_payload, "status": "skipped_existing"}

    keep = filter_overexposed(night.intensity, cfg.filter.overexposed_mean_threshold)
    if len(keep) < cfg.filter.min_frames:
        payload = {**base_payload, "status": "insufficient_data",
                   "n_frames_used": len(keep)}
        write_json_sidecar(paths.json, payload)
        return payload

    if count_low_intensity_frames(night.intensity, cfg.filter.low_intensity_threshold) \
       < cfg.filter.min_usable_at_low:
        payload = {**base_payload, "status": "overexposed",
                   "n_frames_used": len(keep)}
        write_json_sidecar(paths.json, payload)
        return payload

    frames = night.intensity[..., keep]
    times = night.times[keep]
    time_seconds = np.array(
        [(t - night.times[keep[0]]).total_seconds() for t in times], dtype=float
    )
    secs_of_day = np.array(
        [t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6 for t in times],
        dtype=float,
    )

    cal = lookup_calibration(cfg.calibration, night.date)
    band_cfg = cfg.bands[night.band]

    vmin, vmax = color_range_zenith(
        frames, x0=cal.x0, y0=cal.y0,
        low_threshold=cfg.filter.low_intensity_threshold,
        percentiles=cfg.filter.color_range_percentiles,
    )
    fps = fps_for_cadence(secs_of_day, cfg.movie.fps_normal, cfg.movie.fps_low_cadence,
                          threshold_minutes=2.0)

    written: dict[str, str] = {}
    t_start = _time.time()

    if "keogram" in only:
        we, sn = extract_slices(frames, x0=cal.x0, y0=cal.y0)
        hrs = wrap_time_hours(times)
        render_keogram(
            we=we, sn=sn, times=hrs,
            x0=cal.x0, y0=cal.y0, R=cal.R,
            altitude_km=band_cfg.altitude_km, fov_deg=cfg.image.fov_deg,
            image_size=cfg.image.size_px,
            vmin=vmin, vmax=vmax,
            title=f"Keogram of {night.band} Airglow @{cfg.site.name}",
            date_label=night.date.isoformat(),
            out_path=paths.keogram, dpi=cfg.keogram.dpi,
            colormap=cfg.keogram.colormap,
            colormap_crop=cfg.keogram.colormap_crop,
            distance_ticks_km=cfg.keogram.distance_ticks_km,
            distance_tick_labels=cfg.keogram.distance_tick_labels,
        )
        written["keogram"] = paths.keogram.name

    if "raw" in only:
        def _raw_iter():
            n = frames.shape[2]
            for i in range(n):
                t = times[i]
                yield compose_movie_frame(
                    image=frames[:, :, i], vmin=vmin, vmax=vmax,
                    x0=cal.x0, y0=cal.y0, R=cal.R,
                    date_str=t.strftime("%Y-%m-%d"),
                    time_str=t.strftime("%H:%M:%S UT"),
                    site_label=f"{cfg.site.name} {night.band} {cfg.site.label}",
                    markers=[{"name": m.name, "az_deg": m.az_deg, "r_frac": m.r_frac}
                             for m in cfg.markers],
                    colormap_name=cfg.movie.colormap,
                    colormap_crop=cfg.movie.colormap_crop,
                    is_first_or_last=(i == 0 or i == n - 1),
                )
        write_h264_video(_raw_iter(), paths.movie_raw,
                         fps=fps, crf=cfg.movie.crf,
                         pix_fmt=cfg.movie.pix_fmt, codec=cfg.movie.codec)
        written["movie_raw"] = paths.movie_raw.name

    if "diff" in only:
        diff = running_mean_subtract(frames, time_seconds, cfg.difference.window_minutes)
        sigma = float(np.std(diff))
        clip = cfg.difference.clip_sigma * sigma
        d_vmin, d_vmax = -clip, +clip
        def _diff_iter():
            n = diff.shape[2]
            for i in range(n):
                t = times[i]
                yield compose_movie_frame(
                    image=diff[:, :, i], vmin=d_vmin, vmax=d_vmax,
                    x0=cal.x0, y0=cal.y0, R=cal.R,
                    date_str=t.strftime("%Y-%m-%d"),
                    time_str=t.strftime("%H:%M:%S UT") + " (Δ)",
                    site_label=f"{cfg.site.name} {night.band} difference",
                    markers=[],
                    colormap_name=cfg.movie.colormap,
                    colormap_crop=cfg.movie.colormap_crop,
                    is_first_or_last=(i == 0 or i == n - 1),
                )
        write_h264_video(_diff_iter(), paths.movie_diff,
                         fps=fps, crf=cfg.movie.crf,
                         pix_fmt=cfg.movie.pix_fmt, codec=cfg.movie.codec)
        written["movie_diff"] = paths.movie_diff.name

    elapsed = _time.time() - t_start
    payload = {
        **base_payload,
        "status": "ok",
        "start_ut": times[0].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "end_ut": times[-1].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_frames_used": int(frames.shape[2]),
        "display_range": [vmin, vmax],
        "fps": fps,
        "calibration": {"x0": cal.x0, "y0": cal.y0, "R": cal.R, "P": cal.P},
        "files": written,
        "elapsed_seconds": round(elapsed, 2),
    }
    write_json_sidecar(paths.json, payload)
    log.info("[%s %s] %d/%d frames, vmin=%.0f vmax=%.0f, %.1fs",
             night.date, night.band, frames.shape[2], night.intensity.shape[2],
             vmin, vmax, elapsed)
    return payload
