"""Per-night driver: orchestrates read → filter → render → JSON sidecar."""
from __future__ import annotations
import logging
import time as _time
from datetime import timezone
from pathlib import Path
import numpy as np
from .config import Config
from .io import output_paths, read_night, write_json_sidecar
from .geometry import lookup_calibration
from .intensity import (
    filter_overexposed, count_low_intensity_frames,
    color_range_zenith, fps_for_cadence, matlab_get_range,
)
from .difference import previous_frame_difference
from .enhancement import (
    relative_perturbation, robust_limits, symmetric_limits,
    sigma_clipped_symmetric_limits, signed_brightness_curve, spatial_median3,
)
from .keogram import extract_slices, scaled_keogram_width, wrap_time_hours, render_keogram
from .movie import write_h264_video
from .overlays import compose_movie_frame
from .report import write_contact_sheet, write_html_report
from . import __version__

log = logging.getLogger(__name__)

def process_night(nc_path: str | Path, out_dir: str | Path, cfg: Config, *,
                  keogram_out_dir: str | Path | None = None,
                  overwrite: bool = False, only: set[str] | None = None,
                  dry_run: bool = False, style: str = "matlab",
                  resume_artifacts: bool = True,
                  clean_overlay: bool = False) -> dict:
    """Process one night. Always writes a JSON sidecar (even on skip).
    `only` ⊆ {"keogram","raw","diff"}; default = all three."""
    if style not in {"matlab", "modern"}:
        raise ValueError("style must be 'matlab' or 'modern'")
    only = only or {"keogram", "raw", "diff"}
    night = read_night(nc_path)
    paths = output_paths(out_dir, night.band, night.date, style=style,
                         keogram_out_dir=keogram_out_dir)
    paths.dir.mkdir(parents=True, exist_ok=True)
    modern = style == "modern"

    base_payload = {
        "date": night.date.isoformat(),
        "band": night.band,
        "site": cfg.site.name,
        "style": style,
        "generator_version": __version__,
        "n_frames_total": int(night.intensity.shape[2]),
    }

    if dry_run:
        return {**base_payload, "status": "dry_run", "files_planned": {
            "keogram": paths.keogram.name,
            "movie_raw": paths.movie_raw.name,
            "movie_diff": paths.movie_diff.name,
            "movie_wave": paths.movie_wave.name,
            "contact_sheet": paths.contact_sheet.name,
            "report": paths.report.name,
        }}

    if paths.json.exists() and not overwrite and not resume_artifacts:
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
    cal = lookup_calibration(cfg.calibration, night.date)
    band_cfg = cfg.bands[night.band]

    vmin, vmax = color_range_zenith(
        frames, x0=cal.x0, y0=cal.y0,
        low_threshold=cfg.filter.low_intensity_threshold,
        percentiles=cfg.filter.color_range_percentiles,
    )
    fps = fps_for_cadence(time_seconds, cfg.movie.fps_normal, cfg.movie.fps_low_cadence,
                          threshold_minutes=2.0)

    written: dict[str, str] = {}
    skipped: dict[str, str] = {}
    t_start = _time.time()
    hrs_for_layout = wrap_time_hours(times)
    keogram_width_px = scaled_keogram_width(
        hrs_for_layout,
        cfg.keogram.full_night_hours,
        cfg.keogram.full_night_width_px,
        cfg.keogram.min_width_px,
    )
    movie_width_px = int(round(keogram_width_px / 2))
    if movie_width_px % 2:
        movie_width_px += 1
    output_size = (
        max(2, movie_width_px)
        if modern else cfg.movie.output_size_px
    )
    crf = cfg.movie.web_crf if modern else cfg.movie.crf
    wave_cube: np.ndarray | None = None

    def _should_write(kind: str, path: Path) -> bool:
        if overwrite or not resume_artifacts or not path.exists():
            return True
        skipped[kind] = path.name
        log.info("[%s %s] %s exists, skipping", night.date, night.band, path.name)
        return False

    def _wave_cube() -> np.ndarray:
        nonlocal wave_cube
        if wave_cube is None:
            wave_cube = relative_perturbation(
                frames, time_seconds, cfg.wave.window_minutes, scale=100.0 * cfg.wave.gain
            )
            if cfg.wave.spatial_median:
                wave_cube = spatial_median3(wave_cube)
        return wave_cube

    if "keogram" in only:
        if _should_write("keogram", paths.keogram):
            we, sn = extract_slices(frames, x0=cal.x0, y0=cal.y0)
            kvmin = kvmax = None
            cmap = cfg.keogram.colormap
            crop = cfg.keogram.colormap_crop
            value_label = "Intensity"
            render_keogram(
                we=we, sn=sn, times=hrs_for_layout,
                x0=cal.x0, y0=cal.y0, R=cal.R,
                altitude_km=band_cfg.altitude_km, fov_deg=cfg.image.fov_deg,
                image_size=cfg.image.size_px,
                vmin=kvmin, vmax=kvmax,
                title=f"Keogram of {night.band} Airglow @{cfg.site.name}",
                date_label=times[-1].date().isoformat(),
                out_path=paths.keogram,
                dpi=cfg.keogram.modern_dpi if modern else cfg.keogram.dpi,
                colormap=cmap,
                colormap_crop=crop,
                distance_ticks_km=cfg.keogram.distance_ticks_km,
                distance_tick_labels=cfg.keogram.distance_tick_labels,
                gap_factor=None,
                show_colorbar=modern and cfg.keogram.show_colorbar,
                value_label=value_label,
                contour=modern and cfg.keogram.contour,
                contour_levels=cfg.keogram.contour_levels,
                interpolation=cfg.keogram.interpolation if modern else "nearest",
                full_night_hours=cfg.keogram.full_night_hours,
                full_night_width_px=cfg.keogram.full_night_width_px,
                height_px=cfg.keogram.height_px,
                min_width_px=cfg.keogram.min_width_px,
            )
            written["keogram"] = paths.keogram.name

    if "raw" in only:
        if _should_write("movie_raw", paths.movie_raw):
            raw_vmin, raw_vmax = robust_limits(frames, (1, 99)) if modern else (None, None)
            def _raw_iter():
                n = frames.shape[2]
                for i in range(n):
                    t = times[i]
                    frame_vmin, frame_vmax = (raw_vmin, raw_vmax) if modern else matlab_get_range(frames[:, :, i])
                    yield compose_movie_frame(
                        image=frames[:, :, i], vmin=frame_vmin, vmax=frame_vmax,
                        x0=cal.x0, y0=cal.y0, R=cal.R,
                        date_str=t.strftime("%Y-%m-%d"),
                        time_str=t.strftime("%H:%M:%S") + "UT",
                        site_label=cfg.site.label,
                        site_name=cfg.site.name,
                        band=night.band,
                        markers=[{"name": m.name, "az_deg": m.az_deg, "r_frac": m.r_frac}
                                 for m in cfg.markers],
                        colormap_name=cfg.movie.colormap,
                        colormap_crop=cfg.movie.colormap_crop,
                        is_first_or_last=(i == 0 or i == n - 1),
                        output_size=output_size,
                        clean_overlay=clean_overlay,
                        label_font_size=15 if modern else 13,
                    )
            write_h264_video(_raw_iter(), paths.movie_raw,
                             fps=fps, crf=crf,
                             pix_fmt=cfg.movie.pix_fmt, codec=cfg.movie.codec)
            written["movie_raw"] = paths.movie_raw.name

    if "diff" in only:
        diff = previous_frame_difference(frames)
        diff_times = times[1:]
        if _should_write("movie_diff", paths.movie_diff):
            d_global = (
                sigma_clipped_symmetric_limits(diff, sigma=cfg.difference.clip_sigma)
                if modern else None
            )
            def _diff_iter():
                n = diff.shape[2]
                for i in range(n):
                    t = diff_times[i]
                    d_vmin, d_vmax = d_global if modern else matlab_get_range(diff[:, :, i])
                    yield compose_movie_frame(
                        image=diff[:, :, i], vmin=d_vmin, vmax=d_vmax,
                        x0=cal.x0, y0=cal.y0, R=cal.R,
                        date_str=t.strftime("%Y-%m-%d"),
                        time_str=t.strftime("%H:%M:%S") + "UT",
                        site_label=cfg.site.label,
                        site_name=cfg.site.name,
                        band=night.band if not modern else f"{night.band} TD",
                        markers=[],
                        colormap_name=cfg.movie.colormap,
                        colormap_crop=cfg.movie.colormap_crop,
                        is_first_or_last=(i == 0 or i == n - 1),
                        output_size=output_size,
                        clean_overlay=clean_overlay,
                        label_font_size=15 if modern else 13,
                    )
            write_h264_video(_diff_iter(), paths.movie_diff,
                             fps=fps, crf=crf,
                             pix_fmt=cfg.movie.pix_fmt, codec=cfg.movie.codec)
            written["movie_diff"] = paths.movie_diff.name

    if "wave" in only and modern:
        if _should_write("movie_wave", paths.movie_wave):
            wave = _wave_cube()
            w_vmin, w_vmax = symmetric_limits(wave, cfg.wave.movie_percentile)
            w_limit = max(abs(w_vmin), abs(w_vmax))
            wave_display = signed_brightness_curve(wave, w_limit, cfg.wave.brightness_curve)
            def _wave_iter():
                n = wave_display.shape[2]
                for i in range(n):
                    t = times[i]
                    yield compose_movie_frame(
                        image=wave_display[:, :, i], vmin=-w_limit, vmax=w_limit,
                        x0=cal.x0, y0=cal.y0, R=cal.R,
                        date_str=t.strftime("%Y-%m-%d"),
                        time_str=t.strftime("%H:%M:%S") + "UT",
                        site_label=cfg.site.label,
                        site_name=cfg.site.name,
                        band=f"{night.band} wave",
                        markers=[],
                        colormap_name=cfg.keogram.wave_colormap,
                        colormap_crop=(cfg.keogram.colormap_crop
                                       if cfg.keogram.wave_colormap == "gray" else (0, 128)),
                        is_first_or_last=(i == 0 or i == n - 1),
                        output_size=output_size,
                        clean_overlay=clean_overlay,
                        label_font_size=15 if modern else 13,
                    )
            write_h264_video(_wave_iter(), paths.movie_wave,
                             fps=fps, crf=crf,
                             pix_fmt=cfg.movie.pix_fmt, codec=cfg.movie.codec)
            written["movie_wave"] = paths.movie_wave.name

    if "contact" in only and modern:
        if _should_write("contact_sheet", paths.contact_sheet):
            contact_diff = previous_frame_difference(frames)
            contact_times = times[1:]
            d_vmin, d_vmax = sigma_clipped_symmetric_limits(
                contact_diff, sigma=cfg.difference.clip_sigma
            )
            n = contact_diff.shape[2]
            count = min(cfg.movie.contact_sheet_frames, n)
            idxs = np.linspace(0, n - 1, count, dtype=int)
            sheet_frames = []
            for i in idxs:
                t = contact_times[i]
                rgb = compose_movie_frame(
                    image=contact_diff[:, :, i], vmin=d_vmin, vmax=d_vmax,
                    x0=cal.x0, y0=cal.y0, R=cal.R,
                    date_str=t.strftime("%Y-%m-%d"),
                    time_str=t.strftime("%H:%M:%S") + "UT",
                    site_label=cfg.site.label,
                    site_name=cfg.site.name,
                    band=f"{night.band} TD",
                    markers=[],
                    colormap_name=cfg.movie.colormap,
                    colormap_crop=cfg.movie.colormap_crop,
                    is_first_or_last=(i == 0 or i == n - 1),
                    output_size=360,
                    clean_overlay=True,
                    label_font_size=15,
                )
                sheet_frames.append((rgb, t.strftime("%H:%M UT")))
            write_contact_sheet(sheet_frames, paths.contact_sheet)
            written["contact_sheet"] = paths.contact_sheet.name

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
        "skipped_files": skipped,
        "elapsed_seconds": round(elapsed, 2),
    }
    if "report" in only and modern:
        if _should_write("report", paths.report):
            report_files = {**written, **skipped}
            write_html_report(paths.report, payload, report_files)
            written["report"] = paths.report.name
            payload["files"] = written
    write_json_sidecar(paths.json, payload)
    log.info("[%s %s] %d/%d frames, vmin=%.0f vmax=%.0f, %.1fs",
             night.date, night.band, frames.shape[2], night.intensity.shape[2],
             vmin, vmax, elapsed)
    return payload
