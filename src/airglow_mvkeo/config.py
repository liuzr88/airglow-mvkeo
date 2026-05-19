"""Typed config loaded from TOML."""
from __future__ import annotations
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

@dataclass(frozen=True)
class SiteConfig:
    name: str
    lat_deg: float
    lon_deg: float
    label: str
    ut_offset_hours: float

@dataclass(frozen=True)
class BandConfig:
    altitude_km: float

@dataclass(frozen=True)
class ImageConfig:
    size_px: int
    fov_deg: float

@dataclass(frozen=True)
class CalibrationEntry:
    since: date
    x0: int
    y0: int
    R: int
    P: float

@dataclass(frozen=True)
class MarkerConfig:
    name: str
    az_deg: float
    r_frac: float

@dataclass(frozen=True)
class PathsConfig:
    data_root: str
    nc_root: str
    mv_root: str
    kg_root: str

@dataclass(frozen=True)
class MovieConfig:
    fps_normal: int
    fps_low_cadence: int
    crf: int
    web_crf: int
    codec: str
    pix_fmt: str
    colormap: str
    colormap_crop: tuple[int, int]
    output_size_px: int
    web_output_size_px: int
    contact_sheet_frames: int

@dataclass(frozen=True)
class KeogramConfig:
    format: str
    dpi: int
    colormap: str
    colormap_crop: tuple[int, int]
    distance_ticks_km: tuple[float, ...]
    distance_tick_labels: tuple[str, ...]
    gap_threshold_factor: float
    modern_dpi: int
    modern_colormap: str
    wave_colormap: str
    smooth_time_bins: int
    show_colorbar: bool
    contour: bool
    contour_levels: int
    interpolation: str
    full_night_hours: float
    full_night_width_px: int
    height_px: int
    min_width_px: int

@dataclass(frozen=True)
class DifferenceConfig:
    window_minutes: float
    clip_sigma: float

@dataclass(frozen=True)
class WaveConfig:
    window_minutes: float
    movie_percentile: float
    keogram_percentile: float
    gain: float
    spatial_median: bool
    brightness_curve: float

@dataclass(frozen=True)
class FilterConfig:
    overexposed_mean_threshold: float
    min_frames: int
    min_usable_at_low: int
    low_intensity_threshold: float
    color_range_percentiles: tuple[float, float]

@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    site: SiteConfig
    bands: dict[str, BandConfig]
    image: ImageConfig
    calibration: list[CalibrationEntry]
    markers: list[MarkerConfig]
    movie: MovieConfig
    keogram: KeogramConfig
    difference: DifferenceConfig
    wave: WaveConfig
    filter: FilterConfig

def load_config(path: str | Path) -> Config:
    p = Path(path)
    with p.open("rb") as f:
        raw = tomllib.load(f)
    paths_raw = raw.get("paths", {})
    movie_raw = raw["movie"]
    keo_raw = raw["keogram"]
    cfg = Config(
        paths=PathsConfig(
            data_root=paths_raw.get("data_root", "/Users/alanliu/OneDriveResearch/Data/ALOASI"),
            nc_root=paths_raw.get("nc_root", "NC"),
            mv_root=paths_raw.get("mv_root", "MV"),
            kg_root=paths_raw.get("kg_root", "KG"),
        ),
        site=SiteConfig(**raw["site"]),
        bands={k: BandConfig(**v) for k, v in raw["bands"].items()},
        image=ImageConfig(**raw["image"]),
        calibration=sorted(
            [CalibrationEntry(since=date.fromisoformat(c["since"]),
                              x0=c["x0"], y0=c["y0"], R=c["R"], P=c["P"])
             for c in raw["calibration"]],
            key=lambda c: c.since,
        ),
        markers=[MarkerConfig(**m) for m in raw["markers"]],
        movie=MovieConfig(
            fps_normal=movie_raw["fps_normal"],
            fps_low_cadence=movie_raw["fps_low_cadence"],
            crf=movie_raw["crf"],
            web_crf=movie_raw.get("web_crf", 24),
            codec=movie_raw["codec"],
            pix_fmt=movie_raw["pix_fmt"],
            colormap=movie_raw["colormap"],
            colormap_crop=tuple(movie_raw["colormap_crop"]),
            output_size_px=movie_raw.get("output_size_px", 540),
            web_output_size_px=movie_raw.get("web_output_size_px", 540),
            contact_sheet_frames=movie_raw.get("contact_sheet_frames", 12),
        ),
        keogram=KeogramConfig(
            format=keo_raw["format"],
            dpi=keo_raw["dpi"],
            colormap=keo_raw["colormap"],
            colormap_crop=tuple(keo_raw["colormap_crop"]),
            distance_ticks_km=tuple(keo_raw["distance_ticks_km"]),
            distance_tick_labels=tuple(keo_raw["distance_tick_labels"]),
            gap_threshold_factor=keo_raw["gap_threshold_factor"],
            modern_dpi=keo_raw.get("modern_dpi", keo_raw["dpi"]),
            modern_colormap=keo_raw.get("modern_colormap", keo_raw["colormap"]),
            wave_colormap=keo_raw.get("wave_colormap", "RdBu_r"),
            smooth_time_bins=keo_raw.get("smooth_time_bins", 3),
            show_colorbar=keo_raw.get("show_colorbar", True),
            contour=keo_raw.get("contour", False),
            contour_levels=keo_raw.get("contour_levels", 64),
            interpolation=keo_raw.get("interpolation", "nearest"),
            full_night_hours=keo_raw.get("full_night_hours", 10.0),
            full_night_width_px=keo_raw.get("full_night_width_px", 1136),
            height_px=keo_raw.get("height_px", 850),
            min_width_px=keo_raw.get("min_width_px", 480),
        ),
        difference=DifferenceConfig(**raw["difference"]),
        wave=WaveConfig(**raw.get("wave", {
            "window_minutes": 60,
            "movie_percentile": 99,
            "keogram_percentile": 98,
            "gain": 1.0,
            "spatial_median": True,
            "brightness_curve": 0.0,
        })),
        filter=FilterConfig(
            overexposed_mean_threshold=raw["filter"]["overexposed_mean_threshold"],
            min_frames=raw["filter"]["min_frames"],
            min_usable_at_low=raw["filter"]["min_usable_at_low"],
            low_intensity_threshold=raw["filter"]["low_intensity_threshold"],
            color_range_percentiles=tuple(raw["filter"]["color_range_percentiles"]),
        ),
    )
    if not cfg.calibration:
        raise ValueError("calibration table is empty")
    return cfg
