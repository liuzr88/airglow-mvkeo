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
class MovieConfig:
    fps_normal: int
    fps_low_cadence: int
    crf: int
    codec: str
    pix_fmt: str
    colormap: str
    colormap_crop: tuple[int, int]

@dataclass(frozen=True)
class KeogramConfig:
    format: str
    dpi: int
    colormap: str
    colormap_crop: tuple[int, int]
    distance_ticks_km: tuple[float, ...]
    distance_tick_labels: tuple[str, ...]
    gap_threshold_factor: float

@dataclass(frozen=True)
class DifferenceConfig:
    window_minutes: float
    clip_sigma: float

@dataclass(frozen=True)
class FilterConfig:
    overexposed_mean_threshold: float
    min_frames: int
    min_usable_at_low: int
    low_intensity_threshold: float
    color_range_percentiles: tuple[float, float]

@dataclass(frozen=True)
class Config:
    site: SiteConfig
    bands: dict[str, BandConfig]
    image: ImageConfig
    calibration: list[CalibrationEntry]
    markers: list[MarkerConfig]
    movie: MovieConfig
    keogram: KeogramConfig
    difference: DifferenceConfig
    filter: FilterConfig

def load_config(path: str | Path) -> Config:
    p = Path(path)
    with p.open("rb") as f:
        raw = tomllib.load(f)
    cfg = Config(
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
            fps_normal=raw["movie"]["fps_normal"],
            fps_low_cadence=raw["movie"]["fps_low_cadence"],
            crf=raw["movie"]["crf"],
            codec=raw["movie"]["codec"],
            pix_fmt=raw["movie"]["pix_fmt"],
            colormap=raw["movie"]["colormap"],
            colormap_crop=tuple(raw["movie"]["colormap_crop"]),
        ),
        keogram=KeogramConfig(
            format=raw["keogram"]["format"],
            dpi=raw["keogram"]["dpi"],
            colormap=raw["keogram"]["colormap"],
            colormap_crop=tuple(raw["keogram"]["colormap_crop"]),
            distance_ticks_km=tuple(raw["keogram"]["distance_ticks_km"]),
            distance_tick_labels=tuple(raw["keogram"]["distance_tick_labels"]),
            gap_threshold_factor=raw["keogram"]["gap_threshold_factor"],
        ),
        difference=DifferenceConfig(**raw["difference"]),
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
