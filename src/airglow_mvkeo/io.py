"""NetCDF input, JSON sidecar output, output path layout."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any
import numpy as np
import xarray as xr

# MATLAB datenum: days since year 0000-01-00 proleptic. datetime(1,1,1) is
# MATLAB datenum 367. Therefore:
#   datetime_utc = datetime(1,1,1) + timedelta(days=datenum - 367)
_MATLAB_EPOCH_OFFSET = 367

def matlab_datenum_to_datetime(d):
    """Convert MATLAB datenum (scalar or ndarray) to UTC datetime(s)."""
    base = datetime(1, 1, 1, tzinfo=timezone.utc)
    if np.ndim(d) == 0:
        return base + timedelta(days=float(d) - _MATLAB_EPOCH_OFFSET)
    arr = np.asarray(d, dtype=float)
    return np.array(
        [base + timedelta(days=float(x) - _MATLAB_EPOCH_OFFSET) for x in arr.ravel()]
    ).reshape(arr.shape)

_NAME_RE = re.compile(r"^(?P<band>OH|O5)(?P<date>\d{8})", re.IGNORECASE)

@dataclass(frozen=True)
class NightData:
    intensity: np.ndarray
    times: np.ndarray
    band: str
    date: date
    source_path: Path

@dataclass(frozen=True)
class OutputPaths:
    dir: Path
    keogram: Path
    movie_raw: Path
    movie_diff: Path
    movie_wave: Path
    contact_sheet: Path
    report: Path
    json: Path

def parse_filename(p: Path) -> tuple[str, date]:
    m = _NAME_RE.match(p.stem)
    if not m:
        raise ValueError(f"unrecognized filename pattern: {p.name}")
    band = m.group("band").upper()
    s = m.group("date")
    return band, date(int(s[:4]), int(s[4:6]), int(s[6:8]))

def day_of_year_to_datetime(t: float, year: int) -> datetime:
    """Convert a days-from-Jan-1 value (as stored in Step-3 NetCDF files) to a
    UTC datetime.  ``t=0.0`` is midnight on Jan 1 of *year*."""
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=float(t))


def read_night(path: str | Path) -> NightData:
    p = Path(path)
    band, d = parse_filename(p)
    with xr.open_dataset(p) as ds:
        # MATLAB's AirglowFITS2NC.m declared dims {'x','y'} but assigned them
        # in column-major order, so the dim named 'x' is the displayed vertical
        # image axis and 'y' is the displayed horizontal image axis. The result
        # is an image cube indexed as (row, column, time).
        intensity = np.asarray(
            ds["intensity"].transpose("x", "y", "time").values, dtype=np.float32
        )
        # Time values are days-from-Jan-1 of the observation year (not raw
        # MATLAB datenums).  Read the year scalar; fall back to the filename.
        if "year" in ds:
            year = int(ds["year"].values)
        else:
            year = d.year
        times = np.array(
            [day_of_year_to_datetime(float(t), year) for t in ds["time"].values]
        )
    return NightData(intensity=intensity, times=times, band=band, date=d, source_path=p)

def output_paths(out_dir: str | Path, band: str, d: date, *,
                 style: str = "matlab") -> OutputPaths:
    out = Path(out_dir) / f"{d.year:04d}"
    s = d.strftime("%Y%m%d")
    suffix = "" if style == "matlab" else "_web"
    return OutputPaths(
        dir=out,
        keogram=out / f"{band}Keog{s}{suffix}.jpg",
        movie_raw=out / f"{band}{s}{suffix}.mp4",
        movie_diff=out / f"{band}{s}_TD{suffix}.mp4",
        movie_wave=out / f"{band}{s}_wave{suffix}.mp4",
        contact_sheet=out / f"{band}{s}{suffix}_contact.jpg",
        report=out / f"{band}{s}{suffix}_report.html",
        json=out / f"{band}{s}{suffix}.json",
    )

def write_json_sidecar(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
