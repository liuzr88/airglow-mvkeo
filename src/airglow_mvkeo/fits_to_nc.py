#!/usr/bin/env python3
"""
Convert ALO airglow FITS image folders to the same NetCDF structure written by
AirglowFITS2NC.m.

The output NetCDF4 files contain:
  intensity(time, y, x) uint16, deflate level 5
  x(x) uint16
  y(y) uint16
  year scalar uint16
  time(time) float64, days from beginning of the year
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
from netCDF4 import Dataset


NX = 512
NY = 512
MIN_VALID_FITS_BYTES = 529_920
REPEATED_IMAGE_GAP_DAYS = 0.5 / 60.0 / 24.0


@dataclass(frozen=True)
class FitsImage:
    path: Path
    date_obs: datetime
    image: np.ndarray


def matlab_datestr_now() -> str:
    """Approximate MATLAB datestr(now), for example '19-Nov-2018 16:27:36'."""
    return datetime.now().strftime("%d-%b-%Y %H:%M:%S")


def days_from_year_start(dt: datetime, year: int) -> float:
    return (dt - datetime(year, 1, 1)).total_seconds() / 86400.0


def split_fits_value(raw_value: str) -> str:
    """Split a FITS card value from its comment, respecting quoted strings."""
    in_quote = False
    result: list[str] = []
    i = 0
    while i < len(raw_value):
        ch = raw_value[i]
        if ch == "'":
            in_quote = not in_quote
            result.append(ch)
        elif ch == "/" and not in_quote:
            break
        else:
            result.append(ch)
        i += 1
    return "".join(result).strip()


def parse_fits_scalar(value: str):
    value = split_fits_value(value)
    if value.startswith("'") and "'" in value[1:]:
        return value[1 : value.find("'", 1)]
    if value in {"T", "F"}:
        return value == "T"
    try:
        if any(ch in value for ch in ".EeDd"):
            return float(value.replace("D", "E").replace("d", "e"))
        return int(value)
    except ValueError:
        return value.strip()


def read_primary_header(raw: bytes) -> tuple[dict[str, object], int]:
    cards: list[str] = []
    end_offset: int | None = None

    for offset in range(0, len(raw), 80):
        card = raw[offset : offset + 80].decode("ascii", errors="replace")
        cards.append(card)
        if card.startswith("END"):
            end_offset = offset + 80
            break

    if end_offset is None:
        raise ValueError("FITS header END card not found")

    header: dict[str, object] = {}
    for card in cards:
        if "=" not in card[:10]:
            continue
        key = card[:8].strip()
        header[key] = parse_fits_scalar(card[10:])

    data_offset = int(math.ceil(end_offset / 2880.0) * 2880)
    return header, data_offset


def fits_dtype(bitpix: int) -> np.dtype:
    dtype_by_bitpix = {
        8: np.dtype("u1"),
        16: np.dtype(">i2"),
        32: np.dtype(">i4"),
        64: np.dtype(">i8"),
        -32: np.dtype(">f4"),
        -64: np.dtype(">f8"),
    }
    try:
        return dtype_by_bitpix[bitpix]
    except KeyError as exc:
        raise ValueError(f"Unsupported BITPIX value: {bitpix}") from exc


def read_fits_image(path: Path, legacy_int16_ceiling: bool = False) -> FitsImage:
    raw = path.read_bytes()
    header, data_offset = read_primary_header(raw)

    bitpix = int(header["BITPIX"])
    naxis = int(header["NAXIS"])
    if naxis != 2:
        raise ValueError(f"Expected 2-D primary image, got NAXIS={naxis}")

    n1 = int(header["NAXIS1"])
    n2 = int(header["NAXIS2"])
    count = n1 * n2
    dtype = fits_dtype(bitpix)

    stored = np.frombuffer(raw, dtype=dtype, count=count, offset=data_offset)
    if stored.size != count:
        raise ValueError("FITS image payload is shorter than expected")

    data = stored.reshape((n2, n1)).astype(np.float64, copy=False)
    bscale = float(header.get("BSCALE", 1.0))
    bzero = float(header.get("BZERO", 0.0))
    if bscale != 1.0 or bzero != 0.0:
        data = data * bscale + bzero
        if legacy_int16_ceiling and bitpix == 16 and bzero == 32768.0:
            # Older MATLAB/FITS-reader outputs in this archive capped unsigned
            # 16-bit physical values at int16 max before uint16 NetCDF writing.
            data = np.clip(data, 0, np.iinfo(np.int16).max)

    date_obs_raw = header.get("DATE-OBS")
    if not isinstance(date_obs_raw, str) or not date_obs_raw:
        raise ValueError("DATE-OBS keyword not found")

    return FitsImage(
        path=path,
        date_obs=datetime.strptime(date_obs_raw[:19], "%Y-%m-%dT%H:%M:%S"),
        image=data,
    )


def orient_image_like_matlab(data: np.ndarray) -> np.ndarray:
    """
    Match the NetCDF orientation produced by:
        Intensity_array(:,:,nc) = fliplr(data')

    MATLAB writes that x/y/time array to NetCDF in a way that appears to Python
    readers as intensity(time, y, x). The equivalent payload plane is flipud.
    """
    return np.flipud(data)


def existing_nc_looks_good(path: Path) -> bool:
    with Dataset(path) as nc:
        time = np.asarray(nc.variables["time"][:], dtype=np.float64)
    return time.size > 1 and np.mean(np.diff(time)) * 24.0 * 60.0 > 0.5


def iter_night_folders(fits_year_dir: Path, year: int) -> list[Path]:
    prefix = f"{year:04d}"
    return sorted(path for path in fits_year_dir.iterdir() if path.is_dir() and path.name.startswith(prefix))


def write_netcdf(
    nc_path: Path,
    year: int,
    airglow_flag: str,
    times: np.ndarray,
    intensity: np.ndarray,
) -> None:
    if intensity.dtype != np.uint16:
        raise TypeError("intensity must be uint16")

    nc_path.parent.mkdir(parents=True, exist_ok=True)
    if nc_path.exists():
        nc_path.unlink()

    n_image, n_y, n_x = intensity.shape
    with Dataset(nc_path, "w", format="NETCDF4") as nc:
        nc.createDimension("x", n_x)
        nc.createDimension("y", n_y)
        nc.createDimension("time", n_image)

        intensity_var = nc.createVariable(
            "intensity",
            "u2",
            ("time", "y", "x"),
            zlib=True,
            complevel=5,
        )
        intensity_var.longname = f"{airglow_flag} airglow intensity"
        intensity_var[:] = intensity

        x_var = nc.createVariable("x", "u2", ("x",))
        x_var.longname = "x pixel (west to east)"
        x_var[:] = np.arange(1, n_x + 1, dtype=np.uint16)

        y_var = nc.createVariable("y", "u2", ("y",))
        y_var.longname = "y pixel (south to north)"
        y_var[:] = np.arange(1, n_y + 1, dtype=np.uint16)

        year_var = nc.createVariable("year", "u2")
        year_var[...] = np.uint16(year)

        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = "days from beginning of the year"
        time_var[:] = times

        nc.Author = "Alan Liu"
        nc.setncattr("Creation Time", matlab_datestr_now())
        nc.setncattr("Creation Location", "ERAU")
        nc.Version = "V1.0"


def convert_night(
    night_dir: Path,
    nc_year_dir: Path,
    year: int,
    airglow_flag: str,
    overwrite: bool,
    legacy_int16_ceiling: bool = False,
) -> bool:
    nc_path = nc_year_dir / f"{airglow_flag}{night_dir.name}.nc"

    if nc_path.exists() and not overwrite:
        try:
            if existing_nc_looks_good(nc_path):
                print(f"{night_dir.name} {airglow_flag} already exists and looks good")
                return False
        except Exception:
            pass

    fits_files = sorted(night_dir.glob(f"{airglow_flag}*fit"))
    if not fits_files:
        print(f"{night_dir.name} No {airglow_flag} Airglow Data")
        return False
    if len(fits_files) < 2:
        print(f"{night_dir.name} {airglow_flag} Not enough images. Discard!")
        return False

    images: list[np.ndarray] = []
    times: list[float] = []

    for fits_path in fits_files:
        if fits_path.stat().st_size < MIN_VALID_FITS_BYTES:
            continue
        try:
            fits_image = read_fits_image(fits_path, legacy_int16_ceiling=legacy_int16_ceiling)
        except Exception as exc:
            print(f"{fits_path.name} skipped: {exc}")
            continue

        if fits_image.image.shape != (NY, NX):
            print(f"{fits_path.name} Incorrect image size. Discard!")
            continue

        images.append(orient_image_like_matlab(fits_image.image))
        times.append(days_from_year_start(fits_image.date_obs, year))

    if len(images) < 2:
        print(f"{night_dir.name} {airglow_flag} Not enough valid images. Discard!")
        return False

    time_array = np.asarray(times, dtype=np.float64)
    intensity = np.stack(images, axis=0)

    repeated = np.flatnonzero(np.diff(time_array) < REPEATED_IMAGE_GAP_DAYS)
    if repeated.size:
        print(f"{night_dir.name} {airglow_flag} repeated image indices: {repeated + 1}")
        keep = np.ones(time_array.shape, dtype=bool)
        keep[repeated] = False
        time_array = time_array[keep]
        intensity = intensity[keep, :, :]

    if intensity.size == 0:
        print(f"{night_dir.name} {airglow_flag} No valid images after filtering. Discard!")
        return False

    if np.nanmax(intensity) >= 2**16 or np.nanmin(intensity) < 0:
        raise ValueError("Intensity_array value outside of [0,65535]")

    write_netcdf(nc_path, year, airglow_flag, time_array, intensity.astype(np.uint16))
    return True


def convert_year(
    year: int = 2020,
    airglow_flag: str = "OH",
    overwrite: bool = True,
    root_dir: Path = Path("/Users/alanliu/OneDriveResearch/Data"),
    data_root: Path | None = None,
    nc_root: Path | None = None,
    legacy_int16_ceiling: bool = False,
) -> None:
    data_root = data_root if data_root is not None else root_dir / "ALOASI"
    nc_root = nc_root if nc_root is not None else root_dir / "ALOASI" / "NC"

    fits_year_dir = data_root / f"{year:04d}"
    nc_year_dir = nc_root / f"{year:04d}"

    if not fits_year_dir.is_dir():
        raise FileNotFoundError(f"FITS year directory not found: {fits_year_dir}")

    folders = iter_night_folders(fits_year_dir, year)
    for index, night_dir in enumerate(folders, start=1):
        start = perf_counter()
        print(f"Converting {night_dir.name} {index:3d}/{len(folders):3d} nights. ", end="", flush=True)
        converted = convert_night(
            night_dir,
            nc_year_dir,
            year,
            airglow_flag,
            overwrite,
            legacy_int16_ceiling=legacy_int16_ceiling,
        )
        if converted:
            print(f"Elapsed time {perf_counter() - start:5.2f} s")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("year", nargs="?", type=int, default=2020)
    parser.add_argument("airglow_flag", nargs="?", default="OH")
    parser.add_argument("--overwrite", type=int, choices=(0, 1), default=1)
    parser.add_argument("--root-dir", type=Path, default=Path("/Users/alanliu/OneDriveResearch/Data"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--nc-root", type=Path)
    parser.add_argument(
        "--legacy-int16-ceiling",
        action="store_true",
        help="Reproduce older outputs where unsigned 16-bit FITS values were capped at 32767.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    convert_year(
        year=args.year,
        airglow_flag=args.airglow_flag,
        overwrite=bool(args.overwrite),
        root_dir=args.root_dir,
        data_root=args.data_root,
        nc_root=args.nc_root,
        legacy_int16_ceiling=args.legacy_int16_ceiling,
    )


if __name__ == "__main__":
    main()
