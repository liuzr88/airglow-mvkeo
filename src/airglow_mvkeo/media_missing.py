"""Create missing keograms and movies from existing NetCDF files."""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import load_config
from .io import SUPPORTED_BANDS, output_paths, parse_filename
from .processing import process_night

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "alo.toml"
CORE_ARTIFACTS = {"keogram", "raw", "diff"}


def discover_years(nc_root: Path) -> list[int]:
    if not nc_root.is_dir():
        return []
    return sorted(
        int(path.name)
        for path in nc_root.iterdir()
        if path.is_dir() and len(path.name) == 4 and path.name.isdigit()
    )


def discover_nc_files(nc_root: Path, years: Iterable[int], band: str) -> list[Path]:
    bands = SUPPORTED_BANDS if band == "all" else (band,)
    files: list[Path] = []
    for year in years:
        year_dir = nc_root / f"{year:04d}"
        for one_band in bands:
            files.extend(sorted(year_dir.glob(f"{one_band}*.nc")))
    return files


def current_month_bounds(now: datetime | None = None) -> tuple[datetime, datetime]:
    """Return local-time bounds for the current calendar month."""
    now = now.astimezone() if now else datetime.now().astimezone()
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


def file_is_from_current_month(path: Path, *, now: datetime | None = None) -> bool:
    """Use file modification time as the media creation freshness check."""
    start, end = current_month_bounds(now)
    mtime = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    return start <= mtime < end


def artifacts_to_create_for_nc(
    nc_path: Path,
    *,
    mv_root: Path,
    kg_root: Path,
    style: str,
    now: datetime | None = None,
) -> dict[str, str]:
    band, date = parse_filename(nc_path)
    paths = output_paths(mv_root, band, date, style=style, keogram_out_dir=kg_root)
    artifact_paths = {
        "keogram": paths.keogram,
        "raw": paths.movie_raw,
        "diff": paths.movie_diff,
    }
    needed: dict[str, str] = {}
    for artifact, path in artifact_paths.items():
        if not path.exists():
            needed[artifact] = "missing"
        elif not file_is_from_current_month(path, now=now):
            needed[artifact] = "older_than_current_month"
    return needed


def _worker(args):
    nc_path, mv_root, kg_root, cfg, style, clean_overlay, artifacts = args
    try:
        return process_night(
            nc_path,
            mv_root,
            cfg,
            keogram_out_dir=kg_root,
            overwrite=True,
            only=artifacts,
            style=style,
            resume_artifacts=True,
            clean_overlay=clean_overlay,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "io_error",
            "source": str(nc_path),
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(),
        }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("years", nargs="*", type=int, help="Years to scan. Omit to scan all NC year folders.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML config path.")
    parser.add_argument("--band", choices=[*SUPPORTED_BANDS, "all"], default="all")
    parser.add_argument("--style", choices=["matlab", "modern", "web"], default="modern")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate media even when files already exist.")
    parser.add_argument("--clean-overlay", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = load_config(args.config)
    style = "modern" if args.style == "web" else args.style

    data_root = Path(cfg.paths.data_root).expanduser()
    nc_root = data_root / cfg.paths.nc_root
    mv_root = data_root / cfg.paths.mv_root
    kg_root = data_root / cfg.paths.kg_root
    years = args.years or discover_years(nc_root)
    if not years:
        raise FileNotFoundError(f"No NetCDF year folders found under: {nc_root}")

    nc_files = discover_nc_files(nc_root, years, args.band)
    to_process: list[tuple[Path, set[str]]] = []
    planned: dict[str, dict[str, str]] = {}
    now = datetime.now().astimezone()
    for nc_path in nc_files:
        needed = (
            {artifact: "overwrite" for artifact in CORE_ARTIFACTS}
            if args.overwrite
            else artifacts_to_create_for_nc(
                nc_path, mv_root=mv_root, kg_root=kg_root, style=style, now=now
            )
        )
        if needed:
            to_process.append((nc_path, set(needed)))
            planned[str(nc_path)] = needed

    summary = {
        "years": years,
        "band": args.band,
        "style": style,
        "nc_files_found": len(nc_files),
        "nights_to_process": len(to_process),
        "mv_root": str(mv_root),
        "kg_root": str(kg_root),
    }
    if args.dry_run:
        print(json.dumps({**summary, "planned": planned}, indent=2))
        return 0

    if not to_process:
        print(json.dumps({**summary, "status": "nothing_missing"}, indent=2))
        return 0

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    worker_args = [
        (path, mv_root, kg_root, cfg, style, args.clean_overlay, artifacts)
        for path, artifacts in to_process
    ]
    if workers == 1:
        results = [_worker(item) for item in worker_args]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_worker, worker_args)

    final = {
        **summary,
        "status": "ok",
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "errors": sum(1 for r in results if r.get("status") == "io_error"),
        "insufficient": sum(1 for r in results if r.get("status") == "insufficient_data"),
        "overexposed": sum(1 for r in results if r.get("status") == "overexposed"),
    }
    print(json.dumps(final, indent=2))
    return 0 if final["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
