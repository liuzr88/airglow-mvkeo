"""Year-directory orchestrator with multiprocessing."""
from __future__ import annotations
import logging
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from typing import Iterable
from .config import Config
from .processing import process_night

log = logging.getLogger(__name__)

def _worker(args):
    nc_path, out_dir, cfg, overwrite, only, dry_run = args
    try:
        return process_night(nc_path, out_dir, cfg,
                             overwrite=overwrite, only=only, dry_run=dry_run)
    except Exception as e:  # noqa: BLE001
        return {
            "status": "io_error",
            "source": str(nc_path),
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }

def discover_files(year_dir: str | Path, band: str | None = None) -> list[Path]:
    p = Path(year_dir)
    files: list[Path] = []
    bands = ["OH", "O5"] if band in (None, "all") else [band]
    for b in bands:
        files.extend(sorted(p.glob(f"{b}*.nc")))
    return files

def run_year(year_dir: str | Path, out_dir: str | Path, cfg: Config, *,
             band: str | None = None, workers: int | None = None,
             overwrite: bool = False, only: Iterable[str] | None = None,
             dry_run: bool = False) -> list[dict]:
    files = discover_files(year_dir, band)
    if not files:
        log.warning("no files found in %s", year_dir)
        return []
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    only_set = set(only) if only else None
    args_list = [(f, Path(out_dir), cfg, overwrite, only_set, dry_run) for f in files]
    log.info("processing %d files with %d workers", len(files), workers)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(_worker, args_list)
    return results
