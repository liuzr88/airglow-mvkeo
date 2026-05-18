"""argparse entrypoint."""
from __future__ import annotations
import argparse
import json
import logging
import sys
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/airglow_mvkeo_matplotlib")

from .config import load_config
from .batch import run_year
from .processing import process_night

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "alo.toml"

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", type=Path, help="output root directory")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML config path")
    p.add_argument("--band", choices=["OH", "O5", "all"], default="all")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only", choices=["keogram", "raw", "diff", "wave", "contact", "report", "all"], default="all")
    p.add_argument("--style", choices=["matlab", "modern", "web"], default="matlab",
                   help="matlab preserves validated legacy visuals; modern/web adds wave-enhanced web artifacts")
    p.add_argument("--clean-overlay", action="store_true",
                   help="use a minimal movie overlay for presentation/web embeds")
    p.add_argument("--no-resume-artifacts", action="store_true",
                   help="skip whole nights when the JSON sidecar exists instead of resuming missing artifacts")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-file", type=Path)

def _setup_logging(verbose: bool, log_file: Path | None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=handlers)

def _only_set(s: str) -> set[str] | None:
    return None if s == "all" else {s}

def _style(s: str) -> str:
    return "modern" if s == "web" else s

def _default_out(cfg, style: str) -> Path:
    root = Path(cfg.paths.data_root)
    return root / cfg.paths.mv_root

def _date_file(cfg, sdate: str, band: str) -> Path:
    if len(sdate) != 8 or not sdate.isdigit():
        raise ValueError("date must be YYYYMMDD")
    root = Path(cfg.paths.data_root)
    year = sdate[:4]
    return root / cfg.paths.nc_root / year / f"{band}{sdate}.nc"

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="airglow-mvkeo")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("run-night", help="process a single NetCDF file")
    p1.add_argument("file", type=Path)
    _add_common(p1)

    p0 = sub.add_parser("run-date", help="process one date from configured data_root/NC/YYYY")
    p0.add_argument("date", help="YYYYMMDD")
    _add_common(p0)

    p2 = sub.add_parser("run-year", help="process every NC file in a year directory")
    p2.add_argument("year_dir", type=Path)
    p2.add_argument("--workers", type=int, default=None)
    _add_common(p2)

    args = p.parse_args(argv)
    _setup_logging(args.verbose, args.log_file)
    cfg = load_config(args.config)
    only = _only_set(args.only)
    style = _style(args.style)
    out = args.out or _default_out(cfg, style)
    resume_artifacts = not args.no_resume_artifacts

    if args.cmd == "run-night":
        result = process_night(args.file, out, cfg,
                               overwrite=args.overwrite, only=only, dry_run=args.dry_run,
                               style=style, resume_artifacts=resume_artifacts,
                               clean_overlay=args.clean_overlay)
        print(json.dumps(result, indent=2))
        return 0 if result.get("status") in ("ok", "dry_run", "skipped_existing") else 1

    if args.cmd == "run-date":
        bands = ["OH", "O5"] if args.band == "all" else [args.band]
        results = []
        for band in bands:
            fn = _date_file(cfg, args.date, band)
            if not fn.exists():
                results.append({"status": "io_error", "source": str(fn), "error": "file not found"})
                continue
            results.append(process_night(fn, out, cfg,
                                         overwrite=args.overwrite, only=only,
                                         dry_run=args.dry_run, style=style,
                                         resume_artifacts=resume_artifacts,
                                         clean_overlay=args.clean_overlay))
        print(json.dumps(results[0] if len(results) == 1 else results, indent=2))
        return 0 if all(r.get("status") in ("ok", "dry_run", "skipped_existing") for r in results) else 1

    results = run_year(args.year_dir, out, cfg,
                       band=args.band, workers=args.workers,
                       overwrite=args.overwrite, only=only, dry_run=args.dry_run,
                       style=style, resume_artifacts=resume_artifacts,
                       clean_overlay=args.clean_overlay)
    summary = {"total": len(results),
               "ok": sum(1 for r in results if r.get("status") == "ok"),
               "skipped": sum(1 for r in results if r.get("status") == "skipped_existing"),
               "errors": sum(1 for r in results if r.get("status") == "io_error"),
               "insufficient": sum(1 for r in results if r.get("status") == "insufficient_data"),
               "overexposed": sum(1 for r in results if r.get("status") == "overexposed")}
    print(json.dumps(summary, indent=2))
    return 0 if summary["errors"] == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
