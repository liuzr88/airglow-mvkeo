"""argparse entrypoint."""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from .config import load_config
from .batch import run_year
from .processing import process_night

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "alo.toml"

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", required=True, type=Path, help="output root directory")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML config path")
    p.add_argument("--band", choices=["OH", "O5", "all"], default="all")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only", choices=["keogram", "raw", "diff", "all"], default="all")
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

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="airglow-mvkeo")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("run-night", help="process a single NetCDF file")
    p1.add_argument("file", type=Path)
    _add_common(p1)

    p2 = sub.add_parser("run-year", help="process every NC file in a year directory")
    p2.add_argument("year_dir", type=Path)
    p2.add_argument("--workers", type=int, default=None)
    _add_common(p2)

    args = p.parse_args(argv)
    _setup_logging(args.verbose, args.log_file)
    cfg = load_config(args.config)
    only = _only_set(args.only)

    if args.cmd == "run-night":
        result = process_night(args.file, args.out, cfg,
                               overwrite=args.overwrite, only=only, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
        return 0 if result.get("status") in ("ok", "dry_run", "skipped_existing") else 1

    results = run_year(args.year_dir, args.out, cfg,
                       band=args.band, workers=args.workers,
                       overwrite=args.overwrite, only=only, dry_run=args.dry_run)
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
