# airglow-mvkeo

Generate keograms, raw movies, and difference movies from per-night ALO airglow NetCDF files. Replaces the MATLAB `3_MovieKeo` pipeline.

## Requirements
- Python 3.11+
- `ffmpeg` and `ffprobe` on PATH
- `uv` (recommended) or pip

## Install
```bash
cd airglow-mvkeo
uv venv && uv pip install -e .
```

## Usage

### One night
```bash
uv run airglow-mvkeo run-night /path/OH20180101.nc --out ./out
```

### A whole year
```bash
uv run airglow-mvkeo run-year /path/2018 --out ./out --workers 7
```

### Flags
- `--band OH|O5|all` (default: all)
- `--only keogram|raw|diff|all` (default: all) — regenerate just one artifact
- `--overwrite` — redo even if JSON sidecar exists
- `--dry-run` — list planned outputs, render nothing
- `--config /path/to/other.toml` — non-default site/calibration

## Output layout
```
<OUT_DIR>/<YYYY>/<MM>/
  {OH|O5}Keog<YYYYMMDD>.jpg
  {OH|O5}Orig<YYYYMMDD>.mp4
  {OH|O5}Diff<YYYYMMDD>.mp4
  {OH|O5}<YYYYMMDD>.json
```

## Naming change vs. MATLAB
MATLAB output uses a `(v2)` suffix (e.g. `OHKeog20180101(v2).jpg`); this pipeline drops it. The website ingestion path must be updated in lockstep.

## Smoke test
```bash
uv run airglow-mvkeo run-night /path/to/OH20180101.nc --out /tmp/smoke --verbose
```
Open the resulting JPG and MP4s in a viewer to verify visual quality. On macOS: `open /tmp/smoke/2018/01/OHKeog20180101.jpg`. On Linux: `xdg-open ...`.

## Tests
```bash
uv run pytest -q              # unit + integration
uv run pytest -m visual       # opt-in visual regression (requires fixtures)
```

## Architecture
- `config.py` — TOML config (site, bands, calibration table, render settings)
- `io.py` — NetCDF read (with MATLAB datenum decoding), JSON sidecar write
- `geometry.py` — pixel ↔ km mapping at airglow altitude, calibration lookup
- `intensity.py` — over-exposure filter, percentile color range, fps selection
- `difference.py` — running-mean detrending for wave enhancement
- `overlays.py` — PIL primitives (zenith ring, crosshair, labels, markers)
- `movie.py` — ffmpeg-piped H.264 writer (deadlock-safe, `yuv420p` for browsers)
- `keogram.py` — slice extraction, gap insertion, 2-panel matplotlib render
- `processing.py` — per-night driver gluing read → filter → render → JSON
- `batch.py` — multiprocessing year orchestrator
- `cli.py` — argparse entrypoint
