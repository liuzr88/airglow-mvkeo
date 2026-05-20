# airglow-mvkeo

Generate keograms, raw movies, and difference movies from per-night ALO airglow NetCDF files. Replaces the MATLAB `3_MovieKeo` pipeline.

## Requirements
- Python 3.11+
- `uv` (recommended) or pip
- Optional: system `ffmpeg` / `ffprobe`. If `ffmpeg` is not on PATH, the package uses `imageio-ffmpeg`.

## Install
```bash
cd airglow-mvkeo
uv venv && uv pip install -e .
```

## Usage

### Convert FITS to NetCDF
To scan all available years and all science bands, converting only missing or bad NetCDF files:
```bash
uv run airglow-convert-missing
```

This default command reads FITS folders from `~/OneDriveResearch/Data/ALOASI/<YYYY>/<YYYYMMDD>/`, writes to `~/OneDriveResearch/Data/ALOASI/NC/<YYYY>/`, and processes `OH O5 O6 O2 Na` with overwrite off.

For a specific year:
```bash
uv run airglow-fits-to-nc-batch 2025 \
  --overwrite 0
```

This reads nightly FITS folders from `<DATA_ROOT>/<YYYY>/<YYYYMMDD>/` and writes NetCDF files to `<DATA_ROOT>/NC/<YYYY>/`. Use `--overwrite 1` to rebuild existing NetCDF files.

For one channel only:
```bash
uv run airglow-fits-to-nc 2025 OH \
  --data-root ~/OneDriveResearch/Data/ALOASI \
  --nc-root ~/OneDriveResearch/Data/ALOASI/NC \
  --overwrite 0
```

### One night
```bash
uv run airglow-mvkeo run-night /path/OH20180101.nc --out ./out
```

### Create Missing Media
To scan all existing NetCDF year folders and create only missing keograms, raw movies, and TD movies in the standard `KG`/`MV` directories:
```bash
uv run airglow-create-missing-media
```

Existing media files are considered current only if their file timestamp is in the current calendar month. Missing files and older files are regenerated; files from this month are left untouched.

Useful limits:
```bash
uv run airglow-create-missing-media 2025 --band all --workers 4
uv run airglow-create-missing-media 2025 --band O6 --dry-run
```

### One date from the configured ALO data root
```bash
uv run airglow-mvkeo run-date 20231010 --band OH --style matlab
uv run airglow-mvkeo run-date 20231010 --band OH --style modern --clean-overlay
```

### A whole year
```bash
uv run airglow-mvkeo run-year /path/2018 --out ./out --workers 7
```

### Flags
- `--band OH|O5|O6|O2|Na|all` (default: all)
- `--style matlab|modern|web` (default: matlab) — `matlab` preserves validated legacy output; `modern`/`web` adds web-sized movies, reports, and duration-scaled keograms
- `--only keogram|raw|diff|wave|contact|report|all` (default: all) — regenerate just one artifact
- `--clean-overlay` — reduce movie overlay clutter for web embeds and presentations
- `--overwrite` — redo even if JSON sidecar exists
- `--no-resume-artifacts` — skip whole nights when sidecar exists; by default missing artifacts resume individually
- `--dry-run` — list planned outputs, render nothing
- `--config /path/to/other.toml` — non-default site/calibration

## Output layout
```
<DATA_ROOT>/KG/<YYYY>/
  {OH|O5|O6|O2|Na}Keog<YYYYMMDD>.jpg

<DATA_ROOT>/MV/<YYYY>/
  {OH|O5|O6|O2|Na}<YYYYMMDD>.mp4
  {OH|O5|O6|O2|Na}<YYYYMMDD>_TD.mp4
  {OH|O5|O6|O2|Na}<YYYYMMDD>.json
```

Modern/web style uses the same publishing names:
```
<DATA_ROOT>/KG/<YYYY>/
  {OH|O5|O6|O2|Na}Keog<YYYYMMDD>.jpg

<DATA_ROOT>/MV/<YYYY>/
  {OH|O5|O6|O2|Na}<YYYYMMDD>.mp4
  {OH|O5|O6|O2|Na}<YYYYMMDD>_TD.mp4
  {OH|O5|O6|O2|Na}<YYYYMMDD>.json
```

The optional `--only contact` and `--only report` products create a contact sheet and an HTML review page in `MV/<YYYY>/`. They are QA/convenience files and are not required for normal publishing.

## MATLAB Compatibility
Movie names match `CreateMovNC.m`: raw movies are named like `OH20231122.mp4`, and previous-frame difference movies are named like `OH20231122_TD.mp4`.

## Keograms
Keograms are simple centerline stacks from the raw image cube. The W-E panel uses the center image row, `frames[y0, :, :]`; the S-N panel uses the center image column, `frames[:, x0, :]`. No wave filtering, temporal smoothing, brightness curve, contouring, or artificial gap columns are applied to the sampled values. The renderer only maps those samples to grayscale display levels and draws the axes/labels.

Keogram widths are scaled by observing duration so horizontal pixels have a consistent time meaning from night to night, while the vertical size stays fixed across nights. In the default ALO config, a full 10-hour night is 1704 px wide by 850 px tall; shorter nights are proportionally narrower, with a minimum width for readability. Modern movie frames are sized to half the keogram width so raw and TD movies can sit side by side under a full-width keogram.

## Wave Enhancement
The previous-frame TD movie is the preferred web movie for wave structure. `--only wave` remains available in modern/web style as an experimental movie product, but the default keogram is intentionally raw-centerline rather than wave-enhanced.

## Smoke test
```bash
uv run airglow-mvkeo run-date 20231010 --band OH --style modern --out /tmp/smoke --verbose
```
Open the resulting files in a viewer. The optional report is named like `/tmp/smoke/2023/OH20231010_report.html`.

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
- `enhancement.py` — modern wave-enhancement and artifact cleanup helpers
- `difference.py` — previous-frame differencing for MATLAB-compatible TD movies
- `overlays.py` — PIL primitives (zenith ring, crosshair, labels, markers)
- `movie.py` — ffmpeg-piped H.264 writer (deadlock-safe, `yuv420p` for browsers)
- `keogram.py` — slice extraction, gap insertion, 2-panel matplotlib render
- `report.py` — contact sheet and HTML report output
- `fits_to_nc.py` — FITS night folders to MATLAB-compatible NetCDF files
- `fits_to_nc_batch.py` — multi-year, multi-channel FITS-to-NetCDF runner
- `media_missing.py` — create missing KG/MV products from existing NetCDF files
- `processing.py` — per-night driver gluing read → filter → render → JSON
- `batch.py` — multiprocessing year orchestrator
- `cli.py` — argparse entrypoint
