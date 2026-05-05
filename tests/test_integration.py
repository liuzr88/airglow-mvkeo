import json
import shutil
import subprocess
from datetime import date as _date
from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from dataclasses import replace
from airglow_mvkeo.config import load_config, CalibrationEntry, ImageConfig, KeogramConfig
from airglow_mvkeo.processing import process_night

CFG_PATH = Path(__file__).parent.parent / "config" / "alo.toml"
ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(not (ffmpeg and ffprobe), reason="ffmpeg/ffprobe not on PATH")

def _make_synthetic_nc(path: Path, n_frames: int = 30, h: int = 64, w: int = 64):
    rng = np.random.default_rng(0)
    intensity = rng.normal(2000, 150, size=(h, w, n_frames)).astype(np.float32)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for i in range(n_frames):
        cx, cy = w/2 + (i - n_frames/2) * 0.5, h/2
        blob = 800.0 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 4**2))
        intensity[..., i] += blob
    # MATLAB datenum 737061 corresponds to 2018-01-01 (verify in test).
    base = 737061.0 + 3/24
    time = base + np.arange(n_frames) / (24 * 60)
    ds = xr.Dataset({
        "intensity": (("y", "x", "t"), intensity),
        "time": (("t",), time),
    })
    ds.to_netcdf(path)

def test_process_night_synthetic_end_to_end(tmp_path):
    nc = tmp_path / "OH20180101.nc"
    _make_synthetic_nc(nc)
    cfg = load_config(CFG_PATH)

    # Override image size, calibration, and keogram tick range for the synthetic
    # 64x64 image. Default ticks at +-800 km don't make sense at this scale.
    small_ticks_km = (-50.0, -20.0, 0.0, 20.0, 50.0)
    small_tick_labels = ("-50", "-20", "0", "+20", "50")
    cfg = replace(cfg,
                  image=ImageConfig(size_px=64, fov_deg=180),
                  calibration=[CalibrationEntry(_date(2018, 1, 1), 32, 32, 30, 1.6)],
                  keogram=replace(cfg.keogram,
                                  distance_ticks_km=small_ticks_km,
                                  distance_tick_labels=small_tick_labels))

    out_dir = tmp_path / "out"
    result = process_night(nc, out_dir, cfg)

    assert result["status"] == "ok"
    assert result["n_frames_used"] == 30
    assert result["calibration"]["x0"] == 32

    # All three output files exist at the expected paths
    p = out_dir / "2018" / "01"
    assert (p / "OHKeog20180101.jpg").exists()
    assert (p / "OHOrig20180101.mp4").exists()
    assert (p / "OHDiff20180101.mp4").exists()

    sidecar = json.loads((p / "OH20180101.json").read_text())
    assert sidecar["band"] == "OH"
    assert sidecar["status"] == "ok"
    assert sidecar["files"]["keogram"] == "OHKeog20180101.jpg"
    assert sidecar["files"]["movie_raw"] == "OHOrig20180101.mp4"
    assert sidecar["files"]["movie_diff"] == "OHDiff20180101.mp4"

    # Verify mp4 duration: 30 frames @ 30 fps ~= 1 sec
    r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(p / "OHOrig20180101.mp4")],
                       capture_output=True, text=True, check=True)
    duration = float(r.stdout.strip())
    assert 0.8 < duration < 2.0, f"raw movie duration {duration} outside expected range"


def test_idempotent_skip_when_sidecar_exists(tmp_path):
    """Second run without --overwrite should skip and return status='skipped_existing'."""
    nc = tmp_path / "OH20180101.nc"
    _make_synthetic_nc(nc)
    cfg = load_config(CFG_PATH)
    cfg = replace(cfg,
                  image=ImageConfig(size_px=64, fov_deg=180),
                  calibration=[CalibrationEntry(_date(2018, 1, 1), 32, 32, 30, 1.6)],
                  keogram=replace(cfg.keogram,
                                  distance_ticks_km=(-50.0, 0.0, 50.0),
                                  distance_tick_labels=("-50", "0", "50")))
    out_dir = tmp_path / "out"
    first = process_night(nc, out_dir, cfg)
    assert first["status"] == "ok"
    second = process_night(nc, out_dir, cfg, overwrite=False)
    assert second["status"] == "skipped_existing"
    third = process_night(nc, out_dir, cfg, overwrite=True)
    assert third["status"] == "ok"


def test_dry_run_does_not_write_files(tmp_path):
    nc = tmp_path / "O520180101.nc"
    _make_synthetic_nc(nc)
    cfg = load_config(CFG_PATH)
    cfg = replace(cfg,
                  image=ImageConfig(size_px=64, fov_deg=180),
                  calibration=[CalibrationEntry(_date(2018, 1, 1), 32, 32, 30, 1.6)],
                  keogram=replace(cfg.keogram,
                                  distance_ticks_km=(-50.0, 0.0, 50.0),
                                  distance_tick_labels=("-50", "0", "50")))
    out_dir = tmp_path / "out"
    result = process_night(nc, out_dir, cfg, dry_run=True)
    assert result["status"] == "dry_run"
    assert "files_planned" in result
    # Nothing should have been written
    assert not (out_dir / "2018" / "01" / "O5Keog20180101.jpg").exists()
    assert not (out_dir / "2018" / "01" / "O520180101.json").exists()
