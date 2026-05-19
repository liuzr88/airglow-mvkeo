from datetime import date
from pathlib import Path
import pytest
from airglow_mvkeo.config import load_config, Config

CFG = Path(__file__).parent.parent / "config" / "alo.toml"

def test_load_alo_config():
    cfg = load_config(CFG)
    assert cfg.site.name == "ALO"
    assert cfg.bands["OH"].altitude_km == 87
    assert cfg.bands["O5"].altitude_km == 96
    assert cfg.bands["O6"].altitude_km == 250
    assert cfg.bands["O2"].altitude_km == 94
    assert cfg.bands["Na"].altitude_km == 90
    assert cfg.image.size_px == 512

def test_calibration_sorted_ascending():
    cfg = load_config(CFG)
    dates = [c.since for c in cfg.calibration]
    assert dates == sorted(dates)

def test_calibration_non_empty():
    cfg = load_config(CFG)
    assert len(cfg.calibration) >= 1

def test_filter_thresholds_present():
    cfg = load_config(CFG)
    assert cfg.filter.overexposed_mean_threshold == 1e4
    assert cfg.filter.low_intensity_threshold == 5e3
    assert cfg.filter.color_range_percentiles == (10, 80)

def test_movie_pix_fmt_yuv420p():
    cfg = load_config(CFG)
    assert cfg.movie.pix_fmt == "yuv420p"
    assert cfg.movie.web_output_size_px == cfg.keogram.full_night_width_px // 2
    assert cfg.keogram.full_night_width_px == 1704
    assert cfg.keogram.height_px == 850

def test_invalid_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")
