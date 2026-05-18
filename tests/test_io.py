import json
from datetime import datetime, date, timezone
from pathlib import Path
import numpy as np
import xarray as xr
from airglow_mvkeo.io import (
    read_night, write_json_sidecar, output_paths,
    matlab_datenum_to_datetime,
)

def test_datenum_known_value():
    # MATLAB datenum 737791 -> 2020-01-01
    dt = matlab_datenum_to_datetime(737791.0)
    assert dt.year == 2020 and dt.month == 1 and dt.day == 1

def test_datenum_array():
    arr = np.array([737791.0, 737791.5])
    dts = matlab_datenum_to_datetime(arr)
    assert dts[0].day == 1 and dts[1].hour == 12

def test_read_night_minimal_nc(tmp_path):
    n = 5
    # Real schema: intensity has dims (time, y, x) — shape (N, Y, X)
    intensity = np.arange(n * 3 * 4, dtype=np.float32).reshape(n, 3, 4)
    # time values are days-from-Jan-1 (0.0 = Jan 1 00:00 UTC)
    time = np.array([i / (24 * 60) for i in range(n)])
    ds = xr.Dataset({
        "intensity": (("time", "y", "x"), intensity),
        "time": (("time",), time),
        "year": ((), np.uint16(2020)),
    })
    p = tmp_path / "OH20200101.nc"
    ds.to_netcdf(p)
    night = read_night(p)
    # read_night returns an image cube indexed as (row, column, time).
    assert night.intensity.shape == (4, 3, 5)
    assert np.array_equal(night.intensity[:, :, 0], intensity[0].T)
    assert night.times[0].year == 2020
    assert night.band == "OH"
    assert night.date == date(2020, 1, 1)

def test_output_paths_layout(tmp_path):
    p = output_paths(tmp_path, band="OH", d=date(2018, 1, 1))
    assert p.dir == tmp_path / "2018"
    assert p.keogram.name == "OHKeog20180101.jpg"
    assert p.movie_raw.name == "OH20180101.mp4"
    assert p.movie_diff.name == "OH20180101_TD.mp4"
    assert p.json.name == "OH20180101.json"

def test_output_paths_can_split_keogram_root(tmp_path):
    mv = tmp_path / "MV"
    kg = tmp_path / "KG"
    p = output_paths(mv, band="OH", d=date(2018, 1, 1),
                     style="modern", keogram_out_dir=kg)
    assert p.keogram == kg / "2018" / "OHKeog20180101.jpg"
    assert p.movie_raw == mv / "2018" / "OH20180101.mp4"
    assert p.movie_diff == mv / "2018" / "OH20180101_TD.mp4"
    assert p.report == mv / "2018" / "OH20180101_report.html"

def test_write_json_sidecar(tmp_path):
    p = tmp_path / "x.json"
    write_json_sidecar(p, {
        "date": "2018-01-01", "band": "OH", "status": "ok",
        "n_frames_total": 10, "n_frames_used": 8,
    })
    data = json.loads(p.read_text())
    assert data["status"] == "ok" and data["n_frames_used"] == 8
