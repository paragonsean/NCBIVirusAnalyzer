from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import wildfire_downloader_streamlit as app


CHARLOTTESVILLE_BBOX = (-78.60, 37.95, -78.35, 38.15)  # west, south, east, north
CHARLOTTESVILLE_CDS_AREA = (38.15, -78.60, 37.95, -78.35)  # north, west, south, east


def _require_live_network(monkeypatch):
    import os

    if os.environ.get("RUN_NETWORK_TESTS") != "1":
        pytest.skip("Set RUN_NETWORK_TESTS=1 to run live download tests.")


def _read_env_value(name: str) -> str:
    env_path = Path(".env")
    if not env_path.is_file():
        return ""
    prefix = f"{name}="
    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.startswith(prefix):
            return text[len(prefix):].strip().strip("'\"")
    return ""


@pytest.mark.network
def test_live_firms_download_charlottesville(monkeypatch, tmp_path):
    _require_live_network(monkeypatch)
    map_key = _read_env_value("FIRMS_MAP_KEY")
    if not map_key:
        pytest.skip("FIRMS_MAP_KEY is not set in .env.")

    class DummyProgress:
        def progress(self, _value):
            return None

    monkeypatch.setattr(app.st, "progress", lambda _value: DummyProgress())
    monkeypatch.setattr(app.st, "write", lambda *_args, **_kwargs: None)

    out_csv = tmp_path / "firms_charlottesville.csv"
    df = app._download_firms_area(
        map_key=map_key,
        source="MODIS_SP",
        bbox=CHARLOTTESVILLE_BBOX,
        start=date(2026, 1, 1),
        end=date(2026, 1, 5),
        output_csv=out_csv,
    )

    assert out_csv.is_file()
    saved = pd.read_csv(out_csv)
    assert list(saved.columns) == list(df.columns)
    assert {"latitude", "longitude", "acq_date"}.issubset(saved.columns)


@pytest.mark.network
def test_live_grace_download_charlottesville(monkeypatch, tmp_path):
    _require_live_network(monkeypatch)
    files = app._search_download_earthaccess(
        short_name="GRACEDADM_CLSM025GL_7D",
        version="3.0",
        temporal=("2025-01-01", "2025-01-08"),
        bbox=CHARLOTTESVILLE_BBOX,
        output_dir=tmp_path,
        max_results=1,
    )

    assert len(files) == 1
    path = Path(files[0])
    assert path.is_file()
    assert path.stat().st_size > 0
    assert path.suffix in {".nc4", ".nc"}


@pytest.mark.network
def test_live_mcd64a1_download_charlottesville(monkeypatch, tmp_path):
    _require_live_network(monkeypatch)
    files = app._search_download_earthaccess(
        short_name="MCD64A1",
        temporal=("2025-01-01", "2025-01-31"),
        bbox=CHARLOTTESVILLE_BBOX,
        output_dir=tmp_path,
        max_results=1,
    )

    assert len(files) == 1
    path = Path(files[0])
    assert path.is_file()
    assert path.stat().st_size > 0


@pytest.mark.network
def test_live_era5_download_charlottesville(monkeypatch, tmp_path):
    _require_live_network(monkeypatch)
    pytest.importorskip("cdsapi")
    from wildfire.ingest.era5 import request_era5_monthly

    out_nc = tmp_path / "era5_charlottesville_2025_01.nc"
    result = request_era5_monthly(
        out_nc,
        years=[2025],
        months=[1],
        area=CHARLOTTESVILLE_CDS_AREA,
    )

    path = Path(result)
    assert path.is_file()
    assert path.stat().st_size > 0
