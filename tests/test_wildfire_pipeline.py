import json
import sys
import types
from datetime import date

import numpy as np
import pandas as pd
import pytest

from wildfire.features.build import build_feature_table, assert_time_split_no_leakage
from wildfire.ingest.era5 import calculate_vpd_kpa, request_era5_monthly
from wildfire.ingest.firms import load_firms_monthly
from wildfire.ingest.grace import download_grace
from wildfire.ingest.mcd64a1 import aggregate_burned_area_table, download_mcd64a1
from wildfire.models.backtest import run_regression_backtest
from wildfire.regions import assign_grid_region, normalize_longitude
import wildfire_downloader_streamlit as wildfire_app


def test_normalize_longitude_and_region_assignment():
    assert normalize_longitude(250) == pytest.approx(-110)
    region = assign_grid_region(34.2, -118.4, resolution=1.0)
    assert region["region_id"] == "lat34.00_lon-119.00"
    assert region["region_lat_center"] == pytest.approx(34.5)


def test_load_firms_monthly_filters_and_aggregates(tmp_path):
    csv_path = tmp_path / "firms.csv"
    pd.DataFrame(
        [
            {
                "latitude": 34.1,
                "longitude": -118.2,
                "acq_date": "2026-07-01",
                "brightness": 330,
                "confidence": "h",
            },
            {
                "latitude": 34.4,
                "longitude": -118.6,
                "acq_date": "2026-07-02",
                "brightness": 340,
                "confidence": "n",
            },
            {
                "latitude": -20.0,
                "longitude": 130.0,
                "acq_date": "2026-07-03",
                "brightness": 300,
                "confidence": "h",
            },
        ]
    ).to_csv(csv_path, index=False)

    monthly = load_firms_monthly(csv_path, region_resolution=1.0)

    assert len(monthly) == 1
    assert monthly.loc[0, "fire_count"] == 2
    assert monthly.loc[0, "year"] == 2026
    assert monthly.loc[0, "month"] == 7
    assert monthly.loc[0, "brightness_mean"] == pytest.approx(335)
    assert monthly.loc[0, "confidence_mean"] == pytest.approx(75)


def test_calculate_vpd_kpa_positive_and_zero_when_saturated():
    t2m = np.array([303.15, 293.15])
    d2m = np.array([293.15, 293.15])
    vpd = calculate_vpd_kpa(t2m, d2m)
    assert vpd[0] > 0
    assert vpd[1] == pytest.approx(0)


def test_burned_area_table_aggregates_monthly():
    pixels = pd.DataFrame(
        [
            {"latitude": 40.1, "longitude": -120.1, "date": "2026-08-01", "burned_area_ha": 12.5},
            {"latitude": 40.2, "longitude": -120.2, "date": "2026-08-02", "burned_area_ha": 7.5},
        ]
    )
    monthly = aggregate_burned_area_table(pixels, region_resolution=1.0)
    assert len(monthly) == 1
    assert monthly.loc[0, "burned_area_ha"] == pytest.approx(20.0)


def test_build_feature_table_adds_lags_and_no_leakage_check():
    fire = pd.DataFrame(
        [
            {"region_id": "r1", "year": 2025, "month": 11, "fire_count": 2, "region_lat_center": 40, "region_lon_center": -120},
            {"region_id": "r1", "year": 2025, "month": 12, "fire_count": 3, "region_lat_center": 40, "region_lon_center": -120},
            {"region_id": "r1", "year": 2026, "month": 1, "fire_count": 8, "region_lat_center": 40, "region_lon_center": -120},
        ]
    )
    grace = pd.DataFrame(
        [
            {"region_id": "r1", "year": 2025, "month": 11, "gws_inst_mean": 45, "rtzsm_inst_mean": 50},
            {"region_id": "r1", "year": 2025, "month": 12, "gws_inst_mean": 35, "rtzsm_inst_mean": 40},
            {"region_id": "r1", "year": 2026, "month": 1, "gws_inst_mean": 25, "rtzsm_inst_mean": 30},
        ]
    )
    era5 = pd.DataFrame(
        [
            {"region_id": "r1", "year": 2025, "month": 11, "vpd_mean": 1.2, "vpd_max": 2.0},
            {"region_id": "r1", "year": 2025, "month": 12, "vpd_mean": 1.4, "vpd_max": 2.2},
            {"region_id": "r1", "year": 2026, "month": 1, "vpd_mean": 1.8, "vpd_max": 2.8},
        ]
    )

    table = build_feature_table(fire, grace_monthly=grace, era5_monthly=era5)

    assert "gws_inst_mean_lag1" in table.columns
    assert table.loc[2, "fire_count_lag1"] == 3
    assert "severe_fire_month" in table.columns
    assert_time_split_no_leakage(table, 2026, ["gws_inst_mean_lag1", "vpd_mean_lag1"])


def test_regression_backtest_writes_outputs(tmp_path):
    pytest.importorskip("sklearn", exc_type=ImportError)
    rows = []
    for year in [2024, 2025, 2026]:
        for month in range(1, 5):
            rows.append(
                {
                    "region_id": "r1",
                    "year": year,
                    "month": month,
                    "region_lat_center": 40.5,
                    "region_lon_center": -120.5,
                    "fire_count": month + (year - 2024),
                    "vpd_mean_lag1": float(month),
                    "gws_inst_mean_lag1": float(50 - month),
                }
            )
    table = pd.DataFrame(rows)

    metrics = run_regression_backtest(table, tmp_path, backtest_year=2026)

    assert metrics["train_rows"] == 8
    assert metrics["test_rows"] == 4
    with open(metrics["metrics_json"], encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["backtest_year"] == 2026


def test_firms_download_builds_chunked_area_api_urls(monkeypatch, tmp_path):
    urls = []

    class DummyProgress:
        def progress(self, _value):
            return None

    def fake_read_csv(url):
        urls.append(url)
        return pd.DataFrame(
            [
                {
                    "latitude": 36.2837,
                    "longitude": -91.9276,
                    "brightness": 305.9,
                    "acq_date": "2026-01-01",
                    "confidence": 62,
                }
            ]
        )

    monkeypatch.setattr(wildfire_app.st, "progress", lambda _value: DummyProgress())
    monkeypatch.setattr(wildfire_app.st, "write", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wildfire_app.pd, "read_csv", fake_read_csv)

    output_csv = tmp_path / "firms.csv"
    result = wildfire_app._download_firms_area(
        map_key="KEY",
        source="MODIS_SP",
        bbox=(-130.0, 17.0, -74.0, 50.0),
        start=date(2026, 1, 1),
        end=date(2026, 1, 6),
        output_csv=output_csv,
    )

    assert len(urls) == 2
    assert urls[0].endswith("/KEY/MODIS_SP/-130.0,17.0,-74.0,50.0/5/2026-01-01")
    assert urls[1].endswith("/KEY/MODIS_SP/-130.0,17.0,-74.0,50.0/1/2026-01-06")
    assert len(result) == 1
    assert output_csv.is_file()


def test_grace_download_uses_earthaccess_search_and_download(monkeypatch, tmp_path):
    calls = {}

    fake_earthaccess = types.SimpleNamespace()
    fake_earthaccess.login = lambda: calls.setdefault("login", True)

    def fake_search_data(**kwargs):
        calls["search"] = kwargs
        return ["granule"]

    def fake_download(results, local_path):
        calls["download"] = (results, local_path)
        return [str(tmp_path / "grace.nc4")]

    fake_earthaccess.search_data = fake_search_data
    fake_earthaccess.download = fake_download
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

    files = download_grace(tmp_path, temporal=("2025-01-01", "2025-01-08"))

    assert calls["login"] is True
    assert calls["search"]["short_name"] == "GRACEDADM_CLSM025GL_7D"
    assert calls["search"]["version"] == "3.0"
    assert calls["search"]["temporal"] == ("2025-01-01", "2025-01-08")
    assert calls["download"] == (["granule"], str(tmp_path))
    assert files == [str(tmp_path / "grace.nc4")]


def test_mcd64a1_download_uses_earthaccess_search_and_download(monkeypatch, tmp_path):
    calls = {}

    fake_earthaccess = types.SimpleNamespace()
    fake_earthaccess.login = lambda: calls.setdefault("login", True)

    def fake_search_data(**kwargs):
        calls["search"] = kwargs
        return ["mcd_granule"]

    def fake_download(results, local_path):
        calls["download"] = (results, local_path)
        return [str(tmp_path / "mcd64a1.hdf")]

    fake_earthaccess.search_data = fake_search_data
    fake_earthaccess.download = fake_download
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

    files = download_mcd64a1(tmp_path, temporal=("2025-01-01", "2025-01-31"))

    assert calls["login"] is True
    assert calls["search"]["short_name"] == "MCD64A1"
    assert calls["search"]["temporal"] == ("2025-01-01", "2025-01-31")
    assert calls["download"] == (["mcd_granule"], str(tmp_path))
    assert files == [str(tmp_path / "mcd64a1.hdf")]


def test_era5_download_builds_cds_request(monkeypatch, tmp_path):
    calls = {}

    class FakeClient:
        def retrieve(self, dataset, request, target):
            calls["dataset"] = dataset
            calls["request"] = request
            calls["target"] = target

    monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=FakeClient))

    output_path = tmp_path / "era5.nc"
    result = request_era5_monthly(
        output_path,
        years=[2025, 2026],
        months=[1, 7],
        area=(50.0, -130.0, 17.0, -74.0),
    )

    assert result == str(output_path)
    assert calls["dataset"] == "reanalysis-era5-single-levels-monthly-means"
    assert calls["target"] == str(output_path)
    assert calls["request"]["product_type"] == "monthly_averaged_reanalysis"
    assert calls["request"]["variable"] == ["2m_temperature", "2m_dewpoint_temperature"]
    assert calls["request"]["year"] == ["2025", "2026"]
    assert calls["request"]["month"] == ["01", "07"]
    assert calls["request"]["area"] == [50.0, -130.0, 17.0, -74.0]


def test_feature_input_discovery_prefers_download_folders(tmp_path):
    raw = tmp_path / "data" / "raw"
    firms_dir = raw / "firms"
    grace_dir = raw / "grace"
    era5_dir = raw / "era5"
    mcd_dir = raw / "mcd64a1"
    for directory in [firms_dir, grace_dir, era5_dir, mcd_dir]:
        directory.mkdir(parents=True)

    firms = firms_dir / "firms_north_america.csv"
    grace = grace_dir / "GRACEDADM_CLSM025GL_7D.A20250101.030.nc4"
    era5 = era5_dir / "era5_t2m_d2m_2025_01_01.nc"
    burned = mcd_dir / "mcd64a1_burned_area_pixels.csv"
    for path in [firms, grace, era5, burned]:
        path.write_text("x", encoding="utf-8")

    discovered = wildfire_app._discover_feature_inputs(raw)

    assert discovered["firms_csv"] == str(firms)
    assert discovered["grace_files"] == [str(grace)]
    assert discovered["era5_files"] == [str(era5)]
    assert discovered["burned_area_csv"] == str(burned)


def test_download_ids_include_bbox_and_are_stable():
    bbox = (-130.9375, 17.817045, -74.375, 50.797242)
    first = wildfire_app._download_id(
        "firms",
        bbox,
        date(2026, 1, 1),
        date(2026, 1, 5),
        extra="MODIS_SP",
    )
    second = wildfire_app._download_id(
        "firms",
        bbox,
        date(2026, 1, 1),
        date(2026, 1, 5),
        extra="MODIS_SP",
    )
    changed_bbox = wildfire_app._download_id(
        "firms",
        (-78.60, 37.95, -78.35, 38.15),
        date(2026, 1, 1),
        date(2026, 1, 5),
        extra="MODIS_SP",
    )

    assert first == second
    assert first != changed_bbox
    assert "wm130p9375" in first
    assert "s17p8170" in first


def test_manifest_matching_detects_existing_file(tmp_path):
    output = tmp_path / "firms_id.csv"
    output.write_text("latitude,longitude,acq_date\n", encoding="utf-8")
    wildfire_app._write_manifest(output, {"download_id": "abc123", "product": "FIRMS"})

    assert wildfire_app._existing_matching_file(output, "abc123")
    assert not wildfire_app._existing_matching_file(output, "different")


def test_firms_download_reuses_matching_manifest(monkeypatch, tmp_path):
    output = tmp_path / "firms.csv"
    output.write_text("latitude,longitude,acq_date\n", encoding="utf-8")
    bbox = (-78.60, 37.95, -78.35, 38.15)
    download_id = wildfire_app._download_id(
        "firms",
        bbox,
        date(2026, 1, 1),
        date(2026, 1, 5),
        extra="MODIS_SP",
    )
    wildfire_app._write_manifest(output, {"download_id": download_id})
    original_read_csv = pd.read_csv

    def fail_read_url(value):
        if str(value).startswith("https://"):
            raise AssertionError("FIRMS API should not be called for matching existing file")
        return original_read_csv(value)

    monkeypatch.setattr(wildfire_app.st, "info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wildfire_app.pd, "read_csv", fail_read_url)

    df = wildfire_app._download_firms_area(
        map_key="KEY",
        source="MODIS_SP",
        bbox=bbox,
        start=date(2026, 1, 1),
        end=date(2026, 1, 5),
        output_csv=output,
    )

    assert list(df.columns) == ["latitude", "longitude", "acq_date"]


def test_prepare_firms_map_points_filters_and_limits(tmp_path):
    csv_path = tmp_path / "firms_map.csv"
    pd.DataFrame(
        [
            {"latitude": 38.03, "longitude": -78.48, "acq_date": "2026-01-01", "confidence": 20, "brightness": 300},
            {"latitude": 38.04, "longitude": -78.47, "acq_date": "2026-01-02", "confidence": 80, "brightness": 320},
            {"latitude": 38.05, "longitude": -78.46, "acq_date": "2026-01-03", "confidence": 90, "brightness": 330},
        ]
    ).to_csv(csv_path, index=False)

    points = wildfire_app._prepare_firms_map_points(
        csv_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 3),
        min_confidence=50,
        max_points=1,
    )

    assert len(points) == 1
    assert points.loc[0, "confidence"] == 90
    assert points.loc[0, "acq_date"].date() == date(2026, 1, 3)
