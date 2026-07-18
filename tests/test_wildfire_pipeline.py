import json
import sys
import types
from datetime import date

import numpy as np
import pandas as pd
import pytest

from wildfire.features.build import build_feature_table, assert_time_split_no_leakage
from wildfire.features.fire_growth import (
    attach_days_since_ignition,
    attach_frp_metrics,
    attach_spread_rate_metrics,
    build_growth_grid,
    classify_fire_growth,
)
from wildfire.features.fire_spread import (
    build_event_snapshot,
    build_event_spread_summary,
    build_event_timeline,
    cluster_firms_events,
)
from wildfire.ui.map_viewer import _extract_selected_event_id
from wildfire.features.link_fires_burned import (
    aggregate_firms_burned_links,
    link_firms_to_burned,
)
from wildfire.ui.common import ProgressTracker, resolve_local_cache, write_manifest
from wildfire.ui.downloads import combine_similar_firms_points
from wildfire.ingest.era5 import calculate_vpd_kpa, request_era5_monthly
from wildfire.ingest.firms import load_firms_monthly
from wildfire.ingest.grace import download_grace
from wildfire.ingest.mcd64a1 import (
    _vectorize_burn_array,
    acquisition_date_from_filename,
    aggregate_burned_area_table,
    burned_area_polygons_path,
    download_mcd64a1,
    tile_hv_from_filename,
)
from wildfire.models.backtest import run_regression_backtest
from wildfire.regions import assign_grid_region, normalize_longitude
from wildfire.ui.map_viewer import (
    FirmsMapViewer,
    available_fire_dates,
    available_timeline_dates,
    filter_burned_area_for_frame,
    filter_points_for_frame,
    prepare_burned_area_map_points,
    prepare_burned_area_map_polygons,
    prepare_event_map_layers,
)
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


def test_mcd64a1_acquisition_date_from_filename():
    parsed = acquisition_date_from_filename("MCD64A1.A2025001.h11v04.061.test.hdf")
    assert parsed is not None
    assert parsed.date() == date(2025, 1, 1)


def test_mcd64a1_tile_hv_from_filename():
    assert tile_hv_from_filename("MCD64A1.A2026001.h10v04.061.hdf") == (10, 4)
    assert tile_hv_from_filename("no_tile_here.hdf") is None


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
        def progress(self, _value, text=None):
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

    monkeypatch.setattr(wildfire_app.st, "progress", lambda *_args, **_kwargs: DummyProgress())
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


def test_resolve_local_cache_reuses_covering_date_range(tmp_path):
    bbox = (-78.60, 37.95, -78.35, 38.15)
    wide = tmp_path / "firms_wide.csv"
    wide.write_text("latitude,longitude,acq_date\n1,2,2026-01-01\n", encoding="utf-8")
    write_manifest(
        wide,
        {
            "download_id": "wide_id",
            "product": "FIRMS",
            "source": "MODIS_SP",
            "bbox": list(bbox),
            "start": "2026-01-01",
            "end": "2026-12-31",
        },
    )
    preferred = tmp_path / "firms_narrow.csv"
    found = resolve_local_cache(
        preferred,
        "narrow_id",
        product="FIRMS",
        bbox=bbox,
        start=date(2026, 3, 1),
        end=date(2026, 3, 31),
        required_fields={"source": "MODIS_SP"},
    )
    assert found == wide


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

    unlimited = wildfire_app._prepare_firms_map_points(
        csv_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 3),
        min_confidence=50,
        max_points=None,
    )
    points = wildfire_app._prepare_firms_map_points(
        csv_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 3),
        min_confidence=50,
        max_points=1,
    )

    assert len(unlimited) == 2
    assert len(points) == 1
    assert points.loc[0, "confidence"] == 90
    assert points.loc[0, "acq_date"].date() == date(2026, 1, 3)


def test_timelapse_helpers_filter_frame_window():
    points = pd.DataFrame(
        {
            "latitude": [38.0, 38.1, 38.2],
            "longitude": [-78.5, -78.4, -78.3],
            "acq_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"]),
        }
    )

    dates = available_fire_dates(points)
    frame = filter_points_for_frame(points, date(2026, 1, 4), trailing_days=3)

    assert dates == [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 4)]
    assert len(frame) == 3
    assert frame["acq_date"].dt.date.tolist() == [
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 4),
    ]
    assert frame["age_days"].tolist() == [3, 2, 0]
    assert FirmsMapViewer._pydeck_age_color(0) == [255, 0, 0, 210]
    assert FirmsMapViewer._pydeck_age_color(2) == [40, 120, 255, 190]
    assert FirmsMapViewer._pydeck_age_color(3) == [255, 255, 255, 170]


def test_prepare_burned_area_map_points_and_frame_filter(tmp_path):
    csv_path = tmp_path / "burned_area.csv"
    pd.DataFrame(
        [
            {"latitude": 38.0, "longitude": -78.5, "date": "2026-01-01", "burned_area_ha": 10},
            {"latitude": 38.1, "longitude": -78.4, "date": "2026-01-03", "burned_area_ha": 25},
            {"latitude": 38.2, "longitude": -78.3, "date": "2026-01-04", "burned_area_ha": 0},
        ]
    ).to_csv(csv_path, index=False)

    points = prepare_burned_area_map_points(
        csv_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 4),
    )
    frame = filter_burned_area_for_frame(points, date(2026, 1, 3), trailing_days=2)

    assert len(points) == 2
    assert points["burned_area_ha"].sum() == 35
    assert "burned_area_radius" in points.columns
    assert len(frame) == 2
    assert frame["age_days"].tolist() == [2, 0]
    assert "burned_area_color" in frame.columns


def test_vectorize_burn_array_dissolves_adjacent_same_day_pixels():
    from datetime import datetime

    from rasterio.transform import from_origin

    burn = np.zeros((6, 6), dtype=np.int32)
    burn[2:4, 2:5] = 40  # connected same-day scar
    burn[5, 5] = 41  # separate day
    transform = from_origin(-79.0, 39.0, 0.01, 0.01)
    rows = _vectorize_burn_array(
        burn,
        transform,
        datetime(2026, 1, 1),
        "synthetic.hdf",
        project_from_sinu=False,
        bbox=(-80.0, 38.0, -78.0, 40.0),
    )
    assert len(rows) >= 2
    assert all("polygon" in row and len(row["polygon"][0]) >= 4 for row in rows)
    assert sum(row["burned_area_ha"] for row in rows) > 0
    dates = {row["date"] for row in rows}
    assert "2026-02-09" in dates  # day-of-year 40
    assert "2026-02-10" in dates


def test_fire_growth_frp_and_grid_separate_small_from_large():
    dets = pd.DataFrame(
        [
            {"event_id": "small", "latitude": 38.0, "longitude": -78.5, "acq_date": "2026-07-01", "frp": 8.0},
            {"event_id": "big", "latitude": 40.0, "longitude": -120.0, "acq_date": "2026-07-01", "frp": 40.0},
            {"event_id": "big", "latitude": 40.01, "longitude": -120.01, "acq_date": "2026-07-01", "frp": 55.0},
            {"event_id": "big", "latitude": 40.02, "longitude": -120.02, "acq_date": "2026-07-02", "frp": 120.0},
            {"event_id": "big", "latitude": 40.03, "longitude": -120.03, "acq_date": "2026-07-02", "frp": 150.0},
            {"event_id": "big", "latitude": 40.04, "longitude": -120.04, "acq_date": "2026-07-03", "frp": 200.0},
        ]
    )
    events = pd.DataFrame(
        [
            {"event_id": "small", "detection_count": 1, "duration_days": 1, "footprint_ha": 176.7},
            {"event_id": "big", "detection_count": 5, "duration_days": 3, "footprint_ha": 8000.0},
        ]
    )
    enriched = attach_frp_metrics(events, dets)
    assert enriched.loc[enriched.event_id == "small", "growth_class"].iloc[0] == "small / local"
    assert enriched.loc[enriched.event_id == "big", "frp_trend_mw"].iloc[0] > 0
    assert "large" in enriched.loc[enriched.event_id == "big", "growth_class"].iloc[0]
    big = enriched.loc[enriched.event_id == "big"].iloc[0]
    assert bool(big["is_spreading"]) is True
    assert big["spread_km_per_day"] > 0
    assert big["spread_status"] in {"spreading", "spreading fast"}
    small = enriched.loc[enriched.event_id == "small"].iloc[0]
    assert bool(small["is_spreading"]) is False
    assert small["spread_status"] == "too early / local"
    aged = attach_days_since_ignition(dets)
    assert aged.loc[aged.event_id == "big", "days_since_ignition"].max() == 2
    grid = build_growth_grid(aged, resolution_deg=0.02)
    assert len(grid) >= 2
    assert grid["frp_sum"].sum() == pytest.approx(dets["frp"].sum())
    assert classify_fire_growth(pd.Series({"detection_count": 1, "duration_days": 1, "frp_sum": 5})) == "small / local"

    held = attach_spread_rate_metrics(
        pd.DataFrame(
            [{"event_id": "hold", "detection_count": 2, "duration_days": 2, "footprint_ha": 120.0}]
        ),
        pd.DataFrame(
            [
                {"event_id": "hold", "latitude": 38.0, "longitude": -78.5, "acq_date": "2026-07-01"},
                {"event_id": "hold", "latitude": 38.0, "longitude": -78.5, "acq_date": "2026-07-02"},
            ]
        ),
    )
    assert held.iloc[0]["spread_status"] == "holding / not spreading"


def test_prepare_event_map_layers_sizes_all_events():
    events = pd.DataFrame(
        [
            {
                "event_id": "event_0001",
                "footprint_ha": 100.0,
                "burned_area_ha": 20.0,
                "detection_count": 3,
                "polygon": [[[-78.5, 38.0], [-78.4, 38.0], [-78.4, 38.1], [-78.5, 38.1], [-78.5, 38.0]]],
                "fill_color": [200, 80, 40, 90],
                "end_date": "2026-07-03",
            },
            {
                "event_id": "event_0002",
                "footprint_ha": 10000.0,
                "burned_area_ha": 5000.0,
                "detection_count": 50,
                "polygon": [[[-120.5, 40.0], [-120.0, 40.0], [-120.0, 40.5], [-120.5, 40.5], [-120.5, 40.0]]],
                "fill_color": [40, 120, 200, 90],
                "end_date": "2026-07-04",
            },
        ]
    )
    markers, detail = prepare_event_map_layers(events, max_detail_polygons=1)
    assert len(markers) == 2
    assert markers["radius"].min() > 0
    assert float(markers.loc[markers.event_id == "event_0002", "radius"].iloc[0]) > float(
        markers.loc[markers.event_id == "event_0001", "radius"].iloc[0]
    )
    assert len(detail) == 1
    assert detail.iloc[0]["event_id"] == "event_0002"


def test_prepare_burned_area_map_polygons_sidecar(tmp_path):
    csv_path = tmp_path / "burned_area.csv"
    csv_path.write_text("latitude,longitude,date,burned_area_ha\n", encoding="utf-8")
    poly_path = burned_area_polygons_path(csv_path)
    assert poly_path.name == "burned_area.polygons.json"
    payload = [
        {
            "date": "2026-01-02",
            "burned_area_ha": 50.0,
            "polygon": [[[-78.5, 38.0], [-78.4, 38.0], [-78.4, 38.1], [-78.5, 38.1], [-78.5, 38.0]]],
            "fill_color": [120, 55, 15, 140],
        }
    ]
    poly_path.write_text(json.dumps(payload), encoding="utf-8")
    polys = prepare_burned_area_map_polygons(poly_path, start=date(2026, 1, 1), end=date(2026, 1, 3))
    assert len(polys) == 1
    assert polys.loc[0, "burned_area_ha"] == pytest.approx(50.0)
    assert len(polys.loc[0, "polygon"][0]) == 5


def test_link_firms_to_burned_matches_nearby_same_day():
    firms = pd.DataFrame(
        [
            {"latitude": 38.0, "longitude": -78.5, "acq_date": "2026-07-01"},
            {"latitude": 38.0, "longitude": -78.5, "acq_date": "2026-07-10"},
            {"latitude": 40.0, "longitude": -120.0, "acq_date": "2026-07-01"},
        ]
    )
    burned = pd.DataFrame(
        [
            {"latitude": 38.001, "longitude": -78.501, "date": "2026-07-01", "burned_area_ha": 21.5},
            {"latitude": 38.001, "longitude": -78.501, "date": "2026-07-01", "burned_area_ha": 5.0},
        ]
    )
    linked = link_firms_to_burned(firms, burned, max_km=1.5, day_tolerance=1)
    assert bool(linked.loc[0, "matched_burned"]) is True
    assert linked.loc[0, "nearest_burn_km"] < 1.5
    assert linked.loc[0, "matched_burned_area_ha"] == pytest.approx(21.5)
    assert bool(linked.loc[1, "matched_burned"]) is False  # wrong day
    assert bool(linked.loc[2, "matched_burned"]) is False  # too far


def test_available_timeline_dates_unions_firms_and_burned():
    firms = pd.DataFrame(
        {
            "latitude": [38.0],
            "longitude": [-78.5],
            "acq_date": pd.to_datetime(["2026-01-02"]),
        }
    )
    burned = pd.DataFrame(
        {
            "latitude": [38.1, 38.2],
            "longitude": [-78.4, -78.3],
            "date": pd.to_datetime(["2026-01-01", "2026-01-03"]),
            "burned_area_ha": [10.0, 12.0],
        }
    )
    dates = available_timeline_dates(firms, burned)
    assert dates == [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)]


def test_aggregate_firms_burned_links_monthly_metrics():
    linked = pd.DataFrame(
        [
            {
                "latitude": 38.1,
                "longitude": -78.4,
                "acq_date": "2026-07-01",
                "matched_burned": True,
                "matched_burned_area_ha": 10.0,
                "_matched_burn_key": "a",
            },
            {
                "latitude": 38.2,
                "longitude": -78.3,
                "acq_date": "2026-07-02",
                "matched_burned": True,
                "matched_burned_area_ha": 10.0,
                "_matched_burn_key": "a",
            },
            {
                "latitude": 38.3,
                "longitude": -78.2,
                "acq_date": "2026-07-03",
                "matched_burned": False,
                "matched_burned_area_ha": np.nan,
                "_matched_burn_key": pd.NA,
            },
        ]
    )
    monthly = aggregate_firms_burned_links(linked, region_resolution=1.0)
    assert len(monthly) == 1
    assert monthly.loc[0, "fires_matched_count"] == 2
    assert monthly.loc[0, "fire_match_rate"] == pytest.approx(2 / 3)
    assert monthly.loc[0, "burned_area_ha_matched"] == pytest.approx(10.0)


def test_progress_tracker_advances_and_completes(monkeypatch):
    updates = []

    class DummyBar:
        def progress(self, value, text=None):
            updates.append((float(value), text))

    class DummySlot:
        def markdown(self, value):
            updates.append(("md", value))

        def caption(self, value):
            updates.append(("cap", value))

    monkeypatch.setattr("wildfire.ui.common.st.progress", lambda *_a, **_k: DummyBar())
    monkeypatch.setattr("wildfire.ui.common.st.empty", lambda: DummySlot())

    tracker = ProgressTracker(["One", "Two"])
    tracker.advance("start")
    tracker.tick(1, 2, "halfway")
    tracker.advance("next")
    tracker.complete("done")

    assert any(value >= 1.0 for value, _text in updates if isinstance(value, float))
    assert any(isinstance(item, tuple) and item[0] == "cap" and "Done" in item[1] for item in updates)


def test_extract_selected_event_id_from_pydeck_payload():
    class FakeSelection:
        def __init__(self, objects):
            self.selection = {"objects": objects}

    assert _extract_selected_event_id(FakeSelection({"event_footprints": [{"event_id": "event_0007"}]})) == "event_0007"
    assert _extract_selected_event_id(FakeSelection({"firms_points": [{"event_id": "event_0003"}]})) == "event_0003"
    assert _extract_selected_event_id(None) is None


def test_event_timeline_grows_detections_and_burn():
    firms = pd.DataFrame(
        [
            {"latitude": 38.00, "longitude": -78.50, "acq_date": "2026-07-01", "event_id": "event_0001"},
            {"latitude": 38.02, "longitude": -78.48, "acq_date": "2026-07-02", "event_id": "event_0001"},
            {"latitude": 38.04, "longitude": -78.46, "acq_date": "2026-07-03", "event_id": "event_0001"},
        ]
    )
    burned = pd.DataFrame(
        [
            {"latitude": 38.01, "longitude": -78.49, "date": "2026-07-02", "burned_area_ha": 20.0},
            {"latitude": 38.03, "longitude": -78.47, "date": "2026-07-03", "burned_area_ha": 15.0},
        ]
    )
    timeline = build_event_timeline(firms, burned, "event_0001")
    assert list(timeline["detection_count"]) == [1, 2, 3]
    assert timeline.loc[0, "burned_area_ha"] == pytest.approx(0.0)
    assert timeline.loc[2, "burned_area_ha"] >= 20.0
    snap = build_event_snapshot(firms, burned, "event_0001", date(2026, 7, 3))
    assert snap["detection_count"] == 3
    assert snap["days_since_start"] == 2
    assert snap["footprint_ha"] > 0


def test_combine_similar_firms_points_merges_cross_satellite_same_day():
    frames = [
        pd.DataFrame(
            [
                {
                    "latitude": 34.1000,
                    "longitude": -118.2000,
                    "acq_date": "2026-07-01",
                    "acq_time": "1200",
                    "frp": 40.0,
                    "brightness": 330.0,
                    "confidence": 80,
                    "satellite": "Terra",
                    "instrument": "MODIS",
                    "firms_source": "MODIS_NRT",
                },
                {
                    "latitude": 35.0000,
                    "longitude": -119.0000,
                    "acq_date": "2026-07-01",
                    "acq_time": "1300",
                    "frp": 5.0,
                    "brightness": 310.0,
                    "confidence": 50,
                    "satellite": "Terra",
                    "instrument": "MODIS",
                    "firms_source": "MODIS_NRT",
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "latitude": 34.1005,
                    "longitude": -118.2005,
                    "acq_date": "2026-07-01",
                    "acq_time": "1210",
                    "frp": 55.0,
                    "brightness": 340.0,
                    "confidence": 90,
                    "satellite": "N",
                    "instrument": "VIIRS",
                    "firms_source": "VIIRS_NOAA20_NRT",
                },
            ]
        ),
    ]
    combined = combine_similar_firms_points(frames, max_km=1.0)
    assert len(combined) == 2
    near = combined.sort_values("frp", ascending=False).iloc[0]
    assert near["frp"] == pytest.approx(55.0)
    assert near["merged_detections"] == 2
    assert "MODIS_NRT" in str(near["firms_sources"])
    assert "VIIRS_NOAA20_NRT" in str(near["firms_sources"])


def test_cluster_firms_events_groups_nearby_spread():
    firms = pd.DataFrame(
        [
            {"latitude": 38.00, "longitude": -78.50, "acq_date": "2026-07-01"},
            {"latitude": 38.02, "longitude": -78.48, "acq_date": "2026-07-02"},
            {"latitude": 40.00, "longitude": -120.00, "acq_date": "2026-07-01"},
        ]
    )
    clustered = cluster_firms_events(firms, max_km=15, max_gap_days=3)
    east = clustered[clustered["longitude"] > -100]
    west = clustered[clustered["longitude"] < -100]
    assert east["event_id"].nunique() == 1
    assert len(east) == 2
    assert west["event_id"].nunique() == 1
    assert east["event_id"].iloc[0] != west["event_id"].iloc[0]


def test_build_event_spread_summary_burned_and_overlap():
    firms = pd.DataFrame(
        [
            {"latitude": 38.00, "longitude": -78.50, "acq_date": "2026-07-01"},
            {"latitude": 38.05, "longitude": -78.45, "acq_date": "2026-07-03"},
            {"latitude": 38.02, "longitude": -78.48, "acq_date": "2026-07-02"},
        ]
    )
    burned = pd.DataFrame(
        [
            {"latitude": 38.01, "longitude": -78.49, "date": "2026-07-02", "burned_area_ha": 20.0},
            {"latitude": 39.50, "longitude": -79.50, "date": "2026-07-02", "burned_area_ha": 50.0},
        ]
    )
    clustered, events, metrics = build_event_spread_summary(
        firms,
        burned,
        as_of=date(2026, 7, 3),
        max_km=20,
        max_gap_days=3,
    )
    assert metrics["event_count"] >= 1
    assert metrics["total_footprint_ha"] > 0
    assert metrics["total_burned_area_ha"] == pytest.approx(20.0)
    assert not events.empty
    assert events.loc[0, "burned_area_ha"] == pytest.approx(20.0)
    assert clustered["event_id"].nunique() >= 1


def test_build_feature_table_merges_linked_metrics():
    fire = pd.DataFrame(
        [
            {
                "region_id": "r1",
                "year": 2025,
                "month": 7,
                "fire_count": 3,
                "region_lat_center": 38.5,
                "region_lon_center": -78.5,
            }
        ]
    )
    links = pd.DataFrame(
        [
            {
                "region_id": "r1",
                "year": 2025,
                "month": 7,
                "region_lat_center": 38.5,
                "region_lon_center": -78.5,
                "fires_matched_count": 2,
                "fire_match_rate": 2 / 3,
                "burned_area_ha_matched": 15.0,
            }
        ]
    )
    table = build_feature_table(fire, firms_burned_links_monthly=links)
    assert table.loc[0, "fires_matched_count"] == 2
    assert table.loc[0, "fire_match_rate"] == pytest.approx(2 / 3)
    assert table.loc[0, "burned_area_ha_matched"] == pytest.approx(15.0)
    assert "fires_matched_count_lag1" in table.columns
