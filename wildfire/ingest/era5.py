"""Copernicus ERA5 atmospheric aridity ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from wildfire.regions import NORTH_AMERICA_BOUNDS, assign_grid_region, normalize_longitude


ERA5_DATASET = "reanalysis-era5-single-levels-monthly-means"
ERA5_VARIABLES = ("2m_temperature", "2m_dewpoint_temperature")


def calculate_vpd_kpa(t2m_kelvin, d2m_kelvin):
    """Calculate vapor pressure deficit from temperature and dewpoint.

    Inputs are Kelvin, matching ERA5. Output is kPa and clipped at zero.
    """

    t_c = np.asarray(t2m_kelvin, dtype=float) - 273.15
    td_c = np.asarray(d2m_kelvin, dtype=float) - 273.15
    saturation = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
    actual = 0.6108 * np.exp((17.27 * td_c) / (td_c + 237.3))
    return np.clip(saturation - actual, 0.0, None)


def request_era5_monthly(
    output_path: str | Path,
    years: Iterable[int],
    months: Iterable[int] = range(1, 13),
    area: tuple[float, float, float, float] | None = None,
) -> str:
    """Request ERA5 monthly means through the CDS API.

    Requires a configured Copernicus CDS API key. The `area` tuple is north,
    west, south, east, matching CDS conventions.
    """

    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("Install cdsapi to request ERA5 data.") from exc

    request_area = area or (
        NORTH_AMERICA_BOUNDS.max_lat,
        NORTH_AMERICA_BOUNDS.min_lon,
        NORTH_AMERICA_BOUNDS.min_lat,
        NORTH_AMERICA_BOUNDS.max_lon,
    )
    client = cdsapi.Client()
    client.retrieve(
        ERA5_DATASET,
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": list(ERA5_VARIABLES),
            "year": [str(year) for year in years],
            "month": [f"{int(month):02d}" for month in months],
            "time": "00:00",
            "area": list(request_area),
            "format": "netcdf",
        },
        str(output_path),
    )
    return str(output_path)


def _open_dataset(paths: str | Path | Iterable[str | Path]):
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("Install xarray and netCDF support to read ERA5 files.") from exc

    if isinstance(paths, (str, Path)):
        return xr.open_dataset(paths)
    path_list = [str(path) for path in paths]
    if not path_list:
        raise ValueError("No ERA5 files were provided.")
    return xr.open_mfdataset(path_list, combine="by_coords")


def _pick_variable(ds, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in ds:
            return name
    raise ValueError(f"None of the expected ERA5 variables were present: {', '.join(candidates)}")


def load_era5_monthly(
    paths: str | Path | Iterable[str | Path],
    region_resolution: float = 1.0,
) -> pd.DataFrame:
    """Load ERA5 NetCDF and aggregate VPD/temperature by month-region."""

    ds = _open_dataset(paths)
    t_name = _pick_variable(ds, ("t2m", "2m_temperature"))
    d_name = _pick_variable(ds, ("d2m", "2m_dewpoint_temperature"))
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    time_name = "valid_time" if "valid_time" in ds.coords else "time"

    subset = ds[[t_name, d_name]].rename(
        {lat_name: "lat", lon_name: "lon", time_name: "time"}
    )
    subset = subset.assign_coords(lon=[normalize_longitude(v) for v in subset["lon"].values])
    subset = subset.sortby("lon")
    subset = subset.sel(
        lon=slice(NORTH_AMERICA_BOUNDS.min_lon, NORTH_AMERICA_BOUNDS.max_lon),
        lat=slice(NORTH_AMERICA_BOUNDS.max_lat, NORTH_AMERICA_BOUNDS.min_lat)
        if subset["lat"].values[0] > subset["lat"].values[-1]
        else slice(NORTH_AMERICA_BOUNDS.min_lat, NORTH_AMERICA_BOUNDS.max_lat),
    )
    subset = subset.assign(vpd=(subset[t_name].dims, calculate_vpd_kpa(subset[t_name], subset[d_name])))

    flat = subset[[t_name, d_name, "vpd"]].to_dataframe().reset_index()
    flat = flat.dropna(subset=["lat", "lon"])
    flat["year"] = pd.to_datetime(flat["time"]).dt.year.astype(int)
    flat["month"] = pd.to_datetime(flat["time"]).dt.month.astype(int)
    flat["t2m_c"] = flat[t_name] - 273.15
    flat["d2m_c"] = flat[d_name] - 273.15

    regions = flat.apply(
        lambda row: assign_grid_region(row["lat"], row["lon"], region_resolution),
        axis=1,
        result_type="expand",
    )
    flat = pd.concat([flat.reset_index(drop=True), regions.reset_index(drop=True)], axis=1)
    return (
        flat.groupby(
            ["region_id", "year", "month", "region_lat_center", "region_lon_center"],
            as_index=False,
        )
        .agg(
            t2m_c_mean=("t2m_c", "mean"),
            d2m_c_mean=("d2m_c", "mean"),
            vpd_mean=("vpd", "mean"),
            vpd_max=("vpd", "max"),
        )
        .sort_values(["region_id", "year", "month"])
        .reset_index(drop=True)
    )
