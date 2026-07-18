"""MCD64A1 burned-area helpers.

The full MCD64A1 product is distributed as gridded HDF/NetCDF-like files. This
module provides a table-level aggregator used by tests and downstream code after
pixels have been extracted to latitude, longitude, date, and burned-area fields.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wildfire.regions import NORTH_AMERICA_BOUNDS, assign_grid_region, normalize_longitude


def download_mcd64a1(output_dir: str | Path, temporal: tuple[str, str]):
    """Search and download MCD64A1 granules with earthaccess."""

    try:
        import earthaccess
    except ImportError as exc:
        raise RuntimeError("Install earthaccess to download MCD64A1 data.") from exc

    earthaccess.login()
    results = earthaccess.search_data(
        short_name="MCD64A1",
        temporal=temporal,
        bounding_box=(
            NORTH_AMERICA_BOUNDS.min_lon,
            NORTH_AMERICA_BOUNDS.min_lat,
            NORTH_AMERICA_BOUNDS.max_lon,
            NORTH_AMERICA_BOUNDS.max_lat,
        ),
    )
    return earthaccess.download(results, local_path=str(output_dir))


def aggregate_burned_area_table(
    pixels: pd.DataFrame,
    region_resolution: float = 1.0,
) -> pd.DataFrame:
    """Aggregate extracted MCD64A1 burned-area pixels by month-region."""

    required = {"latitude", "longitude", "date", "burned_area_ha"}
    missing = sorted(required.difference(pixels.columns))
    if missing:
        raise ValueError(f"Burned-area table missing columns: {', '.join(missing)}")

    work = pixels.copy()
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce").map(normalize_longitude)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["burned_area_ha"] = pd.to_numeric(work["burned_area_ha"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["latitude", "longitude", "date"])
    work = work[
        work.apply(
            lambda row: NORTH_AMERICA_BOUNDS.contains(row["longitude"], row["latitude"]),
            axis=1,
        )
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=["region_id", "year", "month", "burned_area_ha"])

    regions = work.apply(
        lambda row: assign_grid_region(row["latitude"], row["longitude"], region_resolution),
        axis=1,
        result_type="expand",
    )
    work = pd.concat([work.reset_index(drop=True), regions.reset_index(drop=True)], axis=1)
    work["year"] = work["date"].dt.year.astype(int)
    work["month"] = work["date"].dt.month.astype(int)
    return (
        work.groupby(
            ["region_id", "year", "month", "region_lat_center", "region_lon_center"],
            as_index=False,
        )
        .agg(burned_area_ha=("burned_area_ha", "sum"))
        .sort_values(["region_id", "year", "month"])
        .reset_index(drop=True)
    )
