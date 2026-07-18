"""NASA GRACE/GRACE-FO groundwater and soil moisture ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from wildfire.regions import NORTH_AMERICA_BOUNDS, assign_grid_region, normalize_longitude


GRACE_SHORT_NAME = "GRACEDADM_CLSM025GL_7D"
GRACE_VERSION = "3.0"
GRACE_VARIABLES = ("gws_inst", "rtzsm_inst")


def download_grace(
    output_dir: str | Path,
    temporal: tuple[str, str],
    bounding_box: tuple[float, float, float, float] | None = None,
):
    """Download GRACE granules with earthaccess.

    Requires a configured NASA Earthdata Login. This wrapper is intentionally
    small so tests can exercise local file processing without network access.
    """

    try:
        import earthaccess
    except ImportError as exc:
        raise RuntimeError("Install earthaccess to download GRACE data.") from exc

    earthaccess.login()
    bbox = bounding_box or (
        NORTH_AMERICA_BOUNDS.min_lon,
        NORTH_AMERICA_BOUNDS.min_lat,
        NORTH_AMERICA_BOUNDS.max_lon,
        NORTH_AMERICA_BOUNDS.max_lat,
    )
    results = earthaccess.search_data(
        short_name=GRACE_SHORT_NAME,
        version=GRACE_VERSION,
        temporal=temporal,
        bounding_box=bbox,
    )
    return earthaccess.download(results, local_path=str(output_dir))


def _open_dataset(paths: str | Path | Iterable[str | Path]):
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("Install xarray and netCDF support to read GRACE files.") from exc

    if isinstance(paths, (str, Path)):
        return xr.open_dataset(paths)
    path_list = [str(path) for path in paths]
    if not path_list:
        raise ValueError("No GRACE files were provided.")
    return xr.open_mfdataset(path_list, combine="by_coords")


def load_grace_monthly(
    paths: str | Path | Iterable[str | Path],
    region_resolution: float = 1.0,
) -> pd.DataFrame:
    """Load GRACE NetCDF files and aggregate weekly grids to month-region rows."""

    ds = _open_dataset(paths)
    missing = [name for name in GRACE_VARIABLES if name not in ds]
    if missing:
        raise ValueError(f"GRACE dataset missing variables: {', '.join(missing)}")

    subset = ds[list(GRACE_VARIABLES)]
    if "lon" in subset.coords:
        subset = subset.assign_coords(lon=[normalize_longitude(v) for v in subset["lon"].values])
        subset = subset.sortby("lon")
        subset = subset.sel(
            lon=slice(NORTH_AMERICA_BOUNDS.min_lon, NORTH_AMERICA_BOUNDS.max_lon),
            lat=slice(NORTH_AMERICA_BOUNDS.min_lat, NORTH_AMERICA_BOUNDS.max_lat),
        )
    monthly = subset.resample(time="MS").mean()
    flat = monthly.to_dataframe().reset_index()
    flat = flat.dropna(subset=["lat", "lon"])
    flat["year"] = pd.to_datetime(flat["time"]).dt.year.astype(int)
    flat["month"] = pd.to_datetime(flat["time"]).dt.month.astype(int)

    regions = flat.apply(
        lambda row: assign_grid_region(row["lat"], row["lon"], region_resolution),
        axis=1,
        result_type="expand",
    )
    flat = pd.concat([flat.reset_index(drop=True), regions.reset_index(drop=True)], axis=1)
    grouped = (
        flat.groupby(
            ["region_id", "year", "month", "region_lat_center", "region_lon_center"],
            as_index=False,
        )
        .agg(
            gws_inst_mean=("gws_inst", "mean"),
            gws_inst_p10=("gws_inst", lambda s: s.quantile(0.10)),
            rtzsm_inst_mean=("rtzsm_inst", "mean"),
            rtzsm_inst_p10=("rtzsm_inst", lambda s: s.quantile(0.10)),
        )
        .sort_values(["region_id", "year", "month"])
        .reset_index(drop=True)
    )
    return grouped
