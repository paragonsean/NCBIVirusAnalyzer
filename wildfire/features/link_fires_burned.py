"""Spatially and temporally link FIRMS detections to MCD64A1 burned pixels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wildfire.regions import assign_grid_region, normalize_longitude

EARTH_RADIUS_KM = 6371.0
DEFAULT_MAX_KM = 1.5
DEFAULT_DAY_TOLERANCE = 1


def _normalize_firms(firms: pd.DataFrame) -> pd.DataFrame:
    work = firms.copy()
    if "acq_date" not in work.columns and "date" in work.columns:
        work = work.rename(columns={"date": "acq_date"})
    required = {"latitude", "longitude", "acq_date"}
    missing = sorted(required.difference(work.columns))
    if missing:
        raise ValueError(f"FIRMS table missing columns: {', '.join(missing)}")
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce").map(normalize_longitude)
    work["acq_date"] = pd.to_datetime(work["acq_date"], errors="coerce")
    work = work.dropna(subset=["latitude", "longitude", "acq_date"]).copy()
    work["acq_date"] = work["acq_date"].dt.normalize()
    work["_firms_idx"] = np.arange(len(work), dtype=np.int64)
    return work.reset_index(drop=True)


def _normalize_burned(burned: pd.DataFrame) -> pd.DataFrame:
    work = burned.copy()
    required = {"latitude", "longitude", "date", "burned_area_ha"}
    missing = sorted(required.difference(work.columns))
    if missing:
        raise ValueError(f"Burned-area table missing columns: {', '.join(missing)}")
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce").map(normalize_longitude)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["burned_area_ha"] = pd.to_numeric(work["burned_area_ha"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["latitude", "longitude", "date"]).copy()
    work = work[work["burned_area_ha"] > 0].copy()
    work["date"] = work["date"].dt.normalize()
    work["_burn_idx"] = np.arange(len(work), dtype=np.int64)
    work["_burn_key"] = (
        work["latitude"].round(5).astype(str)
        + "|"
        + work["longitude"].round(5).astype(str)
        + "|"
        + work["date"].dt.strftime("%Y-%m-%d")
    )
    return work.reset_index(drop=True)


def link_firms_to_burned(
    firms: pd.DataFrame,
    burned: pd.DataFrame,
    max_km: float = DEFAULT_MAX_KM,
    day_tolerance: int = DEFAULT_DAY_TOLERANCE,
) -> pd.DataFrame:
    """Attach nearest burned-pixel match to each FIRMS detection.

    A match requires both:
    - |acq_date - burn_date| <= day_tolerance
    - haversine distance <= max_km
    """

    try:
        from sklearn.neighbors import BallTree
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to link FIRMS detections to burned area.") from exc

    firms_norm = _normalize_firms(firms)
    out = firms_norm.copy()
    out["matched_burned"] = False
    out["nearest_burn_km"] = np.nan
    out["matched_burned_area_ha"] = np.nan
    out["burn_date"] = pd.NaT
    out["_matched_burn_key"] = pd.Series([pd.NA] * len(out), dtype="object")

    if firms_norm.empty or burned is None or len(burned) == 0:
        return out.drop(columns=["_firms_idx"])

    burned_norm = _normalize_burned(burned)
    if burned_norm.empty:
        return out.drop(columns=["_firms_idx"])

    day_tolerance = max(0, int(day_tolerance))
    out = out.set_index("_firms_idx", drop=False)

    for fire_day, fire_group in firms_norm.groupby(firms_norm["acq_date"], sort=False):
        day_start = fire_day - pd.Timedelta(days=day_tolerance)
        day_end = fire_day + pd.Timedelta(days=day_tolerance)
        burn_group = burned_norm[
            (burned_norm["date"] >= day_start) & (burned_norm["date"] <= day_end)
        ]
        if burn_group.empty:
            continue

        tree = BallTree(
            np.radians(burn_group[["latitude", "longitude"]].to_numpy(dtype=float)),
            metric="haversine",
        )
        fire_coords = np.radians(fire_group[["latitude", "longitude"]].to_numpy(dtype=float))
        distances, indices = tree.query(fire_coords, k=1)
        distances_km = distances[:, 0] * EARTH_RADIUS_KM
        nearest_idx = indices[:, 0]

        for local_i, (dist_km, burn_pos) in enumerate(zip(distances_km, nearest_idx)):
            if float(dist_km) > float(max_km):
                continue
            burn_row = burn_group.iloc[int(burn_pos)]
            firms_idx = int(fire_group.iloc[local_i]["_firms_idx"])
            out.at[firms_idx, "matched_burned"] = True
            out.at[firms_idx, "nearest_burn_km"] = float(dist_km)
            out.at[firms_idx, "matched_burned_area_ha"] = float(burn_row["burned_area_ha"])
            out.at[firms_idx, "burn_date"] = burn_row["date"]
            out.at[firms_idx, "_matched_burn_key"] = burn_row["_burn_key"]

    out["matched_burned"] = out["matched_burned"].astype(bool)
    return out.reset_index(drop=True).drop(columns=["_firms_idx"])


def aggregate_firms_burned_links(
    linked_firms: pd.DataFrame,
    region_resolution: float = 1.0,
) -> pd.DataFrame:
    """Aggregate linked FIRMS↔burned matches to region-month features."""

    empty = pd.DataFrame(
        columns=[
            "region_id",
            "year",
            "month",
            "region_lat_center",
            "region_lon_center",
            "fires_matched_count",
            "fire_match_rate",
            "burned_area_ha_matched",
        ]
    )
    if linked_firms is None or linked_firms.empty:
        return empty

    work = linked_firms.copy()
    if "acq_date" not in work.columns:
        raise ValueError("Linked FIRMS table must include acq_date")
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce").map(normalize_longitude)
    work["acq_date"] = pd.to_datetime(work["acq_date"], errors="coerce")
    if "matched_burned" not in work:
        work["matched_burned"] = False
    work["matched_burned"] = work["matched_burned"].fillna(False).astype(bool)
    if "matched_burned_area_ha" not in work:
        work["matched_burned_area_ha"] = np.nan
    if "_matched_burn_key" not in work:
        work["_matched_burn_key"] = pd.NA
    work = work.dropna(subset=["latitude", "longitude", "acq_date"]).copy()
    if work.empty:
        return empty

    regions = work.apply(
        lambda row: assign_grid_region(row["latitude"], row["longitude"], region_resolution),
        axis=1,
        result_type="expand",
    )
    work = pd.concat([work.reset_index(drop=True), regions.reset_index(drop=True)], axis=1)
    work["year"] = work["acq_date"].dt.year.astype(int)
    work["month"] = work["acq_date"].dt.month.astype(int)

    group_cols = ["region_id", "year", "month", "region_lat_center", "region_lon_center"]
    rows = []
    for keys, group in work.groupby(group_cols, sort=False):
        fire_count = len(group)
        matched = group[group["matched_burned"]]
        fires_matched = int(len(matched))
        if matched.empty:
            burned_ha = 0.0
        elif matched["_matched_burn_key"].isna().all():
            burned_ha = float(matched["matched_burned_area_ha"].sum())
        else:
            burned_ha = float(
                matched.drop_duplicates(subset=["_matched_burn_key"])["matched_burned_area_ha"].sum()
            )
        rows.append(
            {
                "region_id": keys[0],
                "year": int(keys[1]),
                "month": int(keys[2]),
                "region_lat_center": keys[3],
                "region_lon_center": keys[4],
                "fires_matched_count": fires_matched,
                "fire_match_rate": (fires_matched / fire_count) if fire_count else 0.0,
                "burned_area_ha_matched": burned_ha,
            }
        )
    return pd.DataFrame(rows).sort_values(["region_id", "year", "month"]).reset_index(drop=True)


def build_firms_burned_link_monthly(
    firms_csv: str | Path,
    burned_area_csv: str | Path,
    region_resolution: float = 1.0,
    min_confidence: float | None = None,
    max_km: float = DEFAULT_MAX_KM,
    day_tolerance: int = DEFAULT_DAY_TOLERANCE,
) -> pd.DataFrame:
    """Read FIRMS + burned CSVs, link detections, and aggregate monthly."""

    from wildfire.ingest.firms import filter_north_america, read_firms_csv

    firms = filter_north_america(read_firms_csv(firms_csv))
    if min_confidence is not None and "confidence" in firms:
        firms = firms[firms["confidence"] >= float(min_confidence)].copy()
    burned = pd.read_csv(burned_area_csv, low_memory=False)
    linked = link_firms_to_burned(firms, burned, max_km=max_km, day_tolerance=day_tolerance)
    return aggregate_firms_burned_links(linked, region_resolution=region_resolution)
