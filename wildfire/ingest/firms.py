"""NASA FIRMS active-fire CSV ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wildfire.regions import NORTH_AMERICA_BOUNDS, assign_grid_region, normalize_longitude


COLUMN_ALIASES = {
    "latitude": "latitude",
    "lat": "latitude",
    "longitude": "longitude",
    "lon": "longitude",
    "long": "longitude",
    "acq_date": "acq_date",
    "acquisition_date": "acq_date",
    "date": "acq_date",
    "brightness": "brightness",
    "bright_ti4": "brightness",
    "bright_t31": "brightness_t31",
    "confidence": "confidence",
    "instrument": "instrument",
    "satellite": "satellite",
}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename)


def _confidence_to_numeric(value) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().lower()
    if text in {"l", "low"}:
        return 30.0
    if text in {"n", "nominal"}:
        return 60.0
    if text in {"h", "high"}:
        return 90.0
    return pd.to_numeric(value, errors="coerce")


def read_firms_csv(path: str | Path) -> pd.DataFrame:
    """Read a FIRMS CSV and normalize the core columns."""

    df = pd.read_csv(path, low_memory=False)
    df = _canonicalize_columns(df)
    required = {"latitude", "longitude", "acq_date"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"FIRMS CSV missing required columns: {', '.join(missing)}")

    out = df.copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce").map(normalize_longitude)
    out["acq_date"] = pd.to_datetime(out["acq_date"], errors="coerce")
    if "brightness" in out:
        out["brightness"] = pd.to_numeric(out["brightness"], errors="coerce")
    if "confidence" in out:
        out["confidence"] = out["confidence"].map(_confidence_to_numeric)
    return out.dropna(subset=["latitude", "longitude", "acq_date"])


def filter_north_america(df: pd.DataFrame) -> pd.DataFrame:
    """Keep FIRMS rows inside the North America modeling extent."""

    mask = df.apply(
        lambda row: NORTH_AMERICA_BOUNDS.contains(row["longitude"], row["latitude"]),
        axis=1,
    )
    return df.loc[mask].copy()


def aggregate_monthly_fire_counts(
    df: pd.DataFrame,
    region_resolution: float = 1.0,
    min_confidence: float | None = None,
) -> pd.DataFrame:
    """Aggregate active-fire detections to region-month targets."""

    work = df.copy()
    if min_confidence is not None and "confidence" in work:
        work = work[work["confidence"] >= float(min_confidence)].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "region_id",
                "year",
                "month",
                "fire_count",
                "brightness_mean",
                "confidence_mean",
                "region_lat_center",
                "region_lon_center",
            ]
        )

    regions = work.apply(
        lambda row: assign_grid_region(row["latitude"], row["longitude"], region_resolution),
        axis=1,
        result_type="expand",
    )
    work = pd.concat([work.reset_index(drop=True), regions.reset_index(drop=True)], axis=1)
    work["year"] = work["acq_date"].dt.year.astype(int)
    work["month"] = work["acq_date"].dt.month.astype(int)

    aggregations = {"fire_count": ("acq_date", "size")}
    if "brightness" in work:
        aggregations["brightness_mean"] = ("brightness", "mean")
    if "confidence" in work:
        aggregations["confidence_mean"] = ("confidence", "mean")

    group_cols = [
        "region_id",
        "year",
        "month",
        "region_lat_center",
        "region_lon_center",
    ]
    out = work.groupby(group_cols, as_index=False).agg(**aggregations)
    for col in ["brightness_mean", "confidence_mean"]:
        if col not in out:
            out[col] = float("nan")
    return out.sort_values(["region_id", "year", "month"]).reset_index(drop=True)


def load_firms_monthly(
    path: str | Path,
    region_resolution: float = 1.0,
    min_confidence: float | None = None,
) -> pd.DataFrame:
    """Read, filter, and aggregate a FIRMS archive CSV."""

    return aggregate_monthly_fire_counts(
        filter_north_america(read_firms_csv(path)),
        region_resolution=region_resolution,
        min_confidence=min_confidence,
    )
