"""Build monthly wildfire modeling tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


KEY_COLUMNS = ["region_id", "year", "month"]
REGION_META_COLUMNS = ["region_lat_center", "region_lon_center"]
DEFAULT_LAG_COLUMNS = [
    "gws_inst_mean",
    "gws_inst_p10",
    "rtzsm_inst_mean",
    "rtzsm_inst_p10",
    "vpd_mean",
    "vpd_max",
    "t2m_c_mean",
    "fire_count",
    "burned_area_ha",
]


def _coerce_month_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in KEY_COLUMNS:
        if col not in out:
            raise ValueError(f"Monthly table missing required column: {col}")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["region_id", "year", "month"]).copy()
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    return out


def _merge_monthly(left: pd.DataFrame, right: pd.DataFrame | None) -> pd.DataFrame:
    if right is None or right.empty:
        return left
    rhs = _coerce_month_columns(right)
    duplicate_meta = [col for col in REGION_META_COLUMNS if col in left.columns and col in rhs.columns]
    rhs = rhs.drop(columns=duplicate_meta)
    return left.merge(rhs, on=KEY_COLUMNS, how="left")


def add_lag_features(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    lags: Iterable[int] = (1, 2, 3),
) -> pd.DataFrame:
    """Add region-local lag features for selected columns."""

    out = df.sort_values(["region_id", "year", "month"]).copy()
    cols = [col for col in (columns or DEFAULT_LAG_COLUMNS) if col in out.columns]
    grouped = out.groupby("region_id", sort=False)
    for col in cols:
        for lag in lags:
            out[f"{col}_lag{lag}"] = grouped[col].shift(int(lag))
    return out


def add_rolling_drought_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple multi-month drought/aridity indicators."""

    out = df.sort_values(["region_id", "year", "month"]).copy()
    grouped = out.groupby("region_id", sort=False)
    if "gws_inst_mean" in out:
        out["gws_inst_mean_roll3"] = grouped["gws_inst_mean"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
    if "rtzsm_inst_mean" in out:
        out["rtzsm_inst_mean_roll3"] = grouped["rtzsm_inst_mean"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
    if "vpd_mean" in out:
        out["vpd_mean_roll3"] = grouped["vpd_mean"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
    return out


def add_severe_fire_label(
    df: pd.DataFrame,
    target_column: str = "fire_count",
    quantile: float = 0.80,
) -> pd.DataFrame:
    """Add a binary severe-fire-month label by region-local target quantile."""

    out = df.copy()
    if target_column not in out:
        return out

    def label_region(values: pd.Series) -> pd.Series:
        positive = values.fillna(0)
        threshold = positive.quantile(quantile)
        if threshold <= 0:
            threshold = 1
        return (positive >= threshold).astype(int)

    out["severe_fire_month"] = out.groupby("region_id", group_keys=False)[
        target_column
    ].apply(label_region)
    return out


def build_feature_table(
    fire_monthly: pd.DataFrame,
    grace_monthly: pd.DataFrame | None = None,
    era5_monthly: pd.DataFrame | None = None,
    burned_area_monthly: pd.DataFrame | None = None,
    target_column: str = "fire_count",
    lags: Iterable[int] = (1, 2, 3),
) -> pd.DataFrame:
    """Join monthly target and predictor tables into one modeling table."""

    base = _coerce_month_columns(fire_monthly)
    if "fire_count" not in base:
        base["fire_count"] = 0
    table = _merge_monthly(base, burned_area_monthly)
    table = _merge_monthly(table, grace_monthly)
    table = _merge_monthly(table, era5_monthly)

    numeric_cols = [col for col in table.columns if col not in KEY_COLUMNS and col != "region_id"]
    for col in numeric_cols:
        if table[col].dtype == object:
            converted = pd.to_numeric(table[col], errors="coerce")
            if converted.notna().any():
                table[col] = converted

    table = table.sort_values(["region_id", "year", "month"]).reset_index(drop=True)
    table = add_lag_features(table, lags=lags)
    table = add_rolling_drought_features(table)
    table = add_severe_fire_label(table, target_column=target_column)
    return table


def save_feature_table(df: pd.DataFrame, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def assert_time_split_no_leakage(
    df: pd.DataFrame,
    backtest_year: int,
    feature_columns: Iterable[str] | None = None,
) -> None:
    """Validate that train/test years are split before the backtest year."""

    if df.empty:
        raise ValueError("Feature table is empty.")
    if "year" not in df:
        raise ValueError("Feature table must contain a year column.")
    train = df[df["year"] < int(backtest_year)]
    test = df[df["year"] == int(backtest_year)]
    if train.empty:
        raise ValueError(f"No training rows exist before {backtest_year}.")
    if test.empty:
        raise ValueError(f"No backtest rows exist for {backtest_year}.")
    if train["year"].max() >= int(backtest_year):
        raise ValueError("Training data overlaps the backtest year.")
    if feature_columns:
        unavailable = sorted(set(feature_columns).difference(df.columns))
        if unavailable:
            raise ValueError(f"Missing feature columns: {', '.join(unavailable)}")
