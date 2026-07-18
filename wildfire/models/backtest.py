"""Baseline tabular wildfire forecasting and backtesting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from wildfire.features.build import assert_time_split_no_leakage


DEFAULT_FEATURE_COLUMNS = [
    "month",
    "region_lat_center",
    "region_lon_center",
    "brightness_mean",
    "confidence_mean",
    "gws_inst_mean",
    "gws_inst_p10",
    "rtzsm_inst_mean",
    "rtzsm_inst_p10",
    "vpd_mean",
    "vpd_max",
    "t2m_c_mean",
    "d2m_c_mean",
    "gws_inst_mean_lag1",
    "gws_inst_mean_lag2",
    "gws_inst_mean_lag3",
    "rtzsm_inst_mean_lag1",
    "rtzsm_inst_mean_lag2",
    "rtzsm_inst_mean_lag3",
    "vpd_mean_lag1",
    "vpd_mean_lag2",
    "vpd_mean_lag3",
    "fire_count_lag1",
    "fire_count_lag2",
    "fire_count_lag3",
    "burned_area_ha_lag1",
    "burned_area_ha_lag2",
    "burned_area_ha_lag3",
    "gws_inst_mean_roll3",
    "rtzsm_inst_mean_roll3",
    "vpd_mean_roll3",
]


def available_feature_columns(df: pd.DataFrame, requested: Iterable[str] | None = None) -> list[str]:
    candidates = list(requested or DEFAULT_FEATURE_COLUMNS)
    return [col for col in candidates if col in df.columns]


def _load_sklearn():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            accuracy_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to train wildfire models.") from exc
    return {
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "accuracy_score": accuracy_score,
        "roc_auc_score": roc_auc_score,
        "make_pipeline": make_pipeline,
    }


def _prepare_xy(df: pd.DataFrame, feature_columns: list[str], target_column: str):
    X = df[feature_columns].copy()
    y = pd.to_numeric(df[target_column], errors="coerce").fillna(0)
    return X, y


def _top_risk_recall(result_df: pd.DataFrame, target_column: str, fraction: float = 0.20) -> float:
    if result_df.empty:
        return float("nan")
    n = max(1, int(np.ceil(len(result_df) * fraction)))
    top_pred = result_df.nlargest(n, "prediction")
    top_actual_regions = set(result_df.nlargest(n, target_column).index)
    if not top_actual_regions:
        return float("nan")
    return len(set(top_pred.index).intersection(top_actual_regions)) / len(top_actual_regions)


def run_regression_backtest(
    feature_table: pd.DataFrame,
    output_dir: str | Path,
    target_column: str = "fire_count",
    backtest_year: int = 2026,
    feature_columns: Iterable[str] | None = None,
    random_state: int = 42,
) -> dict:
    """Train on years before `backtest_year` and score that held-out year."""

    if target_column not in feature_table:
        raise ValueError(f"Feature table missing target column: {target_column}")
    features = available_feature_columns(feature_table, feature_columns)
    if not features:
        raise ValueError("No usable model features were found.")
    assert_time_split_no_leakage(feature_table, backtest_year, features)

    deps = _load_sklearn()
    train = feature_table[feature_table["year"] < int(backtest_year)].copy()
    test = feature_table[feature_table["year"] == int(backtest_year)].copy()
    X_train, y_train = _prepare_xy(train, features, target_column)
    X_test, y_test = _prepare_xy(test, features, target_column)

    model = deps["make_pipeline"](
        deps["SimpleImputer"](strategy="median"),
        deps["HistGradientBoostingRegressor"](random_state=random_state),
    )
    model.fit(X_train, y_train)
    predictions = np.clip(model.predict(X_test), 0.0, None)

    results = test[
        ["region_id", "year", "month"]
        + [col for col in ["region_lat_center", "region_lon_center"] if col in test]
    ].copy()
    results[f"actual_{target_column}"] = y_test.to_numpy()
    results["prediction"] = predictions
    results["absolute_error"] = np.abs(results[f"actual_{target_column}"] - results["prediction"])
    results_for_metric = results.rename(columns={f"actual_{target_column}": target_column})

    mse = deps["mean_squared_error"](y_test, predictions)
    metrics = {
        "target_column": target_column,
        "backtest_year": int(backtest_year),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_columns": features,
        "mae": float(deps["mean_absolute_error"](y_test, predictions)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(deps["r2_score"](y_test, predictions)) if len(test) > 1 else float("nan"),
        "top_20pct_recall": float(_top_risk_recall(results_for_metric, target_column)),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / f"wildfire_backtest_{backtest_year}_{target_column}.csv"
    metrics_path = out_dir / f"wildfire_backtest_{backtest_year}_{target_column}_metrics.json"
    results.to_csv(predictions_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    metrics["predictions_csv"] = str(predictions_path)
    metrics["metrics_json"] = str(metrics_path)
    return metrics


def run_severe_fire_classifier_backtest(
    feature_table: pd.DataFrame,
    output_dir: str | Path,
    label_column: str = "severe_fire_month",
    backtest_year: int = 2026,
    feature_columns: Iterable[str] | None = None,
    random_state: int = 42,
) -> dict:
    """Train a binary severe-fire-month classifier and backtest a held-out year."""

    if label_column not in feature_table:
        raise ValueError(f"Feature table missing label column: {label_column}")
    features = available_feature_columns(feature_table, feature_columns)
    assert_time_split_no_leakage(feature_table, backtest_year, features)

    deps = _load_sklearn()
    train = feature_table[feature_table["year"] < int(backtest_year)].copy()
    test = feature_table[feature_table["year"] == int(backtest_year)].copy()
    X_train, y_train = _prepare_xy(train, features, label_column)
    X_test, y_test = _prepare_xy(test, features, label_column)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    model = deps["make_pipeline"](
        deps["SimpleImputer"](strategy="median"),
        deps["RandomForestClassifier"](
            n_estimators=200,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced_subsample",
        ),
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results = test[["region_id", "year", "month"]].copy()
    results[f"actual_{label_column}"] = y_test.to_numpy()
    results["probability"] = probabilities
    results["prediction"] = predictions
    metrics = {
        "label_column": label_column,
        "backtest_year": int(backtest_year),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_columns": features,
        "accuracy": float(deps["accuracy_score"](y_test, predictions)),
    }
    if len(set(y_test)) > 1:
        metrics["roc_auc"] = float(deps["roc_auc_score"](y_test, probabilities))
    else:
        metrics["roc_auc"] = float("nan")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / f"wildfire_severe_backtest_{backtest_year}.csv"
    metrics_path = out_dir / f"wildfire_severe_backtest_{backtest_year}_metrics.json"
    results.to_csv(predictions_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    metrics["predictions_csv"] = str(predictions_path)
    metrics["metrics_json"] = str(metrics_path)
    return metrics
