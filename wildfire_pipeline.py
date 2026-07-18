"""Command-line entry point for the wildfire ML pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wildfire.features.build import build_feature_table, save_feature_table
from wildfire.ingest.era5 import load_era5_monthly
from wildfire.ingest.firms import load_firms_monthly
from wildfire.ingest.grace import load_grace_monthly
from wildfire.ingest.mcd64a1 import aggregate_burned_area_table
from wildfire.models.backtest import run_regression_backtest, run_severe_fire_classifier_backtest


def build_features_command(args) -> None:
    fire = load_firms_monthly(
        args.firms_csv,
        region_resolution=args.region_resolution,
        min_confidence=args.min_confidence,
    )
    grace = load_grace_monthly(args.grace_nc, args.region_resolution) if args.grace_nc else None
    era5 = load_era5_monthly(args.era5_nc, args.region_resolution) if args.era5_nc else None
    burned = None
    if args.burned_area_csv:
        burned_pixels = pd.read_csv(args.burned_area_csv)
        burned = aggregate_burned_area_table(burned_pixels, args.region_resolution)
    table = build_feature_table(
        fire,
        grace_monthly=grace,
        era5_monthly=era5,
        burned_area_monthly=burned,
        target_column=args.target_column,
    )
    path = save_feature_table(table, args.output_csv)
    print(f"Wrote feature table: {path} ({len(table):,} rows)")


def backtest_command(args) -> None:
    table = pd.read_csv(args.feature_csv)
    metrics = run_regression_backtest(
        table,
        args.output_dir,
        target_column=args.target_column,
        backtest_year=args.backtest_year,
    )
    if args.classifier:
        classifier_metrics = run_severe_fire_classifier_backtest(
            table,
            args.output_dir,
            backtest_year=args.backtest_year,
        )
        metrics["classifier"] = classifier_metrics
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="North America wildfire ML pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-features", help="Build monthly region-level features")
    build.add_argument("--firms-csv", required=True, help="FIRMS archive CSV path")
    build.add_argument("--grace-nc", nargs="*", help="GRACE NetCDF files")
    build.add_argument("--era5-nc", nargs="*", help="ERA5 NetCDF files")
    build.add_argument("--burned-area-csv", help="Optional extracted MCD64A1 pixel table")
    build.add_argument("--output-csv", required=True, help="Output feature CSV")
    build.add_argument("--target-column", default="fire_count")
    build.add_argument("--region-resolution", type=float, default=1.0)
    build.add_argument("--min-confidence", type=float)
    build.set_defaults(func=build_features_command)

    backtest = sub.add_parser("backtest", help="Run held-out-year model backtest")
    backtest.add_argument("--feature-csv", required=True)
    backtest.add_argument("--output-dir", default=str(Path("wildfire_output")))
    backtest.add_argument("--target-column", default="fire_count")
    backtest.add_argument("--backtest-year", type=int, default=2026)
    backtest.add_argument("--classifier", action="store_true")
    backtest.set_defaults(func=backtest_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
