"""Simple plots for wildfire backtest outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_actual_vs_predicted(backtest_csv: str | Path, output_png: str | Path) -> str:
    """Create an actual-vs-predicted scatter plot from a backtest CSV."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to create wildfire plots.") from exc

    df = pd.read_csv(backtest_csv)
    actual_cols = [col for col in df.columns if col.startswith("actual_")]
    if not actual_cols or "prediction" not in df:
        raise ValueError("Backtest CSV must include an actual_* column and prediction.")

    actual_col = actual_cols[0]
    path = Path(output_png)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[actual_col], df["prediction"], alpha=0.65)
    max_value = max(float(df[actual_col].max()), float(df["prediction"].max()), 1.0)
    ax.plot([0, max_value], [0, max_value], color="black", linewidth=1)
    ax.set_xlabel(actual_col)
    ax.set_ylabel("prediction")
    ax.set_title("Wildfire Backtest: Actual vs Predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
