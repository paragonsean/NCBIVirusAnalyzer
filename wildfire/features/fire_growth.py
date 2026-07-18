"""Infer fire growth and intensity from FIRMS / MCD14DL point detections.

Raw MODIS points are 1 km footprints. Clustering, FRP, and date-coded fronts
distinguish a small agricultural burn from a large expanding wildfire.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _ensure_frp(detections: pd.DataFrame) -> pd.Series:
    if "frp" not in detections.columns:
        return pd.Series(0.0, index=detections.index, dtype=float)
    return pd.to_numeric(detections["frp"], errors="coerce").fillna(0.0)


def attach_frp_metrics(events: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
    """Add FRP totals / trend, spread rates, and a simple growth class to per-event rows."""

    if events is None or events.empty:
        return events
    out = events.copy()
    if detections is None or detections.empty or "event_id" not in detections.columns:
        out["frp_sum"] = 0.0
        out["frp_max"] = 0.0
        out["frp_mean"] = 0.0
        out["frp_trend_mw"] = 0.0
        out["growth_class"] = "unknown"
        out["is_spreading"] = False
        out["spread_status"] = "unknown"
        out["spread_km_per_day"] = 0.0
        out["footprint_ha_per_day"] = 0.0
        out["new_cells_per_day"] = 0.0
        return out

    work = detections.copy()
    work["event_id"] = work["event_id"].astype(str)
    work["acq_date"] = pd.to_datetime(work["acq_date"], errors="coerce")
    work["frp"] = _ensure_frp(work)
    work = work.dropna(subset=["acq_date", "event_id"])

    rows = []
    for event_id, group in work.groupby("event_id", sort=False):
        frp = group["frp"]
        by_day = group.groupby(group["acq_date"].dt.normalize())["frp"].sum().sort_index()
        trend = float(by_day.iloc[-1] - by_day.iloc[0]) if len(by_day) >= 2 else 0.0
        rows.append(
            {
                "event_id": event_id,
                "frp_sum": float(frp.sum()),
                "frp_max": float(frp.max()) if len(frp) else 0.0,
                "frp_mean": float(frp.mean()) if len(frp) else 0.0,
                "frp_trend_mw": trend,
                "active_days": int(len(by_day)),
            }
        )
    frp_df = pd.DataFrame(rows)
    out["event_id"] = out["event_id"].astype(str)
    out = out.merge(frp_df, on="event_id", how="left")
    for col in ("frp_sum", "frp_max", "frp_mean", "frp_trend_mw"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["active_days"] = pd.to_numeric(out.get("active_days", 1), errors="coerce").fillna(1).astype(int)
    out = attach_spread_rate_metrics(out, work)
    out["growth_class"] = out.apply(classify_fire_growth, axis=1)
    return out


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    r = 6371.0
    lat1_r = np.radians(lat1.astype(float))
    lon1_r = np.radians(lon1.astype(float))
    lat2_r = math.radians(float(lat2))
    lon2_r = math.radians(float(lon2))
    dlat = lat1_r - lat2_r
    dlon = lon1_r - lon2_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * math.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def attach_spread_rate_metrics(
    events: pd.DataFrame,
    detections: pd.DataFrame,
    grid_deg: float = 0.01,
) -> pd.DataFrame:
    """Estimate whether each event is spreading and how fast (km/day, ha/day).

    Uses day-0 centroid → farthest later detection for radial speed, and unique
    ~1 km grid cells for area expansion. Same-day-only fires are "too early".
    """

    if events is None or events.empty:
        return events
    out = events.copy()
    out["event_id"] = out["event_id"].astype(str)

    empty = {
        "spread_km_per_day": 0.0,
        "max_radius_km": 0.0,
        "footprint_ha_per_day": 0.0,
        "new_cells_per_day": 0.0,
        "latest_new_cells": 0,
        "is_spreading": False,
        "spread_status": "too early / local",
    }
    if detections is None or detections.empty or "event_id" not in detections.columns:
        for k, v in empty.items():
            out[k] = v
        return out

    work = detections.copy()
    work["event_id"] = work["event_id"].astype(str)
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["acq_date"] = pd.to_datetime(work["acq_date"], errors="coerce")
    work = work.dropna(subset=["event_id", "latitude", "longitude", "acq_date"])
    if work.empty:
        for k, v in empty.items():
            out[k] = v
        return out

    res = max(1e-4, float(grid_deg))
    # ~111 km/deg * grid_deg → cell side; area ≈ side² in ha near mid-latitudes.
    cell_ha = (111.0 * res) ** 2 * 100.0
    rows = []
    for event_id, group in work.groupby("event_id", sort=False):
        group = group.sort_values("acq_date")
        days = group["acq_date"].dt.normalize()
        unique_days = sorted(days.unique())
        n_days = len(unique_days)
        first_day = unique_days[0]
        day0 = group.loc[days == first_day]
        ign_lat = float(day0["latitude"].mean())
        ign_lon = float(day0["longitude"].mean())
        radii = _haversine_km(
            group["latitude"].to_numpy(),
            group["longitude"].to_numpy(),
            ign_lat,
            ign_lon,
        )
        max_radius = float(np.max(radii)) if len(radii) else 0.0

        cells = (
            (np.floor(group["longitude"].to_numpy() / res) * res).round(6).astype(str)
            + "_"
            + (np.floor(group["latitude"].to_numpy() / res) * res).round(6).astype(str)
        )
        seen: set[str] = set()
        cells_by_day = []
        for day in unique_days:
            mask = days == day
            day_cells = set(cells[mask.to_numpy()])
            new = day_cells - seen
            seen |= day_cells
            cells_by_day.append(len(new))
        elapsed = max(0, n_days - 1)
        total_cells = len(seen)
        day0_cells = cells_by_day[0] if cells_by_day else 0
        new_cells_total = max(0, total_cells - day0_cells)
        latest_new = int(cells_by_day[-1]) if n_days >= 2 else 0

        spread_km_day = (max_radius / elapsed) if elapsed > 0 else 0.0
        new_cells_day = (new_cells_total / elapsed) if elapsed > 0 else 0.0
        rows.append(
            {
                "event_id": event_id,
                "spread_km_per_day": float(spread_km_day),
                "max_radius_km": float(max_radius),
                "new_cells_per_day": float(new_cells_day),
                "latest_new_cells": latest_new,
                "grid_cells": int(total_cells),
                "grid_cell_ha_proxy": float(total_cells * cell_ha),
                "_elapsed_days": elapsed,
                "_day0_cells": day0_cells,
            }
        )

    rate_df = pd.DataFrame(rows)
    out = out.merge(rate_df, on="event_id", how="left")
    for col in (
        "spread_km_per_day",
        "max_radius_km",
        "new_cells_per_day",
        "grid_cell_ha_proxy",
    ):
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)
    out["latest_new_cells"] = pd.to_numeric(out.get("latest_new_cells"), errors="coerce").fillna(0).astype(int)
    out["grid_cells"] = pd.to_numeric(out.get("grid_cells"), errors="coerce").fillna(0).astype(int)
    elapsed = pd.to_numeric(out.get("_elapsed_days"), errors="coerce").fillna(0)
    footprint = pd.to_numeric(out.get("footprint_ha"), errors="coerce").fillna(0.0)
    # Growth rate: footprint gained after day 0, divided by elapsed days.
    day0_proxy = pd.to_numeric(out.get("_day0_cells"), errors="coerce").fillna(1).clip(lower=1) * cell_ha
    gained = (footprint - day0_proxy).clip(lower=0.0)
    # If footprint looks smaller than day-0 proxy (buffer math), fall back to grid growth.
    grid_gained = (out["grid_cell_ha_proxy"] - day0_proxy).clip(lower=0.0)
    use_grid = gained <= 0
    gained = gained.where(~use_grid, grid_gained)
    out["footprint_ha_per_day"] = np.where(elapsed > 0, gained / elapsed, 0.0)

    statuses = []
    spreading = []
    for _, row in out.iterrows():
        el = int(row.get("_elapsed_days") or 0)
        km_day = float(row.get("spread_km_per_day") or 0.0)
        cells_day = float(row.get("new_cells_per_day") or 0.0)
        latest = int(row.get("latest_new_cells") or 0)
        ha_day = float(row.get("footprint_ha_per_day") or 0.0)
        if el <= 0:
            statuses.append("too early / local")
            spreading.append(False)
            continue
        growing = km_day >= 0.5 or cells_day >= 1.0 or latest >= 1 or ha_day >= 50.0
        spreading.append(bool(growing))
        if km_day >= 5.0 or cells_day >= 8.0 or ha_day >= 2_000.0:
            statuses.append("spreading fast")
        elif growing:
            statuses.append("spreading")
        else:
            statuses.append("holding / not spreading")
    out["spread_status"] = statuses
    out["is_spreading"] = spreading
    out = out.drop(columns=[c for c in ("_elapsed_days", "_day0_cells") if c in out.columns])
    return out


def classify_fire_growth(row: pd.Series) -> str:
    """Label events so small burns and large growing fires are not treated alike."""

    dets = int(row.get("detection_count") or 0)
    days = int(row.get("duration_days") or row.get("active_days") or 1)
    footprint = float(row.get("footprint_ha") or 0.0)
    frp_sum = float(row.get("frp_sum") or 0.0)
    frp_trend = float(row.get("frp_trend_mw") or 0.0)
    spread_status = str(row.get("spread_status") or "")
    km_day = float(row.get("spread_km_per_day") or 0.0)

    # ~1 MODIS pixel ≈ 100 ha; a 10-acre (~4 ha) burn rarely lights many adjacent cells.
    if dets <= 2 and days <= 1 and frp_sum < 50:
        return "small / local"
    if dets >= 20 or footprint >= 5_000 or frp_sum >= 500:
        if days >= 2 or frp_trend > 0 or dets >= 40 or km_day >= 0.5:
            return "large growing"
        return "large intense"
    if spread_status == "spreading fast":
        return "expanding fast"
    if days >= 2 and (dets >= 5 or frp_trend > 0 or km_day >= 0.5 or "spreading" in spread_status):
        return "expanding"
    return "moderate"


def attach_days_since_ignition(detections: pd.DataFrame) -> pd.DataFrame:
    """Age each point relative to its event's first detection (core → front)."""

    if detections is None or detections.empty:
        return detections
    work = detections.copy()
    if "event_id" not in work.columns:
        work["days_since_ignition"] = 0
        return work
    work["acq_date"] = pd.to_datetime(work["acq_date"], errors="coerce")
    starts = work.groupby("event_id")["acq_date"].transform("min")
    work["days_since_ignition"] = (work["acq_date"].dt.normalize() - starts.dt.normalize()).dt.days
    work["days_since_ignition"] = work["days_since_ignition"].clip(lower=0).fillna(0).astype(int)
    return work


def build_growth_grid(
    detections: pd.DataFrame,
    resolution_deg: float = 0.01,
) -> pd.DataFrame:
    """Aggregate detections onto a lon/lat grid by day (scatter → expansion cells).

    Each cell is roughly resolution_deg degrees (~1 km near mid-latitudes when
    resolution_deg≈0.01), matching the MODIS active-fire footprint scale.
    """

    if detections is None or detections.empty:
        return pd.DataFrame(
            columns=[
                "grid_lon",
                "grid_lat",
                "acq_date",
                "detection_count",
                "frp_sum",
                "event_count",
            ]
        )

    work = detections.copy()
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["acq_date"] = pd.to_datetime(work["acq_date"], errors="coerce")
    work["frp"] = _ensure_frp(work)
    work = work.dropna(subset=["latitude", "longitude", "acq_date"])
    if work.empty:
        return build_growth_grid(pd.DataFrame())

    res = max(1e-4, float(resolution_deg))
    work["grid_lon"] = (np.floor(work["longitude"] / res) * res + res / 2.0).round(6)
    work["grid_lat"] = (np.floor(work["latitude"] / res) * res + res / 2.0).round(6)
    work["acq_date"] = work["acq_date"].dt.normalize()

    agg = (
        work.groupby(["grid_lon", "grid_lat", "acq_date"], as_index=False)
        .agg(
            detection_count=("latitude", "size"),
            frp_sum=("frp", "sum"),
            event_count=("event_id", "nunique") if "event_id" in work.columns else ("latitude", "size"),
        )
        .sort_values(["acq_date", "frp_sum"], ascending=[True, False])
        .reset_index(drop=True)
    )
    agg["date"] = agg["acq_date"].dt.strftime("%Y-%m-%d")
    # Visual radius for pydeck scatter of grid cells.
    agg["radius"] = (np.sqrt(agg["detection_count"].clip(lower=1)) * 600.0).clip(400.0, 20_000.0)
    return agg


def growth_front_color(days_since_ignition: float, max_days: float = 7.0) -> list[int]:
    """Color core (old) → advancing front (new): blue/purple → orange → red/yellow."""

    if pd.isna(days_since_ignition):
        return [180, 180, 180, 180]
    span = max(1.0, float(max_days))
    t = min(1.0, max(0.0, float(days_since_ignition) / span))
    # Cool core → hot front
    r = int(40 + t * (255 - 40))
    g = int(80 + t * (40 - 80) + (1 - abs(2 * t - 1)) * 120)
    b = int(220 - t * 200)
    return [r, max(0, min(255, g)), max(0, min(255, b)), 210]


def frp_radius_meters(frp_sum: float, footprint_ha: float = 0.0) -> float:
    """Marker radius from intensity + footprint so large FRP fires read bigger."""

    frp = max(0.0, float(frp_sum))
    foot = max(1.0, float(footprint_ha))
    # Blend area scale with FRP (MW).
    return float(
        min(
            80_000.0,
            max(500.0, math.sqrt(foot) * 80.0 + math.sqrt(frp + 1.0) * 120.0),
        )
    )
