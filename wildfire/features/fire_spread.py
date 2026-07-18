"""Cluster FIRMS detections into fire events and measure spread / overlap / burn."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

from wildfire.features.fire_growth import attach_frp_metrics
from wildfire.regions import normalize_longitude

EARTH_RADIUS_KM = 6371.0
DEFAULT_EVENT_MAX_KM = 15.0
DEFAULT_EVENT_GAP_DAYS = 3
METERS_PER_DEG_LAT = 111_320.0
MAX_HULL_POINTS = 250
MAX_OVERLAP_EVENTS = 40
FOOTPRINT_BUFFER_M = 750.0
# Discrete circle used for Minkowski-sum "buffer" (avoids Shapely buffer()).
FOOTPRINT_BUFFER_SEGS = 12
FOOTPRINT_WORKERS = max(1, min(16, (os.cpu_count() or 4)))
_BUFFER_ANGLES = np.linspace(0.0, 2.0 * np.pi, FOOTPRINT_BUFFER_SEGS, endpoint=False)
_BUFFER_COS = np.cos(_BUFFER_ANGLES)
_BUFFER_SIN = np.sin(_BUFFER_ANGLES)
_BUFFER_OFFSETS_M = np.column_stack(
    (FOOTPRINT_BUFFER_M * _BUFFER_COS, FOOTPRINT_BUFFER_M * _BUFFER_SIN)
)
_SINGLE_FOOTPRINT_HA = math.pi * (FOOTPRINT_BUFFER_M**2) / 10_000.0


def _prepare_firms(firms: pd.DataFrame) -> pd.DataFrame:
    work = firms.copy()
    if "acq_date" not in work.columns and "date" in work.columns:
        work = work.rename(columns={"date": "acq_date"})
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce").map(normalize_longitude)
    work["acq_date"] = pd.to_datetime(work["acq_date"], errors="coerce")
    work = work.dropna(subset=["latitude", "longitude", "acq_date"]).copy()
    work["acq_date"] = work["acq_date"].dt.normalize()
    return work.sort_values("acq_date").reset_index(drop=True)


def cluster_firms_events(
    firms: pd.DataFrame,
    max_km: float = DEFAULT_EVENT_MAX_KM,
    max_gap_days: int = DEFAULT_EVENT_GAP_DAYS,
    progress_callback=None,
) -> pd.DataFrame:
    """Assign space-time fire event IDs to FIRMS detections.

    Same-day detections are clustered with DBSCAN, then each day-cluster joins
    the nearest prior event within ``max_km`` and ``max_gap_days`` (or starts a
    new event). This stays fast for tens of thousands of points.
    """

    try:
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import BallTree
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to cluster FIRMS fire events.") from exc

    work = _prepare_firms(firms)
    if work.empty:
        out = work.copy()
        out["event_id"] = pd.Series(dtype="object")
        return out

    max_gap_days = max(0, int(max_gap_days))
    max_radians = float(max_km) / EARTH_RADIUS_KM
    coords = np.radians(work[["latitude", "longitude"]].to_numpy(dtype=float))
    event_of = np.full(len(work), -1, dtype=np.int64)
    next_event = 0

    unique_days = sorted(pd.unique(work["acq_date"]))
    day_to_indices: dict = {}
    for day, group in work.groupby("acq_date", sort=True):
        day_to_indices[day] = group.index.to_numpy(dtype=np.int64)

    total_days = max(1, len(unique_days))
    for day_i, day in enumerate(unique_days):
        if progress_callback is not None and (day_i % 5 == 0 or day_i + 1 == total_days):
            day_label = pd.Timestamp(day).date().isoformat()
            progress_callback(day_i, total_days, f"day {day_label}")

        day_idxs = day_to_indices[day]
        day_coords = coords[day_idxs]
        if len(day_idxs) == 1:
            local_labels = np.array([0], dtype=np.int64)
        else:
            local_labels = DBSCAN(
                eps=max_radians,
                min_samples=1,
                metric="haversine",
            ).fit_predict(day_coords)

        # Prior detections still eligible for linking.
        window_start = day - pd.Timedelta(days=max_gap_days)
        prev_parts = [
            day_to_indices[other]
            for other in unique_days
            if window_start <= other < day
        ]
        prev_idxs = np.concatenate(prev_parts) if prev_parts else np.array([], dtype=np.int64)
        prev_tree = None
        if len(prev_idxs):
            prev_tree = BallTree(coords[prev_idxs], metric="haversine")

        for local_id in np.unique(local_labels):
            members = day_idxs[local_labels == local_id]
            member_coords = coords[members]
            centroid = member_coords.mean(axis=0, keepdims=True)

            assigned = None
            if prev_tree is not None:
                dist, ind = prev_tree.query(centroid, k=1)
                if float(dist[0, 0]) * EARTH_RADIUS_KM <= float(max_km):
                    assigned = int(event_of[prev_idxs[int(ind[0, 0])]])

            if assigned is None or assigned < 0:
                assigned = next_event
                next_event += 1
            event_of[members] = assigned

    root_order = pd.unique(event_of)
    root_to_event = {int(root): f"event_{idx + 1:04d}" for idx, root in enumerate(root_order)}
    work["event_id"] = [root_to_event[int(r)] for r in event_of]
    if progress_callback is not None:
        progress_callback(total_days, total_days, f"{work['event_id'].nunique():,} events")
    return work


def _shoelace_area_m2(ring_xy: np.ndarray) -> float:
    """Polygon area in m² from a closed or open ring in local meters."""

    if ring_xy is None or len(ring_xy) < 3:
        return 0.0
    x = ring_xy[:, 0]
    y = ring_xy[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _points_to_polygon(lons, lats):
    """Build a lon/lat footprint via convex hull ⊕ discrete circle (no Shapely buffer).

    This Minkowski-sum approximation matches a ~750 m buffer closely enough for
    map footprints / burned-area joins, and is far cheaper for tens of thousands
    of mostly single-point events.
    """

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if len(lons) == 0:
        return None, 0.0

    if len(lons) > MAX_HULL_POINTS:
        idx = np.linspace(0, len(lons) - 1, MAX_HULL_POINTS, dtype=int)
        lons = lons[idx]
        lats = lats[idx]

    lat0 = float(np.mean(lats))
    cos_lat = max(0.2, math.cos(math.radians(lat0)))
    scale_x = METERS_PER_DEG_LAT * cos_lat
    xs = lons * scale_x
    ys = lats * METERS_PER_DEG_LAT
    x0 = float(xs[0])
    y0 = float(ys[0])
    pts = np.column_stack((xs - x0, ys - y0))

    if len(pts) == 1:
        # Hot path: most FIRMS "events" are isolated single detections.
        ring_m = pts[0] + _BUFFER_OFFSETS_M
    else:
        # Expand every detection by the buffer circle, then take the outer hull.
        expanded = (pts[:, None, :] + _BUFFER_OFFSETS_M[None, :, :]).reshape(-1, 2)
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(expanded, qhull_options="QJ")
            ring_m = expanded[hull.vertices]
        except Exception:
            center = pts.mean(axis=0)
            ring_m = center + _BUFFER_OFFSETS_M

    ring_m = np.vstack((ring_m, ring_m[0]))
    footprint_ha = _shoelace_area_m2(ring_m) / 10_000.0
    polygon = [
        [float((x + x0) / scale_x), float((y + y0) / METERS_PER_DEG_LAT)]
        for x, y in ring_m
    ]
    return polygon, float(footprint_ha)


def _point_in_polygon_mask(lons, lats, polygon_lonlat) -> np.ndarray:
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if not polygon_lonlat or len(polygon_lonlat) < 3 or len(lons) == 0:
        return np.zeros(len(lons), dtype=bool)

    pts = np.column_stack((lons, lats))
    ring = np.asarray(polygon_lonlat, dtype=float)
    minx, miny = ring[:, 0].min(), ring[:, 1].min()
    maxx, maxy = ring[:, 0].max(), ring[:, 1].max()
    candidates = (lons >= minx) & (lons <= maxx) & (lats >= miny) & (lats <= maxy)
    mask = np.zeros(len(lons), dtype=bool)
    if not candidates.any():
        return mask

    try:
        from matplotlib.path import Path

        mask[candidates] = Path(ring).contains_points(pts[candidates])
        return mask
    except Exception:
        from shapely import contains_xy
        from shapely.geometry import Polygon

        poly = Polygon(polygon_lonlat)
        if not poly.is_valid:
            poly = poly.buffer(0)
        mask[candidates] = contains_xy(poly, lons[candidates], lats[candidates])
        return mask


def _burned_arrays(burned_work: pd.DataFrame | None):
    """Return (tree, lons, lats, areas) for fast burned lookups, or Nones."""

    if burned_work is None or burned_work.empty:
        return None, None, None, None
    try:
        from sklearn.neighbors import BallTree
    except ImportError:
        lons = burned_work["longitude"].to_numpy(dtype=float)
        lats = burned_work["latitude"].to_numpy(dtype=float)
        areas = burned_work["burned_area_ha"].to_numpy(dtype=float)
        return None, lons, lats, areas

    lons = burned_work["longitude"].to_numpy(dtype=float)
    lats = burned_work["latitude"].to_numpy(dtype=float)
    areas = burned_work["burned_area_ha"].to_numpy(dtype=float)
    tree = BallTree(np.radians(np.column_stack((lats, lons))), metric="haversine")
    return tree, lons, lats, areas


def _burned_inside_polygon(
    tree,
    burned_lons,
    burned_lats,
    burned_ha,
    polygon,
    *,
    center_lon: float | None = None,
    center_lat: float | None = None,
    detection_count: int = 0,
) -> tuple[float, int]:
    """Sum burned area inside a footprint using a BallTree candidate filter."""

    if polygon is None or burned_lons is None or len(burned_lons) == 0:
        return 0.0, 0

    arr = np.asarray(polygon, dtype=float)
    if arr.ndim != 2 or len(arr) < 3:
        return 0.0, 0

    lon0 = float(center_lon) if center_lon is not None else float(arr[:, 0].mean())
    lat0 = float(center_lat) if center_lat is not None else float(arr[:, 1].mean())

    # Single-detection footprints are circles: radius query is exact enough.
    if tree is not None and int(detection_count) <= 1:
        idxs = tree.query_radius(
            np.radians([[lat0, lon0]]),
            r=(FOOTPRINT_BUFFER_M / 1000.0) / EARTH_RADIUS_KM,
        )[0]
        if len(idxs) == 0:
            return 0.0, 0
        return float(burned_ha[idxs].sum()), int(len(idxs))

    cos_lat = max(0.2, math.cos(math.radians(lat0)))
    width_km = max(0.01, (arr[:, 0].max() - arr[:, 0].min()) * METERS_PER_DEG_LAT * cos_lat / 1000.0)
    height_km = max(0.01, (arr[:, 1].max() - arr[:, 1].min()) * METERS_PER_DEG_LAT / 1000.0)
    radius_km = 0.5 * math.hypot(width_km, height_km) + 1.5

    if tree is not None:
        idxs = tree.query_radius(
            np.radians([[lat0, lon0]]),
            r=radius_km / EARTH_RADIUS_KM,
        )[0]
        if len(idxs) == 0:
            return 0.0, 0
        cand_lons = burned_lons[idxs]
        cand_lats = burned_lats[idxs]
        cand_ha = burned_ha[idxs]
        mask = _point_in_polygon_mask(cand_lons, cand_lats, polygon)
        if not mask.any():
            return 0.0, 0
        return float(cand_ha[mask].sum()), int(mask.sum())

    mask = _point_in_polygon_mask(burned_lons, burned_lats, polygon)
    if not mask.any():
        return 0.0, 0
    return float(burned_ha[mask].sum()), int(mask.sum())


def _summarize_one_event(args) -> dict:
    """Worker: build one event footprint and burned-in-footprint totals."""

    (
        event_id,
        lons,
        lats,
        start_iso,
        end_iso,
        duration_days,
        detection_count,
        tree,
        burned_lons,
        burned_lats,
        burned_ha,
    ) = args
    polygon, footprint_ha = _points_to_polygon(lons, lats)
    burned_area_ha, burned_count = _burned_inside_polygon(
        tree,
        burned_lons,
        burned_lats,
        burned_ha,
        polygon,
        center_lon=float(np.mean(lons)) if len(lons) else None,
        center_lat=float(np.mean(lats)) if len(lats) else None,
        detection_count=int(detection_count),
    )
    return {
        "event_id": event_id,
        "start_date": start_iso,
        "end_date": end_iso,
        "duration_days": duration_days,
        "detection_count": detection_count,
        "footprint_ha": float(footprint_ha),
        "burned_area_ha": float(burned_area_ha),
        "burned_pixel_count": int(burned_count),
        "polygon": [polygon] if polygon is not None else None,
        "fill_color": event_color(str(event_id)),
        "_polygon_ring": polygon,
    }


def event_color(event_id: str) -> list[int]:
    """Stable RGBA color for an event id."""

    digest = abs(hash(str(event_id)))
    r = 40 + (digest % 180)
    g = 40 + ((digest // 7) % 180)
    b = 40 + ((digest // 13) % 180)
    return [int(r), int(g), int(b), 90]


def _overlap_metrics(polygons: dict) -> tuple[int, float]:
    from shapely.geometry import Polygon

    # Rank by rough bbox area and only compare the largest events.
    scored = []
    for eid, ring in polygons.items():
        if not ring or len(ring) < 3:
            continue
        arr = np.asarray(ring, dtype=float)
        bbox_area = max(1e-12, (arr[:, 0].max() - arr[:, 0].min()) * (arr[:, 1].max() - arr[:, 1].min()))
        scored.append((bbox_area, eid, ring))
    scored.sort(reverse=True)
    scored = scored[:MAX_OVERLAP_EVENTS]
    if len(scored) < 2:
        return 0, 0.0

    geoms = []
    for _area, eid, ring in scored:
        geom = Polygon(ring)
        if not geom.is_valid:
            geom = geom.buffer(0)
        geoms.append((eid, ring, geom))

    overlap_pairs = 0
    overlap_ha = 0.0
    for i, (_eid_a, ring_a, geom_a) in enumerate(geoms):
        for _eid_b, _ring_b, geom_b in geoms[i + 1 :]:
            if not geom_a.bounds or not geom_b.bounds:
                continue
            # Cheap bbox reject.
            minx_a, miny_a, maxx_a, maxy_a = geom_a.bounds
            minx_b, miny_b, maxx_b, maxy_b = geom_b.bounds
            if maxx_a < minx_b or maxx_b < minx_a or maxy_a < miny_b or maxy_b < miny_a:
                continue
            inter = geom_a.intersection(geom_b)
            if inter.is_empty:
                continue
            coords = np.asarray(ring_a, dtype=float)
            lat0 = float(np.mean(coords[:, 1]))
            cos_lat = max(0.2, math.cos(math.radians(lat0)))

            def _area_ha(geom):
                if geom.is_empty:
                    return 0.0
                if geom.geom_type == "Polygon":
                    xs = [c[0] * METERS_PER_DEG_LAT * cos_lat for c in geom.exterior.coords]
                    ys = [c[1] * METERS_PER_DEG_LAT for c in geom.exterior.coords]
                    from shapely.geometry import Polygon as PolyM

                    return float(PolyM(list(zip(xs, ys))).area) / 10_000.0
                if geom.geom_type == "MultiPolygon":
                    return sum(_area_ha(g) for g in geom.geoms)
                return 0.0

            area = _area_ha(inter)
            if area > 0:
                overlap_pairs += 1
                overlap_ha += area
    return overlap_pairs, overlap_ha


def build_event_spread_summary(
    firms: pd.DataFrame,
    burned: pd.DataFrame | None = None,
    as_of: date | None = None,
    max_km: float = DEFAULT_EVENT_MAX_KM,
    max_gap_days: int = DEFAULT_EVENT_GAP_DAYS,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Cluster fires and summarize spread, burned land, and event overlaps.

    Returns:
        clustered detections, per-event summary (with polygon rings), metrics dict
    """

    def _progress(done, total, detail=""):
        if progress_callback is not None:
            progress_callback(done, total, detail)

    _progress(0, 4, "clustering detections")
    clustered = cluster_firms_events(
        firms,
        max_km=max_km,
        max_gap_days=max_gap_days,
        progress_callback=lambda d, t, detail: _progress(0, 4, f"clustering ({d}/{t}) {detail}"),
    )
    if clustered.empty:
        empty_events = pd.DataFrame(
            columns=[
                "event_id",
                "start_date",
                "end_date",
                "duration_days",
                "detection_count",
                "footprint_ha",
                "burned_area_ha",
                "burned_pixel_count",
                "polygon",
                "fill_color",
            ]
        )
        return clustered, empty_events, {
            "event_count": 0,
            "total_footprint_ha": 0.0,
            "total_burned_area_ha": 0.0,
            "overlapping_event_pairs": 0,
            "event_overlap_ha": 0.0,
        }

    as_of_ts = pd.Timestamp(as_of) if as_of is not None else clustered["acq_date"].max()
    active = clustered[clustered["acq_date"] <= as_of_ts].copy()
    if active.empty:
        return build_event_spread_summary(clustered.iloc[0:0], burned, as_of, max_km, max_gap_days, progress_callback)

    burned_work = None
    if burned is not None and not burned.empty:
        _progress(1, 4, "preparing burned pixels")
        burned_work = burned.copy()
        burned_work["latitude"] = pd.to_numeric(burned_work["latitude"], errors="coerce")
        burned_work["longitude"] = pd.to_numeric(burned_work["longitude"], errors="coerce").map(normalize_longitude)
        burned_work["date"] = pd.to_datetime(burned_work["date"], errors="coerce")
        burned_work["burned_area_ha"] = pd.to_numeric(burned_work["burned_area_ha"], errors="coerce").fillna(0.0)
        burned_work = burned_work.dropna(subset=["latitude", "longitude", "date"])
        burned_work = burned_work[burned_work["burned_area_ha"] > 0]
        burned_work = burned_work[burned_work["date"] <= as_of_ts]

    _progress(2, 4, "building event footprints")
    tree, burned_lons, burned_lats, burned_ha = _burned_arrays(burned_work)

    stats = (
        active.groupby("event_id", sort=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            start_date=("acq_date", "min"),
            end_date=("acq_date", "max"),
            detection_count=("acq_date", "size"),
        )
        .reset_index()
    )
    total_events = len(stats)
    single_mask = stats["detection_count"].to_numpy() == 1
    singles = stats.loc[single_mask]
    multis = stats.loc[~single_mask]

    event_rows: list[dict] = []
    polygons: dict = {}

    # Fast path: ~most events are isolated single detections.
    if not singles.empty:
        _progress(2, 4, f"footprints singles 0/{total_events}")
        s_lats = singles["latitude"].to_numpy(dtype=float)
        s_lons = singles["longitude"].to_numpy(dtype=float)
        n_single = len(singles)
        burned_sums = np.zeros(n_single, dtype=float)
        burned_counts = np.zeros(n_single, dtype=np.int64)
        if tree is not None and burned_ha is not None:
            neighbor_lists = tree.query_radius(
                np.radians(np.column_stack((s_lats, s_lons))),
                r=(FOOTPRINT_BUFFER_M / 1000.0) / EARTH_RADIUS_KM,
            )
            for i, idxs in enumerate(neighbor_lists):
                if len(idxs):
                    burned_sums[i] = float(burned_ha[idxs].sum())
                    burned_counts[i] = int(len(idxs))

        cos_lat = np.maximum(0.2, np.cos(np.radians(s_lats)))
        dlon = (FOOTPRINT_BUFFER_M / (METERS_PER_DEG_LAT * cos_lat))[:, None] * _BUFFER_COS[None, :]
        dlat = (FOOTPRINT_BUFFER_M / METERS_PER_DEG_LAT) * _BUFFER_SIN[None, :]
        poly_lons = s_lons[:, None] + dlon
        poly_lats = s_lats[:, None] + dlat

        starts = pd.to_datetime(singles["start_date"]).dt.date
        ends = pd.to_datetime(singles["end_date"]).dt.date
        event_ids = singles["event_id"].to_numpy()
        for i in range(n_single):
            ring = [
                [float(poly_lons[i, j]), float(poly_lats[i, j])]
                for j in range(FOOTPRINT_BUFFER_SEGS)
            ]
            ring.append(ring[0])
            eid = event_ids[i]
            event_rows.append(
                {
                    "event_id": eid,
                    "start_date": starts.iloc[i].isoformat(),
                    "end_date": ends.iloc[i].isoformat(),
                    "duration_days": 1,
                    "detection_count": 1,
                    "footprint_ha": float(_SINGLE_FOOTPRINT_HA),
                    "burned_area_ha": float(burned_sums[i]),
                    "burned_pixel_count": int(burned_counts[i]),
                    "polygon": [ring],
                    "fill_color": event_color(str(eid)),
                }
            )
            polygons[eid] = ring
        _progress(2, 4, f"footprints {len(event_rows)}/{total_events}")

    # Multi-detection events: hull ⊕ buffer (usually a small minority).
    if not multis.empty:
        multi_ids = set(multis["event_id"].astype(str))
        multi_groups = [
            (eid, grp)
            for eid, grp in active.groupby("event_id", sort=False)
            if str(eid) in multi_ids
        ]
        jobs = []
        for event_id, group in multi_groups:
            start = group["acq_date"].min()
            end = group["acq_date"].max()
            jobs.append(
                (
                    event_id,
                    group["longitude"].to_numpy(dtype=float),
                    group["latitude"].to_numpy(dtype=float),
                    start.date().isoformat(),
                    end.date().isoformat(),
                    int((end - start).days) + 1,
                    int(len(group)),
                    tree,
                    burned_lons,
                    burned_lats,
                    burned_ha,
                )
            )
        workers = 1 if len(jobs) < 64 else FOOTPRINT_WORKERS
        done_base = len(event_rows)
        if workers == 1:
            for idx, row in enumerate(map(_summarize_one_event, jobs)):
                ring = row.pop("_polygon_ring", None)
                event_rows.append(row)
                if ring is not None:
                    polygons[row["event_id"]] = ring
                if progress_callback is not None and (idx % 100 == 0 or idx + 1 == len(jobs)):
                    _progress(2, 4, f"footprints {done_base + idx + 1}/{total_events}")
        else:
            chunk = max(32, len(jobs) // max(1, workers * 4))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for start_i in range(0, len(jobs), chunk):
                    batch = jobs[start_i : start_i + chunk]
                    for row in pool.map(_summarize_one_event, batch):
                        ring = row.pop("_polygon_ring", None)
                        event_rows.append(row)
                        if ring is not None:
                            polygons[row["event_id"]] = ring
                    _progress(2, 4, f"footprints {min(done_base + start_i + len(batch), total_events)}/{total_events}")

    events = pd.DataFrame(event_rows)
    if not events.empty:
        events = attach_frp_metrics(events, active)
        events = events.sort_values(
            ["frp_sum", "burned_area_ha", "footprint_ha", "detection_count"],
            ascending=False,
        ).reset_index(drop=True)

    _progress(3, 4, "measuring event overlaps")
    try:
        overlap_pairs, overlap_ha = _overlap_metrics(polygons)
    except Exception:
        overlap_pairs, overlap_ha = 0, 0.0

    metrics = {
        "event_count": int(len(events)),
        "total_footprint_ha": float(events["footprint_ha"].sum()) if not events.empty else 0.0,
        "total_burned_area_ha": float(events["burned_area_ha"].sum()) if not events.empty else 0.0,
        "total_frp_mw": float(events["frp_sum"].sum()) if not events.empty and "frp_sum" in events else 0.0,
        "large_growing_events": int((events.get("growth_class") == "large growing").sum())
        if not events.empty and "growth_class" in events
        else 0,
        "spreading_events": int(events["is_spreading"].sum())
        if not events.empty and "is_spreading" in events
        else 0,
        "fastest_spread_km_per_day": float(events["spread_km_per_day"].max())
        if not events.empty and "spread_km_per_day" in events
        else 0.0,
        "overlapping_event_pairs": int(overlap_pairs),
        "event_overlap_ha": float(overlap_ha),
        "as_of": as_of_ts.date().isoformat(),
    }
    _progress(4, 4, f"{metrics['event_count']:,} events ready")
    return clustered, events, metrics


def event_detections(clustered: pd.DataFrame, event_id: str) -> pd.DataFrame:
    """Return FIRMS detections belonging to one event, sorted by time."""

    if clustered is None or clustered.empty or "event_id" not in clustered.columns:
        return pd.DataFrame()
    work = clustered[clustered["event_id"].astype(str) == str(event_id)].copy()
    if work.empty:
        return work
    work["acq_date"] = pd.to_datetime(work["acq_date"], errors="coerce")
    return work.dropna(subset=["acq_date"]).sort_values("acq_date").reset_index(drop=True)


def build_event_snapshot(
    clustered: pd.DataFrame,
    burned: pd.DataFrame | None,
    event_id: str,
    as_of: date,
) -> dict:
    """Build cumulative FIRMS footprint and burned area for one event as of a day."""

    detections = event_detections(clustered, event_id)
    if detections.empty:
        return {
            "event_id": str(event_id),
            "as_of": as_of.isoformat(),
            "detections": detections,
            "active_detections": detections,
            "burned_inside": pd.DataFrame(),
            "polygon": None,
            "footprint_ha": 0.0,
            "burned_area_ha": 0.0,
            "detection_count": 0,
            "days_since_start": 0,
            "start_date": None,
            "end_date": None,
        }

    start_ts = detections["acq_date"].min()
    end_ts = detections["acq_date"].max()
    as_of_ts = pd.Timestamp(as_of)
    active = detections[detections["acq_date"] <= as_of_ts].copy()
    polygon, footprint_ha = _points_to_polygon(active["longitude"], active["latitude"]) if not active.empty else (None, 0.0)

    burned_inside = pd.DataFrame()
    burned_ha = 0.0
    if burned is not None and not burned.empty and polygon is not None:
        burned_work = burned.copy()
        burned_work["latitude"] = pd.to_numeric(burned_work["latitude"], errors="coerce")
        burned_work["longitude"] = pd.to_numeric(burned_work["longitude"], errors="coerce").map(normalize_longitude)
        burned_work["date"] = pd.to_datetime(burned_work["date"], errors="coerce")
        burned_work["burned_area_ha"] = pd.to_numeric(burned_work["burned_area_ha"], errors="coerce").fillna(0.0)
        burned_work = burned_work.dropna(subset=["latitude", "longitude", "date"])
        burned_work = burned_work[(burned_work["burned_area_ha"] > 0) & (burned_work["date"] <= as_of_ts)]
        if not burned_work.empty:
            tree, blons, blats, bha = _burned_arrays(burned_work)
            if tree is not None:
                arr = np.asarray(polygon, dtype=float)
                lon0 = float(arr[:, 0].mean())
                lat0 = float(arr[:, 1].mean())
                cos_lat = max(0.2, math.cos(math.radians(lat0)))
                width_km = max(0.01, (arr[:, 0].max() - arr[:, 0].min()) * METERS_PER_DEG_LAT * cos_lat / 1000.0)
                height_km = max(0.01, (arr[:, 1].max() - arr[:, 1].min()) * METERS_PER_DEG_LAT / 1000.0)
                radius_km = 0.5 * math.hypot(width_km, height_km) + 1.5
                idxs = tree.query_radius(
                    np.radians([[lat0, lon0]]),
                    r=radius_km / EARTH_RADIUS_KM,
                )[0]
                if len(idxs):
                    subset = burned_work.iloc[idxs]
                    mask = _point_in_polygon_mask(
                        subset["longitude"].to_numpy(dtype=float),
                        subset["latitude"].to_numpy(dtype=float),
                        polygon,
                    )
                    burned_inside = subset.loc[mask].copy().reset_index(drop=True)
            else:
                mask = _point_in_polygon_mask(blons, blats, polygon)
                burned_inside = burned_work.loc[mask].copy().reset_index(drop=True)
            burned_ha = float(burned_inside["burned_area_ha"].sum()) if not burned_inside.empty else 0.0

    if not active.empty:
        active = active.copy()
        active["age_days"] = (as_of_ts.normalize() - active["acq_date"].dt.normalize()).dt.days

    return {
        "event_id": str(event_id),
        "as_of": as_of.isoformat(),
        "detections": detections,
        "active_detections": active,
        "burned_inside": burned_inside,
        "polygon": polygon,
        "footprint_ha": float(footprint_ha),
        "burned_area_ha": float(burned_ha),
        "detection_count": int(len(active)),
        "days_since_start": int((as_of_ts.normalize() - start_ts.normalize()).days),
        "start_date": start_ts.date(),
        "end_date": end_ts.date(),
    }


def build_event_timeline(
    clustered: pd.DataFrame,
    burned: pd.DataFrame | None,
    event_id: str,
) -> pd.DataFrame:
    """Day-by-day growth history for one fire event."""

    detections = event_detections(clustered, event_id)
    if detections.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "days_since_start",
                "new_detections",
                "detection_count",
                "footprint_ha",
                "burned_area_ha",
                "burned_pixel_count",
            ]
        )

    dates = sorted(detections["acq_date"].dt.date.unique().tolist())
    # Include burned days inside the event window so burn growth is visible.
    if burned is not None and not burned.empty:
        start_d, end_d = dates[0], dates[-1]
        burned_dates = pd.to_datetime(burned["date"], errors="coerce").dt.date.dropna()
        extra = sorted({d for d in burned_dates if start_d <= d <= end_d})
        dates = sorted(set(dates).union(extra))

    rows = []
    prev_count = 0
    for day in dates:
        snap = build_event_snapshot(clustered, burned, event_id, day)
        count = int(snap["detection_count"])
        rows.append(
            {
                "date": day.isoformat(),
                "days_since_start": snap["days_since_start"],
                "new_detections": max(0, count - prev_count),
                "detection_count": count,
                "footprint_ha": snap["footprint_ha"],
                "burned_area_ha": snap["burned_area_ha"],
                "burned_pixel_count": int(len(snap["burned_inside"])),
            }
        )
        prev_count = count
    return pd.DataFrame(rows)
