"""FIRMS map viewer UI and data preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from wildfire.features.fire_growth import (
    attach_days_since_ignition,
    build_growth_grid,
    frp_radius_meters,
    growth_front_color,
)
from wildfire.features.fire_spread import (
    DEFAULT_EVENT_GAP_DAYS,
    DEFAULT_EVENT_MAX_KM,
    build_event_snapshot,
    build_event_spread_summary,
    build_event_timeline,
    event_color,
    event_detections,
)
from wildfire.features.link_fires_burned import DEFAULT_DAY_TOLERANCE, DEFAULT_MAX_KM, link_firms_to_burned
from wildfire.ingest.mcd64a1 import burned_area_polygons_path


# Detailed hull polygons for the largest fires only (keeps WebSocket payload manageable).
MAX_DETAIL_EVENT_POLYGONS = 1500
# Visual radius: meters ≈ scale * sqrt(ha); clipped so tiny fires stay visible.
EVENT_SIZE_RADIUS_SCALE = 95.0
EVENT_SIZE_RADIUS_MIN_M = 500.0
EVENT_SIZE_RADIUS_MAX_M = 80_000.0


def _polygon_centroid_lonlat(polygon) -> tuple[float, float] | None:
    """Centroid of the first ring in a pydeck polygon value."""

    try:
        ring = polygon[0] if polygon and isinstance(polygon[0][0], (list, tuple)) else polygon
        lons = [float(pt[0]) for pt in ring]
        lats = [float(pt[1]) for pt in ring]
        if not lons:
            return None
        return float(np.mean(lons)), float(np.mean(lats))
    except Exception:
        return None


def prepare_event_map_layers(
    event_polygons: pd.DataFrame,
    max_detail_polygons: int = MAX_DETAIL_EVENT_POLYGONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build size-scaled markers for every event, plus a capped set of detail hulls.

    Returns:
        (event_markers, detail_polygons)
    """

    if event_polygons is None or event_polygons.empty:
        empty = pd.DataFrame()
        return empty, empty

    work = event_polygons.dropna(subset=["polygon"]).copy()
    if work.empty:
        return work, work

    if "footprint_ha" not in work.columns:
        work["footprint_ha"] = 0.0
    else:
        work["footprint_ha"] = pd.to_numeric(work["footprint_ha"], errors="coerce").fillna(0.0)
    if "burned_area_ha" not in work.columns:
        work["burned_area_ha"] = 0.0
    else:
        work["burned_area_ha"] = pd.to_numeric(work["burned_area_ha"], errors="coerce").fillna(0.0)
    if "detection_count" not in work.columns:
        work["detection_count"] = 1
    else:
        work["detection_count"] = pd.to_numeric(work["detection_count"], errors="coerce").fillna(1).astype(int)
    work["size_ha"] = work[["footprint_ha", "burned_area_ha"]].max(axis=1).clip(lower=1.0)
    if "frp_sum" not in work.columns:
        work["frp_sum"] = 0.0
    else:
        work["frp_sum"] = pd.to_numeric(work["frp_sum"], errors="coerce").fillna(0.0)
    work["radius"] = [
        frp_radius_meters(frp, foot)
        for frp, foot in zip(work["frp_sum"].to_numpy(), work["size_ha"].to_numpy())
    ]

    centroids = work["polygon"].map(_polygon_centroid_lonlat)
    work["longitude"] = centroids.map(lambda c: c[0] if c else np.nan)
    work["latitude"] = centroids.map(lambda c: c[1] if c else np.nan)
    markers = work.dropna(subset=["latitude", "longitude"]).copy()
    if "fill_color" not in markers.columns:
        markers["fill_color"] = markers["event_id"].map(lambda eid: event_color(str(eid)))
    # Stronger fill for size markers so fires read clearly at continental zoom.
    markers["marker_color"] = markers["fill_color"].map(
        lambda rgba: list(rgba[:3]) + [200] if isinstance(rgba, (list, tuple)) else [255, 120, 40, 200]
    )
    if "end_date" in markers.columns:
        markers["date"] = markers["end_date"].astype(str)
    elif "start_date" in markers.columns:
        markers["date"] = markers["start_date"].astype(str)
    else:
        markers["date"] = ""
    markers["line_color"] = [[255, 255, 255, 160]] * len(markers)

    detail = (
        markers.sort_values(["size_ha", "detection_count"], ascending=False)
        .head(int(max_detail_polygons))
        .copy()
    )
    return markers, detail


def _extract_selected_event_id(selection) -> str | None:
    """Pull an event_id from a Streamlit pydeck selection payload."""

    if selection is None:
        return None
    payload = getattr(selection, "selection", selection)
    if payload is None:
        return None
    objects = None
    if isinstance(payload, dict):
        objects = payload.get("objects")
    else:
        objects = getattr(payload, "objects", None)
    if not objects:
        return None

    candidates = []
    if isinstance(objects, dict):
        for values in objects.values():
            if isinstance(values, list):
                candidates.extend(values)
            elif values is not None:
                candidates.append(values)
    elif isinstance(objects, list):
        candidates.extend(objects)

    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        event_id = obj.get("event_id")
        if event_id is not None and str(event_id).strip():
            return str(event_id)
    return None


def prepare_firms_map_points(
    csv_path: str | Path,
    start: date | None = None,
    end: date | None = None,
    min_confidence: float | None = None,
    max_points: int | None = None,
) -> pd.DataFrame:
    """Load and filter FIRMS detections for map rendering."""

    df = pd.read_csv(csv_path, low_memory=False)
    required = {"latitude", "longitude", "acq_date"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("FIRMS CSV is missing columns: " + ", ".join(sorted(missing)))

    work = df.copy()
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["acq_date"] = pd.to_datetime(work["acq_date"], errors="coerce")
    if "confidence" in work:
        work["confidence"] = pd.to_numeric(work["confidence"], errors="coerce")
    if "brightness" in work:
        work["brightness"] = pd.to_numeric(work["brightness"], errors="coerce")
    if "frp" in work:
        work["frp"] = pd.to_numeric(work["frp"], errors="coerce").fillna(0.0)
    else:
        work["frp"] = 0.0
    work = work.dropna(subset=["latitude", "longitude", "acq_date"])
    if start:
        work = work[work["acq_date"].dt.date >= start]
    if end:
        work = work[work["acq_date"].dt.date <= end]
    if min_confidence is not None and "confidence" in work:
        work = work[work["confidence"] >= float(min_confidence)]
    work = work.sort_values("acq_date", ascending=False)
    if max_points and max_points > 0 and len(work) > max_points:
        work = work.head(int(max_points))
    return work.reset_index(drop=True)


def available_fire_dates(points: pd.DataFrame) -> list[date]:
    """Return sorted unique acquisition dates from prepared FIRMS points."""

    if points.empty:
        return []
    return sorted(points["acq_date"].dt.date.dropna().unique().tolist())


def available_burned_dates(points: pd.DataFrame) -> list[date]:
    """Return sorted unique burn dates from prepared burned-area points."""

    if points is None or points.empty or "date" not in points.columns:
        return []
    return sorted(points["date"].dt.date.dropna().unique().tolist())


def available_timeline_dates(
    firms_points: pd.DataFrame,
    burned_points: pd.DataFrame | None = None,
) -> list[date]:
    """Union of FIRMS and burned-area dates for synced timelapse frames."""

    dates = set(available_fire_dates(firms_points))
    dates.update(available_burned_dates(burned_points))
    return sorted(dates)


def filter_points_for_frame(points: pd.DataFrame, frame_date: date, trailing_days: int = 1) -> pd.DataFrame:
    """Keep detections active for a single timelapse frame."""

    if points.empty:
        return points
    trailing_days = max(1, int(trailing_days))
    frame_ts = pd.Timestamp(frame_date)
    start_ts = frame_ts - pd.Timedelta(days=trailing_days)
    mask = (points["acq_date"] >= start_ts) & (points["acq_date"] <= frame_ts)
    frame_points = points.loc[mask].copy()
    frame_points["age_days"] = (frame_ts.normalize() - frame_points["acq_date"].dt.normalize()).dt.days
    frame_points = frame_points[(frame_points["age_days"] >= 0) & (frame_points["age_days"] <= trailing_days)]
    return frame_points.reset_index(drop=True)


def prepare_burned_area_map_points(
    csv_path: str | Path,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Load extracted MCD64A1 burned-area pixels for map rendering."""

    df = pd.read_csv(csv_path, low_memory=False)
    required = {"latitude", "longitude", "date", "burned_area_ha"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Burned-area CSV is missing columns: " + ", ".join(sorted(missing)))

    work = df.copy()
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["burned_area_ha"] = pd.to_numeric(work["burned_area_ha"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["latitude", "longitude", "date"])
    work = work[work["burned_area_ha"] > 0].copy()
    if start:
        work = work[work["date"].dt.date >= start]
    if end:
        work = work[work["date"].dt.date <= end]
    work["burned_area_radius"] = work["burned_area_ha"].clip(lower=1).pow(0.5) * 120
    work["burned_area_color"] = [[85, 45, 10, 135]] * len(work)
    return work.reset_index(drop=True)


def prepare_burned_area_map_polygons(
    polygons_json: str | Path,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Load vectorized MCD64A1 burned-area scar borders for PolygonLayer rendering."""

    path = Path(polygons_json)
    if not path.is_file():
        return pd.DataFrame(columns=["date", "burned_area_ha", "polygon", "fill_color"])

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "features" in payload:
        rows = []
        for feat in payload.get("features") or []:
            props = feat.get("properties") or {}
            geom = feat.get("geometry") or {}
            if geom.get("type") == "Polygon":
                rows.append(
                    {
                        "date": props.get("date", ""),
                        "burned_area_ha": props.get("burned_area_ha", 0.0),
                        "polygon": geom.get("coordinates") or [],
                        "fill_color": props.get("fill_color") or [120, 55, 15, 140],
                    }
                )
            elif geom.get("type") == "MultiPolygon":
                for poly in geom.get("coordinates") or []:
                    rows.append(
                        {
                            "date": props.get("date", ""),
                            "burned_area_ha": props.get("burned_area_ha", 0.0),
                            "polygon": poly,
                            "fill_color": props.get("fill_color") or [120, 55, 15, 140],
                        }
                    )
        work = pd.DataFrame(rows)
    else:
        work = pd.DataFrame(payload)

    if work.empty:
        return work

    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    if "burned_area_ha" not in work.columns:
        work["burned_area_ha"] = 0.0
    else:
        work["burned_area_ha"] = pd.to_numeric(work["burned_area_ha"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["date"])
    work = work[work["polygon"].notna()].copy()
    if start:
        work = work[work["date"].dt.date >= start]
    if end:
        work = work[work["date"].dt.date <= end]
    if "fill_color" not in work.columns:
        work["fill_color"] = [[120, 55, 15, 140]] * len(work)
    work["date_label"] = work["date"].dt.strftime("%Y-%m-%d")
    return work.reset_index(drop=True)


def filter_burned_area_for_frame(points: pd.DataFrame, frame_date: date, trailing_days: int = 1) -> pd.DataFrame:
    """Keep burned pixels active for a single timelapse frame (same window as FIRMS)."""

    if points.empty:
        return points
    trailing_days = max(1, int(trailing_days))
    frame_ts = pd.Timestamp(frame_date)
    start_ts = frame_ts - pd.Timedelta(days=trailing_days)
    mask = (points["date"] >= start_ts) & (points["date"] <= frame_ts)
    frame_points = points.loc[mask].copy()
    frame_points["age_days"] = (frame_ts.normalize() - frame_points["date"].dt.normalize()).dt.days
    frame_points = frame_points[(frame_points["age_days"] >= 0) & (frame_points["age_days"] <= trailing_days)]
    frame_points["burned_area_color"] = frame_points["age_days"].apply(FirmsMapViewer._pydeck_burned_age_color)
    if "fill_color" in frame_points.columns:
        frame_points["fill_color"] = frame_points["age_days"].apply(FirmsMapViewer._pydeck_burned_age_color)
    return frame_points.reset_index(drop=True)


@dataclass
class FirmsMapViewer:
    """Render downloaded FIRMS detections on an interactive map."""

    fast_cluster_threshold: int = 2000

    def render_map(
        self,
        points: pd.DataFrame,
        render_mode: str = "fast",
        burned_area_points: pd.DataFrame | None = None,
        burned_area_polygons: pd.DataFrame | None = None,
        matched_only: bool = False,
        event_polygons: pd.DataFrame | None = None,
        color_mode: str = "auto",
        show_growth_grid: bool = False,
    ):
        linked_points = points
        if burned_area_points is not None and not burned_area_points.empty and not points.empty:
            linked_points = link_firms_to_burned(
                points,
                burned_area_points,
                max_km=DEFAULT_MAX_KM,
                day_tolerance=DEFAULT_DAY_TOLERANCE,
            )
            if "age_days" in points.columns and len(linked_points) == len(points):
                linked_points["age_days"] = points["age_days"].to_numpy()
            if "event_id" in points.columns and len(linked_points) == len(points):
                linked_points["event_id"] = points["event_id"].to_numpy()
            if "days_since_ignition" in points.columns and len(linked_points) == len(points):
                linked_points["days_since_ignition"] = points["days_since_ignition"].to_numpy()
            if "frp" in points.columns and len(linked_points) == len(points):
                linked_points["frp"] = points["frp"].to_numpy()
            self._show_link_metrics(linked_points, burned_area_points)
            if matched_only:
                linked_points = linked_points[linked_points["matched_burned"]].copy()
                if linked_points.empty:
                    st.warning("No FIRMS detections matched burned area in this view.")
                    return linked_points

        if render_mode == "pydeck":
            return self.render_pydeck_map(
                linked_points,
                burned_area_points=burned_area_points,
                burned_area_polygons=burned_area_polygons,
                event_polygons=event_polygons,
                enable_event_selection=event_polygons is not None and not event_polygons.empty,
                color_mode=color_mode,
                show_growth_grid=show_growth_grid,
            )

        try:
            import folium
            from folium.plugins import FastMarkerCluster, MarkerCluster
            from streamlit_folium import st_folium
        except ImportError as exc:
            raise RuntimeError("Install streamlit-folium to use the map viewer.") from exc

        if linked_points.empty:
            st.warning("No FIRMS points to display.")
            return linked_points

        center = [float(linked_points["latitude"].mean()), float(linked_points["longitude"].mean())]
        fmap = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap")

        if render_mode == "fast" or len(linked_points) > self.fast_cluster_threshold:
            locations = linked_points[["latitude", "longitude"]].astype(float).values.tolist()
            FastMarkerCluster(locations, name="FIRMS detections").add_to(fmap)
            st.caption(
                f"Using fast cluster mode for {len(linked_points):,} points. "
                "Popups are disabled in this mode to keep the browser responsive."
            )
        else:
            layer = (
                MarkerCluster(name="FIRMS detections")
                if render_mode == "clustered_popups"
                else folium.FeatureGroup(name="FIRMS detections")
            )
            for _, row in linked_points.iterrows():
                folium.CircleMarker(
                    location=[float(row["latitude"]), float(row["longitude"])],
                    radius=4,
                    color=self._marker_color(row),
                    fill=True,
                    fill_opacity=0.75,
                    popup="<br>".join(self._popup_lines(row)),
                ).add_to(layer)
            layer.add_to(fmap)

        folium.LayerControl().add_to(fmap)
        return st_folium(
            fmap,
            height=650,
            key="firms_map_viewer",
            returned_objects=[],
            use_container_width=True,
        )

    def render_pydeck_map(
        self,
        points: pd.DataFrame,
        burned_area_points: pd.DataFrame | None = None,
        burned_area_polygons: pd.DataFrame | None = None,
        event_polygons: pd.DataFrame | None = None,
        enable_event_selection: bool = False,
        view_state: dict | None = None,
        chart_key: str = "firms_map_viewer_deck",
        color_mode: str = "auto",
        show_growth_grid: bool = False,
    ):
        try:
            import pydeck as pdk
        except ImportError as exc:
            raise RuntimeError("Install pydeck to use the native Streamlit map viewer.") from exc

        has_events = event_polygons is not None and not event_polygons.empty
        has_burn_polys = burned_area_polygons is not None and not burned_area_polygons.empty
        if (
            points.empty
            and (burned_area_points is None or burned_area_points.empty)
            and not has_events
            and not has_burn_polys
        ):
            st.warning("Nothing to plot for this frame.")
            return None

        map_df = points.copy() if not points.empty else pd.DataFrame()
        layers = []
        view_lat = None
        view_lon = None
        view_zoom = 5

        if has_burn_polys:
            burn_poly_df = burned_area_polygons.dropna(subset=["polygon"]).copy()
            if not burn_poly_df.empty:
                if "fill_color" not in burn_poly_df.columns:
                    burn_poly_df["fill_color"] = [[120, 55, 15, 140]] * len(burn_poly_df)
                if "date_label" not in burn_poly_df.columns:
                    burn_poly_df["date_label"] = pd.to_datetime(
                        burn_poly_df["date"], errors="coerce"
                    ).dt.strftime("%Y-%m-%d")
                burn_poly_layer = pdk.Layer(
                    "PolygonLayer",
                    data=burn_poly_df,
                    id="burned_scars",
                    get_polygon="polygon",
                    get_fill_color="fill_color",
                    get_line_color=[255, 220, 160, 220],
                    line_width_min_pixels=2,
                    pickable=True,
                    stroked=True,
                    filled=True,
                    extruded=False,
                    opacity=0.55,
                )
                layers.append(burn_poly_layer)
                st.metric(
                    "Total burned area in view",
                    f"{burn_poly_df['burned_area_ha'].sum():,.1f} ha",
                )
                st.caption(
                    f"{len(burn_poly_df):,} burned-area scar polygon(s) with borders "
                    "(MCD64A1 burn-date dissolve)."
                )
                first_ring = burn_poly_df.iloc[0]["polygon"][0]
                view_lon = float(np.mean([pt[0] for pt in first_ring]))
                view_lat = float(np.mean([pt[1] for pt in first_ring]))

        if has_events:
            event_markers, detail_polys = prepare_event_map_layers(event_polygons)
            if not detail_polys.empty:
                poly_layer = pdk.Layer(
                    "PolygonLayer",
                    data=detail_polys,
                    id="event_footprints",
                    get_polygon="polygon",
                    get_fill_color="fill_color",
                    get_line_color=[255, 255, 255, 180],
                    line_width_min_pixels=2,
                    pickable=True,
                    auto_highlight=True,
                    stroked=True,
                    filled=True,
                    extruded=False,
                    opacity=0.35,
                )
                layers.append(poly_layer)
            if not event_markers.empty:
                size_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=event_markers,
                    id="event_sizes",
                    get_position="[longitude, latitude]",
                    get_fill_color="marker_color",
                    get_radius="radius",
                    pickable=True,
                    auto_highlight=True,
                    opacity=0.55,
                    stroked=True,
                    get_line_color="line_color",
                    line_width_min_pixels=1,
                    radius_min_pixels=3,
                    radius_max_pixels=80,
                )
                layers.append(size_layer)
                st.metric("Fire events in view", f"{len(event_markers):,}")
                st.caption(
                    f"Every fire event is drawn; circle size ∝ √max(footprint, burned ha). "
                    f"Detailed outlines shown for the largest {len(detail_polys):,} events."
                )
                view_lat = float(event_markers["latitude"].mean())
                view_lon = float(event_markers["longitude"].mean())
            elif not map_df.empty:
                view_lat = float(map_df["latitude"].mean())
                view_lon = float(map_df["longitude"].mean())

        # Pixel scatter is a fallback / dense overlay; skip when scar polygons are present
        # to keep map payloads smaller and show continuous burned area instead.
        if (
            not has_burn_polys
            and burned_area_points is not None
            and not burned_area_points.empty
        ):
            burned_df = burned_area_points.copy()
            if "burned_area_radius" not in burned_df:
                burned_df["burned_area_radius"] = burned_df["burned_area_ha"].clip(lower=1).pow(0.5) * 120
            burned_df["date_label"] = pd.to_datetime(burned_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            if "age_days" not in burned_df:
                burned_df["age_days"] = 0
            burned_df["burned_area_color"] = burned_df["age_days"].apply(self._pydeck_burned_age_color)
            burned_layer = pdk.Layer(
                "ScatterplotLayer",
                data=burned_df,
                id="burned_pixels",
                get_position="[longitude, latitude]",
                get_fill_color="burned_area_color",
                get_radius="burned_area_radius",
                pickable=False,
                opacity=0.45,
                stroked=True,
                get_line_color=[255, 255, 255, 120],
                line_width_min_pixels=1,
            )
            layers.append(burned_layer)
            st.metric("Total burned area in view", f"{burned_df['burned_area_ha'].sum():,.1f} ha")
            view_lat = float(burned_df["latitude"].mean())
            view_lon = float(burned_df["longitude"].mean())

        if not map_df.empty:
            if "confidence" not in map_df.columns:
                map_df["confidence"] = 0.0
            else:
                map_df["confidence"] = pd.to_numeric(map_df["confidence"], errors="coerce").fillna(0.0)
            if "brightness" not in map_df.columns:
                map_df["brightness"] = 0.0
            else:
                map_df["brightness"] = pd.to_numeric(map_df["brightness"], errors="coerce").fillna(0.0)
            if "frp" not in map_df.columns:
                map_df["frp"] = 0.0
            else:
                map_df["frp"] = pd.to_numeric(map_df["frp"], errors="coerce").fillna(0.0)
            use_growth = color_mode == "growth_front" or (
                color_mode == "auto" and "days_since_ignition" in map_df.columns
            )
            if use_growth:
                if "days_since_ignition" not in map_df.columns and "event_id" in map_df.columns:
                    map_df = attach_days_since_ignition(map_df)
                max_age = max(1, int(map_df.get("days_since_ignition", pd.Series([0])).max() or 1))
                map_df["color"] = map_df["days_since_ignition"].map(
                    lambda d: growth_front_color(d, max_days=max_age)
                )
                # Point size from FRP so intense pixels read hotter/larger on the front.
                map_df["radius"] = (np.sqrt(map_df["frp"].clip(lower=1.0)) * 90.0).clip(350.0, 12_000.0)
                st.caption(
                    "Growth front coloring: cool = ignition core (older), hot = advancing front (newer). "
                    "Point size ∝ √FRP (MW)."
                )
            else:
                conf_scale = 8 if has_events else 20
                map_df["radius"] = (map_df["confidence"].clip(lower=10, upper=100) * conf_scale).astype(float)
            if "matched_burned" not in map_df.columns:
                map_df["matched_burned"] = False
            map_df["matched_burned"] = map_df["matched_burned"].fillna(False).astype(bool)
            if "nearest_burn_km" not in map_df.columns:
                map_df["nearest_burn_km"] = np.nan
            else:
                map_df["nearest_burn_km"] = pd.to_numeric(map_df["nearest_burn_km"], errors="coerce")
            if "matched_burned_area_ha" not in map_df.columns:
                map_df["matched_burned_area_ha"] = 0.0
            else:
                map_df["matched_burned_area_ha"] = pd.to_numeric(
                    map_df["matched_burned_area_ha"], errors="coerce"
                ).fillna(0.0)
            if "color" not in map_df.columns:
                if "event_id" in map_df and map_df["event_id"].notna().any():
                    map_df["color"] = map_df["event_id"].map(
                        lambda eid: event_color(str(eid))[:3] + [160]
                    )
                elif "age_days" in map_df:
                    map_df["color"] = [
                        self._pydeck_matched_color(age) if matched else self._pydeck_age_color(age)
                        for age, matched in zip(map_df["age_days"], map_df["matched_burned"])
                    ]
                else:
                    map_df["color"] = [
                        self._pydeck_matched_color(0) if matched else self._pydeck_color(conf)
                        for conf, matched in zip(map_df["confidence"], map_df["matched_burned"])
                    ]
            map_df["date"] = map_df["acq_date"].dt.strftime("%Y-%m-%d")
            map_df["line_color"] = map_df["matched_burned"].map(
                lambda matched: [255, 215, 0, 220] if matched else [0, 0, 0, 0]
            )
            if "event_id" not in map_df:
                map_df["event_id"] = ""
            if "footprint_ha" not in map_df.columns:
                map_df["footprint_ha"] = ""
            if "burned_area_ha" not in map_df.columns:
                map_df["burned_area_ha"] = ""
            if "detection_count" not in map_df.columns:
                map_df["detection_count"] = ""
            if "size_ha" not in map_df.columns:
                map_df["size_ha"] = ""
            if "days_since_ignition" not in map_df.columns:
                map_df["days_since_ignition"] = ""
            if show_growth_grid:
                grid = build_growth_grid(map_df, resolution_deg=0.01)
                if not grid.empty:
                    max_day = grid["acq_date"].max()
                    grid["age_days"] = (max_day - grid["acq_date"]).dt.days
                    max_age = max(1, int(grid["age_days"].max() or 1))
                    # Invert: newest cells are the front (age 0).
                    grid["color"] = grid["age_days"].map(
                        lambda a: growth_front_color(max_age - int(a), max_days=max_age)
                    )
                    grid_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=grid,
                        id="growth_grid",
                        get_position="[grid_lon, grid_lat]",
                        get_fill_color="color",
                        get_radius="radius",
                        pickable=True,
                        opacity=0.45,
                        stroked=True,
                        get_line_color=[255, 255, 255, 100],
                        line_width_min_pixels=1,
                    )
                    layers.append(grid_layer)
                    st.caption(
                        f"Growth grid: {len(grid):,} ~1 km cells aggregated by day "
                        f"(color = expansion age, weight uses FRP)."
                    )
            fire_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                id="firms_points",
                get_position="[longitude, latitude]",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                auto_highlight=True,
                opacity=0.55 if use_growth else (0.35 if has_events else 0.7),
                stroked=True,
                get_line_color="line_color",
                line_width_min_pixels=2,
            )
            layers.append(fire_layer)
            if view_lat is None:
                view_lat = float(map_df["latitude"].mean())
                view_lon = float(map_df["longitude"].mean())

        if view_state:
            view_lat = float(view_state.get("latitude", view_lat or 40.0))
            view_lon = float(view_state.get("longitude", view_lon or -100.0))
            view_zoom = float(view_state.get("zoom", view_zoom))

        deck_view = pdk.ViewState(
            latitude=view_lat if view_lat is not None else 40.0,
            longitude=view_lon if view_lon is not None else -100.0,
            zoom=view_zoom,
            pitch=0,
        )
        tooltip = {
            "html": (
                "<b>Event:</b> {event_id}<br/>"
                "<b>Date:</b> {date}<br/>"
                "<b>Days since ignition:</b> {days_since_ignition}<br/>"
                "<b>FRP (MW):</b> {frp}<br/>"
                "<b>Detections:</b> {detection_count}<br/>"
                "<b>Size ha:</b> {size_ha}<br/>"
                "<b>Footprint ha:</b> {footprint_ha}<br/>"
                "<b>Burned ha:</b> {burned_area_ha}<br/>"
                "<b>Lat/Lon:</b> {latitude}, {longitude}<br/>"
                "<b>Confidence:</b> {confidence}<br/>"
                "<b>Matched burned:</b> {matched_burned}<br/>"
                "<i>Click a fire circle to inspect its story</i>"
            )
        }
        if enable_event_selection:
            st.caption("Click a sized fire circle (or outline) to inspect how that fire grew and burned.")
        else:
            st.caption("Using native Streamlit/PyDeck rendering to avoid Folium iframe warnings.")

        chart = st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=deck_view,
                layers=layers,
                tooltip=tooltip,
            ),
            use_container_width=True,
            key=chart_key,
            on_select="rerun" if enable_event_selection else "ignore",
            selection_mode="single-object",
        )
        if enable_event_selection:
            selected_id = _extract_selected_event_id(chart)
            if selected_id:
                st.session_state["selected_event_id"] = selected_id
        return chart

    def render(
        self,
        firms_csv: str | Path,
        start: date,
        end: date,
        min_confidence: float,
        max_points: int | None,
        render_mode: str = "fast",
        burned_area_csv: str | Path | None = None,
        matched_only: bool = False,
        analyze_spread: bool = False,
        progress=None,
        color_mode: str = "auto",
        show_growth_grid: bool = False,
        **legacy_kwargs,
    ):
        if "use_cluster" in legacy_kwargs and render_mode == "fast":
            render_mode = "clustered_popups" if legacy_kwargs["use_cluster"] else "individual_popups"
        if not Path(firms_csv).is_file():
            raise ValueError("Provide a valid FIRMS CSV.")
        if progress is not None:
            progress.advance("reading FIRMS CSV")
        points = prepare_firms_map_points(
            firms_csv,
            start=start,
            end=end,
            min_confidence=min_confidence or None,
            max_points=max_points,
        )

        burned_points = None
        burned_polygons = None
        if burned_area_csv and Path(burned_area_csv).is_file():
            if progress is not None:
                progress.advance("reading burned-area CSV")
            burned_points = prepare_burned_area_map_points(burned_area_csv, start=start, end=end)
            poly_path = burned_area_polygons_path(burned_area_csv)
            burned_polygons = prepare_burned_area_map_polygons(poly_path, start=start, end=end)
            if burned_polygons is not None and not burned_polygons.empty:
                st.info(
                    f"Loaded {len(burned_polygons):,} burned-area scar polygon(s) "
                    f"and {len(burned_points):,} pixel(s) for linking."
                )
            else:
                st.info(
                    f"Loaded {len(burned_points):,} burned-area point(s). "
                    "Re-extract MCD64A1 to also build scar border polygons."
                )

        if (
            points.empty
            and (burned_points is None or burned_points.empty)
            and (burned_polygons is None or burned_polygons.empty)
        ):
            st.warning("No FIRMS detections or burned-area pixels matched the selected filters.")
            if progress is not None:
                progress.complete("Nothing to map")
            return points

        if not points.empty:
            st.success(f"Loaded {len(points):,} FIRMS point(s).")

        event_polygons = None
        if analyze_spread and not points.empty:
            if progress is not None:
                progress.advance(f"clustering {len(points):,} detections")
            clustered, event_polygons, metrics = build_event_spread_summary(
                points,
                burned_points,
                as_of=end,
                max_km=DEFAULT_EVENT_MAX_KM,
                max_gap_days=DEFAULT_EVENT_GAP_DAYS,
                progress_callback=progress.tick if progress is not None else None,
            )
            merge_pts = points.copy()
            merge_pts["acq_date"] = pd.to_datetime(merge_pts["acq_date"]).dt.normalize()
            clustered = clustered.copy()
            clustered["acq_date"] = pd.to_datetime(clustered["acq_date"]).dt.normalize()
            points = merge_pts.merge(
                clustered[["latitude", "longitude", "acq_date", "event_id"]],
                on=["latitude", "longitude", "acq_date"],
                how="left",
            )
            points = attach_days_since_ignition(points)
            self._store_event_story_context(clustered, burned_points, event_polygons)
            self._show_spread_metrics(metrics, event_polygons)

        if progress is not None:
            progress.advance("drawing map layers")
        self.render_map(
            points,
            render_mode=render_mode,
            burned_area_points=burned_points,
            burned_area_polygons=burned_polygons,
            matched_only=matched_only,
            event_polygons=event_polygons,
            color_mode=color_mode,
            show_growth_grid=show_growth_grid,
        )
        if analyze_spread and event_polygons is not None and not event_polygons.empty:
            self.render_event_inspector()
        if progress is not None:
            progress.complete("Map ready")
        if not points.empty:
            st.dataframe(points.head(500), width="stretch")
        return points

    def render_timelapse(
        self,
        firms_csv: str | Path,
        start: date,
        end: date,
        min_confidence: float,
        max_points: int | None,
        frame_index: int,
        trailing_days: int = 1,
        render_mode: str = "pydeck",
        burned_area_csv: str | Path | None = None,
        matched_only: bool = False,
        analyze_spread: bool = False,
        progress=None,
        color_mode: str = "auto",
        show_growth_grid: bool = False,
    ) -> pd.DataFrame:
        if not Path(firms_csv).is_file():
            raise ValueError("Provide a valid FIRMS CSV.")
        if progress is not None:
            progress.advance("reading FIRMS CSV")
        points = prepare_firms_map_points(
            firms_csv,
            start=start,
            end=end,
            min_confidence=min_confidence or None,
            max_points=max_points,
        )
        burned_points = None
        burned_polygons = None
        if burned_area_csv and Path(burned_area_csv).is_file():
            if progress is not None:
                progress.advance("reading burned-area CSV")
            burned_points = prepare_burned_area_map_points(burned_area_csv, start=start, end=end)
            burned_polygons = prepare_burned_area_map_polygons(
                burned_area_polygons_path(burned_area_csv),
                start=start,
                end=end,
            )

        dates = available_timeline_dates(points, burned_points)
        if not dates and (burned_polygons is None or burned_polygons.empty):
            st.warning("No FIRMS detections or burned-area dates matched the selected filters.")
            if progress is not None:
                progress.complete("Nothing to map")
            return points
        if not dates:
            dates = sorted(pd.to_datetime(burned_polygons["date"]).dt.date.unique().tolist())

        safe_index = max(0, min(int(frame_index), len(dates) - 1))
        frame_date = dates[safe_index]
        frame_points = (
            filter_points_for_frame(points, frame_date, trailing_days=trailing_days)
            if not points.empty
            else points
        )
        burned_frame = None
        if burned_points is not None and not burned_points.empty:
            burned_frame = filter_burned_area_for_frame(
                burned_points,
                frame_date,
                trailing_days=trailing_days,
            )
        burned_poly_frame = None
        if burned_polygons is not None and not burned_polygons.empty:
            burned_poly_frame = filter_burned_area_for_frame(
                burned_polygons,
                frame_date,
                trailing_days=trailing_days,
            )

        st.success(
            f"Frame {safe_index + 1:,}/{len(dates):,}: {frame_date.isoformat()} "
            f"({len(frame_points):,} fire point(s)"
            + (f", {len(burned_poly_frame):,} burned scar(s)" if burned_poly_frame is not None else "")
            + (f", {len(burned_frame):,} burned pixel(s)" if burned_frame is not None and burned_poly_frame is None else "")
            + ")"
        )
        if (
            frame_points.empty
            and (burned_frame is None or burned_frame.empty)
            and (burned_poly_frame is None or burned_poly_frame.empty)
        ):
            st.warning("No detections or burned pixels are active in this frame.")
            if progress is not None:
                progress.complete("Empty frame")
            return frame_points

        event_polygons = None
        if analyze_spread and not points.empty:
            if progress is not None:
                progress.advance(f"clustering through {frame_date.isoformat()}")
            clustered, event_polygons, metrics = build_event_spread_summary(
                points,
                burned_points,
                as_of=frame_date,
                max_km=DEFAULT_EVENT_MAX_KM,
                max_gap_days=DEFAULT_EVENT_GAP_DAYS,
                progress_callback=progress.tick if progress is not None else None,
            )
            if not frame_points.empty:
                merge_pts = frame_points.copy()
                merge_pts["acq_date"] = pd.to_datetime(merge_pts["acq_date"]).dt.normalize()
                clustered = clustered.copy()
                clustered["acq_date"] = pd.to_datetime(clustered["acq_date"]).dt.normalize()
                frame_points = merge_pts.merge(
                    clustered[["latitude", "longitude", "acq_date", "event_id"]],
                    on=["latitude", "longitude", "acq_date"],
                    how="left",
                )
                frame_points = attach_days_since_ignition(frame_points)
            self._store_event_story_context(clustered, burned_points, event_polygons)
            self._show_spread_metrics(metrics, event_polygons)

        if progress is not None:
            progress.advance("drawing map layers")
        self.render_map(
            frame_points,
            render_mode=render_mode,
            burned_area_points=burned_frame,
            burned_area_polygons=burned_poly_frame,
            matched_only=matched_only,
            event_polygons=event_polygons,
            color_mode=color_mode,
            show_growth_grid=show_growth_grid,
        )
        if analyze_spread and event_polygons is not None and not event_polygons.empty:
            self.render_event_inspector()
        if progress is not None:
            progress.complete("Map ready")
        if not frame_points.empty:
            st.dataframe(frame_points.head(500), width="stretch")
        return frame_points

    @staticmethod
    def _store_event_story_context(
        clustered: pd.DataFrame,
        burned_points: pd.DataFrame | None,
        events: pd.DataFrame | None,
    ) -> None:
        st.session_state["event_story_clustered"] = clustered
        st.session_state["event_story_burned"] = burned_points
        st.session_state["event_story_events"] = events

    def render_event_inspector(self) -> None:
        """Inspect one fire event from first FIRMS detection through growth/burn."""

        clustered = st.session_state.get("event_story_clustered")
        burned = st.session_state.get("event_story_burned")
        events = st.session_state.get("event_story_events")
        if clustered is None or events is None or events.empty:
            return

        st.subheader("Event story")
        st.caption(
            "Select an event (or click it on the map) and scrub the day slider to watch "
            "FIRMS detections accumulate, the footprint grow, and MCD64 burned area fill in."
        )

        show_cols = [
            c
            for c in [
                "event_id",
                "spread_status",
                "spread_km_per_day",
                "footprint_ha_per_day",
                "growth_class",
                "start_date",
                "end_date",
                "duration_days",
                "detection_count",
                "footprint_ha",
                "burned_area_ha",
                "frp_sum",
                "frp_trend_mw",
                "burned_pixel_count",
            ]
            if c in events.columns
        ]
        table = events[show_cols].copy()
        size_cols = [c for c in ("footprint_ha", "burned_area_ha", "frp_sum") if c in table.columns]
        table["size_ha"] = table[size_cols].max(axis=1) if size_cols else 0.0
        # Prefer spreading / faster fires when ranking the event list.
        if "is_spreading" in events.columns:
            table["_is_spreading"] = events["is_spreading"].to_numpy()
        if "spread_km_per_day" in table.columns:
            table = table.sort_values(
                [c for c in ("_is_spreading", "spread_km_per_day", "size_ha", "detection_count") if c in table.columns],
                ascending=[False, False, False, False],
            ).reset_index(drop=True)
        else:
            table = table.sort_values(["size_ha", "detection_count"], ascending=False).reset_index(drop=True)
        if "_is_spreading" in table.columns:
            table = table.drop(columns=["_is_spreading"])
        all_ids = table["event_id"].astype(str).tolist()
        selected = st.session_state.get("selected_event_id")
        if selected not in all_ids and all_ids:
            selected = all_ids[0]
            st.session_state["selected_event_id"] = selected
        # Dropdown stays responsive: largest fires + currently selected.
        top_n = min(2000, len(all_ids))
        event_ids = table["event_id"].astype(str).head(top_n).tolist()
        if selected and selected not in event_ids:
            event_ids = [selected] + event_ids
        if selected in event_ids:
            st.session_state["event_story_select"] = selected

        st.caption(
            f"{len(table):,} fire events (sorted by size). "
            f"Dropdown lists the largest {min(top_n, len(table)):,}; click the map or table for others."
        )
        choice = st.selectbox(
            "Fire event",
            options=event_ids,
            format_func=lambda eid: (
                f"{eid} · "
                f"{int(table.loc[table.event_id.astype(str) == eid, 'detection_count'].iloc[0]):,} dets · "
                f"{float(table.loc[table.event_id.astype(str) == eid, 'size_ha'].iloc[0]):,.0f} ha size · "
                f"{float(table.loc[table.event_id.astype(str) == eid, 'burned_area_ha'].iloc[0]):,.0f} ha burned"
            ),
            key="event_story_select",
        )
        st.session_state["selected_event_id"] = choice

        event_table = st.dataframe(
            table.drop(columns=["size_ha"]),
            width="stretch",
            hide_index=True,
            height=360,
            on_select="rerun",
            selection_mode="single-row",
            key="event_story_table",
        )
        try:
            rows = event_table.selection.rows
            if rows:
                picked = str(table.iloc[int(rows[0])]["event_id"])
                st.session_state["selected_event_id"] = picked
                st.session_state["event_story_select"] = picked
                choice = picked
        except Exception:
            pass

        detections = event_detections(clustered, choice)
        if detections.empty:
            st.warning("No FIRMS detections found for that event.")
            return

        timeline = build_event_timeline(clustered, burned, choice)
        start_date = detections["acq_date"].min().date()
        end_date = detections["acq_date"].max().date()
        day_options = [pd.Timestamp(value).date() for value in timeline["date"]] if not timeline.empty else [start_date]
        default_day = day_options[-1]
        day_index = st.slider(
            "Event day",
            min_value=0,
            max_value=max(0, len(day_options) - 1),
            value=max(0, len(day_options) - 1),
            help=f"Scrub from first detection ({start_date}) to last activity ({end_date}).",
            key=f"event_story_day_{choice}",
        )
        as_of = day_options[day_index]
        snap = build_event_snapshot(clustered, burned, choice, as_of)

        cols = st.columns(4)
        cols[0].metric("Days since first detection", f"{snap['days_since_start']:,}")
        cols[1].metric("FIRMS detections so far", f"{snap['detection_count']:,}")
        cols[2].metric("Spread footprint", f"{snap['footprint_ha']:,.0f} ha")
        cols[3].metric("Burned inside footprint", f"{snap['burned_area_ha']:,.0f} ha")
        st.markdown(
            f"**{choice}** first detected **{start_date}**, grown through **{as_of}** "
            f"({snap['detection_count']:,} detections, {snap['footprint_ha']:,.0f} ha footprint, "
            f"{snap['burned_area_ha']:,.0f} ha burned)."
        )

        active = snap["active_detections"]
        burned_inside = snap["burned_inside"]
        if burned_inside is not None and not burned_inside.empty:
            burned_inside = burned_inside.copy()
            burned_inside["burned_area_radius"] = burned_inside["burned_area_ha"].clip(lower=1).pow(0.5) * 120
            burned_inside["age_days"] = (
                pd.Timestamp(as_of).normalize() - pd.to_datetime(burned_inside["date"]).dt.normalize()
            ).dt.days.clip(lower=0)

        poly_df = None
        if snap["polygon"] is not None:
            poly_df = pd.DataFrame(
                [
                    {
                        "event_id": choice,
                        "polygon": [snap["polygon"]],
                        "fill_color": event_color(choice),
                        "footprint_ha": snap["footprint_ha"],
                        "burned_area_ha": snap["burned_area_ha"],
                    }
                ]
            )

        view = None
        if active is not None and not active.empty:
            view = {
                "latitude": float(active["latitude"].mean()),
                "longitude": float(active["longitude"].mean()),
                "zoom": 7,
            }
        self.render_pydeck_map(
            active if active is not None else pd.DataFrame(),
            burned_area_points=burned_inside,
            event_polygons=poly_df,
            enable_event_selection=False,
            view_state=view,
            chart_key=f"event_story_deck_{choice}_{as_of.isoformat()}",
        )
        if not timeline.empty:
            st.line_chart(
                timeline.set_index("date")[["detection_count", "footprint_ha", "burned_area_ha"]],
                height=220,
            )
            st.dataframe(timeline, width="stretch", hide_index=True)

    @staticmethod
    def _show_spread_metrics(metrics: dict, events: pd.DataFrame | None) -> None:
        cols = st.columns(5)
        cols[0].metric("Fire events", f"{metrics.get('event_count', 0):,}")
        cols[1].metric("Spread footprint", f"{metrics.get('total_footprint_ha', 0):,.0f} ha")
        cols[2].metric("Land burned (MCD64)", f"{metrics.get('total_burned_area_ha', 0):,.0f} ha")
        cols[3].metric("Total FRP", f"{metrics.get('total_frp_mw', 0):,.0f} MW")
        cols[4].metric("Large growing", f"{metrics.get('large_growing_events', 0):,}")
        st.caption(
            f"As of {metrics.get('as_of', '')}: events are space–time clusters of 1 km MODIS/VIIRS "
            f"points (≤{DEFAULT_EVENT_MAX_KM:g} km, ≤{DEFAULT_EVENT_GAP_DAYS} day gap). "
            "FRP (MW) measures intensity; growth class separates small local burns from expanding fires. "
            "Use Event story to scrub one fire's core → front."
        )

    @staticmethod
    def _show_link_metrics(linked_points: pd.DataFrame, burned_points: pd.DataFrame) -> None:
        fire_count = len(linked_points)
        burned_count = len(burned_points)
        matched = int(linked_points["matched_burned"].fillna(False).sum()) if fire_count else 0
        match_rate = (matched / fire_count) if fire_count else 0.0
        if fire_count and "matched_burned_area_ha" in linked_points.columns:
            matched_mask = linked_points["matched_burned"].fillna(False).astype(bool)
            linked_ha = float(
                pd.to_numeric(linked_points.loc[matched_mask, "matched_burned_area_ha"], errors="coerce")
                .fillna(0)
                .sum()
            )
        else:
            linked_ha = 0.0
        cols = st.columns(4)
        cols[0].metric("Fires in view", f"{fire_count:,}")
        cols[1].metric("Burned pixels", f"{burned_count:,}")
        cols[2].metric("Matched fires", f"{matched:,}")
        cols[3].metric("Fire match rate", f"{match_rate:.0%}")
        st.caption(f"Burned area linked to fires (sum of nearest pixel ha): {linked_ha:,.1f} ha")

    @staticmethod
    def _marker_color(row) -> str:
        if bool(row.get("matched_burned")):
            return "gold"
        confidence = row.get("confidence")
        if pd.notna(confidence):
            if confidence < 40:
                return "orange"
            if confidence < 70:
                return "darkred"
        return "red"

    @staticmethod
    def _pydeck_color(confidence) -> list[int]:
        if pd.isna(confidence):
            return [220, 80, 40, 180]
        if confidence < 40:
            return [255, 165, 0, 170]
        if confidence < 70:
            return [190, 40, 30, 180]
        return [255, 0, 0, 190]

    @staticmethod
    def _pydeck_age_color(age_days) -> list[int]:
        if pd.isna(age_days) or age_days <= 0:
            return [255, 0, 0, 210]
        if age_days <= 2:
            return [40, 120, 255, 190]
        return [255, 255, 255, 170]

    @staticmethod
    def _pydeck_matched_color(age_days) -> list[int]:
        if pd.isna(age_days) or age_days <= 0:
            return [255, 200, 40, 230]
        if age_days <= 2:
            return [255, 170, 40, 200]
        return [255, 220, 120, 180]

    @staticmethod
    def _pydeck_burned_age_color(age_days) -> list[int]:
        if pd.isna(age_days) or age_days <= 0:
            return [120, 50, 10, 180]
        if age_days <= 2:
            return [90, 45, 20, 140]
        return [70, 50, 35, 100]

    @staticmethod
    def _popup_lines(row) -> list[str]:
        lines = [
            f"Date: {row['acq_date'].date()}",
            f"Lat/Lon: {row['latitude']:.4f}, {row['longitude']:.4f}",
        ]
        confidence = row.get("confidence")
        brightness = row.get("brightness")
        if pd.notna(confidence):
            lines.append(f"Confidence: {confidence:.0f}")
        if pd.notna(brightness):
            lines.append(f"Brightness: {brightness:.1f}")
        if "matched_burned" in row:
            lines.append(f"Matched burned: {bool(row['matched_burned'])}")
        nearest = row.get("nearest_burn_km")
        if pd.notna(nearest):
            lines.append(f"Nearest burn km: {nearest:.2f}")
        linked_ha = row.get("matched_burned_area_ha")
        if pd.notna(linked_ha):
            lines.append(f"Linked burned ha: {linked_ha:.1f}")
        for col in ["satellite", "instrument", "frp", "daynight"]:
            if col in row and pd.notna(row[col]):
                lines.append(f"{col}: {row[col]}")
        return lines
