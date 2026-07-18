"""FIRMS map viewer UI and data preparation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st


def prepare_firms_map_points(
    csv_path: str | Path,
    start: date | None = None,
    end: date | None = None,
    min_confidence: float | None = None,
    max_points: int = 5000,
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
    work = work.dropna(subset=["latitude", "longitude", "acq_date"])
    if start:
        work = work[work["acq_date"].dt.date >= start]
    if end:
        work = work[work["acq_date"].dt.date <= end]
    if min_confidence is not None and "confidence" in work:
        work = work[work["confidence"] >= float(min_confidence)]
    work = work.sort_values("acq_date", ascending=False)
    if max_points and len(work) > max_points:
        work = work.head(int(max_points))
    return work.reset_index(drop=True)


@dataclass
class FirmsMapViewer:
    """Render downloaded FIRMS detections on an interactive map."""

    fast_cluster_threshold: int = 2000

    def render_map(self, points: pd.DataFrame, render_mode: str = "fast"):
        try:
            import folium
            from folium.plugins import FastMarkerCluster, MarkerCluster
            from streamlit_folium import st_folium
        except ImportError as exc:
            raise RuntimeError("Install streamlit-folium to use the map viewer.") from exc

        center = [float(points["latitude"].mean()), float(points["longitude"].mean())]
        fmap = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap")

        if render_mode == "fast" or len(points) > self.fast_cluster_threshold:
            locations = points[["latitude", "longitude"]].astype(float).values.tolist()
            FastMarkerCluster(locations, name="FIRMS detections").add_to(fmap)
            st.caption(
                f"Using fast cluster mode for {len(points):,} points. "
                "Popups are disabled in this mode to keep the browser responsive."
            )
        else:
            layer = (
                MarkerCluster(name="FIRMS detections")
                if render_mode == "clustered_popups"
                else folium.FeatureGroup(name="FIRMS detections")
            )
            for _, row in points.iterrows():
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

    def render(
        self,
        firms_csv: str | Path,
        start: date,
        end: date,
        min_confidence: float,
        max_points: int,
        render_mode: str = "fast",
        **legacy_kwargs,
    ):
        if "use_cluster" in legacy_kwargs and render_mode == "fast":
            render_mode = "clustered_popups" if legacy_kwargs["use_cluster"] else "individual_popups"
        if not Path(firms_csv).is_file():
            raise ValueError("Provide a valid FIRMS CSV.")
        points = prepare_firms_map_points(
            firms_csv,
            start=start,
            end=end,
            min_confidence=min_confidence or None,
            max_points=int(max_points),
        )
        if points.empty:
            st.warning("No FIRMS detections matched the selected filters.")
            return points

        st.success(f"Loaded {len(points):,} FIRMS point(s).")
        self.render_map(points, render_mode=render_mode)
        st.dataframe(points.head(500), width="stretch")
        return points

    @staticmethod
    def _marker_color(row) -> str:
        confidence = row.get("confidence")
        if pd.notna(confidence):
            if confidence < 40:
                return "orange"
            if confidence < 70:
                return "darkred"
        return "red"

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
        for col in ["satellite", "instrument", "frp", "daynight"]:
            if col in row and pd.notna(row[col]):
                lines.append(f"{col}: {row[col]}")
        return lines
