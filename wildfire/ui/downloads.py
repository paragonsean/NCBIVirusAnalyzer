"""Download wrappers used by the Streamlit wildfire app."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st

from wildfire.ingest.mcd64a1 import burned_area_polygons_path, extract_burned_area_pixels
from wildfire.ui.common import (
    DEFAULT_FIRMS_DIR,
    DEFAULT_MCD64A1_DIR,
    download_id,
    ensure_dir,
    resolve_local_cache,
    write_manifest,
)


FIRMS_SOURCES = [
    "MODIS_SP",
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
    "MODIS_NRT",
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
]

# MCD14DL is the MODIS NRT active-fire vector product exposed via FIRMS as MODIS_NRT.
MCD14DL_SOURCE = "MODIS_NRT"
MCD14DL_NRT_SOURCES = [
    "MODIS_NRT",
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
]

FIRMS_COLUMNS = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_t31",
    "frp",
    "daynight",
    "type",
]


def date_chunks(start: date, end: date, max_days: int = 5):
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=max_days - 1))
        yield current, (chunk_end - current).days + 1
        current = chunk_end + timedelta(days=1)


def months_by_year(start: date, end: date) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    current = date(start.year, start.month, 1)
    final = date(end.year, end.month, 1)
    while current <= final:
        out.setdefault(current.year, []).append(current.month)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return out


def download_firms_area(
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    output_csv: Path,
    force: bool = False,
    progress_callback=None,
) -> pd.DataFrame:
    firms_id = download_id("firms", bbox, start, end, extra=source)
    if not force:
        cached = resolve_local_cache(
            output_csv,
            firms_id,
            product="FIRMS",
            bbox=bbox,
            start=start,
            end=end,
            required_fields={"source": source},
            allow_covering_dates=True,
        )
        if cached is not None:
            st.info(f"Using local FIRMS cache (no re-download): {cached}")
            return pd.read_csv(cached)

    west, south, east, north = bbox
    area = f"{west},{south},{east},{north}"
    frames = []
    columns = None
    chunks = list(date_chunks(start, end, max_days=5))
    progress = None if progress_callback is not None else st.progress(0)
    for idx, (chunk_start, day_range) in enumerate(chunks, start=1):
        url = (
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{quote(map_key)}/{source}/{area}/{day_range}/{chunk_start.isoformat()}"
        )
        detail = f"{source} {chunk_start.isoformat()} ({day_range} day(s))"
        if progress_callback is not None:
            progress_callback(idx - 1, max(1, len(chunks)), detail)
        else:
            st.write(f"Requesting {detail}")
        frame = pd.read_csv(url)
        columns = columns or list(frame.columns)
        frames.append(frame)
        if progress_callback is not None:
            progress_callback(idx, max(1, len(chunks)), detail)
        else:
            progress.progress(idx / max(1, len(chunks)))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns or FIRMS_COLUMNS)
    if combined.empty and len(combined.columns) == 0:
        combined = pd.DataFrame(columns=FIRMS_COLUMNS)
    if not combined.empty:
        combined = combined.drop_duplicates()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    write_manifest(
        output_csv,
        {
            "download_id": firms_id,
            "product": "FIRMS",
            "source": source,
            "bbox": bbox,
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
    )
    return combined


def combine_similar_firms_points(
    frames: pd.DataFrame | list[pd.DataFrame],
    max_km: float = 1.0,
) -> pd.DataFrame:
    """Merge near-duplicate detections across satellites (same day, ≤ max_km).

    Keeps one row per space–time cluster with max FRP/brightness/confidence and
    lists contributing satellites / FIRMS sources.
    """

    if isinstance(frames, list):
        if not frames:
            return pd.DataFrame(columns=FIRMS_COLUMNS)
        work = pd.concat(frames, ignore_index=True)
    else:
        work = frames.copy()
    if work.empty:
        return work

    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["acq_date"] = pd.to_datetime(work["acq_date"], errors="coerce")
    for col in ("frp", "brightness", "bright_t31", "confidence"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["latitude", "longitude", "acq_date"]).copy()
    if work.empty:
        return work

    if "firms_source" not in work.columns:
        work["firms_source"] = work.get("source", pd.Series([""] * len(work)))

    try:
        from sklearn.neighbors import BallTree
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to combine similar FIRMS points.") from exc

    earth_radius_km = 6371.0
    radius = float(max_km) / earth_radius_km
    parts = []
    for day, day_df in work.groupby(work["acq_date"].dt.normalize(), sort=True):
        day_df = day_df.reset_index(drop=True)
        if len(day_df) == 1:
            row = day_df.iloc[0].to_dict()
            row["acq_date"] = pd.Timestamp(day)
            row["source_count"] = 1
            row["firms_sources"] = str(row.get("firms_source") or "")
            parts.append(row)
            continue

        coords = np.radians(day_df[["latitude", "longitude"]].to_numpy(dtype=float))
        tree = BallTree(coords, metric="haversine")
        neighbors = tree.query_radius(coords, r=radius)
        parent = np.arange(len(day_df), dtype=np.int64)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i, idxs in enumerate(neighbors):
            for j in idxs:
                if j > i:
                    union(i, int(j))

        roots = np.array([find(i) for i in range(len(day_df))], dtype=np.int64)
        day_df = day_df.copy()
        day_df["_cluster"] = roots
        for _, group in day_df.groupby("_cluster", sort=False):
            sources = sorted({str(s) for s in group["firms_source"].dropna().astype(str) if str(s)})
            sats = sorted({str(s) for s in group.get("satellite", pd.Series(dtype=object)).dropna().astype(str) if str(s)})
            instruments = sorted(
                {str(s) for s in group.get("instrument", pd.Series(dtype=object)).dropna().astype(str) if str(s)}
            )
            frp = group["frp"] if "frp" in group else pd.Series([0.0])
            # Prefer the highest-FRP detection as the representative location.
            seed = group.loc[frp.fillna(-1).idxmax()] if len(group) else group.iloc[0]
            conf = group["confidence"] if "confidence" in group else pd.Series(dtype=float)
            # Numeric confidence when possible; keep categorical VIIRS letters via mode.
            conf_num = pd.to_numeric(conf, errors="coerce")
            parts.append(
                {
                    "latitude": float(group["latitude"].mean()),
                    "longitude": float(group["longitude"].mean()),
                    "brightness": float(pd.to_numeric(group.get("brightness"), errors="coerce").max())
                    if "brightness" in group
                    else np.nan,
                    "scan": seed.get("scan"),
                    "track": seed.get("track"),
                    "acq_date": pd.Timestamp(day),
                    "acq_time": seed.get("acq_time"),
                    "satellite": ",".join(sats) if sats else seed.get("satellite"),
                    "instrument": ",".join(instruments) if instruments else seed.get("instrument"),
                    "confidence": float(conf_num.max()) if conf_num.notna().any() else seed.get("confidence"),
                    "version": seed.get("version"),
                    "bright_t31": float(pd.to_numeric(group.get("bright_t31"), errors="coerce").max())
                    if "bright_t31" in group
                    else np.nan,
                    # Max FRP avoids double-counting the same fire seen by multiple sensors.
                    "frp": float(frp.fillna(0).max()),
                    "daynight": seed.get("daynight"),
                    "type": seed.get("type"),
                    "firms_source": ",".join(sources),
                    "firms_sources": ",".join(sources),
                    "source_count": int(max(1, len(sources))),
                    "merged_detections": int(len(group)),
                }
            )

    out = pd.DataFrame(parts)
    if not out.empty:
        out = out.sort_values(["acq_date", "frp"], ascending=[True, False]).reset_index(drop=True)
    return out


def download_firms_multi_source(
    map_key: str,
    sources: list[str],
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    output_csv: Path,
    force: bool = False,
    combine_km: float = 1.0,
    progress_callback=None,
) -> pd.DataFrame:
    """Download multiple FIRMS sources, combine nearby same-day points, cache locally."""

    sources = [str(s).strip() for s in sources if str(s).strip()]
    if not sources:
        raise ValueError("Select at least one FIRMS source.")

    source_key = "+".join(sorted(sources))
    firms_id = download_id("firms_multi", bbox, start, end, extra=f"{source_key}|{combine_km:g}km")
    if not force:
        cached = resolve_local_cache(
            output_csv,
            firms_id,
            product="FIRMS",
            bbox=bbox,
            start=start,
            end=end,
            required_fields={"source": source_key},
            allow_covering_dates=True,
        )
        if cached is not None:
            st.info(f"Using local multi-satellite FIRMS cache (no re-download): {cached}")
            return pd.read_csv(cached)

    frames = []
    total = max(1, len(sources))
    for idx, source in enumerate(sources, start=1):
        detail = f"{source} ({idx}/{total})"
        if progress_callback is not None:
            progress_callback(idx - 1, total, detail)
        else:
            st.write(f"Downloading {detail}")
        # Per-source sidecar cache under firms/multi_parts to avoid re-hitting the API.
        part_id = download_id("firms", bbox, start, end, extra=source)
        part_csv = DEFAULT_FIRMS_DIR / "multi_parts" / f"{part_id}.csv"
        part = download_firms_area(
            map_key=map_key,
            source=source,
            bbox=bbox,
            start=start,
            end=end,
            output_csv=part_csv,
            force=force,
            progress_callback=None,
        )
        if not part.empty:
            part = part.copy()
            part["firms_source"] = source
            frames.append(part)
        if progress_callback is not None:
            progress_callback(idx, total, detail)

    raw_count = int(sum(len(f) for f in frames))
    combined = combine_similar_firms_points(frames, max_km=combine_km)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    write_manifest(
        output_csv,
        {
            "download_id": firms_id,
            "product": "FIRMS",
            "source": source_key,
            "sources": sources,
            "bbox": list(bbox),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "combine_km": float(combine_km),
            "raw_detections": raw_count,
            "combined_detections": int(len(combined)),
        },
    )
    st.caption(
        f"Combined {raw_count:,} raw detections from {len(sources)} source(s) "
        f"→ {len(combined):,} unique points (≤{combine_km:g} km, same day)."
    )
    return combined


def search_download_earthaccess(
    short_name: str,
    temporal: tuple[str, str],
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    version: str | None = None,
    max_results: int | None = None,
    download_id: str | None = None,
    force: bool = False,
) -> list[str]:
    try:
        import earthaccess
    except ImportError as exc:
        raise RuntimeError("Install earthaccess in the active environment.") from exc

    manifest = output_dir / f"{download_id}.manifest.json" if download_id else None
    if download_id and manifest and manifest.is_file() and not force:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        files = [path for path in data.get("files", []) if Path(path).is_file()]
        if data.get("download_id") == download_id and files:
            st.info(f"Using {len(files):,} existing matching {short_name} file(s).")
            return files

    earthaccess.login(strategy="netrc", persist=False)
    kwargs = {
        "short_name": short_name,
        "temporal": temporal,
        "bounding_box": bbox,
    }
    if version:
        kwargs["version"] = version
    if max_results:
        kwargs["count"] = int(max_results)
    results = earthaccess.search_data(**kwargs)
    st.write(f"Found {len(results):,} granule(s).")
    if not results:
        return []
    files = earthaccess.download(results, local_path=str(output_dir))
    if download_id and manifest:
        manifest.write_text(
            json.dumps(
                {
                    "download_id": download_id,
                    "product": short_name,
                    "version": version,
                    "temporal": temporal,
                    "bbox": bbox,
                    "max_results": max_results,
                    "files": [str(path) for path in files],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return files


def ensure_map_data_cached(
    *,
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    include_burned: bool = True,
    max_mcd_results: int | None = None,
    force: bool = False,
    progress_callback=None,
) -> dict:
    """Download FIRMS (+ optional MCD64A1 extract) into local cache; skip when present."""

    firms_id = download_id("firms", bbox, start, end, extra=source)
    firms_csv = DEFAULT_FIRMS_DIR / f"{firms_id}.csv"
    result = {
        "firms_csv": str(firms_csv),
        "firms_status": "cached",
        "burned_area_csv": "",
        "burned_polygons": "",
        "burned_status": "skipped",
    }

    def _tick(done, total, detail=""):
        if progress_callback is not None:
            progress_callback(done, total, detail)

    total_steps = 2 if include_burned else 1
    cached_firms = None if force else resolve_local_cache(
        firms_csv,
        firms_id,
        product="FIRMS",
        bbox=bbox,
        start=start,
        end=end,
        required_fields={"source": source},
    )
    if cached_firms is not None:
        result["firms_csv"] = str(cached_firms)
        result["firms_status"] = "cached"
        _tick(1, total_steps, f"FIRMS cache hit: {Path(cached_firms).name}")
    else:
        _tick(0, total_steps, "downloading FIRMS")
        ensure_dir(DEFAULT_FIRMS_DIR)
        download_firms_area(
            map_key=map_key,
            source=source,
            bbox=bbox,
            start=start,
            end=end,
            output_csv=firms_csv,
            force=force,
            progress_callback=progress_callback,
        )
        result["firms_csv"] = str(firms_csv)
        result["firms_status"] = "downloaded"

    if not include_burned:
        return result

    burned_id = download_id("mcd64a1_pixels", bbox, start, end, extra="burned_area")
    burned_csv = DEFAULT_MCD64A1_DIR / f"{burned_id}.csv"
    poly_path = burned_area_polygons_path(burned_csv)
    result["burned_area_csv"] = str(burned_csv)
    result["burned_polygons"] = str(poly_path)

    cached_burned = None if force else resolve_local_cache(
        burned_csv,
        burned_id,
        product="mcd64a1_pixels",
        bbox=bbox,
        start=start,
        end=end,
    )
    if (
        cached_burned is not None
        and burned_area_polygons_path(cached_burned).is_file()
        and not force
    ):
        result["burned_area_csv"] = str(cached_burned)
        result["burned_polygons"] = str(burned_area_polygons_path(cached_burned))
        result["burned_status"] = "cached"
        _tick(2, 2, f"burned-area cache hit: {Path(cached_burned).name}")
        return result

    if burned_csv.is_file() and poly_path.is_file() and not force:
        result["burned_status"] = "cached"
        _tick(2, 2, f"burned-area cache hit: {burned_csv.name}")
        return result

    _tick(1, 2, "fetching MCD64A1 granules")
    mcd_dir = ensure_dir(DEFAULT_MCD64A1_DIR / burned_id)
    granules = search_download_earthaccess(
        short_name="MCD64A1",
        temporal=(start.isoformat(), end.isoformat()),
        bbox=bbox,
        output_dir=mcd_dir,
        max_results=max_mcd_results,
        download_id=burned_id,
        force=force,
    )
    hdf_granules = [
        path
        for path in granules
        if Path(path).is_file() and Path(path).suffix.lower() in {".hdf", ".hdf4", ".hdfeos"}
    ]
    if not hdf_granules:
        raise RuntimeError("No MCD64A1 .hdf granules found to extract burned-area cache.")
    _tick(1, 2, f"extracting {len(hdf_granules):,} granule(s)")
    extract_burned_area_pixels(
        hdf_granules,
        bbox,
        burned_csv,
        progress_callback=progress_callback,
        output_polygons_json=poly_path,
        download_id=burned_id,
        start=start.isoformat(),
        end=end.isoformat(),
    )
    result["burned_status"] = "downloaded"
    _tick(2, 2, "burned-area cache stored")
    return result
