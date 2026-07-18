"""
Streamlit app for downloading wildfire predictor/target data.

Run with:
    streamlit run wildfire_downloader_streamlit.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

from wildfire.features.build import build_feature_table, save_feature_table
from wildfire.ingest.era5 import ERA5_DATASET, ERA5_VARIABLES, load_era5_monthly
from wildfire.ingest.firms import load_firms_monthly
from wildfire.ingest.grace import GRACE_SHORT_NAME, GRACE_VERSION, load_grace_monthly
from wildfire.ingest.mcd64a1 import aggregate_burned_area_table
from wildfire.models.backtest import run_regression_backtest, run_severe_fire_classifier_backtest
from wildfire.regions import NORTH_AMERICA_BOUNDS
from wildfire.ui.app_controller import StreamlitWorkflowApp
from wildfire.ui.map_viewer import FirmsMapViewer, prepare_firms_map_points


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


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_FIRMS_DIR = DEFAULT_RAW_DIR / "firms"
DEFAULT_GRACE_DIR = DEFAULT_RAW_DIR / "grace"
DEFAULT_ERA5_DIR = DEFAULT_RAW_DIR / "era5"
DEFAULT_MCD64A1_DIR = DEFAULT_RAW_DIR / "mcd64a1"


def _default_bbox_text() -> str:
    return "-130.937500,17.817045,-74.375000,50.797242"


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _format_bbox_id(bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    return f"w{west:.4f}_s{south:.4f}_e{east:.4f}_n{north:.4f}".replace("-", "m").replace(".", "p")


def _download_id(
    product: str,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    extra: str = "",
) -> str:
    payload = {
        "product": product,
        "bbox": [round(float(value), 6) for value in bbox],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "extra": extra,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{_slug(product)}_{start:%Y%m%d}_{end:%Y%m%d}_{_format_bbox_id(bbox)}_{digest}"


def _manifest_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_suffix(path.suffix + ".manifest.json")


def _write_manifest(output_path: str | Path, metadata: dict) -> None:
    manifest = dict(metadata)
    manifest["output"] = str(output_path)
    manifest["download_id"] = manifest.get("download_id") or Path(output_path).stem
    _manifest_path(output_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _existing_matching_file(path: str | Path, download_id: str) -> bool:
    output = Path(path)
    manifest = _manifest_path(output)
    if not output.is_file() or not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("download_id") == download_id


def _read_env_value(name: str, env_path: str | Path = ".env") -> str:
    """Read one value from a local .env file without adding a dependency."""

    path = Path(env_path)
    if not path.is_file():
        return ""
    prefix = f"{name}="
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        value = line[len(prefix):].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return ""


def _bbox_from_folium_drawings(drawings) -> tuple[float, float, float, float] | None:
    if not drawings:
        return None
    geometry = drawings[-1].get("geometry", {})
    coords = geometry.get("coordinates")
    if not coords:
        return None
    if geometry.get("type") == "Polygon":
        points = coords[0]
    elif geometry.get("type") == "Point":
        return None
    else:
        points = coords
    lons = [float(point[0]) for point in points]
    lats = [float(point[1]) for point in points]
    return min(lons), min(lats), max(lons), max(lats)


def _bbox_picker(label: str, key: str, default_bbox: str | None = None) -> str:
    """Render an optional map rectangle picker and return bbox text."""

    default_text = default_bbox or _default_bbox_text()
    st.markdown(f"**{label}**")
    st.caption(
        "Use the map edit tool to drag the rectangle corners, or draw a new rectangle. "
        "The bbox field updates from the latest rectangle. Manual edits still work."
    )

    try:
        import folium
        from folium.plugins import Draw
        from streamlit_folium import st_folium
    except ImportError:
        st.info("Install streamlit-folium to enable map drawing. Using text bbox input only.")
        return st.text_input(
            "Bounding box west,south,east,north",
            value=default_text,
            key=f"{key}_bbox_text",
        )

    west, south, east, north = _parse_bbox(
        st.session_state.get(f"{key}_bbox_text", default_text)
    )
    center = [(south + north) / 2.0, (west + east) / 2.0]
    fmap = folium.Map(location=center, zoom_start=3, tiles="OpenStreetMap")
    editable_group = folium.FeatureGroup(name=f"{key}_editable_bbox")
    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color="#3388ff",
        fill=True,
        fill_opacity=0.08,
        tooltip="Drag corners with the edit tool",
    ).add_to(editable_group)
    editable_group.add_to(fmap)
    Draw(
        export=False,
        feature_group=editable_group,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)
    result = st_folium(
        fmap,
        height=360,
        width=None,
        key=f"{key}_map",
        returned_objects=["all_drawings"],
    )
    drawn_bbox = _bbox_from_folium_drawings(result.get("all_drawings"))
    if drawn_bbox:
        default_text = ",".join(f"{value:.6f}" for value in drawn_bbox)
        st.session_state[f"{key}_bbox_text"] = default_text
        st.success(f"Selected bbox: {default_text}")

    return st.text_input(
        "Bounding box west,south,east,north",
        value=st.session_state.get(f"{key}_bbox_text", default_text),
        key=f"{key}_bbox_text",
    )


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _files_by_mtime(directory: str | Path, patterns: list[str]) -> list[Path]:
    path = Path(directory)
    if not path.is_dir():
        return []
    files: list[Path] = []
    for pattern in patterns:
        files.extend(item for item in path.rglob(pattern) if item.is_file())
    return sorted(set(files), key=lambda item: item.stat().st_mtime, reverse=True)


def _discover_feature_inputs(raw_dir: str | Path = DEFAULT_RAW_DIR) -> dict:
    """Find downloaded files that can feed the feature-table builder."""

    root = Path(raw_dir)
    firms_candidates = []
    exact_firms = root / "firms_north_america.csv"
    if exact_firms.is_file():
        firms_candidates.append(exact_firms)
    firms_candidates.extend(_files_by_mtime(root / "firms", ["*.csv"]))
    firms_candidates.extend(
        path for path in _files_by_mtime(root, ["*firms*.csv", "*fire*.csv"]) if path != exact_firms
    )

    grace_files = _files_by_mtime(root / "grace", ["*.nc", "*.nc4", "*.cdf"])
    era5_files = _files_by_mtime(root / "era5", ["*.nc", "*.nc4", "*.cdf"])
    burned_candidates = _files_by_mtime(
        root / "mcd64a1",
        ["*burn*.csv", "*mcd64*.csv", "*.csv"],
    )
    return {
        "firms_csv": str(firms_candidates[0]) if firms_candidates else "",
        "grace_files": [str(path) for path in sorted(grace_files)],
        "era5_files": [str(path) for path in sorted(era5_files)],
        "burned_area_csv": str(burned_candidates[0]) if burned_candidates else "",
    }


def _date_chunks(start: date, end: date, max_days: int = 5):
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=max_days - 1))
        yield current, (chunk_end - current).days + 1
        current = chunk_end + timedelta(days=1)


def _parse_bbox(text: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("Bounding box must contain four comma-separated numbers.")
    west, south, east, north = parts
    if west >= east or south >= north:
        raise ValueError("Bounding box must be west,south,east,north.")
    return west, south, east, north


def _months_by_year(start: date, end: date) -> dict[int, list[int]]:
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


def _download_firms_area(
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    output_csv: Path,
    force: bool = False,
) -> pd.DataFrame:
    download_id = _download_id("firms", bbox, start, end, extra=source)
    if not force and _existing_matching_file(output_csv, download_id):
        st.info(f"Using existing matching FIRMS file: {output_csv}")
        return pd.read_csv(output_csv)

    west, south, east, north = bbox
    area = f"{west},{south},{east},{north}"
    frames = []
    columns = None
    progress = st.progress(0)
    chunks = list(_date_chunks(start, end, max_days=5))
    for idx, (chunk_start, day_range) in enumerate(chunks, start=1):
        url = (
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{quote(map_key)}/{source}/{area}/{day_range}/{chunk_start.isoformat()}"
        )
        st.write(f"Requesting {source} {chunk_start.isoformat()} for {day_range} day(s)")
        frame = pd.read_csv(url)
        columns = columns or list(frame.columns)
        frames.append(frame)
        progress.progress(idx / len(chunks))
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=columns or FIRMS_COLUMNS)
    )
    if combined.empty and len(combined.columns) == 0:
        combined = pd.DataFrame(columns=FIRMS_COLUMNS)
    if not combined.empty:
        combined = combined.drop_duplicates()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    _write_manifest(
        output_csv,
        {
            "download_id": download_id,
            "product": "FIRMS",
            "source": source,
            "bbox": bbox,
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
    )
    return combined


def _prepare_firms_map_points(
    csv_path: str | Path,
    start: date | None = None,
    end: date | None = None,
    min_confidence: float | None = None,
    max_points: int = 5000,
) -> pd.DataFrame:
    """Compatibility wrapper for tests and older imports."""

    return prepare_firms_map_points(
        csv_path,
        start=start,
        end=end,
        min_confidence=min_confidence,
        max_points=max_points,
    )


def _search_download_earthaccess(
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


def tab_firms():
    st.subheader("FIRMS Active Fire CSV")
    st.caption(
        "Downloads active fire detections through the FIRMS Area API. "
        "For very long 2015-present pulls, the FIRMS archive portal may still be faster."
    )
    col1, col2 = st.columns(2)
    with col1:
        map_key = st.text_input(
            "FIRMS MAP_KEY",
            value=_read_env_value("FIRMS_MAP_KEY"),
            type="password",
        )
        source = st.selectbox("Source", FIRMS_SOURCES, index=0)
        start = st.date_input("Start date", value=date(2026, 1, 1), key="firms_start")
    with col2:
        end = st.date_input("End date", value=date.today(), key="firms_end")
    bbox_text = _bbox_picker("FIRMS bounding box", "firms")
    bbox = _parse_bbox(bbox_text)
    firms_id = _download_id("firms", bbox, start, end, extra=source)
    output_csv = st.text_input("Output CSV", value=str(DEFAULT_FIRMS_DIR / f"{firms_id}.csv"))
    force = st.checkbox("Force redownload even if matching file exists", value=False, key="firms_force")
    if _existing_matching_file(output_csv, firms_id):
        st.info("A matching FIRMS file already exists. Download will be skipped unless force is enabled.")

    if st.button("Download FIRMS CSV", type="primary"):
        try:
            if not map_key.strip():
                raise ValueError("Enter a FIRMS MAP_KEY.")
            if start > end:
                raise ValueError("Start date must be before end date.")
            df = _download_firms_area(map_key.strip(), source, bbox, start, end, Path(output_csv), force=force)
            st.success(f"Saved {len(df):,} row(s) to {output_csv}")
            if not df.empty:
                st.dataframe(df.head(200), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

    st.markdown("**Archive alternative**")
    st.write(
        "If you already requested a bulk FIRMS archive CSV, put the file path below and use "
        "the Build Features tab."
    )


def tab_grace():
    st.subheader("GRACE Groundwater / Root Zone Moisture")
    st.caption(f"NASA GES DISC product `{GRACE_SHORT_NAME}` version `{GRACE_VERSION}`.")
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", value=date(2025, 1, 1), key="grace_start")
        end = st.date_input("End date", value=date(2025, 1, 31), key="grace_end")
        max_results = st.number_input("Max granules (0 = all)", min_value=0, value=1)
    with col2:
        output_dir = st.text_input("Output directory", value=str(DEFAULT_GRACE_DIR))
    bbox_text = _bbox_picker("GRACE bounding box", "grace")
    bbox = _parse_bbox(bbox_text)
    grace_id = _download_id(
        "grace",
        bbox,
        start,
        end,
        extra=f"{GRACE_SHORT_NAME}_{GRACE_VERSION}_{int(max_results) or 'all'}",
    )
    grace_dir = Path(output_dir) / grace_id
    force = st.checkbox("Force redownload even if matching files exist", value=False, key="grace_force")
    if (grace_dir / f"{grace_id}.manifest.json").is_file():
        st.info("A matching GRACE download manifest exists. Download will be skipped unless force is enabled.")

    if st.button("Download GRACE", type="primary"):
        try:
            if start > end:
                raise ValueError("Start date must be before end date.")
            files = _search_download_earthaccess(
                short_name=GRACE_SHORT_NAME,
                version=GRACE_VERSION,
                temporal=(start.isoformat(), end.isoformat()),
                bbox=bbox,
                output_dir=_ensure_dir(grace_dir),
                max_results=int(max_results) or None,
                download_id=grace_id,
                force=force,
            )
            st.success(f"Downloaded {len(files):,} file(s).")
            for path in files:
                st.code(str(path))
        except Exception as exc:
            st.error(str(exc))


def tab_era5():
    st.subheader("ERA5 Temperature / Dewpoint")
    st.caption(
        f"Copernicus CDS `{ERA5_DATASET}` with variables "
        f"`{', '.join(ERA5_VARIABLES)}` for VPD calculation."
    )
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", value=date(2025, 1, 1), key="era5_start")
        end = st.date_input("End date", value=date(2025, 1, 31), key="era5_end")
    with col2:
        output_dir = st.text_input("Output directory", value=str(DEFAULT_ERA5_DIR))
    bbox_text = _bbox_picker("ERA5 bounding box", "era5")
    bbox = _parse_bbox(bbox_text)
    era5_id = _download_id("era5", bbox, start, end, extra="_".join(ERA5_VARIABLES))
    force = st.checkbox("Force redownload even if matching files exist", value=False, key="era5_force")
    if (Path(output_dir) / f"{era5_id}.manifest.json").is_file():
        st.info("A matching ERA5 manifest exists. Download will be skipped unless force is enabled.")

    if st.button("Download ERA5", type="primary"):
        try:
            if start > end:
                raise ValueError("Start date must be before end date.")
            if _existing_matching_file(Path(output_dir) / f"{era5_id}.nc", era5_id) and not force:
                existing = Path(output_dir) / f"{era5_id}.nc"
                st.info(f"Using existing matching ERA5 file: {existing}")
                st.success("Downloaded 1 ERA5 file(s).")
                st.code(str(existing))
                return
            west, south, east, north = bbox
            area = [north, west, south, east]
            try:
                import cdsapi
            except ImportError as exc:
                raise RuntimeError("Install cdsapi in the active environment.") from exc

            out_dir = _ensure_dir(output_dir)
            client = cdsapi.Client()
            saved = []
            for year, months in _months_by_year(start, end).items():
                year_id = _download_id(
                    "era5",
                    bbox,
                    date(year, min(months), 1),
                    date(year, max(months), 1),
                    extra="_".join(ERA5_VARIABLES),
                )
                target = out_dir / f"{year_id}.nc"
                if _existing_matching_file(target, year_id) and not force:
                    st.info(f"Using existing matching ERA5 file: {target}")
                    saved.append(target)
                    continue
                request = {
                    "product_type": "monthly_averaged_reanalysis",
                    "variable": list(ERA5_VARIABLES),
                    "year": str(year),
                    "month": [f"{month:02d}" for month in months],
                    "time": "00:00",
                    "area": area,
                    "format": "netcdf",
                }
                st.write(f"Requesting ERA5 {year}: {', '.join(request['month'])}")
                client.retrieve(ERA5_DATASET, request, str(target))
                _write_manifest(
                    target,
                    {
                        "download_id": year_id,
                        "product": "ERA5",
                        "dataset": ERA5_DATASET,
                        "variables": list(ERA5_VARIABLES),
                        "year": year,
                        "months": months,
                        "bbox": bbox,
                        "area": area,
                    },
                )
                saved.append(target)
            st.success(f"Downloaded {len(saved):,} ERA5 file(s).")
            for path in saved:
                st.code(str(path))
        except Exception as exc:
            st.error(str(exc))


def tab_mcd64a1():
    st.subheader("MCD64A1 Monthly Burned Area")
    st.caption("Downloads MCD64A1 granules through NASA Earthdata. Pixel extraction is a later processing step.")
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", value=date(2025, 1, 1), key="mcd_start")
        end = st.date_input("End date", value=date(2025, 1, 31), key="mcd_end")
        max_results = st.number_input("Max granules (0 = all)", min_value=0, value=1, key="mcd_max")
    with col2:
        output_dir = st.text_input("Output directory", value=str(DEFAULT_MCD64A1_DIR))
    bbox_text = _bbox_picker("MCD64A1 bounding box", "mcd")
    bbox = _parse_bbox(bbox_text)
    mcd_id = _download_id("mcd64a1", bbox, start, end, extra=f"{int(max_results) or 'all'}")
    mcd_dir = Path(output_dir) / mcd_id
    force = st.checkbox("Force redownload even if matching files exist", value=False, key="mcd_force")
    if (mcd_dir / f"{mcd_id}.manifest.json").is_file():
        st.info("A matching MCD64A1 download manifest exists. Download will be skipped unless force is enabled.")

    if st.button("Download MCD64A1", type="primary"):
        try:
            if start > end:
                raise ValueError("Start date must be before end date.")
            files = _search_download_earthaccess(
                short_name="MCD64A1",
                temporal=(start.isoformat(), end.isoformat()),
                bbox=bbox,
                output_dir=_ensure_dir(mcd_dir),
                max_results=int(max_results) or None,
                download_id=mcd_id,
                force=force,
            )
            st.success(f"Downloaded {len(files):,} file(s).")
            for path in files:
                st.code(str(path))
        except Exception as exc:
            st.error(str(exc))


def tab_build_features():
    st.subheader("Build Feature Table")
    st.caption("Join downloaded target and predictor files into one monthly modeling CSV.")
    auto_detect = st.checkbox("Automatically use files from download folders", value=True)
    raw_dir = st.text_input("Raw download root", value=str(DEFAULT_RAW_DIR))
    discovered = _discover_feature_inputs(raw_dir) if auto_detect else {
        "firms_csv": "",
        "grace_files": [],
        "era5_files": [],
        "burned_area_csv": "",
    }

    if auto_detect:
        with st.expander("Discovered inputs", expanded=True):
            st.write("FIRMS CSV:", discovered["firms_csv"] or "Not found")
            st.write(f"GRACE NetCDF files: {len(discovered['grace_files'])}")
            for path in discovered["grace_files"][:10]:
                st.code(path)
            if len(discovered["grace_files"]) > 10:
                st.caption(f"...and {len(discovered['grace_files']) - 10} more")
            st.write(f"ERA5 NetCDF files: {len(discovered['era5_files'])}")
            for path in discovered["era5_files"][:10]:
                st.code(path)
            if len(discovered["era5_files"]) > 10:
                st.caption(f"...and {len(discovered['era5_files']) - 10} more")
            st.write("Burned-area pixel CSV:", discovered["burned_area_csv"] or "Not found")

    firms_csv = st.text_input(
        "FIRMS CSV",
        value=discovered["firms_csv"] or str(DEFAULT_FIRMS_DIR / "firms_north_america.csv"),
    )
    grace_files = st.text_area(
        "GRACE NetCDF files (one per line)",
        value="\n".join(discovered["grace_files"]),
    )
    era5_files = st.text_area(
        "ERA5 NetCDF files (one per line)",
        value="\n".join(discovered["era5_files"]),
    )
    burned_area_csv = st.text_input(
        "Extracted MCD64A1 burned-area pixel CSV (optional)",
        value=discovered["burned_area_csv"],
    )
    output_csv = st.text_input("Output feature CSV", value=str(Path("data/processed/wildfire_monthly_training.csv")))
    col1, col2 = st.columns(2)
    with col1:
        region_resolution = st.number_input("Region resolution degrees", min_value=0.25, value=1.0, step=0.25)
    with col2:
        min_confidence = st.number_input("Minimum FIRMS confidence (0 = no filter)", min_value=0.0, value=0.0)

    if st.button("Build Features", type="primary"):
        try:
            if not Path(firms_csv).is_file():
                raise ValueError("Provide a valid FIRMS CSV.")
            fire = load_firms_monthly(
                firms_csv,
                region_resolution=region_resolution,
                min_confidence=min_confidence or None,
            )
            grace_paths = [line.strip() for line in grace_files.splitlines() if line.strip()]
            era5_paths = [line.strip() for line in era5_files.splitlines() if line.strip()]
            grace = load_grace_monthly(grace_paths, region_resolution) if grace_paths else None
            era5 = load_era5_monthly(era5_paths, region_resolution) if era5_paths else None
            burned = None
            if burned_area_csv.strip():
                burned = aggregate_burned_area_table(pd.read_csv(burned_area_csv), region_resolution)
            table = build_feature_table(fire, grace_monthly=grace, era5_monthly=era5, burned_area_monthly=burned)
            path = save_feature_table(table, output_csv)
            st.success(f"Saved {len(table):,} feature row(s) to {path}")
            st.dataframe(table.head(200), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))


def tab_backtest():
    st.subheader("Backtest")
    st.caption("Run the baseline wildfire model on a held-out year.")
    feature_csv = st.text_input("Feature CSV", value=str(Path("data/processed/wildfire_monthly_training.csv")))
    output_dir = st.text_input("Output directory", value=str(Path("wildfire_output")))
    col1, col2, col3 = st.columns(3)
    with col1:
        target_column = st.selectbox("Target", ["fire_count", "burned_area_ha"], index=0)
    with col2:
        backtest_year = st.number_input("Backtest year", min_value=2000, max_value=2100, value=2026)
    with col3:
        run_classifier = st.checkbox("Also run severe-fire classifier", value=True)

    if st.button("Run Backtest", type="primary"):
        try:
            if not Path(feature_csv).is_file():
                raise ValueError("Provide a valid feature CSV.")
            table = pd.read_csv(feature_csv)
            metrics = run_regression_backtest(
                table,
                output_dir=output_dir,
                target_column=target_column,
                backtest_year=int(backtest_year),
            )
            if run_classifier:
                metrics["classifier"] = run_severe_fire_classifier_backtest(
                    table,
                    output_dir=output_dir,
                    backtest_year=int(backtest_year),
                )
            st.success("Backtest complete.")
            st.json(metrics)
        except Exception as exc:
            st.error(str(exc))


def tab_map_viewer():
    st.subheader("Map Viewer")
    st.caption("Load downloaded FIRMS detections onto an interactive map.")
    discovered = _discover_feature_inputs(DEFAULT_RAW_DIR)
    firms_csv = st.text_input(
        "FIRMS CSV",
        value=discovered["firms_csv"] or str(DEFAULT_FIRMS_DIR / "firms_north_america.csv"),
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start = st.date_input("Start date filter", value=date(2026, 1, 1), key="map_start")
    with col2:
        end = st.date_input("End date filter", value=date.today(), key="map_end")
    with col3:
        min_confidence = st.number_input("Minimum confidence", min_value=0.0, max_value=100.0, value=0.0)
    with col4:
        max_points = st.number_input("Max points to draw", min_value=100, max_value=100000, value=5000, step=100)

    render_mode_label = st.selectbox(
        "Map render mode",
        ["Fast clusters (stable, no popups)", "Clustered popups", "Individual popup markers"],
        index=0,
    )
    render_mode = {
        "Fast clusters (stable, no popups)": "fast",
        "Clustered popups": "clustered_popups",
        "Individual popup markers": "individual_popups",
    }[render_mode_label]
    if st.button("Load Map", type="primary"):
        try:
            FirmsMapViewer().render(
                firms_csv,
                start,
                end,
                min_confidence or None,
                int(max_points),
                render_mode,
            )
        except Exception as exc:
            st.error(str(exc))


def main():
    StreamlitWorkflowApp(
        title="Wildfire Earthdata Downloader",
        caption="Download FIRMS, GRACE, ERA5, and MCD64A1 data for wildfire ML backtesting.",
        workflows={
            "FIRMS active fire CSV": tab_firms,
            "GRACE groundwater/root-zone moisture": tab_grace,
            "ERA5 temperature/dewpoint": tab_era5,
            "MCD64A1 burned area": tab_mcd64a1,
            "Map viewer": tab_map_viewer,
            "Build feature table": tab_build_features,
            "Run backtest": tab_backtest,
        },
    ).run()


if __name__ == "__main__":
    main()
