"""Class-based Streamlit workflow pages for the wildfire app."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from wildfire.features.build import build_feature_table, save_feature_table
from wildfire.features.link_fires_burned import build_firms_burned_link_monthly
from wildfire.ingest.era5 import ERA5_DATASET, ERA5_VARIABLES, load_era5_monthly
from wildfire.ingest.firms import load_firms_monthly
from wildfire.ingest.grace import GRACE_SHORT_NAME, GRACE_VERSION, load_grace_monthly
from wildfire.ingest.mcd64a1 import (
    aggregate_burned_area_table,
    burned_area_polygons_path,
    extract_burned_area_pixels,
)
from wildfire.models.backtest import run_regression_backtest, run_severe_fire_classifier_backtest
from wildfire.ui.common import (
    DEFAULT_ERA5_DIR,
    DEFAULT_FIRMS_DIR,
    DEFAULT_GRACE_DIR,
    DEFAULT_MCD64A1_DIR,
    DEFAULT_RAW_DIR,
    ProgressTracker,
    bbox_picker,
    discover_feature_inputs,
    download_id,
    ensure_dir,
    existing_matching_file,
    parse_bbox,
    read_env_value,
    resolve_local_cache,
    write_manifest,
)
from wildfire.ui.downloads import (
    FIRMS_SOURCES,
    MCD14DL_NRT_SOURCES,
    download_firms_area,
    download_firms_multi_source,
    ensure_map_data_cached,
    months_by_year,
    search_download_earthaccess,
)
from wildfire.ui.map_viewer import (
    FirmsMapViewer,
    available_timeline_dates,
    prepare_burned_area_map_points,
    prepare_firms_map_points,
)
from wildfire.features.fire_spread import build_event_spread_summary


class WildfireWorkflowPages:
    def render_firms(self):
        st.subheader("FIRMS Active Fire CSV")
        st.caption(
            "Downloads active fire detections through the FIRMS Area API. "
            "For very long 2015-present pulls, the FIRMS archive portal may still be faster."
        )
        col1, col2 = st.columns(2)
        with col1:
            map_key = st.text_input(
                "FIRMS MAP_KEY",
                value=read_env_value("FIRMS_MAP_KEY"),
                type="password",
            )
            source = st.selectbox("Source", FIRMS_SOURCES, index=0)
            start = st.date_input("Start date", value=date(2026, 1, 1), key="firms_start")
        with col2:
            end = st.date_input("End date", value=date.today(), key="firms_end")
        bbox_text = bbox_picker("FIRMS bounding box", "firms")
        bbox = parse_bbox(bbox_text)
        firms_id = download_id("firms", bbox, start, end, extra=source)
        output_csv = st.text_input("Output CSV", value=str(DEFAULT_FIRMS_DIR / f"{firms_id}.csv"))
        force = st.checkbox("Force redownload even if matching file exists", value=False, key="firms_force")
        if existing_matching_file(output_csv, firms_id):
            st.info("A matching FIRMS file already exists. Download will be skipped unless force is enabled.")

        if st.button("Download FIRMS CSV", type="primary"):
            try:
                if not map_key.strip():
                    raise ValueError("Enter a FIRMS MAP_KEY.")
                if start > end:
                    raise ValueError("Start date must be before end date.")
                df = download_firms_area(map_key.strip(), source, bbox, start, end, Path(output_csv), force=force)
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


    def render_grace(self):
        st.subheader("GRACE Groundwater / Root Zone Moisture")
        st.caption(f"NASA GES DISC product `{GRACE_SHORT_NAME}` version `{GRACE_VERSION}`.")
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start date", value=date(2025, 1, 1), key="grace_start")
            end = st.date_input("End date", value=date(2025, 1, 31), key="grace_end")
            max_results = st.number_input("Max granules (0 = all)", min_value=0, value=1)
        with col2:
            output_dir = st.text_input("Output directory", value=str(DEFAULT_GRACE_DIR))
        bbox_text = bbox_picker("GRACE bounding box", "grace")
        bbox = parse_bbox(bbox_text)
        grace_id = download_id(
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
                files = search_download_earthaccess(
                    short_name=GRACE_SHORT_NAME,
                    version=GRACE_VERSION,
                    temporal=(start.isoformat(), end.isoformat()),
                    bbox=bbox,
                    output_dir=ensure_dir(grace_dir),
                    max_results=int(max_results) or None,
                    download_id=grace_id,
                    force=force,
                )
                st.success(f"Downloaded {len(files):,} file(s).")
                for path in files:
                    st.code(str(path))
            except Exception as exc:
                st.error(str(exc))


    def render_era5(self):
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
        bbox_text = bbox_picker("ERA5 bounding box", "era5")
        bbox = parse_bbox(bbox_text)
        era5_id = download_id("era5", bbox, start, end, extra="_".join(ERA5_VARIABLES))
        force = st.checkbox("Force redownload even if matching files exist", value=False, key="era5_force")
        if (Path(output_dir) / f"{era5_id}.manifest.json").is_file():
            st.info("A matching ERA5 manifest exists. Download will be skipped unless force is enabled.")

        if st.button("Download ERA5", type="primary"):
            try:
                if start > end:
                    raise ValueError("Start date must be before end date.")
                if existing_matching_file(Path(output_dir) / f"{era5_id}.nc", era5_id) and not force:
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

                out_dir = ensure_dir(output_dir)
                client = cdsapi.Client()
                saved = []
                for year, months in months_by_year(start, end).items():
                    year_id = download_id(
                        "era5",
                        bbox,
                        date(year, min(months), 1),
                        date(year, max(months), 1),
                        extra="_".join(ERA5_VARIABLES),
                    )
                    target = out_dir / f"{year_id}.nc"
                    if existing_matching_file(target, year_id) and not force:
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
                    write_manifest(
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


    def render_mcd64a1(self):
        st.subheader("MCD64A1 Monthly Burned Area")
        st.caption("Downloads MCD64A1 granules through NASA Earthdata. Pixel extraction is a later processing step.")
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start date", value=date(2025, 1, 1), key="mcd_start")
            end = st.date_input("End date", value=date(2025, 1, 31), key="mcd_end")
            max_results = st.number_input("Max granules (0 = all)", min_value=0, value=1, key="mcd_max")
        with col2:
            output_dir = st.text_input("Output directory", value=str(DEFAULT_MCD64A1_DIR))
        bbox_text = bbox_picker("MCD64A1 bounding box", "mcd")
        bbox = parse_bbox(bbox_text)
        mcd_id = download_id("mcd64a1", bbox, start, end, extra=f"{int(max_results) or 'all'}")
        mcd_dir = Path(output_dir) / mcd_id
        force = st.checkbox("Force redownload even if matching files exist", value=False, key="mcd_force")
        if (mcd_dir / f"{mcd_id}.manifest.json").is_file():
            st.info("A matching MCD64A1 download manifest exists. Download will be skipped unless force is enabled.")

        if st.button("Download MCD64A1", type="primary"):
            try:
                if start > end:
                    raise ValueError("Start date must be before end date.")
                files = search_download_earthaccess(
                    short_name="MCD64A1",
                    temporal=(start.isoformat(), end.isoformat()),
                    bbox=bbox,
                    output_dir=ensure_dir(mcd_dir),
                    max_results=int(max_results) or None,
                    download_id=mcd_id,
                    force=force,
                )
                st.success(f"Downloaded {len(files):,} file(s).")
                for path in files:
                    st.code(str(path))
            except Exception as exc:
                st.error(str(exc))


    def render_build_features(self):
        st.subheader("Build Feature Table")
        st.caption("Join downloaded target and predictor files into one monthly modeling CSV.")
        auto_detect = st.checkbox("Automatically use files from download folders", value=True)
        raw_dir = st.text_input("Raw download root", value=str(DEFAULT_RAW_DIR))
        discovered = discover_feature_inputs(raw_dir) if auto_detect else {
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
                links = None
                if burned_area_csv.strip():
                    if not Path(burned_area_csv).is_file():
                        raise ValueError("Provide a valid burned-area pixel CSV.")
                    burned = aggregate_burned_area_table(pd.read_csv(burned_area_csv), region_resolution)
                    links = build_firms_burned_link_monthly(
                        firms_csv,
                        burned_area_csv,
                        region_resolution=region_resolution,
                        min_confidence=min_confidence or None,
                    )
                table = build_feature_table(
                    fire,
                    grace_monthly=grace,
                    era5_monthly=era5,
                    burned_area_monthly=burned,
                    firms_burned_links_monthly=links,
                )
                path = save_feature_table(table, output_csv)
                st.success(f"Saved {len(table):,} feature row(s) to {path}")
                st.dataframe(table.head(200), use_container_width=True)
            except Exception as exc:
                st.error(str(exc))


    def render_backtest(self):
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


    def render_map_viewer(self):
        st.subheader("Map Viewer")
        st.caption(
            "Watch FIRMS fire starts spread over time, see event overlaps, "
            "and measure burned land with MCD64A1."
        )
        discovered = discover_feature_inputs(DEFAULT_RAW_DIR)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            start = st.date_input("Start date filter", value=date(2026, 1, 1), key="map_start")
        with col2:
            end = st.date_input("End date filter", value=date.today(), key="map_end")
        with col3:
            min_confidence = st.number_input("Minimum confidence", min_value=0.0, max_value=100.0, value=0.0)
        with col4:
            limit_points = st.checkbox("Limit points", value=False)
        max_points = None
        if limit_points:
            max_points = st.number_input("Max points to draw", min_value=100, max_value=1000000, value=5000, step=100)
        else:
            st.caption("Map will load all matching points.")
        auto_download = st.checkbox(
            "Automatically download FIRMS/MCD64 for the selected bbox and dates",
            value=True,
            help=(
                "Stores downloads under data/raw/ and reuses them next time. "
                "Exact matches and wider date-range caches for the same bbox are reused "
                "(no re-download)."
            ),
        )
        map_key = read_env_value("FIRMS_MAP_KEY")
        source = st.selectbox("FIRMS source for auto-download", FIRMS_SOURCES, index=0)
        bbox_text = bbox_picker("Map download bounding box", "map_download")
        bbox = parse_bbox(bbox_text)
        firms_id = download_id("firms", bbox, start, end, extra=source)
        default_map_csv = DEFAULT_FIRMS_DIR / f"{firms_id}.csv"
        cached_firms = resolve_local_cache(
            default_map_csv,
            firms_id,
            product="FIRMS",
            bbox=bbox,
            start=start,
            end=end,
            required_fields={"source": source},
        )
        if auto_download:
            firms_csv = str(cached_firms or default_map_csv)
            if cached_firms is not None:
                st.caption(f"FIRMS local cache ready: `{Path(cached_firms).name}`")
            else:
                st.info(f"FIRMS will download once to: `{default_map_csv}` (kept under data/raw/firms)")
        else:
            firms_csv = st.text_input(
                "FIRMS CSV",
                value=discovered["firms_csv"] or str(cached_firms or default_map_csv),
                key="map_firms_csv_manual",
            )
        show_burned_area = st.checkbox("Overlay burned area", value=bool(discovered["burned_area_csv"]))
        burned_area_csv = ""
        burned_id = ""
        max_mcd_results = 0
        if show_burned_area:
            burned_id = download_id("mcd64a1_pixels", bbox, start, end, extra="burned_area")
            default_burned_csv = DEFAULT_MCD64A1_DIR / f"{burned_id}.csv"
            cached_burned = resolve_local_cache(
                default_burned_csv,
                burned_id,
                product="mcd64a1_pixels",
                bbox=bbox,
                start=start,
                end=end,
            )
            if auto_download:
                burned_area_csv = str(cached_burned or default_burned_csv)
                poly_path = burned_area_polygons_path(burned_area_csv)
                if Path(burned_area_csv).is_file() and poly_path.is_file():
                    st.caption(
                        f"Burned-area local cache ready: `{Path(burned_area_csv).name}` "
                        f"+ `{poly_path.name}`"
                    )
                elif Path(burned_area_csv).is_file():
                    st.info(
                        f"Pixel CSV exists without scar borders — sync/reload will extract "
                        f"polygons to `{poly_path.name}`."
                    )
                else:
                    st.info(
                        f"Burned-area will download once to: `{default_burned_csv}` "
                        "(granules + extracted pixels/borders kept under data/raw/mcd64a1)"
                    )
            else:
                burned_area_csv = st.text_input(
                    "Burned-area pixel CSV",
                    value=discovered["burned_area_csv"] or str(cached_burned or default_burned_csv),
                    help="Expected columns: latitude, longitude, date, burned_area_ha. "
                    "Scar border polygons are written beside it as *.polygons.json.",
                    key="map_burned_csv_manual",
                )
            max_mcd_results = st.number_input(
                "Max MCD64A1 granules for auto-download (0 = all)",
                min_value=0,
                value=0,
                key="map_mcd_max",
            )

        with st.expander("Local data cache", expanded=False):
            st.markdown(
                f"- FIRMS folder: `{DEFAULT_FIRMS_DIR}`\n"
                f"- MCD64A1 folder: `{DEFAULT_MCD64A1_DIR}`\n"
                "- Files are keyed by bbox + dates + source and reused automatically.\n"
                "- Shrinking the date range reuses a wider cache for the same bbox when available."
            )
            force_refresh = st.checkbox("Force re-download (ignore local cache)", value=False)
            if st.button("Download & store locally", type="secondary"):
                if not map_key:
                    st.error("FIRMS_MAP_KEY is missing from .env.")
                else:
                    try:
                        progress = ProgressTracker(["Cache FIRMS / burned area"])
                        cached = ensure_map_data_cached(
                            map_key=map_key,
                            source=source,
                            bbox=bbox,
                            start=start,
                            end=end,
                            include_burned=show_burned_area,
                            max_mcd_results=int(max_mcd_results) or None,
                            force=force_refresh,
                            progress_callback=progress.tick,
                        )
                        progress.complete("Local cache ready")
                        st.success(
                            f"FIRMS: {cached['firms_status']} → `{cached['firms_csv']}`"
                            + (
                                f" | Burned: {cached['burned_status']} → `{cached['burned_area_csv']}`"
                                if show_burned_area
                                else ""
                            )
                        )
                        if auto_download:
                            firms_csv = cached["firms_csv"]
                            if show_burned_area:
                                burned_area_csv = cached["burned_area_csv"]
                    except Exception as exc:
                        st.error(str(exc))
        matched_only = st.checkbox(
            "Show only FIRMS fires matched to burned area",
            value=False,
            disabled=not show_burned_area,
            help="Requires burned-area overlay. Matches within 1.5 km and ±1 day.",
        )
        analyze_spread = st.checkbox(
            "Analyze fire spread, overlap, and burned land",
            value=True,
            help=(
                "Groups FIRMS detections into fire events, draws growing footprints, "
                "measures event overlap, and sums MCD64A1 burned area inside each footprint. "
                "Use time-lapse to watch spread grow day by day."
            ),
        )

        selection_key = f"{firms_id}|{burned_id}|{auto_download}|{show_burned_area}"
        previous_selection = st.session_state.get("map_viewer_selection_key")
        if previous_selection != selection_key:
            st.session_state["map_viewer_selection_key"] = selection_key
            # Changing bbox/dates/source while the map is live should pull matching data.
            if st.session_state.get("map_viewer_loaded", False) and auto_download:
                st.caption("Selection changed — refreshing downloads for the new bbox/dates.")

        render_mode_label = st.selectbox(
            "Map render mode",
            [
                "Native Streamlit map (recommended)",
                "Folium fast clusters",
                "Folium clustered popups",
                "Folium individual popup markers",
            ],
            index=0,
        )
        render_mode = {
            "Native Streamlit map (recommended)": "pydeck",
            "Folium fast clusters": "fast",
            "Folium clustered popups": "clustered_popups",
            "Folium individual popup markers": "individual_popups",
        }[render_mode_label]
        timelapse = st.checkbox("Enable time-lapse mode", value=False)
        frame_index = 0
        trailing_days = 1
        if timelapse:
            date_count = 1
            date_preview = ""
            try:
                preview_points = (
                    prepare_firms_map_points(
                        firms_csv,
                        start=start,
                        end=end,
                        min_confidence=min_confidence or None,
                        max_points=max_points,
                    )
                    if Path(firms_csv).is_file()
                    else pd.DataFrame()
                )
                preview_burned = None
                if show_burned_area and burned_area_csv and Path(burned_area_csv).is_file():
                    preview_burned = prepare_burned_area_map_points(
                        burned_area_csv,
                        start=start,
                        end=end,
                    )
                dates = available_timeline_dates(preview_points, preview_burned)
                date_count = max(1, len(dates))
                if dates:
                    date_preview = f"{dates[0].isoformat()} to {dates[-1].isoformat()}"
            except Exception:
                date_count = 1
            col_a, col_b = st.columns(2)
            with col_a:
                frame_index = st.slider(
                    "Time-lapse frame",
                    min_value=0,
                    max_value=max(0, date_count - 1),
                    value=0,
                    help="Frames are the union of FIRMS and burned-area dates when overlay is enabled.",
                )
                if date_preview:
                    st.caption(f"{date_count:,} frame date(s): {date_preview}")
            with col_b:
                trailing_days = st.number_input(
                    "Days each fire remains visible",
                    min_value=1,
                    max_value=30,
                    value=3,
                )
        col_load, col_clear = st.columns(2)
        with col_load:
            if st.button("Load Map / Start Auto Reload", type="primary"):
                st.session_state["map_viewer_loaded"] = True
        with col_clear:
            if st.button("Clear Map / Stop Auto Reload"):
                st.session_state["map_viewer_loaded"] = False

        if st.session_state.get("map_viewer_loaded", False):
            try:
                firms_cached = resolve_local_cache(
                    firms_csv,
                    firms_id,
                    product="FIRMS",
                    bbox=bbox,
                    start=start,
                    end=end,
                    required_fields={"source": source},
                )
                if firms_cached is not None:
                    firms_csv = str(firms_cached)
                need_firms_download = bool(auto_download and firms_cached is None)

                poly_path = (
                    burned_area_polygons_path(burned_area_csv) if burned_area_csv else None
                )
                burned_cached = None
                if show_burned_area and burned_area_csv:
                    burned_cached = resolve_local_cache(
                        burned_area_csv,
                        burned_id or Path(burned_area_csv).stem,
                        product="mcd64a1_pixels",
                        bbox=bbox,
                        start=start,
                        end=end,
                    )
                    if burned_cached is not None:
                        burned_area_csv = str(burned_cached)
                        poly_path = burned_area_polygons_path(burned_area_csv)
                need_burned_download = bool(
                    auto_download
                    and show_burned_area
                    and burned_area_csv
                    and (
                        burned_cached is None
                        or poly_path is None
                        or not Path(poly_path).is_file()
                    )
                )
                steps: list[str] = []
                if need_firms_download:
                    steps.append("Download FIRMS")
                if need_burned_download:
                    steps.extend(
                        ["Fetch MCD64A1 granules", "Extract burned-area pixels + scar borders"]
                    )
                steps.append("Load FIRMS points")
                if show_burned_area and burned_area_csv:
                    steps.append("Load burned-area scars")
                if analyze_spread:
                    steps.append("Analyze fire spread")
                steps.append("Render map")

                progress = ProgressTracker(steps)

                if need_firms_download:
                    if not map_key:
                        raise ValueError("FIRMS_MAP_KEY is missing from .env; cannot auto-download map data.")
                    progress.advance("requesting area API chunks")
                    ensure_dir(DEFAULT_FIRMS_DIR)
                    downloaded = download_firms_area(
                        map_key=map_key,
                        source=source,
                        bbox=bbox,
                        start=start,
                        end=end,
                        output_csv=Path(firms_csv),
                        progress_callback=progress.tick,
                    )
                    st.success(
                        f"Stored {len(downloaded):,} FIRMS row(s) locally at `{firms_csv}` "
                        "(reused next time)."
                    )
                elif auto_download:
                    st.caption(f"Using local FIRMS cache: `{Path(firms_csv).name}`")
                if need_burned_download:
                    burned_id = burned_id or download_id(
                        "mcd64a1_pixels", bbox, start, end, extra="burned_area"
                    )
                    mcd_dir = DEFAULT_MCD64A1_DIR / burned_id
                    progress.advance("searching / reusing Earthdata granules")
                    st.info(
                        "Burned-area cache incomplete; downloading MCD64A1 granules and extracting "
                        "pixels plus scar border polygons into local storage."
                    )
                    granules = search_download_earthaccess(
                        short_name="MCD64A1",
                        temporal=(start.isoformat(), end.isoformat()),
                        bbox=bbox,
                        output_dir=ensure_dir(mcd_dir),
                        max_results=int(max_mcd_results) or None,
                        download_id=burned_id,
                    )
                    hdf_granules = [
                        path
                        for path in granules
                        if Path(path).is_file() and Path(path).suffix.lower() in {".hdf", ".hdf4", ".hdfeos"}
                    ]
                    if not hdf_granules:
                        raise RuntimeError("No MCD64A1 .hdf granules found to extract burned-area pixels.")
                    progress.advance(f"{len(hdf_granules):,} HDF4 granule(s)")
                    extract_burned_area_pixels(
                        hdf_granules,
                        bbox,
                        burned_area_csv,
                        progress_callback=progress.tick,
                        output_polygons_json=poly_path,
                        download_id=burned_id,
                        start=start.isoformat(),
                        end=end.isoformat(),
                    )
                    st.success(
                        f"Stored burned-area cache locally: "
                        f"`{Path(burned_area_csv).name}` + `{Path(poly_path).name}`"
                    )
                elif auto_download and show_burned_area and burned_area_csv:
                    st.caption(f"Using local burned-area cache: `{Path(burned_area_csv).name}`")
                viewer = FirmsMapViewer()
                if timelapse:
                    viewer.render_timelapse(
                        firms_csv,
                        start,
                        end,
                        min_confidence or None,
                        max_points,
                        frame_index=frame_index,
                        trailing_days=int(trailing_days),
                        render_mode=render_mode,
                        burned_area_csv=burned_area_csv or None,
                        matched_only=matched_only,
                        analyze_spread=analyze_spread,
                        progress=progress,
                    )
                else:
                    viewer.render(
                        firms_csv,
                        start,
                        end,
                        min_confidence or None,
                        max_points,
                        render_mode,
                        burned_area_csv=burned_area_csv or None,
                        matched_only=matched_only,
                        analyze_spread=analyze_spread,
                        progress=progress,
                    )
            except Exception as exc:
                st.error(str(exc))
        else:
            st.info("Click 'Load Map / Start Auto Reload' once. After that, changes to time-lapse, frame, render mode, filters, or overlays will redraw automatically.")

    def render_mcd14dl(self):
        """Near-real-time MCD14DL (MODIS NRT) fire growth and tracking."""

        st.subheader("MCD14DL real-time fire growth")
        st.caption(
            "Tracks near-real-time MODIS active fires (MCD14DL via FIRMS `MODIS_NRT`). "
            "Downloads the latest detections into `data/raw/firms`, clusters them into growing "
            "fire events, and lets you scrub day-by-day spread. Optional VIIRS NRT sources "
            "can be mixed in for denser tracking."
        )

        map_key = read_env_value("FIRMS_MAP_KEY")
        col1, col2 = st.columns(2)
        with col1:
            lookback_days = st.slider(
                "Lookback window (days)",
                min_value=1,
                max_value=10,
                value=3,
                help="FIRMS NRT area API is fetched in ≤5-day chunks; wider windows are stitched locally.",
            )
        with col2:
            min_confidence = st.number_input(
                "Minimum confidence",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                key="mcd14dl_min_conf",
            )

        sources = st.multiselect(
            "FIRMS NRT satellites",
            MCD14DL_NRT_SOURCES,
            default=list(MCD14DL_NRT_SOURCES),
            help=(
                "Downloads all selected NRT products (MODIS MCD14DL + VIIRS SNPP/NOAA-20/NOAA-21) "
                "and merges near-duplicate same-day points."
            ),
            key="mcd14dl_sources",
        )
        if not sources:
            st.warning("Select at least one FIRMS NRT satellite.")
            return
        combine_km = st.slider(
            "Combine similar points within (km)",
            min_value=0.25,
            max_value=3.0,
            value=1.0,
            step=0.25,
            help="Same-day detections from any satellite within this distance become one point (max FRP kept).",
            key="mcd14dl_combine_km",
        )

        end = date.today()
        start = end - timedelta(days=int(lookback_days) - 1)
        st.caption(f"Tracking window: **{start.isoformat()}** → **{end.isoformat()}** (rolling)")

        bbox_text = bbox_picker("MCD14DL tracking bounding box", "mcd14dl")
        bbox = parse_bbox(bbox_text)
        source_key = "+".join(sorted(sources))
        nrt_id = download_id(
            "mcd14dl_multi",
            bbox,
            start,
            end,
            extra=f"{source_key}|{float(combine_km):g}km",
        )
        output_csv = DEFAULT_FIRMS_DIR / f"{nrt_id}.csv"
        cached = resolve_local_cache(
            output_csv,
            nrt_id,
            product="FIRMS",
            bbox=bbox,
            start=start,
            end=end,
            required_fields={"source": source_key},
        )
        if cached is not None:
            st.caption(f"Local multi-satellite NRT cache ready: `{Path(cached).name}`")
            firms_csv = Path(cached)
        else:
            st.info(
                f"Latest multi-satellite NRT pull will be stored at `{output_csv}` "
                "and reused until the window moves."
            )
            firms_csv = output_csv

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            force_refresh = st.checkbox("Force refresh from FIRMS", value=False, key="mcd14dl_force")
        with col_b:
            analyze_spread = st.checkbox(
                "Cluster fire growth / event sizes",
                value=True,
                key="mcd14dl_spread",
            )
        with col_c:
            timelapse = st.checkbox("Day-by-day growth scrubber", value=True, key="mcd14dl_timelapse")
        with col_d:
            show_growth_grid = st.checkbox(
                "~1 km growth grid",
                value=True,
                key="mcd14dl_grid",
                help="Aggregate detections into daily grid cells so scatter becomes an expansion surface.",
            )
        st.caption(
            "Growth tracking: pull all selected FIRMS NRT satellites, merge same-day points within "
            f"{float(combine_km):g} km, then measure spread (km/day + ha/day) and label "
            "spreading vs holding fires (cool core → hot front on the map)."
        )

        trailing_days = 3
        frame_index = 0
        if timelapse:
            trailing_days = st.number_input(
                "Days each detection stays visible",
                min_value=1,
                max_value=14,
                value=3,
                key="mcd14dl_trail",
            )

        sync = st.button("Sync latest MCD14DL / NRT", type="primary")
        live = st.session_state.get("mcd14dl_live", False)
        if sync:
            st.session_state["mcd14dl_live"] = True
            live = True

        if not live:
            st.info("Click **Sync latest MCD14DL / NRT** to download/store the latest fires and open the tracker.")
            return

        try:
            if not map_key:
                raise ValueError("FIRMS_MAP_KEY is missing from .env; required for MCD14DL/NRT.")

            need_download = force_refresh or cached is None or not Path(firms_csv).is_file()
            steps = []
            if need_download:
                steps.append("Download all FIRMS NRT satellites")
            steps.extend(["Load detections", "Track fire growth", "Render map"])
            progress = ProgressTracker(steps)

            if need_download:
                progress.advance("requesting FIRMS NRT from all selected satellites")
                ensure_dir(DEFAULT_FIRMS_DIR)
                df = download_firms_multi_source(
                    map_key=map_key,
                    sources=list(sources),
                    bbox=bbox,
                    start=start,
                    end=end,
                    output_csv=output_csv,
                    force=force_refresh,
                    combine_km=float(combine_km),
                    progress_callback=progress.tick,
                )
                firms_csv = output_csv
                st.success(
                    f"Stored {len(df):,} combined NRT point(s) from {len(sources)} satellite source(s) "
                    f"at `{output_csv.name}`."
                )
            else:
                progress.advance("using local NRT cache")
                st.caption(f"Using local cache `{Path(firms_csv).name}` (no re-download).")

            progress.advance("reading detections")
            points = prepare_firms_map_points(
                firms_csv,
                start=start,
                end=end,
                min_confidence=min_confidence or None,
            )
            if points.empty:
                progress.complete("No NRT fires in window")
                st.warning("No MCD14DL/NRT detections in this bbox/window.")
                return

            dates = available_timeline_dates(points, None)
            if timelapse and dates:
                frame_index = st.slider(
                    "Growth frame",
                    min_value=0,
                    max_value=max(0, len(dates) - 1),
                    value=max(0, len(dates) - 1),
                    help="Scrub from first detection day through the latest NRT day.",
                    key="mcd14dl_frame",
                )
                st.caption(
                    f"{len(dates):,} day(s) in window · showing through "
                    f"**{dates[min(frame_index, len(dates) - 1)].isoformat()}**"
                )

            if analyze_spread:
                progress.advance("clustering growing events")
                as_of = dates[min(frame_index, len(dates) - 1)] if dates else end
                _clustered, events, metrics = build_event_spread_summary(
                    points,
                    None,
                    as_of=as_of,
                    progress_callback=progress.tick,
                )
                cols = st.columns(6)
                cols[0].metric("Active events", f"{metrics.get('event_count', 0):,}")
                cols[1].metric("Spreading now", f"{metrics.get('spreading_events', 0):,}")
                cols[2].metric(
                    "Fastest spread",
                    f"{metrics.get('fastest_spread_km_per_day', 0):.1f} km/day",
                )
                cols[3].metric("Footprint", f"{metrics.get('total_footprint_ha', 0):,.0f} ha")
                cols[4].metric("Total FRP", f"{metrics.get('total_frp_mw', 0):,.0f} MW")
                cols[5].metric("Large growing", f"{metrics.get('large_growing_events', 0):,}")
                if events is not None and not events.empty:
                    status_counts = (
                        events["spread_status"].value_counts().to_dict()
                        if "spread_status" in events
                        else {}
                    )
                    class_counts = events["growth_class"].value_counts().to_dict() if "growth_class" in events else {}
                    st.caption(
                        "Spread: "
                        + (" · ".join(f"{k}: {v:,}" for k, v in status_counts.items()) or "n/a")
                        + "  |  Class: "
                        + (" · ".join(f"{k}: {v:,}" for k, v in class_counts.items()) or "n/a")
                    )
                    ranked = events.copy()
                    if "spread_km_per_day" in ranked.columns:
                        ranked = ranked.sort_values(
                            ["is_spreading", "spread_km_per_day", "footprint_ha_per_day", "frp_sum"],
                            ascending=[False, False, False, False],
                        )
                    show_cols = [
                        c
                        for c in [
                            "event_id",
                            "spread_status",
                            "spread_km_per_day",
                            "footprint_ha_per_day",
                            "new_cells_per_day",
                            "max_radius_km",
                            "latest_new_cells",
                            "growth_class",
                            "start_date",
                            "end_date",
                            "duration_days",
                            "detection_count",
                            "footprint_ha",
                            "frp_sum",
                            "frp_trend_mw",
                        ]
                        if c in ranked.columns
                    ]
                    st.dataframe(ranked[show_cols].head(75), width="stretch", hide_index=True)
                    rate_cols = [c for c in ("spread_km_per_day", "footprint_ha_per_day") if c in ranked.columns]
                    if rate_cols:
                        st.caption("Top events by radial spread (km/day) and area growth (ha/day)")
                        chart_df = ranked.head(30).set_index("event_id")[rate_cols]
                        st.bar_chart(chart_df)
            else:
                progress.advance("skipping clustering")

            progress.advance("drawing tracker map")
            viewer = FirmsMapViewer()
            if timelapse:
                viewer.render_timelapse(
                    firms_csv,
                    start,
                    end,
                    min_confidence or None,
                    None,
                    frame_index=frame_index,
                    trailing_days=int(trailing_days),
                    render_mode="pydeck",
                    burned_area_csv=None,
                    matched_only=False,
                    analyze_spread=analyze_spread,
                    progress=None,
                    color_mode="growth_front",
                    show_growth_grid=show_growth_grid,
                )
            else:
                viewer.render(
                    firms_csv,
                    start,
                    end,
                    min_confidence or None,
                    None,
                    "pydeck",
                    burned_area_csv=None,
                    matched_only=False,
                    analyze_spread=analyze_spread,
                    progress=None,
                    color_mode="growth_front",
                    show_growth_grid=show_growth_grid,
                )
            progress.complete("NRT tracker ready")
            st.caption(
                "Tip: leave this tab open and click **Sync latest MCD14DL / NRT** again to pull newer detections. "
                "Same-day pulls reuse the local file unless Force refresh is checked."
            )
        except Exception as exc:
            st.error(str(exc))

