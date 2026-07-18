"""MCD64A1 burned-area helpers.

The full MCD64A1 product is distributed as HDF-EOS4 (HDF4) granules. This module
reads Burn Date SDS layers with pyhdf (GDAL/rasterio often lack the HDF4 driver
on Windows), converts MODIS sinusoidal tile coordinates to lat/lon, vectorizes
burn scars into polygons, and aggregates pixels for map display and monthly
region features.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from wildfire.regions import NORTH_AMERICA_BOUNDS, assign_grid_region, normalize_longitude

# MODIS sinusoidal grid constants (MCD64 Collection 6/6.1 User Guide, Appendix B).
_MODIS_R = 6371007.181
_MODIS_T = 1111950.0
_MODIS_XMIN = -20015109.0
_MODIS_YMAX = 10007555.0
_MODIS_W_500M = _MODIS_T / 2400.0
_TILE_RE = re.compile(r"h(\d{2})v(\d{2})", re.IGNORECASE)

def download_mcd64a1(output_dir: str | Path, temporal: tuple[str, str]):
    """Search and download MCD64A1 granules with earthaccess."""

    try:
        import earthaccess
    except ImportError as exc:
        raise RuntimeError("Install earthaccess to download MCD64A1 data.") from exc

    earthaccess.login()
    results = earthaccess.search_data(
        short_name="MCD64A1",
        temporal=temporal,
        bounding_box=(
            NORTH_AMERICA_BOUNDS.min_lon,
            NORTH_AMERICA_BOUNDS.min_lat,
            NORTH_AMERICA_BOUNDS.max_lon,
            NORTH_AMERICA_BOUNDS.max_lat,
        ),
    )
    return earthaccess.download(results, local_path=str(output_dir))

def acquisition_date_from_filename(path: str | Path) -> datetime | None:
    """Extract the MCD64A1 acquisition month from names like MCD64A1.A2025001..."""

    name = Path(path).name
    marker = ".A"
    if marker not in name:
        return None
    token = name.split(marker, 1)[1][:7]
    if len(token) != 7 or not token.isdigit():
        return None
    year = int(token[:4])
    day_of_year = int(token[4:])
    return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

def tile_hv_from_filename(path: str | Path) -> tuple[int, int] | None:
    """Parse MODIS tile horizontal/vertical indices from a granule filename."""

    match = _TILE_RE.search(Path(path).name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))

def _sinu_xy_to_latlon(x, y):
    """Convert MODIS sinusoidal projection meters to WGS84 degrees (vectorized)."""

    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lat_rad = y / _MODIS_R
    # Avoid divide-by-zero at the poles; cos(lat) is near zero there anyway.
    cos_lat = np.cos(lat_rad)
    lon_rad = np.divide(x, _MODIS_R * cos_lat, out=np.zeros_like(x), where=np.abs(cos_lat) > 1e-12)
    return np.degrees(lat_rad), np.degrees(lon_rad)

def _pixel_centers_latlon(tile_h: int, tile_v: int, rows, cols):
    """Return lat/lon centers for MODIS 500 m tile row/col indices."""

    import numpy as np

    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)
    x = (cols + 0.5) * _MODIS_W_500M + tile_h * _MODIS_T + _MODIS_XMIN
    y = _MODIS_YMAX - (rows + 0.5) * _MODIS_W_500M - tile_v * _MODIS_T
    return _sinu_xy_to_latlon(x, y)


def burned_area_polygons_path(pixel_csv: str | Path) -> Path:
    """Sidecar JSON path for vectorized burned-area borders next to the pixel CSV."""

    path = Path(pixel_csv)
    return path.with_name(f"{path.stem}.polygons.json")


def _tile_affine(tile_h: int, tile_v: int):
    """Affine transform from MODIS 500 m tile pixel space to sinusoidal meters."""

    from rasterio.transform import Affine

    x_origin = tile_h * _MODIS_T + _MODIS_XMIN
    y_origin = _MODIS_YMAX - tile_v * _MODIS_T
    return Affine(_MODIS_W_500M, 0.0, x_origin, 0.0, -_MODIS_W_500M, y_origin)


def _sinu_ring_to_lonlat(ring):
    import numpy as np

    xs = np.asarray([pt[0] for pt in ring], dtype=np.float64)
    ys = np.asarray([pt[1] for pt in ring], dtype=np.float64)
    lats, lons = _sinu_xy_to_latlon(xs, ys)
    return [[float(lon), float(lat)] for lon, lat in zip(lons, lats)]


def _project_sinu_geojson_to_wgs84(geom: dict) -> dict:
    """Project a GeoJSON geometry from MODIS sinusoidal meters to WGS84 lon/lat."""

    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Polygon":
        return {"type": "Polygon", "coordinates": [_sinu_ring_to_lonlat(ring) for ring in coords]}
    if gtype == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [[_sinu_ring_to_lonlat(ring) for ring in poly] for poly in coords],
        }
    raise ValueError(f"Unsupported geometry type for burned-area borders: {gtype}")


def _shapely_to_pydeck_rings(geom):
    """Yield pydeck polygon values (list of rings) for Polygon / MultiPolygon parts."""

    if geom is None or geom.is_empty:
        return
    gtype = geom.geom_type
    if gtype == "GeometryCollection":
        for part in geom.geoms:
            yield from _shapely_to_pydeck_rings(part)
        return
    if gtype == "Polygon":
        rings = [list(map(list, geom.exterior.coords))]
        for interior in geom.interiors:
            rings.append(list(map(list, interior.coords)))
        yield rings
        return
    if gtype == "MultiPolygon":
        for part in geom.geoms:
            yield from _shapely_to_pydeck_rings(part)
        return


def _burn_date_iso(acquisition_date: datetime | None, burn_day: int) -> str:
    if acquisition_date is None or int(burn_day) <= 0:
        return ""
    burned_date = datetime(acquisition_date.year, 1, 1) + timedelta(days=int(burn_day) - 1)
    return burned_date.date().isoformat()


def _vectorize_burn_array(
    burn,
    transform,
    acquisition_date: datetime | None,
    source_name: str,
    *,
    project_from_sinu: bool,
    bbox: tuple[float, float, float, float] | None = None,
) -> list[dict]:
    """Turn a Burn Date array into dissolved scar polygons (borders + filled area)."""

    import numpy as np
    from rasterio import features
    from shapely.geometry import box, mapping, shape

    mask = np.asarray(burn) > 0
    if not mask.any():
        return []

    clip = box(*bbox) if bbox is not None else None
    burn_int = np.where(mask, np.asarray(burn, dtype=np.int32), 0)
    rows: list[dict] = []

    for geom_json, value in features.shapes(
        burn_int,
        mask=mask,
        transform=transform,
        connectivity=8,
    ):
        burn_day = int(value)
        if burn_day <= 0:
            continue
        geom_native = shape(geom_json)
        if geom_native.is_empty:
            continue
        area_ha = float(geom_native.area) / 10_000.0
        if project_from_sinu:
            geom_ll = shape(_project_sinu_geojson_to_wgs84(mapping(geom_native)))
        else:
            geom_ll = geom_native
        if not geom_ll.is_valid:
            geom_ll = geom_ll.buffer(0)
        clipped = geom_ll.intersection(clip) if clip is not None else geom_ll
        if clipped.is_empty:
            continue
        if clip is not None and geom_ll.area > 0 and clipped.area < geom_ll.area:
            area_ha *= float(clipped.area / geom_ll.area)
        date_iso = _burn_date_iso(acquisition_date, burn_day)
        parts = []
        part_weights = []
        for rings in _shapely_to_pydeck_rings(clipped):
            if not rings or len(rings[0]) < 4:
                continue
            part = shape({"type": "Polygon", "coordinates": rings})
            parts.append(rings)
            part_weights.append(max(float(part.area), 0.0))
        weight_sum = sum(part_weights) or 1.0
        for rings, weight in zip(parts, part_weights):
            rows.append(
                {
                    "date": date_iso,
                    "burned_area_ha": float(area_ha * (weight / weight_sum)),
                    "polygon": rings,
                    "fill_color": [120, 55, 15, 140],
                    "source_file": source_name,
                }
            )
    return rows


def _extract_polygons_from_hdf4(
    granule_path: Path,
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    tile = tile_hv_from_filename(granule_path)
    if tile is None:
        raise ValueError(f"Could not parse MODIS tile hXXvYY from {granule_path.name}")
    tile_h, tile_v = tile
    burn = _read_burn_date_hdf4(granule_path)
    return _vectorize_burn_array(
        burn,
        _tile_affine(tile_h, tile_v),
        acquisition_date_from_filename(granule_path),
        granule_path.name,
        project_from_sinu=True,
        bbox=bbox,
    )


def _extract_polygons_from_rasterio(
    granule_path: Path,
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import transform as warp_xy
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
        from shapely.geometry import box, shape
    except ImportError as exc:
        raise RuntimeError("Install rasterio and shapely to extract burned-area borders.") from exc

    west, south, east, north = bbox
    acquisition_date = acquisition_date_from_filename(granule_path)
    with rasterio.open(granule_path) as container:
        burn_dataset = _pick_burn_date_dataset(container)
    burn_path = burn_dataset or str(granule_path)

    with rasterio.open(burn_path) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, west, south, east, north, densify_pts=21)
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        try:
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        except rasterio.errors.WindowError:
            return []
        if window.width <= 0 or window.height <= 0:
            return []
        burn = src.read(1, window=window, masked=True)
        data = np.where(~burn.mask, burn.data, 0)
        transform = src.window_transform(window)
        is_lonlat = bool(src.crs) and src.crs.to_epsg() == 4326
        rows = _vectorize_burn_array(
            data,
            transform,
            acquisition_date,
            granule_path.name,
            project_from_sinu=False,
            bbox=bbox if is_lonlat else None,
        )
        if is_lonlat:
            return rows

        clip = box(west, south, east, north)
        projected = []
        for row in rows:
            rings_ll = []
            for ring in row["polygon"]:
                xs = [pt[0] for pt in ring]
                ys = [pt[1] for pt in ring]
                lons, lats = warp_xy(src.crs, "EPSG:4326", xs, ys)
                rings_ll.append([[float(lon), float(lat)] for lon, lat in zip(lons, lats)])
            geom = shape({"type": "Polygon", "coordinates": rings_ll})
            if not geom.is_valid:
                geom = geom.buffer(0)
            clipped = geom.intersection(clip)
            if clipped.is_empty:
                continue
            for rings in _shapely_to_pydeck_rings(clipped):
                if not rings or len(rings[0]) < 4:
                    continue
                projected.append(
                    {
                        "date": row["date"],
                        "burned_area_ha": float(row["burned_area_ha"]),
                        "polygon": rings,
                        "fill_color": row["fill_color"],
                        "source_file": row["source_file"],
                    }
                )
        return projected

def _read_burn_date_hdf4(path: Path):
    """Read the Burn Date SDS from an MCD64A1 HDF4 granule via pyhdf."""

    try:
        from pyhdf.SD import SD, SDC
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Install pyhdf to read MCD64A1 HDF4 granules "
            "(conda install -c conda-forge pyhdf)."
        ) from exc

    sd = SD(str(Path(path).resolve()), SDC.READ)
    try:
        datasets = sd.datasets()
        name = None
        for candidate in ("Burn Date", "Burn_Date", "burndate"):
            if candidate in datasets:
                name = candidate
                break
        if name is None:
            for candidate in datasets:
                lowered = candidate.lower().replace("_", " ")
                if "burn" in lowered and "date" in lowered:
                    name = candidate
                    break
        if name is None:
            raise ValueError(f"No Burn Date SDS found in {path.name}")
        return np.asarray(sd.select(name)[:])
    finally:
        sd.end()

def _pick_burn_date_dataset(dataset):
    subdatasets = getattr(dataset, "subdatasets", [])
    for subdataset in subdatasets:
        lowered = subdataset.lower()
        if "burn date" in lowered or "burn_date" in lowered or "burndate" in lowered:
            return subdataset
    if subdatasets:
        for subdataset in subdatasets:
            if "burn" in subdataset.lower() and "date" in subdataset.lower():
                return subdataset
    return None

def _extract_from_hdf4(
    granule_path: Path,
    bbox: tuple[float, float, float, float],
    max_pixels_per_granule: int,
) -> list[dict]:
    import numpy as np

    tile = tile_hv_from_filename(granule_path)
    if tile is None:
        raise ValueError(f"Could not parse MODIS tile hXXvYY from {granule_path.name}")
    tile_h, tile_v = tile
    acquisition_date = acquisition_date_from_filename(granule_path)
    burn = _read_burn_date_hdf4(granule_path)
    mask = burn > 0
    if not mask.any():
        return []

    rr, cc = np.where(mask)
    if len(rr) > max_pixels_per_granule:
        step = max(1, len(rr) // max_pixels_per_granule)
        rr = rr[::step][:max_pixels_per_granule]
        cc = cc[::step][:max_pixels_per_granule]

    lats, lons = _pixel_centers_latlon(tile_h, tile_v, rr, cc)
    west, south, east, north = bbox
    in_bbox = (lons >= west) & (lons <= east) & (lats >= south) & (lats <= north)
    if not in_bbox.any():
        return []

    rr = rr[in_bbox]
    cc = cc[in_bbox]
    lats = lats[in_bbox]
    lons = lons[in_bbox]
    burn_values = burn[rr, cc]
    pixel_area_ha = (_MODIS_W_500M * _MODIS_W_500M) / 10_000.0

    rows = []
    for lat, lon, burn_day in zip(lats, lons, burn_values):
        if acquisition_date is not None:
            burned_date = datetime(acquisition_date.year, 1, 1) + timedelta(days=int(burn_day) - 1)
        else:
            burned_date = None
        rows.append(
            {
                "latitude": float(lat),
                "longitude": float(lon),
                "date": burned_date.date().isoformat() if burned_date else "",
                "burned_area_ha": float(pixel_area_ha),
                "source_file": granule_path.name,
            }
        )
    return rows

def _extract_from_rasterio(
    granule_path: Path,
    bbox: tuple[float, float, float, float],
    max_pixels_per_granule: int,
) -> list[dict]:
    """Fallback for GeoTIFF / GDAL-readable burned-area rasters."""

    try:
        import numpy as np
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.warp import transform, transform_bounds
    except ImportError as exc:
        raise RuntimeError("Install rasterio and numpy to extract burned-area GeoTIFFs.") from exc

    west, south, east, north = bbox
    acquisition_date = acquisition_date_from_filename(granule_path)
    with rasterio.open(granule_path) as container:
        burn_dataset = _pick_burn_date_dataset(container)
    burn_path = burn_dataset or str(granule_path)

    with rasterio.open(burn_path) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, west, south, east, north, densify_pts=21)
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        try:
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        except rasterio.errors.WindowError:
            return []
        if window.width <= 0 or window.height <= 0:
            return []
        burn = src.read(1, window=window, masked=True)
        mask = (~burn.mask) & (burn.data > 0)
        if not mask.any():
            return []

        rr, cc = np.where(mask)
        if len(rr) > max_pixels_per_granule:
            step = max(1, len(rr) // max_pixels_per_granule)
            rr = rr[::step][:max_pixels_per_granule]
            cc = cc[::step][:max_pixels_per_granule]

        transform_window = src.window_transform(window)
        xs, ys = rasterio.transform.xy(transform_window, rr, cc, offset="center")
        lons, lats = transform(src.crs, "EPSG:4326", xs, ys)
        pixel_area_ha = abs(transform_window.a * transform_window.e) / 10_000.0
        burn_values = burn.data[rr, cc]

        rows = []
        for lat, lon, burn_day in zip(lats, lons, burn_values):
            if acquisition_date is not None:
                burned_date = datetime(acquisition_date.year, 1, 1) + timedelta(days=int(burn_day) - 1)
            else:
                burned_date = None
            rows.append(
                {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "date": burned_date.date().isoformat() if burned_date else "",
                    "burned_area_ha": float(pixel_area_ha),
                    "source_file": granule_path.name,
                }
            )
        return rows

def _is_hdf4_granule(path: Path) -> bool:
    """True for MCD64A1 HDF-EOS4 granules (never open these with rasterio/GDAL)."""

    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".hdf", ".hdf4", ".hdfeos"}:
        return True
    return name.startswith("mcd64a1") and ".hdf" in name


def extract_burned_area_pixels(
    granule_paths: list[str | Path],
    bbox: tuple[float, float, float, float],
    output_csv: str | Path,
    max_pixels_per_granule: int = 250_000,
    progress_callback=None,
    output_polygons_json: str | Path | None = None,
    download_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> str:
    """Extract burned pixels and scar border polygons from MCD64A1 granules.

    Writes:
      - pixel CSV: latitude, longitude, date, burned_area_ha, source_file
      - polygons JSON sidecar: dissolved burn-date polygons for map fill/borders
      - manifest JSON so later runs reuse the local cache

    HDF4 granules are read with pyhdf; GeoTIFF/other rasters use rasterio.
    Polygon extraction uses the full burn mask (not pixel subsampling).
    """

    paths = [Path(path) for path in granule_paths if Path(path).is_file()]
    rows: list[dict] = []
    polygon_rows: list[dict] = []
    total = max(1, len(paths))
    for idx, granule_path in enumerate(paths, start=1):
        if progress_callback is not None:
            progress_callback(idx - 1, total, granule_path.name)
        if _is_hdf4_granule(granule_path):
            rows.extend(_extract_from_hdf4(granule_path, bbox, max_pixels_per_granule))
            polygon_rows.extend(_extract_polygons_from_hdf4(granule_path, bbox))
        else:
            rows.extend(_extract_from_rasterio(granule_path, bbox, max_pixels_per_granule))
            polygon_rows.extend(_extract_polygons_from_rasterio(granule_path, bbox))
        if progress_callback is not None:
            progress_callback(idx, total, granule_path.name)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=["latitude", "longitude", "date", "burned_area_ha", "source_file"],
    ).to_csv(output_path, index=False)

    polygons_path = Path(output_polygons_json) if output_polygons_json else burned_area_polygons_path(output_path)
    polygons_path.parent.mkdir(parents=True, exist_ok=True)
    polygons_path.write_text(json.dumps(polygon_rows), encoding="utf-8")
    manifest = {
        "download_id": download_id or output_path.stem,
        "product": "mcd64a1_pixels",
        "bbox": list(bbox),
        "start": start,
        "end": end,
        "output": str(output_path),
        "polygons": str(polygons_path),
        "granule_count": len(paths),
        "pixel_count": len(rows),
        "polygon_count": len(polygon_rows),
    }
    Path(str(output_path) + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(output_path)

def aggregate_burned_area_table(
    pixels: pd.DataFrame,
    region_resolution: float = 1.0,
) -> pd.DataFrame:
    """Aggregate extracted MCD64A1 burned-area pixels by month-region."""

    required = {"latitude", "longitude", "date", "burned_area_ha"}
    missing = sorted(required.difference(pixels.columns))
    if missing:
        raise ValueError(f"Burned-area table missing columns: {', '.join(missing)}")

    work = pixels.copy()
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce").map(normalize_longitude)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["burned_area_ha"] = pd.to_numeric(work["burned_area_ha"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["latitude", "longitude", "date"])
    work = work[
        work.apply(
            lambda row: NORTH_AMERICA_BOUNDS.contains(row["longitude"], row["latitude"]),
            axis=1,
        )
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=["region_id", "year", "month", "burned_area_ha"])

    regions = work.apply(
        lambda row: assign_grid_region(row["latitude"], row["longitude"], region_resolution),
        axis=1,
        result_type="expand",
    )
    work = pd.concat([work.reset_index(drop=True), regions.reset_index(drop=True)], axis=1)
    work["year"] = work["date"].dt.year.astype(int)
    work["month"] = work["date"].dt.month.astype(int)
    return (
        work.groupby(
            ["region_id", "year", "month", "region_lat_center", "region_lon_center"],
            as_index=False,
        )
        .agg(burned_area_ha=("burned_area_ha", "sum"))
        .sort_values(["region_id", "year", "month"])
        .reset_index(drop=True)
    )

