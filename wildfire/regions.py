"""Region helpers for North America monthly wildfire modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Bounds:
    """Longitude/latitude bounding box."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def contains(self, lon: float, lat: float) -> bool:
        return (
            self.min_lon <= float(lon) <= self.max_lon
            and self.min_lat <= float(lat) <= self.max_lat
        )


# Conservative North America extent, including Alaska, Canada, CONUS, Mexico,
# Central America, and nearby Caribbean islands.
NORTH_AMERICA_BOUNDS = Bounds(min_lon=-170.0, min_lat=5.0, max_lon=-50.0, max_lat=85.0)


def normalize_longitude(lon: float) -> float:
    """Normalize longitudes to the [-180, 180] range."""

    value = float(lon)
    if value > 180.0:
        value = ((value + 180.0) % 360.0) - 180.0
    return value


def grid_cell_origin(value: float, resolution: float) -> float:
    """Return the lower-left coordinate of a regular grid cell."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    return math.floor(float(value) / resolution) * resolution


def assign_grid_region(lat: float, lon: float, resolution: float = 1.0) -> dict:
    """Assign a point to a regular latitude/longitude region."""

    norm_lon = normalize_longitude(lon)
    lat_min = grid_cell_origin(lat, resolution)
    lon_min = grid_cell_origin(norm_lon, resolution)
    lat_max = lat_min + resolution
    lon_max = lon_min + resolution
    return {
        "region_id": f"lat{lat_min:.2f}_lon{lon_min:.2f}",
        "region_lat_min": lat_min,
        "region_lat_max": lat_max,
        "region_lon_min": lon_min,
        "region_lon_max": lon_max,
        "region_lat_center": lat_min + resolution / 2.0,
        "region_lon_center": lon_min + resolution / 2.0,
    }
