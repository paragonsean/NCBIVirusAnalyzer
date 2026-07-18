"""Shared Streamlit UI helpers for wildfire workflows."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import streamlit as st

from wildfire.regions import NORTH_AMERICA_BOUNDS


@dataclass
class ProgressTracker:
    """Stage-based progress bar with elapsed time and simple ETA."""

    steps: list[str]
    _bar: object = field(init=False, repr=False)
    _label: object = field(init=False, repr=False)
    _meta: object = field(init=False, repr=False)
    _index: int = field(init=False, default=-1)
    _started_at: float = field(init=False, default=0.0)
    _step_started_at: float = field(init=False, default=0.0)
    _step_durations: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not self.steps:
            self.steps = ["Working"]
        self._bar = st.progress(0)
        self._label = st.empty()
        self._meta = st.empty()
        self._started_at = time.perf_counter()
        self._step_started_at = self._started_at
        self._render(0.0, "Starting…")

    @property
    def total_steps(self) -> int:
        return max(1, len(self.steps))

    def _format_seconds(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes, rem = divmod(int(seconds), 60)
        if minutes < 60:
            return f"{minutes}m {rem:02d}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m"

    def _eta_seconds(self, fraction: float) -> float | None:
        elapsed = time.perf_counter() - self._started_at
        if self._step_durations and self._index >= 0:
            remaining_steps = max(0, self.total_steps - (self._index + 1))
            avg = sum(self._step_durations) / len(self._step_durations)
            # Blend step-average ETA with overall fraction ETA when available.
            step_eta = remaining_steps * avg
            if fraction > 0.05 and elapsed > 0.5:
                frac_eta = elapsed * (1.0 - fraction) / fraction
                return 0.6 * step_eta + 0.4 * frac_eta
            return step_eta
        if fraction <= 0.05 or elapsed < 0.5:
            return None
        return elapsed * (1.0 - fraction) / fraction

    def _render(self, fraction: float, message: str) -> None:
        fraction = min(1.0, max(0.0, float(fraction)))
        try:
            self._bar.progress(fraction, text=message)
        except TypeError:
            self._bar.progress(fraction)
        self._label.markdown(f"**{message}**")
        elapsed = time.perf_counter() - self._started_at
        eta = self._eta_seconds(fraction)
        step_no = min(max(self._index + 1, 1), self.total_steps)
        if fraction >= 1.0:
            self._meta.caption(f"Done in {self._format_seconds(elapsed)}")
        elif eta is None:
            self._meta.caption(
                f"Step {step_no}/{self.total_steps} · "
                f"elapsed {self._format_seconds(elapsed)} · estimating remaining time…"
            )
        else:
            self._meta.caption(
                f"Step {step_no}/{self.total_steps} · "
                f"elapsed {self._format_seconds(elapsed)} · "
                f"~{self._format_seconds(eta)} remaining"
            )

    def advance(self, detail: str = "") -> None:
        """Move to the next planned step."""

        if self._index >= 0:
            self._step_durations.append(time.perf_counter() - self._step_started_at)
        self._index = min(self._index + 1, self.total_steps - 1)
        self._step_started_at = time.perf_counter()
        label = self.steps[self._index]
        message = f"{label}: {detail}" if detail else label
        self._render(self._index / self.total_steps, message)

    def tick(self, done: int, total: int, detail: str = "") -> None:
        """Update progress within the current step."""

        if self._index < 0:
            self.advance(detail)
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        width = 1.0 / self.total_steps
        fraction = (self._index / self.total_steps) + width * (done / total)
        label = self.steps[self._index]
        message = f"{label}: {detail}" if detail else f"{label} ({done}/{total})"
        self._render(fraction, message)

    def complete(self, message: str = "Map ready") -> None:
        if self._index >= 0:
            self._step_durations.append(time.perf_counter() - self._step_started_at)
        self._index = self.total_steps - 1
        self._render(1.0, message)


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_FIRMS_DIR = DEFAULT_RAW_DIR / "firms"
DEFAULT_GRACE_DIR = DEFAULT_RAW_DIR / "grace"
DEFAULT_ERA5_DIR = DEFAULT_RAW_DIR / "era5"
DEFAULT_MCD64A1_DIR = DEFAULT_RAW_DIR / "mcd64a1"


def default_bbox_text() -> str:
    return "-130.937500,17.817045,-74.375000,50.797242"


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def format_bbox_id(bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    return f"w{west:.4f}_s{south:.4f}_e{east:.4f}_n{north:.4f}".replace("-", "m").replace(".", "p")


def download_id(
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
    return f"{slug(product)}_{start:%Y%m%d}_{end:%Y%m%d}_{format_bbox_id(bbox)}_{digest}"


def manifest_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_suffix(path.suffix + ".manifest.json")


def write_manifest(output_path: str | Path, metadata: dict) -> None:
    manifest = dict(metadata)
    manifest["output"] = str(output_path)
    manifest["download_id"] = manifest.get("download_id") or Path(output_path).stem
    manifest_path(output_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def existing_matching_file(path: str | Path, expected_download_id: str) -> bool:
    output = Path(path)
    manifest = manifest_path(output)
    if not output.is_file() or not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("download_id") == expected_download_id


def _bbox_close(
    left: tuple[float, float, float, float] | list[float],
    right: tuple[float, float, float, float] | list[float],
    tol: float = 1e-4,
) -> bool:
    if left is None or right is None or len(left) != 4 or len(right) != 4:
        return False
    return all(abs(float(a) - float(b)) <= tol for a, b in zip(left, right))


def resolve_local_cache(
    preferred_path: str | Path,
    expected_download_id: str,
    *,
    product: str,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    required_fields: dict | None = None,
    allow_covering_dates: bool = True,
) -> Path | None:
    """Return a local file that satisfies the request without re-downloading.

    Prefers an exact download_id match, then any same-product/bbox cache whose
    date range fully covers the requested window (so shrinking dates reuses data).
    """

    preferred = Path(preferred_path)
    if existing_matching_file(preferred, expected_download_id):
        return preferred
    if preferred.is_file() and not manifest_path(preferred).is_file():
        # Legacy file without manifest: reuse only for the exact preferred path.
        return preferred

    directory = preferred.parent
    if not directory.is_dir():
        return None

    required_fields = required_fields or {}
    best_path: Path | None = None
    best_span_days: int | None = None

    for manifest in directory.glob("*.manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(data.get("product", "")).lower() != str(product).lower():
            continue
        if not _bbox_close(data.get("bbox") or [], bbox):
            continue
        if any(str(data.get(key, "")) != str(value) for key, value in required_fields.items()):
            continue

        output = Path(data.get("output") or "")
        if not output.is_file():
            # Common layout: foo.csv.manifest.json -> foo.csv
            stem = manifest.name
            if stem.endswith(".manifest.json"):
                output = manifest.with_name(stem[: -len(".manifest.json")])
        if not output.is_file():
            continue

        if data.get("download_id") == expected_download_id:
            return output

        if not allow_covering_dates:
            continue
        try:
            cached_start = date.fromisoformat(str(data.get("start")))
            cached_end = date.fromisoformat(str(data.get("end")))
        except ValueError:
            continue
        if cached_start <= start and cached_end >= end:
            span = (cached_end - cached_start).days
            if best_span_days is None or span < best_span_days:
                best_span_days = span
                best_path = output
    return best_path


def read_env_value(name: str, env_path: str | Path = ".env") -> str:
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


def bbox_from_folium_drawings(drawings) -> tuple[float, float, float, float] | None:
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


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("Bounding box must contain four comma-separated numbers.")
    west, south, east, north = parts
    if west >= east or south >= north:
        raise ValueError("Bounding box must be west,south,east,north.")
    return west, south, east, north


def bbox_picker(label: str, key: str, default_bbox: str | None = None) -> str:
    default_text = default_bbox or default_bbox_text()
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

    west, south, east, north = parse_bbox(st.session_state.get(f"{key}_bbox_text", default_text))
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
    drawn_bbox = bbox_from_folium_drawings(result.get("all_drawings"))
    if drawn_bbox:
        default_text = ",".join(f"{value:.6f}" for value in drawn_bbox)
        st.session_state[f"{key}_bbox_text"] = default_text
        st.success(f"Selected bbox: {default_text}")

    return st.text_input(
        "Bounding box west,south,east,north",
        value=st.session_state.get(f"{key}_bbox_text", default_text),
        key=f"{key}_bbox_text",
    )


def ensure_dir(path: str | Path) -> Path:
    out = Path(path).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    return out


def files_by_mtime(directory: str | Path, patterns: list[str]) -> list[Path]:
    path = Path(directory)
    if not path.is_dir():
        return []
    files: list[Path] = []
    for pattern in patterns:
        files.extend(item for item in path.rglob(pattern) if item.is_file())
    return sorted(set(files), key=lambda item: item.stat().st_mtime, reverse=True)


def discover_feature_inputs(raw_dir: str | Path = DEFAULT_RAW_DIR) -> dict:
    root = Path(raw_dir)
    firms_candidates = []
    exact_firms = root / "firms_north_america.csv"
    if exact_firms.is_file():
        firms_candidates.append(exact_firms)
    firms_candidates.extend(files_by_mtime(root / "firms", ["*.csv"]))
    firms_candidates.extend(
        path for path in files_by_mtime(root, ["*firms*.csv", "*fire*.csv"]) if path != exact_firms
    )
    grace_files = files_by_mtime(root / "grace", ["*.nc", "*.nc4", "*.cdf"])
    era5_files = files_by_mtime(root / "era5", ["*.nc", "*.nc4", "*.cdf"])
    burned_candidates = files_by_mtime(root / "mcd64a1", ["*burn*.csv", "*mcd64*.csv", "*.csv"])
    return {
        "firms_csv": str(firms_candidates[0]) if firms_candidates else "",
        "grace_files": [str(path) for path in sorted(grace_files)],
        "era5_files": [str(path) for path in sorted(era5_files)],
        "burned_area_csv": str(burned_candidates[0]) if burned_candidates else "",
    }
