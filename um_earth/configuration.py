"""Config generation helpers shared by CLI and legacy wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .regions import RegionDefinition


@dataclass(frozen=True)
class ConfigOptions:
    start_date: str
    end_date: str
    nx1: int
    nx2: int
    nx3: int
    nghost: int = 3
    x1_max: float = 20000.0
    x2_extent: float | None = None
    x3_extent: float | None = None
    tlim: int = 86400


def validate_date_format(date_str: str) -> None:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        raise ValueError(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.")


def render_config(template: str, region: RegionDefinition, options: ConfigOptions) -> str:
    validate_date_format(options.start_date)
    validate_date_format(options.end_date)

    extents = region.extents_meters()
    center = region.center
    x2_extent = options.x2_extent if options.x2_extent is not None else extents["x2_extent"]
    x3_extent = options.x3_extent if options.x3_extent is not None else extents["x3_extent"]

    replacements = {
        "location_name": region.name,
        "location_description": f"Location: {region.name}",
        "center_latitude": center["latitude"],
        "center_longitude": center["longitude"],
        "x1_max": options.x1_max,
        "x2_extent": x2_extent,
        "x3_extent": x3_extent,
        "x1_max_km": options.x1_max / 1000.0,
        "x2_extent_km": x2_extent / 1000.0,
        "x3_extent_km": x3_extent / 1000.0,
        "nx1": options.nx1,
        "nx2": options.nx2,
        "nx3": options.nx3,
        "nghost": options.nghost,
        "start_date": options.start_date,
        "end_date": options.end_date,
        "tlim": options.tlim,
    }

    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def default_output_path(base_dir: str | Path, region: RegionDefinition) -> Path:
    return Path(base_dir) / f"{region.region_id}.yaml"

