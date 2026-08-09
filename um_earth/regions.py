"""Shared region parsing for KML- and CSV-defined forecast domains."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import re
from pathlib import Path
import xml.etree.ElementTree as ET


KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


@dataclass(frozen=True)
class RegionDefinition:
    """Normalized region definition used across the pipeline."""

    region_id: str
    name: str
    polygon: list[list[float]]
    source: Path

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        lons = [point[0] for point in self.polygon]
        lats = [point[1] for point in self.polygon]
        return (min(lons), min(lats), max(lons), max(lats))

    @property
    def center(self) -> dict[str, float]:
        lon_min, lat_min, lon_max, lat_max = self.bounds
        return {
            "longitude": (lon_min + lon_max) / 2.0,
            "latitude": (lat_min + lat_max) / 2.0,
        }

    def extents_meters(self) -> dict[str, float]:
        import math

        lon_min, lat_min, lon_max, lat_max = self.bounds
        avg_lat = (lat_min + lat_max) / 2.0
        return {
            "x2_extent": (lon_max - lon_min) * 111000.0 * math.cos(math.radians(avg_lat)),
            "x3_extent": (lat_max - lat_min) * 111000.0,
        }


def slugify_identifier(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-")
    if not text:
        raise ValueError("Could not derive a valid region identifier")
    return text.lower()


def _parse_coordinate_text(raw_text: str) -> list[list[float]]:
    points: list[list[float]] = []
    for chunk in raw_text.split():
        pieces = chunk.split(",")
        if len(pieces) < 2:
            continue
        points.append([float(pieces[0]), float(pieces[1])])
    return points


def load_region_from_kml(kml_file: str | Path, region_id: str | None = None) -> RegionDefinition:
    path = Path(kml_file).resolve()
    root = ET.parse(path).getroot()

    polygons: list[list[list[float]]] = []
    for elem in root.findall(".//kml:Polygon//kml:coordinates", KML_NS):
        coords = _parse_coordinate_text(elem.text or "")
        if coords:
            polygons.append(coords)

    if not polygons:
        # Fall back to all coordinate sequences if the KML is line-heavy.
        all_points: list[list[float]] = []
        for elem in root.findall(".//kml:coordinates", KML_NS):
            all_points.extend(_parse_coordinate_text(elem.text or ""))
        if not all_points:
            raise ValueError(f"No coordinates found in KML file: {path}")
        polygons = [all_points]

    polygon = max(polygons, key=len)
    name_node = root.find(".//kml:Document/kml:name", KML_NS)
    name = (name_node.text or path.stem).strip() if name_node is not None else path.stem

    return RegionDefinition(
        region_id=region_id or slugify_identifier(path.stem),
        name=name,
        polygon=polygon,
        source=path,
    )


def load_regions_from_csv(locations_file: str | Path) -> dict[str, RegionDefinition]:
    path = Path(locations_file).resolve()
    with path.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.strip().startswith("#") and line.strip()]

    reader = csv.DictReader(io.StringIO("".join(lines)), delimiter=",", skipinitialspace=True)
    regions: dict[str, RegionDefinition] = {}
    for row in reader:
        latmin = float(row["Latmin"])
        latmax = float(row["Latmax"])
        lonmin = float(row["Lonmin"])
        lonmax = float(row["Lonmax"])
        polygon = [
            [lonmin, latmin],
            [lonmin, latmax],
            [lonmax, latmax],
            [lonmax, latmin],
        ]
        region_id = row["Name"].strip()
        regions[region_id] = RegionDefinition(
            region_id=region_id,
            name=row["Description"].strip(),
            polygon=polygon,
            source=path,
        )
    return regions


def load_region(
    *,
    region_kml: str | Path | None = None,
    location_id: str | None = None,
    locations_file: str | Path | None = None,
) -> RegionDefinition:
    if region_kml:
        return load_region_from_kml(region_kml, region_id=location_id)
    if not location_id:
        raise ValueError("location_id is required when region_kml is not provided")
    if not locations_file:
        raise ValueError("locations_file is required when region_kml is not provided")
    regions = load_regions_from_csv(locations_file)
    if location_id not in regions:
        available = ", ".join(sorted(regions))
        raise ValueError(f"Unknown location '{location_id}'. Available: {available}")
    return regions[location_id]
