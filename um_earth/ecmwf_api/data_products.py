"""Helpers for ECMWF intermediate file naming and discovery."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from pathlib import Path


VALID_DATA_SOURCES = ("era5", "forecast")


@dataclass(frozen=True)
class ProductFileSet:
    source: str
    stem: str
    dynamics: str
    densities: str
    density: str | None = None


def validate_data_source(source: str) -> str:
    normalized = str(source).strip().lower()
    if normalized not in VALID_DATA_SOURCES:
        raise ValueError(
            f"Unsupported data source: {source}. Expected one of {VALID_DATA_SOURCES}."
        )
    return normalized


def product_prefix(source: str) -> str:
    return validate_data_source(source)


def build_product_stem(date_str: str, cycle: str | None = None) -> str:
    return f"{date_str}_{cycle}" if cycle else date_str


def dynamics_filename(source: str, stem: str) -> str:
    return f"{product_prefix(source)}_hourly_dynamics_{stem}.nc"


def densities_filename(source: str, stem: str) -> str:
    return f"{product_prefix(source)}_hourly_densities_{stem}.nc"


def density_filename(source: str, stem: str) -> str:
    return f"{product_prefix(source)}_density_{stem}.nc"


def discover_product_files(
    data_dir: str | os.PathLike[str],
    stem: str | None = None,
    *,
    require_density: bool = True,
) -> dict[str, ProductFileSet]:
    """Discover ERA5/forecast intermediate NetCDF products."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")

    if stem is not None:
        patterns = [f"*_hourly_dynamics_{stem}.nc"]
    else:
        patterns = ["*_hourly_dynamics_*.nc"]

    dynamics_files: list[str] = []
    for pattern in patterns:
        dynamics_files.extend(sorted(glob.glob(str(root / pattern))))

    if not dynamics_files:
        expected = f"*_hourly_dynamics_{stem}.nc" if stem else "*_hourly_dynamics_*.nc"
        raise FileNotFoundError(
            f"No dynamics files found in {root}. Expected files matching: {expected}"
        )

    discovered: dict[str, ProductFileSet] = {}
    missing: list[str] = []
    for dynamics_path in dynamics_files:
        basename = os.path.basename(dynamics_path)
        prefix, suffix = basename.split("_hourly_dynamics_", 1)
        file_source = validate_data_source(prefix)
        file_stem = suffix.removesuffix(".nc")

        densities_path = root / densities_filename(file_source, file_stem)
        density_path = root / density_filename(file_source, file_stem)
        if not densities_path.exists():
            missing.append(densities_path.name)
            continue
        if require_density and not density_path.exists():
            missing.append(density_path.name)
            continue

        discovered[file_stem] = ProductFileSet(
            source=file_source,
            stem=file_stem,
            dynamics=dynamics_path,
            densities=str(densities_path),
            density=str(density_path) if density_path.exists() else None,
        )

    if missing:
        raise FileNotFoundError(
            f"Missing required ECMWF files in {root}:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )
    if not discovered:
        raise FileNotFoundError(f"No complete ECMWF product sets found in {root}")
    return discovered
