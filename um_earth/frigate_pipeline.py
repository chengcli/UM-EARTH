"""KML-driven FRIGATE preparation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import yaml

from .configuration import ConfigOptions, render_config, validate_date_format
from .regions import RegionDefinition, load_region


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = PROJECT_ROOT / "config_template.yaml"
DEFAULT_WORKSPACE_ROOT = Path("/home/chengcli/data/2025.FRIGATE") / "runs"
DEFAULT_MIN_DOMAIN_DEGREES = 1.2
DEFAULT_TARGET_RESOLUTIONS_KM = (2.4, 1.2, 0.6, 0.3)
DEFAULT_ERA5_TIMES = ("00:00", "06:00", "12:00", "18:00")
DEFAULT_X1_MAX_METERS = 10_000.0
DEFAULT_NX1 = 50
DEFAULT_TIMEOUT = 3600


@dataclass(frozen=True)
class PreparedDomain:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    x2_extent_meters: float
    x3_extent_meters: float
    nx2: int
    nx3: int
    dx2_meters: float
    dx3_meters: float


def build_resolution_products(**kwargs):
    from .topography.split_merge import build_resolution_products as impl

    return impl(**kwargs)


def _meters_per_degree_lon(latitude: float) -> float:
    return 111_000.0 * math.cos(math.radians(latitude))


def _bounds_extents_meters(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    lon_min, lat_min, lon_max, lat_max = bounds
    avg_lat = (lat_min + lat_max) / 2.0
    return (
        (lat_max - lat_min) * 111_000.0,
        (lon_max - lon_min) * _meters_per_degree_lon(avg_lat),
    )


def expand_bounds(
    bounds: tuple[float, float, float, float],
    min_size_degrees: float = DEFAULT_MIN_DOMAIN_DEGREES,
) -> tuple[float, float, float, float]:
    lon_min, lat_min, lon_max, lat_max = bounds
    center_lon = (lon_min + lon_max) / 2.0
    center_lat = (lat_min + lat_max) / 2.0
    lon_span = max(lon_max - lon_min, min_size_degrees)
    lat_span = max(lat_max - lat_min, min_size_degrees)

    return (
        max(-180.0, center_lon - lon_span / 2.0),
        max(-90.0, center_lat - lat_span / 2.0),
        min(180.0, center_lon + lon_span / 2.0),
        min(90.0, center_lat + lat_span / 2.0),
    )


def build_prepared_domain(
    region: RegionDefinition,
    base_resolution_km: float = DEFAULT_TARGET_RESOLUTIONS_KM[0],
    min_size_degrees: float = DEFAULT_MIN_DOMAIN_DEGREES,
) -> PreparedDomain:
    padded_bounds = expand_bounds(region.bounds, min_size_degrees=min_size_degrees)
    x2_extent, x3_extent = _bounds_extents_meters(padded_bounds)
    spacing_meters = base_resolution_km * 1000.0
    nx2 = max(1, round(x2_extent / spacing_meters))
    nx3 = max(1, round(x3_extent / spacing_meters))
    return PreparedDomain(
        lat_min=padded_bounds[1],
        lat_max=padded_bounds[3],
        lon_min=padded_bounds[0],
        lon_max=padded_bounds[2],
        x2_extent_meters=x2_extent,
        x3_extent_meters=x3_extent,
        nx2=nx2,
        nx3=nx3,
        dx2_meters=x2_extent / nx2,
        dx3_meters=x3_extent / nx3,
    )


def write_region_digest(
    output_file: Path,
    region: RegionDefinition,
    prepared: PreparedDomain,
    *,
    date: str,
    era5_times: Iterable[str] = DEFAULT_ERA5_TIMES,
    target_resolutions_km: Iterable[float] = DEFAULT_TARGET_RESOLUTIONS_KM,
) -> dict[str, object]:
    native_bounds = region.bounds
    native_extents = region.extents_meters()
    payload: dict[str, object] = {
        "region_id": region.region_id,
        "name": region.name,
        "source_kml": str(region.source),
        "polygon_vertex_count": len(region.polygon),
        "date": date,
        "requested_era5_times_utc": list(era5_times),
        "target_resolutions_km": list(target_resolutions_km),
        "bounds": {
            "lon_min": native_bounds[0],
            "lat_min": native_bounds[1],
            "lon_max": native_bounds[2],
            "lat_max": native_bounds[3],
        },
        "center": region.center,
        "kml_extent_degrees": {
            "latitude": native_bounds[3] - native_bounds[1],
            "longitude": native_bounds[2] - native_bounds[0],
        },
        "kml_extent_meters": native_extents,
        "padded_bounds": {
            "lon_min": prepared.lon_min,
            "lat_min": prepared.lat_min,
            "lon_max": prepared.lon_max,
            "lat_max": prepared.lat_max,
        },
        "padded_extent_degrees": {
            "latitude": prepared.lat_max - prepared.lat_min,
            "longitude": prepared.lon_max - prepared.lon_min,
        },
        "padded_extent_meters": {
            "x2_extent": prepared.x2_extent_meters,
            "x3_extent": prepared.x3_extent_meters,
        },
        "simulation_grid": {
            "nx1": DEFAULT_NX1,
            "nx2": prepared.nx2,
            "nx3": prepared.nx3,
            "x1_max_meters": DEFAULT_X1_MAX_METERS,
            "dx2_meters": prepared.dx2_meters,
            "dx3_meters": prepared.dx3_meters,
        },
    }
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_config(
    output_file: Path,
    region: RegionDefinition,
    prepared: PreparedDomain,
    *,
    date: str,
    template_file: Path = DEFAULT_TEMPLATE,
    nb2: int = 1,
    nb3: int = 1,
) -> str:
    rendered = render_config(
        template_file.read_text(encoding="utf-8"),
        region,
        ConfigOptions(
            start_date=date,
            end_date=date,
            nx1=DEFAULT_NX1,
            nx2=prepared.nx2,
            nx3=prepared.nx3,
            x1_max=DEFAULT_X1_MAX_METERS,
            x2_extent=prepared.x2_extent_meters,
            x3_extent=prepared.x3_extent_meters,
            nb2=nb2,
            nb3=nb3,
        ),
    )
    output_file.write_text(rendered, encoding="utf-8")
    return rendered


def _run_python(script: Path, args: list[str]) -> None:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not pythonpath else str(PROJECT_ROOT) + os.pathsep + pythonpath
    subprocess.run([sys.executable, str(script), *args], check=True, cwd=PROJECT_ROOT, env=env)


def run_topography_download(region: RegionDefinition, raw_out_dir: Path, *, skip_download: bool = False) -> None:
    script = PROJECT_ROOT / "um_earth" / "topography" / "download_crop.py"
    args = ["--region-kml", str(region.source), region.region_id, "--out", str(raw_out_dir)]
    if skip_download:
        args.append("--skip-download")
    _run_python(script, args)


def run_initial_condition_pipeline(
    region: RegionDefinition,
    config_path: Path,
    era5_out_dir: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    times: Iterable[str] = DEFAULT_ERA5_TIMES,
    data_source: str = "era5",
    forecast_input_dir: str | None = None,
    forecast_cycle: str | None = None,
    forecast_leads: Iterable[int] | None = None,
    decompose_ny: int = 1,
    decompose_nx: int = 1,
) -> None:
    script = PROJECT_ROOT / "prepare_initial_condition.py"
    args = [
        region.region_id,
        "--region-kml",
        str(region.source),
        "--config",
        str(config_path),
        "--output-base",
        str(era5_out_dir),
        "--timeout",
        str(timeout),
        "--data-source",
        data_source,
    ]
    args.extend(["--nY", str(decompose_ny), "--nX", str(decompose_nx)])
    if data_source == "era5":
        args.extend(["--times", *list(times)])
    else:
        if forecast_input_dir is None:
            raise ValueError("forecast_input_dir is required when data_source='forecast'")
        args.extend(["--forecast-input-dir", forecast_input_dir])
        if forecast_cycle is not None:
            args.extend(["--forecast-cycle", forecast_cycle])
        if forecast_leads is not None:
            args.extend(["--forecast-leads", *(str(hour) for hour in forecast_leads)])
    _run_python(script, args)


def find_era5_output_dir(output_base: Path) -> Path | None:
    if output_base.exists() and any(output_base.glob("*_hourly_dynamics_*.nc")):
        return output_base
    candidates = [path for path in output_base.glob("*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _count_matching_files(directory: Path | None, pattern: str) -> int:
    if directory is None or not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def generate_verification_plots(
    plots_dir: Path,
    region: RegionDefinition,
    prepared: PreparedDomain,
    *,
    date: str,
    era5_times: Iterable[str],
    era5_output_dir: Path | None,
    topography_products: dict[str, Path],
) -> None:
    import matplotlib.pyplot as plt
    import torch

    plots_dir.mkdir(parents=True, exist_ok=True)
    region_lon = [point[0] for point in region.polygon]
    region_lat = [point[1] for point in region.polygon]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([*region_lon, region_lon[0]], [*region_lat, region_lat[0]], color="tab:blue", linewidth=2, label="KML polygon")
    ax.plot(
        [prepared.lon_min, prepared.lon_max, prepared.lon_max, prepared.lon_min, prepared.lon_min],
        [prepared.lat_min, prepared.lat_min, prepared.lat_max, prepared.lat_max, prepared.lat_min],
        color="tab:orange",
        linewidth=2,
        label="Padded domain",
    )
    ax.scatter(region.center["longitude"], region.center["latitude"], color="black", s=25, label="Center")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{region.region_id} domain overview")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "kml_domain_overview.png", dpi=200)
    plt.close(fig)

    dynamics_count = _count_matching_files(era5_output_dir, "*_hourly_dynamics_*.nc")
    densities_count = _count_matching_files(era5_output_dir, "*_hourly_densities_*.nc")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["densities", "dynamics"], [densities_count, dynamics_count], color=["tab:green", "tab:purple"])
    ax.set_ylabel("Files")
    ax.set_title(f"ECMWF input coverage for {date}")
    ax.text(0.02, 0.95, "UTC times: " + ", ".join(era5_times), transform=ax.transAxes, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(plots_dir / "era5_fetch_summary.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (label, path) in zip(axes.flat, sorted(topography_products.items())):
        module = torch.jit.load(str(path))
        topo = module.topography.detach().cpu().numpy()
        image = ax.imshow(topo, cmap="terrain", origin="upper")
        ax.set_title(label)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(f"Topography products for {region.region_id}")
    fig.tight_layout()
    fig.savefig(plots_dir / "topography_resolutions.png", dpi=200)
    plt.close(fig)


def write_run_manifest(
    output_file: Path,
    *,
    region: RegionDefinition,
    date: str,
    run_dir: Path,
    config_path: Path,
    digest_path: Path,
    topography_products: dict[str, Path],
    era5_output_dir: Path | None,
    data_source: str = "era5",
    forecast_input_dir: str | None = None,
    forecast_cycle: str | None = None,
    forecast_leads: Iterable[int] | None = None,
) -> None:
    payload = {
        "region_id": region.region_id,
        "date": date,
        "data_source": data_source,
        "run_dir": str(run_dir),
        "simulation_input_path": str(config_path),
        "digest_path": str(digest_path),
        "era5_output_dir": str(era5_output_dir) if era5_output_dir else None,
        "forecast_input_dir": forecast_input_dir,
        "forecast_cycle": forecast_cycle,
        "forecast_leads": list(forecast_leads) if forecast_leads is not None else None,
        "topography_products": {label: str(path) for label, path in topography_products.items()},
    }
    output_file.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def run_frigate_prepare(
    *,
    region_kml: str,
    date: str,
    workspace_root: str | Path = DEFAULT_WORKSPACE_ROOT,
    location_id: str | None = None,
    skip_download: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    data_source: str = "era5",
    forecast_input_dir: str | None = None,
    forecast_cycle: str | None = None,
    forecast_leads: Iterable[int] | None = None,
    decompose_ny: int = 1,
    decompose_nx: int = 1,
) -> Path:
    validate_date_format(date)
    region = load_region(region_kml=region_kml, location_id=location_id)
    prepared = build_prepared_domain(region)

    run_dir = Path(workspace_root) / f"{region.region_id}-{date}"
    raw_topography_dir = run_dir / "topography" / "raw"
    refined_topography_dir = run_dir / "topography" / "products"
    era5_out_dir = run_dir / "era5"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)

    digest_path = run_dir / "region_digest.json"
    write_region_digest(digest_path, region, prepared, date=date)

    # This single YAML is both the pipeline config and the simulation input.
    config_path = run_dir / f"{region.region_id}.yaml"
    write_config(
        config_path,
        region,
        prepared,
        date=date,
        nb2=decompose_ny,
        nb3=decompose_nx,
    )

    run_topography_download(region, raw_topography_dir, skip_download=skip_download)
    merged_tif = raw_topography_dir / region.region_id / f"{region.region_id}_merged_10m.tif"
    if not merged_tif.exists():
        raise FileNotFoundError(f"Expected merged topography file was not created: {merged_tif}")

    topography_products = build_resolution_products(
        merged_tif=merged_tif,
        out_dir=refined_topography_dir,
        region_id=region.region_id,
        lat_bounds=(prepared.lat_min, prepared.lat_max),
        lon_bounds=(prepared.lon_min, prepared.lon_max),
        target_resolutions_km=DEFAULT_TARGET_RESOLUTIONS_KM,
    )

    run_initial_condition_pipeline(
        region,
        config_path,
        era5_out_dir,
        timeout=timeout,
        times=DEFAULT_ERA5_TIMES,
        data_source=data_source,
        forecast_input_dir=forecast_input_dir,
        forecast_cycle=forecast_cycle,
        forecast_leads=forecast_leads,
        decompose_ny=decompose_ny,
        decompose_nx=decompose_nx,
    )
    era5_output_dir = find_era5_output_dir(era5_out_dir)

    generate_verification_plots(
        plots_dir,
        region,
        prepared,
        date=date,
        era5_times=DEFAULT_ERA5_TIMES,
        era5_output_dir=era5_output_dir,
        topography_products=topography_products,
    )
    write_run_manifest(
        run_dir / "run_manifest.yaml",
        region=region,
        date=date,
        run_dir=run_dir,
        config_path=config_path,
        digest_path=digest_path,
        topography_products=topography_products,
        era5_output_dir=era5_output_dir,
        data_source=data_source,
        forecast_input_dir=forecast_input_dir,
        forecast_cycle=forecast_cycle,
        forecast_leads=forecast_leads,
    )
    return run_dir
