#!/usr/bin/env python3
"""Plot the staged topography products in a 2x2 comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from netCDF4 import Dataset

from um_earth.diagnostics.topo_utils import load_topography
from um_earth.regions import load_region_from_kml


TOPO_LABELS = ("2p4km", "1p2km", "0p6km", "0p3km")
EARTH_RADIUS = 6_371_000.0


def discover_topography_files(products_dir: Path, region: str) -> list[Path]:
    files = []
    for label in TOPO_LABELS:
        topo_file = products_dir / f"{region}_topo_{label}.pt"
        if not topo_file.exists():
            raise FileNotFoundError(f"Missing topography file: {topo_file}")
        files.append(topo_file)
    return files


def kml_bounding_box(kml_file: Path) -> tuple[float, float, float, float]:
    region = load_region_from_kml(kml_file)
    lons = [point[0] for point in region.polygon]
    lats = [point[1] for point in region.polygon]
    return min(lons), max(lons), min(lats), max(lats)


def meters_to_lonlat(
    x2f: np.ndarray,
    x3f: np.ndarray,
    center_lon: float,
    center_lat: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    x2_centered = x2f - 0.5 * (float(x2f[0]) + float(x2f[-1]))
    x3_centered = x3f - 0.5 * (float(x3f[0]) + float(x3f[-1]))
    lat = center_lat + np.degrees(x3_centered / radius)
    lon = center_lon + np.degrees(x2_centered / (radius * np.cos(np.radians(center_lat))))
    return lon, lat


def load_forecast_grid_overlay(regridded_file: Path) -> dict[str, np.ndarray | float]:
    with Dataset(regridded_file) as ds:
        x2f = np.asarray(ds.variables["x2f"][:], dtype=float)
        x3f = np.asarray(ds.variables["x3f"][:], dtype=float)
        center_lon = float(getattr(ds, "center_longitude"))
        center_lat = float(getattr(ds, "center_latitude"))
        radius = float(getattr(ds, "planet_radius", EARTH_RADIUS))

    lonf, latf = meters_to_lonlat(x2f, x3f, center_lon, center_lat, radius)
    return {
        "lonf": lonf,
        "latf": latf,
    }


def infer_cell_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    deltas = np.diff(centers)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * deltas[0]
    edges[-1] = centers[-1] + 0.5 * deltas[-1]
    return edges


def load_source_grid_overlay(source_grid_file: Path) -> dict[str, np.ndarray]:
    with Dataset(source_grid_file) as ds:
        lat = np.asarray(ds.variables["latitude"][:], dtype=float)
        lon = np.asarray(ds.variables["longitude"][:], dtype=float)
    return {
        "lonf": infer_cell_edges(lon),
        "latf": infer_cell_edges(lat),
    }


def draw_grid_lines(
    axis: plt.Axes,
    lonf: np.ndarray,
    latf: np.ndarray,
    *,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    for lon in lonf:
        axis.plot([lon, lon], [latf[0], latf[-1]], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    for lat in latf:
        axis.plot([lonf[0], lonf[-1]], [lat, lat], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def draw_forecast_grid(
    axis: plt.Axes,
    coarse_overlay: dict[str, np.ndarray | float],
    source_overlay: dict[str, np.ndarray] | None,
) -> None:
    lonf = np.asarray(coarse_overlay["lonf"], dtype=float)
    latf = np.asarray(coarse_overlay["latf"], dtype=float)

    if source_overlay is not None:
        source_lonf = np.asarray(source_overlay["lonf"], dtype=float)
        source_latf = np.asarray(source_overlay["latf"], dtype=float)
        x0, x1 = axis.get_xlim()
        y0, y1 = axis.get_ylim()
        lon_min, lon_max = min(x0, x1), max(x0, x1)
        lat_min, lat_max = min(y0, y1), max(y0, y1)
        source_lonf = source_lonf[(source_lonf >= lon_min - 0.25) & (source_lonf <= lon_max + 0.25)]
        source_latf = source_latf[(source_latf >= lat_min - 0.25) & (source_latf <= lat_max + 0.25)]
        draw_grid_lines(
            axis,
            source_lonf,
            source_latf,
            color="#bdbdbd",
            linewidth=0.45,
            alpha=0.7,
            zorder=2,
        )
    else:
        source_lonf = source_latf = None

    draw_grid_lines(
        axis,
        lonf,
        latf,
        color="#8a8a8a",
        linewidth=0.35,
        alpha=0.45,
        zorder=3,
    )

    ic = max(0, len(lonf) // 2 - 1)
    jc = max(0, len(latf) // 2 - 1)
    coarse_lon0 = lonf[ic]
    coarse_lon1 = lonf[ic + 1]
    coarse_lat0 = latf[jc]
    coarse_lat1 = latf[jc + 1]
    inner_lonf = np.array([0.5 * (coarse_lon0 + coarse_lon1)])
    inner_latf = np.array([0.5 * (coarse_lat0 + coarse_lat1)])

    axis.add_patch(
        Rectangle(
            (coarse_lon0, coarse_lat0),
            coarse_lon1 - coarse_lon0,
            coarse_lat1 - coarse_lat0,
            fill=False,
            edgecolor="#4d4d4d",
            linewidth=1.0,
            zorder=4,
        )
    )

    for lon in inner_lonf:
        axis.plot([lon, lon], [coarse_lat0, coarse_lat1], color="#4d4d4d", linewidth=0.8, zorder=4)
    for lat in inner_latf:
        axis.plot([coarse_lon0, coarse_lon1], [lat, lat], color="#4d4d4d", linewidth=0.8, zorder=4)


def plot_topography_stages(
    products_dir: Path,
    region: str,
    output_path: Path,
    kml_file: Path | None = None,
    regridded_file: Path | None = None,
    source_grid_file: Path | None = None,
) -> None:
    topo_files = discover_topography_files(products_dir, region)
    topo_payloads = [load_topography(str(path)) for path in topo_files]
    topo_maps = [payload["topography"].astype(float) for payload in topo_payloads]
    bbox = kml_bounding_box(kml_file) if kml_file is not None else None
    overlay = load_forecast_grid_overlay(regridded_file) if regridded_file is not None else None
    source_overlay = load_source_grid_overlay(source_grid_file) if source_grid_file is not None else None

    global_min = min(float(np.min(topo)) for topo in topo_maps)
    global_max = max(float(np.max(topo)) for topo in topo_maps)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for axis, label, payload, topo in zip(axes.flat, TOPO_LABELS, topo_payloads, topo_maps):
        lon_bounds = payload["lon_bounds"]
        lat_bounds = payload["lat_bounds"]
        row0_is_north = bool(payload.get("row0_is_north"))
        extent = None
        if lon_bounds is not None and lat_bounds is not None:
            extent = [
                float(lon_bounds[0]),
                float(lon_bounds[1]),
                float(lat_bounds[0]),
                float(lat_bounds[1]),
            ]
        image = axis.imshow(
            topo,
            cmap="terrain",
            vmin=global_min,
            vmax=global_max,
            aspect="auto",
            extent=extent,
            origin="upper" if row0_is_north else "lower",
        )
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        if bbox is not None:
            lon_min, lon_max, lat_min, lat_max = bbox
            axis.add_patch(
                Rectangle(
                    (lon_min, lat_min),
                    lon_max - lon_min,
                    lat_max - lat_min,
                    fill=False,
                    edgecolor="red",
                    linewidth=2.0,
                )
            )
        if overlay is not None:
            draw_forecast_grid(axis, overlay, source_overlay)
        axis.text(
            0.02,
            0.02,
            f"{label}\nshape={topo.shape[1]} x {topo.shape[0]}",
            transform=axis.transAxes,
            fontsize=9,
            color="white",
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 3},
        )

    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    colorbar.set_label("Elevation (m)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a 2x2 plot of the staged topography resolutions."
    )
    parser.add_argument(
        "--products-dir",
        required=True,
        help="Directory containing <region>_topo_<label>.pt products.",
    )
    parser.add_argument(
        "--region",
        required=True,
        help="Region prefix used in the topography product filenames.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output image path.",
    )
    parser.add_argument(
        "--kml-file",
        help="Optional KML file whose bounding region will be drawn as a red box.",
    )
    parser.add_argument(
        "--regridded-file",
        help="Optional regridded forecast NetCDF whose x2f/x3f cell boundaries are overlaid.",
    )
    parser.add_argument(
        "--source-grid-file",
        help="Optional source forecast NetCDF whose native latitude/longitude grid is overlaid.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    plot_topography_stages(
        products_dir=Path(args.products_dir),
        region=args.region,
        output_path=Path(args.output),
        kml_file=Path(args.kml_file) if args.kml_file else None,
        regridded_file=Path(args.regridded_file) if args.regridded_file else None,
        source_grid_file=Path(args.source_grid_file) if args.source_grid_file else None,
    )
    print(f"Wrote topography stage plot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
