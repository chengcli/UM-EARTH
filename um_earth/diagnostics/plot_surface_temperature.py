#!/usr/bin/env python3
"""Plot a two-panel surface-temperature comparison for FRIGATE forecast cases."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from matplotlib.patches import Rectangle

try:
    from um_earth.diagnostics.topo_utils import load_topography
except ImportError:  # pragma: no cover - direct script execution fallback
    from topo_utils import load_topography
try:
    from um_earth.regions import load_region_from_kml
except ImportError:  # pragma: no cover - direct script execution fallback
    from regions import load_region_from_kml


EARTH_RADIUS = 6_371_000.0


def _load_center_from_config(config_file: str) -> tuple[float, float]:
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)
    geometry = config["geometry"]
    return float(geometry["center_longitude"]), float(geometry["center_latitude"])


def _kml_bounding_box(kml_file: str) -> tuple[float, float, float, float]:
    region = load_region_from_kml(kml_file)
    lons = [point[0] for point in region.polygon]
    lats = [point[1] for point in region.polygon]
    return min(lons), max(lons), min(lats), max(lats)


def _meters_to_lonlat(ds: xr.Dataset, config_file: str) -> tuple[np.ndarray, np.ndarray]:
    x2 = np.asarray(ds["x2"].values, dtype=float)
    x3 = np.asarray(ds["x3"].values, dtype=float)
    center_lon = ds.attrs.get("center_longitude")
    center_lat = ds.attrs.get("center_latitude")
    if center_lon is None or center_lat is None:
        center_lon, center_lat = _load_center_from_config(config_file)
    center_lon = float(center_lon)
    center_lat = float(center_lat)
    radius = float(ds.attrs.get("planet_radius", EARTH_RADIUS))

    x2_centered = x2 - 0.5 * (x2[0] + x2[-1])
    x3_centered = x3 - 0.5 * (x3[0] + x3[-1])
    lat = center_lat + np.degrees(x3_centered / radius)
    lon = center_lon + np.degrees(x2_centered / (radius * np.cos(np.radians(center_lat))))
    return lon, lat


def _load_named_topography(topo_dir: str, location_prefix: str, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topo_file = f"{topo_dir.rstrip('/')}/{location_prefix}_topo_{label}.pt"
    topo_data = load_topography(topo_file)
    topo = topo_data["topography"].astype(float)
    lon = np.asarray(topo_data["lon"], dtype=float)
    lat = np.asarray(topo_data["lat"], dtype=float)
    row0_is_north = bool(topo_data.get("row0_is_north"))
    if row0_is_north:
        topo = np.flip(topo, axis=0)
        lat = lat[::-1].copy()
    return topo, lon, lat


def _load_source_surface_temperature(
    source_sfc_file: str,
    topo_lon: np.ndarray,
    topo_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(source_sfc_file, engine="cfgrib", backend_kwargs={"indexpath": ""})
    if "t2m" not in ds:
        raise KeyError(f"Surface GRIB file does not contain 't2m': {source_sfc_file}")

    t2m = ds["t2m"]
    if "step" in t2m.dims:
        t2m = t2m.isel(step=0)
    if "time" in t2m.dims:
        t2m = t2m.isel(time=0)

    lat_name = "latitude" if "latitude" in t2m.coords else "lat"
    lon_name = "longitude" if "longitude" in t2m.coords else "lon"
    lat = np.asarray(t2m[lat_name].values, dtype=float)
    lon = np.asarray(t2m[lon_name].values, dtype=float)

    lon_pad = max(abs(float(np.diff(topo_lon).mean())), 0.25)
    lat_pad = max(abs(float(np.diff(topo_lat).mean())), 0.25)
    lon_mask = (lon >= topo_lon.min() - lon_pad) & (lon <= topo_lon.max() + lon_pad)
    lat_mask = (lat >= topo_lat.min() - lat_pad) & (lat <= topo_lat.max() + lat_pad)

    t2m_crop = np.asarray(t2m.values[np.ix_(lat_mask, lon_mask)], dtype=float)
    return lon[lon_mask], lat[lat_mask], t2m_crop


def _load_first_fluid_temperature(
    runtime_file: str,
    topo: np.ndarray,
    config_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(runtime_file)
    temp = ds["temp"].isel(time=0)
    lon, lat = _meters_to_lonlat(ds, config_file)
    x1 = np.asarray(ds["x1"].values, dtype=float)

    first_fluid = np.sum(x1[np.newaxis, np.newaxis, :] < topo[:, :, np.newaxis], axis=2)
    first_fluid = np.clip(first_fluid, 0, len(x1) - 1).astype(int)

    temp_values = np.asarray(temp.values, dtype=float)  # (x1, x3, x2)
    surface = np.take_along_axis(temp_values, first_fluid[np.newaxis, :, :], axis=0)[0]
    ds.close()
    return lon, lat, surface


def plot_surface_temperature(
    source_sfc_file: str,
    runtime_file: str,
    config_file: str,
    topo_dir: str,
    location_prefix: str,
    kml_file: str | None = None,
    output_file: str | None = None,
) -> None:
    topo_for_mask, _, _ = _load_named_topography(topo_dir, location_prefix, "2p4km")
    topo_for_overlay, topo_lon, topo_lat = _load_named_topography(topo_dir, location_prefix, "0p3km")
    left_lon, left_lat, left_temp = _load_source_surface_temperature(source_sfc_file, topo_lon, topo_lat)
    right_lon, right_lat, right_temp = _load_first_fluid_temperature(runtime_file, topo_for_mask, config_file)

    topo_km = topo_for_overlay / 1000.0
    topo_X, topo_Y = np.meshgrid(topo_lon, topo_lat)
    left_X, left_Y = np.meshgrid(left_lon, left_lat)
    right_X, right_Y = np.meshgrid(right_lon, right_lat)

    vmin = min(float(np.nanmin(left_temp)), float(np.nanmin(right_temp)))
    vmax = max(float(np.nanmax(left_temp)), float(np.nanmax(right_temp)))
    bbox = _kml_bounding_box(kml_file) if kml_file is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    contour = None
    for ax, X, Y, temp in (
        (axes[0], left_X, left_Y, left_temp),
        (axes[1], right_X, right_Y, right_temp),
    ):
        contour = ax.contourf(X, Y, temp, levels=20, cmap="inferno", vmin=vmin, vmax=vmax, zorder=1)
        contour_lines = ax.contour(
            X,
            Y,
            temp,
            levels=10,
            colors="black",
            linewidths=1.5,
            alpha=0.85,
            zorder=2,
        )
        ax.clabel(contour_lines, inline=True, fontsize=8)
        topo_lines = ax.contour(
            topo_X,
            topo_Y,
            topo_km,
            levels=8,
            colors="#6f6f6f",
            linewidths=1.0,
            alpha=0.8,
            linestyles="solid",
            zorder=3,
        )
        ax.clabel(topo_lines, inline=True, fontsize=7, fmt="%.1f km")
        ax.set_xlabel("Longitude")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        if bbox is not None:
            lon_min, lon_max, lat_min, lat_max = bbox
            ax.add_patch(
                Rectangle(
                    (lon_min, lat_min),
                    lon_max - lon_min,
                    lat_max - lat_min,
                    fill=False,
                    edgecolor="red",
                    linewidth=2.0,
                    zorder=4,
                )
            )

    axes[0].set_ylabel("Latitude")
    plt.colorbar(contour, ax=axes.ravel().tolist(), label="Temperature (K)", shrink=0.7)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ECMWF surface temperature against FRIGATE first-fluid-cell temperature."
    )
    parser.add_argument("source_sfc_file", help="Original ECMWF surface GRIB2 file containing 't2m'.")
    parser.add_argument("runtime_file", help="FRIGATE runtime out2 NetCDF file.")
    parser.add_argument("--config", required=True, help="FRIGATE YAML config used for lon/lat conversion.")
    parser.add_argument("--topo-dir", required=True, help="Directory containing topography .pt files.")
    parser.add_argument("--location", required=True, help="Location prefix for topography files (e.g., pte1b).")
    parser.add_argument("--kml-file", help="Optional KML file used to draw the red region box.")
    parser.add_argument("-o", "--output", help="Output plot file (PNG).")
    args = parser.parse_args()

    plot_surface_temperature(
        args.source_sfc_file,
        args.runtime_file,
        args.config,
        args.topo_dir,
        args.location,
        kml_file=args.kml_file,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
