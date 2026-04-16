#!/usr/bin/env python3
"""Plot a two-panel common-level wind vector comparison for FRIGATE forecast cases."""

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


def _crop_mask(lon: np.ndarray, lat: np.ndarray, topo_lon: np.ndarray, topo_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon_pad = max(abs(float(np.diff(topo_lon).mean())), 0.25)
    lat_pad = max(abs(float(np.diff(topo_lat).mean())), 0.25)
    lon_mask = (lon >= topo_lon.min() - lon_pad) & (lon <= topo_lon.max() + lon_pad)
    lat_mask = (lat >= topo_lat.min() - lat_pad) & (lat <= topo_lat.max() + lat_pad)
    return lon_mask, lat_mask


def _crop_field_2d(field: np.ndarray, lon_mask: np.ndarray, lat_mask: np.ndarray) -> np.ndarray:
    return np.asarray(field[np.ix_(lat_mask, lon_mask)], dtype=float)


def _load_source_common_pressure_wind(
    source_pl_file: str,
    source_sfc_file: str,
    topo_lon: np.ndarray,
    topo_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    sfc = xr.open_dataset(source_sfc_file, engine="cfgrib", backend_kwargs={"indexpath": ""})
    sp = sfc["sp"]
    if "step" in sp.dims:
        sp = sp.isel(step=0)
    if "time" in sp.dims:
        sp = sp.isel(time=0)
    lon = np.asarray(sp["longitude"].values, dtype=float)
    lat = np.asarray(sp["latitude"].values, dtype=float)
    lon_mask, lat_mask = _crop_mask(lon, lat, topo_lon, topo_lat)
    sp_crop = _crop_field_2d(np.asarray(sp.values, dtype=float), lon_mask, lat_mask)
    sfc.close()

    pl = xr.open_dataset(source_pl_file, engine="cfgrib", backend_kwargs={"indexpath": ""})
    levels = np.asarray(pl["isobaricInhPa"].values, dtype=float)
    min_surface_pressure_hpa = float(np.nanmin(sp_crop) / 100.0)
    valid_levels = levels[levels <= min_surface_pressure_hpa]
    if valid_levels.size == 0:
        raise ValueError("No ECMWF pressure level lies above the surface everywhere in the plotted domain.")
    common_level_hpa = float(np.max(valid_levels))

    u = pl["u"].sel(isobaricInhPa=common_level_hpa)
    v = pl["v"].sel(isobaricInhPa=common_level_hpa)
    if "step" in u.dims:
        u = u.isel(step=0)
        v = v.isel(step=0)
    u_crop = _crop_field_2d(np.asarray(u.values, dtype=float), lon_mask, lat_mask)
    v_crop = _crop_field_2d(np.asarray(v.values, dtype=float), lon_mask, lat_mask)
    pl.close()
    return lon[lon_mask], lat[lat_mask], u_crop, v_crop, common_level_hpa


def _load_runtime_common_altitude_wind(
    runtime_file: str,
    topo: np.ndarray,
    config_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    ds = xr.open_dataset(runtime_file)
    vel2 = ds["vel2"].isel(time=0)
    vel3 = ds["vel3"].isel(time=0)
    lon, lat = _meters_to_lonlat(ds, config_file)
    x1 = np.asarray(ds["x1"].values, dtype=float)
    common_altitude_m = float(np.min(x1[x1 > np.max(topo)]))
    level_index = int(np.where(x1 == common_altitude_m)[0][0])
    u_level = np.asarray(vel2.isel(x1=level_index).values, dtype=float)
    v_level = np.asarray(vel3.isel(x1=level_index).values, dtype=float)
    ds.close()
    return lon, lat, u_level, v_level, common_altitude_m


def _quiver_slice(nx: int, ny: int, target: int = 20) -> tuple[slice, slice]:
    step_x = max(1, nx // target)
    step_y = max(1, ny // target)
    return slice(None, None, step_x), slice(None, None, step_y)


def _quiver_scale_reference(max_speed: float) -> float:
    # Smaller scale values produce longer arrows. Tie the scale to the plotted
    # wind range so vector length carries magnitude clearly across both panels.
    return max(1.0, max_speed * 14.0)


def plot_surface_wind(
    source_pl_file: str,
    source_sfc_file: str,
    runtime_file: str,
    config_file: str,
    topo_dir: str,
    location_prefix: str,
    kml_file: str | None = None,
    output_file: str | None = None,
) -> None:
    topo_for_runtime, _, _ = _load_named_topography(topo_dir, location_prefix, "2p4km")
    topo_for_overlay, topo_lon, topo_lat = _load_named_topography(topo_dir, location_prefix, "0p3km")

    left_lon, left_lat, left_u, left_v, common_level_hpa = _load_source_common_pressure_wind(
        source_pl_file, source_sfc_file, topo_lon, topo_lat
    )
    right_lon, right_lat, right_u, right_v, common_altitude_m = _load_runtime_common_altitude_wind(
        runtime_file, topo_for_runtime, config_file
    )

    left_speed = np.hypot(left_u, left_v)
    right_speed = np.hypot(right_u, right_v)
    topo_km = topo_for_overlay / 1000.0

    topo_X, topo_Y = np.meshgrid(topo_lon, topo_lat)
    left_X, left_Y = np.meshgrid(left_lon, left_lat)
    right_X, right_Y = np.meshgrid(right_lon, right_lat)

    vmin = min(float(np.nanmin(left_speed)), float(np.nanmin(right_speed)))
    vmax = max(float(np.nanmax(left_speed)), float(np.nanmax(right_speed)))
    bbox = _kml_bounding_box(kml_file) if kml_file is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    contour = None
    quiver_scale = _quiver_scale_reference(vmax)
    panels = (
        (axes[0], left_X, left_Y, left_u, left_v, left_speed, f"ECMWF {common_level_hpa:.0f} hPa"),
        (axes[1], right_X, right_Y, right_u, right_v, right_speed, f"FRIGATE z = {common_altitude_m:.0f} m"),
    )
    for ax, X, Y, u, v, speed, title in panels:
        contour = ax.contourf(X, Y, speed, levels=20, cmap="cividis", vmin=vmin, vmax=vmax, zorder=1)
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

        sx, sy = _quiver_slice(X.shape[1], X.shape[0], target=18)
        ax.quiver(
            X[sy, sx],
            Y[sy, sx],
            u[sy, sx],
            v[sy, sx],
            color="black",
            angles="xy",
            scale_units="xy",
            scale=quiver_scale,
            width=0.0022,
            headwidth=3.0,
            headlength=4.0,
            headaxislength=3.5,
            alpha=0.85,
            zorder=4,
        )
        ax.set_title(title)
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
                    zorder=5,
                )
            )

    axes[0].set_ylabel("Latitude")
    plt.colorbar(contour, ax=axes.ravel().tolist(), label="Wind speed (m s$^{-1}$)", shrink=0.7)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ECMWF common-pressure wind against FRIGATE common-altitude wind vectors."
    )
    parser.add_argument("source_pl_file", help="Original ECMWF pressure-level GRIB2 file containing u/v.")
    parser.add_argument("source_sfc_file", help="Original ECMWF surface GRIB2 file containing sp.")
    parser.add_argument("runtime_file", help="FRIGATE runtime out1 NetCDF file.")
    parser.add_argument("--config", required=True, help="FRIGATE YAML config used for lon/lat conversion.")
    parser.add_argument("--topo-dir", required=True, help="Directory containing topography .pt files.")
    parser.add_argument("--location", required=True, help="Location prefix for topography files (e.g., pte1b).")
    parser.add_argument("--kml-file", help="Optional KML file used to draw the red region box.")
    parser.add_argument("-o", "--output", help="Output plot file (PNG).")
    args = parser.parse_args()

    plot_surface_wind(
        args.source_pl_file,
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
