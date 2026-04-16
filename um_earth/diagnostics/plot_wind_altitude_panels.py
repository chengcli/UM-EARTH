#!/usr/bin/env python3
"""Plot horizontal wind vectors and vertical velocity at multiple altitudes."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, time as dtime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from scipy.interpolate import RegularGridInterpolator

try:
    from um_earth.diagnostics.topo_utils import load_topography
except ImportError:  # pragma: no cover
    from topo_utils import load_topography


EARTH_RADIUS = 6_371_000.0
TARGET_HEIGHTS_M = (2100.0, 2500.0, 2900.0, 3300.0, 3700.0, 4100.0)
VERTICAL_VELOCITY_LIMIT = 1.2


def _load_center_and_start(config_file: str) -> tuple[float, float, datetime]:
    with open(config_file, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    geometry = config["geometry"]
    integration = config["integration"]
    start_date = integration["start-date"]
    if hasattr(start_date, "year"):
        start_dt = datetime.combine(start_date, dtime.min, tzinfo=UTC)
    else:
        start_dt = datetime.strptime(str(start_date), "%Y-%m-%d").replace(tzinfo=UTC)
    return float(geometry["center_longitude"]), float(geometry["center_latitude"]), start_dt


def _meters_to_lonlat(ds: xr.Dataset, config_file: str) -> tuple[np.ndarray, np.ndarray, datetime]:
    x2 = np.asarray(ds["x2"].values, dtype=float)
    x3 = np.asarray(ds["x3"].values, dtype=float)
    center_lon = ds.attrs.get("center_longitude")
    center_lat = ds.attrs.get("center_latitude")
    config_lon, config_lat, start_dt = _load_center_and_start(config_file)
    if center_lon is None:
        center_lon = config_lon
    if center_lat is None:
        center_lat = config_lat
    center_lon = float(center_lon)
    center_lat = float(center_lat)
    radius = float(ds.attrs.get("planet_radius", EARTH_RADIUS))

    x2_centered = x2 - 0.5 * (x2[0] + x2[-1])
    x3_centered = x3 - 0.5 * (x3[0] + x3[-1])
    lat = center_lat + np.degrees(x3_centered / radius)
    lon = center_lon + np.degrees(x2_centered / (radius * np.cos(np.radians(center_lat))))
    sim_seconds = float(np.asarray(ds["time"].values).reshape(-1)[0])
    return lon, lat, start_dt + timedelta(seconds=sim_seconds)


def _oriented_topography(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topo = np.asarray(payload["topography"], dtype=float)
    lon = np.asarray(payload["lon"], dtype=float)
    lat = np.asarray(payload["lat"], dtype=float)
    if bool(payload.get("row0_is_north")):
        topo = np.flip(topo, axis=0)
        lat = lat[::-1].copy()
    return topo, lon, lat


def _select_topography_file(topo_dir: str, location: str, target_shape: tuple[int, int]) -> Path:
    candidates = sorted(Path(topo_dir).glob(f"{location}_topo_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No topography files found in {topo_dir} for {location}")

    def score(path: Path) -> tuple[int, int]:
        payload = load_topography(str(path))
        topo = np.asarray(payload["topography"])
        shape = topo.shape
        return (abs(shape[0] - target_shape[0]) + abs(shape[1] - target_shape[1]), len(path.stem))

    return min(candidates, key=score)


def _topography_on_runtime_grid(
    topo_file: str,
    runtime_lon: np.ndarray,
    runtime_lat: np.ndarray,
) -> np.ndarray:
    payload = load_topography(topo_file)
    topo, topo_lon, topo_lat = _oriented_topography(payload)
    interp = RegularGridInterpolator(
        (topo_lat, topo_lon),
        topo,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    lat_grid, lon_grid = np.meshgrid(runtime_lat, runtime_lon, indexing="ij")
    points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
    return interp(points).reshape(len(runtime_lat), len(runtime_lon))


def _nearest_indices(x1: np.ndarray, heights_m: tuple[float, ...]) -> list[int]:
    return [int(np.argmin(np.abs(x1 - height))) for height in heights_m]


def _quiver_step(nx: int, ny: int, target: int = 18) -> tuple[slice, slice]:
    step_x = max(1, nx // target)
    step_y = max(1, ny // target)
    return slice(None, None, step_x), slice(None, None, step_y)


def _quiver_scale(max_speed: float) -> float:
    return max(1.0, max_speed * 14.0)


def _quiver_reference_speed(max_speed: float) -> float:
    if max_speed <= 5.0:
        return 2.0
    if max_speed <= 10.0:
        return 5.0
    if max_speed <= 20.0:
        return 10.0
    return 20.0


def plot_wind_altitude_panels(
    input_file: str,
    config_file: str,
    topo_dir: str,
    location: str,
    *,
    output_file: str | None = None,
    contour_levels: int = 21,
) -> None:
    ds = xr.open_dataset(input_file)
    lon, lat, valid_dt = _meters_to_lonlat(ds, config_file)
    x1 = np.asarray(ds["x1"].values, dtype=float)
    vel1 = np.asarray(ds["vel1"].isel(time=0).values, dtype=float)
    vel2 = np.asarray(ds["vel2"].isel(time=0).values, dtype=float)
    vel3 = np.asarray(ds["vel3"].isel(time=0).values, dtype=float)

    topo_file = _select_topography_file(topo_dir, location, (len(lat), len(lon)))
    topo = _topography_on_runtime_grid(str(topo_file), lon, lat) / 1000.0

    indices = _nearest_indices(x1, TARGET_HEIGHTS_M)
    selected_heights_km = [x1[idx] / 1000.0 for idx in indices]
    w_slices = [vel1[idx] for idx in indices]
    u_slices = [vel2[idx] for idx in indices]
    v_slices = [vel3[idx] for idx in indices]
    speed_slices = [np.hypot(u, v) for u, v in zip(u_slices, v_slices)]

    max_speed = max(float(np.nanmax(speed)) for speed in speed_slices)
    quiver_scale = _quiver_scale(max_speed)
    quiver_ref = _quiver_reference_speed(max_speed)

    X, Y = np.meshgrid(lon, lat)
    sx, sy = _quiver_step(len(lon), len(lat), target=18)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    contour = None
    quiver_artist = None
    for ax, idx, z_km, w, u, v in zip(axes.flat, indices, selected_heights_km, w_slices, u_slices, v_slices):
        contour = ax.contourf(
            X,
            Y,
            w,
            levels=np.linspace(-VERTICAL_VELOCITY_LIMIT, VERTICAL_VELOCITY_LIMIT, contour_levels),
            cmap="RdBu_r",
            extend="both",
            zorder=1,
        )
        topo_lines = ax.contour(
            X,
            Y,
            topo,
            levels=8,
            colors="#555555",
            linewidths=0.8,
            alpha=0.85,
            zorder=2,
        )
        ax.clabel(topo_lines, inline=True, fontsize=7, fmt="%.1f km")
        quiver_artist = ax.quiver(
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
            alpha=0.9,
            zorder=3,
        )
        ax.set_title(f"UTC {valid_dt.strftime('%Y-%m-%d %H:%M:%S')} | z = {z_km:.1f} km")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    cbar = plt.colorbar(contour, ax=axes.ravel().tolist(), label="Vertical velocity (m s$^{-1}$)", shrink=0.8)
    if quiver_artist is not None:
        fig.canvas.draw()
        cbar_box = cbar.ax.get_position()
        key = axes[0, 0].quiverkey(
            quiver_artist,
            X=0.5 * (cbar_box.x0 + cbar_box.x1),
            Y=max(0.02, cbar_box.y0 - 0.060),
            U=quiver_ref,
            label=f"{quiver_ref:.0f} m s$^{{-1}}$",
            labelpos="S",
            coordinates="figure",
        )
        fig.add_artist(key)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    plt.close(fig)
    ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a 2x3 grid of horizontal wind vectors with topography overlay and "
            "vertical velocity background at fixed altitude levels."
        )
    )
    parser.add_argument("input_file", help="FRIGATE out1 NetCDF file.")
    parser.add_argument("--config", required=True, help="FRIGATE YAML config for lon/lat and UTC time.")
    parser.add_argument("--topo-dir", required=True, help="Directory containing topography .pt files.")
    parser.add_argument("--location", required=True, help="Topography location prefix, e.g. pte1b.")
    parser.add_argument("-o", "--output", help="Output plot file (PNG).")
    args = parser.parse_args()

    plot_wind_altitude_panels(
        args.input_file,
        args.config,
        args.topo_dir,
        args.location,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
