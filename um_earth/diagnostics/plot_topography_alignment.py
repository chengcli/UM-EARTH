#!/usr/bin/env python3
"""Plot KML/domain alignment against the stored topography geographic metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from um_earth.diagnostics.topo_utils import load_topography
from um_earth.regions import load_region_from_kml


def plot_alignment(kml_file: Path, topo_file: Path, output_path: Path) -> None:
    region = load_region_from_kml(kml_file)
    topo = load_topography(str(topo_file))

    lon_min, lon_max = map(float, topo["lon_bounds"])
    lat_min, lat_max = map(float, topo["lat_bounds"])
    lon = topo["lon"]
    lat = topo["lat"]

    region_lon = [point[0] for point in region.polygon]
    region_lat = [point[1] for point in region.polygon]

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    ax.plot(
        [*region_lon, region_lon[0]],
        [*region_lat, region_lat[0]],
        color="red",
        linewidth=2.0,
        label="KML polygon",
    )
    ax.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        color="black",
        linewidth=1.8,
        linestyle="--",
        label="Topography bounds",
    )

    ax.scatter(
        [lon[0], lon[-1], lon[-1], lon[0]],
        [lat[0], lat[0], lat[-1], lat[-1]],
        color=["tab:blue", "tab:green", "tab:orange", "tab:purple"],
        s=60,
        zorder=3,
    )
    ax.text(lon[0], lat[0], " row0,col0 ", color="tab:blue", fontsize=9, ha="left", va="bottom")
    ax.text(lon[-1], lat[0], " row0,last ", color="tab:green", fontsize=9, ha="right", va="bottom")
    ax.text(lon[-1], lat[-1], " last,last ", color="tab:orange", fontsize=9, ha="right", va="top")
    ax.text(lon[0], lat[-1], " last,0 ", color="tab:purple", fontsize=9, ha="left", va="top")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="best")

    pad_lon = 0.05 * max(lon_max - lon_min, 1.0e-6)
    pad_lat = 0.05 * max(lat_max - lat_min, 1.0e-6)
    ax.set_xlim(min(min(region_lon), lon_min) - pad_lon, max(max(region_lon), lon_max) + pad_lon)
    ax.set_ylim(min(min(region_lat), lat_min) - pad_lat, max(max(region_lat), lat_max) + pad_lat)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot KML polygon alignment against topography geographic metadata."
    )
    parser.add_argument("--kml-file", required=True, help="KML polygon file.")
    parser.add_argument("--topo-file", required=True, help="Topography product .pt file.")
    parser.add_argument("-o", "--output", required=True, help="Output image path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    plot_alignment(Path(args.kml_file), Path(args.topo_file), Path(args.output))
    print(f"Wrote topography alignment plot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
