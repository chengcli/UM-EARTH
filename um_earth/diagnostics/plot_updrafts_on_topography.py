#!/usr/bin/env python3
"""Plot updraft locations over topography in a multi-page PDF."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

from um_earth.diagnostics.topo_utils import load_topography


PANELS_PER_PAGE = 6


@dataclass(frozen=True)
class UpdraftPoint:
    """Aggregated updraft data for one horizontal location at one time."""

    x: float
    y: float
    max_mean_vel1: float
    segment_count: int
    min_altitude: float
    max_altitude: float


def load_domain_extent(input_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load x2 and x3 coordinates from the first out1 file in a directory."""
    out1_files = sorted(input_dir.glob("*.out1.*.nc"))
    if not out1_files:
        raise FileNotFoundError(f"No out1 NetCDF files found in {input_dir}")

    with xr.open_dataset(out1_files[0]) as dataset:
        x2 = dataset["x2"].values.astype(float)
        x3 = dataset["x3"].values.astype(float)
    return x2, x3


def load_updraft_points(csv_path: Path) -> dict[str, list[UpdraftPoint]]:
    """Load and aggregate updraft rows by time snapshot and x-y location."""
    grouped: dict[str, dict[tuple[float, float], dict[str, float]]] = defaultdict(dict)

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            time_key = row["time_utc"]
            x = float(row["x"])
            y = float(row["y"])
            key = (x, y)
            current = grouped[time_key].get(key)
            mean_vel1 = float(row["mean_vel1"])
            min_altitude = float(row["min_altitude"])
            max_altitude = float(row["max_altitude"])
            if current is None:
                grouped[time_key][key] = {
                    "max_mean_vel1": mean_vel1,
                    "segment_count": 1.0,
                    "min_altitude": min_altitude,
                    "max_altitude": max_altitude,
                }
            else:
                current["max_mean_vel1"] = max(current["max_mean_vel1"], mean_vel1)
                current["segment_count"] += 1.0
                current["min_altitude"] = min(current["min_altitude"], min_altitude)
                current["max_altitude"] = max(current["max_altitude"], max_altitude)

    result: dict[str, list[UpdraftPoint]] = {}
    for time_key, point_map in grouped.items():
        points = [
            UpdraftPoint(
                x=x,
                y=y,
                max_mean_vel1=values["max_mean_vel1"],
                segment_count=int(values["segment_count"]),
                min_altitude=values["min_altitude"],
                max_altitude=values["max_altitude"],
            )
            for (x, y), values in sorted(point_map.items(), key=lambda item: (item[0][1], item[0][0]))
        ]
        result[time_key] = points
    return dict(sorted(result.items(), key=lambda item: item[0]))


def _marker_sizes(points: list[UpdraftPoint]) -> np.ndarray:
    counts = np.array([point.segment_count for point in points], dtype=float)
    return 18.0 + 12.0 * np.sqrt(counts)


def plot_updrafts_pdf(
    csv_path: Path,
    topo_file: Path,
    input_dir: Path,
    output_pdf: Path,
) -> int:
    """Render a multi-page PDF with six time snapshots per page."""
    x2, x3 = load_domain_extent(input_dir)
    topo_data = load_topography(str(topo_file))
    topo_array = topo_data["topography"]
    updrafts_by_time = load_updraft_points(csv_path)
    if not updrafts_by_time:
        raise ValueError(f"No updraft rows found in {csv_path}")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    times = list(updrafts_by_time.keys())
    extent = [float(x2.min()), float(x2.max()), float(x3.min()), float(x3.max())]
    topo_min = float(np.min(topo_array))
    topo_max = float(np.max(topo_array))
    max_vel = max(
        point.max_mean_vel1
        for points in updrafts_by_time.values()
        for point in points
    )

    with PdfPages(output_pdf) as pdf:
        for page_start in range(0, len(times), PANELS_PER_PAGE):
            fig, axes = plt.subplots(3, 2, figsize=(11, 14), constrained_layout=True)
            axes_flat = axes.flatten()
            page_times = times[page_start : page_start + PANELS_PER_PAGE]

            for axis, time_key in zip(axes_flat, page_times):
                points = updrafts_by_time[time_key]
                axis.imshow(
                    topo_array,
                    extent=extent,
                    origin="lower",
                    cmap="terrain",
                    vmin=topo_min,
                    vmax=topo_max,
                    aspect="auto",
                )
                scatter = axis.scatter(
                    [point.x for point in points],
                    [point.y for point in points],
                    c=[point.max_mean_vel1 for point in points],
                    s=_marker_sizes(points),
                    cmap="plasma",
                    vmin=1.0,
                    vmax=max_vel,
                    alpha=0.85,
                    edgecolors="black",
                    linewidths=0.2,
                )
                axis.set_title(f"{time_key}\n{len(points)} updraft locations", fontsize=10)
                axis.set_xlabel("X (m)")
                axis.set_ylabel("Y (m)")
                axis.set_xlim(extent[0], extent[1])
                axis.set_ylim(extent[2], extent[3])
                axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.4)

            for axis in axes_flat[len(page_times):]:
                axis.axis("off")

            fig.suptitle("Updraft Locations Over Topography", fontsize=16)
            colorbar = fig.colorbar(
                scatter,
                ax=axes_flat.tolist(),
                location="right",
                shrink=0.9,
                pad=0.01,
            )
            colorbar.set_label("Max mean vel1 at location (m/s)")
            pdf.savefig(fig)
            plt.close(fig)

    return len(times)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot updraft locations over topography into a multi-page PDF"
    )
    parser.add_argument("csv_path", help="CSV file created by extract-updrafts")
    parser.add_argument("--topo-file", required=True, help="Topography .pt file")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the source out1 NetCDF files",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output PDF file (default: <csv_path stem>_topography.pdf)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    csv_path = Path(args.csv_path)
    topo_file = Path(args.topo_file)
    input_dir = Path(args.input_dir)
    output_pdf = (
        Path(args.output)
        if args.output
        else csv_path.with_name(f"{csv_path.stem}_topography.pdf")
    )
    snapshot_count = plot_updrafts_pdf(csv_path, topo_file, input_dir, output_pdf)
    print(f"Wrote {snapshot_count} snapshots to {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
