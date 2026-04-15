#!/usr/bin/env python3
"""Plot the staged topography products in a 2x2 comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from um_earth.diagnostics.topo_utils import load_topography
from um_earth.regions import load_region_from_kml


TOPO_LABELS = ("2p4km", "1p2km", "0p6km", "0p3km")


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


def plot_topography_stages(
    products_dir: Path,
    region: str,
    output_path: Path,
    kml_file: Path | None = None,
) -> None:
    topo_files = discover_topography_files(products_dir, region)
    topo_payloads = [load_topography(str(path)) for path in topo_files]
    topo_maps = [payload["topography"].astype(float) for payload in topo_payloads]
    bbox = kml_bounding_box(kml_file) if kml_file is not None else None

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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    plot_topography_stages(
        products_dir=Path(args.products_dir),
        region=args.region,
        output_path=Path(args.output),
        kml_file=Path(args.kml_file) if args.kml_file else None,
    )
    print(f"Wrote topography stage plot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
