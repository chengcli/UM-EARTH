#!/usr/bin/env python3
"""Legacy wrapper for `um-earth config generate`."""

from __future__ import annotations

import argparse
from pathlib import Path

from um_earth.configuration import ConfigOptions, default_output_path, render_config
from um_earth.regions import load_region, load_regions_from_csv


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate location-specific YAML configuration files")
    parser.add_argument("location_id", nargs="?")
    parser.add_argument("--region-kml")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--nx1", type=int)
    parser.add_argument("--nx2", type=int)
    parser.add_argument("--nx3", type=int)
    parser.add_argument("--nghost", type=int, default=3)
    parser.add_argument("--x1-max", type=float, default=20000.0)
    parser.add_argument("--x2-extent", type=float)
    parser.add_argument("--x3-extent", type=float)
    parser.add_argument("--tlim", type=int, default=86400)
    parser.add_argument("--output")
    parser.add_argument("--locations-file", default="locations.csv")
    parser.add_argument("--template-file", default="config_template.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    locations_file = Path(args.locations_file)
    if not locations_file.is_absolute():
        locations_file = SCRIPT_DIR / locations_file

    if args.list:
        for region_id, region in sorted(load_regions_from_csv(locations_file).items()):
            print(f"{region_id}: {region.name}")
        return 0

    missing = [name for name in ("start_date", "end_date", "nx1", "nx2", "nx3") if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments: " + ", ".join("--" + item.replace("_", "-") for item in missing))

    template_file = Path(args.template_file)
    if not template_file.is_absolute():
        template_file = SCRIPT_DIR / template_file

    region = load_region(
        region_kml=args.region_kml,
        location_id=args.location_id,
        locations_file=locations_file,
    )
    rendered = render_config(
        template_file.read_text(encoding="utf-8"),
        region,
        ConfigOptions(
            start_date=args.start_date,
            end_date=args.end_date,
            nx1=args.nx1,
            nx2=args.nx2,
            nx3=args.nx3,
            nghost=args.nghost,
            x1_max=args.x1_max,
            x2_extent=args.x2_extent,
            x3_extent=args.x3_extent,
            tlim=args.tlim,
        ),
    )
    output_file = Path(args.output) if args.output else default_output_path(Path.cwd(), region)
    output_file.write_text(rendered, encoding="utf-8")
    print(f"\033[92m[OK]\033[0m Configuration file generated: {output_file}")
    print(f"  Location: {region.name}")
    print(f"  Time window: {args.start_date} to {args.end_date}")
    print(f"  Grid: {args.nx1} x {args.nx2} x {args.nx3} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
