#!/usr/bin/env python3
"""Extract vertically contiguous updraft segments from UM-EARTH output."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr


DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})$")


@dataclass(frozen=True)
class UpdraftSegment:
    """A single contiguous vertical updraft segment for one x-y column."""

    start_index: int
    end_index: int


def iter_updraft_segments(mask: np.ndarray) -> Iterable[UpdraftSegment]:
    """Yield contiguous True runs from a 1D boolean mask."""
    if mask.ndim != 1:
        raise ValueError("Updraft mask must be 1-dimensional")

    start: int | None = None
    for idx, is_updraft in enumerate(mask):
        if is_updraft and start is None:
            start = idx
        elif not is_updraft and start is not None:
            yield UpdraftSegment(start_index=start, end_index=idx - 1)
            start = None

    if start is not None:
        yield UpdraftSegment(start_index=start, end_index=mask.shape[0] - 1)


def infer_start_datetime(input_dir: Path) -> datetime:
    """Infer the simulation start date from the input directory name."""
    match = DATE_PATTERN.search(input_dir.name)
    if match is None:
        raise ValueError(
            f"Could not infer simulation date from directory name: {input_dir}"
        )
    return datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)


def infer_default_output_path(input_dir: Path) -> Path:
    """Return the default CSV output path for a directory."""
    return input_dir / "updrafts.csv"


def _format_time_utc(start_time: datetime, time_s: float) -> str:
    timestamp = start_time + timedelta(seconds=float(time_s))
    return timestamp.isoformat().replace("+00:00", "Z")


def find_out1_files(input_dir: Path) -> list[Path]:
    """Return all sorted out1 files in a directory."""
    return sorted(input_dir.glob("*.out1.*.nc"))


def extract_updraft_rows(input_file: Path, *, threshold: float, start_time: datetime) -> list[dict[str, float | str]]:
    """Extract CSV rows for all updraft segments in a single NetCDF file."""
    rows: list[dict[str, float | str]] = []
    with xr.open_dataset(input_file) as dataset:
        if "vel1" not in dataset:
            return rows

        vel1 = dataset["vel1"].isel(time=0).values
        x1 = dataset["x1"].values
        x2 = dataset["x2"].values
        x3 = dataset["x3"].values
        time_s = float(dataset["time"].values[0])
        time_utc = _format_time_utc(start_time, time_s)

        for y_index, y_value in enumerate(x3):
            for x_index, x_value in enumerate(x2):
                column = vel1[:, y_index, x_index]
                for segment in iter_updraft_segments(column > threshold):
                    values = column[segment.start_index : segment.end_index + 1]
                    rows.append(
                        {
                            "time_s": time_s,
                            "time_utc": time_utc,
                            "x": float(x_value),
                            "y": float(y_value),
                            "min_altitude": float(x1[segment.start_index]),
                            "max_altitude": float(x1[segment.end_index]),
                            "mean_vel1": float(np.mean(values)),
                            "std_vel1": float(np.std(values)),
                        }
                    )

    return rows


def extract_directory(input_dir: Path, output_csv: Path, *, threshold: float = 1.0) -> int:
    """Process a directory of out1 files and write one CSV row per updraft."""
    input_dir = input_dir.resolve()
    output_csv = output_csv.resolve()
    start_time = infer_start_datetime(input_dir)
    out1_files = find_out1_files(input_dir)
    if not out1_files:
        raise FileNotFoundError(f"No out1 NetCDF files found in {input_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_s",
        "time_utc",
        "x",
        "y",
        "min_altitude",
        "max_altitude",
        "mean_vel1",
        "std_vel1",
    ]

    row_count = 0
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for input_file in out1_files:
            for row in extract_updraft_rows(input_file, threshold=threshold, start_time=start_time):
                writer.writerow(row)
                row_count += 1

    return row_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract vertically contiguous updrafts from out1 NetCDF files"
    )
    parser.add_argument("input_dir", help="Directory containing out1 NetCDF files")
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV file (default: <input_dir>/updrafts.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Updraft threshold for vel1 in m/s (default: 1.0)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output) if args.output else infer_default_output_path(input_dir)
    row_count = extract_directory(input_dir, output_csv, threshold=args.threshold)
    print(f"Wrote {row_count} updraft rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
