import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from um_earth.diagnostics.extract_updrafts import (
    UpdraftSegment,
    extract_directory,
    infer_default_output_path,
    infer_start_datetime,
    iter_updraft_segments,
)


def _write_out1_file(path: Path, *, time_s: float, vel1: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> None:
    dataset = xr.Dataset(
        {
            "vel1": (["time", "x1", "x3", "x2"], vel1[np.newaxis, :, :, :]),
        },
        coords={
            "time": np.array([time_s], dtype=np.float32),
            "x1": x1.astype(np.float32),
            "x2": x2.astype(np.float32),
            "x3": x3.astype(np.float32),
        },
    )
    dataset.to_netcdf(path)


class TestIterUpdraftSegments(unittest.TestCase):
    def test_splits_contiguous_true_runs(self) -> None:
        mask = np.array([False, True, True, False, True, False, True, True, True])
        segments = list(iter_updraft_segments(mask))
        self.assertEqual(
            segments,
            [
                UpdraftSegment(start_index=1, end_index=2),
                UpdraftSegment(start_index=4, end_index=4),
                UpdraftSegment(start_index=6, end_index=8),
            ],
        )


class TestExtractDirectory(unittest.TestCase):
    def test_extracts_expected_rows_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "ws-site1-2026-02-09"
            input_dir.mkdir()
            x1 = np.array([100.0, 300.0, 500.0, 700.0], dtype=np.float32)
            x2 = np.array([1000.0, 3000.0], dtype=np.float32)
            x3 = np.array([2000.0, 4000.0], dtype=np.float32)

            vel1_file0 = np.zeros((4, 2, 2), dtype=np.float32)
            vel1_file0[1:3, 0, 0] = np.array([1.2, 1.6], dtype=np.float32)
            vel1_file0[0, 1, 1] = 1.1
            vel1_file0[2, 1, 1] = 1.5
            _write_out1_file(
                input_dir / "ws-site1-2026-02-09.out1.00000.nc",
                time_s=0.0,
                vel1=vel1_file0,
                x1=x1,
                x2=x2,
                x3=x3,
            )

            vel1_file1 = np.zeros((4, 2, 2), dtype=np.float32)
            vel1_file1[:, 0, 1] = np.array([0.5, 1.3, 1.5, 0.7], dtype=np.float32)
            _write_out1_file(
                input_dir / "ws-site1-2026-02-09.out1.00001.nc",
                time_s=3600.0,
                vel1=vel1_file1,
                x1=x1,
                x2=x2,
                x3=x3,
            )

            xr.Dataset({"temp": (["time"], np.array([280.0], dtype=np.float32))}).to_netcdf(
                input_dir / "ws-site1-2026-02-09.out2.00000.nc"
            )

            output_csv = infer_default_output_path(input_dir)
            row_count = extract_directory(input_dir, output_csv)

            self.assertEqual(row_count, 4)
            with output_csv.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 4)

            self.assertEqual(rows[0]["time_s"], "0.0")
            self.assertEqual(rows[0]["time_utc"], "2026-02-09T00:00:00Z")
            self.assertEqual(rows[0]["x"], "1000.0")
            self.assertEqual(rows[0]["y"], "2000.0")
            self.assertEqual(rows[0]["min_altitude"], "300.0")
            self.assertEqual(rows[0]["max_altitude"], "500.0")
            self.assertAlmostEqual(float(rows[0]["mean_vel1"]), 1.4, places=6)
            self.assertAlmostEqual(float(rows[0]["std_vel1"]), 0.2, places=6)

            self.assertEqual(rows[1]["x"], "3000.0")
            self.assertEqual(rows[1]["y"], "4000.0")
            self.assertEqual(rows[1]["min_altitude"], "100.0")
            self.assertEqual(rows[1]["max_altitude"], "100.0")

            self.assertEqual(rows[2]["x"], "3000.0")
            self.assertEqual(rows[2]["y"], "4000.0")
            self.assertEqual(rows[2]["min_altitude"], "500.0")
            self.assertEqual(rows[2]["max_altitude"], "500.0")

            self.assertEqual(rows[3]["time_s"], "3600.0")
            self.assertEqual(rows[3]["time_utc"], "2026-02-09T01:00:00Z")
            self.assertEqual(rows[3]["x"], "3000.0")
            self.assertEqual(rows[3]["y"], "2000.0")
            self.assertEqual(rows[3]["min_altitude"], "300.0")
            self.assertEqual(rows[3]["max_altitude"], "500.0")
            self.assertAlmostEqual(float(rows[3]["mean_vel1"]), 1.4, places=6)

    def test_infer_start_datetime_from_directory_name(self) -> None:
        timestamp = infer_start_datetime(Path("/tmp/ws-site1-2026-02-09"))
        self.assertEqual(timestamp, datetime(2026, 2, 9, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
