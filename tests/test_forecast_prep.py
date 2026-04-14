from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import yaml

import prepare_initial_condition as prepare_module
from um_earth.cli import main as cli_main
from um_earth.ecmwf_api.ingest_forecast_data import ingest_forecast_cycle, select_forecast_cycle
from um_earth.frigate_pipeline import run_frigate_prepare


def _write_forecast_pair(root: Path, date_str: str = "20260412", cycle: str = "00") -> None:
    for kind in ("pl", "sfc"):
        (root / f"ifs_{date_str}_{cycle}_{kind}.grib2").write_text(kind, encoding="utf-8")


def test_select_forecast_cycle_requires_pl_and_sfc(tmp_path):
    _write_forecast_pair(tmp_path, cycle="06")
    date_str, cycle, files = select_forecast_cycle(tmp_path, date_str="20260412", cycle="06")

    assert date_str == "20260412"
    assert cycle == "06"
    assert set(files) == {"pl", "sfc"}


def test_ingest_forecast_cycle_writes_expected_netcdf_outputs(tmp_path, monkeypatch):
    _write_forecast_pair(tmp_path)

    latitudes = np.array([34.0, 33.5], dtype=np.float32)
    longitudes = np.array([-107.0, -106.5], dtype=np.float32)
    levels = np.array([1000, 900], dtype=np.int32)
    steps = np.array([np.timedelta64(hour, "h") for hour in (0, 6, 12, 18)])
    valid_time = np.array(
        [np.datetime64("2026-04-12T00:00") + step for step in steps],
        dtype="datetime64[ns]",
    )
    shape = (4, 2, 2, 2)

    def fake_open(path: Path):
        if path.name.endswith("_pl.grib2"):
            data_vars = {
                name: (("step", "isobaricInhPa", "latitude", "longitude"), np.ones(shape, dtype=np.float32))
                for name in ("t", "u", "v", "w", "q")
            }
            return xr.Dataset(
                data_vars=data_vars,
                coords={
                    "step": steps,
                    "valid_time": ("step", valid_time),
                    "isobaricInhPa": levels,
                    "latitude": latitudes,
                    "longitude": longitudes,
                },
            )
        return xr.Dataset(
            data_vars={"z": (("latitude", "longitude"), np.full((2, 2), 98.0665, dtype=np.float32))},
            coords={"latitude": latitudes, "longitude": longitudes},
        )

    monkeypatch.setattr("um_earth.ecmwf_api.ingest_forecast_data._open_grib_dataset", fake_open)

    result = ingest_forecast_cycle(tmp_path, tmp_path / "out", date_str="20260412", cycle="00")

    dynamics_path = Path(result["dynamics"])
    densities_path = Path(result["densities"])
    topo_path = dynamics_path.parent / "forecast_topography_20260412_00.nc"

    assert dynamics_path.name == "forecast_hourly_dynamics_20260412_00.nc"
    assert densities_path.name == "forecast_hourly_densities_20260412_00.nc"
    assert topo_path.exists()

    dynamics = xr.open_dataset(dynamics_path)
    assert tuple(dynamics["t"].dims) == ("time", "pressure_level", "latitude", "longitude")
    assert len(dynamics["time"]) == 4
    dynamics.close()

    topo = xr.open_dataset(topo_path)
    np.testing.assert_allclose(topo["topography"].values, np.full((2, 2), 10.0, dtype=np.float32))
    topo.close()


def test_infer_forecast_cycle_from_written_products(tmp_path):
    out_dir = tmp_path / "forecast"
    out_dir.mkdir()
    (out_dir / "forecast_hourly_dynamics_20260412_06.nc").write_text("dyn", encoding="utf-8")
    (out_dir / "forecast_hourly_densities_20260412_06.nc").write_text("den", encoding="utf-8")

    assert prepare_module.infer_forecast_cycle(out_dir, "20260412") == "06"


def test_cli_init_prepare_forwards_forecast_arguments(monkeypatch, tmp_path):
    captured: list[str] = []

    def fake_run_python(script, args):
        captured.extend(args)
        return 0

    monkeypatch.setattr("um_earth.cli._run_python", fake_run_python)

    rc = cli_main(
        [
            "init",
            "prepare",
            "--region-kml",
            str(tmp_path / "dummy.kml"),
            "--config",
            str(tmp_path / "cfg.yaml"),
            "--output-base",
            str(tmp_path / "out"),
            "--data-source",
            "forecast",
            "--forecast-input-dir",
            str(tmp_path / "forecast"),
            "--forecast-cycle",
            "00",
            "--forecast-leads",
            "0",
            "6",
            "12",
            "18",
        ]
    )

    assert rc == 0
    assert "--data-source" in captured
    assert "forecast" in captured
    assert "--forecast-input-dir" in captured
    assert "--forecast-cycle" in captured


def test_run_frigate_prepare_records_forecast_manifest(tmp_path, monkeypatch):
    kml_path = tmp_path / "pte1b.kml"
    kml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>pte1b</name><Polygon><outerBoundaryIs><LinearRing><coordinates>
-106.8,33.2,0 -106.2,33.2,0 -106.2,33.7,0 -106.8,33.7,0 -106.8,33.2,0
</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>""",
        encoding="utf-8",
    )

    def fake_topography_download(region, raw_out_dir, *, skip_download=False):
        location_dir = raw_out_dir / region.region_id
        location_dir.mkdir(parents=True, exist_ok=True)
        (location_dir / f"{region.region_id}_merged_10m.tif").write_text("fake", encoding="utf-8")

    def fake_build_resolution_products(**kwargs):
        out_dir = kwargs["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for label in ("2p4km", "1p2km", "0p6km", "0p3km"):
            path = out_dir / f"{kwargs['region_id']}_topo_{label}.pt"
            path.write_text(label, encoding="utf-8")
            outputs[label] = path
        return outputs

    def fake_run_initial_condition_pipeline(region, config_path, era5_out_dir, **kwargs):
        output_dir = era5_out_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "forecast_hourly_dynamics_20260412_00.nc").write_text("dynamics", encoding="utf-8")
        (output_dir / "forecast_hourly_densities_20260412_00.nc").write_text("densities", encoding="utf-8")

    def fake_generate_verification_plots(plots_dir, *args, **kwargs):
        plots_dir.mkdir(parents=True, exist_ok=True)
        (plots_dir / "topography_resolutions.png").write_text("plot", encoding="utf-8")

    monkeypatch.setattr("um_earth.frigate_pipeline.run_topography_download", fake_topography_download)
    monkeypatch.setattr("um_earth.frigate_pipeline.build_resolution_products", fake_build_resolution_products)
    monkeypatch.setattr("um_earth.frigate_pipeline.run_initial_condition_pipeline", fake_run_initial_condition_pipeline)
    monkeypatch.setattr("um_earth.frigate_pipeline.generate_verification_plots", fake_generate_verification_plots)

    run_dir = run_frigate_prepare(
        region_kml=str(kml_path),
        date="2026-04-12",
        workspace_root=tmp_path,
        data_source="forecast",
        forecast_input_dir="/home/chengcli/data/2025.FRIGATE/ECWMF_prediction_data/20260412",
        forecast_cycle="00",
        forecast_leads=[0, 6, 12, 18],
    )

    manifest = yaml.safe_load((run_dir / "run_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["data_source"] == "forecast"
    assert manifest["forecast_cycle"] == "00"
    assert manifest["forecast_input_dir"].endswith("20260412")
