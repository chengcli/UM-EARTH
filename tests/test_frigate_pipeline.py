from __future__ import annotations

import json
from pathlib import Path

import yaml

from um_earth.frigate_pipeline import (
    build_prepared_domain,
    expand_bounds,
    run_frigate_prepare,
    write_config,
    write_region_digest,
)
from um_earth.regions import load_region_from_kml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def locate_sacramento_kml() -> Path:
    candidates = [
        WORKSPACE_ROOT / "sacramento_valley.kml",
        WORKSPACE_ROOT / "data" / "2025.FRIGATE" / "sacramento_valley.kml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate sacramento_valley.kml in workspace fixtures")


def test_expand_bounds_enforces_minimum_extent():
    bounds = (-121.2, 38.0, -120.9, 38.2)
    lon_min, lat_min, lon_max, lat_max = expand_bounds(bounds, min_size_degrees=1.5)

    assert round(lat_max - lat_min, 6) == 1.5
    assert round(lon_max - lon_min, 6) == 1.5


def test_write_digest_and_config(tmp_path):
    region = load_region_from_kml(locate_sacramento_kml())
    prepared = build_prepared_domain(region)

    digest_path = tmp_path / "digest.json"
    payload = write_region_digest(digest_path, region, prepared, date="2025-02-01")
    loaded = json.loads(digest_path.read_text(encoding="utf-8"))
    assert loaded["region_id"] == region.region_id
    assert loaded["padded_extent_degrees"]["latitude"] >= 1.5
    assert payload["simulation_grid"]["nx2"] == prepared.nx2

    config_path = tmp_path / "config.yaml"
    rendered = write_config(config_path, region, prepared, date="2025-02-01")
    parsed = yaml.safe_load(rendered)
    assert parsed["integration"]["start-date"].isoformat() == "2025-02-01"
    assert parsed["geometry"]["cells"]["nx2"] == prepared.nx2


def test_run_frigate_prepare_creates_artifacts(tmp_path, monkeypatch):
    kml_path = locate_sacramento_kml()

    def fake_topography_download(region, raw_out_dir, *, skip_download=False):
        location_dir = raw_out_dir / region.region_id
        location_dir.mkdir(parents=True, exist_ok=True)
        (location_dir / f"{region.region_id}_merged_10m.tif").write_text("fake", encoding="utf-8")

    def fake_build_resolution_products(**kwargs):
        out_dir = kwargs["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for label in ("2p4km", "1p2km", "0p6km", "0p3km"):
            path = out_dir / f"{kwargs['region_id']}_{label}.pt"
            path.write_text(label, encoding="utf-8")
            outputs[label] = path
        return outputs

    def fake_run_initial_condition_pipeline(region, config_path, era5_out_dir, *, timeout, times):
        output_dir = era5_out_dir / "38.00N_39.50N_122.00W_120.50W"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "era5_hourly_densities_20250201.nc").write_text("density", encoding="utf-8")
        (output_dir / "era5_hourly_dynamics_20250201.nc").write_text("dynamics", encoding="utf-8")

    def fake_generate_verification_plots(plots_dir, *args, **kwargs):
        plots_dir.mkdir(parents=True, exist_ok=True)
        for name in ("kml_domain_overview.png", "era5_fetch_summary.png", "topography_resolutions.png"):
            (plots_dir / name).write_text(name, encoding="utf-8")

    monkeypatch.setattr("um_earth.frigate_pipeline.run_topography_download", fake_topography_download)
    monkeypatch.setattr("um_earth.frigate_pipeline.build_resolution_products", fake_build_resolution_products)
    monkeypatch.setattr("um_earth.frigate_pipeline.run_initial_condition_pipeline", fake_run_initial_condition_pipeline)
    monkeypatch.setattr("um_earth.frigate_pipeline.generate_verification_plots", fake_generate_verification_plots)

    run_dir = run_frigate_prepare(
        region_kml=str(kml_path),
        date="2025-02-01",
        workspace_root=tmp_path,
    )

    manifest = yaml.safe_load((run_dir / "run_manifest.yaml").read_text(encoding="utf-8"))
    assert (run_dir / "region_digest.json").exists()
    assert (run_dir / "run_manifest.yaml").exists()
    assert (run_dir / "plots" / "topography_resolutions.png").exists()
    assert (run_dir / "topography" / "products").exists()
    assert manifest["simulation_input_path"].endswith("sacramento_valley.yaml")
