from __future__ import annotations

from pathlib import Path

import math
import pytest
import yaml

from um_earth.cli import main
from um_earth.configuration import ConfigOptions, EARTH_ROTATION_RATE, render_config
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


def test_load_region_from_kml():
    region = load_region_from_kml(locate_sacramento_kml())
    lon_min, lat_min, lon_max, lat_max = region.bounds

    assert region.region_id == "sacramento_valley"
    assert lon_min < lon_max
    assert lat_min < lat_max
    assert 38.49 < lat_min < 38.50
    assert 38.50 < lat_max < 38.51


def test_render_config_from_kml():
    region = load_region_from_kml(locate_sacramento_kml())
    template = (PROJECT_ROOT / "config_template.yaml").read_text(encoding="utf-8")

    rendered = render_config(
        template,
        region,
        ConfigOptions(
            start_date="2025-02-01",
            end_date="2025-02-02",
            nx1=32,
            nx2=48,
            nx3=64,
        ),
    )
    parsed = yaml.safe_load(rendered)

    assert parsed["geometry"]["center_latitude"] > 38.49
    assert parsed["integration"]["start-date"].isoformat() == "2025-02-01"
    assert parsed["integration"]["end-date"].isoformat() == "2025-02-02"
    assert parsed["integration"]["cfl"] == 0.9
    assert parsed["geometry"]["cells"]["nx2"] == 48
    expected_omega1 = EARTH_ROTATION_RATE * math.sin(math.radians(parsed["geometry"]["center_latitude"]))
    assert parsed["forcing"]["coriolis"]["type"] == "xyz"
    assert parsed["forcing"]["coriolis"]["omega1"] == pytest.approx(expected_omega1)
    assert parsed["forcing"]["coriolis"]["omega2"] == 0.0
    assert parsed["forcing"]["coriolis"]["omega3"] == 0.0


def test_cli_config_generate_from_kml(tmp_path, monkeypatch):
    output = tmp_path / "sacramento.yaml"
    monkeypatch.chdir(tmp_path)

    rc = main(
        [
            "config",
            "generate",
            "--region-kml",
            str(locate_sacramento_kml()),
            "--start-date",
            "2025-02-01",
            "--end-date",
            "2025-02-02",
            "--nx1",
            "16",
            "--nx2",
            "32",
            "--nx3",
            "48",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["geometry"]["cells"]["nx3"] == 48
    assert config["integration"]["cfl"] == 0.9
    assert config["integration"]["start-date"].isoformat() == "2025-02-01"


def test_cli_pipeline_frigate_prepare(tmp_path, monkeypatch):
    called = {}

    def fake_run_frigate_prepare(**kwargs):
        called.update(kwargs)
        run_dir = tmp_path / "sacramento_valley-2025-02-01"
        run_dir.mkdir()
        return run_dir

    monkeypatch.setattr("um_earth.cli.run_frigate_prepare", fake_run_frigate_prepare)

    rc = main(
        [
            "pipeline",
            "frigate-prepare",
            "--region-kml",
            str(locate_sacramento_kml()),
            "--date",
            "2025-02-01",
            "--workspace-root",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert called["date"] == "2025-02-01"
