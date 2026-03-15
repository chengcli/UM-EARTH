from __future__ import annotations

from pathlib import Path

import yaml

from um_earth.cli import main
from um_earth.configuration import ConfigOptions, render_config
from um_earth.regions import load_region_from_kml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SACRAMENTO_KML = WORKSPACE_ROOT / "data" / "2025.FRIGATE" / "sacramento_valley.kml"


def test_load_region_from_kml():
    region = load_region_from_kml(SACRAMENTO_KML)
    lon_min, lat_min, lon_max, lat_max = region.bounds

    assert region.region_id == "sacramento_valley"
    assert lon_min < lon_max
    assert lat_min < lat_max
    assert 38.49 < lat_min < 38.50
    assert 38.50 < lat_max < 38.51


def test_render_config_from_kml():
    region = load_region_from_kml(SACRAMENTO_KML)
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
    assert parsed["geometry"]["cells"]["nx2"] == 48


def test_cli_config_generate_from_kml(tmp_path, monkeypatch):
    output = tmp_path / "sacramento.yaml"
    monkeypatch.chdir(tmp_path)

    rc = main(
        [
            "config",
            "generate",
            "--region-kml",
            str(SACRAMENTO_KML),
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
    assert config["integration"]["start-date"].isoformat() == "2025-02-01"
