"""Unified CLI for the UM-EARTH workflow."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .configuration import ConfigOptions, default_output_path, render_config
from .frigate_pipeline import DEFAULT_WORKSPACE_ROOT, run_frigate_prepare
from .regions import load_region, load_regions_from_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = PROJECT_ROOT / "config_template.yaml"
DEFAULT_LOCATIONS = PROJECT_ROOT / "locations.csv"


def _run_python(script: Path, args: list[str]) -> int:
    cmd = [sys.executable, str(script), *args]
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not pythonpath else str(PROJECT_ROOT) + os.pathsep + pythonpath
    return subprocess.run(cmd, check=False, cwd=PROJECT_ROOT, env=env).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="um-earth", description="UM-EARTH pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    region_p = sub.add_parser("region", help="Inspect region definitions")
    region_sub = region_p.add_subparsers(dest="region_command", required=True)
    inspect_p = region_sub.add_parser("inspect", help="Inspect a KML or CSV-backed region")
    inspect_p.add_argument("--region-kml")
    inspect_p.add_argument("--location-id")
    inspect_p.add_argument("--locations-file", default=str(DEFAULT_LOCATIONS))
    inspect_p.set_defaults(handler=handle_region_inspect)

    config_p = sub.add_parser("config", help="Generate snapy input YAML")
    config_sub = config_p.add_subparsers(dest="config_command", required=True)
    generate_p = config_sub.add_parser("generate", help="Generate a config file")
    add_region_arguments(generate_p)
    add_config_arguments(generate_p)
    generate_p.add_argument("--template-file", default=str(DEFAULT_TEMPLATE))
    generate_p.add_argument("--output")
    generate_p.set_defaults(handler=handle_config_generate)

    topo_p = sub.add_parser("topo", help="Build topography inputs")
    topo_sub = topo_p.add_subparsers(dest="topo_command", required=True)
    topo_build = topo_sub.add_parser("build", help="Run topography preparation")
    add_region_arguments(topo_build)
    topo_build.add_argument("--out")
    topo_build.add_argument("--skip-download", action="store_true")
    topo_build.set_defaults(handler=handle_topo_build)

    init_p = sub.add_parser("init", help="Prepare ECMWF initial conditions")
    init_sub = init_p.add_subparsers(dest="init_command", required=True)
    init_prepare = init_sub.add_parser("prepare", help="Run the ERA5 preparation pipeline")
    add_region_arguments(init_prepare)
    init_prepare.add_argument("--config", required=True)
    init_prepare.add_argument("--output-base", default=".")
    init_prepare.add_argument("--start-from", type=int, default=1)
    init_prepare.add_argument("--stop-after", type=int, default=6)
    init_prepare.add_argument("--timeout", type=int, default=3600)
    init_prepare.add_argument("--nX", type=int, default=1)
    init_prepare.add_argument("--nY", type=int, default=1)
    init_prepare.add_argument("--times", nargs="+")
    init_prepare.set_defaults(handler=handle_init_prepare)

    forecast_p = sub.add_parser("forecast", help="Run the snapy forecast")
    forecast_sub = forecast_p.add_subparsers(dest="forecast_command", required=True)
    forecast_run = forecast_sub.add_parser("run", help="Run the forecast")
    forecast_run.add_argument("--config", required=True)
    forecast_run.add_argument("--input-dir", default="./input")
    forecast_run.add_argument("--output-dir", default="./output")
    forecast_run.set_defaults(handler=handle_forecast_run)

    diag_p = sub.add_parser("diagnostics", help="Generate diagnostics")
    diag_sub = diag_p.add_subparsers(dest="diagnostics_command", required=True)
    diag_plot = diag_sub.add_parser("plot", help="Generate all plots")
    diag_plot.add_argument("input_dir")
    diag_plot.add_argument("--output-dir")
    diag_plot.add_argument("--output-pdf")
    diag_plot.add_argument("--topo-dir")
    diag_plot.add_argument("--location")
    diag_plot.add_argument("--all-times", action="store_true")
    diag_plot.set_defaults(handler=handle_diagnostics_plot)

    pipe_p = sub.add_parser("pipeline", help="Run the full pipeline")
    pipe_sub = pipe_p.add_subparsers(dest="pipeline_command", required=True)
    pipe_run = pipe_sub.add_parser("run", help="Run config, topo, init, and optionally forecast")
    add_region_arguments(pipe_run)
    add_config_arguments(pipe_run)
    pipe_run.add_argument("--template-file", default=str(DEFAULT_TEMPLATE))
    pipe_run.add_argument("--config-output")
    pipe_run.add_argument("--output-base", default=".")
    pipe_run.add_argument("--skip-download", action="store_true")
    pipe_run.add_argument("--skip-forecast", action="store_true")
    pipe_run.add_argument("--timeout", type=int, default=3600)
    pipe_run.add_argument("--nX", type=int, default=1)
    pipe_run.add_argument("--nY", type=int, default=1)
    pipe_run.set_defaults(handler=handle_pipeline_run)

    frigate_prepare = pipe_sub.add_parser("frigate-prepare", help="Prepare a FRIGATE run from a KML and date")
    frigate_prepare.add_argument("--region-kml", required=True)
    frigate_prepare.add_argument("--date", required=True)
    frigate_prepare.add_argument("--location-id")
    frigate_prepare.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    frigate_prepare.add_argument("--skip-download", action="store_true")
    frigate_prepare.add_argument("--timeout", type=int, default=3600)
    frigate_prepare.set_defaults(handler=handle_pipeline_frigate_prepare)
    return parser


def add_region_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--region-kml")
    parser.add_argument("--location-id")
    parser.add_argument("--locations-file", default=str(DEFAULT_LOCATIONS))


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--nx1", type=int, required=True)
    parser.add_argument("--nx2", type=int, required=True)
    parser.add_argument("--nx3", type=int, required=True)
    parser.add_argument("--nghost", type=int, default=3)
    parser.add_argument("--x1-max", type=float, default=20000.0)
    parser.add_argument("--x2-extent", type=float)
    parser.add_argument("--x3-extent", type=float)
    parser.add_argument("--tlim", type=int, default=86400)


def resolve_region(args: argparse.Namespace):
    return load_region(
        region_kml=args.region_kml,
        location_id=args.location_id,
        locations_file=args.locations_file,
    )


def handle_region_inspect(args: argparse.Namespace) -> int:
    if not args.region_kml and not args.location_id:
        regions = load_regions_from_csv(args.locations_file)
        for region_id, region in sorted(regions.items()):
            print(f"{region_id}: {region.name}")
        return 0

    region = resolve_region(args)
    lon_min, lat_min, lon_max, lat_max = region.bounds
    center = region.center
    print(f"region_id: {region.region_id}")
    print(f"name: {region.name}")
    print(f"source: {region.source}")
    print(f"bounds: lon=({lon_min}, {lon_max}) lat=({lat_min}, {lat_max})")
    print(f"center: lat={center['latitude']}, lon={center['longitude']}")
    return 0


def handle_config_generate(args: argparse.Namespace) -> int:
    region = resolve_region(args)
    template = Path(args.template_file).read_text(encoding="utf-8")
    options = ConfigOptions(
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
    )
    output = Path(args.output) if args.output else default_output_path(Path.cwd(), region)
    output.write_text(render_config(template, region, options), encoding="utf-8")
    print(output)
    return 0


def handle_topo_build(args: argparse.Namespace) -> int:
    script = PROJECT_ROOT / "um_earth" / "topography" / "download_crop.py"
    cmd = []
    if args.location_id:
        cmd.append(args.location_id)
    if args.region_kml:
        cmd.extend(["--region-kml", args.region_kml])
    cmd.extend(["--locations", args.locations_file])
    if args.out:
        cmd.extend(["--out", args.out])
    if args.skip_download:
        cmd.append("--skip-download")
    return _run_python(script, cmd)


def handle_init_prepare(args: argparse.Namespace) -> int:
    script = PROJECT_ROOT / "prepare_initial_condition.py"
    cmd = []
    if args.location_id:
        cmd.append(args.location_id)
    if args.region_kml:
        cmd.extend(["--region-kml", args.region_kml])
    cmd.extend(["--config", args.config, "--output-base", args.output_base])
    cmd.extend(["--start-from", str(args.start_from), "--stop-after", str(args.stop_after)])
    cmd.extend(["--timeout", str(args.timeout), "--nX", str(args.nX), "--nY", str(args.nY)])
    if args.times:
        cmd.extend(["--times", *args.times])
    cmd.extend(["--locations-file", args.locations_file])
    return _run_python(script, cmd)


def handle_forecast_run(args: argparse.Namespace) -> int:
    script = PROJECT_ROOT / "run_frigate_prediction.py"
    return _run_python(script, ["--config", args.config, "--input_dir", args.input_dir, "--output_dir", args.output_dir])


def handle_diagnostics_plot(args: argparse.Namespace) -> int:
    script = PROJECT_ROOT / "um_earth" / "diagnostics" / "generate_all_plots.py"
    cmd = [args.input_dir]
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.output_pdf:
        cmd.extend(["--output-pdf", args.output_pdf])
    if args.topo_dir:
        cmd.extend(["--topo-dir", args.topo_dir])
    if args.location:
        cmd.extend(["--location", args.location])
    if args.all_times:
        cmd.append("--all-times")
    return _run_python(script, cmd)


def handle_pipeline_run(args: argparse.Namespace) -> int:
    region = resolve_region(args)
    config_path = Path(args.config_output) if args.config_output else default_output_path(Path.cwd(), region)
    rc = handle_config_generate(
        argparse.Namespace(
            region_kml=args.region_kml,
            location_id=args.location_id or region.region_id,
            locations_file=args.locations_file,
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
            template_file=args.template_file,
            output=str(config_path),
        )
    )
    if rc:
        return rc

    topo_rc = handle_topo_build(
        argparse.Namespace(
            location_id=args.location_id or region.region_id,
            region_kml=args.region_kml,
            locations_file=args.locations_file,
            out=None,
            skip_download=args.skip_download,
        )
    )
    if topo_rc:
        return topo_rc

    init_rc = handle_init_prepare(
        argparse.Namespace(
            location_id=args.location_id or region.region_id,
            region_kml=args.region_kml,
            locations_file=args.locations_file,
            config=str(config_path),
            output_base=args.output_base,
            start_from=1,
            stop_after=6,
            timeout=args.timeout,
            nX=args.nX,
            nY=args.nY,
            times=None,
        )
    )
    if init_rc or args.skip_forecast:
        return init_rc

    return handle_forecast_run(
        argparse.Namespace(
            config=str(config_path),
            input_dir=str(Path(args.output_base)),
            output_dir=str(Path.cwd() / "output"),
        )
    )


def handle_pipeline_frigate_prepare(args: argparse.Namespace) -> int:
    run_dir = run_frigate_prepare(
        region_kml=args.region_kml,
        date=args.date,
        workspace_root=args.workspace_root,
        location_id=args.location_id,
        skip_download=args.skip_download,
        timeout=args.timeout,
    )
    print(run_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
