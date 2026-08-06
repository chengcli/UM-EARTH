#!/usr/bin/env python3
"""Decode ECMWF forecast GRIB2 files into the NetCDF contract used downstream."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import xarray as xr

try:
    from .data_products import (
        build_product_stem,
        densities_filename,
        dynamics_filename,
    )
except ImportError:  # pragma: no cover - script execution path
    from data_products import (
        build_product_stem,
        densities_filename,
        dynamics_filename,
    )


FORECAST_FILE_RE = re.compile(r"^ifs_(?P<date>\d{8})_(?P<cycle>\d{2})_(?P<kind>pl|sfc)\.grib2$")
HRES_FILE_RE = re.compile(
    r"^.+_fc_(?P<base>\d{8}T\d{6}Z)_(?P<valid>\d{8}T\d{6}Z)_(?P<lead>\d+)h$"
)
PRESSURE_VARIABLES = ("t", "u", "v", "w", "q", "ciwc", "clwc", "cswc", "crwc")
SURFACE_TOPO_VARS = ("z", "orog", "topography", "elevation")


def discover_forecast_inputs(input_dir: str | Path) -> dict[str, dict[str, dict[str, Path]]]:
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Forecast input directory not found: {root}")

    discovered: dict[str, dict[str, dict[str, Path]]] = {}
    for path in sorted(root.glob("*.grib2")):
        match = FORECAST_FILE_RE.match(path.name)
        if not match:
            continue
        date_str = match.group("date")
        cycle = match.group("cycle")
        kind = match.group("kind")
        discovered.setdefault(date_str, {}).setdefault(cycle, {})[kind] = path
    return discovered


def discover_hres_inputs(
    input_dir: str | Path,
    *,
    file_prefix: str | None = None,
) -> dict[str, dict[str, dict[int, Path]]]:
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Forecast input directory not found: {root}")

    discovered: dict[str, dict[str, dict[int, Path]]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if file_prefix is not None and not path.name.startswith(file_prefix):
            continue
        match = HRES_FILE_RE.match(path.name)
        if not match:
            continue
        base = match.group("base")
        date_str = base[:8]
        cycle = base[9:11]
        lead = int(match.group("lead"))
        valid = match.group("valid")
        expected_valid = np.datetime64(
            f"{base[:4]}-{base[4:6]}-{base[6:8]}T{base[9:11]}:{base[11:13]}:{base[13:15]}"
        ) + np.timedelta64(lead, "h")
        actual_valid = np.datetime64(
            f"{valid[:4]}-{valid[4:6]}-{valid[6:8]}T{valid[9:11]}:{valid[11:13]}:{valid[13:15]}"
        )
        if actual_valid != expected_valid:
            raise ValueError(
                f"HRES filename has inconsistent valid time and lead: {path.name}"
            )
        cycle_files = discovered.setdefault(date_str, {}).setdefault(cycle, {})
        if lead in cycle_files:
            raise ValueError(
                f"Duplicate HRES lead {lead} for {date_str} cycle {cycle}: "
                f"{cycle_files[lead].name}, {path.name}"
            )
        cycle_files[lead] = path
    return discovered


def select_forecast_cycle(
    input_dir: str | Path,
    *,
    date_str: str | None = None,
    cycle: str | None = None,
) -> tuple[str, str, dict[str, Path]]:
    discovered = discover_forecast_inputs(input_dir)
    if not discovered:
        raise FileNotFoundError(f"No forecast GRIB2 files found in {input_dir}")

    if date_str is None:
        date_str = sorted(discovered)[0]
    if date_str not in discovered:
        raise FileNotFoundError(f"No forecast data found for date {date_str} in {input_dir}")

    cycles = discovered[date_str]
    if cycle is None:
        cycle = sorted(cycles)[0]
    if cycle not in cycles:
        raise FileNotFoundError(f"No forecast data found for cycle {cycle} on {date_str}")

    files = cycles[cycle]
    missing = [kind for kind in ("pl", "sfc") if kind not in files]
    if missing:
        raise FileNotFoundError(
            f"Missing required forecast GRIB files for {date_str} cycle {cycle}: {missing}"
        )
    return date_str, cycle, files


def select_hres_cycle(
    input_dir: str | Path,
    *,
    date_str: str | None = None,
    cycle: str | None = None,
    lead_hours: Iterable[int] = (0, 6, 12, 18),
    file_prefix: str | None = None,
) -> tuple[str, str, dict[int, Path]]:
    discovered = discover_hres_inputs(input_dir, file_prefix=file_prefix)
    if not discovered:
        raise FileNotFoundError(f"No HRES delivery files found in {input_dir}")

    if date_str is None:
        date_str = sorted(discovered)[-1]
    if date_str not in discovered:
        raise FileNotFoundError(f"No HRES data found for date {date_str} in {input_dir}")

    cycles = discovered[date_str]
    if cycle is None:
        cycle = sorted(cycles)[-1]
    if cycle not in cycles:
        raise FileNotFoundError(f"No HRES data found for cycle {cycle} on {date_str}")

    files = cycles[cycle]
    requested = list(lead_hours)
    missing = [lead for lead in requested if lead not in files]
    if missing:
        raise ValueError(
            f"Requested HRES lead hours are unavailable for {date_str} cycle {cycle}: {missing}"
        )
    return date_str, cycle, {lead: files[lead] for lead in requested}


def _open_grib_dataset(
    path: Path,
    *,
    filter_by_keys: dict[str, str] | None = None,
) -> xr.Dataset:
    backend_kwargs: dict[str, object] = {"indexpath": ""}
    if filter_by_keys:
        backend_kwargs["filter_by_keys"] = filter_by_keys
        backend_kwargs["errors"] = "ignore"
    try:
        return xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs=backend_kwargs,
        )
    except ValueError as exc:
        raise ImportError(
            "Forecast GRIB support requires the 'cfgrib' backend and ecCodes. "
            "Install cfgrib and eccodes before running forecast preparation."
        ) from exc


def _open_hres_pressure_file(path: Path) -> xr.Dataset:
    ds = _rename_standard_coords(
        _open_grib_dataset(
            path,
            filter_by_keys={"typeOfLevel": "isobaricInhPa"},
        )
    )
    if "pressure_level" not in ds.coords:
        raise ValueError(f"HRES file is missing pressure levels: {path}")
    ordered_pressure_vars = [name for name in PRESSURE_VARIABLES if name in ds.data_vars]
    if "t" not in ordered_pressure_vars or "q" not in ordered_pressure_vars:
        raise ValueError(f"HRES pressure data must contain at least 't' and 'q': {path}")
    subset = ds[ordered_pressure_vars].load()
    for var_name in ("ciwc", "clwc", "cswc", "crwc"):
        if var_name not in subset:
            subset[var_name] = xr.zeros_like(subset["q"])
    valid_time = np.datetime64(ds["valid_time"].values)
    subset = subset.expand_dims(time=[valid_time])
    return subset.transpose("time", "pressure_level", "latitude", "longitude")


def _ingest_hres_cycle(
    input_dir: str | Path,
    output_root: Path,
    *,
    date_str: str | None,
    cycle: str | None,
    lead_hours: Iterable[int],
    file_prefix: str | None,
) -> dict[str, str]:
    leads = list(lead_hours)
    resolved_date, resolved_cycle, files = select_hres_cycle(
        input_dir,
        date_str=date_str,
        cycle=cycle,
        lead_hours=leads,
        file_prefix=file_prefix,
    )
    slices = [_open_hres_pressure_file(files[lead]) for lead in leads]
    ds_pressure = xr.concat(
        slices,
        dim="time",
        coords="minimal",
        compat="override",
    )
    stem = build_product_stem(resolved_date, resolved_cycle)

    dynamics = ds_pressure[[name for name in ("t", "u", "v", "w") if name in ds_pressure.data_vars]]
    densities = ds_pressure[["q", "ciwc", "clwc", "cswc", "crwc"]]
    dynamics_path = output_root / dynamics_filename("forecast", stem)
    densities_path = output_root / densities_filename("forecast", stem)
    dynamics.to_netcdf(dynamics_path)
    densities.to_netcdf(densities_path)

    valid_times = [str(value) for value in ds_pressure["time"].values]
    return {
        "date": resolved_date,
        "cycle": resolved_cycle,
        "stem": stem,
        "dynamics": str(dynamics_path),
        "densities": str(densities_path),
        "valid_times": ", ".join(valid_times),
        "format": "hres-delivery",
    }


def _normalize_step_values(step_values: np.ndarray, requested_hours: Iterable[int]) -> np.ndarray:
    requested = list(requested_hours)
    if step_values.dtype.kind == "m":
        requested_values = np.array([np.timedelta64(hour, "h") for hour in requested], dtype=step_values.dtype)
    else:
        requested_values = np.array(requested, dtype=step_values.dtype)
    missing = [hour for hour, raw in zip(requested, requested_values, strict=True) if raw not in step_values]
    if missing:
        raise ValueError(f"Requested forecast lead hours are unavailable: {missing}")
    return requested_values


def _assign_time_axis(ds: xr.Dataset, lead_hours: Iterable[int]) -> xr.Dataset:
    lead_hours = list(lead_hours)
    if "step" in ds.dims:
        requested_steps = _normalize_step_values(ds["step"].values, lead_hours)
        ds = ds.sel(step=requested_steps)
        if "valid_time" in ds.coords:
            time_values = ds["valid_time"].values
        elif "time" in ds.coords:
            base_time = ds["time"].values
            if np.ndim(base_time) == 0:
                if ds["step"].dtype.kind == "m":
                    time_values = base_time + ds["step"].values
                else:
                    time_values = np.array(
                        [np.datetime64(base_time) + np.timedelta64(int(hour), "h") for hour in ds["step"].values]
                    )
            else:
                time_values = base_time
        else:
            raise ValueError("Forecast GRIB dataset is missing both 'step' and 'valid_time' coordinates.")

        ds = ds.assign_coords(time=("step", time_values))
        ds = ds.swap_dims({"step": "time"})
        ds = ds.drop_vars("step", errors="ignore")
    elif "time" in ds.dims:
        available = ds["time"].values
        if len(available) < len(lead_hours):
            raise ValueError(
                f"Forecast dataset contains {len(available)} time slices, "
                f"but {len(lead_hours)} were requested."
            )
        ds = ds.isel(time=slice(0, len(lead_hours)))
    else:
        ds = ds.expand_dims(time=[0])
    return ds


def _rename_standard_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_map: dict[str, str] = {}
    if "isobaricInhPa" in ds.coords:
        rename_map["isobaricInhPa"] = "pressure_level"
    if "latitude" not in ds.coords and "lat" in ds.coords:
        rename_map["lat"] = "latitude"
    if "longitude" not in ds.coords and "lon" in ds.coords:
        rename_map["lon"] = "longitude"
    return ds.rename(rename_map) if rename_map else ds


def _normalize_pressure_dataset(ds: xr.Dataset, lead_hours: Iterable[int]) -> xr.Dataset:
    ds = _rename_standard_coords(ds)
    ds = _assign_time_axis(ds, lead_hours)
    if "pressure_level" not in ds.coords:
        raise ValueError("Pressure-level forecast dataset is missing the pressure-level coordinate.")
    ordered_pressure_vars = [name for name in PRESSURE_VARIABLES if name in ds.data_vars]
    if "t" not in ordered_pressure_vars or "q" not in ordered_pressure_vars:
        raise ValueError("Forecast pressure-level file must contain at least 't' and 'q'.")
    subset = ds[ordered_pressure_vars]
    for var_name in ("ciwc", "clwc", "cswc", "crwc"):
        if var_name not in subset:
            subset[var_name] = xr.zeros_like(subset["q"])
    return subset.transpose("time", "pressure_level", "latitude", "longitude")


def _maybe_write_surface_topography(ds: xr.Dataset, output_dir: Path, stem: str) -> None:
    ds = _rename_standard_coords(ds)
    topo_var = next((name for name in SURFACE_TOPO_VARS if name in ds.data_vars), None)
    if topo_var is None:
        return
    topo = ds[topo_var]
    if "time" in topo.dims:
        topo = topo.isel(time=0, drop=True)
    if "step" in topo.dims:
        topo = topo.isel(step=0, drop=True)
    if topo.ndim != 2:
        return
    values = topo.values.astype(np.float32)
    units = str(topo.attrs.get("units", "")).lower()
    if topo_var == "z" or "m2" in units:
        values = values / 9.80665
    topo_ds = xr.Dataset({"topography": (("latitude", "longitude"), values)}, coords={
        "latitude": ds["latitude"].values,
        "longitude": ds["longitude"].values,
    })
    topo_ds.to_netcdf(output_dir / f"forecast_topography_{stem}.nc")


def ingest_forecast_cycle(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    date_str: str | None = None,
    cycle: str | None = None,
    lead_hours: Iterable[int] = (0, 6, 12, 18),
    file_prefix: str | None = None,
) -> dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    lead_hours = list(lead_hours)

    if discover_hres_inputs(input_dir, file_prefix=file_prefix):
        return _ingest_hres_cycle(
            input_dir,
            output_root,
            date_str=date_str,
            cycle=cycle,
            lead_hours=lead_hours,
            file_prefix=file_prefix,
        )

    resolved_date, resolved_cycle, files = select_forecast_cycle(
        input_dir,
        date_str=date_str,
        cycle=cycle,
    )
    stem = build_product_stem(resolved_date, resolved_cycle)
    ds_pressure = _normalize_pressure_dataset(_open_grib_dataset(files["pl"]), lead_hours)
    ds_surface = _open_grib_dataset(files["sfc"])

    dynamics = ds_pressure[[name for name in ("t", "u", "v", "w") if name in ds_pressure.data_vars]]
    densities = ds_pressure[["q", "ciwc", "clwc", "cswc", "crwc"]]

    dynamics_path = output_root / dynamics_filename("forecast", stem)
    densities_path = output_root / densities_filename("forecast", stem)
    dynamics.to_netcdf(dynamics_path)
    densities.to_netcdf(densities_path)
    _maybe_write_surface_topography(ds_surface, output_root, stem)

    valid_times = [str(value) for value in ds_pressure["time"].values]
    return {
        "date": resolved_date,
        "cycle": resolved_cycle,
        "stem": stem,
        "dynamics": str(dynamics_path),
        "densities": str(densities_path),
        "valid_times": ", ".join(valid_times),
        "format": "legacy-pair",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ECMWF forecast GRIB2 files into NetCDF intermediates.")
    parser.add_argument("--input-dir", required=True, help="Directory containing ifs_*_{pl,sfc}.grib2 files.")
    parser.add_argument("--output-dir", required=True, help="Directory for NetCDF outputs.")
    parser.add_argument("--date", default=None, help="Forecast date in YYYYMMDD form. Defaults to earliest available.")
    parser.add_argument("--cycle", default=None, help="Forecast cycle hour (00/06/12/18). Defaults to earliest available.")
    parser.add_argument("--lead-hours", nargs="+", type=int, default=[0, 6, 12, 18], help="Forecast lead hours to extract.")
    parser.add_argument(
        "--file-prefix",
        default=None,
        help="Restrict HRES delivery discovery to filenames with this prefix.",
    )
    args = parser.parse_args()

    result = ingest_forecast_cycle(
        args.input_dir,
        args.output_dir,
        date_str=args.date,
        cycle=args.cycle,
        lead_hours=args.lead_hours,
        file_prefix=args.file_prefix,
    )
    print(f"[OK] Wrote forecast NetCDF intermediates for {result['date']} cycle {result['cycle']}")
    print(f"     Dynamics:  {result['dynamics']}")
    print(f"     Densities: {result['densities']}")
    print(f"     Valid times: {result['valid_times']}")


if __name__ == "__main__":
    main()
