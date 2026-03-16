# FRIGATE Pipeline

This document summarizes the current KML-driven FRIGATE workflow in `UM-EARTH`.

The pipeline has three major stages:

1. Region preparation from a KML file
2. ERA5 and topography preprocessing for initial conditions
3. Forecast startup and prediction with staged refinement

## Inputs

Required inputs:

- A region KML file
- A target date in `YYYY-MM-DD` format

Example KML used during development:

- `/home/chengcli/scix/workspace/sacramento_valley.kml`

Default run root:

- `/home/chengcli/data/2025.FRIGATE/runs`

## Stage 1: Prepare FRIGATE Run

Use the CLI pipeline entrypoint:

```bash
python3 -m um_earth.cli pipeline frigate-prepare \
  --region-kml /home/chengcli/scix/workspace/sacramento_valley.kml \
  --date 2026-03-07
```

This creates a run directory:

- `/home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07`

Main outputs:

- `region_digest.json`
- `run_manifest.yaml`
- `sacramento_valley.yaml`
- `plots/`
- `topography/products/`
- `era5/<bounds>/`

The generated `sacramento_valley.yaml` is the canonical simulation input. There is no separate config-generation step required after `frigate-prepare`.

## Stage 2: Initial Condition Artifacts

The preparation stage drives the existing `prepare_initial_condition.py` workflow and writes the regridded ERA5 tensors used by the forecast driver.

Typical ERA5 output directory:

- `/home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07/era5/37.60N_39.40N_122.58W_120.79W`

Key files:

- `era5_hourly_densities_20260307.nc`
- `era5_hourly_dynamics_20260307.nc`
- `era5_density_20260307.nc`
- `regridded_sacramento_valley_20260307.nc`
- `regridded_sacramento_valley_20260307_blocks/regridded_sacramento_valley_20260307_block_0_0.nc`
- `regridded_sacramento_valley_20260307_tensors/regridded_sacramento_valley_20260307_block_0_0.part`

Topography products:

- `sacramento_valley_topo_2p4km.pt`
- `sacramento_valley_topo_1p2km.pt`
- `sacramento_valley_topo_0p6km.pt`
- `sacramento_valley_topo_0p3km.pt`

## Stage 3: Forecast Driver

The forecast driver is:

- [`run_frigate_prediction.py`](/home/chengcli/scix/workspace/UM-EARTH/run_frigate_prediction.py)

Current behavior:

- Uses `snapy.Mesh` as the main runtime object
- Uses the `paddle` `shallow_splash.py` distributed initialization pattern
- Uses `snapy` C++ and Python bindings as the source of truth for mesh construction
- Keeps the immersed-boundary-style `topo` mask for damping inside terrain
- Starts from the ERA5 tensor restart
- Runs an initial hydrostatic adjustment
- Runs staged spinup with successive mesh refinement
- Nudges back toward later ECMWF states
- Continues with forecast prediction

### Trial Run

Short CPU trial:

```bash
env PYTHONFAULTHANDLER=1 python3 /home/chengcli/scix/workspace/UM-EARTH/run_frigate_prediction.py \
  -c /home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07/sacramento_valley.yaml \
  -i /home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07/era5/37.60N_39.40N_122.58W_120.79W/regridded_sacramento_valley_20260307_tensors \
  -o /home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07/forecast_trial_v3_cpu \
  --device cpu \
  --hydrostatic-duration 10 \
  --spinup-chunk-duration 10 \
  --prediction-duration 20
```

Short CUDA trial:

```bash
env MASTER_PORT=29502 PYTHONFAULTHANDLER=1 python3 /home/chengcli/scix/workspace/UM-EARTH/run_frigate_prediction.py \
  -c /home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07/sacramento_valley.yaml \
  -i /home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07/era5/37.60N_39.40N_122.58W_120.79W/regridded_sacramento_valley_20260307_tensors \
  -o /home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07/forecast_trial_v3_cuda \
  --device cuda \
  --hydrostatic-duration 10 \
  --spinup-chunk-duration 10 \
  --prediction-duration 20
```

Notes:

- Use a distinct `MASTER_PORT` if another run is already active.
- The driver now uses `mesh.options.device_str()` as the source of truth for the mesh device after construction.

### Production Run

Example detached 24-hour CUDA run:

```bash
mkdir -p /home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07/forecast_production_24h

setsid bash -lc '
  exec env MASTER_PORT=29504 PYTHONFAULTHANDLER=1 \
    python3 /home/chengcli/scix/workspace/UM-EARTH/run_frigate_prediction.py \
      -c /home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07/sacramento_valley.yaml \
      -i /home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07/era5/37.60N_39.40N_122.58W_120.79W/regridded_sacramento_valley_20260307_tensors \
      -o /home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07/forecast_production_24h \
      --device cuda \
      --prediction-duration 86400 \
      </dev/null \
      >> /home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07/forecast_production_24h/run.log 2>&1
' &
```

Useful monitoring commands:

```bash
pgrep -af 'run_frigate_prediction.py|forecast_production_24h'
ps -p <PID> -o pid,etime,%cpu,%mem,stat,cmd
tail -n 40 /home/chengcli/data/2025.FRIGATE/runs/sacramento_valley-2026-03-07/forecast_production_24h/run.log
```

## Output Layout

Typical forecast output directory contents:

- `sacramento_valley.00000.restart`
- `sacramento_valley.final.restart`
- `sacramento_valley.out1.00000.nc`
- `sacramento_valley.out2.00000.nc`
- `sacramento_valley.refined.yaml`
- `run.log`

## Source Of Truth

For forecast runtime behavior, consult only:

- `paddle`: `example_py_scripts/shallow_splash.py`
- `paddle`: `example_py_scripts/shallow_splash.yaml`
- `snapy`: C++ source under `src/`
- `snapy`: Python bindings under `python/snapy/`

Older Python examples in sibling repos should not be treated as authoritative for the current `snapy` API.
