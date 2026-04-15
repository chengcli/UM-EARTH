# FRIGATE Pipeline

This document describes the current FRIGATE workflow in `UM-EARTH` for both
reanalysis-driven and forecast-driven runs.

The operational pattern now has three stages:

1. Fetch or ingest data and prepare restart inputs.
2. Run a low-resolution two-GPU check.
3. Run the full refined two-GPU case.

The examples below use the active `pte1b` forecast cases:

* one-GPU forecast case: `2026-04-12`
* two-GPU forecast case: `2026-04-13`

## Inputs

Required inputs:

* a region KML file
* a target date in `YYYY-MM-DD` format
* either ERA5 access or a local forecast-data directory

Example region:

* `/home/chengcli/data/2025.FRIGATE/pte1b.kml`

Default run root:

* `/home/chengcli/data/2025.FRIGATE/runs`

## Stage 1: Fetch Or Ingest Data

FRIGATE supports two data-source modes.

### Option A: ERA5

Use this when you want to download reanalysis fields through the existing ERA5
pipeline.

Example:

```bash
python3 -m um_earth.cli pipeline frigate-prepare \
  --region-kml /home/chengcli/data/2025.FRIGATE/pte1b.kml \
  --date 2026-04-01 \
  --data-source era5
```

Typical outputs:

* `<run>/<region>.yaml`
* `<run>/topography/products/*.pt`
* `<run>/era5/<bounds>/era5_hourly_dynamics_<date>.nc`
* `<run>/era5/<bounds>/era5_hourly_densities_<date>.nc`
* `<run>/era5/<bounds>/era5_density_<date>.nc`
* `<run>/era5/<bounds>/regridded_<region>_<date>.nc`
* `<run>/era5/<bounds>/regridded_<region>_<date>_tensors/*.part`

### Option B: Prediction Data

Use this when forecast GRIB2 files already exist locally.

Example source directory:

* `/home/chengcli/data/2025.FRIGATE/ECWMF_prediction_data/20260413`

For forecast mode, `prepare_initial_condition.py` ingests the forecast GRIB
files, writes intermediate NetCDF products, decomposes the regridded domain,
and produces runtime restart inputs.

Example for the April 13 two-GPU case:

```bash
python /home/chengcli/scix/workspace/UM-EARTH/prepare_initial_condition.py \
  pte1b \
  --region-kml /home/chengcli/data/2025.FRIGATE/pte1b.kml \
  --config /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/pte1b.yaml \
  --output-base /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_input \
  --data-source forecast \
  --forecast-input-dir /home/chengcli/data/2025.FRIGATE/ECWMF_prediction_data/20260413 \
  --forecast-cycle 00 \
  --forecast-leads 0 6 12 18 \
  --nY 2 --nX 1 \
  --timeout 7200
```

Typical forecast outputs:

* `forecast_hourly_dynamics_<date>_<cycle>.nc`
* `forecast_hourly_densities_<date>_<cycle>.nc`
* `forecast_density_<date>_<cycle>.nc`
* `regridded_<region>_<date>_<cycle>.nc`
* `regridded_<region>_<date>_<cycle>_blocks/*.nc`

For a one-block run:

* `regridded_<region>_<date>_<cycle>_tensors/*.part`

For a two-block run:

* `regridded_<region>_<date>_<cycle>_tensors/<basename>.block0.00000.part`
* `regridded_<region>_<date>_<cycle>_tensors/<basename>.block1.00000.part`
* `regridded_<region>_<date>_<cycle>_tensors/<basename>.restart`

## Run Directory Layout

Typical prepared run tree:

```text
runs/<region>-<date>/
  <region>.yaml
  region_digest.json
  run_manifest.yaml
  plots/
  topography/
    raw/
    products/
  era5/ or forecast_input/
```

For production forecast execution, additional output directories are created,
for example:

```text
forecast_lowres_ghost_24h/
forecast_lowres_ghost_24h_2gpu/
forecast_refined_ghost_24h/
forecast_refined_ghost_24h_2gpu/
```

## Stage 2: Low-Resolution Check With 2 GPUs

Before launching the refined case, run a low-resolution two-GPU check. This
verifies:

* the bundled restart file can be read by rank
* NCCL initialization works
* block-local topography loading works
* ghost-zone forcing works on the decomposed mesh

### Configuration Requirements

For the current two-GPU workflow:

* `distribute.nb2: 2`
* `distribute.nb3: 1`
* preparation must use `--nY 2 --nX 1`
* runtime launch must use `torchrun --nproc-per-node=2`

### Example Low-Resolution Smoke Check

This is the short April 13 two-GPU validation run:

```bash
env CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29531 PYTHONFAULTHANDLER=1 \
  torchrun --nproc-per-node=2 \
  /home/chengcli/scix/workspace/UM-EARTH/run_frigate_prediction.py \
    -c /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/pte1b.yaml \
    -i /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_input/regridded_pte1b_20260413_00_tensors \
    -o /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_lowres_ghost_2gpu_smoke \
    --topography-dir /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/topography/products \
    --device cuda \
    --refinement-mode none \
    --forcing-mode ghost \
    --hydrostatic-duration 60 \
    --prediction-duration 1800 \
    2>&1 | tee /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_lowres_ghost_2gpu_smoke/run.log
```

Expected behavior:

* both ranks start on separate GPUs
* each rank loads a local restart block
* hydrostatic adjustment completes
* a short forecast segment advances and writes outputs

### Example Full Low-Resolution Two-GPU Run

```bash
env CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29532 PYTHONFAULTHANDLER=1 \
  torchrun --nproc-per-node=2 \
  /home/chengcli/scix/workspace/UM-EARTH/run_frigate_prediction.py \
    -c /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/pte1b.yaml \
    -i /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_input/regridded_pte1b_20260413_00_tensors \
    -o /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_lowres_ghost_24h_2gpu \
    --topography-dir /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/topography/products \
    --device cuda \
    --refinement-mode none \
    --forcing-mode ghost \
    --prediction-duration 86400 \
    2>&1 | tee /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_lowres_ghost_24h_2gpu/run.log
```

## Stage 3: Full Refined Case

The current refined forecast path supports staged refinement together with
ghost-zone forcing.

Schedule:

* start from the 00h slice on the base mesh
* refine at 06h
* refine again at 12h
* apply a ghost-only refresh at 18h
* continue on the `0p6km` mesh for the remaining forecast segment

For a full 24-hour forecast using the `00/06/12/18` forecast slices, launch the
refined run with:

* `--refinement-mode staged`
* `--forcing-mode ghost`
* `--prediction-duration 21600`

The `21600 s` value is intentional. The staged `06/12/18` schedule covers the
first 18 hours, and the final prediction segment covers the last 6 hours.

### Example Refined Two-GPU Run

```bash
env CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29533 PYTHONFAULTHANDLER=1 \
  torchrun --nproc-per-node=2 \
  /home/chengcli/scix/workspace/UM-EARTH/run_frigate_prediction.py \
    -c /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/pte1b.yaml \
    -i /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_input/regridded_pte1b_20260413_00_tensors \
    -o /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_refined_ghost_24h_2gpu \
    --topography-dir /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/topography/products \
    --device cuda \
    --refinement-mode staged \
    --forcing-mode ghost \
    --prediction-duration 21600 \
    2>&1 | tee /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_refined_ghost_24h_2gpu/run.log
```

## Example Detached Scripts

For production runs, use a detached launcher that survives logout. `tmux` is the
most reliable option in this environment.

### Example Script: Low-Resolution Two-GPU Run

```bash
cat > /tmp/pte1b_20260413_lowres_2gpu.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
run_dir=/home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13
out_dir=$run_dir/forecast_lowres_ghost_24h_2gpu
mkdir -p "$out_dir"
exec env CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29532 PYTHONFAULTHANDLER=1 \
  torchrun --nproc-per-node=2 \
  /home/chengcli/scix/workspace/UM-EARTH/run_frigate_prediction.py \
  -c "$run_dir/pte1b.yaml" \
  -i "$run_dir/forecast_input/regridded_pte1b_20260413_00_tensors" \
  -o "$out_dir" \
  --topography-dir "$run_dir/topography/products" \
  --device cuda \
  --refinement-mode none \
  --forcing-mode ghost \
  --prediction-duration 86400 \
  >> "$out_dir/run.log" 2>&1
EOF

chmod +x /tmp/pte1b_20260413_lowres_2gpu.sh
tmux new-session -d -s pte1b_20260413_lowres_2gpu /usr/bin/bash /tmp/pte1b_20260413_lowres_2gpu.sh
```

### Example Script: Refined Two-GPU Run

```bash
cat > /tmp/pte1b_20260413_refined_2gpu.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
run_dir=/home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13
out_dir=$run_dir/forecast_refined_ghost_24h_2gpu
mkdir -p "$out_dir"
exec env CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29533 PYTHONFAULTHANDLER=1 \
  torchrun --nproc-per-node=2 \
  /home/chengcli/scix/workspace/UM-EARTH/run_frigate_prediction.py \
  -c "$run_dir/pte1b.yaml" \
  -i "$run_dir/forecast_input/regridded_pte1b_20260413_00_tensors" \
  -o "$out_dir" \
  --topography-dir "$run_dir/topography/products" \
  --device cuda \
  --refinement-mode staged \
  --forcing-mode ghost \
  --prediction-duration 21600 \
  >> "$out_dir/run.log" 2>&1
EOF

chmod +x /tmp/pte1b_20260413_refined_2gpu.sh
tmux new-session -d -s pte1b_20260413_refined_2gpu /usr/bin/bash /tmp/pte1b_20260413_refined_2gpu.sh
```

### Reattach Or Monitor

```bash
tmux ls
tmux attach -t pte1b_20260413_refined_2gpu
tail -n 80 /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_refined_ghost_24h_2gpu/run.log
```

## Notes

### One-GPU Versus Two-GPU Inputs

One-GPU runs use a single-block input and typically launch `python
run_frigate_prediction.py` directly.

Two-GPU runs use:

* `nb2: 2`
* `nb3: 1`
* `--nY 2 --nX 1` during preparation
* a bundled `.restart` tarball plus rank-local `.part` files
* `torchrun --nproc-per-node=2` at runtime

### Why `nvidia-smi` Shows Three Python Processes

A two-GPU launch often shows three Python processes:

* one `torchrun` launcher
* rank 0 worker
* rank 1 worker

That is expected.

### Source Of Truth

For runtime behavior, treat the following as authoritative:

* `run_frigate_prediction.py`
* `prepare_initial_condition.py`
* `um_earth/frigate_pipeline.py`
* `snapy` C++ source under `src/`
* `snapy` Python bindings under `python/snapy/`
