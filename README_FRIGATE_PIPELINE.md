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
* current refined 78-hour two-GPU case: `2026-05-25`

## Inputs

Required inputs:

* a region KML file
* a target date in `YYYY-MM-DD` format
* either ERA5 access or a local forecast-data directory
* forecast-mode runs require `cfgrib` and ecCodes in the active Python
  environment

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
forecast_refined_ghost_78h_2gpu/
forecast_refined_ghost_78h_2gpu_refine18/
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
* at 18h, either apply a ghost-only refresh or pass `--refine-at-18h` to
  refine onto the `0p3km` mesh
* continue for the final prediction segment

For a full 24-hour forecast using the `00/06/12/18` forecast slices, launch the
refined run with:

* `--refinement-mode staged`
* `--forcing-mode ghost`
* `--prediction-duration 21600`

The `21600 s` value is intentional. The staged `06/12/18` schedule covers the
first 18 hours, and the final prediction segment covers the last 6 hours.

For a 78-hour forecast using the same `00/06/12/18` forecast slices, use:

* `--refinement-mode staged`
* `--forcing-mode ghost`
* `--refine-at-18h` if the final sync should refine onto `0p3km`
* `--prediction-duration 216000`

The `216000 s` value is the 60-hour segment after the initial 18-hour staged
sync window, so the total forecast span is 78 hours.

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

### Current 78-Hour Two-GPU Run With 18h Refinement

The current one-off May 25 run is captured by:

```bash
/home/chengcli/scix/workspace/UM-EARTH/scripts/run_pte1b_20260525_refined_78h_2gpu_refine18.sh
```

It prepares and launches:

* forecast source: `/home/chengcli/data/2025.FRIGATE/ECWMF_prediction_data/20260525`
* run directory: `/home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-05-25`
* restart bundle: `forecast_input/regridded_pte1b_20260525_00_tensors/regridded_pte1b_20260525_00.restart`
* output directory: `forecast_refined_ghost_78h_2gpu_refine18`
* tmux session: `pte1b_20260525_refined_78h_2gpu_refine18`
* master port: `29525`

The runtime command uses:

```bash
env CUDA_VISIBLE_DEVICES=0,1 PYTHONFAULTHANDLER=1 \
  torchrun --nproc-per-node=2 --master-port 29525 \
  /home/chengcli/scix/workspace/UM-EARTH/run_frigate_prediction.py \
    -c /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-05-25/pte1b.yaml \
    -i /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-05-25/forecast_input/regridded_pte1b_20260525_00_tensors \
    -o /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-05-25/forecast_refined_ghost_78h_2gpu_refine18 \
    --topography-dir /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-05-25/topography/products \
    --device cuda \
    --refinement-mode staged \
    --forcing-mode ghost \
    --refine-at-18h \
    --prediction-duration 216000
```

## Example Detached Scripts

For production runs, use a detached launcher that survives logout. `tmux` is the
most reliable option in this environment.

The repo also carries maintained helper launchers:

* `scripts/run_frigate_prepare.sh YYYYMMDD`
* `scripts/run_frigate_pipeline.sh YYYYMMDD`

These mirror the current `pte1b` forecast workflow and use the axis-correct
April 14 template run (`pte1b-2026-04-14-axisfix`) when copying the base
configuration and topography products.

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

For the current May 25 78-hour run:

```bash
tmux attach -t pte1b_20260525_refined_78h_2gpu_refine18
tail -n 80 /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-05-25/forecast_refined_ghost_78h_2gpu_refine18/run.log
```

## Daily Automation

A daily refined `pte1b` run for a 78-hour forecast window is available via
`crontab`. In the current user crontab the entry is present but commented out.

Repo files:

* launcher: `/home/chengcli/scix/workspace/UM-EARTH/scripts/run_daily_pte1b_refined_78h.sh`
* crontab file: `/home/chengcli/scix/workspace/UM-EARTH/scripts/cron/pte1b_refined_78h.crontab`
* cron log target: `/home/chengcli/data/2025.FRIGATE/cron/pte1b_daily_refined_78h.log`

Crontab entry from the repo file:

```cron
CRON_TZ=America/Detroit
0 14 * * * /bin/bash /home/chengcli/scix/workspace/UM-EARTH/scripts/run_daily_pte1b_refined_78h.sh >> /home/chengcli/data/2025.FRIGATE/cron/pte1b_daily_refined_78h.log 2>&1
```

Behavior:

* target date defaults to the current ET calendar day
* unless `FORECAST_YMD` is set, the launcher picks the newest forecast
  directory on or before the previous ET calendar day
* source forecast root is `/home/chengcli/data/2025.FRIGATE/ECWMF_prediction_data`
* run directory is `/home/chengcli/data/2025.FRIGATE/runs/pte1b-YYYY-MM-DD`
* topography is copied from `/home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-01` by default
* preparation uses forecast mode with `--nY 2 --nX 1`
* runtime uses two GPUs with `torchrun --nproc-per-node=2`
* runtime mode is `--refinement-mode staged --forcing-mode ghost`
* daily staged refinement occurs at `06h` and `12h`, with a ghost-only update at `18h`
* `--prediction-duration 216000` is used so the total forecast span is 78 hours

Manual test example:

```bash
/home/chengcli/scix/workspace/UM-EARTH/scripts/run_daily_pte1b_refined_78h.sh 20260525
```

Install or refresh the cron job with:

```bash
crontab /home/chengcli/scix/workspace/UM-EARTH/scripts/cron/pte1b_refined_78h.crontab
crontab -l
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
