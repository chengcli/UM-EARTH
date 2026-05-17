#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-15}"
OUTPUT_DIR="${RUN_ROOT}/diagnostics/hourly_wind_24h_48h"
RUN_OUTPUT_DIR="${RUN_ROOT}/forecast_refined_ghost_60h_2gpu_solar"
CONFIG_FILE="${RUN_OUTPUT_DIR}/pte1b.refined.yaml"
TOPO_DIR="${RUN_ROOT}/topography/products"
LOCATION="pte1b"

mkdir -p "${OUTPUT_DIR}"

for hour in $(seq -w 24 48); do
  python3 /home/chengcli/scix/workspace/UM-EARTH/scripts/plot_wind_altitude_panels.py \
    "${RUN_OUTPUT_DIR}/pte1b.out1.${hour}.nc" \
    --config "${CONFIG_FILE}" \
    --topo-dir "${TOPO_DIR}" \
    --location "${LOCATION}" \
    --output "${OUTPUT_DIR}/wind_altitude_panels_${hour}.png"
done
