#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/data00/2025.FRIGATE}"
RUNS_DIR="$BASE_DIR/runs"
WORKSPACE=/home/chengcli/scix/workspace/UM-EARTH
PYENV_ROOT=/home/chengcli/pyenv
PYTHON_BIN="$PYENV_ROOT/bin/python"
TORCHRUN_BIN="$PYENV_ROOT/bin/torchrun"
KML_FILE="$BASE_DIR/pte1b.kml"
HRES_DIR="$BASE_DIR/ECMWF_prediction_data/HRES"
TOPO_SOURCE_RUN="${TOPO_SOURCE_RUN:-$RUNS_DIR/pte1b-2026-04-01}"

TARGET_YMD="${1:?Usage: $0 YYYYMMDD}"
TARGET_DATE=$("$PYTHON_BIN" -c \
  'from datetime import datetime; import sys; print(datetime.strptime(sys.argv[1], "%Y%m%d").strftime("%Y-%m-%d"))' \
  "$TARGET_YMD")
FORECAST_CYCLE=00
FORECAST_START_UTC="${TARGET_DATE}T20:00:00Z"
FORECAST_END_SECONDS=201600
PREDICTION_DURATION=136800
RUN_DIR="$RUNS_DIR/pte1b-${TARGET_DATE}-hres00-lead20"
FORECAST_INPUT_DIR="$RUN_DIR/forecast_input"
TENSOR_DIR="$FORECAST_INPUT_DIR/regridded_pte1b_${TARGET_YMD}_${FORECAST_CYCLE}_tensors"
RESTART_BUNDLE="$TENSOR_DIR/regridded_pte1b_${TARGET_YMD}_${FORECAST_CYCLE}.restart"
TOPOGRAPHY_DIR="$RUN_DIR/topography/products"
RAW_TOPOGRAPHY_DIR="$RUN_DIR/topography/raw/pte1b"
SMOKE_DIR="$RUN_DIR/forecast_lowres_hres_2gpu_smoke"
OUTPUT_DIR="$RUN_DIR/forecast_refined_hres_hourly_56h_2gpu"
RUNTIME_RESTART="$OUTPUT_DIR/pte1b.final.restart"
REFINED_CONFIG="$OUTPUT_DIR/pte1b.refined.yaml"
COMPLETE_MARKER="$OUTPUT_DIR/run.complete"
SESSION_NAME="pte1b_${TARGET_YMD}_00_lead20_hres_refined_56h_2gpu"
DEFAULT_MASTER_PORT=$((29000 + 10#${TARGET_YMD:4:4}))
MASTER_PORT="${MASTER_PORT:-$DEFAULT_MASTER_PORT}"
SMOKE_PORT="${SMOKE_PORT:-$((MASTER_PORT - 1))}"
SOURCE_MERGED_TIF="$TOPO_SOURCE_RUN/topography/raw/pte1b/pte1b_merged_10m.tif"
SOURCE_TOPOGRAPHY_DIR="$TOPO_SOURCE_RUN/topography/products"

export PATH="$PYENV_ROOT/bin:$PATH"

if [[ ! -x "$PYTHON_BIN" || ! -x "$TORCHRUN_BIN" ]]; then
  echo "Required Python environment is incomplete under $PYENV_ROOT" >&2
  exit 1
fi
if [[ ! -d "$HRES_DIR" ]]; then
  echo "HRES input directory not found: $HRES_DIR" >&2
  exit 1
fi
if [[ ! -f "$KML_FILE" ]]; then
  echo "pte1b KML file not found: $KML_FILE" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_MERGED_TIF" ]]; then
  echo "Source merged topography not found: $SOURCE_MERGED_TIF" >&2
  exit 1
fi

mkdir -p "$RUN_DIR" "$FORECAST_INPUT_DIR" "$TOPOGRAPHY_DIR" "$RAW_TOPOGRAPHY_DIR"

"$PYTHON_BIN" - "$WORKSPACE" "$KML_FILE" "$RUN_DIR/pte1b.yaml" "$TARGET_DATE" <<'PY'
from pathlib import Path
import sys

workspace = Path(sys.argv[1])
kml_file = Path(sys.argv[2])
config_path = Path(sys.argv[3])
target_date = sys.argv[4]
sys.path.insert(0, str(workspace))

from um_earth.frigate_pipeline import build_prepared_domain, write_config
from um_earth.regions import load_region

region = load_region(region_kml=kml_file, location_id="pte1b")
prepared = build_prepared_domain(region)
write_config(config_path, region, prepared, date=target_date, nb2=2, nb3=1)
PY

cp "$SOURCE_MERGED_TIF" "$RAW_TOPOGRAPHY_DIR/pte1b_merged_10m.tif"
if [[ -f "$SOURCE_TOPOGRAPHY_DIR/pte1b_topo_2p4km.pt" \
   && -f "$SOURCE_TOPOGRAPHY_DIR/pte1b_topo_1p2km.pt" \
   && -f "$SOURCE_TOPOGRAPHY_DIR/pte1b_topo_0p6km.pt" \
   && -f "$SOURCE_TOPOGRAPHY_DIR/pte1b_topo_0p3km.pt" ]]; then
  cp "$SOURCE_TOPOGRAPHY_DIR"/pte1b_topo_*.pt "$TOPOGRAPHY_DIR"/
else
  echo "Required staged topography products are missing in $SOURCE_TOPOGRAPHY_DIR" >&2
  exit 1
fi

# Lead 20 is simulation time zero. The 6/12/18-hour refinement states are
# therefore delivery leads 26/32/38, followed by hourly forcing through lead 76.
FORECAST_LEADS=(20 26 32)
for lead in $(seq 38 76); do
  FORECAST_LEADS+=("$lead")
done

if [[ ! -f "$RESTART_BUNDLE" ]]; then
  "$PYTHON_BIN" "$WORKSPACE/prepare_initial_condition.py" \
    pte1b \
    --region-kml "$KML_FILE" \
    --config "$RUN_DIR/pte1b.yaml" \
    --output-base "$FORECAST_INPUT_DIR" \
    --data-source forecast \
    --forecast-input-dir "$HRES_DIR" \
    --forecast-file-prefix uom_a1_ \
    --forecast-cycle "$FORECAST_CYCLE" \
    --forecast-leads "${FORECAST_LEADS[@]}" \
    --nY 2 --nX 1 \
    --timeout 14400
fi

if [[ ! -f "$RESTART_BUNDLE" ]]; then
  echo "Prepared restart bundle not found: $RESTART_BUNDLE" >&2
  exit 1
fi

if [[ ! -f "$SMOKE_DIR/smoke.ok" ]]; then
  mkdir -p "$SMOKE_DIR"
  env \
    DEVICE=cuda \
    BACKEND=ucx \
    CUDA_VISIBLE_DEVICES=0,1 \
    PYTHONFAULTHANDLER=1 \
    "$TORCHRUN_BIN" --nproc-per-node=2 --master-port "$SMOKE_PORT" \
      "$WORKSPACE/run_frigate_prediction.py" \
      -c "$RUN_DIR/pte1b.yaml" \
      -i "$TENSOR_DIR" \
      -o "$SMOKE_DIR" \
      --topography-dir "$TOPOGRAPHY_DIR" \
      --device cuda \
      --refinement-mode none \
      --forcing-mode ghost \
      --hydrostatic-duration 60 \
      --prediction-duration 1800 \
      --forecast-start-time-utc "$FORECAST_START_UTC" \
      2>&1 | tee "$SMOKE_DIR/run.log"
  touch "$SMOKE_DIR/smoke.ok"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Production session already running: $SESSION_NAME"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
if [[ -f "$COMPLETE_MARKER" ]]; then
  echo "Production run is already complete: $COMPLETE_MARKER"
  exit 0
fi

if [[ -f "$RUNTIME_RESTART" && -f "$REFINED_CONFIG" ]]; then
  echo "Resuming production from: $RUNTIME_RESTART"
  cmd="cd $WORKSPACE && set -o pipefail && env DEVICE=cuda BACKEND=ucx CUDA_VISIBLE_DEVICES=0,1 PYTHONFAULTHANDLER=1 PATH=$PYENV_ROOT/bin:\$PATH $TORCHRUN_BIN --nproc-per-node=2 --master-port $MASTER_PORT $WORKSPACE/run_frigate_prediction.py -c $REFINED_CONFIG -i $RUNTIME_RESTART --boundary-input $TENSOR_DIR --boundary-config $RUN_DIR/pte1b.yaml -o $OUTPUT_DIR --device cuda --refinement-mode staged --forcing-mode ghost --prediction-duration $PREDICTION_DURATION --forecast-end-time-seconds $FORECAST_END_SECONDS --forecast-start-time-utc $FORECAST_START_UTC 2>&1 | tee -a $OUTPUT_DIR/run.log && touch $COMPLETE_MARKER"
else
  cmd="cd $WORKSPACE && set -o pipefail && env DEVICE=cuda BACKEND=ucx CUDA_VISIBLE_DEVICES=0,1 PYTHONFAULTHANDLER=1 PATH=$PYENV_ROOT/bin:\$PATH $TORCHRUN_BIN --nproc-per-node=2 --master-port $MASTER_PORT $WORKSPACE/run_frigate_prediction.py -c $RUN_DIR/pte1b.yaml -i $TENSOR_DIR -o $OUTPUT_DIR --topography-dir $TOPOGRAPHY_DIR --device cuda --refinement-mode staged --forcing-mode ghost --hydrostatic-duration 600 --spinup-chunk-duration 21600 --prediction-duration $PREDICTION_DURATION --forecast-end-time-seconds $FORECAST_END_SECONDS --forecast-start-time-utc $FORECAST_START_UTC 2>&1 | tee $OUTPUT_DIR/run.log && touch $COMPLETE_MARKER"
fi

tmux new-session -d -s "$SESSION_NAME" "$cmd"

echo "Started production session: $SESSION_NAME"
echo "Master port: $MASTER_PORT"
echo "Run directory: $RUN_DIR"
echo "Log: $OUTPUT_DIR/run.log"
