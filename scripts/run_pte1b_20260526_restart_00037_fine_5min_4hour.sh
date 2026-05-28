#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/home/chengcli/data/2025.FRIGATE}"
WORKSPACE=/home/chengcli/scix/workspace/UM-EARTH
PYENV_ROOT=/home/chengcli/pyenv
PYTHON_BIN="$PYENV_ROOT/bin/python"
TORCHRUN_BIN="$PYENV_ROOT/bin/torchrun"

RUN_DIR="${RUN_DIR:-$BASE_DIR/runs/pte1b-2026-05-26}"
SOURCE_OUTPUT_DIR="${SOURCE_OUTPUT_DIR:-$RUN_DIR/forecast_refined_ghost_78h_2gpu_refine18}"
RESTART_FILE="${RESTART_FILE:-$SOURCE_OUTPUT_DIR/pte1b.00037.restart}"
SOURCE_CONFIG="${SOURCE_CONFIG:-$SOURCE_OUTPUT_DIR/pte1b.refined.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/forecast_fine_output_5min_4hour}"
FINE_CONFIG="$OUTPUT_DIR/pte1b.fine_5min.yaml"
SESSION_NAME="${SESSION_NAME:-pte1b_20260526_restart_00037_fine_5min_4hour}"
MASTER_PORT="${MASTER_PORT:-29637}"
PREDICTION_DURATION="${PREDICTION_DURATION:-14400}"
OUTPUT_DT="${OUTPUT_DT:-300.0}"

export PATH="$PYENV_ROOT/bin:$PATH"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$TORCHRUN_BIN" ]]; then
  echo "torchrun executable not found: $TORCHRUN_BIN" >&2
  exit 1
fi

if [[ ! -f "$RESTART_FILE" ]]; then
  echo "Restart file not found: $RESTART_FILE" >&2
  exit 1
fi

if [[ ! -f "$SOURCE_CONFIG" ]]; then
  echo "Source config not found: $SOURCE_CONFIG" >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session already running: $SESSION_NAME"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" - "$SOURCE_CONFIG" "$FINE_CONFIG" "$OUTPUT_DT" <<'PY'
from pathlib import Path
import sys

import yaml

source_config = Path(sys.argv[1])
fine_config = Path(sys.argv[2])
output_dt = float(sys.argv[3])

config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
outputs = []
for output in config.get("outputs", []):
    if output.get("type") == "restart":
        continue
    updated = dict(output)
    updated["dt"] = output_dt
    outputs.append(updated)

if not outputs:
    raise SystemExit(f"{source_config} has no non-restart outputs to write")

config["outputs"] = outputs
fine_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

echo "Restart file: $RESTART_FILE"
echo "Source config: $SOURCE_CONFIG"
echo "Fine config: $FINE_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Output cadence: ${OUTPUT_DT}s for non-restart outputs"
echo "Prediction duration: ${PREDICTION_DURATION}s"

cmd="cd $WORKSPACE && env CUDA_VISIBLE_DEVICES=0,1 PYTHONFAULTHANDLER=1 PATH=$PYENV_ROOT/bin:\$PATH $TORCHRUN_BIN --nproc-per-node=2 --master-port $MASTER_PORT $WORKSPACE/run_frigate_prediction.py -c $FINE_CONFIG -i $RESTART_FILE -o $OUTPUT_DIR --device cuda --prediction-duration $PREDICTION_DURATION 2>&1 | tee $OUTPUT_DIR/run.log"

tmux new-session -d -s "$SESSION_NAME" "$cmd"

echo "Started session: $SESSION_NAME"
echo "Master port: $MASTER_PORT"
echo "Log: $OUTPUT_DIR/run.log"
