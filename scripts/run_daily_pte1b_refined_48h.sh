#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=/home/chengcli/data/2025.FRIGATE
RUNS_DIR="$BASE_DIR/runs"
TEMPLATE_RUN="$RUNS_DIR/pte1b-2026-04-14"
WORKSPACE=/home/chengcli/scix/workspace/UM-EARTH
KML_FILE="$BASE_DIR/pte1b.kml"

target_ymd=${1:-$(TZ=America/Detroit date +%Y%m%d)}
target_date=$(python - "$target_ymd" <<'PY'
from datetime import datetime
import sys

date = datetime.strptime(sys.argv[1], "%Y%m%d")
print(date.strftime("%Y-%m-%d"))
PY
)

forecast_dir="$BASE_DIR/ECWMF_prediction_data/$target_ymd"
run_dir="$RUNS_DIR/pte1b-$target_date"
forecast_input_dir="$run_dir/forecast_input"
tensor_dir="$forecast_input_dir/regridded_pte1b_${target_ymd}_00_tensors"
restart_bundle="$tensor_dir/regridded_pte1b_${target_ymd}_00.restart"
output_dir="$run_dir/forecast_refined_ghost_48h_2gpu"
topography_dir="$run_dir/topography/products"
session_name="pte1b_${target_ymd}_refined_48h_2gpu"
master_port=$((29000 + 10#${target_ymd:4:4}))

echo "Target date: $target_date"
echo "Forecast source: $forecast_dir"
echo "Run directory: $run_dir"

if [[ ! -d "$forecast_dir" ]]; then
  echo "Forecast input directory does not exist: $forecast_dir" >&2
  exit 1
fi

if [[ ! -f "$TEMPLATE_RUN/pte1b.yaml" ]]; then
  echo "Template config not found: $TEMPLATE_RUN/pte1b.yaml" >&2
  exit 1
fi

mkdir -p "$run_dir" "$topography_dir" "$forecast_input_dir"

if [[ ! -f "$run_dir/pte1b.yaml" ]]; then
  cp "$TEMPLATE_RUN/pte1b.yaml" "$run_dir/pte1b.yaml"
fi

python - "$run_dir/pte1b.yaml" "$target_date" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
target_date = sys.argv[2]
text = path.read_text()
text = re.sub(r"2026-\d{2}-\d{2}", target_date, text)
text = re.sub(r"nb2:\s*\d+", "nb2: 2", text)
text = re.sub(r"nb3:\s*\d+", "nb3: 1", text)
path.write_text(text)
PY

if ! compgen -G "$topography_dir/pte1b_topo_*.pt" > /dev/null; then
  cp "$TEMPLATE_RUN"/topography/products/pte1b_topo_*.pt "$topography_dir"/
fi

if [[ ! -f "$restart_bundle" ]]; then
  python "$WORKSPACE/prepare_initial_condition.py" \
    pte1b \
    --region-kml "$KML_FILE" \
    --config "$run_dir/pte1b.yaml" \
    --output-base "$forecast_input_dir" \
    --data-source forecast \
    --forecast-input-dir "$forecast_dir" \
    --forecast-cycle 00 \
    --forecast-leads 0 6 12 18 \
    --nY 2 --nX 1 \
    --timeout 7200
fi

if [[ ! -f "$restart_bundle" ]]; then
  echo "Prepared restart bundle not found: $restart_bundle" >&2
  exit 1
fi

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "Session already running: $session_name"
  exit 0
fi

mkdir -p "$output_dir"

cmd="cd $WORKSPACE && env CUDA_VISIBLE_DEVICES=0,1 PYTHONFAULTHANDLER=1 torchrun --nproc-per-node=2 --master-port $master_port $WORKSPACE/run_frigate_prediction.py -c $run_dir/pte1b.yaml -i $tensor_dir -o $output_dir --topography-dir $topography_dir --device cuda --refinement-mode staged --forcing-mode ghost --prediction-duration 108000 2>&1 | tee $output_dir/run.log"

tmux new-session -d -s "$session_name" "$cmd"

echo "Started session: $session_name"
echo "Master port: $master_port"
echo "Log: $output_dir/run.log"
