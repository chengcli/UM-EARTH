#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/data00/2025.FRIGATE}"
RUNS_DIR="$BASE_DIR/runs"
TOPO_SOURCE_RUN="${TOPO_SOURCE_RUN:-$RUNS_DIR/pte1b-2026-04-01}"
WORKSPACE=/home/chengcli/scix/workspace/UM-EARTH
KML_FILE="$BASE_DIR/pte1b.kml"
FORECAST_ROOT="$BASE_DIR/ECWMF_prediction_data"
PYENV_ROOT=/home/chengcli/pyenv
PYTHON_BIN="$PYENV_ROOT/bin/python"
TORCHRUN_BIN="$PYENV_ROOT/bin/torchrun"

export PATH="$PYENV_ROOT/bin:$PATH"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$TORCHRUN_BIN" ]]; then
  echo "torchrun executable not found: $TORCHRUN_BIN" >&2
  exit 1
fi

requested_ymd=${1:-$(TZ=America/Detroit date +%Y%m%d)}
if [[ -n "${FORECAST_YMD:-}" ]]; then
  target_ymd="$FORECAST_YMD"
else
  target_ymd=$("$PYTHON_BIN" - "$FORECAST_ROOT" "$requested_ymd" <<'PY'
from datetime import datetime, timedelta
from pathlib import Path
import sys

forecast_root = Path(sys.argv[1])
requested = sys.argv[2]
requested_dt = datetime.strptime(requested, "%Y%m%d")
source_cutoff = (requested_dt - timedelta(days=1)).strftime("%Y%m%d")
candidates = sorted(
    path.name
    for path in forecast_root.iterdir()
    if path.is_dir() and len(path.name) == 8 and path.name.isdigit() and path.name <= source_cutoff
)
if not candidates:
    raise SystemExit(
        f"No forecast directories available on or before previous day {source_cutoff} in {forecast_root}"
    )
print(candidates[-1])
PY
)
fi
target_date=$("$PYTHON_BIN" - "$target_ymd" <<'PY'
from datetime import datetime
import sys

date = datetime.strptime(sys.argv[1], "%Y%m%d")
print(date.strftime("%Y-%m-%d"))
PY
)

forecast_dir="$FORECAST_ROOT/$target_ymd"
run_dir="$RUNS_DIR/pte1b-$target_date"
forecast_input_dir="$run_dir/forecast_input"
tensor_dir="$forecast_input_dir/regridded_pte1b_${target_ymd}_00_tensors"
restart_bundle="$tensor_dir/regridded_pte1b_${target_ymd}_00.restart"
output_dir="$run_dir/forecast_refined_ghost_78h_2gpu"
topography_dir="$run_dir/topography/products"
raw_topography_dir="$run_dir/topography/raw/pte1b"
session_name="pte1b_${target_ymd}_refined_78h_2gpu"
master_port=$((29000 + 10#${target_ymd:4:4}))
source_merged_tif="$TOPO_SOURCE_RUN/topography/raw/pte1b/pte1b_merged_10m.tif"
source_topography_dir="$TOPO_SOURCE_RUN/topography/products"

echo "Execution date: $requested_ymd"
echo "Forecast date: $target_ymd"
echo "Target date: $target_date"
echo "Forecast source: $forecast_dir"
echo "Run directory: $run_dir"

if [[ ! -f "$source_merged_tif" ]]; then
  echo "Source merged topography file not found: $source_merged_tif" >&2
  exit 1
fi

mkdir -p "$run_dir" "$topography_dir" "$forecast_input_dir" "$raw_topography_dir"

"$PYTHON_BIN" - "$WORKSPACE" "$KML_FILE" "$run_dir/pte1b.yaml" "$target_date" <<'PY'
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

cp "$source_merged_tif" "$raw_topography_dir/pte1b_merged_10m.tif"

if [[ -f "$source_topography_dir/pte1b_topo_2p4km.pt" \
   && -f "$source_topography_dir/pte1b_topo_1p2km.pt" \
   && -f "$source_topography_dir/pte1b_topo_0p6km.pt" \
   && -f "$source_topography_dir/pte1b_topo_0p3km.pt" ]]; then
  cp "$source_topography_dir"/pte1b_topo_*.pt "$topography_dir"/
else
  "$PYTHON_BIN" - "$WORKSPACE" "$KML_FILE" "$raw_topography_dir/pte1b_merged_10m.tif" "$topography_dir" <<'PY'
from pathlib import Path
import sys

workspace = Path(sys.argv[1])
kml_file = Path(sys.argv[2])
merged_tif = Path(sys.argv[3])
topography_dir = Path(sys.argv[4])

sys.path.insert(0, str(workspace))

from um_earth.frigate_pipeline import (
    DEFAULT_TARGET_RESOLUTIONS_KM,
    build_prepared_domain,
    build_resolution_products,
)
from um_earth.regions import load_region

region = load_region(region_kml=kml_file, location_id="pte1b")
prepared = build_prepared_domain(region)
build_resolution_products(
    merged_tif=merged_tif,
    out_dir=topography_dir,
    region_id=region.region_id,
    lat_bounds=(prepared.lat_min, prepared.lat_max),
    lon_bounds=(prepared.lon_min, prepared.lon_max),
    target_resolutions_km=DEFAULT_TARGET_RESOLUTIONS_KM,
)
PY
fi

if [[ ! -f "$restart_bundle" ]]; then
  "$PYTHON_BIN" "$WORKSPACE/prepare_initial_condition.py" \
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

cmd="cd $WORKSPACE && env CUDA_VISIBLE_DEVICES=0,1 PYTHONFAULTHANDLER=1 PATH=$PYENV_ROOT/bin:\$PATH $TORCHRUN_BIN --nproc-per-node=2 --master-port $master_port $WORKSPACE/run_frigate_prediction.py -c $run_dir/pte1b.yaml -i $tensor_dir -o $output_dir --topography-dir $topography_dir --device cuda --refinement-mode staged --forcing-mode ghost --prediction-duration 216000 2>&1 | tee $output_dir/run.log"

tmux new-session -d -s "$session_name" "$cmd"

echo "Started session: $session_name"
echo "Master port: $master_port"
echo "Log: $output_dir/run.log"
