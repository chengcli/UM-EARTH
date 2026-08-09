#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/data00/2025.FRIGATE}"
WORKSPACE=/home/chengcli/scix/workspace/UM-EARTH
PYTHON_BIN=/home/chengcli/pyenv/bin/python
REGION_ID=pte1b-extended
KML_FILE="$WORKSPACE/pte1b-extended.kml"
TOPO_RUN="$BASE_DIR/runs/pte1b-extended-topography"
RAW_ROOT="$TOPO_RUN/topography/raw"
PRODUCT_DIR="$TOPO_RUN/topography/products"
MERGED_TIF="$RAW_ROOT/$REGION_ID/${REGION_ID}_merged_10m.tif"

mkdir -p "$RAW_ROOT" "$PRODUCT_DIR"

if [[ ! -f "$MERGED_TIF" ]]; then
  PYTHONPATH="$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$WORKSPACE/um_earth/topography/download_crop.py" \
    "$REGION_ID" --region-kml "$KML_FILE" --out "$RAW_ROOT"
fi

PYTHONPATH="$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" - "$WORKSPACE" "$KML_FILE" "$MERGED_TIF" "$PRODUCT_DIR" "$REGION_ID" <<'PY'
from pathlib import Path
import sys

workspace, kml_file, merged_tif, product_dir, region_id = map(Path, sys.argv[1:])
sys.path.insert(0, str(workspace))

from um_earth.frigate_pipeline import (
    DEFAULT_TARGET_RESOLUTIONS_KM,
    build_prepared_domain,
    build_resolution_products,
)
from um_earth.regions import load_region

region = load_region(region_kml=kml_file, location_id=str(region_id))
prepared = build_prepared_domain(region)
build_resolution_products(
    merged_tif=merged_tif,
    out_dir=product_dir,
    region_id=region.region_id,
    lat_bounds=(prepared.lat_min, prepared.lat_max),
    lon_bounds=(prepared.lon_min, prepared.lon_max),
    target_resolutions_km=DEFAULT_TARGET_RESOLUTIONS_KM,
)
print(f"Prepared bounds: {prepared.lat_min}, {prepared.lat_max}, {prepared.lon_min}, {prepared.lon_max}")
PY
