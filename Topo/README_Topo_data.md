# Topography Preprocessing Pipeline (USGS 3DEP → TorchScript)

It is **two-step topography preprocessing pipeline** based on **USGS 3DEP (10 m) DEM data**

## Overview

```text
USGS 3DEP DEM tiles
↓
[ Script 1 ] Download / Clip / Merge / Coarsen
↓
Coarse DEM (e.g. 60×60)
↓
[ Script 2 ] Refine / Interpolate
↓
Refined DEM (2x / 4x / 8x …)
```
## Python Dependencies (Required Before Running)

Before running any scripts, install the required packages:

```bash
pip install numpy rasterio requests torch scipy
```
## Script 1 `Topo_download_crop.py`  

### Download / Clip / Merge / Coarsen DEM

This script performs the **complete preprocessing workflow** for USGS 3DEP DEM data:

1. Query the **USGS TNM Access API** to identify DEM tiles intersecting a target region  
2. Download 10 m (1/3 arc-second) DEM GeoTIFF tiles *(optional skip)*  
3. Clip each tile to the exact bounding box defined in `locations.csv`  
4. Merge all clipped tiles into one seamless GeoTIFF  
5. Coarsen the merged DEM to an `N × N` lat/lon grid (default: **60 × 60**)  
6. Save the result as:
   - GeoTIFF (`*_mean.tif`)
   - **2D TorchScript (`*_mean.pt`)**

The workflow is **restart-safe** and the script can be safely re-run multiple times.
If intermediate or final output files already exist, the corresponding steps will be
**automatically skipped** to avoid redundant computation.

### Input files

1. `locations.csv`: Defines geographic bounding boxes for each location.
2. USGS 3DEP DEM tiles (downloaded automatically)

### Usage

```bash
python Topo_download_crop.py <location-id> [options]
```
Example:

```bash
python Topo_download_crop.py ws-site1
```

### Output structure 

```text
Topo/Data/Raw/<location-id>/
├── clip_*.tif
├── <location-id>_merged_10m.tif
├── <location-id>_topo_60x60_mean.tif
└── <location-id>_topo_60x60_mean.pt
```

### Automatic Skip Logic


### TorchScript format (Script 1)

Saved `.pt` file contains **2D tensors**:

```text
topography : (latitude, longitude)   float32
lat_bounds : (2,)                    float32
lon_bounds : (2,)                    float32
```

## Script 2 `Topo_split_merge.py` 

### DEM Resolution Refinement (Interpolation)

This script refines a coarse DEM (e.g. 60×60) to a higher resolution using interpolation, and saves only TorchScript outputs.

Feature:

1. Supports power-of-two refinement factors: `2x`, `4x`, `8x`, ...
2. Uses `scipy.RegularGridInterpolator`
3. Optional test-window mode
4. Memory safety check before large interpolations

### Input

1. Coarsened DEM GeoTIFF: `<location-id>_topo_60x60_mean.tif`
2. `locations.csv`: Defines geographic bounding boxes for each location.

### Usage

```bash
python Topo_split_merge.py <location-id> <factor>
```

Example:
```bash
python Topo_split_merge.py ws-site1 2x
python Topo_split_merge.py ws-site1 4x

# Test mode (crop small window):
python Topo_split_merge.py ws-site1 8x --test-window 512

# Interpolation method: (default: linear)
python Topo_split_merge.py ws-site1 4x --method cubic
```

### Output
```text
Topo/Data/Split/<location-id>/
└── <location-id>_refined_<factor>x.pt
```

### TorchScript format (Script 2)
Saved .pt file contains **2D tensors**:
```text
topography : (latitude, longitude)   float32
lat_bounds : (2,)                    float32
lon_bounds : (2,)                    float32
```