# This script performs the complete preprocessing workflow for USGS 3DEP DEM data:

# 1. Query the USGS TNM Access API to identify all 3DEP DEM tiles intersecting
#    a user-defined geographic region.
# 2. Download the corresponding 10 m (1/3 arc-second) DEM GeoTIFF tiles
#    (unless the download step is explicitly skipped).
# 3. Clip each DEM tile to the exact bounding box defined in locations.csv
#    using window-based raster clipping.
# 4. Merge all clipped tiles into a single, seamless GeoTIFF covering
#    the target region.

# Usage:
#     python Topo_download_crop.py <location-id> [options]

# To skip the download step and reuse existing DEM tiles:
#     python Topo_download_crop.py <location-id> --skip-download

# Notes:
# - Input DEM: USGS 3DEP National Elevation Dataset (NED), 10 m resolution
# - Coordinate Reference System (CRS): EPSG:4269 (geographic coordinates)
# - Units: degrees (horizontal), meters (elevation)
# - Bounding boxes are read from locations.csv
# - DEM tiles are clipped conservatively to guarantee full spatial coverage
# - Requires `seamless-3dep` for downloading USGS 3DEP GeoTIFF files
#   (install with: pip install seamless-3dep)



# 1. download DEM file
#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse
import subprocess
import requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import Window
from rasterio.merge import merge
from rasterio.transform import from_origin
import math
import torch

from um_earth.regions import load_region


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_LOCATIONS = PROJECT_ROOT / "locations.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "topography" / "raw"


def polygon_to_bbox(polygon):
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    return (min(lons), min(lats), max(lons), max(lats))


def find_existing_raw_tifs(save_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in [*save_dir.rglob("*.tif"), *save_dir.rglob("*.tiff")]
        if not path.name.startswith("clip_")
    )


def query_tnm_products(download_bbox) -> list[dict]:
    endpoint = "https://tnmaccess.nationalmap.gov/api/v1/products"
    dataset_names = (
        "National Elevation Dataset (NED) 1/3 arc-second Current",
        "Digital Elevation Model (DEM) 1/3 arc-second",
    )
    params_base = {
        "bbox": f"{download_bbox[0]},{download_bbox[1]},{download_bbox[2]},{download_bbox[3]}",
        "max": 200,
    }

    last_error = None
    for dataset in dataset_names:
        params = dict(params_base)
        params["datasets"] = dataset
        try:
            response = requests.get(endpoint, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            last_error = exc
            continue

        items = response.json().get("items", [])
        if items:
            return items

    if last_error is not None:
        raise last_error
    return []


def download_3dep_image_server_tif(bbox, output_file: Path) -> None:
    lon_min, lat_min, lon_max, lat_max = bbox
    avg_lat = (lat_min + lat_max) / 2.0
    width_m = (lon_max - lon_min) * 111_000.0 * math.cos(math.radians(avg_lat))
    height_m = (lat_max - lat_min) * 111_000.0
    width_px = max(256, min(8000, math.ceil(width_m / 10.0)))
    height_px = max(256, min(8000, math.ceil(height_m / 10.0)))

    params = {
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "bboxSR": 4326,
        "imageSR": 4326,
        "format": "tiff",
        "pixelType": "F32",
        "size": f"{width_px},{height_px}",
        "f": "image",
    }
    endpoint = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"

    print("\n[Fallback] Downloading DEM from USGS 3DEP ImageServer...")
    print(f"Image size: {width_px} x {height_px}")
    response = requests.get(endpoint, params=params, timeout=300)
    response.raise_for_status()
    output_file.write_bytes(response.content)
    print("Saved merged DEM:", output_file.resolve())

def save_tensors(tensor_map: dict[str, torch.Tensor], filename: str):
    class TensorModule(torch.nn.Module):
        def __init__(self, tensors):
            super().__init__()
            for name, tensor in tensors.items():
                self.register_buffer(name, tensor)

    module = TensorModule(tensor_map)
    scripted = torch.jit.script(module)
    scripted.save(filename)


# 1.2 argparse：

def parse_args():
    ap = argparse.ArgumentParser(
        description="Download USGS 3DEP DEM using either a KML region or locations.csv (wget-based)"
    )
    ap.add_argument(
        "location_id",
        nargs="?",
        help="Location ID in locations.csv (e.g., ws-site1)"
    )
    ap.add_argument(
        "--region-kml",
        type=Path,
        help="Path to a KML file describing the forecast region",
    )
    ap.add_argument(
       "--locations",
        type=Path,
        default=DEFAULT_LOCATIONS,
        help="Path to locations.csv (default: UM-EARTH/locations.csv)"
    )
    ap.add_argument(
    "--skip-download",
    action="store_true",
    help="Skip download step and use existing tif files"
    )

    ap.add_argument(
         "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory (default: UM-EARTH/data/topography/raw)"
    
    )

    return ap.parse_args()



# 1.3 BBox → TNM API → wget
def main():
    args = parse_args()
    region = load_region(
        region_kml=args.region_kml,
        location_id=args.location_id,
        locations_file=args.locations,
    )
    bbox = polygon_to_bbox(region.polygon)
    download_bbox = (
        bbox[0], # left
        bbox[1],   # bottom
        bbox[2], # right
        bbox[3], # Top
        )
    print("Locations file:", args.locations.resolve())
    print("Output dir:    ", args.out.resolve())
    print("Location:      ", region.region_id, "-", region.name)
    print("BBox (W,S,E,N):", bbox)

    args.out.mkdir(parents=True, exist_ok=True)


    save_dir = args.out / region.region_id
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Save to location:", save_dir.resolve())

    existing_raw_tifs = find_existing_raw_tifs(save_dir)
    merged_fp = save_dir / f"{region.region_id}_merged_10m.tif"
    have_direct_merged = merged_fp.exists()
    if args.skip_download and existing_raw_tifs:
        print("\n[SKIP] Found existing raw tif files. Reusing them without querying USGS TNM API.")
        items = []
    else:
        print("\nQuerying USGS TNM API...")
        try:
            items = query_tnm_products(download_bbox)
        except requests.RequestException as exc:
            print(f"TNM query failed: {exc}")
            items = []

        if items:
            print(f"Found {len(items)} DEM file(s). Starting download...\n")
        else:
            download_3dep_image_server_tif(bbox, merged_fp)
            have_direct_merged = True


    # Download DEM tiles (optional)

    if args.skip_download:
        print("\n[SKIP] Download step skipped. Using existing tif files.")
    else:
        print(f"\nFound {len(items)} DEM file(s). Starting download...\n")

        for i, item in enumerate(items, 1):
            url = item.get("downloadURL")
            if not url:
                continue

            print("=" * 70)
            print(f"[{i}/{len(items)}] {url}")

            subprocess.run(
                ["wget", "-c", url, "-P", str(save_dir)],
                check=True
        )

    print("\nAll downloads completed.")



    # 2) Clip each DEM tile to the bounding box
    import rasterio
    import math
    from rasterio.windows import Window
    from rasterio.windows import from_bounds
    from rasterio.merge import merge

    print("\nClipping tiles to bbox window...")

    # Search recursively for GeoTIFF files (in case of subdirectories) 

    clipped_tifs = sorted(save_dir.glob("clip_*.tif"))

    if have_direct_merged and merged_fp.exists():
        print("[SKIP] Using directly downloaded merged DEM.")
    elif clipped_tifs:
        print(f"[SKIP] Found {len(clipped_tifs)} existing clipped tiles. Reusing them.")
    else:
        print("[RUN] No clipped tiles found. Performing clipping...")

        # Search recursively for raw GeoTIFF files
        raw_tifs = sorted(list(save_dir.rglob("*.tif")) + list(save_dir.rglob("*.tiff")))
        if not raw_tifs:
            print(f"No tif files found in {save_dir}")
            sys.exit(1)

        clipped_tifs = []

        for fp in raw_tifs:
            if fp.name.startswith("clip_"):
                continue  # 防止重复 clip clip_*.tif

            clip_fp = save_dir / f"clip_{fp.name}"

            with rasterio.open(fp) as src:
                win0 = from_bounds(*bbox, transform=src.transform)

                win = Window(
                    col_off=math.floor(win0.col_off),
                    row_off=math.floor(win0.row_off),
                    width=math.ceil(win0.width),
                    height=math.ceil(win0.height),
                )

                from rasterio.errors import WindowError
                try:
                    win = win.intersection(Window(0, 0, src.width, src.height))
                except WindowError:
                    print(f"  skip (no overlap): {fp.name}")
                    continue

                if win.width <= 0 or win.height <= 0:
                    continue

                data = src.read(1, window=win)
                if data.size == 0:
                    continue

                new_transform = rasterio.windows.transform(win, src.transform)

                profile = src.profile.copy()
                profile.update(
                    height=data.shape[0],
                    width=data.shape[1],
                    transform=new_transform,
                    count=1,
                )

            with rasterio.open(clip_fp, "w", **profile) as dst:
                dst.write(data, 1)

            clipped_tifs.append(clip_fp)
            print(f"  clipped: {clip_fp.name}")

        if not clipped_tifs:
            print("No clipped tiles were produced.")
            sys.exit(1)



    # 3 Merge all clipped tiles into a single GeoTIFF

    # 3) Merge all clipped tiles into a single GeoTIFF
    if merged_fp.exists():
        print("\n[SKIP] Merged DEM already exists. Reusing:", merged_fp.resolve())
    else:
        print("\nMerging clipped tiles...")

        if not clipped_tifs:
            print("ERROR: No clipped tiles found for merging.", file=sys.stderr)
            sys.exit(1)

        datasets = [rasterio.open(str(p)) for p in clipped_tifs]
        mosaic, out_transform = merge(datasets)

        out_profile = datasets[0].profile.copy()
        out_profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_transform,
            count=1,
        )

        for ds in datasets:
            ds.close()

        with rasterio.open(merged_fp, "w", **out_profile) as dst:
            dst.write(mosaic)

        print("\nDONE.")
        print("Merged DEM:", merged_fp.resolve())

        
    # 4 Coarsen merged DEM to an NxN lat/lon grid by computing cell mean
    print("\nComputing coarse-grid cell mean from merged DEM...")

    lon_min, lat_min, lon_max, lat_max = bbox  # bbox = (W,S,E,N)

    nx = ny = 60
    out_tif = save_dir / f"{region.region_id}_topo_{nx}x{ny}_mean.tif"

    if out_tif.exists():
        print("[SKIP] Coarse-mean DEM already exists:")
        print("       ", out_tif.resolve())

        with rasterio.open(out_tif) as ds:
            cell_mean_northup = ds.read(1).astype(np.float32)
            crs = ds.crs
    else:
        print("\nComputing coarse-grid cell mean from merged DEM...")

        lon_min, lat_min, lon_max, lat_max = bbox

        lon_edges = np.linspace(lon_min, lon_max, nx + 1)
        lat_edges = np.linspace(lat_min, lat_max, ny + 1)

        cell_mean = np.full((ny, nx), np.nan, dtype=np.float32)

        with rasterio.open(merged_fp) as ds:
            crs = ds.crs

            for j in range(ny):      # south -> north
                for i in range(nx):
                    w, e = lon_edges[i], lon_edges[i + 1]
                    s, n = lat_edges[j], lat_edges[j + 1]

                    win = from_bounds(w, s, e, n, ds.transform)
                    try:
                        win = win.intersection(Window(0, 0, ds.width, ds.height))
                    except Exception:
                        continue

                    if win.width <= 0 or win.height <= 0:
                        continue

                    data = ds.read(1, window=win, masked=True)
                    if data.size > 0 and not np.all(data.mask):
                        cell_mean[j, i] = float(data.mean())

        # north-up
        cell_mean_northup = np.flipud(cell_mean).copy()

        dx = (lon_max - lon_min) / nx
        dy = (lat_max - lat_min) / ny
        transform = from_origin(lon_min, lat_max, dx, dy)

        profile2 = {
            "driver": "GTiff",
            "height": ny,
            "width": nx,
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "nodata": -9999.0,
        }

        to_write = np.where(
            np.isfinite(cell_mean_northup),
            cell_mean_northup,
            profile2["nodata"]
        ).astype(np.float32)

        with rasterio.open(out_tif, "w", **profile2) as dst:
            dst.write(to_write, 1)

        print("Saved coarse-mean GeoTIFF:", out_tif.resolve())

    

    # 5 Save coarse-mean as TorchScript .pt (north-up)
    # Convention: row 0 = North, latitude is descending (N -> S)
    topo_northup = cell_mean_northup.astype(np.float32)

    lat = np.linspace(lat_max, lat_min, ny).astype(np.float32)  # descending: N -> S
    lon = np.linspace(lon_min, lon_max, nx).astype(np.float32)  # increasing: W -> E

    tensor_map = {
        "topography": torch.from_numpy(topo_northup),            # (ny, nx)
        "lat": torch.from_numpy(lat),                            # (ny,)
        "lon": torch.from_numpy(lon),                            # (nx,)
        "lat_bounds": torch.tensor([lat_min, lat_max], dtype=torch.float32),
        "lon_bounds": torch.tensor([lon_min, lon_max], dtype=torch.float32),
        "row0_is_north": torch.tensor([1], dtype=torch.int8),
    }

    out_pt = save_dir / f"{region.region_id}_topo_{nx}x{ny}_mean.pt"
    print("Writing TorchScript:", out_pt.resolve())
    save_tensors(tensor_map, str(out_pt))

    print("Saved:", out_pt.resolve())
    print("topography shape:", tuple(topo_northup.shape), "dtype:", topo_northup.dtype)
    print("lat first/last:", float(lat[0]), float(lat[-1]))  # should be lat_max -> lat_min
    print("lon first/last:", float(lon[0]), float(lon[-1]))  # should be lon_min -> lon_max



if __name__ == "__main__":
    main()
