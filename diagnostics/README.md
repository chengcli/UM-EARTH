# Diagnostic Plotting Scripts

This directory contains Python scripts for creating diagnostic plots from UM-EARTH model output files (NetCDF format).

## New Features

- **Unit Conversions**: Plots now use km for spatial dimensions, cm/s for vertical velocity, m/s for horizontal velocity, and hours for time
- **Topographic Contours**: Optional topography overlays from .pt files
- **Equal Aspect Ratio**: Horizontal dimensions maintain equal aspect ratio
- **PDF Bookmarks**: When processing multiple time steps, plots are grouped by hour with section bookmarks

## Quick Start - Master Script

To generate all diagnostic plots and combine them into a single PDF:

```bash
python generate_all_plots.py <directory_with_netcdf_files>
```

This will:
1. Find all `out1*.nc` and `out2*.nc` files in the directory
2. Generate all diagnostic plots for each file
3. Combine all plots into a single PDF file (`diagnostic_plots.pdf`)

### Examples

```bash
# Process single time index (default: 0)
python generate_all_plots.py ws-site1-2026-02-08/

# Process all time indices with topography overlay
python generate_all_plots.py ws-site1-2026-02-08/ --all-times \
  --topo-dir Topo/Data/Split/ws-site1/ --location ws-site1

# Specify output directory and PDF name
python generate_all_plots.py ws-site1-2026-02-08/ -d output/ -p results.pdf

# Process specific time index and keep PNG files
python generate_all_plots.py ws-site1-2026-02-08/ -t 5 --no-cleanup

# Adjust arrow density for velocity plots
python generate_all_plots.py ws-site1-2026-02-08/ -s 3
```

### Master Script Options

- `-d, --output-dir`: Output directory for plots (default: same as input)
- `-p, --pdf`: Output PDF filename (default: diagnostic_plots.pdf)
- `-t, --time`: Time index to plot (default: 0). Ignored if --all-times is set.
- `-s, --skip`: Skip factor for velocity arrows (default: 2)
- `--no-cleanup`: Keep individual PNG files after creating PDF
- `--topo-dir`: Directory containing topography .pt files
- `--location`: Location prefix for topography files (e.g., ws-site1)
- `--all-times`: Process all time indices and group by hour in PDF

## Topography Files

Topography files should be PyTorch tensors (.pt) with the following structure:

```python
{
    'topography': torch.Tensor,  # 2D array of elevations in meters
    'lat_bounds': torch.Tensor,  # [min_lat, max_lat]
    'lon_bounds': torch.Tensor   # [min_lon, max_lon]
}
```

The script will automatically match topography resolution to the NetCDF grid. Supported resolutions:
- `<location>_0.pt`: 60x60
- `<location>_1.pt`: 120x120
- `<location>_2.pt`: 240x240
- `<location>_3.pt`: 480x480

## Requirements

Install the required Python packages:

```bash
pip install xarray numpy matplotlib pillow torch scipy netCDF4
```

Key dependencies:
- `xarray` - for reading NetCDF files
- `numpy` - for numerical operations
- `matplotlib` - for plotting
- `Pillow` (PIL) - for PDF generation
- `torch` - for loading topography data

## Output File Format

The scripts expect two types of NetCDF output files:

- **out1.nc**: Contains dynamics variables (pressure, velocities, water content)
- **out2.nc**: Contains thermodynamic variables (temperature, potential temperature, water paths)

## Available Plotting Scripts

All plotting scripts now support topography overlays and use improved units (km for spatial dimensions, cm/s for vertical velocity, m/s for horizontal velocity, hours for time).

### 1. Surface Pressure Map
```bash
python plot_surface_pressure.py <out1.nc> -o surface_pressure.png \
  --topo-dir <topo_directory> --location <location_prefix>
```

Creates a contour plot of surface pressure (lowest model level) in hPa.

### 2. Vertical Velocity Maps
```bash
python plot_vertical_velocity.py <out1.nc> -o vertical_velocity.png \
  --topo-dir <topo_directory> --location <location_prefix>
```

Creates contour plots of vertical velocity (in cm/s) at multiple heights:
- Surface (0 km)
- 1.6 km
- 2 km
- 3 km
- 4 km

### 3. Horizontal Velocity Vector Maps
```bash
python plot_horizontal_velocity.py <out1.nc> -o horizontal_velocity.png \
  --topo-dir <topo_directory> --location <location_prefix>
```

Creates vector/quiver plots of horizontal velocity (in m/s) at multiple heights:
- Surface (0 km)
- 1.6 km
- 2 km
- 3 km
- 4 km

Optional arguments:
- `-s, --skip N`: Skip factor for arrows (default: 2) to reduce arrow density

### 4. Water Path Maps
```bash
python plot_water_paths.py <out2.nc> -o water_paths.png \
  --topo-dir <topo_directory> --location <location_prefix>
```

Creates contour plots for:
- Water cloud path (liquid water) in kg/m²
- Water vapor path in kg/m²

### 5. Lift Condensation Level (LCL)
```bash
python plot_lcl.py <out1.nc> <out2.nc> -o lcl.png \
  --topo-dir <topo_directory> --location <location_prefix>
```

Creates a contour plot of the Lift Condensation Level (LCL) in km, calculated from surface temperature, pressure, and relative humidity.

### 6. Virtual Potential Temperature Maps
```bash
python plot_theta_v.py <out2.nc> -o theta_v.png \
  --topo-dir <topo_directory> --location <location_prefix>
```

Creates contour plots of virtual potential temperature (in K) at multiple heights:
- Surface (0 km)
- 1.6 km
- 2 km
- 3 km
- 4 km

## Common Options

All scripts support the following options:

- `-o, --output`: Output file path (PNG format). If not specified, displays interactively.
- `-t, --time`: Time index to plot (default: 0)
- `--topo-dir`: Directory containing topography .pt files (optional)
- `--location`: Location prefix for topography files (optional, e.g., ws-site1)

## Example Usage

Generate all diagnostic plots for a simulation with topography:

```bash
# Surface pressure
python plot_surface_pressure.py out1.nc -o surface_pressure.png \
  --topo-dir Topo/Data/Split/ws-site1 --location ws-site1

# Vertical velocity at multiple levels
python plot_vertical_velocity.py out1.nc -o vertical_velocity.png \
  --topo-dir Topo/Data/Split/ws-site1 --location ws-site1

# Horizontal velocity vectors
python plot_horizontal_velocity.py out1.nc -o horizontal_velocity.png -s 3 \
  --topo-dir Topo/Data/Split/ws-site1 --location ws-site1

# Water paths
python plot_water_paths.py out2.nc -o water_paths.png \
  --topo-dir Topo/Data/Split/ws-site1 --location ws-site1

# Lift Condensation Level
python plot_lcl.py out1.nc out2.nc -o lcl.png \
  --topo-dir Topo/Data/Split/ws-site1 --location ws-site1

# Virtual potential temperature
python plot_theta_v.py out2.nc -o theta_v.png \
  --topo-dir Topo/Data/Split/ws-site1 --location ws-site1
```

## Output

All plots are saved as high-resolution PNG files (300 DPI) suitable for publications and presentations.

## Data Structure

The scripts expect the following data structure in the NetCDF files:

**out1.nc variables:**
- `press`: Pressure (Pa) - shape: (time, x1, x3, x2)
- `vel1`: Vertical velocity (m/s) - shape: (time, x1, x3, x2) - displayed as cm/s
- `vel2`: Horizontal velocity component (m/s) - shape: (time, x1, x3, x2)
- `vel3`: Horizontal velocity component (m/s) - shape: (time, x1, x3, x2)

**out2.nc variables:**
- `temp`: Temperature (K) - shape: (time, x1, x3, x2)
- `theta_v`: Virtual potential temperature (K) - shape: (time, x1, x3, x2)
- `rh_H2O_l_`: Relative humidity - shape: (time, x1, x3, x2)
- `path_H2O`: Water vapor path (kg/m²) - shape: (time, x3, x2)
- `path_H2O_l_`: Water cloud path (kg/m²) - shape: (time, x3, x2)

**Coordinates:**
- `time`: Time (s) - displayed as hours
- `x1`: Z-coordinate / height (m) - displayed as km
- `x2`: X-coordinate (m) - displayed as km
- `x3`: Y-coordinate (m) - displayed as km

## Notes

- The scripts automatically find the nearest vertical level to the target heights
- All plots include contour lines with labels for easier reading
- Colorbars are included to show the data range
- Grid lines are added for reference
- Topography contours (brown lines) are overlaid when topography files are provided
- Equal aspect ratio is maintained for horizontal spatial dimensions
- PDF output groups plots by hour when using --all-times option
