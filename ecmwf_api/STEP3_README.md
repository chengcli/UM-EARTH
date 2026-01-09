# ECMWF Data Pipeline - Step 3: Regrid to Cartesian Coordinates

This document describes Step 3 of the ECMWF data fetching and curation pipeline, which regrids ERA5 variables from pressure-lat-lon grids to Cartesian cell-centered coordinates with ghost zones.

## Overview

Step 3 takes the ERA5 data downloaded in Step 1 and the computed densities from Step 2, and regrids all variables to a uniform Cartesian grid suitable for atmospheric modeling with finite volume methods.

## Prerequisites

Before running Step 3, you must have:

1. **Step 1 completed**: ERA5 data files downloaded
   - `era5_hourly_dynamics_YYYYMMDD.nc`
   - `era5_hourly_densities_YYYYMMDD.nc`

2. **Step 2 completed**: Density calculated
   - `era5_density_YYYYMMDD.nc`

3. **YAML configuration file** specifying the Cartesian grid geometry

## YAML Configuration Format

The configuration file must specify a Cartesian grid with ghost zones:

```yaml
geometry:
  type: cartesian
  bounds:
    x1min: 0.0          # meters (bottom of domain, including ghost)
    x1max: 10000.0      # meters (top of domain, including ghost)
    x2min: -20000.0     # meters (south edge, including ghost)
    x2max: 20000.0      # meters (north edge, including ghost)
    x3min: -30000.0     # meters (west edge, including ghost)
    x3max: 30000.0      # meters (east edge, including ghost)
  cells:
    nx1: 100            # number of interior cells in Z direction
    nx2: 200            # number of interior cells in Y direction
    nx3: 300            # number of interior cells in X direction
    nghost: 3           # number of ghost cells on each side
  center_latitude: 32.5    # degrees north (center of domain)
  center_longitude: -106.3 # degrees east (center of domain)
```

### Coordinate System

- **x1**: Z direction (vertical, height in meters, positive upward)
- **x2**: Y direction (north-south, distance in meters, positive northward)
- **x3**: X direction (east-west, distance in meters, positive eastward)

### Ghost Zones

The `bounds` include ghost zones on all sides. For example, with `nx1=100` and `nghost=3`:
- Total cells in x1 direction: 100 + 2×3 = 106
- Interior cells: indices 3 to 102 (inclusive)
- Ghost cells: indices 0-2 (bottom) and 103-105 (top)

The domain limits (`x1min`, `x1max`, etc.) are the boundaries including ghost cells.

## Usage

### Basic Usage

```bash
python regrid_era5_to_cartesian.py config.yaml data_directory/ --output regridded_era5.nc
```

### Arguments

- `config.yaml`: Path to YAML configuration file
- `data_directory/`: Directory containing ERA5 files from steps 1 and 2
- `--output`: Output NetCDF file path (default: `regridded_era5.nc`)
- `--date`: Optional specific date to process (YYYYMMDD format)

### Example

```bash
# Regrid data for a specific date
python regrid_era5_to_cartesian.py earth.yaml ./32.00N_33.00N_107.00W_106.00W/ \
    --output regridded_20240101.nc --date 20240101

# Regrid first available date
python regrid_era5_to_cartesian.py earth.yaml ./32.00N_33.00N_107.00W_106.00W/ \
    --output regridded_era5.nc
```

## Output Format

The output NetCDF file contains:

### Dimensions

- `time`: Number of time steps
- `x1`: Number of cell centers in Z direction (including ghost)
- `x2`: Number of cell centers in Y direction (including ghost)
- `x3`: Number of cell centers in X direction (including ghost)
- `x1f`: Number of cell interfaces in Z direction (x1 + 1)
- `x2f`: Number of cell interfaces in Y direction (x2 + 1)
- `x3f`: Number of cell interfaces in X direction (x3 + 1)

### Coordinates

**Cell Centers** (where variables are defined):
- `time`: Time values (hours since reference)
- `x1(x1)`: Height at cell centers [meters]
- `x2(x2)`: Y coordinate at cell centers [meters]
- `x3(x3)`: X coordinate at cell centers [meters]

**Cell Interfaces** (cell boundaries):
- `x1f(x1f)`: Height at cell interfaces [meters]
- `x2f(x2f)`: Y coordinate at cell interfaces [meters]
- `x3f(x3f)`: X coordinate at cell interfaces [meters]

Note: `len(x1f) = len(x1) + 1`, same for x2f and x3f.

### Variables

Most variables are stored on cell centers with dimensions `(time, x1, x2, x3)`:

**From dynamics file:**
- `u`: U component of wind [m/s]
- `v`: V component of wind [m/s]
- `w`: Vertical velocity [Pa/s]
- `t`: Temperature [K]
- `z`: Geopotential [m²/s²]

**From densities file:**
- `q`: Specific humidity [kg/kg]
- `ciwc`: Specific cloud ice water content [kg/kg]
- `cswc`: Specific snow water content [kg/kg]
- `clwc`: Specific cloud liquid water content [kg/kg]
- `crwc`: Specific rain water content [kg/kg]

**From density file:**
- `rho`: Air density [kg/m³]

**Special variable on vertical interfaces:**
- `pressure_level`: Pressure at cell interfaces [Pa] with dimensions `(time, x1f, x2, x3)`

Note: The `pressure_level` variable is stored at vertical cell interfaces (x1f) rather than centers (x1), making it suitable for flux calculations and boundary conditions in finite volume methods.

### Metadata

Global attributes include:
- `center_latitude`, `center_longitude`: Domain center coordinates
- `nx1_interior`, `nx2_interior`, `nx3_interior`: Interior cell counts
- `nghost`: Number of ghost cells per side
- `planet_radius`, `planet_gravity`: Physical constants used
- `history`: Processing history
- `coordinate_system`: Grid type description

## Algorithm

The regridding process follows these steps:

1. **Parse configuration**: Extract geometry information from YAML file
2. **Compute coordinates**: Calculate cell centers and interfaces including ghost zones
3. **Load data**: Read ERA5 files from steps 1 and 2
4. **Compute heights**: Calculate height at each pressure level using hydrostatic equation
5. **Vertical interpolation**: Interpolate from pressure levels to uniform height grid
6. **Coordinate transformation**: Convert lat-lon to local Cartesian coordinates (Y, X)
7. **Horizontal interpolation**: Interpolate to desired Cartesian grid
8. **Save output**: Write regridded data with metadata to NetCDF

The regridding uses the functions from `regrid.py` which implement:
- Conservative interpolation methods
- Proper handling of topography
- Parallel processing for performance
- Ghost zone inclusion

## Performance

The script uses parallel processing automatically:
- Parallelizes across multiple variables
- Parallelizes vertical and horizontal interpolation
- Automatically determines optimal number of workers

For large datasets, expect:
- ~1-5 minutes for typical domain sizes (100×200×300 cells)
- Longer for finer grids or many time steps
- Memory usage proportional to grid size

## Error Handling

The script will raise errors if:

1. **Configuration errors:**
   - Missing or invalid YAML fields
   - Non-cartesian geometry type
   - Missing center coordinates

2. **File errors:**
   - Data directory not found
   - Missing ERA5 files (dynamics, densities, or density)
   - Cannot read NetCDF files

3. **Domain errors:**
   - Output domain exceeds input domain (extrapolation not allowed by default)
   - Incompatible grid dimensions

## Example Workflow

Complete pipeline from configuration to regridded data:

```bash
# Step 1: Fetch ERA5 data
python fetch_era5_pipeline.py earth.yaml --output-base ./data/

# Step 2: Calculate density
python calculate_density.py --input-dir ./data/32.00N_33.00N_107.00W_106.00W/ \
                           --output-dir ./data/32.00N_33.00N_107.00W_106.00W/

# Step 3: Regrid to Cartesian
python regrid_era5_to_cartesian.py earth.yaml \
                                   ./data/32.00N_33.00N_107.00W_106.00W/ \
                                   --output ./output/regridded_era5.nc
```

## Testing

Run unit tests to verify functionality:

```bash
python test_regrid_era5_to_cartesian.py
```

Run example to see demonstration:

```bash
python example_regrid_era5_to_cartesian.py
```

## References

- `regrid.py`: Core regridding functions
- `fetch_era5_pipeline.py`: Step 1 of the pipeline
- `calculate_density.py`: Step 2 of the pipeline
- ECMWF ERA5 documentation: https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5

## Notes

- The script assumes Earth's radius (6371 km) and gravity (9.80665 m/s²)
- Ghost zones are essential for finite volume methods with stencil operations
- All coordinates follow the right-handed Cartesian system
- Variables are stored at cell centers, suitable for cell-centered finite volume schemes
- Cell interfaces are provided for flux calculations and boundary conditions
