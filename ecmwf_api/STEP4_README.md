# ECMWF Data Pipeline - Step 4: Compute Hydrostatic Pressure

This document describes Step 4 of the ECMWF data fetching and curation pipeline, which computes hydrostatically balanced pressure at cell centers from the regridded data produced in Step 3.

## Overview

Step 4 takes the regridded ERA5 data from Step 3 (which contains pressure at cell interfaces and density at cell centers) and computes a new pressure field at cell centers using hydrostatic balance integration. This new pressure field has better hydrostatic balance properties than the original pressure.

## Prerequisites

Before running Step 4, you must have:

1. **Step 3 completed**: Regridded NetCDF file with variables:
   - `pressure_level`: Pressure at vertical cell interfaces (Pa)
   - `rho`: Density at cell centers (kg/m³)

2. **YAML configuration file** with gravity specification:
   ```yaml
   forcing:
     const-gravity:
       grav1: -9.8  # Negative because gravity points downward
   ```

## Algorithm

The script performs the following steps:

### 1. Read Gravity from Configuration

The gravity value is read from the YAML configuration file under `forcing.const-gravity.grav1`. Note that `grav1` is typically negative (pointing downward), but the script uses the absolute value for calculations.

### 2. Hydrostatic Balance Integration

Starting from the top level (highest altitude) and integrating downward:

```
pf[i-1] = pf[i] + rho[i] * grav * dz[i]
```

where:
- `pf[i]` is pressure at the i-th vertical interface
- `rho[i]` is density at the i-th cell center
- `grav` is the gravitational acceleration (positive value)
- `dz[i]` is the vertical spacing between interfaces

The integration starts with `pf[top]` equal to the original `pressure_level[top]` from Step 3, ensuring consistency at the top boundary.

### 3. Geometric Mean for Cell Centers

Cell center pressure is computed as the geometric mean of the interface pressures:

```
p[i] = sqrt(pf[i] * pf[i+1])
```

where:
- `p[i]` is the pressure at the i-th cell center
- `pf[i]` is the pressure at the interface below cell i
- `pf[i+1]` is the pressure at the interface above cell i

The geometric mean provides better representation of the exponential pressure profile in the atmosphere compared to arithmetic mean.

### 4. Augment NetCDF File

The computed pressure field `p` is added to the NetCDF file as a new variable with:
- Dimensions: `(time, x1, x2, x3)` (at cell centers)
- Units: Pa (Pascals)
- Standard name: `air_pressure`
- Description: Hydrostatically balanced pressure computed from density and interface pressures

## Usage

### Basic Usage

```bash
python compute_hydrostatic_pressure.py config.yaml regridded.nc
```

### Arguments

- `config.yaml`: Path to YAML configuration file containing gravity
- `regridded.nc`: Path to regridded NetCDF file from Step 3

### Example

```bash
# Compute hydrostatic pressure for regridded data
python compute_hydrostatic_pressure.py earth.yaml regridded_era5.nc
```

The script modifies the NetCDF file in-place, adding the new variable `p`.

## Output

The script augments the existing NetCDF file with:

### New Variable: `p`

- **Dimensions**: `(time, x1, x2, x3)` - at cell centers
- **Units**: Pa (Pascals)
- **Data type**: float32
- **Attributes**:
  - `standard_name`: "air_pressure"
  - `long_name`: "Hydrostatically balanced pressure at cell centers"
  - `description`: Details about computation method

### Updated Metadata

- **history**: Appended with timestamp and description of Step 4 processing

## Physical Interpretation

### Why Hydrostatic Balance?

The original pressure from ERA5 may have small inconsistencies with the density field due to:
1. Interpolation errors during regridding
2. Different vertical coordinate systems
3. Temporal evolution between pressure and density fields

By recomputing pressure from hydrostatic balance using the density field, we ensure:
- Consistency between pressure and density
- Better vertical force balance
- More stable initial conditions for atmospheric models

### Why Geometric Mean?

The geometric mean is used because:
1. Pressure decreases exponentially with altitude in the atmosphere
2. For an exponential profile `p(z) = p0 * exp(-z/H)`, the geometric mean of boundary values equals the value at the center
3. It preserves the physical structure better than arithmetic mean

## Comparison with Step 3 Pressure

After running Step 4, the NetCDF file contains two pressure fields:

1. **pressure_level**: Original pressure at vertical interfaces
   - Dimensions: `(time, x1f, x2, x3)`
   - From ERA5 data, regridded to Cartesian grid
   - At cell interfaces (boundaries)

2. **p**: Hydrostatically balanced pressure at cell centers
   - Dimensions: `(time, x1, x2, x3)`
   - Computed from density using hydrostatic balance
   - At cell centers
   - Better consistency with density field

For most atmospheric modeling applications, use the new `p` variable for initial conditions and diagnostics.

## Error Handling

The script will raise errors if:

1. **Configuration errors:**
   - Missing or invalid YAML file
   - Missing `forcing.const-gravity.grav1` field
   - Zero or invalid gravity value

2. **File errors:**
   - NetCDF file not found
   - Missing required variables (`pressure_level`, `rho`)
   - Missing required coordinates (`x1`, `x1f`)

3. **Data errors:**
   - Incompatible array shapes
   - NaN or negative values in pressure/density

## Testing

Run unit tests to verify functionality:

```bash
python -m unittest test_compute_hydrostatic_pressure
```

The test suite includes:
- YAML configuration parsing
- Gravity extraction from config
- NetCDF file loading
- Hydrostatic pressure computation
- NetCDF file augmentation
- Geometric mean verification

## Example Workflow

Complete pipeline from configuration to hydrostatic pressure:

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

# Step 4: Compute hydrostatic pressure
python compute_hydrostatic_pressure.py earth.yaml ./output/regridded_era5.nc
```

## Performance

Step 4 is computationally efficient:
- Processes in memory without intermediate files
- Modifies NetCDF file in-place
- Typical runtime: seconds to minutes depending on grid size
- Memory usage: Proportional to number of cells

For a typical domain (100×200×300 cells):
- Runtime: ~1-5 seconds
- Memory: ~1-2 GB

## Integration with Step 3

### Updated Step 3 Behavior

Step 3 (`regrid_era5_to_cartesian.py`) has been updated to:
1. Read gravity from `forcing.const-gravity.grav1` in YAML config
2. Use the configured gravity for hydrostatic balance calculations
3. Fall back to default Earth gravity (9.80665 m/s²) if not specified
4. Save the gravity value in NetCDF metadata

This ensures consistency between Step 3 and Step 4 when computing height grids and pressure fields.

## References

- `compute_hydrostatic_pressure.py`: Step 4 implementation
- `regrid_era5_to_cartesian.py`: Step 3 implementation (updated)
- `regrid.py`: Core regridding functions
- `earth.yaml`: Example configuration file

## Notes

- The script modifies the NetCDF file in-place, so keep a backup if needed
- The `grav1` value in the YAML is typically negative (downward), but magnitude is used
- The geometric mean ensures exponential pressure profile is preserved
- Cell center pressure `p` should be used for atmospheric model initial conditions
- Interface pressure `pressure_level` can still be used for flux calculations
