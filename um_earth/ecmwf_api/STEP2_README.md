# ECMWF Data Pipeline - Step 2: Calculate Air Density

This document describes Step 2 of the ECMWF data fetching and curation pipeline, which calculates the total air density from ERA5 dynamics and density variables downloaded in Step 1.

## Overview

Step 2 takes the ERA5 data files from Step 1 (dynamics and densities) and computes the total air density by solving a system of linear equations that account for:
- Dry air density
- Water vapor density
- Cloud and precipitation density (ice, snow, liquid water, rain)

The computed density is essential for atmospheric modeling as it provides a consistent density field that satisfies the ideal gas law with the given temperature and pressure fields.

## Prerequisites

Before running Step 2, you must have:

1. **Step 1 completed**: ERA5 data files downloaded
   - `era5_hourly_dynamics_YYYYMMDD.nc` (contains temperature and pressure)
   - `era5_hourly_densities_YYYYMMDD.nc` (contains humidity and cloud content)

2. **Python packages installed**:
   ```bash
   pip install numpy netCDF4
   ```

## Physical Theory

The script solves three linear equations to determine the density components:

### Equation 1: Ideal Gas Law

```
rho_d / m_d + rho_v / m_v = P / (Rgas * T)
```

This relates the dry air and water vapor densities to pressure and temperature through the ideal gas law for a mixture.

### Equation 2: Water Vapor Fraction

```
rho_v = q * (rho_d + rho_v + rho_c)
```

This defines water vapor density in terms of specific humidity `q` (mass of water vapor per unit mass of total air).

### Equation 3: Cloud Fraction

```
rho_c = (ciwc + cswc + clwc + crwc) * (rho_d + rho_v + rho_c)
```

This defines cloud density from the sum of cloud ice, snow, liquid water, and rain water content.

### Total Density

Once the three density components are calculated, the total air density is:

```
rho = rho_d + rho_v + rho_c
```

### Physical Constants

- **Rgas** = 8.31446 J/(mol·K) - Ideal gas constant
- **m_d** = 28.96e-3 kg/mol - Molecular weight of dry air
- **m_v** = 18.0e-3 kg/mol - Molecular weight of water vapor

### Variables from ERA5

**From dynamics file:**
- `t`: Temperature (K)
- `level`: Pressure levels (hPa), converted to Pa

**From densities file:**
- `q`: Specific humidity (kg/kg)
- `ciwc`: Specific cloud ice water content (kg/kg)
- `cswc`: Specific snow water content (kg/kg)
- `clwc`: Specific cloud liquid water content (kg/kg)
- `crwc`: Specific rain water content (kg/kg)

## Usage

The script supports two modes of operation:

### Mode 1: Single File Processing

Process a single pair of dynamics and densities files:

```bash
python calculate_density.py \
    --dynamics-file era5_hourly_dynamics_20240101.nc \
    --densities-file era5_hourly_densities_20240101.nc \
    --output era5_density_20240101.nc
```

**Arguments:**
- `--dynamics-file`: Path to dynamics NetCDF file (required for single mode)
- `--densities-file`: Path to densities NetCDF file (required for single mode)
- `--output`: Output NetCDF file path (required for single mode)

### Mode 2: Batch Directory Processing

Process all matching file pairs in a directory:

```bash
python calculate_density.py \
    --input-dir ./32.00N_33.00N_107.00W_106.00W \
    --output-dir ./densities
```

**Arguments:**
- `--input-dir`: Directory containing ERA5 files from Step 1 (required for batch mode)
- `--output-dir`: Directory for output files (required for batch mode)

The script automatically finds all matching pairs:
- `era5_hourly_dynamics_YYYYMMDD.nc`
- `era5_hourly_densities_YYYYMMDD.nc`

And creates output files:
- `era5_density_YYYYMMDD.nc`

**Note**: The output directory will be created if it doesn't exist.

## Examples

### Example 1: White Sands Missile Range - Batch Processing

After running Step 1 for White Sands:

```bash
# Step 1 created directory: 32.00N_33.00N_107.00W_106.00W/
# with files: era5_hourly_dynamics_*.nc and era5_hourly_densities_*.nc

# Step 2: Calculate density for all days
python calculate_density.py \
    --input-dir ./32.00N_33.00N_107.00W_106.00W \
    --output-dir ./32.00N_33.00N_107.00W_106.00W
```

This processes all days in the directory and saves density files alongside the original data.

### Example 2: Ann Arbor - Single Day Processing

Process a single day:

```bash
python calculate_density.py \
    --dynamics-file ./42.00N_43.00N_84.00W_83.00W/era5_hourly_dynamics_20240601.nc \
    --densities-file ./42.00N_43.00N_84.00W_83.00W/era5_hourly_densities_20240601.nc \
    --output ./42.00N_43.00N_84.00W_83.00W/era5_density_20240601.nc
```

### Example 3: Separate Output Directory

Save density files to a different location:

```bash
python calculate_density.py \
    --input-dir ./era5_data/32.00N_33.00N_107.00W_106.00W \
    --output-dir ./processed/densities
```

## Output Format

Each output NetCDF file contains the computed density fields and coordinates from the input files.

### Variables

The output file includes four density variables, all with dimensions `(time, level, latitude, longitude)`:

| Variable | Units | Description |
|----------|-------|-------------|
| `rho` | kg/m³ | Total air density |
| `rho_d` | kg/m³ | Dry air density component |
| `rho_v` | kg/m³ | Water vapor density component |
| `rho_c` | kg/m³ | Cloud density component (ice + snow + liquid + rain) |

### Coordinates

Coordinates are copied from the input files:
- `time`: Time dimension (hours since reference)
- `level`: Pressure levels (hPa)
- `latitude`: Latitude (degrees_north)
- `longitude`: Longitude (degrees_east)

### Metadata

Global attributes include:
- Physical constants used in calculations (`Rgas`, `m_dry_air`, `m_water_vapor`)
- Calculation method description
- Reference to physical theory
- CF-1.8 conventions compliance
- Processing timestamp and history

### Example: Reading Output Data

```python
import netCDF4 as nc

# Open the density file
ds = nc.Dataset('era5_density_20240101.nc', 'r')

# Access variables
rho_total = ds.variables['rho'][:]      # Total density (kg/m³)
rho_dry = ds.variables['rho_d'][:]      # Dry air component
rho_vapor = ds.variables['rho_v'][:]    # Water vapor component
rho_cloud = ds.variables['rho_c'][:]    # Cloud component

# Access coordinates
time = ds.variables['time'][:]
pressure = ds.variables['level'][:]
lat = ds.variables['latitude'][:]
lon = ds.variables['longitude'][:]

print(f"Total density shape: {rho_total.shape}")
print(f"Density range: {rho_total.min():.4f} to {rho_total.max():.4f} kg/m³")

ds.close()
```

## Algorithm Details

### Step-by-Step Process

1. **Load data from input files**
   - Extract temperature and pressure from dynamics file
   - Extract humidity and cloud content from densities file
   - Validate dimensions match between files

2. **Convert pressure to Pascals**
   - ERA5 provides pressure levels in hPa (hectopascals)
   - Convert to Pa for calculations: `P_pa = P_hpa * 100`

3. **Calculate total cloud content**
   - Sum all cloud species: `cloud_content = ciwc + cswc + clwc + crwc`

4. **Solve the linear system**
   - Use the three equations described above
   - Solve algebraically for `rho_d`, `rho_v`, and `rho_c`
   - Handle edge cases (zero humidity, zero cloud content)

5. **Compute total density**
   - Sum the three components: `rho = rho_d + rho_v + rho_c`

6. **Save results to NetCDF**
   - Create output file with CF-1.8 conventions
   - Include all four density variables
   - Copy coordinates and time information
   - Add comprehensive metadata

### Mathematical Solution

The three equations form a linear system that can be solved algebraically:

```python
# Define coefficients
A = P / (Rgas * T)
B = q
C = cloud_content

# Solve for components
denominator = 1 / m_d + B / m_v + C / m_d
rho_d = A * m_d / (1 + B * (m_d/m_v - 1) + C)
rho_v = B * (rho_d + rho_v + rho_c)
rho_c = C * (rho_d + rho_v + rho_c)

# Total density
rho_total = rho_d + rho_v + rho_c
```

The actual implementation uses a more numerically stable formulation.

## Physical Interpretation

### Typical Density Ranges

At standard conditions (sea level, 15°C):
- **Total density**: ~1.225 kg/m³
- **Dry air**: ~1.2 kg/m³ (dominates in most conditions)
- **Water vapor**: ~0.01-0.02 kg/m³ (varies with humidity)
- **Cloud content**: ~0.0001-0.001 kg/m³ (significant only in clouds)

### Altitude Variation

Density decreases exponentially with altitude:
- **Sea level (1000 hPa)**: ~1.2 kg/m³
- **1.5 km (850 hPa)**: ~1.0 kg/m³
- **3 km (700 hPa)**: ~0.9 kg/m³
- **5.5 km (500 hPa)**: ~0.7 kg/m³
- **10 km (300 hPa)**: ~0.4 kg/m³

### Humidity Effects

Water vapor has a lower molecular weight than dry air, so:
- Higher humidity → slightly lower total density
- Humid air is actually less dense than dry air at the same pressure and temperature
- This effect is small but important for accurate atmospheric modeling

## Performance

### Single File Processing

Typical processing times:
- **Small domain** (50×50 grid points): ~1-2 seconds
- **Medium domain** (100×100 grid points): ~3-5 seconds
- **Large domain** (200×200 grid points): ~10-15 seconds

Memory usage scales with grid size: ~50 MB per 100×100×37 grid.

### Batch Processing

For multiple days:
- Each day is processed sequentially
- Progress is reported for each file
- Failed files are reported but don't stop processing
- Total time ≈ (number of days) × (single file time)

### Optimization Tips

- Use batch mode for multiple days (avoids Python startup overhead)
- Keep input and output in the same directory to minimize file I/O
- For very large datasets, process on a machine with adequate RAM (≥4 GB recommended)

## Error Handling

### Common Errors and Solutions

**1. Missing Input Files**
```
Error: Dynamics file not found
```
**Solution**: Verify Step 1 completed successfully and files exist in the specified directory.

**2. File Naming Mismatch**
```
Warning: No matching densities file for dynamics file
```
**Solution**: Ensure both `era5_hourly_dynamics_*.nc` and `era5_hourly_densities_*.nc` files exist for each date.

**3. Dimension Mismatch**
```
Error: Dimension mismatch between dynamics and densities files
```
**Solution**: Re-download the data from Step 1. Both files for a given date should have identical dimensions.

**4. Invalid Values**
```
Warning: Non-finite values detected in output
```
**Solution**: Check input data for NaN or Inf values. May indicate corrupted downloads from Step 1.

**5. Missing Variables**
```
Error: Required variable 't' not found in dynamics file
```
**Solution**: Verify the dynamics file was downloaded correctly with all required variables.

## Validation

### Checking Output Quality

After running Step 2, validate the results:

```python
import netCDF4 as nc
import numpy as np

ds = nc.Dataset('era5_density_20240101.nc', 'r')
rho = ds.variables['rho'][:]

# Check for invalid values
print(f"Any NaN values: {np.any(np.isnan(rho))}")
print(f"Any negative values: {np.any(rho < 0)}")

# Check reasonable range (should be positive, typically 0.1 to 1.5 kg/m³)
print(f"Min density: {np.nanmin(rho):.4f} kg/m³")
print(f"Max density: {np.nanmax(rho):.4f} kg/m³")
print(f"Mean density: {np.nanmean(rho):.4f} kg/m³")

ds.close()
```

Expected results:
- No NaN values
- All positive values
- Reasonable range: 0.1 to 1.5 kg/m³ (depending on altitude range)

## Next Steps

After completing Step 2, proceed to:

**Step 3**: Regrid data to Cartesian coordinates
- See [STEP3_README.md](STEP3_README.md) for instructions
- Script: `regrid_era5_to_cartesian.py`
- This step transforms the pressure-level data to a height-based Cartesian grid

**Step 4**: Compute hydrostatic pressure
- See [STEP4_README.md](STEP4_README.md) for instructions
- Script: `compute_hydrostatic_pressure.py`
- This step ensures hydrostatic balance in the regridded data

## Testing

Run the density calculation tests:

```bash
python test_calculate_density.py
```

The test suite covers:
- Physical constants validation
- Density equation solver correctness
- NetCDF file I/O operations
- Command-line interface
- Error handling for edge cases
- Numerical stability

Run the example script:

```bash
python example_calculate_density.py
```

This demonstrates:
- Single file processing
- Batch directory processing
- Custom processing with intermediate access
- Direct calculation with synthetic data

## Python API

You can also use the calculation functions directly in Python code:

```python
from calculate_density import (
    load_netcdf_data,
    calculate_total_density,
    save_density_netcdf,
    solve_density_equations
)

# Load data from files
data = load_netcdf_data('dynamics.nc', 'densities.nc')

# Calculate density components
rho_total, rho_d, rho_v, rho_c, pressure = calculate_total_density(data)

# Save results
save_density_netcdf('output.nc', data, rho_total, rho_d, rho_v, rho_c)

# Or use the equation solver directly with custom arrays
import numpy as np

temperature = np.array([288., 285., 280.])  # K
pressure_pa = np.array([101325., 95000., 85000.])  # Pa
specific_humidity = np.array([0.01, 0.008, 0.005])  # kg/kg
cloud_content = np.array([0.0001, 0.0002, 0.0001])  # kg/kg

rho_total, rho_d, rho_v, rho_c = solve_density_equations(
    temperature, pressure_pa, specific_humidity, cloud_content
)

print(f"Total density: {rho_total} kg/m³")
print(f"Dry air: {rho_d} kg/m³")
print(f"Water vapor: {rho_v} kg/m³")
print(f"Clouds: {rho_c} kg/m³")
```

## Notes

- The calculation assumes ideal gas behavior, which is valid for Earth's atmosphere
- Cloud density is typically small compared to dry air and water vapor
- The method is numerically stable for typical atmospheric conditions
- Output files use CF-1.8 metadata conventions for compatibility with analysis tools
- All calculations preserve the dimensions and coordinates from input files

## References

- Main README: [README_ECMWF.md](README_ECMWF.md)
- Step 1: [STEP1_README.md](STEP1_README.md)
- Step 3: [STEP3_README.md](STEP3_README.md)
- Wallace and Hobbs (2006), "Atmospheric Science: An Introductory Survey"
- CF Conventions: [https://cfconventions.org/](https://cfconventions.org/)

## Support

For issues with:
- **This script**: Open an issue in this repository
- **Atmospheric physics**: Consult atmospheric science textbooks or literature
- **NetCDF format**: See NetCDF documentation at [https://www.unidata.ucar.edu/software/netcdf/](https://www.unidata.ucar.edu/software/netcdf/)
