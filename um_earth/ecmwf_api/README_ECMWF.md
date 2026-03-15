# ECMWF Weather Data API

This directory contains an API for fetching and processing ECMWF ERA5 weather data from the Climate Data Store (CDS).

## Overview

The ECMWF Weather API provides a Python interface to:
- Query and download ERA5 reanalysis weather data
- Specify geographical bounds (latitude/longitude)
- Select atmospheric variables (temperature, wind, humidity, etc.)
- Define time windows and pressure levels
- Process and load the downloaded data

## Complete Data Curation Pipeline

For atmospheric modeling with finite volume methods, use the complete 4-step pipeline:

**Step 1: Fetch ERA5 Data** ([STEP1_README.md](STEP1_README.md))
- Script: `fetch_era5_pipeline.py`
- Downloads ERA5 data based on YAML configuration
- Automatically calculates geographic bounds from Cartesian domain
- Includes ghost zones and buffer regions

**Step 2: Calculate Air Density** ([STEP2_README.md](STEP2_README.md))
- Script: `calculate_density.py`
- Computes total air density from dynamics and densities data
- Solves ideal gas law with moisture and cloud content
- Outputs density components and total density

**Step 3: Regrid to Cartesian Coordinates** ([STEP3_README.md](STEP3_README.md))
- Script: `regrid_era5_to_cartesian.py`
- Transforms pressure-level data to height-based Cartesian grid
- Handles ghost zones for finite volume methods
- Creates cell-centered and cell-interface coordinates

**Step 4: Compute Hydrostatic Pressure** ([STEP4_README.md](STEP4_README.md))
- Script: `compute_hydrostatic_pressure.py`
- Ensures hydrostatic balance in regridded data
- Computes pressure at cell centers from density
- Augments NetCDF file with balanced pressure field

### Example Configurations

Ready-to-use configurations are provided:
- `example_white_sands.yaml`: White Sands Missile Range, New Mexico
- `example_ann_arbor.yaml`: Ann Arbor, Michigan
- `example_pipeline_usage.yaml`: General example with detailed comments

## File Structure

### Core API Files
- `ecmwf_weather_api.py`: Main API class for fetching and loading ERA5 data
- `ecmwf_utils.py`: Utility functions for validation, date handling, and parallel downloads
- `regrid.py`: Functions for regridding pressure-level data to height coordinates

### Convenience Scripts
- `fetch_era5_hourly_dynamics.py`: Download dynamics variables (wind, temperature, etc.)
- `fetch_era5_hourly_densities.py`: Download density variables (humidity, cloud content, etc.)
- `fetch_era5_pipeline.py`: End-to-end pipeline to fetch ERA5 data from YAML configuration (Step 1)
- `calculate_density.py`: Calculate total air density from downloaded data (Step 2)

### Examples
- `example_ecmwf_usage.py`: Demonstrates various API usage patterns
- `example_regrid.py`: Shows how to regrid downloaded data
- `example_calculate_density.py`: Demonstrates air density calculation
- `example_white_sands.yaml`: Complete configuration for White Sands Missile Range, NM
- `example_ann_arbor.yaml`: Complete configuration for Ann Arbor, MI
- `example_pipeline_usage.yaml`: General configuration with detailed comments

### Tests
- `test_ecmwf_weather_api.py`: Tests for main API functionality
- `test_ecmwf_utils.py`: Tests for utility functions
- `test_fetch_scripts.py`: Tests for convenience scripts
- `test_fetch_era5_pipeline.py`: Tests for the YAML configuration pipeline
- `test_regrid.py`: Tests for regridding functions
- `test_calculate_density.py`: Tests for density calculation

### Documentation
- `README_ECMWF.md`: This documentation file (overview and API reference)
- `STEP1_README.md`: Step 1 - Fetch ERA5 data from YAML configuration
- `STEP2_README.md`: Step 2 - Calculate air density from downloaded data
- `STEP3_README.md`: Step 3 - Regrid to Cartesian coordinates
- `STEP4_README.md`: Step 4 - Compute hydrostatic pressure
- `PARALLEL_REGRID.md`: Parallel regridding implementation details

### Configuration
- `requirements.txt`: Python package dependencies

## Installation

### Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The API requires:
- `cdsapi` - Climate Data Store API client
- `xarray` - For loading and processing NetCDF data
- `netCDF4` - NetCDF file format support
- `numpy` - Numerical operations (optional, for data processing)
- `PyYAML` - For parsing YAML configuration files (required for pipeline script)

### CDS API Key Setup

To use this API, you need a free CDS API key:

1. **Register**: Create an account at [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)

2. **Following instructions**: Follow the instructions to set up your API key at [https://cds.climate.copernicus.eu/how-to-api](https://cds.climate.copernicus.eu/how-to-api)

3. **Configure Authentication** (choose one method):

   **Option A: Environment Variable**
   ```bash
   export CDSAPI_KEY="your-uid:your-api-key"
   ```

   **Option B: Configuration File**
   
   Create `~/.cdsapirc` with:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: your-uid:your-api-key
   ```

## Quick Start

### ECMWF/CDS Licenses

ECMWF/CDS requires users to agree to their terms of use for each data product. If some
API requests fail, an error message will prompt you to log in to the CDS website and accept the license.
Look for the URL in the error message to find the relevant license page and accept the terms.

### Basic Usage

```python
from ecmwf_weather_api import ECMWFWeatherAPI

# Initialize the API (uses credentials from environment or ~/.cdsapirc)
api = ECMWFWeatherAPI()

# Fetch weather data
output_file = api.fetch_weather_data(
    latmin=32.0,       # Minimum latitude
    latmax=33.5,       # Maximum latitude
    lonmin=-106.8,     # Minimum longitude
    lonmax=-105.8,     # Maximum longitude
    start_date="2024-01-01",
    end_date="2024-01-02",
    variables=["temperature", "u_component_of_wind", "v_component_of_wind"],
    pressure_levels=[1000, 925, 850, 700, 500]  # hPa
)

print(f"Data saved to: {output_file}")

# Load the downloaded data
data = api.load_data(output_file)
print(f"Variables: {data['variables'].keys()}")
print(f"Coordinates: {data['coordinates'].keys()}")
```

### Alternative: Explicit API Key

```python
# Provide API key directly
api = ECMWFWeatherAPI(api_key="your-uid:your-api-key")
```

### Monitor Download Progress

You can go to the CDS webportal to monitor the status of your data requests:

[https://cds.climate.copernicus.eu/requests](https://cds.climate.copernicus.eu/requests)

## API Reference

### ECMWFWeatherAPI Class

#### Initialization

```python
api = ECMWFWeatherAPI(api_key=None, api_url=None)
```

**Parameters:**
- `api_key` (str, optional): CDS API key. If not provided, reads from `CDSAPI_KEY` environment variable or `~/.cdsapirc`
- `api_url` (str, optional): CDS API URL. Default: `https://cds.climate.copernicus.eu/api/v2`

#### fetch_weather_data()

Fetch ECMWF ERA5 weather data from CDS.

```python
output_file = api.fetch_weather_data(
    latmin, latmax, lonmin, lonmax,
    start_date, end_date,
    variables,
    pressure_levels=None,
    output_file=None,
    times=None,
    product_type='reanalysis',
    format='netcdf'
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `latmin` | float | Yes | Minimum latitude (-90 to 90) |
| `latmax` | float | Yes | Maximum latitude (-90 to 90) |
| `lonmin` | float | Yes | Minimum longitude (-180 to 180) |
| `lonmax` | float | Yes | Maximum longitude (-180 to 180) |
| `start_date` | str | Yes | Start date in format 'YYYY-MM-DD' |
| `end_date` | str | Yes | End date in format 'YYYY-MM-DD' |
| `variables` | list[str] | Yes | List of atmospheric variables to fetch |
| `pressure_levels` | list[int] | No | Pressure levels in hPa. Default: [1000, 925, 850, 700, 500, 300, 200] |
| `output_file` | str | No | Output file path. If None, creates temporary file |
| `times` | list[str] | No | Times in format 'HH:MM'. Default: ['00:00', '06:00', '12:00', '18:00'] |
| `product_type` | str | No | 'reanalysis' or 'ensemble_members'. Default: 'reanalysis' |
| `format` | str | No | 'netcdf' or 'grib'. Default: 'netcdf' |

**Returns:** Path to downloaded file (str)

**Raises:**
- `ValueError`: If input parameters are invalid
- `RuntimeError`: If data retrieval fails

#### load_data()

Load downloaded weather data from file.

```python
data = api.load_data(file_path)
```

**Parameters:**
- `file_path` (str): Path to NetCDF or GRIB file

**Returns:** Dictionary containing:
```python
{
    'variables': {
        'temperature': np.ndarray,  # (time, level, lat, lon)
        'u_component_of_wind': np.ndarray,
        # ... other variables
    },
    'coordinates': {
        'time': np.ndarray,
        'level': np.ndarray,  # pressure levels
        'latitude': np.ndarray,
        'longitude': np.ndarray
    },
    'metadata': dict  # Dataset attributes
}
```

## Available Variables

### Common Variable Names

The API accepts both ECMWF names and common aliases:

| Alias | ECMWF Name | Description |
|-------|------------|-------------|
| `temperature`, `temp` | `temperature` | Air temperature (K) |
| `u`, `u_wind` | `u_component_of_wind` | U-component of wind (m/s) |
| `v`, `v_wind` | `v_component_of_wind` | V-component of wind (m/s) |
| `humidity`, `relative_humidity` | `relative_humidity` | Relative humidity (%) |
| `specific_humidity` | `specific_humidity` | Specific humidity (kg/kg) |
| `geopotential` | `geopotential` | Geopotential (m²/s²) |

For a complete list of available variables, see:
[ERA5 Pressure Levels Documentation](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels)

### Standard Pressure Levels (hPa)

Available pressure levels: 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000

## Examples

### Example 1: Basic Weather Data

```python
from ecmwf_weather_api import ECMWFWeatherAPI

api = ECMWFWeatherAPI()

# Fetch data for White Sands, New Mexico
data_file = api.fetch_weather_data(
    latmin=32.0, latmax=33.5,
    lonmin=-106.8, lonmax=-105.8,
    start_date="2024-01-01",
    end_date="2024-01-02",
    variables=["temperature", "u_wind", "v_wind"],
    pressure_levels=[1000, 850, 500]
)

# Load and inspect
data = api.load_data(data_file)
print(f"Temperature shape: {data['variables']['temperature'].shape}")
```

### Example 2: Specific Times

```python
# Fetch data only at specific times of day
data_file = api.fetch_weather_data(
    latmin=30.0, latmax=35.0,
    lonmin=-110.0, lonmax=-105.0,
    start_date="2024-01-01",
    end_date="2024-01-01",
    variables=["temperature"],
    pressure_levels=[850],
    times=["00:00", "12:00"]  # Only midnight and noon UTC
)
```

### Example 3: High-Resolution Vertical Profile

```python
# Fetch detailed vertical profile
data_file = api.fetch_weather_data(
    latmin=32.0, latmax=33.0,
    lonmin=-106.0, lonmax=-105.0,
    start_date="2024-01-01",
    end_date="2024-01-01",
    variables=["temperature", "geopotential"],
    pressure_levels=[1000, 975, 950, 925, 900, 850, 800, 750, 700, 
                     650, 600, 550, 500, 450, 400, 300, 250, 200]
)
```

### Example 4: Multiple Variables

```python
# Fetch comprehensive atmospheric state
data_file = api.fetch_weather_data(
    latmin=30.0, latmax=40.0,
    lonmin=-110.0, lonmax=-100.0,
    start_date="2024-01-01",
    end_date="2024-01-01",
    variables=[
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "geopotential",
        "relative_humidity",
        "specific_humidity"
    ],
    pressure_levels=[1000, 850, 700, 500, 300]
)
```

## Complete Example Script

See `example_ecmwf_usage.py` for a complete working example with multiple use cases.

Run it with:
```bash
python example_ecmwf_usage.py
```

## Convenience Scripts

Two convenient scripts are provided for common download tasks. Both scripts use **parallel downloads** where each day is downloaded as a separate job, significantly improving download speed. Each day is saved as a separate NetCDF file.

### 1. Download Dynamics Variables (`fetch_era5_hourly_dynamics.py`)

Downloads dynamics-related atmospheric variables at all 37 standard ERA5 pressure levels:
1. Temperature
2. U-component of wind
3. V-component of wind
4. Vertical velocity
5. Divergence
6. Vorticity
7. Potential vorticity
8. Geopotential

**Usage:**
```bash
python fetch_era5_hourly_dynamics.py \
    --latmin 32.0 --latmax 33.5 \
    --lonmin -106.8 --lonmax -105.8 \
    --start-date 2024-01-01 \
    --end-date 2024-01-02 \
    --output ./output_directory
```

**Optional arguments:**
- `--times`: Specify particular times (e.g., `--times 00:00 12:00`)
- `--jobs`: Number of parallel download jobs (default: 4). Each job downloads one day.
- `--api-key`: Provide API key directly
- `--api-url`: Specify custom API URL

**Example with custom parallelism:**
```bash
python fetch_era5_hourly_dynamics.py \
    --latmin 32.0 --latmax 33.5 \
    --lonmin -106.8 --lonmax -105.8 \
    --start-date 2024-01-01 \
    --end-date 2024-01-07 \
    --jobs 7 \
    --output ./output_directory
```

**Output:** Files are named `era5_hourly_dynamics_YYYYMMDD.nc` for each day.

### 2. Download Density Variables (`fetch_era5_hourly_densities.py`)

Downloads density-related variables at all 37 standard ERA5 pressure levels:
1. Specific cloud ice water content
2. Specific humidity
3. Specific snow water content
4. Specific cloud liquid water content
5. Specific rain water content
6. Fraction of cloud cover
7. Relative humidity

**Usage:**
```bash
python fetch_era5_hourly_densities.py \
    --latmin 32.0 --latmax 33.5 \
    --lonmin -106.8 --lonmax -105.8 \
    --start-date 2024-01-01 \
    --end-date 2024-01-02 \
    --output ./output_directory
```

**Optional arguments:**
- `--times`: Specify particular times (e.g., `--times 00:00 12:00`)
- `--jobs`: Number of parallel download jobs (default: 4). Each job downloads one day.
- `--api-key`: Provide API key directly
- `--api-url`: Specify custom API URL

**Output:** Files are named `era5_hourly_densities_YYYYMMDD.nc` for each day.

### Parallel Download Architecture

Both scripts implement a **parallel download strategy**:
- Each day is downloaded as a separate job (with all pressure levels)
- Multiple jobs run in parallel (configurable with `--jobs`, default: 4)
- Each day is saved as a separate NetCDF file in the output directory
- Files are named with the format: `{prefix}_YYYYMMDD.nc`
  - Dynamics: `era5_hourly_dynamics_20240101.nc`
  - Densities: `era5_hourly_densities_20240101.nc`
- Failed downloads are reported but don't prevent other downloads from completing

This approach provides several benefits:
- **Faster downloads**: Multiple days download simultaneously
- **Better resilience**: If one day fails, others continue
- **Progress tracking**: See real-time progress for each day
- **Flexible parallelism**: Adjust the number of parallel jobs based on your network and CDS quota
- **Efficient for date ranges**: Parallelizes along the time dimension, making it ideal for multi-day downloads
- **No post-processing**: Files are ready to use immediately, no combining step needed
- **Clear naming**: Files are clearly identified by date and variable type

**Note:** Both scripts download data at all 37 standard ERA5 pressure levels (1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000 hPa) for each day in the specified date range.

## Utility Functions

The `ecmwf_utils.py` module provides utility functions used by the fetch scripts:

### Validation Functions

- `validate_date_format(date_str)`: Validate date strings in YYYY-MM-DD format
- `validate_region_bounds(latmin, latmax, lonmin, lonmax)`: Validate geographical bounds
- `validate_pressure_levels(pressure_levels)`: Validate and convert pressure levels to strings
- `validate_variable_names(variables)`: Validate ERA5 variable names

### Helper Functions

- `generate_date_list(start_date, end_date)`: Generate list of dates between two dates
- `fetch_single_day(...)`: Download data for a single day (used for parallel downloads)
- `add_common_arguments(parser)`: Add common command-line arguments to argparse parser

### Constants

- `STANDARD_PRESSURE_LEVELS`: All 37 ERA5 pressure levels (1-1000 hPa)
- `STANDARD_TIMES`: Default times ['00:00', '06:00', '12:00', '18:00']
- `STANDARD_VARIABLES`: All available dynamics and density variables

## Testing

Run the test suite:

```bash
python test_ecmwf_weather_api.py       # Tests for main API
python test_ecmwf_utils.py             # Tests for utility functions
python test_fetch_scripts.py           # Tests for fetch scripts
python test_regrid.py                  # Tests for regridding functions
```

The tests include:
- API initialization with various credential configurations
- Input validation (bounds, dates, pressure levels, variables)
- Date range generation and handling
- Variable name validation
- Mock data fetching (no actual API calls)
- Data loading functionality
- Fetch script argument parsing and execution
- Error handling in scripts
- Regridding and interpolation functions

## Error Handling

The API includes comprehensive error handling:

```python
try:
    data_file = api.fetch_weather_data(...)
except ValueError as e:
    # Invalid input parameters
    print(f"Invalid input: {e}")
except RuntimeError as e:
    # API request failed
    print(f"Request failed: {e}")
except ImportError as e:
    # Missing required packages
    print(f"Missing dependency: {e}")
```

## Notes

### Download Times

- Data retrieval from CDS can take several minutes depending on:
  - Size of the requested region
  - Number of variables and pressure levels
  - Time range
  - CDS server load
  
- For large requests, consider:
  - Breaking into smaller time windows
  - Fetching fewer variables per request
  - Using coarser pressure level spacing

### Data Format

- Default output format is NetCDF4 (`.nc`)
- Also supports GRIB format (set `format='grib'`)
- NetCDF is recommended for easier data processing with xarray/numpy

### Coordinate System

- Latitudes: -90 (South Pole) to +90 (North Pole)
- Longitudes: -180 (West) to +180 (East)
- Pressure levels: Surface (1000 hPa) to stratosphere (1 hPa)
- Times: UTC time zone

## Integration with Existing Code

This API is designed to work with the existing weather data processing pipeline in this repository:

```python
from ecmwf_weather_api import ECMWFWeatherAPI
from get_us_weather import interpolate_to_grid
import torch

# 1. Fetch raw ERA5 data
api = ECMWFWeatherAPI()
data_file = api.fetch_weather_data(...)

# 2. Load the data
raw_data = api.load_data(data_file)

# 3. Process with existing pipeline
# (Convert to format expected by interpolate_to_grid)
# ... additional processing code ...

# 4. Use with simulation
# ... integration with white_sand_crm.py or other simulations ...
```

## YAML Configuration Pipeline

The `fetch_era5_pipeline.py` script provides an end-to-end pipeline for fetching ERA5 data based on YAML configuration files. This is particularly useful for atmospheric simulation setups where domain geometry and time period are defined in configuration files.

### Pipeline Features

- **Automatic domain calculation**: Converts Cartesian domain geometry to lat-lon coordinates
- **Ghost zone handling**: Includes ghost cells in the fetch region
- **Buffer zones**: Adds configurable buffer (default 10%) around the domain
- **Integrated fetching**: Automatically fetches both densities and dynamics data
- **Smart directory naming**: Creates output directories with clear geographic identifiers

### Configuration File Format

The pipeline expects a YAML configuration file with the following structure:

```yaml
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 10.e3, x2min: 0., x2max: 20.e3, x3min: 0., x3max: 10.e3}
  cells: {nx1: 100, nx2: 300, nx3: 150, nghost: 3}
  center_latitude: 30.
  center_longitude: -110.

integration:
  start-date: 2024-01-01
  end-date: 2024-01-02
```

**Coordinate system:**
- `x1`: Z direction (vertical, in meters)
- `x2`: Y direction (north-south, in meters)
- `x3`: X direction (east-west, in meters)
- `bounds`: Domain boundaries including ghost zones
- `cells`: Grid cell configuration (nx1, nx2, nx3 are interior cells, nghost is number of ghost cells on each side)
- `center_latitude`, `center_longitude`: Geographic center of the computational domain

### Usage

```bash
python fetch_era5_pipeline.py config.yaml
```

With custom output directory:
```bash
python fetch_era5_pipeline.py config.yaml --output-base ./data
```

### Example Output

For the example configuration above, the pipeline will:
1. Calculate domain extent: ~20 km north-south × ~10 km east-west centered at (30°N, 110°W)
2. Add 10% buffer zone to ensure adequate boundary data
3. Fetch ERA5 densities and dynamics data for the region
4. Save output in directory named: `29.89N_30.11N_110.06W_109.94W`

Each output directory contains:
- `era5_hourly_densities_YYYYMMDD.nc`: Density variables for each day
- `era5_hourly_dynamics_YYYYMMDD.nc`: Dynamics variables for each day

### Pipeline Workflow

```
YAML Config → Parse Geometry → Calculate Lat-Lon → Add Buffer → Fetch Data
                    ↓                  ↓                ↓            ↓
              Extract center    Convert meters    10% expansion   Densities
              and bounds        to degrees        around domain   & Dynamics
```

### Testing

Run the pipeline tests:
```bash
python test_fetch_era5_pipeline.py
```

The test suite covers:
- YAML parsing and validation
- Geometry extraction and conversion
- Lat-lon limit calculations
- Buffer zone addition
- Directory naming conventions
- Error handling

## Data Curation Pipeline - Step 2: Calculate Air Density

After downloading ERA5 data using the pipeline (Step 1), you can calculate the total air density from the dynamics and density variables using `calculate_density.py`.

### Theory

The script solves three linear equations to determine the density components:

1. **Ideal gas law**: `rho_d / m_d + rho_v / m_v = P / (Rgas * T)`
2. **Water vapor fraction**: `rho_v = q * (rho_d + rho_v + rho_c)`
3. **Cloud fraction**: `rho_c = (ciwc + cswc + clwc + crwc) * (rho_d + rho_v + rho_c)`

Where:
- `rho_d`: Dry air density (kg/m³)
- `rho_v`: Water vapor density (kg/m³)
- `rho_c`: Cloud density (kg/m³)
- `P`: Total pressure (Pa)
- `T`: Temperature (K)
- `Rgas`: Ideal gas constant = 8.31446 J/(mol·K)
- `m_d`: Molecular weight of dry air = 28.96e-3 kg/mol
- `m_v`: Molecular weight of water vapor = 18.0e-3 kg/mol
- `q`: Specific humidity (dimensionless)
- `ciwc, cswc, clwc, crwc`: Cloud ice, snow, liquid, and rain water content

The total air density is: `rho = rho_d + rho_v + rho_c`

### Usage

The script supports two modes:

#### Single File Mode

Process a single pair of dynamics and densities files:

```bash
python calculate_density.py \
    --dynamics-file era5_hourly_dynamics_20240101.nc \
    --densities-file era5_hourly_densities_20240101.nc \
    --output era5_density_20240101.nc
```

#### Batch Directory Mode

Process all matching file pairs in a directory:

```bash
python calculate_density.py \
    --input-dir ./29.19N_30.81N_110.93W_109.07W \
    --output-dir ./densities
```

This will automatically find all matching pairs:
- `era5_hourly_dynamics_YYYYMMDD.nc`
- `era5_hourly_densities_YYYYMMDD.nc`

And create output files:
- `era5_density_YYYYMMDD.nc`

### Output Format

The output NetCDF file contains:

**Variables:**
- `rho`: Total air density (kg/m³)
- `rho_d`: Dry air density component (kg/m³)
- `rho_v`: Water vapor density component (kg/m³)
- `rho_c`: Cloud density component (kg/m³)

**Coordinates:**
- `time`: Time dimension
- `level`: Pressure levels (hPa)
- `latitude`: Latitude (degrees_north)
- `longitude`: Longitude (degrees_east)

**Metadata:**
- Physical constants used in calculations
- CF-1.8 conventions compliance
- Calculation method and references

### Complete Pipeline Example

Here's a complete workflow from YAML configuration to density calculation:

```bash
# Step 1: Fetch ERA5 data using YAML configuration
python fetch_era5_pipeline.py earth.yaml

# This creates a directory like: 29.19N_30.81N_110.93W_109.07W/
# with files: era5_hourly_dynamics_*.nc and era5_hourly_densities_*.nc

# Step 2: Calculate air density for all downloaded data
python calculate_density.py \
    --input-dir ./29.19N_30.81N_110.93W_109.07W \
    --output-dir ./densities

# Output directory now contains: era5_density_YYYYMMDD.nc files
```

### Python API

You can also use the calculation functions directly in Python:

```python
from calculate_density import (
    load_netcdf_data,
    calculate_total_density,
    save_density_netcdf,
    solve_density_equations
)

# Load data
data = load_netcdf_data('dynamics.nc', 'densities.nc')

# Calculate density
rho_total, rho_d, rho_v, rho_c, pressure = calculate_total_density(data)

# Save results
save_density_netcdf('output.nc', data, rho_total, rho_d, rho_v, rho_c)

# Or use the equation solver directly with your own arrays
import numpy as np
temperature = np.array([...])  # K
pressure_pa = np.array([...])  # Pa
q = np.array([...])            # specific humidity
cloud_content = np.array([...])  # total cloud content

rho_total, rho_d, rho_v, rho_c = solve_density_equations(
    temperature, pressure_pa, q, cloud_content
)
```

### Examples

Run the example script to see various usage patterns:

```bash
python example_calculate_density.py
```

The example demonstrates:
1. Single file processing
2. Batch directory processing
3. Custom processing with intermediate access
4. Direct calculation with synthetic data

### Testing

Run the density calculation tests:

```bash
python test_calculate_density.py
```

The test suite covers:
- Physical constants validation
- Density equation solver correctness
- Data loading from NetCDF files
- NetCDF output generation
- Command-line interface
- Error handling
- Edge cases and extreme values

### Requirements

The density calculation requires:
- NumPy (numerical operations)
- netCDF4 (file I/O)

Install with:
```bash
pip install numpy netCDF4
```

Or use the main requirements file:
```bash
pip install -r requirements.txt
```

## License

This API interfaces with ECMWF's Copernicus Climate Data Store. Users must comply with the [CDS Terms of Use](https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf).

## References

- [ERA5 Documentation](https://confluence.ecmwf.int/display/CKB/ERA5)
- [CDS API Documentation](https://cds.climate.copernicus.eu/api-how-to)
- [ERA5 Pressure Levels Dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels)

## Support

For issues with:
- **This API**: Open an issue in this repository
- **CDS Service**: Contact ECMWF support at [https://support.ecmwf.int/](https://support.ecmwf.int/)
- **ERA5 Data**: See [ERA5 documentation](https://confluence.ecmwf.int/display/CKB/ERA5)
