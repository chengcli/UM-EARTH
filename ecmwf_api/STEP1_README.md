# ECMWF Data Pipeline - Step 1: Fetch ERA5 Data

This document describes Step 1 of the ECMWF data fetching and curation pipeline, which downloads ERA5 reanalysis data from the Copernicus Climate Data Store (CDS) based on a YAML configuration file.

## Overview

Step 1 reads a YAML configuration file that specifies the computational domain geometry and integration time period, then automatically:
1. Calculates the geographic region (latitude-longitude bounds) that covers the Cartesian domain
2. Includes ghost zones in the calculation
3. Adds a 10% buffer zone around the domain for boundary conditions
4. Fetches ERA5 hourly densities and dynamics data for the specified time period
5. Saves the data in organized NetCDF files by date

## Prerequisites

### 1. Python Packages

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- `cdsapi` - Climate Data Store API client
- `PyYAML` - YAML configuration parsing
- `xarray` - NetCDF data handling
- `netCDF4` - NetCDF file format support

### 2. CDS API Key

You must have a free CDS API key to download ERA5 data:

1. **Register** at [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)
2. **Follow instructions** at [https://cds.climate.copernicus.eu/how-to-api](https://cds.climate.copernicus.eu/how-to-api)
3. **Accept license terms** for ERA5 pressure level data at [https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels)

**Configure your API key** (choose one method):

**Option A: Environment Variable**
```bash
export CDSAPI_KEY="your-uid:your-api-key"
```

**Option B: Configuration File**

Create `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: your-uid:your-api-key
```

## YAML Configuration Format

The pipeline requires a YAML configuration file with the following structure:

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

integration:
  start-date: 2024-01-01   # YYYY-MM-DD format
  end-date: 2024-01-02     # YYYY-MM-DD format
```

### Coordinate System

- **x1**: Z direction (vertical, height in meters, positive upward)
- **x2**: Y direction (north-south, distance in meters, positive northward)
- **x3**: X direction (east-west, distance in meters, positive eastward)

### Domain Specifications

- **bounds**: Domain boundaries **including** ghost zones on all sides
- **cells**: Grid cell configuration where `nx1`, `nx2`, `nx3` are interior (non-ghost) cell counts
- **nghost**: Number of ghost cells on each side of the domain
- **center_latitude**, **center_longitude**: Geographic center of the computational domain

For example, with `nx1=100` and `nghost=3`:
- Total cells in x1 direction: 100 + 2×3 = 106
- Interior cells: indices 3 to 102 (inclusive)
- Ghost cells: indices 0-2 (bottom) and 103-105 (top)

### Minimum Domain Size

The horizontal domain must be **at least 100 km** (approximately 1 degree) in both directions to ensure adequate ERA5 data coverage. Smaller domains may not have sufficient grid points for reliable interpolation.

## Usage

### Basic Usage

```bash
python fetch_era5_pipeline.py config.yaml
```

### Custom Output Directory

```bash
python fetch_era5_pipeline.py config.yaml --output-base ./my_data
```

### Arguments

- `config.yaml`: Path to YAML configuration file (required)
- `--output-base`: Base directory for output (default: current directory)
- `--api-key`: CDS API key (overrides environment/config file)
- `--api-url`: CDS API URL (default: https://cds.climate.copernicus.eu/api/v2)
- `--jobs`: Number of parallel download jobs (default: 4)

## Examples

### Example 1: White Sands Missile Range, NM

```bash
python fetch_era5_pipeline.py example_white_sands.yaml
```

This will:
1. Calculate domain: centered at (32.5°N, 106.3°W), covering ~150×150 km
2. Add 10% buffer for boundary conditions
3. Fetch ERA5 data for January 1-2, 2024
4. Save to directory named with lat-lon bounds (e.g., `32.00N_33.00N_107.00W_106.00W/`)

### Example 2: Ann Arbor, MI

```bash
python fetch_era5_pipeline.py example_ann_arbor.yaml
```

This will:
1. Calculate domain: centered at (42.3°N, 83.7°W), covering ~100×100 km
2. Add 10% buffer for boundary conditions
3. Fetch ERA5 data for the specified time period
4. Save to directory with geographic identifiers

### Example 3: Custom Output Location

```bash
python fetch_era5_pipeline.py earth.yaml --output-base /data/era5/
```

This saves the output directory under `/data/era5/` instead of the current directory.

## Pipeline Workflow

The pipeline executes the following steps automatically:

```
1. Parse YAML Configuration
   └── Extract geometry (bounds, cells, center coordinates)
   └── Extract integration (start-date, end-date)
   └── Validate configuration

2. Calculate Geographic Bounds
   └── Convert Cartesian bounds to lat-lon coordinates
   └── Include ghost zones in calculation
   └── Validate minimum domain size (≥100 km)

3. Add Buffer Zone
   └── Expand domain by 10% in all horizontal directions
   └── Ensure adequate boundary data for interpolation

4. Create Output Directory
   └── Name: {latmin}N_{latmax}N_{lonmin}W_{lonmax}W
   └── Example: 32.00N_33.00N_107.00W_106.00W

5. Fetch ERA5 Densities
   └── Variables: specific humidity, cloud water content, etc.
   └── All 37 standard pressure levels
   └── Parallel download (one file per day)

6. Fetch ERA5 Dynamics
   └── Variables: temperature, wind, geopotential, etc.
   └── All 37 standard pressure levels
   └── Parallel download (one file per day)
```

## Output Files

The pipeline creates an output directory named by the geographic bounds and saves the following files:

```
{latmin}N_{latmax}N_{lonmin}W_{lonmax}W/
├── era5_hourly_densities_20240101.nc
├── era5_hourly_densities_20240102.nc
├── era5_hourly_dynamics_20240101.nc
├── era5_hourly_dynamics_20240102.nc
└── ... (one file pair per day)
```

### File Naming Convention

- **Densities**: `era5_hourly_densities_YYYYMMDD.nc`
- **Dynamics**: `era5_hourly_dynamics_YYYYMMDD.nc`

### Densities File Variables

Each densities file contains at all 37 ERA5 pressure levels:
- `specific_humidity` (q) - kg/kg
- `specific_cloud_ice_water_content` (ciwc) - kg/kg
- `specific_snow_water_content` (cswc) - kg/kg
- `specific_cloud_liquid_water_content` (clwc) - kg/kg
- `specific_rain_water_content` (crwc) - kg/kg
- `fraction_of_cloud_cover` - dimensionless
- `relative_humidity` - %

### Dynamics File Variables

Each dynamics file contains at all 37 ERA5 pressure levels:
- `temperature` (t) - K
- `u_component_of_wind` (u) - m/s
- `v_component_of_wind` (v) - m/s
- `vertical_velocity` (w) - Pa/s
- `divergence` (d) - 1/s
- `vorticity` (vo) - 1/s
- `potential_vorticity` (pv) - K·m²/(kg·s)
- `geopotential` (z) - m²/s²

### Coordinates

All files include the following coordinates:
- `time`: Hours since reference time
- `level`: Pressure levels (hPa) - 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000
- `latitude`: Degrees north
- `longitude`: Degrees east

## Parallel Downloads

The pipeline uses parallel downloads for efficiency:
- Each day is downloaded as a separate job
- Multiple days download simultaneously (default: 4 parallel jobs)
- Configurable with `--jobs` parameter
- Failed downloads don't prevent other downloads from completing

### Adjusting Parallelism

```bash
# Download 7 days with 7 parallel jobs
python fetch_era5_pipeline.py config.yaml --jobs 7
```

**Note**: Be mindful of CDS server load and your API quota. The default of 4 parallel jobs is a good balance.

## Download Times

Data retrieval times depend on:
- **Date range**: More days = longer download time
- **Domain size**: Larger regions = more data = longer time
- **CDS server load**: Peak usage times may be slower
- **Parallel jobs**: More jobs = faster completion (up to a point)

Typical download times:
- **Single day, small domain** (~100×100 km): 5-10 minutes
- **Week of data, medium domain** (~200×200 km): 30-60 minutes
- **Month of data, large domain** (~500×500 km): 2-4 hours

### Monitoring Progress

Monitor your requests at: [https://cds.climate.copernicus.eu/requests](https://cds.climate.copernicus.eu/requests)

You can see:
- Queued requests
- Active downloads
- Completed downloads
- Failed requests

## Error Handling

### Common Errors and Solutions

**1. Missing API Key**
```
Error: CDS API key not found
```
**Solution**: Set up your API key using one of the methods described in Prerequisites.

**2. License Not Accepted**
```
Error: You have not accepted the license for this dataset
```
**Solution**: Visit the ERA5 dataset page and accept the terms of use: [https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels)

**3. Domain Too Small**
```
Error: Domain size must be at least 100 km in both horizontal directions
```
**Solution**: Increase the domain size in your YAML configuration. The horizontal extent (x2 and x3 directions) must each be at least 100 km.

**4. Invalid Date Format**
```
Error: Invalid date format. Expected YYYY-MM-DD
```
**Solution**: Ensure dates in your YAML configuration follow the format `2024-01-01`.

**5. Missing Configuration Fields**
```
Error: Missing required field: geometry.center_latitude
```
**Solution**: Verify your YAML configuration includes all required fields as shown in the format section above.

## Next Steps

After completing Step 1, proceed to:

**Step 2**: Calculate air density from the downloaded data
- See [STEP2_README.md](STEP2_README.md) for instructions
- Script: `calculate_density.py`

**Step 3**: Regrid data to Cartesian coordinates
- See [STEP3_README.md](STEP3_README.md) for instructions
- Script: `regrid_era5_to_cartesian.py`

**Step 4**: Compute hydrostatic pressure
- See [STEP4_README.md](STEP4_README.md) for instructions
- Script: `compute_hydrostatic_pressure.py`

## Testing

Run the pipeline tests:
```bash
python test_fetch_era5_pipeline.py
```

The test suite covers:
- YAML parsing and validation
- Geometry extraction and conversion
- Geographic bound calculations
- Buffer zone addition
- Directory naming
- Error handling

## Example Configurations

See the following example configuration files:
- `example_white_sands.yaml` - White Sands Missile Range, New Mexico
- `example_ann_arbor.yaml` - Ann Arbor, Michigan
- `example_pipeline_usage.yaml` - General example with detailed comments

## Notes

- Ghost zones are automatically included in the geographic region calculation
- The 10% buffer ensures adequate boundary data for interpolation in later steps
- All downloads use UTC time zone
- Data is saved in NetCDF4 format for efficient storage and processing
- Each day is saved as a separate file to facilitate parallel processing and error recovery

## References

- Main README: [README_ECMWF.md](README_ECMWF.md)
- ERA5 Documentation: [https://confluence.ecmwf.int/display/CKB/ERA5](https://confluence.ecmwf.int/display/CKB/ERA5)
- CDS API Documentation: [https://cds.climate.copernicus.eu/api-how-to](https://cds.climate.copernicus.eu/api-how-to)
- ERA5 Pressure Levels Dataset: [https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels)

## Support

For issues with:
- **This pipeline**: Open an issue in this repository
- **CDS Service**: Contact ECMWF support at [https://support.ecmwf.int/](https://support.ecmwf.int/)
- **ERA5 Data**: See [ERA5 documentation](https://confluence.ecmwf.int/display/CKB/ERA5)
