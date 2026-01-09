# Initial condition configuration system

This directory contains a system for configuring atmospheric initial configurations across multiple Earth locations.

## Overview

The initial condition configuration system consists of:

1. **Location Table** (`locations.csv`): Tab-delimited file defining location identifiers, names, and polygon bounds
2. **Configuration Template** (`config_template.yaml`): Template YAML file with placeholders for location-specific values
3. **Configuration Generator** (`generate_config.py`): Generates location-specific YAML files from the template
4. **Download Script** (`prepare_initial_condition.py`): Single script that works with any configured location

## Quick Start

### List Available Locations

```bash
python generate_config.py --list
```

This shows all configured locations with their geographic bounds and calculated center points.

### Generate a Configuration File

Required parameters (horizontal extents calculated automatically from polygon):

```bash
# Generate config for Ann Arbor (x2/x3 extents calculated from polygon)
python generate_config.py ann-arbor \
    --start-date 2025-11-01 --end-date 2025-11-02 \
    --nx1 150 --nx2 200 --nx3 200 \
    --output ann-arbor.yaml

# Generate config for White Sands (x2/x3 extents calculated from polygon)
python generate_config.py white-sands \
    --start-date 2025-10-01 --end-date 2025-10-02 \
    --nx1 150 --nx2 400 --nx3 300 \
    --output white_sands.yaml

# Optional: Override calculated extents if needed
python generate_config.py ann-arbor \
    --start-date 2025-11-01 --end-date 2025-11-02 \
    --nx1 150 --nx2 200 --nx3 200 \
    --x1-max 15000 --tlim 86400 \
    --x2-extent 130000 --x3-extent 95000 \
    --output custom_ann_arbor.yaml
```

### Download and Process Data

```bash
# Download Ann Arbor data (uses ann-arbor.yaml)
python prepare_initial_condition.py ann-arbor

# Download White Sands data (uses white_sands/white_sands.yaml)
python prepare_initial_condition.py white-sands

# Use custom configuration file
python prepare_initial_condition.py ann-arbor --config my_custom_config.yaml

# Run only first 2 steps
python prepare_initial_condition.py white-sands --stop-after 2
```

## Location Table Format

The `locations.csv` file is a comma-delimited file with bounding box information:

```
Name,Description,Latmin,Latmax,Lonmin,Lonmax
ann-arbor,Ann Arbor Michigan,41.75,42.85,-84.25,-83.15
white-sands,White Sands New Mexico,32.1,34.1,-107.2,-105.7
```

Fields:
- **Name**: Unique identifier (letters, numbers, dashes, underscores only)
- **Description**: Human-readable description/name
- **Latmin**: Minimum latitude of bounding box
- **Latmax**: Maximum latitude of bounding box
- **Lonmin**: Minimum longitude of bounding box (negative for West, positive for East)
- **Lonmax**: Maximum longitude of bounding box (negative for West, positive for East)

Notes:
- Center point is calculated automatically from bounding box
- Horizontal domain extents (x2, x3) are calculated automatically from bounding box
- Simulated rectangular domain uses the specified bounding box

### US States Database

A pre-configured location database for all US states is available in `us_states.csv`. This file contains 52 entries:
- 50 US states
- District of Columbia
- Puerto Rico

Each state is defined by its actual boundary polygon (typically 13-50 vertices representing the true state shape). The scripts automatically calculate rectangular simulation bounds from these polygons. You can use these for state-level simulations:

```bash
# List all US states
python generate_config.py --locations-file us_states.csv --list

# Generate config for California
python generate_config.py california \
    --locations-file us_states.csv \
    --start-date 2025-01-01 --end-date 2025-01-02 \
    --nx1 150 --nx2 300 --nx3 300
```

## Configuration Generator Options

The `generate_config.py` script requires most parameters to be explicitly specified:

### Basic Options
- `location_id`: Required location identifier (e.g., `ann-arbor`)
- `--list`: List all available locations
- `--output PATH`: Output file path (default: `<location-id>.yaml`)

### Time Options (REQUIRED)
- `--start-date YYYY-MM-DD`: Simulation start date
- `--end-date YYYY-MM-DD`: Simulation end date
- `--tlim SECONDS`: Simulation time limit

### Grid Resolution Options (REQUIRED)
- `--nx1 INT`: Vertical cells
- `--nx2 INT`: North-south cells
- `--nx3 INT`: East-west cells
- `--nghost INT`: Ghost cells per side (default: 3)

### Domain Size Options
- `--x1-max METERS`: Vertical extent (REQUIRED)
- `--x2-extent METERS`: North-south extent (OPTIONAL - calculated from polygon if not specified)
- `--x3-extent METERS`: East-west extent (OPTIONAL - calculated from polygon if not specified)

**Note**: Horizontal extents (x2, x3) are automatically calculated from the polygon bounds using standard Earth radius approximations (1° latitude ≈ 111 km, 1° longitude ≈ 111 km × cos(latitude)). You can override these by explicitly providing values.

### File Paths
- `--locations-file PATH`: Custom locations table (default: `locations.csv`)
- `--template-file PATH`: Custom template file (default: `config_template.yaml`)

## Download Script Options

The `prepare_initial_condition.py` script accepts:

- `location_id`: Required location identifier
- `--config PATH`: Configuration file (default: searches location subdirectory)
- `--output-base PATH`: Base directory for output (default: current directory)
- `--stop-after N`: Stop after step N (1-4)
- `--timeout SECONDS`: Timeout per step (default: 3600)
- `--locations-file PATH`: Custom locations table

## Adding New Locations

To add a new location:

1. Edit `locations.csv` and add a new tab-delimited line
2. Define the polygon bounds (vertices in counterclockwise order as lon,lat pairs separated by semicolons)
3. Generate a config file with all required parameters
4. Download data: `python prepare_initial_condition.py <new-location-id>`

### Example: Adding a New Location

Add to `locations.csv`:
```
my-location	My Test Site	-100.0,30.0;-100.0,31.0;-99.0,31.0;-99.0,30.0
```

Then generate config (extents calculated automatically):
```bash
python generate_config.py my-location \
    --start-date 2025-01-01 --end-date 2025-01-02 \
    --nx1 150 --nx2 200 --nx3 200 \
    --x1-max 15000 --tlim 86400
```

## Location Identifier Rules

Location identifiers must:
- Contain only letters, numbers, dashes (`-`), and underscores (`_`)
- Be unique across all locations
- Not contain spaces or special characters
- Be used consistently in file names and directory names

Valid examples:
- `ann-arbor`
- `white-sands`
- `test_site_123`
- `location-A1`

Invalid examples:
- `Ann Arbor` (contains space)
- `white.sands` (contains period)
- `site@location` (contains @)

## Simulated Domain vs. Geographic Bounds

- **Geographic bounds (polygon)**: The actual area of interest (can be any shape)
- **Simulated domain (rectangular)**: The computational grid (always rectangular)
- The simulated rectangular domain **must contain** the polygon bounds

For example, if your polygon defines an irregular test area, the rectangular simulation domain will encompass it with some buffer space.

## Pipeline Steps

The download script executes 4 steps:

1. **Fetch ERA5 Data**: Download from ECMWF Climate Data Store
2. **Calculate Air Density**: Compute density from dynamics and moisture
3. **Regrid to Cartesian**: Transform to height-based grid with ghost zones
4. **Compute Hydrostatic Pressure**: Ensure hydrostatic balance

Each step can be run independently or stopped at any point using `--stop-after`.

## Examples

### Generate Custom Configuration

```bash
# High-resolution Ann Arbor simulation
python generate_config.py ann-arbor \
    --nx1 200 --nx2 400 --nx3 400 \
    --output ann_arbor_highres.yaml

# Multi-day White Sands simulation
python generate_config.py white-sands \
    --start-date 2025-10-01 \
    --end-date 2025-10-05 \
    --tlim 432000 \
    --output white_sands_5day.yaml
```

### Download with Custom Settings

```bash
# Download with increased timeout (2 hours per step)
python prepare_initial_condition.py ann-arbor --timeout 7200

# Download to specific directory
python prepare_initial_condition.py white-sands --output-base ./my_data

# Run only data download step
python prepare_initial_condition.py ann-arbor --stop-after 1
```
