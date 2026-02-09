# Diagnostic Plotting Scripts

This directory contains Python scripts for creating diagnostic plots from UM-EARTH model output files (NetCDF format).

## Requirements

Install the required Python packages:

```bash
pip install -r ../ecmwf_api/requirements.txt
```

Key dependencies:
- `xarray` - for reading NetCDF files
- `numpy` - for numerical operations
- `matplotlib` - for plotting

## Output File Format

The scripts expect two types of NetCDF output files:

- **out1.nc**: Contains dynamics variables (pressure, velocities, water content)
- **out2.nc**: Contains thermodynamic variables (temperature, potential temperature, water paths)

## Available Plotting Scripts

### 1. Surface Pressure Map
```bash
python plot_surface_pressure.py <out1.nc> -o surface_pressure.png
```

Creates a contour plot of surface pressure (lowest model level).

### 2. Vertical Velocity Maps
```bash
python plot_vertical_velocity.py <out1.nc> -o vertical_velocity.png
```

Creates contour plots of mean vertical velocity at multiple heights:
- Surface (0 m)
- 1.6 km
- 2 km
- 3 km
- 4 km

### 3. Horizontal Velocity Vector Maps
```bash
python plot_horizontal_velocity.py <out1.nc> -o horizontal_velocity.png
```

Creates vector/quiver plots of horizontal velocity (vel2, vel3) at multiple heights:
- Surface (0 m)
- 1.6 km
- 2 km
- 3 km
- 4 km

Optional arguments:
- `-s, --skip N`: Skip factor for arrows (default: 2) to reduce arrow density

### 4. Water Path Maps
```bash
python plot_water_paths.py <out2.nc> -o water_paths.png
```

Creates contour plots for:
- Water cloud path (liquid water)
- Water vapor path

### 5. Lift Condensation Level (LCL)
```bash
python plot_lcl.py <out1.nc> <out2.nc> -o lcl.png
```

Creates a contour plot of the Lift Condensation Level (LCL) calculated from surface temperature, pressure, and relative humidity.

### 6. Virtual Potential Temperature Maps
```bash
python plot_theta_v.py <out2.nc> -o theta_v.png
```

Creates contour plots of virtual potential temperature at multiple heights:
- Surface (0 m)
- 1.6 km
- 2 km
- 3 km
- 4 km

## Common Options

All scripts support the following options:

- `-o, --output`: Output file path (PNG format). If not specified, displays interactively.
- `-t, --time`: Time index to plot (default: 0)

## Example Usage

Generate all diagnostic plots for a simulation:

```bash
# Surface pressure
python plot_surface_pressure.py out1.nc -o surface_pressure.png

# Vertical velocity at multiple levels
python plot_vertical_velocity.py out1.nc -o vertical_velocity.png

# Horizontal velocity vectors
python plot_horizontal_velocity.py out1.nc -o horizontal_velocity.png -s 3

# Water paths
python plot_water_paths.py out2.nc -o water_paths.png

# Lift Condensation Level
python plot_lcl.py out1.nc out2.nc -o lcl.png

# Virtual potential temperature
python plot_theta_v.py out2.nc -o theta_v.png
```

## Output

All plots are saved as high-resolution PNG files (300 DPI) suitable for publications and presentations.

## Data Structure

The scripts expect the following data structure in the NetCDF files:

**out1.nc variables:**
- `press`: Pressure (Pa) - shape: (time, x1, x3, x2)
- `vel1`: Vertical velocity (m/s) - shape: (time, x1, x3, x2)
- `vel2`: Horizontal velocity component (m/s) - shape: (time, x1, x3, x2)
- `vel3`: Horizontal velocity component (m/s) - shape: (time, x1, x3, x2)

**out2.nc variables:**
- `temp`: Temperature (K) - shape: (time, x1, x3, x2)
- `theta_v`: Virtual potential temperature (K) - shape: (time, x1, x3, x2)
- `rh_H2O_l_`: Relative humidity - shape: (time, x1, x3, x2)
- `path_H2O`: Water vapor path (kg/m²) - shape: (time, x3, x2)
- `path_H2O_l_`: Water cloud path (kg/m²) - shape: (time, x3, x2)

**Coordinates:**
- `time`: Time (s)
- `x1`: Z-coordinate / height (m)
- `x2`: X-coordinate (m)
- `x3`: Y-coordinate (m)

## Notes

- The scripts automatically find the nearest vertical level to the target heights
- All plots include contour lines with labels for easier reading
- Colorbars are included to show the data range
- Grid lines are added for reference
