#!/usr/bin/env python3
"""
ECMWF data fetching and curation pipeline - Step 3.

This script regrids ERA5 variables on pressure levels to Cartesian coordinate cells.
It reads a YAML configuration file specifying the Cartesian grid geometry (including
ghost zones), loads ERA5 data from steps 1 and 2, and interpolates all variables
to cell-centered locations on the Cartesian grid.

The script performs the following steps:
1. Parse YAML configuration to extract geometry information
2. Calculate cell centers and interfaces (including ghost zones)
3. Load ERA5 data files (dynamics, densities, and computed density)
4. Use regrid.py functions to interpolate variables to Cartesian grid
5. Save regridded data to NetCDF with metadata and coordinates

Usage:
    python regrid_era5_to_cartesian.py <config.yaml> <data_dir> [--output output.nc]
    
    python regrid_era5_to_cartesian.py earth.yaml ./data_folder --output regridded_era5.nc

Requirements:
    - PyYAML for YAML parsing
    - xarray, netCDF4 for NetCDF I/O
    - numpy, scipy for numerical operations
    - All dependencies from regrid.py
"""

import argparse
import sys
import os
import yaml
import glob
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import warnings

# Add current directory to path for importing local modules
sys.path.insert(0, os.path.dirname(__file__))

from regrid import (
    compute_height_grid,
    regrid_multiple_variables,
    regrid_pressure_to_height,
    save_regridded_data_to_netcdf,
)


# Physical constants
EARTH_RADIUS = 6371.0e3  # Earth radius in meters
EARTH_GRAVITY_DEFAULT = 9.80665  # Default Earth gravity in m/s^2 (used if not in config)


def parse_yaml_config(yaml_file: str) -> Dict:
    """
    Parse YAML configuration file.
    
    Args:
        yaml_file: Path to YAML configuration file
        
    Returns:
        Dictionary containing parsed YAML content
        
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Configuration file not found: {yaml_file}")
    
    with open(yaml_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file: {e}")
    
    return config


def extract_gravity(config: Dict) -> float:
    """
    Extract gravity value from configuration.
    
    Args:
        config: Parsed YAML configuration dictionary
        
    Returns:
        Gravity value (positive, in m/s^2). Returns default if not found.
    """
    try:
        if 'forcing' in config:
            forcing = config['forcing']
            if 'const-gravity' in forcing:
                const_gravity = forcing['const-gravity']
                if 'grav1' in const_gravity:
                    grav1 = float(const_gravity['grav1'])
                    # grav1 is typically negative (pointing downward), convert to positive
                    gravity = abs(grav1)
                    # Validate non-zero
                    if gravity > 0:
                        return gravity
    except (KeyError, TypeError, ValueError):
        pass
    
    # Return default if not found or parsing fails
    print(f"   Warning: Could not read gravity from config, using default: {EARTH_GRAVITY_DEFAULT} m/s²")
    return EARTH_GRAVITY_DEFAULT


def extract_geometry_info(config: Dict) -> Dict:
    """
    Extract geometry information from configuration.
    
    Args:
        config: Parsed YAML configuration dictionary
        
    Returns:
        Dictionary containing geometry information with:
            - bounds: Dict with x1min, x1max, x2min, x2max, x3min, x3max
            - cells: Dict with nx1, nx2, nx3, nghost
            - center_latitude: float
            - center_longitude: float
        
    Raises:
        ValueError: If required geometry fields are missing or invalid
    """
    if 'geometry' not in config:
        raise ValueError("Configuration must contain 'geometry' field")
    
    geometry = config['geometry']
    
    # Validate geometry type
    if geometry.get('type') != 'cartesian':
        raise ValueError(f"Only 'cartesian' geometry type is supported, got: {geometry.get('type')}")
    
    # Extract bounds
    if 'bounds' not in geometry:
        raise ValueError("Geometry must contain 'bounds' field")
    
    bounds = geometry['bounds']
    required_bounds = ['x1min', 'x1max', 'x2min', 'x2max', 'x3min', 'x3max']
    for bound in required_bounds:
        if bound not in bounds:
            raise ValueError(f"Bounds must contain '{bound}' field")
    
    # Extract cells
    if 'cells' not in geometry:
        raise ValueError("Geometry must contain 'cells' field")
    
    cells = geometry['cells']
    required_cells = ['nx1', 'nx2', 'nx3', 'nghost']
    for cell in required_cells:
        if cell not in cells:
            raise ValueError(f"Cells must contain '{cell}' field")
    
    # Extract center coordinates
    if 'center_latitude' not in geometry:
        raise ValueError("Geometry must contain 'center_latitude' field")
    if 'center_longitude' not in geometry:
        raise ValueError("Geometry must contain 'center_longitude' field")
    
    # Convert bounds to float
    bounds_float = {
        'x1min': float(bounds['x1min']),
        'x1max': float(bounds['x1max']),
        'x2min': float(bounds['x2min']),
        'x2max': float(bounds['x2max']),
        'x3min': float(bounds['x3min']),
        'x3max': float(bounds['x3max'])
    }
    
    # Convert cells to int
    cells_int = {
        'nx1': int(cells['nx1']),
        'nx2': int(cells['nx2']),
        'nx3': int(cells['nx3']),
        'nghost': int(cells['nghost'])
    }
    
    return {
        'bounds': bounds_float,
        'cells': cells_int,
        'center_latitude': float(geometry['center_latitude']),
        'center_longitude': float(geometry['center_longitude'])
    }


def compute_cell_coordinates(geometry: Dict) -> Tuple:
    """
    Compute cell center and interface coordinates including ghost zones.
    
    The domain bounds include ghost zones. We compute:
    - Cell interfaces (x1f, x2f, x3f) at the boundaries between cells
    - Cell centers (x1, x2, x3) at the midpoints of cells
    
    Args:
        geometry: Dictionary containing geometry information
        
    Returns:
        Tuple of (x1, x1f, x2, x2f, x3, x3f) where:
            x1, x2, x3: Cell center coordinates (1D arrays)
            x1f, x2f, x3f: Cell interface coordinates (1D arrays, length + 1)
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("NumPy is required. Install with: pip install numpy")
    
    bounds = geometry['bounds']
    cells = geometry['cells']
    
    # Extract bounds
    x1min, x1max = bounds['x1min'], bounds['x1max']
    x2min, x2max = bounds['x2min'], bounds['x2max']
    x3min, x3max = bounds['x3min'], bounds['x3max']
    
    # Extract cell counts (interior + ghost)
    nx1 = cells['nx1']
    nx2 = cells['nx2']
    nx3 = cells['nx3']
    nghost = cells['nghost']
    
    # Total number of cells including ghost zones
    nx1_total = nx1 + 2 * nghost
    nx2_total = nx2 + 2 * nghost
    nx3_total = nx3 + 2 * nghost
    
    # Cell interfaces (boundaries) - uniform spacing
    # len(x1f) = nx1_total + 1
    x1f = np.linspace(x1min, x1max, nx1_total + 1)
    x2f = np.linspace(x2min, x2max, nx2_total + 1)
    x3f = np.linspace(x3min, x3max, nx3_total + 1)
    
    # Cell centers (midpoints)
    # len(x1) = nx1_total
    x1 = 0.5 * (x1f[:-1] + x1f[1:])
    x2 = 0.5 * (x2f[:-1] + x2f[1:])
    x3 = 0.5 * (x3f[:-1] + x3f[1:])
    
    return x1, x1f, x2, x2f, x3, x3f


def find_era5_files(data_dir: str, date_str: Optional[str] = None) -> Dict[str, str]:
    """
    Find ERA5 data files in the directory.
    
    Looks for:
        - era5_hourly_dynamics_YYYYMMDD.nc
        - era5_hourly_densities_YYYYMMDD.nc
        - era5_density_YYYYMMDD.nc
    
    Args:
        data_dir: Directory containing ERA5 data files
        date_str: Optional specific date string (YYYYMMDD). If None, finds all dates.
        
    Returns:
        Dictionary mapping date strings to file paths:
            {
                'YYYYMMDD': {
                    'dynamics': '/path/to/era5_hourly_dynamics_YYYYMMDD.nc',
                    'densities': '/path/to/era5_hourly_densities_YYYYMMDD.nc',
                    'density': '/path/to/era5_density_YYYYMMDD.nc'
                }
            }
        
    Raises:
        FileNotFoundError: If no matching files are found or if required files are missing
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find dynamics files
    if date_str:
        dynamics_pattern = os.path.join(data_dir, f'era5_hourly_dynamics_{date_str}.nc')
        dynamics_files = glob.glob(dynamics_pattern)
    else:
        dynamics_pattern = os.path.join(data_dir, 'era5_hourly_dynamics_*.nc')
        dynamics_files = sorted(glob.glob(dynamics_pattern))
    
    if not dynamics_files:
        raise FileNotFoundError(
            f"No ERA5 dynamics files found in {data_dir}. "
            f"Expected files matching: era5_hourly_dynamics_YYYYMMDD.nc"
        )
    
    # Build file dictionary
    file_dict = {}
    missing_files = []
    
    for dynamics_file in dynamics_files:
        # Extract date from filename
        basename = os.path.basename(dynamics_file)
        date_str_found = basename.replace('era5_hourly_dynamics_', '').replace('.nc', '')
        
        # Find corresponding files
        densities_file = os.path.join(data_dir, f'era5_hourly_densities_{date_str_found}.nc')
        density_file = os.path.join(data_dir, f'era5_density_{date_str_found}.nc')
        
        # Check if all files exist
        files_found = {
            'dynamics': dynamics_file,
            'densities': densities_file if os.path.exists(densities_file) else None,
            'density': density_file if os.path.exists(density_file) else None
        }
        
        if files_found['densities'] is None:
            missing_files.append(f"era5_hourly_densities_{date_str_found}.nc")
        if files_found['density'] is None:
            missing_files.append(f"era5_density_{date_str_found}.nc")
        
        if files_found['densities'] and files_found['density']:
            file_dict[date_str_found] = files_found
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required ERA5 files in {data_dir}:\n" +
            "\n".join(f"  - {f}" for f in missing_files) +
            "\n\nPlease run steps 1 and 2 of the pipeline first."
        )
    
    if not file_dict:
        raise FileNotFoundError(
            f"No complete set of ERA5 files found in {data_dir}"
        )
    
    return file_dict


def load_era5_data(file_paths: Dict[str, str]) -> Tuple:
    """
    Load ERA5 data from NetCDF files.
    
    Args:
        file_paths: Dictionary with keys 'dynamics', 'densities', 'density'
                   containing paths to NetCDF files
        
    Returns:
        Tuple of (variables_dict, coords_dict, metadata_dict) where:
            variables_dict: Dict mapping variable names to (T, P, Lat, Lon) arrays
            coords_dict: Dict with 'time', 'plev', 'lats', 'lons' arrays
            metadata_dict: Dict with metadata attributes
    """
    try:
        import xarray as xr
        import numpy as np
    except ImportError:
        raise ImportError(
            "xarray is required for loading NetCDF files. "
            "Install with: pip install xarray netCDF4"
        )
    
    print("\nLoading ERA5 data files...")
    
    # Load dynamics file
    print(f"  Loading dynamics: {os.path.basename(file_paths['dynamics'])}")
    ds_dynamics = xr.open_dataset(file_paths['dynamics'])
    
    # Load densities file
    print(f"  Loading densities: {os.path.basename(file_paths['densities'])}")
    ds_densities = xr.open_dataset(file_paths['densities'])
    
    # Load computed density file
    print(f"  Loading density: {os.path.basename(file_paths['density'])}")
    ds_density = xr.open_dataset(file_paths['density'])
    
    # Extract coordinates
    # ERA5 uses 'time', 'pressure_level' (pressure in Pa or hPa), 'latitude', 'longitude'
    # Different files might use slightly different naming conventions
    
    # Time coordinate
    time_var = 'time' if 'time' in ds_dynamics.coords else 'valid_time'
    time = ds_dynamics[time_var].values
    
    # Pressure levels - might be 'pressure_level', 'plev', or 'pressure'
    if 'pressure_level' in ds_dynamics.coords:
        plev = ds_dynamics['pressure_level'].values
    elif 'plev' in ds_dynamics.coords:
        plev = ds_dynamics['plev'].values
    elif 'pressure' in ds_dynamics.coords:
        plev = ds_dynamics['pressure'].values
    else:
        raise ValueError("Cannot find pressure level coordinate in dynamics file")
    
    # Convert pressure to Pa if in hPa
    if np.max(plev) < 2000:  # Likely in hPa
        plev = plev * 100.0  # Convert to Pa
    
    # Latitude and longitude
    if 'latitude' in ds_dynamics.coords:
        lats = ds_dynamics['latitude'].values
        lons = ds_dynamics['longitude'].values
    elif 'lat' in ds_dynamics.coords:
        lats = ds_dynamics['lat'].values
        lons = ds_dynamics['lon'].values
    else:
        raise ValueError("Cannot find latitude/longitude coordinates in dynamics file")
    
    # Collect all variables
    variables = {}
    
    # From dynamics file - typical variables: u, v, w, t (temperature), z (geopotential)
    dynamics_vars = ['u', 'v', 'w', 't', 'z']
    for var in dynamics_vars:
        if var in ds_dynamics.data_vars:
            variables[var] = ds_dynamics[var].values
            print(f"  Found dynamics variable: {var}, shape: {ds_dynamics[var].shape}")
    
    # From densities file - typical variables: q (specific humidity), cloud water contents
    densities_vars = ['q', 'ciwc', 'cswc', 'clwc', 'crwc']
    for var in densities_vars:
        if var in ds_densities.data_vars:
            variables[var] = ds_densities[var].values
            print(f"  Found densities variable: {var}, shape: {ds_densities[var].shape}")
    
    # From density file - computed total density
    if 'rho' in ds_density.data_vars:
        variables['rho'] = ds_density['rho'].values
        print(f"  Found computed density: rho, shape: {ds_density['rho'].shape}")
    elif 'density' in ds_density.data_vars:
        variables['rho'] = ds_density['density'].values
        print(f"  Found computed density: density, shape: {ds_density['density'].shape}")
    else:
        raise ValueError("Cannot find density variable in density file")
    
    coords_dict = {
        'time': time,
        'plev': plev,
        'lats': lats,
        'lons': lons
    }
    
    # Extract metadata
    metadata = {
        'source': 'ECMWF ERA5 Reanalysis',
    }
    
    # Try to extract global attributes
    for attr in ['source', 'institution', 'history']:
        if attr in ds_dynamics.attrs:
            metadata[attr] = ds_dynamics.attrs[attr]
    
    # Close datasets
    ds_dynamics.close()
    ds_densities.close()
    ds_density.close()
    
    print(f"\nLoaded {len(variables)} variables")
    print(f"Time steps: {len(time)}")
    print(f"Pressure levels: {len(plev)}")
    print(f"Lat x Lon: {len(lats)} x {len(lons)}")
    
    return variables, coords_dict, metadata


def load_topography(data_dir: str, lats, lons) -> 'np.ndarray':
    """
    Load topography data. If not available, return zeros.
    
    Args:
        data_dir: Directory containing data files
        lats: Latitude array
        lons: Longitude array
        
    Returns:
        Topography array of shape (Lat, Lon)
    """
    try:
        import xarray as xr
        import numpy as np
    except ImportError:
        raise ImportError("xarray and numpy are required")
    
    # Look for topography file
    topo_files = glob.glob(os.path.join(data_dir, '*topo*.nc')) + \
                 glob.glob(os.path.join(data_dir, '*geopotential*.nc')) + \
                 glob.glob(os.path.join(data_dir, '*elevation*.nc'))
    
    if topo_files:
        print(f"\nLoading topography from: {os.path.basename(topo_files[0])}")
        ds_topo = xr.open_dataset(topo_files[0])
        
        # Try to find topography variable
        for var_name in ['z', 'geopotential', 'topography', 'elevation', 'orog']:
            if var_name in ds_topo.data_vars:
                topo = ds_topo[var_name].values
                if topo.ndim == 2:
                    # Already (lat, lon)
                    if topo.shape == (len(lats), len(lons)):
                        ds_topo.close()
                        return topo
                elif topo.ndim == 3 and topo.shape[0] == 1:
                    # (1, lat, lon)
                    ds_topo.close()
                    return topo[0, :, :]
        
        ds_topo.close()
    
    print("\nTopography not found, using flat surface (elevation = 0)")
    return np.zeros((len(lats), len(lons)))


def regrid_era5_to_cartesian(
    config_file: str,
    data_dir: str,
    output_file: str,
    date_str: Optional[str] = None
) -> None:
    """
    Main function to regrid ERA5 data to Cartesian coordinates.
    
    Args:
        config_file: Path to YAML configuration file
        data_dir: Directory containing ERA5 data files
        output_file: Path to output NetCDF file
        date_str: Optional specific date (YYYYMMDD) to process
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("NumPy is required")
    
    print("="*70)
    print("ECMWF Data Curation - Step 3: Regrid to Cartesian Coordinates")
    print("="*70)
    
    # Step 1: Parse configuration
    print(f"\n1. Reading configuration from: {config_file}")
    config = parse_yaml_config(config_file)
    geometry = extract_geometry_info(config)
    gravity = extract_gravity(config)
    
    print(f"   Center: ({geometry['center_latitude']:.4f}°, {geometry['center_longitude']:.4f}°)")
    print(f"   Gravity: {gravity:.6f} m/s²")
    print(f"   Domain bounds (with ghost zones):")
    print(f"     x1 (Z): [{geometry['bounds']['x1min']:.1f}, {geometry['bounds']['x1max']:.1f}] m")
    print(f"     x2 (Y): [{geometry['bounds']['x2min']:.1f}, {geometry['bounds']['x2max']:.1f}] m")
    print(f"     x3 (X): [{geometry['bounds']['x3min']:.1f}, {geometry['bounds']['x3max']:.1f}] m")
    print(f"   Grid cells: nx1={geometry['cells']['nx1']}, nx2={geometry['cells']['nx2']}, "
          f"nx3={geometry['cells']['nx3']}, nghost={geometry['cells']['nghost']}")
    
    # Step 2: Compute cell coordinates
    print("\n2. Computing cell coordinates (including ghost zones)...")
    x1, x1f, x2, x2f, x3, x3f = compute_cell_coordinates(geometry)
    
    print(f"   Cell centers: x1({len(x1)}), x2({len(x2)}), x3({len(x3)})")
    print(f"   Cell interfaces: x1f({len(x1f)}), x2f({len(x2f)}), x3f({len(x3f)})")
    print(f"   x1 range: [{x1[0]:.1f}, {x1[-1]:.1f}] m")
    print(f"   x2 range: [{x2[0]:.1f}, {x2[-1]:.1f}] m")
    print(f"   x3 range: [{x3[0]:.1f}, {x3[-1]:.1f}] m")
    
    # Step 3: Find ERA5 files
    print(f"\n3. Finding ERA5 data files in: {data_dir}")
    file_dict = find_era5_files(data_dir, date_str)
    
    print(f"   Found data for {len(file_dict)} date(s):")
    for date in sorted(file_dict.keys()):
        print(f"     {date}")
    
    # For now, process the first date (or specified date)
    # In future, could extend to process multiple dates
    if date_str and date_str in file_dict:
        process_date = date_str
    else:
        process_date = sorted(file_dict.keys())[0]
    
    print(f"\n4. Processing date: {process_date}")
    file_paths = file_dict[process_date]
    
    # Step 4: Load ERA5 data
    variables, coords, metadata = load_era5_data(file_paths)
    
    time = coords['time']
    plev = coords['plev']
    lats = coords['lats']
    lons = coords['lons']
    
    # Step 5: Load topography
    topo_ll = load_topography(data_dir, lats, lons)
    
    # Step 6: Extract density for height computation
    print("\n5. Computing height grid...")
    rho_tpll = variables['rho']
    
    # Compute height grid once for all variables
    z_tpll = compute_height_grid(rho_tpll, topo_ll, plev, gravity)
    print(f"   Height grid computed: shape {z_tpll.shape}")
    print(f"   Height range: [{np.min(z_tpll):.1f}, {np.max(z_tpll):.1f}] m")
    
    # Step 7: Regrid all variables
    print("\n6. Regridding variables to Cartesian grid...")
    print("   This may take a while for large datasets...")
    
    # Use cell centers for regridding (not interfaces)
    # But we'll save both centers and interfaces in the output
    regridded_vars = regrid_multiple_variables(
        variables,
        rho_tpll,
        topo_ll,
        plev,
        lats,
        lons,
        x1,  # Use cell centers
        x2,
        x3,
        gravity,
        EARTH_RADIUS,
        bounds_error=False,  # Allow NaNs outside domain
        z_tpll=z_tpll,
        n_jobs=-1  # Use all CPUs
    )
    
    print(f"   Regridded {len(regridded_vars)} variables")
    for var_name, var_data in regridded_vars.items():
        print(f"     {var_name}: shape {var_data.shape}, "
              f"range [{np.nanmin(var_data):.3e}, {np.nanmax(var_data):.3e}]")
    
    # Step 7b: Regrid pressure to cell interfaces
    print("\n6b. Regridding pressure to vertical cell interfaces...")
    
    # Broadcast 1D pressure to 4D (T, P, Lat, Lon)
    T, P, Lat, Lon = rho_tpll.shape
    pressure_tpll = np.broadcast_to(plev[np.newaxis, :, np.newaxis, np.newaxis], 
                                     (T, P, Lat, Lon)).copy()
    
    # Regrid pressure to cell interfaces (x1f) instead of centers
    # Note: x1f has one more element than x1
    pressure_at_interfaces = regrid_pressure_to_height(
        pressure_tpll,
        rho_tpll,
        topo_ll,
        plev,
        lats,
        lons,
        x1f,  # Use cell interfaces for vertical coordinate
        x2,   # Use cell centers for horizontal
        x3,
        gravity,
        EARTH_RADIUS,
        bounds_error=False,
        z_tpll=z_tpll,
        n_jobs=-1
    )
    
    # Add to regridded variables with special name
    regridded_vars['pressure_level'] = pressure_at_interfaces
    
    print(f"   Pressure regridded to interfaces: shape {pressure_at_interfaces.shape}")
    print(f"   Pressure range: [{np.nanmin(pressure_at_interfaces):.1f}, "
          f"{np.nanmax(pressure_at_interfaces):.1f}] Pa")
    
    # Step 8: Prepare coordinates for output
    print("\n7. Preparing output NetCDF file...")
    
    # Convert time to hours since reference
    # If time is already in a numeric format, use it; otherwise convert
    if time.dtype.kind == 'M':  # datetime64
        # Convert to hours since 1900-01-01
        reference_time = np.datetime64('1900-01-01T00:00:00')
        time_hours = (time - reference_time) / np.timedelta64(1, 'h')
    else:
        # Assume it's already numeric
        time_hours = time.astype(float)
    
    output_coords = {
        'time': time_hours,
        'x1': x1,      # Cell centers
        'x1f': x1f,    # Cell interfaces
        'x2': x2,
        'x2f': x2f,
        'x3': x3,
        'x3f': x3f,
    }
    
    # Prepare metadata
    output_metadata = {
        **metadata,
        'center_latitude': geometry['center_latitude'],
        'center_longitude': geometry['center_longitude'],
        'planet_radius': EARTH_RADIUS,
        'planet_gravity': gravity,
        'time_units': 'hours since 1900-01-01 00:00:00',
        'nx1': geometry['cells']['nx1'],
        'nx2': geometry['cells']['nx2'],
        'nx3': geometry['cells']['nx3'],
        'nghost': geometry['cells']['nghost'],
    }
    
    # Add variable-specific metadata
    var_metadata = {
        'u': {'units': 'm s-1', 'long_name': 'U component of wind', 'standard_name': 'eastward_wind'},
        'v': {'units': 'm s-1', 'long_name': 'V component of wind', 'standard_name': 'northward_wind'},
        'w': {'units': 'Pa s-1', 'long_name': 'Vertical velocity', 'standard_name': 'lagrangian_tendency_of_air_pressure'},
        't': {'units': 'K', 'long_name': 'Temperature', 'standard_name': 'air_temperature'},
        'z': {'units': 'm2 s-2', 'long_name': 'Geopotential'},
        'q': {'units': 'kg kg-1', 'long_name': 'Specific humidity', 'standard_name': 'specific_humidity'},
        'rho': {'units': 'kg m-3', 'long_name': 'Air density', 'standard_name': 'air_density'},
        'ciwc': {'units': 'kg kg-1', 'long_name': 'Specific cloud ice water content'},
        'cswc': {'units': 'kg kg-1', 'long_name': 'Specific snow water content'},
        'clwc': {'units': 'kg kg-1', 'long_name': 'Specific cloud liquid water content'},
        'crwc': {'units': 'kg kg-1', 'long_name': 'Specific rain water content'},
        'pressure_level': {'units': 'Pa', 'long_name': 'Pressure at cell interfaces', 'standard_name': 'air_pressure'},
    }
    
    for var_name, var_info in var_metadata.items():
        if var_name in regridded_vars:
            for key, value in var_info.items():
                output_metadata[f'{var_name}_{key}'] = value
    
    # Processing history
    processing_history = (
        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}: "
        f"Regridded from ECMWF ERA5 pressure-lat-lon grids to Cartesian distance grids. "
        f"Original domain: lat [{lats[0]:.4f}, {lats[-1]:.4f}] deg, "
        f"lon [{lons[0]:.4f}, {lons[-1]:.4f}] deg, "
        f"pressure [{plev[-1]/100:.0f}, {plev[0]/100:.0f}] hPa. "
        f"Output domain: x1 [{x1[0]:.1f}, {x1[-1]:.1f}] m, "
        f"x2 [{x2[0]:.1f}, {x2[-1]:.1f}] m, "
        f"x3 [{x3[0]:.1f}, {x3[-1]:.1f}] m. "
        f"Center: ({geometry['center_latitude']:.4f}, {geometry['center_longitude']:.4f}). "
        f"Ghost zones: {geometry['cells']['nghost']}. "
        f"Date: {process_date}."
    )
    
    # Step 9: Save to NetCDF with modified function to include both centers and interfaces
    print(f"   Writing to: {output_file}")
    save_regridded_data_with_interfaces(
        output_file,
        regridded_vars,
        output_coords,
        output_metadata,
        processing_history
    )
    
    print("\n" + "="*70)
    print("Regridding completed successfully!")
    print("="*70)
    print(f"\nOutput file: {output_file}")
    print(f"Variables: {', '.join(regridded_vars.keys())}")
    print(f"Shape: (T={len(time)}, Z={len(x1)}, Y={len(x2)}, X={len(x3)})")
    print(f"Grid includes {geometry['cells']['nghost']} ghost cells on each side")


def save_regridded_data_with_interfaces(
    filename: str,
    variables: Dict,
    coordinates: Dict,
    metadata: Dict,
    processing_history: str
) -> None:
    """
    Save regridded data to NetCDF including both cell centers and interfaces.
    
    This is similar to save_regridded_data_to_netcdf from regrid.py but includes
    interface coordinates.
    
    Args:
        filename: Output NetCDF file path
        variables: Dict of variables with shape (T, Z, Y, X)
        coordinates: Dict with 'time', 'x1', 'x1f', 'x2', 'x2f', 'x3', 'x3f'
        metadata: Dict with metadata
        processing_history: Processing history string
    """
    try:
        from netCDF4 import Dataset
        import numpy as np
    except ImportError:
        raise ImportError("netCDF4 is required. Install with: pip install netCDF4")
    
    with Dataset(filename, "w", format="NETCDF4") as ncfile:
        # Get dimensions from first variable
        first_var = next(iter(variables.values()))
        T, Z, Y, X = first_var.shape
        
        # Create dimensions for centers
        ncfile.createDimension("time", T)
        ncfile.createDimension("x1", Z)  # Height centers
        ncfile.createDimension("x2", Y)  # Y centers
        ncfile.createDimension("x3", X)  # X centers
        
        # Create dimensions for interfaces
        ncfile.createDimension("x1f", Z + 1)  # Height interfaces
        ncfile.createDimension("x2f", Y + 1)  # Y interfaces
        ncfile.createDimension("x3f", X + 1)  # X interfaces
        
        # Create coordinate variables for centers
        time_var = ncfile.createVariable("time", "f8", ("time",))
        x1_var = ncfile.createVariable("x1", "f8", ("x1",))
        x2_var = ncfile.createVariable("x2", "f8", ("x2",))
        x3_var = ncfile.createVariable("x3", "f8", ("x3",))
        
        # Create coordinate variables for interfaces
        x1f_var = ncfile.createVariable("x1f", "f8", ("x1f",))
        x2f_var = ncfile.createVariable("x2f", "f8", ("x2f",))
        x3f_var = ncfile.createVariable("x3f", "f8", ("x3f",))
        
        # Set center coordinate attributes
        time_var.axis = "T"
        time_var.long_name = "time"
        time_var.units = metadata.get('time_units', 'hours since 1900-01-01 00:00:00')
        time_var[:] = coordinates['time'].astype("f8")
        
        x1_var.axis = "Z"
        x1_var.long_name = "cell_center_height"
        x1_var.units = "meters"
        x1_var.positive = "up"
        x1_var.description = "Height at cell centers"
        x1_var[:] = coordinates['x1'].astype("f8")
        
        x2_var.axis = "Y"
        x2_var.long_name = "cell_center_y_coordinate"
        x2_var.units = "meters"
        x2_var.standard_name = "projection_y_coordinate"
        x2_var.description = "Y coordinate at cell centers (North-South)"
        x2_var[:] = coordinates['x2'].astype("f8")
        
        x3_var.axis = "X"
        x3_var.long_name = "cell_center_x_coordinate"
        x3_var.units = "meters"
        x3_var.standard_name = "projection_x_coordinate"
        x3_var.description = "X coordinate at cell centers (East-West)"
        x3_var[:] = coordinates['x3'].astype("f8")
        
        # Set interface coordinate attributes
        x1f_var.long_name = "cell_interface_height"
        x1f_var.units = "meters"
        x1f_var.positive = "up"
        x1f_var.description = "Height at cell interfaces (boundaries)"
        x1f_var[:] = coordinates['x1f'].astype("f8")
        
        x2f_var.long_name = "cell_interface_y_coordinate"
        x2f_var.units = "meters"
        x2f_var.description = "Y coordinate at cell interfaces (boundaries)"
        x2f_var[:] = coordinates['x2f'].astype("f8")
        
        x3f_var.long_name = "cell_interface_x_coordinate"
        x3f_var.units = "meters"
        x3f_var.description = "X coordinate at cell interfaces (boundaries)"
        x3f_var[:] = coordinates['x3f'].astype("f8")
        
        # Create data variables
        for var_name, var_data in variables.items():
            # Special handling for pressure_level which is on cell interfaces
            if var_name == 'pressure_level':
                # pressure_level is on vertical interfaces: (T, x1f, x2, x3)
                Zf = Z + 1
                if var_data.shape != (T, Zf, Y, X):
                    raise ValueError(
                        f"Variable '{var_name}' has shape {var_data.shape}, "
                        f"expected ({T}, {Zf}, {Y}, {X})"
                    )
                
                var = ncfile.createVariable(var_name, "f4", ("time", "x1f", "x2", "x3"),
                                           zlib=True, complevel=4)
                var[:] = var_data.astype("f4")
            else:
                # Regular variables on cell centers
                if var_data.shape != (T, Z, Y, X):
                    raise ValueError(
                        f"Variable '{var_name}' has shape {var_data.shape}, "
                        f"expected ({T}, {Z}, {Y}, {X})"
                    )
                
                var = ncfile.createVariable(var_name, "f4", ("time", "x1", "x2", "x3"),
                                           zlib=True, complevel=4)
                var[:] = var_data.astype("f4")
            
            # Set variable attributes from metadata
            if f"{var_name}_units" in metadata:
                var.units = metadata[f"{var_name}_units"]
            if f"{var_name}_long_name" in metadata:
                var.long_name = metadata[f"{var_name}_long_name"]
            if f"{var_name}_standard_name" in metadata:
                var.standard_name = metadata[f"{var_name}_standard_name"]
        
        # Set global attributes
        ncfile.title = "Regridded ECMWF ERA5 Data on Cartesian Grid"
        ncfile.institution = metadata.get('institution', 'Generated by ECMWF regridding pipeline')
        ncfile.source = metadata.get('source', 'ECMWF ERA5')
        ncfile.conventions = "CF-1.8"
        ncfile.creation_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Grid information
        ncfile.coordinate_system = "Local Cartesian projection"
        ncfile.grid_description = (
            "Finite volume grid with cell centers and interfaces. "
            "x1 (height) in meters positive upward, "
            "x2 (Y) in meters positive northward, "
            "x3 (X) in meters positive eastward. "
            "Ghost cells included on all sides."
        )
        
        # Add geometry metadata
        if 'center_latitude' in metadata:
            ncfile.center_latitude = float(metadata['center_latitude'])
        if 'center_longitude' in metadata:
            ncfile.center_longitude = float(metadata['center_longitude'])
        if 'nx1' in metadata:
            ncfile.nx1_interior = int(metadata['nx1'])
        if 'nx2' in metadata:
            ncfile.nx2_interior = int(metadata['nx2'])
        if 'nx3' in metadata:
            ncfile.nx3_interior = int(metadata['nx3'])
        if 'nghost' in metadata:
            ncfile.nghost = int(metadata['nghost'])
        if 'planet_radius' in metadata:
            ncfile.planet_radius = float(metadata['planet_radius'])
            ncfile.planet_radius_units = "meters"
        if 'planet_gravity' in metadata:
            ncfile.planet_gravity = float(metadata['planet_gravity'])
            ncfile.planet_gravity_units = "m s-2"
        
        # Add processing history
        ncfile.history = processing_history


def main():
    """Main function to execute regridding pipeline."""
    parser = argparse.ArgumentParser(
        description='ECMWF data curation pipeline - Step 3: Regrid to Cartesian coordinates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script regrids ERA5 variables from pressure-lat-lon grids to Cartesian
cell-centered coordinates, including ghost zones.

Example:
    python regrid_era5_to_cartesian.py earth.yaml ./data_folder --output regridded_era5.nc

The script will:
  1. Parse YAML configuration to extract geometry
  2. Compute cell centers and interfaces (with ghost zones)
  3. Load ERA5 data from steps 1 and 2
  4. Regrid all variables to Cartesian grid
  5. Save output NetCDF with metadata and coordinates
        """
    )
    
    parser.add_argument('config_file', type=str,
                       help='Path to YAML configuration file')
    parser.add_argument('data_dir', type=str,
                       help='Directory containing ERA5 data files from steps 1 and 2')
    parser.add_argument('--output', '-o', type=str, default='regridded_era5.nc',
                       help='Output NetCDF file path (default: regridded_era5.nc)')
    parser.add_argument('--date', type=str, default=None,
                       help='Specific date to process (YYYYMMDD). If not specified, processes first available date.')
    
    args = parser.parse_args()
    
    try:
        regrid_era5_to_cartesian(
            args.config_file,
            args.data_dir,
            args.output,
            args.date
        )
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        sys.exit(1)
    
    except yaml.YAMLError as e:
        print(f"\n✗ YAML parsing error: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
