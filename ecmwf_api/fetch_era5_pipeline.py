#!/usr/bin/env python3
"""
ECMWF data fetching and curation pipeline - Step 1.

This script reads a YAML configuration file with geometry and integration fields,
calculates the region limits in lat-lon coordinates that cover all ghost zones,
adds a 10% buffer zone, and fetches ECMWF ERA5 hourly data on pressure levels.

The script performs the following steps:
1. Parse YAML configuration to extract geometry and integration fields
2. Calculate lat-lon region limits from Cartesian domain (including ghost zones)
3. Add 10% buffer zone to soundings
4. Fetch ERA5 hourly densities data using existing fetch_era5_hourly_densities.py
5. Fetch ERA5 hourly dynamics data using existing fetch_era5_hourly_dynamics.py
6. Save outputs in a folder named "LATMIN_LATMAX_LONMIN_LONMAX" with S/N/W/E postfixes

Usage:
    python fetch_era5_pipeline.py <config.yaml>
    
    python fetch_era5_pipeline.py earth.yaml

Requirements:
    - PyYAML for YAML parsing
    - All dependencies from fetch_era5_hourly_densities.py and fetch_era5_hourly_dynamics.py
"""

import argparse
import sys
import os
import yaml
import subprocess
from typing import Dict, Tuple

# Add current directory to path for importing local modules
sys.path.insert(0, os.path.dirname(__file__))

from ecmwf_utils import validate_date_format


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


def extract_geometry_info(config: Dict) -> Dict:
    """
    Extract geometry information from configuration.
    
    Args:
        config: Parsed YAML configuration dictionary
        
    Returns:
        Dictionary containing geometry information
        
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
    
    # Convert bounds to float (YAML might parse them as strings or floats)
    bounds_float = {
        'x1min': float(bounds['x1min']),
        'x1max': float(bounds['x1max']),
        'x2min': float(bounds['x2min']),
        'x2max': float(bounds['x2max']),
        'x3min': float(bounds['x3min']),
        'x3max': float(bounds['x3max'])
    }
    
    # Convert cells to int (YAML should parse them as ints, but be safe)
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


def extract_integration_info(config: Dict) -> Dict:
    """
    Extract integration information from configuration.
    
    Args:
        config: Parsed YAML configuration dictionary
        
    Returns:
        Dictionary containing integration information
        
    Raises:
        ValueError: If required integration fields are missing or invalid
    """
    if 'integration' not in config:
        raise ValueError("Configuration must contain 'integration' field")
    
    integration = config['integration']
    
    # Extract start date
    if 'start-date' not in integration:
        raise ValueError("Integration must contain 'start-date' field")
    
    start_date = integration['start-date']
    # Convert to string if YAML parsed it as a date object
    if hasattr(start_date, 'strftime'):
        start_date = start_date.strftime('%Y-%m-%d')
    else:
        start_date = str(start_date)
    
    if not validate_date_format(start_date):
        raise ValueError(f"Invalid start-date format: {start_date}. Use YYYY-MM-DD")
    
    # end-date is optional; if not provided, use start-date
    end_date = integration.get('end-date', start_date)
    # Convert to string if YAML parsed it as a date object
    if hasattr(end_date, 'strftime'):
        end_date = end_date.strftime('%Y-%m-%d')
    else:
        end_date = str(end_date)
    
    if not validate_date_format(end_date):
        raise ValueError(f"Invalid end-date format: {end_date}. Use YYYY-MM-DD")
    
    return {
        'start_date': start_date,
        'end_date': end_date
    }


def calculate_latlon_limits(geometry: Dict) -> Tuple[float, float, float, float]:
    """
    Calculate lat-lon region limits from Cartesian domain.
    
    The domain uses Cartesian coordinates where:
    - x1 is Z direction (vertical, in meters)
    - x2 is Y direction (north-south, in meters)
    - x3 is X direction (east-west, in meters)
    
    The bounds include ghost zones. We need to calculate the lat-lon limits
    that cover the entire domain including ghost zones.
    
    Args:
        geometry: Dictionary containing geometry information
        
    Returns:
        Tuple of (latmin, latmax, lonmin, lonmax) in degrees
    """
    bounds = geometry['bounds']
    center_lat = geometry['center_latitude']
    center_lon = geometry['center_longitude']
    
    # Extract Y (north-south) and X (east-west) bounds
    # x2 is Y direction (north-south in meters)
    # x3 is X direction (east-west in meters)
    y_min = bounds['x2min']  # meters
    y_max = bounds['x2max']  # meters
    x_min = bounds['x3min']  # meters
    x_max = bounds['x3max']  # meters
    
    # Convert meters to degrees using approximate conversion
    # At the equator: 1 degree latitude ≈ 111,320 meters
    # 1 degree longitude ≈ 111,320 * cos(latitude) meters
    # We use the center latitude for longitude conversion
    
    import math
    
    meters_per_degree_lat = 111320.0  # meters per degree latitude
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(center_lat))
    
    # Calculate lat-lon extents from the center
    # Y direction corresponds to latitude (north-south)
    # delta_lat_min = y_min / meters_per_degree_lat  # Unused variable removed
    
    # X direction corresponds to longitude (east-west)
    delta_lon_min = x_min / meters_per_degree_lon
    delta_lon_max = x_max / meters_per_degree_lon
    
    # Calculate absolute lat-lon limits
    # Note: y_min, y_max, x_min, x_max are distances from center
    # For a domain that starts at 0, the center is at (y_max-y_min)/2
    # But the problem states bounds are domain boundaries including ghost cells
    # So we interpret these as offsets from center
    
    # Calculate domain center offsets
    y_center_offset = (y_min + y_max) / 2.0
    x_center_offset = (x_min + x_max) / 2.0
    
    # Calculate extents relative to center
    y_extent = (y_max - y_min) / 2.0
    x_extent = (x_max - x_min) / 2.0
    
    # Convert to lat-lon extents from center
    lat_extent = y_extent / meters_per_degree_lat
    lon_extent = x_extent / meters_per_degree_lon
    
    # Calculate lat-lon limits
    latmin = center_lat - lat_extent
    latmax = center_lat + lat_extent
    lonmin = center_lon - lon_extent
    lonmax = center_lon + lon_extent
    
    return latmin, latmax, lonmin, lonmax


def validate_domain_size(latmin: float, latmax: float, 
                        lonmin: float, lonmax: float,
                        min_size_degrees: float = 1.0) -> None:
    """
    Validate that the horizontal domain size meets minimum requirements.
    
    The domain must be at least 100 km (approximately 1 degree) in both
    latitude and longitude directions.
    
    Args:
        latmin, latmax, lonmin, lonmax: Domain limits in degrees
        min_size_degrees: Minimum size in degrees (default: 1.0 degree ≈ 100 km)
        
    Raises:
        ValueError: If domain size is too small in either direction
    """
    lat_range = latmax - latmin
    lon_range = lonmax - lonmin
    
    # Convert to approximate kilometers for error message
    # 1 degree ≈ 111.32 km
    km_per_degree = 111.32
    lat_range_km = lat_range * km_per_degree
    lon_range_km = lon_range * km_per_degree
    min_size_km = min_size_degrees * km_per_degree
    
    errors = []
    
    if lat_range < min_size_degrees:
        errors.append(
            f"North-South extent ({lat_range:.4f}° ≈ {lat_range_km:.1f} km) is less than "
            f"minimum required {min_size_degrees:.1f}° ≈ {min_size_km:.1f} km"
        )
    
    if lon_range < min_size_degrees:
        errors.append(
            f"East-West extent ({lon_range:.4f}° ≈ {lon_range_km:.1f} km) is less than "
            f"minimum required {min_size_degrees:.1f}° ≈ {min_size_km:.1f} km"
        )
    
    if errors:
        error_msg = "Domain size validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        error_msg += (
            "\n\nThe horizontal domain size must be at least 100 km (1 degree) in both directions. "
            "Please increase the domain bounds in your YAML configuration."
        )
        raise ValueError(error_msg)


def add_buffer_zone(latmin: float, latmax: float, 
                   lonmin: float, lonmax: float,
                   buffer_percent: float = 0.10) -> Tuple[float, float, float, float]:
    """
    Add buffer zone to region limits.
    
    Args:
        latmin, latmax, lonmin, lonmax: Original region limits in degrees
        buffer_percent: Buffer percentage (default: 0.10 for 10%)
        
    Returns:
        Tuple of (latmin, latmax, lonmin, lonmax) with buffer added
    """
    lat_range = latmax - latmin
    lon_range = lonmax - lonmin
    
    lat_buffer = lat_range * buffer_percent
    lon_buffer = lon_range * buffer_percent
    
    buffered_latmin = max(-90.0, latmin - lat_buffer)
    buffered_latmax = min(90.0, latmax + lat_buffer)
    buffered_lonmin = max(-180.0, lonmin - lon_buffer)
    buffered_lonmax = min(180.0, lonmax + lon_buffer)
    
    return buffered_latmin, buffered_latmax, buffered_lonmin, buffered_lonmax


def format_lat_lon_string(value: float, is_latitude: bool) -> str:
    """
    Format latitude or longitude value with appropriate postfix.
    
    Args:
        value: Latitude or longitude value
        is_latitude: True if value is latitude, False if longitude
        
    Returns:
        Formatted string with postfix (e.g., "30.5N", "110.2W")
    """
    abs_value = abs(value)
    
    if is_latitude:
        postfix = 'N' if value >= 0 else 'S'
    else:
        postfix = 'E' if value >= 0 else 'W'
    
    # Format with 2 decimal places
    return f"{abs_value:.2f}{postfix}"


def generate_output_dirname(latmin: float, latmax: float,
                            lonmin: float, lonmax: float) -> str:
    """
    Generate output directory name in format LATMIN_LATMAX_LONMIN_LONMAX.
    
    Args:
        latmin, latmax, lonmin, lonmax: Region limits in degrees
        
    Returns:
        Directory name string (e.g., "30.00N_40.00N_110.00W_100.00W")
    """
    latmin_str = format_lat_lon_string(latmin, is_latitude=True)
    latmax_str = format_lat_lon_string(latmax, is_latitude=True)
    lonmin_str = format_lat_lon_string(lonmin, is_latitude=False)
    lonmax_str = format_lat_lon_string(lonmax, is_latitude=False)
    
    return f"{latmin_str}_{latmax_str}_{lonmin_str}_{lonmax_str}"


def fetch_era5_data(latmin: float, latmax: float, lonmin: float, lonmax: float,
                   start_date: str, end_date: str, output_dir: str) -> None:
    """
    Fetch ERA5 hourly data using existing fetch scripts.
    
    Args:
        latmin, latmax, lonmin, lonmax: Region limits in degrees
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_dir: Output directory path
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Fetch densities data
    print("\n" + "="*70)
    print("Fetching ERA5 Hourly Densities Data")
    print("="*70)
    
    densities_script = os.path.join(script_dir, 'fetch_era5_hourly_densities.py')
    densities_cmd = [
        'python3', densities_script,
        '--latmin', str(latmin),
        '--latmax', str(latmax),
        '--lonmin', str(lonmin),
        '--lonmax', str(lonmax),
        '--start-date', start_date,
        '--end-date', end_date,
        '--output', output_dir
    ]
    
    try:
        result = subprocess.run(densities_cmd, check=True, capture_output=False, text=True)
        print("\n✓ Densities data fetch completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Densities data fetch failed: {e}")
        raise RuntimeError(f"Failed to fetch densities data: {e}")
    
    # Fetch dynamics data
    print("\n" + "="*70)
    print("Fetching ERA5 Hourly Dynamics Data")
    print("="*70)
    
    dynamics_script = os.path.join(script_dir, 'fetch_era5_hourly_dynamics.py')
    dynamics_cmd = [
        'python3', dynamics_script,
        '--latmin', str(latmin),
        '--latmax', str(latmax),
        '--lonmin', str(lonmin),
        '--lonmax', str(lonmax),
        '--start-date', start_date,
        '--end-date', end_date,
        '--output', output_dir
    ]
    
    try:
        result = subprocess.run(dynamics_cmd, check=True, capture_output=False, text=True)
        print("\n✓ Dynamics data fetch completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Dynamics data fetch failed: {e}")
        raise RuntimeError(f"Failed to fetch dynamics data: {e}")


def main():
    """Main function to execute ECMWF data fetching pipeline."""
    parser = argparse.ArgumentParser(
        description='ECMWF data fetching and curation pipeline - Step 1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script reads a YAML configuration file with geometry and integration fields,
calculates the region limits in lat-lon coordinates, and fetches ERA5 data.

Example:
    python fetch_era5_pipeline.py earth.yaml

The script will:
  1. Parse YAML configuration
  2. Calculate lat-lon limits from Cartesian domain (with ghost zones)
  3. Add 10% buffer zone
  4. Fetch ERA5 hourly densities and dynamics data
  5. Save output in folder named "LATMIN_LATMAX_LONMIN_LONMAX"
        """
    )
    
    parser.add_argument('config_file', type=str,
                       help='Path to YAML configuration file')
    parser.add_argument('--output-base', type=str, default='.',
                       help='Base directory for output (default: current directory)')
    
    args = parser.parse_args()
    
    try:
        # Parse YAML configuration
        print("="*70)
        print("ECMWF Data Fetching Pipeline - Step 1")
        print("="*70)
        print(f"\nReading configuration from: {args.config_file}")
        
        config = parse_yaml_config(args.config_file)
        
        # Extract geometry information
        print("\nExtracting geometry information...")
        geometry = extract_geometry_info(config)
        
        print(f"  Center: ({geometry['center_latitude']}°, {geometry['center_longitude']}°)")
        print(f"  Domain bounds (with ghost zones):")
        print(f"    x1 (Z): [{geometry['bounds']['x1min']}, {geometry['bounds']['x1max']}] m")
        print(f"    x2 (Y): [{geometry['bounds']['x2min']}, {geometry['bounds']['x2max']}] m")
        print(f"    x3 (X): [{geometry['bounds']['x3min']}, {geometry['bounds']['x3max']}] m")
        print(f"  Grid cells: nx1={geometry['cells']['nx1']}, nx2={geometry['cells']['nx2']}, "
              f"nx3={geometry['cells']['nx3']}, nghost={geometry['cells']['nghost']}")
        
        # Extract integration information
        print("\nExtracting integration information...")
        integration = extract_integration_info(config)
        print(f"  Start date: {integration['start_date']}")
        print(f"  End date: {integration['end_date']}")
        
        # Calculate lat-lon limits
        print("\nCalculating lat-lon region limits...")
        latmin, latmax, lonmin, lonmax = calculate_latlon_limits(geometry)
        print(f"  Original limits (with ghost zones):")
        print(f"    Latitude:  [{latmin:.4f}, {latmax:.4f}]")
        print(f"    Longitude: [{lonmin:.4f}, {lonmax:.4f}]")
        
        # Validate domain size
        print("\nValidating domain size...")
        validate_domain_size(latmin, latmax, lonmin, lonmax, min_size_degrees=1.0)
        print(f"  ✓ Domain size validation passed")
        
        # Add buffer zone
        print("\nAdding 10% buffer zone...")
        latmin_buf, latmax_buf, lonmin_buf, lonmax_buf = add_buffer_zone(
            latmin, latmax, lonmin, lonmax, buffer_percent=0.10
        )
        print(f"  Buffered limits:")
        print(f"    Latitude:  [{latmin_buf:.4f}, {latmax_buf:.4f}]")
        print(f"    Longitude: [{lonmin_buf:.4f}, {lonmax_buf:.4f}]")
        
        # Generate output directory name
        output_dirname = generate_output_dirname(latmin_buf, latmax_buf, lonmin_buf, lonmax_buf)
        output_dir = os.path.join(args.output_base, output_dirname)
        
        print(f"\nOutput directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Fetch ERA5 data
        fetch_era5_data(
            latmin_buf, latmax_buf, lonmin_buf, lonmax_buf,
            integration['start_date'], integration['end_date'],
            output_dir
        )
        
        print("\n" + "="*70)
        print("Pipeline completed successfully!")
        print("="*70)
        print(f"\nData saved to: {output_dir}")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        sys.exit(1)
    
    except yaml.YAMLError as e:
        print(f"\n✗ YAML parsing error: {e}")
        sys.exit(1)
    
    except RuntimeError as e:
        print(f"\n✗ Runtime error: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
