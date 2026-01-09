"""
Collection of utility functions for downloading ECMWF weather data.

Functions include:
    - validate_date_format: Validate date strings.
    - validate_region_bounds: Validate geographical bounds.
    - validate_pressure_levels: Validate and convert pressure levels.
    - validate_variable_names: Validate variable names.
    - add_common_arguments: Add common command-line arguments to an argparse parser.
    - generate_date_list: Generate a list of dates between two given dates.
    - fetch_single_day: Download weather data for a single day using ECMWFWeatherAPI.
"""

import os
from datetime import datetime, timedelta
from typing import List

STANDARD_PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
    225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
    775, 800, 825, 850, 875, 900, 925, 950, 975, 1000
]

STANDARD_TIMES = [
    '00:00', '06:00', '12:00', '18:00'
]

STANDARD_VARIABLES = [
    # dynamics variables
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'vertical_velocity',
    'divergence',
    'vorticity',
    'potential_vorticity',
    'geopotential',
    # density variables
    'specific_cloud_ice_water_content',
    'specific_humidity',
    'specific_snow_water_content',
    'specific_cloud_liquid_water_content',
    'specific_rain_water_content',
    'fraction_of_cloud_cover',
    'relative_humidity'
]

############## Validations ###############

def validate_date_format(date_str: str) -> bool:
    """Validate date format."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_region_bounds(
        latmin: float, latmax: float, 
        lonmin: float, lonmax: float) -> None:
    """
    Validate geographical bounds.
    
    Args:
        latmin: Minimum latitude (-90 to 90).
        latmax: Maximum latitude (-90 to 90).
        lonmin: Minimum longitude (-180 to 180).
        lonmax: Maximum longitude (-180 to 180).
    
    Raises:
        ValueError: If bounds are invalid.
    """
    if not (-90 <= latmin <= 90) or not (-90 <= latmax <= 90):
        raise ValueError(f"Latitude must be between -90 and 90. Got: {latmin}, {latmax}")
    
    if latmin >= latmax:
        raise ValueError(f"latmin must be less than latmax. Got: {latmin} >= {latmax}")
    
    if not (-180 <= lonmin <= 180) or not (-180 <= lonmax <= 180):
        raise ValueError(f"Longitude must be between -180 and 180. Got: {lonmin}, {lonmax}")
    
    if lonmin >= lonmax:
        raise ValueError(f"lonmin must be less than lonmax. Got: {lonmin} >= {lonmax}")

def validate_pressure_levels(pressure_levels: List[int]) -> List[str]:
    """
    Validate pressure levels and convert to strings.
    
    Args:
        pressure_levels: List of pressure levels in hPa.
    
    Returns:
        List of pressure levels as strings.
    
    Raises:
        ValueError: If pressure levels are invalid.
    """
    for level in pressure_levels:
        if level not in STANDARD_PRESSURE_LEVELS:
            raise ValueError(
                f"Pressure level {level} hPa is not available. "
                f"Available levels: {STANDARD_PRESSURE_LEVELS}"
            )
    return [str(level) for level in pressure_levels]

def validate_variable_names(variables: List[str]) -> None:
    """
    Validate variable names.
    
    Args:
        variables: List of variable names.
    
    Raises:
        ValueError: If variable names are invalid.
    """
    for var in variables:
        if var not in STANDARD_VARIABLES:
            raise ValueError(
                f"Variable '{var}' is not available. "
                f"Available variables: {STANDARD_VARIABLES}"
            )

########### Command Line Arguments ###########

def add_common_arguments(parser):
    """Add common command-line arguments to the parser."""
    # Required arguments
    parser.add_argument('--latmin', type=float, required=True,
                        help='Minimum latitude (-90 to 90)')
    parser.add_argument('--latmax', type=float, required=True,
                        help='Maximum latitude (-90 to 90)')
    parser.add_argument('--lonmin', type=float, required=True,
                        help='Minimum longitude (-180 to 180)')
    parser.add_argument('--lonmax', type=float, required=True,
                        help='Maximum longitude (-180 to 180)')
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, required=True,
                        help='End date in YYYY-MM-DD format')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for NetCDF files (one file per day)')
    parser.add_argument('--times', nargs='+', default=None,
                        help='Specific times to download (e.g., 00:00 06:00 12:00 18:00). '
                             'Default: all available times (every 6 hours)')
    parser.add_argument('--jobs', type=int, default=4,
                        help='Number of parallel download jobs (default: 4). '
                             'Each job downloads one day.')
    parser.add_argument('--api-key', type=str, default=None,
                        help='CDS API key (if not using environment variable or ~/.cdsapirc)')
    parser.add_argument('--api-url', type=str, default=None,
                        help='CDS API URL (default: https://cds.climate.copernicus.eu/api)')

    return parser

########### Download Wrappers ###########

def generate_date_list(start_date: str, end_date: str):
    """
    Generate list of dates from start_date to end_date (inclusive).
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        List of date strings in YYYY-MM-DD format
    """
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates

def fetch_single_day(api,
        date_str: str, 
        variables: List[str],
        pressure_levels: List[int],
        latmin: float,
        latmax: float,
        lonmin: float, 
        lonmax: float,
        times: List[str],
        output_dir: str,
        day_idx: int,
        total_days: int,
        prefix: str = 'era5'):
    """
    Download data for a single day.
    
    Args:
        api: ECMWFWeatherAPI instance
        date_str: Date in YYYY-MM-DD format
        variables: List of variable names
        pressure_levels: List of pressure levels in hPa
        latmin, latmax, lonmin, lonmax: Geographic bounds
        times: List of times
        output_dir: Output directory for storing files
        day_idx: Index of this day (for progress tracking)
        total_days: Total number of days
        prefix: Prefix for output filenames
    
    Returns:
        output_file_path or None on error
    """
    try:
        # format YYYY-MM-DD to YYYYMMDD
        date_str_nodash = date_str.replace('-', '')
        filename = os.path.join(output_dir, f'{prefix}_{date_str_nodash}.nc')
        
        print(f"[{day_idx+1}/{total_days}] Downloading {date_str}...")
        
        output_file = api.fetch_weather_data(
            latmin=latmin,
            latmax=latmax,
            lonmin=lonmin,
            lonmax=lonmax,
            start_date=date_str,
            end_date=date_str,
            variables=variables,
            pressure_levels=pressure_levels,
            output_file=filename,
            times=times
        )
        
        print(f"[{day_idx+1}/{total_days}] ✓ Completed {date_str} -> {os.path.basename(output_file)}")
        return output_file
    
    except Exception as e:
        print(f"[{day_idx+1}/{total_days}] ✗ Failed {date_str}: {e}")
        return None
