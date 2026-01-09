"""
ECMWF Weather Data Fetching and Processing API

This module provides an API for fetching and processing ECMWF ERA5 weather data.
Given geographical bounds, atmospheric variables, and a time window, it queries
and fetches public ECMWF weather data using the Climate Data Store (CDS) API.

Requirements:
    - cdsapi: Climate Data Store API client
    - API key: Required for authentication with ECMWF CDS

Usage:
    from ecmwf_weather_api import ECMWFWeatherAPI
    
    api = ECMWFWeatherAPI(api_key="your-api-key", api_url="https://cds.climate.copernicus.eu/api")
    
    data = api.fetch_weather_data(
        latmin=30.0, latmax=40.0,
        lonmin=-110.0, lonmax=-100.0,
        start_date="2024-01-01",
        end_date="2024-01-02",
        variables=["temperature", "u_component_of_wind", "v_component_of_wind"],
        pressure_levels=[1000, 925, 850, 700, 500]
    )
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import tempfile
from ecmwf_utils import (
        validate_region_bounds, 
        validate_pressure_levels,
        validate_variable_names
        )

try:
    import cdsapi
    CDS_AVAILABLE = True
except ImportError:
    CDS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

# ECMWF API endpoints
ECMWF_CDS_API_URL = "https://cds.climate.copernicus.eu/api"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECMWFWeatherAPI:
    """
    API for fetching and processing ECMWF ERA5 weather data.
    
    This class provides methods to query and download weather data from the
    ECMWF Climate Data Store (CDS) given geographical bounds, atmospheric
    variables, and a time window.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize the ECMWF Weather API.
        
        Args:
            api_key: CDS API key. If not provided, will try to read from
                    environment variable CDSAPI_KEY or ~/.cdsapirc file.
            api_url: CDS API URL. If not provided, uses default CDS URL.
        
        Raises:
            ImportError: If cdsapi package is not installed.
            ValueError: If API key is not provided and cannot be found.
        """
        if not CDS_AVAILABLE:
            raise ImportError(
                "cdsapi package is required. Install it with: pip install cdsapi"
            )
        
        # Configure API credentials
        self.api_key = api_key or os.environ.get('CDSAPI_KEY')
        self.api_url = api_url or os.environ.get('CDSAPI_URL', ECMWF_CDS_API_URL)
        
        # Initialize CDS client
        if self.api_key:
            self.client = cdsapi.Client(url=self.api_url, key=self.api_key)
        else:
            # Try using default configuration from ~/.cdsapirc
            try:
                self.client = cdsapi.Client()
                logger.info("Using CDS API credentials from ~/.cdsapirc")
            except Exception as e:
                raise ValueError(
                    "API key not provided and could not be found in environment "
                    "variable CDSAPI_KEY or ~/.cdsapirc file. "
                    "Please provide an API key or configure ~/.cdsapirc"
                ) from e
        
        logger.info("ECMWF Weather API initialized successfully")
    
    
    def _parse_date_range(self, start_date: str, end_date: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Parse date range and generate list of years, months, and days.
        
        Args:
            start_date: Start date in format 'YYYY-MM-DD'.
            end_date: End date in format 'YYYY-MM-DD'.
        
        Returns:
            Tuple of (years, months, days) as lists of strings.
        
        Raises:
            ValueError: If dates are invalid or in wrong format.
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(
                f"Dates must be in format 'YYYY-MM-DD'. Got: {start_date}, {end_date}"
            ) from e
        
        if start_dt > end_dt:
            raise ValueError(f"start_date must be before end_date. Got: {start_date} > {end_date}")
        
        # Generate list of years, months, and days for the actual date range
        years = set()
        months = set()
        days = set()
        
        current = start_dt
        while current <= end_dt:
            years.add(str(current.year))
            months.add(f"{current.month:02d}")
            days.add(f"{current.day:02d}")
            current += timedelta(days=1)
        
        return sorted(list(years)), sorted(list(months)), sorted(list(days))
    
    def fetch_weather_data(
        self,
        latmin: float,
        latmax: float,
        lonmin: float,
        lonmax: float,
        start_date: str,
        end_date: str,
        variables: List[str],
        pressure_levels: Optional[List[int]] = None,
        output_file: Optional[str] = None,
        times: Optional[List[str]] = None,
        product_type: str = 'reanalysis',
        format: str = 'netcdf'
    ) -> str:
        """
        Fetch ECMWF ERA5 weather data.
        
        Args:
            latmin: Minimum latitude (-90 to 90).
            latmax: Maximum latitude (-90 to 90).
            lonmin: Minimum longitude (-180 to 180).
            lonmax: Maximum longitude (-180 to 180).
            start_date: Start date in format 'YYYY-MM-DD'.
            end_date: End date in format 'YYYY-MM-DD'.
            variables: List of atmospheric variables to fetch.
            pressure_levels: List of pressure levels in hPa. If None, uses
                           [1000, 925, 850, 700, 500, 300, 200].
            output_file: Path to save the downloaded data. If None, creates
                        a temporary file.
            times: List of times in format 'HH:MM' (e.g., ['00:00', '12:00']).
                  If None, fetches all available times.
            product_type: Type of product ('reanalysis' or 'ensemble_members').
            format: Output format ('netcdf' or 'grib').
        
        Returns:
            Path to the downloaded data file.
        
        Raises:
            ValueError: If input parameters are invalid.
            RuntimeError: If data retrieval fails.
        """
        logger.info("Starting ECMWF weather data fetch")
        
        # Validate region bounds
        validate_region_bounds(latmin, latmax, lonmin, lonmax)
        
        # Set default pressure levels if not provided
        if pressure_levels is None:
            pressure_levels = [1000, 925, 850, 700, 500, 300, 200]
        
        # Validate and normalize inputs
        pressure_level_strs = validate_pressure_levels(pressure_levels)

        # Validate variable names
        validate_variable_names(variables)

        # Parse date range
        years, months, days = self._parse_date_range(start_date, end_date)
        
        # Set default times if not provided (every 6 hours)
        if times is None:
            times = ['00:00', '06:00', '12:00', '18:00']
        
        # Create output file path
        if output_file is None:
            # Use NamedTemporaryFile to securely create the file
            temp_file = tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False)
            output_file = temp_file.name
            temp_file.close()
        
        # Construct the request
        request = {
            'product_type': product_type,
            'format': format,
            'variable': variables,
            'pressure_level': pressure_level_strs,
            'year': years,
            'month': months,
            'day': days,
            'time': times,
            'area': [latmax, lonmin, latmin, lonmax],  # [North, West, South, East]
        }
        
        logger.info(f"Fetching data with request: {request}")
        logger.info(f"Output file: {output_file}")
        
        try:
            # Retrieve the data
            result = self.client.retrieve(
                'reanalysis-era5-pressure-levels',
                request,
                output_file
            )
            
            # Extract request_id from the result object
            request_id = None
            if hasattr(result, 'request_id'):
                request_id = result.request_id
            elif hasattr(result, 'reply') and isinstance(result.reply, dict):
                request_id = result.reply.get('request_id')
            
            logger.info(f"Successfully downloaded data to {output_file}")
            if request_id:
                logger.info(f"Request ID: {request_id}")
            
            return output_file
        
        except Exception as e:
            logger.error(f"Failed to fetch weather data: {e}")
            raise RuntimeError(f"Failed to fetch weather data: {e}") from e
    
    def load_data(self, file_path: str) -> Dict:
        """
        Load fetched weather data from file.
        
        Args:
            file_path: Path to the data file (NetCDF or GRIB format).
        
        Returns:
            Dictionary containing the loaded data with keys:
                - 'variables': Dictionary of variable arrays
                - 'coordinates': Dictionary of coordinate arrays (lat, lon, level, time)
                - 'metadata': Dictionary of metadata
        
        Raises:
            ImportError: If xarray is not installed.
            FileNotFoundError: If file does not exist.
        """
        if not XARRAY_AVAILABLE:
            raise ImportError(
                "xarray package is required for loading data. "
                "Install it with: pip install xarray netCDF4"
            )
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Load the dataset
            ds = xr.open_dataset(file_path)
            
            # Extract data
            result = {
                'variables': {},
                'coordinates': {},
                'metadata': dict(ds.attrs)
            }
            
            # Extract coordinate variables
            for coord_name in ['latitude', 'longitude', 'level', 'time']:
                if coord_name in ds.coords:
                    result['coordinates'][coord_name] = ds[coord_name].values
            
            # Extract data variables
            for var_name in ds.data_vars:
                result['variables'][var_name] = ds[var_name].values
            
            logger.info(f"Successfully loaded data with variables: {list(result['variables'].keys())}")
            return result
        
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise RuntimeError(f"Failed to load data: {e}") from e


def create_api(api_key: Optional[str] = None, api_url: Optional[str] = None) -> ECMWFWeatherAPI:
    """
    Convenience function to create an ECMWF Weather API instance.
    
    Args:
        api_key: CDS API key. If not provided, will try to read from
                environment variable CDSAPI_KEY or ~/.cdsapirc file.
        api_url: CDS API URL. If not provided, uses default CDS URL.
    
    Returns:
        Initialized ECMWFWeatherAPI instance.
    """
    return ECMWFWeatherAPI(api_key=api_key, api_url=api_url)


if __name__ == "__main__":
    # Example usage
    print("ECMWF Weather API Module")
    print("========================")
    print("\nThis module provides an API for fetching ECMWF ERA5 weather data.")
    print("\nExample usage:")
    print("""
    from ecmwf_weather_api import ECMWFWeatherAPI
    
    # Initialize the API (requires API key)
    api = ECMWFWeatherAPI(api_key="your-api-key")
    
    # Fetch weather data
    output_file = api.fetch_weather_data(
        latmin=30.0, latmax=40.0,
        lonmin=-110.0, lonmax=-100.0,
        start_date="2024-01-01",
        end_date="2024-01-02",
        variables=["temperature", "u_component_of_wind", "v_component_of_wind"],
        pressure_levels=[1000, 925, 850, 700, 500]
    )
    
    # Load the downloaded data
    data = api.load_data(output_file)
    print(data['variables'].keys())
    """)
