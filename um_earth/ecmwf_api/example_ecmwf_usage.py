"""
Example usage of the ECMWF Weather API

This script demonstrates how to use the ECMWF Weather API to fetch
and process weather data from the Climate Data Store.

Before running this script:
1. Install required packages: pip install -r requirements.txt
2. Register for a CDS API key at: https://cds.climate.copernicus.eu/
3. Set up your API key either:
   - As an environment variable: export CDSAPI_KEY="your-key:your-uid"
   - In ~/.cdsapirc file:
     url: https://cds.climate.copernicus.eu/api/v2
     key: your-key:your-uid
"""

import os
from ecmwf_weather_api import ECMWFWeatherAPI


def example_basic_usage():
    """
    Basic example: Fetch temperature and wind data for a region.
    """
    print("=" * 60)
    print("Example 1: Basic Weather Data Fetch")
    print("=" * 60)
    
    # Initialize the API
    # Option 1: With explicit API key
    # api = ECMWFWeatherAPI(api_key="your-key:your-uid")
    
    # Option 2: Using environment variable or ~/.cdsapirc
    try:
        api = ECMWFWeatherAPI()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease set up your CDS API key:")
        print("1. Register at: https://cds.climate.copernicus.eu/")
        print("2. Get your API key from: https://cds.climate.copernicus.eu/user")
        print("3. Set environment variable: export CDSAPI_KEY='your-key:your-uid'")
        return
    
    # Define the region of interest (White Sands, New Mexico area)
    latmin = 32.0
    latmax = 33.5
    lonmin = -106.8
    lonmax = -105.8
    
    # Define the time window
    start_date = "2024-01-01"
    end_date = "2024-01-02"
    
    # Define atmospheric variables
    variables = [
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "geopotential"
    ]
    
    # Define pressure levels (in hPa)
    pressure_levels = [1000, 925, 850, 700, 500, 300]
    
    print(f"\nFetching data for:")
    print(f"  Region: lat [{latmin}, {latmax}], lon [{lonmin}, {lonmax}]")
    print(f"  Time: {start_date} to {end_date}")
    print(f"  Variables: {variables}")
    print(f"  Pressure levels: {pressure_levels} hPa")
    print("\nThis may take a few minutes...")
    
    try:
        # Fetch the data
        output_file = api.fetch_weather_data(
            latmin=latmin,
            latmax=latmax,
            lonmin=lonmin,
            lonmax=lonmax,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            pressure_levels=pressure_levels,
            output_file="white_sands_weather.nc"
        )
        
        print(f"\n✓ Data successfully downloaded to: {output_file}")
        
        # Load and inspect the data
        data = api.load_data(output_file)
        
        print(f"\nData summary:")
        print(f"  Variables: {list(data['variables'].keys())}")
        print(f"  Coordinates: {list(data['coordinates'].keys())}")
        
        for coord_name, coord_values in data['coordinates'].items():
            print(f"  {coord_name}: shape {coord_values.shape}, range [{coord_values.min():.2f}, {coord_values.max():.2f}]")
        
        return data
    
    except Exception as e:
        print(f"\n✗ Error fetching data: {e}")
        return None


def example_custom_times():
    """
    Example: Fetch data for specific times of day.
    """
    print("\n" + "=" * 60)
    print("Example 2: Fetch Data for Specific Times")
    print("=" * 60)
    
    try:
        api = ECMWFWeatherAPI()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print("\nFetching data only at 00:00 and 12:00 UTC...")
    
    try:
        output_file = api.fetch_weather_data(
            latmin=30.0,
            latmax=35.0,
            lonmin=-110.0,
            lonmax=-105.0,
            start_date="2024-01-01",
            end_date="2024-01-01",
            variables=["temperature"],
            pressure_levels=[850, 500],
            times=["00:00", "12:00"],
            output_file="specific_times_weather.nc"
        )
        
        print(f"✓ Data downloaded to: {output_file}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_multiple_regions():
    """
    Example: Fetch data for multiple regions sequentially.
    """
    print("\n" + "=" * 60)
    print("Example 3: Fetch Data for Multiple Regions")
    print("=" * 60)
    
    try:
        api = ECMWFWeatherAPI()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    regions = {
        "white_sands": {"latmin": 32.0, "latmax": 33.5, "lonmin": -106.8, "lonmax": -105.8},
        "grand_canyon": {"latmin": 35.8, "latmax": 36.5, "lonmin": -112.5, "lonmax": -111.8},
    }
    
    for region_name, bounds in regions.items():
        print(f"\nFetching data for {region_name}...")
        
        try:
            output_file = api.fetch_weather_data(
                latmin=bounds["latmin"],
                latmax=bounds["latmax"],
                lonmin=bounds["lonmin"],
                lonmax=bounds["lonmax"],
                start_date="2024-01-01",
                end_date="2024-01-01",
                variables=["temperature"],
                pressure_levels=[850],
                output_file=f"{region_name}_weather.nc"
            )
            
            print(f"  ✓ Downloaded to: {output_file}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")


def example_available_variables():
    """
    Display available variable mappings.
    """
    print("\n" + "=" * 60)
    print("Available Variable Aliases")
    print("=" * 60)
    
    print("\nCommon variable names and their ECMWF equivalents:")
    print("  'temperature' or 'temp' -> 'temperature'")
    print("  'u' or 'u_wind' -> 'u_component_of_wind'")
    print("  'v' or 'v_wind' -> 'v_component_of_wind'")
    print("  'humidity' or 'relative_humidity' -> 'relative_humidity'")
    print("  'specific_humidity' -> 'specific_humidity'")
    print("  'geopotential' -> 'geopotential'")
    
    print("\nFor more variables, see:")
    print("https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels")


def main():
    """
    Main function to run examples.
    """
    print("\n" + "=" * 60)
    print("ECMWF Weather API - Example Usage")
    print("=" * 60)
    
    # Check if API key is configured
    api_key_configured = (
        os.environ.get('CDSAPI_KEY') is not None or
        os.path.exists(os.path.expanduser('~/.cdsapirc'))
    )
    
    if not api_key_configured:
        print("\n⚠ Warning: CDS API key not found!")
        print("\nTo use this API, you need to:")
        print("1. Register at: https://cds.climate.copernicus.eu/")
        print("2. Get your API key from: https://cds.climate.copernicus.eu/user")
        print("3. Set up credentials:")
        print("   Option A: Set environment variable:")
        print("     export CDSAPI_KEY='your-key:your-uid'")
        print("   Option B: Create ~/.cdsapirc file:")
        print("     url: https://cds.climate.copernicus.eu/api/v2")
        print("     key: your-key:your-uid")
        print("\nExample only mode - showing available functionality:\n")
        example_available_variables()
        return
    
    # Run examples
    print("\nNote: These examples will make real API requests to CDS.")
    print("Data download may take several minutes.\n")
    
    # Run basic example
    example_basic_usage()
    
    # Uncomment to run additional examples:
    # example_custom_times()
    # example_multiple_regions()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
