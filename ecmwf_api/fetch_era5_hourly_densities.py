#!/usr/bin/env python3
"""
Convenient script to download density-related variables from ERA5.

This script downloads density variables at all standard ERA5 pressure levels for a specified region and time period.

The download is split into separate jobs - one per day - which run in parallel
for faster data retrieval. Each day is saved as a separate NetCDF file.
By default, the files are saved in the current directory with filenames being:
    era5_hourly_densities_YYYYMMDD.nc

Variables downloaded:
    1. Specific cloud ice water content
    2. Specific humidity
    3. Specific snow water content
    4. Specific cloud liquid water content
    5. Specific rain water content
    6. Fraction of cloud cover
    7. Relative humidity

Usage:
    python fetch_era5_hourly_densities.py \
            --latmin 30.0 --latmax 40.0 \
            --lonmin -110.0 --lonmax -100.0 \
            --start-date 2024-01-01 \
            --end-date 2024-01-02 \
            --output ./output_dir

Requirements:
    - ECMWF CDS API key configured (see README_ECMWF.md)
    - Required packages: cdsapi, netCDF4, numpy
"""

import argparse
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add current directory to path for importing local module
sys.path.insert(0, os.path.dirname(__file__))

from ecmwf_weather_api import ECMWFWeatherAPI
from ecmwf_utils import (
        validate_date_format,
        add_common_arguments,
        generate_date_list,
        fetch_single_day
        )

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Download ERA5 density variables at all pressure levels.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Variables downloaded:
  1. Specific cloud ice water content
  2. Specific humidity
  3. Specific snow water content
  4. Specific cloud liquid water content
  5. Specific rain water content
  6. Fraction of cloud cover
  7. Relative humidity

Examples:
  # Download data for White Sands, New Mexico
  python fetch_era5_hourly_densities.py --latmin 32.0 --latmax 33.5 \\
                                    --lonmin -106.8 --lonmax -105.8 \\
                                    --start-date 2024-01-01 \\
                                    --end-date 2024-01-02 \\
                                    --output ./white_sands

  # Download data for a larger region with specific times
  python fetch_era5_hourly_densities.py --latmin 30.0 --latmax 40.0 \\
                                    --lonmin -120.0 --lonmax -100.0 \\
                                    --start-date 2024-01-01 \\
                                    --end-date 2024-01-03 \\
                                    --times 00:00 12:00 \\
                                    --output ./30N_40N_100W_120W

Note: Download may take several minutes depending on request size and CDS server load.
        """
    )

    parser = add_common_arguments(parser)
    return parser.parse_args()

def main():
    """Main function to download ERA5 density-related variables."""
    args = parse_arguments()
    
    # Validate date formats
    if not validate_date_format(args.start_date):
        print(f"Error: Invalid start date format '{args.start_date}'. Use YYYY-MM-DD.")
        sys.exit(1)
    
    if not validate_date_format(args.end_date):
        print(f"Error: Invalid end date format '{args.end_date}'. Use YYYY-MM-DD.")
        sys.exit(1)
    
    print("=" * 70)
    print("ERA5 Hourly Density Variables Download")
    print("=" * 70)
    print(f"\nRegion:")
    print(f"  Latitude:  [{args.latmin}, {args.latmax}]")
    print(f"  Longitude: [{args.lonmin}, {args.lonmax}]")
    print(f"\nTime Period:")
    print(f"  Start: {args.start_date}")
    print(f"  End:   {args.end_date}")
    print(f"\nVariables:")
    print(f"  1. Specific cloud ice water content")
    print(f"  2. Specific humidity")
    print(f"  3. Specific snow water content")
    print(f"  4. Specific cloud liquid water content")
    print(f"  5. Specific rain water content")
    print(f"  6. Fraction of cloud cover")
    print(f"  7. Relative humidity")
    print(f"\nPressure Levels: All 37 standard ERA5 levels")
    print(f"  (1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,")
    print(f"   225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,")
    print(f"   775, 800, 825, 850, 875, 900, 925, 950, 975, 1000 hPa)")
    
    if args.times:
        print(f"\nTimes: {', '.join(args.times)}")
    else:
        print(f"\nTimes: All available (00:00, 06:00, 12:00, 18:00 UTC)")
    
    print(f"\nOutput folder: {args.output}")
    print("\n" + "=" * 70)
    print("\nInitializing ECMWF API...")
    
    try:
        # Initialize the API
        api = ECMWFWeatherAPI(api_key=args.api_key, api_url=args.api_url)
        
        print("✓ API initialized successfully")
        
        # Define density-related variables to download
        # These are the official ECMWF ERA5 variable names
        variables = [
            'specific_cloud_ice_water_content',
            'specific_humidity',
            'specific_snow_water_content',
            'specific_cloud_liquid_water_content',
            'specific_rain_water_content',
            'fraction_of_cloud_cover',
            'relative_humidity'
        ]
        
        # All standard ERA5 pressure levels
        pressure_levels = [
            1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
            225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
            775, 800, 825, 850, 875, 900, 925, 950, 975, 1000
        ]
        
        # Generate list of dates to download
        dates = generate_date_list(args.start_date, args.end_date)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)
        
        print(f"\nDownloading {len(dates)} day(s) using {args.jobs} parallel jobs...")
        print(f"(Each job downloads one day with all pressure levels. This may take several minutes.)")
        print(f"Output directory: {args.output}")
        
        # Download each day in parallel
        results = []
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {}
            for idx, date_str in enumerate(dates):
                future = executor.submit(
                    fetch_single_day,
                    api, date_str, variables, pressure_levels,
                    args.latmin, args.latmax, args.lonmin, args.lonmax,
                    args.times, args.output, idx, len(dates),
                    prefix='era5_hourly_densities'
                )
                futures[future] = date_str
            
            # Collect results as they complete
            for future in as_completed(futures):
                filepath = future.result()
                if filepath:
                    results.append(filepath)
        
        # Report summary
        print(f"\n{'='*70}")
        print(f"Download Summary:")
        print(f"{'='*70}")
        print(f"Total days requested: {len(dates)}")
        print(f"Successfully downloaded: {len(results)}")
        print(f"Failed: {len(dates) - len(results)}")
        
        if results:
            print(f"\nDownloaded files:")
            for filepath in sorted(results):
                filename = os.path.basename(filepath)
                print(f"  {filename}")
        
        if len(results) < len(dates):
            print(f"\n Warning: Some downloads failed")
            sys.exit(1)
        
        print("\n" + "=" * 70)
        print("Download completed successfully!")
        print("=" * 70)
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease check your input parameters and try again.")
        sys.exit(1)
    
    except RuntimeError as e:
        print(f"\n✗ Error: {e}")
        print("\nData retrieval failed. This could be due to:")
        print("  - Invalid CDS API credentials")
        print("  - Network connectivity issues")
        print("  - CDS server problems")
        print("  - Need to accept CDS license terms")
        print("\nPlease check your configuration and try again.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
