#!/usr/bin/env python3
"""
Earth Location Configuration Generator

This script generates location-specific YAML configuration files from:
1. A location table (locations.yaml) with geographic bounds
2. A template YAML configuration file
3. Command-line arguments for customization

The location identifier should be a unique string that can contain letters,
numbers, dashes, underscores, but not special characters.

Usage:
    python generate_config.py <location-id> [options]

Examples:
    # Generate config for Ann Arbor with defaults
    python generate_config.py ann-arbor

    # Generate config with custom time window
    python generate_config.py ann-arbor --start-date 2025-11-01 --end-date 2025-11-03

    # Generate config with custom resolution
    python generate_config.py white-sands --nx2 500 --nx3 400

    # Generate config with all custom parameters
    python generate_config.py ann-arbor \\
        --start-date 2025-12-01 \\
        --end-date 2025-12-02 \\
        --nx1 200 --nx2 300 --nx3 300 \\
        --output custom_ann_arbor.yaml
"""

import argparse
import csv
import io
import re
import sys
from pathlib import Path
import yaml


def load_locations(locations_file):
    """Load location definitions from CSV file."""
    locations = {}
    with open(locations_file, 'r') as f:
        # Skip comment lines
        lines = [line for line in f if not line.strip().startswith('#')]
        
    # Parse CSV from non-comment lines (comma-delimited)
    csv_data = io.StringIO(''.join(lines))
    reader = csv.DictReader(csv_data, delimiter=',', skipinitialspace=True)
    
    for row in reader:
        # Extract bounding box coordinates
        latmin = float(row['Latmin'])
        latmax = float(row['Latmax'])
        lonmin = float(row['Lonmin'])
        lonmax = float(row['Lonmax'])
        
        # Create polygon from bounding box (counterclockwise rectangle)
        polygon = [
            [lonmin, latmin],  # SW corner
            [lonmin, latmax],  # NW corner
            [lonmax, latmax],  # NE corner
            [lonmax, latmin],  # SE corner
        ]
        
        # Use Name field as location_id and Description as the display name
        location_id = row['Name']
        name = row['Description']
        
        locations[location_id] = {
            'name': name,
            'polygon': polygon
        }
    
    return locations


def calculate_center(polygon):
    """Calculate the center point from polygon vertices."""
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    return {
        'longitude': sum(lons) / len(lons),
        'latitude': sum(lats) / len(lats)
    }


def calculate_domain_extents(polygon):
    """
    Calculate domain extents (x2, x3) from polygon bounds in meters.
    
    Uses approximate conversion: 1 degree latitude = 111 km, 
    1 degree longitude = 111 km * cos(latitude).
    
    Returns:
        dict: {'x2_extent': north-south extent in meters, 
               'x3_extent': east-west extent in meters}
    """
    import math
    
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    
    # Get bounds
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    
    # Calculate north-south extent (latitude)
    # 1 degree latitude ≈ 111 km
    lat_extent_deg = lat_max - lat_min
    x2_extent = lat_extent_deg * 111000.0  # meters
    
    # Calculate east-west extent (longitude)
    # 1 degree longitude = 111 km * cos(latitude)
    # Use average latitude for the calculation
    avg_lat = (lat_min + lat_max) / 2.0
    lon_extent_deg = lon_max - lon_min
    x3_extent = lon_extent_deg * 111000.0 * math.cos(math.radians(avg_lat))  # meters
    
    return {
        'x2_extent': x2_extent,
        'x3_extent': x3_extent
    }


def load_template(template_file):
    """Load the template configuration file."""
    with open(template_file, 'r') as f:
        return f.read()


def validate_location_id(location_id):
    """
    Validate that location ID contains only allowed characters.
    
    Allowed: letters, numbers, dashes, underscores
    Not allowed: special characters, spaces
    """
    if not re.match(r'^[a-zA-Z0-9_-]+$', location_id):
        raise ValueError(
            f"Invalid location ID '{location_id}'. "
            "Location IDs can only contain letters, numbers, dashes, and underscores."
        )


def validate_date_format(date_str):
    """Validate date format is YYYY-MM-DD."""
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        raise ValueError(
            f"Invalid date format '{date_str}'. "
            "Dates must be in YYYY-MM-DD format."
        )


def generate_config(location_id, locations, template, args):
    """
    Generate a configuration file for the specified location.
    
    Args:
        location_id: Location identifier
        locations: Dictionary of location definitions
        template: Template string with placeholders
        args: Parsed command-line arguments
        
    Returns:
        Generated configuration string
    """
    # Validate location exists
    if location_id not in locations:
        available = ', '.join(sorted(locations.keys()))
        raise ValueError(
            f"Location '{location_id}' not found. "
            f"Available locations: {available}"
        )
    
    location = locations[location_id]
    
    # Check required arguments
    if not args.start_date:
        raise ValueError("--start-date is required")
    if not args.end_date:
        raise ValueError("--end-date is required")
    if not args.nx1:
        raise ValueError("--nx1 (vertical cells) is required")
    if not args.nx2:
        raise ValueError("--nx2 (north-south cells) is required")
    if not args.nx3:
        raise ValueError("--nx3 (east-west cells) is required")
    
    start_date = args.start_date
    end_date = args.end_date
    
    # Validate dates
    validate_date_format(start_date)
    validate_date_format(end_date)
    
    # Grid resolution (all required)
    nx1 = args.nx1
    nx2 = args.nx2
    nx3 = args.nx3
    nghost = args.nghost if args.nghost is not None else 3  # Default nghost to 3
    
    # Domain height
    x1_max = args.x1_max
    
    # Calculate horizontal extents from polygon bounds (or use overrides if provided)
    domain_extents = calculate_domain_extents(location['polygon'])
    x2_extent = args.x2_extent if args.x2_extent is not None else domain_extents['x2_extent']
    x3_extent = args.x3_extent if args.x3_extent is not None else domain_extents['x3_extent']
    
    # Time limit (default to 86400 seconds if not provided)
    tlim = args.tlim
    
    # Calculate center from polygon
    center = calculate_center(location['polygon'])
    
    # Build replacement dictionary
    replacements = {
        'location_name': location['name'],
        'location_description': f"Location: {location['name']}",
        'center_latitude': center['latitude'],
        'center_longitude': center['longitude'],
        'x1_max': x1_max,
        'x2_extent': x2_extent,
        'x3_extent': x3_extent,
        'x1_max_km': x1_max / 1000.0,
        'x2_extent_km': x2_extent / 1000.0,
        'x3_extent_km': x3_extent / 1000.0,
        'nx1': nx1,
        'nx2': nx2,
        'nx3': nx3,
        'nghost': nghost,
        'start_date': start_date,
        'end_date': end_date,
        'tlim': tlim,
    }
    
    # Replace placeholders in template
    config = template
    for key, value in replacements.items():
        placeholder = '{' + key + '}'
        config = config.replace(placeholder, str(value))
    
    return config


def list_locations(locations):
    """Print available locations."""
    print("\nAvailable Locations:")
    print("=" * 70)
    
    for loc_id, loc_data in sorted(locations.items()):
        print(f"\n{loc_id}")
        print(f"  Name: {loc_data['name']}")
        
        # Calculate and show center
        center = calculate_center(loc_data['polygon'])
        print(f"  Center: {center['latitude']:.2f}°N, "
              f"{abs(center['longitude']):.2f}°"
              f"{'W' if center['longitude'] < 0 else 'E'}")
        
        # Show polygon bounds
        polygon = loc_data['polygon']
        lons = [p[0] for p in polygon]
        lats = [p[1] for p in polygon]
        print(f"  Bounds: {min(lats)}°N to {max(lats)}°N, "
              f"{abs(max(lons))}°W to {abs(min(lons))}°W")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate location-specific YAML configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'location_id',
        nargs='?',
        help="Location identifier (e.g., 'ann-arbor', 'white-sands')"
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help="List available locations and exit"
    )
    
    parser.add_argument(
        '--start-date',
        help="Start date for simulation (YYYY-MM-DD format) [REQUIRED unless --list]"
    )
    
    parser.add_argument(
        '--end-date',
        help="End date for simulation (YYYY-MM-DD format) [REQUIRED unless --list]"
    )
    
    parser.add_argument(
        '--nx1',
        type=int,
        help="Number of interior cells in vertical direction [REQUIRED unless --list]"
    )
    
    parser.add_argument(
        '--nx2',
        type=int,
        help="Number of interior cells in north-south direction [REQUIRED unless --list]"
    )
    
    parser.add_argument(
        '--nx3',
        type=int,
        help="Number of interior cells in east-west direction [REQUIRED unless --list]"
    )
    
    parser.add_argument(
        '--nghost',
        type=int,
        default=3,
        help="Number of ghost cells on each side (default: 3)"
    )
    
    parser.add_argument(
        '--x1-max',
        type=float,
        default=20000.0,
        help="Vertical extent in meters [default: 20000]"
    )
    
    parser.add_argument(
        '--x2-extent',
        type=float,
        help="North-south extent in meters [OPTIONAL - calculated from polygon if not specified]"
    )
    
    parser.add_argument(
        '--x3-extent',
        type=float,
        help="East-west extent in meters [OPTIONAL - calculated from polygon if not specified]"
    )
    
    parser.add_argument(
        '--tlim',
        type=int,
        default=86400,
        help="Simulation time limit in seconds [default: 86400 (1 day)]"
    )
    
    parser.add_argument(
        '--output',
        help="Output file path (default: <location_id>.yaml)"
    )
    
    parser.add_argument(
        '--locations-file',
        default='locations.csv',
        help="Path to locations table file (default: locations.csv)"
    )
    
    parser.add_argument(
        '--template-file',
        default='config_template.yaml',
        help="Path to template file (default: config_template.yaml)"
    )
    
    args = parser.parse_args()
    
    # Determine script directory
    script_dir = Path(__file__).parent
    
    # Resolve file paths
    locations_file = Path(args.locations_file)
    if not locations_file.is_absolute():
        locations_file = script_dir / locations_file
    
    template_file = Path(args.template_file)
    if not template_file.is_absolute():
        template_file = script_dir / template_file
    
    # Check if files exist
    if not locations_file.exists():
        print(f"ERROR: Locations file not found: {locations_file}")
        return 1
    
    if not template_file.exists():
        print(f"ERROR: Template file not found: {template_file}")
        return 1
    
    # Load data
    try:
        locations = load_locations(locations_file)
        template = load_template(template_file)
    except Exception as e:
        print(f"ERROR: Failed to load files: {e}")
        return 1
    
    # Handle --list option
    if args.list:
        list_locations(locations)
        return 0
    
    # Require location_id if not listing
    if not args.location_id:
        parser.print_help()
        print("\nERROR: location_id is required (or use --list to see available locations)")
        return 1
    
    location_id = args.location_id
    
    # Validate location ID
    try:
        validate_location_id(location_id)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    
    # Generate configuration
    try:
        config = generate_config(location_id, locations, template, args)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f"{location_id}.yaml")
    
    # Write configuration file
    try:
        output_file.write_text(config)
        print(f"✓ Configuration file generated: {output_file}")
        print(f"  Location: {locations[location_id]['name']}")
        print(f"  Time window: {args.start_date} to {args.end_date}")
        print(f"  Grid: {args.nx1} × {args.nx2} × {args.nx3} cells")
    except Exception as e:
        print(f"ERROR: Failed to write output file: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
