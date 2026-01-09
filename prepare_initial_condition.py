#!/usr/bin/env python3
"""
Unified Earth Weather Data Download and Processing Script

This script downloads and processes ERA5 weather data for any configured
location using the complete ECMWF data curation pipeline.

The script executes all steps of the pipeline automatically:
Step 1: Fetch ERA5 data (dynamics and densities)
Step 2: Calculate air density from downloaded data
Step 3: Regrid to Cartesian coordinates
Step 4: Compute hydrostatic pressure
Step 5: Domain decomposition (optional, use --nX, --nY flag)
Step 6: Convert splitted NetCDF to PyTorch tensors (restart files)

Usage:
    python prepare_initial_condition.py <location-id> [options]

Examples:
    # Download Ann Arbor data with defaults
    python prepare_initial_condition.py ann-arbor

    # Download White Sands data with defaults
    python prepare_initial_condition.py white-sands

    # Use custom configuration file
    python prepare_initial_condition.py ann-arbor --config my_custom_config.yaml

    # Run only first 2 steps
    python prepare_initial_condition.py white-sands --stop-after 2
    
    # Run with domain decomposition into 4x4 blocks and convert to tensors
    python prepare_initial_condition.py ann-arbor --nX 4 --nY 4

    # Use custom timeout
    python prepare_initial_condition.py ann-arbor --timeout 7200

Requirements:
    - ECMWF CDS API credentials configured (~/.cdsapirc or CDSAPI_KEY env var)
    - Python packages: cdsapi, xarray, netCDF4, numpy, scipy, PyYAML, torch
    - See ecmwf_api/requirements.txt for complete list

For setup instructions, see:
    - ecmwf_api/README_ECMWF.md
    - README.md in location directories
"""

import argparse
import csv
import glob
import io
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path

# Add parent directory to path for importing ECMWF modules
SCRIPT_DIR = Path(__file__).parent
ECMWF_DIR = SCRIPT_DIR / "ecmwf_api"
sys.path.insert(0, str(ECMWF_DIR))


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


def check_cds_credentials():
    """Check if CDS API credentials are configured."""
    cdsapirc_path = Path.home() / ".cdsapirc"
    has_env_key = "CDSAPI_KEY" in os.environ
    has_config_file = cdsapirc_path.exists()
    
    if not has_env_key and not has_config_file:
        print("ERROR: CDS API credentials not found.")
        print()
        print("Please configure your ECMWF CDS API credentials:")
        print("1. Register at https://cds.climate.copernicus.eu/")
        print("2. Get your API key from https://cds.climate.copernicus.eu/how-to-api")
        print("3. Configure authentication:")
        print()
        print("   Option A: Set environment variable")
        print("   export CDSAPI_KEY='your-uid:your-api-key'")
        print()
        print("   Option B: Create ~/.cdsapirc file with:")
        print("   url: https://cds.climate.copernicus.eu/api")
        print("   key: your-uid:your-api-key")
        print()
        return False
    
    return True


def run_step_with_timeout(step_name, command, timeout_seconds=3600):
    """
    Run a pipeline step with error handling and timeout.
    
    Args:
        step_name: Name of the step for display
        command: Command to execute (list of arguments)
        timeout_seconds: Maximum time to wait for completion
        
    Returns:
        True if successful, False if failed or timed out
    """
    print(f"\n{'='*70}")
    print(f"RUNNING {step_name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(str(c) for c in command)}")
    print(f"Timeout: {timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")
    print()
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True,
            timeout=timeout_seconds
        )
        print(f"\n\033[92m[OK]\033[0m {step_name} completed successfully")
        return True
    except subprocess.TimeoutExpired:
        print(f"\n\033[91m[ERROR]\033[0m {step_name} timed out after {timeout_seconds} seconds")
        print(f"   Consider increasing timeout with --timeout option")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n\033[91m[ERROR]\033[0m {step_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"\n\033[91m[ERROR]\033[0m Command not found: {e}")
        print(f"Make sure the ECMWF scripts are in: {ECMWF_DIR}")
        return False


def find_output_directory(output_base):
    """
    Find the output directory created by Step 1.
    
    The directory name follows the pattern: LATMIN_LATMAX_LONMIN_LONMAX
    Example: 41.75N_42.85N_84.25W_83.15W
    
    Args:
        output_base: Base directory where output was created
        
    Returns:
        Path to output directory, or None if not found
    """
    # Look for directories with the expected pattern
    pattern = os.path.join(output_base, "*N_*N_*W_*W")
    matches = glob.glob(pattern)
    
    if not matches:
        # Also try patterns with E longitude
        pattern = os.path.join(output_base, "*N_*N_*E_*E")
        matches = glob.glob(pattern)
    
    if not matches:
        # Try mixed patterns
        pattern = os.path.join(output_base, "*N_*N_*")
        matches = glob.glob(pattern)
        # Filter to only valid lat-lon directory names
        matches = [m for m in matches if os.path.isdir(m) and 
                  any(c in os.path.basename(m) for c in ['N', 'S', 'E', 'W'])]
    
    if not matches:
        return None
    
    # If multiple matches, use the most recently created
    if len(matches) > 1:
        matches.sort(key=os.path.getmtime, reverse=True)
    
    return Path(matches[0])


def check_step1_files(output_dir):
    """Check if Step 1 output files exist."""
    if not output_dir.exists():
        return False
    
    # Look for dynamics and densities files
    dynamics_files = list(output_dir.glob("era5_hourly_dynamics_*.nc"))
    densities_files = list(output_dir.glob("era5_hourly_densities_*.nc"))
    
    return len(dynamics_files) > 0 and len(densities_files) > 0


def check_step2_files(output_dir):
    """Check if Step 2 output files exist."""
    if not output_dir.exists():
        return False
    
    # Look for density files
    density_files = list(output_dir.glob("era5_density_*.nc"))
    
    return len(density_files) > 0


def check_step3_files(output_dir, location_id, end_date):
    """Check if Step 3 output file exists."""
    regridded_file = output_dir / f"regridded_{location_id}_{end_date}.nc"
    return regridded_file.exists()


def check_step5_files(blocks_dir):
    """Check if Step 5 output files exist (decomposed blocks)."""
    if not blocks_dir.exists():
        return False
    
    # Look for block files
    block_files = list(blocks_dir.glob("*_block_*.nc"))
    
    return len(block_files) > 0


def check_step6_files(tensors_dir):
    """Check if Step 6 output files exist (PyTorch tensors)."""
    if not tensors_dir.exists():
        return False
    
    # Look for tensor files
    tensor_files = list(tensors_dir.glob("*_block_*.restart"))
    
    return len(tensor_files) > 0


def wait_for_files(check_function, output_dir, step_name, timeout_seconds=60,
                   check_interval=5, location_id=None, end_date=None):
    """
    Wait for files to appear after a step completes.
    
    Args:
        check_function: Function to check if files exist
        output_dir: Directory to check
        step_name: Name of the step (for logging)
        timeout_seconds: Maximum time to wait
        check_interval: Seconds between checks
        location_id: Optional location ID for step 3
        
    Returns:
        True if files appear, False if timeout
    """
    print(f"\nWaiting for {step_name} output files...")
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        if location_id:
            if check_function(output_dir, location_id, end_date):
                print(f"\033[92m[OK]\033[0m {step_name} output files found")
                return True
        else:
            if check_function(output_dir):
                print(f"\033[92m[OK]\033[0m {step_name} output files found")
                return True
        time.sleep(check_interval)
    
    print(f"\033[91m[ERROR]\033[0m Timeout waiting for {step_name} output files")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and process ERA5 weather data for configured locations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'location_id',
        help="Location identifier (e.g., 'ann-arbor', 'white-sands')"
    )
    
    parser.add_argument(
        "--config",
        help="Path to YAML configuration file (default: <location-id>.yaml in location directory)"
    )
    
    parser.add_argument(
        "--output-base",
        default=".",
        help="Base directory for output files (default: current directory)"
    )

    parser.add_argument(
        "--start-from",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=1,
        help="Start from specified step (1-6)"
    )
    
    parser.add_argument(
        "--stop-after",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=6,
        help="Stop after specified step (1-6)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout for each step in seconds (default: 3600)"
    )
    
    parser.add_argument(
        '--locations-file',
        default='locations.csv',
        help="Path to locations table file (default: locations.csv)"
    )
    
    parser.add_argument(
        '--nY',
        type=int,
        default=1,
        help="Number of blocks in x2 (Y, North-South) direction for decomposition (default: 1)"
    )
    
    parser.add_argument(
        '--nX',
        type=int,
        default=1,
        help="Number of blocks in x3 (X, East-West) direction for decomposition (default: 1)"
    )
    
    args = parser.parse_args()
    
    location_id = args.location_id
    
    # Resolve locations file path
    locations_file = Path(args.locations_file)
    if not locations_file.is_absolute():
        locations_file = SCRIPT_DIR / locations_file
    
    # Load locations to get location name
    try:
        locations = load_locations(locations_file)
        if location_id not in locations:
            available = ', '.join(sorted(locations.keys()))
            print(f"ERROR: Location '{location_id}' not found.")
            print(f"Available locations: {available}")
            return 1
        location_name = locations[location_id]['name']
    except Exception as e:
        print(f"ERROR: Failed to load locations file: {e}")
        return 1
    
    # Resolve config path
    if args.config:
        config_path = Path(args.config)
    else:
        # Try location-specific subdirectory first
        config_path = SCRIPT_DIR / location_id / f"{location_id}.yaml"
        if not config_path.exists():
            # Try current directory
            config_path = SCRIPT_DIR / f"{location_id}.yaml"
    
    if not config_path.is_absolute():
        config_path = SCRIPT_DIR / config_path

    output_base = Path(args.output_base).resolve()
    
    # Check if config file exists
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        print(f"\nTip: Generate a config file using:")
        print(f"  python generate_config.py {location_id}")
        return 1

    # grab end date string from config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
        # check end-date key in config/integration
        if 'integration' not in config_data:
            raise KeyError("Missing 'integration' section in config file")
        if 'end-date' not in config_data['integration']:
            raise KeyError("Missing 'end-date' key in 'integration' section of config file")
        # datetime.date YYYY-MM-DD to YYYYMMDD
        end_date = config_data['integration'].get('end-date').strftime('%Y%m%d')
    
    print("="*70)
    print(f"{location_name} Weather Data Pipeline")
    print("="*70)
    print(f"Location ID: {location_id}")
    print(f"Configuration: {config_path}")
    print(f"End date: {end_date}")
    print(f"Output base: {output_base}")
    print(f"Timeout per step: {args.timeout} seconds ({args.timeout/60:.1f} minutes)")
    print(f"Will start from: Step {args.start_from}")
    print(f"Will stop after: Step {args.stop_after}")
    print()

    # Check credentials before starting
    if not check_cds_credentials():
        return 1

    ####### directory structures #######
    fetch_script = ECMWF_DIR / "fetch_era5_pipeline.py"
    density_script = ECMWF_DIR / "calculate_density.py"
    regrid_script = ECMWF_DIR / "regrid_era5_to_cartesian.py"
    pressure_script = ECMWF_DIR / "compute_hydrostatic_pressure.py"
    decompose_script = ECMWF_DIR / "decompose_domain.py"
    convert_script = ECMWF_DIR / "convert_netcdf_to_tensor.py"
    ####################################
    
    # Step 1: Fetch ERA5 data
    if args.start_from <= 1:
        step1_success = run_step_with_timeout(
            "Step 1: Fetch ERA5 Data",
            ["python3", str(fetch_script), str(config_path), 
             "--output-base", str(output_base)],
            timeout_seconds=args.timeout
        )
        
        if not step1_success:
            print("\n033[91m[ERROR]\033[0m Pipeline failed at Step 1")
            return 1

        print("\nLocating output directory...")
        output_dir = find_output_directory(output_base)

        if not output_dir:
            print("\033[91m[ERROR]\033[0m Could not find output directory")
            print("  Expected directory pattern: LATMIN_LATMAX_LONMIN_LONMAX")
            return 1
        
        print(f"\033[92m[OK]\033[0m Found output directory: {output_dir}")
        
        # Wait for Step 1 files to be fully written
        if not wait_for_files(check_step1_files, output_dir, "Step 1", timeout_seconds=60):
            print("\033[91m[ERROR]\033[0m Step 1 output files not found")
            return 1

    ####### directory structures #######
    blocks_dir = output_dir / f"regridded_{location_id}_{end_date}_blocks"
    regridded_output = output_dir / f"regridded_{location_id}_{end_date}.nc"
    tensors_dir = output_dir / f"regridded_{location_id}_{end_date}_tensors"
    ####################################
    
    if args.stop_after == 1:
        print("\n" + "="*70)
        print("Stopped after Step 1 as requested")
        print("="*70)
        return 0
    
    # Step 2: Calculate air density
    if args.start_from <= 2:
        step2_success = run_step_with_timeout(
            "Step 2: Calculate Air Density",
            ["python3", str(density_script),
             "--input-dir", str(output_dir),
             "--output-dir", str(output_dir)],
            timeout_seconds=args.timeout
        )
        
        if not step2_success:
            print("\n\033[91m[ERROR]\033[0m Pipeline failed at Step 2")
            return 1
        
        # Wait for Step 2 files
        if not wait_for_files(check_step2_files, output_dir, "Step 2", timeout_seconds=30):
            print("\033[91m[ERROR]\033[0m Step 2 output files not found")
            return 1
    
    if args.stop_after == 2:
        print("\n" + "="*70)
        print("Stopped after Step 2 as requested")
        print("="*70)
        return 0
    
    # Step 3: Regrid to Cartesian coordinates
    if args.start_from <= 3:
        step3_success = run_step_with_timeout(
            "Step 3: Regrid to Cartesian",
            ["python3", str(regrid_script),
             str(config_path), str(output_dir),
             "--output", str(regridded_output)],
            timeout_seconds=args.timeout
        )
        
        if not step3_success:
            print("\n\033[91m[ERROR]\033[0m Pipeline failed at Step 3")
            return 1
        
        # Wait for Step 3 file
        if not wait_for_files(check_step3_files, output_dir, "Step 3",
                              timeout_seconds=30, location_id=location_id,
                              end_date=end_date):
            print("\033[91m[ERROR]\033[0m Step 3 output file not found")
            return 1
    
    if args.stop_after == 3:
        print("\n" + "="*70)
        print("Stopped after Step 3 as requested")
        print("="*70)
        return 0
    
    # Step 4: Compute hydrostatic pressure
    if args.start_from <= 4:
        step4_success = run_step_with_timeout(
            "Step 4: Compute Hydrostatic Pressure",
            ["python3", str(pressure_script),
             str(config_path), str(regridded_output)],
            timeout_seconds=args.timeout
        )
        
        if not step4_success:
            print("\n\033[91m[ERROR]\033[0m Pipeline failed at Step 4")
            return 1
    
    if args.stop_after == 4:
        print("\n" + "="*70)
        print("Stopped after Step 4 as requested")
        print("="*70)
        return 0
    
    # Step 5: Domain decomposition (optional)
    if args.start_from <= 5:
        step5_success = run_step_with_timeout(
            "Step 5: Domain Decomposition",
            ["python3", str(decompose_script),
             str(regridded_output),
             str(args.nY),
             str(args.nX),
             "--output-dir", str(blocks_dir)],
            timeout_seconds=args.timeout
        )
        
        if not step5_success:
            print("\n\033[91m[ERROR]\033[0m Pipeline failed at Step 5")
            return 1
        
        # Wait for Step 5 files
        if not wait_for_files(check_step5_files, blocks_dir, "Step 5", timeout_seconds=30):
            print("\033[91m[ERROR]\033[0m Step 5 output files not found")
            return 1
        
    if args.stop_after == 5:
        print("\n" + "="*70)
        print("Stopped after Step 5 as requested")
        print("="*70)
        return 0
        
    # Step 6: Convert NetCDF to PyTorch tensors
    if args.start_from <= 6:
        step6_success = run_step_with_timeout(
            "Step 6: Convert to PyTorch Tensors",
            ["python3", str(convert_script),
             str(blocks_dir),
             "--output-dir", str(tensors_dir)],
            timeout_seconds=args.timeout
        )
        
        if not step6_success:
            print("\n\033[91m[ERROR]\033[0m Pipeline failed at Step 6")
            return 1
        
        # Wait for Step 6 files
        if not wait_for_files(check_step6_files, tensors_dir, "Step 6", timeout_seconds=30):
            print("\033[91m[ERROR]\033[0m Step 6 output files not found")
            return 1
    
    # Success!
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print()
    print(f"\033[92m[OK]\033[0m All data has been processed and saved to: {output_dir}")
    print()
    print("Output files:")
    print(f"  - era5_hourly_dynamics_*.nc (Step 1)")
    print(f"  - era5_hourly_densities_*.nc (Step 1)")
    print(f"  - era5_density_*.nc (Step 2)")
    print(f"  - regridded_{location_id}_{end_date}.nc (Step 3 & 4)")
    
    if args.stop_after == 5 or args.stop_after == 6:
        print(f"  - regridded_{location_id}_{end_date}_blocks/*_block_*_*.nc (Step 5)")
        if args.stop_after != 5:
            print(f"  - regridded_{location_id}_{end_date}_tensors/*_block_*.restart(Step 6)")
            print()
            print(f"\033[92m[OK]\033[0m The PyTorch tensor files in regridded_{location_id}_{end_date}_tensors/ are ready for {location_name} simulations.")
        else:
            print()
            print(f"\033[92m[OK]\033[0m The regridded_{location_id}_{end_date}.nc file and decomposed blocks are ready for {location_name} simulations.")
    else:
        print()
        print(f"\033[92m[OK]\033[0m The regridded_{location_id}_{end_date}.nc file is ready for {location_name} simulations.")
    
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
