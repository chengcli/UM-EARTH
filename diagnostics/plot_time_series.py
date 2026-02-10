#!/usr/bin/env python3
"""
Plot time series data at a specific location given lat/lon coordinates.

This script generates time series plots for:
1. Lifting Condensation Level (LCL) vs time
2. PBL height vs time
3. PBL winds (top) vs time
4. PBL winds (middle) vs time
5. Surface wind vs time

Can process single file pairs or batch process all files in a directory.
"""

import argparse
import csv
import glob
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


# Constants
P0 = 101325.0  # Standard pressure in Pa
KAPPA = 0.286  # R/cp for dry air (dimensionless)
DEFAULT_RH = 0.7  # Default relative humidity (fraction)
ESPY_CONSTANT = 125.0  # meters per degree Celsius in simplified LCL calculation
PBL_THETA_THRESHOLD = 0.5  # K - potential temperature threshold for PBL detection
MAGNUS_A = 17.27  # Magnus formula constant (dimensionless)
MAGNUS_B = 237.7  # Magnus formula constant (degrees Celsius)


def read_location_bounds(locations_csv, location_name):
    """
    Read location bounds from the locations.csv file.
    
    Parameters
    ----------
    locations_csv : str
        Path to locations.csv file
    location_name : str
        Name of the location to look up
    
    Returns
    -------
    dict
        Dictionary with keys: Latmin, Latmax, Lonmin, Lonmax (all floats)
    """
    with open(locations_csv, 'r') as f:
        lines = [line for line in f if not line.strip().startswith('#')]
        reader = csv.DictReader(lines)
        for row in reader:
            if row['Name'].strip() == location_name:
                return {
                    'Latmin': float(row['Latmin']),
                    'Latmax': float(row['Latmax']),
                    'Lonmin': float(row['Lonmin']),
                    'Lonmax': float(row['Lonmax'])
                }
    raise ValueError(f"Location '{location_name}' not found in {locations_csv}")


def latlon_to_cartesian(lat, lon, bounds, x2, x3):
    """
    Convert lat/lon coordinates to cartesian grid indices.
    
    Parameters
    ----------
    lat : float
        Latitude of the point
    lon : float
        Longitude of the point
    bounds : dict
        Dictionary with Latmin, Latmax, Lonmin, Lonmax
    x2 : array
        X-coordinate values from NetCDF (west-east direction, in meters)
    x3 : array
        Y-coordinate values from NetCDF (south-north direction, in meters)
    
    Returns
    -------
    tuple
        (x2_idx, x3_idx, x2_frac, x3_frac) where idx is the lower index and frac is
        the fractional position for interpolation
    """
    # Compute normalized position within the domain
    # x2 corresponds to longitude (west-east)
    # x3 corresponds to latitude (south-north)
    
    lon_norm = (lon - bounds['Lonmin']) / (bounds['Lonmax'] - bounds['Lonmin'])
    lat_norm = (lat - bounds['Latmin']) / (bounds['Latmax'] - bounds['Latmin'])
    
    # Clamp to valid range
    lon_norm = np.clip(lon_norm, 0.0, 1.0)
    lat_norm = np.clip(lat_norm, 0.0, 1.0)
    
    # Convert to cartesian grid indices
    # x2 runs from 0 to max in west-east direction
    # x3 runs from 0 to max in south-north direction
    x2_pos = lon_norm * (len(x2) - 1)
    x3_pos = lat_norm * (len(x3) - 1)
    
    # Get integer indices and fractional parts for interpolation
    x2_idx = int(np.floor(x2_pos))
    x3_idx = int(np.floor(x3_pos))
    
    x2_frac = x2_pos - x2_idx
    x3_frac = x3_pos - x3_idx
    
    # Ensure indices are within bounds
    x2_idx = min(x2_idx, len(x2) - 2)
    x3_idx = min(x3_idx, len(x3) - 2)
    
    return x2_idx, x3_idx, x2_frac, x3_frac


def interpolate_2d(data, x2_idx, x3_idx, x2_frac, x3_frac):
    """
    Perform bilinear interpolation on 2D data.
    
    Parameters
    ----------
    data : array
        2D array with shape (x3, x2)
    x2_idx, x3_idx : int
        Lower indices
    x2_frac, x3_frac : float
        Fractional positions for interpolation
    
    Returns
    -------
    float
        Interpolated value
    """
    # Bilinear interpolation
    v00 = data[x3_idx, x2_idx]
    v10 = data[x3_idx, x2_idx + 1]
    v01 = data[x3_idx + 1, x2_idx]
    v11 = data[x3_idx + 1, x2_idx + 1]
    
    # Interpolate in x2 direction
    v0 = v00 * (1 - x2_frac) + v10 * x2_frac
    v1 = v01 * (1 - x2_frac) + v11 * x2_frac
    
    # Interpolate in x3 direction
    return v0 * (1 - x3_frac) + v1 * x3_frac


def extract_and_interpolate(dataset, var_name, time, x1, x3_idx, x2_idx, x2_frac, x3_frac):
    """
    Extract a variable from a dataset and interpolate to a specific location.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing the variable
    var_name : str
        Name of the variable to extract
    time : int
        Time index
    x1 : int
        Vertical level index
    x3_idx, x2_idx : int
        Lower horizontal indices
    x2_frac, x3_frac : float
        Fractional positions for interpolation
    
    Returns
    -------
    float
        Interpolated value at the specified location
    """
    # Extract 2x2 slice around the target location
    data = dataset[var_name].isel(time=time, x1=x1).values[x3_idx:x3_idx+2, x2_idx:x2_idx+2]
    # Indices are 0, 0 because data has been pre-sliced to a 2x2 interpolation window
    return interpolate_2d(data, 0, 0, x2_frac, x3_frac)


def calculate_lcl(temperature, pressure, rh):
    """
    Calculate the Lifting Condensation Level (LCL) height.
    
    Uses the approximation formula based on temperature and dewpoint.
    
    Parameters
    ----------
    temperature : float or array
        Temperature at surface (K)
    pressure : float or array
        Pressure at surface (Pa)
    rh : float or array
        Relative humidity (fraction 0-1)
    
    Returns
    -------
    float or array
        LCL height in meters
    """
    # Convert temperature to Celsius
    T_c = temperature - 273.15
    
    # Calculate dewpoint using Magnus formula
    rh_clipped = np.clip(rh, 0.01, 0.99)
    
    # Calculate dewpoint
    alpha = (MAGNUS_A * T_c) / (MAGNUS_B + T_c) + np.log(rh_clipped)
    dewpoint = (MAGNUS_B * alpha) / (MAGNUS_A - alpha)
    
    # Calculate LCL height using simplified Espy's formula
    lcl_height = ESPY_CONSTANT * (T_c - dewpoint)
    
    # Ensure non-negative heights
    lcl_height = np.maximum(lcl_height, 0.0)
    
    return lcl_height


def calculate_pbl_height(theta, x1):
    """
    Calculate PBL height.
    
    PBL height is defined by the lowest level at which the potential temperature
    change between two levels is greater than 0.5 K.
    
    Parameters
    ----------
    theta : array
        Potential temperature profile (K) with shape (nz,)
    x1 : array
        Vertical coordinate values (height in meters)
    
    Returns
    -------
    float
        PBL height in meters
    """
    # Calculate differences between adjacent levels
    theta_diff = np.diff(theta)
    
    # Find the first level where the difference exceeds PBL_THETA_THRESHOLD
    pbl_indices = np.where(theta_diff > PBL_THETA_THRESHOLD)[0]
    
    if len(pbl_indices) > 0:
        pbl_idx = pbl_indices[0]
        return x1[pbl_idx]
    else:
        # If no inversion found, return the top level
        return x1[-1]


def extract_time_series(input_file_out1, input_file_out2, lat, lon, 
                        location_name, locations_csv):
    """
    Extract time series data at a specific lat/lon location.
    
    Parameters
    ----------
    input_file_out1 : str
        Path to out1.nc file
    input_file_out2 : str
        Path to out2.nc file
    lat : float
        Latitude of the location
    lon : float
        Longitude of the location
    location_name : str
        Name of the location (for looking up bounds)
    locations_csv : str
        Path to locations.csv file
    
    Returns
    -------
    dict
        Dictionary containing time series data
    """
    # Load datasets
    ds1 = xr.open_dataset(input_file_out1)
    ds2 = xr.open_dataset(input_file_out2)
    
    # Read location bounds
    bounds = read_location_bounds(locations_csv, location_name)
    
    # Get coordinates
    x1 = ds1['x1'].values  # Height (m)
    x2 = ds1['x2'].values  # West-east (m)
    x3 = ds1['x3'].values  # South-north (m)
    time = ds1['time'].values  # Time (s)
    
    # Convert lat/lon to cartesian indices
    x2_idx, x3_idx, x2_frac, x3_frac = latlon_to_cartesian(lat, lon, bounds, x2, x3)
    
    print(f"Location: ({lat}, {lon})")
    print(f"Cartesian indices: x2={x2_idx} ({x2_frac:.2f}), x3={x3_idx} ({x3_frac:.2f})")
    print(f"Cartesian position: x2={x2[x2_idx]/1000:.2f} km, x3={x3[x3_idx]/1000:.2f} km")
    
    # Initialize result arrays
    nt = len(time)
    nz = len(x1)
    
    lcl_series = np.zeros(nt)
    pbl_height_series = np.zeros(nt)
    pbl_wind_top_series = np.zeros(nt)
    pbl_wind_middle_series = np.zeros(nt)
    surface_wind_series = np.zeros(nt)
    
    # Extract time series for each time step
    for t in range(nt):
        # Extract vertical profile at this location and time
        # Data shape is (time, x1, x3, x2)
        
        # For surface variables
        press_interp = extract_and_interpolate(ds1, 'press', t, 0, x3_idx, x2_idx, x2_frac, x3_frac)
        temp_interp = extract_and_interpolate(ds2, 'temp', t, 0, x3_idx, x2_idx, x2_frac, x3_frac)
        
        # Check for relative humidity (rh_H2O_l_ is the NetCDF variable name with trailing underscore)
        if 'rh_H2O_l_' in ds2.data_vars:
            rh_interp = extract_and_interpolate(ds2, 'rh_H2O_l_', t, 0, x3_idx, x2_idx, x2_frac, x3_frac)
        else:
            rh_interp = DEFAULT_RH
        
        # Calculate LCL
        lcl_series[t] = calculate_lcl(temp_interp, press_interp, rh_interp)
        
        # Extract vertical profiles for PBL calculation
        theta_profile = np.zeros(nz)
        vel2_profile = np.zeros(nz)
        vel3_profile = np.zeros(nz)
        
        for k in range(nz):
            if 'theta_v' in ds2.data_vars:
                theta_profile[k] = extract_and_interpolate(ds2, 'theta_v', t, k, x3_idx, x2_idx, x2_frac, x3_frac)
            else:
                # If theta_v not available, calculate from temp and press
                temp_k = extract_and_interpolate(ds2, 'temp', t, k, x3_idx, x2_idx, x2_frac, x3_frac)
                press_k = extract_and_interpolate(ds1, 'press', t, k, x3_idx, x2_idx, x2_frac, x3_frac)
                theta_profile[k] = temp_k * (P0 / press_k) ** KAPPA
            
            vel2_profile[k] = extract_and_interpolate(ds1, 'vel2', t, k, x3_idx, x2_idx, x2_frac, x3_frac)
            vel3_profile[k] = extract_and_interpolate(ds1, 'vel3', t, k, x3_idx, x2_idx, x2_frac, x3_frac)
        
        # Calculate PBL height
        pbl_height = calculate_pbl_height(theta_profile, x1)
        pbl_height_series[t] = pbl_height
        
        # Find the index closest to PBL height
        pbl_idx = np.argmin(np.abs(x1 - pbl_height))
        
        # PBL winds at top
        vel2_top = vel2_profile[pbl_idx]
        vel3_top = vel3_profile[pbl_idx]
        pbl_wind_top_series[t] = np.sqrt(vel2_top**2 + vel3_top**2)
        
        # PBL winds at middle (half of PBL height)
        middle_idx = np.argmin(np.abs(x1 - pbl_height / 2))
        vel2_mid = vel2_profile[middle_idx]
        vel3_mid = vel3_profile[middle_idx]
        pbl_wind_middle_series[t] = np.sqrt(vel2_mid**2 + vel3_mid**2)
        
        # Surface wind
        vel2_surf = vel2_profile[0]
        vel3_surf = vel3_profile[0]
        surface_wind_series[t] = np.sqrt(vel2_surf**2 + vel3_surf**2)
    
    ds1.close()
    ds2.close()
    
    return {
        'time': time,
        'lcl': lcl_series,
        'pbl_height': pbl_height_series,
        'pbl_wind_top': pbl_wind_top_series,
        'pbl_wind_middle': pbl_wind_middle_series,
        'surface_wind': surface_wind_series
    }


def plot_time_series(data, output_file, lat, lon):
    """
    Create time series plots.
    
    Parameters
    ----------
    data : dict
        Dictionary containing time series data
    output_file : str
        Path to save the output plot (PDF or PNG)
    lat : float
        Latitude of the location
    lon : float
        Longitude of the location
    """
    # Convert time from seconds to hours
    time_hours = data['time'] / 3600.0
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(12, 16))
    
    # 1. LCL vs time
    ax = axes[0]
    ax.plot(time_hours, data['lcl'] / 1000.0, 'b-', linewidth=2)
    ax.set_ylabel('LCL Height (km)', fontsize=12)
    ax.set_title(f'Lifting Condensation Level at ({lat:.3f}°, {lon:.3f}°)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 2. PBL height vs time
    ax = axes[1]
    ax.plot(time_hours, data['pbl_height'] / 1000.0, 'r-', linewidth=2)
    ax.set_ylabel('PBL Height (km)', fontsize=12)
    ax.set_title(f'Planetary Boundary Layer Height at ({lat:.3f}°, {lon:.3f}°)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 3. PBL winds (top) vs time
    ax = axes[2]
    ax.plot(time_hours, data['pbl_wind_top'], 'g-', linewidth=2)
    ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
    ax.set_title(f'PBL Winds (Top) at ({lat:.3f}°, {lon:.3f}°)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 4. PBL winds (middle) vs time
    ax = axes[3]
    ax.plot(time_hours, data['pbl_wind_middle'], 'm-', linewidth=2)
    ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
    ax.set_title(f'PBL Winds (Middle) at ({lat:.3f}°, {lon:.3f}°)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 5. Surface wind vs time
    ax = axes[4]
    ax.plot(time_hours, data['surface_wind'], 'c-', linewidth=2)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
    ax.set_title(f'Surface Wind at ({lat:.3f}°, {lon:.3f}°)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Time series plots saved to: {output_file}")
    plt.close()


def find_netcdf_file_pairs(directory):
    """
    Find all matching out1 and out2 NetCDF file pairs in a directory.
    
    Parameters
    ----------
    directory : str
        Directory to search for NetCDF files
    
    Returns
    -------
    list of tuples
        List of (out1_file, out2_file) pairs
    """
    out1_files = sorted(glob.glob(os.path.join(directory, "*out1*.nc")))
    out2_files = sorted(glob.glob(os.path.join(directory, "*out2*.nc")))
    
    # Match files by replacing out1 with out2
    pairs = []
    for out1_file in out1_files:
        basename = os.path.basename(out1_file)
        out2_pattern = basename.replace("out1", "out2")
        matching_out2 = [f for f in out2_files if os.path.basename(f) == out2_pattern]
        
        if matching_out2:
            pairs.append((out1_file, matching_out2[0]))
        else:
            print(f"Warning: No matching out2 file found for {basename}")
    
    return pairs


def combine_plots_to_pdf(plot_files, output_pdf):
    """
    Combine multiple PNG plots into a single PDF file.
    
    Parameters
    ----------
    plot_files : list
        List of PNG file paths to combine
    output_pdf : str
        Output PDF file path
    """
    print(f"\nCombining {len(plot_files)} plot(s) into PDF: {output_pdf}")
    
    with PdfPages(output_pdf, metadata={'Title': 'Time Series Plots'}) as pdf:
        for plot_file in plot_files:
            if os.path.exists(plot_file):
                try:
                    # Load image
                    img = Image.open(plot_file)
                    
                    # Create figure with appropriate size
                    fig = plt.figure(figsize=(11, 8.5))  # Letter size
                    ax = fig.add_subplot(111)
                    ax.imshow(img)
                    ax.axis('off')
                    
                    # Add title with filename
                    filename = os.path.basename(plot_file)
                    fig.suptitle(filename, fontsize=10, y=0.98)
                    
                    plt.tight_layout()
                    
                    # Save to PDF
                    pdf.savefig(fig, dpi=150, bbox_inches='tight')
                    
                    plt.close(fig)
                    
                    print(f"  Added: {filename}")
                except Exception as e:
                    print(f"  Warning: Could not add {plot_file} to PDF: {e}")
    
    print(f"PDF created: {output_pdf}")


def process_directory(directory, lat, lon, location, locations_csv='../locations.csv',
                     output_dir=None, output_pdf='time_series_all.pdf', cleanup=True):
    """
    Process all NetCDF file pairs in a directory and combine into a PDF.
    
    Parameters
    ----------
    directory : str
        Directory containing NetCDF files
    lat : float
        Latitude of the location
    lon : float
        Longitude of the location
    location : str
        Location name from locations.csv
    locations_csv : str
        Path to locations.csv file
    output_dir : str, optional
        Output directory (default: same as input directory)
    output_pdf : str
        Output PDF filename
    cleanup : bool
        Whether to delete individual PNG files after creating PDF
    """
    directory = os.path.abspath(directory)
    
    if output_dir is None:
        output_dir = directory
    else:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    output_pdf_path = os.path.join(output_dir, output_pdf)
    
    print(f"Processing NetCDF files in: {directory}")
    print(f"Output directory: {output_dir}")
    print(f"Output PDF: {output_pdf_path}")
    print(f"Location: ({lat}, {lon}) - {location}")
    print()
    
    # Find file pairs
    file_pairs = find_netcdf_file_pairs(directory)
    
    if not file_pairs:
        print(f"Error: No matching out1/out2 file pairs found in {directory}")
        return False
    
    print(f"Found {len(file_pairs)} file pair(s)")
    print()
    
    # Process each file pair
    plot_files = []
    for i, (out1_file, out2_file) in enumerate(file_pairs, 1):
        basename = os.path.basename(out1_file)
        print(f"[{i}/{len(file_pairs)}] Processing {basename}...")
        
        # Generate output filename
        base_name = basename.replace(".nc", "").replace("out1", "")
        output_file = os.path.join(output_dir, f"{base_name}_timeseries.png")
        
        try:
            # Extract time series
            data = extract_time_series(out1_file, out2_file, lat, lon, location, locations_csv)
            
            # Create plot
            plot_time_series(data, output_file, lat, lon)
            plot_files.append(output_file)
            
        except Exception as e:
            print(f"  Error processing {basename}: {e}")
            continue
    
    print()
    
    # Combine into PDF
    if plot_files:
        combine_plots_to_pdf(plot_files, output_pdf_path)
        
        # Cleanup individual PNG files if requested
        if cleanup:
            print("\nCleaning up individual PNG files...")
            for plot_file in plot_files:
                try:
                    os.remove(plot_file)
                    print(f"  Removed: {os.path.basename(plot_file)}")
                except Exception as e:
                    print(f"  Warning: Could not remove {plot_file}: {e}")
        
        return True
    else:
        print("Error: No plots were successfully generated")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate time series plots at a specific lat/lon location',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file pair
  %(prog)s out1.nc out2.nc 33.5 -106.5 ws-site1 -o timeseries.pdf
  
  # Process all files in a directory
  %(prog)s /path/to/data/ 33.5 -106.5 ws-site1 --batch -o output_dir/
        """
    )
    parser.add_argument('input_file_out1', 
                       help='Input NetCDF file with pressure and velocities (out1.nc) or directory for batch mode')
    parser.add_argument('input_file_out2', nargs='?',
                       help='Input NetCDF file with temperature (out2.nc), not needed in batch mode')
    parser.add_argument('lat', type=float, help='Latitude of the location')
    parser.add_argument('lon', type=float, help='Longitude of the location')
    parser.add_argument('location', help='Location name (from locations.csv)')
    parser.add_argument('-o', '--output', default='time_series.pdf',
                       help='Output plot file (PDF or PNG for single mode, directory or PDF name for batch mode)')
    parser.add_argument('--locations-csv', default='../locations.csv',
                       help='Path to locations.csv file (default: ../locations.csv)')
    parser.add_argument('--batch', action='store_true',
                       help='Process all NetCDF file pairs in the input directory')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Keep individual PNG files (only for batch mode)')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode: process directory
        directory = args.input_file_out1
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a directory")
            return 1
        
        # Determine output directory and PDF name
        if args.output.endswith('.pdf'):
            output_dir = os.path.dirname(args.output) or '.'
            output_pdf = os.path.basename(args.output)
        elif os.path.isdir(args.output):
            output_dir = args.output
            output_pdf = 'time_series_all.pdf'
        else:
            output_dir = '.'
            output_pdf = args.output if args.output != 'time_series.pdf' else 'time_series_all.pdf'
        
        success = process_directory(
            directory, args.lat, args.lon, args.location,
            locations_csv=args.locations_csv,
            output_dir=output_dir,
            output_pdf=output_pdf,
            cleanup=not args.no_cleanup
        )
        
        if success:
            print("\nDone!")
            return 0
        else:
            return 1
    
    else:
        # Single file mode
        if args.input_file_out2 is None:
            print("Error: input_file_out2 is required in single file mode")
            parser.print_help()
            return 1
        
        # Extract time series
        print("Extracting time series data...")
        data = extract_time_series(args.input_file_out1, args.input_file_out2,
                                   args.lat, args.lon, args.location, args.locations_csv)
        
        # Create plots
        print("Creating plots...")
        plot_time_series(data, args.output, args.lat, args.lon)
        
        print("Done!")
        return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
