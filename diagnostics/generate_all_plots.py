#!/usr/bin/env python3
"""
Master script to generate all diagnostic plots from NetCDF output files.

This script:
1. Finds all out1.nc and out2.nc files in a specified directory
2. Generates all diagnostic plots for each file pair across all time indices
3. Combines all plots into a single PDF file with bookmarks for each hour
"""

import argparse
import glob
import os
import sys
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np
import xarray as xr


def find_netcdf_files(directory):
    """
    Find all out1 and out2 NetCDF files in the directory.
    
    Parameters
    ----------
    directory : str
        Directory to search for NetCDF files
    
    Returns
    -------
    tuple
        (list of out1 files, list of out2 files)
    """
    out1_files = sorted(glob.glob(os.path.join(directory, "*out1*.nc")))
    out2_files = sorted(glob.glob(os.path.join(directory, "*out2*.nc")))
    
    return out1_files, out2_files


def generate_plot(script_name, input_files, output_file, time_index=0, skip=2, topo_dir=None, location=None):
    """
    Generate a plot using one of the diagnostic scripts.
    
    Parameters
    ----------
    script_name : str
        Name of the plotting script
    input_files : list
        List of input NetCDF files
    output_file : str
        Output plot file path
    time_index : int
        Time index to plot
    skip : int
        Skip factor for velocity plots
    topo_dir : str, optional
        Directory containing topography .pt files
    location : str, optional
        Location prefix for topography files
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    script_dir = Path(__file__).parent
    script_path = script_dir / script_name
    
    # Build command
    cmd = ["python3", str(script_path)] + input_files + ["-o", output_file, "-t", str(time_index)]
    
    # Add skip parameter for horizontal velocity plot
    if "horizontal_velocity" in script_name:
        cmd.extend(["-s", str(skip)])
    
    # Add topography parameters if provided
    if topo_dir:
        cmd.extend(["--topo-dir", topo_dir])
    if location:
        cmd.extend(["--location", location])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  Warning: {script_name} failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  Warning: {script_name} timed out")
        return False
    except Exception as e:
        print(f"  Warning: {script_name} failed with exception: {e}")
        return False


def combine_plots_to_pdf(plot_groups, output_pdf):
    """
    Combine multiple PNG plots into a single PDF file with bookmarks.
    
    Parameters
    ----------
    plot_groups : list of tuples
        List of (section_name, list_of_plot_files) tuples
    output_pdf : str
        Output PDF file path
    """
    with PdfPages(output_pdf) as pdf:
        for section_name, plot_files in plot_groups:
            # Add a bookmark/outline for this section
            if plot_files:
                print(f"  Adding section: {section_name}")
                
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
                            
                            # Save to PDF - first plot in section creates bookmark
                            if plot_file == plot_files[0]:
                                pdf.savefig(fig, dpi=150, bbox_inches='tight')
                                # Attach bookmark metadata
                                pdf.attach_note(section_name)
                            else:
                                pdf.savefig(fig, dpi=150, bbox_inches='tight')
                            
                            plt.close(fig)
                            
                            print(f"    Added: {filename}")
                        except Exception as e:
                            print(f"    Warning: Could not add {plot_file} to PDF: {e}")


def get_time_indices(netcdf_file):
    """
    Get all time indices from a NetCDF file.
    
    Parameters
    ----------
    netcdf_file : str
        Path to NetCDF file
    
    Returns
    -------
    list
        List of time indices
    """
    try:
        ds = xr.open_dataset(netcdf_file)
        n_times = len(ds['time'])
        ds.close()
        return list(range(n_times))
    except Exception as e:
        print(f"Warning: Could not read time dimension from {netcdf_file}: {e}")
        return [0]


def process_directory(input_dir, output_dir=None, output_pdf=None, time_index=None, 
                      skip=2, cleanup=True, topo_dir=None, location=None, all_times=False):
    """
    Process all NetCDF files in a directory and generate plots.
    
    Parameters
    ----------
    input_dir : str
        Directory containing NetCDF files
    output_dir : str, optional
        Directory for output plots (default: same as input_dir)
    output_pdf : str, optional
        Output PDF filename (default: diagnostic_plots.pdf)
    time_index : int, optional
        Specific time index to plot (default: 0). Ignored if all_times=True.
    skip : int
        Skip factor for velocity plots
    cleanup : bool
        Whether to clean up individual PNG files after creating PDF
    topo_dir : str, optional
        Directory containing topography .pt files
    location : str, optional
        Location prefix for topography files (e.g., 'ws-site1')
    all_times : bool
        If True, process all time indices and group by hour in PDF
    """
    input_dir = os.path.abspath(input_dir)
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    if output_pdf is None:
        output_pdf = os.path.join(output_dir, "diagnostic_plots.pdf")
    else:
        output_pdf = os.path.abspath(output_pdf)
    
    print(f"Processing NetCDF files in: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output PDF: {output_pdf}")
    if topo_dir:
        print(f"Topography directory: {topo_dir}")
    if location:
        print(f"Location prefix: {location}")
    print()
    
    # Find NetCDF files
    out1_files, out2_files = find_netcdf_files(input_dir)
    
    if not out1_files and not out2_files:
        print(f"Error: No NetCDF files found in {input_dir}")
        return False
    
    print(f"Found {len(out1_files)} out1 file(s) and {len(out2_files)} out2 file(s)")
    
    # Determine time indices to process
    if all_times and out1_files:
        time_indices = get_time_indices(out1_files[0])
        print(f"Processing all {len(time_indices)} time indices")
    else:
        if time_index is None:
            time_index = 0
        time_indices = [time_index]
        print(f"Processing time index: {time_index}")
    print()
    
    # Dictionary to group plots by time (for PDF sections)
    plots_by_time = {}
    
    # Process each time index
    for tidx in time_indices:
        print(f"Processing time index {tidx}...")
        plots_for_time = []
        
        # Process each out1 file
        for i, out1_file in enumerate(out1_files, 1):
            basename = os.path.basename(out1_file)
            
            # Generate base name for output files
            base_name = basename.replace(".nc", "").replace("out1", "")
            
            # 1. Surface pressure
            output_file = os.path.join(output_dir, f"{base_name}_t{tidx:03d}_surface_pressure.png")
            if generate_plot("plot_surface_pressure.py", [out1_file], output_file, tidx, skip, topo_dir, location):
                plots_for_time.append(output_file)
            
            # 2. Vertical velocity
            output_file = os.path.join(output_dir, f"{base_name}_t{tidx:03d}_vertical_velocity.png")
            if generate_plot("plot_vertical_velocity.py", [out1_file], output_file, tidx, skip, topo_dir, location):
                plots_for_time.append(output_file)
            
            # 3. Horizontal velocity
            output_file = os.path.join(output_dir, f"{base_name}_t{tidx:03d}_horizontal_velocity.png")
            if generate_plot("plot_horizontal_velocity.py", [out1_file], output_file, tidx, skip, topo_dir, location):
                plots_for_time.append(output_file)
        
        # Process each out2 file
        for i, out2_file in enumerate(out2_files, 1):
            basename = os.path.basename(out2_file)
            
            # Generate base name for output files
            base_name = basename.replace(".nc", "").replace("out2", "")
            
            # 4. Water paths
            output_file = os.path.join(output_dir, f"{base_name}_t{tidx:03d}_water_paths.png")
            if generate_plot("plot_water_paths.py", [out2_file], output_file, tidx, skip, topo_dir, location):
                plots_for_time.append(output_file)
            
            # 5. Virtual potential temperature
            output_file = os.path.join(output_dir, f"{base_name}_t{tidx:03d}_theta_v.png")
            if generate_plot("plot_theta_v.py", [out2_file], output_file, tidx, skip, topo_dir, location):
                plots_for_time.append(output_file)
        
        # Process LCL (requires both out1 and out2)
        for out1_file in out1_files:
            # Try to find matching out2 file
            basename = os.path.basename(out1_file)
            out2_pattern = basename.replace("out1", "out2")
            matching_out2 = [f for f in out2_files if os.path.basename(f) == out2_pattern]
            
            if matching_out2:
                out2_file = matching_out2[0]
                base_name = basename.replace(".nc", "").replace("out1", "")
                
                output_file = os.path.join(output_dir, f"{base_name}_t{tidx:03d}_lcl.png")
                if generate_plot("plot_lcl.py", [out1_file, out2_file], output_file, tidx, skip, topo_dir, location):
                    plots_for_time.append(output_file)
        
        plots_by_time[tidx] = plots_for_time
    
    print()
    
    # Prepare plot groups for PDF (with sections)
    plot_groups = []
    all_plots = []
    
    # Get time values in hours for section names
    if out1_files:
        try:
            ds = xr.open_dataset(out1_files[0])
            time_values_hours = ds['time'].values / 3600.0  # Convert to hours
            ds.close()
        except:
            time_values_hours = None
    else:
        time_values_hours = None
    
    for tidx in sorted(plots_by_time.keys()):
        plots = plots_by_time[tidx]
        if plots:
            # Create section name
            if time_values_hours is not None and tidx < len(time_values_hours):
                section_name = f"Hour {time_values_hours[tidx]:.2f}"
            else:
                section_name = f"Time Index {tidx}"
            
            plot_groups.append((section_name, plots))
            all_plots.extend(plots)
    
    total_plots = len(all_plots)
    print(f"Generated {total_plots} plot(s) total")
    
    if not all_plots:
        print("Error: No plots were generated successfully")
        return False
    
    # Combine all plots into PDF with sections
    print()
    print("Combining plots into PDF with bookmarks...")
    combine_plots_to_pdf(plot_groups, output_pdf)
    
    print()
    print(f"PDF created: {output_pdf}")
    
    print()
    print(f"PDF created: {output_pdf}")
    
    # Cleanup individual PNG files if requested
    if cleanup:
        print()
        print("Cleaning up individual PNG files...")
        for plot_file in all_plots:
            try:
                os.remove(plot_file)
                print(f"  Removed: {os.path.basename(plot_file)}")
            except Exception as e:
                print(f"  Warning: Could not remove {plot_file}: {e}")
    
    print()
    print("Done!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate all diagnostic plots from NetCDF files and combine into PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single time index (default: 0)
  python generate_all_plots.py ws-site1-2026-02-08/
  
  # Process all time indices with topography overlay
  python generate_all_plots.py ws-site1-2026-02-08/ --all-times --topo-dir Topo/Data/Split/ws-site1/ --location ws-site1
  
  # Specify output directory and PDF name
  python generate_all_plots.py ws-site1-2026-02-08/ -d output/ -p results.pdf
  
  # Process specific time index and keep PNG files
  python generate_all_plots.py ws-site1-2026-02-08/ -t 5 --no-cleanup
        """
    )
    parser.add_argument('input_dir', help='Directory containing NetCDF files')
    parser.add_argument('-d', '--output-dir', help='Output directory for plots (default: same as input)')
    parser.add_argument('-p', '--pdf', help='Output PDF filename (default: diagnostic_plots.pdf)')
    parser.add_argument('-t', '--time', type=int, default=None,
                        help='Time index to plot (default: 0). Ignored if --all-times is set.')
    parser.add_argument('-s', '--skip', type=int, default=2,
                        help='Skip factor for velocity arrows (default: 2)')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Keep individual PNG files after creating PDF')
    parser.add_argument('--topo-dir', help='Directory containing topography .pt files')
    parser.add_argument('--location', help='Location prefix for topography files (e.g., ws-site1)')
    parser.add_argument('--all-times', action='store_true',
                        help='Process all time indices and group by hour in PDF')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory not found: {args.input_dir}")
        return 1
    
    success = process_directory(
        args.input_dir,
        output_dir=args.output_dir,
        output_pdf=args.pdf,
        time_index=args.time,
        skip=args.skip,
        cleanup=not args.no_cleanup,
        topo_dir=args.topo_dir,
        location=args.location,
        all_times=args.all_times
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
