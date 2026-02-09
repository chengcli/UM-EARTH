#!/usr/bin/env python3
"""
Master script to generate all diagnostic plots from NetCDF output files.

This script:
1. Finds all out1.nc and out2.nc files in a specified directory
2. Generates all diagnostic plots for each file pair
3. Combines all plots into a single PDF file
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


def generate_plot(script_name, input_files, output_file, time_index=0, skip=2):
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


def combine_plots_to_pdf(plot_files, output_pdf):
    """
    Combine multiple PNG plots into a single PDF file.
    
    Parameters
    ----------
    plot_files : list
        List of PNG file paths
    output_pdf : str
        Output PDF file path
    """
    with PdfPages(output_pdf) as pdf:
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
                    pdf.savefig(fig, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    print(f"  Added to PDF: {filename}")
                except Exception as e:
                    print(f"  Warning: Could not add {plot_file} to PDF: {e}")


def process_directory(input_dir, output_dir=None, output_pdf=None, time_index=0, skip=2, cleanup=True):
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
    time_index : int
        Time index to plot
    skip : int
        Skip factor for velocity plots
    cleanup : bool
        Whether to clean up individual PNG files after creating PDF
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
    print(f"Time index: {time_index}")
    print()
    
    # Find NetCDF files
    out1_files, out2_files = find_netcdf_files(input_dir)
    
    if not out1_files and not out2_files:
        print(f"Error: No NetCDF files found in {input_dir}")
        return False
    
    print(f"Found {len(out1_files)} out1 file(s) and {len(out2_files)} out2 file(s)")
    print()
    
    # List to store all generated plot files
    all_plots = []
    
    # Process each out1 file
    for i, out1_file in enumerate(out1_files, 1):
        basename = os.path.basename(out1_file)
        print(f"[{i}/{len(out1_files)}] Processing {basename}...")
        
        # Generate base name for output files
        base_name = basename.replace(".nc", "").replace("out1", "")
        
        # 1. Surface pressure
        output_file = os.path.join(output_dir, f"{base_name}_surface_pressure.png")
        if generate_plot("plot_surface_pressure.py", [out1_file], output_file, time_index):
            all_plots.append(output_file)
        
        # 2. Vertical velocity
        output_file = os.path.join(output_dir, f"{base_name}_vertical_velocity.png")
        if generate_plot("plot_vertical_velocity.py", [out1_file], output_file, time_index):
            all_plots.append(output_file)
        
        # 3. Horizontal velocity
        output_file = os.path.join(output_dir, f"{base_name}_horizontal_velocity.png")
        if generate_plot("plot_horizontal_velocity.py", [out1_file], output_file, time_index, skip):
            all_plots.append(output_file)
    
    # Process each out2 file
    for i, out2_file in enumerate(out2_files, 1):
        basename = os.path.basename(out2_file)
        print(f"[{i}/{len(out2_files)}] Processing {basename}...")
        
        # Generate base name for output files
        base_name = basename.replace(".nc", "").replace("out2", "")
        
        # 4. Water paths
        output_file = os.path.join(output_dir, f"{base_name}_water_paths.png")
        if generate_plot("plot_water_paths.py", [out2_file], output_file, time_index):
            all_plots.append(output_file)
        
        # 5. Virtual potential temperature
        output_file = os.path.join(output_dir, f"{base_name}_theta_v.png")
        if generate_plot("plot_theta_v.py", [out2_file], output_file, time_index):
            all_plots.append(output_file)
    
    # Process LCL (requires both out1 and out2)
    for out1_file in out1_files:
        # Try to find matching out2 file
        basename = os.path.basename(out1_file)
        out2_pattern = basename.replace("out1", "out2")
        matching_out2 = [f for f in out2_files if os.path.basename(f) == out2_pattern]
        
        if matching_out2:
            out2_file = matching_out2[0]
            base_name = basename.replace(".nc", "").replace("out1", "")
            
            print(f"Processing LCL for {basename}...")
            output_file = os.path.join(output_dir, f"{base_name}_lcl.png")
            if generate_plot("plot_lcl.py", [out1_file, out2_file], output_file, time_index):
                all_plots.append(output_file)
    
    print()
    print(f"Generated {len(all_plots)} plot(s)")
    
    if not all_plots:
        print("Error: No plots were generated successfully")
        return False
    
    # Combine all plots into PDF
    print()
    print("Combining plots into PDF...")
    combine_plots_to_pdf(all_plots, output_pdf)
    
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
  # Process all NetCDF files in a directory
  python generate_all_plots.py ws-site1-2026-02-08/
  
  # Specify output directory and PDF name
  python generate_all_plots.py ws-site1-2026-02-08/ -d output/ -p results.pdf
  
  # Process specific time index and keep PNG files
  python generate_all_plots.py ws-site1-2026-02-08/ -t 5 --no-cleanup
        """
    )
    parser.add_argument('input_dir', help='Directory containing NetCDF files')
    parser.add_argument('-d', '--output-dir', help='Output directory for plots (default: same as input)')
    parser.add_argument('-p', '--pdf', help='Output PDF filename (default: diagnostic_plots.pdf)')
    parser.add_argument('-t', '--time', type=int, default=0,
                        help='Time index to plot (default: 0)')
    parser.add_argument('-s', '--skip', type=int, default=2,
                        help='Skip factor for velocity arrows (default: 2)')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Keep individual PNG files after creating PDF')
    
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
        cleanup=not args.no_cleanup
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
