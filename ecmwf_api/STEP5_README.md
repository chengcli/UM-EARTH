# ECMWF Data Pipeline - Step 5: Domain Decomposition

This document describes Step 5 of the ECMWF data fetching and curation pipeline, which decomposes the regridded NetCDF file from Step 3 (or Step 4 if hydrostatic pressure was computed) into multiple blocks for parallel computation.

## Overview

Step 5 takes the regridded ERA5 data and performs domain decomposition in the horizontal directions (x2, x3). Each resulting block contains a portion of the domain with ghost zones overlapping with neighboring blocks, enabling parallel computation with proper boundary conditions.

The vertical direction (x1) is not decomposed, so each block contains the full vertical column.

## Prerequisites

Before running Step 5, you must have:

1. **Step 3 or 4 completed**: Regridded NetCDF file with:
   - Dimensions: `time`, `x1`, `x2`, `x3` (cell centers)
   - Interface dimensions: `x1f`, `x2f`, `x3f` (cell interfaces)
   - Global attribute: `nghost` (number of ghost cells on each side)
   - Variables regridded to Cartesian grid

2. **Python dependencies**: netCDF4, numpy

## Algorithm

The script performs the following steps:

### 1. Read Metadata

The script reads the input NetCDF file to extract:
- Grid dimensions (time, x1, x2, x3)
- Ghost zone size (`nghost` global attribute)
- Interior dimensions (`nx2_interior`, `nx3_interior` attributes)

### 2. Calculate Block Boundaries

For each horizontal direction (x2, x3):

1. Divide the interior domain into `n_blocks` equal (or nearly equal) parts
2. For each block, add ghost zones on both sides:
   - First block: includes original left ghost zone + interior + overlap ghost
   - Middle blocks: overlap ghost + interior + overlap ghost
   - Last block: overlap ghost + interior + original right ghost zone

### 3. Extract and Save Blocks

For each block:

1. Create new NetCDF file with dimensions matching the block size
2. Copy all coordinate variables, slicing horizontal coordinates
3. Copy all data variables, extracting the appropriate horizontal slice
4. Copy global attributes and add block-specific metadata
5. Add processing history

## Usage

### Basic Usage

```bash
python decompose_domain.py <input_file> <n_blocks_x2> <n_blocks_x3>
```

### Arguments

- `input_file`: Path to regridded NetCDF file from Step 3 or 4
- `n_blocks_x2`: Number of blocks in x2 (Y, North-South) direction
- `n_blocks_x3`: Number of blocks in x3 (X, East-West) direction
- `--output-dir` or `-o`: Optional output directory (default: same as input file)

### Examples

```bash
# Decompose into 4x4 blocks (16 blocks total)
python decompose_domain.py regridded_ann-arbor_20251102.nc 4 4

# Decompose into 2x3 blocks with custom output directory
python decompose_domain.py regridded_data.nc 2 3 --output-dir ./blocks/

# Single direction decomposition (4 blocks in Y, 1 in X)
python decompose_domain.py regridded_data.nc 4 1
```

## Output

### File Naming Convention

Output files are named: `{input_base}_block_{i2}_{j3}.nc`

Where:
- `{input_base}` is the input filename without `.nc` extension
- `{i2}` is the block index in x2 direction (0-based)
- `{j3}` is the block index in x3 direction (0-based)

Example:
- Input: `regridded_ann-arbor_20251102.nc`
- Output (4x4): 
  - `regridded_ann-arbor_20251102_block_0_0.nc`
  - `regridded_ann-arbor_20251102_block_0_1.nc`
  - ...
  - `regridded_ann-arbor_20251102_block_3_3.nc`

### Block Structure

Each block file contains:

#### Dimensions
- `time`: Same as input (no decomposition in time)
- `x1`: Same as input (no vertical decomposition)
- `x2`: Block size in Y direction (with ghost zones)
- `x3`: Block size in X direction (with ghost zones)
- `x1f`, `x2f`, `x3f`: Interface dimensions

#### Variables
All variables from input file, sliced in horizontal directions:
- Variables on cell centers: `(time, x1, x2, x3)`
- Variables on vertical interfaces: `(time, x1f, x2, x3)`

#### Coordinates
- `time`: Full time coordinate (copied from input)
- `x1`, `x1f`: Full vertical coordinates (copied from input)
- `x2`, `x2f`: Subset of Y coordinates for this block
- `x3`, `x3f`: Subset of X coordinates for this block

#### Block Metadata
Additional global attributes:
- `block_index_x2`: Block index in x2 direction
- `block_index_x3`: Block index in x3 direction
- `block_x2_start`: Start index in x2 (in input file coordinates)
- `block_x2_end`: End index in x2 (in input file coordinates)
- `block_x3_start`: Start index in x3 (in input file coordinates)
- `block_x3_end`: End index in x3 (in input file coordinates)

## Example Scenario

### Input File
- Filename: `regridded_ann-arbor_20251102.nc`
- Dimensions: `time=4, x1=156, x2=206, x3=206`
- Ghost zones: `nghost=3` (on each side)
- Interior dimensions: `nx2_interior=200, nx3_interior=200`

### Decomposition: 4x4 blocks

```bash
python decompose_domain.py regridded_ann-arbor_20251102.nc 4 4
```

### Result
- 16 block files created
- Each block has approximately 50x50 interior cells
- Each block includes 3 ghost cells on each side
- Total block dimensions: approximately 56x56 horizontal cells
  - Note: First and last blocks in each direction may be slightly different
  - First blocks: include original ghost zones
  - Last blocks: include original ghost zones
  - Interior blocks: overlap with neighbors

### Block Dimensions

For 200 interior cells divided into 4 blocks with 3 ghost zones:

| Block | x2 Range | x2 Size | x3 Range | x3 Size |
|-------|----------|---------|----------|---------|
| (0,0) | [0:56]   | 56      | [0:56]   | 56      |
| (0,1) | [0:56]   | 56      | [50:106] | 56      |
| (0,2) | [0:56]   | 56      | [100:156]| 56      |
| (0,3) | [0:56]   | 56      | [150:206]| 56      |
| (1,0) | [50:106] | 56      | [0:56]   | 56      |
| ...   | ...      | ...     | ...      | ...     |
| (3,3) | [150:206]| 56      | [150:206]| 56      |

Note: Each interior block has 50 interior cells + 6 ghost cells (3 on each side) = 56 total cells

## Ghost Zone Overlap

Ghost zones enable parallel computation with proper boundary conditions:

### First Block in a Direction
- Includes original left ghost zone (from full domain)
- Interior cells
- Right overlap ghost zone (overlaps with next block)

### Middle Blocks
- Left overlap ghost zone (overlaps with previous block)
- Interior cells
- Right overlap ghost zone (overlaps with next block)

### Last Block in a Direction
- Left overlap ghost zone (overlaps with previous block)
- Interior cells
- Includes original right ghost zone (from full domain)

### Overlap Size
Ghost zones overlap by `nghost` cells on each side, ensuring that:
- Each block can compute fluxes at interfaces
- Boundary conditions can be properly communicated between blocks
- Parallel computation maintains consistency

## Use Cases

### Parallel Computation
Domain decomposition enables:
- Running atmospheric model simulations in parallel
- Distributing blocks across multiple processors/nodes
- Reducing memory requirements per process
- Scaling to larger domains

### Analysis and Visualization
Decomposed blocks can be:
- Processed independently for analysis
- Visualized separately or recombined
- Used for targeted regional studies

## Error Handling

The script will raise errors if:

1. **File errors:**
   - Input file not found
   - Input file missing required dimensions
   - Input file missing `nghost` attribute

2. **Argument errors:**
   - Number of blocks less than 1
   - Invalid number of blocks (e.g., more blocks than interior cells)

3. **Data errors:**
   - Incompatible array shapes
   - Missing required variables or coordinates

## Testing

Run unit tests to verify functionality:

```bash
python -m unittest test_decompose_domain
```

The test suite includes:
- Block boundary calculations
- Ghost zone overlap verification
- NetCDF file reading and writing
- Data preservation across decomposition
- Various decomposition configurations (1x1, 2x2, 4x4, etc.)

## Complete Workflow Example

From raw data to decomposed blocks:

```bash
# Step 1: Fetch ERA5 data
python fetch_era5_pipeline.py example_ann_arbor.yaml --output-base ./data/

# Step 2: Calculate density
python calculate_density.py --input-dir ./data/41.75N_42.85N_84.25W_83.15W/ \
                           --output-dir ./data/41.75N_42.85N_84.25W_83.15W/

# Step 3: Regrid to Cartesian
python regrid_era5_to_cartesian.py example_ann_arbor.yaml \
                                   ./data/41.75N_42.85N_84.25W_83.15W/ \
                                   --output regridded_ann-arbor_20251102.nc

# Step 4 (optional): Compute hydrostatic pressure
python compute_hydrostatic_pressure.py example_ann_arbor.yaml \
                                      regridded_ann-arbor_20251102.nc

# Step 5: Domain decomposition
python decompose_domain.py regridded_ann-arbor_20251102.nc 4 4 \
                          --output-dir ./blocks/

# Result: 16 block files in ./blocks/ directory
```

## Performance

Step 5 is I/O intensive:
- Reads entire input file
- Writes multiple output files
- Uses NetCDF compression (complevel=4) for output

Typical performance:
- For a 4x4 decomposition of 200MB file: ~30-60 seconds
- Memory usage: proportional to largest data variable
- Scales linearly with number of blocks

For very large domains or many blocks, consider:
- Using SSD storage for faster I/O
- Running on systems with sufficient memory
- Enabling parallel I/O if supported by file system

## Integration with Atmospheric Models

The decomposed blocks are ready for use with atmospheric models that support:
- Finite volume methods
- Ghost zone communication
- Cartesian coordinate systems
- NetCDF input

Each block can be assigned to a separate processor/node for parallel simulation.

## References

- `decompose_domain.py`: Step 5 implementation
- `test_decompose_domain.py`: Unit tests
- `regrid_era5_to_cartesian.py`: Step 3 implementation
- `compute_hydrostatic_pressure.py`: Step 4 implementation
- `example_ann_arbor.yaml`: Example configuration file

## Notes

- Domain decomposition is only in horizontal directions (x2, x3)
- Vertical direction (x1) is not decomposed
- Time dimension is not decomposed
- Ghost zones ensure proper parallel computation
- Block indices are 0-based
- Original ghost zones are preserved in first and last blocks
