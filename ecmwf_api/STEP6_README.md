# ECMWF Data Pipeline - Step 6: NetCDF to PyTorch Tensor Conversion

This document describes Step 6 of the ECMWF data fetching and curation pipeline, which converts decomposed NetCDF block files from Step 5 into PyTorch tensor files for use in dynamic atmospheric simulations.

## Overview

Step 6 takes the decomposed block files from Step 5 and converts them into PyTorch tensors suitable for use in atmospheric simulation models. The conversion includes:

1. Reading atmospheric variables from NetCDF files
2. Combining cloud water variables into aggregated quantities
3. Reordering axes to match simulation requirements
4. Saving tensors in LibTorch-compatible format

## Prerequisites

Before running Step 6, you must have:

1. **Step 5 completed**: Decomposed NetCDF block files with:
   - Dimensions: `time`, `x1`, `x2`, `x3` (cell centers)
   - Required variables: `rho`, `w`, `v`, `u`, `p` (or `pressure`), `q`
   - Cloud variables: `ciwc`, `clwc`, `cswc`, `crwc` (optional)

2. **Python dependencies**: PyTorch, netCDF4, numpy

## Algorithm

The script performs the following steps:

### 1. Load Variables from NetCDF

Reads the following variables from each block file:
- `rho`: Air density [kg/m³]
- `w`: Vertical velocity [Pa/s]
- `v`: Meridional wind component [m/s]
- `u`: Zonal wind component [m/s]
- `p` or `pressure` or `pressure_level`: Pressure [Pa]
- `q`: Specific humidity [kg/kg]
- `ciwc`: Cloud ice water content [kg/kg] (optional)
- `clwc`: Cloud liquid water content [kg/kg] (optional)
- `cswc`: Cloud snow water content [kg/kg] (optional)
- `crwc`: Cloud rain water content [kg/kg] (optional)

All variables are expected to have dimensions `(time, x1, x2, x3)`.

### 2. Combine Cloud Variables

Creates aggregated cloud water quantities:
- `q2 = ciwc + clwc` (total cloud ice + liquid water)
- `q3 = cswc + crwc` (total cloud snow + rain water)

If cloud variables are missing, they are set to zero.

### 3. Handle Pressure on Interfaces

If pressure is stored on vertical interfaces (`x1f`), the script automatically averages it to cell centers:
```python
p_centers = 0.5 * (p_interfaces[:-1] + p_interfaces[1:])
```

### 4. Aggregate into hydro_w Tensor

Creates a single tensor `hydro_w` with all variables in a specific order:
- Order: `(rho, w, v, u, p, q, q2, q3)`
- Dimension: `nvar = 8`

### 5. Reorder Axes

Transposes the tensor from NetCDF ordering to simulation ordering:
- Input: `(time, x1, x2, x3)` for each variable
- Stack: `(time, nvar, x1, x2, x3)`
- Output: `(time, nvar, x3, x2, x1)`

This reordering is necessary because atmospheric simulation models often use a different axis convention.

### 6. Save as PyTorch Tensor

Saves the tensor using `torch.jit.script` for LibTorch compatibility:
```python
class TensorModule(torch.nn.Module):
    def __init__(self, tensors):
        super().__init__()
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor)

module = TensorModule({'hydro_w': hydro_w_tensor})
scripted = torch.jit.script(module)
scripted.save(output_file)
```

## Usage

### Basic Usage

```bash
python convert_netcdf_to_tensor.py <input_file.nc> [--output OUTPUT.restart]
```

### Arguments

- `input`: Input NetCDF file or directory containing .nc files
- `--output` or `-o`: Output .restart file (for single file) or directory (for batch mode)
- `--output-dir`: Alternative to --output for directory mode
- `--pattern`: Glob pattern for finding NetCDF files (default: `*.nc`)

### Examples

#### Convert a Single File

```bash
# Convert with automatic output naming
python convert_netcdf_to_tensor.py regridded_block_0_0.nc

# Convert with custom output name
python convert_netcdf_to_tensor.py regridded_block_0_0.nc --output block_0_0.restart
```

#### Convert All Files in a Directory

```bash
# Convert all .nc files in directory
python convert_netcdf_to_tensor.py ./blocks/

# Convert with custom output directory
python convert_netcdf_to_tensor.py ./blocks/ --output-dir ./tensors/

# Convert only specific files
python convert_netcdf_to_tensor.py ./blocks/ --pattern "*block_*.nc"
```

## Output

### File Naming Convention

Output files have the same basename as input files with `.restart` extension:
- Input: `regridded_block_0_0.nc`
- Output: `regridded_block_0_0.restart`

### Tensor Structure

Each `.restart` file contains a `TensorModule` with a single tensor:

#### hydro_w Tensor

- **Name**: `hydro_w`
- **Shape**: `(time, nvar, x3, x2, x1)`
- **dtype**: `torch.float32` or `torch.float64` (depends on input)
- **Variable order in nvar dimension**:
  - Index 0: `rho` (density)
  - Index 1: `w` (vertical velocity)
  - Index 2: `v` (meridional wind)
  - Index 3: `u` (zonal wind)
  - Index 4: `p` (pressure)
  - Index 5: `q` (specific humidity)
  - Index 6: `q2` (cloud ice + liquid water)
  - Index 7: `q3` (cloud snow + rain water)

### Loading Tensors

To load and use the tensors in PyTorch:

```python
import torch

# Load the module
module = torch.jit.load('regridded_block_0_0.restart')

# Access the tensor
hydro_w = module.hydro_w

# Get dimensions
n_time, n_var, n_x3, n_x2, n_x1 = hydro_w.shape

# Access specific variables
rho = hydro_w[:, 0, :, :, :]  # density
w = hydro_w[:, 1, :, :, :]    # vertical velocity
v = hydro_w[:, 2, :, :, :]    # meridional wind
u = hydro_w[:, 3, :, :, :]    # zonal wind
p = hydro_w[:, 4, :, :, :]    # pressure
q = hydro_w[:, 5, :, :, :]    # specific humidity
q2 = hydro_w[:, 6, :, :, :]   # cloud ice + liquid
q3 = hydro_w[:, 7, :, :, :]   # cloud snow + rain
```

To load in LibTorch (C++):

```cpp
#include <torch/script.h>

// Load the module
torch::jit::script::Module module = torch::jit::load("regridded_block_0_0.restart");

// Access the tensor
torch::Tensor hydro_w = module.attr("hydro_w").toTensor();

// Access specific variables (example: density)
auto rho = hydro_w.index({torch::indexing::Slice(), 0, 
                          torch::indexing::Slice(), 
                          torch::indexing::Slice(), 
                          torch::indexing::Slice()});
```

## Example Scenario

### Input File

- Filename: `regridded_ann-arbor_20251102_block_0_0.nc`
- Dimensions: `time=4, x1=56, x2=56, x3=56`
- Variables: `rho`, `u`, `v`, `w`, `p`, `q`, `ciwc`, `clwc`, `cswc`, `crwc`

### Conversion

```bash
python convert_netcdf_to_tensor.py regridded_ann-arbor_20251102_block_0_0.nc
```

### Output

- Filename: `regridded_ann-arbor_20251102_block_0_0.restart`
- Tensor: `hydro_w` with shape `(4, 8, 56, 56, 56)`
- Memory: Approximately 450 MB (4 × 8 × 56³ × 4 bytes)

### Processing Multiple Blocks

```bash
# Convert all 16 blocks from 4x4 decomposition
python convert_netcdf_to_tensor.py ./blocks/ --output-dir ./tensors/

# Result: 16 .restart files in ./tensors/ directory
```

## Error Handling

The script provides informative error messages for common issues:

### Missing Required Variables

```
Error: Missing required variables: ['u', 'v']
Available variables: ['rho', 'w', 'p', 'q', ...]
```

### File Not Found

```
Error: Input file not found: /path/to/file.nc
```

### Dimension Mismatch

```
Error: Variable 'rho' has unexpected shape (4, 56, 56, 60), 
       expected (4, 56, 56, 56)
```

### Missing Cloud Variables

```
Warning: Missing cloud variables: ['ciwc', 'clwc', 'cswc', 'crwc']
Will use zeros for missing variables
```

The script continues execution with zeros for missing cloud variables, since they may not always be available.

## Testing

Run unit tests to verify functionality:

```bash
python -m unittest test_convert_netcdf_to_tensor -v
```

The test suite includes:
- Basic conversion functionality
- Tensor shape verification
- Variable ordering validation
- Cloud variable combination
- Axis reordering verification
- Error handling
- Directory batch processing
- Pressure on interfaces handling

## Performance

Step 6 is memory and I/O intensive:

### Memory Usage

- Loads entire NetCDF file into memory
- Creates numpy arrays and PyTorch tensors
- Peak memory: ~2-3× the size of the tensor

### Typical Performance

- Single file (56×56×56, 4 time steps): ~1-2 seconds
- Batch processing (16 blocks): ~20-30 seconds
- Scales linearly with number of files and block size

### Optimization Tips

1. **Process in batches**: Convert multiple files in one command
2. **Use SSD storage**: Faster I/O for reading NetCDF and writing tensors
3. **Sufficient memory**: Ensure at least 2× block size in RAM
4. **Parallel processing**: For many blocks, use multiple processes:
   ```bash
   # Process blocks 0-7
   python convert_netcdf_to_tensor.py ./blocks/ --pattern "*block_[0-7]_*.nc" &
   
   # Process blocks 8-15
   python convert_netcdf_to_tensor.py ./blocks/ --pattern "*block_[8-9]_*.nc" \
                                     --pattern "*block_1[0-5]_*.nc" &
   ```

## Integration with Simulation Models

The tensor files are ready for use with atmospheric models that:
- Accept PyTorch/LibTorch tensors as input
- Use finite volume methods
- Support domain decomposition
- Expect variables in the order: (rho, w, v, u, p, q, q2, q3)
- Use axis ordering: (time, variable, x3, x2, x1)

Each tensor file corresponds to one block from the domain decomposition and can be:
- Loaded independently for parallel simulation
- Combined with other blocks for full domain simulation
- Used as initial conditions for model runs

## Complete Workflow Example

From raw data to simulation-ready tensors:

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

# Step 6: Convert to tensors
python convert_netcdf_to_tensor.py ./blocks/ --output-dir ./tensors/

# Result: 16 .restart files ready for simulation in ./tensors/ directory
```

## Coordinate System Notes

### Axis Conventions

The tensor uses a specific axis ordering that differs from NetCDF:

| Format   | Dimensions                     | Notes                           |
|----------|--------------------------------|---------------------------------|
| NetCDF   | (time, x1, x2, x3)            | x1=height, x2=Y, x3=X           |
| Tensor   | (time, nvar, x3, x2, x1)      | Reordered for simulation        |

### Coordinate Mapping

- **x1** (NetCDF) → **x1** (Tensor, last dimension): Vertical/height
- **x2** (NetCDF) → **x2** (Tensor, 3rd dimension): Y/North-South
- **x3** (NetCDF) → **x3** (Tensor, 2nd dimension): X/East-West

This reordering ensures compatibility with simulation models that expect specific memory layouts for efficient computation.

## References

- `convert_netcdf_to_tensor.py`: Step 6 implementation
- `test_convert_netcdf_to_tensor.py`: Unit tests
- `decompose_domain.py`: Step 5 implementation
- `STEP5_README.md`: Step 5 documentation

## Notes

- Variable order in `hydro_w` is fixed: (rho, w, v, u, p, q, q2, q3)
- Axis order is reordered from (time, x1, x2, x3) to (time, nvar, x3, x2, x1)
- Cloud variables are optional; zeros are used if missing
- Pressure can be on cell centers or interfaces
- Tensors are saved using torch.jit.script for LibTorch compatibility
- Output files use the same basename as input files with .restart extension
