# Surface Topography Module

A Python module for downloading, caching, and manipulating topographic elevation data.

## Features

- **Download topographic data**: Fetches elevation data for any geographic region
- **Automatic caching**: Downloaded data is cached locally for fast subsequent access
- **Resolution management**: 
  - `split()`: Increase resolution by 2x using interpolation
  - `merge()`: Decrease resolution by 2x using averaging
- **Flexible interpolation**: Supports linear and cubic interpolation methods

## Installation

The module requires the following dependencies:

```bash
pip install numpy scipy requests
```

## Usage

### Basic Usage

```python
from topography import get_topography, split, merge

# Download topography data for a region
topo = get_topography(
    lat_bounds=(30.0, 40.0),      # Latitude range (degrees)
    lon_bounds=(-120.0, -110.0),  # Longitude range (degrees)
    nlat=100,                      # Number of cells in latitude
    nlon=100                       # Number of cells in longitude
)

# Access the elevation data
print(topo.data.shape)  # (100, 100)
print(f"Elevation range: {topo.data.min():.1f} - {topo.data.max():.1f} meters")
```

### Increasing Resolution (Split)

```python
# Double the resolution using linear interpolation
topo_high_res = split(topo, method='linear')
print(topo_high_res.data.shape)  # (200, 200)

# Or use cubic interpolation for smoother results
topo_cubic = split(topo, method='cubic')
```

### Decreasing Resolution (Merge)

```python
# Halve the resolution by averaging 2x2 blocks
topo_low_res = merge(topo_high_res)
print(topo_low_res.data.shape)  # (100, 100)
```

### Working with Cache

```python
import os

# Specify a custom cache directory
cache_dir = os.path.expanduser("~/my_topo_cache")

# First call downloads and caches data
topo1 = get_topography(
    lat_bounds=(30.0, 40.0),
    lon_bounds=(-120.0, -110.0),
    nlat=100,
    nlon=100,
    cache_dir=cache_dir
)

# Second call loads from cache (much faster!)
topo2 = get_topography(
    lat_bounds=(30.0, 40.0),
    lon_bounds=(-120.0, -110.0),
    nlat=100,
    nlon=100,
    cache_dir=cache_dir
)

# Disable caching if needed
topo_no_cache = get_topography(
    lat_bounds=(30.0, 40.0),
    lon_bounds=(-120.0, -110.0),
    nlat=100,
    nlon=100,
    use_cache=False
)
```

## API Reference

### TopographyData

Container class for topographic elevation data.

**Attributes:**
- `data` (np.ndarray): 2D array of elevation values in meters
- `lat_bounds` (tuple): (min_lat, max_lat) in degrees
- `lon_bounds` (tuple): (min_lon, max_lon) in degrees
- `nlat` (int): Number of cells in latitude dimension
- `nlon` (int): Number of cells in longitude dimension

### get_topography()

```python
get_topography(
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
    nlat: int,
    nlon: int,
    cache_dir: str = "~/.cache/topography",
    use_cache: bool = True
) -> TopographyData
```

Download and return topographic elevation data for a geographic region.

**Parameters:**
- `lat_bounds`: Latitude range (min_lat, max_lat) in degrees [-90, 90]
- `lon_bounds`: Longitude range (min_lon, max_lon) in degrees [-180, 180]
- `nlat`: Number of cells in latitude dimension (must be positive)
- `nlon`: Number of cells in longitude dimension (must be positive)
- `cache_dir`: Directory for caching downloaded data
- `use_cache`: Whether to use cached data if available

**Returns:**
- `TopographyData` object containing elevation data

**Raises:**
- `ValueError`: If bounds are invalid or out of range
- `RuntimeError`: If download fails

### split()

```python
split(
    topo_data: TopographyData,
    method: Literal['linear', 'cubic'] = 'linear',
    cache_dir: str = "~/.cache/topography",
    use_cache: bool = True
) -> TopographyData
```

Increase resolution by a factor of 2 using interpolation.

**Parameters:**
- `topo_data`: TopographyData object to increase resolution
- `method`: Interpolation method ('linear' or 'cubic')
- `cache_dir`: Directory for caching
- `use_cache`: Whether to use cached data if available

**Returns:**
- `TopographyData` with doubled resolution

**Raises:**
- `ValueError`: If method is invalid

### merge()

```python
merge(topo_data: TopographyData) -> TopographyData
```

Decrease resolution by a factor of 2 by averaging 2x2 blocks.

**Parameters:**
- `topo_data`: TopographyData object to decrease resolution

**Returns:**
- `TopographyData` with halved resolution

**Raises:**
- `ValueError`: If dimensions are not even

## Examples

### Example 1: White Sands, New Mexico

```python
from topography import get_topography, split

# Download topography for White Sands
white_sands = get_topography(
    lat_bounds=(32.5, 33.5),
    lon_bounds=(-106.5, -106.0),
    nlat=50,
    nlon=50
)

# Increase resolution
white_sands_hires = split(white_sands, method='cubic')

print(f"Low res shape: {white_sands.data.shape}")
print(f"High res shape: {white_sands_hires.data.shape}")
```

### Example 2: Ann Arbor, Michigan

```python
from topography import get_topography, merge

# Download high-resolution topography
ann_arbor = get_topography(
    lat_bounds=(42.0, 42.5),
    lon_bounds=(-83.8, -83.3),
    nlat=200,
    nlon=200
)

# Reduce resolution for coarse simulation
ann_arbor_coarse = merge(ann_arbor)
ann_arbor_very_coarse = merge(ann_arbor_coarse)

print(f"Original: {ann_arbor.data.shape}")
print(f"Coarse: {ann_arbor_coarse.data.shape}")
print(f"Very coarse: {ann_arbor_very_coarse.data.shape}")
```

### Example 3: Progressive Refinement

```python
from topography import get_topography, split

# Start with coarse resolution
region = get_topography(
    lat_bounds=(35.0, 45.0),
    lon_bounds=(-100.0, -90.0),
    nlat=10,
    nlon=10
)

# Progressively refine
level1 = split(region, method='linear')   # 20x20
level2 = split(level1, method='linear')   # 40x40
level3 = split(level2, method='cubic')    # 80x80

print("Progressive refinement:")
for i, topo in enumerate([region, level1, level2, level3]):
    print(f"  Level {i}: {topo.data.shape}")
```

## Cache Management

By default, the module caches downloaded data in `~/.cache/topography/`. Each unique combination of bounds and resolution is cached separately.

Cache files are stored as pickled Python objects with MD5-hashed filenames based on the parameters.

To clear the cache:

```bash
rm -rf ~/.cache/topography/
```

## Testing

Run the test suite:

```bash
cd topography
python -m unittest test_topography.py
```

Or run with verbose output:

```bash
python -m unittest test_topography.py -v
```

## Notes

- The module currently uses synthetic topography data for demonstration purposes. In a production environment, you would integrate with a real topography API like OpenTopography SRTM.
- Elevation values are in meters above sea level.
- The split operation uses scipy's RegularGridInterpolator for high-quality interpolation.
- The merge operation uses simple averaging of 2x2 blocks for downsampling.

## License

This module is part of the UM-EARTH project.
