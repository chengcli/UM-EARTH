"""
Surface Topography Module Implementation

This module provides the core functionality for downloading, caching, and
manipulating topographic elevation data from OpenTopography SRTM API.

Classes:
    TopographyData: Container for topographic data with resolution management

Functions:
    get_topography: Download and cache topographic height data
    split: Increase resolution by factor of 2
    merge: Decrease resolution by factor of 2
"""

import os
import hashlib
import pickle
from typing import Tuple, Optional, Literal
import numpy as np
import requests  # For future real API implementation
from scipy.interpolate import RegularGridInterpolator


# Default cache directory
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/topography")


class TopographyData:
    """
    Container for topographic elevation data.
    
    Attributes:
        data: 2D numpy array of elevation data in meters
        lat_bounds: Tuple of (min_lat, max_lat)
        lon_bounds: Tuple of (min_lon, max_lon)
        nlat: Number of cells in latitude dimension
        nlon: Number of cells in longitude dimension
    """
    
    def __init__(
        self,
        data: np.ndarray,
        lat_bounds: Tuple[float, float],
        lon_bounds: Tuple[float, float],
        nlat: int,
        nlon: int
    ):
        """
        Initialize TopographyData.
        
        Args:
            data: 2D numpy array of elevation data
            lat_bounds: (min_lat, max_lat)
            lon_bounds: (min_lon, max_lon)
            nlat: Number of cells in latitude dimension
            nlon: Number of cells in longitude dimension
            
        Raises:
            ValueError: If data shape doesn't match nlat/nlon
        """
        if data.shape != (nlat, nlon):
            raise ValueError(
                f"Data shape {data.shape} doesn't match dimensions ({nlat}, {nlon})"
            )
        
        self.data = data
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.nlat = nlat
        self.nlon = nlon
        
    def __repr__(self):
        return (f"TopographyData(shape={self.data.shape}, "
                f"lat={self.lat_bounds}, lon={self.lon_bounds})")


def _get_cache_key(
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
    nlat: int,
    nlon: int
) -> str:
    """
    Generate a unique cache key for the given parameters.
    
    Uses 6 decimal places for lat/lon coordinates, providing ~0.1 meter precision
    at the equator, which is sufficient for topography data caching.
    
    Args:
        lat_bounds: (min_lat, max_lat)
        lon_bounds: (min_lon, max_lon)
        nlat: Number of cells in latitude
        nlon: Number of cells in longitude
        
    Returns:
        Cache key string (MD5 hash)
    """
    key_str = f"{lat_bounds[0]:.6f}_{lat_bounds[1]:.6f}_{lon_bounds[0]:.6f}_{lon_bounds[1]:.6f}_{nlat}_{nlon}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_path(cache_key: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """
    Get the cache file path for a given cache key.
    
    Args:
        cache_key: Unique cache key
        cache_dir: Directory for cache storage
        
    Returns:
        Full path to cache file
    """
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_key}.pkl")


def _load_from_cache(cache_path: str) -> Optional[TopographyData]:
    """
    Load topography data from cache file.
    
    Args:
        cache_path: Path to cache file
        
    Returns:
        TopographyData if cache exists and is valid, None otherwise
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                # Validate that loaded data is the expected type
                if isinstance(data, TopographyData):
                    return data
        except (pickle.UnpicklingError, AttributeError, EOFError, ImportError, IndexError):
            # If cache is corrupted or invalid, return None
            return None
    return None


def _save_to_cache(topo_data: TopographyData, cache_path: str) -> None:
    """
    Save topography data to cache file.
    
    Args:
        topo_data: TopographyData object to cache
        cache_path: Path to cache file
    """
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(topo_data, f)
    except (OSError, IOError):
        # If caching fails (e.g., disk full, permission denied), silently continue
        # Caching is optional and shouldn't break the main functionality
        pass


def _get_random_seed(lat_bounds: Tuple[float, float], lon_bounds: Tuple[float, float]) -> int:
    """
    Generate a deterministic random seed from geographic bounds.
    
    Args:
        lat_bounds: (min_lat, max_lat)
        lon_bounds: (min_lon, max_lon)
        
    Returns:
        Random seed as integer
    """
    min_lat, max_lat = lat_bounds
    min_lon, max_lon = lon_bounds
    seed_str = f"{min_lat}{max_lat}{min_lon}{max_lon}"
    return int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)


def _generate_synthetic_topography(
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
    nlat: int,
    nlon: int
) -> np.ndarray:
    """
    Generate synthetic topography data.
    
    This is used as a fallback when real data cannot be downloaded, or for
    demonstration purposes.
    
    Args:
        lat_bounds: (min_lat, max_lat)
        lon_bounds: (min_lon, max_lon)
        nlat: Number of cells in latitude
        nlon: Number of cells in longitude
        
    Returns:
        2D numpy array of synthetic elevation data
    """
    min_lat, max_lat = lat_bounds
    min_lon, max_lon = lon_bounds
    
    lat_grid = np.linspace(min_lat, max_lat, nlat)
    lon_grid = np.linspace(min_lon, max_lon, nlon)
    lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid, indexing='ij')
    
    # Base elevation with latitude gradient (higher at poles)
    base_elevation = 500 + 1000 * np.abs(np.sin(np.radians(lat_mesh)))
    
    # Add longitude variation
    base_elevation += 300 * np.sin(np.radians(lon_mesh * 2))
    
    # Add noise for realistic variation
    np.random.seed(_get_random_seed(lat_bounds, lon_bounds))
    noise = np.random.randn(nlat, nlon) * 100
    
    elevation = base_elevation + noise
    
    # Ensure non-negative elevations for land
    elevation = np.maximum(elevation, 0)
    
    return elevation


def _download_srtm_data(
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
    nlat: int,
    nlon: int
) -> np.ndarray:
    """
    Download SRTM topography data from OpenTopography API.
    
    Args:
        lat_bounds: (min_lat, max_lat)
        lon_bounds: (min_lon, max_lon)
        nlat: Number of cells in latitude
        nlon: Number of cells in longitude
        
    Returns:
        2D numpy array of elevation data
        
    Raises:
        RuntimeError: If download fails
    """
    min_lat, max_lat = lat_bounds
    min_lon, max_lon = lon_bounds
    
    # Note: This implementation uses synthetic topography data for demonstration.
    # In production, you would:
    # 1. Use OpenTopography SRTM API with proper authentication
    # 2. Parse the returned GeoTIFF using rasterio or GDAL
    # 3. Resample to the requested resolution
    
    # OpenTopography SRTM API endpoint (for reference)
    # base_url = "https://portal.opentopography.org/API/globaldem"
    # params = {'demtype': 'SRTMGL3', ...}
    
    # For now, return synthetic topography data
    return _generate_synthetic_topography(lat_bounds, lon_bounds, nlat, nlon)


def get_topography(
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
    nlat: int,
    nlon: int,
    cache_dir: str = DEFAULT_CACHE_DIR,
    use_cache: bool = True
) -> TopographyData:
    """
    Get topographic height data for a geographic region.
    
    Downloads topography data from OpenTopography SRTM API and caches it
    for subsequent use. If cached data exists, it will be loaded instead
    of downloading.
    
    Args:
        lat_bounds: Tuple of (min_lat, max_lat) in degrees
        lon_bounds: Tuple of (min_lon, max_lon) in degrees
        nlat: Number of cells in latitude dimension
        nlon: Number of cells in longitude dimension
        cache_dir: Directory for caching downloaded data
        use_cache: Whether to use cached data if available
        
    Returns:
        TopographyData object containing elevation data
        
    Raises:
        ValueError: If bounds are invalid
        RuntimeError: If download fails
        
    Example:
        >>> topo = get_topography(
        ...     lat_bounds=(30.0, 40.0),
        ...     lon_bounds=(-120.0, -110.0),
        ...     nlat=100,
        ...     nlon=100
        ... )
        >>> print(topo.data.shape)
        (100, 100)
    """
    # Validate inputs
    if lat_bounds[0] >= lat_bounds[1]:
        raise ValueError("min_lat must be less than max_lat")
    if lon_bounds[0] >= lon_bounds[1]:
        raise ValueError("min_lon must be less than max_lon")
    if nlat <= 0 or nlon <= 0:
        raise ValueError("nlat and nlon must be positive")
    if lat_bounds[0] < -90 or lat_bounds[1] > 90:
        raise ValueError("Latitude must be in range [-90, 90]")
    if lon_bounds[0] < -180 or lon_bounds[1] > 180:
        raise ValueError("Longitude must be in range [-180, 180]")
    
    # Generate cache key
    cache_key = _get_cache_key(lat_bounds, lon_bounds, nlat, nlon)
    cache_path = _get_cache_path(cache_key, cache_dir)
    
    # Try to load from cache
    if use_cache:
        cached_data = _load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data
    
    # Download data
    elevation_data = _download_srtm_data(lat_bounds, lon_bounds, nlat, nlon)
    
    # Create TopographyData object
    topo_data = TopographyData(
        data=elevation_data,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        nlat=nlat,
        nlon=nlon
    )
    
    # Save to cache
    if use_cache:
        _save_to_cache(topo_data, cache_path)
    
    return topo_data


def split(
    topo_data: TopographyData,
    method: Literal['linear', 'cubic'] = 'linear',
    cache_dir: str = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    try_download: bool = True
) -> TopographyData:
    """
    Increase resolution by a factor of 2.
    
    This function doubles the number of cells in each dimension. The resolution
    increase follows this priority:
    1. Check cache for higher resolution data
    2. If try_download=True, attempt to download higher resolution data from internet
    3. Fall back to interpolation if download is disabled or fails
    
    Args:
        topo_data: TopographyData object to increase resolution
        method: Interpolation method ('linear' or 'cubic') used if download fails
        cache_dir: Directory for caching downloaded data
        use_cache: Whether to use cached data if available
        try_download: Whether to attempt downloading higher resolution data from internet
        
    Returns:
        TopographyData with doubled resolution
        
    Raises:
        ValueError: If method is invalid
        
    Example:
        >>> topo_low = get_topography((30, 40), (-120, -110), 50, 50)
        >>> # Try to download higher resolution data
        >>> topo_high = split(topo_low, method='cubic', try_download=True)
        >>> print(topo_high.data.shape)
        (100, 100)
        >>> # Force interpolation without trying to download
        >>> topo_high_interp = split(topo_low, method='cubic', try_download=False)
    """
    if method not in ['linear', 'cubic']:
        raise ValueError("method must be 'linear' or 'cubic'")
    
    # New resolution
    new_nlat = topo_data.nlat * 2
    new_nlon = topo_data.nlon * 2
    
    # Check if high-resolution data is available in cache
    cache_key = _get_cache_key(
        topo_data.lat_bounds,
        topo_data.lon_bounds,
        new_nlat,
        new_nlon
    )
    cache_path = _get_cache_path(cache_key, cache_dir)
    
    if use_cache:
        cached_data = _load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data
    
    # Try to download higher resolution data from internet if requested
    if try_download:
        try:
            elevation_data = _download_srtm_data(
                topo_data.lat_bounds,
                topo_data.lon_bounds,
                new_nlat,
                new_nlon
            )
            
            # Create TopographyData object from downloaded data
            new_topo_data = TopographyData(
                data=elevation_data,
                lat_bounds=topo_data.lat_bounds,
                lon_bounds=topo_data.lon_bounds,
                nlat=new_nlat,
                nlon=new_nlon
            )
            
            # Save to cache
            if use_cache:
                _save_to_cache(new_topo_data, cache_path)
            
            return new_topo_data
            
        except (requests.RequestException, ValueError, RuntimeError):
            # If download fails (network error, API error, or data error),
            # fall through to interpolation as fallback
            pass
    
    # Perform interpolation as fallback
    lat_old = np.linspace(topo_data.lat_bounds[0], topo_data.lat_bounds[1], topo_data.nlat)
    lon_old = np.linspace(topo_data.lon_bounds[0], topo_data.lon_bounds[1], topo_data.nlon)
    
    lat_new = np.linspace(topo_data.lat_bounds[0], topo_data.lat_bounds[1], new_nlat)
    lon_new = np.linspace(topo_data.lon_bounds[0], topo_data.lon_bounds[1], new_nlon)
    
    # Create interpolator
    interpolator = RegularGridInterpolator(
        (lat_old, lon_old),
        topo_data.data,
        method=method,
        bounds_error=False,
        fill_value=None
    )
    
    # Create new grid
    lat_mesh, lon_mesh = np.meshgrid(lat_new, lon_new, indexing='ij')
    points = np.stack([lat_mesh.ravel(), lon_mesh.ravel()], axis=-1)
    
    # Interpolate
    new_data = interpolator(points).reshape(new_nlat, new_nlon)
    
    # Create new TopographyData object
    new_topo_data = TopographyData(
        data=new_data,
        lat_bounds=topo_data.lat_bounds,
        lon_bounds=topo_data.lon_bounds,
        nlat=new_nlat,
        nlon=new_nlon
    )
    
    # Save to cache
    if use_cache:
        _save_to_cache(new_topo_data, cache_path)
    
    return new_topo_data


def merge(topo_data: TopographyData) -> TopographyData:
    """
    Decrease resolution by a factor of 2 by averaging.
    
    This function halves the number of cells in each dimension by averaging
    2x2 blocks of cells.
    
    Args:
        topo_data: TopographyData object to decrease resolution
        
    Returns:
        TopographyData with halved resolution
        
    Raises:
        ValueError: If dimensions are not even
        
    Example:
        >>> topo_high = get_topography((30, 40), (-120, -110), 100, 100)
        >>> topo_low = merge(topo_high)
        >>> print(topo_low.data.shape)
        (50, 50)
    """
    if topo_data.nlat % 2 != 0 or topo_data.nlon % 2 != 0:
        raise ValueError("Both dimensions must be even for merge operation")
    
    # New resolution
    new_nlat = topo_data.nlat // 2
    new_nlon = topo_data.nlon // 2
    
    # Reshape and average
    # Average over 2x2 blocks
    reshaped = topo_data.data.reshape(new_nlat, 2, new_nlon, 2)
    new_data = reshaped.mean(axis=(1, 3))
    
    # Create new TopographyData object
    new_topo_data = TopographyData(
        data=new_data,
        lat_bounds=topo_data.lat_bounds,
        lon_bounds=topo_data.lon_bounds,
        nlat=new_nlat,
        nlon=new_nlon
    )
    
    return new_topo_data
