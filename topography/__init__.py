"""
Surface Topography Module

This module provides functionality for downloading, caching, and manipulating
topographic elevation data.

Functions:
    get_topography: Download and return topographic height data
    split: Increase resolution by factor of 2 with interpolation
    merge: Decrease resolution by factor of 2 by averaging
"""

from .topography import TopographyData, get_topography, split, merge

__all__ = ['TopographyData', 'get_topography', 'split', 'merge']
