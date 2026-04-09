"""
ECMWF Weather Data API

This package provides tools for fetching and processing ECMWF ERA5 weather data
from the Climate Data Store (CDS).
"""

from .ecmwf_weather_api import ECMWFWeatherAPI, create_api

__all__ = ['ECMWFWeatherAPI', 'create_api']
__version__ = '1.0.0'
