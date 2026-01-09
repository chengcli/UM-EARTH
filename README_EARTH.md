# Earth's weather and climate API

This directory contains an API for accessing and analyzing Earth's weather and climate data. The API provides several endpoints for retrieving current weather conditions, historical climate data, and forecasts.

## Installation

For accessing the ECMWF Climate Data Store (CDS), read the instructions at `ecmwf_api/README_ECMWF.md`.

## Regional Weather Prediction Pipelines

### White Sands Missile Range
Complete weather data pipeline for the White Sands test area in New Mexico (October 2025).
- **Location**: `white_sands/`
- **Documentation**: See `white_sands/README.md`
- **Configuration**: `white_sands/white_sands.yaml`
- **Quick start**: `cd white_sands && python download_white_sands_data.py`

### Ann Arbor, Michigan
Complete weather data pipeline for Ann Arbor, Michigan (November 2025).
- **Location**: `ann_arbor/`
- **Documentation**: See `ann_arbor/README.md`
- **Configuration**: `ann_arbor/ann_arbor.yaml`
- **Quick start**: `cd ann_arbor && python download_ann_arbor_data.py`

### ECMWF Data Curation
General-purpose ECMWF ERA5 data access and processing tools.
- **Location**: `ecmwf_api/`
- **Documentation**: See `ecmwf_api/README_ECMWF.md`
- **Complete 4-step pipeline**: Fetch, Calculate Density, Regrid, Compute Pressure

## Configuration files

The following configuration files might be created:
- `$HOME/.cdsapirc`: Configuration file for accessing CDS API. Follow instructions at [https://cds.climate.copernicus.eu/api-how-to](https://cds.climate.copernicus.eu/api-how-to) to set it up.
