ECMWF Preprocessing Pipeline
============================

Purpose
-------

The ECMWF preprocessing stack converts atmospheric source data into a regridded
restart tensor usable by the forecast runtime. The pipeline is orchestrated by
``prepare_initial_condition.py`` and backed by the modules in
``um_earth.ecmwf_api``.

Supported Sources
-----------------

The pipeline supports two source modes:

``era5``
   Fetch ERA5 reanalysis data directly from the Climate Data Store.

``forecast``
   Ingest already-downloaded ECMWF forecast GRIB2 files from a local directory.

Shared Output Contract
----------------------

Both source modes converge to the same downstream artifacts:

* source-specific NetCDF intermediates
* regridded Cartesian NetCDF
* decomposed block NetCDF files
* ``.part`` tensor files containing ``hydro_w``

That shared contract is what keeps ``run_frigate_prediction.py`` independent of
the upstream source mode.

Pipeline Steps
--------------

Step 1
   ERA5 fetch or forecast GRIB2 ingest.

Step 2
   Air-density calculation from temperature, humidity, and cloud content.

Step 3
   Regrid pressure-level lat/lon data to the Cartesian model grid.

Step 4
   Compute a hydrostatically balanced pressure field at cell centers.

Step 5
   Decompose the regridded NetCDF file into per-block files.

Step 6
   Convert decomposed NetCDF blocks into LibTorch-compatible ``.part`` files.

Forecast Ingest Path
--------------------

The forecast-specific ingest stage lives in:

* :mod:`um_earth.ecmwf_api.ingest_forecast_data`

It:

* discovers ``ifs_YYYYMMDD_<cycle>_{pl,sfc}.grib2`` files
* selects one cycle
* filters to the requested lead hours
* decodes with ``cfgrib``/``eccodes``
* normalizes variable and coordinate names
* writes forecast-specific NetCDF intermediates

Naming Conventions
------------------

ERA5 products look like:

.. code-block:: text

   era5_hourly_dynamics_YYYYMMDD.nc
   era5_hourly_densities_YYYYMMDD.nc
   era5_density_YYYYMMDD.nc

Forecast products look like:

.. code-block:: text

   forecast_hourly_dynamics_YYYYMMDD_CC.nc
   forecast_hourly_densities_YYYYMMDD_CC.nc
   forecast_density_YYYYMMDD_CC.nc

where ``CC`` is the selected forecast cycle, for example ``00``.

Regridded and tensor outputs follow the common pattern:

.. code-block:: text

   regridded_<region>_<stem>.nc
   regridded_<region>_<stem>_blocks/
   regridded_<region>_<stem>_tensors/

Typical Forecast Preparation Command
------------------------------------

.. code-block:: bash

   python prepare_initial_condition.py \
     pte1b \
     --region-kml /path/to/pte1b.kml \
     --config /path/to/pte1b.yaml \
     --output-base /path/to/forecast_input \
     --data-source forecast \
     --forecast-input-dir /path/to/ECMWF_prediction_data/20260412 \
     --forecast-cycle 00 \
     --forecast-leads 0 6 12 18

Operational Caveat
------------------

The current forecast ingest path does not yet emit a dedicated NetCDF topography
field for the atmospheric regridding stage. In that case the regridding code
falls back to a flat surface while the actual forecast runtime still uses the
prepared terrain products from ``topography/products/``.
