Command-Line Interface
======================

Entry Point
-----------

The package exposes one main CLI:

.. code-block:: bash

   um-earth

It is implemented in :mod:`um_earth.cli` and organized into top-level command
groups.

Region Commands
---------------

Inspect region definitions from either a KML file or the CSV-backed location
table:

.. code-block:: bash

   um-earth region inspect --region-kml /path/to/region.kml
   um-earth region inspect --location-id pte1b

Configuration Commands
----------------------

Generate a simulation YAML from a region and target grid:

.. code-block:: bash

   um-earth config generate \
     --region-kml /path/to/region.kml \
     --start-date 2026-04-12 \
     --end-date 2026-04-12 \
     --nx1 50 --nx2 56 --nx3 46 \
     --output pte1b.yaml

Topography Commands
-------------------

Prepare terrain inputs:

.. code-block:: bash

   um-earth topo build \
     --region-kml /path/to/region.kml \
     --out ./topography/raw

Initial-Condition Commands
--------------------------

The main preprocessing command is:

.. code-block:: bash

   um-earth init prepare ...

ERA5 mode:

.. code-block:: bash

   um-earth init prepare \
     --region-kml /path/to/region.kml \
     --config ./pte1b.yaml \
     --output-base ./era5 \
     --data-source era5 \
     --times 00:00 06:00 12:00 18:00

Forecast mode:

.. code-block:: bash

   um-earth init prepare \
     --region-kml /path/to/region.kml \
     --config ./pte1b.yaml \
     --output-base ./forecast_input \
     --data-source forecast \
     --forecast-input-dir /path/to/ECMWF_prediction_data/20260412 \
     --forecast-cycle 00 \
     --forecast-leads 0 6 12 18

Key options:

``--start-from`` / ``--stop-after``
   Restart or stop the preprocessing pipeline at a specific numbered step.

``--nX`` / ``--nY``
   Control horizontal domain decomposition during NetCDF block generation.

``--data-source``
   Selects ERA5 reanalysis or forecast GRIB2 ingestion.

Forecast Commands
-----------------

The packaged CLI currently exposes a thin forecast wrapper:

.. code-block:: bash

   um-earth forecast run \
     --config ./pte1b.yaml \
     --input-dir ./forecast_input/regridded_pte1b_20260412_00_tensors \
     --output-dir ./forecast_output

For production forecasting, operators often call
``run_frigate_prediction.py`` directly to access the full set of runtime
arguments, including device selection, forcing mode, and refinement mode.

Pipeline Commands
-----------------

``um-earth pipeline run``
   Runs config generation, topography, initial-condition prep, and optionally
   the forecast.

``um-earth pipeline frigate-prepare``
   Creates a FRIGATE-style prepared run directory from a KML file and date.

Example:

.. code-block:: bash

   um-earth pipeline frigate-prepare \
     --region-kml /path/to/pte1b.kml \
     --date 2026-04-12 \
     --data-source forecast \
     --forecast-input-dir /path/to/ECMWF_prediction_data/20260412 \
     --forecast-cycle 00 \
     --forecast-leads 0 6 12 18

Diagnostics Commands
--------------------

The diagnostics CLI supports both plot generation and CSV extraction:

.. code-block:: bash

   um-earth diagnostics plot ./forecast_output
   um-earth diagnostics extract-updrafts ./forecast_output --output updrafts.csv
   um-earth diagnostics plot-updrafts updrafts.csv --topo-file topo.pt --input-dir ./forecast_output
