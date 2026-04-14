Overview
========

Architecture
------------

UM-EARTH is split into a few clear layers:

``um_earth.regions``
   Parses KML- and CSV-defined forecast regions into a common in-memory shape.

``um_earth.configuration``
   Renders the simulation YAML used both by preprocessing scripts and by the
   runtime forecast driver.

``um_earth.frigate_pipeline``
   Builds a prepared FRIGATE run directory from a KML file and date, including
   domain expansion, topography products, and ECMWF-derived initial conditions.

``um_earth.ecmwf_api``
   Contains the ECMWF data curation pipeline. It supports both ERA5 reanalysis
   inputs and forecast GRIB2 ingestion, then converts those data into the
   regridded NetCDF and ``.part`` tensor artifacts used by the forecast driver.

``run_frigate_prediction.py``
   Executes the actual forecast using ``snapy`` and optional low-resolution
   ghost forcing or staged refinement.

Typical Workflow
----------------

The normal operator flow is:

1. Define or select a region.
2. Generate or reuse a simulation YAML.
3. Prepare topography products.
4. Prepare initial conditions from ECMWF ERA5 or forecast data.
5. Launch the forecast run.
6. Inspect or post-process forecast outputs.

There are two common entry patterns:

``um-earth pipeline frigate-prepare``
   High-level FRIGATE workflow that writes a prepared run tree under a run
   directory.

``um-earth init prepare``
   Lower-level initial-condition preparation command that runs the ECMWF
   preprocessing pipeline directly.

Prepared Run Layout
-------------------

A typical prepared FRIGATE run directory contains:

.. code-block:: text

   <run-dir>/
     <region>.yaml
     region_digest.json
     run_manifest.yaml
     plots/
     topography/
       raw/
       products/
     era5/ or forecast_input/
       ... NetCDF intermediates ...
       ... regridded products ...
       ... *_tensors/*.part ...

Data Contracts
--------------

The most important interface boundary in the project is between preprocessing
and the forecast runtime:

* Preprocessing must emit a ``.part`` tensor artifact containing a ``hydro_w``
  tensor with four time slices.
* The forecast runtime consumes that artifact plus the prepared topography
  products.

That contract allows the ECMWF ingest stage to change independently from the
runtime forecast driver as long as the tensor output layout remains stable.
