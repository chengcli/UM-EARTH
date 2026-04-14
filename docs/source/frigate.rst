FRIGATE Workflow
================

Purpose
-------

The FRIGATE workflow packages the multi-step domain-preparation process into a
repeatable run directory layout. It is implemented in
:mod:`um_earth.frigate_pipeline`.

What ``frigate-prepare`` Does
-----------------------------

``um-earth pipeline frigate-prepare`` performs the following tasks:

1. Load and normalize the region from a KML file.
2. Expand the region bounds to enforce a minimum horizontal extent.
3. Compute a prepared low-resolution domain at the target base resolution.
4. Render the simulation YAML for that domain.
5. Build topography products at multiple target resolutions.
6. Run the initial-condition preparation pipeline.
7. Write a digest, manifest, and verification plots into the run tree.

Prepared Domain Rules
---------------------

The FRIGATE code chooses a prepared domain using:

* minimum horizontal size in degrees
* a base target resolution in kilometers
* a fixed low-resolution vertical depth and number of vertical cells

The most important defaults are defined in :mod:`um_earth.frigate_pipeline`:

* ``DEFAULT_MIN_DOMAIN_DEGREES``
* ``DEFAULT_TARGET_RESOLUTIONS_KM``
* ``DEFAULT_X1_MAX_METERS``
* ``DEFAULT_NX1``

Run Tree Structure
------------------

The prepared run directory usually looks like:

.. code-block:: text

   runs/<region-id>-<date>/
     <region-id>.yaml
     region_digest.json
     run_manifest.yaml
     plots/
     topography/
       raw/
       products/
     era5/ or forecast_input/

Important Files
---------------

``region_digest.json``
   Captures the native and padded region bounds, the generated low-resolution
   grid, and the requested preparation settings.

``run_manifest.yaml``
   Records the important paths for the prepared run, including:

   * simulation input YAML
   * topography products
   * data source
   * forecast input directory and cycle, when applicable

``plots/``
   Contains quick visual verification products for the domain and topography.

ERA5 Vs Forecast Inputs
-----------------------

The FRIGATE wrapper now supports both input paths:

``data_source=era5``
   Downloads ERA5 using the existing CDS-backed workflow.

``data_source=forecast``
   Ingests local forecast GRIB2 files and writes the same downstream tensor
   contract used by the forecast runtime.

Operational Pattern
-------------------

Typical operator flow:

1. Run ``frigate-prepare`` once for a target date.
2. Inspect the prepared YAML, plots, and tensor outputs.
3. Launch ``run_frigate_prediction.py`` against the prepared run tree.

This split keeps heavyweight forecast execution separate from domain and input
preparation, which is especially useful for detached GPU production runs.
