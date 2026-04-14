Installation
============

Python Package
--------------

The project declares a lightweight Python package in ``pyproject.toml``:

.. code-block:: bash

   pip install -e .

The published package metadata currently lists only the lightweight Python
dependencies used by the CLI and configuration helpers. The full runtime stack
for preprocessing and forecasting is larger and includes additional scientific
and domain-specific packages.

Core Python Dependencies
------------------------

Commonly required packages include:

* ``PyYAML``
* ``matplotlib``
* ``numpy``
* ``xarray``
* ``netCDF4``
* ``scipy``
* ``torch``

ECMWF Preprocessing Dependencies
--------------------------------

For ERA5-based preparation, the ECMWF pipeline also needs:

* ``cdsapi``

For forecast GRIB2 ingestion, the forecast path also needs:

* ``cfgrib``
* ``eccodes``

The ECMWF-specific requirements file lives at:

* ``um_earth/ecmwf_api/requirements.txt``

Forecast Runtime Dependencies
-----------------------------

The runtime forecast driver additionally depends on project-specific modeling
packages that are not vendored here:

* ``snapy``
* ``paddle``
* ``kintera``

These must already be installed in the execution environment for
``run_frigate_prediction.py`` to work.

ECMWF Credentials
-----------------

ERA5 downloads require a Climate Data Store API key. The project accepts either:

* a ``CDSAPI_KEY`` environment variable, or
* a ``~/.cdsapirc`` file

The forecast GRIB2 ingestion path does not require CDS credentials because it
starts from already-downloaded forecast files.

Documentation Dependencies
--------------------------

To build this documentation site:

.. code-block:: bash

   pip install -r docs/requirements.txt
   make -C docs html

The built site is written to:

.. code-block:: text

   docs/build/html/
