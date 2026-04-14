Diagnostics And Analysis
========================

CLI Entry Points
----------------

Diagnostics are exposed through ``um-earth diagnostics``:

* ``plot``
* ``extract-updrafts``
* ``plot-updrafts``

Plot Generation
---------------

Generate the standard plot bundle from a forecast output directory:

.. code-block:: bash

   um-earth diagnostics plot /path/to/forecast_output

Optional flags support:

* custom output directory
* combined PDF output
* topography directory
* location label
* plotting all times instead of the default subset

Updraft Extraction
------------------

Extract updraft segments to CSV:

.. code-block:: bash

   um-earth diagnostics extract-updrafts \
     /path/to/forecast_output \
     --output updrafts.csv \
     --threshold 1.0

Plot Updraft Locations
----------------------

Overlay extracted updrafts on terrain:

.. code-block:: bash

   um-earth diagnostics plot-updrafts \
     updrafts.csv \
     --topo-file /path/to/pte1b_topo_2p4km.pt \
     --input-dir /path/to/forecast_output

Diagnostic Modules
------------------

The diagnostics package includes utilities for:

* vertical velocity
* horizontal velocity
* water paths
* theta-v
* LCL
* surface pressure
* topography overlays

Relevant modules live under:

* ``um_earth.diagnostics``

Operational Usage
-----------------

The diagnostics commands are designed to run after forecast completion, but they
can also be used on partially complete output directories when enough NetCDF
files are already available.
