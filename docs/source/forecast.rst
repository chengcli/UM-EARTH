Forecast Runtime
================

Driver
------

Forecast execution is handled by:

* ``run_frigate_prediction.py``

This script is the source of truth for runtime forecast behavior, including:

* restart tensor loading
* topography loading
* hydrostatic adjustment
* low-resolution ghost forcing
* staged refinement mode

Required Inputs
---------------

The runtime needs:

* a simulation YAML
* a directory or file containing the prepared ``.part`` tensor input
* a directory containing the topography ``*.pt`` products

Default Runtime Modes
---------------------

The script supports two broad execution styles:

``--refinement-mode staged``
   Runs the original multi-stage refinement workflow.

``--refinement-mode none``
   Runs on a single mesh for the full forecast.

Forcing modes:

``--forcing-mode nudge``
   Nudges the whole domain toward later ECMWF states.

``--forcing-mode ghost``
   Refreshes lateral ghost zones from later ECMWF states.

Low-Resolution Production Pattern
---------------------------------

A common operational launch for prepared forecast tensors is:

.. code-block:: bash

   python run_frigate_prediction.py \
     -c /path/to/pte1b.yaml \
     -i /path/to/regridded_pte1b_20260412_00_tensors \
     -o /path/to/forecast_lowres_ghost_24h \
     --topography-dir /path/to/topography/products \
     --device cuda \
     --refinement-mode none \
     --forcing-mode ghost \
     --prediction-duration 86400

Outputs
-------

Typical runtime outputs include:

* ``<region>.00000.restart``
* ``<region>.final.restart``
* ``<region>.out1.*.nc``
* ``<region>.out2.*.nc``
* ``run.log``

Device Selection
----------------

The driver supports:

* ``--device cpu``
* ``--device cuda``
* ``--device auto``

Distributed initialization is handled internally using ``torch.distributed`` and
the backend implied by the mesh configuration.

Monitoring Detached Runs
------------------------

Common operator checks:

.. code-block:: bash

   pgrep -af 'run_frigate_prediction.py|forecast_lowres_ghost_24h'
   ps -p <PID> -o pid,etime,%cpu,%mem,stat,cmd
   tail -n 80 /path/to/forecast_output/run.log
