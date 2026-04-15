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
* a directory or file containing the prepared restart input
* a directory containing the topography ``*.pt`` products

Prepared restart inputs may be either:

* a single ``.part`` file for a one-block run
* a directory containing one bundled ``.restart`` tarball plus per-rank
  ``.part`` members for a decomposed run

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

Single-GPU Low-Resolution Run
-----------------------------

The simplest production path is a single-block low-resolution forecast on one
GPU. This uses the same preparation pipeline as the original forecast workflow
and launches ``run_frigate_prediction.py`` directly.

Preparation pattern:

.. code-block:: bash

   python prepare_initial_condition.py \
     pte1b \
     --region-kml /home/chengcli/data/2025.FRIGATE/pte1b.kml \
     --config /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-12/pte1b.yaml \
     --output-base /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-12/forecast_input \
     --data-source forecast \
     --forecast-input-dir /home/chengcli/data/2025.FRIGATE/ECWMF_prediction_data/20260412 \
     --forecast-cycle 00 \
     --forecast-leads 0 6 12 18 \
     --timeout 3600

This produces a single-block tensor input such as:

* ``regridded_pte1b_20260412_00_tensors/regridded_pte1b_20260412_00_block_0_0.part``

Runtime pattern:

.. code-block:: bash

   python run_frigate_prediction.py \
     -c /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-12/pte1b.yaml \
     -i /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-12/forecast_input/regridded_pte1b_20260412_00_tensors \
     -o /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-12/forecast_lowres_ghost_24h \
     --topography-dir /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-12/topography/products \
     --device cuda \
     --refinement-mode none \
     --forcing-mode ghost \
     --prediction-duration 86400

Two-GPU Low-Resolution Run
--------------------------

For a two-GPU run, the preparation stage must decompose the regridded file into
two x2 slabs and emit a bundled restart tarball. The simulation YAML must also
set ``distribute.nb2: 2`` and ``distribute.nb3: 1``.

Preparation pattern:

.. code-block:: bash

   python prepare_initial_condition.py \
     pte1b \
     --region-kml /home/chengcli/data/2025.FRIGATE/pte1b.kml \
     --config /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/pte1b.yaml \
     --output-base /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_input \
     --data-source forecast \
     --forecast-input-dir /home/chengcli/data/2025.FRIGATE/ECWMF_prediction_data/20260413 \
     --forecast-cycle 00 \
     --forecast-leads 0 6 12 18 \
     --nY 2 --nX 1 \
     --timeout 7200

The resulting tensor directory contains:

* ``regridded_pte1b_20260413_00.block0.00000.part``
* ``regridded_pte1b_20260413_00.block1.00000.part``
* ``regridded_pte1b_20260413_00.restart``

At runtime, pass the tensor directory or the bundled ``.restart`` file. Each
rank loads its own block from the restart tarball by rank id.

Runtime pattern:

.. code-block:: bash

   env CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29532 PYTHONFAULTHANDLER=1 \
     torchrun --nproc-per-node=2 \
     /home/chengcli/scix/workspace/UM-EARTH/run_frigate_prediction.py \
       -c /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/pte1b.yaml \
       -i /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_input/regridded_pte1b_20260413_00_tensors \
       -o /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/forecast_lowres_ghost_24h_2gpu \
       --topography-dir /home/chengcli/data/2025.FRIGATE/runs/pte1b-2026-04-13/topography/products \
       --device cuda \
       --refinement-mode none \
       --forcing-mode ghost \
       --prediction-duration 86400

Two-GPU Notes
-------------

The two-GPU workflow assumes:

* slab decomposition along ``x2``
* exactly two ranks for the current operational path
* ``torchrun --nproc-per-node=2``
* a bundled restart tarball produced by the preparation stage

When inspecting GPUs with ``nvidia-smi``, you may see three Python processes:

* one ``torchrun`` launcher
* rank 0 worker
* rank 1 worker

That is expected and does not mean three simulation ranks are running.

Low-Resolution Detached Pattern
-------------------------------

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

For decomposed two-GPU runs, the prepared input side also includes:

* ``<basename>.block0.00000.part``
* ``<basename>.block1.00000.part``
* ``<basename>.restart``

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
