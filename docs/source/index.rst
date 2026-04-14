UM-EARTH Documentation
======================

UM-EARTH is a regional weather forecast pipeline built around ECMWF atmospheric
data preprocessing, terrain preparation, and ``snapy``-based forecast execution.
The codebase supports both generic CLI-driven workflows and a FRIGATE-specific
prepared-run layout.

This documentation site is organized around the actual operator workflow:
prepare a domain, build initial conditions, run a forecast, and inspect the
outputs. A compact API reference is also included for the core Python modules
that define those workflows.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   installation
   cli
   frigate
   ecmwf_pipeline
   forecast
   diagnostics
   api

Indices And Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
