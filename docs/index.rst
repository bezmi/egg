egg
===

egg aims to be an excellent grid generator.

The Python front-end declares geometry and a rough multiblock topology,
then runs the pipeline (TFI init -> untangle -> TMOP quality optimisation)
on the C++ backend. See the ``examples/`` tree for complete 2D and 3D
set-ups.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   webui
   cad

.. toctree::
   :maxdepth: 2
   :caption: Worked examples

   examples/egg
   examples/egg-svg
   examples/capsule-fire-ii
   examples/capsule-phoebus

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/geometry
   api/topology
   api/pipeline
   api/smoothing
   api/bindings
   api/core
