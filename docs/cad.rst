CAD import (3D)
===============

:mod:`egg.io.cad` turns build123d / OCCT shapes into egg geometry entities:
each face of an imported STEP/BREP solid becomes a
:class:`~egg.geometry.surfaces3d.BSplineSurface` carrying its UV trim loops,
and straight edges become :class:`~egg.geometry.analytic3d.Line3`. Domain
volumes and boolean carving of the flow region are authored in build123d
directly; the resulting solid is handed to the adapter. The heavy conversion
(OCCT surface to NURBS, trim wires to UV polylines) runs once at import, so
the solver never sees OCCT.

Requirements
------------

STEP/BREP import and the other OCCT-backed operations need
`build123d <https://build123d.readthedocs.io>`_, which is **not** a core
dependency. Install the optional ``cad`` group:

.. code-block:: bash

   uv sync --group cad

Only the CAD import path needs it. 3D set-ups built from analytic primitives
(:class:`~egg.geometry.analytic3d.Sphere`, ``Plane``, ``Line3``) or from
hand-authored NURBS surfaces carry no OCCT dependency, and the device solve
carries none regardless.

.. note::

   A ``--force-reinstall`` C++ rebuild drops the optional groups, and re-adding
   one with ``uv sync --group cad`` reverts the core to its default precision.
   To hold a precision (fp32 is recommended on the GPU) and the groups together,
   set ``SKBUILD_CMAKE_DEFINE`` once; see ``DEVELOPING.md``.

API
---

.. automodule:: egg.io.cad
   :members:
