# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""egg: structured multi-block grid generation with TMOP smoothing/untangling.

The main entry points are re-exported here (lazily, so ``import egg`` stays
light)::

    from egg import TopologyBuilder, PipelineConfig, generate

Build a topology with :class:`TopologyBuilder`, then run the full pipeline
with :func:`generate` (or drive it step-wise with :func:`generate_steps` /
:func:`run_pipeline`). Deeper layers live in their subpackages: geometry
entities in :mod:`egg.geometry`, TMOP targets and the smoother bridge in
:mod:`egg.smoothing`, visualization and export in :mod:`egg.io`.
"""

_EXPORTS = {
    "TopologyBuilder": "egg.topology.builder",
    "PipelineConfig": "egg.pipeline",
    "PipelineReport": "egg.pipeline",
    "generate": "egg.pipeline",
    "generate_steps": "egg.pipeline",
    "run_pipeline": "egg.pipeline",
    "drain_steps": "egg.pipeline",
    "build_boundary_layer_target": "egg.smoothing",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'egg' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
