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

    from egg import TopologyBuilder, Untangle, JacobiSmoother, run_pipeline

Build a topology with :class:`TopologyBuilder`, then run a stage list with
:func:`run_pipeline` / :func:`generate_steps` (or the whole pipeline with
:func:`generate`). The pipeline stages, the smoother config objects
(:class:`InterfaceOrtho`, :class:`InterfaceC2`, :class:`Directional`,
:class:`FasParams`), the config enums (:class:`TmopMetric`, :class:`Device`,
...), and the exception types (:class:`EggValidationError`) are all re-exported
here too. Deeper layers live in their subpackages: geometry entities in
:mod:`egg.geometry`, TMOP targets and the smoother bridge in
:mod:`egg.smoothing`, visualization and export in :mod:`egg.io`.
"""

_EXPORTS = {
    "TopologyBuilder": "egg.topology.builder",
    "Block": "egg.topology.builder",
    "BlockFace": "egg.topology.builder",
    "BlockArray": "egg.topology.builder",
    "Face": "egg.topology.faces",
    "ExplicitTopology": "egg.topology.explicit",
    "editable": "egg.topology.explicit",
    "PipelineConfig": "egg.pipeline",
    "PipelineReport": "egg.pipeline",
    "generate": "egg.pipeline",
    "generate_steps": "egg.pipeline",
    "run_pipeline": "egg.pipeline",
    "drain_steps": "egg.pipeline",
    "build_topology_target": "egg.smoothing",
    "webui_print": "egg._webui_print",
    # Bundle an out-of-tree file into a .eggy archive (opt-in; use .path).
    "file_import": "egg.io.deps",
    # Pipeline stages and smoothers, so a script builds a stage list from a
    # single `import egg` (or `from egg import Untangle, JacobiSmoother, ...`).
    "Stage": "egg.pipeline",
    "MeshState": "egg.pipeline",
    "Untangle": "egg.pipeline",
    "JacobiSmoother": "egg.pipeline",
    "FasSmoother": "egg.pipeline",
    "Presmooth": "egg.pipeline",
    "ControlPointSmoother": "egg.pipeline",
    "Pin": "egg.pipeline",
    "Respace": "egg.pipeline",
    "Refit": "egg.pipeline",
    "Resample": "egg.pipeline",
    "Save": "egg.pipeline",
    # Validated config objects for the composed smoother terms.
    "InterfaceOrtho": "egg.smoothing.config_types",
    "InterfaceC2": "egg.smoothing.config_types",
    "Directional": "egg.smoothing.config_types",
    "FasParams": "egg.smoothing.config_types",
    # Closed value sets for config knobs.
    "OrthoMode": "egg.enums",
    "TmopMetric": "egg.enums",
    "TmopSmoother": "egg.enums",
    "ControlOrtho": "egg.enums",
    "PresmoothSmoother": "egg.enums",
    "Device": "egg.enums",
    # Exception types.
    "EggError": "egg.errors",
    "EggValidationError": "egg.errors",
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
