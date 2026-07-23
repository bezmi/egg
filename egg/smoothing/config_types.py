# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Typed, validated config objects for the composed smoother terms.

The composed-term configs (``interface_ortho``, ``interface_c2``,
``directional``) and the FAS schedule (``fas_params``) used to be untyped dicts
splatted into the sample builders, so a typo'd key was silently dropped and a
bad value crashed deep in the C++ core. They are now frozen dataclasses: a type
checker flags a wrong field name or value type at the call site.

Value ranges (which the type system cannot express) are checked at runtime, but
NOT in ``__post_init__``: construction stays cheap and never raises, so building
one of these in a live-reloaded script (the web UI re-runs the file on every
edit) does not spam errors while a value is half-typed. Instead the pipeline
collects every config's errors once, at the start of a run, and aborts. Use
:func:`config_errors` to collect a value's problems, or :meth:`coerce` to
normalize a validated value to an instance.

This module imports only :mod:`egg.enums` and :mod:`egg.errors` (both
egg-import-free), so it is safe to import anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, TypedDict, TypeVar

import numpy as np

from egg.enums import OrthoMode, enum_errors
from egg.errors import EggValidationError

if TYPE_CHECKING:
    pass

__all__ = [
    "InterfaceOrtho",
    "InterfaceC2",
    "Directional",
    "FasParams",
    "EnergyStencil",
    "FlatWire",
    "config_errors",
]

_T = TypeVar("_T", bound="_ConfigBase")


def _ge_error(value: object, lo: float, field: str, cls_name: str) -> str | None:
    """Return an error message if ``value`` is not a finite number >= ``lo``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{cls_name}.{field} must be a number, got {type(value).__name__}"
    if not np.isfinite(value):
        return f"{cls_name}.{field} must be finite, got {value!r}"
    if value < lo:
        return f"{cls_name}.{field} must be >= {lo}, got {value!r}"
    return None


class _ConfigBase:
    """Shared :meth:`coerce` and validation plumbing for the config dataclasses."""

    def _errors(self) -> list[str]:
        """Value-range problems with this instance (empty if valid)."""
        return []

    @classmethod
    def coerce(cls: type[_T], value: _T | Mapping[str, object] | None) -> _T | None:
        """Return ``value`` as an instance of ``cls`` (or ``None``).

        Accepts an instance (returned unchanged), a mapping (built into an
        instance), or ``None``. Used to normalize an already-validated value
        before use; call :func:`config_errors` first to validate.
        """
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
            known = {k: v for k, v in value.items() if k in allowed}
            return cls(**known)
        raise EggValidationError(
            f"{cls.__name__} must be a {cls.__name__} or a dict of its fields, "
            f"got {type(value).__name__}"
        )


@dataclass(frozen=True)
class InterfaceOrtho(_ConfigBase):
    """Block-interface orthogonality/continuity term (2D).

    Forwarded to :func:`egg.smoothing.interface_ortho.interface_ortho_samples`.
    """

    mode: OrthoMode = OrthoMode.NORMAL
    weight: float = 1.0
    n_layers: int = 3
    cluster_relax: float = 0.0

    def _errors(self) -> list[str]:
        errs = enum_errors(OrthoMode, self.mode, "InterfaceOrtho.mode")
        for e in (
            _ge_error(self.weight, 0.0, "weight", "InterfaceOrtho"),
            _ge_error(self.n_layers, 1.0, "n_layers", "InterfaceOrtho"),
            _ge_error(self.cluster_relax, 0.0, "cluster_relax", "InterfaceOrtho"),
        ):
            if e:
                errs.append(e)
        if isinstance(self.cluster_relax, (int, float)) and self.cluster_relax > 1.0:
            errs.append(
                f"InterfaceOrtho.cluster_relax must be in [0, 1], "
                f"got {self.cluster_relax!r}"
            )
        return errs


@dataclass(frozen=True)
class InterfaceC2(_ConfigBase):
    """Block-interface curvature-continuity term (2D).

    Forwarded to :func:`egg.smoothing.interface_c2.curvature_windows`.
    """

    weight: float = 1.0
    iface_boost: float = 1.0
    interface_only: bool = False
    singularity_weight: float = 0.0

    def _errors(self) -> list[str]:
        errs = [
            e
            for e in (
                _ge_error(self.weight, 0.0, "weight", "InterfaceC2"),
                _ge_error(self.iface_boost, 0.0, "iface_boost", "InterfaceC2"),
                _ge_error(
                    self.singularity_weight, 0.0, "singularity_weight", "InterfaceC2"
                ),
            )
            if e
        ]
        if not isinstance(self.interface_only, bool):
            errs.append(
                f"InterfaceC2.interface_only must be a bool, "
                f"got {type(self.interface_only).__name__}"
            )
        return errs


@dataclass(frozen=True)
class Directional(_ConfigBase):
    """Directional soft-energy terms over parallel chains and fan frames.

    Forwarded to
    :func:`egg.smoothing.directional.build_directional_samples`. The defaults
    are the pipeline's opinionated scale (``energy_scale=200``,
    ``lambda_fair=1``), so ``Directional()`` reproduces the auto-enabled
    behaviour and overriding one field keeps the others at that scale.
    """

    lambda_parallel: float = 1.0
    lambda_line: float = 1.0
    lambda_stem: float = 1.0
    lambda_fair: float = 1.0
    energy_scale: float = 200.0
    eps: float = 1e-12

    def _errors(self) -> list[str]:
        return [
            e
            for e in (
                _ge_error(getattr(self, f.name), 0.0, f.name, "Directional")
                for f in fields(self)
            )
            if e
        ]


@dataclass(frozen=True)
class FasParams(_ConfigBase):
    """FAS V-cycle schedule forwarded to ``session.run_fas``."""

    nu_pre: int = 2
    nu_post: int = 2
    nu_coarse: int = 32
    max_levels: int = 32

    def _errors(self) -> list[str]:
        return [
            e
            for e in (
                _ge_error(self.nu_pre, 0.0, "nu_pre", "FasParams"),
                _ge_error(self.nu_post, 0.0, "nu_post", "FasParams"),
                _ge_error(self.nu_coarse, 1.0, "nu_coarse", "FasParams"),
                _ge_error(self.max_levels, 1.0, "max_levels", "FasParams"),
            )
            if e
        ]


def config_errors(
    cls: type[_ConfigBase],
    value: object,
    name: str,
) -> list[str]:
    """Collect every problem with a config ``value`` without raising.

    ``value`` may be an instance of ``cls``, a mapping of its fields, or
    ``None`` (no term). Reports unknown keys and out-of-range values, so the
    pipeline can gather all config errors and abort once before it runs.
    """
    if value is None:
        return []
    if isinstance(value, cls):
        return value._errors()
    if isinstance(value, Mapping):
        allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        errs: list[str] = []
        unknown = set(value) - allowed
        if unknown:
            errs.append(
                f"{name} got unknown key(s) {sorted(unknown)}; "
                f"valid keys are {sorted(allowed)}"
            )
        known = {k: v for k, v in value.items() if k in allowed}
        errs += cls(**known)._errors()
        return errs
    return [
        f"{name} must be a {cls.__name__} or a dict of its fields, "
        f"got {type(value).__name__}"
    ]


# --- Internal wire shapes (SweepContext fields) --------------------------


class EnergyStencil(TypedDict):
    """Global (cell, corner)-sample stencil arrays built in ``build_sweep_context``."""

    gc: np.ndarray
    gn0: np.ndarray
    gn1: np.ndarray
    s0: np.ndarray
    s1: np.ndarray
    W_inv: np.ndarray


class CellStencil(TypedDict):
    """The dimension-general cell/sample membership tables from
    :func:`egg.smoothing.flat_context.cell_stencil`.

    ``gn``/``s`` hold one array per axis (``d`` entries); the ``m_*`` arrays are
    the flattened node/sample/role membership lists."""

    gc: np.ndarray
    gn: list[np.ndarray]
    s: list[np.ndarray]
    m_node: np.ndarray
    m_sid: np.ndarray
    m_role: np.ndarray
    nc: int
    ns: int
    ncell: int


class FlatWire(TypedDict):
    """The ragged C++ wire from :func:`egg.smoothing.flat_context.build_flat_context`.

    The per-group and energy-stencil dicts are genuinely heterogeneous (``int``
    counts alongside ``np.ndarray`` tables keyed by kernel-template presence), so
    their values are typed ``object`` rather than enumerated.
    """

    groups: list[dict[str, object]]
    energy_stencil: dict[str, object]
