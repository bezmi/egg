# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Closed value sets for config knobs, as string enums.

Each knob with a fixed set of valid strings is a :class:`enum.StrEnum`, so a
wrong value is a static type error at the call site. The members are real
strings, so they compare equal to their value and flow through the C++ wire and
downstream string checks unchanged. :func:`coerce_enum` accepts either a member
or a plain string (validated), so scripts that still pass strings keep working
and an invalid value is rejected before any solving runs.

This module imports only :mod:`egg.errors` (itself egg-import-free), so it is
safe to import anywhere.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar

from egg.errors import EggValidationError

__all__ = [
    "OrthoMode",
    "TmopMetric",
    "TmopSmoother",
    "ControlOrtho",
    "PresmoothSmoother",
    "Device",
    "coerce_enum",
    "enum_errors",
]


class OrthoMode(StrEnum):
    """``interface_ortho`` sampling mode."""

    NORMAL = "normal"
    CONTINUOUS = "continuous"


class TmopMetric(StrEnum):
    """TMOP objective: scale-invariant shape, or shape plus target size."""

    SHAPE = "shape"
    SHAPE_SIZE = "shape_size"


class TmopSmoother(StrEnum):
    """Which smoother runs the TMOP quality phase."""

    JACOBI = "jacobi"
    FAS = "fas"
    CONTROL_POINT = "control_point"


class ControlOrtho(StrEnum):
    """Seam-orthogonality mode for the control-point smoother."""

    OFF = "off"
    PENALTY = "penalty"
    HARD = "hard"


class PresmoothSmoother(StrEnum):
    """Smoother for the control-point pre-fit nodal smooth."""

    AUTO = "auto"
    JACOBI = "jacobi"
    FAS = "fas"


class Device(StrEnum):
    """Compute device selection."""

    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


_E = TypeVar("_E", bound=StrEnum)


def coerce_enum(enum_cls: type[_E], value: object, field: str) -> _E:
    """Return ``value`` as a member of ``enum_cls``, or raise.

    Accepts a member (returned unchanged) or a plain string (looked up by
    value). An invalid value raises :class:`EggValidationError` naming the
    valid set, so a typo fails before the pipeline runs.
    """
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            pass
    valid = [e.value for e in enum_cls]
    raise EggValidationError(f"{field} must be one of {valid}, got {value!r}")


def enum_errors(enum_cls: type[_E], value: object, field: str) -> list[str]:
    """Return a one-item message list if ``value`` is not a valid ``enum_cls``.

    The non-raising form of :func:`coerce_enum`, for collecting all config
    problems before the pipeline aborts.
    """
    if isinstance(value, enum_cls):
        return []
    if isinstance(value, str):
        try:
            enum_cls(value)
            return []
        except ValueError:
            pass
    valid = [e.value for e in enum_cls]
    return [f"{field} must be one of {valid}, got {value!r}"]
