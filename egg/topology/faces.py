# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Named block faces (compass nomenclature) for the topology front-end.

Every face of a structured block is an ``(axis, side)`` pair; :class:`Face`
gives the six of them the gdtk/Eilmer compass names (``west``/``east`` = i-/+,
``south``/``north`` = j-/+, ``bottom``/``top`` = k-/+). 2D grids use the first
four, 3D adds ``bottom``/``top``. There are no compass names beyond 3D, so for
``d > 3`` (which structured grids never are in practice) the raw ``(axis, side)``
integer form remains the universal fallback everywhere a Face is accepted.
"""

from __future__ import annotations

from enum import StrEnum


class Face(StrEnum):
    """A named block face; resolves to an ``(axis, side)`` selector via
    :attr:`axis` / :attr:`side`. Accepted (as the enum or its name string)
    anywhere a topology method takes an ``(axis, side)`` pair."""

    WEST = "west"
    EAST = "east"
    SOUTH = "south"
    NORTH = "north"
    BOTTOM = "bottom"
    TOP = "top"

    @property
    def axis(self) -> int:
        """Logical axis the face lies on (0 = i, 1 = j, 2 = k)."""
        return _AXIS_SIDE[self][0]

    @property
    def side(self) -> int:
        """0 = low side, 1 = high side."""
        return _AXIS_SIDE[self][1]


_AXIS_SIDE: dict[Face, tuple[int, int]] = {
    Face.WEST: (0, 0),
    Face.EAST: (0, 1),
    Face.SOUTH: (1, 0),
    Face.NORTH: (1, 1),
    Face.BOTTOM: (2, 0),
    Face.TOP: (2, 1),
}

#: Face name strings accepted wherever a :class:`Face` is.
FACE_NAMES: frozenset[str] = frozenset(f.value for f in Face)


def is_face(value: object) -> bool:
    """Whether ``value`` is a :class:`Face` or one of its name strings."""
    return isinstance(value, Face) or (isinstance(value, str) and value in FACE_NAMES)


def as_face(value: Face | str) -> Face:
    """Coerce a :class:`Face` or its name to a Face; raise on anything else."""
    if isinstance(value, Face):
        return value
    if isinstance(value, str) and value in FACE_NAMES:
        return Face(value)
    raise ValueError(f"not a face name: {value!r}; expected one of {sorted(FACE_NAMES)}")
