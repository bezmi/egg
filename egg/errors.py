# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Package exception types and the accumulate-then-raise validation helper.

Everything a user's grid-generation script can get wrong should fail here, at
the point the mistake is made, with a message naming the offending field,
rather than crashing deep in the solver after a long run. This module imports
nothing from ``egg`` so it is safe to import at the top of any other module.

:class:`EggValidationError` multiply-inherits :class:`ValueError`, so existing
``except ValueError`` handlers and callers keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "EggError",
    "EggValidationError",
    "Diagnostic",
    "raise_if",
]


class EggError(Exception):
    """Base class for every exception egg raises on purpose."""


class EggValidationError(EggError, ValueError):
    """A user input was rejected before any expensive work ran.

    Inherits :class:`ValueError` so code that already catches ``ValueError``
    (and the many call sites that ``raise ValueError`` today) stay compatible.
    """


@dataclass
class Diagnostic:
    """One rejected-input reason, collected so several can be reported at once.

    ``msg`` is the human-facing explanation (it should name the offending
    field); ``where`` optionally localises it (node ids, an ``(block, axis)``
    pair, etc.) for a front-end to highlight.
    """

    msg: str
    where: tuple[int, ...] = ()


def raise_if(diags: Iterable[Diagnostic]) -> None:
    """Raise a single :class:`EggValidationError` if any diagnostics were collected.

    The accumulate-then-raise pattern (report every problem found in one pass,
    then fail once) so a user fixing a topology sees all the mismatches at once
    instead of one-per-run.
    """
    msgs = [d.msg for d in diags]
    if msgs:
        raise EggValidationError("; ".join(msgs))
