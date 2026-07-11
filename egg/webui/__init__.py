# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Ambient API for scripts running inside the egg web UI.

Scripts exec with ``__name__ == "__egg_webui__"`` and ``egg`` importable,
so inside the guard a script can do::

    if __name__ == "__egg_webui__":
        import egg.webui as egg_webui

        egg_webui.run(grid, generate_steps(grid, config=cfg,
                                           untangle_direct=False))

which registers the grid and the *lazy, unconsumed* steps generator with
the UI: the run button consumes it and streams the relaxing mesh — the
web equivalent of handing ``steps`` to the CLI driver's ``finish()``.
Nothing runs at registration time, so live re-renders while typing stay
cheap. Without this call the run button does nothing: the UI never
invents a pipeline behind the script's back.
"""

from __future__ import annotations

from egg import webui_print

_pending: tuple | None = None


def print(*args, **kwargs) -> None:  # noqa: A001 — ambient UI print, shadows the builtin by design
    """``print`` that streams to the web UI's live log while a run is going.

    A thin alias for :func:`egg.webui_print` for use inside a script's
    ``if __name__ == "__egg_webui__":`` block, alongside :func:`run` and
    :func:`editable`. Formats like the builtin :func:`print`; a no-op when the
    script runs headless, so the same call is silent outside the UI. Prefer
    ``egg.webui_print`` in code shared with the CLI ``main()`` (that module
    can't import ``egg_webui``); both feed the same UI sink.
    """
    webui_print(*args, **kwargs)


def editable(value, *, choices=None, label=None):
    """Mark a literal as editable from the UI's run-parameters panel.

    Returns ``value`` unchanged — at runtime this is an identity function.
    The panel works on the *source*: it finds ``editable(...)`` calls in
    the script's AST and exposes the first argument's literal for editing
    in place (the surrounding call, formatting, and comments survive every
    edit). Works anywhere in the script, not just the guard dict::

        N_RINGS = egg_webui.editable(3, label="dipole rings")   # int box
        a = dict(
            metric=egg_webui.editable("shape_size",
                                      choices=["shape", "shape_size"]),
            dipole=egg_webui.editable(True),                    # checkbox
        )

    The input type follows the literal: bool -> checkbox, int/float ->
    numeric text box, str -> text box; ``choices`` (a list of literals of
    the same type) renders a dropdown instead. ``label`` overrides the
    name shown in the panel, which otherwise comes from the assignment
    target, dict key, or keyword argument the call sits in.
    """
    return value


def params(mapping=None, /, **kwargs):
    """Mark the run-parameters dict for the web UI's run panel.

    Returns the parameters as a plain dict (``mapping`` merged with
    ``kwargs``) — at runtime just ``dict(...)``. The web UI finds the
    ``params(...)`` call in the source and turns its entries into the
    run-parameters strip (bool -> checkbox, int/float -> number box, str ->
    text box), editing each in place. Use it instead of a bare
    ``a = dict(...)`` so the panel binds to an *explicit* marker, not to
    "the first dict in the guard"::

        a = egg_webui.params(
            sweeps_per_delta=20,
            metric=egg_webui.editable("shape", choices=["shape", "shape_size"]),
            device="cpu",
        )

    ``editable(...)`` still works inside for choices/labels; an entry that is
    not a plain literal is shown read-only.
    """
    out = dict(mapping or {})
    out.update(kwargs)
    return out


def run(grid, steps) -> None:
    """Hand the UI a grid and the pipeline steps that will relax it.

    ``steps`` is an unconsumed generator (e.g. from
    :func:`egg.pipeline.generate_steps`, with ``untangle_direct=False``
    so the untangle phase animates); the UI drives it when — and only
    when — the run button is pressed.
    """
    global _pending
    _pending = (grid, steps)


def _reset() -> None:
    global _pending
    _pending = None


def _take() -> tuple | None:
    global _pending
    p, _pending = _pending, None
    return p
