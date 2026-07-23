# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""A ``print`` that reaches the egg web UI when one is driving the script.

:func:`webui_print` formats its arguments exactly like the builtin
:func:`print`, but instead of writing to ``stdout`` it hands the text to a
*sink* the UI installs. Headless — the CLI, a test, or any run with no sink
attached — it is a no-op, so the same call is silent outside the web UI and a
script can be sprinkled with UI-only progress messages that cost nothing in a
batch run. This mirrors :func:`egg.editable`: importable from the core package,
inert on its own, meaningful only when the UI is present.

The sink is process-global module state (see :func:`_set_sink`). The web UI's
run worker installs one that streams each message to the browser as it arrives;
the render worker installs one that folds it into the render's captured stdout.
Neither the CLI nor the server process installs a sink, so :func:`webui_print`
stays a no-op there.
"""

from __future__ import annotations

import io
from typing import Callable

# Installed by the web UI worker processes; ``None`` means headless (no-op).
_sink: Callable[[str], object] | None = None


def _set_sink(fn: Callable[[str], object] | None) -> None:
    """Install (or, with ``None``, clear) the process-wide UI print sink.

    Called by the web UI's worker processes; not part of the script-facing
    API. ``fn`` receives the already-formatted text of one
    :func:`webui_print` call (including its trailing ``end``).
    """
    global _sink
    _sink = fn


def webui_print(*args, sep: str = " ", end: str = "\n", **_kwargs) -> None:
    """``print`` that shows in the egg web UI while a run drives the script.

    Formats ``args`` like the builtin :func:`print` (honouring ``sep`` and
    ``end``) and routes the text to the UI: the live log during a run, the
    render's *stdout* panel while editing. A no-op when the script runs
    headless (the CLI, or any context with no UI attached), so the identical
    call is silent outside the web UI. A ``file`` / ``flush`` keyword is
    accepted and ignored — the destination is the UI, not a stream.
    """
    fn = _sink
    if fn is None:
        return
    buf = io.StringIO()
    print(*args, sep=sep, end=end, file=buf)
    fn(buf.getvalue())
