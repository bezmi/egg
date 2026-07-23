# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Solver worker: one run in its own process (fasthtml-free).

The server starts ``python worker.py <script-file> [<script-path>]`` per
run, so the pipeline — script exec, the SYCL runtime, every sweep — never
executes inside the server process; a hung or crashed solve can't take
the UI down, and a hard kill is always available. The script must have
registered its pipeline explicitly via ``egg_webui.run(grid, steps)``;
this worker only consumes what the script handed over.

Frames go to the parent as pickles over the *original* stdout; fd 1 is
re-pointed at stderr before the script runs, so plain ``print``\\ s from
the script or the pipeline land in the server log instead of corrupting
the frame channel. A script that wants its progress *in the UI* calls
:func:`egg.webui_print` (``egg_webui.print``) instead; a sink installed
here forwards each such message to the parent as ``("print", text)``.
Child → parent messages:

- ``("step", phase, info, global_nodes)`` after every pipeline chunk,
- ``("print", text)`` an ``egg.webui_print`` call from the script/pipeline,
- ``("done",)`` on normal end (including after a stop),
- ``("fatal", msg)`` script error / nothing to smooth,
- ``("error", traceback)`` the pipeline raised.

A ``stop`` line on stdin stops the run at the next chunk boundary; stdin
EOF (the parent died, e.g. a dev-server reload) counts as stop too, so no
orphan keeps computing.
"""

from __future__ import annotations

import os
import pickle
import select
import sys
import traceback
from pathlib import Path


def _claim_stdout():
    """The frame channel; anything else writing fd 1 goes to stderr."""
    chan = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    return chan


def _stop_requested() -> bool:
    """True once the parent wrote to stdin ("stop") or closed it (EOF)."""
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(readable)


def main() -> None:
    chan = _claim_stdout()

    def send(msg: tuple) -> None:
        pickle.dump(msg, chan, protocol=pickle.HIGHEST_PROTOCOL)
        chan.flush()

    try:
        import numpy as np

        from .scene import exec_script

        from egg._webui_print import _set_sink

        # egg.webui_print(...) anywhere in the script or the pipeline streams to
        # the UI; installed before the exec so module-level messages count too.
        _set_sink(lambda text: send(("print", text)))

        code = Path(sys.argv[1]).read_text()
        path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
        # A resume run passes the cached node coordinates (a .npy) as argv[3];
        # the grid is seeded from them before the (freshly re-exec'd) stages run.
        resume_npy = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
        ns, _out, err = exec_script(code, path)
        if err is not None:
            send(("fatal", "script error — fix it first"))
            return
        reg = ns.get("__egg_webui_run__")
        if reg is None:
            send(
                (
                    "fatal",
                    "script registered no run — call egg_webui.run(grid, steps)"
                    ' in its `if __name__ == "__egg_webui__":` block',
                )
            )
            return
        grid, steps = reg
        # Seed the grid from the resume cache before the generator is iterated
        # (generate_steps builds its MeshState lazily on the first step, so this
        # is the state the stages see). Shape-guarded: an edit that changed the
        # topology (different node count) can't be resumed, so run fresh instead.
        if resume_npy:
            X = np.load(resume_npy)
            if X.shape == grid.global_nodes.shape:
                grid.global_nodes[...] = X
                for bi, blk in enumerate(grid.blocks):
                    blk.nodes[...] = X[grid.block_dof_maps[bi]]

        from .scene import control_net_state

        for phase, info in steps:
            stopped = _stop_requested()
            # Net state first: the reader renders on the step frame, so the
            # lattice overlay animates in the same frame as the grid.
            net_state = control_net_state(grid)
            if net_state is not None:
                send(("net_state", net_state))
            send(("step", phase, dict(info), np.asarray(grid.global_nodes)))
            if stopped:
                break
        # A control_point run leaves its solved net on the grid; ship the
        # persisted form so the server can overlay it and serve the export.
        net = getattr(grid, "control_net", None)
        if net is not None:
            import io

            from egg.io import save_control_net

            buf = io.BytesIO()
            save_control_net(net, buf)
            send(("net", buf.getvalue()))
        send(("done",))
    except Exception:
        send(("error", traceback.format_exc()))


if __name__ == "__main__":
    main()
