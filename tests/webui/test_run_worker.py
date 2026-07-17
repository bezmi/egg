# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Run-worker frame protocol: step frames, streamed control-net states, and
the terminal net/done frames of a control_point run."""

import os
import pickle
import subprocess
import sys

import numpy as np
import pytest


def _has_cpp() -> bool:
    try:
        from egg._cpp import cpp_core  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cpp(),
    reason="egg._cpp.cpp_core not built (requires cmake build)",
)

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")

_SCRIPT = """
import sys

sys.path.insert(0, {examples!r})
from topologies import build_circle_in_rectangle

import egg.webui as egg_webui
from egg.pipeline import generate_steps

topo, ents = build_circle_in_rectangle()
grid = topo.initialize_grid()

if __name__ == "__egg_webui__":
    egg_webui.run(
        grid,
        generate_steps(
            grid,
            tmop_smoother="control_point",
            control_presmooth=20,
            control_max_outer=5,
        ),
    )
"""


def test_control_run_streams_net_state_frames(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(
        _SCRIPT.format(examples=os.path.join(_REPO, "examples", "2D", "circles"))
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "egg.webui.worker", str(script), ""],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        cwd=_REPO,
    )
    frames = []
    try:
        while True:
            try:
                frames.append(pickle.load(proc.stdout))
            except EOFError:
                break
    finally:
        proc.stdin.close()
        proc.wait(timeout=60)

    kinds = [f[0] for f in frames]
    assert kinds[-1] == "done"
    assert "net" in kinds  # persisted npz for the export
    assert "net_state" in kinds  # streamed overlay updates

    # Every net_state immediately precedes a step frame (the reader updates
    # the overlay, then renders on the step), and its lattices are 2D blocks.
    for i, f in enumerate(frames):
        if f[0] != "net_state":
            continue
        assert frames[i + 1][0] == "step"
        assert all(
            isinstance(c, np.ndarray) and c.ndim == 3 and c.shape[-1] == 2 for c in f[1]
        )
    # The control phase streams the net; step frames for the control phase
    # exist and each is paired with a preceding net_state.
    ctrl_steps = [
        i for i, f in enumerate(frames) if f[0] == "step" and f[1] == "control"
    ]
    assert ctrl_steps
    assert all(frames[i - 1][0] == "net_state" for i in ctrl_steps)
