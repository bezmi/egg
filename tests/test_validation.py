# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Pipeline preflight: bad configs and a bad grid abort before any solving.

The abort path raises in the mandatory validation stage, before the first
event and before any reduction, so these run without the C++ core.
"""

import os
import sys

import numpy as np
import pytest

from egg.errors import EggValidationError
from egg.pipeline import JacobiSmoother, Untangle, generate_steps
from egg.topology.builder import TopologyBuilder

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "examples", "2D", "circles")
)


def _grid():
    from topologies import build_circle_in_rectangle

    topo, _ents = build_circle_in_rectangle()
    return topo.initialize_grid()


def test_building_the_generator_does_not_validate():
    # Creating the generator (and constructing the stages) must not raise, even
    # with bad config: the web UI re-runs the script continuously.
    grid = _grid()
    generate_steps(grid, stages=[Untangle(), JacobiSmoother(metric="bad", chunk=0)])


def test_bad_config_aborts_before_init_with_all_errors():
    grid = _grid()
    gen = generate_steps(
        grid,
        stages=[
            Untangle(shrink=2.0),
            JacobiSmoother(metric="bad", chunk=0, interface_ortho={"weght": 1.0}),
        ],
    )
    with pytest.raises(EggValidationError) as ei:
        next(gen)
    msg = str(ei.value)
    # every problem is reported together, and nothing ran (no init event)
    assert "shrink" in msg
    assert "metric" in msg
    assert "chunk" in msg
    assert "weght" in msg


def test_non_finite_grid_aborts_before_init():
    grid = _grid()
    grid.global_nodes[0] = np.nan
    gen = generate_steps(grid, stages=[Untangle(), JacobiSmoother()])
    with pytest.raises(EggValidationError) as ei:
        next(gen)
    assert "non-finite" in str(ei.value)


def test_bad_device_aborts_before_solving():
    grid = _grid()
    with pytest.raises(EggValidationError):
        next(generate_steps(grid, stages=[Untangle()], device="cuda"))


def _grid_3d():
    from egg.topology.builder import TopologyBuilder

    corners = {
        "a00": (0, 0, 0), "a01": (0, 0, 1), "a10": (0, 1, 0), "a11": (0, 1, 1),
        "b00": (1, 0, 0), "b01": (1, 0, 1), "b10": (1, 1, 0), "b11": (1, 1, 1),
    }
    tb = TopologyBuilder(d=3)
    for name, pos in corners.items():
        tb.add_corner(name, pos)
    tb.add_block(
        "A",
        corners=("a00", "a01", "a10", "a11", "b00", "b01", "b10", "b11"),
        resolutions=(3, 3, 3),
    )
    return tb.build().initialize_grid()


def test_interface_terms_rejected_on_3d_grid():
    grid = _grid_3d()
    gen = generate_steps(
        grid, stages=[Untangle(), JacobiSmoother(interface_ortho={"weight": 1.0})]
    )
    with pytest.raises(EggValidationError) as ei:
        next(gen)
    assert "2D-only" in str(ei.value)
