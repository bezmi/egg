# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""DeclusterSingularities target wrapper: passthrough + size-preserving blend."""

import numpy as np

from egg.smoothing.targets import AnisotropicTarget, DeclusterSingularities
from egg.topology.builder import TopologyBuilder


def _lr_grid(res=(5, 5)):
    b = TopologyBuilder(d=2)
    for name, pos in [
        ("A", (0.0, 0.0)),
        ("D", (0.0, 2.0)),
        ("B", (2.0, 0.0)),
        ("C", (2.0, 2.0)),
        ("E", (4.0, 0.0)),
        ("F", (4.0, 2.0)),
    ]:
        b.add_corner(name, pos, fixed=True)
    b.add_block("L", ("A", "D", "B", "C"), res)
    b.add_block("R", ("B", "C", "E", "F"), res)
    b.connect("L", 0, 1, "R", 0, 0)
    return b.build().initialize_grid()


def test_passthrough_without_singularities():
    """With no singular nodes the wrapper returns the base W untouched."""
    grid = _lr_grid()
    assert not grid.topology.singularities
    base = AnisotropicTarget((0.2, 1.0))  # strongly anisotropic
    dec = DeclusterSingularities(base, grid, radius=3)
    W = dec(0, grid.blocks[0], (0, 0), (0, 0))
    assert np.allclose(W, base(0, grid.blocks[0], (0, 0), (0, 0)))


def test_blend_is_size_preserving_and_positive():
    """The blend formula moves an anisotropic W toward isotropy at fixed size."""
    grid = _lr_grid()
    base = AnisotropicTarget((0.25, 1.0))  # det = 0.25, edges 0.25 and 1.0
    dec = DeclusterSingularities(base, grid, radius=1)
    W0 = base(0, grid.blocks[0], (0, 0), (0, 0))
    # inject a full-strength blend on one cell and evaluate the wrapper there.
    dec._cell_lam[(0, (0, 0))] = 1.0
    W1 = dec(0, grid.blocks[0], (0, 0), (0, 0))
    h = abs(np.linalg.det(W0)) ** 0.5
    assert np.allclose(W1, h * np.eye(2))  # fully isotropic at the same size
    assert np.linalg.det(W1) > 0.0
    assert abs(np.linalg.det(W1) - np.linalg.det(W0)) < 1e-12  # size preserved
