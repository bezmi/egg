# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""svg_topology: block layout inferred from a straight-line SVG wireframe."""

from pathlib import Path

import numpy as np
import pytest

from egg.geometry import svg_import, svg_topology

EGG_SVG = (
    Path(__file__).resolve().parents[2] / "examples" / "2D" / "egg-svg" / "egg.svg"
)


def _svg(body: str, viewbox: str = "0 0 4 4") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'viewBox="{viewbox}">{body}</svg>'
    )


def _seg(a, b, label=""):
    lab = f'inkscape:label="{label}" ' if label else ""
    return f'<path {lab}d="M {a[0]},{a[1]} L {b[0]},{b[1]}" />'


def test_two_quads_from_wireframe():
    """A 2x1 wireframe yields two conforming quad blocks sharing one edge."""
    # y-up model coords equal svg coords here (viewBox height 2, flip handled).
    wire = "".join(
        _seg(*e)
        for e in [
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
            ((0, 2), (1, 2)),
            ((1, 2), (2, 2)),
            ((0, 0), (0, 2)),
            ((1, 0), (1, 2)),  # the shared interior edge
            ((2, 0), (2, 2)),
        ]
    )
    body = f'<g inkscape:label="topology">{wire}</g>'
    dom = svg_import(_svg(body, viewbox="0 0 2 2"))
    b, ents = svg_topology(dom, res=4)
    assert len(b._block_specs) == 2
    assert len(b._corners) == 6
    assert ents == {}  # no geometry layer -> no associations
    topo = b.build()  # conforming shared edge, builds cleanly
    assert len(topo.singularities) == 0


def test_egg_svg_reconstructs_o_grid():
    dom = svg_import(EGG_SVG)
    assert sum(1 for it in dom if it.group == "topology") == 32

    b, ents = svg_topology(dom, res=10)
    assert set(ents) == {"inflow", "outflow", "wall_bottom", "wall_top", "egg"}
    assert len(b._block_specs) == 12
    assert len(b._corners) == 20

    # Exactly the four domain corners are pinned (each lies on two walls).
    fixed = [n for n, c in b._corners.items() if getattr(c, "fixed", False)]
    assert len(fixed) == 4

    # Every wall/egg boundary is tagged; the egg carries the four O-ring faces.
    # The markers auto-derive from the associated entities' tags in build(), so
    # inspect the built topology rather than the pre-build builder dict.
    tags = {k: len(v) for k, v in b.build().boundary_tags.items()}
    assert tags["egg"] == 4
    assert set(tags) == {"egg", "inflow", "outflow", "wall_bottom", "wall_top"}


def test_egg_svg_singularities_and_grid():
    dom = svg_import(EGG_SVG)
    b, _ = svg_topology(dom, res=10)
    topo = b.build()

    # The O-ring/mid-square junctions are the four 5-valent singularities;
    # the 3-valent inner-ring nodes sit on the egg boundary, so they are
    # regular, not flagged.
    assert [s.valence for s in topo.singularities] == [5, 5, 5, 5]

    grid = topo.initialize_grid()
    dets = []
    for block in grid.blocks:
        nd = np.asarray(block.nodes, dtype=float)
        ni, nj = block.logical_shape[0], block.logical_shape[1]
        for i in range(ni - 1):
            for j in range(nj - 1):
                a = nd[i + 1, j] - nd[i, j]
                c = nd[i, j + 1] - nd[i, j]
                dets.append(a[0] * c[1] - a[1] * c[0])
    assert min(dets) > 0.0  # TFI init is unfolded everywhere


def test_missing_group_raises():
    dom = svg_import(EGG_SVG)
    with pytest.raises(ValueError, match="no wireframe found"):
        svg_topology(dom, group="does-not-exist")
