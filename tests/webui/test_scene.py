# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Smoke tests for the web UI's fasthtml-free core (webui/scene.py).

Covers the contract every shipped 2D example must satisfy — exec without
error, register a run via ``egg_webui.run``, harvest into a drawable
scene — plus the render layers (curves, points, control cages, topology)
and SVG well-formedness. No web stack: ``scene.py`` is deliberately
importable without fasthtml, which is what makes this suite possible.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from egg.webui import scene

REPO = Path(__file__).resolve().parents[2]
EXAMPLES_2D = REPO / "examples" / "2D"

# Every shipped 2D example: (path, expects a control cage?)
EXAMPLES = {
    "annulus/annulus.py": True,
    "capsule-fire-II/capsule.py": True,
    "capsule-phoebus/phoebus.py": True,
    "circles/good-topo.py": True,
    "circles/good-topo_side-by-side.py": True,
    "circles/untangle.py": True,
    "circles/untangle_side-by-side.py": True,
    "egg/egg.py": True,
    # the SVG-imported egg has no construction Vector3s on purpose —
    # its construction points live in the Inkscape file
    "egg-svg/egg_svg.py": False,
    "spline_blob/blob.py": True,
}


@pytest.fixture(scope="module")
def harvested():
    """exec + harvest every example once (module-scoped: exec is the
    expensive part, and the assertions are all read-only)."""
    out = {}
    for rel in EXAMPLES:
        src = EXAMPLES_2D / rel
        ns, _stdout, err = scene.exec_script(src.read_text(), str(src))
        out[rel] = (ns, err, scene.harvest(ns))
    return out


@pytest.mark.parametrize("rel", sorted(EXAMPLES))
def test_example_execs_and_registers_a_run(harvested, rel):
    ns, err, _h = harvested[rel]
    assert err is None, f"{rel} raised:\n{err}"
    reg = ns.get("__egg_webui_run__")
    assert reg is not None, f"{rel} registered no run (egg_webui.run missing?)"
    grid, steps = reg
    assert np.isfinite(np.asarray(grid.global_nodes)).all()
    # the steps generator must be lazy — nothing consumed at registration
    assert hasattr(steps, "__next__")


@pytest.mark.parametrize("rel", sorted(EXAMPLES))
def test_example_harvests_a_drawable_scene(harvested, rel):
    _ns, _err, h = harvested[rel]
    s = h.scene
    assert s.curves, f"{rel}: no geometry curves harvested"
    assert h.topo is not None and h.grid is not None
    assert s.topo_blocks and s.topo_corners
    assert s.grid_blocks, f"{rel}: grid layer empty"
    assert not s.warnings, f"{rel}: {s.warnings}"
    if EXAMPLES[rel]:
        assert s.ctrl_cages, f"{rel}: expected construction-point cages"
    else:
        assert not s.ctrl_cages


@pytest.mark.parametrize("rel", sorted(EXAMPLES))
@pytest.mark.parametrize("mode", ["grid", "topo"])
def test_render_is_wellformed_svg_with_layers(harvested, rel, mode):
    _ns, _err, h = harvested[rel]
    svg = scene.render_svg(h.scene, mode=mode)
    root = ET.fromstring(svg)  # well-formed XML
    assert root.tag.endswith("svg")
    classes = {g.get("class") for g in root.iter("{http://www.w3.org/2000/svg}g")}
    assert "layer-curves" in classes
    assert "layer-points" in classes
    assert "layer-ctrl" in classes
    if mode == "grid":
        assert "layer-grid" in classes


def test_blob_spline_cage_is_closed(harvested):
    _ns, _err, h = harvested["spline_blob/blob.py"]
    spline_cages = [
        (name, cage) for name, cage, _k in h.scene.ctrl_cages if cage.shape[0] > 4
    ]
    assert len(spline_cages) == 1
    _name, cage = spline_cages[0]
    # 16 through-points plus the restored closing chord vertex
    assert cage.shape[0] == 17
    np.testing.assert_allclose(cage[0], cage[-1])


def test_phoebus_wall_recurses_into_segments(harvested):
    _ns, _err, h = harvested["capsule-phoebus/phoebus.py"]
    names = [name for name, _c, _k in h.scene.ctrl_cages]
    # arc-line-arc-line Polyline: one cage per segment
    assert sum("wall" in n for n in names) == 4
    # arcs carry p0/centre/p1 (3 vertices), lines p0/p1 (2)
    wall_sizes = sorted(c.shape[0] for n, c, _k in h.scene.ctrl_cages if "wall" in n)
    assert wall_sizes == [2, 2, 3, 3]
    # the inflow Bezier keeps its 4-point control polygon
    inflow = [c for n, c, _k in h.scene.ctrl_cages if "inflow" in n]
    assert len(inflow) == 1 and inflow[0].shape[0] == 4


def test_namespace_vector3s_harvest_as_points():
    code = (
        "from egg.geometry import Vector3, Edge, Line\n"
        "a = Vector3(0, 0, fixed=True)\n"
        "b = Vector3(4, 0)\n"
        "pts = [Vector3(1, 1), Vector3(2, 2)]\n"
        "e = Edge(Line(p0=a, p1=b))\n"
    )
    ns, _out, err = scene.exec_script(code, None)
    assert err is None
    h = scene.harvest(ns)
    kinds = {name: kind for name, kind, _x, _y in h.scene.points}
    assert kinds["a"] == "fixed"
    assert kinds["b"] == "point"
    assert sum(k == "point" for k in kinds.values()) == 3
    # the Line's construction points form a 2-vertex cage (markers, no line)
    assert any(c.shape[0] == 2 for _n, c, _k in h.scene.ctrl_cages)


def test_sparkline_draws_previous_run_faded():
    hist = {"energy": [10.0, 5.0, 3.0], "min det": [0.1, 0.2]}
    prev = {"energy": [12.0, 6.0, 4.0, 2.0]}
    svg = scene.render_sparkline(hist, prev=prev)
    ET.fromstring(svg)
    assert svg.count("spark-prev") == 1  # only the matching series
    assert ">12<" in svg  # y labels span the combined range
    assert "spark-prev" not in scene.render_sparkline(hist)


def test_svg_domain_harvests_labeled_curves():
    svg_file = EXAMPLES_2D / "egg-svg" / "egg.svg"
    code = f"from egg.geometry import svg_import\ndom = svg_import({str(svg_file)!r})\n"
    ns, _out, err = scene.exec_script(code, None)
    assert err is None
    h = scene.harvest(ns)
    names = {c[0] for c in h.scene.curves}
    assert {"dom['egg']", "dom['inflow']", "dom['wall_top']"} <= names


def test_editable_topology_harvests_to_topo():
    code = (
        "from egg.topology import ExplicitTopology\n"
        "topo = ExplicitTopology(connectivity={\n"
        "    'nodes': {'a': {'xy': [0, 0]}, 'b': {'xy': [1, 0]},\n"
        "              'c': {'xy': [1, 1]}, 'd': {'xy': [0, 1]}},\n"
        "    'edges': [{'a': 'a', 'b': 'b'}, {'a': 'b', 'b': 'c'},\n"
        "              {'a': 'c', 'b': 'd'}, {'a': 'd', 'b': 'a'}],\n"
        "    'res': 3})\n"
    )
    ns, _out, err = scene.exec_script(code, None)
    assert err is None
    h = scene.harvest(ns, init_grid=False)
    assert h.editable is not None
    assert h.topo is not None
    assert h.diagnostics == []
    assert len(h.scene.topo_blocks) == 1


def test_editable_topology_invalid_surfaces_diagnostic():
    code = (
        "from egg.topology import ExplicitTopology\n"
        "topo = ExplicitTopology(connectivity={\n"
        "    'nodes': {'a': {'xy': [0, 0]}, 'b': {'xy': [1, 0]}, 'c': {'xy': [0.5, 1]}},\n"
        "    'edges': [{'a': 'a', 'b': 'b'}, {'a': 'b', 'b': 'c'}, {'a': 'c', 'b': 'a'}]})\n"
    )
    ns, _out, err = scene.exec_script(code, None)
    assert err is None
    h = scene.harvest(ns, init_grid=False)
    assert h.topo is None
    assert h.editable is not None
    assert any(w.startswith("topology:") for w in h.scene.warnings)


def test_editable_grid_mismatch_warns():
    """A grid not built from the in-scope editable topology is a silent footgun
    — the harvest flags it."""
    code = (
        "from egg.topology import ExplicitTopology, TopologyBuilder\n"
        "b = TopologyBuilder(d=2)\n"
        "for nm, xy in [('s0',(0,0)),('s1',(1,0)),('s2',(0,1)),('s3',(1,1))]:\n"
        "    b.add_corner(nm, xy)\n"
        "b.add_block('b0', sw='s0', se='s1', nw='s2', ne='s3', res=(4, 4))\n"
        "grid = b.build().initialize_grid()\n"  # base grid, smoothed as-is
        "et = ExplicitTopology(base=b, connectivity={\n"
        "    'nodes': {'M': {'split': ['s0','s1'], 't': 0.5},\n"
        "              'N': {'split': ['s2','s3'], 't': 0.5}},\n"
        "    'edges': [{'a': 'M', 'b': 'N'}]})\n"  # bifurcates -> different topo
    )
    ns, _out, err = scene.exec_script(code, None)
    assert err is None
    h = scene.harvest(ns, init_grid=True)
    assert any("not built from the edited topology" in w for w in h.scene.warnings)


def test_edit_mode_renders_topology_layers():
    code = (
        "from egg.topology import ExplicitTopology\n"
        "topo = ExplicitTopology(connectivity={\n"
        "    'nodes': {'a': {'xy': [0, 0]}, 'b': {'xy': [1, 0]},\n"
        "              'c': {'xy': [1, 1]}, 'd': {'xy': [0, 1]}},\n"
        "    'edges': [{'a': 'a', 'b': 'b'}, {'a': 'b', 'b': 'c'},\n"
        "              {'a': 'c', 'b': 'd'}, {'a': 'd', 'b': 'a'}],\n"
        "    'res': 3})\n"
    )
    r = scene.build_scene(code, mode="edit")
    assert r.error is None
    root = ET.fromstring(r.svg)
    assert root.get("class") == "mode-edit"
    classes = {g.get("class") for g in root.iter("{http://www.w3.org/2000/svg}g")}
    assert {"layer-blocks", "layer-corners"} <= classes


def test_edit_mode_emits_structured_blocking_data():
    code = (
        "from egg.topology import ExplicitTopology, editable\n"
        "from egg.geometry import Circle\n"
        "c = Circle(center=(9, 9), radius=1.0)\n"
        "topo = ExplicitTopology(geometry={'ring': c}, connectivity=editable({\n"
        "    'nodes': {'a': {'xy': [0, 0]}, 'b': {'xy': [1, 0]},\n"
        "              'c': {'xy': [1, 1]}, 'd': {'xy': [0, 1]}},\n"
        "    'edges': [{'a': 'a', 'b': 'b'}, {'a': 'b', 'b': 'c'},\n"
        "              {'a': 'c', 'b': 'd'}, {'a': 'd', 'b': 'a'}], 'res': 3}))\n"
    )
    r = scene.build_scene(code, mode="edit")
    assert r.error is None
    ed = r.edit_data
    assert ed is not None
    assert ed["editable"] is True
    assert set(ed["blocking"]["nodes"]) == {"a", "b", "c", "d"}
    assert [g["label"] for g in ed["geometry"]] == ["ring"]
    assert len(ed["geometry"][0]["points"]) > 1  # sampled polyline for hit-testing
    assert ed["diagnostics"] == []
    # a bare (unwrapped) connectivity is not editable
    plain = code.replace("editable(", "(")  # editable({...}) -> ({...})
    assert scene.build_scene(plain, mode="edit").edit_data["editable"] is False


def test_validate_blocking_reports_green_and_red():
    code = (
        "from egg.topology import ExplicitTopology, editable\n"
        "topo = ExplicitTopology(connectivity=editable({'nodes': {}, 'edges': []}))\n"
    )
    good = {
        "nodes": {
            "a": {"xy": [0, 0]},
            "b": {"xy": [1, 0]},
            "c": {"xy": [1, 1]},
            "d": {"xy": [0, 1]},
        },
        "edges": [
            {"a": "a", "b": "b"},
            {"a": "b", "b": "c"},
            {"a": "c", "b": "d"},
            {"a": "d", "b": "a"},
        ],
        "res": 3,
    }
    assert scene.validate_blocking(code, good) == []  # green

    bad = {
        "nodes": {"a": {"xy": [0, 0]}, "b": {"xy": [1, 0]}, "c": {"xy": [0.5, 1]}},
        "edges": [{"a": "a", "b": "b"}, {"a": "b", "b": "c"}, {"a": "c", "b": "a"}],
    }
    red = scene.validate_blocking(code, bad)
    assert red and any(d["kind"] in ("non_quad_face", "no_blocks") for d in red)


def test_set_editable_blocking_round_trips_and_is_idempotent():
    code = (
        "from egg.topology import ExplicitTopology, editable\n"
        "topo = ExplicitTopology(connectivity=editable({'nodes': {}, 'edges': []}))\n"
    )
    blocking = {
        "nodes": {"a": {"xy": [0, 0]}, "b": {"xy": [1, 0]}},
        "edges": [{"a": "a", "b": "b"}],
        "res": 5,
    }
    new = scene.set_editable_blocking(code, blocking)
    # still editable, and re-execs to the committed blocking
    assert scene.explicit_topology_source(new)["editable"] is True
    ns, _o, err = scene.exec_script(new, None)
    assert err is None
    h = scene.harvest(ns, init_grid=False)
    assert set(h.editable.connectivity["nodes"]) == {"a", "b"}
    assert h.editable.connectivity["res"] == 5
    # committing the same blocking again changes nothing (no source churn)
    assert scene.set_editable_blocking(new, blocking) == new


def test_set_editable_blocking_refuses_unwrapped():
    code = (
        "from egg.topology import ExplicitTopology\n"
        "topo = ExplicitTopology(connectivity={'nodes': {}, 'edges': []})\n"
    )
    with pytest.raises(ValueError):
        scene.set_editable_blocking(code, {"nodes": {}, "edges": []})


def test_editable_topology_wins_over_a_coexisting_grid():
    """A script that also builds a run grid still shows its ExplicitTopology."""
    code = (
        "from egg.topology import ExplicitTopology\n"
        "from egg.topology.builder import TopologyBuilder\n"
        "gb = TopologyBuilder(d=2)\n"
        "for nm, xy in [('a', (0, 0)), ('b', (1, 0)), ('c', (1, 1)), ('d', (0, 1))]:\n"
        "    gb.add_corner(nm, xy, fixed=False)\n"
        "gb.add_block('q', sw='a', se='b', nw='d', ne='c', res=(3, 3))\n"
        "grid = gb.build().initialize_grid()\n"  # a run grid in the namespace
        "et = ExplicitTopology(connectivity={\n"
        "    'nodes': {'p0': {'xy': [5, 5]}, 'p1': {'xy': [6, 5]},\n"
        "              'p2': {'xy': [6, 6]}, 'p3': {'xy': [5, 6]}},\n"
        "    'edges': [{'a': 'p0', 'b': 'p1'}, {'a': 'p1', 'b': 'p2'},\n"
        "              {'a': 'p2', 'b': 'p3'}, {'a': 'p3', 'b': 'p0'}], 'res': 2})\n"
    )
    ns, _out, err = scene.exec_script(code, None)
    assert err is None
    h = scene.harvest(ns, init_grid=False)
    assert h.editable is not None  # captured despite the grid
    assert h.grid is not None  # the run grid is still there for the grid view
    assert {"p0", "p2"} <= set(h.topo.corners)  # the editable flatten is the topo


def test_explicit_topology_source_detects_the_wrap():
    wrapped = (
        "from egg.topology import ExplicitTopology, editable\n"
        "t = ExplicitTopology(connectivity=editable({'nodes': {}}))\n"
    )
    info = scene.explicit_topology_source(wrapped)
    assert info["editable"] is True
    assert info["span"] is not None  # the blocking literal inside editable(...)

    plain = (
        "from egg.topology import ExplicitTopology\n"
        "t = ExplicitTopology(connectivity={'nodes': {}})\n"
    )
    assert scene.explicit_topology_source(plain)["editable"] is False

    assert scene.explicit_topology_source("x = 1") is None


def test_script_error_is_reported_not_raised():
    ns, _out, err = scene.exec_script("import egg\n1/0\n", None)
    assert err is not None and "ZeroDivisionError" in err
    h = scene.harvest(ns)  # a broken script still harvests what it defined
    assert h.scene.curves == []


def test_empty_scene_renders_placeholder():
    svg = scene.render_svg(scene.Scene())
    assert "nothing to draw yet" in svg
    ET.fromstring(svg)


def _grid_line_count(svg: str) -> int:
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    return sum(
        len(g.findall(f"{ns}polyline"))
        for g in root.iter(f"{ns}g")
        if (g.get("class") or "").startswith("grid-lines")
    )


def test_grid_quality_on_reference_cells():
    # unit squares: 90° corners, zero skew, AR 1 — exactly
    xs, ys = np.meshgrid(np.arange(5.0), np.arange(4.0), indexing="ij")
    q = scene.grid_quality([np.stack([xs, ys], axis=-1)])
    assert q["orthogonality"] == (90.0, 0.0, 90.0)
    assert q["skewness"] == (0.0, 0.0, 0.0)
    assert q["aspect ratio"] == (1.0, 0.0, 1.0)
    # stretch x by 3: AR 3, still orthogonal
    q = scene.grid_quality([np.stack([3.0 * xs, ys], axis=-1)])
    assert q["aspect ratio"][0] == pytest.approx(3.0)
    assert q["orthogonality"][2] == pytest.approx(90.0)
    # shear: 45° worst angle, skew 0.5
    sheared = np.stack([xs + ys, ys], axis=-1)
    q = scene.grid_quality([sheared])
    assert q["orthogonality"][2] == pytest.approx(45.0)
    assert q["skewness"][2] == pytest.approx(0.5)
    assert scene.grid_quality([]) is None


def test_grid_view_carries_block_and_total_cell_counts(harvested):
    _ns, _err, h = harvested["spline_blob/blob.py"]
    svg = scene.render_svg(h.scene, mode="grid")
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    labels = [
        g.get("data-label")
        for g in root.iter(f"{ns}g")
        if g.get("class") == "grid-block"
    ]
    assert len(labels) == len(h.scene.grid_blocks)
    assert any(lab.startswith("o_s: ") for lab in labels)
    per_block = [(b.shape[0] - 1) * (b.shape[1] - 1) for b in h.scene.grid_blocks]
    assert f"= {per_block[0]}" in labels[0]
    assert f"grid: {sum(per_block)} cells" in root.get("data-cells")


def test_dense_grid_preview_is_decimated():
    xs, ys = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 40), indexing="ij")
    block = np.stack([xs, ys], axis=-1)
    s = scene.Scene(grid_blocks=[block])
    assert _grid_line_count(scene.render_svg(s)) == scene.MAX_GRID_LINES + 40
    # small blocks draw every line
    s = scene.Scene(grid_blocks=[block[:20, :10]])
    assert _grid_line_count(scene.render_svg(s)) == 30
    # boundary lines survive decimation
    idx = scene._line_indices(400)
    assert idx[0] == 0 and idx[-1] == 399


# --- guard parameter panel: parse + span-exact rewrite ---


def test_guard_params_of_the_egg_example():
    code = (EXAMPLES_2D / "egg/egg.py").read_text()
    by_name = {p.name: p for p in scene.guard_params(code)}
    assert by_name["bl_first_height"].text == "5.0e-3"
    assert by_name["bl_first_height"].kind == "float"
    assert by_name["tmop_sweeps"].kind == "int"
    assert by_name["smoother"].kind == "str"
    assert by_name["device"].text == '"cpu"'


def test_set_guard_param_rewrites_only_that_literal():
    code = (EXAMPLES_2D / "egg/egg.py").read_text()
    new = scene.set_guard_param(code, "tmop_sweeps", "120")
    changed = [(a, b) for a, b in zip(code.splitlines(), new.splitlines()) if a != b]
    assert changed == [("        tmop_sweeps=5000,", "        tmop_sweeps=120,")]
    # user spellings survive; strings get quoted
    new = scene.set_guard_param(code, "bl_first_height", "2.5e-3")
    assert "bl_first_height=2.5e-3," in new
    new = scene.set_guard_param(code, "smoother", "fas")
    assert "smoother='fas'," in new


def test_set_guard_param_synthetic_kinds_and_errors():
    code = (
        'if __name__ == "__egg_webui__":\n'
        "    a = egg_webui.params(n=-3, flag=True, name='x')\n"
    )
    kinds = {p.name: p.kind for p in scene.guard_params(code)}
    assert kinds == {"n": "int", "flag": "bool", "name": "str"}
    assert "flag=False" in scene.set_guard_param(code, "flag", "false")
    assert "n=-7" in scene.set_guard_param(code, "n", "-7")
    with pytest.raises(ValueError):
        scene.set_guard_param(code, "n", "not-a-number")
    with pytest.raises(ValueError):
        scene.set_guard_param(code, "missing", "1")
    # scripts without a guard dict have no panel
    assert scene.guard_params("x = 1\n") == []
    assert scene.guard_params("if __name__ == '__main__':\n    a = dict(n=1)\n") == []


def test_set_guard_param_roundtrips_through_exec():
    code = (EXAMPLES_2D / "spline_blob/blob.py").read_text()
    new = scene.set_guard_param(code, "tmop_sweeps", "8")
    ns, _out, err = scene.exec_script(new, str(EXAMPLES_2D / "spline_blob/blob.py"))
    assert err is None and ns.get("__egg_webui_run__") is not None


def test_editable_marks_are_params_with_context_names():
    code = (
        "import egg.webui as egg_webui\n"
        "N = egg_webui.editable(3, label='rings')\n"
        "W = editable(0.5)\n"
        'if __name__ == "__egg_webui__":\n'
        "    a = egg_webui.params(\n"
        "        n=1,\n"
        "        metric=egg_webui.editable('shape_size',\n"
        "                                  choices=['shape', 'shape_size']),\n"
        "        dipole=egg_webui.editable(True),\n"
        "    )\n"
        "    f(kw=editable('x'))\n"
    )
    ps = {p.name: p for p in scene.guard_params(code)}
    assert ps["rings"].kind == "int" and ps["rings"].text == "3"
    assert ps["W"].kind == "float"  # assignment-target name
    assert ps["metric"].choices == ("shape", "shape_size")  # dict-key name
    assert ps["dipole"].kind == "bool" and ps["dipole"].choices is None
    assert ps["kw"].kind == "str"  # keyword-argument name
    assert ps["n"].kind == "int"  # params() entries still work
    # source order: module-level marks precede the guard entries
    names = [p.name for p in scene.guard_params(code)]
    assert names.index("rings") < names.index("n")


def test_editable_rewrite_preserves_the_call_and_checks_choices():
    code = (
        "import egg.webui as egg_webui\n"
        "m = egg_webui.editable('shape_size', choices=['shape', 'shape_size'])\n"
    )
    new = scene.set_guard_param(code, "m", "shape")
    assert "egg_webui.editable('shape', choices=['shape', 'shape_size'])" in new
    with pytest.raises(ValueError, match="one of"):
        scene.set_guard_param(code, "m", "sizeshape")


def test_editable_duplicate_names_get_suffixes():
    code = "x = editable(1)\ny = dict(x=editable(2))\n"
    names = [p.name for p in scene.guard_params(code)]
    assert names == ["x", "x#2"]
    assert "editable(9)" in scene.set_guard_param(code, "x#2", "9")


def test_editable_runtime_is_identity():
    import egg.webui as egg_webui

    assert egg_webui.editable(3, choices=[1, 3], label="n") == 3
    assert egg_webui.editable("a") == "a"


def test_capsule_guard_has_metric_dropdown_and_dipole_toggle():
    code = (EXAMPLES_2D / "capsule-fire-II/capsule.py").read_text()
    ps = {p.name: p for p in scene.guard_params(code)}
    assert ps["metric"].choices == ("shape", "shape_size")
    assert ps["corner dipole"].kind == "bool"
    new = scene.set_guard_param(code, "corner dipole", "false")
    assert "dipole=egg_webui.editable(False, label=" in new
