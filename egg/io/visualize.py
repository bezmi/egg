"""PyVista (2D+3D grids) + matplotlib (scalar diagnostics)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv

    from egg.geometry.base import GeometryEntity
    from egg.topology.block_topology import BlockTopology

__all__ = [
    "plot_geometry_entity",
    "plot_projection",
    "plot_topology",
    "plot_grid",
    "plot_quality",
    "LiveGridView",
]


def _get_plotter(show: bool) -> "pv.Plotter | None":
    """Return a PyVista Plotter if show=True, otherwise None."""
    if not show:
        return None
    import pyvista as pv

    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter()
    return plotter


def plot_geometry_entity(
    entity: "GeometryEntity",
    n_curve: int = 200,
    n_normals: int = 8,
    *,
    show: bool = True,
) -> Optional["pv.Plotter"]:
    """Render a geometry entity with normal and tangent vectors.

    Parameters
    ----------
    entity : GeometryEntity
        The entity to render (Circle, LineSegment, etc.).
    n_curve : int
        Number of sample points along the curve.
    n_normals : int
        Number of normal/tangent arrow pairs to display.
    show : bool
        If True, open an interactive PyVista window.
    """
    plotter = _get_plotter(show)
    if plotter is None:
        return None

    import pyvista as pv

    # Sample curve points
    if isinstance(entity, type) or entity.dim == 1:
        t_vals = np.linspace(0, 1, n_curve)
        if hasattr(entity, "start") and hasattr(entity, "end"):
            pts = np.array([entity.project(entity.start + t * (entity.end - entity.start)) for t in t_vals])
        else:
            # Circle: sample roughly around it
            angles = np.linspace(0, 2 * np.pi, n_curve)
            pts = np.column_stack([
                entity.center[0] + entity.radius * np.cos(angles),
                entity.center[1] + entity.radius * np.sin(angles),
            ])
    else:
        # Generic: just project some reference points (not exhaustive)
        pts = np.array([entity.project(np.array([0.0, 0.0]))])

    # Pad to 3D for PyVista
    if pts.shape[1] == 2:
        pts_3d = np.column_stack([pts, np.zeros(len(pts))])
    else:
        pts_3d = pts

    # Render as a polyline (not triangulated surface)
    curve = pv.lines_from_points(pts_3d)
    plotter.add_mesh(curve, color="black", line_width=2)

    # Normals and tangents at sample points
    sample_idxs = np.linspace(0, n_curve - 1, n_normals, dtype=int)
    for idx in sample_idxs:
        q = pts[idx]
        q_3d = pts_3d[idx]
        scale = 0.3 * np.linalg.norm(entity.project(np.array([5.0, 5.0])) - q) * 0.2 + 0.15

        try:
            n = entity.normal(q)
            n_3d = np.append(n, 0.0) if len(n) == 2 else n
            arrow_n = pv.Arrow(start=q_3d, direction=n_3d * scale, scale=0.08)
            plotter.add_mesh(arrow_n, color="red", label="Normal")
        except (NotImplementedError, Exception):
            pass

        try:
            t = entity.tangent_space(q)[:, 0]
            t_3d = np.append(t, 0.0) if len(t) == 2 else t
            arrow_t = pv.Arrow(start=q_3d, direction=t_3d * scale, scale=0.08)
            plotter.add_mesh(arrow_t, color="blue", label="Tangent")
        except (NotImplementedError, Exception):
            pass

    plotter.view_xy()
    plotter.show_axes()
    if show:
        plotter.show()
    return plotter


def plot_projection(
    entity: "GeometryEntity",
    points: np.ndarray,
    *,
    show: bool = True,
) -> Optional["pv.Plotter"]:
    """Show source points and their projections onto the entity.

    Parameters
    ----------
    entity : GeometryEntity
    points : ndarray, shape (N, d)
        Source points to project.
    show : bool
    """
    plotter = _get_plotter(show)
    if plotter is None:
        return None

    import pyvista as pv

    projected = np.array([entity.project(p) for p in points])

    # Pad to 3D
    if points.shape[1] == 2:
        points_3d = np.column_stack([points, np.zeros(len(points))])
        proj_3d = np.column_stack([projected, np.zeros(len(projected))])
    else:
        points_3d = points
        proj_3d = projected

    # Source points
    src_pd = pv.PolyData(points_3d)
    plotter.add_mesh(src_pd, color="orange", point_size=8, render_points_as_spheres=True)

    # Projected points
    dst_pd = pv.PolyData(proj_3d)
    plotter.add_mesh(dst_pd, color="green", point_size=10, render_points_as_spheres=True)

    # Connection lines
    for i in range(len(points)):
        line = pv.Line(points_3d[i], proj_3d[i])
        plotter.add_mesh(line, color="gray", line_width=1)

    # Entity curve
    if plotter is not None and entity.dim == 1:
        _add_entity_curve(plotter, entity)

    plotter.view_xy()
    plotter.show_axes()
    if show:
        plotter.show()
    return plotter


def _add_entity_curve(plotter: "pv.Plotter", entity: "GeometryEntity") -> None:
    """Add the entity curve to the plotter (internal helper)."""
    import pyvista as pv

    if hasattr(entity, "center") and hasattr(entity, "radius"):
        angles = np.linspace(0, 2 * np.pi, 200)
        pts = np.column_stack([
            entity.center[0] + entity.radius * np.cos(angles),
            entity.center[1] + entity.radius * np.sin(angles),
            np.zeros(200),
        ])
    elif hasattr(entity, "start") and hasattr(entity, "end"):
        t_vals = np.linspace(0, 1, 200)
        pts = np.column_stack([
            entity.start[0] + t_vals * (entity.end[0] - entity.start[0]),
            entity.start[1] + t_vals * (entity.end[1] - entity.start[1]),
            np.zeros(200),
        ])
    else:
        return
    curve = pv.PolyData(pts)
    plotter.add_mesh(curve, color="black", line_width=2)


def plot_topology(
    topology: "BlockTopology",
    highlight_singularities: bool = True,
    *,
    show: bool = True,
) -> Optional["pv.Plotter"]:
    """Render declared topology: corners, block boundaries, shared interfaces.

    Parameters
    ----------
    topology : BlockTopology
    highlight_singularities : bool
        If True, mark singularity nodes with red markers.
    show : bool
    """
    plotter = _get_plotter(show)
    if plotter is None:
        return None

    import pyvista as pv

    for name, spec in topology.block_specs.items():
        corner_positions = []
        for cname in spec.corner_names:
            corner_positions.append(topology.corners[cname].position)

        # Pad to 3D
        cpts = np.array(corner_positions)
        if cpts.shape[1] == 2:
            cpts = np.column_stack([cpts, np.zeros(len(cpts))])

        # Draw corner-to-corner edges in product order to show block outline
        d = topology.d
        from itertools import product

        corners_order = list(product((0, 1), repeat=d))
        for i, offset_a in enumerate(corners_order):
            for offset_b in corners_order[i + 1:]:
                diff = sum(abs(a - b) for a, b in zip(offset_a, offset_b))
                if diff == 1:
                    name_a = spec.corner_names[corners_order.index(offset_a)]
                    name_b = spec.corner_names[corners_order.index(offset_b)]
                    pos_a = topology.corners[name_a].position
                    pos_b = topology.corners[name_b].position
                    pa = np.append(pos_a, 0.0) if len(pos_a) == 2 else pos_a
                    pb = np.append(pos_b, 0.0) if len(pos_b) == 2 else pos_b
                    line = pv.Line(pa, pb)
                    plotter.add_mesh(line, color="black", line_width=2)

    # Corners
    for name, corner in topology.corners.items():
        pos = np.append(corner.position, 0.0) if len(corner.position) == 2 else corner.position
        color = "blue" if corner.fixed else "green"
        sphere = pv.Sphere(radius=0.08, center=pos)
        plotter.add_mesh(sphere, color=color)
        plotter.add_point_labels(
            np.array([pos]), [name], font_size=12, point_size=0,
            shape_opacity=0.5, always_visible=True,
        )

    # Singularities
    if highlight_singularities:
        for s in topology.singularities:
            spec = topology.block_specs[s.block_name]
            shape = spec.logical_shape
            # Map logical_idx to the corresponding corner position
            corner_indices = spec._corner_logical_indices(topology.d)
            pos = None
            for ci_logical, cname in zip(corner_indices, spec.corner_names):
                actual = tuple(
                    0 if c == 0 else shape[dim] - 1
                    for dim, c in enumerate(ci_logical)
                )
                if actual == s.logical_idx:
                    corner = topology.corners[cname]
                    pos = corner.position
                    pos = np.append(pos, 0.0) if len(pos) == 2 else pos
                    break
            if pos is None:
                pos = np.zeros(3)
            box = pv.Cube(center=pos, x_length=0.16, y_length=0.16, z_length=0.16)
            plotter.add_mesh(box, color="red", style="wireframe", line_width=2)
            plotter.add_point_labels(
                np.array([pos + np.array([0.15, 0.1, 0.0])]),
                [f"v={s.valence}"],
                font_size=10, point_size=0, always_visible=True,
            )

    plotter.view_xy()
    plotter.show_axes()
    if show:
        plotter.show()
    return plotter


def plot_grid(grid, *, show: bool = True) -> Optional["pv.Plotter"]:
    """Render a structured multiblock grid as wireframe.

    Each block gets a distinct color. Works for 2D and 3D grids.
    """
    plotter = _get_plotter(show)
    if plotter is None:
        return None

    import pyvista as pv

    # Greedy-colour the block adjacency graph so adjacent blocks never
    # share a wireframe colour.
    block_names = list(grid.topology.block_specs.keys())
    block_index: dict[str, int] = {name: i for i, name in enumerate(block_names)}
    adj: dict[int, set[int]] = {bi: set() for bi in range(len(block_names))}
    for conn in grid.topology.interface_connections:
        a = block_index.get(conn.face_a.block_name)
        b = block_index.get(conn.face_b.block_name)
        if a is not None and b is not None and a != b:
            adj[a].add(b)
            adj[b].add(a)
    colour_idx: dict[int, int] = {}
    for bi in range(len(block_names)):
        forbidden = {colour_idx.get(nb, -1) for nb in adj[bi]}
        c = 0
        while c in forbidden:
            c += 1
        colour_idx[bi] = c

    block_colors = ["cyan", "magenta", "yellow", "lime", "orange", "white"]
    for bi, block in enumerate(grid.blocks):
        color = block_colors[colour_idx[bi] % len(block_colors)]
        nodes = block.nodes

        if nodes.ndim - 1 == 2:
            # 2D grid: pad z=0
            x = np.asarray(nodes[..., 0], dtype=float)
            y = np.asarray(nodes[..., 1], dtype=float)
            z = np.zeros_like(x)
            sg = pv.StructuredGrid(x, y, z)
        elif nodes.ndim - 1 == 3:
            x = np.asarray(nodes[..., 0], dtype=float)
            y = np.asarray(nodes[..., 1], dtype=float)
            z = np.asarray(nodes[..., 2], dtype=float)
            sg = pv.StructuredGrid(x, y, z)
        else:
            continue

        plotter.add_mesh(sg, color=color, style="wireframe", line_width=2)

    plotter.view_xy()
    plotter.show_axes()
    if show:
        plotter.show()
    return plotter


def plot_quality_field(
    grid,
    target_fn,
    metric: str = "shape_2d",
    field: str = "mu",
    *,
    show: bool = True,
) -> Optional["pv.Plotter"]:
    """Render grid cells colored by quality metric.

    Parameters
    ----------
    grid : MultiBlockGrid
    target_fn : callable
    metric : str
    field : str
        "mu" for metric value, "det" for Jacobian determinant.
    show : bool
    """
    plotter = _get_plotter(show)
    if plotter is None:
        return None

    import pyvista as pv

    from egg.smoothing.jacobian import compute_jacobian
    from egg.smoothing.metrics import metric_value

    scalars_all = []
    mesh_blocks = []

    for block in grid.blocks:
        nodes = block.nodes
        d = block.d
        shape = block.logical_shape

        if d != 2:
            continue

        values = np.zeros((shape[0] - 1, shape[1] - 1))
        for i in range(shape[0] - 1):
            for j in range(shape[1] - 1):
                A = compute_jacobian(nodes, (i, j))
                if field == "det":
                    values[i, j] = float(np.linalg.det(A))
                else:
                    W = target_fn((i, j), (0,) * d)
                    T = A @ np.linalg.inv(W)
                    values[i, j] = metric_value(T, metric)

        # Build quadrilateral mesh for this block
        pts = nodes.reshape(-1, d)
        if d == 2:
            pts_3d = np.column_stack([pts, np.zeros(len(pts))])

            ni, nj = shape[0], shape[1]
            cells = []
            for i in range(ni - 1):
                for j in range(nj - 1):
                    a = i * nj + j
                    b = i * nj + j + 1
                    c = (i + 1) * nj + j + 1
                    d_cell = (i + 1) * nj + j
                    cells.append([4, a, b, c, d_cell])

            if cells:
                faces = np.hstack(cells)
                surf = pv.PolyData(pts_3d, faces)
                surf.cell_data["quality"] = values.ravel()
                scalars_all.extend(values.ravel())
                plotter.add_mesh(
                    surf, scalars="quality", cmap="viridis",
                    show_scalar_bar=(len(mesh_blocks) == 0), clim=None,
                )
                mesh_blocks.append(surf)

    plotter.view_xy()
    plotter.show_axes()
    if show:
        plotter.show()
    return plotter


def plot_energy_history(energy_history: list[float], *, show: bool = True) -> None:
    """Plot energy vs sweep number using matplotlib.

    Parameters
    ----------
    energy_history : list[float]
    show : bool
    """
    if not show:
        return

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(energy_history) + 1), energy_history, "b-o", markersize=4)
    plt.xlabel("Sweep")
    plt.ylabel("Energy F(x)")
    plt.title("TMOP Local Relaxation Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_quality(grid, *, show: bool = True):
    """Plot quality metrics (deferred to M3)."""
    raise NotImplementedError("Quality plotting deferred to M3 (metrics)")


def plot_grid_live(
    grid,
    boundary_tags: dict[int, str] | None = None,
    *,
    show: bool = True,
    sweep_title: str = "",
    geometry_entities: list | None = None,
) -> Optional["pv.Plotter"]:
    """Render filled cells with wireframe overlay and colored boundary nodes.

    Parameters
    ----------
    grid : MultiBlockGrid
    boundary_tags : dict[int, str] or None
        Maps global DOF index to "circle", "outer", or "interior".
    show : bool
    sweep_title : str
        Title shown on the plotter.
    geometry_entities : list of GeometryEntity or None
        Entities drawn as black lines (circle, rectangle edges, etc.).
    """
    plotter = _get_plotter(show)
    if plotter is None:
        return None

    _add_grid_meshes(plotter, grid, boundary_tags, geometry_entities)
    if sweep_title:
        plotter.add_title(sweep_title, font_size=14)

    plotter.view_xy()
    plotter.show_axes()
    if show:
        plotter.show()
    return plotter


def _add_grid_meshes(
    plotter: "pv.Plotter",
    grid,
    boundary_tags: dict[int, str] | None = None,
    geometry_entities: list | None = None,
) -> None:
    """Add all grid meshes to an existing plotter (no show)."""
    import pyvista as pv

    block_fill_colors = [
        "lightblue", "lightcoral", "lightgreen", "lightyellow",
        "lightsalmon", "plum", "palegreen", "wheat",
    ]
    block_edge_colors = ["cyan", "magenta", "yellow", "lime", "orange", "white"]

    if boundary_tags is None:
        boundary_tags = {}

    if geometry_entities:
        for entity in geometry_entities:
            _add_entity_curve(plotter, entity)

    for bi, block in enumerate(grid.blocks):
        nodes = block.nodes
        d = block.d
        shape = block.logical_shape

        if d != 2:
            continue

        pts = nodes.reshape(-1, d)
        pts_3d = np.column_stack([pts, np.zeros(len(pts))])

        ni, nj = shape[0], shape[1]
        cells = []
        for i in range(ni - 1):
            for j in range(nj - 1):
                a = i * nj + j
                b = i * nj + j + 1
                c = (i + 1) * nj + j + 1
                d_c = (i + 1) * nj + j
                cells.append([4, a, b, c, d_c])

        if cells:
            faces_array = np.hstack(cells)
            surf = pv.PolyData(pts_3d, faces_array)
            plotter.add_mesh(
                surf, color=block_fill_colors[bi % len(block_fill_colors)],
                opacity=0.35, show_edges=True, edge_color="gray",
                line_width=0.5,
            )

        x = np.asarray(nodes[..., 0], dtype=float)
        y = np.asarray(nodes[..., 1], dtype=float)
        z = np.zeros_like(x)
        sg = pv.StructuredGrid(x, y, z)
        plotter.add_mesh(
            sg, color=block_edge_colors[bi % len(block_edge_colors)],
            style="wireframe", line_width=1.5,
        )

    for bi, block in enumerate(grid.blocks):
        dof_map_local = grid.block_dof_maps[bi]
        nodes = block.nodes
        shape = block.logical_shape

        for i in range(shape[0]):
            for j in range(shape[1]):
                gidx = int(dof_map_local[i, j])
                tag = boundary_tags.get(gidx, "")
                pos = nodes[i, j]
                p3 = np.append(pos, 0.0)

                if tag in ("circle", "outer"):
                    color = "red" if tag == "circle" else "blue"
                    sphere = pv.Sphere(radius=0.045, center=p3)
                    plotter.add_mesh(sphere, color=color, render_points_as_spheres=True)


class LiveGridView:
    """Animated grid view that updates mesh points in place (no per-frame rebuild).

    ``_add_grid_meshes`` rebuilds every actor each frame — a fill + wireframe per
    block plus one ``pv.Sphere`` mesh per boundary node — after ``plotter.clear()``.
    For an animated relaxation that is the dominant cost. This builds each actor
    once and, on :meth:`update`, only overwrites the stored meshes' ``.points`` and
    issues a single render, which is orders of magnitude cheaper.

    Boundary markers are drawn as two point clouds (circle / outer) rendered as
    spheres, rather than one sphere mesh per node.
    """

    _FILL_COLORS = [
        "lightblue", "lightcoral", "lightgreen", "lightyellow",
        "lightsalmon", "plum", "palegreen", "wheat",
    ]
    _EDGE_COLORS = ["cyan", "magenta", "yellow", "lime", "orange", "white"]

    def __init__(
        self,
        plotter: "pv.Plotter",
        grid,
        boundary_tags: dict[int, str] | None = None,
        geometry_entities: list | None = None,
        show_edge_verts: bool = True,
    ) -> None:
        import pyvista as pv

        self.plotter = plotter
        self.grid = grid
        boundary_tags = boundary_tags or {}

        if geometry_entities:
            for entity in geometry_entities:
                _add_entity_curve(plotter, entity)

        # Greedy-colour the block adjacency graph so adjacent blocks never
        # share a fill colour.
        block_names = list(grid.topology.block_specs.keys())
        block_index: dict[str, int] = {name: i for i, name in enumerate(block_names)}
        adj: dict[int, set[int]] = {bi: set() for bi in range(len(block_names))}
        for conn in grid.topology.interface_connections:
            a = block_index.get(conn.face_a.block_name)
            b = block_index.get(conn.face_b.block_name)
            if a is not None and b is not None and a != b:
                adj[a].add(b)
                adj[b].add(a)
        fill_colour_idx: dict[int, int] = {}
        for bi in range(len(block_names)):
            forbidden = {fill_colour_idx.get(nb, -1) for nb in adj[bi]}
            c = 0
            while c in forbidden:
                c += 1
            fill_colour_idx[bi] = c

        # Per-block fill (PolyData) + wireframe (StructuredGrid), built once.
        self._fills: list = []          # (surf, block_idx)
        self._wires: list = []          # (sg, block_idx)
        for bi, block in enumerate(grid.blocks):
            if block.d != 2:
                continue
            nodes = block.nodes
            ni, nj = block.logical_shape[0], block.logical_shape[1]

            pts_3d = self._fill_points(nodes)
            cells = []
            for i in range(ni - 1):
                for j in range(nj - 1):
                    a = i * nj + j
                    cells.append([4, a, a + 1, (i + 1) * nj + j + 1, (i + 1) * nj + j])
            if cells:
                surf = pv.PolyData(pts_3d, np.hstack(cells))
                plotter.add_mesh(
                    surf, color=self._FILL_COLORS[fill_colour_idx[bi] % len(self._FILL_COLORS)],
                    opacity=0.35, show_edges=True, edge_color="gray", line_width=0.5,
                )
                self._fills.append((surf, bi))

            sg = pv.StructuredGrid(*self._wire_coords(nodes))
            plotter.add_mesh(
                sg, color=self._EDGE_COLORS[bi % len(self._EDGE_COLORS)],
                style="wireframe", line_width=1.5,
            )
            self._wires.append((sg, bi))

        # Boundary markers: one point cloud per tag, updated in place.
        self._clouds: list = []         # (cloud, gids)
        if show_edge_verts:
            for tag, color in (("circle", "red"), ("outer", "blue")):
                gids = np.array(
                    [g for g, t in boundary_tags.items() if t == tag], dtype=int
                )
                if gids.size == 0:
                    continue
                cloud = pv.PolyData(self._cloud_points(gids))
                plotter.add_mesh(
                    cloud, color=color, point_size=10, render_points_as_spheres=True,
                )
                self._clouds.append((cloud, gids))

    @staticmethod
    def _fill_points(nodes) -> np.ndarray:
        pts = nodes.reshape(-1, 2)
        return np.column_stack([pts, np.zeros(len(pts))])

    @staticmethod
    def _wire_coords(nodes):
        x = np.asarray(nodes[..., 0], dtype=float)
        y = np.asarray(nodes[..., 1], dtype=float)
        return x, y, np.zeros_like(x)

    def _cloud_points(self, gids) -> np.ndarray:
        pos = self.grid.global_nodes[gids]
        return np.column_stack([pos, np.zeros(len(pos))])

    def update(self, render: bool = True) -> None:
        """Overwrite stored mesh points from current node positions and render."""
        for surf, bi in self._fills:
            surf.points = self._fill_points(self.grid.blocks[bi].nodes)
        for sg, bi in self._wires:
            x, y, z = self._wire_coords(self.grid.blocks[bi].nodes)
            sg.points = np.column_stack(
                [x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")]
            )
        for cloud, gids in self._clouds:
            cloud.points = self._cloud_points(gids)
        if render:
            self.plotter.render()


def animate_pipeline(grid, geometry_entities, steps, *, title="pipeline",
                     show_edge_verts=True):
    """Drive a :func:`egg.pipeline.generate_steps` generator from a PyVista
    timer so the grid animates the folded → untangled → smoothed transition.

    The window opens on the (tangled) initial grid and advances one step per
    timer tick. For the untangle phase to animate per δ, build the generator with
    ``untangle_direct=False``.
    """
    import pyvista as pv
    from egg.geometry.analytic2d import Circle

    def _fmt(info: dict) -> str:
        return " ".join(
            f"{k}={v:.4e}" if isinstance(v, float) else f"{k}={v}"
            for k, v in info.items())

    tags = {g: ("circle" if isinstance(e, Circle) else "outer")
            for g, e in grid.dof_constraints.items()}

    plotter = pv.Plotter()
    view = LiveGridView(plotter, grid, tags, geometry_entities,
                        show_edge_verts=show_edge_verts)
    plotter.view_xy()
    plotter.show_axes()
    plotter.add_text(f"{title}: tangled start", name="t", font_size=12)

    state = {"done": False}

    def cb(_step):
        if state["done"]:
            return
        try:
            phase, info = next(steps)
        except StopIteration:
            state["done"] = True
            plotter.add_text(f"{title}: done", name="t", font_size=12)
            return
        txt = f"{phase}: {_fmt(info)}"
        print("  " + txt)
        plotter.add_text(txt, name="t", font_size=12)
        view.update()

    # One step per tick; max_steps is a generous upper bound (generator ends first).
    plotter.add_timer_event(max_steps=100000, duration=200, callback=cb)
    plotter.show()
