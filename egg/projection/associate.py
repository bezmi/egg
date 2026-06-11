"""Node↔entity tagging hierarchy: classify each DOF as free / sliding / fixed."""

from __future__ import annotations

from itertools import product
from typing import Any

__all__ = ["ConstraintHierarchy", "build_dof_constraints"]


def build_dof_constraints(topology, grid):
    """Classify every global DOF from the face↔entity associations.

    A DOF that lies on the associated faces of:

      * 0 entities    -> free (interior / non-geometry node)
      * 1 entity      -> sliding on that entity (orthogonal projection kept)
      * >= 2 entities -> fixed (a geometry corner where two entities meet,
                         e.g. a domain corner shared by two walls)

    Entities are compared by identity, so a node shared by two faces that are
    associated with the *same* entity (e.g. a wall landing point shared by an
    edge block and a corner block) is still classified as sliding.

    Returns
    -------
    dof_constraints : dict[int, entity]   sliding DOFs -> their entity
    fixed_dofs : set[int]                 DOFs to lock (>= 2 entities)
    """
    d = topology.d
    block_names = list(topology.block_specs.keys())
    dof_entities: dict[int, list[Any]] = {}

    for assoc in topology.associations:
        spec = topology.block_specs[assoc.face.block_name]
        axis, side = assoc.face.axis, assoc.face.side
        entity = assoc.entity
        fixed = 0 if side == 0 else spec.logical_shape[axis] - 1
        free_axes = [a for a in range(d) if a != axis]
        bi = block_names.index(assoc.face.block_name)
        dof_map = grid.block_dof_maps[bi]

        for free_idx in product(*[range(spec.logical_shape[a]) for a in free_axes]):
            logical = [0] * d
            logical[axis] = fixed
            for j, a in enumerate(free_axes):
                logical[a] = free_idx[j]
            gidx = int(dof_map[tuple(logical)])
            ents = dof_entities.setdefault(gidx, [])
            if not any(e is entity for e in ents):
                ents.append(entity)

    dof_constraints: dict[int, Any] = {}
    fixed_dofs: set[int] = set()
    for gidx, ents in dof_entities.items():
        if len(ents) == 1:
            dof_constraints[gidx] = ents[0]
        else:  # >= 2 entities meet here -> fixed corner
            fixed_dofs.add(gidx)
    return dof_constraints, fixed_dofs


class ConstraintHierarchy:
    """Thin holder for per-DOF geometry constraints (sliding + fixed sets)."""

    def __init__(self, dof_constraints: dict[int, Any], fixed_dofs: set[int]):
        self.dof_constraints = dict(dof_constraints)
        self.fixed_dofs = set(fixed_dofs)

    @classmethod
    def from_topology(cls, topology, grid) -> "ConstraintHierarchy":
        return cls(*build_dof_constraints(topology, grid))
