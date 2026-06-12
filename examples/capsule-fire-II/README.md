# Capsule FIRE II

FIRE II capsule forebody, ported from gdtk's lmr 2D `capsule-fire-II` case.
The Lua original places a 4 x 13 control-point net over a channel-e2w guide
patch; those control points (exported to `capsule_ctrl_pts.vts`) are used
here as the corners of a 3 x 12 multiblock topology. Boundary faces are
snapped to the gdtk-defined paths and tagged `inflow`, `wall`, `symmetry`
and `outflow` for SU2 export.

```sh
uv run capsule.py --plot-live              # animated relaxation
uv run capsule.py --plot-topology          # block layout only
uv run capsule.py --export capsule.su2     # final mesh as SU2
```

`draw_topology.py` plots the raw control net from the VTS file.
