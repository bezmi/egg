# Capsule FIRE II

FIRE II capsule forebody, ported from gdtk's lmr 2D `capsule-fire-II` case.
The domain between the gdtk-defined paths (outer arc, capsule body,
stagnation line, exit line) is filled with a 3 x 12 block array whose
corners are placed parametrically on the paths; where the Lua original
shapes the interior with a hand-tuned `ControlPointPatch` net, the TMOP
smoothing pass does that job. Boundary faces are tagged `inflow`, `wall`,
`symmetry` and `outflow` for SU2 export.

```sh
uv run capsule.py --plot-live              # animated relaxation
uv run capsule.py --plot-topology          # block layout only
uv run capsule.py --export capsule.su2     # final mesh as SU2
```
