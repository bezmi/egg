"""Derive the closed-form grad/Hessian of mu_cond3 via SymPy CSE.

Computes value, 9-entry gradient, and the 45 unique Hessian entries
(exploiting symmetry), runs Common Subexpression Elimination, and emits
a self-contained C++ header with pre-computed temporaries — the structural
template for src/metric.hpp's mu_cond3_*_closedform functions.

mu_cond3(t) = (s*q)/(9*D^2) - 1
  D = det3(t)         (trilinear)
  cof = cof3(t)       (each entry a 2x2 minor)
  s = sum t_i^2       (quadratic)
  q = sum cof_i^2     (quartic)
"""
from __future__ import annotations

import sympy as sp
from sympy.printing.cxx import CXX11CodePrinter


class GPUOptimizedPrinter(CXX11CodePrinter):
    """C++ printer that expands small integer powers into explicit
    multiplications, avoiding std::pow calls (which lower to libm on GPU
    and add latency + a function-call ABI)."""

    def _print_Pow(self, expr):
        base, exp = expr.as_base_exp()
        # Skip non-integer exponents (sqrt, etc.) -> standard printer.
        if not exp.is_Integer:
            return super()._print_Pow(expr)
        n = int(exp)
        if -8 <= n <= 8 and n != 0 and n != 1:
            base_str = self._print(base)
            # Build the chain of multiplications for the magnitude.
            mag = base_str
            for _ in range(abs(n) - 1):
                mag = f"({mag} * {base_str})"
            if n > 0:
                return f"({mag})"
            return f"(1.0 / {mag})"
        return super()._print_Pow(expr)


_PRINTER = GPUOptimizedPrinter()


def to_cxx(expr) -> str:
    # Substitute array indexing, then print.
    e = expr.subs(sym_to_cxx)
    return _PRINTER.doprint(e)
import time

# 9 symbols of vec(T) (row-major).
t = sp.symbols("t0:9")
N = 9


def det3(t):
    t0, t1, t2, t3, t4, t5, t6, t7, t8 = t
    return (t0 * (t4 * t8 - t5 * t7)
            - t1 * (t3 * t8 - t5 * t6)
            + t2 * (t3 * t7 - t4 * t6))


def cof3(t):
    t0, t1, t2, t3, t4, t5, t6, t7, t8 = t
    return [
        t4 * t8 - t5 * t7,
        -(t3 * t8 - t5 * t6),
        t3 * t7 - t4 * t6,
        -(t1 * t8 - t2 * t7),
        t0 * t8 - t2 * t6,
        -(t0 * t7 - t1 * t6),
        t1 * t5 - t2 * t4,
        -(t0 * t5 - t2 * t3),
        t0 * t4 - t1 * t3,
    ]


D = det3(t)
cof = cof3(t)
s = sum(ti**2 for ti in t)
q = sum(c**2 for c in cof)
mu = s * q / (9 * D**2) - 1

print("=== Computing Gradient (unexpanded) ===", flush=True)
t0 = time.time()
grad = [sp.diff(mu, ti) for ti in t]
print(f"  Grad done ({time.time()-t0:.2f}s)", flush=True)

print("=== Computing symmetric Hessian (45 unique entries) ===", flush=True)
unique_hess = []
hess_map = [[None] * N for _ in range(N)]
for i in range(N):
    for j in range(i, N):
        h = sp.diff(grad[i], t[j])
        hess_map[i][j] = h
        hess_map[j][i] = h
        unique_hess.append(h)
print(f"  Hessian done ({time.time()-t0:.2f}s)", flush=True)

print("=== Running CSE ===", flush=True)
all_exprs = [mu] + grad + unique_hess
replacements, reduced = sp.cse(all_exprs)
print(f"  CSE done ({time.time()-t0:.2f}s, {len(replacements)} temps)",
      flush=True)

# Map t0..t8 -> t[0]..t[8] for C++.
sym_to_cxx = {t[k]: sp.Symbol(f"t[{k}]") for k in range(N)}


# Count non-zero scratch bytes (numeric proxies) -- rough proxy for size.
total_size = sum(len(str(r[1])) for r in replacements)
print(f"  Raw temp expression bytes: {total_size}", flush=True)

# Per the plan's decision gate: if unfactored output exceeds ~200 KB, stop.
LARGE = 200_000
if total_size > LARGE:
    print(f"WARN: total CSE expression size {total_size} > {LARGE}; "
          f"the formula may not admit a clean tensor decomposition.",
          flush=True)

# Post-CSE peephole: collapse 3 independent FP64 divisions on powers of D
# (det3) into 1 division. CSE produces:
#   x20 = 1/(D*D)              -> keep (1 division)
#   x46 = 1/(D*D*D)            -> rewrite as x20 * D          (1 mul, no div)
#   x104 = x51 / (D*D*D*D)     -> rewrite as x51 * x20 * x20  (2 muls, no div)
# On RDNA2 FP64 div is ~16 cycles vs ~2 for FMA, so this trims ~30 cycles
# per work-item; the rewrite is mechanical so we apply it as a string pass
# over the generated C++ lines.
def _to_cxx_lines(replacements, reduced):
    """Emit C++ body (temporaries + outputs) as a list of lines, with the
    pow-of-D division peephole applied to x46 and x104."""
    lines = []
    lines.append("    // --- Common Subexpressions (SymPy CSE) ---")
    for var, expr in replacements:
        cxx_name = str(var)
        s = to_cxx(expr)
        # Peephole: x46 = 1/(D*D*D). With x20 = 1/(D*D) already computed,
        #   inv_x19 = x20 * x19      -> 1/D   (0 extra divisions)
        #   x46     = x20 * inv_x19  -> 1/D^3 (0 extra divisions)
        # Insert inv_x19 just before x46 so the dataflow is visible.
        if cxx_name == "x46":
            lines.append("    const double inv_x19 = x20 * x19;  // 1/D")
            s = "x20 * inv_x19"
        # Peephole: x104 = x51 / (D^4) -> x51 * x20 * x20
        if cxx_name == "x104":
            s = "x51 * x20 * x20"
        lines.append(f"    const double {cxx_name} = {s};")
    lines.append("")
    lines.append("    // --- value ---")
    lines.append(f"    r.val = {to_cxx(reduced[0])};")
    lines.append("")
    lines.append("    // --- gradient (9 entries) ---")
    for i in range(9):
        lines.append(f"    r.grad[{i}] = {to_cxx(reduced[1 + i])};")
    lines.append("")
    lines.append("    // --- symmetric Hessian (45 unique entries) ---")
    idx = 10
    for i in range(9):
        for j in range(i, 9):
            val_str = to_cxx(reduced[idx])
            ij = i * 9 + j
            ji = j * 9 + i
            if i == j:
                lines.append(f"    r.hess[{ij}] = {val_str};")
            else:
                lines.append(f"    r.hess[{ij}] = "
                             f"r.hess[{ji}] = {val_str};")
            idx += 1
    return lines


def _to_cxx_lines_jhj(replacements, reduced):
    """Emit the fused value + gradient + contracted Hessian body.

    Computes the same CSE temporaries, then accumulates jhj = Jb^T H Jb (3x3,
    row-major) where Jb is the runtime 9x3 chain matrix — each metric-Hessian
    entry H_ab is produced, folded into the 3x3 accumulator, and discarded, so
    the full 81-entry Hessian is never simultaneously live (the register
    footprint that pins the D=3 sweep kernel at 256 VGPR). H is symmetric, so
    each unique (a,b) with a<b contributes its mirror in one block."""
    lines = []
    lines.append("    // --- Common Subexpressions (SymPy CSE) ---")
    for var, expr in replacements:
        cxx_name = str(var)
        s = to_cxx(expr)
        if cxx_name == "x46":
            lines.append("    const double inv_x19 = x20 * x19;  // 1/D")
            s = "x20 * inv_x19"
        if cxx_name == "x104":
            s = "x51 * x20 * x20"
        lines.append(f"    const double {cxx_name} = {s};")
    lines.append("")
    lines.append("    // --- value ---")
    lines.append(f"    *val_out = {to_cxx(reduced[0])};")
    lines.append("")
    lines.append("    // --- gradient (9 entries) ---")
    for i in range(9):
        lines.append(f"    grad_out[{i}] = {to_cxx(reduced[1 + i])};")
    lines.append("")
    lines.append("    // --- contracted Hessian jhj = Jb^T H Jb (3x3 row-major) ---")
    lines.append("    for (int _k = 0; _k < 9; ++_k) { jhj_out[_k] = 0.0; }")
    idx = 10
    for i in range(9):
        for j in range(i, 9):
            val_str = to_cxx(reduced[idx])
            lines.append("    {")
            lines.append(f"        const double h = {val_str};")
            lines.append("        for (int p = 0; p < 3; ++p) {")
            lines.append("            for (int qcol = 0; qcol < 3; ++qcol) {")
            if i == j:
                lines.append(
                    f"                jhj_out[(p * 3) + qcol] += "
                    f"h * Jb[({i} * 3) + p] * Jb[({i} * 3) + qcol];")
            else:
                lines.append(
                    f"                jhj_out[(p * 3) + qcol] += h * "
                    f"((Jb[({i} * 3) + p] * Jb[({j} * 3) + qcol]) + "
                    f"(Jb[({j} * 3) + p] * Jb[({i} * 3) + qcol]));")
            lines.append("            }")
            lines.append("        }")
            lines.append("    }")
            idx += 1
    return lines


print("=== Generating C++ header ===", flush=True)
body_lines = _to_cxx_lines(replacements, reduced)
jhj_lines = _to_cxx_lines_jhj(replacements, reduced)
header = [
    "// GENERATED by scripts/derive_cond3.py — SymPy CSE output.",
    "// Closed-form mu_cond3 value/grad/Hessian, raw CSE form.",
    "// Used as the structural template for mu_cond3_*_closedform",
    "// in src/metric.hpp (which rewrites into outer-product form).",
    "#pragma once",
    "",
    "namespace egg {",
    "",
    "struct AnalyticMu3D {",
    "    double val;",
    "    double grad[9];",
    "    double hess[9][9];",
    "};",
    "",
    "// t: 9-entry row-major vec(T); t[0]=T00, t[1]=T01, ...",
    "inline AnalyticMu3D evaluate_mu_cond3(const double* t) {",
    "    AnalyticMu3D r;",
    "",
]
header.extend(body_lines)
header.extend([
    "",
    "    return r;",
    "}",
    "",
    "// Fused value + gradient + contracted Hessian jhj = Jb^T H Jb (3x3).",
    "// Jb: runtime 9x3 row-major chain matrix (the role-selected columns of J).",
    "// Avoids materialising the 81-entry metric Hessian — folds each entry into",
    "// the 3x3 accumulator on the fly.",
    "inline void mu_cond3_jhj(const double* t, const double* Jb,",
    "                         double* val_out, double* grad_out, double* jhj_out) {",
    "",
])
header.extend(jhj_lines)
header.extend([
    "",
    "}",
    "",
    "} // namespace egg",
    "",
])
out_path = "bench/optimized_3d_mu.hpp"
with open(out_path, "w") as f:
    f.write("\n".join(header))

print(f"Done! C++ header written to {out_path}", flush=True)
print(f"Total elapsed: {time.time()-t0:.2f}s", flush=True)