# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

# ruff: noqa: E402  (script: imports interleaved with generation stages)
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
    and add latency + a function-call ABI).

    Every floating literal is emitted with the egg ``_r`` suffix so it carries
    ``egg::real`` precision: in an ``egg::real`` expression a bare ``2.0`` is a
    ``double`` and silently promotes the whole subexpression to double (then
    narrows back) — defeating the fp32 register/bandwidth win. Integer literals
    stay bare (``3 * x`` does not force double)."""

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
            return f"(1.0_r / {mag})"
        return super()._print_Pow(expr)

    def _print_Float(self, expr):
        # e.g. 9.0 -> 9.0_r ; 1.0e-5 -> 1.0e-5_r (the _r UDL takes long double).
        return super()._print_Float(expr) + "_r"

    def _print_Rational(self, expr):
        # SymPy keeps exact fractions (2/9, 4/9, ...). Emit both halves as real
        # literals so the division stays in egg::real, not double.
        return f"{expr.p}.0_r/{expr.q}.0_r"


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
    return (
        t0 * (t4 * t8 - t5 * t7) - t1 * (t3 * t8 - t5 * t6) + t2 * (t3 * t7 - t4 * t6)
    )


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
print(f"  Grad done ({time.time() - t0:.2f}s)", flush=True)

print("=== Computing symmetric Hessian (45 unique entries) ===", flush=True)
unique_hess = []
hess_map = [[None] * N for _ in range(N)]
for i in range(N):
    for j in range(i, N):
        h = sp.diff(grad[i], t[j])
        hess_map[i][j] = h
        hess_map[j][i] = h
        unique_hess.append(h)
print(f"  Hessian done ({time.time() - t0:.2f}s)", flush=True)

print("=== Running CSE ===", flush=True)
all_exprs = [mu] + grad + unique_hess
replacements, reduced = sp.cse(all_exprs)
print(f"  CSE done ({time.time() - t0:.2f}s, {len(replacements)} temps)", flush=True)

# Map t0..t8 -> t[0]..t[8] for C++.
sym_to_cxx = {t[k]: sp.Symbol(f"t[{k}]") for k in range(N)}


# Count non-zero scratch bytes (numeric proxies) -- rough proxy for size.
total_size = sum(len(str(r[1])) for r in replacements)
print(f"  Raw temp expression bytes: {total_size}", flush=True)

# Per the plan's decision gate: if unfactored output exceeds ~200 KB, stop.
LARGE = 200_000
if total_size > LARGE:
    print(
        f"WARN: total CSE expression size {total_size} > {LARGE}; "
        f"the formula may not admit a clean tensor decomposition.",
        flush=True,
    )


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
            lines.append("    const real inv_x19 = x20 * x19;  // 1/D")
            s = "x20 * inv_x19"
        # Peephole: x104 = x51 / (D^4) -> x51 * x20 * x20
        if cxx_name == "x104":
            s = "x51 * x20 * x20"
        lines.append(f"    const real {cxx_name} = {s};")
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
                lines.append(f"    r.hess[{ij}] = r.hess[{ji}] = {val_str};")
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
            lines.append("    const real inv_x19 = x20 * x19;  // 1/D")
            s = "x20 * inv_x19"
        if cxx_name == "x104":
            s = "x51 * x20 * x20"
        lines.append(f"    const real {cxx_name} = {s};")
    lines.append("")
    lines.append("    // --- value ---")
    lines.append(f"    *val_out = {to_cxx(reduced[0])};")
    lines.append("")
    lines.append("    // --- gradient (9 entries) ---")
    for i in range(9):
        lines.append(f"    grad_out[{i}] = {to_cxx(reduced[1 + i])};")
    lines.append("")
    lines.append("    // --- contracted Hessian jhj = Jb^T H Jb (3x3 row-major) ---")
    lines.append("    for (int _k = 0; _k < 9; ++_k) { jhj_out[_k] = 0.0_r; }")
    idx = 10
    for i in range(9):
        for j in range(i, 9):
            val_str = to_cxx(reduced[idx])
            lines.append("    {")
            lines.append(f"        const real h = {val_str};")
            lines.append("        for (int p = 0; p < 3; ++p) {")
            lines.append("            for (int qcol = 0; qcol < 3; ++qcol) {")
            if i == j:
                lines.append(
                    f"                jhj_out[(p * 3) + qcol] += "
                    f"h * Jb[({i} * 3) + p] * Jb[({i} * 3) + qcol];"
                )
            else:
                lines.append(
                    f"                jhj_out[(p * 3) + qcol] += h * "
                    f"((Jb[({i} * 3) + p] * Jb[({j} * 3) + qcol]) + "
                    f"(Jb[({j} * 3) + p] * Jb[({i} * 3) + qcol]));"
                )
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
    "    real val;",
    "    real grad[9];",
    "    real hess[9][9];",
    "};",
    "",
    "// t: 9-entry row-major vec(T); t[0]=T00, t[1]=T01, ...",
    "inline AnalyticMu3D evaluate_mu_cond3(const real* t) {",
    "    AnalyticMu3D r;",
    "",
]
header.extend(body_lines)
header.extend(
    [
        "",
        "    return r;",
        "}",
        "",
        "// Fused value + gradient + contracted Hessian jhj = Jb^T H Jb (3x3).",
        "// Jb: runtime 9x3 row-major chain matrix (the role-selected columns of J).",
        "// Avoids materialising the 81-entry metric Hessian — folds each entry into",
        "// the 3x3 accumulator on the fly.",
        "inline void mu_cond3_jhj(const real* t, const real* Jb,",
        "                         real* val_out, real* grad_out, real* jhj_out) {",
        "",
    ]
)
header.extend(jhj_lines)
header.extend(
    [
        "",
        "}",
        "",
        "} // namespace egg",
        "",
    ]
)
out_path = "bench/optimized_3d_mu.hpp"
with open(out_path, "w") as f:
    f.write("\n".join(header))

print(f"Done! C++ header written to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Directional second-derivative form of jhj = Jb^T H Jb (the VGPR fix).
#
# The full-Hessian jhj (kept above as the reference) CSEs all 45 unique H entries
# from one shared pool (~230 temps), all simultaneously live -> the 256-VGPR pin.
# Instead, contract H via 6 scalar directional second derivatives:
#     D2(w) := w^T H w = mu''(t; w)                 (one scalar per direction)
#     jhj[i][i] = D2(v_i)                           (v_i = Jb[:,i])
#     jhj[i][j] = 1/2 (D2(v_i+v_j) - D2(v_i) - D2(v_j))   (polarization)
#
# D2 is expressed analytically through the building blocks D=det, s=sum t^2,
# q=sum cof^2 (f = s q / 9D^2, mu = f - 1, D2 = f'') and the cofactor directional
# derivatives. A FLAT w^T H w SymPy contraction does NOT help: it CSEs into one
# giant return referencing ~218 simultaneously-live temps (measured: 256 VGPR /
# 520 B scratch, WORSE than the full Hessian). The structured analytic form with
# the cofactors/s/q/D precomputed ONCE and shared across the 6 directions
# measured 248 VGPR / 96 B scratch / 0.873 ms (off the cap; was 256/296/1.23).
# d2dir/jhj are therefore emitted as a fixed analytic template (below), NOT CSE,
# and self-checked numerically against w^T H w here. Only valgrad is CSE-emitted.
# ---------------------------------------------------------------------------
print("=== Directional 2nd-derivative jhj generation ===", flush=True)
w = sp.symbols("w0:9")
sym_to_cxx.update({w[k]: sp.Symbol(f"w[{k}]") for k in range(N)})

# --- numeric self-check: analytic D2 (the template's math) vs w^T H w ---------
import random

_idx = [
    (4, 8, 5, 7),
    (5, 6, 3, 8),
    (3, 7, 4, 6),
    (2, 7, 1, 8),
    (0, 8, 2, 6),
    (1, 6, 0, 7),
    (1, 5, 2, 4),
    (2, 3, 0, 5),
    (0, 4, 1, 3),
]
_cof = [t[a] * t[b] - t[c] * t[d] for (a, b, c, d) in _idx]
_cp = [w[a] * t[b] + t[a] * w[b] - w[c] * t[d] - t[c] * w[d] for (a, b, c, d) in _idx]
_cpp = [2 * (w[a] * w[b] - w[c] * w[d]) for (a, b, c, d) in _idx]
_D = t[0] * _cof[0] + t[1] * _cof[1] + t[2] * _cof[2]
_s = sum(ti**2 for ti in t)
_q = sum(c**2 for c in _cof)
_sp = 2 * sum(t[i] * w[i] for i in range(N))
_spp = 2 * sum(w[i] * w[i] for i in range(N))
_Dp = sum(w[i] * _cof[i] for i in range(N))
_Dpp = sum(w[i] * _cp[i] for i in range(N))
_qp = 2 * sum(_cof[i] * _cp[i] for i in range(N))
_qpp = 2 * (
    sum(_cp[i] * _cp[i] for i in range(N)) + sum(_cof[i] * _cpp[i] for i in range(N))
)
_h = 9 * _D**2
_g = _s * _q
_gp = _sp * _q + _s * _qp
_gpp = _spp * _q + 2 * _sp * _qp + _s * _qpp
_hp = 18 * _D * _Dp
_hpp = 18 * (_Dp**2 + _D * _Dpp)
_D2_template = (
    _gpp / _h - 2 * _gp * _hp / _h**2 - _g * _hpp / _h**2 + 2 * _g * _hp**2 / _h**3
)
_D2_ref = (sp.Matrix(w).T * sp.hessian(mu, t) * sp.Matrix(w))[0]
_f_t = sp.lambdify(t + w, _D2_template, "math")
_f_r = sp.lambdify(t + w, _D2_ref, "math")
random.seed(0)
_maxerr = 0.0
for _ in range(3000):
    _v = [random.uniform(-2, 2) for _ in range(18)]
    _a, _b = _f_r(*_v), _f_t(*_v)
    _maxerr = max(_maxerr, abs(_a - _b) / (1 + abs(_a)))
assert _maxerr < 1e-9, f"analytic D2 template mismatch vs w^T H w: {_maxerr}"
print(f"  analytic D2 template self-check OK (max rel err {_maxerr:.2e})", flush=True)

# value + gradient only (CSE; pool dies after writing to memory).
repl_vg, red_vg = sp.cse([mu] + grad)
print(f"  valgrad CSE done ({time.time() - t0:.2f}s, {len(repl_vg)} temps)", flush=True)


def _emit_cse_temps(replacements):
    return [f"    const real {var} = {to_cxx(expr)};" for var, expr in replacements]


# d2dir + jhj: fixed analytic template (precompute-once, self-checked above).
D2DIR_JHJ_TEMPLATE = r"""// mu''(t; w): the 2nd directional derivative of mu_cond3 at t along the 9-vector
// w. The cofactors cof[9] and the scalars s, q, D=det are precomputed ONCE by
// mu_cond3_jhj and passed in (shared across all 6 directions), so this body is
// only the cheap w-contractions -> small (force-inlined) 6x working set.
// mu = f - 1, f = s q / (9 D^2); D2 = f''. cof_i = t[a]t[b] - t[c]t[d] with the
// (a,b,c,d) map below; cp = d_w cof_i, cpp = d^2_w cof_i.
__attribute__((noinline)) inline real mu_cond3_d2dir(
  const real* t, const real* cof, real s, real q, real D, const real* w)
{
    // Cofactor directional derivatives. Index map (a,b,c,d) per i:
    //  0:(4,8,5,7) 1:(5,6,3,8) 2:(3,7,4,6) 3:(2,7,1,8) 4:(0,8,2,6)
    //  5:(1,6,0,7) 6:(1,5,2,4) 7:(2,3,0,5) 8:(0,4,1,3)
    const real cp0 = (w[4] * t[8]) + (t[4] * w[8]) - (w[5] * t[7]) - (t[5] * w[7]);
    const real cp1 = (w[5] * t[6]) + (t[5] * w[6]) - (w[3] * t[8]) - (t[3] * w[8]);
    const real cp2 = (w[3] * t[7]) + (t[3] * w[7]) - (w[4] * t[6]) - (t[4] * w[6]);
    const real cp3 = (w[2] * t[7]) + (t[2] * w[7]) - (w[1] * t[8]) - (t[1] * w[8]);
    const real cp4 = (w[0] * t[8]) + (t[0] * w[8]) - (w[2] * t[6]) - (t[2] * w[6]);
    const real cp5 = (w[1] * t[6]) + (t[1] * w[6]) - (w[0] * t[7]) - (t[0] * w[7]);
    const real cp6 = (w[1] * t[5]) + (t[1] * w[5]) - (w[2] * t[4]) - (t[2] * w[4]);
    const real cp7 = (w[2] * t[3]) + (t[2] * w[3]) - (w[0] * t[5]) - (t[0] * w[5]);
    const real cp8 = (w[0] * t[4]) + (t[0] * w[4]) - (w[1] * t[3]) - (t[1] * w[3]);
    const real cpp0 = 2.0_r * ((w[4] * w[8]) - (w[5] * w[7]));
    const real cpp1 = 2.0_r * ((w[5] * w[6]) - (w[3] * w[8]));
    const real cpp2 = 2.0_r * ((w[3] * w[7]) - (w[4] * w[6]));
    const real cpp3 = 2.0_r * ((w[2] * w[7]) - (w[1] * w[8]));
    const real cpp4 = 2.0_r * ((w[0] * w[8]) - (w[2] * w[6]));
    const real cpp5 = 2.0_r * ((w[1] * w[6]) - (w[0] * w[7]));
    const real cpp6 = 2.0_r * ((w[1] * w[5]) - (w[2] * w[4]));
    const real cpp7 = 2.0_r * ((w[2] * w[3]) - (w[0] * w[5]));
    const real cpp8 = 2.0_r * ((w[0] * w[4]) - (w[1] * w[3]));

    // Directional 1st/2nd derivatives of the building blocks s, D, q along w.
    const real sp = 2.0_r * ((t[0] * w[0]) + (t[1] * w[1]) + (t[2] * w[2]) + (t[3] * w[3]) +
                             (t[4] * w[4]) + (t[5] * w[5]) + (t[6] * w[6]) + (t[7] * w[7]) +
                             (t[8] * w[8]));
    const real spp = 2.0_r * ((w[0] * w[0]) + (w[1] * w[1]) + (w[2] * w[2]) + (w[3] * w[3]) +
                              (w[4] * w[4]) + (w[5] * w[5]) + (w[6] * w[6]) + (w[7] * w[7]) +
                              (w[8] * w[8]));
    const real Dp = (w[0] * cof[0]) + (w[1] * cof[1]) + (w[2] * cof[2]) + (w[3] * cof[3]) +
                      (w[4] * cof[4]) + (w[5] * cof[5]) + (w[6] * cof[6]) + (w[7] * cof[7]) +
                      (w[8] * cof[8]);
    const real Dpp = (w[0] * cp0) + (w[1] * cp1) + (w[2] * cp2) + (w[3] * cp3) + (w[4] * cp4) +
                       (w[5] * cp5) + (w[6] * cp6) + (w[7] * cp7) + (w[8] * cp8);
    const real qp = 2.0_r * ((cof[0] * cp0) + (cof[1] * cp1) + (cof[2] * cp2) + (cof[3] * cp3) +
                             (cof[4] * cp4) + (cof[5] * cp5) + (cof[6] * cp6) + (cof[7] * cp7) +
                             (cof[8] * cp8));
    const real qpp = 2.0_r * ((cp0 * cp0) + (cp1 * cp1) + (cp2 * cp2) + (cp3 * cp3) + (cp4 * cp4) +
                              (cp5 * cp5) + (cp6 * cp6) + (cp7 * cp7) + (cp8 * cp8) +
                              (cof[0] * cpp0) + (cof[1] * cpp1) + (cof[2] * cpp2) +
                              (cof[3] * cpp3) + (cof[4] * cpp4) + (cof[5] * cpp5) +
                              (cof[6] * cpp6) + (cof[7] * cpp7) + (cof[8] * cpp8));

    // f = s q / (9 D^2); D2 = f'' via product/quotient rule on g = s q, h = 9 D^2.
    const real h = 9.0_r * D * D;
    const real invh = 1.0_r / h;
    const real hp = 18.0_r * D * Dp;
    const real hpp = 18.0_r * ((Dp * Dp) + (D * Dpp));
    const real g = s * q;
    const real gp = (sp * q) + (s * qp);
    const real gpp = (spp * q) + (2.0_r * sp * qp) + (s * qpp);
    return (gpp * invh) - (2.0_r * gp * hp * invh * invh) - (g * hpp * invh * invh) +
           (2.0_r * g * hp * hp * invh * invh * invh);
}

%VALGRAD%

// Fused value + gradient + contracted Hessian jhj = Jb^T H Jb (3x3 row-major) via
// 6 directional 2nd derivatives + polarization. Jb is the runtime 9x3 row-major
// chain matrix; column i is v_i (v_i[k] = Jb[k*3 + i]). Never forms the 81-entry
// Hessian. The cofactors / s / q / D are computed ONCE here and shared by the 6
// mu_cond3_d2dir calls.
__attribute__((noinline)) inline void mu_cond3_jhj(
  const real* t, const real* Jb, real* val_out, real* grad_out, real* jhj_out)
{
    mu_cond3_valgrad(t, val_out, grad_out);

    // Shared building blocks (computed once, reused for all 6 polarization dirs).
    real cof[9];
    cof[0] = (t[4] * t[8]) - (t[5] * t[7]);
    cof[1] = (t[5] * t[6]) - (t[3] * t[8]);
    cof[2] = (t[3] * t[7]) - (t[4] * t[6]);
    cof[3] = (t[2] * t[7]) - (t[1] * t[8]);
    cof[4] = (t[0] * t[8]) - (t[2] * t[6]);
    cof[5] = (t[1] * t[6]) - (t[0] * t[7]);
    cof[6] = (t[1] * t[5]) - (t[2] * t[4]);
    cof[7] = (t[2] * t[3]) - (t[0] * t[5]);
    cof[8] = (t[0] * t[4]) - (t[1] * t[3]);
    const real s = (t[0] * t[0]) + (t[1] * t[1]) + (t[2] * t[2]) + (t[3] * t[3]) +
                     (t[4] * t[4]) + (t[5] * t[5]) + (t[6] * t[6]) + (t[7] * t[7]) + (t[8] * t[8]);
    const real q = (cof[0] * cof[0]) + (cof[1] * cof[1]) + (cof[2] * cof[2]) + (cof[3] * cof[3]) +
                     (cof[4] * cof[4]) + (cof[5] * cof[5]) + (cof[6] * cof[6]) + (cof[7] * cof[7]) +
                     (cof[8] * cof[8]);
    const real D = (t[0] * cof[0]) + (t[1] * cof[1]) + (t[2] * cof[2]);

    // The 3 columns of Jb are the polarization directions v0, v1, v2.
    real v0[9], v1[9], v2[9];
    for (int k = 0; k < 9; ++k) {
        v0[k] = Jb[(k * 3) + 0];
        v1[k] = Jb[(k * 3) + 1];
        v2[k] = Jb[(k * 3) + 2];
    }
    const real d00 = mu_cond3_d2dir(t, cof, s, q, D, v0);
    const real d11 = mu_cond3_d2dir(t, cof, s, q, D, v1);
    const real d22 = mu_cond3_d2dir(t, cof, s, q, D, v2);

    // Off-diagonals by polarization; reuse one scratch direction (low liveness).
    real w[9];
    for (int k = 0; k < 9; ++k) { w[k] = v0[k] + v1[k]; }
    const real j01 = 0.5_r * (mu_cond3_d2dir(t, cof, s, q, D, w) - d00 - d11);
    for (int k = 0; k < 9; ++k) { w[k] = v0[k] + v2[k]; }
    const real j02 = 0.5_r * (mu_cond3_d2dir(t, cof, s, q, D, w) - d00 - d22);
    for (int k = 0; k < 9; ++k) { w[k] = v1[k] + v2[k]; }
    const real j12 = 0.5_r * (mu_cond3_d2dir(t, cof, s, q, D, w) - d11 - d22);

    jhj_out[0] = d00; jhj_out[1] = j01; jhj_out[2] = j02;
    jhj_out[3] = j01; jhj_out[4] = d11; jhj_out[5] = j12;
    jhj_out[6] = j02; jhj_out[7] = j12; jhj_out[8] = d22;
}"""

valgrad_lines = [
    "// value + gradient of mu_cond3 at t, written to memory (CSE pool dies here).",
    "__attribute__((noinline)) inline void mu_cond3_valgrad(",
    "  const real* t, real* val_out, real* grad_out)",
    "{",
    "    // --- Common Subexpressions (SymPy CSE) ---",
]
valgrad_lines += _emit_cse_temps(repl_vg)
valgrad_lines += ["", f"    *val_out = {to_cxx(red_vg[0])};"]
valgrad_lines += [f"    grad_out[{i}] = {to_cxx(red_vg[1 + i])};" for i in range(9)]
valgrad_lines += ["}"]

body = D2DIR_JHJ_TEMPLATE.replace("%VALGRAD%", "\n".join(valgrad_lines))
dir_header = "\n".join(
    [
        "// GENERATED by bench/derive_cond3.py — directional 2nd-derivative jhj.",
        "// Paste mu_cond3_d2dir / mu_cond3_valgrad / mu_cond3_jhj into src/metric.hpp",
        "// (replacing the full-Hessian mu_cond3_jhj). Keep mu_cond3_eval as the parity",
        "// reference. clang-format -i src/metric.hpp afterward.",
        "#pragma once",
        "",
        "namespace egg {",
        "",
        body,
        "",
        "} // namespace egg",
        "",
    ]
)
dir_path = "bench/optimized_3d_mu_directional.hpp"
with open(dir_path, "w") as f:
    f.write(dir_header)
print(f"Done! Directional jhj header written to {dir_path}", flush=True)
print(f"Total elapsed: {time.time() - t0:.2f}s", flush=True)
