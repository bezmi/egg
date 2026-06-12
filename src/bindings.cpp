#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "geometry.hpp"
#include "metric.hpp"
#include "patch.hpp"
#include "solve.hpp"
#include "sweep.hpp"

#include <numeric>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>

#ifndef NDEBUG
    #include <unordered_set>
#endif
#include <variant>

namespace
{
using std::invalid_argument;

sycl::queue select_queue(const std::string& device)
{
    // cpu_selector_v / gpu_selector_v throw sycl::exception with a terse message
    // when no matching device is present; rewrap with a clear, actionable error.
    try {
        if (device == "cpu") {
            return sycl::queue {sycl::cpu_selector_v};
        } else if (device == "gpu") {
            return sycl::queue {sycl::gpu_selector_v};
        }
        return sycl::queue {};  // "auto" — default selector
    } catch (const sycl::exception& e) {
        throw std::runtime_error("select_queue: no SYCL device available for device='" + device +
                                 "' (" + e.what() +
                                 "). Try device='cpu', or check the AdaptiveCpp install / "
                                 "ACPP_VISIBILITY_MASK.");
    }
}

// num_nodes from a flat X array, with a hard shape check: X must be a 1-D array
// whose length is a multiple of d (d coordinates per node). Catches the silent-UB
// case where an (N, d) array is passed and num_nodes is miscomputed (#1).
std::size_t
  checked_num_nodes(const py::array_t<double, py::array::c_style | py::array::forcecast>& X_arr,
                    int d)
{
    auto buf = X_arr.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument(
          "X must be a flat 1-D array of length d*N; got ndim=" + std::to_string(buf.ndim) +
          " (did you pass an (N, d) array? ravel() it first).");
    }
    if (buf.shape[0] % d != 0) {
        throw std::invalid_argument("X length must be a multiple of d (" + std::to_string(d) +
                                    " coordinates per node); got " + std::to_string(buf.shape[0]) +
                                    ".");
    }
    return static_cast<std::size_t>(buf.shape[0]) / d;
}

// Host-side validation of an unpacked context against num_nodes: array-length
// invariants, the ragged sum(P_of) == total_samples contract, and index bounds
// for every gc/gn0/gn1/dof_idx. Throws std::invalid_argument before any device
// work, so a malformed context can never reach the kernels as silent OOB / UB
// (#1). This is the single biggest guard for a memory-safe core.
template <int D> void validate_context(const egg::SweepContextHostT<D>& host)
{
    const std::size_t N = host.num_nodes;

    auto check_index = [N](const std::vector<int>& idx, const char* name, const std::string& grp) {
        for (int v : idx) {
            if (v < 0 || static_cast<std::size_t>(v) >= N) {
                throw std::invalid_argument("validate_context: " + grp + " " + name + " index " +
                                            std::to_string(v) + " out of range [0, " +
                                            std::to_string(N) + ")");
            }
        }
    };

    if (host.X.size() != N * D) {
        throw std::invalid_argument("validate_context: X size " + std::to_string(host.X.size()) +
                                    " != d*num_nodes " + std::to_string(N * D));
    }

    for (std::size_t gi = 0; gi < host.groups.size(); ++gi) {
        const auto& g = host.groups[gi];
        const std::string grp = "group[" + std::to_string(gi) + "]";
        const std::size_t ts = g.total_samples;
        auto eq = [&](std::size_t got, std::size_t want, const std::string& what) {
            if (got != want) {
                throw invalid_argument("validate_context: " + grp + " " + what + " size " +
                                       std::to_string(got) + " != " + std::to_string(want));
            }
        };
        eq(g.gc.size(), ts, "gc");
        for (int k = 0; k < D; ++k) {
            eq(g.gn[k].size(), ts, "gn" + std::to_string(k));
            eq(g.s[k].size(), ts, "s" + std::to_string(k));
        }
        eq(g.role.size(), ts, "role");
        eq(g.W_inv.size(), ts * egg::dim::wInv(D), "W_inv");
        eq(g.J.size(), ts * egg::dim::jSize(D), "J");
        eq(g.dof_idx.size(), g.ndof, "dof_idx");
        eq(g.tag.size(), g.ndof, "tag");
        eq(g.P_of.size(), g.ndof, "P_of");
        eq(g.params.size(), g.ndof * egg::kParamPad, "params");

        // role selects the per-corner stencil branch (corner / neighbour axis /
        // absent) and, via c0 = D*role, the J columns in patch_eval — an
        // out-of-range role would read past jCols into the next sample's J.
        for (int r : g.role) {
            if (r < -1 || r > egg::dim::nbrs(D)) {
                throw std::invalid_argument("validate_context: " + grp + " role " +
                                            std::to_string(r) + " out of range [-1, " +
                                            std::to_string(egg::dim::nbrs(D)) + "]");
            }
        }

        std::size_t sumP = 0;
        for (int p : g.P_of) {
            if (p < 0) {
                throw std::invalid_argument("validate_context: " + grp + " negative P_of");
            }
            sumP += static_cast<std::size_t>(p);
        }
        if (sumP != ts) {
            throw std::invalid_argument("validate_context: " + grp + " sum(P_of) " +
                                        std::to_string(sumP) + " != total_samples " +
                                        std::to_string(ts));
        }

        check_index(g.gc, "gc", grp);
        for (int k = 0; k < D; ++k) {
            check_index(g.gn[k], ("gn" + std::to_string(k)).c_str(), grp);
        }
        check_index(g.dof_idx, "dof_idx", grp);
    }

    const auto& es = host.energy_stencil;
    const std::size_t ns = es.num_samples;
    auto eqs = [&](std::size_t got, std::size_t want, const std::string& what) {
        if (got != want) {
            throw std::invalid_argument("validate_context: energy_stencil " + what + " size " +
                                        std::to_string(got) + " != " + std::to_string(want));
        }
    };
    eqs(es.gc.size(), ns, "gc");
    for (int k = 0; k < D; ++k) {
        eqs(es.gn[k].size(), ns, "gn" + std::to_string(k));
        eqs(es.s[k].size(), ns, "s" + std::to_string(k));
    }
    eqs(es.W_inv.size(), ns * egg::dim::wInv(D), "W_inv");
    check_index(es.gc, "gc", "energy_stencil");
    for (int k = 0; k < D; ++k) {
        check_index(es.gn[k], ("gn" + std::to_string(k)).c_str(), "energy_stencil");
    }
}

#ifndef NDEBUG
// Debug-only check of the host colouring contract: within one colour group no
// DOF's patch may reference another concurrently-moved DOF, otherwise the
// same-colour parallel scatter races against a neighbour's update (#6). Compiled
// out in Release (NDEBUG), where the host promises a valid greedy colouring.
template <int D> void assert_group_race_free(const egg::SweepContextHostT<D>& host)
{
    for (std::size_t gi = 0; gi < host.groups.size(); ++gi) {
        const auto& g = host.groups[gi];
        const std::unordered_set<int> moved(g.dof_idx.begin(), g.dof_idx.end());
        for (std::size_t d = 0; d < g.ndof; ++d) {
            const int self = g.dof_idx[d];
            const std::size_t off = static_cast<std::size_t>(g.sample_offset[d]);
            const std::size_t P = static_cast<std::size_t>(g.P_of[d]);
            for (std::size_t p = off; p < off + P; ++p) {
                bool ref = g.gc[p] != self && moved.count(g.gc[p]);
                int culprit = g.gc[p];
                for (int k = 0; k < D && !ref; ++k) {
                    if (g.gn[k][p] != self && moved.count(g.gn[k][p])) {
                        ref = true;
                        culprit = g.gn[k][p];
                    }
                }
                if (ref) {
                    throw std::logic_error("race-free colouring violated: group " +
                                           std::to_string(gi) + " DOF " + std::to_string(self) +
                                           " patch references concurrently-moved DOF " +
                                           std::to_string(culprit));
                }
            }
        }
    }
}
#endif

// Extract a contiguous int32 numpy array from a dict key, returning a vector.
std::vector<int> extract_int(const py::dict& d, const std::string& key)
{
    auto arr = d[key.c_str()].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
    return std::vector<int>(arr.data(), arr.data() + arr.size());
}

// Extract a contiguous float64 numpy array from a dict key, returning a vector.
std::vector<double> extract_double(const py::dict& d, const std::string& key)
{
    auto arr =
      d[key.c_str()].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    return std::vector<double>(arr.data(), arr.data() + arr.size());
}

template <int D>
egg::SweepContextHostT<D>
  unpack_context(const py::dict& ctx_arrays, const double* X_data, std::size_t num_nodes)
{
    using namespace egg;

    SweepContextHostT<D> host;
    host.num_nodes = num_nodes;
    host.X.assign(X_data, X_data + (num_nodes * D));

    // Groups — one ragged group per colour. Per-sample arrays are flat over the
    // concatenated DOFs (DOF-major, variable P per DOF); per-DOF arrays carry the
    // patch size P_of and we derive the sample_offset prefix sum here. The wire
    // format keeps per-axis keys gn0..gn{D-1} / s0..s{D-1}; we map them into the
    // gn[k]/s[k] array slots. See cpp_backend_plan.md §5 and flatten_context.
    auto groups_list = ctx_arrays["groups"].cast<py::list>();
    for (auto g_item : groups_list) {
        auto gd = g_item.cast<py::dict>();
        SweepGroupHostT<D> sg;
        sg.ndof = gd["D"].cast<std::size_t>();

        sg.gc = extract_int(gd, "gc");
        for (int k = 0; k < D; ++k) {
            sg.gn[k] = extract_int(gd, "gn" + std::to_string(k));
            sg.s[k] = extract_double(gd, "s" + std::to_string(k));
        }
        sg.W_inv = extract_double(gd, "W_inv");
        sg.role = extract_int(gd, "role");
        sg.J = extract_double(gd, "J");
        sg.dof_idx = extract_int(gd, "dof_idx");
        sg.tag = extract_int(gd, "tag");
        sg.P_of = extract_int(gd, "P_of");
        sg.params = extract_double(gd, "params");
        // Optional: variable-length entity data (B-spline nets/knots). Absent for
        // contexts with only fixed-size entities.
        if (gd.contains("arena")) { sg.arena = extract_double(gd, "arena"); }

        sg.total_samples = sg.gc.size();

        // sample_offset[d] = Σ_{k<d} P_of[k] (exclusive prefix sum).
        sg.sample_offset.resize(sg.ndof);
        std::exclusive_scan(sg.P_of.begin(), sg.P_of.end(), sg.sample_offset.begin(), 0);

        host.groups.push_back(std::move(sg));
    }

    // Energy stencil
    auto es = ctx_arrays["energy_stencil"].cast<py::dict>();
    host.energy_stencil.num_samples = es["num_samples"].cast<std::size_t>();
    host.energy_stencil.gc = extract_int(es, "gc");
    for (int k = 0; k < D; ++k) {
        host.energy_stencil.gn[k] = extract_int(es, "gn" + std::to_string(k));
        host.energy_stencil.s[k] = extract_double(es, "s" + std::to_string(k));
    }
    host.energy_stencil.W_inv = extract_double(es, "W_inv");

    // Fail fast on a malformed context before any device allocation / kernel.
    validate_context<D>(host);
#ifndef NDEBUG
    assert_group_race_free<D>(host);
#endif

    return host;
}

// Runtime dispatch on the input's actual dimension d: 2 and 3 are built.
int require_supported_dim(int d)
{
    if (d != 2 && d != 3) {
        throw std::invalid_argument("egg C++ core: dimension must be 2 or 3, got " +
                                    std::to_string(d));
    }
    return d;
}

// Persistent device-resident session: the context is uploaded once and X is
// kept resident across .run() calls. Thin RAII wrapper over ExecutorT<D>,
// which already owns the in-order queue and keeps X device-resident; the
// runtime dimension picks the variant alternative at construction.
class CppSweepSession
{
  public:
    CppSweepSession(const py::dict& ctx_arrays,
                    const py::array_t<double, py::array::c_style | py::array::forcecast>& X0,
                    const std::string& device,
                    int dim) :
        dim_(require_supported_dim(dim)), num_nodes_(checked_num_nodes(X0, dim_)),
        exec_(dim_ == 2 ? ExecVariant {std::in_place_type<egg::ExecutorT<2>>,
                                       select_queue(device),
                                       unpack_context<2>(ctx_arrays, X0.data(), num_nodes_)}
                        : ExecVariant {std::in_place_type<egg::ExecutorT<3>>,
                                       select_queue(device),
                                       unpack_context<3>(ctx_arrays, X0.data(), num_nodes_)})
    {
    }

    // Run n_sweeps on the resident X. No upload/download.
    std::tuple<py::array_t<double>, py::array_t<double>>
      run(int n_sweeps, const std::string& phase, double delta)
    {
        auto [energies, mindets] = std::visit(
          [&]<int D>(egg::ExecutorT<D>& exec) {
              const egg::ObjectiveKindT<D> objective = egg::make_objective<D>(phase, delta);
              return exec.run_sweeps(n_sweeps, objective);
          },
          exec_);
        py::array_t<double> e_ret(static_cast<py::ssize_t>(n_sweeps), energies.data());
        py::array_t<double> m_ret(static_cast<py::ssize_t>(n_sweeps), mindets.data());
        return std::make_tuple(e_ret, m_ret);
    }

    // Download a host copy of the resident X.
    py::array_t<double> get_X()
    {
        std::vector<double> X_out(num_nodes_ * dim_);
        std::visit([&](auto& exec) { exec.ctx().download_X(X_out.data()); }, exec_);
        return py::array_t<double>(static_cast<py::ssize_t>(num_nodes_ * dim_), X_out.data());
    }

    // Re-upload X to the device.
    void set_X(const py::array_t<double, py::array::c_style | py::array::forcecast>& X_arr)
    {
        auto buf = X_arr.request();
        if (static_cast<std::size_t>(buf.shape[0]) != num_nodes_ * dim_) {
            throw std::invalid_argument("set_X: shape mismatch with session num_nodes");
        }
        std::vector<double> host(X_arr.data(), X_arr.data() + num_nodes_ * dim_);
        std::visit([&](auto& exec) { exec.ctx().upload_X(host); }, exec_);
    }

  private:
    using ExecVariant = std::variant<egg::ExecutorT<2>, egg::ExecutorT<3>>;
    int dim_;
    std::size_t num_nodes_;
    ExecVariant exec_;
};

}  // namespace

PYBIND11_MODULE(cpp_core, m)
{
    m.doc() = "egg C++ compute core (AdaptiveCpp).";
    m.def("ping", [] { return 0; }, "Liveness check; returns 0.");

    py::class_<CppSweepSession>(
      m,
      "CppSweepSession",
      "Persistent device-resident smoothing session.\n\n"
      "The flattened context is uploaded once at construction and X is kept\n"
      "device-resident across .run() calls — no per-call upload/download or\n"
      "re-JIT, so steady-state per-sweep cost can be measured warm.")
      .def(py::init<py::dict,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    const std::string&,
                    int>(),
           py::arg("ctx_arrays"),
           py::arg("X"),
           py::kw_only(),
           py::arg("device") = "auto",
           py::arg("dim") = 2)
      .def("run",
           &CppSweepSession::run,
           py::arg("n_sweeps"),
           py::kw_only(),
           py::arg("phase") = "barrier",
           py::arg("delta") = 0.0,
           "Run n_sweeps on the resident X; returns (energies, mindets).")
      .def("get_X",
           &CppSweepSession::get_X,
           "Download a host copy of the resident X, shape (N*2,).")
      .def("set_X",
           &CppSweepSession::set_X,
           py::arg("X"),
           "Re-upload X to the device, shape (N*2,).");

    m.def(
      "cpp_sweep",
      [](const py::dict& ctx_arrays,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& X_arr,
         int n_sweeps,
         const std::string& device,
         const std::string& phase,
         double delta,
         int dim) -> std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> {
          // Runtime dispatch on the requested dimension (2 or 3).
          require_supported_dim(dim);

          const auto run = [&]<int D>() {
              // Select queue
              sycl::queue q = select_queue(device);

              // Unpack context (X shape is checked here; the context is validated
              // inside unpack_context before any device work).
              const std::size_t num_nodes = checked_num_nodes(X_arr, dim);
              auto host_ctx = unpack_context<D>(ctx_arrays, X_arr.data(), num_nodes);

              // Build device context + executor
              egg::ExecutorT<D> exec(q, host_ctx);

              // Dispatch the objective kind (barrier / δ-untangle) once via the variant.
              const egg::ObjectiveKindT<D> objective = egg::make_objective<D>(phase, delta);

              // Run sweeps
              auto [energies, mindets] = exec.run_sweeps(n_sweeps, objective);

              // Copy X back
              std::vector<double> X_out(num_nodes * D);
              exec.ctx().download_X(X_out.data());

              // Return (X_out, energies, mindets) as numpy arrays. pybind11 copies
              // by default when returning from raw pointers — correct here (the
              // vectors go out of scope).
              py::array_t<double> X_ret(static_cast<py::ssize_t>(num_nodes * D), X_out.data());
              py::array_t<double> e_ret(static_cast<py::ssize_t>(n_sweeps), energies.data());
              py::array_t<double> m_ret(static_cast<py::ssize_t>(n_sweeps), mindets.data());
              return std::make_tuple(X_ret, e_ret, m_ret);
          };
          return dim == 2 ? run.template operator()<2>() : run.template operator()<3>();
      },
      py::arg("ctx_arrays"),
      py::arg("X"),
      py::arg("n_sweeps"),
      py::kw_only(),
      py::arg("device") = "auto",
      py::arg("phase") = "barrier",
      py::arg("delta") = 0.0,
      py::arg("dim") = 2,
      "Run n_sweeps of colored Gauss-Seidel smoothing.\n\n"
      "Mirrors build_fused_multisweep.run(): same X0 and sweep count\n"
      "produce matching per-sweep energies and min det A.\n\n"
      "Parameters\n"
      "----------\n"
      "ctx_arrays : dict\n"
      "    Flattened context (groups + energy_stencil) from cpp_backend.flatten_context.\n"
      "X : ndarray, shape (N*2,)\n"
      "    Initial node positions, flat.\n"
      "n_sweeps : int\n"
      "    Number of sweeps to run.\n"
      "device : str, optional\n"
      "    'auto' (default), 'cpu', or 'gpu'.\n"
      "phase : str, optional\n"
      "    'barrier' (default) or 'untangle' (δ-continuation surrogate).\n"
      "delta : float, optional\n"
      "    δ for the untangle surrogate (ignored for the barrier phase).\n\n"
      "Returns\n"
      "-------\n"
      "X_out : ndarray, shape (N*2,)\n"
      "    Final node positions.\n"
      "energies : ndarray, shape (n_sweeps,)\n"
      "    Per-sweep total energy.\n"
      "mindets : ndarray, shape (n_sweeps,)\n"
      "    Per-sweep min det A.");

    // --------------------------------------------------------------------------
    // Oracle primitives — host-side wrappers for golden-generator reference values.
    // SYCL-free; these call the inline C++ header functions directly.
    // --------------------------------------------------------------------------

    m.def(
      "metric_eval",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& t_arr)
        -> std::tuple<double, py::array_t<double>, py::array_t<double>> {
          const auto* t = t_arr.data();
          egg::VecT vt {t[0], t[1], t[2], t[3]};
          egg::ShapeObjective objective;
          double mu = objective.value(vt);
          auto g = objective.grad(vt);
          auto h = objective.hess(vt);
          py::array_t<double> grad(4, g.data());
          py::array_t<double> hess(16, h.data());
          return std::make_tuple(mu, grad, hess);
      },
      py::arg("t"),
      "Evaluate shape_2d metric. Returns (mu, grad(4,), hess(16,)).");

    m.def(
      "geometry_project",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& p_arr,
         int tag,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& params_arr)
        -> py::array_t<double> {
          egg::Pt p {p_arr.data()[0], p_arr.data()[1]};
          egg::Pt proj = egg::project(p, tag, params_arr.data());
          return py::array_t<double>(2, proj.data());
      },
      py::arg("p"),
      py::arg("tag"),
      py::arg("params"),
      "Project point onto entity. Returns proj(2,).");

    m.def(
      "geometry_tangent",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& p_arr,
         int tag,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& params_arr)
        -> py::array_t<double> {
          egg::Pt p {p_arr.data()[0], p_arr.data()[1]};
          egg::Pt tang = egg::tangent_space(p, tag, params_arr.data());
          return py::array_t<double>(2, tang.data());
      },
      py::arg("p"),
      py::arg("tag"),
      py::arg("params"),
      "Tangent direction at point on entity. Returns tang(2,).");

    m.def(
      "patch_eval",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& X_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& gc_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& gn0_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& gn1_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& s0_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& s1_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& W_inv_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& role_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& J_arr)
        -> std::tuple<py::array_t<double>, py::array_t<double>, double, double> {
          egg::PatchView pv;
          pv.P = static_cast<int>(gc_arr.size());
          pv.gc = gc_arr.data();
          pv.gn[0] = gn0_arr.data();
          pv.gn[1] = gn1_arr.data();
          pv.s[0] = s0_arr.data();
          pv.s[1] = s1_arr.data();
          pv.W_inv = W_inv_arr.data();
          pv.role = role_arr.data();
          pv.J = J_arr.data();
          auto result = egg::patch_eval(pv, X_arr.data());
          py::array_t<double> grad(2, result.grad.data());
          py::array_t<double> hess(4, result.hess.data());
          return std::make_tuple(grad, hess, result.energy, result.mindet);
      },
      py::arg("X"),
      py::arg("gc"),
      py::arg("gn0"),
      py::arg("gn1"),
      py::arg("s0"),
      py::arg("s1"),
      py::arg("W_inv"),
      py::arg("role"),
      py::arg("J"),
      "Per-DOF patch evaluation. Returns (grad(2,), hess(4,), energy, mindet).");

    m.def(
      "energy_mindet",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& X_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& gc_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& gn0_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& gn1_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& s0_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& s1_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& W_inv_arr)
        -> std::tuple<double, double> {
          egg::StencilSampleView sv;
          sv.P = static_cast<int>(gc_arr.size());
          sv.gc = gc_arr.data();
          sv.gn[0] = gn0_arr.data();
          sv.gn[1] = gn1_arr.data();
          sv.s[0] = s0_arr.data();
          sv.s[1] = s1_arr.data();
          sv.W_inv = W_inv_arr.data();
          double energy = 0.0, mindet = 0.0;
          egg::patch_energy_mindet<2>(sv, X_arr.data(), energy, mindet);
          return std::make_tuple(energy, mindet);
      },
      py::arg("X"),
      py::arg("gc"),
      py::arg("gn0"),
      py::arg("gn1"),
      py::arg("s0"),
      py::arg("s1"),
      py::arg("W_inv"),
      "Cheap patch energy + min-det only. Returns (energy, mindet).");

    m.def(
      "newton_step",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& grad_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& hess_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& pos_arr,
         int tag,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& params_arr)
        -> py::array_t<double> {
          const auto* g = grad_arr.data();
          const auto* h = hess_arr.data();
          egg::Vec2 gv {g[0], g[1]};
          egg::Mat2 hv {h[0], h[1], h[2], h[3]};
          egg::Pt pos {pos_arr.data()[0], pos_arr.data()[1]};
          egg::Vec2 delta = egg::newton_delta<2>(gv, hv, pos, tag, params_arr.data());
          return py::array_t<double>(2, delta.data());
      },
      py::arg("grad"),
      py::arg("hess"),
      py::arg("pos"),
      py::arg("tag"),
      py::arg("params"),
      "Tangent-reduced Newton step. Returns delta(2,).");
}
