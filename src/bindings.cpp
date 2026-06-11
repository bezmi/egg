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
#include <sycl/sycl.hpp>

namespace
{

sycl::queue select_queue(const std::string& device)
{
    if (device == "cpu") {
        return sycl::queue {sycl::cpu_selector_v};
    } else if (device == "gpu") {
        return sycl::queue {sycl::gpu_selector_v};
    }
    return sycl::queue {};  // "auto" — default selector
}

// Extract a contiguous int32 numpy array from a dict key, returning a vector.
std::vector<int> extract_int(py::dict d, const std::string& key)
{
    auto arr = d[key.c_str()].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
    return std::vector<int>(arr.data(), arr.data() + arr.size());
}

// Extract a contiguous float64 numpy array from a dict key, returning a vector.
std::vector<double> extract_double(py::dict d, const std::string& key)
{
    auto arr =
      d[key.c_str()].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    return std::vector<double>(arr.data(), arr.data() + arr.size());
}

egg::SweepContextHost
  unpack_context(py::dict ctx_arrays, const double* X_data, std::size_t num_nodes)
{
    using namespace egg;

    SweepContextHost host;
    host.num_nodes = num_nodes;
    host.X.assign(X_data, X_data + num_nodes * 2);

    // Groups — one ragged group per colour. Per-sample arrays are flat over the
    // concatenated DOFs (DOF-major, variable P per DOF); per-DOF arrays carry the
    // patch size P_of and we derive the sample_offset prefix sum here. See
    // cpp_backend_plan.md §5 and flatten_context.
    py::list groups_list = ctx_arrays["groups"].cast<py::list>();
    for (auto g_item : groups_list) {
        py::dict gd = g_item.cast<py::dict>();
        SweepGroupHost sg;
        sg.D = gd["D"].cast<std::size_t>();

        sg.gc = extract_int(gd, "gc");
        sg.gn0 = extract_int(gd, "gn0");
        sg.gn1 = extract_int(gd, "gn1");
        sg.s0 = extract_double(gd, "s0");
        sg.s1 = extract_double(gd, "s1");
        sg.W_inv = extract_double(gd, "W_inv");
        sg.role = extract_int(gd, "role");
        sg.J = extract_double(gd, "J");
        sg.dof_idx = extract_int(gd, "dof_idx");
        sg.tag = extract_int(gd, "tag");
        sg.P_of = extract_int(gd, "P_of");
        sg.params = extract_double(gd, "params");

        sg.total_samples = sg.gc.size();

        // sample_offset[d] = Σ_{k<d} P_of[k] (exclusive prefix sum).
        sg.sample_offset.resize(sg.D);
        std::exclusive_scan(sg.P_of.begin(), sg.P_of.end(), sg.sample_offset.begin(), 0);

        host.groups.push_back(std::move(sg));
    }

    // Energy stencil
    py::dict es = ctx_arrays["energy_stencil"].cast<py::dict>();
    host.energy_stencil.num_samples = es["num_samples"].cast<std::size_t>();
    host.energy_stencil.gc = extract_int(es, "gc");
    host.energy_stencil.gn0 = extract_int(es, "gn0");
    host.energy_stencil.gn1 = extract_int(es, "gn1");
    host.energy_stencil.s0 = extract_double(es, "s0");
    host.energy_stencil.s1 = extract_double(es, "s1");
    host.energy_stencil.W_inv = extract_double(es, "W_inv");

    return host;
}

// Persistent device-resident session: the context is uploaded once and X is
// kept resident across .run() calls
// Thin RAII wrapper over Executor, which already owns the in-order queue and
// keeps X device-resident.
class CppSweepSession
{
  public:
    CppSweepSession(py::dict ctx_arrays,
                    py::array_t<double, py::array::c_style | py::array::forcecast> X0,
                    const std::string& device) :
        num_nodes_(static_cast<std::size_t>(X0.request().shape[0]) / 2),
        exec_(select_queue(device), unpack_context(ctx_arrays, X0.data(), num_nodes_))
    {
    }

    // Run n_sweeps on the resident X. No upload/download.
    std::tuple<py::array_t<double>, py::array_t<double>>
      run(int n_sweeps, const std::string& phase, double delta)
    {
        const egg::ObjectiveKind objective = egg::make_objective(phase, delta);
        auto [energies, mindets] = exec_.run_sweeps(n_sweeps, objective);
        py::array_t<double> e_ret(static_cast<py::ssize_t>(n_sweeps), energies.data());
        py::array_t<double> m_ret(static_cast<py::ssize_t>(n_sweeps), mindets.data());
        return std::make_tuple(e_ret, m_ret);
    }

    // Download a host copy of the resident X.
    py::array_t<double> get_X()
    {
        std::vector<double> X_out(num_nodes_ * 2);
        exec_.ctx().download_X(X_out.data());
        return py::array_t<double>(static_cast<py::ssize_t>(num_nodes_ * 2), X_out.data());
    }

    // Re-upload X to the device.
    void set_X(py::array_t<double, py::array::c_style | py::array::forcecast> X_arr)
    {
        auto buf = X_arr.request();
        if (static_cast<std::size_t>(buf.shape[0]) != num_nodes_ * 2) {
            throw std::invalid_argument("set_X: shape mismatch with session num_nodes");
        }
        std::vector<double> host(X_arr.data(), X_arr.data() + num_nodes_ * 2);
        exec_.ctx().upload_X(host);
    }

  private:
    std::size_t num_nodes_;
    egg::Executor exec_;
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
                    const std::string&>(),
           py::arg("ctx_arrays"),
           py::arg("X"),
           py::kw_only(),
           py::arg("device") = "auto")
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
      [](
        py::dict ctx_arrays,
        py::array_t<double, py::array::c_style | py::array::forcecast> X_arr,
        int n_sweeps,
        const std::string& device,
        const std::string& phase,
        double delta) -> std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> {
          // Select queue
          sycl::queue q = select_queue(device);

          // Unpack context
          auto X_buf = X_arr.request();
          const std::size_t num_nodes = X_buf.shape[0] / 2;
          auto host_ctx = unpack_context(ctx_arrays, X_arr.data(), num_nodes);

          // Build device context + executor
          egg::Executor exec(q, host_ctx);

          // Dispatch the objective kind (barrier / δ-untangle) once via the variant.
          const egg::ObjectiveKind objective = egg::make_objective(phase, delta);

          // Run sweeps
          auto [energies, mindets] = exec.run_sweeps(n_sweeps, objective);

          // Copy X back
          std::vector<double> X_out(num_nodes * 2);
          exec.ctx().download_X(X_out.data());

          // Return (X_out, energies, mindets) as numpy arrays
          py::array_t<double> X_ret(static_cast<py::ssize_t>(num_nodes * 2), X_out.data());
          py::array_t<double> e_ret(static_cast<py::ssize_t>(n_sweeps), energies.data());
          py::array_t<double> m_ret(static_cast<py::ssize_t>(n_sweeps), mindets.data());

          // pybind11 copies by default when returning from raw pointers — that is
          // the correct behaviour here (the vectors go out of scope).
          return std::make_tuple(X_ret, e_ret, m_ret);
      },
      py::arg("ctx_arrays"),
      py::arg("X"),
      py::arg("n_sweeps"),
      py::kw_only(),
      py::arg("device") = "auto",
      py::arg("phase") = "barrier",
      py::arg("delta") = 0.0,
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
      [](py::array_t<double, py::array::c_style | py::array::forcecast> t_arr)
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
      [](py::array_t<double, py::array::c_style | py::array::forcecast> p_arr,
         int tag,
         py::array_t<double, py::array::c_style | py::array::forcecast> params_arr)
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
      [](py::array_t<double, py::array::c_style | py::array::forcecast> p_arr,
         int tag,
         py::array_t<double, py::array::c_style | py::array::forcecast> params_arr)
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
      [](py::array_t<double, py::array::c_style | py::array::forcecast> X_arr,
         py::array_t<int, py::array::c_style | py::array::forcecast> gc_arr,
         py::array_t<int, py::array::c_style | py::array::forcecast> gn0_arr,
         py::array_t<int, py::array::c_style | py::array::forcecast> gn1_arr,
         py::array_t<double, py::array::c_style | py::array::forcecast> s0_arr,
         py::array_t<double, py::array::c_style | py::array::forcecast> s1_arr,
         py::array_t<double, py::array::c_style | py::array::forcecast> W_inv_arr,
         py::array_t<int, py::array::c_style | py::array::forcecast> role_arr,
         py::array_t<double, py::array::c_style | py::array::forcecast> J_arr)
        -> std::tuple<py::array_t<double>, py::array_t<double>, double, double> {
          egg::PatchView pv;
          pv.P = static_cast<int>(gc_arr.size());
          pv.gc = gc_arr.data();
          pv.gn0 = gn0_arr.data();
          pv.gn1 = gn1_arr.data();
          pv.s0 = s0_arr.data();
          pv.s1 = s1_arr.data();
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
      [](py::array_t<double, py::array::c_style | py::array::forcecast> X_arr,
         py::array_t<int, py::array::c_style | py::array::forcecast> gc_arr,
         py::array_t<int, py::array::c_style | py::array::forcecast> gn0_arr,
         py::array_t<int, py::array::c_style | py::array::forcecast> gn1_arr,
         py::array_t<double, py::array::c_style | py::array::forcecast> s0_arr,
         py::array_t<double, py::array::c_style | py::array::forcecast> s1_arr,
         py::array_t<double, py::array::c_style | py::array::forcecast> W_inv_arr)
        -> std::tuple<double, double> {
          egg::PatchView pv;
          pv.P = static_cast<int>(gc_arr.size());
          pv.gc = gc_arr.data();
          pv.gn0 = gn0_arr.data();
          pv.gn1 = gn1_arr.data();
          pv.s0 = s0_arr.data();
          pv.s1 = s1_arr.data();
          pv.W_inv = W_inv_arr.data();
          pv.role = gc_arr.data();  // dummy, not used by energy_mindet
          pv.J = W_inv_arr.data();  // dummy, not used by energy_mindet
          double energy = 0.0, mindet = 0.0;
          egg::patch_energy_mindet(pv, X_arr.data(), energy, mindet);
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
      [](py::array_t<double, py::array::c_style | py::array::forcecast> grad_arr,
         py::array_t<double, py::array::c_style | py::array::forcecast> hess_arr,
         py::array_t<double, py::array::c_style | py::array::forcecast> pos_arr,
         int tag,
         py::array_t<double, py::array::c_style | py::array::forcecast> params_arr)
        -> py::array_t<double> {
          const auto* g = grad_arr.data();
          const auto* h = hess_arr.data();
          egg::Vec2 gv {g[0], g[1]};
          egg::Mat2 hv {h[0], h[1], h[2], h[3]};
          egg::Pt pos {pos_arr.data()[0], pos_arr.data()[1]};
          egg::Vec2 delta = egg::newton_delta(gv, hv, pos, tag, params_arr.data());
          return py::array_t<double>(2, delta.data());
      },
      py::arg("grad"),
      py::arg("hess"),
      py::arg("pos"),
      py::arg("tag"),
      py::arg("params"),
      "Tangent-reduced Newton step. Returns delta(2,).");
}
