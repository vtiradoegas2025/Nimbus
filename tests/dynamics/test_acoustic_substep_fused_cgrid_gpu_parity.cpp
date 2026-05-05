/**
 * @file test_acoustic_substep_fused_cgrid_gpu_parity.cpp
 * @brief Fused-vs-sequential parity for the C-grid acoustic substep
 *        (Phase C.9.6 of docs/CoordinateBackend_Plan.md).
 *
 * Phase C.9.6 records the C.9.4 pressure pipeline + a compute-to-compute
 * barrier + the C.9.5 momentum pipeline in ONE Vulkan command buffer
 * submission, halving the GPU dispatch overhead per acoustic substep
 * (~16 substeps per dynamics step in Klemp-Wilhelmson).
 *
 * The fused path must produce IDENTICAL results to two back-to-back
 * single-substep dispatches: the underlying shaders are the same and
 * fed the same float inputs, so the outputs should be bit-exact equal.
 * Any drift would indicate:
 *   - a missing/wrong compute-to-compute memory barrier between the
 *     pressure and momentum sub-dispatches (momentum reads stale rho/p),
 *   - in-place buffer aliasing corrupting the read side of one of the
 *     sub-shaders (writes happening before reads within a thread),
 *   - a push-constant or descriptor-binding mistake (wrong slot index,
 *     wrong shader operating on wrong data).
 *
 * Verification setups:
 *
 *   1. Smooth pressure perturbation around hydrostatic. p = p0_base +
 *      Gaussian bump in (r, z). Both sub-shaders do non-trivial work,
 *      and any race or stale-read hazard between them shows up as a
 *      delta in u / v / w (which depend on the post-pressure rho / p).
 *
 *   2. Hydrostatic state. The fused path must also preserve the
 *      hydrostatic gate (|w_out| < 1e-6 m/s); equivalently, fused
 *      and sequential w_out match within the same float-precision
 *      tolerance the single-substep tests already use.
 *
 *   3. Multi-step composition. Run N=4 fused substeps vs N=4 sequential
 *      pairs. A single-step parity bug might mask if the small drift
 *      cancels by symmetry; iterating amplifies any divergence.
 */

#include "catch2/catch.hpp"

#include "compute/compute_backend.hpp"
#include "core/coordinate_system.hpp"
#include "core/field3d.hpp"
#include "core/grid_geometry.hpp"
#include "core/runtime_config.hpp"
#include "core/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kGamma = 1.4;
constexpr double kTropoScaleHeight = 8000.0;
constexpr double kSurfacePressure = 100000.0;
constexpr double kSurfaceDensity = 1.0;

void setup_cgrid_grid(int nr, int nth, int nz, double dr_m, double dz_m, double dt_s)
{
    NR     = nr;
    NTH    = nth;
    NZ     = nz;
    dr     = dr_m;
    dz     = dz_m;
    dt     = dt_s;
    dtheta = 2.0 * kPi / static_cast<double>(nth);

    global_coordinate_system = CoordinateSystem::Cylindrical;
    global_stagger_type      = StaggerType::CGrid;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    global_coordinate_system,
                                    global_stagger_type);
}

bool try_init_gpu()
{
    std::string error;
    global_compute_backend_config.backend = "vulkan";
    global_compute_backend_config.allow_fallback = false;
    bool ok = initialize_compute_backend_runtime(error);
    if (!ok)
    {
        global_compute_backend_config.backend = "cpu";
        global_compute_backend_config.allow_fallback = true;
        initialize_compute_backend_runtime(error);
        return false;
    }
    return true;
}

void shutdown_backend()
{
    shutdown_compute_backend_runtime();
}

void install_hydrostatic_p0_base()
{
    p0_base.resize(static_cast<std::size_t>(NZ));
    for (int k = 0; k < NZ; ++k)
    {
        const double z = (static_cast<double>(k) + 0.5) * dz;
        p0_base[static_cast<std::size_t>(k)] =
            kSurfacePressure * std::exp(-z / kTropoScaleHeight);
    }
}

std::vector<float> p0_base_to_floats()
{
    std::vector<float> result(static_cast<std::size_t>(NZ), 0.0f);
    for (int k = 0; k < NZ; ++k)
    {
        result[static_cast<std::size_t>(k)] =
            static_cast<float>(p0_base[static_cast<std::size_t>(k)]);
    }
    return result;
}

struct AcousticState
{
    Field3D u, v, w, rho, p;
    AcousticState(int nr, int nth, int nz)
        : u(nr, nth, nz, 0.0f),
          v(nr, nth, nz, 0.0f),
          w(nr, nth, nz, 0.0f),
          rho(nr, nth, nz, static_cast<float>(kSurfaceDensity)),
          p(nr, nth, nz)
    {}
};

void install_perturbation_state(AcousticState& s,
                                double i_center, double sigma_i,
                                double k_decay,
                                double p_amplitude,
                                double u_amplitude)
{
    for (int i = 0; i < NR; ++i)
    {
        const double di = (static_cast<double>(i) - i_center) / sigma_i;
        const double bump_r = std::exp(-0.5 * di * di);
        for (int k = 0; k < NZ; ++k)
        {
            const double dk = static_cast<double>(k) / k_decay;
            const double bump_z = std::exp(-dk * dk);
            const double bump = bump_r * bump_z;
            const float p0_k = static_cast<float>(
                p0_base[static_cast<std::size_t>(k)]);
            const float p_pert = static_cast<float>(p_amplitude * bump);
            const float u_amp  = static_cast<float>(u_amplitude * bump);
            for (int j = 0; j < NTH; ++j)
            {
                s.p(i, j, k) = p0_k + p_pert;
                // Non-zero u so that divergence is non-trivial and
                // the pressure substep produces visible rho / p
                // changes; otherwise (u = v = w = 0) div = 0 and the
                // pressure substep is an identity. The momentum
                // substep is still exercised by the p_pert gradient.
                s.u(i, j, k) = u_amp;
            }
        }
    }
}

void install_hydrostatic_only_state(AcousticState& s)
{
    for (int k = 0; k < NZ; ++k)
    {
        const float p0_k = static_cast<float>(
            p0_base[static_cast<std::size_t>(k)]);
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                s.p(i, j, k) = p0_k;
    }
}

bool run_sequential(ComputeBackend* backend,
                    AcousticState& s,
                    const std::vector<float>& p0f,
                    float dr_f, float dtheta_f, float dz_f,
                    float gamma_f, float dt_f,
                    float rho_floor, float p_floor)
{
    // C.9.4: pressure substep, in-place on rho, p.
    const bool pressure_ok = backend->dispatch_acoustic_pressure_cgrid(
        s.u.data(), s.v.data(), s.w.data(),
        s.rho.data(), s.p.data(),
        s.rho.data(), s.p.data(),
        NR, NTH, NZ, dr_f, dtheta_f, dz_f,
        gamma_f, dt_f, rho_floor, p_floor);
    if (!pressure_ok) return false;

    // C.9.5: momentum substep, in-place on u, v, w.
    const bool momentum_ok = backend->dispatch_acoustic_momentum_cgrid(
        s.rho.data(), s.p.data(),
        p0f.data(), NZ,
        s.u.data(), s.v.data(), s.w.data(),
        s.u.data(), s.v.data(), s.w.data(),
        NR, NTH, NZ, dr_f, dtheta_f, dz_f,
        dt_f, wind_horizontal_abs_max_ms, wind_vertical_abs_max_ms);
    return momentum_ok;
}

bool run_fused(ComputeBackend* backend,
               AcousticState& s,
               const std::vector<float>& p0f,
               float dr_f, float dtheta_f, float dz_f,
               float gamma_f, float dt_f,
               float rho_floor, float p_floor)
{
    return backend->dispatch_acoustic_substep_fused_cgrid(
        s.u.data(), s.v.data(), s.w.data(),
        s.rho.data(), s.p.data(),
        p0f.data(), NZ,
        NR, NTH, NZ, dr_f, dtheta_f, dz_f,
        gamma_f, dt_f, rho_floor, p_floor,
        wind_horizontal_abs_max_ms, wind_vertical_abs_max_ms);
}

struct FieldParity
{
    double max_abs = 0.0;
    int worst_i = -1, worst_j = -1, worst_k = -1;
    float worst_seq = 0.0f, worst_fused = 0.0f;
};

FieldParity compare_all(const Field3D& seq, const Field3D& fused)
{
    FieldParity err;
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const double diff = std::fabs(static_cast<double>(seq(i, j, k)) -
                                              static_cast<double>(fused(i, j, k)));
                if (diff > err.max_abs)
                {
                    err.max_abs = diff;
                    err.worst_i = i;
                    err.worst_j = j;
                    err.worst_k = k;
                    err.worst_seq = seq(i, j, k);
                    err.worst_fused = fused(i, j, k);
                }
            }
        }
    }
    return err;
}

}  // namespace

TEST_CASE("C.9.6 GPU C-grid fused acoustic substep parity",
          "[vulkan][gpu_parity][cgrid][acoustic][fused]")
{
    setup_cgrid_grid(/*nr=*/16, /*nth=*/8, /*nz=*/16,
                     /*dr=*/100.0, /*dz=*/250.0, /*dt=*/1.0e-3);
    install_hydrostatic_p0_base();

    if (!try_init_gpu())
    {
        WARN("GPU unavailable -- skipping C-grid fused acoustic substep parity");
        return;
    }

    auto* backend = mutable_compute_backend();
    if (backend == nullptr || !backend->supports_acoustic_substep_fused_cgrid_dispatch())
    {
        WARN("GPU does not support C-grid fused acoustic substep -- skipping");
        shutdown_backend();
        return;
    }

    const float dr_f      = static_cast<float>(dr);
    const float dtheta_f  = static_cast<float>(dtheta);
    const float dz_f      = static_cast<float>(dz);
    const float dt_f      = static_cast<float>(dt);
    const float gamma_f   = static_cast<float>(kGamma);
    const float rho_floor = density_min_kgm3;
    const float p_floor   = pressure_min_pa;

    SECTION("Smooth pressure perturbation: fused matches sequential bit-exact")
    {
        // Two independent state copies. Run sequential on one, fused on
        // the other, then compare cell-by-cell. The shaders are
        // deterministic and fed the same float inputs, so we expect
        // bit-exact agreement.
        AcousticState seq(NR, NTH, NZ);
        AcousticState fused(NR, NTH, NZ);

        install_perturbation_state(seq,   /*i_center=*/8.0, /*sigma_i=*/3.0,
                                          /*k_decay=*/6.0, /*p_amplitude=*/500.0,
                                          /*u_amplitude=*/5.0);
        install_perturbation_state(fused, /*i_center=*/8.0, /*sigma_i=*/3.0,
                                          /*k_decay=*/6.0, /*p_amplitude=*/500.0,
                                          /*u_amplitude=*/5.0);

        // Snapshot inputs for the motion-sanity assertion.
        Field3D p_input  = fused.p;
        Field3D u_input  = fused.u;

        const std::vector<float> p0f = p0_base_to_floats();

        REQUIRE(run_sequential(backend, seq,   p0f, dr_f, dtheta_f, dz_f,
                               gamma_f, dt_f, rho_floor, p_floor));
        REQUIRE(run_fused     (backend, fused, p0f, dr_f, dtheta_f, dz_f,
                               gamma_f, dt_f, rho_floor, p_floor));

        // Sanity: kernel actually moved the state (separate from the
        // fused-vs-sequential parity check, which would pass even if
        // both paths were silent no-ops).
        double max_motion_p = 0.0;
        double max_motion_u = 0.0;
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    max_motion_p = std::max(max_motion_p,
                        std::fabs(static_cast<double>(fused.p(i, j, k))
                                - static_cast<double>(p_input (i, j, k))));
                    max_motion_u = std::max(max_motion_u,
                        std::fabs(static_cast<double>(fused.u(i, j, k))
                                - static_cast<double>(u_input (i, j, k))));
                }
        INFO("kernel motion: max|delta_p|=" << max_motion_p
             << " max|delta_u|=" << max_motion_u);
        REQUIRE(max_motion_p > 0.5);
        REQUIRE(max_motion_u > 1.0e-4);

        // Parity: bit-exact agreement. Same shaders, same inputs, no
        // host round-trip in the fused path; outputs must match
        // exactly. Any drift indicates a barrier or aliasing bug.
        const auto u_par   = compare_all(seq.u,   fused.u);
        const auto v_par   = compare_all(seq.v,   fused.v);
        const auto w_par   = compare_all(seq.w,   fused.w);
        const auto rho_par = compare_all(seq.rho, fused.rho);
        const auto p_par   = compare_all(seq.p,   fused.p);
        INFO("u: max_abs=" << u_par.max_abs
             << " at (" << u_par.worst_i << "," << u_par.worst_j
             << "," << u_par.worst_k << ") seq=" << u_par.worst_seq
             << " fused=" << u_par.worst_fused);
        INFO("v: max_abs=" << v_par.max_abs);
        INFO("w: max_abs=" << w_par.max_abs);
        INFO("rho: max_abs=" << rho_par.max_abs);
        INFO("p: max_abs=" << p_par.max_abs);
        REQUIRE(u_par.max_abs   == 0.0);
        REQUIRE(v_par.max_abs   == 0.0);
        REQUIRE(w_par.max_abs   == 0.0);
        REQUIRE(rho_par.max_abs == 0.0);
        REQUIRE(p_par.max_abs   == 0.0);

        shutdown_backend();
        REQUIRE(try_init_gpu());
    }

    SECTION("Hydrostatic state: fused preserves the |dw/dt| < 1e-3 m/s/s gate")
    {
        // p = p0_base everywhere, u = v = w = 0. The fused path runs
        // the same C.9.5 momentum shader internally as the
        // single-substep parity test, so it must hit the same
        // float-precision residual on w (~1e-8 m/s) and bit-exact zero
        // on u, v.
        AcousticState fused(NR, NTH, NZ);
        install_hydrostatic_only_state(fused);

        const std::vector<float> p0f = p0_base_to_floats();

        REQUIRE(run_fused(backend, fused, p0f, dr_f, dtheta_f, dz_f,
                          gamma_f, dt_f, rho_floor, p_floor));

        double max_u = 0.0, max_v = 0.0, max_w = 0.0;
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    max_u = std::max(max_u, std::fabs(static_cast<double>(fused.u(i, j, k))));
                    max_v = std::max(max_v, std::fabs(static_cast<double>(fused.v(i, j, k))));
                    max_w = std::max(max_w, std::fabs(static_cast<double>(fused.w(i, j, k))));
                }
        INFO("Hydrostatic fused: max|u|=" << max_u
             << " max|v|=" << max_v << " max|w|=" << max_w);
        REQUIRE(max_u == 0.0);
        REQUIRE(max_v == 0.0);
        // 1e-6 m/s corresponds to dw/dt < 1e-3 m/s/s, four orders
        // smaller than the missing-subtraction case (~1e-2 m/s).
        REQUIRE(max_w < 1.0e-6);

        shutdown_backend();
        REQUIRE(try_init_gpu());
    }

    SECTION("Multi-step composition: 4 fused substeps == 4 sequential pairs")
    {
        // Iterate to amplify any micro-divergence. A single-step bit-
        // exact test could pass if there's a barrier hazard whose
        // timing happens to match in single-shot dispatches but not
        // when chained.
        AcousticState seq(NR, NTH, NZ);
        AcousticState fused(NR, NTH, NZ);

        install_perturbation_state(seq,   8.0, 3.0, 6.0, 500.0, 5.0);
        install_perturbation_state(fused, 8.0, 3.0, 6.0, 500.0, 5.0);

        const std::vector<float> p0f = p0_base_to_floats();

        constexpr int kNumSubsteps = 4;
        for (int step = 0; step < kNumSubsteps; ++step)
        {
            REQUIRE(run_sequential(backend, seq,   p0f, dr_f, dtheta_f, dz_f,
                                   gamma_f, dt_f, rho_floor, p_floor));
            REQUIRE(run_fused     (backend, fused, p0f, dr_f, dtheta_f, dz_f,
                                   gamma_f, dt_f, rho_floor, p_floor));
        }

        const auto u_par   = compare_all(seq.u,   fused.u);
        const auto v_par   = compare_all(seq.v,   fused.v);
        const auto w_par   = compare_all(seq.w,   fused.w);
        const auto rho_par = compare_all(seq.rho, fused.rho);
        const auto p_par   = compare_all(seq.p,   fused.p);
        INFO("After " << kNumSubsteps << " substeps:");
        INFO("u: max_abs=" << u_par.max_abs);
        INFO("v: max_abs=" << v_par.max_abs);
        INFO("w: max_abs=" << w_par.max_abs);
        INFO("rho: max_abs=" << rho_par.max_abs);
        INFO("p: max_abs=" << p_par.max_abs);
        REQUIRE(u_par.max_abs   == 0.0);
        REQUIRE(v_par.max_abs   == 0.0);
        REQUIRE(w_par.max_abs   == 0.0);
        REQUIRE(rho_par.max_abs == 0.0);
        REQUIRE(p_par.max_abs   == 0.0);
    }

    shutdown_backend();
}
