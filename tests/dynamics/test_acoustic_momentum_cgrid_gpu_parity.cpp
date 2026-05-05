/**
 * @file test_acoustic_momentum_cgrid_gpu_parity.cpp
 * @brief CPU/GPU parity for the fused acoustic momentum substep on the
 *        cylindrical Arakawa C-grid (Phase C.9.5 of
 *        docs/CoordinateBackend_Plan.md).
 *
 * The C.5 CPU kernel SupercellCGridScheme::compute_fast_momentum_tendencies
 * computes per-face pressure gradients in DOUBLE precision and stores
 * the integrated u / v / w back as FLOAT, with a vertical
 * reference-state subtraction (dp/dz - dp0/dz) so that a hydrostatic
 * state generates zero dw/dt. The C.9.5 GPU shader
 * acoustic_momentum_cgrid.comp runs in FLOAT throughout. The parity
 * gate accommodates float-vs-double drift while still catching real
 * shader bugs (wrong stencil, missing reference-state subtraction,
 * mis-applied clamp, etc.).
 *
 * Verification setups:
 *
 *   1. Hydrostatic state. u = v = w = 0, p = p0_base[k] everywhere.
 *      Both dp/dz and dp0/dz are computed from the same float values,
 *      so (dp/dz - dp0/dz) is zero bit-exactly and dw/dt = 0. Without
 *      the reference-state subtraction the shader would generate
 *      dw/dt ~ -g ~ -10 m/s/s, so this test directly proves that the
 *      perturbation-form vertical pressure gradient is wired correctly.
 *      This is what makes split-explicit on a C-grid stable; the
 *      collocated dispatch_acoustic_momentum CANNOT pass this gate.
 *
 *   2. Smooth perturbation around hydrostatic. p = p0_base + small
 *      Gaussian dp(r). Per-face momentum tendencies are non-trivial
 *      and CPU and GPU must agree to within float-vs-double drift.
 *      Includes a sanity assertion that the kernel actually moved the
 *      velocity field above some absolute threshold, separate from
 *      the parity check, to catch the C.9.4-era class of bug where
 *      both CPU reference and GPU shader had the same algebra mistake
 *      and "agreed" on a wrong answer.
 *
 *   3. Boundary passthrough. The CPU C-grid kernel pre-zeros the
 *      tendency arrays for ALL cells and only writes the per-face
 *      valid range; the integration runs over ALL cells with
 *      tendency = 0 outside that range, leaving boundary u/v/w equal
 *      to their input values (modulo the magnitude clamp, which is
 *      identity for in-range inputs). The GPU shader mirrors this:
 *      out-of-range cells must take the passthrough path, NOT the
 *      collocated-style antisymmetric / zero-gradient extrapolation.
 *
 *   4. Structural no-op. Constant p AND constant u/v/w. dp gradients
 *      are zero, so all face tendencies are zero and the output equals
 *      the input bit-exactly (modulo magnitude clamp). Catches any
 *      lurking write to the wrong cell, mis-zeroed tendency, etc.
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
constexpr double kGravity = 9.81;
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

// Hydrostatic reference profile: simple isothermal exponential.
//
// p0(z) = p_sl * exp(-z / H)
//
// is a valid hydrostatic solution for an isothermal atmosphere with
// rho(z) = p0(z) / (R * T). For the parity test we only need that
// p0_base[k] varies smoothly with k; the actual atmosphere physics is
// irrelevant since the shader operates purely on the discretized
// pressure-gradient + reference-state subtraction.
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

// Inline replica of the C-grid CPU production path:
//
//   SupercellCGridScheme::compute_fast_momentum_tendencies +
//   the apply_fast_momentum integration in dynamics.cpp.
//
// Inner arithmetic in double; storage in float. The boundary-cell
// passthrough is implicit in the per-face valid range loops (CPU
// pre-zeros all tendencies, then writes only the interior face range,
// then integrates every cell with the magnitude clamp).
void cpu_reference_acoustic_momentum_cgrid(
    const Field3D& rho_in, const Field3D& p_in,
    const Field3D& u_in, const Field3D& v_in, const Field3D& w_in,
    Field3D& u_out, Field3D& v_out, Field3D& w_out,
    double dr_v, double dtheta_v, double dz_v,
    double dt_small)
{
    u_out.resize(NR, NTH, NZ, 0.0f);
    v_out.resize(NR, NTH, NZ, 0.0f);
    w_out.resize(NR, NTH, NZ, 0.0f);

    Field3D du(NR, NTH, NZ, 0.0f);
    Field3D dv(NR, NTH, NZ, 0.0f);
    Field3D dw(NR, NTH, NZ, 0.0f);

    // du/dt at r-face: i in [0, NR-2], j periodic, k in [1, NZ-2].
    for (int i = 0; i <= NR - 2; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 1; k <= NZ - 2; ++k)
            {
                const double dp_dr =
                    (static_cast<double>(p_in(i + 1, j, k)) -
                     static_cast<double>(p_in(i,     j, k))) / dr_v;
                const double rho_face = 0.5 *
                    (static_cast<double>(rho_in(i,     j, k)) +
                     static_cast<double>(rho_in(i + 1, j, k)));
                const double rho_safe =
                    (std::isfinite(rho_face) && rho_face > 1.0e-6) ? rho_face : 1.0;
                double tend = -dp_dr / rho_safe;
                if (!std::isfinite(tend)) tend = 0.0;
                du(i, j, k) = static_cast<float>(tend);
            }
        }
    }

    // dv/dt at theta-face: i in [1, NR-2], j periodic, k in [1, NZ-2].
    for (int i = 1; i <= NR - 2; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const int j_next = (j + 1) % NTH;
            const double r_inv_i = 1.0 / (static_cast<double>(i) * dr_v);
            for (int k = 1; k <= NZ - 2; ++k)
            {
                const double dp_dth_over_r =
                    (static_cast<double>(p_in(i, j_next, k)) -
                     static_cast<double>(p_in(i, j,      k))) / dtheta_v * r_inv_i;
                const double rho_face = 0.5 *
                    (static_cast<double>(rho_in(i, j,      k)) +
                     static_cast<double>(rho_in(i, j_next, k)));
                const double rho_safe =
                    (std::isfinite(rho_face) && rho_face > 1.0e-6) ? rho_face : 1.0;
                double tend = -dp_dth_over_r / rho_safe;
                if (!std::isfinite(tend)) tend = 0.0;
                dv(i, j, k) = static_cast<float>(tend);
            }
        }
    }

    // dw/dt at z-face: i in [1, NR-2], j periodic, k in [0, NZ-2].
    // Reference-state subtraction in the vertical (perturbation form).
    for (int i = 1; i <= NR - 2; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k <= NZ - 2; ++k)
            {
                const double dp_dz =
                    (static_cast<double>(p_in(i, j, k + 1)) -
                     static_cast<double>(p_in(i, j, k    ))) / dz_v;
                const double dp0_dz =
                    (p0_base[static_cast<std::size_t>(k + 1)] -
                     p0_base[static_cast<std::size_t>(k    )]) / dz_v;
                const double dp_prime_dz = dp_dz - dp0_dz;
                const double rho_face = 0.5 *
                    (static_cast<double>(rho_in(i, j, k    )) +
                     static_cast<double>(rho_in(i, j, k + 1)));
                const double rho_safe =
                    (std::isfinite(rho_face) && rho_face > 1.0e-6) ? rho_face : 1.0;
                double tend = -dp_prime_dz / rho_safe;
                if (!std::isfinite(tend)) tend = 0.0;
                dw(i, j, k) = static_cast<float>(tend);
            }
        }
    }

    // Forward-Euler integrate every cell with magnitude clamp; mirrors
    // the apply_fast_momentum integration in dynamics.cpp.
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                double du_d = static_cast<double>(du(i, j, k));
                double dv_d = static_cast<double>(dv(i, j, k));
                double dw_d = static_cast<double>(dw(i, j, k));
                if (!std::isfinite(du_d)) du_d = 0.0;
                if (!std::isfinite(dv_d)) dv_d = 0.0;
                if (!std::isfinite(dw_d)) dw_d = 0.0;

                u_out(i, j, k) = clamp_wind_horizontal_ms(static_cast<float>(
                    static_cast<double>(u_in(i, j, k)) + du_d * dt_small));
                v_out(i, j, k) = clamp_wind_horizontal_ms(static_cast<float>(
                    static_cast<double>(v_in(i, j, k)) + dv_d * dt_small));
                w_out(i, j, k) = clamp_wind_vertical_ms(static_cast<float>(
                    static_cast<double>(w_in(i, j, k)) + dw_d * dt_small));
            }
        }
    }
}

struct FieldParity
{
    double max_abs = 0.0;
    int worst_i = -1, worst_j = -1, worst_k = -1;
    float worst_cpu = 0.0f, worst_gpu = 0.0f;
};

FieldParity compare_all(const Field3D& cpu, const Field3D& gpu)
{
    FieldParity err;
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const double diff = std::fabs(static_cast<double>(cpu(i, j, k)) -
                                              static_cast<double>(gpu(i, j, k)));
                if (diff > err.max_abs)
                {
                    err.max_abs = diff;
                    err.worst_i = i;
                    err.worst_j = j;
                    err.worst_k = k;
                    err.worst_cpu = cpu(i, j, k);
                    err.worst_gpu = gpu(i, j, k);
                }
            }
        }
    }
    return err;
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

}  // namespace

TEST_CASE("C.9.5 GPU C-grid acoustic momentum parity",
          "[vulkan][gpu_parity][cgrid][acoustic]")
{
    // Acoustic substep timescale: dt_small = 1e-3 s is typical for
    // Klemp-Wilhelmson with cs ~ 350 m/s and dr = 100 m
    // (CFL_acoustic ~ cs * dt_small / dr = 0.0035, well under 1).
    setup_cgrid_grid(/*nr=*/16, /*nth=*/8, /*nz=*/16,
                     /*dr=*/100.0, /*dz=*/250.0, /*dt=*/1.0e-3);
    install_hydrostatic_p0_base();

    if (!try_init_gpu())
    {
        WARN("GPU unavailable -- skipping C-grid acoustic momentum parity");
        return;
    }

    auto* backend = mutable_compute_backend();
    if (backend == nullptr || !backend->supports_acoustic_momentum_cgrid_dispatch())
    {
        WARN("GPU does not support C-grid acoustic momentum -- skipping");
        shutdown_backend();
        return;
    }

    const float dr_f      = static_cast<float>(dr);
    const float dtheta_f  = static_cast<float>(dtheta);
    const float dz_f      = static_cast<float>(dz);
    const float dt_f      = static_cast<float>(dt);

    SECTION("Hydrostatic state: zero dw/dt via reference-state subtraction")
    {
        // u = v = w = 0 everywhere. p = p0_base[k] (no perturbation).
        // The shader computes dp_dz from p_in and dp0_dz from p0_base_b
        // separately; with p_in == p0_base broadcast (bit-identical
        // float storage), (dp/dz - dp0/dz) cancels exactly and
        // dw/dt = 0. du/dt and dv/dt are also zero because p has no
        // horizontal variation. Output u / v / w must equal the
        // (zero) input bit-exactly.
        u.resize(NR, NTH, NZ, 0.0f);
        v.resize(NR, NTH, NZ, 0.0f);
        w.resize(NR, NTH, NZ, 0.0f);

        Field3D rho_field(NR, NTH, NZ, static_cast<float>(kSurfaceDensity));
        Field3D p_field(NR, NTH, NZ);
        for (int k = 0; k < NZ; ++k)
        {
            const float p0_k = static_cast<float>(p0_base[static_cast<std::size_t>(k)]);
            for (int i = 0; i < NR; ++i)
                for (int j = 0; j < NTH; ++j)
                    p_field(i, j, k) = p0_k;
        }

        Field3D u_gpu(NR, NTH, NZ, 0.0f);
        Field3D v_gpu(NR, NTH, NZ, 0.0f);
        Field3D w_gpu(NR, NTH, NZ, 0.0f);

        const std::vector<float> p0f = p0_base_to_floats();

        const bool dispatched = backend->dispatch_acoustic_momentum_cgrid(
            rho_field.data(), p_field.data(),
            p0f.data(), NZ,
            u.data(), v.data(), w.data(),
            u_gpu.data(), v_gpu.data(), w_gpu.data(),
            NR, NTH, NZ, dr_f, dtheta_f, dz_f,
            dt_f, wind_horizontal_abs_max_ms, wind_vertical_abs_max_ms);
        REQUIRE(dispatched);

        const auto u_par = compare_all(u, u_gpu);
        const auto v_par = compare_all(v, v_gpu);
        const auto w_par = compare_all(w, w_gpu);
        INFO("Hydrostatic u: max_abs=" << u_par.max_abs);
        INFO("Hydrostatic v: max_abs=" << v_par.max_abs);
        INFO("Hydrostatic w: max_abs=" << w_par.max_abs
             << " at (" << w_par.worst_i << "," << w_par.worst_j
             << "," << w_par.worst_k
             << ") cpu=" << w_par.worst_cpu << " gpu=" << w_par.worst_gpu);
        // u and v have NO cancellation (p has no horizontal variation),
        // so dp/dr and dp/dth are bit-exact zero: u_out = u_in, v_out
        // = v_in. We require bit-exact zero here.
        REQUIRE(u_par.max_abs == 0.0);
        REQUIRE(v_par.max_abs == 0.0);
        // w involves the (dp/dz - dp0/dz) cancellation. The CPU does
        // it in double and gets exactly 0; the GPU does it in float
        // and gets a residual at the level of float ULP for inputs
        // near 1e5 Pa (a few 1e-8 m/s after one substep). Bit-exact
        // zero is too strict for IEEE-754-permitted GPU float
        // arithmetic (FMA contraction, etc.); the realistic gate is
        // "much smaller than what a missing subtraction would
        // produce". Without subtraction, |dw/dt| ~ g ~ 10 m/s/s, so
        // |w_out| ~ 1e-2 m/s after dt = 1e-3 s. We require
        // |w_out| < 1e-6 m/s -- four orders of magnitude smaller,
        // conclusively proving the subtraction is operating.
        REQUIRE(w_par.max_abs < 1.0e-6);

        shutdown_backend();
        REQUIRE(try_init_gpu());
    }

    SECTION("Smooth radial pressure perturbation around hydrostatic")
    {
        // p = p0_base + Gaussian bump in r-direction (peak +500 Pa).
        // u, v, w start at zero. Per-face tendencies are non-trivial
        // (du/dt has the dp/dr horizontal gradient; dw/dt picks up the
        // (dp/dz - dp0/dz) vertical gradient of the bump's z-decay).
        const double i_center = 8.0;
        const double sigma_i  = 3.0;
        const double k_decay  = 6.0;

        u.resize(NR, NTH, NZ, 0.0f);
        v.resize(NR, NTH, NZ, 0.0f);
        w.resize(NR, NTH, NZ, 0.0f);

        Field3D rho_field(NR, NTH, NZ, static_cast<float>(kSurfaceDensity));
        Field3D p_field(NR, NTH, NZ);

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
                const float p_pert = static_cast<float>(500.0 * bump);
                for (int j = 0; j < NTH; ++j)
                {
                    p_field(i, j, k) = p0_k + p_pert;
                }
            }
        }

        Field3D u_gpu(NR, NTH, NZ, 0.0f);
        Field3D v_gpu(NR, NTH, NZ, 0.0f);
        Field3D w_gpu(NR, NTH, NZ, 0.0f);

        const std::vector<float> p0f = p0_base_to_floats();

        const bool dispatched = backend->dispatch_acoustic_momentum_cgrid(
            rho_field.data(), p_field.data(),
            p0f.data(), NZ,
            u.data(), v.data(), w.data(),
            u_gpu.data(), v_gpu.data(), w_gpu.data(),
            NR, NTH, NZ, dr_f, dtheta_f, dz_f,
            dt_f, wind_horizontal_abs_max_ms, wind_vertical_abs_max_ms);
        REQUIRE(dispatched);

        Field3D u_cpu(NR, NTH, NZ, 0.0f);
        Field3D v_cpu(NR, NTH, NZ, 0.0f);
        Field3D w_cpu(NR, NTH, NZ, 0.0f);
        cpu_reference_acoustic_momentum_cgrid(
            rho_field, p_field, u, v, w,
            u_cpu, v_cpu, w_cpu,
            dr, dtheta, dz, dt);

        // Sanity: kernel actually moved the velocity field. With dp ~ 500 Pa
        // peak over dr = 100 m, du/dr ~ 5 Pa/m, du/dt ~ -5 m/s^2,
        // dt = 1e-3 s -> delta_u ~ 5e-3 m/s. The peak excursion will be
        // somewhat larger because of the 1/r factor in dv and the
        // exponential profile, but 1e-3 m/s is a conservative threshold.
        double max_motion_u = 0.0;
        double max_motion_w = 0.0;
        for (int i = 0; i <= NR - 2; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 1; k <= NZ - 2; ++k)
                    max_motion_u = std::max(max_motion_u,
                        std::fabs(static_cast<double>(u_gpu(i, j, k))));
        for (int i = 1; i <= NR - 2; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k <= NZ - 2; ++k)
                    max_motion_w = std::max(max_motion_w,
                        std::fabs(static_cast<double>(w_gpu(i, j, k))));
        INFO("kernel motion: max|u_gpu| = " << max_motion_u
             << "  max|w_gpu| = " << max_motion_w);
        REQUIRE(max_motion_u > 1.0e-4);
        REQUIRE(max_motion_w > 1.0e-4);

        // Parity gate: 1e-4 m/s absolute. The integrated u/v/w live near
        // 1e-3 m/s after one substep with the perturbation amplitudes
        // chosen here. Float-vs-double drift at 1e-3 m/s is well under
        // 1e-5 m/s; 1e-4 leaves an order of magnitude headroom.
        const auto u_par = compare_all(u_cpu, u_gpu);
        const auto v_par = compare_all(v_cpu, v_gpu);
        const auto w_par = compare_all(w_cpu, w_gpu);
        INFO("Smooth perturb u: max_abs=" << u_par.max_abs
             << " at (" << u_par.worst_i << "," << u_par.worst_j
             << "," << u_par.worst_k
             << ") cpu=" << u_par.worst_cpu << " gpu=" << u_par.worst_gpu);
        INFO("Smooth perturb v: max_abs=" << v_par.max_abs
             << " at (" << v_par.worst_i << "," << v_par.worst_j
             << "," << v_par.worst_k
             << ") cpu=" << v_par.worst_cpu << " gpu=" << v_par.worst_gpu);
        INFO("Smooth perturb w: max_abs=" << w_par.max_abs
             << " at (" << w_par.worst_i << "," << w_par.worst_j
             << "," << w_par.worst_k
             << ") cpu=" << w_par.worst_cpu << " gpu=" << w_par.worst_gpu);
        REQUIRE(u_par.max_abs < 1.0e-4);
        REQUIRE(v_par.max_abs < 1.0e-4);
        REQUIRE(w_par.max_abs < 1.0e-4);

        shutdown_backend();
        REQUIRE(try_init_gpu());
    }

    SECTION("Boundary cells outside per-face range pass through inputs")
    {
        // Set distinctive sentinel velocities at boundary cells, with
        // a non-trivial pressure perturbation in the interior. The
        // shader must NOT modify boundary u/v/w (the per-face
        // is_active gate must zero those tendencies; clamps must be
        // identity for in-range sentinel values).
        const float u_sentinel_lo = -3.5f;
        const float u_sentinel_hi =  4.5f;
        const float v_sentinel_lo = -2.5f;
        const float v_sentinel_hi =  2.0f;
        const float w_sentinel_lo = -1.5f;
        const float w_sentinel_hi =  1.0f;

        u.resize(NR, NTH, NZ, 0.0f);
        v.resize(NR, NTH, NZ, 0.0f);
        w.resize(NR, NTH, NZ, 0.0f);

        // Stamp sentinels onto u-face-out-of-range cells (i = NR - 1
        // OR k in {0, NZ-1}). For u, the valid range is i in [0,
        // NR-2] AND k in [1, NZ-2], so anything outside that is a
        // sentinel slot.
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                u(NR - 1, j, k) = (k % 2 == 0) ? u_sentinel_lo : u_sentinel_hi;
            }
        }
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                u(i, j, 0)      = u_sentinel_lo;
                u(i, j, NZ - 1) = u_sentinel_hi;
                v(i, j, 0)      = v_sentinel_lo;
                v(i, j, NZ - 1) = v_sentinel_hi;
            }
        }
        // v boundary: i in {0, NR-1} OR k in {0, NZ-1}.
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                v(0,      j, k) = v_sentinel_lo;
                v(NR - 1, j, k) = v_sentinel_hi;
            }
        }
        // w boundary: i in {0, NR-1} OR k = NZ-1.
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                w(0,      j, k) = w_sentinel_lo;
                w(NR - 1, j, k) = w_sentinel_hi;
            }
            for (int i = 0; i < NR; ++i)
            {
                w(i, j, NZ - 1) = w_sentinel_hi;
            }
        }

        // Snapshot inputs for boundary comparison.
        Field3D u_input = u;
        Field3D v_input = v;
        Field3D w_input = w;

        // Modest interior pressure perturbation so the shader does
        // meaningful work in the valid face range.
        Field3D rho_field(NR, NTH, NZ, static_cast<float>(kSurfaceDensity));
        Field3D p_field(NR, NTH, NZ);
        for (int i = 0; i < NR; ++i)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const float p0_k = static_cast<float>(
                    p0_base[static_cast<std::size_t>(k)]);
                const float pert = (i >= 4 && i <= 11 && k >= 4 && k <= 11)
                    ? 200.0f : 0.0f;
                for (int j = 0; j < NTH; ++j)
                    p_field(i, j, k) = p0_k + pert;
            }
        }

        Field3D u_gpu(NR, NTH, NZ, 0.0f);
        Field3D v_gpu(NR, NTH, NZ, 0.0f);
        Field3D w_gpu(NR, NTH, NZ, 0.0f);

        const std::vector<float> p0f = p0_base_to_floats();

        const bool dispatched = backend->dispatch_acoustic_momentum_cgrid(
            rho_field.data(), p_field.data(),
            p0f.data(), NZ,
            u.data(), v.data(), w.data(),
            u_gpu.data(), v_gpu.data(), w_gpu.data(),
            NR, NTH, NZ, dr_f, dtheta_f, dz_f,
            dt_f, wind_horizontal_abs_max_ms, wind_vertical_abs_max_ms);
        REQUIRE(dispatched);

        // Per-face out-of-range cells must equal input bit-exactly.
        double max_u_boundary = 0.0;
        double max_v_boundary = 0.0;
        double max_w_boundary = 0.0;
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const bool u_out_of_range =
                        !(i <= NR - 2 && k >= 1 && k <= NZ - 2);
                    const bool v_out_of_range =
                        !(i >= 1 && i <= NR - 2 && k >= 1 && k <= NZ - 2);
                    const bool w_out_of_range =
                        !(i >= 1 && i <= NR - 2 && k <= NZ - 2);
                    if (u_out_of_range)
                    {
                        max_u_boundary = std::max(max_u_boundary,
                            std::fabs(static_cast<double>(u_gpu(i, j, k))
                                    - static_cast<double>(u_input(i, j, k))));
                    }
                    if (v_out_of_range)
                    {
                        max_v_boundary = std::max(max_v_boundary,
                            std::fabs(static_cast<double>(v_gpu(i, j, k))
                                    - static_cast<double>(v_input(i, j, k))));
                    }
                    if (w_out_of_range)
                    {
                        max_w_boundary = std::max(max_w_boundary,
                            std::fabs(static_cast<double>(w_gpu(i, j, k))
                                    - static_cast<double>(w_input(i, j, k))));
                    }
                }
        INFO("Boundary passthrough: u=" << max_u_boundary
             << " v=" << max_v_boundary
             << " w=" << max_w_boundary);
        REQUIRE(max_u_boundary == 0.0);
        REQUIRE(max_v_boundary == 0.0);
        REQUIRE(max_w_boundary == 0.0);

        shutdown_backend();
        REQUIRE(try_init_gpu());
    }

    SECTION("Constant pressure + constant velocity: structural identity")
    {
        // Constant p, constant non-zero u/v/w. dp gradients vanish on
        // every face, so all tendencies are zero. dw uses
        // (dp/dz - dp0/dz) which would be -dp0/dz != 0 if we set
        // p = const, but we set p = p0_base[k] (varying with k) so
        // dp/dz = dp0/dz cancels. Velocity output must equal input.
        const float u_const = 1.5f;
        const float v_const = -0.75f;
        const float w_const = 0.25f;

        u.resize(NR, NTH, NZ, u_const);
        v.resize(NR, NTH, NZ, v_const);
        w.resize(NR, NTH, NZ, w_const);

        Field3D rho_field(NR, NTH, NZ, static_cast<float>(kSurfaceDensity));
        Field3D p_field(NR, NTH, NZ);
        for (int k = 0; k < NZ; ++k)
        {
            const float p0_k = static_cast<float>(p0_base[static_cast<std::size_t>(k)]);
            for (int i = 0; i < NR; ++i)
                for (int j = 0; j < NTH; ++j)
                    p_field(i, j, k) = p0_k;
        }

        Field3D u_gpu(NR, NTH, NZ, 0.0f);
        Field3D v_gpu(NR, NTH, NZ, 0.0f);
        Field3D w_gpu(NR, NTH, NZ, 0.0f);

        const std::vector<float> p0f = p0_base_to_floats();

        const bool dispatched = backend->dispatch_acoustic_momentum_cgrid(
            rho_field.data(), p_field.data(),
            p0f.data(), NZ,
            u.data(), v.data(), w.data(),
            u_gpu.data(), v_gpu.data(), w_gpu.data(),
            NR, NTH, NZ, dr_f, dtheta_f, dz_f,
            dt_f, wind_horizontal_abs_max_ms, wind_vertical_abs_max_ms);
        REQUIRE(dispatched);

        const auto u_par = compare_all(u, u_gpu);
        const auto v_par = compare_all(v, v_gpu);
        const auto w_par = compare_all(w, w_gpu);
        INFO("Const-state u: max_abs=" << u_par.max_abs
             << "  v: max_abs=" << v_par.max_abs
             << "  w: max_abs=" << w_par.max_abs);
        // u and v: no cancellation, bit-exact identity.
        REQUIRE(u_par.max_abs == 0.0);
        REQUIRE(v_par.max_abs == 0.0);
        // w: same float-cancellation residual as the hydrostatic
        // SECTION (see comment there for the analysis).
        REQUIRE(w_par.max_abs < 1.0e-6);
    }

    shutdown_backend();
}
