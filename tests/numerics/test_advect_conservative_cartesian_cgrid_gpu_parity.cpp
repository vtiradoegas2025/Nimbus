/**
 * @file test_advect_conservative_cartesian_cgrid_gpu_parity.cpp
 * @brief CPU/GPU parity for the batched conservative WENO5/MC scalar
 *        transport on the Cartesian Arakawa C-grid (GPU.md
 *        Phase 3.2).
 *
 * The CPU reference is the production conservative scalar kernel (double
 * inner arithmetic, float storage), linked directly into this test. The GPU
 * runs mass-flux / WENO5 face-flux / divergence+recovery in one submission;
 * the former unweighted kappa-diffusion chain is tested as rejected.
 *
 * Sections:
 *   1. Free-stream: a spatially constant scalar under arbitrary winds stays
 *      constant to float round-off.
 *   2. Bump advection (kappa = 0) matches the CPU kernel.
 *   3. Nonzero transport-chained point diffusion is rejected.
 *   4. Multi-scalar batch: three different fields in one submission each
 *      match their own CPU reference (verifies the fixed-work-slot cycling
 *      and the shared mass flux).
 *   5. Determinism: identical dispatches produce bit-identical results.
 */

#include "catch2/catch.hpp"

#include "compute/compute_backend.hpp"
#include "compute/compute_kernel_template.hpp"
#include "compute/parity_check.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/field/field3d.hpp"
#include "core/infra/grid_geometry.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "numerics/advection/advection_cartesian.hpp"
#include "numerics/positive_scalar_budget_contract.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

void setup_cartesian_cgrid_grid(int nx, int ny, int nz,
                                double dx_m, double dy_m, double dz_m)
{
    NR     = nx;
    NTH    = ny;
    NZ     = nz;
    dr     = dx_m;   // Cartesian: dr maps to dx
    dtheta = dy_m;   // Cartesian: dtheta maps to dy (meters)
    dz     = dz_m;

    global_coordinate_system = CoordinateSystem::Cartesian;
    global_stagger_type      = StaggerType::CGrid;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    global_coordinate_system,
                                    global_stagger_type);
    global_lateral_bc_west = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_east = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_south = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_north = tmv::init::config::LateralBC::RigidSlip;
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
    global_domain_motion_mode =
        tmv::init::config::DomainMotionMode::Stationary;
    global_domain_frame = tmv::domain::HorizontalDomainFrame::stationary();
}

bool try_init_gpu()
{
    std::string error;
    global_compute_backend_config.backend = "vulkan";
    global_compute_backend_config.allow_fallback = false;
    bool ok = initialize_compute_backend_runtime(error);
    if (!ok)
    {
        const std::string device_error = error;
        global_compute_backend_config.backend = "cpu";
        global_compute_backend_config.allow_fallback = true;
        initialize_compute_backend_runtime(error);
        const char* required = std::getenv("NIMBUS_GPU_REQUIRED");
        if (required != nullptr && std::string(required) == "1")
        {
            throw std::runtime_error(
                "required physical Vulkan scalar qualification could not "
                "initialize a device: " + device_error);
        }
        return false;
    }
    return true;
}

void shutdown_backend()
{
    shutdown_compute_backend_runtime();
}

/// Fills the GLOBAL u/v/w/rho fields (the CPU kernel reads them directly)
/// with a stratified, gently sheared state.
void fill_global_state()
{
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    rho.resize(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double zf = (static_cast<double>(k) + 0.5) * dz;
                u(i, j, k) = static_cast<float>(4.0 + 0.002 * zf);
                v(i, j, k) = static_cast<float>(-2.5 + 0.001 * zf);
                w(i, j, k) = static_cast<float>(
                    0.4 * std::sin(0.5 * static_cast<double>(i))
                        * std::cos(0.4 * static_cast<double>(j)));
                rho(i, j, k) = static_cast<float>(1.2 * std::exp(-zf / 8500.0));
            }
}

Field3D gaussian_bump_field(double base, double amplitude,
                            double ci, double cj, double ck)
{
    Field3D q(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double di = (static_cast<double>(i) - ci) / 2.5;
                const double dj = (static_cast<double>(j) - cj) / 2.5;
                const double dk = (static_cast<double>(k) - ck) / 2.5;
                const double bump =
                    std::exp(-0.5 * (di * di + dj * dj + dk * dk));
                q(i, j, k) = static_cast<float>(base + amplitude * bump);
            }
    return q;
}

void require_parity(const char* field, const Field3D& cpu_f,
                    const Field3D& gpu_f, double abs_tolerance,
                    nimbus::parity::ComparisonDomain domain)
{
    const auto result = nimbus::parity::compare_field(
        cpu_f.data(), gpu_f.data(), NR, NTH, NZ,
        {0.0, abs_tolerance}, domain);
    INFO(field << ": domain=" << nimbus::parity::domain_name(domain)
         << " max_abs=" << result.max_abs
         << " at (" << result.worst_i << "," << result.worst_j << ","
         << result.worst_k << ") cpu=" << result.worst_cpu
         << " gpu=" << result.worst_gpu
         << " compared=" << result.cells_compared
         << " nonfinite=" << result.nonfinite_cells
         << " invalid="
         << (result.invalid_reason != nullptr ? result.invalid_reason : "none"));
    REQUIRE(result.passed);
}

/// Runs the batched GPU dispatch with the same derived constants the
/// production hook in advection.cpp uses.
bool run_gpu_batch(const float* const* q_in, float* const* q_out,
                   const float* dt_kappas, int n, double dt_step,
                   const int* pd_flags = nullptr,
                   const CartesianCGridTransportContext* context = nullptr,
                   ScalarBoundaryTransportReport* const*
                       boundary_reports = nullptr,
                   const ScalarMoistEnergyKind* energy_kinds = nullptr,
                   ScalarMoistEnergyTransportReportV1* const*
                       energy_reports = nullptr)
{
    static const std::vector<int> kNoPd(16, 0);
    if (pd_flags == nullptr) pd_flags = kNoPd.data();
    const double dx_d = std::max(dr, 1.0e-12);
    const double dy_d = std::max(dtheta, 1.0e-12);
    const double dz_d = std::max(dz, 1.0e-12);
    return dispatch_advect_scalars_conservative_cartesian_cgrid_backend(
        u.data(), v.data(), w.data(), rho.data(),
        q_in, q_out, dt_kappas, pd_flags, n,
        NR, NTH, NZ,
        static_cast<float>(1.0 / dx_d),
        static_cast<float>(1.0 / dy_d),
        static_cast<float>(1.0 / dz_d),
        static_cast<float>(1.0 / (dx_d * dx_d)),
        static_cast<float>(1.0 / (dy_d * dy_d)),
        static_cast<float>(1.0 / (dz_d * dz_d)),
        static_cast<float>(dt_step), density_min_kgm3,
        context != nullptr ? context->density_begin.data() : nullptr,
        context != nullptr ? context->mass_flux_x.data() : nullptr,
        context != nullptr ? context->mass_flux_y.data() : nullptr,
        context != nullptr ? context->mass_flux_z.data() : nullptr,
        context != nullptr,
        nullptr, nullptr, nullptr, 0.0,
        boundary_reports, energy_kinds, energy_reports,
        static_cast<float>(global_domain_frame.velocity_ms.u_ms),
        static_cast<float>(global_domain_frame.velocity_ms.v_ms));
}

/// CPU production reference for one scalar. Nonzero kappa is unavailable.
Field3D cpu_reference(const Field3D& q_src, double dt_step, double kappa)
{
    Field3D advected;
    advect_scalar_conservative_3d_cartesian(q_src, advected, dt_step);
    if (kappa > 0.0)
    {
        Field3D diffused;
        apply_diffusion_kernel_cartesian(advected, diffused, dt_step, kappa);
        return diffused;
    }
    return advected;
}

double interior_rho_q_sum(const Field3D& density, const Field3D& scalar)
{
    double sum = 0.0;
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 0; k < NZ; ++k)
                sum += static_cast<double>(density(i, j, k))
                     * static_cast<double>(scalar(i, j, k));
    return sum;
}

double fp32_operator_gamma(int operations)
{
    const double product =
        static_cast<double>(operations) * std::numeric_limits<float>::epsilon();
    REQUIRE(product < 1.0);
    return product / (1.0 - product);
}

void require_scaled_owner_parity(const char* quantity, double cpu_value,
                                 double gpu_value, double physical_scale)
{
    // This smooth, limiter-active qualification case has fewer than 256
    // sequential FP32 arithmetic/rounding sites from face reconstruction
    // through the committed scalar.  Compare against the standard gamma_n
    // forward-error envelope at the conserved inventory/reservoir scale,
    // never against a cancellation-prone net report value.
    const double tolerance =
        fp32_operator_gamma(256) * std::max(physical_scale, 1.0);
    const double difference = std::abs(gpu_value - cpu_value);
    INFO(quantity << ": cpu=" << cpu_value << " gpu=" << gpu_value
         << " difference=" << difference
         << " physical_scale=" << physical_scale
         << " tolerance=" << tolerance);
    REQUIRE(std::isfinite(cpu_value));
    REQUIRE(std::isfinite(gpu_value));
    REQUIRE(difference <= tolerance);
}

void require_scalar_report_identity(const char* owner,
                                    const ScalarBoundaryTransportReport& report)
{
    REQUIRE(report.complete);
    const double scale = std::max({
        1.0,
        std::abs(report.flux_implied_state_change),
        std::abs(report.committed_state_change),
        std::abs(report.closure_residual),
    });
    INFO(owner << " scalar identity residual=" << report.closure_residual
         << " scale=" << scale);
    REQUIRE(std::abs(
        report.committed_state_change
        - report.flux_implied_state_change
        - report.closure_residual)
        <= 32.0 * std::numeric_limits<double>::epsilon() * scale);
}

void require_energy_report_identity(
    const char* owner, const ScalarMoistEnergyTransportReportV1& report)
{
    REQUIRE(report.complete);
    const double scale = std::max({
        1.0,
        std::abs(report.density_history_remap_j),
        std::abs(report.limited_face_state_change_j),
        std::abs(report.storage_rounding_j),
        std::abs(report.committed_state_change_j),
    });
    INFO(owner << " moist-energy identity residual="
         << report.closure_residual_j << " scale=" << scale);
    REQUIRE(std::abs(
        report.committed_state_change_j
        - report.density_history_remap_j
        - report.limited_face_state_change_j
        - report.storage_rounding_j)
        <= 64.0 * std::numeric_limits<double>::epsilon() * scale);
    REQUIRE(std::abs(
        report.limited_face_state_change_j
        + report.outward_boundary_energy_j
        - report.interior_conversion_j)
        <= 64.0 * std::numeric_limits<double>::epsilon() * scale);
    REQUIRE(std::abs(
        report.requested_state_change_j
        - report.density_history_remap_j
        - report.limited_face_state_change_j)
        <= 64.0 * std::numeric_limits<double>::epsilon() * scale);
}

void require_scalar_owner_report_parity(
    const char* owner,
    const ScalarBoundaryTransportReport& cpu_boundary,
    const ScalarBoundaryTransportReport& gpu_boundary,
    const ScalarMoistEnergyTransportReportV1& cpu_energy,
    const ScalarMoistEnergyTransportReportV1& gpu_energy,
    double scalar_inventory_scale,
    double energy_reservoir_scale)
{
    require_scalar_report_identity(owner, cpu_boundary);
    require_scalar_report_identity(owner, gpu_boundary);
    require_energy_report_identity(owner, cpu_energy);
    require_energy_report_identity(owner, gpu_energy);

    const char* boundary_names[] = {
        "outward inventory", "flux-implied change",
        "committed change", "closure residual",
    };
    const double cpu_boundary_values[] = {
        cpu_boundary.outward_boundary_inventory,
        cpu_boundary.flux_implied_state_change,
        cpu_boundary.committed_state_change,
        cpu_boundary.closure_residual,
    };
    const double gpu_boundary_values[] = {
        gpu_boundary.outward_boundary_inventory,
        gpu_boundary.flux_implied_state_change,
        gpu_boundary.committed_state_change,
        gpu_boundary.closure_residual,
    };
    for (std::size_t n = 0; n < std::size(boundary_names); ++n)
    {
        INFO("scalar owner=" << owner);
        require_scaled_owner_parity(
            boundary_names[n], cpu_boundary_values[n],
            gpu_boundary_values[n], scalar_inventory_scale);
    }

    const char* energy_names[] = {
        "dry internal", "water sensible", "phase reference",
        "carrier kinetic", "gravitational", "subgrid TKE",
        "density remap", "outward boundary", "interior conversion",
        "limited-face change", "requested change", "storage rounding",
        "committed change", "closure residual",
    };
    const double cpu_energy_values[] = {
        cpu_energy.dry_internal_change_j,
        cpu_energy.water_sensible_change_j,
        cpu_energy.phase_reference_change_j,
        cpu_energy.carrier_kinetic_change_j,
        cpu_energy.gravitational_change_j,
        cpu_energy.subgrid_tke_change_j,
        cpu_energy.density_history_remap_j,
        cpu_energy.outward_boundary_energy_j,
        cpu_energy.interior_conversion_j,
        cpu_energy.limited_face_state_change_j,
        cpu_energy.requested_state_change_j,
        cpu_energy.storage_rounding_j,
        cpu_energy.committed_state_change_j,
        cpu_energy.closure_residual_j,
    };
    const double gpu_energy_values[] = {
        gpu_energy.dry_internal_change_j,
        gpu_energy.water_sensible_change_j,
        gpu_energy.phase_reference_change_j,
        gpu_energy.carrier_kinetic_change_j,
        gpu_energy.gravitational_change_j,
        gpu_energy.subgrid_tke_change_j,
        gpu_energy.density_history_remap_j,
        gpu_energy.outward_boundary_energy_j,
        gpu_energy.interior_conversion_j,
        gpu_energy.limited_face_state_change_j,
        gpu_energy.requested_state_change_j,
        gpu_energy.storage_rounding_j,
        gpu_energy.committed_state_change_j,
        gpu_energy.closure_residual_j,
    };
    for (std::size_t n = 0; n < std::size(energy_names); ++n)
    {
        INFO("scalar owner=" << owner);
        require_scaled_owner_parity(
            energy_names[n], cpu_energy_values[n], gpu_energy_values[n],
            energy_reservoir_scale);
    }
}

}  // namespace

TEST_CASE("Cartesian C-grid conservative transport GPU parity (Phase 3.2)",
          "[vulkan][gpu_parity][cgrid][advection][cartesian]")
{
    setup_cartesian_cgrid_grid(/*nx=*/18, /*ny=*/18, /*nz=*/18,
                               /*dx=*/500.0, /*dy=*/750.0, /*dz=*/250.0);
    const double dt_step = 2.0;

    if (!try_init_gpu())
    {
        WARN("GPU unavailable -- skipping conservative transport parity");
        return;
    }

    auto* backend = mutable_compute_backend();
    auto* kernels = backend != nullptr ? backend->advection_kernels() : nullptr;
    if (kernels == nullptr
        || !kernels->supports(AdvectionKernelOp::ConservativeCartesianCgridTransport))
    {
        const char* required = std::getenv("NIMBUS_GPU_REQUIRED");
        if (required != nullptr && std::string(required) == "1")
        {
            shutdown_backend();
            FAIL("required Vulkan device does not expose conservative "
                 "Cartesian C-grid scalar transport");
        }
        WARN("GPU does not support the conservative transport batch -- skipping");
        shutdown_backend();
        return;
    }

    fill_global_state();

    SECTION("free-stream: constant scalar stays constant")
    {
        Field3D q(NR, NTH, NZ, 1.0f);
        Field3D q_gpu = q;
        const float* q_in[1] = { q.data() };
        float* q_out[1] = { q_gpu.data() };
        const float dtk[1] = { 0.0f };
        REQUIRE(run_gpu_batch(q_in, q_out, dtk, 1, dt_step));

        require_parity("free-stream", q, q_gpu, 1.0e-6,
                       nimbus::parity::ComparisonDomain::FullField);
    }

    SECTION("bump advection (kappa = 0) matches the CPU production kernel")
    {
        Field3D q = gaussian_bump_field(0.5, 2.0, 9.0, 9.0, 9.0);
        Field3D q_gpu = q;
        const float* q_in[1] = { q.data() };
        float* q_out[1] = { q_gpu.data() };
        const float dtk[1] = { 0.0f };
        REQUIRE(run_gpu_batch(q_in, q_out, dtk, 1, dt_step));

        const Field3D q_cpu = cpu_reference(q, dt_step, 0.0);

        // Sanity: the kernel moved the bump (winds ~4 m/s over dt = 2 s).
        double max_motion = 0.0;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                    max_motion = std::max(max_motion,
                        std::fabs(static_cast<double>(q_gpu(i, j, k))
                                - static_cast<double>(q(i, j, k))));
        INFO("kernel made motion: max|q_gpu - q_in| = " << max_motion);
        REQUIRE(max_motion > 1.0e-3);

        require_parity("bump", q_cpu, q_gpu, 2.0e-5,
                       nimbus::parity::ComparisonDomain::FullField);
    }

    SECTION("standalone mass flux is invariant under constant frame translation")
    {
        Field3D q = gaussian_bump_field(0.5, 2.0, 9.0, 9.0, 9.0);
        Field3D stationary = q;
        const float* q_in[1] = { q.data() };
        float* stationary_out[1] = { stationary.data() };
        const float dtk[1] = { 0.0f };
        REQUIRE(run_gpu_batch(q_in, stationary_out, dtk, 1, dt_step));

        constexpr double frame_u = 17.25;
        constexpr double frame_v = -8.5;
        for (std::size_t index = 0; index < u.size(); ++index)
        {
            u.data()[index] += static_cast<float>(frame_u);
            v.data()[index] += static_cast<float>(frame_v);
        }
        global_domain_motion_mode =
            tmv::init::config::DomainMotionMode::ConstantVelocity;
        global_domain_frame.velocity_ms.u_ms = frame_u;
        global_domain_frame.velocity_ms.v_ms = frame_v;

        Field3D translated = q;
        float* translated_out[1] = { translated.data() };
        REQUIRE(run_gpu_batch(q_in, translated_out, dtk, 1, dt_step));

        require_parity("translated standalone mass flux", stationary,
                       translated, 2.0e-6,
                       nimbus::parity::ComparisonDomain::FullField);
    }

    SECTION("nonzero transport-chained point diffusion is rejected")
    {
        const double kappa = 0.01;
        Field3D q = gaussian_bump_field(0.5, 2.0, 9.0, 9.0, 9.0);
        Field3D q_gpu = q;
        const float* q_in[1] = { q.data() };
        float* q_out[1] = { q_gpu.data() };
        const float dtk[1] = { static_cast<float>(dt_step * kappa) };
        REQUIRE_FALSE(run_gpu_batch(q_in, q_out, dtk, 1, dt_step));
    }

    SECTION("multi-scalar batch: each scalar matches its own CPU reference")
    {
        Field3D qa = gaussian_bump_field(300.0, 2.0, 6.0, 9.0, 9.0);
        Field3D qb = gaussian_bump_field(0.008, 0.004, 12.0, 9.0, 6.0);
        Field3D qc_f = gaussian_bump_field(0.0, 0.002, 9.0, 6.0, 12.0);

        Field3D ga = qa, gb = qb, gc = qc_f;
        const float* q_in[3] = { qa.data(), qb.data(), qc_f.data() };
        float* q_out[3] = { ga.data(), gb.data(), gc.data() };
        const float dtk[3] = {0.0f, 0.0f, 0.0f};
        REQUIRE(run_gpu_batch(q_in, q_out, dtk, 3, dt_step));

        // The theta-like scalar carries O(300) values: the stencil-tier
        // relative band dominates (1e-4 * 300 = 3e-2); mixing-ratio-scale
        // scalars sit in the absolute band.
        constexpr auto kFull = nimbus::parity::ComparisonDomain::FullField;
        require_parity("batch theta-like",
                       cpu_reference(qa, dt_step, 0.0), ga, 3.0e-2, kFull);
        require_parity("batch qv-like",
                       cpu_reference(qb, dt_step, 0.0), gb, 2.0e-5, kFull);
        require_parity("batch hydrometeor-like",
                       cpu_reference(qc_f, dt_step, 0.0), gc, 2.0e-5, kFull);
    }

    SECTION("determinism: identical dispatches are bit-identical")
    {
        Field3D q = gaussian_bump_field(0.5, 2.0, 9.0, 9.0, 9.0);
        Field3D g1 = q, g2 = q;
        const float* q_in[1] = { q.data() };
        float* q_out1[1] = { g1.data() };
        float* q_out2[1] = { g2.data() };
        const float dtk[1] = {0.0f};
        REQUIRE(run_gpu_batch(q_in, q_out1, dtk, 1, dt_step));
        REQUIRE(run_gpu_batch(q_in, q_out2, dtk, 1, dt_step));
        require_parity("determinism", g1, g2, 0.0,
                       nimbus::parity::ComparisonDomain::FullField);
    }

    SECTION("coupled retained-rho mode preserves free stream and rho-q")
    {
        // Closed, divergent flow: density is advanced with one continuity
        // sample and then retained.  Scalar transport must consume that same
        // averaged face flux instead of rebuilding one from rho_end.
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double xi = static_cast<double>(i) / (NR - 1);
                    const double yj = static_cast<double>(j) / (NTH - 1);
                    const double zk = static_cast<double>(k) / (NZ - 1);
                    u(i, j, k) = (i >= 1 && i <= NR - 3)
                        ? static_cast<float>(12.0 * std::sin(2.0 * 3.141592653589793 * xi))
                        : 0.0f;
                    v(i, j, k) = (j >= 1 && j <= NTH - 3)
                        ? static_cast<float>(-9.0 * std::sin(2.0 * 3.141592653589793 * yj))
                        : 0.0f;
                    w(i, j, k) = (k <= NZ - 2)
                        ? static_cast<float>(2.0 * std::sin(3.141592653589793 * zk))
                        : 0.0f;
                    rho(i, j, k) = static_cast<float>(1.0 + 0.08 * zk);
                }

        const Field3D rho_begin = rho;
        CartesianCGridTransportContext context;
        context.begin(NR, NTH, NZ, dt_step);
        context.accumulate(rho, u, v, w, dt_step);
        context.complete();

        const double inv_dx = 1.0 / dr;
        const double inv_dy = 1.0 / dtheta;
        const double inv_dz = 1.0 / dz;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double mx = context.mass_flux_x(i, j, k)
                                    - context.mass_flux_x(i - 1, j, k);
                    const double my = context.mass_flux_y(i, j, k)
                                    - context.mass_flux_y(i, j - 1, k);
                    const double mz_top = (k <= NZ - 2)
                        ? context.mass_flux_z(i, j, k) : 0.0;
                    const double mz_bot = (k >= 1)
                        ? context.mass_flux_z(i, j, k - 1) : 0.0;
                    rho(i, j, k) = static_cast<float>(
                        static_cast<double>(rho_begin(i, j, k))
                        - dt_step * (mx * inv_dx + my * inv_dy
                                     + (mz_top - mz_bot) * inv_dz));
                }

        Field3D q = gaussian_bump_field(0.002, 0.006, 9.0, 9.0, 9.0);
        Field3D q_gpu = q;
        Field3D q_cpu;
        advect_scalar_conservative_3d_cartesian(
            q, q_cpu, dt_step, /*positive_definite=*/false, &context);
        const float* q_in[1] = { q.data() };
        float* q_out[1] = { q_gpu.data() };
        const float dtk[1] = { 0.0f };
        REQUIRE(run_gpu_batch(q_in, q_out, dtk, 1, dt_step, nullptr, &context));
        require_parity("coupled retained-rho bump", q_cpu, q_gpu, 2.0e-5,
                       nimbus::parity::ComparisonDomain::FullField);

        const double mass_before = interior_rho_q_sum(rho_begin, q);
        const double mass_after = interior_rho_q_sum(rho, q_gpu);
        INFO("coupled rho-q before=" << mass_before << " after=" << mass_after);
        REQUIRE(mass_after == Approx(mass_before).epsilon(3.0e-6));

        Field3D ones(NR, NTH, NZ, 0.004f);
        Field3D ones_gpu = ones;
        const float* one_in[1] = { ones.data() };
        float* one_out[1] = { ones_gpu.data() };
        REQUIRE(run_gpu_batch(one_in, one_out, dtk, 1, dt_step,
                              nullptr, &context));
        double max_constant_error = 0.0;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                    max_constant_error = std::max(
                        max_constant_error,
                        std::abs(static_cast<double>(ones_gpu(i, j, k)) - 0.004));
        INFO("coupled constant max error=" << max_constant_error);
        REQUIRE(max_constant_error < 2.0e-7);
    }

    SECTION("physical device scalar owner reports match the CPU operator at reservoir scale")
    {
        constexpr double report_dt = 0.25;
        global_top_bc = tmv::init::config::TopBC::Rigid;

        p.resize(NR, NTH, NZ, 0.0f);
        theta.resize(NR, NTH, NZ, 0.0f);
        qv.resize(NR, NTH, NZ, 0.0f);
        qc.resize(NR, NTH, NZ, 0.0f);
        qr.resize(NR, NTH, NZ, 0.0f);
        qi.resize(NR, NTH, NZ, 0.0f);
        qs.resize(NR, NTH, NZ, 0.0f);
        qg.resize(NR, NTH, NZ, 0.0f);
        qh.resize(NR, NTH, NZ, 0.0f);
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double x = static_cast<double>(i) / (NR - 1);
                    const double y = static_cast<double>(j) / (NTH - 1);
                    const double z = static_cast<double>(k) / (NZ - 1);
                    rho(i, j, k) = static_cast<float>(
                        1.12 - 0.18 * z + 0.015 * std::sin(5.0 * x + y));
                    p(i, j, k) = static_cast<float>(
                        94000.0 * std::exp(-z * (NZ * dz) / 8500.0));
                    theta(i, j, k) = static_cast<float>(
                        298.0 + 7.0 * z + 0.3 * std::sin(3.0 * x));
                    u(i, j, k) = (i >= 1 && i <= NR - 3)
                        ? static_cast<float>(8.0 + 1.5 * std::sin(4.0 * y))
                        : 0.0f;
                    v(i, j, k) = (j >= 1 && j <= NTH - 3)
                        ? static_cast<float>(-5.0 + 1.0 * std::cos(3.0 * x))
                        : 0.0f;
                    w(i, j, k) = static_cast<float>(
                        0.3 * std::sin(3.0 * x) * std::cos(2.0 * y));
                    if (k == NZ - 1)
                        w(i, j, k) = static_cast<float>(1.5 + 0.2 * x);

                    qv(i, j, k) = static_cast<float>(
                        0.003 + 0.006 * std::exp(
                            -8.0 * ((x - 0.48) * (x - 0.48)
                                   + (y - 0.55) * (y - 0.55)
                                   + (z - 0.72) * (z - 0.72))));
                }

        const Field3D density_begin = rho;
        const Field3D scalar_begin = qv;
        CartesianCGridTransportContext invalid_rigid_context;
        invalid_rigid_context.begin(NR, NTH, NZ, report_dt);
        invalid_rigid_context.accumulate(
            rho, u, v, w, static_cast<double>(static_cast<float>(report_dt)),
            nullptr, nullptr, tmv::AcousticMassFluxPrecision::NativeFloat);
        REQUIRE_THROWS_WITH(
            invalid_rigid_context.complete(),
            Catch::Contains("impermeable C-grid boundary"));

        // A production rigid boundary has already closed the face velocity
        // before the acoustic flux is retained. Use that admissible state for
        // the direct owner comparison after proving contradictory flux fails
        // closed.
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                w(i, j, NZ - 1) = 0.0f;
        CartesianCGridTransportContext context;
        context.begin(NR, NTH, NZ, report_dt);
        // The retained flux history uses the same FP32 time and face-product
        // currency that the production acoustic Vulkan path exports.
        context.accumulate(
            rho, u, v, w, static_cast<double>(static_cast<float>(report_dt)),
            nullptr, nullptr, tmv::AcousticMassFluxPrecision::NativeFloat);

        const double inv_dx = 1.0 / dr;
        const double inv_dy = 1.0 / dtheta;
        const double inv_dz = 1.0 / dz;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double mx =
                        static_cast<double>(context.mass_flux_x(i, j, k))
                        - context.mass_flux_x(i - 1, j, k);
                    const double my =
                        static_cast<double>(context.mass_flux_y(i, j, k))
                        - context.mass_flux_y(i, j - 1, k);
                    const double mz_top =
                        static_cast<double>(context.mass_flux_z(i, j, k));
                    const double mz_bottom = k > 0
                        ? static_cast<double>(
                              context.mass_flux_z(i, j, k - 1))
                        : 0.0;
                    rho(i, j, k) = static_cast<float>(
                        static_cast<double>(density_begin(i, j, k))
                        - report_dt
                            * (mx * inv_dx + my * inv_dy
                               + (mz_top - mz_bottom) * inv_dz));
                }
        context.complete();
        REQUIRE(context.validate(NR, NTH, NZ, report_dt));

        Field3D q_cpu = scalar_begin;
        ScalarBoundaryTransportReport cpu_boundary;
        ScalarMoistEnergyTransportReportV1 cpu_energy;
        advect_scalar_conservative_3d_cartesian(
            q_cpu, q_cpu, report_dt, /*positive_definite=*/true, &context,
            &cpu_boundary, ScalarMoistEnergyKind::WaterVapor, &cpu_energy);

        Field3D q_gpu = scalar_begin;
        const float* q_in[1] = {scalar_begin.data()};
        float* q_out[1] = {q_gpu.data()};
        const float dtk[1] = {0.0f};
        const int positive[1] = {1};
        ScalarBoundaryTransportReport gpu_boundary;
        ScalarBoundaryTransportReport* boundary_reports[1] = {
            &gpu_boundary};
        const ScalarMoistEnergyKind energy_kinds[1] = {
            ScalarMoistEnergyKind::WaterVapor};
        ScalarMoistEnergyTransportReportV1 gpu_energy;
        ScalarMoistEnergyTransportReportV1* energy_reports[1] = {
            &gpu_energy};
        REQUIRE(run_gpu_batch(
            q_in, q_out, dtk, 1, report_dt, positive, &context,
            boundary_reports, energy_kinds, energy_reports));

        double scalar_inventory_scale = 0.0;
        double maximum_scalar = 0.0;
        double maximum_sensible_coefficient = 0.0;
        double maximum_speed_squared = 0.0;
        constexpr double kappa =
            physical_constants::gas_constant_dry_air_jkgk
            / physical_constants::specific_heat_cp_jkgk;
        constexpr double cv_v =
            physical_constants::specific_heat_water_vapor_cp_jkgk
            - physical_constants::gas_constant_water_vapor_jkgk;
        const double cell_volume = dr * dtheta * dz;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double begin_inventory = std::abs(
                        static_cast<double>(density_begin(i, j, k))
                        * scalar_begin(i, j, k));
                    const double cpu_inventory = std::abs(
                        static_cast<double>(rho(i, j, k)) * q_cpu(i, j, k));
                    const double gpu_inventory = std::abs(
                        static_cast<double>(rho(i, j, k)) * q_gpu(i, j, k));
                    scalar_inventory_scale += cell_volume * std::max({
                        begin_inventory, cpu_inventory, gpu_inventory});
                    maximum_scalar = std::max(
                        maximum_scalar,
                        std::abs(static_cast<double>(scalar_begin(i, j, k))));
                    const double exner = std::pow(
                        static_cast<double>(p(i, j, k))
                            / physical_constants::reference_pressure_pa,
                        kappa);
                    const double temperature =
                        static_cast<double>(theta(i, j, k)) * exner;
                    maximum_sensible_coefficient = std::max(
                        maximum_sensible_coefficient,
                        std::abs(cv_v
                            * (temperature
                               - physical_constants::freezing_temperature_k)));
                    maximum_speed_squared = std::max(
                        maximum_speed_squared,
                        static_cast<double>(u(i, j, k)) * u(i, j, k)
                            + static_cast<double>(v(i, j, k)) * v(i, j, k)
                            + static_cast<double>(w(i, j, k)) * w(i, j, k));
                }

        const double phase_coefficient =
            physical_constants::latent_heat_vaporization_jkg
            - physical_constants::gas_constant_water_vapor_jkgk
                * physical_constants::freezing_temperature_k;
        const double sensible_scale =
            scalar_inventory_scale * maximum_sensible_coefficient;
        const double phase_scale =
            scalar_inventory_scale * std::abs(phase_coefficient);
        const double kinetic_scale =
            scalar_inventory_scale * 0.5 * maximum_speed_squared;
        const double gravitational_scale =
            scalar_inventory_scale * physical_constants::gravity_ms2
            * (static_cast<double>(NZ) - 0.5) * dz;
        const double energy_scale =
            sensible_scale + phase_scale + kinetic_scale
            + gravitational_scale;

        require_parity(
            "report-bearing qv commit", q_cpu, q_gpu,
            fp32_operator_gamma(256) * std::max(maximum_scalar, 1.0e-6),
            nimbus::parity::ComparisonDomain::FullField);
        require_scalar_report_identity("CPU", cpu_boundary);
        require_scalar_report_identity("Vulkan", gpu_boundary);
        require_energy_report_identity("CPU", cpu_energy);
        require_energy_report_identity("Vulkan", gpu_energy);
        REQUIRE(cpu_boundary.outward_boundary_inventory == 0.0);
        REQUIRE(gpu_boundary.outward_boundary_inventory == 0.0);

        require_scaled_owner_parity(
            "boundary outward inventory",
            cpu_boundary.outward_boundary_inventory,
            gpu_boundary.outward_boundary_inventory,
            scalar_inventory_scale);
        require_scaled_owner_parity(
            "boundary flux-implied change",
            cpu_boundary.flux_implied_state_change,
            gpu_boundary.flux_implied_state_change,
            scalar_inventory_scale);
        require_scaled_owner_parity(
            "boundary committed change",
            cpu_boundary.committed_state_change,
            gpu_boundary.committed_state_change,
            scalar_inventory_scale);
        require_scaled_owner_parity(
            "boundary closure residual",
            cpu_boundary.closure_residual,
            gpu_boundary.closure_residual,
            scalar_inventory_scale);

        require_scaled_owner_parity(
            "energy dry internal", cpu_energy.dry_internal_change_j,
            gpu_energy.dry_internal_change_j, 1.0);
        require_scaled_owner_parity(
            "energy water sensible", cpu_energy.water_sensible_change_j,
            gpu_energy.water_sensible_change_j, sensible_scale);
        require_scaled_owner_parity(
            "energy phase reference", cpu_energy.phase_reference_change_j,
            gpu_energy.phase_reference_change_j, phase_scale);
        require_scaled_owner_parity(
            "energy carrier kinetic", cpu_energy.carrier_kinetic_change_j,
            gpu_energy.carrier_kinetic_change_j, kinetic_scale);
        require_scaled_owner_parity(
            "energy gravitational", cpu_energy.gravitational_change_j,
            gpu_energy.gravitational_change_j, gravitational_scale);
        require_scaled_owner_parity(
            "energy subgrid TKE", cpu_energy.subgrid_tke_change_j,
            gpu_energy.subgrid_tke_change_j, 1.0);
        require_scaled_owner_parity(
            "energy density remap", cpu_energy.density_history_remap_j,
            gpu_energy.density_history_remap_j, energy_scale);
        require_scaled_owner_parity(
            "energy outward boundary", cpu_energy.outward_boundary_energy_j,
            gpu_energy.outward_boundary_energy_j, energy_scale);
        require_scaled_owner_parity(
            "energy interior conversion", cpu_energy.interior_conversion_j,
            gpu_energy.interior_conversion_j, energy_scale);
        require_scaled_owner_parity(
            "energy limited-face change",
            cpu_energy.limited_face_state_change_j,
            gpu_energy.limited_face_state_change_j, energy_scale);
        require_scaled_owner_parity(
            "energy requested change", cpu_energy.requested_state_change_j,
            gpu_energy.requested_state_change_j, energy_scale);
        require_scaled_owner_parity(
            "energy storage rounding", cpu_energy.storage_rounding_j,
            gpu_energy.storage_rounding_j, energy_scale);
        require_scaled_owner_parity(
            "energy committed change", cpu_energy.committed_state_change_j,
            gpu_energy.committed_state_change_j, energy_scale);
        require_scaled_owner_parity(
            "energy closure residual", cpu_energy.closure_residual_j,
            gpu_energy.closure_residual_j, energy_scale);
    }

    SECTION("physical eight-scalar production batch preserves owner parity and order")
    {
        constexpr double report_dt = 0.25;
        global_top_bc = tmv::init::config::TopBC::Open;
        p.resize(NR, NTH, NZ, 0.0f);
        theta.resize(NR, NTH, NZ, 0.0f);
        qv.resize(NR, NTH, NZ, 0.0f);
        qc.resize(NR, NTH, NZ, 0.0f);
        qr.resize(NR, NTH, NZ, 0.0f);
        qi.resize(NR, NTH, NZ, 0.0f);
        qs.resize(NR, NTH, NZ, 0.0f);
        qg.resize(NR, NTH, NZ, 0.0f);
        qh.resize(NR, NTH, NZ, 0.0f);
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double x = static_cast<double>(i) / (NR - 1);
                    const double y = static_cast<double>(j) / (NTH - 1);
                    const double z = static_cast<double>(k) / (NZ - 1);
                    const double bump = std::exp(
                        -9.0 * ((x - 0.47) * (x - 0.47)
                               + (y - 0.53) * (y - 0.53)
                               + (z - 0.68) * (z - 0.68)));
                    rho(i, j, k) = static_cast<float>(
                        1.14 - 0.2 * z + 0.01 * std::sin(4.0 * x + y));
                    p(i, j, k) = static_cast<float>(
                        95000.0 * std::exp(-z * (NZ * dz) / 8200.0));
                    theta(i, j, k) =
                        static_cast<float>(297.0 + 8.0 * z + 1.8 * bump);
                    qv(i, j, k) =
                        static_cast<float>(0.0035 + 0.005 * bump);
                    qc(i, j, k) =
                        static_cast<float>(2.0e-5 + 6.0e-4 * bump);
                    qr(i, j, k) =
                        static_cast<float>(1.0e-5 + 3.0e-4 * bump);
                    qi(i, j, k) =
                        static_cast<float>(8.0e-6 + 2.0e-4 * bump);
                    qs(i, j, k) =
                        static_cast<float>(5.0e-6 + 1.2e-4 * bump);
                    qg(i, j, k) =
                        static_cast<float>(3.0e-6 + 8.0e-5 * bump);
                    qh(i, j, k) =
                        static_cast<float>(2.0e-6 + 5.0e-5 * bump);
                    u(i, j, k) = (i >= 1 && i <= NR - 3)
                        ? static_cast<float>(7.0 + 1.2 * std::sin(4.0 * y))
                        : 0.0f;
                    v(i, j, k) = (j >= 1 && j <= NTH - 3)
                        ? static_cast<float>(-4.0 + 0.8 * std::cos(3.0 * x))
                        : 0.0f;
                    w(i, j, k) = static_cast<float>(
                        0.25 * std::sin(3.0 * x) * std::cos(2.0 * y));
                    if (k == NZ - 1)
                        w(i, j, k) = static_cast<float>(1.2 + 0.15 * x);
                }

        const Field3D density_begin = rho;
        CartesianCGridTransportContext context;
        context.begin(NR, NTH, NZ, report_dt);
        context.accumulate(
            rho, u, v, w, static_cast<double>(static_cast<float>(report_dt)),
            nullptr, nullptr, tmv::AcousticMassFluxPrecision::NativeFloat);
        const double inv_dx = 1.0 / dr;
        const double inv_dy = 1.0 / dtheta;
        const double inv_dz = 1.0 / dz;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double mx =
                        static_cast<double>(context.mass_flux_x(i, j, k))
                        - context.mass_flux_x(i - 1, j, k);
                    const double my =
                        static_cast<double>(context.mass_flux_y(i, j, k))
                        - context.mass_flux_y(i, j - 1, k);
                    const double mz_top =
                        static_cast<double>(context.mass_flux_z(i, j, k));
                    const double mz_bottom = k > 0
                        ? static_cast<double>(
                              context.mass_flux_z(i, j, k - 1))
                        : 0.0;
                    rho(i, j, k) = static_cast<float>(
                        static_cast<double>(density_begin(i, j, k))
                        - report_dt
                            * (mx * inv_dx + my * inv_dy
                               + (mz_top - mz_bottom) * inv_dz));
                }
        context.complete();
        REQUIRE(context.validate(NR, NTH, NZ, report_dt));

        std::array<Field3D*, 8> fields = {
            &theta, &qv, &qc, &qr, &qi, &qs, &qg, &qh};
        const char* names[8] = {
            "theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh"};
        const ScalarMoistEnergyKind kinds[8] = {
            ScalarMoistEnergyKind::PotentialTemperature,
            ScalarMoistEnergyKind::WaterVapor,
            ScalarMoistEnergyKind::LiquidWater,
            ScalarMoistEnergyKind::LiquidWater,
            ScalarMoistEnergyKind::IceWater,
            ScalarMoistEnergyKind::IceWater,
            ScalarMoistEnergyKind::IceWater,
            ScalarMoistEnergyKind::IceWater,
        };
        std::array<Field3D, 8> initial_fields;
        for (std::size_t s = 0; s < fields.size(); ++s)
            initial_fields[s] = *fields[s];

        std::array<ScalarBoundaryTransportReport, 8> cpu_boundaries{};
        std::array<ScalarMoistEnergyTransportReportV1, 8> cpu_energies{};
        std::array<CartesianScalarTransport, 8> cpu_items{};
        for (std::size_t s = 0; s < fields.size(); ++s)
        {
            cpu_items[s] = {
                fields[s], 0.0, s != 0, &cpu_boundaries[s],
                kinds[s], &cpu_energies[s]};
        }
        advect_scalars_conservative_3d_cartesian(
            cpu_items.data(), cpu_items.size(), report_dt, &context);
        std::array<Field3D, 8> cpu_fields;
        for (std::size_t s = 0; s < fields.size(); ++s)
        {
            cpu_fields[s] = *fields[s];
            *fields[s] = initial_fields[s];
        }

        const float* q_in[8]{};
        float* q_out[8]{};
        float dtk[8]{};
        int positive[8]{};
        ScalarBoundaryTransportReport* gpu_boundary_ptrs[8]{};
        ScalarMoistEnergyTransportReportV1* gpu_energy_ptrs[8]{};
        std::array<ScalarBoundaryTransportReport, 8> gpu_boundaries{};
        std::array<ScalarMoistEnergyTransportReportV1, 8> gpu_energies{};
        for (std::size_t s = 0; s < fields.size(); ++s)
        {
            q_in[s] = fields[s]->data();
            q_out[s] = fields[s]->data();
            positive[s] = s != 0 ? 1 : 0;
            gpu_boundary_ptrs[s] = &gpu_boundaries[s];
            gpu_energy_ptrs[s] = &gpu_energies[s];
        }
        REQUIRE(run_gpu_batch(
            q_in, q_out, dtk, static_cast<int>(fields.size()), report_dt,
            positive, &context, gpu_boundary_ptrs, kinds, gpu_energy_ptrs));

        constexpr double kappa =
            physical_constants::gas_constant_dry_air_jkgk
            / physical_constants::specific_heat_cp_jkgk;
        constexpr double cv_d =
            physical_constants::specific_heat_cp_jkgk
            - physical_constants::gas_constant_dry_air_jkgk;
        constexpr double cv_v =
            physical_constants::specific_heat_water_vapor_cp_jkgk
            - physical_constants::gas_constant_water_vapor_jkgk;
        constexpr double vapor_reference =
            physical_constants::latent_heat_vaporization_jkg
            - physical_constants::gas_constant_water_vapor_jkgk
                * physical_constants::freezing_temperature_k;
        constexpr double ice_reference =
            physical_constants::latent_heat_fusion_jkg;
        const double cell_volume = dr * dtheta * dz;
        double max_u = 0.0, max_v = 0.0, max_w = 0.0;
        double max_vapor_sensible = 0.0;
        double max_liquid_sensible = 0.0;
        double max_ice_sensible = 0.0;
        double max_theta_coefficient = 0.0;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    max_u = std::max(
                        max_u, std::abs(static_cast<double>(u(i, j, k))));
                    max_v = std::max(
                        max_v, std::abs(static_cast<double>(v(i, j, k))));
                    max_w = std::max(
                        max_w, std::abs(static_cast<double>(w(i, j, k))));
                    const double exner = std::pow(
                        static_cast<double>(p(i, j, k))
                            / physical_constants::reference_pressure_pa,
                        kappa);
                    const double theta_scale = std::max({
                        std::abs(static_cast<double>(initial_fields[0](i, j, k))),
                        std::abs(static_cast<double>(cpu_fields[0](i, j, k))),
                        std::abs(static_cast<double>((*fields[0])(i, j, k))),
                    });
                    const double temperature = theta_scale * exner;
                    const double delta_temperature = std::abs(
                        temperature
                        - physical_constants::freezing_temperature_k);
                    max_vapor_sensible = std::max(
                        max_vapor_sensible, cv_v * delta_temperature);
                    max_liquid_sensible = std::max(
                        max_liquid_sensible,
                        physical_constants::specific_heat_liquid_water_jkgk
                            * delta_temperature);
                    max_ice_sensible = std::max(
                        max_ice_sensible,
                        physical_constants::specific_heat_ice_jkgk
                            * delta_temperature);
                    const double liquid =
                        static_cast<double>(initial_fields[2](i, j, k))
                        + initial_fields[3](i, j, k);
                    const double ice =
                        static_cast<double>(initial_fields[4](i, j, k))
                        + initial_fields[5](i, j, k)
                        + initial_fields[6](i, j, k)
                        + initial_fields[7](i, j, k);
                    const double water_heat_capacity =
                        static_cast<double>(initial_fields[1](i, j, k)) * cv_v
                        + liquid
                            * physical_constants::
                                specific_heat_liquid_water_jkgk
                        + ice
                            * physical_constants::specific_heat_ice_jkgk;
                    max_theta_coefficient = std::max(
                        max_theta_coefficient,
                        exner * (cv_d + std::abs(water_heat_capacity)));
                }
        const double kinetic_coefficient =
            0.5 * (max_u * max_u + max_v * max_v + max_w * max_w);
        const double gravitational_coefficient =
            physical_constants::gravity_ms2
            * (static_cast<double>(NZ) - 0.5) * dz;

        for (std::size_t s = 0; s < fields.size(); ++s)
        {
            double inventory_scale = 0.0;
            double maximum_scalar = 0.0;
            for (int i = 1; i < NR - 1; ++i)
                for (int j = 1; j < NTH - 1; ++j)
                    for (int k = 0; k < NZ; ++k)
                    {
                        maximum_scalar = std::max({
                            maximum_scalar,
                            std::abs(static_cast<double>(
                                initial_fields[s](i, j, k))),
                            std::abs(static_cast<double>(
                                cpu_fields[s](i, j, k))),
                            std::abs(static_cast<double>(
                                (*fields[s])(i, j, k))),
                        });
                        inventory_scale += cell_volume * std::max({
                            std::abs(
                                static_cast<double>(density_begin(i, j, k))
                                * initial_fields[s](i, j, k)),
                            std::abs(
                                static_cast<double>(rho(i, j, k))
                                * cpu_fields[s](i, j, k)),
                            std::abs(
                                static_cast<double>(rho(i, j, k))
                                * (*fields[s])(i, j, k)),
                        });
                    }

            double coefficient_scale = 0.0;
            switch (kinds[s])
            {
                case ScalarMoistEnergyKind::PotentialTemperature:
                    coefficient_scale = max_theta_coefficient;
                    break;
                case ScalarMoistEnergyKind::WaterVapor:
                    coefficient_scale =
                        max_vapor_sensible + std::abs(vapor_reference)
                        + kinetic_coefficient + gravitational_coefficient;
                    break;
                case ScalarMoistEnergyKind::LiquidWater:
                    coefficient_scale =
                        max_liquid_sensible + kinetic_coefficient
                        + gravitational_coefficient;
                    break;
                case ScalarMoistEnergyKind::IceWater:
                    coefficient_scale =
                        max_ice_sensible + ice_reference
                        + kinetic_coefficient + gravitational_coefficient;
                    break;
                case ScalarMoistEnergyKind::None:
                    FAIL("production thermodynamic batch contains None kind");
            }
            const double energy_scale = inventory_scale * coefficient_scale;
            INFO("production scalar owner=" << names[s]
                 << " inventory_scale=" << inventory_scale
                 << " energy_scale=" << energy_scale);
            require_parity(
                names[s], cpu_fields[s], *fields[s],
                fp32_operator_gamma(256)
                    * std::max(maximum_scalar, 1.0e-9),
                nimbus::parity::ComparisonDomain::FullField);
            REQUIRE(std::abs(
                cpu_boundaries[s].outward_boundary_inventory) > 0.0);
            REQUIRE(std::abs(
                gpu_boundaries[s].outward_boundary_inventory) > 0.0);
            require_scalar_owner_report_parity(
                names[s], cpu_boundaries[s], gpu_boundaries[s],
                cpu_energies[s], gpu_energies[s],
                inventory_scale, energy_scale);
        }
    }

    SECTION("coupled retained-rho positive-definite transport remains nonnegative over repeated steps")
    {
        // This is the device path used by production after the acoustic
        // solve: scalar transport receives the beginning density and the
        // acoustic loop's retained FP32 face-mass-flux history.  Exercise a
        // sparse tracer repeatedly so a round-off-scale undershoot cannot
        // hide behind a one-step smooth-field parity comparison.
        constexpr double pi = 3.14159265358979323846;
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double xi = static_cast<double>(i) / (NR - 1);
                    const double yj = static_cast<double>(j) / (NTH - 1);
                    const double zk = static_cast<double>(k) / (NZ - 1);
                    rho(i, j, k) = static_cast<float>(
                        0.9 + 0.12 * std::sin(2.0 * pi * xi)
                            * std::cos(2.0 * pi * yj)
                        + 0.08 * zk);
                    // Keep the prescribed compressible flow admissible over
                    // the full 400 s stress interval.  The sparse tracer
                    // still activates the donor limiter at zero/nonzero
                    // interfaces, while density remains a valid denominator
                    // instead of being drained by an indefinitely repeated
                    // analytic divergence field.
                    u(i, j, k) = (i >= 1 && i <= NR - 3)
                        ? static_cast<float>(
                            1.4 * std::sin(2.0 * pi * xi)
                            + 0.35 * std::cos(2.0 * pi * yj))
                        : 0.0f;
                    v(i, j, k) = (j >= 1 && j <= NTH - 3)
                        ? static_cast<float>(
                            -1.2 * std::sin(2.0 * pi * yj)
                            + 0.25 * std::cos(2.0 * pi * zk))
                        : 0.0f;
                    w(i, j, k) = (k <= NZ - 2)
                        ? static_cast<float>(
                            0.4 * std::sin(2.0 * pi * zk)
                            + 0.1 * std::cos(2.0 * pi * xi))
                        : 0.0f;
                }

        Field3D q_cpu(NR, NTH, NZ, 0.0f);
        for (int i = 2; i < NR - 2; ++i)
            for (int j = 2; j < NTH - 2; ++j)
                for (int k = 1; k < NZ - 1; ++k)
                {
                    const unsigned hash = static_cast<unsigned>(
                        i * 73856093u ^ j * 19349663u ^ k * 83492791u);
                    if ((hash % 17u) == 0u)
                    {
                        const int decade =
                            static_cast<int>((hash / 17u) % 6u);
                        q_cpu(i, j, k) = static_cast<float>(
                            1.0e-3 * std::pow(0.1, decade));
                    }
                }
        Field3D q_gpu = q_cpu;
        const float dtk[1] = {0.0f};
        const int pd[1] = {1};
        double worst_mass_relative_error = 0.0;

        for (int step = 1; step <= 200; ++step)
        {
            const Field3D rho_begin = rho;
            CartesianCGridTransportContext context;
            context.begin(NR, NTH, NZ, dt_step);
            context.accumulate(
                rho_begin, u, v, w, dt_step, nullptr, nullptr,
                tmv::AcousticMassFluxPrecision::NativeFloat);
            context.complete();

            for (int i = 1; i < NR - 1; ++i)
                for (int j = 1; j < NTH - 1; ++j)
                    for (int k = 0; k < NZ; ++k)
                    {
                        const double mx =
                            static_cast<double>(context.mass_flux_x(i, j, k))
                            - static_cast<double>(
                                context.mass_flux_x(i - 1, j, k));
                        const double my =
                            static_cast<double>(context.mass_flux_y(i, j, k))
                            - static_cast<double>(
                                context.mass_flux_y(i, j - 1, k));
                        const double mz_top = static_cast<double>(
                            context.mass_flux_z(i, j, k));
                        const double mz_bottom = k >= 1
                            ? static_cast<double>(
                                  context.mass_flux_z(i, j, k - 1))
                            : 0.0;
                        rho(i, j, k) = static_cast<float>(
                            static_cast<double>(rho_begin(i, j, k))
                            - dt_step
                                * (mx / dr + my / dtheta
                                   + (mz_top - mz_bottom) / dz));
                    }

            const double mass_before =
                interior_rho_q_sum(rho_begin, q_gpu);
            Field3D cpu_next;
            advect_scalar_conservative_3d_cartesian(
                q_cpu, cpu_next, dt_step, true, &context);

            const float* q_in[1] = {q_gpu.data()};
            float* q_out[1] = {q_gpu.data()};
            REQUIRE(run_gpu_batch(
                q_in, q_out, dtk, 1, dt_step, pd, &context));
            q_cpu = std::move(cpu_next);

            require_parity(
                "repeated coupled positive-definite transport",
                q_cpu, q_gpu, 2.0e-5,
                nimbus::parity::ComparisonDomain::FullField);

            float gpu_min = std::numeric_limits<float>::infinity();
            int min_i = -1;
            int min_j = -1;
            int min_k = -1;
            for (int i = 1; i < NR - 1; ++i)
                for (int j = 1; j < NTH - 1; ++j)
                    for (int k = 0; k < NZ; ++k)
                        if (q_gpu(i, j, k) < gpu_min)
                        {
                            gpu_min = q_gpu(i, j, k);
                            min_i = i;
                            min_j = j;
                            min_k = k;
                        }
            const double mass_after = interior_rho_q_sum(rho, q_gpu);
            const double relative_mass_error =
                std::abs(mass_after - mass_before)
                / std::max(std::abs(mass_before), 1.0e-30);
            worst_mass_relative_error = std::max(
                worst_mass_relative_error, relative_mass_error);
            INFO("step=" << step << " minimum=" << gpu_min
                 << " at (" << min_i << "," << min_j << "," << min_k
                 << ") relative_mass_error=" << relative_mass_error);
            REQUIRE(gpu_min >= 0.0f);
        }
        INFO("worst repeated coupled rho-q relative error="
             << worst_mass_relative_error);
        REQUIRE(worst_mass_relative_error < 3.0e-6);
    }

    shutdown_backend();
}

TEST_CASE("Positive-definite flux renormalization (science audit M5)",
          "[vulkan][gpu_parity][cgrid][advection][cartesian][pd]")
{
    // A near-zero moisture edge under strong wind undershoots negative with
    // raw WENO5 fluxes; the Skamarock-Weisman renormalization must keep the
    // result >= 0 while conserving the rho-weighted integral exactly (the
    // fluxes are rescaled, never clipped) and leaving smooth fields and
    // free-stream untouched.
    setup_cartesian_cgrid_grid(/*nx=*/18, /*ny=*/18, /*nz=*/18,
                               /*dx=*/500.0, /*dy=*/500.0, /*dz=*/250.0);
    const double dt_step = 8.0;  // aggressive CFL ~0.5 to force undershoot

    fill_global_state();
    // Divergence-free flow (uniform u, zero v/w): rho_check == rho
    // exactly, so the rho-weighted sum below is the precisely
    // conserved quantity (with divergent winds the conserved weight
    // is rho_check, not rho). CFL = 30*8/500 = 0.48 still forces
    // WENO undershoot at the box edges.
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u(i, j, k) = 30.0f;
                v(i, j, k) = 0.0f;
                w(i, j, k) = 0.0f;
            }

    // Sharp box of moisture surrounded by exact zeros.
    Field3D q(NR, NTH, NZ, 0.0f);
    for (int i = 6; i <= 9; ++i)
        for (int j = 6; j <= 11; ++j)
            for (int k = 6; k <= 11; ++k)
                q(i, j, k) = 1.0e-3f;

    auto rho_weighted_sum = [&](const Field3D& f) {
        double s = 0.0;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                    s += static_cast<double>(f(i, j, k))
                       * static_cast<double>(rho(i, j, k));
        return s;
    };

    SECTION("CPU kernel: limiter removes negatives, conserves, free-stream intact")
    {
        Field3D raw, lim;
        advect_scalar_conservative_3d_cartesian(q, raw, dt_step, false);
        advect_scalar_conservative_3d_cartesian(q, lim, dt_step, true);

        float raw_min = 0.0f, lim_min = 0.0f;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    raw_min = std::min(raw_min, raw(i, j, k));
                    lim_min = std::min(lim_min, lim(i, j, k));
                }
        INFO("raw min=" << raw_min << " limited min=" << lim_min);
        REQUIRE(raw_min < 0.0f);       // the test really exercises undershoot
        REQUIRE(lim_min >= 0.0f);      // positivity guaranteed

        // Note: comparing the limited result's integral against the RAW
        // result's (not the input's): one step of divergent transport
        // legitimately changes integral-q; renormalization must not change
        // it further beyond rounding.
        const double sum_raw = rho_weighted_sum(raw);
        const double sum_lim = rho_weighted_sum(lim);
        INFO("rho-weighted sums raw=" << sum_raw << " lim=" << sum_lim);
        REQUIRE(sum_lim == Approx(sum_raw).epsilon(1.0e-8));

        // Free-stream: constant field unaffected by the limiter.
        Field3D ones(NR, NTH, NZ, 1.0f), out_fs;
        advect_scalar_conservative_3d_cartesian(ones, out_fs, dt_step, true);
        double dev = 0.0;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                    dev = std::max(dev,
                        std::abs(static_cast<double>(out_fs(i, j, k)) - 1.0));
        REQUIRE(dev < 1.0e-6);
    }

    SECTION("GPU batch matches the limited CPU kernel")
    {
        if (!try_init_gpu())
        {
            WARN("GPU unavailable -- skipping PD parity");
            return;
        }
        auto* backend = mutable_compute_backend();
        auto* kernels = backend != nullptr ? backend->advection_kernels() : nullptr;
        if (kernels == nullptr
            || !kernels->supports(AdvectionKernelOp::ConservativeCartesianCgridTransport))
        {
            WARN("transport batch unsupported -- skipping");
            shutdown_backend();
            return;
        }
        Field3D q_gpu = q;
        Field3D q_cpu = q;
        Field3D q_separate = q;
        const float* separate_in[1] = { q.data() };
        float* separate_out[1] = { q_separate.data() };
        const float* q_in[1] = { q_gpu.data() };
        float* q_out[1] = { q_gpu.data() };
        const float dtk[1] = { 0.0f };
        const int pd[1] = { 1 };
        REQUIRE(run_gpu_batch(
            separate_in, separate_out, dtk, 1, dt_step, pd));
        float separate_min = 0.0f;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                    separate_min = std::min(
                        separate_min, q_separate(i, j, k));
        INFO("separate input/output minimum=" << separate_min);
        REQUIRE(separate_min >= 0.0f);
        for (int step = 0; step < 12; ++step)
        {
            REQUIRE(run_gpu_batch(q_in, q_out, dtk, 1, dt_step, pd));

            Field3D cpu_next;
            advect_scalar_conservative_3d_cartesian(
                q_cpu, cpu_next, dt_step, true);
            q_cpu = std::move(cpu_next);
            require_parity(
                "repeated positive-definite limiter", q_cpu, q_gpu, 2.0e-5,
                nimbus::parity::ComparisonDomain::FullField);

            float gpu_min = 0.0f;
            int min_i = -1;
            int min_j = -1;
            int min_k = -1;
            for (int i = 1; i < NR - 1; ++i)
                for (int j = 1; j < NTH - 1; ++j)
                    for (int k = 0; k < NZ; ++k)
                        if (q_gpu(i, j, k) < gpu_min)
                        {
                            gpu_min = q_gpu(i, j, k);
                            min_i = i;
                            min_j = j;
                            min_k = k;
                        }
            INFO("step=" << step + 1 << " minimum=" << gpu_min
                 << " at (" << min_i << "," << min_j << "," << min_k
                 << ") cpu=" << (min_i >= 0
                         ? q_cpu(min_i, min_j, min_k) : 0.0f));
            REQUIRE(gpu_min >= 0.0f);
        }
        shutdown_backend();
    }
}

TEST_CASE("Positive-definite Cartesian C-grid transport remains nonnegative over sparse multidirectional steps",
          "[cgrid][advection][cartesian][pd][regression]")
{
    // Pre-fix baseline: nearest-rounded limiter factors produced
    // q_min = -3.58539e-15 at step 2, cell (10,15,13).  The magnitude is
    // round-off scale, but a subsequent max(q,0) would silently add that
    // scalar mass and repeated steps would propagate the sign error.
    setup_cartesian_cgrid_grid(/*nx=*/22, /*ny=*/20, /*nz=*/18,
                               /*dx=*/400.0, /*dy=*/550.0, /*dz=*/300.0);
    constexpr double dt_step = 3.0;
    constexpr int nsteps = 5;

    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    rho.resize(NR, NTH, NZ, 0.0f);

    // Smooth, genuinely multidirectional flow with nonzero local divergence.
    // Normal velocity is zero at the domain faces so the scalar-mass budget
    // telescopes over the updated interior columns.
    constexpr double pi = 3.14159265358979323846;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double xi = static_cast<double>(i) / (NR - 1);
                const double yj = static_cast<double>(j) / (NTH - 1);
                const double zk = static_cast<double>(k) / (NZ - 1);
                rho(i, j, k) = static_cast<float>(
                    0.9 + 0.15 * std::sin(2.0 * pi * xi)
                        * std::cos(2.0 * pi * yj)
                    + 0.05 * zk);

                if (i >= 1 && i <= NR - 3)
                    u(i, j, k) = static_cast<float>(
                        34.0 * std::sin(2.0 * pi * xi)
                        + 9.0 * std::cos(2.0 * pi * yj));
                if (j >= 1 && j <= NTH - 3)
                    v(i, j, k) = static_cast<float>(
                        -29.0 * std::sin(2.0 * pi * yj)
                        + 7.0 * std::cos(2.0 * pi * zk));
                if (k >= 1 && k <= NZ - 2)
                    w(i, j, k) = static_cast<float>(
                        11.0 * std::sin(2.0 * pi * zk)
                        + 3.0 * std::cos(2.0 * pi * xi));
            }

    // Sparse nonnegative tracer with discontinuities and a wide dynamic
    // range. These are the cells where WENO face reconstruction most often
    // needs the outgoing-flux positivity limiter.
    Field3D q(NR, NTH, NZ, 0.0f);
    for (int i = 2; i < NR - 2; ++i)
        for (int j = 2; j < NTH - 2; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                const unsigned hash = static_cast<unsigned>(
                    i * 73856093u ^ j * 19349663u ^ k * 83492791u);
                if ((hash % 19u) == 0u)
                {
                    const int decade = static_cast<int>((hash / 19u) % 5u);
                    q(i, j, k) = static_cast<float>(
                        2.5e-4 * std::pow(0.1, decade));
                }
            }

    float global_min = std::numeric_limits<float>::infinity();
    int min_step = -1, min_i = -1, min_j = -1, min_k = -1;
    double max_relative_mass_error = 0.0;
    double minimum_density = std::numeric_limits<double>::infinity();
    for (int step = 1; step <= nsteps; ++step)
    {
        double mass_before = 0.0;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                    mass_before += static_cast<double>(rho(i, j, k))
                                 * static_cast<double>(q(i, j, k));

        Field3D next;
        advect_scalar_conservative_3d_cartesian(q, next, dt_step, true);

        // Advance density with the same rounded face mass fluxes used by the
        // production scalar kernel.  This makes the repeated divergent-flow
        // test physically consistent and lets it verify the conservative
        // rho*q budget without relying on a post-update clamp.
        Field3D rho_next = rho;
        double mass_after = 0.0;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const auto mfx = [&](int fi) {
                        return static_cast<float>(
                            0.5 * (static_cast<double>(rho(fi, j, k))
                                 + static_cast<double>(rho(fi + 1, j, k)))
                            * static_cast<double>(u(fi, j, k)));
                    };
                    const auto mfy = [&](int fj) {
                        return static_cast<float>(
                            0.5 * (static_cast<double>(rho(i, fj, k))
                                 + static_cast<double>(rho(i, fj + 1, k)))
                            * static_cast<double>(v(i, fj, k)));
                    };
                    const auto mfz = [&](int fk) {
                        return static_cast<float>(
                            0.5 * (static_cast<double>(rho(i, j, fk))
                                 + static_cast<double>(rho(i, j, fk + 1)))
                            * static_cast<double>(w(i, j, fk)));
                    };
                    const double mass_div =
                        (static_cast<double>(mfx(i))
                         - static_cast<double>(mfx(i - 1))) / dr
                        + (static_cast<double>(mfy(j))
                           - static_cast<double>(mfy(j - 1))) / dtheta
                        + (((k <= NZ - 2) ? static_cast<double>(mfz(k)) : 0.0)
                           - ((k >= 1) ? static_cast<double>(mfz(k - 1)) : 0.0)) / dz;
                    rho_next(i, j, k) = static_cast<float>(
                        static_cast<double>(rho(i, j, k)) - dt_step * mass_div);
                    minimum_density = std::min(
                        minimum_density,
                        static_cast<double>(rho_next(i, j, k)));
                    mass_after += static_cast<double>(rho_next(i, j, k))
                                * static_cast<double>(next(i, j, k));

                    if (next(i, j, k) < global_min)
                    {
                        global_min = next(i, j, k);
                        min_step = step;
                        min_i = i;
                        min_j = j;
                        min_k = k;
                    }
                }

        const double scale = std::max(std::abs(mass_before), 1.0e-30);
        max_relative_mass_error = std::max(
            max_relative_mass_error, std::abs(mass_after - mass_before) / scale);
        rho = std::move(rho_next);
        q = std::move(next);
    }

    INFO("minimum=" << global_min << " at step=" << min_step
         << " cell=(" << min_i << "," << min_j << "," << min_k << ")"
         << " minimum_density=" << minimum_density
         << " max_relative_mass_error=" << max_relative_mass_error);
    REQUIRE(minimum_density > static_cast<double>(density_min_kgm3));
    REQUIRE(global_min >= 0.0f);
    REQUIRE(max_relative_mass_error < 2.0e-6);
}

TEST_CASE("Native Cartesian positive scalar budget preserves tiny fields and rejects invalid batches",
          "[vulkan][gpu_parity][cgrid][advection][pd-contract]")
{
    if (!try_init_gpu())
    {
        WARN("Vulkan unavailable -- positive scalar budget contract was not exercised");
        return;
    }
    auto* backend = mutable_compute_backend();
    REQUIRE(backend != nullptr);
    nimbus::testing::exercise_positive_scalar_contract(*backend);
    nimbus::testing::exercise_positive_scalar_scales(*backend);
    shutdown_backend();
}
