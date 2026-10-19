/**
 * @file test_acoustic_cartesian_cgrid_gpu_parity.cpp
 * @brief CPU/GPU parity for the Cartesian Arakawa C-grid acoustic substeps
 *        (GPU.md Phase 3.1): the diagnose-p pressure pass, the
 *        perturbation-p momentum pass, and the fused composite.
 *
 * The CPU production path (CartesianCGrid3DScheme::compute_fast_*_tendencies
 * + the integration loops in dynamics.cpp) runs inner arithmetic in DOUBLE
 * and stores float; the GPU shaders run float throughout. Each section
 * compares full fields against an inline double-precision replica of the
 * CPU path, so boundary semantics (zero-tendency passthrough + universal
 * clamp + universal p re-diagnosis) are verified everywhere, not just at
 * sentinels.
 *
 * Sections:
 *   1. Pressure, hydrostatic identity: winds zero and rho/p equal to the
 *      base profiles -> outputs equal inputs bit-exactly.
 *   2. Pressure, smooth flow: Gaussian u bump + rho perturbation ->
 *      CPU-vs-GPU within float drift.
 *   3. Momentum, hydrostatic identity: p equals p0_base exactly ->
 *      p' = 0 -> velocities unchanged bit-exactly.
 *   4. Momentum, pressure bump: localized p' -> face accelerations match
 *      the CPU within float drift.
 *   5. Fused composite: pressure + momentum in one submission matches the
 *      sequential CPU passes.
 */

#include "catch2/catch.hpp"

#include "boundary_conditions/factory.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_backend_factory.hpp"
#include "compute/parity_check.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/field/field3d.hpp"
#include "core/infra/grid_geometry.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "dynamics/acoustic_energy_report.hpp"
#include "dynamics/device_acoustic_energy_attribution.hpp"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr double kGamma = 1.4;
constexpr float  kPRef  = 100000.0f;  // dynamics p0 reference fallback

struct AcousticBoundaryConfigGuard
{
    using LateralBC = tmv::init::config::LateralBC;
    using BottomBC = tmv::init::config::BottomBC;
    using TopBC = tmv::init::config::TopBC;

    LateralBC west = global_lateral_bc_west;
    LateralBC east = global_lateral_bc_east;
    LateralBC south = global_lateral_bc_south;
    LateralBC north = global_lateral_bc_north;
    BottomBC bottom = global_bottom_bc;
    TopBC top = global_top_bc;

    ~AcousticBoundaryConfigGuard()
    {
        global_lateral_bc_west = west;
        global_lateral_bc_east = east;
        global_lateral_bc_south = south;
        global_lateral_bc_north = north;
        global_bottom_bc = bottom;
        global_top_bc = top;
    }
};

void setup_cartesian_cgrid_grid(int nx, int ny, int nz,
                                double dx_m, double dy_m, double dz_m,
                                double dt_s)
{
    NR     = nx;
    NTH    = ny;
    NZ     = nz;
    dr     = dx_m;   // Cartesian: dr maps to dx
    dtheta = dy_m;   // Cartesian: dtheta maps to dy (meters)
    dz     = dz_m;
    dt     = dt_s;

    global_coordinate_system = CoordinateSystem::Cartesian;
    global_stagger_type      = StaggerType::CGrid;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    global_coordinate_system,
                                    global_stagger_type);
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
                "required physical Vulkan acoustic qualification could not "
                "initialize a device: " + device_error);
        }
        return false;
    }
    return true;
}

bool physical_gpu_required()
{
    const char* required = std::getenv("NIMBUS_GPU_REQUIRED");
    return required != nullptr && std::string(required) == "1";
}

void shutdown_backend()
{
    shutdown_compute_backend_runtime();
}

double fp32_acoustic_gamma(int operations)
{
    const double product =
        static_cast<double>(operations) * std::numeric_limits<float>::epsilon();
    REQUIRE(product < 1.0);
    return product / (1.0 - product);
}

void require_acoustic_value_parity(
    const char* name, double cpu_value, double gpu_value, double scale,
    int fp32_operations = 256)
{
    const double tolerance =
        fp32_acoustic_gamma(fp32_operations) * std::max(scale, 1.0);
    INFO(name << ": cpu=" << cpu_value << " gpu=" << gpu_value
         << " difference=" << std::abs(gpu_value - cpu_value)
         << " physical_scale=" << scale << " tolerance=" << tolerance);
    REQUIRE(std::isfinite(cpu_value));
    REQUIRE(std::isfinite(gpu_value));
    REQUIRE(std::abs(gpu_value - cpu_value) <= tolerance);
}

void require_acoustic_owner_parity(
    const CartesianCGridAcousticEnergyReportV1& cpu,
    const CartesianCGridAcousticEnergyReportV1& gpu)
{
    REQUIRE(cpu.complete);
    REQUIRE(gpu.complete);
    REQUIRE(cpu.substep_count == 1);
    REQUIRE(gpu.substep_count == 1);
    const char* names[] = {
        "continuity dry internal", "continuity water sensible",
        "continuity phase reference", "continuity carrier kinetic",
        "continuity gravitational", "continuity subgrid TKE",
        "continuity carrier request", "pressure requested",
        "pressure finite-step", "pressure storage",
        "pressure committed", "committed inventory",
    };
    const double cpu_values[] = {
        cpu.continuity_eos_dry_internal_change_j,
        cpu.continuity_eos_water_sensible_change_j,
        cpu.continuity_eos_phase_reference_change_j,
        cpu.continuity_carrier_kinetic_change_j,
        cpu.continuity_gravitational_change_j,
        cpu.continuity_subgrid_tke_change_j,
        cpu.continuity_carrier_requested_change_j,
        cpu.pressure_gradient_requested_linear_work_j,
        cpu.pressure_gradient_finite_step_correction_j,
        cpu.pressure_gradient_storage_rounding_j,
        cpu.pressure_gradient_committed_work_j,
        cpu.committed_inventory_change_j,
    };
    const double gpu_values[] = {
        gpu.continuity_eos_dry_internal_change_j,
        gpu.continuity_eos_water_sensible_change_j,
        gpu.continuity_eos_phase_reference_change_j,
        gpu.continuity_carrier_kinetic_change_j,
        gpu.continuity_gravitational_change_j,
        gpu.continuity_subgrid_tke_change_j,
        gpu.continuity_carrier_requested_change_j,
        gpu.pressure_gradient_requested_linear_work_j,
        gpu.pressure_gradient_finite_step_correction_j,
        gpu.pressure_gradient_storage_rounding_j,
        gpu.pressure_gradient_committed_work_j,
        gpu.committed_inventory_change_j,
    };
    const double pressure_conditioning_scale = std::max({
        std::abs(cpu.pressure_gradient_requested_linear_work_j),
        std::abs(gpu.pressure_gradient_requested_linear_work_j),
        std::abs(cpu.pressure_gradient_finite_step_correction_j),
        std::abs(gpu.pressure_gradient_finite_step_correction_j),
        std::abs(cpu.pressure_gradient_committed_work_j),
        std::abs(gpu.pressure_gradient_committed_work_j),
        1.0,
    });
    double scales[std::size(names)] = {};
    for (std::size_t n = 0; n < std::size(names); ++n)
        scales[n] = std::max({
            std::abs(cpu_values[n]), std::abs(gpu_values[n]), 1.0});
    // Storage rounding is a residual of requested, finite-step, and committed
    // work. Its conditioning scale is therefore those parent terms, not the
    // often much smaller residual itself.
    scales[9] = pressure_conditioning_scale;
    for (std::size_t n = 0; n < std::size(names); ++n)
        require_acoustic_value_parity(
            names[n], cpu_values[n], gpu_values[n], scales[n]);

    const double cpu_pressure_identity =
        cpu.pressure_gradient_committed_work_j
        - cpu.pressure_gradient_requested_linear_work_j
        - cpu.pressure_gradient_finite_step_correction_j
        - cpu.pressure_gradient_storage_rounding_j;
    const double gpu_pressure_identity =
        gpu.pressure_gradient_committed_work_j
        - gpu.pressure_gradient_requested_linear_work_j
        - gpu.pressure_gradient_finite_step_correction_j
        - gpu.pressure_gradient_storage_rounding_j;
    const auto inventory_identity =
        [](const CartesianCGridAcousticEnergyReportV1& value)
        {
            return value.committed_inventory_change_j
                - value.continuity_eos_dry_internal_change_j
                - value.continuity_eos_water_sensible_change_j
                - value.continuity_eos_phase_reference_change_j
                - value.continuity_carrier_kinetic_change_j
                - value.continuity_gravitational_change_j
                - value.continuity_subgrid_tke_change_j
                - value.pressure_gradient_committed_work_j;
        };
    const double identity_tolerance =
        64.0 * std::numeric_limits<double>::epsilon()
        * pressure_conditioning_scale;
    REQUIRE(std::abs(cpu_pressure_identity) <= identity_tolerance);
    REQUIRE(std::abs(gpu_pressure_identity) <= identity_tolerance);
    REQUIRE(std::abs(inventory_identity(cpu))
        <= 64.0 * std::numeric_limits<double>::epsilon()
            * scales[11]);
    REQUIRE(std::abs(inventory_identity(gpu))
        <= 64.0 * std::numeric_limits<double>::epsilon()
            * scales[11]);
}

void require_boundary_owner_parity(
    const CartesianCGridAcousticEnergyReportV1& cpu,
    const CartesianCGridAcousticEnergyReportV1& gpu)
{
    REQUIRE(cpu.boundary_complete);
    REQUIRE(gpu.boundary_complete);
    REQUIRE(
        cpu.boundary_application_count == gpu.boundary_application_count);
    const char* names[] = {
        "boundary committed",
        "boundary radiative requested",
        "boundary radiative finite-step",
        "boundary radiative storage",
        "boundary radiative committed",
        "boundary constraint/history",
    };
    const double cpu_values[] = {
        cpu.boundary_committed_work_j,
        cpu.boundary_radiative_requested_linear_work_j,
        cpu.boundary_radiative_finite_step_correction_j,
        cpu.boundary_radiative_storage_rounding_j,
        cpu.boundary_radiative_committed_work_j,
        cpu.boundary_constraint_history_committed_work_j,
    };
    const double gpu_values[] = {
        gpu.boundary_committed_work_j,
        gpu.boundary_radiative_requested_linear_work_j,
        gpu.boundary_radiative_finite_step_correction_j,
        gpu.boundary_radiative_storage_rounding_j,
        gpu.boundary_radiative_committed_work_j,
        gpu.boundary_constraint_history_committed_work_j,
    };
    const double radiative_conditioning_scale = std::max({
        std::abs(cpu_values[1]), std::abs(gpu_values[1]),
        std::abs(cpu_values[2]), std::abs(gpu_values[2]),
        std::abs(cpu_values[4]), std::abs(gpu_values[4]), 1.0});
    const double boundary_conditioning_scale = std::max({
        std::abs(cpu_values[0]), std::abs(gpu_values[0]),
        std::abs(cpu_values[4]), std::abs(gpu_values[4]),
        std::abs(cpu_values[5]), std::abs(gpu_values[5]), 1.0});
    for (std::size_t n = 0; n < std::size(names); ++n)
    {
        double scale =
            std::max({std::abs(cpu_values[n]), std::abs(gpu_values[n]), 1.0});
        if (n == 3)
            scale = radiative_conditioning_scale;
        else if (n == 5)
            scale = boundary_conditioning_scale;
        require_acoustic_value_parity(
            names[n], cpu_values[n], gpu_values[n], scale,
            /*fp32_operations=*/2048);
    }

    const auto radiative_identity =
        [](const CartesianCGridAcousticEnergyReportV1& value)
        {
            return value.boundary_radiative_committed_work_j
                - value.boundary_radiative_requested_linear_work_j
                - value.boundary_radiative_finite_step_correction_j
                - value.boundary_radiative_storage_rounding_j;
        };
    const auto boundary_identity =
        [](const CartesianCGridAcousticEnergyReportV1& value)
        {
            return value.boundary_committed_work_j
                - value.boundary_radiative_committed_work_j
                - value.boundary_constraint_history_committed_work_j;
        };
    REQUIRE(std::abs(radiative_identity(cpu))
        <= 64.0 * std::numeric_limits<double>::epsilon()
            * radiative_conditioning_scale);
    REQUIRE(std::abs(radiative_identity(gpu))
        <= 64.0 * std::numeric_limits<double>::epsilon()
            * radiative_conditioning_scale);
    REQUIRE(std::abs(boundary_identity(cpu))
        <= 64.0 * std::numeric_limits<double>::epsilon()
            * boundary_conditioning_scale);
    REQUIRE(std::abs(boundary_identity(gpu))
        <= 64.0 * std::numeric_limits<double>::epsilon()
            * boundary_conditioning_scale);
}

/// Stratified base profiles (isothermal-like exponential decay), plus the
/// base-state vertical pressure gradient differenced in double before the
/// single fp32 rounding (GPU audit section 2.3; matches dynamics.cpp).
void build_base_profiles(std::vector<float>& rho0_f, std::vector<float>& p0_f,
                         std::vector<float>& dp0_dz_f,
                         std::vector<float>& theta0_f)
{
    rho0_f.resize(static_cast<std::size_t>(NZ));
    p0_f.resize(static_cast<std::size_t>(NZ));
    dp0_dz_f.resize(static_cast<std::size_t>(NZ));
    theta0_f.resize(static_cast<std::size_t>(NZ));
    for (int k = 0; k < NZ; ++k)
    {
        const double z = (static_cast<double>(k) + 0.5) * dz;
        rho0_f[static_cast<std::size_t>(k)] =
            static_cast<float>(1.2 * std::exp(-z / 8500.0));
        p0_f[static_cast<std::size_t>(k)] =
            static_cast<float>(100000.0 * std::exp(-z / 8500.0));
    }
    for (int k = 0; k < NZ; ++k)
    {
        dp0_dz_f[static_cast<std::size_t>(k)] =
            (k + 1 < NZ)
                ? static_cast<float>(
                      (static_cast<double>(p0_f[static_cast<std::size_t>(k + 1)])
                       - static_cast<double>(p0_f[static_cast<std::size_t>(k)])) / dz)
                : 0.0f;
    }

    // Dry-EOS-consistent theta0: with kappa = 1 - 1/gamma,
    // p00 (rho0 R_d theta0 / p00)^gamma == p0 exactly in fp64
    // (SCIENCE.md C1 involution), so the EOS diagnosis returns the
    // base pressure at the base state.
    const double kappa_ref = 1.0 - 1.0 / kGamma;
    for (int k = 0; k < NZ; ++k)
    {
        const double p0_d = static_cast<double>(p0_f[static_cast<std::size_t>(k)]);
        const double rho0_d = static_cast<double>(rho0_f[static_cast<std::size_t>(k)]);
        const double T0_d = p0_d / (287.0 * rho0_d);
        theta0_f[static_cast<std::size_t>(k)] = static_cast<float>(
            T0_d * std::pow(100000.0 / p0_d, kappa_ref));
    }
}

double safe_rho_ref(double rho_val)
{
    return (std::isfinite(rho_val) && rho_val > 1.0e-6) ? rho_val : 1.0;
}

/// Inline double-precision replica of the CPU production pressure path:
/// CartesianCGrid3DScheme::compute_fast_pressure_tendencies (the drho part,
/// div_flux_mass with wall/lid/ground closures) + the apply_fast_pressure
/// integration and diagnose-p loop in dynamics.cpp.
void cpu_reference_pressure(
    const Field3D& uf, const Field3D& vf, const Field3D& wf,
    const Field3D& rho_in, const Field3D& p_in,
    const Field3D& theta_in,
    const std::vector<float>& rho0_f, const std::vector<float>& p0_f,
    double dt_small,
    Field3D& rho_out, Field3D& p_out)
{
    rho_out.resize(NR, NTH, NZ, 0.0f);
    p_out.resize(NR, NTH, NZ, 0.0f);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                double drho = 0.0;
                if (i >= 1 && i <= NR - 2 && j >= 1 && j <= NTH - 2)
                {
                    auto rho_at = [&](int ii, int jj, int kk) {
                        return static_cast<double>(rho_in(ii, jj, kk));
                    };
                    // x mass flux divergence (rigid +x wall at i+1 == NX).
                    double flux_xp = 0.0;
                    if (i + 1 < NR)
                    {
                        flux_xp = 0.5 * (rho_at(i, j, k) + rho_at(i + 1, j, k))
                                * static_cast<double>(uf(i, j, k));
                    }
                    const double flux_xm =
                        0.5 * (rho_at(i - 1, j, k) + rho_at(i, j, k))
                        * static_cast<double>(uf(i - 1, j, k));
                    const double divm_x = (flux_xp - flux_xm) / dr;

                    double flux_yp = 0.0;
                    if (j + 1 < NTH)
                    {
                        flux_yp = 0.5 * (rho_at(i, j, k) + rho_at(i, j + 1, k))
                                * static_cast<double>(vf(i, j, k));
                    }
                    const double flux_ym =
                        0.5 * (rho_at(i, j - 1, k) + rho_at(i, j, k))
                        * static_cast<double>(vf(i, j - 1, k));
                    const double divm_y = (flux_yp - flux_ym) / dtheta;

                    double flux_zp = 0.0;
                    if (k + 1 < NZ)
                    {
                        flux_zp = 0.5 * (rho_at(i, j, k) + rho_at(i, j, k + 1))
                                * static_cast<double>(wf(i, j, k));
                    }
                    double flux_zm = 0.0;
                    if (k > 0)
                    {
                        flux_zm = 0.5 * (rho_at(i, j, k - 1) + rho_at(i, j, k))
                                * static_cast<double>(wf(i, j, k - 1));
                    }
                    const double divm_z = (flux_zp - flux_zm) / dz;

                    drho = -(divm_x + divm_y + divm_z);
                    if (!std::isfinite(drho)) drho = 0.0;
                }

                const double rho0_k = static_cast<double>(
                    rho0_f[static_cast<std::size_t>(k)]);
                double rho_new =
                    static_cast<double>(rho_in(i, j, k)) + drho * dt_small;
                if (!std::isfinite(rho_new) || rho_new <= 0.0)
                {
                    rho_new = std::max(0.1, rho0_k);
                }
                const float rho_stored =
                    clamp_density_kgm3(static_cast<float>(rho_new));
                rho_out(i, j, k) = rho_stored;

                // Full dry rho-theta EOS diagnosis (SCIENCE.md C1),
                // mirroring the production CPU path.
                const double th = static_cast<double>(theta_in(i, j, k));
                double p_new = static_cast<double>(kPRef);
                if (th > 0.0)
                {
                    p_new = 100000.0 * std::pow(
                        static_cast<double>(rho_stored) * 287.0 * th / 100000.0,
                        kGamma);
                }
                if (!std::isfinite(p_new) || p_new <= 0.0)
                {
                    p_new = static_cast<double>(kPRef);
                }
                p_out(i, j, k) = clamp_pressure_pa(static_cast<float>(p_new));
            }
}

/// Inline double-precision replica of the CPU production momentum path:
/// CartesianCGrid3DScheme::compute_fast_momentum_tendencies + the
/// apply_fast_momentum integration loop in dynamics.cpp.
void cpu_reference_momentum(
    const Field3D& rho_in, const Field3D& p_in,
    const std::vector<float>& p0_f,
    const Field3D& u_in, const Field3D& v_in, const Field3D& w_in,
    double dt_small,
    Field3D& u_out, Field3D& v_out, Field3D& w_out)
{
    u_out.resize(NR, NTH, NZ, 0.0f);
    v_out.resize(NR, NTH, NZ, 0.0f);
    w_out.resize(NR, NTH, NZ, 0.0f);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const bool h_interior =
                    (i >= 1 && i <= NR - 2 && j >= 1 && j <= NTH - 2);

                double du = 0.0;
                double dv = 0.0;
                double dw = 0.0;

                // u/v are prognostic at k = 0 (SCIENCE.md M19).
                if (h_interior && k >= 0 && k <= NZ - 2)
                {
                    const double dp_dx =
                        (static_cast<double>(p_in(i + 1, j, k))
                         - static_cast<double>(p_in(i, j, k))) / dr;
                    const double rho_face =
                        0.5 * (static_cast<double>(rho_in(i, j, k))
                             + static_cast<double>(rho_in(i + 1, j, k)));
                    du = -dp_dx / safe_rho_ref(rho_face);
                    if (!std::isfinite(du)) du = 0.0;

                    const double dp_dy =
                        (static_cast<double>(p_in(i, j + 1, k))
                         - static_cast<double>(p_in(i, j, k))) / dtheta;
                    const double rho_face_y =
                        0.5 * (static_cast<double>(rho_in(i, j, k))
                             + static_cast<double>(rho_in(i, j + 1, k)));
                    dv = -dp_dy / safe_rho_ref(rho_face_y);
                    if (!std::isfinite(dv)) dv = 0.0;
                }

                if (h_interior && k >= 0 && k <= NZ - 2)
                {
                    const double dp_dz =
                        (static_cast<double>(p_in(i, j, k + 1))
                         - static_cast<double>(p_in(i, j, k))) / dz;
                    const double dp0_dz =
                        (static_cast<double>(p0_f[static_cast<std::size_t>(k + 1)])
                         - static_cast<double>(p0_f[static_cast<std::size_t>(k)])) / dz;
                    const double rho_face =
                        0.5 * (static_cast<double>(rho_in(i, j, k))
                             + static_cast<double>(rho_in(i, j, k + 1)));
                    dw = -(dp_dz - dp0_dz) / safe_rho_ref(rho_face);
                    if (!std::isfinite(dw)) dw = 0.0;
                }

                u_out(i, j, k) = clamp_wind_horizontal_ms(static_cast<float>(
                    static_cast<double>(u_in(i, j, k)) + du * dt_small));
                v_out(i, j, k) = clamp_wind_horizontal_ms(static_cast<float>(
                    static_cast<double>(v_in(i, j, k)) + dv * dt_small));
                w_out(i, j, k) = clamp_wind_vertical_ms(static_cast<float>(
                    static_cast<double>(w_in(i, j, k)) + dw * dt_small));
            }
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

}  // namespace

TEST_CASE("Cartesian C-grid lateral walls close grid-relative velocity",
          "[dynamics][cartesian][cgrid][boundary][domain-motion][SCI-195]")
{
    setup_cartesian_cgrid_grid(/*nx=*/8, /*ny=*/7, /*nz=*/5,
                               /*dx=*/100.0, /*dy=*/120.0, /*dz=*/80.0,
                               /*dt=*/0.2);
    using LateralBC = tmv::init::config::LateralBC;
    global_lateral_bc_west = LateralBC::RigidSlip;
    global_lateral_bc_east = LateralBC::RigidSlip;
    global_lateral_bc_south = LateralBC::RigidSlip;
    global_lateral_bc_north = LateralBC::RigidSlip;
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
    global_domain_motion_mode =
        tmv::init::config::DomainMotionMode::ConstantVelocity;
    global_domain_frame.velocity_ms = {6.5, -2.25};

    u.resize(NR, NTH, NZ, 9.0f);
    v.resize(NR, NTH, NZ, -4.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    rho.resize(NR, NTH, NZ, 1.0f);
    p.resize(NR, NTH, NZ, 90000.0f);

    auto boundary = create_cartesian_cgrid_bc_scheme();
    boundary->apply_full();
    const int i = NR / 2;
    const int j = NTH / 2;
    const int k = NZ / 2;
    REQUIRE(u[0][j][k] == 6.5f);
    REQUIRE(u[NR - 2][j][k] == 6.5f);
    REQUIRE(u[NR - 1][j][k] == 6.5f);
    REQUIRE(v[0][j][k] == -4.0f);
    REQUIRE(v[NR - 1][j][k] == -4.0f);
    REQUIRE(v[i][0][k] == -2.25f);
    REQUIRE(v[i][NTH - 2][k] == -2.25f);
    REQUIRE(v[i][NTH - 1][k] == -2.25f);
    REQUIRE(u[i][0][k] == 9.0f);
    REQUIRE(u[i][NTH - 1][k] == 9.0f);

    global_lateral_bc_west = LateralBC::Rigid;
    global_lateral_bc_east = LateralBC::Rigid;
    global_lateral_bc_south = LateralBC::Rigid;
    global_lateral_bc_north = LateralBC::Rigid;
    boundary->apply_acoustic(dt);
    REQUIRE(u[0][j][k] == 6.5f);
    REQUIRE(v[0][j][k] == -2.25f);
    REQUIRE(u[i][0][k] == 6.5f);
    REQUIRE(v[i][0][k] == -2.25f);
}

TEST_CASE("Cartesian C-grid radiative phase speed is grid relative",
          "[dynamics][cartesian][cgrid][boundary][radiative][domain-motion][SCI-195]")
{
    setup_cartesian_cgrid_grid(/*nx=*/9, /*ny=*/8, /*nz=*/6,
                               /*dx=*/100.0, /*dy=*/120.0, /*dz=*/80.0,
                               /*dt=*/0.2);
    using LateralBC = tmv::init::config::LateralBC;
    global_lateral_bc_west = LateralBC::Radiative;
    global_lateral_bc_east = LateralBC::Radiative;
    global_lateral_bc_south = LateralBC::Radiative;
    global_lateral_bc_north = LateralBC::Radiative;
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;

    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    rho.resize(NR, NTH, NZ, 0.0f);
    p.resize(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[i][j][k] = static_cast<float>(
                    5.0 + 0.08 * i - 0.03 * j + 0.01 * k);
                v[i][j][k] = static_cast<float>(
                    -2.0 + 0.02 * i + 0.07 * j - 0.015 * k);
                w[i][j][k] = static_cast<float>(0.01 * i - 0.02 * j);
                rho[i][j][k] = static_cast<float>(
                    1.0 + 0.005 * i - 0.002 * j);
                p[i][j][k] = static_cast<float>(
                    90000.0 + 2.0 * i + 3.0 * j - 4.0 * k);
            }
    const Field3D initial_u = u;
    const Field3D initial_v = v;
    const Field3D initial_w = w;
    const Field3D initial_rho = rho;
    const Field3D initial_p = p;

    auto stationary = create_cartesian_cgrid_bc_scheme();
    stationary->apply_acoustic(0.0);
    stationary->seed_acoustic_radiation();
    stationary->apply_acoustic(dt);
    const Field3D reference_u = u;
    const Field3D reference_v = v;
    const Field3D reference_w = w;
    const Field3D reference_rho = rho;
    const Field3D reference_p = p;

    constexpr double frame_u = 6.5;
    constexpr double frame_v = -3.25;
    u = initial_u;
    v = initial_v;
    w = initial_w;
    rho = initial_rho;
    p = initial_p;
    for (std::size_t index = 0; index < u.size(); ++index)
    {
        u.data()[index] = static_cast<float>(
            static_cast<double>(u.data()[index]) + frame_u);
        v.data()[index] = static_cast<float>(
            static_cast<double>(v.data()[index]) + frame_v);
    }
    global_domain_motion_mode =
        tmv::init::config::DomainMotionMode::ConstantVelocity;
    global_domain_frame.velocity_ms = {frame_u, frame_v};
    auto translated = create_cartesian_cgrid_bc_scheme();
    translated->apply_acoustic(0.0);
    translated->seed_acoustic_radiation();
    translated->apply_acoustic(dt);

    for (std::size_t index = 0; index < u.size(); ++index)
    {
        REQUIRE(static_cast<double>(u.data()[index]) - frame_u
                == Approx(reference_u.data()[index]).margin(2.0e-6));
        REQUIRE(static_cast<double>(v.data()[index]) - frame_v
                == Approx(reference_v.data()[index]).margin(2.0e-6));
        REQUIRE(w.data()[index] == reference_w.data()[index]);
        REQUIRE(rho.data()[index] == reference_rho.data()[index]);
        REQUIRE(p.data()[index] == reference_p.data()[index]);
    }
}

TEST_CASE("Cartesian C-grid acoustic GPU parity (GPU audit Phase 3.1)",
          "[vulkan][gpu_parity][cgrid][acoustic][cartesian]")
{
    setup_cartesian_cgrid_grid(/*nx=*/16, /*ny=*/16, /*nz=*/16,
                               /*dx=*/100.0, /*dy=*/100.0, /*dz=*/250.0,
                               /*dt=*/1.0e-3);

    if (!try_init_gpu())
    {
        WARN("GPU unavailable -- skipping Cartesian C-grid acoustic parity");
        return;
    }

    auto* backend = mutable_compute_backend();
    auto* kernels = backend != nullptr ? backend->dynamics_kernels() : nullptr;
    if (kernels == nullptr
        || !kernels->supports(DynamicsKernelOp::AcousticPressureCartesianCgrid)
        || !kernels->supports(DynamicsKernelOp::AcousticMomentumCartesianCgrid))
    {
        if (physical_gpu_required())
        {
            shutdown_backend();
            FAIL("required Vulkan device lacks Cartesian C-grid acoustic "
                 "pressure or momentum support");
        }
        WARN("GPU does not support Cartesian C-grid acoustic -- skipping");
        shutdown_backend();
        return;
    }

    std::vector<float> rho0_f, p0_f, dp0_dz_f, theta0_f;
    build_base_profiles(rho0_f, p0_f, dp0_dz_f, theta0_f);
    Field3D theta_field(NR, NTH, NZ);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                theta_field(i, j, k) = theta0_f[static_cast<std::size_t>(k)];

    const float gamma_f  = static_cast<float>(kGamma);
    const float dx_f     = static_cast<float>(dr);
    const float dy_f     = static_cast<float>(dtheta);
    const float dz_f     = static_cast<float>(dz);
    const float dt_f     = static_cast<float>(dt);
    const float wind_h   = wind_horizontal_abs_max_ms;
    const float wind_v   = wind_vertical_abs_max_ms;

    SECTION("pressure: hydrostatic identity (zero winds, base-state rho/p)")
    {
        Field3D uf(NR, NTH, NZ, 0.0f), vf(NR, NTH, NZ, 0.0f), wf(NR, NTH, NZ, 0.0f);
        Field3D rho_field(NR, NTH, NZ), p_field(NR, NTH, NZ);
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    rho_field(i, j, k) = rho0_f[static_cast<std::size_t>(k)];
                    p_field(i, j, k)   = p0_f[static_cast<std::size_t>(k)];
                }

        Field3D rho_gpu(NR, NTH, NZ, 0.0f), p_gpu(NR, NTH, NZ, 0.0f);
        REQUIRE(kernels->dispatch_acoustic_pressure_cartesian_cgrid(
            AcousticPressureCartesianCgridDispatch{
            uf.data(), vf.data(), wf.data(),
            rho_field.data(), p_field.data(),
            rho0_f.data(), p0_f.data(), NZ,
            theta_field.data(),
            rho_gpu.data(), p_gpu.data(),
            NR, NTH, NZ, dx_f, dy_f, dz_f,
            gamma_f, dt_f,
            density_min_kgm3, density_max_kgm3,
            pressure_min_pa, pressure_max_pa, kPRef
            }));

        // Zero divergence and rho == rho0 exactly: the integration is a
        // passthrough. The full-EOS diagnosis (SCIENCE.md C1)
        // reproduces p0 through pow() at the dry-EOS-consistent base
        // state, exact to a few ulp of 1e5 Pa in fp32 (the involution is
        // exact in real arithmetic; bit-exactness was a property of the
        // retired linear closure only).
        constexpr auto kFull = nimbus::parity::ComparisonDomain::FullField;
        require_parity("identity rho", rho_field, rho_gpu, 0.0, kFull);
        require_parity("identity p", p_field, p_gpu, 5.0e-2, kFull);
    }

    SECTION("pressure: Gaussian flow + rho perturbation matches CPU reference")
    {
        Field3D uf(NR, NTH, NZ, 0.0f), vf(NR, NTH, NZ, 0.0f), wf(NR, NTH, NZ, 0.0f);
        Field3D rho_field(NR, NTH, NZ), p_field(NR, NTH, NZ);
        for (int i = 0; i < NR; ++i)
        {
            const double di = (static_cast<double>(i) - 8.0) / 3.0;
            const double bump = std::exp(-0.5 * di * di);
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    uf(i, j, k) = static_cast<float>(5.0 * bump);
                    rho_field(i, j, k) =
                        rho0_f[static_cast<std::size_t>(k)]
                        + static_cast<float>(0.01 * bump);
                    p_field(i, j, k) = p0_f[static_cast<std::size_t>(k)];
                }
        }

        Field3D rho_gpu(NR, NTH, NZ, 0.0f), p_gpu(NR, NTH, NZ, 0.0f);
        REQUIRE(kernels->dispatch_acoustic_pressure_cartesian_cgrid(
            AcousticPressureCartesianCgridDispatch{
            uf.data(), vf.data(), wf.data(),
            rho_field.data(), p_field.data(),
            rho0_f.data(), p0_f.data(), NZ,
            theta_field.data(),
            rho_gpu.data(), p_gpu.data(),
            NR, NTH, NZ, dx_f, dy_f, dz_f,
            gamma_f, dt_f,
            density_min_kgm3, density_max_kgm3,
            pressure_min_pa, pressure_max_pa, kPRef
            }));

        Field3D rho_cpu, p_cpu;
        cpu_reference_pressure(uf, vf, wf, rho_field, p_field, theta_field,
                               rho0_f, p0_f, dt, rho_cpu, p_cpu);

        // Sanity: the kernel produced motion (drho ~ rho*du/dx ~ 0.05 kg/m3/s
        // over dt=1e-3 -> ~5e-5 kg/m3; the diagnosis amplifies into p by
        // cs0^2 ~ 1.2e5 -> ~5 Pa).
        double max_motion_p = 0.0;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                    max_motion_p = std::max(max_motion_p,
                        std::fabs(static_cast<double>(p_gpu(i, j, k))
                                - static_cast<double>(p_field(i, j, k))));
        INFO("kernel made motion: max|p_gpu - p_in| = " << max_motion_p);
        REQUIRE(max_motion_p > 0.5);

        constexpr auto kFull = nimbus::parity::ComparisonDomain::FullField;
        require_parity("pressure rho", rho_cpu, rho_gpu, 1.0e-6, kFull);
        require_parity("pressure p", p_cpu, p_gpu, 0.5, kFull);
    }

    SECTION("pressure and fused acoustics use grid-relative horizontal velocity")
    {
        constexpr float frame_u = 17.25f;
        constexpr float frame_v = -8.5f;
        Field3D u_base(NR, NTH, NZ, 0.0f);
        Field3D v_base(NR, NTH, NZ, 0.0f);
        Field3D w_base(NR, NTH, NZ, 0.0f);
        Field3D rho_base(NR, NTH, NZ, 0.0f);
        Field3D p_base(NR, NTH, NZ, 0.0f);
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const float x = static_cast<float>(i) / (NR - 1);
                    const float y = static_cast<float>(j) / (NTH - 1);
                    u_base(i, j, k) = 3.0f + 2.0f * x;
                    v_base(i, j, k) = -1.0f + 1.5f * y;
                    w_base(i, j, k) = 0.05f * static_cast<float>(k);
                    rho_base(i, j, k) =
                        rho0_f[static_cast<std::size_t>(k)]
                        + 0.01f * x * y;
                    p_base(i, j, k) = p0_f[static_cast<std::size_t>(k)];
                }

        Field3D u_translated = u_base;
        Field3D v_translated = v_base;
        for (std::size_t index = 0; index < u_base.size(); ++index)
        {
            u_translated.data()[index] += frame_u;
            v_translated.data()[index] += frame_v;
        }

        Field3D rho_stationary(NR, NTH, NZ, 0.0f);
        Field3D p_stationary(NR, NTH, NZ, 0.0f);
        Field3D rho_translated(NR, NTH, NZ, 0.0f);
        Field3D p_translated(NR, NTH, NZ, 0.0f);
        REQUIRE(kernels->dispatch_acoustic_pressure_cartesian_cgrid(
            AcousticPressureCartesianCgridDispatch{
                u_base.data(), v_base.data(), w_base.data(),
                rho_base.data(), p_base.data(), rho0_f.data(), p0_f.data(), NZ,
                theta_field.data(), rho_stationary.data(), p_stationary.data(),
                NR, NTH, NZ, dx_f, dy_f, dz_f, gamma_f, dt_f,
                density_min_kgm3, density_max_kgm3,
                pressure_min_pa, pressure_max_pa, kPRef, 0.0f, 0.0f}));
        REQUIRE(kernels->dispatch_acoustic_pressure_cartesian_cgrid(
            AcousticPressureCartesianCgridDispatch{
                u_translated.data(), v_translated.data(), w_base.data(),
                rho_base.data(), p_base.data(), rho0_f.data(), p0_f.data(), NZ,
                theta_field.data(), rho_translated.data(), p_translated.data(),
                NR, NTH, NZ, dx_f, dy_f, dz_f, gamma_f, dt_f,
                density_min_kgm3, density_max_kgm3,
                pressure_min_pa, pressure_max_pa, kPRef, frame_u, frame_v}));

        constexpr auto kFull = nimbus::parity::ComparisonDomain::FullField;
        require_parity("translated pressure rho", rho_stationary,
                       rho_translated, 2.0e-7, kFull);
        require_parity("translated pressure p", p_stationary,
                       p_translated, 3.0e-2, kFull);

        if (kernels->supports(
                DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid))
        {
            Field3D u_fused_stationary = u_base;
            Field3D v_fused_stationary = v_base;
            Field3D w_fused_stationary = w_base;
            Field3D rho_fused_stationary = rho_base;
            Field3D p_fused_stationary = p_base;
            AcousticSubstepFusedCartesianCgridDispatch stationary_dispatch{
                u_fused_stationary.data(), v_fused_stationary.data(),
                w_fused_stationary.data(), rho_fused_stationary.data(),
                p_fused_stationary.data(), theta_field.data(),
                rho0_f.data(), p0_f.data(), dp0_dz_f.data(), NZ,
                NR, NTH, NZ, dx_f, dy_f, dz_f, gamma_f, dt_f,
                density_min_kgm3, density_max_kgm3,
                pressure_min_pa, pressure_max_pa, kPRef, wind_h, wind_v};
            REQUIRE(kernels->dispatch_acoustic_substep_fused_cartesian_cgrid(
                stationary_dispatch));

            Field3D u_fused_translated = u_translated;
            Field3D v_fused_translated = v_translated;
            Field3D w_fused_translated = w_base;
            Field3D rho_fused_translated = rho_base;
            Field3D p_fused_translated = p_base;
            AcousticSubstepFusedCartesianCgridDispatch translated_dispatch{
                u_fused_translated.data(), v_fused_translated.data(),
                w_fused_translated.data(), rho_fused_translated.data(),
                p_fused_translated.data(), theta_field.data(),
                rho0_f.data(), p0_f.data(), dp0_dz_f.data(), NZ,
                NR, NTH, NZ, dx_f, dy_f, dz_f, gamma_f, dt_f,
                density_min_kgm3, density_max_kgm3,
                pressure_min_pa, pressure_max_pa, kPRef, wind_h, wind_v};
            translated_dispatch.frame_velocity_u_ms = frame_u;
            translated_dispatch.frame_velocity_v_ms = frame_v;
            REQUIRE(kernels->dispatch_acoustic_substep_fused_cartesian_cgrid(
                translated_dispatch));

            for (std::size_t index = 0; index < u_base.size(); ++index)
            {
                u_fused_translated.data()[index] -= frame_u;
                v_fused_translated.data()[index] -= frame_v;
            }
            require_parity("translated fused u", u_fused_stationary,
                           u_fused_translated, 2.0e-6, kFull);
            require_parity("translated fused v", v_fused_stationary,
                           v_fused_translated, 2.0e-6, kFull);
            require_parity("translated fused w", w_fused_stationary,
                           w_fused_translated, 2.0e-6, kFull);
            require_parity("translated fused rho", rho_fused_stationary,
                           rho_fused_translated, 2.0e-7, kFull);
            require_parity("translated fused p", p_fused_stationary,
                           p_fused_translated, 3.0e-2, kFull);
        }

        if (kernels->supports(
                DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid))
        {
            Field3D u_batch_stationary = u_base;
            Field3D v_batch_stationary = v_base;
            Field3D w_batch_stationary = w_base;
            Field3D rho_batch_stationary = rho_base;
            Field3D p_batch_stationary = p_base;
            REQUIRE(kernels->dispatch_acoustic_substeps_batched_cartesian_cgrid(
                AcousticSubstepsBatchedCartesianCgridDispatch{
                    u_batch_stationary.data(), v_batch_stationary.data(),
                    w_batch_stationary.data(), rho_batch_stationary.data(),
                    p_batch_stationary.data(), theta_field.data(),
                    rho0_f.data(), p0_f.data(), dp0_dz_f.data(), NZ,
                    NR, NTH, NZ, dx_f, dy_f, dz_f, gamma_f, dt_f, 1,
                    density_min_kgm3, density_max_kgm3,
                    pressure_min_pa, pressure_max_pa, kPRef, wind_h, wind_v,
                    2, 2, 2, 2, 1, 0}));

            Field3D u_batch_translated = u_translated;
            Field3D v_batch_translated = v_translated;
            Field3D w_batch_translated = w_base;
            Field3D rho_batch_translated = rho_base;
            Field3D p_batch_translated = p_base;
            AcousticSubstepsBatchedCartesianCgridDispatch batch_dispatch{
                u_batch_translated.data(), v_batch_translated.data(),
                w_batch_translated.data(), rho_batch_translated.data(),
                p_batch_translated.data(), theta_field.data(),
                rho0_f.data(), p0_f.data(), dp0_dz_f.data(), NZ,
                NR, NTH, NZ, dx_f, dy_f, dz_f, gamma_f, dt_f, 1,
                density_min_kgm3, density_max_kgm3,
                pressure_min_pa, pressure_max_pa, kPRef, wind_h, wind_v,
                2, 2, 2, 2, 1, 0};
            batch_dispatch.frame_velocity_u_ms = frame_u;
            batch_dispatch.frame_velocity_v_ms = frame_v;
            REQUIRE(kernels->dispatch_acoustic_substeps_batched_cartesian_cgrid(
                batch_dispatch));

            for (std::size_t index = 0; index < u_base.size(); ++index)
            {
                u_batch_translated.data()[index] -= frame_u;
                v_batch_translated.data()[index] -= frame_v;
            }
            require_parity("translated batched boundary u", u_batch_stationary,
                           u_batch_translated, 2.0e-6, kFull);
            require_parity("translated batched boundary v", v_batch_stationary,
                           v_batch_translated, 2.0e-6, kFull);
            require_parity("translated batched boundary w", w_batch_stationary,
                           w_batch_translated, 2.0e-6, kFull);
            require_parity("translated batched boundary rho",
                           rho_batch_stationary, rho_batch_translated,
                           2.0e-7, kFull);
            require_parity("translated batched boundary p", p_batch_stationary,
                           p_batch_translated, 3.0e-2, kFull);
        }
    }

    SECTION("momentum: hydrostatic identity (p equals p0_base exactly)")
    {
        Field3D rho_field(NR, NTH, NZ), p_field(NR, NTH, NZ);
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    rho_field(i, j, k) = rho0_f[static_cast<std::size_t>(k)];
                    p_field(i, j, k)   = p0_f[static_cast<std::size_t>(k)];
                }
        Field3D u_in(NR, NTH, NZ, 3.0f), v_in(NR, NTH, NZ, -2.0f), w_in(NR, NTH, NZ, 0.5f);
        Field3D u_gpu(NR, NTH, NZ, 0.0f), v_gpu(NR, NTH, NZ, 0.0f), w_gpu(NR, NTH, NZ, 0.0f);

        REQUIRE(kernels->dispatch_acoustic_momentum_cartesian_cgrid(
            AcousticMomentumCartesianCgridDispatch{
            rho_field.data(), p_field.data(),
            dp0_dz_f.data(), NZ,
            u_in.data(), v_in.data(), w_in.data(),
            u_gpu.data(), v_gpu.data(), w_gpu.data(),
            NR, NTH, NZ, dx_f, dy_f, dz_f,
            dt_f, wind_h, wind_v
            }));

        // p uniform horizontally: the u/v face gradients difference
        // identical floats, which is exact in any conforming float
        // implementation -> bit-exact passthrough.
        constexpr auto kFull = nimbus::parity::ComparisonDomain::FullField;
        require_parity("identity u", u_in, u_gpu, 0.0, kFull);
        require_parity("identity v", v_in, v_gpu, 0.0, kFull);
        // The w face runs the perturbation-p subtraction against the
        // k-varying base profile; Vulkan does not guarantee bit-exact
        // float arithmetic vs the CPU (MoltenVK/Metal may lower the
        // expression with fused/contracted ops), and the observed
        // residual is exactly 1 ulp of w = 0.5. Gate at 1 ulp: a real
        // stencil bug (wrong face, missing p0 subtraction) produces
        // O(1e-3 m/s) here, four orders above this gate.
        require_parity("identity w", w_in, w_gpu, 6.0e-8, kFull);

        // Determinism IS guaranteed: the same dispatch on the same input
        // must reproduce bit-exactly (catches cross-thread races in the
        // in-place buffer design).
        Field3D u_gpu2(NR, NTH, NZ, 0.0f), v_gpu2(NR, NTH, NZ, 0.0f), w_gpu2(NR, NTH, NZ, 0.0f);
        REQUIRE(kernels->dispatch_acoustic_momentum_cartesian_cgrid(
            AcousticMomentumCartesianCgridDispatch{
            rho_field.data(), p_field.data(),
            dp0_dz_f.data(), NZ,
            u_in.data(), v_in.data(), w_in.data(),
            u_gpu2.data(), v_gpu2.data(), w_gpu2.data(),
            NR, NTH, NZ, dx_f, dy_f, dz_f,
            dt_f, wind_h, wind_v
            }));
        require_parity("deterministic u", u_gpu, u_gpu2, 0.0, kFull);
        require_parity("deterministic v", v_gpu, v_gpu2, 0.0, kFull);
        require_parity("deterministic w", w_gpu, w_gpu2, 0.0, kFull);
    }

    SECTION("momentum: pressure bump accelerations match CPU reference")
    {
        Field3D rho_field(NR, NTH, NZ), p_field(NR, NTH, NZ);
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double di = (static_cast<double>(i) - 8.0) / 2.5;
                    const double dj = (static_cast<double>(j) - 8.0) / 2.5;
                    const double dk = (static_cast<double>(k) - 8.0) / 2.5;
                    const double bump =
                        std::exp(-0.5 * (di * di + dj * dj + dk * dk));
                    rho_field(i, j, k) = rho0_f[static_cast<std::size_t>(k)];
                    p_field(i, j, k) = p0_f[static_cast<std::size_t>(k)]
                                     + static_cast<float>(100.0 * bump);
                }
        Field3D u_in(NR, NTH, NZ, 0.0f), v_in(NR, NTH, NZ, 0.0f), w_in(NR, NTH, NZ, 0.0f);
        Field3D u_gpu(NR, NTH, NZ, 0.0f), v_gpu(NR, NTH, NZ, 0.0f), w_gpu(NR, NTH, NZ, 0.0f);

        REQUIRE(kernels->dispatch_acoustic_momentum_cartesian_cgrid(
            AcousticMomentumCartesianCgridDispatch{
            rho_field.data(), p_field.data(),
            dp0_dz_f.data(), NZ,
            u_in.data(), v_in.data(), w_in.data(),
            u_gpu.data(), v_gpu.data(), w_gpu.data(),
            NR, NTH, NZ, dx_f, dy_f, dz_f,
            dt_f, wind_h, wind_v
            }));

        Field3D u_cpu, v_cpu, w_cpu;
        cpu_reference_momentum(rho_field, p_field, p0_f,
                               u_in, v_in, w_in, dt,
                               u_cpu, v_cpu, w_cpu);

        // Sanity: gradient of a 100 Pa bump over 2.5 cells of 100 m at
        // rho ~ 1.2 -> |du/dt| ~ 0.3 m/s^2 -> ~3e-4 m/s over dt = 1e-3.
        double max_motion_u = 0.0;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 1; k < NZ - 1; ++k)
                    max_motion_u = std::max(max_motion_u,
                        std::fabs(static_cast<double>(u_gpu(i, j, k))));
        INFO("kernel made motion: max|u_gpu| = " << max_motion_u);
        REQUIRE(max_motion_u > 1.0e-5);

        constexpr auto kFull = nimbus::parity::ComparisonDomain::FullField;
        require_parity("momentum u", u_cpu, u_gpu, 1.0e-5, kFull);
        require_parity("momentum v", v_cpu, v_gpu, 1.0e-5, kFull);
        require_parity("momentum w", w_cpu, w_gpu, 1.0e-5, kFull);
    }

    SECTION("fused: composite substep matches sequential CPU passes")
    {
        if (!kernels->supports(DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid))
        {
            if (physical_gpu_required())
                FAIL("required Vulkan device lacks fused Cartesian C-grid "
                     "acoustic support");
            WARN("fused Cartesian C-grid dispatch unavailable -- skipping");
        }
        else
        {
            Field3D uf(NR, NTH, NZ, 0.0f), vf(NR, NTH, NZ, 0.0f), wf(NR, NTH, NZ, 0.0f);
            Field3D rho_field(NR, NTH, NZ), p_field(NR, NTH, NZ);
            for (int i = 0; i < NR; ++i)
            {
                const double di = (static_cast<double>(i) - 8.0) / 3.0;
                const double bump = std::exp(-0.5 * di * di);
                for (int j = 0; j < NTH; ++j)
                    for (int k = 0; k < NZ; ++k)
                    {
                        uf(i, j, k) = static_cast<float>(5.0 * bump);
                        rho_field(i, j, k) =
                            rho0_f[static_cast<std::size_t>(k)]
                            + static_cast<float>(0.01 * bump);
                        p_field(i, j, k) = p0_f[static_cast<std::size_t>(k)];
                    }
            }

            // GPU fused: in-place on copies.
            Field3D u_g = uf, v_g = vf, w_g = wf;
            Field3D rho_g = rho_field, p_g = p_field;
            REQUIRE(kernels->dispatch_acoustic_substep_fused_cartesian_cgrid(
                AcousticSubstepFusedCartesianCgridDispatch{
                u_g.data(), v_g.data(), w_g.data(),
                rho_g.data(), p_g.data(),
                theta_field.data(),
                rho0_f.data(), p0_f.data(), dp0_dz_f.data(), NZ,
                NR, NTH, NZ, dx_f, dy_f, dz_f,
                gamma_f, dt_f,
                density_min_kgm3, density_max_kgm3,
                pressure_min_pa, pressure_max_pa, kPRef,
                wind_h, wind_v
                }));

            // CPU sequence: pressure pass, then momentum reads the
            // UPDATED rho/p (the forward-backward ordering).
            Field3D rho_cpu, p_cpu;
            cpu_reference_pressure(uf, vf, wf, rho_field, p_field, theta_field,
                                   rho0_f, p0_f, dt, rho_cpu, p_cpu);
            Field3D u_cpu, v_cpu, w_cpu;
            cpu_reference_momentum(rho_cpu, p_cpu, p0_f,
                                   uf, vf, wf, dt,
                                   u_cpu, v_cpu, w_cpu);

            constexpr auto kFull = nimbus::parity::ComparisonDomain::FullField;
            require_parity("fused rho", rho_cpu, rho_g, 1.0e-6, kFull);
            require_parity("fused p", p_cpu, p_g, 0.5, kFull);
            require_parity("fused u", u_cpu, u_g, 1.0e-5, kFull);
            require_parity("fused v", v_cpu, v_g, 1.0e-5, kFull);
            require_parity("fused w", w_cpu, w_g, 1.0e-5, kFull);

            // Direct owner audit: run the report-bearing CPU fused operator
            // from the identical five-field input, then attribute the actual
            // shader endpoint through the same public helper used by
            // production. This compares operator-time requests as well as
            // endpoint inventory changes.
            Field3D qv_owner(NR, NTH, NZ, 0.0f);
            Field3D qc_owner(NR, NTH, NZ, 0.0f);
            Field3D qr_owner(NR, NTH, NZ, 0.0f);
            Field3D qi_owner(NR, NTH, NZ, 0.0f);
            Field3D qs_owner(NR, NTH, NZ, 0.0f);
            Field3D qg_owner(NR, NTH, NZ, 0.0f);
            Field3D qh_owner(NR, NTH, NZ, 0.0f);
            Field3D tke_owner(NR, NTH, NZ, 0.0f);
            for (int i = 0; i < NR; ++i)
                for (int j = 0; j < NTH; ++j)
                    for (int k = 0; k < NZ; ++k)
                    {
                        const double x = static_cast<double>(i) / (NR - 1);
                        const double y = static_cast<double>(j) / (NTH - 1);
                        const double z = static_cast<double>(k) / (NZ - 1);
                        const double moist_bump = std::exp(
                            -8.0 * ((x - 0.48) * (x - 0.48)
                                   + (y - 0.52) * (y - 0.52)
                                   + (z - 0.6) * (z - 0.6)));
                        qv_owner(i, j, k) =
                            static_cast<float>(0.004 + 0.003 * moist_bump);
                        qc_owner(i, j, k) =
                            static_cast<float>(2.0e-5 + 3.0e-4 * moist_bump);
                        qr_owner(i, j, k) =
                            static_cast<float>(1.0e-5 + 1.0e-4 * moist_bump);
                        qi_owner(i, j, k) =
                            static_cast<float>(5.0e-6 + 8.0e-5 * moist_bump);
                        qs_owner(i, j, k) =
                            static_cast<float>(3.0e-6 + 5.0e-5 * moist_bump);
                        qg_owner(i, j, k) =
                            static_cast<float>(2.0e-6 + 3.0e-5 * moist_bump);
                        qh_owner(i, j, k) =
                            static_cast<float>(1.0e-6 + 2.0e-5 * moist_bump);
                        tke_owner(i, j, k) =
                            static_cast<float>(0.05 + 0.02 * moist_bump);
                    }
            std::vector<double> rho0_d(rho0_f.begin(), rho0_f.end());
            std::vector<double> p0_d(p0_f.begin(), p0_f.end());
            Field3D cpu_u = uf, cpu_v = vf, cpu_w = wf;
            Field3D cpu_rho = rho_field, cpu_p = p_field;
            Field3D cpu_drho(NR, NTH, NZ, 0.0f);
            CartesianCGridAcousticEnergyReportV1 cpu_report;
            AcousticSubstepFusedCartesianCgridDispatch cpu_dispatch;
            cpu_dispatch.u = cpu_u.data();
            cpu_dispatch.v = cpu_v.data();
            cpu_dispatch.w = cpu_w.data();
            cpu_dispatch.rho = cpu_rho.data();
            cpu_dispatch.p = cpu_p.data();
            cpu_dispatch.theta_data = theta_field.data();
            cpu_dispatch.rho0_base_data = rho0_f.data();
            cpu_dispatch.p0_base_data = p0_f.data();
            cpu_dispatch.dp0_dz_data = dp0_dz_f.data();
            cpu_dispatch.base_len = NZ;
            cpu_dispatch.nx = NR;
            cpu_dispatch.ny = NTH;
            cpu_dispatch.nz = NZ;
            cpu_dispatch.gamma_val = gamma_f;
            cpu_dispatch.dt_small = dt_f;
            cpu_dispatch.rho_min = density_min_kgm3;
            cpu_dispatch.rho_max = density_max_kgm3;
            cpu_dispatch.p_min = pressure_min_pa;
            cpu_dispatch.p_max = pressure_max_pa;
            cpu_dispatch.p_ref_fallback = kPRef;
            cpu_dispatch.wind_clamp_h = wind_h;
            cpu_dispatch.wind_clamp_v = wind_v;
            cpu_dispatch.rho0_base_host = rho0_d.data();
            cpu_dispatch.p0_base_host = p0_d.data();
            cpu_dispatch.dx_host = dr;
            cpu_dispatch.dy_host = dtheta;
            cpu_dispatch.dz_host = dz;
            cpu_dispatch.dt_small_host = dt;
            cpu_dispatch.gamma_host = kGamma;
            cpu_dispatch.drho_scratch = cpu_drho.data();
            cpu_dispatch.qv_host = qv_owner.data();
            cpu_dispatch.qc_host = qc_owner.data();
            cpu_dispatch.qr_host = qr_owner.data();
            cpu_dispatch.qi_host = qi_owner.data();
            cpu_dispatch.qs_host = qs_owner.data();
            cpu_dispatch.qg_host = qg_owner.data();
            cpu_dispatch.qh_host = qh_owner.data();
            cpu_dispatch.tke_host = tke_owner.data();
            cpu_dispatch.energy_report_host = &cpu_report;
            auto cpu_backend = create_cpu_compute_backend();
            std::string cpu_error;
            REQUIRE(cpu_backend->initialize(cpu_error));
            REQUIRE(cpu_backend->dynamics_kernels()
                ->dispatch_acoustic_substep_fused_cartesian_cgrid(
                    cpu_dispatch));

            require_parity("owner rho", cpu_rho, rho_g, 1.0e-6, kFull);
            require_parity("owner p", cpu_p, p_g, 0.5, kFull);
            require_parity("owner u", cpu_u, u_g, 1.0e-5, kFull);
            require_parity("owner v", cpu_v, v_g, 1.0e-5, kFull);
            require_parity("owner w", cpu_w, w_g, 1.0e-5, kFull);

            GridStorageState grid_state(
                NR, NTH, NZ, dr, dz, dt, dtheta);
            grid_state.geometry.initialize(
                NR, NTH, NZ, dr, dz, dtheta,
                CoordinateSystem::Cartesian, StaggerType::CGrid);
            SimulationStorageState gpu_storage(std::move(grid_state));
            gpu_storage.fields.u = u_g;
            gpu_storage.fields.v = v_g;
            gpu_storage.fields.w = w_g;
            gpu_storage.fields.rho = rho_g;
            gpu_storage.fields.p = p_g;
            gpu_storage.fields.theta = theta_field;
            gpu_storage.fields.qv = qv_owner;
            gpu_storage.fields.qc = qc_owner;
            gpu_storage.fields.qr = qr_owner;
            gpu_storage.fields.qi = qi_owner;
            gpu_storage.fields.qs = qs_owner;
            gpu_storage.fields.qg = qg_owner;
            gpu_storage.fields.qh = qh_owner;
            gpu_storage.fields.tke = tke_owner;
            gpu_storage.base.rho0 = rho0_d;
            gpu_storage.base.p0 = p0_d;
            auto gpu_state = make_simulation_state_view(gpu_storage);
            CartesianCGridAcousticEnergyReportV1 gpu_report;
            tmv::device_acoustic_energy::
                attribute_flat_cgrid_fused_commit(
                    gpu_state, uf, vf, wf, rho_field, p_field,
                    dt_f, dx_f, dy_f, dz_f, gpu_report);

            require_acoustic_owner_parity(cpu_report, gpu_report);

            // The production helper is intentionally flat/uniform. A mapped
            // or stretched geometry must fail rather than silently combine
            // the flat shader equation with different physical volumes.
            gpu_storage.grid.geometry.z_faces[1] += 1.0;
            auto nonuniform_state =
                make_simulation_state_view(gpu_storage);
            CartesianCGridAcousticEnergyReportV1 rejected_report;
            REQUIRE_THROWS_AS(
                tmv::device_acoustic_energy::
                    attribute_flat_cgrid_fused_commit(
                        nonuniform_state, uf, vf, wf, rho_field, p_field,
                        dt_f, dx_f, dy_f, dz_f, rejected_report),
                std::invalid_argument);
            gpu_storage.grid.geometry.z_faces[1] -= 1.0;
            cpu_backend->shutdown();
        }
    }

    SECTION("batched pressure exports the continuity face-mass average")
    {
        if (!kernels->supports(
                DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid)
            || !kernels->supports_batched_cartesian_cgrid_mass_flux_export())
        {
            if (physical_gpu_required())
                FAIL("required Vulkan device lacks batched Cartesian C-grid "
                     "mass-flux export support");
            WARN("batched Cartesian C-grid mass-flux export unavailable -- skipping");
        }
        else
        {
            constexpr double pi = 3.14159265358979323846;
            constexpr float frame_u = 17.25f;
            constexpr float frame_v = -8.5f;
            Field3D u_g(NR, NTH, NZ, 0.0f);
            Field3D v_g(NR, NTH, NZ, 0.0f);
            Field3D w_g(NR, NTH, NZ, 0.0f);
            Field3D rho_g(NR, NTH, NZ, 0.0f);
            Field3D p_g(NR, NTH, NZ, 0.0f);
            for (int i = 0; i < NR; ++i)
                for (int j = 0; j < NTH; ++j)
                    for (int k = 0; k < NZ; ++k)
                    {
                        const double x = static_cast<double>(i) / (NR - 1);
                        const double y = static_cast<double>(j) / (NTH - 1);
                        const double z = static_cast<double>(k) / (NZ - 1);
                        u_g(i, j, k) = frame_u
                            + static_cast<float>(5.0 * std::sin(2.0 * pi * x));
                        v_g(i, j, k) = frame_v
                            + static_cast<float>(-3.0 * std::cos(2.0 * pi * y));
                        w_g(i, j, k) = static_cast<float>(0.7 * std::sin(pi * z));
                        rho_g(i, j, k) = rho0_f[static_cast<std::size_t>(k)]
                            + static_cast<float>(0.01 * x * y);
                        p_g(i, j, k) = p0_f[static_cast<std::size_t>(k)];
                    }

            const Field3D u_before = u_g;
            const Field3D v_before = v_g;
            const Field3D w_before = w_g;
            const Field3D rho_before = rho_g;
            Field3D expected_x(NR, NTH, NZ, 0.0f);
            Field3D expected_y(NR, NTH, NZ, 0.0f);
            Field3D expected_z(NR, NTH, NZ, 0.0f);
            for (int i = 0; i < NR; ++i)
                for (int j = 0; j < NTH; ++j)
                    for (int k = 0; k < NZ; ++k)
                    {
                        if (i + 1 < NR)
                            expected_x(i, j, k) = static_cast<float>(
                                0.5 * (static_cast<double>(rho_before(i, j, k))
                                     + static_cast<double>(rho_before(i + 1, j, k)))
                                * (static_cast<double>(u_before(i, j, k))
                                   - frame_u));
                        if (j + 1 < NTH)
                            expected_y(i, j, k) = static_cast<float>(
                                0.5 * (static_cast<double>(rho_before(i, j, k))
                                     + static_cast<double>(rho_before(i, j + 1, k)))
                                * (static_cast<double>(v_before(i, j, k))
                                   - frame_v));
                        if (k + 1 < NZ)
                            expected_z(i, j, k) = static_cast<float>(
                                0.5 * (static_cast<double>(rho_before(i, j, k))
                                     + static_cast<double>(rho_before(i, j, k + 1)))
                                * static_cast<double>(w_before(i, j, k)));
                    }

            Field3D mavg_x(NR, NTH, NZ, 0.0f);
            Field3D mavg_y(NR, NTH, NZ, 0.0f);
            Field3D mavg_z(NR, NTH, NZ, 0.0f);
            REQUIRE(kernels->dispatch_acoustic_substeps_batched_cartesian_cgrid(
                AcousticSubstepsBatchedCartesianCgridDispatch{
                    u_g.data(), v_g.data(), w_g.data(), rho_g.data(), p_g.data(),
                    theta_field.data(),
                    rho0_f.data(), p0_f.data(), dp0_dz_f.data(), NZ,
                    NR, NTH, NZ, dx_f, dy_f, dz_f,
                    gamma_f, dt_f, 1,
                    density_min_kgm3, density_max_kgm3,
                    pressure_min_pa, pressure_max_pa, kPRef,
                    wind_h, wind_v,
                    2, 2, 2, 2, 0, 0,
                    mavg_x.data(), mavg_y.data(), mavg_z.data(), true,
                    frame_u, frame_v
                }));

            constexpr auto kFull = nimbus::parity::ComparisonDomain::FullField;
            require_parity("batched mass flux x", expected_x, mavg_x, 2.0e-6, kFull);
            require_parity("batched mass flux y", expected_y, mavg_y, 2.0e-6, kFull);
            require_parity("batched mass flux z", expected_z, mavg_z, 2.0e-6, kFull);
        }
    }

    SECTION("separate attributed lifecycle matches batched KW78 trajectory")
    {
        if (!kernels->supports(
                DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid))
        {
            if (physical_gpu_required())
                FAIL("required Vulkan device lacks batched Cartesian C-grid "
                     "acoustic support");
            WARN("batched Cartesian C-grid acoustics unavailable -- skipping "
                 "SCI-162 lifecycle parity");
        }
        else
        {
            using LateralBC = tmv::init::config::LateralBC;
            using BottomBC = tmv::init::config::BottomBC;
            using TopBC = tmv::init::config::TopBC;
            AcousticBoundaryConfigGuard config_guard;
            global_lateral_bc_west = LateralBC::Radiative;
            global_lateral_bc_east = LateralBC::Radiative;
            global_lateral_bc_south = LateralBC::Radiative;
            global_lateral_bc_north = LateralBC::Radiative;
            global_bottom_bc = BottomBC::RigidSlip;
            global_top_bc = TopBC::Rigid;

            u.resize(NR, NTH, NZ, 0.0f);
            v.resize(NR, NTH, NZ, 0.0f);
            w.resize(NR, NTH, NZ, 0.0f);
            rho.resize(NR, NTH, NZ, 0.0f);
            p.resize(NR, NTH, NZ, 0.0f);
            qv.resize(NR, NTH, NZ, 0.0f);
            qc.resize(NR, NTH, NZ, 0.0f);
            qr.resize(NR, NTH, NZ, 0.0f);
            qi.resize(NR, NTH, NZ, 0.0f);
            qs.resize(NR, NTH, NZ, 0.0f);
            qg.resize(NR, NTH, NZ, 0.0f);
            qh.resize(NR, NTH, NZ, 0.0f);
            constexpr double pi = 3.14159265358979323846;
            for (int i = 0; i < NR; ++i)
                for (int j = 0; j < NTH; ++j)
                    for (int k = 0; k < NZ; ++k)
                    {
                        const double x =
                            static_cast<double>(i) / (NR - 1);
                        const double y =
                            static_cast<double>(j) / (NTH - 1);
                        const double z =
                            static_cast<double>(k) / (NZ - 1);
                        u(i, j, k) = static_cast<float>(
                            8.0 + 0.7 * std::sin(2.0 * pi * x)
                            + 0.2 * y);
                        v(i, j, k) = static_cast<float>(
                            -5.0 + 0.5 * std::cos(2.0 * pi * y)
                            - 0.15 * x);
                        w(i, j, k) = static_cast<float>(
                            0.25 * std::sin(pi * z));
                        rho(i, j, k) =
                            rho0_f[static_cast<std::size_t>(k)]
                            + static_cast<float>(
                                0.003 * std::sin(pi * x)
                                * std::sin(pi * y));
                        p(i, j, k) =
                            p0_f[static_cast<std::size_t>(k)]
                            + static_cast<float>(
                                25.0 * std::sin(pi * x)
                                * std::sin(pi * y)
                                * std::sin(pi * z));
                    }

            const Field3D initial_u = u;
            const Field3D initial_v = v;
            const Field3D initial_w = w;
            const Field3D initial_rho = rho;
            const Field3D initial_p = p;
            auto boundary = create_cartesian_cgrid_bc_scheme();
            CartesianCGridAcousticEnergyReportV1 boundary_report;
            boundary->apply_acoustic_attributed(0.0, boundary_report);
            boundary->seed_acoustic_radiation();

            Field3D u_batched = u;
            Field3D v_batched = v;
            Field3D w_batched = w;
            Field3D rho_batched = rho;
            Field3D p_batched = p;
            constexpr int n_substeps = 4;
            REQUIRE(kernels->dispatch_acoustic_substeps_batched_cartesian_cgrid(
                AcousticSubstepsBatchedCartesianCgridDispatch{
                    u_batched.data(), v_batched.data(), w_batched.data(),
                    rho_batched.data(), p_batched.data(),
                    theta_field.data(),
                    rho0_f.data(), p0_f.data(), dp0_dz_f.data(), NZ,
                    NR, NTH, NZ, dx_f, dy_f, dz_f,
                    gamma_f, dt_f, n_substeps,
                    density_min_kgm3, density_max_kgm3,
                    pressure_min_pa, pressure_max_pa, kPRef,
                    wind_h, wind_v,
                    static_cast<int>(LateralBC::Radiative),
                    static_cast<int>(LateralBC::Radiative),
                    static_cast<int>(LateralBC::Radiative),
                    static_cast<int>(LateralBC::Radiative),
                    static_cast<int>(BottomBC::RigidSlip),
                    static_cast<int>(TopBC::Rigid)
                }));

            Field3D rho_next(NR, NTH, NZ, 0.0f);
            Field3D p_next(NR, NTH, NZ, 0.0f);
            for (int substep = 0; substep < n_substeps; ++substep)
            {
                REQUIRE(kernels->dispatch_acoustic_pressure_cartesian_cgrid(
                    AcousticPressureCartesianCgridDispatch{
                        u.data(), v.data(), w.data(),
                        rho.data(), p.data(),
                        rho0_f.data(), p0_f.data(), NZ,
                        theta_field.data(),
                        rho_next.data(), p_next.data(),
                        NR, NTH, NZ, dx_f, dy_f, dz_f,
                        gamma_f, dt_f,
                        density_min_kgm3, density_max_kgm3,
                        pressure_min_pa, pressure_max_pa, kPRef
                    }));
                rho = rho_next;
                p = p_next;
                REQUIRE(kernels->dispatch_acoustic_momentum_cartesian_cgrid(
                    AcousticMomentumCartesianCgridDispatch{
                        rho.data(), p.data(),
                        dp0_dz_f.data(), NZ,
                        u.data(), v.data(), w.data(),
                        u.data(), v.data(), w.data(),
                        NR, NTH, NZ, dx_f, dy_f, dz_f,
                        dt_f, wind_h, wind_v
                    }));
                boundary->apply_acoustic_attributed(
                    static_cast<double>(dt_f), boundary_report);
            }

            constexpr auto kFull =
                nimbus::parity::ComparisonDomain::FullField;
            require_parity(
                "SCI-162 separate/batched u", u_batched, u, 2.0e-5, kFull);
            require_parity(
                "SCI-162 separate/batched v", v_batched, v, 2.0e-5, kFull);
            require_parity(
                "SCI-162 separate/batched w", w_batched, w, 2.0e-5, kFull);
            require_parity(
                "SCI-162 separate/batched rho",
                rho_batched, rho, 2.0e-6, kFull);
            require_parity(
                "SCI-162 separate/batched p",
                p_batched, p, 0.5, kFull);
            REQUIRE(boundary_report.boundary_complete);
            REQUIRE(
                boundary_report.boundary_application_count
                == static_cast<std::uint64_t>(n_substeps + 1));
            REQUIRE(
                boundary_report.boundary_radiative_committed_work_j
                == Approx(
                    boundary_report
                        .boundary_radiative_requested_linear_work_j
                    + boundary_report
                        .boundary_radiative_finite_step_correction_j
                    + boundary_report
                        .boundary_radiative_storage_rounding_j)
                       .epsilon(2.0e-12));
            REQUIRE(
                boundary_report.boundary_committed_work_j
                == Approx(
                    boundary_report.boundary_radiative_committed_work_j
                    + boundary_report
                        .boundary_constraint_history_committed_work_j)
                       .epsilon(2.0e-12));
            REQUIRE(std::abs(
                boundary_report
                    .boundary_radiative_requested_linear_work_j) > 1.0);

            // Repeat the same attributed lifecycle through the production
            // CPU acoustic kernels and an independently seeded boundary
            // object. The boundary scheme is host-side for both execution
            // paths, but its work is conditioned by the acoustic endpoint;
            // comparing reports therefore closes the actual mixed
            // Vulkan-acoustic/CPU-boundary ownership seam.
            const Field3D gpu_separate_u = u;
            const Field3D gpu_separate_v = v;
            const Field3D gpu_separate_w = w;
            const Field3D gpu_separate_rho = rho;
            const Field3D gpu_separate_p = p;
            u = initial_u;
            v = initial_v;
            w = initial_w;
            rho = initial_rho;
            p = initial_p;
            auto cpu_boundary = create_cartesian_cgrid_bc_scheme();
            CartesianCGridAcousticEnergyReportV1 cpu_boundary_report;
            cpu_boundary->apply_acoustic_attributed(
                0.0, cpu_boundary_report);
            cpu_boundary->seed_acoustic_radiation();
            auto cpu_backend = create_cpu_compute_backend();
            std::string cpu_error;
            REQUIRE(cpu_backend->initialize(cpu_error));
            auto* cpu_kernels = cpu_backend->dynamics_kernels();
            REQUIRE(cpu_kernels != nullptr);
            std::vector<double> cpu_rho0(
                rho0_f.begin(), rho0_f.end());
            std::vector<double> cpu_p0(
                p0_f.begin(), p0_f.end());
            Field3D cpu_drho_scratch(NR, NTH, NZ, 0.0f);
            for (int substep = 0; substep < n_substeps; ++substep)
            {
                AcousticSubstepFusedCartesianCgridDispatch cpu_dispatch;
                cpu_dispatch.u = u.data();
                cpu_dispatch.v = v.data();
                cpu_dispatch.w = w.data();
                cpu_dispatch.rho = rho.data();
                cpu_dispatch.p = p.data();
                cpu_dispatch.theta_data = theta_field.data();
                cpu_dispatch.rho0_base_data = rho0_f.data();
                cpu_dispatch.p0_base_data = p0_f.data();
                cpu_dispatch.dp0_dz_data = dp0_dz_f.data();
                cpu_dispatch.base_len = NZ;
                cpu_dispatch.nx = NR;
                cpu_dispatch.ny = NTH;
                cpu_dispatch.nz = NZ;
                cpu_dispatch.dx = dx_f;
                cpu_dispatch.dy = dy_f;
                cpu_dispatch.dz = dz_f;
                cpu_dispatch.gamma_val = gamma_f;
                cpu_dispatch.dt_small = dt_f;
                cpu_dispatch.rho_min = density_min_kgm3;
                cpu_dispatch.rho_max = density_max_kgm3;
                cpu_dispatch.p_min = pressure_min_pa;
                cpu_dispatch.p_max = pressure_max_pa;
                cpu_dispatch.p_ref_fallback = kPRef;
                cpu_dispatch.wind_clamp_h = wind_h;
                cpu_dispatch.wind_clamp_v = wind_v;
                cpu_dispatch.rho0_base_host = cpu_rho0.data();
                cpu_dispatch.p0_base_host = cpu_p0.data();
                cpu_dispatch.dx_host = dr;
                cpu_dispatch.dy_host = dtheta;
                cpu_dispatch.dz_host = dz;
                cpu_dispatch.dt_small_host = dt;
                cpu_dispatch.gamma_host = kGamma;
                cpu_dispatch.drho_scratch =
                    cpu_drho_scratch.data();
                REQUIRE(cpu_kernels
                    ->dispatch_acoustic_substep_fused_cartesian_cgrid(
                        cpu_dispatch));
                cpu_boundary->apply_acoustic_attributed(
                    static_cast<double>(dt_f), cpu_boundary_report);
            }
            require_parity(
                "boundary owner CPU/Vulkan u",
                u, gpu_separate_u, 2.0e-5, kFull);
            require_parity(
                "boundary owner CPU/Vulkan v",
                v, gpu_separate_v, 2.0e-5, kFull);
            require_parity(
                "boundary owner CPU/Vulkan w",
                w, gpu_separate_w, 2.0e-5, kFull);
            require_parity(
                "boundary owner CPU/Vulkan rho",
                rho, gpu_separate_rho, 2.0e-6, kFull);
            require_parity(
                "boundary owner CPU/Vulkan p",
                p, gpu_separate_p, 0.5, kFull);
            require_boundary_owner_parity(
                cpu_boundary_report, boundary_report);
            cpu_backend->shutdown();
        }
    }

    shutdown_backend();
}
