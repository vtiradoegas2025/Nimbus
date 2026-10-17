#include "dynamics/moving_cgrid_transport.hpp"
#include "numerics/advection/advection_cartesian.hpp"
/**
 * @file test_cartesian_cgrid_slow_tendencies_gpu_parity.cpp
 * @brief CPU/GPU parity for the Cartesian C-grid slow tendencies.
 *
 * Mirrors test_supercell_cgrid_slow_tendencies_gpu_parity.cpp but for the
 * Cartesian C-grid scheme: no 1/r curvature, no axis ghost, and UPWIND-biased
 * momentum advection (Wicker-Skamarock) rather than centered. Compares the
 * CPU compute_slow_tendencies against the GPU dispatch on identical inputs --
 * an isolated check that the cartesian_cgrid_slow_tendencies.comp kernel
 * matches the CPU math (within fp32-vs-fp64 drift), with no full-sim or other
 * GPU-kernel confounds.
 */

#include "catch2/catch.hpp"

#include "compute/compute_backend.hpp"
#include "compute/compute_backend_factory.hpp"
#include "compute/compute_kernel_template.hpp"
#include "compute/parity_check.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/field/field3d.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/grid_geometry.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "dynamics/acoustic_energy_report.hpp"
#include "dynamics/device_acoustic_energy_attribution.hpp"
#include "dynamics/device_slow_energy_attribution.hpp"
#include "dynamics/schemes/klemp_wilhelmson_split/cartesian_cgrid_3d.hpp"
#include "core/orchestration/dynamics/thermodynamic_closure.hpp"
#include "numerics/runtime.hpp"
#include "terrain/base/topography.hpp"
#include "terrain/conservative_transport.hpp"

#include <cstdlib>
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kG  = 9.80665;

void setup_cart_grid(int nx, int ny, int nz, double dx_m, double dy_m, double dz_m, double dt_s)
{
    NR     = nx;   // NX
    NTH    = ny;   // NY
    NZ     = nz;
    dr     = dx_m; // dx
    dtheta = dy_m; // dy
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

    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    rho.resize(NR, NTH, NZ, 1.0f);
    p.resize(NR, NTH, NZ, 100000.0f);
    theta.resize(NR, NTH, NZ, 300.0f);

    // Deterministic Coriolis state (M1): off unless a test enables it.
    coriolis_f = 0.0;
    u0_base.clear();
    v0_base.clear();
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
                "required physical Vulkan slow-path qualification could not "
                "initialize a device: " + device_error);
        }
        return false;
    }
    return true;
}

void shutdown_backend() { shutdown_compute_backend_runtime(); }

struct SlowApplyObserverProbe
{
    const float* old_u = nullptr;
    const float* old_v = nullptr;
    const float* old_w = nullptr;
    const float* old_p = nullptr;
    std::vector<float> committed_u;
    std::vector<float> committed_v;
    std::vector<float> committed_w;
    std::vector<float> committed_p;
    std::size_t count = 0;
    bool called = false;
    bool old_host_untouched = false;
};

struct SlowEnergyObserverContext
{
    const SimulationStateView* state = nullptr;
    const Field3D* du = nullptr;
    const Field3D* dv = nullptr;
    const Field3D* dw = nullptr;
    const SlowMomentumAdvectionTendencyOutputs* advection = nullptr;
    float dt_large = 0.0f;
    tmv::device_slow_energy::AttributionResult result;
    bool called = false;
};

void observe_slow_energy(
    void* raw,
    const float* committed_u,
    const float* committed_v,
    const float* committed_w,
    const float* committed_p)
{
    auto& context = *static_cast<SlowEnergyObserverContext*>(raw);
    context.result =
        tmv::device_slow_energy::attribute_completed_commit(
            *context.state,
            *context.du,
            *context.dv,
            *context.dw,
            context.advection,
            context.dt_large,
            /*attribute_momentum=*/true,
            /*attribute_pbl=*/true,
            /*wrap_x=*/false,
            /*wrap_y=*/false,
            committed_u,
            committed_v,
            committed_w,
            committed_p);
    context.called = true;
}

tmv::device_slow_energy::AttributionResult
reference_flat_cgrid_slow_commit(
    const SimulationStateView& state,
    const Field3D& du,
    const Field3D& dv,
    const Field3D& dw,
    const Field3D& dp,
    const SlowMomentumAdvectionTendencyOutputs& advection,
    double dt_large,
    Field3D& committed_u,
    Field3D& committed_v,
    Field3D& committed_w,
    Field3D& committed_p)
{
    const auto& fields = state.fields;
    const auto& tendencies = state.tendencies;
    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    const double volume =
        state.grid.dr * state.grid.dtheta * state.grid.dz;

    committed_u = fields.u;
    committed_v = fields.v;
    committed_w = fields.w;
    committed_p = fields.p;
    for (int i = 0; i < state.grid.nr; ++i)
        for (int j = 0; j < state.grid.nth; ++j)
            for (int k = 0; k < state.grid.nz; ++k)
            {
                committed_u[i][j][k] = static_cast<float>(
                    static_cast<double>(fields.u[i][j][k])
                    + dt_large * (
                        static_cast<double>(du[i][j][k])
                        + static_cast<double>(
                            tendencies.du_dt_pbl[i][j][k])));
                committed_v[i][j][k] = static_cast<float>(
                    static_cast<double>(fields.v[i][j][k])
                    + dt_large * (
                        static_cast<double>(dv[i][j][k])
                        + static_cast<double>(
                            tendencies.dv_dt_pbl[i][j][k])));
                committed_w[i][j][k] = static_cast<float>(
                    static_cast<double>(fields.w[i][j][k])
                    + dt_large * static_cast<double>(dw[i][j][k]));
                committed_p[i][j][k] = static_cast<float>(
                    static_cast<double>(fields.p[i][j][k])
                    + dt_large * static_cast<double>(dp[i][j][k]));
            }

    std::array<double, 11> sums{};
    const auto moist_mass = [&](int i, int j, int k)
    {
        const double q_total =
            static_cast<double>(fields.qv[i][j][k])
            + static_cast<double>(fields.qc[i][j][k])
            + static_cast<double>(fields.qr[i][j][k])
            + static_cast<double>(fields.qi[i][j][k])
            + static_cast<double>(fields.qs[i][j][k])
            + static_cast<double>(fields.qg[i][j][k])
            + static_cast<double>(fields.qh[i][j][k]);
        return static_cast<double>(fields.rho[i][j][k])
            * volume * (1.0 + q_total);
    };
    const auto dry_mass = [&](int i, int j, int k)
    {
        return static_cast<double>(fields.rho[i][j][k]) * volume;
    };
    const auto accumulate_momentum =
        [&](double mass, double old_velocity, double new_velocity,
            double total_tendency, double advective_tendency)
    {
        const double delta_advection =
            dt_large * advective_tendency;
        const double delta_other =
            dt_large * (total_tendency - advective_tendency);
        sums[0] += mass * old_velocity * delta_advection;
        sums[1] += mass * old_velocity * delta_other;
        sums[2] += 0.5 * mass
            * delta_advection * delta_advection;
        sums[3] += 0.5 * mass * delta_other * delta_other;
        sums[4] += mass * delta_advection * delta_other;
        sums[5] += 0.5 * mass
            * (new_velocity * new_velocity
               - old_velocity * old_velocity);
    };

    constexpr double cv_d =
        physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double c_l =
        physical_constants::specific_heat_liquid_water_jkgk;
    constexpr double c_i =
        physical_constants::specific_heat_ice_jkgk;
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;

    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double cell_mass = moist_mass(i, j, k);
                double u_mass = 0.5 * cell_mass;
                if (i + 1 < domain.i_end)
                    u_mass += 0.5 * moist_mass(i + 1, j, k);
                double v_mass = 0.5 * cell_mass;
                if (j + 1 < domain.j_end)
                    v_mass += 0.5 * moist_mass(i, j + 1, k);
                double w_mass = 0.5 * cell_mass;
                if (k + 1 < domain.k_end)
                    w_mass += 0.5 * moist_mass(i, j, k + 1);
                const double total_u =
                    static_cast<double>(du[i][j][k])
                    + static_cast<double>(
                        tendencies.du_dt_pbl[i][j][k]);
                const double total_v =
                    static_cast<double>(dv[i][j][k])
                    + static_cast<double>(
                        tendencies.dv_dt_pbl[i][j][k]);
                const double total_w = dw[i][j][k];
                accumulate_momentum(
                    u_mass, fields.u[i][j][k],
                    committed_u[i][j][k], total_u,
                    (*advection.du_dt)[i][j][k]);
                accumulate_momentum(
                    v_mass, fields.v[i][j][k],
                    committed_v[i][j][k], total_v,
                    (*advection.dv_dt)[i][j][k]);
                accumulate_momentum(
                    w_mass, fields.w[i][j][k],
                    committed_w[i][j][k], total_w,
                    (*advection.dw_dt)[i][j][k]);

                const double p_old = fields.p[i][j][k];
                const double p_new = committed_p[i][j][k];
                const double theta_k = fields.theta[i][j][k];
                const double old_temperature = theta_k * std::pow(
                    p_old / physical_constants::reference_pressure_pa,
                    kappa);
                const double new_temperature = theta_k * std::pow(
                    p_new / physical_constants::reference_pressure_pa,
                    kappa);
                const double delta_temperature =
                    new_temperature - old_temperature;
                const double q_liquid =
                    static_cast<double>(fields.qc[i][j][k])
                    + fields.qr[i][j][k];
                const double q_ice =
                    static_cast<double>(fields.qi[i][j][k])
                    + fields.qs[i][j][k]
                    + fields.qg[i][j][k]
                    + fields.qh[i][j][k];
                sums[6] += dry_mass(i, j, k) * cv_d
                    * delta_temperature;
                sums[7] += dry_mass(i, j, k)
                    * (fields.qv[i][j][k] * cv_v
                       + q_liquid * c_l + q_ice * c_i)
                    * delta_temperature;

                const double du_dynamics = du[i][j][k];
                const double dv_dynamics = dv[i][j][k];
                const double du_pbl =
                    tendencies.du_dt_pbl[i][j][k];
                const double dv_pbl =
                    tendencies.dv_dt_pbl[i][j][k];
                if (i < domain.i_end - 1)
                {
                    const double mass = 0.5 * (
                        dry_mass(i, j, k)
                        + dry_mass(i + 1, j, k));
                    const double after_dynamics =
                        static_cast<double>(fields.u[i][j][k])
                        + dt_large * du_dynamics;
                    const double delta = dt_large * du_pbl;
                    const double stored_dynamics =
                        static_cast<float>(after_dynamics);
                    const double stored_total =
                        committed_u[i][j][k];
                    sums[8] += mass * after_dynamics * delta;
                    sums[9] += 0.5 * mass * delta * delta;
                    sums[10] += 0.5 * mass
                        * (stored_total * stored_total
                           - stored_dynamics * stored_dynamics);
                }
                if (j < domain.j_end - 1)
                {
                    const double mass = 0.5 * (
                        dry_mass(i, j, k)
                        + dry_mass(i, j + 1, k));
                    const double after_dynamics =
                        static_cast<double>(fields.v[i][j][k])
                        + dt_large * dv_dynamics;
                    const double delta = dt_large * dv_pbl;
                    const double stored_dynamics =
                        static_cast<float>(after_dynamics);
                    const double stored_total =
                        committed_v[i][j][k];
                    sums[8] += mass * after_dynamics * delta;
                    sums[9] += 0.5 * mass * delta * delta;
                    sums[10] += 0.5 * mass
                        * (stored_total * stored_total
                           - stored_dynamics * stored_dynamics);
                }
            }

    tmv::device_slow_energy::AttributionResult result;
    auto& report = result.momentum;
    report.advection_requested_linear_work_j = sums[0];
    report.nonadvection_requested_linear_work_j = sums[1];
    report.advection_finite_step_correction_j = sums[2];
    report.nonadvection_finite_step_correction_j = sums[3];
    report.coupled_finite_step_cross_work_j = sums[4];
    report.committed_kinetic_change_j = sums[5];
    const double requested =
        sums[0] + sums[1] + sums[2] + sums[3] + sums[4];
    report.storage_rounding_j =
        report.committed_kinetic_change_j - requested;
    report.closure_residual_j =
        report.committed_kinetic_change_j
        - (requested + report.storage_rounding_j);
    report.pressure_dry_internal_change_j = sums[6];
    report.pressure_water_sensible_change_j = sums[7];
    report.committed_pressure_energy_change_j = sums[6] + sums[7];
    report.complete = true;
    result.pbl.requested_linear_work_j = sums[8];
    result.pbl.finite_step_correction_j = sums[9];
    result.pbl.committed_work_j = sums[10];
    result.pbl.storage_rounding_j =
        sums[10] - sums[8] - sums[9];
    return result;
}

void require_conditioned_close(
    const char* term,
    double cpu,
    double gpu,
    double parent_scale = 0.0)
{
    const double scale = std::max({
        1.0, std::abs(cpu), std::abs(gpu), std::abs(parent_scale),
    });
    const double tolerance =
        96.0 * std::numeric_limits<float>::epsilon() * scale;
    INFO(term << ": cpu=" << cpu << " gpu=" << gpu
         << " abs=" << std::abs(cpu - gpu)
         << " tolerance=" << tolerance);
    REQUIRE(std::isfinite(cpu));
    REQUIRE(std::isfinite(gpu));
    REQUIRE(std::abs(cpu - gpu) <= tolerance);
}

void observe_slow_apply(
    void* raw,
    const float* committed_u,
    const float* committed_v,
    const float* committed_w,
    const float* committed_p)
{
    auto& probe = *static_cast<SlowApplyObserverProbe*>(raw);
    probe.called = true;
    probe.old_host_untouched =
        probe.old_u[0] == 1.0f
        && probe.old_v[0] == -2.0f
        && probe.old_w[0] == 0.5f
        && probe.old_p[0] == 90000.0f;
    probe.committed_u.assign(committed_u, committed_u + probe.count);
    probe.committed_v.assign(committed_v, committed_v + probe.count);
    probe.committed_w.assign(committed_w, committed_w + probe.count);
    probe.committed_p.assign(committed_p, committed_p + probe.count);
}

void require_parity(const char* field, const Field3D& cpu, const Field3D& gpu,
                    double abs_tolerance,
                    nimbus::parity::ComparisonDomain domain)
{
    const auto result = nimbus::parity::compare_field(
        cpu.data(), gpu.data(), NR, NTH, NZ, {0.0, abs_tolerance}, domain);
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

TEST_CASE(
    "device acoustic carrier request reproduces the FP32 continuity equation",
    "[dynamics][cgrid][cartesian][device-energy][SCI-158]")
{
    constexpr int nx = 5;
    constexpr int ny = 5;
    constexpr int nz = 3;
    constexpr int i = 2;
    constexpr int j = 2;
    constexpr int k = 1;
    constexpr float dt_small = 0.25f;
    constexpr float dx = 10.0f;
    constexpr float dy = 20.0f;
    constexpr float dz = 5.0f;
    Field3D rho(nx, ny, nz, 2.0f);
    Field3D u(nx, ny, nz, 0.0f);
    Field3D v(nx, ny, nz, 0.0f);
    Field3D w(nx, ny, nz, 0.0f);
    u(i, j, k) = 4.0f;
    u(i - 1, j, k) = 1.0f;
    v(i, j, k) = -1.0f;
    v(i, j - 1, k) = 2.0f;
    w(i, j, k) = 0.5f;
    w(i, j, k - 1) = -0.5f;

    const float requested =
        tmv::device_acoustic_energy::requested_density_change(
            rho, u, v, w, nx, ny, nz, i, j, k,
            dt_small, dx, dy, dz);
    REQUIRE(requested == Approx(-0.175f));

    // The shader converts a non-finite assembled divergence to exact zero
    // before applying the timestep.
    u(i, j, k) = std::numeric_limits<float>::infinity();
    REQUIRE(
        tmv::device_acoustic_energy::requested_density_change(
            rho, u, v, w, nx, ny, nz, i, j, k,
            dt_small, dx, dy, dz)
        == 0.0f);
}

TEST_CASE("Cartesian C-grid open top enters conservative continuity",
          "[dynamics][cartesian][cgrid][continuity][open-top][SCI-042]")
{
    setup_cart_grid(/*nx=*/6, /*ny=*/5, /*nz=*/4,
                    /*dx=*/100.0, /*dy=*/80.0, /*dz=*/50.0, /*dt=*/0.25);
    rho0_base.assign(NZ, 1.2);
    p0_base.assign(NZ, 90000.0);
    theta0_base.assign(NZ, 300.0);
    qv0_base.assign(NZ, 0.0);
    rho.resize(NR, NTH, NZ, 1.2f);
    p.resize(NR, NTH, NZ, 90000.0f);
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            w[i][j][NZ - 1] = 0.4f;

    Field3D drho_dt(NR, NTH, NZ, 0.0f);
    Field3D dp_dt(NR, NTH, NZ, 0.0f);
    CartesianCGrid3DScheme scheme;
    scheme.compute_fast_pressure_tendencies(
        u, v, w, rho, p, dt, drho_dt, dp_dt);

    const double expected_top = -1.2 * 0.4 / dz;
    REQUIRE(drho_dt[2][2][NZ - 1]
            == Approx(expected_top).margin(2.0e-9));
    REQUIRE(drho_dt[2][2][NZ - 2] == Approx(0.0));
    long double total_tendency_kg_s = 0.0L;
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 0; k < NZ; ++k)
                total_tendency_kg_s += static_cast<long double>(
                    drho_dt[i][j][k]) * dr * dtheta * dz;
    const double expected_export_kg_s =
        static_cast<double>((NR - 2) * (NTH - 2))
        * dr * dtheta * 1.2 * 0.4;
    REQUIRE(static_cast<double>(total_tendency_kg_s)
            == Approx(-expected_export_kg_s).epsilon(3.0e-7));
}

TEST_CASE("Cartesian C-grid CPU tendencies are invariant under a constant translating frame",
          "[dynamics][cartesian][cgrid][domain-motion][SCI-195]")
{
    setup_cart_grid(/*nx=*/14, /*ny=*/13, /*nz=*/11,
                    /*dx=*/400.0, /*dy=*/350.0, /*dz=*/250.0, /*dt=*/0.1);
    shutdown_compute_backend_runtime();
    coriolis_f = 0.0;
    rho0_base.resize(NZ);
    p0_base.resize(NZ);
    for (int k = 0; k < NZ; ++k)
    {
        rho0_base[k] = 1.15 * std::exp(-static_cast<double>(k) * dz / 8500.0);
        p0_base[k] = 100000.0 * std::exp(-static_cast<double>(k) * dz / 8500.0);
    }
    qv0_base.clear();
    qv.resize(0, 0, 0);
    qc.resize(0, 0, 0);

    Field3D u_stationary(NR, NTH, NZ, 0.0f);
    Field3D v_stationary(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double wave = std::sin(0.29 * i)
                                  * std::cos(0.33 * j);
                rho[i][j][k] = static_cast<float>(
                    rho0_base[k] * (1.0 + 0.003 * wave));
                p[i][j][k] = static_cast<float>(
                    p0_base[k] + 22.0 * wave + 0.15 * i * j);
                theta[i][j][k] = static_cast<float>(300.0 + 0.03 * k);
                u_stationary[i][j][k] = static_cast<float>(
                    4.0 + 0.07 * i - 0.035 * j + 0.012 * k);
                v_stationary[i][j][k] = static_cast<float>(
                    -2.0 + 0.025 * i + 0.065 * j - 0.009 * k);
                w[i][j][k] = static_cast<float>(
                    0.04 * std::sin(0.21 * k) * wave);
            }

    Field3D du_stationary(NR, NTH, NZ, 0.0f);
    Field3D dv_stationary(NR, NTH, NZ, 0.0f);
    Field3D dw_stationary(NR, NTH, NZ, 0.0f);
    Field3D drho_slow_stationary(NR, NTH, NZ, 0.0f);
    Field3D dp_slow_stationary(NR, NTH, NZ, 0.0f);
    Field3D drho_fast_stationary(NR, NTH, NZ, 0.0f);
    Field3D dp_fast_stationary(NR, NTH, NZ, 0.0f);
    CartesianCGrid3DScheme stationary_scheme;
    stationary_scheme.compute_slow_tendencies(
        u_stationary, v_stationary, w, rho, p, theta, dt,
        du_stationary, dv_stationary, dw_stationary,
        drho_slow_stationary, dp_slow_stationary);
    stationary_scheme.compute_fast_pressure_tendencies(
        u_stationary, v_stationary, w, rho, p, dt,
        drho_fast_stationary, dp_fast_stationary);

    constexpr double frame_u = 8.75;
    constexpr double frame_v = -3.625;
    Field3D u_translated = u_stationary;
    Field3D v_translated = v_stationary;
    for (std::size_t index = 0; index < u_translated.size(); ++index)
    {
        u_translated.data()[index] = static_cast<float>(
            static_cast<double>(u_translated.data()[index]) + frame_u);
        v_translated.data()[index] = static_cast<float>(
            static_cast<double>(v_translated.data()[index]) + frame_v);
    }
    global_domain_motion_mode =
        tmv::init::config::DomainMotionMode::ConstantVelocity;
    global_domain_frame.velocity_ms = {frame_u, frame_v};

    Field3D du_translated(NR, NTH, NZ, 0.0f);
    Field3D dv_translated(NR, NTH, NZ, 0.0f);
    Field3D dw_translated(NR, NTH, NZ, 0.0f);
    Field3D drho_slow_translated(NR, NTH, NZ, 0.0f);
    Field3D dp_slow_translated(NR, NTH, NZ, 0.0f);
    Field3D drho_fast_translated(NR, NTH, NZ, 0.0f);
    Field3D dp_fast_translated(NR, NTH, NZ, 0.0f);
    CartesianCGrid3DScheme translated_scheme;
    translated_scheme.compute_slow_tendencies(
        u_translated, v_translated, w, rho, p, theta, dt,
        du_translated, dv_translated, dw_translated,
        drho_slow_translated, dp_slow_translated);
    translated_scheme.compute_fast_pressure_tendencies(
        u_translated, v_translated, w, rho, p, dt,
        drho_fast_translated, dp_fast_translated);

    const auto require_same = [](const Field3D& reference,
                                 const Field3D& candidate,
                                 double margin)
    {
        for (int i = 3; i < NR - 3; ++i)
            for (int j = 3; j < NTH - 3; ++j)
                for (int k = 2; k < NZ - 2; ++k)
                    REQUIRE(candidate[i][j][k]
                            == Approx(reference[i][j][k]).margin(margin));
    };
    require_same(du_stationary, du_translated, 2.0e-7);
    require_same(dv_stationary, dv_translated, 2.0e-7);
    require_same(dw_stationary, dw_translated, 2.0e-7);
    require_same(drho_slow_stationary, drho_slow_translated, 0.0);
    require_same(dp_slow_stationary, dp_slow_translated, 2.0e-3);
    require_same(drho_fast_stationary, drho_fast_translated, 2.0e-9);
    require_same(dp_fast_stationary, dp_fast_translated, 2.0e-3);

    global_domain_motion_mode =
        tmv::init::config::DomainMotionMode::Stationary;
    global_domain_frame = tmv::domain::HorizontalDomainFrame::stationary();
}

TEST_CASE("GPU Cartesian C-grid slow tendencies parity",
          "[vulkan][gpu_parity][cgrid][cartesian][slow]")
{
    setup_cart_grid(/*nx=*/16, /*ny=*/16, /*nz=*/16,
                    /*dx=*/500.0, /*dy=*/500.0, /*dz=*/250.0, /*dt=*/0.1);

    rho0_base.assign(NZ, 1.0);
    p0_base.assign(NZ, 100000.0);
    for (int k = 0; k < NZ; ++k)
    {
        rho0_base[k] = 1.15 * std::exp(-static_cast<double>(k) * dz / 8500.0);
        p0_base[k]   = 1.0e5 * std::exp(-static_cast<double>(k) * dz / 8500.0);
    }
    qv0_base.clear();
    qv.resize(0, 0, 0);
    qc.resize(0, 0, 0);

    if (!try_init_gpu())
    {
        WARN("GPU unavailable -- skipping cartesian_cgrid slow tendencies parity");
        return;
    }

    auto* backend = mutable_compute_backend();
    auto* kernels = backend != nullptr ? backend->dynamics_kernels() : nullptr;
    if (kernels == nullptr || !kernels->supports(DynamicsKernelOp::CartesianCgridSlowTendencies))
    {
        const char* required = std::getenv("NIMBUS_GPU_REQUIRED");
        if (required != nullptr && std::string(required) == "1")
        {
            shutdown_backend();
            FAIL("required Vulkan device lacks Cartesian C-grid slow-tendency "
                 "support");
        }
        WARN("GPU does not support cartesian_cgrid slow tendencies -- skipping");
        shutdown_backend();
        return;
    }

    CartesianCGrid3DScheme scheme;

    auto run_gpu = [&](Field3D& du, Field3D& dv, Field3D& dw,
                       Field3D& drho, Field3D& dp,
                       SlowMomentumAdvectionTendencyOutputs*
                           advection = nullptr,
                       float coriolis_beta = 0.0f) -> bool {
        std::vector<float> rho0_f(rho0_base.begin(), rho0_base.end());
        std::vector<float> qv0_f(1, 0.0f);
        std::vector<float> loading_buf(1, 0.0f);
        // Wind-profile basis for the perturbation Coriolis (M1): zeros,
        // matching the CPU's empty-profile fallback; coriolis_f follows
        // the global so the sections can exercise the term.
        std::vector<float> wind0_f(1, 0.0f);
        return dispatch_cartesian_cgrid_slow_tendencies_backend(
            u.data(), v.data(), w.data(),
            rho.data(), p.data(),
            rho0_f.data(), static_cast<int>(rho0_f.size()),
            qv0_f.data(), static_cast<int>(qv0_f.size()),
            wind0_f.data(), wind0_f.data(),
            static_cast<int>(wind0_f.size()),
            loading_buf.data(), loading_buf.data(),
            du.data(), dv.data(), dw.data(), drho.data(), dp.data(),
            NR, NTH, NZ,
            static_cast<float>(dr), static_cast<float>(dtheta),
            static_cast<float>(dz), static_cast<float>(kG),
            static_cast<float>(coriolis_f),
            /*has_moisture=*/false, /*has_loading=*/false,
            /*slow_rk2_invariant_input_residency=*/false,
            advection != nullptr ? advection->du_dt->data() : nullptr,
            advection != nullptr ? advection->dv_dt->data() : nullptr,
            advection != nullptr ? advection->dw_dt->data() : nullptr,
            static_cast<float>(global_domain_frame.velocity_ms.u_ms),
            static_cast<float>(global_domain_frame.velocity_ms.v_ms),
            coriolis_beta);
    };

    SECTION("Hydrostatic: zero tendencies on both CPU and GPU")
    {
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    rho(i, j, k) = static_cast<float>(rho0_base[k]);
                    p  (i, j, k) = static_cast<float>(p0_base[k]);
                }

        Field3D du_cpu(NR, NTH, NZ, 0.0f), dv_cpu(NR, NTH, NZ, 0.0f),
                dw_cpu(NR, NTH, NZ, 0.0f), drho_cpu(NR, NTH, NZ, 0.0f),
                dp_cpu(NR, NTH, NZ, 0.0f);
        Field3D du_gpu(NR, NTH, NZ, 0.0f), dv_gpu(NR, NTH, NZ, 0.0f),
                dw_gpu(NR, NTH, NZ, 0.0f), drho_gpu(NR, NTH, NZ, 0.0f),
                dp_gpu(NR, NTH, NZ, 0.0f);
        Field3D adv_u_cpu(NR, NTH, NZ, 0.0f);
        Field3D adv_v_cpu(NR, NTH, NZ, 0.0f);
        Field3D adv_w_cpu(NR, NTH, NZ, 0.0f);
        Field3D adv_u_gpu(NR, NTH, NZ, 0.0f);
        Field3D adv_v_gpu(NR, NTH, NZ, 0.0f);
        Field3D adv_w_gpu(NR, NTH, NZ, 0.0f);
        SlowMomentumAdvectionTendencyOutputs cpu_advection{
            &adv_u_cpu, &adv_v_cpu, &adv_w_cpu,
        };
        SlowMomentumAdvectionTendencyOutputs gpu_advection{
            &adv_u_gpu, &adv_v_gpu, &adv_w_gpu,
        };

        shutdown_backend();
        scheme.compute_slow_tendencies(u, v, w, rho, p, theta, dt,
                                       du_cpu, dv_cpu, dw_cpu, drho_cpu, dp_cpu,
                                       &cpu_advection);
        REQUIRE(try_init_gpu());
        REQUIRE(run_gpu(
            du_gpu, dv_gpu, dw_gpu, drho_gpu, dp_gpu,
            &gpu_advection));

        constexpr double kGate = 1.0e-4;
        // The slow kernel owns k=0 momentum tendencies; whole-field output
        // comparison includes that prognostic level and also verifies that
        // unowned edge/top cells remain the prescribed zero initialization.
        constexpr auto kDomain = nimbus::parity::ComparisonDomain::FullField;
        require_parity("hydrostatic du", du_cpu, du_gpu, kGate, kDomain);
        require_parity("hydrostatic dv", dv_cpu, dv_gpu, kGate, kDomain);
        require_parity("hydrostatic dw", dw_cpu, dw_gpu, kGate, kDomain);
        require_parity("hydrostatic drho", drho_cpu, drho_gpu, kGate, kDomain);
        require_parity("hydrostatic dp", dp_cpu, dp_gpu, kGate, kDomain);
        require_parity(
            "hydrostatic advection du", adv_u_cpu, adv_u_gpu,
            kGate, kDomain);
        require_parity(
            "hydrostatic advection dv", adv_v_cpu, adv_v_gpu,
            kGate, kDomain);
        require_parity(
            "hydrostatic advection dw", adv_w_cpu, adv_w_gpu,
            kGate, kDomain);
    }

    SECTION("Sheared Cartesian wind + warm bump (upwind advection + buoyancy)")
    {
        const double cx = 0.5 * (NR - 1), cy = 0.5 * (NTH - 1), cz = 0.5 * (NZ - 1);
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double sx = std::sin(2.0 * kPi * i / NR);
                    const double sy = std::cos(2.0 * kPi * j / NTH);
                    const double sk = std::sin(2.0 * kPi * k / NZ);
                    u(i, j, k) = static_cast<float>(5.0 * sx + 0.3 * k);
                    v(i, j, k) = static_cast<float>(3.0 * sy);
                    w(i, j, k) = static_cast<float>(2.0 * sk * sx);
                    const double r2 = (i - cx) * (i - cx) + (j - cy) * (j - cy)
                                    + (k - cz) * (k - cz);
                    rho(i, j, k) = static_cast<float>(rho0_base[k]
                                   + 0.05 * std::exp(-r2 / 8.0));
                    p(i, j, k) = static_cast<float>(p0_base[k]);
                }

        Field3D du_cpu(NR, NTH, NZ, 0.0f), dv_cpu(NR, NTH, NZ, 0.0f),
                dw_cpu(NR, NTH, NZ, 0.0f), drho_cpu(NR, NTH, NZ, 0.0f),
                dp_cpu(NR, NTH, NZ, 0.0f);
        Field3D du_gpu(NR, NTH, NZ, 0.0f), dv_gpu(NR, NTH, NZ, 0.0f),
                dw_gpu(NR, NTH, NZ, 0.0f), drho_gpu(NR, NTH, NZ, 0.0f),
                dp_gpu(NR, NTH, NZ, 0.0f);
        Field3D adv_u_cpu(NR, NTH, NZ, 0.0f);
        Field3D adv_v_cpu(NR, NTH, NZ, 0.0f);
        Field3D adv_w_cpu(NR, NTH, NZ, 0.0f);
        Field3D adv_u_gpu(NR, NTH, NZ, 0.0f);
        Field3D adv_v_gpu(NR, NTH, NZ, 0.0f);
        Field3D adv_w_gpu(NR, NTH, NZ, 0.0f);
        SlowMomentumAdvectionTendencyOutputs cpu_advection{
            &adv_u_cpu, &adv_v_cpu, &adv_w_cpu,
        };
        SlowMomentumAdvectionTendencyOutputs gpu_advection{
            &adv_u_gpu, &adv_v_gpu, &adv_w_gpu,
        };

        shutdown_backend();
        scheme.compute_slow_tendencies(u, v, w, rho, p, theta, dt,
                                       du_cpu, dv_cpu, dw_cpu, drho_cpu, dp_cpu,
                                       &cpu_advection);
        REQUIRE(try_init_gpu());
        REQUIRE(run_gpu(
            du_gpu, dv_gpu, dw_gpu, drho_gpu, dp_gpu,
            &gpu_advection));

        // O(1 m/s^2) tendencies here; fp32-vs-fp64 over the 5th-order upwind
        // (coeffs up to 60) drifts more than the centered cyl kernel.
        constexpr double kGate = 1.0e-2;
        constexpr auto kDomain = nimbus::parity::ComparisonDomain::FullField;
        require_parity("sheared du", du_cpu, du_gpu, kGate, kDomain);
        require_parity("sheared dv", dv_cpu, dv_gpu, kGate, kDomain);
        require_parity("sheared dw", dw_cpu, dw_gpu, kGate, kDomain);
        require_parity("sheared drho", drho_cpu, drho_gpu, kGate, kDomain);
        require_parity("sheared dp", dp_cpu, dp_gpu, kGate, kDomain);
        require_parity(
            "sheared advection du", adv_u_cpu, adv_u_gpu,
            kGate, kDomain);
        require_parity(
            "sheared advection dv", adv_v_cpu, adv_v_gpu,
            kGate, kDomain);
        require_parity(
            "sheared advection dw", adv_w_cpu, adv_w_gpu,
            kGate, kDomain);
    }

    SECTION("constant frame translation changes advectors, not stored winds")
    {
        constexpr double frame_u = 8.75;
        constexpr double frame_v = -3.625;
        global_domain_motion_mode =
            tmv::init::config::DomainMotionMode::ConstantVelocity;
        global_domain_frame.velocity_ms = {frame_u, frame_v};

        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double sx = std::sin(2.0 * kPi * i / NR);
                    const double sy = std::cos(2.0 * kPi * j / NTH);
                    const double sk = std::sin(2.0 * kPi * k / NZ);
                    u(i, j, k) = static_cast<float>(
                        frame_u + 5.0 * sx + 0.3 * k);
                    v(i, j, k) = static_cast<float>(frame_v + 3.0 * sy);
                    w(i, j, k) = static_cast<float>(2.0 * sk * sx);
                    rho(i, j, k) = static_cast<float>(
                        rho0_base[k] + 0.01 * sx * sy);
                    p(i, j, k) = static_cast<float>(
                        p0_base[k] + 14.0 * sx * sy * sk);
                }

        Field3D du_cpu(NR, NTH, NZ, 0.0f), dv_cpu(NR, NTH, NZ, 0.0f),
                dw_cpu(NR, NTH, NZ, 0.0f), drho_cpu(NR, NTH, NZ, 0.0f),
                dp_cpu(NR, NTH, NZ, 0.0f);
        Field3D du_gpu(NR, NTH, NZ, 0.0f), dv_gpu(NR, NTH, NZ, 0.0f),
                dw_gpu(NR, NTH, NZ, 0.0f), drho_gpu(NR, NTH, NZ, 0.0f),
                dp_gpu(NR, NTH, NZ, 0.0f);

        shutdown_backend();
        scheme.compute_slow_tendencies(
            u, v, w, rho, p, theta, dt,
            du_cpu, dv_cpu, dw_cpu, drho_cpu, dp_cpu);
        REQUIRE(try_init_gpu());
        REQUIRE(run_gpu(du_gpu, dv_gpu, dw_gpu, drho_gpu, dp_gpu));

        constexpr double kGate = 1.0e-2;
        constexpr auto kDomain =
            nimbus::parity::ComparisonDomain::FullField;
        require_parity("translated du", du_cpu, du_gpu, kGate, kDomain);
        require_parity("translated dv", dv_cpu, dv_gpu, kGate, kDomain);
        require_parity("translated dw", dw_cpu, dw_gpu, kGate, kDomain);
        require_parity("translated drho", drho_cpu, drho_gpu, kGate, kDomain);
        require_parity("translated dp", dp_cpu, dp_gpu, kGate, kDomain);
    }

    SECTION("Earth-referenced beta-plane Coriolis matches the CPU operator")
    {
        constexpr double f_y0 = 8.0e-5;
        constexpr double beta = 2.0e-11;
        coriolis_f = f_y0;
        scheme.set_coriolis_profile({f_y0, beta});
        u.fill(10.0f);
        v.fill(-3.0f);
        w.fill(0.0f);
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    rho(i, j, k) = static_cast<float>(rho0_base[k]);
                    p(i, j, k) = static_cast<float>(p0_base[k]);
                }

        Field3D du_cpu(NR, NTH, NZ, 0.0f), dv_cpu(NR, NTH, NZ, 0.0f),
                dw_cpu(NR, NTH, NZ, 0.0f), drho_cpu(NR, NTH, NZ, 0.0f),
                dp_cpu(NR, NTH, NZ, 0.0f);
        Field3D du_gpu(NR, NTH, NZ, 0.0f), dv_gpu(NR, NTH, NZ, 0.0f),
                dw_gpu(NR, NTH, NZ, 0.0f), drho_gpu(NR, NTH, NZ, 0.0f),
                dp_gpu(NR, NTH, NZ, 0.0f);
        shutdown_backend();
        scheme.compute_slow_tendencies(
            u, v, w, rho, p, theta, dt,
            du_cpu, dv_cpu, dw_cpu, drho_cpu, dp_cpu);
        REQUIRE(try_init_gpu());
        REQUIRE(run_gpu(du_gpu, dv_gpu, dw_gpu, drho_gpu, dp_gpu,
                        nullptr, static_cast<float>(beta)));
        constexpr auto domain = nimbus::parity::ComparisonDomain::FullField;
        require_parity("beta-plane du", du_cpu, du_gpu, 2.0e-6, domain);
        require_parity("beta-plane dv", dv_cpu, dv_gpu, 2.0e-6, domain);
    }

    shutdown_backend();
}

TEST_CASE("Cartesian C-grid Coriolis acts on the perturbation wind",
          "[dynamics][cgrid][cartesian][coriolis][m1]")
{
    // SCIENCE.md M1: the C-grid scheme silently ignored
    // coriolis.enabled. The f-plane terms now act on the PERTURBATION
    // wind: du/dt += f (v - v0), dv/dt += -f (u - u0). On a uniform
    // calm-base state with a uniform u perturbation, every advection
    // and buoyancy term is zero and the only tendency is Coriolis.
    setup_cart_grid(/*nx=*/12, /*ny=*/12, /*nz=*/12,
                    /*dx=*/500.0, /*dy=*/500.0, /*dz=*/250.0, /*dt=*/0.1);
    rho0_base.assign(NZ, 1.0);
    p0_base.assign(NZ, 100000.0);
    qv0_base.clear();
    qv.resize(0, 0, 0);
    qc.resize(0, 0, 0);

    const float u_pert = 10.0f;
    u.resize(NR, NTH, NZ, u_pert);

    CartesianCGrid3DScheme scheme;
    Field3D du(NR, NTH, NZ, 0.0f), dv(NR, NTH, NZ, 0.0f),
            dw(NR, NTH, NZ, 0.0f), drho(NR, NTH, NZ, 0.0f),
            dp(NR, NTH, NZ, 0.0f);

    SECTION("enabled: dv/dt = -f u_pert; toggling f off zeroes it")
    {
        coriolis_f = 8.365e-5;  // f-plane at 35 N
        scheme.compute_slow_tendencies(u, v, w, rho, p, theta, dt,
                                       du, dv, dw, drho, dp);

        const int i = 5, j = 5, k = 5;
        REQUIRE(static_cast<double>(dv(i, j, k))
                == Approx(-coriolis_f * u_pert).epsilon(1e-5));
        // No v perturbation: du/dt has no Coriolis contribution.
        REQUIRE(static_cast<double>(du(i, j, k))
                == Approx(0.0).margin(1e-12));

        coriolis_f = 0.0;
        scheme.compute_slow_tendencies(u, v, w, rho, p, theta, dt,
                                       du, dv, dw, drho, dp);
        REQUIRE(static_cast<double>(dv(i, j, k))
                == Approx(0.0).margin(1e-12));
    }

    SECTION("environmental wind carries no Coriolis force")
    {
        // u equals the environmental profile everywhere: the
        // perturbation is zero, so no spin-up of the base state.
        coriolis_f = 8.365e-5;
        u0_base.assign(NZ, static_cast<double>(u_pert));
        v0_base.assign(NZ, 0.0);
        scheme.compute_slow_tendencies(u, v, w, rho, p, theta, dt,
                                       du, dv, dw, drho, dp);

        const int i = 5, j = 5, k = 5;
        REQUIRE(static_cast<double>(dv(i, j, k))
                == Approx(0.0).margin(1e-12));
        coriolis_f = 0.0;
        u0_base.clear();
        v0_base.clear();
    }

    SECTION("beta-plane varies f with staggered Earth y while retaining perturbation winds")
    {
        constexpr double f_y0 = 8.0e-5;
        constexpr double beta = 2.0e-11;
        scheme.set_coriolis_profile({f_y0, beta});
        scheme.compute_slow_tendencies(u, v, w, rho, p, theta, dt,
                                       du, dv, dw, drho, dp);
        const int i = 5, k = 5;
        for (const int j : {2, 8})
        {
            // v[j] lives at the upper y face, (j+1)*dy. The pairwise
            // interpolation in the C-grid operator reduces exactly to this
            // local f for a uniform u perturbation.
            const double local_f = f_y0 + beta * (j + 1.0) * dtheta;
            CHECK(dv(i, j, k)
                  == Approx(-local_f * u_pert).epsilon(2.0e-6));
        }
        CHECK(dv(i, 2, k) != dv(i, 8, k));
    }
}

TEST_CASE("Cartesian C-grid fused CPU acoustic substep matches split reference",
          "[dynamics][cgrid][cartesian][cpu009][acoustic]")
{
    setup_cart_grid(/*nx=*/13, /*ny=*/11, /*nz=*/9,
                    /*dx=*/420.0, /*dy=*/510.0, /*dz=*/230.0, /*dt=*/0.08);
    rho0_base.resize(NZ);
    p0_base.resize(NZ);
    for (int k = 0; k < NZ; ++k)
    {
        rho0_base[k] = 1.17 * std::exp(-static_cast<double>(k) * dz / 8200.0);
        p0_base[k] = 100000.0 * std::exp(-static_cast<double>(k) * dz / 8200.0);
    }

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double x = static_cast<double>(i) - 6.0;
                const double y = static_cast<double>(j) - 5.0;
                const double z = static_cast<double>(k) - 4.0;
                const double bump = std::exp(
                    -0.08 * (x * x + y * y + z * z));
                rho[i][j][k] = static_cast<float>(rho0_base[k]
                    * (1.0 + 0.002 * bump));
                theta[i][j][k] = static_cast<float>(
                    300.0 + 0.35 * bump + 0.01 * k);
                p[i][j][k] = static_cast<float>(p0_base[k] + 18.0 * bump);
                u[i][j][k] = static_cast<float>(2.0 + 0.03 * j - 0.01 * k);
                v[i][j][k] = static_cast<float>(-1.0 + 0.02 * i + 0.01 * k);
                w[i][j][k] = static_cast<float>(0.08 * std::sin(0.4 * k)
                                                 * bump);
            }

    const Field3D u_initial = u, v_initial = v, w_initial = w;
    const Field3D rho_initial = rho, p_initial = p;

    Field3D u_ref = u, v_ref = v, w_ref = w;
    Field3D rho_ref = rho, p_ref = p;
    Field3D drho_ref(NR, NTH, NZ, 0.0f);
    Field3D dp_dead(NR, NTH, NZ, 0.0f);
    Field3D du_ref(NR, NTH, NZ, 0.0f);
    Field3D dv_ref(NR, NTH, NZ, 0.0f);
    Field3D dw_ref(NR, NTH, NZ, 0.0f);
    CartesianCGrid3DScheme reference_scheme;
    reference_scheme.compute_fast_pressure_tendencies(
        u_ref, v_ref, w_ref, rho_ref, p_ref, dt, drho_ref, dp_dead);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                double rho_new = static_cast<double>(rho_ref[i][j][k])
                    + static_cast<double>(drho_ref[i][j][k]) * dt;
                if (!std::isfinite(rho_new) || rho_new <= 0.0)
                    rho_new = std::max(0.1, rho0_base[k]);
                const float rho_stored = clamp_density_kgm3(
                    static_cast<float>(rho_new));
                rho_ref[i][j][k] = rho_stored;

                double p_new = p0;
                const double th = static_cast<double>(theta[i][j][k]);
                if (th > 0.0)
                {
                    p_new = dynamics_constants::p0 * std::pow(
                        static_cast<double>(rho_stored)
                            * dynamics_constants::R_d * th
                            / dynamics_constants::p0,
                        dynamics_constants::gamma);
                }
                if (!std::isfinite(p_new) || p_new <= 0.0) p_new = p0;
                p_ref[i][j][k] = clamp_pressure_pa(
                    static_cast<float>(p_new));
            }

    reference_scheme.compute_fast_momentum_tendencies(
        u_ref, v_ref, w_ref, rho_ref, p_ref, dt,
        du_ref, dv_ref, dw_ref);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u_ref[i][j][k] = clamp_wind_horizontal_ms(static_cast<float>(
                    static_cast<double>(u_ref[i][j][k])
                    + static_cast<double>(du_ref[i][j][k]) * dt));
                v_ref[i][j][k] = clamp_wind_horizontal_ms(static_cast<float>(
                    static_cast<double>(v_ref[i][j][k])
                    + static_cast<double>(dv_ref[i][j][k]) * dt));
                w_ref[i][j][k] = clamp_wind_vertical_ms(static_cast<float>(
                    static_cast<double>(w_ref[i][j][k])
                    + static_cast<double>(dw_ref[i][j][k]) * dt));
            }

    Field3D drho_fused;
    CartesianCGrid3DScheme fused_scheme;
    REQUIRE(fused_scheme.apply_fast_substep_cpu_in_place(
        u, v, w, rho, p, theta, dt, drho_fused));

    for (std::size_t idx = 0; idx < u.size(); ++idx)
    {
        REQUIRE(rho.data()[idx] == rho_ref.data()[idx]);
        REQUIRE(p.data()[idx] == p_ref.data()[idx]);
        REQUIRE(u.data()[idx] == u_ref.data()[idx]);
        REQUIRE(v.data()[idx] == v_ref.data()[idx]);
        REQUIRE(w.data()[idx] == w_ref.data()[idx]);
        REQUIRE(drho_fused.data()[idx] == drho_ref.data()[idx]);
    }

    // CPU-015: the same fused operation is a first-class backend family.
    // Host-native views preserve the direct kernel's fp64 geometry/profile
    // inputs and caller-owned scratch, so the interface route is exact.
    u = u_initial;
    v = v_initial;
    w = w_initial;
    rho = rho_initial;
    p = p_initial;
    Field3D drho_backend(NR, NTH, NZ, 0.0f);
    Field3D qv_energy(NR, NTH, NZ, 0.0f);
    Field3D qc_energy(NR, NTH, NZ, 0.0f);
    Field3D qr_energy(NR, NTH, NZ, 0.0f);
    Field3D qi_energy(NR, NTH, NZ, 0.0f);
    Field3D qs_energy(NR, NTH, NZ, 0.0f);
    Field3D qg_energy(NR, NTH, NZ, 0.0f);
    Field3D qh_energy(NR, NTH, NZ, 0.0f);
    Field3D tke_energy(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                qv_energy[i][j][k] =
                    static_cast<float>(0.009 + 1.0e-5 * (i + j + k));
                qc_energy[i][j][k] =
                    static_cast<float>(2.0e-4 * std::exp(-0.1 * k));
                qr_energy[i][j][k] = 3.0e-5f;
                qi_energy[i][j][k] = 1.0e-5f;
                qs_energy[i][j][k] = 2.0e-5f;
                qg_energy[i][j][k] = 4.0e-6f;
                qh_energy[i][j][k] = 1.0e-6f;
                tke_energy[i][j][k] =
                    static_cast<float>(0.15 + 0.002 * k);
            }

    const auto moist_energy_v1 = [&](const Field3D& u_field,
                                     const Field3D& v_field,
                                     const Field3D& w_field,
                                     const Field3D& rho_field,
                                     const Field3D& p_field)
    {
        constexpr double cv_d =
            physical_constants::specific_heat_cp_jkgk
            - physical_constants::gas_constant_dry_air_jkgk;
        constexpr double cv_v =
            physical_constants::specific_heat_water_vapor_cp_jkgk
            - physical_constants::gas_constant_water_vapor_jkgk;
        constexpr double c_l =
            physical_constants::specific_heat_liquid_water_jkgk;
        constexpr double c_i =
            physical_constants::specific_heat_ice_jkgk;
        constexpr double t_ref =
            physical_constants::freezing_temperature_k;
        constexpr double vapor_reference =
            physical_constants::latent_heat_vaporization_jkg
            - physical_constants::gas_constant_water_vapor_jkgk * t_ref;
        constexpr double ice_reference =
            -physical_constants::latent_heat_fusion_jkg;
        constexpr double kappa =
            physical_constants::gas_constant_dry_air_jkgk
            / physical_constants::specific_heat_cp_jkgk;
        const double volume = dr * dtheta * dz;
        double total = 0.0;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double rho_d = rho_field[i][j][k];
                    const double pressure = p_field[i][j][k];
                    const double temperature =
                        static_cast<double>(theta[i][j][k]) * std::pow(
                            pressure
                                / physical_constants::reference_pressure_pa,
                            kappa);
                    const double qv_k = qv_energy[i][j][k];
                    const double q_liquid =
                        static_cast<double>(qc_energy[i][j][k])
                        + static_cast<double>(qr_energy[i][j][k]);
                    const double q_ice =
                        static_cast<double>(qi_energy[i][j][k])
                        + static_cast<double>(qs_energy[i][j][k])
                        + static_cast<double>(qg_energy[i][j][k])
                        + static_cast<double>(qh_energy[i][j][k]);
                    const double q_total = qv_k + q_liquid + q_ice;
                    const double dry_mass = rho_d * volume;
                    const double mixture_mass =
                        dry_mass * (1.0 + q_total);
                    const double delta_t = temperature - t_ref;
                    const double u_w = u_field[i - 1][j][k];
                    const double u_e = u_field[i][j][k];
                    const double v_s = v_field[i][j - 1][k];
                    const double v_n = v_field[i][j][k];
                    const double w_b =
                        k == 0 ? 0.0 : w_field[i][j][k - 1];
                    const double w_t = w_field[i][j][k];
                    total += dry_mass * cv_d * delta_t;
                    total += dry_mass * delta_t
                        * (qv_k * cv_v + q_liquid * c_l + q_ice * c_i);
                    total += dry_mass
                        * (qv_k * vapor_reference + q_ice * ice_reference);
                    total += 0.25 * mixture_mass
                        * (u_w * u_w + u_e * u_e
                           + v_s * v_s + v_n * v_n
                           + w_b * w_b + w_t * w_t);
                    total += mixture_mass * physical_constants::gravity_ms2
                        * (static_cast<double>(k) + 0.5) * dz;
                    total += dry_mass
                        * static_cast<double>(tke_energy[i][j][k]);
                }
        return total;
    };
    const double energy_before =
        moist_energy_v1(u, v, w, rho, p);
    CartesianCGridAcousticEnergyReportV1 acoustic_energy_report;
    AcousticSubstepFusedCartesianCgridDispatch dispatch;
    dispatch.u = u.data();
    dispatch.v = v.data();
    dispatch.w = w.data();
    dispatch.rho = rho.data();
    dispatch.p = p.data();
    dispatch.theta_data = theta.data();
    dispatch.base_len = NZ;
    dispatch.nx = NR;
    dispatch.ny = NTH;
    dispatch.nz = NZ;
    dispatch.gamma_val = static_cast<float>(dynamics_constants::gamma);
    dispatch.rho_min = density_min_kgm3;
    dispatch.rho_max = density_max_kgm3;
    dispatch.p_min = pressure_min_pa;
    dispatch.p_max = pressure_max_pa;
    dispatch.p_ref_fallback = static_cast<float>(dynamics_constants::p0);
    dispatch.wind_clamp_h = wind_horizontal_abs_max_ms;
    dispatch.wind_clamp_v = wind_vertical_abs_max_ms;
    dispatch.rho0_base_host = rho0_base.data();
    dispatch.p0_base_host = p0_base.data();
    dispatch.dx_host = dr;
    dispatch.dy_host = dtheta;
    dispatch.dz_host = dz;
    dispatch.dt_small_host = dt;
    dispatch.gamma_host = dynamics_constants::gamma;
    dispatch.drho_scratch = drho_backend.data();
    dispatch.qv_host = qv_energy.data();
    dispatch.qc_host = qc_energy.data();
    dispatch.qr_host = qr_energy.data();
    dispatch.qi_host = qi_energy.data();
    dispatch.qs_host = qs_energy.data();
    dispatch.qg_host = qg_energy.data();
    dispatch.qh_host = qh_energy.data();
    dispatch.tke_host = tke_energy.data();

    auto cpu_backend = create_cpu_compute_backend();
    std::string backend_error;
    REQUIRE(cpu_backend->initialize(backend_error));
    REQUIRE(cpu_backend->dynamics_kernels() != nullptr);
    REQUIRE(cpu_backend->dynamics_kernels()
        ->dispatch_acoustic_substep_fused_cartesian_cgrid(dispatch));
    for (std::size_t idx = 0; idx < u.size(); ++idx)
    {
        REQUIRE(rho.data()[idx] == rho_ref.data()[idx]);
        REQUIRE(p.data()[idx] == p_ref.data()[idx]);
        REQUIRE(u.data()[idx] == u_ref.data()[idx]);
        REQUIRE(v.data()[idx] == v_ref.data()[idx]);
        REQUIRE(w.data()[idx] == w_ref.data()[idx]);
        REQUIRE(drho_backend.data()[idx] == drho_ref.data()[idx]);
    }

    // Science-disabled dispatch stays on the original reduction-free kernel.
    u = u_initial;
    v = v_initial;
    w = w_initial;
    rho = rho_initial;
    p = p_initial;
    dispatch.energy_report_host = &acoustic_energy_report;
    REQUIRE(cpu_backend->dynamics_kernels()
        ->dispatch_acoustic_substep_fused_cartesian_cgrid(dispatch));
    for (std::size_t idx = 0; idx < u.size(); ++idx)
    {
        REQUIRE(rho.data()[idx] == rho_ref.data()[idx]);
        REQUIRE(p.data()[idx] == p_ref.data()[idx]);
        REQUIRE(u.data()[idx] == u_ref.data()[idx]);
        REQUIRE(v.data()[idx] == v_ref.data()[idx]);
        REQUIRE(w.data()[idx] == w_ref.data()[idx]);
        REQUIRE(drho_backend.data()[idx] == drho_ref.data()[idx]);
    }
    const double energy_after =
        moist_energy_v1(u, v, w, rho, p);
    REQUIRE(acoustic_energy_report.complete);
    REQUIRE(acoustic_energy_report.substep_count == 1);
    REQUIRE(
        acoustic_energy_report.pressure_gradient_committed_work_j
        == Approx(
            acoustic_energy_report.pressure_gradient_requested_linear_work_j
            + acoustic_energy_report
                  .pressure_gradient_finite_step_correction_j
            + acoustic_energy_report.pressure_gradient_storage_rounding_j)
               .epsilon(1.0e-12));
    REQUIRE(acoustic_energy_report.committed_inventory_change_j
            == Approx(energy_after - energy_before).epsilon(2.0e-12));

    REQUIRE(cpu_backend->dynamics_kernels()
        ->dispatch_acoustic_substep_fused_cartesian_cgrid(dispatch));
    const double energy_after_two =
        moist_energy_v1(u, v, w, rho, p);
    REQUIRE(acoustic_energy_report.complete);
    REQUIRE(acoustic_energy_report.substep_count == 2);
    REQUIRE(acoustic_energy_report.committed_inventory_change_j
            == Approx(energy_after_two - energy_before).epsilon(2.0e-12));

    // Attribution validates the full SCI-140 state before the fused kernel
    // mutates any prognostic field.
    const Field3D u_before_reject = u;
    const Field3D v_before_reject = v;
    const Field3D w_before_reject = w;
    const Field3D rho_before_reject = rho;
    const Field3D p_before_reject = p;
    qv_energy[2][2][2] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        cpu_backend->dynamics_kernels()
            ->dispatch_acoustic_substep_fused_cartesian_cgrid(dispatch),
        std::invalid_argument);
    for (std::size_t idx = 0; idx < u.size(); ++idx)
    {
        REQUIRE(u.data()[idx] == u_before_reject.data()[idx]);
        REQUIRE(v.data()[idx] == v_before_reject.data()[idx]);
        REQUIRE(w.data()[idx] == w_before_reject.data()[idx]);
        REQUIRE(rho.data()[idx] == rho_before_reject.data()[idx]);
        REQUIRE(p.data()[idx] == p_before_reject.data()[idx]);
    }
    REQUIRE(acoustic_energy_report.substep_count == 2);
    cpu_backend->shutdown();
}

TEST_CASE("device slow apply exposes committed buffers before host copy-back",
          "[dynamics][cgrid][cartesian][slow-apply][SCI-160]")
{
    if (!try_init_gpu())
    {
        WARN("No usable Vulkan compute device; SCI-160 physical observer test skipped");
        return;
    }

    constexpr int nx = 2;
    constexpr int ny = 2;
    constexpr int nz = 2;
    constexpr std::size_t count = nx * ny * nz;
    std::vector<float> u(count, 1.0f);
    std::vector<float> v(count, -2.0f);
    std::vector<float> w(count, 0.5f);
    std::vector<float> pressure(count, 90000.0f);
    std::vector<float> du(count, 0.25f);
    std::vector<float> dv(count, -0.5f);
    std::vector<float> dw(count, 1.0f);
    std::vector<float> dp(count, 10.0f);
    std::vector<float> du_pbl(count, 0.125f);
    std::vector<float> dv_pbl(count, 0.25f);
    SlowApplyObserverProbe probe;
    probe.old_u = u.data();
    probe.old_v = v.data();
    probe.old_w = w.data();
    probe.old_p = pressure.data();
    probe.count = count;

    REQUIRE(dispatch_slow_apply_backend(
        u.data(), v.data(), w.data(), pressure.data(),
        du.data(), dv.data(), dw.data(), dp.data(),
        du_pbl.data(), dv_pbl.data(),
        nx, ny, nz, 0.5f,
        &observe_slow_apply, &probe));
    REQUIRE(probe.called);
    REQUIRE(probe.old_host_untouched);
    REQUIRE(probe.committed_u.size() == count);
    for (std::size_t index = 0; index < count; ++index)
    {
        const float expected_u =
            1.0f + (0.25f + 0.125f) * 0.5f;
        const float expected_v =
            -2.0f + (-0.5f + 0.25f) * 0.5f;
        const float expected_w = 0.5f + 1.0f * 0.5f;
        const float expected_p = 90000.0f + 10.0f * 0.5f;
        REQUIRE(probe.committed_u[index] == expected_u);
        REQUIRE(probe.committed_v[index] == expected_v);
        REQUIRE(probe.committed_w[index] == expected_w);
        REQUIRE(probe.committed_p[index] == expected_p);
        REQUIRE(u[index] == probe.committed_u[index]);
        REQUIRE(v[index] == probe.committed_v[index]);
        REQUIRE(w[index] == probe.committed_w[index]);
        REQUIRE(pressure[index] == probe.committed_p[index]);
    }
    shutdown_backend();
}

TEST_CASE(
    "physical Vulkan slow and PBL commit matches the CPU owner and closes",
    "[dynamics][cgrid][cartesian][slow-owner][pbl-owner][SCI-170]")
{
    setup_cart_grid(
        /*nx=*/10, /*ny=*/9, /*nz=*/7,
        /*dx=*/420.0, /*dy=*/575.0, /*dz=*/210.0, /*dt=*/0.35);
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    du_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    dv_dt_pbl.resize(NR, NTH, NZ, 0.0f);

    Field3D du(NR, NTH, NZ, 0.0f);
    Field3D dv(NR, NTH, NZ, 0.0f);
    Field3D dw(NR, NTH, NZ, 0.0f);
    Field3D dp(NR, NTH, NZ, 0.0f);
    Field3D adv_u(NR, NTH, NZ, 0.0f);
    Field3D adv_v(NR, NTH, NZ, 0.0f);
    Field3D adv_w(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double phase =
                    0.31 * i - 0.23 * j + 0.19 * k;
                u[i][j][k] = static_cast<float>(
                    7.0 + 0.3 * k + 0.8 * std::sin(phase));
                v[i][j][k] = static_cast<float>(
                    -2.0 + 0.15 * k + 0.6 * std::cos(phase));
                w[i][j][k] = static_cast<float>(
                    0.35 * std::sin(0.7 * phase));
                rho[i][j][k] = static_cast<float>(
                    1.16 * std::exp(-0.024 * k)
                    * (1.0 + 0.002 * std::cos(phase)));
                p[i][j][k] = static_cast<float>(
                    100000.0 * std::exp(-0.025 * k)
                    + 18.0 * std::sin(phase));
                theta[i][j][k] = static_cast<float>(
                    299.0 + 0.55 * k + 0.08 * std::cos(phase));
                qv[i][j][k] = static_cast<float>(
                    0.012 * std::exp(-0.09 * k)
                    + 8.0e-5 * std::sin(phase));
                qc[i][j][k] =
                    (i + j + k) % 5 == 0 ? 1.7e-4f : 0.0f;
                qr[i][j][k] =
                    (2 * i + j + k) % 9 == 0 ? 7.0e-5f : 0.0f;
                qi[i][j][k] =
                    (i + 2 * j + k) % 11 == 0 ? 3.0e-5f : 0.0f;
                qs[i][j][k] = 1.0e-5f;
                qg[i][j][k] = 2.0e-6f;
                qh[i][j][k] = 5.0e-7f;

                adv_u[i][j][k] = static_cast<float>(
                    -0.018 + 0.003 * std::sin(phase));
                adv_v[i][j][k] = static_cast<float>(
                    0.012 - 0.002 * std::cos(phase));
                adv_w[i][j][k] = static_cast<float>(
                    0.004 * std::sin(1.3 * phase));
                du[i][j][k] = static_cast<float>(
                    adv_u[i][j][k]
                    + 0.006 * std::cos(0.8 * phase));
                dv[i][j][k] = static_cast<float>(
                    adv_v[i][j][k]
                    - 0.005 * std::sin(0.9 * phase));
                dw[i][j][k] = static_cast<float>(
                    adv_w[i][j][k]
                    + 0.003 * std::cos(phase));
                dp[i][j][k] = static_cast<float>(
                    7.0 * std::sin(phase) - 2.0 * std::cos(0.5 * phase));
                du_dt_pbl[i][j][k] = static_cast<float>(
                    -0.0045 - 0.001 * std::sin(phase));
                dv_dt_pbl[i][j][k] = static_cast<float>(
                    0.0035 + 0.0008 * std::cos(phase));
            }

    const SlowMomentumAdvectionTendencyOutputs advection{
        &adv_u, &adv_v, &adv_w,
    };
    auto state = global_simulation_state();
    Field3D cpu_u, cpu_v, cpu_w, cpu_p;
    const auto cpu_report = reference_flat_cgrid_slow_commit(
        state, du, dv, dw, dp, advection, dt,
        cpu_u, cpu_v, cpu_w, cpu_p);

    REQUIRE_NOTHROW(tmv::device_slow_energy::validate_request(
        state, du, dv, dw, &advection, static_cast<float>(dt),
        /*attribute_momentum=*/true, /*attribute_pbl=*/true));
    REQUIRE_THROWS_AS(
        tmv::device_slow_energy::validate_request(
            state, du, dv, dw, nullptr, static_cast<float>(dt),
            /*attribute_momentum=*/true, /*attribute_pbl=*/true),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        tmv::device_slow_energy::attribute_completed_commit(
            state, du, dv, dw, &advection, static_cast<float>(dt),
            true, true, false, false,
            nullptr, cpu_v.data(), cpu_w.data(), cpu_p.data()),
        std::invalid_argument);

    if (!try_init_gpu())
    {
        WARN("No usable Vulkan compute device; SCI-170 physical owner test skipped");
        return;
    }
    SlowEnergyObserverContext observer{
        &state, &du, &dv, &dw, &advection, static_cast<float>(dt),
    };
    REQUIRE(dispatch_slow_apply_backend(
        u.data(), v.data(), w.data(), p.data(),
        du.data(), dv.data(), dw.data(), dp.data(),
        du_dt_pbl.data(), dv_dt_pbl.data(),
        NR, NTH, NZ, static_cast<float>(dt),
        &observe_slow_energy, &observer));
    REQUIRE(observer.called);

    constexpr auto full = nimbus::parity::ComparisonDomain::FullField;
    require_parity("slow commit u", cpu_u, u, 2.0e-6, full);
    require_parity("slow commit v", cpu_v, v, 2.0e-6, full);
    require_parity("slow commit w", cpu_w, w, 2.0e-6, full);
    require_parity("slow commit p", cpu_p, p, 1.0e-2, full);

    const auto& cpu = cpu_report.momentum;
    const auto& gpu = observer.result.momentum;
    REQUIRE(cpu.complete);
    REQUIRE(gpu.complete);
    require_conditioned_close(
        "advection requested",
        cpu.advection_requested_linear_work_j,
        gpu.advection_requested_linear_work_j);
    require_conditioned_close(
        "nonadvection requested",
        cpu.nonadvection_requested_linear_work_j,
        gpu.nonadvection_requested_linear_work_j);
    require_conditioned_close(
        "advection finite",
        cpu.advection_finite_step_correction_j,
        gpu.advection_finite_step_correction_j);
    require_conditioned_close(
        "nonadvection finite",
        cpu.nonadvection_finite_step_correction_j,
        gpu.nonadvection_finite_step_correction_j);
    require_conditioned_close(
        "coupled finite cross",
        cpu.coupled_finite_step_cross_work_j,
        gpu.coupled_finite_step_cross_work_j);
    require_conditioned_close(
        "committed kinetic",
        cpu.committed_kinetic_change_j,
        gpu.committed_kinetic_change_j);
    const double cpu_requested =
        cpu.advection_requested_linear_work_j
        + cpu.nonadvection_requested_linear_work_j
        + cpu.advection_finite_step_correction_j
        + cpu.nonadvection_finite_step_correction_j
        + cpu.coupled_finite_step_cross_work_j;
    const double gpu_requested =
        gpu.advection_requested_linear_work_j
        + gpu.nonadvection_requested_linear_work_j
        + gpu.advection_finite_step_correction_j
        + gpu.nonadvection_finite_step_correction_j
        + gpu.coupled_finite_step_cross_work_j;
    require_conditioned_close(
        "kinetic storage",
        cpu.storage_rounding_j,
        gpu.storage_rounding_j,
        std::max(std::abs(cpu_requested), std::abs(gpu_requested)));
    require_conditioned_close(
        "kinetic closure residual",
        cpu.closure_residual_j,
        gpu.closure_residual_j,
        std::max(
            std::abs(cpu.committed_kinetic_change_j),
            std::abs(gpu.committed_kinetic_change_j)));
    require_conditioned_close(
        "pressure dry internal",
        cpu.pressure_dry_internal_change_j,
        gpu.pressure_dry_internal_change_j);
    require_conditioned_close(
        "pressure water sensible",
        cpu.pressure_water_sensible_change_j,
        gpu.pressure_water_sensible_change_j);
    require_conditioned_close(
        "committed pressure",
        cpu.committed_pressure_energy_change_j,
        gpu.committed_pressure_energy_change_j);

    const auto& cpu_pbl = cpu_report.pbl;
    const auto& gpu_pbl = observer.result.pbl;
    require_conditioned_close(
        "PBL requested",
        cpu_pbl.requested_linear_work_j,
        gpu_pbl.requested_linear_work_j);
    require_conditioned_close(
        "PBL finite",
        cpu_pbl.finite_step_correction_j,
        gpu_pbl.finite_step_correction_j);
    require_conditioned_close(
        "PBL committed",
        cpu_pbl.committed_work_j,
        gpu_pbl.committed_work_j);
    require_conditioned_close(
        "PBL storage",
        cpu_pbl.storage_rounding_j,
        gpu_pbl.storage_rounding_j,
        std::max(
            std::abs(cpu_pbl.requested_linear_work_j),
            std::abs(gpu_pbl.requested_linear_work_j)));

    const double kinetic_scale = std::max(
        1.0, std::abs(gpu.committed_kinetic_change_j));
    REQUIRE(std::abs(
        gpu.committed_kinetic_change_j
        - (gpu_requested + gpu.storage_rounding_j))
        <= 8.0 * std::numeric_limits<double>::epsilon()
            * kinetic_scale);
    REQUIRE(std::abs(gpu.closure_residual_j)
        <= 8.0 * std::numeric_limits<double>::epsilon()
            * kinetic_scale);
    REQUIRE(gpu.committed_pressure_energy_change_j
        == gpu.pressure_dry_internal_change_j
            + gpu.pressure_water_sensible_change_j);
    const double pbl_scale =
        std::max(1.0, std::abs(gpu_pbl.committed_work_j));
    REQUIRE(std::abs(
        gpu_pbl.committed_work_j
        - (gpu_pbl.requested_linear_work_j
           + gpu_pbl.finite_step_correction_j
           + gpu_pbl.storage_rounding_j))
        <= 8.0 * std::numeric_limits<double>::epsilon()
            * pbl_scale);
    shutdown_backend();
}

TEST_CASE("Cartesian C-grid slow RK2 density pair is identically inactive",
          "[dynamics][cgrid][cartesian][cpu010][slow-rk2]")
{
    setup_cart_grid(/*nx=*/9, /*ny=*/8, /*nz=*/7,
                    /*dx=*/500.0, /*dy=*/600.0, /*dz=*/250.0, /*dt=*/0.1);
    rho0_base.assign(NZ, 1.0);
    p0_base.assign(NZ, 100000.0);
    qv0_base.clear();
    qv.resize(0, 0, 0);
    qc.resize(0, 0, 0);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[i][j][k] = static_cast<float>(1.0 + 0.1 * i);
                v[i][j][k] = static_cast<float>(-0.5 + 0.1 * j);
                w[i][j][k] = static_cast<float>(0.02 * k);
                rho[i][j][k] = static_cast<float>(1.0 + 0.001 * (i + j + k));
                p[i][j][k] = static_cast<float>(100000.0 + i - 2.0 * j + k);
            }

    CartesianCGrid3DScheme scheme;
    Field3D du(NR, NTH, NZ, 0.0f), dv(NR, NTH, NZ, 0.0f);
    Field3D dw(NR, NTH, NZ, 0.0f), drho(NR, NTH, NZ, 123.0f);
    Field3D dp(NR, NTH, NZ, 0.0f);
    scheme.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt, du, dv, dw, drho, dp);

    for (std::size_t idx = 0; idx < drho.size(); ++idx)
    {
        REQUIRE(drho.data()[idx] == 0.0f);
    }
}

TEST_CASE("Vulkan C-grid density publication preserves zeros aliases and rejected outputs",
          "[vulkan][gpu_parity][cgrid][slow][density-publication]")
{
    if (!try_init_gpu())
    {
        WARN("Physical Vulkan density-publication test skipped: no device");
        return;
    }
    struct Cleanup { ~Cleanup() { shutdown_backend(); } } cleanup;
    auto* backend = mutable_compute_backend();
    auto* kernels = backend->dynamics_kernels();
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports(DynamicsKernelOp::CartesianCgridSlowTendencies));
    constexpr int nx = 7, ny = 6, nz = 5;
    constexpr std::size_t count = nx * ny * nz;
    std::vector<float> wind_u(count), wind_v(count), wind_w(count), density(count, 1.0f);
    std::vector<float> pressure(count), vapor(count, 0.01f), loading(count, 0.002f);
    std::vector<float> rho_base(nz, 1.0f), qv_base(nz, 0.009f), wind_base(nz, 0.0f);
    for (std::size_t n = 0; n < count; ++n)
    {
        wind_u[n] = 1.0f + 0.01f * n;
        wind_v[n] = -0.5f + 0.02f * n;
        wind_w[n] = 0.03f * (n % nz);
        pressure[n] = 90000.0f + n;
    }
    std::array<std::vector<float>, 8> output;
    for (auto& field : output)
        field.assign(count, std::numeric_limits<float>::quiet_NaN());
    CartesianCgridSlowTendenciesDispatch dispatch{};
    dispatch.u_data = wind_u.data();
    dispatch.v_data = wind_v.data();
    dispatch.w_data = wind_w.data();
    dispatch.rho_data = density.data();
    dispatch.p_data = pressure.data();
    dispatch.rho0_base_data = rho_base.data();
    dispatch.rho0_base_len = nz;
    dispatch.qv0_base_data = qv_base.data();
    dispatch.qv0_base_len = nz;
    dispatch.u0_base_data = dispatch.v0_base_data = wind_base.data();
    dispatch.wind0_base_len = nz;
    dispatch.qv_data = vapor.data();
    dispatch.loading_data = loading.data();
    dispatch.du_dt_data = output[0].data();
    dispatch.dv_dt_data = output[1].data();
    dispatch.dw_dt_data = output[2].data();
    dispatch.drho_dt_data = output[3].data();
    dispatch.dp_dt_data = output[4].data();
    dispatch.du_advection_dt_data = output[5].data();
    dispatch.dv_advection_dt_data = output[6].data();
    dispatch.dw_advection_dt_data = output[7].data();
    dispatch.nr = nx;
    dispatch.nth = ny;
    dispatch.nz = nz;
    dispatch.dx = 500.0f;
    dispatch.dy = 600.0f;
    dispatch.dz = 250.0f;
    dispatch.g = 9.80665f;
    dispatch.coriolis_f_val = 8.0e-5f;
    dispatch.has_moisture = dispatch.has_loading = true;
    const auto run = [&](const CartesianCgridSlowTendenciesDispatch& request)
    {
        return kernels->dispatch_cartesian_cgrid_slow_tendencies(request);
    };
    const auto require_zero = [&](const float* values)
    {
        for (std::size_t n = 0; n < count; ++n)
        {
            REQUIRE(values[n] == 0.0f);
            REQUIRE_FALSE(std::signbit(values[n]));
        }
    };
    REQUIRE(run(dispatch));
    const auto reference = output;
    require_zero(output[3].data());
    REQUIRE(std::any_of(output[4].begin(), output[4].end(), [](float value) { return value != 0.0f; }));

    SECTION("Repeated dry and moist dispatches overwrite poisoned storage")
    {
        for (bool moist : {false, true})
        {
            for (auto& field : output)
                std::fill(field.begin(), field.end(), -123.0f);
            dispatch.has_moisture = dispatch.has_loading = moist;
            dispatch.du_advection_dt_data = dispatch.dv_advection_dt_data = dispatch.dw_advection_dt_data = nullptr;
            REQUIRE(run(dispatch));
            require_zero(output[3].data());
            if (moist)
                for (std::size_t f = 0; f < 5; ++f)
                    REQUIRE(output[f] == reference[f]);
        }
    }
    SECTION("Output aliases retain ordered publication")
    {
        dispatch.drho_dt_data = output[0].data();
        REQUIRE(run(dispatch));
        require_zero(output[0].data());
        dispatch.drho_dt_data = output[4].data();
        REQUIRE(run(dispatch));
        REQUIRE(output[4] == reference[4]);
        dispatch.drho_dt_data = output[5].data();
        REQUIRE(run(dispatch));
        REQUIRE(output[5] == reference[5]);
    }
    SECTION("Input aliases are published only after native evaluation")
    {
        dispatch.drho_dt_data = wind_u.data();
        REQUIRE(run(dispatch));
        require_zero(wind_u.data());
        for (std::size_t f : {0u, 1u, 2u, 4u, 5u, 6u, 7u})
            REQUIRE(output[f] == reference[f]);
    }
    SECTION("Invalid requests leave every output byte unchanged")
    {
        for (auto& field : output)
            std::fill(field.begin(), field.end(), -321.0f);
        const auto before = output;
        const auto completed = backend->successful_dispatch_count();
        auto invalid = dispatch;
        invalid.nr = 0;
        REQUIRE_FALSE(run(invalid));
        invalid = dispatch;
        invalid.coriolis_f_val = std::numeric_limits<float>::quiet_NaN();
        REQUIRE_FALSE(run(invalid));
        invalid = dispatch;
        invalid.rho0_base_data = nullptr;
        REQUIRE_FALSE(run(invalid));
        invalid = dispatch;
        invalid.dw_advection_dt_data = nullptr;
        REQUIRE_FALSE(run(invalid));
        REQUIRE(output == before);
        REQUIRE(backend->successful_dispatch_count() == completed);
    }
    SECTION("Shared slow-apply observer rejection preserves host inputs")
    {
        const auto before_u = wind_u, before_v = wind_v, before_w = wind_w, before_p = pressure;
        const auto reject = [](void*, const float*, const float*, const float*, const float*)
        {
            throw std::runtime_error("rejected completed slow output");
        };
        REQUIRE_THROWS_AS(dispatch_slow_apply_backend(
            wind_u.data(), wind_v.data(), wind_w.data(), pressure.data(),
            output[0].data(), output[1].data(), output[2].data(), output[4].data(),
            output[3].data(), output[3].data(), nx, ny, nz, 0.1f, reject, nullptr), std::runtime_error);
        REQUIRE(wind_u == before_u);
        REQUIRE(wind_v == before_v);
        REQUIRE(wind_w == before_w);
        REQUIRE(pressure == before_p);
    }
}

TEST_CASE("Cartesian C-grid exposes operator-identical slow advection tendencies",
          "[dynamics][cgrid][cartesian][slow-momentum-energy][SCI-149]")
{
    setup_cart_grid(/*nx=*/10, /*ny=*/9, /*nz=*/8,
                    /*dx=*/400.0, /*dy=*/550.0, /*dz=*/225.0, /*dt=*/0.2);
    rho0_base.assign(NZ, 1.0);
    p0_base.assign(NZ, 100000.0);
    theta0_base.assign(NZ, 300.0);
    qv0_base.clear();
    qv.resize(0, 0, 0);
    qc.resize(0, 0, 0);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[i][j][k] = static_cast<float>(
                    3.0 + 0.2 * i + 0.03 * j);
                v[i][j][k] = static_cast<float>(
                    -1.0 + 0.04 * i - 0.15 * j);
                w[i][j][k] = static_cast<float>(
                    0.02 * i - 0.01 * j + 0.04 * k);
            }

    Field3D du(NR, NTH, NZ, 0.0f), dv(NR, NTH, NZ, 0.0f);
    Field3D dw(NR, NTH, NZ, 0.0f), drho(NR, NTH, NZ, 0.0f);
    Field3D dp(NR, NTH, NZ, 0.0f);
    Field3D adu(NR, NTH, NZ, 7.0f), adv(NR, NTH, NZ, 7.0f);
    Field3D adw(NR, NTH, NZ, 7.0f);
    SlowMomentumAdvectionTendencyOutputs outputs{&adu, &adv, &adw};
    CartesianCGrid3DScheme scheme;
    scheme.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt,
        du, dv, dw, drho, dp, &outputs);

    const int i = 4, j = 4, k = 3;
    REQUIRE(std::abs(static_cast<double>(adu[i][j][k])) > 0.0);
    REQUIRE(std::abs(static_cast<double>(adv[i][j][k])) > 0.0);
    REQUIRE(du[i][j][k] == adu[i][j][k]);
    REQUIRE(dv[i][j][k] == adv[i][j][k]);
    REQUIRE(dw[i][j][k] == adw[i][j][k]);
    REQUIRE(adu[0][0][0] == 0.0f);
    REQUIRE(adv[0][0][0] == 0.0f);
    REQUIRE(adw[0][0][NZ - 1] == 0.0f);

    SlowMomentumAdvectionTendencyOutputs partial{&adu, nullptr, &adw};
    REQUIRE_THROWS_AS(
        scheme.compute_slow_tendencies(
            u, v, w, rho, p, theta, dt,
            du, dv, dw, drho, dp, &partial),
        std::invalid_argument);
}

TEST_CASE("Cartesian C-grid fused slow passes are bit-exact across thread counts",
          "[dynamics][cgrid][cartesian][cpu019][threads]")
{
    setup_cart_grid(/*nx=*/18, /*ny=*/17, /*nz=*/12,
                    /*dx=*/450.0, /*dy=*/575.0, /*dz=*/225.0, /*dt=*/0.2);
    rho0_base.resize(NZ);
    p0_base.resize(NZ);
    qv0_base.resize(NZ);
    u0_base.resize(NZ);
    v0_base.resize(NZ);
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    coriolis_f = 1.0e-4;
    for (int k = 0; k < NZ; ++k)
    {
        rho0_base[k] = 1.16 * std::exp(-static_cast<double>(k) * dz / 8500.0);
        p0_base[k] = 1.0e5 * std::exp(-static_cast<double>(k) * dz / 8500.0);
        qv0_base[k] = 0.012 * std::exp(-0.08 * k);
        u0_base[k] = 3.0 + 0.4 * k;
        v0_base[k] = -1.0 + 0.2 * k;
    }
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double phase = 0.17 * i - 0.11 * j + 0.23 * k;
                u[i][j][k] = static_cast<float>(u0_base[k] + std::sin(phase));
                v[i][j][k] = static_cast<float>(v0_base[k] + std::cos(phase));
                w[i][j][k] = static_cast<float>(0.25 * std::sin(0.7 * phase));
                rho[i][j][k] = static_cast<float>(
                    rho0_base[k] * (1.0 + 0.002 * std::cos(phase)));
                p[i][j][k] = static_cast<float>(p0_base[k] + 20.0 * std::sin(phase));
                theta[i][j][k] = static_cast<float>(300.0 + 0.03 * k);
                qv[i][j][k] = static_cast<float>(
                    qv0_base[k] + 1.0e-4 * std::sin(phase));
                qc[i][j][k] = static_cast<float>(
                    ((i + j + k) % 7 == 0) ? 2.0e-4 : 0.0);
                qr[i][j][k] = static_cast<float>(
                    ((i + 2 * j + k) % 11 == 0) ? 1.0e-4 : 0.0);
            }

    shutdown_compute_backend_runtime();
    global_compute_backend_config.backend = "cpu";
    global_compute_backend_config.allow_fallback = true;
    std::string backend_error;
    REQUIRE(initialize_compute_backend_runtime(backend_error));

    Field3D du_one(NR, NTH, NZ, 9.0f), dv_one(NR, NTH, NZ, 9.0f);
    Field3D dw_one(NR, NTH, NZ, 9.0f), drho_one(NR, NTH, NZ, 9.0f);
    Field3D dp_one(NR, NTH, NZ, 9.0f);
    Field3D du_many(NR, NTH, NZ, -9.0f), dv_many(NR, NTH, NZ, -9.0f);
    Field3D dw_many(NR, NTH, NZ, -9.0f), drho_many(NR, NTH, NZ, -9.0f);
    Field3D dp_many(NR, NTH, NZ, -9.0f);

#ifdef _OPENMP
    const int previous_dynamic = omp_get_dynamic();
    const int previous_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    CartesianCGrid3DScheme scheme;
    scheme.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt,
        du_one, dv_one, dw_one, drho_one, dp_one);
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif
    scheme.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt,
        du_many, dv_many, dw_many, drho_many, dp_many);

    bool exact = true;
    for (std::size_t index = 0; index < du_one.size(); ++index)
    {
        exact = exact && du_one.data()[index] == du_many.data()[index]
                      && dv_one.data()[index] == dv_many.data()[index]
                      && dw_one.data()[index] == dw_many.data()[index]
                      && drho_one.data()[index] == drho_many.data()[index]
                      && dp_one.data()[index] == dp_many.data()[index];
    }
#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
#endif
    shutdown_compute_backend_runtime();
    REQUIRE(exact);
}

TEST_CASE("Cartesian C-grid terrain pressure force is well balanced",
          "[dynamics][cartesian][cgrid][terrain][well-balanced][TE-002]")
{
    setup_cart_grid(/*nx=*/9, /*ny=*/8, /*nz=*/6,
                    /*dx=*/800.0, /*dy=*/900.0, /*dz=*/1000.0, /*dt=*/0.1);
    qv = Field3D{};
    qc = Field3D{};
    qr = Field3D{};
    qi = Field3D{};
    qs = Field3D{};
    qg = Field3D{};
    qh = Field3D{};

    TerrainConfig cfg;
    cfg.coord_id = "smoothed";
    cfg.ztop = static_cast<double>(NZ) * dz;
    cfg.smoothing_decay = 0.7;
    Topography2D topo;
    topography::initialize_topography(topo, NR, NTH);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            topo.h[i][j] = 300.0
                + 65.0 * std::sin(0.6 * static_cast<double>(i))
                       * std::cos(0.4 * static_cast<double>(j));

    TerrainMetrics3D metrics;
    topography::initialize_metrics(metrics, NR, NTH, NZ);
    topography::build_conservative_face_metrics(
        cfg, topo, topography::build_zeta_interfaces(NZ, cfg.ztop),
        dr, dtheta, metrics);

    p0_base.resize(NZ);
    rho0_base.resize(NZ);
    theta0_base.resize(NZ);
    for (int k = 0; k < NZ; ++k)
    {
        const double height = (static_cast<double>(k) + 0.5) * dz;
        p0_base[k] = 100000.0 * std::exp(-height / 8200.0);
        theta0_base[k] = 300.0 + 0.003 * height;
    }
    for (int k = 0; k < NZ; ++k)
    {
        const double height = (static_cast<double>(k) + 0.5) * dz;
        rho0_base[k] = terrain_transport::reference_density_fp32(
            p0_base, theta0_base, height, dz);
    }
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                p[i][j][k] = static_cast<float>(
                    terrain_transport::reference_profile_value_fp32(
                        p0_base, metrics.z(i, j, k), dz));
                rho[i][j][k] = static_cast<float>(
                    terrain_transport::reference_density_fp32(
                        p0_base, theta0_base, metrics.z(i, j, k), dz));
                theta[i][j][k] = static_cast<float>(
                    terrain_transport::reference_profile_value_fp32(
                        theta0_base, metrics.z(i, j, k), dz));
                u[i][j][k] = 0.0f;
                v[i][j][k] = 0.0f;
                w[i][j][k] = 0.0f;
            }

    TerrainRuntimeState terrain_runtime;
    terrain_runtime.config = cfg;
    terrain_runtime.config.scheme_id = "synthetic_terrain";
    terrain_runtime.topography = topo;
    terrain_runtime.metrics = metrics;
    auto execution = global_simulation_state();
    execution.terrain_runtime = &terrain_runtime;

    const Field3D rho_before_closure = rho;
    apply_initial_thermodynamic_closure(execution);
    bool closure_exact = true;
    for (std::size_t index = 0; index < rho.size(); ++index)
        closure_exact = closure_exact
            && rho.data()[index] == rho_before_closure.data()[index];
    REQUIRE(closure_exact);

    Field3D du(NR, NTH, NZ, 7.0f), dv(NR, NTH, NZ, 7.0f);
    Field3D dw(NR, NTH, NZ, 7.0f);
    CartesianCGrid3DScheme scheme(execution);
    scheme.compute_fast_momentum_tendencies(
        u, v, w, rho, p, /*dt_small=*/0.1, du, dv, dw);

    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 0; k < NZ - 1; ++k)
            {
                CHECK(du[i][j][k] == 0.0f);
                CHECK(dv[i][j][k] == 0.0f);
                CHECK(dw[i][j][k] == 0.0f);
            }

    Field3D slow_du(NR, NTH, NZ, 7.0f), slow_dv(NR, NTH, NZ, 7.0f);
    Field3D slow_dw(NR, NTH, NZ, 7.0f), slow_drho(NR, NTH, NZ, 7.0f);
    Field3D slow_dp(NR, NTH, NZ, 7.0f);
    scheme.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt,
        slow_du, slow_dv, slow_dw, slow_drho, slow_dp);
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 0; k < NZ - 1; ++k)
            {
                CHECK(slow_du[i][j][k] == 0.0f);
                CHECK(slow_dv[i][j][k] == 0.0f);
                CHECK(slow_dw[i][j][k] == 0.0f);
                CHECK(slow_drho[i][j][k] == 0.0f);
                CHECK(slow_dp[i][j][k] == 0.0f);
            }

    // Away from the impermeable lower face, a horizontally uniform physical
    // wind must not acquire momentum or spuriously advect the hydrostatic
    // pressure merely because constant-zeta surfaces slope through space. The
    // first level is intentionally excluded: w=0 there is not tangent to a
    // sloping ground and must be adjusted by the boundary dynamics.
    for (std::size_t index = 0; index < u.size(); ++index)
    {
        u.data()[index] = 8.0f;
        v.data()[index] = -3.0f;
    }
    scheme.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt,
        slow_du, slow_dv, slow_dw, slow_drho, slow_dp);
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                CHECK(slow_du[i][j][k] == 0.0f);
                CHECK(slow_dv[i][j][k] == 0.0f);
                CHECK(slow_dw[i][j][k] == 0.0f);
                CHECK(slow_drho[i][j][k] == 0.0f);
                CHECK(slow_dp[i][j][k] == 0.0f);
            }

    // The same physical wind is exactly at rest relative to a grid-attached
    // terrain map translating at (8,-3) m/s.  This includes the ground cell:
    // its impermeable physical w is based on grid-relative horizontal wind.
    global_domain_motion_mode =
        tmv::init::config::DomainMotionMode::ConstantVelocity;
    global_domain_frame.velocity_ms = {8.0, -3.0};
    scheme.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt,
        slow_du, slow_dv, slow_dw, slow_drho, slow_dp);
    Field3D fast_drho(NR, NTH, NZ, 7.0f);
    Field3D fast_dp(NR, NTH, NZ, 7.0f);
    scheme.compute_fast_pressure_tendencies(
        u, v, w, rho, p, dt, fast_drho, fast_dp);
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 0; k < NZ - 1; ++k)
            {
                CHECK(slow_du[i][j][k] == 0.0f);
                CHECK(slow_dv[i][j][k] == 0.0f);
                CHECK(slow_dw[i][j][k] == 0.0f);
                CHECK(slow_drho[i][j][k] == 0.0f);
                CHECK(slow_dp[i][j][k] == 0.0f);
                CHECK(fast_drho[i][j][k] == 0.0f);
                CHECK(fast_dp[i][j][k] == 0.0f);
            }
    global_domain_motion_mode =
        tmv::init::config::DomainMotionMode::Stationary;
    global_domain_frame = tmv::domain::HorizontalDomainFrame::stationary();

    Field3D density_scratch;
    const Field3D u_before = u;
    CHECK_FALSE(scheme.apply_fast_substep_cpu_in_place(
        u, v, w, rho, p, theta, 0.01, density_scratch));
    bool state_untouched = true;
    for (std::size_t index = 0; index < u.size(); ++index)
        state_untouched = state_untouched
            && u.data()[index] == u_before.data()[index];
    CHECK(state_untouched);
    CHECK(density_scratch.empty());
}

TEST_CASE("C-grid momentum interface capture is operator-authored and atomic",
          "[dynamics][cartesian][cgrid][momentum-interface-flux]")
{
    setup_cart_grid(9, 8, 4, 100.0, 80.0, 50.0, 0.2);
    rho0_base.assign(NZ, 1.25);
    p0_base.assign(NZ, 100000.0);
    theta0_base.assign(NZ, 300.0);
    qv = {}; qc = {}; qr = {}; qi = {}; qs = {}; qg = {}; qh = {};
    qv0_base.clear();
    rho.fill(1.25f);
    p.fill(100000.0f);
    theta.fill(300.0f);
    u.fill(0.0f);
    v.fill(0.0f);
    w.fill(0.0f);

    const tmv::CartesianInterfaceFootprint footprint{2, 6, 2, 6, 0, 3};
    CartesianCGridMomentumFluxRegister calm_register;
    calm_register.configure(NR, NTH, NZ, footprint);
    CartesianCGrid3DScheme calm_scheme;
    calm_scheme.set_momentum_interface_flux_register(&calm_register);
    Field3D du(NR, NTH, NZ, 7.0f), dv(NR, NTH, NZ, 7.0f);
    Field3D dw(NR, NTH, NZ, 7.0f), drho_out(NR, NTH, NZ, 7.0f);
    Field3D dp_out(NR, NTH, NZ, 7.0f);
    Field3D adv_u(NR, NTH, NZ, 7.0f), adv_v(NR, NTH, NZ, 7.0f);
    Field3D adv_w(NR, NTH, NZ, 7.0f);
    SlowMomentumAdvectionTendencyOutputs advection{&adv_u, &adv_v, &adv_w};
    calm_scheme.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt,
        du, dv, dw, drho_out, dp_out, &advection);
    for (std::size_t index = 0; index < du.size(); ++index)
    {
        REQUIRE(du.data()[index] == 0.0f);
        REQUIRE(dv.data()[index] == 0.0f);
        REQUIRE(dw.data()[index] == 0.0f);
        REQUIRE(adv_u.data()[index] == 0.0f);
        REQUIRE(adv_v.data()[index] == 0.0f);
        REQUIRE(adv_w.data()[index] == 0.0f);
    }
    REQUIRE(calm_register.advective_fluxes().contribution_count == 1);
    for (const auto* side : {
             &calm_register.advective_fluxes().west,
             &calm_register.advective_fluxes().east,
             &calm_register.advective_fluxes().south,
             &calm_register.advective_fluxes().north})
        for (const auto& component : side->component)
            for (double value : component)
                REQUIRE(value == 0.0);
    REQUIRE(calm_register.advective_fluxes().source_impulse ==
            std::array<double, 3>{});

    Field3D literal_u(NR, NTH, NZ, 0.0f);
    Field3D literal_v(NR, NTH, NZ, 0.0f);
    Field3D literal_w(NR, NTH, NZ, 0.0f);
    literal_u[footprint.i_begin - 1][footprint.j_begin][NZ - 1] = 2.0f;
    literal_u[footprint.i_end - 1][footprint.j_begin][NZ - 1] = 3.0f;
    literal_v[footprint.i_begin][footprint.j_begin - 1][NZ - 1] = -4.0f;
    literal_v[footprint.i_begin][footprint.j_end - 1][NZ - 1] = 5.0f;
    literal_w[footprint.i_begin][footprint.j_begin][NZ - 1] = 4.0f;
    const auto literal_inventory = calm_register.native_dual_inventory(
        rho, literal_u, literal_v, literal_w, dr, dtheta, dz);
    REQUIRE(literal_inventory[0] == Approx(2.5e6));
    REQUIRE(literal_inventory[1] == Approx(0.5e6));
    REQUIRE(literal_inventory[2] == Approx(1.0e6));

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                u[i][j][k] = static_cast<float>(1.0 + 0.35 * i);

    CartesianCGridMomentumFluxRegister register_one;
    register_one.configure(NR, NTH, NZ, footprint);
    CartesianCGrid3DScheme scheme_one;
    scheme_one.set_momentum_interface_flux_register(&register_one);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    scheme_one.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt,
        du, dv, dw, drho_out, dp_out, &advection);

    namespace mf = tmv::cartesian_cgrid_momentum_flux;
    const int literal_j = footprint.j_begin;
    const int literal_k = footprint.k_begin;
    REQUIRE(register_one.advective_fluxes().west.component[0][0] ==
            dt * dtheta * dz * mf::x_flux_x(
                rho, u, footprint.i_begin, literal_j, literal_k, 0.0));
    const int cell_i = 4;
    const double mass_div = (
        mf::x_normal_mass(rho, u, cell_i + 1, literal_j, literal_k, 0.0) -
        mf::x_normal_mass(rho, u, cell_i, literal_j, literal_k, 0.0)) / dr;
    const double momentum_div = (
        mf::x_flux_x(rho, u, cell_i + 1, literal_j, literal_k, 0.0) -
        mf::x_flux_x(rho, u, cell_i, literal_j, literal_k, 0.0)) / dr;
    const double expected_adv = (-momentum_div +
        static_cast<double>(u[cell_i][literal_j][literal_k]) * mass_div) /
        mf::rho_x(rho, cell_i, literal_j, literal_k);
    REQUIRE(adv_u[cell_i][literal_j][literal_k] ==
            static_cast<float>(expected_adv));

    double momentum_change_rate = 0.0;
    for (int i = 1; i < NR - 1; ++i)
    {
        const double local_mass_div = (
            mf::x_normal_mass(rho, u, i + 1, literal_j, literal_k, 0.0) -
            mf::x_normal_mass(rho, u, i, literal_j, literal_k, 0.0)) / dr;
        momentum_change_rate += (
            mf::rho_x(rho, i, literal_j, literal_k) *
                static_cast<double>(adv_u[i][literal_j][literal_k]) -
            static_cast<double>(u[i][literal_j][literal_k]) *
                local_mass_div) * dr * dtheta * dz;
    }
    const double boundary_rate = -(
        mf::x_flux_x(rho, u, NR - 1, literal_j, literal_k, 0.0) -
        mf::x_flux_x(rho, u, 1, literal_j, literal_k, 0.0)) * dtheta * dz;
    REQUIRE(boundary_rate != 0.0);
    REQUIRE(momentum_change_rate == Approx(boundary_rate).margin(3.0e-3));

    const auto operator_inventory_before = register_one.native_dual_inventory(
        rho, u, v, w, dr, dtheta, dz);
    Field3D committed_u = u, committed_v = v, committed_w = w;
    for (std::size_t index = 0; index < committed_u.size(); ++index)
    {
        committed_u.data()[index] = static_cast<float>(
            static_cast<double>(committed_u.data()[index]) +
            dt * static_cast<double>(du.data()[index]));
        committed_v.data()[index] = static_cast<float>(
            static_cast<double>(committed_v.data()[index]) +
            dt * static_cast<double>(dv.data()[index]));
        committed_w.data()[index] = static_cast<float>(
            static_cast<double>(committed_w.data()[index]) +
            dt * static_cast<double>(dw.data()[index]));
    }
    const auto operator_inventory_after = register_one.native_dual_inventory(
        rho, committed_u, committed_v, committed_w, dr, dtheta, dz);
    const auto operator_impulse = register_one.last_committed_operator_impulse();
    const auto requested_impulse = register_one.native_dual_tendency_impulse(
        rho, du, dv, dw, dt, dr, dtheta, dz);
    std::array<double, 3> storage_impulse{};
    for (std::size_t component = 0; component < 3; ++component)
        storage_impulse[component] =
            operator_inventory_after[component] -
            operator_inventory_before[component] - requested_impulse[component];
    register_one.accumulate_operator_storage_impulse(storage_impulse);
    for (std::size_t component = 0; component < 3; ++component)
        REQUIRE(operator_inventory_after[component] -
                    operator_inventory_before[component] ==
                Approx(operator_impulse[component] +
                       (requested_impulse[component] -
                        operator_impulse[component]) +
                       register_one.advective_fluxes().storage_impulse[component]));

    CartesianCGridMomentumFluxRegister two_stage_register;
    two_stage_register.configure(NR, NTH, NZ, footprint);
    for (int stage = 0; stage < 2; ++stage)
    {
        two_stage_register.stage_advective(
            rho, u, v, w, 0.5 * dt, dr, dtheta, dz);
        two_stage_register.validate_staged_commit();
        two_stage_register.commit_staged();
    }
    const auto two_stage_pending =
        two_stage_register.pending_committed_operator_impulse();
    const auto last_half_stage =
        two_stage_register.last_committed_operator_impulse();
    for (std::size_t component = 0; component < 3; ++component)
        REQUIRE(two_stage_pending[component] ==
                Approx(2.0 * last_half_stage[component]));
    two_stage_register.accumulate_operator_storage_impulse({});
    REQUIRE_THROWS(two_stage_register.pending_committed_operator_impulse());

    const Field3D du_one = du, dv_one = dv, dw_one = dw;
    CartesianCGridMomentumFluxRegister register_four;
    register_four.configure(NR, NTH, NZ, footprint);
    CartesianCGrid3DScheme scheme_four;
    scheme_four.set_momentum_interface_flux_register(&register_four);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    scheme_four.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt,
        du, dv, dw, drho_out, dp_out, &advection);
    REQUIRE(std::memcmp(du.data(), du_one.data(), du.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(dv.data(), dv_one.data(), dv.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(dw.data(), dw_one.data(), dw.size() * sizeof(float)) == 0);
    REQUIRE(register_four.advective_fluxes().west.component ==
            register_one.advective_fluxes().west.component);
    REQUIRE(register_four.advective_fluxes().east.component ==
            register_one.advective_fluxes().east.component);
    for (double value : register_one.advective_fluxes().storage_impulse)
        REQUIRE(std::isfinite(value));

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                u[i][j][k] = static_cast<float>(
                    1.5 + 0.2 * std::sin(0.7 * static_cast<double>(i)));
    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
        {
            u[NR - 2][j][k] = u[0][j][k];
            u[NR - 1][j][k] = u[1][j][k];
        }
    CartesianCGridMomentumFluxRegister periodic_register;
    periodic_register.configure(NR, NTH, NZ, footprint);
    CartesianCGrid3DScheme periodic_scheme;
    periodic_scheme.set_momentum_interface_flux_register(&periodic_register);
    periodic_scheme.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt,
        du, dv, dw, drho_out, dp_out, &advection);
    double periodic_change_rate = 0.0;
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 0; k < NZ - 1; ++k)
            {
                const double local_mass_div = (
                    mf::x_normal_mass(rho, u, i + 1, j, k, 0.0) -
                    mf::x_normal_mass(rho, u, i, j, k, 0.0)) / dr;
                periodic_change_rate += (
                    mf::rho_x(rho, i, j, k) *
                        static_cast<double>(adv_u[i][j][k]) -
                    static_cast<double>(u[i][j][k]) * local_mass_div) *
                    dr * dtheta * dz;
            }
    REQUIRE(periodic_change_rate == Approx(0.0).margin(2.0e-2));

    u.fill(5.0f);
    v.fill(0.0f);
    w.fill(0.0f);
    coriolis_f = 1.0e-3;
    CartesianCGridMomentumFluxRegister source_register;
    source_register.configure(NR, NTH, NZ, footprint);
    CartesianCGrid3DScheme source_scheme;
    source_scheme.set_momentum_interface_flux_register(&source_register);
    source_scheme.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt,
        du, dv, dw, drho_out, dp_out, &advection);
    REQUIRE(adv_v[3][3][1] == 0.0f);
    REQUIRE(dv[3][3][1] == -0.005f);
    double expected_y_source = 0.0;
    for (int i = footprint.i_begin; i < footprint.i_end; ++i)
        for (int j = footprint.j_begin; j < footprint.j_end; ++j)
            for (int k = footprint.k_begin; k < footprint.k_end; ++k)
                expected_y_source += mf::rho_y(rho, i, j, k) *
                    static_cast<double>(dv[i][j][k] - adv_v[i][j][k]) *
                    dr * dtheta * dz * dt;
    REQUIRE(source_register.advective_fluxes().source_impulse[1] ==
            expected_y_source);
    REQUIRE(source_register.pressure_fluxes().contribution_count == 0);
    coriolis_f = 0.0;

    CartesianCGridMomentumFluxRegister moving_register;
    moving_register.configure(NR, NTH, NZ, footprint);
    CartesianCGrid3DScheme moving_scheme;
    moving_scheme.set_momentum_interface_flux_register(&moving_register);
    du.fill(31.0f);
    const Field3D moving_du_before = du;
    global_domain_frame.velocity_ms = {1.0, 0.0};
    REQUIRE_THROWS_WITH(
        moving_scheme.compute_slow_tendencies(
            u, v, w, rho, p, theta, dt,
            du, dv, dw, drho_out, dp_out, &advection),
        Catch::Matchers::Contains("stationary flat level"));
    REQUIRE(std::memcmp(du.data(), moving_du_before.data(),
                        du.size() * sizeof(float)) == 0);
    REQUIRE(moving_register.advective_fluxes().contribution_count == 0);
    global_domain_frame = tmv::domain::HorizontalDomainFrame::stationary();

    Field3D ref_du(NR, NTH, NZ, 0.0f), ref_dv(NR, NTH, NZ, 0.0f);
    Field3D ref_dw(NR, NTH, NZ, 0.0f), ref_drho(NR, NTH, NZ, 0.0f);
    Field3D ref_dp(NR, NTH, NZ, 0.0f);
    CartesianCGrid3DScheme ordinary_scheme;
    ordinary_scheme.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt,
        ref_du, ref_dv, ref_dw, ref_drho, ref_dp);
    Field3D null_du(NR, NTH, NZ, 0.0f), null_dv(NR, NTH, NZ, 0.0f);
    Field3D null_dw(NR, NTH, NZ, 0.0f), null_drho(NR, NTH, NZ, 0.0f);
    Field3D null_dp(NR, NTH, NZ, 0.0f);
    CartesianCGrid3DScheme explicit_null_scheme;
    explicit_null_scheme.set_momentum_interface_flux_register(nullptr);
    explicit_null_scheme.compute_slow_tendencies(
        u, v, w, rho, p, theta, dt,
        null_du, null_dv, null_dw, null_drho, null_dp);
    REQUIRE(std::memcmp(ref_du.data(), null_du.data(), ref_du.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(ref_dv.data(), null_dv.data(), ref_dv.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(ref_dw.data(), null_dw.data(), ref_dw.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(ref_dp.data(), null_dp.data(), ref_dp.size() * sizeof(float)) == 0);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                p[i][j][k] = static_cast<float>(100000.0 + 7.0 * i - 3.0 * j);
    scheme_one.compute_fast_momentum_tendencies(
        u, v, w, rho, p, 0.05, du, dv, dw);
    REQUIRE(register_one.pressure_fluxes().contribution_count == 1);
    REQUIRE(register_one.advective_fluxes().contribution_count == 1);
    REQUIRE(register_one.pressure_fluxes().west.component[0][0] ==
            0.05 * dtheta * dz * static_cast<double>(
                p[footprint.i_begin][literal_j][literal_k]));
    REQUIRE(register_one.pressure_fluxes().west.component[1][0] == 0.0);
    const float expected_fast_u = static_cast<float>(
        -7.0 / (dr * 1.25));
    REQUIRE(du[3][3][1] == expected_fast_u);
    const auto pressure_inventory_before = register_one.native_dual_inventory(
        rho, u, v, w, dr, dtheta, dz);
    committed_u = u;
    committed_v = v;
    committed_w = w;
    for (std::size_t index = 0; index < committed_u.size(); ++index)
    {
        committed_u.data()[index] = static_cast<float>(
            static_cast<double>(committed_u.data()[index]) +
            0.05 * static_cast<double>(du.data()[index]));
        committed_v.data()[index] = static_cast<float>(
            static_cast<double>(committed_v.data()[index]) +
            0.05 * static_cast<double>(dv.data()[index]));
        committed_w.data()[index] = static_cast<float>(
            static_cast<double>(committed_w.data()[index]) +
            0.05 * static_cast<double>(dw.data()[index]));
    }
    const auto pressure_inventory_after = register_one.native_dual_inventory(
        rho, committed_u, committed_v, committed_w, dr, dtheta, dz);
    const auto pressure_requested =
        register_one.native_dual_tendency_impulse(
            rho, du, dv, dw, 0.05, dr, dtheta, dz);
    std::array<double, 3> pressure_storage{};
    for (std::size_t component = 0; component < 3; ++component)
        pressure_storage[component] =
            pressure_inventory_after[component] -
            pressure_inventory_before[component] - pressure_requested[component];
    register_one.accumulate_operator_storage_impulse(pressure_storage);
    for (std::size_t component = 0; component < 3; ++component)
        REQUIRE(pressure_inventory_after[component] -
                    pressure_inventory_before[component] ==
                Approx(pressure_requested[component] +
                       register_one.pressure_fluxes().storage_impulse[component]));

    CartesianCGridMomentumFluxRegister storage_overflow_register;
    storage_overflow_register.configure(NR, NTH, NZ, footprint);
    storage_overflow_register.stage_pressure(p, 0.05, dr, dtheta, dz);
    storage_overflow_register.validate_staged_commit();
    storage_overflow_register.commit_staged();
    const double half_max = 0.75 * std::numeric_limits<double>::max();
    storage_overflow_register.accumulate_operator_storage_impulse(
        {half_max, 0.0, 0.0});
    const auto retained_storage =
        storage_overflow_register.pressure_fluxes().storage_impulse;
    REQUIRE_THROWS(storage_overflow_register.accumulate_operator_storage_impulse(
        {half_max, 0.0, 0.0}));
    REQUIRE(storage_overflow_register.pressure_fluxes().storage_impulse ==
            retained_storage);

    rho.fill(1.25f);
    p.fill(100000.0f);
    u.fill(1.0e18f);
    v.fill(0.0f);
    w.fill(0.0f);
    CartesianCGridMomentumFluxRegister overflow_register;
    overflow_register.configure(NR, NTH, NZ, footprint);
    constexpr double huge_dt = 2.0e268;
    overflow_register.stage_advective(rho, u, v, w, huge_dt, dr, dtheta, dz);
    overflow_register.validate_staged_commit();
    overflow_register.commit_staged();
    const auto committed_before = overflow_register.advective_fluxes();
    CartesianCGrid3DScheme overflow_scheme;
    overflow_scheme.set_momentum_interface_flux_register(&overflow_register);
    du.fill(17.0f); dv.fill(18.0f); dw.fill(19.0f);
    drho_out.fill(20.0f); dp_out.fill(21.0f);
    const Field3D du_before = du, dv_before = dv, dw_before = dw;
    REQUIRE_THROWS_WITH(
        overflow_scheme.compute_slow_tendencies(
            u, v, w, rho, p, theta, huge_dt,
            du, dv, dw, drho_out, dp_out, &advection),
        Catch::Matchers::Contains("cumulative strip overflowed"));
    REQUIRE(std::memcmp(du.data(), du_before.data(), du.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(dv.data(), dv_before.data(), dv.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(dw.data(), dw_before.data(), dw.size() * sizeof(float)) == 0);
    REQUIRE(overflow_register.advective_fluxes().west.component ==
            committed_before.west.component);
    REQUIRE(overflow_register.advective_fluxes().contribution_count ==
            committed_before.contribution_count);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

TEST_CASE("C-grid moving density stage preserves an Earth-resting atmosphere",
          "[cgrid][cpu][moving_mesh][continuity]")
{
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    struct RestoreThreads { int count; ~RestoreThreads() { omp_set_num_threads(count); } } restore{saved_threads};
    omp_set_num_threads(1);
#endif
    constexpr int nx = 7, ny = 6, nz_local = 5;
    SimulationStorageState storage{{nx,ny,nz_local,400.0,1600.0,0.1,700.0}};
    storage.grid.geometry.initialize(nx,ny,nz_local,400.0,1600.0,700.0,
        CoordinateSystem::Cartesian,StaggerType::CGrid);
    storage.configuration.stagger_type = StaggerType::CGrid;
    storage.configuration.domain_frame.earth_origin_at_t0_m = {1250.0,-2250.0};
    storage.configuration.domain_frame.velocity_ms = {75.0,-40.0};
    auto& terrain = storage.runtime.terrain;
    terrain.config.scheme_id = "bell";
    terrain.config.coord_id = "smoothed";
    terrain.config.horizontal_placement_id = "earth_fixed";
    terrain.config.ztop = 8000.0;
    terrain.config.bell.h0 = GENERATE(0.0, 600.0);
    terrain.config.bell.a = 1500.0;
    terrain.config.bell.axisymmetric = true;
    terrain.config.feature_center_earth_x_m = 2000.0;
    terrain.config.feature_center_earth_y_m = -1000.0;
    tmv::EarthFixedTerrainBuildRequest request;
    request.nr=nx; request.nth=ny; request.nz=nz_local;
    request.dx_m=400.0; request.dy_m=700.0;
    request.grid_origin_earth_m = storage.configuration.domain_frame.earth_origin_at_t0_m;
    request.feature_center_earth_m = {2000.0,-1000.0};
    request.sampling = tmv::EarthFixedTerrainSampling::CellAverageBilinear;
    auto initial = tmv::prepare_earth_fixed_terrain_commit(terrain,request);
    REQUIRE(tmv::commit_earth_fixed_terrain(initial,terrain));
    for (auto* field : {&storage.fields.rho,&storage.fields.u,&storage.fields.v,&storage.fields.w})
        field->resize(nx,ny,nz_local,0.0f);
    storage.fields.rho.fill(1.25f);
    auto state = make_simulation_state_view(storage);
    CartesianCGrid3DScheme scheme(state);
    storage.configuration.lateral_bc_west = tmv::init::config::LateralBC::ZeroGradient;
    storage.configuration.lateral_bc_east = tmv::init::config::LateralBC::ZeroGradient;
    storage.configuration.lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    storage.configuration.lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;
    const auto motion = tmv::prepare_earth_fixed_terrain_motion(terrain,*state.domain_frame,0.0,1.0);
    const auto old_geometry = terrain.metrics.J.values;
    const auto uniform = scheme.prepare_moving_density_step(motion);
    for (std::size_t n=0; n<uniform.density.size(); ++n)
    {
        REQUIRE(uniform.density.data()[n] == 1.25f);
        REQUIRE(storage.fields.rho.data()[n] == 1.25f);
    }
    REQUIRE(terrain.metrics.J.values == old_geometry);
    REQUIRE(storage.time_s == 0.0);
    const double balance = uniform.inventory.final_inventory - uniform.inventory.initial_inventory
        + uniform.inventory.boundary_outflow - uniform.inventory.storage_rounding;
    REQUIRE(balance == Approx(0.0).margin(1.0e-4));
    CartesianCGridTransportContext transport;
    transport.begin(state,nx,ny,nz_local,1.0,&motion.terrain.geometry.metrics,75.0,-40.0);
    transport.set_beginning_geometry(terrain.metrics);
    transport.capture_density_begin(storage.fields.rho);
    auto bad_flux = uniform.integrated_relative_mass_flux;
    bad_flux.x(nx-1,ny-2,nz_local-1) = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(transport.accumulate_integrated_moving_fluxes(bad_flux,1.0));
    REQUIRE(transport.substep_count() == 0);
    REQUIRE(transport.accumulated_dt() == 0.0);
    REQUIRE_THROWS(transport.accumulate(state,storage.fields.rho,storage.fields.u,storage.fields.v,storage.fields.w,1.0));
    transport.accumulate_integrated_moving_fluxes(uniform.integrated_relative_mass_flux,1.0);
    transport.complete(state);
    storage.fields.rho = uniform.density;
    Field3D vapor(nx,ny,nz_local,0.0125f), transported;
    ScalarBoundaryTransportReport vapor_report;
    advect_scalar_conservative_3d_cartesian(state,vapor,transported,1.0,true,&transport,&vapor_report);
    for (int i=1; i<nx-1; ++i)
        for (int j=1; j<ny-1; ++j)
            for (int k=0; k<nz_local; ++k)
                REQUIRE(transported(i,j,k) == Approx(0.0125f).margin(3.0e-9));
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto parallel_density = scheme.prepare_moving_density_step(motion);
    Field3D parallel_vapor;
    advect_scalar_conservative_3d_cartesian(state,vapor,parallel_vapor,1.0,true,&transport);
    for (std::size_t n=0; n<uniform.density.size(); ++n)
        REQUIRE(parallel_density.density.data()[n] == uniform.density.data()[n]);
    for (int i=1; i<nx-1; ++i)
        for (int j=1; j<ny-1; ++j)
            for (int k=0; k<nz_local; ++k)
                REQUIRE(parallel_vapor(i,j,k) == transported(i,j,k));
    const auto mass_report = transport.boundary_mass_report(state,storage.fields.rho,400.0,700.0,1600.0);
    REQUIRE(mass_report.committed_state_change_kg == Approx(uniform.inventory.final_inventory-uniform.inventory.initial_inventory).margin(1.0e-4));
    REQUIRE(vapor_report.outward_boundary_inventory == Approx(mass_report.outward_boundary_mass_kg*0.0125).epsilon(2.0e-6));
    REQUIRE(vapor_report.committed_state_change == Approx(mass_report.committed_state_change_kg*0.0125).epsilon(5.0e-6));
    for (int i=0; i<nx; ++i)
        for (int j=0; j<ny; ++j)
            for (int k=0; k<nz_local; ++k)
            {
                storage.fields.u(i,j,k) = 8.0f + 0.3f*i - 0.2f*j;
                storage.fields.v(i,j,k) = -4.0f + 0.1f*i + 0.4f*j;
                storage.fields.w(i,j,k) = 2.0f + 0.1f*i - 0.2f*k;
            }
    const auto momenta = tmv::prepare_moving_cgrid_momentum_transport(
        state, motion.terrain.geometry.metrics, uniform.density,
        uniform.integrated_relative_mass_flux);
    if (terrain.config.bell.h0 == 0.0)
    {
        constexpr int i=2, j=2, k=2;
        CHECK(momenta.u(i,j,k) == Approx(storage.fields.u(i,j,k)
            + 0.3*75.0/400.0 - 0.2*(-40.0)/700.0).margin(2.0e-6));
        CHECK(momenta.v(i,j,k) == Approx(storage.fields.v(i,j,k)
            + 0.1*75.0/400.0 + 0.4*(-40.0)/700.0).margin(2.0e-6));
        CHECK(momenta.w(i,j,k) == Approx(storage.fields.w(i,j,k)
            + 0.1*75.0/400.0).margin(2.0e-6));
    }
    for (const auto& ledger : momenta.ledger)
    {
        const double impulse_scale = std::max(1.0, std::abs(ledger.initial_impulse));
        CHECK(std::abs(ledger.final_impulse - ledger.initial_impulse
              + ledger.outward_impulse - ledger.impulse_storage_rounding)
              / impulse_scale < 2.0e-13);
        const double kinetic_scale = std::max(1.0, ledger.initial_kinetic);
        CHECK(std::abs(ledger.final_kinetic - ledger.initial_kinetic
              + ledger.outward_kinetic - ledger.spatial_conversion
              - ledger.finite_step_work - ledger.density_storage_work
              - ledger.velocity_storage_work) / kinetic_scale < 2.0e-13);
    }
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    const auto serial_momenta = tmv::prepare_moving_cgrid_momentum_transport(
        state, motion.terrain.geometry.metrics, uniform.density,
        uniform.integrated_relative_mass_flux);
    for (std::size_t n=0; n<momenta.u.size(); ++n)
    {
        REQUIRE(serial_momenta.u.data()[n] == momenta.u.data()[n]);
        REQUIRE(serial_momenta.v.data()[n] == momenta.v.data()[n]);
        REQUIRE(serial_momenta.w.data()[n] == momenta.w.data()[n]);
    }
    for (std::size_t n=0; n<momenta.ledger.size(); ++n)
    {
        CHECK(serial_momenta.ledger[n].final_impulse == momenta.ledger[n].final_impulse);
        CHECK(serial_momenta.ledger[n].final_kinetic == momenta.ledger[n].final_kinetic);
    }
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    Field3D inconsistent_density = uniform.density;
    inconsistent_density(2,2,2) *= 1.1f;
    REQUIRE_THROWS(tmv::prepare_moving_cgrid_momentum_transport(
        state, motion.terrain.geometry.metrics, inconsistent_density,
        uniform.integrated_relative_mass_flux));
    auto inconsistent_flux = uniform.integrated_relative_mass_flux;
    inconsistent_flux.x(2,2,2) = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(tmv::prepare_moving_cgrid_momentum_transport(
        state, motion.terrain.geometry.metrics, uniform.density, inconsistent_flux));
    storage.fields.u.fill(8.0f);
    storage.fields.v.fill(-4.0f);
    storage.fields.w.fill(2.0f);
    const auto constant_momenta = tmv::prepare_moving_cgrid_momentum_transport(
        state, motion.terrain.geometry.metrics, uniform.density,
        uniform.integrated_relative_mass_flux);
    for (std::size_t n=0; n<constant_momenta.u.size(); ++n)
    {
        REQUIRE(constant_momenta.u.data()[n] == 8.0f);
        REQUIRE(constant_momenta.v.data()[n] == -4.0f);
        REQUIRE(constant_momenta.w.data()[n] == 2.0f);
    }
    storage.fields.u.fill(0.0f);
    storage.fields.v.fill(0.0f);
    storage.fields.w.fill(0.0f);
    for (Field3D* field : {&storage.fields.p, &storage.fields.theta,
            &storage.fields.qv, &storage.fields.qc, &storage.fields.qr,
            &storage.fields.qi, &storage.fields.qs, &storage.fields.qg, &storage.fields.qh})
        field->resize(nx,ny,nz_local,0.0f);
    storage.fields.p.fill(100000.0f);
    Field3D potential_temperature(nx,ny,nz_local,300.0f), theta_result;
    for (int i=0; i<nx; ++i)
        for (int j=0; j<ny; ++j)
            for (int k=0; k<nz_local; ++k)
                potential_temperature(i,j,k) = 300.0f + i + 0.5f*j + k;
    storage.fields.theta = potential_temperature;
    ScalarMoistEnergyTransportReportV1 scalar_energy;
    advect_scalar_conservative_3d_cartesian(state, potential_temperature,
        theta_result, 1.0, false, &transport, nullptr,
        ScalarMoistEnergyKind::PotentialTemperature, &scalar_energy);
    REQUIRE(scalar_energy.complete);
    long double dry_energy_change = 0.0;
    const auto& ending_metrics = motion.terrain.geometry.metrics;
    constexpr double dry_cv = physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    for (int i=1; i<nx-1; ++i)
        for (int j=1; j<ny-1; ++j)
            for (int k=0; k<nz_local; ++k)
                dry_energy_change += static_cast<long double>(uniform.density(i,j,k))
                    * ending_metrics.dx * ending_metrics.dy * ending_metrics.dzeta
                    * ending_metrics.J(i,j,k) * dry_cv
                    * (static_cast<double>(theta_result(i,j,k)) - potential_temperature(i,j,k));
    CHECK(scalar_energy.dry_internal_change_j
          == Approx(static_cast<double>(dry_energy_change)).epsilon(2.0e-12));
    CHECK(scalar_energy.water_sensible_change_j == 0.0);
    CHECK(scalar_energy.carrier_kinetic_change_j == 0.0);
    CHECK(scalar_energy.gravitational_change_j == 0.0);
    CHECK(scalar_energy.closure_residual_j == 0.0);
    CHECK(scalar_energy.committed_state_change_j
          == Approx(scalar_energy.density_history_remap_j
                   + scalar_energy.limited_face_state_change_j
                   + scalar_energy.storage_rounding_j).epsilon(2.0e-12));
    storage.fields.theta.fill(300.0f);
    for (int i=0; i<nx; ++i)
        for (int j=0; j<ny; ++j)
            for (int k=0; k<nz_local; ++k)
                vapor(i,j,k) = 0.005f + 0.0003f*i + 0.0001f*j;
    storage.fields.qv = vapor;
    ScalarMoistEnergyTransportReportV1 water_energy;
    advect_scalar_conservative_3d_cartesian(state, vapor, transported,
        1.0, true, &transport, nullptr, ScalarMoistEnergyKind::WaterVapor, &water_energy);
    REQUIRE(water_energy.complete);
    long double water_mass_change = 0.0, gravitational_change = 0.0;
    for (int i=1; i<nx-1; ++i)
        for (int j=1; j<ny-1; ++j)
            for (int k=0; k<nz_local; ++k)
            {
                const long double change = static_cast<long double>(uniform.density(i,j,k))
                    * ending_metrics.dx * ending_metrics.dy * ending_metrics.dzeta
                    * ending_metrics.J(i,j,k)
                    * (static_cast<double>(transported(i,j,k)) - vapor(i,j,k));
                water_mass_change += change;
                gravitational_change += change * ending_metrics.z(i,j,k)
                    * physical_constants::gravity_ms2;
            }
    constexpr double vapor_cv = physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double vapor_reference = physical_constants::latent_heat_vaporization_jkg
        - physical_constants::gas_constant_water_vapor_jkgk * physical_constants::freezing_temperature_k;
    CHECK(water_energy.water_sensible_change_j == Approx(static_cast<double>(water_mass_change)
        * vapor_cv * (300.0 - physical_constants::freezing_temperature_k)).epsilon(2.0e-12));
    CHECK(water_energy.phase_reference_change_j == Approx(static_cast<double>(water_mass_change)
        * vapor_reference).epsilon(2.0e-12));
    CHECK(water_energy.gravitational_change_j
        == Approx(static_cast<double>(gravitational_change)).epsilon(2.0e-12));
    CHECK(water_energy.carrier_kinetic_change_j == 0.0);
    for (int i=0; i<nx; ++i)
        for (int j=0; j<ny; ++j)
            for (int k=0; k<nz_local; ++k)
                storage.fields.rho(i,j,k) = 1.0f + 0.03f*i + 0.01f*j + 0.02f*k;
    const auto varying = scheme.prepare_moving_density_step(motion);
    REQUIRE(varying.inventory.final_inventory - varying.inventory.initial_inventory
        + varying.inventory.boundary_outflow - varying.inventory.storage_rounding
        == Approx(0.0).margin(1.0e-4));
    auto invalid = motion;
    invalid.swept_volumes.z(1,1,0) = 1.0;
    REQUIRE_THROWS(scheme.prepare_moving_density_step(invalid));
    invalid = motion;
    invalid.end_time_s += 1.0;
    REQUIRE_THROWS(scheme.prepare_moving_density_step(invalid));
    invalid = motion;
    invalid.swept_volumes.x(nx-1,ny-2,nz_local-1) = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(scheme.prepare_moving_density_step(invalid));
    REQUIRE(terrain.metrics.J.values == old_geometry);
    REQUIRE(storage.time_s == 0.0);
}
