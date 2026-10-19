/**
 * @file test_rayleigh_damping_yaml.cpp
 * @brief Phase C3 exit-criterion gate — `damping.{enabled, z_d, alpha, mode}`
 *        YAML keys actually drive simulator behavior.
 *
 * Setup: minimal grid, seed velocity perturbations everywhere, apply the
 * sponge under different settings, check that:
 *   - enabled=false leaves the velocity untouched (no-op)
 *   - alpha=0     leaves the velocity untouched (no relaxation)
 *   - alpha > 0   pulls velocity toward the base state (u_ref=0 here)
 *   - higher alpha relaxes faster
 *   - z_d controls which levels are damped (k below z_d untouched)
 *
 * `reset_rayleigh_damping()` is called between cases to force the lazy
 * `ensure_initialized()` to re-read the globals.
 */

#include "catch2/catch.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_backend_factory.hpp"
#include "compute/compute_kernel_template.hpp"
#include "compute/parity_check.hpp"
#include "core/field/field3d.hpp"
#include "core/infra/rayleigh_damping.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "dynamics/dynamics_base.hpp"
#include "terrain/conservative_transport.hpp"

#include <array>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

// Local stub of `compute_wind_profile` (real def in nimbus_sim.cpp; pulled
// into other tests with their own local copies). Returns zero base-state
// wind so the sponge relaxes toward 0 — what every assertion in this file
// expects.
void compute_wind_profile(const WindProfile& /*profile*/, double /*z*/,
                          double& u, double& v)
{
    u = 0.0;
    v = 0.0;
}

namespace
{

using tmv::init::config::DampingMode;

constexpr int kNR = 4;
constexpr int kNTH = 4;
constexpr int kNZ = 10;
constexpr double kDz = 500.0;        // domain z_top = 10*500 = 5000 m
constexpr float kU_PERTURB = 10.0f;  // distinctive interior value to relax away

void seed_grid_and_fields()
{
    NR = kNR; NTH = kNTH; NZ = kNZ;
    dr = 500.0; dz = kDz; dt = 1.0; dtheta = 500.0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    CoordinateSystem::Cartesian,
                                    StaggerType::Collocated);
    rho0_base.assign(NZ, 1.225);
    p0_base.assign(NZ, 100000.0);
    theta0_base.assign(NZ, 300.0);
    u0_base.assign(NZ, 0.0);
    v0_base.assign(NZ, 0.0);

    rho.resize(NR, NTH, NZ, 1.225f);
    p.resize(NR, NTH, NZ, 100000.0f);
    u.resize(NR, NTH, NZ, kU_PERTURB);   // 10 m/s perturbation everywhere
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    theta.resize(NR, NTH, NZ, 300.0f);
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    tke.resize(NR, NTH, NZ, 0.0f);
}

void reset_damping_globals_to_defaults()
{
    global_damping_enabled       = true;
    global_damping_z_d           = -1.0;
    global_damping_alpha         = 0.05;
    global_damping_mode          = DampingMode::Explicit;
    global_w_only_top_sponge     = true;     // production default since 2026-05-25
    global_w_only_depth_cells    = 8;
    global_w_only_alpha_max      = 0.20;
    global_lateral_sponge_enabled = true;
    global_lateral_sponge_width   = 20;
    global_lateral_sponge_alpha   = 0.05;
    reset_rayleigh_damping();
}

bool try_init_sponge_gpu()
{
    std::string error;
    global_compute_backend_config.backend = "vulkan";
    global_compute_backend_config.allow_fallback = false;
    if (initialize_compute_backend_runtime(error))
        return true;

    const std::string device_error = error;
    global_compute_backend_config.backend = "cpu";
    global_compute_backend_config.allow_fallback = true;
    initialize_compute_backend_runtime(error);
    const char* required = std::getenv("NIMBUS_GPU_REQUIRED");
    if (required != nullptr && std::string(required) == "1")
    {
        throw std::runtime_error(
            "required physical Vulkan sponge-owner qualification could not "
            "initialize a device: " + device_error);
    }
    return false;
}

struct SpongeObserverContext
{
    const SimulationStateView* state = nullptr;
    const float* profiles = nullptr;
    int profiles_len = 0;
    int lateral_width = 0;
    bool do_upper = false;
    bool do_w_only = false;
    bool do_lateral = false;
    float dt_step = 0.0f;
    tmv::SpongeEnergyReportV1 report;
    bool called = false;
};

void observe_sponge_commit(
    void* raw,
    const float* committed_u,
    const float* committed_v,
    const float* committed_w,
    const float* committed_theta)
{
    auto& context = *static_cast<SpongeObserverContext*>(raw);
    context.report =
        attribute_state_relaxation_cartesian_device_commit(
            *context.state,
            context.dt_step,
            context.profiles,
            context.profiles_len,
            context.lateral_width,
            context.do_upper,
            context.do_w_only,
            context.do_lateral,
            committed_u,
            committed_v,
            committed_w,
            committed_theta);
    context.called = true;
}

void require_sponge_field_close(
    const char* name,
    const Field3D& cpu,
    const Field3D& gpu,
    double absolute_tolerance)
{
    const auto comparison = nimbus::parity::compare_field(
        cpu.data(), gpu.data(),
        cpu.size_r(), cpu.size_th(), cpu.size_z(),
        {0.0, absolute_tolerance},
        nimbus::parity::ComparisonDomain::FullField);
    INFO(name << ": max_abs=" << comparison.max_abs
         << " at (" << comparison.worst_i << ","
         << comparison.worst_j << "," << comparison.worst_k << ")"
         << " cpu=" << comparison.worst_cpu
         << " gpu=" << comparison.worst_gpu
         << " nonfinite=" << comparison.nonfinite_cells);
    REQUIRE(comparison.passed);
}

void require_sponge_term_close(
    const char* name,
    double cpu,
    double gpu,
    double parent_scale = 0.0)
{
    const double scale = std::max({
        1.0, std::abs(cpu), std::abs(gpu), std::abs(parent_scale),
    });
    const double tolerance =
        256.0 * std::numeric_limits<float>::epsilon() * scale;
    INFO(name << ": cpu=" << cpu << " gpu=" << gpu
         << " abs=" << std::abs(cpu - gpu)
         << " tolerance=" << tolerance);
    REQUIRE(std::isfinite(cpu));
    REQUIRE(std::isfinite(gpu));
    REQUIRE(std::abs(cpu - gpu) <= tolerance);
}

}  // namespace

TEST_CASE(
    "physical Vulkan sponge commits and every pass report match CPU ownership",
    "[core][rayleigh_damping][gpu-owner][SCI-171]")
{
    if (!try_init_sponge_gpu())
    {
        WARN("No usable Vulkan compute device; SCI-171 physical owner test skipped");
        return;
    }

    struct Case
    {
        const char* name;
        StaggerType staggering;
        bool upper;
        bool w_only;
        bool lateral;
    };
    const Case cases[] = {
        {"A upper", StaggerType::Collocated, true, false, false},
        {"A w-only", StaggerType::Collocated, false, true, false},
        {"A lateral", StaggerType::Collocated, false, false, true},
        {"A overlap", StaggerType::Collocated, true, true, true},
        {"C upper", StaggerType::CGrid, true, false, false},
        {"C w-only", StaggerType::CGrid, false, true, false},
        {"C lateral", StaggerType::CGrid, false, false, true},
        {"C overlap", StaggerType::CGrid, true, true, true},
    };

    constexpr int nr = 8;
    constexpr int nth = 7;
    constexpr int nz = 6;
    constexpr double dx = 420.0;
    constexpr double dy = 575.0;
    constexpr double dz_local = 300.0;
    constexpr double dt_step = 0.65;

    const auto make_storage = [&](StaggerType staggering)
    {
        SimulationStorageState storage{
            {nr, nth, nz, dx, dz_local, dt_step, dy}};
        storage.grid.geometry.initialize(
            nr, nth, nz, dx, dz_local, dy,
            CoordinateSystem::Cartesian, staggering);
        auto& fields = storage.fields;
        fields.rho.resize(nr, nth, nz, 1.0f);
        fields.p.resize(nr, nth, nz, 90000.0f);
        fields.u.resize(nr, nth, nz, 0.0f);
        fields.v.resize(nr, nth, nz, 0.0f);
        fields.w.resize(nr, nth, nz, 0.0f);
        fields.theta.resize(nr, nth, nz, 300.0f);
        fields.qv.resize(nr, nth, nz, 0.0f);
        fields.qc.resize(nr, nth, nz, 0.0f);
        fields.qr.resize(nr, nth, nz, 0.0f);
        fields.qi.resize(nr, nth, nz, 0.0f);
        fields.qs.resize(nr, nth, nz, 0.0f);
        fields.qg.resize(nr, nth, nz, 0.0f);
        fields.qh.resize(nr, nth, nz, 0.0f);
        fields.tke.resize(nr, nth, nz, 0.0f);
        storage.base.rho0.resize(nz);
        storage.base.p0.resize(nz);
        storage.base.theta0.resize(nz);
        storage.base.u0.assign(nz, 0.0);
        storage.base.v0.assign(nz, 0.0);
        for (int k = 0; k < nz; ++k)
        {
            storage.base.rho0[static_cast<std::size_t>(k)] =
                1.18 * std::exp(-0.035 * k);
            storage.base.p0[static_cast<std::size_t>(k)] =
                100000.0 * std::exp(-0.036 * k);
            storage.base.theta0[static_cast<std::size_t>(k)] =
                299.0 + 0.45 * k;
        }
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const double phase =
                        0.27 * i - 0.21 * j + 0.34 * k;
                    fields.rho[i][j][k] = static_cast<float>(
                        storage.base.rho0[static_cast<std::size_t>(k)]
                        * (1.0 + 0.002 * std::cos(phase)));
                    fields.p[i][j][k] = static_cast<float>(
                        storage.base.p0[static_cast<std::size_t>(k)]
                        + 16.0 * std::sin(phase));
                    fields.u[i][j][k] = static_cast<float>(
                        8.0 + 0.4 * k + 0.7 * std::sin(phase));
                    fields.v[i][j][k] = static_cast<float>(
                        -3.0 + 0.2 * k + 0.5 * std::cos(phase));
                    fields.w[i][j][k] = static_cast<float>(
                        0.8 + 0.3 * std::sin(0.8 * phase));
                    fields.theta[i][j][k] = static_cast<float>(
                        304.0 + 0.3 * k + 0.12 * std::cos(phase));
                    fields.qv[i][j][k] = static_cast<float>(
                        0.011 * std::exp(-0.08 * k)
                        + 7.0e-5 * std::sin(phase));
                    fields.qc[i][j][k] =
                        (i + j + k) % 5 == 0 ? 1.5e-4f : 0.0f;
                    fields.qr[i][j][k] =
                        (2 * i + j + k) % 9 == 0 ? 6.0e-5f : 0.0f;
                    fields.qi[i][j][k] =
                        (i + 2 * j + k) % 11 == 0 ? 2.0e-5f : 0.0f;
                    fields.qs[i][j][k] = 8.0e-6f;
                    fields.qg[i][j][k] = 2.0e-6f;
                    fields.qh[i][j][k] = 5.0e-7f;
                    fields.tke[i][j][k] =
                        static_cast<float>(0.1 + 0.01 * k);
                }
        return storage;
    };

    bool checked_fail_closed_input = false;
    for (const auto& test_case : cases)
    {
        INFO("sponge owner case: " << test_case.name);
        // W-only uses the upper master switch with a zero broad-sponge
        // coefficient, matching the production selector.
        global_damping_enabled = test_case.upper || test_case.w_only;
        global_damping_z_d = 650.0;
        global_damping_alpha = test_case.upper ? 0.11 : 0.0;
        global_damping_mode = DampingMode::Explicit;
        global_w_only_top_sponge = test_case.w_only;
        global_w_only_depth_cells = 3;
        global_w_only_alpha_max = 0.23;
        global_lateral_sponge_enabled = test_case.lateral;
        global_lateral_sponge_width = 2;
        global_lateral_sponge_alpha = 0.075;

        SimulationStorageState cpu_storage =
            make_storage(test_case.staggering);
        SimulationStorageState gpu_storage = cpu_storage;
        reset_rayleigh_damping(cpu_storage.runtime.rayleigh);
        reset_rayleigh_damping(gpu_storage.runtime.rayleigh);
        SimulationStateView cpu_state =
            make_simulation_state_view(cpu_storage);
        SimulationStateView gpu_state =
            make_simulation_state_view(gpu_storage);

        tmv::SpongeEnergyReportV1 cpu_report;
        tmv::accumulate_sponge_energy(
            cpu_report,
            apply_rayleigh_damping(cpu_state, dt_step));
        tmv::accumulate_sponge_energy(
            cpu_report,
            apply_lateral_damping(cpu_state, dt_step));
        REQUIRE(cpu_report.complete);

        const auto& alpha = rayleigh_alpha_profile(gpu_state);
        const auto& alpha_upper_w =
            rayleigh_upper_w_alpha_profile(gpu_state);
        const auto& alpha_w_only =
            rayleigh_w_only_alpha_profile(gpu_state);
        const auto& u_ref = rayleigh_u_ref_profile(gpu_state);
        const auto& v_ref = rayleigh_v_ref_profile(gpu_state);
        const auto& lateral_alpha =
            rayleigh_lateral_alpha_profile(gpu_state);
        const int lateral_width =
            static_cast<int>(lateral_alpha.size());
        std::vector<float> profiles(
            static_cast<std::size_t>(7 * nz), 0.0f);
        for (int k = 0; k < nz; ++k)
        {
            const std::size_t ks = static_cast<std::size_t>(k);
            profiles[ks] =
                ks < alpha.size() ? static_cast<float>(alpha[ks]) : 0.0f;
            profiles[static_cast<std::size_t>(nz + k)] =
                ks < alpha_w_only.size()
                    ? static_cast<float>(alpha_w_only[ks]) : 0.0f;
            profiles[static_cast<std::size_t>(2 * nz + k)] =
                ks < u_ref.size() ? static_cast<float>(u_ref[ks]) : 0.0f;
            profiles[static_cast<std::size_t>(3 * nz + k)] =
                ks < v_ref.size() ? static_cast<float>(v_ref[ks]) : 0.0f;
            profiles[static_cast<std::size_t>(4 * nz + k)] =
                static_cast<float>(
                    gpu_storage.base.theta0[ks]);
            profiles[static_cast<std::size_t>(6 * nz + k)] =
                ks < alpha_upper_w.size()
                    ? static_cast<float>(alpha_upper_w[ks]) : 0.0f;
        }
        for (int d = 0; d < lateral_width; ++d)
        {
            profiles[static_cast<std::size_t>(5 * nz + d)] =
                static_cast<float>(
                    lateral_alpha[static_cast<std::size_t>(d)]);
        }

        const bool do_upper =
            test_case.upper || test_case.w_only;
        SpongeObserverContext observer{
            &gpu_state,
            profiles.data(),
            static_cast<int>(profiles.size()),
            lateral_width,
            do_upper,
            test_case.w_only,
            test_case.lateral,
            static_cast<float>(dt_step),
        };
        auto& gpu_fields = gpu_storage.fields;
        REQUIRE(dispatch_state_relaxation_cartesian_backend(
            gpu_fields.u.data(),
            gpu_fields.v.data(),
            gpu_fields.w.data(),
            gpu_fields.theta.data(),
            profiles.data(),
            static_cast<int>(profiles.size()),
            nr, nth, nz, lateral_width,
            do_upper, test_case.w_only, test_case.lateral,
            static_cast<float>(dt_step),
            &observe_sponge_commit,
            &observer));
        REQUIRE(observer.called);
        REQUIRE(observer.report.complete);

        if (!checked_fail_closed_input)
        {
            checked_fail_closed_input = true;
            REQUIRE_THROWS_AS(
                attribute_state_relaxation_cartesian_device_commit(
                    gpu_state, static_cast<float>(dt_step),
                    nullptr, static_cast<int>(profiles.size()),
                    lateral_width, do_upper, test_case.w_only,
                    test_case.lateral,
                    gpu_fields.u.data(), gpu_fields.v.data(),
                    gpu_fields.w.data(), gpu_fields.theta.data()),
                std::invalid_argument);
        }

        require_sponge_field_close(
            "sponge u", cpu_storage.fields.u, gpu_fields.u, 4.0e-6);
        require_sponge_field_close(
            "sponge v", cpu_storage.fields.v, gpu_fields.v, 4.0e-6);
        require_sponge_field_close(
            "sponge w", cpu_storage.fields.w, gpu_fields.w, 4.0e-6);
        require_sponge_field_close(
            "sponge theta",
            cpu_storage.fields.theta, gpu_fields.theta, 4.0e-5);

        const tmv::SpongePassEnergyReportV1* cpu_passes[] = {
            &cpu_report.upper,
            &cpu_report.w_only,
            &cpu_report.lateral,
        };
        const tmv::SpongePassEnergyReportV1* gpu_passes[] = {
            &observer.report.upper,
            &observer.report.w_only,
            &observer.report.lateral,
        };
        const char* pass_names[] = {"upper", "w-only", "lateral"};
        for (int pass_index = 0; pass_index < 3; ++pass_index)
        {
            const auto& cpu_pass = *cpu_passes[pass_index];
            const auto& gpu_pass = *gpu_passes[pass_index];
            INFO("sponge pass: " << pass_names[pass_index]);
            REQUIRE(cpu_pass.complete);
            REQUIRE(gpu_pass.complete);
            REQUIRE(cpu_pass.applied == gpu_pass.applied);
            const double momentum_parent = std::max({
                std::abs(cpu_pass.momentum_requested_linear_work_j),
                std::abs(gpu_pass.momentum_requested_linear_work_j),
                std::abs(cpu_pass.carrier_kinetic_change_j),
                std::abs(gpu_pass.carrier_kinetic_change_j),
            });
            const double energy_parent = std::max({
                std::abs(cpu_pass.committed_energy_change_j),
                std::abs(gpu_pass.committed_energy_change_j),
                std::abs(cpu_pass.dry_internal_change_j),
                std::abs(gpu_pass.dry_internal_change_j),
                std::abs(cpu_pass.carrier_kinetic_change_j),
                std::abs(gpu_pass.carrier_kinetic_change_j),
            });
            require_sponge_term_close(
                "momentum requested",
                cpu_pass.momentum_requested_linear_work_j,
                gpu_pass.momentum_requested_linear_work_j);
            require_sponge_term_close(
                "momentum finite",
                cpu_pass.momentum_finite_step_correction_j,
                gpu_pass.momentum_finite_step_correction_j);
            require_sponge_term_close(
                "momentum storage",
                cpu_pass.momentum_storage_rounding_j,
                gpu_pass.momentum_storage_rounding_j,
                momentum_parent);
            require_sponge_term_close(
                "dry internal",
                cpu_pass.dry_internal_change_j,
                gpu_pass.dry_internal_change_j);
            require_sponge_term_close(
                "water sensible",
                cpu_pass.water_sensible_change_j,
                gpu_pass.water_sensible_change_j);
            require_sponge_term_close(
                "phase reference",
                cpu_pass.phase_reference_change_j,
                gpu_pass.phase_reference_change_j);
            require_sponge_term_close(
                "carrier kinetic",
                cpu_pass.carrier_kinetic_change_j,
                gpu_pass.carrier_kinetic_change_j);
            require_sponge_term_close(
                "gravitational",
                cpu_pass.gravitational_change_j,
                gpu_pass.gravitational_change_j);
            require_sponge_term_close(
                "subgrid TKE",
                cpu_pass.subgrid_tke_change_j,
                gpu_pass.subgrid_tke_change_j);
            require_sponge_term_close(
                "committed energy",
                cpu_pass.committed_energy_change_j,
                gpu_pass.committed_energy_change_j);
            require_sponge_term_close(
                "closure residual",
                cpu_pass.closure_residual_j,
                gpu_pass.closure_residual_j,
                energy_parent);

            const double binary64_momentum_tolerance =
                8.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, momentum_parent);
            REQUIRE(std::abs(
                gpu_pass.carrier_kinetic_change_j
                - (gpu_pass.momentum_requested_linear_work_j
                   + gpu_pass.momentum_finite_step_correction_j
                   + gpu_pass.momentum_storage_rounding_j))
                <= binary64_momentum_tolerance);
            const double recomposed_energy =
                gpu_pass.dry_internal_change_j
                + gpu_pass.water_sensible_change_j
                + gpu_pass.phase_reference_change_j
                + gpu_pass.carrier_kinetic_change_j
                + gpu_pass.gravitational_change_j
                + gpu_pass.subgrid_tke_change_j;
            const double binary64_energy_tolerance =
                8.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, energy_parent);
            REQUIRE(std::abs(
                gpu_pass.committed_energy_change_j
                - recomposed_energy)
                <= binary64_energy_tolerance);
            REQUIRE(std::abs(gpu_pass.closure_residual_j)
                <= binary64_energy_tolerance);
        }
    }
    shutdown_compute_backend_runtime();
    reset_damping_globals_to_defaults();
}

TEST_CASE("device sponge attribution preserves ordered pass ownership",
          "[core][rayleigh_damping][energy][SCI-161]")
{
    seed_grid_and_fields();
    qv.fill(0.01f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[i][j][k] = static_cast<float>(
                    8.0 + 0.2 * i + 0.05 * k);
                v[i][j][k] = static_cast<float>(
                    -3.0 + 0.1 * j - 0.03 * k);
                w[i][j][k] = static_cast<float>(
                    1.0 + 0.02 * (i + j + k));
                theta[i][j][k] = static_cast<float>(304.0 + 0.1 * k);
            }
    constexpr float dt_step = 0.75f;
    constexpr int lateral_width = 1;
    std::vector<float> profiles(
        static_cast<std::size_t>(7 * NZ), 0.0f);
    for (int k = 0; k < NZ; ++k)
    {
        profiles[k] = k >= NZ - 3 ? 0.08f : 0.0f;
        profiles[NZ + k] = k >= NZ - 2 ? 0.18f : 0.0f;
        profiles[2 * NZ + k] = 1.5f;
        profiles[3 * NZ + k] = -0.5f;
        profiles[4 * NZ + k] = 300.0f + 0.2f * k;
        profiles[6 * NZ + k] = k >= NZ - 3 ? 0.11f : 0.0f;
    }
    profiles[5 * NZ] = 0.06f;

    Field3D committed_u = u;
    Field3D committed_v = v;
    Field3D committed_w = w;
    Field3D committed_theta = theta;
    const float dt_safe = std::max(dt_step, 1.0e-12f);
    const auto factor = [&](float alpha)
    {
        return 1.0f - std::exp(-alpha * dt_safe);
    };
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const float u_ref = profiles[2 * NZ + k];
                const float v_ref = profiles[3 * NZ + k];
                const float theta_ref = profiles[4 * NZ + k];
                const float f_upper = factor(profiles[k]);
                const float f_upper_w = factor(profiles[6 * NZ + k]);
                committed_u[i][j][k] +=
                    f_upper * (u_ref - committed_u[i][j][k]);
                committed_v[i][j][k] +=
                    f_upper * (v_ref - committed_v[i][j][k]);
                committed_w[i][j][k] +=
                    f_upper_w * (0.0f - committed_w[i][j][k]);
                committed_theta[i][j][k] +=
                    f_upper * (theta_ref - committed_theta[i][j][k]);

                const float f_w_only = factor(profiles[NZ + k]);
                committed_w[i][j][k] +=
                    f_w_only * (0.0f - committed_w[i][j][k]);

                const int distance = std::min(
                    std::min(i, NR - 1 - i),
                    std::min(j, NTH - 1 - j));
                if (distance < lateral_width)
                {
                    const float f_lateral =
                        factor(profiles[5 * NZ + distance]);
                    committed_u[i][j][k] +=
                        f_lateral * (u_ref - committed_u[i][j][k]);
                    committed_v[i][j][k] +=
                        f_lateral * (v_ref - committed_v[i][j][k]);
                    committed_w[i][j][k] +=
                        f_lateral * (0.0f - committed_w[i][j][k]);
                    committed_theta[i][j][k] +=
                        f_lateral
                        * (theta_ref - committed_theta[i][j][k]);
                }
            }

    const SimulationStateView state = global_simulation_state();
    const tmv::SpongeEnergyReportV1 report =
        attribute_state_relaxation_cartesian_device_commit(
            state, dt_step, profiles.data(),
            static_cast<int>(profiles.size()), lateral_width,
            true, true, true,
            committed_u.data(), committed_v.data(),
            committed_w.data(), committed_theta.data());
    REQUIRE(report.complete);
    REQUIRE(report.upper.complete);
    REQUIRE(report.w_only.complete);
    REQUIRE(report.lateral.complete);
    REQUIRE(report.upper.applied);
    REQUIRE(report.w_only.applied);
    REQUIRE(report.lateral.applied);
    for (const auto* pass :
         {&report.upper, &report.w_only, &report.lateral})
    {
        CHECK(pass->carrier_kinetic_change_j
              == pass->momentum_requested_linear_work_j
               + pass->momentum_finite_step_correction_j
               + pass->momentum_storage_rounding_j);
        CHECK(pass->closure_residual_j == 0.0);
    }

    constexpr double cv_d =
        physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;
    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    const double cell_volume = dr * dtheta * dz;
    double endpoint_change = 0.0;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double dry_mass = rho[i][j][k] * cell_volume;
                const double moist_mass =
                    dry_mass * (1.0 + qv[i][j][k]);
                const double exner = std::pow(
                    p[i][j][k]
                        / physical_constants::reference_pressure_pa,
                    kappa);
                endpoint_change += dry_mass
                    * (cv_d + qv[i][j][k] * cv_v) * exner
                    * (committed_theta[i][j][k] - theta[i][j][k]);
                const double old_u = u[i][j][k];
                const double old_v = v[i][j][k];
                const double old_w = w[i][j][k];
                const double new_u = committed_u[i][j][k];
                const double new_v = committed_v[i][j][k];
                const double new_w = committed_w[i][j][k];
                endpoint_change += 0.5 * moist_mass * (
                    new_u * new_u - old_u * old_u
                    + new_v * new_v - old_v * old_v
                    + new_w * new_w - old_w * old_w);
            }
    const double report_change =
        report.upper.committed_energy_change_j
        + report.w_only.committed_energy_change_j
        + report.lateral.committed_energy_change_j;
    CHECK(report_change == Approx(endpoint_change).epsilon(2.0e-12));

    reset_damping_globals_to_defaults();
}

// ─────────────────────────────────────────────────────────────
// enabled=false short-circuits the sponge
// ─────────────────────────────────────────────────────────────

TEST_CASE("C3 damping.enabled=false leaves velocity untouched",
          "[core][rayleigh_damping][c3]")
{
    seed_grid_and_fields();
    global_damping_enabled = false;
    global_damping_z_d     = 2500.0;   // would be active for k>=5 if enabled
    global_damping_alpha   = 0.5;
    reset_rayleigh_damping();

    apply_rayleigh_damping(1.0);

    // No relaxation: every cell keeps its perturbation value.
    for (int k = 0; k < kNZ; ++k)
    {
        REQUIRE(u[1][1][k] == Approx(kU_PERTURB));
    }
    reset_damping_globals_to_defaults();
}

// ─────────────────────────────────────────────────────────────
// alpha=0 is a no-op even when enabled
// ─────────────────────────────────────────────────────────────

TEST_CASE("C3 damping.alpha=0 produces no relaxation",
          "[core][rayleigh_damping][c3]")
{
    seed_grid_and_fields();
    global_damping_enabled = true;
    global_damping_z_d     = 2500.0;
    global_damping_alpha   = 0.0;
    reset_rayleigh_damping();

    apply_rayleigh_damping(1.0);

    for (int k = 0; k < kNZ; ++k)
    {
        REQUIRE(u[1][1][k] == Approx(kU_PERTURB));
    }
    reset_damping_globals_to_defaults();
}

// ─────────────────────────────────────────────────────────────
// alpha > 0 relaxes toward base state in the sponge zone
// ─────────────────────────────────────────────────────────────

TEST_CASE("C3 damping.alpha>0 relaxes top-of-domain velocity",
          "[core][rayleigh_damping][c3]")
{
    seed_grid_and_fields();
    global_damping_enabled = true;
    global_damping_z_d     = 2500.0;     // sponge covers centers k=5..9 (z=2750..4750)
    global_damping_alpha   = 0.5;        // strong damping
    reset_rayleigh_damping();

    apply_rayleigh_damping(1.0);

    // k=0 (z=250 m) is far below z_d → untouched.
    REQUIRE(u[1][1][0] == Approx(kU_PERTURB));
    // The threshold is evaluated at scalar centers: k=4 is at 2250 m and k=5
    // is at 2750 m. This pins SCI-019's half-cell origin in the sponge path.
    REQUIRE(u[1][1][4] == Approx(kU_PERTURB));
    REQUIRE(u[1][1][5] < kU_PERTURB);
    // Top cell (k=NZ-1=9, z=4750m) is in the sponge → should be relaxed
    // toward 0 (u_ref=0 from the zero-wind hodograph).
    REQUIRE(u[1][1][kNZ - 1] < kU_PERTURB);
    REQUIRE(u[1][1][kNZ - 1] > 0.0f);     // partial relaxation, not zeroed
    reset_damping_globals_to_defaults();
}

// ─────────────────────────────────────────────────────────────
// Higher alpha relaxes faster
// ─────────────────────────────────────────────────────────────

TEST_CASE("C3 higher damping.alpha relaxes faster",
          "[core][rayleigh_damping][c3]")
{
    seed_grid_and_fields();
    global_damping_enabled = true;
    global_damping_z_d     = 2500.0;
    global_damping_alpha   = 0.05;
    reset_rayleigh_damping();
    apply_rayleigh_damping(1.0);
    const float u_small_alpha = u[1][1][kNZ - 1];

    seed_grid_and_fields();
    global_damping_alpha = 0.5;
    reset_rayleigh_damping();
    apply_rayleigh_damping(1.0);
    const float u_big_alpha = u[1][1][kNZ - 1];

    // Bigger alpha → faster decay → smaller residual perturbation.
    REQUIRE(u_big_alpha < u_small_alpha);
    REQUIRE(u_big_alpha < kU_PERTURB);
    reset_damping_globals_to_defaults();
}

TEST_CASE("C3 implicit damping uses the backward-Euler relaxation equation",
          "[core][rayleigh_damping][c3][implicit][reference]")
{
    seed_grid_and_fields();
    global_damping_enabled = true;
    global_damping_z_d = 2500.0;
    global_damping_alpha = 1.0;
    global_damping_mode = DampingMode::Implicit;
    global_w_only_top_sponge = false;
    global_lateral_sponge_enabled = false;
    reset_rayleigh_damping();

    apply_rayleigh_damping(1.0);

    const double alpha = rayleigh_alpha_profile().back();
    const double expected = kU_PERTURB / (1.0 + alpha);
    REQUIRE(u[1][1][kNZ - 1] == Approx(expected).epsilon(1.0e-6));
    reset_damping_globals_to_defaults();
}

// ─────────────────────────────────────────────────────────────
// z_d controls the sponge layer height
// ─────────────────────────────────────────────────────────────

TEST_CASE("C3 raising damping.z_d shrinks the sponge zone",
          "[core][rayleigh_damping][c3]")
{
    // Low z_d → bigger sponge zone → mid-level cells are damped.
    seed_grid_and_fields();
    global_damping_enabled = true;
    global_damping_z_d     = 1000.0;
    global_damping_alpha   = 0.5;
    reset_rayleigh_damping();
    apply_rayleigh_damping(1.0);
    const float u_mid_low_zd = u[1][1][kNZ / 2];

    // High z_d → tiny sponge zone → mid-level cells are NOT damped.
    seed_grid_and_fields();
    global_damping_z_d = 4000.0;
    reset_rayleigh_damping();
    apply_rayleigh_damping(1.0);
    const float u_mid_high_zd = u[1][1][kNZ / 2];

    REQUIRE(u_mid_low_zd  < kU_PERTURB);    // damped
    REQUIRE(u_mid_high_zd == Approx(kU_PERTURB));  // outside sponge
    reset_damping_globals_to_defaults();
}

// ─────────────────────────────────────────────────────────────
// z_d <= 0 sentinel falls back to legacy depth_fraction=0.3
// ─────────────────────────────────────────────────────────────

TEST_CASE("C3 z_d sentinel <=0 preserves legacy depth_fraction behavior",
          "[core][rayleigh_damping][c3][backward_compat]")
{
    seed_grid_and_fields();
    global_damping_enabled = true;
    global_damping_z_d     = -1.0;       // legacy fallback
    global_damping_alpha   = 0.5;
    reset_rayleigh_damping();
    apply_rayleigh_damping(1.0);

    // Legacy: sponge covers top 30% of NZ*dz = 5000m → z > 3500m.
    // k=0 (z=250) untouched. k=NZ-1 (z=4750) damped.
    REQUIRE(u[1][1][0]        == Approx(kU_PERTURB));
    REQUIRE(u[1][1][kNZ - 1]   < kU_PERTURB);
    reset_damping_globals_to_defaults();
}

// ─────────────────────────────────────────────────────────────
// Defaults preserve v0 behavior (enabled, alpha=0.05, legacy fallback)
// ─────────────────────────────────────────────────────────────

TEST_CASE("C3 globals default to legacy v0 sponge configuration",
          "[core][rayleigh_damping][c3][defaults]")
{
    REQUIRE(global_damping_enabled == true);
    REQUIRE(global_damping_z_d     == Approx(-1.0));      // sentinel
    REQUIRE(global_damping_alpha   == Approx(0.05));
    REQUIRE(global_damping_mode    == DampingMode::Explicit);
}

// ─────────────────────────────────────────────────────────────
// Smell #8 closure: w-only top sponge
// ─────────────────────────────────────────────────────────────

namespace
{

// Seed perturbations in EVERY prognostic field so the w-only branch can be
// distinguished from the legacy all-field branch by what stays vs goes.
constexpr float kV_PERTURB     = 5.0f;
constexpr float kW_PERTURB     = 3.0f;
constexpr float kTHETA_PERTURB = 305.0f;   // base 300
constexpr float kRHO_PERTURB   = 1.3f;     // base 1.225
constexpr float kP_PERTURB     = 99500.0f; // base 100000

void seed_grid_and_all_fields()
{
    seed_grid_and_fields();
    for (int i = 0; i < kNR; ++i)
        for (int j = 0; j < kNTH; ++j)
            for (int k = 0; k < kNZ; ++k)
            {
                v[i][j][k]     = kV_PERTURB;
                w[i][j][k]     = kW_PERTURB;
                theta[i][j][k] = kTHETA_PERTURB;
                rho[i][j][k]   = kRHO_PERTURB;
                p[i][j][k]     = kP_PERTURB;
            }
}

}  // namespace

TEST_CASE("w-only top sponge: adds extra w damping in the top depth_cells",
          "[core][rayleigh_damping][smell8][w_only]")
{
    // Isolate the w-only absorber by disabling the legacy all-field path
    // (alpha=0). With only w-only active, w is the only field damped.
    seed_grid_and_all_fields();
    global_damping_enabled    = true;
    global_damping_alpha      = 0.0;     // legacy OFF
    global_w_only_top_sponge  = true;
    global_w_only_depth_cells = 4;       // top 4 cells: k=6,7,8,9
    global_w_only_alpha_max   = 1.0;     // strong, fast relaxation for the test
    reset_rayleigh_damping();

    apply_rayleigh_damping(1.0);

    // Top cell at the lid (k=NZ-1=9) has the strongest damping; w drops
    // significantly toward 0 in one step (factor = 1 - exp(-1*1) ~ 0.63).
    REQUIRE(w[1][1][kNZ - 1] < kW_PERTURB * 0.5f);
    REQUIRE(w[1][1][kNZ - 1] > 0.0f);   // partial relaxation, not zeroed exactly

    // Just inside the band (k=6, ramp toe) sees alpha ~ 0 → w nearly untouched.
    REQUIRE(w[1][1][6] == Approx(kW_PERTURB).margin(0.1f));

    // Below the band (k=5 and lower) — strictly untouched.
    for (int k = 0; k <= 5; ++k)
    {
        REQUIRE(w[1][1][k] == Approx(kW_PERTURB));
    }
    reset_damping_globals_to_defaults();
}

TEST_CASE("w-only top sponge: does not touch u, v, theta, rho, p when legacy alpha=0",
          "[core][rayleigh_damping][smell8][w_only]")
{
    // With the legacy path off (alpha=0), the w-only absorber should be
    // the ONLY damping mechanism. Every horizontal-velocity and
    // thermodynamic field stays bit-exactly preserved — the absorber
    // never touches anything but w. This is the cyclostrophic-balance
    // preservation guarantee.
    seed_grid_and_all_fields();
    global_damping_enabled    = true;
    global_damping_alpha      = 0.0;     // legacy OFF
    global_w_only_top_sponge  = true;
    global_w_only_depth_cells = 4;
    global_w_only_alpha_max   = 1.0;
    reset_rayleigh_damping();

    apply_rayleigh_damping(1.0);

    for (int k = 0; k < kNZ; ++k)
    {
        REQUIRE(u[1][1][k]     == Approx(kU_PERTURB));
        REQUIRE(v[1][1][k]     == Approx(kV_PERTURB));
        REQUIRE(theta[1][1][k] == Approx(kTHETA_PERTURB));
        REQUIRE(rho[1][1][k]   == Approx(kRHO_PERTURB));
        REQUIRE(p[1][1][k]     == Approx(kP_PERTURB));
    }
    reset_damping_globals_to_defaults();
}

TEST_CASE("w-only top sponge: additive with legacy path",
          "[core][rayleigh_damping][smell8][w_only][additive]")
{
    // The w-only absorber is meant to layer ON TOP of the legacy all-field
    // sponge, not replace it. Verify w is more strongly damped when both
    // are active than when only legacy is active. Other fields see only
    // the legacy damping in both runs.
    seed_grid_and_all_fields();
    global_damping_enabled    = true;
    global_damping_z_d        = 3500.0;  // legacy active for z >= 3500 = k >= 7
    global_damping_alpha      = 0.05;    // legacy on at production strength
    global_w_only_top_sponge  = false;   // LEGACY ONLY
    reset_rayleigh_damping();
    apply_rayleigh_damping(1.0);
    const float w_lid_legacy = w[1][1][kNZ - 1];
    const float u_lid_legacy = u[1][1][kNZ - 1];

    seed_grid_and_all_fields();
    global_damping_alpha      = 0.05;    // legacy still on
    global_w_only_top_sponge  = true;    // ADD w-only on top
    global_w_only_depth_cells = 4;
    global_w_only_alpha_max   = 1.0;
    reset_rayleigh_damping();
    apply_rayleigh_damping(1.0);
    const float w_lid_both   = w[1][1][kNZ - 1];
    const float u_lid_both   = u[1][1][kNZ - 1];

    // w sees both layers → more damping than legacy alone.
    REQUIRE(w_lid_both < w_lid_legacy);
    // u sees ONLY the legacy layer (w-only does not touch u) → identical.
    REQUIRE(u_lid_both == Approx(u_lid_legacy));
    reset_damping_globals_to_defaults();
}

TEST_CASE("w-only top sponge: legacy path unchanged when flag OFF",
          "[core][rayleigh_damping][smell8][w_only][backward_compat]")
{
    // Sanity check: with the new flag OFF, the legacy alpha=0.05 +
    // depth_fraction=0.3 path is unchanged. u still relaxes (it always did).
    seed_grid_and_all_fields();
    global_damping_enabled    = true;
    global_w_only_top_sponge  = false;   // OFF — legacy path only
    global_damping_z_d        = -1.0;    // legacy fallback (top 30%)
    global_damping_alpha      = 0.5;     // strong, observable in one step
    reset_rayleigh_damping();

    apply_rayleigh_damping(1.0);

    // Legacy path damps u (and v) toward env wind (=0 from the stub).
    REQUIRE(u[1][1][kNZ - 1]   < kU_PERTURB);
    REQUIRE(v[1][1][kNZ - 1]   < kV_PERTURB);
    // And damps theta toward theta0_base[k]=300 (we seeded 305).
    REQUIRE(theta[1][1][kNZ - 1] < kTHETA_PERTURB);
    // w also damps toward 0.
    REQUIRE(w[1][1][kNZ - 1]   < kW_PERTURB);
    reset_damping_globals_to_defaults();
}

TEST_CASE("w-only top sponge: depth_cells controls the band size",
          "[core][rayleigh_damping][smell8][w_only]")
{
    // Narrow band: top 2 cells only damped → k=7 untouched.
    seed_grid_and_all_fields();
    global_damping_enabled    = true;
    global_damping_alpha      = 0.0;     // isolate w-only
    global_w_only_top_sponge  = true;
    global_w_only_depth_cells = 2;
    global_w_only_alpha_max   = 1.0;
    reset_rayleigh_damping();
    apply_rayleigh_damping(1.0);
    REQUIRE(w[1][1][7] == Approx(kW_PERTURB));         // outside narrow band
    REQUIRE(w[1][1][kNZ - 1] < kW_PERTURB * 0.5f);     // at the lid: damped
    reset_damping_globals_to_defaults();

    // Wide band: top 6 cells → k=7 IS now in the band → must be damped.
    seed_grid_and_all_fields();
    global_damping_alpha      = 0.0;     // isolate w-only
    global_w_only_top_sponge  = true;
    global_w_only_depth_cells = 6;
    global_w_only_alpha_max   = 1.0;
    reset_rayleigh_damping();
    apply_rayleigh_damping(1.0);
    REQUIRE(w[1][1][7] < kW_PERTURB);                  // inside wide band
    reset_damping_globals_to_defaults();
}

TEST_CASE("w-only top sponge: defaults match production configuration",
          "[core][rayleigh_damping][smell8][w_only][defaults]")
{
    // Production defaults since 2026-05-25 (Smell #8 closure landed):
    // additive w-only absorber ON, 8-cell band, alpha_max=0.20 1/s.
    // Verified by 7/7 cgrid integration suite with strict improvements
    // over legacy (max_w drops 17-72% in calm fixtures, cyclostrophic
    // vortex peak preserved bit-exactly).
    REQUIRE(global_w_only_top_sponge  == true);
    REQUIRE(global_w_only_depth_cells == 8);
    REQUIRE(global_w_only_alpha_max   == Approx(0.20));
}

TEST_CASE("rayleigh damping can operate on an explicit local state",
          "[core][rayleigh_damping][state]")
{
    seed_grid_and_all_fields();
    u[1][1][kNZ - 1] = 123.0f;

    int local_nr = 2;
    int local_nth = 2;
    int local_nz = 6;
    double local_dr = 500.0;
    double local_dz = kDz;
    double local_dt = 1.0;
    double local_dtheta = 500.0;
    GridGeometry local_geometry;
    local_geometry.initialize(local_nr, local_nth, local_nz,
                              local_dr, local_dz, local_dtheta,
                              CoordinateSystem::Cartesian,
                              StaggerType::Collocated);

    Field3D local_rho(local_nr, local_nth, local_nz, kRHO_PERTURB);
    Field3D local_p(local_nr, local_nth, local_nz, kP_PERTURB);
    Field3D local_u(local_nr, local_nth, local_nz, kU_PERTURB);
    Field3D local_w(local_nr, local_nth, local_nz, kW_PERTURB);
    Field3D local_v(local_nr, local_nth, local_nz, kV_PERTURB);
    Field3D local_tracer(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_theta(local_nr, local_nth, local_nz, kTHETA_PERTURB);
    Field3D local_qv(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qc(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qr(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qi(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qs(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qh(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qg(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_tke(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtheta_dt_rad(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtheta_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dqv_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_du_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dv_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtke_dt_pbl(local_nr, local_nth, local_nz, 0.0f);

    std::vector<double> local_rho0(local_nz, 1.225);
    std::vector<double> local_p0(local_nz, 100000.0);
    std::vector<double> local_qv0(local_nz, 0.0);
    std::vector<double> local_theta0(local_nz, 300.0);
    std::vector<double> local_u0(local_nz, 0.0);
    std::vector<double> local_v0(local_nz, 0.0);
    double local_coriolis = 0.0;
    double local_time = 0.0;
    RayleighDampingRuntimeState local_rayleigh_runtime;

    SimulationStateView local_state{
        {local_nr, local_nth, local_nz, local_dr, local_dz, local_dt, local_dtheta,
         local_geometry},
        {local_rho, local_p, local_u, local_w, local_v, local_tracer,
         local_theta, local_qv, local_qc, local_qr, local_qi, local_qs,
         local_qh, local_qg, local_tke},
        {local_rho0, local_p0, local_qv0, local_theta0, local_u0, local_v0},
        {local_dtheta_dt_rad, local_dtheta_dt_pbl, local_dqv_dt_pbl,
         local_du_dt_pbl, local_dv_dt_pbl, local_dtke_dt_pbl},
        local_coriolis,
        local_time,
        {}, nullptr, nullptr, nullptr, nullptr, &local_rayleigh_runtime};

    RuntimeConfigurationState local_configuration;
    local_configuration.damping_enabled   = true;
    local_configuration.damping_z_d       = 1000.0;
    local_configuration.damping_alpha     = 0.5;
    local_configuration.w_only_top_sponge = false;
    local_state.configuration             = &local_configuration;
    local_state.domain_frame              = &local_configuration.domain_frame;
    reset_rayleigh_damping(local_rayleigh_runtime);

    apply_rayleigh_damping(local_state, 1.0);

    REQUIRE(local_u[1][1][local_nz - 1] < kU_PERTURB);
    REQUIRE(local_v[1][1][local_nz - 1] < kV_PERTURB);
    REQUIRE(local_theta[1][1][local_nz - 1] < kTHETA_PERTURB);
    REQUIRE(u[1][1][kNZ - 1] == Approx(123.0f));

    int other_nr = local_nr;
    int other_nth = local_nth;
    int other_nz = local_nz;
    double other_dr = local_dr;
    double other_dz = 250.0;
    double other_dt = local_dt;
    double other_dtheta = local_dtheta;
    GridGeometry other_geometry;
    other_geometry.initialize(other_nr, other_nth, other_nz,
                              other_dr, other_dz, other_dtheta,
                              CoordinateSystem::Cartesian,
                              StaggerType::Collocated);
    RayleighDampingRuntimeState other_rayleigh_runtime;
    SimulationStateView other_state{
        {other_nr, other_nth, other_nz, other_dr, other_dz, other_dt,
         other_dtheta, other_geometry},
        local_state.fields,
        local_state.base,
        local_state.tendencies,
        local_coriolis,
        local_time,
        {}, nullptr, nullptr, nullptr, nullptr, &other_rayleigh_runtime};

    other_state.configuration = &local_configuration;
    other_state.domain_frame  = &local_configuration.domain_frame;

    const std::vector<double> local_profile =
        rayleigh_alpha_profile(local_state);
    const std::vector<double> other_profile =
        rayleigh_alpha_profile(other_state);
    REQUIRE(local_profile != other_profile);
    REQUIRE(rayleigh_alpha_profile(local_state) == local_profile);
    reset_damping_globals_to_defaults();
}

TEST_CASE("sponge report closes exact collocated moist energy",
          "[core][rayleigh_damping][energy][SCI-155]")
{
    seed_grid_and_all_fields();
    qv.fill(0.01f);
    global_damping_enabled = true;
    global_damping_z_d = 2500.0;
    global_damping_alpha = 0.2;
    global_w_only_top_sponge = false;
    global_lateral_sponge_enabled = false;
    reset_rayleigh_damping();

    constexpr double cv_d =
        physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;
    const auto cell_energies = [&]()
    {
        std::vector<std::array<double, 2>> energies;
        const double volume = dr * dtheta * dz;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 1; k < NZ - 1; ++k)
                {
                    const double dry_mass = rho[i][j][k] * volume;
                    const double vapor = qv[i][j][k];
                    const double temperature = theta[i][j][k] * std::pow(
                        p[i][j][k]
                            / physical_constants::reference_pressure_pa,
                        kappa);
                    const double mixture_mass =
                        dry_mass * (1.0 + vapor);
                    const double stored_u = u[i][j][k];
                    const double stored_v = v[i][j][k];
                    const double stored_w = w[i][j][k];
                    energies.push_back(
                        {dry_mass * (cv_d + vapor * cv_v) * temperature,
                         0.5 * mixture_mass * (stored_u * stored_u + stored_v * stored_v + stored_w * stored_w)});
                }
        return energies;
    };

    const auto before = cell_energies();

    const tmv::SpongeEnergyReportV1 energy =
        apply_rayleigh_damping(1.0);

    const auto after = cell_energies();
    REQUIRE(after.size() == before.size());
    // Subtract per-cell reservoirs rather than the large whole-domain inventory.
    double thermal_change = 0.0;
    double kinetic_change = 0.0;
    for (std::size_t cell = 0; cell < before.size(); ++cell)
    {
        thermal_change += after[cell][0] - before[cell][0];
        kinetic_change += after[cell][1] - before[cell][1];
    }

    REQUIRE(energy.complete);
    REQUIRE(energy.upper.complete);
    REQUIRE(energy.upper.applied);
    REQUIRE_FALSE(energy.w_only.applied);
    CHECK(energy.upper.carrier_kinetic_change_j
          == Approx(
              energy.upper.momentum_requested_linear_work_j
              + energy.upper.momentum_finite_step_correction_j
              + energy.upper.momentum_storage_rounding_j));
    CHECK(energy.upper.committed_energy_change_j == Approx(thermal_change + kinetic_change).epsilon(1.0e-9));
    CHECK(energy.upper.dry_internal_change_j + energy.upper.water_sensible_change_j ==
          Approx(thermal_change).epsilon(1.0e-9));
    CHECK(energy.upper.carrier_kinetic_change_j == Approx(kinetic_change).epsilon(1.0e-9));
    CHECK(energy.upper.closure_residual_j == 0.0);

    reset_damping_globals_to_defaults();
}

TEST_CASE("C-grid upper sponge uses vertical-face height and closes energy",
          "[core][rayleigh_damping][cgrid][energy][SCI-155]")
{
    constexpr int nr = 5, nth = 5, nz = 4;
    constexpr double spacing = 1000.0;
    SimulationStorageState storage{
        {nr, nth, nz, spacing, spacing, 1.0, spacing}};
    storage.grid.geometry.initialize(
        nr, nth, nz, spacing, spacing, spacing,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& f = storage.fields;
    f.rho.resize(nr, nth, nz, 1.1f);
    f.p.resize(nr, nth, nz, 90000.0f);
    f.u.resize(nr, nth, nz, 12.0f);
    f.v.resize(nr, nth, nz, -5.0f);
    f.w.resize(nr, nth, nz, 3.0f);
    f.theta.resize(nr, nth, nz, 305.0f);
    f.qv.resize(nr, nth, nz, 0.01f);
    f.qc.resize(nr, nth, nz, 0.0f);
    f.qr.resize(nr, nth, nz, 0.0f);
    f.qi.resize(nr, nth, nz, 0.0f);
    f.qs.resize(nr, nth, nz, 0.0f);
    f.qg.resize(nr, nth, nz, 0.0f);
    f.qh.resize(nr, nth, nz, 0.0f);
    f.tke.resize(nr, nth, nz, 0.0f);
    storage.base.theta0.assign(nz, 300.0);

    storage.configuration.damping_enabled        = true;
    storage.configuration.damping_z_d            = 2500.0;
    storage.configuration.damping_alpha          = 0.2;
    storage.configuration.w_only_top_sponge      = false;
    storage.configuration.lateral_sponge_enabled = false;
    storage.configuration.lateral_bc_west        = tmv::init::config::LateralBC::Rigid;
    storage.configuration.lateral_bc_east        = tmv::init::config::LateralBC::Rigid;
    storage.configuration.lateral_bc_south       = tmv::init::config::LateralBC::Rigid;
    storage.configuration.lateral_bc_north       = tmv::init::config::LateralBC::Rigid;
    reset_rayleigh_damping(storage.runtime.rayleigh);
    SimulationStateView state = make_simulation_state_view(storage);

    constexpr double cv_d =
        physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;
    const float theta_at_onset = f.theta(2, 2, 2);
    const float w_above_onset = f.w(2, 2, 2);
    const Field3D u_initial = f.u;
    const Field3D v_initial = f.v;
    const Field3D w_initial = f.w;
    const Field3D theta_initial = f.theta;
    const tmv::SpongeEnergyReportV1 energy =
        apply_rayleigh_damping(state, 1.0);
    double direct_thermal_change = 0.0;
    double direct_kinetic_change = 0.0;
    const double volume = spacing * spacing * spacing;
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double dry_mass = f.rho(i, j, k) * volume;
                const double vapor = f.qv(i, j, k);
                const double mixture_mass = dry_mass * (1.0 + vapor);
                const double exner = std::pow(
                    f.p(i, j, k)
                        / physical_constants::reference_pressure_pa,
                    kappa);
                direct_thermal_change += dry_mass * (cv_d + vapor * cv_v)
                    * (f.theta(i, j, k) - theta_initial(i, j, k))
                    * exner;
                const auto kinetic_sum =
                    [&](const Field3D& uf, const Field3D& vf,
                        const Field3D& wf)
                {
                    const double w_bottom =
                        k == 0 ? 0.0 : wf(i, j, k - 1);
                    const double u_w = uf(i - 1, j, k);
                    const double u_e = uf(i, j, k);
                    const double v_s = vf(i, j - 1, k);
                    const double v_n = vf(i, j, k);
                    const double w_top = wf(i, j, k);
                    return u_w * u_w + u_e * u_e
                        + v_s * v_s + v_n * v_n
                        + w_bottom * w_bottom
                        + w_top * w_top;
                };
                direct_kinetic_change += 0.25 * mixture_mass
                    * (kinetic_sum(f.u, f.v, f.w)
                       - kinetic_sum(u_initial, v_initial, w_initial));
            }
    const double direct_change =
        direct_thermal_change + direct_kinetic_change;

    REQUIRE(f.theta(2, 2, 2) == theta_at_onset);
    REQUIRE(f.w(2, 2, 2) < w_above_onset);
    REQUIRE(energy.upper.complete);
    INFO("thermal difference="
         << (energy.upper.dry_internal_change_j
             + energy.upper.water_sensible_change_j
             - direct_thermal_change)
         << " kinetic difference="
         << (energy.upper.carrier_kinetic_change_j
             - direct_kinetic_change));
    CHECK(energy.upper.committed_energy_change_j
          == Approx(direct_change).epsilon(2.0e-12));
    CHECK(energy.upper.carrier_kinetic_change_j
          == Approx(
              energy.upper.momentum_requested_linear_work_j
              + energy.upper.momentum_finite_step_correction_j
              + energy.upper.momentum_storage_rounding_j));

    reset_damping_globals_to_defaults();
}

TEST_CASE("terrain Rayleigh sponge follows physical height within each column",
          "[core][rayleigh_damping][terrain][physical-height]")
{
    constexpr int nr = 4, nth = 4, nz = 4;
    constexpr double dx = 500.0, dy = 500.0, nominal_dz = 1000.0;
    constexpr double z_top = 4000.0;
    SimulationStorageState storage{
        {nr, nth, nz, dx, nominal_dz, 1.0, dy}};
    storage.grid.geometry.initialize(
        nr, nth, nz, dx, nominal_dz, dy,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.fields.rho.resize(nr, nth, nz, 1.0f);
    storage.fields.p.resize(nr, nth, nz, 90000.0f);
    storage.fields.u.resize(nr, nth, nz, 10.0f);
    storage.fields.v.resize(nr, nth, nz, 0.0f);
    storage.fields.w.resize(nr, nth, nz, 0.0f);
    storage.fields.theta.resize(nr, nth, nz, 305.0f);
    storage.base.theta0.assign(nz, 300.0);

    auto& terrain = storage.runtime.terrain;
    terrain.config.scheme_id = "bell";
    terrain.metrics.z.resize(nr, nth, nz, 0.0);
    terrain.metrics.z_face.resize(nr, nth, nz + 1, 0.0);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            const double ground = i >= 2 ? 1000.0 : 0.0;
            const double thickness = (z_top - ground) / nz;
            for (int k = 0; k <= nz; ++k)
                terrain.metrics.z_face(i, j, k) = ground + k * thickness;
            for (int k = 0; k < nz; ++k)
                terrain.metrics.z(i, j, k) =
                    ground + (static_cast<double>(k) + 0.5) * thickness;
        }

    storage.configuration.damping_enabled   = true;
    storage.configuration.damping_z_d       = 2500.0;
    storage.configuration.damping_alpha     = 0.5;
    storage.configuration.w_only_top_sponge = false;
    reset_rayleigh_damping(storage.runtime.rayleigh);
    SimulationStateView state = make_simulation_state_view(storage);

    apply_rayleigh_damping(state, 1.0);

    // At the same computational level, the low column center is exactly at
    // sponge onset while the elevated column center is 375 m into the layer.
    REQUIRE(storage.fields.theta(1, 1, 2) == Approx(305.0f));
    const double fraction = 375.0 / 1500.0;
    const double sine = std::sin(0.5 * 3.14159265358979323846 * fraction);
    const double alpha    = storage.configuration.damping_alpha * sine * sine;
    const double expected = 300.0 + 5.0 * std::exp(-alpha);
    REQUIRE(storage.fields.theta(2, 1, 2) ==
            Approx(expected).epsilon(1.0e-6));
    REQUIRE(storage.fields.theta(2, 1, 2) <
            storage.fields.theta(1, 1, 2));

    reset_damping_globals_to_defaults();
}

TEST_CASE("terrain lateral sponge samples references at staggered physical heights",
          "[core][rayleigh_damping][terrain][lateral][physical-height]")
{
    constexpr int nr = 4, nth = 4, nz = 4;
    constexpr double dx = 500.0, dy = 500.0, nominal_dz = 1000.0;
    constexpr double z_top = 4000.0;
    SimulationStorageState storage{
        {nr, nth, nz, dx, nominal_dz, 1.0, dy}};
    storage.grid.geometry.initialize(
        nr, nth, nz, dx, nominal_dz, dy,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.fields.rho.resize(nr, nth, nz, 1.0f);
    storage.fields.p.resize(nr, nth, nz, 90000.0f);
    storage.fields.u.resize(nr, nth, nz, 10.0f);
    storage.fields.v.resize(nr, nth, nz, -4.0f);
    storage.fields.w.resize(nr, nth, nz, 2.0f);
    storage.fields.theta.resize(nr, nth, nz, 305.0f);
    storage.fields.qv.resize(nr, nth, nz, 0.01f);
    storage.fields.qc.resize(nr, nth, nz, 0.0f);
    storage.fields.qr.resize(nr, nth, nz, 0.0f);
    storage.fields.qi.resize(nr, nth, nz, 0.0f);
    storage.fields.qs.resize(nr, nth, nz, 0.0f);
    storage.fields.qg.resize(nr, nth, nz, 0.0f);
    storage.fields.qh.resize(nr, nth, nz, 0.0f);
    storage.fields.tke.resize(nr, nth, nz, 0.0f);
    storage.base.theta0 = {300.0, 310.0, 320.0, 330.0};

    auto& terrain = storage.runtime.terrain;
    terrain.config.scheme_id = "bell";
    terrain.metrics.z.resize(nr, nth, nz, 0.0);
    terrain.metrics.z_face.resize(nr, nth, nz + 1, 0.0);
    terrain.metrics.Gx_face.resize(nr, nth, nz + 1, -0.1);
    terrain.metrics.Gy_face.resize(nr, nth, nz + 1, 0.05);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            const double ground = i >= 2 ? 1000.0 : 0.0;
            const double thickness = (z_top - ground) / nz;
            for (int k = 0; k <= nz; ++k)
                terrain.metrics.z_face(i, j, k) = ground + k * thickness;
            for (int k = 0; k < nz; ++k)
                terrain.metrics.z(i, j, k) =
                    ground + (static_cast<double>(k) + 0.5) * thickness;
        }

    storage.configuration.lateral_sponge_enabled = true;
    storage.configuration.lateral_sponge_width   = 1;
    storage.configuration.lateral_sponge_alpha   = 0.5;
    reset_rayleigh_damping(storage.runtime.rayleigh);
    SimulationStateView state = make_simulation_state_view(storage);

    const Field3D u_initial = storage.fields.u;
    const Field3D v_initial = storage.fields.v;
    const Field3D w_initial = storage.fields.w;
    const Field3D theta_initial = storage.fields.theta;
    const tmv::SpongeEnergyReportV1 energy =
        apply_lateral_damping(state, 1.0);

    const double factor = 1.0 - std::exp(-0.5);
    for (const int i : {0, nr - 1})
    {
        const double z_scalar = terrain.metrics.z(i, 1, 1);
        const double reference =
            terrain_transport::reference_profile_value_fp32(
                storage.base.theta0, z_scalar, nominal_dz);
        const double expected_theta = 305.0 + factor * (reference - 305.0);
        REQUIRE(storage.fields.theta(i, 1, 1) ==
                Approx(expected_theta).epsilon(1.0e-6));
        REQUIRE(storage.fields.u(i, 1, 1) ==
                Approx(10.0 * std::exp(-0.5)).epsilon(1.0e-6));
        REQUIRE(storage.fields.v(i, 1, 1) ==
                Approx(-4.0 * std::exp(-0.5)).epsilon(1.0e-6));
        REQUIRE(storage.fields.w(i, 1, 1) ==
                Approx(2.0 * std::exp(-0.5)).epsilon(1.0e-6));
    }
    REQUIRE(storage.fields.theta(0, 1, 1) !=
            Approx(storage.fields.theta(nr - 1, 1, 1)));
    // The width-one band must not mutate the physical interior.
    REQUIRE(storage.fields.theta(1, 1, 1) == Approx(305.0f));
    REQUIRE(energy.complete);
    REQUIRE(energy.lateral.complete);
    REQUIRE(energy.lateral.applied);

    constexpr double cv_d =
        physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;
    double direct_change = 0.0;
    const auto kinetic_sum =
        [&](const Field3D& uf, const Field3D& vf, const Field3D& wf,
            int i, int j, int k)
    {
        const double u_w = uf(i - 1, j, k);
        const double u_e = uf(i, j, k);
        const double v_s = vf(i, j - 1, k);
        const double v_n = vf(i, j, k);
        const double w_bottom = k == 0
            ? 0.5 * (u_w + u_e)
                    * (-terrain.metrics.Gx_face(i, j, 0))
                + 0.5 * (v_s + v_n)
                    * (-terrain.metrics.Gy_face(i, j, 0))
            : static_cast<double>(wf(i, j, k - 1));
        const double w_top = wf(i, j, k);
        return u_w * u_w + u_e * u_e
            + v_s * v_s + v_n * v_n
            + w_bottom * w_bottom + w_top * w_top;
    };
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double volume = dx * dy * (
                    terrain.metrics.z_face(i, j, k + 1)
                    - terrain.metrics.z_face(i, j, k));
                const double dry_mass =
                    storage.fields.rho(i, j, k) * volume;
                const double vapor = storage.fields.qv(i, j, k);
                const double mixture_mass = dry_mass * (1.0 + vapor);
                const double exner = std::pow(
                    storage.fields.p(i, j, k)
                        / physical_constants::reference_pressure_pa,
                    kappa);
                direct_change += dry_mass * (cv_d + vapor * cv_v)
                    * (storage.fields.theta(i, j, k)
                       - theta_initial(i, j, k)) * exner;
                direct_change += 0.25 * mixture_mass * (
                    kinetic_sum(
                        storage.fields.u, storage.fields.v,
                        storage.fields.w, i, j, k)
                    - kinetic_sum(
                        u_initial, v_initial, w_initial, i, j, k));
            }
    CHECK(energy.lateral.committed_energy_change_j
          == Approx(direct_change).epsilon(2.0e-12));
    CHECK(energy.lateral.carrier_kinetic_change_j
          == Approx(
              energy.lateral.momentum_requested_linear_work_j
              + energy.lateral.momentum_finite_step_correction_j
              + energy.lateral.momentum_storage_rounding_j));

    reset_damping_globals_to_defaults();
}
