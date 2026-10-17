/**
 * @file test_cpu_backend.cpp
 * @brief Backend-neutral CPU capability, determinism, and runtime tests.
 */

#include "catch2/catch.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_backend_factory.hpp"
#include "dynamics/acoustic_energy_report.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

#ifdef _OPENMP
class ScopedOpenmpSettings
{
public:
    ScopedOpenmpSettings()
        : dynamic_(omp_get_dynamic()), threads_(omp_get_max_threads())
    {
        omp_set_dynamic(0);
    }

    ~ScopedOpenmpSettings()
    {
        omp_set_num_threads(threads_);
        omp_set_dynamic(dynamic_);
    }

    void set_threads(int threads) { omp_set_num_threads(threads); }

private:
    int dynamic_ = 0;
    int threads_ = 1;
};
#endif

std::array<double, 24> acoustic_report_values(
    const CartesianCGridAcousticEnergyReportV1& report)
{
    return {
        report.continuity_eos_dry_internal_change_j,
        report.continuity_eos_water_sensible_change_j,
        report.continuity_eos_phase_reference_change_j,
        report.continuity_carrier_kinetic_change_j,
        report.continuity_gravitational_change_j,
        report.continuity_subgrid_tke_change_j,
        report.continuity_carrier_requested_change_j,
        report.continuity_carrier_outward_boundary_transport_j,
        report.continuity_carrier_interior_conversion_j,
        report.continuity_carrier_storage_rounding_j,
        report.slow_momentum_advection_requested_linear_work_j,
        report.kinetic_transport_interior_conversion_j,
        report.kinetic_transport_closure_residual_j,
        report.pressure_gradient_requested_linear_work_j,
        report.pressure_gradient_finite_step_correction_j,
        report.pressure_gradient_storage_rounding_j,
        report.pressure_gradient_committed_work_j,
        report.boundary_committed_work_j,
        report.boundary_radiative_requested_linear_work_j,
        report.boundary_radiative_finite_step_correction_j,
        report.boundary_radiative_storage_rounding_j,
        report.boundary_radiative_committed_work_j,
        report.boundary_constraint_history_committed_work_j,
        report.committed_inventory_change_j,
    };
}

bool bitwise_equal(
    const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    return lhs.size() == rhs.size()
        && (lhs.empty()
            || std::memcmp(lhs.data(), rhs.data(),
                           lhs.size() * sizeof(float)) == 0);
}

} // namespace

TEST_CASE("Active CPU backend exposes fused host kernels without accelerator preparation",
          "[compute][backend][cpu-fallback-tax]")
{
    const auto saved = global_compute_backend_config;
    global_compute_backend_config.backend = "cpu";
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(error));
    REQUIRE(active_compute_backend_supports(
        AdvectionKernelOp::ConservativeCartesianCgridTransport));
    REQUIRE(active_compute_backend_supports_supplied_cartesian_cgrid_mass_flux());
    REQUIRE(active_compute_backend_supports(
        DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid));
    REQUIRE_FALSE(active_compute_backend_supports(
        DynamicsKernelOp::CartesianCgridSlowTendencies));
    REQUIRE_FALSE(active_compute_backend_supports(
        DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid));
    REQUIRE_FALSE(active_compute_backend_supports(
        DynamicsKernelOp::StateRelaxationCartesian));
    REQUIRE_FALSE(active_compute_backend_supports(
        TurbulenceKernelOp::Smagorinsky));
    const ComputeBackend* cpu_backend = active_compute_backend();
    REQUIRE(cpu_backend != nullptr);
    const ComputeMemoryProperties memory = cpu_backend->memory_properties();
    REQUIRE_FALSE(memory.accelerator);
    REQUIRE(memory.device_local_heap_bytes == 0);
    REQUIRE(memory.max_storage_buffer_bytes == 0);
    shutdown_compute_backend_runtime();
    global_compute_backend_config = saved;
}

TEST_CASE("CPU fused acoustic is bitwise exact across diagnostics and OpenMP threads",
          "[compute][backend][cpu][determinism][science]")
{
    constexpr int nx = 9;
    constexpr int ny = 7;
    constexpr int nz = 5;
    constexpr std::size_t count =
        static_cast<std::size_t>(nx) * ny * nz;

    struct AcousticState
    {
        std::vector<float> u;
        std::vector<float> v;
        std::vector<float> w;
        std::vector<float> rho;
        std::vector<float> p;
        std::vector<float> drho;
        CartesianCGridAcousticEnergyReportV1 report;
        std::uint64_t dispatches = 0;
    };

    AcousticState initial{
        std::vector<float>(count),
        std::vector<float>(count),
        std::vector<float>(count),
        std::vector<float>(count),
        std::vector<float>(count),
        std::vector<float>(count),
        {},
    };
    std::vector<float> theta(count);
    std::vector<float> qv(count);
    std::vector<float> qc(count);
    std::vector<float> qr(count);
    std::vector<float> qi(count);
    std::vector<float> qs(count);
    std::vector<float> qg(count);
    std::vector<float> qh(count);
    std::vector<float> tke(count);
    std::vector<double> rho0(nz);
    std::vector<double> p0(nz);
    std::vector<float> rho0_float(nz);
    std::vector<float> p0_float(nz);
    std::vector<float> dp0_dz(nz);

    for (int k = 0; k < nz; ++k)
    {
        rho0[k] = 1.16 - 0.025 * k;
        p0[k] = 100500.0 - 1050.0 * k;
        rho0_float[k] = static_cast<float>(rho0[k]);
        p0_float[k] = static_cast<float>(p0[k]);
        dp0_dz[k] = k + 1 < nz
            ? static_cast<float>((p0[k + 1] - p0[k]) / 125.0)
            : dp0_dz[k - 1];
    }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t c =
                    (static_cast<std::size_t>(i) * ny + j) * nz + k;
                const float shape =
                    static_cast<float>(3 * i - 2 * j + 5 * k);
                initial.u[c] = 4.0f + 0.013f * shape;
                initial.v[c] = -1.5f + 0.009f * shape;
                initial.w[c] = 0.015f * static_cast<float>(i - j)
                    + 0.006f * static_cast<float>(k);
                initial.rho[c] = rho0_float[k] + 1.0e-4f * shape;
                initial.p[c] = p0_float[k] + 0.37f * shape;
                theta[c] = 299.0f + 0.08f * shape;
                qv[c] = 0.012f + 1.0e-6f * static_cast<float>(c % 13);
                qc[c] = 2.0e-4f + 1.0e-7f * static_cast<float>(c % 7);
                qr[c] = 8.0e-5f + 1.0e-7f * static_cast<float>(c % 5);
                qi[c] = 5.0e-5f + 1.0e-8f * static_cast<float>(c % 11);
                qs[c] = 3.0e-5f + 1.0e-8f * static_cast<float>(c % 3);
                qg[c] = 2.0e-5f + 1.0e-8f * static_cast<float>(c % 17);
                qh[c] = 1.0e-5f + 1.0e-8f * static_cast<float>(c % 19);
                tke[c] = 0.25f + 0.001f * static_cast<float>(c % 23);
            }

    auto run = [&](int threads, bool attribute_energy)
    {
        AcousticState state = initial;
        auto backend = create_cpu_compute_backend();
        std::string error;
        REQUIRE(backend->initialize(error));
        auto* dynamics = backend->dynamics_kernels();
        REQUIRE(dynamics != nullptr);

#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
        AcousticSubstepFusedCartesianCgridDispatch dispatch{};
        dispatch.u = state.u.data();
        dispatch.v = state.v.data();
        dispatch.w = state.w.data();
        dispatch.rho = state.rho.data();
        dispatch.p = state.p.data();
        dispatch.theta_data = theta.data();
        dispatch.rho0_base_data = rho0_float.data();
        dispatch.p0_base_data = p0_float.data();
        dispatch.dp0_dz_data = dp0_dz.data();
        dispatch.base_len = nz;
        dispatch.nx = nx;
        dispatch.ny = ny;
        dispatch.nz = nz;
        dispatch.dx = 175.0f;
        dispatch.dy = 150.0f;
        dispatch.dz = 125.0f;
        dispatch.gamma_val = 1.4f;
        dispatch.dt_small = 0.04f;
        dispatch.rho_min = 0.1f;
        dispatch.rho_max = 2.0f;
        dispatch.p_min = 1000.0f;
        dispatch.p_max = 120000.0f;
        dispatch.p_ref_fallback = 100000.0f;
        dispatch.wind_clamp_h = 200.0f;
        dispatch.wind_clamp_v = 100.0f;
        dispatch.rho0_base_host = rho0.data();
        dispatch.p0_base_host = p0.data();
        dispatch.dx_host = 175.0;
        dispatch.dy_host = 150.0;
        dispatch.dz_host = 125.0;
        dispatch.dt_small_host = 0.04;
        dispatch.gamma_host = 1.4;
        dispatch.drho_scratch = state.drho.data();
        dispatch.qv_host = qv.data();
        dispatch.qc_host = qc.data();
        dispatch.qr_host = qr.data();
        dispatch.qi_host = qi.data();
        dispatch.qs_host = qs.data();
        dispatch.qg_host = qg.data();
        dispatch.qh_host = qh.data();
        dispatch.tke_host = tke.data();
        dispatch.energy_report_host = attribute_energy
            ? &state.report : nullptr;

        for (int substep = 0; substep < 3; ++substep)
            REQUIRE(dynamics
                ->dispatch_acoustic_substep_fused_cartesian_cgrid(dispatch));
        state.dispatches = backend->successful_kernel_dispatch_count(
            nimbus::compute::KernelId::AcousticPressureCartesianCgrid);
        backend->shutdown();
        return state;
    };

#ifdef _OPENMP
    ScopedOpenmpSettings openmp;
#endif
    const AcousticState one_thread = run(1, true);
    const AcousticState four_threads = run(4, true);
    const AcousticState unattributed = run(4, false);

    REQUIRE(one_thread.u == four_threads.u);
    REQUIRE(one_thread.v == four_threads.v);
    REQUIRE(one_thread.w == four_threads.w);
    REQUIRE(one_thread.rho == four_threads.rho);
    REQUIRE(one_thread.p == four_threads.p);
    REQUIRE(one_thread.drho == four_threads.drho);
    REQUIRE(acoustic_report_values(one_thread.report)
            == acoustic_report_values(four_threads.report));
    REQUIRE(one_thread.report.kinetic_transport_complete
            == four_threads.report.kinetic_transport_complete);
    REQUIRE(one_thread.report.boundary_application_count
            == four_threads.report.boundary_application_count);
    REQUIRE(one_thread.report.boundary_complete
            == four_threads.report.boundary_complete);
    REQUIRE(one_thread.report.substep_count == 3);
    REQUIRE(one_thread.report.substep_count
            == four_threads.report.substep_count);
    REQUIRE(one_thread.report.complete);
    REQUIRE(one_thread.report.complete == four_threads.report.complete);
    REQUIRE(one_thread.dispatches == 3);
    REQUIRE(one_thread.dispatches == four_threads.dispatches);

    REQUIRE(bitwise_equal(unattributed.u, four_threads.u));
    REQUIRE(bitwise_equal(unattributed.v, four_threads.v));
    REQUIRE(bitwise_equal(unattributed.w, four_threads.w));
    REQUIRE(bitwise_equal(unattributed.rho, four_threads.rho));
    REQUIRE(bitwise_equal(unattributed.p, four_threads.p));
    REQUIRE(bitwise_equal(unattributed.drho, four_threads.drho));
    REQUIRE(unattributed.dispatches == four_threads.dispatches);
}

TEST_CASE("CPU fused acoustic rejects invalid clamp contracts without mutation",
          "[compute][backend][cpu][acoustic][validation][sentinel]")
{
    constexpr int nx = 3;
    constexpr int ny = 3;
    constexpr int nz = 2;
    constexpr std::size_t count =
        static_cast<std::size_t>(nx) * ny * nz;
    using Dispatch = AcousticSubstepFusedCartesianCgridDispatch;
    struct InvalidCase
    {
        const char* label;
        void (*corrupt)(Dispatch&);
    };
    const InvalidCase invalid_cases[] = {
        {"non-finite rho lower bound", [](Dispatch& d)
            { d.rho_min = std::numeric_limits<float>::quiet_NaN(); }},
        {"non-finite rho upper bound", [](Dispatch& d)
            { d.rho_max = std::numeric_limits<float>::infinity(); }},
        {"inverted rho bounds", [](Dispatch& d)
            { d.rho_min = d.rho_max + 1.0f; }},
        {"non-positive rho lower bound", [](Dispatch& d)
            { d.rho_min = 0.0f; }},
        {"non-finite pressure lower bound", [](Dispatch& d)
            { d.p_min = std::numeric_limits<float>::quiet_NaN(); }},
        {"non-finite pressure upper bound", [](Dispatch& d)
            { d.p_max = std::numeric_limits<float>::infinity(); }},
        {"inverted pressure bounds", [](Dispatch& d)
            { d.p_min = d.p_max + 1.0f; }},
        {"non-positive pressure lower bound", [](Dispatch& d)
            { d.p_min = 0.0f; }},
        {"non-finite pressure fallback", [](Dispatch& d)
            { d.p_ref_fallback = std::numeric_limits<float>::infinity(); }},
        {"non-positive pressure fallback", [](Dispatch& d)
            { d.p_ref_fallback = 0.0f; }},
        {"non-finite host dx", [](Dispatch& d)
            { d.dx_host = std::numeric_limits<double>::infinity(); }},
        {"non-finite host dy", [](Dispatch& d)
            { d.dy_host = std::numeric_limits<double>::quiet_NaN(); }},
        {"non-finite host dz", [](Dispatch& d)
            { d.dz_host = std::numeric_limits<double>::infinity(); }},
        {"non-finite host timestep", [](Dispatch& d)
            { d.dt_small_host = std::numeric_limits<double>::infinity(); }},
        {"non-finite host gamma", [](Dispatch& d)
            { d.gamma_host = std::numeric_limits<double>::infinity(); }},
        {"negative horizontal wind clamp", [](Dispatch& d)
            { d.wind_clamp_h = -1.0f; }},
        {"non-finite vertical wind clamp", [](Dispatch& d)
            { d.wind_clamp_v = std::numeric_limits<float>::infinity(); }},
    };

    for (const InvalidCase& invalid_case : invalid_cases)
    {
        INFO(invalid_case.label);
        std::vector<float> u(count, 1.0f);
        std::vector<float> v(count, -2.0f);
        std::vector<float> w(count, 0.25f);
        std::vector<float> rho(count, 1.1f);
        std::vector<float> p(count, 95000.0f);
        std::vector<float> theta(count, 300.0f);
        std::vector<float> drho(count, 37.0f);
        std::vector<float> water(count, 0.0f);
        std::vector<float> tke(count, 0.1f);
        const std::array<double, nz> rho0{{1.1, 1.0}};
        const std::array<double, nz> p0{{95000.0, 94000.0}};

        CartesianCGridAcousticEnergyReportV1 report{};
        report.continuity_eos_dry_internal_change_j = 11.0;
        report.pressure_gradient_committed_work_j = -7.0;
        report.boundary_application_count = 1;
        report.boundary_complete = true;
        report.committed_inventory_change_j = 19.0;
        report.substep_count = 2;
        report.complete = true;

        Dispatch dispatch{};
        dispatch.u = u.data();
        dispatch.v = v.data();
        dispatch.w = w.data();
        dispatch.rho = rho.data();
        dispatch.p = p.data();
        dispatch.theta_data = theta.data();
        dispatch.base_len = nz;
        dispatch.nx = nx;
        dispatch.ny = ny;
        dispatch.nz = nz;
        dispatch.rho_min = 0.1f;
        dispatch.rho_max = 2.0f;
        dispatch.p_min = 1000.0f;
        dispatch.p_max = 120000.0f;
        dispatch.p_ref_fallback = 100000.0f;
        dispatch.wind_clamp_h = 200.0f;
        dispatch.wind_clamp_v = 100.0f;
        dispatch.rho0_base_host = rho0.data();
        dispatch.p0_base_host = p0.data();
        dispatch.dx_host = 175.0;
        dispatch.dy_host = 150.0;
        dispatch.dz_host = 125.0;
        dispatch.dt_small_host = 0.04;
        dispatch.gamma_host = 1.4;
        dispatch.drho_scratch = drho.data();
        dispatch.qv_host = water.data();
        dispatch.qc_host = water.data();
        dispatch.qr_host = water.data();
        dispatch.qi_host = water.data();
        dispatch.qs_host = water.data();
        dispatch.qg_host = water.data();
        dispatch.qh_host = water.data();
        dispatch.tke_host = tke.data();
        dispatch.energy_report_host = &report;

        const auto u_before = u;
        const auto v_before = v;
        const auto w_before = w;
        const auto rho_before = rho;
        const auto p_before = p;
        const auto drho_before = drho;
        const auto report_before = report;

        auto backend = create_cpu_compute_backend();
        std::string error;
        REQUIRE(backend->initialize(error));
        auto* dynamics = backend->dynamics_kernels();
        REQUIRE(dynamics != nullptr);
        const auto dispatches_before =
            backend->successful_kernel_dispatch_count(
                nimbus::compute::KernelId::AcousticPressureCartesianCgrid);

        invalid_case.corrupt(dispatch);
        REQUIRE_FALSE(
            dynamics->dispatch_acoustic_substep_fused_cartesian_cgrid(
                dispatch));
        REQUIRE(u == u_before);
        REQUIRE(v == v_before);
        REQUIRE(w == w_before);
        REQUIRE(rho == rho_before);
        REQUIRE(p == p_before);
        REQUIRE(drho == drho_before);
        REQUIRE(acoustic_report_values(report)
                == acoustic_report_values(report_before));
        REQUIRE(report.kinetic_transport_complete
                == report_before.kinetic_transport_complete);
        REQUIRE(report.boundary_application_count
                == report_before.boundary_application_count);
        REQUIRE(report.boundary_complete == report_before.boundary_complete);
        REQUIRE(report.substep_count == report_before.substep_count);
        REQUIRE(report.complete == report_before.complete);
        REQUIRE(backend->successful_kernel_dispatch_count(
                    nimbus::compute::KernelId::
                        AcousticPressureCartesianCgrid)
                == dispatches_before);
        backend->shutdown();
    }
}

TEST_CASE("Caller-owned compute runtimes isolate lifecycle and snapshot safely",
          "[compute][backend][runtime][ownership]")
{
    ComputeBackendRuntimeState first;
    ComputeBackendRuntimeState second;
    first.config.backend = "cpu";
    second.config.backend = "cpu";

    std::string first_error;
    std::string second_error;
    REQUIRE(initialize_compute_backend_runtime(
        first, CoordinateSystem::Cartesian, first_error));
    REQUIRE(initialize_compute_backend_runtime(
        second, CoordinateSystem::Cartesian, second_error));
    REQUIRE(active_compute_backend(first) != nullptr);
    REQUIRE(active_compute_backend(second) != nullptr);
    REQUIRE(active_compute_backend(first) != active_compute_backend(second));
    REQUIRE(active_compute_backend_kind(first) == ComputeBackendKind::Cpu);
    REQUIRE(std::string(requested_compute_backend_name(first)) == "cpu");
    REQUIRE_FALSE(compute_backend_fallback_active(first));

    ComputeBackendRuntimeState snapshot(first);
    REQUIRE(snapshot.config.backend == "cpu");
    REQUIRE(snapshot.backend == nullptr);
    REQUIRE(snapshot.active_kind == ComputeBackendKind::Cpu);
    snapshot.config.device_index = 7;
    REQUIRE(first.config.device_index == -1);

    shutdown_compute_backend_runtime(first);
    REQUIRE(active_compute_backend(first) == nullptr);
    REQUIRE(active_compute_backend(second) != nullptr);
    shutdown_compute_backend_runtime(second);
}
