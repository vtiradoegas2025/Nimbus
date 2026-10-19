#include "catch2/catch.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_kernel_template.hpp"
#include "core/field/field3d.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "numerics/advection/advection.hpp"
#include "numerics/advection/advection_base.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

class RecordingVerticalFluxBackend final : public ComputeBackend,
                                           public AdvectionComputeKernels
{
public:
    bool accept_dispatch = true;
    int dispatch_count = 0;

    // Deliberately identifies as Metal: routing must use capabilities, not a
    // backend-name allowlist.
    std::string name() const override { return "metal"; }
    bool initialize(std::string&) override { return true; }
    void shutdown() override {}
    AdvectionComputeKernels* advection_kernels() override { return this; }
    const AdvectionComputeKernels* advection_kernels() const override
    {
        return this;
    }
    bool supports(AdvectionKernelOp op) const override
    {
        return op == AdvectionKernelOp::VerticalFlux;
    }
    bool dispatch_vertical_flux(const VerticalFluxDispatch& dispatch) override
    {
        ++dispatch_count;
        const std::size_t cells = static_cast<std::size_t>(dispatch.nr)
            * static_cast<std::size_t>(dispatch.nth)
            * static_cast<std::size_t>(dispatch.nz);
        if (!accept_dispatch)
        {
            // A rejecting backend is allowed to dirty its candidate outputs;
            // the wrapper must not publish them to its caller.
            std::fill(dispatch.dqdt_data, dispatch.dqdt_data + cells, -919.0f);
            if (dispatch.result != nullptr)
            {
                dispatch.result->max_cfl_z = 919.0;
                dispatch.result->suggested_dt = -919.0;
            }
            return false;
        }
        std::fill(dispatch.dqdt_data, dispatch.dqdt_data + cells, 0.0f);
        if (dispatch.result != nullptr)
        {
            *dispatch.result = {};
        }
        return true;
    }
};

class ScopedAdvectionProcessState
{
public:
    ScopedAdvectionProcessState()
        : previous_(std::move(global_simulation_storage)),
          previous_template_(active_vertical_flux_template_id())
    {
        global_simulation_storage = SimulationStorageState{};
        set_active_vertical_flux_template_id("tvd_vertical_flux_v2");
    }

    ~ScopedAdvectionProcessState()
    {
        global_simulation_storage = std::move(previous_);
        set_active_vertical_flux_template_id(previous_template_);
    }

private:
    SimulationStorageState previous_;
    std::string previous_template_;
};

RecordingVerticalFluxBackend* install_recording_metal_backend(bool accepts)
{
    auto backend = std::make_unique<RecordingVerticalFluxBackend>();
    backend->accept_dispatch = accepts;
    RecordingVerticalFluxBackend* raw = backend.get();
    global_compute_backend_runtime_state.backend = std::move(backend);
    global_compute_backend_runtime_state.active_kind = ComputeBackendKind::Metal;
    global_compute_backend_runtime_state.requested_kind = ComputeBackendKind::Metal;
    global_compute_backend_runtime_state.fallback_active = false;
    global_compute_backend_config.backend = "metal";
    global_compute_backend_config.allow_fallback = false;
    return raw;
}

void setup_split_advection_state()
{
    NR = 6;
    NTH = 4;
    NZ = 6;
    dr = 1.0;
    dtheta = 1.0;
    dz = 1.0;
    dt = 0.1;
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_stagger_type = StaggerType::Collocated;
    global_grid_metrics = {};
    global_grid_metrics.dx = 1.0;
    global_grid_metrics.dy = 1.0;
    global_grid_metrics.dz.assign(static_cast<std::size_t>(NZ), 1.0);
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.25f);
    rho.resize(NR, NTH, NZ, 1.0f);
}

void setup_small_grid()
{
    NR = 16;
    NTH = 16;
    NZ = 16;
    dr = 1000.0;
    dz = 500.0;
    dt = 0.5;
    dtheta = 2.0 * 3.14159265358979323846 / NTH;
}

std::vector<float> weno5_vertical_tendency(const std::vector<float>& profile,
                                           float velocity)
{
    NR = 1;
    NTH = 1;
    NZ = static_cast<int>(profile.size());

    Field3D q(NR, NTH, NZ, 0.0f);
    Field3D w(NR, NTH, NZ, velocity);
    for (int k = 0; k < NZ; ++k)
    {
        q[0][0][k] = profile[static_cast<std::size_t>(k)];
    }

    GridMetrics grid;
    grid.dx = 1.0;
    grid.dy = 1.0;
    grid.dz.assign(static_cast<std::size_t>(NZ), 1.0);
    grid.z_int.resize(static_cast<std::size_t>(NZ + 1));
    for (int k = 0; k <= NZ; ++k)
    {
        grid.z_int[static_cast<std::size_t>(k)] = static_cast<double>(k);
    }

    auto scheme = create_advection_scheme("weno5");
    AdvectionConfig cfg;
    cfg.scheme_id = "weno5";
    cfg.positivity = false;
    scheme->initialize(cfg);

    AdvectionStateView state;
    state.q = &q;
    state.w = &w;
    state.grid = &grid;

    AdvectionTendencies tendencies;
    scheme->compute_flux_divergence(cfg, state, tendencies, nullptr);

    std::vector<float> result(static_cast<std::size_t>(NZ));
    for (int k = 0; k < NZ; ++k)
    {
        result[static_cast<std::size_t>(k)] = tendencies.dqdt_adv[0][0][k];
    }
    return result;
}

} // namespace

TEST_CASE("TVD scheme creation via factory", "[numerics][advection]")
{
    setup_small_grid();
    auto scheme = create_advection_scheme("tvd");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("WENO5 scheme creation via factory", "[numerics][advection]")
{
    setup_small_grid();
    auto scheme = create_advection_scheme("weno5");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("Advection descriptors define executable capabilities",
          "[numerics][advection][descriptor]")
{
    const auto& descriptors = get_advection_scheme_descriptors();
    REQUIRE(descriptors.size() == 3);
    for (const auto& descriptor : descriptors)
        REQUIRE(has_complete_scheme_capabilities(descriptor));

    const auto* tvd = find_scheme_descriptor(descriptors, "tvd");
    REQUIRE(tvd != nullptr);
    REQUIRE(tvd->capabilities.gpu_implementation_coverage
            == tmv::SchemeCoverage::None);
    REQUIRE(tvd->capabilities.cpu_gpu_parity_coverage
            == tmv::SchemeCoverage::None);
    REQUIRE(std::find(tvd->capabilities.backends.begin(),
                      tvd->capabilities.backends.end(), "gpu")
            != tvd->capabilities.backends.end());

    const auto* weno5 = find_scheme_descriptor(descriptors, "weno5");
    REQUIRE(weno5 != nullptr);
    REQUIRE(weno5->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(weno5->capabilities.gpu_implementation_coverage
            == tmv::SchemeCoverage::Covered);
    REQUIRE(weno5->capabilities.cpu_gpu_parity_coverage
            == tmv::SchemeCoverage::Covered);

    const auto* none = find_scheme_descriptor(descriptors, "none");
    REQUIRE(none != nullptr);
    REQUIRE(none->constructible);
    REQUIRE(none->runtime_selectable);
    REQUIRE(none->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(none->capabilities.solver_high_water_delta_fields == -9);
    REQUIRE(create_advection_scheme("disabled")->name() == "none");
    REQUIRE(create_advection_scheme("WENO")->name() == "weno5");
}

TEST_CASE("No-advection operator validates state and publishes exact zero tendency",
          "[numerics][advection][none][failure-atomic]")
{
    Field3D q(2, 2, 3, 0.25f);
    Field3D w(2, 2, 3, 1.0f);
    GridMetrics grid;
    grid.dx = 1.0;
    grid.dy = 1.0;
    grid.dz.assign(3, 1.0);
    grid.z_int = {0.0, 1.0, 2.0, 3.0};

    AdvectionConfig cfg;
    cfg.scheme_id = "none";
    AdvectionStateView state;
    state.q = &q;
    state.w = &w;
    state.grid = &grid;
    AdvectionTendencies tendencies;
    AdvectionDiagnostics diagnostics;
    auto scheme = create_advection_scheme("none");
    REQUIRE_NOTHROW(scheme->initialize(cfg));
    REQUIRE_NOTHROW(scheme->compute_flux_divergence(
        cfg, state, tendencies, &diagnostics));
    REQUIRE(tendencies.dqdt_adv.size() == q.size());
    for (std::size_t index = 0; index < q.size(); ++index)
        REQUIRE(tendencies.dqdt_adv.data()[index] == 0.0f);
    REQUIRE(std::isinf(scheme->suggest_dt(cfg, state)));

    const Field3D prior = tendencies.dqdt_adv;
    q.data()[4] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_WITH(
        scheme->compute_flux_divergence(cfg, state, tendencies, nullptr),
        "no-advection operator received non-finite state");
    REQUIRE(tendencies.dqdt_adv.data()[4] == prior.data()[4]);
}

TEST_CASE("WENO5 vertical transport is symmetric under flow reversal",
          "[numerics][advection][weno5][reconstruction]")
{
    const std::vector<float> profile = {
        0.10f, 0.35f, 1.20f, 0.45f, 2.10f, 1.65f,
        0.70f, 1.85f, 0.25f, 0.95f, 1.40f, 0.55f
    };
    std::vector<float> reversed(profile.rbegin(), profile.rend());

    const auto negative_flow = weno5_vertical_tendency(profile, -0.4f);
    const auto reversed_positive_flow = weno5_vertical_tendency(reversed, 0.4f);

    REQUIRE(negative_flow.size() == reversed_positive_flow.size());
    for (std::size_t k = 0; k < negative_flow.size(); ++k)
    {
        CAPTURE(k, negative_flow[k],
                reversed_positive_flow[negative_flow.size() - 1 - k]);
        REQUIRE(negative_flow[k] ==
                Approx(reversed_positive_flow[negative_flow.size() - 1 - k])
                    .margin(2.0e-6));
    }
}

TEST_CASE("WENO5 call configuration owns positivity and publication",
          "[numerics][advection][weno5][contract]")
{
    constexpr int nr = 5;
    constexpr int nth = 4;
    constexpr int nz = 12;
    Field3D q(nr, nth, nz, 0.0f);
    Field3D w_field(nr, nth, nz, 0.8f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            q[i][j][4] = 1.0f + 0.1f * static_cast<float>(i + j);

    GridMetrics metrics;
    metrics.dx = 1.0;
    metrics.dy = 1.0;
    metrics.dz.assign(nz, 1.0);
    AdvectionStateView state;
    state.q = &q;
    state.w = &w_field;
    state.grid = &metrics;

    auto scheme = create_advection_scheme("weno5");
    AdvectionConfig initialized;
    initialized.scheme_id = "weno5";
    initialized.positivity = false;
    scheme->initialize(initialized);

    AdvectionConfig requested = initialized;
    requested.positivity = true;
    requested.positivity_dt = 2.0;
    AdvectionTendencies one_worker;
    AdvectionDiagnostics one_diagnostics;
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    scheme->compute_flux_divergence(
        requested, state, one_worker, &one_diagnostics);
    for (std::size_t index = 0; index < q.size(); ++index)
    {
        REQUIRE(q.data()[index] + static_cast<float>(requested.positivity_dt)
                * one_worker.dqdt_adv.data()[index] >= -1.0e-7f);
    }

    AdvectionTendencies four_workers;
    AdvectionDiagnostics four_diagnostics;
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    scheme->compute_flux_divergence(
        requested, state, four_workers, &four_diagnostics);
#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
#endif
    REQUIRE(four_workers.dqdt_adv.size() == one_worker.dqdt_adv.size());
    REQUIRE(std::memcmp(four_workers.dqdt_adv.data(),
                        one_worker.dqdt_adv.data(),
                        one_worker.dqdt_adv.size() * sizeof(float)) == 0);
    REQUIRE(four_diagnostics.max_cfl_z == one_diagnostics.max_cfl_z);
    REQUIRE(four_diagnostics.suggested_dt == one_diagnostics.suggested_dt);

    four_workers.dqdt_adv.resize(2, 3, 4, 17.25f);
    const Field3D sentinel = four_workers.dqdt_adv;
    four_diagnostics.max_cfl_z = 23.5;
    four_diagnostics.suggested_dt = 47.0;
    metrics.dz[7] = 0.0;
    REQUIRE_THROWS_AS(
        scheme->compute_flux_divergence(
            requested, state, four_workers, &four_diagnostics),
        std::invalid_argument);
    REQUIRE(four_workers.dqdt_adv.size() == sentinel.size());
    REQUIRE(std::memcmp(four_workers.dqdt_adv.data(), sentinel.data(),
                        sentinel.size() * sizeof(float)) == 0);
    REQUIRE(four_diagnostics.max_cfl_z == 23.5);
    REQUIRE(four_diagnostics.suggested_dt == 47.0);
}

TEST_CASE("TVD scheme initializes with config", "[numerics][advection]")
{
    setup_small_grid();
    auto scheme = create_advection_scheme("tvd");

    AdvectionConfig cfg;
    cfg.scheme_id = "tvd";
    cfg.limiter_id = "mc";

    REQUIRE_NOTHROW(scheme->initialize(cfg));
}

TEST_CASE("TVD scheme accepts all limiter types", "[numerics][advection]")
{
    setup_small_grid();

    for (const auto& limiter : {"mc", "vanleer", "superbee", "universal"})
    {
        auto scheme = create_advection_scheme("tvd");
        AdvectionConfig cfg;
        cfg.limiter_id = limiter;
        REQUIRE_NOTHROW(scheme->initialize(cfg));
    }
}

TEST_CASE("vertical advection rejects unknown schemes metrics and nonfinite state",
          "[numerics][advection][config-truth]")
{
    AdvectionConfig cfg;
    cfg.limiter_id = "not-a-limiter";
    auto tvd = create_advection_scheme("tvd");
    REQUIRE_THROWS_AS(tvd->initialize(cfg), std::invalid_argument);

    cfg.limiter_id = "mc";
    cfg.positivity_dt = 0.0;
    REQUIRE_THROWS_AS(tvd->initialize(cfg), std::invalid_argument);

    cfg.positivity_dt = 1.0;
    tvd->initialize(cfg);
    Field3D q(1, 1, 3, 1.0f);
    Field3D w_field(1, 1, 3, 0.0f);
    GridMetrics metrics;
    metrics.dz = {1.0, 0.0, 1.0};
    AdvectionStateView state;
    state.q = &q;
    state.w = &w_field;
    state.grid = &metrics;
    AdvectionTendencies tendencies;
    REQUIRE_THROWS_AS(
        tvd->compute_flux_divergence(cfg, state, tendencies, nullptr),
        std::invalid_argument);

    metrics.dz = {1.0, 1.0, 1.0};
    q[0][0][1] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        tvd->compute_flux_divergence(cfg, state, tendencies, nullptr),
        std::invalid_argument);

    AdvectionConfig weno_cfg;
    weno_cfg.weno_order = 3;
    auto weno = create_advection_scheme("weno5");
    REQUIRE_THROWS_AS(weno->initialize(weno_cfg), std::invalid_argument);
}

TEST_CASE("Cartesian vertical schemes reject mapped terrain physical-w input",
          "[numerics][advection][terrain][guard]")
{
    Field3D q(1, 1, 3, 1.0f);
    Field3D w_field(1, 1, 3, 0.25f);
    GridMetrics metrics;
    metrics.dz = {10.0, 20.0, 30.0};
    metrics.terrain_metrics_active = true;

    AdvectionStateView state;
    state.q = &q;
    state.w = &w_field;
    state.grid = &metrics;

    AdvectionConfig cfg;
    cfg.positivity_dt = 1.0;
    AdvectionTendencies tendencies;

    for (const char* id : {"tvd", "weno5"})
    {
        auto scheme = create_advection_scheme(id);
        cfg.scheme_id = id;
        scheme->initialize(cfg);
        REQUIRE_THROWS_AS(
            scheme->compute_flux_divergence(cfg, state, tendencies, nullptr),
            std::invalid_argument);
        REQUIRE_THROWS_AS(scheme->suggest_dt(cfg, state),
                          std::invalid_argument);
    }

    REQUIRE_FALSE(dispatch_vertical_flux_template_backend(
        cfg, state, tendencies, nullptr));
}

TEST_CASE("vertical flux backend reports rejected native dispatch without CPU substitution",
          "[numerics][advection][backend][strict]")
{
    ScopedAdvectionProcessState restore;
    setup_split_advection_state();
    RecordingVerticalFluxBackend* backend =
        install_recording_metal_backend(false);

    Field3D q(NR, NTH, NZ, 1.0f);
    AdvectionStateView state;
    state.q = &q;
    state.w = &w;
    state.grid = &global_grid_metrics;
    AdvectionConfig cfg;
    cfg.positivity_dt = dt;
    AdvectionTendencies tendencies;
    tendencies.dqdt_adv.resize(2, 3, 4, 17.25f);
    const Field3D tendency_before = tendencies.dqdt_adv;
    AdvectionDiagnostics diagnostics;
    diagnostics.max_cfl_z = 23.5;
    diagnostics.suggested_dt = 47.0;

    REQUIRE_FALSE(dispatch_vertical_flux_template_backend(
        cfg, state, tendencies, &diagnostics));
    REQUIRE(backend->dispatch_count == 1);
    REQUIRE(tendencies.dqdt_adv.size_r() == tendency_before.size_r());
    REQUIRE(tendencies.dqdt_adv.size_th() == tendency_before.size_th());
    REQUIRE(tendencies.dqdt_adv.size_z() == tendency_before.size_z());
    REQUIRE(std::memcmp(
        tendencies.dqdt_adv.data(), tendency_before.data(),
        tendencies.dqdt_adv.size() * sizeof(float)) == 0);
    REQUIRE(diagnostics.max_cfl_z == 23.5);
    REQUIRE(diagnostics.suggested_dt == 47.0);
}

TEST_CASE("vertical flux template retains direct CPU execution without an accelerator",
          "[numerics][advection][backend][cpu]")
{
    ScopedAdvectionProcessState restore;
    setup_split_advection_state();

    Field3D q(NR, NTH, NZ, 1.0f);
    AdvectionStateView state;
    state.q = &q;
    state.w = &w;
    state.grid = &global_grid_metrics;
    AdvectionConfig cfg;
    cfg.positivity_dt = dt;
    AdvectionTendencies tendencies;

    REQUIRE(dispatch_vertical_flux_template_backend(
        cfg, state, tendencies, nullptr));
    REQUIRE(tendencies.dqdt_adv.size_r() == NR);
    REQUIRE(tendencies.dqdt_adv.size_th() == NTH);
    REQUIRE(tendencies.dqdt_adv.size_z() == NZ);
}

TEST_CASE("vertical flux CPU templates are bitwise deterministic",
          "[numerics][advection][backend][cpu][bitwise]")
{
    ScopedAdvectionProcessState restore;
    setup_split_advection_state();

    Field3D q(NR, NTH, NZ, 0.0f);
    for (std::size_t index = 0; index < q.size(); ++index)
    {
        q.data()[index] = static_cast<float>(
            0.125 + 0.03125 * static_cast<double>((index * 7U) % 19U));
        w.data()[index] = static_cast<float>(
            -0.45 + 0.075 * static_cast<double>((index * 5U) % 13U));
    }
    global_grid_metrics.dz = {0.75, 1.0, 1.5, 0.875, 1.25, 2.0};

    AdvectionStateView state;
    state.q = &q;
    state.w = &w;
    state.grid = &global_grid_metrics;
    AdvectionConfig cfg;
    cfg.limiter_id = "superbee";
    cfg.positivity = true;
    cfg.positivity_dt = 0.2;

    for (const char* template_id : {
             "tvd_vertical_flux_v1", "tvd_vertical_flux_v2"})
    {
        CAPTURE(template_id);
        set_active_vertical_flux_template_id(template_id);
        AdvectionTendencies first;
        AdvectionTendencies second;

        REQUIRE(dispatch_vertical_flux_template_backend(
            cfg, state, first, nullptr));
        REQUIRE(dispatch_vertical_flux_template_backend(
            cfg, state, second, nullptr));
        REQUIRE(first.dqdt_adv.size() == second.dqdt_adv.size());
        REQUIRE(std::memcmp(
            first.dqdt_adv.data(), second.dqdt_adv.data(),
            first.dqdt_adv.size() * sizeof(float)) == 0);
    }
}

TEST_CASE("Metal vertical flux capability routes natively and rejects failure in strict mode",
          "[numerics][advection][backend][metal][strict]")
{
    SECTION("a Metal-capable backend is selected without name gating")
    {
        ScopedAdvectionProcessState restore;
        setup_split_advection_state();
        RecordingVerticalFluxBackend* backend =
            install_recording_metal_backend(true);
        AdvectionRuntimeState runtime;
        AdvectionConfig cfg;
        cfg.scheme_id = "tvd";
        runtime.config = cfg;
        runtime.scheme = create_advection_scheme("tvd");
        runtime.scheme->initialize(runtime.config);
        Field3D scalar(NR, NTH, NZ, 1.0f);

        REQUIRE_NOTHROW(advect_scalar_3d(
            scalar, dt, 0.0, false, nullptr, &runtime));
        REQUIRE(backend->dispatch_count == 1);
    }

    SECTION("a rejected Metal dispatch cannot be hidden by CPU fallback")
    {
        ScopedAdvectionProcessState restore;
        setup_split_advection_state();
        RecordingVerticalFluxBackend* backend =
            install_recording_metal_backend(false);
        global_validation_policy.mode = tmv::GuardMode::Strict;
        AdvectionRuntimeState runtime;
        AdvectionConfig cfg;
        cfg.scheme_id = "tvd";
        runtime.config = cfg;
        runtime.scheme = create_advection_scheme("tvd");
        runtime.scheme->initialize(runtime.config);
        Field3D scalar(NR, NTH, NZ, 1.0f);

        REQUIRE_THROWS_AS(
            advect_scalar_3d(scalar, dt, 0.0, false, nullptr, &runtime),
            std::runtime_error);
        REQUIRE(backend->dispatch_count == 1);
    }
}
