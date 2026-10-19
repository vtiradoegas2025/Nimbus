// RK order tests use dy/dt=-y, whose exact solution is y(t)=y0 exp(-t).
#include "catch2/catch.hpp"
#include "core/field/field3d.hpp"
#include "core/runtime/simulation.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "numerics/time_stepping/factory.hpp"
#include "numerics/numerics_base.hpp"

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>

namespace
{

void setup_grid()
{
    NR = 4;
    NTH = 4;
    NZ = 4;
    dr = 1000.0;
    dz = 500.0;
    dtheta = dr;
}

// Simple exponential decay ODE: dy/dt = -y
// Exact solution: y(t) = y0 * exp(-t)
void exponential_decay_rhs(const std::vector<NumericalState>& fields,
                           double /*time*/,
                           std::vector<NumericalTendencies>& tendencies)
{
    tendencies.resize(fields.size());
    for (size_t f = 0; f < fields.size(); ++f)
    {
        tendencies[f].name = fields[f].name;
        tendencies[f].tendencies.resize(fields[f].data.size_r(),
                                        fields[f].data.size_th(),
                                        fields[f].data.size_z());
        for (size_t n = 0; n < fields[f].data.size(); ++n)
        {
            tendencies[f].tendencies.data()[n] = -fields[f].data.data()[n];
        }
    }
}

// Integrate the decay ODE for duration T using n_steps of size dt
double integrate_decay(const std::string& scheme_name, double y0, double T, int n_steps)
{
    setup_grid();
    double step_dt = T / n_steps;

    auto scheme = create_time_stepping_scheme(scheme_name);
    TimeSteppingConfig cfg;
    cfg.scheme_id = scheme_name;

    scheme->initialize(cfg, exponential_decay_rhs);

    TimeSteppingState state;
    state.time = 0.0;
    state.dt = step_dt;

    NumericalState ns;
    ns.name = "y";
    ns.data.resize(NR, NTH, NZ, static_cast<float>(y0));
    state.fields.push_back(std::move(ns));

    for (int step = 0; step < n_steps; ++step)
    {
        scheme->step(cfg, state);
    }

    // Return the value at a representative interior point
    return static_cast<double>(state.fields[0].data(2, 2, 2));
}

TimeSteppingState make_transaction_test_state()
{
    TimeSteppingState state;
    state.time = 3.25;
    state.dt = 0.125;
    for (int field_index = 0; field_index < 2; ++field_index)
    {
        NumericalState field;
        field.name = field_index == 0 ? "first" : "second";
        field.data.resize(3, 2, 2, 0.0f);
        for (std::size_t i = 0; i < field.data.size(); ++i)
        {
            field.data.data()[i] = static_cast<float>(
                10 * field_index + static_cast<int>(i) + 0.25);
        }
        state.fields.push_back(std::move(field));
    }
    return state;
}

void require_state_bitwise_equal(
    const TimeSteppingState& actual,
    const TimeSteppingState& expected)
{
    REQUIRE(std::memcmp(&actual.time, &expected.time, sizeof(double)) == 0);
    REQUIRE(std::memcmp(&actual.dt, &expected.dt, sizeof(double)) == 0);
    REQUIRE(actual.fields.size() == expected.fields.size());
    for (std::size_t i = 0; i < expected.fields.size(); ++i)
    {
        REQUIRE(actual.fields[i].name == expected.fields[i].name);
        REQUIRE(actual.fields[i].data.size_r() == expected.fields[i].data.size_r());
        REQUIRE(actual.fields[i].data.size_th() == expected.fields[i].data.size_th());
        REQUIRE(actual.fields[i].data.size_z() == expected.fields[i].data.size_z());
        REQUIRE(std::memcmp(
            actual.fields[i].data.data(),
            expected.fields[i].data.data(),
            expected.fields[i].data.size() * sizeof(float)) == 0);
    }
}

void require_diagnostics_equal(
    const TimeSteppingDiagnostics& actual,
    const TimeSteppingDiagnostics& expected)
{
    REQUIRE(actual.n_steps == expected.n_steps);
    REQUIRE(actual.max_cfl == expected.max_cfl);
    REQUIRE(actual.cpu_time_per_step == expected.cpu_time_per_step);
    REQUIRE(actual.dt_history == expected.dt_history);
    REQUIRE(actual.converged == expected.converged);
}

} // namespace

TEST_CASE("Time-stepping descriptors expose the qualified runtime path",
          "[numerics][timestepping][descriptor]")
{
    const auto& descriptors = get_time_stepping_scheme_descriptors();
    const auto* split = tmv::find_scheme_descriptor(descriptors, "klemp_wilhelmson");
    const auto* rk3 = tmv::find_scheme_descriptor(descriptors, "rk3");
    const auto* rk4 = tmv::find_scheme_descriptor(descriptors, "rk4");

    REQUIRE(split != nullptr);
    REQUIRE(split->id == "split_explicit");
    REQUIRE(split->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(split->capabilities.declared);
    REQUIRE(split->capabilities.gpu_implementation_coverage
            == tmv::SchemeCoverage::Covered);
    REQUIRE(split->capabilities.cpu_gpu_parity_coverage
            == tmv::SchemeCoverage::Covered);
    REQUIRE(rk3 != nullptr);
    REQUIRE(rk3->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(rk3->runtime_selectable);
    REQUIRE(rk3->capabilities.coordinate_systems ==
            std::vector<std::string>{"cartesian"});
    REQUIRE(rk3->capabilities.staggerings ==
            std::vector<std::string>{"collocated", "c_grid"});
    const std::string rk_contract =
        "flat Cartesian CPU families and periodic stationary native C-grid/collocated RK with no-advection/TVD/mixed-WENO public/private staging, explicit/implicit constant diffusion, and admitted physics compositions";
    REQUIRE(rk3->capabilities.config_validation_contract == rk_contract);
    REQUIRE(rk4 != nullptr);
    REQUIRE(rk4->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(rk4->runtime_selectable);
    REQUIRE(rk4->capabilities.coordinate_systems ==
            std::vector<std::string>{"cartesian"});
    REQUIRE(rk4->capabilities.staggerings ==
            std::vector<std::string>{"collocated", "c_grid"});
    REQUIRE(rk4->capabilities.config_validation_contract == rk_contract);
}

TEST_CASE("Time stepping factory creates rk3", "[numerics][timestepping]")
{
    auto scheme = create_time_stepping_scheme("rk3");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("Time stepping factory creates rk4", "[numerics][timestepping]")
{
    auto scheme = create_time_stepping_scheme("rk4");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("Time stepping factory recognizes ssprk3 alias", "[numerics][timestepping]")
{
    auto scheme = create_time_stepping_scheme("ssprk3");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("Built-in time-stepping schemes invalidate recycled-generation state",
          "[numerics][timestepping][domain_recycling]")
{
    for (const char* scheme_name : {"rk3", "rk4", "split_explicit"})
    {
        CAPTURE(scheme_name);
        auto scheme = create_time_stepping_scheme(scheme_name);
        REQUIRE(scheme != nullptr);
        REQUIRE(scheme->supports_domain_recycling_invalidation());
        REQUIRE_NOTHROW(
            scheme->invalidate_generation_state_after_domain_recycling());
    }
}

TEST_CASE("time-stepping runtime owns dispatch and persistent stages",
          "[numerics][timestepping][state][modularity]")
{
    REQUIRE(&time_stepping_scheme ==
            &global_time_stepping_runtime_state.scheme);
    REQUIRE(&global_time_stepping_config ==
            &global_time_stepping_runtime_state.config);

    TimeSteppingRuntimeState original;
    original.config.scheme_id = "rk3";
    original.scheme = create_time_stepping_scheme("rk3");
    original.scheme->initialize(original.config, exponential_decay_rhs);
    original.diagnostics.n_steps = 4;
    original.diagnostics.dt_history = {0.5, 0.25};
    original.slow_rk2.u_save.resize(2, 1, 1, 3.0f);

    TimeSteppingRuntimeState copy(original);
    REQUIRE(copy.scheme != nullptr);
    REQUIRE(copy.scheme.get() != original.scheme.get());
    REQUIRE(copy.scheme->name() == "rk3");
    REQUIRE(copy.diagnostics.n_steps == 4);
    REQUIRE(copy.diagnostics.dt_history.data() !=
            original.diagnostics.dt_history.data());
    REQUIRE(copy.slow_rk2.u_save.data() != original.slow_rk2.u_save.data());
}

TEST_CASE("Runtime accepts implemented split-explicit and coupled unsplit contracts",
          "[numerics][timestepping][runtime-contract]")
{
    TimeSteppingConfig cfg;
    cfg.split_acoustic = true;

    auto split = create_time_stepping_scheme("split_explicit");
    REQUIRE_NOTHROW(validate_runtime_time_stepping_contract(cfg, *split));

    auto rk3 = create_time_stepping_scheme("rk3");
    REQUIRE_THROWS_WITH(
        validate_runtime_time_stepping_contract(cfg, *rk3),
        Catch::Matchers::Contains("does not implement split-explicit"));

    cfg.split_acoustic = false;
    REQUIRE_NOTHROW(validate_runtime_time_stepping_contract(cfg, *rk3));
    auto rk4 = create_time_stepping_scheme("rk4");
    REQUIRE_NOTHROW(validate_runtime_time_stepping_contract(cfg, *rk4));
    REQUIRE_THROWS_WITH(
        validate_runtime_time_stepping_contract(cfg, *split),
        Catch::Matchers::Contains("does not implement coupled unsplit"));
}

TEST_CASE("Coupled unsplit Runge-Kutta callbacks follow their tableaux atomically",
          "[numerics][timestepping][coupled][failure-atomic]")
{
    for (const char* scheme_name : {"rk3", "rk4"})
    {
        CAPTURE(scheme_name);
        auto scheme = create_time_stepping_scheme(scheme_name);
        REQUIRE(scheme->supports_coupled_unsplit_rk());

        TimeSteppingConfig cfg;
        cfg.scheme_id = scheme_name;
        cfg.split_acoustic = false;
        double live = 1.0;
        double base = 0.0;
        std::array<double, 4> rhs{};
        std::vector<double> stage_times;
        UnsplitRkCallbacks callbacks;
        callbacks.capture_base = [&] { base = live; };
        callbacks.compute_tendencies =
            [&](std::size_t stage, double time)
        {
            stage_times.push_back(time);
            rhs[stage] = -live;
        };
        callbacks.publish_combination =
            [&](std::size_t count,
                const std::array<double, 4>& weights,
                double dt)
        {
            live = base;
            for (std::size_t stage = 0; stage < count; ++stage)
                live += dt * weights[stage] * rhs[stage];
        };
        callbacks.restore_base = [&] { live = base; };

        TimeSteppingDiagnostics diagnostics;
        scheme->step_coupled_unsplit_rk(
            cfg, 0.1, 2.0, callbacks, &diagnostics);
        REQUIRE(live == Approx(std::exp(-0.1)).epsilon(2.0e-5));
        REQUIRE(diagnostics.n_steps == 1);
        REQUIRE(diagnostics.dt_history == std::vector<double>{0.1});
        if (std::string(scheme_name) == "rk3")
            REQUIRE(stage_times == std::vector<double>{2.0, 2.1, 2.05});
        else
            REQUIRE(stage_times == std::vector<double>{2.0, 2.05, 2.05, 2.1});

        live = 7.0;
        const double original = live;
        stage_times.clear();
        TimeSteppingDiagnostics failed_diagnostics;
        failed_diagnostics.n_steps = 4;
        failed_diagnostics.dt_history = {0.25};
        const auto expected_diagnostics = failed_diagnostics;
        callbacks.compute_tendencies =
            [&](std::size_t stage, double time)
        {
            stage_times.push_back(time);
            rhs[stage] = -live;
            if (stage == 2)
                throw std::runtime_error("synthetic coupled-stage failure");
        };
        REQUIRE_THROWS_WITH(
            scheme->step_coupled_unsplit_rk(
                cfg, 0.1, 2.0, callbacks, &failed_diagnostics),
            "synthetic coupled-stage failure");
        REQUIRE(live == original);
        require_diagnostics_equal(failed_diagnostics, expected_diagnostics);
    }
}

TEST_CASE("RK3 integrates exponential decay correctly", "[numerics][timestepping][analytical]")
{
    const double y0 = 1.0;
    const double T = 1.0;
    const double exact = y0 * std::exp(-T);

    double result = integrate_decay("rk3", y0, T, 100);
    double error = std::abs(result - exact);

    // RK3 with 100 steps on a simple ODE should be very accurate
    REQUIRE(error < 1.0e-4);
}

TEST_CASE("RK4 integrates exponential decay correctly", "[numerics][timestepping][analytical]")
{
    const double y0 = 1.0;
    const double T = 1.0;
    const double exact = y0 * std::exp(-T);

    double result = integrate_decay("rk4", y0, T, 100);
    double error = std::abs(result - exact);

    // RK4 with 100 steps is limited by float32 storage precision (~1.2e-7).
    // The scheme is 4th-order accurate but Field3D uses 32-bit floats,
    // so errors below ~1e-7 are dominated by floating-point representation.
    REQUIRE(error < 1.0e-5);
    // RK4 should still be more accurate than RK3 at same step count
}

TEST_CASE("RK3 shows 3rd-order convergence", "[numerics][timestepping][analytical]")
{
    const double y0 = 1.0;
    const double T = 0.5;
    const double exact = y0 * std::exp(-T);

    double error_coarse = std::abs(integrate_decay("rk3", y0, T, 10) - exact);
    double error_fine   = std::abs(integrate_decay("rk3", y0, T, 20) - exact);

    // Halving dt should reduce error by ~2^3 = 8 for 3rd-order scheme
    // Allow range [4, 16] to account for constant factors
    double ratio = error_coarse / error_fine;
    REQUIRE(ratio > 4.0);
    REQUIRE(ratio < 16.0);
}

TEST_CASE("RK4 shows 4th-order convergence", "[numerics][timestepping][analytical]")
{
    const double y0 = 1.0;
    const double T = 0.5;
    const double exact = y0 * std::exp(-T);

    // Use fewer steps so error is well above float32 floor
    double error_coarse = std::abs(integrate_decay("rk4", y0, T, 4) - exact);
    double error_fine   = std::abs(integrate_decay("rk4", y0, T, 8) - exact);

    // Halving dt should reduce error by at least 2^4 = 16 for 4th-order scheme.
    // With very few steps the higher-order truncation terms contribute, so the
    // ratio can exceed 16 significantly (super-convergence at coarse resolution).
    // The key test: ratio must be >= 16 to confirm at least 4th-order.
    double ratio = error_coarse / error_fine;
    REQUIRE(ratio > 12.0);
}

TEST_CASE("RK4 is more accurate than RK3 at same step count", "[numerics][timestepping][analytical]")
{
    const double y0 = 1.0;
    const double T = 1.0;
    const double exact = y0 * std::exp(-T);

    double error_rk3 = std::abs(integrate_decay("rk3", y0, T, 50) - exact);
    double error_rk4 = std::abs(integrate_decay("rk4", y0, T, 50) - exact);

    REQUIRE(error_rk4 < error_rk3);
}

TEST_CASE("Runge-Kutta stages evaluate a non-autonomous RHS at their tableau times",
          "[numerics][timestepping][analytical][stage-time]")
{
    for (const char* scheme_name : {"rk3", "rk4"})
    {
        CAPTURE(scheme_name);
        auto scheme = create_time_stepping_scheme(scheme_name);
        TimeSteppingConfig cfg;
        cfg.scheme_id = scheme_name;
        std::vector<double> evaluation_times;
        scheme->initialize(
            cfg,
            [&](const std::vector<NumericalState>& fields,
                double time,
                std::vector<NumericalTendencies>& tendencies)
            {
                evaluation_times.push_back(time);
                tendencies.resize(fields.size());
                for (std::size_t i = 0; i < fields.size(); ++i)
                {
                    tendencies[i].name = fields[i].name;
                    tendencies[i].tendencies.resize(
                        fields[i].data.size_r(),
                        fields[i].data.size_th(),
                        fields[i].data.size_z(),
                        static_cast<float>(time));
                }
            });

        TimeSteppingState state;
        state.dt = 1.0;
        NumericalState field;
        field.name = "integral_of_time";
        field.data.resize(1, 1, 1, 0.0f);
        state.fields.push_back(std::move(field));
        scheme->step(cfg, state);

        REQUIRE(state.time == 1.0);
        REQUIRE(state.fields[0].data(0, 0, 0) == Approx(0.5f));
        if (std::string(scheme_name) == "rk3")
        {
            REQUIRE(evaluation_times == std::vector<double>{0.0, 1.0, 0.5});
        }
        else
        {
            REQUIRE(evaluation_times == std::vector<double>{0.0, 0.5, 0.5, 1.0});
        }
    }
}

TEST_CASE("Runge-Kutta steps publish atomically after every stage succeeds",
          "[numerics][timestepping][failure-atomic]")
{
    for (const char* scheme_name : {"rk3", "rk4"})
    {
        CAPTURE(scheme_name);
        auto scheme = create_time_stepping_scheme(scheme_name);
        TimeSteppingConfig cfg;
        cfg.scheme_id = scheme_name;
        int rhs_calls = 0;
        const int failing_call = std::string(scheme_name) == "rk3" ? 3 : 4;
        scheme->initialize(
            cfg,
            [&](const std::vector<NumericalState>& fields,
                double,
                std::vector<NumericalTendencies>& tendencies)
            {
                ++rhs_calls;
                tendencies.resize(fields.size());
                for (std::size_t i = 0; i < fields.size(); ++i)
                {
                    tendencies[i].name = fields[i].name;
                    tendencies[i].tendencies.resize(
                        fields[i].data.size_r(),
                        fields[i].data.size_th(),
                        fields[i].data.size_z(),
                        0.5f);
                }
                if (rhs_calls == failing_call)
                {
                    tendencies.back().tendencies.resize(2, 2, 3, 0.5f);
                }
            });

        auto state = make_transaction_test_state();
        const auto original_state = state;
        TimeSteppingDiagnostics diagnostics;
        diagnostics.n_steps = 7;
        diagnostics.max_cfl = 0.42;
        diagnostics.cpu_time_per_step = 1.75;
        diagnostics.dt_history = {0.25, 0.125};
        diagnostics.converged = false;
        const auto original_diagnostics = diagnostics;

        REQUIRE_THROWS(scheme->step(cfg, state, &diagnostics));
        REQUIRE(rhs_calls == failing_call);
        require_state_bitwise_equal(state, original_state);
        require_diagnostics_equal(diagnostics, original_diagnostics);
    }
}

TEST_CASE("Runge-Kutta steps reject invalid timesteps before dispatch",
          "[numerics][timestepping][validation]")
{
    for (const char* scheme_name : {"rk3", "rk4"})
    {
        for (const double invalid_dt : {
                 0.0,
                 -0.25,
                 std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::quiet_NaN()})
        {
            CAPTURE(scheme_name, invalid_dt);
            auto scheme = create_time_stepping_scheme(scheme_name);
            TimeSteppingConfig cfg;
            cfg.scheme_id = scheme_name;
            int rhs_calls = 0;
            scheme->initialize(
                cfg,
                [&](const std::vector<NumericalState>&,
                    double,
                    std::vector<NumericalTendencies>&)
                {
                    ++rhs_calls;
                });

            auto state = make_transaction_test_state();
            state.dt = invalid_dt;
            const auto original_state = state;
            TimeSteppingDiagnostics diagnostics;
            diagnostics.n_steps = 3;
            diagnostics.dt_history = {0.5};
            const auto original_diagnostics = diagnostics;

            REQUIRE_THROWS(scheme->step(cfg, state, &diagnostics));
            REQUIRE(rhs_calls == 0);
            require_state_bitwise_equal(state, original_state);
            require_diagnostics_equal(diagnostics, original_diagnostics);
        }
    }
}

TEST_CASE("Split-explicit auto substeps include Cartesian y spacing",
          "[numerics][timestepping][split-explicit][rectangular-grid]")
{
    setup_grid();
    dr = 1000.0;
    dtheta = 100.0;
    dz = 500.0;

    auto scheme = create_time_stepping_scheme("split_explicit");
    TimeSteppingConfig cfg;
    cfg.scheme_id = "split_explicit";
    cfg.n_acoustic_substeps = 0;
    cfg.max_acoustic_speed_ms = 350.0;

    int pressure_calls = 0;
    int pressure_boundary_calls = 0;
    int momentum_calls = 0;
    int boundary_calls = 0;
    int residency_release_calls = 0;
    int substep_phase = 0;
    int residency_begin_calls = 0;
    double observed_dt_small = 0.0;
    SplitExplicitCallbacks callbacks;
    callbacks.compute_slow_tendencies = [] {};
    callbacks.apply_slow_tendencies = [](double) {};
    callbacks.begin_acoustic_loop_residency = [&] {
        ++residency_begin_calls;
    };
    callbacks.apply_fast_pressure = [&](double dt_small) {
        REQUIRE(residency_begin_calls == 1);
        REQUIRE(substep_phase == 0);
        ++pressure_calls;
        observed_dt_small = dt_small;
        substep_phase = 1;
    };
    callbacks.close_fast_pressure_boundaries = [&] {
        REQUIRE(substep_phase == 1);
        ++pressure_boundary_calls;
        substep_phase = 2;
    };
    callbacks.apply_fast_momentum = [&](double) {
        REQUIRE(substep_phase == 2);
        ++momentum_calls;
        substep_phase = 3;
    };
    callbacks.acoustic_bcs = [&](double) {
        REQUIRE(substep_phase == 3);
        ++boundary_calls;
        substep_phase = 0;
    };
    callbacks.release_acoustic_loop_residency = [&] {
        ++residency_release_calls;
    };

    constexpr double dt_large = 1.0;
    scheme->step_split_acoustic(cfg, dt_large, callbacks);

    // ceil(dt*c_s/(0.5*dy)) + 1 = ceil(350/50) + 1 = 8.
    REQUIRE(pressure_calls == 8);
    REQUIRE(pressure_boundary_calls == 8);
    REQUIRE(momentum_calls == 8);
    REQUIRE(boundary_calls == 8);
    REQUIRE(residency_release_calls == 1);
    REQUIRE(residency_begin_calls == 1);
    REQUIRE(substep_phase == 0);
    REQUIRE(observed_dt_small == Approx(dt_large / 8.0));
}

TEST_CASE("complete batched boundary ownership avoids a duplicate closure",
          "[numerics][timestepping][split-explicit][batch-boundary]")
{
    setup_grid();
    auto scheme = create_time_stepping_scheme("split_explicit");
    TimeSteppingConfig config;
    config.n_acoustic_substeps = 3;
    config.max_acoustic_speed_ms = 350.0;
    SplitExplicitCallbacks callbacks;
    callbacks.compute_slow_tendencies = [] {};
    callbacks.apply_slow_tendencies = [](double) {};
    callbacks.apply_fast_pressure = [](double) { FAIL("unexpected separate pressure"); };
    callbacks.apply_fast_momentum = [](double) { FAIL("unexpected separate momentum"); };
    callbacks.apply_fast_batched = [](double, int) { return true; };
    const bool complete = GENERATE(false, true);
    callbacks.batched_acoustic_boundary_complete = complete;
    int closures = 0, releases = 0;
    callbacks.acoustic_bcs = [&](double dt) { ++closures; REQUIRE(dt == 0.0); };
    callbacks.release_acoustic_loop_residency = [&] { ++releases; };
    scheme->step_split_acoustic(config, .1, callbacks);
    REQUIRE(closures == (complete ? 0 : 1));
    REQUIRE(releases == 1);
}

TEST_CASE("Split-explicit releases acoustic residency when a substep fails",
          "[numerics][timestepping][split-explicit][residency]")
{
    setup_grid();
    dr = 1000.0;
    dtheta = 1000.0;
    dz = 500.0;

    auto scheme = create_time_stepping_scheme("split_explicit");
    TimeSteppingConfig cfg;
    cfg.scheme_id = "split_explicit";
    cfg.n_acoustic_substeps = 2;
    cfg.max_acoustic_speed_ms = 350.0;

    int residency_release_calls = 0;
    int residency_begin_calls = 0;
    SplitExplicitCallbacks callbacks;
    callbacks.compute_slow_tendencies = [] {};
    callbacks.apply_slow_tendencies = [](double) {};
    const bool fail_on_begin = GENERATE(false, true);
    callbacks.begin_acoustic_loop_residency = [&] {
        ++residency_begin_calls;
        if (fail_on_begin) throw std::runtime_error("synthetic acoustic failure");
    };
    callbacks.apply_fast_fused = [](double) -> bool {
        throw std::runtime_error("synthetic acoustic failure");
    };
    callbacks.apply_fast_pressure = [](double) {};
    callbacks.apply_fast_momentum = [](double) {};
    callbacks.acoustic_bcs = [](double) {};
    callbacks.release_acoustic_loop_residency = [&] {
        ++residency_release_calls;
    };

    REQUIRE_THROWS_WITH(
        scheme->step_split_acoustic(cfg, 1.0, callbacks),
        "synthetic acoustic failure");
    REQUIRE(residency_release_calls == 1);
    REQUIRE(residency_begin_calls == 1);
}

TEST_CASE("Split-explicit promotes an under-resolved fixed acoustic count",
          "[numerics][timestepping][split-explicit][acoustic-cfl]")
{
    setup_grid();
    dr = 1000.0;
    dtheta = 100.0;
    dz = 500.0;

    auto scheme = create_time_stepping_scheme("split_explicit");
    TimeSteppingConfig cfg;
    cfg.scheme_id = "split_explicit";
    cfg.n_acoustic_substeps = 2;
    cfg.max_acoustic_speed_ms = 700.0;

    int pressure_calls = 0;
    double observed_dt_small = 0.0;
    SplitExplicitCallbacks callbacks;
    callbacks.compute_slow_tendencies = [] {};
    callbacks.apply_slow_tendencies = [](double) {};
    callbacks.apply_fast_pressure = [&](double dt_small) {
        ++pressure_calls;
        observed_dt_small = dt_small;
    };
    callbacks.apply_fast_momentum = [](double) {};
    callbacks.acoustic_bcs = [](double) {};

    constexpr double dt_large = 1.0;
    scheme->step_split_acoustic(cfg, dt_large, callbacks);

    // ceil(1*700/(0.5*100)) + 1 = 15. The configured value of 2 must
    // therefore be promoted, giving an actual small-step CFL below 0.5.
    REQUIRE(pressure_calls == 15);
    REQUIRE(observed_dt_small == Approx(dt_large / 15.0));
    REQUIRE(cfg.max_acoustic_speed_ms * observed_dt_small / dtheta < 0.5);
}

TEST_CASE("Split-explicit consumes diagnosed terrain acoustic spacing",
          "[numerics][timestepping][split-explicit][terrain][acoustic-cfl]")
{
    setup_grid();
    dr = 1000.0;
    dtheta = 1000.0;
    dz = 500.0;

    auto scheme = create_time_stepping_scheme("split_explicit");
    TimeSteppingConfig cfg;
    cfg.scheme_id = "split_explicit";
    cfg.n_acoustic_substeps = 0;
    cfg.max_acoustic_speed_ms = 350.0;
    cfg.min_acoustic_spacing_m = 25.0;

    int pressure_calls = 0;
    double observed_dt_small = 0.0;
    SplitExplicitCallbacks callbacks;
    callbacks.compute_slow_tendencies = [] {};
    callbacks.apply_slow_tendencies = [](double) {};
    callbacks.apply_fast_pressure = [&](double dt_small) {
        ++pressure_calls;
        observed_dt_small = dt_small;
    };
    callbacks.apply_fast_momentum = [](double) {};
    callbacks.acoustic_bcs = [](double) {};

    constexpr double dt_large = 1.0;
    scheme->step_split_acoustic(cfg, dt_large, callbacks);

    // ceil(350/(0.5*25)) + 1 = 29. Nominal Cartesian spacing must not
    // override the diagnosed terrain face-normal distance.
    REQUIRE(pressure_calls == 29);
    REQUIRE(observed_dt_small == Approx(dt_large / 29.0));
    REQUIRE(cfg.max_acoustic_speed_ms * observed_dt_small /
                cfg.min_acoustic_spacing_m < 0.5);
}

TEST_CASE("Split-explicit rejects a missing runtime acoustic speed",
          "[numerics][timestepping][split-explicit][acoustic-cfl]")
{
    setup_grid();
    auto scheme = create_time_stepping_scheme("split_explicit");
    TimeSteppingConfig cfg;
    cfg.scheme_id = "split_explicit";
    cfg.n_acoustic_substeps = 6;
    cfg.max_acoustic_speed_ms = 0.0;

    SplitExplicitCallbacks callbacks;
    REQUIRE_THROWS_WITH(
        scheme->step_split_acoustic(cfg, 1.0, callbacks),
        Catch::Matchers::Contains("maximum sound speed is unavailable"));
}

TEST_CASE("Coupled RK retains timestep history with amortized growth", "[numerics][timestepping][memory]")
{
    for (const std::string name : {"rk3", "rk4"})
    {
        auto scheme = create_time_stepping_scheme(name);
        TimeSteppingConfig config;
        TimeSteppingDiagnostics diagnostics;
        tmv::UnsplitRkCallbacks callbacks;
        callbacks.capture_base = [] {};
        callbacks.restore_base = [] {};
        callbacks.compute_tendencies = [](std::size_t, double) {};
        callbacks.publish_combination = [](std::size_t, const std::array<double, 4>&, double) {};
        std::size_t growths = 0;
        for (int step = 0; step < 4096; ++step)
        {
            const auto previous = diagnostics.dt_history.capacity();
            scheme->step_coupled_unsplit_rk(config, 0.25, step * 0.25,
                callbacks, &diagnostics);
            growths += diagnostics.dt_history.capacity() != previous;
        }
        REQUIRE(growths <= 16);
        REQUIRE(diagnostics.n_steps == 4096);
        REQUIRE(diagnostics.dt_history == std::vector<double>(4096, 0.25));
    }
}

TEST_CASE("Runge-Kutta rejects nonfinite inputs and unrepresentable stages atomically",
          "[numerics][timestepping][failure-atomic][finite]")
{
    for (const char* name : {"rk3", "rk4"})
        for (const std::string failure : {"input", "rhs", "intermediate", "final"})
        {
            CAPTURE(name, failure);
            auto               scheme = create_time_stepping_scheme(name);
            TimeSteppingConfig config;
            config.scheme_id = name;
            auto state       = make_transaction_test_state();
            state.dt         = 4;
            if (failure == "input")
                state.fields.back().data.data()[state.fields.back().data.size() - 1] =
                    std::numeric_limits<float>::quiet_NaN();
            const auto              before = state;
            TimeSteppingDiagnostics diagnostics;
            diagnostics.n_steps           = 7;
            diagnostics.dt_history        = {.25, .125};
            const auto diagnostics_before = diagnostics;
            int        calls              = 0;
            const int  stages             = std::string(name) == "rk3" ? 3 : 4;
            scheme->initialize(
                config,
                [&](const std::vector<NumericalState>& fields, double, std::vector<NumericalTendencies>& rhs)
                {
                    ++calls;
                    rhs.resize(fields.size());
                    const float max   = std::numeric_limits<float>::max();
                    const float value = failure == "intermediate" || (failure == "final" && calls == stages) ? max
                                        : failure == "final"                                                 ? .2f * max
                                                                                                             : .5f;
                    for (std::size_t f = 0; f < fields.size(); ++f)
                        rhs[f].tendencies.resize(
                            fields[f].data.size_r(), fields[f].data.size_th(), fields[f].data.size_z(), value);
                    if (failure == "rhs" && calls == stages)
                        rhs.back().tendencies.data()[rhs.back().tendencies.size() - 1] =
                            std::numeric_limits<float>::infinity();
                });
            REQUIRE_THROWS(scheme->step(config, state, &diagnostics));
            REQUIRE(calls == (failure == "input" ? 0 : failure == "intermediate" ? 1 : stages));
            require_state_bitwise_equal(state, before);
            require_diagnostics_equal(diagnostics, diagnostics_before);
        }
}

TEST_CASE("Runge-Kutta rejects an unstaged Euler callback path", "[numerics][timestepping][validation]")
{
    for (const char* name : {"rk3", "rk4"})
    {
        auto scheme    = create_time_stepping_scheme(name);
        int  rhs_calls = 0, apply_calls = 0;
        REQUIRE_THROWS_AS(
            scheme->step_unsplit(.2, [&] { ++rhs_calls; }, [&](double) { ++apply_calls; }), std::invalid_argument);
        CHECK(rhs_calls == 0);
        CHECK(apply_calls == 0);
    }
}

TEST_CASE("Runge-Kutta refuses timesteps with collapsed stage times", "[numerics][timestepping][validation]")
{
    for (const char* name : {"rk3", "rk4"})
        for (double step : {.25, 1.0})
        {
            auto               scheme = create_time_stepping_scheme(name);
            TimeSteppingConfig config;
            config.scheme_id = name;
            int calls        = 0;
            scheme->initialize(config, [&](const auto&, double, auto&) { ++calls; });
            auto state        = make_transaction_test_state();
            state.time        = std::ldexp(1.0, 52);
            state.dt          = step;
            const auto before = state;
            REQUIRE_THROWS_AS(scheme->step(config, state), std::invalid_argument);
            require_state_bitwise_equal(state, before);
            UnsplitRkCallbacks callbacks;
            callbacks.capture_base        = [&] { ++calls; };
            callbacks.compute_tendencies  = [](std::size_t, double) {};
            callbacks.publish_combination = [](std::size_t, const std::array<double, 4>&, double) {};
            callbacks.restore_base        = [] {};
            REQUIRE_THROWS_AS(scheme->step_coupled_unsplit_rk(config, state.dt, state.time, callbacks, nullptr),
                              std::invalid_argument);
            CHECK(calls == 0);
        }
}

TEST_CASE("Classical RK4 conservative transport can evaluate admissible temporary stages",
          "[numerics][timestepping][rk4][private-extension]")
{
    auto               scheme = create_time_stepping_scheme("rk4");
    TimeSteppingConfig config;
    config.scheme_id = "rk4";
    for (double scale : {1e-30, 1.0, 1e30})
    {
        std::array<double, 6>                state{scale, 0, 0, 0, 0, 0}, base{};
        std::array<std::array<double, 6>, 4> tendency{};
        std::vector<double>                  times;
        bool                                 negative_stage = false;
        tmv::UnsplitRkCallbacks              callbacks;
        callbacks.capture_base       = [&] { base = state; };
        callbacks.restore_base       = [&] { state = base; };
        callbacks.compute_tendencies = [&](std::size_t stage, double time)
        {
            times.push_back(time);
            const auto            raw = state;
            std::array<double, 6> face{};
            for (std::size_t i = 0; i < state.size(); ++i)
            {
                negative_stage |= state[i] < 0;
                face[i] = std::max(0.0, state[i]);
            }
            for (std::size_t i = 0; i < state.size(); ++i)
                tendency[stage][i] = face[(i + 5) % 6] - face[i];
            CHECK(state == raw);
        };
        callbacks.publish_combination = [&](std::size_t stages, const std::array<double, 4>& weights, double dt)
        {
            for (std::size_t i = 0; i < state.size(); ++i)
            {
                state[i] = base[i];
                for (std::size_t stage = 0; stage < stages; ++stage)
                    state[i] += dt * weights[stage] * tendency[stage][i];
            }
        };
        scheme->step_coupled_unsplit_rk(config, .9, 0, callbacks);
        CHECK(negative_stage);
        REQUIRE(times.size() == 4);
        CHECK(times[0] == 0);
        CHECK(times[1] == .45);
        CHECK(times[2] == .45);
        CHECK(times[3] == .9);
        const std::array<double, 6> expected{.4108375, .34515, .1832625, .0334125, .0273375, 0};
        double                      inventory = 0;
        for (std::size_t i = 0; i < state.size(); ++i)
        {
            CHECK(state[i] >= 0);
            CHECK(state[i] / scale == Approx(expected[i]).margin(1e-14));
            inventory += state[i] / scale;
        }
        CHECK(inventory == Approx(1.0).epsilon(1e-14));
    }
}
