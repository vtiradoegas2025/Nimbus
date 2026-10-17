#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "core/infra/nested_feedback.hpp"
#include "core/runtime/simulation_state.hpp"
#include "dynamics/eos_pressure.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>

TEST_CASE("Phase-coupled reflux preserves total water and thermal energy after cloud evaporation",
          "[core][nested_grid][two_way][phase_projection]")
{
    const auto relationship =
        tmv::domain::prepare_one_way_nest_relationship({12, 10, 4, 400, 300, .2, 2, 2, 2, 4, 3, 6, 6, 4, 1, 1});
    PrognosticFieldStorage parent, child;
    const auto             initialize = [](PrognosticFieldStorage& fields, int nx, int ny)
    {
        for (auto* f : {&fields.rho,
                        &fields.p,
                        &fields.theta,
                        &fields.u,
                        &fields.v,
                        &fields.w,
                        &fields.qv,
                        &fields.qc,
                        &fields.qr,
                        &fields.qi,
                        &fields.qs,
                        &fields.qg,
                        &fields.qh,
                        &fields.tracer,
                        &fields.tke})
            f->resize(nx, ny, 4, 0);
        fields.rho.fill(1);
        fields.theta.fill(300);
        fields.p.fill(float(tmv::dynamics_eos::pressure_fp64(1, 300)));
        fields.qv.fill(.01f);
    };
    initialize(parent, 14, 12);
    initialize(child, 8, 8);
    tmv::CartesianDryMassInterfaceFluxes mass;
    tmv::CartesianThetaInterfaceFluxes   theta;
    const auto                           scalar = [](auto& flux)
    {
        flux.west.assign(12, 0);
        flux.east.assign(12, 0);
        flux.south.assign(12, 0);
        flux.north.assign(12, 0);
        flux.accumulated_dt_s   = .2;
        flux.contribution_count = 1;
    };
    scalar(mass);
    scalar(theta);
    tmv::CartesianMomentumInterfaceFluxes momentum;
    for (auto* side : {&momentum.west, &momentum.east, &momentum.south, &momentum.north})
        for (auto& values : side->component)
            values.assign(9, 0);
    momentum.accumulated_dt_s   = .2;
    momentum.contribution_count = 1;
    std::array<tmv::CartesianThetaInterfaceFluxes, 7> coarse, fine;
    NestedMoistFeedbackInputs                         water;
    for (std::size_t f = 0; f < 7; ++f)
    {
        scalar(coarse[f]);
        scalar(fine[f]);
        water.parent_flux[f] = &coarse[f];
        water.child_flux[f]  = &fine[f];
    }
    const auto       inventory = nested_two_way_child_inventory_flat(relationship, {100, 100, 100, 100}, child);
    constexpr double volume    = 400 * 300 * 100;
    constexpr double deficit   = .0001 * volume;
    water.child_begin[0]       = inventory[0] * double(.01f) - deficit;
    water.physics_source[0]    = deficit;
    water.physics_source[1]    = -deficit;
    fine[1].west[0]            = deficit;
    const double                pressure_before = parent.p(4, 4, 0);
    ComputeBackendRuntimeState  runtime;
    ComputeBackendRuntimeState* selected = nullptr;
    if (const char* backend = std::getenv("NIMBUS_NESTED_FEEDBACK_BACKEND"))
    {
        runtime.config.backend        = backend;
        runtime.config.allow_fallback = false;
        std::string error;
        const bool  initialized = initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error);
        INFO(error);
        REQUIRE(initialized);
        selected = &runtime;
    }
    NestedTwoWayFeedbackReportV1 report;
    std::array<Field3D, 7>       fine_candidates;
    apply_nested_two_way_feedback_flat(relationship,
                                       {100, 100, 100, 100},
                                       mass,
                                       mass,
                                       theta,
                                       theta,
                                       momentum,
                                       momentum,
                                       momentum,
                                       momentum,
                                       {},
                                       {},
                                       {},
                                       inventory,
                                       parent,
                                       child,
                                       report,
                                       &water,
                                       &fine_candidates,
                                       selected);
    REQUIRE(report.complete);
    REQUIRE(report.phase_projection_cells == 1);
    REQUIRE(valid_nested_phase_feedback_report(report));
    CHECK(parent.qc(4, 4, 0) == 0);
    CHECK(parent.qv(4, 4, 0) == Approx(double(.01f) - .0001).margin(1e-9));
    CHECK(parent.theta(4, 4, 0) > 300);
    CHECK(report.water[0].parent.phase_exchange_inventory == Approx(-deficit).epsilon(2e-7));
    CHECK(report.water[1].parent.phase_exchange_inventory == Approx(deficit).epsilon(2e-7));
    namespace pc                     = physical_constants;
    constexpr double kappa           = pc::gas_constant_dry_air_jkgk / pc::specific_heat_cp_jkgk;
    constexpr double cvd             = pc::specific_heat_cp_jkgk - pc::gas_constant_dry_air_jkgk;
    constexpr double cvv             = pc::specific_heat_water_vapor_cp_jkgk - pc::gas_constant_water_vapor_jkgk;
    constexpr double cl              = pc::specific_heat_liquid_water_jkgk;
    constexpr double tr              = pc::freezing_temperature_k;
    constexpr double ev              = pc::latent_heat_vaporization_jkg - pc::gas_constant_water_vapor_jkgk * tr;
    const double     old_temperature = 300 * std::pow(pressure_before / pc::reference_pressure_pa, kappa);
    const double     cv_before       = cvd + double(.01f) * cvv + double(float(-.0001)) * cl;
    const double     exchange        = -double(float(-.0001));
    const double     expected        = old_temperature + exchange * (ev + (cvv - cl) * (old_temperature - tr)) /
                                                             (cvd + (double(.01f) - exchange) * cvv);
    const double     actual = parent.theta(4, 4, 0) * std::pow(parent.p(4, 4, 0) / pc::reference_pressure_pa, kappa);
    CHECK(actual == Approx(expected).margin(5e-5));
    const double before = cv_before * (old_temperature - tr) + double(.01f) * ev;
    const double after  = (cvd + parent.qv(4, 4, 0) * cvv) * (actual - tr) + parent.qv(4, 4, 0) * ev;
    CHECK((after - before) * volume == Approx(report.phase_thermal_roundoff_j).margin(.1));
    for (const auto& item : report.water)
        CHECK(valid_nested_water_feedback_report(item));
    auto invalid = report;
    invalid.water[1].parent.phase_exchange_inventory += 1;
    CHECK_FALSE(valid_nested_phase_feedback_report(invalid));
}
