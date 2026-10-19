#include "catch2/catch.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "microphysics/runtime.hpp"
#include "microphysics/schemes/milbrandt/milbrandt.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

constexpr double kDuration = 0.05;
constexpr double kRainCoefficient = 523.598775598;

float reference_z(float q, float n)
{
    const long double alpha = 2.0L;
    const long double lambda = std::cbrt(
        kRainCoefficient * static_cast<long double>(n)
        * std::tgamma(alpha + 4.0L) / (q * std::tgamma(alpha + 1.0L)));
    return static_cast<float>(
        n * std::tgamma(alpha + 7.0L)
        / (std::tgamma(alpha + 1.0L) * std::pow(lambda, 6.0L)));
}

bool same_bytes(const Field3D& a, const Field3D& b)
{
    return a.size_r() == b.size_r() && a.size_th() == b.size_th()
        && a.size_z() == b.size_z() && a.size() == b.size()
        && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

class ObservedMilbrandt final : public tmv::MilbrandtScheme
{
public:
    bool reject = false;
    bool accepted_called = false;

    std::unique_ptr<MicrophysicsScheme> clone() const override
    {
        return std::make_unique<ObservedMilbrandt>(*this);
    }

    bool prepare_post_primary_private_commit(
        double dt,
        const tmv::MicrophysicsAcceptedPrimaryFields& accepted,
        std::string& error) override
    {
        if (!MilbrandtScheme::prepare_post_primary_private_commit(dt, accepted, error))
        {
            return false;
        }
        accepted_called = true;
        if (reject)
        {
            error = "injected after live freezing accepted-primary validation";
            return false;
        }
        return true;
    }
};

std::array<Field3D*, 15> public_fields()
{
    return {&rho, &p, &u, &v, &w, &theta, &qv, &qc,
            &qr, &qi, &qs, &qg, &qh, &tracer, &tke};
}

void set_temperature(double temperature)
{
    for (int i = 0; i < NR; ++i)
    {
        theta(i, 0, 0) = static_cast<float>(
            thermodynamics::temperature_to_theta(temperature, p(i, 0, 0)));
    }
}

void initialize_state(bool hail = true, bool observed = false,
                      double temperature = 253.15)
{
    NR = 2;
    NTH = 1;
    NZ = 1;
    dr = 2.0;
    dtheta = 3.0;
    dz = 5.0;
    dt = kDuration;
    simulation_time = 0.0;
    global_grid_geometry.initialize(
        NR, NTH, NZ, dr, dz, dtheta,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    global_gpu_apply_microphysics = false;
    global_validation_policy = {};
    global_validation_policy.mode = tmv::GuardMode::Strict;
    global_validation_policy.fail_on = tmv::GuardFailOn::Both;
    global_simulation_storage.configuration.microphysics_scheme = "milbrandt";
    global_simulation_storage.configuration.microphysics_config = {};
    global_simulation_storage.runtime.boundary_layer = {};
    global_simulation_storage.runtime.chaos = {};
    global_simulation_storage.runtime.terrain = {};
    global_simulation_storage.runtime.radiation = {};
    global_simulation_storage.runtime.compute = {};
    global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Cpu;
    global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Cpu;
    global_simulation_storage.runtime.boundary_layer.config.scheme_id = "none";
    global_simulation_storage.runtime.chaos.config.scheme_id = "none";
    global_simulation_storage.runtime.terrain.config.scheme_id = "none";
    global_simulation_storage.runtime.radiation.config.scheme_id = "none";
    global_microphysics_runtime_state.theta_storage_residual = {};
    global_microphysics_runtime_state.last_water_repair_report = {};
    global_microphysics_runtime_state.last_thermodynamic_repair_report = {};
    global_microphysics_runtime_state.last_conservation_report = {};
    for (Field3D* field : public_fields())
    {
        field->resize(NR, NTH, NZ, 0.0f);
    }
    for (Field3D* field : {&dtheta_dt_rad, &dtheta_dt_pbl, &dqv_dt_pbl,
                           &dqc_dt_pbl, &dqi_dt_pbl, &du_dt_pbl,
                           &dv_dt_pbl, &dtke_dt_pbl})
    {
        field->resize(NR, NTH, NZ, 0.0f);
    }
    global_simulation_storage.surface.precipitation_rate_mm_h.assign(2, 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_mm = {0.25f, 0.75f};
    global_simulation_storage.surface.accumulated_rainfall_residual_mm = {0.0f, 0.0f};
    global_simulation_storage.surface.accumulated_rainfall_last_update_s = 0.0;
    rho0_base.assign(1, 1.0);
    p0_base.assign(1, 90000.0);
    qv0_base.assign(1, 0.001);
    theta0_base.assign(1, 300.0);
    u0_base.assign(1, 0.0);
    v0_base.assign(1, 0.0);
    Field3D rain_number(NR, NTH, NZ, 0.0f);
    Field3D rain_z(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
    {
        rho(i, 0, 0) = i == 0 ? 0.65f : 1.3f;
        p(i, 0, 0) = i == 0 ? 90000.0f : 70000.0f;
        qv(i, 0, 0) = static_cast<float>(
            0.5 * thermodynamics::saturation_mixing_ratio_ice(
                temperature, p(i, 0, 0)));
        qr(i, 0, 0) = i == 0 ? 8.0e-4f : 1.2e-3f;
        rain_number(i, 0, 0) = i == 0 ? 7.0e-5f : 1.3e-4f;
        rain_z(i, 0, 0) = reference_z(qr(i, 0, 0), rain_number(i, 0, 0));
    }
    set_temperature(temperature);
    std::unique_ptr<tmv::MilbrandtScheme> scheme = observed
        ? std::unique_ptr<tmv::MilbrandtScheme>(std::make_unique<ObservedMilbrandt>())
        : std::make_unique<tmv::MilbrandtScheme>();
    tmv::MicrophysicsConfig configuration;
    configuration.milbrandt.hail_processes = hail;
    scheme->configure(configuration);
    std::string error;
    REQUIRE(scheme->enable_internal_triple_rain_freezing_for_test(
        global_simulation_state(), rain_number, rain_z, error));
    microphysics_scheme = std::move(scheme);
}

struct Snapshot
{
    std::array<Field3D, 15> fields;
    std::array<Field3D, 11> moments;
    std::array<Field3D, 8> tendencies;
    Field3D theta_residual;
    tmv::MicrophysicsWaterRepairReport water;
    tmv::MicrophysicsThermodynamicRepairReport heat;
    tmv::MicrophysicsConservationReport conservation;
    std::vector<float> rate;
    std::vector<float> rainfall;
    std::vector<float> rainfall_residual;
    double rainfall_time;

    Snapshot()
        : theta_residual(global_microphysics_runtime_state.theta_storage_residual),
          water(global_microphysics_runtime_state.last_water_repair_report),
          heat(global_microphysics_runtime_state.last_thermodynamic_repair_report),
          conservation(global_microphysics_runtime_state.last_conservation_report),
          rate(global_simulation_storage.surface.precipitation_rate_mm_h),
          rainfall(global_simulation_storage.surface.accumulated_rainfall_mm),
          rainfall_residual(global_simulation_storage.surface.accumulated_rainfall_residual_mm),
          rainfall_time(global_simulation_storage.surface.accumulated_rainfall_last_update_s)
    {
        const auto owners = public_fields();
        const auto& scheme = dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
        for (std::size_t index = 0; index < fields.size(); ++index)
        {
            fields[index] = *owners[index];
        }
        moments[0] = scheme.internal_triple_cloud_number_for_test();
        for (std::size_t species = 0; species < 5; ++species)
        {
            moments[1 + species] = scheme.internal_triple_number_for_test(species);
            moments[6 + species] = scheme.internal_triple_reflectivity_for_test(species);
        }
        const auto& t = global_microphysics_runtime_state.tendencies;
        tendencies = {t.dtheta_dt, t.dqv_dt, t.dqc_dt, t.dqr_dt,
                      t.dqi_dt, t.dqs_dt, t.dqg_dt, t.dqh_dt};
    }
};

void check_report_identity(const Snapshot& before)
{
    const auto& water = global_microphysics_runtime_state.last_water_repair_report;
    CHECK(water.nonfinite_tendency_count == before.water.nonfinite_tendency_count);
    for (std::size_t species = 0; species < water.species.size(); ++species)
    {
        const auto& a = water.species[species];
        const auto& b = before.water.species[species];
        CHECK(a.storage_rounding_kg == b.storage_rounding_kg);
        CHECK(a.lower_bound_correction_kg == b.lower_bound_correction_kg);
        CHECK(a.lower_bound_roundoff_correction_kg == b.lower_bound_roundoff_correction_kg);
        CHECK(a.upper_bound_correction_kg == b.upper_bound_correction_kg);
        CHECK(a.nonfinite_correction_kg == b.nonfinite_correction_kg);
        CHECK(a.lower_bound_count == b.lower_bound_count);
        CHECK(a.lower_bound_roundoff_count == b.lower_bound_roundoff_count);
        CHECK(a.upper_bound_count == b.upper_bound_count);
        CHECK(a.nonfinite_count == b.nonfinite_count);
    }
    const auto& heat = global_microphysics_runtime_state.last_thermodynamic_repair_report;
    const auto& b = before.heat;
    CHECK(heat.microphysics_heat_j == b.microphysics_heat_j);
    CHECK(heat.radiation_heat_j == b.radiation_heat_j);
    CHECK(heat.boundary_layer_heat_j == b.boundary_layer_heat_j);
    CHECK(heat.theta_tendency_limit_correction_j == b.theta_tendency_limit_correction_j);
    CHECK(heat.theta_nonfinite_correction_j == b.theta_nonfinite_correction_j);
    CHECK(heat.theta_storage_rounding_j == b.theta_storage_rounding_j);
    CHECK(heat.theta_storage_residual_release_j == b.theta_storage_residual_release_j);
    CHECK(heat.theta_storage_residual_j == b.theta_storage_residual_j);
    CHECK(heat.theta_bound_correction_j == b.theta_bound_correction_j);
    CHECK(heat.microphysics_water_tendency_kg == b.microphysics_water_tendency_kg);
    CHECK(heat.microphysics_water_tendency_kg_by_species == b.microphysics_water_tendency_kg_by_species);
    CHECK(heat.boundary_layer_vapor_tendency_kg == b.boundary_layer_vapor_tendency_kg);
    CHECK(heat.boundary_layer_cloud_liquid_tendency_kg == b.boundary_layer_cloud_liquid_tendency_kg);
    CHECK(heat.boundary_layer_cloud_ice_tendency_kg == b.boundary_layer_cloud_ice_tendency_kg);
    CHECK(heat.moist_dry_internal_change_j == b.moist_dry_internal_change_j);
    CHECK(heat.moist_water_sensible_change_j == b.moist_water_sensible_change_j);
    CHECK(heat.moist_phase_reference_change_j == b.moist_phase_reference_change_j);
    CHECK(heat.moist_carrier_kinetic_change_j == b.moist_carrier_kinetic_change_j);
    CHECK(heat.moist_gravitational_change_j == b.moist_gravitational_change_j);
    CHECK(heat.moist_subgrid_tke_change_j == b.moist_subgrid_tke_change_j);
    CHECK(heat.moist_committed_energy_change_j == b.moist_committed_energy_change_j);
    CHECK(heat.moist_energy_closure_residual_j == b.moist_energy_closure_residual_j);
    CHECK(heat.moist_energy_complete == b.moist_energy_complete);
    CHECK(heat.theta_tendency_limit_count == b.theta_tendency_limit_count);
    CHECK(heat.theta_nonfinite_count == b.theta_nonfinite_count);
    CHECK(heat.theta_lower_bound_count == b.theta_lower_bound_count);
    CHECK(heat.theta_upper_bound_count == b.theta_upper_bound_count);
    const auto& c = global_microphysics_runtime_state.last_conservation_report;
    CHECK(c.active == before.conservation.active);
    CHECK(c.tendency_imbalance_kg == before.conservation.tendency_imbalance_kg);
    CHECK(c.baseline_q_kg == before.conservation.baseline_q_kg);
    CHECK(c.surface_precipitation_kg == before.conservation.surface_precipitation_kg);
    CHECK(c.relative_imbalance == before.conservation.relative_imbalance);
}

void check_snapshot(const Snapshot& before)
{
    const Snapshot after;
    for (std::size_t index = 0; index < before.fields.size(); ++index)
    {
        CHECK(same_bytes(after.fields[index], before.fields[index]));
    }
    for (std::size_t index = 0; index < before.moments.size(); ++index)
    {
        CHECK(same_bytes(after.moments[index], before.moments[index]));
    }
    for (std::size_t index = 0; index < before.tendencies.size(); ++index)
    {
        CHECK(same_bytes(after.tendencies[index], before.tendencies[index]));
    }
    CHECK(same_bytes(after.theta_residual, before.theta_residual));
    CHECK(after.rate == before.rate);
    CHECK(after.rainfall == before.rainfall);
    CHECK(after.rainfall_residual == before.rainfall_residual);
    CHECK(after.rainfall_time == before.rainfall_time);
    check_report_identity(before);
}

} // namespace

TEST_CASE("live triple rain freezing closes published q N Z and pressure-dependent heat",
          "[physics][milbrandt][triple_cold_internal][direct]")
{
    for (const bool hail : {false, true})
    {
        std::array<std::unique_ptr<Snapshot>, 2> after;
        for (int run = 0; run < 2; ++run)
        {
#ifdef _OPENMP
            omp_set_num_threads(run == 0 ? 1 : 4);
#endif
            initialize_state(hail, true);
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), kDuration));
            const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
            REQUIRE(scheme.accepted_called);
            const std::size_t destination = hail ? 4 : 3;
            const Field3D& frozen = hail ? qh : qg;
            const long double frozen_c = hail ? 471.238898038L : 209.439510239L;
            long double heat_sum = 0.0L;
            for (int i = 0; i < NR; ++i)
            {
                const long double old_q = before.fields[8](i, 0, 0);
                const long double old_n = before.moments[1](i, 0, 0);
                const long double old_z = before.moments[6](i, 0, 0);
                const double temperature = thermodynamics::theta_to_temperature(
                    before.fields[5](i, 0, 0), p(i, 0, 0));
                const long double frozen_n = std::min(old_n,
                    kDuration * 100.0L * (1.0L - std::exp(
                        0.66L * (temperature - microphysics_constants::T0)))
                    * old_q / microphysics_constants::rho_w);
                const long double frozen_q = std::min(old_q, frozen_n * old_q / old_n);
                const long double fraction = frozen_n / old_n;
                REQUIRE(fraction > 0.0L);
                REQUIRE(fraction < 1.0L);
                CHECK(qr(i, 0, 0) == Approx(static_cast<double>(old_q - frozen_q)).epsilon(4.0e-7));
                CHECK(frozen(i, 0, 0) == Approx(static_cast<double>(frozen_q)).epsilon(4.0e-7));
                CHECK(scheme.internal_triple_number_for_test(0)(i, 0, 0)
                      == Approx(static_cast<double>(old_n - frozen_n)).epsilon(4.0e-7));
                CHECK(scheme.internal_triple_number_for_test(destination)(i, 0, 0)
                      == Approx(static_cast<double>(frozen_n)).epsilon(4.0e-7));
                const long double frozen_z = fraction * old_z
                    * std::pow(kRainCoefficient / frozen_c, 2.0L);
                CHECK(scheme.internal_triple_reflectivity_for_test(0)(i, 0, 0)
                      == Approx(static_cast<double>((1.0L - fraction) * old_z)).epsilon(4.0e-7));
                CHECK(scheme.internal_triple_reflectivity_for_test(destination)(i, 0, 0)
                      == Approx(static_cast<double>(frozen_z)).epsilon(4.0e-7));
                const long double new_q = qr(i, 0, 0) + static_cast<long double>(frozen(i, 0, 0));
                const long double new_n = scheme.internal_triple_number_for_test(0)(i, 0, 0)
                    + static_cast<long double>(scheme.internal_triple_number_for_test(destination)(i, 0, 0));
                const long double scaled_z = kRainCoefficient * kRainCoefficient
                    * scheme.internal_triple_reflectivity_for_test(0)(i, 0, 0)
                    + frozen_c * frozen_c
                    * scheme.internal_triple_reflectivity_for_test(destination)(i, 0, 0);
                for (const long double scale : {1.0L, static_cast<long double>(rho(i, 0, 0))})
                {
                    CHECK(static_cast<double>(scale * new_q) == Approx(static_cast<double>(scale * old_q)).epsilon(4.0e-7));
                    CHECK(static_cast<double>(scale * new_n) == Approx(static_cast<double>(scale * old_n)).epsilon(4.0e-7));
                    CHECK(static_cast<double>(scale * scaled_z)
                          == Approx(static_cast<double>(scale * kRainCoefficient * kRainCoefficient * old_z)).epsilon(4.0e-7));
                }
                const long double heat_increment = microphysics_constants::L_f
                    / microphysics_constants::cp * frozen_q;
                const double expected_theta = before.fields[5](i, 0, 0)
                    + static_cast<double>(heat_increment) * std::pow(
                        microphysics_constants::p0 / p(i, 0, 0),
                        microphysics_constants::R_d / microphysics_constants::cp);
                CHECK(theta(i, 0, 0) == Approx(expected_theta).margin(3.1e-5));
                CHECK(theta(i, 0, 0) > before.fields[5](i, 0, 0));
                heat_sum += rho(i, 0, 0) * dr * dtheta * dz
                    * microphysics_constants::L_f * frozen_q;
                CHECK(qv(i, 0, 0) == before.fields[6](i, 0, 0));
                CHECK(qc(i, 0, 0) == 0.0f);
                for (std::size_t species = 1; species < 5; ++species)
                {
                    if (species != destination)
                    {
                        CHECK(scheme.internal_triple_number_for_test(species)(i, 0, 0) == 0.0f);
                        CHECK(scheme.internal_triple_reflectivity_for_test(species)(i, 0, 0) == 0.0f);
                    }
                }
                for (std::size_t species = 0; species < 5; ++species)
                {
                    const auto surface = scheme.internal_triple_surface_for_test(i, 0, species);
                    CHECK(surface.mass == 0.0);
                    CHECK(surface.number == 0.0);
                    CHECK(surface.reflectivity == 0.0);
                }
            }
            CHECK(global_microphysics_runtime_state.last_thermodynamic_repair_report.microphysics_heat_j
                  == Approx(static_cast<double>(heat_sum)).epsilon(2.0e-6));
            CHECK(global_microphysics_runtime_state.last_water_repair_report.lower_bound_count() == 0);
            CHECK(global_microphysics_runtime_state.last_water_repair_report.upper_bound_count() == 0);
            CHECK(global_microphysics_runtime_state.last_conservation_report.surface_precipitation_kg == 0.0);
            CHECK(global_simulation_storage.surface.accumulated_rainfall_mm == before.rainfall);
            CHECK(global_simulation_storage.surface.precipitation_rate_mm_h == before.rate);
            after[run] = std::make_unique<Snapshot>();
        }
        for (std::size_t index = 0; index < after[0]->fields.size(); ++index)
        {
            CHECK(same_bytes(after[0]->fields[index], after[1]->fields[index]));
        }
        for (std::size_t index = 0; index < after[0]->moments.size(); ++index)
        {
            CHECK(same_bytes(after[0]->moments[index], after[1]->moments[index]));
        }
        CHECK(after[0]->rainfall == after[1]->rainfall);
        CHECK(after[0]->rate == after[1]->rate);
        CHECK(after[0]->heat.microphysics_heat_j == after[1]->heat.microphysics_heat_j);
    }
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

TEST_CASE("live triple freezing hook failure restores every public private report and fallout owner",
          "[physics][milbrandt][triple_cold_internal][direct]")
{
    initialize_state(true, true, 268.15);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), kDuration));
    set_temperature(253.15);
    for (int i = 0; i < NR; ++i)
    {
        qv(i, 0, 0) = static_cast<float>(
            0.5 * thermodynamics::saturation_mixing_ratio_ice(253.15, p(i, 0, 0)));
    }
    dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).reject = true;
    const Snapshot before;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), kDuration),
        "post-primary microphysics private candidate rejected: "
        "injected after live freezing accepted-primary validation");
    check_snapshot(before);
    const auto& scheme = dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
    for (int i = 0; i < NR; ++i)
    {
        for (std::size_t species = 0; species < 5; ++species)
        {
            const auto surface = scheme.internal_triple_surface_for_test(i, 0, species);
            CHECK(surface.mass == 0.0);
            CHECK(surface.number == 0.0);
            CHECK(surface.reflectivity == 0.0);
        }
    }
}

TEST_CASE("live triple freezing rejects a late newly active unowned cold event atomically",
          "[physics][milbrandt][triple_cold_internal][direct]")
{
    initialize_state(true, true);
    qv(1, 0, 0) = static_cast<float>(2.0 * thermodynamics::saturation_mixing_ratio_ice(
        thermodynamics::theta_to_temperature(theta(1, 0, 0), p(1, 0, 0)), p(1, 0, 0)));
    const Snapshot before;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), kDuration),
        "internal Milbrandt RainFreezingOnly observed an unowned cold channel");
    check_snapshot(before);
    CHECK_FALSE(dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme).accepted_called);
    std::array<Field3D, 8> unpublished;
    for (auto& field : unpublished)
    {
        field.resize(NR, NTH, NZ, 42.0f);
    }
    auto& rejected = dynamic_cast<tmv::MilbrandtScheme&>(*microphysics_scheme);
    REQUIRE_THROWS_WITH(rejected.compute_tendencies(
        p, theta, rho, qv, qc, qr, qi, qs, qg, qh, kDuration,
        unpublished[0], unpublished[1], unpublished[2], unpublished[3],
        unpublished[4], unpublished[5], unpublished[6], unpublished[7]),
        "internal Milbrandt RainFreezingOnly observed an unowned cold channel");
    for (const auto& field : unpublished)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42.0f);
        }
    }
    std::string error;
    CHECK_FALSE(rejected.validate_prognostic_update(kDuration, error));
    CHECK_FALSE(rejected.commit_prepared_private_candidate());
    qv(1, 0, 0) = before.fields[6](0, 0, 0);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), kDuration));
}

TEST_CASE("live triple freezing zero event still validates all inputs and rejects unowned continuation",
          "[physics][milbrandt][triple_cold_internal][direct]")
{
    for (int mutation = 0; mutation < 11; ++mutation)
    {
        initialize_state(true, true, 268.15);
        switch (mutation)
        {
            case 0: rho(1, 0, 0) = 0.0f; break;
            case 1: p(1, 0, 0) = 0.0f; break;
            case 2: theta(1, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
            case 3: qv(1, 0, 0) = -1.0f; break;
            case 4: qc(1, 0, 0) = 1.0e-4f; break;
            case 5: qi(1, 0, 0) = 1.0e-4f; break;
            case 6: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
            case 7: dtheta_dt_rad(1, 0, 0) = 1.0f; break;
            case 8: qr(1, 0, 0) = -1.0f; break;
            case 9: qr(1, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
            case 10: set_temperature(280.0); break;
        }
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), kDuration));
        check_snapshot(before);
    }
    initialize_state();
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), kDuration));
    const Snapshot before;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), kDuration),
        "internal Milbrandt RainFreezingOnly runtime owner changed: "
        "internal Milbrandt RainFreezingOnly has an unowned cloud or frozen category");
    check_snapshot(before);
    const auto& scheme = dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
    CHECK_THROWS(scheme.portable_state());
    CHECK_THROWS(scheme.clone_parameters());
    CHECK_THROWS(microphysics_scheme->prognostic_scalars_for_transport());
}

TEST_CASE("internal triple freezing activation rejects unsupported parameters and moment state",
          "[physics][milbrandt][triple_cold_internal][direct]")
{
    initialize_state();
    const auto& initial = dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
    const Field3D rain_n(initial.internal_triple_number_for_test(0));
    const Field3D rain_z(initial.internal_triple_reflectivity_for_test(0));
    for (int mutation = 0; mutation < 10; ++mutation)
    {
        Field3D n(rain_n);
        Field3D z(rain_z);
        auto scheme = mutation == 0
            ? std::make_unique<tmv::MilbrandtScheme>(1.0)
            : std::make_unique<tmv::MilbrandtScheme>();
        switch (mutation)
        {
            case 1: n(1, 0, 0) = 0.0f; break;
            case 2: z(1, 0, 0) = 0.0f; break;
            case 3: z(1, 0, 0) = -1.0f; break;
            case 4: n(1, 0, 0) = std::numeric_limits<float>::infinity(); break;
            case 5: z.resize(1, 1, 1, 0.0f); break;
            case 6: p(1, 0, 0) = 200001.0f; break;
            case 7: set_temperature(280.0); break;
            case 8: qv(1, 0, 0) = 0.1f; break;
            case 9: qh(1, 0, 0) = 1.0e-4f; break;
        }
        const Field3D n_before(n);
        const Field3D z_before(z);
        std::string error;
        CHECK_FALSE(scheme->enable_internal_triple_rain_freezing_for_test(
            global_simulation_state(), n, z, error));
        CHECK_FALSE(error.empty());
        CHECK_FALSE(scheme->owns_post_primary_private_commit());
        CHECK(same_bytes(n, n_before));
        CHECK(same_bytes(z, z_before));
        p(1, 0, 0) = 70000.0f;
        qv(1, 0, 0) = 0.001f;
        qh(1, 0, 0) = 0.0f;
        set_temperature(253.15);
        qv(1, 0, 0) = static_cast<float>(0.5 * thermodynamics::saturation_mixing_ratio_ice(
            253.15, p(1, 0, 0)));
    }
}
