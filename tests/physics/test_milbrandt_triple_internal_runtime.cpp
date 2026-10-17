#include "catch2/catch.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "microphysics/runtime.hpp"
#include "microphysics/schemes/milbrandt/milbrandt.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_adapter.hpp"

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

float reference_z(float q, float n, double c, double d,
                  double alpha = 2.0)
{
    const double lambda = std::pow(
        c * n * std::tgamma(alpha + d + 1.0)
            / (q * std::tgamma(alpha + 1.0)),
        1.0 / d);
    return static_cast<float>(
        n * std::tgamma(alpha + 7.0)
            / (std::tgamma(alpha + 1.0) * std::pow(lambda, 6.0)));
}

bool same_bytes(const Field3D& a, const Field3D& b)
{
    return a.size_r() == b.size_r() && a.size_th() == b.size_th()
        && a.size_z() == b.size_z() && a.size() == b.size()
        && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

class LateRejectMilbrandt final : public tmv::MilbrandtScheme
{
public:
    std::unique_ptr<MicrophysicsScheme> clone() const override
    {
        return std::make_unique<LateRejectMilbrandt>(*this);
    }

    bool prepare_post_primary_private_commit(
        double dt,
        const tmv::MicrophysicsAcceptedPrimaryFields& accepted,
        std::string& error) override
    {
        if (!MilbrandtScheme::prepare_post_primary_private_commit(
                dt, accepted, error))
        {
            return false;
        }
        error = "injected after five-Z accepted-primary validation";
        return false;
    }
};

void initialize_state(bool late_reject)
{
    NR = 2;
    NTH = 1;
    NZ = 2;
    dr = 2.0;
    dtheta = 3.0;
    dz = 5.0;
    dt = 0.05;
    simulation_time = 0.0;
    global_grid_geometry.initialize(
        NR, NTH, NZ, dr, dz, dtheta,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    global_gpu_apply_microphysics = false;
    global_validation_policy = {};
    global_validation_policy.mode = tmv::GuardMode::Sanitize;
    const auto resize = [](Field3D& field, float value)
    {
        field.resize(NR, NTH, NZ, value);
    };
    for (Field3D* field : {&rho, &p, &u, &v, &w, &tracer, &theta,
                           &qv, &qc, &qr, &qi, &qs, &qg, &qh, &tke})
    {
        resize(*field, 0.0f);
    }
    resize(rho, 1.0f);
    resize(p, 90000.0f);
    resize(theta, 300.0f);
    resize(qv, 0.010f);
    for (Field3D* field : {&dtheta_dt_rad, &dtheta_dt_pbl, &dqv_dt_pbl,
                           &dqc_dt_pbl, &dqi_dt_pbl, &du_dt_pbl,
                           &dv_dt_pbl, &dtke_dt_pbl})
    {
        resize(*field, 0.0f);
    }
    global_simulation_storage.surface.precipitation_rate_mm_h.assign(2, 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_mm.assign(2, 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_residual_mm.assign(2, 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_last_update_s = 0.0;
    rho0_base.assign(2, 1.0);
    p0_base.assign(2, 90000.0);
    qv0_base.assign(2, 0.010);
    theta0_base.assign(2, 300.0);
    u0_base.assign(2, 0.0);
    v0_base.assign(2, 0.0);
    global_radiation_runtime_state.config.scheme_id = "none";
    global_radiation_runtime_state.held_atmospheric_power_w = 0.0;
    global_radiation_runtime_state.held_cell_power_w.release_storage();
    global_simulation_storage.runtime.boundary_layer.config.clear_sky_only = true;

    const std::array<Field3D*, 5> mass{&qr, &qi, &qs, &qg, &qh};
    std::array<Field3D, 5> number;
    std::array<Field3D, 5> reflectivity;
    for (std::size_t species = 0; species < 5; ++species)
    {
        number[species].resize(NR, NTH, NZ, 0.0f);
        reflectivity[species].resize(NR, NTH, NZ, 0.0f);
        const auto parameters = tmv::milbrandt_detail::
            current_nimbus_triple_sedimentation_parameters(
                static_cast<tmv::milbrandt_detail::TripleMomentSpecies>(species));
        for (int i = 0; i < NR; ++i)
        {
            for (int k = 0; k < NZ; ++k)
            {
                rho(i, 0, k) = 1.0f + 0.1f * k;
                (*mass[species])(i, 0, k) = static_cast<float>(
                    (i + 1) * (k + 1) * (species + 1) * 1.0e-6);
                number[species](i, 0, k) = static_cast<float>(
                    (i + 1) * (k + 1) * (species + 1) * 10000.0);
                reflectivity[species](i, 0, k) = reference_z(
                    (*mass[species])(i, 0, k), number[species](i, 0, k),
                    parameters.mass_law.coefficient,
                    parameters.mass_law.exponent);
            }
        }
    }
    std::unique_ptr<tmv::MilbrandtScheme> scheme = late_reject
        ? std::unique_ptr<tmv::MilbrandtScheme>(
            std::make_unique<LateRejectMilbrandt>())
        : std::make_unique<tmv::MilbrandtScheme>();
    std::string error;
    const std::array<const Field3D*, 5> number_refs{
        &number[0], &number[1], &number[2], &number[3], &number[4]};
    const std::array<const Field3D*, 5> z_refs{
        &reflectivity[0], &reflectivity[1], &reflectivity[2],
        &reflectivity[3], &reflectivity[4]};
    if (!scheme->enable_internal_triple_sedimentation_for_test(
            rho, {&qr, &qi, &qs, &qg, &qh}, number_refs, z_refs, error))
    {
        throw std::runtime_error(error);
    }
    microphysics_scheme = std::move(scheme);
}

void initialize_warm_state(bool late_reject, bool coupled = false)
{
    NR = 2;
    NTH = 1;
    NZ = 1;
    dr = 2.0;
    dtheta = 3.0;
    dz = 5.0;
    dt = 0.05;
    simulation_time = 0.0;
    global_grid_geometry.initialize(
        NR, NTH, NZ, dr, dz, dtheta,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    global_gpu_apply_microphysics = false;
    global_validation_policy = {};
    global_validation_policy.mode = tmv::GuardMode::Strict;
    global_validation_policy.fail_on = tmv::GuardFailOn::Both;
    global_simulation_storage.configuration.microphysics_scheme = "milbrandt";
    global_simulation_storage.runtime.boundary_layer.config.scheme_id = "none";
    global_simulation_storage.runtime.chaos.config.scheme_id = "none";
    global_simulation_storage.runtime.terrain.config.scheme_id = "none";
    global_simulation_storage.runtime.terrain.earth_fixed_placement.active = false;
    global_simulation_storage.runtime.terrain.metrics = {};
    global_simulation_storage.runtime.compute.active_kind =
        ComputeBackendKind::Cpu;
    global_simulation_storage.runtime.compute.requested_kind =
        ComputeBackendKind::Cpu;
    global_simulation_storage.runtime.compute.fallback_active = false;
    const auto resize = [](Field3D& field, float value)
    {
        field.resize(NR, NTH, NZ, value);
    };
    for (Field3D* field : {&rho, &p, &u, &v, &w, &tracer, &theta,
                           &qv, &qc, &qr, &qi, &qs, &qg, &qh, &tke})
    {
        resize(*field, 0.0f);
    }
    for (Field3D* field : {&dtheta_dt_rad, &dtheta_dt_pbl, &dqv_dt_pbl,
                           &dqc_dt_pbl, &dqi_dt_pbl, &du_dt_pbl,
                           &dv_dt_pbl, &dtke_dt_pbl})
    {
        resize(*field, 0.0f);
    }
    global_simulation_storage.runtime.boundary_layer
        .held_u_force_n.release_storage();
    global_simulation_storage.runtime.boundary_layer
        .held_v_force_n.release_storage();
    global_simulation_storage.runtime.boundary_layer
        .held_theta_inventory_rate_kgk_s.release_storage();
    global_simulation_storage.runtime.boundary_layer
        .held_vapor_mass_rate_kg_s.release_storage();
    global_simulation_storage.runtime.boundary_layer
        .held_cloud_liquid_mass_rate_kg_s.release_storage();
    global_simulation_storage.runtime.boundary_layer
        .held_cloud_ice_mass_rate_kg_s.release_storage();
    global_simulation_storage.surface.precipitation_rate_mm_h.assign(2, 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_mm.assign(2, 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_residual_mm.assign(2, 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_last_update_s = 0.0;
    rho0_base.assign(1, 1.0);
    p0_base.assign(1, 90000.0);
    qv0_base.assign(1, 0.010);
    theta0_base.assign(1, 300.0);
    u0_base.assign(1, 0.0);
    v0_base.assign(1, 0.0);
    global_radiation_runtime_state.config.scheme_id = "none";
    global_radiation_runtime_state.held_atmospheric_power_w = 0.0;
    global_radiation_runtime_state.held_cell_power_w.release_storage();
    global_radiation_runtime_state.held_shortwave_incoming_power_w = 0.0;
    global_radiation_runtime_state.held_shortwave_atmospheric_absorbed_power_w = 0.0;
    global_radiation_runtime_state.held_surface_absorbed_power_w = 0.0;
    global_radiation_runtime_state.held_top_escape_power_w = 0.0;
    global_radiation_runtime_state.held_lateral_escape_power_w = 0.0;
    global_radiation_runtime_state.held_unresolved_reflected_power_w = 0.0;
    global_simulation_storage.runtime.chaos.pbl_base_tendencies_valid = false;
    global_simulation_storage.runtime.chaos.base_du_dt_pbl.release_storage();
    global_simulation_storage.runtime.chaos.base_dv_dt_pbl.release_storage();
    global_simulation_storage.runtime.chaos.base_dtheta_dt_pbl.release_storage();
    global_simulation_storage.runtime.chaos.base_dqv_dt_pbl.release_storage();
    global_simulation_storage.runtime.chaos.base_dtke_dt_pbl.release_storage();
    global_microphysics_runtime_state.theta_storage_residual = {};
    global_microphysics_runtime_state.last_water_repair_report = {};
    global_microphysics_runtime_state.last_thermodynamic_repair_report = {};
    global_microphysics_runtime_state.last_conservation_report = {};

    Field3D cloud_number(NR, NTH, NZ, 1.0e5f);
    Field3D rain_number(NR, NTH, NZ, 1.0e3f);
    Field3D rain_z(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
    {
        const float pressure = i == 0 ? 90000.0f : 70000.0f;
        rho(i, 0, 0) = 1.0f;
        p(i, 0, 0) = pressure;
        theta(i, 0, 0) = static_cast<float>(
            thermodynamics::temperature_to_theta(280.0, pressure));
        qv(i, 0, 0) = static_cast<float>(
            0.5 * thermodynamics::saturation_mixing_ratio_water(
                280.0, pressure));
        qc(i, 0, 0) = 0.001f;
        qr(i, 0, 0) = 1.0e-4f;
        rain_z(i, 0, 0) = reference_z(
            qr(i, 0, 0), rain_number(i, 0, 0),
            523.598775598, 3.0, 20.0);
    }
    std::unique_ptr<tmv::MilbrandtScheme> scheme = late_reject
        ? std::unique_ptr<tmv::MilbrandtScheme>(
            std::make_unique<LateRejectMilbrandt>())
        : std::make_unique<tmv::MilbrandtScheme>();
    std::string error;
    auto state = global_simulation_state();
    const bool activated = coupled
        ? scheme->enable_internal_triple_warm_rain_sedimentation_for_test(
              state, cloud_number, rain_number, rain_z, error)
        : scheme->enable_internal_triple_warm_for_test(
              state, cloud_number, rain_number, rain_z, error);
    if (!activated)
    {
        throw std::runtime_error(error);
    }
    microphysics_scheme = std::move(scheme);
}

} // namespace

TEST_CASE("internal five-Z reaches direct accepted-primary and surface commit",
          "[physics][milbrandt][triple_internal][direct]")
{
    std::array<std::array<Field3D, 5>, 2> mass_after;
    std::array<std::array<Field3D, 5>, 2> number_after;
    std::array<std::array<Field3D, 5>, 2> z_after;
    std::array<std::vector<float>, 2> rainfall_after;
    for (int run = 0; run < 2; ++run)
    {
#ifdef _OPENMP
        omp_set_num_threads(run == 0 ? 1 : 4);
#endif
        initialize_state(false);
        const auto& initial = dynamic_cast<const tmv::MilbrandtScheme&>(
            *microphysics_scheme);
        const std::array<const Field3D*, 5> mass_initial{
            &qr, &qi, &qs, &qg, &qh};
        std::array<Field3D, 5> mass_before;
        std::array<Field3D, 5> number_before;
        std::array<Field3D, 5> z_before;
        for (std::size_t species = 0; species < 5; ++species)
        {
            mass_before[species] = *mass_initial[species];
            number_before[species] =
                initial.internal_triple_number_for_test(species);
            z_before[species] =
                initial.internal_triple_reflectivity_for_test(species);
        }
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 0.05));
        const auto& scheme = dynamic_cast<const tmv::MilbrandtScheme&>(
            *microphysics_scheme);
        const std::array<const Field3D*, 5> mass{
            &qr, &qi, &qs, &qg, &qh};
        for (std::size_t species = 0; species < 5; ++species)
        {
            mass_after[run][species] = *mass[species];
            number_after[run][species] =
                scheme.internal_triple_number_for_test(species);
            z_after[run][species] =
                scheme.internal_triple_reflectivity_for_test(species);
            const auto surface = scheme.internal_triple_surface_for_test(1, 0, species);
            CHECK(surface.mass > 0.0);
            CHECK(surface.number > 0.0);
            CHECK(surface.reflectivity > 0.0);
            for (int i = 0; i < NR; ++i)
            {
                const auto local_surface =
                    scheme.internal_triple_surface_for_test(i, 0, species);
                const std::array<const Field3D*, 3> before{
                    &mass_before[species], &number_before[species],
                    &z_before[species]};
                const std::array<const Field3D*, 3> after{
                    mass[species],
                    &scheme.internal_triple_number_for_test(species),
                    &scheme.internal_triple_reflectivity_for_test(species)};
                const std::array<double, 3> surface_values{
                    local_surface.mass, local_surface.number,
                    local_surface.reflectivity};
                for (std::size_t moment = 0; moment < 3; ++moment)
                {
                    double initial_inventory = 0.0;
                    double final_inventory = 0.0;
                    for (int k = 0; k < NZ; ++k)
                    {
                        const double scale = rho(i, 0, k) * 5.0;
                        initial_inventory += scale * (*before[moment])(i, 0, k);
                        final_inventory += scale * (*after[moment])(i, 0, k);
                    }
                    const double residual = initial_inventory
                        - final_inventory - surface_values[moment];
                    const double scale = std::max(
                        {std::abs(initial_inventory),
                         std::abs(final_inventory),
                         std::abs(surface_values[moment])});
                    CHECK(std::abs(residual) <=
                          64.0 * std::numeric_limits<float>::epsilon()
                          * scale);
                }
            }
        }
        rainfall_after[run] =
            global_simulation_storage.surface.accumulated_rainfall_mm;
        CHECK(rainfall_after[run][0] > 0.0f);
        CHECK(rainfall_after[run][1] > 0.0f);
    }
    for (std::size_t species = 0; species < 5; ++species)
    {
        CHECK(same_bytes(mass_after[0][species], mass_after[1][species]));
        CHECK(same_bytes(number_after[0][species], number_after[1][species]));
        CHECK(same_bytes(z_after[0][species], z_after[1][species]));
    }
    CHECK(rainfall_after[0] == rainfall_after[1]);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

TEST_CASE("internal five-Z late accepted-primary failure rolls direct step back",
          "[physics][milbrandt][triple_internal][direct]")
{
    initialize_state(true);
    const std::array<Field3D*, 15> fields{
        &rho, &p, &u, &v, &w, &theta, &qv, &qc,
        &qr, &qi, &qs, &qg, &qh, &tracer, &tke};
    std::array<Field3D, 15> before;
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        before[index] = *fields[index];
    }
    const auto& original = dynamic_cast<const tmv::MilbrandtScheme&>(
        *microphysics_scheme);
    std::array<Field3D, 5> number_before;
    std::array<Field3D, 5> z_before;
    for (std::size_t species = 0; species < 5; ++species)
    {
        number_before[species] = original.internal_triple_number_for_test(species);
        z_before[species] = original.internal_triple_reflectivity_for_test(species);
    }
    const auto rate_before =
        global_simulation_storage.surface.precipitation_rate_mm_h;
    const auto rainfall_before =
        global_simulation_storage.surface.accumulated_rainfall_mm;
    const auto residual_before =
        global_simulation_storage.surface.accumulated_rainfall_residual_mm;
    const double time_before =
        global_simulation_storage.surface.accumulated_rainfall_last_update_s;
    const auto water_before =
        global_microphysics_runtime_state.last_water_repair_report;
    const auto heat_before =
        global_microphysics_runtime_state.last_thermodynamic_repair_report;
    const auto conservation_before =
        global_microphysics_runtime_state.last_conservation_report;

    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 0.05),
        "post-primary microphysics private candidate rejected: "
        "injected after five-Z accepted-primary validation");
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        CHECK(same_bytes(*fields[index], before[index]));
    }
    const auto& restored = dynamic_cast<const tmv::MilbrandtScheme&>(
        *microphysics_scheme);
    for (std::size_t species = 0; species < 5; ++species)
    {
        CHECK(same_bytes(restored.internal_triple_number_for_test(species),
                         number_before[species]));
        CHECK(same_bytes(restored.internal_triple_reflectivity_for_test(species),
                         z_before[species]));
    }
    CHECK(global_simulation_storage.surface.precipitation_rate_mm_h == rate_before);
    CHECK(global_simulation_storage.surface.accumulated_rainfall_mm == rainfall_before);
    CHECK(global_simulation_storage.surface.accumulated_rainfall_residual_mm == residual_before);
    CHECK(global_simulation_storage.surface.accumulated_rainfall_last_update_s == time_before);
    CHECK(global_microphysics_runtime_state.last_water_repair_report.nonfinite_count()
          == water_before.nonfinite_count());
    CHECK(global_microphysics_runtime_state.last_thermodynamic_repair_report
              .moist_committed_energy_change_j
          == heat_before.moist_committed_energy_change_j);
    CHECK(global_microphysics_runtime_state.last_conservation_report
              .surface_precipitation_kg
          == conservation_before.surface_precipitation_kg);
}

TEST_CASE("internal WarmOnly direct step closes water and accepted rain moments",
          "[physics][milbrandt][triple_warm_internal][direct]")
{
    std::array<std::array<Field3D, 7>, 2> after;
    std::array<std::vector<float>, 2> rainfall;
    for (int run = 0; run < 2; ++run)
    {
#ifdef _OPENMP
        omp_set_num_threads(run == 0 ? 1 : 4);
#endif
        initialize_warm_state(false);
        const Field3D qv_before = qv;
        const Field3D qc_before = qc;
        const Field3D qr_before = qr;
        const Field3D theta_before = theta;
        const auto& initial_scheme =
            dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
        const Field3D nc_before =
            initial_scheme.internal_triple_cloud_number_for_test();
        const Field3D nr_before =
            initial_scheme.internal_triple_number_for_test(0);
        const Field3D z_before =
            initial_scheme.internal_triple_reflectivity_for_test(0);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 0.05));
        const auto& scheme =
            dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
        for (int i = 0; i < NR; ++i)
        {
            const double pressure = p(i, 0, 0);
            const double temperature =
                thermodynamics::theta_to_temperature(
                    theta_before(i, 0, 0), pressure);
            const tmv::milbrandt_detail::WarmRainCell cell{
                static_cast<float>(temperature), p(i, 0, 0),
                rho(i, 0, 0), qv_before(i, 0, 0),
                qc_before(i, 0, 0), qr_before(i, 0, 0),
                nc_before(i, 0, 0), nr_before(i, 0, 0)};
            const tmv::milbrandt_detail::RainTripleMomentState initial{
                qr_before(i, 0, 0), nr_before(i, 0, 0),
                z_before(i, 0, 0)};
            tmv::milbrandt_detail::WarmRainTendencies rates{};
            tmv::milbrandt_detail::RainTripleMomentState rain{};
            std::string error;
            REQUIRE(tmv::milbrandt_detail::evaluate_triple_only_warm_cell(
                tmv::milbrandt_detail::warm_rain_parameters(
                    0.0, 523.598775598, 3.0, 149.1, 0.5),
                cell, 0.05, {}, initial, rates, rain, error));
            tmv::milbrandt_detail::TripleWarmRainResult produced;
            tmv::milbrandt_detail::triple_warm_rain_cell(
                tmv::milbrandt_detail::warm_rain_parameters(
                    0.0, 523.598775598, 3.0, 149.1, 0.5),
                cell, 0.05, {}, produced);
            double evaporation_rate = 0.0;
            for (std::size_t event = 0;
                 event < produced.events().size; ++event)
            {
                if (produced.events().events[event].kind ==
                    tmv::milbrandt_detail::WarmRainAcceptedEventKind::Evaporation)
                {
                    evaporation_rate += produced.events().events[event]
                        .rain_mass_rate_per_kg_s;
                }
            }
            REQUIRE(evaporation_rate < 0.0);
            const double expected_temperature_rate =
                microphysics_constants::L_v /
                    microphysics_constants::cp * evaporation_rate;
            CHECK(rates.temperature == Approx(expected_temperature_rate)
                .epsilon(2.0e-6));
            CHECK(qv(i, 0, 0) == Approx(
                static_cast<double>(qv_before(i, 0, 0))
                + 0.05 * rates.vapor).epsilon(2.0e-6));
            CHECK(qc(i, 0, 0) == Approx(
                static_cast<double>(qc_before(i, 0, 0))
                + 0.05 * rates.cloud).epsilon(2.0e-6));
            CHECK(qr(i, 0, 0) == Approx(rain.mass_per_kg)
                .epsilon(2.0e-6));
            CHECK(scheme.internal_triple_number_for_test(0)(i, 0, 0)
                  == rain.number_per_kg);
            CHECK(scheme.internal_triple_reflectivity_for_test(0)(i, 0, 0)
                  == rain.reflectivity_per_kg);
            const double expected_theta =
                static_cast<double>(theta_before(i, 0, 0))
                + 0.05 * expected_temperature_rate *
                    std::pow(microphysics_constants::p0 / pressure,
                             microphysics_constants::R_d /
                                 microphysics_constants::cp);
            CHECK(theta(i, 0, 0) == Approx(expected_theta)
                .epsilon(2.0e-6));
            const double old_water = qv_before(i, 0, 0) +
                qc_before(i, 0, 0) + qr_before(i, 0, 0);
            const double new_water = qv(i, 0, 0) +
                qc(i, 0, 0) + qr(i, 0, 0);
            CHECK(old_water == Approx(new_water).epsilon(2.0e-6));
            CHECK(scheme.internal_triple_cloud_number_for_test()(i, 0, 0)
                  >= 0.0f);
            CHECK(scheme.internal_triple_surface_for_test(i, 0, 0).mass
                  == 0.0);
        }
        CHECK(global_microphysics_runtime_state.last_water_repair_report
                  .lower_bound_count() == 0);
        CHECK(global_microphysics_runtime_state.last_water_repair_report
                  .upper_bound_count() == 0);
        CHECK(global_microphysics_runtime_state.last_thermodynamic_repair_report
                  .theta_lower_bound_count == 0);
        after[run] = {qv, qc, qr, theta,
                      scheme.internal_triple_cloud_number_for_test(),
                      scheme.internal_triple_number_for_test(0),
                      scheme.internal_triple_reflectivity_for_test(0)};
        rainfall[run] =
            global_simulation_storage.surface.accumulated_rainfall_mm;
        CHECK(rainfall[run].size() == 2);
        CHECK(rainfall[run][0] == 0.0f);
        CHECK(rainfall[run][1] == 0.0f);
    }
    for (std::size_t field = 0; field < after[0].size(); ++field)
    {
        CHECK(same_bytes(after[0][field], after[1][field]));
    }
    CHECK(rainfall[0] == rainfall[1]);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

TEST_CASE("internal WarmOnly late accepted-primary rejection rolls all owners back",
          "[physics][milbrandt][triple_warm_internal][direct]")
{
    initialize_warm_state(true);
    const std::array<Field3D*, 15> fields{
        &rho, &p, &u, &v, &w, &theta, &qv, &qc,
        &qr, &qi, &qs, &qg, &qh, &tracer, &tke};
    std::array<Field3D, 15> before;
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        before[index] = *fields[index];
    }
    const auto& initial =
        dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
    const Field3D nc_before =
        initial.internal_triple_cloud_number_for_test();
    const Field3D nr_before =
        initial.internal_triple_number_for_test(0);
    const Field3D z_before =
        initial.internal_triple_reflectivity_for_test(0);
    const auto rate_before =
        global_simulation_storage.surface.precipitation_rate_mm_h;
    const auto rainfall_before =
        global_simulation_storage.surface.accumulated_rainfall_mm;
    const auto residual_before =
        global_simulation_storage.surface.accumulated_rainfall_residual_mm;
    const double time_before =
        global_simulation_storage.surface.accumulated_rainfall_last_update_s;
    const auto water_before =
        global_microphysics_runtime_state.last_water_repair_report;
    const auto heat_before =
        global_microphysics_runtime_state.last_thermodynamic_repair_report;
    const auto conservation_before =
        global_microphysics_runtime_state.last_conservation_report;
    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 0.05),
        "post-primary microphysics private candidate rejected: "
        "injected after five-Z accepted-primary validation");
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        CHECK(same_bytes(*fields[index], before[index]));
    }
    const auto& restored =
        dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
    CHECK(same_bytes(restored.internal_triple_cloud_number_for_test(),
                     nc_before));
    CHECK(same_bytes(restored.internal_triple_number_for_test(0),
                     nr_before));
    CHECK(same_bytes(restored.internal_triple_reflectivity_for_test(0),
                     z_before));
    CHECK(global_simulation_storage.surface.precipitation_rate_mm_h
          == rate_before);
    CHECK(global_simulation_storage.surface.accumulated_rainfall_mm
          == rainfall_before);
    CHECK(global_simulation_storage.surface.accumulated_rainfall_residual_mm
          == residual_before);
    CHECK(global_simulation_storage.surface.accumulated_rainfall_last_update_s
          == time_before);
    CHECK(global_microphysics_runtime_state.last_water_repair_report
              .lower_bound_count() == water_before.lower_bound_count());
    CHECK(global_microphysics_runtime_state.last_thermodynamic_repair_report
              .moist_committed_energy_change_j
          == heat_before.moist_committed_energy_change_j);
    CHECK(global_microphysics_runtime_state.last_conservation_report
              .surface_precipitation_kg
          == conservation_before.surface_precipitation_kg);
}

TEST_CASE("internal warm-rain sedimentation direct step closes coupled surface moments",
          "[physics][milbrandt][triple_rain_coupled][direct]")
{
    std::array<std::array<Field3D, 7>, 2> after;
    std::array<std::vector<float>, 2> rainfall;
    std::array<std::vector<float>, 2> rain_rate;
    for (int run = 0; run < 2; ++run)
    {
#ifdef _OPENMP
        omp_set_num_threads(run == 0 ? 1 : 4);
#endif
        initialize_warm_state(false, true);
        const Field3D qv_before(qv);
        const Field3D qc_before(qc);
        const Field3D qr_before(qr);
        const Field3D theta_before(theta);
        const auto& initial_scheme =
            dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
        const Field3D nc_before(
            initial_scheme.internal_triple_cloud_number_for_test());
        const Field3D nr_before(
            initial_scheme.internal_triple_number_for_test(0));
        const Field3D z_before(
            initial_scheme.internal_triple_reflectivity_for_test(0));
        CHECK(initial_scheme.explicit_terminal_velocity_mps(
                  0, 1.0e-4, 1.0) == 16.0);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 0.05));
        const auto& scheme =
            dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
        for (int i = 0; i < NR; ++i)
        {
            const double pressure = p(i, 0, 0);
            const double temperature =
                thermodynamics::theta_to_temperature(
                    theta_before(i, 0, 0), pressure);
            const tmv::milbrandt_detail::WarmRainCell cell{
                static_cast<float>(temperature), p(i, 0, 0),
                rho(i, 0, 0), qv_before(i, 0, 0),
                qc_before(i, 0, 0), qr_before(i, 0, 0),
                nc_before(i, 0, 0), nr_before(i, 0, 0)};
            const tmv::milbrandt_detail::RainTripleMomentState initial{
                qr_before(i, 0, 0), nr_before(i, 0, 0),
                z_before(i, 0, 0)};
            tmv::milbrandt_detail::WarmRainTendencies warm_rates{};
            tmv::milbrandt_detail::RainTripleMomentState warm{};
            std::string error;
            REQUIRE(tmv::milbrandt_detail::evaluate_triple_only_warm_cell(
                tmv::milbrandt_detail::warm_rain_parameters(
                    0.0, 523.598775598, 3.0, 149.1, 0.5),
                cell, 0.05, {}, initial,
                warm_rates, warm, error));
            const float warm_q = static_cast<float>(
                static_cast<double>(qr_before(i, 0, 0))
                + 0.05 * warm_rates.rain);
            const float warm_n = warm.number_per_kg;
            const float warm_z = warm.reflectivity_per_kg;
            const float live_density = rho(i, 0, 0);
            const double thickness = 5.0;
            float expected_q = 0.0f;
            float expected_n = 0.0f;
            float expected_z = 0.0f;
            tmv::milbrandt_detail::TripleMomentSurfaceInventory surface;
            REQUIRE(tmv::milbrandt_detail::sediment_triple_moment_column(
                1, &warm_q, &warm_n, &warm_z, &live_density,
                &thickness, 0.05,
                tmv::milbrandt_detail::current_nimbus_triple_sedimentation_parameters(
                    tmv::milbrandt_detail::TripleMomentSpecies::Rain),
                &expected_q, &expected_n, &expected_z,
                surface, error));
            CHECK(qr(i, 0, 0) == Approx(expected_q).epsilon(2.0e-6));
            CHECK(scheme.internal_triple_number_for_test(0)(i, 0, 0)
                  == expected_n);
            CHECK(scheme.internal_triple_reflectivity_for_test(0)(i, 0, 0)
                  == expected_z);
            const auto committed =
                scheme.internal_triple_surface_for_test(i, 0, 0);
            CHECK(committed.mass == surface.mass);
            CHECK(committed.number == surface.number);
            CHECK(committed.reflectivity == surface.reflectivity);
            const double old_water = qv_before(i, 0, 0)
                + qc_before(i, 0, 0) + qr_before(i, 0, 0);
            const double new_water = qv(i, 0, 0)
                + qc(i, 0, 0) + qr(i, 0, 0);
            const double residual =
                live_density * thickness * (old_water - new_water)
                - surface.mass;
            CHECK(std::abs(residual) <=
                  64.0 * std::numeric_limits<float>::epsilon()
                  * live_density * thickness * old_water);
            CHECK(std::abs(live_density * thickness *
                           (warm_n - expected_n) - surface.number)
                  <= 64.0 * std::numeric_limits<float>::epsilon()
                     * live_density * thickness * warm_n);
            CHECK(std::abs(live_density * thickness *
                           (warm_z - expected_z) - surface.reflectivity)
                  <= 64.0 * std::numeric_limits<float>::epsilon()
                     * live_density * thickness * warm_z);
            const double expected_theta =
                static_cast<double>(theta_before(i, 0, 0))
                + 0.05 * warm_rates.temperature *
                    std::pow(microphysics_constants::p0 / pressure,
                             microphysics_constants::R_d /
                                 microphysics_constants::cp);
            CHECK(theta(i, 0, 0) == Approx(expected_theta)
                .epsilon(2.0e-6));
            CHECK(global_simulation_storage.surface
                      .accumulated_rainfall_mm[static_cast<std::size_t>(i)]
                  == Approx(surface.mass).epsilon(2.0e-6));
        }
        after[run] = {qv, qc, qr, theta,
                      scheme.internal_triple_cloud_number_for_test(),
                      scheme.internal_triple_number_for_test(0),
                      scheme.internal_triple_reflectivity_for_test(0)};
        rainfall[run] =
            global_simulation_storage.surface.accumulated_rainfall_mm;
        rain_rate[run] =
            global_simulation_storage.surface.precipitation_rate_mm_h;
    }
    for (std::size_t field = 0; field < after[0].size(); ++field)
    {
        CHECK(same_bytes(after[0][field], after[1][field]));
    }
    CHECK(rainfall[0] == rainfall[1]);
    CHECK(rain_rate[0] == rain_rate[1]);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

TEST_CASE("internal warm-rain sedimentation late hook failure restores all owners",
          "[physics][milbrandt][triple_rain_coupled][direct]")
{
    initialize_warm_state(true, true);
    const std::array<Field3D*, 15> fields{
        &rho, &p, &u, &v, &w, &theta, &qv, &qc,
        &qr, &qi, &qs, &qg, &qh, &tracer, &tke};
    std::array<Field3D, 15> before;
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        before[index] = *fields[index];
    }
    const auto& initial =
        dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
    const Field3D nc_before(
        initial.internal_triple_cloud_number_for_test());
    const Field3D nr_before(
        initial.internal_triple_number_for_test(0));
    const Field3D z_before(
        initial.internal_triple_reflectivity_for_test(0));
    const auto precipitation_rate_before =
        global_simulation_storage.surface.precipitation_rate_mm_h;
    const auto rainfall_before =
        global_simulation_storage.surface.accumulated_rainfall_mm;
    const auto residual_before =
        global_simulation_storage.surface.accumulated_rainfall_residual_mm;
    const double precipitation_time_before =
        global_simulation_storage.surface.accumulated_rainfall_last_update_s;
    const auto water_before =
        global_microphysics_runtime_state.last_water_repair_report;
    const auto heat_before =
        global_microphysics_runtime_state.last_thermodynamic_repair_report;
    const auto conservation_before =
        global_microphysics_runtime_state.last_conservation_report;
    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 0.05),
        "post-primary microphysics private candidate rejected: "
        "injected after five-Z accepted-primary validation");
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        CHECK(same_bytes(*fields[index], before[index]));
    }
    const auto& restored =
        dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
    CHECK(same_bytes(restored.internal_triple_cloud_number_for_test(),
                     nc_before));
    CHECK(same_bytes(restored.internal_triple_number_for_test(0),
                     nr_before));
    CHECK(same_bytes(restored.internal_triple_reflectivity_for_test(0),
                     z_before));
    CHECK(global_simulation_storage.surface.precipitation_rate_mm_h
          == precipitation_rate_before);
    CHECK(global_simulation_storage.surface.accumulated_rainfall_mm
          == rainfall_before);
    CHECK(global_simulation_storage.surface.accumulated_rainfall_residual_mm
          == residual_before);
    CHECK(global_simulation_storage.surface.accumulated_rainfall_last_update_s
          == precipitation_time_before);
    CHECK(global_microphysics_runtime_state.last_water_repair_report
              .lower_bound_count() == water_before.lower_bound_count());
    CHECK(global_microphysics_runtime_state.last_thermodynamic_repair_report
              .moist_committed_energy_change_j
          == heat_before.moist_committed_energy_change_j);
    CHECK(global_microphysics_runtime_state.last_conservation_report
              .surface_precipitation_kg
          == conservation_before.surface_precipitation_kg);
}

TEST_CASE("internal rain coupling uses the guarded outer 0.8 dz over 16 schedule",
          "[physics][milbrandt][triple_rain_coupled][direct]")
{
    initialize_warm_state(false, true);
    tmv::reset_microphysics_perf_stats();
    const double expected_limit = 0.8 * dz / 16.0;
    REQUIRE(expected_limit == Approx(0.25));
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 0.5));
    CHECK(tmv::microphysics_perf_snapshot().sedimentation_substeps == 2);
    CHECK(simulation_time == 0.0);
    const auto& continuous_scheme =
        dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
    const std::array<Field3D, 7> continuous{
        qv, qc, qr, theta,
        continuous_scheme.internal_triple_cloud_number_for_test(),
        continuous_scheme.internal_triple_number_for_test(0),
        continuous_scheme.internal_triple_reflectivity_for_test(0)};
    const auto continuous_rainfall =
        global_simulation_storage.surface.accumulated_rainfall_mm;
    const auto continuous_residual =
        global_simulation_storage.surface.accumulated_rainfall_residual_mm;
    CHECK(global_simulation_storage.surface.accumulated_rainfall_last_update_s
          == 0.5);

    initialize_warm_state(false, true);
    tmv::reset_microphysics_perf_stats();
    REQUIRE_NOTHROW(step_microphysics(
        global_simulation_state(), expected_limit));
    CHECK(tmv::microphysics_perf_snapshot().sedimentation_substeps == 1);
    simulation_time = expected_limit;
    REQUIRE_NOTHROW(step_microphysics(
        global_simulation_state(), 0.5 - expected_limit));
    CHECK(tmv::microphysics_perf_snapshot().sedimentation_substeps == 2);
    CHECK(global_simulation_storage.surface.accumulated_rainfall_last_update_s
          == 0.5);
    const auto& resumed_scheme =
        dynamic_cast<const tmv::MilbrandtScheme&>(*microphysics_scheme);
    const std::array<Field3D, 7> matched_schedule{
        qv, qc, qr, theta,
        resumed_scheme.internal_triple_cloud_number_for_test(),
        resumed_scheme.internal_triple_number_for_test(0),
        resumed_scheme.internal_triple_reflectivity_for_test(0)};
    for (std::size_t field = 0; field < continuous.size(); ++field)
    {
        CHECK(same_bytes(continuous[field], matched_schedule[field]));
    }
    CHECK(global_simulation_storage.surface.accumulated_rainfall_mm
          == continuous_rainfall);
    CHECK(global_simulation_storage.surface.accumulated_rainfall_residual_mm
          == continuous_residual);
}
