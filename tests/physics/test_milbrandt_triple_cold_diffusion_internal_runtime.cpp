#include "catch2/catch.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "microphysics/runtime.hpp"
#include "microphysics/schemes/milbrandt/milbrandt.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

using Species = tmv::milbrandt_detail::TripleMomentSpecies;
constexpr double kDuration = 0.05;
constexpr const char* kRejectedHook = "injected after live diffusion accepted-primary validation";

struct ReferenceLaw
{
    long double mass;
    long double fall;
    long double exponent;
};

ReferenceLaw reference_law(Species species)
{
    return species == Species::Graupel
        ? ReferenceLaw{209.439510239L, 19.3L, .37L}
        : ReferenceLaw{471.238898038L, 206.89L, .6384L};
}

std::size_t species_index(Species species)
{
    return static_cast<std::size_t>(species);
}

Field3D& selected_mass(Species species)
{
    return species == Species::Graupel ? qg : qh;
}

long double shape_ratio(long double alpha)
{
    return (alpha + 4) * (alpha + 5) * (alpha + 6)
        / ((alpha + 1) * (alpha + 2) * (alpha + 3));
}

float reference_z(float q, float n, Species species, long double alpha)
{
    const long double c = reference_law(species).mass;
    return static_cast<float>(shape_ratio(alpha) * q * q / (c * c * n));
}

long double reference_shape(long double q, long double n, long double z,
                            Species species)
{
    const long double c = reference_law(species).mass;
    const long double ratio = c * c * n * z / (q * q);
    REQUIRE(ratio > 1.0L);
    long double lower = -1.0L + 1e-9L;
    long double upper = 1e6L;
    for (int iteration = 0; iteration < 120; ++iteration)
    {
        const long double middle = (lower + upper) / 2;
        if (shape_ratio(middle) > ratio)
            lower = middle;
        else
            upper = middle;
    }
    return (lower + upper) / 2;
}

long double reference_qsi(long double temperature, long double pressure)
{
    const long double tc = temperature - microphysics_constants::T0;
    const long double vapor_pressure = 611.15L
        * std::exp((23.036L - tc / 333.7L) * tc / (279.82L + tc));
    return .62198L * vapor_pressure / (pressure - vapor_pressure);
}

long double numerical_gamma(long double order)
{
    constexpr int intervals = 32768;
    constexpr long double end = 160.0L;
    constexpr long double width = end / intervals;
    long double sum = 0;
    for (int index = 1; index < intervals; ++index)
    {
        const long double x = index * width;
        sum += (index % 2 ? 4.0L : 2.0L) * std::pow(x, order) * std::exp(-x);
    }
    return (sum + std::pow(end, order) * std::exp(-end)) * width / 3;
}

long double reference_increment(long double temperature, long double pressure,
                                long double density, long double vapor,
                                long double q, long double n,
                                long double alpha, Species species,
                                long double duration)
{
    if (q == 0 && n == 0)
        return 0;
    const auto law = reference_law(species);
    const long double tc = temperature - microphysics_constants::T0;
    const long double qsi = reference_qsi(temperature, pressure);
    const long double diffusivity = (2.2157e-5L + .0155e-5L * tc) * 1e5L / pressure;
    const long double conductivity = 2.3971e-2L + .0078e-2L * tc;
    const long double viscosity = 1.72e-5L * 393.0L / (temperature + 120.0L)
        * std::pow(temperature / microphysics_constants::T0, 1.5L) / density;
    const long double lambda = std::cbrt(law.mass * n
        * (alpha + 1) * (alpha + 2) * (alpha + 3) / q);
    const long double intercept = density * n * std::pow(lambda, alpha + 1)
        / std::tgamma(alpha + 1);
    const long double ventilation = .78L * numerical_gamma(alpha + 1)
        / std::pow(lambda, alpha + 2)
        + .30L * std::cbrt(viscosity / diffusivity)
        * std::sqrt(std::sqrt(1.225L / density) * law.fall / viscosity)
        * numerical_gamma(alpha + 1.5L + .5L * law.exponent)
        / std::pow(lambda, alpha + 2.5L + .5L * law.exponent);
    const long double resistance = microphysics_constants::L_s * microphysics_constants::L_s
        / (conductivity * microphysics_constants::R_v * temperature * temperature)
        + 1 / (density * qsi * diffusivity);
    long double increment = duration / density * 2 * std::acos(-1.0L)
        * (vapor / qsi - 1) * intercept * ventilation / resistance;
    if (vapor >= qsi)
    {
        const long double limit = (vapor - qsi) / (1
            + microphysics_constants::L_s * microphysics_constants::L_s * qsi
            / (microphysics_constants::cp * microphysics_constants::R_v
               * (temperature - 7.66L) * (temperature - 7.66L)));
        return std::clamp(increment, 0.0L, std::max(limit, 0.0L));
    }
    return std::clamp(increment, -q, 0.0L);
}

bool same_bytes(const Field3D& a, const Field3D& b)
{
    return a.size_r() == b.size_r() && a.size_th() == b.size_th()
        && a.size_z() == b.size_z() && a.size() == b.size()
        && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

class ObservedMilbrandt final : public tmv::MilbrandtScheme
{
public:
    bool reject = false;
    bool accepted_called = false;
    bool surface_committed = false;

    std::unique_ptr<MicrophysicsScheme> clone() const override
    {
        return std::make_unique<ObservedMilbrandt>(*this);
    }

    bool prepare_post_primary_private_commit(
        double duration,
        const tmv::MicrophysicsAcceptedPrimaryFields& accepted,
        std::string& error) override
    {
        if (!MilbrandtScheme::prepare_post_primary_private_commit(duration, accepted, error))
            return false;
        accepted_called = true;
        if (reject)
        {
            error = kRejectedHook;
            return false;
        }
        return true;
    }

    bool commit_prepared_private_candidate() noexcept override
    {
        if (!MilbrandtScheme::commit_prepared_private_candidate())
            return false;
        surface_committed = true;
        return true;
    }
};

std::array<Field3D*, 15> public_fields()
{
    return {&rho, &p, &u, &v, &w, &theta, &qv, &qc,
            &qr, &qi, &qs, &qg, &qh, &tracer, &tke};
}

std::array<Field3D*, 8> source_fields()
{
    return {&dtheta_dt_rad, &dtheta_dt_pbl, &dqv_dt_pbl, &dqc_dt_pbl,
            &dqi_dt_pbl, &du_dt_pbl, &dv_dt_pbl, &dtke_dt_pbl};
}

std::array<Field3D*, 12> held_fields()
{
    auto& b = global_simulation_storage.runtime.boundary_layer;
    auto& r = global_simulation_storage.runtime.radiation;
    auto& c = global_simulation_storage.runtime.chaos;
    return {&b.held_u_force_n, &b.held_v_force_n,
        &b.held_theta_inventory_rate_kgk_s, &b.held_vapor_mass_rate_kg_s,
        &b.held_cloud_liquid_mass_rate_kg_s, &b.held_cloud_ice_mass_rate_kg_s,
        &r.held_cell_power_w, &c.base_du_dt_pbl, &c.base_dv_dt_pbl,
        &c.base_dtheta_dt_pbl, &c.base_dqv_dt_pbl, &c.base_dtke_dt_pbl};
}

std::array<double*, 7> held_radiation_powers()
{
    auto& r = global_simulation_storage.runtime.radiation;
    return {&r.held_atmospheric_power_w, &r.held_shortwave_incoming_power_w,
        &r.held_shortwave_atmospheric_absorbed_power_w, &r.held_surface_absorbed_power_w,
        &r.held_top_escape_power_w, &r.held_lateral_escape_power_w,
        &r.held_unresolved_reflected_power_w};
}

void initialize_state(Species species, long double alpha = 2,
                      double saturation = .75, float mass = 5e-4f,
                      float number = 10.0f)
{
    NR = 6;
    NTH = NZ = 1;
    dr = 2;
    dtheta = 3;
    dz = 5;
    dt = kDuration;
    simulation_time = 0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    global_gpu_apply_microphysics = false;
    global_validation_policy = {};
    global_validation_policy.mode = tmv::GuardMode::Strict;
    global_validation_policy.fail_on = tmv::GuardFailOn::Both;
    global_simulation_storage.configuration.microphysics_scheme = "milbrandt";
    global_simulation_storage.configuration.microphysics_config = {};
    global_simulation_storage.configuration.domain_frame = {};
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
    global_microphysics_runtime_state.tendencies = {};
    global_microphysics_runtime_state.theta_storage_residual = {};
    global_microphysics_runtime_state.last_water_repair_report = {};
    global_microphysics_runtime_state.last_thermodynamic_repair_report = {};
    global_microphysics_runtime_state.last_conservation_report = {};
    for (Field3D* field : public_fields())
        field->resize(NR, NTH, NZ, 0.0f);
    for (Field3D* field : source_fields())
        field->resize(NR, NTH, NZ, 0.0f);
    global_simulation_storage.surface.precipitation_rate_mm_h.assign(NR, 0);
    global_simulation_storage.surface.accumulated_rainfall_mm = {.25f, .5f, .75f, 1.0f, 1.25f, 1.5f};
    global_simulation_storage.surface.accumulated_rainfall_residual_mm.assign(NR, 0);
    global_simulation_storage.surface.accumulated_rainfall_last_update_s = 0;
    rho0_base.assign(1, 1);
    p0_base.assign(1, 90000);
    qv0_base.assign(1, .001);
    theta0_base.assign(1, 300);
    u0_base.assign(1, 0);
    v0_base.assign(1, 0);
    Field3D supplied_n(NR, NTH, NZ, number);
    Field3D supplied_z(NR, NTH, NZ, mass == 0 && number == 0
        ? 0.0f : reference_z(mass, number, species, alpha));
    for (int i = 0; i < NR; ++i)
    {
        rho(i, 0, 0) = i % 2 ? .65f : 1.3f;
        p(i, 0, 0) = i < 3 ? 90000.0f : 70000.0f;
        theta(i, 0, 0) = static_cast<float>(
            thermodynamics::temperature_to_theta(253.15, p(i, 0, 0)));
        const double temperature = thermodynamics::theta_to_temperature(
            theta(i, 0, 0), p(i, 0, 0));
        qv(i, 0, 0) = static_cast<float>(saturation
            * thermodynamics::saturation_mixing_ratio_ice(temperature, p(i, 0, 0)));
        selected_mass(species)(i, 0, 0) = mass;
    }
    auto scheme = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig configuration;
    configuration.milbrandt.hail_processes = true;
    scheme->configure(configuration);
    std::string error;
    INFO(error);
    REQUIRE(scheme->enable_internal_triple_frozen_diffusion_for_test(
        global_simulation_state(), species, supplied_n, supplied_z, error));
    microphysics_scheme = std::move(scheme);
}

struct Snapshot
{
    std::array<Field3D, 15> fields;
    std::array<Field3D, 11> moments;
    std::array<Field3D, 8> tendencies;
    std::array<Field3D, 8> sources;
    std::array<Field3D, 12> held;
    std::array<double, 7> radiation_powers;
    double noise_elapsed;
    bool pbl_base_valid;
    Field3D theta_residual;
    tmv::MicrophysicsWaterRepairReport water;
    tmv::MicrophysicsThermodynamicRepairReport heat;
    tmv::MicrophysicsConservationReport conservation;
    bool surface_committed;
    std::vector<std::array<tmv::milbrandt_detail::TripleMomentSurfaceInventory, 5>> surface;
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
        const auto source_owners = source_fields();
        const auto held_owners = held_fields();
        const auto power_owners = held_radiation_powers();
        const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
        for (std::size_t index = 0; index < fields.size(); ++index)
            fields[index] = *owners[index];
        for (std::size_t index = 0; index < sources.size(); ++index)
            sources[index] = *source_owners[index];
        for (std::size_t index = 0; index < held.size(); ++index)
            held[index] = *held_owners[index];
        for (std::size_t index = 0; index < radiation_powers.size(); ++index)
            radiation_powers[index] = *power_owners[index];
        noise_elapsed = global_simulation_storage.runtime.chaos.noise_elapsed_s;
        pbl_base_valid = global_simulation_storage.runtime.chaos.pbl_base_tendencies_valid;
        moments[0] = scheme.internal_triple_cloud_number_for_test();
        for (std::size_t species = 0; species < 5; ++species)
        {
            moments[1 + species] = scheme.internal_triple_number_for_test(species);
            moments[6 + species] = scheme.internal_triple_reflectivity_for_test(species);
        }
        const auto& t = global_microphysics_runtime_state.tendencies;
        tendencies = {t.dtheta_dt, t.dqv_dt, t.dqc_dt, t.dqr_dt,
                      t.dqi_dt, t.dqs_dt, t.dqg_dt, t.dqh_dt};
        surface_committed = scheme.surface_committed;
        if (surface_committed)
        {
            surface.resize(NR * NTH);
            for (int i = 0; i < NR; ++i)
                for (int j = 0; j < NTH; ++j)
                    for (std::size_t species = 0; species < 5; ++species)
                        surface[i * NTH + j][species] = scheme.internal_triple_surface_for_test(i, j, species);
        }
    }
};

auto thermal_values(const tmv::MicrophysicsThermodynamicRepairReport& r)
{
    return std::tie(r.microphysics_heat_j, r.radiation_heat_j, r.boundary_layer_heat_j,
        r.theta_tendency_limit_correction_j, r.theta_nonfinite_correction_j,
        r.theta_storage_rounding_j, r.theta_storage_residual_release_j,
        r.theta_storage_residual_j, r.theta_bound_correction_j,
        r.microphysics_water_tendency_kg, r.microphysics_water_tendency_kg_by_species,
        r.boundary_layer_vapor_tendency_kg, r.boundary_layer_cloud_liquid_tendency_kg,
        r.boundary_layer_cloud_ice_tendency_kg, r.moist_dry_internal_change_j,
        r.moist_water_sensible_change_j, r.moist_phase_reference_change_j,
        r.moist_carrier_kinetic_change_j, r.moist_gravitational_change_j,
        r.moist_subgrid_tke_change_j, r.moist_committed_energy_change_j,
        r.moist_energy_closure_residual_j, r.moist_energy_complete,
        r.theta_tendency_limit_count, r.theta_nonfinite_count,
        r.theta_lower_bound_count, r.theta_upper_bound_count);
}

void check_snapshot(const Snapshot& expected, const Snapshot& actual = Snapshot())
{
    for (std::size_t index = 0; index < expected.fields.size(); ++index)
        CHECK(same_bytes(actual.fields[index], expected.fields[index]));
    for (std::size_t index = 0; index < expected.moments.size(); ++index)
        CHECK(same_bytes(actual.moments[index], expected.moments[index]));
    for (std::size_t index = 0; index < expected.tendencies.size(); ++index)
        CHECK(same_bytes(actual.tendencies[index], expected.tendencies[index]));
    for (std::size_t index = 0; index < expected.sources.size(); ++index)
        CHECK(same_bytes(actual.sources[index], expected.sources[index]));
    for (std::size_t index = 0; index < expected.held.size(); ++index)
        CHECK(same_bytes(actual.held[index], expected.held[index]));
    CHECK(actual.radiation_powers == expected.radiation_powers);
    CHECK(actual.noise_elapsed == expected.noise_elapsed);
    CHECK(actual.pbl_base_valid == expected.pbl_base_valid);
    CHECK(same_bytes(actual.theta_residual, expected.theta_residual));
    CHECK(actual.rate == expected.rate);
    CHECK(actual.rainfall == expected.rainfall);
    CHECK(actual.rainfall_residual == expected.rainfall_residual);
    CHECK(actual.rainfall_time == expected.rainfall_time);
    CHECK(actual.surface_committed == expected.surface_committed);
    CHECK(actual.water.nonfinite_tendency_count == expected.water.nonfinite_tendency_count);
    for (std::size_t index = 0; index < actual.water.species.size(); ++index)
    {
        const auto& a = actual.water.species[index];
        const auto& b = expected.water.species[index];
        CHECK(std::tie(a.storage_rounding_kg, a.lower_bound_correction_kg,
            a.lower_bound_roundoff_correction_kg, a.upper_bound_correction_kg,
            a.nonfinite_correction_kg, a.lower_bound_count, a.lower_bound_roundoff_count,
            a.upper_bound_count, a.nonfinite_count)
            == std::tie(b.storage_rounding_kg, b.lower_bound_correction_kg,
            b.lower_bound_roundoff_correction_kg, b.upper_bound_correction_kg,
            b.nonfinite_correction_kg, b.lower_bound_count, b.lower_bound_roundoff_count,
            b.upper_bound_count, b.nonfinite_count));
    }
    CHECK(thermal_values(actual.heat) == thermal_values(expected.heat));
    const auto& a = actual.conservation;
    const auto& b = expected.conservation;
    CHECK(std::tie(a.active, a.tendency_imbalance_kg, a.baseline_q_kg,
        a.surface_precipitation_kg, a.relative_imbalance)
        == std::tie(b.active, b.tendency_imbalance_kg, b.baseline_q_kg,
        b.surface_precipitation_kg, b.relative_imbalance));
    REQUIRE(actual.surface.size() == expected.surface.size());
    for (std::size_t column = 0; column < actual.surface.size(); ++column)
        for (std::size_t species = 0; species < 5; ++species)
        {
            const auto& x = actual.surface[column][species];
            const auto& y = expected.surface[column][species];
            CHECK(std::tie(x.mass, x.number, x.reflectivity)
                == std::tie(y.mass, y.number, y.reflectivity));
        }
}

void check_no_fallout(const Snapshot& before, double duration)
{
    const Snapshot after;
    CHECK(after.rate == before.rate);
    CHECK(after.rainfall == before.rainfall);
    CHECK(after.rainfall_residual == before.rainfall_residual);
    CHECK(after.rainfall_time == simulation_time + duration);
    CHECK(after.conservation.surface_precipitation_kg == 0);
    REQUIRE(after.surface_committed);
    REQUIRE(after.surface.size() == static_cast<std::size_t>(NR * NTH));
    for (const auto& column : after.surface)
        for (const auto& inventory : column)
        {
            CHECK(inventory.mass == 0);
            CHECK(inventory.number == 0);
            CHECK(inventory.reflectivity == 0);
        }
}

void check_reference_step(const Snapshot& before, Species species, double duration)
{
    const std::size_t selected = species_index(species);
    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    REQUIRE(scheme.accepted_called);
    const auto& t = global_microphysics_runtime_state.tendencies;
    const std::array<const Field3D*, 8> tendencies{
        &t.dtheta_dt, &t.dqv_dt, &t.dqc_dt, &t.dqr_dt,
        &t.dqi_dt, &t.dqs_dt, &t.dqg_dt, &t.dqh_dt};
    const auto physical = physical_scalar_domain(global_simulation_state().grid);
    long double heat = 0;
    for (int i = physical.i_begin; i < physical.i_end; ++i)
    {
        const long double old_q = before.fields[8 + selected](i, 0, 0);
        const long double old_n = before.moments[1 + selected](i, 0, 0);
        const long double old_z = before.moments[6 + selected](i, 0, 0);
        const long double old_vapor = before.fields[6](i, 0, 0);
        const long double pressure = before.fields[1](i, 0, 0);
        const long double density = before.fields[0](i, 0, 0);
        const long double exner = std::pow(pressure / microphysics_constants::p0,
            static_cast<long double>(microphysics_constants::R_d) / microphysics_constants::cp);
        const long double temperature = before.fields[5](i, 0, 0) * exner;
        const long double alpha = old_q == 0 ? 0
            : reference_shape(old_q, old_n, old_z, species);
        const long double dq = reference_increment(temperature, pressure, density,
            old_vapor, old_q, old_n, alpha, species, duration);
        const long double dn = dq < 0 ? old_n * dq / old_q : 0;
        const long double dz_increment = old_q == 0 ? 0
            : old_z * (2 * dq / old_q - dn / old_n);
        INFO("cell " << i << ", species " << selected << ", alpha " << static_cast<double>(alpha));
        CHECK(selected_mass(species)(i, 0, 0)
            == Approx(static_cast<double>(old_q + dq)).epsilon(4e-7).margin(1e-22));
        CHECK(scheme.internal_triple_number_for_test(selected)(i, 0, 0)
            == Approx(static_cast<double>(old_n + dn)).epsilon(4e-7).margin(1e-22));
        CHECK(scheme.internal_triple_reflectivity_for_test(selected)(i, 0, 0)
            == Approx(static_cast<double>(old_z + dz_increment)).epsilon(4e-7).margin(1e-30));
        CHECK(tendencies[3 + selected]->operator()(i, 0, 0)
            == Approx(static_cast<double>(dq / duration)).epsilon(3e-6).margin(1e-18));
        CHECK(t.dqv_dt(i, 0, 0) == -(*tendencies[3 + selected])(i, 0, 0));
        CHECK(t.dtheta_dt(i, 0, 0)
            == Approx(static_cast<double>(microphysics_constants::L_s / microphysics_constants::cp
                * dq / duration / exner)).epsilon(3e-6).margin(1e-15));
        CHECK(qv(i, 0, 0)
            == Approx(static_cast<double>(old_vapor - dq)).epsilon(4e-7).margin(1e-22));
        CHECK(theta(i, 0, 0) == Approx(static_cast<double>(before.fields[5](i, 0, 0)
            + microphysics_constants::L_s / microphysics_constants::cp * dq / exner)).margin(3.1e-5));
        for (const long double scale : {1.0L, density})
        {
            CHECK(static_cast<double>(scale * (selected_mass(species)(i, 0, 0)
                + static_cast<long double>(qv(i, 0, 0))))
                == Approx(static_cast<double>(scale * (old_q + old_vapor))).epsilon(4e-7));
        }
        heat += density * dr * dtheta * dz * microphysics_constants::L_s * dq;
        CHECK(scheme.internal_triple_cloud_number_for_test()(i, 0, 0) == 0);
        for (std::size_t other = 0; other < 5; ++other)
            if (other != selected)
            {
                CHECK(public_fields()[8 + other]->operator()(i, 0, 0) == 0);
                CHECK(scheme.internal_triple_number_for_test(other)(i, 0, 0) == 0);
                CHECK(scheme.internal_triple_reflectivity_for_test(other)(i, 0, 0) == 0);
                CHECK(tendencies[3 + other]->operator()(i, 0, 0) == 0);
            }
        CHECK(qc(i, 0, 0) == 0);
        CHECK(t.dqc_dt(i, 0, 0) == 0);
    }
    const Snapshot after;
    for (int i = 0; i < NR; ++i)
        if (i < physical.i_begin || i >= physical.i_end)
        {
            for (std::size_t field = 0; field < before.fields.size(); ++field)
                CHECK(std::memcmp(&after.fields[field](i, 0, 0),
                    &before.fields[field](i, 0, 0), sizeof(float)) == 0);
            for (std::size_t moment = 0; moment < before.moments.size(); ++moment)
                CHECK(std::memcmp(&after.moments[moment](i, 0, 0),
                    &before.moments[moment](i, 0, 0), sizeof(float)) == 0);
            for (const auto* tendency : tendencies)
                CHECK(tendency->operator()(i, 0, 0) == 0);
        }
    for (const std::size_t field : {0, 1, 2, 3, 4, 13, 14})
        CHECK(same_bytes(after.fields[field], before.fields[field]));
    CHECK(after.heat.microphysics_heat_j == Approx(static_cast<double>(heat)).epsilon(3e-6).margin(1e-12));
    CHECK(after.water.lower_bound_count() == 0);
    CHECK(after.water.lower_bound_roundoff_count() == 0);
    CHECK(after.water.upper_bound_count() == 0);
    CHECK(after.water.nonfinite_count() == 0);
    CHECK(after.water.nonfinite_tendency_count == 0);
    CHECK(after.heat.theta_tendency_limit_count == 0);
    CHECK(after.heat.theta_nonfinite_count == 0);
    CHECK(after.heat.theta_lower_bound_count == 0);
    CHECK(after.heat.theta_upper_bound_count == 0);
    check_no_fallout(before, duration);
}

std::array<Field3D, 8> unpublished_fields()
{
    std::array<Field3D, 8> fields;
    for (auto& field : fields)
        field.resize(NR, NTH, NZ, 42.0f);
    return fields;
}

void evaluate(ObservedMilbrandt& scheme, double duration,
              std::array<Field3D, 8>& fields, const Field3D& vapor = qv)
{
    scheme.compute_tendencies(p, theta, rho, vapor, qc, qr, qi, qs, qg, qh,
        duration, fields[0], fields[1], fields[2], fields[3],
        fields[4], fields[5], fields[6], fields[7]);
}

void check_unpublished(const std::array<Field3D, 8>& fields)
{
    for (const auto& field : fields)
        for (std::size_t index = 0; index < field.size(); ++index)
            CHECK(field.data()[index] == 42.0f);
}

std::unique_ptr<tmv::MilbrandtScheme> parameter_mutation(int mutation)
{
    std::array<double, 25> p{
        0, 0, 0, 0, 0, 523.598775598, 3, 440, 3, .1597, 2.078,
        209.439510239, 3, 471.238898038, 3,
        149.1, .5, 71.34, .6635, 11.72, .41, 19.3, .37, 206.89, .6384};
    if (mutation < 25)
        p[mutation] = std::nextafter(p[mutation], std::numeric_limits<double>::infinity());
    return std::make_unique<tmv::MilbrandtScheme>(
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], mutation != 25);
}

void check_portable_identity(const PortableSchemeState& before,
                             const tmv::MilbrandtScheme& scheme)
{
    const auto after = scheme.portable_state();
    CHECK(after.version == before.version);
    CHECK(after.integers == before.integers);
    CHECK(after.reals == before.reals);
    REQUIRE(after.fields.size() == before.fields.size());
    for (std::size_t index = 0; index < before.fields.size(); ++index)
        CHECK(same_bytes(after.fields[index], before.fields[index]));
}

} // namespace

TEST_CASE("live frozen triple diffusion closes independent B1 B3 ventilation and pressure heat",
          "[physics][milbrandt][triple_cold_diffusion_internal][direct]")
{
    for (const Species species : {Species::Graupel, Species::Hail})
        for (const long double alpha : {0.0L, 2.0L, 20.0L})
            for (const double saturation : {.75, 1.0, 1.03})
            {
                std::array<std::unique_ptr<Snapshot>, 2> after;
                for (int run = 0; run < 2; ++run)
                {
#ifdef _OPENMP
                    omp_set_num_threads(run == 0 ? 1 : 4);
#endif
                    initialize_state(species, alpha, saturation);
                    const Snapshot before;
                    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), kDuration));
                    check_reference_step(before, species, kDuration);
                    after[run] = std::make_unique<Snapshot>();
                }
                check_snapshot(*after[0], *after[1]);
            }
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

TEST_CASE("live frozen diffusion recomputes gamma shape through successive cold deposition steps",
          "[physics][milbrandt][triple_cold_diffusion_internal][direct]")
{
    constexpr double duration = 2048.0;
    for (const Species species : {Species::Graupel, Species::Hail})
    {
        initialize_state(species, 2, 1.07, 1e-4f);
        const std::size_t selected = species_index(species);
        const Snapshot initial;
        const int physical_i = physical_scalar_domain(global_simulation_state().grid).i_begin;
        const long double initial_alpha = reference_shape(initial.fields[8 + selected](physical_i, 0, 0),
            initial.moments[1 + selected](physical_i, 0, 0), initial.moments[6 + selected](physical_i, 0, 0), species);
        for (int step = 0; step < 3; ++step)
        {
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            check_reference_step(before, species, duration);
            const Snapshot after;
            const long double alpha = reference_shape(after.fields[8 + selected](physical_i, 0, 0),
                after.moments[1 + selected](physical_i, 0, 0), after.moments[6 + selected](physical_i, 0, 0), species);
            CHECK(alpha > initial_alpha + 1e-6L);
            simulation_time += duration;
        }
    }
}

TEST_CASE("live frozen sublimation algebraically extinguishes q N Z and permits empty continuation",
          "[physics][milbrandt][triple_cold_diffusion_internal][direct]")
{
    constexpr double duration = 1048576.0;
    for (const Species species : {Species::Graupel, Species::Hail})
    {
        initialize_state(species, 2, 0, 1e-7f, .01f);
        const Snapshot before;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
        check_reference_step(before, species, duration);
        const Snapshot drained;
        const std::size_t selected = species_index(species);
        const auto physical = physical_scalar_domain(global_simulation_state().grid);
        for (int i = physical.i_begin; i < physical.i_end; ++i)
        {
            CHECK(drained.fields[8 + selected](i, 0, 0) == 0);
            CHECK(drained.moments[1 + selected](i, 0, 0) == 0);
            CHECK(drained.moments[6 + selected](i, 0, 0) == 0);
            CHECK(drained.fields[6](i, 0, 0) == before.fields[8 + selected](i, 0, 0));
        }
        simulation_time += duration;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), kDuration));
        check_reference_step(drained, species, kDuration);
        const Snapshot empty;
        for (std::size_t index = 0; index < drained.fields.size(); ++index)
            CHECK(same_bytes(empty.fields[index], drained.fields[index]));
        for (std::size_t index = 0; index < drained.moments.size(); ++index)
            CHECK(same_bytes(empty.moments[index], drained.moments[index]));
    }
}

TEST_CASE("live frozen diffusion late accepted-primary rejection preserves every owner and recovers",
          "[physics][milbrandt][triple_cold_diffusion_internal][direct]")
{
    for (const Species species : {Species::Graupel, Species::Hail})
    {
        initialize_state(species);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), kDuration));
        simulation_time += kDuration;
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).reject = true;
        const Snapshot before;
        REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), kDuration),
            std::string("post-primary microphysics private candidate rejected: ") + kRejectedHook);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).reject = false;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), kDuration));
        check_reference_step(before, species, kDuration);
    }
}

TEST_CASE("live frozen diffusion rejects late named nucleation and hail reclassification atomically",
          "[physics][milbrandt][triple_cold_diffusion_internal][direct]")
{
    for (int mutation = 0; mutation < 3; ++mutation)
    {
        const Species species = mutation == 0 ? Species::Graupel : Species::Hail;
        initialize_state(species);
        const Snapshot valid;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        if (mutation < 2)
            qv(last, 0, 0) = static_cast<float>(1.2
                * thermodynamics::saturation_mixing_ratio_ice(
                    thermodynamics::theta_to_temperature(theta(last, 0, 0), p(last, 0, 0)), p(last, 0, 0)));
        else
        {
            qh(last, 0, 0) = 1e-7f;
            auto& n = const_cast<Field3D&>(scheme.internal_triple_number_for_test(4));
            auto& z = const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(4));
            n(last, 0, 0) = 1.0f;
            z(last, 0, 0) = reference_z(qh(last, 0, 0), n(last, 0, 0), species, 2);
        }
        const Snapshot before;
        const std::string channel = mutation < 2 ? "vapor_to_ice" : "hail_to_graupel";
        REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), kDuration),
            Catch::Matchers::Contains("unowned cold channel: " + channel));
        check_snapshot(before);
        CHECK_FALSE(dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme).accepted_called);
        auto unpublished = unpublished_fields();
        auto& rejected = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        REQUIRE_THROWS_WITH(evaluate(rejected, kDuration, unpublished),
            Catch::Matchers::Contains("unowned cold channel: " + channel));
        check_unpublished(unpublished);
        std::string error;
        CHECK_FALSE(rejected.validate_prognostic_update(kDuration, error));
        CHECK_FALSE(rejected.commit_prepared_private_candidate());
        qv = valid.fields[6];
        qh = valid.fields[12];
        const std::size_t selected = species_index(species);
        const_cast<Field3D&>(rejected.internal_triple_number_for_test(selected)) = valid.moments[1 + selected];
        const_cast<Field3D&>(rejected.internal_triple_reflectivity_for_test(selected)) = valid.moments[6 + selected];
        const Snapshot recovered;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), kDuration));
        check_reference_step(recovered, species, kDuration);
    }
}

TEST_CASE("frozen diffusion activation rejects exact parameter moment shape and source mutations",
          "[physics][milbrandt][triple_cold_diffusion_internal][direct]")
{
    for (const Species species : {Species::Graupel, Species::Hail})
    {
        initialize_state(species);
        const std::size_t selected = species_index(species);
        const Snapshot valid;
        for (int mutation = 0; mutation < 26; ++mutation)
        {
            auto candidate = parameter_mutation(mutation);
            const auto before = candidate->portable_state();
            std::string error;
            INFO("parameter mutation " << mutation << ", species " << selected);
            CHECK_FALSE(candidate->enable_internal_triple_frozen_diffusion_for_test(
                global_simulation_state(), species, valid.moments[1 + selected],
                valid.moments[6 + selected], error));
            CHECK_FALSE(error.empty());
            CHECK_FALSE(candidate->owns_post_primary_private_commit());
            check_portable_identity(before, *candidate);
            check_snapshot(valid);
        }
        for (int mutation = 0; mutation < 18; ++mutation)
        {
            initialize_state(species);
            const Snapshot source;
            Field3D n(source.moments[1 + selected]);
            Field3D z(source.moments[6 + selected]);
            tmv::MilbrandtScheme candidate;
            auto state = global_simulation_state();
            Species requested = species;
            const int last = physical_scalar_domain(state.grid).i_end - 1;
            switch (mutation)
            {
                case 0: n(last, 0, 0) = 0; break;
                case 1: z(last, 0, 0) = 0; break;
                case 2: n(last, 0, 0) = -1; break;
                case 3: z(last, 0, 0) = -1; break;
                case 4: n(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 5: z(last, 0, 0) = std::numeric_limits<float>::infinity(); break;
                case 6: n.resize(1, 1, 1, 1); break;
                case 7: z.resize(1, 1, 1, 1); break;
                case 8: z(last, 0, 0) *= .01f; break;
                case 9: selected_mass(species)(last, 0, 0) = 0; break;
                case 10: p(last, 0, 0) = 200001; break;
                case 11: theta(last, 0, 0) = static_cast<float>(thermodynamics::temperature_to_theta(280, p(last, 0, 0))); break;
                case 12: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                case 13: state.storage_owner = nullptr; break;
                case 14: requested = Species::Rain; break;
                case 15: dtheta_dt_rad(last, 0, 0) = 1; break;
                case 16:
                case 17:
                {
                    candidate.prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
                    const auto scalars = candidate.prognostic_scalars_for_transport();
                    if (mutation == 16)
                        scalars[0].field->operator()(last, 0, 0) = 1;
                    else
                        scalars[1].field->resize(1, 1, 1, 0);
                    break;
                }
            }
            const Snapshot before;
            const Field3D n_before(n);
            const Field3D z_before(z);
            const auto private_before = candidate.portable_state();
            std::string error;
            INFO("activation mutation " << mutation << ", species " << selected);
            CHECK_FALSE(candidate.enable_internal_triple_frozen_diffusion_for_test(state, requested, n, z, error));
            CHECK_FALSE(error.empty());
            CHECK_FALSE(candidate.owns_post_primary_private_commit());
            CHECK(same_bytes(n, n_before));
            CHECK(same_bytes(z, z_before));
            check_portable_identity(private_before, candidate);
            check_snapshot(before);
        }
    }
}

TEST_CASE("empty frozen diffusion validates public private backend and held source owners atomically",
          "[physics][milbrandt][triple_cold_diffusion_internal][direct]")
{
    for (const Species species : {Species::Graupel, Species::Hail})
        for (int mutation = 0; mutation < 53; ++mutation)
        {
            initialize_state(species, 2, 1, 0, 0);
            auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
            const std::size_t selected = species_index(species);
            const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
            auto& number = const_cast<Field3D&>(scheme.internal_triple_number_for_test(selected));
            auto& z = const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(selected));
            if (mutation >= 29 && mutation < 34)
                source_fields()[mutation - 29]->operator()(last, 0, 0) = 1;
            else if (mutation >= 34 && mutation < 46)
            {
                auto* field = held_fields()[mutation - 34];
                field->resize(NR, NTH, NZ, 0);
                field->operator()(last, 0, 0) = 1;
            }
            else if (mutation >= 46)
                *held_radiation_powers()[mutation - 46] = 1;
            else
                switch (mutation)
                {
                    case 0: rho(last, 0, 0) = 0; break;
                    case 1: p(last, 0, 0) = 0; break;
                    case 2: p(last, 0, 0) = 200001; break;
                    case 3: theta(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                    case 4: theta(last, 0, 0) = static_cast<float>(thermodynamics::temperature_to_theta(280, p(last, 0, 0))); break;
                    case 5: qv(last, 0, 0) = -1; break;
                    case 6: qv(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                    case 7: selected_mass(species)(last, 0, 0) = -1; break;
                    case 8: selected_mass(species)(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                    case 9: qc(last, 0, 0) = 1e-4f; break;
                    case 10: qr(last, 0, 0) = 1e-4f; break;
                    case 11: qi(last, 0, 0) = 1e-4f; break;
                    case 12: qs(last, 0, 0) = 1e-4f; break;
                    case 13: selected_mass(species == Species::Graupel ? Species::Hail : Species::Graupel)(last, 0, 0) = 1e-4f; break;
                    case 14: qv.resize(1, 1, 1, 0); break;
                    case 15: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                    case 16: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
                    case 17: global_simulation_storage.runtime.compute.fallback_active = true; break;
                    case 18: global_simulation_storage.configuration.microphysics_config.milbrandt.triple_moment = true; break;
                    case 19: global_validation_policy.mode = tmv::GuardMode::Warn; break;
                    case 20: global_validation_policy.fail_on = tmv::GuardFailOn::NonFinite; break;
                    case 21: global_simulation_storage.configuration.domain_frame.velocity_ms.u_ms = 1; break;
                    case 22: global_simulation_storage.runtime.chaos.noise_elapsed_s = 1; break;
                    case 23: global_simulation_storage.runtime.chaos.pbl_base_tendencies_valid = true; break;
                    case 24: number(last, 0, 0) = 1; break;
                    case 25: z(last, 0, 0) = std::numeric_limits<float>::denorm_min(); break;
                    case 26: number.resize(1, 1, 1, 0); break;
                    case 27: const_cast<Field3D&>(scheme.internal_triple_number_for_test(0))(last, 0, 0) = 1; break;
                    case 28: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = 1; break;
                }
            const Snapshot before;
            INFO("zero-event mutation " << mutation << ", species " << selected);
            REQUIRE_THROWS(step_microphysics(global_simulation_state(), kDuration));
            check_snapshot(before);
            CHECK_FALSE(dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme).accepted_called);
        }
}

TEST_CASE("frozen diffusion candidate lifecycle rejects stale unprepared detached and failed evaluations",
          "[physics][milbrandt][triple_cold_diffusion_internal][direct]")
{
    for (const Species species : {Species::Graupel, Species::Hail})
    {
        initialize_state(species, 2, 1, 0, 0);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        const Snapshot before;
        std::string error;
        CHECK_THROWS_WITH(scheme.internal_triple_surface_for_test(1, 0, species_index(species)),
            "internal Milbrandt triple surface inventory is not committed");
        CHECK_FALSE(scheme.validate_prognostic_update(kDuration, error));
        CHECK_FALSE(scheme.commit_prepared_private_candidate());
        const auto identity = scheme.parameter_identity();
        tmv::MicrophysicsConfig config;
        config.milbrandt.hail_processes = false;
        CHECK_THROWS(scheme.configure(config));
        CHECK(scheme.parameter_identity() == identity);
        CHECK_THROWS(scheme.portable_state());
        CHECK_THROWS(scheme.clone_parameters());
        CHECK_THROWS(scheme.prognostic_scalars_for_transport());
        auto detached_outputs = unpublished_fields();
        const Field3D detached_vapor(qv);
        CHECK_THROWS_WITH(evaluate(scheme, kDuration, detached_outputs, detached_vapor),
            Catch::Matchers::Contains("runtime owner changed"));
        check_unpublished(detached_outputs);
        check_snapshot(before);
        for (const double invalid_dt : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::quiet_NaN()})
        {
            auto outputs = unpublished_fields();
            CHECK_THROWS(evaluate(scheme, invalid_dt, outputs));
            check_unpublished(outputs);
            CHECK_FALSE(scheme.validate_prognostic_update(kDuration, error));
            CHECK_FALSE(scheme.commit_prepared_private_candidate());
            check_snapshot(before);
        }
        auto evaluated = unpublished_fields();
        REQUIRE_NOTHROW(evaluate(scheme, kDuration, evaluated));
        CHECK(scheme.validate_prognostic_update(kDuration, error));
        CHECK_FALSE(scheme.validate_prognostic_update(2 * kDuration, error));
        CHECK_FALSE(scheme.commit_prepared_private_candidate());
        tmv::MicrophysicsWaterRepairReport water;
        tmv::MicrophysicsThermodynamicRepairReport heat;
        const tmv::MicrophysicsAcceptedPrimaryFields accepted{
            rho, qv, qc, qr, qi, qs, qg, qh, &water, &heat};
        CHECK_FALSE(scheme.prepare_post_primary_private_commit(2 * kDuration, accepted, error));
        CHECK_FALSE(scheme.commit_prepared_private_candidate());
        for (std::size_t owner = 0; owner < 8; ++owner)
        {
            std::array<const Field3D*, 8> fields{&rho, &qv, &qc, &qr, &qi, &qs, &qg, &qh};
            const Field3D detached(*fields[owner]);
            fields[owner] = &detached;
            const tmv::MicrophysicsAcceptedPrimaryFields wrong_owner{
                *fields[0], *fields[1], *fields[2], *fields[3], *fields[4],
                *fields[5], *fields[6], *fields[7], &water, &heat};
            CHECK_FALSE(scheme.prepare_post_primary_private_commit(kDuration, wrong_owner, error));
            CHECK_FALSE(scheme.commit_prepared_private_candidate());
            check_snapshot(before);
        }
        water.species[0].lower_bound_count = 1;
        CHECK_FALSE(scheme.prepare_post_primary_private_commit(kDuration, accepted, error));
        CHECK_FALSE(scheme.commit_prepared_private_candidate());
        water = {};
        heat.theta_tendency_limit_count = 1;
        CHECK_FALSE(scheme.prepare_post_primary_private_commit(kDuration, accepted, error));
        CHECK_FALSE(scheme.commit_prepared_private_candidate());
        heat = {};
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        qv(last, 0, 0) *= 1.2f;
        const Snapshot failed_source;
        auto failed_outputs = unpublished_fields();
        CHECK_THROWS_WITH(evaluate(scheme, kDuration, failed_outputs),
            Catch::Matchers::Contains("unowned cold channel: vapor_to_ice"));
        check_unpublished(failed_outputs);
        check_snapshot(failed_source);
        CHECK_FALSE(scheme.validate_prognostic_update(kDuration, error));
        CHECK_FALSE(scheme.prepare_post_primary_private_commit(kDuration, accepted, error));
        CHECK_FALSE(scheme.commit_prepared_private_candidate());
        qv = before.fields[6];
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), kDuration));
        check_reference_step(before, species, kDuration);
        auto& committed = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        CHECK_FALSE(committed.validate_prognostic_update(kDuration, error));
        CHECK_FALSE(committed.commit_prepared_private_candidate());
    }
}
