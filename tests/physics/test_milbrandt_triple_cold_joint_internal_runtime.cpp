#include "catch2/catch.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "microphysics/runtime.hpp"
#include "microphysics/schemes/milbrandt/milbrandt.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_diffusion.hpp"
#include "microphysics/schemes/milbrandt/triple_melting.hpp"
#include "microphysics/schemes/milbrandt/triple_ice_melting.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_primary.hpp"
#include "microphysics/schemes/milbrandt/triple_cloud_adjustment.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_cloud.hpp"
#include "microphysics/schemes/milbrandt/triple_sedimentation.hpp"
#include "microphysics/schemes/milbrandt/triple_frozen_collection.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_cloud_collection.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_cloud_diffusion.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_riming.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_ice_graupel.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_graupel_conversions.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_sll.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_collected_sll.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "numerics/advection/advection_base.hpp"
#include "support/milbrandt_snow_melting_reference.hpp"
#include "support/milbrandt_cloud_adjustment_reference.hpp"
#include "support/milbrandt_cold_riming_reference.hpp"
#include "support/milbrandt_cold_ice_graupel_reference.hpp"
#include "support/milbrandt_cold_graupel_conversions_reference.hpp"
#include "support/milbrandt_cold_homogeneous_reference.hpp"
#include "support/milbrandt_cold_sll_reference.hpp"
#include "support/milbrandt_cold_collected_sll_reference.hpp"
#include "support/milbrandt_cold_rain_sll_reference.hpp"
#include "support/milbrandt_rain_collision_reference.hpp"
#include "support/milbrandt_hail_wet_growth_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

using namespace tmv::milbrandt_detail;
using State = ColdJointFrozenDiffusionTripleState;
using Result = ColdJointFrozenDiffusionTripleResult;
using Population = std::array<long double, 3>;
constexpr std::array<long double, 2> kMass{209.439510239L, 471.238898038L};
constexpr std::array<long double, 2> kFall{19.3L, 206.89L};
constexpr std::array<long double, 2> kFallExponent{.37L, .6384L};

long double shape_ratio(long double alpha)
{
    return (alpha + 4) * (alpha + 5) * (alpha + 6)
        / ((alpha + 1) * (alpha + 2) * (alpha + 3));
}

ColdFrozenDiffusionTripleState population(std::size_t species, long double alpha,
                                         double q = 5e-4f, double n = 10.0f)
{
    if (q == 0 && n == 0)
    {
        return {};
    }
    const float z = static_cast<float>(shape_ratio(alpha) * q * q
        / (kMass[species] * kMass[species] * n));
    return {q, n, z};
}

long double reference_shape(const ColdFrozenDiffusionTripleState& source,
                            std::size_t species)
{
    if (source.mass_per_kg == 0)
    {
        return 0;
    }
    const long double q = source.mass_per_kg;
    const long double ratio = kMass[species] * kMass[species]
        * source.number_per_kg * source.reflectivity_per_kg / (q * q);
    REQUIRE(ratio > 1);
    long double lower = -1.0L + 1e-9L;
    long double upper = 1e6L;
    for (int iteration = 0; iteration < 120; ++iteration)
    {
        const long double middle = (lower + upper) / 2;
        if (shape_ratio(middle) > ratio)
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
    }
    return (lower + upper) / 2;
}

std::array<double, 16> cell(const State& initial, double density = .65,
                           double pressure = 90000, double saturation = .75,
                           double temperature = 253.15)
{
    std::array<double, 16> result{};
    result[0] = temperature;
    result[1] = pressure;
    result[2] = density;
    result[3] = saturation * thermodynamics::saturation_mixing_ratio_ice(temperature, pressure);
    result[8] = initial.graupel.mass_per_kg;
    result[9] = initial.hail.mass_per_kg;
    result[14] = initial.graupel.number_per_kg;
    result[15] = initial.hail.number_per_kg;
    return result;
}

long double numerical_gamma(long double order)
{
    constexpr int intervals = 32768;
    constexpr long double end = 160;
    constexpr long double width = end / intervals;
    long double sum = 0;
    for (int index = 1; index < intervals; ++index)
    {
        const long double x = index * width;
        sum += (index % 2 ? 4.0L : 2.0L) * std::pow(x, order) * std::exp(-x);
    }
    return (sum + std::pow(end, order) * std::exp(-end)) * width / 3;
}

long double reference_increment(const std::array<double, 16>& input,
                                const ColdFrozenDiffusionTripleState& source,
                                std::size_t species, long double duration)
{
    if (source.mass_per_kg == 0)
    {
        return 0;
    }
    const long double alpha = reference_shape(source, species);
    const long double q = source.mass_per_kg;
    const long double n = source.number_per_kg;
    const long double t = input[0];
    const long double tc = t - microphysics_constants::T0;
    const long double vapor_pressure = 611.15L
        * std::exp((23.036L - tc / 333.7L) * tc / (279.82L + tc));
    const long double qsi = .62198L * vapor_pressure / (input[1] - vapor_pressure);
    const long double density = input[2];
    const long double vapor = input[3];
    const long double diffusivity = (2.2157e-5L + .0155e-5L * tc) * 1e5L / input[1];
    const long double conductivity = 2.3971e-2L + .0078e-2L * tc;
    const long double viscosity = 1.72e-5L * 393.0L / (t + 120)
        * std::pow(t / microphysics_constants::T0, 1.5L) / density;
    const long double lambda = std::cbrt(kMass[species] * n
        * (alpha + 1) * (alpha + 2) * (alpha + 3) / q);
    const long double intercept = density * n * std::pow(lambda, alpha + 1)
        / std::tgamma(alpha + 1);
    const long double ventilation = .78L * numerical_gamma(alpha + 1)
        / std::pow(lambda, alpha + 2)
        + .30L * std::cbrt(viscosity / diffusivity)
        * std::sqrt(std::sqrt(1.225L / density) * kFall[species] / viscosity)
        * numerical_gamma(alpha + 1.5L + .5L * kFallExponent[species])
        / std::pow(lambda, alpha + 2.5L + .5L * kFallExponent[species]);
    const long double resistance = microphysics_constants::L_s * microphysics_constants::L_s
        / (conductivity * microphysics_constants::R_v * t * t)
        + 1 / (density * qsi * diffusivity);
    const long double raw = duration / density * 2 * std::acos(-1.0L)
        * (vapor / qsi - 1) * intercept * ventilation / resistance;
    if (vapor >= qsi)
    {
        const long double response = 1 + microphysics_constants::L_s * microphysics_constants::L_s
            * qsi / (microphysics_constants::cp * microphysics_constants::R_v
                     * (t - 7.66L) * (t - 7.66L));
        return std::clamp(raw, 0.0L, std::max((vapor - qsi) / response, 0.0L));
    }
    return std::clamp(raw, -q, 0.0L);
}

struct Reference
{
    std::array<long double, 2> dq;
    std::array<long double, 2> dn;
    std::array<Population, 2> diffusion;
    std::array<Population, 2> final;
    bool converted = false;
};

Reference reference(const std::array<double, 16>& input, const State& initial,
                     double duration)
{
    Reference result;
    const std::array<ColdFrozenDiffusionTripleState, 2> source{initial.graupel, initial.hail};
    for (std::size_t species = 0; species < 2; ++species)
    {
        result.dq[species] = reference_increment(input, source[species], species, duration);
    }
    for (int pass = 0; pass < 2; ++pass)
    {
        const long double available = input[3] + std::max(-result.dq[0], 0.0L)
            + std::max(-result.dq[1], 0.0L);
        const long double demand = std::max(result.dq[0], 0.0L)
            + std::max(result.dq[1], 0.0L);
        if (demand > available)
        {
            for (auto& increment : result.dq)
            {
                if (increment > 0)
                {
                    increment *= available / demand;
                }
            }
        }
    }
    for (std::size_t species = 0; species < 2; ++species)
    {
        const auto& s = source[species];
        const long double q = s.mass_per_kg;
        const long double n = s.number_per_kg;
        const long double z = s.reflectivity_per_kg;
        result.dn[species] = result.dq[species] < 0 ? n * result.dq[species] / q : 0;
        const long double dz = q == 0 ? 0
            : z * (2 * result.dq[species] / q - result.dn[species] / n);
        result.diffusion[species] = {q + result.dq[species], n + result.dn[species], z + dz};
    }
    result.final = result.diffusion;
    const auto& h = result.diffusion[1];
    if (h[0] > 0 && std::cbrt(h[0] / (kMass[1] * h[1])) < 1e-3L)
    {
        result.converted = true;
        result.final[0][0] += h[0];
        result.final[0][1] += h[1];
        result.final[0][2] += std::pow(kMass[1] / kMass[0], 2) * h[2];
        result.final[1] = {};
    }
    return result;
}

bool same_bytes(const Field3D& a, const Field3D& b)
{
    return a.size_r() == b.size_r() && a.size_th() == b.size_th()
        && a.size_z() == b.size_z() && a.size() == b.size()
        && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

int getter_attempts = 0;
int accepted_attempts = 0;

class ObservedMilbrandt final : public tmv::MilbrandtScheme
{
public:
    bool reject = false;
    bool accepted_called = false;
    bool surface_committed = false;
    int getter_mutation = 0;
    int late_mutation = 0;
    bool wrong_wind_declaration = false;
    mutable bool getter_called = false;
    mutable tmv::MicrophysicsPrimaryEndpointCandidate endpoint;

    const Field3D* observed_velocity_source() const noexcept
    {
        return vertical_velocity_field();
    }

    const Field3D* protected_primary_endpoint_wind_source() const noexcept override
    {
        return wrong_wind_declaration ? &p : MilbrandtScheme::protected_primary_endpoint_wind_source();
    }

    std::unique_ptr<MicrophysicsScheme> clone() const override
    {
        return std::make_unique<ObservedMilbrandt>(*this);
    }

    bool prepare_primary_endpoint_commit(
        const SimulationStateView& state, double duration,
        tmv::MicrophysicsPrimaryEndpointCandidate& candidate,
        std::string& error) const override
    {
        ++getter_attempts;
        if (!MilbrandtScheme::prepare_primary_endpoint_commit(state, duration, candidate, error))
        {
            return false;
        }
        getter_called = true;
        endpoint = candidate;
        const std::size_t last = static_cast<std::size_t>(physical_scalar_domain(state.grid).i_end - 1);
        switch (getter_mutation)
        {
            case 1: candidate.owner = nullptr; break;
            case 2: candidate.generation = 0; break;
            case 3: ++candidate.generation; break;
            case 4: candidate.duration_s = std::nextafter(duration, 1.0); break;
            case 5: ++candidate.nr; break;
            case 6: ++candidate.nth; break;
            case 7: ++candidate.nz; break;
            case 8: candidate.source_fields[0] = &state.fields.p; break;
            case 9: candidate.source_fields[3] = nullptr; break;
            case 10: candidate.water_after[0].pop_back(); break;
            case 11: candidate.theta_increment.pop_back(); break;
            case 12: candidate.water_after[0][last] = std::numeric_limits<double>::quiet_NaN(); break;
            case 13: candidate.water_after[5][last] = -1; break;
            case 14: candidate.theta_increment[last] = std::numeric_limits<double>::infinity(); break;
            case 15: candidate.water_after[1][last] = 1e-4; break;
            case 16: candidate.water_after[5][last] = std::nextafter(candidate.water_after[5][last], 1.0); break;
            case 17: candidate.theta_increment[last] = std::nextafter(candidate.theta_increment[last], 1.0); break;
            case 18: state.fields.rho.data()[last] = 0; break;
            case 19: state.fields.p.data()[last] = 0; break;
            case 20: state.fields.theta.data()[last] = std::numeric_limits<float>::quiet_NaN(); break;
            case 21: candidate.water_after[2][last] = std::nextafter(candidate.water_after[2][last], 1.0); break;
            case 22: candidate.source_fields[5] = &state.fields.qg; break;
            case 23: candidate.water_after[3][last] = std::nextafter(candidate.water_after[3][last], 1.0); break;
            case 24: candidate.water_after[4][last] = std::nextafter(candidate.water_after[4][last], 1.0); break;
            case 25: candidate.water_after[1][last] = std::nextafter(candidate.water_after[1][last], 1.0); break;
            case 26: state.fields.w.data()[last] = std::nextafter(state.fields.w.data()[last], 1.0f); break;
            case 27: const_cast<ObservedMilbrandt&>(*this).set_vertical_velocity_field(state.fields.w, true); break;
            case 28: const_cast<ObservedMilbrandt&>(*this).set_vertical_velocity_field(state.fields.p, false); break;
        }
        return true;
    }

    bool prepare_post_primary_private_commit(
        double duration, const tmv::MicrophysicsAcceptedPrimaryFields& accepted,
        std::string& error) override
    {
        ++accepted_attempts;
        if (!MilbrandtScheme::prepare_post_primary_private_commit(duration, accepted, error))
        {
            return false;
        }
        accepted_called = true;
        if (late_mutation == 1)
        {
            const_cast<tmv::MicrophysicsThermodynamicRepairReport&>(*accepted.thermodynamic_repairs)
                .microphysics_heat_j += 1;
        }
        if (late_mutation == 2)
        {
            const_cast<Field3D&>(internal_triple_number_for_test(3))(1, 0, 0) += 10000;
            error = "injected after joint frozen private moment validation";
            return false;
        }
        if (late_mutation == 3)
        {
            w(1, 0, 0) = std::nextafter(w(1, 0, 0), 100.0f);
        }
        if (late_mutation == 4)
        {
            set_vertical_velocity_field(w, true);
        }
        if (late_mutation == 5)
        {
            set_vertical_velocity_field(p, false);
        }
        if (late_mutation == 6)
        {
            const_cast<tmv::MicrophysicsThermodynamicRepairReport&>(*accepted.thermodynamic_repairs)
                .theta_storage_rounding_j += 1;
        }
        if (late_mutation == 7)
        {
            const_cast<tmv::MicrophysicsWaterRepairReport&>(*accepted.water_repairs)
                .species[3].storage_rounding_kg += 1;
        }
        if (late_mutation == 8)
        {
            const_cast<Field3D&>(internal_triple_reflectivity_for_test(1))(1, 0, 0) = 0;
            error = "injected after frozen collection reflectivity validation";
            return false;
        }
        if (reject)
        {
            error = "injected after joint frozen endpoint accepted-primary validation";
            return false;
        }
        return true;
    }

    bool commit_prepared_private_candidate() noexcept override
    {
        if (!MilbrandtScheme::commit_prepared_private_candidate())
        {
            return false;
        }
        surface_committed = true;
        return true;
    }
};

struct Profile
{
    float graupel_mass = 1e-4f;
    float graupel_number = 2e5f;
    float hail_mass = 5e-4f;
    float hail_number = 1e6f;
    long double graupel_alpha = 2;
    long double hail_alpha = 2;
    double saturation = .75;
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

void initialize_state(const Profile& profile = {})
{
    getter_attempts = 0;
    accepted_attempts = 0;
    NR = 66;
    NTH = NZ = 1;
    dr = 2;
    dtheta = 3;
    dz = 5;
    dt = .05;
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
    global_simulation_storage.runtime.advection = {};
    global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Cpu;
    global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Cpu;
    global_simulation_storage.runtime.boundary_layer.config.scheme_id = "none";
    global_simulation_storage.runtime.chaos.config.scheme_id = "none";
    global_simulation_storage.runtime.terrain.config.scheme_id = "none";
    global_simulation_storage.runtime.radiation.config.scheme_id = "none";
    auto& advection = global_simulation_storage.runtime.advection;
    advection.nr = NR;
    advection.nth = NTH;
    advection.nz = NZ;
    advection.hydrometeor_active = {true, false, true, false, true, false};
    advection.activity_current = true;
    advection.metadata_consumptions = 7;
    advection.fallback_scans = 11;
    global_microphysics_runtime_state.tendencies = {};
    global_microphysics_runtime_state.theta_storage_residual = {};
    global_microphysics_runtime_state.last_water_repair_report = {};
    global_microphysics_runtime_state.last_thermodynamic_repair_report = {};
    global_microphysics_runtime_state.last_conservation_report = {};
    for (auto* field : public_fields())
    {
        field->resize(NR, NTH, NZ, 0);
    }
    for (auto* field : source_fields())
    {
        field->resize(NR, NTH, NZ, 0);
    }
    global_simulation_storage.surface.precipitation_rate_mm_h.assign(NR, 0);
    global_simulation_storage.surface.accumulated_rainfall_mm.assign(NR, .25f);
    global_simulation_storage.surface.accumulated_rainfall_residual_mm.assign(NR, 0);
    global_simulation_storage.surface.accumulated_rainfall_last_update_s = 0;
    rho0_base.assign(1, 1);
    p0_base.assign(1, 90000);
    qv0_base.assign(1, .001);
    theta0_base.assign(1, 300);
    u0_base.assign(1, 0);
    v0_base.assign(1, 0);
    const auto g = population(0, profile.graupel_alpha, profile.graupel_mass, profile.graupel_number);
    const auto h = population(1, profile.hail_alpha, profile.hail_mass, profile.hail_number);
    Field3D ng(NR, NTH, NZ, profile.graupel_number);
    Field3D zg(NR, NTH, NZ, static_cast<float>(g.reflectivity_per_kg));
    Field3D nh(NR, NTH, NZ, profile.hail_number);
    Field3D zh(NR, NTH, NZ, static_cast<float>(h.reflectivity_per_kg));
    for (int i = 0; i < NR; ++i)
    {
        rho(i, 0, 0) = i % 2 ? .65f : 1.3f;
        p(i, 0, 0) = i % 4 < 2 ? 90000 : 70000;
        theta(i, 0, 0) = static_cast<float>(thermodynamics::temperature_to_theta(253.15, p(i, 0, 0)));
        const double temperature = thermodynamics::theta_to_temperature(theta(i, 0, 0), p(i, 0, 0));
        qv(i, 0, 0) = static_cast<float>(profile.saturation
            * thermodynamics::saturation_mixing_ratio_ice(temperature, p(i, 0, 0)));
        qg(i, 0, 0) = profile.graupel_mass;
        qh(i, 0, 0) = profile.hail_mass;
        u(i, 0, 0) = .25f;
        v(i, 0, 0) = -.5f;
        tke(i, 0, 0) = .125f;
    }
    auto scheme = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = true;
    scheme->configure(config);
    std::string error;
    const bool activated = scheme->enable_internal_triple_frozen_composition_for_test(
        global_simulation_state(), ng, zg, nh, zh, error);
    INFO(error);
    REQUIRE(activated);
    REQUIRE(scheme->owns_primary_endpoint_commit());
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
    std::tuple<int, int, int, std::array<bool, tmv::AdvectionRuntimeState::hydrometeor_count>, bool> activity;
    std::uint64_t metadata_consumptions;
    std::uint64_t fallback_scans;
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
        const auto& advection = global_simulation_storage.runtime.advection;
        activity = {advection.nr, advection.nth, advection.nz, advection.hydrometeor_active, advection.activity_current};
        metadata_consumptions = advection.metadata_consumptions;
        fallback_scans = advection.fallback_scans;
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
    CHECK(actual.activity == expected.activity);
    CHECK(actual.metadata_consumptions == expected.metadata_consumptions);
    CHECK(actual.fallback_scans == expected.fallback_scans);
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

void check_reference_step(const Snapshot& before, double duration)
{
    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    REQUIRE(scheme.getter_called);
    REQUIRE(scheme.accepted_called);
    const auto& candidate = scheme.endpoint;
    REQUIRE(candidate.owner == &scheme);
    REQUIRE(candidate.generation != 0);
    REQUIRE(candidate.duration_s == duration);
    REQUIRE(candidate.nr == NR);
    REQUIRE(candidate.nth == NTH);
    REQUIRE(candidate.nz == NZ);
    REQUIRE(candidate.theta_increment.size() == rho.size());
    const std::array<const Field3D*, 10> expected_sources{
        &rho, &p, &theta, &qv, &qc, &qr, &qi, &qs, &qg, &qh};
    CHECK(candidate.source_fields == expected_sources);
    const auto physical = physical_scalar_domain(global_simulation_state().grid);
    const std::array<const Field3D*, 7> water{&qv, &qc, &qr, &qi, &qs, &qg, &qh};
    for (const auto& field : candidate.water_after)
    {
        REQUIRE(field.size() == rho.size());
    }
    std::array<Reference, 4> cached;
    std::array<std::array<double, 10>, 4> signatures{};
    std::array<bool, 4> cached_valid{};
    std::array<long double, 7> storage_rounding{};
    std::array<long double, 7> requested_water{};
    std::array<long double, 6> moist{};
    long double requested_heat = 0;
    long double theta_rounding = 0;
    const auto& tendencies = global_microphysics_runtime_state.tendencies;
    for (int i = physical.i_begin; i < physical.i_end; ++i)
    {
        const std::size_t index = static_cast<std::size_t>(i);
        const State source{
            {before.fields[11](i, 0, 0), before.moments[4](i, 0, 0), before.moments[9](i, 0, 0)},
            {before.fields[12](i, 0, 0), before.moments[5](i, 0, 0), before.moments[10](i, 0, 0)}};
        auto input = cell(source, before.fields[0](i, 0, 0), before.fields[1](i, 0, 0), 1,
            thermodynamics::theta_to_temperature(before.fields[5](i, 0, 0), before.fields[1](i, 0, 0)));
        input[3] = before.fields[6](i, 0, 0);
        const std::array<double, 10> signature{input[0], input[1], input[2], input[3],
            source.graupel.mass_per_kg, source.graupel.number_per_kg, source.graupel.reflectivity_per_kg,
            source.hail.mass_per_kg, source.hail.number_per_kg, source.hail.reflectivity_per_kg};
        const std::size_t slot = static_cast<std::size_t>(i % 4);
        if (!cached_valid[slot])
        {
            cached[slot] = reference(input, source, duration);
            signatures[slot] = signature;
            cached_valid[slot] = true;
        }
        REQUIRE(signatures[slot] == signature);
        const auto& expected = cached[slot];
        const long double sum_diffusion = expected.dq[0] + expected.dq[1];
        const double exner = std::pow(input[1] / microphysics_constants::p0,
            microphysics_constants::R_d / microphysics_constants::cp);
        const double dry_mass = input[2] * dr * dtheta * dz;
        const double heat_factor = dry_mass * microphysics_constants::cp * exner;
        requested_heat += dry_mass * microphysics_constants::L_s * sum_diffusion;
        INFO("physical cell " << i << ", duration " << duration);
        CHECK(qg(i, 0, 0) == Approx(static_cast<double>(expected.final[0][0])).epsilon(4e-7).margin(1e-22));
        CHECK(qh(i, 0, 0) == Approx(static_cast<double>(expected.final[1][0])).epsilon(4e-7).margin(1e-22));
        CHECK(qv(i, 0, 0) == Approx(static_cast<double>(input[3] - sum_diffusion)).epsilon(4e-7).margin(1e-22));
        for (std::size_t frozen = 0; frozen < 2; ++frozen)
        {
            CHECK(scheme.internal_triple_number_for_test(3 + frozen)(i, 0, 0)
                == Approx(static_cast<double>(expected.final[frozen][1])).epsilon(4e-7).margin(1e-22));
            CHECK(scheme.internal_triple_reflectivity_for_test(3 + frozen)(i, 0, 0)
                == Approx(static_cast<double>(expected.final[frozen][2])).epsilon(4e-7).margin(1e-30));
        }
        CHECK(candidate.theta_increment[index] == Approx(static_cast<double>(
            microphysics_constants::L_s / microphysics_constants::cp * sum_diffusion / exner)).epsilon(3e-6).margin(1e-16));
        const double theta_exact = static_cast<double>(before.fields[5](i, 0, 0)) + candidate.theta_increment[index];
        CHECK(theta(i, 0, 0) == static_cast<float>(theta_exact));
        theta_rounding += (static_cast<double>(static_cast<float>(theta_exact)) - theta_exact) * heat_factor;
        long double total_after = 0;
        long double total_before = 0;
        for (std::size_t species = 0; species < water.size(); ++species)
        {
            const double endpoint = candidate.water_after[species][index];
            const double old = before.fields[6 + species](i, 0, 0);
            const float rounded = static_cast<float>(endpoint);
            CHECK(water[species]->operator()(i, 0, 0) == rounded);
            storage_rounding[species] += (static_cast<double>(rounded) - endpoint) * dry_mass;
            requested_water[species] += (endpoint - old) * dry_mass;
            total_after += rounded;
            total_before += old;
        }
        CHECK(static_cast<double>(dry_mass * total_after)
            == Approx(static_cast<double>(dry_mass * total_before)).epsilon(4e-7));
        const long double scaled_z = kMass[0] * kMass[0] * scheme.internal_triple_reflectivity_for_test(3)(i, 0, 0)
            + kMass[1] * kMass[1] * scheme.internal_triple_reflectivity_for_test(4)(i, 0, 0);
        CHECK(static_cast<double>(scaled_z) == Approx(static_cast<double>(kMass[0] * kMass[0]
            * expected.diffusion[0][2] + kMass[1] * kMass[1] * expected.diffusion[1][2])).epsilon(4e-7).margin(1e-25));
        const long double old_ice = before.fields[9](i, 0, 0) + before.fields[10](i, 0, 0)
            + static_cast<long double>(before.fields[11](i, 0, 0)) + before.fields[12](i, 0, 0);
        const long double new_ice = qi(i, 0, 0) + qs(i, 0, 0) + static_cast<long double>(qg(i, 0, 0)) + qh(i, 0, 0);
        const long double old_liquid = before.fields[7](i, 0, 0) + static_cast<long double>(before.fields[8](i, 0, 0));
        const long double new_liquid = qc(i, 0, 0) + static_cast<long double>(qr(i, 0, 0));
        const long double old_t = before.fields[5](i, 0, 0) * static_cast<long double>(exner);
        const long double new_t = theta(i, 0, 0) * static_cast<long double>(exner);
        const long double cv_v = physical_constants::specific_heat_water_vapor_cp_jkgk - microphysics_constants::R_v;
        moist[0] += dry_mass * (microphysics_constants::cp - microphysics_constants::R_d) * (new_t - old_t);
        moist[1] += dry_mass * ((new_t - microphysics_constants::T0)
            * (qv(i, 0, 0) * cv_v + new_liquid * 4186 + new_ice * 2106)
            - (old_t - microphysics_constants::T0)
            * (input[3] * cv_v + old_liquid * 4186 + old_ice * 2106));
        moist[2] += dry_mass * ((qv(i, 0, 0) - static_cast<long double>(input[3]))
            * (microphysics_constants::L_v - microphysics_constants::R_v * microphysics_constants::T0)
            - (new_ice - old_ice) * microphysics_constants::L_f);
        const long double kinetic = .5L * (std::pow(static_cast<long double>(u(i, 0, 0)), 2)
            + std::pow(static_cast<long double>(v(i, 0, 0)), 2) + std::pow(static_cast<long double>(w(i, 0, 0)), 2));
        const long double height = global_grid_geometry.z.empty() ? .5L * dz : global_grid_geometry.z[0];
        moist[3] += dry_mass * (total_after - total_before) * kinetic;
        moist[4] += dry_mass * (total_after - total_before) * physical_constants::gravity_ms2 * height;
        CHECK(qc(i, 0, 0) == 0);
        CHECK(qr(i, 0, 0) == 0);
        CHECK(qi(i, 0, 0) == 0);
        CHECK(qs(i, 0, 0) == 0);
        CHECK(tendencies.dqc_dt(i, 0, 0) == 0);
        CHECK(tendencies.dqr_dt(i, 0, 0) == 0);
        CHECK(tendencies.dqi_dt(i, 0, 0) == 0);
        CHECK(tendencies.dqs_dt(i, 0, 0) == 0);
    }
    const Snapshot after;
    for (const std::size_t field : {0, 1, 2, 3, 4, 13, 14})
    {
        CHECK(same_bytes(after.fields[field], before.fields[field]));
    }
    for (int i : {0, NR - 1})
    {
        for (std::size_t field = 0; field < before.fields.size(); ++field)
        {
            CHECK(std::memcmp(&after.fields[field](i, 0, 0), &before.fields[field](i, 0, 0), sizeof(float)) == 0);
        }
        for (std::size_t moment = 0; moment < before.moments.size(); ++moment)
        {
            CHECK(std::memcmp(&after.moments[moment](i, 0, 0), &before.moments[moment](i, 0, 0), sizeof(float)) == 0);
        }
    }
    CHECK(same_bytes(after.theta_residual, before.theta_residual));
    for (std::size_t species = 0; species < storage_rounding.size(); ++species)
    {
        CHECK(after.water.species[species].storage_rounding_kg
            == Approx(static_cast<double>(storage_rounding[species])).epsilon(3e-6).margin(1e-15));
        CHECK(after.heat.microphysics_water_tendency_kg_by_species[species]
            == Approx(static_cast<double>(requested_water[species])).epsilon(3e-6).margin(1e-12));
    }
    CHECK(after.heat.microphysics_heat_j == Approx(static_cast<double>(requested_heat)).epsilon(3e-6).margin(1e-12));
    CHECK(after.heat.theta_storage_rounding_j == Approx(static_cast<double>(theta_rounding)).epsilon(3e-6).margin(1e-12));
    CHECK(after.heat.theta_storage_residual_release_j == 0);
    CHECK(after.heat.theta_storage_residual_j == 0);
    const std::array<double, 6> measured_moist{after.heat.moist_dry_internal_change_j,
        after.heat.moist_water_sensible_change_j, after.heat.moist_phase_reference_change_j,
        after.heat.moist_carrier_kinetic_change_j, after.heat.moist_gravitational_change_j,
        after.heat.moist_subgrid_tke_change_j};
    for (std::size_t reservoir = 0; reservoir < moist.size(); ++reservoir)
    {
        CHECK(measured_moist[reservoir] == Approx(static_cast<double>(moist[reservoir])).epsilon(3e-6).margin(1e-10));
    }
    CHECK(after.heat.moist_energy_complete);
    CHECK(after.heat.moist_energy_closure_residual_j == 0);
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

} // namespace

TEST_CASE("live joint frozen endpoints remove whole hail exactly across non dyadic and dyadic durations",
          "[physics][milbrandt][triple_cold_joint_internal][direct]")
{
    for (const bool existing_graupel : {false, true})
    {
        for (const float donor : {5e-4f, 4e-4f})
        {
            for (const double duration : {.05, .1, .3, .0625, 1.0})
            {
                std::array<std::unique_ptr<Snapshot>, 2> after;
                for (int run = 0; run < 2; ++run)
                {
#ifdef _OPENMP
                    omp_set_num_threads(run == 0 ? 1 : 4);
#endif
                    Profile profile;
                    profile.hail_mass = donor;
                    profile.graupel_mass = existing_graupel ? 1e-4f : 0;
                    profile.graupel_number = existing_graupel ? 2e5f : 0;
                    initialize_state(profile);
                    const Snapshot before;
                    const float legacy_rate = static_cast<float>(-static_cast<double>(donor) / duration);
                    const double legacy_candidate = static_cast<double>(donor) + duration * legacy_rate;
                    if (duration == .0625 || duration == 1)
                    {
                        CHECK(legacy_candidate == 0);
                    }
                    else
                    {
                        CHECK(legacy_candidate != 0);
                        CHECK((legacy_candidate < 0) == (donor == 5e-4f ? duration != .3 : duration == .3));
                    }
                    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                    check_reference_step(before, duration);
                    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
                    const auto physical = physical_scalar_domain(global_simulation_state().grid);
                    for (int i = physical.i_begin; i < physical.i_end; ++i)
                    {
                        CHECK(qh(i, 0, 0) == 0);
                        CHECK(scheme.internal_triple_number_for_test(4)(i, 0, 0) == 0);
                        CHECK(scheme.internal_triple_reflectivity_for_test(4)(i, 0, 0) == 0);
                        CHECK(global_microphysics_runtime_state.tendencies.dqh_dt(i, 0, 0) == legacy_rate);
                    }
                    after[run] = std::make_unique<Snapshot>();
                }
                check_snapshot(*after[0], *after[1]);
            }
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

TEST_CASE("live joint endpoint late accepted-primary rejection restores all owners and recovers",
          "[physics][milbrandt][triple_cold_joint_internal][direct]")
{
    initialize_state();
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
    simulation_time += .05;
    dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).reject = true;
    const Snapshot before;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), .3),
        "post-primary microphysics private candidate rejected: injected after joint frozen endpoint accepted-primary validation");
    check_snapshot(before);
    dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).reject = false;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .3));
    check_reference_step(before, .3);
}

TEST_CASE("joint primary endpoint records and protected source writes reject before accepted publication",
          "[physics][milbrandt][triple_cold_joint_internal][direct]")
{
    for (int mutation = 1; mutation <= 20; ++mutation)
    {
        initialize_state();
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        INFO("getter mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .05));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
        check_reference_step(before, .05);
    }
}

TEST_CASE("joint primary endpoint ledger and private tampering roll back after base acceptance",
          "[physics][milbrandt][triple_cold_joint_internal][direct]")
{
    for (const int mutation : {1, 2})
    {
        initialize_state();
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
        simulation_time += .05;
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).late_mutation = mutation;
        const Snapshot before;
        const int calls_before = accepted_attempts;
        const std::string expected = mutation == 1
            ? "primary microphysics accepted attribution changed"
            : "post-primary microphysics private candidate rejected: injected after joint frozen private moment validation";
        REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), .1), expected);
        CHECK(accepted_attempts == calls_before + 1);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
        check_reference_step(before, .1);
    }
}

TEST_CASE("zero joint events validate backend public private and held source owners before the getter",
          "[physics][milbrandt][triple_cold_joint_internal][direct]")
{
    for (int mutation = 0; mutation < 40; ++mutation)
    {
        Profile profile;
        profile.graupel_mass = profile.graupel_number = profile.hail_mass = profile.hail_number = 0;
        initialize_state(profile);
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        if (mutation >= 21 && mutation < 33)
        {
            auto* held = held_fields()[mutation - 21];
            held->resize(NR, NTH, NZ, 0);
            held->operator()(last, 0, 0) = 1;
        }
        else if (mutation >= 33)
        {
            *held_radiation_powers()[mutation - 33] = 1;
        }
        else
        {
            switch (mutation)
            {
                case 0: rho(last, 0, 0) = 0; break;
                case 1: p(last, 0, 0) = 0; break;
                case 2: theta(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 3: qv(last, 0, 0) = -1; break;
                case 4: qc(last, 0, 0) = 1e-4f; break;
                case 5: qr(last, 0, 0) = 1e-4f; break;
                case 6: qi(last, 0, 0) = 1e-4f; break;
                case 7: qs(last, 0, 0) = 1e-4f; break;
                case 8: qv.resize(1, 1, 1, 0); break;
                case 9: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                case 10: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
                case 11: global_simulation_storage.runtime.compute.fallback_active = true; break;
                case 12: global_simulation_storage.configuration.microphysics_config.milbrandt.triple_moment = true; break;
                case 13: global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes = false; break;
                case 14: global_validation_policy.mode = tmv::GuardMode::Warn; break;
                case 15: const_cast<Field3D&>(scheme.internal_triple_number_for_test(3))(last, 0, 0) = 1; break;
                case 16: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(4))(last, 0, 0) = 1e-8f; break;
                case 17: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = 1; break;
                case 18: dtheta_dt_rad(last, 0, 0) = 1; break;
                case 19: global_simulation_storage.runtime.chaos.noise_elapsed_s = 1; break;
                case 20: qv(last, 0, 0) *= 2; break;
            }
        }
        const Snapshot before;
        INFO("zero joint mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .05));
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
    }
}

TEST_CASE("joint candidate generations and failed evaluations cannot prepare a stale private endpoint",
          "[physics][milbrandt][triple_cold_joint_internal][direct]")
{
    initialize_state();
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    const Snapshot before;
    std::string error;
    tmv::MicrophysicsPrimaryEndpointCandidate absent;
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, absent, error));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    auto compute = [&](std::array<Field3D, 8>& out, const Field3D& vapor)
    {
        scheme.compute_tendencies(p, theta, rho, vapor, qc, qr, qi, qs, qg, qh, .05,
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    };
    auto sentinels = [&]()
    {
        std::array<Field3D, 8> result;
        for (auto& field : result)
        {
            field.resize(NR, NTH, NZ, 42);
        }
        return result;
    };
    auto evaluated = sentinels();
    REQUIRE_NOTHROW(compute(evaluated, qv));
    tmv::MicrophysicsPrimaryEndpointCandidate first;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    CHECK(scheme.validate_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), .1, first, error));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    REQUIRE_NOTHROW(compute(evaluated, qv));
    tmv::MicrophysicsPrimaryEndpointCandidate second;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, second, error));
    CHECK(second.generation > first.generation);
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    const Field3D detached_vapor(qv);
    auto unpublished = sentinels();
    CHECK_THROWS(compute(unpublished, detached_vapor));
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), .05, second, error));
    for (const auto& field : unpublished)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
    qv(last, 0, 0) *= 2;
    const Snapshot bad_source;
    unpublished = sentinels();
    CHECK_THROWS_WITH(compute(unpublished, qv), Catch::Matchers::Contains("vapor_to_ice"));
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, absent, error));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    for (const auto& field : unpublished)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    check_snapshot(bad_source);
    qv = before.fields[6];
    const auto identity = scheme.parameter_identity();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = false;
    CHECK_THROWS(scheme.configure(config));
    CHECK(scheme.parameter_identity() == identity);
    CHECK_THROWS(scheme.portable_state());
    CHECK_THROWS(scheme.clone_parameters());
    CHECK_THROWS(scheme.prognostic_scalars_for_transport());
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
    check_reference_step(before, .05);
    auto& committed = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    CHECK_FALSE(committed.validate_primary_endpoint_commit(global_simulation_state(), .05, second, error));
    CHECK_FALSE(committed.commit_prepared_private_candidate());
}

TEST_CASE("live joint diffusion continues cold populations and algebraically extinguishes both species",
          "[physics][milbrandt][triple_cold_joint_internal][direct]")
{
    Profile profile;
    profile.graupel_mass = profile.hail_mass = 1e-4f;
    profile.graupel_number = profile.hail_number = 10;
    profile.hail_alpha = 20;
    for (const double saturation : {.75, 1.07})
    {
        profile.saturation = saturation;
        initialize_state(profile);
        for (int step = 0; step < 3; ++step)
        {
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 2048));
            check_reference_step(before, 2048);
            simulation_time += 2048;
        }
    }
    profile.graupel_mass = profile.hail_mass = 1e-7f;
    profile.graupel_number = profile.hail_number = .01f;
    profile.saturation = 0;
    initialize_state(profile);
    const Snapshot before;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1048576));
    check_reference_step(before, 1048576);
    simulation_time += 1048576;
    const auto physical = physical_scalar_domain(global_simulation_state().grid);
    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    for (int i = physical.i_begin; i < physical.i_end; ++i)
    {
        CHECK(qg(i, 0, 0) == 0);
        CHECK(qh(i, 0, 0) == 0);
        for (std::size_t species : {3, 4})
        {
            CHECK(scheme.internal_triple_number_for_test(species)(i, 0, 0) == 0);
            CHECK(scheme.internal_triple_reflectivity_for_test(species)(i, 0, 0) == 0);
        }
    }
    const Snapshot drained;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
    check_reference_step(drained, .1);
    simulation_time += .1;
    const Snapshot empty;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .3));
    check_reference_step(empty, .3);
}

TEST_CASE("joint activation rejects unsupported laws moments and source profiles without taking ownership",
          "[physics][milbrandt][triple_cold_joint_internal][direct]")
{
    initialize_state();
    const Snapshot source;
    for (int mutation = 0; mutation < 26; ++mutation)
    {
        std::array<double, 25> p{
            0, 0, 0, 0, 0, 523.598775598, 3, 440, 3, .1597, 2.078,
            209.439510239, 3, 471.238898038, 3,
            149.1, .5, 71.34, .6635, 11.72, .41, 19.3, .37, 206.89, .6384};
        if (mutation < 25)
        {
            p[mutation] = std::nextafter(p[mutation], std::numeric_limits<double>::infinity());
        }
        tmv::MilbrandtScheme candidate(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
            p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
            p[20], p[21], p[22], p[23], p[24], mutation != 25);
        const auto identity = candidate.parameter_identity();
        std::string error;
        INFO("joint cold law mutation " << mutation);
        CHECK_FALSE(candidate.enable_internal_triple_frozen_composition_for_test(global_simulation_state(),
            source.moments[4], source.moments[9], source.moments[5], source.moments[10], error));
        CHECK_FALSE(error.empty());
        CHECK_FALSE(candidate.owns_primary_endpoint_commit());
        CHECK_FALSE(candidate.owns_post_primary_private_commit());
        CHECK(candidate.parameter_identity() == identity);
        check_snapshot(source);
    }
    for (int mutation = 0; mutation < 14; ++mutation)
    {
        initialize_state();
        const Snapshot valid;
        Field3D ng(valid.moments[4]);
        Field3D zg(valid.moments[9]);
        Field3D nh(valid.moments[5]);
        Field3D zh(valid.moments[10]);
        tmv::MilbrandtScheme candidate;
        auto state = global_simulation_state();
        const int last = physical_scalar_domain(state.grid).i_end - 1;
        switch (mutation)
        {
            case 0: ng(last, 0, 0) = 0; break;
            case 1: nh(last, 0, 0) = 0; break;
            case 2: zg(last, 0, 0) = 0; break;
            case 3: zh(last, 0, 0) = std::numeric_limits<float>::infinity(); break;
            case 4: ng.resize(1, 1, 1, 0); break;
            case 5: zh.resize(1, 1, 1, 0); break;
            case 6: qg(last, 0, 0) = 0; break;
            case 7: qh(last, 0, 0) = 0; break;
            case 8: qc(last, 0, 0) = 1e-4f; break;
            case 9: state.storage_owner = nullptr; break;
            case 10: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
            case 11: global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes = false; break;
            case 12: dtheta_dt_rad(last, 0, 0) = 1; break;
            case 13:
            {
                candidate.prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
                candidate.prognostic_scalars_for_transport()[0].field->operator()(last, 0, 0) = 1;
                break;
            }
        }
        const Snapshot before;
        const std::array<Field3D, 4> moment_before{ng, zg, nh, zh};
        std::string error;
        INFO("joint activation mutation " << mutation);
        CHECK_FALSE(candidate.enable_internal_triple_frozen_composition_for_test(state, ng, zg, nh, zh, error));
        CHECK_FALSE(error.empty());
        CHECK_FALSE(candidate.owns_primary_endpoint_commit());
        CHECK_FALSE(candidate.owns_post_primary_private_commit());
        const std::array<const Field3D*, 4> moments{&ng, &zg, &nh, &zh};
        for (std::size_t index = 0; index < moments.size(); ++index)
        {
            CHECK(same_bytes(*moments[index], moment_before[index]));
        }
        check_snapshot(before);
    }
}

TEST_CASE("a pure joint shared vapor event that leaves the cold regime cannot publish live endpoints",
          "[physics][milbrandt][triple_cold_joint_internal][direct]")
{
    Profile profile;
    profile.graupel_mass = profile.hail_mass = .02f;
    profile.graupel_number = profile.hail_number = 1e5f;
    initialize_state(profile);
    for (int i = 0; i < NR; ++i)
    {
        p(i, 0, 0) = 200000;
        theta(i, 0, 0) = static_cast<float>(thermodynamics::temperature_to_theta(270.15, p(i, 0, 0)));
        qv(i, 0, 0) = static_cast<float>(4 * thermodynamics::saturation_mixing_ratio_ice(
            thermodynamics::theta_to_temperature(theta(i, 0, 0), p(i, 0, 0)), p(i, 0, 0)));
    }
    const Snapshot before;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), 1e7),
        Catch::Matchers::Contains("final heat"));
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(before);
}

namespace melting_live
{

constexpr long double rain_mass_coefficient = 523.598775598L;

struct Profile
{
    ::Profile frozen{5e-4f, 10.0f, 5e-4f, 10.0f, 2, 20, 1};
    float rain_mass = 1e-4f;
    float rain_number = 10.0f;
    double temperature = 283.15;
    double humidity = 1.0;
};

void initialize(const Profile& profile = {})
{
    initialize_state(profile.frozen);
    const auto g = population(0, profile.frozen.graupel_alpha,
        profile.frozen.graupel_mass, profile.frozen.graupel_number);
    const auto h = population(1, profile.frozen.hail_alpha,
        profile.frozen.hail_mass, profile.frozen.hail_number);
    const float zr = profile.rain_mass == 0 && profile.rain_number == 0 ? 0.0f
        : static_cast<float>(shape_ratio(2) * profile.rain_mass * profile.rain_mass
            / (rain_mass_coefficient * rain_mass_coefficient * profile.rain_number));
    Field3D nr(NR, NTH, NZ, profile.rain_number);
    Field3D rain_z(NR, NTH, NZ, zr);
    Field3D ng(NR, NTH, NZ, profile.frozen.graupel_number);
    Field3D zg(NR, NTH, NZ, static_cast<float>(g.reflectivity_per_kg));
    Field3D nh(NR, NTH, NZ, profile.frozen.hail_number);
    Field3D zh(NR, NTH, NZ, static_cast<float>(h.reflectivity_per_kg));
    for (int i = 0; i < NR; ++i)
    {
        theta(i, 0, 0) = static_cast<float>(thermodynamics::temperature_to_theta(
            profile.temperature, p(i, 0, 0)));
        qv(i, 0, 0) = static_cast<float>(profile.humidity
            * thermodynamics::saturation_mixing_ratio_water(microphysics_constants::T0, p(i, 0, 0)));
        qr(i, 0, 0) = profile.rain_mass;
    }
    auto scheme = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = true;
    scheme->configure(config);
    std::string error;
    const bool activated = scheme->enable_internal_triple_melting_for_test(
        global_simulation_state(), nr, rain_z, ng, zg, nh, zh, error);
    INFO(error);
    REQUIRE(activated);
    REQUIRE(scheme->owns_primary_endpoint_commit());
    microphysics_scheme = std::move(scheme);
}

long double melted_mass(const Snapshot& source, int i, std::size_t donor, double duration)
{
    const ColdFrozenDiffusionTripleState frozen{source.fields[11 + donor](i, 0, 0),
        source.moments[4 + donor](i, 0, 0), source.moments[9 + donor](i, 0, 0)};
    const long double pressure = source.fields[1](i, 0, 0);
    const long double density = source.fields[0](i, 0, 0);
    const long double temperature = source.fields[5](i, 0, 0)
        * std::pow(pressure / microphysics_constants::p0,
                   static_cast<long double>(microphysics_constants::R_d) / microphysics_constants::cp);
    const long double tc = temperature - microphysics_constants::T0;
    if (frozen.mass_per_kg == 0 || tc <= 0 || (donor == 1 && tc <= 5))
    {
        return 0;
    }
    const long double alpha = reference_shape(frozen, donor);
    const long double diffusivity = (2.2157e-5L + .0155e-5L * tc) * 1e5L / pressure;
    const long double conductivity = 2.3971e-2L + .0078e-2L * tc;
    const long double viscosity = 1.72e-5L * 393.0L / (temperature + 120)
        * std::pow(temperature / microphysics_constants::T0, 1.5L) / density;
    const long double lambda = std::cbrt(kMass[donor] * frozen.number_per_kg
        * (alpha + 1) * (alpha + 2) * (alpha + 3) / frozen.mass_per_kg);
    const long double intercept = density * frozen.number_per_kg
        * std::pow(lambda, alpha + 1) / std::tgamma(alpha + 1);
    const long double ventilation = .78L * numerical_gamma(alpha + 1)
        / std::pow(lambda, alpha + 2)
        + .30L * std::cbrt(viscosity / diffusivity)
        * std::sqrt(std::sqrt(1.225L / density) * kFall[donor] / viscosity)
        * numerical_gamma(alpha + 1.5L + .5L * kFallExponent[donor])
        / std::pow(lambda, alpha + 2.5L + .5L * kFallExponent[donor]);
    const long double saturation = .62198L * 611.21L / (pressure - 611.21L);
    const long double heat = conductivity * tc - density * microphysics_constants::L_v
        * diffusivity * (saturation - source.fields[6](i, 0, 0));
    const long double raw = duration * 2 * std::acos(-1.0L)
        / (density * microphysics_constants::L_f) * intercept * ventilation * heat;
    return std::clamp(raw, 0.0L, static_cast<long double>(frozen.mass_per_kg));
}

void check_step(const Snapshot& before, double duration, bool ice_only = false, bool snow_only = false)
{
    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    REQUIRE(scheme.getter_called);
    REQUIRE(scheme.accepted_called);
    const auto& candidate = scheme.endpoint;
    REQUIRE(candidate.owner == &scheme);
    REQUIRE(candidate.generation != 0);
    REQUIRE(candidate.duration_s == duration);
    REQUIRE(candidate.nr == NR);
    REQUIRE(candidate.nth == NTH);
    REQUIRE(candidate.nz == NZ);
    const std::array<const Field3D*, 10> bindings{&rho, &p, &theta, &qv, &qc, &qr, &qi, &qs, &qg, &qh};
    CHECK(candidate.source_fields == bindings);
    std::array<long double, 7> rounding{};
    std::array<long double, 7> requested{};
    std::array<long double, 6> moist{};
    long double heat = 0;
    long double theta_rounding = 0;
    std::array<std::array<long double, 2>, 4> cache{};
    std::array<snow_melting_reference::Result, 4> snow_cache{};
    std::array<bool, 4> cached{};
    const auto physical = physical_scalar_domain(global_simulation_state().grid);
    for (int i = physical.i_begin; i < physical.i_end; ++i)
    {
        const std::size_t index = static_cast<std::size_t>(i);
        const std::size_t slot = static_cast<std::size_t>(i % 4);
        if (!cached[slot])
        {
            if (snow_only)
            {
                const tmv::milbrandt_detail::MeltingTripleMomentState source{
                    before.fields[10](i, 0, 0), before.moments[3](i, 0, 0), before.moments[8](i, 0, 0)};
                snow_cache[slot] = snow_melting_reference::evaluate(source,
                    thermodynamics::theta_to_temperature(before.fields[5](i, 0, 0), before.fields[1](i, 0, 0)),
                    before.fields[1](i, 0, 0), before.fields[0](i, 0, 0), before.fields[6](i, 0, 0), duration);
                cache[slot] = {snow_cache[slot].melted_mass, 0};
            }
            else
            {
                cache[slot] = ice_only
                ? std::array<long double, 2>{thermodynamics::theta_to_temperature(
                    before.fields[5](i, 0, 0), before.fields[1](i, 0, 0)) > microphysics_constants::T0
                    ? before.fields[9](i, 0, 0) : 0.0L, 0.0L}
                : std::array<long double, 2>{melted_mass(before, i, 0, duration), melted_mass(before, i, 1, duration)};
            }
            cached[slot] = true;
        }
        const auto& melted = cache[slot];
        const std::array<std::size_t, 3> moment_species = snow_only
            ? std::array<std::size_t, 3>{0, 2, 3} : ice_only
            ? std::array<std::size_t, 3>{0, 1, 3} : std::array<std::size_t, 3>{0, 3, 4};
        const std::array<std::size_t, 3> mass_fields = snow_only
            ? std::array<std::size_t, 3>{8, 10, 11} : ice_only
            ? std::array<std::size_t, 3>{8, 9, 11} : std::array<std::size_t, 3>{8, 11, 12};
        std::array<Population, 3> endpoint{{
            {before.fields[8](i, 0, 0), before.moments[1](i, 0, 0), before.moments[6](i, 0, 0)},
            {before.fields[mass_fields[1]](i, 0, 0), before.moments[1 + moment_species[1]](i, 0, 0),
                before.moments[6 + moment_species[1]](i, 0, 0)},
            {before.fields[mass_fields[2]](i, 0, 0), before.moments[1 + moment_species[2]](i, 0, 0),
                before.moments[6 + moment_species[2]](i, 0, 0)}}};
        long double old_number = 0;
        long double old_scaled_z = 0;
        long double new_number = 0;
        long double new_scaled_z = 0;
        const std::array<long double, 3> coefficients = snow_only
            ? std::array<long double, 3>{rain_mass_coefficient, .1597L, kMass[0]} : ice_only
            ? std::array<long double, 3>{rain_mass_coefficient, 440.0L, kMass[0]}
            : std::array<long double, 3>{rain_mass_coefficient, kMass[0], kMass[1]};
        for (std::size_t species = 0; species < 3; ++species)
        {
            old_number += endpoint[species][1];
            old_scaled_z += coefficients[species] * coefficients[species] * endpoint[species][2];
        }
        const long double snow_mass_before = endpoint[1][0];
        const long double old_mass_squared = coefficients[0] * coefficients[0] * endpoint[0][2]
            + snow_cache[slot].mass_squared;
        for (std::size_t donor = 0; donor < 2; ++donor)
        {
            auto& frozen = endpoint[1 + donor];
            const long double fraction = frozen[0] == 0 ? 0 : melted[donor] / frozen[0];
            const long double transferred_number = frozen[1] * fraction;
            const long double transferred_z = frozen[2] * fraction;
            frozen[0] -= melted[donor];
            frozen[1] -= transferred_number;
            frozen[2] -= transferred_z;
            endpoint[0][0] += melted[donor];
            endpoint[0][1] += transferred_number;
            endpoint[0][2] += snow_only && donor == 0 ? snow_cache[slot].rain_z_gain
                : std::pow(coefficients[1 + donor] / rain_mass_coefficient, 2) * transferred_z;
        }
        const std::array<const Field3D*, 3> mass = snow_only
            ? std::array<const Field3D*, 3>{&qr, &qs, &qg} : ice_only
            ? std::array<const Field3D*, 3>{&qr, &qi, &qg} : std::array<const Field3D*, 3>{&qr, &qg, &qh};
        for (std::size_t species = 0; species < 3; ++species)
        {
            const auto number = scheme.internal_triple_number_for_test(moment_species[species])(i, 0, 0);
            const auto z = scheme.internal_triple_reflectivity_for_test(moment_species[species])(i, 0, 0);
            CHECK(mass[species]->operator()(i, 0, 0) == Approx(static_cast<double>(endpoint[species][0])).epsilon(4e-7).margin(1e-22));
            CHECK(number == Approx(static_cast<double>(endpoint[species][1])).epsilon(4e-7).margin(1e-22));
            CHECK(z == Approx(static_cast<double>(endpoint[species][2])).epsilon(4e-7).margin(1e-30));
            new_number += number;
            new_scaled_z += coefficients[species] * coefficients[species] * z;
        }
        CHECK(static_cast<double>(new_number) == Approx(static_cast<double>(old_number)).epsilon(4e-7));
        if (snow_only)
        {
            const long double fraction = snow_mass_before > 0 ? melted[0] / snow_mass_before : 0;
            const long double after_mass_squared = rain_mass_coefficient * rain_mass_coefficient
                * scheme.internal_triple_reflectivity_for_test(0)(i, 0, 0)
                + (1 - fraction) * snow_cache[slot].mass_squared;
            CHECK(static_cast<double>(after_mass_squared)
                == Approx(static_cast<double>(old_mass_squared)).epsilon(4e-7).margin(1e-25));
        }
        else
        {
            CHECK(static_cast<double>(new_scaled_z) == Approx(static_cast<double>(old_scaled_z)).epsilon(4e-7).margin(1e-25));
        }
        const double exner = std::pow(static_cast<double>(before.fields[1](i, 0, 0)) / microphysics_constants::p0,
            microphysics_constants::R_d / microphysics_constants::cp);
        const long double dry_mass = static_cast<long double>(before.fields[0](i, 0, 0)) * dr * dtheta * dz;
        const long double heat_factor = dry_mass * microphysics_constants::cp * exner;
        heat -= dry_mass * microphysics_constants::L_f * (melted[0] + melted[1]);
        CHECK(candidate.theta_increment[index] == Approx(static_cast<double>(
            -microphysics_constants::L_f / microphysics_constants::cp
            * (melted[0] + melted[1]) / exner)).epsilon(3e-6).margin(1e-16));
        const double theta_exact = static_cast<double>(before.fields[5](i, 0, 0)) + candidate.theta_increment[index];
        CHECK(theta(i, 0, 0) == static_cast<float>(theta_exact));
        theta_rounding += (static_cast<double>(theta(i, 0, 0)) - theta_exact) * heat_factor;
        long double total_before = 0;
        long double total_after = 0;
        for (std::size_t species = 0; species < 7; ++species)
        {
            const double old = before.fields[6 + species](i, 0, 0);
            const double exact = candidate.water_after[species][index];
            const float stored = bindings[3 + species]->operator()(i, 0, 0);
            CHECK(stored == static_cast<float>(exact));
            rounding[species] += (static_cast<double>(stored) - exact) * dry_mass;
            requested[species] += (exact - old) * dry_mass;
            total_before += old;
            total_after += stored;
        }
        CHECK(static_cast<double>(total_after * dry_mass)
            == Approx(static_cast<double>(total_before * dry_mass)).epsilon(4e-7));
        CHECK(qv(i, 0, 0) == before.fields[6](i, 0, 0));
        CHECK(qc(i, 0, 0) == 0);
        CHECK(qi(i, 0, 0) == 0);
        if (!snow_only)
        {
            CHECK(qs(i, 0, 0) == 0);
        }
        const long double old_ice = static_cast<long double>(before.fields[9](i, 0, 0))
            + before.fields[10](i, 0, 0) + before.fields[11](i, 0, 0) + before.fields[12](i, 0, 0);
        const long double new_ice = static_cast<long double>(qi(i, 0, 0)) + qs(i, 0, 0) + qg(i, 0, 0) + qh(i, 0, 0);
        const long double old_t = before.fields[5](i, 0, 0) * static_cast<long double>(exner);
        const long double new_t = theta(i, 0, 0) * static_cast<long double>(exner);
        const long double cv_v = physical_constants::specific_heat_water_vapor_cp_jkgk - microphysics_constants::R_v;
        moist[0] += dry_mass * (microphysics_constants::cp - microphysics_constants::R_d) * (new_t - old_t);
        moist[1] += dry_mass * ((new_t - microphysics_constants::T0)
            * (qv(i, 0, 0) * cv_v + static_cast<long double>(qr(i, 0, 0)) * 4186 + new_ice * 2106)
            - (old_t - microphysics_constants::T0)
            * (before.fields[6](i, 0, 0) * cv_v + static_cast<long double>(before.fields[8](i, 0, 0)) * 4186 + old_ice * 2106));
        moist[2] -= dry_mass * (new_ice - old_ice) * microphysics_constants::L_f;
        const long double kinetic = .5L * (std::pow(static_cast<long double>(u(i, 0, 0)), 2)
            + std::pow(static_cast<long double>(v(i, 0, 0)), 2) + std::pow(static_cast<long double>(w(i, 0, 0)), 2));
        const long double height = global_grid_geometry.z.empty() ? .5L * dz : global_grid_geometry.z[0];
        moist[3] += dry_mass * (total_after - total_before) * kinetic;
        moist[4] += dry_mass * (total_after - total_before) * physical_constants::gravity_ms2 * height;
    }
    const Snapshot after;
    for (std::size_t species = 0; species < 7; ++species)
    {
        CHECK(after.water.species[species].storage_rounding_kg
            == Approx(static_cast<double>(rounding[species])).epsilon(3e-6).margin(1e-15));
        CHECK(after.heat.microphysics_water_tendency_kg_by_species[species]
            == Approx(static_cast<double>(requested[species])).epsilon(3e-6).margin(1e-12));
    }
    CHECK(after.heat.microphysics_heat_j == Approx(static_cast<double>(heat)).epsilon(3e-6).margin(1e-12));
    CHECK(after.heat.theta_storage_rounding_j == Approx(static_cast<double>(theta_rounding)).epsilon(3e-6).margin(1e-12));
    const std::array<double, 6> measured{after.heat.moist_dry_internal_change_j, after.heat.moist_water_sensible_change_j,
        after.heat.moist_phase_reference_change_j, after.heat.moist_carrier_kinetic_change_j,
        after.heat.moist_gravitational_change_j, after.heat.moist_subgrid_tke_change_j};
    for (std::size_t index = 0; index < moist.size(); ++index)
    {
        INFO("moist reservoir " << index << ", duration " << duration);
        CHECK(measured[index] == Approx(static_cast<double>(moist[index])).epsilon(3e-6).margin(1e-10));
    }
    CHECK(after.heat.moist_energy_complete);
    CHECK(after.heat.moist_energy_closure_residual_j == 0);
    CHECK(after.water.lower_bound_count() == 0);
    CHECK(after.water.lower_bound_roundoff_count() == 0);
    CHECK(after.water.upper_bound_count() == 0);
    CHECK(after.water.nonfinite_count() == 0);
    CHECK(after.heat.theta_tendency_limit_count == 0);
    CHECK(after.heat.theta_nonfinite_count == 0);
    CHECK(after.heat.theta_lower_bound_count == 0);
    CHECK(after.heat.theta_upper_bound_count == 0);
    CHECK(same_bytes(after.theta_residual, before.theta_residual));
    for (std::size_t field : {0, 1, 2, 3, 4, 6, 7, 9, 10, 13, 14})
    {
        if ((ice_only && field == 9) || (snow_only && field == 10))
        {
            continue;
        }
        CHECK(same_bytes(after.fields[field], before.fields[field]));
    }
    if (ice_only || snow_only)
    {
        CHECK(same_bytes(after.fields[11], before.fields[11]));
        CHECK(same_bytes(after.fields[12], before.fields[12]));
    }
    for (int i : {0, NR - 1})
    {
        for (std::size_t field = 0; field < before.fields.size(); ++field)
        {
            CHECK(std::memcmp(&after.fields[field](i, 0, 0), &before.fields[field](i, 0, 0), sizeof(float)) == 0);
        }
        for (std::size_t moment = 0; moment < before.moments.size(); ++moment)
        {
            CHECK(std::memcmp(&after.moments[moment](i, 0, 0), &before.moments[moment](i, 0, 0), sizeof(float)) == 0);
        }
    }
    check_no_fallout(before, duration);
}

} // namespace melting_live

TEST_CASE("live melting exact primary endpoints preserve B1 B3 B5 and latent cooling",
          "[physics][milbrandt][triple_melting_internal][direct]")
{
    for (bool receiver : {false, true})
    {
        for (double duration : {.05, .1, .3, 1000.0})
        {
            melting_live::Profile profile;
            if (!receiver)
            {
                profile.rain_mass = 0;
                profile.rain_number = 0;
            }
            melting_live::initialize(profile);
            global_microphysics_runtime_state.theta_storage_residual.resize(NR, NTH, NZ, .0125f);
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            melting_live::check_step(before, duration);
            if (duration == 1000)
            {
                const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
                for (int i = 1; i < NR - 1; ++i)
                {
                    CHECK(qg(i, 0, 0) == 0);
                    CHECK(qh(i, 0, 0) == 0);
                    for (std::size_t species : {3, 4})
                    {
                        CHECK(scheme.internal_triple_number_for_test(species)(i, 0, 0) == 0);
                        CHECK(scheme.internal_triple_reflectivity_for_test(species)(i, 0, 0) == 0);
                    }
                }
            }
        }
    }
}

TEST_CASE("live melting partial humid and graupel only activation use independent references",
          "[physics][milbrandt][triple_melting_internal][direct]")
{
    for (double temperature : {277.0, 283.15})
    {
        for (double humidity : {.5, 1.1})
        {
            melting_live::Profile profile;
            profile.temperature = temperature;
            profile.humidity = humidity;
            profile.frozen.graupel_alpha = 20;
            profile.frozen.hail_alpha = 2;
            melting_live::initialize(profile);
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
            melting_live::check_step(before, .1);
            if (temperature == 277)
            {
                const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
                CHECK(same_bytes(qh, before.fields[12]));
                CHECK(same_bytes(scheme.internal_triple_number_for_test(4), before.moments[5]));
                CHECK(same_bytes(scheme.internal_triple_reflectivity_for_test(4), before.moments[10]));
            }
        }
    }
}

TEST_CASE("live melting continuation extinction and one four worker endpoints are exact",
          "[physics][milbrandt][triple_melting_internal][direct]")
{
#ifdef _OPENMP
    const int old_dynamic = omp_get_dynamic();
    const int old_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    std::array<std::unique_ptr<Snapshot>, 2> result;
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        omp_set_num_threads(workers);
        int observed = 0;
#pragma omp parallel
        {
#pragma omp single
            observed = omp_get_num_threads();
        }
        REQUIRE(observed == workers);
        melting_live::initialize();
        global_microphysics_runtime_state.theta_storage_residual.resize(NR, NTH, NZ, .0125f);
        for (double duration : {.05, .1, .3, 1000.0, .05})
        {
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            melting_live::check_step(before, duration);
            simulation_time += duration;
        }
        result[static_cast<std::size_t>(run)] = std::make_unique<Snapshot>();
        CHECK(qg(1, 0, 0) == 0);
        CHECK(qh(1, 0, 0) == 0);
    }
    check_snapshot(*result[0], *result[1]);
    omp_set_num_threads(old_threads);
    omp_set_dynamic(old_dynamic);
#else
    FAIL("live melting worker evidence requires OpenMP");
#endif
}

TEST_CASE("live melting protected sources candidates and late private reports reject atomically and recover",
          "[physics][milbrandt][triple_melting_internal][direct]")
{
    for (int mutation = 1; mutation <= 22; ++mutation)
    {
        melting_live::initialize();
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        INFO("melting getter mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .05));
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
        melting_live::check_step(before, .05);
    }
    for (int mutation = 0; mutation < 3; ++mutation)
    {
        melting_live::initialize();
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
        simulation_time += .05;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        const int calls_before = accepted_attempts;
        INFO("melting late rejection " << mutation);
        const std::string expected = mutation == 1
            ? "primary microphysics accepted attribution changed"
            : mutation == 2
            ? "post-primary microphysics private candidate rejected: injected after joint frozen private moment validation"
            : "post-primary microphysics private candidate rejected: injected after joint frozen endpoint accepted-primary validation";
        REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), .1), expected);
        CHECK(accepted_attempts == calls_before + 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
        melting_live::check_step(before, .1);
    }
}

TEST_CASE("live melting zero events validate every public private backend and held source owner",
          "[physics][milbrandt][triple_melting_internal][direct]")
{
    for (int mutation = 0; mutation < 41; ++mutation)
    {
        melting_live::Profile profile;
        profile.rain_mass = profile.rain_number = 0;
        profile.frozen.graupel_mass = profile.frozen.graupel_number = 0;
        profile.frozen.hail_mass = profile.frozen.hail_number = 0;
        melting_live::initialize(profile);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        if (mutation >= 22 && mutation < 34)
        {
            auto* field = held_fields()[mutation - 22];
            field->resize(NR, NTH, NZ, 0);
            field->operator()(last, 0, 0) = 1;
        }
        else if (mutation >= 34)
        {
            *held_radiation_powers()[mutation - 34] = 1;
        }
        else
        {
            switch (mutation)
            {
                case 0: rho(last, 0, 0) = 0; break;
                case 1: p(last, 0, 0) = 0; break;
                case 2: theta(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 3: qv(last, 0, 0) = -1; break;
                case 4: qc(last, 0, 0) = 1e-4f; break;
                case 5: qi(last, 0, 0) = 1e-4f; break;
                case 6: qs(last, 0, 0) = 1e-4f; break;
                case 7: qr(last, 0, 0) = 1e-4f; break;
                case 8: qv.resize(1, 1, 1, 0); break;
                case 9: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                case 10: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
                case 11: global_simulation_storage.runtime.compute.fallback_active = true; break;
                case 12: global_simulation_storage.configuration.microphysics_config.milbrandt.triple_moment = true; break;
                case 13: global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes = false; break;
                case 14: global_validation_policy.mode = tmv::GuardMode::Warn; break;
                case 15: const_cast<Field3D&>(scheme.internal_triple_number_for_test(0))(last, 0, 0) = 1; break;
                case 16: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(0))(last, 0, 0) = 1e-8f; break;
                case 17: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = 1; break;
                case 18: const_cast<Field3D&>(scheme.internal_triple_number_for_test(1))(last, 0, 0) = 1; break;
                case 19: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(2))(last, 0, 0) = 1e-8f; break;
                case 20: dtheta_dt_rad(last, 0, 0) = 1; break;
                case 21: global_simulation_storage.runtime.chaos.noise_elapsed_s = 1; break;
            }
        }
        const Snapshot before;
        INFO("melting zero event mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .05));
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
    }
}

TEST_CASE("live melting generations detached sources failed phase exits and portable owners remain guarded",
          "[physics][milbrandt][triple_melting_internal][direct]")
{
    melting_live::initialize();
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    std::string error;
    tmv::MicrophysicsPrimaryEndpointCandidate absent;
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, absent, error));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    auto compute = [&](std::array<Field3D, 8>& out, const Field3D& rain)
    {
        scheme.compute_tendencies(p, theta, rho, qv, qc, rain, qi, qs, qg, qh, .05,
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    };
    std::array<Field3D, 8> out;
    REQUIRE_NOTHROW(compute(out, qr));
    tmv::MicrophysicsPrimaryEndpointCandidate first;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    CHECK(scheme.validate_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), .1, first, error));
    REQUIRE_NOTHROW(compute(out, qr));
    tmv::MicrophysicsPrimaryEndpointCandidate second;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, second, error));
    CHECK(second.generation > first.generation);
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    const Field3D detached(qr);
    for (auto& field : out)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    CHECK_THROWS(compute(out, detached));
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), .05, second, error));
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, absent, error));
    for (const auto& field : out)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    CHECK_THROWS(scheme.portable_state());
    CHECK_THROWS(scheme.clone_parameters());
    CHECK_THROWS(scheme.prognostic_scalars_for_transport());
    const Snapshot before;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
    melting_live::check_step(before, .05);
    CHECK_FALSE(dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme)
        .validate_primary_endpoint_commit(global_simulation_state(), .05, second, error));

    melting_live::Profile crossing;
    crossing.temperature = 273.20;
    crossing.humidity = 1.1;
    crossing.frozen.graupel_mass = .02f;
    crossing.frozen.graupel_number = 1e5f;
    crossing.frozen.hail_mass = crossing.frozen.hail_number = 0;
    melting_live::initialize(crossing);
    const Snapshot warm;
    CHECK(melting_live::melted_mass(warm, 1, 0, 1e7) == Approx(.02f));
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), 1e7), Catch::Matchers::Contains("final heat"));
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(warm);
}

TEST_CASE("live melting activation binds exact laws and rain moment owners without mutation",
          "[physics][milbrandt][triple_melting_internal][direct]")
{
    melting_live::initialize();
    const Snapshot source;
    for (int mutation = 0; mutation < 26; ++mutation)
    {
        std::array<double, 25> parameters{
            0, 0, 0, 0, 0, 523.598775598, 3, 440, 3, .1597, 2.078,
            209.439510239, 3, 471.238898038, 3,
            149.1, .5, 71.34, .6635, 11.72, .41, 19.3, .37, 206.89, .6384};
        if (mutation < 25)
        {
            parameters[mutation] = std::nextafter(parameters[mutation], std::numeric_limits<double>::infinity());
        }
        const auto& a = parameters;
        tmv::MilbrandtScheme candidate(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9],
            a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19],
            a[20], a[21], a[22], a[23], a[24], mutation != 25);
        const auto identity = candidate.parameter_identity();
        std::string error;
        CHECK_FALSE(candidate.enable_internal_triple_melting_for_test(global_simulation_state(),
            source.moments[1], source.moments[6], source.moments[4], source.moments[9],
            source.moments[5], source.moments[10], error));
        CHECK_FALSE(error.empty());
        CHECK_FALSE(candidate.owns_primary_endpoint_commit());
        CHECK_FALSE(candidate.owns_post_primary_private_commit());
        CHECK(candidate.parameter_identity() == identity);
        check_snapshot(source);
    }
    for (int mutation = 0; mutation < 6; ++mutation)
    {
        melting_live::initialize();
        const Snapshot valid;
        Field3D nr(valid.moments[1]);
        Field3D zr(valid.moments[6]);
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        switch (mutation)
        {
            case 0: nr(last, 0, 0) = 0; break;
            case 1: zr(last, 0, 0) = 0; break;
            case 2: zr(last, 0, 0) = std::numeric_limits<float>::infinity(); break;
            case 3: nr.resize(1, 1, 1, 0); break;
            case 4: zr.resize(1, 1, 1, 0); break;
            case 5: theta(last, 0, 0) = static_cast<float>(thermodynamics::temperature_to_theta(250, p(last, 0, 0))); break;
        }
        const Snapshot before;
        const Field3D nr_before(nr);
        const Field3D zr_before(zr);
        tmv::MilbrandtScheme candidate;
        std::string error;
        CHECK_FALSE(candidate.enable_internal_triple_melting_for_test(global_simulation_state(), nr, zr,
            valid.moments[4], valid.moments[9], valid.moments[5], valid.moments[10], error));
        CHECK_FALSE(error.empty());
        CHECK_FALSE(candidate.owns_primary_endpoint_commit());
        CHECK(same_bytes(nr, nr_before));
        CHECK(same_bytes(zr, zr_before));
        check_snapshot(before);
    }
}

namespace ice_melting_live
{

void initialize(bool receiver = true, float ice_mass = 5e-4f,
                float ice_number = 10.0f, long double alpha = 2,
                double temperature = 283.15)
{
    melting_live::Profile profile;
    profile.frozen.graupel_mass = profile.frozen.graupel_number = 0;
    profile.frozen.hail_mass = profile.frozen.hail_number = 0;
    profile.rain_mass = receiver ? 1e-4f : 0;
    profile.rain_number = receiver ? 10.0f : 0;
    profile.temperature = temperature;
    melting_live::initialize(profile);
    const auto& previous = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    const Field3D nr(previous.internal_triple_number_for_test(0));
    const Field3D zr(previous.internal_triple_reflectivity_for_test(0));
    const float ice_z = ice_mass == 0 && ice_number == 0 ? 0.0f
        : static_cast<float>(shape_ratio(alpha) * static_cast<long double>(ice_mass) * ice_mass
            / (440.0L * 440.0L * ice_number));
    Field3D ni(NR, NTH, NZ, ice_number);
    Field3D zi(NR, NTH, NZ, ice_z);
    for (int i = 0; i < NR; ++i)
    {
        qi(i, 0, 0) = ice_mass;
    }
    auto scheme = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = true;
    scheme->configure(config);
    std::string error;
    const bool activated = scheme->enable_internal_triple_ice_melting_for_test(
        global_simulation_state(), nr, zr, ni, zi, error);
    INFO(error);
    REQUIRE(activated);
    REQUIRE(scheme->owns_primary_endpoint_commit());
    microphysics_scheme = std::move(scheme);
}

} // namespace ice_melting_live

TEST_CASE("live pristine ice melting commits whole populations and exact latent cooling at non dyadic durations",
          "[physics][milbrandt][triple_ice_melting_internal][direct]")
{
    for (bool receiver : {false, true})
    {
        for (long double alpha : {0.0L, 2.0L, 20.0L})
        {
            for (double duration : {.05, .1, .3})
            {
                ice_melting_live::initialize(receiver, 5e-4f, 10.0f, alpha);
                global_microphysics_runtime_state.theta_storage_residual.resize(NR, NTH, NZ, .0125f);
                const Snapshot before;
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                melting_live::check_step(before, duration, true);
                const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
                for (int i = 1; i < NR - 1; ++i)
                {
                    CHECK(qi(i, 0, 0) == 0);
                    CHECK(scheme.internal_triple_number_for_test(1)(i, 0, 0) == 0);
                    CHECK(scheme.internal_triple_reflectivity_for_test(1)(i, 0, 0) == 0);
                    CHECK(global_microphysics_runtime_state.tendencies.dqi_dt(i, 0, 0)
                        == static_cast<float>(-static_cast<double>(before.fields[9](i, 0, 0)) / duration));
                }
            }
        }
    }
}

TEST_CASE("live ice melting extinction empty continuation and actual workers preserve every owner",
          "[physics][milbrandt][triple_ice_melting_internal][direct]")
{
#ifdef _OPENMP
    const int old_threads = omp_get_max_threads();
    const int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    std::array<std::unique_ptr<Snapshot>, 2> after;
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        omp_set_num_threads(workers);
        int observed = 0;
#pragma omp parallel
        {
#pragma omp single
            observed = omp_get_num_threads();
        }
        REQUIRE(observed == workers);
        ice_melting_live::initialize();
        global_microphysics_runtime_state.theta_storage_residual.resize(NR, NTH, NZ, .0125f);
        for (double duration : {.05, .1, .3})
        {
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            melting_live::check_step(before, duration, true);
            simulation_time += duration;
        }
        after[static_cast<std::size_t>(run)] = std::make_unique<Snapshot>();
    }
    check_snapshot(*after[0], *after[1]);
    omp_set_num_threads(old_threads);
    omp_set_dynamic(old_dynamic);
#else
    FAIL("live ice melting worker evidence requires OpenMP");
#endif
}

TEST_CASE("live ice melting record source and late ledger private failures roll back and recover",
          "[physics][milbrandt][triple_ice_melting_internal][direct]")
{
    for (int mutation = 1; mutation <= 23; ++mutation)
    {
        ice_melting_live::initialize();
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .05));
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
        melting_live::check_step(before, .05, true);
    }
    for (int mutation = 0; mutation < 3; ++mutation)
    {
        ice_melting_live::initialize();
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
        simulation_time += .05;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        const int calls_before = accepted_attempts;
        const std::string expected = mutation == 1
            ? "primary microphysics accepted attribution changed"
            : mutation == 2
            ? "post-primary microphysics private candidate rejected: injected after joint frozen private moment validation"
            : "post-primary microphysics private candidate rejected: injected after joint frozen endpoint accepted-primary validation";
        REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), .1), expected);
        CHECK(accepted_attempts == calls_before + 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
        melting_live::check_step(before, .1, true);
    }
}

TEST_CASE("live ice melting zero events reject malformed and unowned profiles before the getter",
          "[physics][milbrandt][triple_ice_melting_internal][direct]")
{
    for (int mutation = 0; mutation < 43; ++mutation)
    {
        ice_melting_live::initialize(false, 0, 0);
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        if (mutation >= 24 && mutation < 36)
        {
            auto* field = held_fields()[mutation - 24];
            field->resize(NR, NTH, NZ, 0);
            field->operator()(last, 0, 0) = 1;
        }
        else if (mutation >= 36)
        {
            *held_radiation_powers()[mutation - 36] = 1;
        }
        else
        {
            switch (mutation)
            {
                case 0: rho(last, 0, 0) = 0; break;
                case 1: p(last, 0, 0) = 0; break;
                case 2: theta(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 3: qv(last, 0, 0) = -1; break;
                case 4: qc(last, 0, 0) = 1e-4f; break;
                case 5: qs(last, 0, 0) = 1e-4f; break;
                case 6: qg(last, 0, 0) = 1e-4f; break;
                case 7: qh(last, 0, 0) = 1e-4f; break;
                case 8: qi(last, 0, 0) = 1e-4f; break;
                case 9: qv.resize(1, 1, 1, 0); break;
                case 10: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                case 11: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
                case 12: global_simulation_storage.runtime.compute.fallback_active = true; break;
                case 13: global_simulation_storage.configuration.microphysics_config.milbrandt.triple_moment = true; break;
                case 14: global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes = false; break;
                case 15: global_validation_policy.mode = tmv::GuardMode::Warn; break;
                case 16: const_cast<Field3D&>(scheme.internal_triple_number_for_test(1))(last, 0, 0) = 1; break;
                case 17: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(1))(last, 0, 0) = 1e-8f; break;
                case 18: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = 1; break;
                case 19: const_cast<Field3D&>(scheme.internal_triple_number_for_test(2))(last, 0, 0) = 1; break;
                case 20: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(3))(last, 0, 0) = 1e-8f; break;
                case 21: dtheta_dt_rad(last, 0, 0) = 1; break;
                case 22: global_simulation_storage.runtime.chaos.noise_elapsed_s = 1; break;
                case 23: theta(last, 0, 0) = static_cast<float>(thermodynamics::temperature_to_theta(250, p(last, 0, 0))); break;
            }
        }
        const Snapshot before;
        INFO("ice zero mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .05));
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
    }
}

TEST_CASE("live ice melting candidate lifecycle detached fields phase exits and portable guards are atomic",
          "[physics][milbrandt][triple_ice_melting_internal][direct]")
{
    ice_melting_live::initialize();
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    std::string error;
    tmv::MicrophysicsPrimaryEndpointCandidate absent;
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, absent, error));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    auto compute = [&](std::array<Field3D, 8>& out, const Field3D& ice)
    {
        scheme.compute_tendencies(p, theta, rho, qv, qc, qr, ice, qs, qg, qh, .05,
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    };
    std::array<Field3D, 8> out;
    REQUIRE_NOTHROW(compute(out, qi));
    tmv::MicrophysicsPrimaryEndpointCandidate first;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    CHECK(scheme.validate_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    REQUIRE_NOTHROW(compute(out, qi));
    tmv::MicrophysicsPrimaryEndpointCandidate second;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, second, error));
    CHECK(second.generation > first.generation);
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), .1, second, error));
    const Field3D detached(qi);
    for (auto& field : out)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    CHECK_THROWS(compute(out, detached));
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, absent, error));
    for (const auto& field : out)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    CHECK_THROWS(scheme.portable_state());
    CHECK_THROWS(scheme.clone_parameters());
    CHECK_THROWS(scheme.prognostic_scalars_for_transport());
    const Snapshot before;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
    melting_live::check_step(before, .05, true);
    CHECK_FALSE(dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme)
        .validate_primary_endpoint_commit(global_simulation_state(), .05, second, error));
    ice_melting_live::initialize(true, .02f, 1e5f, 2, 273.20);
    const Snapshot warm;
    CHECK(-microphysics_constants::L_f / microphysics_constants::cp * .02
        + thermodynamics::theta_to_temperature(theta(1, 0, 0), p(1, 0, 0)) < microphysics_constants::T0);
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), .1), Catch::Matchers::Contains("final heat"));
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(warm);
}

TEST_CASE("live ice melting activation rejects malformed moments exact law and unowned fields without changes",
          "[physics][milbrandt][triple_ice_melting_internal][direct]")
{
    for (int mutation = 0; mutation < 9; ++mutation)
    {
        ice_melting_live::initialize();
        const Snapshot valid;
        Field3D ni(valid.moments[2]);
        Field3D zi(valid.moments[7]);
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        switch (mutation)
        {
            case 0: ni(last, 0, 0) = 0; break;
            case 1: zi(last, 0, 0) = 0; break;
            case 2: zi(last, 0, 0) = std::numeric_limits<float>::infinity(); break;
            case 3: ni.resize(1, 1, 1, 0); break;
            case 4: zi.resize(1, 1, 1, 0); break;
            case 5: qs(last, 0, 0) = 1e-4f; break;
            case 6: qg(last, 0, 0) = 1e-4f; break;
            case 7: qh(last, 0, 0) = 1e-4f; break;
            case 8: theta(last, 0, 0) = static_cast<float>(thermodynamics::temperature_to_theta(250, p(last, 0, 0))); break;
        }
        const Snapshot before;
        const Field3D ni_before(ni);
        const Field3D zi_before(zi);
        tmv::MilbrandtScheme candidate(0, 0, 0, 0, 0);
        std::string error;
        CHECK_FALSE(candidate.enable_internal_triple_ice_melting_for_test(global_simulation_state(),
            valid.moments[1], valid.moments[6], ni, zi, error));
        CHECK_FALSE(error.empty());
        CHECK_FALSE(candidate.owns_primary_endpoint_commit());
        CHECK(same_bytes(ni, ni_before));
        CHECK(same_bytes(zi, zi_before));
        check_snapshot(before);
    }
    ice_melting_live::initialize();
    const Snapshot source;
    const double changed_ice_mass_law = std::nextafter(440.0, 1000.0);
    tmv::MilbrandtScheme changed(0, 0, 0, 0, 0, 523.598775598, 3, changed_ice_mass_law);
    std::string error;
    CHECK_FALSE(changed.enable_internal_triple_ice_melting_for_test(global_simulation_state(),
        source.moments[1], source.moments[6], source.moments[2], source.moments[7], error));
    CHECK_FALSE(changed.owns_primary_endpoint_commit());
    check_snapshot(source);
}

namespace warm_primary_live
{

void initialize(bool limited = false)
{
    ice_melting_live::initialize(true, 0, 0, 2, 280);
    const float cloud = limited ? .01f : .001f;
    const float rain = limited ? .002f : 1e-4f;
    const float rain_number = limited ? 1e5f : 1e3f;
    Field3D nc(NR, NTH, NZ, 1e5f);
    Field3D nr(NR, NTH, NZ, rain_number);
    Field3D zr(NR, NTH, NZ, static_cast<float>(20.0L * rain * rain
        / (523.598775598L * 523.598775598L * rain_number)));
    for (int i = 0; i < NR; ++i)
    {
        qc(i, 0, 0) = cloud;
        qr(i, 0, 0) = rain;
        qv(i, 0, 0) = static_cast<float>(.5 * thermodynamics::saturation_mixing_ratio_water(
            static_cast<float>(thermodynamics::theta_to_temperature(theta(i, 0, 0), p(i, 0, 0))), p(i, 0, 0)));
    }
    auto scheme = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = true;
    scheme->configure(config);
    std::string error;
    const bool activated = scheme->enable_internal_triple_warm_primary_for_test(
        global_simulation_state(), nc, nr, zr, error);
    INFO(error);
    REQUIRE(activated);
    REQUIRE(scheme->owns_primary_endpoint_commit());
    microphysics_scheme = std::move(scheme);
}

void check_step(const Snapshot& before, double duration, bool cloud_adjustment = false,
                bool warm_cloud = false, bool dynamic_warm_cloud = false,
                bool subcycled_warm_cloud = false)
{
    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    REQUIRE(scheme.getter_called);
    REQUIRE(scheme.accepted_called);
    const auto& record = scheme.endpoint;
    REQUIRE(record.owner == &scheme);
    REQUIRE(record.generation != 0);
    REQUIRE(record.duration_s == duration);
    const std::array<const Field3D*, 10> fields{&rho, &p, &theta, &qv, &qc, &qr, &qi, &qs, &qg, &qh};
    CHECK(record.source_fields == fields);
    std::array<long double, 7> rounding{};
    std::array<long double, 7> requested{};
    std::array<long double, 6> moist{};
    long double heat = 0;
    long double theta_rounding = 0;
    const auto parameters = tmv::milbrandt_detail::warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    const auto physical = physical_scalar_domain(global_simulation_state().grid);
    for (int i = physical.i_begin; i < physical.i_end; ++i)
    {
        const tmv::milbrandt_detail::WarmRainCell cell{
            static_cast<float>(thermodynamics::theta_to_temperature(before.fields[5](i, 0, 0), before.fields[1](i, 0, 0))),
            before.fields[1](i, 0, 0), before.fields[0](i, 0, 0), before.fields[6](i, 0, 0),
            before.fields[7](i, 0, 0), before.fields[8](i, 0, 0), before.moments[0](i, 0, 0), before.moments[1](i, 0, 0)};
        const tmv::milbrandt_detail::RainTripleMomentState initial{cell.rain, cell.rain_number, before.moments[6](i, 0, 0)};
        tmv::milbrandt_detail::ExactTripleWarmCellResult expected;
        std::string error;
        bool accepted = false;
        if (subcycled_warm_cloud)
        {
            const tmv::milbrandt_detail::DoubleWarmRainCell source{
                thermodynamics::theta_to_temperature(before.fields[5](i, 0, 0), before.fields[1](i, 0, 0)),
                cell.pressure, cell.density, cell.vapor, cell.cloud, cell.rain, cell.cloud_number, cell.rain_number};
            tmv::milbrandt_detail::ExactTripleSubcycledWarmCloudCellResult produced;
            accepted = tmv::milbrandt_detail::produce_exact_triple_subcycled_warm_cloud_cell(
                parameters, source, before.fields[4](i, 0, 0), duration, initial, produced, error);
            if (accepted)
            {
                const auto& e = produced.cloud.endpoints();
                const auto& controlled = produced.warm.producer;
                CHECK(controlled.completed_duration_s() == duration);
                CHECK(controlled.accepted_steps() > 0);
                CHECK(controlled.accepted_steps() <= controlled.options().max_accepted_steps);
                CHECK(controlled.trials() <= controlled.options().max_trials);
                CHECK(controlled.source_cell().temperature == e.temperature_after_k);
                CHECK(controlled.source_cell().vapor == e.vapor_per_kg);
                CHECK(controlled.source_cell().cloud == e.cloud_per_kg);
                CHECK(controlled.source_cell().cloud_number == e.cloud_number_per_kg);
                for (const auto& step : controlled.steps())
                {
                    CHECK(step.duration_s > 0);
                    CHECK(step.error_norm <= 1);
                }
                const auto independent = cloud_adjustment_reference::evaluate({source.temperature, source.pressure,
                    source.density, source.vapor, source.cloud, source.cloud_number, before.fields[4](i, 0, 0)});
                CHECK(e.vapor_per_kg == Approx(static_cast<double>(independent.vapor)).epsilon(3e-12).margin(3e-15));
                CHECK(e.cloud_per_kg == Approx(static_cast<double>(independent.cloud)).epsilon(3e-12).margin(3e-15));
                CHECK(e.cloud_number_per_kg == Approx(static_cast<double>(independent.number)).epsilon(3e-12).margin(1e-12));
                expected.endpoints = produced.endpoints;
                expected.endpoints.temperature_increment_k = produced.temperature_increment_k;
                expected.rounded_rain = produced.rounded_rain;
            }
        }
        else if (dynamic_warm_cloud)
        {
            const tmv::milbrandt_detail::DoubleWarmRainCell source{
                thermodynamics::theta_to_temperature(before.fields[5](i, 0, 0), before.fields[1](i, 0, 0)),
                cell.pressure, cell.density, cell.vapor, cell.cloud, cell.rain, cell.cloud_number, cell.rain_number};
            tmv::milbrandt_detail::ExactTripleDynamicWarmCloudCellResult produced;
            accepted = tmv::milbrandt_detail::produce_exact_triple_dynamic_warm_cloud_cell(
                parameters, source, before.fields[4](i, 0, 0), duration, initial, produced, error);
            if (accepted)
            {
                const auto& e = produced.cloud.endpoints();
                const auto independent = cloud_adjustment_reference::evaluate({source.temperature, source.pressure,
                    source.density, source.vapor, source.cloud, source.cloud_number, before.fields[4](i, 0, 0)});
                CHECK(e.vapor_per_kg == Approx(static_cast<double>(independent.vapor)).epsilon(3e-12).margin(3e-15));
                CHECK(e.cloud_per_kg == Approx(static_cast<double>(independent.cloud)).epsilon(3e-12).margin(3e-15));
                CHECK(e.cloud_number_per_kg == Approx(static_cast<double>(independent.number)).epsilon(3e-12).margin(1e-12));
                expected.endpoints = produced.endpoints;
                expected.endpoints.temperature_increment_k = produced.temperature_increment_k;
                expected.rounded_rain = produced.rounded_rain;
            }
        }
        else if (warm_cloud)
        {
            const tmv::milbrandt_detail::DoubleWarmRainCell source{
                thermodynamics::theta_to_temperature(before.fields[5](i, 0, 0), before.fields[1](i, 0, 0)),
                cell.pressure, cell.density, cell.vapor, cell.cloud, cell.rain, cell.cloud_number, cell.rain_number};
            tmv::milbrandt_detail::ExactTripleWarmCloudCellResult produced;
            accepted = tmv::milbrandt_detail::produce_exact_triple_warm_cloud_cell(
                parameters, source, before.fields[4](i, 0, 0), duration, initial, produced, error);
            if (accepted)
            {
                const auto& e = produced.cloud.endpoints();
                const auto independent = cloud_adjustment_reference::evaluate({source.temperature,source.pressure,
                    source.density,source.vapor,source.cloud,source.cloud_number,before.fields[4](i, 0, 0)});
                CHECK(e.vapor_per_kg == Approx(static_cast<double>(independent.vapor)).epsilon(3e-12).margin(3e-15));
                CHECK(e.cloud_per_kg == Approx(static_cast<double>(independent.cloud)).epsilon(3e-12).margin(3e-15));
                CHECK(e.cloud_number_per_kg == Approx(static_cast<double>(independent.number)).epsilon(3e-12).margin(1e-12));
                CHECK(produced.warm.producer.source_cell().cloud == e.cloud_per_kg);
                CHECK(produced.warm.producer.source_cell().cloud_number == e.cloud_number_per_kg);
                CHECK(produced.warm.producer.source_cell().temperature == e.temperature_after_k);
                expected.endpoints = produced.endpoints;
                expected.endpoints.temperature_increment_k = produced.temperature_increment_k;
                expected.rounded_rain = produced.rounded_rain;
            }
        }
        else if (cloud_adjustment)
        {
            const tmv::milbrandt_detail::TripleCloudAdjustmentCell source{
                thermodynamics::theta_to_temperature(before.fields[5](i, 0, 0), before.fields[1](i, 0, 0)),
                cell.pressure, cell.density, cell.vapor, cell.cloud, cell.cloud_number, before.fields[4](i, 0, 0)};
            tmv::milbrandt_detail::TripleCloudAdjustmentResult produced;
            accepted = tmv::milbrandt_detail::produce_exact_triple_cloud_adjustment(source, duration, produced, error);
            if (accepted)
            {
                const auto& e = produced.endpoints();
                const auto independent = cloud_adjustment_reference::evaluate(source);
                CHECK(e.vapor_per_kg == Approx(static_cast<double>(independent.vapor)).epsilon(3e-12).margin(3e-15));
                CHECK(e.cloud_per_kg == Approx(static_cast<double>(independent.cloud)).epsilon(3e-12).margin(3e-15));
                CHECK(e.cloud_number_per_kg == Approx(static_cast<double>(independent.number)).epsilon(3e-12).margin(1e-12));
                CHECK(e.temperature_increment_k == Approx(static_cast<double>(independent.heat)).epsilon(3e-12).margin(1e-11));
                expected.endpoints.vapor_per_kg = e.vapor_per_kg;
                expected.endpoints.cloud_per_kg = e.cloud_per_kg;
                expected.endpoints.rain_per_kg = cell.rain;
                expected.endpoints.cloud_number_per_kg = e.cloud_number_per_kg;
                expected.endpoints.rain_number_per_kg = cell.rain_number;
                expected.endpoints.temperature_increment_k = e.temperature_increment_k;
                expected.rounded_rain = initial;
            }
        }
        else
        {
            accepted = tmv::milbrandt_detail::evaluate_exact_triple_warm_cell(parameters, cell, duration, initial, expected, error);
        }
        INFO(error);
        REQUIRE(accepted);
        const auto& e = expected.endpoints;
        CHECK(record.water_after[0][i] == e.vapor_per_kg);
        CHECK(record.water_after[1][i] == e.cloud_per_kg);
        CHECK(record.water_after[2][i] == e.rain_per_kg);
        CHECK(qv(i, 0, 0) == static_cast<float>(e.vapor_per_kg));
        CHECK(qc(i, 0, 0) == static_cast<float>(e.cloud_per_kg));
        CHECK(qr(i, 0, 0) == static_cast<float>(e.rain_per_kg));
        CHECK(scheme.internal_triple_cloud_number_for_test()(i, 0, 0) == static_cast<float>(e.cloud_number_per_kg));
        CHECK(scheme.internal_triple_number_for_test(0)(i, 0, 0) == expected.rounded_rain.number_per_kg);
        CHECK(scheme.internal_triple_reflectivity_for_test(0)(i, 0, 0) == expected.rounded_rain.reflectivity_per_kg);
        const double exner = std::pow(static_cast<double>(cell.pressure) / microphysics_constants::p0,
            microphysics_constants::R_d / microphysics_constants::cp);
        const long double dry_mass = static_cast<long double>(cell.density) * dr * dtheta * dz;
        const long double factor = dry_mass * microphysics_constants::cp * exner;
        CHECK(record.theta_increment[i] == thermodynamics::temperature_tendency_to_theta(e.temperature_increment_k, cell.pressure));
        const double exact_theta = static_cast<double>(before.fields[5](i, 0, 0)) + record.theta_increment[i];
        CHECK(theta(i, 0, 0) == static_cast<float>(exact_theta));
        theta_rounding += (static_cast<double>(theta(i, 0, 0)) - exact_theta) * factor;
        heat += dry_mass * microphysics_constants::cp * e.temperature_increment_k;
        long double total_before = 0;
        long double total_after = 0;
        for (std::size_t species = 0; species < 7; ++species)
        {
            const double exact = record.water_after[species][i];
            const float stored = fields[3 + species]->operator()(i, 0, 0);
            const double old = before.fields[6 + species](i, 0, 0);
            CHECK(stored == static_cast<float>(exact));
            rounding[species] += (static_cast<double>(stored) - exact) * dry_mass;
            requested[species] += (exact - old) * dry_mass;
            total_before += old;
            total_after += stored;
        }
        CHECK(static_cast<double>(total_after * dry_mass) == Approx(static_cast<double>(total_before * dry_mass)).epsilon(4e-7));
        const long double old_t = static_cast<long double>(before.fields[5](i, 0, 0)) * exner;
        const long double new_t = static_cast<long double>(theta(i, 0, 0)) * exner;
        const long double old_liquid = static_cast<long double>(cell.cloud) + cell.rain;
        const long double new_liquid = static_cast<long double>(qc(i, 0, 0)) + qr(i, 0, 0);
        const long double cv_v = physical_constants::specific_heat_water_vapor_cp_jkgk - microphysics_constants::R_v;
        moist[0] += dry_mass * (microphysics_constants::cp - microphysics_constants::R_d) * (new_t - old_t);
        moist[1] += dry_mass * ((new_t - microphysics_constants::T0) * (qv(i, 0, 0) * cv_v + new_liquid * 4186)
            - (old_t - microphysics_constants::T0) * (cell.vapor * cv_v + old_liquid * 4186));
        moist[2] += dry_mass * (static_cast<long double>(qv(i, 0, 0)) - cell.vapor)
            * (microphysics_constants::L_v - microphysics_constants::R_v * microphysics_constants::T0);
        const long double kinetic = .5L * (std::pow(static_cast<long double>(u(i, 0, 0)), 2)
            + std::pow(static_cast<long double>(v(i, 0, 0)), 2) + std::pow(static_cast<long double>(w(i, 0, 0)), 2));
        const long double height = global_grid_geometry.z.empty() ? .5L * dz : global_grid_geometry.z[0];
        moist[3] += dry_mass * (total_after - total_before) * kinetic;
        moist[4] += dry_mass * (total_after - total_before) * physical_constants::gravity_ms2 * height;
    }
    const Snapshot after;
    for (std::size_t species = 0; species < 7; ++species)
    {
        CHECK(after.water.species[species].storage_rounding_kg == Approx(static_cast<double>(rounding[species])).epsilon(3e-6).margin(1e-15));
        CHECK(after.heat.microphysics_water_tendency_kg_by_species[species] == Approx(static_cast<double>(requested[species])).epsilon(3e-6).margin(1e-12));
    }
    CHECK(after.heat.microphysics_heat_j == Approx(static_cast<double>(heat)).epsilon(3e-6).margin(1e-12));
    CHECK(after.heat.theta_storage_rounding_j == Approx(static_cast<double>(theta_rounding)).epsilon(3e-6).margin(1e-12));
    const std::array<double, 6> values{after.heat.moist_dry_internal_change_j, after.heat.moist_water_sensible_change_j,
        after.heat.moist_phase_reference_change_j, after.heat.moist_carrier_kinetic_change_j,
        after.heat.moist_gravitational_change_j, after.heat.moist_subgrid_tke_change_j};
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        INFO("warm moist reservoir " << index);
        CHECK(values[index] == Approx(static_cast<double>(moist[index])).epsilon(3e-6).margin(1e-10));
    }
    CHECK(after.heat.moist_energy_complete);
    CHECK(after.heat.moist_energy_closure_residual_j == 0);
    CHECK(after.water.lower_bound_count() == 0);
    CHECK(after.water.nonfinite_count() == 0);
    CHECK(after.heat.theta_tendency_limit_count == 0);
    CHECK(same_bytes(after.theta_residual, before.theta_residual));
    for (std::size_t field : {0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14})
    {
        CHECK(same_bytes(after.fields[field], before.fields[field]));
    }
    check_no_fallout(before, duration);
}

} // namespace warm_primary_live

TEST_CASE("exact warm primary commits producer endpoints and independent ledger attribution",
          "[physics][milbrandt][triple_warm_primary_internal][direct]")
{
    for (bool limited : {false, true})
    {
        for (double duration : {.05, .1, .3})
        {
            warm_primary_live::initialize(limited);
            global_microphysics_runtime_state.theta_storage_residual.resize(NR, NTH, NZ, .0125f);
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            warm_primary_live::check_step(before, duration);
        }
    }
}

TEST_CASE("exact warm primary continuation and actual workers preserve all scientific owners",
          "[physics][milbrandt][triple_warm_primary_internal][direct]")
{
#ifdef _OPENMP
    const int old_threads = omp_get_max_threads();
    const int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    std::array<std::unique_ptr<Snapshot>, 2> after;
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        omp_set_num_threads(workers);
        int observed = 0;
#pragma omp parallel
        {
#pragma omp single
            observed = omp_get_num_threads();
        }
        REQUIRE(observed == workers);
        warm_primary_live::initialize(true);
        for (double duration : {.05, .1, .3})
        {
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            warm_primary_live::check_step(before, duration);
            simulation_time += duration;
        }
        after[static_cast<std::size_t>(run)] = std::make_unique<Snapshot>();
    }
    check_snapshot(*after[0], *after[1]);
    omp_set_num_threads(old_threads);
    omp_set_dynamic(old_dynamic);
#else
    FAIL("warm primary worker evidence requires OpenMP");
#endif
}

TEST_CASE("exact warm primary returned records and late private ledger failures restore and recover",
          "[physics][milbrandt][triple_warm_primary_internal][direct]")
{
    for (int mutation = 1; mutation <= 25; ++mutation)
    {
        warm_primary_live::initialize();
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .05));
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
        warm_primary_live::check_step(before, .05);
    }
    for (int mutation = 0; mutation < 3; ++mutation)
    {
        warm_primary_live::initialize();
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
        simulation_time += .05;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        const int calls_before = accepted_attempts;
        const std::string expected = mutation == 1 ? "primary microphysics accepted attribution changed"
            : mutation == 2 ? "post-primary microphysics private candidate rejected: injected after joint frozen private moment validation"
            : "post-primary microphysics private candidate rejected: injected after joint frozen endpoint accepted-primary validation";
        REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), .1), expected);
        CHECK(accepted_attempts == calls_before + 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
        warm_primary_live::check_step(before, .1);
    }
}

TEST_CASE("exact warm primary zero events still validate cloud private frozen backend and held owners",
          "[physics][milbrandt][triple_warm_primary_internal][direct]")
{
    for (int mutation = 0; mutation < 39; ++mutation)
    {
        warm_primary_live::initialize();
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        qv.fill(0);
        qc.fill(0);
        qr.fill(0);
        const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test()).fill(0);
        const_cast<Field3D&>(scheme.internal_triple_number_for_test(0)).fill(0);
        const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(0)).fill(0);
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        if (mutation >= 20 && mutation < 32)
        {
            auto* field = held_fields()[mutation - 20];
            field->resize(NR, NTH, NZ, 0);
            field->operator()(last, 0, 0) = 1;
        }
        else if (mutation >= 32)
        {
            *held_radiation_powers()[mutation - 32] = 1;
        }
        else
        {
            switch (mutation)
            {
                case 0: rho(last, 0, 0) = 0; break;
                case 1: p(last, 0, 0) = 0; break;
                case 2: theta(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 3: qv(last, 0, 0) = -1; break;
                case 4: qc(last, 0, 0) = 1e-4f; break;
                case 5: qi(last, 0, 0) = 1e-4f; break;
                case 6: qs(last, 0, 0) = 1e-4f; break;
                case 7: qg(last, 0, 0) = 1e-4f; break;
                case 8: qh(last, 0, 0) = 1e-4f; break;
                case 9: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                case 10: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
                case 11: global_simulation_storage.runtime.compute.fallback_active = true; break;
                case 12: global_validation_policy.mode = tmv::GuardMode::Warn; break;
                case 13: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = -1; break;
                case 14: const_cast<Field3D&>(scheme.internal_triple_number_for_test(0))(last, 0, 0) = 1; break;
                case 15: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(0))(last, 0, 0) = 1e-9f; break;
                case 16: const_cast<Field3D&>(scheme.internal_triple_number_for_test(1))(last, 0, 0) = 1; break;
                case 17: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(2))(last, 0, 0) = 1e-9f; break;
                case 18: dtheta_dt_rad(last, 0, 0) = 1; break;
                case 19: global_simulation_storage.runtime.chaos.noise_elapsed_s = 1; break;
            }
        }
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .05));
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
    }
}

TEST_CASE("exact warm primary generations detached sources and portable selectors stay guarded",
          "[physics][milbrandt][triple_warm_primary_internal][direct]")
{
    warm_primary_live::initialize();
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    std::string error;
    tmv::MicrophysicsPrimaryEndpointCandidate absent;
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, absent, error));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    auto compute = [&](std::array<Field3D, 8>& out, const Field3D& cloud)
    {
        scheme.compute_tendencies(p, theta, rho, qv, cloud, qr, qi, qs, qg, qh, .05,
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    };
    std::array<Field3D, 8> out;
    REQUIRE_NOTHROW(compute(out, qc));
    tmv::MicrophysicsPrimaryEndpointCandidate first;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    REQUIRE_NOTHROW(compute(out, qc));
    tmv::MicrophysicsPrimaryEndpointCandidate second;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, second, error));
    CHECK(second.generation > first.generation);
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    const Field3D detached(qc);
    for (auto& field : out)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    CHECK_THROWS(compute(out, detached));
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, absent, error));
    for (const auto& field : out)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    CHECK_THROWS(scheme.portable_state());
    CHECK_THROWS(scheme.clone_parameters());
    CHECK_THROWS(scheme.prognostic_scalars_for_transport());
    const Snapshot before;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
    warm_primary_live::check_step(before, .05);
    CHECK_FALSE(dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme)
        .validate_primary_endpoint_commit(global_simulation_state(), .05, second, error));
}

TEST_CASE("exact warm primary full cloud budget and unsupported phase exits are explicit",
          "[physics][milbrandt][triple_warm_primary_internal][direct]")
{
    warm_primary_live::initialize(true);
    const Snapshot before;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
    warm_primary_live::check_step(before, .1);
    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    for (int i = 1; i < NR - 1; ++i)
    {
        CHECK(qc(i, 0, 0) == 0);
        CHECK(scheme.internal_triple_cloud_number_for_test()(i, 0, 0) >= 0);
    }
    warm_primary_live::initialize();
    for (int i = 0; i < NR; ++i)
    {
        rho(i, 0, 0) = .65f;
        p(i, 0, 0) = 90000;
        theta(i, 0, 0) = static_cast<float>(thermodynamics::temperature_to_theta(273.20, p(i, 0, 0)));
        qv(i, 0, 0) = 0;
    }
    const auto& pending = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    const tmv::milbrandt_detail::WarmRainCell cell{
        static_cast<float>(thermodynamics::theta_to_temperature(theta(1, 0, 0), p(1, 0, 0))),
        p(1, 0, 0), rho(1, 0, 0), qv(1, 0, 0), qc(1, 0, 0), qr(1, 0, 0),
        pending.internal_triple_cloud_number_for_test()(1, 0, 0), pending.internal_triple_number_for_test(0)(1, 0, 0)};
    const tmv::milbrandt_detail::RainTripleMomentState initial{cell.rain, cell.rain_number,
        pending.internal_triple_reflectivity_for_test(0)(1, 0, 0)};
    double crossing_duration = 0;
    for (double duration : {10.0, 30.0, 60.0, 100.0, 200.0})
    {
        tmv::milbrandt_detail::ExactTripleWarmCellResult candidate;
        std::string error;
        if (tmv::milbrandt_detail::evaluate_exact_triple_warm_cell(
                tmv::milbrandt_detail::warm_rain_parameters(0, 523.598775598, 3, 149.1, .5),
                cell, duration, initial, candidate, error) &&
            cell.temperature + candidate.endpoints.temperature_increment_k < microphysics_constants::T0)
        {
            crossing_duration = duration;
            break;
        }
    }
    REQUIRE(crossing_duration > 0);
    const Snapshot warm;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), crossing_duration),
        Catch::Matchers::Contains("final heat"));
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(warm);
}

namespace warm_cloud_live
{

void initialize(int scenario = 0)
{
    ice_melting_live::initialize(false, 0, 0, 2, 290.125);
    const float rain_mass = scenario == 2 ? 1e-4f : scenario == 3 ? .002f : 0;
    const float rain_number = scenario == 2 ? 1e3f : scenario == 3 ? 1e5f : 0;
    Field3D nc(NR, NTH, NZ, 1e5f);
    Field3D nr(NR, NTH, NZ, rain_number);
    Field3D zr(NR, NTH, NZ, rain_mass == 0 ? 0.0f : static_cast<float>(20.0L * rain_mass * rain_mass
        / (523.598775598L * 523.598775598L * rain_number)));
    for (int i = 0; i < NR; ++i)
    {
        const double temperature = thermodynamics::theta_to_temperature(theta(i, 0, 0), p(i, 0, 0));
        const double qs = thermodynamics::saturation_mixing_ratio_water(temperature, p(i, 0, 0));
        w(i, 0, 0) = .5f;
        qc(i, 0, 0) = scenario == 0 ? .03f : scenario == 2 ? .001f : .01f;
        qv(i, 0, 0) = static_cast<float>((scenario == 0 ? 1.2 : scenario == 2 ? 1.05 : .8) * qs);
        qr(i, 0, 0) = rain_mass;
    }
    auto scheme = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = true;
    scheme->configure(config);
    std::string error;
    const bool activated = scheme->enable_internal_triple_warm_cloud_primary_for_test(global_simulation_state(), nc, nr, zr, error);
    INFO(error);
    REQUIRE(activated);
    REQUIRE(scheme->owns_primary_endpoint_commit());
    microphysics_scheme = std::move(scheme);
    global_microphysics_runtime_state.theta_storage_residual.resize(NR, NTH, NZ, .0125f);
}

void initialize_small_rain(bool cloud_present, long double alpha)
{
    ice_melting_live::initialize(false, 0, 0, 2, 290.125);
    constexpr float rain_number = 1e5f;
    constexpr long double mass_coefficient = 523.598775598L;
    constexpr std::array<long double, 3> diameters{40e-6L, 60e-6L, 80e-6L};
    Field3D nc(NR, NTH, NZ, cloud_present ? 1e7f : 0.0f);
    Field3D nr(NR, NTH, NZ, rain_number);
    Field3D zr(NR, NTH, NZ, 0);
    for (int i = 0; i < NR; ++i)
    {
        const double temperature = thermodynamics::theta_to_temperature(theta(i, 0, 0), p(i, 0, 0));
        const double saturation = thermodynamics::saturation_mixing_ratio_water(temperature, p(i, 0, 0));
        float stored_vapor = static_cast<float>(saturation);
        if (static_cast<double>(stored_vapor) > saturation)
        {
            stored_vapor = std::nextafter(stored_vapor, 0.0f);
        }
        const long double diameter = diameters[static_cast<std::size_t>(i) % diameters.size()];
        qr(i, 0, 0) = static_cast<float>(mass_coefficient * rain_number * diameter * diameter * diameter);
        zr(i, 0, 0) = static_cast<float>(shape_ratio(alpha) * qr(i, 0, 0) * qr(i, 0, 0)
            / (mass_coefficient * mass_coefficient * rain_number));
        qc(i, 0, 0) = cloud_present ? 5e-5f : 0.0f;
        qv(i, 0, 0) = stored_vapor;
        w(i, 0, 0) = .5f;
    }
    auto scheme = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = true;
    scheme->configure(config);
    std::string error;
    const bool activated = scheme->enable_internal_triple_warm_cloud_primary_for_test(
        global_simulation_state(), nc, nr, zr, error);
    INFO(error);
    REQUIRE(activated);
    REQUIRE(scheme->owns_primary_endpoint_commit());
    microphysics_scheme = std::move(scheme);
    global_microphysics_runtime_state.theta_storage_residual.resize(NR, NTH, NZ, .0125f);
}

void check_small_rain_transfer(const Snapshot& before, double duration)
{
    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    const auto physical = physical_scalar_domain(global_simulation_state().grid);
    const auto close = [](long double expected, long double actual)
    {
        const long double scale = std::max({std::abs(expected), std::abs(actual), 1e-30L});
        CHECK(std::abs(expected - actual) <= 64 * std::numeric_limits<double>::epsilon() * scale);
    };
    for (int i = physical.i_begin; i < physical.i_end; ++i)
    {
        const DoubleWarmRainCell source{
            thermodynamics::theta_to_temperature(before.fields[5](i, 0, 0), before.fields[1](i, 0, 0)),
            before.fields[1](i, 0, 0), before.fields[0](i, 0, 0), before.fields[6](i, 0, 0),
            before.fields[7](i, 0, 0), before.fields[8](i, 0, 0),
            before.moments[0](i, 0, 0), before.moments[1](i, 0, 0)};
        const RainTripleMomentState initial{
            static_cast<float>(source.rain), static_cast<float>(source.rain_number), before.moments[6](i, 0, 0)};
        ExactTripleWarmCloudCellResult produced;
        std::string error;
        const bool accepted = produce_exact_triple_warm_cloud_cell(parameters, source,
            before.fields[4](i, 0, 0), duration, initial, produced, error);
        INFO("small-rain cell " << i << ", duration " << duration << ", " << error);
        REQUIRE(accepted);
        const auto& transfer = produced.warm.endpoints.reclassification;
        REQUIRE(transfer.present);
        REQUIRE(transfer.rain_mass_before_per_kg > 0);
        REQUIRE(transfer.rain_number_before_per_m3 > 0);
        const long double diameter = std::cbrt(static_cast<long double>(source.density)
            * transfer.rain_mass_before_per_kg
            / (transfer.rain_number_before_per_m3 * 523.598775598L));
        CHECK(diameter < 82e-6L);
        long double rain_mass = source.rain;
        long double rain_number = static_cast<long double>(source.density) * source.rain_number;
        long double cloud_mass = produced.cloud.endpoints().cloud_per_kg;
        long double cloud_number = static_cast<long double>(source.density)
            * produced.cloud.endpoints().cloud_number_per_kg;
        int transfer_count = 0;
        for (std::size_t index = 0; index < produced.warm.producer.events().size; ++index)
        {
            const auto& event = produced.warm.producer.events().events[index];
            CHECK(event.kind != WarmRainAcceptedEventKind::RainReset);
            CHECK(event.kind != WarmRainAcceptedEventKind::Evaporation);
            if (event.kind == WarmRainAcceptedEventKind::RainToCloud)
            {
                ++transfer_count;
                close(duration * static_cast<long double>(event.cloud_mass_rate_per_kg_s),
                    transfer.rain_mass_before_per_kg);
                close(-duration * static_cast<long double>(event.rain_mass_rate_per_kg_s),
                    transfer.rain_mass_before_per_kg);
                close(duration * static_cast<long double>(event.cloud_number_rate_per_m3_s),
                    transfer.rain_number_before_per_m3);
                close(-duration * static_cast<long double>(event.rain_number_rate_per_m3_s),
                    transfer.rain_number_before_per_m3);
                CHECK(event.vapor_mass_rate_per_kg_s == 0);
                CHECK(event.temperature_rate_k_s == 0);
            }
            else
            {
                rain_mass += duration * static_cast<long double>(event.rain_mass_rate_per_kg_s);
                rain_number += duration * static_cast<long double>(event.rain_number_rate_per_m3_s);
                cloud_mass += duration * static_cast<long double>(event.cloud_mass_rate_per_kg_s);
                cloud_number += duration * static_cast<long double>(event.cloud_number_rate_per_m3_s);
            }
        }
        CHECK(transfer_count == 1);
        close(rain_mass, transfer.rain_mass_before_per_kg);
        close(rain_number, transfer.rain_number_before_per_m3);
        close(cloud_mass, transfer.cloud_mass_before_per_kg);
        close(cloud_number, transfer.cloud_number_before_per_m3);
        CHECK(produced.endpoints.cloud_per_kg ==
            transfer.cloud_mass_before_per_kg + transfer.rain_mass_before_per_kg);
        CHECK(produced.endpoints.cloud_number_per_kg ==
            (transfer.cloud_number_before_per_m3 + transfer.rain_number_before_per_m3) / source.density);
        CHECK(produced.warm.endpoints.temperature_increment_k == 0);
        CHECK(produced.temperature_increment_k == produced.cloud.endpoints().temperature_increment_k);
        CHECK(produced.endpoints.rain_per_kg == 0);
        CHECK(produced.endpoints.rain_number_per_kg == 0);
        CHECK(produced.rain.mass_per_kg == 0);
        CHECK(produced.rain.number_per_kg == 0);
        CHECK(produced.rain.reflectivity_per_kg == 0);
        CHECK(qr(i, 0, 0) == 0);
        CHECK(scheme.internal_triple_number_for_test(0)(i, 0, 0) == 0);
        CHECK(scheme.internal_triple_reflectivity_for_test(0)(i, 0, 0) == 0);
        CHECK(qc(i, 0, 0) == static_cast<float>(cloud_mass + rain_mass));
        CHECK(scheme.internal_triple_cloud_number_for_test()(i, 0, 0)
            == static_cast<float>((cloud_number + rain_number) / source.density));
        close(static_cast<long double>(source.vapor) + source.cloud + source.rain,
            static_cast<long double>(produced.endpoints.vapor_per_kg) + produced.endpoints.cloud_per_kg);
        if (source.cloud == 0)
        {
            CHECK(produced.cloud.endpoints().cloud_per_kg == 0);
            CHECK(transfer.cloud_mass_before_per_kg == 0);
            CHECK(transfer.cloud_number_before_per_m3 == 0);
            CHECK(produced.temperature_increment_k == 0);
            CHECK(theta(i, 0, 0) == before.fields[5](i, 0, 0));
            CHECK(qv(i, 0, 0) == before.fields[6](i, 0, 0));
            CHECK(qc(i, 0, 0) > 0);
        }
    }
}

std::string small_rain_continuation_diagnostics(const Snapshot& before, double duration)
{
    std::ostringstream diagnostics;
    diagnostics << std::setprecision(17);
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    const auto physical = physical_scalar_domain(global_simulation_state().grid);
    int reset_cells = 0;
    for (int i = physical.i_begin; i < physical.i_end; ++i)
    {
        const DoubleWarmRainCell source{
            thermodynamics::theta_to_temperature(before.fields[5](i, 0, 0), before.fields[1](i, 0, 0)),
            before.fields[1](i, 0, 0), before.fields[0](i, 0, 0), before.fields[6](i, 0, 0),
            before.fields[7](i, 0, 0), before.fields[8](i, 0, 0),
            before.moments[0](i, 0, 0), before.moments[1](i, 0, 0)};
        TripleCloudAdjustmentResult cloud;
        std::string error;
        if (!produce_exact_triple_cloud_adjustment({source.temperature, source.pressure, source.density,
                source.vapor, source.cloud, source.cloud_number, before.fields[4](i, 0, 0)}, duration, cloud, error))
        {
            diagnostics << "cloud source rejected at cell " << i << ": " << error;
            continue;
        }
        const auto& adjusted = cloud.endpoints();
        const DoubleWarmRainCell warm_source{adjusted.temperature_after_k, source.pressure, source.density,
            adjusted.vapor_per_kg, adjusted.cloud_per_kg, source.rain,
            adjusted.cloud_number_per_kg, source.rain_number};
        TripleWarmRainDoubleSourceResult warm;
        triple_warm_rain_cell_exact_double_source(parameters, warm_source, duration, {}, warm);
        for (std::size_t index = 0; index < warm.events().size; ++index)
        {
            const auto& event = warm.events().events[index];
            if (event.kind != WarmRainAcceptedEventKind::RainReset)
            {
                continue;
            }
            ++reset_cells;
            if (reset_cells != 1)
            {
                continue;
            }
            const double rain_mass = -duration * event.rain_mass_rate_per_kg_s;
            const double rain_number = -duration * event.rain_number_rate_per_m3_s;
            diagnostics << "first RainReset at cell " << i << ", source T=" << source.temperature
                << ", p=" << source.pressure << ", rho=" << source.density << ", qv=" << source.vapor
                << ", qc=" << source.cloud << ", qr=" << source.rain << ", Nc=" << source.cloud_number
                << ", Nr=" << source.rain_number << ", Zr=" << before.moments[6](i, 0, 0)
                << ", admitted cloud qc=" << warm_source.cloud << ", Nc=" << warm_source.cloud_number
                << ", pre-reset qr=" << rain_mass << ", volumetric Nr=" << rain_number
                << ", mean mass diameter=" << std::cbrt(source.density * rain_mass / (523.598775598 * rain_number))
                << ", journal=";
            for (std::size_t entry = 0; entry < warm.events().size; ++entry)
            {
                const auto& admitted = warm.events().events[entry];
                diagnostics << " [kind " << static_cast<int>(admitted.kind)
                    << ", rain mass rate " << admitted.rain_mass_rate_per_kg_s
                    << ", rain number rate " << admitted.rain_number_rate_per_m3_s
                    << ", seed diameter " << admitted.seed_diameter_m << ']';
            }
        }
    }
    diagnostics << "; RainReset cells=" << reset_cells;
    return diagnostics.str();
}

} // namespace warm_cloud_live

TEST_CASE("composed live warm owner commits both unrounded stages and complete scientific ledgers",
          "[physics][milbrandt][triple_warm_cloud_internal][direct]")
{
    bool saw_new_rain = false;
    for (int scenario : {0,1,2,3})
    {
        for (double duration : {.05,.1,.3})
        {
            warm_cloud_live::initialize(scenario);
            const Snapshot before;
            INFO("composed scenario " << scenario << ", duration " << duration);
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            warm_primary_live::check_step(before, duration, false, true);
            CHECK(dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme).observed_velocity_source() == &w);
            for (int i = 1; i < NR - 1; ++i)
            {
                saw_new_rain |= before.fields[8](i,0,0) == 0 && qr(i,0,0) > 0;
            }
        }
    }
    CHECK(saw_new_rain);
}

TEST_CASE("composed warm continuation and actual workers keep identical state and moment owners",
          "[physics][milbrandt][triple_warm_cloud_internal][direct]")
{
#ifdef _OPENMP
    const int old_threads = omp_get_max_threads();
    const int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    std::array<std::unique_ptr<Snapshot>,2> after;
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        omp_set_num_threads(workers);
        int observed = 0;
#pragma omp parallel
        {
#pragma omp single
            observed = omp_get_num_threads();
        }
        REQUIRE(observed == workers);
        warm_cloud_live::initialize(2);
        for (double duration : {.05,.1,.3})
        {
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            warm_primary_live::check_step(before, duration, false, true);
            simulation_time += duration;
        }
        after[static_cast<std::size_t>(run)] = std::make_unique<Snapshot>();
    }
    check_snapshot(*after[0], *after[1]);
    omp_set_num_threads(old_threads);
    omp_set_dynamic(old_dynamic);
#else
    FAIL("composed warm worker evidence requires OpenMP");
#endif
}

TEST_CASE("composed warm records wind and late private failures roll back both stages and recover",
          "[physics][milbrandt][triple_warm_cloud_internal][direct]")
{
    for (int mutation : {1,2,3,12,14,17,18,19,20,21,25,26,27,28})
    {
        warm_cloud_live::initialize(2);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        INFO("composed getter mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .1));
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
        warm_primary_live::check_step(before,.1,false,true);
    }
    for (int mutation = 0; mutation < 6; ++mutation)
    {
        warm_cloud_live::initialize(2);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        INFO("composed late mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(),.1));
        CHECK(accepted_attempts == 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(),.1));
        warm_primary_live::check_step(before,.1,false,true);
    }
}

TEST_CASE("composed warm zero intervals still reject unowned state selectors and bad source profiles",
          "[physics][milbrandt][triple_warm_cloud_internal][direct]")
{
    for (int mutation = 0; mutation < 10; ++mutation)
    {
        warm_cloud_live::initialize(2);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        qv.fill(0); qc.fill(0); qr.fill(0);
        const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test()).fill(0);
        const_cast<Field3D&>(scheme.internal_triple_number_for_test(0)).fill(0);
        const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(0)).fill(0);
        const int last = NR - 2;
        switch (mutation)
        {
            case 0: w(last,0,0) = std::numeric_limits<float>::quiet_NaN(); break;
            case 1: rho(last,0,0) = 0; break;
            case 2: p(last,0,0) = 0; break;
            case 3: qc(last,0,0) = .001f; break;
            case 4: qi(last,0,0) = .001f; break;
            case 5: const_cast<Field3D&>(scheme.internal_triple_number_for_test(2))(last,0,0) = 1; break;
            case 6: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
            case 7: global_validation_policy.mode = tmv::GuardMode::Warn; break;
            case 8: dtheta_dt_rad(last,0,0) = 1; break;
            case 9: qv(last,0,0) = -1; break;
        }
        const Snapshot before;
        INFO("composed zero mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .1));
        CHECK(getter_attempts == 0);
        check_snapshot(before);
    }
    warm_cloud_live::initialize(2);
    CHECK_THROWS(microphysics_scheme->portable_state());
    CHECK_THROWS(microphysics_scheme->clone_parameters());
    CHECK_THROWS(microphysics_scheme->prognostic_scalars_for_transport());
}

TEST_CASE("composed warm small rain transfers its complete mass and number without cloud reflectivity ownership",
          "[physics][milbrandt][triple_warm_cloud_internal][triple_rain_reclassification_internal][direct]")
{
    for (bool cloud_present : {false, true})
    {
        for (long double alpha : {0.0L, 2.0L, 20.0L})
        {
            for (double duration : {.05, .1, .3})
            {
                warm_cloud_live::initialize_small_rain(cloud_present, alpha);
                const Snapshot before;
                INFO("small-rain cloud " << cloud_present << ", alpha " << alpha << ", duration " << duration);
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                warm_primary_live::check_step(before, duration, false, true);
                warm_cloud_live::check_small_rain_transfer(before, duration);
            }
        }
    }
}

TEST_CASE("composed warm small rain continuation is exact across actual one and four workers",
          "[physics][milbrandt][triple_warm_cloud_internal][triple_rain_reclassification_internal][direct]")
{
#ifdef _OPENMP
    const int old_threads = omp_get_max_threads();
    const int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    for (bool cloud_present : {false, true})
    {
        std::array<std::unique_ptr<Snapshot>, 2> after;
        for (int run = 0; run < 2; ++run)
        {
            bool saw_subthreshold_rain = false;
            const int workers = run == 0 ? 1 : 4;
            omp_set_num_threads(workers);
            int observed = 0;
#pragma omp parallel
            {
#pragma omp single
                observed = omp_get_num_threads();
            }
            REQUIRE(observed == workers);
            warm_cloud_live::initialize_small_rain(cloud_present, 20);
            for (double duration : {.05, .1, .3})
            {
                const Snapshot before;
                INFO("small-rain continuation cloud " << cloud_present << ", workers " << workers
                    << ", elapsed " << simulation_time << ", duration " << duration);
                INFO(warm_cloud_live::small_rain_continuation_diagnostics(before, duration));
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                warm_primary_live::check_step(before, duration, false, true);
                if (simulation_time == 0)
                {
                    warm_cloud_live::check_small_rain_transfer(before, duration);
                }
                const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
                for (int i = 1; i < NR - 1; ++i)
                {
                    const double mass = qr(i, 0, 0);
                    const double number = scheme.internal_triple_number_for_test(0)(i, 0, 0);
                    const double reflectivity = scheme.internal_triple_reflectivity_for_test(0)(i, 0, 0);
                    if (simulation_time == 0 || mass == 0)
                    {
                        CHECK(mass == 0);
                        CHECK(number == 0);
                        CHECK(reflectivity == 0);
                    }
                    else
                    {
                        REQUIRE(mass > 0);
                        REQUIRE(number > 0);
                        REQUIRE(reflectivity > 0);
                        double alpha = 0;
                        double lambda = 0;
                        std::string error;
                        const bool valid = diagnose_triple_gamma_shape(mass, number, reflectivity,
                            current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Rain).mass_law,
                            alpha, lambda, error);
                        INFO("continued rain q/N/Z=" << mass << '/' << number << '/' << reflectivity << ": " << error);
                        REQUIRE(valid);
                        CHECK(alpha > -1);
                        CHECK(alpha <= 200);
                        CHECK(std::isfinite(lambda));
                        CHECK(lambda > 0);
                        saw_subthreshold_rain = saw_subthreshold_rain || mass < 1e-14
                            || static_cast<double>(rho(i, 0, 0)) * number < 1e-3;
                    }
                }
                simulation_time += duration;
            }
            if (!cloud_present)
            {
                CHECK(saw_subthreshold_rain);
            }
            after[static_cast<std::size_t>(run)] = std::make_unique<Snapshot>();
        }
        check_snapshot(*after[0], *after[1]);
    }
    omp_set_num_threads(old_threads);
    omp_set_dynamic(old_dynamic);
#else
    FAIL("small-rain composed warm worker evidence requires OpenMP");
#endif
}

TEST_CASE("composed warm small rain returned records and late owners roll back transfer and recover",
          "[physics][milbrandt][triple_warm_cloud_internal][triple_rain_reclassification_internal][direct]")
{
    for (int mutation : {15, 21, 25, 26, 27, 28})
    {
        warm_cloud_live::initialize_small_rain(true, 2);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        INFO("small-rain getter mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .1));
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
        warm_primary_live::check_step(before, .1, false, true);
        warm_cloud_live::check_small_rain_transfer(before, .1);
    }
    for (int mutation = 0; mutation < 6; ++mutation)
    {
        warm_cloud_live::initialize_small_rain(false, 20);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        INFO("small-rain late mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .1));
        CHECK(accepted_attempts == 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
        warm_primary_live::check_step(before, .1, false, true);
        warm_cloud_live::check_small_rain_transfer(before, .1);
    }
}

namespace dynamic_warm_cloud_live
{

void initialize(int scenario = 1, long double alpha = 2)
{
    ice_melting_live::initialize(false, 0, 0, 2, 290.125);
    const float rain_mass = scenario == 0 ? 0 : scenario == 2 ? .002f : scenario == 4 ? .003f : 1e-4f;
    const float rain_number = scenario == 0 ? 0 : scenario == 2 ? 100.0f : scenario == 4 ? 10.0f : 1e3f;
    Field3D nc(NR, NTH, NZ, scenario == 3 || scenario == 4 ? 0 : 1e5f);
    Field3D nr(NR, NTH, NZ, rain_number);
    Field3D zr(NR, NTH, NZ, 0);
    if (rain_mass > 0)
    {
        const long double exact_z = shape_ratio(alpha) * rain_mass * rain_mass
            / (523.598775598L * 523.598775598L * rain_number);
        float represented_z = static_cast<float>(exact_z);
        if (alpha == 200 && static_cast<long double>(represented_z) < exact_z)
        {
            represented_z = std::nextafter(represented_z, std::numeric_limits<float>::infinity());
        }
        zr.fill(represented_z);
        double represented_alpha = 0;
        double represented_lambda = 0;
        std::string error;
        const bool valid = diagnose_triple_gamma_shape(rain_mass, rain_number, represented_z,
            current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Rain).mass_law,
            represented_alpha, represented_lambda, error);
        INFO("dynamic represented source alpha " << represented_alpha << ", requested " << alpha << ": " << error);
        REQUIRE(valid);
        CHECK(represented_alpha <= 200);
        CHECK(represented_alpha == Approx(static_cast<double>(alpha)).epsilon(2e-5).margin(2e-6));
    }
    for (int i = 0; i < NR; ++i)
    {
        const double temperature = thermodynamics::theta_to_temperature(theta(i, 0, 0), p(i, 0, 0));
        const double saturation = thermodynamics::saturation_mixing_ratio_water(temperature, p(i, 0, 0));
        const double relative_humidity = scenario == 0 ? 1.2 : scenario == 1 ? 1.05 : .4;
        qc(i, 0, 0) = scenario == 0 ? .03f : scenario == 1 ? .001f : scenario == 2 ? .01f : 0;
        qv(i, 0, 0) = static_cast<float>(relative_humidity * saturation);
        qr(i, 0, 0) = rain_mass;
        w(i, 0, 0) = scenario == 3 || scenario == 4 ? 0 : .5f;
    }
    auto scheme = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = true;
    scheme->configure(config);
    std::string error;
    const bool activated = scheme->enable_internal_triple_dynamic_warm_cloud_primary_for_test(
        global_simulation_state(), nc, nr, zr, error);
    INFO(error);
    REQUIRE(activated);
    REQUIRE(scheme->owns_primary_endpoint_commit());
    REQUIRE(scheme->protected_primary_endpoint_wind_source() == &w);
    microphysics_scheme = std::move(scheme);
    global_microphysics_runtime_state.theta_storage_residual.resize(NR, NTH, NZ, .0125f);
}

ExactTripleDynamicWarmCloudCellResult replay(const Snapshot& before, int i, double duration)
{
    const DoubleWarmRainCell source{
        thermodynamics::theta_to_temperature(before.fields[5](i, 0, 0), before.fields[1](i, 0, 0)),
        before.fields[1](i, 0, 0), before.fields[0](i, 0, 0), before.fields[6](i, 0, 0),
        before.fields[7](i, 0, 0), before.fields[8](i, 0, 0),
        before.moments[0](i, 0, 0), before.moments[1](i, 0, 0)};
    const RainTripleMomentState initial{
        static_cast<float>(source.rain), static_cast<float>(source.rain_number), before.moments[6](i, 0, 0)};
    ExactTripleDynamicWarmCloudCellResult result;
    std::string error;
    const bool accepted = produce_exact_triple_dynamic_warm_cloud_cell(
        warm_rain_parameters(0, 523.598775598, 3, 149.1, .5), source,
        before.fields[4](i, 0, 0), duration, initial, result, error);
    INFO("dynamic replay cell " << i << ", duration " << duration << ": " << error);
    REQUIRE(accepted);
    return result;
}

void check_phase_sources(const ExactTripleDynamicWarmCloudCellResult& result)
{
    const auto& dynamic = result.warm.producer;
    const auto& producer = dynamic.producer();
    CHECK(producer.uses_dynamic_spectrum());
    CHECK(producer.source_cell().cloud == result.cloud.endpoints().cloud_per_kg);
    CHECK(producer.source_cell().cloud_number == result.cloud.endpoints().cloud_number_per_kg);
    CHECK(producer.source_cell().temperature == result.cloud.endpoints().temperature_after_k);
    CHECK(producer.source_cell().vapor == result.cloud.endpoints().vapor_per_kg);
    CHECK(dynamic.pre_physical_event_count() <= dynamic.pre_evaporation_event_count());
    CHECK(dynamic.pre_evaporation_event_count() <= producer.events().size);
    const auto check = [](const TripleWarmRainSpectrum* spectrum, const ExactRainTripleMomentState& owner)
    {
        if (owner.mass_per_kg > 0)
        {
            REQUIRE(spectrum != nullptr);
            CHECK(spectrum->source().mass_per_kg == owner.mass_per_kg);
            CHECK(spectrum->source().number_per_kg == owner.number_per_kg);
            CHECK(spectrum->source().reflectivity_per_kg == owner.reflectivity_per_kg);
            CHECK(spectrum->alpha() > -1);
            CHECK(spectrum->alpha() <= 200);
            CHECK(spectrum->lambda_per_m() > 0);
        }
        else
        {
            CHECK(spectrum == nullptr);
        }
    };
    check(dynamic.pre_physical_spectrum(), dynamic.pre_physical_rain());
    check(dynamic.pre_evaporation_spectrum(), dynamic.pre_evaporation_rain());
}

struct BreakupEulerBounds
{
    double minimum_positive_reflectivity_duration_s = std::numeric_limits<double>::infinity();
    double minimum_admissible_duration_s = std::numeric_limits<double>::infinity();
    double maximum_relative_number_rate_s = 0;
    bool negative_reflectivity = false;
    bool unrealizable_moments = false;
    std::string diagnostic;
};

BreakupEulerBounds breakup_euler_bounds(const Snapshot& before, double duration)
{
    constexpr double coefficient = 523.598775598;
    BreakupEulerBounds result;
    std::ostringstream diagnostic;
    diagnostic << std::setprecision(17);
    const auto physical = physical_scalar_domain(global_simulation_state().grid);
    for (int i = physical.i_begin; i < physical.i_end; ++i)
    {
        REQUIRE(before.fields[7](i, 0, 0) == 0);
        REQUIRE(before.moments[0](i, 0, 0) == 0);
        const double density = before.fields[0](i, 0, 0);
        const double mass = before.fields[8](i, 0, 0);
        const double number = before.moments[1](i, 0, 0);
        const double reflectivity = before.moments[6](i, 0, 0);
        REQUIRE(mass > 0);
        REQUIRE(number > 0);
        REQUIRE(reflectivity > 0);
        const double diameter = std::cbrt(mass / (coefficient * number));
        const double corrected_number = diameter > .003
            ? number * std::max(1 + 2e4 * (diameter - .003) * (diameter - .003),
                                std::pow(diameter / .005, 3))
            : number;
        const double corrected_diameter = std::cbrt(mass / (coefficient * corrected_number));
        const double equivalent_inverse_slope = corrected_diameter / std::cbrt(6.0);
        const double efficiency = equivalent_inverse_slope > 300e-6
            ? 2 - std::exp(2300 * (equivalent_inverse_slope - 300e-6)) : 1;
        const double relative_number_rate = -efficiency * 3030 * density * mass / coefficient;
        const double corrected_reflectivity = reflectivity * number / corrected_number;
        const double next_number = corrected_number * (1 + duration * relative_number_rate);
        const double next_reflectivity = corrected_reflectivity * (1 - duration * relative_number_rate);
        const double source_ratio = coefficient * coefficient * corrected_number * corrected_reflectivity / (mass * mass);
        const double next_ratio = coefficient * coefficient * next_number * next_reflectivity / (mass * mass);
        const double bound = relative_number_rate > 0 ? 1 / relative_number_rate
            : std::numeric_limits<double>::infinity();
        const double cauchy_bound = relative_number_rate > 0
            ? std::sqrt(std::max(0.0, 1 - 1 / source_ratio)) / relative_number_rate : bound;
        const double gamma_bound = relative_number_rate > 0
            ? std::sqrt(std::max(0.0, 1 - static_cast<double>(shape_ratio(200)) / source_ratio)) / relative_number_rate : bound;
        result.minimum_positive_reflectivity_duration_s = std::min(result.minimum_positive_reflectivity_duration_s, bound);
        result.minimum_admissible_duration_s = std::min(result.minimum_admissible_duration_s,
                                                       std::min({bound, cauchy_bound, gamma_bound}));
        result.maximum_relative_number_rate_s = std::max(result.maximum_relative_number_rate_s, relative_number_rate);
        if (next_reflectivity <= 0)
        {
            result.negative_reflectivity = true;
        }
        result.unrealizable_moments |= next_reflectivity <= 0 || next_ratio < 1;
        if (i <= 2)
        {
            diagnostic << " cell=" << i << " rho=" << density << " q/N/Z=" << mass << '/' << number << '/' << reflectivity
                << " pre-N/Z=" << corrected_number << '/' << corrected_reflectivity
                << " pre-D=" << corrected_diameter << " inverse-slope=" << equivalent_inverse_slope
                << " efficiency=" << efficiency << " relative-breakup-rate=" << relative_number_rate
                << " positive-Z-dt-bound=" << bound << " Cauchy-dt-bound=" << cauchy_bound
                << " alpha200-dt-bound=" << gamma_bound << " dt=" << duration
                << " B3-q/N/Z=" << mass << '/' << next_number << '/' << next_reflectivity
                << " G-before/after=" << source_ratio << '/' << next_ratio << ';';
        }
    }
    result.diagnostic = diagnostic.str();
    return result;
}

} // namespace dynamic_warm_cloud_live

TEST_CASE("dynamic composed warm owner uses phase-bound spectra and complete scientific ledgers",
          "[physics][milbrandt][triple_dynamic_warm_cloud_internal][direct]")
{
    std::array<bool, 16> observed_events{};
    bool saw_new_rain = false;
    bool saw_rediagnosis = false;
    for (int scenario : {0, 1, 2, 3, 4})
    {
        for (long double alpha : {0.0L, 2.0L, 20.0L})
        {
            const std::array<double, 3> intervals = scenario == 4
                ? std::array<double, 3>{.025, .05, .1} : std::array<double, 3>{.05, .1, .3};
            for (double requested_duration : intervals)
            {
                dynamic_warm_cloud_live::initialize(scenario, alpha);
                const Snapshot before;
                double duration = requested_duration;
                INFO("dynamic scenario " << scenario << ", alpha " << alpha << ", duration " << duration);
                if (scenario == 4)
                {
                    const auto requested_bounds = dynamic_warm_cloud_live::breakup_euler_bounds(before, duration);
                    duration = std::min(duration, .5 * requested_bounds.minimum_admissible_duration_s);
                    const auto bounds = dynamic_warm_cloud_live::breakup_euler_bounds(before, duration);
                    INFO(bounds.diagnostic);
                    REQUIRE_FALSE(bounds.negative_reflectivity);
                    REQUIRE_FALSE(bounds.unrealizable_moments);
                    REQUIRE(duration > 0);
                    REQUIRE(duration < bounds.minimum_admissible_duration_s);
                }
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                warm_primary_live::check_step(before, duration, false, false, true);
                for (int i = 1; i < NR - 1; ++i)
                {
                    const auto produced = dynamic_warm_cloud_live::replay(before, i, duration);
                    dynamic_warm_cloud_live::check_phase_sources(produced);
                    const auto& dynamic = produced.warm.producer;
                    for (std::size_t entry = 0; entry < dynamic.producer().events().size; ++entry)
                    {
                        observed_events[static_cast<std::size_t>(dynamic.producer().events().events[entry].kind)] = true;
                    }
                    if (dynamic.pre_physical_spectrum() && dynamic.pre_evaporation_spectrum())
                    {
                        saw_rediagnosis |= dynamic.pre_physical_spectrum()->alpha()
                            != dynamic.pre_evaporation_spectrum()->alpha();
                    }
                    saw_new_rain |= before.fields[8](i, 0, 0) == 0 && qr(i, 0, 0) > 0;
                }
            }
        }
    }
    for (auto kind : {WarmRainAcceptedEventKind::Autoconversion, WarmRainAcceptedEventKind::Accretion,
                      WarmRainAcceptedEventKind::CloudSelfCollection, WarmRainAcceptedEventKind::RainSelfCollection,
                      WarmRainAcceptedEventKind::PreRainNumberRegularization, WarmRainAcceptedEventKind::Evaporation,
                      WarmRainAcceptedEventKind::RainBreakup})
    {
        INFO("dynamic required event " << static_cast<int>(kind));
        CHECK(observed_events[static_cast<std::size_t>(kind)]);
    }
    CHECK(saw_new_rain);
    CHECK(saw_rediagnosis);
    for (double duration : {.05, .1, .3})
    {
        dynamic_warm_cloud_live::initialize(1, 200);
        const Snapshot before;
        INFO("dynamic shape-200 represented accretion source, duration " << duration);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
        warm_primary_live::check_step(before, duration, false, false, true);
        dynamic_warm_cloud_live::check_phase_sources(dynamic_warm_cloud_live::replay(before, 1, duration));
    }
}

TEST_CASE("dynamic composed warm continuation is exact across actual one and four workers",
          "[physics][milbrandt][triple_dynamic_warm_cloud_internal][direct]")
{
#ifdef _OPENMP
    const int old_threads = omp_get_max_threads();
    const int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    for (int scenario : {0, 1, 2, 3, 4})
    {
        std::array<std::unique_ptr<Snapshot>, 2> after;
        for (int run = 0; run < 2; ++run)
        {
            const int workers = run == 0 ? 1 : 4;
            omp_set_num_threads(workers);
            int observed = 0;
#pragma omp parallel
            {
#pragma omp single
                observed = omp_get_num_threads();
            }
            REQUIRE(observed == workers);
            dynamic_warm_cloud_live::initialize(scenario, 20);
            const std::array<double, 3> intervals = scenario == 4
                ? std::array<double, 3>{.025, .05, .1} : std::array<double, 3>{.05, .1, .3};
            for (double requested_duration : intervals)
            {
                const Snapshot before;
                double duration = requested_duration;
                if (scenario == 4)
                {
                    const auto initial_bounds = dynamic_warm_cloud_live::breakup_euler_bounds(before, duration);
                    duration = std::min(duration, .5 * initial_bounds.minimum_admissible_duration_s);
                    const auto admitted_bounds = dynamic_warm_cloud_live::breakup_euler_bounds(before, duration);
                    INFO(admitted_bounds.diagnostic);
                    REQUIRE_FALSE(admitted_bounds.negative_reflectivity);
                    REQUIRE_FALSE(admitted_bounds.unrealizable_moments);
                    REQUIRE(duration > 0);
                    REQUIRE(duration < admitted_bounds.minimum_admissible_duration_s);
                }
                INFO("dynamic continuation scenario " << scenario << ", workers " << workers
                    << ", elapsed " << simulation_time << ", duration " << duration);
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                warm_primary_live::check_step(before, duration, false, false, true);
                simulation_time += duration;
            }
            after[static_cast<std::size_t>(run)] = std::make_unique<Snapshot>();
        }
        check_snapshot(*after[0], *after[1]);
    }
    omp_set_num_threads(old_threads);
    omp_set_dynamic(old_dynamic);
#else
    FAIL("dynamic composed warm worker evidence requires OpenMP");
#endif
}

TEST_CASE("dynamic heavy breakup rejects its original unrealizable intervals atomically and recovers",
          "[physics][milbrandt][triple_dynamic_warm_cloud_internal][direct]")
{
    for (bool continued : {false, true})
    {
        dynamic_warm_cloud_live::initialize(4, continued ? 20 : 0);
        if (continued)
        {
            for (double duration : {.05, .1})
            {
                const Snapshot before;
                const auto bounds = dynamic_warm_cloud_live::breakup_euler_bounds(before, duration);
                INFO(bounds.diagnostic);
                REQUIRE_FALSE(bounds.negative_reflectivity);
                REQUIRE_FALSE(bounds.unrealizable_moments);
                REQUIRE(duration < bounds.minimum_admissible_duration_s);
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                warm_primary_live::check_step(before, duration, false, false, true);
                simulation_time += duration;
            }
        }
        const Snapshot before;
        const auto invalid_bounds = dynamic_warm_cloud_live::breakup_euler_bounds(before, .3);
        INFO("original heavy-breakup rejection after continuation=" << continued << ':' << invalid_bounds.diagnostic);
        std::cout << "Retained heavy-breakup scientific rejection, continued=" << continued
                  << ':' << invalid_bounds.diagnostic << '\n';
        REQUIRE(invalid_bounds.unrealizable_moments);
        REQUIRE(.3 > invalid_bounds.minimum_admissible_duration_s);
        const int old_getters = getter_attempts;
        const int old_accepts = accepted_attempts;
        REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), .3),
            Catch::Matchers::Contains("simultaneous rain physical phase is unrealizable"));
        CHECK(getter_attempts == old_getters);
        CHECK(accepted_attempts == old_accepts);
        check_snapshot(before);
        const double recovery_duration = std::min(.05, .5 * invalid_bounds.minimum_admissible_duration_s);
        const auto recovery_bounds = dynamic_warm_cloud_live::breakup_euler_bounds(before, recovery_duration);
        INFO(recovery_bounds.diagnostic);
        REQUIRE_FALSE(recovery_bounds.negative_reflectivity);
        REQUIRE_FALSE(recovery_bounds.unrealizable_moments);
        REQUIRE(recovery_duration > 0);
        REQUIRE(recovery_duration < recovery_bounds.minimum_admissible_duration_s);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), recovery_duration));
        warm_primary_live::check_step(before, recovery_duration, false, false, true);
    }
}

TEST_CASE("dynamic composed warm record source wind and late private failures roll back and recover",
          "[physics][milbrandt][triple_dynamic_warm_cloud_internal][direct]")
{
    for (int mutation = 1; mutation <= 28; ++mutation)
    {
        dynamic_warm_cloud_live::initialize();
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        INFO("dynamic getter mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .1));
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
        warm_primary_live::check_step(before, .1, false, false, true);
    }
    for (int mutation = 0; mutation < 6; ++mutation)
    {
        dynamic_warm_cloud_live::initialize();
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        INFO("dynamic late mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .1));
        CHECK(accepted_attempts == 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
        warm_primary_live::check_step(before, .1, false, false, true);
    }
}

TEST_CASE("dynamic composed warm zero events still validate all owners and unsupported selectors",
          "[physics][milbrandt][triple_dynamic_warm_cloud_internal][direct]")
{
    for (int mutation = 0; mutation < 43; ++mutation)
    {
        dynamic_warm_cloud_live::initialize();
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        qv.fill(0); qc.fill(0); qr.fill(0);
        const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test()).fill(0);
        const_cast<Field3D&>(scheme.internal_triple_number_for_test(0)).fill(0);
        const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(0)).fill(0);
        const int last = NR - 2;
        if (mutation >= 20 && mutation < 32)
        {
            auto* field = held_fields()[mutation - 20];
            field->resize(NR, NTH, NZ, 0);
            field->operator()(last, 0, 0) = 1;
        }
        else if (mutation >= 32 && mutation < 39)
        {
            *held_radiation_powers()[mutation - 32] = 1;
        }
        else
        {
            switch (mutation)
            {
                case 0: rho(last, 0, 0) = 0; break;
                case 1: p(last, 0, 0) = 0; break;
                case 2: theta(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 3: qv(last, 0, 0) = -1; break;
                case 4: qc(last, 0, 0) = 1e-4f; break;
                case 5: qi(last, 0, 0) = 1e-4f; break;
                case 6: qs(last, 0, 0) = 1e-4f; break;
                case 7: qg(last, 0, 0) = 1e-4f; break;
                case 8: qh(last, 0, 0) = 1e-4f; break;
                case 9: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                case 10: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
                case 11: global_simulation_storage.runtime.compute.fallback_active = true; break;
                case 12: global_validation_policy.mode = tmv::GuardMode::Warn; break;
                case 13: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = -1; break;
                case 14: const_cast<Field3D&>(scheme.internal_triple_number_for_test(0))(last, 0, 0) = 1; break;
                case 15: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(0))(last, 0, 0) = 1e-9f; break;
                case 16: const_cast<Field3D&>(scheme.internal_triple_number_for_test(1))(last, 0, 0) = 1; break;
                case 17: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(2))(last, 0, 0) = 1e-9f; break;
                case 18: dtheta_dt_rad(last, 0, 0) = 1; break;
                case 19: global_simulation_storage.runtime.chaos.noise_elapsed_s = 1; break;
                case 39: w(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 40: theta(last, 0, 0) = static_cast<float>(401 * std::pow(microphysics_constants::p0 / p(last, 0, 0),
                    microphysics_constants::R_d / microphysics_constants::cp)); break;
                case 41: global_simulation_storage.runtime.terrain.earth_fixed_placement.active = true; break;
                case 42: dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).wrong_wind_declaration = true; break;
            }
        }
        const Snapshot before;
        INFO("dynamic zero-event mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .1));
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
    }
    dynamic_warm_cloud_live::initialize();
    CHECK_THROWS(microphysics_scheme->portable_state());
    CHECK_THROWS(microphysics_scheme->clone_parameters());
    CHECK_THROWS(microphysics_scheme->prognostic_scalars_for_transport());
}

TEST_CASE("dynamic composed warm generations clones and detached sources cannot transplant a primary record",
          "[physics][milbrandt][triple_dynamic_warm_cloud_internal][direct]")
{
    dynamic_warm_cloud_live::initialize();
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    std::string error;
    tmv::MicrophysicsPrimaryEndpointCandidate absent;
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, absent, error));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    auto compute = [&](std::array<Field3D, 8>& out, const Field3D& cloud)
    {
        scheme.compute_tendencies(p, theta, rho, qv, cloud, qr, qi, qs, qg, qh, .05,
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    };
    std::array<Field3D, 8> out;
    REQUIRE_NOTHROW(compute(out, qc));
    tmv::MicrophysicsPrimaryEndpointCandidate first;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    auto clone = scheme.clone();
    CHECK_FALSE(clone->validate_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    REQUIRE_NOTHROW(compute(out, qc));
    tmv::MicrophysicsPrimaryEndpointCandidate second;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, second, error));
    CHECK(second.generation > first.generation);
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    const Field3D detached(qc);
    for (auto& field : out)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    CHECK_THROWS(compute(out, detached));
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, absent, error));
    for (const auto& field : out)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    const Snapshot before;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
    warm_primary_live::check_step(before, .05, false, false, true);
    CHECK_FALSE(dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme)
        .validate_primary_endpoint_commit(global_simulation_state(), .05, second, error));
}

namespace subcycled_warm_cloud_live
{

Field3D bounded_field(const Field3D& source)
{
    if (source.size() == 0)
    {
        return source;
    }
    REQUIRE(source.size_r() >= 10);
    REQUIRE(source.size_th() == 1);
    REQUIRE(source.size_z() == 1);
    Field3D result(10, 1, 1, 0);
    for (int i = 0; i < 10; ++i)
    {
        result(i, 0, 0) = source(i, 0, 0);
    }
    return result;
}

void activate_current_source()
{
    const auto& previous = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    const Field3D nc = bounded_field(previous.internal_triple_cloud_number_for_test());
    const Field3D nr = bounded_field(previous.internal_triple_number_for_test(0));
    const Field3D zr = bounded_field(previous.internal_triple_reflectivity_for_test(0));
    for (auto* field : public_fields())
    {
        *field = bounded_field(*field);
    }
    for (auto* field : source_fields())
    {
        *field = bounded_field(*field);
    }
    for (auto* field : held_fields())
    {
        *field = bounded_field(*field);
    }
    auto& tendencies = global_microphysics_runtime_state.tendencies;
    for (Field3D* field : {&tendencies.dtheta_dt, &tendencies.dqv_dt, &tendencies.dqc_dt,
                          &tendencies.dqr_dt, &tendencies.dqi_dt, &tendencies.dqs_dt,
                          &tendencies.dqg_dt, &tendencies.dqh_dt,
                          &global_microphysics_runtime_state.theta_storage_residual})
    {
        *field = bounded_field(*field);
    }
    NR = 10;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    global_simulation_storage.runtime.advection.nr = NR;
    global_simulation_storage.surface.precipitation_rate_mm_h.resize(NR);
    global_simulation_storage.surface.accumulated_rainfall_mm.resize(NR);
    global_simulation_storage.surface.accumulated_rainfall_residual_mm.resize(NR);
    auto scheme = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = true;
    scheme->configure(config);
    std::string error;
    const bool activated = scheme->enable_internal_triple_subcycled_warm_cloud_primary_for_test(
        global_simulation_state(), nc, nr, zr, error);
    INFO(error);
    REQUIRE(activated);
    REQUIRE(scheme->owns_primary_endpoint_commit());
    REQUIRE(scheme->protected_primary_endpoint_wind_source() == &w);
    microphysics_scheme = std::move(scheme);
}

void initialize(int scenario = 1, long double alpha = 2)
{
    dynamic_warm_cloud_live::initialize(scenario, alpha);
    activate_current_source();
}

void check_admitted_rain()
{
    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    for (int i = 1; i < NR - 1; ++i)
    {
        const double mass = qr(i, 0, 0);
        const double number = scheme.internal_triple_number_for_test(0)(i, 0, 0);
        const double reflectivity = scheme.internal_triple_reflectivity_for_test(0)(i, 0, 0);
        if (mass == 0)
        {
            CHECK(number == 0);
            CHECK(reflectivity == 0);
        }
        else
        {
            REQUIRE(number > 0);
            REQUIRE(reflectivity > 0);
            const long double ratio = 523.598775598L * 523.598775598L * number * reflectivity
                / (static_cast<long double>(mass) * mass);
            CHECK(ratio >= shape_ratio(200) * (1 - 8e-7L));
            CHECK(ratio > 1);
            double alpha = 0;
            double lambda = 0;
            std::string error;
            const bool valid = diagnose_triple_gamma_shape(mass, number, reflectivity,
                current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Rain).mass_law,
                alpha, lambda, error);
            INFO(error);
            REQUIRE(valid);
            CHECK(alpha > -1);
            CHECK(alpha <= 200);
            CHECK(std::isfinite(lambda));
            CHECK(lambda > 0);
        }
    }
}

} // namespace subcycled_warm_cloud_live

TEST_CASE("subcycled composed warm admits the original two heavy breakup intervals with exact ledgers",
          "[physics][milbrandt][triple_subcycled_warm_cloud_internal][direct]")
{
    for (bool continued : {false, true})
    {
        dynamic_warm_cloud_live::initialize(4, continued ? 20 : 0);
        if (continued)
        {
            for (double duration : {.05, .1})
            {
                const Snapshot before;
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                warm_primary_live::check_step(before, duration, false, false, true);
                simulation_time += duration;
            }
        }
        const Snapshot original;
        const auto bounds = dynamic_warm_cloud_live::breakup_euler_bounds(original, .3);
        INFO("retained original heavy source, continued=" << continued << ':' << bounds.diagnostic);
        REQUIRE(bounds.unrealizable_moments);
        REQUIRE(.3 > bounds.minimum_admissible_duration_s);
        subcycled_warm_cloud_live::activate_current_source();
        const Snapshot before;
        REQUIRE(NR == 10);
        for (int i = 1; i <= 4; ++i)
        {
            INFO("unchanged original heavy source class " << i << ", continued=" << continued);
            CHECK(before.fields[0](i, 0, 0) == (i % 2 ? .65f : 1.3f));
            CHECK(before.fields[1](i, 0, 0) == (i % 4 < 2 ? 90000.0f : 70000.0f));
            for (std::size_t field = 0; field < before.fields.size(); ++field)
            {
                CHECK(before.fields[field](i, 0, 0) == original.fields[field](i, 0, 0));
            }
            for (std::size_t moment = 0; moment < before.moments.size(); ++moment)
            {
                CHECK(before.moments[moment](i, 0, 0) == original.moments[moment](i, 0, 0));
            }
        }
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .3));
        warm_primary_live::check_step(before, .3, false, false, false, true);
        subcycled_warm_cloud_live::check_admitted_rain();
        const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
        for (int i = 1; i < NR - 1; ++i)
        {
            CHECK(scheme.internal_triple_number_for_test(0)(i, 0, 0)
                > before.moments[1](i, 0, 0));
        }
    }
}

TEST_CASE("subcycled composed warm continuation is identical with actual one and four workers",
          "[physics][milbrandt][triple_subcycled_warm_cloud_internal][direct]")
{
#ifdef _OPENMP
    const int old_threads = omp_get_max_threads();
    const int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    std::array<std::unique_ptr<Snapshot>, 2> after;
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        omp_set_num_threads(workers);
        int observed = 0;
#pragma omp parallel
        {
#pragma omp single
            observed = omp_get_num_threads();
        }
        REQUIRE(observed == workers);
        subcycled_warm_cloud_live::initialize(2, 20);
        for (double duration : {.05, .1, .3})
        {
            const Snapshot before;
            INFO("subcycled continuation workers=" << workers << ", elapsed=" << simulation_time
                << ", duration=" << duration);
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            warm_primary_live::check_step(before, duration, false, false, false, true);
            subcycled_warm_cloud_live::check_admitted_rain();
            simulation_time += duration;
        }
        after[static_cast<std::size_t>(run)] = std::make_unique<Snapshot>();
    }
    check_snapshot(*after[0], *after[1]);
    omp_set_num_threads(old_threads);
    omp_set_dynamic(old_dynamic);
#else
    FAIL("subcycled composed warm worker evidence requires OpenMP");
#endif
}

TEST_CASE("subcycled composed warm source record wind and late failures roll back and recover",
          "[physics][milbrandt][triple_subcycled_warm_cloud_internal][direct]")
{
    for (int mutation : {1, 15, 21, 25, 27, 28})
    {
        subcycled_warm_cloud_live::initialize();
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        INFO("subcycled getter mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .05));
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
        warm_primary_live::check_step(before, .05, false, false, false, true);
    }
    for (int mutation = 0; mutation < 6; ++mutation)
    {
        subcycled_warm_cloud_live::initialize();
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        INFO("subcycled late mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .05));
        CHECK(accepted_attempts == 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
        warm_primary_live::check_step(before, .05, false, false, false, true);
    }
}

TEST_CASE("subcycled zero events validate all state owners and unsupported profiles",
          "[physics][milbrandt][triple_subcycled_warm_cloud_internal][direct]")
{
    for (int mutation = 0; mutation < 43; ++mutation)
    {
        subcycled_warm_cloud_live::initialize();
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        qv.fill(0); qc.fill(0); qr.fill(0);
        const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test()).fill(0);
        const_cast<Field3D&>(scheme.internal_triple_number_for_test(0)).fill(0);
        const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(0)).fill(0);
        const int last = NR - 2;
        if (mutation >= 20 && mutation < 32)
        {
            auto* field = held_fields()[mutation - 20];
            field->resize(NR, NTH, NZ, 0);
            field->operator()(last, 0, 0) = 1;
        }
        else if (mutation >= 32 && mutation < 39)
        {
            *held_radiation_powers()[mutation - 32] = 1;
        }
        else
        {
            switch (mutation)
            {
                case 0: rho(last, 0, 0) = 0; break;
                case 1: p(last, 0, 0) = 0; break;
                case 2: theta(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 3: qv(last, 0, 0) = -1; break;
                case 4: qc(last, 0, 0) = 1e-4f; break;
                case 5: qi(last, 0, 0) = 1e-4f; break;
                case 6: qs(last, 0, 0) = 1e-4f; break;
                case 7: qg(last, 0, 0) = 1e-4f; break;
                case 8: qh(last, 0, 0) = 1e-4f; break;
                case 9: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                case 10: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
                case 11: global_simulation_storage.runtime.compute.fallback_active = true; break;
                case 12: global_validation_policy.mode = tmv::GuardMode::Warn; break;
                case 13: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = -1; break;
                case 14: const_cast<Field3D&>(scheme.internal_triple_number_for_test(0))(last, 0, 0) = 1; break;
                case 15: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(0))(last, 0, 0) = 1e-9f; break;
                case 16: const_cast<Field3D&>(scheme.internal_triple_number_for_test(1))(last, 0, 0) = 1; break;
                case 17: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(2))(last, 0, 0) = 1e-9f; break;
                case 18: dtheta_dt_rad(last, 0, 0) = 1; break;
                case 19: global_simulation_storage.runtime.chaos.noise_elapsed_s = 1; break;
                case 39: w(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 40:
                {
                    theta(last, 0, 0) = static_cast<float>(401 * std::pow(microphysics_constants::p0 / p(last, 0, 0),
                        microphysics_constants::R_d / microphysics_constants::cp));
                    REQUIRE(static_cast<double>(theta(last, 0, 0)) * std::pow(p(last, 0, 0) / microphysics_constants::p0,
                        microphysics_constants::R_d / microphysics_constants::cp) > 400);
                    break;
                }
                case 41: global_simulation_storage.runtime.terrain.earth_fixed_placement.active = true; break;
                case 42: scheme.wrong_wind_declaration = true; break;
            }
        }
        const Snapshot before;
        INFO("subcycled zero-event mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .05));
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
    }
    subcycled_warm_cloud_live::initialize();
    CHECK_THROWS(microphysics_scheme->portable_state());
    CHECK_THROWS(microphysics_scheme->clone_parameters());
    CHECK_THROWS(microphysics_scheme->prognostic_scalars_for_transport());
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    const Snapshot source;
    std::string error;
    CHECK_FALSE(scheme.enable_internal_triple_subcycled_warm_cloud_primary_for_test(global_simulation_state(),
        source.moments[0], source.moments[1], source.moments[6], error));
    CHECK_FALSE(error.empty());
    check_snapshot(source);
    tmv::MilbrandtScheme candidate;
    auto oversized = global_simulation_state();
    const int original_nr = NR;
    oversized.grid.nr = 4097;
    CHECK_FALSE(candidate.enable_internal_triple_subcycled_warm_cloud_primary_for_test(oversized,
        source.moments[0], source.moments[1], source.moments[6], error));
    NR = original_nr;
    CHECK_FALSE(candidate.owns_primary_endpoint_commit());
    CHECK_FALSE(error.empty());
    check_snapshot(source);
}

namespace cloud_adjustment_live
{

void initialize(int scenario = 0)
{
    ice_melting_live::initialize(false, 0, 0, 2, 290);
    Field3D nc(NR, NTH, NZ, scenario == 0 ? 0 : 1e5f);
    for (int i = 0; i < NR; ++i)
    {
        const double temperature = thermodynamics::theta_to_temperature(theta(i, 0, 0), p(i, 0, 0));
        const double qs = thermodynamics::saturation_mixing_ratio_water(temperature, p(i, 0, 0));
        w(i, 0, 0) = i % 3 == 0 ? 0.0f : i % 3 == 1 ? .5f : 5.0f;
        qc(i, 0, 0) = scenario == 0 ? 0 : scenario == 1 ? .02f : 1e-4f;
        qv(i, 0, 0) = scenario == 0 ? static_cast<float>(1.2 * qs)
            : scenario == 1 ? static_cast<float>(.4 * qs) : 0;
    }
    auto scheme = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = true;
    scheme->configure(config);
    std::string error;
    const bool activated = scheme->enable_internal_triple_cloud_adjustment_for_test(global_simulation_state(), nc, error);
    INFO(error);
    REQUIRE(activated);
    REQUIRE(scheme->owns_primary_endpoint_commit());
    microphysics_scheme = std::move(scheme);
    global_microphysics_runtime_state.theta_storage_residual.resize(NR, NTH, NZ, .0125f);
}

} // namespace cloud_adjustment_live

TEST_CASE("live cloud adjustment commits exact saturation activation and population heat ledgers",
          "[physics][milbrandt][triple_cloud_adjustment_internal][direct]")
{
    for (int scenario : {0, 1, 2})
    {
        for (double duration : {.05, .1, .3})
        {
            cloud_adjustment_live::initialize(scenario);
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            warm_primary_live::check_step(before, duration, true);
            const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
            CHECK(scheme.observed_velocity_source() == &w);
            if (scenario == 2)
            {
                for (int i = 1; i < NR - 1; ++i)
                {
                    CHECK(qc(i, 0, 0) == 0);
                    CHECK(scheme.internal_triple_cloud_number_for_test()(i, 0, 0) == 0);
                }
            }
        }
    }
}

TEST_CASE("live cloud adjustment actual workers and empty continuation preserve all owners",
          "[physics][milbrandt][triple_cloud_adjustment_internal][direct]")
{
#ifdef _OPENMP
    const int old_threads = omp_get_max_threads();
    const int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    std::array<std::unique_ptr<Snapshot>, 2> after;
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        omp_set_num_threads(workers);
        int observed = 0;
#pragma omp parallel
        {
#pragma omp single
            observed = omp_get_num_threads();
        }
        REQUIRE(observed == workers);
        cloud_adjustment_live::initialize(2);
        for (double duration : {.05, .1, .3})
        {
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            warm_primary_live::check_step(before, duration, true);
            CHECK(dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme).observed_velocity_source() == &w);
            simulation_time += duration;
        }
        after[static_cast<std::size_t>(run)] = std::make_unique<Snapshot>();
    }
    check_snapshot(*after[0], *after[1]);
    omp_set_num_threads(old_threads);
    omp_set_dynamic(old_dynamic);
#else
    FAIL("cloud adjustment worker evidence requires OpenMP");
#endif
}

TEST_CASE("cloud activation protects its wind source and context through the last private move",
          "[physics][milbrandt][triple_cloud_adjustment_internal][direct]")
{
    for (int mutation : {1, 2, 3, 12, 14, 17, 18, 19, 20, 21, 25, 26, 27, 28})
    {
        INFO("cloud getter mutation " << mutation);
        cloud_adjustment_live::initialize();
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .1));
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
        warm_primary_live::check_step(before, .1, true);
    }
    for (int mutation = 0; mutation < 6; ++mutation)
    {
        INFO("cloud late mutation " << mutation);
        cloud_adjustment_live::initialize();
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .1));
        CHECK(accepted_attempts == 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
        CHECK(dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme).observed_velocity_source() == &w);
        warm_primary_live::check_step(before, .1, true);
    }
}

TEST_CASE("cloud adjustment zero source invalid profiles phase exits and portable selectors remain guarded",
          "[physics][milbrandt][triple_cloud_adjustment_internal][direct]")
{
    for (int mutation = 0; mutation < 9; ++mutation)
    {
        INFO("cloud zero-event mutation " << mutation);
        cloud_adjustment_live::initialize(2);
        qc.fill(0);
        qv.fill(0);
        const int last = NR - 2;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test()).fill(0);
        switch (mutation)
        {
            case 0: w(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
            case 1:
                theta(last, 0, 0) = static_cast<float>(410 / std::pow(
                    static_cast<double>(p(last, 0, 0)) / microphysics_constants::p0,
                    microphysics_constants::R_d / microphysics_constants::cp));
                REQUIRE(thermodynamics::theta_to_temperature(theta(last, 0, 0), p(last, 0, 0)) > 400);
                break;
            case 2: qr(last, 0, 0) = 1e-4f; break;
            case 3: const_cast<Field3D&>(scheme.internal_triple_number_for_test(0))(last, 0, 0) = 1; break;
            case 4: rho(last, 0, 0) = 0; break;
            case 5: qv(last, 0, 0) = -1; break;
            case 6: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
            case 7: dtheta_dt_rad(last, 0, 0) = 1; break;
            case 8: global_validation_policy.mode = tmv::GuardMode::Warn; break;
        }
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .1));
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
    }
    cloud_adjustment_live::initialize(2);
    for (int i = 0; i < NR; ++i)
    {
        theta(i, 0, 0) = static_cast<float>(thermodynamics::temperature_to_theta(273.2, p(i, 0, 0)));
        qc(i, 0, 0) = .1f;
    }
    const Snapshot warm;
    REQUIRE_THROWS(step_microphysics(global_simulation_state(), .1));
    CHECK(getter_attempts == 0);
    check_snapshot(warm);
    cloud_adjustment_live::initialize();
    auto& declared = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    declared.wrong_wind_declaration = true;
    const Snapshot declared_before;
    REQUIRE_THROWS(step_microphysics(global_simulation_state(), .1));
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(declared_before);
    declared.wrong_wind_declaration = false;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
    warm_primary_live::check_step(declared_before, .1, true);
    CHECK_THROWS(microphysics_scheme->portable_state());
    CHECK_THROWS(microphysics_scheme->clone_parameters());
    CHECK_THROWS(microphysics_scheme->prognostic_scalars_for_transport());
}

namespace warm_evaporation_live
{

void initialize()
{
    warm_primary_live::initialize();
    qc.fill(0);
    qv.fill(0);
    for (int i = 0; i < NR; ++i)
    {
        theta(i, 0, 0) = static_cast<float>(thermodynamics::temperature_to_theta(288, p(i, 0, 0)));
    }
    global_microphysics_runtime_state.theta_storage_residual.resize(NR, NTH, NZ, .0125f);
}

void check_extinction(const Snapshot& before)
{
    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    const auto physical = physical_scalar_domain(global_simulation_state().grid);
    for (int i = physical.i_begin; i < physical.i_end; ++i)
    {
        CHECK(qr(i, 0, 0) == 0);
        CHECK(scheme.internal_triple_number_for_test(0)(i, 0, 0) == 0);
        CHECK(scheme.internal_triple_reflectivity_for_test(0)(i, 0, 0) == 0);
        CHECK(qv(i, 0, 0) == before.fields[6](i, 0, 0) + before.fields[8](i, 0, 0));
        CHECK(scheme.internal_triple_cloud_number_for_test()(i, 0, 0) == before.moments[0](i, 0, 0));
    }
}

} // namespace warm_evaporation_live

TEST_CASE("live admitted rain extinction continuation and actual workers preserve all owners",
          "[physics][milbrandt][triple_warm_evaporation_internal][direct]")
{
#ifdef _OPENMP
    const int old_threads = omp_get_max_threads();
    const int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    std::array<std::unique_ptr<Snapshot>, 2> after;
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        omp_set_num_threads(workers);
        int observed = 0;
#pragma omp parallel
        {
#pragma omp single
            observed = omp_get_num_threads();
        }
        REQUIRE(observed == workers);
        warm_evaporation_live::initialize();
        for (double duration : {.05, 1000.0, .3})
        {
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            warm_primary_live::check_step(before, duration);
            if (duration == 1000)
            {
                warm_evaporation_live::check_extinction(before);
            }
            if (duration == .3)
            {
                const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
                const auto physical = physical_scalar_domain(global_simulation_state().grid);
                for (int i = physical.i_begin; i < physical.i_end; ++i)
                {
                    CHECK(scheme.endpoint.theta_increment[i] == 0);
                    CHECK(qv(i, 0, 0) == before.fields[6](i, 0, 0));
                }
            }
            simulation_time += duration;
        }
        after[static_cast<std::size_t>(run)] = std::make_unique<Snapshot>();
    }
    check_snapshot(*after[0], *after[1]);
    omp_set_num_threads(old_threads);
    omp_set_dynamic(old_dynamic);
#else
    FAIL("rain extinction worker evidence requires OpenMP");
#endif
}

TEST_CASE("live whole rain evaporation rejects altered records and late private failures atomically",
          "[physics][milbrandt][triple_warm_evaporation_internal][direct]")
{
    for (int mutation : {1, 2, 3, 12, 14, 17, 18, 19, 20, 21, 25})
    {
        warm_evaporation_live::initialize();
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1000));
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1000));
        warm_primary_live::check_step(before, 1000);
        warm_evaporation_live::check_extinction(before);
    }
    for (int mutation = 0; mutation < 3; ++mutation)
    {
        warm_evaporation_live::initialize();
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        const std::string expected = mutation == 1 ? "primary microphysics accepted attribution changed"
            : mutation == 2 ? "post-primary microphysics private candidate rejected: injected after joint frozen private moment validation"
            : "post-primary microphysics private candidate rejected: injected after joint frozen endpoint accepted-primary validation";
        REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), 1000), expected);
        CHECK(accepted_attempts == 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1000));
        warm_primary_live::check_step(before, 1000);
        warm_evaporation_live::check_extinction(before);
    }
}

namespace snow_melting_live
{

void initialize(bool receiver = true, float snow_mass = 5e-4f, float snow_number = 10.0f,
                long double alpha = 2, double temperature = 283.15, double humidity = 1.1)
{
    ice_melting_live::initialize(receiver, 0, 0, 2, temperature);
    const auto& previous = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    const Field3D nr(previous.internal_triple_number_for_test(0));
    const Field3D zr(previous.internal_triple_reflectivity_for_test(0));
    const auto source = snow_melting_reference::population(alpha, snow_mass, snow_number);
    Field3D ns(NR, NTH, NZ, snow_number);
    Field3D zs(NR, NTH, NZ, static_cast<float>(source.reflectivity_per_kg));
    for (int i = 0; i < NR; ++i)
    {
        qs(i, 0, 0) = snow_mass;
        qv(i, 0, 0) = static_cast<float>(humidity * thermodynamics::saturation_mixing_ratio_water(
            microphysics_constants::T0, p(i, 0, 0)));
    }
    auto scheme = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = true;
    scheme->configure(config);
    std::string error;
    const bool activated = scheme->enable_internal_triple_snow_melting_for_test(
        global_simulation_state(), nr, zr, ns, zs, error);
    INFO(error);
    REQUIRE(activated);
    REQUIRE(scheme->owns_primary_endpoint_commit());
    microphysics_scheme = std::move(scheme);
}

} // namespace snow_melting_live

TEST_CASE("live noncubic snow melting commits independent mass squared and latent heat references",
          "[physics][milbrandt][triple_snow_melting_internal][direct]")
{
    for (bool receiver : {false, true})
    {
        for (long double alpha : {2.0L, 20.0L, 200.0L})
        {
            for (double duration : {.1, 1e6})
            {
                snow_melting_live::initialize(receiver, 5e-4f, 10.0f, alpha);
                global_microphysics_runtime_state.theta_storage_residual.resize(NR, NTH, NZ, .0125f);
                const Snapshot before;
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                melting_live::check_step(before, duration, false, true);
                if (duration == 1e6)
                {
                    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
                    for (int i = 1; i < NR - 1; ++i)
                    {
                        CHECK(qs(i, 0, 0) == 0);
                        CHECK(scheme.internal_triple_number_for_test(2)(i, 0, 0) == 0);
                        CHECK(scheme.internal_triple_reflectivity_for_test(2)(i, 0, 0) == 0);
                    }
                }
            }
        }
    }
}

TEST_CASE("live snow continuation extinction empty interval and actual workers preserve science bytes",
          "[physics][milbrandt][triple_snow_melting_internal][direct]")
{
#ifdef _OPENMP
    const int old_threads = omp_get_max_threads();
    const int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    std::array<std::unique_ptr<Snapshot>, 2> after;
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        omp_set_num_threads(workers);
        int observed = 0;
#pragma omp parallel
        {
#pragma omp single
            observed = omp_get_num_threads();
        }
        REQUIRE(observed == workers);
        snow_melting_live::initialize();
        for (double duration : {.05, .1, .3, 1e6, .1})
        {
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            melting_live::check_step(before, duration, false, true);
            simulation_time += duration;
        }
        after[static_cast<std::size_t>(run)] = std::make_unique<Snapshot>();
        CHECK(qs(1, 0, 0) == 0);
    }
    check_snapshot(*after[0], *after[1]);
    omp_set_num_threads(old_threads);
    omp_set_dynamic(old_dynamic);
#else
    FAIL("live snow melting worker evidence requires OpenMP");
#endif
}

TEST_CASE("live snow altered records protected source and late private ledger failures recover atomically",
          "[physics][milbrandt][triple_snow_melting_internal][direct]")
{
    for (int mutation = 1; mutation <= 24; ++mutation)
    {
        snow_melting_live::initialize();
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .05));
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
        melting_live::check_step(before, .05, false, true);
    }
    for (int mutation = 0; mutation < 3; ++mutation)
    {
        snow_melting_live::initialize();
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
        simulation_time += .05;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        const int calls_before = accepted_attempts;
        const std::string expected = mutation == 1
            ? "primary microphysics accepted attribution changed"
            : mutation == 2
            ? "post-primary microphysics private candidate rejected: injected after joint frozen private moment validation"
            : "post-primary microphysics private candidate rejected: injected after joint frozen endpoint accepted-primary validation";
        REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), .1), expected);
        CHECK(accepted_attempts == calls_before + 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .1));
        melting_live::check_step(before, .1, false, true);
    }
}

TEST_CASE("live snow zero events cannot bypass public private backend and held source checks",
          "[physics][milbrandt][triple_snow_melting_internal][direct]")
{
    for (int mutation = 0; mutation < 41; ++mutation)
    {
        snow_melting_live::initialize(false, 0, 0);
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        if (mutation >= 22 && mutation < 34)
        {
            auto* field = held_fields()[mutation - 22];
            field->resize(NR, NTH, NZ, 0);
            field->operator()(last, 0, 0) = 1;
        }
        else if (mutation >= 34)
        {
            *held_radiation_powers()[mutation - 34] = 1;
        }
        else
        {
            switch (mutation)
            {
                case 0: rho(last, 0, 0) = 0; break;
                case 1: p(last, 0, 0) = 0; break;
                case 2: theta(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 3: qv(last, 0, 0) = -1; break;
                case 4: qc(last, 0, 0) = 1e-4f; break;
                case 5: qi(last, 0, 0) = 1e-4f; break;
                case 6: qg(last, 0, 0) = 1e-4f; break;
                case 7: qh(last, 0, 0) = 1e-4f; break;
                case 8: qs(last, 0, 0) = 1e-4f; break;
                case 9: qv.resize(1, 1, 1, 0); break;
                case 10: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                case 11: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
                case 12: global_simulation_storage.runtime.compute.fallback_active = true; break;
                case 13: global_simulation_storage.configuration.microphysics_config.milbrandt.triple_moment = true; break;
                case 14: global_validation_policy.mode = tmv::GuardMode::Warn; break;
                case 15: const_cast<Field3D&>(scheme.internal_triple_number_for_test(2))(last, 0, 0) = 1; break;
                case 16: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(2))(last, 0, 0) = 1e-8f; break;
                case 17: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = 1; break;
                case 18: const_cast<Field3D&>(scheme.internal_triple_number_for_test(1))(last, 0, 0) = 1; break;
                case 19: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(4))(last, 0, 0) = 1e-8f; break;
                case 20: dtheta_dt_rad(last, 0, 0) = 1; break;
                case 21: global_simulation_storage.runtime.chaos.noise_elapsed_s = 1; break;
            }
        }
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), .05));
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
    }
}

TEST_CASE("live snow candidates detached fields stale commits phase exits and activation laws remain guarded",
          "[physics][milbrandt][triple_snow_melting_internal][direct]")
{
    snow_melting_live::initialize();
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    std::string error;
    tmv::MicrophysicsPrimaryEndpointCandidate absent;
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, absent, error));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    auto compute = [&](std::array<Field3D, 8>& out, const Field3D& snow)
    {
        scheme.compute_tendencies(p, theta, rho, qv, qc, qr, qi, snow, qg, qh, .05,
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    };
    std::array<Field3D, 8> out;
    REQUIRE_NOTHROW(compute(out, qs));
    tmv::MicrophysicsPrimaryEndpointCandidate first;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    REQUIRE_NOTHROW(compute(out, qs));
    tmv::MicrophysicsPrimaryEndpointCandidate second;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, second, error));
    CHECK(second.generation > first.generation);
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), .05, first, error));
    const Field3D detached(qs);
    for (auto& field : out)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    CHECK_THROWS(compute(out, detached));
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), .05, absent, error));
    for (const auto& field : out)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    CHECK_THROWS(scheme.portable_state());
    CHECK_THROWS(scheme.clone_parameters());
    CHECK_THROWS(scheme.prognostic_scalars_for_transport());
    const Snapshot before;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .05));
    melting_live::check_step(before, .05, false, true);
    CHECK_FALSE(dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme)
        .validate_primary_endpoint_commit(global_simulation_state(), .05, second, error));
    snow_melting_live::initialize(true, .02f, 1e5f, 2, 273.20);
    const Snapshot warm;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), 1e7), Catch::Matchers::Contains("final heat"));
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(warm);
    snow_melting_live::initialize();
    const Snapshot source;
    tmv::MilbrandtScheme changed(0, 0, 0, 0, 0, 523.598775598, 3, 440, 3, .1597,
        std::nextafter(2.078, 3.0));
    CHECK_FALSE(changed.enable_internal_triple_snow_melting_for_test(global_simulation_state(),
        source.moments[1], source.moments[6], source.moments[3], source.moments[8], error));
    CHECK_FALSE(changed.owns_primary_endpoint_commit());
    check_snapshot(source);
}

namespace frozen_collection_live
{

constexpr std::array<long double, 4> mass_coefficients{440, .1597L, 209.439510239L, 471.238898038L};
constexpr std::array<long double, 4> mass_exponents{3, 2.078L, 3, 3};

tmv::milbrandt_detail::ColdParameters parameters(bool hail)
{
    const auto& values = tmv::milbrandt_detail::default_cold_parameters;
    return {values[0], values[1], values[2], values[3], values[4],
        values[5], values[6], values[7], values[8], values[9],
        values[10], values[11], values[12], values[13], values[14],
        values[15], values[16], values[17], values[18], values[19],
        values[20], values[21], values[22], values[23], values[24], hail};
}

float reflectivity(std::size_t species, float mass, float number, long double alpha)
{
    if (mass == 0)
    {
        return 0;
    }
    const long double log_lambda = (std::log(mass_coefficients[species] * number / mass)
        + std::lgamma(alpha + mass_exponents[species] + 1) - std::lgamma(alpha + 1))
        / mass_exponents[species];
    const float result = static_cast<float>(std::exp(std::log(static_cast<long double>(number))
        + std::lgamma(alpha + 7) - std::lgamma(alpha + 1) - 6 * log_lambda));
    return alpha == 200 ? std::nextafter(result, std::numeric_limits<float>::infinity()) : result;
}

void initialize(int scenario = 0, long double alpha = 2, bool hail = true)
{
    initialize_state();
    for (auto* field : public_fields())
    {
        *field = subcycled_warm_cloud_live::bounded_field(*field);
    }
    for (auto* field : source_fields())
    {
        *field = subcycled_warm_cloud_live::bounded_field(*field);
    }
    for (auto* field : held_fields())
    {
        *field = subcycled_warm_cloud_live::bounded_field(*field);
    }
    auto& tendencies = global_microphysics_runtime_state.tendencies;
    for (Field3D* field : {&tendencies.dtheta_dt, &tendencies.dqv_dt, &tendencies.dqc_dt,
                         &tendencies.dqr_dt, &tendencies.dqi_dt, &tendencies.dqs_dt,
                         &tendencies.dqg_dt, &tendencies.dqh_dt,
                         &global_microphysics_runtime_state.theta_storage_residual})
    {
        *field = subcycled_warm_cloud_live::bounded_field(*field);
    }
    NR = 10;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    global_simulation_storage.runtime.advection.nr = NR;
    global_simulation_storage.surface.precipitation_rate_mm_h.resize(NR);
    global_simulation_storage.surface.accumulated_rainfall_mm.resize(NR);
    global_simulation_storage.surface.accumulated_rainfall_residual_mm.resize(NR);
    global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes = hail;
    const std::array<float, 4> initial_mass{4e-5f, 8e-4f, 5e-4f, 4e-4f};
    const std::array<float, 4> initial_number{1e5f, 1e4f, 1e3f, 1e2f};
    const std::array<bool, 4> active{scenario == 0 || scenario <= 3,
        scenario == 0 || scenario == 1 || scenario == 4 || scenario == 5,
        scenario == 0 || scenario == 2, scenario == 0 || scenario == 3 || scenario == 4};
    std::array<Field3D, 4> number;
    std::array<Field3D, 4> z;
    const std::array<Field3D*, 4> mass{&qi, &qs, &qg, &qh};
    for (std::size_t species = 0; species < mass.size(); ++species)
    {
        const float q = active[species] && scenario != 6 ? initial_mass[species] : 0;
        const float n = q > 0 ? initial_number[species] : 0;
        mass[species]->resize(NR, NTH, NZ, q);
        number[species].resize(NR, NTH, NZ, n);
        z[species].resize(NR, NTH, NZ, reflectivity(species, q, n, alpha));
    }
    qc.resize(NR, NTH, NZ, 0);
    qr.resize(NR, NTH, NZ, 0);
    auto scheme = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = hail;
    scheme->configure(config);
    std::string error;
    const bool activated = scheme->enable_internal_triple_frozen_collection_for_test(
        global_simulation_state(), number[0], z[0], number[1], z[1], number[2], z[2], number[3], z[3], error);
    INFO(error);
    REQUIRE(activated);
    REQUIRE(scheme->owns_primary_endpoint_commit());
    REQUIRE(scheme->protected_primary_endpoint_wind_source() == nullptr);
    microphysics_scheme = std::move(scheme);
}

FrozenCollectionResult capture(const Snapshot& before, int i, double duration)
{
    FrozenCollectionCell source;
    source.temperature_k = thermodynamics::theta_to_temperature(before.fields[5](i, 0, 0), before.fields[1](i, 0, 0));
    source.pressure_pa = before.fields[1](i, 0, 0);
    source.dry_density_kg_m3 = before.fields[0](i, 0, 0);
    for (std::size_t species = 0; species < 4; ++species)
    {
        source.initial.frozen[species] = {before.fields[species + 9](i, 0, 0),
            before.moments[species + 2](i, 0, 0), before.moments[species + 7](i, 0, 0)};
    }
    FrozenCollectionResult result;
    std::string error;
    const bool ok = evaluate_exact_triple_frozen_collection_cell(parameters(
        global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes), source, duration, result, error);
    INFO(error);
    REQUIRE(ok);
    return result;
}

void check_step(const Snapshot& before, double duration)
{
    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    REQUIRE(scheme.getter_called);
    REQUIRE(scheme.accepted_called);
    const auto& candidate = scheme.endpoint;
    REQUIRE(candidate.owner == &scheme);
    REQUIRE(candidate.duration_s == duration);
    REQUIRE(candidate.generation != 0);
    REQUIRE(candidate.nr == NR);
    REQUIRE(candidate.nth == NTH);
    REQUIRE(candidate.nz == NZ);
    const std::array<const Field3D*, 10> bindings{&rho, &p, &theta, &qv, &qc, &qr, &qi, &qs, &qg, &qh};
    CHECK(candidate.source_fields == bindings);
    const auto physical = physical_scalar_domain(global_simulation_state().grid);
    const std::array<const Field3D*, 7> water{&qv, &qc, &qr, &qi, &qs, &qg, &qh};
    const auto& tendencies = global_microphysics_runtime_state.tendencies;
    std::array<long double, 7> rounding{};
    std::array<long double, 7> requested{};
    std::array<long double, 6> moist{};
    for (int i = physical.i_begin; i < physical.i_end; ++i)
    {
        const auto result = capture(before, i, duration);
        const long double dry_mass = static_cast<long double>(before.fields[0](i, 0, 0)) * dr * dtheta * dz;
        long double total_before = 0;
        long double total_unrounded = 0;
        long double total_after = 0;
        CHECK(candidate.theta_increment[i] == 0);
        CHECK(result.temperature_increment_k == 0);
        CHECK(tendencies.dtheta_dt(i, 0, 0) == 0);
        CHECK(tendencies.dqv_dt(i, 0, 0) == 0);
        CHECK(tendencies.dqc_dt(i, 0, 0) == 0);
        CHECK(tendencies.dqr_dt(i, 0, 0) == 0);
        for (std::size_t species = 0; species < water.size(); ++species)
        {
            const double old = before.fields[species + 6](i, 0, 0);
            const double endpoint = candidate.water_after[species][i];
            const float stored = (*water[species])(i, 0, 0);
            CHECK(stored == static_cast<float>(endpoint));
            if (species < 3)
            {
                CHECK(endpoint == old);
            }
            else
            {
                CHECK(endpoint == result.unrounded.frozen[species - 3].mass_per_kg);
                CHECK(stored == result.rounded[species - 3].mass_per_kg);
                CHECK(scheme.internal_triple_number_for_test(species - 2)(i, 0, 0)
                    == result.rounded[species - 3].number_per_kg);
                CHECK(scheme.internal_triple_reflectivity_for_test(species - 2)(i, 0, 0)
                    == result.rounded[species - 3].reflectivity_per_kg);
            }
            rounding[species] += (static_cast<long double>(stored) - endpoint) * dry_mass;
            requested[species] += (static_cast<long double>(endpoint) - old) * dry_mass;
            total_before += old;
            total_unrounded += endpoint;
            total_after += stored;
        }
        CHECK(static_cast<double>(total_unrounded) == Approx(static_cast<double>(total_before)).epsilon(8e-15));
        CHECK(static_cast<double>(dry_mass * total_unrounded)
            == Approx(static_cast<double>(dry_mass * total_before)).epsilon(8e-15));
        const long double frozen_before = total_before - before.fields[6](i, 0, 0);
        const long double frozen_after = total_after - qv(i, 0, 0);
        const long double t = thermodynamics::theta_to_temperature(theta(i, 0, 0), p(i, 0, 0));
        moist[1] += dry_mass * (t - microphysics_constants::T0) * (frozen_after - frozen_before) * 2106;
        moist[2] -= dry_mass * (frozen_after - frozen_before) * microphysics_constants::L_f;
        const long double kinetic = .5L * (std::pow(static_cast<long double>(u(i, 0, 0)), 2)
            + std::pow(static_cast<long double>(v(i, 0, 0)), 2) + std::pow(static_cast<long double>(w(i, 0, 0)), 2));
        moist[3] += dry_mass * (total_after - total_before) * kinetic;
        const long double height = global_grid_geometry.z.empty() ? .5L * dz : global_grid_geometry.z[0];
        moist[4] += dry_mass * (total_after - total_before) * physical_constants::gravity_ms2 * height;
    }
    const Snapshot after;
    for (const std::size_t field : {0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14})
    {
        CHECK(same_bytes(after.fields[field], before.fields[field]));
    }
    for (const std::size_t moment : {0, 1, 6})
    {
        CHECK(same_bytes(after.moments[moment], before.moments[moment]));
    }
    for (const int i : {0, NR - 1})
    {
        for (std::size_t field = 0; field < before.fields.size(); ++field)
        {
            CHECK(std::memcmp(&after.fields[field](i, 0, 0), &before.fields[field](i, 0, 0), sizeof(float)) == 0);
        }
        for (std::size_t moment = 0; moment < before.moments.size(); ++moment)
        {
            CHECK(std::memcmp(&after.moments[moment](i, 0, 0), &before.moments[moment](i, 0, 0), sizeof(float)) == 0);
        }
    }
    CHECK(same_bytes(after.theta_residual, before.theta_residual));
    for (std::size_t species = 0; species < water.size(); ++species)
    {
        CHECK(after.water.species[species].storage_rounding_kg
            == Approx(static_cast<double>(rounding[species])).epsilon(3e-6).margin(1e-15));
        CHECK(after.heat.microphysics_water_tendency_kg_by_species[species]
            == Approx(static_cast<double>(requested[species])).epsilon(3e-6).margin(1e-12));
    }
    CHECK(after.heat.microphysics_heat_j == 0);
    CHECK(after.heat.theta_storage_rounding_j == 0);
    CHECK(after.heat.theta_storage_residual_release_j == 0);
    CHECK(after.heat.theta_storage_residual_j == 0);
    const std::array<double, 6> measured{after.heat.moist_dry_internal_change_j,
        after.heat.moist_water_sensible_change_j, after.heat.moist_phase_reference_change_j,
        after.heat.moist_carrier_kinetic_change_j, after.heat.moist_gravitational_change_j,
        after.heat.moist_subgrid_tke_change_j};
    for (std::size_t reservoir = 0; reservoir < moist.size(); ++reservoir)
    {
        CHECK(measured[reservoir] == Approx(static_cast<double>(moist[reservoir])).epsilon(3e-6).margin(1e-10));
    }
    CHECK(after.heat.moist_energy_complete);
    CHECK(after.heat.moist_energy_closure_residual_j == 0);
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

} // namespace frozen_collection_live

TEST_CASE("live frozen collection owns all five channels with independent water volume and no heat closure",
          "[physics][milbrandt][triple_frozen_collection_internal][direct]")
{
    for (const long double alpha : {0.0L, 2.0L, 20.0L, 200.0L})
    {
        for (int scenario = 0; scenario <= 5; ++scenario)
        {
            for (const double factor : {1.0, 2.0, 5.0})
            {
                const double duration = (alpha == 200 ? 1e-8 : 1e-4) * factor;
                INFO("shape " << alpha << ", scenario " << scenario << ", duration " << duration);
                frozen_collection_live::initialize(scenario, alpha);
                const Snapshot before;
                const auto producer = frozen_collection_live::capture(before, 1, duration).producer;
                REQUIRE(producer.event_count() == 5);
                if (scenario == 0)
                {
                    for (const auto& channel : producer.channels())
                    {
                        CHECK(channel.admitted_number_increment_per_kg > 0);
                    }
                }
                if (scenario > 0)
                {
                    CHECK(producer.channels()[scenario - 1].admitted_number_increment_per_kg > 0);
                }
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                frozen_collection_live::check_step(before, duration);
            }
        }
    }
    frozen_collection_live::initialize(0, 2, false);
    const Snapshot before;
    const auto producer = frozen_collection_live::capture(before, 1, 1e-4).producer;
    CHECK_FALSE(producer.hail_processes());
    CHECK(producer.channels()[2].admitted_mass_increment_per_kg > 0);
    CHECK(producer.channels()[3].admitted_mass_increment_per_kg > 0);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    frozen_collection_live::check_step(before, 1e-4);
}

TEST_CASE("live frozen collection continuation matches actual one and four workers exactly",
          "[physics][milbrandt][triple_frozen_collection_internal][direct]")
{
#ifndef _OPENMP
    FAIL("frozen collection worker evidence requires OpenMP");
#endif
#ifdef _OPENMP
    const int original_threads = omp_get_max_threads();
    const int original_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
#endif
    std::unique_ptr<Snapshot> serial;
    for (const int threads : {1, 4})
    {
#ifdef _OPENMP
        omp_set_num_threads(threads);
        int actual = 0;
#pragma omp parallel
        {
#pragma omp single
            actual = omp_get_num_threads();
        }
        CHECK(actual == threads);
#endif
        frozen_collection_live::initialize();
        for (const double duration : {1e-4, 2e-4, 5e-4})
        {
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            frozen_collection_live::check_step(before, duration);
            simulation_time += duration;
        }
        if (threads == 1)
        {
            serial = std::make_unique<Snapshot>();
        }
        else
        {
            check_snapshot(*serial);
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(original_threads);
    omp_set_dynamic(original_dynamic);
#endif
}

TEST_CASE("live frozen collection altered records and late ledger private failures roll back and recover",
          "[physics][milbrandt][triple_frozen_collection_internal][direct]")
{
    for (int mutation = 1; mutation <= 25; ++mutation)
    {
        frozen_collection_live::initialize();
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        INFO("frozen collection getter mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        frozen_collection_live::check_step(before, 1e-4);
    }
    for (const int mutation : {0, 1, 2, 6, 7, 8})
    {
        frozen_collection_live::initialize();
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        INFO("frozen collection late mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(accepted_attempts == 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        frozen_collection_live::check_step(before, 1e-4);
    }
}

TEST_CASE("live frozen collection zero events reject all malformed source backend and held owners",
          "[physics][milbrandt][triple_frozen_collection_internal][direct]")
{
    for (int mutation = 0; mutation < 44; ++mutation)
    {
        frozen_collection_live::initialize(6);
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        if (mutation >= 25 && mutation < 37)
        {
            auto* held = held_fields()[mutation - 25];
            held->resize(NR, NTH, NZ, 0);
            (*held)(last, 0, 0) = 1;
        }
        else if (mutation >= 37)
        {
            *held_radiation_powers()[mutation - 37] = 1;
        }
        else
        {
            switch (mutation)
            {
                case 0: rho(last, 0, 0) = 0; break;
                case 1: p(last, 0, 0) = 0; break;
                case 2: theta(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 3: qv(last, 0, 0) = -1; break;
                case 4: qc(last, 0, 0) = 1e-4f; break;
                case 5: qr(last, 0, 0) = 1e-4f; break;
                case 6: qi(last, 0, 0) = 1e-4f; break;
                case 7: qs(last, 0, 0) = 1e-4f; break;
                case 8: qv.resize(1, 1, 1, 0); break;
                case 9: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                case 10: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
                case 11: global_simulation_storage.runtime.compute.fallback_active = true; break;
                case 12: global_simulation_storage.configuration.microphysics_config.milbrandt.triple_moment = true; break;
                case 13: global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes = false; break;
                case 14: global_validation_policy.mode = tmv::GuardMode::Warn; break;
                case 15: const_cast<Field3D&>(scheme.internal_triple_number_for_test(1))(last, 0, 0) = 1; break;
                case 16: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(2))(last, 0, 0) = 1e-8f; break;
                case 17: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = 1; break;
                case 18: dtheta_dt_rad(last, 0, 0) = 1; break;
                case 19: global_simulation_storage.runtime.chaos.noise_elapsed_s = 1; break;
                case 20: qg(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 21: qh(last, 0, 0) = -1; break;
                case 22: const_cast<Field3D&>(scheme.internal_triple_number_for_test(0))(last, 0, 0) = 1; break;
                case 23: theta(last, 0, 0) = static_cast<float>(thermodynamics::temperature_to_theta(280, p(last, 0, 0))); break;
                case 24: global_validation_policy.fail_on = tmv::GuardFailOn::NonFinite; break;
            }
        }
        const Snapshot before;
        INFO("frozen collection zero mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
    }
}

TEST_CASE("live frozen collection stale detached portable transport and activation owners remain guarded",
          "[physics][milbrandt][triple_frozen_collection_internal][direct]")
{
    frozen_collection_live::initialize();
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    std::string error;
    tmv::MicrophysicsPrimaryEndpointCandidate absent;
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, absent, error));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    auto compute = [&](std::array<Field3D, 8>& out, const Field3D& ice)
    {
        scheme.compute_tendencies(p, theta, rho, qv, qc, qr, ice, qs, qg, qh, 1e-4,
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    };
    std::array<Field3D, 8> out;
    REQUIRE_NOTHROW(compute(out, qi));
    tmv::MicrophysicsPrimaryEndpointCandidate first;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    REQUIRE_NOTHROW(compute(out, qi));
    tmv::MicrophysicsPrimaryEndpointCandidate second;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, second, error));
    CHECK(second.generation > first.generation);
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    const Field3D detached(qi);
    for (auto& field : out)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    CHECK_THROWS(compute(out, detached));
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, absent, error));
    for (const auto& field : out)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    CHECK_THROWS(scheme.portable_state());
    CHECK_THROWS(scheme.clone_parameters());
    CHECK_THROWS(scheme.prognostic_scalars_for_transport());
    const Snapshot before;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    frozen_collection_live::check_step(before, 1e-4);
    CHECK_FALSE(dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme)
        .validate_primary_endpoint_commit(global_simulation_state(), 1e-4, second, error));
    CHECK_FALSE(dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).commit_prepared_private_candidate());
    frozen_collection_live::initialize();
    const Snapshot source;
    tmv::MilbrandtScheme changed(0, 0, 0, 0, 0, 523.598775598, 3, 440, 3, .1597,
        std::nextafter(2.078, 3.0));
    CHECK_FALSE(changed.enable_internal_triple_frozen_collection_for_test(global_simulation_state(),
        source.moments[2], source.moments[7], source.moments[3], source.moments[8],
        source.moments[4], source.moments[9], source.moments[5], source.moments[10], error));
    CHECK_FALSE(changed.owns_primary_endpoint_commit());
    check_snapshot(source);
}

namespace cold_cloud_collection_live
{

void initialize(int scenario = 0, long double alpha = 2, bool hail = true,
                float cloud_mass = 1e-3f, float cloud_number = 1e8f)
{
    frozen_collection_live::initialize(0, alpha, hail);
    const Snapshot source;
    qi.resize(NR, NTH, NZ, 0);
    qc.resize(NR, NTH, NZ, cloud_mass);
    Field3D nc(NR, NTH, NZ, cloud_number);
    std::array<Field3D, 3> number;
    std::array<Field3D, 3> z;
    const std::array<Field3D*, 3> mass{&qs, &qg, &qh};
    for (std::size_t species = 0; species < 3; ++species)
    {
        number[species] = source.moments[species + 3];
        z[species] = source.moments[species + 8];
        if (scenario != 0 && scenario != static_cast<int>(species + 1))
        {
            mass[species]->resize(NR, NTH, NZ, 0);
            number[species].resize(NR, NTH, NZ, 0);
            z[species].resize(NR, NTH, NZ, 0);
        }
    }
    auto scheme = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = hail;
    scheme->configure(config);
    std::string error;
    const bool activated = scheme->enable_internal_triple_cold_cloud_collection_for_test(
        global_simulation_state(), nc, number[0], z[0], number[1], z[1], number[2], z[2], error);
    INFO(error);
    REQUIRE(activated);
    REQUIRE(scheme->owns_primary_endpoint_commit());
    REQUIRE(scheme->protected_primary_endpoint_wind_source() == nullptr);
    microphysics_scheme = std::move(scheme);
}

ColdCloudCollectionResult capture(const Snapshot& before, int i, double duration)
{
    ColdCloudCollectionCell source;
    source.pressure_pa = before.fields[1](i, 0, 0);
    source.temperature_k = static_cast<double>(before.fields[5](i, 0, 0))
        * std::pow(source.pressure_pa / microphysics_constants::p0,
                   microphysics_constants::R_d / microphysics_constants::cp);
    source.dry_density_kg_m3 = before.fields[0](i, 0, 0);
    source.initial.cloud_mass_per_kg = before.fields[7](i, 0, 0);
    source.initial.cloud_number_per_kg = before.moments[0](i, 0, 0);
    for (std::size_t species = 0; species < 3; ++species)
    {
        source.initial.frozen[species] = {before.fields[species + 10](i, 0, 0),
            before.moments[species + 3](i, 0, 0), before.moments[species + 8](i, 0, 0)};
    }
    ColdCloudCollectionResult result;
    std::string error;
    const bool ok = evaluate_exact_triple_cold_cloud_collection_cell(frozen_collection_live::parameters(
        global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes), source, duration, result, error);
    INFO(error);
    REQUIRE(ok);
    return result;
}

ColdCloudDiffusionResult capture_coupled(const Snapshot& before, int i, double duration)
{
    ColdCloudDiffusionCell source;
    source.pressure_pa = before.fields[1](i, 0, 0);
    source.temperature_k = static_cast<double>(before.fields[5](i, 0, 0))
        * std::pow(source.pressure_pa / microphysics_constants::p0,
                   microphysics_constants::R_d / microphysics_constants::cp);
    source.dry_density_kg_m3 = before.fields[0](i, 0, 0);
    source.vapor_mass_per_kg = before.fields[6](i, 0, 0);
    source.initial.cloud_mass_per_kg = before.fields[7](i, 0, 0);
    source.initial.cloud_number_per_kg = before.moments[0](i, 0, 0);
    for (std::size_t species = 0; species < 3; ++species)
    {
        source.initial.frozen[species] = {before.fields[species + 10](i, 0, 0),
            before.moments[species + 3](i, 0, 0), before.moments[species + 8](i, 0, 0)};
    }
    ColdCloudDiffusionResult result;
    std::string error;
    const bool ok = evaluate_exact_triple_cold_cloud_diffusion_cell(frozen_collection_live::parameters(
        global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes), source, duration, result, error);
    INFO(error);
    REQUIRE(ok);
    return result;
}

void check_step(const Snapshot& before, double duration, bool coupled = false)
{
    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    REQUIRE(scheme.getter_called);
    REQUIRE(scheme.accepted_called);
    const auto& candidate = scheme.endpoint;
    REQUIRE(candidate.owner == &scheme);
    REQUIRE(candidate.duration_s == duration);
    REQUIRE(candidate.generation != 0);
    const std::array<const Field3D*, 10> bindings{&rho, &p, &theta, &qv, &qc, &qr, &qi, &qs, &qg, &qh};
    CHECK(candidate.source_fields == bindings);
    const auto physical = physical_scalar_domain(global_simulation_state().grid);
    const std::array<const Field3D*, 7> water{&qv, &qc, &qr, &qi, &qs, &qg, &qh};
    std::array<long double, 7> rounding{};
    std::array<long double, 7> requested{};
    std::array<long double, 6> moist{};
    long double latent_heat = 0;
    long double theta_rounding = 0;
    for (int i = physical.i_begin; i < physical.i_end; ++i)
    {
        ColdCloudCollectionResult result;
        double vapor_endpoint = before.fields[6](i, 0, 0);
        if (coupled)
        {
            const auto combined = capture_coupled(before, i, duration);
            result.unrounded = combined.unrounded;
            result.rounded = combined.rounded;
            result.rounded_cloud_mass_per_kg = combined.rounded_cloud_mass_per_kg;
            result.rounded_cloud_number_per_kg = combined.rounded_cloud_number_per_kg;
            vapor_endpoint = combined.vapor_mass_per_kg;
        }
        else
        {
            result = capture(before, i, duration);
        }
        const long double dry_mass = static_cast<long double>(before.fields[0](i, 0, 0)) * dr * dtheta * dz;
        const double exner = std::pow(static_cast<double>(p(i, 0, 0)) / microphysics_constants::p0,
            microphysics_constants::R_d / microphysics_constants::cp);
        const long double heat_factor = dry_mass * microphysics_constants::cp * exner;
        const double cloud_loss = static_cast<double>(before.fields[7](i, 0, 0)) - candidate.water_after[1][i];
        const double deposition = static_cast<double>(before.fields[6](i, 0, 0)) - candidate.water_after[0][i];
        CHECK(cloud_loss >= 0);
        CHECK(cloud_loss <= before.fields[7](i, 0, 0));
        CHECK(candidate.theta_increment[i] * exner
            == Approx((microphysics_constants::L_f * cloud_loss + microphysics_constants::L_s * deposition)
                / microphysics_constants::cp).epsilon(8e-12).margin(1e-14));
        CHECK(candidate.water_after[0][i] == vapor_endpoint);
        CHECK(candidate.water_after[1][i] == result.unrounded.cloud_mass_per_kg);
        CHECK(qc(i, 0, 0) == result.rounded_cloud_mass_per_kg);
        CHECK(scheme.internal_triple_cloud_number_for_test()(i, 0, 0) == result.rounded_cloud_number_per_kg);
        CHECK(scheme.internal_triple_cloud_number_for_test()(i, 0, 0) <= before.moments[0](i, 0, 0));
        const double theta_exact = static_cast<double>(before.fields[5](i, 0, 0)) + candidate.theta_increment[i];
        CHECK(theta(i, 0, 0) == static_cast<float>(theta_exact));
        CHECK(theta_exact * exner >= 150);
        CHECK(theta_exact * exner < microphysics_constants::T0);
        CHECK(static_cast<double>(theta(i, 0, 0)) * exner < microphysics_constants::T0);
        latent_heat += dry_mass * microphysics_constants::L_f * cloud_loss;
        if (coupled)
        {
            latent_heat += dry_mass * microphysics_constants::L_s * deposition;
        }
        theta_rounding += (static_cast<long double>(theta(i, 0, 0)) - theta_exact) * heat_factor;
        long double total_before = 0;
        long double total_unrounded = 0;
        long double total_after = 0;
        for (std::size_t species = 0; species < water.size(); ++species)
        {
            const double old = before.fields[species + 6](i, 0, 0);
            const double endpoint = candidate.water_after[species][i];
            const float stored = (*water[species])(i, 0, 0);
            CHECK(stored == static_cast<float>(endpoint));
            if ((!coupled && species == 0) || species == 2 || species == 3)
            {
                CHECK(endpoint == old);
            }
            else if (species >= 4)
            {
                CHECK(endpoint == result.unrounded.frozen[species - 4].mass_per_kg);
                CHECK(scheme.internal_triple_number_for_test(species - 2)(i, 0, 0)
                    == (coupled ? result.rounded[species - 4].number_per_kg : before.moments[species - 1](i, 0, 0)));
                CHECK(scheme.internal_triple_reflectivity_for_test(species - 2)(i, 0, 0)
                    == result.rounded[species - 4].reflectivity_per_kg);
            }
            rounding[species] += (static_cast<long double>(stored) - endpoint) * dry_mass;
            requested[species] += (static_cast<long double>(endpoint) - old) * dry_mass;
            total_before += old;
            total_unrounded += endpoint;
            total_after += stored;
        }
        CHECK(static_cast<double>(total_unrounded) == Approx(static_cast<double>(total_before)).epsilon(8e-15));
        CHECK(static_cast<double>(dry_mass * total_unrounded)
            == Approx(static_cast<double>(dry_mass * total_before)).epsilon(8e-15));
        const long double old_t = static_cast<long double>(before.fields[5](i, 0, 0)) * exner;
        const long double new_t = static_cast<long double>(theta(i, 0, 0)) * exner;
        const long double old_liquid = before.fields[7](i, 0, 0);
        const long double new_liquid = qc(i, 0, 0);
        const long double old_ice = static_cast<long double>(before.fields[10](i, 0, 0))
            + before.fields[11](i, 0, 0) + before.fields[12](i, 0, 0);
        const long double new_ice = static_cast<long double>(qs(i, 0, 0)) + qg(i, 0, 0) + qh(i, 0, 0);
        const long double cv_v = physical_constants::specific_heat_water_vapor_cp_jkgk - microphysics_constants::R_v;
        moist[0] += dry_mass * (microphysics_constants::cp - microphysics_constants::R_d) * (new_t - old_t);
        moist[1] += dry_mass * ((new_t - microphysics_constants::T0)
            * (qv(i, 0, 0) * cv_v + new_liquid * 4186 + new_ice * 2106)
            - (old_t - microphysics_constants::T0)
            * (before.fields[6](i, 0, 0) * cv_v + old_liquid * 4186 + old_ice * 2106));
        moist[2] -= dry_mass * (new_ice - old_ice) * microphysics_constants::L_f;
        if (coupled)
        {
            moist[2] += dry_mass * (static_cast<long double>(qv(i, 0, 0)) - before.fields[6](i, 0, 0))
                * (microphysics_constants::L_v - microphysics_constants::R_v * microphysics_constants::T0);
        }
        const long double kinetic = .5L * (std::pow(static_cast<long double>(u(i, 0, 0)), 2)
            + std::pow(static_cast<long double>(v(i, 0, 0)), 2) + std::pow(static_cast<long double>(w(i, 0, 0)), 2));
        moist[3] += dry_mass * (total_after - total_before) * kinetic;
        const long double height = global_grid_geometry.z.empty() ? .5L * dz : global_grid_geometry.z[0];
        moist[4] += dry_mass * (total_after - total_before) * physical_constants::gravity_ms2 * height;
    }
    const Snapshot after;
    for (const std::size_t field : {0, 1, 2, 3, 4, 8, 9, 13, 14})
    {
        CHECK(same_bytes(after.fields[field], before.fields[field]));
    }
    if (!coupled)
    {
        CHECK(same_bytes(after.fields[6], before.fields[6]));
    }
    for (const std::size_t moment : {1, 2, 6, 7})
    {
        CHECK(same_bytes(after.moments[moment], before.moments[moment]));
    }
    for (const int i : {0, NR - 1})
    {
        for (std::size_t field = 0; field < before.fields.size(); ++field)
        {
            CHECK(std::memcmp(&after.fields[field](i, 0, 0), &before.fields[field](i, 0, 0), sizeof(float)) == 0);
        }
        for (std::size_t moment = 0; moment < before.moments.size(); ++moment)
        {
            CHECK(std::memcmp(&after.moments[moment](i, 0, 0), &before.moments[moment](i, 0, 0), sizeof(float)) == 0);
        }
    }
    for (std::size_t species = 0; species < water.size(); ++species)
    {
        CHECK(after.water.species[species].storage_rounding_kg
            == Approx(static_cast<double>(rounding[species])).epsilon(3e-6).margin(1e-15));
        CHECK(after.heat.microphysics_water_tendency_kg_by_species[species]
            == Approx(static_cast<double>(requested[species])).epsilon(3e-6).margin(1e-12));
    }
    CHECK(after.heat.microphysics_heat_j == Approx(static_cast<double>(latent_heat)).epsilon(3e-6).margin(1e-10));
    CHECK(after.heat.theta_storage_rounding_j == Approx(static_cast<double>(theta_rounding)).epsilon(3e-6).margin(1e-10));
    CHECK(after.heat.theta_storage_residual_release_j == 0);
    CHECK(after.heat.theta_storage_residual_j == 0);
    const std::array<double, 6> measured{after.heat.moist_dry_internal_change_j,
        after.heat.moist_water_sensible_change_j, after.heat.moist_phase_reference_change_j,
        after.heat.moist_carrier_kinetic_change_j, after.heat.moist_gravitational_change_j,
        after.heat.moist_subgrid_tke_change_j};
    for (std::size_t reservoir = 0; reservoir < moist.size(); ++reservoir)
    {
        CHECK(measured[reservoir] == Approx(static_cast<double>(moist[reservoir])).epsilon(3e-6).margin(1e-10));
    }
    CHECK(after.heat.moist_energy_complete);
    CHECK(after.heat.moist_energy_closure_residual_j == 0);
    CHECK(after.water.lower_bound_count() == 0);
    CHECK(after.water.lower_bound_roundoff_count() == 0);
    CHECK(after.water.upper_bound_count() == 0);
    CHECK(after.water.nonfinite_count() == 0);
    CHECK(after.heat.theta_tendency_limit_count == 0);
    CHECK(after.heat.theta_nonfinite_count == 0);
    check_no_fallout(before, duration);
}

} // namespace cold_cloud_collection_live

TEST_CASE("live cold cloud collection freezes three competing donors with independent latent and reservoir ledgers",
          "[physics][milbrandt][triple_cold_cloud_collection_internal][direct]")
{
    for (const long double alpha : {0.0L, 2.0L, 20.0L, 200.0L})
    {
        for (int scenario = 0; scenario <= 3; ++scenario)
        {
            const double duration = alpha == 200 ? 1e-8 : 1e-3;
            INFO("cloud collection shape " << alpha << ", scenario " << scenario);
            cold_cloud_collection_live::initialize(scenario, alpha);
            const Snapshot before;
            const auto producer = cold_cloud_collection_live::capture(before, 1, duration).producer;
            CHECK(producer.event_count() == 3);
            if (scenario == 0)
            {
                for (const auto& channel : producer.channels())
                {
                    CHECK(channel.admitted_mass_increment_per_kg > 0);
                    CHECK(channel.admitted_number_increment_per_kg > 0);
                }
            }
            else
            {
                CHECK(producer.channels()[scenario - 1].admitted_mass_increment_per_kg > 0);
            }
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            cold_cloud_collection_live::check_step(before, duration);
        }
    }
    cold_cloud_collection_live::initialize(0, 2, false);
    const Snapshot before;
    const auto result = cold_cloud_collection_live::capture(before, 1, 1e-3);
    CHECK_FALSE(result.producer.hail_processes());
    CHECK(result.mass_increment_per_kg[2] > 0);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-3));
    cold_cloud_collection_live::check_step(before, 1e-3);
    cold_cloud_collection_live::initialize(0, 2, true, 0, 1e8f);
    const Snapshot empty;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-3));
    cold_cloud_collection_live::check_step(empty, 1e-3);
}

TEST_CASE("live cold cloud collection continues identically with actual one and four workers",
          "[physics][milbrandt][triple_cold_cloud_collection_internal][direct]")
{
#ifdef _OPENMP
    const int original_threads = omp_get_max_threads();
    const int original_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    std::array<std::unique_ptr<Snapshot>, 2> final;
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        omp_set_num_threads(workers);
        int actual = 0;
#pragma omp parallel
        {
#pragma omp single
            actual = omp_get_num_threads();
        }
        REQUIRE(actual == workers);
        cold_cloud_collection_live::initialize();
        for (const double duration : {1e-3, 2e-3, 5e-3})
        {
            const Snapshot before;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            cold_cloud_collection_live::check_step(before, duration);
            simulation_time += duration;
        }
        final[run] = std::make_unique<Snapshot>();
    }
    check_snapshot(*final[0], *final[1]);
    omp_set_num_threads(original_threads);
    omp_set_dynamic(original_dynamic);
#else
    FAIL("cold cloud collection worker evidence requires OpenMP");
#endif
}

TEST_CASE("live cold cloud collection early record and late ledger failures restore cloud and frozen owners",
          "[physics][milbrandt][triple_cold_cloud_collection_internal][direct]")
{
    for (int mutation = 1; mutation <= 25; ++mutation)
    {
        cold_cloud_collection_live::initialize();
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        INFO("cold cloud getter mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-3));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-3));
        cold_cloud_collection_live::check_step(before, 1e-3);
    }
    for (const int mutation : {0, 1, 2, 6, 7, 8})
    {
        cold_cloud_collection_live::initialize();
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        INFO("cold cloud late mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-3));
        CHECK(accepted_attempts == 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-3));
        cold_cloud_collection_live::check_step(before, 1e-3);
    }
}

TEST_CASE("live cold cloud collection zero events validate raw temperature all private and held owners",
          "[physics][milbrandt][triple_cold_cloud_collection_internal][direct]")
{
    for (int mutation = 0; mutation < 48; ++mutation)
    {
        cold_cloud_collection_live::initialize(4);
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        if (mutation >= 29 && mutation < 41)
        {
            auto* held = held_fields()[mutation - 29];
            held->resize(NR, NTH, NZ, 0);
            (*held)(last, 0, 0) = 1;
        }
        else if (mutation >= 41)
        {
            *held_radiation_powers()[mutation - 41] = 1;
        }
        else
        {
            switch (mutation)
            {
                case 0: rho(last, 0, 0) = 0; break;
                case 1: p(last, 0, 0) = 0; break;
                case 2: theta(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 3: qv(last, 0, 0) = -1; break;
                case 4: qc(last, 0, 0) = -1; break;
                case 5: qr(last, 0, 0) = 1e-4f; break;
                case 6: qi(last, 0, 0) = 1e-4f; break;
                case 7: qs(last, 0, 0) = 1e-4f; break;
                case 8: qv.resize(1, 1, 1, 0); break;
                case 9: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                case 10: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
                case 11: global_simulation_storage.runtime.compute.fallback_active = true; break;
                case 12: global_simulation_storage.configuration.microphysics_config.milbrandt.triple_moment = true; break;
                case 13: global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes = false; break;
                case 14: global_validation_policy.mode = tmv::GuardMode::Warn; break;
                case 15: const_cast<Field3D&>(scheme.internal_triple_number_for_test(1))(last, 0, 0) = 1; break;
                case 16: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(2))(last, 0, 0) = 1e-8f; break;
                case 17: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = 0; break;
                case 18: dtheta_dt_rad(last, 0, 0) = 1; break;
                case 19: global_simulation_storage.runtime.chaos.noise_elapsed_s = 1; break;
                case 20: qg(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 21: qh(last, 0, 0) = -1; break;
                case 22: const_cast<Field3D&>(scheme.internal_triple_number_for_test(0))(last, 0, 0) = 1; break;
                case 23: theta(last, 0, 0) = 149; p(last, 0, 0) = 100000; break;
                case 24: theta(last, 0, 0) = -1; p(last, 0, 0) = 200000; break;
                case 25: theta(last, 0, 0) = 701; break;
                case 26: theta(last, 0, 0) = static_cast<float>(280 / std::pow(static_cast<double>(p(last, 0, 0)) / microphysics_constants::p0,
                    microphysics_constants::R_d / microphysics_constants::cp)); break;
                case 27: global_validation_policy.fail_on = tmv::GuardFailOn::NonFinite; break;
                case 28: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
            }
        }
        const Snapshot before;
        INFO("cold cloud zero mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-3));
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
    }
}

TEST_CASE("live cold cloud collection binds private cloud source generations and portable selection",
          "[physics][milbrandt][triple_cold_cloud_collection_internal][direct]")
{
    cold_cloud_collection_live::initialize();
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    std::string error;
    tmv::MicrophysicsPrimaryEndpointCandidate absent;
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-3, absent, error));
    std::array<Field3D, 8> out;
    auto compute = [&](const Field3D& cloud)
    {
        scheme.compute_tendencies(p, theta, rho, qv, cloud, qr, qi, qs, qg, qh, 1e-3,
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    };
    REQUIRE_NOTHROW(compute(qc));
    tmv::MicrophysicsPrimaryEndpointCandidate first;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-3, first, error));
    auto& nc = const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test());
    const float original_nc = nc(1, 0, 0);
    nc(1, 0, 0) = std::nextafter(original_nc, std::numeric_limits<float>::infinity());
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-3, first, error));
    nc(1, 0, 0) = original_nc;
    REQUIRE_NOTHROW(compute(qc));
    tmv::MicrophysicsPrimaryEndpointCandidate second;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-3, second, error));
    CHECK(second.generation > first.generation);
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-3, first, error));
    const Field3D detached(qc);
    for (auto& field : out)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    CHECK_THROWS(compute(detached));
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-3, absent, error));
    for (const auto& field : out)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    CHECK_THROWS(scheme.portable_state());
    CHECK_THROWS(scheme.clone_parameters());
    CHECK_THROWS(scheme.prognostic_scalars_for_transport());
    const auto identity = scheme.parameter_identity();
    tmv::MicrophysicsConfig changed;
    changed.milbrandt.hail_processes = false;
    CHECK_THROWS(scheme.configure(changed));
    CHECK(scheme.parameter_identity() == identity);
    const Snapshot before;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-3));
    cold_cloud_collection_live::check_step(before, 1e-3);
    CHECK_FALSE(dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme)
        .validate_primary_endpoint_commit(global_simulation_state(), 1e-3, second, error));
    CHECK_FALSE(dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).commit_prepared_private_candidate());
}

TEST_CASE("live cold cloud collection activation rejects raw theta and laws without taking ownership",
          "[physics][milbrandt][triple_cold_cloud_collection_internal][direct]")
{
    const std::array<std::array<float, 2>, 4> invalid_values{{
        {149, 100000}, {-1, 200000}, {701, 100000},
        {std::numeric_limits<float>::quiet_NaN(), 100000}}};
    for (const auto invalid : invalid_values)
    {
        cold_cloud_collection_live::initialize(4);
        const Snapshot original;
        theta(1, 0, 0) = invalid[0];
        p(1, 0, 0) = invalid[1];
        const Snapshot bad;
        tmv::MilbrandtScheme candidate;
        std::string error;
        CHECK_FALSE(candidate.enable_internal_triple_cold_cloud_collection_for_test(global_simulation_state(),
            bad.moments[0], bad.moments[3], bad.moments[8], bad.moments[4], bad.moments[9],
            bad.moments[5], bad.moments[10], error));
        CHECK_FALSE(candidate.owns_primary_endpoint_commit());
        check_snapshot(bad);
        theta = original.fields[5];
        p = original.fields[1];
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-3));
        cold_cloud_collection_live::check_step(original, 1e-3);
    }
    cold_cloud_collection_live::initialize();
    const Snapshot source;
    tmv::MilbrandtScheme changed(0, 0, 0, 0, 0, 523.598775598, 3, 440, 3, .1597,
        std::nextafter(2.078, 3.0));
    std::string error;
    CHECK_FALSE(changed.enable_internal_triple_cold_cloud_collection_for_test(global_simulation_state(),
        source.moments[0], source.moments[3], source.moments[8], source.moments[4], source.moments[9],
        source.moments[5], source.moments[10], error));
    CHECK_FALSE(changed.owns_primary_endpoint_commit());
    check_snapshot(source);
}

TEST_CASE("live cold cloud collection rejects stored theta phase and raw theta range exits atomically",
          "[physics][milbrandt][triple_cold_cloud_collection_internal][direct]")
{
    cold_cloud_collection_live::initialize();
    const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
    bool selected = false;
    double selected_exner = 0;
    float lower_theta = 0;
    float upper_theta = 0;
    for (const float pressure : {65000.0f, 70000.0f, 75000.0f, 80000.0f, 85000.0f, 90000.0f, 95000.0f})
    {
        const double exner = std::pow(static_cast<double>(pressure) / microphysics_constants::p0,
            microphysics_constants::R_d / microphysics_constants::cp);
        const double threshold = microphysics_constants::T0 / exner;
        const float nearest = static_cast<float>(threshold);
        const float lower = static_cast<double>(nearest) < threshold
            ? nearest : std::nextafter(nearest, -std::numeric_limits<float>::infinity());
        const float upper = std::nextafter(lower, std::numeric_limits<float>::infinity());
        if (threshold > (static_cast<double>(lower) + upper) * .5)
        {
            p(last, 0, 0) = pressure;
            theta(last, 0, 0) = lower;
            selected_exner = exner;
            lower_theta = lower;
            upper_theta = upper;
            selected = true;
            break;
        }
    }
    REQUIRE(selected);
    const Snapshot near_freezing;
    const double source_temperature = static_cast<double>(lower_theta) * selected_exner;
    REQUIRE(source_temperature < microphysics_constants::T0);
    const double midpoint_temperature = (static_cast<double>(lower_theta) + upper_theta) * .5 * selected_exner;
    const double target_temperature = (midpoint_temperature + microphysics_constants::T0) * .5;
    const auto probe = cold_cloud_collection_live::capture(near_freezing, last, 1e-6);
    REQUIRE(probe.temperature_increment_k > 0);
    const double duration = (target_temperature - source_temperature) / (probe.temperature_increment_k / 1e-6);
    const auto admissible = cold_cloud_collection_live::capture(near_freezing, last, duration);
    const double theta_increment = admissible.temperature_increment_k / selected_exner;
    REQUIRE(source_temperature + admissible.temperature_increment_k < microphysics_constants::T0);
    REQUIRE(static_cast<double>(static_cast<float>(static_cast<double>(lower_theta) + theta_increment))
        * selected_exner >= microphysics_constants::T0);
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), duration), Catch::Matchers::Contains("final heat"));
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(near_freezing);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration * .1));
    cold_cloud_collection_live::check_step(near_freezing, duration * .1);

    cold_cloud_collection_live::initialize(3, 2, true, .04f, 1e8f);
    for (int i = 0; i < NR; ++i)
    {
        p(i, 0, 0) = 500;
        theta(i, 0, 0) = 690;
        qh(i, 0, 0) = .02f;
    }
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    const float z = frozen_collection_live::reflectivity(3, .02f, 100, 2);
    const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(4)).resize(NR, NTH, NZ, z);
    const Snapshot high_theta;
    const double exner = std::pow(500.0 / microphysics_constants::p0,
        microphysics_constants::R_d / microphysics_constants::cp);
    REQUIRE(690 * exner >= 150);
    REQUIRE(690 * exner < microphysics_constants::T0);
    const auto rate_probe = cold_cloud_collection_live::capture(high_theta, 1, 1e-6);
    REQUIRE(rate_probe.temperature_increment_k > 0);
    const double large_duration = 20 * exner / (rate_probe.temperature_increment_k / 1e-6);
    const auto valid_cold = cold_cloud_collection_live::capture(high_theta, 1, large_duration);
    REQUIRE(690 * exner + valid_cold.temperature_increment_k < microphysics_constants::T0);
    REQUIRE(690 + valid_cold.temperature_increment_k / exner > 700);
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), large_duration), Catch::Matchers::Contains("final heat"));
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(high_theta);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), large_duration * .1));
    cold_cloud_collection_live::check_step(high_theta, large_duration * .1);
}

namespace internal_thermal_live
{

void clear_water_and_moments()
{
    for (Field3D* field : {&qv, &qc, &qr, &qi, &qs, &qg, &qh})
    {
        field->resize(NR, NTH, NZ, 0);
    }
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test()).resize(NR, NTH, NZ, 0);
    for (std::size_t species = 0; species < 5; ++species)
    {
        const_cast<Field3D&>(scheme.internal_triple_number_for_test(species)).resize(NR, NTH, NZ, 0);
        const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(species)).resize(NR, NTH, NZ, 0);
    }
}

bool activate(ObservedMilbrandt& candidate, int mode, const Snapshot& source, std::string& error)
{
    const auto state = global_simulation_state();
    const auto& n = source.moments;
    switch (mode)
    {
        case 0: return candidate.enable_internal_triple_frozen_composition_for_test(state, n[4], n[9], n[5], n[10], error);
        case 1: return candidate.enable_internal_triple_melting_for_test(state, n[1], n[6], n[4], n[9], n[5], n[10], error);
        case 2: return candidate.enable_internal_triple_ice_melting_for_test(state, n[1], n[6], n[2], n[7], error);
        case 3: return candidate.enable_internal_triple_snow_melting_for_test(state, n[1], n[6], n[3], n[8], error);
        case 4: return candidate.enable_internal_triple_warm_primary_for_test(state, n[0], n[1], n[6], error);
        case 5: return candidate.enable_internal_triple_cloud_adjustment_for_test(state, n[0], error);
        case 6: return candidate.enable_internal_triple_warm_cloud_primary_for_test(state, n[0], n[1], n[6], error);
        case 7: return candidate.enable_internal_triple_dynamic_warm_cloud_primary_for_test(state, n[0], n[1], n[6], error);
        case 8: return candidate.enable_internal_triple_subcycled_warm_cloud_primary_for_test(state, n[0], n[1], n[6], error);
        case 9: return candidate.enable_internal_triple_frozen_collection_for_test(state,
            n[2], n[7], n[3], n[8], n[4], n[9], n[5], n[10], error);
        case 10: return candidate.enable_internal_triple_cold_cloud_collection_for_test(state,
            n[0], n[3], n[8], n[4], n[9], n[5], n[10], error);
        case 11: return candidate.enable_internal_triple_warm_for_test(state, n[0], n[1], n[6], error);
        case 12: return candidate.enable_internal_triple_warm_rain_sedimentation_for_test(state, n[0], n[1], n[6], error);
        case 13: return candidate.enable_internal_triple_rain_freezing_for_test(state, n[1], n[6], error);
        case 14: return candidate.enable_internal_triple_frozen_diffusion_for_test(state,
            TripleMomentSpecies::Graupel, n[4], n[9], error);
        case 15: return candidate.enable_internal_triple_frozen_diffusion_for_test(state,
            TripleMomentSpecies::Hail, n[5], n[10], error);
        case 16: return candidate.enable_internal_triple_cold_cloud_diffusion_for_test(state,
            n[0], n[3], n[8], n[4], n[9], n[5], n[10], error);
    }
    return false;
}

void initialize(int mode)
{
    switch (mode)
    {
        case 0: initialize_state(); break;
        case 1: melting_live::initialize(); break;
        case 2: ice_melting_live::initialize(); break;
        case 3: snow_melting_live::initialize(); break;
        case 4: warm_primary_live::initialize(); break;
        case 5: cloud_adjustment_live::initialize(); break;
        case 6: warm_cloud_live::initialize(); break;
        case 7: dynamic_warm_cloud_live::initialize(); break;
        case 8: subcycled_warm_cloud_live::initialize(); break;
        case 9: frozen_collection_live::initialize(); break;
        case 10: cold_cloud_collection_live::initialize(); break;
        case 16: cold_cloud_collection_live::initialize(); break;
        case 11:
        case 12: warm_primary_live::initialize(); break;
        default: initialize_state(); break;
    }
    clear_water_and_moments();
    const Snapshot source;
    auto candidate = std::make_unique<ObservedMilbrandt>();
    std::string error;
    const bool selected = activate(*candidate, mode, source, error);
    INFO(error);
    REQUIRE(selected);
    microphysics_scheme = std::move(candidate);
}

void check_empty_recovery(const Snapshot& before)
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
    CHECK(after.heat.microphysics_heat_j == 0);
    CHECK(after.heat.theta_storage_rounding_j == 0);
    CHECK(after.heat.moist_energy_closure_residual_j == 0);
    CHECK(after.water.nonfinite_tendency_count == 0);
    CHECK(after.water.lower_bound_count() == 0);
    CHECK(after.water.upper_bound_count() == 0);
}

void compute(std::array<Field3D, 8>& out, double duration)
{
    dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).compute_tendencies(
        p, theta, rho, qv, qc, qr, qi, qs, qg, qh, duration,
        out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
}

} // namespace internal_thermal_live

TEST_CASE("internal triple thermal owners reject unclamped source theta before activation or zero publication",
          "[physics][milbrandt][triple_internal_thermal][direct]")
{
    const std::array<std::array<float, 2>, 7> invalid_values{{
        {149, 100000}, {-1, 200000}, {701, 500},
        {std::numeric_limits<float>::quiet_NaN(), 100000},
        {std::numeric_limits<float>::infinity(), 100000}, {149, 200000}, {701, 100000}}};
    for (int mode = 0; mode < 17; ++mode)
    {
        for (const auto invalid : invalid_values)
        {
            internal_thermal_live::initialize(mode);
            const Snapshot original;
            const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
            theta(last, 0, 0) = invalid[0];
            p(last, 0, 0) = invalid[1];
            const Snapshot bad;
            ObservedMilbrandt candidate;
            std::string error;
            INFO("thermal mode " << mode << ", theta " << invalid[0] << ", pressure " << invalid[1]);
            CHECK_FALSE(internal_thermal_live::activate(candidate, mode, bad, error));
            CHECK_FALSE(candidate.owns_post_primary_private_commit());
            check_snapshot(bad);
            REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
            CHECK(getter_attempts == 0);
            CHECK(accepted_attempts == 0);
            check_snapshot(bad);
            theta = original.fields[5];
            p = original.fields[1];
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
            internal_thermal_live::check_empty_recovery(original);
        }
    }
}

TEST_CASE("internal warm and rain freezing reject real stored theta phase crossings atomically and recover",
          "[physics][milbrandt][triple_internal_thermal][direct]")
{
    for (const int mode : {11, 12, 13})
    {
        internal_thermal_live::initialize(mode);
        const bool warm = mode != 13;
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        bool selected = false;
        double selected_exner = 0;
        float lower = 0;
        float upper = 0;
        for (const float pressure : {65000.0f, 70000.0f, 75000.0f, 80000.0f, 85000.0f, 90000.0f, 95000.0f})
        {
            const double exner = std::pow(static_cast<double>(pressure) / microphysics_constants::p0,
                microphysics_constants::R_d / microphysics_constants::cp);
            const double threshold = microphysics_constants::T0 / exner;
            const float nearest = static_cast<float>(threshold);
            const float floor = nearest < threshold ? nearest : std::nextafter(nearest, -std::numeric_limits<float>::infinity());
            const float ceil = std::nextafter(floor, std::numeric_limits<float>::infinity());
            const double midpoint = (static_cast<double>(floor) + ceil) * .5;
            if (warm ? midpoint > threshold : midpoint < threshold)
            {
                p(last, 0, 0) = pressure;
                theta(last, 0, 0) = warm ? ceil : static_cast<float>((microphysics_constants::T0 - 11) / exner);
                lower = floor;
                upper = ceil;
                selected_exner = exner;
                selected = true;
                break;
            }
        }
        REQUIRE(selected);
        qr(last, 0, 0) = warm ? 1e-4f : .04f;
        const float number = warm ? 1e3f : 1e-4f;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        const_cast<Field3D&>(scheme.internal_triple_number_for_test(0))(last, 0, 0) = number;
        const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(0))(last, 0, 0)
            = static_cast<float>(shape_ratio(warm ? 20 : 2) * static_cast<long double>(qr(last, 0, 0)) * qr(last, 0, 0)
                / (523.598775598L * 523.598775598L * number));
        const Snapshot before;
        std::array<Field3D, 8> probe;
        REQUIRE_NOTHROW(internal_thermal_live::compute(probe, 1e-8));
        const double rate = probe[0](last, 0, 0);
        REQUIRE((warm ? rate < 0 : rate > 0));
        const double target_theta = ((static_cast<double>(lower) + upper) * .5
            + microphysics_constants::T0 / selected_exner) * .5;
        const double duration = (target_theta - theta(last, 0, 0)) / rate;
        REQUIRE(duration > 0);
        const double unrounded = static_cast<double>(theta(last, 0, 0)) + duration * rate;
        const float stored = static_cast<float>(unrounded);
        REQUIRE((warm ? unrounded * selected_exner > microphysics_constants::T0
                      : unrounded * selected_exner < microphysics_constants::T0));
        REQUIRE((warm ? static_cast<double>(stored) * selected_exner <= microphysics_constants::T0
                      : static_cast<double>(stored) * selected_exner >= microphysics_constants::T0));
        INFO("thermal phase mode " << mode << ", duration " << duration);
        REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), duration),
            Catch::Matchers::Contains(warm ? "temperature" : "final heat"));
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration * .01));
    }
}

namespace cold_cloud_diffusion_live
{

void initialize(int scenario = 0, long double alpha = 2, bool hail = true,
                double saturation = .75, float cloud = 1e-3f)
{
    cold_cloud_collection_live::initialize(scenario, alpha, hail, cloud);
    for (int i = 0; i < NR; ++i)
    {
        const double temperature = static_cast<double>(theta(i, 0, 0))
            * std::pow(static_cast<double>(p(i, 0, 0)) / microphysics_constants::p0,
                       microphysics_constants::R_d / microphysics_constants::cp);
        qv(i, 0, 0) = static_cast<float>(saturation
            * thermodynamics::saturation_mixing_ratio_ice(temperature, p(i, 0, 0)));
    }
    const Snapshot source;
    auto candidate = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = hail;
    candidate->configure(config);
    std::string error;
    const bool selected = candidate->enable_internal_triple_cold_cloud_diffusion_for_test(global_simulation_state(),
        source.moments[0], source.moments[3], source.moments[8], source.moments[4], source.moments[9],
        source.moments[5], source.moments[10], error);
    INFO(error);
    REQUIRE(selected);
    REQUIRE(candidate->owns_primary_endpoint_commit());
    REQUIRE(candidate->protected_primary_endpoint_wind_source() == nullptr);
    microphysics_scheme = std::move(candidate);
}

} // namespace cold_cloud_diffusion_live

TEST_CASE("live collection coupled frozen diffusion publishes one combined latent and moist energy endpoint",
          "[physics][milbrandt][triple_cold_cloud_diffusion_internal][direct]")
{
    for (const long double alpha : {0.0L, 2.0L, 20.0L, 200.0L})
    {
        for (const double saturation : {.75, 1.0, 1.05})
        {
            for (int scenario = 0; scenario <= 3; ++scenario)
            {
                const double duration = alpha == 200 ? 1e-8 : 1e-4;
                INFO("coupled shape " << alpha << ", saturation " << saturation << ", scenario " << scenario);
                cold_cloud_diffusion_live::initialize(scenario, alpha, true, saturation);
                const Snapshot before;
                const auto result = cold_cloud_collection_live::capture_coupled(before, 1, duration);
                CHECK(result.producer.event_count() == 3);
                CHECK(result.producer.collection_producer().event_count() == 3);
                CHECK(result.producer.thermodynamics().relaxation_number <= 1);
                const auto& collection = result.producer.collection_producer().channels();
                const auto& diffusion = result.producer.channels();
                double collected = 0;
                double deposited = 0;
                for (std::size_t species = 0; species < 3; ++species)
                {
                    collected += collection[species].admitted_mass_increment_per_kg;
                    deposited += diffusion[species].admitted_mass_increment_per_kg;
                }
                CHECK(result.temperature_increment_k == Approx(
                    (microphysics_constants::L_f * collected + microphysics_constants::L_s * deposited)
                    / microphysics_constants::cp).epsilon(8e-12).margin(1e-14));
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                cold_cloud_collection_live::check_step(before, duration, true);
            }
        }
    }
    for (const bool hail : {false, true})
    {
        cold_cloud_diffusion_live::initialize(0, 2, hail, 1.05, 0);
        const Snapshot before;
        const auto result = cold_cloud_collection_live::capture_coupled(before, 1, 1e-4);
        CHECK(result.producer.collection_producer().hail_processes() == hail);
        CHECK(result.cloud_mass_increment_per_kg == 0);
        CHECK(result.cloud_number_increment_per_kg == 0);
        CHECK(result.vapor_mass_increment_per_kg < 0);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_cloud_collection_live::check_step(before, 1e-4, true);
    }
}

TEST_CASE("live coupled cloud diffusion continues identically with actual one and four workers",
          "[physics][milbrandt][triple_cold_cloud_diffusion_internal][direct]")
{
#ifdef _OPENMP
    const int original_threads = omp_get_max_threads();
    const int original_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    for (const double saturation : {.75, 1.05})
    {
        std::array<std::unique_ptr<Snapshot>, 2> final;
        for (int run = 0; run < 2; ++run)
        {
            const int workers = run == 0 ? 1 : 4;
            omp_set_num_threads(workers);
            int actual = 0;
#pragma omp parallel
            {
#pragma omp single
                actual = omp_get_num_threads();
            }
            REQUIRE(actual == workers);
            cold_cloud_diffusion_live::initialize(0, 2, true, saturation);
            for (const double duration : {1e-4, 2e-4, 5e-4})
            {
                const Snapshot before;
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                cold_cloud_collection_live::check_step(before, duration, true);
                simulation_time += duration;
            }
            final[run] = std::make_unique<Snapshot>();
        }
        check_snapshot(*final[0], *final[1]);
    }
    omp_set_num_threads(original_threads);
    omp_set_dynamic(original_dynamic);
#else
    FAIL("coupled cold cloud diffusion worker evidence requires OpenMP");
#endif
}

TEST_CASE("live coupled cloud diffusion rejected records and late failures restore every source and endpoint owner",
          "[physics][milbrandt][triple_cold_cloud_diffusion_internal][direct]")
{
    for (int mutation = 1; mutation <= 25; ++mutation)
    {
        cold_cloud_diffusion_live::initialize();
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        INFO("coupled getter mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_cloud_collection_live::check_step(before, 1e-4, true);
    }
    for (const int mutation : {0, 1, 2, 6, 7, 8})
    {
        cold_cloud_diffusion_live::initialize();
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        INFO("coupled late mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(accepted_attempts == 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_cloud_collection_live::check_step(before, 1e-4, true);
    }
}

TEST_CASE("live coupled cloud diffusion validates zero source context native and held owners before publication",
          "[physics][milbrandt][triple_cold_cloud_diffusion_internal][direct]")
{
    for (int mutation = 0; mutation < 35; ++mutation)
    {
        cold_cloud_diffusion_live::initialize(4, 2, true, 1, 0);
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        if (mutation >= 16 && mutation < 28)
        {
            auto* held = held_fields()[mutation - 16];
            held->resize(NR, NTH, NZ, 0);
            (*held)(last, 0, 0) = 1;
        }
        else if (mutation >= 28)
        {
            *held_radiation_powers()[mutation - 28] = 1;
        }
        else
        {
            switch (mutation)
            {
                case 0: qv(last, 0, 0) = -1; break;
                case 1: qv(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 2: qr(last, 0, 0) = 1e-4f; break;
                case 3: qi(last, 0, 0) = 1e-4f; break;
                case 4: rho(last, 0, 0) = 0; break;
                case 5: p(last, 0, 0) = 0; break;
                case 6: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = -1; break;
                case 7: const_cast<Field3D&>(scheme.internal_triple_number_for_test(0))(last, 0, 0) = 1; break;
                case 8: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(1))(last, 0, 0) = 1; break;
                case 9: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                case 10: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
                case 11: global_simulation_storage.runtime.compute.fallback_active = true; break;
                case 12: global_simulation_storage.configuration.microphysics_config.milbrandt.triple_moment = true; break;
                case 13: global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes = false; break;
                case 14: global_validation_policy.mode = tmv::GuardMode::Warn; break;
                case 15: scheme.set_require_native_dispatch(true); break;
            }
        }
        const Snapshot bad;
        INFO("coupled zero mutation " << mutation);
        if (mutation == 15)
        {
            std::array<Field3D, 8> unpublished;
            for (auto& field : unpublished)
            {
                field.resize(NR, NTH, NZ, 42);
            }
            REQUIRE_THROWS_WITH(scheme.compute_tendencies(p, theta, rho, qv, qc, qr, qi, qs, qg, qh, 1e-4,
                unpublished[0], unpublished[1], unpublished[2], unpublished[3],
                unpublished[4], unpublished[5], unpublished[6], unpublished[7]),
                Catch::Matchers::Contains("live CPU direct-step owner"));
            for (const auto& field : unpublished)
            {
                for (std::size_t index = 0; index < field.size(); ++index)
                {
                    CHECK(field.data()[index] == 42);
                }
            }
        }
        else
        {
            REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        }
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(bad);
    }
}

TEST_CASE("live coupled diffusion source journals resource guards and stale endpoints reject atomically and recover",
          "[physics][milbrandt][triple_cold_cloud_diffusion_internal][direct]")
{
    cold_cloud_diffusion_live::initialize();
    const Snapshot before;
    const auto result = cold_cloud_collection_live::capture_coupled(before, 1, 1e-4);
    REQUIRE(result.producer.thermodynamics().relaxation_number > 0);
    const double unsupported = 1.1e-4 / result.producer.thermodynamics().relaxation_number;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), unsupported), Catch::Matchers::Contains("relaxation"));
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(before);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_cloud_collection_live::check_step(before, 1e-4, true);

    cold_cloud_diffusion_live::initialize();
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    std::array<Field3D, 8> out;
    std::string error;
    tmv::MicrophysicsPrimaryEndpointCandidate absent;
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, absent, error));
    internal_thermal_live::compute(out, 1e-4);
    tmv::MicrophysicsPrimaryEndpointCandidate first;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    const float old_vapor = qv(1, 0, 0);
    qv(1, 0, 0) = std::nextafter(old_vapor, std::numeric_limits<float>::infinity());
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    qv(1, 0, 0) = old_vapor;
    auto& nc = const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test());
    const float old_nc = nc(1, 0, 0);
    nc(1, 0, 0) = std::nextafter(old_nc, std::numeric_limits<float>::infinity());
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    nc(1, 0, 0) = old_nc;
    internal_thermal_live::compute(out, 1e-4);
    tmv::MicrophysicsPrimaryEndpointCandidate second;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, second, error));
    CHECK(second.generation > first.generation);
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    const Field3D detached(qv);
    for (auto& field : out)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    CHECK_THROWS(scheme.compute_tendencies(p, theta, rho, detached, qc, qr, qi, qs, qg, qh, 1e-4,
        out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]));
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, absent, error));
    for (const auto& field : out)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    CHECK_THROWS(scheme.portable_state());
    CHECK_THROWS(scheme.clone_parameters());
    CHECK_THROWS(scheme.prognostic_scalars_for_transport());
    const Snapshot recovered;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_cloud_collection_live::check_step(recovered, 1e-4, true);
    CHECK_FALSE(dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme)
        .validate_primary_endpoint_commit(global_simulation_state(), 1e-4, second, error));
}

TEST_CASE("ordinary double cold rejection crosses worker boundaries without publishing state",
          "[physics][milbrandt][double_cold_runtime][direct]")
{
#ifdef _OPENMP
    struct ThreadRestore
    {
        int threads = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~ThreadRestore()
        {
            omp_set_num_threads(threads);
            omp_set_dynamic(dynamic);
        }
    } restore;
    for (int workers : {1, 4})
    {
        omp_set_dynamic(0);
        omp_set_num_threads(workers);
        int actual_workers = 0;
#pragma omp parallel
        {
#pragma omp single
            actual_workers = omp_get_num_threads();
        }
        REQUIRE(actual_workers == workers);
        initialize_state();
        p.resize(NR, NTH, NZ, 100000);
        rho.resize(NR, NTH, NZ, 1);
        theta.resize(NR, NTH, NZ, 272.15f);
        qv.resize(NR, NTH, NZ, static_cast<float>(thermodynamics::saturation_mixing_ratio_water(272.15f, 100000)));
        qc.resize(NR, NTH, NZ, .001f);
        qs.resize(NR, NTH, NZ, .003f);
        for (Field3D* field : {&qr, &qi, &qg, &qh})
        {
            field->resize(NR, NTH, NZ, 0);
        }
        microphysics_scheme = std::make_unique<tmv::MilbrandtScheme>();
        auto* scheme = dynamic_cast<tmv::MilbrandtScheme*>(microphysics_scheme.get());
        scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
        auto moments = scheme->prognostic_scalars_for_transport();
        moments[0].field->resize(NR, NTH, NZ, 2e8f);
        moments[3].field->resize(NR, NTH, NZ, 1e-20f);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), .2));
        std::cout << "old source accepted workers=" << workers << '\n';

        initialize_state();
        p.resize(NR, NTH, NZ, 100000);
        rho.resize(NR, NTH, NZ, 1);
        theta.resize(NR, NTH, NZ, 272.15f);
        qv.resize(NR, NTH, NZ, static_cast<float>(thermodynamics::saturation_mixing_ratio_water(272.15f, 100000)));
        for (Field3D* field : {&qc, &qr, &qi, &qs, &qg, &qh})
        {
            field->resize(NR, NTH, NZ, 0);
        }
        constexpr std::array<double, 16> cell{265.80990600585938, 88899.6875, 0.94878780841827393,
            0, 3.4392439118846596e-8, 1.3518631931219716e-5, 0.014146114699542522,
            0.015282873064279556, 2.7798416340374388e-5, 0.010872337035834789,
            10540.935546875, 24819536, 4956.21484375, 676.070068359375, 264328.34375, 52801356};
        const int index = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        p(index, 0, 0) = static_cast<float>(cell[1]);
        rho(index, 0, 0) = static_cast<float>(cell[2]);
        theta(index, 0, 0) = static_cast<float>(cell[0] / std::pow(cell[1] / 100000.0, 287.0 / 1004.0));
        qv(index, 0, 0) = static_cast<float>(thermodynamics::saturation_mixing_ratio_water(cell[0], cell[1]));
        const std::array<Field3D*, 6> mass{&qc, &qr, &qi, &qs, &qg, &qh};
        for (std::size_t species = 0; species < mass.size(); ++species)
        {
            (*mass[species])(index, 0, 0) = static_cast<float>(cell[4 + species]);
        }
        microphysics_scheme = std::make_unique<tmv::MilbrandtScheme>();
        scheme = dynamic_cast<tmv::MilbrandtScheme*>(microphysics_scheme.get());
        scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
        moments = scheme->prognostic_scalars_for_transport();
        for (std::size_t species = 0; species < moments.size(); ++species)
        {
            (*moments[species].field)(index, 0, 0) = static_cast<float>(cell[10 + species]);
        }
        const auto private_before = scheme->portable_state();
        const auto owners = public_fields();
        std::array<Field3D, 15> before;
        for (std::size_t field = 0; field < before.size(); ++field)
        {
            before[field] = *owners[field];
        }
        const auto runtime_before = global_microphysics_runtime_state;
        std::cout << std::setprecision(17) << "rejection workers=" << workers << " index=" << index
            << " theta=" << theta(index, 0, 0) << " qv=" << qv(index, 0, 0) << '\n';
        try
        {
            step_microphysics(global_simulation_state(), 1);
            FAIL("finite ordinary cold rejection was accepted");
        }
        catch (const ColdSnowRimingAdmissionError& error)
        {
            REQUIRE(error.kind() == ColdSnowRimingAdmissionFailure::InvalidEndpoint);
            std::cout << "rejection kind=" << static_cast<int>(error.kind()) << " message=" << error.what() << '\n';
        }
        for (std::size_t field = 0; field < before.size(); ++field)
        {
            CHECK(same_bytes(*owners[field], before[field]));
        }
        const auto private_after = microphysics_scheme->portable_state();
        REQUIRE(private_after.fields.size() == private_before.fields.size());
        for (std::size_t field = 0; field < private_before.fields.size(); ++field)
        {
            if (!same_bytes(private_before.fields[field], private_after.fields[field]))
            {
                std::cout << "changed private field=" << field << " last_before="
                    << private_before.fields[field](index, 0, 0) << " last_after="
                    << private_after.fields[field](index, 0, 0) << '\n';
            }
            CHECK(same_bytes(private_before.fields[field], private_after.fields[field]));
        }
        CHECK(private_before.integers == private_after.integers);
        CHECK(private_before.reals == private_after.reals);
        CHECK(thermal_values(global_microphysics_runtime_state.last_thermodynamic_repair_report)
            == thermal_values(runtime_before.last_thermodynamic_repair_report));
        CHECK(std::memcmp(&global_microphysics_runtime_state.last_water_repair_report,
            &runtime_before.last_water_repair_report, sizeof(tmv::MicrophysicsWaterRepairReport)) == 0);
        CHECK(std::memcmp(&global_microphysics_runtime_state.last_conservation_report,
            &runtime_before.last_conservation_report, sizeof(tmv::MicrophysicsConservationReport)) == 0);
        CHECK(same_bytes(global_microphysics_runtime_state.theta_storage_residual, runtime_before.theta_storage_residual));
        p(index, 0, 0) = 100000;
        rho(index, 0, 0) = 1;
        theta(index, 0, 0) = 272.15f;
        qv(index, 0, 0) = static_cast<float>(thermodynamics::saturation_mixing_ratio_water(272.15f, 100000));
        for (auto* field : mass)
        {
            (*field)(index, 0, 0) = 0;
        }
        qc(index, 0, 0) = 2e-4f;
        qs(index, 0, 0) = .003f;
        moments = microphysics_scheme->prognostic_scalars_for_transport();
        for (const auto& moment : moments)
        {
            (*moment.field)(index, 0, 0) = 0;
        }
        (*moments[0].field)(index, 0, 0) = 2e8f;
        (*moments[3].field)(index, 0, 0) = 5000;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-3));
        std::cout << "recovery accepted workers=" << workers << '\n';
    }
#else
    FAIL("ordinary double cold worker evidence requires OpenMP");
#endif
}

TEST_CASE("internal thermal owners reject a direct native requirement without changing outputs or private state",
          "[physics][milbrandt][triple_internal_thermal][direct]")
{
    for (int mode = 0; mode < 17; ++mode)
    {
        internal_thermal_live::initialize(mode);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.set_require_native_dispatch(true);
        const Snapshot before;
        std::array<Field3D, 8> unpublished;
        for (auto& field : unpublished)
        {
            field.resize(NR, NTH, NZ, 42);
        }
        INFO("direct native requirement thermal mode " << mode);
        REQUIRE_THROWS_WITH(internal_thermal_live::compute(unpublished, 1e-4),
            Catch::Matchers::Contains("live CPU direct-step owner"));
        for (const auto& field : unpublished)
        {
            for (std::size_t index = 0; index < field.size(); ++index)
            {
                CHECK(field.data()[index] == 42);
            }
        }
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        scheme.set_require_native_dispatch(false);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        internal_thermal_live::check_empty_recovery(before);
    }
}

namespace cold_riming_live
{

namespace reference = milbrandt_cold_riming_reference;

void initialize(int scenario = 0, long double alpha = 2, double temperature = 268.15,
                double saturation = 1, bool hail = true, float cloud = 1e-3f,
                int conversion_mode = 0, double pressure_override = 0, bool no_rain = false)
{
    const bool hail_shape_mode = conversion_mode >= 4;
    frozen_collection_live::initialize(0, hail_shape_mode ? std::min(alpha, 199.9L) : alpha, hail);
    const Snapshot old;
    std::array<Field3D, 4> number;
    std::array<Field3D, 4> z;
    const std::array<Field3D*, 4> mass{&qi, &qs, &qg, &qh};
    for (std::size_t species = 0; species < 4; ++species)
    {
        number[species] = old.moments[species + 2];
        z[species] = old.moments[species + 7];
        const bool selected = scenario == 0 || (scenario == 1 && species > 0)
            || (scenario >= 2 && scenario <= 5 && species == static_cast<std::size_t>(scenario - 2))
            || (hail_shape_mode && scenario == 7 && species >= 2);
        if (!selected)
        {
            mass[species]->resize(NR, NTH, NZ, 0);
            number[species].resize(NR, NTH, NZ, 0);
            z[species].resize(NR, NTH, NZ, 0);
        }
        else if (hail_shape_mode && alpha > 200 && species >= 2)
        {
            for (int i = 0; i < NR; ++i)
            {
                z[species](i, 0, 0) = frozen_collection_live::reflectivity(
                    species, (*mass[species])(i, 0, 0), number[species](i, 0, 0), alpha);
            }
        }
    }
    qc.resize(NR, NTH, NZ, cloud);
    Field3D nc(NR, NTH, NZ, 1e8f);
    for (int i = 0; i < NR; ++i)
    {
        if (pressure_override > 0)
        {
            p(i, 0, 0) = static_cast<float>(pressure_override);
        }
        const double exner = std::pow(static_cast<double>(p(i, 0, 0)) / microphysics_constants::p0,
            microphysics_constants::R_d / microphysics_constants::cp);
        theta(i, 0, 0) = static_cast<float>(temperature / exner);
        qv(i, 0, 0) = static_cast<float>(saturation * reference::thermal::saturation(
            static_cast<double>(theta(i, 0, 0)) * exner, p(i, 0, 0)).mass);
    }
    auto candidate = std::make_unique<ObservedMilbrandt>();
    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = hail;
    candidate->configure(config);
    const std::array<const Field3D*, 4> n{&number[0], &number[1], &number[2], &number[3]};
    const std::array<const Field3D*, 4> moments{&z[0], &z[1], &z[2], &z[3]};
    std::string error;
    REQUIRE(conversion_mode >= 0);
    REQUIRE(conversion_mode <= 8);
    Field3D rain_number(NR, NTH, NZ, no_rain ? 0.0f : 4000.0f);
    Field3D rain_z(NR, NTH, NZ, 0);
    if (conversion_mode >= 6)
    {
        qr.resize(NR, NTH, NZ, no_rain ? 0.0f : .002f);
        for (int i = 0; i < NR; ++i)
            rain_z(i, 0, 0) = no_rain ? 0.0f : static_cast<float>(milbrandt_cold_rain_sll_reference::rain_population(qr(i, 0, 0),
                rain_number(i, 0, 0), 2).reflectivity_per_kg);
    }
    const bool selected = conversion_mode == 8
        ? candidate->enable_internal_triple_cold_wet_growth_sll_for_test(global_simulation_state(), nc,
            {&rain_number, n[0], n[1], n[2], n[3]}, {&rain_z, moments[0], moments[1], moments[2], moments[3]}, error)
        : conversion_mode == 7
        ? candidate->enable_internal_triple_cold_collision_sll_for_test(global_simulation_state(), nc,
            {&rain_number, n[0], n[1], n[2], n[3]}, {&rain_z, moments[0], moments[1], moments[2], moments[3]}, error)
        : conversion_mode == 6
        ? candidate->enable_internal_triple_cold_rain_sll_for_test(global_simulation_state(), nc,
            {&rain_number, n[0], n[1], n[2], n[3]}, {&rain_z, moments[0], moments[1], moments[2], moments[3]}, error)
        : conversion_mode == 5
        ? candidate->enable_internal_triple_cold_collected_sll_for_test(
            global_simulation_state(), nc, n, moments, error)
        : conversion_mode == 4
        ? candidate->enable_internal_triple_cold_sll_for_test(
            global_simulation_state(), nc, n, moments, error)
        : conversion_mode == 3
        ? candidate->enable_internal_triple_cold_homogeneous_for_test(
            global_simulation_state(), nc, n, moments, error)
        : conversion_mode == 2
        ? candidate->enable_internal_triple_cold_graupel_conversions_for_test(
            global_simulation_state(), nc, n, moments, error)
        : conversion_mode == 1
        ? candidate->enable_internal_triple_cold_ice_graupel_for_test(
            global_simulation_state(), nc, n, moments, error)
        : candidate->enable_internal_triple_cold_riming_for_test(
            global_simulation_state(), nc, n, moments, error);
    INFO(error);
    REQUIRE(selected);
    REQUIRE(candidate->owns_primary_endpoint_commit());
    REQUIRE(candidate->protected_primary_endpoint_wind_source() == nullptr);
    microphysics_scheme = std::move(candidate);
}

ColdRimingCell source(const Snapshot& before, int i)
{
    ColdRimingCell result;
    result.pressure_pa = before.fields[1](i, 0, 0);
    const double exner = std::pow(result.pressure_pa / microphysics_constants::p0,
        microphysics_constants::R_d / microphysics_constants::cp);
    result.temperature_k = static_cast<double>(before.fields[5](i, 0, 0)) * exner;
    result.dry_density_kg_m3 = before.fields[0](i, 0, 0);
    result.initial.vapor_mass_per_kg = before.fields[6](i, 0, 0);
    result.initial.cloud_mass_per_kg = before.fields[7](i, 0, 0);
    result.initial.cloud_number_per_kg = before.moments[0](i, 0, 0);
    for (std::size_t species = 0; species < 4; ++species)
    {
        result.initial.frozen[species] = {before.fields[species + 9](i, 0, 0),
            before.moments[species + 2](i, 0, 0), before.moments[species + 7](i, 0, 0)};
    }
    result.initial.rain = {before.fields[8](i, 0, 0), before.moments[1](i, 0, 0), before.moments[6](i, 0, 0)};
    return result;
}

ColdRimingResult capture(const Snapshot& before, int i, double duration, int conversion_mode = 0)
{
    ColdRimingResult result;
    std::string error;
    const auto parameters = frozen_collection_live::parameters(
        global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes);
    bool accepted = false;
    REQUIRE(conversion_mode >= 0);
    REQUIRE(conversion_mode <= 8);
    if (conversion_mode >= 6)
    {
        ColdCollectedSllResult converted;
        accepted = conversion_mode == 8
            ? evaluate_exact_triple_cold_wet_growth_sll_cell(parameters, source(before, i), duration, converted, error)
            : conversion_mode == 7
            ? evaluate_exact_triple_cold_collision_sll_cell(parameters, source(before, i), duration, converted, error)
            : evaluate_exact_triple_cold_rain_sll_cell(parameters, source(before, i), duration, converted, error);
        result = converted.combined;
    }
    else if (conversion_mode == 5)
    {
        ColdCollectedSllResult converted;
        accepted = evaluate_exact_triple_cold_collected_sll_cell(parameters, source(before, i),
            duration, converted, error);
        result = converted.combined;
    }
    else if (conversion_mode == 4)
    {
        ColdSllResult converted;
        accepted = evaluate_exact_triple_cold_sll_cell(parameters, source(before, i),
            duration, converted, error);
        result = converted.combined;
    }
    else if (conversion_mode == 3 || conversion_mode == 2)
    {
        ColdGraupelConversionsResult converted;
        accepted = evaluate_exact_triple_cold_graupel_conversions_cell(parameters,
            source(before, i), duration, converted, error, 2, nullptr, conversion_mode == 3);
        result = converted.combined;
    }
    else if (conversion_mode == 1)
    {
        ColdIceGraupelResult converted;
        accepted = evaluate_exact_triple_cold_ice_graupel_cell(parameters,
            source(before, i), duration, converted, error);
        result = converted.combined;
    }
    else
    {
        accepted = evaluate_exact_triple_cold_riming_cell(parameters,
            source(before, i), duration, result, error);
    }
    INFO(error);
    REQUIRE(accepted);
    return result;
}

void check_step(const Snapshot& before, double duration, int conversion_mode = 0)
{
    const auto& scheme = dynamic_cast<const ObservedMilbrandt&>(*microphysics_scheme);
    REQUIRE(scheme.getter_called);
    REQUIRE(scheme.accepted_called);
    const auto& candidate = scheme.endpoint;
    REQUIRE(candidate.owner == &scheme);
    REQUIRE(candidate.duration_s == duration);
    REQUIRE(candidate.generation != 0);
    const std::array<const Field3D*, 10> bindings{&rho, &p, &theta, &qv, &qc, &qr, &qi, &qs, &qg, &qh};
    CHECK(candidate.source_fields == bindings);
    const auto physical = physical_scalar_domain(global_simulation_state().grid);
    const std::array<const Field3D*, 7> water{&qv, &qc, &qr, &qi, &qs, &qg, &qh};
    std::array<long double, 7> rounding{};
    std::array<long double, 7> requested{};
    std::array<long double, 6> moist{};
    long double latent_heat = 0;
    long double theta_rounding = 0;
    for (int i = physical.i_begin; i < physical.i_end; ++i)
    {
        const auto cell = source(before, i);
        const auto rain_reference = conversion_mode == 8
            ? static_cast<milbrandt_cold_rain_sll_reference::Reference>(milbrandt_hail_wet_growth_reference::evaluate(cell, duration,
                global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes))
            : conversion_mode == 7 ? milbrandt_rain_collision_reference::evaluate_coupled(cell, duration,
            global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes)
            : conversion_mode == 6 ? milbrandt_cold_rain_sll_reference::evaluate(cell, duration,
            global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes)
            : milbrandt_cold_rain_sll_reference::Reference{};
        const reference::Reference expected = conversion_mode >= 6 ? static_cast<reference::Reference>(rain_reference)
            : conversion_mode == 5
            ? static_cast<reference::Reference>(milbrandt_cold_collected_sll_reference::evaluate(cell, duration,
                global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes))
            : conversion_mode == 4
            ? static_cast<reference::Reference>(milbrandt_cold_sll_reference::evaluate(cell, duration,
                global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes))
            : conversion_mode == 3
            ? static_cast<reference::Reference>(milbrandt_cold_homogeneous_reference::evaluate(cell, duration))
            : conversion_mode == 2
            ? static_cast<reference::Reference>(milbrandt_cold_graupel_conversions_reference::evaluate(cell, duration))
            : conversion_mode == 1
            ? static_cast<reference::Reference>(milbrandt_cold_ice_graupel_reference::evaluate(cell, duration))
            : reference::evaluate(cell, duration);
        const auto actual = capture(before, i, duration, conversion_mode);
        const long double dry_mass = static_cast<long double>(cell.dry_density_kg_m3) * dr * dtheta * dz;
        const double exner = std::pow(cell.pressure_pa / microphysics_constants::p0,
            microphysics_constants::R_d / microphysics_constants::cp);
        const long double heat_factor = dry_mass * microphysics_constants::cp * exner;
        const auto near = [&](double value, long double oracle, long double scale)
        {
            CHECK(std::abs(static_cast<long double>(value) - oracle)
                  <= 3e-8L * std::max(std::abs(oracle), scale) + 1e-35L);
        };
        near(actual.temperature_increment_k, expected.temperature_increment, 1e-12L);
        near(actual.cloud_mass_increment_per_kg, expected.cloud_increment, 1e-22L);
        near(actual.cloud_number_increment_per_kg, expected.cloud_number_increment, 1e-20L);
        near(actual.vapor_mass_increment_per_kg, expected.vapor_increment, 1e-22L);
        CHECK(candidate.water_after[0][i] == actual.unrounded.vapor_mass_per_kg);
        CHECK(candidate.water_after[1][i] == actual.unrounded.cloud_mass_per_kg);
        if (conversion_mode >= 6)
        {
            near(actual.rain_mass_increment_per_kg, rain_reference.rain_mass, 1e-22L);
            near(actual.rain_number_increment_per_kg, rain_reference.rain_number, 1e-20L);
            near(actual.rain_reflectivity_increment_per_kg, rain_reference.rain_z, 1e-40L);
            CHECK(candidate.water_after[2][i] == actual.unrounded.rain.mass_per_kg);
            CHECK(qr(i, 0, 0) == actual.rounded_rain.mass_per_kg);
            CHECK(scheme.internal_triple_number_for_test(0)(i, 0, 0) == actual.rounded_rain.number_per_kg);
            CHECK(scheme.internal_triple_reflectivity_for_test(0)(i, 0, 0) == actual.rounded_rain.reflectivity_per_kg);
        }
        else CHECK(candidate.water_after[2][i] == before.fields[8](i, 0, 0));
        CHECK(qv(i, 0, 0) == actual.rounded_vapor_mass_per_kg);
        CHECK(qc(i, 0, 0) == actual.rounded_cloud_mass_per_kg);
        CHECK(scheme.internal_triple_cloud_number_for_test()(i, 0, 0) == actual.rounded_cloud_number_per_kg);
        for (std::size_t species = 0; species < 4; ++species)
        {
            near(actual.mass_increment_per_kg[species], expected.mass_increment[species], 1e-22L);
            near(actual.number_increment_per_kg[species], expected.number_increment[species], 1e-20L);
            near(actual.reflectivity_increment_per_kg[species], expected.z_increment[species], 1e-40L);
            const auto& channel = actual.producer.channels()[species];
            near(channel.collection_mass_per_kg, expected.collected[species], 1e-22L);
            near(channel.diffusion_mass_per_kg, expected.diffusion[species], 1e-22L);
            near(channel.splinter_number_per_kg, expected.splinter_number[species], 1e-20L);
            CHECK(candidate.water_after[species + 3][i] == actual.unrounded.frozen[species].mass_per_kg);
            CHECK(scheme.internal_triple_number_for_test(species + 1)(i, 0, 0) == actual.rounded[species].number_per_kg);
            CHECK(scheme.internal_triple_reflectivity_for_test(species + 1)(i, 0, 0) == actual.rounded[species].reflectivity_per_kg);
        }
        CHECK(actual.producer.channels()[3].splinter_number_per_kg == 0);
        CHECK(candidate.theta_increment[i] * exner == Approx(actual.temperature_increment_k).epsilon(8e-12).margin(1e-14));
        const double theta_exact = static_cast<double>(before.fields[5](i, 0, 0)) + candidate.theta_increment[i];
        CHECK(theta(i, 0, 0) == static_cast<float>(theta_exact));
        CHECK(theta_exact * exner >= 150);
        CHECK(theta_exact * exner < microphysics_constants::T0);
        CHECK(static_cast<double>(theta(i, 0, 0)) * exner < microphysics_constants::T0);
        latent_heat += dry_mass * actual.temperature_increment_k * microphysics_constants::cp;
        theta_rounding += (static_cast<long double>(theta(i, 0, 0)) - theta_exact) * heat_factor;
        long double total_before = 0;
        long double total_unrounded = 0;
        long double total_after = 0;
        for (std::size_t species = 0; species < water.size(); ++species)
        {
            const double old = before.fields[species + 6](i, 0, 0);
            const double endpoint = candidate.water_after[species][i];
            const float stored = (*water[species])(i, 0, 0);
            CHECK(stored == static_cast<float>(endpoint));
            rounding[species] += (static_cast<long double>(stored) - endpoint) * dry_mass;
            requested[species] += (static_cast<long double>(endpoint) - old) * dry_mass;
            total_before += old;
            total_unrounded += endpoint;
            total_after += stored;
        }
        CHECK(static_cast<double>(total_unrounded) == Approx(static_cast<double>(total_before)).epsilon(8e-15));
        const long double old_t = static_cast<long double>(before.fields[5](i, 0, 0)) * exner;
        const long double new_t = static_cast<long double>(theta(i, 0, 0)) * exner;
        long double old_ice = 0;
        long double new_ice = 0;
        for (std::size_t species = 0; species < 4; ++species)
        {
            old_ice += before.fields[species + 9](i, 0, 0);
            new_ice += (*water[species + 3])(i, 0, 0);
        }
        const long double old_liquid = static_cast<long double>(before.fields[7](i, 0, 0))
            + before.fields[8](i, 0, 0);
        const long double new_liquid = static_cast<long double>(qc(i, 0, 0)) + qr(i, 0, 0);
        const long double cv_v = physical_constants::specific_heat_water_vapor_cp_jkgk - microphysics_constants::R_v;
        moist[0] += dry_mass * (microphysics_constants::cp - microphysics_constants::R_d) * (new_t - old_t);
        moist[1] += dry_mass * ((new_t - microphysics_constants::T0)
            * (qv(i, 0, 0) * cv_v + new_liquid * 4186 + new_ice * 2106)
            - (old_t - microphysics_constants::T0)
            * (before.fields[6](i, 0, 0) * cv_v + old_liquid * 4186 + old_ice * 2106));
        moist[2] += dry_mass * ((static_cast<long double>(qv(i, 0, 0)) - before.fields[6](i, 0, 0))
            * (microphysics_constants::L_v - microphysics_constants::R_v * microphysics_constants::T0)
            - (new_ice - old_ice) * microphysics_constants::L_f);
        const long double kinetic = .5L * (std::pow(static_cast<long double>(u(i, 0, 0)), 2)
            + std::pow(static_cast<long double>(v(i, 0, 0)), 2) + std::pow(static_cast<long double>(w(i, 0, 0)), 2));
        moist[3] += dry_mass * (total_after - total_before) * kinetic;
        const long double height = global_grid_geometry.z.empty() ? .5L * dz : global_grid_geometry.z[0];
        moist[4] += dry_mass * (total_after - total_before) * physical_constants::gravity_ms2 * height;
    }
    const Snapshot after;
    for (const std::size_t field : {0, 1, 2, 3, 4, 8, 13, 14})
    {
        if (conversion_mode < 6 || field != 8) CHECK(same_bytes(after.fields[field], before.fields[field]));
    }
    for (const std::size_t moment : {1, 6})
    {
        if (conversion_mode < 6) CHECK(same_bytes(after.moments[moment], before.moments[moment]));
    }
    for (const int i : {0, NR - 1})
    {
        for (std::size_t field = 0; field < before.fields.size(); ++field)
        {
            CHECK(std::memcmp(&after.fields[field](i, 0, 0), &before.fields[field](i, 0, 0), sizeof(float)) == 0);
        }
        for (std::size_t moment = 0; moment < before.moments.size(); ++moment)
        {
            CHECK(std::memcmp(&after.moments[moment](i, 0, 0), &before.moments[moment](i, 0, 0), sizeof(float)) == 0);
        }
    }
    for (std::size_t species = 0; species < water.size(); ++species)
    {
        CHECK(after.water.species[species].storage_rounding_kg == Approx(static_cast<double>(rounding[species])).epsilon(3e-6).margin(1e-15));
        CHECK(after.heat.microphysics_water_tendency_kg_by_species[species] == Approx(static_cast<double>(requested[species])).epsilon(3e-6).margin(1e-12));
    }
    CHECK(after.heat.microphysics_heat_j == Approx(static_cast<double>(latent_heat)).epsilon(3e-6).margin(1e-10));
    CHECK(after.heat.theta_storage_rounding_j == Approx(static_cast<double>(theta_rounding)).epsilon(3e-6).margin(1e-10));
    const std::array<double, 6> measured{after.heat.moist_dry_internal_change_j,
        after.heat.moist_water_sensible_change_j, after.heat.moist_phase_reference_change_j,
        after.heat.moist_carrier_kinetic_change_j, after.heat.moist_gravitational_change_j,
        after.heat.moist_subgrid_tke_change_j};
    for (std::size_t reservoir = 0; reservoir < moist.size(); ++reservoir)
    {
        CAPTURE(reservoir);
        CHECK(measured[reservoir] == Approx(static_cast<double>(moist[reservoir])).epsilon(3e-6).margin(1e-10));
    }
    CHECK(after.heat.moist_energy_complete);
    CHECK(after.heat.moist_energy_closure_residual_j == 0);
    CHECK(after.water.lower_bound_count() == 0);
    CHECK(after.water.lower_bound_roundoff_count() == 0);
    CHECK(after.water.upper_bound_count() == 0);
    CHECK(after.water.nonfinite_count() == 0);
    CHECK(after.heat.theta_tendency_limit_count == 0);
    CHECK(after.heat.theta_nonfinite_count == 0);
    check_no_fallout(before, duration);
}

} // namespace cold_riming_live

TEST_CASE("live four channel cold riming creates and self multiplies ice with independent full reservoir closure",
          "[physics][milbrandt][triple_cold_riming_internal][direct]")
{
    for (int scenario = 0; scenario <= 5; ++scenario)
    {
        cold_riming_live::initialize(scenario);
        const Snapshot before;
        const auto result = cold_riming_live::capture(before, 1, 1e-4);
        CHECK(result.producer.event_count() == 4);
        const auto& channels = result.producer.channels();
        CHECK(channels[3].splinter_number_per_kg == 0);
        if (scenario == 1 || scenario == 3 || scenario == 4)
        {
            REQUIRE(before.fields[9](1, 0, 0) == 0);
            CHECK(result.unrounded.frozen[0].mass_per_kg > 0);
            CHECK(result.unrounded.frozen[0].number_per_kg > 0);
            CHECK(result.unrounded.frozen[0].reflectivity_per_kg > 0);
        }
        if (scenario == 2)
        {
            REQUIRE(channels[0].splinter_number_per_kg > 0);
            CHECK(result.number_increment_per_kg[0] > channels[0].diffusion_number_per_kg);
            CHECK(result.mass_increment_per_kg[0]
                  == channels[0].collection_mass_per_kg + channels[0].diffusion_mass_per_kg);
        }
        if (scenario == 5)
        {
            CHECK(result.unrounded.frozen[0].mass_per_kg == 0);
            CHECK(result.unrounded.frozen[0].number_per_kg == 0);
            CHECK(result.unrounded.frozen[0].reflectivity_per_kg == 0);
        }
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4);
    }
    for (double temperature : {263.15, 266.15, 270.15, 271.15})
    {
        cold_riming_live::initialize(0, 2, temperature);
        const Snapshot before;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4);
    }
    cold_riming_live::initialize(0, 2, 268.15, 1, false);
    const Snapshot before;
    CHECK_FALSE(cold_riming_live::capture(before, 1, 1e-4).producer.hail_processes());
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(before, 1e-4);
}

TEST_CASE("live cold riming continuations are exact with actual one and four workers",
          "[physics][milbrandt][triple_cold_riming_internal][direct]")
{
#ifdef _OPENMP
    struct ThreadRestore
    {
        int workers = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~ThreadRestore()
        {
            omp_set_num_threads(workers);
            omp_set_dynamic(dynamic);
        }
    } restore;
    omp_set_dynamic(0);
    for (int scenario : {0, 1})
    {
        std::array<std::unique_ptr<Snapshot>, 2> final;
        for (int run = 0; run < 2; ++run)
        {
            const int workers = run == 0 ? 1 : 4;
            omp_set_num_threads(workers);
            int observed = 0;
#pragma omp parallel
            {
#pragma omp single
                observed = omp_get_num_threads();
            }
            REQUIRE(observed == workers);
            cold_riming_live::initialize(scenario);
            for (double duration : {1e-4, 2e-4, 5e-4})
            {
                const Snapshot before;
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                cold_riming_live::check_step(before, duration);
                simulation_time += duration;
            }
            final[run] = std::make_unique<Snapshot>();
        }
        check_snapshot(*final[0], *final[1]);
    }
#else
    FAIL("cold riming worker evidence requires OpenMP");
#endif
}

TEST_CASE("live cold riming rejects altered primary records and late ledger or moment owners atomically",
          "[physics][milbrandt][triple_cold_riming_internal][direct]")
{
    for (int mutation = 1; mutation <= 25; ++mutation)
    {
        cold_riming_live::initialize();
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.getter_mutation = mutation;
        const Snapshot before;
        INFO("cold riming primary mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4);
    }
    for (int mutation : {0, 1, 2, 6, 7, 8})
    {
        cold_riming_live::initialize();
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        INFO("cold riming late mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4);
    }
}

TEST_CASE("live cold riming zero sources still validate private moments native policy and held owners",
          "[physics][milbrandt][triple_cold_riming_internal][direct]")
{
    for (int mutation = 0; mutation < 38; ++mutation)
    {
        cold_riming_live::initialize(6, 2, 268.15, 1, true, 0);
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        if (mutation >= 19 && mutation < 31)
        {
            auto* held = held_fields()[mutation - 19];
            held->resize(NR, NTH, NZ, 0);
            (*held)(last, 0, 0) = 1;
        }
        else if (mutation >= 31)
        {
            *held_radiation_powers()[mutation - 31] = 1;
        }
        else
        {
            switch (mutation)
            {
                case 0: qv(last, 0, 0) = -1; break;
                case 1: qv(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 2: qr(last, 0, 0) = 1e-4f; break;
                case 3: qi(last, 0, 0) = 1e-4f; break;
                case 4: rho(last, 0, 0) = 0; break;
                case 5: p(last, 0, 0) = 0; break;
                case 6: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = -1; break;
                case 7: const_cast<Field3D&>(scheme.internal_triple_number_for_test(0))(last, 0, 0) = 1; break;
                case 8: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(1))(last, 0, 0) = 1; break;
                case 9: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                case 10: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
                case 11: global_simulation_storage.runtime.compute.fallback_active = true; break;
                case 12: global_simulation_storage.configuration.microphysics_config.milbrandt.triple_moment = true; break;
                case 13: global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes = false; break;
                case 14: global_validation_policy.mode = tmv::GuardMode::Warn; break;
                case 15: scheme.set_require_native_dispatch(true); break;
                case 16: theta(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 17: theta(last, 0, 0) = 0; break;
                case 18: theta(last, 0, 0) = 701; break;
            }
        }
        const Snapshot bad;
        INFO("cold riming zero mutation " << mutation);
        if (mutation == 15)
        {
            std::array<Field3D, 8> unpublished;
            for (auto& field : unpublished)
            {
                field.resize(NR, NTH, NZ, 42);
            }
            REQUIRE_THROWS_WITH(internal_thermal_live::compute(unpublished, 1e-4),
                Catch::Matchers::Contains("live CPU direct-step owner"));
            for (const auto& field : unpublished)
            {
                for (std::size_t index = 0; index < field.size(); ++index)
                {
                    CHECK(field.data()[index] == 42);
                }
            }
        }
        else
        {
            REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        }
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(bad);
    }
}

TEST_CASE("live cold riming source interval and detached or stale records reject and recover",
          "[physics][milbrandt][triple_cold_riming_internal][direct]")
{
    cold_riming_live::initialize(1);
    const Snapshot before;
    const auto result = cold_riming_live::capture(before, 1, 1e-4);
    REQUIRE(result.producer.relaxation_number() > 0);
    const double unsupported = 1.1e-4 / result.producer.relaxation_number();
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), unsupported), Catch::Matchers::Contains("relaxation"));
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(before);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(before, 1e-4);
    cold_riming_live::initialize();
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    std::array<Field3D, 8> out;
    std::string error;
    tmv::MicrophysicsPrimaryEndpointCandidate first;
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    internal_thermal_live::compute(out, 1e-4);
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    auto& ni = const_cast<Field3D&>(scheme.internal_triple_number_for_test(1));
    const float old_number = ni(1, 0, 0);
    ni(1, 0, 0) = std::nextafter(old_number, std::numeric_limits<float>::infinity());
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    ni(1, 0, 0) = old_number;
    const float old_cloud = qc(1, 0, 0);
    qc(1, 0, 0) = std::nextafter(old_cloud, std::numeric_limits<float>::infinity());
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    qc(1, 0, 0) = old_cloud;
    internal_thermal_live::compute(out, 1e-4);
    tmv::MicrophysicsPrimaryEndpointCandidate second;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, second, error));
    CHECK(second.generation > first.generation);
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    const Field3D detached(qi);
    for (auto& field : out)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    REQUIRE_THROWS(scheme.compute_tendencies(p, theta, rho, qv, qc, qr, detached, qs, qg, qh, 1e-4,
        out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]));
    for (const auto& field : out)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    CHECK_THROWS(scheme.portable_state());
    CHECK_THROWS(scheme.clone_parameters());
    CHECK_THROWS(scheme.prognostic_scalars_for_transport());
    const Snapshot recovered;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(recovered, 1e-4);
    CHECK_FALSE(dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme)
        .validate_primary_endpoint_commit(global_simulation_state(), 1e-4, second, error));
}

TEST_CASE("live cold riming enforces actual raw and stored thermal phase boundaries atomically",
          "[physics][milbrandt][triple_cold_riming_internal][direct]")
{
    cold_riming_live::initialize();
    const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
    bool selected = false;
    double selected_exner = 0;
    float lower_theta = 0;
    float upper_theta = 0;
    for (float pressure : {65000.0f, 70000.0f, 75000.0f, 80000.0f, 85000.0f, 90000.0f, 95000.0f})
    {
        const double exner = std::pow(static_cast<double>(pressure) / microphysics_constants::p0,
            microphysics_constants::R_d / microphysics_constants::cp);
        const double threshold = microphysics_constants::T0 / exner;
        const float nearest = static_cast<float>(threshold);
        const float lower = static_cast<double>(nearest) < threshold
            ? nearest : std::nextafter(nearest, -std::numeric_limits<float>::infinity());
        const float upper = std::nextafter(lower, std::numeric_limits<float>::infinity());
        if (threshold > (static_cast<double>(lower) + upper) * .5)
        {
            p(last, 0, 0) = pressure;
            theta(last, 0, 0) = lower;
            selected_exner = exner;
            lower_theta = lower;
            upper_theta = upper;
            qv(last, 0, 0) = static_cast<float>(cold_riming_live::reference::thermal::saturation(
                static_cast<double>(lower) * exner, pressure).mass);
            selected = true;
            break;
        }
    }
    REQUIRE(selected);
    const Snapshot before;
    const double source_temperature = static_cast<double>(lower_theta) * selected_exner;
    const double midpoint = (static_cast<double>(lower_theta) + upper_theta) * .5 * selected_exner;
    const double target = (midpoint + microphysics_constants::T0) * .5;
    const auto probe = cold_riming_live::capture(before, last, 1e-6);
    REQUIRE(probe.temperature_increment_k > 0);
    const double duration = (target - source_temperature) / (probe.temperature_increment_k / 1e-6);
    const auto pure = cold_riming_live::capture(before, last, duration);
    REQUIRE(source_temperature + pure.temperature_increment_k < microphysics_constants::T0);
    REQUIRE(static_cast<double>(static_cast<float>(static_cast<double>(lower_theta)
        + pure.temperature_increment_k / selected_exner)) * selected_exner >= microphysics_constants::T0);
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), duration), Catch::Matchers::Contains("final heat"));
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(before);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration * .1));
    cold_riming_live::check_step(before, duration * .1);
}

TEST_CASE("live ice to graupel conversion closes all moments and reservoirs without duplicate heat",
          "[physics][milbrandt][triple_cold_ice_graupel_internal][direct]")
{
    bool active = false;
    bool created = false;
    for (int scenario : {0, 2})
    {
        for (long double alpha : {0.0L, 2.0L, 20.0L})
        {
            for (bool hail : {false, true})
            {
                cold_riming_live::initialize(scenario, alpha, 268.15, 1, hail, 1e-3f, 1);
                const Snapshot before;
                const auto cell = cold_riming_live::source(before, 1);
                const auto parameters = frozen_collection_live::parameters(hail);
                ColdIceGraupelResult converted;
                ColdRimingResult prefix;
                std::string error;
                REQUIRE(evaluate_exact_triple_cold_ice_graupel_cell(parameters, cell, 1e-4, converted, error));
                REQUIRE(evaluate_exact_triple_cold_riming_cell(parameters, cell, 1e-4, prefix, error));
                const auto expected = milbrandt_cold_ice_graupel_reference::evaluate(cell, 1e-4);
                CHECK(converted.conversion.raw_mass_transfer_per_kg()
                    == Approx(static_cast<double>(expected.conversion_mass)).epsilon(3e-8).margin(1e-22));
                CHECK(converted.combined.temperature_increment_k == prefix.temperature_increment_k);
                CHECK(converted.combined.cloud_mass_increment_per_kg == prefix.cloud_mass_increment_per_kg);
                CHECK(converted.combined.cloud_number_increment_per_kg == prefix.cloud_number_increment_per_kg);
                CHECK(converted.combined.vapor_mass_increment_per_kg == prefix.vapor_mass_increment_per_kg);
                active |= converted.conversion.raw_mass_transfer_per_kg() > 0;
                if (scenario == 2)
                {
                    REQUIRE(before.fields[11](1, 0, 0) == 0);
                    REQUIRE(before.moments[4](1, 0, 0) == 0);
                    REQUIRE(before.moments[9](1, 0, 0) == 0);
                    REQUIRE(converted.conversion.raw_mass_transfer_per_kg() > 0);
                    REQUIRE(converted.combined.unrounded.frozen[2].mass_per_kg > 0);
                    REQUIRE(converted.combined.unrounded.frozen[2].number_per_kg > 0);
                    REQUIRE(converted.combined.unrounded.frozen[2].reflectivity_per_kg > 0);
                    created = true;
                }
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
                cold_riming_live::check_step(before, 1e-4, 1);
                const Snapshot continued;
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 2e-4));
                cold_riming_live::check_step(continued, 2e-4, 1);
            }
        }
    }
    CHECK(active);
    CHECK(created);
}

namespace cold_conversion_live
{

bool enable(tmv::MilbrandtScheme& candidate, const SimulationStateView& state,
            const Snapshot& source, int conversion_mode, std::string& error)
{
    const std::array<const Field3D*, 4> number{&source.moments[2], &source.moments[3],
        &source.moments[4], &source.moments[5]};
    const std::array<const Field3D*, 4> z{&source.moments[7], &source.moments[8],
        &source.moments[9], &source.moments[10]};
    return conversion_mode == 5
        ? candidate.enable_internal_triple_cold_collected_sll_for_test(
            state, source.moments[0], number, z, error)
        : conversion_mode == 4
        ? candidate.enable_internal_triple_cold_sll_for_test(
            state, source.moments[0], number, z, error)
        : conversion_mode == 3
        ? candidate.enable_internal_triple_cold_homogeneous_for_test(
            state, source.moments[0], number, z, error)
        : conversion_mode == 2
        ? candidate.enable_internal_triple_cold_graupel_conversions_for_test(
            state, source.moments[0], number, z, error)
        : candidate.enable_internal_triple_cold_ice_graupel_for_test(
            state, source.moments[0], number, z, error);
}

void check_workers(int conversion_mode)
{
#ifdef _OPENMP
    struct ThreadRestore
    {
        int workers = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~ThreadRestore()
        {
            omp_set_num_threads(workers);
            omp_set_dynamic(dynamic);
        }
    } restore;
    omp_set_dynamic(0);
    for (int scenario : {0, conversion_mode == 2 ? 3 : 2})
    {
        std::array<std::unique_ptr<Snapshot>, 2> final;
        for (int run = 0; run < 2; ++run)
        {
            const int workers = run == 0 ? 1 : 4;
            omp_set_num_threads(workers);
            int observed = 0;
#pragma omp parallel
            {
#pragma omp single
                observed = omp_get_num_threads();
            }
            REQUIRE(observed == workers);
            cold_riming_live::initialize(scenario, scenario == 3 ? 199.9L : 2.0L,
                268.15, 1, true, 1e-3f, conversion_mode);
            for (double duration : {1e-4, 2e-4, 5e-4})
            {
                const Snapshot before;
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                cold_riming_live::check_step(before, duration, conversion_mode);
                simulation_time += duration;
            }
            final[run] = std::make_unique<Snapshot>();
        }
        check_snapshot(*final[0], *final[1]);
    }
#else
    FAIL("cold conversion worker evidence requires OpenMP");
#endif
}

void check_rollback(int conversion_mode)
{
    for (int mutation = 1; mutation <= 25; ++mutation)
    {
        cold_riming_live::initialize(0, 2, 268.15, 1, true, 1e-3f, conversion_mode);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        INFO("cold conversion mode " << conversion_mode << ", early mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, conversion_mode);
    }
    for (int mutation : {0, 1, 2, 6, 7, 8})
    {
        cold_riming_live::initialize(0, 2, 268.15, 1, true, 1e-3f, conversion_mode);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        INFO("cold conversion mode " << conversion_mode << ", late mutation " << mutation);
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, conversion_mode);
    }
}

void check_zero_guards(int conversion_mode)
{
    for (int mutation = 0; mutation < 55; ++mutation)
    {
        cold_riming_live::initialize(6, 2, 268.15, 1, true, 0, conversion_mode);
        const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        if (mutation >= 31 && mutation < 43)
        {
            auto* held = held_fields()[mutation - 31];
            held->resize(NR, NTH, NZ, 0);
            (*held)(last, 0, 0) = 1;
        }
        else if (mutation >= 43 && mutation < 50)
        {
            *held_radiation_powers()[mutation - 43] = 1;
        }
        else
        {
            switch (mutation)
            {
                case 0: qv(last, 0, 0) = -1; break;
                case 1: qv(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 2: qr(last, 0, 0) = 1e-4f; break;
                case 3: qi(last, 0, 0) = 1e-4f; break;
                case 4: rho(last, 0, 0) = 0; break;
                case 5: p(last, 0, 0) = 0; break;
                case 6: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = -1; break;
                case 7: const_cast<Field3D&>(scheme.internal_triple_number_for_test(0))(last, 0, 0) = 1; break;
                case 8: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(1))(last, 0, 0) = 1; break;
                case 9: global_simulation_storage.runtime.compute.active_kind = ComputeBackendKind::Metal; break;
                case 10: global_simulation_storage.runtime.compute.requested_kind = ComputeBackendKind::Vulkan; break;
                case 11: global_simulation_storage.runtime.compute.fallback_active = true; break;
                case 12: global_simulation_storage.configuration.microphysics_config.milbrandt.triple_moment = true; break;
                case 13: global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes = false; break;
                case 14: global_validation_policy.mode = tmv::GuardMode::Warn; break;
                case 15: scheme.set_require_native_dispatch(true); break;
                case 16: theta(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 17: theta(last, 0, 0) = 149; p(last, 0, 0) = 100000; break;
                case 18: theta(last, 0, 0) = -1; p(last, 0, 0) = 200000; break;
                case 19: theta(last, 0, 0) = 701; p(last, 0, 0) = 500; break;
                case 20: theta(last, 0, 0) = 149; p(last, 0, 0) = 200000; break;
                case 21:
                    theta(last, 0, 0) = static_cast<float>(280 / std::pow(static_cast<double>(p(last, 0, 0))
                        / microphysics_constants::p0, microphysics_constants::R_d / microphysics_constants::cp));
                    break;
                case 22: global_validation_policy.fail_on = tmv::GuardFailOn::NonFinite; break;
                case 23:
                    qc(last, 0, 0) = 1e-4f;
                    const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = 0;
                    break;
                case 24: qv.resize(1, 1, 1, 0); break;
                case 25: qg(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 26: qh(last, 0, 0) = -1; break;
                case 27: global_simulation_storage.runtime.chaos.noise_elapsed_s = 1; break;
                case 28: global_simulation_storage.runtime.terrain.earth_fixed_placement.active = true; break;
                case 29: global_simulation_storage.configuration.domain_frame.velocity_ms.u_ms = 1; break;
                case 30: dtheta_dt_rad(last, 0, 0) = 1; break;
                case 50: qg(last, 0, 0) = 1e-4f; break;
                case 51: const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test())(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 52: rho(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
                case 53: const_cast<Field3D&>(scheme.internal_triple_number_for_test(3))(last, 0, 0) = 1; break;
                case 54: const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(4))(last, 0, 0) = std::numeric_limits<float>::quiet_NaN(); break;
            }
        }
        const Snapshot bad;
        INFO("cold conversion mode " << conversion_mode << ", zero mutation " << mutation);
        if (mutation == 15)
        {
            std::array<Field3D, 8> unpublished;
            for (auto& field : unpublished)
            {
                field.resize(NR, NTH, NZ, 42);
            }
            REQUIRE_THROWS_WITH(internal_thermal_live::compute(unpublished, 1e-4),
                Catch::Matchers::Contains("live CPU direct-step owner"));
            for (const auto& field : unpublished)
            {
                for (std::size_t index = 0; index < field.size(); ++index)
                {
                    CHECK(field.data()[index] == 42);
                }
            }
        }
        else
        {
            REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        }
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(bad);
    }
}

void check_bindings(int conversion_mode)
{
    cold_riming_live::initialize(0, 2, 268.15, 1, true, 1e-3f, conversion_mode);
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    std::array<Field3D, 8> out;
    std::string error;
    tmv::MicrophysicsPrimaryEndpointCandidate first;
    CHECK_FALSE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    internal_thermal_live::compute(out, 1e-4);
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    const std::array<Field3D*, 5> private_sources{
        &const_cast<Field3D&>(scheme.internal_triple_cloud_number_for_test()),
        &const_cast<Field3D&>(scheme.internal_triple_number_for_test(1)),
        &const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(2)),
        &const_cast<Field3D&>(scheme.internal_triple_number_for_test(3)),
        &const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(4))};
    for (auto* field : private_sources)
    {
        const float old = (*field)(1, 0, 0);
        (*field)(1, 0, 0) = std::nextafter(old, std::numeric_limits<float>::infinity());
        CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
        (*field)(1, 0, 0) = old;
    }
    for (Field3D* field : {&qv, &qc, &qi, &qs, &qg, &qh})
    {
        const float old = (*field)(1, 0, 0);
        (*field)(1, 0, 0) = std::nextafter(old, std::numeric_limits<float>::infinity());
        CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
        (*field)(1, 0, 0) = old;
    }
    internal_thermal_live::compute(out, 1e-4);
    tmv::MicrophysicsPrimaryEndpointCandidate second;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, second, error));
    CHECK(second.generation > first.generation);
    CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-4, first, error));
    const Field3D detached(qi);
    for (auto& field : out)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    const Snapshot before_detached;
    REQUIRE_THROWS(scheme.compute_tendencies(p, theta, rho, qv, qc, qr, detached, qs, qg, qh, 1e-4,
        out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]));
    for (const auto& field : out)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    check_snapshot(before_detached);
    CHECK_THROWS(scheme.portable_state());
    CHECK_THROWS(scheme.clone_parameters());
    CHECK_THROWS(scheme.prognostic_scalars_for_transport());
    const auto identity = scheme.parameter_identity();
    tmv::MicrophysicsConfig changed;
    changed.milbrandt.hail_processes = false;
    CHECK_THROWS(scheme.configure(changed));
    CHECK(scheme.parameter_identity() == identity);
    const Snapshot recovered;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(recovered, 1e-4, conversion_mode);
    CHECK_FALSE(dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme)
        .validate_primary_endpoint_commit(global_simulation_state(), 1e-4, second, error));
}

void check_activation(int conversion_mode)
{
    const std::array<std::array<float, 2>, 6> invalid_values{{
        {149, 100000}, {-1, 200000}, {701, 500}, {149, 200000},
        {std::numeric_limits<float>::quiet_NaN(), 100000},
        {std::numeric_limits<float>::infinity(), 100000}}};
    for (const auto invalid : invalid_values)
    {
        cold_riming_live::initialize(6, 2, 268.15, 1, true, 0, conversion_mode);
        const Snapshot original;
        theta(1, 0, 0) = invalid[0];
        p(1, 0, 0) = invalid[1];
        const Snapshot bad;
        tmv::MilbrandtScheme candidate;
        const auto identity = candidate.parameter_identity();
        std::string error;
        CHECK_FALSE(enable(candidate, global_simulation_state(), bad, conversion_mode, error));
        CHECK_FALSE(error.empty());
        CHECK_FALSE(candidate.owns_primary_endpoint_commit());
        CHECK_FALSE(candidate.owns_post_primary_private_commit());
        CHECK(candidate.parameter_identity() == identity);
        check_snapshot(bad);
        theta = original.fields[5];
        p = original.fields[1];
        REQUIRE(enable(candidate, global_simulation_state(), original, conversion_mode, error));
        CHECK(candidate.owns_primary_endpoint_commit());
        check_snapshot(original);
    }
    cold_riming_live::initialize(0, 2, 268.15, 1, true, 1e-3f, conversion_mode);
    const Snapshot source;
    for (std::size_t mutation = 0; mutation < default_cold_parameters.size(); ++mutation)
    {
        auto p = default_cold_parameters;
        p[mutation] = std::nextafter(p[mutation], std::numeric_limits<double>::infinity());
        tmv::MilbrandtScheme candidate(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
            p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
            p[20], p[21], p[22], p[23], p[24], true);
        const auto identity = candidate.parameter_identity();
        std::string error;
        CHECK_FALSE(enable(candidate, global_simulation_state(), source, conversion_mode, error));
        CHECK_FALSE(error.empty());
        CHECK_FALSE(candidate.owns_primary_endpoint_commit());
        CHECK_FALSE(candidate.owns_post_primary_private_commit());
        CHECK(candidate.parameter_identity() == identity);
        check_snapshot(source);
    }
    for (int mutation = 0; mutation < 4; ++mutation)
    {
        tmv::MilbrandtScheme candidate;
        auto state = global_simulation_state();
        const int old_nr = NR;
        switch (mutation)
        {
            case 0: state.storage_owner = nullptr; break;
            case 1: state.grid.nr = 4097; break;
            case 2: state.compute_runtime = nullptr; break;
            case 3: state.terrain_runtime = nullptr; break;
        }
        std::string error;
        CHECK_FALSE(enable(candidate, state, source, conversion_mode, error));
        NR = old_nr;
        CHECK_FALSE(candidate.owns_primary_endpoint_commit());
        CHECK_FALSE(candidate.owns_post_primary_private_commit());
        check_snapshot(source);
    }
}

void check_thermal_boundary(int conversion_mode, int scenario = 0)
{
    cold_riming_live::initialize(scenario, 2, 268.15, 1, true, 1e-3f, conversion_mode);
    const int last = physical_scalar_domain(global_simulation_state().grid).i_end - 1;
    bool selected = false;
    double selected_exner = 0;
    float lower_theta = 0;
    float upper_theta = 0;
    for (float pressure : {65000.0f, 70000.0f, 75000.0f, 80000.0f, 85000.0f, 90000.0f, 95000.0f})
    {
        const double exner = std::pow(static_cast<double>(pressure) / microphysics_constants::p0,
            microphysics_constants::R_d / microphysics_constants::cp);
        const double threshold = microphysics_constants::T0 / exner;
        const float nearest = static_cast<float>(threshold);
        const float lower = static_cast<double>(nearest) < threshold
            ? nearest : std::nextafter(nearest, -std::numeric_limits<float>::infinity());
        const float upper = std::nextafter(lower, std::numeric_limits<float>::infinity());
        if (threshold > (static_cast<double>(lower) + upper) * .5)
        {
            p(last, 0, 0) = pressure;
            theta(last, 0, 0) = lower;
            selected_exner = exner;
            lower_theta = lower;
            upper_theta = upper;
            qv(last, 0, 0) = static_cast<float>(cold_riming_live::reference::thermal::saturation(
                static_cast<double>(lower) * exner, pressure).mass);
            selected = true;
            break;
        }
    }
    REQUIRE(selected);
    const Snapshot before;
    const double source_temperature = static_cast<double>(lower_theta) * selected_exner;
    const double midpoint = (static_cast<double>(lower_theta) + upper_theta) * .5 * selected_exner;
    const double target = (midpoint + microphysics_constants::T0) * .5;
    const auto probe = cold_riming_live::capture(before, last, 1e-6, conversion_mode);
    REQUIRE(probe.temperature_increment_k > 0);
    const double duration = (target - source_temperature) / (probe.temperature_increment_k / 1e-6);
    const auto pure = cold_riming_live::capture(before, last, duration, conversion_mode);
    REQUIRE(source_temperature + pure.temperature_increment_k < microphysics_constants::T0);
    REQUIRE(static_cast<double>(static_cast<float>(static_cast<double>(lower_theta)
        + pure.temperature_increment_k / selected_exner)) * selected_exner >= microphysics_constants::T0);
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), duration), Catch::Matchers::Contains("final heat"));
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(before);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration * .1));
    cold_riming_live::check_step(before, duration * .1, conversion_mode);
}

} // namespace cold_conversion_live

TEST_CASE("live ice to graupel continues identically with actual one and four workers",
          "[physics][milbrandt][triple_cold_ice_graupel_internal][direct]")
{
    cold_conversion_live::check_workers(1);
}

TEST_CASE("live ice to graupel primary and late failures restore every owner before recovery",
          "[physics][milbrandt][triple_cold_ice_graupel_internal][direct]")
{
    cold_conversion_live::check_rollback(1);
}

TEST_CASE("live ice to graupel zero events validate raw thermal native policy and unowned sources",
          "[physics][milbrandt][triple_cold_ice_graupel_internal][direct]")
{
    cold_conversion_live::check_zero_guards(1);
    cold_conversion_live::check_activation(1);
}

TEST_CASE("live ice to graupel binds private and public sources generations and guarded transport",
          "[physics][milbrandt][triple_cold_ice_graupel_internal][direct]")
{
    cold_conversion_live::check_bindings(1);
}

TEST_CASE("live ice to graupel rejects actual stored thermal phase crossing atomically and recovers",
          "[physics][milbrandt][triple_cold_ice_graupel_internal][direct]")
{
    cold_conversion_live::check_thermal_boundary(1);
}

TEST_CASE("live coupled graupel conversions conserve noncubic snow moments and all water heat reservoirs",
          "[physics][milbrandt][triple_cold_graupel_conversions_internal][direct]")
{
    bool ice_active = false;
    bool snow_active = false;
    for (int scenario : {0, 2, 3})
    {
        for (long double alpha : {0.0L, 2.0L, 20.0L})
        {
            for (bool hail : {false, true})
            {
                cold_riming_live::initialize(scenario, alpha, 268.15, 1, hail, 1e-3f, 2);
                const Snapshot before;
                const auto cell = cold_riming_live::source(before, 1);
                const auto parameters = frozen_collection_live::parameters(hail);
                ColdGraupelConversionsResult converted;
                ColdRimingResult prefix;
                std::string error;
                REQUIRE(evaluate_exact_triple_cold_graupel_conversions_cell(parameters, cell, 1e-4, converted, error));
                REQUIRE(evaluate_exact_triple_cold_riming_cell(parameters, cell, 1e-4, prefix, error,
                    4, nullptr, ColdSpectrumDomain::SnowConvertedGraupel512));
                const auto expected = milbrandt_cold_graupel_conversions_reference::evaluate(cell, 1e-4);
                CHECK(converted.ice_conversion.raw_mass_transfer_per_kg()
                    == Approx(static_cast<double>(expected.conversion_mass)).epsilon(3e-8).margin(1e-22));
                CHECK(converted.snow_conversion.raw_mass_transfer_per_kg()
                    == Approx(static_cast<double>(expected.snow_conversion_mass)).epsilon(3e-8).margin(1e-22));
                CHECK(converted.snow_conversion.donor_number_loss_per_kg()
                    == Approx(static_cast<double>(expected.snow_conversion_number)).epsilon(3e-8).margin(1e-20));
                CHECK(converted.snow_conversion.source_mass_squared_per_kg()
                    == Approx(static_cast<double>(expected.snow_mass_squared)).epsilon(3e-8).margin(1e-30));
                CHECK(converted.combined.producer.spectrum_domain() == ColdSpectrumDomain::SnowConvertedGraupel512);
                CHECK(converted.combined.temperature_increment_k == prefix.temperature_increment_k);
                CHECK(converted.combined.cloud_mass_increment_per_kg == prefix.cloud_mass_increment_per_kg);
                CHECK(converted.combined.cloud_number_increment_per_kg == prefix.cloud_number_increment_per_kg);
                CHECK(converted.combined.vapor_mass_increment_per_kg == prefix.vapor_mass_increment_per_kg);
                ice_active |= converted.ice_conversion.raw_mass_transfer_per_kg() > 0;
                snow_active |= converted.snow_conversion.raw_mass_transfer_per_kg() > 0;
                if (scenario == 3)
                {
                    REQUIRE(converted.ice_conversion.raw_mass_transfer_per_kg() == 0);
                    REQUIRE(converted.snow_conversion.raw_mass_transfer_per_kg() > 0);
                    REQUIRE(before.fields[11](1, 0, 0) == 0);
                    CHECK(converted.combined.unrounded.frozen[2].mass_per_kg > 0);
                    CHECK(converted.combined.unrounded.frozen[2].number_per_kg > 0);
                    CHECK(converted.combined.unrounded.frozen[2].reflectivity_per_kg > 0);
                }
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
                cold_riming_live::check_step(before, 1e-4, 2);
            }
        }
    }
    CHECK(ice_active);
    CHECK(snow_active);
}

TEST_CASE("live snow conversion creates represented high shape graupel and retains it on continuation",
          "[physics][milbrandt][triple_cold_graupel_conversions_internal][direct]")
{
    cold_riming_live::initialize(3, 199.9L, 271.15, 1, true, 1e-3f, 2);
    for (double duration : {1e-4, 2e-4, 5e-4})
    {
        const Snapshot before;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
        cold_riming_live::check_step(before, duration, 2);
        const Snapshot after;
        for (int i = 1; i < NR - 1; ++i)
        {
            const FrozenCollectionPopulation graupel{after.fields[11](i, 0, 0),
                after.moments[4](i, 0, 0), after.moments[9](i, 0, 0)};
            REQUIRE(graupel.mass_per_kg > 0);
            REQUIRE(graupel.number_per_kg > 0);
            REQUIRE(graupel.reflectivity_per_kg > 0);
            const long double represented = cold_riming_live::reference::collection::frozen::represented_alpha(graupel, 2);
            INFO("stored graupel shape " << static_cast<double>(represented));
            REQUIRE(represented > 200);
            REQUIRE(represented < 512);
        }
        tmv::MilbrandtScheme old_domain;
        std::string error;
        CHECK_FALSE(cold_conversion_live::enable(old_domain, global_simulation_state(), after, 1, error));
        CHECK_FALSE(old_domain.owns_primary_endpoint_commit());
        check_snapshot(after);
        simulation_time += duration;
    }
}

TEST_CASE("live graupel domain permits only bounded graupel extension without clipping moments",
          "[physics][milbrandt][triple_cold_graupel_conversions_internal][direct]")
{
    for (long double alpha : {201.0L, 418.72L, 500.0L})
    {
        cold_riming_live::initialize(0, 2, 268.15, 1, true, 1e-3f, 2);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        auto& zg = const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(3));
        for (int i = 1; i < NR - 1; ++i)
        {
            zg(i, 0, 0) = frozen_collection_live::reflectivity(2, qg(i, 0, 0),
                scheme.internal_triple_number_for_test(3)(i, 0, 0), alpha);
        }
        const Snapshot before;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-7));
        cold_riming_live::check_step(before, 1e-7, 2);
    }
    for (std::size_t species = 0; species < 4; ++species)
    {
        cold_riming_live::initialize(0, 2, 268.15, 1, true, 1e-3f, 2);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        auto& z = const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(species + 1));
        const int last = NR - 2;
        const std::array<Field3D*, 4> mass{&qi, &qs, &qg, &qh};
        const float old = z(last, 0, 0);
        z(last, 0, 0) = frozen_collection_live::reflectivity(species, (*mass[species])(last, 0, 0),
            scheme.internal_triple_number_for_test(species + 1)(last, 0, 0), species == 2 ? 513 : 201);
        const Snapshot bad;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-7));
        CHECK(getter_attempts == 0);
        CHECK(accepted_attempts == 0);
        check_snapshot(bad);
        const_cast<Field3D&>(dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme)
            .internal_triple_reflectivity_for_test(species + 1))(last, 0, 0) = old;
        const Snapshot recovered;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-7));
        cold_riming_live::check_step(recovered, 1e-7, 2);
    }
}

TEST_CASE("live coupled graupel conversions continue identically with actual one and four workers",
          "[physics][milbrandt][triple_cold_graupel_conversions_internal][direct]")
{
    cold_conversion_live::check_workers(2);
}

TEST_CASE("live coupled graupel conversions restore all owners after early and late failure",
          "[physics][milbrandt][triple_cold_graupel_conversions_internal][direct]")
{
    cold_conversion_live::check_rollback(2);
}

TEST_CASE("live coupled graupel zero events validate thermal policy and every unowned source",
          "[physics][milbrandt][triple_cold_graupel_conversions_internal][direct]")
{
    cold_conversion_live::check_zero_guards(2);
    cold_conversion_live::check_activation(2);
}

TEST_CASE("live coupled graupel conversions bind all sources generations and guarded transport",
          "[physics][milbrandt][triple_cold_graupel_conversions_internal][direct]")
{
    cold_conversion_live::check_bindings(2);
}

TEST_CASE("live coupled graupel conversions reject stored thermal crossing atomically and recover",
          "[physics][milbrandt][triple_cold_graupel_conversions_internal][direct]")
{
    cold_conversion_live::check_thermal_boundary(2);
}

namespace cold_homogeneous_live
{

namespace reference = milbrandt_cold_homogeneous_reference;

void check_journal(const Snapshot& before, double duration)
{
    const auto cell = cold_riming_live::source(before, 1);
    const auto expected = reference::evaluate(cell, duration);
    const auto actual = cold_riming_live::capture(before, 1, duration, 3);
    REQUIRE(actual.producer.homogeneous_freezing_enabled());
    REQUIRE(actual.producer.event_count() == 5);
    REQUIRE(actual.producer.homogeneous_freezing().event_count() == 1);
    const auto near = [](double value, long double oracle, long double scale)
    {
        CHECK(std::abs(static_cast<long double>(value) - oracle)
            <= 3e-8L * std::max(std::abs(oracle), scale) + 1e-35L);
    };
    const auto& raw = actual.producer.homogeneous_freezing().channel();
    near(raw.fraction, expected.raw_fraction, 1e-25L);
    near(raw.mean_volume_cm3, expected.mean_volume_cm3, 1e-25L);
    near(raw.source_mass_squared_per_kg, expected.source_cloud_mass_squared, 1e-35L);
    near(raw.raw_mass_increment_per_kg, expected.raw_freezing[0], 1e-30L);
    near(raw.raw_number_increment_per_kg, expected.raw_freezing[1], 1e-25L);
    near(raw.raw_reflectivity_increment_per_kg, expected.raw_freezing[2], 1e-40L);
    near(actual.producer.cloud_admission_factor(), expected.factor, 1e-25L);
    for (std::size_t moment = 0; moment < 3; ++moment)
    {
        near(actual.producer.homogeneous_increments_per_kg()[moment], expected.admitted_freezing[moment],
            moment == 2 ? 1e-40L : 1e-25L);
    }
    double mass = actual.producer.homogeneous_increments_per_kg()[0];
    double number = actual.producer.homogeneous_increments_per_kg()[1];
    for (const auto& channel : actual.producer.channels())
    {
        mass += channel.collection_mass_per_kg;
        number += channel.collection_number_per_kg;
    }
    CHECK(mass <= cell.initial.cloud_mass_per_kg);
    CHECK(number <= cell.initial.cloud_number_per_kg);
    near(actual.cloud_mass_increment_per_kg, -mass, 1e-30L);
    near(actual.cloud_number_increment_per_kg, -number, 1e-25L);
    near(actual.temperature_increment_k, (microphysics_constants::L_f * mass
        - microphysics_constants::L_s * actual.vapor_mass_increment_per_kg) / microphysics_constants::cp, 1e-25L);
}

reference::State integrate(const ColdRimingCell& source, double duration, int steps)
{
    auto cell = source;
    std::string error;
    const auto parameters = frozen_collection_live::parameters(true);
    for (int step = 0; step < steps; ++step)
    {
        ColdGraupelConversionsResult result;
        REQUIRE(evaluate_exact_triple_cold_graupel_conversions_cell(parameters, cell,
            duration / steps, result, error, 2, nullptr, true));
        REQUIRE(result.combined.producer.cloud_admission_factor() == 1);
        cell.temperature_k += result.combined.temperature_increment_k;
        cell.initial = result.combined.unrounded;
    }
    return reference::riming::state(cell);
}

long double scaled_difference(const reference::State& a, const reference::State& b,
                              const reference::State& initial, const reference::State& target)
{
    long double result = 0;
    for (std::size_t index = 0; index < a.size(); ++index)
    {
        const long double floor = index == 0 ? 1e-8L : index == 3 || (index >= 4 && index % 3 == 2)
            ? 1e-8L : index >= 4 && index % 3 == 0 ? 1e-38L : 1e-22L;
        const long double scale = std::max(std::abs(target[index] - initial[index]), floor);
        result = std::max(result, std::abs(a[index] - b[index]) / scale);
    }
    return result;
}

} // namespace cold_homogeneous_live

TEST_CASE("live composed homogeneous cloud freezing preserves empty inactive half and full source journals",
          "[physics][milbrandt][triple_cold_homogeneous_internal][direct]")
{
    for (double temperature : {243.16, 237.15, 222.15, 150.01})
    {
        INFO("homogeneous temperature " << temperature);
        cold_riming_live::initialize(6, 2, temperature, 1, true, 1e-3f, 3,
            temperature == 150.01 ? 30000 : 0);
        const Snapshot before;
        const auto cell = cold_riming_live::source(before, 1);
        const auto one_second = cold_homogeneous_live::reference::evaluate(cell, 1);
        const double duration = temperature == 237.15
            ? static_cast<double>(std::log(2.0L) / one_second.hazard) : 1e-4;
        const auto result = cold_riming_live::capture(before, 1, duration, 3);
        if (temperature == 243.16)
        {
            CHECK(result.producer.homogeneous_freezing().channel().fraction == 0);
            CHECK(result.unrounded.frozen[0].mass_per_kg == 0);
        }
        else
        {
            CHECK(result.unrounded.frozen[0].mass_per_kg > 0);
            CHECK(result.unrounded.frozen[0].number_per_kg > 0);
            CHECK(result.unrounded.frozen[0].reflectivity_per_kg > 0);
            CHECK(result.producer.channels()[0].raw_collection_mass_per_kg == 0);
            CHECK(result.producer.channels()[0].diffusion_mass_per_kg == 0);
            CHECK(result.producer.homogeneous_freezing().channel().fraction
                == Approx(temperature == 237.15 ? .5 : 1).epsilon(3e-8));
        }
        cold_homogeneous_live::check_journal(before, duration);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
        cold_riming_live::check_step(before, duration, 3);
    }
    for (float cloud : {0.0f, 1e-15f})
    {
        cold_riming_live::initialize(6, 2, 237.15, 1, true, cloud, 3);
        const Snapshot before;
        CHECK(cold_riming_live::capture(before, 1, 1e-4, 3).producer.homogeneous_freezing().channel().fraction == 0);
        cold_homogeneous_live::check_journal(before, 1e-4);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 3);
    }
}

TEST_CASE("live homogeneous freezing shares one paired cloud budget with all four collectors",
          "[physics][milbrandt][triple_cold_homogeneous_internal][direct]")
{
    for (bool hail : {false, true})
    {
        for (double temperature : {239.15, 222.15})
        {
            cold_riming_live::initialize(0, 2, temperature, 1, hail, 1e-3f, 3);
            const Snapshot before;
            const auto result = cold_riming_live::capture(before, 1, 1e-4, 3);
            CHECK(result.producer.hail_processes() == hail);
            REQUIRE(result.producer.homogeneous_freezing().channel().raw_mass_increment_per_kg > 0);
            for (const auto& channel : result.producer.channels())
            {
                REQUIRE(channel.raw_collection_mass_per_kg > 0);
                REQUIRE(channel.raw_collection_number_per_kg > 0);
                CHECK(channel.collection_mass_per_kg == channel.raw_collection_mass_per_kg
                    * result.producer.cloud_admission_factor());
                CHECK(channel.collection_number_per_kg == channel.raw_collection_number_per_kg
                    * result.producer.cloud_admission_factor());
            }
            if (temperature == 222.15)
            {
                REQUIRE(result.producer.cloud_admission_factor() < 1);
            }
            cold_homogeneous_live::check_journal(before, 1e-4);
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
            cold_riming_live::check_step(before, 1e-4, 3);
        }
    }
}

TEST_CASE("live composed homogeneous freezing continues identically with actual one and four workers",
          "[physics][milbrandt][triple_cold_homogeneous_internal][direct]")
{
#ifdef _OPENMP
    struct ThreadRestore
    {
        int workers = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~ThreadRestore()
        {
            omp_set_num_threads(workers);
            omp_set_dynamic(dynamic);
        }
    } restore;
    omp_set_dynamic(0);
    for (int scenario : {0, 6})
    {
        std::array<std::unique_ptr<Snapshot>, 2> final;
        for (int run = 0; run < 2; ++run)
        {
            const int workers = run == 0 ? 1 : 4;
            omp_set_num_threads(workers);
            int observed = 0;
#pragma omp parallel
            {
#pragma omp single
                observed = omp_get_num_threads();
            }
            REQUIRE(observed == workers);
            cold_riming_live::initialize(scenario, 2, 237.15, 1, true, 1e-3f, 3);
            for (double duration : {1e-5, 2e-5, 5e-5})
            {
                const Snapshot before;
                cold_homogeneous_live::check_journal(before, duration);
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                cold_riming_live::check_step(before, duration, 3);
                simulation_time += duration;
            }
            final[run] = std::make_unique<Snapshot>();
        }
        check_snapshot(*final[0], *final[1]);
    }
#else
    FAIL("homogeneous cloud freezing worker evidence requires OpenMP");
#endif
}

TEST_CASE("live composed homogeneous freezing rolls back active early and late endpoint failures",
          "[physics][milbrandt][triple_cold_homogeneous_internal][direct]")
{
    for (int mutation = 1; mutation <= 25; ++mutation)
    {
        cold_riming_live::initialize(0, 2, 237.15, 1, true, 1e-3f, 3);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = mutation;
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == 0);
        check_snapshot(before);
        dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme).getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 3);
    }
    for (int mutation : {0, 1, 2, 6, 7, 8})
    {
        cold_riming_live::initialize(0, 2, 237.15, 1, true, 1e-3f, 3);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.reject = mutation == 0;
        scheme.late_mutation = mutation;
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == 1);
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 3);
    }
}

TEST_CASE("live composed homogeneous owner validates source policy resources and capture feature identity",
          "[physics][milbrandt][triple_cold_homogeneous_internal][direct]")
{
    cold_conversion_live::check_zero_guards(3);
    cold_conversion_live::check_activation(3);
    cold_conversion_live::check_bindings(3);
    cold_riming_live::initialize(6, 2, 237.15, 1, true, 1e-3f, 3);
    const Snapshot before;
    const auto cell = cold_riming_live::source(before, 1);
    const auto result = cold_riming_live::capture(before, 1, 1e-4, 3);
    ColdRimingResult unpublished;
    unpublished.cloud_mass_increment_per_kg = 42;
    ColdRimingFailure failure = ColdRimingFailure::None;
    std::string error;
    const auto parameters = frozen_collection_live::parameters(true);
    CHECK_FALSE(apply_exact_triple_cold_riming_cell(parameters, cell, 1e-4, result.producer,
        unpublished, error, 4, &failure, ColdSpectrumDomain::SnowConvertedGraupel512, false));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    CHECK(unpublished.cloud_mass_increment_per_kg == 42);
    CHECK_FALSE(apply_exact_triple_cold_riming_cell(parameters, cell, 1e-4, result.producer,
        unpublished, error, 4, &failure, ColdSpectrumDomain::SnowConvertedGraupel512, true));
    CHECK(failure == ColdRimingFailure::Capacity);
    CHECK(unpublished.cloud_mass_increment_per_kg == 42);
    auto changed = cell;
    changed.initial.cloud_number_per_kg = std::nextafter(changed.initial.cloud_number_per_kg,
        std::numeric_limits<double>::infinity());
    CHECK_FALSE(apply_exact_triple_cold_riming_cell(parameters, changed, 1e-4, result.producer,
        unpublished, error, 5, &failure, ColdSpectrumDomain::SnowConvertedGraupel512, true));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    CHECK(unpublished.cloud_mass_increment_per_kg == 42);
    check_snapshot(before);
}

TEST_CASE("live composed homogeneous owner rejects raw and stored thermal crossings atomically",
          "[physics][milbrandt][triple_cold_homogeneous_internal][direct]")
{
    cold_conversion_live::check_thermal_boundary(3);
    cold_riming_live::initialize(6, 2, 222.15, 1, true, 1e-3f, 3);
    const Snapshot before;
    auto cell = cold_riming_live::source(before, 1);
    cell.initial.cloud_mass_per_kg = .2;
    ColdGraupelConversionsResult unpublished;
    unpublished.combined.cloud_mass_increment_per_kg = 42;
    ColdRimingFailure failure = ColdRimingFailure::None;
    std::string error;
    CHECK_FALSE(evaluate_exact_triple_cold_graupel_conversions_cell(frozen_collection_live::parameters(true),
        cell, 1e-4, unpublished, error, 2, &failure, true));
    CHECK(failure == ColdRimingFailure::IntervalRejected);
    CHECK(unpublished.combined.cloud_mass_increment_per_kg == 42);
    check_snapshot(before);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(before, 1e-4, 3);
}

TEST_CASE("composed finite hazard homogeneous freezing refines against independent full state RK4",
          "[physics][milbrandt][triple_cold_homogeneous_internal][reference]")
{
    cold_riming_live::initialize(6, 2, 237.15, 1, true, 1e-3f, 3);
    const Snapshot before;
    const auto cell = cold_riming_live::source(before, 1);
    constexpr double duration = .004;
    const auto initial = cold_homogeneous_live::reference::riming::state(cell);
    const auto target = cold_homogeneous_live::reference::rk4(cell, duration, 2048);
    const auto target_half = cold_homogeneous_live::reference::rk4(cell, duration, 1024);
    const auto coarse = cold_homogeneous_live::integrate(cell, duration, 16);
    const auto medium = cold_homogeneous_live::integrate(cell, duration, 32);
    const auto fine = cold_homogeneous_live::integrate(cell, duration, 64);
    const auto error = [&](const cold_homogeneous_live::reference::State& candidate)
    {
        return cold_homogeneous_live::scaled_difference(candidate, target, initial, target);
    };
    const long double coarse_error = error(coarse);
    const long double medium_error = error(medium);
    const long double fine_error = error(fine);
    CAPTURE(coarse_error, medium_error, fine_error);
    REQUIRE(coarse_error > 1e-9L);
    CHECK(medium_error < .75L * coarse_error);
    CHECK(fine_error < .75L * medium_error);
    CHECK(cold_homogeneous_live::scaled_difference(target_half, target, initial, target) < 1e-8L);
    CHECK(target[2] < initial[2]);
    CHECK(target[4] > initial[4]);
    CHECK(target[5] > initial[5]);
    CHECK(target[6] > initial[6]);
    CHECK(target[0] > initial[0]);
    check_snapshot(before);
}

namespace cold_sll_live
{

namespace reference = milbrandt_cold_sll_reference;

ColdSllResult capture(const Snapshot& before, int i, double duration)
{
    ColdSllResult result;
    std::string error;
    const auto parameters = frozen_collection_live::parameters(
        global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes);
    REQUIRE(evaluate_exact_triple_cold_sll_cell(parameters, cold_riming_live::source(before, i),
        duration, result, error));
    return result;
}

void check_journal(const Snapshot& before, double duration)
{
    const auto cell = cold_riming_live::source(before, 1);
    const bool hail = global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes;
    const auto expected = reference::evaluate(cell, duration, hail);
    const auto actual = capture(before, 1, duration);
    const auto& conversion = actual.hail_conversion;
    REQUIRE(conversion.event_count() == 1);
    CHECK(conversion.hail_processes() == hail);
    CHECK(conversion.spectrum_domain() == ColdSpectrumDomain::GraupelAndHail512);
    CHECK(actual.combined.producer.spectrum_domain() == ColdSpectrumDomain::GraupelAndHail512);
    CHECK(actual.combined.producer.homogeneous_freezing_enabled());
    CHECK(actual.combined.producer.event_count() == 5);
    CHECK(conversion.source().rain_mass_per_kg == 0);
    CHECK(conversion.source().admitted_accretion_mass_per_kg[0]
        == actual.combined.producer.channels()[2].collection_mass_per_kg);
    CHECK(conversion.source().admitted_accretion_mass_per_kg[1] == 0);
    CHECK(conversion.source().admitted_accretion_mass_per_kg[2] == 0);
    CHECK(conversion.source().graupel_diffusion_mass_per_kg
        == actual.combined.producer.channels()[2].diffusion_mass_per_kg);
    CHECK(conversion.source().graupel_diffusion_number_per_kg
        == actual.combined.producer.channels()[2].diffusion_number_per_kg);
    for (std::size_t species = 0; species < 4; ++species)
    {
        CHECK(conversion.source().source.initial.frozen[species].mass_per_kg == cell.initial.frozen[species].mass_per_kg);
        CHECK(conversion.source().source.initial.frozen[species].number_per_kg == cell.initial.frozen[species].number_per_kg);
        CHECK(conversion.source().source.initial.frozen[species].reflectivity_per_kg == cell.initial.frozen[species].reflectivity_per_kg);
    }
    const auto near = [](double value, long double oracle, long double scale)
    {
        CHECK(std::abs(static_cast<long double>(value) - oracle)
            <= 3e-8L * std::max(std::abs(oracle), scale) + 1e-35L);
    };
    near(conversion.loading_denominator(), expected.sll.loading, 1e-22L);
    near(conversion.raw_mass_transfer_per_kg(), expected.sll.raw, 1e-22L);
    near(conversion.admitted_mass_transfer_per_kg(), expected.sll.admitted, 1e-22L);
    near(conversion.admitted_fraction(), expected.sll.fraction, 1e-22L);
    for (std::size_t species = 0; species < 4; ++species)
    {
        near(conversion.mass_increment_per_kg()[species], expected.sll.mass[species], 1e-22L);
        near(conversion.number_increment_per_kg()[species], expected.sll.number[species], 1e-20L);
        near(conversion.reflectivity_increment_per_kg()[species], expected.sll.reflectivity[species], 1e-40L);
    }
    CHECK(conversion.mass_increment_per_kg()[2] + conversion.mass_increment_per_kg()[3] == 0);
    CHECK(conversion.number_increment_per_kg()[2] + conversion.number_increment_per_kg()[3] == 0);
    near(actual.combined.temperature_increment_k, expected.temperature_increment, 1e-15L);
    const auto prefix = reference::homogeneous::evaluate(cell, duration);
    near(actual.combined.temperature_increment_k, prefix.temperature_increment, 1e-15L);
}

reference::State integrate(const ColdRimingCell& source, double duration, int steps)
{
    auto cell = source;
    std::string error;
    for (int step = 0; step < steps; ++step)
    {
        ColdSllResult result;
        REQUIRE(evaluate_exact_triple_cold_sll_cell(frozen_collection_live::parameters(true), cell,
            duration / steps, result, error));
        REQUIRE(result.combined.producer.cloud_admission_factor() == 1);
        REQUIRE(result.hail_conversion.raw_mass_transfer_per_kg() == result.hail_conversion.admitted_mass_transfer_per_kg());
        cell.temperature_k += result.combined.temperature_increment_k;
        cell.initial = result.combined.unrounded;
    }
    return reference::riming::state(cell);
}

} // namespace cold_sll_live

TEST_CASE("live cloud fed SLL binds original source cloud loading without fresh conversion funding or heat",
          "[physics][milbrandt][triple_cold_sll_internal][direct]")
{
    for (int scenario : {0, 2, 3, 4, 6})
    {
        cold_riming_live::initialize(scenario, 2, 268.15, 1, true, 1e-3f, 4);
        const Snapshot before;
        const auto result = cold_sll_live::capture(before, 1, 1e-4);
        if (scenario == 0 || scenario == 4)
        {
            REQUIRE(result.hail_conversion.admitted_mass_transfer_per_kg() > 0);
        }
        else
        {
            CHECK(result.hail_conversion.admitted_mass_transfer_per_kg() == 0);
            CHECK(result.hail_conversion.regime() == GraupelHailSllRegime::NoGraupel);
            if (scenario == 2 || scenario == 3)
            {
                REQUIRE(result.combined.unrounded.frozen[2].mass_per_kg > 0);
                CHECK(result.combined.unrounded.frozen[3].mass_per_kg == 0);
            }
        }
        cold_sll_live::check_journal(before, 1e-4);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 4);
    }
    for (float cloud : {0.0f, 1e-15f})
    {
        cold_riming_live::initialize(4, 2, 268.15, 1, true, cloud, 4);
        const Snapshot before;
        CHECK(cold_sll_live::capture(before, 1, 1e-4).hail_conversion.admitted_mass_transfer_per_kg() == 0);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 4);
    }
    cold_riming_live::initialize(4, 2, 272.15, 1, false, .01f, 4);
    const Snapshot disabled;
    const auto result = cold_sll_live::capture(disabled, 1, 1e-4);
    CHECK(result.hail_conversion.regime() == GraupelHailSllRegime::HailDisabled);
    CHECK(result.hail_conversion.admitted_mass_transfer_per_kg() == 0);
    cold_sll_live::check_journal(disabled, 1e-4);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(disabled, 1e-4, 4);
}

TEST_CASE("live cloud fed SLL admits represented G and H above200 through conservative continuation",
          "[physics][milbrandt][triple_cold_sll_internal][direct]")
{
    for (int scenario : {4, 7})
    {
        cold_riming_live::initialize(scenario, 300, 272.15, 1, true, .01f, 4);
        for (double duration : {1e-4, 2e-4, 5e-4})
        {
            const Snapshot before;
            const auto result = cold_sll_live::capture(before, 1, duration);
            REQUIRE(result.hail_conversion.admitted_mass_transfer_per_kg() > 0);
            cold_sll_live::check_journal(before, duration);
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
            cold_riming_live::check_step(before, duration, 4);
            const Snapshot after;
            for (int i = 1; i < NR - 1; ++i)
            {
                for (std::size_t species : {2u, 3u})
                {
                    const FrozenCollectionPopulation population{after.fields[species + 9](i, 0, 0),
                        after.moments[species + 2](i, 0, 0), after.moments[species + 7](i, 0, 0)};
                    REQUIRE(population.mass_per_kg > 0);
                    const long double alpha = cold_sll_live::reference::riming::collection::frozen::represented_alpha(population, species);
                    CAPTURE(scenario, species, alpha);
                    CHECK(alpha > 200);
                    CHECK(alpha <= 512);
                }
            }
            tmv::MilbrandtScheme old_domain;
            std::string error;
            CHECK_FALSE(cold_conversion_live::enable(old_domain, global_simulation_state(), after, 3, error));
            CHECK_FALSE(error.empty());
            check_snapshot(after);
        }
    }
}

TEST_CASE("live cloud fed SLL continues with actual one and four workers without publication differences",
          "[physics][milbrandt][triple_cold_sll_internal][direct]")
{
#ifdef _OPENMP
    struct ThreadRestore
    {
        int workers = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~ThreadRestore()
        {
            omp_set_num_threads(workers);
            omp_set_dynamic(dynamic);
        }
    } restore;
    omp_set_dynamic(0);
    for (int scenario : {0, 4})
    {
        std::array<std::unique_ptr<Snapshot>, 2> final;
        for (int run = 0; run < 2; ++run)
        {
            const int workers = run == 0 ? 1 : 4;
            omp_set_num_threads(workers);
            int observed = 0;
#pragma omp parallel
            {
#pragma omp single
                observed = omp_get_num_threads();
            }
            REQUIRE(observed == workers);
            cold_riming_live::initialize(scenario, scenario == 4 ? 300 : 2, 272.15, 1, true, .01f, 4);
            for (double duration : {1e-4, 2e-4, 5e-4})
            {
                const Snapshot before;
                REQUIRE(cold_sll_live::capture(before, 1, duration).hail_conversion.admitted_mass_transfer_per_kg() > 0);
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                cold_riming_live::check_step(before, duration, 4);
                simulation_time += duration;
            }
            final[run] = std::make_unique<Snapshot>();
        }
        check_snapshot(*final[0], *final[1]);
    }
#else
    FAIL("cloud fed SLL worker evidence requires OpenMP");
#endif
}

TEST_CASE("live cloud fed SLL restores every owner after active early and late failures",
          "[physics][milbrandt][triple_cold_sll_internal][direct]")
{
    cold_conversion_live::check_rollback(4);
}

TEST_CASE("live cloud fed SLL validates public native policy resources and exact source journals",
          "[physics][milbrandt][triple_cold_sll_internal][direct]")
{
    cold_conversion_live::check_zero_guards(4);
    cold_conversion_live::check_activation(4);
    cold_conversion_live::check_bindings(4);
    cold_riming_live::initialize(4, 2, 272.15, 1, true, .01f, 4);
    const Snapshot before;
    const auto cell = cold_riming_live::source(before, 1);
    const auto journal = cold_sll_live::capture(before, 1, 1e-4);
    ColdSllResult unpublished;
    unpublished.combined.cloud_mass_increment_per_kg = 42;
    ColdRimingFailure failure = ColdRimingFailure::None;
    std::string error;
    auto changed = cell;
    changed.initial.frozen[2].number_per_kg = std::nextafter(changed.initial.frozen[2].number_per_kg,
        std::numeric_limits<double>::infinity());
    CHECK_FALSE(apply_exact_triple_cold_sll_cell(frozen_collection_live::parameters(true), changed, 1e-4,
        journal.ice_conversion, journal.snow_conversion, journal.hail_conversion, unpublished, error, 3, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    CHECK(unpublished.combined.cloud_mass_increment_per_kg == 42);
    CHECK_FALSE(apply_exact_triple_cold_sll_cell(frozen_collection_live::parameters(true), cell, 1e-4,
        journal.ice_conversion, journal.snow_conversion, journal.hail_conversion, unpublished, error, 2, &failure));
    CHECK(failure == ColdRimingFailure::Capacity);
    CHECK(unpublished.combined.cloud_mass_increment_per_kg == 42);
    CHECK_FALSE(apply_exact_triple_cold_sll_cell(frozen_collection_live::parameters(false), cell, 1e-4,
        journal.ice_conversion, journal.snow_conversion, journal.hail_conversion, unpublished, error, 3, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    CHECK(unpublished.combined.cloud_mass_increment_per_kg == 42);
    check_snapshot(before);
}

TEST_CASE("live cloud fed SLL retains strict raw and stored thermal guards without extra conversion heat",
          "[physics][milbrandt][triple_cold_sll_internal][direct]")
{
    cold_conversion_live::check_thermal_boundary(4, 3);
    cold_riming_live::initialize(6, 2, 222.15, 1, true, 1e-3f, 4);
    const Snapshot before;
    auto cell = cold_riming_live::source(before, 1);
    cell.initial.cloud_mass_per_kg = .2;
    ColdSllResult unpublished;
    unpublished.combined.cloud_mass_increment_per_kg = 42;
    ColdRimingFailure failure = ColdRimingFailure::None;
    std::string error;
    CHECK_FALSE(evaluate_exact_triple_cold_sll_cell(frozen_collection_live::parameters(true), cell, 1e-4,
        unpublished, error, 3, &failure));
    CHECK(failure == ColdRimingFailure::IntervalRejected);
    CHECK(unpublished.combined.cloud_mass_increment_per_kg == 42);
    check_snapshot(before);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(before, 1e-4, 4);
}

TEST_CASE("cloud fed SLL full state integration refines toward the independent conservative moment ODE",
          "[physics][milbrandt][triple_cold_sll_internal][reference]")
{
    cold_riming_live::initialize(4, 2, 272.15, 1, true, .01f, 4);
    const Snapshot before;
    const auto cell = cold_riming_live::source(before, 1);
    constexpr double duration = .01;
    const auto initial = cold_sll_live::reference::riming::state(cell);
    const auto target = cold_sll_live::reference::rk4(cell, duration, 2048);
    const auto target_half = cold_sll_live::reference::rk4(cell, duration, 1024);
    const auto coarse = cold_sll_live::integrate(cell, duration, 16);
    const auto medium = cold_sll_live::integrate(cell, duration, 32);
    const auto fine = cold_sll_live::integrate(cell, duration, 64);
    const auto error = [&](const cold_sll_live::reference::State& candidate)
    {
        return cold_homogeneous_live::scaled_difference(candidate, target, initial, target);
    };
    const long double coarse_error = error(coarse);
    const long double medium_error = error(medium);
    const long double fine_error = error(fine);
    CAPTURE(coarse_error, medium_error, fine_error);
    REQUIRE(coarse_error > 1e-9L);
    CHECK(medium_error < .75L * coarse_error);
    CHECK(fine_error < .75L * medium_error);
    CHECK(cold_homogeneous_live::scaled_difference(target_half, target, initial, target) < 1e-8L);
    CHECK(target[13] > initial[13]);
    CHECK(target[14] > initial[14]);
    CHECK(target[15] > initial[15]);
    check_snapshot(before);
}

namespace cold_collected_sll_live
{

ColdCollectedSllResult capture(const Snapshot& before, int i, double duration)
{
    ColdCollectedSllResult result;
    std::string error;
    const auto parameters = frozen_collection_live::parameters(
        global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes);
    const bool accepted = evaluate_exact_triple_cold_collected_sll_cell(parameters,
        cold_riming_live::source(before, i), duration, result, error);
    INFO(error);
    REQUIRE(accepted);
    return result;
}

} // namespace cold_collected_sll_live

TEST_CASE("live collected SLL feeds the accepted original ice collision to hail without a second heat or cloud debit",
          "[physics][milbrandt][triple_cold_collected_sll_internal][direct]")
{
    for (bool hail : {false, true})
    {
        cold_riming_live::initialize(0, 2, 268.15, 1, hail, 1e-3f, 5);
        const Snapshot before;
        const auto actual = cold_collected_sll_live::capture(before, 1, 1e-4);
        const auto& collection = actual.collection.channels();
        REQUIRE(collection[1].admitted_mass_increment_per_kg > 0);
        CHECK(actual.collection.spectrum_domain() == ColdSpectrumDomain::GraupelAndHail512);
        CHECK(actual.hail_conversion.source().admitted_accretion_mass_per_kg[2]
            == collection[1].admitted_mass_increment_per_kg);
        CHECK(actual.hail_conversion.source().admitted_accretion_mass_per_kg[1] == 0);
        CHECK(actual.hail_conversion.source().admitted_accretion_mass_per_kg[0]
            == actual.combined.producer.channels()[2].collection_mass_per_kg);
        const auto cloud_only = cold_sll_live::capture(before, 1, 1e-4);
        CHECK(actual.combined.temperature_increment_k == cloud_only.combined.temperature_increment_k);
        CHECK(actual.combined.cloud_mass_increment_per_kg == cloud_only.combined.cloud_mass_increment_per_kg);
        CHECK(actual.combined.cloud_number_increment_per_kg == cloud_only.combined.cloud_number_increment_per_kg);
        CHECK(actual.combined.vapor_mass_increment_per_kg == cloud_only.combined.vapor_mass_increment_per_kg);
        if (hail)
        {
            REQUIRE(actual.hail_conversion.raw_mass_transfer_per_kg() > cloud_only.hail_conversion.raw_mass_transfer_per_kg());
        }
        else
        {
            CHECK(actual.hail_conversion.regime() == GraupelHailSllRegime::HailDisabled);
            CHECK(actual.hail_conversion.admitted_mass_transfer_per_kg() == 0);
            REQUIRE(collection[2].admitted_mass_increment_per_kg > 0);
        }
        const auto oracle = milbrandt_cold_collected_sll_reference::evaluate(cold_riming_live::source(before, 1), 1e-4, hail);
        REQUIRE(milbrandt_cold_collected_sll_reference::original_donors_admissible(cold_riming_live::source(before, 1), oracle));
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 5);
    }
}

TEST_CASE("live collected SLL continues with original collisions and higher G H shapes on actual one and four workers",
          "[physics][milbrandt][triple_cold_collected_sll_internal][direct]")
{
#ifdef _OPENMP
    struct ThreadRestore
    {
        int workers = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~ThreadRestore()
        {
            omp_set_num_threads(workers);
            omp_set_dynamic(dynamic);
        }
    } restore;
    omp_set_dynamic(0);
    for (int scenario : {0, 7})
    {
        std::array<std::unique_ptr<Snapshot>, 2> final;
        for (int run = 0; run < 2; ++run)
        {
            const int workers = run == 0 ? 1 : 4;
            omp_set_num_threads(workers);
            int observed = 0;
#pragma omp parallel
            {
#pragma omp single
                observed = omp_get_num_threads();
            }
            REQUIRE(observed == workers);
            cold_riming_live::initialize(scenario, scenario == 7 ? 300 : 2, 272.15, 1, true, .01f, 5);
            for (double duration : {1e-4, 2e-4, 5e-4})
            {
                const Snapshot before;
                const auto actual = cold_collected_sll_live::capture(before, 1, duration);
                REQUIRE(actual.hail_conversion.admitted_mass_transfer_per_kg() > 0);
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                cold_riming_live::check_step(before, duration, 5);
                if (scenario == 7)
                {
                    const Snapshot after;
                    for (std::size_t species : {2u, 3u})
                    {
                        const FrozenCollectionPopulation population{after.fields[species + 9](1, 0, 0),
                            after.moments[species + 2](1, 0, 0), after.moments[species + 7](1, 0, 0)};
                        const long double alpha = milbrandt_frozen_collection_reference::represented_alpha(population, species);
                        CHECK(alpha > 200);
                        CHECK(alpha <= 512);
                    }
                }
                simulation_time += duration;
            }
            final[run] = std::make_unique<Snapshot>();
        }
        check_snapshot(*final[0], *final[1]);
    }
#else
    FAIL("collected SLL worker evidence requires OpenMP");
#endif
}

TEST_CASE("live collected SLL restores every owner after early and late active failures and recovers",
          "[physics][milbrandt][triple_cold_collected_sll_internal][direct]")
{
    for (int mutation : {0, 1, 2})
    {
        cold_riming_live::initialize(0, 2, 268.15, 1, true, 1e-3f, 5);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        if (mutation == 0)
        {
            scheme.getter_mutation = 1;
        }
        else
        {
            scheme.reject = mutation == 1;
            scheme.late_mutation = mutation == 2 ? 2 : 0;
        }
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == (mutation == 0 ? 0 : 1));
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.getter_mutation = 0;
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 5);
    }
}

TEST_CASE("live collected SLL retains exact resource and source capture guards without native fallback",
          "[physics][milbrandt][triple_cold_collected_sll_internal][direct]")
{
    cold_riming_live::initialize(0, 2, 268.15, 1, true, 1e-3f, 5);
    const Snapshot before;
    const auto source = cold_riming_live::source(before, 1);
    const auto journal = cold_collected_sll_live::capture(before, 1, 1e-4);
    ColdCollectedSllResult unpublished;
    unpublished.combined.cloud_mass_increment_per_kg = 42;
    ColdRimingFailure failure;
    std::string error;
    CHECK_FALSE(evaluate_exact_triple_cold_collected_sll_cell(frozen_collection_live::parameters(true), source,
        1e-4, unpublished, error, 7, &failure));
    CHECK(failure == ColdRimingFailure::Capacity);
    CHECK(unpublished.combined.cloud_mass_increment_per_kg == 42);
    auto changed = source;
    changed.initial.frozen[0].number_per_kg = std::nextafter(changed.initial.frozen[0].number_per_kg,
        std::numeric_limits<double>::infinity());
    CHECK_FALSE(apply_exact_triple_cold_collected_sll_cell(frozen_collection_live::parameters(true), changed, 1e6,
        journal.collection, journal.ice_conversion, journal.snow_conversion, journal.hail_conversion,
        unpublished, error, 8, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    CHECK(unpublished.combined.cloud_mass_increment_per_kg == 42);
    check_snapshot(before);
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    scheme.set_require_native_dispatch(true);
    const Snapshot guarded;
    std::array<Field3D, 8> rates;
    for (auto& field : rates)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    REQUIRE_THROWS_WITH(internal_thermal_live::compute(rates, 1e-4),
        Catch::Matchers::Contains("live CPU direct-step owner"));
    for (const auto& field : rates)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(guarded);
    scheme.set_require_native_dispatch(false);
    const Snapshot recovered;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(recovered, 1e-4, 5);
}

namespace cold_rain_sll_live
{
ColdCollectedSllResult capture(const Snapshot& before, int i, double duration)
{
    ColdCollectedSllResult result;
    std::string error;
    const bool ok = evaluate_exact_triple_cold_rain_sll_cell(frozen_collection_live::parameters(
        global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes),
        cold_riming_live::source(before, i), duration, result, error);
    INFO(error);
    REQUIRE(ok);
    return result;
}
}

TEST_CASE("live rain SLL publishes independently checked rain frozen water and heat once",
          "[physics][milbrandt][triple_cold_rain_sll_internal][direct]")
{
    for (bool hail : {false, true})
    for (double temperature : {272.15, 268.15, 238.15})
    {
        cold_riming_live::initialize(0, 2, temperature, 1, hail, 1e-3f, 6);
        const Snapshot before;
        const auto actual = cold_rain_sll_live::capture(before, 1, 1e-4);
        REQUIRE(actual.combined.producer.rain_collection().channels()[2].mass_per_kg > 0);
        REQUIRE(actual.combined.producer.rain_collection().channels()[3].mass_per_kg > 0);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 6);
    }
}

TEST_CASE("live rain SLL continues with original collisions and higher G H shapes on actual one and four workers",
          "[physics][milbrandt][triple_cold_rain_sll_internal][direct]")
{
#ifdef _OPENMP
    struct ThreadRestore
    {
        int workers = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~ThreadRestore()
        {
            omp_set_num_threads(workers);
            omp_set_dynamic(dynamic);
        }
    } restore;
    omp_set_dynamic(0);
    for (int scenario : {0, 7})
    {
        std::array<std::unique_ptr<Snapshot>, 2> final;
        for (int run = 0; run < 2; ++run)
        {
            const int workers = run == 0 ? 1 : 4;
            omp_set_num_threads(workers);
            int observed = 0;
#pragma omp parallel
            {
#pragma omp single
                observed = omp_get_num_threads();
            }
            REQUIRE(observed == workers);
            cold_riming_live::initialize(scenario, scenario == 7 ? 300 : 2, 272.15, 1, true, .01f, 6);
            for (double duration : {1e-4, 2e-4, 5e-4})
            {
                const Snapshot before;
                const auto actual = cold_rain_sll_live::capture(before, 1, duration);
                REQUIRE(actual.hail_conversion.admitted_mass_transfer_per_kg() > 0);
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                cold_riming_live::check_step(before, duration, 6);
                if (scenario == 7)
                {
                    const Snapshot after;
                    for (std::size_t species : {2u, 3u})
                    {
                        const FrozenCollectionPopulation population{after.fields[species + 9](1, 0, 0),
                            after.moments[species + 2](1, 0, 0), after.moments[species + 7](1, 0, 0)};
                        const long double alpha = milbrandt_frozen_collection_reference::represented_alpha(population, species);
                        CHECK(alpha > 200);
                        CHECK(alpha <= 512);
                    }
                }
                simulation_time += duration;
            }
            final[run] = std::make_unique<Snapshot>();
        }
        check_snapshot(*final[0], *final[1]);
    }
#else
    FAIL("collected SLL worker evidence requires OpenMP");
#endif
}

TEST_CASE("live rain SLL restores every owner after early and late active failures and recovers",
          "[physics][milbrandt][triple_cold_rain_sll_internal][direct]")
{
    for (int mutation : {0, 1, 2})
    {
        cold_riming_live::initialize(0, 2, 268.15, 1, true, 1e-3f, 6);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        if (mutation == 0)
        {
            scheme.getter_mutation = 1;
        }
        else
        {
            scheme.reject = mutation == 1;
            scheme.late_mutation = mutation == 2 ? 2 : 0;
        }
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == (mutation == 0 ? 0 : 1));
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.getter_mutation = 0;
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 6);
    }
}

TEST_CASE("live rain SLL retains exact resource and source capture guards without native fallback",
          "[physics][milbrandt][triple_cold_rain_sll_internal][direct]")
{
    cold_riming_live::initialize(0, 2, 268.15, 1, true, 1e-3f, 6);
    const Snapshot before;
    const auto source = cold_riming_live::source(before, 1);
    const auto journal = cold_rain_sll_live::capture(before, 1, 1e-4);
    ColdCollectedSllResult unpublished;
    unpublished.combined.cloud_mass_increment_per_kg = 42;
    ColdRimingFailure failure;
    std::string error;
    CHECK_FALSE(evaluate_exact_triple_cold_rain_sll_cell(frozen_collection_live::parameters(true), source,
        1e-4, unpublished, error, 11, &failure));
    CHECK(failure == ColdRimingFailure::Capacity);
    CHECK(unpublished.combined.cloud_mass_increment_per_kg == 42);
    auto changed = source;
    changed.initial.frozen[0].number_per_kg = std::nextafter(changed.initial.frozen[0].number_per_kg,
        std::numeric_limits<double>::infinity());
    CHECK_FALSE(apply_exact_triple_cold_rain_sll_cell(frozen_collection_live::parameters(true), changed, 1e6,
        journal.collection, journal.ice_conversion, journal.snow_conversion, journal.hail_conversion,
        unpublished, error, 12, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    CHECK(unpublished.combined.cloud_mass_increment_per_kg == 42);
    check_snapshot(before);
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    scheme.set_require_native_dispatch(true);
    const Snapshot guarded;
    std::array<Field3D, 8> rates;
    for (auto& field : rates)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    REQUIRE_THROWS_WITH(internal_thermal_live::compute(rates, 1e-4),
        Catch::Matchers::Contains("live CPU direct-step owner"));
    for (const auto& field : rates)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(guarded);
    scheme.set_require_native_dispatch(false);
    const Snapshot recovered;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(recovered, 1e-4, 6);
}

TEST_CASE("live rain SLL binds rain water number and reflectivity before publishing a generation",
          "[physics][milbrandt][triple_cold_rain_sll_internal][direct]")
{
    for (int mutation : {21, 22})
    {
        cold_riming_live::initialize(0, 2, 272.15, 1, true, 1e-3f, 6);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.getter_mutation = mutation;
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 6);
    }
    cold_riming_live::initialize(0, 2, 272.15, 1, true, 1e-3f, 6);
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    std::array<Field3D, 8> rates;
    internal_thermal_live::compute(rates, 1e-4);
    tmv::MicrophysicsPrimaryEndpointCandidate record;
    std::string error;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, record, error));
    std::array<Field3D*, 3> sources{&qr,
        &const_cast<Field3D&>(scheme.internal_triple_number_for_test(0)),
        &const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(0))};
    for (auto* field : sources)
    {
        const float value = (*field)(1, 0, 0);
        (*field)(1, 0, 0) = std::nextafter(value, std::numeric_limits<float>::infinity());
        CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-4, record, error));
        (*field)(1, 0, 0) = value;
    }
    const Snapshot before;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(before, 1e-4, 6);
}

namespace cold_collision_sll_live
{
ColdCollectedSllResult capture(const Snapshot& before, int i, double duration)
{
    ColdCollectedSllResult result;
    std::string error;
    const bool ok = evaluate_exact_triple_cold_collision_sll_cell(frozen_collection_live::parameters(
        global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes),
        cold_riming_live::source(before, i), duration, result, error);
    INFO(error);
    REQUIRE(ok);
    return result;
}
}

TEST_CASE("live collision SLL publishes independently checked rain frozen water and heat once",
          "[physics][milbrandt][triple_cold_collision_sll_internal][direct]")
{
    for (bool hail : {false, true})
    for (double temperature : {272.15, 268.15, 238.15})
    {
        cold_riming_live::initialize(0, 2, temperature, 1, hail, 1e-3f, 7);
        const Snapshot before;
        const auto actual = cold_collision_sll_live::capture(before, 1, 1e-4);
        REQUIRE(actual.combined.producer.rain_collection().channels()[2].mass_per_kg > 0);
        REQUIRE(actual.combined.producer.rain_collection().channels()[3].mass_per_kg > 0);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 7);
    }
}

TEST_CASE("live collision SLL continues with original collisions and higher G H shapes on actual one and four workers",
          "[physics][milbrandt][triple_cold_collision_sll_internal][direct]")
{
#ifdef _OPENMP
    struct ThreadRestore
    {
        int workers = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~ThreadRestore()
        {
            omp_set_num_threads(workers);
            omp_set_dynamic(dynamic);
        }
    } restore;
    omp_set_dynamic(0);
    for (int scenario : {0, 7})
    {
        std::array<std::unique_ptr<Snapshot>, 2> final;
        for (int run = 0; run < 2; ++run)
        {
            const int workers = run == 0 ? 1 : 4;
            omp_set_num_threads(workers);
            int observed = 0;
#pragma omp parallel
            {
#pragma omp single
                observed = omp_get_num_threads();
            }
            REQUIRE(observed == workers);
            cold_riming_live::initialize(scenario, scenario == 7 ? 300 : 2, 272.15, 1, true, .01f, 7);
            for (double duration : {1e-4, 2e-4, 5e-4})
            {
                const Snapshot before;
                const auto actual = cold_collision_sll_live::capture(before, 1, duration);
                REQUIRE(actual.hail_conversion.admitted_mass_transfer_per_kg() > 0);
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                cold_riming_live::check_step(before, duration, 7);
                if (scenario == 7)
                {
                    const Snapshot after;
                    for (std::size_t species : {2u, 3u})
                    {
                        const FrozenCollectionPopulation population{after.fields[species + 9](1, 0, 0),
                            after.moments[species + 2](1, 0, 0), after.moments[species + 7](1, 0, 0)};
                        const long double alpha = milbrandt_frozen_collection_reference::represented_alpha(population, species);
                        CHECK(alpha > 200);
                        CHECK(alpha <= 512);
                    }
                }
                simulation_time += duration;
            }
            final[run] = std::make_unique<Snapshot>();
        }
        check_snapshot(*final[0], *final[1]);
    }
#else
    FAIL("collected SLL worker evidence requires OpenMP");
#endif
}

TEST_CASE("live collision SLL restores every owner after early and late active failures and recovers",
          "[physics][milbrandt][triple_cold_collision_sll_internal][direct]")
{
    for (int mutation : {0, 1, 2})
    {
        cold_riming_live::initialize(0, 2, 268.15, 1, true, 1e-3f, 7);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        if (mutation == 0)
        {
            scheme.getter_mutation = 1;
        }
        else
        {
            scheme.reject = mutation == 1;
            scheme.late_mutation = mutation == 2 ? 2 : 0;
        }
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == (mutation == 0 ? 0 : 1));
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.getter_mutation = 0;
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 7);
    }
}

TEST_CASE("live collision SLL retains exact resource and source capture guards without native fallback",
          "[physics][milbrandt][triple_cold_collision_sll_internal][direct]")
{
    cold_riming_live::initialize(0, 2, 268.15, 1, true, 1e-3f, 7);
    const Snapshot before;
    const auto source = cold_riming_live::source(before, 1);
    const auto journal = cold_collision_sll_live::capture(before, 1, 1e-4);
    ColdCollectedSllResult unpublished;
    unpublished.combined.cloud_mass_increment_per_kg = 42;
    ColdRimingFailure failure;
    std::string error;
    CHECK_FALSE(evaluate_exact_triple_cold_collision_sll_cell(frozen_collection_live::parameters(true), source,
        1e-4, unpublished, error, 14, &failure));
    CHECK(failure == ColdRimingFailure::Capacity);
    CHECK(unpublished.combined.cloud_mass_increment_per_kg == 42);
    auto changed = source;
    changed.initial.frozen[0].number_per_kg = std::nextafter(changed.initial.frozen[0].number_per_kg,
        std::numeric_limits<double>::infinity());
    CHECK_FALSE(apply_exact_triple_cold_collision_sll_cell(frozen_collection_live::parameters(true), changed, 1e6,
        journal.collection, journal.ice_conversion, journal.snow_conversion, journal.hail_conversion,
        unpublished, error, 15, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    CHECK(unpublished.combined.cloud_mass_increment_per_kg == 42);
    check_snapshot(before);
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    scheme.set_require_native_dispatch(true);
    const Snapshot guarded;
    std::array<Field3D, 8> rates;
    for (auto& field : rates)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    REQUIRE_THROWS_WITH(internal_thermal_live::compute(rates, 1e-4),
        Catch::Matchers::Contains("live CPU direct-step owner"));
    for (const auto& field : rates)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(guarded);
    scheme.set_require_native_dispatch(false);
    const Snapshot recovered;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(recovered, 1e-4, 7);
}

TEST_CASE("live collision SLL binds rain water number and reflectivity before publishing a generation",
          "[physics][milbrandt][triple_cold_collision_sll_internal][direct]")
{
    for (int mutation : {21, 22})
    {
        cold_riming_live::initialize(0, 2, 272.15, 1, true, 1e-3f, 7);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.getter_mutation = mutation;
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 7);
    }
    cold_riming_live::initialize(0, 2, 272.15, 1, true, 1e-3f, 7);
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    std::array<Field3D, 8> rates;
    internal_thermal_live::compute(rates, 1e-4);
    tmv::MicrophysicsPrimaryEndpointCandidate record;
    std::string error;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, record, error));
    std::array<Field3D*, 3> sources{&qr,
        &const_cast<Field3D&>(scheme.internal_triple_number_for_test(0)),
        &const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(0))};
    for (auto* field : sources)
    {
        const float value = (*field)(1, 0, 0);
        (*field)(1, 0, 0) = std::nextafter(value, std::numeric_limits<float>::infinity());
        CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-4, record, error));
        (*field)(1, 0, 0) = value;
    }
    const Snapshot before;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(before, 1e-4, 7);
}

namespace cold_wet_growth_sll_live
{
ColdCollectedSllResult capture(const Snapshot& before, int i, double duration)
{
    ColdCollectedSllResult result;
    std::string error;
    const bool ok = evaluate_exact_triple_cold_wet_growth_sll_cell(frozen_collection_live::parameters(
        global_simulation_storage.configuration.microphysics_config.milbrandt.hail_processes),
        cold_riming_live::source(before, i), duration, result, error);
    INFO(error);
    REQUIRE(ok);
    return result;
}
}

TEST_CASE("live wet growth SLL publishes independently checked rain frozen water and heat once",
          "[physics][milbrandt][triple_cold_wet_growth_internal][direct]")
{
    for (bool hail : {false, true})
    for (double temperature : {272.15, 268.15, 238.15})
    {
        cold_riming_live::initialize(0, 2, temperature, 1, hail, 3e-2f, 8);
        const Snapshot before;
        const auto actual = cold_wet_growth_sll_live::capture(before, 1, 1e-4);
        REQUIRE(actual.combined.producer.rain_collection().channels()[2].mass_per_kg > 0);
        REQUIRE(actual.combined.producer.rain_collection().channels()[3].mass_per_kg > 0);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 8);
    }
}

TEST_CASE("live wet growth SLL continues with original collisions and higher G H shapes on actual one and four workers",
          "[physics][milbrandt][triple_cold_wet_growth_internal][direct]")
{
#ifdef _OPENMP
    struct ThreadRestore
    {
        int workers = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~ThreadRestore()
        {
            omp_set_num_threads(workers);
            omp_set_dynamic(dynamic);
        }
    } restore;
    omp_set_dynamic(0);
    for (int scenario : {0, 7})
    {
        std::array<std::unique_ptr<Snapshot>, 2> final;
        for (int run = 0; run < 2; ++run)
        {
            const int workers = run == 0 ? 1 : 4;
            omp_set_num_threads(workers);
            int observed = 0;
#pragma omp parallel
            {
#pragma omp single
                observed = omp_get_num_threads();
            }
            REQUIRE(observed == workers);
            cold_riming_live::initialize(scenario, scenario == 7 ? 300 : 2, 272.15, 1, true, .01f, 8);
            for (double duration : {1e-4, 2e-4, 5e-4})
            {
                const Snapshot before;
                const auto actual = cold_wet_growth_sll_live::capture(before, 1, duration);
                REQUIRE(actual.hail_conversion.admitted_mass_transfer_per_kg() > 0);
                REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), duration));
                cold_riming_live::check_step(before, duration, 8);
                if (scenario == 7)
                {
                    const Snapshot after;
                    for (std::size_t species : {2u, 3u})
                    {
                        const FrozenCollectionPopulation population{after.fields[species + 9](1, 0, 0),
                            after.moments[species + 2](1, 0, 0), after.moments[species + 7](1, 0, 0)};
                        const long double alpha = milbrandt_frozen_collection_reference::represented_alpha(population, species);
                        CHECK(alpha > 200);
                        CHECK(alpha <= 512);
                    }
                }
                simulation_time += duration;
            }
            final[run] = std::make_unique<Snapshot>();
        }
        check_snapshot(*final[0], *final[1]);
    }
#else
    FAIL("collected SLL worker evidence requires OpenMP");
#endif
}

TEST_CASE("live wet growth SLL restores every owner after early and late active failures and recovers",
          "[physics][milbrandt][triple_cold_wet_growth_internal][direct]")
{
    for (int mutation : {0, 1, 2})
    {
        cold_riming_live::initialize(0, 2, 268.15, 1, true, 3e-2f, 8);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        if (mutation == 0)
        {
            scheme.getter_mutation = 1;
        }
        else
        {
            scheme.reject = mutation == 1;
            scheme.late_mutation = mutation == 2 ? 2 : 0;
        }
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        CHECK(getter_attempts == 1);
        CHECK(accepted_attempts == (mutation == 0 ? 0 : 1));
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.getter_mutation = 0;
        recovered.reject = false;
        recovered.late_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 8);
    }
}

TEST_CASE("live wet growth SLL retains exact resource and source capture guards without native fallback",
          "[physics][milbrandt][triple_cold_wet_growth_internal][direct]")
{
    cold_riming_live::initialize(0, 2, 268.15, 1, true, 3e-2f, 8);
    const Snapshot before;
    const auto source = cold_riming_live::source(before, 1);
    const auto journal = cold_wet_growth_sll_live::capture(before, 1, 1e-4);
    ColdCollectedSllResult unpublished;
    unpublished.combined.cloud_mass_increment_per_kg = 42;
    ColdRimingFailure failure;
    std::string error;
    CHECK_FALSE(evaluate_exact_triple_cold_wet_growth_sll_cell(frozen_collection_live::parameters(true), source,
        1e-4, unpublished, error, 15, &failure));
    CHECK(failure == ColdRimingFailure::Capacity);
    CHECK(unpublished.combined.cloud_mass_increment_per_kg == 42);
    auto changed = source;
    changed.initial.frozen[0].number_per_kg = std::nextafter(changed.initial.frozen[0].number_per_kg,
        std::numeric_limits<double>::infinity());
    CHECK_FALSE(apply_exact_triple_cold_wet_growth_sll_cell(frozen_collection_live::parameters(true), changed, 1e6,
        journal.collection, journal.ice_conversion, journal.snow_conversion, journal.hail_conversion,
        unpublished, error, 16, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    CHECK(unpublished.combined.cloud_mass_increment_per_kg == 42);
    check_snapshot(before);
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    scheme.set_require_native_dispatch(true);
    const Snapshot guarded;
    std::array<Field3D, 8> rates;
    for (auto& field : rates)
    {
        field.resize(NR, NTH, NZ, 42);
    }
    REQUIRE_THROWS_WITH(internal_thermal_live::compute(rates, 1e-4),
        Catch::Matchers::Contains("live CPU direct-step owner"));
    for (const auto& field : rates)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == 42);
        }
    }
    CHECK(getter_attempts == 0);
    CHECK(accepted_attempts == 0);
    check_snapshot(guarded);
    scheme.set_require_native_dispatch(false);
    const Snapshot recovered;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(recovered, 1e-4, 8);
}

TEST_CASE("live wet growth SLL binds rain water number and reflectivity before publishing a generation",
          "[physics][milbrandt][triple_cold_wet_growth_internal][direct]")
{
    for (int mutation : {21, 22})
    {
        cold_riming_live::initialize(0, 2, 272.15, 1, true, 3e-2f, 8);
        auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        scheme.getter_mutation = mutation;
        const Snapshot before;
        REQUIRE_THROWS(step_microphysics(global_simulation_state(), 1e-4));
        check_snapshot(before);
        auto& recovered = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
        recovered.getter_mutation = 0;
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 8);
    }
    cold_riming_live::initialize(0, 2, 272.15, 1, true, 3e-2f, 8);
    auto& scheme = dynamic_cast<ObservedMilbrandt&>(*microphysics_scheme);
    std::array<Field3D, 8> rates;
    internal_thermal_live::compute(rates, 1e-4);
    tmv::MicrophysicsPrimaryEndpointCandidate record;
    std::string error;
    REQUIRE(scheme.prepare_primary_endpoint_commit(global_simulation_state(), 1e-4, record, error));
    std::array<Field3D*, 3> sources{&qr,
        &const_cast<Field3D&>(scheme.internal_triple_number_for_test(0)),
        &const_cast<Field3D&>(scheme.internal_triple_reflectivity_for_test(0))};
    for (auto* field : sources)
    {
        const float value = (*field)(1, 0, 0);
        (*field)(1, 0, 0) = std::nextafter(value, std::numeric_limits<float>::infinity());
        CHECK_FALSE(scheme.validate_primary_endpoint_commit(global_simulation_state(), 1e-4, record, error));
        (*field)(1, 0, 0) = value;
    }
    const Snapshot before;
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
    cold_riming_live::check_step(before, 1e-4, 8);
}

TEST_CASE("live wet hail returns captured cloud to a new rain spectrum with only retained freezing heat",
          "[physics][milbrandt][triple_cold_wet_growth_internal][direct]")
{
    for (bool hail : {false, true})
    {
        cold_riming_live::initialize(5, 2, 272.15, 1, hail, .005f, 8, 0, true);
        const Snapshot before;
        const auto captured = cold_wet_growth_sll_live::capture(before, 1, 1e-4);
        const auto& wet = captured.combined.producer.wet_growth();
        REQUIRE(wet.regime() == HailWetGrowthRegime::Wet);
        REQUIRE(wet.shed_mass_per_kg() > 0);
        REQUIRE(wet.source().dry_collection_per_kg[1] == 0);
        REQUIRE(wet.source().wet_collection_per_kg[0] > wet.source().dry_collection_per_kg[0]);
        REQUIRE(captured.combined.unrounded.rain.mass_per_kg > 0);
        REQUIRE(captured.combined.unrounded.rain.number_per_kg > 0);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
        cold_riming_live::check_step(before, 1e-4, 8);
        for (int step = 0; step < 3; ++step)
        {
            const Snapshot continuation;
            REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1e-4));
            cold_riming_live::check_step(continuation, 1e-4, 8);
        }
    }
}
