#pragma once

#include "cold.hpp"
#include "triple_cold_cloud_collection.hpp"
#include "triple_hail_surface_condensation.hpp"
#include "triple_hail_shared_vapor.hpp"
#include "triple_rain_collection.hpp"

namespace tmv::milbrandt_detail
{

struct HailLiquidSurfaceState
{
    HailSurfaceState hail;
    float cloud_number = 0.0f;
    std::array<float, 2> ice_snow_number{};
    std::array<float, 2> ice_snow_reflectivity{};
};

struct HailLiquidSurfaceInput
{
    HailLiquidSurfaceState before;
    bool hail_processes = true;
    double competing_vapor_transfer = 0.0;
    double dry_hail_diffusion = 0.0;
    std::uint64_t generation = 0;
    double duration_s = 0.0;
    bool joint_vapor = false;
    double vertical_velocity_ms = 0.0;
};

struct HailLiquidSurfaceReceipt
{
    double normalized_ventilation = 0.0;
    double vapor_relaxation_number = 0.0;
    double surface_vapor_transfer = 0.0;
    double conductive_heat = 0.0;
    double vapor_heat = 0.0;
    double liquid_sensible_heat = 0.0;
    double collected_frozen_sensible_heat = 0.0;
    double collected_frozen_mass = 0.0;
    double dry_collected_frozen_mass = 0.0;
    double cloud_collected_mass = 0.0;
    double cloud_collected_number = 0.0;
    double rain_collected_mass = 0.0;
    double rain_collected_number = 0.0;
    double rain_reflectivity_removed = 0.0;
    double dry_liquid_mass = 0.0;
    double dry_full_freezing_heat = 0.0;
    double frozen_liquid_mass = 0.0;
    double melted_hail_mass = 0.0;
    double melted_hail_number = 0.0;
    double hail_reflectivity_removed = 0.0;
    double melt_rain_reflectivity = 0.0;
    double shed_liquid_mass = 0.0;
    double shed_rain_number = 0.0;
    double shed_rain_reflectivity = 0.0;
    double cloud_number_after = 0.0;
    double rain_number_after = 0.0;
    double rain_reflectivity_after = 0.0;
    double hail_number_after = 0.0;
    double hail_reflectivity_after = 0.0;
    FrozenCollectionIncrements frozen_collection;
    std::array<double, 2> ice_snow_number_after{};
    std::array<double, 2> ice_snow_reflectivity_after{};
    HailSharedVaporAdmission shared_vapor;
    ColdCloudEosEquilibrium cloud_equilibrium;
    double snow_deposition_reflectivity = 0.0;
    double stored_cloud_saturation_residual = 0.0;
};

class HailLiquidSurfaceTransaction
{
public:
    SourceEosEnergyTransaction::Stage stage() const noexcept { return energy_.stage(); }
    HailLiquidSurfaceInput input() const noexcept { return input_; }
    HailLiquidSurfaceReceipt receipt() const noexcept { return receipt_; }
    SourceEosEnergyInput energy_input() const noexcept { return energy_.input(); }
    SourceEosEnergyReceipt energy_receipt() const noexcept { return energy_.receipt(); }
    HailLiquidSurfaceState source_endpoint() const noexcept { return source_endpoint_; }
    HailLiquidSurfaceState eos_endpoint() const noexcept { return eos_endpoint_; }
    ColdCloudCollectionProducer cloud_collection() const { return cloud_; }
    ColdCloudCollectionProducer dry_cloud_collection() const { return dry_cloud_; }
    RainCollectionProducer rain_collection() const { return rain_; }
    FrozenCollectionProducer frozen_collection() const { return frozen_; }
    FrozenCollectionProducer dry_frozen_collection() const { return dry_frozen_; }
    ColdCloudDiffusionProducer snow_diffusion() const { return snow_diffusion_; }

    bool prepare(const HailLiquidSurfaceInput& input, std::string& error)
    {
        using namespace physical_constants;
        const auto reject = [&](const char* message) { error = message; return false; };
        if (stage() != SourceEosEnergyTransaction::Stage::Empty)
            return reject("hail liquid surface transaction already owns a sequence");
        if (input.competing_vapor_transfer != 0.0 || input.dry_hail_diffusion != 0.0)
            return reject("hail liquid surface shared vapor composition is not owned");
        if (input.joint_vapor && !std::isfinite(input.vertical_velocity_ms))
            return reject("hail shared vapor cloud activation velocity is invalid");
        const auto& before = input.before;
        const auto& source = before.hail;
        if (source.thermal.water[5] != 0.0f)
            return reject("hail liquid surface has an unowned graupel collector");
        if (!std::isfinite(before.cloud_number) || before.cloud_number < 0.0f ||
            (source.thermal.water[1] > 0.0f && before.cloud_number <= 0.0f))
            return reject("hail liquid surface cloud number is invalid");
        SourceEosEnergyInput energy_input;
        energy_input.before = source.thermal;
        std::copy(source.thermal.water.begin(), source.thermal.water.end(), energy_input.accepted_water.begin());
        energy_input.generation = input.generation;
        energy_input.duration_s = input.duration_s;
        SourceEosEnergyTransaction context;
        if (!context.prepare(energy_input, error)) return false;
        const double temperature = context.receipt().temperature_before;
        HailLiquidSurfaceTransaction candidate;
        candidate.input_ = input;
        candidate.source_endpoint_ = before;
        auto& r = candidate.receipt_;
        if (!budget(donors(before), temperature, source.thermal.pressure, input.duration_s, input.hail_processes,
                input.joint_vapor, r, candidate.cloud_, candidate.dry_cloud_, candidate.rain_, candidate.frozen_,
                candidate.dry_frozen_, candidate.snow_diffusion_, error)) return false;
        if (!evaluate_exact_frozen_collection_increments(parameters(input.hail_processes), candidate.frozen_.source(),
                input.duration_s, candidate.frozen_, r.frozen_collection, error, 5, nullptr,
                ColdSpectrumDomain::GraupelAndHail512, true)) return false;
        for (std::size_t i = 0; i < 2; ++i)
        {
            const double q = source.thermal.water[i + 3], n = before.ice_snow_number[i], z = before.ice_snow_reflectivity[i];
            const auto species = i == 0 ? TripleMomentSpecies::Ice : TripleMomentSpecies::Snow;
            const double inbound = i == 1 ? candidate.frozen_.channels()[0].admitted_mass_increment_per_kg : 0.0;
            const double gross_mass = r.frozen_collection.mass_increment_per_kg[i] - inbound;
            const double number = r.frozen_collection.number_increment_per_kg[i];
            const double power = 6.0 / current_nimbus_triple_sedimentation_parameters(species).mass_law.exponent;
            const double gross_z = q > 0.0 ? z * (power * gross_mass / q + (1.0 - power) * number / n) : 0.0;
            if (!population(q + gross_mass, n + number, z + gross_z, species, error, false))
                return reject(i == 0 ? "hail liquid surface gross original ice donor is inadmissible before returns" :
                    "hail liquid surface gross original snow donor is inadmissible before returns");
            energy_input.accepted_water[i + 3] += r.frozen_collection.mass_increment_per_kg[i];
            r.ice_snow_number_after[i] = n + number;
            r.ice_snow_reflectivity_after[i] = z + r.frozen_collection.reflectivity_increment_per_kg[i];
        }
        const double hail_mass = source.thermal.water[6];
        if (hail_mass > 0.0)
        {
            const double tc = temperature - freezing_temperature_k;
            const double heat = r.conductive_heat + r.vapor_heat + r.liquid_sensible_heat + r.collected_frozen_sensible_heat;
            if (heat >= 0.0) r.melted_hail_mass = heat / (latent_heat_fusion_jkg - specific_heat_ice_jkgk * tc);
            else r.frozen_liquid_mass = -heat / latent_heat_fusion_jkg;
            if ((heat > 0.0 && r.melted_hail_mass == 0.0) || (heat < 0.0 && r.frozen_liquid_mass == 0.0))
                return reject("hail liquid surface phase transfer is unrepresentable");
            if (!std::isfinite(r.melted_hail_mass) || r.melted_hail_mass > hail_mass)
                return reject("hail liquid surface exhausts original hail");
            r.shed_liquid_mass = r.cloud_collected_mass + r.rain_collected_mass +
                r.surface_vapor_transfer - r.frozen_liquid_mass;
            if (!(r.shed_liquid_mass > 0.0))
                return reject("hail liquid surface requires an unowned wet/dry transition");
            const double qr = source.thermal.water[2];
            if (qr > 0.0)
                r.rain_reflectivity_removed = source.rain_reflectivity *
                    (2.0 * r.rain_collected_mass / qr - r.rain_collected_number / source.rain_number);
            if (!population(qr - r.rain_collected_mass, source.rain_number - r.rain_collected_number,
                    source.rain_reflectivity - r.rain_reflectivity_removed, TripleMomentSpecies::Rain, error, false))
                return reject("hail liquid surface gross original-rain donor is inadmissible before returns");
            const double melt_fraction = r.melted_hail_mass / hail_mass;
            r.melted_hail_number = source.hail_number * melt_fraction;
            r.hail_reflectivity_removed = source.hail_reflectivity * melt_fraction;
            const double ch = current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Hail).mass_law.coefficient;
            const double cr = current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Rain).mass_law.coefficient;
            r.melt_rain_reflectivity = (ch / cr) * (ch / cr) * r.hail_reflectivity_removed;
            const double drop_mass = cr * 1e-3 * 1e-3 * 1e-3;
            r.shed_rain_number = r.shed_liquid_mass / drop_mass;
            r.shed_rain_reflectivity = 5.6 * r.shed_liquid_mass * drop_mass / (cr * cr);
            if (!(r.shed_rain_number > 0.0) || !(r.shed_rain_reflectivity > 0.0))
                return reject("hail liquid surface shed moments are unrepresentable");
            r.cloud_number_after = before.cloud_number - r.cloud_collected_number;
            r.rain_number_after = source.rain_number - r.rain_collected_number + r.shed_rain_number + r.melted_hail_number;
            r.rain_reflectivity_after = source.rain_reflectivity - r.rain_reflectivity_removed +
                r.shed_rain_reflectivity + r.melt_rain_reflectivity;
            r.hail_number_after = source.hail_number * (1.0 - melt_fraction);
            r.hail_reflectivity_after = source.hail_reflectivity * (1.0 - melt_fraction) +
                2.0 * source.hail_reflectivity * r.frozen_liquid_mass / hail_mass +
                r.frozen_collection.reflectivity_increment_per_kg[3];
            energy_input.accepted_water[0] -= r.surface_vapor_transfer;
            energy_input.accepted_water[1] -= r.cloud_collected_mass;
            energy_input.accepted_water[2] += -r.rain_collected_mass + r.shed_liquid_mass + r.melted_hail_mass;
            energy_input.accepted_water[6] += r.frozen_liquid_mass - r.melted_hail_mass + r.collected_frozen_mass;
        }
        else
        {
            r.cloud_number_after = before.cloud_number;
            r.rain_number_after = source.rain_number;
            r.rain_reflectivity_after = source.rain_reflectivity;
        }
        if (input.joint_vapor)
        {
            const double deposition = r.shared_vapor.snow_transfer;
            energy_input.accepted_water[0] -= deposition;
            energy_input.accepted_water[4] += deposition;
            if (deposition > 0.0)
            {
                const double power = 6.0 / current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Snow).mass_law.exponent;
                r.snow_deposition_reflectivity = power * before.ice_snow_reflectivity[1] * deposition / source.thermal.water[4];
                r.ice_snow_reflectivity_after[1] += r.snow_deposition_reflectivity;
            }
            if (!solve_cold_cloud_eos_equilibrium(source.thermal, energy_input.accepted_water,
                    context.receipt().temperature_eos_reference, energy_input.accepted_water[1], r.cloud_number_after,
                    r.shared_vapor.original_vapor_remaining, input.vertical_velocity_ms, r.cloud_equilibrium, error)) return false;
            energy_input.accepted_water[0] -= r.cloud_equilibrium.cloud_transfer;
            energy_input.accepted_water[1] += r.cloud_equilibrium.cloud_transfer;
            r.cloud_number_after = r.cloud_equilibrium.cloud_number;
        }
        if (!candidate.energy_.prepare(energy_input, error)) return false;
        candidate.source_endpoint_.hail.thermal = candidate.energy_.source_endpoint();
        candidate.source_endpoint_.cloud_number = static_cast<float>(r.cloud_number_after);
        candidate.source_endpoint_.hail.rain_number = static_cast<float>(r.rain_number_after);
        candidate.source_endpoint_.hail.rain_reflectivity = static_cast<float>(r.rain_reflectivity_after);
        candidate.source_endpoint_.hail.hail_number = static_cast<float>(r.hail_number_after);
        candidate.source_endpoint_.hail.hail_reflectivity = static_cast<float>(r.hail_reflectivity_after);
        for (std::size_t i = 0; i < 2; ++i)
        {
            candidate.source_endpoint_.ice_snow_number[i] = static_cast<float>(r.ice_snow_number_after[i]);
            candidate.source_endpoint_.ice_snow_reflectivity[i] = static_cast<float>(r.ice_snow_reflectivity_after[i]);
        }
        candidate.eos_endpoint_ = candidate.source_endpoint_;
        candidate.eos_endpoint_.hail.thermal = candidate.energy_.eos_endpoint();
        if (input.joint_vapor)
        {
            r.stored_cloud_saturation_residual = candidate.eos_endpoint_.hail.thermal.water[0] -
                thermodynamics::saturation_mixing_ratio_water(candidate.energy_.receipt().temperature_eos_stored,
                    candidate.eos_endpoint_.hail.thermal.pressure);
            if (!std::isfinite(r.stored_cloud_saturation_residual))
                return reject("hail shared vapor stored saturation residual is unrepresentable");
        }
        for (bool stored : {false, true})
        {
            const auto& end = candidate.source_endpoint_;
            for (std::size_t i = 0; i < 2; ++i)
                if (!population(stored ? end.hail.thermal.water[i + 3] : energy_input.accepted_water[i + 3],
                        stored ? end.ice_snow_number[i] : r.ice_snow_number_after[i],
                        stored ? end.ice_snow_reflectivity[i] : r.ice_snow_reflectivity_after[i],
                        i == 0 ? TripleMomentSpecies::Ice : TripleMomentSpecies::Snow, error))
                    return reject("hail liquid surface ice/snow endpoint moments are inadmissible");
            if (!cloud_population(stored ? end.hail.thermal.water[1] : energy_input.accepted_water[1],
                    stored ? end.cloud_number : r.cloud_number_after) ||
                !population(stored ? end.hail.thermal.water[2] : energy_input.accepted_water[2],
                    stored ? end.hail.rain_number : r.rain_number_after,
                    stored ? end.hail.rain_reflectivity : r.rain_reflectivity_after, TripleMomentSpecies::Rain, error) ||
                !population(stored ? end.hail.thermal.water[6] : energy_input.accepted_water[6],
                    stored ? end.hail.hail_number : r.hail_number_after,
                    stored ? end.hail.hail_reflectivity : r.hail_reflectivity_after, TripleMomentSpecies::Hail, error))
                return reject("hail liquid surface endpoint moments are inadmissible");
        }
        const auto thermal = candidate.energy_.receipt();
        const double staged = source_eos_energy_detail::temperature(source.thermal.theta + thermal.theta_increment, source.thermal.pressure);
        const double staged_stored = source_eos_energy_detail::temperature(candidate.source_endpoint_.hail.thermal.theta, source.thermal.pressure);
        const std::array<std::array<double, 2>, 4> conditions{{{staged, source.thermal.pressure},
            {staged_stored, source.thermal.pressure},
            {thermal.temperature_eos_target, source.thermal.density * gas_constant_dry_air_jkgk * thermal.temperature_eos_target},
            {thermal.temperature_eos_stored, candidate.eos_endpoint_.hail.thermal.pressure}}};
        for (const auto& state : conditions)
            if (!(state[0] > freezing_temperature_k - 40.0 && state[0] < freezing_temperature_k))
                return reject("hail liquid surface interval crosses its thermal domain");
        if (energy_input.accepted_water[6] > 0.0 || (input.joint_vapor && energy_input.accepted_water[4] > 0.0))
        {
            Donors ideal{source.thermal.density, energy_input.accepted_water[0], energy_input.accepted_water[1],
                r.cloud_number_after, {energy_input.accepted_water[2], r.rain_number_after, r.rain_reflectivity_after},
                {energy_input.accepted_water[6], r.hail_number_after, r.hail_reflectivity_after}};
            for (std::size_t i = 0; i < 2; ++i)
                ideal.ice_snow[i] = {energy_input.accepted_water[i + 3], r.ice_snow_number_after[i], r.ice_snow_reflectivity_after[i]};
            const std::array<Donors, 4> endpoints{ideal, donors(candidate.source_endpoint_), ideal, donors(candidate.eos_endpoint_)};
            for (std::size_t i = 0; i < endpoints.size(); ++i)
            {
                HailLiquidSurfaceReceipt endpoint_budget;
                ColdCloudCollectionProducer cloud, dry;
                RainCollectionProducer rain;
                FrozenCollectionProducer frozen, dry_frozen;
                ColdCloudDiffusionProducer snow_diffusion;
                if (!budget(endpoints[i], conditions[i][0], conditions[i][1], input.duration_s,
                        input.hail_processes, input.joint_vapor, endpoint_budget, cloud, dry, rain, frozen,
                        dry_frozen, snow_diffusion, error))
                    return reject("hail liquid surface ideal or stored endpoint requires an unowned wet/dry transition");
            }
        }
        *this = candidate;
        error.clear();
        return true;
    }

    bool accept_source(const HailLiquidSurfaceState& before, const HailLiquidSurfaceState& after,
                       std::uint64_t generation, std::string& error)
    {
        if (!same_moments(before, input_.before) || !same_moments(after, source_endpoint_))
        { error = "hail liquid source moments differ from their captured donor or endpoint"; return false; }
        return energy_.accept_source(before.hail.thermal, after.hail.thermal, generation, error);
    }

    bool accept_eos(const HailLiquidSurfaceState& before, const HailLiquidSurfaceState& after,
                    std::uint64_t generation, std::string& error)
    {
        if (!same_moments(before, source_endpoint_) || !same_moments(after, eos_endpoint_))
        { error = "hail liquid moments changed before their EOS endpoint"; return false; }
        return energy_.accept_eos(before.hail.thermal, after.hail.thermal, generation, error);
    }

private:
    struct Donors
    {
        double density, vapor, cloud_mass, cloud_number;
        FrozenCollectionPopulation rain, hail;
        std::array<FrozenCollectionPopulation, 2> ice_snow{};
    };

    static Donors donors(const HailLiquidSurfaceState& source)
    {
        const auto& h = source.hail;
        return {h.thermal.density, h.thermal.water[0], h.thermal.water[1], source.cloud_number,
            {h.thermal.water[2], h.rain_number, h.rain_reflectivity},
            {h.thermal.water[6], h.hail_number, h.hail_reflectivity},
            {{{h.thermal.water[3], source.ice_snow_number[0], source.ice_snow_reflectivity[0]},
              {h.thermal.water[4], source.ice_snow_number[1], source.ice_snow_reflectivity[1]}}}};
    }

    static ColdParameters parameters(bool hail)
    {
        const auto& p = default_cold_parameters;
        return {p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],
            p[13],p[14],p[15],p[16],p[17],p[18],p[19],p[20],p[21],p[22],p[23],p[24],hail};
    }

    static bool compose_vapor(const Donors& state, double temperature, double pressure, double dt,
                              bool hail_processes, HailLiquidSurfaceReceipt& r,
                              ColdCloudDiffusionProducer& snow_diffusion, std::string& error)
    {
        ColdCloudDiffusionCell snow{temperature, pressure, state.density, state.vapor, {}};
        snow.initial.frozen[0] = state.ice_snow[1];
        if (!produce_exact_cold_cloud_diffusion_admission(parameters(hail_processes), snow, dt, snow_diffusion,
                error, 3, nullptr, ColdSpectrumDomain::GraupelAndHail512) ||
            !admit_hail_shared_vapor(state.vapor, r.surface_vapor_transfer,
                snow_diffusion.channels()[0].raw_mass_increment_per_kg, r.shared_vapor, error)) return false;
        r.surface_vapor_transfer = r.shared_vapor.surface_transfer;
        return true;
    }

    static bool budget(const Donors& state, double temperature, double pressure, double dt, bool hail_processes, bool joint_vapor,
                       HailLiquidSurfaceReceipt& r, ColdCloudCollectionProducer& cloud, ColdCloudCollectionProducer& dry,
                       RainCollectionProducer& rain, FrozenCollectionProducer& frozen,
                       FrozenCollectionProducer& dry_frozen, ColdCloudDiffusionProducer& snow_diffusion, std::string& error)
    {
        using namespace physical_constants;
        const double rho = state.density, tc = temperature - freezing_temperature_k;
        if (!(tc > -40.0 && tc < 0.0)) { error = "hail liquid surface source is outside the cold domain"; return false; }
        const auto original_hail = state.hail;
        const auto p = parameters(hail_processes);
        constexpr auto domain = ColdSpectrumDomain::GraupelAndHail512;
        ColdCloudCollectionCell cloud_source{temperature, pressure, rho, {}};
        cloud_source.initial.cloud_mass_per_kg = state.cloud_mass;
        cloud_source.initial.cloud_number_per_kg = state.cloud_number;
        cloud_source.initial.frozen[2] = original_hail;
        RainCollectionCell rain_source{{temperature, pressure, rho, {}}, state.rain};
        rain_source.frozen.initial.frozen[3] = original_hail;
        FrozenCollectionCell frozen_source{temperature, pressure, rho, {}};
        frozen_source.initial.frozen[0] = state.ice_snow[0];
        frozen_source.initial.frozen[1] = state.ice_snow[1];
        frozen_source.initial.frozen[3] = original_hail;
        if (!produce_exact_cold_cloud_collection_admission(p, cloud_source, dt, cloud, error, 3, nullptr, domain, true) ||
            !produce_exact_cold_cloud_collection_admission(p, cloud_source, dt, dry, error, 3, nullptr, domain, false) ||
            !produce_exact_rain_collection_admission(p, rain_source, dt, rain, error, 4, nullptr, domain) ||
            !produce_exact_frozen_collection_admission(p, frozen_source, dt, frozen, error, 5, nullptr, domain, true) ||
            !produce_exact_frozen_collection_admission(p, frozen_source, dt, dry_frozen, error, 5, nullptr, domain, false)) return false;
        if (dry_frozen.channels()[3].admitted_mass_increment_per_kg > state.ice_snow[1].mass_per_kg)
        { error = "hail liquid dry-onset witness spends newly transferred snow"; return false; }
        r.cloud_collected_mass = cloud.channels()[2].admitted_mass_increment_per_kg;
        r.cloud_collected_number = cloud.channels()[2].admitted_number_increment_per_kg;
        r.rain_collected_mass = rain.channels()[3].mass_per_kg;
        r.rain_collected_number = rain.channels()[3].number_per_kg;
        r.dry_liquid_mass = dry.channels()[2].admitted_mass_increment_per_kg + r.rain_collected_mass;
        r.collected_frozen_mass = frozen.channels()[2].admitted_mass_increment_per_kg + frozen.channels()[3].admitted_mass_increment_per_kg;
        r.dry_collected_frozen_mass = dry_frozen.channels()[2].admitted_mass_increment_per_kg + dry_frozen.channels()[3].admitted_mass_increment_per_kg;
        if (original_hail.mass_per_kg == 0.0)
            return !joint_vapor || compose_vapor(state, temperature, pressure, dt, hail_processes, r, snow_diffusion, error);
        double alpha = 0.0, lambda = 0.0;
        const auto hail = current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Hail);
        if (!diagnose_triple_gamma_shape(original_hail.mass_per_kg, original_hail.number_per_kg,
                original_hail.reflectivity_per_kg, hail.mass_law, alpha, lambda, error)) return false;
        const auto air = diagnose_air_transport(temperature, pressure, rho);
        const double order = 1.5 + .5 * hail.fall_exponent;
        const double ventilated = std::exp(std::lgamma(alpha + order + 1.0) - std::lgamma(alpha + 1.0) - order * std::log(lambda));
        r.normalized_ventilation = .78 * (alpha + 1.0) / lambda + .30 * air.schmidt_one_third *
            std::sqrt(air.density_factor * hail.fall_coefficient / air.kinematic_viscosity) * ventilated;
        const double conductance = dt * 2.0 * pi * original_hail.number_per_kg * r.normalized_ventilation;
        r.vapor_relaxation_number = conductance * rho * air.vapor_diffusivity;
        const double vapor_excess = state.vapor - thermodynamics::saturation_mixing_ratio_water(freezing_temperature_k, pressure);
        r.surface_vapor_transfer = r.vapor_relaxation_number * vapor_excess;
        if (joint_vapor && !compose_vapor(state, temperature, pressure, dt, hail_processes, r, snow_diffusion, error)) return false;
        r.conductive_heat = conductance * air.thermal_conductivity * tc;
        r.vapor_heat = latent_heat_vaporization_jkg * r.surface_vapor_transfer;
        const double liquid = r.cloud_collected_mass + r.rain_collected_mass;
        r.liquid_sensible_heat = specific_heat_liquid_water_jkgk * tc * liquid;
        r.collected_frozen_sensible_heat = specific_heat_ice_jkgk * tc * r.collected_frozen_mass;
        const double available = liquid + r.surface_vapor_transfer;
        const double heat = r.conductive_heat + r.vapor_heat + r.liquid_sensible_heat + r.collected_frozen_sensible_heat;
        const double freezing = heat < 0.0 ? -heat / latent_heat_fusion_jkg : 0.0;
        r.dry_full_freezing_heat = r.conductive_heat + r.vapor_heat + specific_heat_liquid_water_jkgk * tc * r.dry_liquid_mass +
            specific_heat_ice_jkgk * tc * r.dry_collected_frozen_mass +
            latent_heat_fusion_jkg * (r.dry_liquid_mass + r.surface_vapor_transfer);
        if (!std::isfinite(conductance) || conductance <= 0.0 || r.conductive_heat == 0.0 ||
            !std::isfinite(r.vapor_relaxation_number) || r.vapor_relaxation_number > 1.0 ||
            (vapor_excess != 0.0 && r.surface_vapor_transfer == 0.0) ||
            !std::isfinite(heat) || !std::isfinite(available) || !(available > freezing) ||
            !(r.dry_liquid_mass + r.surface_vapor_transfer > 0.0) || !(r.dry_full_freezing_heat > 0.0))
        { error = "hail liquid surface requires a strictly wet admitted budget and owned onset"; return false; }
        return true;
    }

    static bool cloud_population(double q, double n)
    {
        return std::isfinite(q) && q >= 0.0 && std::isfinite(n) && n >= 0.0 &&
            std::isfinite(static_cast<float>(n)) && (q == 0.0 || n > 0.0);
    }

    static bool population(double q, double n, double z, TripleMomentSpecies species, std::string& error, bool endpoint = true)
    {
        if (q == 0.0 && n == 0.0 && z == 0.0) return true;
        double alpha = 0.0, lambda = 0.0;
        return diagnose_triple_gamma_shape(q, n, z, current_nimbus_triple_sedimentation_parameters(species).mass_law,
            alpha, lambda, error) && (!endpoint || (alpha <= cold_spectrum_shape_limit(ColdSpectrumDomain::GraupelAndHail512, species) &&
            std::isfinite(static_cast<float>(q)) && std::isfinite(static_cast<float>(n)) && std::isfinite(static_cast<float>(z)) &&
            static_cast<float>(q) > 0.0f && static_cast<float>(n) > 0.0f && static_cast<float>(z) > 0.0f));
    }

    static bool same_moments(const HailLiquidSurfaceState& a, const HailLiquidSurfaceState& b)
    {
        const std::array<float, 9> x{a.cloud_number, a.hail.rain_number, a.hail.rain_reflectivity, a.hail.hail_number, a.hail.hail_reflectivity,
            a.ice_snow_number[0], a.ice_snow_number[1], a.ice_snow_reflectivity[0], a.ice_snow_reflectivity[1]};
        const std::array<float, 9> y{b.cloud_number, b.hail.rain_number, b.hail.rain_reflectivity, b.hail.hail_number, b.hail.hail_reflectivity,
            b.ice_snow_number[0], b.ice_snow_number[1], b.ice_snow_reflectivity[0], b.ice_snow_reflectivity[1]};
        return std::memcmp(x.data(), y.data(), sizeof(x)) == 0;
    }

    HailLiquidSurfaceInput input_{};
    HailLiquidSurfaceReceipt receipt_{};
    HailLiquidSurfaceState source_endpoint_{}, eos_endpoint_{};
    ColdCloudCollectionProducer cloud_, dry_cloud_;
    RainCollectionProducer rain_;
    FrozenCollectionProducer frozen_, dry_frozen_;
    ColdCloudDiffusionProducer snow_diffusion_;
    SourceEosEnergyTransaction energy_;
};

} // namespace tmv::milbrandt_detail
