#pragma once

#include "distribution.hpp"
#include "triple_cold_spectrum_domain.hpp"
#include "triple_source_eos_energy.hpp"

namespace tmv::milbrandt_detail
{

struct HailSurfaceState
{
    SourceEosEnergyState thermal;
    float rain_number = 0.0f;
    float rain_reflectivity = 0.0f;
    float hail_number = 0.0f;
    float hail_reflectivity = 0.0f;
};

struct HailSurfaceCondensationInput
{
    HailSurfaceState before;
    std::array<double, 4> admitted_collection{};
    double dry_hail_diffusion = 0.0;
    std::uint64_t generation = 0;
    double duration_s = 0.0;
};

struct HailSurfaceCondensationReceipt
{
    double normalized_ventilation = 0.0;
    double vapor_relaxation_number = 0.0;
    double condensed_mass = 0.0;
    double conductive_heat = 0.0;
    double condensation_heat = 0.0;
    double melted_mass = 0.0;
    double melted_number = 0.0;
    double hail_reflectivity_removed = 0.0;
    double melted_rain_reflectivity = 0.0;
    double material_warming_heat = 0.0;
    double melting_heat = 0.0;
    double condensate_rain_number = 0.0;
    double condensate_rain_reflectivity = 0.0;
    double rain_number_after = 0.0;
    double rain_reflectivity_after = 0.0;
    double hail_number_after = 0.0;
    double hail_reflectivity_after = 0.0;
};

class HailSurfaceCondensationTransaction
{
public:
    SourceEosEnergyTransaction::Stage stage() const noexcept { return energy_.stage(); }
    HailSurfaceCondensationInput input() const noexcept { return input_; }
    HailSurfaceCondensationReceipt receipt() const noexcept { return receipt_; }
    SourceEosEnergyReceipt energy_receipt() const noexcept { return energy_.receipt(); }
    SourceEosEnergyInput energy_input() const noexcept { return energy_.input(); }
    HailSurfaceState source_endpoint() const noexcept { return source_endpoint_; }
    HailSurfaceState eos_endpoint() const noexcept { return eos_endpoint_; }

    bool prepare(const HailSurfaceCondensationInput& input, std::string& error)
    {
        using namespace physical_constants;
        const auto reject = [&](const char* message) { error = message; return false; };
        if (stage() != SourceEosEnergyTransaction::Stage::Empty)
            return reject("hail surface transaction already owns a sequence");
        for (double value : input.admitted_collection)
            if (value != 0.0) return reject("hail surface collected-material coupling is not owned");
        if (input.dry_hail_diffusion != 0.0)
            return reject("hail surface cannot coexist with dry hail diffusion");
        const auto& source = input.before;
        SourceEosEnergyInput energy_input;
        energy_input.before = source.thermal;
        std::copy(source.thermal.water.begin(), source.thermal.water.end(), energy_input.accepted_water.begin());
        energy_input.generation = input.generation;
        energy_input.duration_s = input.duration_s;
        SourceEosEnergyTransaction validated;
        if (!validated.prepare(energy_input, error)) return false;
        const double temperature = validated.receipt().temperature_before;
        const double tc = temperature - freezing_temperature_k;
        if (!(tc > -40.0 && tc < 0.0))
            return reject("hail surface condensation requires the bounded cold thermal domain");
        double alpha = 0.0, lambda = 0.0, unused_alpha = 0.0, unused_lambda = 0.0;
        if (!population(source.thermal.water[2], source.rain_number, source.rain_reflectivity,
                TripleMomentSpecies::Rain, unused_alpha, unused_lambda, error) ||
            !population(source.thermal.water[6], source.hail_number, source.hail_reflectivity,
                TripleMomentSpecies::Hail, alpha, lambda, error)) return false;
        HailSurfaceCondensationTransaction candidate;
        candidate.input_ = input;
        candidate.source_endpoint_ = source;
        auto& receipt = candidate.receipt_;
        const double hail_mass = source.thermal.water[6];
        if (hail_mass > 0.0)
        {
            const auto hail = current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Hail);
            const auto rain = current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Rain);
            const auto air = diagnose_air_transport(temperature, source.thermal.pressure, source.thermal.density);
            const double first = (alpha + 1.0) / lambda;
            const double order = 1.5 + .5 * hail.fall_exponent;
            const double ventilated = std::exp(std::lgamma(alpha + order + 1.0) - std::lgamma(alpha + 1.0) - order * std::log(lambda));
            receipt.normalized_ventilation = .78 * first + .30 * air.schmidt_one_third *
                std::sqrt(air.density_factor * hail.fall_coefficient / air.kinematic_viscosity) * ventilated;
            const double conductance = input.duration_s * 2.0 * pi * source.hail_number * receipt.normalized_ventilation;
            const double saturation = thermodynamics::saturation_mixing_ratio_water(freezing_temperature_k, source.thermal.pressure);
            receipt.vapor_relaxation_number = conductance * source.thermal.density * air.vapor_diffusivity;
            receipt.condensed_mass = receipt.vapor_relaxation_number * (source.thermal.water[0] - saturation);
            receipt.conductive_heat = conductance * air.thermal_conductivity * tc;
            receipt.condensation_heat = latent_heat_vaporization_jkg * receipt.condensed_mass;
            const double heat = receipt.conductive_heat + receipt.condensation_heat;
            if (!std::isfinite(receipt.normalized_ventilation) || receipt.normalized_ventilation <= 0.0 ||
                !std::isfinite(conductance) || conductance <= 0.0 || receipt.conductive_heat == 0.0 ||
                !std::isfinite(receipt.vapor_relaxation_number) || receipt.vapor_relaxation_number > 1.0 ||
                !std::isfinite(receipt.condensed_mass) || receipt.condensed_mass < 0.0 || !std::isfinite(heat) || heat < 0.0)
                return reject("hail surface requires a condensing interface with nonnegative melting heat");
            if (source.thermal.water[0] > saturation && receipt.condensed_mass == 0.0)
                return reject("hail surface condensation is unrepresentable");
            // Converted hail enters at ambient T; a zero-storage 0 C interface must warm it before melting.
            receipt.melted_mass = heat / (latent_heat_fusion_jkg - specific_heat_ice_jkgk * tc);
            if (heat > 0.0 && receipt.melted_mass == 0.0)
                return reject("hail surface melting is unrepresentable");
            if (receipt.condensed_mass > source.thermal.water[0] || receipt.melted_mass > hail_mass)
                return reject("hail surface interval exhausts an original vapor or hail donor");
            const double fraction = receipt.melted_mass / hail_mass;
            receipt.melted_number = source.hail_number * fraction;
            receipt.hail_reflectivity_removed = source.hail_reflectivity * fraction;
            const double density_ratio = hail.mass_law.coefficient / rain.mass_law.coefficient;
            receipt.melted_rain_reflectivity = density_ratio * density_ratio * receipt.hail_reflectivity_removed;
            receipt.material_warming_heat = -specific_heat_ice_jkgk * tc * receipt.melted_mass;
            receipt.melting_heat = latent_heat_fusion_jkg * receipt.melted_mass;
            constexpr double diameter = 1e-3;
            const double drop_mass = rain.mass_law.coefficient * diameter * diameter * diameter;
            receipt.condensate_rain_number = receipt.condensed_mass / drop_mass;
            receipt.condensate_rain_reflectivity = 5.6 * receipt.condensed_mass * drop_mass /
                (rain.mass_law.coefficient * rain.mass_law.coefficient);
            energy_input.accepted_water[0] -= receipt.condensed_mass;
            energy_input.accepted_water[2] += receipt.condensed_mass + receipt.melted_mass;
            energy_input.accepted_water[6] -= receipt.melted_mass;
            receipt.rain_number_after = source.rain_number + receipt.melted_number + receipt.condensate_rain_number;
            receipt.rain_reflectivity_after = source.rain_reflectivity + receipt.melted_rain_reflectivity + receipt.condensate_rain_reflectivity;
            receipt.hail_number_after = source.hail_number * (1.0 - fraction);
            receipt.hail_reflectivity_after = source.hail_reflectivity * (1.0 - fraction);
        }
        else
        {
            receipt.rain_number_after = source.rain_number;
            receipt.rain_reflectivity_after = source.rain_reflectivity;
        }
        if (!candidate.energy_.prepare(energy_input, error)) return false;
        candidate.source_endpoint_.thermal = candidate.energy_.source_endpoint();
        candidate.source_endpoint_.rain_number = static_cast<float>(receipt.rain_number_after);
        candidate.source_endpoint_.rain_reflectivity = static_cast<float>(receipt.rain_reflectivity_after);
        candidate.source_endpoint_.hail_number = static_cast<float>(receipt.hail_number_after);
        candidate.source_endpoint_.hail_reflectivity = static_cast<float>(receipt.hail_reflectivity_after);
        candidate.eos_endpoint_ = candidate.source_endpoint_;
        candidate.eos_endpoint_.thermal = candidate.energy_.eos_endpoint();
        const auto thermal = candidate.energy_.receipt();
        const double staged_temperature = source_eos_energy_detail::temperature(
            static_cast<double>(source.thermal.theta) + thermal.theta_increment, source.thermal.pressure);
        const double staged_stored_temperature = source_eos_energy_detail::temperature(
            candidate.source_endpoint_.thermal.theta, candidate.source_endpoint_.thermal.pressure);
        if (staged_temperature >= freezing_temperature_k || staged_stored_temperature >= freezing_temperature_k ||
            candidate.energy_.receipt().temperature_eos_target >= freezing_temperature_k ||
            candidate.energy_.receipt().temperature_eos_stored >= freezing_temperature_k)
            return reject("hail surface endpoint requires cold/warm process coupling");
        if (energy_input.accepted_water[6] > 0.0 &&
            (!condensing_melt_domain(staged_temperature, source.thermal.pressure, source.thermal.density, energy_input.accepted_water[0]) ||
             !condensing_melt_domain(staged_stored_temperature, source.thermal.pressure, source.thermal.density,
                 candidate.source_endpoint_.thermal.water[0]) ||
             !condensing_melt_domain(thermal.temperature_eos_target,
                 source.thermal.density * gas_constant_dry_air_jkgk * thermal.temperature_eos_target,
                 source.thermal.density, energy_input.accepted_water[0]) ||
             !condensing_melt_domain(thermal.temperature_eos_stored, candidate.eos_endpoint_.thermal.pressure,
                 source.thermal.density, candidate.eos_endpoint_.thermal.water[0])))
            return reject("hail surface interval crosses its condensing melting regime");
        for (bool stored : {false, true})
        {
            const auto& endpoint = candidate.source_endpoint_;
            if (!population(stored ? endpoint.thermal.water[2] : energy_input.accepted_water[2],
                    stored ? endpoint.rain_number : receipt.rain_number_after,
                    stored ? endpoint.rain_reflectivity : receipt.rain_reflectivity_after,
                    TripleMomentSpecies::Rain, unused_alpha, unused_lambda, error) ||
                !population(stored ? endpoint.thermal.water[6] : energy_input.accepted_water[6],
                    stored ? endpoint.hail_number : receipt.hail_number_after,
                    stored ? endpoint.hail_reflectivity : receipt.hail_reflectivity_after,
                    TripleMomentSpecies::Hail, unused_alpha, unused_lambda, error)) return false;
        }
        *this = candidate;
        error.clear();
        return true;
    }

    bool accept_source(const HailSurfaceState& before, const HailSurfaceState& after,
                       std::uint64_t generation, std::string& error)
    {
        if (!same_moments(before, input_.before) || !same_moments(after, source_endpoint_))
        {
            error = "hail surface source moment owner differs";
            return false;
        }
        return energy_.accept_source(before.thermal, after.thermal, generation, error);
    }

    bool accept_eos(const HailSurfaceState& before, const HailSurfaceState& after,
                    std::uint64_t generation, std::string& error)
    {
        if (!same_moments(before, source_endpoint_) || !same_moments(after, eos_endpoint_))
        {
            error = "hail surface moments changed before EOS";
            return false;
        }
        return energy_.accept_eos(before.thermal, after.thermal, generation, error);
    }

private:
    static bool condensing_melt_domain(double temperature, double pressure, double density, double vapor)
    {
        const double tc = temperature - physical_constants::freezing_temperature_k;
        if (!(tc > -40.0 && tc < 0.0)) return false;
        const auto air = diagnose_air_transport(temperature, pressure, density);
        const double excess = vapor - thermodynamics::saturation_mixing_ratio_water(
            physical_constants::freezing_temperature_k, pressure);
        const double heat = air.thermal_conductivity * tc +
            physical_constants::latent_heat_vaporization_jkg * density * air.vapor_diffusivity * excess;
        return std::isfinite(excess) && excess >= 0.0 && std::isfinite(heat) && heat >= 0.0;
    }

    static bool same_moments(const HailSurfaceState& a, const HailSurfaceState& b)
    {
        const std::array<float, 4> x{a.rain_number, a.rain_reflectivity, a.hail_number, a.hail_reflectivity};
        const std::array<float, 4> y{b.rain_number, b.rain_reflectivity, b.hail_number, b.hail_reflectivity};
        return std::memcmp(x.data(), y.data(), sizeof(x)) == 0;
    }

    static bool population(double q, double n, double z, TripleMomentSpecies species,
                           double& alpha, double& lambda, std::string& error)
    {
        if (q == 0.0 && n == 0.0 && z == 0.0) { alpha = lambda = 0.0; return true; }
        const auto law = current_nimbus_triple_sedimentation_parameters(species).mass_law;
        if (!diagnose_triple_gamma_shape(q, n, z, law, alpha, lambda, error) ||
            !std::isfinite(static_cast<float>(q)) || !std::isfinite(static_cast<float>(n)) || !std::isfinite(static_cast<float>(z)) ||
            static_cast<float>(q) == 0.0f || static_cast<float>(n) == 0.0f || static_cast<float>(z) == 0.0f ||
            alpha > cold_spectrum_shape_limit(ColdSpectrumDomain::GraupelAndHail512, species))
        {
            error = "hail surface source or endpoint q/N/Z is outside its bounded spectrum";
            return false;
        }
        return true;
    }

    HailSurfaceCondensationInput input_{};
    HailSurfaceCondensationReceipt receipt_{};
    HailSurfaceState source_endpoint_{}, eos_endpoint_{};
    SourceEosEnergyTransaction energy_;
};

} // namespace tmv::milbrandt_detail
