#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tmv::milbrandt_detail
{

enum class ColdFrozenDiffusionSpecies
{
    Graupel,
    Hail
};

struct ColdFrozenDiffusionAcceptedEvent
{
    ColdFrozenDiffusionSpecies species = ColdFrozenDiffusionSpecies::Graupel;
    double entry_mass_per_kg = 0.0;
    double entry_number_per_kg = 0.0;
    double source_alpha = 0.0;
    double admitted_mass_increment_per_kg = 0.0;
    double admitted_number_increment_per_kg = 0.0;
    double dry_density_kg_m3 = 0.0;
    double duration_s = 0.0;
};

struct ColdFrozenDiffusionAcceptedEventBuffer
{
    ColdFrozenDiffusionSpecies species = ColdFrozenDiffusionSpecies::Graupel;
    std::array<ColdFrozenDiffusionAcceptedEvent, 1> events{};
    std::size_t size = 0;
    std::size_t capacity = 1;
};

static_assert(std::is_trivially_copyable_v<ColdFrozenDiffusionAcceptedEvent>);

inline ColdFrozenDiffusionAcceptedEvent make_published_frozen_diffusion_event(
    ColdFrozenDiffusionSpecies species,
    double mass, double number, double alpha,
    double mass_increment, double number_increment,
    double density, double duration)
{
    if ((species != ColdFrozenDiffusionSpecies::Graupel &&
         species != ColdFrozenDiffusionSpecies::Hail) ||
        !std::isfinite(mass) || mass < 0.0 ||
        !std::isfinite(number) || number < 0.0 ||
        !std::isfinite(alpha) || alpha <= -1.0 ||
        !std::isfinite(mass_increment) || mass_increment < -mass ||
        !std::isfinite(number_increment) || number_increment < -number ||
        !std::isfinite(density) || density <= 0.0 ||
        !std::isfinite(duration) || duration <= 0.0 ||
        ((mass == 0.0 || number == 0.0) &&
         (mass != 0.0 || number != 0.0 ||
          mass_increment != 0.0 || number_increment != 0.0)))
    {
        throw std::runtime_error("published frozen-diffusion event inputs are invalid");
    }
    if (mass_increment >= 0.0)
    {
        if (number_increment != 0.0)
        {
            throw std::runtime_error("published frozen deposition cannot change number");
        }
    }
    else
    {
        const double mass_fraction = mass_increment / mass;
        const double number_fraction = number_increment / number;
        const double scale = std::max(std::abs(mass_fraction), 1.0e-30);
        if (!(number_increment < 0.0) ||
            std::abs(mass_fraction - number_fraction) >
                64.0 * std::numeric_limits<double>::epsilon() * scale)
        {
            throw std::runtime_error("published frozen sublimation q/N fractions differ");
        }
    }
    return {species, mass, number, alpha, mass_increment, number_increment,
            density, duration};
}

} // namespace tmv::milbrandt_detail
