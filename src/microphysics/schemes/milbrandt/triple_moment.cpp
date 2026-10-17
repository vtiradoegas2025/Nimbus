#include "triple_moment.hpp"

#include <cmath>
#include <limits>
#include <utility>

namespace tmv::milbrandt_detail
{
namespace
{

bool same_shape(const Field3D& left, const Field3D& right) noexcept
{
    return left.size_r() == right.size_r() &&
           left.size_th() == right.size_th() &&
           left.size_z() == right.size_z();
}

bool valid_mass_law(TripleMomentMassLaw law) noexcept
{
    return std::isfinite(law.coefficient) && law.coefficient > 0.0 &&
           std::isfinite(law.exponent) && law.exponent > 0.0 &&
           law.exponent < 6.0;
}

bool valid_represented_moment(double density,
                              double mass,
                              double number,
                              double reflectivity,
                              TripleMomentMassLaw law) noexcept
{
    if (!std::isfinite(density) || density <= 0.0 ||
        !std::isfinite(mass) || mass < 0.0 ||
        !std::isfinite(number) || number < 0.0 ||
        !std::isfinite(reflectivity) || reflectivity < 0.0 ||
        !valid_mass_law(law))
    {
        return false;
    }
    if (mass == 0.0 || number == 0.0 || reflectivity == 0.0)
    {
        return mass == 0.0 && number == 0.0 && reflectivity == 0.0;
    }
    const double log_lower = std::log(number) +
        (6.0 / law.exponent) *
            (std::log(mass) - std::log(law.coefficient) - std::log(number));
    return std::isfinite(log_lower) && std::log(reflectivity) >= log_lower;
}

bool valid_fields(const Field3D& density,
                  const TripleMomentFieldRefs& mass,
                  const TripleMomentFieldRefs& number,
                  const TripleMomentMassLaws& laws,
                  std::string& error)
{
    if (density.empty())
    {
        error = "triple-moment density field is empty";
        return false;
    }
    for (std::size_t species = 0; species < mass.size(); ++species)
    {
        if (!mass[species] || !number[species] ||
            !same_shape(density, *mass[species]) ||
            !same_shape(density, *number[species]) ||
            !valid_mass_law(laws[species]))
        {
            error = "triple-moment input shape or mass law is invalid";
            return false;
        }
    }
    return true;
}

} // namespace

bool gamma_sixth_moment_per_kg(double density,
                              double mass_mixing_ratio,
                              double number_per_kg,
                              double alpha,
                              TripleMomentMassLaw mass_law,
                              float& result) noexcept
{
    if (!std::isfinite(density) || density <= 0.0 ||
        !std::isfinite(mass_mixing_ratio) || mass_mixing_ratio < 0.0 ||
        !std::isfinite(number_per_kg) || number_per_kg < 0.0 ||
        !std::isfinite(alpha) || alpha <= -1.0 ||
        !valid_mass_law(mass_law))
    {
        return false;
    }
    if (mass_mixing_ratio == 0.0 || number_per_kg == 0.0)
    {
        if (mass_mixing_ratio != 0.0 || number_per_kg != 0.0)
        {
            return false;
        }
        result = 0.0f;
        return true;
    }

    const double log_number_volume = std::log(density) + std::log(number_per_kg);
    const double log_mass_volume = std::log(density) + std::log(mass_mixing_ratio);
    const double log_gamma_number = std::lgamma(alpha + 1.0);
    const double log_lambda =
        (std::log(mass_law.coefficient) + log_number_volume +
         std::lgamma(alpha + mass_law.exponent + 1.0) -
         log_gamma_number - log_mass_volume) / mass_law.exponent;
    const double log_reflectivity_per_kg =
        log_number_volume - 6.0 * log_lambda +
        std::lgamma(alpha + 7.0) - log_gamma_number - std::log(density);
    if (!std::isfinite(log_reflectivity_per_kg) ||
        log_reflectivity_per_kg > std::log(std::numeric_limits<float>::max()) ||
        log_reflectivity_per_kg < std::log(std::numeric_limits<float>::denorm_min()))
    {
        return false;
    }
    const float candidate = static_cast<float>(std::exp(log_reflectivity_per_kg));
    if (!std::isfinite(candidate) || candidate <= 0.0f)
    {
        return false;
    }
    result = candidate;
    return true;
}

bool TripleMomentCandidateState::prepare(
    const Field3D& density,
    const TripleMomentFieldRefs& mass,
    const TripleMomentFieldRefs& number,
    const TripleMomentMassLaws& mass_laws,
    const std::array<double, 5>& initial_alpha,
    std::string& error)
{
    if (!valid_fields(density, mass, number, mass_laws, error))
    {
        return false;
    }
    std::array<Field3D, 5> candidate;
    std::array<Field3D, 5> candidate_tendencies;
    for (std::size_t species = 0; species < candidate.size(); ++species)
    {
        candidate[species].resize(density.size_r(), density.size_th(), density.size_z(), 0.0f);
        candidate_tendencies[species].resize(density.size_r(), density.size_th(), density.size_z(), 0.0f);
        for (std::size_t cell = 0; cell < density.size(); ++cell)
        {
            float represented = 0.0f;
            if (!gamma_sixth_moment_per_kg(
                    density.data()[cell], mass[species]->data()[cell],
                    number[species]->data()[cell], initial_alpha[species],
                    mass_laws[species], represented))
            {
                error = "triple-moment initial gamma state is invalid";
                return false;
            }
            candidate[species].data()[cell] = represented;
        }
    }
    moments_ = std::move(candidate);
    tendencies_ = std::move(candidate_tendencies);
    error.clear();
    return true;
}

bool TripleMomentCandidateState::commit(
    double dt,
    const Field3D& density,
    const TripleMomentFieldRefs& mass_after,
    const TripleMomentFieldRefs& number_after,
    const TripleMomentMassLaws& mass_laws,
    std::string& error)
{
    error.clear();
    if (!std::isfinite(dt) || dt <= 0.0 ||
        !valid_fields(density, mass_after, number_after, mass_laws, error))
    {
        if (error.empty())
        {
            error = "triple-moment timestep is invalid";
        }
        return false;
    }
    std::array<Field3D, 5> candidate;
    for (std::size_t species = 0; species < candidate.size(); ++species)
    {
        if (!same_shape(density, moments_[species]) ||
            !same_shape(density, tendencies_[species]))
        {
            error = "triple-moment candidate shape is invalid";
            return false;
        }
        candidate[species] = moments_[species];
        for (std::size_t cell = 0; cell < density.size(); ++cell)
        {
            const double updated =
                static_cast<double>(moments_[species].data()[cell]) +
                dt * static_cast<double>(tendencies_[species].data()[cell]);
            if (!valid_represented_moment(
                    density.data()[cell], mass_after[species]->data()[cell],
                    number_after[species]->data()[cell], updated,
                    mass_laws[species]) ||
                updated > std::numeric_limits<float>::max())
            {
                error = "triple-moment candidate is nonfinite or unrealizable";
                return false;
            }
            const float rounded = static_cast<float>(updated);
            if (!valid_represented_moment(
                    density.data()[cell], mass_after[species]->data()[cell],
                    number_after[species]->data()[cell], rounded,
                    mass_laws[species]))
            {
                error = "triple-moment candidate is not representable";
                return false;
            }
            candidate[species].data()[cell] = rounded;
        }
    }
    moments_ = std::move(candidate);
    error.clear();
    return true;
}

const Field3D& TripleMomentCandidateState::moment(std::size_t species) const
{
    return moments_.at(species);
}

Field3D& TripleMomentCandidateState::tendency(std::size_t species)
{
    return tendencies_.at(species);
}

} // namespace tmv::milbrandt_detail
