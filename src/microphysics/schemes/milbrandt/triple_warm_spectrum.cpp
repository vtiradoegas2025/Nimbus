#include "triple_warm_spectrum.hpp"

#include "triple_sedimentation.hpp"

#include <cmath>

namespace tmv::milbrandt_detail
{
namespace
{

bool positive_finite(double value) noexcept
{
    return std::isfinite(value) && value > 0.0;
}

double cubic_log_shape_ratio(double alpha) noexcept
{
    return std::log1p(3.0 / (alpha + 1.0)) +
        std::log1p(3.0 / (alpha + 2.0)) +
        std::log1p(3.0 / (alpha + 3.0));
}

bool refine_cubic_spectrum(const ExactRainTripleMomentState& source,
                           double mass_coefficient, double& alpha,
                           double& lambda) noexcept
{
    int mass_exponent = 0;
    int number_exponent = 0;
    int reflectivity_exponent = 0;
    int coefficient_exponent = 0;
    const double mass_fraction = std::frexp(source.mass_per_kg, &mass_exponent);
    const double number_fraction = std::frexp(source.number_per_kg, &number_exponent);
    const double reflectivity_fraction = std::frexp(
        source.reflectivity_per_kg, &reflectivity_exponent);
    const double coefficient_fraction = std::frexp(mass_coefficient, &coefficient_exponent);
    const double coefficient_over_mass = coefficient_fraction / mass_fraction;
    const double shape_ratio = std::ldexp(
        reflectivity_fraction * number_fraction *
            coefficient_over_mass * coefficient_over_mass,
        reflectivity_exponent + number_exponent +
            2 * (coefficient_exponent - mass_exponent));
    const double log_ratio = std::log(shape_ratio);
    if (!positive_finite(log_ratio))
    {
        return false;
    }
    double lower = -1.0 + 1.0e-9;
    double upper = 1.0;
    if (!(log_ratio < cubic_log_shape_ratio(lower)))
    {
        return false;
    }
    while (cubic_log_shape_ratio(upper) > log_ratio && upper < 1.0e6)
    {
        upper *= 2.0;
    }
    if (upper > 1.0e6 || cubic_log_shape_ratio(upper) > log_ratio)
    {
        return false;
    }
    for (int iteration = 0; iteration < 80; ++iteration)
    {
        const double middle = 0.5 * (lower + upper);
        if (cubic_log_shape_ratio(middle) > log_ratio)
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
    }
    const double candidate_alpha = 0.5 * (lower + upper);
    const double candidate_lambda = std::exp(
        (std::log(mass_coefficient) + std::log(source.number_per_kg) +
         std::log(candidate_alpha + 1.0) +
         std::log(candidate_alpha + 2.0) +
         std::log(candidate_alpha + 3.0) -
         std::log(source.mass_per_kg)) / 3.0);
    if (!std::isfinite(candidate_alpha) || !positive_finite(candidate_lambda))
    {
        return false;
    }
    alpha = candidate_alpha;
    lambda = candidate_lambda;
    return true;
}

bool normalized_moment(double alpha, double lambda, double order,
                       double& result) noexcept
{
    double candidate = 1.0;
    if (order == std::floor(order))
    {
        for (int factor = 1; factor <= static_cast<int>(order); ++factor)
        {
            candidate *= (alpha + factor) / lambda;
        }
    }
    else
    {
        const double log_moment = std::lgamma(alpha + order + 1.0) -
            std::lgamma(alpha + 1.0) - order * std::log(lambda);
        if (!std::isfinite(log_moment))
        {
            return false;
        }
        candidate = std::exp(log_moment);
    }
    if (!positive_finite(candidate))
    {
        return false;
    }
    result = candidate;
    return true;
}

} // namespace

bool TripleWarmRainSpectrum::normalized_diameter_moment(
    double order, double& result, std::string& error) const
{
    if (!valid_ || !std::isfinite(order) || order < 0.0 || order > 6.0)
    {
        error = "triple warm rain spectrum or diameter order is invalid";
        return false;
    }
    double candidate = 0.0;
    if (!normalized_moment(alpha_, lambda_per_m_, order, candidate))
    {
        error = "triple warm rain diameter moment is not representable";
        return false;
    }
    result = candidate;
    error.clear();
    return true;
}

bool diagnose_triple_warm_rain_spectrum(
    const ExactRainTripleMomentState& source,
    TripleWarmRainSpectrum& result,
    std::string& error)
{
    const auto defaults = current_nimbus_triple_sedimentation_parameters(
        TripleMomentSpecies::Rain);
    TripleWarmRainSpectrum candidate;
    if (!diagnose_triple_gamma_shape(
            source.mass_per_kg, source.number_per_kg,
            source.reflectivity_per_kg, defaults.mass_law,
            candidate.alpha_, candidate.lambda_per_m_, error))
    {
        return false;
    }
    // Cubic rain admits a polynomial closure, avoiding high-shape lgamma cancellation.
    if (!refine_cubic_spectrum(source, defaults.mass_law.coefficient,
                               candidate.alpha_, candidate.lambda_per_m_))
    {
        error = "triple warm rain cubic spectrum is outside its admitted bracket";
        return false;
    }
    candidate.mean_mass_diameter_m_ = std::exp(
        (std::log(source.mass_per_kg) -
         std::log(defaults.mass_law.coefficient) -
         std::log(source.number_per_kg)) / defaults.mass_law.exponent);
    candidate.alpha_zero_equivalent_inverse_slope_m_ =
        candidate.mean_mass_diameter_m_ / std::cbrt(6.0);
    if (!positive_finite(candidate.mean_mass_diameter_m_) ||
        !positive_finite(candidate.alpha_zero_equivalent_inverse_slope_m_))
    {
        error = "triple warm rain mean diameter is not representable";
        return false;
    }
    const std::array<double, 4> orders{
        1.0, 3.0, 6.0, 1.5 + 0.5 * defaults.fall_exponent};
    for (std::size_t index = 0; index < orders.size(); ++index)
    {
        if (!normalized_moment(candidate.alpha_, candidate.lambda_per_m_,
                               orders[index], candidate.moments_[index]))
        {
            error = "triple warm rain normalized diameter moment is not representable";
            return false;
        }
    }
    candidate.source_ = source;
    candidate.valid_ = true;
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
