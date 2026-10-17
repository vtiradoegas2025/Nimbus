#include "triple_sedimentation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <stdexcept>

namespace tmv::milbrandt_detail
{
namespace
{

double log_gamma_shape_ratio(double alpha, double exponent) noexcept
{
    return std::lgamma(alpha + 7.0) - std::lgamma(alpha + 1.0) +
        (6.0 / exponent) *
            (std::lgamma(alpha + 1.0) -
             std::lgamma(alpha + exponent + 1.0));
}

bool positive_finite(double value) noexcept
{
    return std::isfinite(value) && value > 0.0;
}

bool overlaps(const float* first, const float* second, int levels) noexcept
{
    const auto a = reinterpret_cast<std::uintptr_t>(first);
    const auto b = reinterpret_cast<std::uintptr_t>(second);
    const auto bytes = static_cast<std::uintptr_t>(levels) * sizeof(float);
    return a <= b ? b - a < bytes : a - b < bytes;
}

} // namespace

TripleMomentSedimentationParameters
current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies species)
{
    switch (species)
    {
    case TripleMomentSpecies::Rain:
        return {{523.598775598, 3.0}, 149.1, 0.5, 16.0, 5.0e-3, 4096};
    case TripleMomentSpecies::Ice:
        return {{440.0, 3.0}, 71.34, 0.6635, 2.0, 5.0e-3, 4096};
    case TripleMomentSpecies::Snow:
        return {{0.1597, 2.078}, 11.72, 0.41, 4.0, 5.0e-3, 4096};
    case TripleMomentSpecies::Graupel:
        return {{209.439510239, 3.0}, 19.3, 0.37, 6.0, 2.0e-3, 4096};
    case TripleMomentSpecies::Hail:
        return {{471.238898038, 3.0}, 206.89, 0.6384, 25.0, 80.0e-3, 4096};
    }
    throw std::invalid_argument("unknown Milbrandt triple-moment sedimentation species");
}

bool diagnose_triple_gamma_shape(double mass_per_kg,
                                 double number_per_kg,
                                 double reflectivity_per_kg,
                                 TripleMomentMassLaw mass_law,
                                 double& alpha,
                                 double& lambda,
                                 std::string& error) noexcept
{
    if (!positive_finite(mass_per_kg) || !positive_finite(number_per_kg) ||
        !positive_finite(reflectivity_per_kg) ||
        !positive_finite(mass_law.coefficient) ||
        !positive_finite(mass_law.exponent) || mass_law.exponent >= 6.0)
    {
        error = "hail q/N/Z or mass law is invalid";
        return false;
    }
    const double log_ratio =
        std::log(reflectivity_per_kg) - std::log(number_per_kg) -
        (6.0 / mass_law.exponent) *
            (std::log(mass_per_kg) - std::log(mass_law.coefficient) -
             std::log(number_per_kg));
    if (!std::isfinite(log_ratio) || log_ratio <= 0.0)
    {
        error = "hail q/N/Z violates the sixth-moment realizability bound";
        return false;
    }
    double lower = -1.0 + 1.0e-9;
    double upper = 1.0;
    if (!(log_ratio < log_gamma_shape_ratio(lower, mass_law.exponent)))
    {
        error = "hail gamma shape is outside the bounded inversion bracket";
        return false;
    }
    while (log_gamma_shape_ratio(upper, mass_law.exponent) > log_ratio &&
           upper < 1.0e6)
    {
        upper *= 2.0;
    }
    if (upper > 1.0e6 ||
        log_gamma_shape_ratio(upper, mass_law.exponent) > log_ratio)
    {
        error = "hail gamma shape is outside the bounded inversion bracket";
        return false;
    }
    for (int iteration = 0; iteration < 80; ++iteration)
    {
        const double middle = 0.5 * (lower + upper);
        if (log_gamma_shape_ratio(middle, mass_law.exponent) > log_ratio)
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
    }
    const double candidate_alpha = 0.5 * (lower + upper);
    const double log_lambda =
        (std::log(mass_law.coefficient) + std::log(number_per_kg) +
         std::lgamma(candidate_alpha + mass_law.exponent + 1.0) -
         std::lgamma(candidate_alpha + 1.0) - std::log(mass_per_kg)) /
        mass_law.exponent;
    const double candidate_lambda = std::exp(log_lambda);
    if (!std::isfinite(candidate_alpha) || !positive_finite(candidate_lambda))
    {
        error = "hail gamma shape or slope is nonfinite";
        return false;
    }
    alpha = candidate_alpha;
    lambda = candidate_lambda;
    error.clear();
    return true;
}

bool triple_moment_fall_speeds(double density,
                               double alpha,
                               double lambda,
                               const TripleMomentSedimentationParameters& parameters,
                               std::array<double, 3>& speeds) noexcept
{
    if (!positive_finite(density) || !std::isfinite(alpha) || alpha <= -1.0 ||
        !positive_finite(lambda) ||
        !positive_finite(parameters.fall_coefficient) ||
        !std::isfinite(parameters.fall_exponent) ||
        parameters.fall_exponent < 0.0 ||
        !positive_finite(parameters.maximum_speed) ||
        !positive_finite(parameters.maximum_diameter) ||
        !positive_finite(parameters.mass_law.exponent) ||
        parameters.mass_law.exponent >= 6.0)
    {
        return false;
    }
    const std::array<double, 3> orders{0.0, parameters.mass_law.exponent, 6.0};
    std::array<double, 3> candidate{};
    const double log_common =
        std::log(parameters.fall_coefficient) +
        0.5 * std::log(1.225 / density) -
        parameters.fall_exponent * std::log(lambda);
    for (std::size_t moment = 0; moment < orders.size(); ++moment)
    {
        const double log_speed = log_common +
            std::lgamma(alpha + orders[moment] +
                        parameters.fall_exponent + 1.0) -
            std::lgamma(alpha + orders[moment] + 1.0);
        if (!std::isfinite(log_speed))
        {
            return false;
        }
        const double unbounded_speed = std::exp(log_speed);
        if (!std::isfinite(unbounded_speed) || unbounded_speed < 0.0)
        {
            return false;
        }
        candidate[moment] = std::min(parameters.maximum_speed, unbounded_speed);
    }
    speeds = candidate;
    return true;
}

bool triple_hail_fall_speeds(double density,
                             double alpha,
                             double lambda,
                             const TripleHailSedimentationParameters& parameters,
                             std::array<double, 3>& speeds) noexcept
{
    return triple_moment_fall_speeds(
        density, alpha, lambda, parameters, speeds);
}

bool sediment_triple_moment_column(
    int levels,
    const float* mass,
    const float* number,
    const float* reflectivity,
    const float* density,
    const double* thickness,
    double dt,
    const TripleMomentSedimentationParameters& parameters,
    float* mass_after,
    float* number_after,
    float* reflectivity_after,
    TripleMomentSurfaceInventory& surface,
    std::string& error)
{
    if (levels <= 0 || levels > 4096 ||
        !mass || !number || !reflectivity || !density || !thickness ||
        !mass_after || !number_after || !reflectivity_after ||
        !positive_finite(dt) || parameters.maximum_substeps <= 0 ||
        parameters.maximum_substeps > 4096 ||
        !positive_finite(parameters.mass_law.coefficient) ||
        !positive_finite(parameters.mass_law.exponent) ||
        parameters.mass_law.exponent >= 6.0 ||
        !positive_finite(parameters.fall_coefficient) ||
        !std::isfinite(parameters.fall_exponent) ||
        parameters.fall_exponent < 0.0 ||
        !positive_finite(parameters.maximum_speed) ||
        overlaps(mass_after, number_after, levels) ||
        overlaps(mass_after, reflectivity_after, levels) ||
        overlaps(number_after, reflectivity_after, levels))
    {
        error = "triple-moment sedimentation extent, pointer, timestep or resource cap is invalid";
        return false;
    }
    if (static_cast<std::int64_t>(levels) * parameters.maximum_substeps > 65536)
    {
        error = "triple-moment sedimentation exceeds the 65536 cell-substep resource budget";
        return false;
    }
    for (const float* input : {mass, number, reflectivity, density})
    {
        for (float* output : {mass_after, number_after, reflectivity_after})
        {
            if (overlaps(input, output, levels))
            {
                error = "triple-moment sedimentation output aliases input";
                return false;
            }
        }
    }

    std::array<std::vector<double>, 3> work;
    std::array<std::vector<double>, 3> next;
    for (std::size_t moment = 0; moment < work.size(); ++moment)
    {
        work[moment].resize(static_cast<std::size_t>(levels));
        next[moment].resize(static_cast<std::size_t>(levels));
    }
    for (int level = 0; level < levels; ++level)
    {
        if (!positive_finite(density[level]) ||
            !positive_finite(thickness[level]))
        {
            error = "triple-moment sedimentation density or thickness is invalid";
            return false;
        }
        const double values[3]{mass[level], number[level], reflectivity[level]};
        if (values[0] == 0.0 && values[1] == 0.0 && values[2] == 0.0)
        {
            continue;
        }
        double alpha = 0.0;
        double lambda = 0.0;
        if (!diagnose_triple_gamma_shape(values[0], values[1], values[2],
                                         parameters.mass_law, alpha, lambda,
                                         error))
        {
            return false;
        }
        for (std::size_t moment = 0; moment < work.size(); ++moment)
        {
            work[moment][static_cast<std::size_t>(level)] = values[moment];
        }
    }

    TripleMomentSurfaceInventory candidate_surface{};
    double remaining = dt;
    std::vector<std::array<double, 3>> velocities(static_cast<std::size_t>(levels));
    for (int substep = 0; remaining > 1.0e-12 * dt; ++substep)
    {
        if (substep >= parameters.maximum_substeps)
        {
            error = "triple-moment sedimentation exceeded its bounded CFL substep cap";
            return false;
        }
        std::fill(velocities.begin(), velocities.end(), std::array<double, 3>{});
        double stable = remaining;
        bool active = false;
        for (int level = 0; level < levels; ++level)
        {
            const std::size_t index = static_cast<std::size_t>(level);
            if (work[0][index] == 0.0 && work[1][index] == 0.0 &&
                work[2][index] == 0.0)
            {
                continue;
            }
            double alpha = 0.0;
            double lambda = 0.0;
            if (!diagnose_triple_gamma_shape(
                    work[0][index], work[1][index], work[2][index],
                    parameters.mass_law, alpha, lambda, error) ||
                !triple_moment_fall_speeds(density[level], alpha, lambda,
                                           parameters, velocities[index]))
            {
                if (error.empty())
                {
                    error = "triple-moment sedimentation fall speed is invalid";
                }
                return false;
            }
            active = true;
            for (double speed : velocities[index])
            {
                if (speed > 0.0)
                {
                    stable = std::min(stable, 0.8 * thickness[level] / speed);
                }
            }
        }
        if (!active)
        {
            break;
        }
        if (!positive_finite(stable))
        {
            error = "triple-moment sedimentation CFL timestep is invalid";
            return false;
        }
        next = work;
        for (int level = 0; level < levels; ++level)
        {
            const std::size_t index = static_cast<std::size_t>(level);
            const double inventory_scale =
                static_cast<double>(density[level]) * thickness[level];
            for (std::size_t moment = 0; moment < work.size(); ++moment)
            {
                const double outgoing =
                    work[moment][index] * velocities[index][moment] * stable /
                    thickness[level];
                next[moment][index] -= outgoing;
                if (level > 0)
                {
                    next[moment][index - 1] += outgoing * inventory_scale /
                        (static_cast<double>(density[level - 1]) *
                         thickness[level - 1]);
                }
                else
                {
                    double* const surface_moment = moment == 0
                        ? &candidate_surface.mass
                        : moment == 1 ? &candidate_surface.number
                                      : &candidate_surface.reflectivity;
                    *surface_moment += outgoing * inventory_scale;
                }
            }
        }
        for (int level = 0; level < levels; ++level)
        {
            const std::size_t index = static_cast<std::size_t>(level);
            const double q = next[0][index];
            const double n = next[1][index];
            const double z = next[2][index];
            if (q == 0.0 && n == 0.0 && z == 0.0)
            {
                continue;
            }
            double alpha = 0.0;
            double lambda = 0.0;
            if (!diagnose_triple_gamma_shape(q, n, z, parameters.mass_law,
                                             alpha, lambda, error))
            {
                return false;
            }
        }
        work.swap(next);
        remaining -= stable;
    }

    std::array<std::vector<float>, 3> output;
    for (std::size_t moment = 0; moment < output.size(); ++moment)
    {
        output[moment].resize(static_cast<std::size_t>(levels));
        for (int level = 0; level < levels; ++level)
        {
            const double value = work[moment][static_cast<std::size_t>(level)];
            if (!std::isfinite(value) || value < 0.0 ||
                value > std::numeric_limits<float>::max() ||
                (value > 0.0 &&
                 value < std::numeric_limits<float>::denorm_min()))
            {
                error = "triple-moment sedimentation output is not representable";
                return false;
            }
            output[moment][static_cast<std::size_t>(level)] =
                static_cast<float>(value);
        }
    }
    for (int level = 0; level < levels; ++level)
    {
        const std::size_t index = static_cast<std::size_t>(level);
        const double q = output[0][index];
        const double n = output[1][index];
        const double z = output[2][index];
        if (q == 0.0 && n == 0.0 && z == 0.0)
        {
            continue;
        }
        double alpha = 0.0;
        double lambda = 0.0;
        if (!diagnose_triple_gamma_shape(q, n, z, parameters.mass_law,
                                         alpha, lambda, error))
        {
            error = "triple-moment sedimentation rounded output is unrealizable";
            return false;
        }
    }
    if (!std::isfinite(candidate_surface.mass) ||
        !std::isfinite(candidate_surface.number) ||
        !std::isfinite(candidate_surface.reflectivity))
    {
        error = "triple-moment sedimentation surface inventory is nonfinite";
        return false;
    }
    for (int level = 0; level < levels; ++level)
    {
        const std::size_t index = static_cast<std::size_t>(level);
        mass_after[level] = output[0][index];
        number_after[level] = output[1][index];
        reflectivity_after[level] = output[2][index];
    }
    surface = candidate_surface;
    error.clear();
    return true;
}

bool sediment_triple_hail_column(
    int levels,
    const float* mass,
    const float* number,
    const float* reflectivity,
    const float* density,
    const double* thickness,
    double dt,
    const TripleHailSedimentationParameters& parameters,
    float* mass_after,
    float* number_after,
    float* reflectivity_after,
    TripleHailSurfaceInventory& surface,
    std::string& error)
{
    return sediment_triple_moment_column(
        levels, mass, number, reflectivity, density, thickness, dt,
        parameters, mass_after, number_after, reflectivity_after,
        surface, error);
}

} // namespace tmv::milbrandt_detail
