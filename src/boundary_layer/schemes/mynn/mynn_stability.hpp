#pragma once

namespace tmv::mynn
{

// Dimensional inputs at a model-layer interface. The sign convention follows
// Nakanishi and Niino: gm >= 0 is shear production divided by Sm*L*q, while
// gh > 0 denotes convectively unstable buoyancy production.
struct Level25StabilityInput
{
    double gm_s2           = 0.0;
    double gh_s2           = 0.0;
    double mixing_length_m = 0.0;
    double q_squared_m2s2  = 0.0;
};

struct Level25Stability
{
    double sm                  = 0.0;
    double sh                  = 0.0;
    double gradient_richardson = 0.0;
    double flux_richardson     = 0.0;
    double equilibrium_limiter = 1.0;
};

/**
 * Nakanishi--Niino level-2.5 stability functions.
 *
 * This is the dry/moist-unsaturated form of the closure in the authors'
 * reference implementation. Moisture enters through the caller's virtual
 * potential-temperature gradient. Invalid dimensional inputs are rejected.
 */
Level25Stability level25_stability_functions(const Level25StabilityInput& input);

} // namespace tmv::mynn
