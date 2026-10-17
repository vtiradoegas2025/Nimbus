#include "mynn_stability.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tmv::mynn
{
namespace
{

// Nakanishi--Niino (2009) closure constants. The dependent constants are
// evaluated from the same relationships as the authors' reference model,
// avoiding internally inconsistent rounded values.
constexpr double g1                         = 0.235;
constexpr double b1                         = 24.0;
constexpr double b2                         = 15.0;
constexpr double c2                         = 0.75;
constexpr double c3                         = 0.352;
constexpr double c5                         = 0.2;
constexpr double turbulent_prandtl_unstable = 0.74;

const double a1 = b1 * (1.0 - 3.0 * g1) / 6.0;
const double c1 = g1 - 1.0 / (3.0 * a1 * std::cbrt(b1));
const double a2 = a1 * (g1 - c1) / (g1 * turbulent_prandtl_unstable);
const double g2 = b2 / b1 * (1.0 - c3) + 2.0 * a1 / b1 * (3.0 - 2.0 * c2);

struct Level2Stability
{
    double sm = 0.0;
    double sh = 0.0;
    double ri = 0.0;
    double rf = 0.0;
};

Level2Stability level2_stability(double gm, double gh)
{
    const double rfc = g1 / (g1 + g2);
    const double f1  = b1 * (g1 - c1) + 3.0 * a2 * (1.0 - c2) * (1.0 - c5) + 2.0 * a1 * (3.0 - 2.0 * c2);
    const double f2  = b1 * (g1 + g2) - 3.0 * a1 * (1.0 - c2);
    const double rf1 = b1 * (g1 - c1) / f1;
    const double rf2 = b1 * g1 / f2;
    const double smc = a1 / a2 * f1 / f2;
    const double shc = 3.0 * a2 * (g1 + g2);
    const double ri1 = 0.5 / smc;
    const double ri2 = rf1 * smc;
    const double ri3 = 4.0 * rf2 * smc - 2.0 * ri2;
    const double ri4 = ri2 * ri2;

    const double ri           = -gh / std::max(gm, 1.0e-10);
    const double discriminant = std::max(0.0, ri * ri - ri3 * ri + ri4);
    const double rf           = std::min(ri1 * (ri + ri2 - std::sqrt(discriminant)), rfc);
    const double sh           = shc * (rfc - rf) / (1.0 - rf);
    const double sm           = smc * (rf1 - rf) / (rf2 - rf) * sh;
    return {sm, sh, ri, rf};
}

} // namespace

Level25Stability level25_stability_functions(const Level25StabilityInput& input)
{
    if (!std::isfinite(input.gm_s2) || input.gm_s2 < 0.0 || !std::isfinite(input.gh_s2) ||
        !std::isfinite(input.mixing_length_m) || input.mixing_length_m <= 0.0 || !std::isfinite(input.q_squared_m2s2) ||
        input.q_squared_m2s2 <= 0.0)
    {
        throw std::invalid_argument("MYNN stability requires finite Gm/Gh and positive finite L/q^2");
    }

    const Level2Stability level2         = level2_stability(input.gm_s2, input.gh_s2);
    const double          length_squared = input.mixing_length_m * input.mixing_length_m;
    const double          q_squared      = input.q_squared_m2s2;
    const double equilibrium_q_squared   = b1 * length_squared * (level2.sm * input.gm_s2 + level2.sh * input.gh_s2);

    double sm   = level2.sm;
    double sh   = level2.sh;
    double qdiv = 1.0;
    if (equilibrium_q_squared > 0.0 && q_squared < equilibrium_q_squared)
    {
        // Helfand--Labraga realizability correction used by MYNN level 2.5.
        qdiv = std::sqrt(q_squared / equilibrium_q_squared);
        sm *= qdiv;
        sh *= qdiv;
    }
    else
    {
        const double e5c         = 6.0 * a1 * a1;
        const double e1c         = 3.0 * a2 * b2 * (1.0 - c3);
        const double e2c         = 9.0 * a1 * a2 * (1.0 - c2);
        const double e3c         = 9.0 * a2 * a2 * (1.0 - c2) * (1.0 - c5);
        const double e4c         = 12.0 * a1 * a2 * (1.0 - c2);
        const double gmel        = input.gm_s2 * length_squared;
        const double ghel        = input.gh_s2 * length_squared;
        const double e1          = q_squared - e1c * ghel;
        const double e2          = q_squared - e2c * ghel;
        const double e3          = e1 + e3c * ghel;
        const double e4          = e1 - e4c * ghel;
        const double denominator = e2 * e4 + e3 * e5c * gmel;
        if (!std::isfinite(denominator) || denominator <= 1.0e-20)
            throw std::runtime_error("MYNN level-2.5 stability denominator is non-positive");
        sm = q_squared * a1 * (e3 - 3.0 * c1 * e4) / denominator;
        sh = q_squared * a2 * (e2 + 3.0 * c1 * e5c * gmel) / denominator;
    }

    if (!std::isfinite(sm) || !std::isfinite(sh) || sm < 0.0 || sh < 0.0)
        throw std::runtime_error("MYNN level-2.5 stability produced an unrealizable coefficient");
    return {sm, sh, level2.ri, level2.rf, qdiv};
}

} // namespace tmv::mynn
