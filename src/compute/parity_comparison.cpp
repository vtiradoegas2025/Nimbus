// Pure CPU/GPU parity comparison policy and field traversal.
#include "compute/parity_check.hpp"

#include <cmath>
#include <limits>

namespace nimbus::parity {

namespace {

struct DomainBounds
{
    int i_begin = 0;
    int i_end = 0;
    int j_begin = 0;
    int j_end = 0;
    int k_begin = 0;
    int k_end = 0;
    const char* invalid_reason = nullptr;
};

DomainBounds bounds_for(ComparisonDomain domain, int nr, int nth, int nz)
{
    DomainBounds bounds;
    if (nr <= 0 || nth <= 0 || nz <= 0)
    {
        bounds.invalid_reason = "invalid_dimensions";
        return bounds;
    }

    switch (domain)
    {
        case ComparisonDomain::StrictInterior:
            if (nr < 3 || nth < 3 || nz < 3)
            {
                bounds.invalid_reason = "empty_strict_interior";
                return bounds;
            }
            bounds = {1, nr - 1, 1, nth - 1, 1, nz - 1, nullptr};
            break;
        case ComparisonDomain::FullField:
            bounds = {0, nr, 0, nth, 0, nz, nullptr};
            break;
        case ComparisonDomain::BoundaryShell:
            // The rectangular bounds cover the field; the loop below skips
            // strict-interior cells so edge/corner cells are counted once.
            bounds = {0, nr, 0, nth, 0, nz, nullptr};
            break;
        case ComparisonDomain::WestFace:
            bounds = {0, 1, 0, nth, 0, nz, nullptr};
            break;
        case ComparisonDomain::EastFace:
            bounds = {nr - 1, nr, 0, nth, 0, nz, nullptr};
            break;
        case ComparisonDomain::SouthFace:
            bounds = {0, nr, 0, 1, 0, nz, nullptr};
            break;
        case ComparisonDomain::NorthFace:
            bounds = {0, nr, nth - 1, nth, 0, nz, nullptr};
            break;
        case ComparisonDomain::BottomFace:
            bounds = {0, nr, 0, nth, 0, 1, nullptr};
            break;
        case ComparisonDomain::TopFace:
            bounds = {0, nr, 0, nth, nz - 1, nz, nullptr};
            break;
        default:
            bounds.invalid_reason = "invalid_comparison_domain";
            break;
    }
    return bounds;
}

}  // namespace

Tolerance tolerance_for(Tier tier)
{
    // GPU.md section 2.3 tiered parity tolerances.
    //
    // The Stencil absolute band is calibrated to the derived fp32 noise
    // floor of the acoustic composite: the diagnosed pressure carries
    // ~1 ulp (7.8e-3 Pa at 1e5 Pa) of independent per-cell rounding
    // between the fp64 CPU and fp32 GPU; the momentum face-difference
    // amplifies it by dt/(dz*rho) -> ~1.2e-5 m/s per substep at beast-like
    // parameters (measured 1.216e-5 vs derived 1.15e-5 on the calm
    // fixture). The gate sits 4x above that floor and ~20x below the
    // smallest real-bug signal observed (a wrong base-state subtraction
    // produced O(1e-3)).
    // The MicrophysicsHeating absolute band is the mixing-ratio band
    // propagated through the latent-heating chain: a mixing-ratio
    // difference epsilon produces a heating difference
    // epsilon * (L/cp) * Exner (the spurious /T amplifier was removed
    // with SCIENCE.md C3, so the band grew by ~T). Worst case
    // (L_s/cp = 2812, Exner at 100 hPa = 1.93):
    // 1e-7 * 2812 * 1.93 ~ 5.4e-4 -> 6e-4.
    //
    // Horizon scaling (check_field's `horizon` parameter): a batched
    // kernel verified once after N sequential substeps accumulates the
    // per-substep rounding divergence near-LINEARLY, not as a sqrt(N)
    // random walk - the fp64 CPU and fp32 GPU make systematically
    // different roundings at the same cells every substep, so the gap
    // compounds. Measured on the calm fixture: the batched 8-substep
    // acoustic loop's w divergence is 8.6e-5 ~ 7.1x the 1.2e-5
    // single-substep floor (linear bound 8x; sqrt bound 2.8x would be
    // breached). Scaling the absolute band by N preserves the 4x
    // gate-to-floor headroom at every horizon; real-bug signals scale
    // with N too (they compound through the same loop), so the
    // floor-to-bug separation is horizon-invariant. The RELATIVE band
    // scales identically: the KW78 radiative faces (SCIENCE.md
    // M17) gave boundary cells per-substep recursive state (face ->
    // boundary-cell pressure -> next substep's momentum -> face), so
    // systematic per-substep rounding now compounds in the relative
    // measure too - observed 1.8e-4 over an 8-substep horizon at a
    // radiated north-face cell, i.e. ~2.3e-5 per substep against the
    // 1e-4 single-application band.
    switch (tier)
    {
        case Tier::Pointwise:           return {1e-5, 1e-6};
        case Tier::Stencil:             return {1e-4, 5e-5};
        case Tier::Microphysics:        return {1e-4, 1e-7};
        case Tier::MicrophysicsHeating: return {1e-4, 6e-4};
    }
    return {1e-5, 1e-6};
}

const char* tier_name(Tier tier)
{
    switch (tier)
    {
        case Tier::Pointwise:           return "pointwise";
        case Tier::Stencil:             return "stencil";
        case Tier::Microphysics:        return "microphys";
        case Tier::MicrophysicsHeating: return "mp-heating";
    }
    return "unknown";
}

const char* domain_name(ComparisonDomain domain)
{
    switch (domain)
    {
        case ComparisonDomain::StrictInterior: return "interior";
        case ComparisonDomain::FullField:      return "full";
        case ComparisonDomain::BoundaryShell:  return "boundary-shell";
        case ComparisonDomain::WestFace:       return "west-face";
        case ComparisonDomain::EastFace:       return "east-face";
        case ComparisonDomain::SouthFace:      return "south-face";
        case ComparisonDomain::NorthFace:      return "north-face";
        case ComparisonDomain::BottomFace:     return "bottom-face";
        case ComparisonDomain::TopFace:        return "top-face";
    }
    return "invalid";
}

FieldComparison compare_field(const float* cpu, const float* gpu,
                              int nr, int nth, int nz,
                              Tolerance tolerance,
                              ComparisonDomain domain)
{
    FieldComparison result;
    DomainBounds bounds = bounds_for(domain, nr, nth, nz);
    if (bounds.invalid_reason == nullptr
        && (!std::isfinite(tolerance.abs) || tolerance.abs < 0.0
            || !std::isfinite(tolerance.rel) || tolerance.rel < 0.0))
    {
        bounds.invalid_reason = "invalid_tolerance";
    }
    if (bounds.invalid_reason == nullptr && (cpu == nullptr || gpu == nullptr))
    {
        bounds.invalid_reason = "null_input";
    }
    result.invalid_reason = bounds.invalid_reason;

    for (int i = bounds.i_begin;
         result.invalid_reason == nullptr && i < bounds.i_end; ++i)
    {
        for (int j = bounds.j_begin; j < bounds.j_end; ++j)
        {
            for (int k = bounds.k_begin; k < bounds.k_end; ++k)
            {
                if (domain == ComparisonDomain::BoundaryShell
                    && i > 0 && i < nr - 1
                    && j > 0 && j < nth - 1
                    && k > 0 && k < nz - 1)
                {
                    continue;
                }
                const std::size_t idx =
                    (static_cast<std::size_t>(i) * static_cast<std::size_t>(nth)
                     + static_cast<std::size_t>(j))
                    * static_cast<std::size_t>(nz) + static_cast<std::size_t>(k);
                const double c = static_cast<double>(cpu[idx]);
                const double g = static_cast<double>(gpu[idx]);
                ++result.cells_compared;

                // Matching NaNs or infinities are invalid scientific state,
                // not agreement.  Record the first such cell as the worst
                // location so callers never lose the diagnostic to NaN's
                // unordered comparison semantics.
                if (!std::isfinite(c) || !std::isfinite(g))
                {
                    ++result.nonfinite_cells;
                    if (result.nonfinite_cells == 1)
                    {
                        result.max_abs = std::numeric_limits<double>::infinity();
                        result.max_rel = std::numeric_limits<double>::infinity();
                        result.worst_i = i;
                        result.worst_j = j;
                        result.worst_k = k;
                        result.worst_cpu = static_cast<float>(c);
                        result.worst_gpu = static_cast<float>(g);
                    }
                    continue;
                }

                const double abs_diff = std::abs(g - c);
                if (abs_diff > tolerance.abs + tolerance.rel * std::abs(c))
                {
                    ++result.tolerance_breaches;
                }
                if (abs_diff > result.max_abs)
                {
                    result.max_abs = abs_diff;
                    result.max_rel = (std::abs(c) > 1e-300)
                        ? abs_diff / std::abs(c) : abs_diff;
                    result.worst_i = i;
                    result.worst_j = j;
                    result.worst_k = k;
                    result.worst_cpu = static_cast<float>(c);
                    result.worst_gpu = static_cast<float>(g);
                }
            }
        }
    }

    if (result.invalid_reason == nullptr && result.cells_compared == 0)
    {
        result.invalid_reason = "empty_domain";
    }
    result.passed = result.invalid_reason == nullptr
                 && result.nonfinite_cells == 0
                 && result.tolerance_breaches == 0;
    return result;
}

}  // namespace nimbus::parity
