#pragma once

/**
 * @file parity_types.hpp
 * @brief Lightweight scientific comparison types shared by manifests and the
 *        runtime parity harness.
 */

namespace nimbus::parity
{

enum class Tier
{
    // Pointwise: abs <= 1e-6 + 1e-5*|reference|.
    Pointwise,
    // Stencils combine high-order terms and pressure-gradient rounding.
    Stencil,
    // Microphysics permits a calibrated finite branch-edge allowance.
    Microphysics,
    // Heating includes latent-heat/Exner amplification.
    MicrophysicsHeating,
};

// Domains are contractual: using the wrong one can hide boundary drift.
enum class ComparisonDomain
{
    StrictInterior,
    FullField,
    BoundaryShell, // Disjoint union of all six faces.
    WestFace,
    EastFace,
    SouthFace,
    NorthFace,
    BottomFace,
    TopFace,
};

struct Tolerance
{
    double rel;
    double abs;
};

} // namespace nimbus::parity
