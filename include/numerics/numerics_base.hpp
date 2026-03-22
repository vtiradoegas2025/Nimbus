#pragma once

#include <functional>
#include <string>
#include <vector>

#include "core/field3d.hpp"

struct GridMetrics;

/**
 * @file numerics_base.hpp
 * @brief Common numerics interfaces shared by advection, diffusion, and time stepping.
 *
 * Defines foundational data containers and the base polymorphic
 * interface used by numerical schemes.
 * This header keeps cross-module numerics contracts in one place.
 */

namespace numerics_constants
{
inline constexpr double epsilon = 1e-12;
inline constexpr double weno_epsilon = 1e-6;
inline constexpr double cfl_target = 0.8;
}

struct NumericalState
{
    Field3D data;
    std::string name;
};

struct NumericalTendencies
{
    Field3D tendencies;
    std::string name;
};

using RHSFunction =
    std::function<void(const std::vector<NumericalState>&, double, std::vector<NumericalTendencies>&)>;

class NumericalSchemeBase
{
public:
    /**
     * @brief Virtual destructor for polymorphic cleanup.
     */
    virtual ~NumericalSchemeBase() = default;

    /**
     * @brief Returns the scheme identifier used for diagnostics.
     * @return Scheme name.
     */
    virtual std::string name() const = 0;

    /**
     * @brief Performs scheme-specific initialization.
     */
    virtual void initialize() = 0;
};

extern GridMetrics global_grid_metrics;
