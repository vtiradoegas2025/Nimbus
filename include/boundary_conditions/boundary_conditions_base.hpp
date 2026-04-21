#pragma once

#include <string>

/**
 * @file boundary_conditions_base.hpp
 * @brief Abstract base for coordinate-specific boundary condition schemes.
 *
 * Implementations handle lateral, axis, and vertical BCs for their
 * coordinate system. Selected at startup via factory, called each step.
 */

class BoundaryConditionScheme
{
public:
    virtual ~BoundaryConditionScheme() = default;

    /// Apply full BCs to all prognostic fields (u, v, w, rho, p, theta, moisture).
    virtual void apply_full() = 0;

    /// Lightweight BCs for acoustic substeps (u, v, w, rho, p only).
    virtual void apply_acoustic() = 0;

    virtual std::string get_scheme_name() const = 0;
};
