/**
 * @file velocity_staging.hpp
 * @brief Dependency plan for C-grid face-to-center output staging.
 */

#pragma once

#include "core/output/field_registry.hpp"

struct CenteredVelocityStagingPlan
{
    bool u = false;
    bool v = false;
    bool w = false;

    int field_count() const noexcept
    {
        return static_cast<int>(u) + static_cast<int>(v) + static_cast<int>(w);
    }
};

inline CenteredVelocityStagingPlan centered_velocity_staging_plan(
    bool c_grid,
    OutputFieldDependency dependencies)
{
    if (!c_grid)
    {
        return {};
    }
    return {
        output_field_has_dependency(dependencies, OutputFieldDependency::CenteredU),
        output_field_has_dependency(dependencies, OutputFieldDependency::CenteredV),
        output_field_has_dependency(dependencies, OutputFieldDependency::CenteredW),
    };
}
