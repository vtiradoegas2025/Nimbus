/**
 * @file thermodynamic_closure.cpp
 * @brief Initial-state pressure-controlled thermodynamic closure.
 */

#include "core/orchestration/dynamics/thermodynamic_closure.hpp"

#include "core/infra/physical_constants.hpp"
#include "core/runtime/simulation_state.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

void apply_initial_thermodynamic_closure(const SimulationStateView& state)
{
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk /
        physical_constants::specific_heat_cp_jkgk;

    const auto& grid = state.grid;
    auto& fields = state.fields;

    int invalid_count = 0;
    int out_of_range_count = 0;
    #pragma omp parallel for collapse(2) reduction(+:invalid_count,out_of_range_count)
    for (int i = 0; i < grid.nr; ++i)
    {
        for (int j = 0; j < grid.nth; ++j)
        {
            for (int k = 0; k < grid.nz; ++k)
            {
                const double pressure = static_cast<double>(fields.p[i][j][k]);
                const double potential_temperature =
                    static_cast<double>(fields.theta[i][j][k]);
                if (!std::isfinite(pressure) || pressure <= 0.0 ||
                    !std::isfinite(potential_temperature) ||
                    potential_temperature <= 0.0)
                {
                    ++invalid_count;
                    continue;
                }

                const double temperature = potential_temperature * std::pow(
                    pressure / physical_constants::reference_pressure_pa,
                    kappa);
                const double diagnosed_density = pressure /
                    (physical_constants::gas_constant_dry_air_jkgk * temperature);
                if (!std::isfinite(diagnosed_density) ||
                    diagnosed_density < 0.01 || diagnosed_density > 5.0)
                {
                    ++out_of_range_count;
                }
            }
        }
    }

    if (invalid_count > 0 || out_of_range_count > 0)
    {
        throw std::runtime_error(
            "initial thermodynamic closure rejected " +
            std::to_string(invalid_count) +
            " cells with invalid pressure/theta and " +
            std::to_string(out_of_range_count) +
            " cells with diagnosed density outside [0.01, 5] kg/m^3");
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid.nr; ++i)
    {
        for (int j = 0; j < grid.nth; ++j)
        {
            for (int k = 0; k < grid.nz; ++k)
            {
                const double pressure = static_cast<double>(fields.p[i][j][k]);
                const double potential_temperature =
                    static_cast<double>(fields.theta[i][j][k]);
                const double temperature = potential_temperature * std::pow(
                    pressure / physical_constants::reference_pressure_pa,
                    kappa);
                const double diagnosed_density = pressure /
                    (physical_constants::gas_constant_dry_air_jkgk * temperature);
                fields.rho[i][j][k] = static_cast<float>(diagnosed_density);
            }
        }
    }
}
