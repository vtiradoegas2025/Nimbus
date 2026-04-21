/**
 * @file thermodynamics.cpp
 * @brief Implementation for the microphysics module.
 *
 * Provides executable logic for the microphysics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/microphysics subsystem.
 */

#include "thermodynamics.hpp"
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Converts the potential temperature field to the temperature field.
 */
void thermodynamics::convert_theta_to_temperature_field(
    const Field3D& theta,
    const Field3D& p,
    Field3D& temperature
) 
{
    int NR = theta.size_r();
    if (NR == 0) return;
    int NTH = theta.size_th();
    if (NTH == 0) return;
    int NZ = theta.size_z();

    temperature.resize(NR, NTH, NZ, 0.0f);

    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                temperature[i][j][k] = static_cast<float>(
                    theta_to_temperature(static_cast<double>(theta[i][j][k]), static_cast<double>(p[i][j][k]))
                );
            }
        }
    }
}

/**
 * @brief Converts the temperature field to the potential temperature field.
 */
void thermodynamics::convert_temperature_to_theta_field(
    const Field3D& temperature,
    const Field3D& p,
    Field3D& theta
)
{
    int NR = temperature.size_r();
    if (NR == 0) return;
    int NTH = temperature.size_th();
    if (NTH == 0) return;
    int NZ = temperature.size_z();

    theta.resize(NR, NTH, NZ, 0.0f);

    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                theta[i][j][k] = static_cast<float>(
                    temperature_to_theta(static_cast<double>(temperature[i][j][k]), static_cast<double>(p[i][j][k]))
                );
            }
        }
    }
}

/**
 * @brief Updates the density field from the equation of state.
 */
void thermodynamics::update_density_from_eos(
    const Field3D& p,
    const Field3D& theta,
    Field3D& rho)
{
    const int nr = p.size_r();
    if (nr == 0) return;
    const int nth = p.size_th();
    if (nth == 0) return;
    const int nz = p.size_z();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                rho[i][j][k] = static_cast<float>(
                    density_from_eos(
                        static_cast<double>(p[i][j][k]),
                        static_cast<double>(theta[i][j][k])));
            }
        }
    }
}
