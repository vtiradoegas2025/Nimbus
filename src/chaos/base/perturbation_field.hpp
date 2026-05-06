/**
 * @file perturbation_field.hpp
 * @brief Declarations for the chaos module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the chaos runtime and scheme implementations.
 * This file is part of the src/chaos subsystem.
 *
 * All perturbation operations use contiguous Field3D storage exclusively.
 */

#pragma once
#include "random_generator.hpp"
#include "core/field/field3d.hpp"
#include <string>
#include <tuple>

namespace chaos
{

/**
 * @brief Generate white noise perturbation field in contiguous Field3D layout
 * @param rng Random number generator
 * @param NR Radial grid points
 * @param NTH Azimuthal grid points
 * @param NZ Vertical grid points
 * @param stream_key Stream identifier for reproducibility
 * @param field_name Field identifier for stream separation
 * @return Field3D with N(0,1) values
 */
Field3D generate_white_noise_field3d(
    ChaosRNG& rng,
    int NR,
    int NTH,
    int NZ,
    uint64_t stream_key,
    const std::string& field_name = ""
);

/**
 * @brief Scale perturbation field by specified amplitude
 * @param field Input/output perturbation field
 * @param sigma Scaling factor (standard deviation)
 */
void scale_perturbation_field(Field3D& field, double sigma);

/**
 * @brief Ensure perturbation field has unit variance (renormalization)
 * @param field Input/output perturbation field
 * @return Realized variance of the field before renormalization
 */
double renormalize_to_unit_variance(Field3D& field);

/**
 * @brief Compute statistical properties of a Field3D perturbation field
 * @param field Perturbation field
 * @return Tuple of (mean, variance, min, max)
 */
std::tuple<double, double, double, double> compute_field_statistics(const Field3D& field);

/**
 * @brief Evolve perturbation field using AR(1) process
 * @param xi Current perturbation field (modified in-place)
 * @param xi_prev Previous perturbation field
 * @param rho_t Temporal correlation coefficient (exp(-dt/tau_t))
 * @param rng Random number generator for innovation term
 * @param stream_key Stream identifier for reproducibility
 * @param time_step Current time step counter
 */
void evolve_ar1_3d(
    Field3D& xi, const Field3D& xi_prev, double rho_t,
    ChaosRNG& rng, uint64_t stream_key, uint64_t time_step
);

/**
 * @brief Compute temporal correlation coefficient from decorrelation time
 * @param dt Timestep size
 * @param tau_t Decorrelation time
 * @return rho_t = exp(-dt/tau_t)
 */
double compute_temporal_correlation(double dt, double tau_t);

}
