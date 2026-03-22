/**
 * @file perturbation_field.cpp
 * @brief Implementation for the chaos module.
 *
 * Provides executable logic for the chaos runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/chaos subsystem.
 *
 * All perturbation operations use contiguous Field3D storage exclusively.
 */

#include "perturbation_field.hpp"
#include "util/simd_utils.hpp"
#include <cmath>
#include <algorithm>

namespace chaos
{

Field3D generate_white_noise_field3d(
    ChaosRNG& rng,
    int NR,
    int NTH,
    int NZ,
    uint64_t stream_key,
    const std::string& field_name
)
{
    Field3D field(NR, NTH, NZ);
    rng.fill_field_normal(field, stream_key, field_name);
    return field;
}

void scale_perturbation_field(Field3D& field, double sigma)
{
    if (field.empty())
    {
        return;
    }

    float* values = field.data();
    const int count = static_cast<int>(field.size());
    simd_utils::scale_vectors(static_cast<float>(sigma), values, values, count);
}

double renormalize_to_unit_variance(Field3D& field)
{
    if (field.empty())
    {
        return 0.0;
    }

    const size_t total_points = field.size();
    const float* values = field.data();
    double sum = 0.0;
    double sum_sq = 0.0;
    for (size_t idx = 0; idx < total_points; ++idx)
    {
        const double val = static_cast<double>(values[idx]);
        sum += val;
        sum_sq += val * val;
    }

    const double mean = sum / static_cast<double>(total_points);
    const double variance = (sum_sq / static_cast<double>(total_points)) - (mean * mean);
    if (variance <= 1e-12)
    {
        return variance;
    }

    const double scale_factor = 1.0 / std::sqrt(variance);
    float* writable = field.data();
    for (size_t idx = 0; idx < total_points; ++idx)
    {
        const double centered = static_cast<double>(writable[idx]) - mean;
        writable[idx] = static_cast<float>(centered * scale_factor);
    }

    return variance;
}

std::tuple<double, double, double, double> compute_field_statistics(const Field3D& field)
{
    if (field.empty())
    {
        return {0.0, 0.0, 0.0, 0.0};
    }

    const int total_points = static_cast<int>(field.size());
    const float* values = field.data();

    // Min/max via SIMD (float-precision is exact for min/max)
    float fmin, fmax;
    simd_utils::reduce_min_max(values, total_points, &fmin, &fmax);

    // Sum and sum_sq in double precision for accumulation accuracy
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int idx = 0; idx < total_points; ++idx)
    {
        const double val = static_cast<double>(values[idx]);
        sum += val;
        sum_sq += val * val;
    }

    const double mean = sum / static_cast<double>(total_points);
    const double variance = (sum_sq / static_cast<double>(total_points)) - (mean * mean);

    return {mean, variance, static_cast<double>(fmin), static_cast<double>(fmax)};
}

void evolve_ar1_3d(
    Field3D& xi,
    const Field3D& xi_prev,
    double rho_t,
    ChaosRNG& rng,
    uint64_t stream_key,
    uint64_t time_step
)
{
    if (xi.empty() || xi_prev.empty() ||
        xi.size_r() != xi_prev.size_r() ||
        xi.size_th() != xi_prev.size_th() ||
        xi.size_z() != xi_prev.size_z())
    {
        return;
    }

    const size_t total_points = xi.size();
    const auto innovation = rng.normal_stream(total_points, stream_key + time_step);
    const double innovation_scale = std::sqrt(std::max(0.0, 1.0 - rho_t * rho_t));

    float* current = xi.data();
    const float* previous = xi_prev.data();
    for (size_t idx = 0; idx < total_points; ++idx)
    {
        current[idx] = static_cast<float>(
            rho_t * static_cast<double>(previous[idx]) +
            innovation_scale * innovation[idx]
        );
    }
}

double compute_temporal_correlation(double dt, double tau_t)
{
    if (tau_t <= 0.0)
    {
        return 0.0;
    }
    return std::exp(-dt / tau_t);
}

}
