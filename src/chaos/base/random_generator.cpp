/**
 * @file random_generator.cpp
 * @brief Implementation for the chaos module.
 *
 * Provides executable logic for the chaos runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/chaos subsystem.
 */

#include "random_generator.hpp"
#include "terrain/terrain_base.hpp"
#include "turbulence/turbulence_base.hpp"
#include "util/log.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace tmv::chaos
{

namespace {

double vertical_taper_factor(const std::string& taper_id,
                             double height_agl, double z1, double z2)
{
    if (taper_id == "none") return 1.0;
    if (taper_id == "pbl_only") return height_agl <= z1 ? 1.0 : 0.0;
    if (taper_id != "cosine")
        throw std::invalid_argument("unknown chaos vertical taper '" + taper_id + "'");
    if (height_agl <= z1) return 1.0;
    if (height_agl >= z2) return 0.0;
    if (!(z2 > z1)) return height_agl <= z1 ? 1.0 : 0.0;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double phase = pi * (height_agl - z1) / (z2 - z1);
    return 0.5 * (1.0 + std::cos(phase));
}

uint64_t mix_field_key(uint64_t stream_key, const std::string& field_name)
{
    uint64_t field_key = stream_key;
    for (char c : field_name)
    {
        field_key = field_key * 31ULL + static_cast<uint64_t>(c);
    }
    return field_key;
}

}

ChaosRNG::ChaosRNG(uint64_t base_seed, int member_id)
    : base_seed_(base_seed), member_id_(member_id),
      uniform_dist_(0.0, 1.0)
{
    base_generator_.seed(base_seed_ + static_cast<uint64_t>(member_id_));
}

std::vector<double> ChaosRNG::normal_stream(size_t count, uint64_t stream_key)
{
    return normal_stream(count, stream_key, 0);
}

std::vector<double> ChaosRNG::normal_stream(
    size_t count,
    uint64_t stream_key,
    uint64_t sequence_key
)
{
    std::vector<double> result(count);

    const uint64_t stream_seed = make_stream_seed(stream_key, sequence_key);
    auto generator = create_stream_generator(stream_seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (size_t i = 0; i < count; ++i)
    {
        result[i] = distribution(generator);
    }

    return result;
}

double ChaosRNG::normal(uint64_t stream_key)
{
    return normal(stream_key, 0);
}

double ChaosRNG::normal(uint64_t stream_key, uint64_t sequence_key)
{
    const uint64_t stream_seed = make_stream_seed(stream_key, sequence_key);
    auto generator = create_stream_generator(stream_seed);
    std::normal_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}

std::vector<double> ChaosRNG::uniform_stream(size_t count, uint64_t stream_key)
{
    std::vector<double> result(count);

    uint64_t stream_seed = make_stream_seed(stream_key, 0);
    auto generator = create_stream_generator(stream_seed);

    for (size_t i = 0; i < count; ++i)
    {
        result[i] = uniform_dist_(generator);
    }

    return result;
}

double ChaosRNG::uniform(uint64_t stream_key)
{
    uint64_t stream_seed = make_stream_seed(stream_key, 0);
    auto generator = create_stream_generator(stream_seed);
    return uniform_dist_(generator);
}

void ChaosRNG::fill_field_normal(
    Field3D& field,
    uint64_t stream_key,
    const std::string& field_name
)
{
    fill_field_normal(field, stream_key, 0, field_name);
}

void ChaosRNG::fill_field_normal(
    Field3D& field,
    uint64_t stream_key,
    uint64_t sequence_key,
    const std::string& field_name
)
{
    if (field.empty())
    {
        return;
    }

    const uint64_t field_key = mix_field_key(stream_key, field_name);
    const uint64_t stream_seed = make_stream_seed(field_key, sequence_key);
    auto generator = create_stream_generator(stream_seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    float* values = field.data();
    const size_t total_points = field.size();
    for (size_t idx = 0; idx < total_points; ++idx)
    {
        values[idx] = static_cast<float>(distribution(generator));
    }
}

void ChaosRNG::reset()
{
    base_generator_.seed(base_seed_ + static_cast<uint64_t>(member_id_));
}

uint64_t ChaosRNG::make_stream_seed(uint64_t stream_key, uint64_t sub_key) const
{
    uint64_t combined = base_seed_;
    combined = combined * 6364136223846793005ULL + static_cast<uint64_t>(member_id_);
    combined = combined * 6364136223846793005ULL + stream_key;
    combined = combined * 6364136223846793005ULL + sub_key;
    return combined;
}

std::mt19937_64 ChaosRNG::create_stream_generator(uint64_t stream_seed) const
{
    std::mt19937_64 generator;
    generator.seed(stream_seed);
    return generator;
}

void bound_perturbation_field(
    Field3D& xi,
    double xi_max,
    bool use_tanh
)
{
    if (xi.empty())
    {
        return;
    }

    const double xi_max_eff = (std::isfinite(xi_max) && xi_max > 0.0) ? xi_max : 1.0;

    float* values = xi.data();
    const size_t total_points = xi.size();
    for (size_t idx = 0; idx < total_points; ++idx)
    {
        double val = static_cast<double>(values[idx]);
        if (use_tanh)
        {
            val = xi_max_eff * std::tanh(val / xi_max_eff);
        }
        else
        {
            val = std::max(-xi_max_eff, std::min(xi_max_eff, val));
        }
        values[idx] = static_cast<float>(val);
    }
}

void apply_vertical_taper(
    Field3D& xi,
    const std::vector<double>& z_levels,
    const std::string& taper_id,
    double z1,
    double z2
)
{
    if (xi.empty() || z_levels.empty())
    {
        return;
    }

    if (taper_id == "none")
    {
        return;
    }

    const int nr = xi.size_r();
    const int nth = xi.size_th();
    const int nz = xi.size_z();
    if (nz <= 0 || static_cast<size_t>(nz) > z_levels.size())
    {
        return;
    }

    for (int k = 0; k < nz; ++k)
    {
        const double z = z_levels[static_cast<size_t>(k)];
        const double taper_factor = vertical_taper_factor(
            taper_id, z, z1, z2);

        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                xi(i, j, k) = static_cast<float>(static_cast<double>(xi(i, j, k)) * taper_factor);
            }
        }
    }
}

void apply_vertical_taper(
    Field3D& xi,
    const GridMetrics& grid,
    const std::string& taper_id,
    double z1,
    double z2)
{
    if (xi.empty() || taper_id == "none") return;

    const int nr = xi.size_r();
    const int nth = xi.size_th();
    const int nz = xi.size_z();
    if (!grid.terrain_metrics_active)
    {
        std::vector<double> z_levels(static_cast<std::size_t>(nz), 0.0);
        if (grid.z_int.size() >= static_cast<std::size_t>(nz + 1))
        {
            for (int k = 0; k < nz; ++k)
                z_levels[static_cast<std::size_t>(k)] = 0.5 * (
                    grid.z_int[static_cast<std::size_t>(k)]
                    + grid.z_int[static_cast<std::size_t>(k + 1)]);
        }
        else
        {
            if (grid.dz.size() < static_cast<std::size_t>(nz))
                throw std::invalid_argument(
                    "chaos vertical taper requires complete flat-grid heights");
            double lower = 0.0;
            for (int k = 0; k < nz; ++k)
            {
                const double thickness = grid.dz[static_cast<std::size_t>(k)];
                if (!std::isfinite(thickness) || thickness <= 0.0)
                    throw std::invalid_argument(
                        "chaos vertical taper requires positive finite layer thicknesses");
                z_levels[static_cast<std::size_t>(k)] = lower + 0.5 * thickness;
                lower += thickness;
            }
        }
        apply_vertical_taper(xi, z_levels, taper_id, z1, z2);
        return;
    }

    const TerrainMetrics3D* metrics = grid.terrain_metrics;
    if (metrics == nullptr
        || metrics->z.size_r() != nr || metrics->z.size_th() != nth
        || metrics->z.size_z() != nz || metrics->z_face.size_r() != nr
        || metrics->z_face.size_th() != nth
        || metrics->z_face.size_z() != nz + 1)
    {
        throw std::invalid_argument(
            "chaos vertical taper requires complete terrain physical heights");
    }

    // Validate the complete geometry before mutating xi so malformed terrain
    // remains transactional and fail-closed.
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            const double ground = metrics->z_face(i, j, 0);
            if (!std::isfinite(ground))
                throw std::invalid_argument(
                    "chaos vertical taper requires finite terrain height");
            for (int k = 0; k < nz; ++k)
            {
                const double height_agl = metrics->z(i, j, k) - ground;
                if (!std::isfinite(height_agl) || height_agl <= 0.0)
                    throw std::invalid_argument(
                        "chaos vertical taper requires positive finite above-ground heights");
            }
        }

#pragma omp parallel for collapse(2) schedule(static) \
    if(static_cast<long long>(nr) * nth * nz >= 4096)
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            const double ground = metrics->z_face(i, j, 0);
            for (int k = 0; k < nz; ++k)
            {
                const double factor = vertical_taper_factor(
                    taper_id, metrics->z(i, j, k) - ground, z1, z2);
                xi(i, j, k) = static_cast<float>(
                    static_cast<double>(xi(i, j, k)) * factor);
            }
        }
}

void apply_horizontal_mask(
    Field3D& xi,
    const std::vector<std::vector<double>>& mask
)
{
    if (xi.empty() || mask.empty() || mask[0].empty())
    {
        return;
    }

    const int nr = xi.size_r();
    const int nth = xi.size_th();
    const int nz = xi.size_z();

    if (mask.size() != static_cast<size_t>(nr) ||
        mask[0].size() != static_cast<size_t>(nth))
    {
        tmv::log_warn("Mask dimensions don't match perturbation field");
        return;
    }

    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            const double mask_val = mask[static_cast<size_t>(i)][static_cast<size_t>(j)];
            for (int k = 0; k < nz; ++k)
            {
                xi(i, j, k) = static_cast<float>(static_cast<double>(xi(i, j, k)) * mask_val);
            }
        }
    }
}

}
