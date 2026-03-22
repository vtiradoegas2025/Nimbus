/**
 * @file random_generator.cpp
 * @brief Implementation for the chaos module.
 *
 * Provides executable logic for the chaos runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/chaos subsystem.
 */

#include "random_generator.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace chaos
{

namespace {

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
      normal_dist_(0.0, 1.0), uniform_dist_(0.0, 1.0)
{
    base_generator_.seed(base_seed_ + static_cast<uint64_t>(member_id_));
}

std::vector<double> ChaosRNG::normal_stream(size_t count, uint64_t stream_key)
{
    std::vector<double> result(count);

    uint64_t stream_seed = make_stream_seed(stream_key, 0);
    auto generator = create_stream_generator(stream_seed);

    for (size_t i = 0; i < count; ++i)
    {
        result[i] = normal_dist_(generator);
    }

    return result;
}

double ChaosRNG::normal(uint64_t stream_key)
{
    uint64_t stream_seed = make_stream_seed(stream_key, 0);
    auto generator = create_stream_generator(stream_seed);
    return normal_dist_(generator);
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
    if (field.empty())
    {
        return;
    }

    const uint64_t field_key = mix_field_key(stream_key, field_name);
    const uint64_t stream_seed = make_stream_seed(field_key, 0);
    auto generator = create_stream_generator(stream_seed);

    float* values = field.data();
    const size_t total_points = field.size();
    for (size_t idx = 0; idx < total_points; ++idx)
    {
        values[idx] = static_cast<float>(normal_dist_(generator));
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
        double taper_factor = 1.0;

        if (taper_id == "pbl_only")
        {
            taper_factor = (z <= z1) ? 1.0 : 0.0;
        }
        else if (taper_id == "cosine")
        {
            if (z <= z1)
            {
                taper_factor = 1.0;
            }
            else if (z >= z2)
            {
                taper_factor = 0.0;
            }
            else
            {
                if (z2 > z1)
                {
                    const double phase = M_PI * (z - z1) / (z2 - z1);
                    taper_factor = 0.5 * (1.0 + std::cos(phase));
                }
                else
                {
                    taper_factor = (z <= z1) ? 1.0 : 0.0;
                }
            }
        }

        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                xi(i, j, k) = static_cast<float>(static_cast<double>(xi(i, j, k)) * taper_factor);
            }
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
        std::cerr << "Warning: Mask dimensions don't match perturbation field" << std::endl;
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
