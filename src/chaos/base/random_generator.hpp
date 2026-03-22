/**
 * @file random_generator.hpp
 * @brief Declarations for the chaos module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the chaos runtime and scheme implementations.
 * This file is part of the src/chaos subsystem.
 */

#pragma once
#include <random>
#include <cstdint>
#include <vector>
#include <string>
#include "core/field3d.hpp"

namespace chaos
{

/**
 * @brief Reproducible random number generator for chaos perturbations
 *
 * Provides deterministic random sequences based on seed, member_id, and stream keys.
 * Uses MT19937-64 for high-quality, fast random numbers with good statistical properties.
 */
class ChaosRNG
{
public:
    /**
     * @brief Initialize RNG with base seed and member ID
     * @param base_seed Base random seed
     * @param member_id Ensemble member identifier
     */
    ChaosRNG(uint64_t base_seed = 42, int member_id = 0);

    /**
     * @brief Generate a stream of standard normal random variables
     * @param count Number of samples to generate
     * @param stream_key Additional key for stream separation (field name, timestep, etc.)
     * @return Vector of N(0,1) random variables
     */
    std::vector<double> normal_stream(size_t count, uint64_t stream_key = 0);

    /**
     * @brief Generate a single standard normal random variable
     * @param stream_key Additional key for stream separation
     * @return Single N(0,1) random variable
     */
    double normal(uint64_t stream_key = 0);

    /**
     * @brief Generate uniform random variables in [0,1)
     * @param count Number of samples to generate
     * @param stream_key Additional key for stream separation
     * @return Vector of uniform random variables
     */
    std::vector<double> uniform_stream(size_t count, uint64_t stream_key = 0);

    /**
     * @brief Generate a single uniform random variable in [0,1)
     * @param stream_key Additional key for stream separation
     * @return Single uniform random variable
     */
    double uniform(uint64_t stream_key = 0);

    /**
     * @brief Fill a Field3D with independent normal random variables
     * @param field Output field to fill
     * @param stream_key Base stream key
     * @param field_name Field identifier for stream separation
     */
    void fill_field_normal(Field3D& field, uint64_t stream_key, const std::string& field_name = "");

    /**
     * @brief Reset RNG state (for reproducibility testing)
     */
    void reset();

private:
    uint64_t base_seed_;
    int member_id_;
    std::mt19937_64 base_generator_;
    std::normal_distribution<double> normal_dist_;
    std::uniform_real_distribution<double> uniform_dist_;

    /**
     * @brief Create a deterministic stream seed from components
     * @param stream_key Primary stream identifier
     * @param sub_key Secondary identifier (field, spatial index, etc.)
     * @return Deterministic 64-bit seed
     */
    uint64_t make_stream_seed(uint64_t stream_key, uint64_t sub_key = 0) const;

    /**
     * @brief Create a new generator instance for a specific stream
     * @param stream_seed Seed for this stream
     * @return New MT19937-64 generator instance
     */
    std::mt19937_64 create_stream_generator(uint64_t stream_seed) const;
};

/**
 * @brief Apply bounding to perturbation field
 * @param xi Input/output perturbation field
 * @param xi_max Maximum absolute value (clipping threshold)
 * @param use_tanh If true, use smooth tanh saturation; if false, use hard clipping
 */
void bound_perturbation_field(Field3D& xi, double xi_max, bool use_tanh = true);

/**
 * @brief Apply vertical tapering to 3D perturbation field
 * @param xi Input/output perturbation field
 * @param z_levels Vertical coordinate levels [nz]
 * @param taper_id Tapering function: "none", "pbl_only", "cosine"
 * @param z1 Lower taper height
 * @param z2 Upper taper height
 */
void apply_vertical_taper(Field3D& xi, const std::vector<double>& z_levels, const std::string& taper_id, double z1, double z2);

/**
 * @brief Apply horizontal mask to perturbation field
 * @param xi Input/output perturbation field
 * @param mask Mask field (0.0 = no perturbation, 1.0 = full perturbation)
 */
void apply_horizontal_mask(Field3D& xi, const std::vector<std::vector<double>>& mask);

}
