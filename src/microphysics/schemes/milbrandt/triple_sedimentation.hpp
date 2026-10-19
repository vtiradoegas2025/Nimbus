#pragma once

#include "triple_moment.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace tmv::milbrandt_detail
{

enum class TripleMomentSpecies : std::size_t
{
    Rain,
    Ice,
    Snow,
    Graupel,
    Hail
};

struct TripleMomentSedimentationParameters
{
    TripleMomentMassLaw mass_law{471.238898038, 3.0};
    double fall_coefficient = 206.89;
    double fall_exponent = 0.6384;
    double maximum_speed = 25.0;
    double maximum_diameter = 80.0e-3;
    int maximum_substeps = 4096;
};

struct TripleMomentSurfaceInventory
{
    double mass = 0.0;
    double number = 0.0;
    double reflectivity = 0.0;
};

using TripleHailSedimentationParameters = TripleMomentSedimentationParameters;
using TripleHailSurfaceInventory = TripleMomentSurfaceInventory;

TripleMomentSedimentationParameters
current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies species);

bool diagnose_triple_gamma_shape(double mass_per_kg,
                                 double number_per_kg,
                                 double reflectivity_per_kg,
                                 TripleMomentMassLaw mass_law,
                                 double& alpha,
                                 double& lambda,
                                 std::string& error) noexcept;

bool triple_moment_fall_speeds(double density,
                               double alpha,
                               double lambda,
                               const TripleMomentSedimentationParameters& parameters,
                               std::array<double, 3>& speeds) noexcept;

bool triple_hail_fall_speeds(double density,
                             double alpha,
                             double lambda,
                             const TripleHailSedimentationParameters& parameters,
                             std::array<double, 3>& speeds) noexcept;

bool sediment_triple_moment_column(int levels,
                                  const float* mass,
                                  const float* number,
                                  const float* reflectivity,
                                  const float* density,
                                  const double* thickness,
                                  double dt,
                                  const TripleMomentSedimentationParameters& parameters,
                                  float* mass_after,
                                  float* number_after,
                                  float* reflectivity_after,
                                  TripleMomentSurfaceInventory& surface,
                                  std::string& error);

bool sediment_triple_hail_column(int levels,
                                const float* mass,
                                const float* number,
                                const float* reflectivity,
                                const float* density,
                                const double* thickness,
                                double dt,
                                const TripleHailSedimentationParameters& parameters,
                                float* mass_after,
                                float* number_after,
                                float* reflectivity_after,
                                TripleHailSurfaceInventory& surface,
                                std::string& error);

} // namespace tmv::milbrandt_detail
