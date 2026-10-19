#pragma once

#include "rrtmg_atmosphere.hpp"
#include "rrtmg_coefficients.hpp"
#include "rrtmg_preprocessing.hpp"

#include <cstddef>
#include <string>
#include <utility>

namespace tmv::rrtmg::optics_internal
{

struct KeySpecies
{
    const GasProfile* first           = nullptr;
    const GasProfile* second          = nullptr;
    std::size_t       first_absorber  = 0;
    std::size_t       second_absorber = 0;
};

const GasProfile* find_gas(const PreparedAtmosphere& atmosphere, const std::string& name) noexcept;

std::size_t absorber_index(const SpectralGrid& grid, const std::string& name);

KeySpecies key_species(const SpectralGrid&       grid,
                       const PreparedAtmosphere& atmosphere,
                       const std::string&        first,
                       const std::string&        second);

double interpolate_key_absorption(const CoefficientArray&   table,
                                  std::size_t               bands,
                                  std::size_t               pressure_count,
                                  std::size_t               ratio_count,
                                  std::size_t               band,
                                  std::size_t               local_g_point,
                                  const LayerInterpolation& interpolation,
                                  const KeySpecies&         species,
                                  const CoefficientArray*   reference_amounts,
                                  std::size_t               layer,
                                  double                    fixed_reference_ratio    = 1.0,
                                  bool                      fourth_order_ratio_edges = true);

double interpolate_ratio_table(const CoefficientArray& table,
                               std::size_t             bands,
                               std::size_t             ratio_count,
                               std::size_t             band,
                               std::size_t             local_g_point,
                               double                  eta);

double interpolate_continuum(const CoefficientArray& table,
                             std::size_t             bands,
                             std::size_t             temperatures,
                             std::size_t             band,
                             std::size_t             local_g_point,
                             std::size_t             temperature_index,
                             double                  temperature_fraction);

double interpolate_minor_absorption(const CoefficientArray& table,
                                    std::size_t             bands,
                                    std::size_t             ratio_count,
                                    std::size_t             band,
                                    std::size_t             absorber,
                                    std::size_t             local_g_point,
                                    std::size_t             temperature_index,
                                    double                  temperature_fraction,
                                    double                  eta);

double minor_cross_section(const CoefficientArray& table,
                           std::size_t             bands,
                           std::size_t             band,
                           std::size_t             absorber,
                           std::size_t             local_g_point,
                           std::size_t             ratio_count = 9);

double longwave_minor_optical_depth(const LongwaveCoefficients& coefficients,
                                    const PreparedAtmosphere&   atmosphere,
                                    const LayerInterpolation&   interpolation,
                                    std::size_t                 band,
                                    std::size_t                 local_g_point,
                                    std::size_t                 layer);

double scaled_column(const GasProfile& gas, std::size_t layer) noexcept;

void require_finite_nonnegative(double value, const char* quantity);

} // namespace tmv::rrtmg::optics_internal
