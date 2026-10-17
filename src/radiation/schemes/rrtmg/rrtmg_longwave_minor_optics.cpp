#include "rrtmg_optics_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tmv::rrtmg::optics_internal
{
namespace
{

constexpr std::size_t absorber_n2    = 0;
constexpr std::size_t absorber_ccl4  = 1;
constexpr std::size_t absorber_cfc11 = 2;
constexpr std::size_t absorber_cfc12 = 3;
constexpr std::size_t absorber_cfc22 = 4;
constexpr std::size_t absorber_h2o   = 5;
constexpr std::size_t absorber_co2   = 6;
constexpr std::size_t absorber_o3    = 7;
constexpr std::size_t absorber_n2o   = 8;
constexpr std::size_t absorber_co    = 9;
constexpr std::size_t absorber_ch4   = 10;
constexpr std::size_t absorber_o2    = 11;
constexpr double      column_scale   = 1.0e-20;
constexpr double      one_minus      = 1.0 - 1.0e-12;

const GasProfile& gas_profile(const PreparedAtmosphere& atmosphere, std::size_t absorber)
{
    switch (absorber)
    {
    case absorber_n2:
        return atmosphere.gases.nitrogen;
    case absorber_ccl4:
        return atmosphere.gases.ccl4;
    case absorber_cfc11:
        return atmosphere.gases.cfc11;
    case absorber_cfc12:
        return atmosphere.gases.cfc12;
    case absorber_cfc22:
        return atmosphere.gases.cfc22;
    case absorber_h2o:
        return atmosphere.gases.water_vapor;
    case absorber_co2:
        return atmosphere.gases.carbon_dioxide;
    case absorber_o3:
        return atmosphere.gases.ozone;
    case absorber_n2o:
        return atmosphere.gases.nitrous_oxide;
    case absorber_co:
        return atmosphere.gases.carbon_monoxide;
    case absorber_ch4:
        return atmosphere.gases.methane;
    case absorber_o2:
        return atmosphere.gases.oxygen;
    default:
        throw std::logic_error("RRTMG minor absorber index is invalid");
    }
}

double reference_vmr(const LongwaveCoefficients& coefficients, std::size_t pressure, std::size_t absorber)
{
    constexpr std::size_t absorbers = 12;
    if (coefficients.absorber_amount_mls.shape != std::vector<std::size_t>{59, absorbers} ||
        coefficients.absorber_amount_mls.values.size() != 59 * absorbers || pressure >= 59 || absorber >= absorbers)
        throw std::invalid_argument("RRTMG reference-atmosphere table is malformed");
    const double value = coefficients.absorber_amount_mls.values[pressure * absorbers + absorber];
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument("RRTMG reference-atmosphere value is invalid");
    return value;
}

double mixture_eta(const LongwaveCoefficients& coefficients,
                   const PreparedAtmosphere&   atmosphere,
                   std::size_t                 layer,
                   std::size_t                 first_absorber,
                   std::size_t                 second_absorber,
                   std::size_t                 mapping_pressure)
{
    const double first_reference  = reference_vmr(coefficients, mapping_pressure, first_absorber);
    const double second_reference = reference_vmr(coefficients, mapping_pressure, second_absorber);
    if (!(second_reference > 0.0))
        throw std::invalid_argument("RRTMG minor-gas reference ratio is invalid");
    const double first  = scaled_column(gas_profile(atmosphere, first_absorber), layer);
    const double second = scaled_column(gas_profile(atmosphere, second_absorber), layer);
    const double total  = first + (first_reference / second_reference) * second;
    return total > 0.0 ? std::clamp(first / total, 0.0, one_minus) : 0.0;
}

double adjusted_column(const LongwaveCoefficients& coefficients,
                       const PreparedAtmosphere&   atmosphere,
                       const LayerInterpolation&   interpolation,
                       std::size_t                 layer,
                       std::size_t                 absorber,
                       double                      threshold,
                       double                      offset,
                       double                      exponent,
                       double                      adjustment_reference = 0.0)
{
    const GasProfile& gas        = gas_profile(atmosphere, absorber);
    const double      current    = scaled_column(gas, layer);
    const double      dry_column = atmosphere.dry_air_column_molecules_cm2[layer];
    if (!(dry_column > 0.0))
        throw std::invalid_argument("RRTMG dry-air column is invalid");
    const std::size_t reference_pressure  = std::min(interpolation.pressure_index + 1, std::size_t{58});
    const double      tabulated_reference = reference_vmr(coefficients, reference_pressure, absorber);
    const double      ratio_reference     = adjustment_reference > 0.0 ? adjustment_reference : tabulated_reference;
    if (!(ratio_reference > 0.0))
        return current;
    const double actual_vmr = gas.column_molecules_cm2[layer] / dry_column;
    const double ratio      = actual_vmr / ratio_reference;
    if (!(ratio > threshold))
        return current;
    const double factor = offset + std::pow(ratio - offset, exponent);
    return factor * ratio_reference * dry_column * column_scale;
}

double broadening_fraction(const PreparedAtmosphere& atmosphere, std::size_t layer)
{
    const auto&  gases    = atmosphere.gases;
    const double declared = gases.carbon_dioxide.volume_mixing_ratio[layer] + gases.ozone.volume_mixing_ratio[layer] +
                            gases.nitrous_oxide.volume_mixing_ratio[layer] +
                            gases.carbon_monoxide.volume_mixing_ratio[layer] +
                            gases.methane.volume_mixing_ratio[layer] + gases.oxygen.volume_mixing_ratio[layer];
    return std::max(0.0, 1.0 - declared);
}

} // namespace

double longwave_minor_optical_depth(const LongwaveCoefficients& coefficients,
                                    const PreparedAtmosphere&   atmosphere,
                                    const LayerInterpolation&   interpolation,
                                    std::size_t                 band,
                                    std::size_t                 local_g_point,
                                    std::size_t                 layer)
{
    constexpr std::size_t bands = 16;
    if (band >= bands || local_g_point >= 16 || layer >= atmosphere.layer_pressure_pa.size())
        throw std::invalid_argument("RRTMG longwave minor-gas coordinates are invalid");
    const bool              lower       = interpolation.lower_atmosphere;
    const std::size_t       ratio_count = lower ? 9 : 5;
    const CoefficientArray& table = lower ? coefficients.minor_absorption_lower : coefficients.minor_absorption_upper;
    const auto              coefficient = [&](std::size_t absorber, double eta)
    {
        return interpolate_minor_absorption(table,
                                            bands,
                                            ratio_count,
                                            band,
                                            absorber,
                                            local_g_point,
                                            interpolation.minor_temperature_index,
                                            interpolation.minor_temperature_fraction,
                                            eta);
    };
    const auto cross_section = [&](std::size_t absorber)
    {
        // Cross sections occupy the first temperature/ratio cell and are
        // pressure independent in the AER coefficient schema.
        return minor_cross_section(coefficients.minor_absorption_lower, bands, band, absorber, local_g_point);
    };
    const auto column = [&](std::size_t absorber) { return scaled_column(gas_profile(atmosphere, absorber), layer); };

    double optical_depth = 0.0;
    switch (band)
    {
    case 0: // N2, both regimes.
        optical_depth = atmosphere.dry_air_column_molecules_cm2[layer] * column_scale *
                        broadening_fraction(atmosphere, layer) * interpolation.nitrogen_minor_scale *
                        coefficient(absorber_n2, 0.0);
        break;
    case 2: // N2O mapped on H2O/CO2.
    {
        const double eta = mixture_eta(coefficients, atmosphere, layer, absorber_h2o, absorber_co2, lower ? 2 : 12);
        optical_depth = adjusted_column(coefficients, atmosphere, interpolation, layer, absorber_n2o, 1.5, 0.5, 0.65) *
                        coefficient(absorber_n2o, eta);
        break;
    }
    case 4: // O3 plus the CCl4 cross section.
        if (lower)
        {
            const double eta = mixture_eta(coefficients, atmosphere, layer, absorber_h2o, absorber_co2, 6);
            optical_depth += column(absorber_o3) * coefficient(absorber_o3, eta);
        }
        optical_depth += column(absorber_ccl4) * cross_section(absorber_ccl4);
        break;
    case 5: // CO2 below the tropopause; CFC11/CFC12 in both regimes.
        if (lower)
            optical_depth +=
                adjusted_column(coefficients, atmosphere, interpolation, layer, absorber_co2, 3.0, 2.0, 0.77) *
                coefficient(absorber_co2, 0.0);
        optical_depth += column(absorber_cfc11) * cross_section(absorber_cfc11);
        optical_depth += column(absorber_cfc12) * cross_section(absorber_cfc12);
        break;
    case 6: // CO2 mapped on H2O/O3 below; unmixed above.
    {
        const double eta = lower ? mixture_eta(coefficients, atmosphere, layer, absorber_h2o, absorber_o3, 2) : 0.0;
        optical_depth =
            adjusted_column(
                coefficients, atmosphere, interpolation, layer, absorber_co2, 3.0, lower ? 3.0 : 2.0, 0.79) *
            coefficient(absorber_co2, eta);
        break;
    }
    case 7: // CO2/O3/N2O plus CFC12/CFC22 cross sections.
        optical_depth += adjusted_column(coefficients, atmosphere, interpolation, layer, absorber_co2, 3.0, 2.0, 0.65) *
                         coefficient(absorber_co2, 0.0);
        if (lower)
            optical_depth += column(absorber_o3) * coefficient(absorber_o3, 0.0);
        optical_depth += column(absorber_n2o) * coefficient(absorber_n2o, 0.0);
        optical_depth += column(absorber_cfc12) * cross_section(absorber_cfc12);
        optical_depth += column(absorber_cfc22) * cross_section(absorber_cfc22);
        break;
    case 8: // N2O mapped on H2O/CH4 below; unmixed above.
    {
        const double eta = lower ? mixture_eta(coefficients, atmosphere, layer, absorber_h2o, absorber_ch4, 2) : 0.0;
        optical_depth = adjusted_column(coefficients, atmosphere, interpolation, layer, absorber_n2o, 1.5, 0.5, 0.65) *
                        coefficient(absorber_n2o, eta);
        break;
    }
    case 10: // O2 collision-induced absorption.
        optical_depth = column(absorber_o2) * interpolation.minor_scale * coefficient(absorber_o2, 0.0);
        break;
    case 12: // CO2/CO below; O3 above.
        if (lower)
        {
            const double co2_eta = mixture_eta(coefficients, atmosphere, layer, absorber_h2o, absorber_n2o, 0);
            const double co_eta  = mixture_eta(coefficients, atmosphere, layer, absorber_h2o, absorber_n2o, 2);
            optical_depth +=
                adjusted_column(coefficients, atmosphere, interpolation, layer, absorber_co2, 3.0, 2.0, 0.68, 3.55e-4) *
                coefficient(absorber_co2, co2_eta);
            optical_depth += column(absorber_co) * coefficient(absorber_co, co_eta);
        }
        else
        {
            optical_depth += column(absorber_o3) * coefficient(absorber_o3, 0.0);
        }
        break;
    case 14: // N2 mapped on N2O/CO2 below; no upper-atmosphere term.
        if (lower)
        {
            const double eta = mixture_eta(coefficients, atmosphere, layer, absorber_n2o, absorber_co2, 0);
            optical_depth = atmosphere.dry_air_column_molecules_cm2[layer] * column_scale * interpolation.minor_scale *
                            broadening_fraction(atmosphere, layer) * coefficient(absorber_n2, eta);
        }
        break;
    default:
        break;
    }
    require_finite_nonnegative(optical_depth, "longwave minor-gas optical depth");
    return optical_depth;
}

} // namespace tmv::rrtmg::optics_internal
