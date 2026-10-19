#include "radiation/column_evaluation.hpp"

#include "core/infra/physical_constants.hpp"
#include "radiation/base/radiative_transfer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tmv
{
namespace
{

const std::vector<double>& require_vector(
    const std::vector<double>* values,
    const char* name)
{
    if (values == nullptr)
        throw std::invalid_argument(
            std::string("prepared radiation column has no ") + name);
    return *values;
}

} // namespace

RadiationColumnStateView prepare_radiation_column_state(
    const RadiationConfig& config,
    const RadiationPreparedColumnView& column,
    RadiationColumnWorkspace& workspace)
{
    const auto& pressure = require_vector(column.pressure_pa, "pressure");
    const auto& temperature = require_vector(column.temperature_k, "temperature");
    const auto& density = require_vector(column.density_kgm3, "density");
    const auto& z_center = require_vector(column.z_center_m, "center geometry");
    const auto& z_interface = require_vector(column.z_interface_m, "interface geometry");
    const auto& layer_thickness = require_vector(
        column.layer_thickness_m, "layer thickness");
    const std::size_t nz = pressure.size();
    if (nz == 0 || density.size() != nz || layer_thickness.size() != nz)
        throw RadiationColumnEvaluationError(
            RadiationColumnEvaluationFailure::MalformedTendency,
            "prepared radiation closure vectors have inconsistent shapes");
    if (nz == 0 || temperature.size() != nz || density.size() != nz
        || z_center.size() != nz || layer_thickness.size() != nz
        || z_interface.size() != nz + 1)
        throw std::invalid_argument(
            "prepared radiation column vectors have inconsistent shapes");
    const auto require_optional_shape = [nz](
        const std::vector<double>* values, const char* name)
    {
        if (values != nullptr && values->size() != nz)
            throw std::invalid_argument(
                std::string("prepared radiation column has malformed ") + name);
    };
    require_optional_shape(column.vapor_mixing_ratio, "qv");
    require_optional_shape(column.cloud_liquid_mixing_ratio, "qc");
    require_optional_shape(column.cloud_ice_mixing_ratio, "qi");
    if (!std::isfinite(column.cosine_solar_zenith)
        || column.cosine_solar_zenith < 0.0
        || column.cosine_solar_zenith > 1.0
        || !std::isfinite(column.surface_temperature_k)
        || (config.do_lw && (column.surface_temperature_k < 150.0
                             || column.surface_temperature_k > 400.0))
        || !std::isfinite(column.surface_albedo)
        || column.surface_albedo < 0.0 || column.surface_albedo > 1.0
        || !std::isfinite(column.surface_emissivity)
        || column.surface_emissivity < 0.0
        || column.surface_emissivity > 1.0
        || !std::isfinite(column.horizontal_cell_area_m2)
        || column.horizontal_cell_area_m2 <= 0.0)
        throw std::invalid_argument(
            "prepared radiation column has invalid forcing or surface values");
    for (std::size_t k = 0; k < nz; ++k)
    {
        if (!std::isfinite(pressure[k]) || pressure[k] <= 0.0
            || !std::isfinite(temperature[k]) || temperature[k] <= 0.0
            || !std::isfinite(density[k]) || density[k] <= 0.0
            || !std::isfinite(z_center[k])
            || !std::isfinite(layer_thickness[k])
            || layer_thickness[k] <= 0.0
            || (k > 0 && pressure[k] > pressure[k - 1]))
            throw std::invalid_argument(
                "prepared radiation column contains unphysical state");
    }
    for (std::size_t k = 0; k < z_interface.size(); ++k)
        if (!std::isfinite(z_interface[k])
            || (k > 0 && z_interface[k] <= z_interface[k - 1]))
            throw std::invalid_argument(
                "prepared radiation column interfaces must increase with height");
    for (std::size_t k = 0; k < nz; ++k)
    {
        const double geometric_thickness =
            z_interface[k + 1] - z_interface[k];
        const double thickness_scale = std::max(
            1.0, std::max(std::abs(geometric_thickness),
                          std::abs(layer_thickness[k])));
        if (!(z_center[k] > z_interface[k])
            || !(z_center[k] < z_interface[k + 1])
            || std::abs(layer_thickness[k] - geometric_thickness)
                   > 1.0e-12 * thickness_scale)
            throw std::invalid_argument(
                "prepared radiation column center/thickness geometry is inconsistent");
    }

    workspace.p_interface.resize(nz + 1);
    workspace.theta_tendency.resize(nz);
    workspace.cell_power_w.resize(nz);

    RadiationColumnStateView state;
    state.p = &pressure;
    state.T = &temperature;
    state.rho = &density;
    state.qv = column.vapor_mixing_ratio;
    state.qc = column.cloud_liquid_mixing_ratio;
    state.qi = column.cloud_ice_mixing_ratio;
    state.dz = &layer_thickness;
    state.z_int = &z_interface;
    state.mu0 = column.cosine_solar_zenith;
    state.S0 = physical_constants::solar_constant_wm2;
    state.Tsfc = column.surface_temperature_k;
    state.albedo_sw = column.surface_albedo;
    state.emissivity_lw = column.surface_emissivity;
    state.workspace = &workspace;
    radiative_transfer::diagnose_pressure_interfaces(
        pressure, z_center, z_interface, workspace.p_interface);
    state.p_int = &workspace.p_interface;
    return state;
}

double finalize_prepared_radiation_column(
    const RadiationPreparedColumnView& column,
    RadiationColumnWorkspace& workspace,
    const RadiationColumnTendencies& tendencies)
{
    const auto& pressure = require_vector(column.pressure_pa, "pressure");
    const auto& density = require_vector(column.density_kgm3, "density");
    const auto& layer_thickness = require_vector(
        column.layer_thickness_m, "layer thickness");
    const std::size_t nz = pressure.size();
    if (tendencies.dTdt_rad.size() != nz)
        throw RadiationColumnEvaluationError(
            RadiationColumnEvaluationFailure::MalformedTendency,
            "radiation scheme returned a malformed tendency column");

    double integrated_power_wm2 = 0.0;
    for (std::size_t k = 0; k < nz; ++k)
        integrated_power_wm2 += density[k]
            * physical_constants::specific_heat_cp_jkgk
            * layer_thickness[k] * tendencies.dTdt_rad[k];
    const double reported_power_wm2 = tendencies.atmospheric_power_total_wm2;
    const double power_scale = std::max(
        1.0, std::max(std::abs(integrated_power_wm2),
                      std::abs(reported_power_wm2)));
    if (!std::isfinite(reported_power_wm2)
        || !std::isfinite(integrated_power_wm2)
        || std::abs(integrated_power_wm2 - reported_power_wm2)
               > 1.0e-11 * power_scale)
        throw RadiationColumnEvaluationError(
            RadiationColumnEvaluationFailure::PowerClosure,
            "radiation tendency does not close its interface-flux power");

    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk /
        physical_constants::specific_heat_cp_jkgk;
    double stored_column_power_w = 0.0;
    for (std::size_t k = 0; k < nz; ++k)
    {
        const double theta_scale = std::pow(
            physical_constants::reference_pressure_pa / pressure[k], kappa);
        const float theta_tendency = static_cast<float>(
            tendencies.dTdt_rad[k] * theta_scale);
        if (!std::isfinite(theta_tendency))
            throw RadiationColumnEvaluationError(
                RadiationColumnEvaluationFailure::NonFiniteThetaTendency,
                "radiation scheme produced a non-finite theta tendency");
        const float cell_power_w = static_cast<float>(
            density[k] * physical_constants::specific_heat_cp_jkgk
            * layer_thickness[k] * tendencies.dTdt_rad[k]
            * column.horizontal_cell_area_m2);
        if (!std::isfinite(cell_power_w))
            throw RadiationColumnEvaluationError(
                RadiationColumnEvaluationFailure::NonFiniteCellPower,
                "radiation scheme produced non-finite cell power");
        workspace.theta_tendency[k] = theta_tendency;
        workspace.cell_power_w[k] = cell_power_w;
        stored_column_power_w += static_cast<double>(cell_power_w);
    }
    if (!std::isfinite(stored_column_power_w))
        throw RadiationColumnEvaluationError(
            RadiationColumnEvaluationFailure::NonFiniteCellPower,
            "radiation column stored power overflowed");
    return stored_column_power_w;
}

double evaluate_prepared_radiation_column(
    RadiationSchemeBase& scheme,
    const RadiationConfig& config,
    const RadiationPreparedColumnView& column,
    RadiationColumnWorkspace& workspace,
    RadiationColumnTendencies& tendencies,
    RadiationColumnFluxes* fluxes)
{
    const RadiationColumnStateView state =
        prepare_radiation_column_state(config, column, workspace);
    try
    {
        scheme.compute_column(config, state, tendencies, fluxes);
    }
    catch (const std::exception& error)
    {
        const std::string detail =
            std::string("radiation scheme rejected the column state: ") +
            error.what();
        throw RadiationColumnEvaluationError(
            RadiationColumnEvaluationFailure::SchemeRejected,
            detail.c_str());
    }
    return finalize_prepared_radiation_column(
        column, workspace, tendencies);
}

const char* radiation_prepared_column_algorithm_id() noexcept
{
    return "radiation_column_flux_power_fp32_storage_v1";
}

} // namespace tmv
