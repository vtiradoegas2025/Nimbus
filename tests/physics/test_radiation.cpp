#include "catch2/catch.hpp"
#include "radiation/radiation_base.hpp"
#include "radiation/base/radiative_transfer.hpp"
#include "radiation/schemes/rrtmg/rrtmg_atmosphere.hpp"
#include "radiation/schemes/rrtmg/rrtmg_cloud_optics.hpp"
#include "radiation/schemes/rrtmg/rrtmg_coefficients.hpp"
#include "radiation/schemes/rrtmg/rrtmg_optics.hpp"
#include "radiation/schemes/rrtmg/rrtmg_optics_internal.hpp"
#include "radiation/schemes/rrtmg/rrtmg_preprocessing.hpp"
#include "radiation/schemes/rrtmg/rrtmg_transfer.hpp"
#include "radiation/schemes/simple_grey/simple_grey.hpp"
#include "core/runtime/simulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

std::vector<std::vector<double>> read_numeric_csv(const char* path, std::size_t expected_columns)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error(std::string("cannot open reference fixture: ") + path);

    std::vector<std::vector<double>> rows;
    std::string                      line;
    while (std::getline(input, line))
    {
        if (line.empty() || line.front() == '#')
            continue;
        if (line.find_first_not_of("0123456789+-.eE,") != std::string::npos)
            continue;
        std::vector<double> row;
        std::stringstream   fields(line);
        std::string         field;
        while (std::getline(fields, field, ','))
            row.push_back(std::stod(field));
        if (row.size() != expected_columns)
            throw std::runtime_error(std::string("malformed reference fixture row: ") + path);
        rows.push_back(std::move(row));
    }
    return rows;
}

struct MlsClearColumn
{
    std::vector<double> pressure;
    std::vector<double> interface_pressure{101300.0};
    std::vector<double> temperature;
    std::vector<double> interface_temperature{294.2};
    std::vector<double> density;
    std::vector<double> thickness;
    std::vector<double> vapor;
    std::vector<double> nitrogen;
    std::vector<double> carbon_dioxide;
    std::vector<double> ozone;
    std::vector<double> nitrous_oxide;
    std::vector<double> carbon_monoxide;
    std::vector<double> methane;
    std::vector<double> oxygen;
    std::vector<double> liquid;
    std::vector<double> ice;

    RadiationColumnStateView view() const
    {
        RadiationColumnStateView column;
        column.p                   = &pressure;
        column.p_int               = &interface_pressure;
        column.T                   = &temperature;
        column.T_int               = &interface_temperature;
        column.rho                 = &density;
        column.dz                  = &thickness;
        column.qv                  = &vapor;
        column.nitrogen_vmr        = &nitrogen;
        column.carbon_dioxide_vmr  = &carbon_dioxide;
        column.ozone_vmr           = &ozone;
        column.nitrous_oxide_vmr   = &nitrous_oxide;
        column.carbon_monoxide_vmr = &carbon_monoxide;
        column.methane_vmr         = &methane;
        column.oxygen_vmr          = &oxygen;
        column.qc                  = liquid.empty() ? nullptr : &liquid;
        column.qi                  = ice.empty() ? nullptr : &ice;
        column.Tsfc                = 294.2;
        return column;
    }
};

MlsClearColumn read_mls_clear_column()
{
    const auto rows = read_numeric_csv("tests/data/rrtmg_mls_clear_lw_layers.csv", 12);
    if (rows.size() != 51)
        throw std::runtime_error("official MLS reference fixture must contain 51 layers");

    MlsClearColumn result;
    result.thickness.assign(rows.size(), 1.0);
    for (const auto& layer : rows)
    {
        result.pressure.push_back(layer[0]);
        result.temperature.push_back(layer[1]);
        result.vapor.push_back(layer[2] * 18.01528 / 28.966);
        result.carbon_dioxide.push_back(layer[3]);
        result.ozone.push_back(layer[4]);
        result.nitrous_oxide.push_back(layer[5]);
        result.carbon_monoxide.push_back(layer[6]);
        result.methane.push_back(layer[7]);
        result.oxygen.push_back(layer[8]);
        const double broadening_fraction = 1.0 - layer[3] - layer[4] - layer[5] - layer[6] - layer[7] - layer[8];
        result.nitrogen.push_back(broadening_fraction);
        const double dry_air_column = layer[9] / broadening_fraction;
        result.density.push_back(dry_air_column * 28.966e-3 * 1.0e4 / 6.02214076e23);
        result.interface_pressure.push_back(layer[10]);
        result.interface_temperature.push_back(layer[11]);
    }
    return result;
}

void add_reference_ebert_curry_clouds(MlsClearColumn& column)
{
    column.liquid.assign(column.pressure.size(), 0.0);
    column.ice.assign(column.pressure.size(), 0.0);
    const auto set_paths = [&column](std::size_t one_based_layer, double liquid_path_gm2, double ice_path_gm2)
    {
        const std::size_t layer       = one_based_layer - 1;
        const double      mass_column = 1000.0 * column.density[layer] * column.thickness[layer];
        column.liquid[layer]          = liquid_path_gm2 / mass_column;
        column.ice[layer]             = ice_path_gm2 / mass_column;
    };
    set_paths(4, 5.0, 5.0);
    set_paths(5, 5.0, 5.0);
    set_paths(12, 2.5, 2.5);
    set_paths(13, 2.5, 2.5);
}

} // namespace

TEST_CASE("Radiation factory creates simple_grey scheme", "[physics][radiation]")
{
    auto scheme = create_radiation_scheme("simple_grey");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("Radiation factory throws for unknown scheme", "[physics][radiation]")
{
    REQUIRE_THROWS(create_radiation_scheme("nonexistent_scheme_xyz"));
}

TEST_CASE("Radiation descriptors expose ready native RRTMG", "[physics][radiation][descriptor]")
{
    const auto& descriptors = get_radiation_scheme_descriptors();
    const auto* rrtmg       = tmv::find_scheme_descriptor(descriptors, "rrtmg_lw_sw");
    REQUIRE(rrtmg != nullptr);
    REQUIRE(rrtmg->id == "rrtmg");
    REQUIRE(rrtmg->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(rrtmg->constructible);
    REQUIRE(rrtmg->runtime_selectable);
    REQUIRE(rrtmg->capabilities.declared);
    REQUIRE(rrtmg->capabilities.backends == std::vector<std::string>{"cpu", "gpu"});
    REQUIRE(rrtmg->capabilities.gpu_implementation_coverage == tmv::SchemeCoverage::Covered);
    REQUIRE(rrtmg->capabilities.cpu_gpu_parity_coverage == tmv::SchemeCoverage::Covered);
    for (const std::string field : {"qv", "qc", "qi"})
    {
        REQUIRE(std::find(rrtmg->capabilities.required_state_fields.begin(),
                          rrtmg->capabilities.required_state_fields.end(),
                          field) != rrtmg->capabilities.required_state_fields.end());
    }
    REQUIRE(rrtmg->capabilities.config_validation_contract.find("rrtmg_composition") != std::string::npos);
    REQUIRE(rrtmg->caveat.find("overcast MLS") != std::string::npos);
    REQUIRE(is_radiation_scheme_implemented("rrtmg"));
    REQUIRE(tmv::find_scheme_descriptor(descriptors, "rrtmgp") == nullptr);

    const auto* grey = tmv::find_scheme_descriptor(descriptors, "grey");
    REQUIRE(grey != nullptr);
    REQUIRE(grey->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(grey->runtime_selectable);
    REQUIRE(grey->capabilities.gpu_implementation_coverage == tmv::SchemeCoverage::Covered);
    REQUIRE(grey->capabilities.cpu_gpu_parity_coverage == tmv::SchemeCoverage::Covered);
    REQUIRE(grey->capabilities.produced_state_fields == std::vector<std::string>{"dT_dt_rad", "dT_dt_lw", "dT_dt_sw"});
    for (const std::string field : {"qv", "qc", "qi"})
    {
        REQUIRE(std::find(grey->capabilities.required_state_fields.begin(),
                          grey->capabilities.required_state_fields.end(),
                          field) !=
                grey->capabilities.required_state_fields.end());
    }
    REQUIRE(grey->capabilities.config_validation_contract.find(
                "versioned_absorbing_water_cloud_optics") !=
            std::string::npos);
}

TEST_CASE("RRTMG native column preparation converts gases and resolved clouds", "[physics][radiation][rrtmg][column]")
{
    std::vector<double>      pressure{90000.0, 70000.0, 50000.0};
    std::vector<double>      interface_pressure{100000.0, 80000.0, 60000.0, 40000.0};
    std::vector<double>      temperature{290.0, 270.0, 250.0};
    std::vector<double>      density{1.0, 0.8, 0.6};
    std::vector<double>      thickness{100.0, 200.0, 300.0};
    std::vector<double>      vapor{0.010, 0.005, 0.001};
    std::vector<double>      liquid{0.0, 1.0e-3, 0.0};
    std::vector<double>      ice{0.0, 0.0, 2.0e-4};
    RadiationColumnStateView column;
    column.p     = &pressure;
    column.p_int = &interface_pressure;
    column.T     = &temperature;
    column.rho   = &density;
    column.dz    = &thickness;
    column.qv    = &vapor;
    column.qc    = &liquid;
    column.qi    = &ice;
    column.Tsfc  = 294.0;

    RadiationConfig config;
    config.clear_sky_only  = false;
    config.rrtmg_ozone_vmr = 4.0e-6;
    const auto prepared    = tmv::rrtmg::prepare_atmosphere(config, column);

    REQUIRE(prepared.layer_pressure_pa == pressure);
    REQUIRE(prepared.interface_pressure_pa == interface_pressure);
    REQUIRE(prepared.interface_temperature_k.front() == 294.0);
    REQUIRE(prepared.interface_temperature_k.back() == 250.0);
    REQUIRE(prepared.gases.water_vapor.volume_mixing_ratio[0] == Approx(0.010 * 28.966 / 18.01528).epsilon(1.0e-14));
    REQUIRE(prepared.gases.ozone.volume_mixing_ratio == std::vector<double>(3, 4.0e-6));
    REQUIRE(prepared.gases.carbon_dioxide.volume_mixing_ratio == std::vector<double>(3, 420.0e-6));
    const double dry_column = 1.0 * 100.0 * 6.02214076e23 / (28.966e-3) / 1.0e4;
    REQUIRE(prepared.dry_air_column_molecules_cm2[0] == Approx(dry_column).epsilon(1.0e-14));
    REQUIRE(prepared.gases.water_vapor.column_molecules_cm2[0] ==
            Approx(dry_column * 0.010 * 28.966 / 18.01528).epsilon(1.0e-14));
    REQUIRE(prepared.gases.oxygen.column_molecules_cm2[0] == Approx(dry_column * 0.2095).epsilon(1.0e-14));
    REQUIRE(prepared.cloud_fraction == std::vector<double>{0.0, 1.0, 1.0});
    REQUIRE(prepared.liquid_water_path_gm2[1] == Approx(160.0).epsilon(1.0e-14));
    REQUIRE(prepared.ice_water_path_gm2[2] == Approx(36.0).epsilon(1.0e-14));
    REQUIRE(prepared.liquid_effective_radius_um == std::vector<double>(3, 10.0));
    REQUIRE(prepared.ice_effective_radius_um == std::vector<double>(3, 30.0));

    config.clear_sky_only = true;
    column.qc             = nullptr;
    column.qi             = nullptr;
    const auto clear      = tmv::rrtmg::prepare_atmosphere(config, column);
    REQUIRE(clear.cloud_fraction == std::vector<double>(3, 0.0));
    REQUIRE(clear.liquid_water_path_gm2 == std::vector<double>(3, 0.0));
    REQUIRE(clear.ice_water_path_gm2 == std::vector<double>(3, 0.0));
}

TEST_CASE("RRTMG native column preparation fails closed on incomplete science state",
          "[physics][radiation][rrtmg][integrity]")
{
    std::vector<double>      pressure{90000.0};
    std::vector<double>      interface_pressure{100000.0, 80000.0};
    std::vector<double>      temperature{290.0};
    std::vector<double>      density{1.0};
    std::vector<double>      thickness{100.0};
    std::vector<double>      vapor{0.01};
    RadiationColumnStateView column;
    column.p     = &pressure;
    column.p_int = &interface_pressure;
    column.T     = &temperature;
    column.rho   = &density;
    column.dz    = &thickness;
    column.qv    = &vapor;
    RadiationConfig config;
    config.clear_sky_only = false;
    REQUIRE_THROWS_AS(tmv::rrtmg::prepare_atmosphere(config, column), std::invalid_argument);

    config.clear_sky_only = true;
    config.rrtmg_co2_vmr  = -1.0;
    REQUIRE_THROWS_AS(tmv::rrtmg::prepare_atmosphere(config, column), std::invalid_argument);
}

TEST_CASE("RRTMG column preparation accepts validated layer trace-gas profiles", "[physics][radiation][rrtmg][column]")
{
    std::vector<double> pressure{90000.0, 70000.0, 50000.0};
    std::vector<double> interface_pressure{100000.0, 80000.0, 60000.0, 40000.0};
    std::vector<double> temperature{290.0, 270.0, 250.0};
    std::vector<double> density{1.0, 0.8, 0.6};
    std::vector<double> thickness{100.0, 200.0, 300.0};
    std::vector<double> vapor{0.010, 0.005, 0.001};
    std::vector<double> ozone{2.0e-8, 4.0e-6, 8.0e-6};
    std::vector<double> carbon_dioxide{350.0e-6, 360.0e-6, 370.0e-6};
    std::vector<double> interface_temperature{294.0, 280.0, 260.0, 240.0};

    RadiationColumnStateView column;
    column.p                  = &pressure;
    column.p_int              = &interface_pressure;
    column.T                  = &temperature;
    column.rho                = &density;
    column.dz                 = &thickness;
    column.qv                 = &vapor;
    column.T_int              = &interface_temperature;
    column.ozone_vmr          = &ozone;
    column.carbon_dioxide_vmr = &carbon_dioxide;

    RadiationConfig config;
    config.clear_sky_only  = true;
    config.rrtmg_ozone_vmr = 1.0e-6;
    const auto prepared    = tmv::rrtmg::prepare_atmosphere(config, column);

    REQUIRE(prepared.gases.ozone.volume_mixing_ratio == ozone);
    REQUIRE(prepared.interface_temperature_k == interface_temperature);
    REQUIRE(prepared.surface_temperature_k == column.Tsfc);
    REQUIRE(prepared.gases.carbon_dioxide.volume_mixing_ratio == carbon_dioxide);
    REQUIRE(prepared.gases.methane.volume_mixing_ratio == std::vector<double>(3, config.rrtmg_ch4_vmr));
    REQUIRE(prepared.gases.ozone.column_molecules_cm2[1] ==
            Approx(prepared.dry_air_column_molecules_cm2[1] * ozone[1]).epsilon(1.0e-14));

    std::vector<double> wrong_shape{1.0e-6, 2.0e-6};
    column.ozone_vmr = &wrong_shape;
    REQUIRE_THROWS_AS(tmv::rrtmg::prepare_atmosphere(config, column), std::invalid_argument);

    std::vector<double> excessive_nitrogen{0.99, 0.99, 0.99};
    column.ozone_vmr    = &ozone;
    column.nitrogen_vmr = &excessive_nitrogen;
    REQUIRE_THROWS_AS(tmv::rrtmg::prepare_atmosphere(config, column), std::invalid_argument);
}

TEST_CASE("RRTMG binary-species interpolation preserves the reference edge stencil",
          "[physics][radiation][rrtmg][optics][interpolation]")
{
    tmv::rrtmg::CoefficientArray table;
    table.shape = {2, 1, 16, 13, 5, 9};
    table.values.assign(2 * 1 * 16 * 13 * 5 * 9, 0.0);
    const auto set_ratio_value = [&table](std::size_t ratio, double value)
    {
        const std::size_t index = ((((1 * 16) * 13) * 5) * 9) + ratio;
        table.values[index]     = value;
    };
    set_ratio_value(0, 1.0);
    set_ratio_value(1, 2.0);
    set_ratio_value(2, 4.0);
    set_ratio_value(4, 8.0);
    set_ratio_value(6, 10.0);
    set_ratio_value(7, 20.0);
    set_ratio_value(8, 40.0);

    tmv::rrtmg::LayerInterpolation interpolation;
    interpolation.lower_atmosphere             = true;
    interpolation.pressure_temperature_weights = {1.0, 0.0, 0.0, 0.0};
    tmv::rrtmg::GasProfile first;
    tmv::rrtmg::GasProfile second;
    first.column_molecules_cm2  = {0.05e20};
    second.column_molecules_cm2 = {0.95e20};
    tmv::rrtmg::optics_internal::KeySpecies species;
    species.first  = &first;
    species.second = &second;

    const auto interpolate = [&]
    {
        return tmv::rrtmg::optics_internal::interpolate_key_absorption(
            table, 1, 13, 9, 0, 0, interpolation, species, nullptr, 0);
    };
    const double low_p        = 8.0 * 0.05 - 1.0;
    const double low_p4       = low_p * low_p * low_p * low_p;
    const double low_expected = low_p4 * 1.0 + (1.0 - low_p - 2.0 * low_p4) * 2.0 + (low_p + low_p4) * 4.0;
    REQUIRE(interpolate() == Approx(low_expected).epsilon(1.0e-15));

    first.column_molecules_cm2  = {0.50e20};
    second.column_molecules_cm2 = {0.50e20};
    REQUIRE(interpolate() == Approx(8.0).epsilon(1.0e-15));

    first.column_molecules_cm2  = {0.95e20};
    second.column_molecules_cm2 = {0.05e20};
    const double high_p         = -(8.0 * 0.95 - 7.0);
    const double high_p4        = high_p * high_p * high_p * high_p;
    const double high_expected  = (high_p + high_p4) * 10.0 + (1.0 - high_p - 2.0 * high_p4) * 20.0 + high_p4 * 40.0;
    REQUIRE(interpolate() == Approx(high_expected).epsilon(1.0e-15));
}

TEST_CASE("RRTMG native coefficient loading fails explicitly when data are absent",
          "[physics][radiation][rrtmg][coefficients][integrity]")
{
    if (tmv::rrtmg::coefficient_loading_available())
    {
        REQUIRE_THROWS_AS(tmv::rrtmg::load_longwave_coefficients("this-RRTMG-coefficient-file-does-not-exist.nc"),
                          std::runtime_error);
    }
    else
    {
        REQUIRE_THROWS_WITH(tmv::rrtmg::load_longwave_coefficients("rrtmg_lw.nc"),
                            Catch::Contains("requires a Nimbus build with netCDF"));
    }
}

TEST_CASE("RRTMG native loader validates official coefficient schemas",
          "[physics][radiation][rrtmg][coefficients][external]")
{
    const char* longwave_path  = std::getenv("NIMBUS_RRTMG_LW_COEFFICIENTS");
    const char* shortwave_path = std::getenv("NIMBUS_RRTMG_SW_COEFFICIENTS");
    if (longwave_path == nullptr || shortwave_path == nullptr)
    {
        SUCCEED("official RRTMG coefficient paths were not supplied");
        return;
    }

    const auto longwave = tmv::rrtmg::load_longwave_coefficients(longwave_path);
    REQUIRE(longwave.identity.model_version == "rrtm_lw_v3.01");
    REQUIRE(longwave.identity.continuum_version == "CKD_2.4");
    REQUIRE(longwave.grid.band_numbers.front() == 1);
    REQUIRE(longwave.grid.band_numbers.back() == 16);
    REQUIRE(std::accumulate(longwave.grid.reduced_g_points_per_band.begin(),
                            longwave.grid.reduced_g_points_per_band.end(),
                            0) == 140);
    REQUIRE(longwave.grid.reduced_g_point_offsets.back() == 140);
    REQUIRE(longwave.grid.absorber_names[5] == "H2O");
    REQUIRE(longwave.key_absorption_lower.shape == std::vector<std::size_t>{2, 16, 16, 13, 5, 9});
    REQUIRE(longwave.integrated_planck_function.shape == std::vector<std::size_t>{16, 181});

    constexpr std::size_t                                                                    coefficient_workers = 4;
    std::array<std::shared_ptr<const tmv::rrtmg::LongwaveCoefficients>, coefficient_workers> shared_longwave;
#pragma omp parallel for num_threads(4) schedule(static)
    for (int worker = 0; worker < static_cast<int>(coefficient_workers); ++worker)
        shared_longwave[static_cast<std::size_t>(worker)] = tmv::rrtmg::shared_longwave_coefficients(longwave_path);
    REQUIRE(shared_longwave.front() != nullptr);
    for (const auto& coefficients : shared_longwave)
        REQUIRE(coefficients.get() == shared_longwave.front().get());

    std::vector<double>      pressure{90000.0, 70000.0, 50000.0};
    std::vector<double>      interface_pressure{100000.0, 80000.0, 60000.0, 40000.0};
    std::vector<double>      temperature{290.0, 270.0, 250.0};
    std::vector<double>      density{1.0, 0.8, 0.6};
    std::vector<double>      thickness{100.0, 200.0, 300.0};
    std::vector<double>      vapor{0.010, 0.005, 0.001};
    RadiationColumnStateView column;
    column.p     = &pressure;
    column.p_int = &interface_pressure;
    column.T     = &temperature;
    column.rho   = &density;
    column.dz    = &thickness;
    column.qv    = &vapor;
    column.Tsfc  = 294.0;
    RadiationConfig config;
    config.clear_sky_only  = true;
    const auto atmosphere  = tmv::rrtmg::prepare_atmosphere(config, column);
    const auto preparation = tmv::rrtmg::prepare_longwave(longwave, atmosphere, 0.97);
    REQUIRE(preparation.interpolation.size() == pressure.size());
    REQUIRE(preparation.layer_planck.size() == pressure.size() * 16);
    REQUIRE(preparation.interface_planck.size() == interface_pressure.size() * 16);
    REQUIRE(preparation.surface_planck.size() == 16);
    REQUIRE(preparation.precipitable_water_cm > 0.0);
    for (const auto& layer : preparation.interpolation)
    {
        REQUIRE(std::accumulate(layer.pressure_temperature_weights.begin(),
                                layer.pressure_temperature_weights.end(),
                                0.0) == Approx(1.0).margin(1.0e-14));
        REQUIRE(layer.lower_atmosphere);
        REQUIRE(layer.water_self_factor >= 0.0);
        REQUIRE(layer.water_foreign_factor >= 0.0);
    }
    for (double value : preparation.layer_planck)
        REQUIRE(std::isfinite(value));

    const auto longwave_optics = tmv::rrtmg::compute_longwave_gas_optics(longwave, atmosphere, preparation);
    REQUIRE(longwave_optics.layers == pressure.size());
    REQUIRE(longwave_optics.g_points == 140);
    REQUIRE(longwave_optics.absorption_optical_depth.size() == pressure.size() * 140);
    REQUIRE(std::all_of(longwave_optics.absorption_optical_depth.begin(),
                        longwave_optics.absorption_optical_depth.end(),
                        [](double value) { return std::isfinite(value) && value >= 0.0; }));

    RadiationConfig trace_gas_config         = config;
    trace_gas_config.rrtmg_ozone_vmr         = 4.0e-6;
    trace_gas_config.rrtmg_co_vmr            = 120.0e-9;
    trace_gas_config.rrtmg_cfc11_vmr         = 0.25e-9;
    trace_gas_config.rrtmg_cfc12_vmr         = 0.50e-9;
    trace_gas_config.rrtmg_cfc22_vmr         = 0.20e-9;
    trace_gas_config.rrtmg_ccl4_vmr          = 0.10e-9;
    const auto             trace_atmosphere  = tmv::rrtmg::prepare_atmosphere(trace_gas_config, column);
    const auto             trace_preparation = tmv::rrtmg::prepare_longwave(longwave, trace_atmosphere, 0.97);
    std::array<double, 16> minor_band_depth{};
    for (std::size_t band = 0; band < minor_band_depth.size(); ++band)
    {
        const std::size_t g_points = static_cast<std::size_t>(longwave.grid.reduced_g_points_per_band[band]);
        for (std::size_t layer = 0; layer < pressure.size(); ++layer)
            for (std::size_t g_point = 0; g_point < g_points; ++g_point)
                minor_band_depth[band] += tmv::rrtmg::optics_internal::longwave_minor_optical_depth(
                    longwave, trace_atmosphere, trace_preparation.interpolation[layer], band, g_point, layer);
    }
    for (std::size_t band : {0u, 2u, 4u, 5u, 6u, 7u, 8u, 10u, 12u, 14u})
    {
        INFO("longwave minor-gas band " << band + 1);
        REQUIRE(minor_band_depth[band] > 0.0);
    }
    for (std::size_t band : {1u, 3u, 9u, 11u, 13u, 15u})
    {
        INFO("longwave band without a minor-gas term " << band + 1);
        REQUIRE(minor_band_depth[band] == 0.0);
    }
    const auto longwave_fluxes = tmv::rrtmg::solve_longwave(longwave, atmosphere, preparation, longwave_optics, 0.97);
    REQUIRE(longwave_fluxes.upward_wm2.size() == pressure.size() + 1);
    REQUIRE(longwave_fluxes.downward_wm2.size() == pressure.size() + 1);
    REQUIRE(longwave_fluxes.upward_wm2.front() > 0.0);
    REQUIRE(longwave_fluxes.downward_wm2.back() == 0.0);

    auto malformed = atmosphere;
    malformed.gases.methane.column_molecules_cm2.pop_back();
    REQUIRE_THROWS_AS(tmv::rrtmg::prepare_longwave(longwave, malformed, 0.97), std::invalid_argument);

    const auto shortwave = tmv::rrtmg::load_shortwave_coefficients(shortwave_path);
    REQUIRE(shortwave.identity.model_version == "rrtm_sw_v2.5");
    REQUIRE(shortwave.identity.continuum_version == "CKD_2.4");
    REQUIRE(shortwave.grid.band_numbers.front() == 16);
    REQUIRE(shortwave.grid.band_numbers.back() == 29);
    REQUIRE(std::accumulate(shortwave.grid.reduced_g_points_per_band.begin(),
                            shortwave.grid.reduced_g_points_per_band.end(),
                            0) == 112);
    REQUIRE(shortwave.grid.reduced_g_point_offsets.back() == 112);
    REQUIRE(shortwave.grid.absorber_names[7] == "O3");
    REQUIRE(shortwave.solar_source_lower.shape == std::vector<std::size_t>{2, 14, 9, 16});
    REQUIRE(shortwave.rayleigh_extinction_upper.shape == std::vector<std::size_t>{2, 14, 5, 16});
    const auto shared_shortwave_a = tmv::rrtmg::shared_shortwave_coefficients(shortwave_path);
    const auto shared_shortwave_b = tmv::rrtmg::shared_shortwave_coefficients(shortwave_path);
    REQUIRE(shared_shortwave_a != nullptr);
    REQUIRE(shared_shortwave_a.get() == shared_shortwave_b.get());
    const auto shortwave_optics =
        tmv::rrtmg::compute_shortwave_gas_optics(shortwave, atmosphere, preparation.interpolation);
    REQUIRE(shortwave_optics.layers == pressure.size());
    REQUIRE(shortwave_optics.g_points == 112);
    REQUIRE(shortwave_optics.absorption_optical_depth.size() == pressure.size() * 112);
    REQUIRE(std::all_of(shortwave_optics.rayleigh_optical_depth.begin(),
                        shortwave_optics.rayleigh_optical_depth.end(),
                        [](double value) { return std::isfinite(value) && value >= 0.0; }));
    const auto shortwave_fluxes = tmv::rrtmg::solve_shortwave(shortwave, shortwave_optics, 0.7, 1361.0, 0.2);
    REQUIRE(shortwave_fluxes.upward_wm2.size() == pressure.size() + 1);
    REQUIRE(shortwave_fluxes.downward_wm2.back() == Approx(0.7 * 1361.0).epsilon(1.0e-13));
    REQUIRE(shortwave_fluxes.upward_wm2.back() <= shortwave_fluxes.downward_wm2.back());

    auto scheme = create_radiation_scheme("rrtmg");
    REQUIRE(scheme != nullptr);
    config.scheme_id     = "rrtmg";
    config.do_lw         = true;
    config.do_sw         = true;
    column.mu0           = 0.7;
    column.S0            = 1361.0;
    column.emissivity_lw = 0.97;
    column.albedo_sw     = 0.2;
    scheme->initialize(config);
    RadiationColumnTendencies tendencies;
    RadiationColumnFluxes     fluxes;
    scheme->compute_column(config, column, tendencies, &fluxes);
    REQUIRE(tendencies.dTdt_rad.size() == pressure.size());
    REQUIRE(fluxes.Fnet_total.size() == pressure.size() + 1);
    REQUIRE(std::all_of(
        tendencies.dTdt_rad.begin(), tendencies.dTdt_rad.end(), [](double value) { return std::isfinite(value); }));

    RadiationColumnWorkspace reusable_workspace;
    column.workspace = &reusable_workspace;
    RadiationColumnTendencies workspace_first;
    RadiationColumnTendencies workspace_second;
    scheme->compute_column(config, column, workspace_first);
    const double* const net_flux_storage  = reusable_workspace.flux_net.data();
    const std::size_t   net_flux_capacity = reusable_workspace.flux_net.capacity();
    scheme->compute_column(config, column, workspace_second);
    REQUIRE(reusable_workspace.flux_net.size() == pressure.size() + 1);
    REQUIRE(reusable_workspace.flux_net.data() == net_flux_storage);
    REQUIRE(reusable_workspace.flux_net.capacity() == net_flux_capacity);
    REQUIRE(workspace_first.dTdt_rad == tendencies.dTdt_rad);
    REQUIRE(workspace_second.dTdt_rad == tendencies.dTdt_rad);
    column.workspace = nullptr;

    constexpr std::size_t                                          worker_count = 4;
    std::array<std::unique_ptr<RadiationSchemeBase>, worker_count> workers;
    std::array<RadiationColumnTendencies, worker_count>            worker_tendencies;
    std::array<RadiationColumnFluxes, worker_count>                worker_fluxes;
    std::array<bool, worker_count>                                 worker_succeeded{};
    for (auto& worker : workers)
        worker = scheme->clone();

#pragma omp parallel for num_threads(4) schedule(static)
    for (int worker = 0; worker < static_cast<int>(worker_count); ++worker)
    {
        try
        {
            const std::size_t index = static_cast<std::size_t>(worker);
            workers[index]->compute_column(config, column, worker_tendencies[index], &worker_fluxes[index]);
            worker_succeeded[index] = true;
        }
        catch (...)
        {
            worker_succeeded[static_cast<std::size_t>(worker)] = false;
        }
    }

    const auto vector_is_bitwise_equal = [](const std::vector<double>& left, const std::vector<double>& right)
    {
        return left.size() == right.size() &&
               (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(double)) == 0);
    };
    const auto scalar_is_bitwise_equal = [](double left, double right)
    { return std::memcmp(&left, &right, sizeof(double)) == 0; };
    for (std::size_t worker = 0; worker < worker_count; ++worker)
    {
        REQUIRE(worker_succeeded[worker]);
        REQUIRE(vector_is_bitwise_equal(worker_tendencies[worker].dTdt_rad, tendencies.dTdt_rad));
        REQUIRE(vector_is_bitwise_equal(worker_tendencies[worker].dTdt_lw, tendencies.dTdt_lw));
        REQUIRE(vector_is_bitwise_equal(worker_tendencies[worker].dTdt_sw, tendencies.dTdt_sw));
        REQUIRE(scalar_is_bitwise_equal(worker_tendencies[worker].atmospheric_power_lw_wm2,
                                        tendencies.atmospheric_power_lw_wm2));
        REQUIRE(scalar_is_bitwise_equal(worker_tendencies[worker].atmospheric_power_sw_wm2,
                                        tendencies.atmospheric_power_sw_wm2));
        REQUIRE(scalar_is_bitwise_equal(worker_tendencies[worker].atmospheric_power_total_wm2,
                                        tendencies.atmospheric_power_total_wm2));
        REQUIRE(vector_is_bitwise_equal(worker_fluxes[worker].Fup_lw, fluxes.Fup_lw));
        REQUIRE(vector_is_bitwise_equal(worker_fluxes[worker].Fdn_lw, fluxes.Fdn_lw));
        REQUIRE(vector_is_bitwise_equal(worker_fluxes[worker].Fup_sw, fluxes.Fup_sw));
        REQUIRE(vector_is_bitwise_equal(worker_fluxes[worker].Fdn_sw, fluxes.Fdn_sw));
        REQUIRE(vector_is_bitwise_equal(worker_fluxes[worker].Fnet_lw, fluxes.Fnet_lw));
        REQUIRE(vector_is_bitwise_equal(worker_fluxes[worker].Fnet_sw, fluxes.Fnet_sw));
        REQUIRE(vector_is_bitwise_equal(worker_fluxes[worker].Fnet_total, fluxes.Fnet_total));
    }
}

TEST_CASE("RRTMG native longwave transfer matches the official MLS clear-sky column",
          "[physics][radiation][rrtmg][reference][external]")
{
    const char* longwave_path = std::getenv("NIMBUS_RRTMG_LW_COEFFICIENTS");
    if (longwave_path == nullptr)
    {
        SUCCEED("official RRTMG longwave coefficient path was not supplied");
        return;
    }

    const auto fixture          = read_mls_clear_column();
    const auto reference_fluxes = read_numeric_csv("tests/data/rrtmg_mls_clear_lw_fluxes.csv", 3);
    REQUIRE(reference_fluxes.size() == 52);
    RadiationColumnStateView column = fixture.view();
    column.emissivity_lw            = 1.0;

    RadiationConfig config;
    config.clear_sky_only   = true;
    const auto coefficients = tmv::rrtmg::load_longwave_coefficients(longwave_path);
    const auto atmosphere   = tmv::rrtmg::prepare_atmosphere(config, column);
    const auto preparation  = tmv::rrtmg::prepare_longwave(coefficients, atmosphere, column.emissivity_lw);
    const auto optics       = tmv::rrtmg::compute_longwave_gas_optics(coefficients, atmosphere, preparation);
    const auto fluxes = tmv::rrtmg::solve_longwave(coefficients, atmosphere, preparation, optics, column.emissivity_lw);

    double maximum_upward_error   = 0.0;
    double maximum_downward_error = 0.0;
    for (std::size_t interface = 0; interface < reference_fluxes.size(); ++interface)
    {
        REQUIRE(fixture.interface_pressure[interface] == Approx(reference_fluxes[interface][0]).epsilon(5.0e-4));
        maximum_upward_error =
            std::max(maximum_upward_error, std::abs(fluxes.upward_wm2[interface] - reference_fluxes[interface][1]));
        maximum_downward_error =
            std::max(maximum_downward_error, std::abs(fluxes.downward_wm2[interface] - reference_fluxes[interface][2]));
    }
    INFO("surface native/reference upward=" << fluxes.upward_wm2.front() << '/' << reference_fluxes.front()[1]);
    INFO("surface native/reference downward=" << fluxes.downward_wm2.front() << '/' << reference_fluxes.front()[2]);
    INFO("TOA native/reference upward=" << fluxes.upward_wm2.back() << '/' << reference_fluxes.back()[1]);
    INFO("maximum upward error=" << maximum_upward_error);
    INFO("maximum downward error=" << maximum_downward_error);
    REQUIRE(maximum_upward_error <= 0.02);
    REQUIRE(maximum_downward_error <= 0.02);
}

TEST_CASE("RRTMG native shortwave transfer matches the official MLS clear-sky column",
          "[physics][radiation][rrtmg][reference][external]")
{
    const char* longwave_path  = std::getenv("NIMBUS_RRTMG_LW_COEFFICIENTS");
    const char* shortwave_path = std::getenv("NIMBUS_RRTMG_SW_COEFFICIENTS");
    if (longwave_path == nullptr || shortwave_path == nullptr)
    {
        SUCCEED("official RRTMG coefficient paths were not supplied");
        return;
    }

    const auto fixture          = read_mls_clear_column();
    const auto reference_fluxes = read_numeric_csv("tests/data/rrtmg_mls_clear_sw_fluxes.csv", 3);
    REQUIRE(reference_fluxes.size() == 52);

    RadiationColumnStateView column = fixture.view();
    column.albedo_sw                = 0.2;
    RadiationConfig config;
    config.clear_sky_only             = true;
    const auto longwave_coefficients  = tmv::rrtmg::load_longwave_coefficients(longwave_path);
    const auto shortwave_coefficients = tmv::rrtmg::load_shortwave_coefficients(shortwave_path);
    const auto atmosphere             = tmv::rrtmg::prepare_atmosphere(config, column);
    const auto preparation            = tmv::rrtmg::prepare_longwave(longwave_coefficients, atmosphere, 1.0);
    const auto optics =
        tmv::rrtmg::compute_shortwave_gas_optics(shortwave_coefficients, atmosphere, preparation.interpolation);
    constexpr double reference_solar_irradiance_wm2 = 1360.8491;
    const double     native_source_sum =
        std::accumulate(optics.solar_source_fraction.begin(), optics.solar_source_fraction.end(), 0.0);
    constexpr std::array<double, 14> reference_band_incident{
        11.9937,
        20.2197,
        23.7440,
        22.4748,
        56.3444,
        104.2111,
        24.6282,
        344.8128,
        216.0557,
        344.4769,
        129.9930,
        45.8410,
        2.8274,
        13.2264,
    };
    std::ostringstream               source_report;
    constexpr std::array<double, 14> reference_surface_upward{
        1.2223,
        0.0623,
        3.5281,
        2.8379,
        5.8594,
        9.6105,
        4.5675,
        55.6633,
        39.6131,
        64.2547,
        22.8214,
        3.9394,
        0.0,
        0.9027,
    };
    constexpr std::array<double, 14> reference_surface_downward{
        6.1114,
        0.3117,
        17.6403,
        14.1895,
        29.2970,
        48.0526,
        22.8374,
        278.3167,
        198.0657,
        321.2733,
        114.1070,
        19.6971,
        0.0,
        4.5136,
    };
    double maximum_band_surface_upward_error   = 0.0;
    double maximum_band_surface_downward_error = 0.0;
    for (std::size_t band = 0; band < reference_band_incident.size(); ++band)
    {
        const std::size_t begin       = shortwave_coefficients.grid.reduced_g_point_offsets[band];
        const std::size_t end         = shortwave_coefficients.grid.reduced_g_point_offsets[band + 1];
        const double      band_source = std::accumulate(
            optics.solar_source_fraction.begin() + begin, optics.solar_source_fraction.begin() + end, 0.0);
        source_report << "band " << band + 16 << " native/reference incident="
                      << reference_solar_irradiance_wm2 * band_source / native_source_sum << '/'
                      << reference_band_incident[band];
        auto band_optics = optics;
        std::fill(band_optics.solar_source_fraction.begin(), band_optics.solar_source_fraction.begin() + begin, 0.0);
        std::fill(band_optics.solar_source_fraction.begin() + end, band_optics.solar_source_fraction.end(), 0.0);
        const auto band_fluxes = tmv::rrtmg::solve_shortwave(
            shortwave_coefficients, band_optics, 1.0, reference_band_incident[band], column.albedo_sw);
        maximum_band_surface_upward_error =
            std::max(maximum_band_surface_upward_error,
                     std::abs(band_fluxes.upward_wm2.front() - reference_surface_upward[band]));
        maximum_band_surface_downward_error =
            std::max(maximum_band_surface_downward_error,
                     std::abs(band_fluxes.downward_wm2.front() - reference_surface_downward[band]));
        source_report << ", surface up=" << band_fluxes.upward_wm2.front() << '/' << reference_surface_upward[band]
                      << ", down=" << band_fluxes.downward_wm2.front() << '/' << reference_surface_downward[band]
                      << '\n';
    }
    const auto fluxes = tmv::rrtmg::solve_shortwave(
        shortwave_coefficients, optics, 1.0, reference_solar_irradiance_wm2, column.albedo_sw);

    double maximum_upward_error   = 0.0;
    double maximum_downward_error = 0.0;
    for (std::size_t interface = 0; interface < reference_fluxes.size(); ++interface)
    {
        REQUIRE(fixture.interface_pressure[interface] == Approx(reference_fluxes[interface][0]).epsilon(5.0e-4));
        maximum_upward_error =
            std::max(maximum_upward_error, std::abs(fluxes.upward_wm2[interface] - reference_fluxes[interface][1]));
        maximum_downward_error =
            std::max(maximum_downward_error, std::abs(fluxes.downward_wm2[interface] - reference_fluxes[interface][2]));
    }
    INFO("surface native/reference upward=" << fluxes.upward_wm2.front() << '/' << reference_fluxes.front()[1]);
    INFO(source_report.str());
    INFO("surface native/reference downward=" << fluxes.downward_wm2.front() << '/' << reference_fluxes.front()[2]);
    INFO("TOA native/reference upward=" << fluxes.upward_wm2.back() << '/' << reference_fluxes.back()[1]);
    INFO("maximum upward error=" << maximum_upward_error);
    INFO("maximum downward error=" << maximum_downward_error);
    INFO("maximum band surface upward error=" << maximum_band_surface_upward_error);
    INFO("maximum band surface downward error=" << maximum_band_surface_downward_error);
    REQUIRE(maximum_band_surface_upward_error <= 0.003);
    REQUIRE(maximum_band_surface_downward_error <= 0.003);
    REQUIRE(maximum_upward_error <= 0.02);
    REQUIRE(maximum_downward_error <= 0.02);
}

TEST_CASE("RRTMG native cloud transfer matches an official overcast Ebert-Curry MLS column",
          "[physics][radiation][rrtmg][cloud][reference][external]")
{
    const char* longwave_path  = std::getenv("NIMBUS_RRTMG_LW_COEFFICIENTS");
    const char* shortwave_path = std::getenv("NIMBUS_RRTMG_SW_COEFFICIENTS");
    if (longwave_path == nullptr || shortwave_path == nullptr)
    {
        SUCCEED("official RRTMG coefficient paths were not supplied");
        return;
    }

    auto fixture = read_mls_clear_column();
    add_reference_ebert_curry_clouds(fixture);
    const auto longwave_reference =
        read_numeric_csv("tests/data/rrtmg_mls_cloud_ebert_curry_lw_fluxes.csv", 3);
    const auto shortwave_reference =
        read_numeric_csv("tests/data/rrtmg_mls_cloud_ebert_curry_sw_fluxes.csv", 3);
    REQUIRE(longwave_reference.size() == 52);
    REQUIRE(shortwave_reference.size() == 52);

    RadiationColumnStateView column = fixture.view();
    column.emissivity_lw            = 1.0;
    column.albedo_sw                = 0.2;
    RadiationConfig config;
    config.clear_sky_only                    = false;
    config.rrtmg_liquid_effective_radius_um = 10.0;
    config.rrtmg_ice_effective_radius_um    = 30.0;

    const auto longwave_coefficients  = tmv::rrtmg::load_longwave_coefficients(longwave_path);
    const auto shortwave_coefficients = tmv::rrtmg::load_shortwave_coefficients(shortwave_path);
    const auto atmosphere             = tmv::rrtmg::prepare_atmosphere(config, column);
    REQUIRE(atmosphere.cloud_fraction[3] == 1.0);
    REQUIRE(atmosphere.liquid_water_path_gm2[3] == Approx(5.0).epsilon(1.0e-14));
    REQUIRE(atmosphere.ice_water_path_gm2[3] == Approx(5.0).epsilon(1.0e-14));
    REQUIRE(atmosphere.liquid_water_path_gm2[4] == Approx(5.0).epsilon(1.0e-14));
    REQUIRE(atmosphere.ice_water_path_gm2[4] == Approx(5.0).epsilon(1.0e-14));

    const auto preparation = tmv::rrtmg::prepare_longwave(longwave_coefficients, atmosphere, column.emissivity_lw);
    const auto longwave_gas =
        tmv::rrtmg::compute_longwave_gas_optics(longwave_coefficients, atmosphere, preparation);
    const auto longwave_cloud = tmv::rrtmg::compute_longwave_cloud_optics(atmosphere);
    const auto longwave_fluxes = tmv::rrtmg::solve_longwave(
        longwave_coefficients, atmosphere, preparation, longwave_gas, column.emissivity_lw, &longwave_cloud);

    const auto shortwave_gas = tmv::rrtmg::compute_shortwave_gas_optics(
        shortwave_coefficients, atmosphere, preparation.interpolation);
    const auto shortwave_cloud = tmv::rrtmg::compute_shortwave_cloud_optics(shortwave_coefficients, atmosphere);
    auto negligible_cloud = atmosphere;
    negligible_cloud.cloud_fraction[0] = 1.0;
    negligible_cloud.liquid_water_path_gm2[0] = 1.0e-30;
    negligible_cloud.ice_water_path_gm2[0] = 1.0e-30;
    const auto negligible_optics = tmv::rrtmg::compute_shortwave_cloud_optics(
        shortwave_coefficients, negligible_cloud);
    for (std::size_t band = 0; band < 14; ++band)
    {
        REQUIRE(negligible_optics.optical_depth[band] == 0.0);
        REQUIRE(negligible_optics.single_scattering_albedo[band] == 0.0);
        REQUIRE(negligible_optics.asymmetry[band] == 0.0);
    }
    constexpr double reference_solar_irradiance_wm2 = 1360.8491;
    const auto shortwave_fluxes = tmv::rrtmg::solve_shortwave(shortwave_coefficients,
                                                              shortwave_gas,
                                                              1.0,
                                                              reference_solar_irradiance_wm2,
                                                              column.albedo_sw,
                                                              &shortwave_cloud);

    double maximum_lw_upward_error   = 0.0;
    double maximum_lw_downward_error = 0.0;
    double maximum_sw_upward_error   = 0.0;
    double maximum_sw_downward_error = 0.0;
    std::size_t maximum_lw_upward_interface   = 0;
    std::size_t maximum_lw_downward_interface = 0;
    for (std::size_t interface = 0; interface < longwave_reference.size(); ++interface)
    {
        REQUIRE(fixture.interface_pressure[interface] == Approx(longwave_reference[interface][0]).epsilon(5.0e-4));
        REQUIRE(fixture.interface_pressure[interface] == Approx(shortwave_reference[interface][0]).epsilon(5.0e-4));
        const double lw_upward_error =
            std::abs(longwave_fluxes.upward_wm2[interface] - longwave_reference[interface][1]);
        const double lw_downward_error =
            std::abs(longwave_fluxes.downward_wm2[interface] - longwave_reference[interface][2]);
        if (lw_upward_error > maximum_lw_upward_error)
        {
            maximum_lw_upward_error     = lw_upward_error;
            maximum_lw_upward_interface = interface;
        }
        if (lw_downward_error > maximum_lw_downward_error)
        {
            maximum_lw_downward_error     = lw_downward_error;
            maximum_lw_downward_interface = interface;
        }
        maximum_sw_upward_error = std::max(
            maximum_sw_upward_error,
            std::abs(shortwave_fluxes.upward_wm2[interface] - shortwave_reference[interface][1]));
        maximum_sw_downward_error = std::max(
            maximum_sw_downward_error,
            std::abs(shortwave_fluxes.downward_wm2[interface] - shortwave_reference[interface][2]));
    }
    INFO("maximum cloudy LW upward error=" << maximum_lw_upward_error);
    INFO("maximum cloudy LW downward error=" << maximum_lw_downward_error);
    INFO("LW upward max native/reference=" << longwave_fluxes.upward_wm2[maximum_lw_upward_interface] << '/'
                                            << longwave_reference[maximum_lw_upward_interface][1] << " at interface "
                                            << maximum_lw_upward_interface);
    INFO("LW downward max native/reference=" << longwave_fluxes.downward_wm2[maximum_lw_downward_interface] << '/'
                                              << longwave_reference[maximum_lw_downward_interface][2]
                                              << " at interface " << maximum_lw_downward_interface);
    INFO("maximum cloudy SW upward error=" << maximum_sw_upward_error);
    INFO("maximum cloudy SW downward error=" << maximum_sw_downward_error);
    REQUIRE(maximum_lw_upward_error <= 0.05);
    REQUIRE(maximum_lw_downward_error <= 0.05);
    REQUIRE(maximum_sw_upward_error <= 0.05);
    REQUIRE(maximum_sw_downward_error <= 0.05);
}

TEST_CASE("RRTMG cloudy columns fail closed without publishing partial state",
          "[physics][radiation][rrtmg][cloud][transactional][external]")
{
    if (std::getenv("NIMBUS_RRTMG_LW_COEFFICIENTS") == nullptr ||
        std::getenv("NIMBUS_RRTMG_SW_COEFFICIENTS") == nullptr)
    {
        SUCCEED("official RRTMG coefficient paths were not supplied");
        return;
    }

    auto fixture = read_mls_clear_column();
    add_reference_ebert_curry_clouds(fixture);
    fixture.ice.back() = std::numeric_limits<double>::quiet_NaN();
    RadiationColumnStateView column = fixture.view();
    column.mu0                       = 1.0;
    column.S0                        = 1360.8491;
    column.emissivity_lw             = 1.0;
    column.albedo_sw                 = 0.2;

    RadiationConfig config;
    config.scheme_id      = "rrtmg";
    config.clear_sky_only = false;
    auto scheme           = create_radiation_scheme("rrtmg");
    scheme->initialize(config);

    RadiationColumnTendencies tendencies;
    tendencies.dTdt_rad                   = {1.0, 2.0};
    tendencies.dTdt_lw                    = {3.0};
    tendencies.dTdt_sw                    = {4.0};
    tendencies.atmospheric_power_lw_wm2   = 5.0;
    tendencies.atmospheric_power_sw_wm2   = 6.0;
    tendencies.atmospheric_power_total_wm2 = 7.0;
    RadiationColumnFluxes fluxes;
    fluxes.Fup_lw     = {8.0};
    fluxes.Fdn_lw     = {9.0};
    fluxes.Fup_sw     = {10.0};
    fluxes.Fdn_sw     = {11.0};
    fluxes.Fnet_lw    = {12.0};
    fluxes.Fnet_sw    = {13.0};
    fluxes.Fnet_total = {14.0};
    const auto held_tendencies = tendencies;
    const auto held_fluxes      = fluxes;

    REQUIRE_THROWS_AS(scheme->compute_column(config, column, tendencies, &fluxes), std::invalid_argument);
    REQUIRE(tendencies.dTdt_rad == held_tendencies.dTdt_rad);
    REQUIRE(tendencies.dTdt_lw == held_tendencies.dTdt_lw);
    REQUIRE(tendencies.dTdt_sw == held_tendencies.dTdt_sw);
    REQUIRE(tendencies.atmospheric_power_lw_wm2 == held_tendencies.atmospheric_power_lw_wm2);
    REQUIRE(tendencies.atmospheric_power_sw_wm2 == held_tendencies.atmospheric_power_sw_wm2);
    REQUIRE(tendencies.atmospheric_power_total_wm2 == held_tendencies.atmospheric_power_total_wm2);
    REQUIRE(fluxes.Fup_lw == held_fluxes.Fup_lw);
    REQUIRE(fluxes.Fdn_lw == held_fluxes.Fdn_lw);
    REQUIRE(fluxes.Fup_sw == held_fluxes.Fup_sw);
    REQUIRE(fluxes.Fdn_sw == held_fluxes.Fdn_sw);
    REQUIRE(fluxes.Fnet_lw == held_fluxes.Fnet_lw);
    REQUIRE(fluxes.Fnet_sw == held_fluxes.Fnet_sw);
    REQUIRE(fluxes.Fnet_total == held_fluxes.Fnet_total);
}

TEST_CASE("RRTMG publishes complete mapped shortwave spectral optics atomically",
          "[physics][radiation][rrtmg][shortwave][terrain][spectral][transactional][external]")
{
    if (std::getenv("NIMBUS_RRTMG_LW_COEFFICIENTS") == nullptr ||
        std::getenv("NIMBUS_RRTMG_SW_COEFFICIENTS") == nullptr)
    {
        SUCCEED("official RRTMG coefficient paths were not supplied");
        return;
    }

    auto fixture = read_mls_clear_column();
    add_reference_ebert_curry_clouds(fixture);
    RadiationColumnStateView column = fixture.view();
    column.mu0 = 0.7;
    column.S0 = 1360.8491;
    column.emissivity_lw = 1.0;
    column.albedo_sw = 0.2;

    RadiationConfig config;
    config.scheme_id = "rrtmg";
    config.do_lw = true;
    config.do_sw = true;
    config.clear_sky_only = false;
    auto scheme = create_radiation_scheme("rrtmg");
    scheme->initialize(config);

    tmv::MappedShortwaveColumnOptics optics;
    REQUIRE(scheme->compute_mapped_shortwave_optics(config, column, optics));
    REQUIRE(optics.layers == fixture.pressure.size());
    REQUIRE(optics.g_points == 112);
    REQUIRE(optics.absorption_optical_depth.size() == optics.layers * 112);
    REQUIRE(optics.scattering_optical_depth.size() == optics.layers * 112);
    REQUIRE(optics.scattering_asymmetry.size() == optics.layers * 112);
    REQUIRE(optics.solar_source_fraction.size() == 112);
    REQUIRE(std::accumulate(optics.solar_source_fraction.begin(),
                            optics.solar_source_fraction.end(), 0.0) ==
            Approx(1.0).margin(1.0e-12));
    REQUIRE(std::all_of(
        optics.absorption_optical_depth.begin(),
        optics.absorption_optical_depth.end(),
        [](double value) { return std::isfinite(value) && value >= 0.0; }));
    REQUIRE(std::all_of(
        optics.scattering_optical_depth.begin(),
        optics.scattering_optical_depth.end(),
        [](double value) { return std::isfinite(value) && value >= 0.0; }));
    REQUIRE(std::all_of(
        optics.scattering_asymmetry.begin(),
        optics.scattering_asymmetry.end(),
        [](double value)
        { return std::isfinite(value) && value >= 0.0 && value <= 1.0; }));
    REQUIRE(std::any_of(
        optics.scattering_optical_depth.begin(),
        optics.scattering_optical_depth.end(),
        [](double value) { return value > 0.0; }));

    const auto held = optics;
    fixture.ice.back() = std::numeric_limits<double>::quiet_NaN();
    column = fixture.view();
    REQUIRE_THROWS_AS(
        scheme->compute_mapped_shortwave_optics(config, column, optics),
        std::invalid_argument);
    REQUIRE(optics.layers == held.layers);
    REQUIRE(optics.g_points == held.g_points);
    REQUIRE(optics.absorption_optical_depth == held.absorption_optical_depth);
    REQUIRE(optics.scattering_optical_depth == held.scattering_optical_depth);
    REQUIRE(optics.scattering_asymmetry == held.scattering_asymmetry);
    REQUIRE(optics.solar_source_fraction == held.solar_source_fraction);
}

TEST_CASE("Mapped spectral terrain owner approaches the flat RRTMG column limit",
          "[physics][radiation][rrtmg][shortwave][terrain][spectral][flat-limit][external]")
{
    if (std::getenv("NIMBUS_RRTMG_LW_COEFFICIENTS") == nullptr ||
        std::getenv("NIMBUS_RRTMG_SW_COEFFICIENTS") == nullptr)
    {
        SUCCEED("official RRTMG coefficient paths were not supplied");
        return;
    }
    const auto fixture = read_mls_clear_column();
    RadiationColumnStateView column = fixture.view();
    column.mu0 = 1.0;
    column.S0 = 1360.8491;
    column.emissivity_lw = 1.0;
    column.albedo_sw = 0.2;
    RadiationConfig config;
    config.scheme_id = "rrtmg";
    config.do_lw = true;
    config.do_sw = true;
    config.clear_sky_only = true;
    auto scheme = create_radiation_scheme("rrtmg");
    scheme->initialize(config);
    tmv::MappedShortwaveColumnOptics optics;
    REQUIRE(scheme->compute_mapped_shortwave_optics(config, column, optics));

    constexpr std::size_t coarse_layers = 3;
    tmv::rrtmg::ShortwaveOptics coarse;
    coarse.layers = coarse_layers;
    coarse.g_points = optics.g_points;
    coarse.absorption_optical_depth.assign(coarse_layers * optics.g_points, 0.0);
    coarse.rayleigh_optical_depth.assign(coarse_layers * optics.g_points, 0.0);
    coarse.solar_source_fraction = optics.solar_source_fraction;
    for (std::size_t layer = 0; layer < optics.layers; ++layer)
    {
        const std::size_t coarse_layer = layer * coarse_layers / optics.layers;
        for (std::size_t g = 0; g < optics.g_points; ++g)
        {
            coarse.absorption_optical_depth[coarse_layer * optics.g_points + g] +=
                optics.absorption_optical_depth[layer * optics.g_points + g];
            coarse.rayleigh_optical_depth[coarse_layer * optics.g_points + g] +=
                optics.scattering_optical_depth[layer * optics.g_points + g];
        }
    }
    const auto shortwave_coefficients = tmv::rrtmg::load_shortwave_coefficients(
        std::getenv("NIMBUS_RRTMG_SW_COEFFICIENTS"));
    const auto reference = tmv::rrtmg::solve_shortwave(
        shortwave_coefficients, coarse, column.mu0, column.S0,
        column.albedo_sw);
    const double reference_atmosphere = -(
        (reference.upward_wm2.back() - reference.downward_wm2.back()) -
        (reference.upward_wm2.front() - reference.downward_wm2.front()));
    const double reference_surface =
        (1.0 - column.albedo_sw) * reference.downward_wm2.front();
    const double reference_top = reference.upward_wm2.back();

    std::vector<double> interfaces(coarse_layers + 1, 0.0);
    for (std::size_t k = 0; k < coarse_layers; ++k)
        interfaces[k + 1] = static_cast<double>(k + 1);
    std::vector<double> asymmetry(coarse_layers * optics.g_points, 0.0);
    const radiative_transfer::MappedSpectralShortwaveGridView grid{
        coarse.absorption_optical_depth.data(),
        coarse.rayleigh_optical_depth.data(),
        asymmetry.data(),
        optics.solar_source_fraction.data(), interfaces.data(),
        1, 1, coarse_layers, optics.g_points, 1.0, 1.0,
        radiative_transfer::TerrainShadowBoundary::Periodic,
        radiative_transfer::TerrainShadowBoundary::Periodic};
    radiative_transfer::LambertianMultipleReflectionOptions options;
    options.quadrature = {1, 4};
    options.albedo = column.albedo_sw;
    options.maximum_bounces = 64;
    options.relative_residual_tolerance = 1.0e-8;
    options.absolute_residual_tolerance_w = 1.0e-9;
    const std::vector<double> slopes{0.0};
    radiative_transfer::MappedSpectralTerrainWorkspace workspace;
    radiative_transfer::MappedSpectralTerrainResult result;
    radiative_transfer::evaluate_mapped_spectral_terrain_shortwave_reference(
        grid, {0.0, 0.0, 1.0}, column.S0, slopes.data(), slopes.data(),
        1.0, options, workspace, result);
    INFO("mapped/reference atmosphere=" << result.atmospheric_absorbed_power_w_total
         << '/' << reference_atmosphere);
    INFO("mapped/reference surface=" << result.surface_absorbed_power_w_total
         << '/' << reference_surface);
    INFO("mapped/reference top=" << result.top_escape_power_w
         << '/' << reference_top);
    REQUIRE(result.converged);
    REQUIRE(result.atmospheric_absorbed_power_w_total ==
            Approx(reference_atmosphere).epsilon(0.08));
    REQUIRE(result.surface_absorbed_power_w_total ==
            Approx(reference_surface).epsilon(0.08));
    REQUIRE(result.top_escape_power_w == Approx(reference_top).epsilon(0.15));
    REQUIRE(std::abs(result.closure_residual_w) <= 1.0e-6);
}

TEST_CASE("Idealized solar forcing uses explicit geometry and local time", "[physics][radiation][solar]")
{
    RadiationConfig config;
    config.solar_forcing               = "idealized_diurnal";
    config.latitude_deg                = 0.0;
    config.declination_deg             = 0.0;
    config.local_solar_time_at_start_h = 0.0;

    REQUIRE(idealized_solar_cosine_zenith(config, 0.0) == Approx(0.0));
    REQUIRE(idealized_solar_cosine_zenith(config, 6.0 * 3600.0) == Approx(0.0).margin(1.0e-14));
    REQUIRE(idealized_solar_cosine_zenith(config, 12.0 * 3600.0) == Approx(1.0).margin(1.0e-14));
    REQUIRE(idealized_solar_cosine_zenith(config, 18.0 * 3600.0) == Approx(0.0).margin(1.0e-14));
    REQUIRE(idealized_solar_cosine_zenith(config, 21.0 * 3600.0) == Approx(0.0));

    config.declination_deg = 30.0;
    REQUIRE_THROWS_AS(idealized_solar_cosine_zenith(config, 0.0), std::invalid_argument);
    config.declination_deg             = 0.0;
    config.local_solar_time_at_start_h = 24.0;
    REQUIRE_THROWS_AS(idealized_solar_cosine_zenith(config, 0.0), std::invalid_argument);
}

TEST_CASE("Idealized solar geometry exposes a unit ENU sun direction", "[physics][radiation][solar][geometry]")
{
    RadiationConfig config;
    config.solar_forcing               = "idealized_diurnal";
    config.latitude_deg                = 0.0;
    config.declination_deg             = 0.0;
    config.local_solar_time_at_start_h = 0.0;

    const auto sunrise = tmv::idealized_solar_geometry(config, 6.0 * 3600.0);
    const auto noon    = tmv::idealized_solar_geometry(config, 12.0 * 3600.0);
    const auto sunset  = tmv::idealized_solar_geometry(config, 18.0 * 3600.0);
    REQUIRE(sunrise.east == Approx(1.0).margin(1.0e-14));
    REQUIRE(sunrise.north == Approx(0.0).margin(1.0e-14));
    REQUIRE(sunrise.up == Approx(0.0).margin(1.0e-14));
    REQUIRE(noon.east == Approx(0.0).margin(1.0e-14));
    REQUIRE(noon.north == Approx(0.0).margin(1.0e-14));
    REQUIRE(noon.up == Approx(1.0).margin(1.0e-14));
    REQUIRE(sunset.east == Approx(-1.0).margin(1.0e-14));
    REQUIRE(sunset.north == Approx(0.0).margin(1.0e-14));
    REQUIRE(sunset.up == Approx(0.0).margin(1.0e-14));

    config.latitude_deg                 = 35.0;
    config.declination_deg              = 15.0;
    const auto       midlatitude_noon   = tmv::idealized_solar_geometry(config, 12.0 * 3600.0);
    constexpr double degrees_to_radians = radiation_constants::pi / 180.0;
    REQUIRE(midlatitude_noon.east == Approx(0.0).margin(1.0e-14));
    REQUIRE(midlatitude_noon.north == Approx(std::sin(-20.0 * degrees_to_radians)).epsilon(1.0e-14));
    REQUIRE(midlatitude_noon.up == Approx(std::cos(20.0 * degrees_to_radians)).epsilon(1.0e-14));
    const double direction_norm =
        std::sqrt(midlatitude_noon.east * midlatitude_noon.east + midlatitude_noon.north * midlatitude_noon.north +
                  midlatitude_noon.up * midlatitude_noon.up);
    REQUIRE(direction_norm == Approx(1.0).epsilon(1.0e-14));
}

TEST_CASE("Lambertian shortwave transmission matches hemispheric quadrature",
          "[physics][radiation][shortwave][analytical][reference]")
{
    const auto independent_midpoint_quadrature = [](double optical_depth)
    {
        // Independently integrate 2*integral_0^1 mu*exp(-tau/mu) dmu.
        // Midpoints avoid evaluating the removable endpoint at mu=0.
        constexpr int intervals = 100000;
        long double   sum       = 0.0L;
        for (int index = 0; index < intervals; ++index)
        {
            const long double mu = (static_cast<long double>(index) + 0.5L) / intervals;
            sum += 2.0L * mu * std::exp(-static_cast<long double>(optical_depth) / mu);
        }
        return static_cast<double>(sum / intervals);
    };

    REQUIRE(radiative_transfer::lambertian_diffuse_transmittance(0.0) == Approx(1.0));
    REQUIRE(radiative_transfer::lambertian_diffuse_transmittance(1.0) ==
            Approx(0.21938393439552027368).epsilon(2.0e-14));
    for (double optical_depth : {0.02, 0.2, 1.0, 1.7, 6.0, 100.0})
    {
        INFO("Lambertian optical depth " << optical_depth);
        const double expected = independent_midpoint_quadrature(optical_depth);
        const double actual   = radiative_transfer::lambertian_diffuse_transmittance(optical_depth);
        REQUIRE(actual == Approx(expected).margin(2.0e-10));
    }

    const double thin     = radiative_transfer::lambertian_diffuse_transmittance(0.1);
    const double moderate = radiative_transfer::lambertian_diffuse_transmittance(1.0);
    const double thick    = radiative_transfer::lambertian_diffuse_transmittance(10.0);
    REQUIRE(1.0 > thin);
    REQUIRE(thin > moderate);
    REQUIRE(moderate > thick);
    REQUIRE(thick >= 0.0);
    REQUIRE_THROWS_AS(radiative_transfer::lambertian_diffuse_transmittance(-0.1), std::invalid_argument);
    REQUIRE_THROWS_AS(radiative_transfer::lambertian_diffuse_transmittance(std::numeric_limits<double>::quiet_NaN()),
                      std::invalid_argument);
}

TEST_CASE("Sloped Lambertian direct boundary closes actual and horizontal area power",
          "[physics][radiation][shortwave][terrain][geometry][conservation]")
{
    constexpr double             beam_normal_wm2 = 1000.0;
    constexpr double             albedo          = 0.2;
    const IdealizedSolarGeometry overhead{0.0, 0.0, 1.0};
    const auto                   flat = radiative_transfer::evaluate_sloped_lambertian_direct_boundary(
        beam_normal_wm2, overhead, 0.0, 0.0, albedo, true);
    REQUIRE(flat.surface_area_per_horizontal_area == Approx(1.0));
    REQUIRE(flat.cosine_incidence == Approx(1.0));
    REQUIRE(flat.incident_irradiance_actual_wm2 == Approx(beam_normal_wm2));
    REQUIRE(flat.incident_power_horizontal_wm2 == Approx(beam_normal_wm2));
    REQUIRE(flat.reflected_power_horizontal_wm2 == Approx(200.0));
    REQUIRE(flat.absorbed_power_horizontal_wm2 == Approx(800.0));
    REQUIRE(flat.reflected_radiance_wm2sr == Approx(200.0 / radiation_constants::pi).epsilon(1.0e-14));

    const double                 inverse_sqrt_two = 1.0 / std::sqrt(2.0);
    const IdealizedSolarGeometry from_west{-inverse_sqrt_two, 0.0, inverse_sqrt_two};
    const auto                   west_facing = radiative_transfer::evaluate_sloped_lambertian_direct_boundary(
        beam_normal_wm2, from_west, 1.0, 0.0, albedo, true);
    REQUIRE(west_facing.surface_area_per_horizontal_area == Approx(std::sqrt(2.0)).epsilon(1.0e-14));
    REQUIRE(west_facing.cosine_incidence == Approx(1.0).epsilon(1.0e-14));
    REQUIRE(west_facing.incident_irradiance_actual_wm2 == Approx(beam_normal_wm2).epsilon(1.0e-14));
    REQUIRE(west_facing.incident_power_horizontal_wm2 == Approx(beam_normal_wm2 * std::sqrt(2.0)).epsilon(1.0e-14));
    REQUIRE(west_facing.incident_power_horizontal_wm2 ==
            west_facing.absorbed_power_horizontal_wm2 + west_facing.reflected_power_horizontal_wm2);
    REQUIRE(west_facing.reflected_radiance_wm2sr * radiation_constants::pi *
                west_facing.surface_area_per_horizontal_area ==
            Approx(west_facing.reflected_power_horizontal_wm2).epsilon(1.0e-14));

    const IdealizedSolarGeometry from_east{inverse_sqrt_two, 0.0, inverse_sqrt_two};
    const auto                   self_shaded = radiative_transfer::evaluate_sloped_lambertian_direct_boundary(
        beam_normal_wm2, from_east, 1.0, 0.0, albedo, true);
    REQUIRE(self_shaded.cosine_incidence == Approx(0.0).margin(1.0e-15));
    REQUIRE(self_shaded.incident_power_horizontal_wm2 == Approx(0.0).margin(1.0e-12));

    const auto occluded = radiative_transfer::evaluate_sloped_lambertian_direct_boundary(
        beam_normal_wm2, overhead, 0.5, -0.25, albedo, false);
    REQUIRE(occluded.incident_power_horizontal_wm2 == 0.0);
    REQUIRE(occluded.absorbed_power_horizontal_wm2 == 0.0);
    REQUIRE(occluded.reflected_power_horizontal_wm2 == 0.0);

    const IdealizedSolarGeometry below_horizon{0.0, 0.0, -1.0};
    const auto                   night = radiative_transfer::evaluate_sloped_lambertian_direct_boundary(
        beam_normal_wm2, below_horizon, 0.0, 0.0, albedo, true);
    REQUIRE(night.incident_power_horizontal_wm2 == 0.0);

    REQUIRE_THROWS_AS(radiative_transfer::evaluate_sloped_lambertian_direct_boundary(
                          beam_normal_wm2, {0.0, 0.0, 0.5}, 0.0, 0.0, albedo, true),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(radiative_transfer::evaluate_sloped_lambertian_direct_boundary(
                          beam_normal_wm2, overhead, std::numeric_limits<double>::infinity(), 0.0, albedo, true),
                      std::invalid_argument);
}

TEST_CASE("Sloped Lambertian surface field matches the independent boundary result",
          "[physics][radiation][shortwave][terrain][surface][field][conservation]")
{
    using radiative_transfer::SlopedLambertianSurfaceFieldResult;
    using radiative_transfer::SlopedLambertianSurfaceFieldWorkspace;

    constexpr double             beam_normal_wm2    = 875.0;
    constexpr double             horizontal_area_m2 = 240.0;
    constexpr double             slope_x            = 0.75;
    constexpr double             slope_y            = -0.25;
    constexpr double             albedo             = 0.32;
    const IdealizedSolarGeometry sun{-0.6, 0.0, 0.8};
    const auto                   boundary = radiative_transfer::evaluate_sloped_lambertian_direct_boundary(
        beam_normal_wm2, sun, slope_x, slope_y, albedo, true);
    const std::vector<double>             incident_power{boundary.incident_power_horizontal_wm2 * horizontal_area_m2};
    SlopedLambertianSurfaceFieldWorkspace workspace;
    SlopedLambertianSurfaceFieldResult    field;
    radiative_transfer::evaluate_sloped_lambertian_surface_field(
        incident_power, &slope_x, &slope_y, 1, horizontal_area_m2, albedo, workspace, field);

    REQUIRE(field.absorbed_power_w.size() == 1);
    REQUIRE(field.reflected_power_w.size() == 1);
    REQUIRE(field.reflected_radiance_wm2sr.size() == 1);
    REQUIRE(field.incident_total_power_w == incident_power[0]);
    REQUIRE(field.absorbed_power_w[0] == boundary.absorbed_power_horizontal_wm2 * horizontal_area_m2);
    REQUIRE(field.reflected_power_w[0] == boundary.reflected_power_horizontal_wm2 * horizontal_area_m2);
    REQUIRE(field.reflected_radiance_wm2sr[0] == Approx(boundary.reflected_radiance_wm2sr).epsilon(2.0e-15));
    REQUIRE(field.absorbed_power_w[0] + field.reflected_power_w[0] == incident_power[0]);
    REQUIRE(field.absorbed_total_power_w + field.reflected_total_power_w == field.incident_total_power_w);
    REQUIRE(field.closure_residual_w == 0.0);
}

TEST_CASE("Sloped Lambertian surface field closes every cell and publishes transactionally",
          "[physics][radiation][shortwave][terrain][surface][field][transactional]")
{
    using radiative_transfer::SlopedLambertianSurfaceFieldResult;
    using radiative_transfer::SlopedLambertianSurfaceFieldWorkspace;

    constexpr double                      horizontal_area_m2 = 30.0;
    constexpr double                      albedo             = 0.25;
    const std::vector<double>             incident_power{0.0, 120.0, 900.0, 55.0};
    const std::vector<double>             slope_x{0.0, 0.5, -2.0, 3.0};
    const std::vector<double>             slope_y{0.0, -1.5, 0.25, -4.0};
    SlopedLambertianSurfaceFieldWorkspace workspace;
    SlopedLambertianSurfaceFieldResult    result;
    radiative_transfer::evaluate_sloped_lambertian_surface_field(incident_power,
                                                                 slope_x.data(),
                                                                 slope_y.data(),
                                                                 incident_power.size(),
                                                                 horizontal_area_m2,
                                                                 albedo,
                                                                 workspace,
                                                                 result);

    for (std::size_t index = 0; index < incident_power.size(); ++index)
    {
        INFO("sloped Lambertian surface cell " << index);
        const double actual_area = horizontal_area_m2 * std::hypot(std::hypot(slope_x[index], slope_y[index]), 1.0);
        REQUIRE(result.absorbed_power_w[index] + result.reflected_power_w[index] == incident_power[index]);
        REQUIRE(result.reflected_power_w[index] == albedo * incident_power[index]);
        REQUIRE(result.reflected_radiance_wm2sr[index] * radiation_constants::pi * actual_area ==
                Approx(result.reflected_power_w[index]).epsilon(2.0e-15));
    }
    REQUIRE(result.incident_total_power_w == 1075.0);
    REQUIRE(result.absorbed_total_power_w == 806.25);
    REQUIRE(result.reflected_total_power_w == 268.75);
    REQUIRE(result.closure_residual_w == 0.0);

    // Warm both transactional vector banks, then prove steady-cadence reuse.
    radiative_transfer::evaluate_sloped_lambertian_surface_field(incident_power,
                                                                 slope_x.data(),
                                                                 slope_y.data(),
                                                                 incident_power.size(),
                                                                 horizontal_area_m2,
                                                                 albedo,
                                                                 workspace,
                                                                 result);
    const auto        committed_absorbed  = result.absorbed_power_w;
    const auto        committed_reflected = result.reflected_power_w;
    const auto        committed_radiance  = result.reflected_radiance_wm2sr;
    const std::size_t absorbed_capacity   = workspace.scratch_absorbed_power_w.capacity();
    const std::size_t reflected_capacity  = workspace.scratch_reflected_power_w.capacity();
    const std::size_t radiance_capacity   = workspace.scratch_reflected_radiance_wm2sr.capacity();
    radiative_transfer::evaluate_sloped_lambertian_surface_field(incident_power,
                                                                 slope_x.data(),
                                                                 slope_y.data(),
                                                                 incident_power.size(),
                                                                 horizontal_area_m2,
                                                                 albedo,
                                                                 workspace,
                                                                 result);
    REQUIRE(result.absorbed_power_w == committed_absorbed);
    REQUIRE(result.reflected_power_w == committed_reflected);
    REQUIRE(result.reflected_radiance_wm2sr == committed_radiance);
    REQUIRE(workspace.scratch_absorbed_power_w.capacity() == absorbed_capacity);
    REQUIRE(workspace.scratch_reflected_power_w.capacity() == reflected_capacity);
    REQUIRE(workspace.scratch_reflected_radiance_wm2sr.capacity() == radiance_capacity);

    auto malformed_slope = slope_x;
    malformed_slope[2]   = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(radiative_transfer::evaluate_sloped_lambertian_surface_field(incident_power,
                                                                                   malformed_slope.data(),
                                                                                   slope_y.data(),
                                                                                   incident_power.size(),
                                                                                   horizontal_area_m2,
                                                                                   albedo,
                                                                                   workspace,
                                                                                   result),
                      std::invalid_argument);
    REQUIRE(result.absorbed_power_w == committed_absorbed);
    REQUIRE(result.reflected_power_w == committed_reflected);
    REQUIRE(result.reflected_radiance_wm2sr == committed_radiance);

    auto negative_power = incident_power;
    negative_power[1]   = -1.0;
    REQUIRE_THROWS_AS(radiative_transfer::evaluate_sloped_lambertian_surface_field(negative_power,
                                                                                   slope_x.data(),
                                                                                   slope_y.data(),
                                                                                   negative_power.size(),
                                                                                   horizontal_area_m2,
                                                                                   albedo,
                                                                                   workspace,
                                                                                   result),
                      std::invalid_argument);
    REQUIRE(result.incident_total_power_w == 1075.0);
    REQUIRE_THROWS_AS(radiative_transfer::evaluate_sloped_lambertian_surface_field(incident_power,
                                                                                   slope_x.data(),
                                                                                   slope_y.data(),
                                                                                   incident_power.size(),
                                                                                   horizontal_area_m2,
                                                                                   1.01,
                                                                                   workspace,
                                                                                   result),
                      std::invalid_argument);
    REQUIRE(result.reflected_total_power_w == 268.75);
}

TEST_CASE("Terrain shadow reference resolves bilinear occlusion and boundary semantics",
          "[physics][radiation][shortwave][terrain][shadow][reference]")
{
    std::vector<double>                      heights(4 * 3, 0.0);
    radiative_transfer::TerrainHeightMapView terrain{heights.data(),
                                                     4,
                                                     3,
                                                     1.0,
                                                     1.0,
                                                     radiative_transfer::TerrainShadowBoundary::Open,
                                                     radiative_transfer::TerrainShadowBoundary::Open};
    const IdealizedSolarGeometry             low_sun{-std::sqrt(0.75), 0.0, 0.5};
    REQUIRE(radiative_transfer::terrain_height_map_visible_to_sun_reference(terrain, 3, 1, low_sun));

    for (std::size_t y = 0; y < terrain.ny; ++y)
        heights[1 * terrain.ny + y] = 10.0;
    REQUIRE_FALSE(radiative_transfer::terrain_height_map_visible_to_sun_reference(terrain, 3, 1, low_sun));
    const IdealizedSolarGeometry high_sun{-0.1, 0.0, std::sqrt(0.99)};
    REQUIRE(radiative_transfer::terrain_height_map_visible_to_sun_reference(terrain, 3, 1, high_sun));
    // An open boundary carries no invented terrain beyond its represented
    // extent; the western edge therefore sees a western sun.
    REQUIRE(radiative_transfer::terrain_height_map_visible_to_sun_reference(terrain, 0, 1, low_sun));

    std::vector<double> periodic_heights(4 * 2, 0.0);
    for (std::size_t y = 0; y < 2; ++y)
        periodic_heights[3 * 2 + y] = 10.0;
    radiative_transfer::TerrainHeightMapView periodic{periodic_heights.data(),
                                                      4,
                                                      2,
                                                      1.0,
                                                      1.0,
                                                      radiative_transfer::TerrainShadowBoundary::Periodic,
                                                      radiative_transfer::TerrainShadowBoundary::Open};
    REQUIRE_FALSE(radiative_transfer::terrain_height_map_visible_to_sun_reference(periodic, 0, 0, low_sun));
    periodic.x_boundary = radiative_transfer::TerrainShadowBoundary::Open;
    REQUIRE(radiative_transfer::terrain_height_map_visible_to_sun_reference(periodic, 0, 0, low_sun));

    // A bilinear saddle has zero height at both diagonal endpoints but rises
    // inside the patch. Endpoint-only horizon tests would miss this blocker.
    const std::vector<double>                saddle_heights{0.0, 10.0, 10.0, 0.0};
    radiative_transfer::TerrainHeightMapView saddle{saddle_heights.data(),
                                                    2,
                                                    2,
                                                    1.0,
                                                    1.0,
                                                    radiative_transfer::TerrainShadowBoundary::Open,
                                                    radiative_transfer::TerrainShadowBoundary::Open};
    const IdealizedSolarGeometry             diagonal_sun{0.7, 0.7, std::sqrt(0.02)};
    REQUIRE_FALSE(radiative_transfer::terrain_height_map_visible_to_sun_reference(saddle, 0, 0, diagonal_sun));

    REQUIRE_FALSE(radiative_transfer::terrain_height_map_visible_to_sun_reference(terrain, 2, 1, {0.0, 0.0, -1.0}));
    heights[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(radiative_transfer::terrain_height_map_visible_to_sun_reference(terrain, 2, 1, low_sun),
                      std::invalid_argument);
}

TEST_CASE("Terrain shadow CPU field matches every independent reference ray",
          "[physics][radiation][shortwave][terrain][shadow][cpu][parity]")
{
    constexpr std::size_t nx = 19;
    constexpr std::size_t ny = 17;
    std::vector<double>   heights(nx * ny, 0.0);
    for (std::size_t i = 0; i < nx; ++i)
        for (std::size_t j = 0; j < ny; ++j)
            heights[i * ny + j] = 8.0 * std::sin(0.71 * i) + 5.0 * std::cos(0.43 * j) + 3.0 * std::sin(0.19 * i * j);
    radiative_transfer::TerrainHeightMapView terrain{heights.data(),
                                                     nx,
                                                     ny,
                                                     120.0,
                                                     85.0,
                                                     radiative_transfer::TerrainShadowBoundary::Open,
                                                     radiative_transfer::TerrainShadowBoundary::Periodic};
    const IdealizedSolarGeometry             sun{-0.72, 0.48, std::sqrt(1.0 - 0.72 * 0.72 - 0.48 * 0.48)};

    radiative_transfer::TerrainShadowMaskWorkspace workspace;
    workspace.scratch_visibility.resize(nx * ny, 99);
    std::vector<unsigned char>                      visibility(nx * ny, 88);
    const std::size_t                               scratch_capacity   = workspace.scratch_visibility.capacity();
    const std::size_t                               published_capacity = visibility.capacity();
    radiative_transfer::TerrainShadowTraversalStats stats;
    radiative_transfer::terrain_height_map_visibility_field_cpu(terrain, sun, workspace, visibility, &stats);

    REQUIRE(visibility.size() == nx * ny);
    REQUIRE(stats.height_values_validated == nx * ny);
    REQUIRE(stats.rays_traced == nx * ny);
    REQUIRE(stats.hierarchy_nodes_built > 0);
    REQUIRE(stats.hierarchy_nodes_built < 2 * (nx - 1) * ny);
    REQUIRE(stats.hierarchy_blocks_skipped > 0);
    REQUIRE(stats.bilinear_patches_tested > 0);
    std::uint64_t reference_patches = 0;
    for (std::size_t i = 0; i < nx; ++i)
        for (std::size_t j = 0; j < ny; ++j)
        {
            INFO("terrain visibility cell " << i << "," << j);
            std::uint64_t local_reference_patches = 0;
            const bool    reference               = radiative_transfer::terrain_height_map_visible_to_sun_reference(
                terrain, i, j, sun, &local_reference_patches);
            reference_patches += local_reference_patches;
            REQUIRE((visibility[i * ny + j] != 0) == reference);
        }
    REQUIRE(stats.bilinear_patches_tested < reference_patches);

    // Both publication buffers were pre-sized. Repeated cadence updates rotate
    // them without growing either allocation.
    const std::size_t hierarchy_capacity         = workspace.hierarchy_max_height_m.capacity();
    const std::size_t hierarchy_offsets_capacity = workspace.hierarchy_level_offset.capacity();
    radiative_transfer::terrain_height_map_visibility_field_cpu(terrain, sun, workspace, visibility, nullptr);
    REQUIRE(workspace.scratch_visibility.capacity() == scratch_capacity);
    REQUIRE(visibility.capacity() == published_capacity);
    REQUIRE(workspace.hierarchy_max_height_m.capacity() == hierarchy_capacity);
    REQUIRE(workspace.hierarchy_level_offset.capacity() == hierarchy_offsets_capacity);

    const std::vector<unsigned char> committed       = visibility;
    const auto                       committed_stats = stats;
    heights[4 * ny + 3]                              = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(
        radiative_transfer::terrain_height_map_visibility_field_cpu(terrain, sun, workspace, visibility, &stats),
        std::invalid_argument);
    REQUIRE(visibility == committed);
    REQUIRE(stats.height_values_validated == committed_stats.height_values_validated);
    REQUIRE(stats.rays_traced == committed_stats.rays_traced);
    REQUIRE(stats.hierarchy_nodes_built == committed_stats.hierarchy_nodes_built);
    REQUIRE(stats.hierarchy_blocks_skipped == committed_stats.hierarchy_blocks_skipped);
    REQUIRE(stats.bilinear_patches_tested == committed_stats.bilinear_patches_tested);
}

TEST_CASE("Terrain shadow hierarchy matches mixed-boundary low-sun oracle matrix",
          "[physics][radiation][shortwave][terrain][shadow][hierarchy][parity]")
{
    constexpr std::size_t nx = 9;
    constexpr std::size_t ny = 8;
    std::vector<double>   heights(nx * ny, 0.0);
    for (std::size_t i = 0; i < nx; ++i)
        for (std::size_t j = 0; j < ny; ++j)
        {
            heights[i * ny + j] = 5.0 * std::sin(0.83 * i) + 4.0 * std::cos(0.61 * j) + 2.0 * std::sin(0.37 * i * j);
            if ((i + 2 * j) % 7 == 0)
                heights[i * ny + j] += 15.0;
        }
    const std::vector<IdealizedSolarGeometry> suns{
        {0.91, -0.32, std::sqrt(0.0695)},
        {-0.55, 0.72, std::sqrt(0.1791)},
        {0.15, -0.10, std::sqrt(0.9675)},
    };
    const std::vector<std::pair<radiative_transfer::TerrainShadowBoundary, radiative_transfer::TerrainShadowBoundary>>
        boundaries{
            {radiative_transfer::TerrainShadowBoundary::Open, radiative_transfer::TerrainShadowBoundary::Open},
            {radiative_transfer::TerrainShadowBoundary::Periodic, radiative_transfer::TerrainShadowBoundary::Open},
            {radiative_transfer::TerrainShadowBoundary::Open, radiative_transfer::TerrainShadowBoundary::Periodic},
            {radiative_transfer::TerrainShadowBoundary::Periodic, radiative_transfer::TerrainShadowBoundary::Periodic},
        };

    radiative_transfer::TerrainShadowMaskWorkspace workspace;
    std::vector<unsigned char>                     visibility;
    std::uint64_t                                  accelerated_patches = 0;
    std::uint64_t                                  reference_patches   = 0;
    std::uint64_t                                  skipped_blocks      = 0;
    std::uint64_t                                  visible_count       = 0;
    std::uint64_t                                  shadowed_count      = 0;
    for (const auto& boundary : boundaries)
        for (const auto& sun : suns)
        {
            radiative_transfer::TerrainHeightMapView terrain{
                heights.data(), nx, ny, 1.3, 0.9, boundary.first, boundary.second};
            radiative_transfer::TerrainShadowTraversalStats stats;
            radiative_transfer::terrain_height_map_visibility_field_cpu(terrain, sun, workspace, visibility, &stats);
            accelerated_patches += stats.bilinear_patches_tested;
            skipped_blocks += stats.hierarchy_blocks_skipped;
            for (std::size_t i = 0; i < nx; ++i)
                for (std::size_t j = 0; j < ny; ++j)
                {
                    INFO("boundary=" << static_cast<int>(boundary.first) << "," << static_cast<int>(boundary.second)
                                     << " sun=" << sun.east << "," << sun.north << "," << sun.up << " cell=" << i << ","
                                     << j);
                    std::uint64_t patches = 0;
                    const bool    reference =
                        radiative_transfer::terrain_height_map_visible_to_sun_reference(terrain, i, j, sun, &patches);
                    reference_patches += patches;
                    const bool accelerated = visibility[i * ny + j] != 0;
                    REQUIRE(accelerated == reference);
                    visible_count += accelerated ? 1 : 0;
                    shadowed_count += accelerated ? 0 : 1;
                }
        }
    REQUIRE(visible_count > 0);
    REQUIRE(shadowed_count > 0);
    REQUIRE(skipped_blocks > 0);
    REQUIRE(accelerated_patches < reference_patches);

    radiative_transfer::TerrainHeightMapView        open_terrain{heights.data(),
                                                                 nx,
                                                                 ny,
                                                                 1.3,
                                                                 0.9,
                                                                 radiative_transfer::TerrainShadowBoundary::Open,
                                                                 radiative_transfer::TerrainShadowBoundary::Open};
    radiative_transfer::TerrainShadowTraversalStats trivial_stats;
    radiative_transfer::terrain_height_map_visibility_field_cpu(
        open_terrain, {0.0, 0.0, 1.0}, workspace, visibility, &trivial_stats);
    REQUIRE(std::all_of(visibility.begin(), visibility.end(), [](unsigned char value) { return value == 1; }));
    REQUIRE(trivial_stats.height_values_validated == nx * ny);
    REQUIRE(trivial_stats.rays_traced == 0);
    REQUIRE(trivial_stats.hierarchy_nodes_built == 0);
    REQUIRE(trivial_stats.hierarchy_blocks_skipped == 0);
    REQUIRE(trivial_stats.bilinear_patches_tested == 0);

    radiative_transfer::terrain_height_map_visibility_field_cpu(
        open_terrain, {0.0, 0.0, -1.0}, workspace, visibility, &trivial_stats);
    REQUIRE(std::all_of(visibility.begin(), visibility.end(), [](unsigned char value) { return value == 0; }));
    REQUIRE(trivial_stats.rays_traced == 0);
    REQUIRE(trivial_stats.hierarchy_nodes_built == 0);
    REQUIRE(trivial_stats.bilinear_patches_tested == 0);
}

TEST_CASE("Direct-beam ordered path deposits slant optical energy conservatively",
          "[physics][radiation][shortwave][terrain][atmosphere][reference][conservation]")
{
    using radiative_transfer::DirectBeamPathSegment;
    using radiative_transfer::DirectBeamSegmentAbsorption;

    constexpr double                         incident_power_w = 731.25;
    constexpr double                         mu0              = 0.4;
    const std::vector<DirectBeamPathSegment> segments{
        {7, 250.0 / mu0, 0.025, 250.0},
        {11, 400.0 / mu0, 0.16, 400.0},
        {29, 900.0 / mu0, 0.72, 900.0},
    };
    std::vector<DirectBeamSegmentAbsorption> absorption;
    const auto                               result =
        radiative_transfer::attenuate_direct_beam_path_reference(incident_power_w, segments, absorption);

    REQUIRE(absorption.size() == segments.size());
    REQUIRE(result.incident_power_w == incident_power_w);
    REQUIRE(result.total_slant_optical_depth == Approx((0.025 + 0.16 + 0.72) / mu0).epsilon(2.0e-15));
    REQUIRE(result.transmitted_power_w ==
            Approx(incident_power_w * std::exp(-(0.025 + 0.16 + 0.72) / mu0)).epsilon(3.0e-15));
    REQUIRE(result.absorbed_power_w + result.transmitted_power_w == result.incident_power_w);

    double      expected_incoming = incident_power_w;
    long double resolved_absorbed = 0.0L;
    for (std::size_t index = 0; index < segments.size(); ++index)
    {
        const double expected_tau      = segments[index].vertical_layer_optical_depth / mu0;
        const double expected_absorbed = expected_incoming * (-std::expm1(-expected_tau));
        INFO("direct-beam segment " << index);
        REQUIRE(absorption[index].cell_index == segments[index].cell_index);
        REQUIRE(absorption[index].slant_optical_depth == Approx(expected_tau).epsilon(2.0e-15));
        REQUIRE(absorption[index].incoming_power_w == expected_incoming);
        REQUIRE(absorption[index].absorbed_power_w == Approx(expected_absorbed).epsilon(2.0e-15));
        REQUIRE(absorption[index].absorbed_power_w + absorption[index].transmitted_power_w ==
                absorption[index].incoming_power_w);
        expected_incoming = absorption[index].transmitted_power_w;
        resolved_absorbed += absorption[index].absorbed_power_w;
    }
    REQUIRE(static_cast<double>(resolved_absorbed) == Approx(result.absorbed_power_w).epsilon(3.0e-15));
}

TEST_CASE("Direct-beam path is split-invariant, thin-stable, and transactional",
          "[physics][radiation][shortwave][terrain][atmosphere][reference][transaction]")
{
    using radiative_transfer::DirectBeamPathSegment;
    using radiative_transfer::DirectBeamSegmentAbsorption;

    constexpr double                         incident_power_w = 42.0;
    const std::vector<DirectBeamPathSegment> unsplit{
        {3, 1200.0, 1.0e-14, 600.0},
        {5, 800.0, 0.75, 400.0},
    };
    const std::vector<DirectBeamPathSegment> split{
        {3, 300.0, 1.0e-14, 600.0},
        {3, 900.0, 1.0e-14, 600.0},
        {5, 200.0, 0.75, 400.0},
        {5, 600.0, 0.75, 400.0},
    };
    std::vector<DirectBeamSegmentAbsorption> unsplit_absorption;
    std::vector<DirectBeamSegmentAbsorption> split_absorption;
    const auto                               unsplit_result =
        radiative_transfer::attenuate_direct_beam_path_reference(incident_power_w, unsplit, unsplit_absorption);
    const auto split_result =
        radiative_transfer::attenuate_direct_beam_path_reference(incident_power_w, split, split_absorption);
    REQUIRE(split_result.total_slant_optical_depth ==
            Approx(unsplit_result.total_slant_optical_depth).epsilon(2.0e-15));
    REQUIRE(split_result.transmitted_power_w == Approx(unsplit_result.transmitted_power_w).epsilon(2.0e-15));
    REQUIRE(split_result.absorbed_power_w == Approx(unsplit_result.absorbed_power_w).epsilon(2.0e-15));
    REQUIRE(unsplit_absorption.front().absorbed_power_w > 0.0);
    REQUIRE(unsplit_absorption.front().absorbed_power_w == Approx(incident_power_w * 2.0e-14).epsilon(2.0e-14));

    std::vector<DirectBeamSegmentAbsorption> opaque_absorption;
    const auto                               opaque = radiative_transfer::attenuate_direct_beam_path_reference(
        incident_power_w, std::vector<DirectBeamPathSegment>{{9, 1000.0, 1000.0, 1.0}}, opaque_absorption);
    REQUIRE(opaque.transmitted_power_w == 0.0);
    REQUIRE(opaque.absorbed_power_w == incident_power_w);
    REQUIRE(opaque_absorption[0].absorbed_power_w == incident_power_w);

    std::vector<DirectBeamSegmentAbsorption> empty_absorption;
    const auto empty = radiative_transfer::attenuate_direct_beam_path_reference(incident_power_w, {}, empty_absorption);
    REQUIRE(empty_absorption.empty());
    REQUIRE(empty.transmitted_power_w == incident_power_w);
    REQUIRE(empty.absorbed_power_w == 0.0);

    const std::vector<DirectBeamSegmentAbsorption> committed{{99, 1.0, 2.0, 1.0, 1.0}};
    auto                                           publication = committed;
    REQUIRE_THROWS_AS(radiative_transfer::attenuate_direct_beam_path_reference(
                          incident_power_w, std::vector<DirectBeamPathSegment>{{1, 0.0, 0.1, 100.0}}, publication),
                      std::invalid_argument);
    REQUIRE(publication.size() == committed.size());
    REQUIRE(publication[0].cell_index == committed[0].cell_index);
    REQUIRE(publication[0].absorbed_power_w == committed[0].absorbed_power_w);
    REQUIRE_THROWS_AS(radiative_transfer::attenuate_direct_beam_path_reference(-1.0, unsplit, publication),
                      std::invalid_argument);
    REQUIRE(publication[0].cell_index == committed[0].cell_index);
}

TEST_CASE("Bilinear mapped-interface ray intersection resolves true crossings",
          "[physics][radiation][shortwave][terrain][atmosphere][ray][mapped][intersection]")
{
    using radiative_transfer::BilinearMappedInterfacePatch;
    using radiative_transfer::MappedInterfaceCrossing;
    const BilinearMappedInterfacePatch flat{5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0};
    const auto vertical = radiative_transfer::first_bilinear_mapped_interface_intersection_reference(
        flat, 0.25, 0.75, 10.0, {0.0, 0.0, -1.0}, 10.0, MappedInterfaceCrossing::FromAbove);
    REQUIRE(vertical.has_value());
    REQUIRE(*vertical == 5.0);

    // z_interface=x+2y and z_ray=2-0.8s with x=0.6s.
    const BilinearMappedInterfacePatch plane{0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0};
    const auto planar = radiative_transfer::first_bilinear_mapped_interface_intersection_reference(
        plane, 0.0, 0.0, 2.0, {0.6, 0.0, -0.8}, 1.5, MappedInterfaceCrossing::FromAbove);
    REQUIRE(planar.has_value());
    REQUIRE(*planar == Approx(10.0 / 7.0).epsilon(3.0e-15));

    // The saddle becomes 4x(1-x) on x=y. A horizontal diagonal ray at z=0.5
    // intersects twice; crossing sense selects the entry then the exit.
    const BilinearMappedInterfacePatch saddle{0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0};
    const double                       diagonal = std::sqrt(0.5);
    const auto into_saddle = radiative_transfer::first_bilinear_mapped_interface_intersection_reference(
        saddle, 0.0, 0.0, 0.5, {diagonal, diagonal, 0.0}, std::sqrt(2.0), MappedInterfaceCrossing::FromAbove);
    REQUIRE(into_saddle.has_value());
    REQUIRE(*into_saddle == Approx((1.0 - std::sqrt(0.5)) / (2.0 * diagonal)).epsilon(4.0e-15));
    const auto out_of_saddle = radiative_transfer::first_bilinear_mapped_interface_intersection_reference(
        saddle, 0.5, 0.5, 0.5, {diagonal, diagonal, 0.0}, diagonal, MappedInterfaceCrossing::FromBelow);
    REQUIRE(out_of_saddle.has_value());
    REQUIRE(*out_of_saddle == Approx(0.5).epsilon(4.0e-15));
}

TEST_CASE("Bilinear mapped-interface solver rejects tangency and wrong ownership",
          "[physics][radiation][shortwave][terrain][atmosphere][ray][mapped][tangent]")
{
    using radiative_transfer::BilinearMappedInterfacePatch;
    using radiative_transfer::MappedInterfaceCrossing;
    const BilinearMappedInterfacePatch saddle{0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0};
    const double                       diagonal = std::sqrt(0.5);
    const auto tangent = radiative_transfer::first_bilinear_mapped_interface_intersection_reference(
        saddle, 0.0, 0.0, 1.0, {diagonal, diagonal, 0.0}, std::sqrt(2.0), MappedInterfaceCrossing::FromAbove);
    REQUIRE_FALSE(tangent.has_value());

    const BilinearMappedInterfacePatch flat{5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0};
    const auto starts_downward = radiative_transfer::first_bilinear_mapped_interface_intersection_reference(
        flat, 0.5, 0.5, 5.0, {0.0, 0.0, -1.0}, 1.0, MappedInterfaceCrossing::FromAbove);
    REQUIRE(starts_downward.has_value());
    REQUIRE(*starts_downward == 0.0);
    const auto starts_upward = radiative_transfer::first_bilinear_mapped_interface_intersection_reference(
        flat, 0.5, 0.5, 5.0, {0.0, 0.0, 1.0}, 1.0, MappedInterfaceCrossing::FromBelow);
    REQUIRE(starts_upward.has_value());
    REQUIRE(*starts_upward == 0.0);
    const auto parallel_clear = radiative_transfer::first_bilinear_mapped_interface_intersection_reference(
        flat, 0.25, 0.5, 10.0, {1.0, 0.0, 0.0}, 0.5, MappedInterfaceCrossing::FromAbove);
    REQUIRE_FALSE(parallel_clear.has_value());

    REQUIRE_THROWS_AS(radiative_transfer::first_bilinear_mapped_interface_intersection_reference(
                          flat, 0.5, 0.5, 4.0, {0.0, 0.0, -1.0}, 1.0, MappedInterfaceCrossing::FromAbove),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(radiative_transfer::first_bilinear_mapped_interface_intersection_reference(
                          flat, 0.75, 0.5, 10.0, {1.0, 0.0, 0.0}, 0.5, MappedInterfaceCrossing::FromAbove),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(radiative_transfer::first_bilinear_mapped_interface_intersection_reference(
                          flat, 0.5, 0.5, 10.0, {0.0, 0.0, -2.0}, 1.0, MappedInterfaceCrossing::FromAbove),
                      std::invalid_argument);
}

TEST_CASE("Mapped direct-beam flat limit matches rectilinear long characteristic",
          "[physics][radiation][shortwave][terrain][atmosphere][ray][mapped][flat-limit]")
{
    using radiative_transfer::DirectBeamPathSegment;
    using radiative_transfer::DirectBeamSegmentAbsorption;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::RectilinearDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    const std::vector<double> z_interface{0.0, 100.0, 200.0};
    const std::vector<double> vertical_tau{
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
    };
    std::vector<double> mapped_interface(3 * 1 * 3, 0.0);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            mapped_interface[i * 3 + k] = z_interface[k];
    const RectilinearDirectBeamGridView      rectilinear{vertical_tau.data(),
                                                         z_interface.data(),
                                                         3,
                                                         1,
                                                         2,
                                                         100.0,
                                                         100.0,
                                                         TerrainShadowBoundary::Periodic,
                                                         TerrainShadowBoundary::Open};
    const MappedDirectBeamGridView           mapped{vertical_tau.data(),
                                                    mapped_interface.data(),
                                                    3,
                                                    1,
                                                    2,
                                                    100.0,
                                                    100.0,
                                                    TerrainShadowBoundary::Periodic,
                                                    TerrainShadowBoundary::Open};
    const IdealizedSolarGeometry             sun{-0.8, 0.0, 0.6};
    std::vector<DirectBeamPathSegment>       rect_segments;
    std::vector<DirectBeamPathSegment>       mapped_segments;
    std::vector<DirectBeamSegmentAbsorption> rect_absorption;
    std::vector<DirectBeamSegmentAbsorption> mapped_absorption;
    const auto rect_result = radiative_transfer::trace_rectilinear_direct_beam_from_top_reference(
        rectilinear, 0, 0, sun, 600.0, rect_segments, rect_absorption);
    const auto mapped_result = radiative_transfer::trace_mapped_direct_beam_from_top_reference(
        mapped, 0, 0, sun, 600.0, mapped_segments, mapped_absorption);
    REQUIRE(mapped_result.exit == rect_result.exit);
    REQUIRE(mapped_result.terminal_i == rect_result.terminal_i);
    REQUIRE(mapped_result.terminal_j == rect_result.terminal_j);
    REQUIRE(mapped_result.terminal_x_m == Approx(rect_result.terminal_x_m).margin(2.0e-12));
    REQUIRE(mapped_result.terminal_y_m == rect_result.terminal_y_m);
    REQUIRE(mapped_result.terminal_z_m == Approx(rect_result.terminal_z_m).margin(2.0e-12));
    REQUIRE(mapped_segments.size() == rect_segments.size());
    REQUIRE(mapped_absorption.size() == rect_absorption.size());
    for (std::size_t index = 0; index < rect_segments.size(); ++index)
    {
        INFO("mapped flat-limit segment " << index);
        REQUIRE(mapped_segments[index].cell_index == rect_segments[index].cell_index);
        REQUIRE(mapped_segments[index].path_length_m == Approx(rect_segments[index].path_length_m).margin(3.0e-12));
        REQUIRE(mapped_absorption[index].absorbed_power_w ==
                Approx(rect_absorption[index].absorbed_power_w).margin(2.0e-12));
    }
    REQUIRE(mapped_result.energy.total_slant_optical_depth ==
            Approx(rect_result.energy.total_slant_optical_depth).margin(2.0e-14));
    REQUIRE(mapped_result.energy.transmitted_power_w == Approx(rect_result.energy.transmitted_power_w).margin(2.0e-12));
    REQUIRE(mapped_result.energy.absorbed_power_w + mapped_result.energy.transmitted_power_w ==
            mapped_result.energy.incident_power_w);
}

TEST_CASE("Mapped direct beam crosses analytic sloped interfaces conservatively",
          "[physics][radiation][shortwave][terrain][atmosphere][ray][mapped][slope]")
{
    using radiative_transfer::DirectBeamPathSegment;
    using radiative_transfer::DirectBeamRayExit;
    using radiative_transfer::DirectBeamSegmentAbsorption;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    // At scalar centers x=50/150 m, ground is 0/20 m, the middle interface is
    // 100/110 m, and the model top is exactly level at 200 m.
    const std::vector<double> interfaces{
        0.0,
        100.0,
        200.0,
        20.0,
        110.0,
        200.0,
    };
    const std::vector<double>                vertical_tau{0.01, 0.02, 0.10, 0.20};
    const MappedDirectBeamGridView           grid{vertical_tau.data(),
                                                  interfaces.data(),
                                                  2,
                                                  1,
                                                  2,
                                                  100.0,
                                                  100.0,
                                                  TerrainShadowBoundary::Open,
                                                  TerrainShadowBoundary::Open};
    const double                             up = std::sqrt(0.96);
    const IdealizedSolarGeometry             sun{0.2, 0.0, up};
    const double                             expected_middle_path  = 90.0 / (up - 0.02);
    const double                             expected_surface_path = 180.0 / (up - 0.04);
    const double                             expected_surface_x    = 150.0 - 0.2 * expected_surface_path;
    const double                             expected_surface_z    = 20.0 - 0.04 * expected_surface_path;
    std::vector<DirectBeamPathSegment>       segments;
    std::vector<DirectBeamSegmentAbsorption> absorption;
    const auto                               result =
        radiative_transfer::trace_mapped_direct_beam_from_top_reference(grid, 1, 0, sun, 500.0, segments, absorption);
    REQUIRE(result.exit == DirectBeamRayExit::Surface);
    REQUIRE(result.terminal_i == 1);
    REQUIRE(result.terminal_j == 0);
    REQUIRE(result.terminal_x_m == Approx(expected_surface_x).epsilon(5.0e-15));
    REQUIRE(result.terminal_y_m == 50.0);
    REQUIRE(result.terminal_z_m == Approx(expected_surface_z).epsilon(8.0e-15));
    REQUIRE(segments.size() == 2);
    REQUIRE(segments[0].cell_index == 3);
    REQUIRE(segments[1].cell_index == 2);
    REQUIRE(segments[0].path_length_m == Approx(expected_middle_path).epsilon(8.0e-15));
    REQUIRE(segments[1].path_length_m == Approx(expected_surface_path - expected_middle_path).epsilon(8.0e-15));
    const double expected_tau =
        0.20 * expected_middle_path / 90.0 + 0.10 * (expected_surface_path - expected_middle_path) / 90.0;
    REQUIRE(result.energy.total_slant_optical_depth == Approx(expected_tau).epsilon(8.0e-15));
    REQUIRE(result.energy.transmitted_power_w == Approx(500.0 * std::exp(-expected_tau)).epsilon(8.0e-15));
    REQUIRE(result.energy.absorbed_power_w + result.energy.transmitted_power_w == result.energy.incident_power_w);

    const auto committed_segments    = segments;
    const auto committed_absorption  = absorption;
    auto       malformed             = interfaces;
    malformed[4]                     = 10.0; // middle interface below the local ground (20 m)
    MappedDirectBeamGridView invalid = grid;
    invalid.z_interface_m            = malformed.data();
    REQUIRE_THROWS_AS(radiative_transfer::trace_mapped_direct_beam_from_top_reference(
                          invalid, 1, 0, sun, 500.0, segments, absorption),
                      std::invalid_argument);
    REQUIRE(segments.size() == committed_segments.size());
    REQUIRE(absorption.size() == committed_absorption.size());
    REQUIRE(segments[0].cell_index == committed_segments[0].cell_index);
    REQUIRE(absorption[0].absorbed_power_w == committed_absorption[0].absorbed_power_w);
}

TEST_CASE("Mapped direct beam admits a non-level physical top conservatively",
          "[physics][radiation][shortwave][terrain][atmosphere][ray][mapped][topography][conservation]")
{
    using radiative_transfer::DirectBeamFieldResult;
    using radiative_transfer::DirectBeamFieldWorkspace;
    using radiative_transfer::DirectBeamPathSegment;
    using radiative_transfer::DirectBeamRayExit;
    using radiative_transfer::DirectBeamSegmentAbsorption;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    constexpr double spacing = 100.0;
    const std::vector<double> interfaces{
        0.0, 100.0, 180.0,
        20.0, 110.0, 200.0,
    };
    const std::vector<double> tau(4, 0.0);
    const MappedDirectBeamGridView grid{
        tau.data(), interfaces.data(), 2, 1, 2, spacing, spacing,
        TerrainShadowBoundary::Periodic, TerrainShadowBoundary::Periodic};

    DirectBeamFieldWorkspace workspace;
    DirectBeamFieldResult result;
    radiative_transfer::evaluate_mapped_direct_beam_field_reference(
        grid, {0.0, 0.0, 1.0}, 1000.0, workspace, result);
    REQUIRE(result.incoming_top_power_w == 20.0e6);
    REQUIRE(result.surface_incident_total_power_w == 20.0e6);
    REQUIRE(result.atmospheric_absorbed_power_w == 0.0);
    REQUIRE(result.lateral_escape_power_w == 0.0);
    REQUIRE(result.boundary_closure_residual_w == 0.0);

    std::vector<DirectBeamPathSegment> segments;
    std::vector<DirectBeamSegmentAbsorption> absorption;
    const auto upward =
        radiative_transfer::trace_mapped_radiative_ray_from_cell_reference(
            grid, 0, 0, 1, {0.0, 0.0, 1.0}, 75.0,
            segments, absorption);
    REQUIRE(upward.exit == DirectBeamRayExit::LevelTop);
    REQUIRE(upward.terminal_z_m == 180.0);
    REQUIRE(upward.energy.transmitted_power_w == 75.0);
    REQUIRE(upward.energy.absorbed_power_w == 0.0);
}

TEST_CASE("Mapped direct-beam field closes periodic atmospheric and surface power",
          "[physics][radiation][shortwave][terrain][atmosphere][field][periodic][conservation]")
{
    using radiative_transfer::DirectBeamFieldResult;
    using radiative_transfer::DirectBeamFieldWorkspace;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    constexpr std::size_t     nx = 2, ny = 2, nz = 2;
    constexpr double          dx = 10.0, dy = 20.0;
    const std::vector<double> levels{0.0, 100.0, 300.0};
    std::vector<double>       interfaces(nx * ny * (nz + 1), 0.0);
    std::vector<double>       tau(nx * ny * nz, 0.0);
    for (std::size_t column = 0; column < nx * ny; ++column)
    {
        for (std::size_t k = 0; k <= nz; ++k)
            interfaces[column * (nz + 1) + k] = levels[k];
        tau[column * nz]     = 0.1;
        tau[column * nz + 1] = 0.2;
    }
    const MappedDirectBeamGridView grid{tau.data(),
                                        interfaces.data(),
                                        nx,
                                        ny,
                                        nz,
                                        dx,
                                        dy,
                                        TerrainShadowBoundary::Periodic,
                                        TerrainShadowBoundary::Periodic};
    DirectBeamFieldWorkspace       workspace;
    DirectBeamFieldResult          result;
    radiative_transfer::evaluate_mapped_direct_beam_field_reference(grid, {0.0, 0.0, 1.0}, 1000.0, workspace, result);

    constexpr double bundle_power   = 1000.0 * dx * dy;
    constexpr double incoming       = bundle_power * nx * ny;
    const double     top_absorbed   = bundle_power * (-std::expm1(-0.2));
    const double     after_top      = bundle_power - top_absorbed;
    const double     lower_absorbed = after_top * (-std::expm1(-0.1));
    const double     surface        = bundle_power * std::exp(-0.3);
    REQUIRE(result.cell_absorbed_power_w.size() == nx * ny * nz);
    REQUIRE(result.surface_incident_power_w.size() == nx * ny);
    for (std::size_t column = 0; column < nx * ny; ++column)
    {
        REQUIRE(result.cell_absorbed_power_w[column * nz] == Approx(lower_absorbed).epsilon(3.0e-15));
        REQUIRE(result.cell_absorbed_power_w[column * nz + 1] == Approx(top_absorbed).epsilon(3.0e-15));
        REQUIRE(result.surface_incident_power_w[column] == Approx(surface).epsilon(3.0e-15));
    }
    REQUIRE(result.incoming_top_power_w == incoming);
    REQUIRE(result.incoming_side_power_w == 0.0);
    REQUIRE(result.atmospheric_absorbed_power_w == Approx((top_absorbed + lower_absorbed) * nx * ny).epsilon(3.0e-15));
    REQUIRE(result.surface_incident_total_power_w == Approx(surface * nx * ny).epsilon(3.0e-15));
    REQUIRE(result.lateral_escape_power_w == 0.0);
    REQUIRE(result.top_rays == nx * ny);
    REQUIRE(result.side_rays == 0);
    REQUIRE(result.surface_rays == nx * ny);
    REQUIRE(result.escaped_rays == 0);
    REQUIRE(result.interface_values_validated == nx * ny * (nz + 1));
    REQUIRE(result.optical_values_validated == nx * ny * nz);
    REQUIRE(result.boundary_closure_residual_w == Approx(0.0).margin(2.0e-10));

    const auto        committed_cell_power    = result.cell_absorbed_power_w;
    const auto        committed_surface_power = result.surface_incident_power_w;
    const std::size_t ray_segment_capacity    = workspace.ray_segments.capacity();
    const std::size_t ray_absorption_capacity = workspace.ray_absorption.capacity();
    radiative_transfer::evaluate_mapped_direct_beam_field_reference(grid, {0.0, 0.0, 1.0}, 1000.0, workspace, result);
    REQUIRE(result.cell_absorbed_power_w == committed_cell_power);
    REQUIRE(result.surface_incident_power_w == committed_surface_power);
    REQUIRE(workspace.ray_segments.capacity() == ray_segment_capacity);
    REQUIRE(workspace.ray_absorption.capacity() == ray_absorption_capacity);
    REQUIRE(result.interface_values_validated == nx * ny * (nz + 1));
    REQUIRE(result.optical_values_validated == nx * ny * nz);
}

TEST_CASE("Mapped surface radiative ray resolves top escape and atmospheric absorption",
          "[physics][radiation][shortwave][terrain][reflection][ray][top][conservation]")
{
    using radiative_transfer::DirectBeamRayExit;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    constexpr std::size_t                                        nx = 1, ny = 1, nz = 2;
    const std::vector<double>                                    interfaces{0.0, 100.0, 300.0};
    const std::vector<double>                                    tau{0.1, 0.2};
    const MappedDirectBeamGridView                               grid{tau.data(),
                                                                      interfaces.data(),
                                                                      nx,
                                                                      ny,
                                                                      nz,
                                                                      20.0,
                                                                      30.0,
                                                                      TerrainShadowBoundary::Periodic,
                                                                      TerrainShadowBoundary::Periodic};
    std::vector<radiative_transfer::DirectBeamPathSegment>       segments;
    std::vector<radiative_transfer::DirectBeamSegmentAbsorption> absorption;
    const auto ray = radiative_transfer::trace_mapped_radiative_ray_from_surface_reference(
        grid, 0, 0, {0.0, 0.0, 1.0}, 500.0, segments, absorption);

    REQUIRE(ray.exit == DirectBeamRayExit::LevelTop);
    REQUIRE(ray.terminal_i == 0);
    REQUIRE(ray.terminal_j == 0);
    REQUIRE(ray.terminal_z_m == 300.0);
    REQUIRE(segments.size() == 2);
    REQUIRE(absorption.size() == 2);
    REQUIRE(segments[0].cell_index == 0);
    REQUIRE(segments[0].path_length_m == 100.0);
    REQUIRE(segments[1].cell_index == 1);
    REQUIRE(segments[1].path_length_m == 200.0);
    REQUIRE(ray.energy.total_slant_optical_depth == Approx(0.3).epsilon(2.0e-15));
    REQUIRE(ray.energy.transmitted_power_w == Approx(500.0 * std::exp(-0.3)).epsilon(2.0e-15));
    REQUIRE(ray.energy.absorbed_power_w + ray.energy.transmitted_power_w == ray.energy.incident_power_w);
}

TEST_CASE("Mapped surface radiative ray reintercepts continuous terrain",
          "[physics][radiation][shortwave][terrain][reflection][ray][surface]")
{
    using radiative_transfer::DirectBeamRayExit;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    constexpr std::size_t nx = 3, ny = 1, nz = 2;
    // Ground is flat through the first interpolation patch, then rises with
    // dh/dx=2. The ray has dz/dx=0.5 and intersects at x=550/3 m.
    const std::vector<double>      interfaces{0.0, 200.0, 400.0, 0.0, 200.0, 400.0, 200.0, 300.0, 400.0};
    const std::vector<double>      tau(nx * ny * nz, 0.0);
    const MappedDirectBeamGridView grid{tau.data(),
                                        interfaces.data(),
                                        nx,
                                        ny,
                                        nz,
                                        100.0,
                                        100.0,
                                        TerrainShadowBoundary::Open,
                                        TerrainShadowBoundary::Open};
    const double                   inverse_sqrt_five = 1.0 / std::sqrt(5.0);
    const IdealizedSolarGeometry   direction{2.0 * inverse_sqrt_five, 0.0, inverse_sqrt_five};
    std::vector<radiative_transfer::DirectBeamPathSegment>       segments;
    std::vector<radiative_transfer::DirectBeamSegmentAbsorption> absorption;
    const auto ray = radiative_transfer::trace_mapped_radiative_ray_from_surface_reference(
        grid, 0, 0, direction, 75.0, segments, absorption);

    REQUIRE(ray.exit == DirectBeamRayExit::Surface);
    REQUIRE(ray.terminal_i == 1);
    REQUIRE(ray.terminal_j == 0);
    REQUIRE(ray.terminal_x_m == Approx(550.0 / 3.0).epsilon(4.0e-15));
    REQUIRE(ray.terminal_z_m == Approx(200.0 / 3.0).epsilon(4.0e-15));
    REQUIRE(ray.energy.absorbed_power_w == 0.0);
    REQUIRE(ray.energy.transmitted_power_w == 75.0);
    REQUIRE_FALSE(segments.empty());

    const auto committed_segments   = segments;
    const auto committed_absorption = absorption;
    REQUIRE_THROWS_AS(radiative_transfer::trace_mapped_radiative_ray_from_surface_reference(
                          grid, 0, 0, {1.0, 0.0, 1.0}, 75.0, segments, absorption),
                      std::invalid_argument);
    REQUIRE(segments.size() == committed_segments.size());
    REQUIRE(absorption.size() == committed_absorption.size());
    REQUIRE(segments[0].cell_index == committed_segments[0].cell_index);

    MappedDirectBeamGridView periodic = grid;
    periodic.x_boundary               = TerrainShadowBoundary::Periodic;
    periodic.y_boundary               = TerrainShadowBoundary::Periodic;
    REQUIRE_THROWS_AS(radiative_transfer::trace_mapped_radiative_ray_from_surface_reference(
                          periodic, 0, 0, {1.0, 0.0, 0.0}, 75.0, segments, absorption),
                      std::invalid_argument);
}

TEST_CASE("Mapped Lambertian reflection sweep conserves transparent top escape",
          "[physics][radiation][shortwave][terrain][reflection][field][quadrature][conservation]")
{
    using radiative_transfer::LambertianAngularQuadrature;
    using radiative_transfer::LambertianReflectionSweepResult;
    using radiative_transfer::LambertianReflectionSweepWorkspace;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    constexpr std::size_t              nx = 1, ny = 1, nz = 2;
    const std::vector<double>          interfaces{0.0, 100.0, 300.0};
    const std::vector<double>          tau{0.0, 0.0};
    const MappedDirectBeamGridView     grid{tau.data(),
                                            interfaces.data(),
                                            nx,
                                            ny,
                                            nz,
                                            100.0,
                                            100.0,
                                            TerrainShadowBoundary::Periodic,
                                            TerrainShadowBoundary::Periodic};
    const std::vector<double>          reflected{1234.5};
    const std::vector<double>          slope_x{0.0};
    const std::vector<double>          slope_y{0.0};
    LambertianReflectionSweepWorkspace workspace;
    LambertianReflectionSweepResult    result;
    radiative_transfer::evaluate_mapped_lambertian_reflection_sweep_reference(
        grid, reflected, slope_x.data(), slope_y.data(), {3, 8}, workspace, result);

    REQUIRE(result.emitted_power_w == reflected[0]);
    REQUIRE(result.atmospheric_absorbed_power_w == 0.0);
    REQUIRE(result.surface_incident_total_power_w == 0.0);
    REQUIRE(result.top_escape_power_w == Approx(reflected[0]).epsilon(2.0e-15));
    REQUIRE(result.lateral_escape_power_w == 0.0);
    REQUIRE(result.closure_residual_w == Approx(0.0).margin(2.0e-12));
    REQUIRE(result.cell_absorbed_power_w == std::vector<double>{0.0, 0.0});
    REQUIRE(result.surface_incident_power_w == std::vector<double>{0.0});
    REQUIRE(result.rays == 24);
    REQUIRE(result.top_escape_rays == 24);
    REQUIRE(result.surface_rays == 0);
    REQUIRE(result.lateral_escape_rays == 0);
    REQUIRE(result.interface_values_validated == nx * ny * (nz + 1));
    REQUIRE(result.optical_values_validated == nx * ny * nz);

    // Warm both transactional banks and prove steady-cadence reuse.
    radiative_transfer::evaluate_mapped_lambertian_reflection_sweep_reference(
        grid, reflected, slope_x.data(), slope_y.data(), {3, 8}, workspace, result);
    const std::size_t        cell_capacity    = workspace.scratch_cell_absorbed_power_w.capacity();
    const std::size_t        surface_capacity = workspace.scratch_surface_incident_power_w.capacity();
    std::vector<std::size_t> segment_capacities;
    std::vector<std::size_t> absorption_capacities;
    for (const auto& scratch : workspace.ray_batch)
    {
        segment_capacities.push_back(scratch.segments.capacity());
        absorption_capacities.push_back(scratch.absorption.capacity());
    }
    radiative_transfer::evaluate_mapped_lambertian_reflection_sweep_reference(
        grid, reflected, slope_x.data(), slope_y.data(), {3, 8}, workspace, result);
    REQUIRE(workspace.scratch_cell_absorbed_power_w.capacity() == cell_capacity);
    REQUIRE(workspace.scratch_surface_incident_power_w.capacity() == surface_capacity);
    REQUIRE(workspace.ray_batch.size() == 24);
    for (std::size_t slot = 0; slot < workspace.ray_batch.size(); ++slot)
    {
        REQUIRE(workspace.ray_batch[slot].segments.capacity() == segment_capacities[slot]);
        REQUIRE(workspace.ray_batch[slot].absorption.capacity() == absorption_capacities[slot]);
    }
}

TEST_CASE("Mapped Lambertian reflection sweep matches one-node slant absorption",
          "[physics][radiation][shortwave][terrain][reflection][field][atmosphere]")
{
    using radiative_transfer::LambertianReflectionSweepResult;
    using radiative_transfer::LambertianReflectionSweepWorkspace;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    const std::vector<double>          interfaces{0.0, 100.0, 300.0};
    const std::vector<double>          tau{0.1, 0.2};
    const MappedDirectBeamGridView     grid{tau.data(),
                                            interfaces.data(),
                                            1,
                                            1,
                                            2,
                                            100.0,
                                            100.0,
                                            TerrainShadowBoundary::Periodic,
                                            TerrainShadowBoundary::Periodic};
    const std::vector<double>          reflected{1000.0};
    const std::vector<double>          zero_slope{0.0};
    LambertianReflectionSweepWorkspace workspace;
    LambertianReflectionSweepResult    result;
    radiative_transfer::evaluate_mapped_lambertian_reflection_sweep_reference(
        grid, reflected, zero_slope.data(), zero_slope.data(), {1, 4}, workspace, result);

    const double after_lower  = 1000.0 * std::exp(-0.2);
    const double expected_top = after_lower * std::exp(-0.4);
    REQUIRE(result.rays == 4);
    REQUIRE(result.top_escape_rays == 4);
    REQUIRE(result.surface_rays == 0);
    REQUIRE(result.lateral_escape_rays == 0);
    REQUIRE(result.cell_absorbed_power_w[0] == Approx(1000.0 - after_lower).epsilon(3.0e-15));
    REQUIRE(result.cell_absorbed_power_w[1] == Approx(after_lower - expected_top).epsilon(3.0e-15));
    REQUIRE(result.atmospheric_absorbed_power_w == Approx(1000.0 - expected_top).epsilon(3.0e-15));
    REQUIRE(result.top_escape_power_w == Approx(expected_top).epsilon(3.0e-15));
    REQUIRE(result.emitted_power_w == result.atmospheric_absorbed_power_w + result.top_escape_power_w);
    REQUIRE(result.closure_residual_w == 0.0);

    const auto committed_cells   = result.cell_absorbed_power_w;
    const auto committed_surface = result.surface_incident_power_w;
    auto       invalid_slope     = zero_slope;
    invalid_slope[0]             = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(radiative_transfer::evaluate_mapped_lambertian_reflection_sweep_reference(
                          grid, reflected, invalid_slope.data(), zero_slope.data(), {1, 4}, workspace, result),
                      std::invalid_argument);
    REQUIRE(result.cell_absorbed_power_w == committed_cells);
    REQUIRE(result.surface_incident_power_w == committed_surface);
    REQUIRE_THROWS_AS(radiative_transfer::evaluate_mapped_lambertian_reflection_sweep_reference(
                          grid, reflected, zero_slope.data(), zero_slope.data(), {0, 4}, workspace, result),
                      std::invalid_argument);
    REQUIRE(result.top_escape_power_w == Approx(expected_top).epsilon(3.0e-15));
}

TEST_CASE("Mapped Lambertian angular sweep converges to hemispheric transmission",
          "[physics][radiation][shortwave][terrain][reflection][quadrature][convergence]")
{
    using radiative_transfer::LambertianReflectionSweepResult;
    using radiative_transfer::LambertianReflectionSweepWorkspace;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;
    constexpr double                   optical_depth = 0.7;
    const std::vector<double>          interfaces{0.0, 1000.0};
    const std::vector<double>          tau{optical_depth};
    const MappedDirectBeamGridView     grid{tau.data(),
                                            interfaces.data(),
                                            1,
                                            1,
                                            1,
                                            100.0,
                                            100.0,
                                            TerrainShadowBoundary::Periodic,
                                            TerrainShadowBoundary::Periodic};
    const std::vector<double>          source{1.0};
    const std::vector<double>          slopes{0.0};
    LambertianReflectionSweepWorkspace workspace;
    LambertianReflectionSweepResult    result;
    const double                       expected = radiative_transfer::lambertian_diffuse_transmittance(optical_depth);
    double                             previous_error = std::numeric_limits<double>::infinity();
    for (std::size_t cosine_nodes : {4u, 8u, 16u})
    {
        INFO("Lambertian cosine nodes " << cosine_nodes);
        radiative_transfer::evaluate_mapped_lambertian_reflection_sweep_reference(
            grid, source, slopes.data(), slopes.data(), {cosine_nodes, 2}, workspace, result);
        const double error = std::abs(result.top_escape_power_w - expected);
        REQUIRE(error < previous_error);
        previous_error = error;
        REQUIRE(result.atmospheric_absorbed_power_w + result.top_escape_power_w == Approx(1.0).epsilon(3.0e-15));
    }
    REQUIRE(result.top_escape_power_w == Approx(expected).margin(2.0e-10));
}

TEST_CASE("Mapped Lambertian ray batches are bitwise deterministic",
          "[physics][radiation][shortwave][terrain][reflection][parallel][deterministic][allocation]")
{
    using radiative_transfer::LambertianReflectionSweepResult;
    using radiative_transfer::LambertianReflectionSweepWorkspace;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;
    // Exactly reaches the production parallel-work floor, so the four-worker
    // branch exercises a real persistent team rather than a serialized region.
    constexpr std::size_t nx = 64, ny = 32, nz = 2;
    std::vector<double>   interfaces(nx * ny * (nz + 1), 0.0);
    std::vector<double>   tau(nx * ny * nz, 0.0);
    std::vector<double>   source(nx * ny, 0.0);
    std::vector<double>   slope_x(nx * ny, 0.0);
    std::vector<double>   slope_y(nx * ny, 0.0);
    for (std::size_t column = 0; column < nx * ny; ++column)
    {
        interfaces[column * (nz + 1)]     = 0.0;
        interfaces[column * (nz + 1) + 1] = 120.0;
        interfaces[column * (nz + 1) + 2] = 360.0;
        tau[column * nz]                  = 0.01 + 0.001 * static_cast<double>(column % 5);
        tau[column * nz + 1]              = 0.02 + 0.001 * static_cast<double>(column % 3);
        source[column]                    = 100.0 + static_cast<double>(column);
        slope_x[column]                   = 0.02 * static_cast<double>(column % 3);
        slope_y[column]                   = -0.015 * static_cast<double>(column % 4);
    }
    const MappedDirectBeamGridView     grid{tau.data(),
                                            interfaces.data(),
                                            nx,
                                            ny,
                                            nz,
                                            90.0,
                                            110.0,
                                            TerrainShadowBoundary::Periodic,
                                            TerrainShadowBoundary::Periodic};
    LambertianReflectionSweepWorkspace workspace;
    LambertianReflectionSweepResult    result;
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    radiative_transfer::evaluate_mapped_lambertian_reflection_sweep_reference(
        grid, source, slope_x.data(), slope_y.data(), {4, 8}, workspace, result);
    const auto   one_cells         = result.cell_absorbed_power_w;
    const auto   one_surface       = result.surface_incident_power_w;
    const double one_atmosphere    = result.atmospheric_absorbed_power_w;
    const double one_surface_total = result.surface_incident_total_power_w;
    const double one_top           = result.top_escape_power_w;
    const double one_lateral       = result.lateral_escape_power_w;
    const double one_residual      = result.closure_residual_w;
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    radiative_transfer::evaluate_mapped_lambertian_reflection_sweep_reference(
        grid, source, slope_x.data(), slope_y.data(), {4, 8}, workspace, result);
    const bool exact = result.cell_absorbed_power_w == one_cells && result.surface_incident_power_w == one_surface &&
                       result.atmospheric_absorbed_power_w == one_atmosphere &&
                       result.surface_incident_total_power_w == one_surface_total &&
                       result.top_escape_power_w == one_top && result.lateral_escape_power_w == one_lateral &&
                       result.closure_residual_w == one_residual;
    std::vector<std::size_t> segment_capacities;
    std::vector<std::size_t> absorption_capacities;
    for (const auto& scratch : workspace.ray_batch)
    {
        segment_capacities.push_back(scratch.segments.capacity());
        absorption_capacities.push_back(scratch.absorption.capacity());
    }
    radiative_transfer::evaluate_mapped_lambertian_reflection_sweep_reference(
        grid, source, slope_x.data(), slope_y.data(), {4, 8}, workspace, result);
    bool capacities_equal = true;
    for (std::size_t slot = 0; slot < workspace.ray_batch.size(); ++slot)
        capacities_equal = capacities_equal &&
                           workspace.ray_batch[slot].segments.capacity() == segment_capacities[slot] &&
                           workspace.ray_batch[slot].absorption.capacity() == absorption_capacities[slot];
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
    REQUIRE(workspace.ray_batch.size() == 256);
    REQUIRE(result.rays == nx * ny * 32);
    REQUIRE(exact);
    REQUIRE(result.cell_absorbed_power_w == one_cells);
    REQUIRE(result.surface_incident_power_w == one_surface);
    REQUIRE(capacities_equal);
}

TEST_CASE("Mapped Lambertian source clearance fails before parallel tracing",
          "[physics][radiation][shortwave][terrain][reflection][parallel][transactional]")
{
    using radiative_transfer::LambertianReflectionSweepResult;
    using radiative_transfer::LambertianReflectionSweepWorkspace;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    const double                   ground          = 1.0;
    const double                   first_interface = std::nextafter(ground, std::numeric_limits<double>::infinity());
    const std::vector<double>      interfaces{ground, first_interface};
    const std::vector<double>      tau{0.0};
    const std::vector<double>      source{1.0};
    const std::vector<double>      slopes{0.0};
    const MappedDirectBeamGridView grid{tau.data(),
                                        interfaces.data(),
                                        1,
                                        1,
                                        1,
                                        100.0,
                                        100.0,
                                        TerrainShadowBoundary::Periodic,
                                        TerrainShadowBoundary::Periodic};
    LambertianReflectionSweepWorkspace workspace;
    LambertianReflectionSweepResult    result;
    result.cell_absorbed_power_w    = {17.0};
    result.surface_incident_power_w = {23.0};
    result.emitted_power_w          = 31.0;

    REQUIRE_THROWS_AS(radiative_transfer::evaluate_mapped_lambertian_reflection_sweep_reference(
                          grid, source, slopes.data(), slopes.data(), {4, 8}, workspace, result),
                      std::invalid_argument);
    REQUIRE(result.cell_absorbed_power_w == std::vector<double>{17.0});
    REQUIRE(result.surface_incident_power_w == std::vector<double>{23.0});
    REQUIRE(result.emitted_power_w == 31.0);
}

TEST_CASE("Multiple Lambertian terrain reflection closes converged and truncated ledgers",
          "[physics][radiation][shortwave][terrain][reflection][multiple][conservation]")
{
    using radiative_transfer::LambertianMultipleReflectionOptions;
    using radiative_transfer::LambertianMultipleReflectionResult;
    using radiative_transfer::LambertianMultipleReflectionWorkspace;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    constexpr std::size_t                 nx = 3, ny = 1, nz = 2;
    const std::vector<double>             interfaces{0.0, 200.0, 400.0, 0.0, 200.0, 400.0, 200.0, 300.0, 400.0};
    const std::vector<double>             tau(nx * ny * nz, 0.0);
    const MappedDirectBeamGridView        grid{tau.data(),
                                               interfaces.data(),
                                               nx,
                                               ny,
                                               nz,
                                               100.0,
                                               100.0,
                                               TerrainShadowBoundary::Open,
                                               TerrainShadowBoundary::Periodic};
    const std::vector<double>             source{1000.0, 0.0, 0.0};
    const std::vector<double>             slope_x{0.0, 2.0, 0.0};
    const std::vector<double>             slope_y{0.0, 0.0, 0.0};
    LambertianMultipleReflectionWorkspace workspace;
    LambertianMultipleReflectionResult    result;
    LambertianMultipleReflectionOptions   options;
    options.quadrature                    = {1, 4};
    options.albedo                        = 0.5;
    options.maximum_bounces               = 32;
    options.relative_residual_tolerance   = 1.0e-10;
    options.absolute_residual_tolerance_w = 1.0e-10;
    radiative_transfer::evaluate_mapped_lambertian_multiple_reflection_reference(
        grid, source, slope_x.data(), slope_y.data(), options, workspace, result);

    REQUIRE(result.converged);
    REQUIRE(result.completed_bounces >= 2);
    REQUIRE(result.initial_reflected_power_w == 1000.0);
    REQUIRE(result.atmospheric_absorbed_power_w == 0.0);
    REQUIRE(result.surface_absorbed_power_w_total > 0.0);
    REQUIRE(result.lateral_escape_power_w > 0.0);
    REQUIRE(result.top_escape_power_w >= 0.0);
    REQUIRE(result.unresolved_reflected_power_w <= 1.0e-7);
    REQUIRE(result.initial_reflected_power_w ==
            Approx(result.atmospheric_absorbed_power_w + result.surface_absorbed_power_w_total +
                   result.top_escape_power_w + result.lateral_escape_power_w + result.unresolved_reflected_power_w)
                .epsilon(3.0e-14));
    REQUIRE(result.closure_residual_w == Approx(0.0).margin(3.0e-11));
    REQUIRE(result.surface_absorbed_power_w.size() == nx * ny);
    REQUIRE(result.interface_values_validated == nx * ny * (nz + 1));
    REQUIRE(result.optical_values_validated == nx * ny * nz);
    REQUIRE(result.surface_absorbed_power_w[0] + result.surface_absorbed_power_w[1] +
                result.surface_absorbed_power_w[2] ==
            Approx(result.surface_absorbed_power_w_total).epsilon(3.0e-14));

    LambertianMultipleReflectionResult truncated;
    options.maximum_bounces               = 1;
    options.relative_residual_tolerance   = 0.0;
    options.absolute_residual_tolerance_w = 0.0;
    radiative_transfer::evaluate_mapped_lambertian_multiple_reflection_reference(
        grid, source, slope_x.data(), slope_y.data(), options, workspace, truncated);
    REQUIRE_FALSE(truncated.converged);
    REQUIRE(truncated.completed_bounces == 1);
    REQUIRE(truncated.unresolved_reflected_power_w > 0.0);
    REQUIRE(truncated.initial_reflected_power_w ==
            Approx(truncated.surface_absorbed_power_w_total + truncated.top_escape_power_w +
                   truncated.lateral_escape_power_w + truncated.unresolved_reflected_power_w)
                .epsilon(3.0e-14));
}

TEST_CASE("Multiple Lambertian reflection publishes transactionally",
          "[physics][radiation][shortwave][terrain][reflection][multiple][transactional]")
{
    using radiative_transfer::LambertianMultipleReflectionOptions;
    using radiative_transfer::LambertianMultipleReflectionResult;
    using radiative_transfer::LambertianMultipleReflectionWorkspace;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;
    const std::vector<double>             interfaces{0.0, 100.0};
    const std::vector<double>             tau{0.0};
    const MappedDirectBeamGridView        grid{tau.data(),
                                               interfaces.data(),
                                               1,
                                               1,
                                               1,
                                               10.0,
                                               10.0,
                                               TerrainShadowBoundary::Periodic,
                                               TerrainShadowBoundary::Periodic};
    const std::vector<double>             source{50.0};
    const std::vector<double>             slopes{0.0};
    LambertianMultipleReflectionWorkspace workspace;
    LambertianMultipleReflectionResult    result;
    LambertianMultipleReflectionOptions   options;
    options.quadrature = {2, 4};
    options.albedo     = 0.8;
    radiative_transfer::evaluate_mapped_lambertian_multiple_reflection_reference(
        grid, source, slopes.data(), slopes.data(), options, workspace, result);
    REQUIRE(result.converged);
    REQUIRE(result.completed_bounces == 1);
    REQUIRE(result.top_escape_power_w == Approx(50.0).epsilon(2.0e-15));
    REQUIRE(result.unresolved_reflected_power_w == 0.0);

    const auto   committed_cells   = result.cell_absorbed_power_w;
    const auto   committed_surface = result.surface_absorbed_power_w;
    const double committed_escape  = result.top_escape_power_w;
    auto         malformed         = source;
    malformed[0]                   = -1.0;
    REQUIRE_THROWS_AS(radiative_transfer::evaluate_mapped_lambertian_multiple_reflection_reference(
                          grid, malformed, slopes.data(), slopes.data(), options, workspace, result),
                      std::invalid_argument);
    REQUIRE(result.cell_absorbed_power_w == committed_cells);
    REQUIRE(result.surface_absorbed_power_w == committed_surface);
    REQUIRE(result.top_escape_power_w == committed_escape);
    options.albedo = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(radiative_transfer::evaluate_mapped_lambertian_multiple_reflection_reference(
                          grid, source, slopes.data(), slopes.data(), options, workspace, result),
                      std::invalid_argument);
    REQUIRE(result.top_escape_power_w == committed_escape);
}

TEST_CASE("Complete terrain shortwave matches the flat periodic analytic ledger",
          "[physics][radiation][shortwave][terrain][coupled][analytical][conservation]")
{
    using radiative_transfer::LambertianMultipleReflectionOptions;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::MappedTerrainShortwaveResult;
    using radiative_transfer::MappedTerrainShortwaveWorkspace;
    using radiative_transfer::TerrainShadowBoundary;

    constexpr std::size_t               nz             = 2;
    constexpr double                    area_m2        = 200.0;
    constexpr double                    irradiance_wm2 = 1000.0;
    constexpr double                    albedo         = 0.2;
    const std::vector<double>           interfaces{0.0, 100.0, 300.0};
    const std::vector<double>           tau{0.1, 0.2};
    const std::vector<double>           slopes{0.0};
    const MappedDirectBeamGridView      grid{tau.data(),
                                             interfaces.data(),
                                             1,
                                             1,
                                             nz,
                                             10.0,
                                             20.0,
                                             TerrainShadowBoundary::Periodic,
                                             TerrainShadowBoundary::Periodic};
    LambertianMultipleReflectionOptions options;
    options.quadrature                    = {16, 2};
    options.albedo                        = albedo;
    options.maximum_bounces               = 4;
    options.relative_residual_tolerance   = 1.0e-13;
    options.absolute_residual_tolerance_w = 1.0e-12;
    MappedTerrainShortwaveWorkspace workspace;
    MappedTerrainShortwaveResult    result;
    radiative_transfer::evaluate_mapped_terrain_shortwave_reference(
        grid, {0.0, 0.0, 1.0}, irradiance_wm2, slopes.data(), slopes.data(), area_m2, options, workspace, result);

    const double incoming             = irradiance_wm2 * area_m2;
    const double direct_surface       = incoming * std::exp(-0.3);
    const double reflected            = albedo * direct_surface;
    const double diffuse_transmission = radiative_transfer::lambertian_diffuse_transmittance(0.3);
    const double expected_top_escape  = reflected * diffuse_transmission;
    const double expected_atmospheric = incoming - direct_surface + reflected - expected_top_escape;
    const double expected_surface     = (1.0 - albedo) * direct_surface;

    REQUIRE(result.reflection_converged);
    REQUIRE(workspace.reflection_result.completed_bounces == 1);
    REQUIRE(result.incoming_power_w == incoming);
    REQUIRE(result.atmospheric_absorbed_power_w_total == Approx(expected_atmospheric).margin(reflected * 2.0e-10));
    REQUIRE(result.surface_absorbed_power_w_total == Approx(expected_surface).epsilon(3.0e-15));
    REQUIRE(result.top_escape_power_w == Approx(expected_top_escape).margin(reflected * 2.0e-10));
    REQUIRE(result.lateral_escape_power_w == 0.0);
    REQUIRE(result.unresolved_reflected_power_w == 0.0);
    REQUIRE(result.closure_residual_w == Approx(0.0).margin(2.0e-10));
    REQUIRE(result.atmospheric_absorbed_power_w.size() == nz);
    REQUIRE(result.surface_absorbed_power_w == std::vector<double>{expected_surface});
    REQUIRE(result.atmospheric_absorbed_power_w[0] + result.atmospheric_absorbed_power_w[1] ==
            Approx(result.atmospheric_absorbed_power_w_total).epsilon(3.0e-15));
    REQUIRE(result.incoming_power_w == Approx(result.atmospheric_absorbed_power_w_total +
                                              result.surface_absorbed_power_w_total + result.top_escape_power_w)
                                           .epsilon(3.0e-15));
}

TEST_CASE("Complete terrain shortwave closes open terrain and reuses cadence storage",
          "[physics][radiation][shortwave][terrain][coupled][open][transactional][allocation]")
{
    using radiative_transfer::LambertianMultipleReflectionOptions;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::MappedTerrainShortwaveResult;
    using radiative_transfer::MappedTerrainShortwaveWorkspace;
    using radiative_transfer::TerrainShadowBoundary;

    constexpr std::size_t               nx = 3, ny = 1, nz = 2;
    constexpr double                    dx = 100.0, dy = 100.0;
    const std::vector<double>           interfaces{0.0, 200.0, 400.0, 0.0, 200.0, 400.0, 200.0, 300.0, 400.0};
    const std::vector<double>           tau{0.05, 0.10, 0.03, 0.08, 0.02, 0.06};
    const std::vector<double>           slope_x{0.0, 2.0, 0.0};
    const std::vector<double>           slope_y(nx * ny, 0.0);
    const MappedDirectBeamGridView      grid{tau.data(),
                                             interfaces.data(),
                                             nx,
                                             ny,
                                             nz,
                                             dx,
                                             dy,
                                             TerrainShadowBoundary::Open,
                                             TerrainShadowBoundary::Periodic};
    LambertianMultipleReflectionOptions options;
    options.quadrature                    = {4, 8};
    options.albedo                        = 0.45;
    options.maximum_bounces               = 32;
    options.relative_residual_tolerance   = 1.0e-11;
    options.absolute_residual_tolerance_w = 1.0e-6;
    MappedTerrainShortwaveWorkspace workspace;
    MappedTerrainShortwaveResult    result;
    const auto                      evaluate = [&]
    {
        radiative_transfer::evaluate_mapped_terrain_shortwave_reference(
            grid, {-0.6, 0.0, 0.8}, 900.0, slope_x.data(), slope_y.data(), dx * dy, options, workspace, result);
    };
    evaluate();

    REQUIRE(result.reflection_converged);
    REQUIRE(result.incoming_power_w > 0.0);
    REQUIRE(result.atmospheric_absorbed_power_w_total > 0.0);
    REQUIRE(result.surface_absorbed_power_w_total > 0.0);
    REQUIRE(result.lateral_escape_power_w > 0.0);
    REQUIRE(workspace.reflection_result.surface_absorbed_power_w_total > 0.0);
    REQUIRE(result.unresolved_reflected_power_w <=
            std::max(options.absolute_residual_tolerance_w,
                     options.relative_residual_tolerance * workspace.reflection_result.initial_reflected_power_w));
    REQUIRE(result.incoming_power_w ==
            Approx(result.atmospheric_absorbed_power_w_total + result.surface_absorbed_power_w_total +
                   result.top_escape_power_w + result.lateral_escape_power_w + result.unresolved_reflected_power_w)
                .epsilon(4.0e-14));
    REQUIRE(result.closure_residual_w == Approx(0.0).margin(2.0e-7));
    REQUIRE(result.atmospheric_absorbed_power_w.size() == nx * ny * nz);
    REQUIRE(result.surface_absorbed_power_w.size() == nx * ny);

    // Warm both transactional banks, then require stable steady-cadence storage.
    evaluate();
    const auto        committed_atmosphere      = result.atmospheric_absorbed_power_w;
    const auto        committed_surface         = result.surface_absorbed_power_w;
    const double      committed_incoming        = result.incoming_power_w;
    const std::size_t atmosphere_capacity       = workspace.scratch_atmospheric_absorbed_power_w.capacity();
    const std::size_t surface_capacity          = workspace.scratch_surface_absorbed_power_w.capacity();
    const std::size_t direct_batch_capacity     = workspace.direct_workspace.ray_batch.capacity();
    const std::size_t reflection_batch_capacity = workspace.reflection_workspace.sweep_workspace.ray_batch.capacity();
    evaluate();
    REQUIRE(result.atmospheric_absorbed_power_w == committed_atmosphere);
    REQUIRE(result.surface_absorbed_power_w == committed_surface);
    REQUIRE(workspace.scratch_atmospheric_absorbed_power_w.capacity() == atmosphere_capacity);
    REQUIRE(workspace.scratch_surface_absorbed_power_w.capacity() == surface_capacity);
    REQUIRE(workspace.direct_workspace.ray_batch.capacity() == direct_batch_capacity);
    REQUIRE(workspace.reflection_workspace.sweep_workspace.ray_batch.capacity() == reflection_batch_capacity);

    auto invalid_slope = slope_x;
    invalid_slope[1]   = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(
        radiative_transfer::evaluate_mapped_terrain_shortwave_reference(
            grid, {-0.6, 0.0, 0.8}, 900.0, invalid_slope.data(), slope_y.data(), dx * dy, options, workspace, result),
        std::invalid_argument);
    REQUIRE(result.atmospheric_absorbed_power_w == committed_atmosphere);
    REQUIRE(result.surface_absorbed_power_w == committed_surface);
    REQUIRE(result.incoming_power_w == committed_incoming);
}

TEST_CASE("Mapped spectral direct shortwave separates absorption and scattering",
          "[physics][radiation][rrtmg][shortwave][terrain][spectral][direct][conservation][transactional]")
{
    using radiative_transfer::MappedSpectralDirectResult;
    using radiative_transfer::MappedSpectralDirectWorkspace;
    using radiative_transfer::MappedSpectralShortwaveGridView;
    using radiative_transfer::TerrainShadowBoundary;

    const std::vector<double> interfaces{0.0, 1000.0};
    std::vector<double> absorption{0.2, 0.2};
    const std::vector<double> scattering{0.3, 0.3};
    std::vector<double> asymmetry{0.0, 0.7};
    const std::vector<double> source{0.25, 0.75};
    const MappedSpectralShortwaveGridView grid{
        absorption.data(), scattering.data(), asymmetry.data(), source.data(),
        interfaces.data(), 1, 1, 1, 2, 100.0, 100.0,
        TerrainShadowBoundary::Periodic, TerrainShadowBoundary::Periodic};
    MappedSpectralDirectWorkspace workspace;
    MappedSpectralDirectResult result;
    radiative_transfer::evaluate_mapped_spectral_direct_shortwave_reference(
        grid, {0.0, 0.0, 1.0}, 1000.0, workspace, result);

    REQUIRE(result.cell_absorbed_power_w.size() == 1);
    REQUIRE(result.cell_scattered_power_w.size() == 1);
    REQUIRE(result.surface_incident_power_w.size() == 1);
    REQUIRE(result.incoming_power_w == Approx(1.0e7));
    REQUIRE(result.atmospheric_absorbed_power_w /
                result.unresolved_scattered_power_w ==
            Approx(2.0 / 3.0).epsilon(2.0e-15));
    REQUIRE(result.incoming_power_w == Approx(
        result.atmospheric_absorbed_power_w +
        result.unresolved_scattered_power_w +
        result.surface_incident_power_w_total +
        result.top_escape_power_w + result.lateral_escape_power_w)
        .epsilon(3.0e-15));
    REQUIRE(std::abs(result.closure_residual_w) <= 1.0e-8);

    const auto held_absorbed = result.cell_absorbed_power_w;
    const auto held_scattered = result.cell_scattered_power_w;
    const double held_incoming = result.incoming_power_w;
    asymmetry[1] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(
        radiative_transfer::evaluate_mapped_spectral_direct_shortwave_reference(
            grid, {0.0, 0.0, 1.0}, 1000.0, workspace, result),
        std::invalid_argument);
    REQUIRE(result.cell_absorbed_power_w == held_absorbed);
    REQUIRE(result.cell_scattered_power_w == held_scattered);
    REQUIRE(result.incoming_power_w == held_incoming);
}

TEST_CASE("Mapped scattered bundles trace transactionally from cell centers",
          "[physics][radiation][shortwave][terrain][spectral][scattering][ray]")
{
    using radiative_transfer::DirectBeamPathSegment;
    using radiative_transfer::DirectBeamRayExit;
    using radiative_transfer::DirectBeamSegmentAbsorption;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    const std::vector<double> interfaces{0.0, 1000.0};
    const std::vector<double> extinction{0.5};
    const MappedDirectBeamGridView grid{
        extinction.data(), interfaces.data(), 1, 1, 1, 100.0, 100.0,
        TerrainShadowBoundary::Periodic, TerrainShadowBoundary::Periodic};
    std::vector<DirectBeamPathSegment> segments;
    std::vector<DirectBeamSegmentAbsorption> absorption;
    const auto upward =
        radiative_transfer::trace_mapped_radiative_ray_from_cell_reference(
            grid, 0, 0, 0, {0.0, 0.0, 1.0}, 100.0,
            segments, absorption);
    REQUIRE(upward.exit == DirectBeamRayExit::LevelTop);
    REQUIRE(upward.energy.absorbed_power_w > 0.0);
    REQUIRE(upward.energy.absorbed_power_w +
                upward.energy.transmitted_power_w == Approx(100.0));
    REQUIRE(segments.size() == 1);
    REQUIRE(absorption.size() == 1);

    const auto downward =
        radiative_transfer::trace_mapped_radiative_ray_from_cell_reference(
            grid, 0, 0, 0, {0.0, 0.0, -1.0}, 100.0,
            segments, absorption);
    REQUIRE(downward.exit == DirectBeamRayExit::Surface);
    REQUIRE(downward.energy.absorbed_power_w +
                downward.energy.transmitted_power_w == Approx(100.0));
    const auto held_segments = segments;
    const auto held_absorption = absorption;
    REQUIRE_THROWS_AS(
        radiative_transfer::trace_mapped_radiative_ray_from_cell_reference(
            grid, 0, 0, 1, {0.0, 0.0, 1.0}, 100.0,
            segments, absorption),
        std::invalid_argument);
    REQUIRE(segments.size() == held_segments.size());
    REQUIRE(absorption.size() == held_absorption.size());
    REQUIRE(segments.front().cell_index == held_segments.front().cell_index);
    REQUIRE(segments.front().path_length_m == held_segments.front().path_length_m);
    REQUIRE(absorption.front().cell_index == held_absorption.front().cell_index);
    REQUIRE(absorption.front().absorbed_power_w ==
            held_absorption.front().absorbed_power_w);
}

TEST_CASE("Mapped spectral scattering and terrain reflection iterate to closure",
          "[physics][radiation][rrtmg][shortwave][terrain][spectral][scattering][conservation][transactional]")
{
    using radiative_transfer::LambertianMultipleReflectionOptions;
    using radiative_transfer::MappedSpectralShortwaveGridView;
    using radiative_transfer::MappedSpectralTerrainResult;
    using radiative_transfer::MappedSpectralTerrainWorkspace;
    using radiative_transfer::TerrainShadowBoundary;

    const std::vector<double> interfaces{0.0, 1000.0};
    std::vector<double> absorption{0.2};
    const std::vector<double> scattering{0.1};
    std::vector<double> asymmetry{0.7};
    const std::vector<double> source{1.0};
    const std::vector<double> slopes{0.0};
    const MappedSpectralShortwaveGridView grid{
        absorption.data(), scattering.data(), asymmetry.data(), source.data(),
        interfaces.data(), 1, 1, 1, 1, 100.0, 100.0,
        TerrainShadowBoundary::Periodic, TerrainShadowBoundary::Periodic};
    LambertianMultipleReflectionOptions options;
    options.quadrature = {2, 4};
    options.albedo = 0.2;
    options.maximum_bounces = 64;
    options.relative_residual_tolerance = 1.0e-10;
    options.absolute_residual_tolerance_w = 1.0e-6;
    MappedSpectralTerrainWorkspace workspace;
    MappedSpectralTerrainResult result;
    radiative_transfer::evaluate_mapped_spectral_terrain_shortwave_reference(
        grid, {0.0, 0.0, 1.0}, 1000.0, slopes.data(), slopes.data(),
        10000.0, options, workspace, result);

    REQUIRE(result.converged);
    REQUIRE(result.rays > 0);
    REQUIRE(result.completed_iterations > 0);
    REQUIRE(result.atmospheric_absorbed_power_w.size() == 1);
    REQUIRE(result.surface_absorbed_power_w.size() == 1);
    REQUIRE(result.atmospheric_absorbed_power_w_total > 0.0);
    REQUIRE(result.surface_absorbed_power_w_total > 0.0);
    REQUIRE(result.unresolved_scattered_power_w <= 1.0e-3);
    REQUIRE(result.incoming_power_w == Approx(
        result.atmospheric_absorbed_power_w_total +
        result.surface_absorbed_power_w_total + result.top_escape_power_w +
        result.lateral_escape_power_w + result.unresolved_scattered_power_w)
        .epsilon(2.0e-12));
    REQUIRE(std::abs(result.closure_residual_w) <= 1.0e-5);

    const double forward_absorbed = result.atmospheric_absorbed_power_w_total;
    asymmetry[0] = 0.0;
    radiative_transfer::evaluate_mapped_spectral_terrain_shortwave_reference(
        grid, {0.0, 0.0, 1.0}, 1000.0, slopes.data(), slopes.data(),
        10000.0, options, workspace, result);
    REQUIRE(result.converged);
    REQUIRE(result.atmospheric_absorbed_power_w_total != forward_absorbed);

    const auto held_atmosphere = result.atmospheric_absorbed_power_w;
    const double held_incoming = result.incoming_power_w;
    absorption[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(
        radiative_transfer::evaluate_mapped_spectral_terrain_shortwave_reference(
            grid, {0.0, 0.0, 1.0}, 1000.0, slopes.data(), slopes.data(),
            10000.0, options, workspace, result),
        std::invalid_argument);
    REQUIRE(result.atmospheric_absorbed_power_w == held_atmosphere);
    REQUIRE(result.incoming_power_w == held_incoming);
}

TEST_CASE("Mapped spectral geometry cache preserves spectral and iteration arithmetic",
          "[physics][radiation][terrain][spectral][cache]")
{
    using namespace radiative_transfer;
    constexpr std::size_t nx=3, ny=2, nz=3, ng=2;
    std::vector<double> z(nx*ny*(nz+1)), absorption(nx*ny*nz*ng), scattering(absorption.size());
    std::vector<double> asymmetry(absorption.size(),0.4), source(nx*ny*ng), slopes(nx*ny,0.0);
    for (std::size_t c=0;c<nx*ny;++c)
    {
        source[c*ng]=0.3; source[c*ng+1]=0.7;
        for (std::size_t k=0;k<=nz;++k) z[c*(nz+1)+k]=300.0*k+10.0*c*(1.0-static_cast<double>(k)/nz);
        for (std::size_t k=0;k<nz;++k)
            for (std::size_t g=0;g<ng;++g)
            {
                absorption[(c*nz+k)*ng+g]=0.1+0.04*g+0.01*k;
                scattering[(c*nz+k)*ng+g]=0.05+0.02*g;
            }
    }
    MappedSpectralShortwaveGridView grid{absorption.data(),scattering.data(),asymmetry.data(),source.data(),
        z.data(),nx,ny,nz,ng,500.0,500.0,TerrainShadowBoundary::Open,TerrainShadowBoundary::Open};
    LambertianMultipleReflectionOptions options;
    options.quadrature={2,4}; options.albedo=0.2; options.maximum_bounces=64;
    MappedSpectralTerrainWorkspace uncached;
    uncached.geometry_cache_budget_bytes=0;
    MappedSpectralTerrainResult expected;
    evaluate_mapped_spectral_terrain_shortwave_reference(grid,{0.0,0.0,1.0},1000.0,
        slopes.data(),slopes.data(),250000.0,options,uncached,expected);
    REQUIRE(expected.converged);
    for (const std::size_t budget : {std::size_t(1024),mapped_geometry_cache_max_bytes})
    {
        MappedSpectralTerrainWorkspace cached;
        cached.geometry_cache_budget_bytes=budget;
        MappedSpectralTerrainResult result;
        evaluate_mapped_spectral_terrain_shortwave_reference(grid,{0.0,0.0,1.0},1000.0,
            slopes.data(),slopes.data(),250000.0,options,cached,result);
        CHECK(result.atmospheric_absorbed_power_w==expected.atmospheric_absorbed_power_w);
        CHECK(result.surface_absorbed_power_w==expected.surface_absorbed_power_w);
        CHECK(result.closure_residual_w==expected.closure_residual_w);
        CHECK(result.top_escape_power_w==expected.top_escape_power_w);
        CHECK(result.lateral_escape_power_w==expected.lateral_escape_power_w);
        CHECK(result.completed_iterations==expected.completed_iterations);
        CHECK(cached.geometry_cache_bytes<=budget);
        CHECK(cached.geometry_cache_hits>0);
        CHECK(cached.geometry_traces<uncached.geometry_traces);
    }
}

TEST_CASE("Reduced mapped RRTMG angular sweep tracks the production quadrature",
          "[physics][radiation][rrtmg][shortwave][terrain][spectral][quadrature][convergence]")
{
    using radiative_transfer::LambertianMultipleReflectionOptions;
    using radiative_transfer::MappedSpectralShortwaveGridView;
    using radiative_transfer::MappedSpectralTerrainResult;
    using radiative_transfer::MappedSpectralTerrainWorkspace;
    using radiative_transfer::TerrainShadowBoundary;
    constexpr std::size_t nx = 2, ny = 2, nz = 2, g_points = 2;
    const std::size_t columns = nx * ny;
    const std::size_t cells = columns * nz;
    std::vector<double> interfaces(columns * (nz + 1));
    std::vector<double> absorption(cells * g_points);
    std::vector<double> scattering(cells * g_points);
    std::vector<double> asymmetry(cells * g_points);
    std::vector<double> source(columns * g_points);
    std::vector<double> slope_x(columns);
    std::vector<double> slope_y(columns);
    for (std::size_t column = 0; column < columns; ++column)
    {
        interfaces[column * 3] = 20.0 * static_cast<double>(column % 2);
        interfaces[column * 3 + 1] = interfaces[column * 3] + 400.0;
        interfaces[column * 3 + 2] = 1000.0;
        source[column * 2] = 0.35 + 0.05 * static_cast<double>(column % 2);
        source[column * 2 + 1] = 1.0 - source[column * 2];
        slope_x[column] = 0.04 * static_cast<double>(column % 2);
        slope_y[column] = -0.03 * static_cast<double>(column / 2);
    }
    for (std::size_t cell = 0; cell < cells; ++cell)
        for (std::size_t g = 0; g < g_points; ++g)
        {
            absorption[cell * g_points + g] =
                0.04 + 0.01 * static_cast<double>((cell + g) % 3);
            scattering[cell * g_points + g] =
                0.08 + 0.015 * static_cast<double>((2 * cell + g) % 4);
            asymmetry[cell * g_points + g] = 0.55 + 0.1 * g;
        }
    const MappedSpectralShortwaveGridView grid{
        absorption.data(), scattering.data(), asymmetry.data(), source.data(),
        interfaces.data(), nx, ny, nz, g_points, 500.0, 500.0,
        TerrainShadowBoundary::Periodic, TerrainShadowBoundary::Periodic};
    LambertianMultipleReflectionOptions options;
    options.albedo = 0.25;
    options.maximum_bounces = 64;
    options.relative_residual_tolerance = 1.0e-8;
    options.absolute_residual_tolerance_w = 1.0e-6;
    MappedSpectralTerrainWorkspace workspace;
    MappedSpectralTerrainResult reduced;
    MappedSpectralTerrainResult production;
    options.quadrature = {1, 4};
    radiative_transfer::evaluate_mapped_spectral_terrain_shortwave_reference(
        grid, {0.3, -0.4, std::sqrt(0.75)}, 1000.0,
        slope_x.data(), slope_y.data(), 250000.0, options, workspace, reduced);
    options.quadrature = {4, 8};
    radiative_transfer::evaluate_mapped_spectral_terrain_shortwave_reference(
        grid, {0.3, -0.4, std::sqrt(0.75)}, 1000.0,
        slope_x.data(), slope_y.data(), 250000.0, options, workspace, production);
    INFO("reduced/production atmosphere="
         << reduced.atmospheric_absorbed_power_w_total << '/'
         << production.atmospheric_absorbed_power_w_total);
    INFO("reduced/production surface="
         << reduced.surface_absorbed_power_w_total << '/'
         << production.surface_absorbed_power_w_total);
    INFO("reduced/production top=" << reduced.top_escape_power_w << '/'
         << production.top_escape_power_w);
    REQUIRE(reduced.converged);
    REQUIRE(production.converged);
    REQUIRE(reduced.atmospheric_absorbed_power_w_total ==
            Approx(production.atmospheric_absorbed_power_w_total).epsilon(0.05));
    REQUIRE(reduced.surface_absorbed_power_w_total ==
            Approx(production.surface_absorbed_power_w_total).epsilon(0.05));
    REQUIRE(reduced.top_escape_power_w ==
            Approx(production.top_escape_power_w).epsilon(0.08));
}

TEST_CASE("Mapped direct-beam field injects sun-facing open sides without loss",
          "[physics][radiation][shortwave][terrain][atmosphere][field][open][conservation]")
{
    using radiative_transfer::DirectBeamFieldResult;
    using radiative_transfer::DirectBeamFieldWorkspace;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    constexpr std::size_t     nx = 2, ny = 1, nz = 2;
    constexpr double          spacing = 100.0;
    const std::vector<double> levels{0.0, 100.0, 200.0};
    std::vector<double>       interfaces(nx * ny * (nz + 1), 0.0);
    for (std::size_t column = 0; column < nx * ny; ++column)
        for (std::size_t k = 0; k <= nz; ++k)
            interfaces[column * (nz + 1) + k] = levels[k];
    const std::vector<double>      tau(nx * ny * nz, 0.0);
    const MappedDirectBeamGridView grid{tau.data(),
                                        interfaces.data(),
                                        nx,
                                        ny,
                                        nz,
                                        spacing,
                                        spacing,
                                        TerrainShadowBoundary::Open,
                                        TerrainShadowBoundary::Open};
    DirectBeamFieldWorkspace       workspace;
    DirectBeamFieldResult          result;
    radiative_transfer::evaluate_mapped_direct_beam_field_reference(grid, {-0.6, 0.0, 0.8}, 1000.0, workspace, result);
    REQUIRE(result.incoming_top_power_w == 16.0e6);
    REQUIRE(result.incoming_side_power_w == 12.0e6);
    REQUIRE(result.atmospheric_absorbed_power_w == 0.0);
    REQUIRE(result.surface_incident_total_power_w == 20.0e6);
    REQUIRE(result.lateral_escape_power_w == 8.0e6);
    REQUIRE(result.top_rays == 2);
    REQUIRE(result.side_rays == 2);
    REQUIRE(result.surface_rays == 3);
    REQUIRE(result.escaped_rays == 1);
    REQUIRE(result.interface_values_validated == nx * ny * (nz + 1));
    REQUIRE(result.optical_values_validated == nx * ny * nz);
    REQUIRE(result.boundary_closure_residual_w == 0.0);
    REQUIRE(result.surface_incident_power_w[0] == 6.0e6);
    REQUIRE(result.surface_incident_power_w[1] == 14.0e6);

    const auto   committed_surface = result.surface_incident_power_w;
    const double committed_total   = result.surface_incident_total_power_w;
    interfaces[1]                  = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(radiative_transfer::evaluate_mapped_direct_beam_field_reference(
                          grid, {-0.6, 0.0, 0.8}, 1000.0, workspace, result),
                      std::invalid_argument);
    REQUIRE(result.surface_incident_power_w == committed_surface);
    REQUIRE(result.surface_incident_total_power_w == committed_total);
}

TEST_CASE("Mapped direct-beam diagonal field covers both incoming side axes",
          "[physics][radiation][shortwave][terrain][atmosphere][field][open][matrix]")
{
    using radiative_transfer::DirectBeamFieldResult;
    using radiative_transfer::DirectBeamFieldWorkspace;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;
    constexpr std::size_t     nx = 2, ny = 2, nz = 2;
    constexpr double          dx = 10.0, dy = 20.0, top = 300.0;
    const std::vector<double> levels{0.0, 100.0, top};
    std::vector<double>       interfaces(nx * ny * (nz + 1), 0.0);
    for (std::size_t column = 0; column < nx * ny; ++column)
        for (std::size_t k = 0; k <= nz; ++k)
            interfaces[column * (nz + 1) + k] = levels[k];
    const std::vector<double>      tau(nx * ny * nz, 0.0);
    const MappedDirectBeamGridView grid{
        tau.data(), interfaces.data(), nx, ny, nz, dx, dy, TerrainShadowBoundary::Open, TerrainShadowBoundary::Open};
    DirectBeamFieldWorkspace     workspace;
    DirectBeamFieldResult        result;
    const IdealizedSolarGeometry sun{-0.3, -0.4, std::sqrt(0.75)};
    radiative_transfer::evaluate_mapped_direct_beam_field_reference(grid, sun, 500.0, workspace, result);
    const double expected_top    = 500.0 * sun.up * nx * dx * ny * dy;
    const double expected_x_side = 500.0 * 0.3 * ny * dy * top;
    const double expected_y_side = 500.0 * 0.4 * nx * dx * top;
    REQUIRE(result.incoming_top_power_w == Approx(expected_top).epsilon(2.0e-15));
    REQUIRE(result.incoming_side_power_w == Approx(expected_x_side + expected_y_side).epsilon(2.0e-15));
    REQUIRE(result.atmospheric_absorbed_power_w == 0.0);
    REQUIRE(result.surface_incident_total_power_w + result.lateral_escape_power_w ==
            Approx(expected_top + expected_x_side + expected_y_side).epsilon(3.0e-15));
    REQUIRE(result.top_rays == 4);
    REQUIRE(result.side_rays == 8);
    REQUIRE(result.surface_rays + result.escaped_rays == 12);
    REQUIRE(result.boundary_closure_residual_w == Approx(0.0).margin(2.0e-9));
}

TEST_CASE("Mapped direct-beam batches are bitwise deterministic and allocation stable",
          "[physics][radiation][shortwave][terrain][field][parallel][deterministic][allocation]")
{
    using radiative_transfer::DirectBeamFieldResult;
    using radiative_transfer::DirectBeamFieldWorkspace;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;
    constexpr std::size_t nx = 19, ny = 17, nz = 2;
    std::vector<double>   interfaces(nx * ny * (nz + 1), 0.0);
    std::vector<double>   tau(nx * ny * nz, 0.0);
    for (std::size_t column = 0; column < nx * ny; ++column)
    {
        interfaces[column * (nz + 1)]     = 0.0;
        interfaces[column * (nz + 1) + 1] = 100.0;
        interfaces[column * (nz + 1) + 2] = 300.0;
        tau[column * nz]                  = 0.01 + 0.0001 * static_cast<double>(column % 11);
        tau[column * nz + 1]              = 0.02 + 0.0001 * static_cast<double>(column % 7);
    }
    const MappedDirectBeamGridView grid{tau.data(),
                                        interfaces.data(),
                                        nx,
                                        ny,
                                        nz,
                                        80.0,
                                        120.0,
                                        TerrainShadowBoundary::Periodic,
                                        TerrainShadowBoundary::Periodic};
    const IdealizedSolarGeometry   sun{-0.3, -0.4, std::sqrt(0.75)};
    DirectBeamFieldWorkspace       workspace;
    DirectBeamFieldResult          result;
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    radiative_transfer::evaluate_mapped_direct_beam_field_reference(grid, sun, 900.0, workspace, result);
    const auto   one_cell          = result.cell_absorbed_power_w;
    const auto   one_surface       = result.surface_incident_power_w;
    const double one_atmosphere    = result.atmospheric_absorbed_power_w;
    const double one_surface_total = result.surface_incident_total_power_w;
    const double one_residual      = result.boundary_closure_residual_w;
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    radiative_transfer::evaluate_mapped_direct_beam_field_reference(grid, sun, 900.0, workspace, result);
    const bool               cells_equal   = result.cell_absorbed_power_w == one_cell;
    const bool               surface_equal = result.surface_incident_power_w == one_surface;
    const bool               totals_equal  = result.atmospheric_absorbed_power_w == one_atmosphere &&
                                             result.surface_incident_total_power_w == one_surface_total &&
                                             result.boundary_closure_residual_w == one_residual;
    std::vector<std::size_t> segment_capacities;
    std::vector<std::size_t> absorption_capacities;
    for (const auto& scratch : workspace.ray_batch)
    {
        segment_capacities.push_back(scratch.segments.capacity());
        absorption_capacities.push_back(scratch.absorption.capacity());
    }
    radiative_transfer::evaluate_mapped_direct_beam_field_reference(grid, sun, 900.0, workspace, result);
    bool capacities_equal = true;
    for (std::size_t slot = 0; slot < workspace.ray_batch.size(); ++slot)
        capacities_equal = capacities_equal &&
                           workspace.ray_batch[slot].segments.capacity() == segment_capacities[slot] &&
                           workspace.ray_batch[slot].absorption.capacity() == absorption_capacities[slot];
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
    REQUIRE(workspace.ray_batch.size() == 64);
    REQUIRE(result.top_rays == nx * ny);
    REQUIRE(result.side_rays == 0);
    REQUIRE(result.surface_rays == nx * ny);
    REQUIRE(result.escaped_rays == 0);
    REQUIRE(cells_equal);
    REQUIRE(surface_equal);
    REQUIRE(totals_equal);
    REQUIRE(result.cell_absorbed_power_w == one_cell);
    REQUIRE(result.surface_incident_power_w == one_surface);
    REQUIRE(capacities_equal);
}

TEST_CASE("Mapped direct-beam persistent team rejects a failed side bundle transactionally",
          "[physics][radiation][shortwave][terrain][field][parallel][transactional]")
{
    using radiative_transfer::DirectBeamFieldResult;
    using radiative_transfer::DirectBeamFieldWorkspace;
    using radiative_transfer::MappedDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    constexpr std::size_t nx = 16, ny = 16, nz = 1;
    std::vector<double>   interfaces(nx * ny * (nz + 1), 0.0);
    for (std::size_t column = 0; column < nx * ny; ++column)
        interfaces[column * (nz + 1) + 1] = 10.0;
    const std::vector<double>      tau(nx * ny * nz, 0.0);
    const MappedDirectBeamGridView grid{tau.data(),
                                        interfaces.data(),
                                        nx,
                                        ny,
                                        nz,
                                        1.0,
                                        1.0,
                                        TerrainShadowBoundary::Open,
                                        TerrainShadowBoundary::Periodic};
    // The almost-horizontal top bundle is finite (1 W per cell), while the
    // sun-facing open-x side bundle exceeds binary64. The failure is first
    // discovered by a worker after the 256 top rays have already traced.
    const IdealizedSolarGeometry sun{1.0, 0.0, 1.0e-308};
    DirectBeamFieldWorkspace     workspace;
    DirectBeamFieldResult        result;
    result.cell_absorbed_power_w    = {17.0};
    result.surface_incident_power_w = {23.0};
    result.incoming_top_power_w     = 31.0;

    REQUIRE_THROWS_AS(
        radiative_transfer::evaluate_mapped_direct_beam_field_reference(grid, sun, 1.0e308, workspace, result),
        std::runtime_error);
    REQUIRE(result.cell_absorbed_power_w == std::vector<double>{17.0});
    REQUIRE(result.surface_incident_power_w == std::vector<double>{23.0});
    REQUIRE(result.incoming_top_power_w == 31.0);
}

TEST_CASE("Rectilinear 3-D direct beam resolves periodic cell crossings",
          "[physics][radiation][shortwave][atmosphere][ray][rectilinear][reference]")
{
    using radiative_transfer::DirectBeamPathSegment;
    using radiative_transfer::DirectBeamRayExit;
    using radiative_transfer::DirectBeamSegmentAbsorption;
    using radiative_transfer::RectilinearDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    const std::vector<double> z_interface{0.0, 100.0, 200.0};
    // [i][j][k], with one y cell and two vertical layers.
    const std::vector<double> vertical_tau{
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
    };
    const RectilinearDirectBeamGridView      grid{vertical_tau.data(),
                                                  z_interface.data(),
                                                  3,
                                                  1,
                                                  2,
                                                  100.0,
                                                  100.0,
                                                  TerrainShadowBoundary::Periodic,
                                                  TerrainShadowBoundary::Open};
    const IdealizedSolarGeometry             sun{-0.8, 0.0, 0.6};
    constexpr double                         top_normal_irradiance_wm2 = 1000.0;
    constexpr double                         horizontal_area_m2        = 10000.0;
    constexpr double                         incident_power_w = top_normal_irradiance_wm2 * 0.6 * horizontal_area_m2;
    std::vector<DirectBeamPathSegment>       segments;
    std::vector<DirectBeamSegmentAbsorption> absorption;
    const auto result = radiative_transfer::trace_rectilinear_direct_beam_from_top_reference(
        grid, 0, 0, sun, incident_power_w, segments, absorption);

    REQUIRE(result.exit == DirectBeamRayExit::Surface);
    REQUIRE(result.terminal_i == 0);
    REQUIRE(result.terminal_j == 0);
    REQUIRE(result.terminal_x_m == Approx(50.0 / 3.0).epsilon(2.0e-15));
    REQUIRE(result.terminal_y_m == 50.0);
    REQUIRE(result.terminal_z_m == 0.0);
    REQUIRE(segments.size() == 5);
    REQUIRE(absorption.size() == segments.size());
    const std::vector<std::size_t> expected_cells{1, 3, 2, 4, 0};
    const std::vector<double>      expected_lengths{62.5, 625.0 / 6.0, 125.0 / 6.0, 125.0, 125.0 / 6.0};
    double                         total_path_length  = 0.0;
    double                         expected_total_tau = 0.0;
    for (std::size_t index = 0; index < segments.size(); ++index)
    {
        INFO("periodic rectilinear segment " << index);
        REQUIRE(segments[index].cell_index == expected_cells[index]);
        REQUIRE(segments[index].path_length_m == Approx(expected_lengths[index]).epsilon(3.0e-15));
        REQUIRE(absorption[index].cell_index == expected_cells[index]);
        total_path_length += segments[index].path_length_m;
        expected_total_tau += vertical_tau[expected_cells[index]] * expected_lengths[index] / 100.0;
    }
    REQUIRE(total_path_length == Approx(200.0 / 0.6).epsilon(3.0e-15));
    REQUIRE(result.energy.total_slant_optical_depth == Approx(expected_total_tau).epsilon(3.0e-15));
    REQUIRE(result.energy.transmitted_power_w ==
            Approx(incident_power_w * std::exp(-expected_total_tau)).epsilon(3.0e-15));
    REQUIRE(result.energy.absorbed_power_w + result.energy.transmitted_power_w == result.energy.incident_power_w);
}

TEST_CASE("Rectilinear direct beam distinguishes surface and open-side power",
          "[physics][radiation][shortwave][atmosphere][ray][rectilinear][boundary]")
{
    using radiative_transfer::DirectBeamPathSegment;
    using radiative_transfer::DirectBeamRayExit;
    using radiative_transfer::DirectBeamSegmentAbsorption;
    using radiative_transfer::RectilinearDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    const std::vector<double> z_interface{0.0, 100.0, 200.0};
    std::vector<double>       vertical_tau{
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
    };
    RectilinearDirectBeamGridView            grid{vertical_tau.data(),
                                                  z_interface.data(),
                                                  3,
                                                  1,
                                                  2,
                                                  100.0,
                                                  100.0,
                                                  TerrainShadowBoundary::Open,
                                                  TerrainShadowBoundary::Open};
    std::vector<DirectBeamPathSegment>       segments;
    std::vector<DirectBeamSegmentAbsorption> absorption;
    const auto lateral = radiative_transfer::trace_rectilinear_direct_beam_from_top_reference(
        grid, 0, 0, {-0.8, 0.0, 0.6}, 500.0, segments, absorption);
    REQUIRE(lateral.exit == DirectBeamRayExit::OpenLateralBoundary);
    REQUIRE(lateral.terminal_i == 2);
    REQUIRE(lateral.terminal_j == 0);
    REQUIRE(lateral.terminal_x_m == 300.0);
    REQUIRE(lateral.terminal_z_m == Approx(12.5).epsilon(2.0e-15));
    REQUIRE(segments.size() == 4);
    REQUIRE(lateral.energy.absorbed_power_w + lateral.energy.transmitted_power_w == lateral.energy.incident_power_w);

    const auto vertical = radiative_transfer::trace_rectilinear_direct_beam_from_top_reference(
        grid, 2, 0, {0.0, 0.0, 1.0}, 500.0, segments, absorption);
    REQUIRE(vertical.exit == DirectBeamRayExit::Surface);
    REQUIRE(vertical.terminal_i == 2);
    REQUIRE(vertical.terminal_j == 0);
    REQUIRE(segments.size() == 2);
    REQUIRE(segments[0].cell_index == 5);
    REQUIRE(segments[1].cell_index == 4);
    REQUIRE(segments[0].path_length_m == 100.0);
    REQUIRE(segments[1].path_length_m == 100.0);
    REQUIRE(vertical.energy.total_slant_optical_depth == Approx(0.11));
    REQUIRE(vertical.energy.transmitted_power_w == Approx(500.0 * std::exp(-0.11)).epsilon(2.0e-15));

    const std::vector<DirectBeamPathSegment>       committed_segments{{77, 1.0, 1.0, 1.0}};
    const std::vector<DirectBeamSegmentAbsorption> committed_absorption{{77, 1.0, 2.0, 1.0, 1.0}};
    segments        = committed_segments;
    absorption      = committed_absorption;
    vertical_tau[2] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(radiative_transfer::trace_rectilinear_direct_beam_from_top_reference(
                          grid, 0, 0, {0.0, 0.0, 1.0}, 500.0, segments, absorption),
                      std::invalid_argument);
    REQUIRE(segments.size() == 1);
    REQUIRE(segments[0].cell_index == 77);
    REQUIRE(absorption.size() == 1);
    REQUIRE(absorption[0].cell_index == 77);
}

TEST_CASE("Rectilinear periodic ray matrix preserves analytic path and column depth",
          "[physics][radiation][shortwave][atmosphere][ray][rectilinear][matrix]")
{
    using radiative_transfer::DirectBeamPathSegment;
    using radiative_transfer::DirectBeamRayExit;
    using radiative_transfer::DirectBeamSegmentAbsorption;
    using radiative_transfer::RectilinearDirectBeamGridView;
    using radiative_transfer::TerrainShadowBoundary;

    constexpr std::size_t     nx  = 4;
    constexpr std::size_t     ny  = 3;
    constexpr std::size_t     nz  = 3;
    constexpr double          dx  = 80.0;
    constexpr double          dy  = 120.0;
    constexpr double          top = 300.0;
    const std::vector<double> z_interface{0.0, 60.0, 170.0, top};
    const std::vector<double> layer_tau{0.03, 0.07, 0.11};
    std::vector<double>       vertical_tau(nx * ny * nz, 0.0);
    for (std::size_t i = 0; i < nx; ++i)
        for (std::size_t j = 0; j < ny; ++j)
            for (std::size_t k = 0; k < nz; ++k)
                vertical_tau[(i * ny + j) * nz + k] = layer_tau[k];
    const RectilinearDirectBeamGridView       grid{vertical_tau.data(),
                                                   z_interface.data(),
                                                   nx,
                                                   ny,
                                                   nz,
                                                   dx,
                                                   dy,
                                                   TerrainShadowBoundary::Periodic,
                                                   TerrainShadowBoundary::Periodic};
    const std::vector<IdealizedSolarGeometry> directions{
        {0.3, 0.4, std::sqrt(0.75)},
        {-0.6, 0.0, 0.8},
        {0.5, 0.5, std::sqrt(0.5)},
    };
    const auto wrap = [](double coordinate, double extent)
    {
        double result = std::fmod(coordinate, extent);
        if (result < 0.0)
            result += extent;
        return result;
    };

    std::vector<DirectBeamPathSegment>       segments;
    std::vector<DirectBeamSegmentAbsorption> absorption;
    for (const IdealizedSolarGeometry& sun : directions)
    {
        constexpr std::size_t top_i  = 2;
        constexpr std::size_t top_j  = 1;
        const auto            result = radiative_transfer::trace_rectilinear_direct_beam_from_top_reference(
            grid, top_i, top_j, sun, 900.0, segments, absorption);
        const double expected_path = top / sun.up;
        const double expected_x =
            wrap((static_cast<double>(top_i) + 0.5) * dx - sun.east * expected_path, static_cast<double>(nx) * dx);
        const double expected_y =
            wrap((static_cast<double>(top_j) + 0.5) * dy - sun.north * expected_path, static_cast<double>(ny) * dy);
        double resolved_path = 0.0;
        for (const DirectBeamPathSegment& segment : segments)
            resolved_path += segment.path_length_m;
        INFO("periodic matrix sun=" << sun.east << ',' << sun.north << ',' << sun.up);
        REQUIRE(result.exit == DirectBeamRayExit::Surface);
        REQUIRE(result.terminal_x_m == Approx(expected_x).margin(2.0e-12));
        REQUIRE(result.terminal_y_m == Approx(expected_y).margin(2.0e-12));
        REQUIRE(result.terminal_i == static_cast<std::size_t>(std::floor(expected_x / dx)));
        REQUIRE(result.terminal_j == static_cast<std::size_t>(std::floor(expected_y / dy)));
        REQUIRE(resolved_path == Approx(expected_path).epsilon(5.0e-15));
        REQUIRE(result.energy.total_slant_optical_depth == Approx((0.03 + 0.07 + 0.11) / sun.up).epsilon(5.0e-15));
        REQUIRE(result.energy.transmitted_power_w == Approx(900.0 * std::exp(-0.21 / sun.up)).epsilon(5.0e-15));
        REQUIRE(result.energy.absorbed_power_w + result.energy.transmitted_power_w == result.energy.incident_power_w);
        REQUIRE(absorption.size() == segments.size());
    }
}

TEST_CASE("Simple grey column computation produces finite tendencies", "[physics][radiation][analytical]")
{
    auto      scheme = create_radiation_scheme("simple_grey");
    const int nz     = 10;

    // Build a simple isothermal column
    std::vector<double> T(nz, 280.0);
    std::vector<double> p_col(nz);
    std::vector<double> p_int_col(nz + 1);
    std::vector<double> rho_col(nz, 1.0);
    std::vector<double> dz_col(nz, 1000.0);
    std::vector<double> z_int(nz + 1);
    std::vector<double> qv_col(nz, 0.005);

    // Pressure decreasing with height
    for (int k = 0; k < nz; ++k)
    {
        p_col[k] = 100000.0 * std::exp(-(static_cast<double>(k) + 0.5) * 1000.0 / 8500.0);
        z_int[k] = k * 1000.0;
    }
    z_int[nz] = nz * 1000.0;
    for (int k = 0; k <= nz; ++k)
        p_int_col[k] = 100000.0 * std::exp(-static_cast<double>(k) * 1000.0 / 8500.0);

    RadiationConfig cfg;
    cfg.do_lw      = true;
    cfg.do_sw      = true;
    cfg.tau_lw_ref = 3.0;
    cfg.tau_sw_ref = 0.1;
    cfg.n_lw       = 1.0;
    cfg.n_sw       = 1.0;

    RadiationColumnStateView col;
    col.T             = &T;
    col.p             = &p_col;
    col.p_int         = &p_int_col;
    col.rho           = &rho_col;
    col.dz            = &dz_col;
    col.z_int         = &z_int;
    col.qv            = &qv_col;
    col.mu0           = 0.5; // 60 degree solar zenith
    col.S0            = 1366.0;
    col.Tsfc          = 288.0;
    col.albedo_sw     = 0.2;
    col.emissivity_lw = 0.95;

    RadiationColumnTendencies tend;
    RadiationColumnFluxes     fluxes;

    scheme->initialize(cfg);
    scheme->compute_column(cfg, col, tend, &fluxes);

    // All tendencies must be finite
    REQUIRE(tend.dTdt_rad.size() == static_cast<size_t>(nz));
    for (int k = 0; k < nz; ++k)
    {
        REQUIRE(std::isfinite(tend.dTdt_rad[k]));
        REQUIRE(std::isfinite(tend.dTdt_lw[k]));
        REQUIRE(std::isfinite(tend.dTdt_sw[k]));
    }

    // Total tendency = LW + SW at each level
    for (int k = 0; k < nz; ++k)
    {
        REQUIRE(tend.dTdt_rad[k] == Approx(tend.dTdt_lw[k] + tend.dTdt_sw[k]).margin(1e-15));
    }
}

TEST_CASE("Simple grey rejects inconsistent interfaces before changing output",
          "[physics][radiation][analytical][rollback]")
{
    auto            scheme = create_radiation_scheme("simple_grey");
    RadiationConfig cfg;
    scheme->initialize(cfg);

    const std::vector<double> temperature{280.0};
    const std::vector<double> pressure{90000.0};
    const std::vector<double> pressure_interfaces{100000.0, 80000.0};
    const std::vector<double> density{1.0};
    const std::vector<double> thickness{500.0};
    const std::vector<double> inconsistent_heights{0.0, 600.0};
    RadiationColumnStateView  col;
    col.T             = &temperature;
    col.p             = &pressure;
    col.p_int         = &pressure_interfaces;
    col.rho           = &density;
    col.dz            = &thickness;
    col.z_int         = &inconsistent_heights;
    col.mu0           = 0.5;
    col.S0            = 1000.0;
    col.Tsfc          = 290.0;
    col.albedo_sw     = 0.2;
    col.emissivity_lw = 0.95;

    RadiationColumnTendencies published;
    published.dTdt_rad = {7.0};
    published.dTdt_lw  = {8.0};
    published.dTdt_sw  = {9.0};
    REQUIRE_THROWS_AS(scheme->compute_column(cfg, col, published), std::invalid_argument);
    REQUIRE(published.dTdt_rad == std::vector<double>{7.0});
    REQUIRE(published.dTdt_lw == std::vector<double>{8.0});
    REQUIRE(published.dTdt_sw == std::vector<double>{9.0});
}

TEST_CASE("Radiation fluxes have correct sign conventions", "[physics][radiation][analytical]")
{
    auto      scheme = create_radiation_scheme("simple_grey");
    const int nz     = 8;

    std::vector<double> T(nz, 280.0);
    std::vector<double> p_col(nz);
    std::vector<double> p_int_col(nz + 1);
    std::vector<double> rho_col(nz, 1.0);
    std::vector<double> dz_col(nz, 1000.0);
    std::vector<double> z_int(nz + 1);
    std::vector<double> qv_col(nz, 0.005);

    for (int k = 0; k < nz; ++k)
    {
        p_col[k] = 100000.0 * std::exp(-(static_cast<double>(k) + 0.5) * 1000.0 / 8500.0);
        z_int[k] = k * 1000.0;
    }
    z_int[nz] = nz * 1000.0;
    for (int k = 0; k <= nz; ++k)
        p_int_col[k] = 100000.0 * std::exp(-static_cast<double>(k) * 1000.0 / 8500.0);

    RadiationConfig cfg;
    cfg.do_lw      = true;
    cfg.do_sw      = true;
    cfg.tau_lw_ref = 3.0;
    cfg.tau_sw_ref = 0.1;
    cfg.n_lw       = 1.0;
    cfg.n_sw       = 1.0;

    RadiationColumnStateView col;
    col.T             = &T;
    col.p             = &p_col;
    col.p_int         = &p_int_col;
    col.rho           = &rho_col;
    col.dz            = &dz_col;
    col.z_int         = &z_int;
    col.qv            = &qv_col;
    col.mu0           = 0.5;
    col.S0            = 1366.0;
    col.Tsfc          = 288.0;
    col.albedo_sw     = 0.2;
    col.emissivity_lw = 0.95;

    RadiationColumnTendencies tend;
    RadiationColumnFluxes     fluxes;
    // Reused diagnostics must clear the disabled band rather than publishing
    // values retained from a previous column/cadence.
    fluxes.Fup_lw.assign(nz + 1, 99.0);
    fluxes.Fdn_lw.assign(nz + 1, 99.0);
    fluxes.Fnet_lw.assign(nz + 1, 99.0);
    scheme->initialize(cfg);
    scheme->compute_column(cfg, col, tend, &fluxes);

    // SW downward flux at TOA should be mu0 * S0
    REQUIRE(fluxes.Fdn_sw.size() == static_cast<size_t>(nz + 1));
    double expected_toa_sw = col.mu0 * col.S0;
    REQUIRE(fluxes.Fdn_sw[nz] == Approx(expected_toa_sw).epsilon(1e-10));

    // SW downward flux should decrease toward surface (absorption)
    for (int k = nz - 1; k >= 0; --k)
    {
        REQUIRE(fluxes.Fdn_sw[k] <= fluxes.Fdn_sw[k + 1] + 1e-10);
    }

    // LW downward flux at TOA should be zero (no incoming LW from space)
    REQUIRE(fluxes.Fdn_lw[nz] == Approx(0.0).margin(1e-10));

    // All flux values must be finite and non-negative
    for (size_t k = 0; k <= static_cast<size_t>(nz); ++k)
    {
        REQUIRE(std::isfinite(fluxes.Fup_lw[k]));
        REQUIRE(std::isfinite(fluxes.Fdn_lw[k]));
        REQUIRE(std::isfinite(fluxes.Fup_sw[k]));
        REQUIRE(std::isfinite(fluxes.Fdn_sw[k]));
        REQUIRE(fluxes.Fup_lw[k] >= 0.0);
        REQUIRE(fluxes.Fdn_sw[k] >= 0.0);
    }
}

TEST_CASE("One-layer simple-grey fluxes match the closed-form solution", "[physics][radiation][analytical][reference]")
{
    auto            scheme = create_radiation_scheme("simple_grey");
    RadiationConfig cfg;
    cfg.do_lw      = true;
    cfg.do_sw      = true;
    cfg.tau_lw_ref = 0.7;
    cfg.tau_sw_ref = 0.2;
    cfg.n_lw       = 1.0;
    cfg.n_sw       = 1.0;
    scheme->initialize(cfg);

    const std::vector<double> temperature{275.0};
    const std::vector<double> pressure{100000.0};
    const std::vector<double> pressure_interfaces{100000.0, 0.0};
    const std::vector<double> density{1.0};
    const std::vector<double> thickness{500.0};
    const std::vector<double> height_interfaces{0.0, 500.0};
    const std::vector<double> dry_qv{0.0};
    const std::vector<double> moist_qv{0.02};
    RadiationColumnStateView  col;
    col.T             = &temperature;
    col.p             = &pressure;
    col.p_int         = &pressure_interfaces;
    col.rho           = &density;
    col.dz            = &thickness;
    col.z_int         = &height_interfaces;
    col.qv            = &dry_qv;
    col.mu0           = 0.6;
    col.S0            = 1000.0;
    col.Tsfc          = 290.0;
    col.albedo_sw     = 0.25;
    col.emissivity_lw = 0.8;

    RadiationColumnTendencies dry_tendencies;
    RadiationColumnFluxes     dry_fluxes;
    scheme->compute_column(cfg, col, dry_tendencies, &dry_fluxes);

    const double sw_transmission          = std::exp(-cfg.tau_sw_ref / col.mu0);
    const double expected_sw_down_top     = col.mu0 * col.S0;
    const double expected_sw_down_surface = expected_sw_down_top * sw_transmission;
    const double expected_sw_up_surface   = col.albedo_sw * expected_sw_down_surface;
    const double expected_sw_up_top =
        expected_sw_up_surface * radiative_transfer::lambertian_diffuse_transmittance(cfg.tau_sw_ref);
    REQUIRE(dry_fluxes.Fdn_sw[1] == Approx(expected_sw_down_top).epsilon(1.0e-12));
    REQUIRE(dry_fluxes.Fdn_sw[0] == Approx(expected_sw_down_surface).epsilon(1.0e-12));
    REQUIRE(dry_fluxes.Fup_sw[0] == Approx(expected_sw_up_surface).epsilon(1.0e-12));
    REQUIRE(dry_fluxes.Fup_sw[1] == Approx(expected_sw_up_top).epsilon(1.0e-12));

    const double lw_transmission          = std::exp(-cfg.tau_lw_ref);
    const double atmospheric_emission     = radiative_transfer::stefan_boltzmann(temperature[0]);
    const double expected_lw_down_surface = atmospheric_emission * (1.0 - lw_transmission);
    const double expected_lw_up_surface   = col.emissivity_lw * radiative_transfer::stefan_boltzmann(col.Tsfc) +
                                            (1.0 - col.emissivity_lw) * expected_lw_down_surface;
    const double expected_lw_up_top =
        expected_lw_up_surface * lw_transmission + atmospheric_emission * (1.0 - lw_transmission);
    REQUIRE(dry_fluxes.Fdn_lw[1] == Approx(0.0).margin(1.0e-14));
    REQUIRE(dry_fluxes.Fdn_lw[0] == Approx(expected_lw_down_surface).epsilon(1.0e-12));
    REQUIRE(dry_fluxes.Fup_lw[0] == Approx(expected_lw_up_surface).epsilon(1.0e-12));
    REQUIRE(dry_fluxes.Fup_lw[1] == Approx(expected_lw_up_top).epsilon(1.0e-12));

    const double expected_lw_atmospheric_power = -(dry_fluxes.Fnet_lw.back() - dry_fluxes.Fnet_lw.front());
    const double expected_sw_atmospheric_power = -(dry_fluxes.Fnet_sw.back() - dry_fluxes.Fnet_sw.front());
    const double integrated_atmospheric_power  = density[0] * 1004.0 * thickness[0] * dry_tendencies.dTdt_rad[0];
    REQUIRE(dry_tendencies.atmospheric_power_lw_wm2 == Approx(expected_lw_atmospheric_power).epsilon(1.0e-12));
    REQUIRE(dry_tendencies.atmospheric_power_sw_wm2 == Approx(expected_sw_atmospheric_power).epsilon(1.0e-12));
    REQUIRE(dry_tendencies.atmospheric_power_total_wm2 ==
            Approx(expected_lw_atmospheric_power + expected_sw_atmospheric_power).epsilon(1.0e-12));
    REQUIRE(dry_tendencies.atmospheric_power_total_wm2 == Approx(integrated_atmospheric_power).epsilon(1.0e-12));

    // The declared model is pressure-coordinate clear-sky grey radiation.
    // Water vapor is intentionally outside its optical-depth contract.
    col.qv = &moist_qv;
    RadiationColumnTendencies moist_tendencies;
    RadiationColumnFluxes     moist_fluxes;
    scheme->compute_column(cfg, col, moist_tendencies, &moist_fluxes);
    REQUIRE(moist_tendencies.dTdt_rad[0] == Approx(dry_tendencies.dTdt_rad[0]).epsilon(1.0e-14));
    REQUIRE(moist_fluxes.Fnet_total[0] == Approx(dry_fluxes.Fnet_total[0]).epsilon(1.0e-14));
    REQUIRE(moist_fluxes.Fnet_total[1] == Approx(dry_fluxes.Fnet_total[1]).epsilon(1.0e-14));
}

TEST_CASE("Absorbing water-cloud simple grey matches its one-layer closed form",
          "[physics][radiation][analytical][reference][sci352]")
{
    RadiationConfig cfg;
    cfg.clear_sky_only = false;
    cfg.simple_grey_optical_model = "absorbing_water_cloud_path_v1";
    cfg.do_lw = true;
    cfg.do_sw = true;
    cfg.tau_lw_ref = 0.3;
    cfg.tau_sw_ref = 0.1;
    cfg.n_lw = 1.0;
    cfg.n_sw = 1.0;

    const std::vector<double> temperature{270.0};
    const std::vector<double> pressure{100000.0};
    const std::vector<double> pressure_interfaces{100000.0, 0.0};
    const std::vector<double> density{0.8};
    const std::vector<double> thickness{250.0};
    const std::vector<double> height_interfaces{0.0, 250.0};
    const std::vector<double> vapor{0.01};
    const std::vector<double> liquid{2.0e-4};
    const std::vector<double> ice{1.0e-4};

    RadiationColumnStateView column;
    column.T = &temperature;
    column.p = &pressure;
    column.p_int = &pressure_interfaces;
    column.rho = &density;
    column.dz = &thickness;
    column.z_int = &height_interfaces;
    column.qv = &vapor;
    column.qc = &liquid;
    column.qi = &ice;
    column.mu0 = 0.7;
    column.S0 = 1000.0;
    column.Tsfc = 290.0;
    column.albedo_sw = 0.2;
    column.emissivity_lw = 0.9;

    auto scheme = create_radiation_scheme("simple_grey");
    scheme->initialize(cfg);
    RadiationColumnTendencies tendencies;
    RadiationColumnFluxes fluxes;
    scheme->compute_column(cfg, column, tendencies, &fluxes);

    const auto& coefficients =
        tmv::simple_grey_absorbing_water_cloud_coefficients();
    const double mass_path = density[0] * thickness[0];
    const double tau_lw = cfg.tau_lw_ref + mass_path *
        (coefficients.longwave_vapor_m2kg * vapor[0]
         + coefficients.longwave_liquid_m2kg * liquid[0]
         + coefficients.longwave_ice_m2kg * ice[0]);
    const double tau_sw = cfg.tau_sw_ref + mass_path *
        (coefficients.shortwave_vapor_m2kg * vapor[0]
         + coefficients.shortwave_liquid_m2kg * liquid[0]
         + coefficients.shortwave_ice_m2kg * ice[0]);
    const double emission =
        radiative_transfer::stefan_boltzmann(temperature[0]);
    const double expected_lw_down = emission * (1.0 - std::exp(-tau_lw));
    const double expected_lw_up_surface =
        column.emissivity_lw
            * radiative_transfer::stefan_boltzmann(column.Tsfc)
        + (1.0 - column.emissivity_lw) * expected_lw_down;
    const double expected_lw_up_top =
        expected_lw_up_surface * std::exp(-tau_lw)
        + emission * (1.0 - std::exp(-tau_lw));
    const double expected_sw_top = column.mu0 * column.S0;
    const double expected_sw_surface =
        expected_sw_top * std::exp(-tau_sw / column.mu0);
    const double expected_sw_up_surface =
        column.albedo_sw * expected_sw_surface;
    const double expected_sw_up_top = expected_sw_up_surface
        * radiative_transfer::lambertian_diffuse_transmittance(tau_sw);

    REQUIRE(fluxes.Fdn_lw[0] == Approx(expected_lw_down).epsilon(1.0e-13));
    REQUIRE(fluxes.Fup_lw[0] == Approx(expected_lw_up_surface).epsilon(1.0e-13));
    REQUIRE(fluxes.Fup_lw[1] == Approx(expected_lw_up_top).epsilon(1.0e-13));
    REQUIRE(fluxes.Fdn_sw[0] == Approx(expected_sw_surface).epsilon(1.0e-13));
    REQUIRE(fluxes.Fup_sw[0] == Approx(expected_sw_up_surface).epsilon(1.0e-13));
    REQUIRE(fluxes.Fup_sw[1] == Approx(expected_sw_up_top).epsilon(1.0e-13));

    const double net_lower = fluxes.Fnet_total.front();
    const double net_upper = fluxes.Fnet_total.back();
    REQUIRE(tendencies.atmospheric_power_total_wm2 ==
            Approx(-(net_upper - net_lower)).epsilon(1.0e-13));
    REQUIRE(density[0] * 1004.0 * thickness[0]
                * tendencies.dTdt_rad[0] ==
            Approx(tendencies.atmospheric_power_total_wm2).epsilon(1.0e-13));
}

TEST_CASE("Absorbing water-cloud simple grey is species-sensitive and failure atomic",
          "[physics][radiation][guard][rollback][sci352]")
{
    RadiationConfig cfg;
    cfg.clear_sky_only = false;
    cfg.simple_grey_optical_model = "absorbing_water_cloud_path_v1";
    cfg.tau_lw_ref = 0.1;
    cfg.tau_sw_ref = 0.05;
    cfg.n_lw = 1.0;
    cfg.n_sw = 1.0;
    RadiationConfig invalid_coefficients = cfg;
    invalid_coefficients.simple_grey_lw_liquid_absorption_m2kg = -1.0;
    auto invalid_scheme = create_radiation_scheme("simple_grey");
    REQUIRE_THROWS_AS(invalid_scheme->initialize(invalid_coefficients),
                      std::invalid_argument);
    invalid_coefficients = cfg;
    invalid_coefficients.simple_grey_sw_ice_absorption_m2kg =
        std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(invalid_scheme->initialize(invalid_coefficients),
                      std::invalid_argument);
    auto scheme = create_radiation_scheme("simple_grey");
    scheme->initialize(cfg);

    const std::vector<double> temperature{275.0};
    const std::vector<double> pressure{90000.0};
    const std::vector<double> pressure_interfaces{100000.0, 80000.0};
    const std::vector<double> density{1.0};
    const std::vector<double> thickness{500.0};
    const std::vector<double> height_interfaces{0.0, 500.0};
    const std::vector<double> zero{0.0};
    const std::vector<double> vapor{0.015};
    const std::vector<double> liquid{1.0e-4};
    const std::vector<double> ice{1.0e-4};
    const std::vector<double> invalid{-1.0e-4};
    RadiationColumnStateView column;
    column.T = &temperature;
    column.p = &pressure;
    column.p_int = &pressure_interfaces;
    column.rho = &density;
    column.dz = &thickness;
    column.z_int = &height_interfaces;
    column.qv = &zero;
    column.qc = &zero;
    column.qi = &zero;
    column.mu0 = 0.6;
    column.S0 = 1000.0;
    column.Tsfc = 290.0;
    column.albedo_sw = 0.2;
    column.emissivity_lw = 0.95;

    const auto solve = [&]()
    {
        RadiationColumnTendencies tendencies;
        RadiationColumnFluxes fluxes;
        scheme->compute_column(cfg, column, tendencies, &fluxes);
        return std::make_pair(std::move(tendencies), std::move(fluxes));
    };
    const auto dry = solve();
    column.qv = &vapor;
    const auto moist = solve();
    column.qv = &zero;
    column.qc = &liquid;
    const auto cloudy_liquid = solve();
    column.qc = &zero;
    column.qi = &ice;
    const auto cloudy_ice = solve();
    REQUIRE(moist.second.Fdn_sw[0] < dry.second.Fdn_sw[0]);
    REQUIRE(cloudy_liquid.second.Fdn_sw[0] < dry.second.Fdn_sw[0]);
    REQUIRE(cloudy_ice.second.Fdn_sw[0] < dry.second.Fdn_sw[0]);
    REQUIRE(moist.second.Fdn_lw[0] > dry.second.Fdn_lw[0]);
    REQUIRE(cloudy_liquid.second.Fdn_lw[0] > dry.second.Fdn_lw[0]);
    REQUIRE(cloudy_ice.second.Fdn_lw[0] > dry.second.Fdn_lw[0]);

    RadiationColumnTendencies published;
    published.dTdt_rad = {7.0};
    published.dTdt_lw = {8.0};
    published.dTdt_sw = {9.0};
    published.atmospheric_power_total_wm2 = 10.0;
    RadiationColumnFluxes published_fluxes;
    published_fluxes.Fup_lw = {11.0, 12.0};
    column.qi = &invalid;
    REQUIRE_THROWS_AS(
        scheme->compute_column(cfg, column, published, &published_fluxes),
        std::invalid_argument);
    REQUIRE(published.dTdt_rad == std::vector<double>{7.0});
    REQUIRE(published.dTdt_lw == std::vector<double>{8.0});
    REQUIRE(published.dTdt_sw == std::vector<double>{9.0});
    REQUIRE(published.atmospheric_power_total_wm2 == 10.0);
    REQUIRE(published_fluxes.Fup_lw == std::vector<double>{11.0, 12.0});
}

TEST_CASE("Simple-grey atmospheric column matches the published two-stream equations",
          "[physics][radiation][published-reference][RA-005][sci088]")
{
    // Independent benchmark vector from the isothermal closed-form solution
    // of the grey two-stream equations and U(surface)=sigma*Ts^4,
    // D(model top)=0 boundary conditions stated by Thuburn & Craig (2000),
    // J. Atmos. Sci. 57, 17-28:
    // https://doi.org/10.1175/1520-0469(2000)057<0017:SIOTHT>2.0.CO;2
    // The vector was evaluated outside the implementation at the pressure
    // interfaces below; it is intentionally not reconstructed with Nimbus's
    // layer recurrence or optical-depth helpers.
    constexpr std::size_t     nz = 10;
    const std::vector<double> pressure_interfaces{
        100000.0, 91000.0, 82000.0, 73000.0, 64000.0, 55000.0, 46000.0, 37000.0, 28000.0, 19000.0, 10000.0};
    const std::vector<double> expected_upward{390.105153541862,
                                              297.905944437903,
                                              257.473668822749,
                                              239.133182764094,
                                              230.520094071240,
                                              226.329312632374,
                                              224.215892833986,
                                              223.111747554867,
                                              222.515676230450,
                                              222.185932735782,
                                              222.004252824167};
    const std::vector<double> expected_downward{220.835247792350,
                                                220.034306227941,
                                                218.388121969634,
                                                215.152642791337,
                                                209.093316952059,
                                                198.330138394221,
                                                180.307511533136,
                                                152.106315913050,
                                                111.421761177084,
                                                58.5821001973895,
                                                0.0};

    std::vector<double>       pressure(nz, 0.0);
    std::vector<double>       height_interfaces(nz + 1, 0.0);
    std::vector<double>       thickness(nz, 0.0);
    std::vector<double>       density(nz, 0.0);
    const std::vector<double> temperature(nz, 250.0);
    constexpr double          scale_height_m   = 8000.0;
    constexpr double          gas_constant_dry = 287.05;
    for (std::size_t k = 0; k <= nz; ++k)
        height_interfaces[k] = -scale_height_m * std::log(pressure_interfaces[k] / pressure_interfaces.front());
    for (std::size_t k = 0; k < nz; ++k)
    {
        pressure[k]  = std::sqrt(pressure_interfaces[k] * pressure_interfaces[k + 1]);
        thickness[k] = height_interfaces[k + 1] - height_interfaces[k];
        density[k]   = pressure[k] / (gas_constant_dry * temperature[k]);
    }

    RadiationConfig cfg;
    cfg.do_lw      = true;
    cfg.do_sw      = false;
    cfg.tau_lw_ref = 6.0;
    cfg.n_lw       = 1.5;
    auto scheme    = create_radiation_scheme("simple_grey");
    scheme->initialize(cfg);

    RadiationColumnStateView column;
    column.T             = &temperature;
    column.p             = &pressure;
    column.p_int         = &pressure_interfaces;
    column.rho           = &density;
    column.dz            = &thickness;
    column.z_int         = &height_interfaces;
    column.mu0           = 0.0;
    column.S0            = 0.0;
    column.Tsfc          = 288.0;
    column.albedo_sw     = 0.0;
    column.emissivity_lw = 1.0;

    RadiationColumnTendencies tendencies;
    RadiationColumnFluxes     fluxes;
    scheme->compute_column(cfg, column, tendencies, &fluxes);
    REQUIRE(fluxes.Fup_lw.size() == expected_upward.size());
    REQUIRE(fluxes.Fdn_lw.size() == expected_downward.size());
    for (std::size_t k = 0; k <= nz; ++k)
    {
        INFO("published two-stream interface " << k);
        REQUIRE(fluxes.Fup_lw[k] == Approx(expected_upward[k]).epsilon(2.0e-13));
        REQUIRE(fluxes.Fdn_lw[k] == Approx(expected_downward[k]).epsilon(2.0e-13));
    }

    double integrated_power = 0.0;
    for (std::size_t k = 0; k < nz; ++k)
    {
        const double expected_net_lower = expected_upward[k] - expected_downward[k];
        const double expected_net_upper = expected_upward[k + 1] - expected_downward[k + 1];
        const double expected_heating =
            -(expected_net_upper - expected_net_lower) / (density[k] * 1004.0 * thickness[k]);
        REQUIRE(tendencies.dTdt_lw[k] == Approx(expected_heating).epsilon(3.0e-12));
        integrated_power += density[k] * 1004.0 * thickness[k] * tendencies.dTdt_lw[k];
    }
    REQUIRE(tendencies.atmospheric_power_lw_wm2 == Approx(integrated_power).epsilon(2.0e-13));
}

TEST_CASE("Radiation SW with zero optical depth is transparent", "[physics][radiation][analytical]")
{
    auto      scheme = create_radiation_scheme("simple_grey");
    const int nz     = 5;

    std::vector<double> T(nz, 280.0);
    std::vector<double> p_col(nz, 100000.0); // uniform pressure = zero tau gradient
    std::vector<double> p_int_col(nz + 1, 100000.0);
    std::vector<double> rho_col(nz, 1.0);
    std::vector<double> dz_col(nz, 1000.0);
    std::vector<double> z_int(nz + 1);
    std::vector<double> qv_col(nz, 0.0);

    for (int k = 0; k <= nz; ++k)
        z_int[k] = k * 1000.0;

    RadiationConfig cfg;
    cfg.do_lw      = false;
    cfg.do_sw      = true;
    cfg.tau_sw_ref = 0.0; // zero optical depth = transparent atmosphere
    cfg.n_sw       = 1.0;

    RadiationColumnStateView col;
    col.T             = &T;
    col.p             = &p_col;
    col.p_int         = &p_int_col;
    col.rho           = &rho_col;
    col.dz            = &dz_col;
    col.z_int         = &z_int;
    col.qv            = &qv_col;
    col.mu0           = 1.0;
    col.S0            = 1000.0;
    col.Tsfc          = 288.0;
    col.albedo_sw     = 0.0;
    col.emissivity_lw = 0.0;

    RadiationColumnTendencies tend;
    RadiationColumnFluxes     fluxes;
    scheme->initialize(cfg);
    scheme->compute_column(cfg, col, tend, &fluxes);

    // With zero optical depth, SW flux should pass through unchanged
    // Fdn should be approximately S0*mu0 at all levels
    for (int k = 0; k <= nz; ++k)
    {
        REQUIRE(fluxes.Fdn_sw[k] == Approx(1000.0).margin(1.0));
        REQUIRE(fluxes.Fup_lw[k] == 0.0);
        REQUIRE(fluxes.Fdn_lw[k] == 0.0);
        REQUIRE(fluxes.Fnet_lw[k] == 0.0);
    }

    // SW heating should be near zero (no absorption)
    for (int k = 0; k < nz; ++k)
    {
        REQUIRE(std::abs(tend.dTdt_sw[k]) < 1.0e-6);
    }
}

TEST_CASE("Grey LW surface reflects atmospheric downwelling for emissivity below one",
          "[physics][radiation][analytical][conservation]")
{
    const std::vector<double> temperature{280.0, 270.0};
    const std::vector<double> optical_depth{0.8, 0.6};
    constexpr double          surface_temperature = 300.0;
    constexpr double          emissivity          = 0.6;
    std::vector<double>       upward, downward;

    radiative_transfer::grey_lw_two_stream(
        temperature, optical_depth, surface_temperature, emissivity, upward, downward);

    REQUIRE(downward[0] > 0.0);
    const double expected_surface_up =
        emissivity * radiative_transfer::stefan_boltzmann(surface_temperature) + (1.0 - emissivity) * downward[0];
    REQUIRE(upward[0] == Approx(expected_surface_up).epsilon(1.0e-12));
}

TEST_CASE("Radiation column heating closes the signed net-flux budget",
          "[physics][radiation][analytical][conservation]")
{
    const std::vector<double> rho{1.2, 0.9, 0.6};
    const std::vector<double> dz_layers{200.0, 400.0, 800.0};
    const std::vector<double> net_flux{120.0, 95.0, 60.0, 35.0};
    std::vector<double>       heating;
    radiative_transfer::heating_rate_from_flux_divergence(rho, dz_layers, net_flux, heating);

    const double residual = radiative_transfer::check_energy_conservation(rho, dz_layers, heating, net_flux);
    INFO("column energy residual = " << residual << " W/m2");
    REQUIRE(residual == Approx(0.0).margin(1.0e-11));
}

TEST_CASE("Simple-grey coupled column closes energy and converges with radiation cadence",
          "[physics][radiation][analytical][conservation][cadence][RA-005]")
{
    struct CoupledResult
    {
        std::vector<double> temperature;
        double              relative_energy_residual = 0.0;
    };
    const auto integrate = [](double radiation_cadence_s)
    {
        constexpr std::size_t nz            = 6;
        constexpr double      step_s        = 10.0;
        constexpr double      duration_s    = 6.0 * 3600.0;
        constexpr double      cp            = 1004.0;
        const int             cadence_steps = static_cast<int>(std::llround(radiation_cadence_s / step_s));
        REQUIRE(cadence_steps > 0);
        REQUIRE(static_cast<double>(cadence_steps) * step_s == Approx(radiation_cadence_s));

        std::vector<double>       temperature{288.0, 282.0, 276.0, 270.0, 264.0, 258.0};
        const std::vector<double> initial_temperature = temperature;
        const std::vector<double> density{1.15, 1.02, 0.90, 0.78, 0.66, 0.54};
        const std::vector<double> thickness(nz, 750.0);
        std::vector<double>       height_interfaces(nz + 1, 0.0);
        std::vector<double>       pressure_interfaces(nz + 1, 0.0);
        std::vector<double>       pressure(nz, 0.0);
        for (std::size_t k = 0; k <= nz; ++k)
        {
            height_interfaces[k]   = 750.0 * static_cast<double>(k);
            pressure_interfaces[k] = 100000.0 * std::exp(-height_interfaces[k] / 8200.0);
        }
        for (std::size_t k = 0; k < nz; ++k)
        {
            pressure[k] = std::sqrt(pressure_interfaces[k] * pressure_interfaces[k + 1]);
        }

        RadiationConfig cfg;
        cfg.do_lw      = true;
        cfg.do_sw      = false;
        cfg.tau_lw_ref = 2.5;
        cfg.n_lw       = 1.0;
        auto scheme    = create_radiation_scheme("simple_grey");
        scheme->initialize(cfg);

        RadiationColumnStateView col;
        col.T             = &temperature;
        col.p             = &pressure;
        col.p_int         = &pressure_interfaces;
        col.rho           = &density;
        col.dz            = &thickness;
        col.z_int         = &height_interfaces;
        col.mu0           = 0.0;
        col.S0            = 0.0;
        col.Tsfc          = 290.0;
        col.albedo_sw     = 0.2;
        col.emissivity_lw = 0.96;

        RadiationColumnTendencies held_tendency;
        RadiationColumnFluxes     held_fluxes;
        long double               accumulated_boundary_energy = 0.0L;
        const int                 total_steps                 = static_cast<int>(duration_s / step_s);
        for (int step = 0; step < total_steps; ++step)
        {
            if (step % cadence_steps == 0)
            {
                scheme->compute_column(cfg, col, held_tendency, &held_fluxes);
            }
            const double boundary_power = -(held_fluxes.Fnet_total.back() - held_fluxes.Fnet_total.front());
            REQUIRE(held_tendency.atmospheric_power_total_wm2 == Approx(boundary_power).epsilon(1.0e-12));
            accumulated_boundary_energy += step_s * boundary_power;
            for (std::size_t k = 0; k < nz; ++k)
            {
                temperature[k] += step_s * held_tendency.dTdt_rad[k];
            }
        }

        long double column_energy_change = 0.0L;
        long double energy_scale         = 0.0L;
        for (std::size_t k = 0; k < nz; ++k)
        {
            const double heat_capacity = density[k] * cp * thickness[k];
            column_energy_change += heat_capacity * (temperature[k] - initial_temperature[k]);
            energy_scale += heat_capacity * initial_temperature[k];
        }
        CoupledResult result;
        result.temperature = std::move(temperature);
        result.relative_energy_residual =
            std::abs(static_cast<double>((column_energy_change - accumulated_boundary_energy) / energy_scale));
        return result;
    };

    const CoupledResult reference                  = integrate(10.0);
    const CoupledResult one_minute                 = integrate(60.0);
    const CoupledResult five_minutes               = integrate(300.0);
    const CoupledResult fifteen_minutes            = integrate(900.0);
    const auto          max_temperature_difference = [](const std::vector<double>& lhs, const std::vector<double>& rhs)
    {
        double value = 0.0;
        for (std::size_t k = 0; k < lhs.size(); ++k)
            value = std::max(value, std::abs(lhs[k] - rhs[k]));
        return value;
    };
    const double error_60  = max_temperature_difference(one_minute.temperature, reference.temperature);
    const double error_300 = max_temperature_difference(five_minutes.temperature, reference.temperature);
    const double error_900 = max_temperature_difference(fifteen_minutes.temperature, reference.temperature);
    INFO("six-hour simple-grey cadence max temperature errors [K]: 60s=" << error_60 << " 300s=" << error_300
                                                                         << " 900s=" << error_900);
    INFO("six-hour simple-grey relative energy residuals: reference="
         << reference.relative_energy_residual << " 60s=" << one_minute.relative_energy_residual
         << " 300s=" << five_minutes.relative_energy_residual << " 900s=" << fifteen_minutes.relative_energy_residual);

    REQUIRE(error_60 < error_300);
    REQUIRE(error_300 < error_900);
    REQUIRE(error_900 < 0.01);
    REQUIRE(reference.relative_energy_residual < 1.0e-12);
    REQUIRE(one_minute.relative_energy_residual < 1.0e-12);
    REQUIRE(five_minutes.relative_energy_residual < 1.0e-12);
    REQUIRE(fifteen_minutes.relative_energy_residual < 1.0e-12);
}

TEST_CASE("Radiation diagnoses bounded log-pressure model interfaces", "[physics][radiation][geometry][analytical]")
{
    constexpr double          scale_height_m = 8000.0;
    const std::vector<double> z_center{500.0, 1500.0, 2500.0};
    const std::vector<double> z_interface{0.0, 1000.0, 2000.0, 3000.0};
    std::vector<double>       p_center;
    for (double z : z_center)
        p_center.push_back(100000.0 * std::exp(-z / scale_height_m));

    const std::vector<double> p_interface =
        radiative_transfer::diagnose_pressure_interfaces(p_center, z_center, z_interface);
    REQUIRE(p_interface.size() == z_interface.size());
    for (std::size_t k = 0; k < p_interface.size(); ++k)
    {
        REQUIRE(p_interface[k] == Approx(100000.0 * std::exp(-z_interface[k] / scale_height_m)).epsilon(1.0e-12));
        REQUIRE(p_interface[k] > 0.0);
    }

    const auto tau = radiative_transfer::compute_optical_depth_profile(p_interface, 2.0, 1.0);
    REQUIRE(tau.size() == p_interface.size());
    REQUIRE(tau.back() > 0.0);
    REQUIRE(tau.front() > tau.back());
}

TEST_CASE("Radiation primitives reject malformed or non-physical columns", "[physics][radiation][guard]")
{
    REQUIRE_THROWS_AS(radiative_transfer::compute_optical_depth_profile({100000.0, 110000.0}, 3.0, 1.0),
                      std::invalid_argument);

    std::vector<double> heating;
    REQUIRE_THROWS_AS(
        radiative_transfer::heating_rate_from_flux_divergence({1.0, -1.0}, {100.0, 100.0}, {0.0, 1.0, 2.0}, heating),
        std::invalid_argument);

    std::vector<double> upward, downward;
    REQUIRE_THROWS_AS(radiative_transfer::grey_lw_two_stream({280.0}, {-0.1}, 290.0, 0.9, upward, downward),
                      std::invalid_argument);

    auto            scheme = create_radiation_scheme("simple_grey");
    RadiationConfig cfg;
    scheme->initialize(cfg);
    RadiationColumnStateView  missing_fields;
    RadiationColumnTendencies tendencies;
    REQUIRE_THROWS_AS(scheme->compute_column(cfg, missing_fields, tendencies), std::invalid_argument);
}
