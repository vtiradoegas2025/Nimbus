#include "simple_grey.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{
constexpr const char* clear_sky_model = "pressure_clear_sky_v1";
constexpr const char* absorbing_water_cloud_model =
    "absorbing_water_cloud_path_v1";

tmv::SimpleGreyAbsorptionCoefficients absorption_coefficients(
    const RadiationConfig& cfg)
{
    return {
        cfg.simple_grey_lw_vapor_absorption_m2kg,
        cfg.simple_grey_lw_liquid_absorption_m2kg,
        cfg.simple_grey_lw_ice_absorption_m2kg,
        cfg.simple_grey_sw_vapor_absorption_m2kg,
        cfg.simple_grey_sw_liquid_absorption_m2kg,
        cfg.simple_grey_sw_ice_absorption_m2kg,
    };
}

bool same_coefficients(const tmv::SimpleGreyAbsorptionCoefficients& left,
                       const tmv::SimpleGreyAbsorptionCoefficients& right)
{
    return left.longwave_vapor_m2kg == right.longwave_vapor_m2kg &&
           left.longwave_liquid_m2kg == right.longwave_liquid_m2kg &&
           left.longwave_ice_m2kg == right.longwave_ice_m2kg &&
           left.shortwave_vapor_m2kg == right.shortwave_vapor_m2kg &&
           left.shortwave_liquid_m2kg == right.shortwave_liquid_m2kg &&
           left.shortwave_ice_m2kg == right.shortwave_ice_m2kg;
}

void validate_simple_grey_config(const RadiationConfig& cfg,
                                 bool validate_solar_forcing)
{
    if (!std::isfinite(cfg.dt_radiation) || cfg.dt_radiation <= 0.0 ||
        !std::isfinite(cfg.tau_lw_ref) || cfg.tau_lw_ref < 0.0 ||
        !std::isfinite(cfg.tau_sw_ref) || cfg.tau_sw_ref < 0.0 ||
        !std::isfinite(cfg.n_lw) || cfg.n_lw <= 0.0 ||
        !std::isfinite(cfg.n_sw) || cfg.n_sw <= 0.0 ||
        (!cfg.do_lw && !cfg.do_sw) ||
        (!cfg.use_subcycling && cfg.subcycle_count != 1) ||
        (cfg.use_subcycling &&
         (cfg.subcycle_count < 2 || cfg.subcycle_count > 64)))
    {
        throw std::invalid_argument(
            "simple_grey configuration violates its cadence, band, or optical contract");
    }
    const bool pressure_clear_sky =
        cfg.simple_grey_optical_model == clear_sky_model;
    const bool absorbing_water_cloud =
        cfg.simple_grey_optical_model == absorbing_water_cloud_model;
    if ((!pressure_clear_sky && !absorbing_water_cloud) ||
        (pressure_clear_sky && !cfg.clear_sky_only) ||
        (absorbing_water_cloud && cfg.clear_sky_only))
        throw std::invalid_argument(
            "simple_grey optical model and clear_sky_only selection disagree");
    const auto coefficients = absorption_coefficients(cfg);
    const double values[] = {
        coefficients.longwave_vapor_m2kg,
        coefficients.longwave_liquid_m2kg,
        coefficients.longwave_ice_m2kg,
        coefficients.shortwave_vapor_m2kg,
        coefficients.shortwave_liquid_m2kg,
        coefficients.shortwave_ice_m2kg,
    };
    for (double value : values)
        if (!std::isfinite(value) || value < 0.0)
            throw std::invalid_argument(
                "simple_grey mass-absorption coefficients must be finite and non-negative");
    if (validate_solar_forcing)
        (void)idealized_solar_cosine_zenith(cfg, 12.0 * 3600.0);
}

void add_water_path_optical_depth(
    const RadiationColumnStateView& col,
    const tmv::SimpleGreyAbsorptionCoefficients& coefficients,
    bool longwave,
    std::vector<double>& optical_depth)
{
    const double vapor = longwave
        ? coefficients.longwave_vapor_m2kg
        : coefficients.shortwave_vapor_m2kg;
    const double liquid = longwave
        ? coefficients.longwave_liquid_m2kg
        : coefficients.shortwave_liquid_m2kg;
    const double ice = longwave
        ? coefficients.longwave_ice_m2kg
        : coefficients.shortwave_ice_m2kg;
    for (std::size_t k = 0; k < optical_depth.size(); ++k)
    {
        // rho*dz*q is absorber mass path [kg/m2]; kappa is [m2/kg].
        const double absorber =
            vapor * (*col.qv)[k] + liquid * (*col.qc)[k]
            + ice * (*col.qi)[k];
        optical_depth[k] += (*col.rho)[k] * (*col.dz)[k] * absorber;
        if (!std::isfinite(optical_depth[k]))
            throw std::overflow_error(
                "simple_grey water-path optical depth overflowed");
    }
}
}

namespace tmv
{

SimpleGreyScheme::SimpleGreyScheme()
    : tau_lw_ref_(6.0), tau_sw_ref_(0.22), n_lw_(4.0), n_sw_(2.0),
      optical_model_(clear_sky_model),
      absorption_coefficients_(
          simple_grey_absorbing_water_cloud_coefficients())
{}

int SimpleGreyScheme::required_fields() const 
{
    int requirements = static_cast<int>(RadiationRequirements::TEMPERATURE) |
                       static_cast<int>(RadiationRequirements::SOLAR_GEOMETRY) |
                       static_cast<int>(RadiationRequirements::INTERFACE_GEOMETRY);
    if (optical_model_ == absorbing_water_cloud_model)
        requirements |= static_cast<int>(RadiationRequirements::CLOUD_OPTICS);
    return requirements;
}

void SimpleGreyScheme::initialize(const RadiationConfig& cfg)
{
    validate_simple_grey_config(cfg, true);
    tau_lw_ref_ = cfg.tau_lw_ref;
    tau_sw_ref_ = cfg.tau_sw_ref;
    n_lw_ = cfg.n_lw;
    n_sw_ = cfg.n_sw;
    optical_model_ = cfg.simple_grey_optical_model;
    absorption_coefficients_ = absorption_coefficients(cfg);

    tmv::log_info("Initialized Simple Grey Radiation:");
    tmv::log_info("  LW optical depth: ", tau_lw_ref_, " (exponent: ", n_lw_, ")");
    tmv::log_info("  SW optical depth: ", tau_sw_ref_, " (exponent: ", n_sw_, ")");
    tmv::log_info("  Optical model: ", optical_model_);
    if (optical_model_ == absorbing_water_cloud_model)
    {
        tmv::log_info("  Algorithm: ",
                      simple_grey_absorbing_water_cloud_algorithm_id());
        tmv::log_info(
            "  LW mass absorption [m2/kg] vapor/liquid/ice: ",
            absorption_coefficients_.longwave_vapor_m2kg, "/",
            absorption_coefficients_.longwave_liquid_m2kg, "/",
            absorption_coefficients_.longwave_ice_m2kg);
        tmv::log_info(
            "  SW mass absorption [m2/kg] vapor/liquid/ice: ",
            absorption_coefficients_.shortwave_vapor_m2kg, "/",
            absorption_coefficients_.shortwave_liquid_m2kg, "/",
            absorption_coefficients_.shortwave_ice_m2kg);
    }
}

void SimpleGreyScheme::compute_column(
    const RadiationConfig& cfg,
    const RadiationColumnStateView& col,
    RadiationColumnTendencies& tend,
    RadiationColumnFluxes* fluxes_opt
) 
{
    validate_column(cfg,col);
    const size_t nz=col.rho->size();

    tend.dTdt_rad.resize(nz);
    tend.dTdt_lw.resize(nz);
    tend.dTdt_sw.resize(nz);
    std::fill(tend.dTdt_rad.begin(), tend.dTdt_rad.end(), 0.0);
    std::fill(tend.dTdt_lw.begin(), tend.dTdt_lw.end(), 0.0);
    std::fill(tend.dTdt_sw.begin(), tend.dTdt_sw.end(), 0.0);
    tend.atmospheric_power_lw_wm2 = 0.0;
    tend.atmospheric_power_sw_wm2 = 0.0;
    tend.atmospheric_power_total_wm2 = 0.0;

    if (fluxes_opt) 
    {
        const auto clear_flux = [nz](std::vector<double>& flux)
        {
            flux.resize(nz + 1);
            std::fill(flux.begin(), flux.end(), 0.0);
        };
        clear_flux(fluxes_opt->Fup_lw);
        clear_flux(fluxes_opt->Fdn_lw);
        clear_flux(fluxes_opt->Fup_sw);
        clear_flux(fluxes_opt->Fdn_sw);
        clear_flux(fluxes_opt->Fnet_lw);
        clear_flux(fluxes_opt->Fnet_sw);
        clear_flux(fluxes_opt->Fnet_total);
    }

    RadiationColumnWorkspace local_workspace;
    RadiationColumnWorkspace& workspace =
        col.workspace ? *col.workspace : local_workspace;

    if (cfg.do_lw)
    {
        tend.atmospheric_power_lw_wm2 =
            compute_lw_heating(
                cfg, col, tend.dTdt_lw, fluxes_opt, workspace);
    }

    if (cfg.do_sw)
    {
        tend.atmospheric_power_sw_wm2 =
            compute_sw_heating(
                cfg, col, tend.dTdt_sw, fluxes_opt, workspace);
    }

    tend.atmospheric_power_total_wm2 =
        tend.atmospheric_power_lw_wm2 + tend.atmospheric_power_sw_wm2;

    for (size_t k = 0; k < nz; ++k) 
    {
        tend.dTdt_rad[k] = tend.dTdt_lw[k] + tend.dTdt_sw[k];
    }

    if (fluxes_opt) 
    {
        for (size_t k = 0; k <= nz; ++k) 
        {
            fluxes_opt->Fnet_lw[k] = fluxes_opt->Fup_lw[k] - fluxes_opt->Fdn_lw[k];
            fluxes_opt->Fnet_sw[k] = fluxes_opt->Fup_sw[k] - fluxes_opt->Fdn_sw[k];
            fluxes_opt->Fnet_total[k] = fluxes_opt->Fnet_lw[k] + fluxes_opt->Fnet_sw[k];
        }
    }
}

double SimpleGreyScheme::compute_lw_heating(
    const RadiationConfig& cfg,
    const RadiationColumnStateView& col,
    std::vector<double>& dTdt_lw,
    RadiationColumnFluxes* fluxes_opt,
    RadiationColumnWorkspace& workspace
) 
{
    const size_t nz = col.rho->size();

    radiative_transfer::compute_optical_depth_profile(
        *col.p_int, tau_lw_ref_, n_lw_, workspace.tau_profile);

    radiative_transfer::compute_layer_optical_depths(
        workspace.tau_profile, *col.dz, workspace.tau_layers);

    // Keep the qualified clear-sky recurrence free of additional arithmetic;
    // its historical longwave lane has a bitwise restart contract.
    if (!cfg.clear_sky_only)
        add_water_path_optical_depth(
            col, absorption_coefficients_, true,
            workspace.tau_layers);

    radiative_transfer::grey_lw_two_stream(
        *col.T, workspace.tau_layers, col.Tsfc, col.emissivity_lw,
        workspace.flux_up, workspace.flux_down
    );

    workspace.flux_net.resize(nz + 1);

    for (size_t k = 0; k <= nz; ++k) 
    {
        workspace.flux_net[k] =
            workspace.flux_up[k] - workspace.flux_down[k];
    }

    
    radiative_transfer::heating_rate_from_flux_divergence(
        *col.rho, *col.dz, workspace.flux_net, dTdt_lw
    );

    if (fluxes_opt) 
    {
        fluxes_opt->Fup_lw = workspace.flux_up;
        fluxes_opt->Fdn_lw = workspace.flux_down;
    }
    return -(workspace.flux_net.back() - workspace.flux_net.front());
}

double SimpleGreyScheme::compute_sw_heating(
    const RadiationConfig& cfg,
    const RadiationColumnStateView& col,
    std::vector<double>& dTdt_sw,
    RadiationColumnFluxes* fluxes_opt,
    RadiationColumnWorkspace& workspace
) 
{
    const size_t nz = col.rho->size();

    if (col.mu0 <= 0.0) 
    {
        std::fill(dTdt_sw.begin(), dTdt_sw.end(), 0.0);
        return 0.0;
    }

    compute_simple_grey_shortwave_layer_optical_depths(
        cfg, *col.p_int, *col.dz, *col.rho,
        col.qv, col.qc, col.qi,
        workspace.tau_profile, workspace.tau_layers);

    radiative_transfer::grey_sw_beer_lambert(
        workspace.tau_layers, col.mu0, col.S0, col.albedo_sw,
        workspace.flux_up, workspace.flux_down
    );

    workspace.flux_net.resize(nz + 1);

    for (size_t k = 0; k <= nz; ++k) 
    {
        workspace.flux_net[k] =
            workspace.flux_up[k] - workspace.flux_down[k];
    }

    radiative_transfer::heating_rate_from_flux_divergence(
        *col.rho, *col.dz, workspace.flux_net, dTdt_sw
    );

    if (fluxes_opt)
    {
        fluxes_opt->Fup_sw = workspace.flux_up;
        fluxes_opt->Fdn_sw = workspace.flux_down;
    }
    return -(workspace.flux_net.back() - workspace.flux_net.front());
}

const char* simple_grey_absorbing_water_cloud_algorithm_id() noexcept
{
    return "simple_grey_absorbing_water_cloud_path_v1";
}

const SimpleGreyAbsorptionCoefficients&
simple_grey_absorbing_water_cloud_coefficients() noexcept
{
    static constexpr SimpleGreyAbsorptionCoefficients coefficients{
        0.02, 85.0, 45.0,
        0.001, 20.0, 12.0,
    };
    return coefficients;
}

void compute_simple_grey_shortwave_layer_optical_depths(
    const RadiationConfig& config,
    const std::vector<double>& pressure_interface_pa,
    const std::vector<double>& layer_thickness_m,
    const std::vector<double>& density_kgm3,
    const std::vector<double>* vapor_mixing_ratio,
    const std::vector<double>* cloud_liquid_mixing_ratio,
    const std::vector<double>* cloud_ice_mixing_ratio,
    std::vector<double>& optical_depth_profile,
    std::vector<double>& layer_optical_depth)
{
    radiative_transfer::compute_optical_depth_profile(
        pressure_interface_pa, config.tau_sw_ref, config.n_sw,
        optical_depth_profile);
    radiative_transfer::compute_layer_optical_depths(
        optical_depth_profile, layer_thickness_m, layer_optical_depth);
    if (config.clear_sky_only)
        return;
    RadiationColumnStateView column;
    column.rho = &density_kgm3;
    column.dz = &layer_thickness_m;
    column.qv = vapor_mixing_ratio;
    column.qc = cloud_liquid_mixing_ratio;
    column.qi = cloud_ice_mixing_ratio;
    const std::size_t layers = layer_thickness_m.size();
    if (density_kgm3.size() != layers || vapor_mixing_ratio == nullptr
        || cloud_liquid_mixing_ratio == nullptr
        || cloud_ice_mixing_ratio == nullptr
        || vapor_mixing_ratio->size() != layers
        || cloud_liquid_mixing_ratio->size() != layers
        || cloud_ice_mixing_ratio->size() != layers)
    {
        throw std::invalid_argument(
            "simple_grey terrain shortwave requires complete water-path columns");
    }
    add_water_path_optical_depth(
        column, absorption_coefficients(config), false,
        layer_optical_depth);
}

}  // namespace tmv

namespace tmv
{
void SimpleGreyScheme::validate_column(const RadiationConfig& cfg,const RadiationColumnStateView& col) const
{
    validate_simple_grey_config(cfg, false);
    if (cfg.tau_lw_ref != tau_lw_ref_ || cfg.tau_sw_ref != tau_sw_ref_ ||
        cfg.n_lw != n_lw_ || cfg.n_sw != n_sw_ ||
        cfg.simple_grey_optical_model != optical_model_ ||
        !same_coefficients(absorption_coefficients(cfg),
                           absorption_coefficients_))
    {
        throw std::invalid_argument(
            "simple_grey compute configuration differs from initialized optical parameters");
    }
    if (col.T == nullptr || col.p == nullptr || col.p_int == nullptr ||
        col.rho == nullptr || col.dz == nullptr || col.z_int == nullptr)
    {
        throw std::invalid_argument(
            "simple_grey: T, p, p_int, rho, dz, and z_int column fields are required");
    }
    const size_t nz = col.rho->size();
    if (col.T->size() != nz || col.p->size() != nz ||
        col.p_int->size() != nz + 1 || col.dz->size() != nz ||
        col.z_int->size() != nz + 1 || nz == 0)
    {
        throw std::invalid_argument(
            "simple_grey: required column fields must have identical sizes");
    }
    const bool water_path_model =
        optical_model_ == absorbing_water_cloud_model;
    if (water_path_model &&
        (col.qv == nullptr || col.qc == nullptr || col.qi == nullptr ||
         col.qv->size() != nz || col.qc->size() != nz ||
         col.qi->size() != nz))
        throw std::invalid_argument(
            "simple_grey absorbing water/cloud-path optics require qv, qc, and qi");
    if ((cfg.do_lw && (!std::isfinite(col.Tsfc) || col.Tsfc <= 0.0 ||
                       !std::isfinite(col.emissivity_lw) ||
                       col.emissivity_lw < 0.0 || col.emissivity_lw > 1.0)) ||
        (cfg.do_sw && (!std::isfinite(col.mu0) || col.mu0 < 0.0 || col.mu0 > 1.0 ||
                       !std::isfinite(col.S0) || col.S0 < 0.0 ||
                       !std::isfinite(col.albedo_sw) ||
                       col.albedo_sw < 0.0 || col.albedo_sw > 1.0)))
    {
        throw std::invalid_argument(
            "simple_grey: invalid surface or solar boundary condition");
    }
    for (std::size_t k = 0; k < nz; ++k)
    {
        const double thickness = (*col.z_int)[k + 1] - (*col.z_int)[k];
        const double tolerance = 1.0e-10 * std::max(1.0, std::abs(thickness));
        if (!std::isfinite((*col.T)[k]) || (*col.T)[k] <= 0.0 ||
            !std::isfinite((*col.p)[k]) || (*col.p)[k] <= 0.0 ||
            !std::isfinite((*col.rho)[k]) || (*col.rho)[k] <= 0.0 ||
            !std::isfinite((*col.dz)[k]) || (*col.dz)[k] <= 0.0 ||
            !std::isfinite((*col.z_int)[k]) ||
            !std::isfinite((*col.z_int)[k + 1]) || thickness <= 0.0 ||
            std::abs((*col.dz)[k] - thickness) > tolerance ||
            !std::isfinite((*col.p_int)[k]) || (*col.p_int)[k] <= 0.0 ||
            !std::isfinite((*col.p_int)[k + 1]) || (*col.p_int)[k + 1] < 0.0 ||
            (*col.p_int)[k + 1] > (*col.p_int)[k] ||
            (*col.p)[k] > (*col.p_int)[k] ||
            (*col.p)[k] < (*col.p_int)[k + 1])
        {
            throw std::invalid_argument(
                "simple_grey: column state/interface geometry violates its physical contract");
        }
        if (water_path_model)
        {
            const double qv = (*col.qv)[k];
            const double qc = (*col.qc)[k];
            const double qi = (*col.qi)[k];
            if (!std::isfinite(qv) || !std::isfinite(qc) ||
                !std::isfinite(qi) || qv < 0.0 || qc < 0.0 || qi < 0.0 ||
                qv + qc + qi > 1.0)
                throw std::invalid_argument(
                    "simple_grey water species must be finite non-negative mixing ratios");
        }
    }

}
}
