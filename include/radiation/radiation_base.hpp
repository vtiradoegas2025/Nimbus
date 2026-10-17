#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/field/field3d.hpp"
#include "core/infra/physical_constants.hpp"

// Column-scheme contracts and persistent radiation runtime state.

class RadiationComputeKernels;

namespace tmv
{

struct TerrainShortwaveRuntimeWorkspace;
struct RadiationBatchRuntimeWorkspace;

namespace radiation_constants
{
inline constexpr double sigma = physical_constants::stefan_boltzmann_wm2k4;
inline constexpr double S0    = physical_constants::solar_constant_wm2;
inline constexpr double pi    = physical_constants::pi;
} // namespace radiation_constants

struct RadiationConfig
{
    // Omission must mean no radiative forcing. Active radiation requires an
    // explicit, validated selection and forcing contract.
    std::string scheme_id          = "none";
    bool        do_lw              = true;
    bool        do_sw              = true;
    double      dt_radiation       = 300.0;
    bool        clear_sky_only     = true;
    bool        use_subcycling     = false;
    std::size_t subcycle_count     = 1;
    bool        enable_diagnostics = false;

    std::string simple_grey_optical_model = "pressure_clear_sky_v1";
    double simple_grey_lw_vapor_absorption_m2kg  = 0.02;
    double simple_grey_lw_liquid_absorption_m2kg = 85.0;
    double simple_grey_lw_ice_absorption_m2kg    = 45.0;
    double simple_grey_sw_vapor_absorption_m2kg  = 0.001;
    double simple_grey_sw_liquid_absorption_m2kg = 20.0;
    double simple_grey_sw_ice_absorption_m2kg    = 12.0;

    std::string solar_forcing               = "idealized_diurnal";
    double      latitude_deg                = 35.0;
    double      declination_deg             = 15.0;
    double      local_solar_time_at_start_h = 0.0;

    double tau_lw_ref = 6.0;
    double tau_sw_ref = 0.22;
    double n_lw       = 4.0;
    double n_sw       = 2.0;

    // RRTMG gas composition is explicit. Values are dry-air mole fractions;
    // water vapor is diagnosed from the model mixing-ratio profile. Ozone is
    // deliberately not supplied by a hidden climatology.
    double rrtmg_ozone_vmr                  = 0.0;
    double rrtmg_co2_vmr                    = 420.0e-6;
    double rrtmg_ch4_vmr                    = 1.9e-6;
    double rrtmg_n2o_vmr                    = 0.335e-6;
    double rrtmg_co_vmr                     = 0.0;
    double rrtmg_o2_vmr                     = 0.2095;
    double rrtmg_n2_vmr                     = 0.7808;
    double rrtmg_cfc11_vmr                  = 0.0;
    double rrtmg_cfc12_vmr                  = 0.0;
    double rrtmg_cfc22_vmr                  = 0.0;
    double rrtmg_ccl4_vmr                   = 0.0;
    double rrtmg_liquid_effective_radius_um = 10.0;
    double rrtmg_ice_effective_radius_um    = 30.0;

    // Mapped-terrain shortwave discrete-ordinates and residual contract.
    // These values do not affect the established flat-column solver.
    std::size_t terrain_sw_cosine_nodes                  = 4;
    std::size_t terrain_sw_azimuth_nodes                 = 8;
    std::size_t terrain_sw_maximum_bounces               = 32;
    double      terrain_sw_relative_residual_tolerance   = 1.0e-10;
    double      terrain_sw_absolute_residual_tolerance_w = 1.0e-9;
};

/** Earth-local direction from the surface toward the idealized sun. */
struct IdealizedSolarGeometry
{
    double east  = 0.0;
    double north = 0.0;
    double up    = -1.0;

    [[nodiscard]] double cosine_zenith() const noexcept
    {
        return std::clamp(up, 0.0, 1.0);
    }
};

inline IdealizedSolarGeometry idealized_solar_geometry(const RadiationConfig& config, double simulation_time_s)
{
    if (config.solar_forcing != "idealized_diurnal")
        throw std::invalid_argument("unsupported radiation solar forcing mode");
    if (!std::isfinite(config.latitude_deg) || config.latitude_deg < -90.0 || config.latitude_deg > 90.0 ||
        !std::isfinite(config.declination_deg) || config.declination_deg < -23.5 || config.declination_deg > 23.5 ||
        !std::isfinite(config.local_solar_time_at_start_h) || config.local_solar_time_at_start_h < 0.0 ||
        config.local_solar_time_at_start_h >= 24.0 || !std::isfinite(simulation_time_s) || simulation_time_s < 0.0)
    {
        throw std::invalid_argument("invalid idealized radiation solar geometry");
    }
    const double latitude    = config.latitude_deg * radiation_constants::pi / 180.0;
    const double declination = config.declination_deg * radiation_constants::pi / 180.0;
    double       local_hour  = std::fmod(config.local_solar_time_at_start_h + simulation_time_s / 3600.0, 24.0);
    if (local_hour < 0.0)
        local_hour += 24.0;
    const double hour_angle      = (local_hour - 12.0) * radiation_constants::pi / 12.0;
    const double cos_declination = std::cos(declination);
    return {
        -cos_declination * std::sin(hour_angle),
        std::cos(latitude) * std::sin(declination) - std::sin(latitude) * cos_declination * std::cos(hour_angle),
        std::sin(latitude) * std::sin(declination) + std::cos(latitude) * cos_declination * std::cos(hour_angle),
    };
}

// Reusable per-worker column memory. The runtime owns one instance per active
// CPU worker so a radiation cadence does not allocate for every horizontal
// column. Direct scheme callers may omit it and retain the convenient
// allocate-on-call interface.
struct RadiationColumnWorkspace
{
    // Runtime column reconstruction.
    std::vector<double> p;
    std::vector<double> p_interface;
    std::vector<double> T;
    std::vector<double> rho;
    std::vector<double> qv;
    std::vector<double> qc;
    std::vector<double> qi;
    std::vector<double> z_center;
    std::vector<double> z_interface;
    std::vector<double> layer_thickness;
    std::vector<float>  theta_tendency;
    std::vector<float>  cell_power_w;

    // Radiative-transfer work arrays. Longwave and shortwave execute
    // sequentially within a column and intentionally share these buffers.
    std::vector<double> tau_profile;
    std::vector<double> tau_layers;
    std::vector<double> flux_up;
    std::vector<double> flux_down;
    std::vector<double> flux_net;
};

struct RadiationFluxDiagnostics
{
    std::size_t columns_x = 0;
    std::size_t columns_y = 0;
    std::size_t levels    = 0;

    std::vector<double> upward_longwave_wm2;
    std::vector<double> downward_longwave_wm2;
    std::vector<double> upward_shortwave_wm2;
    std::vector<double> downward_shortwave_wm2;

    [[nodiscard]] bool empty() const noexcept
    {
        return upward_longwave_wm2.empty();
    }
};

inline double idealized_solar_cosine_zenith(const RadiationConfig& config, double simulation_time_s)
{
    return idealized_solar_geometry(config, simulation_time_s).cosine_zenith();
}

enum class RadiationRequirements
{
    TEMPERATURE        = 1,
    THETA_PRESSURE     = 2,
    SOLAR_GEOMETRY     = 4,
    CLOUD_OPTICS       = 8,
    AEROSOL_OPTICS     = 16,
    INTERFACE_GEOMETRY = 32
};

struct RadiationColumnStateView
{
    const std::vector<double>* T     = nullptr;
    const std::vector<double>* theta = nullptr;
    const std::vector<double>* p     = nullptr;
    // Pressure at the same nz+1 physical interfaces as z_int. Simple-grey
    // optical depth is evaluated on these interfaces; scalar-center pressure
    // is never reinterpreted as a surface boundary or extended to p=0 aloft.
    const std::vector<double>* p_int = nullptr;
    // Optional temperature at the same nz+1 interfaces. Schemes that need
    // interface emission derive it from T when this profile is absent.
    const std::vector<double>* T_int = nullptr;
    const std::vector<double>* rho   = nullptr;
    const std::vector<double>* dz    = nullptr;
    const std::vector<double>* z_int = nullptr;

    double mu0 = 0.0;
    double S0  = 1366.0;

    double Tsfc          = 288.0;
    double albedo_sw     = 0.2;
    double emissivity_lw = 0.95;

    const std::vector<double>* qv = nullptr;
    const std::vector<double>* qc = nullptr;
    const std::vector<double>* qi = nullptr;

    // Optional dry-gas profiles for reference atmospheres and coupled
    // chemistry. Values are layer volume mixing ratios. A missing profile
    // uses the corresponding scalar in RadiationConfig.
    const std::vector<double>* nitrogen_vmr        = nullptr;
    const std::vector<double>* ccl4_vmr            = nullptr;
    const std::vector<double>* cfc11_vmr           = nullptr;
    const std::vector<double>* cfc12_vmr           = nullptr;
    const std::vector<double>* cfc22_vmr           = nullptr;
    const std::vector<double>* carbon_dioxide_vmr  = nullptr;
    const std::vector<double>* ozone_vmr           = nullptr;
    const std::vector<double>* nitrous_oxide_vmr   = nullptr;
    const std::vector<double>* carbon_monoxide_vmr = nullptr;
    const std::vector<double>* methane_vmr         = nullptr;
    const std::vector<double>* oxygen_vmr          = nullptr;

    RadiationColumnWorkspace* workspace = nullptr;
};

struct RadiationColumnTendencies
{
    std::vector<double> dTdt_rad;
    std::vector<double> dTdt_lw;
    std::vector<double> dTdt_sw;

    // Signed atmospheric column power implied by the interface fluxes.
    // Positive values heat the resolved atmosphere. Keeping these compact
    // scalars alongside the tendency closes the audit chain without retaining
    // every interface flux for every horizontal column.
    double atmospheric_power_lw_wm2    = 0.0;
    double atmospheric_power_sw_wm2    = 0.0;
    double atmospheric_power_total_wm2 = 0.0;
};

struct RadiationColumnFluxes
{
    std::vector<double> Fup_lw;
    std::vector<double> Fdn_lw;
    std::vector<double> Fup_sw;
    std::vector<double> Fdn_sw;

    std::vector<double> Fnet_lw;
    std::vector<double> Fnet_sw;
    std::vector<double> Fnet_total;
};

/** Layer-major spectral state consumed by mapped shortwave transport. */
struct MappedShortwaveColumnOptics
{
    std::size_t layers = 0;
    std::size_t g_points = 0;
    std::vector<double> absorption_optical_depth;
    std::vector<double> scattering_optical_depth;
    std::vector<double> scattering_asymmetry;
    std::vector<double> solar_source_fraction;
};

class RadiationSchemeBase
{
public:
    virtual ~RadiationSchemeBase() = default;
    virtual std::unique_ptr<RadiationSchemeBase> clone() const
    {
        return nullptr;
    }

    virtual std::string name() const = 0;

    // Bitmask of RadiationRequirements needed by compute_column().
    virtual int required_fields() const = 0;

    virtual void initialize(const RadiationConfig& cfg) = 0;

    // Computes column heating and, when requested, interface flux diagnostics.
    virtual void compute_column(const RadiationConfig&          cfg,
                                const RadiationColumnStateView& col,
                                RadiationColumnTendencies&      tend,
                                RadiationColumnFluxes*          fluxes_opt = nullptr) = 0;

    // Optional spectral state for the native mapped-terrain shortwave owner.
    // Implementations publish only after the complete candidate validates.
    virtual bool compute_mapped_shortwave_optics(
        const RadiationConfig& /*cfg*/,
        const RadiationColumnStateView& /*column*/,
        MappedShortwaveColumnOptics& /*optics*/)
    {
        return false;
    }

    // Optional whole-batch accelerator contract. Implementations must leave
    // every tendency unchanged when returning false or throwing. Runtime
    // orchestration owns bounded chunking and calls this only from the serial
    // command-submission thread after parallel column preparation.
    virtual bool supports_native_column_batch(
        const RadiationConfig& /*cfg*/,
        const RadiationComputeKernels& /*kernels*/) const
    {
        return false;
    }
    virtual bool compute_columns_native_diagnosed(const RadiationConfig& config,
        const RadiationColumnStateView* columns,std::size_t count,RadiationColumnTendencies* tendencies,
        RadiationComputeKernels& kernels,RadiationColumnFluxes* fluxes)
    {
        return fluxes==nullptr && compute_columns_native(config,columns,count,tendencies,kernels);
    }
    virtual bool compute_columns_native(
        const RadiationConfig& /*cfg*/,
        const RadiationColumnStateView* /*columns*/,
        std::size_t /*column_count*/,
        RadiationColumnTendencies* /*tendencies*/,
        RadiationComputeKernels& /*kernels*/)
    {
        return false;
    }
};

struct RadiationWorkerContext
{
    std::unique_ptr<RadiationSchemeBase> scheme;
    RadiationColumnWorkspace             workspace;
    RadiationColumnTendencies            tendencies;
    RadiationColumnFluxes                 fluxes;
    MappedShortwaveColumnOptics           mapped_shortwave_optics;
};

// Bind through SimulationStateView::radiation_runtime for independent runs.
struct RadiationRuntimeState
{
    std::unique_ptr<RadiationSchemeBase> scheme;
    RadiationConfig                      config;
    double                               last_update_time_s = std::numeric_limits<double>::lowest();
    double                               next_update_time_s = 0.0;

    // Sample-and-hold energy ledger. The held power is replaced
    // transactionally with the tendency at a radiation update; the remaining
    // fields advance only after the shared theta application commits.
    // Per-cell power, rather than a theta tendency alone, is the conserved
    // coupling currency: density and Exner pressure may evolve between the
    // radiation sample and the later thermodynamic application.
    Field3D held_cell_power_w;

    // Transactional publication scratch. These fields are deliberately
    // omitted from copy construction and checkpoint serialization: a runtime
    // snapshot contains only committed science state. After warm-up, commit
    // swaps the old published buffers back here so radiation cadence updates
    // do not allocate full model volumes.
    Field3D                             scratch_theta_tendency;
    Field3D                             scratch_cell_power_w;
    std::vector<double>                 scratch_column_power_w;
    std::vector<unsigned char>          scratch_column_errors;
    RadiationFluxDiagnostics            scratch_flux_diagnostics;
    std::vector<RadiationWorkerContext> worker_contexts;
    // Complete mapped-terrain shortwave scratch is intentionally absent from
    // snapshots/checkpoints. shared_ptr permits this public state header to
    // retain only an incomplete implementation type.
    std::shared_ptr<TerrainShortwaveRuntimeWorkspace> terrain_shortwave_workspace;
    // Bounded native-column batch storage is likewise operational scratch and
    // is intentionally omitted from snapshots/checkpoints.
    std::shared_ptr<RadiationBatchRuntimeWorkspace> batch_workspace;

    double        held_atmospheric_power_w                    = 0.0;
    RadiationFluxDiagnostics held_flux_diagnostics;
    double        held_shortwave_incoming_power_w             = 0.0;
    double        held_shortwave_atmospheric_absorbed_power_w = 0.0;
    double        held_surface_absorbed_power_w               = 0.0;
    double        held_top_escape_power_w                     = 0.0;
    double        held_lateral_escape_power_w                 = 0.0;
    double        held_unresolved_reflected_power_w           = 0.0;
    bool          held_reflection_converged                   = true;
    double        last_expected_atmospheric_energy_j          = 0.0;
    double        last_applied_radiation_heat_j               = 0.0;
    double        accumulated_expected_atmospheric_energy_j   = 0.0;
    double        accumulated_applied_radiation_heat_j        = 0.0;
    std::uint64_t applied_step_count                          = 0;

    RadiationRuntimeState() = default;
    RadiationRuntimeState(const RadiationRuntimeState& other)
        : config(other.config), last_update_time_s(other.last_update_time_s),
          next_update_time_s(other.next_update_time_s), held_cell_power_w(other.held_cell_power_w),
          held_atmospheric_power_w(other.held_atmospheric_power_w),
          held_flux_diagnostics(other.held_flux_diagnostics),
          held_shortwave_incoming_power_w(other.held_shortwave_incoming_power_w),
          held_shortwave_atmospheric_absorbed_power_w(other.held_shortwave_atmospheric_absorbed_power_w),
          held_surface_absorbed_power_w(other.held_surface_absorbed_power_w),
          held_top_escape_power_w(other.held_top_escape_power_w),
          held_lateral_escape_power_w(other.held_lateral_escape_power_w),
          held_unresolved_reflected_power_w(other.held_unresolved_reflected_power_w),
          held_reflection_converged(other.held_reflection_converged),
          last_expected_atmospheric_energy_j(other.last_expected_atmospheric_energy_j),
          last_applied_radiation_heat_j(other.last_applied_radiation_heat_j),
          accumulated_expected_atmospheric_energy_j(other.accumulated_expected_atmospheric_energy_j),
          accumulated_applied_radiation_heat_j(other.accumulated_applied_radiation_heat_j),
          applied_step_count(other.applied_step_count)
    {
        if (other.scheme)
        {
            scheme = other.scheme->clone();
            if (!scheme)
            {
                throw std::runtime_error("radiation scheme does not support runtime snapshots: " +
                                         other.scheme->name());
            }
        }
    }
    RadiationRuntimeState& operator=(const RadiationRuntimeState& other)
    {
        if (this != &other)
        {
            RadiationRuntimeState copy(other);
            *this = std::move(copy);
        }
        return *this;
    }
    RadiationRuntimeState(RadiationRuntimeState&&) noexcept            = default;
    RadiationRuntimeState& operator=(RadiationRuntimeState&&) noexcept = default;
};

using RadiationSchemeFactory = std::unique_ptr<RadiationSchemeBase> (*)(const RadiationConfig&);

} // namespace tmv

#include "radiation/factory.hpp"
#include "radiation/runtime.hpp"

namespace radiation_constants = tmv::radiation_constants;
using tmv::canonicalize_radiation_scheme_id;
using tmv::create_radiation_scheme;
using tmv::get_available_radiation_schemes;
using tmv::get_known_radiation_schemes;
using tmv::idealized_solar_cosine_zenith;
using tmv::idealized_solar_geometry;
using tmv::IdealizedSolarGeometry;
using tmv::is_radiation_scheme_implemented;
using tmv::RadiationColumnFluxes;
using tmv::RadiationColumnStateView;
using tmv::RadiationColumnTendencies;
using tmv::RadiationColumnWorkspace;
using tmv::RadiationConfig;
using tmv::RadiationRequirements;
using tmv::RadiationRuntimeState;
using tmv::RadiationSchemeBase;
using tmv::RadiationSchemeFactory;
using tmv::RadiationWorkerContext;
