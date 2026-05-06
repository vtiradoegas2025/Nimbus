#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/infra/physical_constants.hpp"

/**
 * @file radiation_base.hpp
 * @brief Base contracts for radiation column schemes.
 *
 * Defines configuration, column state views, tendency/flux outputs,
 * and polymorphic interfaces used by longwave/shortwave parameterizations.
 * Runtime code uses these contracts to initialize and call a scheme uniformly.
 */

namespace radiation_constants
{
inline constexpr double sigma = physical_constants::stefan_boltzmann_wm2k4;
inline constexpr double S0 = physical_constants::solar_constant_wm2;
inline constexpr double pi = physical_constants::pi;
} // namespace radiation_constants

struct RadiationConfig
{
    std::string scheme_id = "simple_grey";
    bool do_lw = true;
    bool do_sw = true;
    double dt_radiation = 300.0;
    bool clear_sky_only = true;
    bool use_subcycling = false;
    bool enable_diagnostics = false;

    double tau_lw_ref = 6.0;
    double tau_sw_ref = 0.22;
    double n_lw = 4.0;
    double n_sw = 2.0;
};

enum class RadiationRequirements
{
    TEMPERATURE = 1,
    THETA_PRESSURE = 2,
    SOLAR_GEOMETRY = 4,
    CLOUD_OPTICS = 8,
    AEROSOL_OPTICS = 16
};

struct RadiationColumnStateView
{
    const std::vector<double>* T = nullptr;
    const std::vector<double>* theta = nullptr;
    const std::vector<double>* p = nullptr;
    const std::vector<double>* rho = nullptr;
    const std::vector<double>* dz = nullptr;
    const std::vector<double>* z_int = nullptr;

    double mu0 = 0.0;
    double S0 = 1366.0;

    double Tsfc = 288.0;
    double albedo_sw = 0.2;
    double emissivity_lw = 0.95;

    const std::vector<double>* qv = nullptr;
};

struct RadiationColumnTendencies
{
    std::vector<double> dTdt_rad;
    std::vector<double> dTdt_lw;
    std::vector<double> dTdt_sw;
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

class RadiationSchemeBase
{
public:
    /**
     * @brief Virtual destructor for polymorphic cleanup.
     */
    virtual ~RadiationSchemeBase() = default;

    /**
     * @brief Returns the scheme identifier.
     * @return Scheme name.
     */
    virtual std::string name() const = 0;

    /**
     * @brief Returns required input fields for this scheme.
     * @return Bitmask of RadiationRequirements.
     */
    virtual int required_fields() const = 0;

    /**
     * @brief Initializes scheme state and coefficients.
     * @param cfg Radiation configuration.
     */
    virtual void initialize(const RadiationConfig& cfg) = 0;

    /**
     * @brief Computes radiative tendencies for one atmospheric column.
     * @param cfg Radiation configuration.
     * @param col Column state view.
     * @param tend Output heating tendencies.
     * @param fluxes_opt Optional flux diagnostics output.
     */
    virtual void compute_column(const RadiationConfig& cfg,
                                const RadiationColumnStateView& col,
                                RadiationColumnTendencies& tend,
                                RadiationColumnFluxes* fluxes_opt = nullptr) = 0;
};

using RadiationSchemeFactory = std::unique_ptr<RadiationSchemeBase> (*)(const RadiationConfig&);

extern std::unique_ptr<RadiationSchemeBase> radiation_scheme;
extern RadiationConfig global_radiation_config;

/**
 * @brief Creates a radiation scheme by name.
 * @param scheme_name Scheme identifier.
 * @return Owning pointer to a scheme instance.
 */
std::unique_ptr<RadiationSchemeBase> create_radiation_scheme(const std::string& scheme_name);

/**
 * @brief Lists registered radiation schemes.
 * @return Collection of scheme identifiers.
 */
std::vector<std::string> get_available_radiation_schemes();

/**
 * @brief Lists known radiation scheme identifiers (implemented + planned).
 * @return Collection of canonical scheme identifiers.
 */
std::vector<std::string> get_known_radiation_schemes();

/**
 * @brief Canonicalizes radiation scheme aliases.
 * @param scheme_name Raw scheme id from config/runtime.
 * @return Canonical scheme id.
 */
std::string canonicalize_radiation_scheme_id(const std::string& scheme_name);

/**
 * @brief Reports whether a scheme id is implemented in this build.
 * @param scheme_name Raw or canonical scheme id.
 * @return True when runtime implementation exists.
 */
bool is_radiation_scheme_implemented(const std::string& scheme_name);

/**
 * @brief Initializes the global radiation subsystem.
 * @param scheme_name Scheme identifier.
 * @param cfg Optional radiation configuration.
 */
void initialize_radiation(const std::string& scheme_name, const RadiationConfig& cfg = RadiationConfig{});
