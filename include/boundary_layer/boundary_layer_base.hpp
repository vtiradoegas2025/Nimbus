#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/infra/physical_constants.hpp"

/**
 * @file boundary_layer_base.hpp
 * @brief Base interfaces for planetary boundary-layer parameterizations.
 *
 * Declares configuration/state contracts and tendency outputs used by
 * slab, YSU, MYNN, and related boundary-layer schemes.
 * Includes factory and global initialization entry points.
 */

namespace boundary_layer_constants
{
inline constexpr double kappa = physical_constants::von_karman_constant;
inline constexpr double g = physical_constants::gravity_ms2;
inline constexpr double cp = physical_constants::specific_heat_cp_jkgk;
inline constexpr double T0 = physical_constants::freezing_temperature_k;
inline constexpr double pi = physical_constants::pi;
} // namespace boundary_layer_constants

struct BoundaryLayerConfig
{
    std::string scheme_id = "slab";
    double dt_pbl = 60.0;
    bool apply_surface_fluxes = true;
    std::string surface_layer_id = "bulk";
    bool clear_sky_only = true;
    bool enable_nonlocal_transport = false;
    bool enable_tke = false;
    std::string pbl_top_method = "ri";
    double pbl_max_height = 2000.0;
    double min_ustar = 1e-4;
};

struct SurfaceConfig
{
    double z0m = 0.1;
    double z0h = 0.01;
    double albedo = 0.2;
    double emissivity = 0.95;
    double Tsfc = 288.0;
    double qsfc = 0.01;
};

enum class BoundaryLayerRequirements
{
    BASIC = 1,
    MOISTURE = 2,
    TKE = 4,
    CLOUDS = 8
};

struct BoundaryLayerColumnStateView
{
    const std::vector<double>* u = nullptr;
    const std::vector<double>* v = nullptr;
    const std::vector<double>* theta = nullptr;
    const std::vector<double>* qv = nullptr;
    const std::vector<double>* p = nullptr;
    const std::vector<double>* rho = nullptr;
    const std::vector<double>* z_int = nullptr;

    const std::vector<double>* tke = nullptr;
    const std::vector<double>* qc = nullptr;
    const std::vector<double>* qi = nullptr;

    double u_sfc = 0.0;
    double v_sfc = 0.0;
    double theta_sfc = 288.0;
    double qv_sfc = 0.01;
    double z_sfc = 10.0;
};

struct BoundaryLayerTendencies
{
    std::vector<double> dudt_pbl;
    std::vector<double> dvdt_pbl;
    std::vector<double> dthetadt_pbl;
    std::vector<double> dqvdt_pbl;
    std::vector<double> dtkedt_pbl;
};

struct BoundaryLayerDiagnostics
{
    std::vector<double> K_m;
    std::vector<double> K_h;
    std::vector<double> K_e;

    double pbl_height = 0.0;
    double ustar = 0.0;

    double tau_u = 0.0;
    double tau_v = 0.0;
    double H = 0.0;
    double E = 0.0;

    double L = 0.0;
    double rib_bulk = 0.0;

    double Cd = 0.0;
    double Ch = 0.0;
    double Ce = 0.0;
};

class BoundaryLayerSchemeBase
{
public:
    /**
     * @brief Virtual destructor for polymorphic cleanup.
     */
    virtual ~BoundaryLayerSchemeBase() = default;

    /**
     * @brief Returns the scheme identifier.
     * @return Scheme name.
     */
    virtual std::string name() const = 0;

    /**
     * @brief Returns required input fields for this scheme.
     * @return Bitmask of BoundaryLayerRequirements.
     */
    virtual int required_fields() const = 0;

    /**
     * @brief Initializes scheme state and coefficients.
     * @param cfg Boundary-layer configuration.
     */
    virtual void initialize(const BoundaryLayerConfig& cfg) = 0;

    /**
     * @brief Computes tendencies for one vertical column.
     * @param cfg Boundary-layer configuration.
     * @param sfc Surface-layer configuration.
     * @param col Column state view.
     * @param tend Output tendencies.
     * @param diag_opt Optional diagnostics output.
     */
    virtual void compute_column(const BoundaryLayerConfig& cfg,
                                const SurfaceConfig& sfc,
                                const BoundaryLayerColumnStateView& col,
                                BoundaryLayerTendencies& tend,
                                BoundaryLayerDiagnostics* diag_opt = nullptr) = 0;
};

using BoundaryLayerSchemeFactory = std::unique_ptr<BoundaryLayerSchemeBase> (*)(const BoundaryLayerConfig&);

extern std::unique_ptr<BoundaryLayerSchemeBase> boundary_layer_scheme;
extern BoundaryLayerConfig global_boundary_layer_config;
extern SurfaceConfig global_surface_config;

/**
 * @brief Creates a boundary-layer scheme by name.
 * @param scheme_name Scheme identifier.
 * @return Owning pointer to a scheme instance.
 */
std::unique_ptr<BoundaryLayerSchemeBase> create_boundary_layer_scheme(const std::string& scheme_name);

/**
 * @brief Lists registered boundary-layer schemes.
 * @return Collection of scheme identifiers.
 */
std::vector<std::string> get_available_boundary_layer_schemes();

/**
 * @brief Initializes the global boundary-layer subsystem.
 * @param scheme_name Scheme identifier.
 * @param cfg Optional boundary-layer configuration.
 * @param sfc Optional surface configuration.
 */
void initialize_boundary_layer(const std::string& scheme_name,
                               const BoundaryLayerConfig& cfg = BoundaryLayerConfig{},
                               const SurfaceConfig& sfc = SurfaceConfig{});

/**
 * @brief Steps the boundary-layer scheme at the current model time.
 * @param current_time Current simulation time in seconds.
 */
void step_boundary_layer(double current_time);
