#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/field/field3d.hpp"
#include "core/infra/physical_constants.hpp"

/**
 * @file turbulence_base.hpp
 * @brief Base contracts for subgrid turbulence parameterizations.
 *
 * Declares shared configuration/state models and tendency/diagnostic
 * outputs used by Smagorinsky and TKE-based schemes.
 * Runtime code uses this interface to select and step a turbulence model.
 */

struct TerrainMetrics3D;
struct Topography2D;

namespace turbulence_constants
{
inline constexpr double g = physical_constants::gravity_ms2;
inline constexpr double kappa = physical_constants::von_karman_constant;
inline constexpr double C_s_default = 0.18;
inline constexpr double Pr_t_default = 0.7;
inline constexpr double Sc_t_default = 0.7;
} // namespace turbulence_constants

struct TurbulenceConfig
{
    std::string scheme_id = "smagorinsky";
    double dt_sgs = 1.0;
    std::string mode = "3d";
    std::string filter_width = "cubic_root";
    double Cs = turbulence_constants::C_s_default;
    double Pr_t = turbulence_constants::Pr_t_default;
    double Sc_t = turbulence_constants::Sc_t_default;
    bool dynamic_Cs = false;
    std::string stability_correction = "none";
    bool enable_moist_buoyancy = false;
    bool enable_diagnostics = false;
    double nu_t_max = 1000.0;
    double K_max = 1000.0;
};

struct GridMetrics
{
    double dx = 1000.0;
    double dy = 1000.0;
    std::vector<double> dz;
    std::vector<double> z_int;
    bool terrain_metrics_active = false;
    const TerrainMetrics3D* terrain_metrics = nullptr;
    const Topography2D* terrain_topography = nullptr;
};

enum class TurbulenceRequirements
{
    BASIC = 1,
    MOISTURE = 2,
    TKE = 4,
    STABILITY = 8
};

struct TurbulenceStateView
{
    const Field3D* u = nullptr;
    const Field3D* v = nullptr;
    const Field3D* w = nullptr;
    const Field3D* rho = nullptr;
    const Field3D* theta = nullptr;
    const Field3D* qv = nullptr;
    const Field3D* tke = nullptr;

    int NR = 0;
    int NTH = 0;
    int NZ = 0;
};

struct TurbulenceTendencies
{
    Field3D dudt_sgs;
    Field3D dvdt_sgs;
    Field3D dwdt_sgs;
    Field3D dthetadt_sgs;
    Field3D dqvdt_sgs;
    Field3D dtkedt_sgs;
};

struct TurbulenceDiagnostics
{
    Field3D nu_t;
    Field3D K_theta;
    Field3D K_q;
    Field3D K_tke;

    Field3D shear_prod;
    Field3D buoy_prod;
    Field3D dissipation;

    Field3D Cs_eff;
    Field3D Pr_t_eff;
};

class TurbulenceSchemeBase
{
public:
    /**
     * @brief Virtual destructor for polymorphic cleanup.
     */
    virtual ~TurbulenceSchemeBase() = default;

    /**
     * @brief Returns the scheme identifier.
     * @return Scheme name.
     */
    virtual std::string name() const = 0;

    /**
     * @brief Returns required input fields for this scheme.
     * @return Bitmask of TurbulenceRequirements.
     */
    virtual int required_fields() const = 0;

    /**
     * @brief Initializes scheme state and coefficients.
     * @param cfg Turbulence configuration.
     */
    virtual void initialize(const TurbulenceConfig& cfg) = 0;

    /**
     * @brief Computes subgrid tendencies and optional diagnostics.
     * @param cfg Turbulence configuration.
     * @param grid Grid metrics.
     * @param state State view.
     * @param tend Output tendencies.
     * @param diag_opt Optional diagnostics output.
     */
    virtual void compute(const TurbulenceConfig& cfg,
                         const GridMetrics& grid,
                         const TurbulenceStateView& state,
                         TurbulenceTendencies& tend,
                         TurbulenceDiagnostics* diag_opt = nullptr) = 0;
};

using TurbulenceSchemeFactory = std::unique_ptr<TurbulenceSchemeBase> (*)(const TurbulenceConfig&);

extern std::unique_ptr<TurbulenceSchemeBase> turbulence_scheme;
extern TurbulenceConfig global_turbulence_config;

/**
 * @brief Creates a turbulence scheme by name.
 * @param scheme_name Scheme identifier.
 * @return Owning pointer to a scheme instance.
 */
std::unique_ptr<TurbulenceSchemeBase> create_turbulence_scheme(const std::string& scheme_name);

/**
 * @brief Lists registered turbulence schemes.
 * @return Collection of scheme identifiers.
 */
std::vector<std::string> get_available_turbulence_schemes();

/**
 * @brief Initializes the global turbulence subsystem.
 * @param scheme_name Scheme identifier.
 * @param cfg Optional turbulence configuration.
 */
void initialize_turbulence(const std::string& scheme_name, const TurbulenceConfig& cfg = TurbulenceConfig{});

/**
 * @brief Steps turbulence tendencies at current model time.
 * @param current_time Current simulation time in seconds.
 * @param tendencies Output tendency container.
 */
void step_turbulence(double current_time, TurbulenceTendencies& tendencies);
