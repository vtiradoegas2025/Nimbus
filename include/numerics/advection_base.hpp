#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/field3d.hpp"
#include "numerics/numerics_base.hpp"

/**
 * @file advection_base.hpp
 * @brief Base interfaces and shared state models for advection schemes.
 *
 * Defines runtime configuration and state views used by advection
 * implementations.
 * Exposes factory and initialization APIs for scheme selection.
 */

struct AdvectionConfig
{
    std::string scheme_id = "tvd";
    std::string formulation = "flux_form";
    std::string split_mode = "unsplit";
    std::string limiter_id = "mc";
    bool positivity = false;
    double positivity_dt = 1.0;
    double cfl_target = numerics_constants::cfl_target;
    std::string reconstruct_on = "q";
    int weno_order = 5;
};

struct AdvectionStateView
{
    const Field3D* u = nullptr;
    const Field3D* v = nullptr;
    const Field3D* w = nullptr;
    const Field3D* q = nullptr;
    const Field3D* rho = nullptr;
    const GridMetrics* grid = nullptr;
};

struct AdvectionTendencies
{
    Field3D dqdt_adv;
};

struct AdvectionDiagnostics
{
    double max_cfl_x = 0.0;
    double max_cfl_y = 0.0;
    double max_cfl_z = 0.0;
    double suggested_dt = 0.0;
    Field3D fluxes_x;
    Field3D fluxes_y;
    Field3D fluxes_z;
};

class AdvectionSchemeBase : public NumericalSchemeBase
{
public:
    /**
     * @brief Initializes the advection scheme.
     * @param cfg Advection configuration.
     */
    virtual void initialize(const AdvectionConfig& cfg) = 0;

    /**
     * @brief Computes advection tendencies for the supplied state.
     * @param cfg Advection configuration.
     * @param state Read-only state view.
     * @param tendencies Output tendency container.
     * @param diag_opt Optional diagnostics output.
     */
    virtual void compute_flux_divergence(const AdvectionConfig& cfg,
                                         const AdvectionStateView& state,
                                         AdvectionTendencies& tendencies,
                                         AdvectionDiagnostics* diag_opt = nullptr) = 0;

    /**
     * @brief Suggests a stable advection time step.
     * @param cfg Advection configuration.
     * @param state Read-only state view.
     * @return Suggested advection time step in seconds.
     */
    virtual double suggest_dt(const AdvectionConfig& cfg, const AdvectionStateView& state) = 0;
};

using AdvectionSchemeFactory = std::unique_ptr<AdvectionSchemeBase> (*)(const AdvectionConfig&);

/**
 * @brief Creates an advection scheme by name.
 * @param scheme_name Scheme identifier.
 * @return Owning pointer to a scheme instance.
 */
std::unique_ptr<AdvectionSchemeBase> create_advection_scheme(const std::string& scheme_name);

/**
 * @brief Lists registered advection schemes.
 * @return Collection of scheme identifiers.
 */
std::vector<std::string> get_available_advection_schemes();

/**
 * @brief Initializes the global advection subsystem.
 * @param scheme_name Scheme identifier.
 * @param cfg Optional advection configuration.
 */
void initialize_advection(const std::string& scheme_name, const AdvectionConfig& cfg = AdvectionConfig{});
