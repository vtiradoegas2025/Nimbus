#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/field/field3d.hpp"
#include "numerics/numerics_base.hpp"

/**
 * @file diffusion_base.hpp
 * @brief Base interfaces and shared data contracts for diffusion schemes.
 *
 * Defines configuration, state views, tendency containers, and diagnostics
 * used by explicit and implicit diffusion implementations.
 * Also provides scheme factory and subsystem initialization APIs.
 */

struct DiffusionConfig
{
    std::string scheme_id = "explicit";
    std::string operator_type = "laplacian";
    std::string apply_to = "scalars";
    double K_h = 0.0;
    double K_v = 0.0;
    std::string implicit_dim = "none";
    double dt_diffusion = 1.0;
    bool use_variable_K = false;
};

struct DiffusionStateView
{
    const Field3D* u = nullptr;
    const Field3D* v = nullptr;
    const Field3D* w = nullptr;
    const Field3D* theta = nullptr;
    const Field3D* qv = nullptr;
    const Field3D* rho = nullptr;
    const Field3D* K_momentum = nullptr;
    const Field3D* K_scalar = nullptr;
    const GridMetrics* grid = nullptr;
};

struct DiffusionTendencies
{
    Field3D dudt_diff;
    Field3D dvdt_diff;
    Field3D dwdt_diff;
    Field3D dthetadt_diff;
    Field3D dqvdt_diff;
};

struct DiffusionDiagnostics
{
    double max_diffusion_number = 0.0;
    Field3D K_effective;
};

class DiffusionSchemeBase : public NumericalSchemeBase
{
public:
    /**
     * @brief Initializes the diffusion scheme.
     * @param cfg Diffusion configuration.
     */
    virtual void initialize(const DiffusionConfig& cfg) = 0;

    /**
     * @brief Computes diffusion tendencies for the supplied state.
     * @param cfg Diffusion configuration.
     * @param state Read-only state view.
     * @param tendencies Output tendency container.
     * @param diag_opt Optional diagnostics output.
     */
    virtual void compute_diffusion_tendencies(const DiffusionConfig& cfg,
                                              const DiffusionStateView& state,
                                              DiffusionTendencies& tendencies,
                                              DiffusionDiagnostics* diag_opt = nullptr) = 0;

    /**
     * @brief Computes a stability metric for the configured diffusion step.
     * @param cfg Diffusion configuration.
     * @param state Read-only state view.
     * @return Stability metric where values above one are typically unstable.
     */
    virtual double check_stability(const DiffusionConfig& cfg, const DiffusionStateView& state) = 0;
};

using DiffusionSchemeFactory = std::unique_ptr<DiffusionSchemeBase> (*)(const DiffusionConfig&);

/**
 * @brief Creates a diffusion scheme by name.
 * @param scheme_name Scheme identifier.
 * @return Owning pointer to a scheme instance.
 */
std::unique_ptr<DiffusionSchemeBase> create_diffusion_scheme(const std::string& scheme_name);

/**
 * @brief Lists registered diffusion schemes.
 * @return Collection of scheme identifiers.
 */
std::vector<std::string> get_available_diffusion_schemes();

/**
 * @brief Initializes the global diffusion subsystem.
 * @param scheme_name Scheme identifier.
 * @param cfg Optional diffusion configuration.
 */
void initialize_diffusion(const std::string& scheme_name, const DiffusionConfig& cfg = DiffusionConfig{});
