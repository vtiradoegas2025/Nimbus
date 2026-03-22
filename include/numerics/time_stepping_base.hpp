#pragma once

#include <memory>
#include <string>
#include <vector>

#include "numerics/numerics_base.hpp"

/**
 * @file time_stepping_base.hpp
 * @brief Interfaces and shared data models for time-integration schemes.
 *
 * Declares the configuration, state, diagnostics, and factory APIs
 * used by explicit and split time-stepping implementations.
 * The runtime uses this layer to select and drive a scheme uniformly.
 */

struct TimeSteppingConfig
{
    std::string scheme_id = "rk3";
    bool split_acoustic = false;
    int n_acoustic_substeps = 1;
    std::string physics_splitting = "additive";
    double cfl_safety = numerics_constants::cfl_target;
    bool adaptive_dt = false;
    double dt_min = 1e-6;
    double dt_max = 100.0;
};

struct TimeSteppingState
{
    std::vector<NumericalState> fields;
    double time = 0.0;
    double dt = 1.0;
};

struct TimeSteppingDiagnostics
{
    int n_steps = 0;
    double max_cfl = 0.0;
    double cpu_time_per_step = 0.0;
    std::vector<double> dt_history;
    bool converged = true;
};

class TimeSteppingSchemeBase : public NumericalSchemeBase
{
public:
    /**
     * @brief Initializes the time-stepping scheme.
     * @param cfg Scheme configuration.
     * @param rhs_func Right-hand-side callback for prognostic tendencies.
     */
    virtual void initialize(const TimeSteppingConfig& cfg, RHSFunction rhs_func) = 0;

    /**
     * @brief Advances the model state by one time step.
     * @param cfg Scheme configuration.
     * @param state Mutable state advanced in place.
     * @param diag_opt Optional diagnostics output.
     */
    virtual void step(const TimeSteppingConfig& cfg,
                      TimeSteppingState& state,
                      TimeSteppingDiagnostics* diag_opt = nullptr) = 0;

    /**
     * @brief Suggests a stable time step for current conditions.
     * @param cfg Scheme configuration.
     * @param state Current state snapshot.
     * @return Suggested time step in seconds.
     */
    virtual double suggest_dt(const TimeSteppingConfig& cfg, const TimeSteppingState& state) = 0;
};

using TimeSteppingSchemeFactory = std::unique_ptr<TimeSteppingSchemeBase> (*)(const TimeSteppingConfig&);

/**
 * @brief Creates a time-stepping scheme by name.
 * @param scheme_name Scheme identifier.
 * @return Owning pointer to a scheme instance.
 */
std::unique_ptr<TimeSteppingSchemeBase> create_time_stepping_scheme(const std::string& scheme_name);

/**
 * @brief Lists registered time-stepping schemes.
 * @return Collection of scheme identifiers.
 */
std::vector<std::string> get_available_time_stepping_schemes();

/**
 * @brief Initializes the global time-stepping subsystem.
 * @param scheme_name Scheme identifier.
 * @param cfg Optional scheme configuration.
 * @param rhs_func Optional right-hand-side callback.
 */
void initialize_time_stepping(const std::string& scheme_name,
                              const TimeSteppingConfig& cfg = TimeSteppingConfig{},
                              RHSFunction rhs_func = nullptr);
