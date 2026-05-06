#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/field/field3d.hpp"

/**
 * @file chaos_base.hpp
 * @brief Base interfaces and state containers for stochastic perturbations.
 *
 * Defines configuration and API contracts for chaos schemes that perturb
 * initial conditions and/or physics tendencies.
 * Supports deterministic reproducibility with seeded stochastic updates.
 */

struct GridMetrics;

class SimulationState
{
public:
    Field3D* u = nullptr;
    Field3D* v = nullptr;
    Field3D* w = nullptr;
    Field3D* theta = nullptr;
    Field3D* qv = nullptr;
    Field3D* qc = nullptr;
    Field3D* qr = nullptr;
    Field3D* qi = nullptr;
    Field3D* qs = nullptr;
    Field3D* qg = nullptr;
    Field3D* qh = nullptr;

    /**
     * @brief Checks whether required state pointers are present.
     * @return True when mandatory fields are available for perturbation.
     */
    bool valid() const
    {
        return (u != nullptr) &&
               (v != nullptr) &&
               (w != nullptr) &&
               (theta != nullptr) &&
               (qv != nullptr) &&
               (qc != nullptr) &&
               (qr != nullptr);
    }
};

namespace chaos
{

struct ChaosConfig
{
    std::string scheme_id = "none";

    uint64_t seed = 42;
    int member_id = 0;

    double dt_chaos = 60.0;
    double tau_t = 21600.0;

    double Lx = 50000.0;
    double Ly = 50000.0;
    std::string filter_id = "recursive_gaussian";

    double xi_max = 2.0;
    std::string taper_id = "none";
    double taper_z1 = 1000.0;
    double taper_z2 = 3000.0;

    std::vector<std::string> apply_to_ic;
    std::vector<std::string> apply_to_tendencies;

    std::unordered_map<std::string, double> sigma_ic;
    std::unordered_map<std::string, double> alpha_tend;

    /**
     * @brief Builds default perturbation amplitudes.
     */
    ChaosConfig()
    {
        sigma_ic["u"] = 0.5;
        sigma_ic["v"] = 0.5;
        sigma_ic["w"] = 0.1;
        sigma_ic["theta"] = 0.5;
        sigma_ic["qv"] = 0.0005;
        sigma_ic["qc"] = 0.0;
        sigma_ic["qr"] = 0.0;

        alpha_tend["microphysics"] = 0.3;
        alpha_tend["pbl"] = 0.2;
        alpha_tend["diffusion"] = 0.1;
    }
};

struct ChaosStateView
{
    const GridMetrics* grid = nullptr;

    const Field3D* u = nullptr;
    const Field3D* v = nullptr;
    const Field3D* w = nullptr;
    const Field3D* theta = nullptr;
    const Field3D* qv = nullptr;
    const Field3D* qc = nullptr;
    const Field3D* qr = nullptr;

    double pbl_height = 1000.0;
    const std::vector<std::vector<double>>* terrain_mask = nullptr;
};

struct ChaosTendencies
{
    Field3D* dtheta_micro_dt = nullptr;
    Field3D* dqv_micro_dt = nullptr;
    Field3D* dqc_micro_dt = nullptr;
    Field3D* dqr_micro_dt = nullptr;
    Field3D* dqi_micro_dt = nullptr;
    Field3D* dqs_micro_dt = nullptr;
    Field3D* dqg_micro_dt = nullptr;
    Field3D* dqh_micro_dt = nullptr;

    Field3D* du_pbl_dt = nullptr;
    Field3D* dv_pbl_dt = nullptr;
    Field3D* dw_pbl_dt = nullptr;
    Field3D* dtheta_pbl_dt = nullptr;
    Field3D* dqv_pbl_dt = nullptr;

    Field3D* du_diff_dt = nullptr;
    Field3D* dv_diff_dt = nullptr;
    Field3D* dw_diff_dt = nullptr;
    Field3D* dtheta_diff_dt = nullptr;
    Field3D* dqv_diff_dt = nullptr;
};

struct ChaosEffects
{
    std::unordered_map<std::string, std::vector<std::vector<std::vector<float>>>> delta_state;
    std::unordered_map<std::string, std::vector<std::vector<std::vector<float>>>> multipliers;
    std::vector<std::vector<std::vector<float>>> xi_field;
    double realized_variance = 0.0;
    double min_multiplier = 1.0;
    double max_multiplier = 1.0;
};

struct ChaosDiagnostics
{
    double mean_perturbation = 0.0;
    double variance_perturbation = 0.0;
    double correlation_length_x = 0.0;
    double correlation_length_y = 0.0;

    bool negative_moisture_detected = false;
    int negative_moisture_count = 0;
    double min_qv_after_perturbation = 1.0;

    std::vector<std::string> warnings;
    std::vector<std::string> errors;

    double time_noise_update = 0.0;
    double time_apply_ic = 0.0;
    double time_apply_tend = 0.0;
};

class ChaosScheme
{
public:
    /**
     * @brief Virtual destructor for polymorphic cleanup.
     */
    virtual ~ChaosScheme() = default;

    /**
     * @brief Returns the scheme identifier.
     * @return Scheme name.
     */
    virtual std::string name() const = 0;

    /**
     * @brief Initializes scheme internals for the active grid.
     * @param cfg Chaos configuration.
     * @param grid Grid metrics.
     */
    virtual void initialize(const ChaosConfig& cfg, const GridMetrics& grid) = 0;

    /**
     * @brief Applies perturbations to initial conditions.
     * @param cfg Chaos configuration.
     * @param grid Grid metrics.
     * @param state Mutable simulation state.
     * @param diag Optional diagnostics output.
     */
    virtual void apply_initial_conditions(const ChaosConfig& cfg,
                                          const GridMetrics& grid,
                                          SimulationState& state,
                                          ChaosDiagnostics* diag = nullptr)
    {
    }

    /**
     * @brief Applies perturbations to physics tendencies.
     * @param cfg Chaos configuration.
     * @param grid Grid metrics.
     * @param state_view Read-only state view.
     * @param tendencies Mutable tendency set.
     * @param diag Optional diagnostics output.
     */
    virtual void apply_tendencies(const ChaosConfig& cfg,
                                  const GridMetrics& grid,
                                  const ChaosStateView& state_view,
                                  ChaosTendencies& tendencies,
                                  ChaosDiagnostics* diag = nullptr)
    {
    }

    /**
     * @brief Advances stochastic noise fields in time.
     * @param cfg Chaos configuration.
     * @param grid Grid metrics.
     * @param dt Time step in seconds.
     */
    virtual void step_noise(const ChaosConfig& cfg, const GridMetrics& grid, double dt)
    {
    }
};

/**
 * @brief Creates a chaos scheme by name.
 * @param scheme_name Scheme identifier.
 * @return Owning pointer to a scheme instance.
 */
std::unique_ptr<ChaosScheme> create_chaos_scheme(const std::string& scheme_name);

} // namespace chaos
