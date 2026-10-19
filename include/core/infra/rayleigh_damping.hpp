#pragma once

#include "dynamics/sponge_energy_report.hpp"

#include <vector>

struct SimulationStateView;

// Per-simulation lazy damping profiles and initialization state.
struct RayleighDampingRuntimeState
{
    bool upper_initialized = false;
    bool lateral_initialized = false;
    std::vector<double> alpha;
    std::vector<double> u_ref;
    std::vector<double> v_ref;
    std::vector<double> upper_w_alpha;
    std::vector<double> w_only_alpha;
    std::vector<double> lateral_alpha;
};

extern RayleighDampingRuntimeState& global_rayleigh_runtime_state;

tmv::SpongeEnergyReportV1 apply_rayleigh_damping(
    const SimulationStateView& state, double dt_damp);
tmv::SpongeEnergyReportV1 apply_rayleigh_damping(double dt_damp);

tmv::SpongeEnergyReportV1 apply_lateral_damping(
    const SimulationStateView& state, double dt_damp);
tmv::SpongeEnergyReportV1 apply_lateral_damping(double dt_damp);

/**
 * Attribute the flat-Cartesian fused device relaxation from the untouched
 * host state and actual post-fence device outputs. The packed profiles and
 * flags are the exact shader inputs. Intermediate pass states are replayed in
 * FP32; each component's last active pass closes against the actual device
 * value so the three reports telescope to the committed endpoint.
 */
tmv::SpongeEnergyReportV1 attribute_state_relaxation_cartesian_device_commit(
    const SimulationStateView& state,
    float dt_step,
    const float* profiles,
    int profiles_len,
    int lateral_width,
    bool do_upper,
    bool do_w_only,
    bool do_lateral,
    const float* u_committed,
    const float* v_committed,
    const float* w_committed,
    const float* theta_committed);

void reset_rayleigh_damping();
void reset_rayleigh_damping(const SimulationStateView& execution_state);
void reset_rayleigh_damping(RayleighDampingRuntimeState& runtime);

const std::vector<double>& rayleigh_alpha_profile(const SimulationStateView& state);
const std::vector<double>& rayleigh_alpha_profile();

const std::vector<double>& rayleigh_w_only_alpha_profile(const SimulationStateView& state);
const std::vector<double>& rayleigh_w_only_alpha_profile();

const std::vector<double>& rayleigh_upper_w_alpha_profile(
    const SimulationStateView& state);
const std::vector<double>& rayleigh_upper_w_alpha_profile();

const std::vector<double>& rayleigh_u_ref_profile(const SimulationStateView& state);
const std::vector<double>& rayleigh_u_ref_profile();

const std::vector<double>& rayleigh_v_ref_profile(const SimulationStateView& state);
const std::vector<double>& rayleigh_v_ref_profile();

const std::vector<double>& rayleigh_lateral_alpha_profile(const SimulationStateView& state);
const std::vector<double>& rayleigh_lateral_alpha_profile();

bool rayleigh_upper_sponge_enabled();
bool rayleigh_upper_sponge_enabled(const SimulationStateView& execution_state);
bool rayleigh_w_only_sponge_enabled();
bool rayleigh_w_only_sponge_enabled(const SimulationStateView& execution_state);
bool rayleigh_lateral_sponge_enabled();
bool rayleigh_lateral_sponge_enabled(const SimulationStateView& execution_state);

/**
 * Validate that every initialized Rayleigh profile is the exact profile
 * implied by the current vertical grid, hodograph, and damping configuration.
 * A flat horizontal recycling translation does not change any of those inputs, so
 * successful validation proves the runtime can be preserved byte-for-value.
 * The checkpoint-lazy missing upper-w profile is accepted because the normal
 * initializer deterministically reconstructs the complete upper profile set.
 */
void validate_rayleigh_runtime_for_flat_horizontal_recycling(
    const SimulationStateView& state);

const char* rayleigh_horizontal_recycling_algorithm_id() noexcept;
