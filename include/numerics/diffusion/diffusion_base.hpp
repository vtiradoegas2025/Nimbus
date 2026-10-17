#pragma once
#include "core/infra/flow_execution.hpp"

struct ComputeBackendRuntimeState;

struct SimulationStateView;

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/field/field3d.hpp"
#include "numerics/numerics_base.hpp"

// Diffusion scheme contracts and persistent runtime state.

namespace tmv
{

struct DiffusionConfig
{
    std::string scheme_id = "explicit";
    // The production diffusion operators are second order. Keep the selected
    // order in the runtime config so strict-v1 projection cannot silently drop
    // an accepted scientific choice.
    int operator_order = 2;
    std::string operator_type = "laplacian";
    std::string apply_to = "scalars";
    double K_h = 0.0;
    double K_v = 0.0;
    std::string implicit_dim = "none";
    double dt_diffusion = 1.0;
    bool use_variable_K = false;
    std::string coefficient_source = "constant";
    double K_v_surface = 0.0;
    double K_v_top = 0.0;
};

struct DiffusionStateView : FlowExecutionInputs
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
    // True for Arakawa C-grid velocity staggering. Horizontal momentum
    // diffusion uses component-specific control-volume density and ownership;
    // direct operator tests default to the production C-grid contract.
    bool c_grid = true;
    // Horizontal diffusion owns physical boundary-face semantics explicitly;
    // it must not infer them from potentially stale closure storage.
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
    // Maximum active directional coefficient at each cell. This is not an
    // isotropic-K claim when K_h != K_v.
    Field3D K_effective;
};

// Exact resolved kinetic-energy attribution for a committed momentum-
// diffusion update in its native A/C-grid control-volume currency.
struct DiffusionMomentumWorkReport
{
    static constexpr std::uint32_t report_schema_version = 1;
    static constexpr std::uint32_t energy_basis_version = 2;

    double requested_linear_work_j = 0.0;
    double finite_step_correction_j = 0.0;
    double storage_rounding_j = 0.0;
    double committed_work_j = 0.0;
    double moist_dry_internal_change_j = 0.0;
    double moist_water_sensible_change_j = 0.0;
    double moist_phase_reference_change_j = 0.0;
    double moist_carrier_kinetic_change_j = 0.0;
    double moist_gravitational_change_j = 0.0;
    double moist_subgrid_tke_change_j = 0.0;
    double moist_committed_energy_change_j = 0.0;
    double moist_energy_closure_residual_j = 0.0;
    bool moist_energy_complete = false;
};

class DiffusionSchemeBase : public NumericalSchemeBase
{
public:
    using NumericalSchemeBase::initialize;

    virtual std::unique_ptr<DiffusionSchemeBase> clone() const
    {
        return nullptr;
    }

    virtual void initialize(const DiffusionConfig& cfg) = 0;

    // Writes tendencies without modifying the supplied prognostic fields.
    virtual void compute_diffusion_tendencies(const DiffusionConfig& cfg,
                                              const DiffusionStateView& state,
                                              DiffusionTendencies& tendencies,
                                              DiffusionDiagnostics* diag_opt = nullptr) = 0;

    // Values above one exceed the scheme's nominal stability limit.
    virtual double check_stability(const DiffusionConfig& cfg, const DiffusionStateView& state) = 0;
};

using DiffusionSchemeFactory = std::unique_ptr<DiffusionSchemeBase> (*)(const DiffusionConfig&);

struct DiffusionRuntimeState
{
    std::unique_ptr<DiffusionSchemeBase> scheme;
    DiffusionConfig config;
    DiffusionTendencies tendencies;
    DiffusionDiagnostics diagnostics;
    // Reusable transactional state candidates. They are deliberately omitted
    // from copy construction and checkpoint serialization; after commit the
    // previous prognostic buffers are swapped back here for the next step.
    Field3D scratch_u;
    Field3D scratch_v;
    Field3D scratch_w;
    Field3D scratch_theta;
    Field3D scratch_qv;
    Field3D prescribed_vertical_K;
    std::vector<double> prescribed_vertical_dz;
    int prescribed_vertical_nr = 0;
    int prescribed_vertical_nth = 0;
    int prescribed_vertical_nz = 0;
    double prescribed_vertical_surface = 0.0;
    double prescribed_vertical_top = 0.0;
    DiffusionMomentumWorkReport last_momentum_work;
    DiffusionMomentumWorkReport cumulative_moist_energy_work;
    double cumulative_requested_momentum_work_j = 0.0;
    double cumulative_finite_step_momentum_correction_j = 0.0;
    double cumulative_momentum_storage_rounding_j = 0.0;
    double cumulative_committed_momentum_work_j = 0.0;
    std::uint64_t momentum_work_step_count = 0;
    std::uint64_t moist_energy_step_count = 0;

    DiffusionRuntimeState() = default;
    DiffusionRuntimeState(const DiffusionRuntimeState& other)
        : config(other.config),
          tendencies(other.tendencies),
          diagnostics(other.diagnostics),
          last_momentum_work(other.last_momentum_work),
          cumulative_moist_energy_work(other.cumulative_moist_energy_work),
          cumulative_requested_momentum_work_j(
              other.cumulative_requested_momentum_work_j),
          cumulative_finite_step_momentum_correction_j(
              other.cumulative_finite_step_momentum_correction_j),
          cumulative_momentum_storage_rounding_j(
              other.cumulative_momentum_storage_rounding_j),
          cumulative_committed_momentum_work_j(
              other.cumulative_committed_momentum_work_j),
          momentum_work_step_count(other.momentum_work_step_count),
          moist_energy_step_count(other.moist_energy_step_count)
    {
        if (other.scheme)
        {
            scheme = other.scheme->clone();
            if (!scheme)
            {
                throw std::runtime_error(
                    "diffusion scheme does not support runtime snapshots: "
                    + other.scheme->name());
            }
        }
    }
    DiffusionRuntimeState& operator=(const DiffusionRuntimeState& other)
    {
        if (this != &other)
        {
            DiffusionRuntimeState copy(other);
            *this = std::move(copy);
        }
        return *this;
    }
    DiffusionRuntimeState(DiffusionRuntimeState&&) noexcept = default;
    DiffusionRuntimeState& operator=(DiffusionRuntimeState&&) noexcept = default;
};

extern DiffusionRuntimeState& global_diffusion_runtime_state;

// Source-compatible names; the runtime object remains the sole owner.
extern std::unique_ptr<DiffusionSchemeBase>& diffusion_scheme;
extern DiffusionConfig& global_diffusion_config;

void initialize_diffusion(const std::string& scheme_name, const DiffusionConfig& cfg = DiffusionConfig{});
void initialize_diffusion(const SimulationStateView& execution_state, const std::string& scheme_name, const DiffusionConfig& cfg = DiffusionConfig{});
void initialize_diffusion(DiffusionRuntimeState& runtime,
                          const std::string& scheme_name,
                          const DiffusionConfig& cfg = DiffusionConfig{});

} // namespace tmv
#include "numerics/diffusion/factory.hpp"

using tmv::DiffusionConfig;
using tmv::DiffusionStateView;
using tmv::DiffusionTendencies;
using tmv::DiffusionDiagnostics;
using tmv::DiffusionMomentumWorkReport;
using tmv::DiffusionSchemeBase;
using tmv::DiffusionRuntimeState;
using tmv::DiffusionSchemeFactory;
using tmv::global_diffusion_runtime_state;
using tmv::diffusion_scheme;
using tmv::global_diffusion_config;
using tmv::create_diffusion_scheme;
using tmv::get_available_diffusion_schemes;
using tmv::initialize_diffusion;
