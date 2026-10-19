#include "util/transient_workspace.hpp"
#pragma once

struct SimulationStateView;

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/field/field3d.hpp"
#include "numerics/advection/scalar_boundary_report.hpp"
#include "numerics/advection/private_interface_capture.hpp"
#include "numerics/advection/cartesian_interface_flux_register.hpp"
#include "numerics/numerics_base.hpp"

// Advection scheme contracts and persistent transport runtime state.

namespace tmv
{

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

    /// Fused integration target. When set, the TVD v2 kernel writes
    /// dst[k] = src[k] + fused_dt * tendency directly, eliminating the
    /// separate integration pass and halving memory traffic. The caller
    /// sets these before dispatch; the kernel sets fused_completed = true
    /// if fusion was performed. The caller then skips its own integration.
    float* fused_dst = nullptr;
    const float* fused_src = nullptr;
    float fused_dt = 0.0f;
    bool fused_completed = false;
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
    using NumericalSchemeBase::initialize;

    virtual std::unique_ptr<AdvectionSchemeBase> clone() const
    {
        return nullptr;
    }

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

// Scheme selection, reusable transport storage, and activity metadata for one run.
struct AdvectionRuntimeState
{
    tmv::TransientWorkspace<std::vector<Field3D>> private_transport;
    static constexpr std::size_t hydrometeor_count = 6;

    std::unique_ptr<AdvectionSchemeBase> scheme;
    AdvectionConfig config;
    Field3D scalar_scratch_a;
    Field3D scalar_scratch_b;
    AdvectionTendencies vertical_tendencies;
    bool tracer_advected_in_batch = false;
    PositiveDefiniteScalarRoundoffReport tracer_roundoff_report{};
    PositiveDefiniteScalarRoundoffReport cumulative_tracer_roundoff_report{};
    std::uint64_t tracer_roundoff_report_step_count = 0;
    bool tracer_roundoff_cumulative_history_known = true;
    ThermodynamicBoundaryTransportReports thermodynamic_boundary_reports{};
    ThermodynamicBoundaryTransportReports
        cumulative_thermodynamic_boundary_reports{};
    ThermodynamicMoistEnergyTransportReports
        thermodynamic_moist_energy_reports{};
    ThermodynamicMoistEnergyTransportReports
        cumulative_thermodynamic_moist_energy_reports{};
    bool thermodynamic_boundary_reports_requested = false;
    bool thermodynamic_boundary_reports_complete = false;
    bool thermodynamic_moist_energy_reports_requested = false;
    bool thermodynamic_moist_energy_reports_complete = false;
    std::uint64_t thermodynamic_boundary_report_step_count = 0;
    std::uint64_t thermodynamic_moist_energy_report_step_count = 0;
    CartesianInterfaceFluxRegister theta_interface_flux_register;
    CartesianMoistInterfaceCapture moist_interface;
    CartesianPrivateInterfaceCapture      private_interface;
    CartesianDryMassInterfaceFluxRegister dry_mass_interface_flux_register;
    bool theta_interface_flux_capture_enabled = false;
    bool dry_mass_interface_flux_capture_enabled = false;

    std::array<bool, hydrometeor_count> hydrometeor_active{};
    int nr = -1;
    int nth = -1;
    int nz = -1;
    bool activity_current = false;
    std::uint64_t metadata_consumptions = 0;
    std::uint64_t fallback_scans = 0;

    AdvectionRuntimeState() = default;
    AdvectionRuntimeState(const AdvectionRuntimeState& other)
        : config(other.config), scalar_scratch_a(other.scalar_scratch_a), scalar_scratch_b(other.scalar_scratch_b),
          tracer_advected_in_batch(other.tracer_advected_in_batch),
          tracer_roundoff_report(other.tracer_roundoff_report),
          cumulative_tracer_roundoff_report(other.cumulative_tracer_roundoff_report),
          tracer_roundoff_report_step_count(other.tracer_roundoff_report_step_count),
          tracer_roundoff_cumulative_history_known(other.tracer_roundoff_cumulative_history_known),
          thermodynamic_boundary_reports(other.thermodynamic_boundary_reports),
          cumulative_thermodynamic_boundary_reports(other.cumulative_thermodynamic_boundary_reports),
          thermodynamic_moist_energy_reports(other.thermodynamic_moist_energy_reports),
          cumulative_thermodynamic_moist_energy_reports(other.cumulative_thermodynamic_moist_energy_reports),
          thermodynamic_boundary_reports_requested(other.thermodynamic_boundary_reports_requested),
          thermodynamic_boundary_reports_complete(other.thermodynamic_boundary_reports_complete),
          thermodynamic_moist_energy_reports_requested(other.thermodynamic_moist_energy_reports_requested),
          thermodynamic_moist_energy_reports_complete(other.thermodynamic_moist_energy_reports_complete),
          thermodynamic_boundary_report_step_count(other.thermodynamic_boundary_report_step_count),
          thermodynamic_moist_energy_report_step_count(other.thermodynamic_moist_energy_report_step_count),
          theta_interface_flux_register(other.theta_interface_flux_register), moist_interface(other.moist_interface),
          private_interface(other.private_interface),
          dry_mass_interface_flux_register(other.dry_mass_interface_flux_register),
          theta_interface_flux_capture_enabled(other.theta_interface_flux_capture_enabled),
          dry_mass_interface_flux_capture_enabled(other.dry_mass_interface_flux_capture_enabled),
          hydrometeor_active(other.hydrometeor_active), nr(other.nr), nth(other.nth), nz(other.nz),
          activity_current(other.activity_current), metadata_consumptions(other.metadata_consumptions),
          fallback_scans(other.fallback_scans)
    {
        // Fusion pointers are call-local aliases and never survive a snapshot.
        vertical_tendencies.dqdt_adv = other.vertical_tendencies.dqdt_adv;
        if (other.scheme)
        {
            scheme = other.scheme->clone();
            if (!scheme)
            {
                throw std::runtime_error(
                    "advection scheme does not support runtime snapshots: "
                    + other.scheme->name());
            }
        }
    }
    AdvectionRuntimeState& operator=(const AdvectionRuntimeState& other)
    {
        if (this != &other)
        {
            AdvectionRuntimeState copy(other);
            *this = std::move(copy);
        }
        return *this;
    }
    AdvectionRuntimeState(AdvectionRuntimeState&&) noexcept = default;
    AdvectionRuntimeState& operator=(AdvectionRuntimeState&&) noexcept = default;

    void reset_for_shape(int next_nr, int next_nth, int next_nz)
    {
        if (nr == next_nr && nth == next_nth && nz == next_nz)
            return;
        nr = next_nr;
        nth = next_nth;
        nz = next_nz;
        hydrometeor_active.fill(false);
        activity_current = false;
        thermodynamic_boundary_reports = {};
        thermodynamic_boundary_reports_complete = false;
        thermodynamic_moist_energy_reports = {};
        thermodynamic_moist_energy_reports_complete = false;
        tracer_roundoff_report = {};
        theta_interface_flux_register = {};
        moist_interface = {};
        private_interface                           = {};
        dry_mass_interface_flux_register = {};
        theta_interface_flux_capture_enabled = false;
        dry_mass_interface_flux_capture_enabled = false;
    }

    void publish_hydrometeor_activity(
        const std::array<bool, hydrometeor_count>& active,
        int next_nr, int next_nth, int next_nz)
    {
        nr = next_nr;
        nth = next_nth;
        nz = next_nz;
        hydrometeor_active = active;
        activity_current = true;
    }

    void invalidate_hydrometeor_activity()
    {
        activity_current = false;
    }
};

extern AdvectionRuntimeState& global_advection_runtime_state;

// Source-compatible names; the runtime object remains the sole owner.
extern std::unique_ptr<AdvectionSchemeBase>& advection_scheme;
extern AdvectionConfig& global_advection_config;

/**
 * @brief Initializes the global advection subsystem.
 * @param scheme_name Scheme identifier.
 * @param cfg Optional advection configuration.
 */
void initialize_advection(const std::string& scheme_name, const AdvectionConfig& cfg = AdvectionConfig{});
void initialize_advection(const SimulationStateView& execution_state, const std::string& scheme_name, const AdvectionConfig& cfg = AdvectionConfig{});
void initialize_advection(AdvectionRuntimeState& runtime,
                          const std::string& scheme_name,
                          const AdvectionConfig& cfg = AdvectionConfig{});

} // namespace tmv
#include "numerics/advection/factory.hpp"

using tmv::AdvectionConfig;
using tmv::AdvectionStateView;
using tmv::AdvectionTendencies;
using tmv::AdvectionDiagnostics;
using tmv::AdvectionSchemeBase;
using tmv::AdvectionRuntimeState;
using tmv::AdvectionSchemeFactory;
using tmv::global_advection_runtime_state;
using tmv::advection_scheme;
using tmv::global_advection_config;
using tmv::create_advection_scheme;
using tmv::get_available_advection_schemes;
using tmv::initialize_advection;
