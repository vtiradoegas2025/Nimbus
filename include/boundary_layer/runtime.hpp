#pragma once

#include "boundary_layer/boundary_layer_base.hpp"
#include "core/field/field3d.hpp"

#include <limits>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct SimulationStateView;

namespace tmv
{

struct BoundaryLayerMomentumWorkReport
{
    // Counterfactual ordering: resolved dynamics first, then PBL momentum.
    double requested_linear_work_j = 0.0;
    double finite_step_correction_j = 0.0;
    double storage_rounding_j = 0.0;
    double committed_work_j = 0.0;
};

struct BoundaryLayerWorkerContext
{
    std::unique_ptr<BoundaryLayerSchemeBase> scheme;
    BoundaryLayerColumnWorkspace workspace;
    BoundaryLayerTendencies tendencies;
    BoundaryLayerDiagnostics diagnostics;
};

// Reusable candidate storage for one transactional PBL cadence refresh.
// Runtime copies and checkpoints deliberately omit this replaceable state.
struct BoundaryLayerRefreshScratch
{
    Field3D dtheta;
    Field3D dqv;
    Field3D dqc;
    Field3D dqi;
    Field3D du;
    Field3D dv;
    Field3D dtke;
    Field3D u_force;
    Field3D v_force;
    Field3D theta_inventory_rate;
    Field3D vapor_mass_rate;
    Field3D cloud_liquid_mass_rate;
    Field3D cloud_ice_mass_rate;

    std::vector<float> ustar;
    std::vector<float> tau_u;
    std::vector<float> tau_v;
    std::vector<float> sensible_heat_flux;
    std::vector<float> latent_heat_flux;
    std::vector<float> moisture_flux;
    std::vector<float> pbl_height;
    std::vector<float> skin_theta;
    std::vector<float> skin_qv;
    std::vector<SlabColumnState> slab_columns;
    std::vector<BoundaryLayerWorkerContext> batch_columns;
    std::vector<BoundaryLayerColumnStateView> batch_views;
    std::vector<BoundaryLayerTendencies> batch_tendencies;
    std::vector<BoundaryLayerDiagnostics> batch_diagnostics;

    void release_storage() noexcept
    {
        dtheta.release_storage();
        dqv.release_storage();
        dqc.release_storage();
        dqi.release_storage();
        du.release_storage();
        dv.release_storage();
        dtke.release_storage();
        u_force.release_storage();
        v_force.release_storage();
        theta_inventory_rate.release_storage();
        vapor_mass_rate.release_storage();
        cloud_liquid_mass_rate.release_storage();
        cloud_ice_mass_rate.release_storage();
        std::vector<float>().swap(ustar);
        std::vector<float>().swap(tau_u);
        std::vector<float>().swap(tau_v);
        std::vector<float>().swap(sensible_heat_flux);
        std::vector<float>().swap(latent_heat_flux);
        std::vector<float>().swap(moisture_flux);
        std::vector<float>().swap(pbl_height);
        std::vector<float>().swap(skin_theta);
        std::vector<float>().swap(skin_qv);
        std::vector<SlabColumnState>().swap(slab_columns);
        std::vector<BoundaryLayerWorkerContext>().swap(batch_columns);
        std::vector<BoundaryLayerColumnStateView>().swap(batch_views);
        std::vector<BoundaryLayerTendencies>().swap(batch_tendencies);
        std::vector<BoundaryLayerDiagnostics>().swap(batch_diagnostics);
    }
};

// Persistent scheme, cadence, slab, and held-skin state for one simulation.
struct BoundaryLayerRuntimeState
{
    std::unique_ptr<BoundaryLayerSchemeBase> scheme;
    BoundaryLayerConfig config;
    SurfaceConfig surface_config;
    double last_update_time_s = std::numeric_limits<double>::quiet_NaN();
    // Next nominal cadence boundary. Keep this separate from the actual
    // publication time so a dynamics step that crosses the boundary late
    // cannot permanently phase-shift later PBL samples.
    double next_update_time_s = 0.0;
    bool updated_this_step = false;

    std::vector<float> skin_theta_2d;
    std::vector<float> skin_qv_2d;
    bool skin_captured = false;
    int skin_nr = -1;
    int skin_nth = -1;

    std::vector<SlabColumnState> slab_columns;
    int slab_nr = -1;
    int slab_nth = -1;

    // Conserved sample-and-hold coupling currency. Specific tendencies are
    // reconstructed from these rates with the live density immediately before
    // application, so density evolution between PBL refreshes cannot change
    // the held force or scalar inventory source.
    Field3D held_u_force_n;
    Field3D held_v_force_n;
    Field3D held_theta_inventory_rate_kgk_s;
    Field3D held_vapor_mass_rate_kg_s;
    Field3D held_cloud_liquid_mass_rate_kg_s;
    Field3D held_cloud_ice_mass_rate_kg_s;

    BoundaryLayerRefreshScratch refresh_scratch;
    std::vector<BoundaryLayerWorkerContext> worker_contexts;

    BoundaryLayerMomentumWorkReport last_momentum_work;
    double cumulative_requested_momentum_work_j = 0.0;
    double cumulative_finite_step_momentum_correction_j = 0.0;
    double cumulative_momentum_storage_rounding_j = 0.0;
    double cumulative_committed_momentum_work_j = 0.0;
    std::uint64_t momentum_work_step_count = 0;

    BoundaryLayerRuntimeState() = default;
    BoundaryLayerRuntimeState(const BoundaryLayerRuntimeState& other)
        : config(other.config),
          surface_config(other.surface_config),
          last_update_time_s(other.last_update_time_s),
          next_update_time_s(other.next_update_time_s),
          updated_this_step(other.updated_this_step),
          skin_theta_2d(other.skin_theta_2d),
          skin_qv_2d(other.skin_qv_2d),
          skin_captured(other.skin_captured),
          skin_nr(other.skin_nr),
          skin_nth(other.skin_nth),
          slab_columns(other.slab_columns),
          slab_nr(other.slab_nr),
          slab_nth(other.slab_nth),
          held_u_force_n(other.held_u_force_n),
          held_v_force_n(other.held_v_force_n),
          held_theta_inventory_rate_kgk_s(
              other.held_theta_inventory_rate_kgk_s),
          held_vapor_mass_rate_kg_s(other.held_vapor_mass_rate_kg_s),
          held_cloud_liquid_mass_rate_kg_s(
              other.held_cloud_liquid_mass_rate_kg_s),
          held_cloud_ice_mass_rate_kg_s(
              other.held_cloud_ice_mass_rate_kg_s),
          last_momentum_work(other.last_momentum_work),
          cumulative_requested_momentum_work_j(
              other.cumulative_requested_momentum_work_j),
          cumulative_finite_step_momentum_correction_j(
              other.cumulative_finite_step_momentum_correction_j),
          cumulative_momentum_storage_rounding_j(
              other.cumulative_momentum_storage_rounding_j),
          cumulative_committed_momentum_work_j(
              other.cumulative_committed_momentum_work_j),
          momentum_work_step_count(other.momentum_work_step_count)
    {
        if (other.scheme)
        {
            scheme = other.scheme->clone();
            if (!scheme)
            {
                throw std::runtime_error(
                    "boundary-layer scheme does not support runtime snapshots: "
                    + other.scheme->name());
            }
        }
    }
    BoundaryLayerRuntimeState& operator=(const BoundaryLayerRuntimeState& other)
    {
        if (this != &other)
        {
            BoundaryLayerRuntimeState copy(other);
            *this = std::move(copy);
        }
        return *this;
    }
    BoundaryLayerRuntimeState(BoundaryLayerRuntimeState&&) noexcept = default;
    BoundaryLayerRuntimeState& operator=(BoundaryLayerRuntimeState&&) noexcept = default;
};

extern BoundaryLayerRuntimeState& global_boundary_layer_runtime_state;

// Compatibility aliases for the process-owned runtime.
extern std::unique_ptr<BoundaryLayerSchemeBase>& boundary_layer_scheme;
extern BoundaryLayerConfig& global_boundary_layer_config;
extern SurfaceConfig& global_surface_config;

void initialize_boundary_layer(
    const std::string& scheme_name,
    const BoundaryLayerConfig& cfg = BoundaryLayerConfig{},
    const SurfaceConfig& sfc = SurfaceConfig{});
void initialize_boundary_layer(const SimulationStateView& execution_state,
    const std::string& scheme_name,
    const BoundaryLayerConfig& cfg = BoundaryLayerConfig{},
    const SurfaceConfig& sfc = SurfaceConfig{});
void initialize_boundary_layer(
    BoundaryLayerRuntimeState& runtime,
    const std::string& scheme_name,
    const BoundaryLayerConfig& cfg = BoundaryLayerConfig{},
    const SurfaceConfig& sfc = SurfaceConfig{});

// Mapped terrain uses tangent-wind surface traction expressed as generalized
// horizontal forces on the active stagger's physical momentum volumes.
void step_boundary_layer(const SimulationStateView& state, double current_time);
void step_boundary_layer(double current_time);

// Reconstructs specific PBL tendencies from held physical/inventory rates and
// the current density. Called before momentum and scalar application. Active
// terrain reconstructs the same generalized face forces from held sources.
void prepare_boundary_layer_tendencies_for_application(
    const SimulationStateView& state);
// Rebuild every live-density specific tendency after a persistent-source
// shift. dtke_dt_pbl is already shifted as persistent sampled state.
void rebuild_boundary_layer_tendencies_after_domain_recycling(
    const SimulationStateView& state);
const char* boundary_layer_domain_recycling_finalize_algorithm_id() noexcept;
void commit_boundary_layer_momentum_work(
    const SimulationStateView& state,
    const BoundaryLayerMomentumWorkReport& report);

bool boundary_layer_updated_this_step();
bool boundary_layer_updated_this_step(const SimulationStateView& state);

} // namespace tmv

using tmv::BoundaryLayerRuntimeState;
using tmv::BoundaryLayerMomentumWorkReport;
using tmv::boundary_layer_scheme;
using tmv::global_boundary_layer_config;
using tmv::global_surface_config;
using tmv::initialize_boundary_layer;
using tmv::step_boundary_layer;
using tmv::prepare_boundary_layer_tendencies_for_application;
using tmv::rebuild_boundary_layer_tendencies_after_domain_recycling;
using tmv::commit_boundary_layer_momentum_work;
using tmv::boundary_layer_updated_this_step;
