#pragma once

#include "compute/dispatch/terrain.hpp"
#include "dynamics/acoustic_energy_report.hpp"
#include "dynamics/eos_pressure.hpp"
#include "dynamics/final_boundary_energy_report.hpp"
#include "dynamics/slow_momentum_energy_report.hpp"
#include "numerics/advection/scalar_boundary_report.hpp"

#include <cstdint>
#include <array>
#include "core/field/field3d.hpp"
#include "boundary_layer/runtime.hpp"

struct SimulationStateView;
class Field3D;

namespace tmv::terrain_native
{

struct MappedCoreStepReport
{
    std::uint32_t                                  acoustic_substeps            = 0;
    std::uint32_t                                  mass_continuity_dispatches   = 0;
    std::uint32_t                                  slow_tendency_dispatches     = 0;
    std::uint32_t                                  slow_average_dispatches      = 0;
    std::uint32_t                                  fast_momentum_dispatches     = 0;
    std::uint32_t                                  state_update_dispatches      = 0;
    std::uint32_t                                  flux_accumulation_dispatches = 0;
    std::uint32_t                                  scalar_flux_dispatches       = 0;
    std::uint32_t                                  scalar_limiter_dispatches    = 0;
    std::uint32_t                                  scalar_update_dispatches     = 0;
    std::uint32_t                                  boundary_dispatches          = 0;
    std::uint64_t                                  completed_backend_dispatches = 0;
    double                                         accumulated_time_weight      = 0.0;
    CartesianCGridAcousticEnergyReportV1           acoustic;
    CartesianSlowMomentumEnergyReportV1            slow_momentum;
    ThermodynamicBoundaryTransportReports          thermodynamic_boundary;
    ThermodynamicMoistEnergyTransportReports       thermodynamic_moist_energy;
    dynamics_eos::PressureProjectionEnergyReportV1 final_eos;
    FinalBoundaryEnergyReportV1                    final_boundary;
    bool                                           final_pressure_uses_canonical_fp64 = false;
    bool                                           complete                           = false;
};

using MappedCoreCompletedObserver = void (*)(void*                       context,
                                             const Field3D&              u,
                                             const Field3D&              v,
                                             const Field3D&              w,
                                             const Field3D&              rho,
                                             const Field3D&              p,
                                             const Field3D&              theta,
                                             const Field3D&              qv,
                                             const Field3D&              tracer,
                                             const Field3D&              tke,
                                             const Field3D&              density_begin,
                                             const Field3D&              mass_flux_x,
                                             const Field3D&              mass_flux_y,
                                             const Field3D&              mass_flux_z,
                                             const MappedCoreStepReport& report);

struct MappedAcousticResult
{
    CartesianCGridAcousticEnergyReportV1 acoustic;
    CartesianSlowMomentumEnergyReportV1  slow;
    BoundaryLayerMomentumWorkReport      pbl;
    Field3D                              density_begin, mass_flux_x, mass_flux_y, mass_flux_z;
    std::uint64_t                        completed_backend_dispatches = 0;
    bool                                 complete                     = false;
};
bool prepare_mapped_momentum_tendencies(const SimulationStateView& state,
                                       const Field3D& u, const Field3D& v, const Field3D& w,
                                       const Field3D& rho, const Field3D& pressure, bool pressure_only,
                                       std::array<Field3D, 8>& result);

bool advance_mapped_acoustic_core(const SimulationStateView& state,
                                  double                     dt,
                                  int                        acoustic_substeps,
                                  MappedAcousticResult&      result);

bool advance_dry_stationary_mapped_core(const SimulationStateView&  state,
                                        double                      dt_large,
                                        int                         acoustic_substeps,
                                        MappedCoreStepReport&       report,
                                        MappedCoreCompletedObserver completed_observer         = nullptr,
                                        void*                       completed_observer_context = nullptr);

} // namespace tmv::terrain_native
