#pragma once

#include "core/field/field3d.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "core/infra/nested_grid.hpp"
#include "core/infra/nested_terrain_geometry.hpp"
#include "numerics/advection/cartesian_interface_flux_register.hpp"

struct ComputeBackendRuntimeState;

class NestedScalarInventoryExhausted : public std::domain_error
{
public:
    using std::domain_error::domain_error;
};

struct NestedScalarFeedbackReport
{
    double      initial_inventory              = 0;
    double      final_inventory                = 0;
    double      child_synchronized_inventory   = 0;
    double      fine_requested_transfer        = 0;
    double      fine_storage_rounding          = 0;
    double      child_inventory                = 0;
    double      restriction_change             = 0;
    double      interface_mismatch             = 0;
    double      applied_reflux                 = 0;
    double      storage_rounding               = 0;
    double      phase_exchange_inventory       = 0;
    double      closure_residual               = 0;
    double      redistributed_inventory        = 0;
    std::size_t redistributed_cells            = 0;
    double      global_redistributed_inventory = 0;
    std::size_t global_redistributed_cells     = 0;
    std::size_t restricted_cells               = 0;
    std::size_t refluxed_cells                 = 0;
    std::size_t constrained_cells              = 0;
    bool        complete                       = false;
};

inline bool valid_nested_scalar_feedback_report(const NestedScalarFeedbackReport& r) noexcept
{
    if (!r.complete || !r.restricted_cells || !r.refluxed_cells || !std::isfinite(r.redistributed_inventory) ||
        r.redistributed_inventory < 0 ||
        (r.redistributed_cells > r.refluxed_cells && r.redistributed_cells - r.refluxed_cells > r.constrained_cells) ||
        ((r.redistributed_cells == 0) != (r.redistributed_inventory == 0)) ||
        !std::isfinite(r.global_redistributed_inventory) || r.global_redistributed_inventory < 0 ||
        r.global_redistributed_inventory > r.redistributed_inventory ||
        r.global_redistributed_cells > r.redistributed_cells ||
        ((r.global_redistributed_cells == 0) != (r.global_redistributed_inventory == 0)))
        return false;
    double scale = 1;
    for (double value : {r.initial_inventory,
                         r.final_inventory,
                         r.child_inventory,
                         r.child_synchronized_inventory,
                         r.fine_requested_transfer,
                         r.fine_storage_rounding,
                         r.restriction_change,
                         r.interface_mismatch,
                         r.applied_reflux,
                         r.storage_rounding,
                         r.phase_exchange_inventory,
                         r.closure_residual})
    {
        if (!std::isfinite(value))
            return false;
        scale = std::max(scale, std::abs(value));
    }
    const double tolerance = 128 * std::numeric_limits<double>::epsilon() * scale;
    const double residual  = r.final_inventory - r.initial_inventory - r.restriction_change - r.applied_reflux -
                             r.phase_exchange_inventory - r.storage_rounding;
    return r.fine_requested_transfer >= 0 &&
           r.fine_requested_transfer <= r.global_redistributed_inventory + tolerance &&
           std::abs(r.child_synchronized_inventory - r.child_inventory + r.fine_requested_transfer -
                    r.fine_storage_rounding) <= tolerance &&
           std::abs(residual) <= tolerance && std::abs(r.closure_residual - residual) <= tolerance &&
           std::abs(r.interface_mismatch - r.applied_reflux) <= tolerance;
}

struct PreparedNestedScalarFeedback
{
    Field3D                    scalar;
    Field3D                    child_scalar;
    NestedScalarFeedbackReport report;
};

struct NestedScalarFeedbackBounds
{
    const Field3D* parent = nullptr;
    const Field3D* child  = nullptr;
};

PreparedNestedScalarFeedback
prepare_nested_scalar_feedback_flat(const tmv::domain::OneWayNestRelationshipV1& relationship,
                                    const std::vector<double>&                   layer_thickness_m,
                                    const Field3D&                               parent_density_before,
                                    const Field3D&                               parent_density_after,
                                    const Field3D&                               child_density,
                                    const Field3D&                               parent_scalar,
                                    const Field3D&                               child_scalar,
                                    const tmv::CartesianThetaInterfaceFluxes&    parent_flux,
                                    const tmv::CartesianThetaInterfaceFluxes&    child_flux,
                                    bool                                         volume_concentration      = false,
                                    bool                                         positive                  = true,
                                    bool                                         allow_fine_redistribution = false,
                                    ComputeBackendRuntimeState*                  compute                   = nullptr,
                                    NestedScalarFeedbackBounds                   bounds                    = {});

PreparedNestedScalarFeedback
prepare_nested_scalar_feedback_mapped(const tmv::domain::NestedMappedTerrainGeometryV1& geometry,
                                      const Field3D&                                    parent_density_before,
                                      const Field3D&                                    parent_density_after,
                                      const Field3D&                                    child_density,
                                      const Field3D&                                    parent_scalar,
                                      const Field3D&                                    child_scalar,
                                      const tmv::CartesianThetaInterfaceFluxes&         parent_flux,
                                      const tmv::CartesianThetaInterfaceFluxes&         child_flux,
                                      bool                                              volume_concentration = false,
                                      bool                                              positive             = true,
                                      bool                        allow_fine_redistribution                  = false,
                                      ComputeBackendRuntimeState* compute                                    = nullptr,
                                      NestedScalarFeedbackBounds  bounds                                     = {});

struct SimulationStateView;
std::array<double, 8> nested_thermodynamic_inventory(const SimulationStateView& state);
std::vector<double>   nested_private_inventory(const SimulationStateView& state);

double nested_private_carrier_inventory(const tmv::domain::OneWayNestRelationshipV1& relationship,
                                        const std::vector<double>&                   layer_thickness_m,
                                        const Field3D&                               density,
                                        const Field3D&                               scalar,
                                        int                                          shifted_i              = 0,
                                        int                                          shifted_j              = 0,
                                        bool                                         outside_shifted_domain = false,
                                        const TerrainMetrics3D*                      metrics                = nullptr);

struct PrognosticFieldStorage;
std::array<double, 12> nested_primary_inventory(const tmv::domain::OneWayNestRelationshipV1& relationship,
                                                const std::vector<double>&                   layer_thickness_m,
                                                const PrognosticFieldStorage&                fields,
                                                int                                          shifted_i = 0,
                                                int                                          shifted_j = 0,
                                                bool                    outside_shifted_domain         = false,
                                                bool                    absolute_contributions         = false,
                                                const TerrainMetrics3D* metrics                        = nullptr);
