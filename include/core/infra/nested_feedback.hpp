#pragma once

#include "core/infra/nested_grid.hpp"
#include "core/infra/nested_scalar_feedback.hpp"
#include "core/infra/nested_terrain_geometry.hpp"
#include "dynamics/cartesian_cgrid_momentum_flux.hpp"
#include "numerics/advection/cartesian_interface_flux_register.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

struct PrognosticFieldStorage;
struct ComputeBackendRuntimeState;

struct NestedMoistFeedbackInputs
{
    std::array<const tmv::CartesianScalarInterfaceFluxes*, 7> parent_flux{}, child_flux{};
    std::array<double, 7> child_begin{}, interface_source{}, physics_source{}, transport_storage{};
};
struct NestedWaterFeedbackReport
{
    NestedScalarFeedbackReport parent;
    double child_begin = 0, child_transport = 0, interface_source = 0, physics_source = 0, transport_storage = 0;
    double child_change = 0, closure_residual = 0, tolerance = 0;
};

inline bool valid_nested_water_feedback_report(const NestedWaterFeedbackReport& r) noexcept
{
    if (!valid_nested_scalar_feedback_report(r.parent) || r.parent.initial_inventory < 0 ||
        r.parent.final_inventory < 0 || r.parent.child_inventory < 0 || r.parent.child_synchronized_inventory < 0 ||
        r.child_begin < 0)
        return false;
    double scale = 0;
    for (double value : {r.child_begin,
                         r.parent.child_inventory,
                         r.child_transport,
                         r.interface_source,
                         r.physics_source,
                         r.transport_storage})
    {
        if (!std::isfinite(value))
            return false;
        scale = std::max(scale, std::abs(value));
    }
    const double expected_tolerance = 512 * std::numeric_limits<float>::epsilon() * scale;
    const double change             = r.parent.child_inventory - r.child_begin;
    const double residual   = change - r.child_transport - r.interface_source - r.physics_source - r.transport_storage;
    const double arithmetic = 128 * std::numeric_limits<double>::epsilon() * std::max(1.0, scale);
    return std::isfinite(r.child_change) && std::isfinite(r.closure_residual) && r.tolerance == expected_tolerance &&
           std::abs(r.child_change - change) <= arithmetic && std::abs(r.closure_residual - residual) <= arithmetic &&
           std::abs(residual) <= r.tolerance;
}

struct NestedPrivateScalarFeedbackReport
{
    std::string               name;
    NestedWaterFeedbackReport carrier;
};

inline bool
valid_nested_private_feedback_reports(const std::vector<NestedPrivateScalarFeedbackReport>& reports) noexcept
{
    if (reports.size() > 16)
        return false;
    double cloud_return = 0, aerosol_withdrawal = 0;
    for (std::size_t i = 0; i < reports.size(); ++i)
    {
        if (reports[i].name.empty() || reports[i].name.size() >= 64 ||
            !valid_nested_water_feedback_report(reports[i].carrier))
            return false;
        const double exchange = reports[i].carrier.parent.phase_exchange_inventory;
        if (reports[i].name == "thompson.Nc_kg-1" && exchange >= 0)
            cloud_return = exchange;
        else if (reports[i].name == "thompson.Nwfa_kg-1" && exchange <= 0)
            aerosol_withdrawal = -exchange;
        else if (exchange != 0)
            return false;
        for (std::size_t j = 0; j < i; ++j)
            if (reports[i].name == reports[j].name)
                return false;
    }
    return cloud_return == aerosol_withdrawal;
}

struct NestedPrivateScalarFeedbackInput
{
    std::string                                name;
    const Field3D*                             parent      = nullptr;
    const Field3D*                             child       = nullptr;
    const tmv::CartesianScalarInterfaceFluxes* parent_flux = nullptr;
    const tmv::CartesianScalarInterfaceFluxes* child_flux  = nullptr;
    double         child_begin = 0, interface_source = 0, physics_source = 0, transport_storage = 0;
    const Field3D* parent_mass_before = nullptr;
    const Field3D* parent_mass_after  = nullptr;
    const Field3D* child_mass         = nullptr;
};

struct PreparedNestedPrivateFeedback
{
    std::vector<Field3D>                           parent, child;
    std::vector<NestedPrivateScalarFeedbackReport> reports;
};

PreparedNestedPrivateFeedback
prepare_nested_private_feedback(const tmv::domain::OneWayNestRelationshipV1&         relationship,
                                const std::vector<double>&                           layer_thickness,
                                const tmv::domain::NestedMappedTerrainGeometryV1*    geometry,
                                const Field3D&                                       parent_density_before,
                                const Field3D&                                       parent_density_after,
                                const Field3D&                                       child_density,
                                const std::vector<NestedPrivateScalarFeedbackInput>& inputs,
                                ComputeBackendRuntimeState*                          compute                = nullptr,
                                const Field3D*                                       completed_cloud_return = nullptr);

struct NestedPrivateRelocationInventory
{
    std::string name;
    double      before = 0, retired = 0, admitted = 0, boundary_source = 0, after = 0, arithmetic_scale = 0;
};

struct NestedTwoWayRelocationReportV1
{
    bool                   complete                   = false;
    std::uint64_t          movement_sequence          = 0;
    std::uint64_t          feedback_application_count = 0;
    std::array<double, 19> before{}, retired{}, admitted{}, boundary_source{}, after{}, arithmetic_scale{};
    std::vector<NestedPrivateRelocationInventory> private_scalars;
};

inline bool valid_nested_two_way_relocation_report(const NestedTwoWayRelocationReportV1& report) noexcept
{
    if (!report.complete || !report.movement_sequence || !report.feedback_application_count)
        return false;
    for (std::size_t f = 0; f < report.before.size(); ++f)
    {
        if (!std::isfinite(report.arithmetic_scale[f]) || report.arithmetic_scale[f] < 0)
            return false;
        const double scale = std::max(1.0, report.arithmetic_scale[f]);
        for (double value :
             {report.before[f], report.retired[f], report.admitted[f], report.boundary_source[f], report.after[f]})
        {
            if (!std::isfinite(value) || std::abs(value) > scale)
                return false;
        }
        if ((f < 9 &&
             (report.before[f] < 0 || report.retired[f] < 0 || report.admitted[f] < 0 || report.after[f] < 0)) ||
            std::abs(report.after[f] - report.before[f] + report.retired[f] - report.admitted[f] -
                     report.boundary_source[f]) > 128 * std::numeric_limits<double>::epsilon() * scale)
            return false;
    }
    if (report.private_scalars.size() > 16)
        return false;
    for (std::size_t f = 0; f < report.private_scalars.size(); ++f)
    {
        const auto& scalar = report.private_scalars[f];
        if (scalar.name.empty() || scalar.name.size() >= 64 || !std::isfinite(scalar.arithmetic_scale) ||
            scalar.arithmetic_scale < 0 || scalar.before < 0 || scalar.retired < 0 || scalar.admitted < 0 ||
            scalar.after < 0)
            return false;
        const double scale = std::max(1.0, scalar.arithmetic_scale);
        for (double value : {scalar.before, scalar.retired, scalar.admitted, scalar.boundary_source, scalar.after})
            if (!std::isfinite(value) || std::abs(value) > scale)
                return false;
        if (std::abs(scalar.after - scalar.before + scalar.retired - scalar.admitted - scalar.boundary_source) >
            128 * std::numeric_limits<double>::epsilon() * scale)
            return false;
        for (std::size_t previous = 0; previous < f; ++previous)
            if (report.private_scalars[previous].name == scalar.name)
                return false;
    }
    return true;
}

struct NestedTwoWayFeedbackReportV1
{
    static constexpr std::uint32_t schema_version = 12;

    std::vector<NestedPrivateScalarFeedbackReport> private_scalars;
    NestedTwoWayRelocationReportV1                 relocation;
    std::size_t                                    phase_projection_cells       = 0;
    double                                         phase_theta_inventory_change = 0;
    double                                         phase_thermal_roundoff_j     = 0;
    double                                         phase_thermal_scale_j        = 0;
    bool                                           complete                     = false;
    bool                                           moist                        = false;
    std::array<NestedWaterFeedbackReport, 7>       water{};
    std::size_t                                    restricted_scalar_cells = 0;
    std::size_t                                    restricted_x_faces      = 0;
    std::size_t                                    restricted_y_faces      = 0;
    std::size_t                                    restricted_z_faces      = 0;
    std::size_t                                    refluxed_scalar_cells   = 0;
    std::array<double, 5>                          interface_mismatch{};
    std::array<double, 5>                          applied_reflux{};
    std::array<double, 5>                          storage_residual{};
    std::array<double, 5>                          child_interface_source{};
    std::array<double, 5>                          child_dynamics_source{};
    std::array<double, 5>                          child_dynamics_storage{};
    std::array<double, 5>                    child_interface_transport{};
    std::array<double, 5>                    child_interval_change{};
    std::array<double, 5>                    child_composite_residual{};
    std::array<double, 5>                    child_composite_tolerance{};
};

inline bool valid_nested_phase_feedback_report(const NestedTwoWayFeedbackReportV1& report) noexcept
{
    double total = 0, scale = 0;
    for (const auto& water : report.water)
    {
        const double value = water.parent.phase_exchange_inventory;
        if (!std::isfinite(value))
            return false;
        total += value;
        scale += std::abs(value);
    }
    if (!std::isfinite(scale) || !std::isfinite(report.phase_theta_inventory_change) ||
        !std::isfinite(report.phase_thermal_roundoff_j) || !std::isfinite(report.phase_thermal_scale_j) ||
        report.phase_thermal_scale_j < 0)
        return false;
    if (report.phase_projection_cells == 0)
        return scale == 0 && report.phase_theta_inventory_change == 0 && report.phase_thermal_roundoff_j == 0 &&
               report.phase_thermal_scale_j == 0;
    return report.moist && scale > 0 && report.phase_thermal_scale_j > 0 &&
           std::abs(total) <= 128 * std::numeric_limits<double>::epsilon() * scale &&
           std::abs(report.phase_thermal_roundoff_j) <=
               32 * std::numeric_limits<float>::epsilon() * report.phase_thermal_scale_j;
}

std::array<double, 5> nested_two_way_child_inventory_flat(const tmv::domain::OneWayNestRelationshipV1& relationship,
                                                          const std::vector<double>&    layer_thickness_m,
                                                          const PrognosticFieldStorage& child);

std::array<double, 5> nested_two_way_child_inventory_mapped(const tmv::domain::NestedMappedTerrainGeometryV1& geometry,
                                                            const PrognosticFieldStorage&                     child);

void apply_nested_two_way_feedback_flat(const tmv::domain::OneWayNestRelationshipV1& relationship,
                                        const std::vector<double>&                   layer_thickness_m,
                                        const tmv::CartesianDryMassInterfaceFluxes&  parent_mass,
                                        const tmv::CartesianDryMassInterfaceFluxes&  child_mass,
                                        const tmv::CartesianThetaInterfaceFluxes&    parent_theta,
                                        const tmv::CartesianThetaInterfaceFluxes&    child_theta,
                                        const tmv::CartesianMomentumInterfaceFluxes& parent_advective_momentum,
                                        const tmv::CartesianMomentumInterfaceFluxes& parent_pressure_momentum,
                                        const tmv::CartesianMomentumInterfaceFluxes& child_advective_momentum,
                                        const tmv::CartesianMomentumInterfaceFluxes& child_pressure_momentum,
                                        const std::array<double, 5>&                 child_interface_source,
                                        const std::array<double, 5>&                 child_dynamics_source,
                                        const std::array<double, 5>&                 child_dynamics_storage,
                                        const std::array<double, 5>&                 child_interval_begin_inventory,
                                        PrognosticFieldStorage&                      parent,
                                        const PrognosticFieldStorage&                child,
                                        NestedTwoWayFeedbackReportV1&                report,
                                        const NestedMoistFeedbackInputs*             moist            = nullptr,
                                        std::array<Field3D, 7>*                      child_candidates = nullptr,
                                        ComputeBackendRuntimeState*                  compute          = nullptr);

void apply_nested_two_way_feedback_mapped(const tmv::domain::NestedMappedTerrainGeometryV1& geometry,
                                          const tmv::CartesianDryMassInterfaceFluxes&       parent_mass,
                                          const tmv::CartesianDryMassInterfaceFluxes&       child_mass,
                                          const tmv::CartesianThetaInterfaceFluxes&         parent_theta,
                                          const tmv::CartesianThetaInterfaceFluxes&         child_theta,
                                          const tmv::CartesianMomentumInterfaceFluxes&      parent_advective_momentum,
                                          const tmv::CartesianMomentumInterfaceFluxes&      parent_pressure_momentum,
                                          const tmv::CartesianMomentumInterfaceFluxes&      child_advective_momentum,
                                          const tmv::CartesianMomentumInterfaceFluxes&      child_pressure_momentum,
                                          const std::array<double, 5>&                      child_interface_source,
                                          const std::array<double, 5>&                      child_dynamics_source,
                                          const std::array<double, 5>&                      child_dynamics_storage,
                                          const std::array<double, 5>&     child_interval_begin_inventory,
                                          PrognosticFieldStorage&          parent,
                                          const PrognosticFieldStorage&    child,
                                          NestedTwoWayFeedbackReportV1&    report,
                                          const NestedMoistFeedbackInputs* moist            = nullptr,
                                          std::array<Field3D, 7>*          child_candidates = nullptr,
                                          ComputeBackendRuntimeState*      compute          = nullptr);
