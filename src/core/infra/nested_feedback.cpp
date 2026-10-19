#include "compute/native_conservative_reduction.hpp"
#include "core/field/field_bounds.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "core/infra/nested_feedback.hpp"

#include "core/runtime/simulation_state.hpp"
#include "dynamics/eos_pressure.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

[[noreturn]] void reject(const char* reason)
{
    throw std::runtime_error(std::string("nested two-way feedback: ") + reason);
}

int require_supported_refinement(const tmv::domain::OneWayNestRelationshipV1& relationship)
{
    tmv::domain::validate_one_way_nest_relationship(relationship);
    const int ratio = relationship.parent_ratio_x;
    if (relationship.version != 1 || ratio != relationship.parent_ratio_y || (ratio != 2 && ratio != 3) ||
        relationship.parent_time_step_ratio < ratio ||
        relationship.child_cells_x != ratio * relationship.child_parent_span_x ||
        relationship.child_cells_y != ratio * relationship.child_parent_span_y)
    {
        reject("the maintained operator requires isotropic 2:1 or 3:1 horizontal refinement and a sufficient time-step "
               "ratio");
    }
    return ratio;
}

bool shape_is(const Field3D& field, int nx, int ny, int nz)
{
    return field.size_r() == nx + 2 && field.size_th() == ny + 2 && field.size_z() == nz;
}

std::size_t index3(int i, int j, int k, int ny, int nz)
{
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(j)) *
               static_cast<std::size_t>(nz) +
           static_cast<std::size_t>(k);
}

void require_strip_shapes(const tmv::CartesianDryMassInterfaceFluxes& coarse,
                          const tmv::CartesianDryMassInterfaceFluxes& fine)
{
    if (coarse.west.size() != fine.west.size() || coarse.east.size() != fine.east.size() ||
        coarse.south.size() != fine.south.size() || coarse.north.size() != fine.north.size())
        reject("dry-mass coarse/fine interface strips do not match");
}

void require_strip_shapes(const tmv::CartesianThetaInterfaceFluxes& coarse,
                          const tmv::CartesianThetaInterfaceFluxes& fine)
{
    if (coarse.west.size() != fine.west.size() || coarse.east.size() != fine.east.size() ||
        coarse.south.size() != fine.south.size() || coarse.north.size() != fine.north.size())
        reject("theta coarse/fine interface strips do not match");
}

void require_strip_shapes(const tmv::CartesianMomentumInterfaceFluxes& coarse,
                          const tmv::CartesianMomentumInterfaceFluxes& fine)
{
    for (std::size_t component = 0; component < 3; ++component)
        if (coarse.west.component[component].size() != fine.west.component[component].size() ||
            coarse.east.component[component].size() != fine.east.component[component].size() ||
            coarse.south.component[component].size() != fine.south.component[component].size() ||
            coarse.north.component[component].size() != fine.north.component[component].size())
            reject("momentum coarse/fine interface strips do not match");
}

double duration_tolerance(double duration)
{
    return 64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(duration));
}

template <class Fluxes>
void require_duration(const Fluxes& fluxes, double expected, const char* owner)
{
    if (fluxes.contribution_count == 0 || !std::isfinite(fluxes.accumulated_dt_s) ||
        std::abs(fluxes.accumulated_dt_s - expected) > duration_tolerance(expected))
        reject(owner);
}

double horizontal_layer_volume(double dx, double dy, double dz)
{
    const double volume = dx * dy * dz;
    if (!std::isfinite(volume) || volume <= 0.0)
        reject("cell volume is invalid");
    return volume;
}

double parent_u_dual_mass(const Field3D& rho, int face, int j, int k, double volume)
{
    return 0.5 * volume * (static_cast<double>(rho[face][j][k]) + static_cast<double>(rho[face + 1][j][k]));
}

double parent_v_dual_mass(const Field3D& rho, int i, int face, int k, double volume)
{
    return 0.5 * volume * (static_cast<double>(rho[i][face][k]) + static_cast<double>(rho[i][face + 1][k]));
}

double parent_w_dual_mass(const Field3D& rho, int i, int j, int k, double area, const std::vector<double>& dz)
{
    double mass = 0.5 * area * dz[static_cast<std::size_t>(k)] * static_cast<double>(rho[i][j][k]);
    if (k + 1 < rho.size_z())
        mass += 0.5 * area * dz[static_cast<std::size_t>(k + 1)] * static_cast<double>(rho[i][j][k + 1]);
    return mass;
}

double side_value(const tmv::CartesianMomentumInterfaceFluxes& advective,
                  const tmv::CartesianMomentumInterfaceFluxes& pressure,
                  int                                          side,
                  std::size_t                                  component,
                  std::size_t                                  bin)
{
    const auto select =
        [side](const tmv::CartesianMomentumInterfaceFluxes& f) -> const tmv::CartesianMomentumInterfaceSide&
    {
        if (side == 0)
            return f.west;
        if (side == 1)
            return f.east;
        if (side == 2)
            return f.south;
        return f.north;
    };
    return select(advective).component[component][bin] + select(pressure).component[component][bin];
}

} // namespace

std::array<double, 5> nested_two_way_child_inventory_flat(const tmv::domain::OneWayNestRelationshipV1& relationship,
                                                          const std::vector<double>&    layer_thickness_m,
                                                          const PrognosticFieldStorage& child)
{
    require_supported_refinement(relationship);
    const Field3D* required[] = {&child.rho, &child.theta, &child.u, &child.v, &child.w};
    if (layer_thickness_m.size() != static_cast<std::size_t>(relationship.levels_z))
        reject("child inventory geometry is invalid");
    if (!std::isfinite(relationship.child_dx_m) || relationship.child_dx_m <= 0.0 ||
        !std::isfinite(relationship.child_dy_m) || relationship.child_dy_m <= 0.0)
        reject("child inventory horizontal metrics are invalid");
    for (double thickness : layer_thickness_m)
        if (!std::isfinite(thickness) || thickness <= 0.0)
            reject("child inventory vertical metrics are invalid");
    for (const Field3D* field : required)
        if (!shape_is(*field, relationship.child_cells_x, relationship.child_cells_y, relationship.levels_z))
            reject("child inventory field is incomplete or mis-shaped");
    std::array<double, 5> inventory{};
    const double          area = relationship.child_dx_m * relationship.child_dy_m;
    for (int i = 1; i <= relationship.child_cells_x; ++i)
        for (int j = 1; j <= relationship.child_cells_y; ++j)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const double volume = area * layer_thickness_m[static_cast<std::size_t>(k)];
                if (!std::isfinite(child.rho[i][j][k]) || child.rho[i][j][k] <= 0.0f ||
                    !std::isfinite(child.theta[i][j][k]) || child.theta[i][j][k] <= 0.0f)
                    reject("child inventory state is invalid");
                const double mass = child.rho[i][j][k] * volume;
                inventory[0] += mass;
                inventory[1] += mass * child.theta[i][j][k];
            }
    for (int i = 1; i <= relationship.child_cells_x; ++i)
        for (int j = 1; j <= relationship.child_cells_y; ++j)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                if (!std::isfinite(child.w[i][j][k]))
                    reject("child inventory z momentum is invalid");
                inventory[4] += parent_w_dual_mass(child.rho, i, j, k, area, layer_thickness_m) * child.w[i][j][k];
            }
    for (int face = 0; face <= relationship.child_cells_x; ++face)
        for (int j = 1; j <= relationship.child_cells_y; ++j)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                if (!std::isfinite(child.u[face][j][k]))
                    reject("child inventory x momentum is invalid");
                const double volume = area * layer_thickness_m[static_cast<std::size_t>(k)];
                inventory[2] += parent_u_dual_mass(child.rho, face, j, k, volume) * child.u[face][j][k];
            }
    for (int i = 1; i <= relationship.child_cells_x; ++i)
        for (int face = 0; face <= relationship.child_cells_y; ++face)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                if (!std::isfinite(child.v[i][face][k]))
                    reject("child inventory y momentum is invalid");
                const double volume = area * layer_thickness_m[static_cast<std::size_t>(k)];
                inventory[3] += parent_v_dual_mass(child.rho, i, face, k, volume) * child.v[i][face][k];
            }
    for (double value : inventory)
        if (!std::isfinite(value))
            reject("child inventory is non-finite");
    return inventory;
}

std::array<double, 5> nested_two_way_child_inventory_mapped(const tmv::domain::NestedMappedTerrainGeometryV1& geometry,
                                                            const PrognosticFieldStorage&                     child)
{
    const auto& relationship = geometry.relationship;
    require_supported_refinement(relationship);
    const auto& metrics = geometry.child_storage.metrics;
    if (metrics.J.size_r() != relationship.child_cells_x + 2 || metrics.J.size_th() != relationship.child_cells_y + 2 ||
        metrics.J.size_z() != relationship.levels_z)
        reject("mapped child inventory geometry is invalid");
    for (const Field3D* field : {&child.rho, &child.theta, &child.u, &child.v, &child.w})
        if (!shape_is(*field, relationship.child_cells_x, relationship.child_cells_y, relationship.levels_z))
            reject("mapped child inventory field is incomplete or mis-shaped");
    const auto scalar_mass = [&](int i, int j, int k)
    {
        const double value = child.rho[i][j][k] * metrics.J(i, j, k) * metrics.dx * metrics.dy * metrics.dzeta;
        if (!std::isfinite(value) || value <= 0.0)
            reject("mapped child scalar mass is invalid");
        return value;
    };
    std::array<double, 5> inventory{};
    for (int i = 1; i <= relationship.child_cells_x; ++i)
        for (int j = 1; j <= relationship.child_cells_y; ++j)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                if (!std::isfinite(child.theta[i][j][k]) || child.theta[i][j][k] <= 0.0f)
                    reject("mapped child theta inventory is invalid");
                const double mass = scalar_mass(i, j, k);
                inventory[0] += mass;
                inventory[1] += mass * child.theta[i][j][k];
            }
    for (int face = 0; face <= relationship.child_cells_x; ++face)
        for (int j = 1; j <= relationship.child_cells_y; ++j)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                if (!std::isfinite(child.u[face][j][k]))
                    reject("mapped child x momentum is invalid");
                const double mass = 0.5 * (scalar_mass(face, j, k) + scalar_mass(face + 1, j, k));
                inventory[2] += mass * child.u[face][j][k];
            }
    for (int i = 1; i <= relationship.child_cells_x; ++i)
        for (int face = 0; face <= relationship.child_cells_y; ++face)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                if (!std::isfinite(child.v[i][face][k]))
                    reject("mapped child y momentum is invalid");
                const double mass = 0.5 * (scalar_mass(i, face, k) + scalar_mass(i, face + 1, k));
                inventory[3] += mass * child.v[i][face][k];
            }
    for (int i = 1; i <= relationship.child_cells_x; ++i)
        for (int j = 1; j <= relationship.child_cells_y; ++j)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                if (!std::isfinite(child.w[i][j][k]))
                    reject("mapped child z momentum is invalid");
                double mass = 0.5 * scalar_mass(i, j, k);
                if (k + 1 < relationship.levels_z)
                    mass += 0.5 * scalar_mass(i, j, k + 1);
                inventory[4] += mass * child.w[i][j][k];
            }
    for (double value : inventory)
        if (!std::isfinite(value))
            reject("mapped child inventory is non-finite");
    return inventory;
}

namespace
{
template <class Volume>
void project_water_phases(PrognosticFieldStorage&       candidate,
                          NestedTwoWayFeedbackReportV1& report,
                          const Volume&                 volume,
                          ComputeBackendRuntimeState*   compute)
{
    namespace constants    = physical_constants;
    constexpr double cv_d  = constants::specific_heat_cp_jkgk - constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v  = constants::specific_heat_water_vapor_cp_jkgk - constants::gas_constant_water_vapor_jkgk;
    constexpr double tr    = constants::freezing_temperature_k;
    constexpr double kappa = constants::gas_constant_dry_air_jkgk / constants::specific_heat_cp_jkgk;
    const std::array<double, 7>   heat{cv_v,
                                       constants::specific_heat_liquid_water_jkgk,
                                       constants::specific_heat_liquid_water_jkgk,
                                       constants::specific_heat_ice_jkgk,
                                       constants::specific_heat_ice_jkgk,
                                       constants::specific_heat_ice_jkgk,
                                       constants::specific_heat_ice_jkgk};
    const std::array<double, 7>   reference{constants::latent_heat_vaporization_jkg -
                                                constants::gas_constant_water_vapor_jkgk * tr,
                                            0,
                                            0,
                                            -constants::latent_heat_fusion_jkg,
                                            -constants::latent_heat_fusion_jkg,
                                            -constants::latent_heat_fusion_jkg,
                                            -constants::latent_heat_fusion_jkg};
    const std::array<Field3D*, 7> fields{
        &candidate.qv, &candidate.qc, &candidate.qr, &candidate.qi, &candidate.qs, &candidate.qg, &candidate.qh};
    struct Cell
    {
        int                   i, j, k;
        std::array<double, 7> before, exchange;
    };
    std::vector<Cell> pending;
    pending.reserve(256);
    nimbus::compute::NativeConservativeReduction    native(compute);
    std::array<tmv::diagnostics::CompensatedSum, 7> exchanges;
    tmv::diagnostics::CompensatedSum                theta_change, thermal_roundoff, thermal_scale;
    const auto                                      flush = [&]()
    {
        if (native.enabled())
        {
            native.clear();
            for (const auto& cell : pending)
                for (std::size_t f = 0; f < 7; ++f)
                {
                    native.begin_row();
                    native.term(1, cell.before[f], 1);
                    native.term(1, cell.exchange[f], 1);
                    native.denominator();
                    native.term(1, 1, 1);
                    native.end_row();
                }
            native.dispatch();
        }
        std::size_t row = 0;
        for (const auto& cell : pending)
        {
            const auto [i, j, k, before, exchange] = cell;
            const double dry_mass                  = candidate.rho(i, j, k) * volume(i, j, k);
            const float  old_theta                 = candidate.theta(i, j, k);
            const double temperature =
                double(old_theta) * std::pow(candidate.p(i, j, k) / constants::reference_pressure_pa, kappa);
            double cv_before = cv_d, reference_before = 0, cv_change = 0, reference_change = 0;
            for (std::size_t f = 0; f < 7; ++f)
            {
                cv_before += before[f] * heat[f];
                reference_before += before[f] * reference[f];
                cv_change += exchange[f] * heat[f];
                reference_change += exchange[f] * reference[f];
                const float stored  = native.enabled() ? native.stored(row++) : float(before[f] + exchange[f]);
                const float maximum = f == 0 ? qv_max_kgkg : hydrometeor_max_kgkg;
                if (!std::isfinite(stored) || stored < 0 || stored > maximum)
                    reject("coupled water positivity is not representable");
                (*fields[f])(i, j, k) = stored;
                exchanges[f].add(dry_mass * exchange[f]);
            }
            const double cv_after           = cv_before + cv_change;
            const double temperature_change = -((temperature - tr) * cv_change + reference_change) / cv_after;
            const double target_temperature = temperature + temperature_change;
            if (!std::isfinite(cv_before) || cv_before <= 0 || !std::isfinite(cv_after) || cv_after <= 0 ||
                !std::isfinite(target_temperature) || target_temperature <= 0)
                reject("coupled water positivity has no admissible thermal state");
            const float theta =
                float(double(old_theta) +
                      double(old_theta) * std::expm1((1 - kappa) * std::log1p(temperature_change / temperature)));
            if (!std::isfinite(theta) || theta < theta_min_k || theta > theta_max_k)
                reject("coupled water positivity exceeds the potential-temperature bound");
            // Sub-ULP exchanges must not repeatedly re-project an unchanged stored thermodynamic pair.
            const float pressure = theta == old_theta
                                       ? candidate.p(i, j, k)
                                       : float(tmv::dynamics_eos::pressure_fp64(candidate.rho(i, j, k), theta));
            if (!std::isfinite(pressure) || pressure <= 0)
                reject("coupled water positivity pressure is not representable");
            const double stored_temperature =
                double(theta) * std::pow(pressure / constants::reference_pressure_pa, kappa);
            double stored_cv_change = 0, stored_reference_change = 0;
            for (std::size_t f = 0; f < 7; ++f)
            {
                const double change = double((*fields[f])(i, j, k)) - before[f];
                stored_cv_change += change * heat[f];
                stored_reference_change += change * reference[f];
            }
            const double roundoff = dry_mass * ((cv_before + stored_cv_change) * (stored_temperature - temperature) +
                                                (temperature - tr) * stored_cv_change + stored_reference_change);
            const double scale    = dry_mass * (std::abs(cv_before * temperature) + std::abs(reference_before));
            if (!std::isfinite(roundoff) || !std::isfinite(scale) ||
                std::abs(roundoff) > 32 * std::numeric_limits<float>::epsilon() * scale)
                reject("coupled water positivity thermal storage does not close");
            candidate.theta(i, j, k) = theta;
            candidate.p(i, j, k)     = pressure;
            theta_change.add(dry_mass * (double(theta) - old_theta));
            thermal_roundoff.add(roundoff);
            thermal_scale.add(scale);
            ++report.phase_projection_cells;
        }
        pending.clear();
    };
    for (int i = 1; i < candidate.rho.size_r() - 1; ++i)
        for (int j = 1; j < candidate.rho.size_th() - 1; ++j)
            for (int k = 0; k < candidate.rho.size_z(); ++k)
            {
                Cell        cell{i, j, k, {}, {}};
                long double available = 0, deficit = 0;
                for (std::size_t f = 0; f < 7; ++f)
                {
                    const double value = (*fields[f])(i, j, k);
                    cell.before[f]     = value;
                    if (value < 0)
                        deficit -= value;
                    else
                        available += value;
                }
                if (deficit == 0)
                    continue;
                if (!std::isfinite(available) || available < deficit)
                    reject("coupled water positivity exhausts total cell water");
                const long double fraction = deficit / available;
                for (std::size_t f = 0; f < 7; ++f)
                    cell.exchange[f] = cell.before[f] < 0 ? -cell.before[f] : -double(cell.before[f] * fraction);
                pending.push_back(cell);
                if (pending.size() == 256)
                    flush();
            }
    flush();
    if (report.phase_projection_cells == 0)
        return;
    report.phase_theta_inventory_change = theta_change.value();
    report.phase_thermal_roundoff_j     = thermal_roundoff.value();
    report.phase_thermal_scale_j        = thermal_scale.value();
    std::array<tmv::diagnostics::CompensatedSum, 7> inventories;
    for (int i = 1; i < candidate.rho.size_r() - 1; ++i)
        for (int j = 1; j < candidate.rho.size_th() - 1; ++j)
            for (int k = 0; k < candidate.rho.size_z(); ++k)
            {
                const double dry_mass = candidate.rho(i, j, k) * volume(i, j, k);
                for (std::size_t f = 0; f < 7; ++f)
                    inventories[f].add(dry_mass * (*fields[f])(i, j, k));
            }
    for (std::size_t f = 0; f < 7; ++f)
    {
        auto&        p             = report.water[f].parent;
        const double after         = inventories[f].value();
        p.phase_exchange_inventory = exchanges[f].value();
        p.storage_rounding += (after - p.final_inventory) - p.phase_exchange_inventory;
        p.final_inventory  = after;
        p.closure_residual = p.final_inventory - p.initial_inventory - p.restriction_change - p.applied_reflux -
                             p.phase_exchange_inventory - p.storage_rounding;
        if (!valid_nested_water_feedback_report(report.water[f]))
            reject("coupled phase feedback inventory does not close");
    }
    if (!valid_nested_phase_feedback_report(report))
        reject("coupled phase feedback does not conserve total water and thermal energy");
}

template <class Prepare, class Volume>
void prepare_water_feedback(const PrognosticFieldStorage&    parent,
                            const PrognosticFieldStorage&    child,
                            PrognosticFieldStorage&          candidate,
                            NestedTwoWayFeedbackReportV1&    report,
                            const NestedMoistFeedbackInputs* inputs,
                            const Prepare&                   prepare,
                            std::array<Field3D, 7>&          child_candidates,
                            const Volume&                    volume,
                            ComputeBackendRuntimeState*      compute)
{
    if (!inputs)
        return;
    const std::array<const Field3D*, 7> old{
        &parent.qv, &parent.qc, &parent.qr, &parent.qi, &parent.qs, &parent.qg, &parent.qh};
    const std::array<const Field3D*, 7> fine{
        &child.qv, &child.qc, &child.qr, &child.qi, &child.qs, &child.qg, &child.qh};
    const std::array<Field3D*, 7> target{
        &candidate.qv, &candidate.qc, &candidate.qr, &candidate.qi, &candidate.qs, &candidate.qg, &candidate.qh};
    for (std::size_t f = 0; f < old.size(); ++f)
    {
        if (!inputs->parent_flux[f] || !inputs->child_flux[f])
            reject("missing nested water interface owner");
        for (double value : {inputs->child_begin[f],
                             inputs->interface_source[f],
                             inputs->physics_source[f],
                             inputs->transport_storage[f]})
            if (!std::isfinite(value))
                reject("invalid nested water source ledger");
        PreparedNestedScalarFeedback prepared;
        try
        {
            try
            {
                prepared = prepare(*old[f], *fine[f], *inputs->parent_flux[f], *inputs->child_flux[f], true);
            }
            catch (const NestedScalarInventoryExhausted&)
            {
                prepared = prepare(*old[f], *fine[f], *inputs->parent_flux[f], *inputs->child_flux[f], false);
            }
        }
        catch (const std::exception& error)
        {
            static constexpr const char* names[]{"qv", "qc", "qr", "qi", "qs", "qg", "qh"};
            throw std::runtime_error(std::string("nested water ") + names[f] + ": " + error.what());
        }
        auto& result                          = report.water[f];
        result.parent                         = prepared.report;
        result.child_begin                    = inputs->child_begin[f];
        result.interface_source               = inputs->interface_source[f];
        result.physics_source                 = inputs->physics_source[f];
        result.transport_storage              = inputs->transport_storage[f];
        const auto&                      flux = *inputs->child_flux[f];
        tmv::diagnostics::CompensatedSum transport;
        for (std::size_t n = 0; n < flux.west.size(); ++n)
            transport.add(flux.west[n] - flux.east[n]);
        for (std::size_t n = 0; n < flux.south.size(); ++n)
            transport.add(flux.south[n] - flux.north[n]);
        result.child_transport  = transport.value();
        result.child_change     = result.parent.child_inventory - result.child_begin;
        result.closure_residual = result.child_change - result.child_transport - result.interface_source -
                                  result.physics_source - result.transport_storage;
        const double scale      = std::max({std::abs(result.child_begin),
                                            std::abs(result.parent.child_inventory),
                                            std::abs(result.child_transport),
                                            std::abs(result.interface_source),
                                            std::abs(result.physics_source),
                                            std::abs(result.transport_storage)});
        result.tolerance        = 512 * std::numeric_limits<float>::epsilon() * scale;
        if (!std::isfinite(result.closure_residual) || std::abs(result.closure_residual) > result.tolerance)
        {
            std::ostringstream message;
            message << std::setprecision(17) << "child water interval does not close: field=" << f
                    << " before=" << result.child_begin << " after=" << result.parent.child_inventory
                    << " transport=" << result.child_transport << " interface=" << result.interface_source
                    << " physics=" << result.physics_source << " storage=" << result.transport_storage
                    << " residual=" << result.closure_residual << " tolerance=" << result.tolerance;
            reject(message.str().c_str());
        }
        const float maximum = f == 0 ? qv_max_kgkg : hydrometeor_max_kgkg;
        for (std::size_t n = 0; n < prepared.scalar.size(); ++n)
            if (prepared.scalar.data()[n] > maximum)
                reject("nested water feedback exceeds the physical field bound");
        *target[f]          = std::move(prepared.scalar);
        child_candidates[f] = std::move(prepared.child_scalar);
    }
    report.moist = true;
    project_water_phases(candidate, report, volume, compute);
}
void publish_water(PrognosticFieldStorage& parent, PrognosticFieldStorage& candidate) noexcept
{
    using std::swap;
    swap(parent.qv, candidate.qv);
    swap(parent.qc, candidate.qc);
    swap(parent.qr, candidate.qr);
    swap(parent.qi, candidate.qi);
    swap(parent.qs, candidate.qs);
    swap(parent.qg, candidate.qg);
    swap(parent.qh, candidate.qh);
}
} // namespace

static void apply_nested_two_way_feedback(const tmv::domain::OneWayNestRelationshipV1& relationship,
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
                                          const NestedMoistFeedbackInputs*             moist,
                                          const tmv::domain::NestedMappedTerrainGeometryV1* mapped,
                                          std::array<Field3D, 7>*                           child_candidates,
                                          ComputeBackendRuntimeState*                       compute)
{
    const int ratio = require_supported_refinement(relationship);
    if (!mapped && layer_thickness_m.size() != static_cast<std::size_t>(relationship.levels_z))
        reject("vertical metrics do not match the relationship");
    for (double value : layer_thickness_m)
        if (!std::isfinite(value) || value <= 0.0)
            reject("vertical metrics must be finite and positive");

    if (mapped)
    {
        const auto& p = mapped->parent.metrics;
        const auto& c = mapped->child_storage.metrics;
        if (!mapped->generation || p.J.size_r() != relationship.parent_cells_x ||
            p.J.size_th() != relationship.parent_cells_y || p.J.size_z() != relationship.levels_z ||
            c.J.size_r() != relationship.child_cells_x + 2 || c.J.size_th() != relationship.child_cells_y + 2 ||
            c.J.size_z() != relationship.levels_z || p.dx != relationship.parent_dx_m ||
            p.dy != relationship.parent_dy_m || c.dx != relationship.child_dx_m || c.dy != relationship.child_dy_m)
            reject("mapped feedback geometry is incomplete or stale");
        for (const auto* metrics : {&p, &c})
        {
            if (!std::isfinite(metrics->dzeta) || metrics->dzeta <= 0)
                reject("mapped feedback vertical geometry is invalid");
            for (double value : metrics->J.values)
                if (!std::isfinite(value) || value <= 0)
                    reject("mapped feedback cell Jacobian is invalid");
        }
    }
    const auto cell_volume = [&](bool fine, int i, int j, int k)
    {
        if (mapped)
        {
            const auto& metrics = fine ? mapped->child_storage.metrics : mapped->parent.metrics;
            return metrics.dx * metrics.dy * metrics.dzeta * metrics.J(fine ? i : i - 1, fine ? j : j - 1, k);
        }
        return horizontal_layer_volume(fine ? relationship.child_dx_m : relationship.parent_dx_m,
                                       fine ? relationship.child_dy_m : relationship.parent_dy_m,
                                       layer_thickness_m[static_cast<std::size_t>(k)]);
    };
    const auto dual_mass = [&](bool fine, const Field3D& rho, std::size_t component, int i, int j, int k)
    {
        const double first = 0.5 * rho(i, j, k) * cell_volume(fine, i, j, k);
        if (component == 2 && k + 1 == relationship.levels_z)
            return first;
        const int ni = i + (component == 0), nj = j + (component == 1), nk = k + (component == 2);
        return first + 0.5 * rho(ni, nj, nk) * cell_volume(fine, ni, nj, nk);
    };

    const Field3D* parent_fields[] = {&parent.rho, &parent.theta, &parent.p, &parent.u, &parent.v, &parent.w};
    const Field3D* child_fields[]  = {&child.rho, &child.theta, &child.p, &child.u, &child.v, &child.w};
    for (const Field3D* field : parent_fields)
        if (!shape_is(*field, relationship.parent_cells_x, relationship.parent_cells_y, relationship.levels_z))
            reject("parent primary field shape does not match the relationship");
    for (const Field3D* field : child_fields)
        if (!shape_is(*field, relationship.child_cells_x, relationship.child_cells_y, relationship.levels_z))
            reject("child primary field shape does not match the relationship");
    const auto require_finite = [&](const PrognosticFieldStorage& fields, int nx, int ny)
    {
        const Field3D* all[] = {&fields.rho,
                                &fields.p,
                                &fields.u,
                                &fields.v,
                                &fields.w,
                                &fields.theta,
                                &fields.qv,
                                &fields.qc,
                                &fields.qr,
                                &fields.qi,
                                &fields.qs,
                                &fields.qg,
                                &fields.qh,
                                &fields.tracer,
                                &fields.tke};
        for (const Field3D* field : all)
        {
            if (!shape_is(*field, nx, ny, relationship.levels_z))
                reject("primary state is incomplete or mis-shaped");
            for (std::size_t index = 0; index < field->size(); ++index)
                if (!std::isfinite(static_cast<double>(field->data()[index])))
                    reject("primary state contains a non-finite value");
        }
        for (const Field3D* field : {&fields.rho, &fields.p, &fields.theta})
            for (std::size_t index = 0; index < field->size(); ++index)
                if (field->data()[index] <= 0.0f)
                    reject("thermodynamic primary state is not positive");
        const Field3D* unowned[] = {&fields.qv,
                                    &fields.qc,
                                    &fields.qr,
                                    &fields.qi,
                                    &fields.qs,
                                    &fields.qg,
                                    &fields.qh,
                                    &fields.tracer,
                                    &fields.tke};
        const char*    names[]   = {"qv", "qc", "qr", "qi", "qs", "qg", "qh", "tracer", "tke"};
        for (std::size_t field_index = 0; field_index < std::size(unowned); ++field_index)
            for (std::size_t index = 0; index < unowned[field_index]->size(); ++index)
                if (moist && field_index < 7)
                {
                    if (unowned[field_index]->data()[index] < 0)
                        reject("negative nested water reservoir");
                }
                else if (unowned[field_index]->data()[index] != 0.0f)
                    throw std::runtime_error(std::string("nested two-way feedback: the dry operator rejects active ") +
                                             names[field_index] + " carrier");
    };
    require_finite(parent, relationship.parent_cells_x, relationship.parent_cells_y);
    require_finite(child, relationship.child_cells_x, relationship.child_cells_y);
    require_strip_shapes(parent_mass, child_mass);
    require_strip_shapes(parent_theta, child_theta);
    require_strip_shapes(parent_advective_momentum, child_advective_momentum);
    require_strip_shapes(parent_pressure_momentum, child_pressure_momentum);
    const std::size_t scalar_x_count =
        static_cast<std::size_t>(relationship.child_parent_span_y) * static_cast<std::size_t>(relationship.levels_z);
    const std::size_t scalar_y_count =
        static_cast<std::size_t>(relationship.child_parent_span_x) * static_cast<std::size_t>(relationship.levels_z);
    if (parent_mass.west.size() != scalar_x_count || parent_mass.east.size() != scalar_x_count ||
        parent_mass.south.size() != scalar_y_count || parent_mass.north.size() != scalar_y_count ||
        parent_theta.west.size() != scalar_x_count || parent_theta.east.size() != scalar_x_count ||
        parent_theta.south.size() != scalar_y_count || parent_theta.north.size() != scalar_y_count)
        reject("scalar interface strips have the wrong exact extent");
    const std::size_t momentum_x_count = static_cast<std::size_t>(relationship.child_parent_span_y) *
                                         static_cast<std::size_t>(relationship.levels_z - 1);
    const std::size_t momentum_y_count = static_cast<std::size_t>(relationship.child_parent_span_x) *
                                         static_cast<std::size_t>(relationship.levels_z - 1);
    for (std::size_t component = 0; component < 3; ++component)
        for (const auto* fluxes : {&parent_advective_momentum, &parent_pressure_momentum})
            if (fluxes->west.component[component].size() != momentum_x_count ||
                fluxes->east.component[component].size() != momentum_x_count ||
                fluxes->south.component[component].size() != momentum_y_count ||
                fluxes->north.component[component].size() != momentum_y_count)
                reject("momentum interface strips have the wrong exact extent");
    const double interval_s = parent_mass.accumulated_dt_s;
    if (!std::isfinite(interval_s) || interval_s <= 0.0)
        reject("parent dry-mass register has no completed interval");
    require_duration(parent_mass, interval_s, "parent dry-mass register is incomplete");
    require_duration(child_mass, interval_s, "child dry-mass register is incomplete");
    require_duration(parent_theta, interval_s, "parent theta register is incomplete");
    require_duration(child_theta, interval_s, "child theta register is incomplete");
    require_duration(parent_advective_momentum, interval_s, "parent advective-momentum register is incomplete");
    require_duration(child_advective_momentum, interval_s, "child advective-momentum register is incomplete");
    require_duration(parent_pressure_momentum, interval_s, "parent pressure-momentum register is incomplete");
    require_duration(child_pressure_momentum, interval_s, "child pressure-momentum register is incomplete");

    nimbus::compute::NativeConservativeReduction native(compute);
    PrognosticFieldStorage                       candidate   = parent;
    const int                                    parent_nx   = parent.rho.size_r();
    const int                                    parent_ny   = parent.rho.size_th();
    const int                                    nz          = relationship.levels_z;
    const int                                    oi          = relationship.child_origin_parent_cell_i;
    const int                                    oj          = relationship.child_origin_parent_cell_j;
    const int                                    sx          = relationship.child_parent_span_x;
    const int                                    sy          = relationship.child_parent_span_y;
    const int                                    ci0         = 1;
    const int                                    cj0         = 1;
    const std::size_t                            parent_size = parent.rho.size();
    std::vector<double>                          mass_delta(parent_size, 0.0);
    std::vector<double>                          theta_delta(parent_size, 0.0);
    const std::size_t                            levels = static_cast<std::size_t>(nz);

    const auto add_scalar_reflux = [&](int i, int j, int k, double dry, double theta)
    {
        const std::size_t index = index3(i, j, k, parent_ny, nz);
        mass_delta[index] += dry;
        theta_delta[index] += theta;
    };
    for (int local_j = 0; local_j < sy; ++local_j)
        for (int k = 0; k < nz; ++k)
        {
            const std::size_t bin = static_cast<std::size_t>(local_j) * levels + static_cast<std::size_t>(k);
            add_scalar_reflux(oi,
                              oj + local_j + 1,
                              k,
                              parent_mass.west[bin] - child_mass.west[bin],
                              parent_theta.west[bin] - child_theta.west[bin]);
            add_scalar_reflux(oi + sx + 1,
                              oj + local_j + 1,
                              k,
                              child_mass.east[bin] - parent_mass.east[bin],
                              child_theta.east[bin] - parent_theta.east[bin]);
        }
    for (int local_i = 0; local_i < sx; ++local_i)
        for (int k = 0; k < nz; ++k)
        {
            const std::size_t bin = static_cast<std::size_t>(local_i) * levels + static_cast<std::size_t>(k);
            add_scalar_reflux(oi + local_i + 1,
                              oj,
                              k,
                              parent_mass.south[bin] - child_mass.south[bin],
                              parent_theta.south[bin] - child_theta.south[bin]);
            add_scalar_reflux(oi + local_i + 1,
                              oj + sy + 1,
                              k,
                              child_mass.north[bin] - parent_mass.north[bin],
                              child_theta.north[bin] - parent_theta.north[bin]);
        }

    NestedTwoWayFeedbackReportV1                    candidate_report;
    std::array<tmv::diagnostics::CompensatedSum, 5> applied_reflux_sum;
    for (std::size_t component = 0; component < child_interface_source.size(); ++component)
    {
        if (!std::isfinite(child_interface_source[component]))
            reject("child interface source ledger is invalid");
        candidate_report.child_interface_source[component] = child_interface_source[component];
        if (!std::isfinite(child_dynamics_source[component]) ||
            !std::isfinite(child_interval_begin_inventory[component]))
            reject("child interval inventory provenance is invalid");
        candidate_report.child_dynamics_source[component] = child_dynamics_source[component];
        if (!std::isfinite(child_dynamics_storage[component]))
            reject("child dynamics storage ledger is invalid");
        candidate_report.child_dynamics_storage[component] = child_dynamics_storage[component];
    }
    if (native.enabled())
    {
        for (int li = 0; li < sx; ++li)
            for (int lj = 0; lj < sy; ++lj)
                for (int k = 0; k < nz; ++k)
                    for (int field = 0; field < 2; ++field)
                    {
                        native.begin_row();
                        for (int fi = 0; fi < ratio; ++fi)
                            for (int fj = 0; fj < ratio; ++fj)
                            {
                                const int i = 1 + ratio * li + fi, j = 1 + ratio * lj + fj;
                                native.term(child.rho(i, j, k),
                                            field == 0 ? 1 : child.theta(i, j, k),
                                            cell_volume(true, i, j, k));
                            }
                        native.denominator();
                        if (field == 0)
                            native.term(1, 1, cell_volume(false, oi + li + 1, oj + lj + 1, k));
                        else
                            for (int fi = 0; fi < ratio; ++fi)
                                for (int fj = 0; fj < ratio; ++fj)
                                {
                                    const int i = 1 + ratio * li + fi, j = 1 + ratio * lj + fj;
                                    native.term(child.rho(i, j, k), 1, cell_volume(true, i, j, k));
                                }
                        native.end_row();
                    }
        native.dispatch();
    }
    std::size_t native_row = 0;
    for (int local_i = 0; local_i < sx; ++local_i)
        for (int local_j = 0; local_j < sy; ++local_j)
            for (int k = 0; k < nz; ++k)
            {
                double dry_mass        = 0.0;
                double theta_inventory = 0.0;
                if (!native.enabled())
                {
                    for (int fi = 0; fi < ratio; ++fi)
                        for (int fj = 0; fj < ratio; ++fj)
                        {
                            const int    i            = ci0 + ratio * local_i + fi;
                            const int    j            = cj0 + ratio * local_j + fj;
                            const double rho          = child.rho[i][j][k];
                            const double child_volume = cell_volume(true, i, j, k);
                            dry_mass += rho * child_volume;
                            theta_inventory += rho * static_cast<double>(child.theta[i][j][k]) * child_volume;
                        }
                }
                const int    pi            = oi + local_i + 1;
                const int    pj            = oj + local_j + 1;
                const double parent_volume = cell_volume(false, pi, pj, k);
                const float  stored_rho =
                    native.enabled() ? native.stored(native_row++) : static_cast<float>(dry_mass / parent_volume);
                const float stored_theta =
                    native.enabled() ? native.stored(native_row++) : static_cast<float>(theta_inventory / dry_mass);
                if (!std::isfinite(stored_rho) || stored_rho <= 0.0f || !std::isfinite(stored_theta) ||
                    stored_theta <= 0.0f)
                    reject("restricted scalar state is not representable");
                candidate.rho[pi][pj][k]   = stored_rho;
                candidate.theta[pi][pj][k] = stored_theta;
                ++candidate_report.restricted_scalar_cells;
            }

    if (native.enabled())
    {
        native.clear();
        for (int i = 0; i < parent_nx; ++i)
            for (int j = 0; j < parent_ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const auto index = index3(i, j, k, parent_ny, nz);
                    if (mass_delta[index] == 0 && theta_delta[index] == 0)
                        continue;
                    const double volume = cell_volume(false, i, j, k);
                    for (int field = 0; field < 2; ++field)
                    {
                        native.begin_row();
                        native.term(candidate.rho(i, j, k), field == 0 ? 1 : candidate.theta(i, j, k), volume);
                        native.term(1, field == 0 ? mass_delta[index] : theta_delta[index], 1);
                        native.denominator();
                        if (field == 0)
                            native.term(1, 1, volume);
                        else
                        {
                            native.term(candidate.rho(i, j, k), 1, volume);
                            native.term(1, mass_delta[index], 1);
                        }
                        native.end_row();
                    }
                }
        native.dispatch();
        native_row = 0;
    }
    for (int i = 0; i < parent_nx; ++i)
        for (int j = 0; j < parent_ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t index = index3(i, j, k, parent_ny, nz);
                if (mass_delta[index] == 0.0 && theta_delta[index] == 0.0)
                    continue;
                const double volume       = cell_volume(false, i, j, k);
                const double old_mass     = static_cast<double>(candidate.rho[i][j][k]) * volume;
                const double old_theta    = old_mass * candidate.theta[i][j][k];
                const double target_mass  = old_mass + mass_delta[index];
                const double target_theta = old_theta + theta_delta[index];
                if (!std::isfinite(target_mass) || target_mass <= 0.0 || !std::isfinite(target_theta))
                    reject("scalar reflux produces an invalid inventory");
                const float stored_rho =
                    native.enabled() ? native.stored(native_row++) : static_cast<float>(target_mass / volume);
                const float stored_theta =
                    native.enabled() ? native.stored(native_row++) : static_cast<float>(target_theta / target_mass);
                if (!std::isfinite(stored_rho) || stored_rho <= 0.0f || !std::isfinite(stored_theta) ||
                    stored_theta <= 0.0f)
                    reject("scalar reflux is not representable");
                candidate.rho[i][j][k]   = stored_rho;
                candidate.theta[i][j][k] = stored_theta;
                applied_reflux_sum[0].add(mass_delta[index]);
                applied_reflux_sum[1].add(theta_delta[index]);
                candidate_report.storage_residual[0] += target_mass - static_cast<double>(stored_rho) * volume;
                candidate_report.storage_residual[1] +=
                    target_theta - static_cast<double>(stored_rho) * static_cast<double>(stored_theta) * volume;
                ++candidate_report.refluxed_scalar_cells;
            }

    std::array<std::vector<double>, 3>        momentum_target;
    std::array<std::vector<double>, 3>        momentum_delta;
    std::array<std::vector<unsigned char>, 3> momentum_owned;
    for (std::size_t component = 0; component < 3; ++component)
    {
        momentum_target[component].assign(parent_size, 0.0);
        momentum_delta[component].assign(parent_size, 0.0);
        momentum_owned[component].assign(parent_size, 0);
    }

    const auto child_u_momentum = [&](int parent_face, int local_j, int k)
    {
        const double support_left  = std::max(0.0, parent_face - 0.5);
        const double support_right = std::min(static_cast<double>(sx), parent_face + 0.5);
        double       momentum      = 0.0;
        for (int f = std::max(0, static_cast<int>(std::floor(support_left * ratio)));
             f <= std::min(relationship.child_cells_x, static_cast<int>(std::ceil(support_right * ratio)));
             ++f)
            for (int side = 0; side < 2; ++side)
            {
                const int cell = f - 1 + side;
                if (cell < 0 || cell >= relationship.child_cells_x)
                    continue;
                const double half_child_width = 0.5 / ratio;
                const double left =
                    side == 0 ? static_cast<double>(f) / ratio - half_child_width : static_cast<double>(f) / ratio;
                const double right   = left + half_child_width;
                const double overlap = std::max(0.0, std::min(support_right, right) - std::max(support_left, left));
                if (overlap == 0.0)
                    continue;
                for (int fj = 0; fj < ratio; ++fj)
                {
                    const int j = cj0 + ratio * local_j + fj;
                    if (native.enabled())
                        native.term(child.rho(cell + 1, j, k),
                                    child.u(f, j, k),
                                    overlap * ratio * cell_volume(true, cell + 1, j, k));
                    else
                    {
                        const double cell_mass =
                            child.rho[cell + 1][j][k] * overlap * ratio * cell_volume(true, cell + 1, j, k);
                        momentum += cell_mass * child.u[f][j][k];
                    }
                }
            }
        return momentum;
    };
    const auto child_v_momentum = [&](int local_i, int parent_face, int k)
    {
        const double support_left  = std::max(0.0, parent_face - 0.5);
        const double support_right = std::min(static_cast<double>(sy), parent_face + 0.5);
        double       momentum      = 0.0;
        for (int f = std::max(0, static_cast<int>(std::floor(support_left * ratio)));
             f <= std::min(relationship.child_cells_y, static_cast<int>(std::ceil(support_right * ratio)));
             ++f)
            for (int side = 0; side < 2; ++side)
            {
                const int cell = f - 1 + side;
                if (cell < 0 || cell >= relationship.child_cells_y)
                    continue;
                const double half_child_width = 0.5 / ratio;
                const double left =
                    side == 0 ? static_cast<double>(f) / ratio - half_child_width : static_cast<double>(f) / ratio;
                const double right   = left + half_child_width;
                const double overlap = std::max(0.0, std::min(support_right, right) - std::max(support_left, left));
                if (overlap == 0.0)
                    continue;
                for (int fi = 0; fi < ratio; ++fi)
                {
                    const int i = ci0 + ratio * local_i + fi;
                    if (native.enabled())
                        native.term(child.rho(i, cell + 1, k),
                                    child.v(i, f, k),
                                    overlap * ratio * cell_volume(true, i, cell + 1, k));
                    else
                    {
                        const double cell_mass =
                            child.rho[i][cell + 1][k] * overlap * ratio * cell_volume(true, i, cell + 1, k);
                        momentum += cell_mass * child.v[i][f][k];
                    }
                }
            }
        return momentum;
    };

    std::vector<std::pair<std::size_t, std::size_t>> native_momentum_rows;
    if (native.enabled())
        native.clear();
    for (int face = 0; face <= sx; ++face)
        for (int local_j = 0; local_j < sy; ++local_j)
            for (int k = 0; k < nz; ++k)
            {
                const int i = oi + face;
                const int j = oj + local_j + 1;
                if (native.enabled())
                    native.begin_row();
                double target = child_u_momentum(face, local_j, k);
                if (face == 0)
                    target += 0.5 * parent.rho[oi][j][k] * cell_volume(false, oi, j, k) * parent.u[i][j][k];
                if (face == sx)
                    target +=
                        0.5 * parent.rho[oi + sx + 1][j][k] * cell_volume(false, oi + sx + 1, j, k) * parent.u[i][j][k];
                const std::size_t index = index3(i, j, k, parent_ny, nz);
                if (native.enabled())
                {
                    if (face == 0)
                        native.term(parent.rho(oi, j, k), parent.u(i, j, k), .5 * cell_volume(false, oi, j, k));
                    if (face == sx)
                        native.term(parent.rho(oi + sx + 1, j, k),
                                    parent.u(i, j, k),
                                    .5 * cell_volume(false, oi + sx + 1, j, k));
                    native.end_row();
                    native_momentum_rows.emplace_back(0, index);
                }
                momentum_target[0][index] = target;
                momentum_owned[0][index]  = 1;
                ++candidate_report.restricted_x_faces;
            }
    for (int local_i = 0; local_i < sx; ++local_i)
        for (int face = 0; face <= sy; ++face)
            for (int k = 0; k < nz; ++k)
            {
                const int i = oi + local_i + 1;
                const int j = oj + face;
                if (native.enabled())
                    native.begin_row();
                double target = child_v_momentum(local_i, face, k);
                if (face == 0)
                    target += 0.5 * parent.rho[i][oj][k] * cell_volume(false, i, oj, k) * parent.v[i][j][k];
                if (face == sy)
                    target +=
                        0.5 * parent.rho[i][oj + sy + 1][k] * cell_volume(false, i, oj + sy + 1, k) * parent.v[i][j][k];
                const std::size_t index = index3(i, j, k, parent_ny, nz);
                if (native.enabled())
                {
                    if (face == 0)
                        native.term(parent.rho(i, oj, k), parent.v(i, j, k), .5 * cell_volume(false, i, oj, k));
                    if (face == sy)
                        native.term(parent.rho(i, oj + sy + 1, k),
                                    parent.v(i, j, k),
                                    .5 * cell_volume(false, i, oj + sy + 1, k));
                    native.end_row();
                    native_momentum_rows.emplace_back(1, index);
                }
                momentum_target[1][index] = target;
                momentum_owned[1][index]  = 1;
                ++candidate_report.restricted_y_faces;
            }
    for (int local_i = 0; local_i < sx; ++local_i)
        for (int local_j = 0; local_j < sy; ++local_j)
            for (int k = 0; k < nz; ++k)
            {
                const int i      = oi + local_i + 1;
                const int j      = oj + local_j + 1;
                double    target = 0.0;
                if (native.enabled())
                    native.begin_row();
                for (int fi = 0; fi < ratio; ++fi)
                    for (int fj = 0; fj < ratio; ++fj)
                    {
                        const int ci = ci0 + ratio * local_i + fi;
                        const int cj = cj0 + ratio * local_j + fj;
                        if (native.enabled())
                        {
                            native.term(child.rho(ci, cj, k), child.w(ci, cj, k), .5 * cell_volume(true, ci, cj, k));
                            if (k + 1 < nz)
                                native.term(child.rho(ci, cj, k + 1),
                                            child.w(ci, cj, k),
                                            .5 * cell_volume(true, ci, cj, k + 1));
                        }
                        else
                            target += dual_mass(true, child.rho, 2, ci, cj, k) * child.w[ci][cj][k];
                    }
                const std::size_t index = index3(i, j, k, parent_ny, nz);
                if (native.enabled())
                {
                    native.end_row();
                    native_momentum_rows.emplace_back(2, index);
                }
                momentum_target[2][index] = target;
                momentum_owned[2][index]  = 1;
                ++candidate_report.restricted_z_faces;
            }

    if (native.enabled())
    {
        native.dispatch();
        for (std::size_t row = 0; row < native_momentum_rows.size(); ++row)
        {
            const auto [component, index]     = native_momentum_rows[row];
            momentum_target[component][index] = native.inventory(row);
        }
    }

    const auto add_momentum_reflux = [&](std::size_t component, int i, int j, int k, double value)
    { momentum_delta[component][index3(i, j, k, parent_ny, nz)] += value; };
    for (int local_j = 0; local_j < sy; ++local_j)
        for (int k = 0; k < nz - 1; ++k)
        {
            const std::size_t bin =
                static_cast<std::size_t>(local_j) * static_cast<std::size_t>(nz - 1) + static_cast<std::size_t>(k);
            for (std::size_t component = 0; component < 3; ++component)
            {
                const double west = side_value(parent_advective_momentum, parent_pressure_momentum, 0, component, bin) -
                                    side_value(child_advective_momentum, child_pressure_momentum, 0, component, bin);
                const double east = side_value(child_advective_momentum, child_pressure_momentum, 1, component, bin) -
                                    side_value(parent_advective_momentum, parent_pressure_momentum, 1, component, bin);
                const int    west_i = component == 0 ? oi : oi;
                const int    east_i = component == 0 ? oi + sx : oi + sx + 1;
                const int    j      = component == 1 ? oj + local_j + 1 : oj + local_j + 1;
                add_momentum_reflux(component, west_i, j, k, west);
                add_momentum_reflux(component, east_i, j, k, east);
            }
        }
    for (int local_i = 0; local_i < sx; ++local_i)
        for (int k = 0; k < nz - 1; ++k)
        {
            const std::size_t bin =
                static_cast<std::size_t>(local_i) * static_cast<std::size_t>(nz - 1) + static_cast<std::size_t>(k);
            for (std::size_t component = 0; component < 3; ++component)
            {
                const double south =
                    side_value(parent_advective_momentum, parent_pressure_momentum, 2, component, bin) -
                    side_value(child_advective_momentum, child_pressure_momentum, 2, component, bin);
                const double north = side_value(child_advective_momentum, child_pressure_momentum, 3, component, bin) -
                                     side_value(parent_advective_momentum, parent_pressure_momentum, 3, component, bin);
                const int    i     = oi + local_i + 1;
                const int    south_j = component == 1 ? oj : oj;
                const int    north_j = component == 1 ? oj + sy : oj + sy + 1;
                add_momentum_reflux(component, i, south_j, k, south);
                add_momentum_reflux(component, i, north_j, k, north);
            }
        }

    if (native.enabled())
    {
        native.clear();
        const auto add_mass = [&](const Field3D& density, std::size_t component, int i, int j, int k, double value)
        {
            native.term(density(i, j, k), value, .5 * cell_volume(false, i, j, k));
            int ni = i, nj = j, nk = k;
            if (component == 0)
                ++ni;
            else if (component == 1)
                ++nj;
            else
                ++nk;
            if (nk < nz)
                native.term(density(ni, nj, nk), value, .5 * cell_volume(false, ni, nj, nk));
        };
        const std::array<const Field3D*, 3> velocity{&parent.u, &parent.v, &parent.w};
        for (std::size_t component = 0; component < 3; ++component)
            for (int i = 0; i < parent_nx; ++i)
                for (int j = 0; j < parent_ny; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        const auto index = index3(i, j, k, parent_ny, nz);
                        if (!momentum_owned[component][index] && momentum_delta[component][index] == 0)
                            continue;
                        native.begin_row();
                        if (momentum_owned[component][index])
                            native.term(1, momentum_target[component][index], 1);
                        else
                            add_mass(parent.rho, component, i, j, k, (*velocity[component])(i, j, k));
                        native.term(1, momentum_delta[component][index], 1);
                        native.denominator();
                        add_mass(candidate.rho, component, i, j, k, 1);
                        native.end_row();
                    }
        native.dispatch();
        native_row = 0;
    }
    for (std::size_t component = 0; component < 3; ++component)
        for (int i = 0; i < parent_nx; ++i)
            for (int j = 0; j < parent_ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const std::size_t index = index3(i, j, k, parent_ny, nz);
                    if (!momentum_owned[component][index] && momentum_delta[component][index] == 0.0)
                        continue;
                    const double   mass         = dual_mass(false, candidate.rho, component, i, j, k);
                    const Field3D* velocity[]   = {&parent.u, &parent.v, &parent.w};
                    const float    old_velocity = (*velocity[component])(i, j, k);
                    const double   original_mass =
                        momentum_owned[component][index] ? mass : dual_mass(false, parent.rho, component, i, j, k);
                    double target = momentum_owned[component][index]
                                        ? momentum_target[component][index]
                                        : original_mass * static_cast<double>(old_velocity);
                    target += momentum_delta[component][index];
                    const float stored =
                        native.enabled() ? native.stored(native_row++) : static_cast<float>(target / mass);
                    if (!std::isfinite(mass) || mass <= 0.0 || !std::isfinite(stored))
                        reject("momentum feedback is not representable");
                    if (component == 0)
                        candidate.u[i][j][k] = stored;
                    else if (component == 1)
                        candidate.v[i][j][k] = stored;
                    else
                        candidate.w[i][j][k] = stored;
                    applied_reflux_sum[2 + component].add(momentum_delta[component][index]);
                    candidate_report.storage_residual[2 + component] += target - mass * static_cast<double>(stored);
                }

    for (int i = 0; i < parent_nx; ++i)
        for (int j = 0; j < parent_ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t index = index3(i, j, k, parent_ny, nz);
                if (mass_delta[index] == 0.0 && theta_delta[index] == 0.0 &&
                    !(i > oi && i <= oi + sx && j > oj && j <= oj + sy))
                    continue;
                const double pressure =
                    tmv::dynamics_eos::pressure_fp64(candidate.rho[i][j][k], candidate.theta[i][j][k]);
                const float stored = static_cast<float>(pressure);
                if (!std::isfinite(stored) || stored <= 0.0f)
                    reject("feedback pressure diagnosis is not representable");
                candidate.p[i][j][k] = stored;
            }

    for (std::size_t currency = 0; currency < 5; ++currency)
        candidate_report.applied_reflux[currency] = applied_reflux_sum[currency].value();
    const auto oriented_scalar_sum = [](const auto& coarse, const auto& fine)
    {
        tmv::diagnostics::CompensatedSum value;
        for (std::size_t bin = 0; bin < coarse.west.size(); ++bin)
        {
            value.add(coarse.west[bin] - fine.west[bin]);
            value.add(fine.east[bin] - coarse.east[bin]);
        }
        for (std::size_t bin = 0; bin < coarse.south.size(); ++bin)
        {
            value.add(coarse.south[bin] - fine.south[bin]);
            value.add(fine.north[bin] - coarse.north[bin]);
        }
        return value.value();
    };
    candidate_report.interface_mismatch[0] = oriented_scalar_sum(parent_mass, child_mass);
    candidate_report.interface_mismatch[1] = oriented_scalar_sum(parent_theta, child_theta);
    const auto child_scalar_transport      = [](const auto& fluxes)
    {
        tmv::diagnostics::CompensatedSum value;
        for (std::size_t bin = 0; bin < fluxes.west.size(); ++bin)
        {
            value.add(fluxes.west[bin]);
            value.add(-fluxes.east[bin]);
        }
        for (std::size_t bin = 0; bin < fluxes.south.size(); ++bin)
        {
            value.add(fluxes.south[bin]);
            value.add(-fluxes.north[bin]);
        }
        return value.value();
    };
    candidate_report.child_interface_transport[0] = child_scalar_transport(child_mass);
    candidate_report.child_interface_transport[1] = child_scalar_transport(child_theta);
    for (std::size_t component = 0; component < 3; ++component)
    {
        tmv::diagnostics::CompensatedSum value;
        for (std::size_t bin = 0; bin < parent_advective_momentum.west.component[component].size(); ++bin)
        {
            value.add(side_value(parent_advective_momentum, parent_pressure_momentum, 0, component, bin) -
                      side_value(child_advective_momentum, child_pressure_momentum, 0, component, bin));
            value.add(side_value(child_advective_momentum, child_pressure_momentum, 1, component, bin) -
                      side_value(parent_advective_momentum, parent_pressure_momentum, 1, component, bin));
        }
        for (std::size_t bin = 0; bin < parent_advective_momentum.south.component[component].size(); ++bin)
        {
            value.add(side_value(parent_advective_momentum, parent_pressure_momentum, 2, component, bin) -
                      side_value(child_advective_momentum, child_pressure_momentum, 2, component, bin));
            value.add(side_value(child_advective_momentum, child_pressure_momentum, 3, component, bin) -
                      side_value(parent_advective_momentum, parent_pressure_momentum, 3, component, bin));
        }
        candidate_report.interface_mismatch[2 + component] = value.value();
        tmv::diagnostics::CompensatedSum child_transport;
        for (std::size_t bin = 0; bin < child_advective_momentum.west.component[component].size(); ++bin)
        {
            child_transport.add(side_value(child_advective_momentum, child_pressure_momentum, 0, component, bin));
            child_transport.add(-side_value(child_advective_momentum, child_pressure_momentum, 1, component, bin));
        }
        for (std::size_t bin = 0; bin < child_advective_momentum.south.component[component].size(); ++bin)
        {
            child_transport.add(side_value(child_advective_momentum, child_pressure_momentum, 2, component, bin));
            child_transport.add(-side_value(child_advective_momentum, child_pressure_momentum, 3, component, bin));
        }
        candidate_report.child_interface_transport[2 + component] = child_transport.value();
    }
    const auto child_end_inventory = mapped
                                         ? nested_two_way_child_inventory_mapped(*mapped, child)
                                         : nested_two_way_child_inventory_flat(relationship, layer_thickness_m, child);
    for (std::size_t currency = 0; currency < 5; ++currency)
    {
        candidate_report.child_interval_change[currency] =
            child_end_inventory[currency] - child_interval_begin_inventory[currency];
        candidate_report.child_composite_residual[currency] =
            candidate_report.child_interval_change[currency] -
            (candidate_report.child_interface_transport[currency] + candidate_report.child_interface_source[currency] +
             candidate_report.child_dynamics_source[currency] + candidate_report.child_dynamics_storage[currency]);
        const double composite_scale = std::max({1.0,
                                                 std::abs(child_interval_begin_inventory[currency]),
                                                 std::abs(child_end_inventory[currency]),
                                                 std::abs(candidate_report.child_interface_transport[currency]),
                                                 std::abs(candidate_report.child_interface_source[currency]),
                                                 std::abs(candidate_report.child_dynamics_source[currency]),
                                                 std::abs(candidate_report.child_dynamics_storage[currency])});
        // The whole-domain scale already accounts for cell count, so the
        // accumulation allowance must not weaken with resolution.
        candidate_report.child_composite_tolerance[currency] =
            512.0 * std::numeric_limits<float>::epsilon() * composite_scale;
        if (!std::isfinite(candidate_report.interface_mismatch[currency]) ||
            !std::isfinite(candidate_report.applied_reflux[currency]) ||
            !std::isfinite(candidate_report.storage_residual[currency]) ||
            !std::isfinite(candidate_report.child_composite_residual[currency]) ||
            !std::isfinite(candidate_report.child_composite_tolerance[currency]) ||
            std::abs(candidate_report.interface_mismatch[currency] - candidate_report.applied_reflux[currency]) >
                duration_tolerance(candidate_report.interface_mismatch[currency]))
        {
            std::ostringstream detail;
            detail << "interface mismatch does not close against applied reflux: currency=" << currency
                   << " mismatch=" << std::setprecision(17) << candidate_report.interface_mismatch[currency]
                   << " applied=" << candidate_report.applied_reflux[currency] << " difference="
                   << candidate_report.interface_mismatch[currency] - candidate_report.applied_reflux[currency];
            reject(detail.str().c_str());
        }
        if (std::abs(candidate_report.child_composite_residual[currency]) >
            candidate_report.child_composite_tolerance[currency])
        {
            std::ostringstream detail;
            detail << "nested two-way feedback: child currency " << currency
                   << " interval inventory does not close: change=" << candidate_report.child_interval_change[currency]
                   << " transport=" << candidate_report.child_interface_transport[currency]
                   << " interface_source=" << candidate_report.child_interface_source[currency]
                   << " dynamics_source=" << candidate_report.child_dynamics_source[currency]
                   << " dynamics_storage=" << candidate_report.child_dynamics_storage[currency]
                   << " residual=" << candidate_report.child_composite_residual[currency]
                   << " tolerance=" << candidate_report.child_composite_tolerance[currency];
            throw std::runtime_error(detail.str());
        }
    }

    std::array<Field3D, 7> prepared_child;
    prepare_water_feedback(
        parent,
        child,
        candidate,
        candidate_report,
        moist,
        [&](const Field3D& coarse, const Field3D& fine, const auto& coarse_flux, const auto& fine_flux, bool positive)
        {
            if (mapped)
                return prepare_nested_scalar_feedback_mapped(*mapped,
                                                             parent.rho,
                                                             candidate.rho,
                                                             child.rho,
                                                             coarse,
                                                             fine,
                                                             coarse_flux,
                                                             fine_flux,
                                                             false,
                                                             positive,
                                                             child_candidates != nullptr,
                                                             compute);
            return prepare_nested_scalar_feedback_flat(relationship,
                                                       layer_thickness_m,
                                                       parent.rho,
                                                       candidate.rho,
                                                       child.rho,
                                                       coarse,
                                                       fine,
                                                       coarse_flux,
                                                       fine_flux,
                                                       false,
                                                       positive,
                                                       child_candidates != nullptr,
                                                       compute);
        },
        prepared_child,
        [&](int i, int j, int k) { return cell_volume(false, i, j, k); },
        compute);
    candidate_report.complete = true;
    using std::swap;
    swap(parent.rho, candidate.rho);
    swap(parent.theta, candidate.theta);
    swap(parent.p, candidate.p);
    swap(parent.u, candidate.u);
    swap(parent.v, candidate.v);
    swap(parent.w, candidate.w);
    if (moist)
        publish_water(parent, candidate);
    if (child_candidates)
        child_candidates->swap(prepared_child);
    report = candidate_report;
}

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
                                        const NestedMoistFeedbackInputs*             moist,
                                        std::array<Field3D, 7>*                      child_candidates,
                                        ComputeBackendRuntimeState*                  compute)
{
    apply_nested_two_way_feedback(relationship,
                                  layer_thickness_m,
                                  parent_mass,
                                  child_mass,
                                  parent_theta,
                                  child_theta,
                                  parent_advective_momentum,
                                  parent_pressure_momentum,
                                  child_advective_momentum,
                                  child_pressure_momentum,
                                  child_interface_source,
                                  child_dynamics_source,
                                  child_dynamics_storage,
                                  child_interval_begin_inventory,
                                  parent,
                                  child,
                                  report,
                                  moist,
                                  nullptr,
                                  child_candidates,
                                  compute);
}

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
                                          const NestedMoistFeedbackInputs* moist,
                                          std::array<Field3D, 7>*          child_candidates,
                                          ComputeBackendRuntimeState*      compute)
{
    apply_nested_two_way_feedback(geometry.relationship,
                                  {},
                                  parent_mass,
                                  child_mass,
                                  parent_theta,
                                  child_theta,
                                  parent_advective_momentum,
                                  parent_pressure_momentum,
                                  child_advective_momentum,
                                  child_pressure_momentum,
                                  child_interface_source,
                                  child_dynamics_source,
                                  child_dynamics_storage,
                                  child_interval_begin_inventory,
                                  parent,
                                  child,
                                  report,
                                  moist,
                                  &geometry,
                                  child_candidates,
                                  compute);
}

static PreparedNestedPrivateFeedback
prepare_nested_private_feedback_impl(const tmv::domain::OneWayNestRelationshipV1&         relationship,
                                const std::vector<double>&                           layer_thickness,
                                const tmv::domain::NestedMappedTerrainGeometryV1*    geometry,
                                const Field3D&                                       parent_density_before,
                                const Field3D&                                       parent_density_after,
                                const Field3D&                                       child_density,
                                const std::vector<NestedPrivateScalarFeedbackInput>& inputs,
                                ComputeBackendRuntimeState*                          compute)
{
    if (inputs.size() > 16)
        throw std::invalid_argument("too many private feedback carriers");
    PreparedNestedPrivateFeedback candidate;
    candidate.parent.reserve(inputs.size());
    candidate.child.reserve(inputs.size());
    candidate.reports.reserve(inputs.size());
    for (std::size_t f = 0; f < inputs.size(); ++f)
    {
        const auto& input = inputs[f];
        if (input.name.empty() || input.name.size() >= 64 || !input.parent || !input.child || !input.parent_flux ||
            !input.child_flux)
            throw std::invalid_argument("private feedback binding is incomplete");
        for (std::size_t prior = 0; prior < f; ++prior)
            if (inputs[prior].name == input.name || inputs[prior].parent == input.parent ||
                inputs[prior].child == input.child)
                throw std::invalid_argument("private feedback bindings are not unique");
        Field3D                    parent_lower, child_lower;
        NestedScalarFeedbackBounds bounds;
        if (input.parent_mass_before || input.parent_mass_after || input.child_mass)
        {
            if (!input.parent_mass_before || !input.parent_mass_after || !input.child_mass)
                throw std::invalid_argument("private mass/number feedback binding is incomplete");
            const auto same_shape = [](const Field3D& a, const Field3D& b)
            { return a.size_r() == b.size_r() && a.size_th() == b.size_th() && a.size_z() == b.size_z(); };
            if (!same_shape(*input.parent, *input.parent_mass_before) ||
                !same_shape(*input.parent, *input.parent_mass_after) ||
                !same_shape(*input.parent, parent_density_before) || !same_shape(*input.parent, parent_density_after) ||
                !same_shape(*input.child, *input.child_mass) || !same_shape(*input.child, child_density))
                throw std::invalid_argument("private mass/number feedback shapes differ");
            double     minimum_ratio = std::numeric_limits<double>::infinity();
            const auto observe       = [&](const Field3D& mass, const Field3D& number, const Field3D& density)
            {
                for (int i = 1; i < mass.size_r() - 1; ++i)
                    for (int j = 1; j < mass.size_th() - 1; ++j)
                        for (int k = 0; k < mass.size_z(); ++k)
                        {
                            const double q = mass(i, j, k), n = number(i, j, k), rho = density(i, j, k);
                            if (!std::isfinite(q) || q < 0 || !std::isfinite(n) || n < 0 || !std::isfinite(rho) ||
                                rho <= 0)
                                throw std::invalid_argument("private mass/number feedback reservoir is invalid");
                            if (rho * q > 1e-12)
                            {
                                if (n <= 0)
                                {
                                    std::ostringstream message;
                                    message << std::setprecision(17)
                                            << "resolved hydrometeor has no population before feedback: " << input.name
                                            << ", level=" << (&mass == input.parent_mass_before ? "parent" : "child")
                                            << ", cell=" << i << ',' << j << ',' << k << ", mass=" << q
                                            << ", density=" << rho;
                                    throw std::invalid_argument(message.str());
                                }
                                minimum_ratio = std::min(minimum_ratio, n / q);
                            }
                        }
            };
            observe(*input.parent_mass_before, *input.parent, parent_density_before);
            observe(*input.child_mass, *input.child, child_density);
            const auto lower_bound = [&](const Field3D& mass, const Field3D& density, const Field3D* retained)
            {
                Field3D result(mass.size_r(), mass.size_th(), mass.size_z(), 0.0f);
                for (int i = 1; i < mass.size_r() - 1; ++i)
                    for (int j = 1; j < mass.size_th() - 1; ++j)
                        for (int k = 0; k < mass.size_z(); ++k)
                        {
                            const double q = mass(i, j, k), rho = density(i, j, k);
                            if (!std::isfinite(q) || q < 0 || !std::isfinite(rho) || rho <= 0)
                                throw std::invalid_argument("private feedback mass candidate is invalid");
                            if (rho * q <= 1e-12)
                                continue;
                            const double target = minimum_ratio * q;
                            float        stored = static_cast<float>(target);
                            if (stored > target)
                                stored = std::nextafter(stored, 0.0f);
                            if (!std::isfinite(stored) || stored <= 0)
                                throw std::domain_error("private feedback cannot represent population support: " +
                                                        input.name);
                            if (retained)
                                stored = std::min(stored, (*retained)(i, j, k));
                            result(i, j, k) = stored;
                        }
                return result;
            };
            parent_lower = lower_bound(*input.parent_mass_after, parent_density_after, nullptr);
            child_lower  = lower_bound(*input.child_mass, child_density, input.child);
            bounds       = {&parent_lower, &child_lower};
        }
        auto field = [&]()
        {
            try
            {
                return geometry ? prepare_nested_scalar_feedback_mapped(*geometry,
                                                                        parent_density_before,
                                                                        parent_density_after,
                                                                        child_density,
                                                                        *input.parent,
                                                                        *input.child,
                                                                        *input.parent_flux,
                                                                        *input.child_flux,
                                                                        false,
                                                                        true,
                                                                        true,
                                                                        compute,
                                                                        bounds)
                                : prepare_nested_scalar_feedback_flat(relationship,
                                                                      layer_thickness,
                                                                      parent_density_before,
                                                                      parent_density_after,
                                                                      child_density,
                                                                      *input.parent,
                                                                      *input.child,
                                                                      *input.parent_flux,
                                                                      *input.child_flux,
                                                                      false,
                                                                      true,
                                                                      true,
                                                                      compute,
                                                                      bounds);
            }
            catch (const NestedScalarInventoryExhausted& error)
            {
                throw NestedScalarInventoryExhausted(input.name + ": " + error.what());
            }
        }();
        NestedPrivateScalarFeedbackReport report;
        report.name              = input.name;
        auto& budget             = report.carrier;
        budget.parent            = field.report;
        budget.child_begin       = input.child_begin;
        budget.interface_source  = input.interface_source;
        budget.physics_source    = input.physics_source;
        budget.transport_storage = input.transport_storage;
        tmv::diagnostics::CompensatedSum transport;
        for (double v : input.child_flux->west)
            transport.add(v);
        for (double v : input.child_flux->east)
            transport.add(-v);
        for (double v : input.child_flux->south)
            transport.add(v);
        for (double v : input.child_flux->north)
            transport.add(-v);
        budget.child_transport  = transport.value();
        budget.child_change     = budget.parent.child_inventory - budget.child_begin;
        budget.closure_residual = budget.child_change - budget.child_transport - budget.interface_source -
                                  budget.physics_source - budget.transport_storage;
        double scale            = 0;
        for (double v : {budget.child_begin,
                         budget.parent.child_inventory,
                         budget.child_transport,
                         budget.interface_source,
                         budget.physics_source,
                         budget.transport_storage})
            scale = std::max(scale, std::abs(v));
        budget.tolerance = 512 * std::numeric_limits<float>::epsilon() * scale;
        if (!valid_nested_water_feedback_report(budget))
        {
            std::ostringstream message;
            message << std::setprecision(17) << "private feedback inventory does not close: " << input.name
                    << ", residual=" << budget.closure_residual << ", tolerance=" << budget.tolerance
                    << ", child_begin=" << budget.child_begin << ", child_change=" << budget.child_change
                    << ", transport=" << budget.child_transport << ", interface=" << budget.interface_source
                    << ", physics=" << budget.physics_source << ", storage=" << budget.transport_storage
                    << ", parent_residual=" << budget.parent.closure_residual
                    << ", parent_initial=" << budget.parent.initial_inventory
                    << ", parent_final=" << budget.parent.final_inventory;
            throw std::runtime_error(message.str());
        }
        candidate.parent.push_back(std::move(field.scalar));
        candidate.child.push_back(std::move(field.child_scalar));
        candidate.reports.push_back(std::move(report));
    }
    return candidate;
}

PreparedNestedPrivateFeedback
prepare_nested_private_feedback(const tmv::domain::OneWayNestRelationshipV1&         relationship,
                                const std::vector<double>&                           layer_thickness,
                                const tmv::domain::NestedMappedTerrainGeometryV1*    geometry,
                                const Field3D&                                       parent_density_before,
                                const Field3D&                                       parent_density_after,
                                const Field3D&                                       child_density,
                                const std::vector<NestedPrivateScalarFeedbackInput>& inputs,
                                ComputeBackendRuntimeState*                          compute,
                                const Field3D*                                       completed_cloud_return)
{
    if (!completed_cloud_return)
    {
        return prepare_nested_private_feedback_impl(relationship,
                                                    layer_thickness,
                                                    geometry,
                                                    parent_density_before,
                                                    parent_density_after,
                                                    child_density,
                                                    inputs,
                                                    compute);
    }
    const auto& r          = relationship;
    const auto  same_shape = [&](const Field3D& f)
    { return f.size_r() == r.parent_cells_x + 2 && f.size_th() == r.parent_cells_y + 2 && f.size_z() == r.levels_z; };
    if (!same_shape(*completed_cloud_return) || !same_shape(parent_density_before) || !same_shape(parent_density_after))
    {
        throw std::invalid_argument("private reaction feedback shapes differ");
    }
    std::size_t cloud = inputs.size(), aerosol = inputs.size();
    for (std::size_t f = 0; f < inputs.size(); ++f)
    {
        if (inputs[f].name == "thompson.Nc_kg-1")
        {
            cloud = f;
        }
        if (inputs[f].name == "thompson.Nwfa_kg-1")
        {
            aerosol = f;
        }
    }
    if (cloud == inputs.size() || aerosol == inputs.size() || !inputs[cloud].parent ||
        !inputs[cloud].parent_mass_before || !inputs[cloud].parent_mass_after || !inputs[aerosol].parent ||
        !same_shape(*inputs[cloud].parent) || !same_shape(*inputs[aerosol].parent) ||
        !same_shape(*inputs[cloud].parent_mass_before) || !same_shape(*inputs[cloud].parent_mass_after))
    {
        throw std::invalid_argument("private reaction feedback requires the cloud/aerosol pair and cloud mass");
    }
    const auto covered = [&](int i, int j)
    {
        return i > r.child_origin_parent_cell_i && i <= r.child_origin_parent_cell_i + r.child_parent_span_x &&
               j > r.child_origin_parent_cell_j && j <= r.child_origin_parent_cell_j + r.child_parent_span_y;
    };
    const auto volume = [&](int i, int j, int k)
    {
        if (geometry)
        {
            const auto& m = geometry->parent.metrics;
            return m.dx * m.dy * m.dzeta * m.J(i - 1, j - 1, k);
        }
        return r.parent_dx_m * r.parent_dy_m * layer_thickness.at(k);
    };
    // Complete evaporation is a one-to-one particle map. Synchronize its input
    // before replaying that map; covered cells retain the child's source owner.
    Field3D before_reaction = *inputs[cloud].parent;
    for (int i = 1; i <= r.parent_cells_x; ++i)
        for (int j = 1; j <= r.parent_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const float returned = (*completed_cloud_return)(i, j, k);
                if (!std::isfinite(returned) || returned < 0)
                {
                    throw std::invalid_argument("private reaction provenance is invalid");
                }
                if (returned == 0 || covered(i, j))
                {
                    continue;
                }
                if (before_reaction(i, j, k) != 0 || (*inputs[cloud].parent_mass_before)(i, j, k) != 0 ||
                    (*inputs[cloud].parent_mass_after)(i, j, k) != 0 || (*inputs[aerosol].parent)(i, j, k) < returned)
                {
                    throw std::invalid_argument("private reaction provenance does not describe complete evaporation");
                }
                before_reaction(i, j, k) = returned;
            }
    auto replay_inputs          = inputs;
    replay_inputs[cloud].parent = &before_reaction;
    auto result                 = prepare_nested_private_feedback_impl(r,
                                                                       layer_thickness,
                                                                       geometry,
                                                                       parent_density_before,
                                                                       parent_density_after,
                                                                       child_density,
                                                                       replay_inputs,
                                                                       compute);
    struct Cell
    {
        int    i, j, k;
        double mass, withdrawal;
    };
    std::vector<Cell> pending;
    pending.reserve(256);
    nimbus::compute::NativeConservativeReduction native(compute);
    tmv::diagnostics::CompensatedSum             exchanges;
    const auto                                   flush = [&]()
    {
        if (native.enabled())
        {
            native.clear();
            for (const auto& cell : pending)
            {
                native.begin_row();
                native.term(cell.mass, result.parent[aerosol](cell.i, cell.j, cell.k), 1);
                native.term(1, -cell.withdrawal, 1);
                native.denominator();
                native.term(cell.mass, 1, 1);
                native.end_row();
                native.begin_row();
                native.term(1, 0, 1);
                native.denominator();
                native.term(1, 1, 1);
                native.end_row();
            }
            native.dispatch();
        }
        std::size_t row = 0;
        for (const auto& cell : pending)
        {
            const float aerosol_after =
                native.enabled()
                    ? native.stored(row++)
                    : float(double(result.parent[aerosol](cell.i, cell.j, cell.k)) - cell.withdrawal / cell.mass);
            if (!std::isfinite(aerosol_after) || aerosol_after < 0)
            {
                throw NestedScalarInventoryExhausted("reaction feedback exhausts its returned aerosol");
            }
            result.parent[aerosol](cell.i, cell.j, cell.k) = aerosol_after;
            result.parent[cloud](cell.i, cell.j, cell.k)   = native.enabled() ? native.stored(row++) : 0;
            exchanges.add(cell.withdrawal);
        }
        pending.clear();
    };
    for (int i = 1; i <= r.parent_cells_x; ++i)
        for (int j = 1; j <= r.parent_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const float returned = (*completed_cloud_return)(i, j, k);
                if (returned == 0 || covered(i, j))
                {
                    continue;
                }
                const double measure    = volume(i, j, k);
                const double mass       = parent_density_after(i, j, k) * measure;
                const double capacity   = parent_density_before(i, j, k) * measure * returned;
                const float  retained   = result.parent[cloud](i, j, k);
                const double withdrawal = capacity - mass * retained;
                const double rounding =
                    mass * (double(std::nextafter(retained, std::numeric_limits<float>::infinity())) - retained);
                if (!std::isfinite(withdrawal) || withdrawal < -rounding ||
                    withdrawal > mass * result.parent[aerosol](i, j, k))
                {
                    throw NestedScalarInventoryExhausted("reaction feedback cannot extrapolate its source transition");
                }
                pending.push_back({i, j, k, mass, std::max(0.0, withdrawal)});
                if (pending.size() == 256)
                {
                    flush();
                }
            }
    flush();
    for (std::size_t f : {cloud, aerosol})
    {
        tmv::diagnostics::CompensatedSum initial, final;
        for (int i = 1; i <= r.parent_cells_x; ++i)
            for (int j = 1; j <= r.parent_cells_y; ++j)
                for (int k = 0; k < r.levels_z; ++k)
                {
                    const double measure = volume(i, j, k);
                    initial.add(parent_density_before(i, j, k) * measure * (*inputs[f].parent)(i, j, k));
                    final.add(parent_density_after(i, j, k) * measure * result.parent[f](i, j, k));
                }
        auto& report                    = result.reports[f].carrier.parent;
        report.phase_exchange_inventory = f == cloud ? exchanges.value() : -exchanges.value();
        report.storage_rounding += (final.value() - report.final_inventory) -
                                   (initial.value() - report.initial_inventory) - report.phase_exchange_inventory;
        report.initial_inventory = initial.value();
        report.final_inventory   = final.value();
        report.closure_residual  = report.final_inventory - report.initial_inventory - report.restriction_change -
                                   report.applied_reflux - report.phase_exchange_inventory - report.storage_rounding;
    }
    if (!valid_nested_private_feedback_reports(result.reports))
    {
        throw std::runtime_error("private reaction feedback particle ledgers do not close");
    }
    return result;
}
