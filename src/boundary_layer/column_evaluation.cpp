#include "boundary_layer/column_evaluation.hpp"

#include "core/infra/physical_constants.hpp"

#include <cmath>
#include <stdexcept>

namespace tmv
{

void validate_prepared_boundary_layer_column(
    const BoundaryLayerColumnStateView& column,
    double horizontal_cell_area_m2)
{
    const std::size_t nz = column.theta.size();
    if (nz == 0 || column.p.size() != nz || column.qv.size() != nz
        || column.u.size() != nz || column.v.size() != nz
        || column.rho.size() != nz || column.z_int.size() != nz + 1
        || column.workspace == nullptr)
        throw std::invalid_argument(
            "prepared boundary-layer column has inconsistent shapes");
    if (!std::isfinite(horizontal_cell_area_m2)
        || horizontal_cell_area_m2 <= 0.0)
        throw std::invalid_argument(
            "prepared boundary-layer column requires positive finite area");
    for (std::size_t k = 0; k < nz; ++k)
    {
        if (!std::isfinite(column.p[k]) || column.p[k] <= 0.0
            || !std::isfinite(column.theta[k]) || column.theta[k] <= 0.0
            || !std::isfinite(column.qv[k]) || column.qv[k] < 0.0
            || !std::isfinite(column.u[k]) || !std::isfinite(column.v[k])
            || !std::isfinite(column.rho[k]) || column.rho[k] <= 0.0
            || !std::isfinite(column.z_int[k])
            || !std::isfinite(column.z_int[k + 1])
            || column.z_int[k + 1] <= column.z_int[k]
            || (k > 0 && column.p[k] > column.p[k - 1]))
            throw std::invalid_argument(
                "prepared boundary-layer column contains unphysical state");
    }
}

BoundaryLayerStoredSurfaceDiagnostics
store_prepared_boundary_layer_result(
    BoundaryLayerColumnStateView& column,
    double horizontal_cell_area_m2,
    const BoundaryLayerTendencies& tendencies,
    const BoundaryLayerDiagnostics& diagnostics)
{
    const std::size_t nz = column.theta.size();
    if (tendencies.dthetadt_pbl.size() != nz
        || tendencies.dqvdt_pbl.size() != nz
        || tendencies.dqcdt_pbl.size() != nz
        || tendencies.dqidt_pbl.size() != nz
        || tendencies.dudt_pbl.size() != nz
        || tendencies.dvdt_pbl.size() != nz
        || tendencies.dtkedt_pbl.size() != nz)
        throw std::runtime_error(
            "boundary-layer scheme returned malformed tendency sizes");
    if (!std::isfinite(diagnostics.ustar) || diagnostics.ustar < 0.0
        || !std::isfinite(diagnostics.tau_u)
        || !std::isfinite(diagnostics.tau_v)
        || !std::isfinite(diagnostics.H) || !std::isfinite(diagnostics.E)
        || !std::isfinite(diagnostics.pbl_height)
        || diagnostics.pbl_height < 0.0)
        throw std::runtime_error(
            "boundary-layer scheme returned malformed diagnostics");

    BoundaryLayerColumnWorkspace& workspace = *column.workspace;
    workspace.stored_dtheta.resize(nz);
    workspace.stored_dqv.resize(nz);
    workspace.stored_dqc.resize(nz);
    workspace.stored_dqi.resize(nz);
    workspace.stored_du.resize(nz);
    workspace.stored_dv.resize(nz);
    workspace.stored_dtke.resize(nz);
    workspace.stored_u_force_n.resize(nz);
    workspace.stored_v_force_n.resize(nz);
    workspace.stored_theta_inventory_rate_kgk_s.resize(nz);
    workspace.stored_vapor_mass_rate_kg_s.resize(nz);
    workspace.stored_cloud_liquid_mass_rate_kg_s.resize(nz);
    workspace.stored_cloud_ice_mass_rate_kg_s.resize(nz);
    for (std::size_t k = 0; k < nz; ++k)
    {
        const float dtheta = static_cast<float>(tendencies.dthetadt_pbl[k]);
        const float dqv = static_cast<float>(tendencies.dqvdt_pbl[k]);
        const float dqc = static_cast<float>(tendencies.dqcdt_pbl[k]);
        const float dqi = static_cast<float>(tendencies.dqidt_pbl[k]);
        const float du = static_cast<float>(tendencies.dudt_pbl[k]);
        const float dv = static_cast<float>(tendencies.dvdt_pbl[k]);
        const float dtke = static_cast<float>(tendencies.dtkedt_pbl[k]);
        if (!std::isfinite(dtheta) || !std::isfinite(dqv)
            || !std::isfinite(dqc) || !std::isfinite(dqi)
            || !std::isfinite(du) || !std::isfinite(dv)
            || !std::isfinite(dtke))
            throw std::runtime_error(
                "boundary-layer scheme produced a non-finite tendency");
        const double cell_mass_kg = column.rho[k]
            * (column.z_int[k + 1] - column.z_int[k])
            * horizontal_cell_area_m2;
        const float u_force = static_cast<float>(
            cell_mass_kg * static_cast<double>(du));
        const float v_force = static_cast<float>(
            cell_mass_kg * static_cast<double>(dv));
        const float theta_rate = static_cast<float>(
            cell_mass_kg * static_cast<double>(dtheta));
        const float vapor_rate = static_cast<float>(
            cell_mass_kg * static_cast<double>(dqv));
        const float cloud_liquid_rate = static_cast<float>(
            cell_mass_kg * static_cast<double>(dqc));
        const float cloud_ice_rate = static_cast<float>(
            cell_mass_kg * static_cast<double>(dqi));
        if (!std::isfinite(u_force) || !std::isfinite(v_force)
            || !std::isfinite(theta_rate) || !std::isfinite(vapor_rate)
            || !std::isfinite(cloud_liquid_rate)
            || !std::isfinite(cloud_ice_rate))
            throw std::runtime_error(
                "boundary-layer conserved source overflowed published precision");
        workspace.stored_dtheta[k] = dtheta;
        workspace.stored_dqv[k] = dqv;
        workspace.stored_dqc[k] = dqc;
        workspace.stored_dqi[k] = dqi;
        workspace.stored_du[k] = du;
        workspace.stored_dv[k] = dv;
        workspace.stored_dtke[k] = dtke;
        workspace.stored_u_force_n[k] = u_force;
        workspace.stored_v_force_n[k] = v_force;
        workspace.stored_theta_inventory_rate_kgk_s[k] = theta_rate;
        workspace.stored_vapor_mass_rate_kg_s[k] = vapor_rate;
        workspace.stored_cloud_liquid_mass_rate_kg_s[k] =
            cloud_liquid_rate;
        workspace.stored_cloud_ice_mass_rate_kg_s[k] = cloud_ice_rate;
    }

    if (column.slab_state != nullptr && column.slab_state->initialized)
    {
        if (!std::isfinite(column.slab_state->h_m)
            || column.slab_state->h_m <= 0.0
            || !std::isfinite(column.slab_state->theta_m_k)
            || column.slab_state->theta_m_k <= 0.0
            || !std::isfinite(column.slab_state->qv_m_kgkg)
            || column.slab_state->qv_m_kgkg < 0.0)
            throw std::runtime_error(
                "boundary-layer scheme produced invalid column memory");
    }

    BoundaryLayerStoredSurfaceDiagnostics stored;
    stored.ustar = static_cast<float>(diagnostics.ustar);
    stored.tau_u = static_cast<float>(diagnostics.tau_u);
    stored.tau_v = static_cast<float>(diagnostics.tau_v);
    stored.sensible_heat_flux = static_cast<float>(diagnostics.H);
    stored.moisture_flux = static_cast<float>(diagnostics.E);
    stored.latent_heat_flux = static_cast<float>(
        diagnostics.E * physical_constants::latent_heat_vaporization_jkg);
    stored.pbl_height = static_cast<float>(diagnostics.pbl_height);
    const float values[] = {
        stored.ustar, stored.tau_u, stored.tau_v,
        stored.sensible_heat_flux, stored.latent_heat_flux,
        stored.moisture_flux, stored.pbl_height};
    for (float value : values)
        if (!std::isfinite(value))
            throw std::runtime_error(
                "boundary-layer diagnostics overflowed published precision");
    return stored;
}

BoundaryLayerStoredSurfaceDiagnostics
evaluate_prepared_boundary_layer_column(
    BoundaryLayerSchemeBase& scheme,
    const BoundaryLayerConfig& config,
    const SurfaceConfig& surface,
    BoundaryLayerColumnStateView& column,
    double horizontal_cell_area_m2,
    BoundaryLayerTendencies& tendencies,
    BoundaryLayerDiagnostics& diagnostics)
{
    validate_prepared_boundary_layer_column(
        column, horizontal_cell_area_m2);
    tendencies.clear_retain_capacity();
    diagnostics.reset_retain_capacity(false);
    scheme.compute_column(
        config, surface, column, tendencies, &diagnostics);
    return store_prepared_boundary_layer_result(
        column, horizontal_cell_area_m2, tendencies, diagnostics);
}

const char* boundary_layer_prepared_column_algorithm_id() noexcept
{
    return "pbl_column_conserved_source_fp32_storage_v1";
}

} // namespace tmv
