#include "core/runtime/environment_column_refresh.hpp"
#include "core/runtime/simulation_state.hpp"
#include "boundary_layer/column_evaluation.hpp"
#include "radiation/column_evaluation.hpp"
#include "terrain/conservative_transport.hpp"
#include "dynamics/moving_environment_internal.hpp"
#include "compute/compute_backend.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
namespace tmv
{
namespace
{
struct Donor
{
    int                                   i = 0, j = 0;
    std::size_t                           cell = 0;
    std::vector<double>                   p, theta, qv, rho, u, v, zero, temperature, zc, zi, dz;
    SlabColumnState                       slab;
    BoundaryLayerColumnWorkspace          pbl_workspace;
    RadiationColumnWorkspace              rad_workspace;
    RadiationPreparedColumnView           prepared;
    BoundaryLayerStoredSurfaceDiagnostics surface;
};
ExtendedRadiationFluxes extend(SimulationStorageState& s, ComputeBackend* backend)
{
    const int   nx = s.grid.nr, ny = s.grid.nth, nz = s.grid.nz;
    const auto& g = s.runtime.terrain.metrics;
    terrain_transport::validate_geometry(g);
    if (nx < 3 || ny < 3 || nz < 1 || g.z.nr != nx || g.z.nth != ny || g.z.nz != nz)
        throw std::invalid_argument("environmental columns require matching terrain and state grids");
    const auto volume_shape = [&](const Field3D& field)
    {
        if (field.size_r() != nx || field.size_th() != ny || field.size_z() != nz ||
            field.size() != std::size_t(nx) * ny * nz)
            throw std::invalid_argument("environmental column owner has malformed volume storage");
    };
    const auto surface_shape = [&](const std::vector<float>& values)
    {
        if (values.size() != std::size_t(nx) * ny)
            throw std::invalid_argument("environmental column owner has malformed surface storage");
    };
    for (auto* field : {&s.fields.p, &s.fields.theta, &s.fields.rho, &s.fields.qv})
        volume_shape(*field);
    auto& pbl        = s.runtime.boundary_layer;
    auto& rad        = s.runtime.radiation;
    auto& chaos      = s.runtime.chaos;
    auto  pbl_scheme = pbl.scheme && pbl.config.scheme_id != "none" ? pbl.scheme->clone() : nullptr;
    auto  rad_scheme = rad.scheme && rad.config.scheme_id != "none" ? rad.scheme->clone() : nullptr;
    if ((pbl.scheme && pbl.config.scheme_id != "none" && !pbl_scheme) ||
        (rad.scheme && rad.config.scheme_id != "none" && !rad_scheme))
        throw std::runtime_error("environmental physics schemes must support isolated cloning");
    const std::array<Field3D*, 7>            pbl_fields{&pbl.held_u_force_n,
                                                        &pbl.held_v_force_n,
                                                        &pbl.held_theta_inventory_rate_kgk_s,
                                                        &pbl.held_vapor_mass_rate_kg_s,
                                                        &pbl.held_cloud_liquid_mass_rate_kg_s,
                                                        &pbl.held_cloud_ice_mass_rate_kg_s,
                                                        &s.tendencies.dtke_dt_pbl};
    const std::array<std::vector<float>*, 9> surfaces{&pbl.skin_theta_2d,
                                                      &pbl.skin_qv_2d,
                                                      &s.surface.ustar,
                                                      &s.surface.tau_u,
                                                      &s.surface.tau_v,
                                                      &s.surface.sensible_heat_flux,
                                                      &s.surface.latent_heat_flux,
                                                      &s.surface.moisture_flux,
                                                      &s.surface.pbl_height};
    const std::array<Field3D*, 5>            chaos_fields{&chaos.base_du_dt_pbl,
                                                          &chaos.base_dv_dt_pbl,
                                                          &chaos.base_dtheta_dt_pbl,
                                                          &chaos.base_dqv_dt_pbl,
                                                          &chaos.base_dtke_dt_pbl};
    const std::array<std::vector<float>*, 6> chaos_surface{&chaos.base_sfc_ustar,
                                                           &chaos.base_sfc_tau_u,
                                                           &chaos.base_sfc_tau_v,
                                                           &chaos.base_sfc_sensible_heat_flux,
                                                           &chaos.base_sfc_latent_heat_flux,
                                                           &chaos.base_sfc_moisture_flux};
    if (pbl_scheme)
    {
        for (auto* field : pbl_fields)
            volume_shape(*field);
        for (auto* values : surfaces)
            surface_shape(*values);
        if (pbl.slab_columns.size() != std::size_t(nx) * ny)
            throw std::invalid_argument("environmental slabs have malformed storage");
        if (chaos.pbl_base_tendencies_valid)
        {
            for (auto* field : chaos_fields)
                volume_shape(*field);
            for (auto* values : chaos_surface)
                surface_shape(*values);
        }
    }
    if (rad_scheme)
        volume_shape(rad.held_cell_power_w);
    auto* pbl_kernels = backend ? backend->boundary_layer_kernels() : nullptr;
    auto* rad_kernels = backend ? backend->radiation_kernels() : nullptr;
    if (backend &&
        ((pbl_scheme && (!pbl_kernels || !pbl_scheme->supports_native_column_batch(pbl.config, *pbl_kernels))) ||
         (rad_scheme && (!rad_kernels || !rad_scheme->supports_native_column_batch(rad.config, *rad_kernels)))))
        throw std::runtime_error("native environmental-column physics is unavailable");
    auto&                                           flux = rad.held_flux_diagnostics;
    const std::array<const std::vector<double>*, 4> live_flux{&flux.upward_longwave_wm2,
                                                              &flux.downward_longwave_wm2,
                                                              &flux.upward_shortwave_wm2,
                                                              &flux.downward_shortwave_wm2};
    const bool                                      retain = !flux.empty();
    ExtendedRadiationFluxes                         extended;
    for (std::size_t f = 0; f < live_flux.size(); ++f)
    {
        if (!retain)
        {
            if (!live_flux[f]->empty())
                throw std::invalid_argument("environmental radiation interfaces are incomplete");
            continue;
        }
        if (!rad_scheme || flux.columns_x != std::size_t(nx - 2) || flux.columns_y != std::size_t(ny - 2) ||
            flux.levels != std::size_t(nz) || live_flux[f]->size() != std::size_t(nx - 2) * (ny - 2) * (nz + 1))
            throw std::invalid_argument("environmental radiation interfaces have malformed ownership");
        for (double value : *live_flux[f])
            if (!std::isfinite(value))
                throw std::invalid_argument("environmental radiation interface is not finite");
        extended[f].resize(std::size_t(nx) * ny * (nz + 1));
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
                std::copy_n(live_flux[f]->data() + (std::size_t(i - 1) * (ny - 2) + j - 1) * (nz + 1),
                            nz + 1,
                            extended[f].data() + (std::size_t(i) * ny + j) * (nz + 1));
    }
    if (!pbl_scheme && !rad_scheme)
        return extended;
    std::vector<Donor>                        donors(2 * std::size_t(nx) + 2 * std::size_t(ny) - 4);
    std::vector<BoundaryLayerColumnStateView> pbl_views(donors.size());
    std::vector<BoundaryLayerTendencies>      pbl_tendencies(donors.size());
    std::vector<BoundaryLayerDiagnostics>     pbl_diagnostics(donors.size());
    std::vector<RadiationColumnStateView>     rad_views(donors.size());
    std::vector<RadiationColumnTendencies>    rad_tendencies(donors.size());
    std::vector<RadiationColumnFluxes>        rad_flux(donors.size());
    std::vector<double>                       heights;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; j += (i == 0 || i == nx - 1) ? 1 : ny - 1)
            for (int k = 0; k < nz; ++k)
                heights.push_back(g.z(i, j, k));
    std::vector<float> samples;
    if (backend)
    {
        auto* terrain = backend->terrain_kernels();
        if (!terrain)
            throw std::runtime_error("native environmental reference sampling is unavailable");
        samples = moving_environment_detail::sample_reference(s.base, heights, g.dzeta, *terrain);
    }
    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk;
    std::size_t      n     = 0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; j += (i == 0 || i == nx - 1) ? 1 : ny - 1, ++n)
        {
            auto& d = donors[n];
            d.i     = i;
            d.j     = j;
            d.cell  = std::size_t(i) * ny + j;
            for (auto* values : {&d.p, &d.theta, &d.qv, &d.rho, &d.u, &d.v, &d.zero, &d.temperature, &d.zc, &d.dz})
                values->resize(nz, 0);
            d.zi.resize(nz + 1);
            d.pbl_workspace.z_interface.resize(nz + 1);
            for (int k = 0; k < nz; ++k)
            {
                d.p[k]     = s.fields.p(i, j, k);
                d.theta[k] = s.fields.theta(i, j, k);
                d.qv[k]    = s.fields.qv(i, j, k);
                d.rho[k]   = s.fields.rho(i, j, k);
                d.u[k] = backend ? samples[(n * nz + k) * 6 + 3]
                                 : terrain_transport::reference_profile_value_fp32(s.base.u0, g.z(i, j, k), g.dzeta);
                d.v[k] = backend ? samples[(n * nz + k) * 6 + 4]
                                 : terrain_transport::reference_profile_value_fp32(s.base.v0, g.z(i, j, k), g.dzeta);
                d.temperature[k] = d.theta[k] * std::pow(d.p[k] / physical_constants::reference_pressure_pa, kappa);
                d.zc[k]          = g.z(i, j, k);
                d.zi[k]          = g.z_face(i, j, k);
                d.dz[k]          = g.z_face(i, j, k + 1) - d.zi[k];
                d.pbl_workspace.z_interface[k] = d.zi[k] - g.z_face(i, j, 0);
            }
            d.zi[nz]                        = g.z_face(i, j, nz);
            d.pbl_workspace.z_interface[nz] = d.zi[nz] - d.zi[0];
            if (pbl_scheme)
            {
                auto& c = pbl_views[n];
                c.p     = &d.p;
                c.theta = &d.theta;
                c.qv    = &d.qv;
                c.rho   = &d.rho;
                c.u     = &d.u;
                c.v     = &d.v;
                c.tke   = &d.zero;
                c.z_int = &d.pbl_workspace.z_interface;
                c.u_sfc = d.u[0];
                c.v_sfc = d.v[0];
                c.theta_sfc =
                    pbl.surface_config.Tsfc > 0
                        ? pbl.surface_config.Tsfc * std::pow(physical_constants::reference_pressure_pa / d.p[0], kappa)
                        : d.theta[0];
                c.qv_sfc          = pbl.surface_config.qsfc >= 0 ? pbl.surface_config.qsfc : d.qv[0];
                c.z_sfc           = d.zc[0] - d.zi[0];
                c.surface_slope_x = -g.Gx_face(i, j, 0);
                c.surface_slope_y = -g.Gy_face(i, j, 0);
                c.slab_state      = &d.slab;
                c.workspace       = &d.pbl_workspace;
                c.elapsed_time_s  = 0;
                validate_prepared_boundary_layer_column(c, s.grid.dx * s.grid.dy);
            }
            if (rad_scheme)
            {
                auto& c                     = d.prepared;
                c.pressure_pa               = &d.p;
                c.temperature_k             = &d.temperature;
                c.density_kgm3              = &d.rho;
                c.vapor_mixing_ratio        = &d.qv;
                c.cloud_liquid_mixing_ratio = &d.zero;
                c.cloud_ice_mixing_ratio    = &d.zero;
                c.z_center_m                = &d.zc;
                c.z_interface_m             = &d.zi;
                c.layer_thickness_m         = &d.dz;
                c.cosine_solar_zenith       = idealized_solar_cosine_zenith(rad.config, rad.last_update_time_s);
                c.surface_temperature_k     = pbl.surface_config.Tsfc;
                c.surface_albedo            = pbl.surface_config.albedo;
                c.surface_emissivity        = pbl.surface_config.emissivity;
                c.horizontal_cell_area_m2   = s.grid.dx * s.grid.dy;
                rad_views[n]                = prepare_radiation_column_state(rad.config, c, d.rad_workspace);
            }
        }
    for (std::size_t begin = 0; begin < donors.size(); begin += 64)
    {
        const auto count = std::min<std::size_t>(64, donors.size() - begin);
        if (backend)
        {
            if (pbl_scheme && !pbl_scheme->compute_columns_native(pbl.config,
                                                                  pbl.surface_config,
                                                                  pbl_views.data() + begin,
                                                                  count,
                                                                  pbl_tendencies.data() + begin,
                                                                  pbl_diagnostics.data() + begin,
                                                                  *pbl_kernels))
                throw std::runtime_error("native environmental PBL batch failed");
            if (rad_scheme && !rad_scheme->compute_columns_native_diagnosed(rad.config,
                                                                            rad_views.data() + begin,
                                                                            count,
                                                                            rad_tendencies.data() + begin,
                                                                            *rad_kernels,
                                                                            retain ? rad_flux.data() + begin : nullptr))
                throw std::runtime_error("native environmental radiation batch failed");
        }
        else
        {
            for (std::size_t c = begin; c < begin + count; ++c)
            {
                if (pbl_scheme)
                    pbl_scheme->compute_column(
                        pbl.config, pbl.surface_config, pbl_views[c], pbl_tendencies[c], &pbl_diagnostics[c]);
                if (rad_scheme)
                    rad_scheme->compute_column(
                        rad.config, rad_views[c], rad_tendencies[c], retain ? &rad_flux[c] : nullptr);
            }
        }
    }
    for (std::size_t c = 0; c < donors.size(); ++c)
    {
        auto& d = donors[c];
        if (pbl_scheme)
            d.surface = store_prepared_boundary_layer_result(
                pbl_views[c], s.grid.dx * s.grid.dy, pbl_tendencies[c], pbl_diagnostics[c]);
        if (rad_scheme)
        {
            finalize_prepared_radiation_column(d.prepared, d.rad_workspace, rad_tendencies[c]);
            if (retain)
            {
                const auto&                                     f = rad_flux[c];
                const std::array<const std::vector<double>*, 4> values{&f.Fup_lw, &f.Fdn_lw, &f.Fup_sw, &f.Fdn_sw};
                for (std::size_t v = 0; v < values.size(); ++v)
                {
                    if (values[v]->size() != std::size_t(nz + 1))
                        throw std::runtime_error("environmental radiation returned malformed interfaces");
                    for (double value : *values[v])
                        if (!std::isfinite(value))
                            throw std::runtime_error("environmental radiation returned nonfinite interfaces");
                    std::copy(values[v]->begin(), values[v]->end(), extended[v].data() + d.cell * (nz + 1));
                }
            }
        }
    }
    for (auto& d : donors)
    {
        if (pbl_scheme)
        {
            const auto&                                    w = d.pbl_workspace;
            const std::array<const std::vector<float>*, 7> values{&w.stored_u_force_n,
                                                                  &w.stored_v_force_n,
                                                                  &w.stored_theta_inventory_rate_kgk_s,
                                                                  &w.stored_vapor_mass_rate_kg_s,
                                                                  &w.stored_cloud_liquid_mass_rate_kg_s,
                                                                  &w.stored_cloud_ice_mass_rate_kg_s,
                                                                  &w.stored_dtke};
            for (std::size_t f = 0; f < values.size(); ++f)
                std::copy(values[f]->begin(), values[f]->end(), pbl_fields[f]->row_ptr(d.i, d.j));
            pbl.skin_theta_2d[d.cell]    = float(d.theta[0]);
            pbl.skin_qv_2d[d.cell]       = float(d.qv[0]);
            pbl.slab_columns[d.cell]     = d.slab;
            const auto&                a = d.surface;
            const std::array<float, 7> sv{
                a.ustar, a.tau_u, a.tau_v, a.sensible_heat_flux, a.latent_heat_flux, a.moisture_flux, a.pbl_height};
            for (std::size_t f = 0; f < sv.size(); ++f)
                (*surfaces[f + 2])[d.cell] = sv[f];
            if (chaos.pbl_base_tendencies_valid)
            {
                for (std::size_t f = 0; f < chaos_fields.size(); ++f)
                {
                    const auto& v = *values[f == 4 ? 6 : f];
                    std::copy(v.begin(), v.end(), chaos_fields[f]->row_ptr(d.i, d.j));
                }
                for (std::size_t f = 0; f < chaos_surface.size(); ++f)
                    (*chaos_surface[f])[d.cell] = sv[f];
            }
        }
        if (rad_scheme)
            std::copy(d.rad_workspace.cell_power_w.begin(),
                      d.rad_workspace.cell_power_w.end(),
                      rad.held_cell_power_w.row_ptr(d.i, d.j));
    }
    return extended;
}
} // namespace
ExtendedRadiationFluxes extend_environment_columns(SimulationStorageState& state)
{
    return extend(state, nullptr);
}
ExtendedRadiationFluxes extend_environment_columns(SimulationStorageState& state, ComputeBackend& backend)
{
    return extend(state, &backend);
}
} // namespace tmv
