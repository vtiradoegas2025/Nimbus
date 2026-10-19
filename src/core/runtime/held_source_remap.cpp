#include "core/runtime/held_source_remap.hpp"
#include "core/runtime/column_remap_internal.hpp"
#include "core/runtime/simulation_state.hpp"
#include "compute/dispatch/terrain.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace tmv
{
namespace
{
using held_remap_detail::Columns;
using held_remap_detail::Storage;
struct Worker
{
    int                    nx, ny;
    double                 sx, sy;
    TerrainComputeKernels* kernels;
    Columns                run(const Columns& source, int levels, Storage storage) const
    {
        const auto count    = std::size_t(nx) * ny * levels;
        const int  channels = storage == Storage::Slab ? 3 : 1;
        for (int c = 0; c < channels; ++c)
        {
            if (source.values[c].size() != count)
                throw std::invalid_argument("held-column storage has the wrong shape");
            for (double value : source.values[c])
                if (!std::isfinite(value))
                    throw std::invalid_argument("held-column storage is not finite");
        }
        for (std::size_t n = 0; n < count; ++n)
        {
            if (storage == Storage::Slab &&
                (source.values[0][n] <= 0 || source.values[1][n] <= 0 || source.values[2][n] < 0 ||
                 !std::isfinite(source.values[0][n] * source.values[1][n]) ||
                 source.values[0][n] * source.values[1][n] <= 0 ||
                 !std::isfinite(source.values[0][n] * source.values[2][n]) ||
                 (source.values[2][n] > 0 && source.values[0][n] * source.values[2][n] == 0)))
                throw std::invalid_argument("held-column slab inventories are invalid");
            if (storage == Storage::Rain && source.values[0][n] < 0)
                throw std::invalid_argument("held-column rainfall is negative");
        }
        if (kernels)
            return held_remap_detail::remap_native(*kernels, source, nx, ny, levels, sx, sy, storage);
        Columns      result = source;
        const double ax = std::abs(sx), ay = std::abs(sy);
        const auto   currency = [&](std::size_t n, int c)
        { return storage == Storage::Slab && c > 0 ? source.values[0][n] * source.values[c][n] : source.values[c][n]; };
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < levels; ++k)
                {
                    const auto n = (std::size_t(i) * ny + j) * levels + k;
                    const auto x = (std::size_t(std::clamp(i + (sx >= 0 ? 1 : -1), 0, nx - 1)) * ny + j) * levels + k;
                    const auto y = (std::size_t(i) * ny + std::clamp(j + (sy >= 0 ? 1 : -1), 0, ny - 1)) * levels + k;
                    for (int c = 0; c < channels; ++c)
                    {
                        double value = source.values[c][n];
                        if (sx != 0 || sy != 0)
                        {
                            value = (1 - ax - ay) * currency(n, c) + ax * currency(x, c) + ay * currency(y, c);
                            if (storage == Storage::Slab && c > 0)
                                value /= result.values[0][n];
                        }
                        if (!std::isfinite(value))
                            throw std::overflow_error("held-column remapping overflowed");
                        result.values[c][n] = value;
                    }
                }
        if (storage == Storage::Float || storage == Storage::Rain)
        {
            result.rounded.resize(count);
            result.residual.resize(count);
            for (std::size_t n = 0; n < count; ++n)
            {
                result.rounded[n] = static_cast<float>(result.values[0][n]);
                result.residual[n] =
                    storage == Storage::Rain ? static_cast<float>(result.values[0][n] - result.rounded[n]) : 0;
                if (!std::isfinite(result.rounded[n]) || !std::isfinite(result.residual[n]))
                    throw std::overflow_error("held-column remapping overflowed FP32 storage");
            }
        }
        return result;
    }
    template <class Value>
    Columns scalar(const std::vector<Value>& values, int levels, Storage storage) const
    {
        Columns source;
        source.values[0].assign(values.begin(), values.end());
        return run(source, levels, storage);
    }
};
struct VolumeCandidate
{
    Field3D* target;
    Field3D  value;
};
struct SurfaceCandidate
{
    std::vector<float>* target;
    std::vector<float>  value;
};
void remap(
    SimulationStorageState& s, double sx, double sy, ExtendedRadiationFluxes& extended, TerrainComputeKernels* kernels)
{
    const int nx = s.grid.nr, ny = s.grid.nth, nz = s.grid.nz;
    if (nx < 3 || ny < 3 || nz < 1 || nz == std::numeric_limits<int>::max() ||
        std::size_t(nx) > std::numeric_limits<std::size_t>::max() / std::size_t(ny) / std::size_t(nz + 1) ||
        !std::isfinite(sx) || !std::isfinite(sy) || std::abs(sx) + std::abs(sy) > .5)
        throw std::invalid_argument("held-column remapping requires a finite bounded shift and complete grid");
    if (kernels && !kernels->supports_column_remap())
        throw std::runtime_error("native held-column kernels are unavailable");
    Worker                       worker{nx, ny, sx, sy, kernels};
    auto&                        pbl       = s.runtime.boundary_layer;
    auto&                        radiation = s.runtime.radiation;
    auto&                        chaos     = s.runtime.chaos;
    std::vector<VolumeCandidate> volumes;
    for (Field3D* field : {&pbl.held_u_force_n,
                           &pbl.held_v_force_n,
                           &pbl.held_theta_inventory_rate_kgk_s,
                           &pbl.held_vapor_mass_rate_kg_s,
                           &pbl.held_cloud_liquid_mass_rate_kg_s,
                           &pbl.held_cloud_ice_mass_rate_kg_s,
                           &radiation.held_cell_power_w,
                           &s.tendencies.dtke_dt_pbl,
                           &chaos.base_du_dt_pbl,
                           &chaos.base_dv_dt_pbl,
                           &chaos.base_dtheta_dt_pbl,
                           &chaos.base_dqv_dt_pbl,
                           &chaos.base_dtke_dt_pbl})
    {
        if (field->empty())
            continue;
        if (field->size_r() != nx || field->size_th() != ny || field->size_z() != nz ||
            field->size() != std::size_t(nx) * ny * nz)
            throw std::invalid_argument("held-column volume shape is invalid");
        Columns source;
        source.values[0].assign(field->data(), field->data() + field->size());
        const auto result = worker.run(source, nz, Storage::Float);
        Field3D    value(nx, ny, nz, 0);
        std::copy(result.rounded.begin(), result.rounded.end(), value.data());
        volumes.push_back({field, std::move(value)});
    }
    std::vector<SurfaceCandidate> surfaces;
    for (std::vector<float>* field : {&pbl.skin_theta_2d,
                                      &pbl.skin_qv_2d,
                                      &s.surface.ustar,
                                      &s.surface.tau_u,
                                      &s.surface.tau_v,
                                      &s.surface.sensible_heat_flux,
                                      &s.surface.latent_heat_flux,
                                      &s.surface.moisture_flux,
                                      &s.surface.pbl_height,
                                      &s.surface.precipitation_rate_mm_h,
                                      &chaos.base_sfc_ustar,
                                      &chaos.base_sfc_tau_u,
                                      &chaos.base_sfc_tau_v,
                                      &chaos.base_sfc_sensible_heat_flux,
                                      &chaos.base_sfc_latent_heat_flux,
                                      &chaos.base_sfc_moisture_flux})
    {
        if (field->empty())
            continue;
        surfaces.push_back({field, worker.scalar(*field, 1, Storage::Float).rounded});
    }
    auto rain = s.surface.accumulated_rainfall_mm, residual = s.surface.accumulated_rainfall_residual_mm;
    if (rain.size() != residual.size())
        throw std::invalid_argument("held-column rainfall compensation is incomplete");
    if (!rain.empty())
    {
        Columns source;
        source.values[0].resize(rain.size());
        for (std::size_t n = 0; n < rain.size(); ++n)
            source.values[0][n] = double(rain[n]) + residual[n];
        const auto result = worker.run(source, 1, Storage::Rain);
        if (sx != 0 || sy != 0)
        {
            rain     = result.rounded;
            residual = result.residual;
        }
    }
    auto slabs = pbl.slab_columns;
    if (!slabs.empty())
    {
        Columns source;
        for (const auto& column : slabs)
        {
            if (!column.initialized)
                throw std::invalid_argument("held-column slabs are not initialized");
            source.values[0].push_back(column.h_m);
            source.values[1].push_back(column.theta_m_k);
            source.values[2].push_back(column.qv_m_kgkg);
        }
        const auto result = worker.run(source, 1, Storage::Slab);
        for (std::size_t n = 0; n < slabs.size(); ++n)
        {
            if (result.values[0][n] <= 0 || result.values[1][n] <= 0 || result.values[2][n] < 0)
                throw std::domain_error("held-column remapping produced invalid slab storage");
            slabs[n] = {result.values[0][n], result.values[1][n], result.values[2][n], true};
        }
    }
    auto&                                     flux = radiation.held_flux_diagnostics;
    const std::array<std::vector<double>*, 4> flux_fields{&flux.upward_longwave_wm2,
                                                          &flux.downward_longwave_wm2,
                                                          &flux.upward_shortwave_wm2,
                                                          &flux.downward_shortwave_wm2};
    ExtendedRadiationFluxes                   next_extended, next_flux;
    for (std::size_t f = 0; f < flux_fields.size(); ++f)
    {
        if (flux.empty())
        {
            if (!flux_fields[f]->empty() || !extended[f].empty())
                throw std::invalid_argument("held radiation interfaces are incomplete");
            continue;
        }
        if (flux.columns_x != std::size_t(nx - 2) || flux.columns_y != std::size_t(ny - 2) ||
            flux.levels != std::size_t(nz) || flux_fields[f]->size() != std::size_t(nx - 2) * (ny - 2) * (nz + 1))
            throw std::invalid_argument("held radiation interface shape is invalid");
        for (double value : *flux_fields[f])
            if (!std::isfinite(value))
                throw std::invalid_argument("held radiation interface is not finite");
        if (extended[f].size() != std::size_t(nx) * ny * (nz + 1))
            throw std::invalid_argument("extended radiation interface shape is invalid");
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
                for (int k = 0; k <= nz; ++k)
                    if (extended[f][(std::size_t(i) * ny + j) * (nz + 1) + k] !=
                        (*flux_fields[f])[(std::size_t(i - 1) * (ny - 2) + j - 1) * (nz + 1) + k])
                        throw std::invalid_argument("extended radiation interfaces have stale owned values");
        next_extended[f] = worker.scalar(extended[f], nz + 1, Storage::Double).values[0];
        next_flux[f].resize(flux_fields[f]->size());
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
                std::copy_n(next_extended[f].data() + (std::size_t(i) * ny + j) * (nz + 1),
                            nz + 1,
                            next_flux[f].data() + (std::size_t(i - 1) * (ny - 2) + j - 1) * (nz + 1));
    }
    double total = radiation.held_atmospheric_power_w;
    for (const auto& candidate : volumes)
        if (candidate.target == &radiation.held_cell_power_w)
        {
            total = 0;
            for (int i = 1; i < nx - 1; ++i)
                for (int j = 1; j < ny - 1; ++j)
                    for (int k = 0; k < nz; ++k)
                        total += candidate.value(i, j, k);
            if (!std::isfinite(total))
                throw std::overflow_error("held atmospheric power sum overflowed");
        }
    for (auto& candidate : volumes)
        *candidate.target = std::move(candidate.value);
    for (auto& candidate : surfaces)
        *candidate.target = std::move(candidate.value);
    s.surface.accumulated_rainfall_mm          = std::move(rain);
    s.surface.accumulated_rainfall_residual_mm = std::move(residual);
    pbl.slab_columns                           = std::move(slabs);
    if (!flux.empty())
    {
        for (std::size_t f = 0; f < flux_fields.size(); ++f)
            *flux_fields[f] = std::move(next_flux[f]);
        extended = std::move(next_extended);
    }
    radiation.held_atmospheric_power_w = total;
}
} // namespace
void remap_held_owners(SimulationStorageState& s, double sx, double sy, ExtendedRadiationFluxes& extended)
{
    remap(s, sx, sy, extended, nullptr);
}
void remap_held_owners(
    SimulationStorageState& s, double sx, double sy, ExtendedRadiationFluxes& extended, TerrainComputeKernels& kernels)
{
    remap(s, sx, sy, extended, &kernels);
}
} // namespace tmv
