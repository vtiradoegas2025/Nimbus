#pragma once

#include "compute/dispatch/advection.hpp"
#include "compute/kernel_abi.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "numerics/advection/advection_cartesian.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nimbus::compute
{
class CartesianScalarCapture
{
public:
    static constexpr int maximum_scalars = 16;

    bool prepare(const ConservativeCartesianCgridTransportDispatch& d)
    {
        if (d.n_scalars < 1 || d.n_scalars > maximum_scalars || d.nx < 3 || d.ny < 3 || d.nz < 1 || !d.u || !d.v ||
            !d.w || !d.rho_data || !d.q_in || !d.q_out || !d.dt_kappas || !std::isfinite(d.inv_dx) || d.inv_dx <= 0 ||
            !std::isfinite(d.inv_dy) || d.inv_dy <= 0 || !std::isfinite(d.inv_dz) || d.inv_dz <= 0 ||
            !std::isfinite(d.dt_step) || d.dt_step <= 0 || !std::isfinite(d.rho_floor) || d.rho_floor <= 0 ||
            !std::isfinite(d.frame_velocity_u_ms) || !std::isfinite(d.frame_velocity_v_ms))
            return false;
        const auto maximum = std::numeric_limits<std::size_t>::max();
        if (std::size_t(d.nx) > maximum / std::size_t(d.ny) ||
            std::size_t(d.nx) * d.ny > maximum / std::size_t(d.nz) / sizeof(float))
            return false;
        const std::size_t cells       = std::size_t(d.nx) * d.ny * d.nz;
        const std::size_t field_bytes = cells * sizeof(float);
        const auto        overlaps    = [field_bytes](const float* a, const float* b)
        {
            const auto aa = reinterpret_cast<std::uintptr_t>(a);
            const auto bb = reinterpret_cast<std::uintptr_t>(b);
            return a && b && (aa <= bb ? bb - aa : aa - bb) < field_bytes;
        };
        for (int s = 0; s < d.n_scalars; ++s)
        {
            if (!d.q_in[s] || !d.q_out[s] || d.dt_kappas[s] != 0 ||
                (d.q_in[s] != d.q_out[s] && overlaps(d.q_in[s], d.q_out[s])))
                return false;
            for (const float* owner :
                 {d.u, d.v, d.w, d.rho_data, d.density_begin, d.mass_flux_x, d.mass_flux_y, d.mass_flux_z})
                if (overlaps(d.q_out[s], owner))
                    return false;
            for (int prior = 0; prior < s; ++prior)
                if (overlaps(d.q_out[s], d.q_out[prior]) || overlaps(d.q_out[s], d.q_in[prior]) ||
                    overlaps(d.q_out[prior], d.q_in[s]) ||
                    (d.host_boundary_reports && d.host_boundary_reports[s] &&
                     d.host_boundary_reports[s] == d.host_boundary_reports[prior]) ||
                    (d.host_energy_reports && d.host_energy_reports[s] &&
                     d.host_energy_reports[s] == d.host_energy_reports[prior]))
                    return false;
        }
        for (std::size_t index = 0; index < cells; ++index)
        {
            if (!std::isfinite(d.u[index]) || !std::isfinite(d.v[index]) || !std::isfinite(d.w[index]) ||
                !std::isfinite(d.rho_data[index]) || d.rho_data[index] <= 0)
                return false;
            if (d.use_supplied_mass_flux &&
                (!d.density_begin || !d.mass_flux_x || !d.mass_flux_y || !d.mass_flux_z ||
                 !std::isfinite(d.density_begin[index]) || d.density_begin[index] <= 0 ||
                 !std::isfinite(d.mass_flux_x[index]) || !std::isfinite(d.mass_flux_y[index]) ||
                 !std::isfinite(d.mass_flux_z[index])))
                return false;
            for (int s = 0; s < d.n_scalars; ++s)
                if (!std::isfinite(d.q_in[s][index]))
                    return false;
        }
        const std::size_t physical_faces =
            2 * (std::size_t(d.nx - 2) + d.ny - 2) * d.nz + std::size_t(d.nx - 2) * (d.ny - 2);
        for (int scalar = 0; scalar < d.n_scalars; ++scalar)
            if ((d.host_boundary_reports && d.host_boundary_reports[scalar]) ||
                (d.host_energy_reports && d.host_energy_reports[scalar]))
            {
                if (physical_faces > std::size_t(std::numeric_limits<int>::max()) - elements_)
                    return false;
                reports_[scalar] = {int(elements_), report_count_++};
                elements_ += physical_faces;
            }
        for (int s = 0; s < d.n_scalars; ++s)
        {
            auto* target = d.host_interface_registers ? d.host_interface_registers[s] : nullptr;
            if (!target)
                continue;
            if (!d.execution_state || !target->configured() ||
                tmv::physical_grid_metrics::active_terrain_metrics(*d.execution_state) != nullptr)
                return false;
            for (int prior = 0; prior < s; ++prior)
                if (entries_[prior].target == target)
                    return false;
            try
            {
                target->require_compatible(d.nx, d.ny, d.nz);
            }
            catch (const std::invalid_argument&)
            {
                return false;
            }
            const auto& grid = d.execution_state->grid;
            if (grid.nr != d.nx || grid.nth != d.ny || grid.nz != d.nz || !std::isfinite(grid.dr) || grid.dr <= 0 ||
                !std::isfinite(grid.dtheta) || grid.dtheta <= 0 || !std::isfinite(grid.dz) || grid.dz <= 0 ||
                float(1.0 / grid.dr) != d.inv_dx || float(1.0 / grid.dtheta) != d.inv_dy ||
                float(1.0 / grid.dz) != d.inv_dz || !std::isfinite(d.dt_step_host) || d.dt_step_host <= 0 ||
                float(d.dt_step_host) != d.dt_step)
                return false;
            const auto&       f    = target->footprint();
            const std::size_t size = 2 * (std::size_t(f.i_end - f.i_begin) + std::size_t(f.j_end - f.j_begin)) *
                                     std::size_t(f.k_end - f.k_begin);
            if (size > std::size_t(std::numeric_limits<int>::max()) - elements_)
                return false;
            entries_[s] = {target, *target, int(elements_)};
            elements_ += size;
        }
        return true;
    }

    [[nodiscard]] std::size_t bytes() const noexcept
    {
        return std::max<std::size_t>(1, elements_) * sizeof(float);
    }

    [[nodiscard]] bool enabled() const noexcept
    {
        return elements_ != 0;
    }

    void configure(int scalar, abi::AdvectUpdateCartesianCgridPushConstants& constants) const noexcept
    {
        const auto& entry        = entries_[scalar];
        constants.capture_offset = entry.offset;
        constants.report_offset  = reports_[scalar].offset;
        if (!entry.target)
            return;
        const auto& f             = entry.candidate.footprint();
        constants.capture_i_begin = f.i_begin;
        constants.capture_i_end   = f.i_end;
        constants.capture_j_begin = f.j_begin;
        constants.capture_j_end   = f.j_end;
        constants.capture_k_begin = f.k_begin;
        constants.capture_k_end   = f.k_end;
    }

    int report_count() const noexcept
    {
        return report_count_;
    }
    int report_index(int scalar) const noexcept
    {
        return reports_[scalar].index;
    }
    const float* boundary_fluxes(int scalar, const float* fluxes) const noexcept
    {
        return reports_[scalar].offset < 0 ? nullptr : fluxes + reports_[scalar].offset;
    }

    void stage(const ConservativeCartesianCgridTransportDispatch& d, const float* fluxes)
    {
        if (!enabled())
            return;
        for (int s = 0; s < d.n_scalars; ++s)
        {
            auto& entry = entries_[s];
            if (!entry.target)
                continue;
            // The interval clock is FP64; inventory follows the device timestep.
            const double time_weight = double(d.dt_step) / d.dt_step_host;
            const auto&  f           = entry.candidate.footprint();
            const int    nk          = f.k_end - f.k_begin;
            const int    x_size      = (f.j_end - f.j_begin) * nk;
            const int    y_size      = (f.i_end - f.i_begin) * nk;
            entry.candidate.stage_scalar_flux(
                [&](int axis, int i, int j, int k)
                {
                    const int bin = axis == 0 ? (i == f.i_begin - 1 ? 0 : x_size) + (j - f.j_begin) * nk
                                              : 2 * x_size + (j == f.j_begin - 1 ? 0 : y_size) + (i - f.i_begin) * nk;
                    return time_weight * double(fluxes[entry.offset + bin + k - f.k_begin]);
                },
                d.dt_step_host,
                d.execution_state->grid.dr,
                d.execution_state->grid.dtheta,
                d.execution_state->grid.dz);
            entry.candidate.validate_staged_scalar_commit();
            entry.candidate.commit_staged_scalar();
        }
    }

    void commit() noexcept
    {
        for (auto& entry : entries_)
            if (entry.target)
                *entry.target = std::move(entry.candidate);
    }

private:
    struct Entry
    {
        tmv::CartesianInterfaceFluxRegister* target = nullptr;
        tmv::CartesianInterfaceFluxRegister  candidate;
        int                                  offset = -1;
    };
    struct Report
    {
        int offset = -1;
        int index  = -1;
    };
    std::array<Report, maximum_scalars> reports_{};
    int                                 report_count_ = 0;
    std::array<Entry, maximum_scalars>  entries_{};
    std::size_t                         elements_ = 0;
};

template <class Candidate, class Requested, class Copy>
void publish_cartesian_scalar_batch(const ConservativeCartesianCgridTransportDispatch& d,
                                    CartesianScalarCapture&                            capture,
                                    const float*                                       fluxes,
                                    std::size_t                                        cells,
                                    const Candidate&                                   candidate,
                                    tmv::FlatCGridDeviceScalarAttributionWorkspace&    workspace,
                                    const Requested&                                   requested,
                                    const Copy&                                        copy)
{
    capture.stage(d, fluxes);
    for (int s = 0; s < d.n_scalars; ++s)
        for (std::size_t index = 0; index < cells; ++index)
            if (!std::isfinite(candidate(s)[index]))
                throw std::runtime_error("native Cartesian scalar transport produced a non-finite field");

    std::array<std::vector<float>, CartesianScalarCapture::maximum_scalars>                      saved_fields;
    std::array<tmv::ScalarBoundaryTransportReport, CartesianScalarCapture::maximum_scalars>      saved_boundary;
    std::array<tmv::ScalarMoistEnergyTransportReportV1, CartesianScalarCapture::maximum_scalars> saved_energy;
    bool                                                                                         reporting = false;
    for (int s = 0; s < d.n_scalars; ++s)
    {
        if (d.host_boundary_reports && d.host_boundary_reports[s])
        {
            saved_boundary[s] = *d.host_boundary_reports[s];
            reporting         = true;
        }
        if (d.host_energy_reports && d.host_energy_reports[s])
        {
            saved_energy[s] = *d.host_energy_reports[s];
            reporting       = true;
        }
    }
    if (reporting)
        for (int s = 0; s < d.n_scalars; ++s)
            saved_fields[s].assign(d.q_out[s], d.q_out[s] + cells);
    try
    {
        for (int s = 0; s < d.n_scalars; ++s)
        {
            auto*      boundary = d.host_boundary_reports ? d.host_boundary_reports[s] : nullptr;
            auto*      energy   = d.host_energy_reports ? d.host_energy_reports[s] : nullptr;
            const auto kind     = d.host_energy_kinds ? d.host_energy_kinds[s] : tmv::ScalarMoistEnergyKind::None;
            if (boundary || energy)
                tmv::attribute_flat_cgrid_device_scalar_commit(d.execution_state ? *d.execution_state
                                                                                 : global_simulation_state(),
                                                               d.q_in[s],
                                                               candidate(s),
                                                               d.density_begin,
                                                               d.rho_data,
                                                               d.mass_flux_x,
                                                               d.mass_flux_y,
                                                               d.mass_flux_z,
                                                               d.nx,
                                                               d.ny,
                                                               d.nz,
                                                               d.inv_dx,
                                                               d.inv_dy,
                                                               d.inv_dz,
                                                               d.dt_step,
                                                               d.positive_definite && d.positive_definite[s] != 0,
                                                               boundary,
                                                               kind,
                                                               energy,
                                                               workspace,
                                                               capture.boundary_fluxes(s, fluxes),
                                                               requested(s));
            // Moist energy is bilinear: each report sees preceding scalar commits.
            copy(d.q_out[s], candidate(s), cells * sizeof(float));
        }
    }
    catch (...)
    {
        for (int s = 0; s < d.n_scalars; ++s)
        {
            if (reporting)
                std::memcpy(d.q_out[s], saved_fields[s].data(), cells * sizeof(float));
            if (d.host_boundary_reports && d.host_boundary_reports[s])
                *d.host_boundary_reports[s] = saved_boundary[s];
            if (d.host_energy_reports && d.host_energy_reports[s])
                *d.host_energy_reports[s] = saved_energy[s];
        }
        throw;
    }
    capture.commit();
}
} // namespace nimbus::compute
