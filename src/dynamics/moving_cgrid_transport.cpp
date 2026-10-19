#include "dynamics/moving_cgrid_transport.hpp"
#include "dynamics/moving_cgrid_transport_internal.hpp"

#include "core/infra/physical_grid_metrics.hpp"
#include "terrain/conservative_transport.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>
#include <limits>
#include <stdexcept>

namespace tmv
{
namespace
{
using Index = std::array<int, 3>;

double value(const TerrainField3D& field, const Index& cell)
{
    return field(cell[0], cell[1], cell[2]);
}

double value(const Field3D& field, const Index& cell)
{
    return field(cell[0], cell[1], cell[2]);
}

double dual_mass(const TerrainMetrics3D& geometry, const Field3D& density,
                 Index cell, int component, const Index& lower_bound, const Index& upper_bound)
{
    const double lower = cell[component] >= lower_bound[component]
        ? value(geometry.J, cell) * value(density, cell) : 0.0;
    ++cell[component];
    const double upper = cell[component] < upper_bound[component]
        ? value(geometry.J, cell) * value(density, cell) : 0.0;
    return 0.5 * (lower + upper) * geometry.dx * geometry.dy * geometry.dzeta;
}

double dual_flux(const TerrainContravariantFluxes& flux, int direction,
                 int component, Index face, const Index& lower_bound, const Index& upper_bound)
{
    const TerrainField3D& field = direction == 0 ? flux.x
        : direction == 1 ? flux.y : flux.z;
    if (direction == component && face[component] < lower_bound[component])
    {
        face[component] = lower_bound[component];
        return value(field, face);
    }
    if (direction == component && face[component] >= upper_bound[component])
        return value(field, face);
    const double lower = direction == component || face[component] >= lower_bound[component]
        ? value(field, face) : 0.0;
    ++face[component];
    const double upper = direction == component || face[component] < upper_bound[component]
        ? value(field, face) : 0.0;
    return 0.5 * (lower + upper);
}

void require_flux(const TerrainField3D& field, int nr, int nth, int nz)
{
    if (field.nr != nr || field.nth != nth || field.nz != nz
        || field.values.size() != static_cast<std::size_t>(nr) * nth * nz
        || !std::all_of(field.values.begin(), field.values.end(),
                        [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("moving momentum requires complete finite primal mass transfers");
}
} // namespace

const TerrainMetrics3D& moving_transport_detail::validate(const SimulationStateView&        state,
                                                          const TerrainMetrics3D&           after,
                                                          const Field3D&                    density_after,
                                                          const TerrainContravariantFluxes& flux)
{
    const auto* before = physical_grid_metrics::active_terrain_metrics(state);
    if (before == nullptr || !state.grid.geometry.staggered || state.grid.nr < 4 || state.grid.nth < 4 ||
        state.grid.nz < 2 || state.grid.nr == std::numeric_limits<int>::max() ||
        state.grid.nth == std::numeric_limits<int>::max() || state.grid.nz == std::numeric_limits<int>::max() ||
        before->J.nr != state.grid.nr || before->J.nth != state.grid.nth || before->J.nz != state.grid.nz)
        throw std::invalid_argument("moving momentum requires a complete mapped C-grid");
    terrain_transport::validate_cgrid_state(*before, state.fields.rho, state.fields.u, state.fields.v, state.fields.w);
    terrain_transport::validate_cgrid_state(after, density_after, state.fields.u, state.fields.v, state.fields.w);
    if (before->dx != after.dx || before->dy != after.dy || before->dzeta != after.dzeta)
        throw std::invalid_argument("moving momentum cannot change the logical grid partition");
    require_flux(flux.x, state.grid.nr + 1, state.grid.nth, state.grid.nz);
    require_flux(flux.y, state.grid.nr, state.grid.nth + 1, state.grid.nz);
    require_flux(flux.z, state.grid.nr, state.grid.nth, state.grid.nz + 1);
    for (int i = 0; i < state.grid.nr; ++i)
        for (int j = 0; j < state.grid.nth; ++j)
            if (flux.z(i, j, 0) != 0.0)
                throw std::invalid_argument("moving momentum requires an impermeable ground");

    return *before;
}

MovingCGridMomentumTransport prepare_moving_cgrid_momentum_transport(const SimulationStateView&        state,
                                                                     const TerrainMetrics3D&           after,
                                                                     const Field3D&                    density_after,
                                                                     const TerrainContravariantFluxes& flux,
                                                                     bool include_boundary_faces)
{
    if (state.compute_runtime != nullptr &&
        (state.compute_runtime->active_kind != ComputeBackendKind::Cpu || state.compute_runtime->fallback_active))
        throw std::invalid_argument("moving momentum requires CPU-owned execution");
    const auto*                         before = &moving_transport_detail::validate(state, after, density_after, flux);
    MovingCGridMomentumTransport        result{state.fields.u, state.fields.v, state.fields.w, {}};
    const std::array<const Field3D*, 3> input{&state.fields.u, &state.fields.v, &state.fields.w};
    const std::array<Field3D*, 3>       output{&result.u, &result.v, &result.w};
    const auto                          domain = physical_scalar_domain(state.grid);
    const Index                         scalar_begin{domain.i_begin, domain.j_begin, domain.k_begin};
    const Index                         scalar_end{domain.i_end, domain.j_end, domain.k_end};
    const auto                          surface = physical_grid_metrics::surface_velocity_ms(state);
    for (int component = 0; component < 3; ++component)
    {
        Index begin = scalar_begin, end = scalar_end;
        if (include_boundary_faces)
        {
            if (component < 2) --begin[component];
        }
        else
            --end[component];
        using Partial = std::array<long double, 11>;
        const auto slabs = static_cast<std::size_t>(end[0] - begin[0]);
        std::vector<Partial> partials(slabs);
        std::vector<std::exception_ptr> failures(slabs);
        const auto wind = [&](Index cell)
        {
            if (component == 2 && cell[2] == -1)
                return terrain_transport::cgrid_ground_vertical_velocity(
                    *before, state.fields.u, state.fields.v, cell[0], cell[1],
                    surface.u_ms, surface.v_ms);
            cell[0] = std::clamp(cell[0], 0, state.grid.nr - 1);
            cell[1] = std::clamp(cell[1], 0, state.grid.nth - 1);
            cell[2] = std::clamp(cell[2], 0, state.grid.nz - 1);
            return value(*input[component], cell);
        };
        #pragma omp parallel for schedule(static)
        for (int i = begin[0]; i < end[0]; ++i)
        {
            const auto slab = static_cast<std::size_t>(i - begin[0]);
            try
            {
                auto& sum = partials[slab];
                for (int j = begin[1]; j < end[1]; ++j)
                    for (int k = begin[2]; k < end[2]; ++k)
                    {
                        const Index cell{i, j, k};
                        const double old_mass = dual_mass(*before, state.fields.rho, cell, component, scalar_begin, scalar_end);
                        const double new_mass = dual_mass(after, density_after, cell, component, scalar_begin, scalar_end);
                        const double old_wind = wind(cell);
                        if (!std::isfinite(old_mass) || !std::isfinite(new_mass)
                            || old_mass <= 0.0 || new_mass <= 0.0)
                            throw std::domain_error("moving momentum requires positive finite dual masses");
                        double mass_change = 0.0, impulse_change = 0.0;
                        double linear_work = 0.0, outgoing_mass = 0.0;
                        for (int direction = 0; direction < 3; ++direction)
                            for (int side : {-1, 1})
                            {
                                Index face = cell;
                                if (side > 0) ++face[direction];
                                const double transfer = dual_flux(flux, direction, component, face, scalar_begin, scalar_end);
                                const double outward = side * transfer;
                                if (outward == 0.0) continue;
                                Index donor = cell;
                                if (outward < 0.0) donor[direction] += side;
                                double face_wind = wind(donor);
                                if (include_boundary_faces && direction == component
                                    && (face[direction] == scalar_begin[direction]
                                        || face[direction] == scalar_end[direction] - 1))
                                {
                                    Index previous=donor, next=donor;
                                    --previous[direction];
                                    ++next[direction];
                                    const double a=face_wind-wind(previous);
                                    const double b=wind(next)-face_wind;
                                    const double slope=a*b>0.0
                                        ? std::copysign(std::min({0.5*std::abs(a+b),2.0*std::abs(a),2.0*std::abs(b)}),a)
                                        : 0.0;
                                    face_wind+=(transfer>=0.0 ? 0.5 : -0.5)*slope;
                                }
                                mass_change -= outward;
                                impulse_change -= outward * face_wind;
                                linear_work -= outward * (old_wind * face_wind - 0.5 * old_wind * old_wind);
                                outgoing_mass += std::max(0.0, outward);
                                if ((side < 0 && cell[direction] == begin[direction])
                                    || (side > 0 && cell[direction] + 1 == end[direction]))
                                {
                                    sum[2] += outward * face_wind;
                                    sum[6] += 0.5 * outward * face_wind * face_wind;
                                }
                            }
                        if (outgoing_mass > old_mass * (1.0 + 64.0 * std::numeric_limits<double>::epsilon()))
                            throw std::domain_error("moving momentum interval exhausts a donor dual volume");
                        const double mass_rounding = new_mass - old_mass - mass_change;
                        const double mass_scale = std::max({old_mass, new_mass, std::abs(mass_change)});
                        const double mass_tolerance =
                            8.0 * std::numeric_limits<float>::epsilon() * new_mass
                            + 64.0 * std::numeric_limits<double>::epsilon() * mass_scale;
                        if (std::abs(mass_rounding) > mass_tolerance)
                            throw std::invalid_argument("moving momentum density and retained mass transfers disagree");
                        const double requested_impulse = old_mass * old_wind + impulse_change;
                        const double exact_wind = requested_impulse / new_mass;
                        const float stored_wind = static_cast<float>(exact_wind);
                        if (!std::isfinite(stored_wind))
                            throw std::overflow_error("moving momentum velocity cannot be represented");
                        const double old_kinetic = 0.5 * old_mass * old_wind * old_wind;
                        const double exact_kinetic = 0.5 * new_mass * exact_wind * exact_wind;
                        const double stored_kinetic = 0.5 * new_mass * stored_wind * stored_wind;
                        sum[0] += old_mass * old_wind;
                        sum[1] += new_mass * stored_wind;
                        sum[3] += new_mass * stored_wind - requested_impulse;
                        sum[4] += old_kinetic;
                        sum[5] += stored_kinetic;
                        sum[7] += linear_work;
                        sum[8] += 0.5 * new_mass * (exact_wind - old_wind) * (exact_wind - old_wind);
                        sum[9] -= 0.5 * old_wind * old_wind * mass_rounding;
                        sum[10] += stored_kinetic - exact_kinetic;
                        (*output[component])(i, j, k) = stored_wind;
                    }
            }
            catch (...)
            {
                failures[slab] = std::current_exception();
            }
        }
        Partial sum{};
        for (std::size_t slab = 0; slab < slabs; ++slab)
        {
            if (failures[slab]) std::rethrow_exception(failures[slab]);
            for (std::size_t n = 0; n < sum.size(); ++n)
                sum[n] += partials[slab][n];
        }
        for (long double x : sum)
            if (!std::isfinite(static_cast<double>(x)))
                throw std::overflow_error("moving momentum transport ledger overflowed");
        result.ledger[component] = {
            static_cast<double>(sum[0]), static_cast<double>(sum[1]),
            static_cast<double>(sum[2]), static_cast<double>(sum[3]),
            static_cast<double>(sum[4]), static_cast<double>(sum[5]),
            static_cast<double>(sum[6]), static_cast<double>(sum[7] + sum[6]),
            static_cast<double>(sum[8]), static_cast<double>(sum[9]),
            static_cast<double>(sum[10])};
    }
    return result;
}
} // namespace tmv
