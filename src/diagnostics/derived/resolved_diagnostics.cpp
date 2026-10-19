#include "diagnostics/resolved_diagnostics.hpp"

#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/infra/grid_coordinates.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/physical_derivative_kernel.hpp"
#include "diagnostics/physical_reference.hpp"
#include "dynamics/coriolis.hpp"
#include "terrain/conservative_transport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv::diagnostics
{
namespace
{
Field3D pressure_perturbation_field(const SimulationStateView& state)
{
    physical_grid_metrics::validate(state);
    if (state.fields.p.size_r() != state.grid.nr || state.fields.p.size_th() != state.grid.nth ||
        state.fields.p.size_z() != state.grid.nz || state.base.p0.empty())
        throw std::invalid_argument("pressure perturbation requires complete pressure and reference fields");
    Field3D     result(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    const auto* terrain = physical_grid_metrics::active_terrain_metrics(state);
    for (int i = 0; i < state.grid.nr; ++i)
        for (int j = 0; j < state.grid.nth; ++j)
            for (int k = 0; k < state.grid.nz; ++k)
            {
                const float pressure = state.fields.p(i, j, k);
                const float reference =
                    static_cast<float>(reference_at_cell(state.base.p0, terrain, state.grid.dz, i, j, k));
                const float value = pressure - reference;
                if (!std::isfinite(pressure) || pressure <= 0.0f || !std::isfinite(value))
                    throw std::invalid_argument("pressure perturbation is not finite and representable");
                result(i, j, k) = value;
            }
    return result;
}

class PhysicalSnapshot
{
public:
    explicit PhysicalSnapshot(const SimulationStateView& state)
        : state_(state), domain_(physical_scalar_domain(state.grid)),
          terrain_(physical_grid_metrics::active_terrain_metrics(state))
    {
        physical_grid_metrics::validate(state);
        if (state.grid.geometry.coord != CoordinateSystem::Cartesian || !state.diagnostics)
            throw std::invalid_argument("resolved diagnostics require Cartesian geometry and owned diagnostic storage");
        nx = domain_.i_end - domain_.i_begin;
        ny = domain_.j_end - domain_.j_begin;
        nz = domain_.k_end - domain_.k_begin;
        if (nx < 1 || ny < 1 || nz < 1)
            throw std::invalid_argument("resolved diagnostics require a nonempty physical domain");
        for (const Field3D* field :
             {&state.fields.u, &state.fields.v, &state.fields.w, &state.fields.rho, &state.fields.p})
            require_field(*field);
        if (state.configuration && state.budget_level == BudgetLevel::Root)
        {
            using Boundary     = init::config::LateralBC;
            const auto& config = *state.configuration;
            const bool  west   = config.lateral_bc_west == Boundary::Periodic;
            const bool  east   = config.lateral_bc_east == Boundary::Periodic;
            const bool  south  = config.lateral_bc_south == Boundary::Periodic;
            const bool  north  = config.lateral_bc_north == Boundary::Periodic;
            if (west != east || south != north)
                throw std::invalid_argument("diagnostic periodic boundaries must be paired");
            periodic_x = west;
            periodic_y = south;
        }
        ground_velocity_ = physical_grid_metrics::surface_velocity_ms(state);
        const auto count = static_cast<std::size_t>(nx) * ny * nz;
        for (auto& field : velocity)
            field.resize(count);
        height.resize(count);
        density.resize(count);
        volume.resize(count);
        pressure.resize(count);
        perturbation.resize(count);
        const bool reference_available = !state.base.p0.empty();
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const auto n  = index(i, j, k);
                    const int  ii = i + domain_.i_begin, jj = j + domain_.j_begin, kk = k + domain_.k_begin;
                    height[n]   = physical_grid_metrics::scalar_height(state, ii, jj, kk);
                    volume[n]   = physical_grid_metrics::cell_volume(state, ii, jj, kk);
                    density[n]  = state.fields.rho(ii, jj, kk);
                    pressure[n] = state.fields.p(ii, jj, kk);
                    if (!std::isfinite(density[n]) || density[n] <= 0.0 || !std::isfinite(pressure[n]) ||
                        pressure[n] <= 0.0)
                        throw std::invalid_argument(
                            "resolved diagnostics require positive finite density and pressure");
                    perturbation[n] = pressure[n] - (reference_available ? reference(state.base.p0, i, j, k) : 0.0);
                    if (state.grid.geometry.staggered)
                    {
                        velocity[0][n] = .5 * (double(state.fields.u(ii - 1, jj, kk)) + state.fields.u(ii, jj, kk));
                        velocity[1][n] = .5 * (double(state.fields.v(ii, jj - 1, kk)) + state.fields.v(ii, jj, kk));
                        const double lower =
                            kk > 0     ? state.fields.w(ii, jj, kk - 1)
                            : terrain_ ? terrain_transport::cgrid_ground_vertical_velocity(*terrain_,
                                                                                           state.fields.u,
                                                                                           state.fields.v,
                                                                                           ii,
                                                                                           jj,
                                                                                           ground_velocity_.u_ms,
                                                                                           ground_velocity_.v_ms)
                                       : 0.0;
                        velocity[2][n] = .5 * (lower + state.fields.w(ii, jj, kk));
                    }
                    else
                    {
                        velocity[0][n] = state.fields.u(ii, jj, kk);
                        velocity[1][n] = state.fields.v(ii, jj, kk);
                        velocity[2][n] = state.fields.w(ii, jj, kk);
                    }
                    for (const auto& component : velocity)
                        if (!std::isfinite(component[n]))
                            throw std::invalid_argument("resolved diagnostics require finite physical velocity");
                }
    }

    void require_field(const Field3D& field) const
    {
        if (field.size_r() != state_.grid.nr || field.size_th() != state_.grid.nth || field.size_z() != state_.grid.nz)
            throw std::invalid_argument("resolved diagnostic field shape differs from its grid");
    }

    std::size_t index(int i, int j, int k) const
    {
        return (static_cast<std::size_t>(i) * ny + j) * nz + k;
    }

    double sample(const Field3D& field, int i, int j, int k) const
    {
        return field(i + domain_.i_begin, j + domain_.j_begin, k + domain_.k_begin);
    }

    double reference(const std::vector<double>& profile, int i, int j, int k) const
    {
        return reference_at_cell(
            profile, terrain_, state_.grid.dz, i + domain_.i_begin, j + domain_.j_begin, k + domain_.k_begin);
    }

    double reference_density(int i, int j, int k) const
    {
        return terrain_ ? terrain_transport::reference_density_fp32(
                              state_.base.p0, state_.base.theta0, height[index(i, j, k)], state_.grid.dz)
                        : reference(state_.base.rho0, i, j, k);
    }

    PhysicalGradientPoint gradient(const std::vector<double>& field, int i, int j, int k) const
    {
        const auto vertical = [&](const std::vector<double>& values)
        {
            const double center = values[index(i, j, k)];
            return detail::derivative_at_index(
                k,
                nz,
                [&](int z) { return height[index(i, j, z)]; },
                [&](int z) { return values[index(i, j, z)] - center; });
        };
        const auto horizontal = [&](const std::vector<double>& values, int axis)
        {
            const int    position = axis == 0 ? i : j;
            const int    count    = axis == 0 ? nx : ny;
            const bool   periodic = axis == 0 ? periodic_x : periodic_y;
            const double spacing  = axis == 0 ? state_.grid.dr : state_.grid.dtheta;
            const auto   at       = [&](int n) { return axis == 0 ? values[index(n, j, k)] : values[index(i, n, k)]; };
            if (periodic && count > 1)
                return (at((position + 1) % count) - at((position + count - 1) % count)) / (2.0 * spacing);
            const double center = at(position);
            return detail::derivative_at_index(
                position,
                count,
                [&](int n) { return static_cast<double>(n) * spacing; },
                [&](int n) { return at(n) - center; });
        };
        const double z = vertical(field);
        return {horizontal(field, 0) - horizontal(height, 0) * z, horizontal(field, 1) - horizontal(height, 1) * z, z};
    }

    double x(int i) const
    {
        const int n = i + domain_.i_begin;
        return state_.grid.geometry.r.size() == static_cast<std::size_t>(state_.grid.nr)
                   ? state_.grid.geometry.r[static_cast<std::size_t>(n)]
                   : grid_coordinates::scalar_center(n, state_.grid.dr);
    }

    double vertical_strain(int i, int j, int k, const PhysicalGradientPoint& centered_gradient) const
    {
        if (!state_.grid.geometry.staggered)
            return centered_gradient.z;
        const int    ii = i + domain_.i_begin, jj = j + domain_.j_begin, kk = k + domain_.k_begin;
        const double lower =
            kk > 0 ? state_.fields.w(ii, jj, kk - 1)
            : terrain_
                ? terrain_transport::cgrid_ground_vertical_velocity(
                      *terrain_, state_.fields.u, state_.fields.v, ii, jj, ground_velocity_.u_ms, ground_velocity_.v_ms)
                : 0.0;
        return (state_.fields.w(ii, jj, kk) - lower) / physical_grid_metrics::layer_thickness(state_, ii, jj, kk);
    }

    double y(int j) const
    {
        const int n = j + domain_.j_begin;
        return state_.grid.geometry.theta.size() == static_cast<std::size_t>(state_.grid.nth)
                   ? state_.grid.geometry.theta[static_cast<std::size_t>(n)]
                   : grid_coordinates::scalar_center(n, state_.grid.dtheta);
    }

    void store(Field3D& target, const std::vector<double>& values) const
    {
        target.resize(state_.grid.nr, state_.grid.nth, state_.grid.nz, 0.0f);
        for (int i = 0; i < state_.grid.nr; ++i)
            for (int j = 0; j < state_.grid.nth; ++j)
                for (int k = 0; k < state_.grid.nz; ++k)
                {
                    const auto local = [](int value, int start, int count, bool periodic)
                    {
                        const int relative = value - start;
                        return periodic ? (relative % count + count) % count : std::clamp(relative, 0, count - 1);
                    };
                    const double value  = values[index(local(i, domain_.i_begin, nx, periodic_x),
                                                       local(j, domain_.j_begin, ny, periodic_y),
                                                       local(k, domain_.k_begin, nz, false))];
                    const float  stored = static_cast<float>(value);
                    if (!std::isfinite(value) || !std::isfinite(stored))
                        throw std::overflow_error("resolved diagnostic cannot be represented as a finite field");
                    target(i, j, k) = stored;
                }
    }

    int                                nx = 0, ny = 0, nz = 0;
    bool                               periodic_x = false, periodic_y = false;
    std::array<std::vector<double>, 3> velocity;
    std::vector<double>                height, density, volume, pressure, perturbation;

private:
    const SimulationStateView& state_;
    PhysicalCellDomain         domain_;
    const TerrainMetrics3D*    terrain_;
    domain::HorizontalVelocity ground_velocity_;
};

double advect(const std::array<double, 3>& velocity, const PhysicalGradientPoint& gradient)
{
    return -(velocity[0] * gradient.x + velocity[1] * gradient.y + velocity[2] * gradient.z);
}
} // namespace

void compute_resolved_diagnostics(const SimulationStateView&       state,
                                  const ResolvedDiagnosticRequest& request,
                                  ResolvedDiagnosticReport*        report)
{
    if (!request.vorticity && !request.angular_momentum && !request.pressure_perturbation &&
        !request.pressure_components)
        return;
    if (!state.diagnostics)
        throw std::invalid_argument("resolved diagnostics require owned output storage");
    if (request.pressure_perturbation && !request.vorticity && !request.angular_momentum &&
        !request.pressure_components)
    {
        auto candidate = pressure_perturbation_field(state);
        using std::swap;
        swap(state.diagnostics->p_prime, candidate);
        if (report)
            *report = {};
        return;
    }
    PhysicalSnapshot                   grid(state);
    const std::size_t                  count = grid.height.size();
    DiagnosticFieldStorage             candidate;
    ResolvedDiagnosticReport           candidate_report;
    std::array<std::vector<double>, 6> vorticity;
    std::array<std::vector<double>, 2> angular;
    std::array<std::vector<double>, 3> dynamic_force, buoyancy_force;
    PressureProjectionGrid             projection_grid;
    if (request.vorticity)
        for (auto& field : vorticity)
            field.resize(count);
    if (request.angular_momentum)
        for (auto& field : angular)
            field.resize(count);
    if (request.pressure_components)
    {
        for (auto* fields : {&dynamic_force, &buoyancy_force})
            for (auto& field : *fields)
                field.assign(count, 0.0);
        projection_grid = {grid.nx,
                           grid.ny,
                           grid.nz,
                           state.grid.dr,
                           state.grid.dtheta,
                           grid.periodic_x,
                           grid.periodic_y,
                           grid.height,
                           grid.volume,
                           grid.density,
                           std::vector<double>(count)};
        for (const Field3D* field : {&state.fields.qv,
                                     &state.fields.qc,
                                     &state.fields.qr,
                                     &state.fields.qi,
                                     &state.fields.qs,
                                     &state.fields.qg,
                                     &state.fields.qh})
            grid.require_field(*field);
    }
    if ((request.pressure_perturbation || request.pressure_components) && state.base.p0.empty())
        throw std::invalid_argument("pressure diagnostics require a pressure reference profile");

    CartesianCoriolisProfile coriolis;
    if (request.angular_momentum && state.configuration && state.configuration->coriolis_enabled)
    {
        if (!state.domain_frame)
            throw std::invalid_argument("angular momentum diagnostics require their Coriolis frame");
        coriolis =
            make_cartesian_coriolis_profile(true,
                                            state.configuration->coriolis_mode == init::config::CoriolisMode::BetaPlane,
                                            state.configuration->coriolis_latitude_deg,
                                            *state.domain_frame,
                                            state.time_s);
    }
    const double center_x = .5 * (grid.x(0) + grid.x(grid.nx - 1));
    const double center_y = .5 * (grid.y(0) + grid.y(grid.ny - 1));
    for (int i = 0; i < grid.nx; ++i)
        for (int j = 0; j < grid.ny; ++j)
            for (int k = 0; k < grid.nz; ++k)
            {
                const auto                  n = grid.index(i, j, k);
                const std::array<double, 3> velocity{grid.velocity[0][n], grid.velocity[1][n], grid.velocity[2][n]};
                const auto                  gu = grid.gradient(grid.velocity[0], i, j, k);
                const auto                  gv = grid.gradient(grid.velocity[1], i, j, k);
                const auto                  gw = grid.gradient(grid.velocity[2], i, j, k);
                if (request.vorticity)
                {
                    const auto gr   = grid.gradient(grid.density, i, j, k);
                    vorticity[0][n] = gw.y - gv.z;
                    vorticity[1][n] = gu.z - gw.x;
                    vorticity[2][n] = gv.x - gu.y;
                    vorticity[3][n] = vorticity[2][n] * grid.vertical_strain(i, j, k, gw);
                    vorticity[4][n] = vorticity[0][n] * gw.x + vorticity[1][n] * gw.y;
                    const auto gp   = grid.gradient(grid.pressure, i, j, k);
                    vorticity[5][n] = (gr.x * gp.y - gr.y * gp.x) / (grid.density[n] * grid.density[n]);
                }
                if (request.angular_momentum)
                {
                    const double x = grid.x(i) - center_x, y = grid.y(j) - center_y;
                    const auto   gp          = grid.gradient(grid.perturbation, i, j, k);
                    const double f           = coriolis.at_grid_y(grid.y(j));
                    const double reference_u = state.base.u0.empty() ? 0.0 : grid.reference(state.base.u0, i, j, k);
                    const double reference_v = state.base.v0.empty() ? 0.0 : grid.reference(state.base.v0, i, j, k);
                    const double du = advect(velocity, gu) - gp.x / grid.density[n] + f * (velocity[1] - reference_v);
                    const double dv = advect(velocity, gv) - gp.y / grid.density[n] - f * (velocity[0] - reference_u);
                    angular[0][n]   = x * velocity[1] - y * velocity[0];
                    angular[1][n]   = x * dv - y * du;
                }
                if (request.pressure_components)
                {
                    dynamic_force[0][n]            = advect(velocity, gu);
                    dynamic_force[1][n]            = advect(velocity, gv);
                    dynamic_force[2][n]            = advect(velocity, gw);
                    const double reference_density = grid.reference_density(i, j, k);
                    const double reference_vapor   = grid.reference(state.base.qv0, i, j, k);
                    double       condensate        = 0.0;
                    for (const Field3D* field : {&state.fields.qc,
                                                 &state.fields.qr,
                                                 &state.fields.qi,
                                                 &state.fields.qs,
                                                 &state.fields.qg,
                                                 &state.fields.qh})
                    {
                        const double value = grid.sample(*field, i, j, k);
                        if (!std::isfinite(value) || value < 0.0)
                            throw std::invalid_argument("pressure diagnostics require finite nonnegative water");
                        condensate += value;
                    }
                    projection_grid.reference_density[n] = reference_density;
                    buoyancy_force[2][n]                 = density_buoyancy(grid.density[n],
                                                                            reference_density,
                                                                            grid.sample(state.fields.qv, i, j, k),
                                                                            reference_vapor,
                                                                            condensate);
                }
            }
    if (request.vorticity)
    {
        Field3D* fields[] = {&candidate.vorticity_r,
                             &candidate.vorticity_theta,
                             &candidate.vorticity_z,
                             &candidate.stretching_term,
                             &candidate.tilting_term,
                             &candidate.baroclinic_term};
        for (std::size_t n = 0; n < vorticity.size(); ++n)
            grid.store(*fields[n], vorticity[n]);
    }
    if (request.angular_momentum)
    {
        grid.store(candidate.angular_momentum, angular[0]);
        grid.store(candidate.angular_momentum_tendency, angular[1]);
    }
    if (request.pressure_perturbation || request.pressure_components)
        candidate.p_prime = pressure_perturbation_field(state);
    if (request.pressure_components)
    {
        PressureProjection projection(projection_grid);
        grid.store(candidate.dynamic_pressure, projection.solve(dynamic_force, candidate_report.dynamic_pressure));
        grid.store(candidate.buoyancy_pressure, projection.solve(buoyancy_force, candidate_report.buoyancy_pressure));
    }
    using std::swap;
    auto& target = *state.diagnostics;
    if (request.vorticity)
    {
        swap(target.vorticity_r, candidate.vorticity_r);
        swap(target.vorticity_theta, candidate.vorticity_theta);
        swap(target.vorticity_z, candidate.vorticity_z);
        swap(target.stretching_term, candidate.stretching_term);
        swap(target.tilting_term, candidate.tilting_term);
        swap(target.baroclinic_term, candidate.baroclinic_term);
    }
    if (request.angular_momentum)
    {
        swap(target.angular_momentum, candidate.angular_momentum);
        swap(target.angular_momentum_tendency, candidate.angular_momentum_tendency);
    }
    if (request.pressure_perturbation || request.pressure_components)
        swap(target.p_prime, candidate.p_prime);
    if (request.pressure_components)
    {
        swap(target.dynamic_pressure, candidate.dynamic_pressure);
        swap(target.buoyancy_pressure, candidate.buoyancy_pressure);
    }
    if (report)
        *report = candidate_report;
}
} // namespace tmv::diagnostics
