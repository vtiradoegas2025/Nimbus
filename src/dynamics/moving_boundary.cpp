#include "dynamics/moving_boundary_internal.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv
{
namespace
{
void shape(const Field3D& field, int nx, int ny, int nz)
{
    if (field.size_r() != nx || field.size_th() != ny || field.size_z() != nz ||
        field.size() != std::size_t(nx) * ny * nz)
        throw std::invalid_argument("moving boundary state has an invalid shape");
}
void shape(const TerrainField3D& field, int nx, int ny, int nz)
{
    if (field.nr != nx || field.nth != ny || field.nz != nz || field.size() != std::size_t(nx) * ny * nz)
        throw std::invalid_argument("moving boundary geometry has an invalid shape");
}
double volume(const TerrainMetrics3D& g, int i, int j, int k)
{
    return g.dx * g.dy * g.dzeta * g.J(i, j, k);
}
double ground_velocity(const TerrainMetrics3D& g, const Field3D& u, const Field3D& v, int i, int j)
{
    return -0.5 * (double(u(std::max(0, i - 1), j, 0)) + u(i, j, 0)) * g.Gx_face(i, j, 0) -
           0.5 * (double(v(i, std::max(0, j - 1), 0)) + v(i, j, 0)) * g.Gy_face(i, j, 0);
}
void accumulate(std::array<diagnostics::CompensatedSum, 7>& sums, const std::array<double, 7>& values)
{
    for (std::size_t f = 0; f < sums.size(); ++f)
    {
        if (!std::isfinite(values[f]))
            throw std::overflow_error("moving boundary ledger overflowed");
        sums[f].add(values[f]);
    }
}
} // namespace
namespace moving_boundary_detail
{
void validate(
    const TerrainMetrics3D& g, const Field3D& rho, const Field3D& u, const Field3D& v, const Field3D* w, bool ground)
{
    const int nx = g.J.nr, ny = g.J.nth, nz = g.J.nz;
    if (nx < 4 || ny < 4 || nz < 2 || nx == std::numeric_limits<int>::max() || ny == std::numeric_limits<int>::max() ||
        nz == std::numeric_limits<int>::max() ||
        std::size_t(nx) > std::numeric_limits<std::size_t>::max() / std::size_t(ny) / std::size_t(nz))
        throw std::invalid_argument("moving boundary requires a complete C-grid");
    shape(g.J, nx, ny, nz);
    shape(rho, nx, ny, nz);
    shape(u, nx, ny, nz);
    shape(v, nx, ny, nz);
    if (w)
    {
        shape(*w, nx, ny, nz);
        if (w->data() == rho.data() || w->data() == u.data() || w->data() == v.data())
            throw std::invalid_argument("moving ground output aliases its inputs");
    }
    for (double spacing : {g.dx, g.dy, g.dzeta})
        if (!std::isfinite(spacing) || spacing <= 0)
            throw std::invalid_argument("moving boundary spacing is invalid");
    if (ground)
    {
        shape(g.Gx_face, nx, ny, nz + 1);
        shape(g.Gy_face, nx, ny, nz + 1);
    }
    const int first = ground ? 0 : nz - 2;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            if (ground && (!std::isfinite(g.Gx_face(i, j, 0)) || !std::isfinite(g.Gy_face(i, j, 0))))
                throw std::invalid_argument("moving ground slope is not finite");
            for (int k = first; k < first + 2; ++k)
                if (!std::isfinite(g.J(i, j, k)) || g.J(i, j, k) <= 0 || !std::isfinite(rho(i, j, k)) ||
                    rho(i, j, k) <= 0 || !std::isfinite(u(i, j, k)) || !std::isfinite(v(i, j, k)) ||
                    (w && !std::isfinite((*w)(i, j, k))))
                    throw std::invalid_argument("moving boundary requires finite velocities and positive carriers");
        }
}
void validate_flux(const TerrainMetrics3D& g, const TerrainContravariantFluxes& flux)
{
    const int nx = g.J.nr, ny = g.J.nth, nz = g.J.nz;
    shape(flux.x, nx + 1, ny, nz);
    shape(flux.y, nx, ny + 1, nz);
    shape(flux.z, nx, ny, nz + 1);
    for (int i = 0; i <= nx; ++i)
        for (int j = 0; j <= ny; ++j)
        {
            if (j < ny && !std::isfinite(flux.x(i, j, 0)))
                throw std::invalid_argument("moving ground x transfer is invalid");
            if (i < nx && !std::isfinite(flux.y(i, j, 0)))
                throw std::invalid_argument("moving ground y transfer is invalid");
        }
    // The implicit half-cell ledger belongs to horizontal coordinate remapping.
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < 2; ++k)
                if (flux.z(i, j, k) != 0)
                    throw std::invalid_argument("moving ground targets cannot discard bottom vertical transfers");
}
void validate_targets(const TerrainMetrics3D& g, const MovingGroundTargets& targets)
{
    shape(targets.impulse, g.J.nr, g.J.nth, 1);
    for (double value : targets.impulse.values)
        if (!std::isfinite(value))
            throw std::invalid_argument("moving ground impulse target is not finite");
}
void validate_top_outputs(const Field3D& rho, const Field3D& u, const Field3D& v)
{
    if (u.data() == v.data() || u.data() == rho.data() || v.data() == rho.data())
        throw std::invalid_argument("moving top outputs alias each other or the density");
}
void publish_top(const std::array<Field3D, 2>& means, Field3D& u, Field3D& v)
{
    const int nx = u.size_r(), ny = u.size_th(), nz = u.size_z();
    for (int c = 0; c < 2; ++c)
    {
        auto& field = c == 0 ? u : v;
        for (int i = c == 0 ? 0 : 1; i < nx - 1; ++i)
            for (int j = c == 1 ? 0 : 1; j < ny - 1; ++j)
                field(i, j, nz - 2) = field(i, j, nz - 1) = means[c](i, j, 0);
    }
}
MovingBoundaryLedger ledger(const std::array<diagnostics::CompensatedSum, 7>& sums)
{
    for (const auto& sum : sums)
        if (!std::isfinite(sum.value()))
            throw std::overflow_error("moving boundary ledger overflowed");
    return {sums[0].value(),
            sums[1].value(),
            sums[2].value(),
            sums[3].value(),
            sums[4].value(),
            sums[5].value(),
            sums[6].value()};
}
} // namespace moving_boundary_detail
std::array<MovingBoundaryLedger, 2>
project_moving_top_momentum(const TerrainMetrics3D& g, const Field3D& rho, Field3D& u, Field3D& v)
{
    moving_boundary_detail::validate(g, rho, u, v, nullptr, false);
    moving_boundary_detail::validate_top_outputs(rho, u, v);
    const int                           nx = g.J.nr, ny = g.J.nth, nz = g.J.nz;
    std::array<Field3D, 2>              means{Field3D(nx, ny, 1), Field3D(nx, ny, 1)};
    std::array<MovingBoundaryLedger, 2> result;
    const auto                          mass = [&](int i, int j, int k)
    {
        return i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 ? double(rho(i, j, k)) * volume(g, i, j, k) : 0.0;
    };
    for (int c = 0; c < 2; ++c)
    {
        const auto&                                field = c == 0 ? u : v;
        std::array<diagnostics::CompensatedSum, 7> sums{};
        for (int i = c == 0 ? 0 : 1; i < nx - 1; ++i)
            for (int j = c == 1 ? 0 : 1; j < ny - 1; ++j)
            {
                const auto dual = [&](int k)
                {
                    return .5 * (mass(i, j, k) + mass(i + (c == 0), j + (c == 1), k));
                };
                const double lower = dual(nz - 2), upper = dual(nz - 1), total = lower + upper;
                const double a = field(i, j, nz - 2), b = field(i, j, nz - 1);
                const double initial = lower * a + upper * b, exact = initial / total;
                const float  stored = static_cast<float>(exact);
                if (!std::isfinite(total) || total <= 0 || !std::isfinite(stored))
                    throw std::overflow_error("moving top projection is not representable");
                means[c](i, j, 0)  = stored;
                const double final = total * stored;
                accumulate(sums,
                           {initial,
                            initial,
                            final,
                            final - initial,
                            .5 * (lower * a * a + upper * b * b),
                            .5 * total * exact * exact,
                            .5 * total * stored * stored});
            }
        result[c] = moving_boundary_detail::ledger(sums);
    }
    moving_boundary_detail::publish_top(means, u, v);
    return result;
}
MovingGroundTargets prepare_moving_ground_targets(const TerrainMetrics3D&           g,
                                                  const Field3D&                    rho,
                                                  const Field3D&                    u,
                                                  const Field3D&                    v,
                                                  const TerrainContravariantFluxes& flux)
{
    moving_boundary_detail::validate(g, rho, u, v, nullptr, true);
    moving_boundary_detail::validate_flux(g, flux);
    const int           nx = g.J.nr, ny = g.J.nth;
    MovingGroundTargets result;
    result.impulse.resize(nx, ny, 1, 0);
    diagnostics::CompensatedSum initial;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
        {
            const double mass   = .5 * rho(i, j, 0) * volume(g, i, j, 0);
            double       target = mass * ground_velocity(g, u, v, i, j), outgoing = 0;
            initial.add(target);
            for (int side : {-1, 1})
            {
                const double fx = .5 * flux.x(i + (side > 0), j, 0), fy = .5 * flux.y(i, j + (side > 0), 0);
                const int    ix = i + (side > 0 ? (fx < 0) : -(fx > 0)), jy = j + (side > 0 ? (fy < 0) : -(fy > 0));
                const double x = side * fx * ground_velocity(g, u, v, ix, j),
                             y = side * fy * ground_velocity(g, u, v, i, jy);
                target -= x + y;
                outgoing += std::max(0.0, side * fx) + std::max(0.0, side * fy);
                if ((side < 0 && i == 1) || (side > 0 && i == nx - 2))
                    result.outward_impulse += x;
                if ((side < 0 && j == 1) || (side > 0 && j == ny - 2))
                    result.outward_impulse += y;
            }
            if (!std::isfinite(target) || !std::isfinite(mass) || mass <= 0 ||
                outgoing > mass * (1 + 64 * std::numeric_limits<double>::epsilon()))
                throw std::domain_error("moving ground transfer exhausts its carrier or overflows");
            result.impulse(i, j, 0) = target;
        }
    result.initial_impulse = initial.value();
    if (!std::isfinite(result.initial_impulse) || !std::isfinite(result.outward_impulse))
        throw std::overflow_error("moving ground ledger overflowed");
    return result;
}
MovingBoundaryLedger correct_moving_ground_momentum(const TerrainMetrics3D&    g,
                                                    const Field3D&             rho,
                                                    const Field3D&             u,
                                                    const Field3D&             v,
                                                    Field3D&                   w,
                                                    const MovingGroundTargets& targets)
{
    moving_boundary_detail::validate(g, rho, u, v, &w, true);
    moving_boundary_detail::validate_targets(g, targets);
    const int                                  nx = g.J.nr, ny = g.J.nth;
    Field3D                                    candidate(nx, ny, 1);
    std::array<diagnostics::CompensatedSum, 7> sums{};
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
        {
            const double lower = rho(i, j, 0) * volume(g, i, j, 0), upper = rho(i, j, 1) * volume(g, i, j, 1);
            const double mass = .5 * (lower + upper), ground_mass = .5 * lower, ground = ground_velocity(g, u, v, i, j),
                         old        = w(i, j, 0);
            const double correction = targets.impulse(i, j, 0) - ground_mass * ground;
            const double exact      = old + correction / mass;
            const float  stored     = static_cast<float>(exact);
            if (!std::isfinite(mass) || mass <= 0 || !std::isfinite(stored))
                throw std::overflow_error("moving ground correction is not representable");
            candidate(i, j, 0)   = stored;
            const double initial = mass * old + ground_mass * ground, requested = mass * old + targets.impulse(i, j, 0),
                         final         = mass * stored + ground_mass * ground;
            const double ground_energy = .5 * ground_mass * ground * ground;
            accumulate(sums,
                       {initial,
                        requested,
                        final,
                        final - requested,
                        .5 * mass * old * old + ground_energy,
                        .5 * mass * exact * exact + ground_energy,
                        .5 * mass * stored * stored + ground_energy});
        }
    auto result = moving_boundary_detail::ledger(sums);
    // Only publish after every column has a finite correction and ledger.
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            w(i, j, 0) = candidate(i, j, 0);
    return result;
}
} // namespace tmv
