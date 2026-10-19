#include "dynamics/moving_energy_internal.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_constants.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "dynamics/eos_pressure.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv
{
namespace
{
constexpr double rd = physical_constants::gas_constant_dry_air_jkgk;
constexpr double cp = physical_constants::specific_heat_cp_jkgk;
constexpr double cv = cp - rd;
constexpr double cvv =
    physical_constants::specific_heat_water_vapor_cp_jkgk - physical_constants::gas_constant_water_vapor_jkgk;
constexpr double cl   = physical_constants::specific_heat_liquid_water_jkgk;
constexpr double ci   = physical_constants::specific_heat_ice_jkgk;
constexpr double tref = physical_constants::freezing_temperature_k;
constexpr double ev =
    physical_constants::latent_heat_vaporization_jkg - physical_constants::gas_constant_water_vapor_jkgk * tref;
constexpr double ei = -physical_constants::latent_heat_fusion_jkg;
struct EnergyBasis
{
    double capacity, nonthermal;
};
EnergyBasis basis(const PrognosticFieldStorage& f, const TerrainMetrics3D& g, int i, int j, int k)
{
    const double vapor = f.qv(i, j, k), liquid = double(f.qc(i, j, k)) + f.qr(i, j, k);
    const double ice     = double(f.qi(i, j, k)) + f.qs(i, j, k) + f.qg(i, j, k) + f.qh(i, j, k);
    const double density = f.rho(i, j, k);
    const double uw = f.u(std::max(0, i - 1), j, k), ue = f.u(i, j, k), vs = f.v(i, std::max(0, j - 1), k),
                 vn = f.v(i, j, k);
    const double wb =
        k > 0 ? f.w(i, j, k - 1) : -.5 * (uw + ue) * g.Gx_face(i, j, 0) - .5 * (vs + vn) * g.Gy_face(i, j, 0);
    const double wt = f.w(i, j, k), kinetic = .25 * (uw * uw + ue * ue + vs * vs + vn * vn + wb * wb + wt * wt);
    const double capacity = density * (cv + vapor * cvv + liquid * cl + ice * ci);
    const double nonthermal =
        density * ((1 + vapor + liquid + ice) * (kinetic + physical_constants::gravity_ms2 * g.z(i, j, k)) +
                   vapor * ev + ice * ei + (f.tke.empty() ? 0 : f.tke(i, j, k)));
    if (!std::isfinite(capacity) || capacity <= 0 || !std::isfinite(nonthermal))
        throw std::domain_error("moving moist-energy basis is invalid");
    return {capacity, nonthermal};
}
double total(EnergyBasis b, float theta, float pressure)
{
    const double temperature = theta * std::pow(pressure / physical_constants::reference_pressure_pa, rd / cp);
    const double energy      = b.capacity * (temperature - tref) + b.nonthermal;
    if (!std::isfinite(energy))
        throw std::domain_error("moving moist energy is not finite");
    return energy;
}
} // namespace
namespace moving_energy_detail
{
void validate(const PrognosticFieldStorage& f, const TerrainMetrics3D& g)
{
    const int nx = g.J.nr, ny = g.J.nth, nz = g.J.nz;
    if (nx < 3 || ny < 3 || nz < 1 || nz == std::numeric_limits<int>::max() ||
        std::size_t(nx) > std::numeric_limits<std::size_t>::max() / std::size_t(ny) / std::size_t(nz))
        throw std::invalid_argument("moving energy requires a complete C-grid");
    for (double spacing : {g.dx, g.dy, g.dzeta})
        if (!std::isfinite(spacing) || spacing <= 0)
            throw std::invalid_argument("moving energy requires positive finite spacing");
    const double logical = g.dx * g.dy * g.dzeta;
    if (!std::isfinite(logical) || logical <= 0)
        throw std::invalid_argument("moving energy logical volume is not representable");
    const std::size_t count = std::size_t(nx) * ny * nz;
    for (const auto* metric : {&g.J, &g.z})
        if (metric->nr != nx || metric->nth != ny || metric->nz != nz || metric->size() != count)
            throw std::invalid_argument("moving energy geometry shape is invalid");
    for (const auto* metric : {&g.Gx_face, &g.Gy_face})
        if (metric->nr != nx || metric->nth != ny || metric->nz != nz + 1 ||
            metric->size() != std::size_t(nx) * ny * (nz + 1))
            throw std::invalid_argument("moving energy ground geometry shape is invalid");
    const std::array<const Field3D*, 14> fields{
        &f.rho, &f.theta, &f.p, &f.u, &f.v, &f.w, &f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.tke};
    for (std::size_t field = 0; field < fields.size(); ++field)
    {
        const auto& values = *fields[field];
        if (field == 13 && values.empty())
            continue;
        if (values.size_r() != nx || values.size_th() != ny || values.size_z() != nz || values.size() != count)
            throw std::invalid_argument("moving energy state shape is invalid");
        for (std::size_t n = 0; n < count; ++n)
            if (!std::isfinite(values.data()[n]) || (field < 3 && values.data()[n] <= 0) ||
                (field >= 6 && values.data()[n] < 0))
                throw std::invalid_argument("moving energy received invalid thermodynamic state");
    }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            if (!std::isfinite(g.Gx_face(i, j, 0)) || !std::isfinite(g.Gy_face(i, j, 0)))
                throw std::invalid_argument("moving energy ground slope is not finite");
            for (int k = 0; k < nz; ++k)
                if (!std::isfinite(g.z(i, j, k)) || !std::isfinite(g.J(i, j, k)) || g.J(i, j, k) <= 0 ||
                    !std::isfinite(logical * g.J(i, j, k)) || logical * g.J(i, j, k) <= 0)
                    throw std::invalid_argument("moving energy requires finite heights and positive volumes");
        }
}
void validate_target(const TerrainMetrics3D& g, const std::vector<double>& target)
{
    if (target.size() != g.J.size())
        throw std::invalid_argument("moving energy target shape is invalid");
    for (int i = 1; i < g.J.nr - 1; ++i)
        for (int j = 1; j < g.J.nth - 1; ++j)
            for (int k = 0; k < g.J.nz; ++k)
                if (!std::isfinite(target[(std::size_t(i) * g.J.nth + j) * g.J.nz + k]))
                    throw std::invalid_argument("moving energy target is not finite");
}
} // namespace moving_energy_detail
std::vector<double> moving_energy_density(const PrognosticFieldStorage& f, const TerrainMetrics3D& g)
{
    moving_energy_detail::validate(f, g);
    std::vector<double> result(g.J.size());
    for (int i = 0; i < g.J.nr; ++i)
        for (int j = 0; j < g.J.nth; ++j)
            for (int k = 0; k < g.J.nz; ++k)
                result[(std::size_t(i) * g.J.nth + j) * g.J.nz + k] =
                    total(basis(f, g, i, j, k), f.theta(i, j, k), f.p(i, j, k));
    return result;
}
MovingEnergyStorageReport
reconstruct_moving_energy(PrognosticFieldStorage& f, const TerrainMetrics3D& g, const std::vector<double>& target)
{
    moving_energy_detail::validate(f, g);
    moving_energy_detail::validate_target(g, target);
    auto                                       theta = f.theta, pressure = f.p;
    std::array<diagnostics::CompensatedSum, 3> sums;
    for (int i = 1; i < g.J.nr - 1; ++i)
        for (int j = 1; j < g.J.nth - 1; ++j)
            for (int k = 0; k < g.J.nz; ++k)
            {
                const auto   n           = (std::size_t(i) * g.J.nth + j) * g.J.nz + k;
                const auto   b           = basis(f, g, i, j, k);
                const double temperature = tref + (target[n] - b.nonthermal) / b.capacity;
                const double p           = f.rho(i, j, k) * rd * temperature;
                const float  stored_theta =
                    static_cast<float>(temperature * std::pow(physical_constants::reference_pressure_pa / p, rd / cp));
                const float stored_pressure =
                    static_cast<float>(dynamics_eos::pressure_fp64(f.rho(i, j, k), stored_theta));
                if (!std::isfinite(stored_theta) || stored_theta < theta_min_k || stored_theta > theta_max_k ||
                    !std::isfinite(stored_pressure) || stored_pressure <= 0 || temperature <= 0)
                    throw std::domain_error(
                        "coordinate remapping cannot represent its conserved energy thermodynamically");
                theta(i, j, k)         = stored_theta;
                pressure(i, j, k)      = stored_pressure;
                const double volume    = g.dx * g.dy * g.dzeta * g.J(i, j, k);
                const double requested = target[n] * volume, stored = total(b, stored_theta, stored_pressure) * volume;
                sums[0].add(requested);
                sums[1].add(stored);
                sums[2].add(stored - requested);
            }
    for (const auto& sum : sums)
        if (!std::isfinite(sum.value()))
            throw std::overflow_error("moving energy storage ledger overflowed");
    f.theta = std::move(theta);
    f.p     = std::move(pressure);
    return {sums[0].value(), sums[1].value(), sums[2].value()};
}
} // namespace tmv
