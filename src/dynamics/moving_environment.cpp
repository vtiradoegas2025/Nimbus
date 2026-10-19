#include "dynamics/moving_environment_internal.hpp"
#include "terrain/conservative_transport.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
namespace tmv
{
namespace moving_environment_detail
{
Profiles profiles(const BaseStateStorage& base)
{
    return {&base.theta0, &base.p0, &base.qv0, &base.u0, &base.v0};
}
void validate_base(const BaseStateStorage& base)
{
    const auto source = profiles(base);
    if (source[0]->empty() || source[0]->size() > std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("moving environment requires nonempty reference profiles");
    for (std::size_t f = 0; f < source.size(); ++f)
    {
        if (source[f]->size() != source[0]->size())
            throw std::invalid_argument("moving environment reference profiles must share their levels");
        for (double value : *source[f])
            if (!std::isfinite(value) || (f < 2 && value <= 0) || (f == 2 && value < 0))
                throw std::invalid_argument("moving environment reference profile is invalid");
    }
}
void validate(const PrognosticFieldStorage& f, const BaseStateStorage& base, const TerrainMetrics3D& g)
{
    const int nx = g.z.nr, ny = g.z.nth, nz = g.z.nz;
    if (nx < 3 || ny < 3 || nz < 1 ||
        std::size_t(nx) > std::numeric_limits<std::size_t>::max() / std::size_t(ny) / std::size_t(nz) ||
        g.z.size() != std::size_t(nx) * ny * nz || !std::isfinite(g.dzeta) || g.dzeta <= 0)
        throw std::invalid_argument("moving environment requires a complete terrain grid");
    const std::array<const Field3D*, 15> fields{
        &f.rho, &f.theta, &f.p, &f.qv, &f.u, &f.v, &f.w, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.tke, &f.tracer};
    for (std::size_t n = 0; n < fields.size(); ++n)
    {
        const auto* field = fields[n];
        if (n >= 7 && field->empty())
            continue;
        if (field->size_r() != nx || field->size_th() != ny || field->size_z() != nz || field->size() != g.z.size())
            throw std::invalid_argument("moving environment received malformed state storage");
    }
    validate_base(base);
}
std::vector<std::size_t> boundary_cells(const TerrainMetrics3D& g)
{
    std::vector<std::size_t> result;
    result.reserve((2 * std::size_t(g.z.nr) + 2 * std::size_t(g.z.nth) - 4) * g.z.nz);
    for (int i = 0; i < g.z.nr; ++i)
        for (int j = 0; j < g.z.nth; j += (i == 0 || i == g.z.nr - 1) ? 1 : g.z.nth - 1)
            for (int k = 0; k < g.z.nz; ++k)
            {
                const auto n = (std::size_t(i) * g.z.nth + j) * g.z.nz + k;
                if (!std::isfinite(g.z.values[n]))
                    throw std::invalid_argument("moving environment ghost height is not finite");
                result.push_back(n);
            }
    return result;
}
void publish(PrognosticFieldStorage&         f,
             const TerrainMetrics3D&         g,
             const std::vector<std::size_t>& cells,
             const std::vector<float>&       candidates)
{
    if (candidates.size() != 6 * cells.size())
        throw std::invalid_argument("moving environment candidate shape is invalid");
    for (std::size_t n = 0; n < cells.size(); ++n)
    {
        const auto* v = candidates.data() + 6 * n;
        for (int k = 0; k < 6; ++k)
            if (!std::isfinite(v[k]))
                throw std::domain_error("moving environment candidate is not finite");
        if (v[0] <= 0 || v[1] <= 0 || v[2] < 0 || v[5] <= 0)
            throw std::domain_error("moving environment candidate is not physical");
    }
    for (std::size_t n = 0; n < cells.size(); ++n)
    {
        const auto  cell     = cells[n];
        const auto* v        = candidates.data() + 6 * n;
        f.theta.data()[cell] = v[0];
        f.p.data()[cell]     = v[1];
        f.qv.data()[cell]    = v[2];
        f.rho.data()[cell]   = v[5];
        if (cell / (std::size_t(g.z.nth) * g.z.nz) != 0)
            f.u.data()[cell] = v[3];
        if ((cell / g.z.nz) % g.z.nth != 0)
            f.v.data()[cell] = v[4];
        f.w.data()[cell] = 0;
        for (auto* field : {&f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.tke, &f.tracer})
            if (!field->empty())
                field->data()[cell] = 0;
    }
}
} // namespace moving_environment_detail
void fill_moving_environment(PrognosticFieldStorage& f, const BaseStateStorage& base, const TerrainMetrics3D& g)
{
    moving_environment_detail::validate(f, base, g);
    const auto         cells  = moving_environment_detail::boundary_cells(g);
    const auto         source = moving_environment_detail::profiles(base);
    std::vector<float> candidates(6 * cells.size());
    for (std::size_t n = 0; n < cells.size(); ++n)
    {
        const double z = g.z.values[cells[n]];
        for (std::size_t field = 0; field < source.size(); ++field)
            candidates[6 * n + field] =
                static_cast<float>(terrain_transport::reference_profile_value_fp32(*source[field], z, g.dzeta));
        candidates[6 * n + 5] =
            static_cast<float>(terrain_transport::reference_density_fp32(base.p0, base.theta0, z, g.dzeta));
    }
    moving_environment_detail::publish(f, g, cells, candidates);
}
} // namespace tmv
