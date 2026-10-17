#include "dynamics/moving_energy_internal.hpp"
#include "compute/dispatch/terrain.hpp"
#include "compute/moving_energy_contract.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_constants.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv
{
namespace
{
void pack(double value, float& high, float& low)
{
    high = static_cast<float>(value);
    low  = static_cast<float>(value - high);
    if (!std::isfinite(value) || !std::isfinite(high) || !std::isfinite(low))
        throw std::overflow_error("native moving energy exceeds compensated storage");
}
std::vector<float> dispatch(TerrainComputeKernels&        kernels,
                            const PrognosticFieldStorage& f,
                            const TerrainMetrics3D&       g,
                            const std::vector<double>*    target)
{
    moving_energy_detail::validate(f, g);
    if (target)
        moving_energy_detail::validate_target(g, *target);
    if (!kernels.supports_moving_energy())
        throw std::runtime_error("native moving-energy kernels are unavailable");
    MovingEnergyDispatch d;
    auto&                p = d.constants;
    p.nx                   = g.J.nr;
    p.ny                   = g.J.nth;
    p.nz                   = g.J.nz;
    p.operation            = target ? 1 : 0;
    if (g.J.size() > std::size_t(std::numeric_limits<int>::max()) / MovingEnergyDispatch::input_stride)
        throw std::invalid_argument("native moving energy exceeds dispatch limits");
    using namespace physical_constants;
    pack(gas_constant_dry_air_jkgk, p.rd_high, p.rd_low);
    pack(specific_heat_cp_jkgk - gas_constant_dry_air_jkgk, p.cv_high, p.cv_low);
    pack(specific_heat_water_vapor_cp_jkgk - gas_constant_water_vapor_jkgk, p.cvv_high, p.cvv_low);
    pack(specific_heat_liquid_water_jkgk, p.cl_high, p.cl_low);
    pack(specific_heat_ice_jkgk, p.ci_high, p.ci_low);
    pack(freezing_temperature_k, p.tref_high, p.tref_low);
    pack(latent_heat_vaporization_jkg - gas_constant_water_vapor_jkgk * freezing_temperature_k, p.ev_high, p.ev_low);
    pack(-latent_heat_fusion_jkg, p.ei_high, p.ei_low);
    pack(gravity_ms2, p.gravity_high, p.gravity_low);
    pack(reference_pressure_pa, p.p0_high, p.p0_low);
    pack(gas_constant_dry_air_jkgk / specific_heat_cp_jkgk, p.kappa_high, p.kappa_low);
    pack(dry_air_gamma, p.gamma_high, p.gamma_low);
    pack(theta_min_k, p.theta_min_high, p.theta_min_low);
    pack(theta_max_k, p.theta_max_high, p.theta_max_low);
    std::vector<float> input(g.J.size() * MovingEnergyDispatch::input_stride),
        output(g.J.size() * MovingEnergyDispatch::output_stride);
    const std::array<const Field3D*, 14> fields{
        &f.rho, &f.theta, &f.p, &f.u, &f.v, &f.w, &f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.tke};
    for (int i = 0; i < p.nx; ++i)
        for (int j = 0; j < p.ny; ++j)
            for (int k = 0; k < p.nz; ++k)
            {
                const auto n      = (std::size_t(i) * p.ny + j) * p.nz + k;
                auto*      record = input.data() + n * MovingEnergyDispatch::input_stride;
                for (std::size_t field = 0; field < fields.size(); ++field)
                    record[field] = fields[field]->empty() ? 0 : fields[field]->data()[n];
                pack(g.z(i, j, k), record[14], record[15]);
                if (k == 0)
                {
                    pack(g.Gx_face(i, j, 0), record[16], record[17]);
                    pack(g.Gy_face(i, j, 0), record[18], record[19]);
                }
                if (target && i > 0 && i < p.nx - 1 && j > 0 && j < p.ny - 1)
                    pack((*target)[n], record[20], record[21]);
                pack(g.dx * g.dy * g.dzeta * g.J(i, j, k), record[22], record[23]);
            }
    d.input  = input.data();
    d.output = output.data();
    if (!nimbus::compute::moving_energy::validate(d))
        throw std::invalid_argument("native moving energy violates its input contract");
    if (!kernels.dispatch_moving_energy(d))
        throw std::runtime_error("native moving-energy dispatch failed");
    if (!nimbus::compute::moving_energy::finite_candidate(output.data(), d))
        throw std::overflow_error("native moving energy returned an invalid candidate");
    return output;
}
} // namespace
std::vector<double>
moving_energy_density(const PrognosticFieldStorage& f, const TerrainMetrics3D& g, TerrainComputeKernels& kernels)
{
    const auto          output = dispatch(kernels, f, g, nullptr);
    std::vector<double> result(g.J.size());
    for (std::size_t n = 0; n < result.size(); ++n)
    {
        const auto* record = output.data() + n * MovingEnergyDispatch::output_stride;
        result[n]          = double(record[6]) + record[7];
    }
    return result;
}
MovingEnergyStorageReport reconstruct_moving_energy(PrognosticFieldStorage&    f,
                                                    const TerrainMetrics3D&    g,
                                                    const std::vector<double>& target,
                                                    TerrainComputeKernels&     kernels)
{
    const auto                                 output = dispatch(kernels, f, g, &target);
    auto                                       theta = f.theta, pressure = f.p;
    std::array<diagnostics::CompensatedSum, 3> sums;
    for (std::size_t n = 0; n < g.J.size(); ++n)
    {
        const auto* record = output.data() + n * MovingEnergyDispatch::output_stride;
        theta.data()[n]    = record[0];
        pressure.data()[n] = record[1];
        for (std::size_t f = 0; f < sums.size(); ++f)
        {
            sums[f].add(record[8 + 2 * f]);
            sums[f].add(record[9 + 2 * f]);
        }
    }
    for (const auto& sum : sums)
        if (!std::isfinite(sum.value()))
            throw std::overflow_error("native moving energy storage ledger overflowed");
    f.theta = std::move(theta);
    f.p     = std::move(pressure);
    return {sums[0].value(), sums[1].value(), sums[2].value()};
}
} // namespace tmv
