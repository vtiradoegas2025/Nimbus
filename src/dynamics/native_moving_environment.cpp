#include "dynamics/moving_environment_internal.hpp"
#include "compute/dispatch/terrain.hpp"
#include "compute/reference_atmosphere_contract.hpp"
#include "core/infra/physical_constants.hpp"
#include <algorithm>
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
    if (!std::isfinite(high) || !std::isfinite(low) || (value != 0 && high == 0))
        throw std::overflow_error("native reference atmosphere exceeds compensated storage");
}
} // namespace
std::vector<float> moving_environment_detail::sample_reference(const BaseStateStorage&    base,
                                                               const std::vector<double>& heights,
                                                               double                     dz,
                                                               TerrainComputeKernels&     kernels)
{
    moving_environment_detail::validate_base(base);
    if (!std::isfinite(dz) || dz <= 0)
        throw std::invalid_argument("native reference atmosphere requires positive spacing");
    if (heights.empty())
        return {};
    if (!kernels.supports_reference_atmosphere())
        throw std::runtime_error("native reference-atmosphere kernels are unavailable");
    const auto source = moving_environment_detail::profiles(base);
    const auto limit  = std::size_t(std::numeric_limits<int>::max());
    if (heights.size() > limit / 6 || source[0]->size() > (limit - 2 * heights.size()) / 10 ||
        source[0]->size() > (1u << 24))
        throw std::invalid_argument("native reference atmosphere exceeds dispatch limits");
    ReferenceAtmosphereDispatch d;
    auto&                       p = d.constants;
    p.count                       = int(heights.size());
    p.levels                      = int(source[0]->size());
    using namespace physical_constants;
    pack(dz, p.dz_high, p.dz_low);
    pack(gas_constant_dry_air_jkgk, p.rd_high, p.rd_low);
    pack(reference_pressure_pa, p.p0_high, p.p0_low);
    pack(gas_constant_dry_air_jkgk / specific_heat_cp_jkgk, p.kappa_high, p.kappa_low);
    std::array<int*, 5> exponents{
        &p.theta_exponent, &p.pressure_exponent, &p.vapor_exponent, &p.u_exponent, &p.v_exponent};
    std::vector<float> input(nimbus::compute::reference_atmosphere::input_count(d)), output(6 * heights.size());
    for (std::size_t field = 0; field < source.size(); ++field)
    {
        double maximum = 0;
        for (double value : *source[field])
            maximum = std::max(maximum, std::abs(value));
        std::frexp(maximum, exponents[field]);
        if (*exponents[field] > 128)
            throw std::overflow_error("native reference profile exceeds FP32 storage range");
        for (int k = 0; k < p.levels; ++k)
            pack(std::ldexp((*source[field])[k], -*exponents[field]),
                 input[10 * k + 2 * field],
                 input[10 * k + 2 * field + 1]);
    }
    for (std::size_t n = 0; n < heights.size(); ++n)
        pack(heights[n], input[10 * std::size_t(p.levels) + 2 * n], input[10 * std::size_t(p.levels) + 2 * n + 1]);
    d.input  = input.data();
    d.output = output.data();
    if (!nimbus::compute::reference_atmosphere::validate(d))
        throw std::invalid_argument("native reference atmosphere violates its input contract");
    if (!kernels.dispatch_reference_atmosphere(d))
        throw std::runtime_error("native reference-atmosphere dispatch failed");
    if (!nimbus::compute::reference_atmosphere::finite_candidate(output.data(), d))
        throw std::runtime_error("native reference atmosphere returned invalid samples");
    return output;
}
void fill_moving_environment(PrognosticFieldStorage& fields,
                             const BaseStateStorage& base,
                             const TerrainMetrics3D& geometry,
                             TerrainComputeKernels&  kernels)
{
    moving_environment_detail::validate(fields, base, geometry);
    const auto          cells = moving_environment_detail::boundary_cells(geometry);
    std::vector<double> heights;
    heights.reserve(cells.size());
    for (auto cell : cells)
        heights.push_back(geometry.z.values[cell]);
    const auto output = moving_environment_detail::sample_reference(base, heights, geometry.dzeta, kernels);
    moving_environment_detail::publish(fields, geometry, cells, output);
}
} // namespace tmv
