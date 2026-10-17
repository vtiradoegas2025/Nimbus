#include "simple_grey.hpp"
#include "compute/dispatch/radiation.hpp"
#include "compute/simple_grey_contract.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace tmv
{
namespace
{
void pack_grey(double value, float& high, float& low)
{
    high = static_cast<float>(value);
    low  = static_cast<float>(value - high);
    if (!std::isfinite(high) || !std::isfinite(low) || (value != 0 && high == 0))
        throw std::overflow_error("native simple-grey input exceeds compensated storage");
}
} // namespace
bool SimpleGreyScheme::supports_native_column_batch(const RadiationConfig&,
                                                    const RadiationComputeKernels& kernels) const
{
    return kernels.supports(RadiationKernelOp::SimpleGrey);
}
bool SimpleGreyScheme::compute_columns_native(const RadiationConfig&          cfg,
                                              const RadiationColumnStateView* columns,
                                              std::size_t                     count,
                                              RadiationColumnTendencies*      tendencies,
                                              RadiationComputeKernels&        kernels)
{
    return compute_columns_native_diagnosed(cfg, columns, count, tendencies, kernels, nullptr);
}
bool SimpleGreyScheme::compute_columns_native_diagnosed(const RadiationConfig&          cfg,
                                                        const RadiationColumnStateView* columns,
                                                        std::size_t                     count,
                                                        RadiationColumnTendencies*      tendencies,
                                                        RadiationComputeKernels&        kernels,
                                                        RadiationColumnFluxes*          fluxes)
{
    if (!columns || !tendencies || count == 0 || !supports_native_column_batch(cfg, kernels))
        return false;
    validate_column(cfg, columns[0]);
    const auto levels = columns[0].rho->size();
    if (levels > 1000000 || count > std::size_t(std::numeric_limits<int>::max()) / (34 + 18 * (levels + 1)))
        return false;
    for (std::size_t c = 1; c < count; ++c)
    {
        validate_column(cfg, columns[c]);
        if (columns[c].rho->size() != levels)
            throw std::invalid_argument("native simple-grey columns must share their layer count");
    }
    SimpleGreyDispatch d;
    auto&              p = d.constants;
    p.columns            = int(count);
    p.layers             = int(levels);
    p.longwave           = cfg.do_lw;
    p.shortwave          = cfg.do_sw;
    p.water              = !cfg.clear_sky_only;
    std::vector<float> input(nimbus::compute::simple_grey::input_count(d)),
        output(nimbus::compute::simple_grey::output_count(d));
    for (std::size_t c = 0; c < count; ++c)
    {
        const auto&                  col    = columns[c];
        float*                       target = input.data() + c * nimbus::compute::simple_grey::input_stride(d);
        const std::array<double, 17> parameters{col.Tsfc,
                                                col.emissivity_lw,
                                                col.mu0,
                                                col.S0,
                                                col.albedo_sw,
                                                tau_lw_ref_,
                                                n_lw_,
                                                tau_sw_ref_,
                                                n_sw_,
                                                radiation_constants::sigma,
                                                physical_constants::specific_heat_cp_jkgk,
                                                absorption_coefficients_.longwave_vapor_m2kg,
                                                absorption_coefficients_.longwave_liquid_m2kg,
                                                absorption_coefficients_.longwave_ice_m2kg,
                                                absorption_coefficients_.shortwave_vapor_m2kg,
                                                absorption_coefficients_.shortwave_liquid_m2kg,
                                                absorption_coefficients_.shortwave_ice_m2kg};
        for (std::size_t f = 0; f < parameters.size(); ++f)
            pack_grey(parameters[f], target[2 * f], target[2 * f + 1]);
        for (std::size_t k = 0; k <= levels; ++k)
        {
            float* row = target + 34 + 14 * k;
            pack_grey((*col.p_int)[k], row[2], row[3]);
            if (k == levels)
                continue;
            pack_grey((*col.T)[k], row[0], row[1]);
            pack_grey((*col.rho)[k], row[4], row[5]);
            pack_grey((*col.dz)[k], row[6], row[7]);
            if (p.water)
            {
                pack_grey((*col.qv)[k], row[8], row[9]);
                pack_grey((*col.qc)[k], row[10], row[11]);
                pack_grey((*col.qi)[k], row[12], row[13]);
            }
        }
    }
    d.input  = input.data();
    d.output = output.data();
    if (!nimbus::compute::simple_grey::validate(d))
        throw std::invalid_argument("native simple-grey input contract failed");
    if (!kernels.dispatch_simple_grey(d))
        return false;
    if (!nimbus::compute::simple_grey::finite_candidate(output.data(), d))
        throw std::runtime_error("native simple-grey returned invalid fluxes or heating");
    std::vector<RadiationColumnTendencies> next(count);
    std::vector<RadiationColumnFluxes>     next_flux(fluxes ? count : 0);
    for (std::size_t c = 0; c < count; ++c)
    {
        const auto read = [&](std::size_t k, int f)
        {
            const auto n = (c * (levels + 1) + k) * 18 + 2 * f;
            return double(output[n]) + output[n + 1];
        };
        auto& t = next[c];
        t.dTdt_lw.resize(levels);
        t.dTdt_sw.resize(levels);
        t.dTdt_rad.resize(levels);
        for (std::size_t k = 0; k < levels; ++k)
        {
            t.dTdt_lw[k]  = read(k, 4);
            t.dTdt_sw[k]  = read(k, 5);
            t.dTdt_rad[k] = read(k, 6);
        }
        t.atmospheric_power_lw_wm2    = read(0, 7);
        t.atmospheric_power_sw_wm2    = read(0, 8);
        t.atmospheric_power_total_wm2 = t.atmospheric_power_lw_wm2 + t.atmospheric_power_sw_wm2;
        if (fluxes)
        {
            auto& f = next_flux[c];
            for (auto* values : {&f.Fup_lw, &f.Fdn_lw, &f.Fup_sw, &f.Fdn_sw, &f.Fnet_lw, &f.Fnet_sw, &f.Fnet_total})
                values->resize(levels + 1);
            for (std::size_t k = 0; k <= levels; ++k)
            {
                f.Fup_lw[k]     = read(k, 0);
                f.Fdn_lw[k]     = read(k, 1);
                f.Fup_sw[k]     = read(k, 2);
                f.Fdn_sw[k]     = read(k, 3);
                f.Fnet_lw[k]    = f.Fup_lw[k] - f.Fdn_lw[k];
                f.Fnet_sw[k]    = f.Fup_sw[k] - f.Fdn_sw[k];
                f.Fnet_total[k] = f.Fnet_lw[k] + f.Fnet_sw[k];
            }
        }
    }
    for (std::size_t c = 0; c < count; ++c)
    {
        tendencies[c] = std::move(next[c]);
        if (fluxes)
            fluxes[c] = std::move(next_flux[c]);
    }
    return true;
}
} // namespace tmv
