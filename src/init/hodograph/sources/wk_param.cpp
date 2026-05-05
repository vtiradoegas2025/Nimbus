/**
 * @file wk_param.cpp
 * @brief WKParamHodograph implementation. The math is the verbatim
 *        extraction of the previously-inline compute_wind_profile() in
 *        src/core/runtime/tornado_sim.cpp so that constructing this source
 *        with anchors matching global_wind_profile produces identical
 *        winds at every height.
 */

#include "init/hodograph/wk_param.hpp"

#include <cstddef>
#include <stdexcept>

namespace tmv::init
{

WKParamHodograph::WKParamHodograph(WKParamHodographAnchors anchors)
    : anchors_(anchors)
{
}

WindColumn WKParamHodograph::build(const std::vector<double>& z_m) const
{
    if (z_m.empty())
    {
        throw std::invalid_argument("WKParamHodograph::build: z_m is empty");
    }

    constexpr double z_sfc = 0.0;
    constexpr double z_1km = 1000.0;
    constexpr double z_6km = 6000.0;

    WindColumn out;
    out.u_ms.resize(z_m.size());
    out.v_ms.resize(z_m.size());

    for (std::size_t k = 0; k < z_m.size(); ++k)
    {
        const double z = z_m[k];
        if (z <= z_1km)
        {
            const double frac = (z - z_sfc) / (z_1km - z_sfc);
            out.u_ms[k] = anchors_.u_sfc_ms + frac * (anchors_.u_1km_ms - anchors_.u_sfc_ms);
            out.v_ms[k] = anchors_.v_sfc_ms + frac * (anchors_.v_1km_ms - anchors_.v_sfc_ms);
        }
        else if (z <= z_6km)
        {
            const double frac = (z - z_1km) / (z_6km - z_1km);
            out.u_ms[k] = anchors_.u_1km_ms + frac * (anchors_.u_6km_ms - anchors_.u_1km_ms);
            out.v_ms[k] = anchors_.v_1km_ms + frac * (anchors_.v_6km_ms - anchors_.v_1km_ms);
        }
        else
        {
            out.u_ms[k] = anchors_.u_6km_ms;
            out.v_ms[k] = anchors_.v_6km_ms;
        }
    }

    return out;
}

std::string WKParamHodograph::describe() const
{
    return "wk_param";
}

}  // namespace tmv::init
