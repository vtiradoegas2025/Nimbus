#pragma once

#include "init/hodograph/hodograph_source.hpp"

namespace tmv::init
{

/**
 * @brief Anchor points for the WK 3-point parametric hodograph.
 *
 * Linear interpolation between (sfc, 1 km, 6 km) and constant above 6 km,
 * matching the legacy compute_wind_profile() in tornado_sim.cpp. The new
 * source produces identical winds at every height when constructed from
 * the same anchors; this is the regression net for the extraction.
 */
struct WKParamHodographAnchors
{
    double u_sfc_ms = 0.0;
    double v_sfc_ms = 0.0;
    double u_1km_ms = 0.0;
    double v_1km_ms = 0.0;
    double u_6km_ms = 0.0;
    double v_6km_ms = 0.0;
};

/**
 * @brief 3-point linear hodograph (Weisman & Klemp 1982 style).
 *
 *   z <= 1 km : linear interpolation sfc -> 1 km
 *   z <= 6 km : linear interpolation 1 km -> 6 km
 *   z >  6 km : constant at 6 km value
 *
 * The interpolation is performed on Cartesian (u_x, u_y) components at each
 * z_m[k]. Coordinate-aware projection happens in the runtime, not the source.
 */
class WKParamHodograph final : public HodographSource
{
public:
    explicit WKParamHodograph(WKParamHodographAnchors anchors);

    WindColumn build(const std::vector<double>& z_m) const override;
    std::string describe() const override;

private:
    WKParamHodographAnchors anchors_;
};

}  // namespace tmv::init
