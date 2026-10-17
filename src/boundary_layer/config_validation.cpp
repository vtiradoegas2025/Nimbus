#include "util/physics_cadence.hpp"
#include "boundary_layer/config_validation.hpp"
#include "boundary_layer/boundary_layer_base.hpp"

#include <cmath>
#include <stdexcept>

namespace tmv
{

void validate_boundary_layer_configuration(
    const BoundaryLayerConfig& config,
    const SurfaceConfig& surface,
    std::string_view canonical_scheme_id)
{
    if (canonical_scheme_id != "slab" && canonical_scheme_id != "ysu"
        && canonical_scheme_id != "mynn" && canonical_scheme_id != "none")
        throw std::invalid_argument(
            "boundary-layer scheme id is not canonical or supported");
    if (config.scheme_id != canonical_scheme_id)
        throw std::invalid_argument(
            "boundary-layer configuration does not match the selected scheme");
    require_physics_cadence(config.dt_pbl, "boundary-layer dt_pbl");
    if (!std::isfinite(config.pbl_max_height)
        || config.pbl_max_height <= 0.0)
        throw std::invalid_argument(
            "boundary-layer pbl_max_height must be positive and finite");
    if (!std::isfinite(config.min_ustar) || config.min_ustar < 0.0)
        throw std::invalid_argument(
            "boundary-layer min_ustar must be non-negative and finite");
    if (config.surface_layer_id != "bulk"
        && config.surface_layer_id != "monin_obukhov")
        throw std::invalid_argument(
            "boundary-layer surface_layer_id must be bulk or monin_obukhov");
    if (config.pbl_top_method != "ri" &&
        config.pbl_top_method != "theta" &&
        config.pbl_top_method != "tke")
        throw std::invalid_argument(
            "boundary-layer pbl_top_method must be ri, theta, or tke");
    if (!config.clear_sky_only && canonical_scheme_id != "ysu")
        throw std::invalid_argument(
            "boundary-layer cloud-top radiative mixing is available only for YSU");
    const bool scheme_requires_tke = canonical_scheme_id == "mynn";
    if (config.enable_tke != scheme_requires_tke)
        throw std::invalid_argument(
            scheme_requires_tke
                ? "MYNN requires boundary_layer.enable_tke=true"
                : "boundary_layer.enable_tke=true is only valid for MYNN");
    if (config.pbl_top_method == "tke" && !scheme_requires_tke)
        throw std::invalid_argument(
            "boundary-layer pbl_top_method=tke requires MYNN prognostic TKE");

    if (!std::isfinite(surface.z0m) || surface.z0m <= 0.0
        || !std::isfinite(surface.z0h) || surface.z0h <= 0.0)
        throw std::invalid_argument(
            "surface roughness lengths must be positive and finite");
    if (!std::isfinite(surface.albedo) || surface.albedo < 0.0
        || surface.albedo > 1.0 || !std::isfinite(surface.emissivity)
        || surface.emissivity < 0.0 || surface.emissivity > 1.0)
        throw std::invalid_argument(
            "surface albedo and emissivity must be finite and in [0,1]");
    if (!std::isfinite(surface.Tsfc)
        || (surface.Tsfc != -1.0
            && (surface.Tsfc < 150.0 || surface.Tsfc > 400.0)))
        throw std::invalid_argument(
            "surface Tsfc must be the -1 sentinel or a temperature in [150,400] K");
    if (!std::isfinite(surface.qsfc)
        || (surface.qsfc != -1.0
            && (surface.qsfc < 0.0 || surface.qsfc > 0.1)))
        throw std::invalid_argument(
            "surface qsfc must be the -1 sentinel or a mixing ratio in [0,0.1]");
}

} // namespace tmv
