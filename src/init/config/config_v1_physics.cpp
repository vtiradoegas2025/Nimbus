#include "init/config/config_v1.hpp"

#include <algorithm>
#include <cctype>

namespace tmv::init::config
{

namespace
{

std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

} // namespace

// ─────────────────────────────────────────────────────────────
// MicrophysicsScheme
// ─────────────────────────────────────────────────────────────

bool parse_microphysics_scheme(const std::string& s, MicrophysicsScheme& out)
{
    const std::string v = to_lower(s);
    if (v == "none")
    {
        out = MicrophysicsScheme::None;
        return true;
    }
    if (v == "kessler")
    {
        out = MicrophysicsScheme::Kessler;
        return true;
    }
    if (v == "lin")
    {
        out = MicrophysicsScheme::Lin;
        return true;
    }
    if (v == "thompson")
    {
        out = MicrophysicsScheme::Thompson;
        return true;
    }
    if (v == "milbrandt")
    {
        out = MicrophysicsScheme::Milbrandt;
        return true;
    }
    return false;
}

const char* microphysics_scheme_name(MicrophysicsScheme v)
{
    switch (v)
    {
    case MicrophysicsScheme::None:
        return "none";
    case MicrophysicsScheme::Kessler:
        return "kessler";
    case MicrophysicsScheme::Lin:
        return "lin";
    case MicrophysicsScheme::Thompson:
        return "thompson";
    case MicrophysicsScheme::Milbrandt:
        return "milbrandt";
    }
    return "none";
}

// ─────────────────────────────────────────────────────────────
// TurbulenceScheme
// ─────────────────────────────────────────────────────────────

bool parse_turbulence_scheme(const std::string& s, TurbulenceScheme& out)
{
    const std::string v = to_lower(s);
    if (v == "none")
    {
        out = TurbulenceScheme::None;
        return true;
    }
    if (v == "smagorinsky")
    {
        out = TurbulenceScheme::Smagorinsky;
        return true;
    }
    if (v == "tke")
    {
        out = TurbulenceScheme::TKE;
        return true;
    }
    return false;
}

const char* turbulence_scheme_name(TurbulenceScheme v)
{
    switch (v)
    {
    case TurbulenceScheme::None:
        return "none";
    case TurbulenceScheme::Smagorinsky:
        return "smagorinsky";
    case TurbulenceScheme::TKE:
        return "tke";
    }
    return "smagorinsky";
}

// ─────────────────────────────────────────────────────────────
// TurbulenceMode
// ─────────────────────────────────────────────────────────────

bool parse_turbulence_mode(const std::string& s, TurbulenceMode& out)
{
    const std::string v = to_lower(s);
    if (v == "2d")
    {
        out = TurbulenceMode::TwoD;
        return true;
    }
    if (v == "vertical_only")
    {
        out = TurbulenceMode::VerticalOnly;
        return true;
    }
    if (v == "3d")
    {
        out = TurbulenceMode::ThreeD;
        return true;
    }
    return false;
}

const char* turbulence_mode_name(TurbulenceMode v)
{
    switch (v)
    {
    case TurbulenceMode::TwoD:
        return "2d";
    case TurbulenceMode::VerticalOnly:
        return "vertical_only";
    case TurbulenceMode::ThreeD:
        return "3d";
    }
    return "3d";
}

bool parse_turbulence_filter_width(const std::string& s,
                                   TurbulenceFilterWidth& out)
{
    const std::string value = to_lower(s);
    if (value == "cubic_root" || value == "cubic-root")
        out = TurbulenceFilterWidth::CubicRoot;
    else if (value == "dx" || value == "horizontal")
        out = TurbulenceFilterWidth::Horizontal;
    else if (value == "user")
        out = TurbulenceFilterWidth::User;
    else
        return false;
    return true;
}

const char* turbulence_filter_width_name(TurbulenceFilterWidth v)
{
    switch (v)
    {
    case TurbulenceFilterWidth::CubicRoot: return "cubic_root";
    case TurbulenceFilterWidth::Horizontal: return "dx";
    case TurbulenceFilterWidth::User: return "user";
    }
    return "cubic_root";
}

bool parse_turbulence_stability_correction(
    const std::string& s, TurbulenceStabilityCorrection& out)
{
    const std::string value = to_lower(s);
    if (value == "none")
        out = TurbulenceStabilityCorrection::None;
    else if (value == "ri" || value == "richardson")
        out = TurbulenceStabilityCorrection::Richardson;
    else if (value == "tke_based" || value == "tke-based")
        out = TurbulenceStabilityCorrection::TkeBased;
    else
        return false;
    return true;
}

const char* turbulence_stability_correction_name(
    TurbulenceStabilityCorrection v)
{
    switch (v)
    {
    case TurbulenceStabilityCorrection::None: return "none";
    case TurbulenceStabilityCorrection::Richardson: return "ri";
    case TurbulenceStabilityCorrection::TkeBased: return "tke_based";
    }
    return "none";
}

// ─────────────────────────────────────────────────────────────
// AdvectionScheme
// ─────────────────────────────────────────────────────────────

bool parse_advection_scheme(const std::string& s, AdvectionScheme& out)
{
    const std::string v = to_lower(s);
    if (v == "auto")
    {
        out = AdvectionScheme::Auto;
        return true;
    }
    if (v == "none" || v == "off" || v == "disabled")
    {
        out = AdvectionScheme::None;
        return true;
    }
    if (v == "tvd" || v == "mc")
    {
        out = AdvectionScheme::TVD;
        return true;
    }
    if (v == "weno5" || v == "weno")
    {
        out = AdvectionScheme::WENO5;
        return true;
    }
    return false;
}

const char* advection_scheme_name(AdvectionScheme v)
{
    switch (v)
    {
    case AdvectionScheme::Auto:
        return "auto";
    case AdvectionScheme::None:
        return "none";
    case AdvectionScheme::TVD:
        return "tvd";
    case AdvectionScheme::WENO5:
        return "weno5";
    }
    return "auto";
}

const char* resolved_advection_scheme_name(AdvectionScheme v,
                                           StaggerType staggering)
{
    if (v == AdvectionScheme::Auto)
        return staggering == StaggerType::CGrid ? "weno5" : "tvd";
    return advection_scheme_name(v);
}

// ────────────────────────────────────────────────────────────
// DiffusionScheme
// ────────────────────────────────────────────────────────────

bool parse_diffusion_scheme(const std::string& s, DiffusionScheme& out)
{
    const std::string v = to_lower(s);
    if (v == "none")
    {
        out = DiffusionScheme::None;
        return true;
    }
    if (v == "explicit")
    {
        out = DiffusionScheme::Explicit;
        return true;
    }
    if (v == "implicit")
    {
        out = DiffusionScheme::Implicit;
        return true;
    }
    return false;
}

const char* diffusion_scheme_name(DiffusionScheme v)
{
    switch (v)
    {
    case DiffusionScheme::None:
        return "none";
    case DiffusionScheme::Explicit:
        return "explicit";
    case DiffusionScheme::Implicit:
        return "implicit";
    }
    return "explicit";
}

// ─────────────────────────────────────────────────────────────
// DiffusionTarget
// ─────────────────────────────────────────────────────────────

bool parse_diffusion_target(const std::string& s, DiffusionTarget& out)
{
    const std::string v = to_lower(s);
    if (v == "scalars")
    {
        out = DiffusionTarget::Scalars;
        return true;
    }
    if (v == "scalars_and_momentum")
    {
        out = DiffusionTarget::ScalarsAndMomentum;
        return true;
    }
    return false;
}

const char* diffusion_target_name(DiffusionTarget v)
{
    switch (v)
    {
    case DiffusionTarget::Scalars:
        return "scalars";
    case DiffusionTarget::ScalarsAndMomentum:
        return "scalars_and_momentum";
    }
    return "scalars";
}

bool parse_diffusion_coefficient_source(
    const std::string& s, DiffusionCoefficientSource& out)
{
    const std::string v = to_lower(s);
    if (v == "constant")
    {
        out = DiffusionCoefficientSource::Constant;
        return true;
    }
    if (v == "linear_height")
    {
        out = DiffusionCoefficientSource::LinearHeight;
        return true;
    }
    return false;
}

const char* diffusion_coefficient_source_name(DiffusionCoefficientSource v)
{
    switch (v)
    {
    case DiffusionCoefficientSource::Constant:
        return "constant";
    case DiffusionCoefficientSource::LinearHeight:
        return "linear_height";
    }
    return "constant";
}

// ─────────────────────────────────────────────────────────────
// SurfaceScheme
// ─────────────────────────────────────────────────────────────

bool parse_surface_scheme(const std::string& s, SurfaceScheme& out)
{
    const std::string v = to_lower(s);
    if (v == "none")
    {
        out = SurfaceScheme::None;
        return true;
    }
    if (v == "slab")
    {
        out = SurfaceScheme::Slab;
        return true;
    }
    return false;
}

const char* surface_scheme_name(SurfaceScheme v)
{
    switch (v)
    {
    case SurfaceScheme::None:
        return "none";
    case SurfaceScheme::Slab:
        return "slab";
    }
    return "slab";
}

// ─────────────────────────────────────────────────────────────
// BoundaryLayerScheme
// ─────────────────────────────────────────────────────────────

bool parse_boundary_layer_scheme(const std::string& s, BoundaryLayerScheme& out)
{
    const std::string v = to_lower(s);
    if (v == "none")
    {
        out = BoundaryLayerScheme::None;
        return true;
    }
    if (v == "slab")
    {
        out = BoundaryLayerScheme::Slab;
        return true;
    }
    if (v == "ysu")
    {
        out = BoundaryLayerScheme::YSU;
        return true;
    }
    if (v == "mynn")
    {
        out = BoundaryLayerScheme::MYNN;
        return true;
    }
    return false;
}

const char* boundary_layer_scheme_name(BoundaryLayerScheme v)
{
    switch (v)
    {
    case BoundaryLayerScheme::None:
        return "none";
    case BoundaryLayerScheme::Slab:
        return "slab";
    case BoundaryLayerScheme::YSU:
        return "ysu";
    case BoundaryLayerScheme::MYNN:
        return "mynn";
    }
    return "none";
}

bool parse_boundary_layer_top_method(
    const std::string& s, BoundaryLayerTopMethod& out)
{
    const std::string v = to_lower(s);
    if (v == "ri")
        out = BoundaryLayerTopMethod::Richardson;
    else if (v == "theta")
        out = BoundaryLayerTopMethod::PotentialTemperature;
    else if (v == "tke")
        out = BoundaryLayerTopMethod::TKE;
    else
        return false;
    return true;
}

const char* boundary_layer_top_method_name(BoundaryLayerTopMethod v)
{
    switch (v)
    {
    case BoundaryLayerTopMethod::Richardson:
        return "ri";
    case BoundaryLayerTopMethod::PotentialTemperature:
        return "theta";
    case BoundaryLayerTopMethod::TKE:
        return "tke";
    }
    return "ri";
}

// ─────────────────────────────────────────────────────────────
// RadiationScheme
// ─────────────────────────────────────────────────────────────

bool parse_radiation_scheme(const std::string& s, RadiationScheme& out)
{
    const std::string v = to_lower(s);
    if (v == "none")
    {
        out = RadiationScheme::None;
        return true;
    }
    if (v == "simple_grey")
    {
        out = RadiationScheme::SimpleGrey;
        return true;
    }
    if (v == "rrtmg")
    {
        out = RadiationScheme::RRTMG;
        return true;
    }
    return false;
}

const char* radiation_scheme_name(RadiationScheme v)
{
    switch (v)
    {
    case RadiationScheme::None:
        return "none";
    case RadiationScheme::SimpleGrey:
        return "simple_grey";
    case RadiationScheme::RRTMG:
        return "rrtmg";
    }
    return "none";
}

bool parse_radiation_solar_forcing(const std::string& s, RadiationSolarForcing& out)
{
    if (to_lower(s) == "idealized_diurnal")
    {
        out = RadiationSolarForcing::IdealizedDiurnal;
        return true;
    }
    return false;
}

const char* radiation_solar_forcing_name(RadiationSolarForcing v)
{
    switch (v)
    {
    case RadiationSolarForcing::IdealizedDiurnal:
        return "idealized_diurnal";
    }
    return "idealized_diurnal";
}

// ─────────────────────────────────────────────────────────────
// Terrain
// ─────────────────────────────────────────────────────────────

bool parse_terrain_scheme(const std::string& s, TerrainScheme& out)
{
    const std::string v = to_lower(s);
    if (v == "none" || v == "flat")
    {
        out = TerrainScheme::None;
        return true;
    }
    if (v == "bell")
    {
        out = TerrainScheme::Bell;
        return true;
    }
    if (v == "schar" || v == "schaer")
    {
        out = TerrainScheme::Schar;
        return true;
    }
    return false;
}

const char* terrain_scheme_name(TerrainScheme v)
{
    switch (v)
    {
    case TerrainScheme::None:
        return "none";
    case TerrainScheme::Bell:
        return "bell";
    case TerrainScheme::Schar:
        return "schar";
    }
    return "none";
}

bool parse_terrain_coordinate(const std::string& s, TerrainCoordinate& out)
{
    const std::string v = to_lower(s);
    if (v == "btf" || v == "terrain_following" || v == "terrain-following" || v == "sigma")
    {
        out = TerrainCoordinate::BasicTerrainFollowing;
        return true;
    }
    if (v == "smoothed" || v == "smooth")
    {
        out = TerrainCoordinate::Smoothed;
        return true;
    }
    if (v == "cartesian" || v == "flat" || v == "none")
    {
        out = TerrainCoordinate::Cartesian;
        return true;
    }
    return false;
}

const char* terrain_coordinate_name(TerrainCoordinate v)
{
    switch (v)
    {
    case TerrainCoordinate::BasicTerrainFollowing:
        return "btf";
    case TerrainCoordinate::Smoothed:
        return "smoothed";
    case TerrainCoordinate::Cartesian:
        return "cartesian";
    }
    return "btf";
}

bool parse_terrain_horizontal_placement(
    const std::string& s, TerrainHorizontalPlacement& out)
{
    const std::string v = to_lower(s);
    if (v == "grid_attached")
    {
        out = TerrainHorizontalPlacement::GridAttached;
        return true;
    }
    if (v == "earth_fixed")
    {
        out = TerrainHorizontalPlacement::EarthFixed;
        return true;
    }
    return false;
}

const char* terrain_horizontal_placement_name(TerrainHorizontalPlacement v)
{
    switch (v)
    {
    case TerrainHorizontalPlacement::GridAttached:
        return "grid_attached";
    case TerrainHorizontalPlacement::EarthFixed:
        return "earth_fixed";
    }
    return "grid_attached";
}

// ─────────────────────────────────────────────────────────────
} // namespace tmv::init::config
