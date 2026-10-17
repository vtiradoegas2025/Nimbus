/**
 * @file colormap_lut.hpp
 * @brief Scientific colormap lookup tables for scalar-field + radar rendering.
 *
 * Standalone, GPU-agnostic colormap module for S7.1 (Validation Plan §3.1).
 * Produces RGBA8 colors and LUT arrays that a render backend can upload as a
 * 1D R8G8B8A8_UNORM texture. Intentionally free of any Vulkan dependency so it
 * can be unit-tested without a device/loader; the renderer-side wiring
 * (volume.frag / volume_backend) lives on the S7.2 track and is NOT done here.
 *
 * Two colormap families:
 *   - NWS NEXRAD reflectivity scale (dBZ domain, stepped, with a reflectivity
 *     floor below which samples are transparent).
 *   - Perceptual scalar colormaps (viridis / inferno / coolwarm) for generic
 *     fields, sampled over a normalized [0, 1] domain.
 *
 * The reflectivity domain floor aligns with ReflectivityScheme::Z_min_dBZ
 * (-30 dBZ) in src/radar/schemes/reflectivity/reflectivity.hpp; the visible
 * NWS color scale begins at +5 dBZ (kNwsReflectivityFloorDbz).
 */

#pragma once

#include <cstdint>
#include <vector>

namespace vkcpp
{

/**
 * @brief 8-bit-per-channel RGBA color — the storage format of a LUT texel.
 */
struct ColorRGBA8
{
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
    std::uint8_t a = 255;
};

/** @brief Exact equality across all four channels. */
constexpr bool operator==(const ColorRGBA8& lhs, const ColorRGBA8& rhs)
{
    return lhs.r == rhs.r && lhs.g == rhs.g && lhs.b == rhs.b && lhs.a == rhs.a;
}
constexpr bool operator!=(const ColorRGBA8& lhs, const ColorRGBA8& rhs)
{
    return !(lhs == rhs);
}

/**
 * @brief Perceptual scalar colormaps for generic (non-radar) fields.
 *
 * Values mirror matplotlib's reference data so figures read the same as the
 * source-paper colorbars. Sampled over a normalized scalar domain [0, 1].
 */
enum class ScalarColormap
{
    Viridis,
    Inferno,
    Coolwarm
};

/**
 * @brief First visible dBZ on the NWS reflectivity scale.
 *
 * Reflectivity below this floor is considered "no echo" and rendered fully
 * transparent (alpha = 0) — the "reflectivity floor" of S7.1. Forecast radar
 * products conventionally start the color scale near +5 dBZ.
 */
constexpr float kNwsReflectivityFloorDbz = 5.0f;

/** @brief Top of the NWS reflectivity scale; >= this value renders white. */
constexpr float kNwsReflectivityMaxDbz = 75.0f;

/**
 * @brief Map a reflectivity value (dBZ) to a color on the NWS NEXRAD scale.
 *
 * The NWS reflectivity legend is a *stepped* scale: each 5 dBZ band maps to a
 * fixed published color. This reproduces that table exactly (no interpolation),
 * which is what the §3.1 acceptance ("match the published NWS reference table
 * to within 1 LSB") requires.
 *
 * @param dbz       Reflectivity in dBZ.
 * @param floor_dbz Values below this render transparent (alpha 0). Defaults to
 *                  the standard +5 dBZ floor; callers may raise it to suppress
 *                  light returns.
 * @return Opaque banded color at/above the floor, or transparent below it.
 */
ColorRGBA8 nws_reflectivity_color(float dbz, float floor_dbz = kNwsReflectivityFloorDbz);

/**
 * @brief Sample a perceptual scalar colormap at normalized position @p t.
 *
 * @param cmap Which colormap.
 * @param t    Position in [0, 1]; clamped to that range.
 * @return Opaque RGBA color.
 */
ColorRGBA8 sample_scalar_colormap(ScalarColormap cmap, float t);

/**
 * @brief Build a @p size -entry LUT for a perceptual colormap.
 *
 * Entry i samples the colormap at t = i / (size - 1), so the first and last
 * entries are exactly the colormap endpoints. Suitable for direct upload as a
 * 1D R8G8B8A8_UNORM texture.
 *
 * @param cmap Which colormap.
 * @param size Number of texels (>= 2; values < 2 yield an empty vector).
 */
std::vector<ColorRGBA8> build_scalar_lut(ScalarColormap cmap, int size = 256);

/**
 * @brief Build a @p size -entry NWS reflectivity LUT spanning a dBZ domain.
 *
 * Entry i corresponds to dBZ = dbz_min + (dbz_max - dbz_min) * i / (size - 1),
 * colored via nws_reflectivity_color(). The stepped scale is preserved, so the
 * texture shows the discrete NWS bands and a transparent run below @p floor_dbz.
 *
 * @param size     Number of texels (>= 2; values < 2 yield an empty vector).
 * @param dbz_min  dBZ mapped to texel 0.
 * @param dbz_max  dBZ mapped to the final texel.
 * @param floor_dbz Reflectivity floor (see nws_reflectivity_color).
 */
std::vector<ColorRGBA8> build_reflectivity_lut(int size,
                                               float dbz_min,
                                               float dbz_max,
                                               float floor_dbz = kNwsReflectivityFloorDbz);

}  // namespace vkcpp
