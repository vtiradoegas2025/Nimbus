/**
 * @file colormap_lut.cpp
 * @brief Implementation of the scientific colormap LUTs (see colormap_lut.hpp).
 *
 * Reference data:
 *   - NWS NEXRAD reflectivity scale: the standard 16-color reflectivity legend
 *     (5 dBZ bands from +5 to 75+ dBZ). Published, e.g., on the NWS NEXRAD
 *     reflectivity product legend; the same hex values are reproduced in the
 *     unit test (tests/vulkan/test_colormap_lut.cpp) as an independent check.
 *   - viridis / inferno / coolwarm: matplotlib reference colormap data,
 *     sampled at deciles and linearly interpolated. Endpoints match the
 *     canonical matplotlib values exactly after 8-bit quantization.
 */

#include "rendering/colormap_lut.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace vkcpp
{
namespace
{

/** @brief Quantize a normalized [0,1] channel to 8-bit with round-to-nearest. */
std::uint8_t to_u8(float x)
{
    const float clamped = std::clamp(x, 0.0f, 1.0f);
    return static_cast<std::uint8_t>(std::lround(clamped * 255.0f));
}

/* ---------------------------------------------------------------------------
 * NWS NEXRAD reflectivity scale (stepped).
 *
 * Each entry is { lower_dbz_inclusive, color }. Bands are 5 dBZ wide; the
 * final band (>= 75 dBZ) is open-ended. Values below the first band are
 * handled by the floor logic in nws_reflectivity_color().
 * ------------------------------------------------------------------------- */
struct ReflectivityBand
{
    float lower_dbz;
    ColorRGBA8 color;
};

constexpr float kNwsFirstBandDbz = 5.0f;

const std::array<ReflectivityBand, 15> kNwsReflectivityBands = {{
    {5.0f, {4, 233, 231, 255}},     // 5-10  dBZ  light cyan
    {10.0f, {1, 159, 244, 255}},    // 10-15       blue
    {15.0f, {3, 0, 244, 255}},      // 15-20       dark blue
    {20.0f, {2, 253, 2, 255}},      // 20-25       green
    {25.0f, {1, 197, 1, 255}},      // 25-30       green
    {30.0f, {0, 142, 0, 255}},      // 30-35       dark green
    {35.0f, {253, 248, 2, 255}},    // 35-40       yellow
    {40.0f, {229, 188, 0, 255}},    // 40-45       gold
    {45.0f, {253, 149, 0, 255}},    // 45-50       orange
    {50.0f, {253, 0, 0, 255}},      // 50-55       red
    {55.0f, {212, 0, 0, 255}},      // 55-60       dark red
    {60.0f, {188, 0, 0, 255}},      // 60-65       darker red
    {65.0f, {248, 0, 253, 255}},    // 65-70       magenta
    {70.0f, {152, 84, 198, 255}},   // 70-75       purple
    {75.0f, {253, 253, 253, 255}},  // 75+         white (large hail)
}};

/* ---------------------------------------------------------------------------
 * Perceptual colormap control points (matplotlib reference data), evenly
 * spaced over [0, 1]. Endpoints are the canonical matplotlib endpoints.
 * ------------------------------------------------------------------------- */
using Rgb = std::array<float, 3>;

// viridis at deciles t = 0.0 .. 1.0.
const std::array<Rgb, 11> kViridis = {{
    {0.267004f, 0.004874f, 0.329415f},
    {0.282623f, 0.140926f, 0.457517f},
    {0.253935f, 0.265254f, 0.529983f},
    {0.206756f, 0.371758f, 0.553117f},
    {0.163625f, 0.471133f, 0.558148f},
    {0.127568f, 0.566949f, 0.550556f},
    {0.134692f, 0.658636f, 0.517649f},
    {0.266941f, 0.748751f, 0.440573f},
    {0.477504f, 0.821444f, 0.318195f},
    {0.741388f, 0.873449f, 0.149561f},
    {0.993248f, 0.906157f, 0.143936f},
}};

// inferno at deciles t = 0.0 .. 1.0.
const std::array<Rgb, 11> kInferno = {{
    {0.001462f, 0.000466f, 0.013866f},
    {0.087411f, 0.044556f, 0.224813f},
    {0.258234f, 0.038571f, 0.406485f},
    {0.416331f, 0.090203f, 0.432943f},
    {0.578304f, 0.148039f, 0.404411f},
    {0.735683f, 0.215906f, 0.330245f},
    {0.865006f, 0.316822f, 0.226055f},
    {0.954506f, 0.468744f, 0.099874f},
    {0.987622f, 0.645320f, 0.039886f},
    {0.964394f, 0.843848f, 0.273391f},
    {0.988362f, 0.998364f, 0.644924f},
}};

// coolwarm (Moreland diverging) — exact blue/gray/red anchors with two
// intermediate tints. Quartile spacing (5 points).
const std::array<Rgb, 5> kCoolwarm = {{
    {0.229806f, 0.298717f, 0.753683f},  // 0.00  blue
    {0.556700f, 0.690000f, 0.995600f},  // 0.25  light blue
    {0.865400f, 0.865400f, 0.865400f},  // 0.50  neutral gray
    {0.960300f, 0.627600f, 0.518100f},  // 0.75  light red
    {0.705673f, 0.015556f, 0.150233f},  // 1.00  red
}};

/** @brief Linearly interpolate a control-point ramp at normalized @p t. */
template <std::size_t N>
ColorRGBA8 sample_ramp(const std::array<Rgb, N>& ramp, float t)
{
    static_assert(N >= 2, "ramp needs at least two control points");
    const float clamped = std::clamp(t, 0.0f, 1.0f);
    const float scaled = clamped * static_cast<float>(N - 1);
    const std::size_t lo = std::min(static_cast<std::size_t>(scaled), N - 2);
    const float frac = scaled - static_cast<float>(lo);

    const Rgb& a = ramp[lo];
    const Rgb& b = ramp[lo + 1];
    return ColorRGBA8{
        to_u8(a[0] + (b[0] - a[0]) * frac),
        to_u8(a[1] + (b[1] - a[1]) * frac),
        to_u8(a[2] + (b[2] - a[2]) * frac),
        255,
    };
}

}  // namespace

ColorRGBA8 nws_reflectivity_color(float dbz, float floor_dbz)
{
    // Transparent ("no echo") below the floor or below the start of the
    // published scale — the renderer skips these texels.
    if (!std::isfinite(dbz) || dbz < floor_dbz || dbz < kNwsFirstBandDbz)
    {
        return ColorRGBA8{0, 0, 0, 0};
    }

    // Highest band whose lower bound is <= dbz. Bands are ascending, so walk
    // from the top; the >= 75 band catches everything above the scale.
    for (auto it = kNwsReflectivityBands.rbegin(); it != kNwsReflectivityBands.rend(); ++it)
    {
        if (dbz >= it->lower_dbz)
        {
            return it->color;
        }
    }
    // Unreachable given the floor check above, but keep it safe.
    return ColorRGBA8{0, 0, 0, 0};
}

ColorRGBA8 sample_scalar_colormap(ScalarColormap cmap, float t)
{
    switch (cmap)
    {
        case ScalarColormap::Viridis:
            return sample_ramp(kViridis, t);
        case ScalarColormap::Inferno:
            return sample_ramp(kInferno, t);
        case ScalarColormap::Coolwarm:
            return sample_ramp(kCoolwarm, t);
    }
    return ColorRGBA8{0, 0, 0, 255};
}

std::vector<ColorRGBA8> build_scalar_lut(ScalarColormap cmap, int size)
{
    std::vector<ColorRGBA8> lut;
    if (size < 2)
    {
        return lut;
    }
    lut.reserve(static_cast<std::size_t>(size));
    const float denom = static_cast<float>(size - 1);
    for (int i = 0; i < size; ++i)
    {
        lut.push_back(sample_scalar_colormap(cmap, static_cast<float>(i) / denom));
    }
    return lut;
}

std::vector<ColorRGBA8> build_reflectivity_lut(int size,
                                               float dbz_min,
                                               float dbz_max,
                                               float floor_dbz)
{
    std::vector<ColorRGBA8> lut;
    if (size < 2)
    {
        return lut;
    }
    lut.reserve(static_cast<std::size_t>(size));
    const float denom = static_cast<float>(size - 1);
    for (int i = 0; i < size; ++i)
    {
        const float dbz = dbz_min + (dbz_max - dbz_min) * (static_cast<float>(i) / denom);
        lut.push_back(nws_reflectivity_color(dbz, floor_dbz));
    }
    return lut;
}

}  // namespace vkcpp
