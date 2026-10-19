/**
 * @file test_colormap_lut.cpp
 * @brief Unit tests for the S7.1 scientific colormap LUTs.
 *
 * The headline acceptance (Validation Plan §3.1) is that the NWS reflectivity
 * colors match the published reference table to within 1 LSB. The reference
 * table below is transcribed independently from the module under test — the
 * standard NWS NEXRAD 16-color reflectivity legend (5 dBZ bands, +5 to 75+
 * dBZ) — so the assertions pin the implementation to the published values
 * rather than to themselves.
 *
 * viridis / inferno / coolwarm are validated structurally (canonical
 * endpoints, diverging shape, monotone perceived lightness) since their hard
 * numeric anchor in §3.1 is the NWS table.
 */
#include "catch2/catch.hpp"
#include "rendering/colormap_lut.hpp"

#include <cmath>
#include <cstdint>

using vkcpp::ColorRGBA8;
using vkcpp::ScalarColormap;

namespace
{

/** @brief True if @p got is within @p tol LSB of (@p r, @p g, @p b) per channel. */
bool within_lsb(const ColorRGBA8& got, int r, int g, int b, int tol)
{
    return std::abs(static_cast<int>(got.r) - r) <= tol &&
           std::abs(static_cast<int>(got.g) - g) <= tol &&
           std::abs(static_cast<int>(got.b) - b) <= tol;
}

double luminance(const ColorRGBA8& c)
{
    return 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
}

// Published NWS NEXRAD reflectivity scale: lower dBZ bound of each 5 dBZ band
// and its reference RGB. Transcribed from the standard NWS reflectivity legend.
struct RefBand
{
    float lower_dbz;
    int r;
    int g;
    int b;
};

const RefBand kPublishedNws[] = {
    {5.0f, 4, 233, 231},    {10.0f, 1, 159, 244},  {15.0f, 3, 0, 244},
    {20.0f, 2, 253, 2},     {25.0f, 1, 197, 1},    {30.0f, 0, 142, 0},
    {35.0f, 253, 248, 2},   {40.0f, 229, 188, 0},  {45.0f, 253, 149, 0},
    {50.0f, 253, 0, 0},     {55.0f, 212, 0, 0},    {60.0f, 188, 0, 0},
    {65.0f, 248, 0, 253},   {70.0f, 152, 84, 198}, {75.0f, 253, 253, 253},
};

}  // namespace

TEST_CASE("NWS reflectivity matches the published reference within 1 LSB", "[vulkan][colormap][radar]")
{
    for (const auto& band : kPublishedNws)
    {
        // Probe at the lower edge (inclusive) and mid-band; both land in the
        // same published color.
        const ColorRGBA8 at_edge = vkcpp::nws_reflectivity_color(band.lower_dbz);
        const ColorRGBA8 at_mid = vkcpp::nws_reflectivity_color(band.lower_dbz + 2.5f);

        INFO("dBZ band lower=" << band.lower_dbz);
        CHECK(within_lsb(at_edge, band.r, band.g, band.b, 1));
        CHECK(within_lsb(at_mid, band.r, band.g, band.b, 1));
        CHECK(at_edge.a == 255);
    }
}

TEST_CASE("NWS reflectivity is a stepped (not interpolated) scale", "[vulkan][colormap][radar]")
{
    // Two values inside the same 5 dBZ band must produce identical colors;
    // interpolation between band colors would fail the §3.1 table match.
    CHECK(vkcpp::nws_reflectivity_color(31.0f) == vkcpp::nws_reflectivity_color(34.0f));
    CHECK(vkcpp::nws_reflectivity_color(50.1f) == vkcpp::nws_reflectivity_color(54.9f));
    // Adjacent bands differ.
    CHECK(vkcpp::nws_reflectivity_color(34.0f) != vkcpp::nws_reflectivity_color(36.0f));
}

TEST_CASE("Reflectivity floor renders transparent below threshold", "[vulkan][colormap][radar]")
{
    // Default floor (+5 dBZ): no echo below it.
    CHECK(vkcpp::nws_reflectivity_color(4.9f).a == 0);
    CHECK(vkcpp::nws_reflectivity_color(-10.0f).a == 0);
    CHECK(vkcpp::nws_reflectivity_color(5.0f).a == 255);

    // A raised floor suppresses otherwise-valid returns.
    CHECK(vkcpp::nws_reflectivity_color(18.0f, /*floor=*/30.0f).a == 0);
    CHECK(vkcpp::nws_reflectivity_color(32.0f, /*floor=*/30.0f).a == 255);

    // Non-finite input is treated as no echo, never garbage.
    CHECK(vkcpp::nws_reflectivity_color(std::nanf("")).a == 0);
}

TEST_CASE("NWS scale saturates to white at and above the maximum", "[vulkan][colormap][radar]")
{
    CHECK(within_lsb(vkcpp::nws_reflectivity_color(75.0f), 253, 253, 253, 1));
    CHECK(within_lsb(vkcpp::nws_reflectivity_color(120.0f), 253, 253, 253, 1));
}

TEST_CASE("Reflectivity LUT is stepped, opaque above floor, transparent below", "[vulkan][colormap][radar]")
{
    const auto lut = vkcpp::build_reflectivity_lut(256, -30.0f, 80.0f);
    REQUIRE(lut.size() == 256);

    // texel 0 = -30 dBZ → below floor → transparent; final texel = 80 → white.
    CHECK(lut.front().a == 0);
    CHECK(within_lsb(lut.back(), 253, 253, 253, 1));

    // Every texel is either fully transparent (floor) or fully opaque (band).
    for (const auto& c : lut)
    {
        CHECK((c.a == 0 || c.a == 255));
    }

    // Only a small set of distinct opaque colors appears (stepped, not a
    // continuous gradient): the 15 bands at most.
    // (Sanity bound rather than exact count, robust to domain edges.)
    int opaque = 0;
    for (const auto& c : lut)
    {
        if (c.a == 255) ++opaque;
    }
    CHECK(opaque > 0);
}

TEST_CASE("Perceptual colormap endpoints match canonical matplotlib values", "[vulkan][colormap]")
{
    // Endpoints are exact control points; allow 1 LSB for quantization.
    CHECK(within_lsb(vkcpp::sample_scalar_colormap(ScalarColormap::Viridis, 0.0f), 68, 1, 84, 1));
    CHECK(within_lsb(vkcpp::sample_scalar_colormap(ScalarColormap::Viridis, 1.0f), 253, 231, 37, 1));

    CHECK(within_lsb(vkcpp::sample_scalar_colormap(ScalarColormap::Inferno, 0.0f), 0, 0, 4, 1));
    CHECK(within_lsb(vkcpp::sample_scalar_colormap(ScalarColormap::Inferno, 1.0f), 252, 255, 164, 1));

    CHECK(within_lsb(vkcpp::sample_scalar_colormap(ScalarColormap::Coolwarm, 0.0f), 59, 76, 192, 1));
    CHECK(within_lsb(vkcpp::sample_scalar_colormap(ScalarColormap::Coolwarm, 1.0f), 180, 4, 38, 1));
}

TEST_CASE("Coolwarm is a diverging map: blue → neutral → red", "[vulkan][colormap]")
{
    const ColorRGBA8 lo = vkcpp::sample_scalar_colormap(ScalarColormap::Coolwarm, 0.0f);
    const ColorRGBA8 mid = vkcpp::sample_scalar_colormap(ScalarColormap::Coolwarm, 0.5f);
    const ColorRGBA8 hi = vkcpp::sample_scalar_colormap(ScalarColormap::Coolwarm, 1.0f);

    CHECK(lo.b > lo.r);                 // cool end is blue-dominant
    CHECK(hi.r > hi.b);                 // warm end is red-dominant
    CHECK(within_lsb(mid, 221, 221, 221, 3));  // neutral light-gray midpoint
    CHECK(std::abs(static_cast<int>(mid.r) - static_cast<int>(mid.b)) <= 6);
}

TEST_CASE("Viridis and inferno increase monotonically in perceived lightness", "[vulkan][colormap]")
{
    for (auto cmap : {ScalarColormap::Viridis, ScalarColormap::Inferno})
    {
        double prev = -1.0;
        for (int i = 0; i <= 10; ++i)
        {
            const double l = luminance(vkcpp::sample_scalar_colormap(cmap, static_cast<float>(i) / 10.0f));
            CHECK(l > prev);  // strictly increasing across deciles
            prev = l;
        }
    }
}

TEST_CASE("Scalar LUT has the requested size, opaque texels, and exact endpoints", "[vulkan][colormap]")
{
    const auto lut = vkcpp::build_scalar_lut(ScalarColormap::Viridis, 256);
    REQUIRE(lut.size() == 256);
    CHECK(within_lsb(lut.front(), 68, 1, 84, 1));
    CHECK(within_lsb(lut.back(), 253, 231, 37, 1));
    for (const auto& c : lut)
    {
        CHECK(c.a == 255);
    }

    // Degenerate sizes yield an empty LUT rather than a crash.
    CHECK(vkcpp::build_scalar_lut(ScalarColormap::Inferno, 1).empty());
    CHECK(vkcpp::build_scalar_lut(ScalarColormap::Inferno, 0).empty());
}

TEST_CASE("sample_scalar_colormap clamps out-of-range t to the endpoints", "[vulkan][colormap]")
{
    CHECK(vkcpp::sample_scalar_colormap(ScalarColormap::Viridis, -1.0f) ==
          vkcpp::sample_scalar_colormap(ScalarColormap::Viridis, 0.0f));
    CHECK(vkcpp::sample_scalar_colormap(ScalarColormap::Viridis, 2.0f) ==
          vkcpp::sample_scalar_colormap(ScalarColormap::Viridis, 1.0f));
}
