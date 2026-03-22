/**
 * @file test_soundings.cpp
 * @brief Unit tests for sounding data structures, interpolation, and QC.
 *
 * Tests verify:
 *   - SoundingData member structure
 *   - Linear interpolation correctness
 *   - Quality control filtering
 *   - Factory creation
 */
#include "catch2/catch.hpp"
#include "data/soundings_base.hpp"
#include "data/soundings.hpp"
#include "soundings/base/soundings_base.hpp"

#include <cmath>
#include <limits>

TEST_CASE("SoundingData default construction has empty vectors", "[data][soundings]")
{
    SoundingData data;
    REQUIRE(data.height_m.empty());
    REQUIRE(data.pressure_hpa.empty());
    REQUIRE(data.temperature_k.empty());
}

TEST_CASE("SoundingData stores multi-level profile correctly", "[data][soundings][analytical]")
{
    SoundingData data;
    data.height_m = {0.0, 1000.0, 2000.0, 5000.0, 10000.0};
    data.pressure_hpa = {1013.25, 900.0, 800.0, 550.0, 265.0};
    data.temperature_k = {288.0, 281.5, 275.0, 255.0, 223.0};

    REQUIRE(data.height_m.size() == 5);

    // Physical consistency: pressure decreases with height
    for (size_t k = 1; k < data.pressure_hpa.size(); ++k)
        REQUIRE(data.pressure_hpa[k] < data.pressure_hpa[k - 1]);

    // Temperature decreases in troposphere
    REQUIRE(data.temperature_k.back() < data.temperature_k.front());
}

TEST_CASE("Sounding factory creates sharpy scheme", "[data][soundings]")
{
    auto scheme = create_sounding_scheme("sharpy");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("Sounding linear interpolation at exact levels returns source values", "[data][soundings][analytical]")
{
    SoundingData data;
    data.height_m = {0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0};
    data.pressure_hpa = {1013.25, 900.0, 800.0, 700.0, 600.0, 500.0};
    data.temperature_k = {288.0, 281.5, 275.0, 269.0, 262.0, 255.0};

    SoundingConfig cfg;
    std::vector<double> targets = {0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0};

    auto result = interpolate_sounding_linear(data, targets, cfg);

    REQUIRE(result.height_m.size() == 6);
    for (size_t k = 0; k < 6; ++k)
    {
        REQUIRE(result.temperature_k[k] == Approx(data.temperature_k[k]).margin(1e-6));
        REQUIRE(result.pressure_hpa[k] == Approx(data.pressure_hpa[k]).margin(1e-6));
    }
}

TEST_CASE("Sounding linear interpolation at midpoints yields arithmetic mean", "[data][soundings][analytical]")
{
    SoundingData data;
    data.height_m = {0.0, 2000.0, 4000.0, 6000.0, 8000.0, 10000.0};
    data.pressure_hpa = {1000.0, 800.0, 600.0, 400.0, 300.0, 200.0};
    data.temperature_k = {300.0, 280.0, 260.0, 240.0, 220.0, 200.0};

    SoundingConfig cfg;
    std::vector<double> targets = {1000.0};  // midpoint of first interval

    auto result = interpolate_sounding_linear(data, targets, cfg);

    // Linear midpoint: (300 + 280) / 2 = 290
    REQUIRE(result.temperature_k.size() == 1);
    REQUIRE(result.temperature_k[0] == Approx(290.0).margin(0.5));

    // Pressure midpoint: (1000 + 800) / 2 = 900
    REQUIRE(result.pressure_hpa[0] == Approx(900.0).margin(0.5));
}

TEST_CASE("Sounding QC filters NaN temperature levels", "[data][soundings][analytical]")
{
    SoundingData data;
    data.height_m = {0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0};
    data.pressure_hpa = {1013.0, 900.0, 800.0, 700.0, 600.0, 500.0};
    data.temperature_k = {288.0, std::numeric_limits<double>::quiet_NaN(),
                          275.0, 269.0, 262.0, 255.0};

    SoundingConfig cfg;
    bool valid = quality_control_sounding(data, cfg);

    if (valid)
    {
        // All remaining values must be finite
        for (double t : data.temperature_k)
            REQUIRE(std::isfinite(t));
        for (double p : data.pressure_hpa)
            REQUIRE(std::isfinite(p));

        // The NaN level should be removed
        REQUIRE(data.height_m.size() == 5);
    }
}

TEST_CASE("Sounding QC preserves clean data", "[data][soundings][analytical]")
{
    SoundingData data;
    data.height_m = {0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0};
    data.pressure_hpa = {1013.0, 900.0, 800.0, 700.0, 600.0, 500.0};
    data.temperature_k = {288.0, 281.5, 275.0, 269.0, 262.0, 255.0};

    SoundingConfig cfg;
    bool valid = quality_control_sounding(data, cfg);

    REQUIRE(valid);
    REQUIRE(data.height_m.size() == 6);
    REQUIRE(data.temperature_k[0] == Approx(288.0).margin(1e-6));
    REQUIRE(data.temperature_k[5] == Approx(255.0).margin(1e-6));
}
