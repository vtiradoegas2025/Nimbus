#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/triple_cloud_adjustment.hpp"
#include "support/milbrandt_cloud_adjustment_reference.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace
{

using namespace tmv;
using namespace tmv::milbrandt_detail;

void compare(const TripleCloudAdjustmentCell& cell, const TripleCloudAdjustmentResult& result)
{
    const auto expected = cloud_adjustment_reference::evaluate(cell);
    const auto& e = result.endpoints();
    CHECK(e.vapor_per_kg == Approx(static_cast<double>(expected.vapor)).epsilon(3e-12).margin(3e-15));
    CHECK(e.cloud_per_kg == Approx(static_cast<double>(expected.cloud)).epsilon(3e-12).margin(3e-15));
    CHECK(e.cloud_number_per_kg == Approx(static_cast<double>(expected.number)).epsilon(3e-12).margin(1e-12));
    CHECK(e.temperature_increment_k == Approx(static_cast<double>(expected.heat)).epsilon(3e-12).margin(1e-11));
    CHECK(e.temperature_after_k == Approx(static_cast<double>(expected.temperature)).epsilon(3e-13));
    CHECK(e.activated_number_per_m3 == Approx(static_cast<double>(expected.target)).epsilon(3e-12).margin(1e-12));
    CHECK(e.vapor_per_kg + e.cloud_per_kg == Approx(cell.vapor_per_kg + cell.cloud_per_kg).epsilon(2e-14));
    CHECK(e.temperature_increment_k == microphysics_constants::L_v / microphysics_constants::cp * e.admitted_cloud_increment_per_kg);
    CHECK(cell.dry_density_kg_m3 * (e.vapor_per_kg + e.cloud_per_kg)
        == Approx(cell.dry_density_kg_m3 * (cell.vapor_per_kg + cell.cloud_per_kg)).epsilon(2e-14));
}

bool same_result(const TripleCloudAdjustmentResult& a, const TripleCloudAdjustmentResult& b)
{
    return std::memcmp(&a.source_cell(), &b.source_cell(), sizeof(TripleCloudAdjustmentCell)) == 0
        && std::memcmp(&a.endpoints(), &b.endpoints(), sizeof(TripleCloudAdjustmentEndpoints)) == 0
        && a.duration_s() == b.duration_s();
}

} // namespace

TEST_CASE("exact cloud adjustment closes independent saturation activation and evaporative population references",
          "[physics][milbrandt][triple_cloud_adjustment]")
{
    for (double temperature : {280.0, 290.0, 305.0})
    {
        for (double pressure : {65000.0, 90000.0})
        {
            for (double density : {.65, 1.43})
            {
                for (double velocity : {-.05, .001, .5, 5.0})
                {
                    const double qs = thermodynamics::saturation_mixing_ratio_water(temperature, pressure);
                    const std::array<TripleCloudAdjustmentCell, 5> cells{{
                        {temperature, pressure, density, 1.2 * qs, 0, 0, velocity},
                        {temperature, pressure, density, 1.05 * qs, .001, 8e8, velocity},
                        {temperature, pressure, density, .4 * qs, .02, 1e8, velocity},
                        {temperature, pressure, density, 0, 1e-4, 1e5, velocity},
                        {temperature, pressure, density, qs, .001, 1e5, velocity}}};
                    for (const auto& cell : cells)
                    {
                        for (double duration : {.05, .3})
                        {
                            TripleCloudAdjustmentResult result;
                            std::string error;
                            const bool accepted = produce_exact_triple_cloud_adjustment(cell, duration, result, error);
                            INFO("T " << temperature << ", p " << pressure << ", rho " << density << ", w " << velocity
                                << ", vapor " << cell.vapor_per_kg << ", cloud " << cell.cloud_per_kg << ": " << error);
                            REQUIRE(accepted);
                            compare(cell, result);
                            TripleCloudAdjustmentResult replay;
                            REQUIRE(apply_exact_triple_cloud_adjustment(cell, duration, result, replay, error));
                            CHECK(same_result(result, replay));
                            if (cell.vapor_per_kg == 0)
                            {
                                CHECK(result.endpoints().cloud_donor_exhausted);
                                CHECK(result.endpoints().cloud_per_kg == 0);
                                CHECK(result.endpoints().cloud_number_per_kg == 0);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("cloud adjustment retains sharp saturation and one millimeter per second activation branches",
          "[physics][milbrandt][triple_cloud_adjustment]")
{
    const double qs = thermodynamics::saturation_mixing_ratio_water(290, 90000);
    for (double velocity : {std::nextafter(.001, 0.0), .001, std::nextafter(.001, 1.0)})
    {
        for (double vapor : {std::nextafter(qs, 0.0), qs, std::nextafter(qs, 1.0)})
        {
            TripleCloudAdjustmentCell cell{290, 90000, .65, vapor, 1e-4, 1e5, velocity};
            TripleCloudAdjustmentResult result;
            std::string error;
            const bool accepted = produce_exact_triple_cloud_adjustment(cell, .1, result, error);
            INFO(error);
            REQUIRE(accepted);
            const auto& e = result.endpoints();
            if (vapor > qs)
            {
                CHECK(e.admitted_cloud_increment_per_kg > 0);
                CHECK(e.activated_number_per_m3 == Approx(static_cast<double>(
                    cloud_adjustment_reference::activated(velocity, e.temperature_after_k, cell.pressure_pa))).epsilon(3e-12));
                if (velocity <= .001)
                {
                    CHECK(e.activated_number_per_m3 == 2e8);
                }
            }
            else if (vapor < qs)
            {
                CHECK(e.admitted_cloud_increment_per_kg < 0);
                CHECK(e.activated_number_per_m3 == 0);
            }
            else
            {
                CHECK(e.admitted_cloud_increment_per_kg == 0);
                CHECK(e.cloud_number_per_kg == cell.cloud_number_per_kg);
            }
        }
    }
}

TEST_CASE("cloud producer rejects every forged source duration and endpoint without publication",
          "[physics][milbrandt][triple_cloud_adjustment]")
{
    const double qs = thermodynamics::saturation_mixing_ratio_water(290, 90000);
    const TripleCloudAdjustmentCell cell{290, 90000, .65, 1.2 * qs, 0, 0, .5};
    TripleCloudAdjustmentResult baseline;
    std::string error;
    REQUIRE(produce_exact_triple_cloud_adjustment(cell, .1, baseline, error));
    for (int mutation = 0; mutation < 7; ++mutation)
    {
        auto source = cell;
        const std::array<double*, 7> fields{&source.temperature_k, &source.pressure_pa, &source.dry_density_kg_m3,
            &source.vapor_per_kg, &source.cloud_per_kg, &source.cloud_number_per_kg, &source.vertical_velocity_ms};
        *fields[mutation] = std::nextafter(*fields[mutation], 1e20);
        auto result = baseline;
        CHECK_FALSE(apply_exact_triple_cloud_adjustment(source, .1, baseline, result, error));
        CHECK(same_result(result, baseline));
    }
    for (int mutation = 0; mutation < 8; ++mutation)
    {
        auto altered = baseline;
        auto& e = const_cast<TripleCloudAdjustmentEndpoints&>(altered.endpoints());
        const std::array<double*, 7> fields{&e.vapor_per_kg, &e.cloud_per_kg, &e.cloud_number_per_kg,
            &e.temperature_increment_k, &e.temperature_after_k, &e.admitted_cloud_increment_per_kg,
            &e.activated_number_per_m3};
        if (mutation < 7)
        {
            *fields[mutation] = std::nextafter(*fields[mutation], 1e20);
        }
        else
        {
            e.cloud_donor_exhausted = true;
        }
        auto result = baseline;
        CHECK_FALSE(apply_exact_triple_cloud_adjustment(cell, .1, altered, result, error));
        CHECK(same_result(result, baseline));
    }
    for (double duration : {.3, 0.0, -1.0, std::numeric_limits<double>::quiet_NaN()})
    {
        auto result = baseline;
        CHECK_FALSE(apply_exact_triple_cloud_adjustment(cell, duration, baseline, result, error));
        CHECK(same_result(result, baseline));
    }
    auto signed_zero = cell;
    signed_zero.cloud_per_kg = -0.0;
    auto result = baseline;
    CHECK_FALSE(apply_exact_triple_cloud_adjustment(signed_zero, .1, baseline, result, error));
    CHECK(same_result(result, baseline));
}

TEST_CASE("cloud adjustment invalid physical inputs range and thermal exits preserve the previous result",
          "[physics][milbrandt][triple_cloud_adjustment]")
{
    const TripleCloudAdjustmentCell dry{290, 90000, .65, 0, 0, 0, 0};
    TripleCloudAdjustmentResult baseline;
    std::string error;
    REQUIRE(produce_exact_triple_cloud_adjustment(dry, .1, baseline, error));
    for (int mutation = 0; mutation < 15; ++mutation)
    {
        auto cell = dry;
        switch (mutation)
        {
            case 0: cell.temperature_k = microphysics_constants::T0; break;
            case 1: cell.temperature_k = 400; break;
            case 2: cell.pressure_pa = 99; break;
            case 3: cell.pressure_pa = 200001; break;
            case 4: cell.dry_density_kg_m3 = 0; break;
            case 5: cell.dry_density_kg_m3 = -1; break;
            case 6: cell.vapor_per_kg = -1; break;
            case 7: cell.cloud_per_kg = -1; break;
            case 8: cell.cloud_number_per_kg = -1; break;
            case 9: cell.cloud_per_kg = .001; break;
            case 10: cell.vertical_velocity_ms = std::numeric_limits<double>::quiet_NaN(); break;
            case 11: cell.vapor_per_kg = 1; break;
            case 12: cell.temperature_k = 273.2; cell.cloud_per_kg = .1; cell.cloud_number_per_kg = 1e5; break;
            case 13: cell.dry_density_kg_m3 = 1e-300; break;
            case 14: cell.temperature_k = std::numeric_limits<double>::infinity(); break;
        }
        auto result = baseline;
        INFO("cloud source mutation " << mutation);
        CHECK_FALSE(produce_exact_triple_cloud_adjustment(cell, .1, result, error));
        CHECK(same_result(result, baseline));
    }
    for (double duration : {0.0, -1.0, std::numeric_limits<double>::infinity()})
    {
        auto result = baseline;
        CHECK_FALSE(produce_exact_triple_cloud_adjustment(dry, duration, result, error));
        CHECK(same_result(result, baseline));
    }
}
