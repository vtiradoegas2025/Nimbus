#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_homogeneous_cloud_freezing.hpp"
#include "support/milbrandt_homogeneous_cloud_freezing_reference.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_homogeneous_cloud_freezing_reference;

ColdParameters parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

HomogeneousCloudFreezingCell source(double tc = -37, double q = 1e-3, double n = 1e8)
{
    return {273.15 + tc, 85000, .65, q, n};
}

std::array<double, 7> values(const HomogeneousCloudFreezingChannel& channel)
{
    return {channel.fraction, channel.mean_volume_cm3, channel.log10_rate_per_cm3_s,
        channel.source_mass_squared_per_kg, channel.raw_mass_increment_per_kg,
        channel.raw_number_increment_per_kg, channel.raw_reflectivity_increment_per_kg};
}

void same(const HomogeneousCloudFreezingProducer& actual, const HomogeneousCloudFreezingProducer& expected)
{
    REQUIRE(actual.event_count() == expected.event_count());
    REQUIRE(actual.parameters() == expected.parameters());
    REQUIRE(actual.hail_processes() == expected.hail_processes());
    REQUIRE(actual.duration_s() == expected.duration_s());
    const auto& a = actual.source();
    const auto& b = expected.source();
    REQUIRE(a.temperature_k == b.temperature_k);
    REQUIRE(a.pressure_pa == b.pressure_pa);
    REQUIRE(a.dry_density_kg_m3 == b.dry_density_kg_m3);
    REQUIRE(a.cloud_mass_per_kg == b.cloud_mass_per_kg);
    REQUIRE(a.cloud_number_per_kg == b.cloud_number_per_kg);
    const auto av = values(actual.channel());
    const auto bv = values(expected.channel());
    REQUIRE(std::memcmp(av.data(), bv.data(), sizeof(av)) == 0);
}

void near(double actual, long double expected, long double scale = 1e-300L)
{
    INFO("actual " << actual << ", independent " << static_cast<double>(expected));
    REQUIRE(std::isfinite(actual));
    CHECK(std::abs(static_cast<long double>(actual) - expected)
        <= 4e-10L * std::max(std::abs(expected), scale));
}

void check_reference(const HomogeneousCloudFreezingCell& cell, double duration,
                     const HomogeneousCloudFreezingProducer& producer)
{
    const auto expected = reference::evaluate(cell, duration);
    const auto& channel = producer.channel();
    near(channel.fraction, expected.fraction);
    near(channel.mean_volume_cm3, expected.mean_volume_cm3);
    near(channel.log10_rate_per_cm3_s, expected.log10_rate, 1);
    near(channel.source_mass_squared_per_kg, expected.source_mass_squared_per_kg);
    near(channel.raw_mass_increment_per_kg, expected.mass_increment);
    near(channel.raw_number_increment_per_kg, expected.number_increment);
    near(channel.raw_reflectivity_increment_per_kg, expected.z_increment);
    CHECK(channel.raw_mass_increment_per_kg <= cell.cloud_mass_per_kg);
    CHECK(channel.raw_number_increment_per_kg <= cell.cloud_number_per_kg);
    if (channel.fraction > 0)
    {
        near(channel.raw_mass_increment_per_kg / cell.cloud_mass_per_kg, channel.fraction);
        near(channel.raw_number_increment_per_kg / cell.cloud_number_per_kg, channel.fraction);
        near(channel.raw_reflectivity_increment_per_kg * 440 * 440,
            channel.fraction * channel.source_mass_squared_per_kg);
    }
}

HomogeneousCloudFreezingProducer produce(const HomogeneousCloudFreezingCell& cell,
                                        double duration, bool hail = true)
{
    HomogeneousCloudFreezingProducer producer;
    std::string error;
    const bool accepted = produce_exact_homogeneous_cloud_freezing_admission(parameters(hail), cell,
        duration, producer, error);
    INFO(error);
    REQUIRE(accepted);
    REQUIRE(producer.event_count() == 1);
    return producer;
}

} // namespace

TEST_CASE("homogeneous cloud freezing follows independent volume probability and conserved source M2",
          "[physics][milbrandt][triple_homogeneous_cloud_freezing]")
{
    for (double tc : {-30.0, -32.0, -35.0, -37.0, -40.0, -45.0, -50.0})
    {
        for (double q : {1e-8, 1e-4, 5e-3})
        {
            for (double n : {1e4, 1e8})
            {
                for (double duration : {1e-7, .01, 10.0})
                {
                    for (double density : {.5, 1.3})
                    {
                        for (bool hail : {false, true})
                        {
                            auto cell = source(tc, q, n);
                            cell.dry_density_kg_m3 = density;
                            const auto producer = produce(cell, duration, hail);
                            check_reference(cell, duration, producer);
                            CHECK(producer.hail_processes() == hail);
                            CHECK(producer.duration_s() == duration);
                            CHECK(producer.parameters() == default_cold_parameters);
                            std::string error;
                            REQUIRE(validate_exact_homogeneous_cloud_freezing_admission(parameters(hail), cell,
                                duration, producer, error));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("homogeneous cloud mass squared agrees with generalized spectrum quadrature and cubic ice units",
          "[physics][milbrandt][triple_homogeneous_cloud_freezing]")
{
    for (double q : {1e-6, 1e-3})
    {
        for (double n : {1e4, 1e8})
        {
            const auto cell = source(-51, q, n);
            const auto producer = produce(cell, .1);
            const auto& channel = producer.channel();
            near(channel.source_mass_squared_per_kg, reference::cloud_mass_squared_quadrature(q, n));
            near(channel.mean_volume_cm3, (std::acos(-1.0L) / 6 * 1e6L / 523.598775598L) * q / n);
            REQUIRE(channel.fraction == 1);
            CHECK(channel.raw_mass_increment_per_kg == q);
            CHECK(channel.raw_number_increment_per_kg == n);
            near(channel.raw_reflectivity_increment_per_kg,
                reference::cloud_mass_squared_quadrature(q, n) / (440 * 440));
            const long double mass_squared_lost = channel.fraction * channel.source_mass_squared_per_kg;
            near(channel.raw_reflectivity_increment_per_kg * 440 * 440, mass_squared_lost);
        }
    }
}

TEST_CASE("homogeneous freezing retains exact cloud and temperature cutoffs and empty-source contracts",
          "[physics][milbrandt][triple_homogeneous_cloud_freezing]")
{
    for (double q : {0.0, 1e-14, std::nextafter(1e-14, 0.0)})
    {
        const auto cell = source(-51, q, q == 0 ? 0 : 1e8);
        const auto producer = produce(cell, .1);
        CHECK(producer.channel().fraction == 0);
        CHECK(producer.channel().raw_mass_increment_per_kg == 0);
        CHECK(producer.channel().raw_number_increment_per_kg == 0);
        CHECK(producer.channel().raw_reflectivity_increment_per_kg == 0);
        check_reference(cell, .1, producer);
    }
    const auto active = source(-51, std::nextafter(1e-14, std::numeric_limits<double>::infinity()));
    CHECK(produce(active, .1).channel().fraction == 1);
    const auto empty_with_number = source(-51, 0, 1e8);
    CHECK(produce(empty_with_number, .1).channel().source_mass_squared_per_kg == 0);
    auto boundary = source(-30);
    REQUIRE(produce(boundary, .1).channel().fraction > 0);
    boundary.temperature_k = std::nextafter(boundary.temperature_k, std::numeric_limits<double>::infinity());
    CHECK(produce(boundary, .1).channel().fraction == 0);
    boundary = source(-50);
    const double fraction = produce(boundary, 1e-20).channel().fraction;
    REQUIRE(fraction > 0);
    REQUIRE(fraction < 1);
    boundary.temperature_k = std::nextafter(boundary.temperature_k, 0.0);
    CHECK(produce(boundary, 1e-20).channel().fraction == 1);
}

TEST_CASE("homogeneous freezing preserves tiny probabilities and saturates stiff intervals without overflow",
          "[physics][milbrandt][triple_homogeneous_cloud_freezing]")
{
    const auto weak = source(-30);
    const auto producer = produce(weak, 1e-10);
    const auto expected = reference::evaluate(weak, 1e-10);
    REQUIRE(expected.hazard > 0);
    REQUIRE(1.0 - std::exp(-static_cast<double>(expected.hazard)) == 0);
    REQUIRE(producer.channel().fraction > 0);
    REQUIRE(producer.channel().raw_mass_increment_per_kg > 0);
    REQUIRE(producer.channel().raw_reflectivity_increment_per_kg > 0);
    check_reference(weak, 1e-10, producer);
    const auto stiff = produce(source(-40), std::numeric_limits<double>::max());
    REQUIRE(stiff.channel().fraction == 1);
    REQUIRE(stiff.channel().raw_mass_increment_per_kg == stiff.source().cloud_mass_per_kg);
    auto large = source(-51, 1e200, 1e200);
    const auto scaled = produce(large, .1);
    near(scaled.channel().source_mass_squared_per_kg, 1.5e200L);
    REQUIRE(std::isfinite(scaled.channel().raw_reflectivity_increment_per_kg));
    auto environment = weak;
    environment.pressure_pa = 100;
    environment.dry_density_kg_m3 = std::numeric_limits<double>::denorm_min();
    const auto thin = produce(environment, 1e-10);
    CHECK(values(thin.channel()) == values(producer.channel()));
    environment.pressure_pa = 200000;
    environment.dry_density_kg_m3 = 1000;
    CHECK(values(produce(environment, 1e-10).channel()) == values(producer.channel()));
}

TEST_CASE("homogeneous freezing rejects malformed inactive contexts resources and numerical underflow atomically",
          "[physics][milbrandt][triple_homogeneous_cloud_freezing]")
{
    const auto valid = produce(source(), .1);
    for (bool inactive : {false, true})
    {
        for (int mutation = 0; mutation < 17; ++mutation)
        {
            auto cell = inactive ? source(-20, 0, 0) : source();
            double duration = .1;
            auto p = parameters();
            switch (mutation)
            {
                case 0: cell.temperature_k = std::numeric_limits<double>::quiet_NaN(); break;
                case 1: cell.temperature_k = 149; break;
                case 2: cell.temperature_k = 273.15; break;
                case 3: cell.pressure_pa = 99; break;
                case 4: cell.pressure_pa = 200001; break;
                case 5: cell.pressure_pa = std::numeric_limits<double>::infinity(); break;
                case 6: cell.dry_density_kg_m3 = 0; break;
                case 7: cell.dry_density_kg_m3 = -1; break;
                case 8: cell.dry_density_kg_m3 = std::numeric_limits<double>::quiet_NaN(); break;
                case 9: cell.cloud_mass_per_kg = -1; break;
                case 10: cell.cloud_mass_per_kg = std::numeric_limits<double>::quiet_NaN(); break;
                case 11: cell.cloud_number_per_kg = -1; break;
                case 12: cell.cloud_number_per_kg = std::numeric_limits<double>::infinity(); break;
                case 13: cell.cloud_mass_per_kg = 1e-3; cell.cloud_number_per_kg = 0; break;
                case 14: duration = 0; break;
                case 15: duration = std::numeric_limits<double>::quiet_NaN(); break;
                case 16: p.c_i_ = 441; break;
            }
            auto producer = valid;
            std::string error;
            HomogeneousCloudFreezingFailure failure;
            CHECK_FALSE(produce_exact_homogeneous_cloud_freezing_admission(p, cell, duration, producer, error, 1, &failure));
            CHECK(failure == ((mutation >= 9 && mutation <= 13)
                ? HomogeneousCloudFreezingFailure::InvalidSource : HomogeneousCloudFreezingFailure::InvalidContext));
            CHECK_FALSE(error.empty());
            same(producer, valid);
        }
    }
    for (double duration : {std::numeric_limits<double>::denorm_min(), 1e-303})
    {
        auto producer = valid;
        std::string error;
        HomogeneousCloudFreezingFailure failure;
        CHECK_FALSE(produce_exact_homogeneous_cloud_freezing_admission(parameters(), source(-30), duration,
            producer, error, 1, &failure));
        CHECK(failure == HomogeneousCloudFreezingFailure::IntervalRejected);
        same(producer, valid);
    }
    for (auto cell : {source(-51, std::numeric_limits<double>::max(), std::numeric_limits<double>::max()),
                     source(-20, std::numeric_limits<double>::denorm_min(), 1e308)})
    {
        auto producer = valid;
        std::string error;
        HomogeneousCloudFreezingFailure failure;
        CHECK_FALSE(produce_exact_homogeneous_cloud_freezing_admission(parameters(), cell, .1,
            producer, error, 1, &failure));
        CHECK(failure == HomogeneousCloudFreezingFailure::InvalidSource);
        same(producer, valid);
    }
    for (std::size_t capacity : {std::size_t{0}, std::size_t{2}, std::numeric_limits<std::size_t>::max()})
    {
        auto producer = valid;
        std::string error;
        HomogeneousCloudFreezingFailure failure;
        CHECK_FALSE(produce_exact_homogeneous_cloud_freezing_admission(parameters(), source(), .1,
            producer, error, capacity, &failure));
        CHECK(failure == HomogeneousCloudFreezingFailure::Capacity);
        same(producer, valid);
    }
    for (std::size_t mutation = 0; mutation < default_cold_parameters.size(); ++mutation)
    {
        auto p = default_cold_parameters;
        p[mutation] = std::nextafter(p[mutation], std::numeric_limits<double>::infinity());
        const ColdParameters changed{p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
            p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
            p[20], p[21], p[22], p[23], p[24], true};
        auto producer = valid;
        std::string error;
        HomogeneousCloudFreezingFailure failure;
        CHECK_FALSE(produce_exact_homogeneous_cloud_freezing_admission(changed, source(-20, 0, 0), .1,
            producer, error, 1, &failure));
        CHECK(failure == HomogeneousCloudFreezingFailure::InvalidContext);
        same(producer, valid);
    }
    same(produce(source(), .1), valid);
}

TEST_CASE("homogeneous freezing captures every source parameter policy and transfer before replay",
          "[physics][milbrandt][triple_homogeneous_cloud_freezing]")
{
    for (bool inactive : {false, true})
    {
        const auto cell = inactive ? source(-20, 0, 0) : source();
        const auto producer = produce(cell, .1);
        for (int mutation = 0; mutation < 32; ++mutation)
        {
            auto changed = cell;
            double duration = .1;
            auto captured = default_cold_parameters;
            bool hail = true;
            std::array<double*, 5> fields{&changed.temperature_k, &changed.pressure_pa,
                &changed.dry_density_kg_m3, &changed.cloud_mass_per_kg, &changed.cloud_number_per_kg};
            if (mutation < 5)
            {
                *fields[mutation] = std::nextafter(*fields[mutation], std::numeric_limits<double>::infinity());
            }
            else if (mutation < 30)
            {
                captured[mutation - 5] = std::nextafter(captured[mutation - 5], std::numeric_limits<double>::infinity());
            }
            else if (mutation == 30)
            {
                duration = std::nextafter(duration, std::numeric_limits<double>::infinity());
            }
            else
            {
                hail = false;
            }
            const auto& p = captured;
            const ColdParameters context{p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
                p[20], p[21], p[22], p[23], p[24], hail};
            std::string error;
            HomogeneousCloudFreezingFailure failure;
            CHECK_FALSE(validate_exact_homogeneous_cloud_freezing_admission(context, changed, duration,
                producer, error, 1, &failure));
            CHECK(failure == HomogeneousCloudFreezingFailure::InvalidProducer);
            same(producer, produce(cell, .1));
        }
        for (std::size_t mutation = 0; mutation < 7; ++mutation)
        {
            auto forged = producer;
            auto& channel = const_cast<HomogeneousCloudFreezingChannel&>(forged.channel());
            std::array<double*, 7> fields{&channel.fraction, &channel.mean_volume_cm3,
                &channel.log10_rate_per_cm3_s, &channel.source_mass_squared_per_kg,
                &channel.raw_mass_increment_per_kg, &channel.raw_number_increment_per_kg,
                &channel.raw_reflectivity_increment_per_kg};
            *fields[mutation] = std::nextafter(*fields[mutation], std::numeric_limits<double>::infinity());
            std::string error;
            HomogeneousCloudFreezingFailure failure;
            CHECK_FALSE(validate_exact_homogeneous_cloud_freezing_admission(parameters(), cell, .1,
                forged, error, 1, &failure));
            CHECK(failure == HomogeneousCloudFreezingFailure::InvalidProducer);
        }
        std::string error;
        const HomogeneousCloudFreezingProducer absent;
        CHECK_FALSE(validate_exact_homogeneous_cloud_freezing_admission(parameters(), cell, .1, absent, error));
        for (std::size_t capacity : {std::size_t{0}, std::size_t{2}})
        {
            HomogeneousCloudFreezingFailure failure;
            CHECK_FALSE(validate_exact_homogeneous_cloud_freezing_admission(parameters(), cell, .1,
                producer, error, capacity, &failure));
            CHECK(failure == HomogeneousCloudFreezingFailure::Capacity);
        }
        REQUIRE(validate_exact_homogeneous_cloud_freezing_admission(parameters(), cell, .1, producer, error));
    }
    const auto cell = source(-20, 0, 0);
    const auto producer = produce(cell, .1);
    auto changed = cell;
    changed.cloud_mass_per_kg = -0.0;
    std::string error;
    CHECK_FALSE(validate_exact_homogeneous_cloud_freezing_admission(parameters(), changed, .1, producer, error));
}
