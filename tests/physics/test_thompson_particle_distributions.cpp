#include "catch2/catch.hpp"
#include "microphysics/schemes/thompson/particle_distributions.hpp"

#include <limits>

namespace
{

void require_close(double actual, double expected)
{
    REQUIRE(actual == Approx(expected).epsilon(2.0e-12));
}

} // namespace

TEST_CASE("Thompson cloud rain and ice size sorting matches WRF v4.7.1",
          "[microphysics][thompson][distribution][literal]")
{
    const auto cloud = tmv::thompson::cloud_fall_speeds(1.0e-4, 1.0e8, 1.0);
    require_close(cloud.mass_m_s, 0.00739408998412188);
    require_close(cloud.number_m_s, 0.004947516092316845);
    REQUIRE(cloud.mass_m_s > cloud.number_m_s);

    const auto rain = tmv::thompson::rain_fall_speeds(1.0e-3, 1.0e5, 1.0);
    require_close(rain.mass_m_s, 2.698586404752422);
    require_close(rain.number_m_s, 1.759704757984367);
    REQUIRE(rain.mass_m_s > rain.number_m_s);

    const auto ice = tmv::thompson::ice_fall_speeds(2.0e-4, 2.0e5, 0.8);
    require_close(ice.mass_m_s, 0.5161326075397535);
    require_close(ice.number_m_s, 0.32258287971234595);
    REQUIRE(ice.mass_m_s > ice.number_m_s);
}

TEST_CASE("Thompson Field snow and fixed-density graupel distributions are literal",
          "[microphysics][thompson][distribution][literal]")
{
    const auto snow = tmv::thompson::snow_distribution(5.0e-4, 258.15, 0.9);
    require_close(snow.second_moment, 0.007246376811594202);
    require_close(snow.third_moment, 7.93918522258473e-6);
    require_close(snow.number_m3, 156020.48889377533);
    require_close(snow.mass_fall_speed_m_s, 0.8951572865818471);

    const auto graupel = tmv::thompson::graupel_fall_speeds(3.0e-4, 0.9);
    require_close(graupel.mass_m_s, 5.65618030679104);
    require_close(graupel.number_m_s, 1.597221867628539);
}

TEST_CASE("Thompson particle distributions fail closed",
          "[microphysics][thompson][distribution][invalid]")
{
    REQUIRE_THROWS_AS(tmv::thompson::rain_fall_speeds(
                          1.0e-3, -1.0, 1.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(tmv::thompson::snow_distribution(
                          1.0e-3, std::numeric_limits<double>::quiet_NaN(), 1.0),
                      std::invalid_argument);
    const auto clear = tmv::thompson::ice_fall_speeds(0.0, 0.0, 1.0);
    REQUIRE(clear.mass_m_s == 0.0);
    REQUIRE(clear.number_m_s == 0.0);
}
