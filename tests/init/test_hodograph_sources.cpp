/**
 * @file test_hodograph_sources.cpp
 * @brief Verifies HodographSource factory dispatch and concrete sources.
 *
 * Coverage:
 *   - WKParamHodograph reproduces compute_wind_profile() for the legacy
 *     surface / 1 km / 6 km anchor pattern.
 *   - ZeroHodograph returns u = v = 0 at every level.
 *   - Factory parses recognized type strings and dispatches to the right
 *     concrete source. Unknown names fail parse_hodograph_source_type
 *     so the runtime can warn cleanly.
 */

#include "catch2/catch.hpp"
#include "init/hodograph/factory.hpp"
#include "init/hodograph/wk_param.hpp"
#include "init/hodograph/zero.hpp"

#include <vector>

namespace
{

std::vector<double> heights(const std::vector<double>& zs)
{
    return zs;
}

}  // namespace

TEST_CASE("WKParamHodograph reproduces 3-point linear interpolation",
          "[init][hodograph][wk_param]")
{
    tmv::init::WKParamHodographAnchors a;
    a.u_sfc_ms = 4.0;   a.v_sfc_ms = 1.0;
    a.u_1km_ms = 14.0;  a.v_1km_ms = 6.0;
    a.u_6km_ms = 28.0;  a.v_6km_ms = 22.0;

    tmv::init::WKParamHodograph src(a);
    const auto z = heights({0.0, 500.0, 1000.0, 3500.0, 6000.0, 10000.0});
    const auto wind = src.build(z);

    REQUIRE(wind.is_consistent_with(z));

    SECTION("surface anchor exact")
    {
        REQUIRE(wind.u_ms[0] == Approx(4.0).margin(1.0e-12));
        REQUIRE(wind.v_ms[0] == Approx(1.0).margin(1.0e-12));
    }

    SECTION("midway sfc<->1km is linear average")
    {
        // z=500 m -> halfway between sfc and 1 km
        REQUIRE(wind.u_ms[1] == Approx(0.5 * (4.0 + 14.0)).margin(1.0e-12));
        REQUIRE(wind.v_ms[1] == Approx(0.5 * (1.0 + 6.0)).margin(1.0e-12));
    }

    SECTION("1 km anchor exact")
    {
        REQUIRE(wind.u_ms[2] == Approx(14.0).margin(1.0e-12));
        REQUIRE(wind.v_ms[2] == Approx(6.0).margin(1.0e-12));
    }

    SECTION("midway 1km<->6km is linear average")
    {
        // z=3500 m -> halfway between 1 km and 6 km
        REQUIRE(wind.u_ms[3] == Approx(0.5 * (14.0 + 28.0)).margin(1.0e-12));
        REQUIRE(wind.v_ms[3] == Approx(0.5 * (6.0 + 22.0)).margin(1.0e-12));
    }

    SECTION("6 km anchor exact")
    {
        REQUIRE(wind.u_ms[4] == Approx(28.0).margin(1.0e-12));
        REQUIRE(wind.v_ms[4] == Approx(22.0).margin(1.0e-12));
    }

    SECTION("above 6 km, constant at 6 km value")
    {
        REQUIRE(wind.u_ms[5] == Approx(28.0).margin(1.0e-12));
        REQUIRE(wind.v_ms[5] == Approx(22.0).margin(1.0e-12));
    }
}

TEST_CASE("WKParamHodograph describe() returns canonical name",
          "[init][hodograph][wk_param]")
{
    tmv::init::WKParamHodograph src({});
    REQUIRE(src.describe() == "wk_param");
}

TEST_CASE("WKParamHodograph rejects empty z",
          "[init][hodograph][wk_param]")
{
    tmv::init::WKParamHodograph src({});
    REQUIRE_THROWS_AS(src.build({}), std::invalid_argument);
}

TEST_CASE("ZeroHodograph returns zeros at every level",
          "[init][hodograph][zero]")
{
    tmv::init::ZeroHodograph src;
    const auto z = heights({0.0, 100.0, 1000.0, 5000.0, 12000.0});
    const auto wind = src.build(z);

    REQUIRE(wind.is_consistent_with(z));
    for (std::size_t k = 0; k < z.size(); ++k)
    {
        REQUIRE(wind.u_ms[k] == 0.0);
        REQUIRE(wind.v_ms[k] == 0.0);
    }
    REQUIRE(src.describe() == "zero");
}

TEST_CASE("parse_hodograph_source_type recognizes valid names",
          "[init][hodograph][factory]")
{
    using T = tmv::init::HodographSourceConfig::Type;
    T t{};

    REQUIRE(tmv::init::parse_hodograph_source_type("auto", t));
    REQUIRE(t == T::Auto);

    REQUIRE(tmv::init::parse_hodograph_source_type("wk_param", t));
    REQUIRE(t == T::WKParam);

    REQUIRE(tmv::init::parse_hodograph_source_type("WK-PARAM", t));
    REQUIRE(t == T::WKParam);

    REQUIRE(tmv::init::parse_hodograph_source_type("zero", t));
    REQUIRE(t == T::Zero);

    REQUIRE(tmv::init::parse_hodograph_source_type("calm", t));
    REQUIRE(t == T::Zero);

    REQUIRE_FALSE(tmv::init::parse_hodograph_source_type("rubbish", t));
}

TEST_CASE("hodograph_source_type_name returns canonical strings",
          "[init][hodograph][factory]")
{
    using T = tmv::init::HodographSourceConfig::Type;
    REQUIRE(tmv::init::hodograph_source_type_name(T::Auto) == "auto");
    REQUIRE(tmv::init::hodograph_source_type_name(T::WKParam) == "wk_param");
    REQUIRE(tmv::init::hodograph_source_type_name(T::Zero) == "zero");
}

TEST_CASE("make_hodograph_source dispatches by type",
          "[init][hodograph][factory]")
{
    SECTION("Auto and WKParam both produce a wk_param source")
    {
        tmv::init::HodographSourceConfig cfg;
        cfg.type = tmv::init::HodographSourceConfig::Type::Auto;
        cfg.wk_anchors.u_sfc_ms = 1.0;  // arbitrary
        auto src = tmv::init::make_hodograph_source(cfg);
        REQUIRE(src);
        REQUIRE(src->describe() == "wk_param");

        cfg.type = tmv::init::HodographSourceConfig::Type::WKParam;
        src = tmv::init::make_hodograph_source(cfg);
        REQUIRE(src->describe() == "wk_param");
    }

    SECTION("Zero produces a zero source")
    {
        tmv::init::HodographSourceConfig cfg;
        cfg.type = tmv::init::HodographSourceConfig::Type::Zero;
        auto src = tmv::init::make_hodograph_source(cfg);
        REQUIRE(src);
        REQUIRE(src->describe() == "zero");
    }
}
