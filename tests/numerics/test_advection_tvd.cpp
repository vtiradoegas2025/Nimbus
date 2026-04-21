/**
 * @file test_advection_tvd.cpp
 * @brief Unit tests for TVD advection scheme factory and basic behavior.
 */
#include "catch2/catch.hpp"
#include "core/field3d.hpp"
#include "core/simulation.hpp"
#include "numerics/advection/advection_base.hpp"

#include <cmath>

namespace
{

void setup_small_grid()
{
    NR = 16;
    NTH = 16;
    NZ = 16;
    dr = 1000.0;
    dz = 500.0;
    dt = 0.5;
    dtheta = 2.0 * 3.14159265358979323846 / NTH;
}

} // namespace

TEST_CASE("TVD scheme creation via factory", "[numerics][advection]")
{
    setup_small_grid();
    auto scheme = create_advection_scheme("tvd");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("WENO5 scheme creation via factory", "[numerics][advection]")
{
    setup_small_grid();
    auto scheme = create_advection_scheme("weno5");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("TVD scheme initializes with config", "[numerics][advection]")
{
    setup_small_grid();
    auto scheme = create_advection_scheme("tvd");

    AdvectionConfig cfg;
    cfg.scheme_id = "tvd";
    cfg.limiter_id = "mc";

    REQUIRE_NOTHROW(scheme->initialize(cfg));
}

TEST_CASE("TVD scheme accepts all limiter types", "[numerics][advection]")
{
    setup_small_grid();

    for (const auto& limiter : {"mc", "vanleer", "superbee", "universal"})
    {
        auto scheme = create_advection_scheme("tvd");
        AdvectionConfig cfg;
        cfg.limiter_id = limiter;
        REQUIRE_NOTHROW(scheme->initialize(cfg));
    }
}
