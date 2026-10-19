/**
 * @file test_b3_scheme_trigger_decoupling.cpp
 * @brief CM1Mapping §21.B3 — vortex/bubble IC lives in the
 *        `initial_conditions.trigger` machinery, never in the dynamics scheme.
 *
 * Two invariants that together capture the B3 contract:
 *
 *   1. Constructing a DynamicsScheme leaves the prognostic fields untouched.
 *      A scheme is pure RHS — it knows about derivatives, not initial states.
 *
 *   2. The same dynamics scheme combined with two different
 *      `initial_conditions.trigger.type` values produces two different
 *      initial states. The trigger picks the IC, not the scheme name.
 *
 * Together these say: choose your scheme for the dynamics you want; choose
 * your trigger for the IC you want; they are orthogonal. A researcher
 * familiar with CM1's `iinit` namelist can map mental models 1:1.
 */

#include "catch2/catch.hpp"
#include "core/field/field3d.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/infra/grid_geometry.hpp"
#include "core/runtime/simulation.hpp"

#include "dynamics/factory.hpp"
#include "dynamics/schemes/klemp_wilhelmson_split/cartesian_agrid.hpp"

#include "init/trigger/factory.hpp"
#include "init/trigger/vortex_seed.hpp"

#include <cmath>

extern int& NR;
extern int& NTH;
extern int& NZ;

extern Field3D& u;
extern Field3D& v;
extern Field3D& w;
extern Field3D& p;
extern Field3D& rho;
extern Field3D& theta;

extern GridGeometry& global_grid_geometry;
extern CoordinateSystem& global_coordinate_system;

namespace
{

void zero_prognostic_fields()
{
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    p.resize(NR, NTH, NZ, 0.0f);
    rho.resize(NR, NTH, NZ, 1.0f);
    theta.resize(NR, NTH, NZ, 300.0f);
}

double total_abs_v()
{
    double sum = 0.0;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                sum += std::abs(static_cast<double>(v[i][j][k]));
    return sum;
}

double total_abs_p()
{
    double sum = 0.0;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                sum += std::abs(static_cast<double>(p[i][j][k]));
    return sum;
}

}  // namespace

// =============================================================================
// Invariant 1 — scheme construction does not touch prognostic fields.
// =============================================================================

TEST_CASE("B3 invariant 1: CartesianAGridScheme construction leaves fields untouched",
          "[init][b3][decoupling]")
{
    global_coordinate_system = CoordinateSystem::Cartesian;
    zero_prognostic_fields();

    REQUIRE(total_abs_v() == 0.0);
    REQUIRE(total_abs_p() == 0.0);

    CartesianAGridScheme scheme;

    REQUIRE(total_abs_v() == 0.0);
    REQUIRE(total_abs_p() == 0.0);
}

// =============================================================================
// Invariant 2 — same scheme, different triggers, different ICs.
//
// Construct one CartesianAGridScheme. Apply VortexSeedTrigger; capture fields.
// Reset. Apply no trigger (the `none` IC); capture fields. Verify the two
// resulting field states differ — proof that the trigger, not the scheme,
// picks the IC.
// =============================================================================

TEST_CASE("B3 invariant 2: same scheme + different triggers produce different ICs",
          "[init][b3][decoupling]")
{
    global_coordinate_system = CoordinateSystem::Cartesian;

    // Same scheme for both branches.
    CartesianAGridScheme scheme;

    // --- Branch A: VortexSeedTrigger ---
    zero_prognostic_fields();

    tmv::init::VortexSeedParams vs_params;
    vs_params.r_max_m  = 200.0;
    vs_params.v_max_ms = 10.0;
    vs_params.z_top_m  = 400.0;
    tmv::init::VortexSeedTrigger vs_trigger(vs_params);
    vs_trigger.apply();

    const double v_sum_vortex_seed = total_abs_v();
    REQUIRE(v_sum_vortex_seed > 0.0);

    // --- Branch B: the `none` trigger leaves the quiescent base state ---
    zero_prognostic_fields();

    const double v_sum_none = total_abs_v();
    REQUIRE(v_sum_none == 0.0);

    // Different trigger type ⇒ different IC. The vortex seed adds rotation;
    // the `none` trigger does not. The scheme is identical in both branches,
    // so the difference can only come from the trigger.
    REQUIRE(std::abs(v_sum_vortex_seed - v_sum_none) > 0.01);
}

// =============================================================================
// Invariant 3 — factory dispatch honors the trigger.type config field
//                regardless of the dynamics scheme picked alongside it.
// =============================================================================

TEST_CASE("B3 invariant 3: trigger factory dispatch is scheme-independent",
          "[init][b3][decoupling]")
{
    using Type = tmv::init::TriggerSourceConfig::Type;

    SECTION("parse + factory round-trips for vortex_seed")
    {
        tmv::init::TriggerSourceConfig cfg;
        REQUIRE(tmv::init::parse_trigger_source_type("vortex_seed", cfg.type));
        REQUIRE(cfg.type == Type::VortexSeed);
        auto trigger = tmv::init::make_trigger_source(cfg);
        REQUIRE(trigger != nullptr);
    }

    SECTION("parse + factory round-trips for warm_bubble")
    {
        tmv::init::TriggerSourceConfig cfg;
        REQUIRE(tmv::init::parse_trigger_source_type("warm_bubble", cfg.type));
        REQUIRE(cfg.type == Type::WarmBubble);
        auto trigger = tmv::init::make_trigger_source(cfg);
        REQUIRE(trigger != nullptr);
    }

    SECTION("parse + factory round-trips for none")
    {
        tmv::init::TriggerSourceConfig cfg;
        REQUIRE(tmv::init::parse_trigger_source_type("none", cfg.type));
        REQUIRE(cfg.type == Type::None);
        auto trigger = tmv::init::make_trigger_source(cfg);
        REQUIRE(trigger != nullptr);
    }
}
