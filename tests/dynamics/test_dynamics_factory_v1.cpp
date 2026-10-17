#include "catch2/catch.hpp"
#include "dynamics/dynamics_base.hpp"
#include "dynamics/factory.hpp"
#include "init/config/config_v1.hpp"

#include <algorithm>
#include <stdexcept>

using tmv::init::config::ConfigV1;

namespace
{

ConfigV1 make_config(CoordinateSystem coord, StaggerType stagger,
                     bool axisymmetric, const std::string& scheme)
{
    ConfigV1 c;
    c.grid.coordinate_system = coord;
    c.grid.staggering        = stagger;
    c.dynamics.scheme        = scheme;
    c.dynamics.axisymmetric  = axisymmetric;
    return c;
}

}  // namespace

// ─────────────────────────────────────────────────────────────
// klemp_wilhelmson_split (unqualified) → derive compound from axes
// ─────────────────────────────────────────────────────────────

TEST_CASE("B1b unqualified klemp_wilhelmson_split → cartesian_agrid",
          "[dynamics][factory][b1b]")
{
    auto cfg = make_config(CoordinateSystem::Cartesian, StaggerType::Collocated,
                           /*axisym=*/false, "klemp_wilhelmson_split");
    auto scheme = create_dynamics_scheme(cfg);
    REQUIRE(scheme != nullptr);
    REQUIRE(scheme->get_scheme_name() == "klemp_wilhelmson_split_cartesian_agrid");
}

// ─────────────────────────────────────────────────────────────
// Cartesian + cgrid resolves to the Cartesian C-grid scheme
// ─────────────────────────────────────────────────────────────

TEST_CASE("B1b unqualified → cartesian cgrid",
          "[dynamics][factory][b1b]")
{
    auto cfg = make_config(CoordinateSystem::Cartesian, StaggerType::CGrid,
                           /*axisym=*/false, "klemp_wilhelmson_split");
    auto scheme = create_dynamics_scheme(cfg);
    REQUIRE(scheme != nullptr);
    REQUIRE(scheme->get_scheme_name() == "klemp_wilhelmson_split_cartesian_cgrid");
}

// ─────────────────────────────────────────────────────────────
// An unknown scheme name fails loudly (no silent legacy fallback)
// ─────────────────────────────────────────────────────────────

TEST_CASE("B1b unknown scheme name throws",
          "[dynamics][factory][b1b]")
{
    REQUIRE_THROWS_AS(create_dynamics_scheme("not_a_real_scheme"),
                      std::runtime_error);
}

// ─────────────────────────────────────────────────────────────
// Compound canonical names pass through unchanged
// ─────────────────────────────────────────────────────────────

TEST_CASE("B1b compound canonical names pass through",
          "[dynamics][factory][b1b]")
{
    // The compound name on dynamics.scheme is authoritative even when the
    // grid axes would otherwise suggest a different combination — the
    // user's explicit compound choice wins. Here the collocated staggering
    // would derive the agrid scheme, but the explicit cgrid compound name
    // resolves to the C-grid scheme.
    auto cfg = make_config(CoordinateSystem::Cartesian, StaggerType::Collocated,
                           /*axisym=*/false,
                           "klemp_wilhelmson_split_cartesian_cgrid");
    auto scheme = create_dynamics_scheme(cfg);
    REQUIRE(scheme != nullptr);
    REQUIRE(scheme->get_scheme_name() == "klemp_wilhelmson_split_cartesian_cgrid");
}

// ─────────────────────────────────────────────────────────────
// Legacy v0 aliases also pass through
// ─────────────────────────────────────────────────────────────

TEST_CASE("B1b legacy v0 alias 'cartesian' still resolves",
          "[dynamics][factory][b1b]")
{
    // The 'cartesian' alias is authoritative on dynamics.scheme: it resolves
    // to the A-grid scheme regardless of the staggering axis.
    auto cfg = make_config(CoordinateSystem::Cartesian, StaggerType::CGrid,
                           /*axisym=*/false, "cartesian");
    auto scheme = create_dynamics_scheme(cfg);
    REQUIRE(scheme != nullptr);
    REQUIRE(scheme->get_scheme_name() == "klemp_wilhelmson_split_cartesian_agrid");
}

// ─────────────────────────────────────────────────────────────
// String overload still works (legacy callers unchanged)
// ─────────────────────────────────────────────────────────────

TEST_CASE("B1b string overload still resolves canonical + aliases",
          "[dynamics][factory][b1b]")
{
    auto canonical = create_dynamics_scheme(
        "klemp_wilhelmson_split_cartesian_cgrid");
    REQUIRE(canonical != nullptr);
    REQUIRE(canonical->get_scheme_name() == "klemp_wilhelmson_split_cartesian_cgrid");

    auto alias = create_dynamics_scheme("cart");
    REQUIRE(alias != nullptr);
    REQUIRE(alias->get_scheme_name() == "klemp_wilhelmson_split_cartesian_agrid");
}

TEST_CASE("Dynamics descriptors cover every constructible scheme and alias",
          "[dynamics][factory][descriptor]")
{
    const auto ids = get_available_dynamics_schemes();
    const auto& descriptors = get_dynamics_scheme_descriptors();

    REQUIRE(descriptors.size() == ids.size());
    for (const auto& descriptor : descriptors)
    {
        REQUIRE(descriptor.capabilities.declared);
        REQUIRE(descriptor.capabilities.coordinate_systems
                == std::vector<std::string>{"cartesian"});
        REQUIRE_FALSE(descriptor.capabilities.required_state_fields.empty());
        REQUIRE_FALSE(descriptor.capabilities.produced_state_fields.empty());
        REQUIRE(std::find(ids.begin(), ids.end(), descriptor.id) != ids.end());
        auto canonical = create_dynamics_scheme(descriptor.id);
        REQUIRE(canonical->get_scheme_name() == descriptor.id);
        REQUIRE(canonical->supports_domain_recycling_invalidation());
        auto copied = canonical->clone();
        REQUIRE(copied != nullptr);
        REQUIRE(copied.get() != canonical.get());
        REQUIRE(copied->get_scheme_name() == canonical->get_scheme_name());
        copied->invalidate_generation_state_after_domain_recycling();
        for (const auto& alias : descriptor.aliases)
        {
            auto aliased = create_dynamics_scheme(alias);
            REQUIRE(aliased->get_scheme_name() == descriptor.id);
        }
    }

    const auto* cgrid = tmv::find_scheme_descriptor(
        descriptors, tmv::kCartesianCGridDynamicsId);
    REQUIRE(cgrid != nullptr);
    REQUIRE(cgrid->capabilities.staggerings
            == std::vector<std::string>{"c_grid"});
    REQUIRE(cgrid->capabilities.cpu_gpu_parity_coverage
            == tmv::SchemeCoverage::Covered);
}
