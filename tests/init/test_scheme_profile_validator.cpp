/**
 * @file test_scheme_profile_validator.cpp
 * @brief Verification of the per-scheme IC profile registry and validator.
 *
 * Coverage:
 *   - get_scheme_profile returns the right profile for each registered
 *     scheme (and throws for unknown schemes).
 *   - Aliases (mesocyclone, axisymmetric_cgrid, cart, ...) resolve to the
 *     same profile as the canonical name.
 *   - validate() catches coordinate / staggering mismatches with errors.
 *   - validate() catches disallowed sounding / hodograph / trigger types.
 *   - validate() emits a warning when the trigger is allowed but isn't
 *     the recommended one (e.g. tornado scheme + warm_bubble).
 *   - validate() emits a warning when a scheme requires shear and the
 *     parametric WK anchors are all zero.
 *   - validate() passes for legitimate combos.
 */

#include "catch2/catch.hpp"
#include "init/scheme_profile.hpp"

namespace
{

tmv::init::ValidationInputs make_default_supercell_cgrid_inputs()
{
    tmv::init::ValidationInputs vi;
    vi.scheme_id = "supercell_cgrid";
    vi.coordinate = CoordinateSystem::Cylindrical;
    vi.stagger = StaggerType::CGrid;
    vi.sounding.type = tmv::init::SoundingSourceConfig::Type::ParametricCAPE;
    vi.hodograph.type = tmv::init::HodographSourceConfig::Type::Auto;
    vi.hodograph.wk_anchors.u_sfc_ms = 4.0;
    vi.hodograph.wk_anchors.u_1km_ms = 14.0;
    vi.hodograph.wk_anchors.u_6km_ms = 28.0;
    vi.trigger.type = tmv::init::TriggerSourceConfig::Type::WarmBubble;
    return vi;
}

}  // namespace

TEST_CASE("get_scheme_profile returns expected profile per scheme",
          "[init][profile]")
{
    using Coord = tmv::init::CoordinateExpect;
    using Stag = tmv::init::StaggerExpect;
    using Trg = tmv::init::TriggerSourceConfig::Type;

    SECTION("cartesian")
    {
        const auto& p = tmv::init::get_scheme_profile("cartesian");
        REQUIRE(p.scheme_id == "cartesian");
        REQUIRE(p.coordinate == Coord::Cartesian);
        REQUIRE(p.stagger == Stag::Collocated);
        REQUIRE(p.recommended_trigger_type == Trg::WarmBubble);
        REQUIRE(p.requires_nonzero_shear == false);
    }

    SECTION("supercell_cgrid")
    {
        const auto& p = tmv::init::get_scheme_profile("supercell_cgrid");
        REQUIRE(p.scheme_id == "supercell_cgrid");
        REQUIRE(p.coordinate == Coord::Cylindrical);
        REQUIRE(p.stagger == Stag::CGrid);
        REQUIRE(p.recommended_trigger_type == Trg::WarmBubble);
        REQUIRE(p.requires_nonzero_shear == true);
    }

    SECTION("tornado_cgrid recommends vortex_seed")
    {
        const auto& p = tmv::init::get_scheme_profile("tornado_cgrid");
        REQUIRE(p.scheme_id == "tornado_cgrid");
        REQUIRE(p.recommended_trigger_type == Trg::VortexSeed);
        REQUIRE(p.requires_nonzero_shear == false);
    }
}

TEST_CASE("Scheme aliases resolve to the canonical profile",
          "[init][profile]")
{
    REQUIRE(tmv::init::scheme_profile_exists("mesocyclone_cgrid"));
    REQUIRE(tmv::init::scheme_profile_exists("axisymmetric"));
    REQUIRE(tmv::init::scheme_profile_exists("cart"));
    REQUIRE(tmv::init::scheme_profile_exists("cartesian_cpu"));

    const auto& meso = tmv::init::get_scheme_profile("mesocyclone_cgrid");
    const auto& super = tmv::init::get_scheme_profile("supercell_cgrid");
    REQUIRE(meso.scheme_id == super.scheme_id);
}

TEST_CASE("get_scheme_profile throws on unknown scheme",
          "[init][profile]")
{
    REQUIRE_THROWS_AS(tmv::init::get_scheme_profile("nonexistent_scheme"),
                      std::out_of_range);
    REQUIRE(tmv::init::scheme_profile_exists("nonexistent_scheme") == false);
}

TEST_CASE("validate: legitimate supercell_cgrid combo passes",
          "[init][profile][validate]")
{
    auto vi = make_default_supercell_cgrid_inputs();
    const auto r = tmv::init::validate_initial_condition_config(vi);
    REQUIRE(r.ok);
    REQUIRE(r.errors.empty());
    // Recommended trigger is warm_bubble, which is what we set; no warning.
    REQUIRE(r.warnings.empty());
}

TEST_CASE("validate: cartesian scheme + cylindrical coords is an error",
          "[init][profile][validate]")
{
    auto vi = make_default_supercell_cgrid_inputs();
    vi.scheme_id = "cartesian";
    // coordinate stays Cylindrical from default
    const auto r = tmv::init::validate_initial_condition_config(vi);
    REQUIRE_FALSE(r.ok);
    REQUIRE_FALSE(r.errors.empty());
}

TEST_CASE("validate: supercell on c_grid with collocated stagger is an error",
          "[init][profile][validate]")
{
    auto vi = make_default_supercell_cgrid_inputs();
    vi.scheme_id = "supercell_cgrid";
    vi.stagger = StaggerType::Collocated;  // wrong for supercell_cgrid
    const auto r = tmv::init::validate_initial_condition_config(vi);
    REQUIRE_FALSE(r.ok);
}

TEST_CASE("validate: supercell schemes don't allow vortex_seed trigger",
          "[init][profile][validate]")
{
    auto vi = make_default_supercell_cgrid_inputs();
    vi.trigger.type = tmv::init::TriggerSourceConfig::Type::VortexSeed;
    const auto r = tmv::init::validate_initial_condition_config(vi);
    REQUIRE_FALSE(r.ok);
}

TEST_CASE("validate: tornado scheme + warm_bubble is allowed but warns",
          "[init][profile][validate]")
{
    auto vi = make_default_supercell_cgrid_inputs();
    vi.scheme_id = "tornado_cgrid";
    vi.coordinate = CoordinateSystem::Cylindrical;
    vi.stagger = StaggerType::CGrid;
    vi.trigger.type = tmv::init::TriggerSourceConfig::Type::WarmBubble;
    const auto r = tmv::init::validate_initial_condition_config(vi);
    REQUIRE(r.ok);
    REQUIRE_FALSE(r.warnings.empty());
}

TEST_CASE("validate: supercell with all-zero WK anchors warns about shear",
          "[init][profile][validate]")
{
    auto vi = make_default_supercell_cgrid_inputs();
    vi.hodograph.wk_anchors = {};  // every anchor = 0
    const auto r = tmv::init::validate_initial_condition_config(vi);
    REQUIRE(r.ok);  // not an error, just a warning
    bool found_shear_warning = false;
    for (const auto& w : r.warnings)
    {
        if (w.find("nonzero environmental shear") != std::string::npos)
        {
            found_shear_warning = true;
        }
    }
    REQUIRE(found_shear_warning);
}

TEST_CASE("validate: zero hodograph + supercell does not warn about shear",
          "[init][profile][validate]")
{
    // Explicit hodograph: zero is the user's intent — they don't want
    // shear. The validator should not warn since the user is explicitly
    // asking for it (e.g. testing pressure-only flow on a supercell mesh).
    auto vi = make_default_supercell_cgrid_inputs();
    vi.hodograph.type = tmv::init::HodographSourceConfig::Type::Zero;
    vi.hodograph.wk_anchors = {};
    const auto r = tmv::init::validate_initial_condition_config(vi);
    bool found_shear_warning = false;
    for (const auto& w : r.warnings)
    {
        if (w.find("nonzero environmental shear") != std::string::npos)
        {
            found_shear_warning = true;
        }
    }
    REQUIRE_FALSE(found_shear_warning);
}

TEST_CASE("validate: unknown scheme is an error",
          "[init][profile][validate]")
{
    auto vi = make_default_supercell_cgrid_inputs();
    vi.scheme_id = "made_up_scheme";
    const auto r = tmv::init::validate_initial_condition_config(vi);
    REQUIRE_FALSE(r.ok);
    REQUIRE_FALSE(r.errors.empty());
}
