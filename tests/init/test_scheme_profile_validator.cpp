#include "catch2/catch.hpp"
#include "dynamics/scheme_descriptor.hpp"
#include "init/hodograph/factory.hpp"
#include "init/scheme_profile.hpp"
#include "init/sounding/factory.hpp"
#include "init/trigger/factory.hpp"
#include "turbulence/scheme_descriptor.hpp"

#include <algorithm>
#include <array>

namespace
{

tmv::init::ValidationInputs make_default_cartesian_cgrid_inputs()
{
    tmv::init::ValidationInputs vi;
    vi.scheme_id = "klemp_wilhelmson_split_cartesian_cgrid";
    vi.coordinate = CoordinateSystem::Cartesian;
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

    // Legacy scheme names (`cartesian`, `cart`, `cartesian_cpu`) resolve to v1
    // canonical scheme_ids via the Phase B1a alias map. The profile structure
    // preserves the canonical name so downstream code can route on a stable
    // identifier regardless of which alias the user typed.

    SECTION("cartesian (A-grid)")
    {
        const auto& p = tmv::init::get_scheme_profile("cartesian");
        REQUIRE(p.scheme_id == "klemp_wilhelmson_split_cartesian_agrid");
        REQUIRE(p.coordinate == Coord::Cartesian);
        REQUIRE(p.stagger == Stag::Collocated);
        REQUIRE(p.recommended_trigger_type == Trg::WarmBubble);
        REQUIRE(p.requires_nonzero_shear == false);
    }

    SECTION("cartesian cgrid (C-grid)")
    {
        const auto& p =
            tmv::init::get_scheme_profile("klemp_wilhelmson_split_cartesian_cgrid");
        REQUIRE(p.scheme_id == "klemp_wilhelmson_split_cartesian_cgrid");
        REQUIRE(p.coordinate == Coord::Cartesian);
        REQUIRE(p.stagger == Stag::CGrid);
        REQUIRE(p.recommended_trigger_type == Trg::WarmBubble);
        REQUIRE(p.requires_nonzero_shear == false);
    }
}

TEST_CASE("Scheme aliases resolve to the canonical profile",
          "[init][profile]")
{
    REQUIRE(tmv::init::scheme_profile_exists("cartesian"));
    REQUIRE(tmv::init::scheme_profile_exists("cart"));
    REQUIRE(tmv::init::scheme_profile_exists("cartesian_cpu"));

    const auto& cart = tmv::init::get_scheme_profile("cart");
    const auto& cpu = tmv::init::get_scheme_profile("cartesian_cpu");
    const auto& canonical = tmv::init::get_scheme_profile("cartesian");
    REQUIRE(cart.scheme_id == canonical.scheme_id);
    REQUIRE(cpu.scheme_id == canonical.scheme_id);
    REQUIRE(canonical.scheme_id == "klemp_wilhelmson_split_cartesian_agrid");
}

TEST_CASE("Every dynamics descriptor has an initialization profile",
          "[init][profile][descriptor]")
{
    const auto& descriptors = tmv::get_dynamics_scheme_descriptors();
    const auto profile_ids = tmv::init::get_registered_scheme_profile_ids();
    REQUIRE(profile_ids.size() == descriptors.size());

    for (const auto& descriptor : descriptors)
    {
        REQUIRE(tmv::init::scheme_profile_exists(descriptor.id));
        const auto& profile = tmv::init::get_scheme_profile(descriptor.id);
        REQUIRE(profile.scheme_id == descriptor.id);

        std::vector<std::string> sounding_ids;
        for (const auto type : profile.allowed_sounding_types)
            sounding_ids.push_back(tmv::init::sounding_source_type_name(type));
        REQUIRE(sounding_ids == descriptor.capabilities.accepted_sounding_sources);

        std::vector<std::string> hodograph_ids;
        for (const auto type : profile.allowed_hodograph_types)
            hodograph_ids.push_back(tmv::init::hodograph_source_type_name(type));
        REQUIRE(hodograph_ids == descriptor.capabilities.accepted_hodograph_sources);

        std::vector<std::string> trigger_ids;
        for (const auto type : profile.allowed_trigger_types)
            trigger_ids.push_back(tmv::init::trigger_source_type_name(type));
        REQUIRE(trigger_ids == descriptor.capabilities.accepted_trigger_sources);
        REQUIRE(tmv::init::trigger_source_type_name(profile.recommended_trigger_type)
                == descriptor.capabilities.recommended_trigger_source);
        for (const auto& alias : descriptor.aliases)
        {
            REQUIRE(tmv::init::scheme_profile_exists(alias));
            REQUIRE(tmv::init::get_scheme_profile(alias).scheme_id == descriptor.id);
        }
    }
}

TEST_CASE("get_scheme_profile throws on unknown scheme",
          "[init][profile]")
{
    REQUIRE_THROWS_AS(tmv::init::get_scheme_profile("nonexistent_scheme"),
                      std::out_of_range);
    REQUIRE(tmv::init::scheme_profile_exists("nonexistent_scheme") == false);
}

TEST_CASE("validate: legitimate cartesian_cgrid combo passes",
          "[init][profile][validate]")
{
    auto vi = make_default_cartesian_cgrid_inputs();
    const auto r = tmv::init::validate_initial_condition_config(vi);
    REQUIRE(r.ok);
    REQUIRE(r.errors.empty());
    // Recommended trigger is warm_bubble, which is what we set; no warning.
    REQUIRE(r.warnings.empty());
}

TEST_CASE("SCI-026: parametric targets are allowed and constructible",
          "[init][profile][validate][sounding][sci026]")
{
    auto vi = make_default_cartesian_cgrid_inputs();
    vi.sounding.type = tmv::init::SoundingSourceConfig::Type::ParametricTargets;

    const auto validation = tmv::init::validate_initial_condition_config(vi);
    REQUIRE(validation.ok);
    REQUIRE(validation.errors.empty());

    auto source = tmv::init::make_sounding_source(vi.sounding);
    REQUIRE(source != nullptr);
    REQUIRE(source->describe() == "parametric_targets");
}

TEST_CASE("validate: cartesian_cgrid with collocated stagger is an error",
          "[init][profile][validate]")
{
    auto vi = make_default_cartesian_cgrid_inputs();
    vi.stagger = StaggerType::Collocated;  // wrong for the C-grid profile
    const auto r = tmv::init::validate_initial_condition_config(vi);
    REQUIRE_FALSE(r.ok);
    REQUIRE_FALSE(r.errors.empty());
}

TEST_CASE("validate: active SGS follows its declared collocated capability",
          "[init][profile][validate][turbulence][staggering]")
{
    auto vi = make_default_cartesian_cgrid_inputs();
    vi.scheme_id = "klemp_wilhelmson_split_cartesian_agrid";
    vi.stagger = StaggerType::Collocated;
    vi.turbulence_scheme_id = "smagorinsky";
    const auto result = tmv::init::validate_initial_condition_config(vi);
    REQUIRE(result.ok);
    REQUIRE(result.errors.empty());
}

TEST_CASE("validate: cartesian schemes allow vortex_seed trigger (not rejected)",
          "[init][profile][validate]")
{
    // The cartesian profiles list VortexSeed in allowed_trigger_types, so the
    // validator accepts it rather than rejecting it (it is allowed but not the
    // recommended trigger).
    auto vi = make_default_cartesian_cgrid_inputs();
    vi.trigger.type = tmv::init::TriggerSourceConfig::Type::VortexSeed;
    const auto r = tmv::init::validate_initial_condition_config(vi);
    REQUIRE(r.ok);
}

TEST_CASE("validate: allowed-but-not-recommended trigger is allowed but warns",
          "[init][profile][validate]")
{
    // cartesian_cgrid recommends warm_bubble. vortex_seed is allowed, so it is
    // not an error, but using it instead of the recommended trigger warns.
    auto vi = make_default_cartesian_cgrid_inputs();
    vi.trigger.type = tmv::init::TriggerSourceConfig::Type::VortexSeed;
    const auto r = tmv::init::validate_initial_condition_config(vi);
    REQUIRE(r.ok);
    REQUIRE_FALSE(r.warnings.empty());
}

TEST_CASE("validate: unknown scheme is an error",
          "[init][profile][validate]")
{
    auto vi = make_default_cartesian_cgrid_inputs();
    vi.scheme_id = "made_up_scheme";
    const auto r = tmv::init::validate_initial_condition_config(vi);
    REQUIRE_FALSE(r.ok);
    REQUIRE_FALSE(r.errors.empty());
}

TEST_CASE("generated dynamics initialization-source matrix is descriptor exact",
          "[init][profile][validate][matrix][sci136]")
{
    using Sounding = tmv::init::SoundingSourceConfig::Type;
    using Hodograph = tmv::init::HodographSourceConfig::Type;
    using Trigger = tmv::init::TriggerSourceConfig::Type;

    const std::array<Sounding, 3> soundings = {
        Sounding::ParametricCAPE,
        Sounding::ParametricTargets,
        Sounding::File,
    };
    const std::array<Hodograph, 3> hodographs = {
        Hodograph::Auto,
        Hodograph::WKParam,
        Hodograph::Zero,
    };
    const std::array<Trigger, 3> triggers = {
        Trigger::WarmBubble,
        Trigger::None,
        Trigger::VortexSeed,
    };
    const std::array<StaggerType, 2> staggerings = {
        StaggerType::Collocated,
        StaggerType::CGrid,
    };
    const auto contains = [](const std::vector<std::string>& values,
                             const std::string& requested)
    {
        return std::find(values.begin(), values.end(), requested) !=
            values.end();
    };

    std::size_t evaluated = 0;
    for (const auto& dynamics : tmv::get_dynamics_scheme_descriptors())
    for (const auto& turbulence : tmv::get_turbulence_scheme_descriptors())
    for (const auto stagger : staggerings)
    for (const auto sounding : soundings)
    for (const auto hodograph : hodographs)
    for (const auto trigger : triggers)
    {
        tmv::init::ValidationInputs inputs;
        inputs.scheme_id = dynamics.id;
        inputs.turbulence_scheme_id = turbulence.id;
        inputs.coordinate = CoordinateSystem::Cartesian;
        inputs.stagger = stagger;
        inputs.sounding.type = sounding;
        inputs.hodograph.type = hodograph;
        inputs.trigger.type = trigger;

        const std::string staggering =
            stagger == StaggerType::Collocated ? "collocated" : "c_grid";
        const bool expected =
            contains(dynamics.capabilities.staggerings, staggering)
            && turbulence.constructible
            && contains(turbulence.capabilities.staggerings, staggering)
            && contains(
                dynamics.capabilities.accepted_sounding_sources,
                tmv::init::sounding_source_type_name(sounding))
            && contains(
                dynamics.capabilities.accepted_hodograph_sources,
                tmv::init::hodograph_source_type_name(hodograph))
            && contains(
                dynamics.capabilities.accepted_trigger_sources,
                tmv::init::trigger_source_type_name(trigger));
        const auto result =
            tmv::init::validate_initial_condition_config(inputs);
        INFO("dynamics=" << dynamics.id
             << " turbulence=" << turbulence.id
             << " staggering=" << staggering
             << " sounding="
             << tmv::init::sounding_source_type_name(sounding)
             << " hodograph="
             << tmv::init::hodograph_source_type_name(hodograph)
             << " trigger="
             << tmv::init::trigger_source_type_name(trigger));
        REQUIRE(result.ok == expected);
        ++evaluated;
    }

    REQUIRE(evaluated == 324);
}
