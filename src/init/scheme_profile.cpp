/**
 * @file scheme_profile.cpp
 * @brief Per-scheme IC requirement registry + the runtime validator.
 *
 * Adding a new dynamics scheme: register its profile in build_registry()
 * below alongside the existing entries. Adding a new sounding / hodograph
 * / trigger type the scheme can accept: extend the profile's allowed_*
 * lists. The validator does the rest at startup.
 */

#include "init/scheme_profile.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace tmv::init
{

namespace
{

bool contains(const std::vector<SoundingSourceConfig::Type>& v,
              SoundingSourceConfig::Type t)
{
    return std::find(v.begin(), v.end(), t) != v.end();
}
bool contains(const std::vector<HodographSourceConfig::Type>& v,
              HodographSourceConfig::Type t)
{
    return std::find(v.begin(), v.end(), t) != v.end();
}
bool contains(const std::vector<TriggerSourceConfig::Type>& v,
              TriggerSourceConfig::Type t)
{
    return std::find(v.begin(), v.end(), t) != v.end();
}

std::string coordinate_expect_name(CoordinateExpect c)
{
    switch (c)
    {
        case CoordinateExpect::Cartesian:   return "cartesian";
        case CoordinateExpect::Cylindrical: return "cylindrical";
        case CoordinateExpect::Either:      return "either";
    }
    return "unknown";
}

std::string stagger_expect_name(StaggerExpect s)
{
    switch (s)
    {
        case StaggerExpect::Collocated: return "collocated";
        case StaggerExpect::CGrid:      return "c_grid";
        case StaggerExpect::Either:     return "either";
    }
    return "unknown";
}

const std::unordered_map<std::string, InitialConditionProfile>& build_registry()
{
    using Snd = SoundingSourceConfig::Type;
    using Hod = HodographSourceConfig::Type;
    using Trg = TriggerSourceConfig::Type;

    static const std::unordered_map<std::string, InitialConditionProfile> table = {
        {
            "cartesian",
            {
                /*scheme_id=*/             "cartesian",
                /*coordinate=*/            CoordinateExpect::Cartesian,
                /*stagger=*/               StaggerExpect::Collocated,
                /*allowed_sounding_types=*/{Snd::ParametricCAPE, Snd::File, Snd::ParametricTargets},
                /*allowed_hodograph_types=*/{Hod::Auto, Hod::WKParam, Hod::Zero},
                /*allowed_trigger_types=*/ {Trg::WarmBubble, Trg::None, Trg::VortexSeed},
                /*recommended_trigger=*/   Trg::WarmBubble,
                /*requires_nonzero_shear=*/false,
            }
        },
        {
            "supercell",
            {
                "supercell",
                CoordinateExpect::Cylindrical,
                StaggerExpect::Collocated,
                {Snd::ParametricCAPE, Snd::File, Snd::ParametricTargets},
                {Hod::Auto, Hod::WKParam, Hod::Zero},
                {Trg::WarmBubble, Trg::None},
                Trg::WarmBubble,
                /*requires_nonzero_shear=*/true,
            }
        },
        {
            "supercell_cgrid",
            {
                "supercell_cgrid",
                CoordinateExpect::Cylindrical,
                StaggerExpect::CGrid,
                {Snd::ParametricCAPE, Snd::File, Snd::ParametricTargets},
                {Hod::Auto, Hod::WKParam, Hod::Zero},
                {Trg::WarmBubble, Trg::None},
                Trg::WarmBubble,
                /*requires_nonzero_shear=*/true,
            }
        },
        {
            "tornado",
            {
                "tornado",
                CoordinateExpect::Cylindrical,
                StaggerExpect::Collocated,
                {Snd::ParametricCAPE, Snd::File, Snd::ParametricTargets},
                {Hod::Auto, Hod::WKParam, Hod::Zero},
                {Trg::VortexSeed, Trg::WarmBubble, Trg::None},
                Trg::VortexSeed,
                /*requires_nonzero_shear=*/false,
            }
        },
        {
            "tornado_cgrid",
            {
                "tornado_cgrid",
                CoordinateExpect::Cylindrical,
                StaggerExpect::CGrid,
                {Snd::ParametricCAPE, Snd::File, Snd::ParametricTargets},
                {Hod::Auto, Hod::WKParam, Hod::Zero},
                {Trg::VortexSeed, Trg::WarmBubble, Trg::None},
                Trg::VortexSeed,
                /*requires_nonzero_shear=*/false,
            }
        },
    };
    return table;
}

/// Resolves the legacy aliases that src/dynamics/factory.cpp accepts so a
/// user setting `dynamics.scheme: mesocyclone_cgrid` still gets the
/// supercell_cgrid profile. Keep this in sync with the alias list there.
std::string canonicalize(const std::string& scheme_id)
{
    static const std::unordered_map<std::string, std::string> aliases = {
        {"mesocyclone",        "supercell"},
        {"mesocyclone_cgrid",  "supercell_cgrid"},
        {"axisymmetric",       "tornado"},
        {"axisymmetric_cgrid", "tornado_cgrid"},
        {"cart",               "cartesian"},
        {"cartesian_cpu",      "cartesian"},
    };
    auto it = aliases.find(scheme_id);
    return (it == aliases.end()) ? scheme_id : it->second;
}

}  // namespace

const InitialConditionProfile& get_scheme_profile(const std::string& scheme_id)
{
    const auto& table = build_registry();
    const std::string canonical = canonicalize(scheme_id);
    auto it = table.find(canonical);
    if (it == table.end())
    {
        throw std::out_of_range(
            "get_scheme_profile: no IC profile registered for dynamics scheme '"
            + scheme_id + "' (canonical: '" + canonical + "'). Register one in "
            "src/init/scheme_profile.cpp::build_registry().");
    }
    return it->second;
}

bool scheme_profile_exists(const std::string& scheme_id)
{
    const auto& table = build_registry();
    return table.find(canonicalize(scheme_id)) != table.end();
}

ProfileValidationResult validate_initial_condition_config(const ValidationInputs& inputs)
{
    ProfileValidationResult r;

    if (!scheme_profile_exists(inputs.scheme_id))
    {
        std::ostringstream ss;
        ss << "[CONFIG ERROR] Unknown dynamics scheme '" << inputs.scheme_id
           << "'. Recognized: cartesian, supercell, supercell_cgrid, tornado, "
              "tornado_cgrid (plus aliases). Aborting.";
        r.errors.push_back(ss.str());
        r.ok = false;
        return r;
    }

    const InitialConditionProfile& p = get_scheme_profile(inputs.scheme_id);

    // Coordinate compatibility.
    const bool coord_cartesian = (inputs.coordinate == CoordinateSystem::Cartesian);
    const bool coord_cylindrical = (inputs.coordinate == CoordinateSystem::Cylindrical);
    const bool coord_ok =
        (p.coordinate == CoordinateExpect::Either)
        || (p.coordinate == CoordinateExpect::Cartesian && coord_cartesian)
        || (p.coordinate == CoordinateExpect::Cylindrical && coord_cylindrical);
    if (!coord_ok)
    {
        std::ostringstream ss;
        ss << "[CONFIG ERROR] dynamics.scheme=" << p.scheme_id
           << " requires coordinate_system="
           << coordinate_expect_name(p.coordinate) << ", but config says "
           << coordinate_system_name(inputs.coordinate) << ". Aborting.";
        r.errors.push_back(ss.str());
        r.ok = false;
    }

    // Staggering compatibility.
    const bool stagger_collocated = (inputs.stagger == StaggerType::Collocated);
    const bool stagger_cgrid = (inputs.stagger == StaggerType::CGrid);
    const bool stagger_ok =
        (p.stagger == StaggerExpect::Either)
        || (p.stagger == StaggerExpect::Collocated && stagger_collocated)
        || (p.stagger == StaggerExpect::CGrid && stagger_cgrid);
    if (!stagger_ok)
    {
        std::ostringstream ss;
        ss << "[CONFIG ERROR] dynamics.scheme=" << p.scheme_id
           << " requires grid.staggering=" << stagger_expect_name(p.stagger)
           << ", but config says "
           << (stagger_collocated ? "collocated" : "c_grid") << ". Aborting.";
        r.errors.push_back(ss.str());
        r.ok = false;
    }

    // Sounding type allowed.
    if (!p.allowed_sounding_types.empty()
        && !contains(p.allowed_sounding_types, inputs.sounding.type))
    {
        std::ostringstream ss;
        ss << "[CONFIG ERROR] dynamics.scheme=" << p.scheme_id
           << " does not accept environment.sounding.type="
           << sounding_source_type_name(inputs.sounding.type) << ". Aborting.";
        r.errors.push_back(ss.str());
        r.ok = false;
    }

    // Hodograph type allowed.
    if (!p.allowed_hodograph_types.empty()
        && !contains(p.allowed_hodograph_types, inputs.hodograph.type))
    {
        std::ostringstream ss;
        ss << "[CONFIG ERROR] dynamics.scheme=" << p.scheme_id
           << " does not accept environment.hodograph.type="
           << hodograph_source_type_name(inputs.hodograph.type) << ". Aborting.";
        r.errors.push_back(ss.str());
        r.ok = false;
    }

    // Trigger type allowed.
    if (!p.allowed_trigger_types.empty()
        && !contains(p.allowed_trigger_types, inputs.trigger.type))
    {
        std::ostringstream ss;
        ss << "[CONFIG ERROR] dynamics.scheme=" << p.scheme_id
           << " does not accept trigger.type="
           << trigger_source_type_name(inputs.trigger.type)
           << ". Aborting.";
        r.errors.push_back(ss.str());
        r.ok = false;
    }
    else if (inputs.trigger.type != p.recommended_trigger_type)
    {
        std::ostringstream ss;
        ss << "[CONFIG WARNING] dynamics.scheme=" << p.scheme_id
           << " recommends trigger.type="
           << trigger_source_type_name(p.recommended_trigger_type)
           << " for realistic spin-up; got "
           << trigger_source_type_name(inputs.trigger.type)
           << ". Run will continue.";
        r.warnings.push_back(ss.str());
    }

    // Shear sanity check. The hodograph anchors live on the WK branch of
    // the config; if every anchor is identically zero (and the hodograph
    // type isn't explicitly Zero), we warn the user that a scheme that
    // requires nonzero shear will produce a degenerate run.
    if (p.requires_nonzero_shear
        && inputs.hodograph.type != HodographSourceConfig::Type::Zero
        && inputs.sounding.type != SoundingSourceConfig::Type::File)
    {
        const auto& a = inputs.hodograph.wk_anchors;
        const bool all_zero =
            (a.u_sfc_ms == 0.0 && a.v_sfc_ms == 0.0
             && a.u_1km_ms == 0.0 && a.v_1km_ms == 0.0
             && a.u_6km_ms == 0.0 && a.v_6km_ms == 0.0);
        if (all_zero)
        {
            std::ostringstream ss;
            ss << "[CONFIG WARNING] dynamics.scheme=" << p.scheme_id
               << " expects nonzero environmental shear; every WK hodograph "
                  "anchor in environment.hodograph.* is 0. Storm-scale "
                  "convection will not organize without shear.";
            r.warnings.push_back(ss.str());
        }
    }

    return r;
}

}  // namespace tmv::init
