#include "init/scheme_profile.hpp"

#include "dynamics/scheme_descriptor.hpp"
#include "init/hodograph/factory.hpp"
#include "init/sounding/factory.hpp"
#include "init/trigger/factory.hpp"
#include "turbulence/scheme_descriptor.hpp"

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

InitialConditionProfile build_profile(const SchemeDescriptor& descriptor)
{
    InitialConditionProfile profile;
    profile.scheme_id = descriptor.id;
    profile.coordinate = CoordinateExpect::Cartesian;

    const auto& staggerings = descriptor.capabilities.staggerings;
    if (std::find(staggerings.begin(), staggerings.end(), "collocated")
        != staggerings.end())
    {
        profile.stagger = StaggerExpect::Collocated;
    }
    else if (std::find(staggerings.begin(), staggerings.end(), "c_grid")
             != staggerings.end())
    {
        profile.stagger = StaggerExpect::CGrid;
    }
    else
    {
        throw std::logic_error(
            "Dynamics descriptor has no supported profile staggering: "
            + descriptor.id);
    }

    for (const auto& id : descriptor.capabilities.accepted_sounding_sources)
    {
        SoundingSourceConfig::Type type;
        if (!parse_sounding_source_type(id, type))
        {
            throw std::logic_error(
                "Dynamics descriptor has unknown sounding source '" + id + "'");
        }
        profile.allowed_sounding_types.push_back(type);
    }
    for (const auto& id : descriptor.capabilities.accepted_hodograph_sources)
    {
        HodographSourceConfig::Type type;
        if (!parse_hodograph_source_type(id, type))
        {
            throw std::logic_error(
                "Dynamics descriptor has unknown hodograph source '" + id + "'");
        }
        profile.allowed_hodograph_types.push_back(type);
    }
    for (const auto& id : descriptor.capabilities.accepted_trigger_sources)
    {
        TriggerSourceConfig::Type type;
        if (!parse_trigger_source_type(id, type))
        {
            throw std::logic_error(
                "Dynamics descriptor has unknown trigger source '" + id + "'");
        }
        profile.allowed_trigger_types.push_back(type);
    }
    if (!parse_trigger_source_type(
            descriptor.capabilities.recommended_trigger_source,
            profile.recommended_trigger_type))
    {
        throw std::logic_error(
            "Dynamics descriptor has unknown recommended trigger source");
    }
    return profile;
}

const std::unordered_map<std::string, InitialConditionProfile>& build_registry()
{
    // Initialization compatibility is generated from the dynamics catalog.
    static const std::unordered_map<std::string, InitialConditionProfile> table = [] {
        std::unordered_map<std::string, InitialConditionProfile> result;
        for (const auto& descriptor : get_dynamics_scheme_descriptors())
        {
            result.emplace(descriptor.id, build_profile(descriptor));
        }
        return result;
    }();
    return table;
}

std::string canonicalize(const std::string& scheme_id)
{
    return canonicalize_dynamics_scheme_id(scheme_id);
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
            + scheme_id + "' (canonical: '" + canonical + "'). Add its "
            "initialization compatibility to the dynamics descriptor.");
    }
    return it->second;
}

bool scheme_profile_exists(const std::string& scheme_id)
{
    const auto& table = build_registry();
    return table.find(canonicalize(scheme_id)) != table.end();
}

std::vector<std::string> get_registered_scheme_profile_ids()
{
    std::vector<std::string> ids;
    ids.reserve(build_registry().size());
    for (const auto& entry : build_registry())
    {
        ids.push_back(entry.first);
    }
    return ids;
}

ProfileValidationResult validate_initial_condition_config(const ValidationInputs& inputs)
{
    ProfileValidationResult r;

    if (!scheme_profile_exists(inputs.scheme_id))
    {
        std::ostringstream ss;
        ss << "[CONFIG ERROR] Unknown dynamics scheme '" << inputs.scheme_id
           << "' (canonical: '" << canonicalize(inputs.scheme_id) << "'). "
              "Recognized canonical names:";
        const auto& table = build_registry();
        bool first = true;
        for (const auto& [id, _profile] : table)
        {
            ss << (first ? " " : ", ") << id;
            first = false;
        }
        ss << ". Aliases are listed by the dynamics scheme descriptors. Aborting.";
        r.errors.push_back(ss.str());
        r.ok = false;
        return r;
    }

    const InitialConditionProfile& p = get_scheme_profile(inputs.scheme_id);

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

    // Cross-component staggering truth comes from the scheme descriptor.
    // Smagorinsky and TKE both own bounded collocated equations as well as
    // their C-grid paths; a future scheme with narrower storage support is
    // rejected here before fields are allocated or advanced.
    const auto* turbulence_descriptor = find_scheme_descriptor(
        get_turbulence_scheme_descriptors(), inputs.turbulence_scheme_id);
    if (turbulence_descriptor == nullptr || !turbulence_descriptor->constructible)
    {
        r.errors.push_back(
            "[CONFIG ERROR] Unknown turbulence scheme '"
            + inputs.turbulence_scheme_id + "'. Aborting.");
        r.ok = false;
    }
    else
    {
        const std::string requested_stagger = stagger_collocated
            ? "collocated" : "c_grid";
        const auto& supported = turbulence_descriptor->capabilities.staggerings;
        if (std::find(supported.begin(), supported.end(), requested_stagger)
            == supported.end())
        {
            std::ostringstream ss;
            ss << "[CONFIG ERROR] turbulence.scheme=" << turbulence_descriptor->id
               << " does not support grid.staggering=" << requested_stagger
               << "; supported staggering: ";
            for (std::size_t index = 0; index < supported.size(); ++index)
            {
                if (index > 0) ss << ", ";
                ss << supported[index];
            }
            ss << ". Aborting.";
            r.errors.push_back(ss.str());
            r.ok = false;
        }
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
