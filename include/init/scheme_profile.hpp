#pragma once

#include "core/infra/coordinate_system.hpp"
#include "init/hodograph/hodograph_source_config.hpp"
#include "init/sounding/sounding_source_config.hpp"
#include "init/trigger/trigger_source_config.hpp"

#include <string>
#include <vector>

namespace tmv::init
{

// Initial-condition compatibility is validated once at startup. Errors reject
// the run; warnings identify physically weak but intentionally allowed setups.
enum class CoordinateExpect
{
    Cartesian,
    Either,
};

enum class StaggerExpect
{
    Collocated,
    CGrid,
    Either,
};

// Allowed-source lists are hard compatibility boundaries. A non-recommended
// but allowed trigger remains an explicit experimental choice and only warns.
struct InitialConditionProfile
{
    std::string scheme_id;
    CoordinateExpect coordinate = CoordinateExpect::Either;
    StaggerExpect stagger = StaggerExpect::Either;

    std::vector<SoundingSourceConfig::Type> allowed_sounding_types;
    std::vector<HodographSourceConfig::Type> allowed_hodograph_types;
    std::vector<TriggerSourceConfig::Type> allowed_trigger_types;

    TriggerSourceConfig::Type recommended_trigger_type = TriggerSourceConfig::Type::WarmBubble;
    bool requires_nonzero_shear = false;
};

const InitialConditionProfile& get_scheme_profile(const std::string& scheme_id);
bool scheme_profile_exists(const std::string& scheme_id);
std::vector<std::string> get_registered_scheme_profile_ids();

struct ProfileValidationResult
{
    bool ok = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
};

struct ValidationInputs
{
    std::string scheme_id;
    std::string turbulence_scheme_id = "none";
    CoordinateSystem coordinate;
    StaggerType stagger;
    SoundingSourceConfig sounding;
    HodographSourceConfig hodograph;
    TriggerSourceConfig trigger;
};

ProfileValidationResult validate_initial_condition_config(const ValidationInputs& inputs);

}  // namespace tmv::init
