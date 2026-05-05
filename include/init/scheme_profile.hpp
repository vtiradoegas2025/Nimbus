#pragma once

#include "core/coordinate_system.hpp"
#include "init/hodograph/factory.hpp"
#include "init/sounding/factory.hpp"
#include "init/trigger/factory.hpp"

#include <string>
#include <vector>

namespace tmv::init
{

/**
 * @file scheme_profile.hpp
 * @brief Per-dynamics-scheme initial-condition requirements.
 *
 * Each dynamics scheme has implicit requirements about what initial
 * conditions it can run with. Today these are scattered: tornado_sim.cpp
 * checks coordinate-system compatibility for the cartesian scheme; nothing
 * checks that tornado-vortex schemes get a vortex seed; nothing warns
 * when a supercell scheme runs with zero shear. This module makes those
 * requirements explicit, queryable, and validatable.
 *
 * The runtime calls validate(...) once during startup with the parsed
 * config. Errors abort the run; warnings log and continue.
 */

/**
 * @brief Coordinate-system tolerance.
 *
 * Cartesian / Cylindrical pin the run to one system; Either accepts both
 * (currently unused, reserved for future schemes that work on either).
 */
enum class CoordinateExpect
{
    Cartesian,
    Cylindrical,
    Either,
};

enum class StaggerExpect
{
    Collocated,
    CGrid,
    Either,
};

/**
 * @brief What a dynamics scheme accepts as initial conditions.
 *
 * Vectors of allowed types act as whitelists; anything not on the list is
 * a hard error. recommended_trigger_type names the most appropriate
 * trigger for the scheme; using a different (but allowed) trigger emits a
 * warning rather than an error so users can still experiment.
 */
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

/**
 * @brief Returns the profile for a dynamics scheme.
 *
 * @param scheme_id  The active dynamics scheme name (post alias resolution).
 * @return Profile for that scheme. Throws std::out_of_range if no profile
 *         is registered (i.e. a new scheme was added without updating
 *         the IC profile registry).
 */
const InitialConditionProfile& get_scheme_profile(const std::string& scheme_id);

/**
 * @brief Returns true when scheme_id has a registered profile.
 */
bool scheme_profile_exists(const std::string& scheme_id);

/**
 * @brief Outcome of validating a config against a scheme profile.
 *
 * Errors block the run; warnings are advisories. Both are formatted
 * for direct logging — already include scheme name and offending value
 * so the user can see what to change.
 */
struct ProfileValidationResult
{
    bool ok = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
};

/**
 * @brief Inputs to the validator. Mirrors the fields the runtime has
 *        already parsed by the time validate() is called.
 */
struct ValidationInputs
{
    std::string scheme_id;
    CoordinateSystem coordinate;
    StaggerType stagger;
    SoundingSourceConfig sounding;
    HodographSourceConfig hodograph;
    TriggerSourceConfig trigger;
};

/**
 * @brief Validates the parsed config against the active scheme's profile.
 */
ProfileValidationResult validate_initial_condition_config(const ValidationInputs& inputs);

}  // namespace tmv::init
