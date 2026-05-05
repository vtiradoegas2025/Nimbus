#pragma once

#include "init/sounding/parametric_targets.hpp"
#include "init/sounding/sounding_source.hpp"

#include <memory>
#include <string>

namespace tmv::init
{

/**
 * @brief Tagged config for one of the registered SoundingSource types.
 *
 * The runtime fills this from YAML (`environment.sounding.*`) and passes it
 * to make_sounding_source. Adding a new source type means:
 *   1. extending `SoundingSourceConfig::Type`,
 *   2. adding a parameter struct field if the source needs one,
 *   3. wiring a factory case in src/init/sounding/factory.cpp.
 *
 * Branch-specific knobs (parametric.*, file.*, targets.*) are populated
 * only when the matching type is selected; the others are ignored.
 */
struct SoundingSourceConfig
{
    enum class Type
    {
        ParametricCAPE,
        File,
        ParametricTargets,
    };

    Type type = Type::ParametricCAPE;

    // Parametric branch — populated when type == ParametricCAPE.
    struct Parametric
    {
        double cape_target_jkg = 2500.0;
        double surface_theta_k = 300.0;
        double surface_qv_kgkg = 0.014;
        double tropopause_z_m = 12000.0;
    } parametric;

    // File branch — populated when type == File. scheme_id picks the
    // file-format reader registered with src/soundings/factory.cpp
    // (today only "sharpy"). require_winds = true makes a missing
    // file hodograph an error rather than silently falling back to the
    // parametric WK 3-point hodograph.
    //
    // use_fallback_profiles preserves the legacy
    // environment.sounding.use_fallback_profiles semantics: when true and
    // the file load (or interpolation) fails, the runtime silently uses
    // ParametricCAPE built from the legacy environment.* knobs instead.
    // Default false because explicitly choosing type=file is a strong
    // intent — the legacy translation in runtime_config sets true to
    // preserve the historical behavior of pre-existing YAMLs.
    struct File
    {
        std::string path;
        std::string scheme_id = "sharpy";
        bool require_winds = true;
        bool use_fallback_profiles = false;
    } file;

    // Targets branch — populated when type == ParametricTargets.
    // Mirrors the user-facing diagnostic table on a Pivotal/SHARPpy
    // skew-T (CAPE, CIN, LCL, LFC, EL plus surface conditions).
    ParametricTargetsParams targets;
};

/**
 * @brief Resolve a SoundingSourceConfig::Type from a YAML string.
 *
 * Returns false if the string does not match a registered type. Unknown
 * strings should be reported by the caller with the originating YAML path
 * so the user gets a clear error.
 */
bool parse_sounding_source_type(const std::string& s,
                                SoundingSourceConfig::Type& out);

/// Returns the canonical YAML string for a registered type.
std::string sounding_source_type_name(SoundingSourceConfig::Type t);

/**
 * @brief Construct the configured SoundingSource.
 *
 * Defaults to ParametricCAPE when nothing is set in YAML, preserving the
 * historical behavior where every run gets the procedural CAPE column.
 */
std::unique_ptr<SoundingSource> make_sounding_source(const SoundingSourceConfig& cfg);

}  // namespace tmv::init
