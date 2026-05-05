/**
 * @file factory.cpp
 * @brief Dispatcher from SoundingSourceConfig to a concrete SoundingSource.
 *
 * Adding a new source: register the YAML name in parse_sounding_source_type
 * and sounding_source_type_name, then handle the new enumerator in
 * make_sounding_source. Everything else (init wiring, broadcast) is shared.
 */

#include "init/sounding/factory.hpp"
#include "init/sounding/fallback_source.hpp"
#include "init/sounding/file_sounding.hpp"
#include "init/sounding/parametric_cape.hpp"
#include "init/sounding/parametric_targets.hpp"

#include <cctype>
#include <stdexcept>

namespace tmv::init
{

namespace
{

std::string normalize(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
    {
        if (std::isspace(static_cast<unsigned char>(c)))
        {
            continue;
        }
        if (c == '-')
        {
            out.push_back('_');
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

}  // namespace

bool parse_sounding_source_type(const std::string& s, SoundingSourceConfig::Type& out)
{
    const std::string norm = normalize(s);
    if (norm == "parametric_cape" || norm == "parametriccape" || norm == "cape")
    {
        out = SoundingSourceConfig::Type::ParametricCAPE;
        return true;
    }
    if (norm == "file" || norm == "sharpy" || norm == "raob" || norm == "sounding_file")
    {
        out = SoundingSourceConfig::Type::File;
        return true;
    }
    if (norm == "parametric_targets" || norm == "parametrictargets" || norm == "targets"
        || norm == "diagnostics" || norm == "from_table")
    {
        out = SoundingSourceConfig::Type::ParametricTargets;
        return true;
    }
    return false;
}

std::string sounding_source_type_name(SoundingSourceConfig::Type t)
{
    switch (t)
    {
        case SoundingSourceConfig::Type::ParametricCAPE:
            return "parametric_cape";
        case SoundingSourceConfig::Type::File:
            return "file";
        case SoundingSourceConfig::Type::ParametricTargets:
            return "parametric_targets";
    }
    return "unknown";
}

std::unique_ptr<SoundingSource> make_sounding_source(const SoundingSourceConfig& cfg)
{
    switch (cfg.type)
    {
        case SoundingSourceConfig::Type::ParametricCAPE:
        {
            ParametricCAPEParams p;
            p.cape_target_jkg = cfg.parametric.cape_target_jkg;
            p.surface_theta_k = cfg.parametric.surface_theta_k;
            p.surface_qv_kgkg = cfg.parametric.surface_qv_kgkg;
            p.tropopause_z_m = cfg.parametric.tropopause_z_m;
            return std::make_unique<ParametricCAPESoundingSource>(p);
        }
        case SoundingSourceConfig::Type::File:
        {
            FileSoundingParams fp;
            fp.file_path = cfg.file.path;
            fp.scheme_id = cfg.file.scheme_id;
            fp.require_winds = cfg.file.require_winds;
            auto file_source = std::make_unique<FileSoundingSource>(fp);
            if (!cfg.file.use_fallback_profiles)
            {
                return file_source;
            }
            // Legacy `use_fallback_profiles=true` semantics: if the file
            // load fails, build a parametric column from the same legacy
            // environment.* knobs and use that instead. Wrapped via
            // FallbackSoundingSource so the policy lives in one place
            // and applies to any future primary source too.
            ParametricCAPEParams pp;
            pp.cape_target_jkg = cfg.parametric.cape_target_jkg;
            pp.surface_theta_k = cfg.parametric.surface_theta_k;
            pp.surface_qv_kgkg = cfg.parametric.surface_qv_kgkg;
            pp.tropopause_z_m = cfg.parametric.tropopause_z_m;
            auto fallback = std::make_unique<ParametricCAPESoundingSource>(pp);
            return std::make_unique<FallbackSoundingSource>(std::move(file_source),
                                                            std::move(fallback));
        }
        case SoundingSourceConfig::Type::ParametricTargets:
            return std::make_unique<ParametricTargetsSoundingSource>(cfg.targets);
    }
    throw std::logic_error("make_sounding_source: unhandled SoundingSourceConfig::Type");
}

}  // namespace tmv::init
