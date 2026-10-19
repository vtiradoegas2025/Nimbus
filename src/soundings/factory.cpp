#include "soundings/factory.hpp"
#include "soundings/soundings_base.hpp"
#include "schemes/auto/auto_sounding.hpp"
#include "schemes/cm1/cm1_sounding.hpp"
#include "schemes/sharpy/sharpy_sounding.hpp"
#include "util/scheme_factory.hpp"

namespace
{
const tmv::SchemeRegistry<SoundingScheme> registry({
    {"auto", [] { return std::make_unique<AutoSoundingScheme>(); }},
    {"cm1", [] { return std::make_unique<Cm1SoundingScheme>(); }},
    {"sharpy", [] { return std::make_unique<SharpySoundingScheme>(); }},
}, get_sounding_scheme_descriptors());
}

std::unique_ptr<SoundingScheme> create_sharpy_sounding_scheme()
{
    return std::make_unique<SharpySoundingScheme>();
}

std::unique_ptr<SoundingScheme> create_sounding_scheme(const std::string& scheme_id)
{
    const std::string normalized = tmv::canonicalize_scheme_id(
        get_sounding_scheme_descriptors(), scheme_id);

    if (normalized == "none" || normalized.empty())
    {
        return nullptr;
    }

    return registry.create("sounding", scheme_id);
}
