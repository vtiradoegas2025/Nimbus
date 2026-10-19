#include "microphysics/factory.hpp"
#include "microphysics/microphysics_base.hpp"
#include "schemes/kessler/kessler.hpp"
#include "schemes/lin/lin.hpp"
#include "schemes/thompson/thompson.hpp"
#include "schemes/milbrandt/milbrandt.hpp"
#include "schemes/none/none.hpp"
#include "util/scheme_factory.hpp"

#include <cmath>
#include <stdexcept>

namespace tmv
{

namespace
{
const SchemeRegistry<MicrophysicsScheme> registry({
    {"kessler",   [] { return std::make_unique<KesslerScheme>(); }},
    {"lin",       [] { return std::make_unique<LinScheme>(); }},
    {"thompson",  [] { return std::make_unique<ThompsonScheme>(); }},
    {"milbrandt", [] { return std::make_unique<MilbrandtScheme>(); }},
    {"none",      [] { return std::make_unique<microphysics::NoneScheme>(); }},
}, get_microphysics_scheme_descriptors());
}

std::unique_ptr<MicrophysicsScheme> create_microphysics_scheme(const std::string& scheme_name)
{
    return registry.create("microphysics", scheme_name);
}

std::unique_ptr<MicrophysicsScheme>
create_microphysics_scheme(const std::string& scheme_name,
                           const MicrophysicsConfig& config)
{
    const std::string canonical_name = registry.resolve(scheme_name);
    const auto validate = [](double value) {
        if (!std::isfinite(value) || value < 0.0)
        {
            throw std::invalid_argument(
                "microphysics parameters must be finite and non-negative");
        }
    };
    const auto validate_positive = [](double value) {
        if (!std::isfinite(value) || value <= 0.0)
        {
            throw std::invalid_argument(
                "microphysics aerosol concentrations must be finite and positive");
        }
    };
    if (canonical_name != "none" && canonical_name != "milbrandt")
    {
        validate(config.shared.cloud_water_threshold_kgkg);
        validate(config.shared.autoconversion_rate_s_inv);
    }
    if (canonical_name == "kessler")
    {
        validate(config.kessler.accretion_coefficient);
        validate(config.kessler.evaporation_efficiency);
    }
    else if (canonical_name == "lin")
    {
        validate(config.lin.accretion_coefficient);
        validate(config.lin.evaporation_rate_s_inv);
    }
    else if (canonical_name == "thompson")
    {
        validate(config.thompson.evaporation_rate_s_inv);
        validate_positive(config.thompson.water_friendly_aerosol_surface_m3);
        validate_positive(config.thompson.ice_friendly_aerosol_surface_m3);
        if (config.thompson.water_friendly_aerosol_surface_m3 <= 50.0e6
            || config.thompson.ice_friendly_aerosol_surface_m3 <= 0.5e6)
        {
            throw std::invalid_argument(
                "Thompson surface aerosol concentrations must exceed their fixed free-troposphere backgrounds");
        }
    }
    auto selected = registry.create("microphysics", scheme_name);
    selected->configure(config);
    return selected;
}

std::vector<std::string> get_available_microphysics_schemes()
{
    return registry.available_ids();
}

} // namespace tmv
