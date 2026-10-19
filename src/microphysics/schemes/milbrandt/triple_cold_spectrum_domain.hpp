#pragma once

#include "triple_sedimentation.hpp"

namespace tmv::milbrandt_detail
{

enum class ColdSpectrumDomain
{
    Base200,
    SnowConvertedGraupel512,
    GraupelAndHail512
};

inline bool valid_cold_spectrum_domain(ColdSpectrumDomain domain) noexcept
{
    return domain == ColdSpectrumDomain::Base200 || domain == ColdSpectrumDomain::SnowConvertedGraupel512 ||
        domain == ColdSpectrumDomain::GraupelAndHail512;
}

inline double cold_spectrum_shape_limit(ColdSpectrumDomain domain, TripleMomentSpecies species) noexcept
{
    const bool graupel = species == TripleMomentSpecies::Graupel;
    const bool hail = species == TripleMomentSpecies::Hail;
    return (domain == ColdSpectrumDomain::SnowConvertedGraupel512 && graupel) ||
        (domain == ColdSpectrumDomain::GraupelAndHail512 && (graupel || hail)) ? 512.0 : 200.0;
}

} // namespace tmv::milbrandt_detail
