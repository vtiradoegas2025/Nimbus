/**
 * @file fallback_source.cpp
 * @brief FallbackSoundingSource implementation.
 */

#include "init/sounding/fallback_source.hpp"
#include "util/log.hpp"

#include <stdexcept>
#include <utility>

namespace tmv::init
{

FallbackSoundingSource::FallbackSoundingSource(std::unique_ptr<SoundingSource> primary,
                                                std::unique_ptr<SoundingSource> fallback)
    : primary_(std::move(primary)), fallback_(std::move(fallback))
{
    if (!primary_ || !fallback_)
    {
        throw std::invalid_argument(
            "FallbackSoundingSource: primary and fallback must both be non-null");
    }
}

Sounding FallbackSoundingSource::build(const std::vector<double>& z_m, double dz) const
{
    try
    {
        return primary_->build(z_m, dz);
    }
    catch (const std::exception& e)
    {
        tmv::log_warn("Sounding source '", primary_->describe(),
                      "' failed: ", e.what(),
                      ". Falling back to '", fallback_->describe(), "'.");
        return fallback_->build(z_m, dz);
    }
}

std::string FallbackSoundingSource::describe() const
{
    return primary_->describe() + "+fallback(" + fallback_->describe() + ")";
}

}  // namespace tmv::init
