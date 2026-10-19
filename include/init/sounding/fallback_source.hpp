#pragma once

#include "init/sounding/sounding_source.hpp"

#include <memory>

namespace tmv::init
{

/**
 * @brief SoundingSource decorator: try primary, fall back on exception.
 *
 * Wraps two SoundingSource instances. build() invokes the primary; on any
 * std::exception, it logs a warning and delegates to the fallback. This
 * preserves the legacy `environment.sounding.use_fallback_profiles=true`
 * semantics, where a SHARPY load failure used to silently revert to the
 * procedural base state.
 *
 * The decorator intentionally does NOT swallow the fallback's exceptions;
 * if the fallback also fails, that propagates so the runtime sees a real
 * configuration problem.
 *
 * Use the factory to compose: when SoundingSourceConfig::File has
 * use_fallback_profiles=true, make_sounding_source returns a
 * FallbackSoundingSource(file, parametric).
 */
class FallbackSoundingSource final : public SoundingSource
{
public:
    FallbackSoundingSource(std::unique_ptr<SoundingSource> primary,
                           std::unique_ptr<SoundingSource> fallback);

    Sounding build(const std::vector<double>& z_m, double dz) const override;
    std::string describe() const override;

private:
    std::unique_ptr<SoundingSource> primary_;
    std::unique_ptr<SoundingSource> fallback_;
};

}  // namespace tmv::init
