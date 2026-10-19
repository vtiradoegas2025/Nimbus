#pragma once

#include "soundings/soundings_base.hpp"

#include <memory>
#include <string>

class AutoSoundingScheme final : public SoundingScheme
{
public:
    void initialize(const SoundingConfig& config) override;
    SoundingData load_sounding(const std::string& file_path) override;
    SoundingData interpolate_to_heights(
        const SoundingData& sounding,
        const std::vector<double>& target_heights_m) override;

    const SoundingConfig& get_config() const override { return config_; }
    bool is_initialized() const override { return delegate_ != nullptr; }

private:
    SoundingConfig config_;
    std::string resolved_scheme_id_;
    std::unique_ptr<SoundingScheme> delegate_;
};
