#pragma once

#include "triple_cold_riming.hpp"

namespace tmv::milbrandt_detail
{

struct GraupelHailConversionCell
{
    ColdRimingCell source;
    double rain_mass_per_kg = 0.0;
    std::array<double, 3> admitted_accretion_mass_per_kg{};
    double graupel_diffusion_mass_per_kg = 0.0;
    double graupel_diffusion_number_per_kg = 0.0;
};

enum class GraupelHailConversionFailure
{
    None,
    InvalidSource,
    InvalidContext,
    Capacity,
    InvalidProducer,
    IntervalRejected
};

enum class GraupelHailSllRegime
{
    HailDisabled,
    NoGraupel,
    NoAccretion,
    NonpositiveLoading,
    BelowSizeRatio,
    Active
};

class GraupelHailConversionProducer;
struct GraupelHailConversionResult;

bool produce_exact_graupel_hail_conversion_admission(
    const ColdParameters&, const GraupelHailConversionCell&, double duration_s,
    GraupelHailConversionProducer&, std::string& error,
    std::size_t capacity = 1, GraupelHailConversionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

bool evaluate_exact_triple_graupel_hail_conversion_cell(
    const ColdParameters&, const GraupelHailConversionCell&, double duration_s,
    GraupelHailConversionResult&, std::string& error,
    std::size_t capacity = 1, GraupelHailConversionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

bool validate_exact_graupel_hail_conversion_admission(
    const ColdParameters&, const GraupelHailConversionCell&, double duration_s,
    const GraupelHailConversionProducer&, std::string& error,
    std::size_t capacity = 1, GraupelHailConversionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

bool apply_exact_triple_graupel_hail_conversion_cell(
    const ColdParameters&, const GraupelHailConversionCell&, double duration_s,
    const GraupelHailConversionProducer&, GraupelHailConversionResult&, std::string& error,
    std::size_t capacity = 1, GraupelHailConversionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

class GraupelHailConversionProducer
{
public:
    const GraupelHailConversionCell& source() const noexcept
    {
        return source_;
    }
    const std::array<double, 25>& parameters() const noexcept
    {
        return parameters_;
    }
    bool hail_processes() const noexcept
    {
        return hail_processes_;
    }
    ColdSpectrumDomain spectrum_domain() const noexcept
    {
        return spectrum_domain_;
    }
    double duration_s() const noexcept
    {
        return duration_s_;
    }
    GraupelHailSllRegime regime() const noexcept
    {
        return regime_;
    }
    double loading_denominator() const noexcept
    {
        return loading_denominator_;
    }
    double source_alpha() const noexcept
    {
        return source_alpha_;
    }
    double source_lambda_per_m() const noexcept
    {
        return source_lambda_per_m_;
    }
    double mean_mass_diameter_m() const noexcept
    {
        return mean_mass_diameter_m_;
    }
    double hail_embryo_diameter_m() const noexcept
    {
        return hail_embryo_diameter_m_;
    }
    double size_ratio() const noexcept
    {
        return size_ratio_;
    }
    double raw_mass_transfer_per_kg() const noexcept
    {
        return raw_mass_transfer_per_kg_;
    }
    double admitted_mass_transfer_per_kg() const noexcept
    {
        return admitted_mass_transfer_per_kg_;
    }
    double admitted_fraction() const noexcept
    {
        return admitted_fraction_;
    }
    double donor_number_loss_per_kg() const noexcept
    {
        return donor_number_loss_per_kg_;
    }
    double recipient_number_gain_per_kg() const noexcept
    {
        return donor_number_loss_per_kg_;
    }
    const std::array<double, 4>& mass_increment_per_kg() const noexcept
    {
        return mass_increment_per_kg_;
    }
    const std::array<double, 4>& number_increment_per_kg() const noexcept
    {
        return number_increment_per_kg_;
    }
    const std::array<double, 4>& reflectivity_increment_per_kg() const noexcept
    {
        return reflectivity_increment_per_kg_;
    }
    std::size_t event_count() const noexcept
    {
        return valid_ ? 1 : 0;
    }

private:
    friend bool produce_exact_graupel_hail_conversion_admission(
        const ColdParameters&, const GraupelHailConversionCell&, double,
        GraupelHailConversionProducer&, std::string&, std::size_t,
        GraupelHailConversionFailure*, ColdSpectrumDomain);

    GraupelHailConversionCell source_{};
    std::array<double, 25> parameters_{};
    bool hail_processes_ = false;
    ColdSpectrumDomain spectrum_domain_ = ColdSpectrumDomain::Base200;
    double duration_s_ = 0.0;
    GraupelHailSllRegime regime_ = GraupelHailSllRegime::NoGraupel;
    double loading_denominator_ = 0.0;
    double source_alpha_ = 0.0;
    double source_lambda_per_m_ = 0.0;
    double mean_mass_diameter_m_ = 0.0;
    double hail_embryo_diameter_m_ = 0.0;
    double size_ratio_ = 0.0;
    double raw_mass_transfer_per_kg_ = 0.0;
    double admitted_mass_transfer_per_kg_ = 0.0;
    double admitted_fraction_ = 0.0;
    double donor_number_loss_per_kg_ = 0.0;
    std::array<double, 4> mass_increment_per_kg_{};
    std::array<double, 4> number_increment_per_kg_{};
    std::array<double, 4> reflectivity_increment_per_kg_{};
    bool valid_ = false;
};

struct GraupelHailConversionResult
{
    std::array<FrozenCollectionPopulation, 4> unrounded{};
    std::array<FrozenCollectionEndpoint, 4> rounded{};
    std::array<double, 4> mass_increment_per_kg{};
    std::array<double, 4> number_increment_per_kg{};
    std::array<double, 4> reflectivity_increment_per_kg{};
    double cloud_mass_increment_per_kg = 0.0;
    double rain_mass_increment_per_kg = 0.0;
    double vapor_mass_increment_per_kg = 0.0;
    double temperature_increment_k = 0.0;
    GraupelHailConversionProducer producer;
};

} // namespace tmv::milbrandt_detail
