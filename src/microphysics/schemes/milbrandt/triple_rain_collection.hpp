#pragma once

#include "triple_frozen_collection.hpp"

namespace tmv::milbrandt_detail
{

struct RainCollectionCell
{
    FrozenCollectionCell frozen;
    FrozenCollectionPopulation rain;
};

struct RainCollectionChannel
{
    double raw_mass_per_kg = 0.0;
    double raw_number_per_kg = 0.0;
    double mass_per_kg = 0.0;
    double number_per_kg = 0.0;
};

class RainCollectionProducer;

bool produce_exact_rain_collection_admission(
    const ColdParameters&, const RainCollectionCell&, double duration_s,
    RainCollectionProducer&, std::string& error, std::size_t capacity = 4,
    FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

bool validate_exact_rain_collection_admission(
    const ColdParameters&, const RainCollectionCell&, double duration_s,
    const RainCollectionProducer&, std::string& error, std::size_t capacity = 4,
    FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

class RainCollectionProducer
{
public:
    const RainCollectionCell& source() const noexcept { return source_; }
    const std::array<double, 25>& parameters() const noexcept { return parameters_; }
    bool hail_processes() const noexcept { return hail_processes_; }
    ColdSpectrumDomain spectrum_domain() const noexcept { return spectrum_domain_; }
    double duration_s() const noexcept { return duration_s_; }
    double admission_factor() const noexcept { return admission_factor_; }
    const std::array<RainCollectionChannel, 4>& channels() const noexcept { return channels_; }
    const std::array<double, 5>& source_alpha() const noexcept { return source_alpha_; }
    const std::array<double, 5>& source_lambda_per_m() const noexcept { return source_lambda_per_m_; }
    const std::array<double, 5>& mass_weighted_speed_m_s() const noexcept { return mass_weighted_speed_m_s_; }
    std::size_t event_count() const noexcept { return valid_ ? 4 : 0; }

private:
    friend bool produce_exact_rain_collection_admission(
        const ColdParameters&, const RainCollectionCell&, double, RainCollectionProducer&,
        std::string&, std::size_t, FrozenCollectionFailure*, ColdSpectrumDomain);

    RainCollectionCell source_{};
    std::array<double, 25> parameters_{};
    bool hail_processes_ = false;
    ColdSpectrumDomain spectrum_domain_ = ColdSpectrumDomain::Base200;
    double duration_s_ = 0.0;
    double admission_factor_ = 1.0;
    std::array<RainCollectionChannel, 4> channels_{};
    std::array<double, 5> source_alpha_{};
    std::array<double, 5> source_lambda_per_m_{};
    std::array<double, 5> mass_weighted_speed_m_s_{};
    bool valid_ = false;
};

} // namespace tmv::milbrandt_detail
