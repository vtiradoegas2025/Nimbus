#pragma once

#include "triple_cold_riming.hpp"

namespace tmv::milbrandt_detail
{

enum class IceGraupelConversionFailure
{
    None,
    InvalidSource,
    InvalidContext,
    Capacity,
    InvalidProducer,
    IntervalRejected
};

class IceGraupelConversionProducer;

bool produce_exact_ice_graupel_conversion_admission(
    const ColdRimingProducer& riming_source, IceGraupelConversionProducer& producer,
    std::string& error, std::size_t capacity = 1,
    IceGraupelConversionFailure* failure = nullptr);

class IceGraupelConversionProducer
{
public:
    const ColdRimingProducer& riming_source() const noexcept
    {
        return riming_source_;
    }

    double raw_mass_transfer_per_kg() const noexcept
    {
        return raw_mass_transfer_per_kg_;
    }

    double admitted_mass_transfer_per_kg() const noexcept
    {
        return admitted_mass_transfer_per_kg_;
    }

    double donor_number_loss_per_kg() const noexcept
    {
        return donor_number_loss_per_kg_;
    }

    double recipient_number_gain_per_kg() const noexcept
    {
        return recipient_number_gain_per_kg_;
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
    friend bool produce_exact_ice_graupel_conversion_admission(
        const ColdRimingProducer&, IceGraupelConversionProducer&,
        std::string&, std::size_t, IceGraupelConversionFailure*);

    ColdRimingProducer riming_source_{};
    double raw_mass_transfer_per_kg_ = 0.0;
    double admitted_mass_transfer_per_kg_ = 0.0;
    double donor_number_loss_per_kg_ = 0.0;
    double recipient_number_gain_per_kg_ = 0.0;
    std::array<double, 4> mass_increment_per_kg_{};
    std::array<double, 4> number_increment_per_kg_{};
    std::array<double, 4> reflectivity_increment_per_kg_{};
    bool valid_ = false;
};

} // namespace tmv::milbrandt_detail
