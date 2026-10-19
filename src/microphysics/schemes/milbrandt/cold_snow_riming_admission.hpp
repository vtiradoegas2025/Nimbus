#pragma once

#include <array>
#include <stdexcept>

namespace tmv::milbrandt_detail
{

inline constexpr unsigned cold_snow_riming_admission_max_passes = 16;
// Quantize the paired cloud factor before freezing its journal, across host and compensated GPU arithmetic.
inline constexpr double cold_snow_riming_cloud_factor_scale = 0x1p40;
inline constexpr double cold_snow_riming_cloud_factor_quantum = 0x1p-40;
inline constexpr double cold_snow_riming_overdraw_factor = 1.0 - 0x1p-40;

enum class ColdSnowRimingAdmissionFailure
{
    InvalidResource,
    NonfiniteBudget,
    BudgetCap,
    InvalidEndpoint
};

class ColdSnowRimingAdmissionError : public std::runtime_error
{
public:
    ColdSnowRimingAdmissionError(ColdSnowRimingAdmissionFailure kind, const char* message)
        : std::runtime_error(message), kind_(kind)
    {
    }

    ColdSnowRimingAdmissionFailure kind() const noexcept
    {
        return kind_;
    }

private:
    ColdSnowRimingAdmissionFailure kind_;
};

struct ColdSnowRimingAdmissionObservation
{
    bool active = false;
    unsigned admission_passes = 0;
    double temperature_k = 0.0;
    double pressure_pa = 0.0;
    double dry_density_kg_m3 = 0.0;
    double duration_s = 0.0;
    std::array<double, 3> raw_cloud_mass_per_kg{};
    std::array<double, 3> raw_cloud_number_per_kg{};
    std::array<double, 3> pre_admission_cloud_mass_per_kg{};
    std::array<double, 3> pre_admission_cloud_number_per_kg{};
    double cloud_admission_factor = 1.0;
    bool cloud_factor_quantized = false;
    unsigned cloud_factor_backoffs = 0;
    std::array<double, 3> admitted_cloud_mass_per_kg{};
    std::array<double, 3> admitted_cloud_number_per_kg{};
    double raw_ambient_snow_mass_increment_per_kg = 0.0;
    double riming_snow_mass_increment_per_kg = 0.0;
    double raw_heated_snow_mass_increment_per_kg = 0.0;
    double admitted_snow_mass_increment_per_kg = 0.0;
    double admitted_snow_number_increment_per_kg = 0.0;
    bool snow_conversion_predicate = false;
    std::array<double, 7> donor_mass_source_per_kg{};
    std::array<double, 7> donor_mass_sink_per_kg{};
    double snow_number_source_per_kg = 0.0;
    double snow_number_sink_per_kg = 0.0;
    double final_snow_mass_per_kg = 0.0;
    double final_snow_number_per_kg = 0.0;
};

} // namespace tmv::milbrandt_detail
