#pragma once

#include "core/field/field_bounds.hpp"
#include "dynamics/eos_pressure.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace tmv::milbrandt_detail
{

struct SourceEosEnergyState
{
    float density = 0.0f;
    float pressure = 0.0f;
    float theta = 0.0f;
    std::array<float, 7> water{};
};

struct SourceEosEnergyInput
{
    SourceEosEnergyState before;
    std::array<double, 7> accepted_water{};
    std::uint64_t generation = 0;
    double duration_s = 0.0;
};

struct SourceEosEnergyReceipt
{
    double theta_increment = 0.0;
    double temperature_before = 0.0;
    double temperature_eos_reference = 0.0;
    double temperature_eos_target = 0.0;
    double temperature_eos_stored = 0.0;
    double reference_projection_work = 0.0;
    double ideal_phase_energy_change = 0.0;
    double water_rounding_work = 0.0;
    double theta_pressure_rounding_work = 0.0;
    double committed_internal_energy_change = 0.0;
    double stored_water_increment = 0.0;
};

namespace source_eos_energy_detail
{

template <class T>
double capacity(const std::array<T, 7>& q)
{
    using namespace physical_constants;
    double result = specific_heat_cp_jkgk - gas_constant_dry_air_jkgk +
        q[0] * (specific_heat_water_vapor_cp_jkgk - gas_constant_water_vapor_jkgk);
    result += (static_cast<double>(q[1]) + q[2]) * specific_heat_liquid_water_jkgk;
    for (std::size_t index = 3; index < q.size(); ++index)
        result += q[index] * specific_heat_ice_jkgk;
    return result;
}

template <class A, class B>
double composition_work(const std::array<A, 7>& before, const std::array<B, 7>& after, double temperature)
{
    using namespace physical_constants;
    const double tc = temperature - freezing_temperature_k;
    double result = (static_cast<double>(after[0]) - before[0]) *
        ((specific_heat_water_vapor_cp_jkgk - gas_constant_water_vapor_jkgk) * tc +
         latent_heat_vaporization_jkg - gas_constant_water_vapor_jkgk * freezing_temperature_k);
    for (std::size_t index = 1; index < before.size(); ++index)
        result += (static_cast<double>(after[index]) - before[index]) *
            (index < 3 ? specific_heat_liquid_water_jkgk * tc
                       : specific_heat_ice_jkgk * tc - latent_heat_fusion_jkg);
    return result;
}

inline bool same(const SourceEosEnergyState& a, const SourceEosEnergyState& b)
{
    return std::memcmp(&a.density, &b.density, sizeof(float)) == 0 &&
        std::memcmp(&a.pressure, &b.pressure, sizeof(float)) == 0 &&
        std::memcmp(&a.theta, &b.theta, sizeof(float)) == 0 &&
        std::memcmp(a.water.data(), b.water.data(), a.water.size() * sizeof(float)) == 0;
}

inline double temperature(double theta, double pressure)
{
    return theta * std::pow(pressure / physical_constants::reference_pressure_pa,
        physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk);
}

inline bool bounded(double value, double lo, double hi)
{
    return std::isfinite(value) && value >= lo && value <= hi;
}

} // namespace source_eos_energy_detail

class SourceEosEnergyTransaction
{
public:
    enum class Stage { Empty, Prepared, SourceCommitted, EosCommitted };

    Stage stage() const noexcept { return stage_; }
    SourceEosEnergyInput input() const noexcept { return input_; }
    SourceEosEnergyState source_endpoint() const noexcept { return source_endpoint_; }
    SourceEosEnergyState eos_endpoint() const noexcept { return eos_endpoint_; }
    SourceEosEnergyReceipt receipt() const noexcept { return receipt_; }

    bool prepare(const SourceEosEnergyInput& input, std::string& error)
    {
        using namespace source_eos_energy_detail;
        const auto reject = [&](const char* message) { error = message; return false; };
        if (stage_ != Stage::Empty)
            return reject("source/EOS energy transaction already owns a sequence");
        const auto& before = input.before;
        if (input.generation == 0 || !std::isfinite(input.duration_s) || input.duration_s <= 0.0 ||
            !bounded(before.density, density_min_kgm3, density_max_kgm3) ||
            !bounded(before.pressure, pressure_min_pa, pressure_max_pa) ||
            !bounded(before.theta, theta_min_k, theta_max_k))
            return reject("source/EOS energy input or generation is invalid");
        SourceEosEnergyTransaction candidate;
        candidate.input_ = input;
        candidate.source_endpoint_ = before;
        double water_change = 0.0, water_scale = 0.0;
        for (std::size_t index = 0; index < before.water.size(); ++index)
        {
            const double limit = index == 0 ? qv_max_kgkg : hydrometeor_max_kgkg;
            const double q = input.accepted_water[index];
            if (!bounded(before.water[index], 0.0, limit) || !bounded(q, 0.0, limit) ||
                (q != 0.0 && q < std::numeric_limits<float>::denorm_min()))
                return reject("source/EOS energy water is invalid or not representable");
            candidate.source_endpoint_.water[index] = static_cast<float>(q);
            candidate.receipt_.stored_water_increment +=
                static_cast<double>(candidate.source_endpoint_.water[index]) - before.water[index];
            water_change += q - before.water[index];
            water_scale += before.water[index];
        }
        if (std::abs(water_change) > 64.0 * std::numeric_limits<double>::epsilon() * water_scale)
            return reject("source/EOS energy receipt requires a closed water transfer");
        auto& receipt = candidate.receipt_;
        receipt.temperature_before = temperature(before.theta, before.pressure);
        const double reference_pressure = dynamics_eos::pressure_fp64(before.density, before.theta);
        if (!bounded(reference_pressure, pressure_min_pa, pressure_max_pa))
            return reject("source/EOS reference pressure would require repair");
        receipt.temperature_eos_reference = temperature(before.theta, reference_pressure);
        const double source_capacity = capacity(before.water);
        const double accepted_capacity = capacity(input.accepted_water);
        const double phase_work = composition_work(before.water, input.accepted_water, receipt.temperature_eos_reference);
        const double temperature_change = -phase_work / accepted_capacity;
        receipt.temperature_eos_target = receipt.temperature_eos_reference + temperature_change;
        if (!bounded(receipt.temperature_before, 150.0, 400.0) ||
            !bounded(receipt.temperature_eos_reference, 150.0, 400.0) ||
            !bounded(receipt.temperature_eos_target, 150.0, 400.0))
            return reject("source/EOS energy temperature leaves the bounded domain");
        // Invert the subsequent dry EOS, rather than adding a local Cp or Cv temperature correction.
        receipt.theta_increment = before.theta * std::expm1(std::log1p(temperature_change /
            receipt.temperature_eos_reference) / physical_constants::dry_air_gamma);
        const double theta_unrounded = static_cast<double>(before.theta) + receipt.theta_increment;
        const double theta_rate = receipt.theta_increment / input.duration_s;
        if (!bounded(theta_unrounded, theta_min_k, theta_max_k) ||
            !std::isfinite(theta_rate) || std::abs(theta_rate) > std::numeric_limits<float>::max() ||
            (receipt.theta_increment != 0.0 && static_cast<float>(theta_rate) == 0.0f))
            return reject("source/EOS energy theta source is unrepresentable");
        candidate.source_endpoint_.theta = static_cast<float>(theta_unrounded);
        if (!bounded(temperature(theta_unrounded, before.pressure), 150.0, 400.0) ||
            !bounded(temperature(candidate.source_endpoint_.theta, before.pressure), 150.0, 400.0))
            return reject("source/EOS staged temperature leaves the bounded domain");
        candidate.eos_endpoint_ = candidate.source_endpoint_;
        const double eos_pressure = dynamics_eos::pressure_fp64(before.density, candidate.eos_endpoint_.theta);
        if (!bounded(eos_pressure, pressure_min_pa, pressure_max_pa))
            return reject("source/EOS endpoint pressure would require repair");
        candidate.eos_endpoint_.pressure = static_cast<float>(eos_pressure);
        receipt.temperature_eos_stored = temperature(candidate.eos_endpoint_.theta, candidate.eos_endpoint_.pressure);
        if (!bounded(receipt.temperature_eos_stored, 150.0, 400.0))
            return reject("source/EOS stored temperature leaves the bounded domain");
        receipt.reference_projection_work = source_capacity *
            (receipt.temperature_eos_reference - receipt.temperature_before);
        receipt.ideal_phase_energy_change = accepted_capacity *
            (receipt.temperature_eos_target - receipt.temperature_eos_reference) + phase_work;
        receipt.water_rounding_work = composition_work(input.accepted_water,
            candidate.source_endpoint_.water, receipt.temperature_eos_target);
        receipt.theta_pressure_rounding_work = capacity(candidate.source_endpoint_.water) *
            (receipt.temperature_eos_stored - receipt.temperature_eos_target);
        receipt.committed_internal_energy_change = receipt.reference_projection_work + receipt.ideal_phase_energy_change +
            receipt.water_rounding_work + receipt.theta_pressure_rounding_work;
        for (double value : {receipt.reference_projection_work, receipt.ideal_phase_energy_change,
                            receipt.water_rounding_work, receipt.theta_pressure_rounding_work,
                            receipt.committed_internal_energy_change, receipt.stored_water_increment})
            if (!std::isfinite(value)) return reject("source/EOS energy receipt is unrepresentable");
        candidate.stage_ = Stage::Prepared;
        *this = candidate;
        error.clear();
        return true;
    }

    bool accept_source(const SourceEosEnergyState& before, const SourceEosEnergyState& after,
                       std::uint64_t generation, std::string& error)
    {
        if (stage_ != Stage::Prepared || generation != input_.generation ||
            !source_eos_energy_detail::same(before, input_.before) ||
            !source_eos_energy_detail::same(after, source_endpoint_))
        {
            error = "source/EOS energy sequence has a foreign source commit";
            return false;
        }
        stage_ = Stage::SourceCommitted;
        error.clear();
        return true;
    }

    bool accept_eos(const SourceEosEnergyState& before, const SourceEosEnergyState& after,
                    std::uint64_t generation, std::string& error)
    {
        if (stage_ != Stage::SourceCommitted || generation != input_.generation ||
            !source_eos_energy_detail::same(before, source_endpoint_) ||
            !source_eos_energy_detail::same(after, eos_endpoint_))
        {
            error = "source/EOS energy sequence changed before its pressure projection";
            return false;
        }
        stage_ = Stage::EosCommitted;
        error.clear();
        return true;
    }

private:
    Stage stage_ = Stage::Empty;
    SourceEosEnergyInput input_{};
    SourceEosEnergyState source_endpoint_{}, eos_endpoint_{};
    SourceEosEnergyReceipt receipt_{};
};

} // namespace tmv::milbrandt_detail
