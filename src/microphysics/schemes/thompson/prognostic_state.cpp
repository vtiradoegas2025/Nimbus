#include "prognostic_state.hpp"
#include "moment_initialization.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tmv::thompson
{
namespace
{

constexpr std::array<const char*, 5> currency_names = {
    "Nc", "Nr", "Ni", "Nwfa", "Nifa"};

bool same_shape(const Field3D& lhs, const Field3D& rhs)
{
    return lhs.size_r() == rhs.size_r()
        && lhs.size_th() == rhs.size_th()
        && lhs.size_z() == rhs.size_z();
}

bool same_profile(const AerosolProfile& lhs, const AerosolProfile& rhs)
{
    return lhs.water_surface_total_m3 == rhs.water_surface_total_m3
        && lhs.water_free_troposphere_m3 == rhs.water_free_troposphere_m3
        && lhs.ice_surface_total_m3 == rhs.ice_surface_total_m3
        && lhs.ice_free_troposphere_m3 == rhs.ice_free_troposphere_m3;
}

bool valid_profile(const AerosolProfile& profile)
{
    return std::isfinite(profile.water_surface_total_m3)
        && std::isfinite(profile.water_free_troposphere_m3)
        && std::isfinite(profile.ice_surface_total_m3)
        && std::isfinite(profile.ice_free_troposphere_m3)
        && profile.water_surface_total_m3 > profile.water_free_troposphere_m3
        && profile.water_free_troposphere_m3 > 0.0
        && profile.ice_surface_total_m3 > profile.ice_free_troposphere_m3
        && profile.ice_free_troposphere_m3 > 0.0;
}

double fixed_profile(double height_m, double surface_total, double background)
{
    constexpr double boundary_layer_shape = 0.8;
    const double amplitude = surface_total - background;
    const double decay = -std::log(background / amplitude)
                         / boundary_layer_shape;
    return background + amplitude * std::exp(-(height_m / 1000.0) * decay);
}

} // namespace

std::vector<double> flat_profile_sampling_heights(
    int nr,
    int nth,
    int nz,
    const std::vector<double>& layer_thickness_m)
{
    if (nr <= 0 || nth <= 0 || nz <= 0)
        throw std::invalid_argument("invalid Thompson flat-profile grid");
    const auto checked_product = [](std::size_t lhs, std::size_t rhs) {
        if (rhs != 0 && lhs > std::vector<double>().max_size() / rhs)
            throw std::length_error("Thompson flat-profile grid is too large");
        return lhs * rhs;
    };
    const std::size_t horizontal = checked_product(
        static_cast<std::size_t>(nr), static_cast<std::size_t>(nth));
    const std::size_t expected = checked_product(
        horizontal, static_cast<std::size_t>(nz));
    if (layer_thickness_m.size() != expected)
    {
        throw std::invalid_argument("invalid Thompson flat-profile grid");
    }
    std::vector<double> samples(expected);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            const std::size_t base =
                (static_cast<std::size_t>(i) * static_cast<std::size_t>(nth)
                 + static_cast<std::size_t>(j))
                * static_cast<std::size_t>(nz);
            for (int k = 0; k < nz; ++k)
                if (!std::isfinite(layer_thickness_m[base + k])
                    || layer_thickness_m[base + k] <= 0.0)
                {
                    throw std::invalid_argument(
                        "invalid Thompson flat-profile layer thickness");
                }
            if (nz == 1)
            {
                samples[base] = layer_thickness_m[base];
                continue;
            }
            const double first_center = 0.5 * layer_thickness_m[base];
            double lower_face = layer_thickness_m[base];
            for (int k = 1; k < nz; ++k)
            {
                const double center = lower_face
                    + 0.5 * layer_thickness_m[base + k];
                samples[base + k] = center - first_center;
                lower_face += layer_thickness_m[base + k];
            }
            samples[base] = samples[base + 1];
        }
    return samples;
}

void PrognosticState::prepare(const Field3D& rho_d,
                              const Field3D& qc,
                              const Field3D& qr,
                              const Field3D& qi,
                              const std::vector<double>& profile_sample_height_m,
                              AerosolProfile aerosol_profile)
{
    prepare(rho_d, qc, qr, qi, profile_sample_height_m, aerosol_profile,
            PhysicalCellDomain{0, rho_d.size_r(), 0, rho_d.size_th(),
                               0, rho_d.size_z()});
}

void PrognosticState::prepare(const Field3D& rho_d,
                              const Field3D& qc,
                              const Field3D& qr,
                              const Field3D& qi,
                              const std::vector<double>& profile_sample_height_m,
                              AerosolProfile aerosol_profile,
                              const PhysicalCellDomain& domain)
{
    if (!same_shape(rho_d, qc) || !same_shape(rho_d, qr) || !same_shape(rho_d, qi)
        || profile_sample_height_m.size() != rho_d.size())
    {
        throw std::invalid_argument("Thompson prognostic initialization shape mismatch");
    }
    if (domain.empty() || domain.i_begin < 0
        || domain.i_end > rho_d.size_r() || domain.j_begin < 0
        || domain.j_end > rho_d.size_th() || domain.k_begin < 0
        || domain.k_end > rho_d.size_z())
        throw std::invalid_argument(
            "invalid Thompson prognostic physical domain");
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
        const std::size_t index =
            (static_cast<std::size_t>(i) * rho_d.size_th()
             + static_cast<std::size_t>(j)) * rho_d.size_z()
            + static_cast<std::size_t>(k);
        if (!std::isfinite(rho_d.data()[index]) || rho_d.data()[index] <= 0.0f
            || !std::isfinite(qc.data()[index]) || qc.data()[index] < 0.0f
            || !std::isfinite(qr.data()[index]) || qr.data()[index] < 0.0f
            || !std::isfinite(qi.data()[index]) || qi.data()[index] < 0.0f
            || !std::isfinite(profile_sample_height_m[index])
            || profile_sample_height_m[index] < 0.0)
        {
            throw std::invalid_argument("invalid Thompson prognostic initialization state");
        }
            }
    if (!valid_profile(aerosol_profile))
        throw std::invalid_argument("invalid Thompson aerosol initialization profile");
    if (!moments_.front().empty() && aerosol_profile_bound_
        && !same_profile(aerosol_profile_, aerosol_profile))
    {
        throw std::invalid_argument(
            "Thompson aerosol profile cannot change after state initialization");
    }
    if (!moments_.front().empty() && same_shape(moments_.front(), rho_d))
    {
        std::string error;
        if (!validate_process_state(rho_d, domain, error))
            throw std::invalid_argument(error);
        aerosol_profile_ = aerosol_profile;
        aerosol_profile_bound_ = true;
        return;
    }

    PrognosticState candidate;
    candidate.aerosol_profile_ = aerosol_profile;
    candidate.aerosol_profile_bound_ = true;
    for (Field3D& field : candidate.moments_)
        field.resize(rho_d.size_r(), rho_d.size_th(), rho_d.size_z());
    for (Field3D& field : candidate.tendencies_)
        field.resize(rho_d.size_r(), rho_d.size_th(), rho_d.size_z());

    std::atomic<bool> initialization_failed{false};
    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
        if (initialization_failed.load(std::memory_order_relaxed)) continue;
        try
        {
        const std::size_t index =
            (static_cast<std::size_t>(i) * rho_d.size_th()
             + static_cast<std::size_t>(j)) * rho_d.size_z()
            + static_cast<std::size_t>(k);
        const double inverse_density = 1.0 / static_cast<double>(rho_d.data()[index]);
        const auto initialized = initialize_number_moments(
            qc.data()[index], qr.data()[index], qi.data()[index],
            rho_d.data()[index]);
        candidate.moments_[0].data()[index] =
            static_cast<float>(initialized.cloud_kg_dry);
        candidate.moments_[1].data()[index] =
            static_cast<float>(initialized.rain_kg_dry);
        candidate.moments_[2].data()[index] =
            static_cast<float>(initialized.ice_kg_dry);
        candidate.moments_[3].data()[index] = static_cast<float>(
            fixed_profile(profile_sample_height_m[index],
                          aerosol_profile.water_surface_total_m3,
                          aerosol_profile.water_free_troposphere_m3)
            * inverse_density);
        candidate.moments_[4].data()[index] = static_cast<float>(
            fixed_profile(profile_sample_height_m[index],
                          aerosol_profile.ice_surface_total_m3,
                          aerosol_profile.ice_free_troposphere_m3)
            * inverse_density);
        }
        catch (...)
        {
            initialization_failed.store(true, std::memory_order_relaxed);
        }
            }
    if (initialization_failed.load(std::memory_order_relaxed))
        throw std::runtime_error("Thompson prognostic initialization failed");
    for (const Field3D& field : candidate.moments_)
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            const float value = field.data()[index];
            if (!std::isfinite(value) || value < 0.0f)
                throw std::overflow_error(
                    "Thompson prognostic initialization is not representable");
        }
    candidate.invalidate_transient();
    *this = std::move(candidate);
}

MicrophysicsPrognosticScalarList PrognosticState::transport_bindings()
{
    using Storage = MicrophysicsScalarStorage;
    return {{"thompson.Nc_kg-1", &moments_[0], true, Storage::DryAirMixingRatio, MicrophysicsMassSpecies::Cloud},
            {"thompson.Nr_kg-1", &moments_[1], true, Storage::DryAirMixingRatio, MicrophysicsMassSpecies::Rain},
            {"thompson.Ni_kg-1", &moments_[2], true, Storage::DryAirMixingRatio, MicrophysicsMassSpecies::Ice},
            {"thompson.Nwfa_kg-1", &moments_[3], true, Storage::DryAirMixingRatio},
            {"thompson.Nifa_kg-1", &moments_[4], true, Storage::DryAirMixingRatio}};
}

bool PrognosticState::validate_process_state(const Field3D& reference,
                                             std::string& error) const
{
    return validate_process_state(
        reference,
        PhysicalCellDomain{0, reference.size_r(), 0, reference.size_th(),
                           0, reference.size_z()},
        error);
}

bool PrognosticState::validate_process_state(
    const Field3D& reference, const PhysicalCellDomain& domain,
    std::string& error) const
{
    if (reference.empty() || moments_.front().empty())
    {
        error = "Thompson prognostic process state is not initialized";
        return false;
    }
    for (std::size_t currency = 0; currency < moments_.size(); ++currency)
    {
        if (!same_shape(moments_[currency], reference)
            || !same_shape(tendencies_[currency], reference))
        {
            error = std::string("Thompson prognostic process shape mismatch for ")
                + currency_names[currency];
            return false;
        }
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
            const std::size_t index =
                (static_cast<std::size_t>(i) * reference.size_th()
                 + static_cast<std::size_t>(j)) * reference.size_z()
                + static_cast<std::size_t>(k);
            const float moment = moments_[currency].data()[index];
            const float tendency = tendencies_[currency].data()[index];
            if (!std::isfinite(moment) || moment < 0.0f
                || !std::isfinite(tendency))
            {
                error = std::string("invalid Thompson prognostic process value for ")
                    + currency_names[currency];
                return false;
            }
                }
    }
    error.clear();
    return true;
}

bool PrognosticState::validate_update(double dt, std::string& error) const
{
    const Field3D& reference = moments_.front();
    return validate_update(
        dt,
        PhysicalCellDomain{0, reference.size_r(), 0, reference.size_th(),
                           0, reference.size_z()},
        error);
}

bool PrognosticState::validate_update(
    double dt, const PhysicalCellDomain& domain, std::string& error) const
{
    if (!std::isfinite(dt) || dt <= 0.0 || moments_.front().empty())
    {
        error = "Thompson prognostic timestep/state is invalid";
        return false;
    }
    for (std::size_t currency = 0; currency < moments_.size(); ++currency)
    {
        if (!same_shape(moments_[currency], moments_.front())
            || !same_shape(tendencies_[currency], moments_.front()))
        {
            error = "Thompson prognostic currency shape mismatch";
            return false;
        }
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
            const std::size_t index =
                (static_cast<std::size_t>(i) * moments_[currency].size_th()
                 + static_cast<std::size_t>(j))
                    * moments_[currency].size_z()
                + static_cast<std::size_t>(k);
            const double old_value = moments_[currency].data()[index];
            const double tendency = tendencies_[currency].data()[index];
            const double candidate = old_value + dt * tendency;
            const float stored_candidate = static_cast<float>(candidate);
            if (!std::isfinite(old_value) || old_value < 0.0
                || !std::isfinite(tendency) || !std::isfinite(candidate)
                || candidate < 0.0 || !std::isfinite(stored_candidate))
            {
                error = std::string("invalid Thompson prognostic update for ")
                        + currency_names[currency];
                return false;
            }
                }
    }
    error.clear();
    return true;
}

bool PrognosticState::apply_column_multipliers(double                    dt,
                                               const Field3D&            density,
                                               const PhysicalCellDomain& domain,
                                               std::vector<float>&       column_multipliers,
                                               std::string&              error)
{
    const int nr = density.size_r(), nth = density.size_th(), nz = density.size_z();
    if (!std::isfinite(dt) || dt <= 0 || density.empty() || domain.i_begin < 0 || domain.i_begin > domain.i_end ||
        domain.i_end > nr || domain.j_begin < 0 || domain.j_begin > domain.j_end || domain.j_end > nth ||
        domain.k_begin < 0 || domain.k_begin > domain.k_end || domain.k_end > nz ||
        column_multipliers.size() != static_cast<std::size_t>(nr) * nth)
    {
        error = "invalid Thompson stochastic column timestep or layout";
        return false;
    }
    for (float multiplier : column_multipliers)
        if (!std::isfinite(multiplier) || multiplier < 0)
        {
            error = "invalid Thompson stochastic column multiplier";
            return false;
        }
    if (!validate_process_state(density, domain, error))
        return false;
    for (const auto& tendency : tendencies_)
        if (density.data() == tendency.data())
        {
            error = "Thompson stochastic density aliases a mutable tendency";
            return false;
        }

    constexpr std::array<double, 5> maximum_volume_number{1999e6, 1e12, 1e12, 9999e6, 9999e6};
    auto                            admitted = column_multipliers;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double rho = density(i, j, k);
                if (!std::isfinite(rho) || rho <= 0)
                {
                    error = "invalid Thompson stochastic density";
                    return false;
                }
                for (std::size_t field = 0; field < moments_.size(); ++field)
                    if (rho * moments_[field](i, j, k) > maximum_volume_number[field])
                    {
                        error = "Thompson stochastic initial population exceeds process bounds";
                        return false;
                    }
            }
            const auto valid_factor = [&](float factor)
            {
                for (std::size_t field = 0; field < moments_.size(); ++field)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const float  rate      = tendencies_[field](i, j, k) * factor;
                        const double candidate = double(moments_[field](i, j, k)) + dt * double(rate);
                        if (!std::isfinite(rate) || !std::isfinite(candidate) || candidate < 0 ||
                            candidate > std::numeric_limits<float>::max() ||
                            double(float(candidate)) * density(i, j, k) > maximum_volume_number[field])
                            return false;
                    }
                return true;
            };
            float& factor = admitted[static_cast<std::size_t>(i) * nth + j];
            if (factor == 0)
                factor = 0.0f;
            if (valid_factor(factor))
                continue;

            // Positive IEEE float bits preserve order, bounding even subnormal factor searches to 31 probes.
            std::uint32_t lower = 0, upper;
            static_assert(sizeof(upper) == sizeof(factor) && std::numeric_limits<float>::is_iec559 &&
                          std::numeric_limits<float>::digits == 24 && std::numeric_limits<float>::max_exponent == 128);
            std::memcpy(&upper, &factor, sizeof(upper));
            while (upper - lower > 1)
            {
                const std::uint32_t middle = lower + (upper - lower) / 2;
                float               candidate;
                std::memcpy(&candidate, &middle, sizeof(candidate));
                if (valid_factor(candidate))
                    lower = middle;
                else
                    upper = middle;
            }
            std::memcpy(&factor, &lower, sizeof(factor));
        }

    for (auto& tendency : tendencies_)
    {
#pragma omp parallel for collapse(2)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    tendency(i, j, k) *= admitted[static_cast<std::size_t>(i) * nth + j];
    }
    std::copy(admitted.begin(), admitted.end(), column_multipliers.begin());
    error.clear();
    return true;
}

bool PrognosticState::bind_aerosol_profile(AerosolProfile aerosol_profile, std::string& error)
{
    if (!valid_profile(aerosol_profile))
    {
        error = "invalid Thompson aerosol profile identity";
        return false;
    }
    if (aerosol_profile_bound_ && !same_profile(aerosol_profile_, aerosol_profile))
    {
        error = "Thompson aerosol profile identity mismatch";
        return false;
    }
    aerosol_profile_ = aerosol_profile;
    aerosol_profile_bound_ = true;
    error.clear();
    return true;
}

void PrognosticState::commit_update(double dt) noexcept
{
    const Field3D& reference = moments_.front();
    commit_update(
        dt, PhysicalCellDomain{0, reference.size_r(), 0,
                               reference.size_th(), 0, reference.size_z()});
}

void PrognosticState::commit_update(
    double dt, const PhysicalCellDomain& domain) noexcept
{
    if (moments_.front().empty()) return;
    for (std::size_t currency = 0; currency < moments_.size(); ++currency)
    {
        #pragma omp parallel for collapse(2)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double candidate = static_cast<double>(
                        moments_[currency](i, j, k)) + dt * static_cast<double>(
                            tendencies_[currency](i, j, k));
                    moments_[currency](i, j, k) = static_cast<float>(candidate);
                }
    }
}

void PrognosticState::invalidate_transient() noexcept
{
    for (Field3D& field : tendencies_)
        if (!field.empty()) field.fill(0.0f);
}

std::vector<Field3D> PrognosticState::portable_fields() const
{
    return {moments_[0], moments_[1], moments_[2], moments_[3], moments_[4],
            tendencies_[0], tendencies_[1], tendencies_[2], tendencies_[3], tendencies_[4]};
}

bool PrognosticState::restore_fields(const std::vector<Field3D>& fields,
                                     std::string& error)
{
    if (fields.size() != moments_.size() + tendencies_.size())
    {
        error = "invalid Thompson prognostic portable-state field count";
        return false;
    }
    const bool initialized = !fields.front().empty();
    for (std::size_t currency = 0; currency < moments_.size(); ++currency)
    {
        const Field3D& moment = fields[currency];
        const Field3D& tendency = fields[moments_.size() + currency];
        if (moment.empty() == initialized || tendency.empty() == initialized
            || (initialized && (!same_shape(moment, fields.front())
                                || !same_shape(tendency, fields.front()))))
        {
            error = "inconsistent Thompson prognostic portable-state fields";
            return false;
        }
        for (std::size_t index = 0; index < moment.size(); ++index)
        {
            if (!std::isfinite(moment.data()[index]) || moment.data()[index] < 0.0f
                || !std::isfinite(tendency.data()[index]))
            {
                error = "invalid Thompson prognostic portable-state value";
                return false;
            }
        }
    }
    PrognosticState candidate;
    for (std::size_t currency = 0; currency < moments_.size(); ++currency)
    {
        candidate.moments_[currency] = fields[currency];
        candidate.tendencies_[currency] = fields[moments_.size() + currency];
    }
    *this = std::move(candidate);
    error.clear();
    return true;
}

} // namespace tmv::thompson
