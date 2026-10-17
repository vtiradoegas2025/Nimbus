#include "triple_graupel_hail_conversion.hpp"

#include "cold.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tmv::milbrandt_detail
{
namespace
{

constexpr std::array<TripleMomentSpecies, 4> species{
    TripleMomentSpecies::Ice, TripleMomentSpecies::Snow,
    TripleMomentSpecies::Graupel, TripleMomentSpecies::Hail};

bool reject(GraupelHailConversionFailure kind, const char* message,
            std::string& error, GraupelHailConversionFailure* failure)
{
    error = message;
    if (failure)
    {
        *failure = kind;
    }
    return false;
}

bool same(double a, double b) noexcept
{
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

template <std::size_t Count>
bool same(const std::array<double, Count>& a, const std::array<double, Count>& b) noexcept
{
    for (std::size_t index = 0; index < Count; ++index)
    {
        if (!same(a[index], b[index]))
        {
            return false;
        }
    }
    return true;
}

std::array<double, 25> values(const ColdParameters& p) noexcept
{
    return {p.a_g_, p.a_h_, p.a_i_, p.a_r_, p.a_s_, p.alpha_g_, p.alpha_h_, p.alpha_i_,
        p.alpha_r_, p.alpha_s_, p.b_g_, p.b_h_, p.b_i_, p.b_r_, p.b_s_, p.c_g_, p.c_h_,
        p.c_i_, p.c_r_, p.c_s_, p.d_g_, p.d_h_, p.d_i_, p.d_r_, p.d_s_};
}

std::array<double, 27> values(const GraupelHailConversionCell& cell) noexcept
{
    const auto& source = cell.source;
    const auto& initial = source.initial;
    std::array<double, 27> result{source.temperature_k, source.pressure_pa,
        source.dry_density_kg_m3, initial.vapor_mass_per_kg, initial.cloud_mass_per_kg,
        initial.cloud_number_per_kg};
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        result[6 + 3 * index] = initial.frozen[index].mass_per_kg;
        result[7 + 3 * index] = initial.frozen[index].number_per_kg;
        result[8 + 3 * index] = initial.frozen[index].reflectivity_per_kg;
    }
    result[18] = cell.rain_mass_per_kg;
    std::copy(cell.admitted_accretion_mass_per_kg.begin(), cell.admitted_accretion_mass_per_kg.end(), result.begin() + 19);
    result[22] = cell.graupel_diffusion_mass_per_kg;
    result[23] = cell.graupel_diffusion_number_per_kg;
    result[24] = initial.rain.mass_per_kg;
    result[25] = initial.rain.number_per_kg;
    result[26] = initial.rain.reflectivity_per_kg;
    return result;
}

bool population(const FrozenCollectionPopulation& source, TripleMomentSpecies category,
                ColdSpectrumDomain domain, double& alpha, double& lambda, std::string& error)
{
    for (double value : {source.mass_per_kg, source.number_per_kg, source.reflectivity_per_kg})
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            error = "SLL population has nonfinite or negative moments";
            return false;
        }
    }
    alpha = 0.0;
    lambda = 0.0;
    if (source.mass_per_kg == 0.0 && source.number_per_kg == 0.0 && source.reflectivity_per_kg == 0.0)
    {
        return true;
    }
    const auto law = current_nimbus_triple_sedimentation_parameters(category).mass_law;
    if (!diagnose_triple_gamma_shape(source.mass_per_kg, source.number_per_kg,
            source.reflectivity_per_kg, law, alpha, lambda, error))
    {
        return false;
    }
    if (law.exponent == 3.0)
    {
        int qe = 0, ne = 0, ze = 0, ce = 0;
        const double q = std::frexp(source.mass_per_kg, &qe);
        const double n = std::frexp(source.number_per_kg, &ne);
        const double z = std::frexp(source.reflectivity_per_kg, &ze);
        const double c = std::frexp(law.coefficient, &ce);
        const double ratio = std::ldexp(z * n * (c / q) * (c / q), ze + ne + 2 * (ce - qe));
        const double target = std::log(ratio);
        const auto log_ratio = [](double shape)
        {
            return std::log1p(3.0 / (shape + 1.0)) + std::log1p(3.0 / (shape + 2.0)) +
                std::log1p(3.0 / (shape + 3.0));
        };
        double lower = -1.0 + 1e-9;
        double upper = 1e6;
        if (!std::isfinite(target) || target <= log_ratio(upper) || target >= log_ratio(lower))
        {
            error = "SLL cubic population is outside its finite shape bracket";
            return false;
        }
        for (int iteration = 0; iteration < 80; ++iteration)
        {
            const double middle = .5 * (lower + upper);
            if (log_ratio(middle) > target)
            {
                lower = middle;
            }
            else
            {
                upper = middle;
            }
        }
        alpha = .5 * (lower + upper);
        lambda = std::exp((std::log(law.coefficient) + std::log(source.number_per_kg) -
            std::log(source.mass_per_kg) + std::log(alpha + 1.0) + std::log(alpha + 2.0) +
            std::log(alpha + 3.0)) / 3.0);
    }
    if (!std::isfinite(alpha) || alpha > cold_spectrum_shape_limit(domain, category) ||
        !std::isfinite(lambda) || lambda <= 0.0)
    {
        error = "SLL population exceeds its selected spectrum domain or has an invalid slope";
        return false;
    }
    return true;
}

bool stored_population(const FrozenCollectionPopulation& source, TripleMomentSpecies category,
                       ColdSpectrumDomain domain, FrozenCollectionEndpoint& stored, std::string& error)
{
    const FrozenCollectionEndpoint candidate{static_cast<float>(source.mass_per_kg),
        static_cast<float>(source.number_per_kg), static_cast<float>(source.reflectivity_per_kg)};
    if (source.mass_per_kg > 0.0 &&
        (!(candidate.mass_per_kg > 0.0f) || !(candidate.number_per_kg > 0.0f) || !(candidate.reflectivity_per_kg > 0.0f)))
    {
        error = "SLL nonempty population cannot be stored without losing a moment";
        return false;
    }
    double alpha = 0.0, lambda = 0.0;
    if (!population({candidate.mass_per_kg, candidate.number_per_kg, candidate.reflectivity_per_kg},
            category, domain, alpha, lambda, error))
    {
        return false;
    }
    stored = candidate;
    return true;
}

bool same_capture(const GraupelHailConversionProducer& a, const GraupelHailConversionProducer& b) noexcept
{
    return a.event_count() == b.event_count() && same(values(a.source()), values(b.source())) &&
        same(a.parameters(), b.parameters()) && a.hail_processes() == b.hail_processes() &&
        a.spectrum_domain() == b.spectrum_domain() && a.regime() == b.regime() &&
        same(std::array<double, 11>{a.duration_s(), a.loading_denominator(), a.source_alpha(),
                a.source_lambda_per_m(), a.mean_mass_diameter_m(), a.hail_embryo_diameter_m(),
                a.size_ratio(), a.raw_mass_transfer_per_kg(), a.admitted_mass_transfer_per_kg(),
                a.admitted_fraction(), a.donor_number_loss_per_kg()},
            std::array<double, 11>{b.duration_s(), b.loading_denominator(), b.source_alpha(),
                b.source_lambda_per_m(), b.mean_mass_diameter_m(), b.hail_embryo_diameter_m(),
                b.size_ratio(), b.raw_mass_transfer_per_kg(), b.admitted_mass_transfer_per_kg(),
                b.admitted_fraction(), b.donor_number_loss_per_kg()}) &&
        same(a.mass_increment_per_kg(), b.mass_increment_per_kg()) &&
        same(a.number_increment_per_kg(), b.number_increment_per_kg()) &&
        same(a.reflectivity_increment_per_kg(), b.reflectivity_increment_per_kg());
}

bool phase(const GraupelHailConversionProducer& producer, GraupelHailConversionResult& result,
           std::string& error, GraupelHailConversionFailure* failure)
{
    GraupelHailConversionResult candidate;
    candidate.unrounded = producer.source().source.initial.frozen;
    candidate.mass_increment_per_kg = producer.mass_increment_per_kg();
    candidate.number_increment_per_kg = producer.number_increment_per_kg();
    candidate.reflectivity_increment_per_kg = producer.reflectivity_increment_per_kg();
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        auto& endpoint = candidate.unrounded[index];
        endpoint.mass_per_kg += candidate.mass_increment_per_kg[index];
        endpoint.number_per_kg += candidate.number_increment_per_kg[index];
        endpoint.reflectivity_per_kg += candidate.reflectivity_increment_per_kg[index];
        double alpha = 0.0, lambda = 0.0;
        if (!population(endpoint, species[index], producer.spectrum_domain(), alpha, lambda, error) ||
            !stored_population(endpoint, species[index], producer.spectrum_domain(), candidate.rounded[index], error))
        {
            if (failure)
            {
                *failure = GraupelHailConversionFailure::IntervalRejected;
            }
            return false;
        }
    }
    candidate.producer = producer;
    result = candidate;
    error.clear();
    if (failure)
    {
        *failure = GraupelHailConversionFailure::None;
    }
    return true;
}

} // namespace

bool produce_exact_graupel_hail_conversion_admission(
    const ColdParameters& parameters, const GraupelHailConversionCell& cell, double duration_s,
    GraupelHailConversionProducer& producer, std::string& error, std::size_t capacity,
    GraupelHailConversionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    if (failure)
    {
        *failure = GraupelHailConversionFailure::None;
    }
    if (capacity != 1)
    {
        return reject(GraupelHailConversionFailure::Capacity,
            "SLL conversion requires exactly one channel record", error, failure);
    }
    const auto& source = cell.source;
    if (!valid_cold_spectrum_domain(spectrum_domain) || !same(values(parameters), default_cold_parameters) ||
        !std::isfinite(duration_s) || duration_s <= 0.0 || !std::isfinite(source.temperature_k) ||
        source.temperature_k < 150.0 || source.temperature_k >= microphysics_constants::T0 ||
        !std::isfinite(source.pressure_pa) || source.pressure_pa < 100.0 || source.pressure_pa > 200000.0 ||
        !std::isfinite(source.dry_density_kg_m3) || source.dry_density_kg_m3 <= 0.0)
    {
        return reject(GraupelHailConversionFailure::InvalidContext,
            "SLL conversion source context, retained parameters or spectrum domain is invalid", error, failure);
    }
    for (double value : {source.initial.vapor_mass_per_kg, source.initial.cloud_mass_per_kg,
                         source.initial.cloud_number_per_kg, cell.rain_mass_per_kg})
    {
        if (!std::isfinite(value) || value < 0.0 || !std::isfinite(static_cast<float>(value)))
        {
            return reject(GraupelHailConversionFailure::InvalidSource,
                "SLL conversion liquid, vapor or cloud-number source is invalid", error, failure);
        }
    }
    if (source.initial.cloud_mass_per_kg > 0.0 && source.initial.cloud_number_per_kg <= 0.0)
    {
        return reject(GraupelHailConversionFailure::InvalidSource,
            "SLL conversion cloud mass has no cloud-number source", error, failure);
    }
    GraupelHailConversionProducer candidate;
    candidate.source_ = cell;
    candidate.parameters_ = values(parameters);
    candidate.hail_processes_ = parameters.hail_processes;
    candidate.spectrum_domain_ = spectrum_domain;
    candidate.duration_s_ = duration_s;
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        double alpha = 0.0, lambda = 0.0;
        FrozenCollectionEndpoint stored;
        if (!population(source.initial.frozen[index], species[index], spectrum_domain, alpha, lambda, error) ||
            !stored_population(source.initial.frozen[index], species[index], spectrum_domain, stored, error))
        {
            if (failure)
            {
                *failure = GraupelHailConversionFailure::InvalidSource;
            }
            return false;
        }
        if (index == 2)
        {
            candidate.source_alpha_ = alpha;
            candidate.source_lambda_per_m_ = lambda;
        }
    }
    const std::array<double, 3> donors{source.initial.cloud_mass_per_kg, cell.rain_mass_per_kg,
        source.initial.frozen[0].mass_per_kg};
    double accreted = 0.0;
    for (std::size_t index = 0; index < donors.size(); ++index)
    {
        const double amount = cell.admitted_accretion_mass_per_kg[index];
        if (!std::isfinite(amount) || amount < 0.0 || amount > donors[index])
        {
            return reject(GraupelHailConversionFailure::InvalidSource,
                "SLL conversion accretion amount is invalid or exceeds its bound source donor", error, failure);
        }
        accreted += amount;
    }
    const auto& graupel = source.initial.frozen[2];
    const double diffusion = cell.graupel_diffusion_mass_per_kg;
    const double diffusion_number = cell.graupel_diffusion_number_per_kg;
    const double expected_number = graupel.mass_per_kg > 0.0
        ? graupel.number_per_kg * (std::min(diffusion, 0.0) / graupel.mass_per_kg) : 0.0;
    if (!std::isfinite(accreted) || !std::isfinite(diffusion) || !std::isfinite(diffusion_number) ||
        !std::isfinite(expected_number) || diffusion_number != expected_number ||
        (graupel.mass_per_kg == 0.0 && (accreted != 0.0 || diffusion != 0.0)))
    {
        return reject(GraupelHailConversionFailure::InvalidSource,
            "SLL conversion graupel accretion or proportional diffusion record is invalid", error, failure);
    }
    candidate.loading_denominator_ = 1.1e4 * (source.initial.cloud_mass_per_kg + cell.rain_mass_per_kg)
        - 1.3e3 * source.initial.frozen[0].mass_per_kg + .001;
    if (!std::isfinite(candidate.loading_denominator_))
    {
        return reject(GraupelHailConversionFailure::InvalidSource,
            "SLL conversion source loading is unrepresentable", error, failure);
    }
    if (graupel.mass_per_kg > 0.0)
    {
        candidate.mean_mass_diameter_m_ = std::exp((std::log(graupel.mass_per_kg)
            - std::log(parameters.c_g_) - std::log(graupel.number_per_kg)) / 3.0);
    }
    if (!parameters.hail_processes)
    {
        candidate.regime_ = GraupelHailSllRegime::HailDisabled;
    }
    else if (graupel.mass_per_kg == 0.0)
    {
        candidate.regime_ = GraupelHailSllRegime::NoGraupel;
    }
    else if (accreted == 0.0)
    {
        candidate.regime_ = GraupelHailSllRegime::NoAccretion;
    }
    else if (candidate.loading_denominator_ <= 0.0)
    {
        candidate.regime_ = GraupelHailSllRegime::NonpositiveLoading;
    }
    else
    {
        const double cooling = microphysics_constants::T0 - source.temperature_k;
        const double gate = std::log1p(1000.0 * candidate.mean_mass_diameter_m_);
        // The exact ratio gate excludes enormous expm1 arguments without inventing a diameter cap.
        if (cooling > candidate.loading_denominator_ * gate)
        {
            candidate.regime_ = GraupelHailSllRegime::BelowSizeRatio;
        }
        else
        {
            candidate.hail_embryo_diameter_m_ = .01 * std::expm1(cooling / candidate.loading_denominator_);
            candidate.size_ratio_ = candidate.mean_mass_diameter_m_ / candidate.hail_embryo_diameter_m_;
            candidate.regime_ = candidate.size_ratio_ >= .1
                ? GraupelHailSllRegime::Active : GraupelHailSllRegime::BelowSizeRatio;
            if (candidate.regime_ == GraupelHailSllRegime::Active)
            {
                candidate.raw_mass_transfer_per_kg_ = .5 * candidate.size_ratio_ * accreted;
                candidate.admitted_mass_transfer_per_kg_ = std::min(
                    candidate.raw_mass_transfer_per_kg_, graupel.mass_per_kg + accreted);
            }
        }
    }
    const double mass = candidate.admitted_mass_transfer_per_kg_;
    const double old_mass_loss = mass + std::max(-diffusion, 0.0);
    if (!std::isfinite(candidate.raw_mass_transfer_per_kg_) || !std::isfinite(mass) ||
        !std::isfinite(old_mass_loss) || old_mass_loss > graupel.mass_per_kg)
    {
        return reject(GraupelHailConversionFailure::IntervalRejected,
            "SLL conversion cannot fund fractional particles beyond its original graupel mass", error, failure);
    }
    if (mass > 0.0)
    {
        candidate.admitted_fraction_ = mass / graupel.mass_per_kg;
        candidate.donor_number_loss_per_kg_ = graupel.number_per_kg * candidate.admitted_fraction_;
        const double donor_z = graupel.reflectivity_per_kg * candidate.admitted_fraction_;
        const double recipient_z = donor_z * (parameters.c_g_ / parameters.c_h_) * (parameters.c_g_ / parameters.c_h_);
        for (double value : {candidate.admitted_fraction_, candidate.donor_number_loss_per_kg_, donor_z, recipient_z})
        {
            if (!std::isfinite(value) || value <= 0.0)
            {
                return reject(GraupelHailConversionFailure::IntervalRejected,
                    "SLL nonzero fractional transfer is unrepresentable", error, failure);
            }
        }
        candidate.mass_increment_per_kg_[2] = -mass;
        candidate.mass_increment_per_kg_[3] = mass;
        candidate.number_increment_per_kg_[2] = -candidate.donor_number_loss_per_kg_;
        candidate.number_increment_per_kg_[3] = candidate.donor_number_loss_per_kg_;
        candidate.reflectivity_increment_per_kg_[2] = -donor_z;
        candidate.reflectivity_increment_per_kg_[3] = recipient_z;
    }
    const double old_number_loss = candidate.donor_number_loss_per_kg_ + std::max(-diffusion_number, 0.0);
    if (!std::isfinite(old_number_loss) || old_number_loss > graupel.number_per_kg)
    {
        return reject(GraupelHailConversionFailure::IntervalRejected,
            "SLL conversion cannot fund fractional particles beyond its original graupel number", error, failure);
    }
    candidate.valid_ = true;
    producer = candidate;
    error.clear();
    return true;
}

bool evaluate_exact_triple_graupel_hail_conversion_cell(
    const ColdParameters& parameters, const GraupelHailConversionCell& cell, double duration_s,
    GraupelHailConversionResult& result, std::string& error, std::size_t capacity,
    GraupelHailConversionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    GraupelHailConversionProducer producer;
    if (!produce_exact_graupel_hail_conversion_admission(
            parameters, cell, duration_s, producer, error, capacity, failure, spectrum_domain))
    {
        return false;
    }
    return phase(producer, result, error, failure);
}

bool validate_exact_graupel_hail_conversion_admission(
    const ColdParameters& parameters, const GraupelHailConversionCell& cell, double duration_s,
    const GraupelHailConversionProducer& producer, std::string& error,
    std::size_t capacity, GraupelHailConversionFailure* failure,
    ColdSpectrumDomain spectrum_domain)
{
    if (capacity != 1)
    {
        return reject(GraupelHailConversionFailure::Capacity,
            "SLL conversion requires exactly one channel record", error, failure);
    }
    if (producer.event_count() != 1 || !same(values(producer.source()), values(cell)) ||
        !same(producer.parameters(), values(parameters)) || producer.hail_processes() != parameters.hail_processes ||
        producer.spectrum_domain() != spectrum_domain || !same(producer.duration_s(), duration_s))
    {
        return reject(GraupelHailConversionFailure::InvalidProducer,
            "SLL conversion capture differs from its original source, amounts, parameters, interval or domain", error, failure);
    }
    GraupelHailConversionProducer regenerated;
    std::string regeneration_error;
    if (!produce_exact_graupel_hail_conversion_admission(parameters, cell, duration_s, regenerated,
            regeneration_error, capacity, nullptr, spectrum_domain) || !same_capture(producer, regenerated))
    {
        return reject(GraupelHailConversionFailure::InvalidProducer,
            "SLL conversion transfer record differs from its regenerated original source", error, failure);
    }
    error.clear();
    if (failure)
    {
        *failure = GraupelHailConversionFailure::None;
    }
    return true;
}

bool apply_exact_triple_graupel_hail_conversion_cell(
    const ColdParameters& parameters, const GraupelHailConversionCell& cell, double duration_s,
    const GraupelHailConversionProducer& producer, GraupelHailConversionResult& result,
    std::string& error, std::size_t capacity, GraupelHailConversionFailure* failure,
    ColdSpectrumDomain spectrum_domain)
{
    return validate_exact_graupel_hail_conversion_admission(parameters, cell, duration_s,
        producer, error, capacity, failure, spectrum_domain) && phase(producer, result, error, failure);
}

} // namespace tmv::milbrandt_detail
