#include "triple_cold_riming.hpp"
#include "triple_cold_ice_graupel.hpp"
#include "triple_cold_graupel_conversions.hpp"
#include "triple_cold_sll.hpp"
#include "triple_cold_collected_sll.hpp"

#include "cold.hpp"
#include "triple_rime_splintering.hpp"
#include "triple_sedimentation.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace tmv::milbrandt_detail
{
namespace
{

constexpr std::array<TripleMomentSpecies, 4> species{
    TripleMomentSpecies::Ice, TripleMomentSpecies::Snow,
    TripleMomentSpecies::Graupel, TripleMomentSpecies::Hail};

bool reject(ColdRimingFailure kind, const char* message,
            std::string& error, ColdRimingFailure* failure)
{
    error = message;
    if (failure)
    {
        *failure = kind;
    }
    return false;
}

bool positive_finite(double value) noexcept
{
    return std::isfinite(value) && value > 0.0;
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

bool same_source(const ColdRimingCell& a, const ColdRimingCell& b) noexcept
{
    if (!same(std::array<double, 6>{a.temperature_k, a.pressure_pa, a.dry_density_kg_m3,
            a.initial.vapor_mass_per_kg, a.initial.cloud_mass_per_kg, a.initial.cloud_number_per_kg},
        std::array<double, 6>{b.temperature_k, b.pressure_pa, b.dry_density_kg_m3,
            b.initial.vapor_mass_per_kg, b.initial.cloud_mass_per_kg, b.initial.cloud_number_per_kg}))
    {
        return false;
    }
    if (!same(std::array<double, 3>{a.initial.rain.mass_per_kg, a.initial.rain.number_per_kg,
            a.initial.rain.reflectivity_per_kg},
        std::array<double, 3>{b.initial.rain.mass_per_kg, b.initial.rain.number_per_kg,
            b.initial.rain.reflectivity_per_kg})) return false;
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        const auto& x = a.initial.frozen[index];
        const auto& y = b.initial.frozen[index];
        if (!same(std::array<double, 3>{x.mass_per_kg, x.number_per_kg, x.reflectivity_per_kg},
                std::array<double, 3>{y.mass_per_kg, y.number_per_kg, y.reflectivity_per_kg}))
        {
            return false;
        }
    }
    return true;
}

std::array<double, 11> values(const ColdRimingChannel& channel) noexcept
{
    return {channel.raw_collection_mass_per_kg, channel.raw_collection_number_per_kg,
        channel.collection_mass_per_kg, channel.collection_number_per_kg,
        channel.environmental_diffusion_mass_per_kg, channel.riming_diffusion_mass_per_kg,
        channel.diffusion_mass_per_kg, channel.diffusion_number_per_kg,
        channel.splinter_embryo_mass_per_kg, channel.splinter_number_per_kg,
        channel.splinter_reflectivity_per_kg};
}

bool same_capture(const ColdRimingProducer& a, const ColdRimingProducer& b) noexcept
{
    if (a.event_count() != b.event_count() || !same_source(a.source(), b.source()) ||
        !same(a.parameters(), b.parameters()) || a.hail_processes() != b.hail_processes() ||
        a.spectrum_domain() != b.spectrum_domain() ||
        a.homogeneous_freezing_enabled() != b.homogeneous_freezing_enabled() ||
        a.rain_collection_enabled() != b.rain_collection_enabled() ||
        a.rain_collision_enabled() != b.rain_collision_enabled() ||
        a.wet_hail_collection_enabled() != b.wet_hail_collection_enabled() ||
        a.wet_growth_enabled() != b.wet_growth_enabled() ||
        !same(a.riming_feedback(), b.riming_feedback()) ||
        !same(a.homogeneous_increments_per_kg(), b.homogeneous_increments_per_kg()) ||
        !same(a.duration_s(), b.duration_s()) ||
        !same(a.cloud_admission_factor(), b.cloud_admission_factor()) ||
        !same(a.relaxation_number(), b.relaxation_number()))
    {
        return false;
    }
    if (a.rain_collection_enabled())
    {
        const auto& x = a.rain_collection();
        const auto& y = b.rain_collection();
        if (x.event_count() != y.event_count() || !same(x.admission_factor(), y.admission_factor()) ||
            !same(x.source_alpha(), y.source_alpha()) || !same(x.source_lambda_per_m(), y.source_lambda_per_m()) ||
            !same(x.mass_weighted_speed_m_s(), y.mass_weighted_speed_m_s())) return false;
        for (std::size_t index = 0; index < 4; ++index)
        {
            const auto& xc = x.channels()[index];
            const auto& yc = y.channels()[index];
            if (!same(std::array<double, 4>{xc.raw_mass_per_kg, xc.raw_number_per_kg, xc.mass_per_kg, xc.number_per_kg},
                    std::array<double, 4>{yc.raw_mass_per_kg, yc.raw_number_per_kg, yc.mass_per_kg, yc.number_per_kg}))
                return false;
        }
    }
    if (a.rain_collision_enabled())
    {
        const auto& x = a.rain_collision_increments();
        const auto& y = b.rain_collision_increments();
        if (!same(x.frozen.mass_increment_per_kg, y.frozen.mass_increment_per_kg) ||
            !same(x.frozen.number_increment_per_kg, y.frozen.number_increment_per_kg) ||
            !same(x.frozen.reflectivity_increment_per_kg, y.frozen.reflectivity_increment_per_kg) ||
            !same(std::array<double, 4>{x.rain.mass_per_kg, x.rain.number_per_kg, x.rain.reflectivity_per_kg, x.temperature_k},
                std::array<double, 4>{y.rain.mass_per_kg, y.rain.number_per_kg, y.rain.reflectivity_per_kg, y.temperature_k})) return false;
    }
    if (a.wet_growth_enabled())
    {
        const auto& x = a.wet_growth();
        const auto& y = b.wet_growth();
        if (x.regime() != y.regime() ||
            !same(std::array<double, 8>{x.normalized_ventilation_m(), x.atmospheric_capacity_per_kg(), x.frozen_capacity_per_kg(),
                x.growth_capacity_per_kg(), x.retained_liquid_per_kg(), x.shed_mass_per_kg(), x.shed_number_per_kg(), x.shed_reflectivity_per_kg()},
                std::array<double, 8>{y.normalized_ventilation_m(), y.atmospheric_capacity_per_kg(), y.frozen_capacity_per_kg(),
                y.growth_capacity_per_kg(), y.retained_liquid_per_kg(), y.shed_mass_per_kg(), y.shed_number_per_kg(), y.shed_reflectivity_per_kg()})) return false;
    }
    const auto& ah = a.homogeneous_freezing();
    const auto& bh = b.homogeneous_freezing();
    const auto source_values = [](const HomogeneousCloudFreezingCell& cell)
    {
        return std::array<double, 5>{cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3,
            cell.cloud_mass_per_kg, cell.cloud_number_per_kg};
    };
    const auto channel_values = [](const HomogeneousCloudFreezingChannel& channel)
    {
        return std::array<double, 7>{channel.fraction, channel.mean_volume_cm3, channel.log10_rate_per_cm3_s,
            channel.source_mass_squared_per_kg, channel.raw_mass_increment_per_kg,
            channel.raw_number_increment_per_kg, channel.raw_reflectivity_increment_per_kg};
    };
    if (ah.event_count() != bh.event_count() || !same(source_values(ah.source()), source_values(bh.source())) ||
        !same(ah.parameters(), bh.parameters()) || ah.hail_processes() != bh.hail_processes() ||
        !same(ah.duration_s(), bh.duration_s()) || !same(channel_values(ah.channel()), channel_values(bh.channel())))
    {
        return false;
    }
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        if (!same(values(a.channels()[index]), values(b.channels()[index])))
        {
            return false;
        }
    }
    return true;
}

bool bound_source(const ColdParameters& p, const ColdRimingCell& cell, double duration_s,
                  const ColdRimingProducer& producer, ColdSpectrumDomain domain,
                  bool homogeneous_freezing = false, bool rain_collection = false, bool rain_collision = false, bool wet_growth = false) noexcept
{
    const std::array<double, 25> parameters{p.a_g_, p.a_h_, p.a_i_, p.a_r_, p.a_s_,
        p.alpha_g_, p.alpha_h_, p.alpha_i_, p.alpha_r_, p.alpha_s_,
        p.b_g_, p.b_h_, p.b_i_, p.b_r_, p.b_s_, p.c_g_, p.c_h_, p.c_i_, p.c_r_, p.c_s_,
        p.d_g_, p.d_h_, p.d_i_, p.d_r_, p.d_s_};
    return producer.event_count() == (homogeneous_freezing ? 5u : 4u) + (rain_collection ? 4u : 0u) + (rain_collision ? 3u : 0u) + (wet_growth ? 1u : 0u) &&
        producer.homogeneous_freezing_enabled() == homogeneous_freezing && same_source(producer.source(), cell) &&
        same(producer.parameters(), parameters) && producer.hail_processes() == p.hail_processes &&
        producer.spectrum_domain() == domain && same(producer.duration_s(), duration_s) && producer.rain_collection_enabled() == rain_collection && producer.rain_collision_enabled() == rain_collision && producer.wet_growth_enabled() == wet_growth;
}

template <typename Producer>
bool same_conversion(const Producer& a, const Producer& b) noexcept
{
    return a.event_count() == b.event_count() && same_capture(a.riming_source(), b.riming_source()) &&
        same(a.raw_mass_transfer_per_kg(), b.raw_mass_transfer_per_kg()) &&
        same(a.admitted_mass_transfer_per_kg(), b.admitted_mass_transfer_per_kg()) &&
        same(a.donor_number_loss_per_kg(), b.donor_number_loss_per_kg()) &&
        same(a.recipient_number_gain_per_kg(), b.recipient_number_gain_per_kg()) &&
        same(a.mass_increment_per_kg(), b.mass_increment_per_kg()) &&
        same(a.number_increment_per_kg(), b.number_increment_per_kg()) &&
        same(a.reflectivity_increment_per_kg(), b.reflectivity_increment_per_kg());
}

bool same_snow_conversion(const SnowGraupelConversionProducer& a,
                          const SnowGraupelConversionProducer& b) noexcept
{
    return same_conversion(a, b) &&
        same(a.source_mass_squared_per_kg(), b.source_mass_squared_per_kg()) &&
        same(a.source_alpha(), b.source_alpha()) && same(a.source_lambda_per_m(), b.source_lambda_per_m()) &&
        same(a.admitted_fraction(), b.admitted_fraction());
}

double graupel_rain_accretion(const ColdRimingProducer& riming) noexcept
{
    if (!riming.rain_collection_enabled() ||
        (riming.rain_collision_enabled() && riming.rain_collision().channels()[2].destination != TripleMomentSpecies::Graupel))
    {
        return 0.0;
    }
    return riming.admitted_rain_channel(2).mass_per_kg;
}

GraupelHailConversionCell cloud_sll_source(const ColdRimingProducer& riming,
                                         const FrozenCollectionProducer* collection = nullptr)
{
    GraupelHailConversionCell result;
    result.source = riming.source();
    result.admitted_accretion_mass_per_kg[0] = riming.channels()[2].collection_mass_per_kg;
    if (collection)
    {
        result.admitted_accretion_mass_per_kg[2] = collection->channels()[1].admitted_mass_increment_per_kg;
    }
    result.graupel_diffusion_mass_per_kg = riming.channels()[2].diffusion_mass_per_kg;
    result.graupel_diffusion_number_per_kg = riming.channels()[2].diffusion_number_per_kg;
    if (riming.rain_collection_enabled())
    {
        result.rain_mass_per_kg = riming.source().initial.rain.mass_per_kg;
        result.admitted_accretion_mass_per_kg[1] = graupel_rain_accretion(riming);
    }
    return result;
}

bool bound_cloud_sll(const ColdRimingProducer& riming,
                     const GraupelHailConversionProducer& conversion,
                     const FrozenCollectionProducer* collection = nullptr) noexcept
{
    const auto& source = conversion.source();
    return conversion.event_count() == 1 && same_source(source.source, riming.source()) &&
        same(conversion.parameters(), riming.parameters()) && conversion.hail_processes() == riming.hail_processes() &&
        conversion.spectrum_domain() == riming.spectrum_domain() && same(conversion.duration_s(), riming.duration_s()) &&
        same(std::array<double, 6>{source.rain_mass_per_kg, source.admitted_accretion_mass_per_kg[0],
                source.admitted_accretion_mass_per_kg[1], source.admitted_accretion_mass_per_kg[2],
                source.graupel_diffusion_mass_per_kg, source.graupel_diffusion_number_per_kg},
            std::array<double, 6>{riming.rain_collection_enabled() ? riming.source().initial.rain.mass_per_kg : 0.0,
                riming.channels()[2].collection_mass_per_kg,
                graupel_rain_accretion(riming),
                collection ? collection->channels()[1].admitted_mass_increment_per_kg : 0.0,
                riming.channels()[2].diffusion_mass_per_kg, riming.channels()[2].diffusion_number_per_kg});
}

FrozenCollectionCell frozen_collection_source(const ColdRimingCell& source)
{
    return {source.temperature_k, source.pressure_pa, source.dry_density_kg_m3, {source.initial.frozen}};
}

bool original_collected_donors(const ColdRimingProducer& riming, const ColdCollectedSllResult& result,
                               std::string& error, ColdRimingFailure* failure)
{
    const auto& channels = result.collection.channels();
    std::array<double, 3> mass_loss{
        result.ice_conversion.admitted_mass_transfer_per_kg(),
        result.snow_conversion.admitted_mass_transfer_per_kg(),
        result.hail_conversion.admitted_mass_transfer_per_kg()};
    std::array<double, 3> number_loss{
        result.ice_conversion.donor_number_loss_per_kg(),
        result.snow_conversion.donor_number_loss_per_kg(),
        result.hail_conversion.donor_number_loss_per_kg()};
    for (std::size_t index = 0; index < 3; ++index)
    {
        mass_loss[0] += channels[index].admitted_mass_increment_per_kg;
        number_loss[0] += channels[index].admitted_number_increment_per_kg;
    }
    mass_loss[1] += channels[3].admitted_mass_increment_per_kg;
    number_loss[1] += channels[3].admitted_number_increment_per_kg + channels[4].admitted_number_increment_per_kg;
    for (std::size_t index = 0; index < 3; ++index)
    {
        if (riming.rain_collision_enabled())
        {
            mass_loss[index] += riming.rain_collision().channels()[index].frozen.mass_per_kg;
            number_loss[index] += riming.rain_collision().channels()[index].frozen.number_per_kg;
        }
        mass_loss[index] += std::max(-riming.channels()[index].diffusion_mass_per_kg, 0.0);
        number_loss[index] += std::max(-riming.channels()[index].diffusion_number_per_kg, 0.0);
        const auto& source = riming.source().initial.frozen[index];
        if (!std::isfinite(mass_loss[index]) || !std::isfinite(number_loss[index]) ||
            mass_loss[index] > source.mass_per_kg || number_loss[index] > source.number_per_kg)
        {
            return reject(ColdRimingFailure::IntervalRejected,
                "collected SLL sinks exceed an original frozen mass or number budget", error, failure);
        }
    }
    return true;
}

bool valid_population(const FrozenCollectionPopulation& population,
                      TripleMomentSpecies category, std::string& error, ColdSpectrumDomain spectrum_domain)
{
    for (double value : {population.mass_per_kg, population.number_per_kg,
                         population.reflectivity_per_kg})
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            error = "cold riming population is nonfinite or negative";
            return false;
        }
    }
    if (population.mass_per_kg == 0.0 && population.number_per_kg == 0.0 &&
        population.reflectivity_per_kg == 0.0)
    {
        return true;
    }
    const auto law = current_nimbus_triple_sedimentation_parameters(category).mass_law;
    double alpha = 0.0;
    double lambda = 0.0;
    if (!diagnose_triple_gamma_shape(population.mass_per_kg, population.number_per_kg,
            population.reflectivity_per_kg, law, alpha, lambda, error))
    {
        return false;
    }
    if (law.exponent == 3.0)
    {
        int qe = 0, ne = 0, ze = 0, ce = 0;
        const double q = std::frexp(population.mass_per_kg, &qe);
        const double n = std::frexp(population.number_per_kg, &ne);
        const double z = std::frexp(population.reflectivity_per_kg, &ze);
        const double c = std::frexp(law.coefficient, &ce);
        const double target = std::log(std::ldexp(z * n * (c / q) * (c / q), ze + ne + 2 * (ce - qe)));
        const auto log_ratio = [](double shape)
        {
            return std::log1p(3.0 / (shape + 1.0)) + std::log1p(3.0 / (shape + 2.0)) +
                std::log1p(3.0 / (shape + 3.0));
        };
        double lower = -1.0 + 1e-9;
        double upper = 1e6;
        if (!positive_finite(target) || target >= log_ratio(lower) || target <= log_ratio(upper))
        {
            error = "cold riming cubic endpoint is outside its admitted shape bracket";
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
    }
    if (alpha > cold_spectrum_shape_limit(spectrum_domain, category))
    {
        error = "cold riming endpoint shape exceeds its selected spectrum domain";
        return false;
    }
    return true;
}

bool float_range(double value) noexcept
{
    return std::isfinite(value) && value >= 0.0 && value <= std::numeric_limits<float>::max() &&
        (value == 0.0 || value >= std::numeric_limits<float>::denorm_min());
}

bool valid_cloud(double mass, double number) noexcept
{
    return float_range(mass) && float_range(number) && (mass == 0.0 || number > 0.0);
}

bool phase_endpoint(const ColdRimingProducer& producer, ColdRimingResult& result,
                    std::string& error, ColdRimingFailure* failure,
                    const IceGraupelConversionProducer* conversion = nullptr,
                    const SnowGraupelConversionProducer* snow_conversion = nullptr,
                    const GraupelHailConversionProducer* hail_conversion = nullptr,
                    const FrozenCollectionIncrements* collection = nullptr)
{
    const auto& source = producer.source();
    const bool wet = producer.wet_growth_enabled() && producer.wet_growth().regime() == HailWetGrowthRegime::Wet;
    ColdRimingResult candidate;
    candidate.unrounded = source.initial;
    double collection_sum = producer.homogeneous_increments_per_kg()[0];
    double diffusion_sum = 0.0;
    if (producer.homogeneous_freezing_enabled())
    {
        candidate.cloud_number_increment_per_kg = -producer.homogeneous_increments_per_kg()[1];
    }
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        const auto& initial = source.initial.frozen[index];
        const auto& channel = producer.channels()[index];
        if (channel.diffusion_mass_per_kg < -initial.mass_per_kg ||
            channel.diffusion_number_per_kg < -initial.number_per_kg)
        {
            return reject(ColdRimingFailure::IntervalRejected,
                "cold riming diffusion exhausts its source population before secondary ice production", error, failure);
        }
        double growth = channel.collection_mass_per_kg + channel.diffusion_mass_per_kg;
        if (producer.rain_collection_enabled())
        {
            const auto& rain = producer.admitted_rain_channel(index);
            if (!producer.rain_collision_enabled()) growth += rain.mass_per_kg;
            candidate.rain_mass_increment_per_kg -= rain.mass_per_kg;
            candidate.rain_number_increment_per_kg -= rain.number_per_kg;
        }
        if (wet && index == 3) growth = producer.wet_growth().retained_liquid_per_kg() + channel.diffusion_mass_per_kg;
        candidate.mass_increment_per_kg[index] = growth;
        candidate.number_increment_per_kg[index] = channel.diffusion_number_per_kg;
        collection_sum += channel.collection_mass_per_kg;
        diffusion_sum += channel.diffusion_mass_per_kg;
        candidate.cloud_number_increment_per_kg -= channel.collection_number_per_kg;
        if (initial.mass_per_kg > 0.0)
        {
            const double power = 6.0 / current_nimbus_triple_sedimentation_parameters(species[index]).mass_law.exponent;
            candidate.reflectivity_increment_per_kg[index] = initial.reflectivity_per_kg *
                (power * growth / initial.mass_per_kg +
                 (1.0 - power) * channel.diffusion_number_per_kg / initial.number_per_kg);
        }
    }
    if (producer.rain_collision_enabled())
    {
        const auto& increments = producer.rain_collision_increments().frozen;
        for (std::size_t index = 0; index < species.size(); ++index)
        {
            candidate.mass_increment_per_kg[index] += increments.mass_increment_per_kg[index];
            candidate.number_increment_per_kg[index] += increments.number_increment_per_kg[index];
            candidate.reflectivity_increment_per_kg[index] += increments.reflectivity_increment_per_kg[index];
        }
        if (wet)
        {
            const auto& hail = source.initial.frozen[3];
            const double rain = producer.admitted_rain_channel(3).mass_per_kg;
            candidate.mass_increment_per_kg[3] -= rain;
            candidate.reflectivity_increment_per_kg[3] -= 2.0 * hail.reflectivity_per_kg * rain / hail.mass_per_kg;
        }
    }
    if (conversion)
    {
        for (std::size_t index = 0; index < species.size(); ++index)
        {
            candidate.mass_increment_per_kg[index] += conversion->mass_increment_per_kg()[index];
            candidate.number_increment_per_kg[index] += conversion->number_increment_per_kg()[index];
            candidate.reflectivity_increment_per_kg[index] += conversion->reflectivity_increment_per_kg()[index];
        }
    }
    if (snow_conversion)
    {
        for (std::size_t index = 0; index < species.size(); ++index)
        {
            candidate.mass_increment_per_kg[index] += snow_conversion->mass_increment_per_kg()[index];
            candidate.number_increment_per_kg[index] += snow_conversion->number_increment_per_kg()[index];
            candidate.reflectivity_increment_per_kg[index] += snow_conversion->reflectivity_increment_per_kg()[index];
        }
    }
    if (hail_conversion)
    {
        for (std::size_t index = 0; index < species.size(); ++index)
        {
            candidate.mass_increment_per_kg[index] += hail_conversion->mass_increment_per_kg()[index];
            candidate.number_increment_per_kg[index] += hail_conversion->number_increment_per_kg()[index];
            candidate.reflectivity_increment_per_kg[index] += hail_conversion->reflectivity_increment_per_kg()[index];
        }
    }
    if (collection)
    {
        for (std::size_t index = 0; index < species.size(); ++index)
        {
            candidate.mass_increment_per_kg[index] += collection->mass_increment_per_kg[index];
            candidate.number_increment_per_kg[index] += collection->number_increment_per_kg[index];
            candidate.reflectivity_increment_per_kg[index] += collection->reflectivity_increment_per_kg[index];
        }
    }
    for (std::size_t donor = 0; donor < 3; ++donor)
    {
        const auto& channel = producer.channels()[donor];
        candidate.number_increment_per_kg[0] += channel.splinter_number_per_kg;
        candidate.reflectivity_increment_per_kg[0] += channel.splinter_reflectivity_per_kg;
        if (donor != 0)
        {
            candidate.mass_increment_per_kg[0] += channel.splinter_embryo_mass_per_kg;
            candidate.mass_increment_per_kg[donor] -= channel.splinter_embryo_mass_per_kg;
            candidate.reflectivity_increment_per_kg[donor] -= channel.splinter_reflectivity_per_kg;
        }
    }
    if (producer.homogeneous_freezing_enabled())
    {
        candidate.mass_increment_per_kg[0] += producer.homogeneous_increments_per_kg()[0];
        candidate.number_increment_per_kg[0] += producer.homogeneous_increments_per_kg()[1];
        candidate.reflectivity_increment_per_kg[0] += producer.homogeneous_increments_per_kg()[2];
    }
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        const auto& initial = source.initial.frozen[index];
        auto& endpoint = candidate.unrounded.frozen[index];
        endpoint = {initial.mass_per_kg + candidate.mass_increment_per_kg[index],
            initial.number_per_kg + candidate.number_increment_per_kg[index],
            initial.reflectivity_per_kg + candidate.reflectivity_increment_per_kg[index]};
        if (!valid_population(endpoint, species[index], error, producer.spectrum_domain()) ||
            !float_range(endpoint.mass_per_kg) || !float_range(endpoint.number_per_kg) ||
            !float_range(endpoint.reflectivity_per_kg))
        {
            return reject(ColdRimingFailure::IntervalRejected,
                "cold riming coupled q/N/Z endpoint is inadmissible or not FP32 representable", error, failure);
        }
        auto& rounded = candidate.rounded[index];
        rounded = {static_cast<float>(endpoint.mass_per_kg), static_cast<float>(endpoint.number_per_kg),
            static_cast<float>(endpoint.reflectivity_per_kg)};
        if (!valid_population({rounded.mass_per_kg, rounded.number_per_kg, rounded.reflectivity_per_kg},
                species[index], error, producer.spectrum_domain()))
        {
            return reject(ColdRimingFailure::IntervalRejected,
                "cold riming stored q/N/Z endpoint is inadmissible", error, failure);
        }
    }
    candidate.vapor_mass_increment_per_kg = -diffusion_sum;
    candidate.cloud_mass_increment_per_kg = -collection_sum;
    candidate.unrounded.vapor_mass_per_kg += candidate.vapor_mass_increment_per_kg;
    candidate.unrounded.cloud_mass_per_kg += candidate.cloud_mass_increment_per_kg;
    candidate.unrounded.cloud_number_per_kg += candidate.cloud_number_increment_per_kg;
    if (!float_range(candidate.unrounded.vapor_mass_per_kg) ||
        !valid_cloud(candidate.unrounded.cloud_mass_per_kg, candidate.unrounded.cloud_number_per_kg))
    {
        return reject(ColdRimingFailure::IntervalRejected,
            "cold riming vapor/cloud endpoint is inadmissible or not FP32 representable", error, failure);
    }
    candidate.rounded_vapor_mass_per_kg = static_cast<float>(candidate.unrounded.vapor_mass_per_kg);
    candidate.rounded_cloud_mass_per_kg = static_cast<float>(candidate.unrounded.cloud_mass_per_kg);
    candidate.rounded_cloud_number_per_kg = static_cast<float>(candidate.unrounded.cloud_number_per_kg);
    if (!valid_cloud(candidate.rounded_cloud_mass_per_kg, candidate.rounded_cloud_number_per_kg))
    {
        return reject(ColdRimingFailure::IntervalRejected,
            "cold riming stored cloud is number-exhausted", error, failure);
    }
    if (producer.rain_collection_enabled())
    {
        const auto& rain = source.initial.rain;
        if (rain.mass_per_kg > 0.0)
        {
            candidate.rain_reflectivity_increment_per_kg = rain.reflectivity_per_kg *
                (2.0 * candidate.rain_mass_increment_per_kg / rain.mass_per_kg -
                 candidate.rain_number_increment_per_kg / rain.number_per_kg);
        }
        if (wet)
        {
            candidate.rain_mass_increment_per_kg += producer.wet_growth().shed_mass_per_kg();
            candidate.rain_number_increment_per_kg += producer.wet_growth().shed_number_per_kg();
            candidate.rain_reflectivity_increment_per_kg += producer.wet_growth().shed_reflectivity_per_kg();
        }
        candidate.unrounded.rain = {rain.mass_per_kg + candidate.rain_mass_increment_per_kg,
            rain.number_per_kg + candidate.rain_number_increment_per_kg,
            rain.reflectivity_per_kg + candidate.rain_reflectivity_increment_per_kg};
        const auto& endpoint = candidate.unrounded.rain;
        if (!valid_population(endpoint, TripleMomentSpecies::Rain, error, producer.spectrum_domain()) ||
            !float_range(endpoint.mass_per_kg) || !float_range(endpoint.number_per_kg) ||
            !float_range(endpoint.reflectivity_per_kg))
        {
            return reject(ColdRimingFailure::IntervalRejected,
                "rain collection q/N/Z endpoint is inadmissible or not FP32 representable", error, failure);
        }
        candidate.rounded_rain = {static_cast<float>(endpoint.mass_per_kg), static_cast<float>(endpoint.number_per_kg),
            static_cast<float>(endpoint.reflectivity_per_kg)};
        const auto& rounded = candidate.rounded_rain;
        if (!valid_population({rounded.mass_per_kg, rounded.number_per_kg, rounded.reflectivity_per_kg},
                TripleMomentSpecies::Rain, error, producer.spectrum_domain()))
        {
            return reject(ColdRimingFailure::IntervalRejected,
                "rain collection stored q/N/Z endpoint is inadmissible", error, failure);
        }
    }
    candidate.temperature_increment_k =
        (microphysics_constants::L_f * collection_sum + microphysics_constants::L_s * diffusion_sum) /
        microphysics_constants::cp;
    if (producer.rain_collection_enabled())
    {
        candidate.temperature_increment_k -= microphysics_constants::L_f * candidate.rain_mass_increment_per_kg /
            microphysics_constants::cp;
    }
    if (wet)
    {
        double frozen_liquid = producer.homogeneous_increments_per_kg()[0] + producer.wet_growth().retained_liquid_per_kg();
        for (std::size_t index = 0; index < 3; ++index)
            frozen_liquid += producer.channels()[index].collection_mass_per_kg + producer.admitted_rain_channel(index).mass_per_kg;
        candidate.temperature_increment_k = (microphysics_constants::L_f * frozen_liquid +
            microphysics_constants::L_s * diffusion_sum) / microphysics_constants::cp;
    }
    const double temperature = source.temperature_k + candidate.temperature_increment_k;
    if (!std::isfinite(temperature) || temperature < 150.0 || temperature >= microphysics_constants::T0)
    {
        return reject(ColdRimingFailure::IntervalRejected,
            "cold riming heat leaves the cold thermal domain", error, failure);
    }
    long double water_change = static_cast<long double>(candidate.unrounded.vapor_mass_per_kg) -
        source.initial.vapor_mass_per_kg + static_cast<long double>(candidate.unrounded.cloud_mass_per_kg) -
        source.initial.cloud_mass_per_kg;
    long double water_scale = static_cast<long double>(source.initial.vapor_mass_per_kg) + source.initial.cloud_mass_per_kg;
    if (producer.rain_collection_enabled())
    {
        water_change += static_cast<long double>(candidate.unrounded.rain.mass_per_kg) - source.initial.rain.mass_per_kg;
        water_scale += source.initial.rain.mass_per_kg;
    }
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        water_change += static_cast<long double>(candidate.unrounded.frozen[index].mass_per_kg) -
            source.initial.frozen[index].mass_per_kg;
        water_scale += source.initial.frozen[index].mass_per_kg;
    }
    if (std::abs(water_change) > 64.0L * std::numeric_limits<double>::epsilon() * water_scale)
    {
        return reject(ColdRimingFailure::IntervalRejected,
            "cold riming endpoint violates water closure", error, failure);
    }
    candidate.producer = producer;
    result = candidate;
    error.clear();
    if (failure)
    {
        *failure = ColdRimingFailure::None;
    }
    return true;
}

} // namespace

bool validate_cold_riming_source_population(
    const FrozenCollectionPopulation& population, TripleMomentSpecies category,
    ColdSpectrumDomain spectrum_domain, std::string& error)
{
    if (!valid_cold_spectrum_domain(spectrum_domain) ||
        (category != TripleMomentSpecies::Rain && category != TripleMomentSpecies::Ice && category != TripleMomentSpecies::Snow &&
         category != TripleMomentSpecies::Graupel && category != TripleMomentSpecies::Hail))
    {
        error = "cold riming source category or spectrum domain is invalid";
        return false;
    }
    if (!valid_population(population, category, error, spectrum_domain))
    {
        return false;
    }
    error.clear();
    return true;
}

bool produce_exact_cold_riming_admission(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    ColdRimingProducer& producer, std::string& error, std::size_t capacity, ColdRimingFailure* failure,
    ColdSpectrumDomain spectrum_domain, bool homogeneous_freezing, bool rain_collection, bool rain_collision, bool wet_hail_collection)
{
    if (!valid_cold_spectrum_domain(spectrum_domain))
    {
        return reject(ColdRimingFailure::InvalidContext,
            "cold riming spectrum domain is invalid", error, failure);
    }
    if (capacity != species.size() + (homogeneous_freezing ? 1 : 0) + (rain_collection ? 4 : 0) + (rain_collision ? 3 : 0))
    {
        return reject(ColdRimingFailure::Capacity, "cold riming channel capacity differs from its selected processes", error, failure);
    }
    if (rain_collision && !rain_collection)
        return reject(ColdRimingFailure::InvalidContext, "rain collisions require the rain collection owner", error, failure);
    if (!rain_collection && (cell.initial.rain.mass_per_kg != 0.0 ||
        cell.initial.rain.number_per_kg != 0.0 || cell.initial.rain.reflectivity_per_kg != 0.0))
    {
        return reject(ColdRimingFailure::InvalidSource, "cold riming rain requires its collection owner", error, failure);
    }
    ColdCloudDiffusionCell secondary;
    secondary.temperature_k = cell.temperature_k;
    secondary.pressure_pa = cell.pressure_pa;
    secondary.dry_density_kg_m3 = cell.dry_density_kg_m3;
    secondary.vapor_mass_per_kg = cell.initial.vapor_mass_per_kg;
    secondary.initial.cloud_mass_per_kg = cell.initial.cloud_mass_per_kg;
    secondary.initial.cloud_number_per_kg = cell.initial.cloud_number_per_kg;
    std::copy(cell.initial.frozen.begin() + 1, cell.initial.frozen.end(), secondary.initial.frozen.begin());
    ColdCloudDiffusionProducer secondary_producer;
    ColdCloudDiffusionFailure secondary_failure = ColdCloudDiffusionFailure::None;
    if (!produce_exact_cold_cloud_diffusion_admission(parameters, secondary, duration_s,
            secondary_producer, error, 3, &secondary_failure, spectrum_domain))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(secondary_failure);
        }
        return false;
    }
    IceCloudCollectionCell ice{cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3,
        {cell.initial.cloud_mass_per_kg, cell.initial.cloud_number_per_kg, cell.initial.frozen[0]}};
    IceCloudCollectionProducer ice_producer;
    IceCloudCollectionFailure ice_failure = IceCloudCollectionFailure::None;
    if (!produce_exact_ice_cloud_collection_admission(parameters, ice, duration_s,
            ice_producer, error, 1, &ice_failure))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(ice_failure);
        }
        return false;
    }
    ColdCloudCollectionProducer wet_cloud;
    if (wet_hail_collection)
    {
        ColdCloudCollectionFailure wet_failure = ColdCloudCollectionFailure::None;
        if (!produce_exact_cold_cloud_collection_admission(parameters,
                {cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3, secondary.initial}, duration_s,
                wet_cloud, error, 3, &wet_failure, spectrum_domain, true))
        {
            if (failure) *failure = static_cast<ColdRimingFailure>(wet_failure);
            return false;
        }
    }
    ColdRimingProducer candidate;
    candidate.wet_hail_collection_enabled_ = wet_hail_collection;
    candidate.source_ = cell;
    candidate.parameters_ = ice_producer.parameters();
    candidate.hail_processes_ = parameters.hail_processes;
    candidate.spectrum_domain_ = spectrum_domain;
    candidate.homogeneous_freezing_enabled_ = homogeneous_freezing;
    candidate.duration_s_ = duration_s;
    candidate.rain_collection_enabled_ = rain_collection;
    candidate.rain_collision_enabled_ = rain_collision;
    if (rain_collection)
    {
        FrozenCollectionFailure rain_failure = FrozenCollectionFailure::None;
        if (!produce_exact_rain_collection_admission(parameters,
                {frozen_collection_source(cell), cell.initial.rain}, duration_s, candidate.rain_collection_,
                error, 4, &rain_failure, spectrum_domain))
        {
            if (failure) *failure = static_cast<ColdRimingFailure>(rain_failure);
            return false;
        }
    }
    if (rain_collision)
    {
        FrozenCollectionFailure collision_failure = FrozenCollectionFailure::None;
        const RainCollectionCell collision_source{frozen_collection_source(cell), cell.initial.rain};
        if (!produce_exact_rain_collision_admission(parameters, collision_source, duration_s,
                candidate.rain_collision_, error, 7, &collision_failure, spectrum_domain) ||
            !evaluate_exact_rain_collision_increments(parameters, collision_source, duration_s,
                candidate.rain_collision_, candidate.rain_collision_increments_, error, 7, &collision_failure, spectrum_domain))
        {
            if (failure) *failure = static_cast<ColdRimingFailure>(collision_failure);
            return false;
        }
    }
    if (homogeneous_freezing)
    {
        HomogeneousCloudFreezingFailure freezing_failure = HomogeneousCloudFreezingFailure::None;
        const HomogeneousCloudFreezingCell freezing_cell{cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3,
            cell.initial.cloud_mass_per_kg, cell.initial.cloud_number_per_kg};
        if (!produce_exact_homogeneous_cloud_freezing_admission(parameters, freezing_cell, duration_s,
                candidate.homogeneous_freezing_, error, 1, &freezing_failure))
        {
            if (failure)
            {
                *failure = static_cast<ColdRimingFailure>(freezing_failure);
            }
            return false;
        }
    }
    const auto& freezing = candidate.homogeneous_freezing_.channel();
    double mass_sink = freezing.raw_mass_increment_per_kg;
    double number_sink = freezing.raw_number_increment_per_kg;
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        auto& channel = candidate.channels_[index];
        if (index == 0)
        {
            channel.raw_collection_mass_per_kg = ice_producer.channel().raw_mass_increment_per_kg;
            channel.raw_collection_number_per_kg = ice_producer.channel().raw_number_increment_per_kg;
        }
        else
        {
            const auto& raw = wet_hail_collection && index == 3 ? wet_cloud.channels()[2] :
                secondary_producer.collection_producer().channels()[index - 1];
            channel.raw_collection_mass_per_kg = raw.raw_mass_increment_per_kg;
            channel.raw_collection_number_per_kg = raw.raw_number_increment_per_kg;
        }
        mass_sink += channel.raw_collection_mass_per_kg;
        number_sink += channel.raw_collection_number_per_kg;
    }
    if (!std::isfinite(mass_sink) || !std::isfinite(number_sink))
    {
        return reject(ColdRimingFailure::IntervalRejected,
            "cold riming raw shared cloud budget is unrepresentable", error, failure);
    }
    double factor = 1.0;
    if (mass_sink > 0.0)
    {
        factor = std::min(factor, cell.initial.cloud_mass_per_kg / mass_sink);
    }
    if (number_sink > 0.0)
    {
        factor = std::min(factor, cell.initial.cloud_number_per_kg / number_sink);
    }
    if (mass_sink > cell.initial.cloud_mass_per_kg || number_sink > cell.initial.cloud_number_per_kg)
    {
        factor *= cold_snow_riming_overdraw_factor;
    }
    candidate.cloud_admission_factor_ = factor;
    candidate.homogeneous_increments_per_kg_ = {factor * freezing.raw_mass_increment_per_kg,
        factor * freezing.raw_number_increment_per_kg, factor * freezing.raw_reflectivity_increment_per_kg};
    const std::array<double, 3> freezing_raw{freezing.raw_mass_increment_per_kg,
        freezing.raw_number_increment_per_kg, freezing.raw_reflectivity_increment_per_kg};
    for (std::size_t index = 0; index < freezing_raw.size(); ++index)
    {
        const double accepted = candidate.homogeneous_increments_per_kg_[index];
        if (!std::isfinite(accepted) || accepted < 0.0 ||
            (freezing_raw[index] > 0.0 && factor > 0.0 && accepted == 0.0))
        {
            return reject(ColdRimingFailure::IntervalRejected,
                "coupled homogeneous freezing transfer is unrepresentable", error, failure);
        }
    }
    double admitted_mass = candidate.homogeneous_increments_per_kg_[0];
    double admitted_number = candidate.homogeneous_increments_per_kg_[1];
    for (auto& channel : candidate.channels_)
    {
        channel.collection_mass_per_kg = factor * channel.raw_collection_mass_per_kg;
        channel.collection_number_per_kg = factor * channel.raw_collection_number_per_kg;
        admitted_mass += channel.collection_mass_per_kg;
        admitted_number += channel.collection_number_per_kg;
    }
    if (!std::isfinite(factor) || factor < 0.0 || factor > 1.0 ||
        admitted_mass > cell.initial.cloud_mass_per_kg || admitted_number > cell.initial.cloud_number_per_kg)
    {
        return reject(ColdRimingFailure::IntervalRejected,
            "cold riming paired cloud admission did not close", error, failure);
    }
    const auto& thermal = secondary_producer.thermodynamics();
    candidate.relaxation_number_ = thermal.relaxation_number;
    candidate.riming_feedback_ = thermal.riming_feedback;
    if (cell.initial.frozen[0].mass_per_kg > 0.0)
    {
        const double alpha = ice_producer.source_alpha();
        const double lambda = ice_producer.source_lambda_per_m();
        const auto ice_parameters = current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Ice);
        const double order = 1.5 + .5 * ice_parameters.fall_exponent;
        const double ventilated = std::exp(std::lgamma(alpha + order + 1.0) -
            std::lgamma(alpha + 1.0) - order * std::log(lambda));
        const double ventilation = .78 * (alpha + 1.0) / lambda + .30 * thermal.schmidt_one_third *
            std::sqrt(thermal.density_factor * ice_parameters.fall_coefficient /
                thermal.kinematic_viscosity_m2_s) * ventilated;
        // Preserve the retained pristine-ice capacitance; surface-heating feedback has no such multiplier.
        const double conductance = 3.14159265358979323846 * cell.initial.frozen[0].number_per_kg *
            ventilation / thermal.diffusion_resistance_m_s_per_kg;
        const double coefficient = conductance / thermal.saturation_ice_per_kg;
        if (!positive_finite(ventilation) || !positive_finite(coefficient))
        {
            return reject(ColdRimingFailure::InvalidSource,
                "cold riming ice ventilation is unrepresentable", error, failure);
        }
        candidate.channels_[0].environmental_diffusion_mass_per_kg =
            duration_s * conductance * (thermal.saturation_ratio_ice - 1.0);
        candidate.relaxation_number_ += duration_s * coefficient *
            (1.0 + thermal.saturation_temperature_derivative_per_kg_k *
                microphysics_constants::L_s / microphysics_constants::cp);
    }
    if (!std::isfinite(candidate.relaxation_number_) || candidate.relaxation_number_ > 1.0)
    {
        return reject(ColdRimingFailure::IntervalRejected,
            "cold riming interval exceeds the four-population relaxation bound", error, failure);
    }
    RimeSplinteringCell splintering;
    splintering.temperature_k = cell.temperature_k;
    splintering.pressure_pa = cell.pressure_pa;
    splintering.dry_density_kg_m3 = cell.dry_density_kg_m3;
    splintering.cloud_mass_per_kg = cell.initial.cloud_mass_per_kg;
    splintering.initial = cell.initial.frozen;
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        auto& channel = candidate.channels_[index];
        if (index != 0)
        {
            channel.environmental_diffusion_mass_per_kg =
                secondary_producer.channels()[index - 1].environmental_mass_increment_per_kg;
        }
        double liquid_collection = channel.collection_mass_per_kg;
        if (rain_collection) liquid_collection += candidate.admitted_rain_channel(index).mass_per_kg;
        channel.riming_diffusion_mass_per_kg = -thermal.riming_feedback * liquid_collection;
        channel.diffusion_mass_per_kg = channel.environmental_diffusion_mass_per_kg + channel.riming_diffusion_mass_per_kg;
        if (channel.diffusion_mass_per_kg < 0.0)
        {
            const auto& initial = cell.initial.frozen[index];
            channel.diffusion_number_per_kg = initial.number_per_kg * (channel.diffusion_mass_per_kg / initial.mass_per_kg);
        }
        splintering.admitted_cloud_mass_per_kg[index] = channel.collection_mass_per_kg;
        for (double value : values(channel))
        {
            if (!std::isfinite(value))
            {
                return reject(ColdRimingFailure::IntervalRejected,
                    "cold riming coupled transfer is unrepresentable", error, failure);
            }
        }
    }
    RimeSplinteringProducer splinter_producer;
    RimeSplinteringFailure splinter_failure = RimeSplinteringFailure::None;
    if (!produce_exact_rime_splintering_admission(parameters, splintering, duration_s,
            splinter_producer, error, 3, &splinter_failure, spectrum_domain))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(splinter_failure);
        }
        return false;
    }
    for (std::size_t index = 0; index < 3; ++index)
    {
        auto& channel = candidate.channels_[index];
        const auto& event = splinter_producer.channels()[index];
        channel.splinter_embryo_mass_per_kg = event.embryo_mass_per_kg;
        channel.splinter_number_per_kg = event.number_increment_per_kg;
        channel.splinter_reflectivity_per_kg = event.reflectivity_increment_per_kg;
    }
    candidate.valid_ = true;
    producer = candidate;
    error.clear();
    if (failure)
    {
        *failure = ColdRimingFailure::None;
    }
    return true;
}

bool produce_exact_wet_cold_riming_admission(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    ColdRimingProducer& producer, FrozenCollectionProducer& collection, std::string& error,
    std::size_t capacity, ColdRimingFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    if (capacity != 13)
        return reject(ColdRimingFailure::Capacity, "wet cold riming requires thirteen bounded source records", error, failure);
    ColdRimingProducer dry, wet;
    FrozenCollectionProducer dry_frozen, wet_frozen;
    FrozenCollectionFailure frozen_failure = FrozenCollectionFailure::None;
    const auto frozen_source = frozen_collection_source(cell);
    if (!produce_exact_cold_riming_admission(parameters, cell, duration_s, dry, error, 12, failure,
            spectrum_domain, true, true, true) ||
        !produce_exact_frozen_collection_admission(parameters, frozen_source, duration_s, dry_frozen,
            error, 5, &frozen_failure, spectrum_domain))
    {
        if (failure && frozen_failure != FrozenCollectionFailure::None)
            *failure = static_cast<ColdRimingFailure>(frozen_failure);
        return false;
    }
    wet = dry;
    wet_frozen = dry_frozen;
    if (cell.initial.frozen[3].mass_per_kg > 0.0 && cell.temperature_k - microphysics_constants::T0 > -40.0)
    {
        if (!produce_exact_cold_riming_admission(parameters, cell, duration_s, wet, error, 12, failure,
                spectrum_domain, true, true, true, true) ||
            !produce_exact_frozen_collection_admission(parameters, frozen_source, duration_s, wet_frozen,
                error, 5, &frozen_failure, spectrum_domain, true))
        {
            if (failure && frozen_failure != FrozenCollectionFailure::None)
                *failure = static_cast<ColdRimingFailure>(frozen_failure);
            return false;
        }
    }
    const auto collected = [](const ColdRimingProducer& riming, const FrozenCollectionProducer& frozen)
    {
        return std::array<double, 4>{riming.channels()[3].collection_mass_per_kg,
            riming.admitted_rain_channel(3).mass_per_kg, frozen.channels()[2].admitted_mass_increment_per_kg,
            frozen.channels()[3].admitted_mass_increment_per_kg};
    };
    HailWetGrowthProducer growth;
    if (!produce_exact_hail_wet_growth_admission(parameters, {cell, collected(dry, dry_frozen), collected(wet, wet_frozen)},
            duration_s, growth, error, 1, &frozen_failure, spectrum_domain))
    {
        if (failure) *failure = static_cast<ColdRimingFailure>(frozen_failure);
        return false;
    }
    const bool active = growth.regime() == HailWetGrowthRegime::Wet;
    auto candidate = active ? wet : dry;
    if (active)
    {
        auto& hail = candidate.channels_[3];
        hail.riming_diffusion_mass_per_kg = -candidate.riming_feedback_ * growth.retained_liquid_per_kg();
        hail.diffusion_mass_per_kg = hail.environmental_diffusion_mass_per_kg + hail.riming_diffusion_mass_per_kg;
        hail.diffusion_number_per_kg = hail.diffusion_mass_per_kg < 0.0 ? cell.initial.frozen[3].number_per_kg *
            (hail.diffusion_mass_per_kg / cell.initial.frozen[3].mass_per_kg) : 0.0;
        for (double value : values(hail))
            if (!std::isfinite(value))
                return reject(ColdRimingFailure::IntervalRejected, "wet hail feedback is unrepresentable", error, failure);
    }
    candidate.wet_growth_enabled_ = true;
    candidate.wet_growth_ = growth;
    producer = candidate;
    collection = active ? wet_frozen : dry_frozen;
    error.clear();
    if (failure) *failure = ColdRimingFailure::None;
    return true;
}

bool evaluate_exact_triple_cold_riming_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    ColdRimingResult& result, std::string& error, std::size_t capacity, ColdRimingFailure* failure,
    ColdSpectrumDomain spectrum_domain, bool homogeneous_freezing, bool rain_collection, bool rain_collision, bool wet_hail_collection)
{
    ColdRimingProducer producer;
    return produce_exact_cold_riming_admission(parameters, cell, duration_s, producer, error, capacity, failure,
            spectrum_domain, homogeneous_freezing, rain_collection, rain_collision, wet_hail_collection) &&
        phase_endpoint(producer, result, error, failure);
}

bool apply_exact_triple_cold_riming_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    const ColdRimingProducer& producer, ColdRimingResult& result, std::string& error,
    std::size_t capacity, ColdRimingFailure* failure, ColdSpectrumDomain spectrum_domain, bool homogeneous_freezing, bool rain_collection, bool rain_collision, bool wet_hail_collection)
{
    return validate_exact_cold_riming_admission(parameters, cell, duration_s, producer, error, capacity,
            failure, spectrum_domain, homogeneous_freezing, rain_collection, rain_collision, wet_hail_collection) && phase_endpoint(producer, result, error, failure);
}

bool validate_exact_cold_riming_admission(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    const ColdRimingProducer& producer, std::string& error,
    std::size_t capacity, ColdRimingFailure* failure, ColdSpectrumDomain spectrum_domain, bool homogeneous_freezing, bool rain_collection, bool rain_collision, bool wet_hail_collection)
{
    const auto& p = parameters;
    const std::array<double, 25> physical_parameters{p.a_g_, p.a_h_, p.a_i_, p.a_r_, p.a_s_,
        p.alpha_g_, p.alpha_h_, p.alpha_i_, p.alpha_r_, p.alpha_s_,
        p.b_g_, p.b_h_, p.b_i_, p.b_r_, p.b_s_, p.c_g_, p.c_h_, p.c_i_, p.c_r_, p.c_s_,
        p.d_g_, p.d_h_, p.d_i_, p.d_r_, p.d_s_};
    if (producer.event_count() != (homogeneous_freezing ? 5u : 4u) + (rain_collection ? 4u : 0u) + (rain_collision ? 3u : 0u) + (producer.wet_growth_enabled() ? 1u : 0u) ||
        producer.wet_hail_collection_enabled() != wet_hail_collection ||
        producer.rain_collection_enabled() != rain_collection ||
        producer.rain_collision_enabled() != rain_collision ||
        producer.homogeneous_freezing_enabled() != homogeneous_freezing || !same_source(producer.source(), cell) ||
        !same(producer.parameters(), physical_parameters) || producer.hail_processes() != p.hail_processes ||
        producer.spectrum_domain() != spectrum_domain ||
        !same(producer.duration_s(), duration_s))
    {
        return reject(ColdRimingFailure::InvalidProducer,
            "cold riming admission is not bound to this source, interval or parameter set", error, failure);
    }
    if (rain_collection)
    {
        FrozenCollectionFailure rain_failure = FrozenCollectionFailure::None;
        if (!validate_exact_rain_collection_admission(parameters, {frozen_collection_source(cell), cell.initial.rain},
                duration_s, producer.rain_collection(), error, 4, &rain_failure, spectrum_domain))
        {
            if (failure) *failure = static_cast<ColdRimingFailure>(rain_failure);
            return false;
        }
    }
    if (rain_collision)
    {
        FrozenCollectionFailure collision_failure = FrozenCollectionFailure::None;
        if (!validate_exact_rain_collision_admission(parameters, {frozen_collection_source(cell), cell.initial.rain},
                duration_s, producer.rain_collision(), error, 7, &collision_failure, spectrum_domain))
        {
            if (failure) *failure = static_cast<ColdRimingFailure>(collision_failure);
            return false;
        }
    }
    ColdRimingProducer expected;
    FrozenCollectionProducer frozen_capture;
    if (producer.wet_growth_enabled() && (!homogeneous_freezing || !rain_collection || !rain_collision))
        return reject(ColdRimingFailure::InvalidProducer, "wet cold capture lacks its required collision owners", error, failure);
    const bool regenerated = producer.wet_growth_enabled()
        ? produce_exact_wet_cold_riming_admission(parameters, cell, duration_s, expected, frozen_capture, error, capacity, failure, spectrum_domain)
        : produce_exact_cold_riming_admission(parameters, cell, duration_s, expected, error, capacity, failure,
            spectrum_domain, homogeneous_freezing, rain_collection, rain_collision, wet_hail_collection);
    if (!regenerated)
    {
        return false;
    }
    if (producer.wet_growth_enabled())
    {
        FrozenCollectionFailure wet_failure = FrozenCollectionFailure::None;
        if (!validate_exact_hail_wet_growth_admission(parameters, expected.wet_growth().source(), duration_s,
                producer.wet_growth(), error, 1, &wet_failure, spectrum_domain))
        {
            if (failure) *failure = static_cast<ColdRimingFailure>(wet_failure);
            return false;
        }
    }
    if (!same_capture(producer, expected))
    {
        return reject(ColdRimingFailure::InvalidProducer,
            "cold riming admission does not match source, interval, parameters or joint transfers", error, failure);
    }
    error.clear();
    if (failure)
    {
        *failure = ColdRimingFailure::None;
    }
    return true;
}

bool evaluate_exact_triple_cold_ice_graupel_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    ColdIceGraupelResult& result, std::string& error,
    std::size_t capacity, ColdRimingFailure* failure)
{
    ColdRimingProducer riming;
    if (!produce_exact_cold_riming_admission(parameters, cell, duration_s, riming, error, 4, failure))
    {
        return false;
    }
    ColdIceGraupelResult candidate;
    IceGraupelConversionFailure conversion_failure = IceGraupelConversionFailure::None;
    if (!produce_exact_ice_graupel_conversion_admission(riming, candidate.conversion, error, capacity, &conversion_failure))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(conversion_failure);
        }
        return false;
    }
    if (!phase_endpoint(riming, candidate.combined, error, failure, &candidate.conversion))
    {
        return false;
    }
    result = candidate;
    return true;
}

bool apply_exact_triple_cold_ice_graupel_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    const IceGraupelConversionProducer& producer, ColdIceGraupelResult& result,
    std::string& error, std::size_t capacity, ColdRimingFailure* failure)
{
    const auto& p = parameters;
    const std::array<double, 25> physical_parameters{p.a_g_, p.a_h_, p.a_i_, p.a_r_, p.a_s_,
        p.alpha_g_, p.alpha_h_, p.alpha_i_, p.alpha_r_, p.alpha_s_,
        p.b_g_, p.b_h_, p.b_i_, p.b_r_, p.b_s_, p.c_g_, p.c_h_, p.c_i_, p.c_r_, p.c_s_,
        p.d_g_, p.d_h_, p.d_i_, p.d_r_, p.d_s_};
    const auto& prefix = producer.riming_source();
    if (producer.event_count() != 1 || !same_source(prefix.source(), cell) ||
        !same(prefix.parameters(), physical_parameters) || prefix.hail_processes() != p.hail_processes ||
        prefix.spectrum_domain() != ColdSpectrumDomain::Base200 ||
        prefix.homogeneous_freezing_enabled() ||
        !same(prefix.duration_s(), duration_s))
    {
        return reject(ColdRimingFailure::InvalidProducer,
            "ice-to-graupel conversion is not bound to this source, interval or parameter set", error, failure);
    }
    ColdRimingProducer expected;
    if (!produce_exact_cold_riming_admission(parameters, cell, duration_s, expected, error, 4, failure))
    {
        return false;
    }
    if (!same_capture(prefix, expected))
    {
        return reject(ColdRimingFailure::InvalidProducer,
            "ice-to-graupel conversion does not match the accepted coupled riming journal", error, failure);
    }
    IceGraupelConversionProducer expected_conversion;
    IceGraupelConversionFailure conversion_failure = IceGraupelConversionFailure::None;
    if (!produce_exact_ice_graupel_conversion_admission(expected, expected_conversion, error, capacity, &conversion_failure))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(conversion_failure);
        }
        return false;
    }
    if (!same(producer.mass_increment_per_kg(), expected_conversion.mass_increment_per_kg()) ||
        !same(producer.number_increment_per_kg(), expected_conversion.number_increment_per_kg()) ||
        !same(producer.reflectivity_increment_per_kg(), expected_conversion.reflectivity_increment_per_kg()) ||
        !same(producer.raw_mass_transfer_per_kg(), expected_conversion.raw_mass_transfer_per_kg()) ||
        !same(producer.admitted_mass_transfer_per_kg(), expected_conversion.admitted_mass_transfer_per_kg()) ||
        !same(producer.donor_number_loss_per_kg(), expected_conversion.donor_number_loss_per_kg()) ||
        !same(producer.recipient_number_gain_per_kg(), expected_conversion.recipient_number_gain_per_kg()))
    {
        return reject(ColdRimingFailure::InvalidProducer,
            "ice-to-graupel conversion moment transfers differ from the accepted source journal", error, failure);
    }
    ColdIceGraupelResult candidate;
    candidate.conversion = producer;
    if (!phase_endpoint(expected, candidate.combined, error, failure, &candidate.conversion))
    {
        return false;
    }
    result = candidate;
    return true;
}

bool evaluate_exact_triple_cold_graupel_conversions_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    ColdGraupelConversionsResult& result, std::string& error,
    std::size_t capacity, ColdRimingFailure* failure, bool homogeneous_freezing)
{
    if (capacity != 2)
    {
        return reject(ColdRimingFailure::Capacity,
            "coupled graupel conversions require two channel records", error, failure);
    }
    ColdRimingProducer riming;
    if (!produce_exact_cold_riming_admission(parameters, cell, duration_s, riming, error,
            homogeneous_freezing ? 5 : 4, failure, ColdSpectrumDomain::SnowConvertedGraupel512, homogeneous_freezing))
    {
        return false;
    }
    ColdGraupelConversionsResult candidate;
    IceGraupelConversionFailure ice_failure = IceGraupelConversionFailure::None;
    SnowGraupelConversionFailure snow_failure = SnowGraupelConversionFailure::None;
    if (!produce_exact_ice_graupel_conversion_admission(riming, candidate.ice_conversion, error, 1, &ice_failure))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(ice_failure);
        }
        return false;
    }
    if (!produce_exact_snow_graupel_conversion_admission(riming, candidate.snow_conversion, error, 1, &snow_failure))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(snow_failure);
        }
        return false;
    }
    if (!phase_endpoint(riming, candidate.combined, error, failure, &candidate.ice_conversion, &candidate.snow_conversion))
    {
        return false;
    }
    result = candidate;
    return true;
}

bool apply_exact_triple_cold_graupel_conversions_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    const IceGraupelConversionProducer& ice_conversion, const SnowGraupelConversionProducer& snow_conversion,
    ColdGraupelConversionsResult& result, std::string& error,
    std::size_t capacity, ColdRimingFailure* failure, bool homogeneous_freezing)
{
    constexpr auto domain = ColdSpectrumDomain::SnowConvertedGraupel512;
    if (ice_conversion.event_count() != 1 || snow_conversion.event_count() != 1 ||
        !bound_source(parameters, cell, duration_s, ice_conversion.riming_source(), domain, homogeneous_freezing) ||
        !bound_source(parameters, cell, duration_s, snow_conversion.riming_source(), domain, homogeneous_freezing) ||
        !same_capture(ice_conversion.riming_source(), snow_conversion.riming_source()))
    {
        return reject(ColdRimingFailure::InvalidProducer,
            "graupel conversions are not bound to one accepted source, spectrum domain and interval", error, failure);
    }
    if (capacity != 2)
    {
        return reject(ColdRimingFailure::Capacity,
            "coupled graupel conversions require two channel records", error, failure);
    }
    ColdRimingProducer expected;
    if (!produce_exact_cold_riming_admission(parameters, cell, duration_s, expected, error,
            homogeneous_freezing ? 5 : 4, failure, domain, homogeneous_freezing))
    {
        return false;
    }
    ColdGraupelConversionsResult candidate;
    IceGraupelConversionFailure ice_failure = IceGraupelConversionFailure::None;
    SnowGraupelConversionFailure snow_failure = SnowGraupelConversionFailure::None;
    if (!produce_exact_ice_graupel_conversion_admission(expected, candidate.ice_conversion, error, 1, &ice_failure))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(ice_failure);
        }
        return false;
    }
    if (!produce_exact_snow_graupel_conversion_admission(expected, candidate.snow_conversion, error, 1, &snow_failure))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(snow_failure);
        }
        return false;
    }
    if (!same_conversion(ice_conversion, candidate.ice_conversion) ||
        !same_snow_conversion(snow_conversion, candidate.snow_conversion))
    {
        return reject(ColdRimingFailure::InvalidProducer,
            "graupel conversion moment transfers differ from the accepted source journal", error, failure);
    }
    if (!phase_endpoint(expected, candidate.combined, error, failure,
            &candidate.ice_conversion, &candidate.snow_conversion))
    {
        return false;
    }
    result = candidate;
    return true;
}

bool evaluate_exact_triple_cold_sll_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    ColdSllResult& result, std::string& error, std::size_t capacity, ColdRimingFailure* failure)
{
    if (capacity != 3)
    {
        return reject(ColdRimingFailure::Capacity,
            "cloud-fed SLL composition requires three conversion records", error, failure);
    }
    constexpr auto domain = ColdSpectrumDomain::GraupelAndHail512;
    ColdRimingProducer riming;
    if (!produce_exact_cold_riming_admission(parameters, cell, duration_s, riming, error, 5, failure, domain, true))
    {
        return false;
    }
    ColdSllResult candidate;
    IceGraupelConversionFailure ice_failure = IceGraupelConversionFailure::None;
    SnowGraupelConversionFailure snow_failure = SnowGraupelConversionFailure::None;
    GraupelHailConversionFailure hail_failure = GraupelHailConversionFailure::None;
    if (!produce_exact_ice_graupel_conversion_admission(riming, candidate.ice_conversion, error, 1, &ice_failure))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(ice_failure);
        }
        return false;
    }
    if (!produce_exact_snow_graupel_conversion_admission(riming, candidate.snow_conversion, error, 1, &snow_failure))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(snow_failure);
        }
        return false;
    }
    if (!produce_exact_graupel_hail_conversion_admission(parameters, cloud_sll_source(riming), duration_s,
            candidate.hail_conversion, error, 1, &hail_failure, domain))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(hail_failure);
        }
        return false;
    }
    if (!phase_endpoint(riming, candidate.combined, error, failure,
            &candidate.ice_conversion, &candidate.snow_conversion, &candidate.hail_conversion))
    {
        return false;
    }
    result = candidate;
    return true;
}

bool apply_exact_triple_cold_sll_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    const IceGraupelConversionProducer& ice_conversion, const SnowGraupelConversionProducer& snow_conversion,
    const GraupelHailConversionProducer& hail_conversion, ColdSllResult& result, std::string& error,
    std::size_t capacity, ColdRimingFailure* failure)
{
    if (capacity != 3)
    {
        return reject(ColdRimingFailure::Capacity,
            "cloud-fed SLL composition requires three conversion records", error, failure);
    }
    constexpr auto domain = ColdSpectrumDomain::GraupelAndHail512;
    const auto& riming = ice_conversion.riming_source();
    if (ice_conversion.event_count() != 1 || snow_conversion.event_count() != 1 ||
        !bound_source(parameters, cell, duration_s, riming, domain, true) ||
        !bound_source(parameters, cell, duration_s, snow_conversion.riming_source(), domain, true) ||
        !same_capture(riming, snow_conversion.riming_source()) || !bound_cloud_sll(riming, hail_conversion))
    {
        return reject(ColdRimingFailure::InvalidProducer,
            "cloud-fed SLL journals are not bound to one five-way source, accepted cloud/diffusion records and interval", error, failure);
    }
    ColdRimingProducer expected;
    if (!produce_exact_cold_riming_admission(parameters, cell, duration_s, expected, error, 5, failure, domain, true))
    {
        return false;
    }
    ColdSllResult candidate;
    IceGraupelConversionFailure ice_failure = IceGraupelConversionFailure::None;
    SnowGraupelConversionFailure snow_failure = SnowGraupelConversionFailure::None;
    GraupelHailConversionFailure hail_failure = GraupelHailConversionFailure::None;
    if (!produce_exact_ice_graupel_conversion_admission(expected, candidate.ice_conversion, error, 1, &ice_failure))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(ice_failure);
        }
        return false;
    }
    if (!produce_exact_snow_graupel_conversion_admission(expected, candidate.snow_conversion, error, 1, &snow_failure))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(snow_failure);
        }
        return false;
    }
    if (!same_conversion(ice_conversion, candidate.ice_conversion) ||
        !same_snow_conversion(snow_conversion, candidate.snow_conversion))
    {
        return reject(ColdRimingFailure::InvalidProducer,
            "cloud-fed SLL ice/snow conversion records differ from the accepted original source", error, failure);
    }
    if (!validate_exact_graupel_hail_conversion_admission(parameters, cloud_sll_source(expected), duration_s,
            hail_conversion, error, 1, &hail_failure, domain))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(hail_failure);
        }
        return false;
    }
    candidate.hail_conversion = hail_conversion;
    if (!phase_endpoint(expected, candidate.combined, error, failure,
            &candidate.ice_conversion, &candidate.snow_conversion, &candidate.hail_conversion))
    {
        return false;
    }
    result = candidate;
    return true;
}

namespace
{

bool prepare_collected_sll(const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
                           ColdRimingProducer& riming, FrozenCollectionIncrements& increments,
                           ColdCollectedSllResult& candidate, std::string& error, ColdRimingFailure* failure,
                           bool rain_collection = false, bool rain_collision = false, bool wet_growth = false)
{
    constexpr auto domain = ColdSpectrumDomain::GraupelAndHail512;
    const bool admitted = wet_growth
        ? produce_exact_wet_cold_riming_admission(parameters, cell, duration_s, riming, candidate.collection, error, 13, failure, domain)
        : produce_exact_cold_riming_admission(parameters, cell, duration_s, riming, error,
            rain_collision ? 12 : rain_collection ? 9 : 5, failure, domain, true, rain_collection, rain_collision);
    if (!admitted)
    {
        return false;
    }
    const auto frozen_source = frozen_collection_source(cell);
    FrozenCollectionFailure collection_failure = FrozenCollectionFailure::None;
    if ((!wet_growth && !produce_exact_frozen_collection_admission(parameters, frozen_source, duration_s,
            candidate.collection, error, 5, &collection_failure, domain)) ||
        !evaluate_exact_frozen_collection_increments(parameters, frozen_source, duration_s,
            candidate.collection, increments, error, 5, &collection_failure, domain, candidate.collection.wet_hail_collection()))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(collection_failure);
        }
        return false;
    }
    IceGraupelConversionFailure ice_failure = IceGraupelConversionFailure::None;
    if (!produce_exact_ice_graupel_conversion_admission(riming, candidate.ice_conversion, error, 1, &ice_failure))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(ice_failure);
        }
        return false;
    }
    SnowGraupelConversionFailure snow_failure = SnowGraupelConversionFailure::None;
    if (!produce_exact_snow_graupel_conversion_admission(riming, candidate.snow_conversion, error, 1, &snow_failure))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(snow_failure);
        }
        return false;
    }
    GraupelHailConversionFailure hail_failure = GraupelHailConversionFailure::None;
    if (!produce_exact_graupel_hail_conversion_admission(parameters, cloud_sll_source(riming, &candidate.collection),
            duration_s, candidate.hail_conversion, error, 1, &hail_failure, domain))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(hail_failure);
        }
        return false;
    }
    return original_collected_donors(riming, candidate, error, failure);
}

} // namespace

bool evaluate_exact_triple_cold_collected_sll_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    ColdCollectedSllResult& result, std::string& error, std::size_t capacity, ColdRimingFailure* failure)
{
    if (capacity != 8)
    {
        return reject(ColdRimingFailure::Capacity,
            "collected SLL composition requires five collection and three conversion records", error, failure);
    }
    ColdRimingProducer riming;
    FrozenCollectionIncrements increments;
    ColdCollectedSllResult candidate;
    if (!prepare_collected_sll(parameters, cell, duration_s, riming, increments, candidate, error, failure) ||
        !phase_endpoint(riming, candidate.combined, error, failure, &candidate.ice_conversion,
            &candidate.snow_conversion, &candidate.hail_conversion, &increments))
    {
        return false;
    }
    result = candidate;
    return true;
}

static bool apply_collected_sll(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    const FrozenCollectionProducer& collection, const IceGraupelConversionProducer& ice_conversion,
    const SnowGraupelConversionProducer& snow_conversion, const GraupelHailConversionProducer& hail_conversion,
    ColdCollectedSllResult& result, std::string& error, std::size_t capacity, ColdRimingFailure* failure, bool rain_collection, bool rain_collision = false, bool wet_growth = false)
{
    if (capacity != (wet_growth ? 16u : rain_collision ? 15u : rain_collection ? 12u : 8u))
    {
        return reject(ColdRimingFailure::Capacity,
            wet_growth ? "wet growth composition requires sixteen collection, conversion and thermal records" :
            rain_collision ? "collision SLL composition requires twelve collection and three conversion records" :
            rain_collection ? "rain SLL composition requires nine collection and three conversion records" :
                "collected SLL composition requires five collection and three conversion records", error, failure);
    }
    constexpr auto domain = ColdSpectrumDomain::GraupelAndHail512;
    const auto& captured_riming = ice_conversion.riming_source();
    if (ice_conversion.event_count() != 1 || snow_conversion.event_count() != 1 ||
        !bound_source(parameters, cell, duration_s, captured_riming, domain, true, rain_collection, rain_collision, wet_growth) ||
        !bound_source(parameters, cell, duration_s, snow_conversion.riming_source(), domain, true, rain_collection, rain_collision, wet_growth) ||
        collection.wet_hail_collection() != captured_riming.wet_hail_collection_enabled() ||
        !same_capture(captured_riming, snow_conversion.riming_source()) ||
        !bound_cloud_sll(captured_riming, hail_conversion, &collection))
    {
        return reject(ColdRimingFailure::InvalidProducer,
            "collected SLL journals differ from the original source, accepted ice accretion or interval", error, failure);
    }
    if (rain_collection &&
        (!validate_exact_cold_riming_admission(parameters, cell, duration_s, captured_riming, error,
            wet_growth ? 13 : rain_collision ? 12 : 9, failure, domain, true, true, rain_collision, captured_riming.wet_hail_collection_enabled()) ||
         !validate_exact_cold_riming_admission(parameters, cell, duration_s, snow_conversion.riming_source(), error,
            wet_growth ? 13 : rain_collision ? 12 : 9, failure, domain, true, true, rain_collision, captured_riming.wet_hail_collection_enabled()))) return false;
    FrozenCollectionFailure collection_failure = FrozenCollectionFailure::None;
    if (!validate_exact_frozen_collection_admission(parameters, frozen_collection_source(cell), duration_s,
            collection, error, 5, &collection_failure, domain, captured_riming.wet_hail_collection_enabled()))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(collection_failure);
        }
        return false;
    }
    ColdRimingProducer riming;
    FrozenCollectionIncrements increments;
    ColdCollectedSllResult candidate;
    if (!prepare_collected_sll(parameters, cell, duration_s, riming, increments, candidate, error, failure, rain_collection, rain_collision, wet_growth))
    {
        return false;
    }
    if (!same_conversion(ice_conversion, candidate.ice_conversion) ||
        !same_snow_conversion(snow_conversion, candidate.snow_conversion))
    {
        return reject(ColdRimingFailure::InvalidProducer,
            "collected SLL conversion records differ from the accepted source journals", error, failure);
    }
    GraupelHailConversionFailure hail_failure = GraupelHailConversionFailure::None;
    if (!validate_exact_graupel_hail_conversion_admission(parameters, cloud_sll_source(riming, &candidate.collection),
            duration_s, hail_conversion, error, 1, &hail_failure, domain))
    {
        if (failure)
        {
            *failure = static_cast<ColdRimingFailure>(hail_failure);
        }
        return false;
    }
    if (!phase_endpoint(riming, candidate.combined, error, failure, &candidate.ice_conversion,
            &candidate.snow_conversion, &candidate.hail_conversion, &increments))
    {
        return false;
    }
    result = candidate;
    return true;
}


bool apply_exact_triple_cold_collected_sll_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    const FrozenCollectionProducer& collection, const IceGraupelConversionProducer& ice_conversion,
    const SnowGraupelConversionProducer& snow_conversion, const GraupelHailConversionProducer& hail_conversion,
    ColdCollectedSllResult& result, std::string& error, std::size_t capacity, ColdRimingFailure* failure)
{
    return apply_collected_sll(parameters, cell, duration_s, collection, ice_conversion, snow_conversion,
        hail_conversion, result, error, capacity, failure, false);
}

bool evaluate_exact_triple_cold_rain_sll_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    ColdCollectedSllResult& result, std::string& error, std::size_t capacity, ColdRimingFailure* failure)
{
    if (capacity != 12)
    {
        return reject(ColdRimingFailure::Capacity,
            "rain SLL composition requires nine collection and three conversion records", error, failure);
    }
    ColdRimingProducer riming;
    FrozenCollectionIncrements increments;
    ColdCollectedSllResult candidate;
    if (!prepare_collected_sll(parameters, cell, duration_s, riming, increments, candidate, error, failure, true) ||
        !phase_endpoint(riming, candidate.combined, error, failure, &candidate.ice_conversion,
            &candidate.snow_conversion, &candidate.hail_conversion, &increments)) return false;
    result = candidate;
    return true;
}

bool apply_exact_triple_cold_rain_sll_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    const FrozenCollectionProducer& collection, const IceGraupelConversionProducer& ice_conversion,
    const SnowGraupelConversionProducer& snow_conversion, const GraupelHailConversionProducer& hail_conversion,
    ColdCollectedSllResult& result, std::string& error, std::size_t capacity, ColdRimingFailure* failure)
{
    return apply_collected_sll(parameters, cell, duration_s, collection, ice_conversion, snow_conversion,
        hail_conversion, result, error, capacity, failure, true);
}

bool evaluate_exact_triple_cold_collision_sll_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    ColdCollectedSllResult& result, std::string& error, std::size_t capacity, ColdRimingFailure* failure)
{
    if (capacity != 15)
    {
        return reject(ColdRimingFailure::Capacity,
            "rain SLL composition requires twelve collection and three conversion records", error, failure);
    }
    ColdRimingProducer riming;
    FrozenCollectionIncrements increments;
    ColdCollectedSllResult candidate;
    if (!prepare_collected_sll(parameters, cell, duration_s, riming, increments, candidate, error, failure, true, true) ||
        !phase_endpoint(riming, candidate.combined, error, failure, &candidate.ice_conversion,
            &candidate.snow_conversion, &candidate.hail_conversion, &increments)) return false;
    result = candidate;
    return true;
}

bool apply_exact_triple_cold_collision_sll_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    const FrozenCollectionProducer& collection, const IceGraupelConversionProducer& ice_conversion,
    const SnowGraupelConversionProducer& snow_conversion, const GraupelHailConversionProducer& hail_conversion,
    ColdCollectedSllResult& result, std::string& error, std::size_t capacity, ColdRimingFailure* failure)
{
    return apply_collected_sll(parameters, cell, duration_s, collection, ice_conversion, snow_conversion,
        hail_conversion, result, error, capacity, failure, true, true);
}

bool evaluate_exact_triple_cold_wet_growth_sll_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    ColdCollectedSllResult& result, std::string& error, std::size_t capacity, ColdRimingFailure* failure)
{
    if (capacity != 16)
    {
        return reject(ColdRimingFailure::Capacity,
            "wet growth composition requires sixteen collection, conversion and thermal records", error, failure);
    }
    ColdRimingProducer riming;
    FrozenCollectionIncrements increments;
    ColdCollectedSllResult candidate;
    if (!prepare_collected_sll(parameters, cell, duration_s, riming, increments, candidate, error, failure, true, true, true) ||
        !phase_endpoint(riming, candidate.combined, error, failure, &candidate.ice_conversion,
            &candidate.snow_conversion, &candidate.hail_conversion, &increments)) return false;
    result = candidate;
    return true;
}

bool apply_exact_triple_cold_wet_growth_sll_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    const FrozenCollectionProducer& collection, const IceGraupelConversionProducer& ice_conversion,
    const SnowGraupelConversionProducer& snow_conversion, const GraupelHailConversionProducer& hail_conversion,
    ColdCollectedSllResult& result, std::string& error, std::size_t capacity, ColdRimingFailure* failure)
{
    return apply_collected_sll(parameters, cell, duration_s, collection, ice_conversion, snow_conversion,
        hail_conversion, result, error, capacity, failure, true, true, true);
}

} // namespace tmv::milbrandt_detail
