#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_diffusion.hpp"
#include "microphysics/schemes/milbrandt/triple_sedimentation.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace
{

using namespace tmv::milbrandt_detail;

TripleMomentMassLaw law(ColdFrozenDiffusionSpecies species)
{
    return current_nimbus_triple_sedimentation_parameters(
        species == ColdFrozenDiffusionSpecies::Graupel
            ? TripleMomentSpecies::Graupel : TripleMomentSpecies::Hail).mass_law;
}

ColdParameters parameters(ColdFrozenDiffusionSpecies species, double alpha)
{
    const auto& p = default_cold_parameters;
    ColdParameters result{
        p[0], p[1], p[2], p[3], p[4],
        p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14],
        p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], true};
    (species == ColdFrozenDiffusionSpecies::Graupel
        ? result.alpha_g_ : result.alpha_h_) = alpha;
    return result;
}

long double reference_z(long double q, long double n, long double c,
                        long double alpha)
{
    const long double g = (alpha + 4) * (alpha + 5) * (alpha + 6)
        / ((alpha + 1) * (alpha + 2) * (alpha + 3));
    return g * q * q / (c * c * n);
}

ColdFrozenDiffusionTripleState state(ColdFrozenDiffusionSpecies species,
                                    double alpha = 2.0,
                                    float q = 5.0e-4f,
                                    float n = 10.0f)
{
    return {q, n, static_cast<float>(reference_z(q, n, law(species).coefficient, alpha))};
}

double diagnosed_alpha(const ColdFrozenDiffusionTripleState& s,
                       ColdFrozenDiffusionSpecies species)
{
    double alpha = 0.0;
    double lambda = 0.0;
    std::string error;
    REQUIRE(diagnose_triple_gamma_shape(s.mass_per_kg, s.number_per_kg,
        s.reflectivity_per_kg, law(species), alpha, lambda, error));
    return alpha;
}

std::array<double, 16> cell(ColdFrozenDiffusionSpecies species,
                           const ColdFrozenDiffusionTripleState& s,
                           double density = .65, double pressure = 90000,
                           double saturation = .75)
{
    std::array<double, 16> result{};
    result[0] = 253.15;
    result[1] = pressure;
    result[2] = density;
    result[3] = saturation * thermodynamics::saturation_mixing_ratio_ice(result[0], pressure);
    result[species == ColdFrozenDiffusionSpecies::Graupel ? 8 : 9] = s.mass_per_kg;
    result[species == ColdFrozenDiffusionSpecies::Graupel ? 14 : 15] = s.number_per_kg;
    return result;
}

long double numerical_gamma_integral(long double order)
{
    constexpr int count = 32768;
    constexpr long double end = 160.0L;
    constexpr long double width = end / count;
    long double sum = 0.0L;
    for (int index = 1; index < count; ++index)
    {
        const long double x = index * width;
        sum += (index % 2 ? 4.0L : 2.0L) * std::pow(x, order) * std::exp(-x);
    }
    sum += std::pow(end, order) * std::exp(-end);
    return sum * width / 3.0L;
}

long double reference_increment(const std::array<double, 16>& input,
                                ColdFrozenDiffusionSpecies species,
                                long double alpha, long double duration)
{
    const auto mass_law = law(species);
    const auto p = current_nimbus_triple_sedimentation_parameters(
        species == ColdFrozenDiffusionSpecies::Graupel
            ? TripleMomentSpecies::Graupel : TripleMomentSpecies::Hail);
    const long double t = input[0];
    const long double tc = t - microphysics_constants::T0;
    const long double pressure = input[1];
    const long double density = input[2];
    const long double vapor = input[3];
    const long double q = input[species == ColdFrozenDiffusionSpecies::Graupel ? 8 : 9];
    const long double n = input[species == ColdFrozenDiffusionSpecies::Graupel ? 14 : 15];
    const long double qsi = thermodynamics::saturation_mixing_ratio_ice(input[0], input[1]);
    const long double diffusivity = (2.2157e-5L + 0.0155e-5L * tc) * 1.0e5L / pressure;
    const long double conductivity = 2.3971e-2L + 0.0078e-2L * tc;
    const long double viscosity = 1.72e-5L * 393.0L / (t + 120.0L)
        * std::pow(t / microphysics_constants::T0, 1.5L) / density;
    const long double lambda = std::cbrt(
        mass_law.coefficient * n * std::tgamma(alpha + 4) / (q * std::tgamma(alpha + 1)));
    const long double intercept = density * n * std::pow(lambda, alpha + 1)
        / std::tgamma(alpha + 1);
    const long double ventilation = .78L * numerical_gamma_integral(alpha + 1)
        / std::pow(lambda, alpha + 2)
        + .30L * std::cbrt(viscosity / diffusivity)
        * std::sqrt(std::sqrt(1.225L / density) * p.fall_coefficient / viscosity)
        * numerical_gamma_integral(alpha + 1.5L + .5L * p.fall_exponent)
        / std::pow(lambda, alpha + 2.5L + .5L * p.fall_exponent);
    const long double resistance = microphysics_constants::L_s * microphysics_constants::L_s
        / (conductivity * microphysics_constants::R_v * t * t)
        + 1.0L / (density * qsi * diffusivity);
    long double increment = duration / density * 2.0L * std::acos(-1.0L)
        * (vapor / qsi - 1.0L) * intercept * ventilation / resistance;
    if (vapor >= qsi)
    {
        const long double limit = (vapor - qsi) /
            (1.0L + microphysics_constants::L_s * microphysics_constants::L_s * qsi /
                (microphysics_constants::cp * microphysics_constants::R_v * (t - 7.66L) * (t - 7.66L)));
        increment = std::min(std::max(increment, 0.0L), std::max(limit, 0.0L));
    }
    else
    {
        increment = std::min(std::max(increment, -q), 0.0L);
    }
    return increment;
}

} // namespace

TEST_CASE("published frozen diffusion observer agrees with independent ventilation and B1 B3",
          "[physics][milbrandt][triple_cold_diffusion]")
{
    for (auto species : {ColdFrozenDiffusionSpecies::Graupel, ColdFrozenDiffusionSpecies::Hail})
    {
        for (double nominal_alpha : {0.0, 2.0, 20.0})
        {
            const auto initial = state(species, nominal_alpha);
            const double alpha = diagnosed_alpha(initial, species);
            for (double density : {.65, 1.3})
            {
                for (double pressure : {70000.0, 90000.0})
                {
                    for (double saturation : {.75, 1.0, 1.03})
                    {
                        for (double duration : {.05, 2.0})
                        {
                            const auto input = cell(species, initial, density, pressure, saturation);
                            const long double dq = reference_increment(input, species, alpha, duration);
                            const long double dn = dq < 0.0L ? initial.number_per_kg * dq / initial.mass_per_kg : 0.0L;
                            const long double dz = initial.reflectivity_per_kg *
                                (2.0L * dq / initial.mass_per_kg - dn / initial.number_per_kg);
                            std::array<float, 14> rates{};
                            ColdFrozenDiffusionAcceptedEventBuffer observer;
                            observer.species = species;
                            REQUIRE_NOTHROW(published_frozen_diffusion_cold_cell(
                                parameters(species, alpha), input, duration, rates, observer));
                            REQUIRE(observer.size == 1);
                            const auto& event = observer.events[0];
                            CHECK(event.entry_mass_per_kg == initial.mass_per_kg);
                            CHECK(event.entry_number_per_kg == initial.number_per_kg);
                            CHECK(event.source_alpha == alpha);
                            CHECK(event.dry_density_kg_m3 == density);
                            CHECK(event.duration_s == duration);
                            CHECK(event.admitted_mass_increment_per_kg == Approx(static_cast<double>(dq)).epsilon(3e-6).margin(1e-22));
                            CHECK(event.admitted_number_increment_per_kg == Approx(static_cast<double>(dn)).epsilon(3e-6).margin(1e-22));
                            ColdFrozenDiffusionTripleEndpoint endpoint;
                            std::string error;
                            INFO(error);
                            REQUIRE(apply_published_frozen_diffusion(initial, density, duration,
                                event, law(species), endpoint, error));
                            CHECK(endpoint.mass_per_kg == Approx(static_cast<double>(initial.mass_per_kg + dq)).epsilon(4e-7));
                            CHECK(endpoint.number_per_kg == Approx(static_cast<double>(initial.number_per_kg + dn)).epsilon(4e-7));
                            CHECK(endpoint.reflectivity_per_kg == Approx(static_cast<double>(initial.reflectivity_per_kg + dz)).epsilon(4e-7));
                            CHECK(rates[6] == Approx(static_cast<double>(-dq / duration)).epsilon(3e-6).margin(1e-22));
                            CHECK(rates[7] == Approx(static_cast<double>(microphysics_constants::L_s / microphysics_constants::cp * dq / duration)).epsilon(3e-6).margin(1e-22));
                            const std::size_t q_index = species == ColdFrozenDiffusionSpecies::Graupel ? 4 : 5;
                            const std::size_t n_index = species == ColdFrozenDiffusionSpecies::Graupel ? 12 : 13;
                            for (std::size_t index = 0; index < rates.size(); ++index)
                            {
                                if (index != q_index && index != n_index && index != 6 && index != 7)
                                {
                                    CHECK(rates[index] == 0.0f);
                                }
                            }
                            CHECK(rates[q_index] == -rates[6]);
                            CHECK(rates[n_index] == Approx(static_cast<double>(dn / duration)).epsilon(3e-6).margin(1e-22));
                            CHECK(density * endpoint.mass_per_kg == Approx(static_cast<double>(density * (initial.mass_per_kg + dq))).epsilon(4e-7));
                            CHECK(density * endpoint.number_per_kg == Approx(static_cast<double>(density * (initial.number_per_kg + dn))).epsilon(4e-7));
                            CHECK(density * endpoint.reflectivity_per_kg == Approx(static_cast<double>(density * (initial.reflectivity_per_kg + dz))).epsilon(4e-7));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("published B3 diffusion derivative has independent fixed-alpha finite difference",
          "[physics][milbrandt][triple_cold_diffusion]")
{
    for (auto species : {ColdFrozenDiffusionSpecies::Graupel, ColdFrozenDiffusionSpecies::Hail})
    {
        for (double nominal_alpha : {0.0, 2.0, 20.0})
        {
            const auto initial = state(species, nominal_alpha);
            const double alpha = diagnosed_alpha(initial, species);
            for (double fraction : {-.5, -.125, 0.0, .125, .5})
            {
                const double dq = fraction * initial.mass_per_kg;
                const double dn = dq < 0.0 ? fraction * initial.number_per_kg : 0.0;
                const auto event = make_published_frozen_diffusion_event(species,
                    initial.mass_per_kg, initial.number_per_kg, alpha, dq, dn, .65, .2);
                ColdFrozenDiffusionTripleEndpoint endpoint;
                std::string error;
                REQUIRE(apply_published_frozen_diffusion(initial, .65, .2, event, law(species), endpoint, error));
                constexpr long double epsilon = 1e-5L;
                const long double plus = reference_z(initial.mass_per_kg + epsilon * dq,
                    initial.number_per_kg + epsilon * dn, law(species).coefficient, alpha);
                const long double minus = reference_z(initial.mass_per_kg - epsilon * dq,
                    initial.number_per_kg - epsilon * dn, law(species).coefficient, alpha);
                const long double increment = (plus - minus) / (2 * epsilon);
                CHECK(endpoint.reflectivity_per_kg == Approx(static_cast<double>(initial.reflectivity_per_kg + increment)).epsilon(4e-7));
            }
        }
    }
}

TEST_CASE("published sublimation drains all q N Z by algebra and differs only in triple number policy",
          "[physics][milbrandt][triple_cold_diffusion]")
{
    for (auto species : {ColdFrozenDiffusionSpecies::Graupel, ColdFrozenDiffusionSpecies::Hail})
    {
        const auto initial = state(species, 2.0, 1e-7f, .01f);
        const double alpha = diagnosed_alpha(initial, species);
        const auto input = cell(species, initial, .65, 90000, 0.0);
        std::array<float, 14> legacy{};
        cold_cell(parameters(species, alpha), input, 1048576.0, legacy);
        const std::size_t n_index = species == ColdFrozenDiffusionSpecies::Graupel ? 12 : 13;
        CHECK(legacy[n_index] == 0.0f);
        std::array<float, 14> rates{};
        ColdFrozenDiffusionAcceptedEventBuffer observer;
        observer.species = species;
        REQUIRE_NOTHROW(published_frozen_diffusion_cold_cell(parameters(species, alpha),
            input, 1048576.0, rates, observer));
        REQUIRE(observer.size == 1);
        const auto& event = observer.events[0];
        CHECK(event.admitted_mass_increment_per_kg == -initial.mass_per_kg);
        CHECK(event.admitted_number_increment_per_kg == -initial.number_per_kg);
        ColdFrozenDiffusionTripleEndpoint result{42, 42, 42};
        std::string error;
        REQUIRE(apply_published_frozen_diffusion(initial, .65, 1048576.0,
            event, law(species), result, error));
        CHECK(result.mass_per_kg == 0.0f);
        CHECK(result.number_per_kg == 0.0f);
        CHECK(result.reflectivity_per_kg == 0.0f);
        CHECK(rates[n_index] < 0.0f);
        for (std::size_t index = 0; index < rates.size(); ++index)
        {
            if (index != n_index)
            {
                CHECK(rates[index] == legacy[index]);
            }
        }
    }
}

TEST_CASE("published diffusion rejects invalid event provenance and endpoints without publication",
          "[physics][milbrandt][triple_cold_diffusion]")
{
    const auto species = ColdFrozenDiffusionSpecies::Graupel;
    const auto initial = state(species);
    const double alpha = diagnosed_alpha(initial, species);
    for (int mutation = 0; mutation < 14; ++mutation)
    {
        auto source = initial;
        auto mass_law = law(species);
        auto event = make_published_frozen_diffusion_event(species,
            source.mass_per_kg, source.number_per_kg, alpha,
            -.25 * source.mass_per_kg, -.25 * source.number_per_kg, .65, .2);
        switch (mutation)
        {
            case 0: event.entry_mass_per_kg = std::nextafter(event.entry_mass_per_kg, 1.0); break;
            case 1: event.entry_number_per_kg = std::nextafter(event.entry_number_per_kg, 100.0); break;
            case 2: event.source_alpha = std::nextafter(alpha, 100.0); break;
            case 3: event.dry_density_kg_m3 = 1.0; break;
            case 4: event.duration_s = .3; break;
            case 5: event.admitted_number_increment_per_kg = 0.0; break;
            case 6: event.admitted_mass_increment_per_kg = -2 * source.mass_per_kg; break;
            case 7: event.admitted_number_increment_per_kg = .25 * source.number_per_kg; break;
            case 8: mass_law.exponent = 2.078; break;
            case 9: source.reflectivity_per_kg = 0.0; break;
            case 10: source.number_per_kg = std::numeric_limits<double>::quiet_NaN(); break;
            case 11: event.admitted_mass_increment_per_kg = 1e9 * source.mass_per_kg; event.admitted_number_increment_per_kg = 0.0; break;
            case 12: event.species = static_cast<ColdFrozenDiffusionSpecies>(42); break;
            case 13: event.admitted_mass_increment_per_kg = std::numeric_limits<double>::infinity(); break;
        }
        ColdFrozenDiffusionTripleEndpoint result{.0125f, 23.0f, .001f};
        const auto before = result;
        const auto source_before = source;
        std::string error;
        CHECK_FALSE(apply_published_frozen_diffusion(source, .65, .2,
            event, mass_law, result, error));
        CHECK_FALSE(error.empty());
        CHECK(std::memcmp(&result, &before, sizeof(result)) == 0);
        CHECK(std::memcmp(&source, &source_before, sizeof(source)) == 0);
    }
}

TEST_CASE("published diffusion validates zero events and named unowned channels atomically",
          "[physics][milbrandt][triple_cold_diffusion]")
{
    const auto species = ColdFrozenDiffusionSpecies::Graupel;
    const auto initial = state(species);
    const double alpha = diagnosed_alpha(initial, species);
    for (int mutation = 0; mutation < 11; ++mutation)
    {
        auto input = cell(species, initial, .65, 90000, 1.0);
        auto p = parameters(species, alpha);
        std::array<float, 14> rates{};
        ColdFrozenDiffusionAcceptedEventBuffer observer;
        observer.species = species;
        switch (mutation)
        {
            case 0: input[2] = 0.0; break;
            case 1: input[1] = 0.0; break;
            case 2: input[0] = std::numeric_limits<double>::quiet_NaN(); break;
            case 3: input[3] = -1.0; break;
            case 4: p.c_g_ = 100.0; break;
            case 5: observer.capacity = 0; break;
            case 6: observer.capacity = 2; break;
            case 7: input[3] *= 2.0; break;
            case 8: input[9] = 1e-7; input[15] = 100.0; break;
            case 9: input[5] = 1e-4; input[11] = 1000.0; break;
            case 10: rates[12] = 1.0f; break;
        }
        const auto before = rates;
        const auto observer_before = observer;
        CHECK_THROWS(published_frozen_diffusion_cold_cell(p, input, .2, rates, observer));
        CHECK(rates == before);
        CHECK(std::memcmp(&observer, &observer_before, sizeof(observer)) == 0);
    }
    auto input = cell(species, initial);
    std::array<float, 14> rates;
    rates.fill(42.0f);
    ColdFrozenDiffusionTripleEndpoint endpoint{42, 42, 42};
    const auto before = endpoint;
    input[3] *= 4.0;
    std::string error;
    CHECK_FALSE(evaluate_published_triple_frozen_diffusion_cell(
        input, .2, species, initial, rates, endpoint, error));
    for (float rate : rates)
    {
        CHECK(rate == 42.0f);
    }
    CHECK(std::memcmp(&endpoint, &before, sizeof(endpoint)) == 0);
}
