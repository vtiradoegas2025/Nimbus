#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/triple_hail_surface_condensation.hpp"
#include "support/milbrandt_frozen_collection_reference.hpp"
#include "core/runtime/simulation_state.hpp"

#include <thread>
#include <iomanip>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
using namespace tmv::milbrandt_detail;
namespace spectral = milbrandt_frozen_collection_reference;

HailSurfaceCondensationInput source(double temperature = 268.15, float vapor = .015f,
                                    long double shape = 2, bool rain = true)
{
    HailSurfaceCondensationInput input;
    auto& state = input.before;
    state.thermal.pressure = 85000.f;
    state.thermal.density = static_cast<float>(85000 / (287 * temperature));
    state.thermal.theta = static_cast<float>(temperature / std::pow(.85, 287.0 / 1004));
    state.thermal.water = {vapor, .001f, rain ? .001f : 0.f, 1e-6f, 1e-5f, 1e-4f, .001f};
    state.hail_number = 20.f;
    state.hail_reflectivity = static_cast<float>(spectral::population(3, state.thermal.water[6], state.hail_number, shape).reflectivity_per_kg);
    if (rain)
    {
        state.rain_number = 4000.f;
        state.rain_reflectivity = static_cast<float>(5.6L * state.thermal.water[2] * state.thermal.water[2] /
            (523.598775598L * 523.598775598L * state.rain_number));
    }
    input.generation = 1;
    input.duration_s = .01;
    return input;
}

long double temperature(const SourceEosEnergyState& state)
{
    return state.theta * std::pow(state.pressure / 100000.L, 287.L / 1004.L);
}

struct Reference
{
    long double ventilation, condensation, conduction, melting;
};

Reference reference(const HailSurfaceCondensationInput& input, bool quadrature = false)
{
    const auto& state = input.before;
    const long double t = temperature(state.thermal), tc = t - 273.15L;
    const long double rho = state.thermal.density, p = state.thermal.pressure;
    const long double diffusivity = (2.2157e-5L + .0155e-5L * tc) * 1e5L / p;
    const long double conductivity = .023971L + .000078L * tc;
    const long double viscosity = 1.72e-5L * 393 / (t + 120) * std::pow(t / 273.15L, 1.5L) / rho;
    const auto s = spectral::spectrum({state.thermal.water[6], state.hail_number, state.hail_reflectivity}, 3, rho);
    const auto moment = [&](long double k)
    {
        return quadrature ? spectral::quadrature_moment(s, k) : spectral::moment(s, k);
    };
    const long double ventilation = .78L * moment(1) + .30L * std::cbrt(viscosity / diffusivity) *
        std::sqrt(std::sqrt(1.225L / rho) * 206.89L / viscosity) * moment(1.5L + .5L * .6384L);
    const long double b = input.duration_s * 2 * std::acos(-1.L) * state.hail_number * ventilation;
    const long double saturation = std::clamp(.62198L * 611.21L / std::max(p - 611.21L, 1.L), 0.L, .2L);
    const long double c = b * rho * diffusivity * (state.thermal.water[0] - saturation);
    const long double h = b * conductivity * tc;
    long double lo = 0, hi = state.thermal.water[6];
    for (int iteration = 0; iteration < 100; ++iteration)
    {
        const auto mass = (lo + hi) / 2;
        const long double incoming = h + 2500000.L * c + mass * (2106.L * tc - 334000.L);
        if (incoming > 0) lo = mass;
        else hi = mass;
    }
    return {ventilation, c, h, (lo + hi) / 2};
}

template <class T>
long double inventory(long double t, const std::array<T, 7>& q)
{
    long double ice = 0;
    for (std::size_t i = 3; i < q.size(); ++i) ice += q[i];
    return 717.L * t + (t - 273.15L) * (1388.5L * q[0] + 4186.L * (static_cast<long double>(q[1]) + q[2]) + 2106.L * ice)
        + q[0] * (2500000.L - 461.5L * 273.15L) - 334000.L * ice;
}

void near(double actual, long double expected, long double floor = 1e-25L)
{
    CAPTURE(actual, expected);
    CHECK(std::abs(actual - expected) < 3e-8L * std::max(std::abs(expected), floor));
}

HailSurfaceCondensationTransaction prepared(const HailSurfaceCondensationInput& input)
{
    HailSurfaceCondensationTransaction result;
    std::string error;
    const bool ok = result.prepare(input, error);
    INFO(error);
    REQUIRE(ok);
    return result;
}

using IntegratedState = std::array<long double, 7>;

IntegratedState integrated_reference(const HailSurfaceCondensationInput& input, long double duration, int steps)
{
    const auto& initial = input.before;
    const long double rho = initial.thermal.density;
    const long double eos_pressure = 100000.L * std::pow(rho * 287.L * initial.thermal.theta / 100000.L, 1004.L / 717.L);
    const long double t0 = initial.thermal.theta * std::pow(eos_pressure / 100000.L, 287.L / 1004.L);
    const long double total_energy = inventory(t0, initial.thermal.water);
    const auto spectrum = spectral::spectrum({initial.thermal.water[6], initial.hail_number, initial.hail_reflectivity}, 3, rho);
    const long double first = spectral::moment(spectrum, 1), ventilated = spectral::moment(spectrum, 1.5L + .5L * .6384L);
    const long double rain_c = 523.598775598L, drop_mass = rain_c * 1e-9L;
    IntegratedState value{initial.thermal.water[0], initial.thermal.water[2], initial.thermal.water[6],
        initial.rain_number, initial.rain_reflectivity, initial.hail_number, initial.hail_reflectivity};
    const auto derivative = [&](const IntegratedState& x)
    {
        const long double frozen = initial.thermal.water[3] + static_cast<long double>(initial.thermal.water[4]) +
            initial.thermal.water[5] + x[2];
        const long double liquid = initial.thermal.water[1] + x[1];
        const long double cv = 717 + 1388.5L * x[0] + 4186 * liquid + 2106 * frozen;
        const long double phase = (2500000.L - 461.5L * 273.15L) * x[0] - 334000 * frozen;
        const long double t = 273.15L + (total_energy - 717 * 273.15L - phase) / cv;
        const long double tc = t - 273.15L, pressure = rho * 287 * t;
        const long double diffusivity = (2.2157e-5L + .0155e-5L * tc) * 1e5L / pressure;
        const long double conductivity = .023971L + .000078L * tc;
        const long double viscosity = 1.72e-5L * 393 / (t + 120) * std::pow(t / 273.15L, 1.5L) / rho;
        const long double vent = .78L * first + .30L * std::cbrt(viscosity / diffusivity) *
            std::sqrt(std::sqrt(1.225L / rho) * 206.89L / viscosity) * ventilated;
        const long double b = 2 * std::acos(-1.L) * x[5] * vent;
        const long double sat = .62198L * 611.21L / (pressure - 611.21L);
        const long double condensation = b * rho * diffusivity * (x[0] - sat);
        const long double melt = (b * conductivity * tc + 2500000 * condensation) / (334000 - 2106 * tc);
        const long double number = x[5] * melt / x[2], z = x[6] * melt / x[2];
        return IntegratedState{-condensation, condensation + melt, -melt, number + condensation / drop_mass,
            z * std::pow(471.238898038L / rain_c, 2) + 5.6L * condensation * drop_mass / (rain_c * rain_c), -number, -z};
    };
    const long double dt = duration / steps;
    for (int step = 0; step < steps; ++step)
    {
        const auto a = derivative(value);
        const auto stage = [&](const IntegratedState& slope, long double fraction)
        {
            auto next = value;
            for (std::size_t i = 0; i < next.size(); ++i) next[i] += dt * fraction * slope[i];
            return derivative(next);
        };
        const auto b = stage(a, .5L), c = stage(b, .5L), d = stage(c, 1);
        for (std::size_t i = 0; i < value.size(); ++i) value[i] += dt / 6 * (a[i] + 2 * b[i] + 2 * c[i] + d[i]);
    }
    return value;
}
}

TEST_CASE("condensing hail surface closes independent heat mass and moment references",
          "[physics][milbrandt][triple_hail_surface_condensation]")
{
    for (double t : {272.15, 268.15, 253.15})
    for (float vapor : {.015f, .04f})
    for (long double shape : {2.L, 20.L, 350.L})
    for (bool rain : {false, true})
    {
        const auto input = source(t, vapor, shape, rain);
        const auto actual = prepared(input);
        const auto expected = reference(input);
        const auto r = actual.receipt();
        const auto q = actual.energy_input().accepted_water;
        const auto& before = input.before;
        near(r.normalized_ventilation, expected.ventilation);
        near(r.condensed_mass, expected.condensation);
        near(r.conductive_heat, expected.conduction);
        near(r.melted_mass, expected.melting);
        near(r.melted_number, before.hail_number * expected.melting / before.thermal.water[6]);
        near(r.hail_reflectivity_removed, before.hail_reflectivity * expected.melting / before.thermal.water[6]);
        near(r.melted_rain_reflectivity, r.hail_reflectivity_removed * std::pow(471.238898038L / 523.598775598L, 2));
        near(r.condensate_rain_number, expected.condensation / (523.598775598L * 1e-9L));
        near(r.condensate_rain_reflectivity, r.condensate_rain_number * 5.6e-18L);
        near(r.material_warming_heat + r.melting_heat, expected.conduction + 2500000.L * expected.condensation);
        CHECK(r.melted_mass < (r.condensation_heat + r.conductive_heat) / 334000.0);
        long double water = 0;
        for (std::size_t i = 0; i < q.size(); ++i) water += static_cast<long double>(q[i]) - before.thermal.water[i];
        CHECK(std::abs(water) < 2e-17L);
        const auto e = actual.energy_receipt();
        const auto ideal_before = inventory(e.temperature_eos_reference, before.thermal.water);
        const auto ideal_after = inventory(e.temperature_eos_target, q);
        CHECK(std::abs(ideal_after - ideal_before) < 2e-7L);
        const long double source_t = temperature(before.thermal), target_t = e.temperature_eos_target;
        long double original_ice = 0;
        for (std::size_t i = 3; i < before.thermal.water.size(); ++i) original_ice += before.thermal.water[i];
        const long double remaining_cp = 1004 + 1850.L * (before.thermal.water[0] - r.condensed_mass) +
            4186.L * (static_cast<long double>(before.thermal.water[1]) + before.thermal.water[2]) +
            2106.L * (original_ice - r.melted_mass);
        const long double vapor_to_interface = -1850.L * (source_t - 273.15L) * r.condensed_mass;
        const long double liquid_return = 4186.L * (target_t - 273.15L) * (r.condensed_mass + r.melted_mass);
        const long double enthalpy_path = vapor_to_interface + r.conductive_heat + liquid_return +
            remaining_cp * (target_t - source_t);
        const long double gas_pv_change = (287.L + 461.5L * q[0]) * target_t -
            (287.L + 461.5L * before.thermal.water[0]) * source_t;
        CHECK(std::abs(enthalpy_path - gas_pv_change - e.reference_projection_work - e.ideal_phase_energy_change) < 2e-7L);
        const auto end = actual.eos_endpoint().thermal;
        const auto stored = inventory(temperature(end), end.water) - inventory(temperature(before.thermal), before.thermal.water);
        CHECK(std::abs(stored - e.committed_internal_energy_change) < 2e-7L);
    }
    const auto input = source(268.15, .015f, 350, false);
    const auto quadrature = reference(input, true);
    const auto actual = prepared(input);
    near(actual.receipt().normalized_ventilation, quadrature.ventilation);
    near(actual.receipt().condensed_mass, quadrature.condensation);
}

TEST_CASE("condensing hail keeps melt and condensate rain populations separate and owns no new hail",
          "[physics][milbrandt][triple_hail_surface_condensation]")
{
    const auto input = source(268.15, .015f, 2, false);
    const auto actual = prepared(input);
    const auto r = actual.receipt();
    REQUIRE(r.melted_mass > 0);
    REQUIRE(r.condensed_mass > 0);
    const double wrong_all_shed = 5.6 * actual.energy_input().accepted_water[2] * 1e-9 / 523.598775598;
    CHECK(std::abs(wrong_all_shed - r.rain_reflectivity_after) > .1 * r.rain_reflectivity_after);
    CHECK(r.rain_number_after == r.melted_number + r.condensate_rain_number);
    CHECK(r.rain_reflectivity_after == r.melted_rain_reflectivity + r.condensate_rain_reflectivity);
    near(r.hail_number_after / input.before.hail_number,
        actual.energy_input().accepted_water[6] / input.before.thermal.water[6]);
    near(r.hail_reflectivity_after / input.before.hail_reflectivity,
        actual.energy_input().accepted_water[6] / input.before.thermal.water[6]);
    auto absent = input;
    absent.before.thermal.water[6] = absent.before.hail_number = absent.before.hail_reflectivity = 0;
    const auto no_hail = prepared(absent);
    CHECK(no_hail.receipt().condensed_mass == 0);
    CHECK(no_hail.receipt().melted_mass == 0);
    CHECK(no_hail.source_endpoint().thermal.water == absent.before.thermal.water);
    CHECK(no_hail.energy_receipt().theta_increment == 0);
}

TEST_CASE("condensing hail unowned processes donor exhaustion and malformed moments reject atomically",
          "[physics][milbrandt][triple_hail_surface_condensation]")
{
    for (int mutation = 0; mutation < 14; ++mutation)
    {
        auto input = source();
        if (mutation == 0) input.admitted_collection[0] = 1e-8;
        if (mutation == 1) input.admitted_collection[2] = 1e-8;
        if (mutation == 2) input.dry_hail_diffusion = 1e-8;
        if (mutation == 3) input.before.thermal.water[0] = .001f;
        if (mutation == 4) input.before.thermal.water[0] = .0046f;
        if (mutation == 5) input.duration_s = 1e6;
        if (mutation == 6) input.before.hail_number = 0;
        if (mutation == 7) input.before.rain_reflectivity = std::numeric_limits<float>::quiet_NaN();
        if (mutation == 8) input.generation = 0;
        if (mutation == 9) input.duration_s = 0;
        if (mutation == 10) input = source(220.15);
        if (mutation == 11) input = source(280.15);
        if (mutation == 12) input.admitted_collection[3] = std::numeric_limits<double>::infinity();
        if (mutation == 13) input.duration_s = std::numeric_limits<double>::denorm_min();
        HailSurfaceCondensationTransaction transaction;
        std::string error;
        CHECK_FALSE(transaction.prepare(input, error));
        CHECK(transaction.stage() == SourceEosEnergyTransaction::Stage::Empty);
        CHECK(transaction.receipt().condensed_mass == 0.0);
    }
}

TEST_CASE("condensing hail exact original donor exhaustion has a zero hail endpoint and rejects overdraw",
          "[physics][milbrandt][triple_hail_surface_condensation]")
{
    bool found = false;
    for (double t : {272.15, 268.15, 253.15})
    {
        auto input = source(t, .015f, 2, false);
        const auto initial = prepared(input);
        const double estimate = input.duration_s * input.before.thermal.water[6] / initial.receipt().melted_mass;
        double lower = estimate, upper = estimate;
        for (int neighbor = 0; neighbor < 64 && !found; ++neighbor)
        {
            for (double duration : {lower, upper})
            {
                input.duration_s = duration;
                HailSurfaceCondensationTransaction transaction;
                std::string error;
                if (!transaction.prepare(input, error) || transaction.receipt().melted_mass != input.before.thermal.water[6]) continue;
                found = true;
                CHECK(transaction.source_endpoint().thermal.water[6] == 0.0f);
                CHECK(transaction.source_endpoint().hail_number == 0.0f);
                CHECK(transaction.source_endpoint().hail_reflectivity == 0.0f);
                CHECK(transaction.receipt().melted_number == input.before.hail_number);
                input.duration_s *= 1.001;
                HailSurfaceCondensationTransaction overdraw;
                CHECK_FALSE(overdraw.prepare(input, error));
                CHECK(overdraw.stage() == SourceEosEnergyTransaction::Stage::Empty);
                break;
            }
            lower = std::nextafter(lower, 0.0);
            upper = std::nextafter(upper, std::numeric_limits<double>::infinity());
        }
        if (found) break;
    }
    REQUIRE(found);
}

TEST_CASE("condensing hail cold source cannot cross the staged or post EOS thermal domain",
          "[physics][milbrandt][triple_hail_surface_condensation]")
{
    auto post_eos = source();
    post_eos.before.thermal.density *= 1.08f;
    HailSurfaceCondensationTransaction projected;
    std::string error;
    CHECK_FALSE(projected.prepare(post_eos, error));
    CHECK(error.find("cold/warm process coupling") != std::string::npos);
    CHECK(projected.stage() == SourceEosEnergyTransaction::Stage::Empty);

    auto staged = source(272.15, .0051f);
    staged.before.thermal.density = .65f;
    staged.duration_s = 1.0;
    const auto rate = reference(staged);
    const long double saturation = .62198L * 611.21L / (85000.L - 611.21L);
    staged.duration_s = static_cast<double>(.99L * (staged.before.thermal.water[0] - saturation) / rate.condensation);
    HailSurfaceCondensationTransaction intermediate;
    CHECK_FALSE(intermediate.prepare(staged, error));
    CHECK(error.find("cold/warm process coupling") != std::string::npos);
    CHECK(intermediate.stage() == SourceEosEnergyTransaction::Stage::Empty);
}

TEST_CASE("condensing hail moment and thermal generations bind the actual production EOS",
          "[physics][milbrandt][triple_hail_surface_condensation]")
{
    auto input = source(268.15, .015f, 20, false);
    auto transaction = prepared(input);
    std::string error;
    auto stage = transaction.source_endpoint();
    auto bad = stage;
    bad.rain_number = std::nextafter(bad.rain_number, 1e9f);
    CHECK_FALSE(transaction.accept_source(input.before, bad, input.generation, error));
    CHECK(transaction.stage() == SourceEosEnergyTransaction::Stage::Prepared);
    CHECK_FALSE(transaction.accept_source(input.before, stage, input.generation + 1, error));
    REQUIRE(transaction.accept_source(input.before, stage, input.generation, error));
    CHECK_FALSE(transaction.accept_source(input.before, stage, input.generation, error));
    SimulationStorageState storage{{5, 3, 4, 4.0, 5.0, .5, 6.0}};
    storage.grid.geometry.initialize(5, 3, 4, 4.0, 5.0, 6.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& f = storage.fields;
    for (Field3D* field : {&f.u, &f.v, &f.w, &f.rho, &f.p, &f.theta, &f.qv, &f.qc,
            &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.tke}) field->resize(5, 3, 4, 0.f);
    f.rho.fill(stage.thermal.density); f.p.fill(stage.thermal.pressure); f.theta.fill(stage.thermal.theta);
    const std::array<Field3D*, 7> water{&f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh};
    for (std::size_t i = 0; i < water.size(); ++i) water[i]->fill(stage.thermal.water[i]);
    REQUIRE(tmv::dynamics_eos::apply_pressure_projection(make_simulation_state_view(storage)).complete);
    auto after = stage;
    after.thermal.pressure = f.p(1, 1, 0);
    bad = after;
    bad.hail_reflectivity = std::nextafter(bad.hail_reflectivity, 1.f);
    CHECK_FALSE(transaction.accept_eos(stage, bad, input.generation, error));
    bad = stage;
    bad.thermal.theta = std::nextafter(bad.thermal.theta, 400.f);
    CHECK_FALSE(transaction.accept_eos(bad, after, input.generation, error));
    CHECK(transaction.stage() == SourceEosEnergyTransaction::Stage::SourceCommitted);
    REQUIRE(transaction.accept_eos(stage, after, input.generation, error));
    CHECK(transaction.stage() == SourceEosEnergyTransaction::Stage::EosCommitted);
}

TEST_CASE("condensing hail continuation is deterministic and accounts for stored energy without hidden state",
          "[physics][milbrandt][triple_hail_surface_condensation]")
{
    const auto initial = source(268.15, .015f, 20, false);
    std::array<HailSurfaceCondensationTransaction, 4> parallel;
    std::array<std::thread, 4> workers;
    std::array<bool, 4> success{};
    for (std::size_t i = 0; i < workers.size(); ++i)
        workers[i] = std::thread([&, i] { std::string error; success[i] = parallel[i].prepare(initial, error); });
    for (auto& worker : workers) worker.join();
    for (std::size_t i = 0; i < parallel.size(); ++i)
    {
        REQUIRE(success[i]);
        CHECK(parallel[i].receipt().condensed_mass == parallel[0].receipt().condensed_mass);
        CHECK(parallel[i].receipt().rain_reflectivity_after == parallel[0].receipt().rain_reflectivity_after);
        CHECK(parallel[i].energy_receipt().theta_increment == parallel[0].energy_receipt().theta_increment);
    }
    auto next = initial;
    long double energy = 0;
    for (int step = 0; step < 30; ++step)
    {
        auto transaction = prepared(next);
        std::string error;
        REQUIRE(transaction.accept_source(next.before, transaction.source_endpoint(), next.generation, error));
        REQUIRE(transaction.accept_eos(transaction.source_endpoint(), transaction.eos_endpoint(), next.generation, error));
        energy += transaction.energy_receipt().committed_internal_energy_change;
        next.before = transaction.eos_endpoint();
        ++next.generation;
    }
    const auto observed = inventory(temperature(next.before.thermal), next.before.thermal.water) -
        inventory(temperature(initial.before.thermal), initial.before.thermal.water);
    CHECK(std::abs(observed - energy) < 1e-6L);
}

TEST_CASE("condensing hail stored source EOS continuation converges to an independent coupled RK4 integration",
          "[physics][milbrandt][triple_hail_surface_condensation]")
{
    auto initial = source(253.15, .04f, 2, true);
    initial.before.thermal.water[6] = .02f;
    initial.before.hail_number = 400.f;
    initial.before.hail_reflectivity = static_cast<float>(spectral::population(3,
        initial.before.thermal.water[6], initial.before.hail_number, 2).reflectivity_per_kg);
    const auto oracle = integrated_reference(initial, 100, 4096);
    const auto refined_oracle = integrated_reference(initial, 100, 8192);
    CHECK(std::abs(oracle[2] - refined_oracle[2]) < 1e-12L);
    std::array<long double, 3> error{};
    for (int level = 0; level < 3; ++level)
    {
        auto next = initial;
        const int steps = 10 << level;
        next.duration_s = 100.0 / steps;
        for (int step = 0; step < steps; ++step)
        {
            auto transaction = prepared(next);
            std::string reason;
            REQUIRE(transaction.accept_source(next.before, transaction.source_endpoint(), next.generation, reason));
            REQUIRE(transaction.accept_eos(transaction.source_endpoint(), transaction.eos_endpoint(), next.generation, reason));
            next.before = transaction.eos_endpoint();
            ++next.generation;
        }
        error[level] = std::abs(next.before.thermal.water[6] - oracle[2]);
    }
    CAPTURE(error[0], error[1], error[2]);
    std::cout << std::setprecision(17) << "[Hail surface refinement] hail_mass_errors="
        << error[0] << ',' << error[1] << ',' << error[2] << " kg/kg_dry" << std::endl;
    CHECK(error[1] < .8L * error[0]);
    CHECK(error[2] < .8L * error[1]);
}
