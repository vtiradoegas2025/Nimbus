#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/triple_source_eos_energy.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/runtime/simulation_state.hpp"

#include <algorithm>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
using namespace tmv::milbrandt_detail;

SourceEosEnergyInput input(float rho = .8f, float theta = 310.f, double pressure_factor = 1.0)
{
    SourceEosEnergyInput result;
    result.before = {rho, static_cast<float>(pressure_factor * tmv::dynamics_eos::pressure_fp64(rho, theta)), theta,
        {.006f, .005f, .001f, .000002f, .00002f, .0002f, .001f}};
    std::copy(result.before.water.begin(), result.before.water.end(), result.accepted_water.begin());
    result.generation = 19;
    result.duration_s = .01;
    return result;
}

template <class T>
long double internal_energy(long double temperature, const std::array<T, 7>& q)
{
    const long double vapor = q[0], liquid = static_cast<long double>(q[1]) + q[2];
    long double ice = 0;
    for (std::size_t index = 3; index < q.size(); ++index) ice += q[index];
    return 717.L * temperature + (temperature - 273.15L) *
        (1388.5L * vapor + 4186.L * liquid + 2106.L * ice) +
        vapor * (2500000.L - 461.5L * 273.15L) - ice * 334000.L;
}

long double eos_temperature(long double rho, long double theta)
{
    const auto pressure = 100000.L * std::pow(rho * 287.L * theta / 100000.L, 1004.L / 717.L);
    return theta * std::pow(pressure / 100000.L, 287.L / 1004.L);
}

long double inverse_reference(const SourceEosEnergyInput& source)
{
    const auto target = internal_energy(eos_temperature(source.before.density, source.before.theta), source.before.water);
    long double lo = 200, hi = 700;
    for (int iteration = 0; iteration < 100; ++iteration)
    {
        const auto middle = (lo + hi) / 2;
        const auto inventory = internal_energy(eos_temperature(source.before.density, middle), source.accepted_water);
        if (inventory < target) lo = middle;
        else hi = middle;
    }
    return (lo + hi) / 2;
}

long double stored_temperature(const SourceEosEnergyState& state)
{
    return static_cast<long double>(state.theta) * std::pow(state.pressure / 100000.L, 287.L / 1004.L);
}

SourceEosEnergyState sample(const PrognosticFieldStorage& f)
{
    return {f.rho(1, 1, 0), f.p(1, 1, 0), f.theta(1, 1, 0),
        {f.qv(1, 1, 0), f.qc(1, 1, 0), f.qr(1, 1, 0), f.qi(1, 1, 0),
         f.qs(1, 1, 0), f.qg(1, 1, 0), f.qh(1, 1, 0)}};
}

SourceEosEnergyTransaction prepared(const SourceEosEnergyInput& source)
{
    SourceEosEnergyTransaction transaction;
    std::string error;
    REQUIRE(transaction.prepare(source, error));
    INFO(error);
    return transaction;
}

void transfer(SourceEosEnergyInput& source, std::size_t from, std::size_t to, double amount)
{
    source.accepted_water[from] -= amount;
    source.accepted_water[to] += amount;
}

void compare_inventory(const SourceEosEnergyInput& source, const SourceEosEnergyTransaction& transaction)
{
    const auto& receipt = transaction.receipt();
    const auto target = inverse_reference(source);
    const long double proposed = static_cast<long double>(source.before.theta) + receipt.theta_increment;
    CHECK(std::abs(proposed - target) < 2e-10L);
    const auto initial_eos_energy = internal_energy(eos_temperature(source.before.density, source.before.theta), source.before.water);
    const auto final_eos_energy = internal_energy(eos_temperature(source.before.density, proposed), source.accepted_water);
    CHECK(std::abs(final_eos_energy - initial_eos_energy) < 2e-7L);
    CHECK(std::abs(receipt.ideal_phase_energy_change) < 1e-9);
    const auto observed = internal_energy(stored_temperature(transaction.eos_endpoint()), transaction.eos_endpoint().water) -
        internal_energy(stored_temperature(source.before), source.before.water);
    CHECK(std::abs(observed - receipt.committed_internal_energy_change) < 2e-7L);
    const auto water_rounding = internal_energy(receipt.temperature_eos_target, transaction.eos_endpoint().water) -
        internal_energy(receipt.temperature_eos_target, source.accepted_water);
    CHECK(std::abs(water_rounding - receipt.water_rounding_work) < 2e-9L);
}
}

TEST_CASE("triple source EOS mapping matches an independent full-inventory root across phase transfers",
          "[physics][milbrandt][triple_source_eos_energy]")
{
    for (float density : {.5f, .8f, 1.1f})
    for (float theta : {280.f, 310.f, 340.f})
    for (double pressure_factor : {.8, 1., 1.1})
    for (int process = 0; process < 5; ++process)
    {
        auto source = input(density, theta, pressure_factor);
        if (process == 0) transfer(source, 0, 1, 1e-4);
        if (process == 1) transfer(source, 1, 0, 1e-4);
        if (process == 2) transfer(source, 1, 6, 1e-3);
        if (process == 3) transfer(source, 6, 2, 2e-4);
        if (process == 4)
        {
            transfer(source, 0, 2, 2e-5);
            transfer(source, 6, 2, 1e-4);
        }
        CAPTURE(density, theta, pressure_factor, process);
        const auto transaction = prepared(source);
        compare_inventory(source, transaction);
        CHECK(transaction.stage() == SourceEosEnergyTransaction::Stage::Prepared);
    }
}

TEST_CASE("triple source EOS zero transfers preserve theta and isolate the original pressure mismatch",
          "[physics][milbrandt][triple_source_eos_energy]")
{
    for (double pressure_factor : {.8, 1., 1.1})
    {
        auto source = input(.8f, 310.f, pressure_factor);
        const auto unchanged = prepared(source);
        CHECK(unchanged.receipt().theta_increment == 0.0);
        CHECK(unchanged.source_endpoint().theta == source.before.theta);
        CHECK(unchanged.source_endpoint().water == source.before.water);
        CHECK(unchanged.receipt().stored_water_increment == 0.0);
        compare_inventory(source, unchanged);
        transfer(source, 0, 6, 1e-5);
        const auto a = prepared(source);
        source.before.pressure = input().before.pressure;
        const auto b = prepared(source);
        CHECK(a.receipt().theta_increment == b.receipt().theta_increment);
        CHECK(a.eos_endpoint().theta == b.eos_endpoint().theta);
        if (pressure_factor != 1.0)
            CHECK(a.receipt().reference_projection_work != b.receipt().reference_projection_work);
    }
}

TEST_CASE("triple source EOS sub ULP heating exposes stored energy change instead of declaring it conserved",
          "[physics][milbrandt][triple_source_eos_energy]")
{
    auto source = input();
    transfer(source, 1, 6, 1e-9);
    const auto transaction = prepared(source);
    CHECK(transaction.receipt().theta_increment > 0.0);
    CHECK(transaction.source_endpoint().theta == source.before.theta);
    CHECK(transaction.eos_endpoint().water != source.before.water);
    CHECK(std::abs(transaction.receipt().theta_pressure_rounding_work) > 1e-5);
    CHECK(std::abs(transaction.receipt().committed_internal_energy_change) > 1e-5);
    compare_inventory(source, transaction);
}

TEST_CASE("triple source EOS receipt validates actual pressure projection and reports storage work",
          "[physics][milbrandt][triple_source_eos_energy]")
{
    auto source = input(.8f, 310.f, .93);
    transfer(source, 0, 2, 1e-5);
    transfer(source, 6, 2, 1e-4);
    tmv::dynamics_eos::PressureProjectionEnergyReportV1 serial_report;
    std::vector<float> serial_pressure;
#ifdef _OPENMP
    struct ThreadRestore
    {
        int count = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~ThreadRestore() { omp_set_num_threads(count); omp_set_dynamic(dynamic); }
    } thread_restore;
    omp_set_dynamic(0);
#endif
    for (int workers : {1, 4})
    {
#ifdef _OPENMP
        omp_set_num_threads(workers);
        int actual_workers = 0;
#pragma omp parallel reduction(+:actual_workers)
        { ++actual_workers; }
        CHECK(actual_workers == workers);
#endif
        auto transaction = prepared(source);
        SimulationStorageState storage{{5, 3, 4, 4.0, 5.0, .5, 6.0}};
        storage.grid.geometry.initialize(5, 3, 4, 4.0, 5.0, 6.0,
            CoordinateSystem::Cartesian, StaggerType::CGrid);
        auto& f = storage.fields;
        for (Field3D* field : {&f.u, &f.v, &f.w, &f.rho, &f.p, &f.theta,
                &f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.tke})
            field->resize(5, 3, 4, 0.f);
        const auto stage = transaction.source_endpoint();
        f.rho.fill(stage.density);
        f.theta.fill(static_cast<float>(source.before.theta + transaction.receipt().theta_increment));
        f.p.fill(stage.pressure);
        const std::array<Field3D*, 7> water{&f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh};
        for (std::size_t index = 0; index < water.size(); ++index) water[index]->fill(stage.water[index]);
        std::string error;
        REQUIRE(transaction.accept_source(source.before, sample(f), source.generation, error));
        const auto view = make_simulation_state_view(storage);
        const auto report = tmv::dynamics_eos::apply_pressure_projection(view);
        REQUIRE(report.complete);
        const auto after = sample(f);
        REQUIRE(transaction.accept_eos(stage, after, source.generation, error));
        const auto domain = physical_scalar_domain(view.grid);
        const auto cells = (domain.i_end - domain.i_begin) * (domain.j_end - domain.j_begin) * (domain.k_end - domain.k_begin);
        const auto eos_energy = internal_energy(stored_temperature(after), after.water) -
            internal_energy(stored_temperature(stage), stage.water);
        const auto expected = cells * static_cast<long double>(stage.density) * 4 * 5 * 6 * eos_energy;
        CHECK(std::abs(report.committed_energy_change_j - expected) < 1e-6L);
        CHECK(report.closure_residual_j == 0.0);
        CHECK(transaction.stage() == SourceEosEnergyTransaction::Stage::EosCommitted);
        CHECK(transaction.receipt().water_rounding_work != 0.0);
        CHECK(transaction.receipt().theta_pressure_rounding_work != 0.0);
        compare_inventory(source, transaction);
        if (workers == 1)
        {
            serial_report = report;
            serial_pressure.assign(f.p.data(), f.p.data() + f.p.size());
        }
        else
        {
            CHECK(report.dry_internal_change_j == serial_report.dry_internal_change_j);
            CHECK(report.water_sensible_change_j == serial_report.water_sensible_change_j);
            CHECK(report.committed_energy_change_j == serial_report.committed_energy_change_j);
            CHECK(std::memcmp(serial_pressure.data(), f.p.data(), f.p.size() * sizeof(float)) == 0);
        }
    }
}

TEST_CASE("triple source EOS rejection leaves pressure and the pending energy owner recoverable",
          "[physics][milbrandt][triple_source_eos_energy]")
{
    auto source = input(.8f, 310.f, .93);
    transfer(source, 0, 6, 1e-5);
    auto transaction = prepared(source);
    SimulationStorageState storage{{5, 3, 4, 4.0, 5.0, .5, 6.0}};
    storage.grid.geometry.initialize(5, 3, 4, 4.0, 5.0, 6.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& f = storage.fields;
    for (Field3D* field : {&f.u, &f.v, &f.w, &f.rho, &f.p, &f.theta,
            &f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.tke})
        field->resize(5, 3, 4, 0.f);
    const auto stage = transaction.source_endpoint();
    f.rho.fill(stage.density);
    f.theta.fill(stage.theta);
    f.p.fill(stage.pressure);
    const std::array<Field3D*, 7> water{&f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh};
    for (std::size_t index = 0; index < water.size(); ++index) water[index]->fill(stage.water[index]);
    std::string error;
    REQUIRE(transaction.accept_source(source.before, sample(f), source.generation, error));
    const Field3D pressure_before = f.p;
    f.rho(3, 1, 2) = -1.f;
    REQUIRE_THROWS(tmv::dynamics_eos::apply_pressure_projection(make_simulation_state_view(storage)));
    CHECK(std::memcmp(f.p.data(), pressure_before.data(), f.p.size() * sizeof(float)) == 0);
    CHECK_FALSE(transaction.accept_eos(stage, sample(f), source.generation, error));
    CHECK(transaction.stage() == SourceEosEnergyTransaction::Stage::SourceCommitted);
    f.rho(3, 1, 2) = stage.density;
    REQUIRE(tmv::dynamics_eos::apply_pressure_projection(make_simulation_state_view(storage)).complete);
    REQUIRE(transaction.accept_eos(stage, sample(f), source.generation, error));
    CHECK(transaction.stage() == SourceEosEnergyTransaction::Stage::EosCommitted);
}

TEST_CASE("triple source EOS stage and generation binding rejects intervening work atomically",
          "[physics][milbrandt][triple_source_eos_energy]")
{
    auto source = input();
    transfer(source, 0, 6, 1e-5);
    for (int mutation = 0; mutation < 12; ++mutation)
    {
        auto transaction = prepared(source);
        auto before = source.before, stage = transaction.source_endpoint(), endpoint = transaction.eos_endpoint();
        std::string error;
        CHECK_FALSE(transaction.accept_eos(stage, endpoint, source.generation, error));
        CHECK(transaction.stage() == SourceEosEnergyTransaction::Stage::Prepared);
        if (mutation < 6)
        {
            if (mutation == 0) before.theta = std::nextafter(before.theta, 400.f);
            if (mutation == 1) stage.density = std::nextafter(stage.density, 2.f);
            if (mutation == 2) stage.water[0] = std::nextafter(stage.water[0], 1.f);
            if (mutation == 3) stage.pressure = std::nextafter(stage.pressure, 200000.f);
            if (mutation == 4) stage.theta = std::nextafter(stage.theta, 400.f);
            CHECK_FALSE(transaction.accept_source(before, stage, source.generation + (mutation == 5), error));
            CHECK(transaction.stage() == SourceEosEnergyTransaction::Stage::Prepared);
            stage = transaction.source_endpoint();
        }
        REQUIRE(transaction.accept_source(source.before, transaction.source_endpoint(), source.generation, error));
        CHECK_FALSE(transaction.accept_source(source.before, stage, source.generation, error));
        if (mutation == 6) stage.theta = std::nextafter(stage.theta, 400.f);
        if (mutation == 7) stage.water[6] = std::nextafter(stage.water[6], 1.f);
        if (mutation == 8) endpoint.density = std::nextafter(endpoint.density, 2.f);
        if (mutation == 9) endpoint.pressure = std::nextafter(endpoint.pressure, 200000.f);
        if (mutation == 10) endpoint.water[2] = std::nextafter(endpoint.water[2], 1.f);
        if (mutation >= 6)
        {
            CHECK_FALSE(transaction.accept_eos(stage, endpoint, source.generation + (mutation == 11), error));
            CHECK(transaction.stage() == SourceEosEnergyTransaction::Stage::SourceCommitted);
        }
        REQUIRE(transaction.accept_eos(transaction.source_endpoint(), transaction.eos_endpoint(), source.generation, error));
        CHECK_FALSE(transaction.accept_eos(transaction.source_endpoint(), transaction.eos_endpoint(), source.generation, error));
        CHECK(transaction.stage() == SourceEosEnergyTransaction::Stage::EosCommitted);
    }
}

TEST_CASE("triple source EOS rejects invalid inputs without creating a partial receipt",
          "[physics][milbrandt][triple_source_eos_energy]")
{
    for (int mutation = 0; mutation < 13; ++mutation)
    {
        auto source = input();
        if (mutation == 0) source.duration_s = 0;
        if (mutation == 1) source.generation = 0;
        if (mutation == 2) source.before.density = .01f;
        if (mutation == 3) source.before.pressure = 500.f;
        if (mutation == 4) source.before.theta = std::numeric_limits<float>::quiet_NaN();
        if (mutation == 5) source.accepted_water[1] += 1e-5;
        if (mutation == 6) source.accepted_water[1] = -1e-5;
        if (mutation == 7) source.accepted_water[1] = std::numeric_limits<double>::infinity();
        if (mutation == 8) source.accepted_water[1] = .1;
        if (mutation == 9) source.accepted_water[1] = std::numeric_limits<float>::denorm_min() * .5;
        if (mutation == 10) source.before.density = 2.f, source.before.theta = 700.f;
        if (mutation >= 11)
        {
            transfer(source, 0, 6, 1e-5);
            source.duration_s = mutation == 11 ? std::numeric_limits<double>::max() : std::numeric_limits<double>::min();
        }
        SourceEosEnergyTransaction transaction;
        std::string error;
        CHECK_FALSE(transaction.prepare(source, error));
        CHECK(transaction.stage() == SourceEosEnergyTransaction::Stage::Empty);
        CHECK(transaction.receipt().theta_increment == 0.0);
    }
    auto source = input();
    auto transaction = prepared(source);
    std::string error;
    source.generation++;
    CHECK_FALSE(transaction.prepare(source, error));
    CHECK(transaction.input().generation == 19);
}

TEST_CASE("triple source EOS mapping is deterministic and telescopes through continuation",
          "[physics][milbrandt][triple_source_eos_energy]")
{
    auto source = input();
    transfer(source, 1, 6, 1e-4);
    std::array<SourceEosEnergyTransaction, 4> transactions;
    std::array<std::thread, 4> workers;
    std::array<bool, 4> accepted{};
    for (std::size_t index = 0; index < workers.size(); ++index)
        workers[index] = std::thread([&, index] { std::string error; accepted[index] = transactions[index].prepare(source, error); });
    for (auto& worker : workers) worker.join();
    for (std::size_t index = 0; index < transactions.size(); ++index)
    {
        REQUIRE(accepted[index]);
        CHECK(transactions[index].receipt().theta_increment == transactions[0].receipt().theta_increment);
        CHECK(transactions[index].receipt().committed_internal_energy_change == transactions[0].receipt().committed_internal_energy_change);
    }
    const auto initial = source.before;
    long double recorded = 0;
    for (int step = 0; step < 40; ++step)
    {
        std::copy(source.before.water.begin(), source.before.water.end(), source.accepted_water.begin());
        transfer(source, step % 2 ? 6 : 1, step % 2 ? 1 : 6, 1e-5);
        auto transaction = prepared(source);
        std::string error;
        REQUIRE(transaction.accept_source(source.before, transaction.source_endpoint(), source.generation, error));
        REQUIRE(transaction.accept_eos(transaction.source_endpoint(), transaction.eos_endpoint(), source.generation, error));
        recorded += transaction.receipt().committed_internal_energy_change;
        source.before = transaction.eos_endpoint();
        ++source.generation;
    }
    const auto observed = internal_energy(stored_temperature(source.before), source.before.water) -
        internal_energy(stored_temperature(initial), initial.water);
    CHECK(std::abs(observed - recorded) < 1e-6L);
}
