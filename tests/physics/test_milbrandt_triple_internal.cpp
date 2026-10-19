#include "catch2/catch.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "microphysics/schemes/milbrandt/milbrandt.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_adapter.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

using tmv::milbrandt_detail::TripleMomentSpecies;

float independent_z(float q, float n, double c, double d,
                    double alpha = 2.0)
{
    if (q == 0.0f && n == 0.0f)
    {
        return 0.0f;
    }
    const double lambda = std::pow(
        c * n * std::tgamma(alpha + d + 1.0)
            / (q * std::tgamma(alpha + 1.0)),
        1.0 / d);
    return static_cast<float>(
        n * std::tgamma(alpha + 7.0)
            / (std::tgamma(alpha + 1.0) * std::pow(lambda, 6.0)));
}

bool same_field_bytes(const Field3D& a, const Field3D& b)
{
    return a.size_r() == b.size_r() && a.size_th() == b.size_th()
        && a.size_z() == b.size_z() && a.size() == b.size()
        && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct Inputs
{
    Field3D density{1, 1, 2, 1.0f};
    std::array<Field3D, 5> mass;
    std::array<Field3D, 5> number;
    std::array<Field3D, 5> z;

    Inputs(int nr = 1, int nth = 1, int nz = 2)
    {
        density.resize(nr, nth, nz, 1.0f);
        for (std::size_t species = 0; species < 5; ++species)
        {
            mass[species].resize(nr, nth, nz, 0.0f);
            number[species].resize(nr, nth, nz, 0.0f);
            z[species].resize(nr, nth, nz, 0.0f);
        }
    }

    void seed_column(int i, int j)
    {
        for (int k = 0; k < density.size_z(); ++k)
        {
            density(i, j, k) = 1.0f + 0.1f * k;
            for (std::size_t species = 0; species < 5; ++species)
            {
                const auto p = tmv::milbrandt_detail::
                    current_nimbus_triple_sedimentation_parameters(
                        static_cast<TripleMomentSpecies>(species));
                mass[species](i, j, k) =
                    static_cast<float>((k + 1) * (species + 1) * 1.0e-6);
                number[species](i, j, k) =
                    static_cast<float>((k + 1) * (species + 1) * 10000.0);
                z[species](i, j, k) = independent_z(
                    mass[species](i, j, k), number[species](i, j, k),
                    p.mass_law.coefficient, p.mass_law.exponent);
            }
        }
    }

    std::array<const Field3D*, 5> masses() const
    {
        return {&mass[0], &mass[1], &mass[2], &mass[3], &mass[4]};
    }
    std::array<const Field3D*, 5> numbers() const
    {
        return {&number[0], &number[1], &number[2], &number[3], &number[4]};
    }
    std::array<const Field3D*, 5> reflectivities() const
    {
        return {&z[0], &z[1], &z[2], &z[3], &z[4]};
    }
};

struct Tendencies
{
    std::array<Field3D, 8> field;

    void compute(tmv::MilbrandtScheme& scheme, const Inputs& input, double dt)
    {
        Field3D p(input.density.size_r(), input.density.size_th(),
                  input.density.size_z(), 90000.0f);
        Field3D theta(input.density.size_r(), input.density.size_th(),
                      input.density.size_z(), 300.0f);
        Field3D qv(input.density.size_r(), input.density.size_th(),
                   input.density.size_z(), 0.01f);
        Field3D qc(input.density.size_r(), input.density.size_th(),
                   input.density.size_z(), 0.0f);
        scheme.compute_tendencies(
            p, theta, input.density, qv, qc, input.mass[0], input.mass[1],
            input.mass[2], input.mass[3], input.mass[4], dt,
            field[0], field[1], field[2], field[3], field[4], field[5],
            field[6], field[7]);
    }
};

struct WarmFixture
{
    SimulationStorageState storage;
    Field3D cloud_number;
    Field3D rain_number;
    Field3D rain_z;

    explicit WarmFixture(int columns = 1, int levels = 1)
    {
        auto& grid = storage.grid;
        grid.nr = columns;
        grid.nth = 1;
        grid.nz = levels;
        grid.dx = 2.0;
        grid.dy = 3.0;
        grid.dz = 5.0;
        grid.dt = 0.05;
        grid.geometry.initialize(columns, 1, levels,
                                 grid.dx, grid.dz, grid.dy,
                                 CoordinateSystem::Cartesian,
                                 StaggerType::Collocated);
        storage.configuration.microphysics_scheme = "milbrandt";
        storage.configuration.validation_policy.mode = tmv::GuardMode::Strict;
        storage.configuration.validation_policy.fail_on =
            tmv::GuardFailOn::Both;
        storage.runtime.boundary_layer.config.scheme_id = "none";
        storage.runtime.radiation.config.scheme_id = "none";
        storage.runtime.chaos.config.scheme_id = "none";
        storage.runtime.terrain.config.scheme_id = "none";
        auto& f = storage.fields;
        for (Field3D* field : {&f.rho, &f.p, &f.u, &f.v, &f.w,
                               &f.tracer, &f.theta, &f.qv, &f.qc,
                               &f.qr, &f.qi, &f.qs, &f.qg, &f.qh,
                               &f.tke})
        {
            field->resize(columns, 1, levels, 0.0f);
        }
        for (Field3D* field : {&storage.tendencies.dtheta_dt_rad,
                               &storage.tendencies.dtheta_dt_pbl,
                               &storage.tendencies.dqv_dt_pbl,
                               &storage.tendencies.dqc_dt_pbl,
                               &storage.tendencies.dqi_dt_pbl,
                               &storage.tendencies.du_dt_pbl,
                               &storage.tendencies.dv_dt_pbl,
                               &storage.tendencies.dtke_dt_pbl})
        {
            field->resize(columns, 1, levels, 0.0f);
        }
        cloud_number.resize(columns, 1, levels, 1.0e5f);
        rain_number.resize(columns, 1, levels, 1.0e3f);
        rain_z.resize(columns, 1, levels, 0.0f);
        for (int i = 0; i < columns; ++i)
        {
            for (int k = 0; k < levels; ++k)
            {
                const float pressure = (i == 0 ? 90000.0f : 70000.0f)
                    - 1000.0f * k;
                f.rho(i, 0, k) = 1.0f + 0.1f * k;
                f.p(i, 0, k) = pressure;
                f.theta(i, 0, k) = static_cast<float>(
                    thermodynamics::temperature_to_theta(280.0, pressure));
                f.qv(i, 0, k) = static_cast<float>(
                    0.5 * thermodynamics::saturation_mixing_ratio_water(
                        280.0, pressure));
                f.qc(i, 0, k) = 0.001f;
                f.qr(i, 0, k) = 1.0e-4f * (k + 1);
                cloud_number(i, 0, k) = 1.0e5f * (k + 1);
                rain_number(i, 0, k) = 1.0e3f * (k + 1);
                rain_z(i, 0, k) = independent_z(
                    f.qr(i, 0, k), rain_number(i, 0, k),
                    523.598775598, 3.0, 20.0);
            }
        }
        storage.base.rho0.assign(static_cast<std::size_t>(levels), 1.0);
        storage.base.p0.assign(static_cast<std::size_t>(levels), 90000.0);
        storage.base.qv0.assign(static_cast<std::size_t>(levels), 0.01);
        storage.base.theta0.assign(static_cast<std::size_t>(levels), 300.0);
        storage.base.u0.assign(static_cast<std::size_t>(levels), 0.0);
        storage.base.v0.assign(static_cast<std::size_t>(levels), 0.0);
        storage.surface.precipitation_rate_mm_h.assign(
            static_cast<std::size_t>(columns), 0.0f);
        storage.surface.accumulated_rainfall_mm.assign(
            static_cast<std::size_t>(columns), 0.0f);
        storage.surface.accumulated_rainfall_residual_mm.assign(
            static_cast<std::size_t>(columns), 0.0f);
    }

    SimulationStateView view()
    {
        return make_simulation_state_view(storage);
    }
};

} // namespace

TEST_CASE("internal Milbrandt five-Z sedimentation uses the accepted-primary hook",
          "[physics][milbrandt][triple_internal]")
{
    Inputs input;
    input.seed_column(0, 0);
    tmv::MilbrandtScheme scheme;
    std::string error;
    REQUIRE(scheme.enable_internal_triple_sedimentation_for_test(
        input.density, input.masses(), input.numbers(),
        input.reflectivities(), error));
    CHECK(scheme.get_num_prognostic_vars() == 18);
    REQUIRE_FALSE(scheme.commit_prepared_private_candidate());
    scheme.set_vertical_spacing_m(1, 1, 2, {5.0, 7.0});
    Tendencies tendencies;
    constexpr double dt = 0.05;
    REQUIRE_NOTHROW(tendencies.compute(scheme, input, dt));

    std::array<Field3D, 5> accepted_mass = input.mass;
    Field3D qv(1, 1, 2, 0.01f);
    Field3D qc(1, 1, 2, 0.0f);
    for (std::size_t species = 0; species < 5; ++species)
    {
        const auto p = tmv::milbrandt_detail::
            current_nimbus_triple_sedimentation_parameters(
                static_cast<TripleMomentSpecies>(species));
        std::array<float, 2> reference_q{};
        std::array<float, 2> reference_n{};
        std::array<float, 2> reference_z{};
        tmv::milbrandt_detail::TripleMomentSurfaceInventory surface;
        REQUIRE(tmv::milbrandt_detail::sediment_triple_moment_column(
            2, input.mass[species].data(), input.number[species].data(),
            input.z[species].data(), input.density.data(),
            std::array<double, 2>{5.0, 7.0}.data(), dt, p,
            reference_q.data(), reference_n.data(), reference_z.data(),
            surface, error));
        for (int k = 0; k < 2; ++k)
        {
            accepted_mass[species](0, 0, k) = static_cast<float>(
                input.mass[species](0, 0, k)
                + dt * tendencies.field[species + 3](0, 0, k));
            CHECK(accepted_mass[species](0, 0, k) == Approx(reference_q[k]).epsilon(1.0e-6));
        }
        const double old_inventory =
            input.density(0, 0, 0) * 5.0 * input.mass[species](0, 0, 0)
            + input.density(0, 0, 1) * 7.0 * input.mass[species](0, 0, 1);
        const double new_inventory =
            input.density(0, 0, 0) * 5.0 * accepted_mass[species](0, 0, 0)
            + input.density(0, 0, 1) * 7.0 * accepted_mass[species](0, 0, 1);
        CHECK(old_inventory == Approx(new_inventory + surface.mass).epsilon(1.0e-5));
    }
    tmv::MicrophysicsAcceptedPrimaryFields accepted{
        input.density, qv, qc, accepted_mass[0], accepted_mass[1],
        accepted_mass[2], accepted_mass[3], accepted_mass[4]};
    REQUIRE(scheme.prepare_post_primary_private_commit(dt, accepted, error));
    auto copy = scheme.clone();
    REQUIRE(copy);
    CHECK_FALSE(copy->commit_prepared_private_candidate());
    REQUIRE(scheme.commit_prepared_private_candidate());
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    for (std::size_t species = 0; species < 5; ++species)
    {
        const auto p = tmv::milbrandt_detail::
            current_nimbus_triple_sedimentation_parameters(
                static_cast<TripleMomentSpecies>(species));
        std::array<float, 2> reference_q{};
        std::array<float, 2> reference_n{};
        std::array<float, 2> reference_z{};
        tmv::milbrandt_detail::TripleMomentSurfaceInventory surface;
        REQUIRE(tmv::milbrandt_detail::sediment_triple_moment_column(
            2, input.mass[species].data(), input.number[species].data(),
            input.z[species].data(), input.density.data(),
            std::array<double, 2>{5.0, 7.0}.data(), dt, p,
            reference_q.data(), reference_n.data(), reference_z.data(),
            surface, error));
        for (int k = 0; k < 2; ++k)
        {
            CHECK(scheme.internal_triple_number_for_test(species)(0, 0, k)
                  == reference_n[k]);
            CHECK(scheme.internal_triple_reflectivity_for_test(species)(0, 0, k)
                  == reference_z[k]);
        }
        const auto committed = scheme.internal_triple_surface_for_test(0, 0, species);
        CHECK(committed.mass == surface.mass);
        CHECK(committed.number == surface.number);
        CHECK(committed.reflectivity == surface.reflectivity);
    }
}

TEST_CASE("internal Milbrandt five-Z guards unsupported owners and aggregate work",
          "[physics][milbrandt][triple_internal]")
{
    Inputs input;
    input.seed_column(0, 0);
    tmv::MilbrandtScheme scheme;
    std::string error;
    REQUIRE(scheme.enable_internal_triple_sedimentation_for_test(
        input.density, input.masses(), input.numbers(),
        input.reflectivities(), error));
    REQUIRE_THROWS_AS(scheme.prognostic_scalars_for_transport(), std::logic_error);
    REQUIRE_THROWS_AS(scheme.clone_parameters(), std::logic_error);
    REQUIRE_THROWS_AS(scheme.portable_state(), std::logic_error);
    REQUIRE_THROWS_AS(scheme.portable_parameter_state(), std::logic_error);
    CHECK_FALSE(scheme.restore_portable_state({}, error));
    std::vector<float> multipliers(1, 1.0f);
    CHECK_FALSE(scheme.apply_stochastic_column_multipliers(
        0.05, input.density, multipliers, error));
    scheme.invalidate_transient_state_after_domain_recycling();
    scheme.set_vertical_spacing_m(1, 1, 2, {5.0, 7.0});
    Tendencies tendencies;
    REQUIRE_THROWS_AS(tendencies.compute(scheme, input, 0.05), std::runtime_error);

    Inputs edge(6, 1, 16);
    tmv::MilbrandtScheme admitted;
    REQUIRE(admitted.enable_internal_triple_sedimentation_for_test(
        edge.density, edge.masses(), edge.numbers(),
        edge.reflectivities(), error));
    admitted.set_vertical_spacing_m(6, 1, 16, std::vector<double>(96, 5.0));
    Tendencies edge_tendencies;
    REQUIRE_NOTHROW(edge_tendencies.compute(admitted, edge, 0.05));
    Inputs too_large(7, 1, 16);
    tmv::MilbrandtScheme rejected;
    CHECK_FALSE(rejected.enable_internal_triple_sedimentation_for_test(
        too_large.density, too_large.masses(), too_large.numbers(),
        too_large.reflectivities(), error));
    CHECK(error.find("aggregate work budget") != std::string::npos);
    CHECK(rejected.get_num_prognostic_vars() == 13);
}

TEST_CASE("internal Milbrandt late fifth-species column failure is atomic",
          "[physics][milbrandt][triple_internal]")
{
    Inputs input(2, 1, 2);
    input.seed_column(0, 0);
    input.seed_column(1, 0);
    tmv::MilbrandtScheme scheme;
    std::string error;
    REQUIRE(scheme.enable_internal_triple_sedimentation_for_test(
        input.density, input.masses(), input.numbers(),
        input.reflectivities(), error));
    scheme.set_vertical_spacing_m(2, 1, 2, {5.0, 7.0, 5.0, 7.0});
    Tendencies tendencies;
    for (auto& field : tendencies.field)
    {
        field.resize(2, 1, 2, -7.0f);
    }
    const Field3D number_before = scheme.internal_triple_number_for_test(4);
    const Field3D z_before = scheme.internal_triple_reflectivity_for_test(4);
    input.mass[4](1, 0, 1) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(tendencies.compute(scheme, input, 0.05), std::runtime_error);
    for (const auto& field : tendencies.field)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == -7.0f);
        }
    }
    CHECK(same_field_bytes(scheme.internal_triple_number_for_test(4), number_before));
    CHECK(same_field_bytes(scheme.internal_triple_reflectivity_for_test(4), z_before));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
}

TEST_CASE("internal Milbrandt legacy noexcept commit poisons without moving N or Z",
          "[physics][milbrandt][triple_internal]")
{
    Inputs input;
    input.seed_column(0, 0);
    tmv::MilbrandtScheme scheme;
    std::string error;
    REQUIRE(scheme.enable_internal_triple_sedimentation_for_test(
        input.density, input.masses(), input.numbers(),
        input.reflectivities(), error));
    const Field3D n_before = scheme.internal_triple_number_for_test(0);
    const Field3D z_before = scheme.internal_triple_reflectivity_for_test(0);
    scheme.commit_prognostic_update(0.05);
    CHECK(same_field_bytes(scheme.internal_triple_number_for_test(0), n_before));
    CHECK(same_field_bytes(scheme.internal_triple_reflectivity_for_test(0), z_before));
    scheme.set_vertical_spacing_m(1, 1, 2, {5.0, 7.0});
    Tendencies tendencies;
    REQUIRE_THROWS_AS(tendencies.compute(scheme, input, 0.05), std::runtime_error);
}

TEST_CASE("internal Milbrandt five-Z direct columns have worker-setting identity",
          "[physics][milbrandt][triple_internal]")
{
    Inputs input(2, 1, 2);
    input.seed_column(0, 0);
    input.seed_column(1, 0);
    std::array<Tendencies, 2> result;
    for (int run = 0; run < 2; ++run)
    {
#ifdef _OPENMP
        omp_set_num_threads(run == 0 ? 1 : 4);
#endif
        tmv::MilbrandtScheme scheme;
        std::string error;
        REQUIRE(scheme.enable_internal_triple_sedimentation_for_test(
            input.density, input.masses(), input.numbers(),
            input.reflectivities(), error));
        scheme.set_vertical_spacing_m(2, 1, 2, {5.0, 7.0, 5.0, 7.0});
        REQUIRE_NOTHROW(result[run].compute(scheme, input, 0.05));
    }
    for (std::size_t field = 0; field < result[0].field.size(); ++field)
    {
        CHECK(same_field_bytes(result[0].field[field], result[1].field[field]));
    }
}

TEST_CASE("internal Milbrandt post-primary q mismatch and native request fail closed",
          "[physics][milbrandt][triple_internal]")
{
    Inputs input;
    input.seed_column(0, 0);
    tmv::MilbrandtScheme scheme;
    std::string error;
    REQUIRE(scheme.enable_internal_triple_sedimentation_for_test(
        input.density, input.masses(), input.numbers(),
        input.reflectivities(), error));
    scheme.set_vertical_spacing_m(1, 1, 2, {5.0, 7.0});
    Tendencies tendencies;
    scheme.set_require_native_dispatch(true);
    REQUIRE_THROWS_AS(tendencies.compute(scheme, input, 0.05),
                      std::runtime_error);
    scheme.set_require_native_dispatch(false);
    REQUIRE_NOTHROW(tendencies.compute(scheme, input, 0.05));
    std::array<Field3D, 5> accepted_mass = input.mass;
    Field3D qv(1, 1, 2, 0.01f);
    Field3D qc(1, 1, 2, 0.0f);
    for (std::size_t species = 0; species < 5; ++species)
    {
        for (int k = 0; k < 2; ++k)
        {
            accepted_mass[species](0, 0, k) = static_cast<float>(
                input.mass[species](0, 0, k)
                + 0.05 * tendencies.field[species + 3](0, 0, k));
        }
    }
    tmv::MicrophysicsAcceptedPrimaryFields accepted{
        input.density, qv, qc, accepted_mass[0], accepted_mass[1],
        accepted_mass[2], accepted_mass[3], accepted_mass[4]};
    const Field3D n_before = scheme.internal_triple_number_for_test(4);
    const Field3D z_before = scheme.internal_triple_reflectivity_for_test(4);
    accepted_mass[4](0, 0, 1) += 1.0e-4f;
    CHECK_FALSE(scheme.prepare_post_primary_private_commit(0.05, accepted, error));
    CHECK(error.find("accepted mass differs") != std::string::npos);
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    accepted_mass[4](0, 0, 1) = 0.0f;
    CHECK_FALSE(scheme.prepare_post_primary_private_commit(0.05, accepted, error));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    CHECK(same_field_bytes(scheme.internal_triple_number_for_test(4), n_before));
    CHECK(same_field_bytes(scheme.internal_triple_reflectivity_for_test(4), z_before));
}

TEST_CASE("internal Milbrandt WarmOnly stages coherent q N Z and theta before commit",
          "[physics][milbrandt][triple_warm_internal]")
{
    WarmFixture fixture(2);
    auto state = fixture.view();
    tmv::MilbrandtScheme scheme;
    std::string error;
    REQUIRE(scheme.enable_internal_triple_warm_for_test(
        state, fixture.cloud_number, fixture.rain_number,
        fixture.rain_z, error));
    CHECK(scheme.get_num_prognostic_vars() == 18);
    REQUIRE_FALSE(scheme.commit_prepared_private_candidate());
    scheme.set_vertical_spacing_m(2, 1, 1, {5.0, 5.0});
    std::array<Field3D, 8> rates;
    auto& f = fixture.storage.fields;
    REQUIRE_NOTHROW(scheme.compute_tendencies(
        f.p, f.theta, f.rho, f.qv, f.qc, f.qr,
        f.qi, f.qs, f.qg, f.qh, 0.05,
        rates[0], rates[1], rates[2], rates[3],
        rates[4], rates[5], rates[6], rates[7]));
    Field3D qv_after(f.qv);
    Field3D qc_after(f.qc);
    Field3D qr_after(f.qr);
    for (int i = 0; i < 2; ++i)
    {
        const double pressure = f.p(i, 0, 0);
        const double temperature =
            thermodynamics::theta_to_temperature(
                f.theta(i, 0, 0), pressure);
        const tmv::milbrandt_detail::WarmRainCell cell{
            static_cast<float>(temperature), f.p(i, 0, 0),
            f.rho(i, 0, 0), f.qv(i, 0, 0), f.qc(i, 0, 0),
            f.qr(i, 0, 0), fixture.cloud_number(i, 0, 0),
            fixture.rain_number(i, 0, 0)};
        const tmv::milbrandt_detail::RainTripleMomentState initial{
            f.qr(i, 0, 0), fixture.rain_number(i, 0, 0),
            fixture.rain_z(i, 0, 0)};
        tmv::milbrandt_detail::WarmRainTendencies reference_rates{};
        tmv::milbrandt_detail::RainTripleMomentState reference_rain{};
        REQUIRE(tmv::milbrandt_detail::evaluate_triple_only_warm_cell(
            tmv::milbrandt_detail::warm_rain_parameters(
                0.0, 523.598775598, 3.0, 149.1, 0.5),
            cell, 0.05, {}, initial,
            reference_rates, reference_rain, error));
        CHECK(rates[1](i, 0, 0) == reference_rates.vapor);
        CHECK(rates[2](i, 0, 0) == reference_rates.cloud);
        CHECK(rates[3](i, 0, 0) == reference_rates.rain);
        tmv::milbrandt_detail::TripleWarmRainResult produced;
        tmv::milbrandt_detail::triple_warm_rain_cell(
            tmv::milbrandt_detail::warm_rain_parameters(
                0.0, 523.598775598, 3.0, 149.1, 0.5),
            cell, 0.05, {}, produced);
        double evaporation_rate = 0.0;
        for (std::size_t event = 0; event < produced.events().size; ++event)
        {
            if (produced.events().events[event].kind ==
                tmv::milbrandt_detail::WarmRainAcceptedEventKind::Evaporation)
            {
                evaporation_rate +=
                    produced.events().events[event].rain_mass_rate_per_kg_s;
            }
        }
        REQUIRE(evaporation_rate < 0.0);
        const double expected_temperature_rate =
            microphysics_constants::L_v / microphysics_constants::cp
            * evaporation_rate;
        const double expected_theta_rate = expected_temperature_rate *
            std::pow(microphysics_constants::p0 / pressure,
                     microphysics_constants::R_d /
                         microphysics_constants::cp);
        CHECK(rates[0](i, 0, 0) == Approx(expected_theta_rate)
            .epsilon(2.0e-6));
        qv_after(i, 0, 0) = static_cast<float>(
            static_cast<double>(f.qv(i, 0, 0))
            + 0.05 * rates[1](i, 0, 0));
        qc_after(i, 0, 0) = static_cast<float>(
            static_cast<double>(f.qc(i, 0, 0))
            + 0.05 * rates[2](i, 0, 0));
        qr_after(i, 0, 0) = static_cast<float>(
            static_cast<double>(f.qr(i, 0, 0))
            + 0.05 * rates[3](i, 0, 0));
        const double old_water = f.qv(i, 0, 0) +
            f.qc(i, 0, 0) + f.qr(i, 0, 0);
        const double new_water = qv_after(i, 0, 0) +
            qc_after(i, 0, 0) + qr_after(i, 0, 0);
        CHECK(old_water == Approx(new_water).epsilon(2.0e-6));
    }
    tmv::MicrophysicsWaterRepairReport water;
    tmv::MicrophysicsThermodynamicRepairReport heat;
    tmv::MicrophysicsAcceptedPrimaryFields accepted{
        f.rho, qv_after, qc_after, qr_after,
        f.qi, f.qs, f.qg, f.qh, &water, &heat};
    water.species[2].lower_bound_roundoff_count = 1;
    CHECK_FALSE(scheme.prepare_post_primary_private_commit(
        0.05, accepted, error));
    CHECK(error.find("accepted-primary repair") != std::string::npos);
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    water.species[2].lower_bound_roundoff_count = 0;
    heat.theta_lower_bound_count = 1;
    CHECK_FALSE(scheme.prepare_post_primary_private_commit(
        0.05, accepted, error));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    heat.theta_lower_bound_count = 0;
    REQUIRE(scheme.prepare_post_primary_private_commit(
        0.05, accepted, error));
    auto clone = scheme.clone();
    REQUIRE(clone);
    CHECK_FALSE(clone->commit_prepared_private_candidate());
    REQUIRE(scheme.commit_prepared_private_candidate());
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
    for (int i = 0; i < 2; ++i)
    {
        CHECK(scheme.internal_triple_cloud_number_for_test()(i, 0, 0) >= 0.0f);
        CHECK(scheme.internal_triple_number_for_test(0)(i, 0, 0) > 0.0f);
        CHECK(scheme.internal_triple_reflectivity_for_test(0)(i, 0, 0) > 0.0f);
        for (std::size_t other = 1; other < 5; ++other)
        {
            CHECK(scheme.internal_triple_number_for_test(other)(i, 0, 0) == 0.0f);
            CHECK(scheme.internal_triple_reflectivity_for_test(other)(i, 0, 0) == 0.0f);
        }
        CHECK(scheme.internal_triple_surface_for_test(i, 0, 0).mass == 0.0);
    }
}

TEST_CASE("internal Milbrandt WarmOnly rejects active source and invalid profile neighbors",
          "[physics][milbrandt][triple_warm_internal]")
{
    WarmFixture fixture;
    auto state = fixture.view();
    const auto rejected = [&]()
    {
        tmv::MilbrandtScheme scheme;
        std::string error;
        CHECK_FALSE(scheme.enable_internal_triple_warm_for_test(
            state, fixture.cloud_number, fixture.rain_number,
            fixture.rain_z, error));
        CHECK_FALSE(error.empty());
        CHECK(scheme.get_num_prognostic_vars() == 13);
    };
    fixture.storage.runtime.boundary_layer.config.scheme_id = "slab";
    rejected();
    fixture.storage.runtime.boundary_layer.config.scheme_id = "none";
    fixture.storage.runtime.radiation.config.scheme_id = "rrtmg";
    rejected();
    fixture.storage.runtime.radiation.config.scheme_id = "none";
    fixture.storage.runtime.chaos.config.scheme_id = "full_stochastic";
    rejected();
    fixture.storage.runtime.chaos.config.scheme_id = "none";
    fixture.storage.runtime.chaos.noise_elapsed_s = 1.0;
    rejected();
    fixture.storage.runtime.chaos.noise_elapsed_s = 0.0;
    fixture.storage.runtime.terrain.config.scheme_id = "bell";
    rejected();
    fixture.storage.runtime.terrain.config.scheme_id = "none";
    fixture.storage.runtime.compute.active_kind = ComputeBackendKind::Metal;
    rejected();
    fixture.storage.runtime.compute.active_kind = ComputeBackendKind::Cpu;
    fixture.storage.tendencies.dtheta_dt_rad(0, 0, 0) = 1.0f;
    rejected();
    fixture.storage.tendencies.dtheta_dt_rad(0, 0, 0) = 0.0f;
    fixture.storage.tendencies.dqv_dt_pbl(0, 0, 0) = 1.0e-6f;
    rejected();
    fixture.storage.tendencies.dqv_dt_pbl(0, 0, 0) = 0.0f;
    fixture.storage.configuration.validation_policy.mode =
        tmv::GuardMode::Sanitize;
    rejected();
    fixture.storage.configuration.validation_policy.mode =
        tmv::GuardMode::Strict;
    fixture.storage.fields.p(0, 0, 0) = 99.0f;
    rejected();
    fixture.storage.fields.p(0, 0, 0) = 90000.0f;
    fixture.storage.fields.theta(0, 0, 0) = static_cast<float>(
        thermodynamics::temperature_to_theta(260.0, 90000.0));
    rejected();
    fixture.storage.fields.theta(0, 0, 0) = static_cast<float>(
        thermodynamics::temperature_to_theta(280.0, 90000.0));
    fixture.storage.fields.qi(0, 0, 0) = 1.0e-6f;
    rejected();
    fixture.storage.fields.qi(0, 0, 0) = 0.0f;
    fixture.cloud_number(0, 0, 0) = 0.0f;
    rejected();
    fixture.cloud_number(0, 0, 0) = 1.0e5f;
    fixture.rain_z(0, 0, 0) = 0.0f;
    rejected();
    fixture.rain_z(0, 0, 0) = independent_z(
        fixture.storage.fields.qr(0, 0, 0),
        fixture.rain_number(0, 0, 0),
        523.598775598, 3.0, 20.0);
    tmv::MilbrandtScheme admitted;
    std::string error;
    REQUIRE(admitted.enable_internal_triple_warm_for_test(
        state, fixture.cloud_number, fixture.rain_number,
        fixture.rain_z, error));
    CHECK_FALSE(admitted.enable_internal_triple_sedimentation_for_test(
        fixture.storage.fields.rho, {}, {}, {}, error));
    REQUIRE_THROWS_AS(admitted.prognostic_scalars_for_transport(),
                      std::logic_error);
    REQUIRE_THROWS_AS(admitted.portable_state(), std::logic_error);
    REQUIRE_THROWS_AS(admitted.clone_parameters(), std::logic_error);
    std::vector<float> multipliers(1, 1.0f);
    CHECK_FALSE(admitted.apply_stochastic_column_multipliers(
        0.05, fixture.storage.fields.rho, multipliers, error));
    admitted.set_vertical_spacing_m(5.0);
    std::array<Field3D, 8> rates;
    for (auto& field : rates)
    {
        field.resize(1, 1, 1, -7.0f);
    }
    auto& f = fixture.storage.fields;
    fixture.storage.runtime.radiation.held_atmospheric_power_w = 1.0;
    REQUIRE_THROWS_AS(admitted.compute_tendencies(
        f.p, f.theta, f.rho, f.qv, f.qc, f.qr,
        f.qi, f.qs, f.qg, f.qh, 0.05,
        rates[0], rates[1], rates[2], rates[3],
        rates[4], rates[5], rates[6], rates[7]),
        std::runtime_error);
    for (const auto& field : rates)
    {
        CHECK(field(0, 0, 0) == -7.0f);
    }
    fixture.storage.runtime.radiation.held_atmospheric_power_w = 0.0;
    admitted.set_require_native_dispatch(true);
    REQUIRE_THROWS_AS(admitted.compute_tendencies(
        f.p, f.theta, f.rho, f.qv, f.qc, f.qr,
        f.qi, f.qs, f.qg, f.qh, 0.05,
        rates[0], rates[1], rates[2], rates[3],
        rates[4], rates[5], rates[6], rates[7]),
        std::runtime_error);
    admitted.set_require_native_dispatch(false);
    admitted.invalidate_transient_state_after_domain_recycling();
    REQUIRE_THROWS_AS(admitted.compute_tendencies(
        f.p, f.theta, f.rho, f.qv, f.qc, f.qr,
        f.qi, f.qs, f.qg, f.qh, 0.05,
        rates[0], rates[1], rates[2], rates[3],
        rates[4], rates[5], rates[6], rates[7]),
        std::runtime_error);
}

TEST_CASE("internal Milbrandt WarmOnly late rain-to-cloud cell publishes no tendency",
          "[physics][milbrandt][triple_warm_internal]")
{
    WarmFixture fixture(2);
    auto state = fixture.view();
    auto& f = fixture.storage.fields;
    f.qc(1, 0, 0) = 0.0f;
    fixture.cloud_number(1, 0, 0) = 0.0f;
    fixture.rain_number(1, 0, 0) = 1.0e7f;
    fixture.rain_z(1, 0, 0) = independent_z(
        f.qr(1, 0, 0), fixture.rain_number(1, 0, 0),
        523.598775598, 3.0, 20.0);
    tmv::MilbrandtScheme scheme;
    std::string error;
    REQUIRE(scheme.enable_internal_triple_warm_for_test(
        state, fixture.cloud_number, fixture.rain_number,
        fixture.rain_z, error));
    scheme.set_vertical_spacing_m(2, 1, 1, {5.0, 5.0});
    std::array<Field3D, 8> rates;
    for (auto& field : rates)
    {
        field.resize(2, 1, 1, -7.0f);
    }
    const Field3D nc_before = scheme.internal_triple_cloud_number_for_test();
    const Field3D nr_before = scheme.internal_triple_number_for_test(0);
    const Field3D z_before = scheme.internal_triple_reflectivity_for_test(0);
    REQUIRE_THROWS_AS(scheme.compute_tendencies(
        f.p, f.theta, f.rho, f.qv, f.qc, f.qr,
        f.qi, f.qs, f.qg, f.qh, 0.05,
        rates[0], rates[1], rates[2], rates[3],
        rates[4], rates[5], rates[6], rates[7]),
        std::runtime_error);
    for (const auto& field : rates)
    {
        CHECK(field(0, 0, 0) == -7.0f);
        CHECK(field(1, 0, 0) == -7.0f);
    }
    CHECK(same_field_bytes(scheme.internal_triple_cloud_number_for_test(),
                           nc_before));
    CHECK(same_field_bytes(scheme.internal_triple_number_for_test(0),
                           nr_before));
    CHECK(same_field_bytes(scheme.internal_triple_reflectivity_for_test(0),
                           z_before));
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
}

TEST_CASE("internal Milbrandt WarmOnly rejects evaporation across freezing",
          "[physics][milbrandt][triple_warm_internal]")
{
    const auto parameters = tmv::milbrandt_detail::warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    bool found = false;
    double selected_dt = 0.0;
    float selected_temperature = 0.0f;
    float selected_rain = 0.0f;
    float selected_number = 0.0f;
    for (float temperature : {273.151f, 273.16f, 273.2f})
    {
        for (float rain : {1.0e-4f, 1.0e-3f, 5.0e-3f})
        {
            for (float number : {1.0e3f, 1.0e5f})
            {
                for (double dt : {0.05, 0.2, 1.0, 10.0})
                {
                    const tmv::milbrandt_detail::WarmRainCell cell{
                        temperature, 90000.0f, 1.0f, 0.0f,
                        0.0f, rain, 0.0f, number};
                    const tmv::milbrandt_detail::RainTripleMomentState initial{
                        rain, number,
                        independent_z(rain, number,
                                      523.598775598, 3.0, 20.0)};
                    tmv::milbrandt_detail::WarmRainTendencies rates{};
                    tmv::milbrandt_detail::RainTripleMomentState after{};
                    std::string error;
                    if (tmv::milbrandt_detail::evaluate_triple_only_warm_cell(
                            parameters, cell, dt, {}, initial,
                            rates, after, error) &&
                        temperature + dt * rates.temperature <
                            microphysics_constants::T0 - 0.01)
                    {
                        found = true;
                        selected_dt = dt;
                        selected_temperature = temperature;
                        selected_rain = rain;
                        selected_number = number;
                        break;
                    }
                }
                if (found) break;
            }
            if (found) break;
        }
        if (found) break;
    }
    REQUIRE(found);
    WarmFixture fixture;
    auto& f = fixture.storage.fields;
    f.theta(0, 0, 0) = static_cast<float>(
        thermodynamics::temperature_to_theta(
            selected_temperature, f.p(0, 0, 0)));
    f.qv(0, 0, 0) = 0.0f;
    f.qc(0, 0, 0) = 0.0f;
    fixture.cloud_number(0, 0, 0) = 0.0f;
    f.qr(0, 0, 0) = selected_rain;
    fixture.rain_number(0, 0, 0) = selected_number;
    fixture.rain_z(0, 0, 0) = independent_z(
        selected_rain, selected_number,
        523.598775598, 3.0, 20.0);
    auto state = fixture.view();
    tmv::MilbrandtScheme scheme;
    std::string error;
    REQUIRE(scheme.enable_internal_triple_warm_for_test(
        state, fixture.cloud_number, fixture.rain_number,
        fixture.rain_z, error));
    scheme.set_vertical_spacing_m(5.0);
    std::array<Field3D, 8> tendencies;
    for (auto& field : tendencies)
    {
        field.resize(1, 1, 1, -7.0f);
    }
    try
    {
        scheme.compute_tendencies(
            f.p, f.theta, f.rho, f.qv, f.qc, f.qr,
            f.qi, f.qs, f.qg, f.qh, selected_dt,
            tendencies[0], tendencies[1], tendencies[2], tendencies[3],
            tendencies[4], tendencies[5], tendencies[6], tendencies[7]);
        FAIL("WarmOnly crossed freezing without rejection");
    }
    catch (const std::runtime_error& failure)
    {
        CHECK(std::string(failure.what()).find("final temperature")
              != std::string::npos);
    }
    for (const auto& field : tendencies)
    {
        CHECK(field(0, 0, 0) == -7.0f);
    }
    CHECK_FALSE(scheme.commit_prepared_private_candidate());
}

TEST_CASE("internal Milbrandt warm-rain sedimentation composes q N Z with live rho dz",
          "[physics][milbrandt][triple_rain_coupled]")
{
    WarmFixture fixture(2, 3);
    auto state = fixture.view();
    tmv::MilbrandtScheme scheme;
    std::string error;
    REQUIRE(scheme.enable_internal_triple_warm_rain_sedimentation_for_test(
        state, fixture.cloud_number, fixture.rain_number,
        fixture.rain_z, error));
    CHECK(scheme.explicit_terminal_velocity_mps(0, 1.0e-4, 1.0) == 16.0);
    CHECK(scheme.explicit_terminal_velocity_mps(1, 1.0e-4, 1.0) == 0.0);
    const std::array<double, 6> thickness{4.0, 7.0, 5.0,
                                          6.0, 3.0, 8.0};
    scheme.set_vertical_spacing_m(2, 1, 3,
        std::vector<double>(thickness.begin(), thickness.end()));
    auto& f = fixture.storage.fields;
    const Field3D qv_before(f.qv);
    const Field3D qc_before(f.qc);
    const Field3D qr_before(f.qr);
    const Field3D nr_before(fixture.rain_number);
    const Field3D z_before(fixture.rain_z);
    std::array<Field3D, 8> tendencies;
    constexpr double dt = 0.05;
    REQUIRE_NOTHROW(scheme.compute_tendencies(
        f.p, f.theta, f.rho, f.qv, f.qc, f.qr,
        f.qi, f.qs, f.qg, f.qh, dt,
        tendencies[0], tendencies[1], tendencies[2], tendencies[3],
        tendencies[4], tendencies[5], tendencies[6], tendencies[7]));
    Field3D qv_after(qv_before);
    Field3D qc_after(qc_before);
    Field3D qr_after(qr_before);
    std::array<std::array<float, 3>, 2> expected_number{};
    std::array<std::array<float, 3>, 2> expected_z{};
    std::array<tmv::milbrandt_detail::TripleMomentSurfaceInventory, 2>
        expected_surface{};
    for (int i = 0; i < 2; ++i)
    {
        std::array<float, 3> warm_q{};
        std::array<float, 3> warm_n{};
        std::array<float, 3> warm_z{};
        std::array<float, 3> density{};
        std::array<double, 3> dz{};
        for (int k = 0; k < 3; ++k)
        {
            const double temperature =
                thermodynamics::theta_to_temperature(
                    f.theta(i, 0, k), f.p(i, 0, k));
            const tmv::milbrandt_detail::WarmRainCell cell{
                static_cast<float>(temperature), f.p(i, 0, k),
                f.rho(i, 0, k), qv_before(i, 0, k),
                qc_before(i, 0, k), qr_before(i, 0, k),
                fixture.cloud_number(i, 0, k), nr_before(i, 0, k)};
            const tmv::milbrandt_detail::RainTripleMomentState initial{
                qr_before(i, 0, k), nr_before(i, 0, k),
                z_before(i, 0, k)};
            tmv::milbrandt_detail::WarmRainTendencies rates{};
            tmv::milbrandt_detail::RainTripleMomentState warm{};
            REQUIRE(tmv::milbrandt_detail::evaluate_triple_only_warm_cell(
                tmv::milbrandt_detail::warm_rain_parameters(
                    0.0, 523.598775598, 3.0, 149.1, 0.5),
                cell, dt, {}, initial, rates, warm, error));
            qv_after(i, 0, k) = static_cast<float>(
                static_cast<double>(qv_before(i, 0, k))
                + dt * rates.vapor);
            qc_after(i, 0, k) = static_cast<float>(
                static_cast<double>(qc_before(i, 0, k))
                + dt * rates.cloud);
            warm_q[static_cast<std::size_t>(k)] = static_cast<float>(
                static_cast<double>(qr_before(i, 0, k))
                + dt * rates.rain);
            warm_n[static_cast<std::size_t>(k)] = warm.number_per_kg;
            warm_z[static_cast<std::size_t>(k)] = warm.reflectivity_per_kg;
            density[static_cast<std::size_t>(k)] = f.rho(i, 0, k);
            dz[static_cast<std::size_t>(k)] =
                thickness[static_cast<std::size_t>(i * 3 + k)];
            CHECK(tendencies[1](i, 0, k) == rates.vapor);
            CHECK(tendencies[2](i, 0, k) == rates.cloud);
        }
        std::array<float, 3> sediment_q{};
        std::array<float, 3> sediment_n{};
        std::array<float, 3> sediment_z{};
        REQUIRE(tmv::milbrandt_detail::sediment_triple_moment_column(
            3, warm_q.data(), warm_n.data(), warm_z.data(),
            density.data(), dz.data(), dt,
            tmv::milbrandt_detail::current_nimbus_triple_sedimentation_parameters(
                TripleMomentSpecies::Rain),
            sediment_q.data(), sediment_n.data(), sediment_z.data(),
            expected_surface[static_cast<std::size_t>(i)], error));
        double before_water = 0.0;
        double after_water = 0.0;
        std::array<double, 3> warm_inventory{};
        std::array<double, 3> final_inventory{};
        for (int k = 0; k < 3; ++k)
        {
            const std::size_t level = static_cast<std::size_t>(k);
            qr_after(i, 0, k) = static_cast<float>(
                static_cast<double>(qr_before(i, 0, k))
                + dt * tendencies[3](i, 0, k));
            CHECK(qr_after(i, 0, k) ==
                  Approx(sediment_q[level]).epsilon(2.0e-6));
            expected_number[static_cast<std::size_t>(i)][level] =
                sediment_n[level];
            expected_z[static_cast<std::size_t>(i)][level] =
                sediment_z[level];
            const double weight = density[level] * dz[level];
            before_water += weight * (qv_before(i, 0, k)
                + qc_before(i, 0, k) + qr_before(i, 0, k));
            after_water += weight * (qv_after(i, 0, k)
                + qc_after(i, 0, k) + qr_after(i, 0, k));
            warm_inventory[0] += weight * warm_q[level];
            warm_inventory[1] += weight * warm_n[level];
            warm_inventory[2] += weight * warm_z[level];
            final_inventory[0] += weight * sediment_q[level];
            final_inventory[1] += weight * sediment_n[level];
            final_inventory[2] += weight * sediment_z[level];
        }
        const auto& surface = expected_surface[static_cast<std::size_t>(i)];
        CHECK(std::abs(before_water - after_water - surface.mass)
              <= 64.0 * std::numeric_limits<float>::epsilon()
                  * before_water);
        const std::array<double, 3> surface_value{
            surface.mass, surface.number, surface.reflectivity};
        for (std::size_t moment = 0; moment < 3; ++moment)
        {
            CHECK(std::abs(warm_inventory[moment]
                           - final_inventory[moment]
                           - surface_value[moment])
                  <= 64.0 * std::numeric_limits<float>::epsilon()
                      * warm_inventory[moment]);
        }
        CHECK(scheme.bottom_sedimentation_sink_rate(
            f.rho, f.qr, f.qs, f.qg, f.qh, i, 0, dt)
              == Approx(surface.mass /
                        (density[0] * dz[0] * dt)).epsilon(2.0e-6));
    }
    tmv::MicrophysicsWaterRepairReport water;
    tmv::MicrophysicsThermodynamicRepairReport heat;
    tmv::MicrophysicsAcceptedPrimaryFields accepted{
        f.rho, qv_after, qc_after, qr_after,
        f.qi, f.qs, f.qg, f.qh, &water, &heat};
    REQUIRE(scheme.prepare_post_primary_private_commit(dt, accepted, error));
    REQUIRE(scheme.commit_prepared_private_candidate());
    for (int i = 0; i < 2; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            CHECK(scheme.internal_triple_number_for_test(0)(i, 0, k)
                  == expected_number[static_cast<std::size_t>(i)]
                                    [static_cast<std::size_t>(k)]);
            CHECK(scheme.internal_triple_reflectivity_for_test(0)(i, 0, k)
                  == expected_z[static_cast<std::size_t>(i)]
                               [static_cast<std::size_t>(k)]);
        }
        const auto surface = scheme.internal_triple_surface_for_test(i, 0, 0);
        CHECK(surface.mass == expected_surface[static_cast<std::size_t>(i)].mass);
        CHECK(surface.number == expected_surface[static_cast<std::size_t>(i)].number);
        CHECK(surface.reflectivity ==
              expected_surface[static_cast<std::size_t>(i)].reflectivity);
    }
}

TEST_CASE("internal rain coupling bounds high-Courant work and fails late atomically",
          "[physics][milbrandt][triple_rain_coupled]")
{
    WarmFixture high(1, 8);
    high.storage.grid.dz = 1.0;
    high.storage.grid.geometry.initialize(
        1, 1, 8, high.storage.grid.dx,
        high.storage.grid.dz, high.storage.grid.dy,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    for (int k = 0; k < 8; ++k)
    {
        const double pressure = 90000.0;
        high.storage.fields.p(0, 0, k) =
            static_cast<float>(pressure);
        high.storage.fields.theta(0, 0, k) = static_cast<float>(
            thermodynamics::temperature_to_theta(280.0, pressure));
        high.storage.fields.rho(0, 0, k) = 1.0f;
        high.storage.fields.qv(0, 0, k) = static_cast<float>(
            thermodynamics::saturation_mixing_ratio_water(280.0, pressure));
        high.storage.fields.qc(0, 0, k) = 0.0f;
        high.cloud_number(0, 0, k) = 0.0f;
        high.storage.fields.qr(0, 0, k) = 1.0e-4f;
        high.rain_number(0, 0, k) = 1.0e5f;
        high.rain_z(0, 0, k) = independent_z(
            high.storage.fields.qr(0, 0, k),
            high.rain_number(0, 0, k),
            523.598775598, 3.0, 2.0);
    }
    auto high_state = high.view();
    tmv::MilbrandtScheme admitted;
    std::string error;
    REQUIRE(admitted.enable_internal_triple_warm_rain_sedimentation_for_test(
        high_state, high.cloud_number, high.rain_number,
        high.rain_z, error));
    std::vector<double> high_thickness(8);
    for (int k = 0; k < 8; ++k)
    {
        high_thickness[static_cast<std::size_t>(k)] = 1.0 + 0.1 * k;
    }
    admitted.set_vertical_spacing_m(1, 1, 8,
        std::move(high_thickness));
    std::array<Field3D, 8> high_rates;
    auto& high_fields = high.storage.fields;
    REQUIRE_NOTHROW(admitted.compute_tendencies(
        high_fields.p, high_fields.theta, high_fields.rho,
        high_fields.qv, high_fields.qc, high_fields.qr,
        high_fields.qi, high_fields.qs, high_fields.qg,
        high_fields.qh, 1.0,
        high_rates[0], high_rates[1], high_rates[2], high_rates[3],
        high_rates[4], high_rates[5], high_rates[6], high_rates[7]));
    CHECK(admitted.bottom_sedimentation_sink_rate(
        high_fields.rho, high_fields.qr, high_fields.qs,
        high_fields.qg, high_fields.qh, 0, 0, 1.0) > 0.0);

    WarmFixture unrealizable(1, 2);
    unrealizable.storage.grid.dz = 1.0;
    unrealizable.storage.grid.geometry.initialize(
        1, 1, 2, unrealizable.storage.grid.dx,
        unrealizable.storage.grid.dz, unrealizable.storage.grid.dy,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    for (int k = 0; k < 2; ++k)
    {
        const double pressure = unrealizable.storage.fields.p(0, 0, k);
        unrealizable.storage.fields.qv(0, 0, k) = static_cast<float>(
            thermodynamics::saturation_mixing_ratio_water(280.0, pressure));
        unrealizable.storage.fields.qc(0, 0, k) = 0.0f;
        unrealizable.cloud_number(0, 0, k) = 0.0f;
    }
    auto unrealizable_state = unrealizable.view();
    tmv::MilbrandtScheme guarded;
    REQUIRE(guarded.enable_internal_triple_warm_rain_sedimentation_for_test(
        unrealizable_state, unrealizable.cloud_number,
        unrealizable.rain_number, unrealizable.rain_z, error));
    guarded.set_vertical_spacing_m(1.0);
    std::array<Field3D, 8> rejected_rates;
    for (auto& field : rejected_rates)
    {
        field.resize(1, 1, 2, -7.0f);
    }
    auto& unrealizable_fields = unrealizable.storage.fields;
    try
    {
        guarded.compute_tendencies(
            unrealizable_fields.p, unrealizable_fields.theta,
            unrealizable_fields.rho, unrealizable_fields.qv,
            unrealizable_fields.qc, unrealizable_fields.qr,
            unrealizable_fields.qi, unrealizable_fields.qs,
            unrealizable_fields.qg, unrealizable_fields.qh, 1.0,
            rejected_rates[0], rejected_rates[1], rejected_rates[2],
            rejected_rates[3], rejected_rates[4], rejected_rates[5],
            rejected_rates[6], rejected_rates[7]);
        FAIL("unrealizable high-Courant rain column was published");
    }
    catch (const std::runtime_error& failure)
    {
        CHECK(std::string(failure.what()).find(
            "sixth-moment realizability bound") != std::string::npos);
    }
    for (const auto& field : rejected_rates)
    {
        CHECK(field(0, 0, 0) == -7.0f);
        CHECK(field(0, 0, 1) == -7.0f);
    }
    CHECK_FALSE(guarded.commit_prepared_private_candidate());

    WarmFixture over_budget(7, 16);
    auto budget_state = over_budget.view();
    tmv::MilbrandtScheme rejected;
    CHECK_FALSE(rejected.enable_internal_triple_warm_rain_sedimentation_for_test(
        budget_state, over_budget.cloud_number,
        over_budget.rain_number, over_budget.rain_z, error));
    CHECK(error.find("aggregate work budget") != std::string::npos);
    CHECK(rejected.get_num_prognostic_vars() == 13);

    WarmFixture late(2, 2);
    for (int i = 0; i < 2; ++i)
    {
        for (int k = 0; k < 2; ++k)
        {
            const double pressure = late.storage.fields.p(i, 0, k);
            late.storage.fields.qv(i, 0, k) = static_cast<float>(
                thermodynamics::saturation_mixing_ratio_water(
                    280.0, pressure));
            late.storage.fields.qc(i, 0, k) = 0.0f;
            late.cloud_number(i, 0, k) = 0.0f;
        }
    }
    auto late_state = late.view();
    tmv::MilbrandtScheme late_scheme;
    REQUIRE(late_scheme.enable_internal_triple_warm_rain_sedimentation_for_test(
        late_state, late.cloud_number, late.rain_number,
        late.rain_z, error));
    late_scheme.set_vertical_spacing_m(
        2, 1, 2, {5.0, 5.0, 1.0e-6, 1.0e-6});
    std::array<Field3D, 8> late_rates;
    for (auto& field : late_rates)
    {
        field.resize(2, 1, 2, -7.0f);
    }
    const Field3D nr_before = late_scheme.internal_triple_number_for_test(0);
    const Field3D z_before =
        late_scheme.internal_triple_reflectivity_for_test(0);
    auto& late_fields = late.storage.fields;
    REQUIRE_THROWS_AS(late_scheme.compute_tendencies(
        late_fields.p, late_fields.theta, late_fields.rho,
        late_fields.qv, late_fields.qc, late_fields.qr,
        late_fields.qi, late_fields.qs, late_fields.qg,
        late_fields.qh, 1.0,
        late_rates[0], late_rates[1], late_rates[2], late_rates[3],
        late_rates[4], late_rates[5], late_rates[6], late_rates[7]),
        std::runtime_error);
    for (const auto& field : late_rates)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            CHECK(field.data()[index] == -7.0f);
        }
    }
    CHECK(same_field_bytes(
        late_scheme.internal_triple_number_for_test(0), nr_before));
    CHECK(same_field_bytes(
        late_scheme.internal_triple_reflectivity_for_test(0), z_before));
    CHECK_FALSE(late_scheme.commit_prepared_private_candidate());
}
