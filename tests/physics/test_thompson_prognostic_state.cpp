#include "catch2/catch.hpp"
#include "microphysics/schemes/thompson/moment_initialization.hpp"
#include "microphysics/schemes/thompson/prognostic_state.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

Field3D field(float a, float b, float c)
{
    Field3D result(1, 1, 3);
    result[0][0][0] = a;
    result[0][0][1] = b;
    result[0][0][2] = c;
    return result;
}

double expected_profile(double height_m, double surface_total, double background)
{
    const double amplitude = surface_total - background;
    const double decay = -std::log(background / amplitude) / 0.8;
    return background + amplitude * std::exp(-(height_m / 1000.0) * decay);
}

TEST_CASE("Thompson fixed aerosol profile honors its surface and free-air endpoints",
          "[microphysics][thompson][state]")
{
    const Field3D rho = field(1.0f, 1.0f, 1.0f);
    const Field3D zero = field(0.0f, 0.0f, 0.0f);
    tmv::thompson::PrognosticState state;
    state.prepare(rho, zero, zero, zero, {0.0, 100000.0, 1000000.0});

    REQUIRE(state.water_aerosol()[0][0][0]
            == Approx(tmv::thompson::PrognosticState::surface_water_friendly_aerosol_m3));
    REQUIRE(state.ice_aerosol()[0][0][0]
            == Approx(tmv::thompson::PrognosticState::surface_ice_friendly_aerosol_m3));
    REQUIRE(state.water_aerosol()[0][0][2]
            == Approx(tmv::thompson::PrognosticState::free_troposphere_water_friendly_aerosol_m3));
    REQUIRE(state.ice_aerosol()[0][0][2]
            == Approx(tmv::thompson::PrognosticState::free_troposphere_ice_friendly_aerosol_m3));
    REQUIRE(state.water_aerosol()[0][0][0] == 350000000.0f);
    REQUIRE(state.ice_aerosol()[0][0][0] == 2000000.0f);
}

TEST_CASE("Thompson flat-column sampling and aerosol values match WRF v4.7.1",
          "[microphysics][thompson][state][literal]")
{
    const Field3D rho = field(1.0f, 1.0f, 1.0f);
    const Field3D zero = field(0.0f, 0.0f, 0.0f);
    const auto samples = tmv::thompson::flat_profile_sampling_heights(
        1, 1, 3, {100.0, 100.0, 100.0});
    REQUIRE(samples == std::vector<double>{100.0, 100.0, 200.0});
    tmv::thompson::PrognosticState state;
    state.prepare(rho, zero, zero, zero, samples);

    REQUIRE(state.water_aerosol()[0][0][0]
            == Approx(289801750.16493213).epsilon(2.0e-7));
    REQUIRE(state.ice_aerosol()[0][0][0]
            == Approx(1807528.3143076035).epsilon(2.0e-7));
    REQUIRE(state.water_aerosol()[0][0][1]
            == state.water_aerosol()[0][0][0]);
    REQUIRE(state.ice_aerosol()[0][0][1]
            == state.ice_aerosol()[0][0][0]);
    REQUIRE(state.water_aerosol()[0][0][2]
            == Approx(241682931.27388173).epsilon(2.0e-7));
    REQUIRE(state.ice_aerosol()[0][0][2]
            == Approx(1639753.528477389).epsilon(2.0e-7));

    REQUIRE(tmv::thompson::flat_profile_sampling_heights(
                1, 1, 1, {250.0})
            == std::vector<double>{250.0});
    REQUIRE(tmv::thompson::flat_profile_sampling_heights(
                1, 1, 3, {200.0, 400.0, 600.0})
            == std::vector<double>{300.0, 300.0, 800.0});
    REQUIRE_THROWS_AS(tmv::thompson::flat_profile_sampling_heights(
                          -1, 1, 1, {}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(tmv::thompson::flat_profile_sampling_heights(
                          std::numeric_limits<int>::max(),
                          std::numeric_limits<int>::max(),
                          std::numeric_limits<int>::max(), {}),
                      std::length_error);
}

bool bitwise_equal(const Field3D& lhs, const Field3D& rhs)
{
    return lhs.size() == rhs.size()
        && std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0;
}

} // namespace

TEST_CASE("Thompson mp=28 initializes five dry-air prognostic currencies",
          "[microphysics][thompson][state]")
{
    const Field3D rho = field(1.0f, 0.8f, 0.5f);
    const Field3D qc = field(1.0e-4f, 0.0f, 0.0f);
    const Field3D qr = field(0.0f, 2.0e-4f, 0.0f);
    const Field3D qi = field(0.0f, 0.0f, 3.0e-5f);
    tmv::thompson::PrognosticState state;
    state.prepare(rho, qc, qr, qi, {50.0, 150.0, 300.0});

    const auto bindings = state.transport_bindings();
    REQUIRE(bindings.size() == 5);
    REQUIRE(std::string(bindings[0].name) == "thompson.Nc_kg-1");
    REQUIRE(std::string(bindings[4].name) == "thompson.Nifa_kg-1");
    for (const auto& binding : bindings)
        REQUIRE(binding.storage == tmv::MicrophysicsScalarStorage::DryAirMixingRatio);

    REQUIRE(state.cloud_number()[0][0][0]
            == Approx(267559.7437901381).epsilon(2.0e-7));
    REQUIRE(state.rain_number()[0][0][1]
            == Approx(3152.0140201030636).epsilon(2.0e-7));
    REQUIRE(state.ice_number()[0][0][2]
            == Approx(1998000.0).epsilon(2.0e-7));
    for (int k = 0; k < 3; ++k)
    {
        const double density = rho[0][0][k];
        REQUIRE(state.water_aerosol()[0][0][k] * density
                == Approx(expected_profile(
                    std::array<double, 3>{50.0, 150.0, 300.0}[k], 350.0e6, 50.0e6))
                       .epsilon(2.0e-7));
        REQUIRE(state.ice_aerosol()[0][0][k] * density
                == Approx(expected_profile(
                    std::array<double, 3>{50.0, 150.0, 300.0}[k], 2.0e6, 0.5e6))
                       .epsilon(2.0e-7));
    }
}

TEST_CASE("Thompson moment initialization matches WRF distribution bounds",
          "[microphysics][thompson][state][literal]")
{
    REQUIRE(tmv::thompson::normalize_cloud_number_m3(1.0e-4, 0.0)
            == Approx(267559.7437901381).epsilon(2.0e-13));
    REQUIRE(tmv::thompson::normalize_rain_number_m3(1.6e-4, 0.0)
            == Approx(2521.611216082451).epsilon(2.0e-13));
    REQUIRE(tmv::thompson::normalize_ice_number_m3(1.5e-5, 0.0)
            == Approx(999000.0).epsilon(2.0e-13));

    const auto clear = tmv::thompson::initialize_number_moments(
        1.0e-12, 0.0, 0.0, 1.0);
    REQUIRE(clear.cloud_kg_dry == 0.0);
    REQUIRE(clear.rain_kg_dry == 0.0);
    REQUIRE(clear.ice_kg_dry == 0.0);
}

TEST_CASE("Thompson prognostic commit is transactional",
          "[microphysics][thompson][state][atomic]")
{
    const Field3D rho = field(1.0f, 1.0f, 1.0f);
    const Field3D zero = field(0.0f, 0.0f, 0.0f);
    tmv::thompson::PrognosticState state;
    state.prepare(rho, zero, zero, zero, {50.0, 150.0, 250.0});
    state.cloud_number_tendency().fill(2.0f);
    state.water_aerosol_tendency().fill(-1.0e6f);

    const Field3D before = state.water_aerosol();
    std::string error;
    REQUIRE(state.validate_update(2.0, error));
    REQUIRE(bitwise_equal(state.water_aerosol(), before));
    state.commit_update(2.0);
    REQUIRE(state.cloud_number()[0][0][0] == 4.0f);
    REQUIRE(state.water_aerosol()[0][0][0]
            == Approx(before[0][0][0] - 2.0e6f));

    const Field3D committed = state.water_aerosol();
    state.water_aerosol_tendency().fill(-1.0e30f);
    REQUIRE_FALSE(state.validate_update(2.0, error));
    REQUIRE(bitwise_equal(state.water_aerosol(), committed));

    state.water_aerosol_tendency().fill(0.0f);
    state.ice_aerosol_tendency().fill(3.0e38f);
    const Field3D ice_before_overflow = state.ice_aerosol();
    REQUIRE_FALSE(state.validate_update(2.0, error));
    REQUIRE(bitwise_equal(state.ice_aerosol(), ice_before_overflow));
}

TEST_CASE("Thompson initialized aerosol ownership rejects reconfiguration atomically",
          "[microphysics][thompson][state][atomic]")
{
    const Field3D rho = field(1.0f, 1.0f, 1.0f);
    const Field3D zero = field(0.0f, 0.0f, 0.0f);
    tmv::thompson::PrognosticState state;
    state.prepare(rho, zero, zero, zero, {0.0, 100.0, 200.0});
    const Field3D before = state.water_aerosol();

    tmv::thompson::AerosolProfile changed;
    changed.water_surface_total_m3 = 200.0e6;
    REQUIRE_THROWS_AS(
        state.prepare(rho, zero, zero, zero, {0.0, 100.0, 200.0}, changed),
        std::invalid_argument);
    REQUIRE(bitwise_equal(state.water_aerosol(), before));

    Field3D invalid = zero;
    invalid[0][0][2] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        state.prepare(rho, zero, zero, invalid, {0.0, 100.0, 200.0}),
        std::invalid_argument);
    REQUIRE(bitwise_equal(state.water_aerosol(), before));
}

TEST_CASE("Thompson reuse validates every private moment and tendency",
          "[microphysics][thompson][state][invalid][atomic]")
{
    const Field3D rho = field(1.0f, 1.0f, 1.0f);
    const Field3D zero = field(0.0f, 0.0f, 0.0f);
    tmv::thompson::PrognosticState state;
    state.prepare(rho, zero, zero, zero, {50.0, 100.0, 150.0});
    const Field3D cloud_before = state.cloud_number();
    const Field3D aerosol_before = state.water_aerosol();

    auto bindings = state.transport_bindings();
    bindings[4].field->resize(1, 1, 2, 1.0f);
    REQUIRE_THROWS_AS(
        state.prepare(rho, zero, zero, zero, {50.0, 100.0, 150.0}),
        std::invalid_argument);
    REQUIRE(bitwise_equal(state.cloud_number(), cloud_before));
    REQUIRE(bitwise_equal(state.water_aerosol(), aerosol_before));

    tmv::thompson::PrognosticState tendency_state;
    tendency_state.prepare(rho, zero, zero, zero, {50.0, 100.0, 150.0});
    const Field3D tendency_cloud_before = tendency_state.cloud_number();
    tendency_state.ice_aerosol_tendency().resize(1, 1, 2, 0.0f);
    std::string error;
    REQUIRE_FALSE(tendency_state.validate_process_state(rho, error));
    REQUIRE_THROWS_AS(
        tendency_state.prepare(
            rho, zero, zero, zero, {50.0, 100.0, 150.0}),
        std::invalid_argument);
    REQUIRE(bitwise_equal(tendency_state.cloud_number(), tendency_cloud_before));
}

TEST_CASE("Thompson extreme finite initialization fails outside OpenMP atomically",
          "[microphysics][thompson][state][thread][invalid][atomic]")
{
    const float tiny = std::numeric_limits<float>::denorm_min();
    const Field3D rho = field(tiny, tiny, tiny);
    const Field3D cloud = field(1.0f, 1.0f, 1.0f);
    const Field3D zero = field(0.0f, 0.0f, 0.0f);
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
#endif
    for (const int thread_count : {1, 4})
    {
#ifdef _OPENMP
        omp_set_num_threads(thread_count);
#else
        (void)thread_count;
#endif
        tmv::thompson::PrognosticState state;
        REQUIRE_THROWS_AS(
            state.prepare(rho, cloud, zero, zero, {50.0, 100.0, 150.0}),
            std::runtime_error);
        for (const auto& binding : state.transport_bindings())
            REQUIRE(binding.field->empty());
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}

TEST_CASE("Thompson prognostic restore rejects malformed state atomically",
          "[microphysics][thompson][state][checkpoint][atomic]")
{
    const Field3D rho = field(1.0f, 0.9f, 0.8f);
    const Field3D zero = field(0.0f, 0.0f, 0.0f);
    tmv::thompson::PrognosticState source;
    source.prepare(rho, zero, zero, zero, {25.0, 75.0, 125.0});
    source.rain_number_tendency().fill(3.0f);
    const auto portable = source.portable_fields();

    tmv::thompson::PrognosticState restored;
    std::string error;
    REQUIRE(restored.restore_fields(portable, error));
    REQUIRE(bitwise_equal(restored.water_aerosol(), source.water_aerosol()));
    REQUIRE(bitwise_equal(restored.rain_number_tendency(),
                          source.rain_number_tendency()));

    auto malformed = portable;
    malformed[4][0][0][1] = -1.0f;
    const Field3D before = restored.ice_aerosol();
    REQUIRE_FALSE(restored.restore_fields(malformed, error));
    REQUIRE(bitwise_equal(restored.ice_aerosol(), before));
}

TEST_CASE("Thompson fixed aerosol initialization is thread-identical",
          "[microphysics][thompson][state][threads]")
{
    Field3D rho(5, 4, 7);
    Field3D zero(5, 4, 7);
    rho.fill(0.83f);
    zero.fill(0.0f);
    std::vector<double> heights(rho.size());
    for (std::size_t index = 0; index < heights.size(); ++index)
        heights[index] = 25.0 + 10.0 * static_cast<double>(index % 7);

    tmv::thompson::PrognosticState one;
    tmv::thompson::PrognosticState four;
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    one.prepare(rho, zero, zero, zero, heights);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    four.prepare(rho, zero, zero, zero, heights);
    REQUIRE(bitwise_equal(one.water_aerosol(), four.water_aerosol()));
    REQUIRE(bitwise_equal(one.ice_aerosol(), four.ice_aerosol()));
}

TEST_CASE("Thompson stochastic columns admit rounded private reservoirs without partial publication",
          "[microphysics][thompson][stochastic][atomic]")
{
    tmv::thompson::PrognosticState state;
    std::vector<Field3D>           initial(10, Field3D(3, 1, 2, 0));
    initial[5](0, 0, 0) = 1;
    initial[1](1, 0, 0) = 1;
    initial[6](1, 0, 0) = -3;
    initial[4](2, 0, 1) = 4;
    initial[9](2, 0, 1) = -2;
    std::string error;
    REQUIRE(state.restore_fields(initial, error));
    Field3D                  density(3, 1, 2, 1);
    const PhysicalCellDomain domain{0, 3, 0, 1, 0, 2};
    std::vector<float>       factors(3, std::numeric_limits<float>::max());
    const auto               requested = factors;
    density(2, 0, 1)                   = 0;
    CHECK_FALSE(state.apply_column_multipliers(.1, density, domain, factors, error));
    CHECK(factors == requested);
    auto unchanged = state.portable_fields();
    for (std::size_t n = 0; n < initial.size(); ++n)
        CHECK(bitwise_equal(initial[n], unchanged[n]));
    density(2, 0, 1) = 1;
    REQUIRE(state.apply_column_multipliers(.1, density, domain, factors, error));
    const auto scaled = state.portable_fields();
    for (int column = 0; column < 3; ++column)
    {
        const int    currency = column == 0 ? 0 : column == 1 ? 1 : 4;
        const int    k        = column == 2 ? 1 : 0;
        const double old      = initial[currency](column, 0, k);
        const float  source   = initial[currency + 5](column, 0, k);
        const double upper    = currency == 0 ? 1999e6 : currency == 1 ? 1e12 : 9999e6;
        const double endpoint = old + .1 * double(scaled[currency + 5](column, 0, k));
        CHECK(endpoint >= 0);
        CHECK(double(float(endpoint)) <= upper);
        const float  next     = std::nextafter(factors[column], std::numeric_limits<float>::infinity());
        const double rejected = old + .1 * double(float(source * next));
        CHECK((rejected < 0 || double(float(rejected)) > upper));
        CHECK(factors[column] <= requested[column]);
    }
    REQUIRE(state.validate_update(.1, domain, error));
}

TEST_CASE("Thompson zero stochastic factors cannot mask invalid inputs", "[microphysics][thompson][stochastic][atomic]")
{
    tmv::thompson::PrognosticState state;
    std::vector<Field3D>           initial(10, Field3D(2, 1, 2, 0));
    std::string                    error;
    REQUIRE(state.restore_fields(initial, error));
    Field3D                  density(2, 1, 2, 1);
    const PhysicalCellDomain domain{0, 2, 0, 1, 0, 2};
    std::vector<float>       factors(2, 0);
    state.ice_aerosol_tendency()(1, 0, 1) = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(state.apply_column_multipliers(1, density, domain, factors, error));
    CHECK(std::isnan(state.ice_aerosol_tendency()(1, 0, 1)));
    state.ice_aerosol_tendency()(1, 0, 1) = 0;
    auto bindings                         = state.transport_bindings();
    (*bindings[0].field)(1, 0, 1)         = float(1999e6);
    REQUIRE(double((*bindings[0].field)(1, 0, 1)) > 1999e6);
    CHECK_FALSE(state.apply_column_multipliers(1, density, domain, factors, error));
    (*bindings[0].field)(1, 0, 1) = 0;
    CHECK_FALSE(state.apply_column_multipliers(1, density, {0, 3, 0, 1, 0, 2}, factors, error));
    factors[0] = std::numeric_limits<float>::infinity();
    CHECK_FALSE(state.apply_column_multipliers(1, density, domain, factors, error));
    factors[0] = -0.0f;
    REQUIRE(state.apply_column_multipliers(1, density, domain, factors, error));
    CHECK_FALSE(std::signbit(factors[0]));
}

TEST_CASE("Thompson stochastic preparation is restart and thread identical",
          "[microphysics][thompson][stochastic][threads]")
{
    std::vector<Field3D> initial(10, Field3D(5, 4, 7, 0));
    for (std::size_t f = 0; f < 5; ++f)
        for (std::size_t n = 0; n < initial[f].size(); ++n)
        {
            initial[f].data()[n]     = float(10 + n + f);
            initial[f + 5].data()[n] = n % 2 ? -7.1f * float(f + 1) : 13.3f * float(f + 1);
        }
    tmv::thompson::PrognosticState one, four, resumed;
    std::string                    error;
    REQUIRE(one.restore_fields(initial, error));
    REQUIRE(four.restore_fields(initial, error));
    Field3D                  density(5, 4, 7, 1);
    const PhysicalCellDomain domain{0, 5, 0, 4, 0, 7};
    std::vector<float>       first(20, 1.7f), second = first;
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    REQUIRE(one.apply_column_multipliers(.3, density, domain, first, error));
    REQUIRE(resumed.restore_fields(one.portable_fields(), error));
    one.commit_update(.3, domain);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    REQUIRE(four.apply_column_multipliers(.3, density, domain, second, error));
    four.commit_update(.3, domain);
    resumed.commit_update(.3, domain);
#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
#endif
    CHECK(first == second);
    const auto a = one.portable_fields(), b = four.portable_fields(), c = resumed.portable_fields();
    for (std::size_t f = 0; f < a.size(); ++f)
    {
        CHECK(bitwise_equal(a[f], b[f]));
        CHECK(bitwise_equal(a[f], c[f]));
    }
}

TEST_CASE("Thompson stochastic admission rejects density aliasing a tendency", "[thompson][stochastic][atomic]")
{
    tmv::thompson::PrognosticState state;
    std::vector<Field3D> initial(10, Field3D(1, 1, 1, 0));
    initial[3](0, 0, 0) = 3e9f;
    initial[5](0, 0, 0) = 1;
    initial[8](0, 0, 0) = 1e9f;
    std::string error;
    REQUIRE(state.restore_fields(initial, error));
    std::vector<float> factors{2};
    REQUIRE_FALSE(state.apply_column_multipliers(1, state.cloud_number_tendency(), {0, 1, 0, 1, 0, 1}, factors, error));
    CHECK(factors[0] == 2);
    const auto after = state.portable_fields();
    for (std::size_t f = 0; f < initial.size(); ++f)
        CHECK(bitwise_equal(initial[f], after[f]));
}
