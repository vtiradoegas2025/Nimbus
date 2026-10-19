#include "catch2/catch.hpp"

#include "core/runtime/grid_state.hpp"
#include "radar/factory.hpp"
#include "radar/radar.hpp"
#include "radar/runtime.hpp"

#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

class ThrowingRadarOperator final : public RadarSchemeBase
{
public:
    std::unique_ptr<RadarSchemeBase> clone() const override
    {
        return std::make_unique<ThrowingRadarOperator>(*this);
    }

    void initialize(const RadarConfig&, int, int, int) override {}

    void compute(const RadarConfig&, const RadarStateView&, RadarOut& output) override
    {
        output.initialize(2, 2, 2);
        output.ZDR_dB.fill(9.0f);
        throw std::runtime_error("synthetic late radar failure");
    }

    std::string name() const override { return "throwing"; }
};

void require_radar_output_equal(const RadarOut& actual, const RadarOut& expected)
{
    const auto require_field_equal = [](const Field3D& lhs, const Field3D& rhs)
    {
        REQUIRE(lhs.size_r() == rhs.size_r());
        REQUIRE(lhs.size_th() == rhs.size_th());
        REQUIRE(lhs.size_z() == rhs.size_z());
        REQUIRE(std::memcmp(
            lhs.data(), rhs.data(), rhs.size() * sizeof(float)) == 0);
    };
    REQUIRE(actual.NR == expected.NR);
    REQUIRE(actual.NTH == expected.NTH);
    REQUIRE(actual.NZ == expected.NZ);
    require_field_equal(actual.Z_dBZ, expected.Z_dBZ);
    require_field_equal(actual.Vr, expected.Vr);
    require_field_equal(actual.ZDR_dB, expected.ZDR_dB);
    require_field_equal(actual.KDP_deg_km, expected.KDP_deg_km);
    require_field_equal(actual.rhohv, expected.rhohv);
    require_field_equal(actual.Ze_linear, expected.Ze_linear);
    require_field_equal(actual.ZH_dBZ, expected.ZH_dBZ);
    require_field_equal(actual.ZV_dBZ, expected.ZV_dBZ);
    require_field_equal(actual.Ze_rain, expected.Ze_rain);
    require_field_equal(actual.Ze_snow, expected.Ze_snow);
    require_field_equal(actual.Ze_graupel, expected.Ze_graupel);
    require_field_equal(actual.Ze_hail, expected.Ze_hail);
    require_field_equal(actual.Ze_ice, expected.Ze_ice);
}

} // namespace

TEST_CASE("built-in radar operators invalidate recycled-generation state",
          "[radar][runtime][domain_recycling]")
{
    for (const char* scheme_name : {"reflectivity", "velocity", "zdr"})
    {
        CAPTURE(scheme_name);
        auto scheme = create_radar_scheme(scheme_name);
        REQUIRE(scheme);
        REQUIRE(scheme->supports_domain_recycling_invalidation());
        REQUIRE_NOTHROW(
            scheme->invalidate_generation_state_after_domain_recycling());
    }
}

TEST_CASE("radar runtime retains product operators across frames", "[radar][runtime]")
{
    constexpr int nx = 3;
    constexpr int ny = 3;
    constexpr int nz = 3;
    global_grid_geometry.initialize(
        nx, ny, nz, 1000.0, 500.0, 1000.0,
        CoordinateSystem::Cartesian, StaggerType::Collocated);

    Field3D u(nx, ny, nz, 5.0f);
    Field3D v(nx, ny, nz, -2.0f);
    Field3D w(nx, ny, nz, 1.0f);
    Field3D rho(nx, ny, nz, 1.0f);
    Field3D qr(nx, ny, nz, 1.0e-3f);

    RadarStateView state;
    state.NR = nx;
    state.NTH = ny;
    state.NZ = nz;
    state.geometry = &global_grid_geometry;
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.rho = &rho;
    state.qr = &qr;

    RadarRuntimeState runtime;
    RadarOut output;
    RadarRuntimeState reflectivity_only_runtime;
    RadarOut reflectivity_only;
    RadarSystem::compute_reflectivity(
        reflectivity_only_runtime, state, reflectivity_only);
    RadarSystem::compute_all_observables(runtime, state, output);

    REQUIRE(runtime.reflectivity_operator);
    REQUIRE(runtime.velocity_operator);
    REQUIRE(runtime.zdr_operator);
    REQUIRE(output.NR == nx);
    REQUIRE(output.NTH == ny);
    REQUIRE(output.NZ == nz);
    REQUIRE(output.Z_dBZ(1, 1, 1)
            == Approx(reflectivity_only.Z_dBZ(1, 1, 1)));

    auto* const reflectivity = runtime.reflectivity_operator.get();
    auto* const velocity = runtime.velocity_operator.get();
    auto* const zdr = runtime.zdr_operator.get();

    RadarSystem::compute_all_observables(runtime, state, output, 1000.0, 0.0, 0.0);
    REQUIRE(runtime.reflectivity_operator.get() == reflectivity);
    REQUIRE(runtime.velocity_operator.get() == velocity);
    REQUIRE(runtime.zdr_operator.get() == zdr);
    REQUIRE(runtime.radar_x == 1000.0);
    REQUIRE_FALSE(runtime.polarimetric_output.ZDR_dB.empty());

    runtime.selected_config.scheme_id = "reflectivity";
    runtime.selected_scheme = create_radar_scheme("reflectivity");
    runtime.selected_scheme->initialize(runtime.selected_config, nx, ny, nz);
    runtime.selected_output.initialize(nx, ny, nz);
    runtime.selected_output.Z_dBZ[0][0][0] = 47.0f;

    RadarRuntimeState snapshot(runtime);
    REQUIRE(snapshot.selected_scheme);
    REQUIRE(snapshot.selected_scheme.get() != runtime.selected_scheme.get());
    REQUIRE(snapshot.selected_scheme->name() == "reflectivity");
    REQUIRE(snapshot.reflectivity_operator.get() !=
            runtime.reflectivity_operator.get());
    REQUIRE(snapshot.velocity_operator.get() != runtime.velocity_operator.get());
    REQUIRE(snapshot.zdr_operator.get() != runtime.zdr_operator.get());
    REQUIRE(snapshot.selected_output.Z_dBZ.data() !=
            runtime.selected_output.Z_dBZ.data());
    REQUIRE(snapshot.selected_output.Z_dBZ[0][0][0] == Approx(47.0f));
    REQUIRE(snapshot.polarimetric_output.ZDR_dB.data() !=
            runtime.polarimetric_output.ZDR_dB.data());
    REQUIRE(snapshot.polarimetric_output.ZDR_dB(1, 1, 1) ==
            Approx(runtime.polarimetric_output.ZDR_dB(1, 1, 1)));

    RadarOut snapshot_output;
    RadarSystem::compute_all_observables(snapshot, state, snapshot_output,
                                         1000.0, 0.0, 0.0);
    REQUIRE(snapshot_output.NR == nx);
    REQUIRE(snapshot_output.Z_dBZ.size() == output.Z_dBZ.size());

    const float published_dbz = output.Z_dBZ(1, 1, 1);
    rho(1, 1, 1) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        RadarSystem::compute_all_observables(runtime, state, output),
        std::invalid_argument);
    REQUIRE(output.Z_dBZ(1, 1, 1) == published_dbz);
    rho(1, 1, 1) = 1.0f;

    const float published_velocity = output.Vr(1, 1, 1);
    u(1, 1, 1) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        RadarSystem::compute_all_observables(runtime, state, output),
        std::invalid_argument);
    REQUIRE(output.Z_dBZ(1, 1, 1) == published_dbz);
    REQUIRE(output.Vr(1, 1, 1) == published_velocity);
    u(1, 1, 1) = 5.0f;

    RadarStateView resized_state = state;
    resized_state.NR = 2;
    resized_state.NTH = 2;
    resized_state.NZ = 2;
    Field3D resized_rho(2, 2, 2, 1.0f);
    Field3D resized_qr(2, 2, 2, 1.0e-3f);
    resized_state.rho = &resized_rho;
    resized_state.qr = &resized_qr;
    resized_state.u = nullptr;
    resized_state.v = nullptr;
    resized_state.w = nullptr;

    RadarSystem::compute_reflectivity(runtime, resized_state, output);
    REQUIRE(runtime.reflectivity_operator);
    REQUIRE_FALSE(runtime.velocity_operator);
    REQUIRE_FALSE(runtime.zdr_operator);

    runtime.reset();
    REQUIRE_FALSE(runtime.selected_scheme);
    REQUIRE_FALSE(runtime.reflectivity_operator);
    REQUIRE_FALSE(runtime.velocity_operator);
    REQUIRE_FALSE(runtime.zdr_operator);
    REQUIRE(runtime.nr == -1);
}

TEST_CASE("combined radar publication is atomic across late operator failure",
          "[radar][runtime][transactional]")
{
    constexpr int nx = 2;
    constexpr int ny = 2;
    constexpr int nz = 2;
    global_grid_geometry.initialize(
        nx, ny, nz, 1000.0, 500.0, 1000.0,
        CoordinateSystem::Cartesian, StaggerType::Collocated);

    Field3D u(nx, ny, nz, 5.0f);
    Field3D v(nx, ny, nz, -2.0f);
    Field3D w(nx, ny, nz, 1.0f);
    Field3D rho(nx, ny, nz, 1.0f);
    Field3D qr(nx, ny, nz, 1.0e-3f);
    RadarStateView state;
    state.NR = nx;
    state.NTH = ny;
    state.NZ = nz;
    state.geometry = &global_grid_geometry;
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.rho = &rho;
    state.qr = &qr;

    RadarRuntimeState runtime;
    runtime.nr = nx;
    runtime.nth = ny;
    runtime.nz = nz;
    runtime.zdr_operator = std::make_unique<ThrowingRadarOperator>();
    runtime.radar_x = 10.0;
    runtime.radar_y = 20.0;
    runtime.radar_z = 30.0;

    RadarOut output;
    output.initialize(nx, ny, nz);
    output.Z_dBZ.fill(31.0f);
    output.Vr.fill(-7.0f);
    output.ZDR_dB.fill(2.5f);
    output.Ze_linear.fill(1234.0f);
    output.ZH_dBZ.fill(32.0f);
    output.ZV_dBZ.fill(29.5f);
    output.Ze_rain.fill(1.0f);
    output.Ze_snow.fill(2.0f);
    output.Ze_graupel.fill(3.0f);
    output.Ze_hail.fill(4.0f);
    output.Ze_ice.fill(5.0f);
    const RadarOut original = output;

    REQUIRE_THROWS_WITH(
        RadarSystem::compute_all_observables(
            runtime, state, output, 1000.0, 2000.0, 3000.0),
        "synthetic late radar failure");
    require_radar_output_equal(output, original);
    REQUIRE(runtime.radar_x == 10.0);
    REQUIRE(runtime.radar_y == 20.0);
    REQUIRE(runtime.radar_z == 30.0);
}

TEST_CASE("selected reflectivity and velocity publication is atomic",
          "[radar][runtime][transactional][ready]")
{
    constexpr int n = 2;
    global_grid_geometry.initialize(
        n, n, n, 1000.0, 500.0, 1000.0,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    Field3D u(n, n, n, 5.0f), v(n, n, n, -2.0f), w(n, n, n, 1.0f);
    Field3D rho(n, n, n, 1.0f), qr(n, n, n, 1.0e-3f);
    RadarStateView state;
    state.NR = state.NTH = state.NZ = n;
    state.geometry = &global_grid_geometry;
    state.u = &u; state.v = &v; state.w = &w;
    state.rho = &rho; state.qr = &qr;

    RadarRuntimeState runtime;
    runtime.nr = runtime.nth = runtime.nz = n;
    runtime.velocity_operator = std::make_unique<ThrowingRadarOperator>();
    RadarOut output;
    output.initialize(n, n, n);
    output.Z_dBZ.fill(30.0f);
    output.Vr.fill(-4.0f);
    output.Ze_linear.fill(1000.0f);
    const RadarOut original = output;
    RadarProductSelection selection;
    selection.reflectivity = true;
    selection.radial_velocity = true;
    selection.radar_x = 1000.0;

    REQUIRE_THROWS_WITH(
        RadarSystem::compute_observables(runtime, selection, state, output),
        "synthetic late radar failure");
    require_radar_output_equal(output, original);
}

TEST_CASE("ready radar operators are bitwise deterministic across thread counts",
          "[radar][runtime][threads][ready]")
{
    constexpr int nx = 36;
    constexpr int ny = 34;
    constexpr int nz = 28;
    global_grid_geometry.initialize(
        nx, ny, nz, 250.0, 200.0, 300.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    Field3D u(nx, ny, nz), v(nx, ny, nz), w(nx, ny, nz);
    Field3D rho(nx, ny, nz), qr(nx, ny, nz), qs(nx, ny, nz);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                u(i, j, k) = 2.0f + 0.01f * i;
                v(i, j, k) = -1.0f + 0.02f * j;
                w(i, j, k) = 0.005f * k;
                rho(i, j, k) = 1.2f - 0.005f * k;
                qr(i, j, k) = 1.0e-4f * (1 + (i + j + k) % 9);
                qs(i, j, k) = 5.0e-5f * (1 + (2 * i + k) % 7);
            }
    RadarStateView state;
    state.NR = nx; state.NTH = ny; state.NZ = nz;
    state.staggering = StaggerType::CGrid;
    state.geometry = &global_grid_geometry;
    state.u = &u; state.v = &v; state.w = &w;
    state.rho = &rho; state.qr = &qr; state.qs = &qs;
    RadarProductSelection selection;
    selection.reflectivity = true;
    selection.radial_velocity = true;
    selection.apply_scatterer_correction = true;
    selection.radar_x = -1200.0;
    selection.radar_y = 800.0;
    selection.radar_z = 150.0;

    RadarRuntimeState one_runtime;
    RadarRuntimeState four_runtime;
    RadarOut one;
    RadarOut four;
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    RadarSystem::compute_observables(one_runtime, selection, state, one);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    RadarSystem::compute_observables(four_runtime, selection, state, four);
    require_radar_output_equal(four, one);
}

TEST_CASE("radar velocity refreshes Earth geometry after window relocation",
          "[radar][runtime][domain_recycling][geometry]")
{
    constexpr int n = 3;
    global_grid_geometry.initialize(
        n, n, n, 1000.0, 500.0, 1000.0,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    Field3D u(n, n, n, 8.0f), v(n, n, n, 1.0f), w(n, n, n, 0.0f);
    RadarStateView state;
    state.NR = state.NTH = state.NZ = n;
    state.geometry = &global_grid_geometry;
    state.u = &u; state.v = &v; state.w = &w;
    RadarProductSelection selection;
    selection.reflectivity = false;
    selection.radial_velocity = true;
    selection.radar_x = -1000.0;
    selection.radar_y = 500.0;
    RadarRuntimeState runtime;
    RadarOut output;
    RadarSystem::compute_observables(runtime, selection, state, output);
    const float baseline = output.Vr(1, 1, 1);

    state.grid_origin_earth_x_m = 15000.0;
    state.grid_origin_earth_y_m = -6000.0;
    output.invalidate_generation_values();
    runtime.velocity_operator
        ->invalidate_generation_state_after_domain_recycling();
    REQUIRE(output.Vr(1, 1, 1) == 0.0f);
    RadarSystem::compute_observables(runtime, selection, state, output);
    REQUIRE(std::isfinite(output.Vr(1, 1, 1)));
    REQUIRE(output.Vr(1, 1, 1) != Approx(baseline));
}
