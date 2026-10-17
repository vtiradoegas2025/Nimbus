#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/moving_momentum_contract.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/runtime/native_earth_fixed_transport.hpp"
#include "dynamics/native_moving_cgrid_transport.hpp"
#include "terrain/native_earth_geometry.hpp"
#include "terrain/native_conservative_remap.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>

namespace
{
std::array<double, 11> values(const tmv::MovingMomentumTransportLedger& l)
{
    return {l.initial_impulse,
            l.final_impulse,
            l.outward_impulse,
            l.impulse_storage_rounding,
            l.initial_kinetic,
            l.final_kinetic,
            l.outward_kinetic,
            l.spatial_conversion,
            l.finite_step_work,
            l.density_storage_work,
            l.velocity_storage_work};
}
void compare(const tmv::MovingCGridMomentumTransport& actual, const tmv::MovingCGridMomentumTransport& expected)
{
    const std::array<const Field3D*, 3> lhs{&actual.u, &actual.v, &actual.w},
        rhs{&expected.u, &expected.v, &expected.w};
    for (int c = 0; c < 3; ++c)
    {
        CAPTURE(c);
        REQUIRE(lhs[c]->size() == rhs[c]->size());
        for (std::size_t n = 0; n < lhs[c]->size(); ++n)
        {
            if (std::abs(rhs[c]->data()[n]) < std::numeric_limits<float>::min())
                CHECK(lhs[c]->data()[n] == rhs[c]->data()[n]);
            else
                CHECK(lhs[c]->data()[n] == Approx(rhs[c]->data()[n]).epsilon(2e-7));
        }
        const auto   a = values(actual.ledger[c]), b = values(expected.ledger[c]);
        const double impulse = std::max({std::abs(b[0]), std::abs(b[1]), std::abs(b[2]), 1e-90});
        const double kinetic = std::max({std::abs(b[4]), std::abs(b[5]), std::abs(b[6]), 1e-90});
        for (std::size_t f = 0; f < a.size(); ++f)
        {
            CAPTURE(f, a[f], b[f], impulse, kinetic);
            const bool stored = f == 1 || f == 3 || f == 5 || f == 10;
            CHECK(std::abs(a[f] - b[f]) <= (f < 4 ? impulse : kinetic) * (stored ? 3e-7 : 3e-12) + 1e-100);
        }
        CHECK(std::abs(a[1] - a[0] + a[2] - a[3]) <= 5e-12 * impulse + 1e-100);
        CHECK(std::abs(a[5] - a[4] + a[6] - a[7] - a[8] - a[9] - a[10]) <= 5e-12 * kinetic + 1e-100);
    }
}
class RejectSecondMomentum final : public TerrainComputeKernels
{
public:
    explicit RejectSecondMomentum(TerrainComputeKernels& delegate) : delegate_(delegate)
    {
    }
    bool supports_earth_geometry() const override
    {
        return delegate_.supports_earth_geometry();
    }
    bool supports_conservative_remap() const override
    {
        return delegate_.supports_conservative_remap();
    }
    bool supports_moving_momentum() const override
    {
        return delegate_.supports_moving_momentum();
    }
    bool dispatch_earth_geometry(const EarthGeometryDispatch& d) override
    {
        return delegate_.dispatch_earth_geometry(d);
    }
    bool dispatch_conservative_remap(const ConservativeRemapDispatch& d) override
    {
        return delegate_.dispatch_conservative_remap(d);
    }
    bool dispatch_moving_momentum(const MovingMomentumDispatch& d) override
    {
        if (d.constants.operation == 0 && d.constants.component == 1)
            return false;
        const bool result = delegate_.dispatch_moving_momentum(d);
        if (result && d.constants.operation == 0)
            ++completed_components;
        return result;
    }
    int completed_components = 0;

private:
    TerrainComputeKernels& delegate_;
};
void exercise(const char* backend)
{
    using namespace tmv;
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    INFO(error);
    REQUIRE_FALSE(runtime.fallback_active);
    auto* kernels = runtime.backend->terrain_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports_moving_momentum());
    constexpr int nx = 7, ny = 6, nz = 5;
    for (double height : {0.0, 200.0})
        for (int sx : {-1, 0, 1})
            for (int sy : {-1, 0, 1})
                for (bool boundaries : {false, true})
                    for (bool vertical : {false, true})
                    {
                        CAPTURE(backend, height, sx, sy, boundaries, vertical);
                        SimulationStorageState storage{{nx, ny, nz, 400, 350, .1, 275}};
                        storage.grid.geometry.initialize(
                            nx, ny, nz, 400, 350, 275, CoordinateSystem::Cartesian, StaggerType::CGrid);
                        storage.configuration.stagger_type     = StaggerType::CGrid;
                        auto& terrain                          = storage.runtime.terrain;
                        terrain.config.scheme_id               = "bell";
                        terrain.config.coord_id                = "smoothed";
                        terrain.config.horizontal_placement_id = "earth_fixed";
                        terrain.config.ztop                    = nz * 350;
                        terrain.config.bell.h0                 = height;
                        terrain.config.bell.a                  = 850;
                        terrain.config.bell.axisymmetric       = true;
                        EarthFixedTerrainBuildRequest request;
                        request.nr                  = nx;
                        request.nth                 = ny;
                        request.nz                  = nz;
                        request.dx_m                = 400;
                        request.dy_m                = 275;
                        request.grid_origin_earth_m = {-800, -550};
                        request.sampling            = EarthFixedTerrainSampling::CellAverageBilinear;
                        auto initial                = prepare_earth_fixed_terrain_commit(terrain, request, *kernels);
                        REQUIRE(commit_earth_fixed_terrain(initial, terrain));
                        storage.configuration.domain_frame.earth_origin_at_t0_m = request.grid_origin_earth_m;
                        storage.configuration.domain_frame.velocity_ms          = {sx * 12.0, sy * 8.0};
                        for (auto* field :
                             {&storage.fields.rho, &storage.fields.u, &storage.fields.v, &storage.fields.w})
                            field->resize(nx, ny, nz, 0);
                        for (int i = 0; i < nx; ++i)
                            for (int j = 0; j < ny; ++j)
                                for (int k = 0; k < nz; ++k)
                                {
                                    storage.fields.rho(i, j, k) = 1.2f - .05f * k + .003f * i - .002f * j;
                                    storage.fields.u(i, j, k)   = 8 + .3f * i - .2f * j + .1f * k;
                                    storage.fields.v(i, j, k)   = -4 + .1f * i + .4f * j - .03f * k;
                                    storage.fields.w(i, j, k)   = .2f + .03f * i - .04f * k + .01f * j;
                                }
                        auto state  = make_simulation_state_view(storage);
                        auto motion = prepare_earth_fixed_terrain_motion(terrain, *state.domain_frame, 0, 1, *kernels);
                        auto flux =
                            terrain_native::moving_mesh_mass_flux(*kernels, storage.fields.rho, motion.swept_volumes);
                        for (int i = 1; i < nx; ++i)
                            for (int j = 1; j < ny - 1; ++j)
                                for (int k = 0; k < nz; ++k)
                                {
                                    const double swept    = motion.swept_volumes.x(i, j, k);
                                    const double expected = -swept * storage.fields.rho(swept >= 0 ? i : i - 1, j, k);
                                    CHECK(flux.x(i, j, k) == Approx(expected).epsilon(3e-12).margin(1e-20));
                                }
                        for (int i = 1; i < nx - 1; ++i)
                            for (int j = 1; j < ny; ++j)
                                for (int k = 0; k < nz; ++k)
                                {
                                    const double swept    = motion.swept_volumes.y(i, j, k);
                                    const double expected = -swept * storage.fields.rho(i, swept >= 0 ? j : j - 1, k);
                                    CHECK(flux.y(i, j, k) == Approx(expected).epsilon(3e-12).margin(1e-20));
                                }
                        Field3D density;
                        if (vertical)
                        {
                            for (int i = 1; i < nx - 1; ++i)
                                for (int j = 1; j < ny - 1; ++j)
                                    for (int k = 1; k < nz; ++k)
                                        flux.z(i, j, k) =
                                            (sx == 0 ? 1 : sx) * 7000.0 * std::sin(.5 * k + .2 * i - .1 * j);
                            update_moving_mesh_inventory(terrain.metrics,
                                                         motion.terrain.geometry.metrics,
                                                         flux,
                                                         physical_scalar_domain(state.grid),
                                                         storage.fields.rho,
                                                         density,
                                                         0);
                        }
                        else
                            terrain_native::remap_inventory(*kernels,
                                                            storage.fields.rho,
                                                            density,
                                                            storage.fields.rho,
                                                            storage.fields.rho,
                                                            terrain.metrics,
                                                            motion.terrain.geometry.metrics,
                                                            motion.swept_volumes,
                                                            false,
                                                            terrain_native::RemapLowerBound::Positive);
                        const auto original_u = storage.fields.u, original_v = storage.fields.v,
                                   original_w = storage.fields.w;
                        const auto expected   = prepare_moving_cgrid_momentum_transport(
                            state, motion.terrain.geometry.metrics, density, flux, boundaries);
                        const auto actual = prepare_moving_cgrid_momentum_transport(
                            state, motion.terrain.geometry.metrics, density, flux, *kernels, boundaries);
                        compare(actual, expected);
                        if (vertical && height > 0 && sx == 1 && sy == 1)
                        {
                            terrain.config.horizontal_placement_id = "grid_attached";
                            const auto surface_reference           = prepare_moving_cgrid_momentum_transport(
                                state, motion.terrain.geometry.metrics, density, flux, boundaries);
                            const auto surface_native = prepare_moving_cgrid_momentum_transport(
                                state, motion.terrain.geometry.metrics, density, flux, *kernels, boundaries);
                            compare(surface_native, surface_reference);
                            terrain.config.horizontal_placement_id = "earth_fixed";
                        }

                        CHECK(std::memcmp(
                                  storage.fields.u.data(), original_u.data(), original_u.size() * sizeof(float)) == 0);
                        CHECK(std::memcmp(
                                  storage.fields.v.data(), original_v.data(), original_v.size() * sizeof(float)) == 0);
                        CHECK(std::memcmp(
                                  storage.fields.w.data(), original_w.data(), original_w.size() * sizeof(float)) == 0);
                        if (sx == 0 && sy == 0)
                        {
                            const int workers = omp_get_max_threads();
                            omp_set_num_threads(4);
                            const auto repeated = prepare_moving_cgrid_momentum_transport(
                                state, motion.terrain.geometry.metrics, density, flux, *kernels, boundaries);
                            omp_set_num_threads(workers);
                            CHECK(std::memcmp(actual.u.data(), repeated.u.data(), actual.u.size() * sizeof(float)) ==
                                  0);
                            for (int c = 0; c < 3; ++c)
                                CHECK(values(actual.ledger[c]) == values(repeated.ledger[c]));
                        }
                        if (!vertical && sx == 1 && sy == 1)
                        {
                            const auto rho_before      = storage.fields.rho;
                            const auto geometry_before = terrain.metrics.J.values;
                            const auto generation      = terrain.earth_fixed_placement.generation;
                            auto       prepared        = prepare_native_earth_fixed_transport(state, 1, *kernels);
                            const auto reference =
                                prepare_moving_cgrid_momentum_transport(state,
                                                                        prepared.motion.terrain.geometry.metrics,
                                                                        prepared.density,
                                                                        prepared.mass_flux,
                                                                        true);
                            compare(prepared.momentum, reference);
                            CHECK(std::abs(prepared.dry_mass.final_inventory - prepared.dry_mass.initial_inventory +
                                           prepared.dry_mass.boundary_outflow - prepared.dry_mass.storage_rounding) <
                                  3e-12 * prepared.dry_mass.initial_inventory);
                            RejectSecondMomentum reject(*kernels);
                            REQUIRE_THROWS(prepare_native_earth_fixed_transport(state, 1, reject));
                            CHECK(reject.completed_components == 1);
                            CHECK(std::memcmp(storage.fields.rho.data(),
                                              rho_before.data(),
                                              rho_before.size() * sizeof(float)) == 0);
                            CHECK(terrain.metrics.J.values == geometry_before);
                            CHECK(terrain.earth_fixed_placement.generation == generation);

                            auto wrong = density;
                            wrong(2, 2, 2) *= 2;
                            REQUIRE_THROWS(prepare_moving_cgrid_momentum_transport(
                                state, motion.terrain.geometry.metrics, wrong, flux, *kernels, boundaries));
                            CHECK(std::memcmp(storage.fields.u.data(),
                                              original_u.data(),
                                              original_u.size() * sizeof(float)) == 0);
                            CHECK(std::memcmp(storage.fields.v.data(),
                                              original_v.data(),
                                              original_v.size() * sizeof(float)) == 0);
                            CHECK(std::memcmp(storage.fields.w.data(),
                                              original_w.data(),
                                              original_w.size() * sizeof(float)) == 0);
                            auto bad       = flux;
                            bad.z(1, 1, 0) = 1;
                            REQUIRE_THROWS(prepare_moving_cgrid_momentum_transport(
                                state, motion.terrain.geometry.metrics, density, bad, *kernels, boundaries));
                            TerrainComputeKernels unavailable;
                            REQUIRE_THROWS(prepare_moving_cgrid_momentum_transport(
                                state, motion.terrain.geometry.metrics, density, flux, unavailable, boundaries));
                        }
                        if (sx == 1 && sy == 0)
                        {
                            for (int component = 0; component < 3; ++component)
                            {
                                auto* field = component == 0   ? &storage.fields.u
                                              : component == 1 ? &storage.fields.v
                                                               : &storage.fields.w;
                                for (std::size_t n = 0; n < field->size(); ++n)
                                    field->data()[n] =
                                        std::ldexp(float((component == 1 ? -1 : 1) * int(1 + n % 7)), -149);
                            }
                            const auto tiny_ref = prepare_moving_cgrid_momentum_transport(
                                state, motion.terrain.geometry.metrics, density, flux, boundaries);
                            const auto tiny = prepare_moving_cgrid_momentum_transport(
                                state, motion.terrain.geometry.metrics, density, flux, *kernels, boundaries);
                            compare(tiny, tiny_ref);
                            storage.fields.u(0, 0, 0) = 1;
                            const auto mixed_ref      = prepare_moving_cgrid_momentum_transport(
                                state, motion.terrain.geometry.metrics, density, flux, boundaries);
                            const auto mixed = prepare_moving_cgrid_momentum_transport(
                                state, motion.terrain.geometry.metrics, density, flux, *kernels, boundaries);
                            compare(mixed, mixed_ref);
                        }
                    }
    MovingMomentumDispatch d;
    auto&                  p = d.constants;
    p.nx                     = 4;
    p.ny                     = 4;
    p.nz                     = 3;
    p.dx_high = p.dy_high = p.dz_high = 1;
    const auto         count          = nimbus::compute::moving_momentum::cells(d);
    std::vector<float> input(count * MovingMomentumDispatch::input_stride),
        output(count * MovingMomentumDispatch::output_stride, 17);
    d.input  = input.data();
    d.output = output.data();
    for (std::size_t n = 0; n < count; ++n)
    {
        auto* record = input.data() + n * MovingMomentumDispatch::input_stride;
        record[0] = record[1] = record[2] = record[4] = 1;
        record[6]                                     = 2;
        record[7]                                     = -1;
        record[8]                                     = .1f;
    }
    const auto rejected = [&]()
    {
        CHECK_FALSE(kernels->dispatch_moving_momentum(d));
        for (float value : output)
            CHECK(value == 17);
    };
    p.nx = std::numeric_limits<int>::max();
    rejected();
    p.nx    = 4;
    d.input = nullptr;
    rejected();
    d.input  = input.data();
    d.output = input.data() + 1;
    rejected();
    d.output = output.data();
    input[0] = 0;
    rejected();
    input[0] = 1;
    input[3] = 2;
    rejected();
    input[3]          = 0;
    const auto offset = ((1 * 5 + 1) * 4 + 1) * MovingMomentumDispatch::input_stride;
    input[offset + 9] = 100;
    rejected();
    input[offset + 9] = 0;
    for (float flux : {-100.0f, 100.0f})
    {
        for (std::size_t n = 0; n < count; ++n)
            input[n * MovingMomentumDispatch::input_stride + 9] = flux;
        rejected();
    }
    for (std::size_t n = 0; n < count; ++n)
        input[n * MovingMomentumDispatch::input_stride + 9] = 1;
    REQUIRE(kernels->dispatch_moving_momentum(d));
    for (std::size_t n = 0; n < count; ++n)
        CHECK(output[n * MovingMomentumDispatch::output_stride] == 2);
    for (int sign : {-1, 1})
    {
        for (std::size_t n = 0; n < count; ++n)
        {
            input[n * MovingMomentumDispatch::input_stride + 6] = std::ldexp(float(sign * int(1 + n % 7)), -149);
            input[n * MovingMomentumDispatch::input_stride + 9] = .5f;
        }
        REQUIRE(kernels->dispatch_moving_momentum(d));
        for (int j = 1; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
            {
                const auto  n = std::size_t((1 * 5 + j) * 4 + k), previous = n - 20;
                const float expected =
                    static_cast<float>(.5 * (double(input[n * MovingMomentumDispatch::input_stride + 6]) +
                                             input[previous * MovingMomentumDispatch::input_stride + 6]));
                CHECK(output[n * MovingMomentumDispatch::output_stride] == expected);
            }
    }
    for (std::size_t n = 0; n < count; ++n)
        input[n * MovingMomentumDispatch::input_stride + 6] = 2;
    std::fill(output.begin(), output.end(), 17);
    for (std::size_t n = 0; n < count; ++n)
        input[n * MovingMomentumDispatch::input_stride + 9] = 0;
    p.operation       = 1;
    input[offset]     = .5f;
    input[offset + 9] = std::numeric_limits<float>::denorm_min();
    rejected();
    input[offset]     = 1;
    input[offset + 9] = 0;
    p.operation       = 0;
    input[offset + 6] = std::numeric_limits<float>::max();
    rejected();
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MovingMomentum) > 0);
}
} // namespace
TEST_CASE("Native moving C-grid momentum closes impulse and kinetic work on Metal", "[metal][terrain][moving_momentum]")
{
    exercise("metal");
}
TEST_CASE("Native moving C-grid momentum closes impulse and kinetic work on Vulkan",
          "[vulkan][terrain][moving_momentum]")
{
    exercise("vulkan");
}
