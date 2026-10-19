#include "catch2/catch.hpp"
#include "core/runtime/held_source_remap.hpp"
#include "core/runtime/simulation_state.hpp"
#include "compute/column_remap_contract.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>
namespace
{
using namespace tmv;
std::vector<Field3D*> volumes(SimulationStorageState& s)
{
    auto& p = s.runtime.boundary_layer;
    auto& r = s.runtime.radiation;
    auto& c = s.runtime.chaos;
    return {&p.held_u_force_n,
            &p.held_v_force_n,
            &p.held_theta_inventory_rate_kgk_s,
            &p.held_vapor_mass_rate_kg_s,
            &p.held_cloud_liquid_mass_rate_kg_s,
            &p.held_cloud_ice_mass_rate_kg_s,
            &r.held_cell_power_w,
            &s.tendencies.dtke_dt_pbl,
            &c.base_du_dt_pbl,
            &c.base_dv_dt_pbl,
            &c.base_dtheta_dt_pbl,
            &c.base_dqv_dt_pbl,
            &c.base_dtke_dt_pbl};
}
std::vector<std::vector<float>*> surfaces(SimulationStorageState& s)
{
    auto& p = s.runtime.boundary_layer;
    auto& c = s.runtime.chaos;
    return {&p.skin_theta_2d,
            &p.skin_qv_2d,
            &s.surface.ustar,
            &s.surface.tau_u,
            &s.surface.tau_v,
            &s.surface.sensible_heat_flux,
            &s.surface.latent_heat_flux,
            &s.surface.moisture_flux,
            &s.surface.pbl_height,
            &s.surface.precipitation_rate_mm_h,
            &c.base_sfc_ustar,
            &c.base_sfc_tau_u,
            &c.base_sfc_tau_v,
            &c.base_sfc_sensible_heat_flux,
            &c.base_sfc_latent_heat_flux,
            &c.base_sfc_moisture_flux};
}
std::array<std::vector<double>*, 4> fluxes(SimulationStorageState& s)
{
    auto& f = s.runtime.radiation.held_flux_diagnostics;
    return {&f.upward_longwave_wm2, &f.downward_longwave_wm2, &f.upward_shortwave_wm2, &f.downward_shortwave_wm2};
}
struct Fixture
{
    static constexpr int    nx = 5, ny = 4, nz = 3;
    SimulationStorageState  state{{nx, ny, nz, 100, 100, .1, 100}};
    ExtendedRadiationFluxes extended;
    Fixture()
    {
        int field = 0;
        for (auto* v : volumes(state))
        {
            v->resize(nx, ny, nz, 0);
            ++field;
            for (std::size_t n = 0; n < v->size(); ++n)
                v->data()[n] = float(int(n) - 20) * (field % 2 ? -1 : 1) + .125f * field;
        }
        field = 0;
        for (auto* v : surfaces(state))
        {
            v->resize(nx * ny);
            for (std::size_t n = 0; n < v->size(); ++n)
                (*v)[n] = float(++field + 2 * n);
        }
        state.surface.accumulated_rainfall_mm.resize(nx * ny);
        state.surface.accumulated_rainfall_residual_mm.resize(nx * ny);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
            {
                const auto n                                      = i * ny + j;
                state.surface.accumulated_rainfall_mm[n]          = 1e6f + float(n);
                state.surface.accumulated_rainfall_residual_mm[n] = .001f * float(n - 5);
                state.runtime.boundary_layer.slab_columns.push_back(
                    {100.0 + 20 * i + 3 * j, 290.0 + 2 * i - j, .003 + .0003 * j, true});
            }
        auto& f         = state.runtime.radiation.held_flux_diagnostics;
        f.columns_x     = nx - 2;
        f.columns_y     = ny - 2;
        f.levels        = nz;
        const auto live = fluxes(state);
        for (int channel = 0; channel < 4; ++channel)
        {
            extended[channel].resize(nx * ny * (nz + 1));
            live[channel]->resize((nx - 2) * (ny - 2) * (nz + 1));
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    for (int k = 0; k <= nz; ++k)
                    {
                        double value                                   = 10 * channel + 3 * i + .2 * j + .01 * k;
                        extended[channel][(i * ny + j) * (nz + 1) + k] = value;
                        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1)
                            (*live[channel])[((i - 1) * (ny - 2) + j - 1) * (nz + 1) + k] = value;
                    }
        }
        state.time_s                                     = 17;
        state.runtime.radiation.next_update_time_s       = 20;
        state.runtime.boundary_layer.next_update_time_s  = 19;
        state.surface.accumulated_rainfall_last_update_s = 16;
    }
};
std::vector<double> snapshot(Fixture& f)
{
    auto&               s = f.state;
    std::vector<double> result;
    for (auto* v : volumes(s))
        result.insert(result.end(), v->data(), v->data() + v->size());
    for (auto* v : surfaces(s))
        result.insert(result.end(), v->begin(), v->end());
    for (auto* v : {&s.surface.accumulated_rainfall_mm, &s.surface.accumulated_rainfall_residual_mm})
        result.insert(result.end(), v->begin(), v->end());
    for (const auto& slab : s.runtime.boundary_layer.slab_columns)
        for (double value : {slab.h_m, slab.theta_m_k, slab.qv_m_kgkg, double(slab.initialized)})
            result.push_back(value);
    for (auto* v : fluxes(s))
        result.insert(result.end(), v->begin(), v->end());
    result.push_back(s.runtime.radiation.held_atmospheric_power_w);
    return result;
}
void run(Fixture& f, double sx, double sy, TerrainComputeKernels* kernels)
{
    if (kernels)
        remap_held_owners(f.state, sx, sy, f.extended, *kernels);
    else
        remap_held_owners(f.state, sx, sy, f.extended);
}
void analytic(TerrainComputeKernels* kernels)
{
    for (double sx : {-.3, 0.0, .3})
        for (double sy : {-.2, 0.0, .2})
        {
            CAPTURE(sx, sy);
            Fixture    before, after;
            const auto original = snapshot(before);
            run(after, sx, sy, kernels);
            auto&        s  = after.state;
            const double ax = std::abs(sx), ay = std::abs(sy);
            const auto   blend = [&](const auto& at, int i, int j)
            {
                const int x = std::clamp(i + (sx >= 0 ? 1 : -1), 0, 4), y = std::clamp(j + (sy >= 0 ? 1 : -1), 0, 3);
                return (1 - ax - ay) * at(i, j) + ax * at(x, j) + ay * at(i, y);
            };
            const auto v0 = volumes(before.state), v1 = volumes(s);
            for (std::size_t f = 0; f < v0.size(); ++f)
                for (int i = 0; i < 5; ++i)
                    for (int j = 0; j < 4; ++j)
                        for (int k = 0; k < 3; ++k)
                        {
                            const double expected =
                                blend([&](int a, int b) { return double((*v0[f])(a, b, k)); }, i, j);
                            CHECK((*v1[f])(i, j, k) == Approx(expected).margin(1e-5));
                        }
            const auto a0 = surfaces(before.state), a1 = surfaces(s);
            for (std::size_t f = 0; f < a0.size(); ++f)
                for (int i = 0; i < 5; ++i)
                    for (int j = 0; j < 4; ++j)
                        CHECK((*a1[f])[i * 4 + j] ==
                              Approx(blend([&](int a, int b) { return double((*a0[f])[a * 4 + b]); }, i, j))
                                  .epsilon(2e-7));
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 4; ++j)
                {
                    const int    n    = i * 4 + j;
                    const auto&  slab = s.runtime.boundary_layer.slab_columns[n];
                    const double h = blend([&](int a, int b)
                                           { return before.state.runtime.boundary_layer.slab_columns[a * 4 + b].h_m; },
                                           i,
                                           j);
                    const double heat = blend(
                        [&](int a, int b)
                        {
                            const auto& c = before.state.runtime.boundary_layer.slab_columns[a * 4 + b];
                            return c.h_m * c.theta_m_k;
                        },
                        i,
                        j);
                    const double water = blend(
                        [&](int a, int b)
                        {
                            const auto& c = before.state.runtime.boundary_layer.slab_columns[a * 4 + b];
                            return c.h_m * c.qv_m_kgkg;
                        },
                        i,
                        j);
                    CHECK(slab.h_m == Approx(h).epsilon(3e-12));
                    CHECK(slab.h_m * slab.theta_m_k == Approx(heat).epsilon(3e-12));
                    CHECK(slab.h_m * slab.qv_m_kgkg == Approx(water).epsilon(3e-12));
                    const double rain = blend(
                        [&](int a, int b)
                        {
                            return double(before.state.surface.accumulated_rainfall_mm[a * 4 + b]) +
                                   before.state.surface.accumulated_rainfall_residual_mm[a * 4 + b];
                        },
                        i,
                        j);
                    CHECK(double(s.surface.accumulated_rainfall_mm[n]) +
                              s.surface.accumulated_rainfall_residual_mm[n] ==
                          Approx(rain).margin(3e-7));
                }
            const auto live = fluxes(s);
            for (int f = 0; f < 4; ++f)
                for (int i = 0; i < 5; ++i)
                    for (int j = 0; j < 4; ++j)
                        for (int k = 0; k < 4; ++k)
                        {
                            const double expected =
                                blend([&](int a, int b) { return before.extended[f][(a * 4 + b) * 4 + k]; }, i, j);
                            CHECK(after.extended[f][(i * 4 + j) * 4 + k] == Approx(expected).margin(1e-10));
                            if (i > 0 && i < 4 && j > 0 && j < 3)
                                CHECK((*live[f])[((i - 1) * 2 + j - 1) * 4 + k] ==
                                      after.extended[f][(i * 4 + j) * 4 + k]);
                        }
            double total = 0;
            for (int i = 1; i < 4; ++i)
                for (int j = 1; j < 3; ++j)
                    for (int k = 0; k < 3; ++k)
                        total += s.runtime.radiation.held_cell_power_w(i, j, k);
            CHECK(total == s.runtime.radiation.held_atmospheric_power_w);
            CHECK(s.time_s == 17);
            CHECK(s.runtime.radiation.next_update_time_s == 20);
            CHECK(s.runtime.boundary_layer.next_update_time_s == 19);
            CHECK(s.surface.accumulated_rainfall_last_update_s == 16);
            CHECK(snapshot(before) == original);
        }
    Fixture    failed;
    const auto saved          = snapshot(failed);
    const auto extended       = failed.extended;
    failed.extended[3].back() = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(run(failed, .2, .1, kernels));
    CHECK(snapshot(failed) == saved);
    failed.extended = extended;
    REQUIRE_THROWS(run(failed, .4, .2, kernels));
    CHECK(snapshot(failed) == saved);
    CHECK(failed.extended == extended);
    failed.state.surface.accumulated_rainfall_residual_mm.pop_back();
    const auto malformed = snapshot(failed);
    REQUIRE_THROWS(run(failed, .2, .1, kernels));
    CHECK(snapshot(failed) == malformed);
}
void extremes(TerrainComputeKernels* kernels)
{
    Fixture no_motion;
    run(no_motion, 0, 0, nullptr);
    no_motion.state.surface.accumulated_rainfall_residual_mm[0] = 1e-30f;
    const auto saved                                            = snapshot(no_motion);
    const auto ext                                              = no_motion.extended;
    run(no_motion, 0, 0, kernels);
    CHECK(snapshot(no_motion) == saved);
    CHECK(no_motion.extended == ext);

    Fixture tiny, reference;
    for (auto* f : {&tiny, &reference})
    {
        for (auto* v : volumes(f->state))
            for (std::size_t n = 0; n < v->size(); ++n)
                v->data()[n] = std::numeric_limits<float>::denorm_min() * float(int(n % 17) - 8);
        for (auto* v : surfaces(f->state))
            for (std::size_t n = 0; n < v->size(); ++n)
                (*v)[n] = std::numeric_limits<float>::denorm_min() * float(n % 17);
        const auto live = fluxes(f->state);
        for (int c = 0; c < 4; ++c)
        {
            for (auto& value : f->extended[c])
                value *= 1e-220;
            for (auto& value : *live[c])
                value *= 1e-220;
        }
        for (auto& slab : f->state.runtime.boundary_layer.slab_columns)
        {
            slab.h_m *= 1e-100;
            slab.theta_m_k *= 1e100;
        }
    }
    run(reference, .25, -.25, nullptr);
    run(tiny, .25, -.25, kernels);
    const auto a = volumes(tiny.state), b = volumes(reference.state);
    for (std::size_t f = 0; f < a.size(); ++f)
        for (std::size_t n = 0; n < a[f]->size(); ++n)
            CHECK(a[f]->data()[n] == b[f]->data()[n]);
    const auto sa = surfaces(tiny.state), sb = surfaces(reference.state);
    for (std::size_t f = 0; f < sa.size(); ++f)
        CHECK(*sa[f] == *sb[f]);
    for (int c = 0; c < 4; ++c)
        for (std::size_t n = 0; n < tiny.extended[c].size(); ++n)
            CHECK(std::abs(tiny.extended[c][n] - reference.extended[c][n]) <=
                  3e-12 * std::max(std::abs(reference.extended[c][n]), 1e-230));
    const auto& slab_a = tiny.state.runtime.boundary_layer.slab_columns;
    const auto& slab_b = reference.state.runtime.boundary_layer.slab_columns;
    for (std::size_t n = 0; n < slab_a.size(); ++n)
    {
        CHECK(slab_a[n].h_m == Approx(slab_b[n].h_m).epsilon(3e-12));
        CHECK(slab_a[n].theta_m_k == Approx(slab_b[n].theta_m_k).epsilon(3e-12));
        CHECK(slab_a[n].qv_m_kgkg == Approx(slab_b[n].qv_m_kgkg).epsilon(3e-12));
    }
    Fixture mixed, mixed_reference;
    for (auto* fixture : {&mixed, &mixed_reference})
    {
        for (auto* field : volumes(fixture->state))
            for (int i = 0; i < Fixture::nx; ++i)
                for (int j = 0; j < Fixture::ny; ++j)
                    for (int k = 0; k < Fixture::nz; ++k)
                        (*field)(i, j, k) = i == 0 ? 1e10f : (i == 1 ? -2e10f : 1e-36f);
        const auto live = fluxes(fixture->state);
        for (int c = 0; c < 4; ++c)
            for (int i = 0; i < Fixture::nx; ++i)
                for (int j = 0; j < Fixture::ny; ++j)
                    for (int k = 0; k <= Fixture::nz; ++k)
                    {
                        const double value                                                  = i == 0 ? 1e200 : 1e-220;
                        fixture->extended[c][(i * Fixture::ny + j) * (Fixture::nz + 1) + k] = value;
                        if (i > 0 && i < Fixture::nx - 1 && j > 0 && j < Fixture::ny - 1)
                            (*live[c])[((i - 1) * (Fixture::ny - 2) + j - 1) * (Fixture::nz + 1) + k] = value;
                    }
    }
    run(mixed_reference, .25, .25, nullptr);
    run(mixed, .25, .25, kernels);
    const auto mixed_a = volumes(mixed.state), mixed_b = volumes(mixed_reference.state);
    for (std::size_t f = 0; f < mixed_a.size(); ++f)
        for (std::size_t n = 0; n < mixed_a[f]->size(); ++n)
            CHECK(mixed_a[f]->data()[n] == mixed_b[f]->data()[n]);
    for (int c = 0; c < 4; ++c)
        for (std::size_t n = 0; n < mixed.extended[c].size(); ++n)
            CHECK(std::abs(mixed.extended[c][n] - mixed_reference.extended[c][n]) <=
                  3e-12 * std::abs(mixed_reference.extended[c][n]));

    Fixture    stale;
    const auto unchanged = snapshot(stale);
    stale.extended[0][(1 * 4 + 1) * 4] += 1;
    REQUIRE_THROWS(run(stale, .2, .1, kernels));
    CHECK(snapshot(stale) == unchanged);
    Fixture slab_bad;
    slab_bad.state.runtime.boundary_layer.slab_columns.back().initialized = false;
    const auto invalid                                                    = snapshot(slab_bad);
    REQUIRE_THROWS(run(slab_bad, .2, .1, kernels));
    CHECK(snapshot(slab_bad) == invalid);
    SimulationStorageState  empty{{5, 4, 3, 100, 100, .1, 100}};
    ExtendedRadiationFluxes none;
    if (kernels)
        REQUIRE_NOTHROW(remap_held_owners(empty, .2, .1, none, *kernels));
    else
        REQUIRE_NOTHROW(remap_held_owners(empty, .2, .1, none));
}

class Capture final : public TerrainComputeKernels
{
public:
    TerrainComputeKernels&                         backend;
    nimbus::compute::abi::ColumnRemapPushConstants constants{};
    std::vector<float>                             input;
    int                                            calls = 0, fail_after = -1;
    explicit Capture(TerrainComputeKernels& k) : backend(k)
    {
    }
    bool supports_column_remap() const override
    {
        return true;
    }
    bool dispatch_column_remap(const ColumnRemapDispatch& d) override
    {
        if (fail_after >= 0 && calls++ >= fail_after)
            return false;
        constants = d.constants;
        input.assign(d.input, d.input + nimbus::compute::column_remap::cells(d) * ColumnRemapDispatch::input_stride);
        return backend.dispatch_column_remap(d);
    }
};
void hardware(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    INFO(error);
    REQUIRE_FALSE(runtime.fallback_active);
    auto* kernels = runtime.backend->terrain_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports_column_remap());
    analytic(kernels);
    extremes(kernels);
    Fixture   one, four;
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    run(one, .25, -.25, kernels);
    omp_set_num_threads(4);
    run(four, .25, -.25, kernels);
    omp_set_num_threads(threads);
    CHECK(snapshot(one) == snapshot(four));
    CHECK(one.extended == four.extended);
    Capture capture(*kernels);
    Fixture f;
    capture.fail_after = 25;
    const auto saved   = snapshot(f);
    const auto ext     = f.extended;
    REQUIRE_THROWS(run(f, .2, .1, &capture));
    CHECK(snapshot(f) == saved);
    CHECK(f.extended == ext);
    CHECK(capture.calls == 26);
    capture.fail_after = -1;
    run(f, .2, .1, &capture);
    ColumnRemapDispatch d;
    d.constants = capture.constants;
    d.input     = capture.input.data();
    std::vector<float> output(nimbus::compute::column_remap::cells(d) * ColumnRemapDispatch::output_stride, 17);
    d.output            = output.data();
    const auto rejected = [&]()
    {
        CHECK_FALSE(kernels->dispatch_column_remap(d));
        for (float v : output)
            CHECK(v == 17);
    };
    d.constants.nx = std::numeric_limits<int>::max();
    rejected();
    d.constants = capture.constants;
    d.input     = nullptr;
    rejected();
    d.input  = capture.input.data();
    d.output = capture.input.data() + 1;
    rejected();
    d.output            = output.data();
    d.constants.sx_high = .6;
    d.constants.sx_low  = 0;
    rejected();
    d.constants      = capture.constants;
    capture.input[0] = std::numeric_limits<float>::infinity();
    rejected();
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::ColumnRemap) > 0);
}
} // namespace
TEST_CASE("CPU held-source remapping preserves slab and compensated inventories atomically", "[cpu][held_remap]")
{
    analytic(nullptr);
    extremes(nullptr);
}
TEST_CASE("Held-source remapping preserves slab and compensated inventories on Metal", "[metal][held_remap]")
{
    hardware("metal");
}
TEST_CASE("Held-source remapping preserves slab and compensated inventories on Vulkan", "[vulkan][held_remap]")
{
    hardware("vulkan");
}
