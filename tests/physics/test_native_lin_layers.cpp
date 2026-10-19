#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "microphysics/schemes/lin/lin.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include "compute/bridge/residency.hpp"
#include "compute/lin_contract.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
void exercise(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    const bool  initialized = initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error);
    INFO(error);
    REQUIRE(initialized);
    constexpr int nx = 5, ny = 5, nz = 6;
    Field3D       pressure(nx, ny, nz, 100000), theta(nx, ny, nz, 300), rho(nx, ny, nz, 1), vapor(nx, ny, nz, .001f),
        rain(nx, ny, nz, .0001f), snow(nx, ny, nz, 0), graupel(nx, ny, nz, 0), hail(nx, ny, nz, 0), zero(nx, ny, nz, 0);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            rain(i, j, nz - 1) = .01f;
    tmv::LinScheme cpu(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), gpu(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (auto* scheme : {&cpu, &gpu})
    {
        scheme->set_physical_cell_domain(nx, ny, nz, {1, nx - 1, 1, ny - 1, 1, nz - 1});
        scheme->set_vertical_spacing_m(50);
    }
    gpu.bind_execution_owners(runtime, nullptr);
    gpu.set_require_native_dispatch(true);
    std::array<Field3D, 8> expected, actual;
    const auto             calculate = [&](tmv::MicrophysicsScheme& scheme, std::array<Field3D, 8>& rates)
    {
        scheme.compute_tendencies(pressure,
                                  theta,
                                  rho,
                                  vapor,
                                  zero,
                                  rain,
                                  zero,
                                  snow,
                                  graupel,
                                  hail,
                                  .1,
                                  rates[0],
                                  rates[1],
                                  rates[2],
                                  rates[3],
                                  rates[4],
                                  rates[5],
                                  rates[6],
                                  rates[7]);
    };
    calculate(cpu, expected);
    calculate(gpu, actual);
    CAPTURE(backend, expected[3](2, 2, nz - 2), actual[3](2, 2, nz - 2));
    CHECK(actual[3](2, 2, nz - 2) == Approx(expected[3](2, 2, nz - 2)).margin(1e-10));
    std::vector<double> layers(rho.size());
    for (int k_begin : {0, 1})
    {
        const int k_end = k_begin ? nz - 1 : nz;
        for (bool variable : {false, true})
        {
            CAPTURE(k_begin, variable);
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        const bool owned = i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k >= k_begin && k < k_end;
                        const auto n     = (i * ny + j) * nz + k;
                        layers[n]        = 50 + 3.1 * k + .7 * i + .2 * j;
                        rho.data()[n]    = owned ? .8f + .04f * k : std::numeric_limits<float>::quiet_NaN();
                        for (auto* field : {&pressure, &theta, &vapor, &rain, &snow, &graupel, &hail})
                            field->data()[n] = owned ? 0 : std::numeric_limits<float>::quiet_NaN();
                        if (owned)
                        {
                            pressure.data()[n] = 100000;
                            theta.data()[n]    = 300;
                            vapor.data()[n]    = .001f;
                            rain.data()[n]     = .0001f + .00002f * k;
                            snow.data()[n]     = .0002f;
                            graupel.data()[n]  = .0003f;
                            hail.data()[n]     = .0004f;
                        }
                    }
            for (auto* scheme : {&cpu, &gpu})
            {
                scheme->set_physical_cell_domain(nx, ny, nz, {1, nx - 1, 1, ny - 1, k_begin, k_end});
                if (variable)
                    scheme->set_vertical_spacing_m(nx, ny, nz, layers);
                else
                    scheme->set_vertical_spacing_m(50);
            }
            invalidate_all_resident_fields_backend(runtime);
            calculate(cpu, expected);
            for (int repeat = 0; repeat < 3; ++repeat)
            {
                std::array<const float*, 8> destinations;
                for (std::size_t field = 0; field < actual.size(); ++field)
                    destinations[field] = actual[field].data();
                for (auto& field : actual)
                    std::fill(field.data(), field.data() + field.size(), std::numeric_limits<float>::quiet_NaN());
                calculate(gpu, actual);
                if (runtime.backend->microphysics_kernels()->supports_atomic_lin_fused() && !runtime.config.validate_parity)
                    for (std::size_t field = 0; field < actual.size(); ++field)
                        CHECK(actual[field].data() == destinations[field]);
                for (std::size_t field = 0; field < actual.size(); ++field)
                    for (std::size_t n = 0; n < rho.size(); ++n)
                        REQUIRE(actual[field].data()[n] == Approx(expected[field].data()[n]).margin(1e-10));
            }
            auto restored = gpu.clone();
            REQUIRE(restored->compute_runtime() == nullptr);
            REQUIRE(restored->parity_runtime() == nullptr);
            REQUIRE(restored->portable_state().reals == gpu.portable_state().reals);
            REQUIRE_THROWS_WITH(calculate(*restored, actual),
                                "native Lin execution requires an explicit compute owner");
            restored->bind_execution_owners(runtime, nullptr);
            std::array<Field3D, 8> restored_rates;
            calculate(*restored, restored_rates);
            for (std::size_t field = 0; field < actual.size(); ++field)
                REQUIRE(std::memcmp(actual[field].data(), restored_rates[field].data(),
                                    rho.size() * sizeof(float)) == 0);
            for (std::size_t field = 0; field < actual.size(); ++field)
                for (std::size_t n = 0; n < rho.size(); ++n)
                {
                    CAPTURE(field, n);
                    CHECK(actual[field].data()[n] == Approx(expected[field].data()[n]).margin(1e-10));
                }
            double removed = 0, surface = 0;
            for (int i = 1; i < nx - 1; ++i)
                for (int j = 1; j < ny - 1; ++j)
                {
                    const auto bottom = (i * ny + j) * nz + k_begin;
                    surface += rho.data()[bottom] * (variable ? layers[bottom] : 50) *
                               cpu.bottom_sedimentation_sink_rate(rho, rain, snow, graupel, hail, i, j, .1);
                    for (int k = k_begin; k < k_end; ++k)
                    {
                        const auto n = (i * ny + j) * nz + k;
                        removed -= rho.data()[n] * (variable ? layers[n] : 50) *
                                   (double(actual[3].data()[n]) + actual[5].data()[n] + actual[6].data()[n] +
                                    actual[7].data()[n]);
                    }
                }
            CHECK(std::abs(removed - surface) < 5e-6 * surface);
        }
    }
    auto* kernels = runtime.backend->microphysics_kernels();
    REQUIRE(kernels);
    LinSedimentationDispatch d{rain.data(),
                               snow.data(),
                               graupel.data(),
                               hail.data(),
                               rho.data(),
                               actual[3].data(),
                               actual[5].data(),
                               actual[6].data(),
                               actual[7].data(),
                               nx,
                               ny,
                               nz,
                               50,
                               std::numeric_limits<float>::max()};
    d.i_begin = d.j_begin = d.k_begin = 1;
    d.i_end                           = nx - 1;
    d.j_end                           = ny - 1;
    d.k_end                           = nz - 1;
    std::vector<float> layer_storage(layers.begin(), layers.end());
    d.layer_thickness_m                  = layer_storage.data();
    const auto before                    = actual;
    layer_storage[(2 * ny + 2) * nz + 2] = 0;
    CHECK_FALSE(kernels->dispatch_lin_sedimentation(d));
    for (std::size_t field = 0; field < actual.size(); ++field)
        CHECK(std::equal(actual[field].data(), actual[field].data() + actual[field].size(), before[field].data()));
    layer_storage[(2 * ny + 2) * nz + 2] = 50;
    d.k_end                              = nz + 1;
    CHECK_FALSE(kernels->dispatch_lin_sedimentation(d));
    d.k_end  = nz - 1;
    d.dqr_dt = d.dqs_dt;
    CHECK_FALSE(kernels->dispatch_lin_sedimentation(d));
    d.dqr_dt                             = actual[3].data();
    layer_storage[(2 * ny + 2) * nz + 2] = float(layers[(2 * ny + 2) * nz + 2]);
    LinPointwiseDispatch point;
    point.theta_data = theta.data();
    point.p_data     = pressure.data();
    point.qv         = vapor.data();
    point.qc         = zero.data();
    point.qr         = rain.data();
    point.qi         = zero.data();
    point.qs         = snow.data();
    point.qg         = graupel.data();
    point.qh         = hail.data();
    point.dtheta_dt  = actual[0].data();
    point.dqv_dt     = actual[1].data();
    point.dqc_dt     = actual[2].data();
    point.dqr_dt     = actual[3].data();
    point.dqi_dt     = actual[4].data();
    point.dqs_dt     = actual[5].data();
    point.dqg_dt     = actual[6].data();
    point.dqh_dt     = actual[7].data();
    point.nx         = nx;
    point.ny         = ny;
    point.nz         = nz;
    point.dt_step    = .1f;
    point.inv_dt     = 10;
    point.Lv_cp      = float(microphysics_constants::L_v / microphysics_constants::cp);
    point.Lf_cp      = float(microphysics_constants::L_f / microphysics_constants::cp);
    point.Ls_cp      = float(microphysics_constants::L_s / microphysics_constants::cp);
    point.i_begin = point.j_begin = point.k_begin = 1;
    point.i_end                                   = nx - 1;
    point.j_end                                   = ny - 1;
    point.k_end                                   = nz - 1;
    invalidate_all_resident_fields_backend(runtime);
    REQUIRE(kernels->dispatch_lin_pointwise(point));
    REQUIRE(kernels->dispatch_lin_sedimentation(d));
    for (std::size_t field = 0; field < actual.size(); ++field)
        for (std::size_t n = 0; n < rho.size(); ++n)
            CHECK(actual[field].data()[n] == Approx(before[field].data()[n]).margin(1e-10));
    if (kernels->supports(MicrophysicsKernelOp::LinFused))
    {
        for (auto& field : actual)
            field.fill(std::numeric_limits<float>::quiet_NaN());
        REQUIRE(kernels->dispatch_lin_fused({point, d}));
        for (std::size_t field = 0; field < actual.size(); ++field)
            for (std::size_t n = 0; n < rho.size(); ++n)
                CHECK(actual[field].data()[n] == Approx(before[field].data()[n]).margin(1e-10));
        const auto density_before = rho;
        point.dtheta_dt           = rho.data();
        CHECK_FALSE(kernels->dispatch_lin_fused({point, d}));
        CHECK(std::memcmp(rho.data(), density_before.data(), rho.size() * sizeof(float)) == 0);
    }
}
} // namespace
TEST_CASE("Native Lin fallout excludes closure-plane reservoirs on Metal", "[metal][lin][layers]")
{
    exercise("metal");
}
TEST_CASE("Native Lin fallout excludes closure-plane reservoirs on Vulkan", "[vulkan][lin][layers]")
{
    exercise("vulkan");
}

TEST_CASE("Lin validation preserves domain and failure atomicity across worker counts",
          "[lin][contract][thread]")
{
#ifdef _OPENMP
    struct RestoreWorkers
    {
        int count = omp_get_max_threads();
        ~RestoreWorkers() { omp_set_num_threads(count); }
    } restore;
    omp_set_num_threads(GENERATE(1, 4));
#endif
    constexpr int nx = 20, ny = 20, nz = 220;
    constexpr std::size_t count = nx * ny * nz;
    constexpr std::size_t last = ((nx - 2) * ny + ny - 2) * nz + nz - 2;
    std::array<std::vector<float>, 9> inputs;
    std::array<std::vector<float>, 8> outputs;
    for (auto& f : inputs) f.assign(count, .001f);
    inputs[0].assign(count, 300);
    inputs[1].assign(count, 100000);
    for (auto& f : outputs) f.assign(count, -7);
    LinPointwiseDispatch point;
    point.theta_data = inputs[0].data(); point.p_data = inputs[1].data();
    point.qv = inputs[2].data(); point.qc = inputs[3].data(); point.qr = inputs[4].data();
    point.qi = inputs[5].data(); point.qs = inputs[6].data(); point.qg = inputs[7].data(); point.qh = inputs[8].data();
    point.dtheta_dt = outputs[0].data(); point.dqv_dt = outputs[1].data(); point.dqc_dt = outputs[2].data();
    point.dqr_dt = outputs[3].data(); point.dqi_dt = outputs[4].data(); point.dqs_dt = outputs[5].data();
    point.dqg_dt = outputs[6].data(); point.dqh_dt = outputs[7].data();
    point.nx = nx; point.ny = ny; point.nz = nz;
    point.i_begin = point.j_begin = point.k_begin = 1;
    point.i_end = nx - 1; point.j_end = ny - 1; point.k_end = nz - 1;
    point.dt_step = .1f; point.inv_dt = 10;
    point.Lv_cp = point.Lf_cp = point.Ls_cp = 1;
    inputs[0][0] = std::numeric_limits<float>::quiet_NaN();
    std::size_t cells = 0, bytes = 0;
    REQUIRE(nimbus::compute::lin::valid_lin_pointwise(point, cells, bytes));
    inputs[5][last] = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(nimbus::compute::lin::valid_lin_pointwise(point, cells, bytes));
    inputs[5][last] = -.001f;
    CHECK_FALSE(nimbus::compute::lin::valid_lin_pointwise(point, cells, bytes));
    inputs[5][last] = .001f;
    std::vector<float> density(count, 1), layers(count, 50);
    LinSedimentationDispatch sed{point.qr, point.qs, point.qg, point.qh, density.data(),
                                 point.dqr_dt, point.dqs_dt, point.dqg_dt, point.dqh_dt, nx, ny, nz, 50, 100};
    sed.i_begin = sed.j_begin = sed.k_begin = 1;
    sed.i_end = nx - 1; sed.j_end = ny - 1; sed.k_end = nz - 1;
    sed.layer_thickness_m = layers.data();
    REQUIRE(nimbus::compute::lin::valid_lin_sedimentation(sed, cells, bytes));
    density[last] = 0;
    CHECK_FALSE(nimbus::compute::lin::valid_lin_sedimentation(sed, cells, bytes));
    density[last] = 1; layers[last] = -1;
    CHECK_FALSE(nimbus::compute::lin::valid_lin_sedimentation(sed, cells, bytes));
    layers[last] = 50;
    const std::array<const float*, 2> result{outputs[0].data(), outputs[7].data()};
    REQUIRE(nimbus::compute::lin::finite_result(point, result));
    for (const auto& field : outputs)
        CHECK(std::all_of(field.begin(), field.end(), [](float x) { return x == -7; }));
    outputs[7][0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(nimbus::compute::lin::finite_result(point, result));
    outputs[7][last] = std::numeric_limits<float>::infinity();
    CHECK_FALSE(nimbus::compute::lin::finite_result(point, result));
}
