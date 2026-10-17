/**
 * @file test_gpu_parity.cpp
 * @brief GPU parity tests — verify CPU and GPU dispatch produce matching results.
 *
 * When GPU is unavailable, tests report SKIPPED (not failed).
 * When GPU IS available, we compare GPU output against CPU reference
 * with tight tolerances. Divergence means the GPU shader is wrong.
 *
 * Tolerances:
 *   - Kessler pointwise: max_abs < 1e-4 (float32 warm-rain processes)
 *   - Diffusion: max_abs < 1e-4
 */
#include "catch2/catch.hpp"
#include "compute/support/sgs_scalar_boundary.hpp"
#include "compute/support/acoustic_unrepaired.hpp"
#include "compute/compute_backend.hpp"
#include "compute/parity_check.hpp"
#include "microphysics/microphysics_base.hpp"
#include "microphysics/schemes/lin/lin.hpp"
#include "numerics/diffusion/schemes/explicit/explicit.hpp"
#include "core/infra/hardware_info.hpp"
#include "core/infra/domain_recycling.hpp"
#include "core/field/field3d.hpp"
#include "core/runtime/simulation.hpp"
#include "numerics/advection/cartesian_mass_flux.hpp"
#include "boundary_layer/schemes/mynn/mynn_stability.hpp"
#include "boundary_layer/schemes/mynn/mynn.hpp"
#include "compute/support/mynn_batch_fixture.hpp"
#include "compute/support/rrtmg_batch_fixture.hpp"
#include "compute/support/agrid_complete_native_fixture.hpp"
#include "chaos/base/earth_correlated_noise.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace
{

bool try_init_gpu()
{
    std::string error;
    global_compute_backend_config.backend        = "vulkan";
    global_compute_backend_config.allow_fallback = false;
    bool ok                                      = initialize_compute_backend_runtime(error);
    if (!ok)
    {
        global_compute_backend_config.backend        = "cpu";
        global_compute_backend_config.allow_fallback = true;
        initialize_compute_backend_runtime(error);
        return false;
    }
    return true;
}

void shutdown_backend()
{
    shutdown_compute_backend_runtime();
}

void require_parity(const char*                      name,
                    const float*                     cpu,
                    const float*                     gpu,
                    int                              nr,
                    int                              nth,
                    int                              nz,
                    double                           abs_tolerance,
                    nimbus::parity::ComparisonDomain domain)
{
    const auto result = nimbus::parity::compare_field(cpu, gpu, nr, nth, nz, {0.0, abs_tolerance}, domain);
    INFO(name << ": domain=" << nimbus::parity::domain_name(domain) << " max_abs=" << result.max_abs
              << " max_rel=" << result.max_rel << " at (" << result.worst_i << "," << result.worst_j << ","
              << result.worst_k << ") expected=" << result.worst_cpu << " actual=" << result.worst_gpu
              << " compared=" << result.cells_compared << " nonfinite=" << result.nonfinite_cells
              << " invalid=" << (result.invalid_reason != nullptr ? result.invalid_reason : "none"));
    REQUIRE(result.passed);
}

void require_microphysics_apply_parity(MicrophysicsComputeKernels* kernels,
                                       Field3D&                    theta_state,
                                       Field3D&                    qv_state,
                                       Field3D&                    qc_state,
                                       Field3D&                    qr_state,
                                       Field3D&                    qi_state,
                                       Field3D&                    qs_state,
                                       Field3D&                    qg_state,
                                       Field3D&                    qh_state,
                                       const Field3D&              dtheta,
                                       const Field3D&              dqv,
                                       const Field3D&              dqc,
                                       const Field3D&              dqr,
                                       const Field3D&              dqi,
                                       const Field3D&              dqs,
                                       const Field3D&              dqg,
                                       const Field3D&              dqh,
                                       int                         nx,
                                       int                         ny,
                                       int                         nz,
                                       float                       dt_step,
                                       int                         i_begin = 0,
                                       int                         i_end   = -1,
                                       int                         j_begin = 0,
                                       int                         j_end   = -1)
{
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports(MicrophysicsKernelOp::MicrophysicsApply));

    constexpr float theta_fallback = 300.0f;
    constexpr float theta_min      = 150.0f;
    constexpr float theta_max      = 700.0f;
    constexpr float max_theta_step = 20.0f;
    constexpr float qv_max         = 0.1f;
    constexpr float hydro_max      = 0.1f;
    if (i_end < 0)
        i_end = nx;
    if (j_end < 0)
        j_end = ny;

    const std::size_t                 cells = theta_state.size();
    std::vector<float>                zero(cells, 0.0f);
    std::array<float*, 8>             state    = {theta_state.data(),
                                                  qv_state.data(),
                                                  qc_state.data(),
                                                  qr_state.data(),
                                                  qi_state.data(),
                                                  qs_state.data(),
                                                  qg_state.data(),
                                                  qh_state.data()};
    const std::array<const float*, 8> tendency = {
        dtheta.data(), dqv.data(), dqc.data(), dqr.data(), dqi.data(), dqs.data(), dqg.data(), dqh.data()};
    std::array<std::vector<float>, 8> expected;
    for (int f = 0; f < 8; ++f)
    {
        expected[f].assign(state[f], state[f] + cells);
    }
    for (int i = i_begin; i < i_end; ++i)
        for (int j = j_begin; j < j_end; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t n =
                    (static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(j)) *
                        static_cast<std::size_t>(nz) +
                    static_cast<std::size_t>(k);
                const float integrated_theta = tendency[0][n] * dt_step;
                const float limited_theta    = std::clamp(integrated_theta, -max_theta_step, max_theta_step);
                expected[0][n]               = std::clamp(expected[0][n] + limited_theta, theta_min, theta_max);
                expected[1][n]               = std::clamp(expected[1][n] + tendency[1][n] * dt_step, 0.0f, qv_max);
                for (int f = 2; f < 8; ++f)
                {
                    expected[f][n] = std::clamp(expected[f][n] + tendency[f][n] * dt_step, 0.0f, hydro_max);
                }
            }

    std::vector<unsigned int>       stats(static_cast<std::size_t>(10 + nz + nx), 0u);
    const MicrophysicsApplyDispatch dispatch{state[0],
                                             state[1],
                                             state[2],
                                             state[3],
                                             state[4],
                                             state[5],
                                             state[6],
                                             state[7],
                                             tendency[0],
                                             tendency[1],
                                             tendency[2],
                                             tendency[3],
                                             tendency[4],
                                             tendency[5],
                                             tendency[6],
                                             tendency[7],
                                             zero.data(),
                                             zero.data(),
                                             zero.data(),
                                             nx,
                                             ny,
                                             nz,
                                             i_begin,
                                             i_end,
                                             j_begin,
                                             j_end,
                                             dt_step,
                                             theta_fallback,
                                             theta_min,
                                             theta_max,
                                             max_theta_step,
                                             qv_max,
                                             hydro_max,
                                             stats.data(),
                                             static_cast<int>(stats.size())};
    REQUIRE(kernels->dispatch_microphysics_apply(dispatch));

    static const char* const names[8] = {
        "theta apply", "qv apply", "qc apply", "qr apply", "qi apply", "qs apply", "qg apply", "qh apply"};
    for (int f = 0; f < 8; ++f)
    {
        require_parity(
            names[f], expected[f].data(), state[f], nx, ny, nz, 1.0e-6, nimbus::parity::ComparisonDomain::FullField);
    }
}

} // namespace

TEST_CASE("Vulkan MYNN level-2.5 stability matches the canonical CPU closure atomically", "[vulkan][gpu_parity][mynn]")
{
    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping MYNN stability parity");
        return;
    }
    auto* backend = mutable_compute_backend();
    REQUIRE(backend != nullptr);
    REQUIRE(backend->supports_boundary_layer(BoundaryLayerKernelOp::MynnLevel25Stability));
    auto* kernels = backend->boundary_layer_kernels();
    REQUIRE(kernels != nullptr);

    const std::array<float, 6>          gm{{0.04f, 0.04f, 0.04f, 0.04f, 0.0f, 0.5f}};
    const std::array<float, 6>          gh{{-0.002f, 0.0f, 0.004f, -0.002f, 0.001f, -0.01f}};
    const std::array<float, 6>          length{{10.0f, 10.0f, 10.0f, 1.0f, 10.0f, 5.0f}};
    const std::array<float, 6>          q2{{0.01f, 0.01f, 0.01f, 100.0f, 0.01f, 0.05f}};
    std::array<std::array<float, 6>, 5> output;
    for (auto& field : output)
        field.fill(-9876.5f);
    MynnLevel25StabilityDispatch dispatch{gm.data(),
                                          gh.data(),
                                          length.data(),
                                          q2.data(),
                                          output[0].data(),
                                          output[1].data(),
                                          output[2].data(),
                                          output[3].data(),
                                          output[4].data(),
                                          static_cast<int>(gm.size())};
    const auto before = backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MynnLevel25Stability);
    REQUIRE(kernels->dispatch_mynn_level25_stability(dispatch));
    REQUIRE(backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MynnLevel25Stability) == before + 1);
    for (std::size_t index = 0; index < gm.size(); ++index)
    {
        const auto expected = tmv::mynn::level25_stability_functions({gm[index], gh[index], length[index], q2[index]});
        CAPTURE(index, gm[index], gh[index], length[index], q2[index]);
        REQUIRE(output[0][index] == Approx(expected.sm).epsilon(3.0e-5).margin(3.0e-6));
        REQUIRE(output[1][index] == Approx(expected.sh).epsilon(3.0e-5).margin(3.0e-6));
        REQUIRE(output[2][index] == Approx(expected.gradient_richardson).epsilon(3.0e-5).margin(3.0e-6));
        REQUIRE(output[3][index] == Approx(expected.flux_richardson).epsilon(3.0e-5).margin(3.0e-6));
        REQUIRE(output[4][index] == Approx(expected.equilibrium_limiter).epsilon(3.0e-5).margin(3.0e-6));
    }
    const auto preserved    = output;
    auto       invalid_q2   = q2;
    invalid_q2[1]           = 0.0f;
    dispatch.q_squared_m2s2 = invalid_q2.data();
    REQUIRE_FALSE(kernels->dispatch_mynn_level25_stability(dispatch));
    REQUIRE(std::memcmp(output.data(), preserved.data(), sizeof(output)) == 0);
    REQUIRE(backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MynnLevel25Stability) == before + 1);
    shutdown_backend();
}

TEST_CASE("Vulkan executes the composed MYNN column batch with CPU parity", "[vulkan][gpu_parity][mynn][batch]")
{
    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping composed MYNN batch parity");
        return;
    }

    auto* backend = mutable_compute_backend();
    REQUIRE(backend != nullptr);
    auto* kernels = backend->boundary_layer_kernels();
    REQUIRE(kernels != nullptr);

    auto       fixture = make_mynn_batch_fixture();
    const bool stretched = GENERATE(false, true);
    if (stretched)
        for (auto& column : fixture.storage)
            for (std::size_t face = 1; face < column.z_interface.size(); ++face)
                column.z_interface[face] = column.z_interface[face - 1]
                    + 50.0 + 20.0 * static_cast<double>(face);

    MYNNScheme cpu;
    MYNNScheme native;
    cpu.initialize(fixture.config);
    native.initialize(fixture.config);

    std::vector<BoundaryLayerTendencies>  expected(fixture.views.size());
    std::vector<BoundaryLayerDiagnostics> expected_diagnostics(fixture.views.size());
    for (std::size_t column = 0; column < fixture.views.size(); ++column)
    {
        cpu.compute_column(
            fixture.config, fixture.surface, fixture.views[column], expected[column], &expected_diagnostics[column]);
    }

    std::vector<BoundaryLayerTendencies>  actual(fixture.views.size());
    std::vector<BoundaryLayerDiagnostics> actual_diagnostics(fixture.views.size());
    const auto before = backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MynnLevel25Stability);
    REQUIRE(native.compute_columns_native(fixture.config,
                                          fixture.surface,
                                          fixture.views.data(),
                                          fixture.views.size(),
                                          actual.data(),
                                          actual_diagnostics.data(),
                                          *kernels));
    REQUIRE(backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MynnLevel25Stability) == before + 1);

    for (std::size_t column = 0; column < actual.size(); ++column)
    {
        const std::vector<double>* expected_fields[] = {&expected[column].dudt_pbl,
                                                        &expected[column].dvdt_pbl,
                                                        &expected[column].dthetadt_pbl,
                                                        &expected[column].dqvdt_pbl,
                                                        &expected[column].dtkedt_pbl};
        const std::vector<double>* actual_fields[]   = {&actual[column].dudt_pbl,
                                                        &actual[column].dvdt_pbl,
                                                        &actual[column].dthetadt_pbl,
                                                        &actual[column].dqvdt_pbl,
                                                        &actual[column].dtkedt_pbl};
        for (std::size_t field = 0; field < 5; ++field)
        {
            for (std::size_t level = 0; level < expected_fields[field]->size(); ++level)
            {
                CAPTURE(column, field, level);
                REQUIRE((*actual_fields[field])[level] ==
                        Approx((*expected_fields[field])[level]).epsilon(3.0e-5).margin(3.0e-7));
            }
        }
        REQUIRE(actual_diagnostics[column].pbl_height == expected_diagnostics[column].pbl_height);
    }

    shutdown_backend();
}

TEST_CASE("Vulkan RRTMG longwave spectral transfer matches clear and cloudy CPU columns atomically",
          "[vulkan][gpu_parity][rrtmg][longwave]")
{
    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping RRTMG longwave parity");
        return;
    }
    auto* backend = mutable_compute_backend();
    REQUIRE(backend != nullptr);
    REQUIRE(backend->supports_radiation(RadiationKernelOp::RrtmgLongwaveTransfer));
    auto* kernels = backend->radiation_kernels();
    REQUIRE(kernels != nullptr);
    auto                          fixture = make_rrtmg_longwave_fixture(9);
    std::vector<float>            upward(static_cast<std::size_t>(fixture.layers + 1), -4321.0f);
    std::vector<float>            downward(static_cast<std::size_t>(fixture.layers + 1), -4321.0f);
    RrtmgLongwaveTransferDispatch dispatch{fixture.absorption.data(),
                                           fixture.planck_fraction.data(),
                                           fixture.layer_planck.data(),
                                           fixture.interface_planck.data(),
                                           fixture.surface_planck.data(),
                                           fixture.band_width.data(),
                                           nullptr,
                                           upward.data(),
                                           downward.data(),
                                           fixture.layers,
                                           fixture.emissivity,
                                           fixture.precipitable_water_cm};
    const auto before = backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::RrtmgLongwaveTransfer);
    for (const bool cloudy : {false, true})
    {
        dispatch.cloud_absorption_optical_depth = cloudy ? fixture.cloud_absorption.data() : nullptr;
        REQUIRE(kernels->dispatch_rrtmg_longwave_transfer(dispatch));
        const auto& expected = cloudy ? fixture.expected_cloudy : fixture.expected_clear;
        for (std::size_t interface_index = 0; interface_index < upward.size(); ++interface_index)
        {
            CAPTURE(cloudy, interface_index);
            REQUIRE(upward[interface_index] ==
                    Approx(expected.upward_wm2[interface_index]).epsilon(2.0e-4).margin(2.0e-4));
            REQUIRE(downward[interface_index] ==
                    Approx(expected.downward_wm2[interface_index]).epsilon(2.0e-4).margin(2.0e-4));
        }
    }
    REQUIRE(backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::RrtmgLongwaveTransfer) == before + 2);
    const auto preserved_up   = upward;
    const auto preserved_down = downward;
    fixture.band_width[3]     = 0.0f;
    REQUIRE_FALSE(kernels->dispatch_rrtmg_longwave_transfer(dispatch));
    REQUIRE(std::memcmp(upward.data(), preserved_up.data(), upward.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(downward.data(), preserved_down.data(), downward.size() * sizeof(float)) == 0);
    REQUIRE(backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::RrtmgLongwaveTransfer) == before + 2);
    shutdown_backend();
}

TEST_CASE("Vulkan RRTMG shortwave spectral transfer matches clear and cloudy CPU columns atomically",
          "[vulkan][gpu_parity][rrtmg][shortwave]")
{
    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping RRTMG shortwave parity");
        return;
    }
    auto* backend = mutable_compute_backend();
    REQUIRE(backend != nullptr);
    REQUIRE(backend->supports_radiation(RadiationKernelOp::RrtmgShortwaveTransfer));
    auto* kernels = backend->radiation_kernels();
    REQUIRE(kernels != nullptr);
    auto                           fixture = make_rrtmg_shortwave_fixture(9);
    std::vector<float>             upward(static_cast<std::size_t>(fixture.layers + 1), -4321.0f);
    std::vector<float>             downward(static_cast<std::size_t>(fixture.layers + 1), -4321.0f);
    RrtmgShortwaveTransferDispatch dispatch{fixture.absorption.data(),
                                            fixture.rayleigh.data(),
                                            fixture.solar_source.data(),
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            upward.data(),
                                            downward.data(),
                                            fixture.layers,
                                            fixture.cosine_zenith,
                                            fixture.solar_irradiance_wm2,
                                            fixture.surface_albedo};
    const auto before = backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::RrtmgShortwaveTransfer);
    for (const bool cloudy : {false, true})
    {
        dispatch.cloud_optical_depth            = cloudy ? fixture.cloud_depth.data() : nullptr;
        dispatch.cloud_single_scattering_albedo = cloudy ? fixture.cloud_omega.data() : nullptr;
        dispatch.cloud_asymmetry                = cloudy ? fixture.cloud_asymmetry.data() : nullptr;
        REQUIRE(kernels->dispatch_rrtmg_shortwave_transfer(dispatch));
        const auto& expected = cloudy ? fixture.expected_cloudy : fixture.expected_clear;
        for (std::size_t interface_index = 0; interface_index < upward.size(); ++interface_index)
        {
            CAPTURE(cloudy, interface_index);
            REQUIRE(upward[interface_index] ==
                    Approx(expected.upward_wm2[interface_index]).epsilon(4.0e-4).margin(4.0e-4));
            REQUIRE(downward[interface_index] ==
                    Approx(expected.downward_wm2[interface_index]).epsilon(4.0e-4).margin(4.0e-4));
        }
    }
    auto                           thick = make_rrtmg_shortwave_fixture(9, 3);
    std::vector<float>             thick_up(static_cast<std::size_t>(thick.layers + 1), -4321.0f);
    std::vector<float>             thick_down(static_cast<std::size_t>(thick.layers + 1), -4321.0f);
    RrtmgShortwaveTransferDispatch thick_dispatch{thick.absorption.data(),
                                                  thick.rayleigh.data(),
                                                  thick.solar_source.data(),
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  thick_up.data(),
                                                  thick_down.data(),
                                                  thick.layers,
                                                  thick.cosine_zenith,
                                                  thick.solar_irradiance_wm2,
                                                  thick.surface_albedo};
    REQUIRE(kernels->dispatch_rrtmg_shortwave_transfer(thick_dispatch));
    for (std::size_t interface_index = 0; interface_index < thick_up.size(); ++interface_index)
    {
        CAPTURE(interface_index);
        REQUIRE(thick_up[interface_index] ==
                Approx(thick.expected_clear.upward_wm2[interface_index]).epsilon(4.0e-4).margin(4.0e-4));
        REQUIRE(thick_down[interface_index] ==
                Approx(thick.expected_clear.downward_wm2[interface_index]).epsilon(4.0e-4).margin(4.0e-4));
    }
    REQUIRE(backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::RrtmgShortwaveTransfer) == before + 3);
    const auto preserved_up   = upward;
    const auto preserved_down = downward;
    dispatch.cloud_asymmetry  = nullptr;
    REQUIRE_FALSE(kernels->dispatch_rrtmg_shortwave_transfer(dispatch));
    REQUIRE(std::memcmp(upward.data(), preserved_up.data(), upward.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(downward.data(), preserved_down.data(), downward.size() * sizeof(float)) == 0);
    REQUIRE(backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::RrtmgShortwaveTransfer) == before + 3);
    shutdown_backend();
}

TEST_CASE("Vulkan RRTMG batch transfers preserve column isolation and publish atomically",
          "[vulkan][gpu_parity][rrtmg][batch]")
{
    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping RRTMG batch parity");
        return;
    }
    auto* backend = mutable_compute_backend();
    REQUIRE(backend != nullptr);
    auto* kernels = backend->radiation_kernels();
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports(RadiationKernelOp::RrtmgLongwaveTransferBatch));
    REQUIRE(kernels->supports(RadiationKernelOp::RrtmgShortwaveTransferBatch));

    constexpr int                      columns      = 4;
    constexpr int                      layers       = 7;
    const std::size_t                  result_count = static_cast<std::size_t>(columns * (layers + 1));
    auto                               longwave     = make_rrtmg_longwave_batch_fixture(columns, layers);
    std::vector<float>                 lw_up(result_count, -901.0f);
    std::vector<float>                 lw_down(result_count, -902.0f);
    RrtmgLongwaveTransferBatchDispatch lw_dispatch{longwave.absorption.data(),
                                                   longwave.planck_fraction.data(),
                                                   longwave.layer_planck.data(),
                                                   longwave.interface_planck.data(),
                                                   longwave.surface_planck.data(),
                                                   longwave.band_width.data(),
                                                   longwave.cloud_absorption.data(),
                                                   longwave.emissivity.data(),
                                                   longwave.precipitable_water_cm.data(),
                                                   lw_up.data(),
                                                   lw_down.data(),
                                                   columns,
                                                   layers};
    const auto                         lw_before =
        backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::RrtmgLongwaveTransferBatch);
    REQUIRE(kernels->dispatch_rrtmg_longwave_transfer_batch(lw_dispatch));
    for (int column = 0; column < columns; ++column)
        for (int interface_index = 0; interface_index <= layers; ++interface_index)
        {
            const std::size_t index    = static_cast<std::size_t>(column * (layers + 1) + interface_index);
            const auto&       expected = longwave.columns_data[static_cast<std::size_t>(column)].expected_cloudy;
            CAPTURE(column, interface_index);
            REQUIRE(
                lw_up[index] ==
                Approx(expected.upward_wm2[static_cast<std::size_t>(interface_index)]).epsilon(2.0e-4).margin(2.0e-4));
            REQUIRE(lw_down[index] == Approx(expected.downward_wm2[static_cast<std::size_t>(interface_index)])
                                          .epsilon(2.0e-4)
                                          .margin(2.0e-4));
        }
    REQUIRE(backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::RrtmgLongwaveTransferBatch) ==
            lw_before + 1);
    const auto preserved_lw_up   = lw_up;
    const auto preserved_lw_down = lw_down;
    longwave.band_width[4]       = 0.0f;
    REQUIRE_FALSE(kernels->dispatch_rrtmg_longwave_transfer_batch(lw_dispatch));
    REQUIRE(std::memcmp(lw_up.data(), preserved_lw_up.data(), result_count * sizeof(float)) == 0);
    REQUIRE(std::memcmp(lw_down.data(), preserved_lw_down.data(), result_count * sizeof(float)) == 0);

    auto                                shortwave = make_rrtmg_shortwave_batch_fixture(columns, layers);
    std::vector<float>                  sw_up(result_count, -903.0f);
    std::vector<float>                  sw_down(result_count, -904.0f);
    RrtmgShortwaveTransferBatchDispatch sw_dispatch{shortwave.absorption.data(),
                                                    shortwave.rayleigh.data(),
                                                    shortwave.solar_source.data(),
                                                    shortwave.cloud_depth.data(),
                                                    shortwave.cloud_omega.data(),
                                                    shortwave.cloud_asymmetry.data(),
                                                    shortwave.cosine_zenith.data(),
                                                    shortwave.solar_irradiance_wm2.data(),
                                                    shortwave.surface_albedo.data(),
                                                    sw_up.data(),
                                                    sw_down.data(),
                                                    columns,
                                                    layers};
    const auto                          sw_before =
        backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::RrtmgShortwaveTransferBatch);
    REQUIRE(kernels->dispatch_rrtmg_shortwave_transfer_batch(sw_dispatch));
    for (int column = 0; column < columns; ++column)
        for (int interface_index = 0; interface_index <= layers; ++interface_index)
        {
            const std::size_t index    = static_cast<std::size_t>(column * (layers + 1) + interface_index);
            const auto&       expected = shortwave.columns_data[static_cast<std::size_t>(column)].expected_cloudy;
            CAPTURE(column, interface_index);
            REQUIRE(
                sw_up[index] ==
                Approx(expected.upward_wm2[static_cast<std::size_t>(interface_index)]).epsilon(4.0e-4).margin(4.0e-4));
            REQUIRE(sw_down[index] == Approx(expected.downward_wm2[static_cast<std::size_t>(interface_index)])
                                          .epsilon(4.0e-4)
                                          .margin(4.0e-4));
        }
    REQUIRE(backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::RrtmgShortwaveTransferBatch) ==
            sw_before + 1);
    const auto preserved_sw_up                                   = sw_up;
    const auto preserved_sw_down                                 = sw_down;
    shortwave.cloud_omega[static_cast<std::size_t>(layers * 14)] = 1.1f;
    REQUIRE_FALSE(kernels->dispatch_rrtmg_shortwave_transfer_batch(sw_dispatch));
    REQUIRE(std::memcmp(sw_up.data(), preserved_sw_up.data(), result_count * sizeof(float)) == 0);
    REQUIRE(std::memcmp(sw_down.data(), preserved_sw_down.data(), result_count * sizeof(float)) == 0);
    shutdown_backend();
}

// ---- Kessler pointwise GPU parity ----

TEST_CASE("GPU Kessler pointwise parity (tol 1e-4)", "[vulkan][gpu_parity]")
{
    NR  = 16;
    NTH = 16;
    NZ  = 16;
    dz  = 250.0; // sedimentation reads the global: vt_cap = 0.9 dz/dt
                 // and the rho0 stratification both depend on it

    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping Kessler parity test");
        return;
    }

    auto* backend = mutable_compute_backend();
    auto* kernels = backend != nullptr ? backend->microphysics_kernels() : nullptr;
    if (kernels == nullptr || !kernels->supports(MicrophysicsKernelOp::KesslerPointwise))
    {
        WARN("GPU does not support Kessler dispatch — skipping");
        shutdown_backend();
        return;
    }
    // Mirror the production driver lifecycle so pointwise -> sedimentation
    // exact-generation residency is exercised rather than intentionally
    // disabled at generation zero.
    backend->begin_resident_step();

    // Production contract parity (rewritten with SCIENCE.md C5):
    // the SAME scheme call, GPU backend up vs shut down. The old form
    // compared the bare shader on un-adjusted inputs against the full
    // scheme - which only passed because the scheme used to DROP the
    // saturation-adjustment commit on its GPU branch (the C5 bug); once
    // the commit runs on both branches, the bare-shader comparison
    // diverges by exactly the satadj heating.
    theta.resize(NR, NTH, NZ, 300.0f);
    p.resize(NR, NTH, NZ, 100000.0f);
    rho.resize(NR, NTH, NZ, 1.2f);
    qv.resize(NR, NTH, NZ, 0.012f);
    qc.resize(NR, NTH, NZ, 0.002f);
    qr.resize(NR, NTH, NZ, 0.001f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0001f);
    qh.resize(NR, NTH, NZ, 0.00005f);
    // Stratified live density: a uniform rho makes the sedimentation
    // density terms (fall_factor, recv_weight) identity, leaving the
    // shader's stratified path unverified against the CPU.
    rho0_base.resize(NZ);
    for (int k = 0; k < NZ; ++k)
    {
        rho0_base[static_cast<size_t>(k)] = 1.2 * std::exp(-(k * dz) / 8000.0);
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                rho(i, j, k) = static_cast<float>(rho0_base[static_cast<size_t>(k)] *
                                                  (1.0 + 0.02 * std::sin(0.3 * i) * std::cos(0.4 * j)));
    }

    auto kessler = create_microphysics_scheme("kessler");
    kessler->set_vertical_spacing_m(dz);
    Field3D gpu_dt(NR, NTH, NZ, 0.0f), gpu_dqv(NR, NTH, NZ, 0.0f);
    Field3D gpu_dqc(NR, NTH, NZ, 0.0f), gpu_dqr(NR, NTH, NZ, 0.0f);
    Field3D gpu_dqi(NR, NTH, NZ, 0.0f), gpu_dqs(NR, NTH, NZ, 0.0f);
    Field3D gpu_dqg(NR, NTH, NZ, 0.0f), gpu_dqh(NR, NTH, NZ, 0.0f);
    kessler->compute_tendencies(p,
                                theta,
                                rho,
                                qv,
                                qc,
                                qr,
                                qi,
                                qs,
                                qg,
                                qh,
                                1.0,
                                gpu_dt,
                                gpu_dqv,
                                gpu_dqc,
                                gpu_dqr,
                                gpu_dqi,
                                gpu_dqs,
                                gpu_dqg,
                                gpu_dqh);

    // The application must consume Kessler's authoritative rain field.
    // theta/qv/qc/qi/qs/qg/qh are legitimate
    // misses: pointwise physics consumed saturation-adjusted qv/qc working
    // copies, not these prognostic arrays. Preserve the initial state for the
    // CPU tendency reference below.
    std::array<std::vector<float>, 8> initial_state;
    const std::array<float*, 8>       state_ptrs = {
        theta.data(), qv.data(), qc.data(), qr.data(), qi.data(), qs.data(), qg.data(), qh.data()};
    for (int f = 0; f < 8; ++f)
    {
        initial_state[f].assign(state_ptrs[f], state_ptrs[f] + theta.size());
    }
    require_microphysics_apply_parity(kernels,
                                      theta,
                                      qv,
                                      qc,
                                      qr,
                                      qi,
                                      qs,
                                      qg,
                                      qh,
                                      gpu_dt,
                                      gpu_dqv,
                                      gpu_dqc,
                                      gpu_dqr,
                                      gpu_dqi,
                                      gpu_dqs,
                                      gpu_dqg,
                                      gpu_dqh,
                                      NR,
                                      NTH,
                                      NZ,
                                      1.0f);
    for (int f = 0; f < 8; ++f)
    {
        std::copy(initial_state[f].begin(), initial_state[f].end(), state_ptrs[f]);
    }

    // Native C-grid application owns only the scalar interior; its closure
    // storage must remain bit-identical for the boundary-condition owner.
    backend->begin_resident_step();
    require_microphysics_apply_parity(kernels,
                                      theta,
                                      qv,
                                      qc,
                                      qr,
                                      qi,
                                      qs,
                                      qg,
                                      qh,
                                      gpu_dt,
                                      gpu_dqv,
                                      gpu_dqc,
                                      gpu_dqr,
                                      gpu_dqi,
                                      gpu_dqs,
                                      gpu_dqg,
                                      gpu_dqh,
                                      NR,
                                      NTH,
                                      NZ,
                                      1.0f,
                                      1,
                                      NR - 1,
                                      1,
                                      NTH - 1);
    for (int f = 0; f < 8; ++f)
    {
        std::copy(initial_state[f].begin(), initial_state[f].end(), state_ptrs[f]);
    }

    shutdown_backend();

    Field3D cpu_dt(NR, NTH, NZ, 0.0f), cpu_dqv(NR, NTH, NZ, 0.0f);
    Field3D cpu_dqc(NR, NTH, NZ, 0.0f), cpu_dqr(NR, NTH, NZ, 0.0f);
    Field3D cpu_dqi(NR, NTH, NZ, 0.0f), cpu_dqs(NR, NTH, NZ, 0.0f);
    Field3D cpu_dqg(NR, NTH, NZ, 0.0f), cpu_dqh(NR, NTH, NZ, 0.0f);
    kessler->compute_tendencies(p,
                                theta,
                                rho,
                                qv,
                                qc,
                                qr,
                                qi,
                                qs,
                                qg,
                                qh,
                                1.0,
                                cpu_dt,
                                cpu_dqv,
                                cpu_dqc,
                                cpu_dqr,
                                cpu_dqi,
                                cpu_dqs,
                                cpu_dqg,
                                cpu_dqh);

    // Heating is now O(5 K/s) at this state (real latent magnitudes);
    // the fp32 shader vs fp64 CPU band scales with it.
    using nimbus::parity::ComparisonDomain;
    require_parity("dtheta_dt", cpu_dt.data(), gpu_dt.data(), NR, NTH, NZ, 1e-3, ComparisonDomain::FullField);
    require_parity("dqv_dt", cpu_dqv.data(), gpu_dqv.data(), NR, NTH, NZ, 1e-4, ComparisonDomain::FullField);
    require_parity("dqc_dt", cpu_dqc.data(), gpu_dqc.data(), NR, NTH, NZ, 1e-4, ComparisonDomain::FullField);
    require_parity("dqr_dt", cpu_dqr.data(), gpu_dqr.data(), NR, NTH, NZ, 1e-4, ComparisonDomain::FullField);
    require_parity("dqg_dt", cpu_dqg.data(), gpu_dqg.data(), NR, NTH, NZ, 1e-4, ComparisonDomain::FullField);
    require_parity("dqh_dt", cpu_dqh.data(), gpu_dqh.data(), NR, NTH, NZ, 1e-4, ComparisonDomain::FullField);
}

TEST_CASE("GPU Kessler honors collocated physical-cell ownership", "[vulkan][gpu_parity][kessler][agrid][SCI-106]")
{
    shutdown_backend();
    NR                              = 10;
    NTH                             = 9;
    NZ                              = 8;
    dz                              = 180.0;
    const PhysicalCellDomain domain = physical_scalar_domain(NR, NTH, NZ, StaggerType::Collocated);

    theta.resize(NR, NTH, NZ, 0.0f);
    p.resize(NR, NTH, NZ, 0.0f);
    rho.resize(NR, NTH, NZ, 0.0f);
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const bool physical = i >= domain.i_begin && i < domain.i_end && j >= domain.j_begin &&
                                      j < domain.j_end && k >= domain.k_begin && k < domain.k_end;
                if (!physical)
                {
                    // Finite but violently nonphysical closure data: any
                    // accidental processing produces an unmistakable signal.
                    theta(i, j, k) = 650.0f;
                    p(i, j, k)     = 250.0f;
                    rho(i, j, k)   = 0.02f;
                    qv(i, j, k)    = 0.09f;
                    qc(i, j, k)    = 0.08f;
                    qr(i, j, k)    = 0.07f;
                    qg(i, j, k)    = 0.06f;
                    qh(i, j, k)    = 0.05f;
                    continue;
                }
                const float pattern = static_cast<float>((3 * i + 5 * j + 7 * k) % 11);
                p(i, j, k)          = 92000.0f - 700.0f * static_cast<float>(k);
                theta(i, j, k)      = 294.0f + 0.25f * pattern;
                rho(i, j, k)        = 1.12f - 0.035f * static_cast<float>(k) + 0.002f * pattern;
                qv(i, j, k)         = 0.009f + 0.0007f * pattern;
                qc(i, j, k)         = 0.00025f * static_cast<float>((i + j + k) % 9);
                qr(i, j, k)         = 0.00015f * static_cast<float>((2 * i + j + k) % 8);
            }

    auto cpu_scheme = create_microphysics_scheme("kessler");
    cpu_scheme->set_vertical_spacing_m(dz);
    cpu_scheme->set_physical_cell_domain(NR, NTH, NZ, domain);
    Field3D cpu_dt, cpu_dqv, cpu_dqc, cpu_dqr;
    Field3D cpu_dqi, cpu_dqs, cpu_dqg, cpu_dqh;
    cpu_scheme->compute_tendencies(p,
                                   theta,
                                   rho,
                                   qv,
                                   qc,
                                   qr,
                                   qi,
                                   qs,
                                   qg,
                                   qh,
                                   2.0,
                                   cpu_dt,
                                   cpu_dqv,
                                   cpu_dqc,
                                   cpu_dqr,
                                   cpu_dqi,
                                   cpu_dqs,
                                   cpu_dqg,
                                   cpu_dqh);

    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping collocated Kessler parity test");
        return;
    }
    auto* backend = mutable_compute_backend();
    auto* kernels = backend != nullptr ? backend->microphysics_kernels() : nullptr;
    if (kernels == nullptr || !kernels->supports(MicrophysicsKernelOp::KesslerPointwise) ||
        !kernels->supports(MicrophysicsKernelOp::KesslerSedimentation))
    {
        WARN("GPU does not support complete Kessler dispatch — skipping");
        shutdown_backend();
        return;
    }
    backend->begin_resident_step();

    auto gpu_scheme = create_microphysics_scheme("kessler");
    gpu_scheme->set_vertical_spacing_m(dz);
    gpu_scheme->set_physical_cell_domain(NR, NTH, NZ, domain);
    Field3D gpu_dt, gpu_dqv, gpu_dqc, gpu_dqr;
    Field3D gpu_dqi, gpu_dqs, gpu_dqg, gpu_dqh;
    gpu_scheme->compute_tendencies(p,
                                   theta,
                                   rho,
                                   qv,
                                   qc,
                                   qr,
                                   qi,
                                   qs,
                                   qg,
                                   qh,
                                   2.0,
                                   gpu_dt,
                                   gpu_dqv,
                                   gpu_dqc,
                                   gpu_dqr,
                                   gpu_dqi,
                                   gpu_dqs,
                                   gpu_dqg,
                                   gpu_dqh);

    using nimbus::parity::ComparisonDomain;
    require_parity(
        "collocated Kessler dtheta", cpu_dt.data(), gpu_dt.data(), NR, NTH, NZ, 1e-3, ComparisonDomain::FullField);
    require_parity(
        "collocated Kessler dqv", cpu_dqv.data(), gpu_dqv.data(), NR, NTH, NZ, 1e-4, ComparisonDomain::FullField);
    require_parity(
        "collocated Kessler dqc", cpu_dqc.data(), gpu_dqc.data(), NR, NTH, NZ, 1e-4, ComparisonDomain::FullField);
    require_parity(
        "collocated Kessler dqr", cpu_dqr.data(), gpu_dqr.data(), NR, NTH, NZ, 1e-4, ComparisonDomain::FullField);
    require_parity(
        "collocated Kessler dqi", cpu_dqi.data(), gpu_dqi.data(), NR, NTH, NZ, 0.0, ComparisonDomain::FullField);
    require_parity(
        "collocated Kessler dqs", cpu_dqs.data(), gpu_dqs.data(), NR, NTH, NZ, 0.0, ComparisonDomain::FullField);
    require_parity(
        "collocated Kessler dqg", cpu_dqg.data(), gpu_dqg.data(), NR, NTH, NZ, 0.0, ComparisonDomain::FullField);
    require_parity(
        "collocated Kessler dqh", cpu_dqh.data(), gpu_dqh.data(), NR, NTH, NZ, 0.0, ComparisonDomain::FullField);

    bool                                closures_are_exact_zero = true;
    bool                                physical_response       = false;
    const std::array<const Field3D*, 8> gpu_tendencies          = {
        &gpu_dt, &gpu_dqv, &gpu_dqc, &gpu_dqr, &gpu_dqi, &gpu_dqs, &gpu_dqg, &gpu_dqh};
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const bool physical = i >= domain.i_begin && i < domain.i_end && j >= domain.j_begin &&
                                      j < domain.j_end && k >= domain.k_begin && k < domain.k_end;
                if (physical)
                {
                    physical_response =
                        physical_response || std::abs(gpu_dqv(i, j, k)) > 0.0f || std::abs(gpu_dqr(i, j, k)) > 0.0f;
                }
                else
                {
                    for (const Field3D* field : gpu_tendencies)
                        closures_are_exact_zero = closures_are_exact_zero && (*field)(i, j, k) == 0.0f;
                }
            }
    REQUIRE(closures_are_exact_zero);
    REQUIRE(physical_response);
    shutdown_backend();
}

TEST_CASE("GPU diffusion matches live-density FV A/C-grid control volumes", "[vulkan][gpu_parity][diffusion]")
{
    constexpr int nx = 11;
    constexpr int ny = 10;
    constexpr int nz = 8;
    Field3D       rho(nx, ny, nz, 1.0f);
    Field3D       u(nx, ny, nz, 0.0f);
    Field3D       v(nx, ny, nz, 0.0f);
    Field3D       w(nx, ny, nz, 0.0f);
    Field3D       theta(nx, ny, nz, 0.0f);
    Field3D       qv(nx, ny, nz, 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                rho(i, j, k)   = 0.8f + 0.017f * static_cast<float>(i) + 0.009f * static_cast<float>(j) +
                                 0.013f * static_cast<float>(k);
                const float x  = static_cast<float>(i - 4);
                const float y  = static_cast<float>(j - 5);
                const float z  = static_cast<float>(k - 3);
                u(i, j, k)     = 3.0f + 0.11f * x * x - 0.04f * y + 0.03f * z;
                v(i, j, k)     = -2.0f + 0.05f * y * y + 0.02f * x * z;
                w(i, j, k)     = 0.3f + 0.07f * z * z - 0.01f * x * y;
                theta(i, j, k) = 300.0f + 0.08f * x * x + 0.06f * y * y + 0.13f * z * z;
                qv(i, j, k)    = 0.009f + 0.00001f * x * x + 0.00002f * y * y + 0.00003f * z * z;
            }

    GridMetrics metrics;
    metrics.dx = 725.0;
    metrics.dy = 910.0;
    metrics.dz = {80.0, 95.0, 120.0, 155.0, 205.0, 270.0, 350.0, 460.0};

    DiffusionConfig config;
    config.K_h          = 43.0;
    config.K_v          = 71.0;
    config.dt_diffusion = 0.1;
    config.apply_to     = "scalars_and_momentum";
    DiffusionStateView state;
    state.u          = &u;
    state.v          = &v;
    state.w          = &w;
    state.theta      = &theta;
    state.qv         = &qv;
    state.rho        = &rho;
    state.grid       = &metrics;
    state.c_grid     = true;
    state.periodic_x = true;
    state.periodic_y = false;
    ExplicitDiffusionScheme cpu_scheme;
    cpu_scheme.initialize(config);
    DiffusionTendencies cpu;
    cpu_scheme.compute_diffusion_tendencies(config, state, cpu, nullptr);

    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping");
        return;
    }
    auto* backend = mutable_compute_backend();
    auto* kernels = backend != nullptr ? backend->diffusion_kernels() : nullptr;
    if (kernels == nullptr || !kernels->supports(DiffusionKernelOp::Diffusion))
    {
        WARN("GPU does not support diffusion dispatch — skipping");
        shutdown_backend();
        return;
    }

    std::vector<float> dz_device(metrics.dz.begin(), metrics.dz.end());
    Field3D            gpu_u(nx, ny, nz, -99.0f);
    Field3D            gpu_v(nx, ny, nz, -99.0f);
    Field3D            gpu_w(nx, ny, nz, -99.0f);
    Field3D            gpu_theta(nx, ny, nz, -99.0f);
    Field3D            gpu_qv(nx, ny, nz, -99.0f);
    const float*       fields[5]  = {u.data(), v.data(), w.data(), theta.data(), qv.data()};
    float*             outputs[5] = {gpu_u.data(), gpu_v.data(), gpu_w.data(), gpu_theta.data(), gpu_qv.data()};
    const DiffusionFieldLocation locations[5] = {DiffusionFieldLocation::UFace,
                                                 DiffusionFieldLocation::VFace,
                                                 DiffusionFieldLocation::WFace,
                                                 DiffusionFieldLocation::Scalar,
                                                 DiffusionFieldLocation::Scalar};
    DiffusionBatchDispatch request{fields,
                                         outputs,
                                         locations,
                                         5,
                                         rho.data(),
                                         dz_device.data(),
                                         nx,
                                         ny,
                                         nz,
                                         static_cast<float>(metrics.dx),
                                         static_cast<float>(metrics.dy),
                                         static_cast<float>(config.K_h),
                                         static_cast<float>(config.K_v),
                                         true,
                                         true,
                                         false};
    REQUIRE(kernels->dispatch_diffusion_batch(request));
    REQUIRE(gpu_u(0, 0, 0) == 0.0f);
    REQUIRE(gpu_v(0, 0, 0) == 0.0f);
    REQUIRE(gpu_w(0, 0, 0) == 0.0f);

    require_parity("diffusion u",
                   cpu.dudt_diff.data(),
                   gpu_u.data(),
                   nx,
                   ny,
                   nz,
                   2.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("diffusion v",
                   cpu.dvdt_diff.data(),
                   gpu_v.data(),
                   nx,
                   ny,
                   nz,
                   2.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("diffusion w",
                   cpu.dwdt_diff.data(),
                   gpu_w.data(),
                   nx,
                   ny,
                   nz,
                   2.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("diffusion scalar",
                   cpu.dthetadt_diff.data(),
                   gpu_theta.data(),
                   nx,
                   ny,
                   nz,
                   2.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("diffusion qv",
                   cpu.dqvdt_diff.data(),
                   gpu_qv.data(),
                   nx,
                   ny,
                   nz,
                   2.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);

    Field3D scalar_k(nx, ny, nz, 0.0f);
    Field3D momentum_k(nx, ny, nz, 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                scalar_k(i, j, k) = 8.0f + 0.7f * i + 0.3f * j + 1.1f * k;
                momentum_k(i, j, k) = 12.0f + 0.4f * i + 0.9f * j + 0.6f * k;
            }
    DiffusionConfig variable_config = config;
    variable_config.K_h = 17.0;
    variable_config.K_v = 0.0;
    variable_config.use_variable_K = true;
    DiffusionStateView variable_state = state;
    variable_state.K_scalar = &scalar_k;
    variable_state.K_momentum = &momentum_k;
    DiffusionTendencies variable_cpu;
    shutdown_backend();
    cpu_scheme.compute_diffusion_tendencies(
        variable_config, variable_state, variable_cpu, nullptr);
    REQUIRE(try_init_gpu());
    backend = mutable_compute_backend();
    kernels = backend != nullptr ? backend->diffusion_kernels() : nullptr;
    REQUIRE(kernels != nullptr);
    const float* vertical_k[5] = {momentum_k.data(), momentum_k.data(),
                                  momentum_k.data(), scalar_k.data(),
                                  scalar_k.data()};
    request.K_h = static_cast<float>(variable_config.K_h);
    request.K_v = 0.0f;
    request.vertical_K = vertical_k;
    REQUIRE(kernels->dispatch_diffusion_batch(request));
    require_parity("variable-K diffusion u", variable_cpu.dudt_diff.data(),
                   gpu_u.data(), nx, ny, nz, 2.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("variable-K diffusion v", variable_cpu.dvdt_diff.data(),
                   gpu_v.data(), nx, ny, nz, 2.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("variable-K diffusion w", variable_cpu.dwdt_diff.data(),
                   gpu_w.data(), nx, ny, nz, 2.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("variable-K diffusion theta",
                   variable_cpu.dthetadt_diff.data(), gpu_theta.data(),
                   nx, ny, nz, 2.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("variable-K diffusion qv", variable_cpu.dqvdt_diff.data(),
                   gpu_qv.data(), nx, ny, nz, 2.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);

    const Field3D outputs_before_failure[5] = {
        gpu_u, gpu_v, gpu_w, gpu_theta, gpu_qv};
    Field3D invalid_k = scalar_k;
    invalid_k(2, 3, 4) = -1.0f;
    const float* invalid_vertical_k[5] = {
        momentum_k.data(), momentum_k.data(), momentum_k.data(),
        invalid_k.data(), scalar_k.data()};
    request.vertical_K = invalid_vertical_k;
    REQUIRE_FALSE(kernels->dispatch_diffusion_batch(request));
    const Field3D* outputs_after_failure[5] = {
        &gpu_u, &gpu_v, &gpu_w, &gpu_theta, &gpu_qv};
    for (int target = 0; target < 5; ++target)
        REQUIRE(std::memcmp(outputs_before_failure[target].data(),
                            outputs_after_failure[target]->data(),
                            outputs_before_failure[target].size() * sizeof(float)) == 0);
    shutdown_backend();
}

// ---- Lin microphysics GPU parity (GPU.md Phase 3.3) ----

TEST_CASE("GPU Lin pointwise + sedimentation parity", "[vulkan][gpu_parity]")
{
    NR  = 16;
    NTH = 16;
    NZ  = 24;
    dz  = 400.0;

    // Mixed-phase state straddling T0 with every species populated, so all
    // process branches (saturation adjustment, warm rain, freezing,
    // deposition, riming, aggregation, sublimation, melting, fallout) fire
    // somewhere in the column.
    Field3D theta_f(NR, NTH, NZ), p_f(NR, NTH, NZ), rho_f(NR, NTH, NZ);
    Field3D qv_f(NR, NTH, NZ), qc_f(NR, NTH, NZ), qr_f(NR, NTH, NZ);
    Field3D qi_f(NR, NTH, NZ), qs_f(NR, NTH, NZ), qg_f(NR, NTH, NZ);
    Field3D qh_f(NR, NTH, NZ);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double zf    = (static_cast<double>(k) + 0.5) * dz;
                const double horiz = 0.1 * std::sin(0.7 * i) * std::cos(0.5 * j);
                theta_f(i, j, k)   = static_cast<float>(300.0 + 0.004 * zf + horiz);
                p_f(i, j, k)       = static_cast<float>(100000.0 * std::exp(-zf / 8500.0));
                rho_f(i, j, k)     = static_cast<float>(1.2 * std::exp(-zf / 8500.0) * (1.0 + 0.02 * horiz));
                qv_f(i, j, k)      = static_cast<float>(0.012 * std::exp(-zf / 3000.0));
                qc_f(i, j, k)      = static_cast<float>(0.0020 + 0.0005 * horiz);
                qr_f(i, j, k)      = static_cast<float>(0.0010 * std::exp(-zf / 5000.0));
                qi_f(i, j, k)      = static_cast<float>((zf > 4000.0) ? 0.0015 : 0.0);
                qs_f(i, j, k)      = static_cast<float>((zf > 4000.0) ? 0.0008 : 0.0);
                qg_f(i, j, k)      = static_cast<float>((zf > 3000.0) ? 0.0006 : 0.0);
                qh_f(i, j, k)      = static_cast<float>((zf > 3000.0) ? 0.0003 : 0.0);
            }
    const double dt_step = 2.0;

    LinScheme lin;
    lin.set_vertical_spacing_m(dz);
    Field3D c_dtheta, c_dqv, c_dqc, c_dqr, c_dqi, c_dqs, c_dqg, c_dqh;
    Field3D g_dtheta, g_dqv, g_dqc, g_dqr, g_dqi, g_dqs, g_dqg, g_dqh;

    // CPU reference first, with no backend (the GPU hook inside
    // compute_tendencies falls through to the CPU path).
    shutdown_backend();
    lin.compute_tendencies(p_f,
                           theta_f,
                           rho_f,
                           qv_f,
                           qc_f,
                           qr_f,
                           qi_f,
                           qs_f,
                           qg_f,
                           qh_f,
                           dt_step,
                           c_dtheta,
                           c_dqv,
                           c_dqc,
                           c_dqr,
                           c_dqi,
                           c_dqs,
                           c_dqg,
                           c_dqh);

    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping Lin parity test");
        return;
    }
    auto* backend = mutable_compute_backend();
    auto* kernels = backend != nullptr ? backend->microphysics_kernels() : nullptr;
    if (kernels == nullptr || !kernels->supports(MicrophysicsKernelOp::LinPointwise))
    {
        WARN("GPU does not support Lin dispatch — skipping");
        shutdown_backend();
        return;
    }
    REQUIRE(kernels->supports(MicrophysicsKernelOp::LinFused));
    backend->begin_resident_step();

    lin.compute_tendencies(p_f,
                           theta_f,
                           rho_f,
                           qv_f,
                           qc_f,
                           qr_f,
                           qi_f,
                           qs_f,
                           qg_f,
                           qh_f,
                           dt_step,
                           g_dtheta,
                           g_dqv,
                           g_dqc,
                           g_dqr,
                           g_dqi,
                           g_dqs,
                           g_dqg,
                           g_dqh);

    // Sanity: the scheme produced nonzero tendencies on this state.
    double max_dqv = 0.0;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                max_dqv = std::max(max_dqv, std::fabs(static_cast<double>(g_dqv(i, j, k))));
    INFO("max |dqv_dt| = " << max_dqv);
    REQUIRE(max_dqv > 1.0e-9);

    // Microphysics tier (exp/pow, libm vs GPU SFU): mixing ratios at
    // rel 1e-4 + abs 1e-7 per second of tendency; theta heating scales
    // with Lv/cp ~ 2.5e3 K per unit mixing ratio.
    struct Pair
    {
        const char*    name;
        const Field3D* cpu;
        const Field3D* gpu;
        double         tol;
    };
    const Pair pairs[8] = {
        {"dtheta", &c_dtheta, &g_dtheta, 5.0e-4},
        {"dqv", &c_dqv, &g_dqv, 1.0e-7},
        {"dqc", &c_dqc, &g_dqc, 1.0e-7},
        {"dqr", &c_dqr, &g_dqr, 1.0e-7},
        {"dqi", &c_dqi, &g_dqi, 1.0e-7},
        {"dqs", &c_dqs, &g_dqs, 1.0e-7},
        {"dqg", &c_dqg, &g_dqg, 1.0e-7},
        {"dqh", &c_dqh, &g_dqh, 1.0e-7},
    };
    for (const auto& pr : pairs)
        require_parity(
            pr.name, pr.cpu->data(), pr.gpu->data(), NR, NTH, NZ, pr.tol, nimbus::parity::ComparisonDomain::FullField);

    // Lin publishes all eight state fields. The final in-place application
    // therefore qualifies the full resident pointwise -> sedimentation ->
    // apply chain in addition to checking its numerical update.
    require_microphysics_apply_parity(kernels,
                                      theta_f,
                                      qv_f,
                                      qc_f,
                                      qr_f,
                                      qi_f,
                                      qs_f,
                                      qg_f,
                                      qh_f,
                                      g_dtheta,
                                      g_dqv,
                                      g_dqc,
                                      g_dqr,
                                      g_dqi,
                                      g_dqs,
                                      g_dqg,
                                      g_dqh,
                                      NR,
                                      NTH,
                                      NZ,
                                      static_cast<float>(dt_step));

    shutdown_backend();
}

TEST_CASE("GPU directional Smagorinsky matches the closure tensor contract",
          "[vulkan][gpu_parity][turbulence][directional]")
{
    constexpr int     nx    = 9;
    constexpr int     ny    = 9;
    constexpr int     nz    = 7;
    constexpr float   dx    = 100.0f;
    constexpr float   dy    = 120.0f;
    constexpr float   dz    = 80.0f;
    constexpr float   shear = 0.0125f;
    constexpr float   Cs    = 0.18f;
    const std::size_t cells = static_cast<std::size_t>(nx) * ny * nz;
    const auto at = [=](int i, int j, int k) -> std::size_t { return (static_cast<std::size_t>(i) * ny + j) * nz + k; };

    if (!try_init_gpu())
    {
        WARN("GPU unavailable - skipping directional Smagorinsky contract test");
        return;
    }
    auto* backend = mutable_compute_backend();
    auto* kernels = backend != nullptr ? backend->turbulence_kernels() : nullptr;
    if (kernels == nullptr || !kernels->supports(TurbulenceKernelOp::Smagorinsky))
    {
        WARN("GPU does not support Smagorinsky - skipping directional contract test");
        shutdown_backend();
        return;
    }

    std::vector<float> u(cells, 0.0f), v(cells, 0.0f), w(cells, 0.0f);
    std::vector<float> theta(cells, 300.0f), qv(cells, 0.01f);
    std::vector<float> rho(cells, 1.0f);
    std::vector<float> dz_prof(nz, dz), dzc_prof(nz, 2.0f * dz);
    std::vector<float> nu(cells), kth(cells), kq(cells);
    std::vector<float> dudt(cells), dvdt(cells), dwdt(cells);
    std::vector<float> dthetadt(cells), dqvdt(cells);

    // Pure vertical u shear belongs to vertical_only and full 3-D, never to
    // horizontal_only. This catches a coefficient pass that silently measures
    // the full tensor regardless of the selected closure mode.
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                u[at(i, j, k)] = shear * (static_cast<float>(k) + 0.5f) * dz;

    SmagorinskyCoefficientsDispatch coefficients{};
    coefficients.u           = u.data();
    coefficients.v           = v.data();
    coefficients.w           = w.data();
    coefficients.dz_prof     = dz_prof.data();
    coefficients.dzc_prof    = dzc_prof.data();
    coefficients.prof_len    = nz;
    coefficients.nu_t        = nu.data();
    coefficients.K_theta     = kth.data();
    coefficients.K_q         = kq.data();
    coefficients.nx          = nx;
    coefficients.ny          = ny;
    coefficients.nz          = nz;
    coefficients.filter_mode = 0;
    coefficients.periodic_x  = false;
    coefficients.periodic_y  = false;
    coefficients.dx          = dx;
    coefficients.dy          = dy;
    coefficients.Cs          = Cs;
    coefficients.Pr_t        = 1.0f / 3.0f;
    coefficients.Sc_t        = 1.0f / 3.0f;
    coefficients.nu_t_max    = 1000.0f;
    coefficients.K_max       = 1000.0f;

    coefficients.use_horizontal = true;
    coefficients.use_vertical   = false;
    REQUIRE(kernels->dispatch_smagorinsky_coefficients(coefficients));
    REQUIRE(nu[at(4, 4, 3)] == Approx(0.0f).margin(1.0e-10));

    const float delta           = std::cbrt(dx * dy * dz);
    const float expected_nu     = Cs * Cs * delta * delta * shear;
    coefficients.use_horizontal = false;
    coefficients.use_vertical   = true;
    REQUIRE(kernels->dispatch_smagorinsky_coefficients(coefficients));
    REQUIRE(nu[at(4, 4, 3)] == Approx(expected_nu).epsilon(2.0e-5));
    coefficients.use_horizontal = true;
    coefficients.use_vertical   = true;
    REQUIRE(kernels->dispatch_smagorinsky_coefficients(coefficients));
    REQUIRE(nu[at(4, 4, 3)] == Approx(expected_nu).epsilon(2.0e-5));

    // Use a prescribed constant viscosity to isolate the stress pass. In a
    // horizontal closure, x-varying du/dz must not enter the w equation.
    constexpr float cross_gradient = 2.0e-7f;
    std::fill(u.begin(), u.end(), 0.0f);
    std::fill(w.begin(), w.end(), 0.0f);
    std::fill(nu.begin(), nu.end(), 5.0f);
    std::fill(kth.begin(), kth.end(), 0.0f);
    std::fill(kq.begin(), kq.end(), 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                u[at(i, j, k)] = cross_gradient * (i * dx) * ((static_cast<float>(k) + 0.5f) * dz);

    SmagorinskyTendenciesDispatch tendencies{};
    tendencies.u              = u.data();
    tendencies.v              = v.data();
    tendencies.w              = w.data();
    tendencies.theta_data     = theta.data();
    tendencies.qv_data        = qv.data();
    tendencies.nu_t           = nu.data();
    tendencies.K_theta        = kth.data();
    tendencies.K_q            = kq.data();
    tendencies.dz_prof        = dz_prof.data();
    tendencies.rho            = rho.data();
    tendencies.prof_len       = nz;
    tendencies.dudt_sgs       = dudt.data();
    tendencies.dvdt_sgs       = dvdt.data();
    tendencies.dwdt_sgs       = dwdt.data();
    tendencies.dthetadt_sgs   = dthetadt.data();
    tendencies.dqvdt_sgs      = dqvdt.data();
    tendencies.nx             = nx;
    tendencies.ny             = ny;
    tendencies.nz             = nz;
    tendencies.has_qv         = true;
    tendencies.periodic_x     = false;
    tendencies.periodic_y     = false;
    tendencies.dx             = dx;
    tendencies.dy             = dy;
    tendencies.use_horizontal = true;
    tendencies.use_vertical   = false;
    REQUIRE(kernels->dispatch_smagorinsky_tendencies(tendencies));
    REQUIRE(dwdt[at(4, 4, 3)] == Approx(0.0f).margin(1.0e-10));

    // Reciprocal check: z-varying dw/dx cannot enter the u equation in a
    // vertical-only closure.
    std::fill(u.begin(), u.end(), 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                w[at(i, j, k)] = cross_gradient * (i * dx) * ((static_cast<float>(k) + 1.0f) * dz);
    tendencies.use_horizontal = false;
    tendencies.use_vertical   = true;
    REQUIRE(kernels->dispatch_smagorinsky_tendencies(tendencies));
    REQUIRE(dudt[at(4, 4, 3)] == Approx(0.0f).margin(1.0e-10));

    shutdown_backend();
}

TEST_CASE("GPU Smagorinsky to SGS apply preserves resident tendencies", "[vulkan][gpu_parity][turbulence]")
{
    constexpr int     nx      = 17;
    constexpr int     ny      = 13;
    constexpr int     nz      = 9;
    constexpr float   dt_step = 0.25f;
    const std::size_t cells   = static_cast<std::size_t>(nx) * ny * nz;

    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping resident SGS chain test");
        return;
    }
    auto* backend = mutable_compute_backend();
    auto* kernels = backend != nullptr ? backend->turbulence_kernels() : nullptr;
    if (kernels == nullptr || !kernels->supports(TurbulenceKernelOp::Smagorinsky) ||
        !kernels->supports(TurbulenceKernelOp::SgsApply))
    {
        WARN("GPU does not support fused Smagorinsky + SGS apply — skipping");
        shutdown_backend();
        return;
    }
    backend->begin_resident_step();

    std::vector<float> u_f(cells), v_f(cells), w_f(cells), rho_f(cells);
    std::vector<float> theta_f(cells), qv_f(cells), tke_f(cells, 0.2f);
    // Zero is the exact no-subgrid-energy equilibrium; the apply kernel must
    // not manufacture its former 0.001 floor in an otherwise inactive cell.
    tke_f[0] = 0.0f;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t n = (static_cast<std::size_t>(i) * ny + j) * nz + k;
                // Keep activity on both periodic-x edge rows so the device
                // path must use the physical 1 <-> nx-2 seam, not ghosts.
                const bool  interior = i >= 1 && i <= nx - 2 && j >= 4 && j <= ny - 5 && k >= 2 && k <= nz - 3;
                const float wave     = interior ? std::sin(0.3f * i + 0.2f * j + 0.4f * k) : 0.0f;
                u_f[n]               = 4.0f + 0.8f * wave;
                v_f[n]               = -2.0f - 0.6f * wave;
                w_f[n]               = 0.2f * wave;
                theta_f[n]           = 300.0f + 1.5f * wave;
                qv_f[n]              = 0.010f + 2.0e-4f * wave;
                // Poison the old rho0-only implementation: live density has
                // independent horizontal and vertical structure.
                rho_f[n] = std::exp(-static_cast<float>(k) * 100.0f / 8000.0f) * (0.78f + 0.018f * i + 0.011f * j);
            }
    const std::vector<float> initial_u     = u_f;
    const std::vector<float> initial_v     = v_f;
    const std::vector<float> initial_w     = w_f;
    const std::vector<float> initial_theta = theta_f;
    const std::vector<float> initial_qv    = qv_f;
    const std::vector<float> initial_tke   = tke_f;

    std::vector<float> dz_prof(nz, 100.0f), dzc_prof(nz, 200.0f);
    constexpr float    coefficient_sentinel = -1234.0f;
    std::vector<float> nu_t(cells, coefficient_sentinel);
    std::vector<float> K_theta(cells, coefficient_sentinel);
    std::vector<float> K_q(cells, coefficient_sentinel);
    std::vector<float> dudt(cells), dvdt(cells), dwdt(cells);
    std::vector<float> dthetadt(cells), dqvdt(cells);

    const SmagorinskyFusedDispatch smag_dispatch{u_f.data(),
                                                 v_f.data(),
                                                 w_f.data(),
                                                 theta_f.data(),
                                                 qv_f.data(),
                                                 dz_prof.data(),
                                                 dzc_prof.data(),
                                                 rho_f.data(),
                                                 nz,
                                                 nu_t.data(),
                                                 K_theta.data(),
                                                 K_q.data(),
                                                 dudt.data(),
                                                 dvdt.data(),
                                                 dwdt.data(),
                                                 dthetadt.data(),
                                                 dqvdt.data(),
                                                 nx,
                                                 ny,
                                                 nz,
                                                 0,
                                                 true,
                                                 true,
                                                 true,
                                                 true,
                                                 false,
                                                 100.0f,
                                                 120.0f,
                                                 0.18f,
                                                 1.0f / 3.0f,
                                                 1.0f / 3.0f,
                                                 1000.0f,
                                                 1000.0f,
                                                 false};
    REQUIRE(kernels->dispatch_smagorinsky_fused(smag_dispatch));
    const bool coefficients_untouched =
        std::all_of(nu_t.begin(), nu_t.end(), [](float value) { return value == coefficient_sentinel; }) &&
        std::all_of(K_theta.begin(), K_theta.end(), [](float value) { return value == coefficient_sentinel; }) &&
        std::all_of(K_q.begin(), K_q.end(), [](float value) { return value == coefficient_sentinel; });
    REQUIRE(coefficients_untouched);

    double max_abs_tendency  = 0.0;
    bool   tendencies_finite = true;
    for (std::size_t n = 0; n < cells; ++n)
    {
        tendencies_finite = tendencies_finite && std::isfinite(dudt[n]) && std::isfinite(dvdt[n]) &&
                            std::isfinite(dwdt[n]) && std::isfinite(dthetadt[n]) && std::isfinite(dqvdt[n]);
        max_abs_tendency  = std::max(
            max_abs_tendency, std::fabs(static_cast<double>(dudt[n])) + std::fabs(static_cast<double>(dthetadt[n])));
    }
    REQUIRE(tendencies_finite);
    REQUIRE(max_abs_tendency > 1.0e-9);

    const auto at = [=](int i, int j, int k) -> std::size_t { return (static_cast<std::size_t>(i) * ny + j) * nz + k; };
    auto       require_conserved = [](double residual, double l1)
    { REQUIRE(std::fabs(residual) <= 1.0e-4 * std::max(l1, 1.0e-20)); };
    double sum_u = 0.0, l1_u = 0.0;
    double sum_v = 0.0, l1_v = 0.0;
    double sum_w = 0.0, l1_w = 0.0;
    double sum_theta = 0.0, l1_theta = 0.0;
    double sum_qv = 0.0, l1_qv = 0.0;
    for (int i = 1; i <= nx - 2; ++i)
        for (int j = 1; j <= ny - 2; ++j)
            for (int k = 0; k < nz - 1; ++k)
            {
                const int    i1    = (i == nx - 2) ? 1 : i + 1;
                const double value = 0.5 * (rho_f[at(i, j, k)] + rho_f[at(i1, j, k)]) * dudt[at(i, j, k)];
                sum_u += value;
                l1_u += std::fabs(value);
            }
    for (int i = 1; i <= nx - 2; ++i)
        for (int j = 1; j <= ny - 3; ++j)
            for (int k = 0; k < nz - 1; ++k)
            {
                const double value = 0.5 * (rho_f[at(i, j, k)] + rho_f[at(i, j + 1, k)]) * dvdt[at(i, j, k)];
                sum_v += value;
                l1_v += std::fabs(value);
            }
    for (int i = 1; i <= nx - 2; ++i)
        for (int j = 1; j <= ny - 2; ++j)
            for (int k = 0; k < nz - 1; ++k)
            {
                const double value = 0.5 * (rho_f[at(i, j, k)] + rho_f[at(i, j, k + 1)]) * dwdt[at(i, j, k)];
                sum_w += value;
                l1_w += std::fabs(value);
            }
    for (int i = 1; i <= nx - 2; ++i)
        for (int j = 1; j <= ny - 2; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double theta_value = rho_f[at(i, j, k)] * dthetadt[at(i, j, k)];
                const double qv_value    = rho_f[at(i, j, k)] * dqvdt[at(i, j, k)];
                sum_theta += theta_value;
                l1_theta += std::fabs(theta_value);
                sum_qv += qv_value;
                l1_qv += std::fabs(qv_value);
            }
    require_conserved(sum_u, l1_u);
    require_conserved(sum_v, l1_v);
    require_conserved(sum_w, l1_w);
    require_conserved(sum_theta, l1_theta);
    require_conserved(sum_qv, l1_qv);

    std::vector<float> dtkedt(cells, 0.0f), dtkedt_pbl(cells, 0.0f);
    std::vector<float> expected_u(cells), expected_v(cells), expected_w(cells);
    std::vector<float> expected_theta(cells), expected_qv(cells);
    std::vector<float> expected_tke(cells);
    for (std::size_t n = 0; n < cells; ++n)
    {
        expected_u[n]     = std::clamp(initial_u[n] + dudt[n] * dt_step, -250.0f, 250.0f);
        expected_v[n]     = std::clamp(initial_v[n] + dvdt[n] * dt_step, -250.0f, 250.0f);
        expected_w[n]     = std::clamp(initial_w[n] + dwdt[n] * dt_step, -150.0f, 150.0f);
        expected_theta[n] = std::clamp(initial_theta[n] + dthetadt[n] * dt_step, 150.0f, 700.0f);
        expected_qv[n]    = std::clamp(initial_qv[n] + dqvdt[n] * dt_step, 0.0f, 0.1f);
        expected_tke[n]   = std::max(0.0f, initial_tke[n]);
    }

    struct SgsCommitCapture
    {
        std::size_t                       cells  = 0;
        bool                              called = false;
        std::array<std::vector<float>, 6> fields;
    } capture{cells};
    const auto capture_commit = +[](void*        raw_context,
                                    const float* u_committed,
                                    const float* v_committed,
                                    const float* w_committed,
                                    const float* theta_committed,
                                    const float* qv_committed,
                                    const float* tke_committed)
    {
        auto&        result   = *static_cast<SgsCommitCapture*>(raw_context);
        const float* values[] = {
            u_committed,
            v_committed,
            w_committed,
            theta_committed,
            qv_committed,
            tke_committed,
        };
        for (std::size_t field = 0; field < result.fields.size(); ++field)
        {
            result.fields[field].assign(values[field], values[field] + result.cells);
        }
        result.called = true;
    };

    const SgsApplyDispatch apply_dispatch{u_f.data(),
                                          v_f.data(),
                                          w_f.data(),
                                          theta_f.data(),
                                          qv_f.data(),
                                          tke_f.data(),
                                          dudt.data(),
                                          dvdt.data(),
                                          dwdt.data(),
                                          dthetadt.data(),
                                          dqvdt.data(),
                                          dtkedt.data(),
                                          dtkedt_pbl.data(),
                                          nx,
                                          ny,
                                          nz,
                                          true,
                                          true,
                                          dt_step,
                                          300.0f,
                                          150.0f,
                                          700.0f,
                                          0.1f,
                                          250.0f,
                                          150.0f,
                                          true,
                                          true,
                                          capture_commit,
                                          &capture};
    REQUIRE(kernels->dispatch_sgs_apply(apply_dispatch));
    REQUIRE(capture.called);
    REQUIRE(capture.fields[0] == u_f);
    REQUIRE(capture.fields[1] == v_f);
    REQUIRE(capture.fields[2] == w_f);
    REQUIRE(capture.fields[3] == theta_f);
    REQUIRE(capture.fields[4] == qv_f);
    REQUIRE(capture.fields[5] == tke_f);

    require_parity("resident SGS u",
                   expected_u.data(),
                   u_f.data(),
                   nx,
                   ny,
                   nz,
                   1.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("resident SGS v",
                   expected_v.data(),
                   v_f.data(),
                   nx,
                   ny,
                   nz,
                   1.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("resident SGS w",
                   expected_w.data(),
                   w_f.data(),
                   nx,
                   ny,
                   nz,
                   1.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("resident SGS theta",
                   expected_theta.data(),
                   theta_f.data(),
                   nx,
                   ny,
                   nz,
                   1.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("resident SGS qv",
                   expected_qv.data(),
                   qv_f.data(),
                   nx,
                   ny,
                   nz,
                   1.0e-7,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("resident SGS tke",
                   expected_tke.data(),
                   tke_f.data(),
                   nx,
                   ny,
                   nz,
                   1.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);

    // A raw Warn/Strict writer must not clamp or repair its candidates. The
    // post-fence observer is also an atomic veto: throwing there leaves every
    // authoritative host field at the old time level.
    u_f                         = initial_u;
    v_f                         = initial_v;
    w_f                         = initial_w;
    theta_f                     = initial_theta;
    qv_f                        = initial_qv;
    tke_f                       = initial_tke;
    std::vector<float> raw_dudt = dudt;
    raw_dudt[0]                 = 2000.0f;
    const auto reject_commit =
        +[](void*, const float*, const float*, const float*, const float*, const float*, const float*)
    { throw std::runtime_error("reject raw SGS candidate"); };
    const SgsApplyDispatch rejected_dispatch{u_f.data(),
                                             v_f.data(),
                                             w_f.data(),
                                             theta_f.data(),
                                             qv_f.data(),
                                             tke_f.data(),
                                             raw_dudt.data(),
                                             dvdt.data(),
                                             dwdt.data(),
                                             dthetadt.data(),
                                             dqvdt.data(),
                                             dtkedt.data(),
                                             dtkedt_pbl.data(),
                                             nx,
                                             ny,
                                             nz,
                                             true,
                                             true,
                                             dt_step,
                                             300.0f,
                                             150.0f,
                                             700.0f,
                                             0.1f,
                                             250.0f,
                                             150.0f,
                                             false,
                                             false,
                                             reject_commit,
                                             nullptr};
    REQUIRE_THROWS_WITH(kernels->dispatch_sgs_apply(rejected_dispatch), "reject raw SGS candidate");
    REQUIRE(u_f == initial_u);
    REQUIRE(v_f == initial_v);
    REQUIRE(w_f == initial_w);
    REQUIRE(theta_f == initial_theta);
    REQUIRE(qv_f == initial_qv);
    REQUIRE(tke_f == initial_tke);

    shutdown_backend();
}

// ---- Collocated Cartesian tendencies and acoustic pressure/momentum ----

TEST_CASE("Cartesian A-grid tendencies match the complete CPU equation and ABI",
          "[vulkan][gpu_parity][cartesian][agrid]")
{
    constexpr int   nx          = 11;
    constexpr int   ny          = 9;
    constexpr int   nz          = 7;
    constexpr float dx_local    = 120.0f;
    constexpr float dy_local    = 95.0f;
    constexpr float dz_local    = 80.0f;
    constexpr float gravity     = 9.81f;
    constexpr float gamma_local = 1.4f;
    constexpr float f_plane     = 1.0e-4f;

    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping Cartesian A-grid tendency parity");
        return;
    }

    auto* backend = mutable_compute_backend();
    backend->set_coordinate_system(CoordinateSystem::Cartesian);
    auto* kernels = backend->dynamics_kernels();
    if (kernels == nullptr || !kernels->supports(DynamicsKernelOp::CartesianTendencies))
    {
        WARN("Cartesian A-grid tendency pipeline unavailable — skipping");
        shutdown_backend();
        return;
    }

    Field3D            u(nx, ny, nz), v(nx, ny, nz), w(nx, ny, nz);
    Field3D            rho_f(nx, ny, nz), pressure(nx, ny, nz), theta_f(nx, ny, nz);
    Field3D            loading(nx, ny, nz), qv_f(nx, ny, nz);
    std::vector<float> p0(static_cast<std::size_t>(nz));
    std::vector<float> rho0(static_cast<std::size_t>(nz));
    std::vector<float> u0(static_cast<std::size_t>(nz));
    std::vector<float> v0(static_cast<std::size_t>(nz));
    std::vector<float> qv0(static_cast<std::size_t>(nz));
    for (int k = 0; k < nz; ++k)
    {
        p0[static_cast<std::size_t>(k)]   = 100000.0f - 930.0f * k;
        rho0[static_cast<std::size_t>(k)] = 1.18f - 0.015f * k;
        u0[static_cast<std::size_t>(k)]   = 4.0f + 0.2f * k;
        v0[static_cast<std::size_t>(k)]   = -1.0f + 0.1f * k;
        qv0[static_cast<std::size_t>(k)]  = 0.012f - 0.0007f * k;
    }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                u(i, j, k)        = 4.0f + 0.07f * i - 0.03f * j + 0.02f * k;
                v(i, j, k)        = -1.0f + 0.04f * i + 0.05f * j - 0.01f * k;
                w(i, j, k)        = 0.02f * i - 0.015f * j + 0.03f * k;
                rho_f(i, j, k)    = rho0[static_cast<std::size_t>(k)] + 0.002f * i - 0.001f * j;
                pressure(i, j, k) = p0[static_cast<std::size_t>(k)] + 4.0f * i - 3.0f * j + 1.5f * k;
                theta_f(i, j, k)  = 300.0f + 0.1f * k;
                loading(i, j, k)  = 1.0e-4f * static_cast<float>(1 + (i + j + k) % 4);
                qv_f(i, j, k)     = qv0[static_cast<std::size_t>(k)] + 2.0e-4f * i - 1.0e-4f * j;
            }

    constexpr int west_bc  = 2; // rigid
    constexpr int east_bc  = 4; // zero-gradient/open face value
    constexpr int south_bc = 0; // periodic
    constexpr int north_bc = 0; // periodic
    constexpr int top_bc   = 2; // open

    Field3D                     gpu_du(nx, ny, nz, 123.0f), gpu_dv(nx, ny, nz, 123.0f);
    Field3D                     gpu_dw(nx, ny, nz, 123.0f), gpu_drho(nx, ny, nz, 123.0f);
    Field3D                     gpu_dp(nx, ny, nz, 123.0f);
    CartesianTendenciesDispatch dispatch{};
    dispatch.u_x_data         = u.data();
    dispatch.u_y_data         = v.data();
    dispatch.w_data           = w.data();
    dispatch.rho_data         = rho_f.data();
    dispatch.p_data           = pressure.data();
    dispatch.theta_data       = theta_f.data();
    dispatch.p0_base_data     = p0.data();
    dispatch.rho0_base_data   = rho0.data();
    dispatch.loading_data     = loading.data();
    dispatch.u0_base_data     = u0.data();
    dispatch.v0_base_data     = v0.data();
    dispatch.qv_data          = qv_f.data();
    dispatch.qv0_base_data    = qv0.data();
    dispatch.du_x_dt_data     = gpu_du.data();
    dispatch.du_y_dt_data     = gpu_dv.data();
    dispatch.dw_dt_data       = gpu_dw.data();
    dispatch.drho_dt_data     = gpu_drho.data();
    dispatch.dp_dt_data       = gpu_dp.data();
    dispatch.nr               = nx;
    dispatch.nth              = ny;
    dispatch.nz               = nz;
    dispatch.dx               = dx_local;
    dispatch.dy               = dy_local;
    dispatch.dz               = dz_local;
    dispatch.g                = gravity;
    dispatch.gamma_val        = gamma_local;
    dispatch.coriolis_f_val   = f_plane;
    dispatch.lateral_bc_west  = west_bc;
    dispatch.lateral_bc_east  = east_bc;
    dispatch.lateral_bc_south = south_bc;
    dispatch.lateral_bc_north = north_bc;
    dispatch.top_bc           = top_bc;
    dispatch.has_moisture     = true;
    REQUIRE(kernels->dispatch_cartesian_tendencies(dispatch));

    Field3D    cpu_du(nx, ny, nz, 0.0f), cpu_dv(nx, ny, nz, 0.0f);
    Field3D    cpu_dw(nx, ny, nz, 0.0f), cpu_drho(nx, ny, nz, 0.0f);
    Field3D    cpu_dp(nx, ny, nz, 0.0f);
    const auto average_product = [](float rho_lo, float rho_hi, float velocity_lo, float velocity_hi)
    { return 0.25 * (static_cast<double>(rho_lo) + rho_hi) * (static_cast<double>(velocity_lo) + velocity_hi); };
    const auto closed = [](int bc) { return bc == 2 || bc == 3; };
    const auto x_face = [&](int face_i, int j, int k)
    {
        if (face_i == 0)
        {
            if (closed(west_bc))
                return 0.0;
            if (west_bc == 0)
                return average_product(rho_f(nx - 2, j, k), rho_f(1, j, k), u(nx - 2, j, k), u(1, j, k));
            return static_cast<double>(rho_f(0, j, k)) * u(0, j, k);
        }
        if (face_i == nx - 2)
        {
            if (closed(east_bc))
                return 0.0;
            if (east_bc == 0)
                return average_product(rho_f(nx - 2, j, k), rho_f(1, j, k), u(nx - 2, j, k), u(1, j, k));
            return static_cast<double>(rho_f(nx - 1, j, k)) * u(nx - 1, j, k);
        }
        return average_product(rho_f(face_i, j, k), rho_f(face_i + 1, j, k), u(face_i, j, k), u(face_i + 1, j, k));
    };
    const auto y_face = [&](int i, int face_j, int k)
    {
        if (face_j == 0 || face_j == ny - 2)
        {
            const int bc = face_j == 0 ? south_bc : north_bc;
            if (closed(bc))
                return 0.0;
            if (bc == 0)
                return average_product(rho_f(i, ny - 2, k), rho_f(i, 1, k), v(i, ny - 2, k), v(i, 1, k));
            const int boundary_j = face_j == 0 ? 0 : ny - 1;
            return static_cast<double>(rho_f(i, boundary_j, k)) * v(i, boundary_j, k);
        }
        return average_product(rho_f(i, face_j, k), rho_f(i, face_j + 1, k), v(i, face_j, k), v(i, face_j + 1, k));
    };
    const auto z_face = [&](int i, int j, int face_k)
    {
        if (face_k == 0)
            return 0.0;
        if (face_k == nz - 2)
        {
            if (top_bc != 2)
                return 0.0;
            return static_cast<double>(rho_f(i, j, nz - 1)) * w(i, j, nz - 1);
        }
        return average_product(rho_f(i, j, face_k), rho_f(i, j, face_k + 1), w(i, j, face_k), w(i, j, face_k + 1));
    };

    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const double ux        = u(i, j, k);
                const double uy        = v(i, j, k);
                const double wz        = w(i, j, k);
                const double rho_value = rho_f(i, j, k);
                const double inv_2dx   = 1.0 / (2.0 * dx_local);
                const double inv_2dy   = 1.0 / (2.0 * dy_local);
                const double inv_2dz   = 1.0 / (2.0 * dz_local);
                const auto   ddx       = [&](const Field3D& field)
                { return (field(i + 1, j, k) - field(i - 1, j, k)) * inv_2dx; };
                const auto ddy = [&](const Field3D& field)
                { return (field(i, j + 1, k) - field(i, j - 1, k)) * inv_2dy; };
                const auto ddz = [&](const Field3D& field)
                { return (field(i, j, k + 1) - field(i, j, k - 1)) * inv_2dz; };
                const double dux_dx = ddx(u), dux_dy = ddy(u), dux_dz = ddz(u);
                const double duy_dx = ddx(v), duy_dy = ddy(v), duy_dz = ddz(v);
                const double dw_dx = ddx(w), dw_dy = ddy(w), dw_dz = ddz(w);
                const double dp_dx = ddx(pressure), dp_dy = ddy(pressure);
                const double dp_dz = ddz(pressure);
                cpu_du(i, j, k)    = static_cast<float>(-ux * dux_dx - uy * dux_dy - wz * dux_dz - dp_dx / rho_value +
                                                        f_plane * (uy - v0[static_cast<std::size_t>(k)]));
                cpu_dv(i, j, k)    = static_cast<float>(-ux * duy_dx - uy * duy_dy - wz * duy_dz - dp_dy / rho_value -
                                                        f_plane * (ux - u0[static_cast<std::size_t>(k)]));
                const double dp0_dz =
                    (p0[static_cast<std::size_t>(k + 1)] - p0[static_cast<std::size_t>(k - 1)]) * inv_2dz;
                cpu_dw(i, j, k) = static_cast<float>(
                    -ux * dw_dx - uy * dw_dy - wz * dw_dz - (dp_dz - dp0_dz) / rho_value -
                    gravity * (rho_value - rho0[static_cast<std::size_t>(k)]) / rho_value +
                    gravity * 0.608 * (qv_f(i, j, k) - qv0[static_cast<std::size_t>(k)]) - gravity * loading(i, j, k));
                cpu_drho(i, j, k)       = static_cast<float>(-((x_face(i, j, k) - x_face(i - 1, j, k)) / dx_local +
                                                               (y_face(i, j, k) - y_face(i, j - 1, k)) / dy_local +
                                                               (z_face(i, j, k) - z_face(i, j, k - 1)) / dz_local));
                const double divergence = dux_dx + duy_dy + dw_dz;
                cpu_dp(i, j, k) = static_cast<float>(-gamma_local * pressure(i, j, k) * divergence - ux * dp_dx -
                                                     uy * dp_dy - wz * dp_dz);
            }

    using nimbus::parity::ComparisonDomain;
    require_parity(
        "Cartesian A-grid du", cpu_du.data(), gpu_du.data(), nx, ny, nz, 2.0e-5, ComparisonDomain::FullField);
    require_parity(
        "Cartesian A-grid dv", cpu_dv.data(), gpu_dv.data(), nx, ny, nz, 2.0e-5, ComparisonDomain::FullField);
    require_parity(
        "Cartesian A-grid dw", cpu_dw.data(), gpu_dw.data(), nx, ny, nz, 2.0e-5, ComparisonDomain::FullField);
    require_parity(
        "Cartesian A-grid drho", cpu_drho.data(), gpu_drho.data(), nx, ny, nz, 2.0e-6, ComparisonDomain::FullField);
    require_parity(
        "Cartesian A-grid dp", cpu_dp.data(), gpu_dp.data(), nx, ny, nz, 5.0e-2, ComparisonDomain::FullField);
    shutdown_backend();
}

TEST_CASE("Collocated acoustic pressure matches the Cartesian CPU stencil", "[vulkan][gpu_parity][acoustic][cartesian]")
{
    constexpr int   nx          = 12;
    constexpr int   ny          = 10;
    constexpr int   nz          = 8;
    constexpr float dx          = 100.0f;
    constexpr float dy          = 120.0f;
    constexpr float dz_local    = 80.0f;
    constexpr float gamma_local = 1.4f;
    constexpr float dt_small    = 0.1f;

    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping collocated pressure parity test");
        return;
    }

    auto* backend = mutable_compute_backend();
    backend->set_coordinate_system(CoordinateSystem::Cartesian);
    auto* kernels = backend->dynamics_kernels();
    if (kernels == nullptr || !kernels->supports(DynamicsKernelOp::AcousticPressure))
    {
        WARN("Cartesian collocated pressure pipeline unavailable — skipping");
        shutdown_backend();
        return;
    }

    Field3D u_f(nx, ny, nz), v_f(nx, ny, nz), w_f(nx, ny, nz);
    Field3D rho_in(nx, ny, nz, 1.2f), p_in(nx, ny, nz, 100000.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                u_f(i, j, k) = 0.1f * static_cast<float>(i);
                v_f(i, j, k) = -0.03f * static_cast<float>(j);
                w_f(i, j, k) = 0.02f * static_cast<float>(k);
            }

    Field3D gpu_rho(nx, ny, nz, 0.0f), gpu_p(nx, ny, nz, 0.0f);
    REQUIRE(kernels->dispatch_acoustic_pressure(AcousticPressureDispatch{u_f.data(),
                                                                         v_f.data(),
                                                                         w_f.data(),
                                                                         rho_in.data(),
                                                                         p_in.data(),
                                                                         gpu_rho.data(),
                                                                         gpu_p.data(),
                                                                         nx,
                                                                         ny,
                                                                         nz,
                                                                         dx,
                                                                         dy,
                                                                         dz_local,
                                                                         gamma_local,
                                                                         dt_small,
                                                                         1.0e-6f,
                                                                         1.0f}));

    Field3D     cpu_rho(nx, ny, nz, 0.0f), cpu_p(nx, ny, nz, 0.0f);
    const float divergence = 0.1f / dx - 0.03f / dy + 0.02f / dz_local;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                cpu_rho(i, j, k) = rho_in(i, j, k) * (1.0f - divergence * dt_small);
                cpu_p(i, j, k)   = p_in(i, j, k) * (1.0f - gamma_local * divergence * dt_small);
            }

    using nimbus::parity::ComparisonDomain;
    require_parity("collocated acoustic rho",
                   cpu_rho.data(),
                   gpu_rho.data(),
                   nx,
                   ny,
                   nz,
                   2.0e-6,
                   ComparisonDomain::StrictInterior);
    require_parity(
        "collocated acoustic p", cpu_p.data(), gpu_p.data(), nx, ny, nz, 2.0e-2, ComparisonDomain::StrictInterior);
    shutdown_backend();
}

TEST_CASE("Collocated acoustic momentum routes to the Cartesian shader under Cartesian", "[vulkan][gpu_parity]")
{
    NR  = 16;
    NTH = 16;
    NZ  = 16;

    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping collocated momentum routing test");
        return;
    }

    auto* backend = mutable_compute_backend();
    backend->set_coordinate_system(CoordinateSystem::Cartesian);
    auto* kernels = backend != nullptr ? backend->dynamics_kernels() : nullptr;
    if (kernels == nullptr || !kernels->supports(DynamicsKernelOp::AcousticMomentum))
    {
        WARN("Cartesian collocated momentum pipeline unavailable — skipping");
        shutdown_backend();
        return;
    }

    // Uniform rho, p linear in j (a pure y pressure gradient), zero winds.
    // The Cartesian shader's dv/dt = -(1/rho) dp/dy is translation-invariant
    // in i. The cylindrical shader (the pre-fix mis-route) computes
    // -(1/rho) (1/(i*dr)) dp/dtheta, which scales as 1/i across rows —
    // so equality across i rows discriminates the two shaders.
    Field3D rho_f(NR, NTH, NZ, 1.2f);
    Field3D p_f(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                p_f(i, j, k) = 100000.0f + 10.0f * static_cast<float>(j);
    Field3D zero(NR, NTH, NZ, 0.0f);
    Field3D u_out(NR, NTH, NZ, 0.0f);
    Field3D v_out(NR, NTH, NZ, 0.0f);
    Field3D w_out(NR, NTH, NZ, 0.0f);

    const AcousticMomentumDispatch dispatch{rho_f.data(),
                                            p_f.data(),
                                            zero.data(),
                                            zero.data(),
                                            zero.data(),
                                            u_out.data(),
                                            v_out.data(),
                                            w_out.data(),
                                            NR,
                                            NTH,
                                            NZ,
                                            100.0f,
                                            100.0f,
                                            100.0f,
                                            0.1f,
                                            120.0f,
                                            90.0f};
    const bool                     dispatched = kernels->dispatch_acoustic_momentum(dispatch);
    REQUIRE(dispatched);

    const float v_inner = v_out(4, 8, 8);
    const float v_outer = v_out(10, 8, 8);
    INFO("v_out(4,8,8)=" << v_inner << " v_out(10,8,8)=" << v_outer);
    // Non-trivial response to the pressure gradient...
    REQUIRE(std::abs(v_inner) > 1e-12f);
    // ...and identical across i rows (Cartesian translation invariance;
    // the cylindrical shader would differ by a factor of 10/4 here).
    REQUIRE(v_inner == Approx(v_outer).epsilon(1e-5));

    shutdown_backend();
}

// ---- Hardware info population ----

TEST_CASE("Vulkan backend populates HardwareInfo GPU fields", "[vulkan][gpu_parity]")
{
    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping HardwareInfo population test");
        return;
    }

    auto*        backend = mutable_compute_backend();
    HardwareInfo hw;
    backend->populate_hardware_info(hw);
    // Regression: populate_hardware_info was never overridden, so the
    // banner printed "GPU: not available" while Vulkan kernels ran.
    REQUIRE(hw.gpu_available);
    REQUIRE(hw.gpu_device_name != "none");
    REQUIRE_FALSE(hw.gpu_device_name.empty());

    const ComputeMemoryProperties memory = backend->memory_properties();
    REQUIRE(memory.accelerator);
    REQUIRE(memory.device_local_heap_bytes > 0);
    REQUIRE(memory.max_storage_buffer_bytes >= sizeof(float));

    shutdown_backend();
}

TEST_CASE("Vulkan Earth-control evolution preserves Gaussian and AR reconstruction contracts atomically",
          "[vulkan][gpu_parity][chaos][restart][fail_closed]")
{
    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping Earth-control evolution parity");
        return;
    }
    auto* backend = mutable_compute_backend();
    REQUIRE(backend != nullptr);
    REQUIRE(backend->supports_earth_control_evolution());

    constexpr std::uint64_t seed = 91234567ULL;
    constexpr int member = -7;
    constexpr std::uint64_t initial_key = 4001ULL;
    constexpr std::uint64_t innovation_key = 4101ULL;
    constexpr double rho_value = 0.73;
    tmv::chaos::EarthCorrelatedNoise cpu;
    cpu.configure(seed, member, initial_key, innovation_key,
                  900.0, 700.0, 200.0, 250.0);
    cpu.initialize_domain(27, 23, 5);
    const Field3D& cpu_initial = cpu.current_controls();
    const std::size_t count = cpu_initial.size();

    std::vector<float> zero(count, 0.0f);
    std::vector<float> native_initial(count, -91.0f);
    EarthControlEvolutionDispatch dispatch;
    dispatch.previous = zero.data();
    dispatch.destination = native_initial.data();
    dispatch.seed = seed;
    dispatch.member_id = member;
    dispatch.initial_process_key = initial_key;
    dispatch.innovation_process_key = innovation_key;
    dispatch.control_i_min = cpu.control_i_min();
    dispatch.control_j_min = cpu.control_j_min();
    dispatch.control_i_count = cpu_initial.size_r();
    dispatch.control_j_count = cpu_initial.size_th();
    dispatch.nz = cpu_initial.size_z();
    dispatch.mode = EarthControlEvolutionMode::Initialize;
    REQUIRE(backend->dispatch_earth_control_evolution(dispatch));

    double max_initial_error = 0.0;
    for (std::size_t index = 0; index < count; ++index)
        max_initial_error = std::max(
            max_initial_error,
            std::abs(static_cast<double>(native_initial[index]) -
                     static_cast<double>(cpu_initial.data()[index])));
    REQUIRE(max_initial_error <= 3.0e-5);

    for (double correlation : {1.0 - 1e-8, 1.0 - 1e-10, std::nextafter(1.0, 0.0)})
    {
        CAPTURE(correlation);
        auto near_unit = dispatch;
        std::vector<float> evolved(count, -93.0f);
        near_unit.destination = evolved.data();
        near_unit.innovation_process_key = initial_key;
        near_unit.rho = correlation;
        near_unit.mode = EarthControlEvolutionMode::Advance;
        REQUIRE(backend->dispatch_earth_control_evolution(near_unit));
        const double scale = std::sqrt(1.0 - correlation * correlation);
        for (std::size_t index = 0; index < count; ++index)
        {
            const double expected = scale * native_initial[index];
            CHECK(std::abs(double(evolved[index]) - expected) <=
                  8 * std::numeric_limits<float>::epsilon() * std::abs(expected));
        }
        near_unit.mode = EarthControlEvolutionMode::Reconstruct;
        near_unit.innovation_epoch = 1;
        REQUIRE(backend->dispatch_earth_control_evolution(near_unit));
        for (std::size_t index = 0; index < count; ++index)
        {
            const float expected = float((correlation + scale) * native_initial[index]);
            const double quantum = std::abs(double(std::nextafter(expected, 0.0f)) - expected);
            CHECK(std::abs(double(evolved[index]) - expected) <= 2 * quantum);
        }
        near_unit.mode = EarthControlEvolutionMode::Advance;
        near_unit.rho = float(correlation);
        REQUIRE(backend->dispatch_earth_control_evolution(near_unit));
        CHECK(std::all_of(evolved.begin(), evolved.end(), [](float value)
        {
            return value == 0.0f;
        }));
        const auto retained = evolved;
        for (double invalid : {std::nextafter(1.0, 2.0), std::nextafter(0.0, -1.0)})
        {
            near_unit.rho = invalid;
            CHECK_FALSE(backend->dispatch_earth_control_evolution(near_unit));
            CHECK(evolved == retained);
        }
    }

    for (std::uint64_t epoch = 0; epoch < 4; ++epoch)
        cpu.advance(rho_value, epoch);
    std::vector<float> native_reconstructed(count, -92.0f);
    dispatch.destination = native_reconstructed.data();
    dispatch.innovation_epoch = 4;
    dispatch.rho = rho_value;
    dispatch.mode = EarthControlEvolutionMode::Reconstruct;
    REQUIRE(backend->dispatch_earth_control_evolution(dispatch));

    double max_reconstruction_error = 0.0;
    for (std::size_t index = 0; index < count; ++index)
        max_reconstruction_error = std::max(
            max_reconstruction_error,
            std::abs(static_cast<double>(native_reconstructed[index]) -
                     static_cast<double>(cpu.current_controls().data()[index])));
    REQUIRE(max_reconstruction_error <= 8.0e-5);

    constexpr int sample_count = 262144;
    std::vector<float> samples(sample_count, 0.0f);
    std::vector<float> sample_zero(sample_count, 0.0f);
    dispatch.previous = sample_zero.data();
    dispatch.destination = samples.data();
    dispatch.control_i_min = -19;
    dispatch.control_j_min = 23;
    dispatch.control_i_count = 1;
    dispatch.control_j_count = 1;
    dispatch.nz = sample_count;
    dispatch.innovation_epoch = 0;
    dispatch.rho = 0.0f;
    dispatch.mode = EarthControlEvolutionMode::Initialize;
    REQUIRE(backend->dispatch_earth_control_evolution(dispatch));
    double sum = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    std::size_t four_sigma = 0;
    for (float value : samples)
    {
        sum += value;
        sum_sq += static_cast<double>(value) * value;
        max_abs = std::max(max_abs, std::abs(static_cast<double>(value)));
        four_sigma += std::abs(value) > 4.0f ? 1U : 0U;
    }
    const double mean = sum / sample_count;
    const double variance = sum_sq / sample_count - mean * mean;
    REQUIRE(std::abs(mean) < 0.01);
    REQUIRE(std::abs(variance - 1.0) < 0.02);
    REQUIRE(four_sigma >= 5);
    REQUIRE(four_sigma <= 40);
    REQUIRE(max_abs > 4.2);
    REQUIRE(std::sqrt(-2.0 * std::log(
                0.5 / 9007199254740992.0)) > 8.5);

    const std::vector<float> committed = native_reconstructed;
    dispatch.destination = native_reconstructed.data();
    dispatch.control_i_count = 0;
    REQUIRE_FALSE(backend->dispatch_earth_control_evolution(dispatch));
    REQUIRE(native_reconstructed == committed);
    REQUIRE(backend->successful_kernel_dispatch_count(
                nimbus::compute::KernelId::EarthControlEvolution) == 12);
    INFO("max_initial_abs=" << max_initial_error
         << " max_reconstruct_abs=" << max_reconstruction_error);
    shutdown_backend();
}

TEST_CASE("Vulkan homogeneous D5 field shift is bitwise equal to the canonical map",
          "[vulkan][gpu_parity][domain_recycling][D5]")
{
    constexpr int     nr    = 7;
    constexpr int     nth   = 8;
    constexpr int     nz    = 5;
    const std::size_t cells = static_cast<std::size_t>(nr) * nth * nz;

    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping homogeneous D5 field-shift parity");
        return;
    }
    auto* backend = mutable_compute_backend();
    REQUIRE(backend != nullptr);
    backend->set_coordinate_system(CoordinateSystem::Cartesian);
    REQUIRE(backend->supports_homogeneous_domain_field_shift());

    Field3D source(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t index = (static_cast<std::size_t>(i) * nth + j) * nz + k;
                source.data()[index]    = static_cast<float>(10000 * i + 100 * j + k);
            }
    const std::vector<float> leading{-11.25f, -22.5f, -33.75f, -45.0f, -56.25f};

    const std::array<std::array<int, 2>, 2>                   shifts{{
        {{2, -1}},
        {{-2, 2}},
    }};
    const std::array<tmv::domain::HorizontalFieldLocation, 3> locations{{
        tmv::domain::HorizontalFieldLocation::ScalarCenter,
        tmv::domain::HorizontalFieldLocation::XFace,
        tmv::domain::HorizontalFieldLocation::YFace,
    }};

    for (const auto& shift : shifts)
        for (const auto location : locations)
        {
            CAPTURE(shift[0], shift[1], static_cast<int>(location));
            const auto scalar_plan = tmv::domain::plan_whole_cell_shift_for_physical_domain(
                nr, nth, nz, shift[0], shift[1], 1, nr - 1, 1, nth - 1);
            const auto owner_plan = tmv::domain::plan_whole_cell_shift_for_horizontal_location(scalar_plan, location);
            Field3D    expected   = source;
            (void)tmv::domain::execute_homogeneous_field_shift(scalar_plan,
                                                               {{"oracle", &expected, &leading, location}});
            Field3D actual(nr, nth, nz, 987654.0f);

            HomogeneousDomainFieldShiftDispatch dispatch;
            dispatch.source              = source.data();
            dispatch.leading_profile     = leading.data();
            dispatch.destination         = actual.data();
            dispatch.nr                  = nr;
            dispatch.nth                 = nth;
            dispatch.nz                  = nz;
            dispatch.shift_i_cells       = owner_plan.shift_i_cells;
            dispatch.shift_j_cells       = owner_plan.shift_j_cells;
            dispatch.destination_i_begin = owner_plan.destination_i_begin;
            dispatch.destination_i_end   = owner_plan.destination_i_end;
            dispatch.destination_j_begin = owner_plan.destination_j_begin;
            dispatch.destination_j_end   = owner_plan.destination_j_end;
            dispatch.source_i_begin      = owner_plan.source_i_begin;
            dispatch.source_i_end        = owner_plan.source_i_end;
            dispatch.source_j_begin      = owner_plan.source_j_begin;
            dispatch.source_j_end        = owner_plan.source_j_end;

            REQUIRE(backend->dispatch_homogeneous_domain_field_shift(dispatch));
            REQUIRE(std::equal(actual.data(), actual.data() + cells, expected.data()));

            const Field3D before_invalid = actual;
            ++dispatch.source_i_begin;
            REQUIRE_FALSE(backend->dispatch_homogeneous_domain_field_shift(dispatch));
            REQUIRE(std::equal(actual.data(), actual.data() + cells, before_invalid.data()));
        }
    shutdown_backend();
}

TEST_CASE("Vulkan homogeneous D5 resident batch is atomic and uses one submission",
          "[vulkan][gpu_parity][domain_recycling][D5][batch]")
{
    constexpr int     nr          = 8;
    constexpr int     nth         = 9;
    constexpr int     nz          = 4;
    constexpr int     field_count = 9;
    const std::size_t cells       = static_cast<std::size_t>(nr) * nth * nz;

    if (!try_init_gpu())
    {
        WARN("GPU unavailable — skipping homogeneous D5 resident-batch parity");
        return;
    }
    auto* backend = mutable_compute_backend();
    REQUIRE(backend != nullptr);
    backend->set_coordinate_system(CoordinateSystem::Cartesian);
    REQUIRE(backend->supports_homogeneous_domain_field_shift());
    backend->begin_resident_step();

    const auto scalar_plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(nr, nth, nz, 2, -1, 1, nr - 1, 1, nth - 1);
    const std::array<tmv::domain::HorizontalFieldLocation, field_count> locations{{
        tmv::domain::HorizontalFieldLocation::ScalarCenter,
        tmv::domain::HorizontalFieldLocation::XFace,
        tmv::domain::HorizontalFieldLocation::YFace,
        tmv::domain::HorizontalFieldLocation::ScalarCenter,
        tmv::domain::HorizontalFieldLocation::XFace,
        tmv::domain::HorizontalFieldLocation::YFace,
        tmv::domain::HorizontalFieldLocation::ScalarCenter,
        tmv::domain::HorizontalFieldLocation::XFace,
        tmv::domain::HorizontalFieldLocation::YFace,
    }};
    std::array<Field3D, field_count>                                    actual;
    std::array<Field3D, field_count>                                    expected;
    std::array<std::vector<float>, field_count>                         profiles;
    std::array<std::string, field_count>                                id_storage;
    std::array<const char*, field_count>                                ids{};
    std::array<HomogeneousDomainFieldShiftDispatch, field_count>        fields{};

    for (int field = 0; field < field_count; ++field)
    {
        actual[static_cast<std::size_t>(field)] = Field3D(nr, nth, nz, 0.0f);
        for (std::size_t index = 0; index < cells; ++index)
        {
            actual[static_cast<std::size_t>(field)].data()[index] = static_cast<float>(field * 100000 + index * 3 + 1);
        }
        expected[static_cast<std::size_t>(field)] = actual[static_cast<std::size_t>(field)];
        profiles[static_cast<std::size_t>(field)].resize(nz);
        for (int k = 0; k < nz; ++k)
        {
            profiles[static_cast<std::size_t>(field)][static_cast<std::size_t>(k)] =
                static_cast<float>(-1000 - field * 10 - k);
        }
        (void)tmv::domain::execute_homogeneous_field_shift(scalar_plan,
                                                           {{"batch_oracle",
                                                             &expected[static_cast<std::size_t>(field)],
                                                             &profiles[static_cast<std::size_t>(field)],
                                                             locations[static_cast<std::size_t>(field)]}});

        const auto owner_plan = tmv::domain::plan_whole_cell_shift_for_horizontal_location(
            scalar_plan, locations[static_cast<std::size_t>(field)]);
        auto& dispatch                              = fields[static_cast<std::size_t>(field)];
        dispatch.source                             = actual[static_cast<std::size_t>(field)].data();
        dispatch.leading_profile                    = profiles[static_cast<std::size_t>(field)].data();
        dispatch.destination                        = actual[static_cast<std::size_t>(field)].data();
        dispatch.nr                                 = nr;
        dispatch.nth                                = nth;
        dispatch.nz                                 = nz;
        dispatch.shift_i_cells                      = owner_plan.shift_i_cells;
        dispatch.shift_j_cells                      = owner_plan.shift_j_cells;
        dispatch.destination_i_begin                = owner_plan.destination_i_begin;
        dispatch.destination_i_end                  = owner_plan.destination_i_end;
        dispatch.destination_j_begin                = owner_plan.destination_j_begin;
        dispatch.destination_j_end                  = owner_plan.destination_j_end;
        dispatch.source_i_begin                     = owner_plan.source_i_begin;
        dispatch.source_i_end                       = owner_plan.source_i_end;
        dispatch.source_j_begin                     = owner_plan.source_j_begin;
        dispatch.source_j_end                       = owner_plan.source_j_end;
        id_storage[static_cast<std::size_t>(field)] = "d5_batch_owner_" + std::to_string(field);
        ids[static_cast<std::size_t>(field)]        = id_storage[static_cast<std::size_t>(field)].c_str();
    }

    HomogeneousDomainFieldShiftBatchDispatch batch{fields.data(), ids.data(), field_count};
    const std::uint64_t                      before_first = backend->successful_dispatch_count();
    REQUIRE(backend->dispatch_homogeneous_domain_field_shift_batch(batch));
    REQUIRE(backend->successful_dispatch_count() == before_first + 1);
    for (int field = 0; field < field_count; ++field)
    {
        REQUIRE(std::equal(actual[static_cast<std::size_t>(field)].data(),
                           actual[static_cast<std::size_t>(field)].data() + cells,
                           expected[static_cast<std::size_t>(field)].data()));
    }

    // The same host identities remain current on the device. A second whole
    // batch must still cost only one submission and produce the second exact
    // production-map application.
    for (int field = 0; field < field_count; ++field)
    {
        (void)tmv::domain::execute_homogeneous_field_shift(scalar_plan,
                                                           {{"batch_oracle_second",
                                                             &expected[static_cast<std::size_t>(field)],
                                                             &profiles[static_cast<std::size_t>(field)],
                                                             locations[static_cast<std::size_t>(field)]}});
    }
    const std::uint64_t before_second = backend->successful_dispatch_count();
    REQUIRE(backend->dispatch_homogeneous_domain_field_shift_batch(batch));
    REQUIRE(backend->successful_dispatch_count() == before_second + 1);
    for (int field = 0; field < field_count; ++field)
    {
        REQUIRE(std::equal(actual[static_cast<std::size_t>(field)].data(),
                           actual[static_cast<std::size_t>(field)].data() + cells,
                           expected[static_cast<std::size_t>(field)].data()));
    }

    const auto          before_rejection            = actual;
    const std::uint64_t dispatches_before_rejection = backend->successful_dispatch_count();
    ++fields[4].source_i_begin;
    REQUIRE_FALSE(backend->dispatch_homogeneous_domain_field_shift_batch(batch));
    REQUIRE(backend->successful_dispatch_count() == dispatches_before_rejection);
    for (int field = 0; field < field_count; ++field)
    {
        REQUIRE(std::equal(actual[static_cast<std::size_t>(field)].data(),
                           actual[static_cast<std::size_t>(field)].data() + cells,
                           before_rejection[static_cast<std::size_t>(field)].data()));
    }
    shutdown_backend();
}

// ---- Backend config parsing (always runs) ----

TEST_CASE("Vulkan production A-grid continuity exports corrected retained mass flux",
          "[vulkan][gpu_parity][cartesian][agrid][acoustic][mass_flux]")
{
    using tmv::cartesian_mass_flux::CollocatedPressureCoupling;
    using tmv::init::config::BottomBC;
    using tmv::init::config::LateralBC;
    using tmv::init::config::TopBC;
    constexpr int nx = 8, ny = 7, nz = 6;
    constexpr float dx = 110.0f, dy = 90.0f, dz = 70.0f, dt = 0.08f;
    constexpr float frame_u = 1.25f, frame_v = -0.5f;
    if (!try_init_gpu())
    {
        WARN("GPU unavailable - skipping production A-grid continuity parity");
        return;
    }
    ComputeBackend* backend = mutable_compute_backend();
    backend->set_coordinate_system(CoordinateSystem::Cartesian);
    DynamicsComputeKernels* kernels = backend->dynamics_kernels();
    if (kernels == nullptr || !kernels->supports(
            DynamicsKernelOp::AcousticContinuityMassFluxCartesianAgrid))
    {
        WARN("production A-grid continuity pipeline unavailable - skipping");
        shutdown_backend();
        return;
    }

    Field3D u(nx, ny, nz), v(nx, ny, nz), w(nx, ny, nz);
    Field3D rho(nx, ny, nz), pressure(nx, ny, nz);
    Field3D expected_drho(nx, ny, nz, 0.0f);
    Field3D expected_x(nx, ny, nz, 0.0f);
    Field3D expected_y(nx, ny, nz, 0.0f);
    Field3D expected_z(nx, ny, nz, 0.0f);
    Field3D actual_drho(nx, ny, nz, 777.0f);
    Field3D actual_x(nx, ny, nz, 777.0f);
    Field3D actual_y(nx, ny, nz, 777.0f);
    Field3D actual_z(nx, ny, nz, 777.0f);
    std::vector<float> p0(static_cast<std::size_t>(nz));
    std::vector<double> p0_double(static_cast<std::size_t>(nz));
    for (int k = 0; k < nz; ++k)
    {
        p0[static_cast<std::size_t>(k)] = 100000.0f - 820.0f * k;
        p0_double[static_cast<std::size_t>(k)] = p0[static_cast<std::size_t>(k)];
    }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                u(i, j, k) = 3.0f + 0.08f * i - 0.03f * j + 0.01f * k;
                v(i, j, k) = -1.0f + 0.04f * i + 0.06f * j - 0.02f * k;
                w(i, j, k) = 0.01f * i - 0.015f * j + 0.025f * k;
                rho(i, j, k) = 1.2f - 0.025f * k + 0.001f * i - 0.0007f * j;
                const float checker = ((i + j + k) & 1) ? 18.0f : -18.0f;
                pressure(i, j, k) = p0[static_cast<std::size_t>(k)] + checker
                    + 2.5f * i - 1.75f * j + 0.4f * k;
            }
    const CollocatedPressureCoupling coupling{
        &pressure, &p0_double, dt, dx, dy, dz};
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                if (i < nx - 1)
                    expected_x(i, j, k) = static_cast<float>(
                        tmv::cartesian_mass_flux::x_face(
                            rho, u, i, j, k, nx, StaggerType::Collocated,
                            LateralBC::Periodic, LateralBC::Periodic,
                            &coupling, frame_u));
                if (j < ny - 1)
                    expected_y(i, j, k) = static_cast<float>(
                        tmv::cartesian_mass_flux::y_face(
                            rho, v, i, j, k, ny, StaggerType::Collocated,
                            LateralBC::Periodic, LateralBC::Periodic,
                            &coupling, frame_v));
                if (k < nz - 1)
                    expected_z(i, j, k) = static_cast<float>(
                        tmv::cartesian_mass_flux::z_face(
                            rho, w, i, j, k, nz, StaggerType::Collocated,
                            BottomBC::RigidSlip, TopBC::Open, &coupling));
                if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 &&
                    k > 0 && k < nz - 1)
                {
                    const double divergence =
                        (tmv::cartesian_mass_flux::x_face(
                             rho, u, i, j, k, nx, StaggerType::Collocated,
                             LateralBC::Periodic, LateralBC::Periodic,
                             &coupling, frame_u)
                         - tmv::cartesian_mass_flux::x_face(
                             rho, u, i - 1, j, k, nx, StaggerType::Collocated,
                             LateralBC::Periodic, LateralBC::Periodic,
                             &coupling, frame_u)) / dx
                        + (tmv::cartesian_mass_flux::y_face(
                               rho, v, i, j, k, ny, StaggerType::Collocated,
                               LateralBC::Periodic, LateralBC::Periodic,
                               &coupling, frame_v)
                           - tmv::cartesian_mass_flux::y_face(
                               rho, v, i, j - 1, k, ny, StaggerType::Collocated,
                               LateralBC::Periodic, LateralBC::Periodic,
                               &coupling, frame_v)) / dy
                        + (tmv::cartesian_mass_flux::z_face(
                               rho, w, i, j, k, nz, StaggerType::Collocated,
                               BottomBC::RigidSlip, TopBC::Open, &coupling)
                           - tmv::cartesian_mass_flux::z_face(
                               rho, w, i, j, k - 1, nz, StaggerType::Collocated,
                               BottomBC::RigidSlip, TopBC::Open, &coupling)) / dz;
                    expected_drho(i, j, k) = static_cast<float>(-divergence);
                }
            }

    AcousticContinuityMassFluxCartesianAgridDispatch dispatch{
        u.data(), v.data(), w.data(), rho.data(), pressure.data(),
        p0.data(), nz, actual_drho.data(), actual_x.data(), actual_y.data(),
        actual_z.data(), nx, ny, nz, dx, dy, dz, dt,
        static_cast<int>(LateralBC::Periodic),
        static_cast<int>(LateralBC::Periodic),
        static_cast<int>(LateralBC::Periodic),
        static_cast<int>(LateralBC::Periodic),
        static_cast<int>(TopBC::Open), false, 0.0f, frame_u, frame_v};
    REQUIRE(kernels->dispatch_acoustic_continuity_mass_flux_cartesian_agrid(
        dispatch));
    require_parity("A-grid corrected drho", expected_drho.data(), actual_drho.data(),
                   nx, ny, nz, 2.0e-6,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("A-grid corrected x flux", expected_x.data(), actual_x.data(),
                   nx, ny, nz, 2.0e-5,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("A-grid corrected y flux", expected_y.data(), actual_y.data(),
                   nx, ny, nz, 2.0e-5,
                   nimbus::parity::ComparisonDomain::FullField);
    require_parity("A-grid corrected z flux", expected_z.data(), actual_z.data(),
                   nx, ny, nz, 2.0e-5,
                   nimbus::parity::ComparisonDomain::FullField);

    const Field3D committed_drho = actual_drho;
    const Field3D committed_x = actual_x;
    dispatch.nx = 3;
    REQUIRE_FALSE(kernels->dispatch_acoustic_continuity_mass_flux_cartesian_agrid(
        dispatch));
    REQUIRE(std::memcmp(committed_drho.data(), actual_drho.data(),
                        committed_drho.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(committed_x.data(), actual_x.data(),
                        committed_x.size() * sizeof(float)) == 0);
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::AcousticContinuityMassFluxCartesianAgrid) == 1);
    shutdown_backend();
}

TEST_CASE("Vulkan native A-grid completion stages match the CPU operators atomically",
          "[vulkan][gpu_parity][cartesian][agrid][acoustic][complete]")
{
    if (!try_init_gpu())
    {
        WARN("GPU unavailable - skipping complete native A-grid stage parity");
        return;
    }
    ComputeBackend* backend = mutable_compute_backend();
    backend->set_coordinate_system(CoordinateSystem::Cartesian);
    DynamicsComputeKernels* kernels = backend->dynamics_kernels();
    REQUIRE(kernels != nullptr);
    const auto metrics = nimbus::test::run_agrid_complete_native_fixture(*kernels);
    INFO("pressure_max_abs=" << metrics.pressure_max_abs
         << " momentum_max_abs=" << metrics.momentum_max_abs
         << " constraint_max_abs=" << metrics.constraint_max_abs
         << " slow_max_abs=" << metrics.slow_max_abs);
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::AcousticPressureEosCartesianAgrid) == 2);
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::AcousticMomentumCartesianAgrid) == 1);
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::AcousticNullModeConstraintCartesianAgrid) == 3);
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::AcousticBoundaryCartesianAgrid) == 3);
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::CartesianTendencies) == 1);
    shutdown_backend();
}

TEST_CASE("ComputeBackend config parsing", "[vulkan][backend]")
{
    ComputeBackendKind kind;
    REQUIRE(parse_compute_backend_kind("cpu", kind));
    REQUIRE(kind == ComputeBackendKind::Cpu);
    REQUIRE(parse_compute_backend_kind("vulkan", kind));
    REQUIRE(kind == ComputeBackendKind::Vulkan);
    REQUIRE_FALSE(parse_compute_backend_kind("nonexistent", kind));
}

TEST_CASE("Vulkan SGS scalar walls preserve closed and periodic fluxes", "[vulkan][sgs_scalar_boundary]")
{
    REQUIRE(try_init_gpu());
    auto* backend = mutable_compute_backend();
    REQUIRE(backend != nullptr);
    REQUIRE(backend->turbulence_kernels() != nullptr);
    exercise_sgs_scalar_boundary(*backend->turbulence_kernels());
    shutdown_backend();
}

TEST_CASE("Vulkan acoustic validation preserves unrepaired candidates", "[vulkan][acoustic_unrepaired]")
{
    REQUIRE(try_init_gpu());
    auto* backend = mutable_compute_backend();
    REQUIRE(backend != nullptr);
    exercise_acoustic_unrepaired(*backend->dynamics_kernels());
    shutdown_backend();
}
