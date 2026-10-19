#include "catch2/catch.hpp"
#include "compute/support/sgs_scalar_boundary.hpp"
#include "compute/support/acoustic_unrepaired.hpp"

#include "compute/kernel_abi.hpp"
#include "compute/parity_check.hpp"
#include "compute/metal/metal_compute_backend.hpp"
#include "compute/compute_backend_factory.hpp"
#include "core/infra/domain_recycling.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/domain_motion.hpp"
#include "core/infra/hardware_info.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/grid_geometry.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation_state.hpp"
#include "dynamics/acoustic_energy_report.hpp"
#include "dynamics/eos_pressure.hpp"
#include "microphysics/schemes/lin/lin.hpp"
#include "numerics/diffusion/schemes/explicit/explicit.hpp"
#include "numerics/derivatives/derivative_operators.hpp"
#include "numerics/advection/advection_cartesian.hpp"
#include "numerics/positive_scalar_budget_contract.hpp"
#include "numerics/advection/cartesian_mass_flux.hpp"
#include "numerics/advection/scalar_boundary_report.hpp"
#include "terrain/base/topography.hpp"
#include "terrain/conservative_transport.hpp"
#include "terrain/native_mapped_transaction.hpp"
#include "turbulence/base/eddy_viscosity.hpp"
#include "boundary_layer/schemes/mynn/mynn_stability.hpp"
#include "boundary_layer/schemes/mynn/mynn.hpp"
#include "compute/support/mynn_batch_fixture.hpp"
#include "compute/support/rrtmg_batch_fixture.hpp"
#include "compute/support/agrid_complete_native_fixture.hpp"
#include "util/log_profile.hpp"
#include "chaos/base/earth_correlated_noise.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace
{
tmv::init::config::LateralBC       test_lateral_bc_west  = tmv::init::config::LateralBC::Rigid;
tmv::init::config::LateralBC       test_lateral_bc_east  = tmv::init::config::LateralBC::Rigid;
tmv::init::config::LateralBC       test_lateral_bc_south = tmv::init::config::LateralBC::Rigid;
tmv::init::config::LateralBC       test_lateral_bc_north = tmv::init::config::LateralBC::Rigid;
tmv::domain::HorizontalDomainFrame test_domain_frame;
} // namespace

tmv::init::config::LateralBC&       global_lateral_bc_west  = test_lateral_bc_west;
tmv::init::config::LateralBC&       global_lateral_bc_east  = test_lateral_bc_east;
tmv::init::config::LateralBC&       global_lateral_bc_south = test_lateral_bc_south;
tmv::init::config::LateralBC&       global_lateral_bc_north = test_lateral_bc_north;
tmv::domain::HorizontalDomainFrame& global_domain_frame     = test_domain_frame;

namespace
{
int                         test_nr = 1, test_nth = 1, test_nz = 1;
double                      test_dr = 1.0, test_dtheta = 1.0, test_dz = 1.0;
GridGeometry                test_grid_geometry;
StaggerType                 test_stagger_type = StaggerType::CGrid;
tmv::init::config::BottomBC test_bottom_bc    = tmv::init::config::BottomBC::RigidSlip;
tmv::init::config::TopBC    test_top_bc       = tmv::init::config::TopBC::Rigid;
Field3D                     test_u, test_v, test_w, test_rho, test_p, test_theta;
Field3D                     test_qv, test_qc, test_qr, test_qi, test_qs, test_qg, test_qh;
} // namespace
int&                         NR                   = test_nr;
int&                         NTH                  = test_nth;
int&                         NZ                   = test_nz;
double&                      dr                   = test_dr;
double&                      dtheta               = test_dtheta;
double&                      dz                   = test_dz;
GridGeometry&                global_grid_geometry = test_grid_geometry;
StaggerType&                 global_stagger_type  = test_stagger_type;
tmv::init::config::BottomBC& global_bottom_bc     = test_bottom_bc;
tmv::init::config::TopBC&    global_top_bc        = test_top_bc;
Field3D&                     u                    = test_u;
Field3D&                     v                    = test_v;
Field3D&                     w                    = test_w;
Field3D&                     rho                  = test_rho;
Field3D&                     p                    = test_p;
Field3D&                     theta                = test_theta;
Field3D&                     qv                   = test_qv;
Field3D&                     qc                   = test_qc;
Field3D&                     qr                   = test_qr;
Field3D&                     qi                   = test_qi;
Field3D&                     qs                   = test_qs;
Field3D&                     qg                   = test_qg;
Field3D&                     qh                   = test_qh;

SimulationStorageState global_simulation_storage{{1, 1, 1, 1.0, 1.0, 0.1, 1.0}};

namespace
{
ComputeBackendRuntimeState test_compute_backend_runtime;
}
ComputeBackendRuntimeState& global_compute_backend_runtime_state = test_compute_backend_runtime;
ComputeBackendConfig&       global_compute_backend_config        = test_compute_backend_runtime.config;
namespace
{
LogProfile test_log_profile = LogProfile::quiet;
}
LogProfile& global_log_profile = test_log_profile;
bool&       global_perf_timing_enabled = global_simulation_storage.configuration.perf_timing_enabled;

ComputeBackendKind active_compute_backend_kind(const ComputeBackendRuntimeState&)
{
    return ComputeBackendKind::Cpu;
}

bool compute_backend_fallback_active(const ComputeBackendRuntimeState&)
{
    return false;
}

ComputeBackendKind active_compute_backend_kind()
{
    return active_compute_backend_kind(global_compute_backend_runtime_state);
}

bool compute_backend_fallback_active()
{
    return compute_backend_fallback_active(global_compute_backend_runtime_state);
}

ComputeBackend* mutable_compute_backend()
{
    return global_compute_backend_runtime_state.backend.get();
}

ComputeBackend* mutable_compute_backend(ComputeBackendRuntimeState& runtime)
{
    return runtime.backend.get();
}

// The canonical CPU diffusion scheme probes the optional accelerator before
// executing its reference loops. This isolated Metal test binds that probe
// off so the oracle cannot recursively call the backend under test.
bool dispatch_diffusion_batch_backend(ComputeBackendRuntimeState&, const DiffusionBatchDispatch&)
{
    return false;
}

bool dispatch_lin_fused_backend(ComputeBackendRuntimeState&, const LinPointwiseDispatch&, const LinSedimentationDispatch&)
{
    return false;
}

bool dispatch_lin_pointwise_backend(ComputeBackendRuntimeState&, const LinPointwiseDispatch&)
{
    return false;
}

bool dispatch_lin_sedimentation_backend(ComputeBackendRuntimeState&, const LinSedimentationDispatch&)
{
    return false;
}

bool dispatch_lin_pointwise_backend(ComputeBackendRuntimeState&, const float*,
                                    const float*,
                                    const float*,
                                    const float*,
                                    const float*,
                                    const float*,
                                    const float*,
                                    const float*,
                                    const float*,
                                    float*,
                                    float*,
                                    float*,
                                    float*,
                                    float*,
                                    float*,
                                    float*,
                                    float*,
                                    int,
                                    int,
                                    int,
                                    float,
                                    float,
                                    float,
                                    float,
                                    float,
                                    float,
                                    float,
                                    float,
                                    float,
                                    float,
                                    float,
                                    float,
                                    float,
                                    float,
                                    float,
                                    float)
{
    return false;
}

bool dispatch_lin_sedimentation_backend(ComputeBackendRuntimeState&, const float*,
                                        const float*,
                                        const float*,
                                        const float*,
                                        const float*,
                                        float*,
                                        float*,
                                        float*,
                                        float*,
                                        int,
                                        int,
                                        int,
                                        float,
                                        float)
{
    return false;
}

namespace nimbus::parity
{
bool sample(ParityRuntimeState&, bool, const char*)
{
    return false;
}
bool check_field(ParityRuntimeState&, const char*, const char*, const float*, const float*, int, int, int, Tier, ComparisonDomain, int)
{
    return true;
}
} // namespace nimbus::parity

namespace
{

using nimbus::compute::KernelId;
using nimbus::metal::MetalComputeBackend;
using nimbus::metal::MetalKesslerFusedDispatch;

using Location = tmv::domain::HorizontalFieldLocation;

struct Case
{
    const char* name;
    int         shift_i;
    int         shift_j;
};

std::string fnv1a64_hex(const char* text)
{
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const unsigned char* byte = reinterpret_cast<const unsigned char*>(text); *byte != 0; ++byte)
    {
        hash ^= static_cast<std::uint64_t>(*byte);
        hash *= UINT64_C(1099511628211);
    }
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << hash;
    return stream.str();
}

std::string canonical_manifest_fingerprint()
{
    std::ostringstream canonical;
    const auto         append_text = [&canonical](const char* value)
    {
        const char* safe = value == nullptr ? "" : value;
        canonical << std::strlen(safe) << ':' << safe << ';';
    };
    for (const auto& entry : nimbus::compute::kernel_manifest())
    {
        canonical << static_cast<unsigned>(entry.id) << '|';
        append_text(entry.operation_id);
        append_text(entry.kernel_name);
        append_text(entry.artifact_name);
        append_text(entry.artifact_hash);
        canonical << static_cast<unsigned>(entry.resources.descriptor_span) << ',' << entry.resources.read_mask << ','
                  << entry.resources.write_mask << ',' << static_cast<unsigned>(entry.shape) << ','
                  << entry.constants.bytes << ',' << static_cast<unsigned>(entry.constants.alignment) << ','
                  << static_cast<unsigned>(entry.constants.version) << ',' << static_cast<unsigned>(entry.precision)
                  << ',' << static_cast<unsigned>(entry.cpu_fallback) << ',';
        append_text(entry.cpu_reference);
        canonical << static_cast<unsigned>(entry.parity_tier) << ',' << static_cast<unsigned>(entry.comparison_domain)
                  << ',' << static_cast<unsigned>(entry.coordinate) << ',' << static_cast<unsigned>(entry.availability)
                  << ',' << (entry.startup_required ? 1 : 0) << '\n';
    }
    return fnv1a64_hex(canonical.str().c_str());
}

HomogeneousDomainFieldShiftDispatch make_dispatch(const std::vector<float>&              source,
                                                  const std::vector<float>&              profile,
                                                  std::vector<float>&                    destination,
                                                  const tmv::domain::WholeCellShiftPlan& plan)
{
    HomogeneousDomainFieldShiftDispatch dispatch;
    dispatch.source              = source.data();
    dispatch.leading_profile     = profile.data();
    dispatch.destination         = destination.data();
    dispatch.nr                  = plan.nr;
    dispatch.nth                 = plan.nth;
    dispatch.nz                  = plan.nz;
    dispatch.shift_i_cells       = plan.shift_i_cells;
    dispatch.shift_j_cells       = plan.shift_j_cells;
    dispatch.destination_i_begin = plan.destination_i_begin;
    dispatch.destination_i_end   = plan.destination_i_end;
    dispatch.destination_j_begin = plan.destination_j_begin;
    dispatch.destination_j_end   = plan.destination_j_end;
    dispatch.source_i_begin      = plan.source_i_begin;
    dispatch.source_i_end        = plan.source_i_end;
    dispatch.source_j_begin      = plan.source_j_begin;
    dispatch.source_j_end        = plan.source_j_end;
    return dispatch;
}

void initialize_backend(MetalComputeBackend& backend)
{
    std::string error;
    const bool  initialized = backend.initialize(error);
    CAPTURE(error);
    REQUIRE(initialized);
}

float tvd_limiter_oracle(float r, int limiter_id)
{
    if (limiter_id == 0)
        return std::max(0.0f, std::min(r, 1.0f));
    if (limiter_id == 1)
        return (r + std::abs(r)) / (1.0f + std::abs(r));
    if (limiter_id == 2)
        return std::max(0.0f, std::max(std::min(2.0f * r, 1.0f), std::min(r, 2.0f)));
    return std::max(0.0f, std::min(std::min(2.0f * r, 2.0f), (1.0f + r) * 0.5f));
}

float tvd_reconstructed_oracle(const std::vector<float>& q, std::size_t base, int nz, int k, int limiter_id, bool right)
{
    const float center = q[base + static_cast<std::size_t>(k)];
    if (k == 0 || k == nz - 1)
        return center;
    const float next  = q[base + static_cast<std::size_t>(k + 1)];
    const float r     = (center - q[base + static_cast<std::size_t>(k - 1)]) / (next - center + 1.0e-30f);
    const float delta = tvd_limiter_oracle(r, limiter_id) * (next - center);
    return right ? center + 0.5f * delta : center - 0.5f * delta;
}

std::vector<float> tvd_vertical_flux_oracle(const std::vector<float>&  q,
                                            const std::vector<float>&  w,
                                            const std::vector<double>& dz,
                                            int                        nr,
                                            int                        nth,
                                            int                        nz,
                                            int                        limiter_id,
                                            bool                       positivity,
                                            double                     positivity_dt)
{
    std::vector<float> output(q.size(), 0.0f);
    for (int column = 0; column < nr * nth; ++column)
    {
        const std::size_t base = static_cast<std::size_t>(column) * nz;
        for (int k = 0; k < nz; ++k)
        {
            const std::size_t index    = base + static_cast<std::size_t>(k);
            float             tendency = 0.0f;
            if (k < nz - 1)
            {
                const float velocity_right = w[index];
                const float flux_right =
                    velocity_right >= 0.0f
                        ? velocity_right * tvd_reconstructed_oracle(q, base, nz, k, limiter_id, true)
                        : velocity_right * tvd_reconstructed_oracle(q, base, nz, k + 1, limiter_id, false);
                float flux_left = 0.0f;
                if (k > 0)
                {
                    const float velocity_left = w[index - 1];
                    flux_left = velocity_left >= 0.0f
                                    ? velocity_left * tvd_reconstructed_oracle(q, base, nz, k - 1, limiter_id, true)
                                    : velocity_left * tvd_reconstructed_oracle(q, base, nz, k, limiter_id, false);
                }
                tendency = -(flux_right - flux_left) / static_cast<float>(dz[static_cast<std::size_t>(k)]);
            }
            if (positivity)
            {
                if (!std::isfinite(tendency))
                    tendency = 0.0f;
                const float floor_tendency =
                    -q[index] / std::max(static_cast<float>(std::abs(positivity_dt)), 1.0e-12f);
                tendency = std::max(tendency, floor_tendency);
            }
            output[index] = std::isfinite(tendency) ? tendency : 0.0f;
        }
    }
    return output;
}

} // namespace

static_assert(sizeof(nimbus::compute::abi::HomogeneousDomainFieldShiftPushConstants) == 44,
              "Metal parity tests require the canonical 40-byte ABI");
static_assert(sizeof(nimbus::compute::abi::SmagorinskyTendenciesPushConstants) == 44,
              "Metal tendency parity tests require the canonical 44-byte ABI");
static_assert(sizeof(nimbus::compute::abi::SgsApplyPushConstants) == 56,
              "Metal SGS Apply parity tests require the canonical 52-byte ABI");
static_assert(sizeof(nimbus::compute::abi::DiffusionPushConstants) == 64,
              "Metal diffusion parity tests require the canonical 64-byte ABI");
static_assert(sizeof(nimbus::compute::abi::KesslerPointwisePushConstants) == 64,
              "Metal Kessler pointwise parity tests require the canonical 64-byte ABI");
static_assert(sizeof(nimbus::compute::abi::KesslerSedimentationPushConstants) == 64,
              "Metal Kessler sedimentation parity tests require the canonical 64-byte ABI");
static_assert(sizeof(nimbus::compute::abi::LinPointwisePushConstants) == 112,
              "Metal Lin pointwise parity tests require the canonical 112-byte ABI");
static_assert(sizeof(nimbus::compute::abi::LinSedimentationPushConstants) == 64,
              "Metal Lin sedimentation parity tests require the canonical 64-byte ABI");
static_assert(sizeof(nimbus::compute::abi::CartesianTendenciesPushConstants) == 64,
              "Metal A-grid tendency tests require the canonical 64-byte ABI");
static_assert(sizeof(nimbus::compute::abi::AcousticPressurePushConstants) == 40,
              "Metal A-grid pressure tests require the canonical 40-byte ABI");
static_assert(sizeof(nimbus::compute::abi::AcousticMomentumPushConstants) == 40,
              "Metal A-grid momentum tests require the canonical 40-byte ABI");
static_assert(sizeof(nimbus::compute::abi::MicrophysicsApplyPushConstants) == 64,
              "Metal microphysics apply parity tests require the canonical 64-byte ABI");
static_assert(sizeof(nimbus::compute::abi::AdvectionXPushConstants) == 32,
              "Metal advection-x parity tests require the canonical 32-byte ABI");
static_assert(sizeof(nimbus::compute::abi::AdvectionYPushConstants) == 32,
              "Metal advection-y parity tests require the canonical 32-byte ABI");
static_assert(sizeof(nimbus::compute::abi::TvdPushConstants) == 32,
              "Metal TVD vertical-flux parity tests require the canonical 32-byte ABI");
static_assert(sizeof(nimbus::compute::abi::AdvectMassFluxCartesianCgridPushConstants) == 24,
              "Metal conservative mass-flux tests require the canonical 24-byte ABI");
static_assert(sizeof(nimbus::compute::abi::AdvectUpdateCartesianCgridPushConstants) == 76,
              "Metal conservative update tests require the canonical 48-byte ABI");
static_assert(sizeof(nimbus::compute::abi::MappedTerrainStateUpdatePushConstants) == 32,
              "Metal mapped state-update tests require the canonical 32-byte ABI");
static_assert(sizeof(nimbus::compute::abi::MappedTerrainFluxAccumulationPushConstants) == 32,
              "Metal mapped flux-accumulation tests require the canonical 32-byte ABI");
static_assert(sizeof(nimbus::compute::abi::MappedTerrainMassContinuityPushConstants) == 64,
              "Metal mapped mass-continuity tests require the canonical 64-byte ABI");
static_assert(sizeof(nimbus::compute::abi::MappedTerrainFastMomentumPushConstants) == 32,
              "Metal mapped fast-momentum tests require the canonical 32-byte ABI");
static_assert(sizeof(nimbus::compute::abi::MappedTerrainSlowTendenciesPushConstants) == 64,
              "Metal mapped slow-tendency tests require the canonical 64-byte ABI");
static_assert(sizeof(nimbus::compute::abi::MappedTerrainScalarLimiterPushConstants) == 48,
              "Metal mapped scalar-limiter tests require the canonical 48-byte ABI");
static_assert(sizeof(nimbus::compute::abi::MappedTerrainScalarUpdatePushConstants) == 48,
              "Metal mapped scalar-update tests require the canonical 48-byte ABI");
static_assert(sizeof(nimbus::compute::abi::MappedTerrainEosPushConstants) == 16,
              "Metal mapped EOS tests require the canonical 16-byte ABI");
static_assert(sizeof(nimbus::compute::abi::MappedTerrainBoundaryPushConstants) == 48,
              "Metal mapped boundary tests require the canonical 48-byte ABI");

TEST_CASE("Metal shared factory constructs the native backend and honors device selection",
          "[metal][physical][factory]")
{
    auto backend = create_metal_compute_backend();
    REQUIRE(backend);
    REQUIRE(backend->name() == "metal");

    ComputeBackendConfig config;
    config.backend      = "metal";
    config.device_index = 0;
    backend->configure(config);

    std::string error;
    REQUIRE(backend->initialize(error));
    REQUIRE(error.empty());
    REQUIRE(backend->memory_properties().accelerator);
    backend->shutdown();

    config.device_index = std::numeric_limits<int>::max();
    backend->configure(config);
    REQUIRE_FALSE(backend->initialize(error));
    REQUIRE(error.find("out of range") != std::string::npos);
    REQUIRE(backend->successful_dispatch_count() == 0);
}

TEST_CASE("Metal mapped state and retained-flux commits match the accepted Vulkan transaction atomically",
          "[metal][physical][terrain][mapped-state-flux][gpu_parity][atomic]")
{
    constexpr int nx = 5, ny = 4, nz_levels = 3;
    constexpr std::size_t cells = static_cast<std::size_t>(nx * ny * nz_levels);
    constexpr float dt = 0.0375f;
    std::array<std::vector<float>, 11> state;
    for (auto& field : state)
        field.resize(cells);
    for (std::size_t index = 0; index < cells; ++index)
    {
        const float x = static_cast<float>(index);
        state[0][index] = 1.0f + 0.013f * x;
        state[1][index] = -0.5f + 0.009f * x;
        state[2][index] = 0.2f * std::sin(0.17f * x);
        state[3][index] = 1.15f - 0.0013f * x;
        state[4][index] = 90000.0f + 3.0f * x;
        state[5][index] = 298.0f + 0.04f * x;
        state[6][index] = 0.05f * float(int(index % 7) - 3);
        state[7][index] = -0.04f * float(int(index % 5) - 2);
        state[8][index] = 0.03f * float(int(index % 9) - 4);
        state[9][index] = 0.002f * float(int(index % 5) - 2);
        state[10][index] = 2.0f * float(int(index % 11) - 5);
    }

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.terrain_kernels() == &backend);
    REQUIRE(backend.supports_mapped_state_update());
    REQUIRE(backend.supports_mapped_flux_accumulation());
    REQUIRE(backend.supports_mapped_mass_continuity());
    REQUIRE(backend.supports_mapped_fast_momentum());
    REQUIRE(backend.supports_mapped_slow_tendencies());
    REQUIRE(backend.supports_mapped_scalar_transport());
    REQUIRE(backend.supports_mapped_final_state());
    REQUIRE(backend.ready(KernelId::MappedTerrainStateUpdate));
    REQUIRE(backend.ready(KernelId::MappedTerrainFluxAccumulation));

    std::array<std::vector<float>, 5> output;
    for (auto& field : output)
        field.assign(cells, -777.0f);
    MappedTerrainStateUpdateReport state_report{41u,
        MappedTerrainStateUpdateMode::FastMomentum, false};
    MappedTerrainStateUpdateDispatch state_dispatch{};
    state_dispatch.u = state[0].data();
    state_dispatch.v = state[1].data();
    state_dispatch.w = state[2].data();
    state_dispatch.rho = state[3].data();
    state_dispatch.p = state[4].data();
    state_dispatch.theta = state[5].data();
    state_dispatch.du_dt = state[6].data();
    state_dispatch.dv_dt = state[7].data();
    state_dispatch.dw_dt = state[8].data();
    state_dispatch.drho_dt = state[9].data();
    state_dispatch.dp_dt = state[10].data();
    state_dispatch.u_candidate = output[0].data();
    state_dispatch.v_candidate = output[1].data();
    state_dispatch.w_candidate = output[2].data();
    state_dispatch.rho_candidate = output[3].data();
    state_dispatch.p_candidate = output[4].data();
    state_dispatch.nx = nx;
    state_dispatch.ny = ny;
    state_dispatch.nz = nz_levels;
    state_dispatch.dt = dt;
    state_dispatch.report = &state_report;

    for (const auto mode : {MappedTerrainStateUpdateMode::Slow,
                            MappedTerrainStateUpdateMode::ContinuityEos,
                            MappedTerrainStateUpdateMode::FastMomentum})
    {
        state_dispatch.mode = mode;
        REQUIRE(backend.dispatch_mapped_state_update(state_dispatch));
        REQUIRE(state_report.complete);
        REQUIRE(state_report.dispatch_count == 1);
        REQUIRE(state_report.mode == mode);
        for (std::size_t index = 0; index < cells; ++index)
        {
            const float expected_u = mode == MappedTerrainStateUpdateMode::ContinuityEos
                ? state[0][index]
                : state[0][index] + state[6][index] * dt;
            const float expected_v = mode == MappedTerrainStateUpdateMode::ContinuityEos
                ? state[1][index]
                : state[1][index] + state[7][index] * dt;
            const float expected_w = mode == MappedTerrainStateUpdateMode::ContinuityEos
                ? state[2][index]
                : state[2][index] + state[8][index] * dt;
            const float expected_rho = mode == MappedTerrainStateUpdateMode::ContinuityEos
                ? state[3][index] + state[9][index] * dt
                : state[3][index];
            const float expected_p = mode == MappedTerrainStateUpdateMode::Slow
                ? state[4][index] + state[10][index] * dt
                : (mode == MappedTerrainStateUpdateMode::ContinuityEos
                    ? 100000.0f * std::pow(
                        expected_rho * 287.0f * state[5][index] / 100000.0f,
                        1.40027894f)
                    : state[4][index]);
            REQUIRE(std::abs(output[0][index] - expected_u) <= 2.0e-6f);
            REQUIRE(std::abs(output[1][index] - expected_v) <= 2.0e-6f);
            REQUIRE(std::abs(output[2][index] - expected_w) <= 2.0e-6f);
            REQUIRE(std::abs(output[3][index] - expected_rho) <= 2.0e-7f);
            REQUIRE(std::abs(output[4][index] - expected_p) <= 0.05f);
        }
    }

    const auto state_before_veto = output;
    const auto report_before_veto = state_report;
    state_dispatch.completed_observer = +[](void*, const float*, const float*,
                                             const float*, const float*, const float*)
    { throw std::runtime_error("reject Metal mapped state candidate"); };
    REQUIRE_THROWS_WITH(backend.dispatch_mapped_state_update(state_dispatch),
                        "reject Metal mapped state candidate");
    REQUIRE(output == state_before_veto);
    REQUIRE(state_report.dispatch_count == report_before_veto.dispatch_count);
    REQUIRE(state_report.mode == report_before_veto.mode);
    REQUIRE(state_report.complete == report_before_veto.complete);
    state_dispatch.completed_observer = nullptr;

    auto invalid_state = state;
    invalid_state[9][7] = -100.0f;
    state_dispatch.drho_dt = invalid_state[9].data();
    state_dispatch.mode = MappedTerrainStateUpdateMode::ContinuityEos;
    REQUIRE_FALSE(backend.dispatch_mapped_state_update(state_dispatch));
    REQUIRE(output == state_before_veto);
    REQUIRE(state_report.dispatch_count == report_before_veto.dispatch_count);
    state_dispatch.drho_dt = state[9].data();
    state_dispatch.u_candidate = state_dispatch.v_candidate;
    REQUIRE_FALSE(backend.dispatch_mapped_state_update(state_dispatch));
    REQUIRE(output == state_before_veto);
    state_dispatch.u_candidate = output[0].data();
    invalid_state = state;
    invalid_state[0][5] = std::numeric_limits<float>::quiet_NaN();
    state_dispatch.u = invalid_state[0].data();
    const auto state_dispatches_before_invalid = backend.successful_kernel_dispatch_count(
        KernelId::MappedTerrainStateUpdate);
    REQUIRE_FALSE(backend.dispatch_mapped_state_update(state_dispatch));
    REQUIRE(backend.successful_kernel_dispatch_count(
        KernelId::MappedTerrainStateUpdate) == state_dispatches_before_invalid);
    state_dispatch.u = state[0].data();

    std::array<std::vector<float>, 6> flux_input;
    for (auto& field : flux_input)
        field.resize(cells);
    for (std::size_t index = 0; index < cells; ++index)
        for (std::size_t field = 0; field < flux_input.size(); ++field)
            flux_input[field][index] = 0.25f * float(field + 1) +
                0.003f * float(index) * float(int(field % 3) - 1);
    std::array<std::vector<float>, 3> flux_output;
    for (auto& field : flux_output)
        field.assign(cells, -333.0f);
    MappedTerrainFluxAccumulationReport flux_report{29u, false};
    MappedTerrainFluxAccumulationDispatch flux_dispatch{};
    flux_dispatch.accumulated_x = flux_input[0].data();
    flux_dispatch.accumulated_y = flux_input[1].data();
    flux_dispatch.accumulated_zeta = flux_input[2].data();
    flux_dispatch.step_x = flux_input[3].data();
    flux_dispatch.step_y = flux_input[4].data();
    flux_dispatch.step_zeta = flux_input[5].data();
    flux_dispatch.accumulated_x_candidate = flux_output[0].data();
    flux_dispatch.accumulated_y_candidate = flux_output[1].data();
    flux_dispatch.accumulated_zeta_candidate = flux_output[2].data();
    flux_dispatch.nx = nx;
    flux_dispatch.ny = ny;
    flux_dispatch.nz = nz_levels;
    flux_dispatch.time_weight = 1.0f / 6.0f;
    flux_dispatch.report = &flux_report;
    REQUIRE(backend.dispatch_mapped_flux_accumulation(flux_dispatch));
    REQUIRE(flux_report.complete);
    REQUIRE(flux_report.dispatch_count == 1);
    for (std::size_t index = 0; index < cells; ++index)
        for (std::size_t field = 0; field < flux_output.size(); ++field)
        {
            const float expected = flux_input[field][index] +
                flux_input[field + 3][index] * flux_dispatch.time_weight;
            REQUIRE(std::abs(flux_output[field][index] - expected) <= 2.0e-7f);
        }

    const auto flux_before_veto = flux_output;
    const auto flux_report_before_veto = flux_report;
    flux_dispatch.completed_observer = +[](void*, const float*, const float*, const float*)
    { throw std::runtime_error("reject Metal mapped flux candidate"); };
    REQUIRE_THROWS_WITH(backend.dispatch_mapped_flux_accumulation(flux_dispatch),
                        "reject Metal mapped flux candidate");
    REQUIRE(flux_output == flux_before_veto);
    REQUIRE(flux_report.dispatch_count == flux_report_before_veto.dispatch_count);
    REQUIRE(flux_report.complete == flux_report_before_veto.complete);
    flux_dispatch.completed_observer = nullptr;
    flux_dispatch.accumulated_y_candidate = flux_dispatch.accumulated_x_candidate;
    REQUIRE_FALSE(backend.dispatch_mapped_flux_accumulation(flux_dispatch));
    REQUIRE(flux_output == flux_before_veto);
    flux_dispatch.accumulated_y_candidate = flux_output[1].data();
    flux_dispatch.time_weight = 0.0f;
    REQUIRE_FALSE(backend.dispatch_mapped_flux_accumulation(flux_dispatch));
    REQUIRE(flux_output == flux_before_veto);

    REQUIRE(backend.successful_kernel_dispatch_count(
        KernelId::MappedTerrainStateUpdate) == 5);
    REQUIRE(backend.successful_kernel_dispatch_count(
        KernelId::MappedTerrainFluxAccumulation) == 2);
    const char* source = MetalComputeBackend::mapped_state_flux_msl_source();
    const char* source_hash = MetalComputeBackend::mapped_state_flux_msl_source_hash();
    REQUIRE(source != nullptr);
    REQUIRE(source_hash != nullptr);
    REQUIRE(std::strlen(source_hash) == 16);
    REQUIRE(fnv1a64_hex(source) == source_hash);
    REQUIRE(std::string(MetalComputeBackend::metal_artifact_source_hash(
        KernelId::MappedTerrainFluxAccumulation)) == source_hash);
}

TEST_CASE("Metal mapped mass continuity and slow-fast dynamics match production terrain operators atomically",
          "[metal][physical][terrain][mapped-acoustic][gpu_parity][atomic]")
{
    constexpr int mx = 7, my = 6, mz = 5;
    constexpr float dx_m = 700.0f, dy_m = 650.0f;
    constexpr float gamma_value = 1.4f;
    constexpr std::size_t cell_count = static_cast<std::size_t>(mx * my * mz);
    const auto at = [](int i, int j, int k) {
        return (static_cast<std::size_t>(i) * my + j) * mz + k;
    };

    TerrainConfig config;
    config.ztop = 5000.0;
    config.bell.h0 = 220.0;
    config.bell.a = 1600.0;
    config.bell.axisymmetric = true;
    Topography2D topo;
    topography::initialize_topography(topo, mx, my);
    for (int i = 0; i < mx; ++i)
        for (int j = 0; j < my; ++j)
        {
            const auto bell = topography::eval_bell(
                (i - 0.5 * (mx - 1)) * dx_m,
                (j - 0.5 * (my - 1)) * dy_m, config.bell);
            topo.h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = bell.h;
            topo.hx[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = bell.hx;
            topo.hy[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = bell.hy;
        }
    TerrainMetrics3D metrics;
    topography::initialize_metrics(metrics, mx, my, mz);
    topography::build_conservative_face_metrics(
        config, topo, topography::build_zeta_interfaces(mz, config.ztop),
        dx_m, dy_m, metrics);

    Field3D density(mx, my, mz), velocity_u(mx, my, mz), velocity_v(mx, my, mz);
    Field3D velocity_w(mx, my, mz), pressure(mx, my, mz);
    std::vector<float> jacobian(cell_count), jacobian_x(cell_count), jacobian_y(cell_count);
    std::vector<float> metric_x(cell_count), metric_y(cell_count);
    std::vector<float> ground_metric_x(cell_count), ground_metric_y(cell_count);
    std::vector<float> height(cell_count);
    for (int i = 0; i < mx; ++i)
        for (int j = 0; j < my; ++j)
            for (int k = 0; k < mz; ++k)
            {
                const std::size_t n = at(i, j, k);
                density[i][j][k] = 1.12f - 0.045f * k + 8.0e-4f * i - 5.0e-4f * j;
                velocity_u[i][j][k] = 4.0f + 0.11f * i - 0.06f * j + 0.03f * k;
                velocity_v[i][j][k] = -1.3f - 0.04f * i + 0.09f * j - 0.02f * k;
                velocity_w[i][j][k] = k == mz - 1 ? 0.0f :
                    0.06f * std::sin(0.4f * i + 0.7f * j + 0.2f * k);
                pressure[i][j][k] = 96000.0f - 1250.0f * k + 7.0f * i - 4.0f * j;
                jacobian[n] = static_cast<float>(metrics.J(i, j, k));
                jacobian_x[n] = i < mx - 1 ?
                    static_cast<float>(metrics.Jx_face(i + 1, j, k)) : 0.0f;
                jacobian_y[n] = j < my - 1 ?
                    static_cast<float>(metrics.Jy_face(i, j + 1, k)) : 0.0f;
                metric_x[n] = static_cast<float>(metrics.Gx_face(i, j, k + 1));
                metric_y[n] = static_cast<float>(metrics.Gy_face(i, j, k + 1));
                ground_metric_x[n] = static_cast<float>(metrics.Gx_face(i, j, 0));
                ground_metric_y[n] = static_cast<float>(metrics.Gy_face(i, j, 0));
                height[n] = static_cast<float>(metrics.z(i, j, k));
            }

    std::array<std::vector<float>, 6> expected;
    for (auto& field : expected)
        field.assign(cell_count, 0.0f);
    for (int i = 1; i < mx; ++i)
        for (int j = 1; j < my; ++j)
            expected[0][at(i, j, 0)] = static_cast<float>(
                tmv::terrain_transport::cgrid_ground_vertical_velocity(
                    metrics, velocity_u, velocity_v, i, j));
    for (int i = 0; i < mx - 1; ++i)
        for (int j = 0; j < my; ++j)
            for (int k = 0; k < mz; ++k)
                expected[3][at(i, j, k)] = static_cast<float>(
                    tmv::terrain_transport::cgrid_mass_flux_x(
                        metrics, density, velocity_u, i, j, k));
    for (int i = 0; i < mx; ++i)
        for (int j = 0; j < my - 1; ++j)
            for (int k = 0; k < mz; ++k)
                expected[4][at(i, j, k)] = static_cast<float>(
                    tmv::terrain_transport::cgrid_mass_flux_y(
                        metrics, density, velocity_v, i, j, k));
    for (int i = 1; i < mx - 1; ++i)
        for (int j = 1; j < my - 1; ++j)
            for (int k = 0; k < mz; ++k)
            {
                const std::size_t n = at(i, j, k);
                expected[5][n] = static_cast<float>(
                    tmv::terrain_transport::cgrid_mass_flux_zeta(
                        metrics, density, velocity_u, velocity_v, velocity_w,
                        i, j, k));
                expected[1][n] = static_cast<float>(-
                    tmv::terrain_transport::cgrid_mass_divergence(
                        metrics, density, velocity_u, velocity_v, velocity_w,
                        i, j, k));
                expected[2][n] = static_cast<float>(-gamma_value *
                    static_cast<double>(pressure[i][j][k]) *
                    tmv::terrain_transport::cgrid_velocity_divergence(
                        metrics, velocity_u, velocity_v, velocity_w, i, j, k));
            }

    MetalComputeBackend backend;
    initialize_backend(backend);
    std::array<std::vector<float>, 6> output;
    for (std::size_t field = 0; field < output.size(); ++field)
        output[field].assign(cell_count, -10.0f - static_cast<float>(field));
    MappedTerrainMassContinuityReport mass_report{17u, 19u, false};
    MappedTerrainMassContinuityDispatch mass{};
    mass.rho = density.data(); mass.u = velocity_u.data(); mass.v = velocity_v.data();
    mass.w = velocity_w.data(); mass.p = pressure.data(); mass.jacobian = jacobian.data();
    mass.jacobian_x_positive_face = jacobian_x.data();
    mass.jacobian_y_positive_face = jacobian_y.data();
    mass.metric_x_positive_zeta_face = metric_x.data();
    mass.metric_y_positive_zeta_face = metric_y.data();
    mass.metric_x_ground_face = ground_metric_x.data();
    mass.metric_y_ground_face = ground_metric_y.data();
    mass.ground_physical_w = output[0].data(); mass.drho_dt = output[1].data();
    mass.dp_dt = output[2].data(); mass.mass_flux_x_positive = output[3].data();
    mass.mass_flux_y_positive = output[4].data();
    mass.mass_flux_zeta_positive = output[5].data();
    mass.nx = mx; mass.ny = my; mass.nz = mz; mass.dx = dx_m; mass.dy = dy_m;
    mass.dzeta = static_cast<float>(metrics.dzeta); mass.gamma = gamma_value;
    mass.report = &mass_report;
    REQUIRE(backend.dispatch_mapped_mass_continuity(mass));
    REQUIRE(mass_report.complete);
    REQUIRE(mass_report.ground_tangency_dispatch_count == 1);
    REQUIRE(mass_report.mass_flux_continuity_dispatch_count == 1);
    const float tolerance[6] = {2.0e-5f, 2.0e-6f, 2.0e-2f, 2.0e-3f, 2.0e-3f, 2.0e-5f};
    for (std::size_t field = 0; field < output.size(); ++field)
        for (std::size_t n = 0; n < cell_count; ++n)
            REQUIRE(std::abs(output[field][n] - expected[field][n]) <= tolerance[field]);

    const auto mass_before_veto = output;
    const auto mass_report_before_veto = mass_report;
    mass.completed_observer = +[](void*, const float*, const float*, const float*,
                                  const float*, const float*, const float*)
    { throw std::runtime_error("reject Metal mapped continuity candidate"); };
    REQUIRE_THROWS_WITH(backend.dispatch_mapped_mass_continuity(mass),
                        "reject Metal mapped continuity candidate");
    REQUIRE(output == mass_before_veto);
    REQUIRE(mass_report.ground_tangency_dispatch_count ==
            mass_report_before_veto.ground_tangency_dispatch_count);
    REQUIRE(mass_report.mass_flux_continuity_dispatch_count ==
            mass_report_before_veto.mass_flux_continuity_dispatch_count);
    REQUIRE(mass_report.complete == mass_report_before_veto.complete);
    mass.completed_observer = nullptr;
    auto late_metric_x = metric_x;
    late_metric_x[at(3, 3, 2)] = std::numeric_limits<float>::quiet_NaN();
    mass.metric_x_positive_zeta_face = late_metric_x.data();
    REQUIRE_FALSE(backend.dispatch_mapped_mass_continuity(mass));
    REQUIRE(output == mass_before_veto);
    mass.metric_x_positive_zeta_face = metric_x.data();
    auto malformed_jacobian = jacobian;
    malformed_jacobian[at(3, 3, 2)] = std::numeric_limits<float>::quiet_NaN();
    mass.jacobian = malformed_jacobian.data();
    const auto mass_count_before = backend.successful_kernel_dispatch_count(
        KernelId::MappedTerrainMassContinuity);
    REQUIRE_FALSE(backend.dispatch_mapped_mass_continuity(mass));
    REQUIRE(backend.successful_kernel_dispatch_count(
        KernelId::MappedTerrainMassContinuity) == mass_count_before);
    mass.jacobian = jacobian.data();

    std::vector<float> reference_pressure = {
        96000.0f, 94750.0f, 93500.0f, 92250.0f, 91000.0f};
    const std::vector<double> reference(reference_pressure.begin(), reference_pressure.end());
    for (int i = 0; i < mx; ++i)
        for (int j = 0; j < my; ++j)
            for (int k = 0; k < mz; ++k)
            {
                const double base = tmv::terrain_transport::reference_profile_value_fp32(
                    reference, metrics.z(i, j, k), metrics.dzeta);
                pressure[i][j][k] = static_cast<float>(base +
                    24.0 * std::sin(0.31 * i + 0.43 * j + 0.27 * k));
            }
    std::array<std::vector<float>, 3> fast_expected;
    std::array<std::vector<float>, 3> fast_output;
    for (auto& field : fast_expected) field.assign(cell_count, 0.0f);
    for (auto& field : fast_output) field.assign(cell_count, -21.0f);
    for (int i = 1; i < mx - 1; ++i)
        for (int j = 1; j < my - 1; ++j)
            for (int k = 0; k < mz - 1; ++k)
            {
                const std::size_t n = at(i, j, k);
                const double rho_x = 0.5 * (density[i][j][k] + density[i + 1][j][k]);
                const double rho_y = 0.5 * (density[i][j][k] + density[i][j + 1][k]);
                const double rho_z = 0.5 * (density[i][j][k] + density[i][j][k + 1]);
                fast_expected[0][n] = static_cast<float>(-
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_x(
                        metrics, pressure, reference, i, j, k) / rho_x);
                fast_expected[1][n] = static_cast<float>(-
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_y(
                        metrics, pressure, reference, i, j, k) / rho_y);
                fast_expected[2][n] = static_cast<float>(-
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_z(
                        metrics, pressure, reference, i, j, k) / rho_z);
            }
    MappedTerrainFastMomentumReport fast_report{23u, false};
    MappedTerrainFastMomentumDispatch fast{};
    fast.rho = density.data(); fast.p = pressure.data(); fast.physical_height = height.data();
    fast.reference_pressure = reference_pressure.data(); fast.du_dt = fast_output[0].data();
    fast.dv_dt = fast_output[1].data(); fast.dw_dt = fast_output[2].data();
    fast.nx = mx; fast.ny = my; fast.nz = mz; fast.dx = dx_m; fast.dy = dy_m;
    fast.nominal_dz = static_cast<float>(metrics.dzeta); fast.report = &fast_report;
    REQUIRE(backend.dispatch_mapped_fast_momentum(fast));
    REQUIRE(fast_report.complete);
    REQUIRE(fast_report.pressure_force_dispatch_count == 1);
    for (std::size_t field = 0; field < fast_output.size(); ++field)
        for (std::size_t n = 0; n < cell_count; ++n)
            REQUIRE(std::abs(fast_output[field][n] - fast_expected[field][n]) <= 5.0e-5f);

    const auto fast_before_veto = fast_output;
    const auto fast_report_before_veto = fast_report;
    fast.completed_observer = +[](void*, const float*, const float*, const float*)
    { throw std::runtime_error("reject Metal mapped pressure-force candidate"); };
    REQUIRE_THROWS_WITH(backend.dispatch_mapped_fast_momentum(fast),
                        "reject Metal mapped pressure-force candidate");
    REQUIRE(fast_output == fast_before_veto);
    REQUIRE(fast_report.pressure_force_dispatch_count ==
            fast_report_before_veto.pressure_force_dispatch_count);
    REQUIRE(fast_report.complete == fast_report_before_veto.complete);
    fast.completed_observer = nullptr;
    auto late_height = height;
    late_height[at(3, 3, 2)] = late_height[at(3, 3, 1)];
    fast.physical_height = late_height.data();
    REQUIRE_FALSE(backend.dispatch_mapped_fast_momentum(fast));
    REQUIRE(fast_output == fast_before_veto);

    REQUIRE(backend.successful_kernel_dispatch_count(
        KernelId::MappedTerrainMassContinuity) == 6);
    REQUIRE(backend.successful_kernel_dispatch_count(
        KernelId::MappedTerrainFastMomentum) == 3);

    std::vector<float> reference_theta = {
        300.0f, 301.0f, 302.0f, 303.0f, 304.0f};
    std::vector<float> slow_qv(cell_count, 0.0f);
    std::vector<float> slow_reference_qv(mz, 0.0f);
    std::vector<float> slow_loading(cell_count, 0.0f);
    std::array<std::vector<float>, 8> slow_expected;
    std::array<std::vector<float>, 8> slow_output;
    for (auto& field : slow_expected) field.assign(cell_count, 0.0f);
    for (auto& field : slow_output) field.assign(cell_count, -31.0f);
    StaggeredCartesianDerivatives derivatives(
        dx_m, dy_m, metrics.dzeta, mx, my, mz);
    const std::vector<double> theta_reference(
        reference_theta.begin(), reference_theta.end());
    const double inv_dz = 1.0 / metrics.dzeta;
    for (int i = 1; i < mx - 1; ++i)
        for (int j = 1; j < my - 1; ++j)
            for (int k = 0; k < mz - 1; ++k)
            {
                const std::size_t n = at(i, j, k);
                const double ux = velocity_u[i][j][k];
                const double vy_at_x = 0.25 * (
                    velocity_v[i][j - 1][k] + velocity_v[i][j][k] +
                    velocity_v[i + 1][j - 1][k] + velocity_v[i + 1][j][k]);
                const double wx = k == 0 ? 0.25 * (
                    tmv::terrain_transport::cgrid_ground_vertical_velocity(
                        metrics, velocity_u, velocity_v, i, j) +
                    tmv::terrain_transport::cgrid_ground_vertical_velocity(
                        metrics, velocity_u, velocity_v, i + 1, j) +
                    velocity_w[i][j][0] + velocity_w[i + 1][j][0]) :
                    0.25 * (velocity_w[i][j][k - 1] + velocity_w[i][j][k] +
                            velocity_w[i + 1][j][k - 1] + velocity_w[i + 1][j][k]);
                const double zeta_x = tmv::terrain_transport::cgrid_zeta_velocity_at_x_face(
                    metrics, ux, vy_at_x, wx, i, j, k);
                const double duz = k == 0 ? (zeta_x >= 0.0 ? 0.0 :
                    (velocity_u[i][j][1] - velocity_u[i][j][0]) * inv_dz) :
                    derivatives.upwind_dz(velocity_u, zeta_x, i, j, k);
                const double adv_u = -ux * derivatives.upwind_dx(
                    velocity_u, ux, i, j, k) -
                    vy_at_x * derivatives.upwind_dy(velocity_u, vy_at_x, i, j, k) -
                    zeta_x * duz;
                slow_expected[0][n] = slow_expected[5][n] = static_cast<float>(adv_u);

                const double vy = velocity_v[i][j][k];
                const double ux_at_y = 0.25 * (
                    velocity_u[i - 1][j][k] + velocity_u[i][j][k] +
                    velocity_u[i - 1][j + 1][k] + velocity_u[i][j + 1][k]);
                const double wy = k == 0 ? 0.25 * (
                    tmv::terrain_transport::cgrid_ground_vertical_velocity(
                        metrics, velocity_u, velocity_v, i, j) +
                    tmv::terrain_transport::cgrid_ground_vertical_velocity(
                        metrics, velocity_u, velocity_v, i, j + 1) +
                    velocity_w[i][j][0] + velocity_w[i][j + 1][0]) :
                    0.25 * (velocity_w[i][j][k - 1] + velocity_w[i][j][k] +
                            velocity_w[i][j + 1][k - 1] + velocity_w[i][j + 1][k]);
                const double zeta_y = tmv::terrain_transport::cgrid_zeta_velocity_at_y_face(
                    metrics, ux_at_y, vy, wy, i, j, k);
                const double dvz = k == 0 ? (zeta_y >= 0.0 ? 0.0 :
                    (velocity_v[i][j][1] - velocity_v[i][j][0]) * inv_dz) :
                    derivatives.upwind_dz(velocity_v, zeta_y, i, j, k);
                const double adv_v = -ux_at_y * derivatives.upwind_dx(
                    velocity_v, ux_at_y, i, j, k) -
                    vy * derivatives.upwind_dy(velocity_v, vy, i, j, k) -
                    zeta_y * dvz;
                slow_expected[1][n] = slow_expected[6][n] = static_cast<float>(adv_v);

                const double ux_w = 0.25 * (
                    velocity_u[i - 1][j][k] + velocity_u[i][j][k] +
                    velocity_u[i - 1][j][k + 1] + velocity_u[i][j][k + 1]);
                const double vy_w = 0.25 * (
                    velocity_v[i][j - 1][k] + velocity_v[i][j][k] +
                    velocity_v[i][j - 1][k + 1] + velocity_v[i][j][k + 1]);
                const double zeta_w = tmv::terrain_transport::cgrid_zeta_velocity_at_z_face(
                    metrics, ux_w, vy_w, velocity_w[i][j][k], i, j, k);
                const double ground = tmv::terrain_transport::cgrid_ground_vertical_velocity(
                    metrics, velocity_u, velocity_v, i, j);
                const double dwz = k == 0 ? (zeta_w >= 0.0 ?
                    (velocity_w[i][j][0] - ground) * inv_dz :
                    (velocity_w[i][j][1] - velocity_w[i][j][0]) * inv_dz) :
                    derivatives.upwind_dz(velocity_w, zeta_w, i, j, k);
                const double adv_w = -ux_w * derivatives.upwind_dx(
                    velocity_w, ux_w, i, j, k) -
                    vy_w * derivatives.upwind_dy(velocity_w, vy_w, i, j, k) -
                    zeta_w * dwz;
                const double rho_face = 0.5 *
                    (density[i][j][k] + density[i][j][k + 1]);
                const double rho0_face = 0.5 * (
                    tmv::terrain_transport::reference_density_fp32(
                        reference, theta_reference, metrics.z(i, j, k), metrics.dzeta) +
                    tmv::terrain_transport::reference_density_fp32(
                        reference, theta_reference, metrics.z(i, j, k + 1), metrics.dzeta));
                slow_expected[2][n] = static_cast<float>(
                    adv_w - 9.81 * (rho_face - rho0_face) / rho_face);
                slow_expected[7][n] = static_cast<float>(adv_w);
            }
    for (int i = 1; i < mx - 1; ++i)
        for (int j = 1; j < my - 1; ++j)
            for (int k = 1; k < mz - 1; ++k)
            {
                const std::size_t n = at(i, j, k);
                const double center_u = 0.5 *
                    (velocity_u[i][j][k] + velocity_u[i - 1][j][k]);
                const double center_v = 0.5 *
                    (velocity_v[i][j][k] + velocity_v[i][j - 1][k]);
                const double center_w = 0.5 *
                    (velocity_w[i][j][k] + velocity_w[i][j][k - 1]);
                const double gx = 0.5 * (
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_x(
                        metrics, pressure, reference, i - 1, j, k) +
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_x(
                        metrics, pressure, reference, i, j, k));
                const double gy = 0.5 * (
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_y(
                        metrics, pressure, reference, i, j - 1, k) +
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_y(
                        metrics, pressure, reference, i, j, k));
                const double gz = tmv::terrain_transport::cgrid_scalar_gradient_z_center(
                    metrics, pressure, i, j, k);
                slow_expected[4][n] = static_cast<float>(
                    -center_u * gx - center_v * gy - center_w * gz);
            }

    MappedTerrainSlowTendenciesReport slow_report{37u, false, false};
    MappedTerrainSlowTendenciesDispatch slow{};
    slow.u = velocity_u.data(); slow.v = velocity_v.data(); slow.w = velocity_w.data();
    slow.rho = density.data(); slow.p = pressure.data(); slow.physical_height = height.data();
    slow.jacobian_x_positive_face = jacobian_x.data();
    slow.jacobian_y_positive_face = jacobian_y.data();
    slow.metric_x_positive_zeta_face = metric_x.data();
    slow.metric_y_positive_zeta_face = metric_y.data();
    slow.ground_physical_w = output[0].data();
    slow.reference_pressure = reference_pressure.data();
    slow.reference_theta = reference_theta.data(); slow.qv = slow_qv.data();
    slow.reference_qv = slow_reference_qv.data();
    slow.hydrometeor_loading = slow_loading.data();
    slow.du_dt = slow_output[0].data(); slow.dv_dt = slow_output[1].data();
    slow.dw_dt = slow_output[2].data(); slow.drho_dt = slow_output[3].data();
    slow.dp_dt = slow_output[4].data(); slow.du_advection_dt = slow_output[5].data();
    slow.dv_advection_dt = slow_output[6].data(); slow.dw_advection_dt = slow_output[7].data();
    slow.nx = mx; slow.ny = my; slow.nz = mz; slow.dx = dx_m; slow.dy = dy_m;
    slow.dzeta = static_cast<float>(metrics.dzeta);
    slow.nominal_dz = static_cast<float>(metrics.dzeta); slow.gravity = 9.81f;
    slow.report = &slow_report;
    REQUIRE(backend.dispatch_mapped_slow_tendencies(slow));
    REQUIRE(slow_report.complete);
    REQUIRE(slow_report.advection_attribution_complete);
    REQUIRE(slow_report.slow_tendency_dispatch_count == 1);
    const float slow_tolerance[8] = {
        2.0e-5f, 2.0e-5f, 2.0e-5f, 0.0f, 5.0e-2f,
        2.0e-5f, 2.0e-5f, 2.0e-5f};
    for (std::size_t field = 0; field < slow_output.size(); ++field)
        for (std::size_t n = 0; n < cell_count; ++n)
            REQUIRE(std::abs(slow_output[field][n] - slow_expected[field][n]) <=
                    slow_tolerance[field]);

    const auto dry_slow_output = slow_output;
    auto disabled_qv = slow_qv;
    auto disabled_reference_qv = slow_reference_qv;
    auto disabled_loading = slow_loading;
    std::fill(disabled_qv.begin(), disabled_qv.end(),
              std::numeric_limits<float>::quiet_NaN());
    std::fill(disabled_reference_qv.begin(), disabled_reference_qv.end(),
              std::numeric_limits<float>::quiet_NaN());
    std::fill(disabled_loading.begin(), disabled_loading.end(),
              std::numeric_limits<float>::quiet_NaN());
    slow.qv = disabled_qv.data(); slow.reference_qv = disabled_reference_qv.data();
    slow.hydrometeor_loading = disabled_loading.data();
    REQUIRE(backend.dispatch_mapped_slow_tendencies(slow));
    REQUIRE(slow_output == dry_slow_output);

    for (int level = 0; level < mz; ++level)
        slow_reference_qv[static_cast<std::size_t>(level)] = 0.006f - 0.0005f * level;
    for (std::size_t n = 0; n < cell_count; ++n)
    {
        slow_qv[n] = slow_reference_qv[n % mz] + 0.0007f;
        slow_loading[n] = 0.001f + 0.00001f * static_cast<float>(n % 7);
    }
    slow.qv = slow_qv.data(); slow.reference_qv = slow_reference_qv.data();
    slow.hydrometeor_loading = slow_loading.data();
    slow.has_moisture = true; slow.has_loading = true;
    REQUIRE(backend.dispatch_mapped_slow_tendencies(slow));
    const std::vector<double> qv_reference(
        slow_reference_qv.begin(), slow_reference_qv.end());
    for (int i = 1; i < mx - 1; ++i)
        for (int j = 1; j < my - 1; ++j)
            for (int k = 0; k < mz - 1; ++k)
            {
                const std::size_t n = at(i, j, k);
                const double moisture = 9.81 * 0.608 * (
                    0.5 * (slow_qv[n] + slow_qv[at(i, j, k + 1)]) -
                    0.5 * (tmv::terrain_transport::reference_profile_value_fp32(
                        qv_reference, metrics.z(i, j, k), metrics.dzeta) +
                    tmv::terrain_transport::reference_profile_value_fp32(
                        qv_reference, metrics.z(i, j, k + 1), metrics.dzeta)));
                const double load = -9.81 * 0.5 *
                    (slow_loading[n] + slow_loading[at(i, j, k + 1)]);
                REQUIRE(std::abs(slow_output[2][n] -
                    static_cast<float>(slow_expected[2][n] + moisture + load)) <= 2.0e-5f);
            }
    REQUIRE(slow_output[2] != dry_slow_output[2]);

    const auto coupled_slow_before = slow_output;
    const auto slow_report_before = slow_report;
    slow.completed_observer = +[](void*, const float*, const float*, const float*,
                                  const float*, const float*, const float*,
                                  const float*, const float*)
    { throw std::runtime_error("reject Metal mapped slow candidate"); };
    REQUIRE_THROWS_WITH(backend.dispatch_mapped_slow_tendencies(slow),
                        "reject Metal mapped slow candidate");
    REQUIRE(slow_output == coupled_slow_before);
    REQUIRE(slow_report.slow_tendency_dispatch_count ==
            slow_report_before.slow_tendency_dispatch_count);
    REQUIRE(slow_report.advection_attribution_complete ==
            slow_report_before.advection_attribution_complete);
    REQUIRE(slow_report.complete == slow_report_before.complete);
    slow.completed_observer = nullptr;
    auto late_slow_height = height;
    late_slow_height[at(3, 3, 2)] = late_slow_height[at(3, 3, 1)];
    slow.physical_height = late_slow_height.data();
    REQUIRE_FALSE(backend.dispatch_mapped_slow_tendencies(slow));
    REQUIRE(slow_output == coupled_slow_before);
    REQUIRE(backend.successful_kernel_dispatch_count(
        KernelId::MappedTerrainSlowTendencies) == 5);
    const char* slow_source = MetalComputeBackend::mapped_slow_tendencies_msl_source();
    const char* slow_hash = MetalComputeBackend::mapped_slow_tendencies_msl_source_hash();
    REQUIRE(slow_source != nullptr);
    REQUIRE(slow_hash != nullptr);
    REQUIRE(fnv1a64_hex(slow_source) == slow_hash);

    constexpr float scalar_step = 0.4f;
    std::vector<float> density_begin(density.data(), density.data() + cell_count);
    std::vector<float> density_end(cell_count);
    for (std::size_t n = 0; n < cell_count; ++n)
        density_end[n] = density_begin[n] + scalar_step * expected[1][n];
    const auto scalar_case = [&](bool positive_definite,
                                 std::vector<float>& scalar_begin,
                                 std::vector<float>& scalar_end,
                                 MappedTerrainScalarTransportReport& scalar_report,
                                 MappedTerrainScalarTransportDispatch& scalar_dispatch)
    {
        scalar_begin.resize(cell_count);
        scalar_end.assign(cell_count, -41.0f);
        for (int i = 0; i < mx; ++i)
            for (int j = 0; j < my; ++j)
                for (int k = 0; k < mz; ++k)
                {
                    const std::size_t n = at(i, j, k);
                    const float wave = std::exp(-0.24f * (
                        (i - 4.0f) * (i - 4.0f) + (j - 3.0f) * (j - 3.0f) +
                        (k - 2.0f) * (k - 2.0f)));
                    scalar_begin[n] = positive_definite
                        ? 0.001f + 0.006f * wave : 299.0f + 5.0f * wave;
                }
        if (positive_definite)
            scalar_begin[at(4, 3, 2)] = 1.0e-12f;
        scalar_dispatch = {};
        scalar_dispatch.scalar_begin = scalar_begin.data();
        scalar_dispatch.density_begin = density_begin.data();
        scalar_dispatch.density_end = density_end.data();
        scalar_dispatch.mass_flux_x = output[3].data();
        scalar_dispatch.mass_flux_y = output[4].data();
        scalar_dispatch.mass_flux_zeta = output[5].data();
        scalar_dispatch.jacobian = jacobian.data();
        scalar_dispatch.scalar_end = scalar_end.data();
        scalar_dispatch.nx = mx; scalar_dispatch.ny = my; scalar_dispatch.nz = mz;
        scalar_dispatch.positive_definite = positive_definite;
        scalar_dispatch.dx = dx_m; scalar_dispatch.dy = dy_m;
        scalar_dispatch.dzeta = static_cast<float>(metrics.dzeta);
        scalar_dispatch.dt = scalar_step;
        scalar_dispatch.density_floor = density_min_kgm3;
        scalar_dispatch.report = &scalar_report;
    };
    const auto scalar_oracle = [&](const std::vector<float>& scalar_begin,
                                   bool positive_definite)
    {
        SimulationStorageState storage{{mx, my, mz, dx_m, metrics.dzeta, scalar_step, dy_m}};
        storage.configuration.stagger_type = StaggerType::CGrid;
        storage.configuration.lateral_bc_west = tmv::init::config::LateralBC::LegacyOpen;
        storage.configuration.lateral_bc_east = tmv::init::config::LateralBC::LegacyOpen;
        storage.configuration.lateral_bc_south = tmv::init::config::LateralBC::LegacyOpen;
        storage.configuration.lateral_bc_north = tmv::init::config::LateralBC::LegacyOpen;
        storage.configuration.bottom_bc = tmv::init::config::BottomBC::RigidSlip;
        storage.configuration.top_bc = tmv::init::config::TopBC::Rigid;
        storage.grid.geometry.initialize(mx, my, mz, dx_m, metrics.dzeta, dy_m,
                                         CoordinateSystem::Cartesian, StaggerType::CGrid);
        storage.fields.rho.resize(mx, my, mz, 0.0f);
        storage.fields.u = velocity_u; storage.fields.v = velocity_v; storage.fields.w = velocity_w;
        std::copy(density_end.begin(), density_end.end(), storage.fields.rho.data());
        auto view = make_simulation_state_view(storage);
        Field3D source(mx, my, mz), result;
        std::copy(scalar_begin.begin(), scalar_begin.end(), source.data());
        CartesianCGridTransportContext context;
        context.begin(view, mx, my, mz, scalar_step, &metrics);
        context.accumulate(density, velocity_u, velocity_v, velocity_w, scalar_step);
        context.complete(view);
        advect_scalar_conservative_3d_cartesian(
            view, source, result, scalar_step, positive_definite, &context);
        return std::vector<float>(result.data(), result.data() + result.size());
    };
    struct ScalarCapture
    {
        std::array<std::vector<float>, 5> fields;
        std::size_t count = 0;
        bool called = false;
    };
    const auto capture_scalar = +[](void* raw, const float* scalar_end,
                                    const float* flux_x, const float* flux_y,
                                    const float* flux_z, const float* limiter)
    {
        auto& capture = *static_cast<ScalarCapture*>(raw);
        const float* source_fields[5] = {
            scalar_end, flux_x, flux_y, flux_z, limiter};
        for (std::size_t field = 0; field < capture.fields.size(); ++field)
            capture.fields[field].assign(
                source_fields[field], source_fields[field] + capture.count);
        capture.called = true;
    };

    std::vector<float> theta_begin, theta_end;
    MappedTerrainScalarTransportReport theta_report{41u, 43u, 47u, false};
    MappedTerrainScalarTransportDispatch theta_dispatch{};
    scalar_case(false, theta_begin, theta_end, theta_report, theta_dispatch);
    const auto theta_expected = scalar_oracle(theta_begin, false);
    ScalarCapture theta_capture{{}, cell_count, false};
    theta_dispatch.completed_observer = capture_scalar;
    theta_dispatch.completed_observer_context = &theta_capture;
    REQUIRE(backend.dispatch_mapped_scalar_transport(theta_dispatch));
    REQUIRE(theta_report.complete);
    REQUIRE(theta_report.scalar_flux_dispatch_count == 1);
    REQUIRE(theta_report.positive_definite_limiter_dispatch_count == 0);
    REQUIRE(theta_report.scalar_update_dispatch_count == 1);
    REQUIRE(theta_capture.called);
    REQUIRE(theta_capture.fields[0] == theta_end);
    for (std::size_t n = 0; n < cell_count; ++n)
        REQUIRE(std::abs(theta_end[n] - theta_expected[n]) <= 1.0e-4f);

    std::vector<float> qv_begin, qv_end;
    MappedTerrainScalarTransportReport qv_report{51u, 53u, 59u, false};
    MappedTerrainScalarTransportDispatch qv_dispatch{};
    scalar_case(true, qv_begin, qv_end, qv_report, qv_dispatch);
    const auto qv_expected = scalar_oracle(qv_begin, true);
    ScalarCapture qv_capture{{}, cell_count, false};
    qv_dispatch.completed_observer = capture_scalar;
    qv_dispatch.completed_observer_context = &qv_capture;
    REQUIRE(backend.dispatch_mapped_scalar_transport(qv_dispatch));
    REQUIRE(qv_report.complete);
    REQUIRE(qv_report.scalar_flux_dispatch_count == 1);
    REQUIRE(qv_report.positive_definite_limiter_dispatch_count == 1);
    REQUIRE(qv_report.scalar_update_dispatch_count == 1);
    REQUIRE(qv_capture.called);
    REQUIRE(*std::min_element(qv_capture.fields[4].begin(),
                              qv_capture.fields[4].end()) < 1.0f);
    REQUIRE(*std::min_element(qv_end.begin(), qv_end.end()) >= 0.0f);
    for (std::size_t n = 0; n < cell_count; ++n)
        REQUIRE(std::abs(qv_end[n] - qv_expected[n]) <= 2.0e-6f);

    const auto theta_before_veto = theta_end;
    const auto theta_report_before_veto = theta_report;
    theta_dispatch.completed_observer = +[](void*, const float*, const float*,
                                             const float*, const float*, const float*)
    { throw std::runtime_error("reject Metal mapped scalar candidate"); };
    theta_dispatch.completed_observer_context = nullptr;
    REQUIRE_THROWS_WITH(backend.dispatch_mapped_scalar_transport(theta_dispatch),
                        "reject Metal mapped scalar candidate");
    REQUIRE(theta_end == theta_before_veto);
    REQUIRE(theta_report.scalar_flux_dispatch_count ==
            theta_report_before_veto.scalar_flux_dispatch_count);
    REQUIRE(theta_report.positive_definite_limiter_dispatch_count ==
            theta_report_before_veto.positive_definite_limiter_dispatch_count);
    REQUIRE(theta_report.scalar_update_dispatch_count ==
            theta_report_before_veto.scalar_update_dispatch_count);
    REQUIRE(theta_report.complete == theta_report_before_veto.complete);
    theta_dispatch.completed_observer = nullptr;
    auto invalid_density_end = density_end;
    invalid_density_end[at(3, 3, 2)] = 0.0f;
    theta_dispatch.density_end = invalid_density_end.data();
    const auto scalar_count_before_invalid = backend.successful_kernel_dispatch_count(
        KernelId::AdvectConservativeScalarFluxCartesianCgrid);
    REQUIRE_FALSE(backend.dispatch_mapped_scalar_transport(theta_dispatch));
    REQUIRE(theta_end == theta_before_veto);
    REQUIRE(backend.successful_kernel_dispatch_count(
        KernelId::AdvectConservativeScalarFluxCartesianCgrid) ==
        scalar_count_before_invalid);
    REQUIRE(backend.successful_kernel_dispatch_count(
        KernelId::AdvectConservativeScalarFluxCartesianCgrid) == 3);
    REQUIRE(backend.successful_kernel_dispatch_count(
        KernelId::MappedTerrainScalarLimiter) == 1);
    REQUIRE(backend.successful_kernel_dispatch_count(
        KernelId::MappedTerrainScalarUpdate) == 3);
    const char* scalar_source = MetalComputeBackend::mapped_scalar_transport_msl_source();
    const char* scalar_hash = MetalComputeBackend::mapped_scalar_transport_msl_source_hash();
    REQUIRE(scalar_source != nullptr);
    REQUIRE(scalar_hash != nullptr);
    REQUIRE(fnv1a64_hex(scalar_source) == scalar_hash);

    std::array<std::vector<float>, 6> final_input;
    std::array<std::vector<float>, 5> final_output;
    final_input[0].assign(velocity_u.data(), velocity_u.data() + cell_count);
    final_input[1].assign(velocity_v.data(), velocity_v.data() + cell_count);
    final_input[2].assign(velocity_w.data(), velocity_w.data() + cell_count);
    final_input[3].assign(density.data(), density.data() + cell_count);
    final_input[4].assign(pressure.data(), pressure.data() + cell_count);
    final_input[5].resize(cell_count);
    for (int i = 0; i < mx; ++i)
        for (int j = 0; j < my; ++j)
            for (int k = 0; k < mz; ++k)
                final_input[5][at(i, j, k)] =
                    299.0f + 0.7f * k + 0.05f * i - 0.03f * j;
    for (auto& field : final_output)
        field.assign(cell_count, -51.0f);
    const auto final_expected = [&](bool apply_eos)
    {
        std::array<std::vector<float>, 5> expected_final;
        for (auto& field : expected_final)
            field.resize(cell_count);
        std::vector<float> pressure_owner = final_input[4];
        if (apply_eos)
            for (std::size_t n = 0; n < cell_count; ++n)
                pressure_owner[n] = clamp_pressure_pa(static_cast<float>(
                    tmv::dynamics_eos::pressure_fp64(
                        final_input[3][n], final_input[5][n])));
        for (int i = 0; i < mx; ++i)
            for (int j = 0; j < my; ++j)
                for (int k = 0; k < mz; ++k)
                {
                    const int source_i = i == 0 ? mx - 2 : (i == mx - 1 ? 1 : i);
                    const int source_j = j == 0 ? my - 2 : (j == my - 1 ? 1 : j);
                    const int source_k = k == mz - 1 ? mz - 2 : k;
                    const std::size_t n = at(i, j, k);
                    const std::size_t scalar_source = at(source_i, source_j, k);
                    const std::size_t velocity_source = at(source_i, source_j, source_k);
                    expected_final[0][n] = final_input[0][velocity_source];
                    expected_final[1][n] = final_input[1][velocity_source];
                    expected_final[2][n] = k == mz - 1 ? 0.0f :
                        final_input[2][scalar_source];
                    expected_final[3][n] = final_input[3][scalar_source];
                    expected_final[4][n] = pressure_owner[scalar_source];
                }
        return expected_final;
    };
    MappedTerrainFinalStateReport final_report{61u, 67u, false};
    MappedTerrainFinalStateDispatch final_dispatch{};
    final_dispatch.u = final_input[0].data(); final_dispatch.v = final_input[1].data();
    final_dispatch.w = final_input[2].data(); final_dispatch.rho = final_input[3].data();
    final_dispatch.p = final_input[4].data(); final_dispatch.theta = final_input[5].data();
    final_dispatch.u_final = final_output[0].data();
    final_dispatch.v_final = final_output[1].data();
    final_dispatch.w_final = final_output[2].data();
    final_dispatch.rho_final = final_output[3].data();
    final_dispatch.p_final = final_output[4].data();
    final_dispatch.nx = mx; final_dispatch.ny = my; final_dispatch.nz = mz;
    final_dispatch.report = &final_report;
    const auto expected_final = final_expected(true);
    REQUIRE(backend.dispatch_mapped_final_state(final_dispatch));
    REQUIRE(final_report.complete);
    REQUIRE(final_report.eos_dispatch_count == 1);
    REQUIRE(final_report.boundary_dispatch_count == 1);
    for (std::size_t field = 0; field < final_output.size(); ++field)
        for (std::size_t n = 0; n < cell_count; ++n)
            REQUIRE(std::abs(final_output[field][n] - expected_final[field][n]) <=
                    (field == 4 ? 0.05f : 0.0f));
    for (int i = 0; i < mx; ++i)
        for (int j = 0; j < my; ++j)
        {
            REQUIRE(final_output[2][at(i, j, mz - 1)] == 0.0f);
            REQUIRE(final_output[0][at(i, j, mz - 1)] ==
                    final_output[0][at(i, j, mz - 2)]);
            REQUIRE(final_output[1][at(i, j, mz - 1)] ==
                    final_output[1][at(i, j, mz - 2)]);
        }

    const auto final_before_veto = final_output;
    const auto final_report_before_veto = final_report;
    final_dispatch.completed_observer = +[](void*, const float*, const float*,
                                             const float*, const float*, const float*)
    { throw std::runtime_error("reject Metal mapped final-state candidate"); };
    REQUIRE_THROWS_WITH(backend.dispatch_mapped_final_state(final_dispatch),
                        "reject Metal mapped final-state candidate");
    REQUIRE(final_output == final_before_veto);
    REQUIRE(final_report.eos_dispatch_count == final_report_before_veto.eos_dispatch_count);
    REQUIRE(final_report.boundary_dispatch_count ==
            final_report_before_veto.boundary_dispatch_count);
    REQUIRE(final_report.complete == final_report_before_veto.complete);
    final_dispatch.completed_observer = nullptr;
    final_dispatch.v_final = final_dispatch.u_final;
    REQUIRE_FALSE(backend.dispatch_mapped_final_state(final_dispatch));
    REQUIRE(final_output == final_before_veto);
    final_dispatch.v_final = final_output[1].data();
    auto malformed_theta = final_input[5];
    malformed_theta[at(3, 3, 2)] = std::numeric_limits<float>::quiet_NaN();
    final_dispatch.theta = malformed_theta.data();
    REQUIRE_FALSE(backend.dispatch_mapped_final_state(final_dispatch));
    REQUIRE(final_output == final_before_veto);
    final_dispatch.theta = final_input[5].data();

    for (std::size_t n = 0; n < cell_count; ++n)
        final_input[4][n] = clamp_pressure_pa(static_cast<float>(
            tmv::dynamics_eos::pressure_fp64(final_input[3][n], final_input[5][n])));
    final_dispatch.p = final_input[4].data();
    final_dispatch.apply_eos = false;
    const auto expected_boundary_only = final_expected(false);
    REQUIRE(backend.dispatch_mapped_final_state(final_dispatch));
    REQUIRE(final_report.eos_dispatch_count == 0);
    REQUIRE(final_report.boundary_dispatch_count == 1);
    for (std::size_t field = 0; field < final_output.size(); ++field)
        REQUIRE(final_output[field] == expected_boundary_only[field]);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::MappedTerrainEos) == 2);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::MappedTerrainBoundary) == 3);
    const char* final_source = MetalComputeBackend::mapped_final_state_msl_source();
    const char* final_hash = MetalComputeBackend::mapped_final_state_msl_source_hash();
    REQUIRE(final_source != nullptr);
    REQUIRE(final_hash != nullptr);
    REQUIRE(fnv1a64_hex(final_source) == final_hash);

    const char* acoustic_source = MetalComputeBackend::mapped_acoustic_msl_source();
    const char* acoustic_hash = MetalComputeBackend::mapped_acoustic_msl_source_hash();
    REQUIRE(acoustic_source != nullptr);
    REQUIRE(acoustic_hash != nullptr);
    REQUIRE(fnv1a64_hex(acoustic_source) == acoustic_hash);
    REQUIRE(std::string(MetalComputeBackend::metal_artifact_source_hash(
        KernelId::MappedTerrainFastMomentum)) == acoustic_hash);
}

TEST_CASE("Metal dry stationary Bell mapped core commits one canonical transaction",
          "[metal][physical][terrain][mapped-core-transaction][atomic][threading]")
{
    constexpr int tx = 7, ty = 6, tz = 5;
    constexpr double tdx = 700.0, tdy = 650.0, ttop = 5000.0;
    constexpr std::size_t transaction_cells =
        static_cast<std::size_t>(tx * ty * tz);
    TerrainConfig terrain_config;
    terrain_config.scheme_id = "bell";
    terrain_config.ztop = ttop;
    terrain_config.bell.h0 = 220.0;
    terrain_config.bell.a = 1600.0;
    terrain_config.bell.axisymmetric = true;
    Topography2D transaction_topography;
    topography::initialize_topography(transaction_topography, tx, ty);
    for (int i = 0; i < tx; ++i)
        for (int j = 0; j < ty; ++j)
        {
            const auto bell = topography::eval_bell(
                (i - 0.5 * (tx - 1)) * tdx,
                (j - 0.5 * (ty - 1)) * tdy, terrain_config.bell);
            transaction_topography.h[static_cast<std::size_t>(i)]
                                      [static_cast<std::size_t>(j)] = bell.h;
            transaction_topography.hx[static_cast<std::size_t>(i)]
                                       [static_cast<std::size_t>(j)] = bell.hx;
            transaction_topography.hy[static_cast<std::size_t>(i)]
                                       [static_cast<std::size_t>(j)] = bell.hy;
        }
    TerrainMetrics3D transaction_metrics;
    topography::initialize_metrics(transaction_metrics, tx, ty, tz);
    topography::build_conservative_face_metrics(
        terrain_config, transaction_topography,
        topography::build_zeta_interfaces(tz, ttop), tdx, tdy,
        transaction_metrics);

    const auto make_storage = [&]
    {
        SimulationStorageState storage{{
            tx, ty, tz, tdx, transaction_metrics.dzeta, 0.05, tdy}};
        storage.grid.geometry.initialize(
            tx, ty, tz, tdx, transaction_metrics.dzeta, tdy,
            CoordinateSystem::Cartesian, StaggerType::CGrid);
        storage.runtime.terrain.config.scheme_id = "bell";
        storage.runtime.terrain.metrics = transaction_metrics;
        const auto initialize = [&](Field3D& field, float value)
        { field.resize(tx, ty, tz, value); };
        for (Field3D* field : {
                 &storage.fields.qv, &storage.fields.qc, &storage.fields.qr,
                 &storage.fields.qi, &storage.fields.qs, &storage.fields.qg,
                 &storage.fields.qh})
            initialize(*field, 0.0f);
        initialize(storage.fields.tracer, 0.0f);
        initialize(storage.fields.tke, 0.1f);
        for (Field3D* field : {
                 &storage.tendencies.dtheta_dt_rad,
                 &storage.tendencies.dtheta_dt_pbl,
                 &storage.tendencies.dqv_dt_pbl,
                 &storage.tendencies.du_dt_pbl,
                 &storage.tendencies.dv_dt_pbl,
                 &storage.tendencies.dtke_dt_pbl,
                 &storage.tendencies.dqc_dt_pbl,
                 &storage.tendencies.dqi_dt_pbl})
            initialize(*field, 0.0f);
        for (Field3D* field : {
                 &storage.fields.rho, &storage.fields.p, &storage.fields.u,
                 &storage.fields.v, &storage.fields.w, &storage.fields.theta})
            initialize(*field, 0.0f);
        storage.base.rho0.resize(tz);
        storage.base.p0.resize(tz);
        storage.base.qv0.assign(tz, 0.0);
        storage.base.theta0.resize(tz);
        storage.base.u0.assign(tz, 0.0);
        storage.base.v0.assign(tz, 0.0);
        for (int k = 0; k < tz; ++k)
        {
            const double reference_pressure = 96000.0 - 1250.0 * k;
            const double reference_theta = 300.0 + 0.65 * k;
            storage.base.p0[static_cast<std::size_t>(k)] = reference_pressure;
            storage.base.theta0[static_cast<std::size_t>(k)] = reference_theta;
            storage.base.rho0[static_cast<std::size_t>(k)] =
                physical_constants::reference_pressure_pa /
                (physical_constants::gas_constant_dry_air_jkgk * reference_theta) *
                std::pow(reference_pressure /
                         physical_constants::reference_pressure_pa,
                         1.0 / physical_constants::dry_air_gamma);
        }
        for (int i = 0; i < tx; ++i)
            for (int j = 0; j < ty; ++j)
                for (int k = 0; k < tz; ++k)
                {
                    const float theta_value = static_cast<float>(
                        storage.base.theta0[static_cast<std::size_t>(k)] +
                        0.015 * std::sin(0.3 * i + 0.2 * j + 0.4 * k));
                    const float rho_value = static_cast<float>(
                        storage.base.rho0[static_cast<std::size_t>(k)] +
                        8.0e-6 * std::sin(0.2 * i - 0.25 * j + 0.35 * k));
                    storage.fields.rho[i][j][k] = rho_value;
                    storage.fields.theta[i][j][k] = theta_value;
                    storage.fields.p[i][j][k] = clamp_pressure_pa(
                        static_cast<float>(tmv::dynamics_eos::pressure_fp64(
                            rho_value, theta_value)));
                    storage.fields.u[i][j][k] = 1.4f + 0.008f * i - 0.004f * j;
                    storage.fields.v[i][j][k] = -0.5f + 0.003f * i + 0.006f * j;
                    storage.fields.w[i][j][k] = k == tz - 1 ? 0.0f :
                        0.002f * std::sin(0.2f * i + 0.3f * j + 0.4f * k);
                    storage.fields.tracer[i][j][k] = 0.01f +
                        0.001f * std::sin(0.4f * i + 0.3f * j + 0.2f * k);
                    storage.fields.tke[i][j][k] = 0.1f +
                        0.002f * std::cos(0.3f * i - 0.2f * j + 0.4f * k);
                }
        return storage;
    };
    const auto public_fields = [](const SimulationStorageState& storage)
    {
        std::array<std::vector<float>, 9> fields;
        const Field3D* source[] = {
            &storage.fields.u, &storage.fields.v, &storage.fields.w,
            &storage.fields.rho, &storage.fields.p, &storage.fields.theta,
            &storage.fields.qv, &storage.fields.tracer, &storage.fields.tke};
        for (std::size_t field = 0; field < fields.size(); ++field)
            fields[field].assign(source[field]->data(),
                                 source[field]->data() + source[field]->size());
        return fields;
    };
    const auto same_report = [](const tmv::terrain_native::MappedCoreStepReport& a,
                                const tmv::terrain_native::MappedCoreStepReport& b)
    {
        const double acoustic_a[] = {
            a.acoustic.continuity_eos_dry_internal_change_j,
            a.acoustic.continuity_eos_water_sensible_change_j,
            a.acoustic.continuity_eos_phase_reference_change_j,
            a.acoustic.continuity_carrier_kinetic_change_j,
            a.acoustic.continuity_gravitational_change_j,
            a.acoustic.continuity_subgrid_tke_change_j,
            a.acoustic.continuity_carrier_requested_change_j,
            a.acoustic.continuity_carrier_outward_boundary_transport_j,
            a.acoustic.continuity_carrier_interior_conversion_j,
            a.acoustic.continuity_carrier_storage_rounding_j,
            a.acoustic.slow_momentum_advection_requested_linear_work_j,
            a.acoustic.kinetic_transport_interior_conversion_j,
            a.acoustic.kinetic_transport_closure_residual_j,
            a.acoustic.pressure_gradient_requested_linear_work_j,
            a.acoustic.pressure_gradient_finite_step_correction_j,
            a.acoustic.pressure_gradient_storage_rounding_j,
            a.acoustic.pressure_gradient_committed_work_j,
            a.acoustic.boundary_committed_work_j,
            a.acoustic.boundary_radiative_requested_linear_work_j,
            a.acoustic.boundary_radiative_finite_step_correction_j,
            a.acoustic.boundary_radiative_storage_rounding_j,
            a.acoustic.boundary_radiative_committed_work_j,
            a.acoustic.boundary_constraint_history_committed_work_j,
            a.acoustic.committed_inventory_change_j};
        const double acoustic_b[] = {
            b.acoustic.continuity_eos_dry_internal_change_j,
            b.acoustic.continuity_eos_water_sensible_change_j,
            b.acoustic.continuity_eos_phase_reference_change_j,
            b.acoustic.continuity_carrier_kinetic_change_j,
            b.acoustic.continuity_gravitational_change_j,
            b.acoustic.continuity_subgrid_tke_change_j,
            b.acoustic.continuity_carrier_requested_change_j,
            b.acoustic.continuity_carrier_outward_boundary_transport_j,
            b.acoustic.continuity_carrier_interior_conversion_j,
            b.acoustic.continuity_carrier_storage_rounding_j,
            b.acoustic.slow_momentum_advection_requested_linear_work_j,
            b.acoustic.kinetic_transport_interior_conversion_j,
            b.acoustic.kinetic_transport_closure_residual_j,
            b.acoustic.pressure_gradient_requested_linear_work_j,
            b.acoustic.pressure_gradient_finite_step_correction_j,
            b.acoustic.pressure_gradient_storage_rounding_j,
            b.acoustic.pressure_gradient_committed_work_j,
            b.acoustic.boundary_committed_work_j,
            b.acoustic.boundary_radiative_requested_linear_work_j,
            b.acoustic.boundary_radiative_finite_step_correction_j,
            b.acoustic.boundary_radiative_storage_rounding_j,
            b.acoustic.boundary_radiative_committed_work_j,
            b.acoustic.boundary_constraint_history_committed_work_j,
            b.acoustic.committed_inventory_change_j};
        const double slow_a[] = {
            a.slow_momentum.advection_requested_linear_work_j,
            a.slow_momentum.nonadvection_requested_linear_work_j,
            a.slow_momentum.advection_finite_step_correction_j,
            a.slow_momentum.nonadvection_finite_step_correction_j,
            a.slow_momentum.coupled_finite_step_cross_work_j,
            a.slow_momentum.storage_rounding_j,
            a.slow_momentum.committed_kinetic_change_j,
            a.slow_momentum.closure_residual_j,
            a.slow_momentum.pressure_dry_internal_change_j,
            a.slow_momentum.pressure_water_sensible_change_j,
            a.slow_momentum.committed_pressure_energy_change_j};
        const double slow_b[] = {
            b.slow_momentum.advection_requested_linear_work_j,
            b.slow_momentum.nonadvection_requested_linear_work_j,
            b.slow_momentum.advection_finite_step_correction_j,
            b.slow_momentum.nonadvection_finite_step_correction_j,
            b.slow_momentum.coupled_finite_step_cross_work_j,
            b.slow_momentum.storage_rounding_j,
            b.slow_momentum.committed_kinetic_change_j,
            b.slow_momentum.closure_residual_j,
            b.slow_momentum.pressure_dry_internal_change_j,
            b.slow_momentum.pressure_water_sensible_change_j,
            b.slow_momentum.committed_pressure_energy_change_j};
        if (!std::equal(std::begin(acoustic_a), std::end(acoustic_a),
                        std::begin(acoustic_b)) ||
            !std::equal(std::begin(slow_a), std::end(slow_a),
                        std::begin(slow_b)) ||
            a.acoustic.kinetic_transport_complete != b.acoustic.kinetic_transport_complete ||
            a.acoustic.boundary_application_count != b.acoustic.boundary_application_count ||
            a.acoustic.boundary_complete != b.acoustic.boundary_complete ||
            a.acoustic.substep_count != b.acoustic.substep_count ||
            a.acoustic.complete != b.acoustic.complete ||
            a.slow_momentum.complete != b.slow_momentum.complete)
            return false;
        for (std::size_t scalar = 0;
             scalar < tmv::thermodynamic_boundary_report_count; ++scalar)
        {
            const auto& ab = a.thermodynamic_boundary[scalar];
            const auto& bb = b.thermodynamic_boundary[scalar];
            if (ab.outward_boundary_inventory != bb.outward_boundary_inventory ||
                ab.flux_implied_state_change != bb.flux_implied_state_change ||
                ab.committed_state_change != bb.committed_state_change ||
                ab.closure_residual != bb.closure_residual ||
                ab.complete != bb.complete)
                return false;
            const auto& ae = a.thermodynamic_moist_energy[scalar];
            const auto& be = b.thermodynamic_moist_energy[scalar];
            const double energy_a[] = {
                ae.dry_internal_change_j, ae.water_sensible_change_j,
                ae.phase_reference_change_j, ae.carrier_kinetic_change_j,
                ae.gravitational_change_j, ae.subgrid_tke_change_j,
                ae.density_history_remap_j, ae.outward_boundary_energy_j,
                ae.interior_conversion_j, ae.limited_face_state_change_j,
                ae.requested_state_change_j, ae.storage_rounding_j,
                ae.committed_state_change_j, ae.closure_residual_j};
            const double energy_b[] = {
                be.dry_internal_change_j, be.water_sensible_change_j,
                be.phase_reference_change_j, be.carrier_kinetic_change_j,
                be.gravitational_change_j, be.subgrid_tke_change_j,
                be.density_history_remap_j, be.outward_boundary_energy_j,
                be.interior_conversion_j, be.limited_face_state_change_j,
                be.requested_state_change_j, be.storage_rounding_j,
                be.committed_state_change_j, be.closure_residual_j};
            if (!std::equal(std::begin(energy_a), std::end(energy_a),
                            std::begin(energy_b)) || ae.complete != be.complete)
                return false;
        }
        const double eos_a[] = {
            a.final_eos.dry_internal_change_j,
            a.final_eos.water_sensible_change_j,
            a.final_eos.phase_reference_change_j,
            a.final_eos.carrier_kinetic_change_j,
            a.final_eos.gravitational_change_j,
            a.final_eos.subgrid_tke_change_j,
            a.final_eos.committed_energy_change_j,
            a.final_eos.closure_residual_j};
        const double eos_b[] = {
            b.final_eos.dry_internal_change_j,
            b.final_eos.water_sensible_change_j,
            b.final_eos.phase_reference_change_j,
            b.final_eos.carrier_kinetic_change_j,
            b.final_eos.gravitational_change_j,
            b.final_eos.subgrid_tke_change_j,
            b.final_eos.committed_energy_change_j,
            b.final_eos.closure_residual_j};
        const double boundary_a[] = {
            a.final_boundary.constraint_committed_work_j,
            a.final_boundary.dry_internal_change_j,
            a.final_boundary.water_sensible_change_j,
            a.final_boundary.phase_reference_change_j,
            a.final_boundary.carrier_kinetic_change_j,
            a.final_boundary.gravitational_change_j,
            a.final_boundary.subgrid_tke_change_j,
            a.final_boundary.committed_energy_change_j,
            a.final_boundary.closure_residual_j};
        const double boundary_b[] = {
            b.final_boundary.constraint_committed_work_j,
            b.final_boundary.dry_internal_change_j,
            b.final_boundary.water_sensible_change_j,
            b.final_boundary.phase_reference_change_j,
            b.final_boundary.carrier_kinetic_change_j,
            b.final_boundary.gravitational_change_j,
            b.final_boundary.subgrid_tke_change_j,
            b.final_boundary.committed_energy_change_j,
            b.final_boundary.closure_residual_j};
        return a.acoustic_substeps == b.acoustic_substeps &&
            a.mass_continuity_dispatches == b.mass_continuity_dispatches &&
            a.slow_tendency_dispatches == b.slow_tendency_dispatches &&
        a.slow_average_dispatches == b.slow_average_dispatches &&
            a.fast_momentum_dispatches == b.fast_momentum_dispatches &&
            a.state_update_dispatches == b.state_update_dispatches &&
            a.flux_accumulation_dispatches == b.flux_accumulation_dispatches &&
            a.scalar_flux_dispatches == b.scalar_flux_dispatches &&
            a.scalar_limiter_dispatches == b.scalar_limiter_dispatches &&
            a.scalar_update_dispatches == b.scalar_update_dispatches &&
            a.boundary_dispatches == b.boundary_dispatches &&
            a.completed_backend_dispatches == b.completed_backend_dispatches &&
            a.accumulated_time_weight == b.accumulated_time_weight &&
            std::equal(std::begin(eos_a), std::end(eos_a), std::begin(eos_b)) &&
            a.final_eos.complete == b.final_eos.complete &&
            std::equal(std::begin(boundary_a), std::end(boundary_a),
                       std::begin(boundary_b)) &&
            a.final_boundary.applied == b.final_boundary.applied &&
            a.final_boundary.complete == b.final_boundary.complete &&
            a.final_pressure_uses_canonical_fp64 ==
                b.final_pressure_uses_canonical_fp64 &&
            a.complete == b.complete;
    };
    const auto make_runtime = []
    {
        ComputeBackendRuntimeState runtime;
        auto backend = std::make_unique<MetalComputeBackend>();
        std::string error;
        const bool initialized = backend->initialize(error);
        if (!initialized)
            throw std::runtime_error(error);
        runtime.backend = std::move(backend);
        runtime.active_kind = ComputeBackendKind::Metal;
        runtime.requested_kind = ComputeBackendKind::Metal;
        runtime.config.backend = "metal";
        runtime.config.allow_fallback = false;
        return runtime;
    };
    const auto attach_runtime = [](SimulationStorageState& storage,
                                   ComputeBackendRuntimeState& runtime)
    {
        auto view = make_simulation_state_view(storage);
        view.compute_runtime = &runtime;
        return view;
    };

    const int previous_threads = omp_get_max_threads();
    ComputeBackendRuntimeState one_runtime = make_runtime();
    SimulationStorageState one_thread = make_storage();
    auto one_view = attach_runtime(one_thread, one_runtime);
    tmv::terrain_native::MappedCoreStepReport one_report;
    omp_set_num_threads(1);
    REQUIRE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        one_view, 0.03, 3, one_report));
    REQUIRE(one_report.complete);
    REQUIRE(one_report.final_pressure_uses_canonical_fp64);
    REQUIRE(one_report.acoustic_substeps == 3);
    REQUIRE(one_report.mass_continuity_dispatches == 5);
    REQUIRE(one_report.slow_tendency_dispatches == 2);
    REQUIRE(one_report.slow_average_dispatches == 6);
    REQUIRE(one_report.fast_momentum_dispatches == 3);
    REQUIRE(one_report.state_update_dispatches == 8);
    REQUIRE(one_report.flux_accumulation_dispatches == 3);
    REQUIRE(one_report.scalar_flux_dispatches == 4);
    REQUIRE(one_report.scalar_limiter_dispatches == 3);
    REQUIRE(one_report.scalar_update_dispatches == 4);
    REQUIRE(one_report.boundary_dispatches == 8);
    REQUIRE(one_report.completed_backend_dispatches == 39);
    REQUIRE(one_report.accumulated_time_weight == Approx(1.0).margin(4.0e-7));
    REQUIRE(one_report.acoustic.complete);
    REQUIRE(one_report.acoustic.kinetic_transport_complete);
    REQUIRE(one_report.acoustic.boundary_complete);
    REQUIRE(one_report.acoustic.substep_count == 3);
    REQUIRE(one_report.acoustic.boundary_application_count == 4);
    REQUIRE(one_report.slow_momentum.complete);
    REQUIRE(one_report.final_eos.complete);
    REQUIRE(one_report.final_boundary.complete);
    const auto domain = physical_scalar_domain(one_view.grid);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
                REQUIRE(one_thread.fields.p[i][j][k] == clamp_pressure_pa(
                    static_cast<float>(tmv::dynamics_eos::pressure_fp64(
                        one_thread.fields.rho[i][j][k],
                        one_thread.fields.theta[i][j][k]))));
    auto* one_backend = one_runtime.backend.get();
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::MappedTerrainMassContinuity) == 10);
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::MappedTerrainSlowTendencies) == 2);
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::MappedTerrainFastMomentum) == 3);
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::MappedTerrainStateUpdate) == 8);
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::MappedTerrainFluxAccumulation) == 9);
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::AdvectConservativeScalarFluxCartesianCgrid) == 4);
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::AdvectConservativeUpdateCartesianCgrid) == 0);
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::MappedTerrainScalarLimiter) == 3);
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::MappedTerrainScalarUpdate) == 4);
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::MappedTerrainEos) == 0);
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::MappedTerrainBoundary) == 8);
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::DiffuseCartesian) == 0);
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::ArchivedLegacyThompsonLinPointwise) == 0);
    REQUIRE(one_backend->successful_kernel_dispatch_count(
        KernelId::ArchivedLegacyThompsonLinSedimentation) == 0);
    std::uint64_t transaction_kernel_dispatches = 0;
    for (std::size_t index = 0;
         index < nimbus::compute::kernel_count(); ++index)
        transaction_kernel_dispatches +=
            one_backend->successful_kernel_dispatch_count(
                static_cast<KernelId>(index));
    REQUIRE(one_backend->successful_dispatch_count() == 39);
    REQUIRE(transaction_kernel_dispatches == 51);

    ComputeBackendRuntimeState four_runtime = make_runtime();
    SimulationStorageState four_threads = make_storage();
    auto four_view = attach_runtime(four_threads, four_runtime);
    tmv::terrain_native::MappedCoreStepReport four_report;
    omp_set_num_threads(4);
    REQUIRE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        four_view, 0.03, 3, four_report));
    REQUIRE(public_fields(four_threads) == public_fields(one_thread));
    REQUIRE(same_report(four_report, one_report));

    ComputeBackendRuntimeState invalid_runtime = make_runtime();
    SimulationStorageState invalid = make_storage();
    invalid.fields.qv[3][3][2] = 1.0e-6f;
    auto invalid_view = attach_runtime(invalid, invalid_runtime);
    const auto invalid_before = public_fields(invalid);
    tmv::terrain_native::MappedCoreStepReport invalid_report;
    invalid_report.acoustic_substeps = 91;
    const auto invalid_report_before = invalid_report;
    REQUIRE_FALSE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        invalid_view, 0.03, 3, invalid_report));
    REQUIRE(public_fields(invalid) == invalid_before);
    REQUIRE(std::memcmp(&invalid_report, &invalid_report_before,
                        sizeof(invalid_report)) == 0);

    ComputeBackendRuntimeState nonfinite_runtime = make_runtime();
    SimulationStorageState nonfinite = make_storage();
    std::fill_n(nonfinite.fields.u.data(), transaction_cells,
                std::numeric_limits<float>::max() / 16.0f);
    auto nonfinite_view = attach_runtime(nonfinite, nonfinite_runtime);
    const auto nonfinite_before = public_fields(nonfinite);
    tmv::terrain_native::MappedCoreStepReport nonfinite_report;
    nonfinite_report.slow_tendency_dispatches = 59;
    const auto nonfinite_report_before = nonfinite_report;
    REQUIRE_FALSE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        nonfinite_view, 0.03, 3, nonfinite_report));
    REQUIRE(public_fields(nonfinite) == nonfinite_before);
    REQUIRE(std::memcmp(&nonfinite_report, &nonfinite_report_before,
                        sizeof(nonfinite_report)) == 0);

    ComputeBackendRuntimeState alias_runtime = make_runtime();
    SimulationStorageState aliased = make_storage();
    auto alias_source = attach_runtime(aliased, alias_runtime);
    SimulationStateView alias_view{
        alias_source.grid,
        {alias_source.fields.rho, alias_source.fields.p,
         alias_source.fields.u, alias_source.fields.w,
         alias_source.fields.u, alias_source.fields.tracer,
         alias_source.fields.theta, alias_source.fields.qv,
         alias_source.fields.qc, alias_source.fields.qr,
         alias_source.fields.qi, alias_source.fields.qs,
         alias_source.fields.qh, alias_source.fields.qg,
         alias_source.fields.tke},
        alias_source.base, alias_source.tendencies, alias_source.coriolis_f,
        alias_source.time_s, alias_source.surface};
    alias_view.terrain_runtime = alias_source.terrain_runtime;
    alias_view.compute_runtime = &alias_runtime;
    alias_view.configuration = alias_source.configuration;
    alias_view.diagnostics = alias_source.diagnostics;
    alias_view.dynamics_runtime = alias_source.dynamics_runtime;
    alias_view.runtime_owner = alias_source.runtime_owner;
    alias_view.domain_frame = alias_source.domain_frame;
    const auto alias_before = public_fields(aliased);
    tmv::terrain_native::MappedCoreStepReport alias_report;
    alias_report.boundary_dispatches = 63;
    const auto alias_report_before = alias_report;
    REQUIRE_FALSE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        alias_view, 0.03, 3, alias_report));
    REQUIRE(public_fields(aliased) == alias_before);
    REQUIRE(std::memcmp(&alias_report, &alias_report_before,
                        sizeof(alias_report)) == 0);

    ComputeBackendRuntimeState veto_runtime = make_runtime();
    SimulationStorageState vetoed = make_storage();
    auto vetoed_view = attach_runtime(vetoed, veto_runtime);
    const auto vetoed_before = public_fields(vetoed);
    tmv::terrain_native::MappedCoreStepReport vetoed_report;
    vetoed_report.mass_continuity_dispatches = 77;
    const auto vetoed_report_before = vetoed_report;
    REQUIRE_THROWS_WITH(tmv::terrain_native::advance_dry_stationary_mapped_core(
        vetoed_view, 0.03, 3, vetoed_report,
        +[](void*, const Field3D&, const Field3D&, const Field3D&,
            const Field3D&, const Field3D&, const Field3D&, const Field3D&,
            const Field3D&, const Field3D&, const Field3D&, const Field3D&,
            const Field3D&, const Field3D&,
            const tmv::terrain_native::MappedCoreStepReport&)
        { throw std::runtime_error("reject complete Metal mapped core candidate"); }),
        "reject complete Metal mapped core candidate");
    REQUIRE(public_fields(vetoed) == vetoed_before);
    REQUIRE(std::memcmp(&vetoed_report, &vetoed_report_before,
                        sizeof(vetoed_report)) == 0);
    omp_set_num_threads(previous_threads);
}

TEST_CASE("Metal bootstrap initializes repeatedly and reports physical services", "[metal][physical][bootstrap]")
{
    MetalComputeBackend backend;
    for (int generation = 0; generation < 3; ++generation)
    {
        CAPTURE(generation);
        std::string error;
        const bool  initialized = backend.initialize(error);
        CAPTURE(error);
        REQUIRE(initialized);
        REQUIRE(error.empty());
        REQUIRE(backend.available());
        REQUIRE(std::string(backend.api_name()) == "metal");
        REQUIRE_FALSE(std::string(backend.device_name()).empty());
        REQUIRE(backend.supports_homogeneous_domain_field_shift());
        REQUIRE(backend.ready(KernelId::DomainRecyclingHomogeneousFieldShift));
        REQUIRE(backend.pipeline(KernelId::DomainRecyclingHomogeneousFieldShift) != 0);
        REQUIRE(backend.ready(KernelId::Diffusion));

        HardwareInfo hardware;
        backend.populate_hardware_info(hardware);
        REQUIRE(hardware.gpu_available);
        REQUIRE(hardware.gpu_device_name == backend.device_name());

        const ComputeMemoryProperties memory = backend.memory_properties();
        REQUIRE(memory.accelerator);
        REQUIRE(memory.unified_host_device_memory);
        REQUIRE(memory.device_local_heap_bytes > 0);
        REQUIRE(memory.max_storage_buffer_bytes >= 256);

        nimbus::compute::DeviceContext&     device_service     = backend;
        nimbus::compute::BufferAllocator&   allocator_service  = backend;
        nimbus::compute::CommandSubmission& submission_service = backend;
        nimbus::compute::TimestampProvider& timestamp_service  = backend;
        REQUIRE(device_service.available());

        const auto first  = allocator_service.allocate(256);
        const auto second = allocator_service.allocate(128);
        REQUIRE(first != 0);
        REQUIRE(second != 0);
        REQUIRE(first != second);
        REQUIRE(backend.live_shared_allocation_count() == 2);
        REQUIRE(backend.live_shared_allocation_bytes() == 384);
        allocator_service.release(first);
        REQUIRE(backend.live_shared_allocation_count() == 1);
        REQUIRE(backend.live_shared_allocation_bytes() == 128);
        allocator_service.release(second);
        REQUIRE(backend.live_shared_allocation_count() == 0);

        nimbus::compute::DispatchGraph graph;
        REQUIRE(submission_service.submit(graph));
        REQUIRE(backend.last_command_buffer_completed());
        REQUIRE(backend.completed_command_buffer_count() == 1);

        graph.add(KernelId::DomainRecyclingHomogeneousFieldShift, {});
        REQUIRE_FALSE(submission_service.submit(graph));
        REQUIRE(backend.completed_command_buffer_count() == 1);
        graph.clear();
        graph.add(KernelId::Diffusion, {});
        REQUIRE_FALSE(submission_service.submit(graph));
        REQUIRE(backend.completed_command_buffer_count() == 1);

        const auto record = backend.execution_record();
        REQUIRE(record.api_name == "metal");
        REQUIRE(record.device_name == backend.device_name());
        REQUIRE_FALSE(record.operating_system.empty());
        REQUIRE_FALSE(record.host_toolchain.empty());
        REQUIRE(record.msl_source_hash.size() == 16);
        REQUIRE(record.unified_memory);
        REQUIRE(record.completed_command_buffers == 1);
        REQUIRE(record.last_device_milliseconds >= 0.0);
        REQUIRE(timestamp_service.last_device_milliseconds() >= 0.0);

        backend.shutdown();
        REQUIRE_FALSE(backend.available());
        REQUIRE(backend.completed_command_buffer_count() == 0);
        REQUIRE(backend.live_shared_allocation_count() == 0);
    }
}

TEST_CASE("Metal MYNN level-2.5 stability matches the canonical CPU closure atomically",
          "[metal][physical][mynn][parity]")
{
    MetalComputeBackend backend;
    std::string         error;
    REQUIRE(backend.initialize(error));
    REQUIRE(error.empty());
    REQUIRE(backend.supports_boundary_layer(BoundaryLayerKernelOp::MynnLevel25Stability));
    auto* kernels = backend.boundary_layer_kernels();
    REQUIRE(kernels != nullptr);

    const std::array<float, 6>          gm{{0.04f, 0.04f, 0.04f, 0.04f, 0.0f, 0.5f}};
    const std::array<float, 6>          gh{{-0.002f, 0.0f, 0.004f, -0.002f, 0.001f, -0.01f}};
    const std::array<float, 6>          length{{10.0f, 10.0f, 10.0f, 1.0f, 10.0f, 5.0f}};
    const std::array<float, 6>          q2{{0.01f, 0.01f, 0.01f, 100.0f, 0.01f, 0.05f}};
    constexpr float                     sentinel = -9876.5f;
    std::array<std::array<float, 6>, 5> output;
    for (auto& field : output)
        field.fill(sentinel);

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
    const auto                   before = backend.successful_kernel_dispatch_count(KernelId::MynnLevel25Stability);
    REQUIRE(kernels->dispatch_mynn_level25_stability(dispatch));
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::MynnLevel25Stability) == before + 1);
    REQUIRE(backend.last_command_buffer_completed());

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

    const auto preserved  = output;
    auto       invalid_gm = gm;
    invalid_gm[2]         = std::numeric_limits<float>::quiet_NaN();
    dispatch.gm_s2        = invalid_gm.data();
    REQUIRE_FALSE(kernels->dispatch_mynn_level25_stability(dispatch));
    REQUIRE(std::memcmp(output.data(), preserved.data(), sizeof(output)) == 0);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::MynnLevel25Stability) == before + 1);

    dispatch.gm_s2 = gm.data();
    dispatch.sm    = const_cast<float*>(gm.data());
    REQUIRE_FALSE(kernels->dispatch_mynn_level25_stability(dispatch));
    REQUIRE(std::memcmp(output.data(), preserved.data(), sizeof(output)) == 0);
    backend.shutdown();
}

TEST_CASE("Metal executes the composed MYNN column batch with CPU parity", "[metal][physical][mynn][batch][parity]")
{
    MetalComputeBackend backend;
    std::string         error;
    REQUIRE(backend.initialize(error));
    REQUIRE(error.empty());
    auto* kernels = backend.boundary_layer_kernels();
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
    const auto before = backend.successful_kernel_dispatch_count(KernelId::MynnLevel25Stability);
    REQUIRE(native.compute_columns_native(fixture.config,
                                          fixture.surface,
                                          fixture.views.data(),
                                          fixture.views.size(),
                                          actual.data(),
                                          actual_diagnostics.data(),
                                          *kernels));
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::MynnLevel25Stability) == before + 1);
    REQUIRE(backend.last_command_buffer_completed());

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

    backend.shutdown();
}

TEST_CASE("Metal RRTMG longwave spectral transfer matches clear and cloudy CPU columns atomically",
          "[metal][physical][rrtmg][longwave][parity]")
{
    MetalComputeBackend backend;
    std::string         error;
    REQUIRE(backend.initialize(error));
    REQUIRE(error.empty());
    REQUIRE(backend.supports_radiation(RadiationKernelOp::RrtmgLongwaveTransfer));
    auto* kernels = backend.radiation_kernels();
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
    const auto                    before = backend.successful_kernel_dispatch_count(KernelId::RrtmgLongwaveTransfer);
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
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::RrtmgLongwaveTransfer) == before + 2);
    const auto preserved_up   = upward;
    const auto preserved_down = downward;
    fixture.absorption[17]    = -0.1f;
    REQUIRE_FALSE(kernels->dispatch_rrtmg_longwave_transfer(dispatch));
    REQUIRE(std::memcmp(upward.data(), preserved_up.data(), upward.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(downward.data(), preserved_down.data(), downward.size() * sizeof(float)) == 0);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::RrtmgLongwaveTransfer) == before + 2);
    backend.shutdown();
}

TEST_CASE("Metal RRTMG shortwave spectral transfer matches clear and cloudy CPU columns atomically",
          "[metal][physical][rrtmg][shortwave][parity]")
{
    MetalComputeBackend backend;
    std::string         error;
    REQUIRE(backend.initialize(error));
    REQUIRE(error.empty());
    REQUIRE(backend.supports_radiation(RadiationKernelOp::RrtmgShortwaveTransfer));
    auto* kernels = backend.radiation_kernels();
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
    const auto                     before = backend.successful_kernel_dispatch_count(KernelId::RrtmgShortwaveTransfer);
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
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::RrtmgShortwaveTransfer) == before + 3);
    const auto preserved_up                 = upward;
    const auto preserved_down               = downward;
    dispatch.cloud_single_scattering_albedo = nullptr;
    REQUIRE_FALSE(kernels->dispatch_rrtmg_shortwave_transfer(dispatch));
    REQUIRE(std::memcmp(upward.data(), preserved_up.data(), upward.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(downward.data(), preserved_down.data(), downward.size() * sizeof(float)) == 0);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::RrtmgShortwaveTransfer) == before + 3);
    backend.shutdown();
}

TEST_CASE("Metal RRTMG batch transfers preserve column isolation and publish atomically",
          "[metal][physical][rrtmg][batch][parity]")
{
    MetalComputeBackend backend;
    std::string         error;
    REQUIRE(backend.initialize(error));
    REQUIRE(error.empty());
    auto* kernels = backend.radiation_kernels();
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
    const auto lw_before = backend.successful_kernel_dispatch_count(KernelId::RrtmgLongwaveTransferBatch);
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
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::RrtmgLongwaveTransferBatch) == lw_before + 1);
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
    const auto sw_before = backend.successful_kernel_dispatch_count(KernelId::RrtmgShortwaveTransferBatch);
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
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::RrtmgShortwaveTransferBatch) == sw_before + 1);
    const auto preserved_sw_up                                   = sw_up;
    const auto preserved_sw_down                                 = sw_down;
    shortwave.cloud_omega[static_cast<std::size_t>(layers * 14)] = 1.1f;
    REQUIRE_FALSE(kernels->dispatch_rrtmg_shortwave_transfer_batch(sw_dispatch));
    REQUIRE(std::memcmp(sw_up.data(), preserved_sw_up.data(), result_count * sizeof(float)) == 0);
    REQUIRE(std::memcmp(sw_down.data(), preserved_sw_down.data(), result_count * sizeof(float)) == 0);
    REQUIRE(backend.last_command_buffer_completed());
    backend.shutdown();
}

TEST_CASE("Dispatch graph normalizes node-local aliases and validates scheduler hazards", "[metal][dispatch_graph]")
{
    using nimbus::compute::DispatchGraph;
    using nimbus::compute::KernelId;
    using nimbus::compute::ResourceAccess;

    DispatchGraph graph;
    const auto    producer = graph.add(
        KernelId::LinPointwise,
        {{"dqr_dt", ResourceAccess::Write}, {"dqr_dt", ResourceAccess::Read}, {"theta", ResourceAccess::Read}});
    const auto independent = graph.add(KernelId::Diffusion, {{"unrelated", ResourceAccess::Read}});
    const auto consumer    = graph.add(KernelId::LinSedimentation, {{"dqr_dt", ResourceAccess::ReadWrite}});

    REQUIRE(graph.valid());
    REQUIRE(graph.nodes()[producer].fields.size() == 2);
    REQUIRE(graph.nodes()[producer].fields[0].access == ResourceAccess::ReadWrite);
    REQUIRE(graph.nodes()[producer].dependencies.empty());
    REQUIRE_FALSE(graph.depends_on(independent, producer));
    REQUIRE(graph.depends_on(consumer, producer));
    REQUIRE_FALSE(graph.requires_barrier(producer, independent));
    REQUIRE(graph.requires_barrier(producer, consumer));

    DispatchGraph invalid;
    invalid.add(KernelId::Diffusion, {{"", ResourceAccess::Read}});
    REQUIRE_FALSE(invalid.valid());
}

TEST_CASE("Metal runtime MSL source has a stable independently checked hash", "[metal][msl][abi]")
{
    const char* source = MetalComputeBackend::domain_shift_msl_source();
    REQUIRE(source != nullptr);
    REQUIRE(std::strstr(source, "kernel void domain_recycling_homogeneous_field_shift") != nullptr);
    REQUIRE(std::strstr(source, "leading_profile_offset") != nullptr);
    REQUIRE(fnv1a64_hex(source) == MetalComputeBackend::domain_shift_msl_source_hash());
    const char* chaos_source = MetalComputeBackend::earth_control_evolution_msl_source();
    REQUIRE(chaos_source != nullptr);
    REQUIRE(std::strstr(chaos_source, "bits >> 11") != nullptr);
    REQUIRE(std::strstr(chaos_source, "9007199254740992.0") != nullptr);
    REQUIRE(fnv1a64_hex(chaos_source) == MetalComputeBackend::earth_control_evolution_msl_source_hash());
}

TEST_CASE("Metal Earth-control evolution preserves Gaussian and AR reconstruction contracts atomically",
          "[metal][physical][chaos][parity][restart][fail_closed]")
{
    constexpr std::uint64_t seed           = 91234567ULL;
    constexpr int           member         = -7;
    constexpr std::uint64_t initial_key    = 4001ULL;
    constexpr std::uint64_t innovation_key = 4101ULL;
    constexpr double        rho_value      = 0.73;

    tmv::chaos::EarthCorrelatedNoise cpu;
    cpu.configure(seed, member, initial_key, innovation_key, 900.0, 700.0, 200.0, 250.0);
    cpu.initialize_domain(27, 23, 5);
    const Field3D&    cpu_initial = cpu.current_controls();
    const int         ni          = cpu_initial.size_r();
    const int         nj          = cpu_initial.size_th();
    const int         nk          = cpu_initial.size_z();
    const std::size_t count       = cpu_initial.size();

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.supports_earth_control_evolution());

    std::vector<float>            zero(count, 0.0f);
    std::vector<float>            native_initial(count, -91.0f);
    EarthControlEvolutionDispatch dispatch;
    dispatch.previous               = zero.data();
    dispatch.destination            = native_initial.data();
    dispatch.seed                   = seed;
    dispatch.member_id              = member;
    dispatch.initial_process_key    = initial_key;
    dispatch.innovation_process_key = innovation_key;
    dispatch.control_i_min          = cpu.control_i_min();
    dispatch.control_j_min          = cpu.control_j_min();
    dispatch.control_i_count        = ni;
    dispatch.control_j_count        = nj;
    dispatch.nz                     = nk;
    dispatch.mode                   = EarthControlEvolutionMode::Initialize;
    REQUIRE(backend.dispatch_earth_control_evolution(dispatch));

    double max_initial_error = 0.0;
    for (std::size_t index = 0; index < count; ++index)
        max_initial_error = std::max(
            max_initial_error,
            std::abs(static_cast<double>(native_initial[index]) - static_cast<double>(cpu_initial.data()[index])));
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
        REQUIRE(backend.dispatch_earth_control_evolution(near_unit));
        const double scale = std::sqrt(1.0 - correlation * correlation);
        for (std::size_t index = 0; index < count; ++index)
        {
            const double expected = scale * native_initial[index];
            CHECK(std::abs(double(evolved[index]) - expected) <=
                  8 * std::numeric_limits<float>::epsilon() * std::abs(expected));
        }
        near_unit.mode = EarthControlEvolutionMode::Reconstruct;
        near_unit.innovation_epoch = 1;
        REQUIRE(backend.dispatch_earth_control_evolution(near_unit));
        for (std::size_t index = 0; index < count; ++index)
        {
            const float expected = float((correlation + scale) * native_initial[index]);
            const double quantum = std::abs(double(std::nextafter(expected, 0.0f)) - expected);
            CHECK(std::abs(double(evolved[index]) - expected) <= 2 * quantum);
        }
        near_unit.mode = EarthControlEvolutionMode::Advance;
        near_unit.rho = float(correlation);
        REQUIRE(backend.dispatch_earth_control_evolution(near_unit));
        CHECK(std::all_of(evolved.begin(), evolved.end(), [](float value)
        {
            return value == 0.0f;
        }));
        const auto retained = evolved;
        for (double invalid : {std::nextafter(1.0, 2.0), std::nextafter(0.0, -1.0)})
        {
            near_unit.rho = invalid;
            CHECK_FALSE(backend.dispatch_earth_control_evolution(near_unit));
            CHECK(evolved == retained);
        }
    }

    for (std::uint64_t epoch = 0; epoch < 4; ++epoch)
        cpu.advance(rho_value, epoch);
    std::vector<float> native_reconstructed(count, -92.0f);
    dispatch.destination      = native_reconstructed.data();
    dispatch.innovation_epoch = 4;
    dispatch.rho              = rho_value;
    dispatch.mode             = EarthControlEvolutionMode::Reconstruct;
    REQUIRE(backend.dispatch_earth_control_evolution(dispatch));

    double max_reconstruction_error = 0.0;
    for (std::size_t index = 0; index < count; ++index)
        max_reconstruction_error = std::max(max_reconstruction_error,
                                            std::abs(static_cast<double>(native_reconstructed[index]) -
                                                     static_cast<double>(cpu.current_controls().data()[index])));
    REQUIRE(max_reconstruction_error <= 8.0e-5);

    constexpr int      sample_count = 262144;
    std::vector<float> samples(sample_count, 0.0f);
    std::vector<float> sample_zero(sample_count, 0.0f);
    dispatch.previous         = sample_zero.data();
    dispatch.destination      = samples.data();
    dispatch.control_i_min    = -19;
    dispatch.control_j_min    = 23;
    dispatch.control_i_count  = 1;
    dispatch.control_j_count  = 1;
    dispatch.nz               = sample_count;
    dispatch.innovation_epoch = 0;
    dispatch.rho              = 0.0f;
    dispatch.mode             = EarthControlEvolutionMode::Initialize;
    REQUIRE(backend.dispatch_earth_control_evolution(dispatch));
    double      sum        = 0.0;
    double      sum_sq     = 0.0;
    double      max_abs    = 0.0;
    std::size_t four_sigma = 0;
    for (float value : samples)
    {
        sum += value;
        sum_sq += static_cast<double>(value) * value;
        max_abs = std::max(max_abs, std::abs(static_cast<double>(value)));
        four_sigma += std::abs(value) > 4.0f ? 1U : 0U;
    }
    const double mean     = sum / sample_count;
    const double variance = sum_sq / sample_count - mean * mean;
    REQUIRE(std::abs(mean) < 0.01);
    REQUIRE(std::abs(variance - 1.0) < 0.02);
    REQUIRE(four_sigma >= 5);
    REQUIRE(four_sigma <= 40);
    REQUIRE(max_abs > 4.2);
    const double minimum_open_unit = 0.5 / 9007199254740992.0;
    REQUIRE(std::sqrt(-2.0 * std::log(minimum_open_unit)) > 8.5);

    const std::vector<float> committed = native_reconstructed;
    dispatch.destination               = native_reconstructed.data();
    dispatch.control_i_count           = 0;
    REQUIRE_FALSE(backend.dispatch_earth_control_evolution(dispatch));
    REQUIRE(native_reconstructed == committed);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::EarthControlEvolution) == 12);

    std::cout << "[metal-chaos-evidence] controls=" << count << " max_initial_abs=" << max_initial_error
              << " max_reconstruct_abs=" << max_reconstruction_error << " mean=" << mean << " variance=" << variance
              << " abs_gt_4=" << four_sigma << " max_abs=" << max_abs << " dispatches="
              << backend.successful_kernel_dispatch_count(KernelId::EarthControlEvolution) << '\n';
}

TEST_CASE("Metal manifest covers supported kernels and records native precision limits",
          "[metal][physical][manifest][qualification][fail_closed]")
{
    const char* manifest_error = nullptr;
    REQUIRE(nimbus::compute::validate_kernel_manifest(nimbus::compute::kernel_manifest(), &manifest_error));
    REQUIRE(manifest_error == nullptr);

    MetalComputeBackend backend;
    initialize_backend(backend);

    std::uint64_t active   = 0;
    std::uint64_t deferred = 0;
    std::uint64_t orphan   = 0;
    std::uint64_t hardware_limited = 0;
    for (const auto& entry : nimbus::compute::kernel_manifest())
    {
        CAPTURE(entry.operation_id, entry.kernel_name, static_cast<unsigned>(entry.id));
        REQUIRE(entry.artifact_hash != nullptr);
        REQUIRE(std::strlen(entry.artifact_hash) == 16);
        const char* metal_hash = MetalComputeBackend::metal_artifact_source_hash(entry.id);
        if (entry.availability == nimbus::compute::KernelAvailability::Active)
        {
            ++active;
            if (entry.precision == nimbus::compute::KernelPrecision::Float32StorageFloat64Arithmetic)
            {
                ++hardware_limited;
                REQUIRE_FALSE(backend.ready(entry.id));
                REQUIRE(backend.pipeline(entry.id) == 0);
                REQUIRE(metal_hash == nullptr);
                continue;
            }
            REQUIRE(backend.ready(entry.id));
            REQUIRE(backend.pipeline(entry.id) != 0);
            REQUIRE(metal_hash != nullptr);
            REQUIRE(std::strlen(metal_hash) == 16);
        }
        else
        {
            if (entry.availability == nimbus::compute::KernelAvailability::Deferred)
                ++deferred;
            else
                ++orphan;
            REQUIRE_FALSE(backend.ready(entry.id));
            REQUIRE(backend.pipeline(entry.id) == 0);
            REQUIRE(metal_hash == nullptr);
        }
    }

    const auto coverage = backend.manifest_coverage();
    REQUIRE(coverage.canonical_manifest_hash == canonical_manifest_fingerprint());
    REQUIRE(coverage.canonical_manifest_hash == MetalComputeBackend::canonical_kernel_manifest_hash());
    REQUIRE(coverage.manifest_kernels == nimbus::compute::kernel_count());
    REQUIRE(coverage.active_kernels == active);
    REQUIRE(coverage.ready_active_kernels == active - hardware_limited);
    REQUIRE(coverage.hardware_limited_kernels == hardware_limited);
    REQUIRE(hardware_limited == 1);
    REQUIRE(coverage.deferred_kernels == deferred);
    REQUIRE(coverage.orphan_kernels == orphan);
    REQUIRE(coverage.unexpected_available_kernels == 0);
    REQUIRE(coverage.missing_artifact_hashes == 0);
    REQUIRE(coverage.complete);

    const auto record = backend.execution_record();
    REQUIRE(record.canonical_kernel_manifest_hash == coverage.canonical_manifest_hash);
    REQUIRE(record.manifest_kernels == coverage.manifest_kernels);
    REQUIRE(record.active_manifest_kernels == coverage.active_kernels);
    REQUIRE(record.ready_active_manifest_kernels == coverage.ready_active_kernels);
    REQUIRE(record.hardware_limited_manifest_kernels == coverage.hardware_limited_kernels);
    REQUIRE(record.deferred_manifest_kernels == coverage.deferred_kernels);
    REQUIRE(record.orphan_manifest_kernels == coverage.orphan_kernels);
    REQUIRE(record.unexpected_available_manifest_kernels == 0);
    REQUIRE(record.missing_active_artifact_hashes == 0);
    REQUIRE(record.manifest_complete);

    std::cout << "[metal-manifest-evidence] manifest=" << coverage.manifest_kernels
              << " active=" << coverage.active_kernels << " ready=" << coverage.ready_active_kernels
              << " hardware_limited=" << coverage.hardware_limited_kernels
              << " deferred=" << coverage.deferred_kernels << " orphan=" << coverage.orphan_kernels
              << " canonical_fnv1a64=" << coverage.canonical_manifest_hash << '\n';
}

TEST_CASE("Metal shared and private-heap storage modes preserve bitwise domain-shift results",
          "[metal][physical][storage][heap][benchmark][gpu_parity][fail_closed]")
{
    nimbus::metal::MetalStorageBenchmarkConfig config;
    config.nr         = 192;
    config.nth        = 160;
    config.nz         = 24;
    config.iterations = 12;
    config.warmups    = 2;
    config.samples    = 7;

    MetalComputeBackend                        backend;
    nimbus::metal::MetalStorageBenchmarkResult result;
    result.field_bytes = 99;
    REQUIRE_FALSE(backend.run_domain_shift_storage_benchmark(config, result));
    REQUIRE(result.field_bytes == 0);

    initialize_backend(backend);
    REQUIRE(backend.run_domain_shift_storage_benchmark(config, result));
    const std::size_t cells = static_cast<std::size_t>(config.nr) * config.nth * config.nz;
    REQUIRE(result.field_bytes == cells * sizeof(float));
    REQUIRE(result.private_heap_bytes >= 2 * result.field_bytes + static_cast<std::size_t>(config.nz) * sizeof(float));
    REQUIRE(result.dispatches_per_sample == static_cast<std::uint64_t>(config.iterations));
    REQUIRE(result.threadgroup_width == 64);
    REQUIRE(result.shared_bitwise_correct);
    REQUIRE(result.private_bitwise_correct);
    REQUIRE(result.shared.gpu_upload_bytes == 0);
    REQUIRE(result.shared.gpu_download_bytes == 0);
    REQUIRE(result.private_heap.gpu_upload_bytes ==
            result.field_bytes + static_cast<std::size_t>(config.nz) * sizeof(float));
    REQUIRE(result.private_heap.gpu_download_bytes == result.field_bytes);
    REQUIRE(result.shared.command_buffers == static_cast<std::uint64_t>(config.samples));
    REQUIRE(result.private_heap.command_buffers == static_cast<std::uint64_t>(config.samples));
    REQUIRE(result.shared.median_device_milliseconds > 0.0);
    REQUIRE(result.shared.p95_device_milliseconds >= result.shared.median_device_milliseconds);
    REQUIRE(result.shared.median_wall_milliseconds > 0.0);
    REQUIRE(result.shared.p95_wall_milliseconds >= result.shared.median_wall_milliseconds);
    REQUIRE(result.private_heap.median_device_milliseconds > 0.0);
    REQUIRE(result.private_heap.p95_device_milliseconds >= result.private_heap.median_device_milliseconds);
    REQUIRE(result.private_heap.median_wall_milliseconds > 0.0);
    REQUIRE(result.private_heap.p95_wall_milliseconds >= result.private_heap.median_wall_milliseconds);
    REQUIRE(backend.completed_command_buffer_count() == 0);

    const auto valid_result = result;
    config.samples          = 2;
    REQUIRE_FALSE(backend.run_domain_shift_storage_benchmark(config, result));
    REQUIRE(result.field_bytes == 0);
    REQUIRE(backend.completed_command_buffer_count() == 0);
    config.samples    = 7;
    config.iterations = 0;
    REQUIRE_FALSE(backend.run_domain_shift_storage_benchmark(config, result));
    REQUIRE(result.field_bytes == 0);

    std::cout << "[metal-storage-evidence] field_bytes=" << valid_result.field_bytes
              << " private_heap_bytes=" << valid_result.private_heap_bytes
              << " iterations=" << valid_result.dispatches_per_sample
              << " samples=" << valid_result.shared.command_buffers << " shared_bitwise=" << std::boolalpha
              << valid_result.shared_bitwise_correct << " private_bitwise=" << valid_result.private_bitwise_correct
              << " shared_device_median_ms=" << valid_result.shared.median_device_milliseconds
              << " shared_device_p95_ms=" << valid_result.shared.p95_device_milliseconds
              << " private_device_median_ms=" << valid_result.private_heap.median_device_milliseconds
              << " private_device_p95_ms=" << valid_result.private_heap.p95_device_milliseconds
              << " private_upload_bytes=" << valid_result.private_heap.gpu_upload_bytes
              << " private_download_bytes=" << valid_result.private_heap.gpu_download_bytes
              << " winner=unclaimed msl_fnv1a64=" << MetalComputeBackend::domain_shift_msl_source_hash() << '\n';
}

TEST_CASE("Metal workgroup-width matrix preserves storage-mode parity without selecting a winner",
          "[metal][physical][workgroup][storage][benchmark][gpu_parity][fail_closed]")
{
    MetalComputeBackend backend;
    initialize_backend(backend);

    nimbus::metal::MetalStorageBenchmarkConfig config;
    config.nr         = 128;
    config.nth        = 96;
    config.nz         = 16;
    config.iterations = 8;
    config.warmups    = 1;
    config.samples    = 7;
    const std::array<int, 3>                                  widths{{32, 64, 128}};
    std::array<nimbus::metal::MetalStorageBenchmarkResult, 3> results;
    for (std::size_t index = 0; index < widths.size(); ++index)
    {
        CAPTURE(widths[index]);
        config.threadgroup_width = widths[index];
        REQUIRE(backend.run_domain_shift_storage_benchmark(config, results[index]));
        REQUIRE(results[index].threadgroup_width == static_cast<std::uint64_t>(widths[index]));
        REQUIRE(results[index].shared_bitwise_correct);
        REQUIRE(results[index].private_bitwise_correct);
        REQUIRE(results[index].shared.command_buffers == 7);
        REQUIRE(results[index].private_heap.command_buffers == 7);
        REQUIRE(results[index].shared.median_device_milliseconds > 0.0);
        REQUIRE(results[index].shared.p95_device_milliseconds >= results[index].shared.median_device_milliseconds);
        REQUIRE(results[index].private_heap.median_device_milliseconds > 0.0);
        REQUIRE(results[index].private_heap.p95_device_milliseconds >=
                results[index].private_heap.median_device_milliseconds);
        if (index > 0)
        {
            REQUIRE(results[index].field_bytes == results[0].field_bytes);
            REQUIRE(results[index].private_heap_bytes == results[0].private_heap_bytes);
        }
    }

    nimbus::metal::MetalStorageBenchmarkResult invalid;
    invalid.field_bytes      = 7;
    config.threadgroup_width = 0;
    REQUIRE_FALSE(backend.run_domain_shift_storage_benchmark(config, invalid));
    REQUIRE(invalid.field_bytes == 0);
    config.threadgroup_width = 1025;
    REQUIRE_FALSE(backend.run_domain_shift_storage_benchmark(config, invalid));
    REQUIRE(invalid.field_bytes == 0);
    REQUIRE(backend.completed_command_buffer_count() == 0);

    std::cout << "[metal-workgroup-evidence] widths=32,64,128 field_bytes=" << results[0].field_bytes
              << " iterations=" << results[0].dispatches_per_sample << " samples=" << results[0].shared.command_buffers
              << " all_bitwise=true"
              << " w32_shared_median_ms=" << results[0].shared.median_device_milliseconds
              << " w32_shared_p95_ms=" << results[0].shared.p95_device_milliseconds
              << " w32_private_median_ms=" << results[0].private_heap.median_device_milliseconds
              << " w32_private_p95_ms=" << results[0].private_heap.p95_device_milliseconds
              << " w64_shared_median_ms=" << results[1].shared.median_device_milliseconds
              << " w64_shared_p95_ms=" << results[1].shared.p95_device_milliseconds
              << " w64_private_median_ms=" << results[1].private_heap.median_device_milliseconds
              << " w64_private_p95_ms=" << results[1].private_heap.p95_device_milliseconds
              << " w128_shared_median_ms=" << results[2].shared.median_device_milliseconds
              << " w128_shared_p95_ms=" << results[2].shared.p95_device_milliseconds
              << " w128_private_median_ms=" << results[2].private_heap.median_device_milliseconds
              << " w128_private_p95_ms=" << results[2].private_heap.p95_device_milliseconds
              << " winner=unclaimed msl_fnv1a64=" << MetalComputeBackend::domain_shift_msl_source_hash() << '\n';
}

TEST_CASE("Metal submission-depth matrix preserves parity and reports per-dispatch cost",
          "[metal][physical][submission][batch][storage][benchmark][gpu_parity][fail_closed]")
{
    MetalComputeBackend backend;
    initialize_backend(backend);

    nimbus::metal::MetalStorageBenchmarkConfig config;
    config.nr                = 96;
    config.nth               = 80;
    config.nz                = 16;
    config.warmups           = 1;
    config.samples           = 7;
    config.threadgroup_width = 64;
    const std::array<int, 4>                                  depths{{1, 4, 16, 64}};
    std::array<nimbus::metal::MetalStorageBenchmarkResult, 4> results;
    std::array<double, 4>                                     shared_median_per_dispatch{};
    std::array<double, 4>                                     shared_p95_per_dispatch{};
    std::array<double, 4>                                     private_median_per_dispatch{};
    std::array<double, 4>                                     private_p95_per_dispatch{};
    for (std::size_t index = 0; index < depths.size(); ++index)
    {
        CAPTURE(depths[index]);
        config.iterations = depths[index];
        REQUIRE(backend.run_domain_shift_storage_benchmark(config, results[index]));
        REQUIRE(results[index].dispatches_per_sample == static_cast<std::uint64_t>(depths[index]));
        REQUIRE(results[index].threadgroup_width == 64);
        REQUIRE(results[index].shared_bitwise_correct);
        REQUIRE(results[index].private_bitwise_correct);
        REQUIRE(results[index].shared.command_buffers == 7);
        REQUIRE(results[index].private_heap.command_buffers == 7);
        const double denominator           = static_cast<double>(depths[index]);
        shared_median_per_dispatch[index]  = results[index].shared.median_device_milliseconds / denominator;
        shared_p95_per_dispatch[index]     = results[index].shared.p95_device_milliseconds / denominator;
        private_median_per_dispatch[index] = results[index].private_heap.median_device_milliseconds / denominator;
        private_p95_per_dispatch[index]    = results[index].private_heap.p95_device_milliseconds / denominator;
        REQUIRE(shared_median_per_dispatch[index] > 0.0);
        REQUIRE(shared_p95_per_dispatch[index] >= shared_median_per_dispatch[index]);
        REQUIRE(private_median_per_dispatch[index] > 0.0);
        REQUIRE(private_p95_per_dispatch[index] >= private_median_per_dispatch[index]);
        if (index > 0)
        {
            REQUIRE(results[index].field_bytes == results[0].field_bytes);
            REQUIRE(results[index].private_heap_bytes == results[0].private_heap_bytes);
        }
    }

    nimbus::metal::MetalStorageBenchmarkResult invalid;
    invalid.field_bytes = 11;
    config.iterations   = 10001;
    REQUIRE_FALSE(backend.run_domain_shift_storage_benchmark(config, invalid));
    REQUIRE(invalid.field_bytes == 0);
    REQUIRE(backend.completed_command_buffer_count() == 0);

    std::cout << "[metal-submission-evidence] depths=1,4,16,64 field_bytes=" << results[0].field_bytes
              << " samples=" << results[0].shared.command_buffers << " width=64 all_bitwise=true"
              << " d1_shared_median_per_dispatch_ms=" << shared_median_per_dispatch[0]
              << " d1_shared_p95_per_dispatch_ms=" << shared_p95_per_dispatch[0]
              << " d1_private_median_per_dispatch_ms=" << private_median_per_dispatch[0]
              << " d1_private_p95_per_dispatch_ms=" << private_p95_per_dispatch[0]
              << " d4_shared_median_per_dispatch_ms=" << shared_median_per_dispatch[1]
              << " d4_shared_p95_per_dispatch_ms=" << shared_p95_per_dispatch[1]
              << " d4_private_median_per_dispatch_ms=" << private_median_per_dispatch[1]
              << " d4_private_p95_per_dispatch_ms=" << private_p95_per_dispatch[1]
              << " d16_shared_median_per_dispatch_ms=" << shared_median_per_dispatch[2]
              << " d16_shared_p95_per_dispatch_ms=" << shared_p95_per_dispatch[2]
              << " d16_private_median_per_dispatch_ms=" << private_median_per_dispatch[2]
              << " d16_private_p95_per_dispatch_ms=" << private_p95_per_dispatch[2]
              << " d64_shared_median_per_dispatch_ms=" << shared_median_per_dispatch[3]
              << " d64_shared_p95_per_dispatch_ms=" << shared_p95_per_dispatch[3]
              << " d64_private_median_per_dispatch_ms=" << private_median_per_dispatch[3]
              << " d64_private_p95_per_dispatch_ms=" << private_p95_per_dispatch[3]
              << " winner=unclaimed msl_fnv1a64=" << MetalComputeBackend::domain_shift_msl_source_hash() << '\n';
}

TEST_CASE("Metal Smagorinsky coefficients match the canonical CPU operator",
          "[metal][physical][gpu_parity][turbulence][smagorinsky]")
{
    constexpr int     nx       = 9;
    constexpr int     ny       = 8;
    constexpr int     nz       = 6;
    constexpr float   dx       = 120.0f;
    constexpr float   dy       = 150.0f;
    constexpr float   Cs       = 0.18f;
    constexpr float   Pr_t     = 1.0f / 3.0f;
    constexpr float   Sc_t     = 0.4f;
    constexpr float   nu_t_max = 1000.0f;
    constexpr float   K_max    = 1000.0f;
    const std::size_t cells    = static_cast<std::size_t>(nx) * ny * nz;

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.supports_smagorinsky_coefficients());
    REQUIRE(backend.ready(KernelId::SmagorinskyCoefficients));
    REQUIRE(backend.pipeline(KernelId::SmagorinskyCoefficients) != 0);
    REQUIRE(backend.turbulence_kernels() == &backend);
    REQUIRE(backend.supports_turbulence(TurbulenceKernelOp::Smagorinsky));
    REQUIRE(backend.supports_turbulence(TurbulenceKernelOp::SgsApply));

    Field3D u(nx, ny, nz, 0.0f);
    Field3D v(nx, ny, nz, 0.0f);
    Field3D w(nx, ny, nz, 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                u[i][j][k] = 0.11f * i + 0.037f * j + 0.053f * k + 0.002f * i * k;
                v[i][j][k] = -0.043f * i + 0.081f * j + 0.019f * k + 0.001f * j * k;
                w[i][j][k] = 0.014f * i - 0.023f * j + 0.047f * (k + 1) + 0.0015f * i * j;
            }

    std::vector<float> dz_profile(nz);
    std::vector<float> dzc_profile(nz);
    tmv::GridMetrics   grid;
    grid.dx = dx;
    grid.dy = dy;
    grid.dz.resize(nz);
    for (int k = 0; k < nz; ++k)
    {
        dz_profile[static_cast<std::size_t>(k)]  = 80.0f + 7.0f * k;
        dzc_profile[static_cast<std::size_t>(k)] = 2.0f * dz_profile[static_cast<std::size_t>(k)];
        grid.dz[static_cast<std::size_t>(k)]     = dz_profile[static_cast<std::size_t>(k)];
    }
    tmv::TurbulenceStateView state;
    state.u   = &u;
    state.v   = &v;
    state.w   = &w;
    state.NR  = nx;
    state.NTH = ny;
    state.NZ  = nz;

    std::vector<float>              actual_nu(cells, -777.0f);
    std::vector<float>              actual_theta(cells, -888.0f);
    std::vector<float>              actual_q(cells, -999.0f);
    SmagorinskyCoefficientsDispatch dispatch{};
    dispatch.u        = u.data();
    dispatch.v        = v.data();
    dispatch.w        = w.data();
    dispatch.dz_prof  = dz_profile.data();
    dispatch.dzc_prof = dzc_profile.data();
    dispatch.prof_len = nz;
    dispatch.nu_t     = actual_nu.data();
    dispatch.K_theta  = actual_theta.data();
    dispatch.K_q      = actual_q.data();
    dispatch.nx       = nx;
    dispatch.ny       = ny;
    dispatch.nz       = nz;
    dispatch.dx       = dx;
    dispatch.dy       = dy;
    dispatch.Cs       = Cs;
    dispatch.Pr_t     = Pr_t;
    dispatch.Sc_t     = Sc_t;
    dispatch.nu_t_max = nu_t_max;
    dispatch.K_max    = K_max;

    struct CoefficientCase
    {
        const char* mode;
        bool        horizontal;
        bool        vertical;
        bool        periodic;
        int         filter_mode;
    };
    const std::array<CoefficientCase, 4> cases{{
        {"horizontal_only", true, false, false, 0},
        {"vertical_only", false, true, false, 0},
        {"3d", true, true, false, 0},
        {"3d", true, true, true, 1},
    }};

    double maximum_nu_error          = 0.0;
    double maximum_diffusivity_error = 0.0;
    for (const auto& coefficient_case : cases)
    {
        CAPTURE(coefficient_case.mode, coefficient_case.periodic, coefficient_case.filter_mode);
        test_lateral_bc_west =
            coefficient_case.periodic ? tmv::init::config::LateralBC::Periodic : tmv::init::config::LateralBC::Rigid;
        test_lateral_bc_east    = test_lateral_bc_west;
        test_lateral_bc_south   = test_lateral_bc_west;
        test_lateral_bc_north   = test_lateral_bc_west;
        dispatch.use_horizontal = coefficient_case.horizontal;
        dispatch.use_vertical   = coefficient_case.vertical;
        dispatch.periodic_x     = coefficient_case.periodic;
        dispatch.periodic_y     = coefficient_case.periodic;
        state.periodic_x = coefficient_case.periodic;
        state.periodic_y = coefficient_case.periodic;
        dispatch.filter_mode    = coefficient_case.filter_mode;

        std::vector<float> expected_nu(cells);
        std::vector<float> expected_theta(cells);
        std::vector<float> expected_q(cells);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const std::size_t index     = (static_cast<std::size_t>(i) * ny + j) * nz + k;
                    const double      strain_sq = tmv::eddy_viscosity::compute_directional_strain_sq_centered(
                        state, grid, i, j, k, coefficient_case.mode);
                    const double filter_width =
                        coefficient_case.filter_mode == 1
                            ? std::sqrt(static_cast<double>(dx) * dy)
                            : std::cbrt(static_cast<double>(dx) * dy * grid.dz[static_cast<std::size_t>(k)]);
                    double viscosity =
                        tmv::eddy_viscosity::compute_smagorinsky_viscosity(Cs, filter_width, std::sqrt(strain_sq));
                    viscosity                   = std::min(viscosity, static_cast<double>(nu_t_max));
                    double theta_diffusivity    = 0.0;
                    double moisture_diffusivity = 0.0;
                    double tke_diffusivity      = 0.0;
                    tmv::eddy_viscosity::compute_eddy_diffusivities(
                        viscosity, Pr_t, Sc_t, theta_diffusivity, moisture_diffusivity, tke_diffusivity);
                    expected_nu[index]    = static_cast<float>(viscosity);
                    expected_theta[index] = static_cast<float>(std::min(theta_diffusivity, static_cast<double>(K_max)));
                    expected_q[index] = static_cast<float>(std::min(moisture_diffusivity, static_cast<double>(K_max)));
                }

        REQUIRE(backend.dispatch_smagorinsky_coefficients(dispatch));
        for (std::size_t index = 0; index < cells; ++index)
        {
            REQUIRE(std::isfinite(actual_nu[index]));
            REQUIRE(std::isfinite(actual_theta[index]));
            REQUIRE(std::isfinite(actual_q[index]));
            maximum_nu_error =
                std::max(maximum_nu_error, std::abs(static_cast<double>(actual_nu[index]) - expected_nu[index]));
            maximum_diffusivity_error = std::max(
                maximum_diffusivity_error, std::abs(static_cast<double>(actual_theta[index]) - expected_theta[index]));
            maximum_diffusivity_error =
                std::max(maximum_diffusivity_error, std::abs(static_cast<double>(actual_q[index]) - expected_q[index]));
        }
    }
    REQUIRE(maximum_nu_error < 2.0e-5);
    REQUIRE(maximum_diffusivity_error < 8.0e-5);
    REQUIRE(backend.successful_dispatch_count() == cases.size());
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::SmagorinskyCoefficients) == cases.size());

    const auto                      nu_before_rejection    = actual_nu;
    const auto                      theta_before_rejection = actual_theta;
    const auto                      q_before_rejection     = actual_q;
    SmagorinskyCoefficientsDispatch invalid                = dispatch;
    invalid.Pr_t                                           = 0.0f;
    REQUIRE_FALSE(backend.dispatch_smagorinsky_coefficients(invalid));
    invalid         = dispatch;
    invalid.K_theta = invalid.nu_t;
    REQUIRE_FALSE(backend.dispatch_smagorinsky_coefficients(invalid));
    std::vector<float> invalid_dz = dz_profile;
    invalid_dz[2]                 = -1.0f;
    invalid                       = dispatch;
    invalid.dz_prof               = invalid_dz.data();
    REQUIRE_FALSE(backend.dispatch_smagorinsky_coefficients(invalid));
    REQUIRE(std::memcmp(actual_nu.data(), nu_before_rejection.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(actual_theta.data(), theta_before_rejection.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(actual_q.data(), q_before_rejection.data(), cells * sizeof(float)) == 0);
    REQUIRE(backend.successful_dispatch_count() == cases.size());
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::SmagorinskyCoefficients) == cases.size());

    const auto record = backend.execution_record();
    REQUIRE(record.completed_smagorinsky_coefficients_dispatches == cases.size());
    REQUIRE(record.smagorinsky_coefficients_msl_source_hash.size() == 16);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::smagorinsky_coefficients_msl_source()) ==
            record.smagorinsky_coefficients_msl_source_hash);
    std::cout << "[metal-smagorinsky-evidence] cases=" << cases.size() << " max_nu_abs=" << maximum_nu_error
              << " max_diffusivity_abs=" << maximum_diffusivity_error
              << " msl_fnv1a64=" << record.smagorinsky_coefficients_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';

    test_lateral_bc_west  = tmv::init::config::LateralBC::Rigid;
    test_lateral_bc_east  = tmv::init::config::LateralBC::Rigid;
    test_lateral_bc_south = tmv::init::config::LateralBC::Rigid;
    test_lateral_bc_north = tmv::init::config::LateralBC::Rigid;
}

TEST_CASE("Metal conservative Smagorinsky tendencies match the live-density CPU operator",
          "[metal][physical][gpu_parity][turbulence][smagorinsky][tendencies]")
{
    constexpr int     nx    = 9;
    constexpr int     ny    = 8;
    constexpr int     nz    = 6;
    constexpr float   dx    = 120.0f;
    constexpr float   dy    = 150.0f;
    const std::size_t cells = static_cast<std::size_t>(nx) * ny * nz;

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.supports_smagorinsky_tendencies());
    REQUIRE(backend.ready(KernelId::SmagorinskyTendencies));
    REQUIRE(backend.pipeline(KernelId::SmagorinskyTendencies) != 0);
    REQUIRE(backend.turbulence_kernels() == &backend);
    REQUIRE(backend.supports_turbulence(TurbulenceKernelOp::Smagorinsky));
    REQUIRE(backend.supports_turbulence(TurbulenceKernelOp::SgsApply));

    Field3D u(nx, ny, nz, 0.0f);
    Field3D v(nx, ny, nz, 0.0f);
    Field3D w(nx, ny, nz, 0.0f);
    Field3D theta(nx, ny, nz, 0.0f);
    Field3D qv(nx, ny, nz, 0.0f);
    Field3D rho(nx, ny, nz, 0.0f);
    Field3D nu(nx, ny, nz, 0.0f);
    Field3D kth(nx, ny, nz, 0.0f);
    Field3D kq(nx, ny, nz, 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const float wave = std::sin(0.31f * i + 0.23f * j + 0.41f * k);
                u[i][j][k]       = 2.0f + 0.17f * i - 0.08f * j + 0.11f * k + 0.37f * wave;
                v[i][j][k]       = -1.0f - 0.09f * i + 0.13f * j + 0.07f * k - 0.29f * wave;
                w[i][j][k]       = 0.03f * i - 0.04f * j + 0.12f * (k + 1) + 0.16f * wave;
                theta[i][j][k]   = 298.0f + 0.21f * i - 0.14f * j + 0.73f * k + 0.8f * wave;
                qv[i][j][k]      = 0.009f + 0.00013f * i + 0.00007f * j - 0.00019f * k + 0.0003f * wave;
                // Independent horizontal and vertical structure makes any
                // rho0/unity substitution observably wrong.
                rho[i][j][k] = 0.78f + 0.019f * i + 0.012f * j + 0.027f * k + 0.015f * wave;
                nu[i][j][k]  = 2.0f + 0.11f * i + 0.07f * j + 0.16f * k + 0.08f * wave;
                kth[i][j][k] = 3.0f + 0.09f * i + 0.13f * j + 0.12f * k + 0.06f * wave;
                kq[i][j][k]  = 4.0f + 0.08f * i + 0.05f * j + 0.14f * k + 0.04f * wave;
            }

    std::vector<float>  dz_profile(nz);
    std::vector<double> rho0(nz);
    tmv::GridMetrics    grid;
    grid.dx = dx;
    grid.dy = dy;
    grid.dz.resize(nz);
    for (int k = 0; k < nz; ++k)
    {
        dz_profile[static_cast<std::size_t>(k)] = 75.0f + 9.0f * k;
        grid.dz[static_cast<std::size_t>(k)]    = dz_profile[static_cast<std::size_t>(k)];
        rho0[static_cast<std::size_t>(k)]       = 1.0 - 0.035 * k;
    }

    tmv::TurbulenceStateView state;
    state.u     = &u;
    state.v     = &v;
    state.w     = &w;
    state.theta = &theta;
    state.qv    = &qv;
    state.rho   = &rho;
    state.rho0  = &rho0;
    state.NR    = nx;
    state.NTH   = ny;
    state.NZ    = nz;

    std::vector<float>            actual_du(cells, -101.0f);
    std::vector<float>            actual_dv(cells, -102.0f);
    std::vector<float>            actual_dw(cells, -103.0f);
    std::vector<float>            actual_dtheta(cells, -104.0f);
    std::vector<float>            actual_dqv(cells, -105.0f);
    SmagorinskyTendenciesDispatch dispatch{};
    dispatch.u            = u.data();
    dispatch.v            = v.data();
    dispatch.w            = w.data();
    dispatch.theta_data   = theta.data();
    dispatch.qv_data      = qv.data();
    dispatch.nu_t         = nu.data();
    dispatch.K_theta      = kth.data();
    dispatch.K_q          = kq.data();
    dispatch.dz_prof      = dz_profile.data();
    dispatch.rho          = rho.data();
    dispatch.prof_len     = nz;
    dispatch.dudt_sgs     = actual_du.data();
    dispatch.dvdt_sgs     = actual_dv.data();
    dispatch.dwdt_sgs     = actual_dw.data();
    dispatch.dthetadt_sgs = actual_dtheta.data();
    dispatch.dqvdt_sgs    = actual_dqv.data();
    dispatch.nx           = nx;
    dispatch.ny           = ny;
    dispatch.nz           = nz;
    dispatch.dx           = dx;
    dispatch.dy           = dy;

    struct TendencyCase
    {
        const char* mode;
        bool        horizontal;
        bool        vertical;
        bool        periodic;
        bool        has_qv;
    };
    const std::array<TendencyCase, 4> cases{{
        {"horizontal_only", true, false, false, true},
        {"vertical_only", false, true, false, false},
        {"3d", true, true, false, true},
        {"3d", true, true, true, true},
    }};

    double maximum_momentum_error = 0.0;
    double maximum_scalar_error   = 0.0;
    for (const auto& tendency_case : cases)
    {
        CAPTURE(tendency_case.mode, tendency_case.periodic, tendency_case.has_qv);
        state.periodic_x = tendency_case.periodic;
        state.periodic_y = tendency_case.periodic;
        test_lateral_bc_west =
            tendency_case.periodic ? tmv::init::config::LateralBC::Periodic : tmv::init::config::LateralBC::Rigid;
        test_lateral_bc_east  = test_lateral_bc_west;
        test_lateral_bc_south = test_lateral_bc_west;
        test_lateral_bc_north = test_lateral_bc_west;

        tmv::TurbulenceConfig config;
        config.mode = tendency_case.mode;
        Field3D expected_du(nx, ny, nz, 0.0f);
        Field3D expected_dv(nx, ny, nz, 0.0f);
        Field3D expected_dw(nx, ny, nz, 0.0f);
        Field3D expected_dtheta(nx, ny, nz, 0.0f);
        Field3D expected_dqv(nx, ny, nz, 0.0f);
        tmv::eddy_viscosity::compute_momentum_diffusion_tendencies(
            config, state, grid, nu, expected_du, expected_dv, expected_dw);
        tmv::eddy_viscosity::compute_scalar_diffusion_tendencies(config, state, grid, kth, theta, expected_dtheta);
        if (tendency_case.has_qv)
            tmv::eddy_viscosity::compute_scalar_diffusion_tendencies(config, state, grid, kq, qv, expected_dqv);

        dispatch.use_horizontal = tendency_case.horizontal;
        dispatch.use_vertical   = tendency_case.vertical;
        dispatch.periodic_x     = tendency_case.periodic;
        dispatch.periodic_y     = tendency_case.periodic;
        dispatch.has_qv         = tendency_case.has_qv;
        REQUIRE(backend.dispatch_smagorinsky_tendencies(dispatch));

        for (std::size_t index = 0; index < cells; ++index)
        {
            REQUIRE(std::isfinite(actual_du[index]));
            REQUIRE(std::isfinite(actual_dv[index]));
            REQUIRE(std::isfinite(actual_dw[index]));
            REQUIRE(std::isfinite(actual_dtheta[index]));
            REQUIRE(std::isfinite(actual_dqv[index]));
            maximum_momentum_error = std::max(
                maximum_momentum_error, std::abs(static_cast<double>(actual_du[index]) - expected_du.data()[index]));
            maximum_momentum_error = std::max(
                maximum_momentum_error, std::abs(static_cast<double>(actual_dv[index]) - expected_dv.data()[index]));
            maximum_momentum_error = std::max(
                maximum_momentum_error, std::abs(static_cast<double>(actual_dw[index]) - expected_dw.data()[index]));
            maximum_scalar_error =
                std::max(maximum_scalar_error,
                         std::abs(static_cast<double>(actual_dtheta[index]) - expected_dtheta.data()[index]));
            maximum_scalar_error = std::max(
                maximum_scalar_error, std::abs(static_cast<double>(actual_dqv[index]) - expected_dqv.data()[index]));
        }
    }
    REQUIRE(maximum_momentum_error < 2.0e-8);
    REQUIRE(maximum_scalar_error < 2.0e-7);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::SmagorinskyTendencies) == cases.size());

    const auto                    du_before        = actual_du;
    const auto                    dv_before        = actual_dv;
    const auto                    dw_before        = actual_dw;
    const auto                    dtheta_before    = actual_dtheta;
    const auto                    dqv_before       = actual_dqv;
    const auto                    completed_before = backend.successful_dispatch_count();
    SmagorinskyTendenciesDispatch invalid          = dispatch;
    std::vector<float>            invalid_rho(rho.data(), rho.data() + cells);
    invalid_rho[17] = 0.0f;
    invalid.rho     = invalid_rho.data();
    REQUIRE_FALSE(backend.dispatch_smagorinsky_tendencies(invalid));
    std::vector<float> invalid_dz = dz_profile;
    invalid_dz[2]                 = -1.0f;
    invalid                       = dispatch;
    invalid.dz_prof               = invalid_dz.data();
    REQUIRE_FALSE(backend.dispatch_smagorinsky_tendencies(invalid));
    invalid          = dispatch;
    invalid.dvdt_sgs = invalid.dudt_sgs;
    REQUIRE_FALSE(backend.dispatch_smagorinsky_tendencies(invalid));
    invalid    = dispatch;
    invalid.dx = 0.0f;
    REQUIRE_FALSE(backend.dispatch_smagorinsky_tendencies(invalid));
    REQUIRE(std::memcmp(actual_du.data(), du_before.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(actual_dv.data(), dv_before.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(actual_dw.data(), dw_before.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(actual_dtheta.data(), dtheta_before.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(actual_dqv.data(), dqv_before.data(), cells * sizeof(float)) == 0);
    REQUIRE(backend.successful_dispatch_count() == completed_before);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::SmagorinskyTendencies) == cases.size());

    const auto record = backend.execution_record();
    REQUIRE(record.completed_smagorinsky_tendencies_dispatches == cases.size());
    REQUIRE(record.smagorinsky_tendencies_msl_source_hash.size() == 16);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::smagorinsky_tendencies_msl_source()) ==
            record.smagorinsky_tendencies_msl_source_hash);
    std::cout << "[metal-smagorinsky-tendencies-evidence] cases=" << cases.size()
              << " max_momentum_abs=" << maximum_momentum_error << " max_scalar_abs=" << maximum_scalar_error
              << " msl_fnv1a64=" << record.smagorinsky_tendencies_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';

    test_lateral_bc_west  = tmv::init::config::LateralBC::Rigid;
    test_lateral_bc_east  = tmv::init::config::LateralBC::Rigid;
    test_lateral_bc_south = tmv::init::config::LateralBC::Rigid;
    test_lateral_bc_north = tmv::init::config::LateralBC::Rigid;
}

TEST_CASE("Metal fused Smagorinsky chain uses one completed command buffer",
          "[metal][physical][gpu_parity][turbulence][smagorinsky][fused]")
{
    constexpr int      nx    = 7;
    constexpr int      ny    = 7;
    constexpr int      nz    = 5;
    constexpr float    dx    = 95.0f;
    constexpr float    dy    = 125.0f;
    const std::size_t  cells = static_cast<std::size_t>(nx) * ny * nz;
    std::vector<float> u(cells), v(cells), w(cells), theta(cells), qv(cells), rho(cells);
    std::vector<float> dz(nz), dzc(nz);
    for (int k = 0; k < nz; ++k)
    {
        dz[static_cast<std::size_t>(k)]  = 70.0f + 6.0f * k;
        dzc[static_cast<std::size_t>(k)] = 2.0f * dz[static_cast<std::size_t>(k)];
    }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t index = (static_cast<std::size_t>(i) * ny + j) * nz + k;
                const float       wave  = std::sin(0.27f * i + 0.19f * j + 0.35f * k);
                u[index]                = 1.5f + 0.2f * i - 0.07f * j + 0.1f * k + wave;
                v[index]                = -0.8f - 0.08f * i + 0.16f * j + 0.05f * k - 0.4f * wave;
                w[index]                = 0.04f * i - 0.02f * j + 0.09f * (k + 1) + 0.2f * wave;
                theta[index]            = 297.0f + 0.3f * i - 0.1f * j + 0.65f * k + 0.7f * wave;
                qv[index]               = 0.01f + 0.0001f * i + 0.00008f * j - 0.0002f * k + 0.0002f * wave;
                rho[index]              = 0.82f + 0.015f * i + 0.009f * j + 0.025f * k + 0.01f * wave;
            }

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.supports_smagorinsky_fused());
    REQUIRE(backend.turbulence_kernels() == &backend);
    REQUIRE(backend.supports_turbulence(TurbulenceKernelOp::Smagorinsky));

    std::vector<float>              split_nu(cells), split_kth(cells), split_kq(cells);
    SmagorinskyCoefficientsDispatch coefficients{};
    coefficients.u              = u.data();
    coefficients.v              = v.data();
    coefficients.w              = w.data();
    coefficients.dz_prof        = dz.data();
    coefficients.dzc_prof       = dzc.data();
    coefficients.prof_len       = nz;
    coefficients.nu_t           = split_nu.data();
    coefficients.K_theta        = split_kth.data();
    coefficients.K_q            = split_kq.data();
    coefficients.nx             = nx;
    coefficients.ny             = ny;
    coefficients.nz             = nz;
    coefficients.filter_mode    = 0;
    coefficients.use_horizontal = true;
    coefficients.use_vertical   = true;
    coefficients.periodic_x     = true;
    coefficients.periodic_y     = true;
    coefficients.dx             = dx;
    coefficients.dy             = dy;
    coefficients.Cs             = 0.18f;
    coefficients.Pr_t           = 1.0f / 3.0f;
    coefficients.Sc_t           = 0.4f;
    coefficients.nu_t_max       = 1000.0f;
    coefficients.K_max          = 1000.0f;
    REQUIRE(backend.dispatch_smagorinsky_coefficients(coefficients));

    std::vector<float> split_du(cells), split_dv(cells), split_dw(cells), split_dtheta(cells), split_dqv(cells);
    SmagorinskyTendenciesDispatch tendencies{};
    tendencies.u              = u.data();
    tendencies.v              = v.data();
    tendencies.w              = w.data();
    tendencies.theta_data     = theta.data();
    tendencies.qv_data        = qv.data();
    tendencies.nu_t           = split_nu.data();
    tendencies.K_theta        = split_kth.data();
    tendencies.K_q            = split_kq.data();
    tendencies.dz_prof        = dz.data();
    tendencies.rho            = rho.data();
    tendencies.prof_len       = nz;
    tendencies.dudt_sgs       = split_du.data();
    tendencies.dvdt_sgs       = split_dv.data();
    tendencies.dwdt_sgs       = split_dw.data();
    tendencies.dthetadt_sgs   = split_dtheta.data();
    tendencies.dqvdt_sgs      = split_dqv.data();
    tendencies.nx             = nx;
    tendencies.ny             = ny;
    tendencies.nz             = nz;
    tendencies.use_horizontal = true;
    tendencies.use_vertical   = true;
    tendencies.has_qv         = true;
    tendencies.periodic_x     = true;
    tendencies.periodic_y     = true;
    tendencies.dx             = dx;
    tendencies.dy             = dy;
    REQUIRE(backend.dispatch_smagorinsky_tendencies(tendencies));

    std::vector<float>       fused_nu(cells, -1.0f), fused_kth(cells, -2.0f), fused_kq(cells, -3.0f);
    std::vector<float>       fused_du(cells, -4.0f), fused_dv(cells, -5.0f), fused_dw(cells, -6.0f);
    std::vector<float>       fused_dtheta(cells, -7.0f), fused_dqv(cells, -8.0f);
    SmagorinskyFusedDispatch fused{};
    fused.u_data                = u.data();
    fused.v_data                = v.data();
    fused.w_data                = w.data();
    fused.theta_data            = theta.data();
    fused.qv_data               = qv.data();
    fused.dz_prof               = dz.data();
    fused.dzc_prof              = dzc.data();
    fused.rho                   = rho.data();
    fused.prof_len              = nz;
    fused.nu_t                  = fused_nu.data();
    fused.K_theta               = fused_kth.data();
    fused.K_q                   = fused_kq.data();
    fused.dudt_sgs              = fused_du.data();
    fused.dvdt_sgs              = fused_dv.data();
    fused.dwdt_sgs              = fused_dw.data();
    fused.dthetadt_sgs          = fused_dtheta.data();
    fused.dqvdt_sgs             = fused_dqv.data();
    fused.nx                    = nx;
    fused.ny                    = ny;
    fused.nz                    = nz;
    fused.filter_mode           = coefficients.filter_mode;
    fused.use_horizontal        = true;
    fused.use_vertical          = true;
    fused.has_qv                = true;
    fused.periodic_x            = true;
    fused.periodic_y            = true;
    fused.dx_val                = dx;
    fused.dy_val                = dy;
    fused.Cs                    = coefficients.Cs;
    fused.Pr_t                  = coefficients.Pr_t;
    fused.Sc_t                  = coefficients.Sc_t;
    fused.nu_t_max              = coefficients.nu_t_max;
    fused.K_max                 = coefficients.K_max;
    fused.download_coefficients = true;

    backend.begin_resident_step();
    const auto command_buffers_before = backend.completed_command_buffer_count();
    REQUIRE(backend.dispatch_smagorinsky_fused(fused));
    REQUIRE(backend.completed_command_buffer_count() == command_buffers_before + 1);
    REQUIRE(std::memcmp(fused_nu.data(), split_nu.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(fused_kth.data(), split_kth.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(fused_kq.data(), split_kq.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(fused_du.data(), split_du.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(fused_dv.data(), split_dv.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(fused_dw.data(), split_dw.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(fused_dtheta.data(), split_dtheta.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(fused_dqv.data(), split_dqv.data(), cells * sizeof(float)) == 0);

    std::fill(fused_du.begin(), fused_du.end(), -11.0f);
    std::fill(fused_dv.begin(), fused_dv.end(), -12.0f);
    std::fill(fused_dw.begin(), fused_dw.end(), -13.0f);
    std::fill(fused_dtheta.begin(), fused_dtheta.end(), -14.0f);
    std::fill(fused_dqv.begin(), fused_dqv.end(), -15.0f);
    fused.nu_t                  = nullptr;
    fused.K_theta               = nullptr;
    fused.K_q                   = nullptr;
    fused.download_coefficients = false;
    REQUIRE(backend.dispatch_smagorinsky_fused(fused));
    REQUIRE(backend.completed_command_buffer_count() == command_buffers_before + 2);
    REQUIRE(std::memcmp(fused_du.data(), split_du.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(fused_dv.data(), split_dv.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(fused_dw.data(), split_dw.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(fused_dtheta.data(), split_dtheta.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(fused_dqv.data(), split_dqv.data(), cells * sizeof(float)) == 0);

    std::vector<float> applied_u     = u;
    std::vector<float> applied_v     = v;
    std::vector<float> applied_w     = w;
    std::vector<float> applied_theta = theta;
    std::vector<float> applied_qv    = qv;
    std::vector<float> applied_tke(cells, 0.2f);
    std::vector<float> zero_tke_tendency(cells, 0.0f);
    std::vector<float> expected_applied_u(cells), expected_applied_v(cells), expected_applied_w(cells);
    std::vector<float> expected_applied_theta(cells), expected_applied_qv(cells);
    constexpr float    apply_dt = 0.2f;
    for (std::size_t index = 0; index < cells; ++index)
    {
        expected_applied_u[index]     = std::clamp(u[index] + fused_du[index] * apply_dt, -250.0f, 250.0f);
        expected_applied_v[index]     = std::clamp(v[index] + fused_dv[index] * apply_dt, -250.0f, 250.0f);
        expected_applied_w[index]     = std::clamp(w[index] + fused_dw[index] * apply_dt, -150.0f, 150.0f);
        expected_applied_theta[index] = std::clamp(theta[index] + fused_dtheta[index] * apply_dt, 150.0f, 700.0f);
        expected_applied_qv[index]    = std::clamp(qv[index] + fused_dqv[index] * apply_dt, 0.0f, 0.1f);
    }
    SgsApplyDispatch resident_apply{};
    resident_apply.u          = applied_u.data();
    resident_apply.v          = applied_v.data();
    resident_apply.w          = applied_w.data();
    resident_apply.theta_data = applied_theta.data();
    resident_apply.qv_data    = applied_qv.data();
    resident_apply.tke_data   = applied_tke.data();
    resident_apply.dudt       = fused_du.data();
    resident_apply.dvdt       = fused_dv.data();
    resident_apply.dwdt       = fused_dw.data();
    resident_apply.dthetadt   = fused_dtheta.data();
    resident_apply.dqvdt      = fused_dqv.data();
    resident_apply.dtkedt     = zero_tke_tendency.data();
    resident_apply.dtkedt_pbl = zero_tke_tendency.data();
    // The two TKE inputs may legitimately share the immutable zero field.
    // Give the backend distinct identities to retain its general alias guard.
    std::vector<float> zero_pbl_tendency(cells, 0.0f);
    resident_apply.dtkedt_pbl                    = zero_pbl_tendency.data();
    resident_apply.nx                            = nx;
    resident_apply.ny                            = ny;
    resident_apply.nz                            = nz;
    resident_apply.has_qv                        = true;
    resident_apply.has_tke                       = true;
    resident_apply.dt_step                       = apply_dt;
    resident_apply.theta_fallback                = 300.0f;
    resident_apply.theta_min                     = 150.0f;
    resident_apply.theta_max                     = 700.0f;
    resident_apply.qv_max_val                    = 0.1f;
    resident_apply.wind_h_max                    = 250.0f;
    resident_apply.wind_v_max                    = 150.0f;
    resident_apply.resident_tendencies_unchanged = true;
    resident_apply.sanitize                      = true;
    REQUIRE(backend.dispatch_sgs_apply(resident_apply));
    REQUIRE(applied_u == expected_applied_u);
    REQUIRE(applied_v == expected_applied_v);
    REQUIRE(applied_w == expected_applied_w);
    REQUIRE(applied_theta == expected_applied_theta);
    REQUIRE(applied_qv == expected_applied_qv);
    REQUIRE(backend.execution_record().sgs_resident_tendency_reuses == 5);

    backend.invalidate_resident_field("sgs_dudt");
    applied_u     = u;
    applied_v     = v;
    applied_w     = w;
    applied_theta = theta;
    applied_qv    = qv;
    std::fill(applied_tke.begin(), applied_tke.end(), 0.2f);
    REQUIRE(backend.dispatch_sgs_apply(resident_apply));
    REQUIRE(applied_u == expected_applied_u);
    REQUIRE(backend.execution_record().sgs_resident_tendency_reuses == 5);

    const auto         du_before_rejection        = fused_du;
    const auto         completed_before_rejection = backend.completed_command_buffer_count();
    std::vector<float> invalid_rho                = rho;
    invalid_rho[9]                                = 0.0f;
    fused.rho                                     = invalid_rho.data();
    REQUIRE_FALSE(backend.dispatch_smagorinsky_fused(fused));
    REQUIRE(backend.completed_command_buffer_count() == completed_before_rejection);
    REQUIRE(std::memcmp(fused_du.data(), du_before_rejection.data(), cells * sizeof(float)) == 0);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_smagorinsky_fused_dispatches == 2);
    REQUIRE(record.completed_sgs_apply_dispatches == 2);
    REQUIRE(record.sgs_resident_tendency_reuses == 5);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::SmagorinskyCoefficients) == 3);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::SmagorinskyTendencies) == 3);
    std::cout << "[metal-smagorinsky-fused-evidence] fused_submissions="
              << record.completed_smagorinsky_fused_dispatches
              << " command_buffers=" << record.completed_command_buffers << " coefficient_dispatches="
              << backend.successful_kernel_dispatch_count(KernelId::SmagorinskyCoefficients)
              << " tendency_dispatches=" << backend.successful_kernel_dispatch_count(KernelId::SmagorinskyTendencies)
              << " resident_tendency_reuses=" << record.sgs_resident_tendency_reuses
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal SGS Apply publishes six fields atomically with canonical guards",
          "[metal][physical][gpu_parity][turbulence][sgs_apply]")
{
    constexpr int      nx    = 5;
    constexpr int      ny    = 4;
    constexpr int      nz    = 3;
    constexpr float    dt    = 0.25f;
    const std::size_t  cells = static_cast<std::size_t>(nx) * ny * nz;
    std::vector<float> u(cells), v(cells), w(cells), theta(cells), qv(cells), tke(cells);
    std::vector<float> du(cells), dv(cells), dw(cells), dtheta(cells), dqv(cells), dtke(cells), dtke_pbl(cells);
    for (std::size_t index = 0; index < cells; ++index)
    {
        const float x   = static_cast<float>(index);
        u[index]        = -12.0f + 0.7f * x;
        v[index]        = 8.0f - 0.4f * x;
        w[index]        = -3.0f + 0.2f * x;
        theta[index]    = 275.0f + 0.9f * x;
        qv[index]       = 0.004f + 0.0002f * x;
        tke[index]      = 0.1f + 0.01f * x;
        du[index]       = 0.3f - 0.01f * x;
        dv[index]       = -0.2f + 0.008f * x;
        dw[index]       = 0.1f - 0.005f * x;
        dtheta[index]   = 0.4f - 0.012f * x;
        dqv[index]      = 0.0001f - 0.000003f * x;
        dtke[index]     = -0.02f + 0.0004f * x;
        dtke_pbl[index] = 0.005f;
    }
    du[2]      = std::numeric_limits<float>::quiet_NaN();
    u[3]       = std::numeric_limits<float>::infinity();
    dtheta[4]  = std::numeric_limits<float>::infinity();
    theta[5]   = std::numeric_limits<float>::quiet_NaN();
    dqv[6]     = std::numeric_limits<float>::quiet_NaN();
    dtke[7]    = std::numeric_limits<float>::infinity();
    du[8]      = 5000.0f;
    dw[9]      = -5000.0f;
    dtheta[10] = 5000.0f;
    dqv[11]    = -5.0f;
    dtke[12]   = -5000.0f;

    const auto         initial_u     = u;
    const auto         initial_v     = v;
    const auto         initial_w     = w;
    const auto         initial_theta = theta;
    const auto         initial_qv    = qv;
    const auto         initial_tke   = tke;
    const auto         finite        = [](float value) { return std::isfinite(value); };
    const auto         clean         = [&](float value) { return finite(value) ? value : 0.0f; };
    std::vector<float> expected_u(cells), expected_v(cells), expected_w(cells);
    std::vector<float> expected_theta(cells), expected_qv(cells), expected_tke(cells);
    for (std::size_t index = 0; index < cells; ++index)
    {
        const float u_new = initial_u[index] + clean(du[index]) * dt;
        expected_u[index] = finite(u_new) ? std::clamp(u_new, -250.0f, 250.0f) : 0.0f;
        const float v_new = initial_v[index] + clean(dv[index]) * dt;
        expected_v[index] = finite(v_new) ? std::clamp(v_new, -250.0f, 250.0f) : 0.0f;
        const float w_new = initial_w[index] + clean(dw[index]) * dt;
        expected_w[index] = finite(w_new) ? std::clamp(w_new, -150.0f, 150.0f) : 0.0f;
        float theta_new   = initial_theta[index] + clean(dtheta[index]) * dt;
        if (!finite(theta_new))
            theta_new = finite(initial_theta[index]) ? initial_theta[index] : 300.0f;
        expected_theta[index] = std::clamp(theta_new, 150.0f, 700.0f);
        const float qv_new    = initial_qv[index] + clean(dqv[index]) * dt;
        expected_qv[index]    = finite(qv_new) ? std::clamp(qv_new, 0.0f, 0.1f) : 0.0f;
        float tke_new         = initial_tke[index] + (clean(dtke[index]) + clean(dtke_pbl[index])) * dt;
        if (!finite(tke_new))
            tke_new = finite(initial_tke[index]) ? initial_tke[index] : 0.0f;
        expected_tke[index] = std::max(0.0f, tke_new);
    }

    struct ObserverCapture
    {
        std::size_t                              cells          = 0;
        bool                                     called         = false;
        bool                                     host_unchanged = false;
        std::array<const std::vector<float>*, 6> host{};
        std::array<const std::vector<float>*, 6> original{};
        std::array<std::vector<float>, 6>        candidate;
    } capture;
    capture.cells      = cells;
    capture.host       = {&u, &v, &w, &theta, &qv, &tke};
    capture.original   = {&initial_u, &initial_v, &initial_w, &initial_theta, &initial_qv, &initial_tke};
    const auto observe = +[](void*        context,
                             const float* out_u,
                             const float* out_v,
                             const float* out_w,
                             const float* out_theta,
                             const float* out_qv,
                             const float* out_tke)
    {
        auto&        result    = *static_cast<ObserverCapture*>(context);
        const float* fields[6] = {out_u, out_v, out_w, out_theta, out_qv, out_tke};
        result.host_unchanged  = true;
        for (std::size_t field = 0; field < 6; ++field)
        {
            if (std::memcmp(result.host[field]->data(), result.original[field]->data(), result.cells * sizeof(float)) !=
                0)
                result.host_unchanged = false;
            result.candidate[field].assign(fields[field], fields[field] + result.cells);
        }
        result.called = true;
    };

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.supports_sgs_apply());
    REQUIRE(backend.supports_turbulence(TurbulenceKernelOp::SgsApply));
    REQUIRE(backend.ready(KernelId::SgsApply));
    REQUIRE(backend.pipeline(KernelId::SgsApply) != 0);
    SgsApplyDispatch apply{};
    apply.u                          = u.data();
    apply.v                          = v.data();
    apply.w                          = w.data();
    apply.theta_data                 = theta.data();
    apply.qv_data                    = qv.data();
    apply.tke_data                   = tke.data();
    apply.dudt                       = du.data();
    apply.dvdt                       = dv.data();
    apply.dwdt                       = dw.data();
    apply.dthetadt                   = dtheta.data();
    apply.dqvdt                      = dqv.data();
    apply.dtkedt                     = dtke.data();
    apply.dtkedt_pbl                 = dtke_pbl.data();
    apply.nx                         = nx;
    apply.ny                         = ny;
    apply.nz                         = nz;
    apply.has_qv                     = true;
    apply.has_tke                    = true;
    apply.dt_step                    = dt;
    apply.theta_fallback             = 300.0f;
    apply.theta_min                  = 150.0f;
    apply.theta_max                  = 700.0f;
    apply.qv_max_val                 = 0.1f;
    apply.wind_h_max                 = 250.0f;
    apply.wind_v_max                 = 150.0f;
    apply.sanitize                   = true;
    apply.completed_observer         = observe;
    apply.completed_observer_context = &capture;
    REQUIRE(backend.dispatch_sgs_apply(apply));
    REQUIRE(capture.called);
    REQUIRE(capture.host_unchanged);
    REQUIRE(u == expected_u);
    REQUIRE(v == expected_v);
    REQUIRE(w == expected_w);
    REQUIRE(theta == expected_theta);
    REQUIRE(qv == expected_qv);
    REQUIRE(tke == expected_tke);
    REQUIRE(capture.candidate[0] == expected_u);
    REQUIRE(capture.candidate[1] == expected_v);
    REQUIRE(capture.candidate[2] == expected_w);
    REQUIRE(capture.candidate[3] == expected_theta);
    REQUIRE(capture.candidate[4] == expected_qv);
    REQUIRE(capture.candidate[5] == expected_tke);

    u                 = initial_u;
    v                 = initial_v;
    w                 = initial_w;
    theta             = initial_theta;
    qv                = initial_qv;
    tke               = initial_tke;
    const auto reject = +[](void*, const float*, const float*, const float*, const float*, const float*, const float*)
    { throw std::runtime_error("reject raw Metal SGS candidate"); };
    apply.sanitize                   = false;
    apply.completed_observer         = reject;
    apply.completed_observer_context = nullptr;
    const auto completed_before_veto = backend.completed_command_buffer_count();
    REQUIRE_THROWS_WITH(backend.dispatch_sgs_apply(apply), "reject raw Metal SGS candidate");
    REQUIRE(backend.completed_command_buffer_count() == completed_before_veto + 1);
    REQUIRE(std::memcmp(u.data(), initial_u.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(v.data(), initial_v.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(w.data(), initial_w.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(theta.data(), initial_theta.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(qv.data(), initial_qv.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(tke.data(), initial_tke.data(), cells * sizeof(float)) == 0);

    apply.sanitize                      = true;
    apply.completed_observer            = nullptr;
    apply.theta_min                     = 800.0f;
    apply.theta_max                     = 700.0f;
    const auto completed_before_invalid = backend.completed_command_buffer_count();
    REQUIRE_FALSE(backend.dispatch_sgs_apply(apply));
    REQUIRE(backend.completed_command_buffer_count() == completed_before_invalid);
    REQUIRE(std::memcmp(u.data(), initial_u.data(), cells * sizeof(float)) == 0);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_sgs_apply_dispatches == 2);
    REQUIRE(record.sgs_apply_msl_source_hash.size() == 16);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::sgs_apply_msl_source()) == record.sgs_apply_msl_source_hash);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::SgsApply) == 2);
    std::cout << "[metal-sgs-apply-evidence] completed=" << record.completed_sgs_apply_dispatches
              << " observer_veto=atomic msl_fnv1a64=" << record.sgs_apply_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal diffusion matches canonical live-density A/C-grid control volumes",
          "[metal][physical][gpu_parity][diffusion]")
{
    constexpr int     nx    = 11;
    constexpr int     ny    = 10;
    constexpr int     nz    = 8;
    const std::size_t cells = static_cast<std::size_t>(nx) * ny * nz;
    Field3D           rho(nx, ny, nz, 1.0f);
    Field3D           u(nx, ny, nz, 0.0f);
    Field3D           v(nx, ny, nz, 0.0f);
    Field3D           w(nx, ny, nz, 0.0f);
    Field3D           theta(nx, ny, nz, 0.0f);
    Field3D           qv(nx, ny, nz, 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                rho(i, j, k)   = 0.8f + 0.017f * i + 0.009f * j + 0.013f * k;
                const float x  = static_cast<float>(i - 4);
                const float y  = static_cast<float>(j - 5);
                const float z  = static_cast<float>(k - 3);
                u(i, j, k)     = 3.0f + 0.11f * x * x - 0.04f * y + 0.03f * z;
                v(i, j, k)     = -2.0f + 0.05f * y * y + 0.02f * x * z;
                w(i, j, k)     = 0.3f + 0.07f * z * z - 0.01f * x * y;
                theta(i, j, k) = 300.0f + 0.08f * x * x + 0.06f * y * y + 0.13f * z * z;
                qv(i, j, k)    = 0.009f + 0.00001f * x * x + 0.00002f * y * y + 0.00003f * z * z;
            }

    tmv::GridMetrics metrics;
    metrics.dx = 725.0;
    metrics.dy = 910.0;
    metrics.dz = {80.0, 95.0, 120.0, 155.0, 205.0, 270.0, 350.0, 460.0};
    tmv::DiffusionConfig config;
    config.K_h          = 43.0;
    config.K_v          = 71.0;
    config.dt_diffusion = 0.1;
    config.apply_to     = "scalars_and_momentum";
    tmv::DiffusionStateView state;
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
    ExplicitDiffusionScheme cpu;
    cpu.initialize(config);
    tmv::DiffusionTendencies expected;
    global_compute_backend_config.allow_fallback = true;
    cpu.compute_diffusion_tendencies(config, state, expected, nullptr);

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.supports_diffusion_dispatch());
    REQUIRE(backend.diffusion_kernels() == &backend);
    REQUIRE(backend.supports_diffusion(DiffusionKernelOp::Diffusion));
    REQUIRE(backend.ready(KernelId::Diffusion));
    REQUIRE(backend.pipeline(KernelId::Diffusion) != 0);

    std::vector<float>                dz(metrics.dz.begin(), metrics.dz.end());
    std::array<std::vector<float>, 5> actual;
    for (auto& field : actual)
        field.assign(cells, -999.0f);
    const float* fields[5] = {u.data(), v.data(), w.data(), theta.data(), qv.data()};
    float* outputs[5]      = {actual[0].data(), actual[1].data(), actual[2].data(), actual[3].data(), actual[4].data()};
    const DiffusionFieldLocation locations[5] = {DiffusionFieldLocation::UFace,
                                                 DiffusionFieldLocation::VFace,
                                                 DiffusionFieldLocation::WFace,
                                                 DiffusionFieldLocation::Scalar,
                                                 DiffusionFieldLocation::Scalar};
    DiffusionBatchDispatch       batch{fields,
                                       outputs,
                                       locations,
                                       5,
                                       rho.data(),
                                       dz.data(),
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
    const auto                   submissions_before = backend.completed_command_buffer_count();
    REQUIRE(backend.dispatch_diffusion_batch(batch));
    REQUIRE(backend.completed_command_buffer_count() == submissions_before + 1);
    REQUIRE(actual[0][0] == 0.0f);
    REQUIRE(actual[1][0] == 0.0f);
    REQUIRE(actual[2][0] == 0.0f);

    const float* expected_fields[5] = {expected.dudt_diff.data(),
                                       expected.dvdt_diff.data(),
                                       expected.dwdt_diff.data(),
                                       expected.dthetadt_diff.data(),
                                       expected.dqvdt_diff.data()};
    double       maximum_error      = 0.0;
    for (int target = 0; target < 5; ++target)
        for (std::size_t index = 0; index < cells; ++index)
        {
            REQUIRE(std::isfinite(actual[static_cast<std::size_t>(target)][index]));
            maximum_error = std::max(maximum_error,
                                     std::abs(static_cast<double>(actual[static_cast<std::size_t>(target)][index]) -
                                              expected_fields[target][index]));
        }
    REQUIRE(maximum_error < 2.0e-6);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::Diffusion) == 5);

    std::vector<float> single(cells, -77.0f);
    DiffusionDispatch  one{};
    one.field      = theta.data();
    one.rho        = rho.data();
    one.dz         = dz.data();
    one.tendency   = single.data();
    one.nx         = nx;
    one.ny         = ny;
    one.nz         = nz;
    one.dx         = static_cast<float>(metrics.dx);
    one.dy         = static_cast<float>(metrics.dy);
    one.K_h        = static_cast<float>(config.K_h);
    one.K_v        = static_cast<float>(config.K_v);
    one.location   = DiffusionFieldLocation::Scalar;
    one.c_grid     = true;
    one.periodic_x = true;
    one.periodic_y = false;
    REQUIRE(backend.dispatch_diffusion(one));
    REQUIRE(std::memcmp(single.data(), actual[3].data(), cells * sizeof(float)) == 0);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::Diffusion) == 6);

    tmv::DiffusionStateView a_grid_state = state;
    a_grid_state.c_grid                  = false;
    a_grid_state.periodic_x              = false;
    a_grid_state.periodic_y              = true;
    tmv::DiffusionTendencies expected_a_grid;
    cpu.compute_diffusion_tendencies(config, a_grid_state, expected_a_grid, nullptr);
    for (auto& field : actual)
        std::fill(field.begin(), field.end(), -555.0f);
    batch.c_grid     = false;
    batch.periodic_x = false;
    batch.periodic_y = true;
    REQUIRE(backend.dispatch_diffusion_batch(batch));
    const float* expected_a_grid_fields[5] = {expected_a_grid.dudt_diff.data(),
                                              expected_a_grid.dvdt_diff.data(),
                                              expected_a_grid.dwdt_diff.data(),
                                              expected_a_grid.dthetadt_diff.data(),
                                              expected_a_grid.dqvdt_diff.data()};
    for (int target = 0; target < 5; ++target)
        for (std::size_t index = 0; index < cells; ++index)
            maximum_error = std::max(maximum_error,
                                     std::abs(static_cast<double>(actual[static_cast<std::size_t>(target)][index]) -
                                              expected_a_grid_fields[target][index]));
    REQUIRE(maximum_error < 2.0e-6);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::Diffusion) == 11);

    Field3D scalar_k(nx, ny, nz, 0.0f);
    Field3D momentum_k(nx, ny, nz, 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                scalar_k(i, j, k)   = 8.0f + 0.7f * i + 0.3f * j + 1.1f * k;
                momentum_k(i, j, k) = 12.0f + 0.4f * i + 0.9f * j + 0.6f * k;
            }
    tmv::DiffusionConfig variable_config   = config;
    variable_config.K_h                    = 17.0;
    variable_config.K_v                    = 0.0;
    variable_config.use_variable_K         = true;
    tmv::DiffusionStateView variable_state = state;
    variable_state.K_scalar                = &scalar_k;
    variable_state.K_momentum              = &momentum_k;
    tmv::DiffusionTendencies expected_variable;
    cpu.compute_diffusion_tendencies(variable_config, variable_state, expected_variable, nullptr);
    const float* vertical_k[5] = {
        momentum_k.data(), momentum_k.data(), momentum_k.data(), scalar_k.data(), scalar_k.data()};
    batch.K_h        = static_cast<float>(variable_config.K_h);
    batch.K_v        = 0.0f;
    batch.c_grid     = true;
    batch.periodic_x = true;
    batch.periodic_y = false;
    batch.vertical_K = vertical_k;
    REQUIRE(backend.dispatch_diffusion_batch(batch));
    const float* expected_variable_fields[5] = {expected_variable.dudt_diff.data(),
                                                expected_variable.dvdt_diff.data(),
                                                expected_variable.dwdt_diff.data(),
                                                expected_variable.dthetadt_diff.data(),
                                                expected_variable.dqvdt_diff.data()};
    maximum_error                            = 0.0;
    for (int target = 0; target < 5; ++target)
        for (std::size_t index = 0; index < cells; ++index)
            maximum_error = std::max(maximum_error,
                                     std::abs(static_cast<double>(actual[static_cast<std::size_t>(target)][index]) -
                                              expected_variable_fields[target][index]));
    REQUIRE(maximum_error < 2.0e-6);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::Diffusion) == 16);

    const auto         outputs_before_rejection   = actual;
    const auto         completed_before_rejection = backend.completed_command_buffer_count();
    std::vector<float> invalid_rho(rho.data(), rho.data() + cells);
    invalid_rho[33] = 0.0f;
    batch.rho       = invalid_rho.data();
    REQUIRE_FALSE(backend.dispatch_diffusion_batch(batch));
    batch.rho                     = rho.data();
    std::vector<float> invalid_dz = dz;
    invalid_dz[3]                 = -1.0f;
    batch.dz                      = invalid_dz.data();
    REQUIRE_FALSE(backend.dispatch_diffusion_batch(batch));
    batch.dz                           = dz.data();
    Field3D invalid_k                  = scalar_k;
    invalid_k(2, 3, 4)                 = -1.0f;
    const float* invalid_vertical_k[5] = {
        momentum_k.data(), momentum_k.data(), momentum_k.data(), invalid_k.data(), scalar_k.data()};
    batch.vertical_K = invalid_vertical_k;
    REQUIRE_FALSE(backend.dispatch_diffusion_batch(batch));
    batch.vertical_K          = vertical_k;
    float* aliased_outputs[5] = {outputs[0], outputs[0], outputs[2], outputs[3], outputs[4]};
    batch.tendencies          = aliased_outputs;
    REQUIRE_FALSE(backend.dispatch_diffusion_batch(batch));
    batch.tendencies = outputs;
    REQUIRE(backend.completed_command_buffer_count() == completed_before_rejection);
    for (std::size_t target = 0; target < actual.size(); ++target)
        REQUIRE(std::memcmp(actual[target].data(), outputs_before_rejection[target].data(), cells * sizeof(float)) ==
                0);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_diffusion_dispatches == 16);
    REQUIRE(record.completed_diffusion_batches == 3);
    REQUIRE(record.diffusion_msl_source_hash.size() == 16);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::diffusion_msl_source()) == record.diffusion_msl_source_hash);
    std::cout << "[metal-diffusion-evidence] targets=16 submissions=4 max_abs=" << maximum_error
              << " msl_fnv1a64=" << record.diffusion_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal homogeneous field shift is bitwise equal to the CPU oracle",
          "[metal][physical][gpu_parity][domain_recycling][D5]")
{
    constexpr int       nr    = 7;
    constexpr int       nth   = 8;
    constexpr int       nz    = 5;
    const std::size_t   cells = static_cast<std::size_t>(nr) * nth * nz;
    MetalComputeBackend backend;
    initialize_backend(backend);

    std::vector<float> source(cells);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t index = (static_cast<std::size_t>(i) * nth + j) * nz + k;
                source[index]           = static_cast<float>(10000 * i + 100 * j + k);
            }
    const std::vector<float> profile{-11.25f, -22.5f, -33.75f, -45.0f, -56.25f};

    const std::array<Case, 4>     cases{{
        {"positive", 2, 0},
        {"negative", -2, 0},
        {"diagonal", 2, -1},
        {"stagger-opposed", -1, 2},
    }};
    const std::array<Location, 3> locations{{Location::ScalarCenter, Location::XFace, Location::YFace}};

    const auto  wall_start   = std::chrono::steady_clock::now();
    std::size_t parity_cases = 0;
    for (const auto& test_case : cases)
        for (const auto location : locations)
        {
            CAPTURE(test_case.name, test_case.shift_i, test_case.shift_j, static_cast<int>(location));
            const auto scalar_plan = tmv::domain::plan_whole_cell_shift_for_physical_domain(
                nr, nth, nz, test_case.shift_i, test_case.shift_j, 1, nr - 1, 1, nth - 1);
            const auto owner_plan = tmv::domain::plan_whole_cell_shift_for_horizontal_location(scalar_plan, location);
            Field3D    expected(nr, nth, nz, 0.0f);
            std::copy(source.begin(), source.end(), expected.data());
            const auto oracle_report = tmv::domain::execute_homogeneous_field_shift(
                scalar_plan, {{"metal_cpu_oracle", &expected, &profile, location}});
            REQUIRE(oracle_report.fields_shifted == 1);
            std::vector<float> actual(cells, 987654.0f);
            auto               dispatch = make_dispatch(source, profile, actual, owner_plan);
            REQUIRE(backend.dispatch_homogeneous_domain_field_shift(dispatch));
            REQUIRE(std::memcmp(actual.data(), expected.data(), cells * sizeof(float)) == 0);
            ++parity_cases;
        }
    const auto   wall_end = std::chrono::steady_clock::now();
    const double wall_ms  = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    REQUIRE(parity_cases == cases.size() * locations.size());
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::DomainRecyclingHomogeneousFieldShift) == parity_cases);
    REQUIRE(backend.successful_dispatch_count() == parity_cases);
    REQUIRE(backend.completed_command_buffer_count() == parity_cases);
    REQUIRE(backend.last_command_buffer_completed());
    REQUIRE(backend.supported());

    const auto record = backend.execution_record();
    REQUIRE(record.completed_shift_dispatches == parity_cases);
    REQUIRE(record.timestamps_supported);
    REQUIRE(record.total_device_milliseconds >= 0.0);
    REQUIRE(record.timestamped_command_buffers == record.completed_command_buffers);
    REQUIRE(std::isfinite(record.total_submit_wait_milliseconds));
    REQUIRE(record.total_submit_wait_milliseconds > 0.0);
    std::cout << "[metal-evidence] device=\"" << record.device_name << "\" os=\"" << record.operating_system
              << "\" toolchain=\"" << record.host_toolchain << "\" msl_fnv1a64=" << record.msl_source_hash
              << " unified_memory=" << (record.unified_memory ? "true" : "false")
              << " parity=bitwise parity_cases=" << parity_cases
              << " completed_command_buffers=" << record.completed_command_buffers
              << " device_ms=" << record.total_device_milliseconds << " wall_ms=" << wall_ms << '\n';
}

TEST_CASE("Metal resident field-shift batch is atomic and uses one command buffer",
          "[metal][physical][gpu_parity][domain_recycling][D5][batch]")
{
    constexpr int         nr          = 8;
    constexpr int         nth         = 9;
    constexpr int         nz          = 4;
    constexpr std::size_t field_count = 9;
    const std::size_t     cells       = static_cast<std::size_t>(nr) * nth * nz;

    MetalComputeBackend backend;
    initialize_backend(backend);
    backend.begin_resident_step();

    const auto scalar_plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(nr, nth, nz, 2, -1, 1, nr - 1, 1, nth - 1);
    const std::array<Location, field_count> locations{{
        Location::ScalarCenter,
        Location::XFace,
        Location::YFace,
        Location::ScalarCenter,
        Location::XFace,
        Location::YFace,
        Location::ScalarCenter,
        Location::XFace,
        Location::YFace,
    }};

    std::array<std::vector<float>, field_count>                  actual;
    std::array<Field3D, field_count>                             expected;
    std::array<std::vector<float>, field_count>                  profiles;
    std::array<std::string, field_count>                         id_storage;
    std::array<const char*, field_count>                         resident_ids{};
    std::array<HomogeneousDomainFieldShiftDispatch, field_count> fields{};

    for (std::size_t field = 0; field < field_count; ++field)
    {
        actual[field].resize(cells);
        expected[field] = Field3D(nr, nth, nz, 0.0f);
        for (std::size_t index = 0; index < cells; ++index)
        {
            actual[field][index]          = static_cast<float>(field * 100000 + index * 3 + 1);
            expected[field].data()[index] = actual[field][index];
        }
        profiles[field].resize(nz);
        for (int k = 0; k < nz; ++k)
            profiles[field][static_cast<std::size_t>(k)] = static_cast<float>(-1000 - field * 10 - k);

        (void)tmv::domain::execute_homogeneous_field_shift(
            scalar_plan, {{"metal_batch_oracle", &expected[field], &profiles[field], locations[field]}});
        const auto owner_plan =
            tmv::domain::plan_whole_cell_shift_for_horizontal_location(scalar_plan, locations[field]);
        fields[field]       = make_dispatch(actual[field], profiles[field], actual[field], owner_plan);
        id_storage[field]   = "metal_d5_batch_owner_" + std::to_string(field);
        resident_ids[field] = id_storage[field].c_str();
    }

    const HomogeneousDomainFieldShiftBatchDispatch batch{
        fields.data(), resident_ids.data(), static_cast<int>(field_count)};
    REQUIRE(backend.dispatch_homogeneous_domain_field_shift_batch(batch));
    REQUIRE(backend.successful_dispatch_count() == 1);
    REQUIRE(backend.completed_shift_batch_count() == 1);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::DomainRecyclingHomogeneousFieldShift) == field_count);
    REQUIRE(backend.resident_field_count() == field_count);
    REQUIRE(backend.resident_upload_count() == field_count);
    REQUIRE(backend.resident_reuse_count() == 0);
    for (std::size_t field = 0; field < field_count; ++field)
        REQUIRE(std::memcmp(actual[field].data(), expected[field].data(), cells * sizeof(float)) == 0);

    for (std::size_t field = 0; field < field_count; ++field)
    {
        (void)tmv::domain::execute_homogeneous_field_shift(
            scalar_plan, {{"metal_batch_oracle_second", &expected[field], &profiles[field], locations[field]}});
    }
    REQUIRE(backend.dispatch_homogeneous_domain_field_shift_batch(batch));
    REQUIRE(backend.successful_dispatch_count() == 2);
    REQUIRE(backend.completed_shift_batch_count() == 2);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::DomainRecyclingHomogeneousFieldShift) ==
            2 * field_count);
    REQUIRE(backend.resident_field_count() == field_count);
    REQUIRE(backend.resident_upload_count() == field_count);
    REQUIRE(backend.resident_reuse_count() == field_count);
    for (std::size_t field = 0; field < field_count; ++field)
        REQUIRE(std::memcmp(actual[field].data(), expected[field].data(), cells * sizeof(float)) == 0);

    const auto          before_rejection          = actual;
    const std::uint64_t commands_before_rejection = backend.successful_dispatch_count();
    const std::uint64_t kernels_before_rejection =
        backend.successful_kernel_dispatch_count(KernelId::DomainRecyclingHomogeneousFieldShift);
    ++fields[4].source_i_begin;
    REQUIRE_FALSE(backend.dispatch_homogeneous_domain_field_shift_batch(batch));
    REQUIRE(backend.successful_dispatch_count() == commands_before_rejection);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::DomainRecyclingHomogeneousFieldShift) ==
            kernels_before_rejection);
    REQUIRE(backend.completed_shift_batch_count() == 2);
    REQUIRE(backend.resident_field_count() == field_count);
    for (std::size_t field = 0; field < field_count; ++field)
        REQUIRE(std::memcmp(actual[field].data(), before_rejection[field].data(), cells * sizeof(float)) == 0);
    --fields[4].source_i_begin;

    const char* distinct_id = resident_ids[5];
    resident_ids[5]         = resident_ids[4];
    REQUIRE_FALSE(backend.dispatch_homogeneous_domain_field_shift_batch(batch));
    REQUIRE(backend.successful_dispatch_count() == commands_before_rejection);
    REQUIRE(backend.completed_shift_batch_count() == 2);
    for (std::size_t field = 0; field < field_count; ++field)
        REQUIRE(std::memcmp(actual[field].data(), before_rejection[field].data(), cells * sizeof(float)) == 0);
    resident_ids[5] = distinct_id;

    backend.invalidate_resident_field(resident_ids[3]);
    for (std::size_t field = 0; field < field_count; ++field)
    {
        (void)tmv::domain::execute_homogeneous_field_shift(
            scalar_plan, {{"metal_batch_oracle_third", &expected[field], &profiles[field], locations[field]}});
    }
    REQUIRE(backend.dispatch_homogeneous_domain_field_shift_batch(batch));
    REQUIRE(backend.successful_dispatch_count() == 3);
    REQUIRE(backend.completed_shift_batch_count() == 3);
    REQUIRE(backend.resident_upload_count() == field_count + 1);
    REQUIRE(backend.resident_reuse_count() == 2 * field_count - 1);
    for (std::size_t field = 0; field < field_count; ++field)
        REQUIRE(std::memcmp(actual[field].data(), expected[field].data(), cells * sizeof(float)) == 0);

    backend.begin_resident_step();
    for (std::size_t field = 0; field < field_count; ++field)
    {
        (void)tmv::domain::execute_homogeneous_field_shift(
            scalar_plan, {{"metal_batch_oracle_fourth", &expected[field], &profiles[field], locations[field]}});
    }
    REQUIRE(backend.dispatch_homogeneous_domain_field_shift_batch(batch));
    REQUIRE(backend.successful_dispatch_count() == 4);
    REQUIRE(backend.completed_shift_batch_count() == 4);
    REQUIRE(backend.resident_upload_count() == 2 * field_count + 1);
    REQUIRE(backend.resident_reuse_count() == 2 * field_count - 1);
    for (std::size_t field = 0; field < field_count; ++field)
        REQUIRE(std::memcmp(actual[field].data(), expected[field].data(), cells * sizeof(float)) == 0);

    backend.invalidate_all_resident_fields();
    for (std::size_t field = 0; field < field_count; ++field)
    {
        (void)tmv::domain::execute_homogeneous_field_shift(
            scalar_plan, {{"metal_batch_oracle_fifth", &expected[field], &profiles[field], locations[field]}});
    }
    REQUIRE(backend.dispatch_homogeneous_domain_field_shift_batch(batch));
    REQUIRE(backend.successful_dispatch_count() == 5);
    REQUIRE(backend.completed_shift_batch_count() == 5);
    REQUIRE(backend.resident_upload_count() == 3 * field_count + 1);
    REQUIRE(backend.resident_reuse_count() == 2 * field_count - 1);
    for (std::size_t field = 0; field < field_count; ++field)
        REQUIRE(std::memcmp(actual[field].data(), expected[field].data(), cells * sizeof(float)) == 0);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_shift_batches == 5);
    REQUIRE(record.completed_shift_dispatches == 5 * field_count);
    REQUIRE(record.resident_uploads == 3 * field_count + 1);
    REQUIRE(record.resident_reuses == 2 * field_count - 1);
    std::cout << "[metal-batch-evidence] fields=" << field_count << " batches=" << record.completed_shift_batches
              << " kernel_dispatches=" << record.completed_shift_dispatches
              << " command_buffers=" << record.completed_command_buffers
              << " resident_uploads=" << record.resident_uploads << " resident_reuses=" << record.resident_reuses
              << " device_ms=" << record.total_device_milliseconds << '\n';
    backend.shutdown();
    REQUIRE(backend.resident_field_count() == 0);
}

TEST_CASE("Metal invalid domain rectangles fail closed", "[metal][physical][domain_recycling][fail_closed]")
{
    constexpr int       nr    = 7;
    constexpr int       nth   = 8;
    constexpr int       nz    = 5;
    const std::size_t   cells = static_cast<std::size_t>(nr) * nth * nz;
    MetalComputeBackend backend;
    initialize_backend(backend);
    std::vector<float>       source(cells, 12.5f);
    std::vector<float>       profile(nz, -4.25f);
    std::vector<float>       destination(cells, 987654.0f);
    const std::vector<float> original = destination;
    const auto               scalar_plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(nr, nth, nz, 2, -1, 1, nr - 1, 1, nth - 1);
    const auto owner_plan =
        tmv::domain::plan_whole_cell_shift_for_horizontal_location(scalar_plan, Location::ScalarCenter);
    const auto valid = make_dispatch(source, profile, destination, owner_plan);

    std::array<HomogeneousDomainFieldShiftDispatch, 6> invalid{};
    invalid.fill(valid);
    invalid[0].destination_i_begin = -1;
    invalid[1].destination_i_end   = nr + 1;
    invalid[2].destination_j_begin = invalid[2].destination_j_end;
    ++invalid[3].source_i_begin;
    --invalid[4].source_j_end;
    invalid[5].source_j_begin = -1;

    for (std::size_t index = 0; index < invalid.size(); ++index)
    {
        CAPTURE(index);
        REQUIRE_FALSE(backend.dispatch_homogeneous_domain_field_shift(invalid[index]));
        REQUIRE(std::memcmp(destination.data(), original.data(), cells * sizeof(float)) == 0);
    }
    REQUIRE(backend.successful_dispatch_count() == 0);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::DomainRecyclingHomogeneousFieldShift) == 0);
}

namespace
{

struct KesslerPointwiseOracle
{
    std::array<std::vector<float>, 6> tendency;
};

KesslerPointwiseOracle kessler_pointwise_oracle(const KesslerPointwiseDispatch& dispatch)
{
    const std::size_t      cells = static_cast<std::size_t>(dispatch.nr) * dispatch.nth * dispatch.nz;
    KesslerPointwiseOracle result;
    for (auto& field : result.tendency)
        field.assign(cells, 0.0f);
    for (int i = dispatch.i_begin; i < dispatch.i_end; ++i)
        for (int j = dispatch.j_begin; j < dispatch.j_end; ++j)
            for (int k = dispatch.k_begin; k < dispatch.k_end; ++k)
            {
                const std::size_t index       = (static_cast<std::size_t>(i) * dispatch.nth + j) * dispatch.nz + k;
                const float       temperature = dispatch.temperature_data[index];
                const float       pressure    = dispatch.p_data[index];
                const float       qv          = dispatch.qv_data[index];
                const float       qc          = dispatch.qc_data[index];
                const float       qr          = dispatch.qr_data[index];
                const float       rho         = dispatch.rho_d_data[index];
                float             production  = 0.0f;
                if (qc > 0.0f)
                {
                    const float factor =
                        1.0f / (1.0f + dispatch.c_accr * dispatch.dt_step * std::pow(std::max(qr, 0.0f), 0.875f));
                    float amount = qc * (1.0f - factor) +
                                   factor * dispatch.c_auto * dispatch.dt_step * std::max(qc - dispatch.qc0, 0.0f);
                    amount       = std::clamp(amount, 0.0f, qc);
                    production   = amount / dispatch.dt_step;
                    result.tendency[2][index] -= production;
                    result.tendency[3][index] += production;
                }
                if (qr <= 0.0f && production <= 0.0f)
                    continue;
                const float tc    = temperature - dispatch.T0;
                const float es    = 611.21f * std::exp((18.678f - tc / 234.5f) * tc / (257.14f + tc));
                const float qvsat = std::clamp(0.62198f * es / std::max(pressure - es, 1.0f), 0.0f, 0.2f);
                if (qvsat <= 0.0f || qv >= qvsat)
                    continue;
                const float available   = qr + production * dispatch.dt_step;
                const float rcgs        = 1.0e-3f * rho;
                const float content     = rcgs * std::max(available, 0.0f);
                const float ventilation = (1.6f + 124.9f * std::pow(content, 0.2046f)) * std::pow(content, 0.525f);
                const float resistance  = 2.55e8f / (pressure * qvsat) + 5.4e5f;
                const float raw =
                    dispatch.evaporation_efficiency * (ventilation / resistance) * ((qvsat - qv) / (rcgs * qvsat));
                const float gamma =
                    dispatch.Lv_cp * (2.5e6f / 461.5f) * qvsat / std::max(temperature * temperature, 1.0f);
                const float saturation_bound = (qvsat - qv) / (1.0f + gamma);
                const float evaporation =
                    std::clamp(raw, 0.0f, std::min(saturation_bound / dispatch.dt_step, available / dispatch.dt_step));
                result.tendency[0][index] -= dispatch.Lv_cp * evaporation;
                result.tendency[1][index] += evaporation;
                result.tendency[3][index] -= evaporation;
            }
    return result;
}

void kessler_sedimentation_oracle(const KesslerSedimentationDispatch& dispatch, std::vector<float>& dqr)
{
    const float inv_dz = 1.0f / dispatch.dz_val;
    for (int i = dispatch.i_begin; i < dispatch.i_end; ++i)
        for (int j = dispatch.j_begin; j < dispatch.j_end; ++j)
            for (int k = dispatch.k_begin; k < dispatch.k_end; ++k)
            {
                const std::size_t index     = (static_cast<std::size_t>(i) * dispatch.nth + j) * dispatch.nz + k;
                const float       rho       = dispatch.rho_d_data[index];
                const float       rho_below = k > dispatch.k_begin ? dispatch.rho_d_data[index - 1] : rho;
                const float       rain      = std::max(dispatch.qr_data[index], 0.0f);
                if (rain <= 1.0e-9f)
                    continue;
                float velocity   = 36.34f * std::pow(1.0e-3f * rho * rain, 0.1346f);
                velocity         = std::min(velocity * std::sqrt(1.2f / rho), dispatch.Vt_max_rain);
                velocity         = std::min(velocity, dispatch.vt_cap);
                const float flux = velocity * rain;
                dqr[index] -= flux * inv_dz;
                if (k > dispatch.k_begin)
                    dqr[index - 1] += (rho / rho_below) * flux * inv_dz;
            }
}

} // namespace

TEST_CASE("Metal Kessler pointwise matches warm-rain branches and conserves water",
          "[metal][physical][gpu_parity][microphysics][kessler]")
{
    constexpr int      nr    = 6;
    constexpr int      nth   = 5;
    constexpr int      nz    = 7;
    const std::size_t  cells = static_cast<std::size_t>(nr) * nth * nz;
    std::vector<float> temperature(cells), pressure(cells), qv(cells), qc(cells), qr(cells), rho(cells),
        qh(cells, 0.0004f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t index = (static_cast<std::size_t>(i) * nth + j) * nz + k;
                temperature[index]      = 282.0f + 0.7f * k + 0.15f * i;
                pressure[index]         = 92000.0f - 850.0f * k - 50.0f * j;
                rho[index]              = 1.17f - 0.045f * k + 0.003f * i;
                switch (k % 4)
                {
                case 0:
                    qv[index] = 0.018f;
                    qc[index] = 0.0024f;
                    qr[index] = 0.0f;
                    break;
                case 1:
                    qv[index] = 0.002f;
                    qc[index] = 0.0f;
                    qr[index] = 0.0012f;
                    break;
                case 2:
                    qv[index] = 0.018f;
                    qc[index] = 0.0013f;
                    qr[index] = 0.0007f;
                    break;
                default:
                    qv[index] = 0.003f;
                    qc[index] = 0.0021f;
                    qr[index] = 0.0008f;
                    break;
                }
            }
    std::array<std::vector<float>, 6> actual;
    for (auto& field : actual)
        field.assign(cells, -12345.0f);
    KesslerPointwiseDispatch     dispatch{temperature.data(),
                                          pressure.data(),
                                          qv.data(),
                                          qc.data(),
                                          qr.data(),
                                          rho.data(),
                                          qh.data(),
                                          actual[0].data(),
                                          actual[1].data(),
                                          actual[2].data(),
                                          actual[3].data(),
                                          actual[4].data(),
                                          actual[5].data(),
                                          nr,
                                          nth,
                                          nz,
                                          1,
                                          nr - 1,
                                          1,
                                          nth - 1,
                                          1,
                                          nz - 1,
                                          0.001f,
                                          0.001f,
                                          2.2f,
                                          1.0f,
                                          2.5e6f / 1004.0f,
                                          273.15f,
                                          5.0f};
    const KesslerPointwiseOracle expected = kessler_pointwise_oracle(dispatch);

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.microphysics_kernels() == &backend);
    REQUIRE(backend.supports(MicrophysicsKernelOp::KesslerPointwise));
    REQUIRE(backend.supports(MicrophysicsKernelOp::KesslerSedimentation));
    REQUIRE(backend.supports(MicrophysicsKernelOp::LinPointwise));
    REQUIRE(backend.supports(MicrophysicsKernelOp::MicrophysicsApply));
    REQUIRE(backend.ready(KernelId::KesslerPointwise));
    REQUIRE(backend.ready(KernelId::KesslerSedimentation));
    REQUIRE(backend.pipeline(KernelId::KesslerPointwise) != 0);
    REQUIRE(backend.pipeline(KernelId::KesslerSedimentation) != 0);
    REQUIRE(backend.dispatch_kessler_pointwise(dispatch));

    double      maximum_error = 0.0;
    std::size_t active_count  = 0;
    for (std::size_t index = 0; index < cells; ++index)
        for (std::size_t field = 0; field < actual.size(); ++field)
        {
            REQUIRE(std::isfinite(actual[field][index]));
            maximum_error = std::max(
                maximum_error, std::abs(static_cast<double>(actual[field][index] - expected.tendency[field][index])));
        }
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const std::size_t index = (static_cast<std::size_t>(i) * nth + j) * nz + k;
                ++active_count;
                REQUIRE(std::abs(actual[1][index] + actual[2][index] + actual[3][index]) < 2.0e-9f);
                REQUIRE(std::abs(actual[0][index] + dispatch.Lv_cp * actual[1][index]) < 2.0e-5f);
                REQUIRE(actual[4][index] == 0.0f);
                REQUIRE(actual[5][index] == 0.0f);
            }
    REQUIRE(active_count > 0);
    REQUIRE(maximum_error < 2.0e-5);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::KesslerPointwise) == 1);

    // Kessler's MSL contract zeroes inactive pointwise outputs before reading
    // inputs. Production halo cells are therefore outside the validation
    // domain and may carry sentinels owned by another boundary stage.
    const float inactive_pressure = pressure[0];
    pressure[0]                   = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(backend.dispatch_kessler_pointwise(dispatch));
    pressure[0] = inactive_pressure;
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::KesslerPointwise) == 2);

    const auto               unchanged   = actual;
    const auto               submissions = backend.completed_command_buffer_count();
    KesslerPointwiseDispatch invalid     = dispatch;
    invalid.i_begin                      = invalid.i_end;
    REQUIRE_FALSE(backend.dispatch_kessler_pointwise(invalid));
    invalid             = dispatch;
    invalid.dqr_dt_data = invalid.dqv_dt_data;
    REQUIRE_FALSE(backend.dispatch_kessler_pointwise(invalid));
    std::vector<float> invalid_rho                                = rho;
    invalid_rho[(static_cast<std::size_t>(2) * nth + 2) * nz + 2] = 0.0f;
    invalid                                                       = dispatch;
    invalid.rho_d_data                                            = invalid_rho.data();
    REQUIRE_FALSE(backend.dispatch_kessler_pointwise(invalid));
    REQUIRE(backend.completed_command_buffer_count() == submissions);
    for (std::size_t field = 0; field < actual.size(); ++field)
        REQUIRE(std::memcmp(actual[field].data(), unchanged[field].data(), cells * sizeof(float)) == 0);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_kessler_pointwise_dispatches == 2);
    REQUIRE(record.kessler_pointwise_msl_source_hash.size() == 16);
    REQUIRE(record.kessler_sedimentation_msl_source_hash.size() == 16);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::kessler_pointwise_msl_source()) ==
            record.kessler_pointwise_msl_source_hash);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::kessler_sedimentation_msl_source()) ==
            record.kessler_sedimentation_msl_source_hash);
    std::cout << "[metal-kessler-pointwise-evidence] active=" << active_count << " max_abs=" << maximum_error
              << " msl_fnv1a64=" << record.kessler_pointwise_msl_source_hash << '\n';
}

TEST_CASE("Metal Kessler sedimentation accumulates columns with dry-mass conservation",
          "[metal][physical][gpu_parity][microphysics][kessler]")
{
    constexpr int      nr    = 6;
    constexpr int      nth   = 5;
    constexpr int      nz    = 8;
    const std::size_t  cells = static_cast<std::size_t>(nr) * nth * nz;
    std::vector<float> qr(cells), qg(cells, 0.0002f), qh(cells, 0.0001f), rho(cells);
    std::vector<float> dqr(cells), dqg(cells), dqh(cells);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t index = (static_cast<std::size_t>(i) * nth + j) * nz + k;
                qr[index]               = ((i + j + k) % 5 == 0) ? 5.0e-10f : 0.00025f * (1.0f + k);
                rho[index]              = 1.16f - 0.055f * k + 0.004f * i;
                dqr[index]              = 1.0e-5f * static_cast<float>(k - 3);
                dqg[index]              = 0.031f + 1.0e-6f * static_cast<float>(index);
                dqh[index]              = -0.017f - 2.0e-6f * static_cast<float>(index);
            }
    const std::vector<float>     initial_dqr = dqr;
    const std::vector<float>     initial_dqg = dqg;
    const std::vector<float>     initial_dqh = dqh;
    KesslerSedimentationDispatch dispatch{qr.data(),
                                          qg.data(),
                                          qh.data(),
                                          rho.data(),
                                          dqr.data(),
                                          dqg.data(),
                                          dqh.data(),
                                          nr,
                                          nth,
                                          nz,
                                          1,
                                          nr - 1,
                                          1,
                                          nth - 1,
                                          1,
                                          nz - 1,
                                          120.0f,
                                          18.0f,
                                          0.9f * 120.0f / 4.0f,
                                          4.0f};
    std::vector<float>           expected = dqr;
    kessler_sedimentation_oracle(dispatch, expected);

    MetalComputeBackend backend;
    initialize_backend(backend);
    const auto submissions = backend.completed_command_buffer_count();
    REQUIRE(backend.dispatch_kessler_sedimentation(dispatch));
    REQUIRE(backend.completed_command_buffer_count() == submissions + 1);
    double maximum_error = 0.0;
    for (std::size_t index = 0; index < cells; ++index)
    {
        maximum_error = std::max(maximum_error, std::abs(static_cast<double>(dqr[index] - expected[index])));
        REQUIRE(dqg[index] == initial_dqg[index]);
        REQUIRE(dqh[index] == initial_dqh[index]);
    }
    REQUIRE(maximum_error < 2.0e-8);

    double maximum_mass_residual = 0.0;
    for (int i = dispatch.i_begin; i < dispatch.i_end; ++i)
        for (int j = dispatch.j_begin; j < dispatch.j_end; ++j)
        {
            double weighted_delta = 0.0;
            for (int k = dispatch.k_begin; k < dispatch.k_end; ++k)
            {
                const std::size_t index = (static_cast<std::size_t>(i) * nth + j) * nz + k;
                weighted_delta += static_cast<double>(rho[index]) *
                                  static_cast<double>(dqr[index] - initial_dqr[index]) * dispatch.dz_val;
            }
            const std::size_t bottom          = (static_cast<std::size_t>(i) * nth + j) * nz + dispatch.k_begin;
            const float       rain            = std::max(qr[bottom], 0.0f);
            float             bottom_velocity = 0.0f;
            if (rain > 1.0e-9f)
            {
                bottom_velocity = 36.34f * std::pow(1.0e-3f * rho[bottom] * rain, 0.1346f);
                bottom_velocity = std::min(bottom_velocity * std::sqrt(1.2f / rho[bottom]), dispatch.Vt_max_rain);
                bottom_velocity = std::min(bottom_velocity, dispatch.vt_cap);
            }
            const double expected_sink = -static_cast<double>(rho[bottom]) * bottom_velocity * rain;
            maximum_mass_residual      = std::max(maximum_mass_residual, std::abs(weighted_delta - expected_sink));
        }
    REQUIRE(maximum_mass_residual < 2.0e-7);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::KesslerSedimentation) == 1);

    const std::vector<float>     unchanged_dqr = dqr;
    const std::vector<float>     unchanged_dqg = dqg;
    const std::vector<float>     unchanged_dqh = dqh;
    const auto                   completed     = backend.completed_command_buffer_count();
    KesslerSedimentationDispatch invalid       = dispatch;
    invalid.k_begin                            = invalid.k_end;
    REQUIRE_FALSE(backend.dispatch_kessler_sedimentation(invalid));
    invalid        = dispatch;
    invalid.dz_val = 0.0f;
    REQUIRE_FALSE(backend.dispatch_kessler_sedimentation(invalid));
    invalid             = dispatch;
    invalid.dqg_dt_data = invalid.dqr_dt_data;
    REQUIRE_FALSE(backend.dispatch_kessler_sedimentation(invalid));
    std::vector<float> invalid_rho                                = rho;
    invalid_rho[(static_cast<std::size_t>(2) * nth + 2) * nz + 3] = -1.0f;
    invalid                                                       = dispatch;
    invalid.rho_d_data                                            = invalid_rho.data();
    REQUIRE_FALSE(backend.dispatch_kessler_sedimentation(invalid));
    REQUIRE(backend.completed_command_buffer_count() == completed);
    REQUIRE(std::memcmp(dqr.data(), unchanged_dqr.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(dqg.data(), unchanged_dqg.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(dqh.data(), unchanged_dqh.data(), cells * sizeof(float)) == 0);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_kessler_sedimentation_dispatches == 1);
    std::cout << "[metal-kessler-sedimentation-evidence] columns="
              << (dispatch.i_end - dispatch.i_begin) * (dispatch.j_end - dispatch.j_begin)
              << " max_abs=" << maximum_error << " mass_residual=" << maximum_mass_residual
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal fused Kessler keeps intermediate tendencies resident and publishes atomically",
          "[metal][physical][gpu_parity][microphysics][kessler][resident]")
{
    constexpr int      nr    = 7;
    constexpr int      nth   = 6;
    constexpr int      nz    = 8;
    const std::size_t  cells = static_cast<std::size_t>(nr) * nth * nz;
    std::vector<float> temperature(cells), pressure(cells), qv(cells), qc(cells), qr(cells), qg(cells, 0.0002f),
        qh(cells, 0.0001f), rho(cells);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t index = (static_cast<std::size_t>(i) * nth + j) * nz + k;
                temperature[index]      = 280.0f + 0.9f * k + 0.1f * j;
                pressure[index]         = 93000.0f - 900.0f * k;
                rho[index]              = 1.19f - 0.052f * k + 0.002f * i;
                qv[index]               = (k % 3 == 0) ? 0.0025f : 0.016f;
                qc[index]               = (k % 2 == 0) ? 0.0022f : 0.0004f;
                qr[index]               = (k == 2) ? 5.0e-10f : 0.0003f * static_cast<float>(k + 1);
            }
    std::array<std::vector<float>, 6> fused_outputs;
    std::array<std::vector<float>, 6> separate_outputs;
    for (auto& field : fused_outputs)
        field.assign(cells, -8000.0f);
    for (auto& field : separate_outputs)
        field.assign(cells, -7000.0f);

    const auto make_pointwise = [&](std::array<std::vector<float>, 6>& outputs)
    {
        return KesslerPointwiseDispatch{temperature.data(),
                                        pressure.data(),
                                        qv.data(),
                                        qc.data(),
                                        qr.data(),
                                        rho.data(),
                                        qh.data(),
                                        outputs[0].data(),
                                        outputs[1].data(),
                                        outputs[2].data(),
                                        outputs[3].data(),
                                        outputs[4].data(),
                                        outputs[5].data(),
                                        nr,
                                        nth,
                                        nz,
                                        1,
                                        nr - 1,
                                        1,
                                        nth - 1,
                                        1,
                                        nz - 1,
                                        0.001f,
                                        0.0012f,
                                        2.0f,
                                        0.85f,
                                        2.5e6f / 1004.0f,
                                        273.15f,
                                        4.0f};
    };
    const auto make_sedimentation = [&](std::array<std::vector<float>, 6>& outputs)
    {
        return KesslerSedimentationDispatch{qr.data(),
                                            qg.data(),
                                            qh.data(),
                                            rho.data(),
                                            outputs[3].data(),
                                            outputs[4].data(),
                                            outputs[5].data(),
                                            nr,
                                            nth,
                                            nz,
                                            1,
                                            nr - 1,
                                            1,
                                            nth - 1,
                                            1,
                                            nz - 1,
                                            110.0f,
                                            20.0f,
                                            0.9f * 110.0f / 4.0f,
                                            4.0f};
    };

    MetalComputeBackend separate_backend;
    initialize_backend(separate_backend);
    KesslerPointwiseDispatch     separate_pointwise     = make_pointwise(separate_outputs);
    KesslerSedimentationDispatch separate_sedimentation = make_sedimentation(separate_outputs);
    REQUIRE(separate_backend.dispatch_kessler_pointwise(separate_pointwise));
    REQUIRE(separate_backend.dispatch_kessler_sedimentation(separate_sedimentation));
    REQUIRE(separate_backend.completed_command_buffer_count() == 2);

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.supports_kessler_fused());
    MetalKesslerFusedDispatch fused{make_pointwise(fused_outputs), make_sedimentation(fused_outputs)};
    const auto                command_buffers_before = backend.completed_command_buffer_count();
    REQUIRE(backend.dispatch_kessler_fused(fused));
    REQUIRE(backend.completed_command_buffer_count() == command_buffers_before + 1);
    REQUIRE(backend.resident_field_count() == 6);
    for (std::size_t field = 0; field < fused_outputs.size(); ++field)
        REQUIRE(std::memcmp(fused_outputs[field].data(), separate_outputs[field].data(), cells * sizeof(float)) == 0);

    const KesslerPointwiseOracle pointwise_oracle     = kessler_pointwise_oracle(fused.pointwise);
    std::vector<float>           final_rain_oracle    = pointwise_oracle.tendency[3];
    KesslerSedimentationDispatch oracle_sedimentation = fused.sedimentation;
    oracle_sedimentation.dqr_dt_data                  = final_rain_oracle.data();
    kessler_sedimentation_oracle(oracle_sedimentation, final_rain_oracle);
    double maximum_error = 0.0;
    for (std::size_t index = 0; index < cells; ++index)
    {
        maximum_error =
            std::max(maximum_error, std::abs(static_cast<double>(fused_outputs[3][index] - final_rain_oracle[index])));
        for (std::size_t field = 0; field < 3; ++field)
            maximum_error = std::max(
                maximum_error,
                std::abs(static_cast<double>(fused_outputs[field][index] - pointwise_oracle.tendency[field][index])));
    }
    REQUIRE(maximum_error < 2.0e-5);

    // Reuse the same six resident slots through their alternate candidates.
    for (auto& field : fused_outputs)
        std::fill(field.begin(), field.end(), 4444.0f);
    REQUIRE(backend.dispatch_kessler_fused(fused));
    REQUIRE(backend.completed_command_buffer_count() == command_buffers_before + 2);
    REQUIRE(backend.resident_field_count() == 6);
    for (std::size_t field = 0; field < fused_outputs.size(); ++field)
        REQUIRE(std::memcmp(fused_outputs[field].data(), separate_outputs[field].data(), cells * sizeof(float)) == 0);

    const auto                unchanged                  = fused_outputs;
    const auto                completed_before_rejection = backend.completed_command_buffer_count();
    MetalKesslerFusedDispatch invalid                    = fused;
    invalid.sedimentation.j_end                          = invalid.sedimentation.j_begin;
    REQUIRE_FALSE(backend.dispatch_kessler_fused(invalid));
    invalid                           = fused;
    invalid.sedimentation.dqr_dt_data = invalid.pointwise.dqv_dt_data;
    REQUIRE_FALSE(backend.dispatch_kessler_fused(invalid));
    invalid                          = fused;
    std::vector<float> different_rho = rho;
    invalid.sedimentation.rho_d_data = different_rho.data();
    REQUIRE_FALSE(backend.dispatch_kessler_fused(invalid));
    REQUIRE(backend.completed_command_buffer_count() == completed_before_rejection);
    for (std::size_t field = 0; field < fused_outputs.size(); ++field)
        REQUIRE(std::memcmp(fused_outputs[field].data(), unchanged[field].data(), cells * sizeof(float)) == 0);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_kessler_fused_dispatches == 2);
    REQUIRE(record.completed_kessler_pointwise_dispatches == 2);
    REQUIRE(record.completed_kessler_sedimentation_dispatches == 2);
    REQUIRE(record.kessler_resident_tendency_writes == 12);
    std::cout << "[metal-kessler-fused-evidence] fused_submissions=" << record.completed_kessler_fused_dispatches
              << " resident_fields=" << backend.resident_field_count()
              << " resident_writes=" << record.kessler_resident_tendency_writes << " max_abs=" << maximum_error
              << " device_ms=" << record.total_device_milliseconds << '\n';
    backend.invalidate_resident_field("kessler_dqv_dt");
    REQUIRE(backend.resident_field_count() == 6);
    backend.invalidate_all_resident_fields();
    REQUIRE(backend.resident_field_count() == 6);
    backend.shutdown();
    REQUIRE(backend.resident_field_count() == 0);
}

namespace
{

struct MicrophysicsApplyOracle
{
    std::array<std::vector<float>, 8> state;
    std::vector<unsigned int>         stats;
    std::vector<unsigned char>        theta_clamped;
    std::vector<float>                theta_new;
    std::vector<float>                theta_bounded;
    std::vector<float>                dtheta_total;
    std::vector<float>                dtheta_limited;
};

unsigned int float_bits(float value)
{
    unsigned int bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float telemetry requires 32-bit words");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_float(unsigned int bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

MicrophysicsApplyOracle microphysics_apply_oracle(const MicrophysicsApplyDispatch& dispatch)
{
    const std::size_t       cells           = static_cast<std::size_t>(dispatch.nx) * dispatch.ny * dispatch.nz;
    const std::size_t       stats_needed    = static_cast<std::size_t>(10 + dispatch.nz + dispatch.nx);
    float*                  source_state[8] = {dispatch.theta_data,
                                               dispatch.qv_data,
                                               dispatch.qc_data,
                                               dispatch.qr_data,
                                               dispatch.qi_data,
                                               dispatch.qs_data,
                                               dispatch.qg_data,
                                               dispatch.qh_data};
    const float*            tendency[8]     = {dispatch.dtheta,
                                               dispatch.dqv,
                                               dispatch.dqc,
                                               dispatch.dqr,
                                               dispatch.dqi,
                                               dispatch.dqs,
                                               dispatch.dqg,
                                               dispatch.dqh};
    MicrophysicsApplyOracle result;
    for (int field = 0; field < 8; ++field)
        result.state[field].assign(source_state[field], source_state[field] + cells);
    result.stats.assign(stats_needed, 0u);
    result.theta_clamped.assign(cells, 0u);
    result.theta_new.assign(cells, 0.0f);
    result.theta_bounded.assign(cells, 0.0f);
    result.dtheta_total.assign(cells, 0.0f);
    result.dtheta_limited.assign(cells, 0.0f);

    for (int i = dispatch.i_begin; i < dispatch.i_end; ++i)
        for (int j = dispatch.j_begin; j < dispatch.j_end; ++j)
            for (int k = 0; k < dispatch.nz; ++k)
            {
                const std::size_t index = (static_cast<std::size_t>(i) * dispatch.ny + j) * dispatch.nz + k;
                float             guarded[8];
                unsigned int      nonfinite = 0;
                for (int field = 0; field < 8; ++field)
                {
                    guarded[field] = tendency[field][index];
                    if (!std::isfinite(guarded[field]))
                    {
                        guarded[field] = 0.0f;
                        ++nonfinite;
                    }
                }
                result.stats[0] += nonfinite;
                float total = (guarded[0] + dispatch.dtheta_rad[index] + dispatch.dtheta_pbl[index]) * dispatch.dt_step;
                if (!std::isfinite(total))
                {
                    total = 0.0f;
                    ++result.stats[1];
                }
                const float limited = std::clamp(total, -dispatch.max_theta_step, dispatch.max_theta_step);
                if (limited != total)
                    ++result.stats[2];
                const float old_theta        = result.state[0][index];
                const float theta_new        = old_theta + limited;
                const float theta_safe       = std::isfinite(theta_new)
                                                   ? theta_new
                                                   : (std::isfinite(old_theta) ? old_theta : dispatch.theta_fallback);
                const float theta_bounded    = std::clamp(theta_safe, dispatch.theta_min, dispatch.theta_max);
                const bool  clamped          = theta_bounded != theta_new || !std::isfinite(theta_new);
                result.theta_new[index]      = theta_new;
                result.theta_bounded[index]  = theta_bounded;
                result.dtheta_total[index]   = total;
                result.dtheta_limited[index] = limited;
                result.theta_clamped[index]  = clamped ? 1u : 0u;
                result.state[0][index]       = theta_bounded;
                if (clamped)
                {
                    ++result.stats[3];
                    ++result.stats[10 + static_cast<std::size_t>(k)];
                    ++result.stats[10 + static_cast<std::size_t>(dispatch.nz) + static_cast<std::size_t>(i)];
                    if (result.stats[4] == 0u)
                    {
                        result.stats[4] = 1u;
                        result.stats[5] = static_cast<unsigned int>(index);
                        result.stats[6] = float_bits(theta_new);
                        result.stats[7] = float_bits(theta_bounded);
                        result.stats[8] = float_bits(total);
                        result.stats[9] = float_bits(limited);
                    }
                }

                const float qv_new = result.state[1][index] + (guarded[1] + dispatch.dqv_pbl[index]) * dispatch.dt_step;
                result.state[1][index] = std::isfinite(qv_new) ? std::clamp(qv_new, 0.0f, dispatch.qv_max_val) : 0.0f;
                for (int field = 2; field < 8; ++field)
                {
                    const float updated = result.state[field][index] + guarded[field] * dispatch.dt_step;
                    result.state[field][index] =
                        std::isfinite(updated) ? std::clamp(updated, 0.0f, dispatch.hydro_max_val) : 0.0f;
                }
            }
    return result;
}

} // namespace

TEST_CASE("Metal Lin split and fused kernels match the canonical mixed-phase CPU scheme",
          "[metal][physical][gpu_parity][microphysics][lin][resident][apply]")
{
    constexpr int     nx = 6, ny = 5, nz = 24;
    constexpr float   dz_value = 400.0f, dt_step = 2.0f;
    const std::size_t cells = static_cast<std::size_t>(nx) * ny * nz;
    Field3D           theta_f(nx, ny, nz), pressure_f(nx, ny, nz), rho_f(nx, ny, nz);
    Field3D           qv_f(nx, ny, nz), qc_f(nx, ny, nz), qr_f(nx, ny, nz), qi_f(nx, ny, nz);
    Field3D           qs_f(nx, ny, nz), qg_f(nx, ny, nz), qh_f(nx, ny, nz);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double z          = (static_cast<double>(k) + 0.5) * dz_value;
                const double horizontal = 0.1 * std::sin(0.7 * i) * std::cos(0.5 * j);
                theta_f(i, j, k)        = static_cast<float>(300.0 + 0.004 * z + horizontal);
                pressure_f(i, j, k)     = static_cast<float>(100000.0 * std::exp(-z / 8500.0));
                rho_f(i, j, k)          = static_cast<float>(1.2 * std::exp(-z / 8500.0) * (1.0 + 0.02 * horizontal));
                qv_f(i, j, k)           = static_cast<float>(0.012 * std::exp(-z / 3000.0));
                qc_f(i, j, k)           = static_cast<float>(0.0020 + 0.0005 * horizontal);
                qr_f(i, j, k)           = static_cast<float>(0.0010 * std::exp(-z / 5000.0));
                qi_f(i, j, k)           = static_cast<float>(z > 4000.0 ? 0.0015 : 0.0);
                qs_f(i, j, k)           = static_cast<float>(z > 4000.0 ? 0.0008 : 0.0);
                qg_f(i, j, k)           = static_cast<float>(z > 3000.0 ? 0.0006 : 0.0);
                qh_f(i, j, k)           = static_cast<float>(z > 3000.0 ? 0.0003 : 0.0);
            }

    LinScheme oracle_scheme;
    oracle_scheme.set_vertical_spacing_m(dz_value);
    Field3D cpu_dtheta, cpu_dqv, cpu_dqc, cpu_dqr, cpu_dqi, cpu_dqs, cpu_dqg, cpu_dqh;
    oracle_scheme.compute_tendencies(pressure_f,
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
                                     cpu_dtheta,
                                     cpu_dqv,
                                     cpu_dqc,
                                     cpu_dqr,
                                     cpu_dqi,
                                     cpu_dqs,
                                     cpu_dqg,
                                     cpu_dqh);

    std::array<std::vector<float>, 8> split_outputs, fused_outputs;
    for (auto& field : split_outputs)
        field.assign(cells, 0.0f);
    for (auto& field : fused_outputs)
        field.assign(cells, std::numeric_limits<float>::quiet_NaN());
    const auto make_pointwise = [&](std::array<std::vector<float>, 8>& out)
    {
        return LinPointwiseDispatch{theta_f.data(),
                                    pressure_f.data(),
                                    qv_f.data(),
                                    qc_f.data(),
                                    qr_f.data(),
                                    qi_f.data(),
                                    qs_f.data(),
                                    qg_f.data(),
                                    qh_f.data(),
                                    out[0].data(),
                                    out[1].data(),
                                    out[2].data(),
                                    out[3].data(),
                                    out[4].data(),
                                    out[5].data(),
                                    out[6].data(),
                                    out[7].data(),
                                    nx,
                                    ny,
                                    nz,
                                    dt_step,
                                    1.0f / dt_step,
                                    1.0e-3f,
                                    1.0e-3f,
                                    2.2f,
                                    1.0e-3f,
                                    1.0e-3f,
                                    1.0f,
                                    1.0e-3f,
                                    static_cast<float>(microphysics_constants::c_melt_snow_per_K),
                                    static_cast<float>(microphysics_constants::c_melt_graupel_per_K),
                                    static_cast<float>(microphysics_constants::c_melt_hail_per_K),
                                    1.0e-3f,
                                    static_cast<float>(microphysics_constants::L_v / microphysics_constants::cp),
                                    static_cast<float>(microphysics_constants::L_f / microphysics_constants::cp),
                                    static_cast<float>(microphysics_constants::L_s / microphysics_constants::cp)};
    };
    const auto make_sedimentation = [&](std::array<std::vector<float>, 8>& out)
    {
        return LinSedimentationDispatch{qr_f.data(),
                                        qs_f.data(),
                                        qg_f.data(),
                                        qh_f.data(),
                                        rho_f.data(),
                                        out[3].data(),
                                        out[5].data(),
                                        out[6].data(),
                                        out[7].data(),
                                        nx,
                                        ny,
                                        nz,
                                        dz_value,
                                        0.9f * dz_value / dt_step};
    };

    MetalComputeBackend split_backend;
    initialize_backend(split_backend);
    REQUIRE(split_backend.supports(MicrophysicsKernelOp::LinPointwise));
    REQUIRE(split_backend.supports(MicrophysicsKernelOp::LinSedimentation));
    REQUIRE(split_backend.supports(MicrophysicsKernelOp::LinFused));
    REQUIRE(split_backend.ready(KernelId::LinPointwise));
    REQUIRE(split_backend.ready(KernelId::LinSedimentation));
    REQUIRE(split_backend.pipeline(KernelId::LinPointwise) != 0);
    REQUIRE(split_backend.pipeline(KernelId::LinSedimentation) != 0);
    REQUIRE(split_backend.dispatch_lin_pointwise(make_pointwise(split_outputs)));
    REQUIRE(split_backend.dispatch_lin_sedimentation(make_sedimentation(split_outputs)));
    REQUIRE(split_backend.completed_command_buffer_count() == 2);

    MetalComputeBackend backend;
    initialize_backend(backend);
    backend.begin_resident_step();
    LinFusedDispatch fused{make_pointwise(fused_outputs), make_sedimentation(fused_outputs)};
    REQUIRE(backend.dispatch_lin_fused(fused));
    REQUIRE(backend.completed_command_buffer_count() == 1);
    REQUIRE(backend.resident_field_count() == 8);
    for (std::size_t field = 0; field < 8; ++field)
        REQUIRE(std::memcmp(split_outputs[field].data(), fused_outputs[field].data(), cells * sizeof(float)) == 0);

    const float*          cpu[8]                = {cpu_dtheta.data(),
                                                   cpu_dqv.data(),
                                                   cpu_dqc.data(),
                                                   cpu_dqr.data(),
                                                   cpu_dqi.data(),
                                                   cpu_dqs.data(),
                                                   cpu_dqg.data(),
                                                   cpu_dqh.data()};
    double                maximum_heating_error = 0.0, maximum_species_error = 0.0;
    std::array<double, 8> maximum_magnitude{};
    for (std::size_t field = 0; field < 8; ++field)
        for (std::size_t index = 0; index < cells; ++index)
        {
            const double expected = cpu[field][index], actual = fused_outputs[field][index];
            const double error       = std::abs(actual - expected);
            maximum_magnitude[field] = std::max(maximum_magnitude[field], std::abs(actual));
            if (field == 0)
            {
                maximum_heating_error = std::max(maximum_heating_error, error);
                REQUIRE(error <= 6.0e-4 + 1.0e-4 * std::abs(expected));
            }
            else
            {
                maximum_species_error = std::max(maximum_species_error, error);
                REQUIRE(error <= 1.0e-7 + 1.0e-4 * std::abs(expected));
            }
        }
    for (double magnitude : maximum_magnitude)
        REQUIRE(magnitude > 1.0e-10);

    const auto       unchanged        = fused_outputs;
    const auto       completed_before = backend.completed_command_buffer_count();
    LinFusedDispatch invalid          = fused;
    invalid.sedimentation.nz -= 1;
    REQUIRE_FALSE(backend.dispatch_lin_fused(invalid));
    invalid                  = fused;
    invalid.sedimentation.qs = invalid.pointwise.qr;
    REQUIRE_FALSE(backend.dispatch_lin_fused(invalid));
    std::vector<float> invalid_rho(rho_f.data(), rho_f.data() + cells);
    invalid_rho[cells / 2]      = 0.0f;
    invalid                     = fused;
    invalid.sedimentation.rho_d = invalid_rho.data();
    REQUIRE_FALSE(backend.dispatch_lin_fused(invalid));
    LinPointwiseDispatch invalid_point = fused.pointwise;
    invalid_point.inv_dt               = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_FALSE(backend.dispatch_lin_pointwise(invalid_point));
    LinSedimentationDispatch invalid_sed = fused.sedimentation;
    invalid_sed.dz                       = 0.0f;
    REQUIRE_FALSE(backend.dispatch_lin_sedimentation(invalid_sed));
    REQUIRE(backend.completed_command_buffer_count() == completed_before);
    for (std::size_t field = 0; field < 8; ++field)
        REQUIRE(std::memcmp(unchanged[field].data(), fused_outputs[field].data(), cells * sizeof(float)) == 0);

    std::array<std::vector<float>, 8> state;
    state[0].assign(theta_f.data(), theta_f.data() + cells);
    state[1].assign(qv_f.data(), qv_f.data() + cells);
    state[2].assign(qc_f.data(), qc_f.data() + cells);
    state[3].assign(qr_f.data(), qr_f.data() + cells);
    state[4].assign(qi_f.data(), qi_f.data() + cells);
    state[5].assign(qs_f.data(), qs_f.data() + cells);
    state[6].assign(qg_f.data(), qg_f.data() + cells);
    state[7].assign(qh_f.data(), qh_f.data() + cells);
    std::vector<float>        zero(cells, 0.0f);
    std::vector<unsigned int> stats(10 + nz + nx, 0u);
    MicrophysicsApplyDispatch apply{};
    apply.theta_data                             = state[0].data();
    apply.qv_data                                = state[1].data();
    apply.qc_data                                = state[2].data();
    apply.qr_data                                = state[3].data();
    apply.qi_data                                = state[4].data();
    apply.qs_data                                = state[5].data();
    apply.qg_data                                = state[6].data();
    apply.qh_data                                = state[7].data();
    apply.dtheta                                 = fused_outputs[0].data();
    apply.dqv                                    = fused_outputs[1].data();
    apply.dqc                                    = fused_outputs[2].data();
    apply.dqr                                    = fused_outputs[3].data();
    apply.dqi                                    = fused_outputs[4].data();
    apply.dqs                                    = fused_outputs[5].data();
    apply.dqg                                    = fused_outputs[6].data();
    apply.dqh                                    = fused_outputs[7].data();
    apply.dtheta_rad                             = zero.data();
    apply.dtheta_pbl                             = zero.data();
    apply.dqv_pbl                                = zero.data();
    apply.nx                                     = nx;
    apply.ny                                     = ny;
    apply.nz                                     = nz;
    apply.i_begin                                = 0;
    apply.i_end                                  = nx;
    apply.j_begin                                = 0;
    apply.j_end                                  = ny;
    apply.dt_step                                = dt_step;
    apply.theta_fallback                         = 300.0f;
    apply.theta_min                              = 150.0f;
    apply.theta_max                              = 700.0f;
    apply.max_theta_step                         = 20.0f;
    apply.qv_max_val                             = 0.2f;
    apply.hydro_max_val                          = 0.2f;
    apply.stats_out                              = stats.data();
    apply.stats_len                              = static_cast<int>(stats.size());
    const MicrophysicsApplyOracle expected_apply = microphysics_apply_oracle(apply);
    REQUIRE(backend.dispatch_microphysics_apply(apply));
    for (std::size_t field = 0; field < 8; ++field)
        for (std::size_t index = 0; index < cells; ++index)
            REQUIRE(std::abs(state[field][index] - expected_apply.state[field][index]) < 1.0e-6f);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_lin_fused_dispatches == 1);
    REQUIRE(record.completed_lin_pointwise_dispatches == 1);
    REQUIRE(record.completed_lin_sedimentation_dispatches == 1);
    REQUIRE(record.lin_resident_tendency_writes == 8);
    REQUIRE(record.microphysics_apply_resident_tendency_reuses == 8);
    REQUIRE(record.lin_pointwise_msl_source_hash.size() == 16);
    REQUIRE(record.lin_sedimentation_msl_source_hash.size() == 16);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::lin_pointwise_msl_source()) == record.lin_pointwise_msl_source_hash);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::lin_sedimentation_msl_source()) ==
            record.lin_sedimentation_msl_source_hash);
    std::cout << "[metal-lin-evidence] mixed_phase_cells=" << cells
              << " split_command_buffers=2 fused_command_buffers=1 resident_apply_reuses=8"
              << " max_heating_abs=" << maximum_heating_error << " max_species_abs=" << maximum_species_error
              << " pointwise_msl_fnv1a64=" << record.lin_pointwise_msl_source_hash
              << " sedimentation_msl_fnv1a64=" << record.lin_sedimentation_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal MicrophysicsApply matches guarded state and telemetry semantics",
          "[metal][physical][gpu_parity][microphysics][apply]")
{
    constexpr int                     nx    = 8;
    constexpr int                     ny    = 7;
    constexpr int                     nz    = 6;
    const std::size_t                 cells = static_cast<std::size_t>(nx) * ny * nz;
    std::array<std::vector<float>, 8> state;
    std::array<std::vector<float>, 8> tendency;
    for (std::size_t field = 0; field < state.size(); ++field)
    {
        state[field].resize(cells);
        tendency[field].resize(cells);
        for (std::size_t index = 0; index < cells; ++index)
        {
            state[field][index] = field == 0 ? 295.0f + 0.01f * static_cast<float>(index)
                                             : 0.002f * static_cast<float>(field) + 1.0e-7f * static_cast<float>(index);
            tendency[field][index] = field == 0
                                         ? 0.15f * static_cast<float>(static_cast<int>(index % 5) - 2)
                                         : 2.0e-5f * static_cast<float>(static_cast<int>((index + field) % 5) - 2);
        }
    }
    std::vector<float> dtheta_rad(cells, 0.02f);
    std::vector<float> dtheta_pbl(cells, -0.01f);
    std::vector<float> dqv_pbl(cells, 1.0e-5f);
    const auto         flat = [](int i, int j, int k) { return (static_cast<std::size_t>(i) * ny + j) * nz + k; };
    const std::size_t  limited_cell            = flat(2, 2, 1);
    const std::size_t  bounded_cell            = flat(3, 2, 2);
    const std::size_t  nonfinite_tendency_cell = flat(2, 3, 3);
    const std::size_t  nonfinite_total_cell    = flat(4, 3, 4);
    const std::size_t  nonfinite_theta_cell    = flat(5, 4, 2);
    tendency[0][limited_cell]                  = 80.0f;
    state[0][bounded_cell]                     = 695.0f;
    tendency[0][bounded_cell]                  = 30.0f;
    for (auto& field : tendency)
        field[nonfinite_tendency_cell] = std::numeric_limits<float>::quiet_NaN();
    dtheta_rad[nonfinite_total_cell] = std::numeric_limits<float>::infinity();
    state[0][nonfinite_theta_cell]   = std::numeric_limits<float>::quiet_NaN();
    state[1][flat(3, 3, 1)]          = 0.099f;
    tendency[1][flat(3, 3, 1)]       = 0.02f;
    state[2][flat(3, 3, 2)]          = 0.001f;
    tendency[2][flat(3, 3, 2)]       = -0.02f;
    dqv_pbl[flat(4, 2, 1)]           = std::numeric_limits<float>::infinity();

    const int                 stats_needed = 10 + nz + nx;
    std::vector<unsigned int> stats(static_cast<std::size_t>(stats_needed + 3), UINT32_C(0xdeadbeef));
    MicrophysicsApplyDispatch dispatch{};
    dispatch.theta_data                                   = state[0].data();
    dispatch.qv_data                                      = state[1].data();
    dispatch.qc_data                                      = state[2].data();
    dispatch.qr_data                                      = state[3].data();
    dispatch.qi_data                                      = state[4].data();
    dispatch.qs_data                                      = state[5].data();
    dispatch.qg_data                                      = state[6].data();
    dispatch.qh_data                                      = state[7].data();
    dispatch.dtheta                                       = tendency[0].data();
    dispatch.dqv                                          = tendency[1].data();
    dispatch.dqc                                          = tendency[2].data();
    dispatch.dqr                                          = tendency[3].data();
    dispatch.dqi                                          = tendency[4].data();
    dispatch.dqs                                          = tendency[5].data();
    dispatch.dqg                                          = tendency[6].data();
    dispatch.dqh                                          = tendency[7].data();
    dispatch.dtheta_rad                                   = dtheta_rad.data();
    dispatch.dtheta_pbl                                   = dtheta_pbl.data();
    dispatch.dqv_pbl                                      = dqv_pbl.data();
    dispatch.nx                                           = nx;
    dispatch.ny                                           = ny;
    dispatch.nz                                           = nz;
    dispatch.i_begin                                      = 1;
    dispatch.i_end                                        = nx - 1;
    dispatch.j_begin                                      = 1;
    dispatch.j_end                                        = ny - 1;
    dispatch.dt_step                                      = 2.0f;
    dispatch.theta_fallback                               = 300.0f;
    dispatch.theta_min                                    = 150.0f;
    dispatch.theta_max                                    = 700.0f;
    dispatch.max_theta_step                               = 20.0f;
    dispatch.qv_max_val                                   = 0.1f;
    dispatch.hydro_max_val                                = 0.1f;
    dispatch.stats_out                                    = stats.data();
    dispatch.stats_len                                    = static_cast<int>(stats.size());
    const std::array<std::vector<float>, 8> initial_state = state;
    const MicrophysicsApplyOracle           expected      = microphysics_apply_oracle(dispatch);

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.supports_microphysics_apply());
    REQUIRE(backend.supports(MicrophysicsKernelOp::MicrophysicsApply));
    REQUIRE(backend.ready(KernelId::MicrophysicsApply));
    REQUIRE(backend.pipeline(KernelId::MicrophysicsApply) != 0);
    REQUIRE(backend.dispatch_microphysics_apply(dispatch));
    double maximum_error = 0.0;
    for (std::size_t field = 0; field < state.size(); ++field)
        for (std::size_t index = 0; index < cells; ++index)
        {
            REQUIRE(std::isfinite(state[field][index]));
            maximum_error = std::max(maximum_error,
                                     std::abs(static_cast<double>(state[field][index] - expected.state[field][index])));
            const int k   = static_cast<int>(index % nz);
            const int j   = static_cast<int>((index / nz) % ny);
            const int i   = static_cast<int>(index / (ny * nz));
            if (i < dispatch.i_begin || i >= dispatch.i_end || j < dispatch.j_begin || j >= dispatch.j_end)
            {
                REQUIRE(std::memcmp(&state[field][index], &initial_state[field][index], sizeof(float)) == 0);
            }
            (void)k;
        }
    REQUIRE(maximum_error < 1.0e-6);
    for (std::size_t index = 0; index < 5; ++index)
        REQUIRE(stats[index] == expected.stats[index]);
    for (std::size_t index = 10; index < expected.stats.size(); ++index)
        REQUIRE(stats[index] == expected.stats[index]);
    REQUIRE(stats[0] == 8u);
    REQUIRE(stats[1] == 1u);
    REQUIRE(stats[2] >= 2u);
    REQUIRE(stats[3] >= 2u);
    REQUIRE(stats[5] < cells);
    REQUIRE(expected.theta_clamped[stats[5]] != 0u);
    const float reported_theta_new = bits_float(stats[6]);
    if (std::isfinite(expected.theta_new[stats[5]]))
        REQUIRE(reported_theta_new == expected.theta_new[stats[5]]);
    else
        REQUIRE_FALSE(std::isfinite(reported_theta_new));
    REQUIRE(stats[7] == float_bits(state[0][stats[5]]));
    REQUIRE(bits_float(stats[7]) == expected.theta_bounded[stats[5]]);
    REQUIRE(bits_float(stats[8]) == expected.dtheta_total[stats[5]]);
    REQUIRE(bits_float(stats[9]) == expected.dtheta_limited[stats[5]]);
    REQUIRE(stats[static_cast<std::size_t>(stats_needed)] == UINT32_C(0xdeadbeef));
    REQUIRE(stats[static_cast<std::size_t>(stats_needed + 1)] == UINT32_C(0xdeadbeef));
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::MicrophysicsApply) == 1);
    REQUIRE(backend.resident_field_count() == 8);

    const auto                unchanged_state = state;
    const auto                unchanged_stats = stats;
    const auto                completed       = backend.completed_command_buffer_count();
    MicrophysicsApplyDispatch invalid         = dispatch;
    invalid.i_begin                           = invalid.i_end;
    REQUIRE_FALSE(backend.dispatch_microphysics_apply(invalid));
    invalid           = dispatch;
    invalid.stats_len = stats_needed - 1;
    REQUIRE_FALSE(backend.dispatch_microphysics_apply(invalid));
    invalid         = dispatch;
    invalid.qv_data = invalid.theta_data;
    REQUIRE_FALSE(backend.dispatch_microphysics_apply(invalid));
    invalid                = dispatch;
    invalid.max_theta_step = -1.0f;
    REQUIRE_FALSE(backend.dispatch_microphysics_apply(invalid));
    REQUIRE(backend.completed_command_buffer_count() == completed);
    for (std::size_t field = 0; field < state.size(); ++field)
        REQUIRE(std::memcmp(state[field].data(), unchanged_state[field].data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(stats.data(), unchanged_stats.data(), stats.size() * sizeof(unsigned int)) == 0);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_microphysics_apply_dispatches == 1);
    REQUIRE(record.microphysics_apply_msl_source_hash.size() == 16);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::microphysics_apply_msl_source()) ==
            record.microphysics_apply_msl_source_hash);
    std::cout << "[metal-microphysics-apply-evidence] max_abs=" << maximum_error << " nonfinite_reads=" << stats[0]
              << " limited=" << stats[2] << " theta_clamps=" << stats[3]
              << " msl_fnv1a64=" << record.microphysics_apply_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal MicrophysicsApply selectively reuses fused Kessler resident tendencies",
          "[metal][physical][gpu_parity][microphysics][kessler][apply][resident]")
{
    constexpr int                     nx    = 7;
    constexpr int                     ny    = 6;
    constexpr int                     nz    = 7;
    const std::size_t                 cells = static_cast<std::size_t>(nx) * ny * nz;
    std::vector<float>                temperature(cells), pressure(cells), rho(cells);
    std::array<std::vector<float>, 8> state;
    for (auto& field : state)
        field.resize(cells);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t index = (static_cast<std::size_t>(i) * ny + j) * nz + k;
                temperature[index]      = 283.0f + 0.8f * k;
                pressure[index]         = 92500.0f - 920.0f * k;
                rho[index]              = 1.18f - 0.05f * k;
                state[0][index]         = 300.0f + 0.1f * k;
                state[1][index]         = k % 3 == 0 ? 0.0025f : 0.014f;
                state[2][index]         = k % 2 == 0 ? 0.002f : 0.0003f;
                state[3][index]         = 0.00025f * static_cast<float>(k + 1);
                state[4][index]         = 0.0f;
                state[5][index]         = 0.0f;
                state[6][index]         = 0.0001f;
                state[7][index]         = 0.00005f;
            }
    std::array<std::vector<float>, 6> kessler_tendency;
    for (auto& field : kessler_tendency)
        field.assign(cells, -9.0f);
    KesslerPointwiseDispatch     pointwise{temperature.data(),
                                           pressure.data(),
                                           state[1].data(),
                                           state[2].data(),
                                           state[3].data(),
                                           rho.data(),
                                           state[7].data(),
                                           kessler_tendency[0].data(),
                                           kessler_tendency[1].data(),
                                           kessler_tendency[2].data(),
                                           kessler_tendency[3].data(),
                                           kessler_tendency[4].data(),
                                           kessler_tendency[5].data(),
                                           nx,
                                           ny,
                                           nz,
                                           1,
                                           nx - 1,
                                           1,
                                           ny - 1,
                                           1,
                                           nz - 1,
                                           0.001f,
                                           0.001f,
                                           2.1f,
                                           0.9f,
                                           2.5e6f / 1004.0f,
                                           273.15f,
                                           3.0f};
    KesslerSedimentationDispatch sedimentation{state[3].data(),
                                               state[6].data(),
                                               state[7].data(),
                                               rho.data(),
                                               kessler_tendency[3].data(),
                                               kessler_tendency[4].data(),
                                               kessler_tendency[5].data(),
                                               nx,
                                               ny,
                                               nz,
                                               1,
                                               nx - 1,
                                               1,
                                               ny - 1,
                                               1,
                                               nz - 1,
                                               100.0f,
                                               18.0f,
                                               30.0f,
                                               3.0f};

    MetalComputeBackend backend;
    initialize_backend(backend);
    backend.begin_resident_step();
    REQUIRE(backend.dispatch_kessler_fused({pointwise, sedimentation}));
    REQUIRE(backend.resident_field_count() == 6);

    std::vector<float>        zero(cells, 0.0f);
    std::vector<unsigned int> stats(static_cast<std::size_t>(10 + nz + nx), 99u);
    MicrophysicsApplyDispatch apply{};
    apply.theta_data                            = state[0].data();
    apply.qv_data                               = state[1].data();
    apply.qc_data                               = state[2].data();
    apply.qr_data                               = state[3].data();
    apply.qi_data                               = state[4].data();
    apply.qs_data                               = state[5].data();
    apply.qg_data                               = state[6].data();
    apply.qh_data                               = state[7].data();
    apply.dtheta                                = kessler_tendency[0].data();
    apply.dqv                                   = kessler_tendency[1].data();
    apply.dqc                                   = kessler_tendency[2].data();
    apply.dqr                                   = kessler_tendency[3].data();
    apply.dqi                                   = zero.data();
    apply.dqs                                   = zero.data();
    apply.dqg                                   = kessler_tendency[4].data();
    apply.dqh                                   = kessler_tendency[5].data();
    apply.dtheta_rad                            = zero.data();
    apply.dtheta_pbl                            = zero.data();
    apply.dqv_pbl                               = zero.data();
    apply.nx                                    = nx;
    apply.ny                                    = ny;
    apply.nz                                    = nz;
    apply.i_begin                               = 1;
    apply.i_end                                 = nx - 1;
    apply.j_begin                               = 1;
    apply.j_end                                 = ny - 1;
    apply.dt_step                               = 3.0f;
    apply.theta_fallback                        = 300.0f;
    apply.theta_min                             = 150.0f;
    apply.theta_max                             = 700.0f;
    apply.max_theta_step                        = 20.0f;
    apply.qv_max_val                            = 0.1f;
    apply.hydro_max_val                         = 0.1f;
    apply.stats_out                             = stats.data();
    apply.stats_len                             = static_cast<int>(stats.size());
    const auto                    initial_state = state;
    const MicrophysicsApplyOracle expected      = microphysics_apply_oracle(apply);
    REQUIRE(backend.dispatch_microphysics_apply(apply));
    REQUIRE(backend.resident_field_count() == 14);
    for (std::size_t field = 0; field < state.size(); ++field)
        for (std::size_t index = 0; index < cells; ++index)
            REQUIRE(std::abs(state[field][index] - expected.state[field][index]) < 1.0e-6f);
    auto record = backend.execution_record();
    REQUIRE(record.microphysics_apply_resident_tendency_reuses == 6);

    backend.invalidate_resident_field("kessler_dqc_dt");
    state = initial_state;
    std::fill(stats.begin(), stats.end(), 77u);
    REQUIRE(backend.dispatch_microphysics_apply(apply));
    for (std::size_t field = 0; field < state.size(); ++field)
        for (std::size_t index = 0; index < cells; ++index)
            REQUIRE(std::abs(state[field][index] - expected.state[field][index]) < 1.0e-6f);
    record = backend.execution_record();
    REQUIRE(record.completed_microphysics_apply_dispatches == 2);
    REQUIRE(record.microphysics_apply_resident_tendency_reuses == 11);
    REQUIRE(record.completed_command_buffers == 3);
    std::cout << "[metal-kessler-apply-residency-evidence] resident_fields=" << backend.resident_field_count()
              << " first_reuses=6 selective_total_reuses=" << record.microphysics_apply_resident_tendency_reuses
              << " command_buffers=" << record.completed_command_buffers
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal Cartesian X and Y advection match signed upwind CPU operators",
          "[metal][physical][gpu_parity][advection]")
{
    constexpr int      nr    = 10;
    constexpr int      nth   = 9;
    constexpr int      nz    = 7;
    constexpr float    dx    = 125.0f;
    constexpr float    dy    = 160.0f;
    constexpr float    dt    = 2.5f;
    const std::size_t  cells = static_cast<std::size_t>(nr) * nth * nz;
    std::vector<float> source(cells), velocity_x(cells), velocity_y(cells);
    const auto         flat = [](int i, int j, int k) { return (static_cast<std::size_t>(i) * nth + j) * nz + k; };
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t index = flat(i, j, k);
                const float       x     = static_cast<float>(i - 4);
                const float       y     = static_cast<float>(j - 3);
                source[index]           = 300.0f + 0.17f * x * x - 0.11f * y * y + 0.07f * x * y + 0.03f * k;
                const int x_branch      = (i + 2 * j + k) % 3;
                const int y_branch      = (2 * i + j + k) % 3;
                velocity_x[index]       = x_branch == 0 ? -6.5f : (x_branch == 1 ? 0.0f : 8.25f);
                velocity_y[index]       = y_branch == 0 ? 5.75f : (y_branch == 1 ? -7.0f : 0.0f);
            }

    std::vector<float> expected_x = source;
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const std::size_t center   = flat(i, j, k);
                const double      speed    = velocity_x[center];
                double            gradient = 0.0;
                if (speed > 0.0)
                    gradient = (static_cast<double>(source[center]) - source[flat(i - 1, j, k)]) / dx;
                else if (speed < 0.0)
                    gradient = (static_cast<double>(source[flat(i + 1, j, k)]) - source[center]) / dx;
                expected_x[center] = static_cast<float>(static_cast<double>(source[center]) - dt * speed * gradient);
            }
    std::vector<float> expected_y = source;
    for (int i = 0; i < nr; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const std::size_t center   = flat(i, j, k);
                const double      speed    = velocity_y[center];
                double            gradient = 0.0;
                if (speed > 0.0)
                    gradient = (static_cast<double>(source[center]) - source[flat(i, j - 1, k)]) / dy;
                else if (speed < 0.0)
                    gradient = (static_cast<double>(source[flat(i, j + 1, k)]) - source[center]) / dy;
                expected_y[center] = static_cast<float>(static_cast<double>(source[center]) - dt * speed * gradient);
            }

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.advection_kernels() == &backend);
    REQUIRE(backend.supports(AdvectionKernelOp::AdvectionX));
    REQUIRE(backend.supports(AdvectionKernelOp::AdvectionY));
    REQUIRE(backend.supports(AdvectionKernelOp::VerticalFlux));
    REQUIRE(backend.supports(AdvectionKernelOp::ConservativeCartesianCgridTransport));
    REQUIRE(backend.supports(AdvectionKernelOp::BatchedAdvection));
    REQUIRE(backend.ready(KernelId::AdvectX));
    REQUIRE(backend.ready(KernelId::AdvectY));
    REQUIRE(backend.pipeline(KernelId::AdvectX) != 0);
    REQUIRE(backend.pipeline(KernelId::AdvectY) != 0);

    std::vector<float>      actual_x(cells, -1234.0f);
    std::vector<float>      actual_y(cells, -5678.0f);
    ScalarAdvectionDispatch x_dispatch{source.data(), velocity_x.data(), actual_x.data(), nr, nth, nz, dx, dy, dt};
    ScalarAdvectionDispatch y_dispatch{source.data(), velocity_y.data(), actual_y.data(), nr, nth, nz, dy, dx, dt};
    REQUIRE(backend.dispatch_advection_x(x_dispatch));
    REQUIRE(backend.dispatch_advection_y(y_dispatch));
    REQUIRE(backend.completed_command_buffer_count() == 2);

    double      maximum_x_error     = 0.0;
    double      maximum_y_error     = 0.0;
    std::size_t zero_velocity_cells = 0;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t index = flat(i, j, k);
                maximum_x_error =
                    std::max(maximum_x_error, std::abs(static_cast<double>(actual_x[index] - expected_x[index])));
                maximum_y_error =
                    std::max(maximum_y_error, std::abs(static_cast<double>(actual_y[index] - expected_y[index])));
                if (i == 0 || i == nr - 1 || k == 0 || k == nz - 1)
                    REQUIRE(std::memcmp(&actual_x[index], &source[index], sizeof(float)) == 0);
                if (j == 0 || j == nth - 1 || k == 0 || k == nz - 1)
                    REQUIRE(std::memcmp(&actual_y[index], &source[index], sizeof(float)) == 0);
                if (i > 0 && i < nr - 1 && k > 0 && k < nz - 1 && velocity_x[index] == 0.0f)
                {
                    ++zero_velocity_cells;
                    REQUIRE(actual_x[index] == source[index]);
                }
                if (j > 0 && j < nth - 1 && k > 0 && k < nz - 1 && velocity_y[index] == 0.0f)
                {
                    ++zero_velocity_cells;
                    REQUIRE(actual_y[index] == source[index]);
                }
            }
    REQUIRE(zero_velocity_cells > 0);
    // The CPU contract evaluates the expression in double before one fp32
    // rounding; the canonical GPU shaders evaluate it in fp32. At q~300 the
    // resulting admissible difference is one float ULP (3.05176e-5).
    REQUIRE(maximum_x_error <= 4.0e-5);
    REQUIRE(maximum_y_error <= 4.0e-5);

    std::vector<float> inplace = source;
    x_dispatch.src             = inplace.data();
    x_dispatch.dst             = inplace.data();
    REQUIRE(backend.dispatch_advection_x(x_dispatch));
    REQUIRE(std::memcmp(inplace.data(), actual_x.data(), cells * sizeof(float)) == 0);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::AdvectX) == 2);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::AdvectY) == 1);

    const std::vector<float> unchanged = actual_y;
    const auto               completed = backend.completed_command_buffer_count();
    ScalarAdvectionDispatch  invalid   = y_dispatch;
    invalid.primary_spacing            = 0.0f;
    REQUIRE_FALSE(backend.dispatch_advection_y(invalid));
    invalid    = y_dispatch;
    invalid.dt = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_FALSE(backend.dispatch_advection_y(invalid));
    invalid     = y_dispatch;
    invalid.nth = 2;
    REQUIRE_FALSE(backend.dispatch_advection_y(invalid));
    invalid     = y_dispatch;
    invalid.dst = const_cast<float*>(invalid.velocity);
    REQUIRE_FALSE(backend.dispatch_advection_y(invalid));
    invalid     = y_dispatch;
    invalid.src = nullptr;
    REQUIRE_FALSE(backend.dispatch_advection_y(invalid));
    REQUIRE(backend.completed_command_buffer_count() == completed);
    REQUIRE(std::memcmp(actual_y.data(), unchanged.data(), cells * sizeof(float)) == 0);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_advection_x_dispatches == 2);
    REQUIRE(record.completed_advection_y_dispatches == 1);
    REQUIRE(record.advection_x_msl_source_hash.size() == 16);
    REQUIRE(record.advection_y_msl_source_hash.size() == 16);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::advection_x_msl_source()) == record.advection_x_msl_source_hash);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::advection_y_msl_source()) == record.advection_y_msl_source_hash);
    std::cout << "[metal-advection-xy-evidence] x_dispatches=" << record.completed_advection_x_dispatches
              << " y_dispatches=" << record.completed_advection_y_dispatches << " max_x_abs=" << maximum_x_error
              << " max_y_abs=" << maximum_y_error << " x_msl_fnv1a64=" << record.advection_x_msl_source_hash
              << " y_msl_fnv1a64=" << record.advection_y_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal pre-vertical advection batches X then Y atomically in one command buffer",
          "[metal][physical][gpu_parity][advection][batch]")
{
    constexpr int     nr      = 11;
    constexpr int     nth     = 10;
    constexpr int     nz      = 8;
    constexpr float   dr      = 90.0f;
    constexpr float   dtheta  = 135.0f;
    constexpr float   dt_half = 1.75f;
    const std::size_t cells   = static_cast<std::size_t>(nr) * nth * nz;
    const auto        flat    = [](int i, int j, int k) { return (static_cast<std::size_t>(i) * nth + j) * nz + k; };

    std::vector<float> source(cells), u(cells), v(cells);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t index = flat(i, j, k);
                const float       x     = static_cast<float>(i - 5);
                const float       y     = static_cast<float>(j - 4);
                source[index] =
                    275.0f + 0.13f * x * x - 0.09f * y * y + 0.05f * x * y + 0.021f * static_cast<float>(k * k);
                const int u_branch = (i + 2 * j + 3 * k) % 3;
                const int v_branch = (3 * i + j + 2 * k) % 3;
                u[index]           = u_branch == 0 ? -9.0f : (u_branch == 1 ? 0.0f : 7.25f);
                v[index]           = v_branch == 0 ? 6.5f : (v_branch == 1 ? -8.0f : 0.0f);
            }

    // Canonical host oracle: boundary ownership is copied at the start of
    // each directional phase, exactly as the Field3D CPU path does.
    std::vector<float> expected_x = source;
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const std::size_t center   = flat(i, j, k);
                const double      speed    = u[center];
                double            gradient = 0.0;
                if (speed > 0.0)
                    gradient = (static_cast<double>(source[center]) - source[flat(i - 1, j, k)]) / dr;
                else if (speed < 0.0)
                    gradient = (static_cast<double>(source[flat(i + 1, j, k)]) - source[center]) / dr;
                expected_x[center] =
                    static_cast<float>(static_cast<double>(source[center]) - dt_half * speed * gradient);
            }
    std::vector<float> expected = expected_x;
    for (int i = 0; i < nr; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const std::size_t center   = flat(i, j, k);
                const double      speed    = v[center];
                double            gradient = 0.0;
                if (speed > 0.0)
                    gradient = (static_cast<double>(expected_x[center]) - expected_x[flat(i, j - 1, k)]) / dtheta;
                else if (speed < 0.0)
                    gradient = (static_cast<double>(expected_x[flat(i, j + 1, k)]) - expected_x[center]) / dtheta;
                expected[center] =
                    static_cast<float>(static_cast<double>(expected_x[center]) - dt_half * speed * gradient);
            }

    MetalComputeBackend reference_backend;
    initialize_backend(reference_backend);
    std::vector<float>      reference_x(cells, -1.0f);
    std::vector<float>      reference_y(cells, -2.0f);
    ScalarAdvectionDispatch x_dispatch{source.data(), u.data(), reference_x.data(), nr, nth, nz, dr, dtheta, dt_half};
    ScalarAdvectionDispatch y_dispatch{
        reference_x.data(), v.data(), reference_y.data(), nr, nth, nz, dtheta, dr, dt_half};
    REQUIRE(reference_backend.dispatch_advection_x(x_dispatch));
    REQUIRE(reference_backend.dispatch_advection_y(y_dispatch));
    REQUIRE(reference_backend.completed_command_buffer_count() == 2);

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.supports_advection_pre_vertical_batch());
    REQUIRE(backend.supports(AdvectionKernelOp::BatchedAdvection));
    std::vector<float>                actual(cells, -3.0f);
    AdvectionBatchPreVerticalDispatch batch{
        source.data(), actual.data(), u.data(), v.data(), nr, nth, nz, dr, dtheta, dt_half};
    REQUIRE(backend.dispatch_advection_batch_pre_vertical(batch));
    REQUIRE(backend.completed_command_buffer_count() == 1);
    REQUIRE(backend.last_command_buffer_completed());
    REQUIRE(std::memcmp(actual.data(), reference_y.data(), cells * sizeof(float)) == 0);

    double maximum_error = 0.0;
    for (std::size_t index = 0; index < cells; ++index)
        maximum_error = std::max(maximum_error, std::abs(static_cast<double>(actual[index] - expected[index])));
    REQUIRE(maximum_error <= 8.0e-5);

    std::vector<float> inplace = source;
    batch.scalar_in            = inplace.data();
    batch.result_out           = inplace.data();
    REQUIRE(backend.dispatch_advection_batch_pre_vertical(batch));
    REQUIRE(std::memcmp(inplace.data(), actual.data(), cells * sizeof(float)) == 0);
    REQUIRE(backend.completed_command_buffer_count() == 2);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::AdvectX) == 2);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::AdvectY) == 2);

    const std::vector<float> unchanged                  = actual;
    const std::vector<float> unchanged_u                = u;
    const auto               completed                  = backend.completed_command_buffer_count();
    const auto               require_rejected_unchanged = [&](const AdvectionBatchPreVerticalDispatch& invalid)
    {
        REQUIRE_FALSE(backend.dispatch_advection_batch_pre_vertical(invalid));
        REQUIRE(backend.completed_command_buffer_count() == completed);
        REQUIRE(std::memcmp(actual.data(), unchanged.data(), cells * sizeof(float)) == 0);
    };
    AdvectionBatchPreVerticalDispatch invalid{
        source.data(), actual.data(), u.data(), v.data(), nr, nth, nz, dr, dtheta, dt_half};
    invalid.dr = 0.0f;
    require_rejected_unchanged(invalid);
    invalid            = batch;
    invalid.scalar_in  = source.data();
    invalid.result_out = actual.data();
    invalid.dtheta     = std::numeric_limits<float>::infinity();
    require_rejected_unchanged(invalid);
    invalid            = batch;
    invalid.scalar_in  = source.data();
    invalid.result_out = actual.data();
    invalid.dt_half    = std::numeric_limits<float>::quiet_NaN();
    require_rejected_unchanged(invalid);
    invalid            = batch;
    invalid.scalar_in  = source.data();
    invalid.result_out = actual.data();
    invalid.nr         = 2;
    require_rejected_unchanged(invalid);
    invalid            = batch;
    invalid.scalar_in  = nullptr;
    invalid.result_out = actual.data();
    require_rejected_unchanged(invalid);
    invalid            = batch;
    invalid.scalar_in  = source.data();
    invalid.result_out = u.data();
    REQUIRE_FALSE(backend.dispatch_advection_batch_pre_vertical(invalid));
    REQUIRE(std::memcmp(u.data(), unchanged_u.data(), cells * sizeof(float)) == 0);
    REQUIRE(backend.completed_command_buffer_count() == completed);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_advection_pre_vertical_batches == 2);
    REQUIRE(record.completed_advection_x_dispatches == 2);
    REQUIRE(record.completed_advection_y_dispatches == 2);
    std::cout << "[metal-advection-pre-vertical-batch-evidence] batches="
              << record.completed_advection_pre_vertical_batches
              << " command_buffers=" << record.completed_command_buffers
              << " x_dispatches=" << record.completed_advection_x_dispatches
              << " y_dispatches=" << record.completed_advection_y_dispatches << " max_abs=" << maximum_error
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal TVD vertical flux covers every limiter positivity CFL and deep columns",
          "[metal][physical][gpu_parity][advection][vertical_flux]")
{
    constexpr int       nr         = 3;
    constexpr int       nth        = 4;
    constexpr int       nz         = 13;
    constexpr double    cfl_target = 0.72;
    const std::size_t   cells      = static_cast<std::size_t>(nr) * nth * nz;
    std::vector<float>  q(cells), w(cells);
    std::vector<double> dz(nz);
    for (int k = 0; k < nz; ++k)
        dz[static_cast<std::size_t>(k)] = 32.0 + 2.75 * k;
    for (int column = 0; column < nr * nth; ++column)
        for (int k = 0; k < nz; ++k)
        {
            const std::size_t index = static_cast<std::size_t>(column) * nz + k;
            const float       z     = static_cast<float>(k - 5);
            q[index]         = 0.25f + 0.011f * column + 0.007f * z * z - 0.003f * static_cast<float>((column + k) % 4);
            const int branch = (2 * column + k) % 3;
            w[index]         = branch == 0 ? -4.25f : (branch == 1 ? 0.0f : 5.5f);
        }

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.supports(AdvectionKernelOp::VerticalFlux));
    REQUIRE(backend.supports_vertical_flux_dispatch());
    REQUIRE(backend.ready(KernelId::TvdVerticalFlux));
    REQUIRE(backend.pipeline(KernelId::TvdVerticalFlux) != 0);
    REQUIRE(std::strstr(MetalComputeBackend::tvd_vertical_flux_msl_source(), "kernel void tvd_vertical_flux") !=
            nullptr);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::tvd_vertical_flux_msl_source()) ==
            MetalComputeBackend::tvd_vertical_flux_msl_source_hash());

    std::vector<float>         actual(cells, -999.0f);
    VerticalFluxDispatchResult diagnostics{-1.0, -2.0};
    VerticalFluxDispatch       dispatch{};
    dispatch.q_data        = q.data();
    dispatch.w_data        = w.data();
    dispatch.dqdt_data     = actual.data();
    dispatch.nr            = nr;
    dispatch.nth           = nth;
    dispatch.nz            = nz;
    dispatch.dz_data       = dz.data();
    dispatch.dz_count      = nz;
    dispatch.positivity_dt = 3.0;
    dispatch.cfl_target    = cfl_target;
    dispatch.result        = &diagnostics;

    double expected_max_cfl = 0.0;
    for (std::size_t index = 0; index < cells; ++index)
        expected_max_cfl = std::max(expected_max_cfl, std::abs(static_cast<double>(w[index])) / dz[index % nz]);

    double maximum_error = 0.0;
    for (int limiter = 0; limiter <= 4; ++limiter)
    {
        CAPTURE(limiter);
        dispatch.limiter_id = limiter;
        dispatch.positivity = false;
        const auto expected = tvd_vertical_flux_oracle(q, w, dz, nr, nth, nz, limiter, false, dispatch.positivity_dt);
        REQUIRE(backend.dispatch_vertical_flux(dispatch));
        for (std::size_t index = 0; index < cells; ++index)
        {
            maximum_error = std::max(maximum_error, std::abs(static_cast<double>(actual[index] - expected[index])));
            REQUIRE(std::abs(actual[index] - expected[index]) <= 2.0e-6f);
            if (index % nz == nz - 1)
                REQUIRE(actual[index] == 0.0f);
        }
        REQUIRE(diagnostics.max_cfl_z == Approx(expected_max_cfl).epsilon(1.0e-12));
        REQUIRE(diagnostics.suggested_dt == Approx(cfl_target / expected_max_cfl).epsilon(1.0e-12));
    }

    // Force the positivity limiter to bind at the lower boundary.
    std::vector<float> positive_q(cells, 1.0e-3f);
    std::vector<float> outflow(cells, 0.0f);
    for (int column = 0; column < nr * nth; ++column)
        outflow[static_cast<std::size_t>(column) * nz] = 50.0f;
    dispatch.q_data        = positive_q.data();
    dispatch.w_data        = outflow.data();
    dispatch.limiter_id    = 3;
    dispatch.positivity    = true;
    dispatch.positivity_dt = 100.0;
    const auto positive_expected =
        tvd_vertical_flux_oracle(positive_q, outflow, dz, nr, nth, nz, 3, true, dispatch.positivity_dt);
    REQUIRE(backend.dispatch_vertical_flux(dispatch));
    std::size_t limited_cells = 0;
    for (std::size_t index = 0; index < cells; ++index)
    {
        REQUIRE(std::abs(actual[index] - positive_expected[index]) <= 2.0e-8f);
        REQUIRE(positive_q[index] + static_cast<float>(dispatch.positivity_dt) * actual[index] >= -1.0e-9f);
        if (index % nz == 0)
        {
            ++limited_cells;
            REQUIRE(actual[index] == Approx(-1.0e-5f).margin(1.0e-10f));
        }
    }
    REQUIRE(limited_cells == static_cast<std::size_t>(nr * nth));

    // The cell-parallel Metal kernel deliberately has no Vulkan MAX_NZ=256
    // scratch-array ceiling.
    constexpr int       deep_nz = 300;
    std::vector<float>  deep_q(deep_nz), deep_w(deep_nz), deep_actual(deep_nz, -5.0f);
    std::vector<double> deep_dz(deep_nz);
    for (int k = 0; k < deep_nz; ++k)
    {
        deep_q[static_cast<std::size_t>(k)]  = 1.0f + 0.002f * k + 0.00001f * k * k;
        deep_w[static_cast<std::size_t>(k)]  = k % 2 == 0 ? 1.25f : -0.75f;
        deep_dz[static_cast<std::size_t>(k)] = 20.0 + 0.1 * k;
    }
    dispatch.q_data          = deep_q.data();
    dispatch.w_data          = deep_w.data();
    dispatch.dqdt_data       = deep_actual.data();
    dispatch.nr              = 1;
    dispatch.nth             = 1;
    dispatch.nz              = deep_nz;
    dispatch.dz_data         = deep_dz.data();
    dispatch.dz_count        = deep_nz;
    dispatch.positivity      = false;
    dispatch.positivity_dt   = 1.0;
    const auto deep_expected = tvd_vertical_flux_oracle(deep_q, deep_w, deep_dz, 1, 1, deep_nz, 3, false, 1.0);
    REQUIRE(backend.dispatch_vertical_flux(dispatch));
    for (int k = 0; k < deep_nz; ++k)
        REQUIRE(std::abs(deep_actual[static_cast<std::size_t>(k)] - deep_expected[static_cast<std::size_t>(k)]) <=
                2.0e-6f);

    // Shader-side finite guards sanitize contaminated input neighborhoods.
    deep_q[40] = std::numeric_limits<float>::quiet_NaN();
    deep_w[80] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(backend.dispatch_vertical_flux(dispatch));
    for (const float value : deep_actual)
        REQUIRE(std::isfinite(value));

    const std::vector<float>         unchanged             = deep_actual;
    const VerticalFluxDispatchResult unchanged_diagnostics = diagnostics;
    const auto                       completed             = backend.completed_command_buffer_count();
    const auto                       require_rejected      = [&](const VerticalFluxDispatch& invalid)
    {
        REQUIRE_FALSE(backend.dispatch_vertical_flux(invalid));
        REQUIRE(backend.completed_command_buffer_count() == completed);
        REQUIRE(std::memcmp(deep_actual.data(), unchanged.data(), deep_actual.size() * sizeof(float)) == 0);
        REQUIRE(diagnostics.max_cfl_z == unchanged_diagnostics.max_cfl_z);
        REQUIRE(diagnostics.suggested_dt == unchanged_diagnostics.suggested_dt);
    };
    VerticalFluxDispatch invalid = dispatch;
    invalid.dz_count             = deep_nz - 1;
    require_rejected(invalid);
    invalid            = dispatch;
    invalid.limiter_id = 5;
    require_rejected(invalid);
    invalid         = dispatch;
    invalid.dz_data = nullptr;
    require_rejected(invalid);
    invalid               = dispatch;
    invalid.positivity    = true;
    invalid.positivity_dt = 0.0;
    require_rejected(invalid);
    invalid            = dispatch;
    invalid.cfl_target = std::numeric_limits<double>::quiet_NaN();
    require_rejected(invalid);
    invalid           = dispatch;
    invalid.dqdt_data = const_cast<float*>(invalid.q_data);
    REQUIRE_FALSE(backend.dispatch_vertical_flux(invalid));
    REQUIRE(backend.completed_command_buffer_count() == completed);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_vertical_flux_dispatches == 8);
    REQUIRE(record.completed_command_buffers == 8);
    REQUIRE(record.tvd_vertical_flux_msl_source_hash.size() == 16);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::TvdVerticalFlux) == 8);
    std::cout << "[metal-tvd-vertical-flux-evidence] limiters=5 positivity_limited=" << limited_cells
              << " deep_nz=" << deep_nz << " finite_guard=true max_abs=" << maximum_error
              << " msl_fnv1a64=" << record.tvd_vertical_flux_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal post-vertical batch executes Y then X atomically", "[metal][physical][gpu_parity][advection][batch]")
{
    constexpr int      nr      = 9;
    constexpr int      nth     = 8;
    constexpr int      nz      = 7;
    constexpr float    dr      = 110.0f;
    constexpr float    dtheta  = 145.0f;
    constexpr float    dt_half = 1.25f;
    const std::size_t  cells   = static_cast<std::size_t>(nr) * nth * nz;
    std::vector<float> source(cells), u(cells), v(cells);
    for (std::size_t index = 0; index < cells; ++index)
    {
        source[index] = 260.0f + 0.003f * static_cast<float>(index) + 0.17f * static_cast<float>((index / nz) % 5);
        u[index]      = index % 3 == 0 ? -5.0f : (index % 3 == 1 ? 0.0f : 6.25f);
        v[index]      = index % 4 < 2 ? 4.75f : -3.5f;
    }

    MetalComputeBackend reference_backend;
    initialize_backend(reference_backend);
    std::vector<float>      after_y(cells), expected(cells);
    ScalarAdvectionDispatch y_dispatch{source.data(), v.data(), after_y.data(), nr, nth, nz, dtheta, dr, dt_half};
    ScalarAdvectionDispatch x_dispatch{after_y.data(), u.data(), expected.data(), nr, nth, nz, dr, dtheta, dt_half};
    REQUIRE(reference_backend.dispatch_advection_y(y_dispatch));
    REQUIRE(reference_backend.dispatch_advection_x(x_dispatch));

    MetalComputeBackend backend;
    initialize_backend(backend);
    std::vector<float>                 actual(cells, -10.0f);
    AdvectionBatchPostVerticalDispatch batch{};
    batch.scalar_in  = source.data();
    batch.result_out = actual.data();
    batch.u_data     = u.data();
    batch.v_data     = v.data();
    batch.nr         = nr;
    batch.nth        = nth;
    batch.nz         = nz;
    batch.dr         = dr;
    batch.dtheta     = dtheta;
    batch.dz         = 75.0f;
    batch.dt_half    = dt_half;
    batch.dt_full    = 2.0f * dt_half;
    batch.kappa      = 0.0f;
    REQUIRE(backend.dispatch_advection_batch_post_vertical(batch));
    REQUIRE(backend.completed_command_buffer_count() == 1);
    REQUIRE(std::memcmp(actual.data(), expected.data(), cells * sizeof(float)) == 0);

    std::vector<float> inplace = source;
    batch.scalar_in            = inplace.data();
    batch.result_out           = inplace.data();
    REQUIRE(backend.dispatch_advection_batch_post_vertical(batch));
    REQUIRE(std::memcmp(inplace.data(), expected.data(), cells * sizeof(float)) == 0);

    const std::vector<float> unchanged = actual;
    const auto               completed = backend.completed_command_buffer_count();
    batch.scalar_in                    = source.data();
    batch.result_out                   = actual.data();
    batch.kappa                        = 0.01f;
    REQUIRE_FALSE(backend.dispatch_advection_batch_post_vertical(batch));
    REQUIRE(backend.completed_command_buffer_count() == completed);
    REQUIRE(std::memcmp(actual.data(), unchanged.data(), cells * sizeof(float)) == 0);
    batch.kappa   = 0.0f;
    batch.dt_full = 0.0f;
    REQUIRE_FALSE(backend.dispatch_advection_batch_post_vertical(batch));
    REQUIRE(std::memcmp(actual.data(), unchanged.data(), cells * sizeof(float)) == 0);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_advection_post_vertical_batches == 2);
    REQUIRE(record.completed_advection_x_dispatches == 2);
    REQUIRE(record.completed_advection_y_dispatches == 2);
    REQUIRE(record.completed_command_buffers == 2);
    std::cout << "[metal-advection-post-vertical-batch-evidence] batches="
              << record.completed_advection_post_vertical_batches
              << " command_buffers=" << record.completed_command_buffers
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal conservative C-grid transport batches generated and supplied mass flux atomically",
          "[metal][physical][gpu_parity][advection][conservative][residency]")
{
    constexpr int    nx = 10, ny = 9, nz_levels = 7;
    constexpr double dx = 120.0, dy = 150.0, dz_value = 80.0, dt_step = 1.5;
    NR                  = nx;
    NTH                 = ny;
    NZ                  = nz_levels;
    dr                  = dx;
    dtheta              = dy;
    dz                  = dz_value;
    global_stagger_type = StaggerType::CGrid;
    global_grid_geometry.initialize(
        nx, ny, nz_levels, dx, dz_value, dy, CoordinateSystem::Cartesian, StaggerType::CGrid);
    u.resize(nx, ny, nz_levels, 0.0f);
    v.resize(nx, ny, nz_levels, 0.0f);
    w.resize(nx, ny, nz_levels, 0.0f);
    rho.resize(nx, ny, nz_levels, 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_levels; ++k)
            {
                u(i, j, k)   = (i == 0 || i == nx - 2) ? 0.0f : (i < nx / 2 ? 4.0f : -3.25f) + 0.07f * k - 0.03f * j;
                v(i, j, k)   = (j == 0 || j == ny - 2) ? 0.0f : (j < ny / 2 ? -2.0f : 2.5f) + 0.05f * i + 0.02f * k;
                w(i, j, k)   = (k == 0 || k == nz_levels - 1) ? 0.0f : 0.35f * std::sin(0.4f * i) * std::cos(0.3f * j);
                rho(i, j, k) = 1.18f - 0.018f * k + 0.001f * i;
            }
    const std::size_t cells      = rho.size();
    auto              make_field = [&](float base, float amplitude, float ci, float cj, float ck)
    {
        Field3D field(nx, ny, nz_levels, 0.0f);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz_levels; ++k)
                {
                    const float xi = (i - ci) / 2.1f, yj = (j - cj) / 2.0f, zk = (k - ck) / 1.7f;
                    field(i, j, k) = base + amplitude * std::exp(-0.5f * (xi * xi + yj * yj + zk * zk));
                }
        return field;
    };
    Field3D theta_source    = make_field(300.0f, 2.0f, 4.0f, 4.0f, 3.0f);
    Field3D moisture_source = make_field(0.001f, 0.006f, 6.0f, 3.0f, 2.0f);
    Field3D tracer_source   = make_field(-0.2f, 0.8f, 3.0f, 5.0f, 4.0f);
    // Exercise donor depletion across sharp faces in all three directions,
    // including zero-valued positive-definite cells next to the active front.
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_levels; ++k)
                moisture_source(i, j, k) = (i <= nx / 2 && j >= ny / 3 && k <= nz_levels / 2) ? 0.008f : 0.0f;
    SimulationStorageState transport_storage = SimulationStorageState{{nx, ny, nz_levels, dx, dz_value, dt_step, dy}};
    transport_storage.grid.geometry          = global_grid_geometry;
    transport_storage.configuration.coordinate_system = CoordinateSystem::Cartesian;
    transport_storage.configuration.stagger_type      = StaggerType::CGrid;
    transport_storage.configuration.lateral_bc_west   = global_lateral_bc_west;
    transport_storage.configuration.lateral_bc_east   = global_lateral_bc_east;
    transport_storage.configuration.lateral_bc_south  = global_lateral_bc_south;
    transport_storage.configuration.lateral_bc_north  = global_lateral_bc_north;
    transport_storage.fields.rho                      = rho;
    transport_storage.fields.u                        = u;
    transport_storage.fields.v                        = v;
    transport_storage.fields.w                        = w;
    auto    transport_view                            = make_simulation_state_view(transport_storage);
    Field3D theta_cpu, moisture_cpu, tracer_cpu;
    tmv::advect_scalar_conservative_3d_cartesian(transport_view, theta_source, theta_cpu, dt_step, false);
    tmv::advect_scalar_conservative_3d_cartesian(transport_view, moisture_source, moisture_cpu, dt_step, true);
    tmv::advect_scalar_conservative_3d_cartesian(transport_view, tracer_source, tracer_cpu, dt_step, false);

    MetalComputeBackend backend;
    initialize_backend(backend);
    backend.begin_resident_step();
    REQUIRE(backend.supports(AdvectionKernelOp::ConservativeCartesianCgridTransport));
    REQUIRE(backend.supports_supplied_cartesian_cgrid_mass_flux());
    REQUIRE(backend.ready(KernelId::AdvectConservativeMassFluxCartesianCgrid));
    REQUIRE(backend.ready(KernelId::AdvectConservativeScalarFluxCartesianCgrid));
    REQUIRE(backend.ready(KernelId::AdvectConservativeUpdateCartesianCgrid));
    REQUIRE(backend.ready(KernelId::AdvectPdLimiterCartesianCgrid));

    const float* standalone_inputs[3] = {theta_source.data(), moisture_source.data(), tracer_source.data()};
    Field3D      theta_gpu = theta_source, moisture_gpu = moisture_source, tracer_gpu = tracer_source;
    float*       standalone_outputs[3] = {theta_gpu.data(), moisture_gpu.data(), tracer_gpu.data()};
    float        zero_kappas[3]        = {0.0f, 0.0f, 0.0f};
    int          pd_flags[3]           = {0, 1, 0};
    ConservativeCartesianCgridTransportDispatch dispatch{};
    dispatch.execution_state   = &transport_view;
    dispatch.u                 = u.data();
    dispatch.v                 = v.data();
    dispatch.w                 = w.data();
    dispatch.rho_data          = rho.data();
    dispatch.q_in              = standalone_inputs;
    dispatch.q_out             = standalone_outputs;
    dispatch.dt_kappas         = zero_kappas;
    dispatch.positive_definite = pd_flags;
    dispatch.n_scalars         = 3;
    dispatch.nx                = nx;
    dispatch.ny                = ny;
    dispatch.nz                = nz_levels;
    dispatch.inv_dx            = static_cast<float>(1.0 / dx);
    dispatch.inv_dy            = static_cast<float>(1.0 / dy);
    dispatch.inv_dz            = static_cast<float>(1.0 / dz_value);
    dispatch.inv_dx2           = dispatch.inv_dx * dispatch.inv_dx;
    dispatch.inv_dy2           = dispatch.inv_dy * dispatch.inv_dy;
    dispatch.inv_dz2           = dispatch.inv_dz * dispatch.inv_dz;
    dispatch.dt_step           = static_cast<float>(dt_step);
    dispatch.rho_floor         = 1.0e-6f;
    REQUIRE(backend.dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch));
    REQUIRE(backend.completed_command_buffer_count() == 1);
    double max_theta_error = 0.0, max_small_error = 0.0;
    for (std::size_t index = 0; index < cells; ++index)
    {
        max_theta_error =
            std::max(max_theta_error, std::abs(double(theta_gpu.data()[index] - theta_cpu.data()[index])));
        max_small_error =
            std::max(max_small_error, std::abs(double(moisture_gpu.data()[index] - moisture_cpu.data()[index])));
        max_small_error =
            std::max(max_small_error, std::abs(double(tracer_gpu.data()[index] - tracer_cpu.data()[index])));
        REQUIRE(std::abs(theta_gpu.data()[index] - theta_cpu.data()[index]) <= 1.3e-4f);
        REQUIRE(std::abs(moisture_gpu.data()[index] - moisture_cpu.data()[index]) <= 2.0e-5f);
        REQUIRE(std::abs(tracer_gpu.data()[index] - tracer_cpu.data()[index]) <= 2.0e-5f);
        REQUIRE(moisture_gpu.data()[index] >= -1.0e-9f);
        const int i = static_cast<int>(index / (ny * nz_levels));
        const int j = static_cast<int>((index / nz_levels) % ny);
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
        {
            REQUIRE(theta_gpu.data()[index] == theta_source.data()[index]);
            REQUIRE(moisture_gpu.data()[index] == moisture_source.data()[index]);
        }
    }

    Field3D deterministic_a = theta_source, deterministic_b = moisture_source, deterministic_c = tracer_source;
    float*  deterministic_outputs[3] = {deterministic_a.data(), deterministic_b.data(), deterministic_c.data()};
    dispatch.q_out                   = deterministic_outputs;
    REQUIRE(backend.dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch));
    REQUIRE(std::memcmp(theta_gpu.data(), deterministic_a.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(moisture_gpu.data(), deterministic_b.data(), cells * sizeof(float)) == 0);
    REQUIRE(std::memcmp(tracer_gpu.data(), deterministic_c.data(), cells * sizeof(float)) == 0);

    const Field3D                       rho_begin = rho;
    tmv::CartesianCGridTransportContext context;
    context.begin(transport_view, nx, ny, nz_levels, dt_step);
    context.accumulate(transport_view, rho, u, v, w, dt_step);
    context.complete(transport_view);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_levels; ++k)
            {
                const double mx     = context.mass_flux_x(i, j, k) - context.mass_flux_x(i - 1, j, k);
                const double my     = context.mass_flux_y(i, j, k) - context.mass_flux_y(i, j - 1, k);
                const double mz_top = context.mass_flux_z(i, j, k);
                const double mz_bot = k >= 1 ? context.mass_flux_z(i, j, k - 1) : 0.0;
                rho(i, j, k)        = static_cast<float>(rho_begin(i, j, k) -
                                                         dt_step * (mx / dx + my / dy + (mz_top - mz_bot) / dz_value));
            }
    transport_storage.fields.rho = rho;
    backend.invalidate_resident_field("rho");
    Field3D                            coupled_cpu;
    tmv::ScalarBoundaryTransportReport cpu_report;
    tmv::advect_scalar_conservative_3d_cartesian(
        transport_view, moisture_source, coupled_cpu, dt_step, true, &context, &cpu_report);
    Field3D                             coupled_gpu        = moisture_source;
    const float*                        coupled_inputs[1]  = {moisture_source.data()};
    float*                              coupled_outputs[1] = {coupled_gpu.data()};
    tmv::ScalarBoundaryTransportReport  gpu_report;
    tmv::ScalarBoundaryTransportReport* reports[1]    = {&gpu_report};
    int                                 coupled_pd[1] = {1};
    dispatch.q_in                                     = coupled_inputs;
    dispatch.q_out                                    = coupled_outputs;
    dispatch.n_scalars                                = 1;
    dispatch.positive_definite                        = coupled_pd;
    dispatch.use_supplied_mass_flux                   = true;
    dispatch.density_begin                            = context.density_begin.data();
    dispatch.mass_flux_x                              = context.mass_flux_x.data();
    dispatch.mass_flux_y                              = context.mass_flux_y.data();
    dispatch.mass_flux_z                              = context.mass_flux_z.data();
    dispatch.host_boundary_reports                    = reports;
    REQUIRE(backend.dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch));
    REQUIRE(cpu_report.complete);
    REQUIRE(gpu_report.complete);
    REQUIRE(std::isfinite(cpu_report.closure_residual));
    REQUIRE(std::isfinite(gpu_report.closure_residual));
    for (std::size_t index = 0; index < cells; ++index)
    {
        REQUIRE(std::abs(coupled_gpu.data()[index] - coupled_cpu.data()[index]) <= 2.0e-5f);
        REQUIRE(coupled_gpu.data()[index] >= -1.0e-9f);
    }

    const double cell_volume         = dx * dy * dz_value;
    double       inventory_scale     = 1.0;
    double       measured_gpu_change = 0.0;
    for (std::size_t index = 0; index < cells; ++index)
    {
        inventory_scale += std::abs(static_cast<double>(context.density_begin.data()[index]) *
                                    static_cast<double>(moisture_source.data()[index]) * cell_volume);
        measured_gpu_change +=
            (static_cast<double>(rho.data()[index]) * static_cast<double>(coupled_gpu.data()[index]) -
             static_cast<double>(context.density_begin.data()[index]) *
                 static_cast<double>(moisture_source.data()[index])) *
            cell_volume;
    }
    const double report_tolerance    = 512.0 * std::numeric_limits<float>::epsilon() * inventory_scale;
    const double cpu_report_values[] = {
        cpu_report.outward_boundary_inventory,
        cpu_report.flux_implied_state_change,
        cpu_report.committed_state_change,
        cpu_report.closure_residual,
    };
    const double gpu_report_values[] = {
        gpu_report.outward_boundary_inventory,
        gpu_report.flux_implied_state_change,
        gpu_report.committed_state_change,
        gpu_report.closure_residual,
    };
    for (std::size_t term = 0; term < std::size(cpu_report_values); ++term)
    {
        CAPTURE(term, cpu_report_values[term], gpu_report_values[term], inventory_scale, report_tolerance);
        REQUIRE(std::abs(gpu_report_values[term] - cpu_report_values[term]) <= report_tolerance);
    }
    REQUIRE(std::abs(gpu_report.committed_state_change - measured_gpu_change) <= report_tolerance);
    REQUIRE(std::abs(gpu_report.closure_residual) <= report_tolerance);
    REQUIRE(std::abs(gpu_report.committed_state_change - gpu_report.flux_implied_state_change -
                     gpu_report.closure_residual) <= 64.0 * std::numeric_limits<double>::epsilon() *
                                                         std::max({1.0,
                                                                   std::abs(gpu_report.committed_state_change),
                                                                   std::abs(gpu_report.flux_implied_state_change)}));

    Field3D coupled_repeat = moisture_source;
    coupled_outputs[0]     = coupled_repeat.data();
    dispatch.q_out         = coupled_outputs;
    REQUIRE(backend.dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch));
    REQUIRE(std::memcmp(coupled_gpu.data(), coupled_repeat.data(), cells * sizeof(float)) == 0);

    const Field3D unchanged = coupled_repeat;
    const auto    completed = backend.completed_command_buffer_count();
    zero_kappas[0]          = 0.01f;
    REQUIRE_FALSE(backend.dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch));
    REQUIRE(backend.completed_command_buffer_count() == completed);
    REQUIRE(std::memcmp(unchanged.data(), coupled_repeat.data(), cells * sizeof(float)) == 0);
    zero_kappas[0]         = 0.0f;
    dispatch.density_begin = nullptr;
    REQUIRE_FALSE(backend.dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch));
    REQUIRE(std::memcmp(unchanged.data(), coupled_repeat.data(), cells * sizeof(float)) == 0);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_conservative_cgrid_batches == 4);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::AdvectConservativeMassFluxCartesianCgrid) == 2);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::AdvectPdLimiterCartesianCgrid) == 0);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::AdvectConservativeUpdateCartesianCgrid) == 8);
    REQUIRE(record.completed_conservative_cgrid_scalars == 8);
    REQUIRE(record.conservative_cgrid_resident_reuses >= 15);
    REQUIRE(record.conservative_cgrid_transport_msl_source_hash.size() == 16);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::conservative_cgrid_transport_msl_source()) ==
            record.conservative_cgrid_transport_msl_source_hash);
    std::cout << "[metal-conservative-cgrid-evidence] batches=" << record.completed_conservative_cgrid_batches
              << " scalars=" << record.completed_conservative_cgrid_scalars
              << " resident_reuses=" << record.conservative_cgrid_resident_reuses
              << " max_theta_abs=" << max_theta_error << " max_small_abs=" << max_small_error
              << " report_complete=" << std::boolalpha << gpu_report.complete
              << " msl_fnv1a64=" << record.conservative_cgrid_transport_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';

    Field3D      stationary = theta_source, stationary_out = theta_source;
    Field3D      zero_flux(nx, ny, nz_levels, 0.0f);
    const float* stationary_input[]  = {stationary.data()};
    float*       stationary_output[] = {stationary_out.data()};
    int          stationary_pd[]     = {0};
    dispatch.q_in                    = stationary_input;
    dispatch.q_out                   = stationary_output;
    dispatch.positive_definite       = stationary_pd;
    dispatch.density_begin           = rho.data();
    dispatch.rho_data                = rho.data();
    dispatch.mass_flux_x             = zero_flux.data();
    dispatch.mass_flux_y             = zero_flux.data();
    dispatch.mass_flux_z             = zero_flux.data();
    dispatch.host_boundary_reports   = nullptr;
    dispatch.host_context            = nullptr;
    REQUIRE(backend.dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch));
    REQUIRE(std::memcmp(stationary.data(), stationary_out.data(), cells * sizeof(float)) == 0);

    transport_storage.fields.theta = theta_source;
    transport_storage.fields.qv    = moisture_source;
    transport_storage.fields.p.resize(nx, ny, nz_levels, 100000.0f);
    for (auto* field : {&transport_storage.fields.qc,
                        &transport_storage.fields.qr,
                        &transport_storage.fields.qi,
                        &transport_storage.fields.qs,
                        &transport_storage.fields.qg,
                        &transport_storage.fields.qh})
        field->resize(nx, ny, nz_levels, 0.0f);
    const float* energy_inputs[]  = {transport_storage.fields.theta.data(), transport_storage.fields.qv.data()};
    float*       energy_outputs[] = {transport_storage.fields.theta.data(), transport_storage.fields.qv.data()};
    int          energy_pd[]      = {0, 1};
    tmv::ScalarMoistEnergyTransportReportV1  theta_energy, vapor_energy;
    tmv::ScalarMoistEnergyTransportReportV1* energy_reports[] = {&theta_energy, &vapor_energy};
    tmv::ScalarMoistEnergyKind               energy_kinds[]   = {tmv::ScalarMoistEnergyKind::PotentialTemperature,
                                                                 tmv::ScalarMoistEnergyKind::WaterVapor};
    dispatch.n_scalars                                        = 2;
    dispatch.q_in                                             = energy_inputs;
    dispatch.q_out                                            = energy_outputs;
    dispatch.positive_definite                                = energy_pd;
    dispatch.density_begin                                    = context.density_begin.data();
    dispatch.mass_flux_x                                      = context.mass_flux_x.data();
    dispatch.mass_flux_y                                      = context.mass_flux_y.data();
    dispatch.mass_flux_z                                      = context.mass_flux_z.data();
    dispatch.host_energy_kinds                                = energy_kinds;
    dispatch.host_energy_reports                              = energy_reports;
    REQUIRE(backend.dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch));
    REQUIRE(theta_energy.complete);
    REQUIRE(vapor_energy.complete);
    double           measured_sensible = 0.0, cross_term = 0.0;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk - physical_constants::gas_constant_water_vapor_jkgk;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_levels; ++k)
            {
                const double old_theta = theta_source(i, j, k), old_qv = moisture_source(i, j, k);
                const double new_theta = transport_storage.fields.theta(i, j, k);
                const double new_qv    = transport_storage.fields.qv(i, j, k);
                const double mass_cv   = rho(i, j, k) * dx * dy * dz_value * cv_v;
                measured_sensible += mass_cv * (new_qv * (new_theta - physical_constants::freezing_temperature_k) -
                                                old_qv * (old_theta - physical_constants::freezing_temperature_k));
                cross_term += mass_cv * (new_theta - old_theta) * (new_qv - old_qv);
            }
    REQUIRE(std::abs(cross_term) > 1.0e-6 * std::max(1.0, std::abs(measured_sensible)));
    REQUIRE(std::abs(theta_energy.water_sensible_change_j + vapor_energy.water_sensible_change_j - measured_sensible) <=
            1.0e-8 * std::max(1.0, std::abs(measured_sensible)));
}

TEST_CASE("Metal dynamics relaxation and slow apply preserve ordered atomic commits",
          "[metal][physical][gpu_parity][dynamics][relaxation][slow-apply][residency]")
{
    constexpr int         nx = 8, ny = 7, nz_levels = 6;
    constexpr std::size_t cells    = static_cast<std::size_t>(nx * ny * nz_levels);
    constexpr float       dt_relax = 1.75f, dt_slow = 0.4f;
    const auto flat = [](int i, int j, int k) { return static_cast<std::size_t>((i * ny + j) * nz_levels + k); };
    std::vector<float> u_state(cells), v_state(cells), w_state(cells), theta_state(cells);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_levels; ++k)
            {
                const std::size_t n = flat(i, j, k);
                u_state[n]          = 7.0f + 0.3f * i - 0.17f * j + 0.04f * k;
                v_state[n]          = -3.0f + 0.11f * i + 0.2f * j - 0.03f * k;
                w_state[n]          = 0.8f * std::sin(0.31f * i + 0.27f * j + 0.13f * k);
                theta_state[n]      = 297.0f + 0.2f * i - 0.1f * j + 0.7f * k;
            }
    const auto         initial_u = u_state, initial_v = v_state;
    const auto         initial_w = w_state, initial_theta = theta_state;
    std::vector<float> profiles(7 * nz_levels, 0.0f);
    for (int k = 0; k < nz_levels; ++k)
    {
        profiles[k]                 = k >= 3 ? 0.025f * (k - 2) : 0.0f;
        profiles[nz_levels + k]     = k >= 4 ? 0.09f : 0.0f;
        profiles[2 * nz_levels + k] = 5.0f + 0.08f * k;
        profiles[3 * nz_levels + k] = -1.5f + 0.05f * k;
        profiles[4 * nz_levels + k] = 299.0f + 0.65f * k;
        profiles[6 * nz_levels + k] = k >= 3 ? 0.04f * (k - 2) : 0.0f;
    }
    profiles[5 * nz_levels]     = 0.06f;
    profiles[5 * nz_levels + 1] = 0.025f;

    auto relax_cpu = [&](std::vector<float>& uu, std::vector<float>& vv, std::vector<float>& ww, std::vector<float>& tt)
    {
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz_levels; ++k)
                {
                    const std::size_t n       = flat(i, j, k);
                    const float       upper   = 1.0f - std::exp(-profiles[k] * dt_relax);
                    const float       upper_w = 1.0f - std::exp(-profiles[6 * nz_levels + k] * dt_relax);
                    uu[n] += upper * (profiles[2 * nz_levels + k] - uu[n]);
                    vv[n] += upper * (profiles[3 * nz_levels + k] - vv[n]);
                    ww[n] += upper_w * (-ww[n]);
                    tt[n] += upper * (profiles[4 * nz_levels + k] - tt[n]);
                    const float lid = 1.0f - std::exp(-profiles[nz_levels + k] * dt_relax);
                    ww[n] += lid * (-ww[n]);
                    const int distance = std::min(std::min(i, nx - 1 - i), std::min(j, ny - 1 - j));
                    if (distance < 2)
                    {
                        const float side = 1.0f - std::exp(-profiles[5 * nz_levels + distance] * dt_relax);
                        uu[n] += side * (profiles[2 * nz_levels + k] - uu[n]);
                        vv[n] += side * (profiles[3 * nz_levels + k] - vv[n]);
                        ww[n] += side * (-ww[n]);
                        tt[n] += side * (profiles[4 * nz_levels + k] - tt[n]);
                    }
                }
    };
    auto expected_u = initial_u, expected_v = initial_v, expected_w = initial_w, expected_theta = initial_theta;
    relax_cpu(expected_u, expected_v, expected_w, expected_theta);

    MetalComputeBackend backend;
    initialize_backend(backend);
    backend.begin_resident_step();
    REQUIRE(backend.dynamics_kernels() == &backend);
    REQUIRE(backend.supports(DynamicsKernelOp::StateRelaxationCartesian));
    REQUIRE(backend.supports(DynamicsKernelOp::SlowApply));
    REQUIRE(backend.supports(DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid));
    REQUIRE(backend.ready(KernelId::StateRelaxationCartesian));
    REQUIRE(backend.ready(KernelId::SlowApply));
    REQUIRE(backend.pipeline(KernelId::StateRelaxationCartesian) != 0);
    REQUIRE(backend.pipeline(KernelId::SlowApply) != 0);

    StateRelaxationDispatch relaxation{};
    relaxation.u              = u_state.data();
    relaxation.v              = v_state.data();
    relaxation.w              = w_state.data();
    relaxation.theta_data     = theta_state.data();
    relaxation.profiles       = profiles.data();
    relaxation.profiles_len   = static_cast<int>(profiles.size());
    relaxation.nx             = nx;
    relaxation.ny             = ny;
    relaxation.nz             = nz_levels;
    relaxation.lateral_width  = 2;
    relaxation.do_upper       = true;
    relaxation.do_w_only      = true;
    relaxation.do_lateral     = true;
    relaxation.dt_step        = dt_relax;
    const auto throw_observer = +[](void*, const float*, const float*, const float*, const float*)
    { throw std::runtime_error("reject Metal dynamics candidate"); };
    relaxation.completed_observer = throw_observer;
    const auto before_relax_veto  = backend.completed_command_buffer_count();
    REQUIRE_THROWS_WITH(backend.dispatch_state_relaxation_cartesian(relaxation), "reject Metal dynamics candidate");
    REQUIRE(backend.completed_command_buffer_count() == before_relax_veto + 1);
    REQUIRE(u_state == initial_u);
    REQUIRE(v_state == initial_v);
    REQUIRE(w_state == initial_w);
    REQUIRE(theta_state == initial_theta);
    relaxation.completed_observer = nullptr;
    REQUIRE(backend.dispatch_state_relaxation_cartesian(relaxation));
    double max_relax_error = 0.0;
    for (std::size_t n = 0; n < cells; ++n)
    {
        max_relax_error = std::max(max_relax_error, std::abs(double(u_state[n] - expected_u[n])));
        max_relax_error = std::max(max_relax_error, std::abs(double(v_state[n] - expected_v[n])));
        max_relax_error = std::max(max_relax_error, std::abs(double(w_state[n] - expected_w[n])));
        max_relax_error = std::max(max_relax_error, std::abs(double(theta_state[n] - expected_theta[n])));
        REQUIRE(std::abs(u_state[n] - expected_u[n]) <= 2.0e-5f);
        REQUIRE(std::abs(v_state[n] - expected_v[n]) <= 2.0e-5f);
        REQUIRE(std::abs(w_state[n] - expected_w[n]) <= 2.0e-5f);
        REQUIRE(std::abs(theta_state[n] - expected_theta[n]) <= 2.0e-5f);
    }
    // Slow-apply is pointwise-exact from the state it receives. Rebase that
    // oracle after separately accepting Metal exp() within relaxation's
    // manifest tolerance, so a relaxation ULP is not charged to this stage.
    expected_u     = u_state;
    expected_v     = v_state;
    expected_w     = w_state;
    expected_theta = theta_state;

    std::vector<float> pressure(cells), du(cells), dv(cells), dw(cells), dp(cells), du_pbl(cells), dv_pbl(cells);
    for (std::size_t n = 0; n < cells; ++n)
    {
        pressure[n] = 88000.0f + 0.5f * float(n % 17);
        du[n]       = 0.03f * float(int(n % 7) - 3);
        dv[n]       = -0.025f * float(int(n % 5) - 2);
        dw[n]       = 0.012f * float(int(n % 9) - 4);
        dp[n]       = 1.5f * float(int(n % 11) - 5);
        du_pbl[n]   = 0.006f * float(int(n % 3) - 1);
        dv_pbl[n]   = 0.004f * float(int(n % 4) - 2);
    }
    du[3]           = std::numeric_limits<float>::infinity();
    du_pbl[3]       = -std::numeric_limits<float>::infinity();
    dv[5]           = std::numeric_limits<float>::quiet_NaN();
    dw[7]           = std::numeric_limits<float>::infinity();
    dp[9]           = std::numeric_limits<float>::quiet_NaN();
    auto expected_p = pressure;
    auto apply_slow_cpu =
        [&](std::vector<float>& uu, std::vector<float>& vv, std::vector<float>& ww, std::vector<float>& pp)
    {
        for (std::size_t n = 0; n < cells; ++n)
        {
            const float dus = du[n] + du_pbl[n], dvs = dv[n] + dv_pbl[n];
            uu[n] += (std::isfinite(dus) ? dus : 0.0f) * dt_slow;
            vv[n] += (std::isfinite(dvs) ? dvs : 0.0f) * dt_slow;
            ww[n] += (std::isfinite(dw[n]) ? dw[n] : 0.0f) * dt_slow;
            pp[n] += (std::isfinite(dp[n]) ? dp[n] : 0.0f) * dt_slow;
        }
    };
    apply_slow_cpu(expected_u, expected_v, expected_w, expected_p);
    struct Capture
    {
        const std::vector<float>*         old_u;
        std::vector<float>                original_u;
        std::size_t                       count;
        bool                              called        = false;
        bool                              old_untouched = false;
        std::array<std::vector<float>, 4> result;
    } capture{&u_state, u_state, cells, false, false, {}};
    const auto capture_observer = +[](void* raw, const float* a, const float* b, const float* c, const float* d)
    {
        auto& x                = *static_cast<Capture*>(raw);
        x.called               = true;
        x.old_untouched        = std::memcmp(x.old_u->data(), x.original_u.data(), x.count * sizeof(float)) == 0;
        const float* fields[4] = {a, b, c, d};
        for (int f = 0; f < 4; ++f)
            x.result[f].assign(fields[f], fields[f] + x.count);
    };
    SlowApplyDispatch slow{};
    slow.u                          = u_state.data();
    slow.v                          = v_state.data();
    slow.w                          = w_state.data();
    slow.p_data                     = pressure.data();
    slow.du_dt                      = du.data();
    slow.dv_dt                      = dv.data();
    slow.dw_dt                      = dw.data();
    slow.dp_dt                      = dp.data();
    slow.du_dt_pbl                  = du_pbl.data();
    slow.dv_dt_pbl                  = dv_pbl.data();
    slow.nx                         = nx;
    slow.ny                         = ny;
    slow.nz                         = nz_levels;
    slow.dt_large                   = dt_slow;
    slow.completed_observer         = capture_observer;
    slow.completed_observer_context = &capture;
    REQUIRE(backend.dispatch_slow_apply(slow));
    REQUIRE(capture.called);
    REQUIRE(capture.old_untouched);
    REQUIRE(u_state == expected_u);
    REQUIRE(v_state == expected_v);
    REQUIRE(w_state == expected_w);
    REQUIRE(pressure == expected_p);
    REQUIRE(capture.result[0] == expected_u);
    REQUIRE(capture.result[3] == expected_p);

    const auto once_u = u_state, once_v = v_state, once_w = w_state, once_p = pressure;
    apply_slow_cpu(expected_u, expected_v, expected_w, expected_p);
    slow.completed_observer         = throw_observer;
    slow.completed_observer_context = nullptr;
    REQUIRE_THROWS_WITH(backend.dispatch_slow_apply(slow), "reject Metal dynamics candidate");
    REQUIRE(u_state == once_u);
    REQUIRE(v_state == once_v);
    REQUIRE(w_state == once_w);
    REQUIRE(pressure == once_p);
    slow.completed_observer = nullptr;
    REQUIRE(backend.dispatch_slow_apply(slow));
    REQUIRE(u_state == expected_u);
    REQUIRE(v_state == expected_v);
    REQUIRE(w_state == expected_w);
    REQUIRE(pressure == expected_p);

    const auto completed    = backend.completed_command_buffer_count();
    const auto unchanged    = u_state;
    relaxation.profiles_len = 7 * nz_levels - 1;
    REQUIRE_FALSE(backend.dispatch_state_relaxation_cartesian(relaxation));
    REQUIRE(backend.completed_command_buffer_count() == completed);
    REQUIRE(u_state == unchanged);
    relaxation.profiles_len = 7 * nz_levels;
    slow.dt_large           = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_FALSE(backend.dispatch_slow_apply(slow));
    REQUIRE(u_state == unchanged);

    const auto record = backend.execution_record();
    REQUIRE(record.completed_state_relaxation_dispatches == 1);
    REQUIRE(record.completed_slow_apply_dispatches == 2);
    REQUIRE(record.dynamics_state_resident_reuses >= 7);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::StateRelaxationCartesian) == 1);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::SlowApply) == 2);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::dynamics_apply_msl_source()) == record.dynamics_apply_msl_source_hash);
    std::cout << "[metal-dynamics-apply-evidence] relaxation=" << record.completed_state_relaxation_dispatches
              << " slow_apply=" << record.completed_slow_apply_dispatches
              << " resident_reuses=" << record.dynamics_state_resident_reuses << " max_relax_abs=" << max_relax_error
              << " observer_exception=atomic msl_fnv1a64=" << record.dynamics_apply_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal Cartesian C-grid slow tendencies retain a complete slow stage",
          "[metal][physical][gpu_parity][dynamics][cgrid][slow][residency]")
{
    constexpr int         nx = 9, ny = 8, nz_levels = 7;
    constexpr std::size_t cells = static_cast<std::size_t>(nx * ny * nz_levels);
    constexpr float       dx = 420.0f, dy = 510.0f, dz_value = 230.0f, g = 9.81f;
    const auto         at = [](int i, int j, int k) { return static_cast<std::size_t>((i * ny + j) * nz_levels + k); };
    std::vector<float> u_field(cells), v_field(cells), w_field(cells), rho_field(cells), pressure(cells);
    std::vector<float> qv_field(cells), loading(cells), rho0(nz_levels), qv0(nz_levels), u0(nz_levels), v0(nz_levels);
    for (int k = 0; k < nz_levels; ++k)
    {
        rho0[k] = 1.16f - 0.025f * k;
        qv0[k]  = 0.006f - 0.0003f * k;
        u0[k]   = 4.5f + 0.05f * k;
        v0[k]   = -1.2f + 0.03f * k;
    }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_levels; ++k)
            {
                const auto n = at(i, j, k);
                u_field[n]   = 8.0f + 0.27f * i - 0.13f * j + 0.08f * k;
                v_field[n]   = -3.0f + 0.09f * i + 0.21f * j - 0.06f * k;
                w_field[n]   = 0.35f + 0.025f * i - 0.018f * j + 0.04f * k;
                rho_field[n] = rho0[k] + 0.0015f * i - 0.0008f * j;
                pressure[n]  = 93000.0f + 7.0f * i - 5.0f * j - 18.0f * k;
                qv_field[n]  = qv0[k] + 0.00002f * i + 0.00001f * j;
                loading[n]   = 0.0004f + 0.00001f * (i + j + k);
            }
    std::vector<float>  du(cells, 77.0f), dv(cells, 77.0f), dw(cells, 77.0f), drho(cells, 77.0f), dp(cells, 77.0f);
    std::vector<float>  au(cells, 77.0f), av(cells, 77.0f), aw(cells, 77.0f);
    MetalComputeBackend backend;
    initialize_backend(backend);
    backend.begin_resident_step();
    REQUIRE(backend.supports(DynamicsKernelOp::CartesianCgridSlowTendencies));
    REQUIRE(backend.ready(KernelId::CartesianCgridSlowTendencies));
    REQUIRE(backend.pipeline(KernelId::CartesianCgridSlowTendencies) != 0);

    std::vector<float> calm_u(cells), calm_v(cells), calm_w(cells, 0.0f), calm_rho(cells), calm_p(cells, 90000.0f);
    std::vector<float> calm_du(cells, 1.0f), calm_dv(cells, 1.0f), calm_dw(cells, 1.0f), calm_drho(cells, 1.0f),
        calm_dp(cells, 1.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_levels; ++k)
            {
                const auto n = at(i, j, k);
                calm_u[n]    = u0[k];
                calm_v[n]    = v0[k];
                calm_rho[n]  = rho0[k];
            }
    CartesianCgridSlowTendenciesDispatch calm{};
    calm.u_data         = calm_u.data();
    calm.v_data         = calm_v.data();
    calm.w_data         = calm_w.data();
    calm.rho_data       = calm_rho.data();
    calm.p_data         = calm_p.data();
    calm.rho0_base_data = rho0.data();
    calm.rho0_base_len  = nz_levels;
    calm.u0_base_data   = u0.data();
    calm.v0_base_data   = v0.data();
    calm.wind0_base_len = nz_levels;
    calm.du_dt_data     = calm_du.data();
    calm.dv_dt_data     = calm_dv.data();
    calm.dw_dt_data     = calm_dw.data();
    calm.drho_dt_data   = calm_drho.data();
    calm.dp_dt_data     = calm_dp.data();
    calm.nr             = nx;
    calm.nth            = ny;
    calm.nz             = nz_levels;
    calm.dx             = dx;
    calm.dy             = dy;
    calm.dz             = dz_value;
    calm.g              = g;
    calm.coriolis_f_val = 8.365e-5f;
    REQUIRE(backend.dispatch_cartesian_cgrid_slow_tendencies(calm));
    for (std::size_t n = 0; n < cells; ++n)
    {
        REQUIRE(std::abs(calm_du[n]) <= 1.0e-7f);
        REQUIRE(std::abs(calm_dv[n]) <= 1.0e-7f);
        REQUIRE(std::abs(calm_dw[n]) <= 1.0e-7f);
        REQUIRE(calm_drho[n] == 0.0f);
        REQUIRE(std::abs(calm_dp[n]) <= 1.0e-7f);
    }

    CartesianCgridSlowTendenciesDispatch slow_tendency{};
    slow_tendency.u_data                             = u_field.data();
    slow_tendency.v_data                             = v_field.data();
    slow_tendency.w_data                             = w_field.data();
    slow_tendency.rho_data                           = rho_field.data();
    slow_tendency.p_data                             = pressure.data();
    slow_tendency.rho0_base_data                     = rho0.data();
    slow_tendency.rho0_base_len                      = nz_levels;
    slow_tendency.qv0_base_data                      = qv0.data();
    slow_tendency.qv0_base_len                       = nz_levels;
    slow_tendency.u0_base_data                       = u0.data();
    slow_tendency.v0_base_data                       = v0.data();
    slow_tendency.wind0_base_len                     = nz_levels;
    slow_tendency.qv_data                            = qv_field.data();
    slow_tendency.loading_data                       = loading.data();
    slow_tendency.du_dt_data                         = du.data();
    slow_tendency.dv_dt_data                         = dv.data();
    slow_tendency.dw_dt_data                         = dw.data();
    slow_tendency.drho_dt_data                       = drho.data();
    slow_tendency.dp_dt_data                         = dp.data();
    slow_tendency.nr                                 = nx;
    slow_tendency.nth                                = ny;
    slow_tendency.nz                                 = nz_levels;
    slow_tendency.dx                                 = dx;
    slow_tendency.dy                                 = dy;
    slow_tendency.dz                                 = dz_value;
    slow_tendency.g                                  = g;
    slow_tendency.coriolis_f_val                     = 8.365e-5f;
    slow_tendency.coriolis_beta                      = 2.0e-11f;
    slow_tendency.has_moisture                       = true;
    slow_tendency.has_loading                        = true;
    slow_tendency.slow_rk2_invariant_input_residency = true;
    slow_tendency.du_advection_dt_data               = au.data();
    slow_tendency.dv_advection_dt_data               = av.data();
    slow_tendency.dw_advection_dt_data               = aw.data();
    slow_tendency.frame_velocity_u_ms                = 2.25f;
    slow_tendency.frame_velocity_v_ms                = -0.75f;
    REQUIRE(backend.dispatch_cartesian_cgrid_slow_tendencies(slow_tendency));

    const auto up = [](int hw, float m3, float m2, float m1, float c, float p1, float p2, float p3, float a, float inv)
    {
        if (hw <= 0)
            return 0.0f;
        if (hw == 1)
            return a >= 0 ? (c - m1) * inv : (p1 - c) * inv;
        if (hw == 2)
            return a >= 0 ? (2 * p1 + 3 * c - 6 * m1 + m2) * (inv / 6) : (-p2 + 6 * p1 - 3 * c - 2 * m1) * (inv / 6);
        return a >= 0 ? (-2 * m3 + 15 * m2 - 60 * m1 + 20 * c + 30 * p1 - 3 * p2) * (inv / 60)
                      : (2 * p3 - 15 * p2 + 60 * p1 - 20 * c - 30 * m1 + 3 * m2) * (inv / 60);
    };
    std::vector<float> eu(cells, 0), ev(cells, 0), ew(cells, 0), ep(cells, 0), eau(cells, 0), eav(cells, 0),
        eaw(cells, 0);
    const float ix = 1 / dx, iy = 1 / dy, iz = 1 / dz_value;
    for (int i = 1; i <= nx - 2; ++i)
        for (int j = 1; j <= ny - 2; ++j)
            for (int k = 0; k <= nz_levels - 2; ++k)
            {
                const auto n = at(i, j, k);
                int   hx = std::min(i, nx - 1 - i), hy = std::min(j, ny - 1 - j), hz = std::min(k, nz_levels - 1 - k);
                float c = u_field[n], va = .25f * (v_field[at(i, j - 1, k)] + v_field[n] +
                                                   v_field[at(i + 1, j - 1, k)] + v_field[at(i + 1, j, k)]);
                float wa = k == 0 ? .25f * (w_field[at(i, j, 0)] + w_field[at(i + 1, j, 0)])
                                  : .25f * (w_field[at(i, j, k - 1)] + w_field[n] + w_field[at(i + 1, j, k - 1)] +
                                            w_field[at(i + 1, j, k)]);
                float ux = up(hx,
                              hx >= 3 ? u_field[at(i - 3, j, k)] : 0,
                              hx >= 2 ? u_field[at(i - 2, j, k)] : 0,
                              u_field[at(i - 1, j, k)],
                              c,
                              u_field[at(i + 1, j, k)],
                              hx >= 2 ? u_field[at(i + 2, j, k)] : 0,
                              hx >= 3 ? u_field[at(i + 3, j, k)] : 0,
                              c,
                              ix);
                float uy = up(hy,
                              hy >= 3 ? u_field[at(i, j - 3, k)] : 0,
                              hy >= 2 ? u_field[at(i, j - 2, k)] : 0,
                              u_field[at(i, j - 1, k)],
                              c,
                              u_field[at(i, j + 1, k)],
                              hy >= 2 ? u_field[at(i, j + 2, k)] : 0,
                              hy >= 3 ? u_field[at(i, j + 3, k)] : 0,
                              va,
                              iy);
                float uz = k == 0 ? (wa >= 0 ? 0 : (u_field[at(i, j, 1)] - c) * iz)
                                  : up(hz,
                                       hz >= 3 ? u_field[at(i, j, k - 3)] : 0,
                                       hz >= 2 ? u_field[at(i, j, k - 2)] : 0,
                                       u_field[at(i, j, k - 1)],
                                       c,
                                       u_field[at(i, j, k + 1)],
                                       hz >= 2 ? u_field[at(i, j, k + 2)] : 0,
                                       hz >= 3 ? u_field[at(i, j, k + 3)] : 0,
                                       wa,
                                       iz);
                eau[n]   = -(c - 2.25f) * ux - (va + 0.75f) * uy - wa * uz;
                const float f_u_lo =
                    slow_tendency.coriolis_f_val + slow_tendency.coriolis_beta * (float(j) + 0.25f) * dy;
                const float f_u_hi =
                    slow_tendency.coriolis_f_val + slow_tendency.coriolis_beta * (float(j) + 0.75f) * dy;
                eu[n]    = eau[n] +
                           0.25f * (f_u_lo * (v_field[at(i, j - 1, k)] + v_field[at(i + 1, j - 1, k)] - 2.0f * v0[k]) +
                                    f_u_hi * (v_field[at(i, j, k)] + v_field[at(i + 1, j, k)] - 2.0f * v0[k]));
                c        = v_field[n];
                float ua = .25f * (u_field[at(i - 1, j, k)] + u_field[n] + u_field[at(i - 1, j + 1, k)] +
                                   u_field[at(i, j + 1, k)]);
                wa       = k == 0 ? .25f * (w_field[at(i, j, 0)] + w_field[at(i, j + 1, 0)])
                                  : .25f * (w_field[at(i, j, k - 1)] + w_field[n] + w_field[at(i, j + 1, k - 1)] +
                                            w_field[at(i, j + 1, k)]);
                float vx = up(hx,
                              hx >= 3 ? v_field[at(i - 3, j, k)] : 0,
                              hx >= 2 ? v_field[at(i - 2, j, k)] : 0,
                              v_field[at(i - 1, j, k)],
                              c,
                              v_field[at(i + 1, j, k)],
                              hx >= 2 ? v_field[at(i + 2, j, k)] : 0,
                              hx >= 3 ? v_field[at(i + 3, j, k)] : 0,
                              ua,
                              ix);
                float vy = up(hy,
                              hy >= 3 ? v_field[at(i, j - 3, k)] : 0,
                              hy >= 2 ? v_field[at(i, j - 2, k)] : 0,
                              v_field[at(i, j - 1, k)],
                              c,
                              v_field[at(i, j + 1, k)],
                              hy >= 2 ? v_field[at(i, j + 2, k)] : 0,
                              hy >= 3 ? v_field[at(i, j + 3, k)] : 0,
                              c,
                              iy);
                float vz = k == 0 ? (wa >= 0 ? 0 : (v_field[at(i, j, 1)] - c) * iz)
                                  : up(hz,
                                       hz >= 3 ? v_field[at(i, j, k - 3)] : 0,
                                       hz >= 2 ? v_field[at(i, j, k - 2)] : 0,
                                       v_field[at(i, j, k - 1)],
                                       c,
                                       v_field[at(i, j, k + 1)],
                                       hz >= 2 ? v_field[at(i, j, k + 2)] : 0,
                                       hz >= 3 ? v_field[at(i, j, k + 3)] : 0,
                                       wa,
                                       iz);
                eav[n]   = -(ua - 2.25f) * vx - (c + 0.75f) * vy - wa * vz;
                const float f_v_lo =
                    slow_tendency.coriolis_f_val + slow_tendency.coriolis_beta * (float(j) + 0.75f) * dy;
                const float f_v_hi =
                    slow_tendency.coriolis_f_val + slow_tendency.coriolis_beta * (float(j) + 1.25f) * dy;
                ev[n] = eav[n] -
                        0.25f * (f_v_lo * (u_field[at(i - 1, j, k)] + u_field[at(i, j, k)] - 2.0f * u0[k]) +
                                 f_v_hi * (u_field[at(i - 1, j + 1, k)] + u_field[at(i, j + 1, k)] - 2.0f * u0[k]));
                c     = w_field[n];
                ua = .25f *
                     (u_field[at(i - 1, j, k)] + u_field[n] + u_field[at(i - 1, j, k + 1)] + u_field[at(i, j, k + 1)]);
                va = .25f *
                     (v_field[at(i, j - 1, k)] + v_field[n] + v_field[at(i, j - 1, k + 1)] + v_field[at(i, j, k + 1)]);
                float wx = up(hx,
                              hx >= 3 ? w_field[at(i - 3, j, k)] : 0,
                              hx >= 2 ? w_field[at(i - 2, j, k)] : 0,
                              w_field[at(i - 1, j, k)],
                              c,
                              w_field[at(i + 1, j, k)],
                              hx >= 2 ? w_field[at(i + 2, j, k)] : 0,
                              hx >= 3 ? w_field[at(i + 3, j, k)] : 0,
                              ua,
                              ix);
                float wy = up(hy,
                              hy >= 3 ? w_field[at(i, j - 3, k)] : 0,
                              hy >= 2 ? w_field[at(i, j - 2, k)] : 0,
                              w_field[at(i, j - 1, k)],
                              c,
                              w_field[at(i, j + 1, k)],
                              hy >= 2 ? w_field[at(i, j + 2, k)] : 0,
                              hy >= 3 ? w_field[at(i, j + 3, k)] : 0,
                              va,
                              iy);
                float wz = k == 0 ? (c >= 0 ? c * iz : (w_field[at(i, j, 1)] - c) * iz)
                                  : up(hz,
                                       hz >= 3 ? w_field[at(i, j, k - 3)] : 0,
                                       hz >= 2 ? w_field[at(i, j, k - 2)] : 0,
                                       w_field[at(i, j, k - 1)],
                                       c,
                                       w_field[at(i, j, k + 1)],
                                       hz >= 2 ? w_field[at(i, j, k + 2)] : 0,
                                       hz >= 3 ? w_field[at(i, j, k + 3)] : 0,
                                       c,
                                       iz);
                eaw[n]   = -(ua - 2.25f) * wx - (va + 0.75f) * wy - c * wz;
                float rf = .5f * (rho_field[n] + rho_field[at(i, j, k + 1)]), r0 = .5f * (rho0[k] + rho0[k + 1]);
                ew[n] = eaw[n] - g * (rf - r0) / rf +
                        g * .608f * (.5f * (qv_field[n] + qv_field[at(i, j, k + 1)]) - .5f * (qv0[k] + qv0[k + 1])) -
                        g * .5f * (loading[n] + loading[at(i, j, k + 1)]);
            }
    for (int i = 1; i <= nx - 2; ++i)
        for (int j = 1; j <= ny - 2; ++j)
            for (int k = 1; k <= nz_levels - 2; ++k)
            {
                auto  n  = at(i, j, k);
                float uc = .5f * (u_field[n] + u_field[at(i - 1, j, k)]),
                      vc = .5f * (v_field[n] + v_field[at(i, j - 1, k)]),
                      wc = .5f * (w_field[n] + w_field[at(i, j, k - 1)]);
                ep[n]    = -(uc - 2.25f) * (pressure[at(i + 1, j, k)] - pressure[at(i - 1, j, k)]) / (2 * dx) -
                           (vc + 0.75f) * (pressure[at(i, j + 1, k)] - pressure[at(i, j - 1, k)]) / (2 * dy) -
                           wc * (pressure[at(i, j, k + 1)] - pressure[at(i, j, k - 1)]) / (2 * dz_value);
            }
    double max_error = 0;
    for (std::size_t n = 0; n < cells; ++n)
    {
        for (auto pair : {std::pair<float, float>{du[n], eu[n]},
                          {dv[n], ev[n]},
                          {dw[n], ew[n]},
                          {dp[n], ep[n]},
                          {au[n], eau[n]},
                          {av[n], eav[n]},
                          {aw[n], eaw[n]}})
            max_error = std::max(max_error, std::abs(double(pair.first - pair.second)));
        REQUIRE(std::abs(du[n] - eu[n]) <= 1e-2f);
        REQUIRE(std::abs(dv[n] - ev[n]) <= 1e-2f);
        REQUIRE(std::abs(dw[n] - ew[n]) <= 1e-2f);
        REQUIRE(std::abs(dp[n] - ep[n]) <= 1e-2f);
        REQUIRE(std::abs(au[n] - eau[n]) <= 1e-2f);
        REQUIRE(std::abs(av[n] - eav[n]) <= 1e-2f);
        REQUIRE(std::abs(aw[n] - eaw[n]) <= 1e-2f);
        REQUIRE(drho[n] == 0.0f);
    }

    std::vector<float> state_u = u_field, state_v = v_field, state_w = w_field, state_p = pressure, zero(cells, 0.0f);
    auto               expected_u = state_u, expected_v = state_v, expected_w = state_w, expected_p = state_p;
    for (std::size_t n = 0; n < cells; ++n)
    {
        expected_u[n] += du[n] * .2f;
        expected_v[n] += dv[n] * .2f;
        expected_w[n] += dw[n] * .2f;
        expected_p[n] += dp[n] * .2f;
    }
    SlowApplyDispatch apply{};
    apply.u         = state_u.data();
    apply.v         = state_v.data();
    apply.w         = state_w.data();
    apply.p_data    = state_p.data();
    apply.du_dt     = du.data();
    apply.dv_dt     = dv.data();
    apply.dw_dt     = dw.data();
    apply.dp_dt     = dp.data();
    apply.du_dt_pbl = zero.data();
    apply.dv_dt_pbl = zero.data();
    apply.nx        = nx;
    apply.ny        = ny;
    apply.nz        = nz_levels;
    apply.dt_large  = .2f;
    REQUIRE(backend.dispatch_slow_apply(apply));
    for (std::size_t n = 0; n < cells; ++n)
    {
        REQUIRE(std::abs(state_u[n] - expected_u[n]) <= 1e-6f);
        REQUIRE(std::abs(state_v[n] - expected_v[n]) <= 1e-6f);
        REQUIRE(std::abs(state_w[n] - expected_w[n]) <= 1e-6f);
        REQUIRE(std::abs(state_p[n] - expected_p[n]) <= 1e-3f);
    }

    // A Heun predictor changes the prognostic winds/pressure while rho and qv
    // retain both their host identity and values. The second dispatch must
    // consume the changed transient wind and reuse exactly the two invariants.
    const auto first_du    = du;
    auto       predictor_u = u_field;
    predictor_u[at(4, 4, 3)] += 1.25f;
    slow_tendency.u_data = predictor_u.data();
    REQUIRE(backend.dispatch_cartesian_cgrid_slow_tendencies(slow_tendency));
    REQUIRE(du != first_du);
    const auto predictor_du = du, predictor_dv = dv, predictor_dw = dw, predictor_drho = drho, predictor_dp = dp;

    // Selective invalidation forces only rho back across the host/device
    // boundary. qv remains resident, and the numerical result is bitwise
    // identical because neither invariant changed.
    backend.invalidate_resident_field("rho");
    REQUIRE(backend.dispatch_cartesian_cgrid_slow_tendencies(slow_tendency));
    REQUIRE(du == predictor_du);
    REQUIRE(dv == predictor_dv);
    REQUIRE(dw == predictor_dw);
    REQUIRE(drho == predictor_drho);
    REQUIRE(dp == predictor_dp);

    const auto unchanged = du;
    const auto completed = backend.completed_command_buffer_count();
    slow_tendency.nr     = 0;
    REQUIRE_FALSE(backend.dispatch_cartesian_cgrid_slow_tendencies(slow_tendency));
    REQUIRE(backend.completed_command_buffer_count() == completed);
    REQUIRE(du == unchanged);
    slow_tendency.nr                   = nx;
    slow_tendency.dv_advection_dt_data = nullptr;
    REQUIRE_FALSE(backend.dispatch_cartesian_cgrid_slow_tendencies(slow_tendency));
    REQUIRE(backend.completed_command_buffer_count() == completed);
    REQUIRE(du == unchanged);
    slow_tendency.dv_advection_dt_data = av.data();
    slow_tendency.rho0_base_len        = nz_levels - 1;
    REQUIRE_FALSE(backend.dispatch_cartesian_cgrid_slow_tendencies(slow_tendency));
    REQUIRE(du == unchanged);
    const auto record = backend.execution_record();
    REQUIRE(record.completed_cartesian_cgrid_slow_tendency_dispatches == 4);
    REQUIRE(record.slow_tendency_invariant_input_uploads == 3);
    REQUIRE(record.slow_tendency_invariant_input_reuses == 3);
    REQUIRE(record.slow_apply_resident_tendency_reuses == 4);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::CartesianCgridSlowTendencies) == 4);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::cartesian_cgrid_slow_tendencies_msl_source()) ==
            record.cartesian_cgrid_slow_tendencies_msl_source_hash);
    std::cout << "[metal-cgrid-slow-evidence] dispatches=" << record.completed_cartesian_cgrid_slow_tendency_dispatches
              << " invariant_uploads=" << record.slow_tendency_invariant_input_uploads
              << " invariant_reuses=" << record.slow_tendency_invariant_input_reuses
              << " retained_handoff=" << record.slow_apply_resident_tendency_reuses << " max_abs=" << max_error
              << " moist_loading=true beta_plane=true frame=(2.25,-0.75) msl_fnv1a64="
              << record.cartesian_cgrid_slow_tendencies_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal Cartesian A-grid slow and acoustic families preserve the canonical operators",
          "[metal][physical][gpu_parity][dynamics][agrid][resident][batched]")
{
    constexpr int      nx = 7, ny = 6, nz = 5;
    constexpr float    dx = 120.0f, dy = 95.0f, dzv = 80.0f;
    const std::size_t  cells = static_cast<std::size_t>(nx) * ny * nz;
    const auto         at    = [](int i, int j, int k) { return (static_cast<std::size_t>(i) * ny + j) * nz + k; };
    std::vector<float> u(cells), v(cells), w(cells), rho_f(cells), pressure(cells), theta_f(cells), loading(cells),
        qv_f(cells);
    std::vector<float> p0(nz), rho0(nz), u0(nz), v0(nz), qv0(nz);
    for (int k = 0; k < nz; ++k)
    {
        p0[k]   = 100000.0f - 930.0f * k;
        rho0[k] = 1.18f - 0.015f * k;
        u0[k]   = 4.0f + 0.2f * k;
        v0[k]   = -1.0f + 0.1f * k;
        qv0[k]  = 0.012f - 0.0007f * k;
    }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                auto n      = at(i, j, k);
                u[n]        = 4.0f + 0.07f * i - 0.03f * j + 0.02f * k;
                v[n]        = -1.0f + 0.04f * i + 0.05f * j - 0.01f * k;
                w[n]        = 0.02f * i - 0.015f * j + 0.03f * k;
                rho_f[n]    = rho0[k] + 0.002f * i - 0.001f * j;
                pressure[n] = p0[k] + 4.0f * i - 3.0f * j + 1.5f * k;
                theta_f[n]  = 300.0f + 0.1f * k;
                loading[n]  = 1.0e-4f * float(1 + (i + j + k) % 4);
                qv_f[n]     = qv0[k] + 2.0e-4f * i - 1.0e-4f * j;
            }
    std::array<std::vector<float>, 5> out, expected;
    for (auto& f : out)
        f.assign(cells, 123.0f);
    for (auto& f : expected)
        f.assign(cells, 0.0f);
    CartesianTendenciesDispatch slow{u.data(),
                                     v.data(),
                                     w.data(),
                                     rho_f.data(),
                                     pressure.data(),
                                     theta_f.data(),
                                     p0.data(),
                                     rho0.data(),
                                     loading.data(),
                                     u0.data(),
                                     v0.data(),
                                     qv_f.data(),
                                     qv0.data(),
                                     out[0].data(),
                                     out[1].data(),
                                     out[2].data(),
                                     out[3].data(),
                                     out[4].data(),
                                     nx,
                                     ny,
                                     nz,
                                     dx,
                                     dy,
                                     dzv,
                                     9.81f,
                                     1.4f,
                                     1.0e-4f,
                                     2,
                                     4,
                                     0,
                                     0,
                                     2,
                                     true,
                                     2.0e-11f};
    MetalComputeBackend         slow_backend;
    initialize_backend(slow_backend);
    REQUIRE(slow_backend.supports(DynamicsKernelOp::CartesianTendencies));
    REQUIRE(slow_backend.dispatch_cartesian_tendencies(slow));
    REQUIRE(slow_backend.resident_field_count() == 5);
    const auto avg   = [](float ra, float rb, float va, float vb) { return 0.25f * (ra + rb) * (va + vb); };
    const auto xface = [&](int face, int j, int k)
    {
        if (face == 0)
            return 0.0f;
        if (face == nx - 2)
            return rho_f[at(nx - 1, j, k)] * u[at(nx - 1, j, k)];
        return avg(rho_f[at(face, j, k)], rho_f[at(face + 1, j, k)], u[at(face, j, k)], u[at(face + 1, j, k)]);
    };
    const auto yface = [&](int i, int face, int k)
    {
        if (face == 0 || face == ny - 2)
            return avg(rho_f[at(i, ny - 2, k)], rho_f[at(i, 1, k)], v[at(i, ny - 2, k)], v[at(i, 1, k)]);
        return avg(rho_f[at(i, face, k)], rho_f[at(i, face + 1, k)], v[at(i, face, k)], v[at(i, face + 1, k)]);
    };
    const auto zface = [&](int i, int j, int face)
    {
        if (face == 0)
            return 0.0f;
        if (face == nz - 2)
            return rho_f[at(i, j, nz - 1)] * w[at(i, j, nz - 1)];
        return avg(rho_f[at(i, j, face)], rho_f[at(i, j, face + 1)], w[at(i, j, face)], w[at(i, j, face + 1)]);
    };
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                auto  n  = at(i, j, k);
                float ix = 1 / (2 * dx), iy = 1 / (2 * dy), iz = 1 / (2 * dzv);
                auto  ddx = [&](const std::vector<float>& f) { return (f[at(i + 1, j, k)] - f[at(i - 1, j, k)]) * ix; };
                auto  ddy = [&](const std::vector<float>& f) { return (f[at(i, j + 1, k)] - f[at(i, j - 1, k)]) * iy; };
                auto  ddz = [&](const std::vector<float>& f) { return (f[at(i, j, k + 1)] - f[at(i, j, k - 1)]) * iz; };
                float ux = u[n], vy = v[n], wz = w[n], r = rho_f[n], px = ddx(pressure), py = ddy(pressure),
                      pz            = ddz(pressure);
                const float local_f = slow.coriolis_f_val + slow.coriolis_beta * (float(j) + 0.5f) * dy;
                expected[0][n]      = -ux * ddx(u) - vy * ddy(u) - wz * ddz(u) - px / r + local_f * (vy - v0[k]);
                expected[1][n]      = -ux * ddx(v) - vy * ddy(v) - wz * ddz(v) - py / r - local_f * (ux - u0[k]);
                expected[2][n] = -ux * ddx(w) - vy * ddy(w) - wz * ddz(w) - (pz - (p0[k + 1] - p0[k - 1]) * iz) / r -
                                 9.81f * (r - rho0[k]) / r + 9.81f * 0.608f * (qv_f[n] - qv0[k]) - 9.81f * loading[n];
                expected[3][n] =
                    -((xface(i, j, k) - xface(i - 1, j, k)) / dx + (yface(i, j, k) - yface(i, j - 1, k)) / dy +
                      (zface(i, j, k) - zface(i, j, k - 1)) / dzv);
                expected[4][n] = -1.4f * pressure[n] * (ddx(u) + ddy(v) + ddz(w)) - ux * px - vy * py - wz * pz;
            }
    double       slow_max = 0;
    const double tol[5]   = {2e-5, 2e-5, 2e-5, 2e-6, 5e-2};
    for (int f = 0; f < 5; ++f)
        for (std::size_t n = 0; n < cells; ++n)
        {
            slow_max = std::max(slow_max, std::abs(double(out[f][n] - expected[f][n])));
            REQUIRE(std::abs(double(out[f][n] - expected[f][n])) <= tol[f]);
        }

    std::array<std::vector<float>, 5> initial{{u, v, w, rho_f, pressure}}, split = initial, fused = initial,
                                                                           sequential = initial, batched = initial;
    MetalComputeBackend               split_backend;
    initialize_backend(split_backend);
    std::vector<float> next_rho(cells), next_p(cells), next_u(cells), next_v(cells), next_w(cells);
    REQUIRE(split_backend.dispatch_acoustic_pressure({split[0].data(),
                                                      split[1].data(),
                                                      split[2].data(),
                                                      split[3].data(),
                                                      split[4].data(),
                                                      next_rho.data(),
                                                      next_p.data(),
                                                      nx,
                                                      ny,
                                                      nz,
                                                      dx,
                                                      dy,
                                                      dzv,
                                                      1.4f,
                                                      0.1f,
                                                      1e-6f,
                                                      1.0f}));
    REQUIRE(split_backend.dispatch_acoustic_momentum({next_rho.data(),
                                                      next_p.data(),
                                                      split[0].data(),
                                                      split[1].data(),
                                                      split[2].data(),
                                                      next_u.data(),
                                                      next_v.data(),
                                                      next_w.data(),
                                                      nx,
                                                      ny,
                                                      nz,
                                                      dx,
                                                      dy,
                                                      dzv,
                                                      0.1f,
                                                      120.0f,
                                                      90.0f}));
    split = {next_u, next_v, next_w, next_rho, next_p};
    MetalComputeBackend fused_backend;
    initialize_backend(fused_backend);
    REQUIRE(fused_backend.dispatch_acoustic_substep_fused({fused[0].data(),
                                                           fused[1].data(),
                                                           fused[2].data(),
                                                           fused[3].data(),
                                                           fused[4].data(),
                                                           nx,
                                                           ny,
                                                           nz,
                                                           dx,
                                                           dy,
                                                           dzv,
                                                           1.4f,
                                                           0.1f,
                                                           1e-6f,
                                                           1.0f,
                                                           120.0f,
                                                           90.0f}));
    for (int f = 0; f < 5; ++f)
        REQUIRE(std::memcmp(split[f].data(), fused[f].data(), cells * sizeof(float)) == 0);
    MetalComputeBackend sequence_backend;
    initialize_backend(sequence_backend);
    for (int n = 0; n < 3; ++n)
        REQUIRE(sequence_backend.dispatch_acoustic_substep_fused({sequential[0].data(),
                                                                  sequential[1].data(),
                                                                  sequential[2].data(),
                                                                  sequential[3].data(),
                                                                  sequential[4].data(),
                                                                  nx,
                                                                  ny,
                                                                  nz,
                                                                  dx,
                                                                  dy,
                                                                  dzv,
                                                                  1.4f,
                                                                  0.1f,
                                                                  1e-6f,
                                                                  1.0f,
                                                                  120.0f,
                                                                  90.0f}));
    MetalComputeBackend batch_backend;
    initialize_backend(batch_backend);
    REQUIRE(batch_backend.dispatch_acoustic_substeps_batched({batched[0].data(),
                                                              batched[1].data(),
                                                              batched[2].data(),
                                                              batched[3].data(),
                                                              batched[4].data(),
                                                              nx,
                                                              ny,
                                                              nz,
                                                              dx,
                                                              dy,
                                                              dzv,
                                                              1.4f,
                                                              0.1f,
                                                              3,
                                                              1e-6f,
                                                              1.0f,
                                                              120.0f,
                                                              90.0f}));
    for (int f = 0; f < 5; ++f)
        REQUIRE(std::memcmp(sequential[f].data(), batched[f].data(), cells * sizeof(float)) == 0);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            REQUIRE(batched[2][at(i, j, 0)] == 0.0f);
            REQUIRE(batched[2][at(i, j, nz - 1)] == 0.0f);
        }
    const auto unchanged = batched;
    auto       invalid   = AcousticSubstepsBatchedDispatch{batched[0].data(),
                                                           batched[1].data(),
                                                           batched[2].data(),
                                                           batched[3].data(),
                                                           batched[4].data(),
                                                           nx,
                                                           ny,
                                                           nz,
                                                           dx,
                                                           dy,
                                                           dzv,
                                                           1.4f,
                                                           0.1f,
                                                           0,
                                                           1e-6f,
                                                           1.0f,
                                                           120.0f,
                                                           90.0f};
    REQUIRE_FALSE(batch_backend.dispatch_acoustic_substeps_batched(invalid));
    invalid.n_substeps = 1;
    invalid.p          = invalid.rho;
    REQUIRE_FALSE(batch_backend.dispatch_acoustic_substeps_batched(invalid));
    for (int f = 0; f < 5; ++f)
        REQUIRE(std::memcmp(unchanged[f].data(), batched[f].data(), cells * sizeof(float)) == 0);
    auto record = batch_backend.execution_record();
    REQUIRE(record.completed_acoustic_agrid_batched_dispatches == 1);
    REQUIRE(record.completed_acoustic_agrid_pressure_dispatches == 3);
    REQUIRE(record.completed_acoustic_agrid_momentum_dispatches == 3);
    REQUIRE(record.cartesian_agrid_msl_source_hash.size() == 16);
    REQUIRE(record.acoustic_agrid_msl_source_hash.size() == 16);
    std::cout << "[metal-agrid-evidence] slow_cells=" << cells << " slow_max_abs=" << slow_max
              << " beta_plane=true split_buffers=2 fused_buffers=1 batched_substeps=3 command_buffers="
              << batch_backend.completed_command_buffer_count()
              << " slow_msl_fnv1a64=" << slow_backend.execution_record().cartesian_agrid_msl_source_hash
              << " acoustic_msl_fnv1a64=" << record.acoustic_agrid_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal production A-grid continuity exports its authoritative corrected mass flux",
          "[metal][physical][gpu_parity][dynamics][agrid][acoustic][mass_flux]")
{
    using tmv::cartesian_mass_flux::CollocatedPressureCoupling;
    using tmv::init::config::BottomBC;
    using tmv::init::config::LateralBC;
    using tmv::init::config::TopBC;
    constexpr int      nx = 8, ny = 7, nz_levels = 6;
    constexpr float    dx_value = 110.0f, dy_value = 90.0f;
    constexpr float    dz_value = 70.0f, dt_value = 0.08f;
    constexpr float    frame_u = 1.25f, frame_v = -0.5f;
    const std::size_t  cells = static_cast<std::size_t>(nx) * ny * nz_levels;
    const auto         at = [](int i, int j, int k) { return (static_cast<std::size_t>(i) * ny + j) * nz_levels + k; };
    std::vector<float> u_data(cells), v_data(cells), w_data(cells);
    std::vector<float> rho_data(cells), p_data(cells), p0_data(nz_levels);
    for (int k = 0; k < nz_levels; ++k)
        p0_data[static_cast<std::size_t>(k)] = 100000.0f - 820.0f * k;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_levels; ++k)
            {
                const std::size_t n = at(i, j, k);
                u_data[n]           = 3.0f + 0.08f * i - 0.03f * j + 0.01f * k;
                v_data[n]           = -1.0f + 0.04f * i + 0.06f * j - 0.02f * k;
                w_data[n]           = 0.01f * i - 0.015f * j + 0.025f * k;
                rho_data[n]         = 1.2f - 0.025f * k + 0.001f * i - 0.0007f * j;
                const float checker = ((i + j + k) & 1) ? 18.0f : -18.0f;
                p_data[n]           = p0_data[static_cast<std::size_t>(k)] + checker + 2.5f * i - 1.75f * j + 0.4f * k;
            }

    Field3D u_field(nx, ny, nz_levels), v_field(nx, ny, nz_levels);
    Field3D w_field(nx, ny, nz_levels), rho_field(nx, ny, nz_levels);
    Field3D p_field(nx, ny, nz_levels);
    std::memcpy(u_field.data(), u_data.data(), cells * sizeof(float));
    std::memcpy(v_field.data(), v_data.data(), cells * sizeof(float));
    std::memcpy(w_field.data(), w_data.data(), cells * sizeof(float));
    std::memcpy(rho_field.data(), rho_data.data(), cells * sizeof(float));
    std::memcpy(p_field.data(), p_data.data(), cells * sizeof(float));
    std::vector<double>              p0_double(p0_data.begin(), p0_data.end());
    const CollocatedPressureCoupling coupling{&p_field, &p0_double, dt_value, dx_value, dy_value, dz_value};

    std::vector<float>                               drho(cells, 777.0f), mfx(cells, 777.0f);
    std::vector<float>                               mfy(cells, 777.0f), mfz(cells, 777.0f);
    AcousticContinuityMassFluxCartesianAgridDispatch dispatch{u_data.data(),
                                                              v_data.data(),
                                                              w_data.data(),
                                                              rho_data.data(),
                                                              p_data.data(),
                                                              p0_data.data(),
                                                              nz_levels,
                                                              drho.data(),
                                                              mfx.data(),
                                                              mfy.data(),
                                                              mfz.data(),
                                                              nx,
                                                              ny,
                                                              nz_levels,
                                                              dx_value,
                                                              dy_value,
                                                              dz_value,
                                                              dt_value,
                                                              static_cast<int>(LateralBC::Periodic),
                                                              static_cast<int>(LateralBC::Periodic),
                                                              static_cast<int>(LateralBC::Periodic),
                                                              static_cast<int>(LateralBC::Periodic),
                                                              static_cast<int>(TopBC::Open),
                                                              false,
                                                              0.0f,
                                                              frame_u,
                                                              frame_v};

    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.supports(DynamicsKernelOp::AcousticContinuityMassFluxCartesianAgrid));
    REQUIRE(backend.dispatch_acoustic_continuity_mass_flux_cartesian_agrid(dispatch));

    double max_flux_error = 0.0, max_divergence_error = 0.0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_levels; ++k)
            {
                const std::size_t n          = at(i, j, k);
                const double      expected_x = i < nx - 1 ? tmv::cartesian_mass_flux::x_face(rho_field,
                                                                                             u_field,
                                                                                             i,
                                                                                             j,
                                                                                             k,
                                                                                             nx,
                                                                                             StaggerType::Collocated,
                                                                                             LateralBC::Periodic,
                                                                                             LateralBC::Periodic,
                                                                                             &coupling,
                                                                                             frame_u)
                                                          : 0.0;
                const double      expected_y = j < ny - 1 ? tmv::cartesian_mass_flux::y_face(rho_field,
                                                                                             v_field,
                                                                                             i,
                                                                                             j,
                                                                                             k,
                                                                                             ny,
                                                                                             StaggerType::Collocated,
                                                                                             LateralBC::Periodic,
                                                                                             LateralBC::Periodic,
                                                                                             &coupling,
                                                                                             frame_v)
                                                          : 0.0;
                const double expected_z = k < nz_levels - 1 ? tmv::cartesian_mass_flux::z_face(rho_field,
                                                                                               w_field,
                                                                                               i,
                                                                                               j,
                                                                                               k,
                                                                                               nz_levels,
                                                                                               StaggerType::Collocated,
                                                                                               BottomBC::RigidSlip,
                                                                                               TopBC::Open,
                                                                                               &coupling)
                                                            : 0.0;
                max_flux_error          = std::max({max_flux_error,
                                                    std::abs(double(mfx[n]) - expected_x),
                                                    std::abs(double(mfy[n]) - expected_y),
                                                    std::abs(double(mfz[n]) - expected_z)});
                REQUIRE(std::abs(double(mfx[n]) - expected_x) <= 3.0e-5 * std::max(1.0, std::abs(expected_x)));
                REQUIRE(std::abs(double(mfy[n]) - expected_y) <= 3.0e-5 * std::max(1.0, std::abs(expected_y)));
                REQUIRE(std::abs(double(mfz[n]) - expected_z) <= 3.0e-5 * std::max(1.0, std::abs(expected_z)));
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz_levels - 1)
                {
                    REQUIRE(drho[n] == 0.0f);
                }
                else
                {
                    const double retained_divergence = (double(mfx[n]) - double(mfx[at(i - 1, j, k)])) / dx_value +
                                                       (double(mfy[n]) - double(mfy[at(i, j - 1, k)])) / dy_value +
                                                       (double(mfz[n]) - double(mfz[at(i, j, k - 1)])) / dz_value;
                    max_divergence_error =
                        std::max(max_divergence_error, std::abs(double(drho[n]) + retained_divergence));
                    REQUIRE(std::abs(double(drho[n]) + retained_divergence) <= 2.0e-6);
                }
            }

    const auto first_drho = drho;
    const auto first_mfx = mfx, first_mfy = mfy, first_mfz = mfz;
    dispatch.accumulate_mass_flux = true;
    dispatch.mass_flux_weight     = 0.25f;
    REQUIRE(backend.dispatch_acoustic_continuity_mass_flux_cartesian_agrid(dispatch));
    REQUIRE(drho == first_drho);
    for (std::size_t n = 0; n < cells; ++n)
    {
        REQUIRE(mfx[n] == first_mfx[n] + 0.25f * first_mfx[n]);
        REQUIRE(mfy[n] == first_mfy[n] + 0.25f * first_mfy[n]);
        REQUIRE(mfz[n] == first_mfz[n] + 0.25f * first_mfz[n]);
    }

    const auto committed_drho = drho, committed_mfx = mfx;
    const auto completed = backend.completed_command_buffer_count();
    dispatch.nx          = 3;
    REQUIRE_FALSE(backend.dispatch_acoustic_continuity_mass_flux_cartesian_agrid(dispatch));
    REQUIRE(drho == committed_drho);
    REQUIRE(mfx == committed_mfx);
    REQUIRE(backend.completed_command_buffer_count() == completed);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::AcousticContinuityMassFluxCartesianAgrid) == 2);
    const auto record = backend.execution_record();
    REQUIRE(record.completed_acoustic_agrid_continuity_dispatches == 2);
    REQUIRE(record.acoustic_agrid_continuity_msl_source_hash.size() == 16);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::acoustic_agrid_continuity_msl_source()) ==
            record.acoustic_agrid_continuity_msl_source_hash);
    std::cout << "[metal-agrid-continuity] max_flux_abs=" << max_flux_error
              << " max_retained_divergence_abs=" << max_divergence_error
              << " dispatches=2 device_ms=" << record.total_device_milliseconds
              << " msl_fnv1a64=" << record.acoustic_agrid_continuity_msl_source_hash << '\n';
}

TEST_CASE("Metal native A-grid completion stages match the CPU operators atomically",
          "[metal][physical][gpu_parity][dynamics][agrid][acoustic][complete]")
{
    MetalComputeBackend backend;
    initialize_backend(backend);
    const auto metrics = nimbus::test::run_agrid_complete_native_fixture(backend);
    const auto record  = backend.execution_record();
    REQUIRE(record.completed_acoustic_agrid_pressure_eos_dispatches == 1);
    REQUIRE(record.completed_acoustic_agrid_native_momentum_dispatches == 1);
    REQUIRE(record.completed_acoustic_agrid_constraint_dispatches == 1);
    REQUIRE(record.completed_acoustic_agrid_boundary_dispatches == 1);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::AcousticPressureEosCartesianAgrid) == 1);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::AcousticMomentumCartesianAgrid) == 1);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::AcousticNullModeConstraintCartesianAgrid) == 1);
    REQUIRE(backend.successful_kernel_dispatch_count(KernelId::AcousticBoundaryCartesianAgrid) == 1);
    REQUIRE(record.acoustic_agrid_pressure_eos_msl_source_hash.size() == 16);
    REQUIRE(record.acoustic_agrid_native_momentum_msl_source_hash.size() == 16);
    REQUIRE(record.acoustic_agrid_constraint_msl_source_hash.size() == 16);
    REQUIRE(record.acoustic_agrid_boundary_msl_source_hash.size() == 16);
    REQUIRE(record.acoustic_agrid_pressure_eos_msl_source_hash ==
            MetalComputeBackend::metal_artifact_source_hash(KernelId::AcousticPressureEosCartesianAgrid));
    REQUIRE(record.acoustic_agrid_native_momentum_msl_source_hash ==
            MetalComputeBackend::metal_artifact_source_hash(KernelId::AcousticMomentumCartesianAgrid));
    REQUIRE(record.acoustic_agrid_constraint_msl_source_hash ==
            MetalComputeBackend::metal_artifact_source_hash(KernelId::AcousticNullModeConstraintCartesianAgrid));
    REQUIRE(record.acoustic_agrid_boundary_msl_source_hash ==
            MetalComputeBackend::metal_artifact_source_hash(KernelId::AcousticBoundaryCartesianAgrid));
    std::cout << "[metal-agrid-complete] pressure_max_abs=" << metrics.pressure_max_abs
              << " momentum_max_abs=" << metrics.momentum_max_abs
              << " constraint_max_abs=" << metrics.constraint_max_abs << " slow_max_abs=" << metrics.slow_max_abs
              << " command_buffers=" << backend.completed_command_buffer_count()
              << " pressure_eos_msl_fnv1a64=" << record.acoustic_agrid_pressure_eos_msl_source_hash
              << " momentum_msl_fnv1a64=" << record.acoustic_agrid_native_momentum_msl_source_hash
              << " constraint_msl_fnv1a64=" << record.acoustic_agrid_constraint_msl_source_hash
              << " boundary_msl_fnv1a64=" << record.acoustic_agrid_boundary_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal Cartesian C-grid acoustic split and fused stages preserve the canonical physics",
          "[metal][physical][gpu_parity][dynamics][cgrid][acoustic][residency]")
{
    constexpr int         nx = 7, ny = 6, nz_levels = 5;
    constexpr std::size_t cells = nx * ny * nz_levels;
    constexpr float       dx = 360.0f, dy = 440.0f, dz_value = 180.0f, dt_small = .075f, gamma = 1.4f;
    const auto         at = [](int i, int j, int k) { return static_cast<std::size_t>((i * ny + j) * nz_levels + k); };
    std::vector<float> rho0(nz_levels), p0(nz_levels), dp0(nz_levels), theta(cells), u(cells), v(cells), w(cells),
        rho(cells), p(cells);
    for (int k = 0; k < nz_levels; ++k)
    {
        rho0[k] = 1.18f - .035f * k;
        p0[k]   = 100000.0f - 1900.0f * k;
    }
    for (int k = 0; k < nz_levels - 1; ++k)
        dp0[k] = float((double(p0[k + 1]) - double(p0[k])) / double(dz_value));
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_levels; ++k)
            {
                auto n   = at(i, j, k);
                theta[n] = 299.0f + .35f * k + .04f * i - .03f * j;
                rho[n]   = rho0[k] + .0012f * i - .0007f * j + .0003f * k;
                p[n]     = p0[k] + 2.5f * i - 1.75f * j + .4f * k;
                u[n]     = 5.0f + .18f * i - .08f * j + .03f * k;
                v[n]     = -2.0f + .05f * i + .14f * j - .02f * k;
                w[n]     = .12f + .015f * i - .011f * j + .025f * k;
            }
    std::vector<float> split_rho(cells, -1), split_p(cells, -1), split_u(cells, -1), split_v(cells, -1),
        split_w(cells, -1);
    MetalComputeBackend backend;
    initialize_backend(backend);
    backend.begin_resident_step();
    REQUIRE(backend.supports(DynamicsKernelOp::AcousticPressureCartesianCgrid));
    REQUIRE(backend.supports(DynamicsKernelOp::AcousticMomentumCartesianCgrid));
    REQUIRE(backend.supports(DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid));
    AcousticPressureCartesianCgridDispatch pressure{};
    pressure.u                   = u.data();
    pressure.v                   = v.data();
    pressure.w                   = w.data();
    pressure.rho_in              = rho.data();
    pressure.p_in                = p.data();
    pressure.rho0_base_data      = rho0.data();
    pressure.p0_base_data        = p0.data();
    pressure.base_len            = nz_levels;
    pressure.theta_data          = theta.data();
    pressure.rho_out             = split_rho.data();
    pressure.p_out               = split_p.data();
    pressure.nx                  = nx;
    pressure.ny                  = ny;
    pressure.nz                  = nz_levels;
    pressure.dx                  = dx;
    pressure.dy                  = dy;
    pressure.dz                  = dz_value;
    pressure.gamma_val           = gamma;
    pressure.dt_small            = dt_small;
    pressure.rho_min             = .05f;
    pressure.rho_max             = 5.0f;
    pressure.p_min               = 1000.0f;
    pressure.p_max               = 200000.0f;
    pressure.p_ref_fallback      = 100000.0f;
    pressure.frame_velocity_u_ms = 1.25f;
    pressure.frame_velocity_v_ms = -.5f;
    REQUIRE(backend.dispatch_acoustic_pressure_cartesian_cgrid(pressure));
    AcousticMomentumCartesianCgridDispatch momentum{};
    momentum.rho_data     = split_rho.data();
    momentum.p_data       = split_p.data();
    momentum.dp0_dz_data  = dp0.data();
    momentum.base_len     = nz_levels;
    momentum.u_in         = u.data();
    momentum.v_in         = v.data();
    momentum.w_in         = w.data();
    momentum.u_out        = split_u.data();
    momentum.v_out        = split_v.data();
    momentum.w_out        = split_w.data();
    momentum.nx           = nx;
    momentum.ny           = ny;
    momentum.nz           = nz_levels;
    momentum.dx           = dx;
    momentum.dy           = dy;
    momentum.dz           = dz_value;
    momentum.dt_small     = dt_small;
    momentum.wind_clamp_h = 150.0f;
    momentum.wind_clamp_v = 75.0f;
    REQUIRE(backend.dispatch_acoustic_momentum_cartesian_cgrid(momentum));

    std::vector<float> erho(cells), ep(cells), eu(cells), ev(cells), ew(cells);
    double             max_error = 0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_levels; ++k)
            {
                auto  n  = at(i, j, k);
                float dr = 0;
                if (i >= 1 && i <= nx - 2 && j >= 1 && j <= ny - 2)
                {
                    float fxp = .5f * (rho[n] + rho[at(i + 1, j, k)]) * (u[n] - 1.25f),
                          fxm = .5f * (rho[at(i - 1, j, k)] + rho[n]) * (u[at(i - 1, j, k)] - 1.25f);
                    float fyp = .5f * (rho[n] + rho[at(i, j + 1, k)]) * (v[n] + .5f),
                          fym = .5f * (rho[at(i, j - 1, k)] + rho[n]) * (v[at(i, j - 1, k)] + .5f);
                    float fzp = k + 1 < nz_levels ? .5f * (rho[n] + rho[at(i, j, k + 1)]) * w[n] : rho[n] * w[n];
                    float fzm = k > 0 ? .5f * (rho[at(i, j, k - 1)] + rho[n]) * w[at(i, j, k - 1)] : 0;
                    dr        = -((fxp - fxm) / dx + (fyp - fym) / dy + (fzp - fzm) / dz_value);
                    if (!std::isfinite(dr))
                        dr = 0;
                }
                erho[n] = std::clamp(rho[n] + dr * dt_small, .05f, 5.0f);
                ep[n]   = std::clamp(
                    100000.0f * std::pow(erho[n] * 287.0f * theta[n] / 100000.0f, gamma), 1000.0f, 200000.0f);
            }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_levels; ++k)
            {
                auto  n  = at(i, j, k);
                float du = 0, dv = 0, dw = 0;
                bool  active = i >= 1 && i <= nx - 2 && j >= 1 && j <= ny - 2 && k <= nz_levels - 2;
                if (active)
                {
                    du = -((ep[at(i + 1, j, k)] - ep[n]) / dx) / (.5f * (erho[n] + erho[at(i + 1, j, k)]));
                    dv = -((ep[at(i, j + 1, k)] - ep[n]) / dy) / (.5f * (erho[n] + erho[at(i, j + 1, k)]));
                    dw = -(((ep[at(i, j, k + 1)] - ep[n]) / dz_value) - dp0[k]) /
                         (.5f * (erho[n] + erho[at(i, j, k + 1)]));
                }
                eu[n] = std::clamp(u[n] + du * dt_small, -150.0f, 150.0f);
                ev[n] = std::clamp(v[n] + dv * dt_small, -150.0f, 150.0f);
                ew[n] = std::clamp(w[n] + dw * dt_small, -75.0f, 75.0f);
            }
    for (std::size_t n = 0; n < cells; ++n)
    {
        for (auto pair : {std::pair<float, float>{split_rho[n], erho[n]},
                          {split_p[n], ep[n]},
                          {split_u[n], eu[n]},
                          {split_v[n], ev[n]},
                          {split_w[n], ew[n]}})
            max_error = std::max(max_error, std::abs(double(pair.first - pair.second)));
        REQUIRE(std::abs(split_rho[n] - erho[n]) <= 2e-6f);
        REQUIRE(std::abs(split_p[n] - ep[n]) <= .05f);
        REQUIRE(std::abs(split_u[n] - eu[n]) <= 2e-5f);
        REQUIRE(std::abs(split_v[n] - ev[n]) <= 2e-5f);
        REQUIRE(std::abs(split_w[n] - ew[n]) <= 2e-5f);
    }

    AcousticSubstepFusedCartesianCgridDispatch fused{};
    fused.u                   = u.data();
    fused.v                   = v.data();
    fused.w                   = w.data();
    fused.rho                 = rho.data();
    fused.p                   = p.data();
    fused.theta_data          = theta.data();
    fused.rho0_base_data      = rho0.data();
    fused.p0_base_data        = p0.data();
    fused.dp0_dz_data         = dp0.data();
    fused.base_len            = nz_levels;
    fused.nx                  = nx;
    fused.ny                  = ny;
    fused.nz                  = nz_levels;
    fused.dx                  = dx;
    fused.dy                  = dy;
    fused.dz                  = dz_value;
    fused.gamma_val           = gamma;
    fused.dt_small            = dt_small;
    fused.rho_min             = .05f;
    fused.rho_max             = 5.0f;
    fused.p_min               = 1000.0f;
    fused.p_max               = 200000.0f;
    fused.p_ref_fallback      = 100000.0f;
    fused.wind_clamp_h        = 150.0f;
    fused.wind_clamp_v        = 75.0f;
    fused.frame_velocity_u_ms = 1.25f;
    fused.frame_velocity_v_ms = -.5f;
    REQUIRE(backend.dispatch_acoustic_substep_fused_cartesian_cgrid(fused));
    REQUIRE(u == split_u);
    REQUIRE(v == split_v);
    REQUIRE(w == split_w);
    REQUIRE(rho == split_rho);
    REQUIRE(p == split_p);
    const auto committed_u = u, committed_v = v, committed_w = w, committed_rho = rho, committed_p = p;
    const auto completed = backend.completed_command_buffer_count();
    CartesianCGridAcousticEnergyReportV1 unsupported{};
    fused.energy_report_host = &unsupported;
    REQUIRE_FALSE(backend.dispatch_acoustic_substep_fused_cartesian_cgrid(fused));
    REQUIRE(backend.completed_command_buffer_count() == completed);
    REQUIRE(u == committed_u);
    REQUIRE(v == committed_v);
    REQUIRE(w == committed_w);
    REQUIRE(rho == committed_rho);
    REQUIRE(p == committed_p);
    fused.energy_report_host = nullptr;
    REQUIRE(backend.dispatch_acoustic_substep_fused_cartesian_cgrid(fused));
    const auto resident_before = backend.resident_field_count();
    backend.release_acoustic_loop_residency();
    REQUIRE(backend.resident_field_count() + 9 == resident_before);
    const auto unchanged = split_rho;
    pressure.base_len    = nz_levels - 1;
    REQUIRE_FALSE(backend.dispatch_acoustic_pressure_cartesian_cgrid(pressure));
    REQUIRE(split_rho == unchanged);
    const auto record = backend.execution_record();
    REQUIRE(record.completed_acoustic_pressure_dispatches == 3);
    REQUIRE(record.completed_acoustic_momentum_dispatches == 3);
    REQUIRE(record.completed_acoustic_fused_dispatches == 2);
    REQUIRE(record.acoustic_invariant_uploads == 4);
    REQUIRE(record.acoustic_invariant_reuses == 4);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::acoustic_cartesian_cgrid_msl_source()) ==
            record.acoustic_cartesian_cgrid_msl_source_hash);
    std::cout << "[metal-acoustic-cgrid-evidence] split=2 fused=" << record.completed_acoustic_fused_dispatches
              << " invariant_uploads=" << record.acoustic_invariant_uploads
              << " invariant_reuses=" << record.acoustic_invariant_reuses << " max_abs=" << max_error
              << " atomic_report_reject=true msl_fnv1a64=" << record.acoustic_cartesian_cgrid_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal batched acoustics preserve face order and export the acoustic mass-flux history",
          "[metal][physical][gpu_parity][dynamics][cgrid][acoustic][batch][boundary]")
{
    constexpr int         nx = 7, ny = 6, nz_levels = 5, substeps = 3;
    constexpr std::size_t cells = nx * ny * nz_levels;
    constexpr float       dx = 325.0f, dy = 410.0f, dz_value = 175.0f, dt_small = .045f;
    const auto         at = [](int i, int j, int k) { return static_cast<std::size_t>((i * ny + j) * nz_levels + k); };
    std::vector<float> rho0(nz_levels), p0(nz_levels), dp0(nz_levels), theta(cells), base_u(cells), base_v(cells),
        base_w(cells), base_rho(cells), base_p(cells);
    for (int k = 0; k < nz_levels; ++k)
    {
        rho0[k] = 1.2f - .03f * k;
        p0[k]   = 100000.0f - 1700.0f * k;
    }
    for (int k = 0; k < nz_levels - 1; ++k)
        dp0[k] = float((double(p0[k + 1]) - double(p0[k])) / double(dz_value));
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_levels; ++k)
            {
                auto n      = at(i, j, k);
                theta[n]    = 300.0f + .25f * k + .03f * i - .02f * j;
                base_rho[n] = rho0[k] + .0008f * i - .0005f * j;
                base_p[n]   = p0[k] + 1.8f * i - 1.1f * j;
                base_u[n]   = 4.0f + .12f * i - .06f * j + .02f * k;
                base_v[n]   = -1.5f + .04f * i + .1f * j - .015f * k;
                base_w[n]   = .08f + .01f * i - .008f * j + .018f * k;
            }
    auto make_dispatch = [&](std::vector<float>& u,
                             std::vector<float>& v,
                             std::vector<float>& w,
                             std::vector<float>& rho,
                             std::vector<float>& p,
                             std::vector<float>& fx,
                             std::vector<float>& fy,
                             std::vector<float>& fz,
                             int                 count)
    {
        AcousticSubstepsBatchedCartesianCgridDispatch d{};
        d.u                    = u.data();
        d.v                    = v.data();
        d.w                    = w.data();
        d.rho                  = rho.data();
        d.p                    = p.data();
        d.theta_data           = theta.data();
        d.rho0_base_data       = rho0.data();
        d.p0_base_data         = p0.data();
        d.dp0_dz_data          = dp0.data();
        d.base_len             = nz_levels;
        d.nx                   = nx;
        d.ny                   = ny;
        d.nz                   = nz_levels;
        d.dx                   = dx;
        d.dy                   = dy;
        d.dz                   = dz_value;
        d.gamma_val            = 1.4f;
        d.dt_small             = dt_small;
        d.n_substeps           = count;
        d.rho_min              = .05f;
        d.rho_max              = 5.0f;
        d.p_min                = 1000.0f;
        d.p_max                = 200000.0f;
        d.p_ref_fallback       = 100000.0f;
        d.wind_clamp_h         = 150.0f;
        d.wind_clamp_v         = 75.0f;
        d.bc_west              = 0;
        d.bc_east              = 3;
        d.bc_south             = 4;
        d.bc_north             = 2;
        d.bc_bottom            = 0;
        d.bc_top               = 2;
        d.mass_flux_x          = fx.data();
        d.mass_flux_y          = fy.data();
        d.mass_flux_z          = fz.data();
        d.accumulate_mass_flux = true;
        d.frame_velocity_u_ms  = 1.0f;
        d.frame_velocity_v_ms  = -.4f;
        return d;
    };

    auto                bu = base_u, bv = base_v, bw = base_w, br = base_rho, bp = base_p;
    std::vector<float>  bfx(cells, -9), bfy(cells, -9), bfz(cells, -9);
    MetalComputeBackend batched;
    initialize_backend(batched);
    batched.begin_resident_step();
    REQUIRE(batched.supports(DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid));
    REQUIRE(batched.supports_batched_cartesian_cgrid_mass_flux_export());
    auto       batch     = make_dispatch(bu, bv, bw, br, bp, bfx, bfy, bfz, substeps);
    const auto before_cb = batched.completed_command_buffer_count();
    REQUIRE(batched.dispatch_acoustic_substeps_batched_cartesian_cgrid(batch));
    REQUIRE(batched.completed_command_buffer_count() == before_cb + 1);

    auto                ru = base_u, rv = base_v, rw = base_w, rr = base_rho, rp = base_p;
    std::vector<float>  rfx(cells), rfy(cells), rfz(cells), avgx(cells), avgy(cells), avgz(cells);
    MetalComputeBackend reference;
    initialize_backend(reference);
    reference.begin_resident_step();
    for (int step = 0; step < substeps; ++step)
    {
        auto one = make_dispatch(ru, rv, rw, rr, rp, rfx, rfy, rfz, 1);
        REQUIRE(reference.dispatch_acoustic_substeps_batched_cartesian_cgrid(one));
        for (std::size_t n = 0; n < cells; ++n)
        {
            avgx[n] += (1.0f / substeps) * rfx[n];
            avgy[n] += (1.0f / substeps) * rfy[n];
            avgz[n] += (1.0f / substeps) * rfz[n];
        }
    }
    REQUIRE(bu == ru);
    REQUIRE(bv == rv);
    REQUIRE(bw == rw);
    REQUIRE(br == rr);
    REQUIRE(bp == rp);
    double max_flux_error = 0;
    for (std::size_t n = 0; n < cells; ++n)
    {
        max_flux_error = std::max({max_flux_error,
                                   std::abs(double(bfx[n] - avgx[n])),
                                   std::abs(double(bfy[n] - avgy[n])),
                                   std::abs(double(bfz[n] - avgz[n]))});
        REQUIRE(std::abs(bfx[n] - avgx[n]) <= 2e-6f);
        REQUIRE(std::abs(bfy[n] - avgy[n]) <= 2e-6f);
        REQUIRE(std::abs(bfz[n] - avgz[n]) <= 2e-6f);
    }
    for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz_levels; ++k)
        {
            if (j < ny - 1)
                REQUIRE(br[at(0, j, k)] == br[at(nx - 2, j, k)]);
            if (k > 0)
            {
                REQUIRE(bu[at(nx - 2, j, k)] == 1.0f);
                REQUIRE(bu[at(nx - 1, j, k)] == 1.0f);
            }
        }
    for (int i = 0; i < nx; ++i)
        for (int k = 0; k < nz_levels; ++k)
        {
            REQUIRE(bv[at(i, 0, k)] == bv[at(i, 1, k)]);
            if (k > 0)
            {
                REQUIRE(bv[at(i, ny - 2, k)] == -.4f);
                REQUIRE(bv[at(i, ny - 1, k)] == -.4f);
            }
        }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            REQUIRE(bu[at(i, j, 0)] == 0.0f);
            REQUIRE(bv[at(i, j, 0)] == 0.0f);
            REQUIRE(bw[at(i, j, nz_levels - 1)] == bw[at(i, j, nz_levels - 2)]);
        }

    const auto committed_u = bu, committed_fx = bfx;
    const auto completed = batched.completed_command_buffer_count();
    batch.bc_west        = 5;
    REQUIRE_FALSE(batched.dispatch_acoustic_substeps_batched_cartesian_cgrid(batch));
    REQUIRE(batched.completed_command_buffer_count() == completed);
    REQUIRE(bu == committed_u);
    REQUIRE(bfx == committed_fx);
    batch.bc_west     = 0;
    batch.mass_flux_y = nullptr;
    REQUIRE_FALSE(batched.dispatch_acoustic_substeps_batched_cartesian_cgrid(batch));
    REQUIRE(bu == committed_u);
    REQUIRE(bfx == committed_fx);
    const auto record = batched.execution_record();
    REQUIRE(record.completed_acoustic_batched_dispatches == 1);
    REQUIRE(record.completed_acoustic_boundary_face_dispatches == 16);
    REQUIRE(record.completed_acoustic_mass_flux_exports == 1);
    REQUIRE(record.completed_acoustic_mass_flux_pressure_dispatches == 3);
    REQUIRE(batched.successful_kernel_dispatch_count(KernelId::AcousticBcFaceCartesianCgrid) == 16);
    REQUIRE(batched.successful_kernel_dispatch_count(KernelId::AcousticPressureMassFluxCartesianCgrid) == 3);
    REQUIRE(fnv1a64_hex(MetalComputeBackend::acoustic_batched_cartesian_cgrid_msl_source()) ==
            record.acoustic_batched_cartesian_cgrid_msl_source_hash);

    // Directly qualify the stateful KW78 branches away from face-order
    // corners. The radiation buffer is seeded from the pre-step normal
    // faces, then each condition consumes the post-momentum interior value.
    auto                xu = base_u, xv = base_v, xw = base_w, xr = base_rho, xp = base_p;
    std::vector<float>  dummyx(cells), dummyy(cells), dummyz(cells);
    MetalComputeBackend radiative;
    initialize_backend(radiative);
    radiative.begin_resident_step();
    auto rad    = make_dispatch(xu, xv, xw, xr, xp, dummyx, dummyy, dummyz, 1);
    rad.bc_west = rad.bc_east = rad.bc_south = rad.bc_north = 1;
    rad.bc_bottom                                           = 1;
    rad.bc_top                                              = 0;
    rad.accumulate_mass_flux                                = false;
    rad.mass_flux_x = rad.mass_flux_y = rad.mass_flux_z = nullptr;
    REQUIRE(radiative.dispatch_acoustic_substeps_batched_cartesian_cgrid(rad));
    for (int j = 1; j <= ny - 2; ++j)
        for (int k = 1; k <= nz_levels - 2; ++k)
        {
            float old = base_u[at(0, j, k)], phase = old - 1.0f - 30.0f,
                  expected = old - phase * dt_small * (xu[at(1, j, k)] - old) / dx;
            REQUIRE(std::abs(xu[at(0, j, k)] - expected) <= 1e-6f);
            old      = base_u[at(nx - 2, j, k)];
            phase    = old - 1.0f + 30.0f;
            expected = old - phase * dt_small * (old - xu[at(nx - 3, j, k)]) / dx;
            REQUIRE(std::abs(xu[at(nx - 2, j, k)] - expected) <= 1e-6f);
            REQUIRE(xu[at(nx - 1, j, k)] == xu[at(nx - 2, j, k)]);
        }
    for (int i = 1; i <= nx - 2; ++i)
        for (int k = 1; k <= nz_levels - 2; ++k)
        {
            float old = base_v[at(i, 0, k)], phase = old + .4f - 30.0f,
                  expected = old - phase * dt_small * (xv[at(i, 1, k)] - old) / dy;
            REQUIRE(std::abs(xv[at(i, 0, k)] - expected) <= 1e-6f);
            old      = base_v[at(i, ny - 2, k)];
            phase    = old + .4f + 30.0f;
            expected = old - phase * dt_small * (old - xv[at(i, ny - 3, k)]) / dy;
            REQUIRE(std::abs(xv[at(i, ny - 2, k)] - expected) <= 1e-6f);
            REQUIRE(xv[at(i, ny - 1, k)] == xv[at(i, ny - 2, k)]);
        }
    std::cout << "[metal-acoustic-batch-evidence] substeps=" << substeps
              << " command_buffers=1 boundary_faces=" << record.completed_acoustic_boundary_face_dispatches
              << " mass_flux_exports=" << record.completed_acoustic_mass_flux_exports
              << " max_flux_abs=" << max_flux_error
              << " msl_fnv1a64=" << record.acoustic_batched_cartesian_cgrid_msl_source_hash
              << " device_ms=" << record.total_device_milliseconds << '\n';
}

TEST_CASE("Metal SGS scalar walls preserve closed and periodic fluxes", "[metal][sgs_scalar_boundary]")
{
    MetalComputeBackend backend;
    initialize_backend(backend);
    REQUIRE(backend.turbulence_kernels() != nullptr);
    exercise_sgs_scalar_boundary(*backend.turbulence_kernels());
}

TEST_CASE("Metal acoustic validation preserves unrepaired candidates", "[metal][acoustic_unrepaired]")
{
    MetalComputeBackend backend;
    initialize_backend(backend);
    exercise_acoustic_unrepaired(*backend.dynamics_kernels());
}

TEST_CASE("Metal Cartesian positive scalar budget preserves tiny fields and rejects invalid batches",
          "[metal][physical][gpu_parity][advection][pd-contract]")
{
    MetalComputeBackend backend;
    initialize_backend(backend);
    nimbus::testing::exercise_positive_scalar_contract(backend);
    nimbus::testing::exercise_positive_scalar_scales(backend);
}

TEST_CASE("Metal retained-rho positive-definite transport remains nonnegative over repeated steps",
          "[metal][physical][gpu_parity][advection][pd-repeat]")
{
    constexpr int    nx = 18, ny = 18, nz = 18;
    constexpr double dt_step = 2.0;
    NR                       = nx;
    NTH                      = ny;
    NZ                       = nz;
    dr                       = 500.0;
    dtheta                   = 750.0;
    dz                       = 250.0;
    global_stagger_type      = StaggerType::CGrid;
    global_grid_geometry.initialize(nx, ny, nz, dr, dz, dtheta, CoordinateSystem::Cartesian, StaggerType::CGrid);
    SimulationStorageState storage{{nx, ny, nz, dr, dz, dt_step, dtheta}};
    storage.grid.geometry                   = global_grid_geometry;
    storage.configuration.coordinate_system = CoordinateSystem::Cartesian;
    storage.configuration.stagger_type      = StaggerType::CGrid;
    storage.configuration.lateral_bc_west   = tmv::init::config::LateralBC::RigidSlip;
    storage.configuration.lateral_bc_east   = tmv::init::config::LateralBC::RigidSlip;
    storage.configuration.lateral_bc_south  = tmv::init::config::LateralBC::RigidSlip;
    storage.configuration.lateral_bc_north  = tmv::init::config::LateralBC::RigidSlip;
    storage.configuration.bottom_bc         = tmv::init::config::BottomBC::RigidSlip;
    storage.configuration.top_bc            = tmv::init::config::TopBC::Rigid;
    auto& u                                 = storage.fields.u;
    auto& v                                 = storage.fields.v;
    auto& w                                 = storage.fields.w;
    auto& rho                               = storage.fields.rho;
    u.resize(nx, ny, nz, 0.0f);
    v.resize(nx, ny, nz, 0.0f);
    w.resize(nx, ny, nz, 0.0f);
    rho.resize(nx, ny, nz, 0.0f);
    auto                transport_view = make_simulation_state_view(storage);
    MetalComputeBackend backend;
    initialize_backend(backend);
    const auto interior_rho_q_sum = [](const Field3D& density, const Field3D& scalar)
    {
        double sum = 0.0;
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
                for (int k = 0; k < nz; ++k)
                    sum += double(density(i, j, k)) * double(scalar(i, j, k));
        return sum;
    };
    const auto run_metal_batch = [&](const float* const*                   inputs,
                                     float* const*                         outputs,
                                     const float*                          kappas,
                                     int                                   count,
                                     double                                dt,
                                     const int*                            positive,
                                     const CartesianCGridTransportContext* context)
    {
        ConservativeCartesianCgridTransportDispatch d{};
        d.execution_state        = &transport_view;
        d.u                      = u.data();
        d.v                      = v.data();
        d.w                      = w.data();
        d.rho_data               = rho.data();
        d.q_in                   = inputs;
        d.q_out                  = outputs;
        d.dt_kappas              = kappas;
        d.n_scalars              = count;
        d.positive_definite      = positive;
        d.nx                     = nx;
        d.ny                     = ny;
        d.nz                     = nz;
        d.inv_dx                 = float(1.0 / dr);
        d.inv_dy                 = float(1.0 / dtheta);
        d.inv_dz                 = float(1.0 / dz);
        d.inv_dx2                = d.inv_dx * d.inv_dx;
        d.inv_dy2                = d.inv_dy * d.inv_dy;
        d.inv_dz2                = d.inv_dz * d.inv_dz;
        d.dt_step                = float(dt);
        d.rho_floor              = 0.05f;
        d.density_begin          = context->density_begin.data();
        d.mass_flux_x            = context->mass_flux_x.data();
        d.mass_flux_y            = context->mass_flux_y.data();
        d.mass_flux_z            = context->mass_flux_z.data();
        d.use_supplied_mass_flux = true;
        return backend.dispatch_advect_scalars_conservative_cartesian_cgrid(d);
    };

    // This is the device path used by production after the acoustic
    // solve: scalar transport receives the beginning density and the
    // acoustic loop's retained FP32 face-mass-flux history.  Exercise a
    // sparse tracer repeatedly so a round-off-scale undershoot cannot
    // hide behind a one-step smooth-field parity comparison.
    constexpr double pi = 3.14159265358979323846;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double xi = static_cast<double>(i) / (NR - 1);
                const double yj = static_cast<double>(j) / (NTH - 1);
                const double zk = static_cast<double>(k) / (NZ - 1);
                rho(i, j, k) =
                    static_cast<float>(0.9 + 0.12 * std::sin(2.0 * pi * xi) * std::cos(2.0 * pi * yj) + 0.08 * zk);
                // Keep the prescribed compressible flow admissible over
                // the full 400 s stress interval.  The sparse tracer
                // still activates the donor limiter at zero/nonzero
                // interfaces, while density remains a valid denominator
                // instead of being drained by an indefinitely repeated
                // analytic divergence field.
                u(i, j, k) = (i >= 1 && i <= NR - 3)
                                 ? static_cast<float>(1.4 * std::sin(2.0 * pi * xi) + 0.35 * std::cos(2.0 * pi * yj))
                                 : 0.0f;
                v(i, j, k) = (j >= 1 && j <= NTH - 3)
                                 ? static_cast<float>(-1.2 * std::sin(2.0 * pi * yj) + 0.25 * std::cos(2.0 * pi * zk))
                                 : 0.0f;
                w(i, j, k) = (k <= NZ - 2)
                                 ? static_cast<float>(0.4 * std::sin(2.0 * pi * zk) + 0.1 * std::cos(2.0 * pi * xi))
                                 : 0.0f;
            }

    Field3D q_cpu(NR, NTH, NZ, 0.0f);
    for (int i = 2; i < NR - 2; ++i)
        for (int j = 2; j < NTH - 2; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                const unsigned hash = static_cast<unsigned>(i * 73856093u ^ j * 19349663u ^ k * 83492791u);
                if ((hash % 17u) == 0u)
                {
                    const int decade = static_cast<int>((hash / 17u) % 6u);
                    q_cpu(i, j, k)   = static_cast<float>(1.0e-3 * std::pow(0.1, decade));
                }
            }
    Field3D     q_gpu                     = q_cpu;
    const float dtk[1]                    = {0.0f};
    const int   pd[1]                     = {1};
    double      worst_mass_relative_error = 0.0;

    for (int step = 1; step <= 200; ++step)
    {
        const Field3D                  rho_begin = rho;
        CartesianCGridTransportContext context;
        context.begin(transport_view, NR, NTH, NZ, dt_step);
        context.accumulate(
            transport_view, rho_begin, u, v, w, dt_step, nullptr, nullptr, tmv::AcousticMassFluxPrecision::NativeFloat);
        context.complete(transport_view);

        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double mx        = static_cast<double>(context.mass_flux_x(i, j, k)) -
                                             static_cast<double>(context.mass_flux_x(i - 1, j, k));
                    const double my        = static_cast<double>(context.mass_flux_y(i, j, k)) -
                                             static_cast<double>(context.mass_flux_y(i, j - 1, k));
                    const double mz_top    = static_cast<double>(context.mass_flux_z(i, j, k));
                    const double mz_bottom = k >= 1 ? static_cast<double>(context.mass_flux_z(i, j, k - 1)) : 0.0;
                    rho(i, j, k) = static_cast<float>(static_cast<double>(rho_begin(i, j, k)) -
                                                      dt_step * (mx / dr + my / dtheta + (mz_top - mz_bottom) / dz));
                }

        const double mass_before = interior_rho_q_sum(rho_begin, q_gpu);
        Field3D      cpu_next;
        advect_scalar_conservative_3d_cartesian(transport_view, q_cpu, cpu_next, dt_step, true, &context);

        const float* q_in[1]  = {q_gpu.data()};
        float*       q_out[1] = {q_gpu.data()};
        REQUIRE(run_metal_batch(q_in, q_out, dtk, 1, dt_step, pd, &context));
        q_cpu = std::move(cpu_next);

        double maximum_error = 0.0;
        for (std::size_t index = 0; index < q_cpu.size(); ++index)
            maximum_error =
                std::max(maximum_error, std::abs(double(q_cpu.data()[index]) - double(q_gpu.data()[index])));
        INFO("step=" << step << " maximum CPU/Metal error=" << maximum_error);
        REQUIRE(maximum_error <= 2.0e-5);

        float gpu_min = std::numeric_limits<float>::infinity();
        int   min_i   = -1;
        int   min_j   = -1;
        int   min_k   = -1;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                    if (q_gpu(i, j, k) < gpu_min)
                    {
                        gpu_min = q_gpu(i, j, k);
                        min_i   = i;
                        min_j   = j;
                        min_k   = k;
                    }
        const double mass_after = interior_rho_q_sum(rho, q_gpu);
        const double relative_mass_error =
            std::abs(mass_after - mass_before) / std::max(std::abs(mass_before), 1.0e-30);
        worst_mass_relative_error = std::max(worst_mass_relative_error, relative_mass_error);
        INFO("step=" << step << " minimum=" << gpu_min << " at (" << min_i << "," << min_j << "," << min_k
                     << ") relative_mass_error=" << relative_mass_error);
        REQUIRE(gpu_min >= 0.0f);
    }
    INFO("worst repeated coupled rho-q relative error=" << worst_mass_relative_error);
    REQUIRE(worst_mass_relative_error < 3.0e-6);
}
