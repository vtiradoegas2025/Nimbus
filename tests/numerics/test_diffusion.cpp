// The analytical checks use the discrete vertical Laplacian
// K(φ[k-1] - 2φ[k] + φ[k+1]) / dz².
#include "catch2/catch.hpp"
#include "compute/compute_backend.hpp"
#include "core/field/field3d.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/orchestration/physics/diffusion_step.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/diffusion/linear_height_profile.hpp"
#include "numerics/diffusion/schemes/explicit/explicit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

class RecordingDiffusionBackend final : public ComputeBackend,
                                        public DiffusionComputeKernels
{
public:
    std::vector<DiffusionFieldLocation> locations;
    bool valid_contract = true;

    std::string name() const override { return "recording-vulkan"; }
    bool initialize(std::string&) override { return true; }
    void shutdown() override {}
    DiffusionComputeKernels* diffusion_kernels() override { return this; }
    const DiffusionComputeKernels* diffusion_kernels() const override
    {
        return this;
    }
    bool supports(DiffusionKernelOp op) const override
    {
        return op == DiffusionKernelOp::Diffusion;
    }
    bool dispatch_diffusion(const DiffusionDispatch& dispatch) override
    {
        valid_contract = valid_contract && dispatch.field != nullptr &&
            dispatch.rho != nullptr && dispatch.dz != nullptr &&
            dispatch.tendency != nullptr && dispatch.nx > 0 &&
            dispatch.ny > 0 && dispatch.nz > 0 && dispatch.dx > 0.0f &&
            dispatch.dy > 0.0f && dispatch.K_h == 2.0f &&
            dispatch.K_v == 3.0f && dispatch.c_grid &&
            dispatch.periodic_x && !dispatch.periodic_y;
        locations.push_back(dispatch.location);
        const std::size_t cells = static_cast<std::size_t>(dispatch.nx) *
            static_cast<std::size_t>(dispatch.ny) *
            static_cast<std::size_t>(dispatch.nz);
        std::fill(dispatch.tendency, dispatch.tendency + cells,
                  10.0f + static_cast<float>(dispatch.location));
        return true;
    }
};

struct ScopedRecordingBackend
{
    ComputeBackendConfig previous_config = global_compute_backend_config;

    ScopedRecordingBackend(std::unique_ptr<ComputeBackend> backend,
                           ComputeBackendKind kind)
    {
        global_compute_backend_runtime_state.reset();
        global_compute_backend_runtime_state.backend = std::move(backend);
        global_compute_backend_runtime_state.active_kind = kind;
        global_compute_backend_runtime_state.requested_kind = kind;
        global_compute_backend_runtime_state.fallback_active = false;
        global_compute_backend_config.backend = compute_backend_kind_name(kind);
        global_compute_backend_config.allow_fallback = false;
    }

    ~ScopedRecordingBackend()
    {
        global_compute_backend_runtime_state.reset();
        global_compute_backend_config = previous_config;
    }
};

void setup_grid()
{
    NR = 16;
    NTH = 16;
    NZ = 16;
    dr = 1000.0;
    dz = 500.0;
    dt = 0.5;
    dtheta = 2.0 * 3.14159265358979323846 / NTH;
}

bool same_tendency_bits(const DiffusionTendencies& lhs,
                        const DiffusionTendencies& rhs);

TEST_CASE("Full implicit A-grid scalar and momentum diffusion solves the collocated equation",
          "[numerics][diffusion][implicit][horizontal][agrid][collocated]")
{
    constexpr int nx = 7;
    constexpr int ny = 7;
    constexpr int nz_local = 5;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    Field3D rho_field(nx, ny, nz_local, poison);
    Field3D theta_field(nx, ny, nz_local, poison);
    Field3D u_field(nx, ny, nz_local, poison);
    Field3D v_field(nx, ny, nz_local, poison);
    Field3D w_field(nx, ny, nz_local, poison);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz_local - 1; ++k)
            {
                rho_field(i, j, k) = static_cast<float>(
                    0.9 + 0.02 * i + 0.03 * j + 0.04 * k);
                theta_field(i, j, k) =
                    (i == 3 && j == 3 && k == 2) ? 2.0f : 0.0f;
                u_field(i, j, k) = static_cast<float>(
                    1.0 + 0.2 * i * i - 0.1 * j + 0.3 * k);
                v_field(i, j, k) = static_cast<float>(
                    -0.5 + 0.15 * j * j - 0.08 * i - 0.2 * k);
                w_field(i, j, k) = static_cast<float>(
                    0.1 + 0.12 * k * k + 0.03 * i - 0.02 * j);
            }
    GridMetrics grid;
    grid.dx = 3.0;
    grid.dy = 4.0;
    grid.dz = {1.0, 1.3, 1.8, 2.4, 3.1};
    DiffusionConfig cfg;
    cfg.scheme_id = "implicit";
    cfg.K_h = 12.0;
    cfg.K_v = 5.0;
    cfg.dt_diffusion = 10.0;
    cfg.apply_to = "scalars_and_momentum";
    auto implicit_scheme = create_diffusion_scheme("implicit");
    implicit_scheme->initialize(cfg);
    DiffusionStateView state;
    state.theta = &theta_field;
    state.u = &u_field;
    state.v = &v_field;
    state.w = &w_field;
    state.rho = &rho_field;
    state.grid = &grid;
    state.c_grid = false;
    state.periodic_x = false;
    state.periodic_y = false;
    DiffusionTendencies one_thread;
    DiffusionTendencies four_threads;
#ifdef _OPENMP
    const int prior_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    REQUIRE_NOTHROW(implicit_scheme->compute_diffusion_tendencies(
        cfg, state, one_thread, nullptr));
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    REQUIRE_NOTHROW(implicit_scheme->compute_diffusion_tendencies(
        cfg, state, four_threads, nullptr));
#ifdef _OPENMP
    omp_set_num_threads(prior_threads);
#endif
    REQUIRE(same_tendency_bits(one_thread, four_threads));

    Field3D updated_theta = theta_field;
    Field3D updated_u = u_field;
    Field3D updated_v = v_field;
    Field3D updated_w = w_field;
    const Field3D* old_fields[4] = {
        &theta_field, &u_field, &v_field, &w_field};
    const Field3D* rates[4] = {
        &one_thread.dthetadt_diff, &one_thread.dudt_diff,
        &one_thread.dvdt_diff, &one_thread.dwdt_diff};
    Field3D* updated_fields[4] = {
        &updated_theta, &updated_u, &updated_v, &updated_w};
    double kinetic_before = 0.0;
    double kinetic_after = 0.0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz_local - 1; ++k)
            {
                for (int field = 0; field < 4; ++field)
                    (*updated_fields[field])(i, j, k) = static_cast<float>(
                        static_cast<double>((*old_fields[field])(i, j, k))
                        + cfg.dt_diffusion
                            * static_cast<double>((*rates[field])(i, j, k)));
                const double mass = rho_field(i, j, k)
                    * grid.dx * grid.dy
                    * grid.dz[static_cast<std::size_t>(k)];
                kinetic_before += 0.5 * mass * (
                    u_field(i, j, k) * u_field(i, j, k)
                    + v_field(i, j, k) * v_field(i, j, k)
                    + w_field(i, j, k) * w_field(i, j, k));
                kinetic_after += 0.5 * mass * (
                    updated_u(i, j, k) * updated_u(i, j, k)
                    + updated_v(i, j, k) * updated_v(i, j, k)
                    + updated_w(i, j, k) * updated_w(i, j, k));
                REQUIRE(updated_theta(i, j, k) >= -2.0e-5);
                REQUIRE(updated_theta(i, j, k) <= 2.0 + 2.0e-5);
            }
    REQUIRE(kinetic_after < kinetic_before);

    DiffusionConfig explicit_cfg = cfg;
    explicit_cfg.scheme_id = "explicit";
    auto explicit_scheme = create_diffusion_scheme("explicit");
    explicit_scheme->initialize(explicit_cfg);
    DiffusionStateView updated_state = state;
    updated_state.theta = &updated_theta;
    updated_state.u = &updated_u;
    updated_state.v = &updated_v;
    updated_state.w = &updated_w;
    DiffusionTendencies equation;
    explicit_scheme->compute_diffusion_tendencies(
        explicit_cfg, updated_state, equation, nullptr);
    const Field3D* equation_rates[4] = {
        &equation.dthetadt_diff, &equation.dudt_diff,
        &equation.dvdt_diff, &equation.dwdt_diff};
    double maximum_residual = 0.0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz_local - 1; ++k)
                for (int field = 0; field < 4; ++field)
                    maximum_residual = std::max(maximum_residual, std::abs(
                        static_cast<double>((*rates[field])(i, j, k))
                        - static_cast<double>((*equation_rates[field])(
                            i, j, k))));
    REQUIRE(maximum_residual < 2.0e-5);
    REQUIRE(implicit_scheme->check_stability(cfg, state) == 0.0);
}

bool same_field_bits(const Field3D& lhs, const Field3D& rhs)
{
    if (lhs.size_r() != rhs.size_r() || lhs.size_th() != rhs.size_th() ||
        lhs.size_z() != rhs.size_z() || lhs.size() != rhs.size())
    {
        return false;
    }
    return lhs.empty() ||
        std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0;
}

bool same_tendency_bits(const DiffusionTendencies& lhs,
                        const DiffusionTendencies& rhs)
{
    return same_field_bits(lhs.dudt_diff, rhs.dudt_diff) &&
        same_field_bits(lhs.dvdt_diff, rhs.dvdt_diff) &&
        same_field_bits(lhs.dwdt_diff, rhs.dwdt_diff) &&
        same_field_bits(lhs.dthetadt_diff, rhs.dthetadt_diff) &&
        same_field_bits(lhs.dqvdt_diff, rhs.dqvdt_diff);
}

DiffusionTendencies sentinel_tendencies()
{
    DiffusionTendencies tendencies;
    tendencies.dudt_diff = Field3D(2, 2, 2, 11.0f);
    tendencies.dvdt_diff = Field3D(2, 2, 2, 12.0f);
    tendencies.dwdt_diff = Field3D(2, 2, 2, 13.0f);
    tendencies.dthetadt_diff = Field3D(2, 2, 2, 14.0f);
    tendencies.dqvdt_diff = Field3D(2, 2, 2, 15.0f);
    return tendencies;
}

TerrainMetrics3D make_uniform_slope_metrics(
    int nr, int nth, int nz, double dx, double dy, double dzeta,
    double jacobian, double slope_x, double slope_y)
{
    TerrainMetrics3D metrics;
    metrics.dx = dx;
    metrics.dy = dy;
    metrics.dzeta = dzeta;
    metrics.z.resize(nr, nth, nz, 0.0);
    metrics.J.resize(nr, nth, nz, jacobian);
    metrics.mx.resize(nr, nth, nz, -slope_x / jacobian);
    metrics.my.resize(nr, nth, nz, -slope_y / jacobian);
    metrics.zeta.resize(nr, nth, nz, 0.0);
    metrics.z_face.resize(nr, nth, nz + 1, 0.0);
    metrics.Jx_face.resize(nr + 1, nth, nz, jacobian);
    metrics.Jy_face.resize(nr, nth + 1, nz, jacobian);
    metrics.Gx_face.resize(nr, nth, nz + 1, -slope_x);
    metrics.Gy_face.resize(nr, nth, nz + 1, -slope_y);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            const double ground = slope_x * i * dx + slope_y * j * dy;
            for (int k = 0; k <= nz; ++k)
                metrics.z_face(i, j, k) =
                    ground + jacobian * dzeta * k;
            for (int k = 0; k < nz; ++k)
            {
                metrics.z(i, j, k) = ground
                    + jacobian * dzeta * (static_cast<double>(k) + 0.5);
                metrics.zeta(i, j, k) =
                    dzeta * (static_cast<double>(k) + 0.5);
            }
        }
    return metrics;
}

} // namespace

TEST_CASE("shared C-grid dual masses partition stretched scalar cells exactly",
          "[numerics][diffusion][cgrid][mass][sci102]")
{
    SimulationStorageState storage{{3, 3, 3, 2.0, 10.0, 1.0, 3.0}};
    storage.grid.geometry.initialize(
        3, 3, 3, 2.0, 10.0, 3.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.grid.geometry.z_faces = {0.0, 10.0, 40.0, 100.0};
    storage.grid.geometry.z = {5.0, 25.0, 70.0};
    storage.fields.rho.resize(3, 3, 3, 1.0f);
    storage.fields.rho(1, 1, 0) = 2.0f;
    storage.fields.rho(1, 1, 1) = 5.0f;
    storage.fields.rho(2, 1, 0) = 7.0f;

    SimulationStateView state = make_simulation_state_view(storage);
    REQUIRE_NOTHROW(tmv::physical_grid_metrics::validate(state));
    const tmv::physical_grid_metrics::ControlVolumeMassView masses(state);
    const double area = storage.grid.dx * storage.grid.dy;
    const double lower_mass = area * 10.0 * 2.0;
    const double upper_mass = area * 30.0 * 5.0;
    const double right_mass = area * 10.0 * 7.0;

    REQUIRE(masses.scalar_mass_kg(storage.fields.rho, 1, 1, 0)
            == Approx(lower_mass));
    REQUIRE(masses.u_dual_mass_kg(storage.fields.rho, 1, 1, 0, 2)
            == Approx(0.5 * (lower_mass + right_mass)));
    REQUIRE(masses.w_dual_mass_kg(storage.fields.rho, 1, 1, 0)
            == Approx(0.5 * (lower_mass + upper_mass)));

    // Averaging density and thickness separately invents cross terms. Keep a
    // regression witness that is intentionally different for this profile.
    const double cross_term_mass =
        area * 0.25 * (10.0 + 30.0) * (2.0 + 5.0);
    REQUIRE(masses.w_dual_mass_kg(storage.fields.rho, 1, 1, 0)
            != Approx(cross_term_mass));
}

TEST_CASE("C-grid vertical w diffusion uses scalar-center stress and exact dual mass",
          "[numerics][diffusion][cgrid][vertical][mass][sci102]")
{
    constexpr int nx = 4;
    constexpr int ny = 4;
    constexpr int nz_local = 4;
    Field3D rho_field(nx, ny, nz_local, 1.0f);
    Field3D u_field(nx, ny, nz_local, 0.0f);
    Field3D v_field(nx, ny, nz_local, 0.0f);
    Field3D w_field(nx, ny, nz_local, 0.0f);
    const float density_by_level[nz_local] = {1.0f, 2.0f, 4.0f, 8.0f};
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_local; ++k)
                rho_field(i, j, k) = density_by_level[k];
    w_field(1, 1, 1) = 2.0f;
    w_field(1, 1, 2) = 2.0f;

    GridMetrics metrics;
    metrics.dx = 10.0;
    metrics.dy = 10.0;
    metrics.dz = {1.0, 3.0, 5.0, 7.0};
    DiffusionConfig cfg;
    cfg.apply_to = "momentum";
    cfg.K_h = 0.0;
    cfg.K_v = 1.0;
    cfg.dt_diffusion = 0.01;
    ExplicitDiffusionScheme scheme;
    scheme.initialize(cfg);
    DiffusionStateView state;
    state.u = &u_field;
    state.v = &v_field;
    state.w = &w_field;
    state.rho = &rho_field;
    state.grid = &metrics;
    state.c_grid = true;
    DiffusionTendencies tendencies;
    scheme.compute_diffusion_tendencies(cfg, state, tendencies, nullptr);

    const double stress = 2.0 * (2.0 - 0.0) / 3.0;
    const double lower_mass = 0.5 * (1.0 * 1.0 + 2.0 * 3.0);
    const double upper_mass = 0.5 * (2.0 * 3.0 + 4.0 * 5.0);
    REQUIRE(tendencies.dwdt_diff(1, 1, 0)
            == Approx(stress / lower_mass));
    REQUIRE(tendencies.dwdt_diff(1, 1, 1)
            == Approx(-stress / upper_mass));
    REQUIRE(tendencies.dwdt_diff(1, 1, 2) == 0.0f);
}

TEST_CASE("Explicit diffusion scheme creation", "[numerics][diffusion]")
{
    setup_grid();
    auto scheme = create_diffusion_scheme("explicit");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("explicit runtime dispatches the native FV device ABI for every target",
          "[numerics][diffusion][gpu][sci049]")
{
    constexpr int nx = 6;
    constexpr int ny = 6;
    constexpr int nz_local = 5;
    Field3D u(nx, ny, nz_local, 1.0f);
    Field3D v(nx, ny, nz_local, 2.0f);
    Field3D w(nx, ny, nz_local, 3.0f);
    Field3D theta(nx, ny, nz_local, 300.0f);
    Field3D qv(nx, ny, nz_local, 0.01f);
    Field3D rho_field(nx, ny, nz_local, 1.1f);
    GridMetrics metrics;
    metrics.dx = 100.0;
    metrics.dy = 120.0;
    metrics.dz = {40.0, 50.0, 65.0, 85.0, 110.0};

    DiffusionConfig config;
    config.K_h = 2.0;
    config.K_v = 3.0;
    config.dt_diffusion = 0.1;
    config.apply_to = "scalars_and_momentum";
    DiffusionStateView state;
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.theta = &theta;
    state.qv = &qv;
    state.rho = &rho_field;
    state.grid = &metrics;
    state.c_grid = true;
    state.periodic_x = true;
    state.periodic_y = false;

    const ComputeBackendKind accelerator_kinds[] = {
        ComputeBackendKind::Vulkan,
        ComputeBackendKind::Metal,
    };
    for (ComputeBackendKind kind : accelerator_kinds)
    {
        INFO("backend=" << compute_backend_kind_name(kind));
        auto backend = std::make_unique<RecordingDiffusionBackend>();
        auto* recorder = backend.get();
        ScopedRecordingBackend scoped(std::move(backend), kind);
        state.compute_runtime = &global_compute_backend_runtime_state;
        ExplicitDiffusionScheme scheme;
        scheme.initialize(config);
        DiffusionTendencies tendencies;
        REQUIRE_NOTHROW(scheme.compute_diffusion_tendencies(
            config, state, tendencies, nullptr));
        REQUIRE(recorder->valid_contract);
        REQUIRE(recorder->locations == std::vector<DiffusionFieldLocation>{
            DiffusionFieldLocation::UFace, DiffusionFieldLocation::VFace,
            DiffusionFieldLocation::WFace, DiffusionFieldLocation::Scalar,
            DiffusionFieldLocation::Scalar});
        REQUIRE(tendencies.dudt_diff(2, 2, 2) == 11.0f);
        REQUIRE(tendencies.dvdt_diff(2, 2, 2) == 12.0f);
        REQUIRE(tendencies.dwdt_diff(2, 2, 2) == 13.0f);
        REQUIRE(tendencies.dthetadt_diff(2, 2, 2) == 10.0f);
        REQUIRE(tendencies.dqvdt_diff(2, 2, 2) == 10.0f);
    }
}

TEST_CASE("incomplete terrain diffusion geometry fails closed",
          "[numerics][diffusion][terrain][guard]")
{
    constexpr int nx = 6;
    constexpr int ny = 6;
    constexpr int nz_local = 5;
    Field3D theta(nx, ny, nz_local, 300.0f);
    Field3D rho_field(nx, ny, nz_local, 1.0f);
    GridMetrics metrics;
    metrics.dx = 100.0;
    metrics.dy = 100.0;
    metrics.dz.assign(nz_local, 50.0);
    metrics.terrain_metrics_active = true;
    DiffusionConfig config;
    config.K_h = 10.0;
    config.K_v = 0.0;
    DiffusionStateView state;
    state.theta = &theta;
    state.rho = &rho_field;
    state.grid = &metrics;
    DiffusionTendencies tendencies;
    ExplicitDiffusionScheme scheme;
    scheme.initialize(config);
    REQUIRE_THROWS_AS(scheme.compute_diffusion_tendencies(
        config, state, tendencies, nullptr), std::invalid_argument);
    REQUIRE_THROWS_AS(scheme.check_stability(config, state),
                      std::invalid_argument);
}

TEST_CASE("terrain scalar diffusion is conservative and preserves constants",
          "[numerics][diffusion][terrain][conservative]")
{
    constexpr int nx = 7, ny = 7, nz_local = 6;
    const TerrainMetrics3D terrain = make_uniform_slope_metrics(
        nx, ny, nz_local, 120.0, 90.0, 50.0, 0.8, 0.15, -0.08);
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(nz_local, 40.0);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;
    Field3D scalar(nx, ny, nz_local, 3.5f);
    Field3D density(nx, ny, nz_local, 1.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_local; ++k)
                density(i, j, k) = static_cast<float>(
                    0.9 + 0.02 * i + 0.01 * j + 0.015 * k);

    DiffusionConfig config;
    config.K_h = 8.0;
    config.K_v = 3.0;
    config.dt_diffusion = 0.05;
    DiffusionStateView state;
    state.theta = &scalar;
    state.rho = &density;
    state.grid = &grid;
    state.c_grid = true;
    DiffusionTendencies tendencies;
    DiffusionDiagnostics diagnostics;
    ExplicitDiffusionScheme scheme;
    scheme.initialize(config);
    scheme.compute_diffusion_tendencies(config, state, tendencies, &diagnostics);
    for (std::size_t n = 0; n < tendencies.dthetadt_diff.size(); ++n)
        REQUIRE(tendencies.dthetadt_diff.data()[n] == 0.0f);
    REQUIRE(std::isfinite(diagnostics.max_diffusion_number));
    REQUIRE(diagnostics.max_diffusion_number >= 0.0);

    const float poison = std::numeric_limits<float>::quiet_NaN();
    for (int k = 0; k < nz_local; ++k)
    {
        for (int j = 0; j < ny; ++j)
        {
            scalar(0, j, k) = poison;
            scalar(nx - 1, j, k) = poison;
            density(0, j, k) = poison;
            density(nx - 1, j, k) = poison;
        }
        for (int i = 0; i < nx; ++i)
        {
            scalar(i, 0, k) = poison;
            scalar(i, ny - 1, k) = poison;
            density(i, 0, k) = poison;
            density(i, ny - 1, k) = poison;
        }
    }
    REQUIRE_NOTHROW(scheme.compute_diffusion_tendencies(
        config, state, tendencies, &diagnostics));
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_local; ++k)
                REQUIRE(tendencies.dthetadt_diff(i, j, k) == 0.0f);
    REQUIRE(std::isfinite(diagnostics.max_diffusion_number));

    scalar.fill(3.5f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_local; ++k)
                density(i, j, k) = static_cast<float>(
                    0.9 + 0.02 * i + 0.01 * j + 0.015 * k);

    scalar(3, 3, 2) = 7.0f;
    scheme.compute_diffusion_tendencies(config, state, tendencies, &diagnostics);
    REQUIRE(tendencies.dthetadt_diff(3, 3, 2) < 0.0f);
    double inventory_rate = 0.0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
                inventory_rate += static_cast<double>(density(i, j, k))
                    * static_cast<double>(tendencies.dthetadt_diff(i, j, k))
                    * terrain.J(i, j, k) * terrain.dx * terrain.dy
                    * terrain.dzeta;
    REQUIRE(inventory_rate == Approx(0.0).margin(5.0e-3));
}

TEST_CASE("explicit terrain momentum diffusion conserves mapped momentum and dissipates energy",
          "[numerics][diffusion][terrain][momentum][conservation][work]")
{
    constexpr int nx = 12, ny = 12, nz_local = 12;
    TerrainMetrics3D terrain = make_uniform_slope_metrics(
        nx, ny, nz_local, 80.0, 110.0, 45.0, 0.9, 0.12, -0.07);
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(nz_local, terrain.dzeta * 0.9);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;
    grid.staggering = StaggerType::CGrid;

    Field3D rho(nx, ny, nz_local, 1.0f);
    Field3D u(nx, ny, nz_local, 0.0f);
    Field3D v(nx, ny, nz_local, 0.0f);
    Field3D w(nx, ny, nz_local, 0.0f);
    const auto bump = [](int i, int j, int k, float amplitude)
    {
        const auto shape = [](int index)
        {
            if (index < 4 || index > 7) return 0.0;
            return std::sin(3.14159265358979323846 *
                            static_cast<double>(index - 3) / 5.0);
        };
        return amplitude * static_cast<float>(
            shape(i) * shape(j) * shape(k));
    };
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                rho(i, j, k) = static_cast<float>(
                    0.75 + 0.013 * (2 * i + j + k));
                u(i, j, k) = bump(i, j, k, 5.0f);
                v(i, j, k) = bump(j, k, i, -3.0f);
                w(i, j, k) = bump(k, i, j, 2.0f);
            }

    DiffusionConfig cfg;
    cfg.scheme_id = "explicit";
    cfg.K_h = 2.0;
    cfg.K_v = 2.0;
    cfg.dt_diffusion = 0.01;
    cfg.apply_to = "momentum";
    ExplicitDiffusionScheme scheme;
    scheme.initialize(cfg);

    DiffusionStateView state;
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.rho = &rho;
    state.grid = &grid;
    state.c_grid = true;
    DiffusionTendencies tendencies;
    DiffusionDiagnostics diagnostics;
    REQUIRE_NOTHROW(scheme.compute_diffusion_tendencies(
        cfg, state, tendencies, &diagnostics));

    double net[3] = {0.0, 0.0, 0.0};
    double activity[3] = {0.0, 0.0, 0.0};
    double resolved_work = 0.0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                const int ip = std::min(i + 1, nx - 1);
                const int jp = std::min(j + 1, ny - 1);
                const int kp = std::min(k + 1, nz_local - 1);
                const double masses[3] = {
                    0.5 * (static_cast<double>(rho(i, j, k)) *
                               terrain.J(i, j, k) +
                           static_cast<double>(rho(ip, j, k)) *
                               terrain.J(ip, j, k)),
                    0.5 * (static_cast<double>(rho(i, j, k)) *
                               terrain.J(i, j, k) +
                           static_cast<double>(rho(i, jp, k)) *
                               terrain.J(i, jp, k)),
                    0.5 * (static_cast<double>(rho(i, j, k)) *
                               terrain.J(i, j, k) +
                           static_cast<double>(rho(i, j, kp)) *
                               terrain.J(i, j, kp))};
                const double rates[3] = {
                    tendencies.dudt_diff(i, j, k),
                    tendencies.dvdt_diff(i, j, k),
                    tendencies.dwdt_diff(i, j, k)};
                const double values[3] = {
                    u(i, j, k), v(i, j, k), w(i, j, k)};
                for (int component = 0; component < 3; ++component)
                {
                    const double force = masses[component] * rates[component];
                    net[component] += force;
                    activity[component] += std::abs(force);
                    resolved_work += values[component] * force;
                }
            }
    for (int component = 0; component < 3; ++component)
    {
        REQUIRE(activity[component] > 1.0e-7);
        REQUIRE(std::abs(net[component]) <=
                4.0e-6 * activity[component]);
    }
    REQUIRE(resolved_work < 0.0);
    REQUIRE(std::isfinite(diagnostics.max_diffusion_number));
    REQUIRE(diagnostics.max_diffusion_number > 0.0);
    REQUIRE(diagnostics.max_diffusion_number < 1.0);
    for (std::size_t index = 0;
         index < tendencies.dthetadt_diff.size(); ++index)
    {
        REQUIRE(tendencies.dthetadt_diff.data()[index] == 0.0f);
        REQUIRE(tendencies.dqvdt_diff.data()[index] == 0.0f);
    }

    const DiffusionTendencies before = sentinel_tendencies();
    DiffusionTendencies sentinel = before;
    terrain.z_face(6, 6, 4) = terrain.z_face(6, 6, 3);
    REQUIRE_THROWS_AS(scheme.compute_diffusion_tendencies(
        cfg, state, sentinel, nullptr), std::invalid_argument);
    REQUIRE(same_tendency_bits(sentinel, before));
}

TEST_CASE("terrain scalar diffusion has exact flat reduction and metric cancellation",
          "[numerics][diffusion][terrain][flat-reduction][metric]")
{
    constexpr int nx = 7, ny = 7, nz_local = 6;
    const TerrainMetrics3D flat_terrain = make_uniform_slope_metrics(
        nx, ny, nz_local, 120.0, 90.0, 50.0, 1.0, 0.0, 0.0);
    GridMetrics mapped_grid;
    mapped_grid.dx = 120.0;
    mapped_grid.dy = 90.0;
    mapped_grid.dz.assign(nz_local, 50.0);
    mapped_grid.terrain_metrics_active = true;
    mapped_grid.terrain_metrics = &flat_terrain;
    GridMetrics flat_grid = mapped_grid;
    flat_grid.terrain_metrics_active = false;
    flat_grid.terrain_metrics = nullptr;

    Field3D scalar(nx, ny, nz_local, 0.0f);
    Field3D density(nx, ny, nz_local, 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                scalar(i, j, k) = static_cast<float>(
                    0.2 * i * i - 0.1 * j + 0.3 * k * k);
                density(i, j, k) = static_cast<float>(
                    0.9 + 0.01 * i + 0.015 * j + 0.02 * k);
            }
    DiffusionConfig config;
    config.K_h = 4.0;
    config.K_v = 7.0;
    config.dt_diffusion = 0.02;
    DiffusionStateView mapped_state;
    mapped_state.theta = &scalar;
    mapped_state.rho = &density;
    mapped_state.grid = &mapped_grid;
    mapped_state.c_grid = true;
    DiffusionStateView flat_state = mapped_state;
    flat_state.grid = &flat_grid;
    ExplicitDiffusionScheme scheme;
    scheme.initialize(config);
    DiffusionTendencies mapped;
    DiffusionTendencies cartesian;
    scheme.compute_diffusion_tendencies(
        config, mapped_state, mapped, nullptr);
    scheme.compute_diffusion_tendencies(
        config, flat_state, cartesian, nullptr);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_local; ++k)
                REQUIRE(mapped.dthetadt_diff(i, j, k)
                        == Approx(cartesian.dthetadt_diff(i, j, k))
                               .margin(2.0e-9));

    const TerrainMetrics3D slope = make_uniform_slope_metrics(
        nx, ny, nz_local, 120.0, 90.0, 50.0, 0.8, 0.15, -0.08);
    mapped_grid.terrain_metrics = &slope;
    density.fill(1.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_local; ++k)
                scalar(i, j, k) = static_cast<float>(slope.z(i, j, k));
    scheme.compute_diffusion_tendencies(
        config, mapped_state, mapped, nullptr);
    for (int i = 2; i < nx - 2; ++i)
        for (int j = 2; j < ny - 2; ++j)
            for (int k = 1; k < nz_local - 1; ++k)
                REQUIRE(mapped.dthetadt_diff(i, j, k)
                        == Approx(0.0).margin(2.0e-7));
}

TEST_CASE("terrain scalar diffusion owns periodic seams without closure storage",
          "[numerics][diffusion][terrain][periodic][conservation]")
{
    constexpr int nx = 8, ny = 7, nz_local = 5;
    // x is periodic; retaining a physical y slope exercises the mapped zeta
    // cross-flux while keeping the periodic terrain itself single-valued.
    const TerrainMetrics3D terrain = make_uniform_slope_metrics(
        nx, ny, nz_local, 80.0, 110.0, 45.0, 0.9, 0.0, 0.07);
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(nz_local, terrain.dzeta * 0.9);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;

    const float poison = std::numeric_limits<float>::quiet_NaN();
    Field3D scalar(nx, ny, nz_local, poison);
    Field3D density(nx, ny, nz_local, poison);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                scalar(i, j, k) = static_cast<float>(
                    290.0 + 0.35 * i * i - 0.22 * j + 0.16 * i * j
                    + 0.4 * k * k);
                density(i, j, k) = static_cast<float>(
                    0.85 + 0.03 * i + 0.02 * j + 0.025 * k);
            }

    DiffusionConfig config;
    config.K_h = 6.0;
    config.K_v = 2.5;
    config.dt_diffusion = 0.04;
    DiffusionStateView state;
    state.theta = &scalar;
    state.rho = &density;
    state.grid = &grid;
    state.c_grid = true;
    state.periodic_x = true;
    state.periodic_y = false;

    ExplicitDiffusionScheme scheme;
    scheme.initialize(config);
    DiffusionTendencies tendencies;
    DiffusionDiagnostics diagnostics;
    REQUIRE_NOTHROW(scheme.compute_diffusion_tendencies(
        config, state, tendencies, &diagnostics));

    double inventory_rate = 0.0;
    double absolute_rate = 0.0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                REQUIRE(std::isfinite(tendencies.dthetadt_diff(i, j, k)));
                const double weighted = static_cast<double>(density(i, j, k))
                    * terrain.J(i, j, k) * terrain.dx * terrain.dy
                    * terrain.dzeta
                    * static_cast<double>(tendencies.dthetadt_diff(i, j, k));
                inventory_rate += weighted;
                absolute_rate += std::abs(weighted);
            }
    REQUIRE(absolute_rate > 0.0);
    REQUIRE(std::abs(inventory_rate) <= 2.0e-6 * absolute_rate);
    REQUIRE(std::isfinite(diagnostics.max_diffusion_number));
    REQUIRE(diagnostics.max_diffusion_number > 0.0);

    scalar.fill(301.5f);
    // Re-poison the closure planes after fill: a periodic implementation must
    // wrap physical cells 1..N-2 and never make ghost validity a prerequisite.
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
                for (int k = 0; k < nz_local; ++k)
                    scalar(i, j, k) = poison;
    REQUIRE_NOTHROW(scheme.compute_diffusion_tendencies(
        config, state, tendencies, nullptr));
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
                REQUIRE(tendencies.dthetadt_diff(i, j, k) == 0.0f);
}

TEST_CASE("Implicit diffusion scheme creation", "[numerics][diffusion]")
{
    setup_grid();
    auto scheme = create_diffusion_scheme("implicit");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("diffusion runtime owns dispatch and reusable storage",
          "[numerics][diffusion][state][modularity]")
{
    REQUIRE(&diffusion_scheme == &global_diffusion_runtime_state.scheme);
    REQUIRE(&global_diffusion_config == &global_diffusion_runtime_state.config);

    DiffusionRuntimeState original;
    original.config.scheme_id = "explicit";
    original.scheme = create_diffusion_scheme("explicit");
    original.scheme->initialize(original.config);
    original.tendencies.dthetadt_diff.resize(2, 1, 1, 0.5f);
    original.diagnostics.K_effective.resize(2, 1, 1, 2.0f);
    original.last_momentum_work = {-4.0, 0.2, -0.01, -3.81};
    original.cumulative_committed_momentum_work_j = -38.1;
    original.momentum_work_step_count = 10;

    DiffusionRuntimeState copy(original);
    REQUIRE(copy.scheme != nullptr);
    REQUIRE(copy.scheme.get() != original.scheme.get());
    REQUIRE(copy.scheme->name() == "explicit");
    REQUIRE(copy.tendencies.dthetadt_diff.data() !=
            original.tendencies.dthetadt_diff.data());
    REQUIRE(copy.diagnostics.K_effective.data() !=
            original.diagnostics.K_effective.data());
    REQUIRE(copy.last_momentum_work.committed_work_j == Approx(-3.81));
    REQUIRE(copy.cumulative_committed_momentum_work_j == Approx(-38.1));
    REQUIRE(copy.momentum_work_step_count == 10);
}

TEST_CASE("diffusion advances only caller-owned fields and runtime",
          "[numerics][diffusion][state][modularity]")
{
    constexpr int local_nr = 6;
    constexpr int local_nth = 6;
    constexpr int local_nz = 5;
    int nr_local = local_nr;
    int nth_local = local_nth;
    int nz_local = local_nz;
    double dx_local = 100.0;
    double dz_local = 100.0;
    double dt_local = 1.0;
    double dy_local = 100.0;
    GridGeometry geometry_local;
    geometry_local.initialize(nr_local, nth_local, nz_local,
                              dx_local, dz_local, dy_local,
                              CoordinateSystem::Cartesian);

    Field3D rho_local(local_nr, local_nth, local_nz, 1.0f);
    Field3D p_local(local_nr, local_nth, local_nz, 90000.0f);
    Field3D u_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D w_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D v_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D tracer_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D theta_local(local_nr, local_nth, local_nz, 300.0f);
    Field3D qv_local(local_nr, local_nth, local_nz, 0.01f);
    Field3D qc_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D qr_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D qi_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D qs_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D qh_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D qg_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D tke_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D dtheta_rad_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D dtheta_pbl_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D dqv_pbl_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D du_pbl_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D dv_pbl_local(local_nr, local_nth, local_nz, 0.0f);
    Field3D dtke_pbl_local(local_nr, local_nth, local_nz, 0.0f);

    std::vector<double> rho0_local(local_nz, 1.0);
    std::vector<double> p0_local(local_nz, 90000.0);
    std::vector<double> qv0_local(local_nz, 0.01);
    std::vector<double> theta0_local(local_nz, 300.0);
    std::vector<double> u0_local(local_nz, 0.0);
    std::vector<double> v0_local(local_nz, 0.0);
    double coriolis_local = 0.0;
    double time_local = 0.0;

    DiffusionRuntimeState runtime_local;
    runtime_local.config.scheme_id = "explicit";
    runtime_local.config.apply_to = "scalars_and_momentum";
    runtime_local.config.K_h = 10.0;
    runtime_local.config.K_v = 50.0;
    runtime_local.scheme = create_diffusion_scheme("explicit");
    runtime_local.scheme->initialize(runtime_local.config);

    DiffusionRuntimeState other_runtime;
    other_runtime.tendencies.dthetadt_diff.resize(1, 1, 1, 7.0f);

    SimulationStateView state_local{
        {nr_local, nth_local, nz_local, dx_local, dz_local,
         dt_local, dy_local, geometry_local},
        {rho_local, p_local, u_local, w_local, v_local, tracer_local,
         theta_local, qv_local, qc_local, qr_local, qi_local, qs_local,
         qh_local, qg_local, tke_local},
        {rho0_local, p0_local, qv0_local, theta0_local, u0_local, v0_local},
        {dtheta_rad_local, dtheta_pbl_local, dqv_pbl_local,
         du_pbl_local, dv_pbl_local, dtke_pbl_local},
        coriolis_local,
        time_local,
        {}, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        &runtime_local};

    RuntimeConfigurationState local_configuration;
    tmv::domain::HorizontalDomainFrame local_frame;
    state_local.configuration = &local_configuration;
    state_local.domain_frame = &local_frame;
    theta_local[2][2][2] = 302.0f;
    u_local[2][2][2] = 2.0f;
    REQUIRE(runtime_diffusion_enabled(state_local));
    theta.resize(1, 1, 1, 275.0f);
    const float* const original_u_storage = u_local.data();
    const float* const original_theta_storage = theta_local.data();

    apply_runtime_diffusion(state_local, 1.0);
    const float* const first_committed_u_storage = u_local.data();
    const float* const first_committed_theta_storage = theta_local.data();

    REQUIRE(theta_local[2][2][2] < 302.0f);
    REQUIRE(theta_local[2][2][1] > 300.0f);
    REQUIRE(theta_local[2][1][2] > 300.0f);
    REQUIRE(u_local[2][2][2] < 2.0f);
    REQUIRE(u_local[2][1][2] > 0.0f);
    REQUIRE(runtime_local.tendencies.dthetadt_diff.size_z() == local_nz);
    REQUIRE(runtime_local.diagnostics.K_effective[2][2][2] == Approx(50.0f));
    REQUIRE(runtime_local.last_momentum_work.committed_work_j < 0.0);
    REQUIRE(runtime_local.last_momentum_work.committed_work_j == Approx(
        runtime_local.last_momentum_work.requested_linear_work_j +
        runtime_local.last_momentum_work.finite_step_correction_j +
        runtime_local.last_momentum_work.storage_rounding_j));
    REQUIRE(runtime_local.last_momentum_work.moist_energy_complete);
    REQUIRE(runtime_local.last_momentum_work
                .moist_energy_closure_residual_j == 0.0);
    REQUIRE(runtime_local.cumulative_moist_energy_work
                .moist_energy_complete);
    REQUIRE(runtime_local.moist_energy_step_count == 1);
    REQUIRE(runtime_local.momentum_work_step_count == 1);
    REQUIRE(theta[0][0][0] == Approx(275.0f));
    REQUIRE(other_runtime.tendencies.dthetadt_diff.size_z() == 1);
    REQUIRE(other_runtime.tendencies.dthetadt_diff[0][0][0] == Approx(7.0f));
    REQUIRE(runtime_local.scratch_u.data() == original_u_storage);
    REQUIRE(runtime_local.scratch_theta.data() == original_theta_storage);

    // The alternate Cartesian path must traverse the same runtime dispatch
    // and commit boundary, while remaining vertical-only by contract.
    theta_local.fill(300.0f);
    qv_local.fill(0.01f);
    u_local.fill(0.0f);
    v_local.fill(0.0f);
    w_local.fill(0.0f);
    theta_local[2][2][2] = 302.0f;
    u_local[2][2][2] = 2.0f;
    runtime_local.config.scheme_id = "implicit";
    runtime_local.config.K_h = 0.0;
    runtime_local.config.K_v = 50.0;
    runtime_local.scheme = create_diffusion_scheme("implicit");
    runtime_local.scheme->initialize(runtime_local.config);

    REQUIRE(runtime_diffusion_enabled(state_local));
    apply_runtime_diffusion(state_local, 1.0);

    REQUIRE(theta_local[2][2][2] < 302.0f);
    REQUIRE(theta_local[2][2][1] > 300.0f);
    REQUIRE(theta_local[2][1][2] == Approx(300.0f));
    REQUIRE(u_local[2][2][2] < 2.0f);
    REQUIRE(u_local[2][2][1] > 0.0f);
    REQUIRE(u_local[2][1][2] == Approx(0.0f));
    REQUIRE(runtime_local.diagnostics.max_diffusion_number == 0.0);
    REQUIRE(runtime_local.last_momentum_work.committed_work_j < 0.0);
    REQUIRE(runtime_local.last_momentum_work.committed_work_j == Approx(
        runtime_local.last_momentum_work.requested_linear_work_j +
        runtime_local.last_momentum_work.finite_step_correction_j +
        runtime_local.last_momentum_work.storage_rounding_j));
    REQUIRE(runtime_local.momentum_work_step_count == 2);
    REQUIRE(runtime_local.last_momentum_work.moist_energy_complete);
    REQUIRE(runtime_local.last_momentum_work
                .moist_energy_closure_residual_j == 0.0);
    REQUIRE(runtime_local.moist_energy_step_count == 2);
    REQUIRE(u_local.data() == original_u_storage);
    REQUIRE(theta_local.data() == original_theta_storage);
    REQUIRE(runtime_local.scratch_u.data() == first_committed_u_storage);
    REQUIRE(runtime_local.scratch_theta.data()
            == first_committed_theta_storage);

    runtime_local.config.K_v = 0.0;
    REQUIRE_FALSE(runtime_diffusion_enabled(state_local));
}

TEST_CASE("caller-owned terrain reaches conservative scalar diffusion",
          "[numerics][diffusion][terrain][state]")
{
    constexpr int nr = 6, nth = 6, nz = 5;
    SimulationStorageState storage{{nr, nth, nz, 100.0, 80.0, 0.1, 120.0}};
    storage.grid.geometry.initialize(
        nr, nth, nz, 100.0, 80.0, 120.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.fields.rho.resize(nr, nth, nz, 1.0f);
    storage.fields.theta.resize(nr, nth, nz, 300.0f);
    storage.fields.qv.resize(nr, nth, nz, 0.01f);

    auto& terrain = storage.runtime.terrain;
    terrain.config.scheme_id = "bell";
    terrain.metrics = make_uniform_slope_metrics(
        nr, nth, nz, 100.0, 120.0, 80.0, 0.75, 0.1, -0.04);

    auto& runtime = storage.runtime.diffusion;
    runtime.config.scheme_id = "explicit";
    runtime.config.apply_to = "scalars";
    runtime.config.K_h = 2.0;
    runtime.config.K_v = 3.0;
    runtime.scheme = create_diffusion_scheme("explicit");
    runtime.scheme->initialize(runtime.config);
    SimulationStateView state = make_simulation_state_view(storage);

    REQUIRE_NOTHROW(apply_runtime_diffusion(state, 0.1));
    REQUIRE(storage.fields.theta(2, 2, 2) == Approx(300.0f));
    REQUIRE(runtime.momentum_work_step_count == 0);
}

TEST_CASE("Diffusion descriptors expose backend and field contracts",
          "[numerics][diffusion][descriptor]")
{
    const auto& descriptors = get_diffusion_scheme_descriptors();
    REQUIRE(descriptors.size() == 2);
    for (const auto& descriptor : descriptors)
    {
        REQUIRE(descriptor.capabilities.declared);
        REQUIRE_FALSE(descriptor.capabilities.required_state_fields.empty());
        REQUIRE_FALSE(descriptor.capabilities.produced_state_fields.empty());
        REQUIRE_FALSE(descriptor.capabilities.config_validation_contract.empty());
    }

    const auto* explicit_scheme = tmv::find_scheme_descriptor(descriptors, "explicit");
    REQUIRE(explicit_scheme != nullptr);
    REQUIRE(explicit_scheme->capabilities.gpu_implementation_coverage
            == tmv::SchemeCoverage::Covered);
    REQUIRE(explicit_scheme->capabilities.cpu_gpu_parity_coverage
            == tmv::SchemeCoverage::Covered);
    const auto* implicit_scheme = tmv::find_scheme_descriptor(descriptors, "implicit");
    REQUIRE(implicit_scheme != nullptr);
    REQUIRE(implicit_scheme->capabilities.gpu_implementation_coverage
            == tmv::SchemeCoverage::Covered);
}

TEST_CASE("Unavailable no-op diffusion scheme fails closed",
          "[numerics][diffusion][config][sci021]")
{
    setup_grid();
    REQUIRE_THROWS(create_diffusion_scheme("none"));
}

TEST_CASE("Explicit diffusion initializes with config", "[numerics][diffusion]")
{
    setup_grid();
    auto scheme = create_diffusion_scheme("explicit");
    DiffusionConfig cfg;
    cfg.scheme_id = "explicit";
    cfg.K_h = 0.0;
    cfg.K_v = 50.0;
    REQUIRE_NOTHROW(scheme->initialize(cfg));
}

TEST_CASE("Implicit diffusion accepts horizontal scalar and momentum targets",
          "[numerics][diffusion][horizontal][config-truth]")
{
    DiffusionConfig cfg;
    cfg.K_h = 10.0;
    cfg.K_v = 0.0;
    auto explicit_scheme = create_diffusion_scheme("explicit");
    REQUIRE_NOTHROW(explicit_scheme->initialize(cfg));

    cfg.apply_to = "scalars_and_momentum";
    REQUIRE_NOTHROW(explicit_scheme->initialize(cfg));
    cfg.apply_to = "scalars";

    cfg.scheme_id = "implicit";
    auto implicit_scheme = create_diffusion_scheme("implicit");
    REQUIRE_NOTHROW(implicit_scheme->initialize(cfg));
    cfg.apply_to = "scalars_and_momentum";
    REQUIRE_NOTHROW(implicit_scheme->initialize(cfg));
}

TEST_CASE("Explicit horizontal C-grid momentum diffusion conserves momentum and dissipates energy",
          "[numerics][diffusion][horizontal][momentum][cgrid][conservation]")
{
    constexpr int nx = 7;
    constexpr int ny = 8;
    constexpr int nz_local = 5;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    Field3D rho_field(nx, ny, nz_local, poison);
    Field3D u_field(nx, ny, nz_local, poison);
    Field3D v_field(nx, ny, nz_local, poison);
    Field3D w_field(nx, ny, nz_local, poison);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                rho_field[i][j][k] = static_cast<float>(
                    0.8 + 0.07 * i + 0.04 * j + 0.03 * k);
                if (k < nz_local - 1)
                {
                    u_field[i][j][k] = static_cast<float>(
                        2.0 + 0.3 * i * i - 0.2 * j + 0.1 * i * j + 0.05 * k);
                    v_field[i][j][k] = static_cast<float>(
                        -1.0 + 0.15 * i - 0.22 * j * j + 0.08 * i * j - 0.03 * k);
                    w_field[i][j][k] = static_cast<float>(
                        0.4 - 0.11 * i * i + 0.17 * j + 0.06 * i * j + 0.02 * k);
                }
            }

    GridMetrics metrics;
    metrics.dx = 2.5;
    metrics.dy = 3.5;
    metrics.dz.assign(nz_local, 4.0);
    DiffusionConfig cfg;
    cfg.K_h = 3.0;
    cfg.K_v = 0.0;
    cfg.dt_diffusion = 0.01;
    cfg.apply_to = "scalars_and_momentum";
    auto scheme = create_diffusion_scheme("explicit");
    scheme->initialize(cfg);

    DiffusionStateView state;
    state.u = &u_field;
    state.v = &v_field;
    state.w = &w_field;
    state.rho = &rho_field;
    state.grid = &metrics;
    state.c_grid = true;
    state.periodic_x = true;
    state.periodic_y = true;
    DiffusionTendencies tendencies;
    DiffusionDiagnostics diagnostics;
    scheme->compute_diffusion_tendencies(cfg, state, tendencies, &diagnostics);

    double u_momentum = 0.0;
    double v_momentum = 0.0;
    double w_momentum = 0.0;
    double kinetic_work = 0.0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local - 1; ++k)
            {
                const int u_right = i == nx - 2 ? 1 : i + 1;
                const int v_north = j == ny - 2 ? 1 : j + 1;
                const double rho_u = 0.5 * (
                    static_cast<double>(rho_field[i][j][k]) +
                    static_cast<double>(rho_field[u_right][j][k]));
                const double rho_v = 0.5 * (
                    static_cast<double>(rho_field[i][j][k]) +
                    static_cast<double>(rho_field[i][v_north][k]));
                const double w_mass_per_area = 0.5 * (
                    static_cast<double>(rho_field[i][j][k]) +
                    static_cast<double>(rho_field[i][j][k + 1]));
                const double du = tendencies.dudt_diff[i][j][k];
                const double dv = tendencies.dvdt_diff[i][j][k];
                const double dw = tendencies.dwdt_diff[i][j][k];
                u_momentum += rho_u * du;
                v_momentum += rho_v * dv;
                w_momentum += w_mass_per_area * dw;
                kinetic_work += rho_u * static_cast<double>(u_field[i][j][k]) * du
                              + rho_v * static_cast<double>(v_field[i][j][k]) * dv
                              + w_mass_per_area *
                                  static_cast<double>(w_field[i][j][k]) * dw;
            }
    REQUIRE(u_momentum == Approx(0.0).margin(3.0e-5));
    REQUIRE(v_momentum == Approx(0.0).margin(3.0e-5));
    REQUIRE(w_momentum == Approx(0.0).margin(3.0e-5));
    REQUIRE(kinetic_work < 0.0);
    REQUIRE(diagnostics.max_diffusion_number > 0.0);
    REQUIRE(diagnostics.max_diffusion_number < 1.0);
    REQUIRE(tendencies.dudt_diff[0][3][1] == 0.0f);
    REQUIRE(tendencies.dwdt_diff[3][3][nz_local - 1] == 0.0f);

    state.periodic_x = false;
    state.periodic_y = false;
    scheme->compute_diffusion_tendencies(cfg, state, tendencies, nullptr);
    double closed_momentum[3] = {0.0, 0.0, 0.0};
    double closed_work = 0.0;
    for (int i = 1; i < nx - 2; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local - 1; ++k)
            {
                const double density = 0.5 * (
                    static_cast<double>(rho_field[i][j][k]) +
                    static_cast<double>(rho_field[i + 1][j][k]));
                const double du = tendencies.dudt_diff[i][j][k];
                closed_momentum[0] += density * du;
                closed_work += density * static_cast<double>(u_field[i][j][k]) * du;
            }
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 2; ++j)
            for (int k = 0; k < nz_local - 1; ++k)
            {
                const double density = 0.5 * (
                    static_cast<double>(rho_field[i][j][k]) +
                    static_cast<double>(rho_field[i][j + 1][k]));
                const double dv = tendencies.dvdt_diff[i][j][k];
                closed_momentum[1] += density * dv;
                closed_work += density * static_cast<double>(v_field[i][j][k]) * dv;
            }
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local - 1; ++k)
            {
                const double density = 0.5 * (
                    static_cast<double>(rho_field[i][j][k]) +
                    static_cast<double>(rho_field[i][j][k + 1]));
                const double dw = tendencies.dwdt_diff[i][j][k];
                closed_momentum[2] += density * dw;
                closed_work += density * static_cast<double>(w_field[i][j][k]) * dw;
            }
    REQUIRE(closed_momentum[0] == Approx(0.0).margin(3.0e-5));
    REQUIRE(closed_momentum[1] == Approx(0.0).margin(3.0e-5));
    REQUIRE(closed_momentum[2] == Approx(0.0).margin(3.0e-5));
    REQUIRE(closed_work < 0.0);
    REQUIRE(tendencies.dudt_diff[nx - 2][3][1] == 0.0f);
    REQUIRE(tendencies.dvdt_diff[3][ny - 2][1] == 0.0f);
}

TEST_CASE("Explicit horizontal A-grid momentum diffusion ignores closures and closes budgets",
          "[numerics][diffusion][horizontal][momentum][agrid][conservation]")
{
    constexpr int nx = 7;
    constexpr int ny = 7;
    constexpr int nz_local = 6;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    Field3D rho_field(nx, ny, nz_local, poison);
    Field3D u_field(nx, ny, nz_local, poison);
    Field3D v_field(nx, ny, nz_local, poison);
    Field3D w_field(nx, ny, nz_local, poison);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz_local - 1; ++k)
            {
                rho_field[i][j][k] = static_cast<float>(
                    0.9 + 0.08 * i + 0.05 * j + 0.02 * k);
                u_field[i][j][k] = static_cast<float>(
                    3.0 + 0.21 * i * i - 0.18 * j + 0.07 * i * j + 0.03 * k);
                v_field[i][j][k] = static_cast<float>(
                    -2.0 + 0.13 * i - 0.19 * j * j + 0.05 * i * j - 0.02 * k);
                w_field[i][j][k] = static_cast<float>(
                    0.2 - 0.09 * i * i + 0.16 * j + 0.04 * i * j + 0.01 * k);
            }

    GridMetrics metrics;
    metrics.dx = 3.0;
    metrics.dy = 4.0;
    metrics.dz.assign(nz_local, 5.0);
    DiffusionConfig cfg;
    cfg.K_h = 2.0;
    cfg.K_v = 0.0;
    cfg.dt_diffusion = 0.02;
    cfg.apply_to = "scalars_and_momentum";
    auto scheme = create_diffusion_scheme("explicit");
    scheme->initialize(cfg);

    DiffusionStateView state;
    state.u = &u_field;
    state.v = &v_field;
    state.w = &w_field;
    state.rho = &rho_field;
    state.grid = &metrics;
    state.c_grid = false;
    state.periodic_x = true;
    state.periodic_y = true;
    DiffusionTendencies tendencies;
    scheme->compute_diffusion_tendencies(cfg, state, tendencies, nullptr);

    double momentum[3] = {0.0, 0.0, 0.0};
    double kinetic_work = 0.0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz_local - 1; ++k)
            {
                const double density = rho_field[i][j][k];
                const double du = tendencies.dudt_diff[i][j][k];
                const double dv = tendencies.dvdt_diff[i][j][k];
                const double dw = tendencies.dwdt_diff[i][j][k];
                momentum[0] += density * du;
                momentum[1] += density * dv;
                momentum[2] += density * dw;
                kinetic_work += density * (
                    static_cast<double>(u_field[i][j][k]) * du +
                    static_cast<double>(v_field[i][j][k]) * dv +
                    static_cast<double>(w_field[i][j][k]) * dw);
            }
    REQUIRE(momentum[0] == Approx(0.0).margin(3.0e-5));
    REQUIRE(momentum[1] == Approx(0.0).margin(3.0e-5));
    REQUIRE(momentum[2] == Approx(0.0).margin(3.0e-5));
    REQUIRE(kinetic_work < 0.0);
    REQUIRE(tendencies.dudt_diff[3][3][0] == 0.0f);
    REQUIRE(tendencies.dvdt_diff[nx - 1][3][2] == 0.0f);
}

TEST_CASE("Explicit horizontal scalar diffusion is live-density conservative",
          "[numerics][diffusion][horizontal][conservation]")
{
    constexpr int nx = 6;
    constexpr int ny = 7;
    constexpr int nz_local = 3;
    Field3D rho_field(nx, ny, nz_local, 1.0f);
    Field3D theta_field(nx, ny, nz_local, 300.0f);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                rho_field[i][j][k] = static_cast<float>(
                    0.7 + 0.09 * i + 0.04 * j + 0.03 * k);
                theta_field[i][j][k] = static_cast<float>(
                    297.0 + 0.6 * i * i - 0.35 * j + 0.2 * i * j
                    + 0.1 * k);
            }
    // Periodic closure storage in x and y. The production boundary pass owns
    // these planes; pairing them here makes the closed face-flux identity
    // independently inspectable.
    for (int j = 1; j < ny - 1; ++j)
        for (int k = 0; k < nz_local; ++k)
        {
            rho_field[0][j][k] = rho_field[nx - 2][j][k];
            rho_field[nx - 1][j][k] = rho_field[1][j][k];
            theta_field[0][j][k] = theta_field[nx - 2][j][k];
            theta_field[nx - 1][j][k] = theta_field[1][j][k];
        }
    for (int i = 0; i < nx; ++i)
        for (int k = 0; k < nz_local; ++k)
        {
            rho_field[i][0][k] = rho_field[i][ny - 2][k];
            rho_field[i][ny - 1][k] = rho_field[i][1][k];
            theta_field[i][0][k] = theta_field[i][ny - 2][k];
            theta_field[i][ny - 1][k] = theta_field[i][1][k];
        }
    // Boundary semantics are carried explicitly below. Poisoning every
    // closure plane proves neither the operator nor its stability check can
    // silently sample stale scalar storage.
    const float poison = std::numeric_limits<float>::quiet_NaN();
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
                for (int k = 0; k < nz_local; ++k)
                {
                    rho_field[i][j][k] = poison;
                    theta_field[i][j][k] = poison;
                }

    GridMetrics metrics;
    metrics.dx = 2.0;
    metrics.dy = 3.0;
    metrics.dz.assign(nz_local, 5.0);
    DiffusionConfig cfg;
    cfg.K_h = 4.0;
    cfg.K_v = 0.0;
    cfg.dt_diffusion = 0.01;
    auto scheme = create_diffusion_scheme("explicit");
    scheme->initialize(cfg);

    DiffusionStateView state;
    state.theta = &theta_field;
    state.rho = &rho_field;
    state.grid = &metrics;
    state.periodic_x = true;
    state.periodic_y = true;
    DiffusionTendencies tendencies;
    DiffusionDiagnostics diagnostics;
    scheme->compute_diffusion_tendencies(
        cfg, state, tendencies, &diagnostics);

    double rho_theta_tendency = 0.0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
                rho_theta_tendency += static_cast<double>(rho_field[i][j][k])
                    * static_cast<double>(tendencies.dthetadt_diff[i][j][k]);
    REQUIRE(rho_theta_tendency == Approx(0.0).margin(2.0e-5));
    REQUIRE(tendencies.dthetadt_diff[0][2][1] == 0.0f);
    REQUIRE(tendencies.dthetadt_diff[nx - 1][2][1] == 0.0f);
    REQUIRE(diagnostics.max_diffusion_number > 0.0);
    REQUIRE(diagnostics.max_diffusion_number < 1.0);
    REQUIRE(diagnostics.K_effective[2][2][1] == Approx(4.0f));

    state.periodic_x = false;
    state.periodic_y = false;
    scheme->compute_diffusion_tendencies(
        cfg, state, tendencies, nullptr);
    double closed_rho_theta_tendency = 0.0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
                closed_rho_theta_tendency += static_cast<double>(rho_field[i][j][k])
                    * static_cast<double>(tendencies.dthetadt_diff[i][j][k]);
    REQUIRE(closed_rho_theta_tendency == Approx(0.0).margin(2.0e-5));

    theta_field.fill(301.25f);
    scheme->compute_diffusion_tendencies(
        cfg, state, tendencies, nullptr);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
                REQUIRE(tendencies.dthetadt_diff[i][j][k] == 0.0f);
}

TEST_CASE("Explicit vertical momentum diffusion damps and cannot create kinetic energy",
          "[numerics][diffusion][vertical][momentum][energy]")
{
    constexpr int nx = 3;
    constexpr int ny = 3;
    constexpr int nz_local = 5;
    Field3D u_field(nx, ny, nz_local, 0.0f);
    Field3D v_field(nx, ny, nz_local, 0.0f);
    Field3D w_field(nx, ny, nz_local, 0.0f);
    Field3D rho_field(nx, ny, nz_local, 1.0f);
    const float poison = std::numeric_limits<float>::quiet_NaN();
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            u_field[i][j][0] = poison;
            v_field[i][j][0] = poison;
            w_field[i][j][0] = poison;
            rho_field[i][j][0] = poison;
            u_field[i][j][nz_local - 1] = poison;
            v_field[i][j][nz_local - 1] = poison;
            w_field[i][j][nz_local - 1] = poison;
            rho_field[i][j][nz_local - 1] = poison;
        }
    u_field[1][1][2] = 2.0f;

    GridMetrics metrics;
    metrics.dx = 10.0;
    metrics.dy = 10.0;
    metrics.dz.assign(nz_local, 2.0);
    DiffusionConfig cfg;
    cfg.K_h = 0.0;
    cfg.K_v = 4.0;
    cfg.dt_diffusion = 0.1;
    cfg.apply_to = "scalars_and_momentum";
    auto scheme = create_diffusion_scheme("explicit");
    scheme->initialize(cfg);

    DiffusionStateView state;
    state.u = &u_field;
    state.v = &v_field;
    state.w = &w_field;
    state.rho = &rho_field;
    state.grid = &metrics;
    state.c_grid = false;
    DiffusionTendencies tendencies;
    DiffusionDiagnostics diagnostics;
    scheme->compute_diffusion_tendencies(cfg, state, tendencies, &diagnostics);

    REQUIRE(tendencies.dudt_diff[1][1][2] == Approx(-4.0f));
    REQUIRE(tendencies.dudt_diff[1][1][1] == Approx(2.0f));
    REQUIRE(tendencies.dudt_diff[1][1][3] == Approx(2.0f));
    double summed_tendency = 0.0;
    double kinetic_work = 0.0;
    for (int k = 1; k < nz_local - 1; ++k)
    {
        summed_tendency += tendencies.dudt_diff[1][1][k];
        kinetic_work += static_cast<double>(u_field[1][1][k]) *
                        static_cast<double>(tendencies.dudt_diff[1][1][k]);
    }
    REQUIRE(summed_tendency == Approx(0.0));
    REQUIRE(kinetic_work < 0.0);
    REQUIRE(diagnostics.max_diffusion_number == Approx(0.2));
    REQUIRE(tendencies.dudt_diff[1][1][0] == 0.0f);
    REQUIRE(tendencies.dudt_diff[1][1][nz_local - 1] == 0.0f);
}

TEST_CASE("Explicit C-grid vertical diffusion closes live-density variable-K budgets",
          "[numerics][diffusion][vertical][cgrid][variable-k][conservation]")
{
    constexpr int nx = 6;
    constexpr int ny = 6;
    constexpr int nz_local = 5;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    Field3D rho_field(nx, ny, nz_local, poison);
    Field3D theta_field(nx, ny, nz_local, poison);
    Field3D u_field(nx, ny, nz_local, poison);
    Field3D v_field(nx, ny, nz_local, poison);
    Field3D w_field(nx, ny, nz_local, poison);
    Field3D scalar_k(nx, ny, nz_local, poison);
    Field3D momentum_k(nx, ny, nz_local, poison);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                rho_field[i][j][k] = static_cast<float>(
                    0.75 + 0.06 * i + 0.035 * j + 0.08 * k);
                theta_field[i][j][k] = static_cast<float>(
                    298.0 + 0.4 * i - 0.2 * j + 0.3 * k * k);
                scalar_k[i][j][k] = static_cast<float>(
                    1.0 + 0.15 * i + 0.08 * k);
                momentum_k[i][j][k] = static_cast<float>(
                    1.3 + 0.09 * j + 0.11 * k);
                if (k < nz_local - 1)
                {
                    u_field[i][j][k] = static_cast<float>(
                        2.0 + 0.1 * i - 0.04 * j + 0.25 * k * k);
                    v_field[i][j][k] = static_cast<float>(
                        -1.0 + 0.07 * i + 0.03 * j - 0.18 * k * k);
                    w_field[i][j][k] = static_cast<float>(
                        0.2 - 0.05 * i + 0.02 * j + 0.12 * k * k);
                }
            }

    GridMetrics metrics;
    metrics.dx = 3.0;
    metrics.dy = 4.0;
    metrics.dz = {1.5, 2.0, 2.8, 3.6, 4.5};
    DiffusionConfig cfg;
    cfg.K_h = 0.0;
    cfg.K_v = 0.0;
    cfg.dt_diffusion = 0.01;
    cfg.use_variable_K = true;
    cfg.apply_to = "scalars_and_momentum";
    auto scheme = create_diffusion_scheme("explicit");
    scheme->initialize(cfg);

    DiffusionStateView state;
    state.theta = &theta_field;
    state.u = &u_field;
    state.v = &v_field;
    state.w = &w_field;
    state.rho = &rho_field;
    state.K_scalar = &scalar_k;
    state.K_momentum = &momentum_k;
    state.grid = &metrics;
    state.c_grid = true;
    state.periodic_x = true;
    state.periodic_y = true;
    DiffusionTendencies tendencies;
    DiffusionDiagnostics diagnostics;
    scheme->compute_diffusion_tendencies(cfg, state, tendencies, &diagnostics);

    double scalar_inventory = 0.0;
    double momentum_inventory[3] = {0.0, 0.0, 0.0};
    double work = 0.0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
                scalar_inventory += static_cast<double>(rho_field[i][j][k]) *
                    metrics.dz[static_cast<std::size_t>(k)] *
                    static_cast<double>(tendencies.dthetadt_diff[i][j][k]);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local - 1; ++k)
            {
                const int right = i == nx - 2 ? 1 : i + 1;
                const int north = j == ny - 2 ? 1 : j + 1;
                const double rho_u = 0.5 * (
                    static_cast<double>(rho_field[i][j][k]) +
                    static_cast<double>(rho_field[right][j][k]));
                const double rho_v = 0.5 * (
                    static_cast<double>(rho_field[i][j][k]) +
                    static_cast<double>(rho_field[i][north][k]));
                const double dz_uv = metrics.dz[static_cast<std::size_t>(k)];
                const double w_mass_per_area = 0.5 * (
                    static_cast<double>(rho_field[i][j][k]) *
                        metrics.dz[static_cast<std::size_t>(k)] +
                    static_cast<double>(rho_field[i][j][k + 1]) *
                        metrics.dz[static_cast<std::size_t>(k + 1)]);
                const double du = tendencies.dudt_diff[i][j][k];
                const double dv = tendencies.dvdt_diff[i][j][k];
                const double dw = tendencies.dwdt_diff[i][j][k];
                momentum_inventory[0] += rho_u * dz_uv * du;
                momentum_inventory[1] += rho_v * dz_uv * dv;
                momentum_inventory[2] += w_mass_per_area * dw;
                work += rho_u * dz_uv * static_cast<double>(u_field[i][j][k]) * du
                      + rho_v * dz_uv * static_cast<double>(v_field[i][j][k]) * dv
                      + w_mass_per_area *
                          static_cast<double>(w_field[i][j][k]) * dw;
            }
    REQUIRE(scalar_inventory == Approx(0.0).margin(2.0e-5));
    REQUIRE(momentum_inventory[0] == Approx(0.0).margin(2.0e-5));
    REQUIRE(momentum_inventory[1] == Approx(0.0).margin(2.0e-5));
    REQUIRE(momentum_inventory[2] == Approx(0.0).margin(2.0e-5));
    REQUIRE(work < 0.0);
    REQUIRE(diagnostics.max_diffusion_number > 0.0);
    REQUIRE(diagnostics.max_diffusion_number < 1.0);
    REQUIRE(tendencies.dwdt_diff[2][2][nz_local - 1] == 0.0f);
    REQUIRE(diagnostics.K_effective[0][2][2] == 0.0f);
}

TEST_CASE("Linear-height variable diffusion uses scalar-cell heights and conservative faces",
          "[numerics][diffusion][variable-k][profile][threading]")
{
    constexpr int nx = 5;
    constexpr int ny = 5;
    const std::vector<double> layers{1.0, 3.0, 2.0, 6.0};
    Field3D coefficient = tmv::build_linear_height_diffusivity(
        nx, ny, layers, 4.0, 20.0);
    REQUIRE(coefficient(2, 2, 0) == 4.0f);
    REQUIRE(coefficient(2, 2, 1) == Approx(4.0 + 16.0 * 2.0 / 8.5));
    REQUIRE(coefficient(2, 2, 2) == Approx(4.0 + 16.0 * 4.5 / 8.5));
    REQUIRE(coefficient(2, 2, 3) == 20.0f);

    DiffusionRuntimeState runtime;
    runtime.config.use_variable_K = true;
    runtime.config.coefficient_source = "linear_height";
    runtime.config.K_v = 0.0;
    runtime.config.K_v_surface = 4.0;
    runtime.config.K_v_top = 20.0;
    GridMetrics owned_metrics;
    owned_metrics.dz = layers;
    const Field3D* owned = tmv::prescribed_vertical_diffusivity(
        runtime, nx, ny, owned_metrics);
    REQUIRE(owned == &runtime.prescribed_vertical_K);
    REQUIRE((*owned)(2, 2, 0) == 4.0f);
    REQUIRE((*owned)(2, 2, 3) == 20.0f);
    const float prior_middle = (*owned)(2, 2, 1);
    owned_metrics.dz = {2.0, 1.0, 4.0, 5.0};
    owned = tmv::prescribed_vertical_diffusivity(
        runtime, nx, ny, owned_metrics);
    REQUIRE((*owned)(2, 2, 1) != prior_middle);
    REQUIRE(runtime.prescribed_vertical_dz == owned_metrics.dz);
    const Field3D before_invalid_geometry = *owned;
    owned_metrics.terrain_metrics_active = true;
    REQUIRE_THROWS_AS(tmv::prescribed_vertical_diffusivity(
        runtime, nx, ny, owned_metrics), std::invalid_argument);
    REQUIRE(same_field_bits(
        runtime.prescribed_vertical_K, before_invalid_geometry));

    Field3D rho(nx, ny, 4, 1.0f);
    Field3D scalar(nx, ny, 4, 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < 4; ++k)
                scalar(i, j, k) = static_cast<float>(k * k);
    GridMetrics metrics;
    metrics.dx = 10.0;
    metrics.dy = 10.0;
    metrics.dz = layers;
    DiffusionConfig config;
    config.K_h = 0.0;
    config.K_v = 0.0;
    config.use_variable_K = true;
    config.coefficient_source = "linear_height";
    config.K_v_surface = 4.0;
    config.K_v_top = 20.0;
    DiffusionStateView state;
    state.theta = &scalar;
    state.rho = &rho;
    state.K_scalar = &coefficient;
    state.grid = &metrics;
    state.c_grid = true;
    ExplicitDiffusionScheme scheme;
    scheme.initialize(config);

    DiffusionTendencies one_thread;
    DiffusionTendencies four_threads;
#ifdef _OPENMP
    const int prior_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    scheme.compute_diffusion_tendencies(
        config, state, one_thread, nullptr);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    scheme.compute_diffusion_tendencies(
        config, state, four_threads, nullptr);
#ifdef _OPENMP
    omp_set_num_threads(prior_threads);
#endif
    REQUIRE(same_field_bits(one_thread.dthetadt_diff,
                            four_threads.dthetadt_diff));

    double inventory = 0.0;
    for (int k = 0; k < 4; ++k)
    {
        const double lower_flux = k == 0 ? 0.0 :
            0.5 * (coefficient(2, 2, k - 1) + coefficient(2, 2, k)) *
            (scalar(2, 2, k) - scalar(2, 2, k - 1)) /
            (0.5 * (layers[static_cast<std::size_t>(k - 1)] +
                    layers[static_cast<std::size_t>(k)]));
        const double upper_flux = k == 3 ? 0.0 :
            0.5 * (coefficient(2, 2, k) + coefficient(2, 2, k + 1)) *
            (scalar(2, 2, k + 1) - scalar(2, 2, k)) /
            (0.5 * (layers[static_cast<std::size_t>(k)] +
                    layers[static_cast<std::size_t>(k + 1)]));
        const double expected = (upper_flux - lower_flux) /
            layers[static_cast<std::size_t>(k)];
        REQUIRE(one_thread.dthetadt_diff(2, 2, k) ==
                Approx(expected).margin(2.0e-6));
        inventory += layers[static_cast<std::size_t>(k)] *
            one_thread.dthetadt_diff(2, 2, k);
    }
    REQUIRE(inventory == Approx(0.0).margin(2.0e-6));

    const Field3D before = one_thread.dthetadt_diff;
    Field3D invalid_coefficient = coefficient;
    invalid_coefficient(2, 2, 2) = -1.0f;
    state.K_scalar = &invalid_coefficient;
    REQUIRE_THROWS_AS(scheme.compute_diffusion_tendencies(
        config, state, one_thread, nullptr), std::invalid_argument);
    REQUIRE(same_field_bits(one_thread.dthetadt_diff, before));
    REQUIRE_THROWS_AS(tmv::build_linear_height_diffusivity(
        nx, ny, layers, -1.0, 20.0), std::invalid_argument);
}

TEST_CASE("Implicit diffusion initializes with config", "[numerics][diffusion]")
{
    setup_grid();
    auto scheme = create_diffusion_scheme("implicit");
    DiffusionConfig cfg;
    cfg.scheme_id = "implicit";
    cfg.K_h = 0.0;
    cfg.K_v = 100.0;
    REQUIRE_NOTHROW(scheme->initialize(cfg));
}

TEST_CASE("Diffusion implementations reject unimplemented operator order",
          "[numerics][diffusion][config][sci021]")
{
    setup_grid();
    DiffusionConfig cfg;
    cfg.operator_order = 6;

    SECTION("explicit")
    {
        auto scheme = create_diffusion_scheme("explicit");
        REQUIRE_THROWS(scheme->initialize(cfg));
    }

    SECTION("implicit")
    {
        auto scheme = create_diffusion_scheme("implicit");
        REQUIRE_THROWS(scheme->initialize(cfg));
    }
}

TEST_CASE("Implicit diffusion rejects invalid metrics coefficients and state",
          "[numerics][diffusion][config-truth]")
{
    auto scheme = create_diffusion_scheme("implicit");
    DiffusionConfig cfg;
    cfg.scheme_id = "implicit";
    cfg.K_v = 10.0;
    cfg.dt_diffusion = 1.0;
    scheme->initialize(cfg);

    Field3D theta_field(3, 3, 5, 300.0f);
    Field3D rho_field(3, 3, 5, 1.0f);
    GridMetrics metrics;
    metrics.dx = 100.0;
    metrics.dy = 100.0;
    metrics.dz = {50.0, 50.0, 50.0, 50.0, 50.0};
    DiffusionStateView state;
    state.theta = &theta_field;
    state.rho = &rho_field;
    state.grid = &metrics;
    state.c_grid = false;
    DiffusionTendencies tendencies;
    REQUIRE_NOTHROW(
        scheme->compute_diffusion_tendencies(cfg, state, tendencies, nullptr));

    metrics.dz[1] = 0.0;
    REQUIRE_THROWS_AS(
        scheme->compute_diffusion_tendencies(cfg, state, tendencies, nullptr),
        std::invalid_argument);
    metrics.dz[1] = 50.0;

    cfg.use_variable_K = true;
    REQUIRE_THROWS_AS(
        scheme->compute_diffusion_tendencies(cfg, state, tendencies, nullptr),
        std::invalid_argument);
    Field3D scalar_k(3, 3, 5, 10.0f);
    state.K_scalar = &scalar_k;
    scalar_k[1][1][2] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        scheme->compute_diffusion_tendencies(cfg, state, tendencies, nullptr),
        std::invalid_argument);
}

TEST_CASE("Diffusion tendency outputs are transactional on late target failure",
          "[numerics][diffusion][guard][transaction]")
{
    constexpr int nx = 3;
    constexpr int ny = 3;
    constexpr int nz_local = 5;
    Field3D theta_field(nx, ny, nz_local, 300.0f);
    Field3D qv_field(nx, ny, nz_local, 0.01f);
    Field3D rho_field(nx, ny, nz_local, 1.0f);
    // Theta is solved first. This late qv failure exercises rollback after a
    // complete earlier target rather than only an initial preflight failure.
    qv_field[1][1][2] = std::numeric_limits<float>::quiet_NaN();

    GridMetrics metrics;
    metrics.dx = 100.0;
    metrics.dy = 100.0;
    metrics.dz.assign(nz_local, 50.0);
    DiffusionConfig cfg;
    cfg.K_h = 0.0;
    cfg.K_v = 4.0;
    cfg.dt_diffusion = 0.1;
    cfg.apply_to = "scalars";

    DiffusionStateView state;
    state.theta = &theta_field;
    state.qv = &qv_field;
    state.rho = &rho_field;
    state.grid = &metrics;
    state.c_grid = false;

    SECTION("explicit")
    {
        auto scheme = create_diffusion_scheme("explicit");
        scheme->initialize(cfg);
        DiffusionTendencies tendencies = sentinel_tendencies();
        const DiffusionTendencies before = tendencies;
        REQUIRE_THROWS_AS(
            scheme->compute_diffusion_tendencies(
                cfg, state, tendencies, nullptr),
            std::invalid_argument);
        REQUIRE(same_tendency_bits(tendencies, before));
    }

    SECTION("implicit")
    {
        cfg.scheme_id = "implicit";
        auto scheme = create_diffusion_scheme("implicit");
        scheme->initialize(cfg);
        DiffusionTendencies tendencies = sentinel_tendencies();
        const DiffusionTendencies before = tendencies;
        DiffusionDiagnostics diagnostics;
        diagnostics.max_diffusion_number = 19.25;
        diagnostics.K_effective = Field3D(2, 2, 2, 23.5f);
        const DiffusionDiagnostics diagnostics_before = diagnostics;
        REQUIRE_THROWS_AS(
            scheme->compute_diffusion_tendencies(
                cfg, state, tendencies, &diagnostics),
            std::invalid_argument);
        REQUIRE(same_tendency_bits(tendencies, before));
        REQUIRE(diagnostics.max_diffusion_number
                == diagnostics_before.max_diffusion_number);
        REQUIRE(same_field_bits(
            diagnostics.K_effective, diagnostics_before.K_effective));
    }
}

TEST_CASE("Mapped implicit diffusion preserves outputs on invalid scalar state",
          "[numerics][diffusion][implicit][terrain][guard][transaction]")
{
    constexpr int nx = 5;
    constexpr int ny = 5;
    constexpr int nz_local = 4;
    const TerrainMetrics3D terrain = make_uniform_slope_metrics(
        nx, ny, nz_local, 120.0, 150.0, 60.0, 1.35, 0.08, -0.05);
    Field3D theta_field(nx, ny, nz_local, 300.0f);
    Field3D qv_field(nx, ny, nz_local, 0.01f);
    Field3D rho_field(nx, ny, nz_local, 1.0f);
    qv_field(2, 2, 2) = std::numeric_limits<float>::quiet_NaN();

    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(nz_local, terrain.dzeta * 1.35);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;
    grid.staggering = StaggerType::CGrid;

    DiffusionConfig cfg;
    cfg.scheme_id = "implicit";
    cfg.K_h = 0.0;
    cfg.K_v = 4.0;
    cfg.dt_diffusion = 0.1;
    cfg.apply_to = "scalars";
    auto scheme = create_diffusion_scheme("implicit");
    scheme->initialize(cfg);

    DiffusionStateView state;
    state.theta = &theta_field;
    state.qv = &qv_field;
    state.rho = &rho_field;
    state.grid = &grid;
    state.c_grid = true;
    DiffusionTendencies tendencies = sentinel_tendencies();
    const DiffusionTendencies before = tendencies;
    DiffusionDiagnostics diagnostics;
    diagnostics.max_diffusion_number = 19.25;
    diagnostics.K_effective = Field3D(2, 2, 2, 23.5f);
    const DiffusionDiagnostics diagnostics_before = diagnostics;

    REQUIRE_THROWS_AS(
        scheme->compute_diffusion_tendencies(
            cfg, state, tendencies, &diagnostics),
        std::invalid_argument);
    REQUIRE(same_tendency_bits(tendencies, before));
    REQUIRE(diagnostics.max_diffusion_number ==
            diagnostics_before.max_diffusion_number);
    REQUIRE(same_field_bits(
        diagnostics.K_effective, diagnostics_before.K_effective));
}

TEST_CASE("Implicit backward Euler is monotone and ignores A-grid closure storage",
          "[numerics][diffusion][implicit][backward-euler][monotone]")
{
    constexpr int nx = 3;
    constexpr int ny = 3;
    constexpr int nz_local = 5;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    Field3D theta_field(nx, ny, nz_local, poison);
    Field3D rho_field(nx, ny, nz_local, poison);
    theta_field[1][1][1] = 0.0f;
    theta_field[1][1][2] = 2.0f;
    theta_field[1][1][3] = 0.0f;
    rho_field[1][1][1] = 1.0f;
    rho_field[1][1][2] = 1.0f;
    rho_field[1][1][3] = 1.0f;

    GridMetrics metrics;
    metrics.dx = 10.0;
    metrics.dy = 10.0;
    metrics.dz.assign(nz_local, 2.0);
    DiffusionConfig cfg;
    cfg.scheme_id = "implicit";
    cfg.K_h = 0.0;
    cfg.K_v = 4.0;
    cfg.dt_diffusion = 100.0;
    auto scheme = create_diffusion_scheme("implicit");
    scheme->initialize(cfg);

    DiffusionStateView state;
    state.theta = &theta_field;
    state.rho = &rho_field;
    state.grid = &metrics;
    state.c_grid = false;
    DiffusionTendencies tendencies;
    DiffusionDiagnostics diagnostics;
    REQUIRE_NOTHROW(scheme->compute_diffusion_tendencies(
        cfg, state, tendencies, &diagnostics));

    double updated_sum = 0.0;
    for (int k = 1; k < nz_local - 1; ++k)
    {
        const double updated = static_cast<double>(theta_field[1][1][k]) +
            cfg.dt_diffusion * tendencies.dthetadt_diff[1][1][k];
        REQUIRE(std::isfinite(updated));
        REQUIRE(updated >= -1.0e-6);
        REQUIRE(updated <= 2.0 + 1.0e-6);
        REQUIRE(updated == Approx(2.0 / 3.0).margin(0.007));
        updated_sum += updated;
    }
    REQUIRE(updated_sum == Approx(2.0).margin(2.0e-6));
    REQUIRE(tendencies.dthetadt_diff[1][1][0] == 0.0f);
    REQUIRE(tendencies.dthetadt_diff[1][1][nz_local - 1] == 0.0f);
    REQUIRE(tendencies.dthetadt_diff[0][1][2] == 0.0f);
    REQUIRE(diagnostics.K_effective[0][1][2] == 0.0f);
    REQUIRE(diagnostics.max_diffusion_number == 0.0);
    REQUIRE(scheme->check_stability(cfg, state) == 0.0);
}

TEST_CASE("Implicit C-grid variable-K diffusion closes all live-density budgets",
          "[numerics][diffusion][implicit][cgrid][variable-k][conservation]")
{
    constexpr int nx = 5;
    constexpr int ny = 5;
    constexpr int nz_local = 4;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    Field3D rho_field(nx, ny, nz_local, poison);
    Field3D theta_field(nx, ny, nz_local, poison);
    Field3D u_field(nx, ny, nz_local, poison);
    Field3D v_field(nx, ny, nz_local, poison);
    Field3D w_field(nx, ny, nz_local, poison);
    Field3D scalar_k(nx, ny, nz_local, poison);
    Field3D momentum_k(nx, ny, nz_local, poison);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                rho_field[i][j][k] = static_cast<float>(
                    0.8 + 0.07 * i + 0.04 * j + 0.11 * k);
                theta_field[i][j][k] = static_cast<float>(
                    297.0 + 0.2 * i - 0.1 * j + 0.6 * k * k);
                scalar_k[i][j][k] = static_cast<float>(
                    0.7 + 0.12 * i + 0.09 * k);
                momentum_k[i][j][k] = static_cast<float>(
                    1.1 + 0.08 * j + 0.13 * k);
                if (k < nz_local - 1)
                {
                    u_field[i][j][k] = static_cast<float>(
                        2.0 + 0.05 * i - 0.03 * j + 0.8 * k * k);
                    v_field[i][j][k] = static_cast<float>(
                        -1.0 + 0.04 * i + 0.02 * j - 0.55 * k * k);
                    w_field[i][j][k] = static_cast<float>(
                        0.2 - 0.02 * i + 0.01 * j + 0.35 * k * k);
                }
            }

    GridMetrics metrics;
    metrics.dx = 3.0;
    metrics.dy = 4.0;
    metrics.dz = {1.2, 1.9, 2.7, 3.8};
    DiffusionConfig cfg;
    cfg.scheme_id = "implicit";
    cfg.K_h = 0.0;
    cfg.K_v = 0.0;
    cfg.dt_diffusion = 3.0;
    cfg.use_variable_K = true;
    cfg.apply_to = "scalars_and_momentum";
    auto scheme = create_diffusion_scheme("implicit");
    scheme->initialize(cfg);

    DiffusionStateView state;
    state.theta = &theta_field;
    state.u = &u_field;
    state.v = &v_field;
    state.w = &w_field;
    state.rho = &rho_field;
    state.K_scalar = &scalar_k;
    state.K_momentum = &momentum_k;
    state.grid = &metrics;
    state.c_grid = true;
    state.periodic_x = true;
    state.periodic_y = true;
    DiffusionTendencies tendencies;
    DiffusionDiagnostics diagnostics;
    scheme->compute_diffusion_tendencies(cfg, state, tendencies, &diagnostics);

    double scalar_inventory = 0.0;
    double momentum_inventory[3] = {0.0, 0.0, 0.0};
    double kinetic_before = 0.0;
    double kinetic_after = 0.0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                scalar_inventory += static_cast<double>(rho_field[i][j][k]) *
                    metrics.dz[static_cast<std::size_t>(k)] *
                    static_cast<double>(tendencies.dthetadt_diff[i][j][k]);
                const double updated = theta_field[i][j][k] +
                    cfg.dt_diffusion * tendencies.dthetadt_diff[i][j][k];
                REQUIRE(updated >= theta_field[i][j][0] - 1.0e-5);
                REQUIRE(updated <= theta_field[i][j][nz_local - 1] + 1.0e-5);
            }
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local - 1; ++k)
            {
                const int right = i == nx - 2 ? 1 : i + 1;
                const int north = j == ny - 2 ? 1 : j + 1;
                const double rho_u = 0.5 * (rho_field[i][j][k] +
                                            rho_field[right][j][k]);
                const double rho_v = 0.5 * (rho_field[i][j][k] +
                                            rho_field[i][north][k]);
                const double dz_uv = metrics.dz[static_cast<std::size_t>(k)];
                const double w_mass_per_area = 0.5 * (
                    static_cast<double>(rho_field[i][j][k]) *
                        metrics.dz[static_cast<std::size_t>(k)] +
                    static_cast<double>(rho_field[i][j][k + 1]) *
                        metrics.dz[static_cast<std::size_t>(k + 1)]);
                const double weights[3] = {
                    rho_u * dz_uv, rho_v * dz_uv, w_mass_per_area};
                const double old_values[3] = {
                    u_field[i][j][k], v_field[i][j][k], w_field[i][j][k]};
                const double rates[3] = {
                    tendencies.dudt_diff[i][j][k],
                    tendencies.dvdt_diff[i][j][k],
                    tendencies.dwdt_diff[i][j][k]};
                for (int component = 0; component < 3; ++component)
                {
                    momentum_inventory[component] +=
                        weights[component] * rates[component];
                    kinetic_before += 0.5 * weights[component] *
                        old_values[component] * old_values[component];
                    const double updated = old_values[component] +
                        cfg.dt_diffusion * rates[component];
                    kinetic_after += 0.5 * weights[component] *
                        updated * updated;
                }
            }
    REQUIRE(scalar_inventory == Approx(0.0).margin(1.0e-4));
    REQUIRE(momentum_inventory[0] == Approx(0.0).margin(1.0e-4));
    REQUIRE(momentum_inventory[1] == Approx(0.0).margin(1.0e-4));
    REQUIRE(momentum_inventory[2] == Approx(0.0).margin(1.0e-4));
    REQUIRE(kinetic_after < kinetic_before);
    REQUIRE(tendencies.dudt_diff[0][2][1] == 0.0f);
    REQUIRE(tendencies.dwdt_diff[2][2][nz_local - 1] == 0.0f);
    REQUIRE(diagnostics.K_effective[0][2][1] == 0.0f);
    REQUIRE(diagnostics.max_diffusion_number == 0.0);
}

TEST_CASE("Explicit and implicit vertical paths converge to the same C-grid equation",
          "[numerics][diffusion][implicit][explicit][equivalence]")
{
    constexpr int nx = 3;
    constexpr int ny = 3;
    constexpr int nz_local = 4;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    Field3D theta_field(nx, ny, nz_local, poison);
    Field3D rho_field(nx, ny, nz_local, poison);
    Field3D scalar_k(nx, ny, nz_local, poison);
    const float values[nz_local] = {0.2f, 1.3f, -0.4f, 0.7f};
    for (int k = 0; k < nz_local; ++k)
    {
        theta_field[1][1][k] = values[k];
        rho_field[1][1][k] = static_cast<float>(0.8 + 0.17 * k);
        scalar_k[1][1][k] = static_cast<float>(0.6 + 0.21 * k);
    }
    GridMetrics metrics;
    metrics.dx = 10.0;
    metrics.dy = 10.0;
    metrics.dz = {0.9, 1.4, 2.1, 3.0};
    DiffusionConfig cfg;
    cfg.K_h = 0.0;
    cfg.K_v = 0.0;
    cfg.dt_diffusion = 1.0e-4;
    cfg.use_variable_K = true;
    auto explicit_scheme = create_diffusion_scheme("explicit");
    auto implicit_scheme = create_diffusion_scheme("implicit");
    explicit_scheme->initialize(cfg);
    implicit_scheme->initialize(cfg);
    DiffusionStateView state;
    state.theta = &theta_field;
    state.rho = &rho_field;
    state.K_scalar = &scalar_k;
    state.grid = &metrics;
    state.c_grid = true;
    DiffusionTendencies explicit_tendencies;
    DiffusionTendencies implicit_tendencies;
    explicit_scheme->compute_diffusion_tendencies(
        cfg, state, explicit_tendencies, nullptr);
    implicit_scheme->compute_diffusion_tendencies(
        cfg, state, implicit_tendencies, nullptr);
    for (int k = 0; k < nz_local; ++k)
        REQUIRE(implicit_tendencies.dthetadt_diff[1][1][k] ==
                Approx(explicit_tendencies.dthetadt_diff[1][1][k])
                    .margin(0.002));
}

TEST_CASE("Mapped implicit scalar diffusion matches the conservative vertical equation",
          "[numerics][diffusion][implicit][terrain][scalar][conservation]")
{
    constexpr int nx = 5;
    constexpr int ny = 5;
    constexpr int nz_local = 4;
    const TerrainMetrics3D terrain = make_uniform_slope_metrics(
        nx, ny, nz_local, 120.0, 150.0, 60.0, 1.35, 0.08, -0.05);
    Field3D theta_field(nx, ny, nz_local, 0.0f);
    Field3D rho_field(nx, ny, nz_local, 0.0f);
    Field3D scalar_k(nx, ny, nz_local, 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                theta_field(i, j, k) = static_cast<float>(
                    280.0 + 0.7 * k * k);
                rho_field(i, j, k) = static_cast<float>(
                    1.15 - 0.08 * k);
                scalar_k(i, j, k) = static_cast<float>(
                    1.2 + 0.25 * k);
            }

    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(nz_local, terrain.dzeta * 1.35);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;
    grid.staggering = StaggerType::CGrid;

    DiffusionConfig cfg;
    cfg.scheme_id = "implicit";
    cfg.K_h = 0.0;
    cfg.K_v = 0.0;
    cfg.dt_diffusion = 1.0e-4;
    cfg.use_variable_K = true;
    cfg.apply_to = "scalars";
    auto implicit_scheme = create_diffusion_scheme("implicit");
    auto explicit_scheme = create_diffusion_scheme("explicit");
    implicit_scheme->initialize(cfg);
    explicit_scheme->initialize(cfg);

    DiffusionStateView state;
    state.theta = &theta_field;
    state.rho = &rho_field;
    state.K_scalar = &scalar_k;
    state.grid = &grid;
    state.c_grid = true;
    DiffusionTendencies implicit_tendencies;
    DiffusionTendencies explicit_tendencies;
    REQUIRE_NOTHROW(implicit_scheme->compute_diffusion_tendencies(
        cfg, state, implicit_tendencies, nullptr));
    REQUIRE_NOTHROW(explicit_scheme->compute_diffusion_tendencies(
        cfg, state, explicit_tendencies, nullptr));
    REQUIRE(implicit_scheme->check_stability(cfg, state) == 0.0);

    double inventory_rate = 0.0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                REQUIRE(implicit_tendencies.dthetadt_diff(i, j, k) ==
                    Approx(explicit_tendencies.dthetadt_diff(i, j, k))
                        .margin(0.003));
                const double thickness = terrain.z_face(i, j, k + 1)
                    - terrain.z_face(i, j, k);
                inventory_rate += static_cast<double>(rho_field(i, j, k))
                    * thickness
                    * implicit_tendencies.dthetadt_diff(i, j, k);
            }
    REQUIRE(inventory_rate == Approx(0.0).margin(2.0e-3));

    Field3D u_field(nx, ny, nz_local, 0.0f);
    Field3D v_field(nx, ny, nz_local, 0.0f);
    Field3D w_field(nx, ny, nz_local, 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                const int velocity_level = std::min(k, nz_local - 2);
                u_field(i, j, k) = static_cast<float>(
                    1.0 + 0.4 * velocity_level * velocity_level);
                v_field(i, j, k) = static_cast<float>(
                    -2.0 - 0.3 * velocity_level * velocity_level);
                w_field(i, j, k) = k == nz_local - 1 ? 0.0f
                    : static_cast<float>(0.2 + 0.2 * k * k);
            }
    state.u = &u_field;
    state.v = &v_field;
    state.w = &w_field;
    state.K_momentum = &scalar_k;
    state.periodic_x = true;
    state.periodic_y = true;
    cfg.K_h = 8.0;
    cfg.dt_diffusion = 10.0;
    DiffusionTendencies momentum_tendencies;
    REQUIRE_NOTHROW(implicit_scheme->compute_diffusion_tendencies(
        cfg, state, momentum_tendencies, nullptr));
    REQUIRE(implicit_scheme->check_stability(cfg, state) == 0.0);

    Field3D updated_u = u_field;
    Field3D updated_v = v_field;
    Field3D updated_w = w_field;
    for (std::size_t index = 0; index < updated_u.size(); ++index)
    {
        updated_u.data()[index] += cfg.dt_diffusion
            * momentum_tendencies.dudt_diff.data()[index];
        updated_v.data()[index] += cfg.dt_diffusion
            * momentum_tendencies.dvdt_diff.data()[index];
        updated_w.data()[index] += cfg.dt_diffusion
            * momentum_tendencies.dwdt_diff.data()[index];
    }
    DiffusionConfig equation_cfg = cfg;
    equation_cfg.scheme_id = "explicit";
    DiffusionStateView equation_state = state;
    equation_state.u = &updated_u;
    equation_state.v = &updated_v;
    equation_state.w = &updated_w;
    DiffusionTendencies equation_tendencies;
    explicit_scheme->compute_diffusion_tendencies(
        equation_cfg, equation_state, equation_tendencies, nullptr);

    std::array<double, 3> momentum_rate{};
    double maximum_equation_residual = 0.0;
    double kinetic_before = 0.0;
    double kinetic_after = 0.0;
    const auto scalar_mass = [&](int i, int j, int k)
    {
        return static_cast<double>(rho_field(i, j, k))
            * (terrain.z_face(i, j, k + 1)
               - terrain.z_face(i, j, k));
    };
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local - 1; ++k)
            {
                const int right = i == nx - 2 ? 1 : i + 1;
                const int north = j == ny - 2 ? 1 : j + 1;
                const double weights[3] = {
                    0.5 * (scalar_mass(i, j, k)
                         + scalar_mass(right, j, k)),
                    0.5 * (scalar_mass(i, j, k)
                         + scalar_mass(i, north, k)),
                    0.5 * (scalar_mass(i, j, k)
                         + scalar_mass(i, j, k + 1)),
                };
                const double old_values[3] = {
                    u_field(i, j, k), v_field(i, j, k), w_field(i, j, k),
                };
                const double rates[3] = {
                    momentum_tendencies.dudt_diff(i, j, k),
                    momentum_tendencies.dvdt_diff(i, j, k),
                    momentum_tendencies.dwdt_diff(i, j, k),
                };
                const double equation_rates[3] = {
                    equation_tendencies.dudt_diff(i, j, k),
                    equation_tendencies.dvdt_diff(i, j, k),
                    equation_tendencies.dwdt_diff(i, j, k),
                };
                for (int component = 0; component < 3; ++component)
                {
                    maximum_equation_residual = std::max(
                        maximum_equation_residual,
                        std::abs(rates[component]
                                 - equation_rates[component]));
                    momentum_rate[component] +=
                        weights[component] * rates[component];
                    kinetic_before += 0.5 * weights[component]
                        * old_values[component] * old_values[component];
                    const double updated = old_values[component]
                        + cfg.dt_diffusion * rates[component];
                    kinetic_after += 0.5 * weights[component]
                        * updated * updated;
                }
            }
    INFO("maximum backward-Euler momentum residual="
         << maximum_equation_residual);
    INFO("momentum rates=" << momentum_rate[0] << ","
         << momentum_rate[1] << "," << momentum_rate[2]);
    REQUIRE(maximum_equation_residual < 2.0e-5);
    REQUIRE(std::all_of(momentum_rate.begin(), momentum_rate.end(),
                        [](double value) { return std::isfinite(value); }));
    REQUIRE(kinetic_after < kinetic_before);
}

TEST_CASE("Full implicit scalar diffusion is conservative monotone and thread deterministic",
          "[numerics][diffusion][implicit][horizontal][scalar][backward-euler]")
{
    constexpr int nx = 7;
    constexpr int ny = 7;
    constexpr int nz_local = 4;
    Field3D theta_field(nx, ny, nz_local, 0.0f);
    Field3D rho_field(nx, ny, nz_local, 0.0f);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                rho_field(i, j, k) = static_cast<float>(
                    0.8 + 0.03 * i + 0.02 * j + 0.05 * k);
                theta_field(i, j, k) = (i == 3 && j == 3 && k == 2)
                    ? 2.0f : 0.0f;
            }
    GridMetrics grid;
    grid.dx = 3.0;
    grid.dy = 4.0;
    grid.dz = {1.0, 1.4, 1.9, 2.6};
    DiffusionConfig cfg;
    cfg.scheme_id = "implicit";
    cfg.K_h = 12.0;
    cfg.K_v = 5.0;
    cfg.dt_diffusion = 40.0;
    cfg.apply_to = "scalars";
    auto scheme = create_diffusion_scheme("implicit");
    scheme->initialize(cfg);
    DiffusionStateView state;
    state.theta = &theta_field;
    state.rho = &rho_field;
    state.grid = &grid;
    state.c_grid = true;
    state.periodic_x = true;
    state.periodic_y = false;
    DiffusionTendencies one_thread;
    DiffusionTendencies four_threads;
#ifdef _OPENMP
    const int prior_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    REQUIRE_NOTHROW(scheme->compute_diffusion_tendencies(
        cfg, state, one_thread, nullptr));
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    REQUIRE_NOTHROW(scheme->compute_diffusion_tendencies(
        cfg, state, four_threads, nullptr));
#ifdef _OPENMP
    omp_set_num_threads(prior_threads);
#endif
    REQUIRE(same_field_bits(one_thread.dthetadt_diff,
                            four_threads.dthetadt_diff));

    double inventory_rate = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                const double updated = theta_field(i, j, k)
                    + cfg.dt_diffusion
                        * one_thread.dthetadt_diff(i, j, k);
                minimum = std::min(minimum, updated);
                maximum = std::max(maximum, updated);
                inventory_rate += rho_field(i, j, k)
                    * grid.dx * grid.dy
                    * grid.dz[static_cast<std::size_t>(k)]
                    * one_thread.dthetadt_diff(i, j, k);
            }
    REQUIRE(minimum >= -2.0e-5);
    REQUIRE(maximum <= 2.0 + 2.0e-5);
    REQUIRE(maximum < 0.2);
    REQUIRE(inventory_rate == Approx(0.0).margin(2.0e-4));
    REQUIRE(scheme->check_stability(cfg, state) == 0.0);
}

TEST_CASE("Mapped full implicit scalar diffusion solves the production mimetic equation",
          "[numerics][diffusion][implicit][terrain][horizontal][scalar]")
{
    constexpr int nx = 6;
    constexpr int ny = 6;
    constexpr int nz_local = 4;
    const TerrainMetrics3D terrain = make_uniform_slope_metrics(
        nx, ny, nz_local, 120.0, 150.0, 60.0, 1.35, 0.08, -0.05);
    Field3D scalar(nx, ny, nz_local, 0.0f);
    Field3D rho(nx, ny, nz_local, 0.0f);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                scalar(i, j, k) = static_cast<float>(
                    280.0 + 0.3 * i * i - 0.2 * j + 0.5 * k * k);
                rho(i, j, k) = static_cast<float>(
                    1.2 - 0.07 * k + 0.01 * i);
            }
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(nz_local, terrain.dzeta * 1.35);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;
    grid.staggering = StaggerType::CGrid;
    DiffusionConfig cfg;
    cfg.scheme_id = "implicit";
    cfg.K_h = 18.0;
    cfg.K_v = 7.0;
    cfg.dt_diffusion = 6.0;
    cfg.apply_to = "scalars";
    auto implicit_scheme = create_diffusion_scheme("implicit");
    implicit_scheme->initialize(cfg);
    DiffusionStateView state;
    state.theta = &scalar;
    state.rho = &rho;
    state.grid = &grid;
    state.c_grid = true;
    state.periodic_x = true;
    state.periodic_y = true;
    DiffusionTendencies implicit_tendencies;
    REQUIRE_NOTHROW(implicit_scheme->compute_diffusion_tendencies(
        cfg, state, implicit_tendencies, nullptr));

    Field3D updated = scalar;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
                updated(i, j, k) = static_cast<float>(
                    static_cast<double>(scalar(i, j, k))
                    + cfg.dt_diffusion * implicit_tendencies.dthetadt_diff(
                        i, j, k));
    DiffusionConfig explicit_cfg = cfg;
    explicit_cfg.scheme_id = "explicit";
    auto explicit_scheme = create_diffusion_scheme("explicit");
    explicit_scheme->initialize(explicit_cfg);
    DiffusionStateView updated_state = state;
    updated_state.theta = &updated;
    DiffusionTendencies equation_tendencies;
    explicit_scheme->compute_diffusion_tendencies(
        explicit_cfg, updated_state, equation_tendencies, nullptr);
    double maximum_residual = 0.0;
    double inventory_rate = 0.0;
    double inventory_scale = 0.0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                maximum_residual = std::max(maximum_residual, std::abs(
                    static_cast<double>(implicit_tendencies.dthetadt_diff(
                        i, j, k))
                    - static_cast<double>(equation_tendencies.dthetadt_diff(
                        i, j, k))));
                const double contribution = rho(i, j, k)
                    * terrain.J(i, j, k)
                    * terrain.dx * terrain.dy * terrain.dzeta
                    * implicit_tendencies.dthetadt_diff(i, j, k);
                inventory_rate += contribution;
                inventory_scale += rho(i, j, k) * terrain.J(i, j, k)
                    * terrain.dx * terrain.dy * terrain.dzeta
                    * std::abs(static_cast<double>(scalar(i, j, k)));
            }
    REQUIRE(maximum_residual < 8.0e-6);
    REQUIRE(std::abs(inventory_rate * cfg.dt_diffusion) <=
            1.0e-7 * inventory_scale);
}

TEST_CASE("Mapped collocated full implicit diffusion closes its production equation",
          "[numerics][diffusion][implicit][terrain][collocated][conservation][sci417]")
{
    constexpr int nx = 7, ny = 7, nz_local = 7;
    const TerrainMetrics3D terrain = make_uniform_slope_metrics(
        nx, ny, nz_local, 120.0, 150.0, 60.0, 1.35, 0.08, -0.05);
    Field3D theta(nx, ny, nz_local, 0.0f);
    Field3D rho(nx, ny, nz_local, 1.0f);
    Field3D u(nx, ny, nz_local, 0.0f);
    Field3D v(nx, ny, nz_local, 0.0f);
    Field3D w(nx, ny, nz_local, 0.0f);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz_local - 1; ++k)
            {
                rho(i, j, k) = static_cast<float>(
                    1.1 + 0.01 * i - 0.005 * j - 0.01 * k);
                const double shape = (i - 3.0) * (i - 3.0)
                    + 0.5 * (j - 3.0) * (j - 3.0)
                    + 0.25 * (k - 3.0) * (k - 3.0);
                theta(i, j, k) = static_cast<float>(300.0 + 0.2 * shape);
                u(i, j, k) = static_cast<float>(0.3 * shape);
                v(i, j, k) = static_cast<float>(-0.15 * shape);
                w(i, j, k) = static_cast<float>(0.08 * shape);
            }
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(nz_local, terrain.dzeta * 1.35);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;
    grid.staggering = StaggerType::Collocated;
    DiffusionConfig cfg;
    cfg.scheme_id = "implicit";
    cfg.K_h = 20.0;
    cfg.K_v = 8.0;
    cfg.dt_diffusion = 0.1;
    cfg.apply_to = "scalars_and_momentum";
    auto implicit_scheme = create_diffusion_scheme("implicit");
    implicit_scheme->initialize(cfg);
    DiffusionStateView state;
    state.theta = &theta;
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.rho = &rho;
    state.grid = &grid;
    state.c_grid = false;
    DiffusionTendencies tendencies;
    REQUIRE_NOTHROW(implicit_scheme->compute_diffusion_tendencies(
        cfg, state, tendencies, nullptr));
    REQUIRE(implicit_scheme->check_stability(cfg, state) == 0.0);

    Field3D updated_theta = theta;
    Field3D updated_u = u;
    Field3D updated_v = v;
    Field3D updated_w = w;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz_local - 1; ++k)
            {
                updated_theta(i, j, k) += cfg.dt_diffusion
                    * tendencies.dthetadt_diff(i, j, k);
                updated_u(i, j, k) += cfg.dt_diffusion
                    * tendencies.dudt_diff(i, j, k);
                updated_v(i, j, k) += cfg.dt_diffusion
                    * tendencies.dvdt_diff(i, j, k);
                updated_w(i, j, k) += cfg.dt_diffusion
                    * tendencies.dwdt_diff(i, j, k);
            }
    DiffusionConfig explicit_cfg = cfg;
    explicit_cfg.scheme_id = "explicit";
    auto explicit_scheme = create_diffusion_scheme("explicit");
    explicit_scheme->initialize(explicit_cfg);
    DiffusionStateView updated_state = state;
    updated_state.theta = &updated_theta;
    updated_state.u = &updated_u;
    updated_state.v = &updated_v;
    updated_state.w = &updated_w;
    DiffusionTendencies equation;
    REQUIRE_NOTHROW(explicit_scheme->compute_diffusion_tendencies(
        explicit_cfg, updated_state, equation, nullptr));
    double maximum_residual = 0.0;
    long double theta_inventory = 0.0L;
    long double theta_inventory_scale = 0.0L;
    long double x_momentum = 0.0L;
    long double y_momentum = 0.0L;
    long double z_momentum = 0.0L;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_local; ++k)
            {
                const bool physical = i > 0 && i + 1 < nx
                    && j > 0 && j + 1 < ny
                    && k > 0 && k + 1 < nz_local;
                if (!physical)
                {
                    REQUIRE(tendencies.dthetadt_diff(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dudt_diff(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dvdt_diff(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dwdt_diff(i, j, k) == 0.0f);
                    continue;
                }
                maximum_residual = std::max({maximum_residual,
                    std::abs(static_cast<double>(
                        tendencies.dthetadt_diff(i, j, k)
                        - equation.dthetadt_diff(i, j, k))),
                    std::abs(static_cast<double>(
                        tendencies.dudt_diff(i, j, k)
                        - equation.dudt_diff(i, j, k))),
                    std::abs(static_cast<double>(
                        tendencies.dvdt_diff(i, j, k)
                        - equation.dvdt_diff(i, j, k))),
                    std::abs(static_cast<double>(
                        tendencies.dwdt_diff(i, j, k)
                        - equation.dwdt_diff(i, j, k)))});
                const long double mass = static_cast<long double>(rho(i, j, k))
                    * terrain.J(i, j, k) * terrain.dx * terrain.dy
                    * terrain.dzeta;
                theta_inventory += mass * tendencies.dthetadt_diff(i, j, k);
                theta_inventory_scale += mass
                    * std::abs(static_cast<long double>(theta(i, j, k)));
                x_momentum += mass * tendencies.dudt_diff(i, j, k);
                y_momentum += mass * tendencies.dvdt_diff(i, j, k);
                z_momentum += mass * tendencies.dwdt_diff(i, j, k);
            }
    REQUIRE(maximum_residual < 2.0e-4);
    const long double scale = terrain.dx * terrain.dy * terrain.dzeta;
    REQUIRE(std::abs(theta_inventory * cfg.dt_diffusion)
            < 2.0e-7L * theta_inventory_scale);
    REQUIRE(std::abs(x_momentum) < 2.0e-4L * scale);
    REQUIRE(std::abs(y_momentum) < 2.0e-4L * scale);
    REQUIRE(std::abs(z_momentum) < 2.0e-4L * scale);
}

TEST_CASE("Implicit scalar increments survive a large uniform background",
          "[numerics][diffusion][implicit][background]")
{
    constexpr int n = 6;
    GridMetrics   grid;
    grid.dx = grid.dy = 100;
    grid.dz.assign(n, 100);
    DiffusionConfig config;
    config.scheme_id    = "implicit";
    config.K_h          = .1;
    config.K_v          = 0;
    config.dt_diffusion = 1;
    config.apply_to     = "scalars";
    Field3D theta_field(n, n, n, 300), density(n, n, n, 1);
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 0; k < n; ++k)
                theta_field(i, j, k) = (i + j) % 2 ? 301 : 300;
    DiffusionStateView state;
    state.theta      = &theta_field;
    state.rho        = &density;
    state.grid       = &grid;
    state.c_grid     = true;
    state.periodic_x = state.periodic_y = true;
    auto scheme                         = create_diffusion_scheme("implicit");
    scheme->initialize(config);
    DiffusionTendencies result;
    REQUIRE_NOTHROW(scheme->compute_diffusion_tendencies(config, state, result, nullptr));
    const double amplitude = .5 / (1 + 8 * config.dt_diffusion * config.K_h / (grid.dx * grid.dx));
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 0; k < n; ++k)
            {
                const float exact = static_cast<float>(300.5 + ((i + j) % 2 ? amplitude : -amplitude));
                CHECK(result.dthetadt_diff(i, j, k) == exact - theta_field(i, j, k));
                CHECK(result.dthetadt_diff(i, j, k) != 0);
            }
}

TEST_CASE("Implicit momentum increments survive a large uniform background",
          "[numerics][diffusion][implicit][background][momentum]")
{
    constexpr int n = 6;
    GridMetrics   grid;
    grid.dx = grid.dy = 100;
    grid.dz.assign(n, 100);
    DiffusionConfig config;
    config.scheme_id    = "implicit";
    config.K_h          = .2;
    config.K_v          = 0;
    config.dt_diffusion = 1;
    config.apply_to     = "scalars_and_momentum";
    Field3D theta_field(n, n, n, 300), density(n, n, n, 1);
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 0; k < n; ++k)
                theta_field(i, j, k) = j % 2 ? 301 : 300;
    DiffusionStateView state;
    Field3D            zero(n, n, n, 0);
    state.u = &theta_field;
    state.v = state.w = &zero;
    state.rho         = &density;
    state.grid        = &grid;
    state.c_grid      = true;
    state.periodic_x = state.periodic_y = true;
    auto scheme                         = create_diffusion_scheme("implicit");
    scheme->initialize(config);
    DiffusionTendencies result;
    REQUIRE_NOTHROW(scheme->compute_diffusion_tendencies(config, state, result, nullptr));
    const double amplitude = .5 / (1 + 4 * config.dt_diffusion * config.K_h / (grid.dx * grid.dx));
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 0; k < n - 1; ++k)
            {
                const float exact = static_cast<float>(300.5 + (j % 2 ? amplitude : -amplitude));
                CHECK(result.dudt_diff(i, j, k) == exact - theta_field(i, j, k));
                CHECK(result.dudt_diff(i, j, k) != 0);
            }
}
