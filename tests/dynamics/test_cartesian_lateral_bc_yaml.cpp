/**
 * @file test_cartesian_lateral_bc_yaml.cpp
 * @brief Phase C1 exit-criterion gate — `boundary_conditions.lateral.*`
 *        YAML keys actually drive simulator behavior.
 *
 * The §21 exit criterion: "A unit test verifies that changing the YAML
 * key changes the simulator output." We verify this directly at the BC
 * scheme layer (rather than through a full simulation run) so the gate
 * is fast and pinpoints exactly which face/dispatch path regressed.
 *
 * Setup: small (8,8,8) Cartesian grid, seed each interior field with a
 * distinctive value, run the BC scheme under each lateral_bc setting in
 * turn, and inspect the boundary cells.
 *
 * Expected outcomes per setting:
 *   ZeroGradient — face copies interior value (legacy v0 behavior)
 *   Rigid     — face velocity is 0; rho/p/scalars copy interior
 *   RigidSlip — face normal-velocity is 0; tangential + scalars copy
 *   Periodic  — face takes value from opposite interior (wrap)
 *
 * Periodic is set as a pair (west↔east bound together; same for
 * south↔north): setting either face to Periodic activates the wrap on
 * both, per the docstring on the dispatch.
 */

#include "catch2/catch.hpp"
#include "boundary_conditions/factory.hpp"
#include "boundary_conditions/runtime.hpp"
#include "compute/compute_backend.hpp"
#include "core/field/field3d.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "dynamics/acoustic_energy_report.hpp"
#include "dynamics/dynamics_base.hpp"
#include "dynamics/final_boundary_energy_report.hpp"
#include "numerics/runtime.hpp"
#include "terrain/terrain_base.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

using tmv::init::config::LateralBC;

class RecordingRecyclingBackend final : public ComputeBackend
{
public:
    std::string name() const override { return "recording_recycling_backend"; }
    bool initialize(std::string&) override { return true; }
    void shutdown() override {}
    void invalidate_all_resident_fields() noexcept override { ++invalidations; }

    int invalidations = 0;
};

class RecordingRecyclingBoundary final : public BoundaryConditionScheme
{
public:
    RecordingRecyclingBoundary(
        int& rebuilds,
        const RecordingRecyclingBackend& backend,
        bool& residency_was_invalidated)
        : rebuilds_(rebuilds),
          backend_(backend),
          residency_was_invalidated_(residency_was_invalidated)
    {
    }

    void apply_full() override {}
    tmv::FinalBoundaryEnergyReportV1 apply_full_attributed() override
    {
        return {};
    }
    void apply_acoustic(double) override {}
    void rebuild_after_domain_recycling() override
    {
        residency_was_invalidated_ = backend_.invalidations == 1;
        ++rebuilds_;
    }
    std::string get_scheme_name() const override { return "recording_recycling"; }

private:
    int& rebuilds_;
    const RecordingRecyclingBackend& backend_;
    bool& residency_was_invalidated_;
};

class TerrainBindingGuard
{
public:
    explicit TerrainBindingGuard(const TerrainMetrics3D& metrics)
        : previous_(std::move(global_simulation_storage.runtime.terrain))
    {
        auto& terrain = global_simulation_storage.runtime.terrain;
        terrain = {};
        terrain.config.scheme_id = "bell";
        terrain.metrics = metrics;
        global_grid_metrics.terrain_metrics_active = true;
        global_grid_metrics.terrain_metrics = &terrain.metrics;
    }
    ~TerrainBindingGuard()
    {
        global_simulation_storage.runtime.terrain = std::move(previous_);
        global_grid_metrics.terrain_metrics_active = false;
        global_grid_metrics.terrain_metrics = nullptr;
        global_grid_metrics.terrain_topography = nullptr;
    }
private:
    tmv::TerrainRuntimeState previous_;
};

constexpr int kNR = 8;
constexpr int kNTH = 8;
constexpr int kNZ = 8;
constexpr float kU_INTERIOR = 7.0f;
constexpr float kV_INTERIOR = 11.0f;
constexpr float kW_INTERIOR = 13.0f;
constexpr float kRHO_INTERIOR = 1.225f;
constexpr float kP_INTERIOR  = 100000.0f;
constexpr float kTHETA_INTERIOR = 300.0f;

void resize_and_seed_uniform()
{
    tmv::global_boundary_condition_runtime_state.clear();
    NR = kNR; NTH = kNTH; NZ = kNZ;
    dr = 1000.0; dz = 500.0; dt = 0.1; dtheta = 0.0;
    rho0_base.assign(NZ, kRHO_INTERIOR);
    p0_base.assign(NZ, kP_INTERIOR);
    theta0_base.assign(NZ, kTHETA_INTERIOR);
    qv0_base.assign(NZ, 0.0);
    u0_base.assign(NZ, 0.0); v0_base.assign(NZ, 0.0);
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_grid_metrics.terrain_metrics_active = false;
    global_grid_metrics.terrain_metrics = nullptr;
    global_grid_metrics.terrain_topography = nullptr;

    rho.resize(NR, NTH, NZ, kRHO_INTERIOR);
    p.resize(NR, NTH, NZ, kP_INTERIOR);
    u.resize(NR, NTH, NZ, kU_INTERIOR);
    v.resize(NR, NTH, NZ, kV_INTERIOR);
    w.resize(NR, NTH, NZ, kW_INTERIOR);
    theta.resize(NR, NTH, NZ, kTHETA_INTERIOR);
    tracer.resize(NR, NTH, NZ, 0.0f);
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    tke.resize(NR, NTH, NZ, 0.0f);
}

TerrainMetrics3D make_attribution_terrain(double slope_x, double slope_y)
{
    TerrainMetrics3D metrics;
    metrics.z_face.resize(kNR, kNTH, kNZ + 1, 0.0);
    metrics.Gx_face.resize(kNR, kNTH, kNZ + 1, -slope_x);
    metrics.Gy_face.resize(kNR, kNTH, kNZ + 1, -slope_y);
    for (int i = 0; i < kNR; ++i)
        for (int j = 0; j < kNTH; ++j)
        {
            const double ground =
                slope_x * static_cast<double>(i) * dr
                + slope_y * static_cast<double>(j) * dtheta;
            const double thickness =
                350.0 + 3.0 * static_cast<double>(i)
                + 2.0 * static_cast<double>(j);
            for (int k = 0; k <= kNZ; ++k)
                metrics.z_face(i, j, k) =
                    ground + static_cast<double>(k) * thickness;
        }
    return metrics;
}

double mapped_v2_kinetic_energy(const TerrainMetrics3D& metrics)
{
    double energy = 0.0;
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double q_total =
                    static_cast<double>(qv[i][j][k])
                    + static_cast<double>(qc[i][j][k])
                    + static_cast<double>(qr[i][j][k])
                    + static_cast<double>(qi[i][j][k])
                    + static_cast<double>(qs[i][j][k])
                    + static_cast<double>(qg[i][j][k])
                    + static_cast<double>(qh[i][j][k]);
                const double volume = dr * dtheta
                    * (metrics.z_face(i, j, k + 1)
                       - metrics.z_face(i, j, k));
                const double mass =
                    static_cast<double>(rho[i][j][k])
                    * (1.0 + q_total) * volume;
                const double u_w = u[i - 1][j][k];
                const double u_e = u[i][j][k];
                const double v_s = v[i][j - 1][k];
                const double v_n = v[i][j][k];
                const double w_b = k == 0
                    ? 0.5 * (u_w + u_e)
                            * (-metrics.Gx_face(i, j, 0))
                        + 0.5 * (v_s + v_n)
                            * (-metrics.Gy_face(i, j, 0))
                    : static_cast<double>(w[i][j][k - 1]);
                const double w_t = w[i][j][k];
                energy += 0.25 * mass
                    * (u_w * u_w + u_e * u_e
                       + v_s * v_s + v_n * v_n
                       + w_b * w_b + w_t * w_t);
            }
    return energy;
}

/// Mark the periodic-pair-opposite interior cells with distinct values so
/// a "this face copied from the OPPOSITE end" test has signal to compare.
void mark_opposite_x_interior()
{
    for (int j = 0; j < kNTH; ++j)
        for (int k = 0; k < kNZ; ++k)
        {
            u[1][j][k]        = 1.0f;   // for east(NR-1) ← interior(1)
            u[kNR - 2][j][k]  = 2.0f;   // for west(0)    ← interior(NR-2)
        }
}

void mark_opposite_y_interior()
{
    for (int i = 0; i < kNR; ++i)
        for (int k = 0; k < kNZ; ++k)
        {
            v[i][1][k]         = 3.0f;
            v[i][kNTH - 2][k]  = 4.0f;
        }
}

void set_bcs(LateralBC w_, LateralBC e_, LateralBC s_, LateralBC n_)
{
    global_lateral_bc_west  = w_;
    global_lateral_bc_east  = e_;
    global_lateral_bc_south = s_;
    global_lateral_bc_north = n_;
}

void reset_bcs_to_default()
{
    set_bcs(LateralBC::ZeroGradient, LateralBC::ZeroGradient,
            LateralBC::ZeroGradient, LateralBC::ZeroGradient);
}

std::vector<float> run_patterned_cgrid_boundaries(int requested_threads)
{
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(requested_threads);
#else
    (void)requested_threads;
#endif

    resize_and_seed_uniform();
    global_bottom_bc = tmv::init::config::BottomBC::RigidNoSlip;
    global_top_bc = tmv::init::config::TopBC::Open;
    set_bcs(LateralBC::Rigid, LateralBC::ZeroGradient,
            LateralBC::Periodic, LateralBC::RigidSlip);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const float pattern = static_cast<float>(100 * i + 10 * j + k);
                u[i][j][k] = pattern + 1.0f;
                v[i][j][k] = pattern + 2.0f;
                w[i][j][k] = pattern + 3.0f;
                rho[i][j][k] = pattern + 4.0f;
                p[i][j][k] = pattern + 5.0f;
            }

    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    scheme->apply_full();

    std::vector<float> result;
    result.reserve(5 * u.size());
    const auto append = [&](const Field3D& field)
    {
        result.insert(result.end(), field.data(), field.data() + field.size());
    };
    append(u);
    append(v);
    append(w);
    append(rho);
    append(p);
    return result;
}

struct LinearAcousticPulseBudget
{
    double residual_energy_fraction = 0.0;
    double integrated_outward_energy = 0.0;
    double flux_closure_residual = 0.0;
};

LinearAcousticPulseBudget linear_acoustic_pulse_budget(LateralBC east_bc)
{
    constexpr int nx = 160;
    constexpr int ny = 4;
    constexpr int nz = 4;
    constexpr double dx = 10.0;
    constexpr double sound_speed = 30.0;
    constexpr double rho_ref = 1.0;
    constexpr double step_dt = 0.1;
    constexpr int steps = 400;
    constexpr int j_probe = 1;
    constexpr int k_probe = 1;

    NR = nx; NTH = ny; NZ = nz;
    dr = dx; dtheta = dx; dz = 100.0; dt = step_dt;
    global_coordinate_system = CoordinateSystem::Cartesian;
    rho0_base.assign(nz, rho_ref);
    p0_base.assign(nz, 0.0);
    u0_base.assign(nz, 0.0);
    v0_base.assign(nz, 0.0);

    rho.resize(nx, ny, nz, static_cast<float>(rho_ref));
    p.resize(nx, ny, nz, 0.0f);
    u.resize(nx, ny, nz, 0.0f);
    v.resize(nx, ny, nz, 0.0f);
    w.resize(nx, ny, nz, 0.0f);
    theta.resize(nx, ny, nz, 300.0f);
    qv.resize(nx, ny, nz, 0.0f);
    qc.resize(nx, ny, nz, 0.0f);
    qr.resize(nx, ny, nz, 0.0f);
    qi.resize(nx, ny, nz, 0.0f);
    qs.resize(nx, ny, nz, 0.0f);
    qg.resize(nx, ny, nz, 0.0f);
    qh.resize(nx, ny, nz, 0.0f);

    // A small-amplitude right-going acoustic characteristic, centered far
    // enough from either face that the initial boundary state is quiescent.
    constexpr double center = 700.0;
    constexpr double sigma = 80.0;
    for (int i = 1; i <= nx - 2; ++i)
    {
        const double x = static_cast<double>(i) * dx;
        const double pressure =
            std::exp(-0.5 * (x - center) * (x - center) /
                     (sigma * sigma));
        p[i][j_probe][k_probe] = static_cast<float>(pressure);
    }
    for (int i = 0; i <= nx - 2; ++i)
    {
        const double x = (static_cast<double>(i) + 0.5) * dx;
        const double pressure =
            std::exp(-0.5 * (x - center) * (x - center) /
                     (sigma * sigma));
        u[i][j_probe][k_probe] =
            static_cast<float>(pressure / (rho_ref * sound_speed));
    }

    const auto energy = [&]()
    {
        double total = 0.0;
        for (int i = 1; i <= nx - 2; ++i)
        {
            const double pressure = p[i][j_probe][k_probe];
            total += pressure * pressure / (rho_ref * sound_speed * sound_speed);
        }
        for (int i = 0; i <= nx - 2; ++i)
        {
            const double velocity = u[i][j_probe][k_probe];
            total += rho_ref * velocity * velocity;
        }
        return 0.5 * dx * total;
    };

    const auto outward_power = [&]()
    {
        const int east = nx - 2;
        return static_cast<double>(p[east][j_probe][k_probe])
                   * static_cast<double>(u[east][j_probe][k_probe])
             - static_cast<double>(p[1][j_probe][k_probe])
                   * static_cast<double>(u[0][j_probe][k_probe]);
    };

    const double initial_energy = energy();
    double previous_power = outward_power();
    double integrated_outward_energy = 0.0;
    set_bcs(LateralBC::Radiative, east_bc,
            LateralBC::ZeroGradient, LateralBC::ZeroGradient);
    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    scheme->seed_acoustic_radiation();

    // Standard staggered-in-time 1-D linear-acoustic update. The only
    // difference between the two trials is the production east-face closure.
    for (int step = 0; step < steps; ++step)
    {
        for (int i = 1; i <= nx - 3; ++i)
        {
            u[i][j_probe][k_probe] -= static_cast<float>(
                step_dt * (p[i + 1][j_probe][k_probe] -
                           p[i][j_probe][k_probe]) / (rho_ref * dx));
        }
        scheme->apply_acoustic(step_dt);
        for (int i = 1; i <= nx - 2; ++i)
        {
            p[i][j_probe][k_probe] -= static_cast<float>(
                rho_ref * sound_speed * sound_speed * step_dt *
                (u[i][j_probe][k_probe] - u[i - 1][j_probe][k_probe]) / dx);
        }
        scheme->apply_full();
        const double power = outward_power();
        integrated_outward_energy +=
            0.5 * step_dt * (previous_power + power);
        previous_power = power;
    }

    const double final_energy = energy();
    return {
        final_energy / initial_energy,
        integrated_outward_energy,
        std::abs(final_energy + integrated_outward_energy - initial_energy)
            / initial_energy};
}

}  // namespace

TEST_CASE("Completed-step boundary schemes publish complete prognostic write sets",
          "[boundary][residency][ownership]")
{
    const auto field_ids = [](const tmv::BoundaryFieldWriteSet& writes)
    {
        REQUIRE(writes.complete);
        REQUIRE(writes.field_ids != nullptr);
        std::vector<std::string> result;
        result.reserve(writes.count);
        for (std::size_t i = 0; i < writes.count; ++i)
        {
            REQUIRE(writes.field_ids[i] != nullptr);
            result.emplace_back(writes.field_ids[i]);
        }
        return result;
    };

    const auto collocated = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    REQUIRE(field_ids(collocated->full_attributed_write_set()) ==
            std::vector<std::string>{
                "u", "v", "w", "rho", "p", "theta", "qv",
                "qc", "qr", "qi", "qs", "qg", "qh", "tracer", "tke"});

    const auto cgrid = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    REQUIRE(field_ids(cgrid->full_attributed_write_set()) ==
            std::vector<std::string>{"u", "v", "w", "rho", "p"});
}

// ─────────────────────────────────────────────────────────────
// Zero-gradient lateral BC: every face copies interior (legacy v0 behavior)
// ─────────────────────────────────────────────────────────────

TEST_CASE("C1 lateral=ZeroGradient copies interior to all 4 faces",
          "[boundary_conditions][c1][lateral_bc]")
{
    resize_and_seed_uniform();
    set_bcs(LateralBC::ZeroGradient, LateralBC::ZeroGradient,
            LateralBC::ZeroGradient, LateralBC::ZeroGradient);

    auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                   StaggerType::Collocated);
    scheme->apply_full();

    // Pick an arbitrary interior column to sanity-check the face copies.
    const int j = 3, k = 4;
    REQUIRE(u[0][j][k]       == Approx(kU_INTERIOR));
    REQUIRE(u[kNR - 1][j][k] == Approx(kU_INTERIOR));
    REQUIRE(v[0][j][k]       == Approx(kV_INTERIOR));
    REQUIRE(w[0][j][k]       == Approx(kW_INTERIOR));

    const int i = 3;
    REQUIRE(u[i][0][k]        == Approx(kU_INTERIOR));
    REQUIRE(v[i][kNTH - 1][k] == Approx(kV_INTERIOR));

    reset_bcs_to_default();
}

// ─────────────────────────────────────────────────────────────
// Rigid: all 3 velocity components zero at the face
// ─────────────────────────────────────────────────────────────

TEST_CASE("C1 lateral=Rigid zeroes all velocity at the face",
          "[boundary_conditions][c1][lateral_bc]")
{
    SECTION("west=Rigid")
    {
        resize_and_seed_uniform();
        set_bcs(LateralBC::Rigid, LateralBC::ZeroGradient,
                LateralBC::ZeroGradient, LateralBC::ZeroGradient);

        auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                       StaggerType::Collocated);
        scheme->apply_full();

        const int j = 2, k = 5;
        REQUIRE(u[0][j][k] == Approx(0.0f));
        REQUIRE(v[0][j][k] == Approx(0.0f));
        REQUIRE(w[0][j][k] == Approx(0.0f));
        // east face still zero-gradient
        REQUIRE(u[kNR - 1][j][k] == Approx(kU_INTERIOR));
        // Scalars still copy interior
        REQUIRE(rho[0][j][k]   == Approx(kRHO_INTERIOR));
        REQUIRE(theta[0][j][k] == Approx(kTHETA_INTERIOR));
    }

    SECTION("north=Rigid")
    {
        resize_and_seed_uniform();
        set_bcs(LateralBC::ZeroGradient, LateralBC::ZeroGradient,
                LateralBC::ZeroGradient, LateralBC::Rigid);

        auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                       StaggerType::Collocated);
        scheme->apply_full();

        const int i = 4, k = 2;
        REQUIRE(u[i][kNTH - 1][k] == Approx(0.0f));
        REQUIRE(v[i][kNTH - 1][k] == Approx(0.0f));
        REQUIRE(w[i][kNTH - 1][k] == Approx(0.0f));
        // south face still zero-gradient
        REQUIRE(v[i][0][k]        == Approx(kV_INTERIOR));
    }

    reset_bcs_to_default();
}

// ─────────────────────────────────────────────────────────────
// RigidSlip: only normal component zeroed; tangential copies
// ─────────────────────────────────────────────────────────────

TEST_CASE("C1 lateral=RigidSlip zeroes normal velocity but copies tangential",
          "[boundary_conditions][c1][lateral_bc]")
{
    SECTION("east=RigidSlip")
    {
        resize_and_seed_uniform();
        set_bcs(LateralBC::ZeroGradient, LateralBC::RigidSlip,
                LateralBC::ZeroGradient, LateralBC::ZeroGradient);

        auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                       StaggerType::Collocated);
        scheme->apply_full();

        const int j = 2, k = 5;
        // east is an x-face; normal = u; tangential = v, w.
        REQUIRE(u[kNR - 1][j][k] == Approx(0.0f));
        REQUIRE(v[kNR - 1][j][k] == Approx(kV_INTERIOR));
        REQUIRE(w[kNR - 1][j][k] == Approx(kW_INTERIOR));
    }

    SECTION("south=RigidSlip")
    {
        resize_and_seed_uniform();
        set_bcs(LateralBC::ZeroGradient, LateralBC::ZeroGradient,
                LateralBC::RigidSlip, LateralBC::ZeroGradient);

        auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                       StaggerType::Collocated);
        scheme->apply_full();

        const int i = 4, k = 2;
        // south is a y-face; normal = v; tangential = u, w.
        REQUIRE(u[i][0][k] == Approx(kU_INTERIOR));
        REQUIRE(v[i][0][k] == Approx(0.0f));
        REQUIRE(w[i][0][k] == Approx(kW_INTERIOR));
    }

    reset_bcs_to_default();
}

// ─────────────────────────────────────────────────────────────
// Periodic: face copies opposite interior (pair-bound)
// ─────────────────────────────────────────────────────────────

TEST_CASE("C1 lateral=Periodic wraps face values from opposite interior",
          "[boundary_conditions][c1][lateral_bc]")
{
    SECTION("x periodic pair")
    {
        resize_and_seed_uniform();
        mark_opposite_x_interior();
        set_bcs(LateralBC::Periodic, LateralBC::Periodic,
                LateralBC::ZeroGradient, LateralBC::ZeroGradient);

        auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                       StaggerType::Collocated);
        scheme->apply_full();

        const int j = 3, k = 4;
        // west(0) ← interior(NR-2) where u was marked = 2.0
        REQUIRE(u[0][j][k]        == Approx(2.0f));
        // east(NR-1) ← interior(1) where u was marked = 1.0
        REQUIRE(u[kNR - 1][j][k]  == Approx(1.0f));
    }

    SECTION("y periodic pair")
    {
        resize_and_seed_uniform();
        mark_opposite_y_interior();
        set_bcs(LateralBC::ZeroGradient, LateralBC::ZeroGradient,
                LateralBC::Periodic, LateralBC::Periodic);

        auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                       StaggerType::Collocated);
        scheme->apply_full();

        const int i = 2, k = 5;
        REQUIRE(v[i][0][k]         == Approx(4.0f));   // ← interior(NTH-2)
        REQUIRE(v[i][kNTH - 1][k]  == Approx(3.0f));   // ← interior(1)
    }

    reset_bcs_to_default();
}

// ─────────────────────────────────────────────────────────────
// Acoustic-substep dispatch follows the same enum
// ─────────────────────────────────────────────────────────────

TEST_CASE("C1 apply_acoustic respects lateral BC enum",
          "[boundary_conditions][c1][lateral_bc][acoustic]")
{
    resize_and_seed_uniform();
    set_bcs(LateralBC::Rigid, LateralBC::ZeroGradient,
            LateralBC::ZeroGradient, LateralBC::ZeroGradient);

    auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                   StaggerType::Collocated);
    scheme->apply_acoustic(0.0);

    const int j = 1, k = 1;
    REQUIRE(u[0][j][k]       == Approx(0.0f));         // west=Rigid
    REQUIRE(u[kNR - 1][j][k] == Approx(kU_INTERIOR));  // east=ZeroGradient

    reset_bcs_to_default();
}

TEST_CASE("SCI-017 C-grid zero-gradient and radiative closures are distinct",
          "[boundary_conditions][lateral_bc][cgrid][SCI-017]")
{
    SECTION("zero-gradient copies boundary-normal values")
    {
        resize_and_seed_uniform();
        set_bcs(LateralBC::ZeroGradient, LateralBC::ZeroGradient,
                LateralBC::ZeroGradient, LateralBC::ZeroGradient);
        const int j = 3, k = 4;
        u[0][j][k] = -9.0f;
        u[1][j][k] = 6.0f;
        u[kNR - 3][j][k] = 4.0f;
        u[kNR - 2][j][k] = 12.0f;

        auto scheme = create_boundary_condition_scheme(
            CoordinateSystem::Cartesian, StaggerType::CGrid);
        scheme->apply_acoustic(0.1);

        REQUIRE(u[0][j][k] == Approx(6.0f));
        REQUIRE(u[kNR - 2][j][k] == Approx(4.0f));
        REQUIRE(u[kNR - 1][j][k] == Approx(4.0f));
    }

    SECTION("radiative advances from saved face history")
    {
        resize_and_seed_uniform();
        set_bcs(LateralBC::Radiative, LateralBC::Radiative,
                LateralBC::Radiative, LateralBC::Radiative);
        const int j = 3, k = 4;
        u[0][j][k] = 2.0f;
        u[1][j][k] = 8.0f;

        auto scheme = create_boundary_condition_scheme(
            CoordinateSystem::Cartesian, StaggerType::CGrid);
        scheme->seed_acoustic_radiation();
        REQUIRE(tmv::global_boundary_condition_runtime_state
                    .radiative_u_west.size() ==
                static_cast<std::size_t>(kNTH * kNZ));
        REQUIRE(tmv::global_boundary_condition_runtime_state
                    .radiative_u_west[
                        static_cast<std::size_t>(j * kNZ + k)] == 2.0f);
        scheme->apply_acoustic(1.0);

        const double expected = 2.0 - (2.0 - 30.0) * (8.0 - 2.0) / dr;
        REQUIRE(u[0][j][k] == Approx(expected).epsilon(1.0e-6));
        REQUIRE(tmv::global_boundary_condition_runtime_state
                    .radiative_u_west[
                        static_cast<std::size_t>(j * kNZ + k)] ==
                Approx(expected).epsilon(1.0e-6));
        REQUIRE(u[0][j][k] != Approx(u[1][j][k]));
    }

    reset_bcs_to_default();
}

TEST_CASE("D5 C-grid recycling rebuild closes every advected scalar ghost",
          "[boundary_conditions][cgrid][domain-recycling][D5]")
{
    resize_and_seed_uniform();
    set_bcs(LateralBC::Periodic, LateralBC::Periodic,
            LateralBC::ZeroGradient, LateralBC::Radiative);

    const std::array<Field3D*, 10> advected_scalars = {
        &theta, &tracer, &qv, &qc, &qr,
        &qi, &qs, &qg, &qh, &tke,
    };
    for (std::size_t field_index = 0;
         field_index < advected_scalars.size(); ++field_index)
    {
        Field3D& field = *advected_scalars[field_index];
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                    field[i][j][k] = static_cast<float>(
                        1000 * field_index + 100 * i + 10 * j + k);
    }

    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    scheme->rebuild_after_domain_recycling();

    const int j = 3;
    const int i = 4;
    const int k = 2;
    for (const Field3D* field : advected_scalars)
    {
        INFO("advected scalar storage address=" << field);
        REQUIRE((*field)[0][j][k]
                == (*field)[NR - 2][j][k]);
        REQUIRE((*field)[NR - 1][j][k]
                == (*field)[1][j][k]);
        REQUIRE((*field)[i][0][k]
                == (*field)[i][1][k]);
        REQUIRE((*field)[i][NTH - 1][k]
                == (*field)[i][NTH - 2][k]);

        // The y pass owns lateral corners. Its source row was already closed
        // by the x pass, so the southwest corner wraps x and copies y.
        REQUIRE((*field)[0][0][k]
                == (*field)[NR - 2][1][k]);
    }

    reset_bcs_to_default();
}

TEST_CASE("D5 C-grid recycling rebuild reseeds KW78 face history",
          "[boundary_conditions][cgrid][domain-recycling][radiative][D5]")
{
    resize_and_seed_uniform();
    set_bcs(LateralBC::Radiative, LateralBC::Radiative,
            LateralBC::Radiative, LateralBC::Radiative);
    global_domain_frame =
        tmv::domain::HorizontalDomainFrame::stationary();

    const int j = 3;
    const int k = 4;
    u[0][j][k] = 2.0f;
    u[1][j][k] = 8.0f;

    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    scheme->seed_acoustic_radiation();

    // Emulate a completed whole-cell recycle. The current boundary face must
    // replace the retired domain's private KW78 history before integration.
    u[0][j][k] = 20.0f;
    u[1][j][k] = 30.0f;
    scheme->rebuild_after_domain_recycling();
    scheme->apply_acoustic(1.0);

    const double expected =
        20.0 - (20.0 - 30.0) * (30.0 - 20.0) / dr;
    REQUIRE(u[0][j][k] == Approx(expected).epsilon(1.0e-6));

    reset_bcs_to_default();
}

TEST_CASE("D5 runtime invalidates device residency before boundary rebuild",
          "[boundary_conditions][domain-recycling][residency][D5]")
{
    resize_and_seed_uniform();
    SimulationStateView state = global_simulation_state();
    auto backend = std::make_unique<RecordingRecyclingBackend>();
    RecordingRecyclingBackend* recording_backend = backend.get();
    state.compute_runtime->backend = std::move(backend);

    int rebuilds = 0;
    bool residency_was_invalidated = false;
    tmv::bc_scheme =
        std::make_unique<RecordingRecyclingBoundary>(
            rebuilds, *recording_backend, residency_was_invalidated);
    tmv::rebuild_boundary_conditions_after_domain_recycling(state);

    REQUIRE(recording_backend->invalidations == 1);
    REQUIRE(residency_was_invalidated);
    REQUIRE(rebuilds == 1);

    tmv::bc_scheme.reset();
    state.compute_runtime->backend.reset();
}

TEST_CASE("C-grid boundary parallelism preserves serial face and corner ownership",
          "[boundary_conditions][cgrid][acoustic][cpu023]")
{
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
#endif
    const std::vector<float> serial = run_patterned_cgrid_boundaries(1);
    const std::vector<float> parallel = run_patterned_cgrid_boundaries(8);

    REQUIRE(parallel == serial);

#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
#endif
    reset_bcs_to_default();
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
}

TEST_CASE("C-grid open top closes the normal face with zero gradient",
          "[boundary_conditions][cgrid][open-top][SCI-042]")
{
    resize_and_seed_uniform();
    set_bcs(LateralBC::RigidSlip, LateralBC::RigidSlip,
            LateralBC::RigidSlip, LateralBC::RigidSlip);
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Open;
    const int i = 3, j = 2;
    w[i][j][NZ - 2] = 4.25f;
    w[i][j][NZ - 1] = -99.0f;

    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    scheme->apply_acoustic(0.0);

    REQUIRE(w[i][j][NZ - 1] == Approx(4.25f));
    reset_bcs_to_default();
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
}

TEST_CASE("SCI-142 C-grid acoustic boundary commit closes native kinetic inventory",
          "[boundary_conditions][cgrid][acoustic][energy][SCI-142]")
{
    resize_and_seed_uniform();
    dtheta = 750.0;
    set_bcs(LateralBC::RigidSlip, LateralBC::ZeroGradient,
            LateralBC::RigidSlip, LateralBC::ZeroGradient);
    global_bottom_bc = tmv::init::config::BottomBC::RigidNoSlip;
    global_top_bc = tmv::init::config::TopBC::Open;

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                qv[i][j][k] =
                    static_cast<float>(0.008 + 1.0e-5 * (i + j + k));
                u[i][j][k] = static_cast<float>(1.0 + 0.2 * i + 0.03 * k);
                v[i][j][k] = static_cast<float>(-0.5 + 0.1 * j - 0.02 * k);
                w[i][j][k] = static_cast<float>(0.15 * k - 0.01 * i);
            }

    const auto resolved_kinetic = []()
    {
        const double volume = dr * dtheta * dz;
        double energy = 0.0;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const double q_total =
                        static_cast<double>(qv[i][j][k])
                        + static_cast<double>(qc[i][j][k])
                        + static_cast<double>(qr[i][j][k])
                        + static_cast<double>(qi[i][j][k])
                        + static_cast<double>(qs[i][j][k])
                        + static_cast<double>(qg[i][j][k])
                        + static_cast<double>(qh[i][j][k]);
                    const double mass =
                        static_cast<double>(rho[i][j][k])
                        * (1.0 + q_total) * volume;
                    const double u_w = u[i - 1][j][k];
                    const double u_e = u[i][j][k];
                    const double v_s = v[i][j - 1][k];
                    const double v_n = v[i][j][k];
                    const double w_b =
                        k == 0 ? 0.0 : w[i][j][k - 1];
                    const double w_t = w[i][j][k];
                    energy += 0.25 * mass
                        * (u_w * u_w + u_e * u_e
                           + v_s * v_s + v_n * v_n
                           + w_b * w_b + w_t * w_t);
                }
        return energy;
    };

    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    const double before = resolved_kinetic();
    CartesianCGridAcousticEnergyReportV1 report;
    scheme->apply_acoustic_attributed(0.0, report);
    const double after = resolved_kinetic();

    REQUIRE(report.boundary_complete);
    REQUIRE(report.boundary_application_count == 1);
    REQUIRE(report.boundary_committed_work_j
            == Approx(after - before).epsilon(2.0e-12));
    REQUIRE(report.boundary_radiative_requested_linear_work_j == 0.0);
    REQUIRE(report.boundary_radiative_finite_step_correction_j == 0.0);
    REQUIRE(report.boundary_radiative_storage_rounding_j == 0.0);
    REQUIRE(report.boundary_radiative_committed_work_j == 0.0);
    REQUIRE(report.boundary_constraint_history_committed_work_j
            == Approx(after - before).epsilon(2.0e-12));
    REQUIRE(report.committed_inventory_change_j
            == Approx(after - before).epsilon(2.0e-12));

    scheme->apply_acoustic_attributed(0.0, report);
    REQUIRE(report.boundary_application_count == 2);
    REQUIRE(report.boundary_committed_work_j
            == Approx(after - before).epsilon(2.0e-12));

    const Field3D u_before_reject = u;
    qv[1][2][2] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        scheme->apply_acoustic_attributed(0.0, report),
        std::invalid_argument);
    for (std::size_t n = 0; n < u.size(); ++n)
        REQUIRE(u.data()[n] == u_before_reject.data()[n]);
    REQUIRE(report.boundary_application_count == 2);

    reset_bcs_to_default();
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
}

TEST_CASE("SCI-146 A-grid acoustic closure changes only zero-volume halos",
          "[boundary_conditions][agrid][acoustic][energy][SCI-146]")
{
    resize_and_seed_uniform();
    dtheta = 750.0;
    set_bcs(LateralBC::RigidSlip, LateralBC::ZeroGradient,
            LateralBC::Rigid, LateralBC::Periodic);
    global_bottom_bc = tmv::init::config::BottomBC::RigidNoSlip;
    global_top_bc = tmv::init::config::TopBC::Open;

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                rho[i][j][k] =
                    static_cast<float>(0.95 + 0.002 * (i + j + k));
                p[i][j][k] =
                    static_cast<float>(93000.0 + 7.0 * i - 5.0 * j);
                u[i][j][k] =
                    static_cast<float>(1.0 + 0.2 * i - 0.03 * k);
                v[i][j][k] =
                    static_cast<float>(-0.4 + 0.1 * j + 0.02 * k);
                w[i][j][k] =
                    static_cast<float>(0.05 * i - 0.08 * j + 0.04 * k);
            }
    u[0][3][4] = -91.0f;

    const std::array<Field3D, 5> before{{u, v, w, rho, p}};
    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    CartesianAcousticEnergyReportV1 report;
    scheme->apply_acoustic_attributed(0.5, report);

    const std::array<const Field3D*, 5> after{{&u, &v, &w, &rho, &p}};
    for (std::size_t field = 0; field < after.size(); ++field)
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 1; k < NZ - 1; ++k)
                    REQUIRE((*after[field])[i][j][k]
                            == before[field][i][j][k]);

    REQUIRE(u[0][3][4] != -91.0f);
    REQUIRE(report.boundary_complete);
    REQUIRE(report.boundary_application_count == 1);
    REQUIRE(report.boundary_committed_work_j == 0.0);
    REQUIRE(report.boundary_radiative_requested_linear_work_j == 0.0);
    REQUIRE(report.boundary_radiative_finite_step_correction_j == 0.0);
    REQUIRE(report.boundary_radiative_storage_rounding_j == 0.0);
    REQUIRE(report.boundary_radiative_committed_work_j == 0.0);
    REQUIRE(report.boundary_constraint_history_committed_work_j == 0.0);
    REQUIRE(report.committed_inventory_change_j == 0.0);

    reset_bcs_to_default();
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
}

TEST_CASE("SCI-156 A-grid final closure has zero physical inventory work",
          "[boundary_conditions][agrid][final][energy][SCI-156]")
{
    resize_and_seed_uniform();
    dtheta = 750.0;
    set_bcs(LateralBC::RigidSlip, LateralBC::ZeroGradient,
            LateralBC::Rigid, LateralBC::Periodic);
    global_bottom_bc = tmv::init::config::BottomBC::RigidNoSlip;
    global_top_bc = tmv::init::config::TopBC::Open;
    u[0][3][4] = -91.0f;
    theta[2][0][3] = 999.0f;

    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    const tmv::FinalBoundaryEnergyReportV1 report =
        scheme->apply_full_attributed();

    REQUIRE(u[0][3][4] != -91.0f);
    REQUIRE(theta[2][0][3] != 999.0f);
    REQUIRE(report.applied);
    REQUIRE(report.complete);
    REQUIRE(report.constraint_committed_work_j == 0.0);
    REQUIRE(report.dry_internal_change_j == 0.0);
    REQUIRE(report.water_sensible_change_j == 0.0);
    REQUIRE(report.phase_reference_change_j == 0.0);
    REQUIRE(report.carrier_kinetic_change_j == 0.0);
    REQUIRE(report.gravitational_change_j == 0.0);
    REQUIRE(report.subgrid_tke_change_j == 0.0);
    REQUIRE(report.committed_energy_change_j == 0.0);
    REQUIRE(report.closure_residual_j == 0.0);

    reset_bcs_to_default();
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
}

TEST_CASE("SCI-143 KW78 work is separate from boundary constraints",
          "[boundary_conditions][cgrid][acoustic][energy][radiative][SCI-143]")
{
    resize_and_seed_uniform();
    dtheta = 800.0;
    set_bcs(LateralBC::Radiative, LateralBC::Radiative,
            LateralBC::Radiative, LateralBC::Radiative);
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
    u.fill(2.0f);
    v.fill(0.0f);
    w.fill(0.0f);
    qv.fill(0.01f);

    const int j = 3;
    const int k = 4;
    u[0][j][k] = 2.0f;
    u[1][j][k] = 8.0f;
    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    scheme->seed_acoustic_radiation();

    CartesianCGridAcousticEnergyReportV1 report;
    scheme->apply_acoustic_attributed(1.0, report);
    const double expected_velocity =
        2.0 - (2.0 - 30.0) * (8.0 - 2.0) / dr;
    const double scalar_mass =
        static_cast<double>(rho[1][j][k])
        * (1.0 + static_cast<double>(qv[1][j][k]))
        * dr * dtheta * dz;
    const double west_half_face_mass = 0.5 * scalar_mass;
    const double expected_work = 0.5 * west_half_face_mass
        * (expected_velocity * expected_velocity - 4.0);
    const double stored_velocity = static_cast<double>(u[0][j][k]);
    const double delta = stored_velocity - 2.0;
    const double expected_requested =
        west_half_face_mass * 2.0 * delta;
    const double expected_finite =
        0.5 * west_half_face_mass * delta * delta;
    const double expected_committed = 0.5 * west_half_face_mass
        * (stored_velocity * stored_velocity - 4.0);

    REQUIRE(u[0][j][k] == Approx(expected_velocity).epsilon(1.0e-6));
    REQUIRE(report.boundary_complete);
    REQUIRE(report.boundary_application_count == 1);
    REQUIRE(report.boundary_committed_work_j
            == Approx(expected_work).epsilon(2.0e-6));
    REQUIRE(report.boundary_radiative_requested_linear_work_j
            == Approx(expected_requested).epsilon(2.0e-12));
    REQUIRE(report.boundary_radiative_finite_step_correction_j
            == Approx(expected_finite).epsilon(2.0e-12));
    REQUIRE(report.boundary_radiative_committed_work_j
            == Approx(expected_committed).epsilon(2.0e-12));
    REQUIRE(report.boundary_radiative_committed_work_j
            == Approx(
                report.boundary_radiative_requested_linear_work_j
                + report.boundary_radiative_finite_step_correction_j
                + report.boundary_radiative_storage_rounding_j)
                   .epsilon(2.0e-12));
    REQUIRE(report.boundary_constraint_history_committed_work_j
            == Approx(0.0).margin(
                std::abs(expected_committed) * 2.0e-12 + 1.0e-9));
    REQUIRE(report.committed_inventory_change_j
            == Approx(expected_work).epsilon(2.0e-6));

    reset_bcs_to_default();
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
}

TEST_CASE("SCI-145 mapped boundary work closes the moist v2 kinetic inventory",
          "[boundary_conditions][cgrid][terrain][energy][SCI-145]")
{
    resize_and_seed_uniform();
    dtheta = 800.0;
    TerrainMetrics3D metrics = make_attribution_terrain(0.18, -0.11);
    TerrainBindingGuard terrain_binding(metrics);

    set_bcs(LateralBC::Rigid, LateralBC::RigidSlip,
            LateralBC::ZeroGradient, LateralBC::Rigid);
    global_bottom_bc = tmv::init::config::BottomBC::RigidNoSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                rho[i][j][k] =
                    static_cast<float>(1.0 + 0.002 * i + 0.003 * j);
                qv[i][j][k] =
                    static_cast<float>(0.006 + 0.0001 * k);
                qc[i][j][k] = static_cast<float>(0.00002 * (i + j));
                u[i][j][k] =
                    static_cast<float>(3.0 + 0.2 * i - 0.05 * k);
                v[i][j][k] =
                    static_cast<float>(-1.0 + 0.15 * j + 0.03 * k);
                w[i][j][k] =
                    static_cast<float>(0.1 * i - 0.08 * j + 0.02 * k);
            }

    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    const double before = mapped_v2_kinetic_energy(metrics);
    CartesianCGridAcousticEnergyReportV1 report;
    scheme->apply_acoustic_attributed(0.0, report);
    const double after = mapped_v2_kinetic_energy(metrics);
    const double expected = after - before;

    REQUIRE(CartesianCGridAcousticEnergyReportV1::basis_version == 2);
    REQUIRE(report.boundary_complete);
    REQUIRE(report.boundary_application_count == 1);
    REQUIRE(report.boundary_committed_work_j
            == Approx(expected).epsilon(2.0e-11).margin(1.0e-5));
    REQUIRE(report.committed_inventory_change_j
            == Approx(expected).epsilon(2.0e-11).margin(1.0e-5));
    REQUIRE(report.boundary_radiative_requested_linear_work_j == 0.0);
    REQUIRE(report.boundary_radiative_finite_step_correction_j == 0.0);
    REQUIRE(report.boundary_radiative_storage_rounding_j == 0.0);
    REQUIRE(report.boundary_radiative_committed_work_j == 0.0);
    REQUIRE(report.boundary_constraint_history_committed_work_j
            == Approx(expected).epsilon(2.0e-11).margin(1.0e-5));

    reset_bcs_to_default();
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
}

TEST_CASE("SCI-156 mapped final closure owns exact wall and lid work",
          "[boundary_conditions][cgrid][terrain][final][energy][SCI-156]")
{
    resize_and_seed_uniform();
    dtheta = 800.0;
    TerrainMetrics3D metrics = make_attribution_terrain(0.18, -0.11);
    TerrainBindingGuard terrain_binding(metrics);

    set_bcs(LateralBC::Rigid, LateralBC::RigidSlip,
            LateralBC::ZeroGradient, LateralBC::Rigid);
    global_bottom_bc = tmv::init::config::BottomBC::RigidNoSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                rho[i][j][k] =
                    static_cast<float>(1.0 + 0.002 * i + 0.003 * j);
                qv[i][j][k] =
                    static_cast<float>(0.006 + 0.0001 * k);
                qc[i][j][k] = static_cast<float>(0.00002 * (i + j));
                u[i][j][k] =
                    static_cast<float>(3.0 + 0.2 * i - 0.05 * k);
                v[i][j][k] =
                    static_cast<float>(-1.0 + 0.15 * j + 0.03 * k);
                w[i][j][k] =
                    static_cast<float>(0.1 * i - 0.08 * j + 0.02 * k);
            }

    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    const double before = mapped_v2_kinetic_energy(metrics);
    const tmv::FinalBoundaryEnergyReportV1 report =
        scheme->apply_full_attributed();
    const double after = mapped_v2_kinetic_energy(metrics);
    const double expected = after - before;

    REQUIRE(tmv::FinalBoundaryEnergyReportV1::energy_basis_version == 2);
    REQUIRE(report.applied);
    REQUIRE(report.complete);
    REQUIRE(report.constraint_committed_work_j
            == Approx(expected).epsilon(2.0e-11).margin(1.0e-5));
    REQUIRE(report.carrier_kinetic_change_j
            == Approx(expected).epsilon(2.0e-11).margin(1.0e-5));
    REQUIRE(report.committed_energy_change_j
            == Approx(expected).epsilon(2.0e-11).margin(1.0e-5));
    REQUIRE(report.dry_internal_change_j == 0.0);
    REQUIRE(report.water_sensible_change_j == 0.0);
    REQUIRE(report.phase_reference_change_j == 0.0);
    REQUIRE(report.gravitational_change_j == 0.0);
    REQUIRE(report.subgrid_tke_change_j == 0.0);
    REQUIRE(report.closure_residual_j == 0.0);

    reset_bcs_to_default();
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
}

TEST_CASE("SCI-145 mapped KW78 work includes diagnosed ground velocity",
          "[boundary_conditions][cgrid][terrain][energy][radiative][SCI-145]")
{
    resize_and_seed_uniform();
    dtheta = 800.0;
    constexpr double slope_x = 0.2;
    TerrainMetrics3D metrics =
        make_attribution_terrain(slope_x, -0.13);
    TerrainBindingGuard terrain_binding(metrics);

    set_bcs(LateralBC::Radiative, LateralBC::Radiative,
            LateralBC::Radiative, LateralBC::Radiative);
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Open;
    u.fill(2.0f);
    v.fill(0.0f);
    w.fill(0.0f);
    qv.fill(0.01f);
    const int j = 3;
    u[1][j][0] = 8.0f;

    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    scheme->seed_acoustic_radiation();
    const double before = mapped_v2_kinetic_energy(metrics);
    CartesianCGridAcousticEnergyReportV1 report;
    scheme->apply_acoustic_attributed(1.0, report);
    const double after = mapped_v2_kinetic_energy(metrics);

    const double stored_velocity = static_cast<double>(u[0][j][0]);
    const double delta_u = stored_velocity - 2.0;
    const double cell_mass =
        static_cast<double>(rho[1][j][0])
        * (1.0 + static_cast<double>(qv[1][j][0]))
        * dr * dtheta
        * (metrics.z_face(1, j, 1) - metrics.z_face(1, j, 0));
    const double native_mass = 0.5 * cell_mass;
    const double ground_old = 0.5 * (2.0 + 8.0) * slope_x;
    const double delta_ground = 0.5 * delta_u * slope_x;
    const double expected_requested =
        native_mass * 2.0 * delta_u
        + 0.5 * cell_mass * ground_old * delta_ground;
    const double expected_finite =
        0.5 * native_mass * delta_u * delta_u
        + 0.25 * cell_mass * delta_ground * delta_ground;
    const double expected_committed =
        0.5 * native_mass
            * (stored_velocity * stored_velocity - 4.0)
        + 0.25 * cell_mass
            * ((ground_old + delta_ground)
                   * (ground_old + delta_ground)
               - ground_old * ground_old);

    REQUIRE(report.boundary_committed_work_j
            == Approx(after - before).epsilon(2.0e-11).margin(1.0e-5));
    REQUIRE(report.boundary_radiative_requested_linear_work_j
            == Approx(expected_requested).epsilon(2.0e-11).margin(1.0e-5));
    REQUIRE(report.boundary_radiative_finite_step_correction_j
            == Approx(expected_finite).epsilon(2.0e-11).margin(1.0e-5));
    REQUIRE(report.boundary_radiative_committed_work_j
            == Approx(expected_committed).epsilon(2.0e-11).margin(1.0e-5));
    REQUIRE(report.boundary_radiative_committed_work_j
            == Approx(
                report.boundary_radiative_requested_linear_work_j
                + report.boundary_radiative_finite_step_correction_j
                + report.boundary_radiative_storage_rounding_j)
                   .epsilon(2.0e-12).margin(1.0e-5));
    REQUIRE(report.boundary_constraint_history_committed_work_j
            == Approx(0.0).margin(
                std::abs(expected_committed) * 2.0e-11 + 1.0e-5));

    reset_bcs_to_default();
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
}

TEST_CASE("SCI-017 collocated grid rejects radiative closure",
          "[boundary_conditions][lateral_bc][SCI-017]")
{
    resize_and_seed_uniform();
    set_bcs(LateralBC::Radiative, LateralBC::ZeroGradient,
            LateralBC::ZeroGradient, LateralBC::ZeroGradient);
    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    REQUIRE_THROWS_WITH(
        scheme->apply_full(),
        Catch::Matchers::Contains("require Cartesian C-grid"));
    reset_bcs_to_default();
}

TEST_CASE("SCI-017 radiative C-grid face releases an outgoing acoustic pulse",
          "[boundary_conditions][lateral_bc][cgrid][acoustic][SCI-017]")
{
    const LinearAcousticPulseBudget radiative =
        linear_acoustic_pulse_budget(LateralBC::Radiative);
    const LinearAcousticPulseBudget zero_gradient =
        linear_acoustic_pulse_budget(LateralBC::ZeroGradient);
    INFO("acoustic pulse residual energy: radiative="
         << radiative.residual_energy_fraction
         << " zero_gradient=" << zero_gradient.residual_energy_fraction
         << " radiative flux closure=" << radiative.flux_closure_residual);

    REQUIRE(radiative.residual_energy_fraction < 0.10);
    REQUIRE(zero_gradient.residual_energy_fraction > 0.25);
    REQUIRE(radiative.residual_energy_fraction
            < 0.25 * zero_gradient.residual_energy_fraction);
    REQUIRE(radiative.integrated_outward_energy > 0.0);
    REQUIRE(radiative.flux_closure_residual < 0.005);
    reset_bcs_to_default();
}

// ─────────────────────────────────────────────────────────────
// Defaults are preserved for v0 configs (no schema stamp)
// ─────────────────────────────────────────────────────────────

TEST_CASE("C1 globals default to ZeroGradient (v0 backward compat)",
          "[boundary_conditions][c1][lateral_bc][defaults]")
{
    // Smoke check: fresh process state. These globals are zero-initialized
    // to LateralBC::ZeroGradient in runtime_config.cpp; changing this
    // the default would break every v0 config silently.
    REQUIRE(global_lateral_bc_west  == LateralBC::ZeroGradient);
    REQUIRE(global_lateral_bc_east  == LateralBC::ZeroGradient);
    REQUIRE(global_lateral_bc_south == LateralBC::ZeroGradient);
    REQUIRE(global_lateral_bc_north == LateralBC::ZeroGradient);
}

TEST_CASE("Boundary-condition dispatch rejects unsupported grid axes",
          "[boundary_conditions][factory]")
{
    REQUIRE_THROWS_AS(
        create_boundary_condition_scheme(
            static_cast<CoordinateSystem>(99), StaggerType::Collocated),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        create_boundary_condition_scheme(
            CoordinateSystem::Cartesian, static_cast<StaggerType>(99)),
        std::invalid_argument);
}

TEST_CASE("C-grid boundary masses follow physical state across calls and thin grids",
          "[boundary_conditions][cgrid][energy][mass_reuse]")
{
    const auto kinetic=[](const SimulationStorageState& s) {
        double total=0.0;
        const double volume=s.grid.dx*s.grid.dy*s.grid.dz;
        for (int i=1;i<s.grid.nr-1;++i) for (int j=1;j<s.grid.nth-1;++j) for (int k=0;k<s.grid.nz;++k)
        {
            const auto& f=s.fields;
            const double water=double(f.qv(i,j,k))+f.qc(i,j,k)+f.qr(i,j,k)+f.qi(i,j,k)+f.qs(i,j,k)+f.qg(i,j,k)+f.qh(i,j,k);
            const double mass=double(f.rho(i,j,k))*(1.0+water)*volume;
            const double uw=f.u(i-1,j,k),ue=f.u(i,j,k),vs=f.v(i,j-1,k),vn=f.v(i,j,k);
            const double wb=k==0?0.0:double(f.w(i,j,k-1)),wt=f.w(i,j,k);
            total+=0.25*mass*(uw*uw+ue*ue+vs*vs+vn*vn+wb*wb+wt*wt);
        }
        return total;
    };
    for (const auto shape:{std::array<int,3>{3,3,2},{4,4,33},{11,7,9},{31,29,5},{31,29,2}})
        for (auto lateral:{LateralBC::Radiative,LateralBC::Periodic,LateralBC::RigidSlip,LateralBC::ZeroGradient})
            for (auto top:{tmv::init::config::TopBC::Open,tmv::init::config::TopBC::Rigid})
            {
                SimulationStorageState state(GridStorageState(shape[0],shape[1],shape[2],50.0,100.0,0.1,75.0));
                state.configuration.lateral_bc_west=state.configuration.lateral_bc_east=lateral;
                state.configuration.lateral_bc_south=state.configuration.lateral_bc_north=lateral;
                state.configuration.bottom_bc=tmv::init::config::BottomBC::RigidNoSlip;
                state.configuration.top_bc=top;
                auto& f=state.fields;
                for (auto* field:{&f.rho,&f.p,&f.u,&f.v,&f.w,&f.qv,&f.qc,&f.qr,&f.qi,&f.qs,&f.qg,&f.qh})
                    field->resize(shape[0],shape[1],shape[2],0.0f);
                auto scheme=tmv::create_cartesian_cgrid_bc_scheme(make_simulation_state_view(state));
                for (int phase=0;phase<3;++phase)
                {
                    for (int i=0;i<shape[0];++i) for (int j=0;j<shape[1];++j) for (int k=0;k<shape[2];++k)
                    {
                        f.rho(i,j,k)=float(0.8+0.01*i+0.015*j+0.003*k+0.2*phase);
                        f.p(i,j,k)=90000.0f;
                        f.u(i,j,k)=float(1.3+0.2*i+0.13*j+0.3*k);
                        f.v(i,j,k)=float(-0.7+0.11*i-0.09*j+0.21*k);
                        f.w(i,j,k)=float(0.15*i-0.07*j+0.05*k);
                        for (auto* water:{&f.qv,&f.qc,&f.qr,&f.qi,&f.qs,&f.qg,&f.qh})
                            (*water)(i,j,k)=float(0.0003*(1+phase)+0.00001*(i+3*j+7*k));
                    }
                    scheme->seed_acoustic_radiation();
                    const double before=kinetic(state);
                    CartesianCGridAcousticEnergyReportV1 report;
                    scheme->apply_acoustic_attributed(0.025,report);
                    const double after=kinetic(state);
                    CHECK(report.boundary_complete);
                    CHECK(report.boundary_application_count==1);
                    CHECK(report.boundary_committed_work_j==Approx(after-before).epsilon(2e-11).margin(1e-6));
                    CHECK(report.committed_inventory_change_j==report.boundary_committed_work_j);
                    const double final_before=kinetic(state);
                    const auto final=scheme->apply_full_attributed();
                    CHECK(final.complete);
                    CHECK(final.committed_energy_change_j==Approx(kinetic(state)-final_before).epsilon(2e-11).margin(1e-6));
                }
                const auto before=f.u;
                f.qh(1,1,0)=-0.001f;
                CartesianCGridAcousticEnergyReportV1 rejected;
                REQUIRE_THROWS_AS(scheme->apply_acoustic_attributed(0.025,rejected),std::invalid_argument);
                CHECK(rejected.boundary_application_count==0);
                CHECK(std::equal(f.u.data(),f.u.data()+f.u.size(),before.data()));
            }
}
