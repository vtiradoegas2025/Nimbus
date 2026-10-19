#include "dynamics/radiative_face.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_grid_metrics.hpp"
/**
 * @file boundary_conditions_cartesian_cgrid.cpp
 * @brief Arakawa C-grid (staggered) Cartesian boundary conditions.
 *
 * Companion to CartesianCGrid3DScheme. The existing collocated Cartesian BC
 * (boundary_conditions_cartesian.cpp) moves u/v/w in unison at cell centers and
 * cannot be reused for staggered faces, so this is a staggered-aware module.
 *
 * Field placement (matches StaggeredCartesianDerivatives / the dynamics scheme):
 *   u (u_x) at x-face   u[i][j][k] at (x_{i+1/2}, y_j,       z_k)
 *   v (u_y) at y-face   v[i][j][k] at (x_i,       y_{j+1/2}, z_k)
 *   w       at z-face   w[i][j][k] at (x_i,       y_j,       z_{k+1/2})
 *   scalars cell center rho/p/...  at (x_i,       y_j,       z_k)
 *
 * Convention. The dynamics scheme computes interior tendencies on
 * i = 1..NX-2, j = 1..NY-2 and leaves the boundary columns (i in {0, NX-1},
 * j in {0, NY-1}) to this module -- a one-cell ghost layer on each side. The
 * rigid-wall face is the one between the last interior cell and the ghost cell:
 *   west wall  = u[0]      east wall  = u[NX-2]
 *   south wall = v[i][0]   north wall = v[i][NY-2]
 *
 * Lateral closures, per the v1 boundary_conditions.lateral.{west,east,south,north}
 * (global_lateral_bc_*):
 *   Radiative -- Klemp-Wilhelmson (1978) radiative: the boundary-normal
 *                velocity face is owned by the radiation equation
 *                du_b/dt = -(u_b +/- c*) du/dn (c* = 30 m/s, the deep-
 *                convection gravity-wave speed), integrated once per
 *                acoustic substep, held on inflow; ghosts, tangential
 *                velocities and rho/p stay zero-gradient
 *                (SCIENCE.md M17 - plain zero-gradient on the
 *                normal face is reflective and unstable without a
 *                strong lateral sponge).
 *   ZeroGradient -- normal, tangential, and scalar values copy the adjacent
 *                interior value without time-dependent radiation.
 *   Rigid     -- no-slip wall: Earth-relative horizontal wind equals the
 *                translating lateral-wall velocity; vertical wind is zero.
 *   RigidSlip -- free-slip wall: grid-relative normal velocity is zero and
 *                tangential velocity is zero-gradient.
 *   Periodic  -- wrap the one-cell ghost from the opposite interior.
 * rho/p take the same zero-gradient / wrap closure so the i=1 / j=1 stencils
 * have a defined neighbour. Advected scalars normally remain owned by their
 * transport kernel; the domain-recycling rebuild closes their newly replaced
 * ghosts once before the next reconstruction stencil.
 *
 * Vertical closure (ported from CartesianPatchScheme::apply_vertical_bcs):
 *   bottom k=0: RigidNoSlip -> u=v=0; RigidSlip leaves u,v prognostic with an
 *     implicit mirror. The ground face has zero contravariant flux; on terrain
 *     its physical w is u*dh/dx+v*dh/dy, not zero. w[0] is interior.
 *   top k=NZ-1: Open -> u,v zero-gradient; Rigid/RigidWithSponge -> w=0,
 *     u,v zero-gradient.
 *   p and rho are left PROGNOSTIC at the surface and lid. Overwriting them
 *     (zero-gradient rho or a perturbation-pressure Neumann copy) zeroes
 *     dp'/dz at the boundary-adjacent w-face and lets the surface
 *     buoyancy-continuity feedback grow unchecked -- documented in the patch
 *     scheme + project_patch_p_prime_bc_fix. Leaving them prognostic is
 *     exactly balanced at IC (p=p0, rho=rho0 -> zero tendency).
 */

#include "boundary_conditions/boundary_conditions_base.hpp"
#include "boundary_conditions/factory.hpp"
#include "boundary_conditions/runtime_state.hpp"
#include "core/runtime/process_state.hpp"
#include "dynamics/acoustic_energy_report.hpp"
#include "dynamics/runtime.hpp"
#include "numerics/runtime.hpp"
#include "terrain/terrain_base.hpp"
#include <algorithm>
#include <vector>       // u, v, w, rho, p, theta, qv..qh, NR, NTH, NZ
#include "core/runtime/runtime_config.hpp"   // global_lateral_bc_*, global_bottom_bc, global_top_bc
#include "init/config/config_v1.hpp"         // LateralBC, BottomBC, TopBC

#include <memory>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

using tmv::init::config::BottomBC;
using tmv::init::config::LateralBC;
using tmv::init::config::TopBC;

namespace tmv
{

// For a Cartesian grid the global NR/NTH/NZ are NX/NY/NZ. GridGeometry maps
// r/theta/z to scalar centers and the corresponding *_faces arrays to physical
// faces. Aliased locally for readability.

class CartesianCGridBCScheme : public BoundaryConditionScheme
{
    SimulationStateView state_;
public:
    explicit CartesianCGridBCScheme(const SimulationStateView& state) : state_(state) {}

    // theta + moisture are owned by the scalar-advection kernel and are
    // never touched here. The acoustic call advances Radiative
    // boundary-normal faces by dt_small (KW78); the full call only
    // refreshes ghosts/tangentials around the radiation-owned faces.
    void apply_full() override
    {
        require_explicit_lateral_bcs();
        apply_boundaries_attributed_to_momentum(0.0);
    }
    void apply_rk_stage() override
    {
        apply_full();
        rebuild_advected_scalar_ghosts();
    }
    FinalBoundaryEnergyReportV1 apply_full_attributed() override
    {
        require_explicit_lateral_bcs();
        // Boundary closure changes velocities and scalar ghosts, never these physical-cell masses.
        BoundaryMassCache masses(state_);
        const double before = boundary_owned_kinetic_energy_j(&masses);
        apply_boundaries_attributed_to_momentum(0.0);
        const double after = boundary_owned_kinetic_energy_j(&masses);
        const double committed = after - before;
        if (!std::isfinite(committed))
        {
            throw std::runtime_error(
                "C-grid final boundary closure produced non-finite energy work");
        }

        FinalBoundaryEnergyReportV1 report;
        report.constraint_committed_work_j = committed;
        report.carrier_kinetic_change_j = committed;
        report.committed_energy_change_j = committed;
        report.applied = true;
        report.complete = true;
        return report;
    }
    BoundaryFieldWriteSet full_attributed_write_set() const noexcept override
    {
        static const char* const field_ids[] = {
            "u", "v", "w", "rho", "p"};
        return {field_ids, sizeof(field_ids) / sizeof(field_ids[0]), true};
    }
    void apply_acoustic(double dt_small) override
    {
        require_explicit_lateral_bcs();
        apply_boundaries_attributed_to_momentum(dt_small);
    }
    void apply_acoustic_attributed(
        double dt_small,
        CartesianCGridAcousticEnergyReportV1& report) override
    {
        if (!std::isfinite(dt_small) || dt_small < 0.0)
        {
            throw std::invalid_argument(
                "attributed C-grid acoustic boundary closure requires finite dt >= 0");
        }
        const double prior_values[] = {
            report.boundary_committed_work_j,
            report.boundary_radiative_requested_linear_work_j,
            report.boundary_radiative_finite_step_correction_j,
            report.boundary_radiative_storage_rounding_j,
            report.boundary_radiative_committed_work_j,
            report.boundary_constraint_history_committed_work_j,
            report.committed_inventory_change_j,
        };
        for (double value : prior_values)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument(
                    "attributed C-grid acoustic boundary closure received an invalid report");
            }
        }
        if (report.boundary_application_count > 0
            && !report.boundary_complete)
        {
            throw std::invalid_argument(
                "attributed C-grid acoustic boundary closure received a partial report");
        }

        require_explicit_lateral_bcs();
        // Boundary closure changes velocities and scalar ghosts, never these physical-cell masses.
        BoundaryMassCache masses(state_);
        const double before = boundary_owned_kinetic_energy_j(&masses);
        ensure_radiative_state(dt_small);
        const RadiativeEnergyTerms radiative =
            radiative_energy_terms(dt_small, &masses);
        apply_boundaries_attributed_to_momentum(dt_small);
        const double after = boundary_owned_kinetic_energy_j(&masses);
        const double committed = after - before;
        if (!std::isfinite(committed))
        {
            throw std::runtime_error(
                "C-grid acoustic boundary closure produced non-finite energy work");
        }

        CartesianCGridAcousticEnergyReportV1 next = report;
        next.boundary_committed_work_j += committed;
        next.boundary_radiative_requested_linear_work_j +=
            radiative.requested_linear_work_j;
        next.boundary_radiative_finite_step_correction_j +=
            radiative.finite_step_correction_j;
        next.boundary_radiative_storage_rounding_j +=
            radiative.storage_rounding_j;
        next.boundary_radiative_committed_work_j +=
            radiative.committed_work_j;
        next.boundary_constraint_history_committed_work_j +=
            committed - radiative.committed_work_j;
        next.committed_inventory_change_j += committed;
        ++next.boundary_application_count;
        next.boundary_complete = true;
        const double next_values[] = {
            next.boundary_committed_work_j,
            next.boundary_radiative_requested_linear_work_j,
            next.boundary_radiative_finite_step_correction_j,
            next.boundary_radiative_storage_rounding_j,
            next.boundary_radiative_committed_work_j,
            next.boundary_constraint_history_committed_work_j,
            next.committed_inventory_change_j,
        };
        if (std::any_of(
                std::begin(next_values), std::end(next_values),
                [](double value) { return !std::isfinite(value); }))
        {
            throw std::overflow_error(
                "C-grid acoustic boundary energy attribution overflowed");
        }
        report = next;
    }

    // Seed the KW78 radiative-face stash from the live (post-slow-tendency) faces
    // at the START of the acoustic loop. The split-explicit driver calls this
    // after apply_slow_tendencies and before the first acoustic substep, so the
    // radiation's "previous face value" includes this step's slow (advection +
    // Coriolis) update at the boundary - matching the GPU batched path, which
    // seeds its in-shader state from the fields uploaded at batch start (also
    // post-slow). Seeding at the end of the previous step instead left the stash
    // one large-step of slow forcing stale and diverged from the GPU on open
    // boundaries with strong mean flow (the w parity breach at the open corner).
    void seed_acoustic_radiation() override
    {
        seed_radiative_state();
    }

    void rebuild_after_domain_recycling() override
    {
        require_explicit_lateral_bcs();
        apply_boundaries(0.0);
        rebuild_advected_scalar_ghosts();
        seed_radiative_state();
    }

    std::string get_scheme_name() const override { return "cartesian_cgrid"; }

private:
    void apply_boundaries_attributed_to_momentum(double dt_small)
    {
        auto& runtime = (*state_.dynamics_runtime);
        if (!runtime.momentum_interface_flux_capture_enabled)
        {
            apply_boundaries(dt_small);
            return;
        }
        auto& fluxes = runtime.momentum_interface_flux_register;
        const auto before = fluxes.native_dual_inventory(
            state_.fields.rho, state_.fields.u, state_.fields.v, state_.fields.w, state_.grid.dr, state_.grid.dtheta, state_.grid.dz);
        apply_boundaries(dt_small);
        const auto after = fluxes.native_dual_inventory(
            state_.fields.rho, state_.fields.u, state_.fields.v, state_.fields.w, state_.grid.dr, state_.grid.dtheta, state_.grid.dz);
        std::array<double, 3> change{};
        for (std::size_t component = 0; component < change.size(); ++component)
            change[component] = after[component] - before[component];
        fluxes.accumulate_operator_source_impulse(change);
    }

    void require_explicit_lateral_bcs() const
    {
        const LateralBC faces[] = {
            state_.configuration->lateral_bc_west, state_.configuration->lateral_bc_east,
            state_.configuration->lateral_bc_south, state_.configuration->lateral_bc_north
        };
        for (const LateralBC bc : faces)
        {
            if (bc == LateralBC::LegacyOpen)
            {
                throw std::runtime_error(
                    "ambiguous legacy 'open' lateral boundary reached the "
                    "C-grid runtime without configuration projection");
            }
        }
    }

    // KW78 deep-convection gravity-wave phase speed (m/s); CM1 uses the
    // same fixed value for its radiative lateral BCs.
    static constexpr double kCStar = 30.0;

    struct RadiativeEnergyTerms
    {
        double requested_linear_work_j = 0.0;
        double finite_step_correction_j = 0.0;
        double storage_rounding_j = 0.0;
        double committed_work_j = 0.0;
    };

    static void accumulate_radiative_energy_terms(
        RadiativeEnergyTerms& terms, double mass,
        double previous, double committed)
    {
        const double old_velocity = previous;
        const double stored_velocity = committed;
        const double delta = stored_velocity - old_velocity;
        const double requested = mass * old_velocity * delta;
        const double finite = 0.5 * mass * delta * delta;
        const double exact_committed = 0.5 * mass
            * (stored_velocity * stored_velocity
               - old_velocity * old_velocity);
        terms.requested_linear_work_j += requested;
        terms.finite_step_correction_j += finite;
        terms.storage_rounding_j +=
            exact_committed - requested - finite;
        terms.committed_work_j += exact_committed;
    }

    struct BoundaryMassCache
    {
        int nx = 0, ny = 0, nz = 0;
        std::size_t plane = 0, side_x = 0, side_y = 0;
        std::vector<double> values;

        explicit BoundaryMassCache(const SimulationStateView& state)
            : nx(state.grid.nr), ny(state.grid.nth), nz(state.grid.nz)
        {
            if (nx < 3 || ny < 3 || nz < 2) return;
            const auto x = static_cast<std::size_t>(nx - 2);
            const auto y = static_cast<std::size_t>(ny - 2);
            const auto z = static_cast<std::size_t>(nz - 2);
            const auto limit = state.fields.rho.size() / 2;
            if (x > limit / y / 2) return;
            plane = x * y;
            const auto rest = limit - 2 * plane;
            if (z != 0 && (x > rest / z / 2 || y > rest / z / 2 - x)) return;
            side_x = y * z;
            side_y = x * z;
            values.assign(2 * (plane + side_x + side_y), std::numeric_limits<double>::quiet_NaN());
        }

        double* find(int i, int j, int k)
        {
            if (values.empty() || i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1 || k < 0 || k >= nz)
                return nullptr;
            const auto x = static_cast<std::size_t>(i - 1);
            const auto y = static_cast<std::size_t>(j - 1);
            const auto levels = static_cast<std::size_t>(nz - 2);
            if (k == 0) return &values[x * static_cast<std::size_t>(ny - 2) + y];
            if (k == nz - 1) return &values[plane + x * static_cast<std::size_t>(ny - 2) + y];
            const auto z = static_cast<std::size_t>(k - 1);
            if (i == 1) return &values[2 * plane + y * levels + z];
            if (i == nx - 2) return &values[2 * plane + side_x + y * levels + z];
            if (j == 1) return &values[2 * plane + 2 * side_x + x * levels + z];
            if (j == ny - 2) return &values[2 * plane + 2 * side_x + side_y + x * levels + z];
            return nullptr;
        }
    };

    double scalar_mixture_mass_kg(int i, int j, int k, BoundaryMassCache* cache = nullptr) const
    {
        double* cached = cache != nullptr ? cache->find(i, j, k) : nullptr;
        if (cached != nullptr && !std::isnan(*cached)) return *cached;
        const double rho_d = static_cast<double>(state_.fields.rho[i][j][k]);
        const double values[] = {
            rho_d,
            static_cast<double>(state_.fields.qv[i][j][k]),
            static_cast<double>(state_.fields.qc[i][j][k]),
            static_cast<double>(state_.fields.qr[i][j][k]),
            static_cast<double>(state_.fields.qi[i][j][k]),
            static_cast<double>(state_.fields.qs[i][j][k]),
            static_cast<double>(state_.fields.qg[i][j][k]),
            static_cast<double>(state_.fields.qh[i][j][k]),
        };
        for (double value : values)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument(
                    "C-grid acoustic boundary energy attribution rejected non-finite moist state");
            }
        }
        if (rho_d <= 0.0)
        {
            throw std::invalid_argument(
                "C-grid acoustic boundary energy attribution requires positive dry density");
        }
        for (std::size_t n = 1; n < std::size(values); ++n)
        {
            if (values[n] < 0.0)
            {
                throw std::invalid_argument(
                    "C-grid acoustic boundary energy attribution requires nonnegative water");
            }
        }
        double layer_thickness = state_.grid.dz;
        if (tmv::physical_grid_metrics::terrain_is_active(state_))
        {
            const TerrainMetrics3D* metrics =
                tmv::physical_grid_metrics::active_terrain_metrics(state_);
            if (metrics == nullptr
                || metrics->z_face.size_r() != state_.grid.nr
                || metrics->z_face.size_th() != state_.grid.nth
                || metrics->z_face.size_z() != state_.grid.nz + 1)
            {
                throw std::invalid_argument(
                    "C-grid acoustic boundary energy attribution requires complete mapped geometry");
            }
            layer_thickness =
                metrics->z_face(i, j, k + 1)
                - metrics->z_face(i, j, k);
        }
        const double volume = state_.grid.dr * state_.grid.dtheta * layer_thickness;
        if (!std::isfinite(volume) || volume <= 0.0)
        {
            throw std::invalid_argument(
                "C-grid acoustic boundary energy attribution requires positive finite cell volume");
        }
        const double q_total = values[1] + values[2] + values[3]
            + values[4] + values[5] + values[6] + values[7];
        const double mass = rho_d * (1.0 + q_total) * volume;
        if (!std::isfinite(mass) || mass <= 0.0)
        {
            throw std::invalid_argument(
                "C-grid acoustic boundary energy attribution produced invalid mixture mass");
        }
        if (cached != nullptr) *cached = mass;
        return mass;
    }

    const TerrainMetrics3D* attributed_terrain_metrics() const
    {
        if (!tmv::physical_grid_metrics::terrain_is_active(state_))
            return nullptr;
        const TerrainMetrics3D* metrics =
            tmv::physical_grid_metrics::active_terrain_metrics(state_);
        if (metrics == nullptr
            || metrics->z_face.size_r() != state_.grid.nr
            || metrics->z_face.size_th() != state_.grid.nth
            || metrics->z_face.size_z() != state_.grid.nz + 1
            || metrics->Gx_face.size_r() != state_.grid.nr
            || metrics->Gx_face.size_th() != state_.grid.nth
            || metrics->Gx_face.size_z() != state_.grid.nz + 1
            || metrics->Gy_face.size_r() != state_.grid.nr
            || metrics->Gy_face.size_th() != state_.grid.nth
            || metrics->Gy_face.size_z() != state_.grid.nz + 1)
        {
            throw std::invalid_argument(
                "C-grid acoustic boundary energy attribution requires complete mapped surface metrics");
        }
        return metrics;
    }

    double ground_physical_velocity(
        const TerrainMetrics3D& metrics, int i, int j,
        double u_west, double u_east,
        double v_south, double v_north) const
    {
        const double frame_u = tmv::physical_grid_metrics::surface_velocity_ms(state_).u_ms;
        const double frame_v = tmv::physical_grid_metrics::surface_velocity_ms(state_).v_ms;
        const double values[] = {
            metrics.Gx_face(i, j, 0),
            metrics.Gy_face(i, j, 0),
            u_west, u_east, v_south, v_north, frame_u, frame_v,
        };
        if (std::any_of(
                std::begin(values), std::end(values),
                [](double value) { return !std::isfinite(value); }))
        {
            throw std::invalid_argument(
                "C-grid acoustic boundary energy attribution rejected invalid mapped ground state");
        }
        return -(0.5 * (u_west + u_east) - frame_u)
                * metrics.Gx_face(i, j, 0)
            - (0.5 * (v_south + v_north) - frame_v)
                * metrics.Gy_face(i, j, 0);
    }

    double boundary_owned_kinetic_energy_j(BoundaryMassCache* cache = nullptr) const
    {
        const int nx = state_.grid.nr;
        const int ny = state_.grid.nth;
        const int nz = state_.grid.nz;
        if (nx < 3 || ny < 3 || nz < 2)
        {
            throw std::invalid_argument(
                "C-grid acoustic boundary energy attribution requires a complete grid");
        }

        double energy = 0.0;
        const int k_top = nz - 1;
        const auto accumulate_u_face =
            [this, &energy, nx, cache](int i_face, int j, int k)
        {
                    double mass = 0.0;
                    if (i_face >= 1)
                        mass += 0.5 * scalar_mixture_mass_kg(i_face, j, k, cache);
                    if (i_face + 1 < nx - 1)
                        mass += 0.5 * scalar_mixture_mass_kg(i_face + 1, j, k, cache);
                    const double velocity =
                        static_cast<double>(state_.fields.u[i_face][j][k]);
                    if (!std::isfinite(velocity))
                        throw std::invalid_argument(
                            "C-grid acoustic boundary energy attribution rejected non-finite u");
                    energy += 0.5 * mass * velocity * velocity;
        };
        for (int j = 1; j < ny - 1; ++j)
            for (int i_face = 0; i_face < nx - 1; ++i_face)
            {
                if (i_face == 0 || i_face == nx - 2)
                {
                    for (int k = 0; k < nz; ++k)
                        accumulate_u_face(i_face, j, k);
                }
                else
                {
                    accumulate_u_face(i_face, j, 0);
                    accumulate_u_face(i_face, j, k_top);
                }
            }

        const auto accumulate_v_face =
            [this, &energy, ny, cache](int i, int j_face, int k)
        {
                    double mass = 0.0;
                    if (j_face >= 1)
                        mass += 0.5 * scalar_mixture_mass_kg(i, j_face, k, cache);
                    if (j_face + 1 < ny - 1)
                        mass += 0.5 * scalar_mixture_mass_kg(i, j_face + 1, k, cache);
                    const double velocity =
                        static_cast<double>(state_.fields.v[i][j_face][k]);
                    if (!std::isfinite(velocity))
                        throw std::invalid_argument(
                            "C-grid acoustic boundary energy attribution rejected non-finite v");
                    energy += 0.5 * mass * velocity * velocity;
        };
        for (int i = 1; i < nx - 1; ++i)
            for (int j_face = 0; j_face < ny - 1; ++j_face)
            {
                if (j_face == 0 || j_face == ny - 2)
                {
                    for (int k = 0; k < nz; ++k)
                        accumulate_v_face(i, j_face, k);
                }
                else
                {
                    accumulate_v_face(i, j_face, 0);
                    accumulate_v_face(i, j_face, k_top);
                }
            }

        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
            {
                const double mass = 0.5
                    * scalar_mixture_mass_kg(i, j, k_top, cache);
                const double velocity =
                    static_cast<double>(state_.fields.w[i][j][k_top]);
                if (!std::isfinite(velocity))
                    throw std::invalid_argument(
                        "C-grid acoustic boundary energy attribution rejected non-finite w");
                energy += 0.5 * mass * velocity * velocity;
            }
        if (const TerrainMetrics3D* metrics =
                attributed_terrain_metrics())
        {
            for (int i = 1; i < nx - 1; ++i)
                for (int j = 1; j < ny - 1; ++j)
                {
                    const double ground_w = ground_physical_velocity(
                        *metrics, i, j,
                        static_cast<double>(state_.fields.u[i - 1][j][0]),
                        static_cast<double>(state_.fields.u[i][j][0]),
                        static_cast<double>(state_.fields.v[i][j - 1][0]),
                        static_cast<double>(state_.fields.v[i][j][0]));
                    energy += 0.25
                        * scalar_mixture_mass_kg(i, j, 0, cache)
                        * ground_w * ground_w;
                }
        }
        if (!std::isfinite(energy))
        {
            throw std::overflow_error(
                "C-grid acoustic boundary energy inventory overflowed");
        }
        return energy;
    }

    void ensure_radiative_state(double dt_rad)
    {
        if (dt_rad <= 0.0)
            return;
        const auto& state = (*state_.boundary_condition_runtime);
        const std::size_t yz = static_cast<std::size_t>(state_.grid.nth)
            * static_cast<std::size_t>(state_.grid.nz);
        const std::size_t xz = static_cast<std::size_t>(state_.grid.nr)
            * static_cast<std::size_t>(state_.grid.nz);
        if (state.radiative_u_west.size() != yz ||
            state.radiative_u_east.size() != yz ||
            state.radiative_v_south.size() != xz ||
            state.radiative_v_north.size() != xz)
        {
            seed_radiative_state();
        }
    }

    RadiativeEnergyTerms radiative_energy_terms(double dt_rad, BoundaryMassCache* cache = nullptr) const
    {
        const auto& state = (*state_.boundary_condition_runtime);
        RadiativeEnergyTerms terms;
        if (dt_rad <= 0.0)
            return terms;

        const int nx = state_.grid.nr;
        const int ny = state_.grid.nth;
        const int nz = state_.grid.nz;
        const float dt_rad_f = static_cast<float>(dt_rad);
        const float inv_dx_f =
            (state_.grid.dr > 0.0) ? 1.0f / static_cast<float>(state_.grid.dr) : 0.0f;
        const float inv_dy_f =
            (state_.grid.dtheta > 0.0) ? 1.0f / static_cast<float>(state_.grid.dtheta) : 0.0f;
        const float c_star_f = static_cast<float>(kCStar);
        const float frame_u_f = static_cast<float>(
            (*state_.domain_frame).velocity_ms.u_ms);
        const float frame_v_f = static_cast<float>(
            (*state_.domain_frame).velocity_ms.v_ms);

        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t n =
                    static_cast<std::size_t>(j) * nz + k;
                if (state_.configuration->lateral_bc_west == LateralBC::Radiative)
                {
                    const float previous = state.radiative_u_west[n];
                    const float committed = kw78_updated_face(
                        previous, previous - frame_u_f - c_star_f,
                        state_.fields.u[1][j][k] - previous,
                        dt_rad_f, inv_dx_f, true);
                    accumulate_radiative_energy_terms(
                        terms,
                        0.5 * scalar_mixture_mass_kg(1, j, k, cache),
                        previous, committed);
                }
                if (state_.configuration->lateral_bc_east == LateralBC::Radiative)
                {
                    const int face = nx - 2;
                    const float previous = state.radiative_u_east[n];
                    const float committed = kw78_updated_face(
                        previous, previous - frame_u_f + c_star_f,
                        previous - state_.fields.u[face - 1][j][k],
                        dt_rad_f, inv_dx_f, false);
                    accumulate_radiative_energy_terms(
                        terms,
                        0.5 * scalar_mixture_mass_kg(face, j, k, cache),
                        previous, committed);
                }
            }

        for (int i = 1; i < nx - 1; ++i)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t n =
                    static_cast<std::size_t>(i) * nz + k;
                if (state_.configuration->lateral_bc_south == LateralBC::Radiative)
                {
                    const float previous = state.radiative_v_south[n];
                    const float committed = kw78_updated_face(
                        previous, previous - frame_v_f - c_star_f,
                        state_.fields.v[i][1][k] - previous,
                        dt_rad_f, inv_dy_f, true);
                    accumulate_radiative_energy_terms(
                        terms,
                        0.5 * scalar_mixture_mass_kg(i, 1, k, cache),
                        previous, committed);
                }
                if (state_.configuration->lateral_bc_north == LateralBC::Radiative)
                {
                    const int face = ny - 2;
                    const float previous = state.radiative_v_north[n];
                    const float committed = kw78_updated_face(
                        previous, previous - frame_v_f + c_star_f,
                        previous - state_.fields.v[i][face - 1][k],
                        dt_rad_f, inv_dy_f, false);
                    accumulate_radiative_energy_terms(
                        terms,
                        0.5 * scalar_mixture_mass_kg(i, face, k, cache),
                        previous, committed);
                }
            }

        // In terrain coordinates a zero-contravariant-flux ground has the
        // physical velocity w_g = u dh/dx + v dh/dy. A radiative change to a
        // bottom lateral-normal face therefore changes the same basis-v2
        // ground kinetic reservoir even though no stored w face is mutated.
        // Evaluate all four lateral updates together so the corner cross term
        // is assigned exactly once.
        if (const TerrainMetrics3D* metrics =
                attributed_terrain_metrics())
        {
            for (int i = 1; i < nx - 1; ++i)
                for (int j = 1; j < ny - 1; ++j)
                {
                    double u_w_old = static_cast<double>(state_.fields.u[i - 1][j][0]);
                    double u_w_new = u_w_old;
                    double u_e_old = static_cast<double>(state_.fields.u[i][j][0]);
                    double u_e_new = u_e_old;
                    double v_s_old = static_cast<double>(state_.fields.v[i][j - 1][0]);
                    double v_s_new = v_s_old;
                    double v_n_old = static_cast<double>(state_.fields.v[i][j][0]);
                    double v_n_new = v_n_old;

                    if (i == 1
                        && state_.configuration->lateral_bc_west == LateralBC::Radiative)
                    {
                        const std::size_t n =
                            static_cast<std::size_t>(j) * nz;
                        const float previous = state.radiative_u_west[n];
                        const float committed = kw78_updated_face(
                            previous, previous - frame_u_f - c_star_f,
                            state_.fields.u[1][j][0] - previous,
                            dt_rad_f, inv_dx_f, true);
                        u_w_old = static_cast<double>(previous);
                        u_w_new = static_cast<double>(committed);
                    }
                    if (i == nx - 2
                        && state_.configuration->lateral_bc_east == LateralBC::Radiative)
                    {
                        const std::size_t n =
                            static_cast<std::size_t>(j) * nz;
                        const float previous = state.radiative_u_east[n];
                        const float committed = kw78_updated_face(
                            previous, previous - frame_u_f + c_star_f,
                            previous - state_.fields.u[nx - 3][j][0],
                            dt_rad_f, inv_dx_f, false);
                        u_e_old = static_cast<double>(previous);
                        u_e_new = static_cast<double>(committed);
                    }
                    if (j == 1
                        && state_.configuration->lateral_bc_south == LateralBC::Radiative)
                    {
                        const std::size_t n =
                            static_cast<std::size_t>(i) * nz;
                        const float previous = state.radiative_v_south[n];
                        const float committed = kw78_updated_face(
                            previous, previous - frame_v_f - c_star_f,
                            state_.fields.v[i][1][0] - previous,
                            dt_rad_f, inv_dy_f, true);
                        v_s_old = static_cast<double>(previous);
                        v_s_new = static_cast<double>(committed);
                    }
                    if (j == ny - 2
                        && state_.configuration->lateral_bc_north == LateralBC::Radiative)
                    {
                        const std::size_t n =
                            static_cast<std::size_t>(i) * nz;
                        const float previous = state.radiative_v_north[n];
                        const float committed = kw78_updated_face(
                            previous, previous - frame_v_f + c_star_f,
                            previous - state_.fields.v[i][ny - 3][0],
                            dt_rad_f, inv_dy_f, false);
                        v_n_old = static_cast<double>(previous);
                        v_n_new = static_cast<double>(committed);
                    }

                    const double ground_old = ground_physical_velocity(
                        *metrics, i, j,
                        u_w_old, u_e_old, v_s_old, v_n_old);
                    const double ground_new = ground_physical_velocity(
                        *metrics, i, j,
                        u_w_new, u_e_new, v_s_new, v_n_new);
                    accumulate_radiative_energy_terms(
                        terms,
                        0.5 * scalar_mixture_mass_kg(i, j, 0, cache),
                        ground_old, ground_new);
                }
        }

        const double values[] = {
            terms.requested_linear_work_j,
            terms.finite_step_correction_j,
            terms.storage_rounding_j,
            terms.committed_work_j,
        };
        if (std::any_of(
                std::begin(values), std::end(values),
                [](double value) { return !std::isfinite(value); }))
        {
            throw std::overflow_error(
                "C-grid KW78 acoustic-boundary energy attribution overflowed");
        }
        return terms;
    }

    void seed_radiative_state()
    {
        auto& state = (*state_.boundary_condition_runtime);
        const int NX = state_.grid.nr, NY = state_.grid.nth;
        const size_t yz = static_cast<size_t>(NY) * static_cast<size_t>(state_.grid.nz);
        const size_t xz = static_cast<size_t>(NX) * static_cast<size_t>(state_.grid.nz);
        state.radiative_u_west.resize(yz);
        state.radiative_u_east.resize(yz);
        state.radiative_v_south.resize(xz);
        state.radiative_v_north.resize(xz);
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < state_.grid.nz; ++k)
            {
                const size_t n = static_cast<size_t>(j) * state_.grid.nz + k;
                state.radiative_u_west[n] = state_.fields.u[0][j][k];
                state.radiative_u_east[n] = state_.fields.u[NX - 2][j][k];
            }
        for (int i = 0; i < NX; ++i)
            for (int k = 0; k < state_.grid.nz; ++k)
            {
                const size_t n = static_cast<size_t>(i) * state_.grid.nz + k;
                state.radiative_v_south[n] = state_.fields.v[i][0][k];
                state.radiative_v_north[n] = state_.fields.v[i][NY - 2][k];
            }
    }

    // Copy the dynamics scalar prognostics (rho, p) from (si,sj,k) to
    // (di,dj,k). theta + moisture are NOT touched: the scalar-advection kernel
    // owns them and updates the boundary cells too, so a zero-gradient
    // overwrite here would clobber that update and inject a spurious theta
    // tendency on a stratified column.
    void copy_scalars(int di, int dj, int si, int sj, int k) const
    {
        state_.fields.rho[di][dj][k] = state_.fields.rho[si][sj][k];
        state_.fields.p[di][dj][k]   = state_.fields.p[si][sj][k];
    }

    void copy_advected_scalars(
        int di, int dj, int si, int sj, int k) const
    {
        state_.fields.theta[di][dj][k] = state_.fields.theta[si][sj][k];
        state_.fields.tracer[di][dj][k] = state_.fields.tracer[si][sj][k];
        state_.fields.qv[di][dj][k] = state_.fields.qv[si][sj][k];
        state_.fields.qc[di][dj][k] = state_.fields.qc[si][sj][k];
        state_.fields.qr[di][dj][k] = state_.fields.qr[si][sj][k];
        state_.fields.qi[di][dj][k] = state_.fields.qi[si][sj][k];
        state_.fields.qs[di][dj][k] = state_.fields.qs[si][sj][k];
        state_.fields.qg[di][dj][k] = state_.fields.qg[si][sj][k];
        state_.fields.qh[di][dj][k] = state_.fields.qh[si][sj][k];
        state_.fields.tke[di][dj][k] = state_.fields.tke[si][sj][k];
    }

    void rebuild_advected_scalar_ghosts() const
    {
        // Whole-cell recycling replaces ghost storage along with the physical
        // domain. Unlike ordinary completed-step closure, the next transport
        // stencil therefore needs one explicit reconstruction of every
        // advected scalar ghost. Close x first and y second to retain the
        // scheme's established y-owned corner convention.
        const int east_ghost = state_.grid.nr - 1;
        const int north_ghost = state_.grid.nth - 1;
        const int west_source =
            state_.configuration->lateral_bc_west == LateralBC::Periodic ? state_.grid.nr - 2 : 1;
        const int east_source =
            state_.configuration->lateral_bc_east == LateralBC::Periodic ? 1 : state_.grid.nr - 2;
        for (int j = 0; j < state_.grid.nth; ++j)
            for (int k = 0; k < state_.grid.nz; ++k)
            {
                copy_advected_scalars(0, j, west_source, j, k);
                copy_advected_scalars(
                    east_ghost, j, east_source, j, k);
            }

        const int south_source =
            state_.configuration->lateral_bc_south == LateralBC::Periodic ? state_.grid.nth - 2 : 1;
        const int north_source =
            state_.configuration->lateral_bc_north == LateralBC::Periodic ? 1 : state_.grid.nth - 2;
        for (int i = 0; i < state_.grid.nr; ++i)
            for (int k = 0; k < state_.grid.nz; ++k)
            {
                copy_advected_scalars(i, 0, i, south_source, k);
                copy_advected_scalars(
                    i, north_ghost, i, north_source, k);
            }
    }

    void apply_boundaries(double dt_rad)
    {
        // First-ever application before any explicit acoustic seed initializes
        // all four histories once, before workers can update them.
        ensure_radiative_state(dt_rad);

        // Each face owns disjoint cells internally.  Implicit barriers retain
        // the historical west -> east -> south -> north corner ownership, and
        // the vertical pass follows the completed lateral closure.  One team
        // covers the whole boundary application instead of five fork/join
        // regions per acoustic substep.
        #pragma omp parallel
        {
            apply_lateral_parallel(dt_rad);
            apply_vertical_parallel();
        }
    }

    // Called collectively by every thread in apply_boundaries().
    void apply_lateral_parallel(double dt_rad)
    {
        auto& boundary_runtime = (*state_.boundary_condition_runtime);
        const int NX = state_.grid.nr, NY = state_.grid.nth;
        const int levels = state_.grid.nz;
        const bool radiate = (dt_rad > 0.0);
        // fp32 throughout, mirroring acoustic_bc_face_cartesian_cgrid.comp
        // exactly: the radiated faces feed an 8-substep feedback loop, so
        // mixed-precision drift here shows up as a parity breach.
        const float dt_rad_f = static_cast<float>(dt_rad);
        const float inv_dx_f =
            (state_.grid.dr > 0.0) ? 1.0f / static_cast<float>(state_.grid.dr) : 0.0f;
        const float inv_dy_f =
            (state_.grid.dtheta > 0.0) ? 1.0f / static_cast<float>(state_.grid.dtheta) : 0.0f;
        const float frame_u_f = static_cast<float>(
            (*state_.domain_frame).velocity_ms.u_ms);
        const float frame_v_f = static_cast<float>(
            (*state_.domain_frame).velocity_ms.v_ms);

        // ---- West (i = 0 ghost; i = 1 interior; wall face = u[0]) ----
        {
            const LateralBC bc = state_.configuration->lateral_bc_west;
            #pragma omp for collapse(2)
            for (int j = 0; j < NY; ++j)
                for (int k = 0; k < levels; ++k)
                {
                    if (bc == LateralBC::Periodic)
                    {
                        const int s = NX - 2;  // east interior
                        state_.fields.u[0][j][k] = state_.fields.u[s][j][k];
                        state_.fields.v[0][j][k] = state_.fields.v[s][j][k];
                        state_.fields.w[0][j][k] = state_.fields.w[s][j][k];
                        copy_scalars(0, j, s, j, k);
                    }
                    else
                    {
                        if (bc == LateralBC::Radiative)
                        {
                            // KW78: u[0] is radiation-owned. Outgoing
                            // phase at the west face is u - c* < 0.
                            if (radiate)
                            {
                                const size_t n =
                                    static_cast<size_t>(j) * levels + k;
                                const float up = boundary_runtime.radiative_u_west[n];
                                const float phase =
                                    up - frame_u_f
                                    - static_cast<float>(kCStar);
                                const float un = kw78_updated_face(
                                    up, phase, state_.fields.u[1][j][k] - up,
                                    dt_rad_f, inv_dx_f, true);
                                state_.fields.u[0][j][k] = un;
                                boundary_runtime.radiative_u_west[n] = un;
                            }
                        }
                        else if (bc == LateralBC::ZeroGradient)
                        {
                            state_.fields.u[0][j][k] = state_.fields.u[1][j][k];
                        }
                        else
                        {
                            state_.fields.u[0][j][k] = frame_u_f;
                        }
                        if (bc == LateralBC::Rigid) { state_.fields.v[0][j][k] = frame_v_f; state_.fields.w[0][j][k] = 0.0f; }
                        else { state_.fields.v[0][j][k] = state_.fields.v[1][j][k]; state_.fields.w[0][j][k] = state_.fields.w[1][j][k]; }
                        copy_scalars(0, j, 1, j, k);
                    }
                }
        }

        // ---- East (i = NX-1 ghost; i = NX-2 interior; wall face = u[NX-2]) ----
        {
            const LateralBC bc = state_.configuration->lateral_bc_east;
            const int g = NX - 1, a = NX - 2;
            #pragma omp for collapse(2)
            for (int j = 0; j < NY; ++j)
                for (int k = 0; k < levels; ++k)
                {
                    if (bc == LateralBC::Periodic)
                    {
                        const int s = 1;  // west interior
                        state_.fields.u[g][j][k] = state_.fields.u[s][j][k];
                        state_.fields.v[g][j][k] = state_.fields.v[s][j][k];
                        state_.fields.w[g][j][k] = state_.fields.w[s][j][k];
                        copy_scalars(g, j, s, j, k);
                    }
                    else
                    {
                        if (bc == LateralBC::Rigid || bc == LateralBC::RigidSlip)
                            state_.fields.u[a][j][k] = frame_u_f;
                        if (bc == LateralBC::Radiative && radiate)
                        {
                            // KW78: the radiation replaces the interior
                            // kernel's update of the boundary face.
                            // Outgoing phase at the east face is
                            // u + c* > 0.
                            const size_t n = static_cast<size_t>(j) * levels + k;
                            const float up = boundary_runtime.radiative_u_east[n];
                            const float phase =
                                up - frame_u_f
                                + static_cast<float>(kCStar);
                            const float un = kw78_updated_face(
                                up, phase, up - state_.fields.u[a - 1][j][k],
                                dt_rad_f, inv_dx_f, false);
                            state_.fields.u[a][j][k] = un;
                            boundary_runtime.radiative_u_east[n] = un;
                        }
                        else if (bc == LateralBC::ZeroGradient)
                        {
                            state_.fields.u[a][j][k] = state_.fields.u[a - 1][j][k];
                        }
                        state_.fields.u[g][j][k] =
                            (bc == LateralBC::Radiative
                             || bc == LateralBC::ZeroGradient)
                                ? state_.fields.u[a][j][k] : frame_u_f;
                        if (bc == LateralBC::Rigid) { state_.fields.v[g][j][k] = frame_v_f; state_.fields.w[g][j][k] = 0.0f; }
                        else { state_.fields.v[g][j][k] = state_.fields.v[a][j][k]; state_.fields.w[g][j][k] = state_.fields.w[a][j][k]; }
                        copy_scalars(g, j, a, j, k);
                    }
                }
        }

        // ---- South (j = 0 ghost; j = 1 interior; wall face = v[i][0]) ----
        {
            const LateralBC bc = state_.configuration->lateral_bc_south;
            #pragma omp for collapse(2)
            for (int i = 0; i < NX; ++i)
                for (int k = 0; k < levels; ++k)
                {
                    if (bc == LateralBC::Periodic)
                    {
                        const int s = NY - 2;  // north interior
                        state_.fields.v[i][0][k] = state_.fields.v[i][s][k];
                        state_.fields.u[i][0][k] = state_.fields.u[i][s][k];
                        state_.fields.w[i][0][k] = state_.fields.w[i][s][k];
                        copy_scalars(i, 0, i, s, k);
                    }
                    else
                    {
                        if (bc == LateralBC::Radiative)
                        {
                            if (radiate)
                            {
                                const size_t n =
                                    static_cast<size_t>(i) * levels + k;
                                const float vp = boundary_runtime.radiative_v_south[n];
                                const float phase =
                                    vp - frame_v_f
                                    - static_cast<float>(kCStar);
                                const float vn = kw78_updated_face(
                                    vp, phase, state_.fields.v[i][1][k] - vp,
                                    dt_rad_f, inv_dy_f, true);
                                state_.fields.v[i][0][k] = vn;
                                boundary_runtime.radiative_v_south[n] = vn;
                            }
                        }
                        else if (bc == LateralBC::ZeroGradient)
                        {
                            state_.fields.v[i][0][k] = state_.fields.v[i][1][k];
                        }
                        else
                        {
                            state_.fields.v[i][0][k] = frame_v_f;
                        }
                        if (bc == LateralBC::Rigid) { state_.fields.u[i][0][k] = frame_u_f; state_.fields.w[i][0][k] = 0.0f; }
                        else { state_.fields.u[i][0][k] = state_.fields.u[i][1][k]; state_.fields.w[i][0][k] = state_.fields.w[i][1][k]; }
                        copy_scalars(i, 0, i, 1, k);
                    }
                }
        }

        // ---- North (j = NY-1 ghost; j = NY-2 interior; wall face = v[i][NY-2]) ----
        {
            const LateralBC bc = state_.configuration->lateral_bc_north;
            const int g = NY - 1, a = NY - 2;
            #pragma omp for collapse(2)
            for (int i = 0; i < NX; ++i)
                for (int k = 0; k < levels; ++k)
                {
                    if (bc == LateralBC::Periodic)
                    {
                        const int s = 1;  // south interior
                        state_.fields.v[i][g][k] = state_.fields.v[i][s][k];
                        state_.fields.u[i][g][k] = state_.fields.u[i][s][k];
                        state_.fields.w[i][g][k] = state_.fields.w[i][s][k];
                        copy_scalars(i, g, i, s, k);
                    }
                    else
                    {
                        if (bc == LateralBC::Rigid || bc == LateralBC::RigidSlip)
                            state_.fields.v[i][a][k] = frame_v_f;
                        if (bc == LateralBC::Radiative && radiate)
                        {
                            const size_t n = static_cast<size_t>(i) * levels + k;
                            const float vp = boundary_runtime.radiative_v_north[n];
                            const float phase =
                                vp - frame_v_f
                                + static_cast<float>(kCStar);
                            const float vn = kw78_updated_face(
                                vp, phase, vp - state_.fields.v[i][a - 1][k],
                                dt_rad_f, inv_dy_f, false);
                            state_.fields.v[i][a][k] = vn;
                            boundary_runtime.radiative_v_north[n] = vn;
                        }
                        else if (bc == LateralBC::ZeroGradient)
                        {
                            state_.fields.v[i][a][k] = state_.fields.v[i][a - 1][k];
                        }
                        state_.fields.v[i][g][k] =
                            (bc == LateralBC::Radiative
                             || bc == LateralBC::ZeroGradient)
                                ? state_.fields.v[i][a][k] : frame_v_f;
                        if (bc == LateralBC::Rigid) { state_.fields.u[i][g][k] = frame_u_f; state_.fields.w[i][g][k] = 0.0f; }
                        else { state_.fields.u[i][g][k] = state_.fields.u[i][a][k]; state_.fields.w[i][g][k] = state_.fields.w[i][a][k]; }
                        copy_scalars(i, g, i, a, k);
                    }
                }
        }
    }

    // Called collectively by every thread in apply_boundaries().
    void apply_vertical_parallel() const
    {
        const int nx = state_.grid.nr;
        const int ny = state_.grid.nth;
        const int k_top = state_.grid.nz - 1;
        const int k_top_m = k_top - 1;

        #pragma omp for collapse(2)
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
            {
                // Bottom (k=0). w[0] is interior on the C-grid; the ground
                // contravariant flux is implicit and zero, so no surface w is
                // pinned. On terrain the corresponding physical surface
                // velocity is u*dh/dx+v*dh/dy rather than zero.
                // RigidSlip leaves u/v at k=0 PROGNOSTIC (SCIENCE.md
                // M19): free-slip is zero vertical momentum flux at the
                // ground, enforced in-kernel by the mirror ghost - the old
                // k0 := k1 copy enslaved the lowest level and erased
                // low-level shear (near-surface SRH, tornado-genesis
                // relevant). RigidNoSlip pins the level to the ground.
                if (state_.configuration->bottom_bc == BottomBC::RigidNoSlip)
                {
                    state_.fields.u[i][j][0] = 0.0f;
                    state_.fields.v[i][j][0] = 0.0f;
                }

                // Top (k=NZ-1). w[NZ-1] is the lid z-face.
                if (state_.configuration->top_bc == TopBC::Open)
                {
                    // Zero-gradient outflow closure for the boundary-normal
                    // face.  Leaving w[NZ-1] untouched made "open" depend on
                    // stale initialization while continuity simultaneously
                    // treated the same face as sealed.
                    state_.fields.w[i][j][k_top] = state_.fields.w[i][j][k_top_m];
                    state_.fields.u[i][j][k_top] = state_.fields.u[i][j][k_top_m];
                    state_.fields.v[i][j][k_top] = state_.fields.v[i][j][k_top_m];
                }
                else  // Rigid or RigidWithSponge (sponge applied separately)
                {
                    state_.fields.w[i][j][k_top] = 0.0f;
                    state_.fields.u[i][j][k_top] = state_.fields.u[i][j][k_top_m];
                    state_.fields.v[i][j][k_top] = state_.fields.v[i][j][k_top_m];
                }

                // p and rho stay PROGNOSTIC at the surface and lid (see header).
                // theta + moisture are owned by the advection kernel; not touched here.
            }
    }
};

std::unique_ptr<BoundaryConditionScheme> create_cartesian_cgrid_bc_scheme(const SimulationStateView& state)
{
    if (!state.configuration || !state.domain_frame || !state.boundary_condition_runtime || !state.dynamics_runtime)
        throw std::invalid_argument("boundary execution requires a complete level context");
    return std::make_unique<CartesianCGridBCScheme>(state);
}

std::unique_ptr<BoundaryConditionScheme> create_cartesian_cgrid_bc_scheme()
{
    return create_cartesian_cgrid_bc_scheme(global_simulation_state());
}

} // namespace tmv
