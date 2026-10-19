#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_grid_metrics.hpp"
/**
 * @file boundary_conditions_cartesian.cpp
 * @brief Cartesian boundary condition scheme.
 *
 * Lateral boundary type is selected per face by the global enums
 * `global_lateral_bc_{west,east,south,north}` (set from
 * `boundary_conditions.lateral.*` in the v1 YAML, defaulting to
 * ZeroGradient for v0 configs). Supported per-face: ZeroGradient, Rigid,
 * RigidSlip, Periodic. Radiative is rejected for this collocated scheme.
 *
 * Vertical faces remain rigid-lid + rigid-surface with hydrostatic
 * pressure extrapolation (configurable surface in a future C2 phase).
 */

#include "boundary_conditions/boundary_conditions_base.hpp"
#include "boundary_conditions/factory.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/process_state.hpp"
#include "dynamics/acoustic_energy_report.hpp"
#include "dynamics/dynamics_base.hpp"
#include "numerics/runtime.hpp"
#include "terrain/conservative_transport.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{

using tmv::init::config::LateralBC;

}  // namespace

namespace tmv
{

class CartesianBCScheme : public BoundaryConditionScheme
{
    SimulationStateView state_;
void require_explicit_collocated_lateral_bcs()
{
    const LateralBC faces[] = {
        state_.configuration->lateral_bc_west, state_.configuration->lateral_bc_east,
        state_.configuration->lateral_bc_south, state_.configuration->lateral_bc_north
    };
    for (const LateralBC bc : faces)
    {
        if (bc == LateralBC::Radiative)
        {
            throw std::runtime_error(
                "radiative lateral boundaries require Cartesian C-grid "
                "staggering; use zero_gradient for the collocated grid");
        }
        if (bc == LateralBC::LegacyOpen)
        {
            throw std::runtime_error(
                "ambiguous legacy 'open' lateral boundary reached the "
                "collocated runtime without configuration projection");
        }
    }
}

// ─── X faces (west=i=0, east=i=NR-1; normal = u) ─────────────
// ZeroGradient: copy interior → face for every field.
// Rigid: Earth-relative horizontal wind equals the translating lateral wall
// velocity; w=0. RigidSlip applies only the matching normal component.
// All variants share the same scalar-copy block.

inline void copy_x_face_scalars(int j, int k, int dst_i, int src_i)
{
    state_.fields.rho[dst_i][j][k]   = state_.fields.rho[src_i][j][k];
    state_.fields.p[dst_i][j][k]     = state_.fields.p[src_i][j][k];
    state_.fields.theta[dst_i][j][k] = state_.fields.theta[src_i][j][k];
    state_.fields.qv[dst_i][j][k]    = state_.fields.qv[src_i][j][k];
    state_.fields.qc[dst_i][j][k]    = state_.fields.qc[src_i][j][k];
    state_.fields.qr[dst_i][j][k]    = state_.fields.qr[src_i][j][k];
    state_.fields.qi[dst_i][j][k]    = state_.fields.qi[src_i][j][k];
    state_.fields.qs[dst_i][j][k]    = state_.fields.qs[src_i][j][k];
    state_.fields.qg[dst_i][j][k]    = state_.fields.qg[src_i][j][k];
    state_.fields.qh[dst_i][j][k]    = state_.fields.qh[src_i][j][k];
    if (!state_.fields.tracer.empty()) state_.fields.tracer[dst_i][j][k] = state_.fields.tracer[src_i][j][k];
    if (!state_.fields.tke.empty()) state_.fields.tke[dst_i][j][k] = state_.fields.tke[src_i][j][k];
}

inline void apply_x_face_open(int j, int k, int face_i, int interior_i)
{
    state_.fields.u[face_i][j][k] = state_.fields.u[interior_i][j][k];
    state_.fields.v[face_i][j][k] = state_.fields.v[interior_i][j][k];
    state_.fields.w[face_i][j][k] = state_.fields.w[interior_i][j][k];
    copy_x_face_scalars(j, k, face_i, interior_i);
}

inline void apply_x_face_rigid(int j, int k, int face_i, int interior_i)
{
    state_.fields.u[face_i][j][k] = static_cast<float>(
        (*state_.domain_frame).velocity_ms.u_ms);
    state_.fields.v[face_i][j][k] = static_cast<float>(
        (*state_.domain_frame).velocity_ms.v_ms);
    state_.fields.w[face_i][j][k] = 0.0f;
    copy_x_face_scalars(j, k, face_i, interior_i);
}

inline void apply_x_face_rigid_slip(int j, int k, int face_i, int interior_i)
{
    state_.fields.u[face_i][j][k] = static_cast<float>(
        (*state_.domain_frame).velocity_ms.u_ms);
    state_.fields.v[face_i][j][k] = state_.fields.v[interior_i][j][k];
    state_.fields.w[face_i][j][k] = state_.fields.w[interior_i][j][k];
    copy_x_face_scalars(j, k, face_i, interior_i);
}

/// Periodic pair: face cell at one end takes the interior value at the
/// opposite end. west(0) ← interior(NR-2), east(NR-1) ← interior(1).
inline void apply_x_periodic_pair(int j, int k)
{
    state_.fields.u[0][j][k]     = state_.fields.u[state_.grid.nr - 2][j][k];     state_.fields.u[state_.grid.nr - 1][j][k]     = state_.fields.u[1][j][k];
    state_.fields.v[0][j][k]     = state_.fields.v[state_.grid.nr - 2][j][k];     state_.fields.v[state_.grid.nr - 1][j][k]     = state_.fields.v[1][j][k];
    state_.fields.w[0][j][k]     = state_.fields.w[state_.grid.nr - 2][j][k];     state_.fields.w[state_.grid.nr - 1][j][k]     = state_.fields.w[1][j][k];
    state_.fields.rho[0][j][k]   = state_.fields.rho[state_.grid.nr - 2][j][k];   state_.fields.rho[state_.grid.nr - 1][j][k]   = state_.fields.rho[1][j][k];
    state_.fields.p[0][j][k]     = state_.fields.p[state_.grid.nr - 2][j][k];     state_.fields.p[state_.grid.nr - 1][j][k]     = state_.fields.p[1][j][k];
    state_.fields.theta[0][j][k] = state_.fields.theta[state_.grid.nr - 2][j][k]; state_.fields.theta[state_.grid.nr - 1][j][k] = state_.fields.theta[1][j][k];
    state_.fields.qv[0][j][k]    = state_.fields.qv[state_.grid.nr - 2][j][k];    state_.fields.qv[state_.grid.nr - 1][j][k]    = state_.fields.qv[1][j][k];
    state_.fields.qc[0][j][k]    = state_.fields.qc[state_.grid.nr - 2][j][k];    state_.fields.qc[state_.grid.nr - 1][j][k]    = state_.fields.qc[1][j][k];
    state_.fields.qr[0][j][k]    = state_.fields.qr[state_.grid.nr - 2][j][k];    state_.fields.qr[state_.grid.nr - 1][j][k]    = state_.fields.qr[1][j][k];
    state_.fields.qi[0][j][k]    = state_.fields.qi[state_.grid.nr - 2][j][k];    state_.fields.qi[state_.grid.nr - 1][j][k]    = state_.fields.qi[1][j][k];
    state_.fields.qs[0][j][k]    = state_.fields.qs[state_.grid.nr - 2][j][k];    state_.fields.qs[state_.grid.nr - 1][j][k]    = state_.fields.qs[1][j][k];
    state_.fields.qg[0][j][k]    = state_.fields.qg[state_.grid.nr - 2][j][k];    state_.fields.qg[state_.grid.nr - 1][j][k]    = state_.fields.qg[1][j][k];
    state_.fields.qh[0][j][k]    = state_.fields.qh[state_.grid.nr - 2][j][k];    state_.fields.qh[state_.grid.nr - 1][j][k]    = state_.fields.qh[1][j][k];
    if (!state_.fields.tracer.empty()) { state_.fields.tracer[0][j][k] = state_.fields.tracer[state_.grid.nr - 2][j][k]; state_.fields.tracer[state_.grid.nr - 1][j][k] = state_.fields.tracer[1][j][k]; }
    if (!state_.fields.tke.empty()) { state_.fields.tke[0][j][k] = state_.fields.tke[state_.grid.nr - 2][j][k]; state_.fields.tke[state_.grid.nr - 1][j][k] = state_.fields.tke[1][j][k]; }
}

inline void dispatch_x_face(LateralBC bc, int j, int k, int face_i, int interior_i)
{
    switch (bc)
    {
        case LateralBC::ZeroGradient: apply_x_face_open(j, k, face_i, interior_i);       break;
        case LateralBC::Rigid:        apply_x_face_rigid(j, k, face_i, interior_i);      break;
        case LateralBC::RigidSlip:    apply_x_face_rigid_slip(j, k, face_i, interior_i); break;
        case LateralBC::Periodic:     /* handled by apply_x_periodic_pair */              break;
        case LateralBC::Radiative:
        case LateralBC::LegacyOpen:   /* rejected before dispatch */                      break;
    }
}

// ─── Y faces (south=j=0, north=j=NTH-1; normal = v) ──────────

inline void copy_y_face_scalars(int i, int k, int dst_j, int src_j)
{
    state_.fields.rho[i][dst_j][k]   = state_.fields.rho[i][src_j][k];
    state_.fields.p[i][dst_j][k]     = state_.fields.p[i][src_j][k];
    state_.fields.theta[i][dst_j][k] = state_.fields.theta[i][src_j][k];
    state_.fields.qv[i][dst_j][k]    = state_.fields.qv[i][src_j][k];
    state_.fields.qc[i][dst_j][k]    = state_.fields.qc[i][src_j][k];
    state_.fields.qr[i][dst_j][k]    = state_.fields.qr[i][src_j][k];
    state_.fields.qi[i][dst_j][k]    = state_.fields.qi[i][src_j][k];
    state_.fields.qs[i][dst_j][k]    = state_.fields.qs[i][src_j][k];
    state_.fields.qg[i][dst_j][k]    = state_.fields.qg[i][src_j][k];
    state_.fields.qh[i][dst_j][k]    = state_.fields.qh[i][src_j][k];
    if (!state_.fields.tracer.empty()) state_.fields.tracer[i][dst_j][k] = state_.fields.tracer[i][src_j][k];
    if (!state_.fields.tke.empty()) state_.fields.tke[i][dst_j][k] = state_.fields.tke[i][src_j][k];
}

inline void apply_y_face_open(int i, int k, int face_j, int interior_j)
{
    state_.fields.u[i][face_j][k] = state_.fields.u[i][interior_j][k];
    state_.fields.v[i][face_j][k] = state_.fields.v[i][interior_j][k];
    state_.fields.w[i][face_j][k] = state_.fields.w[i][interior_j][k];
    copy_y_face_scalars(i, k, face_j, interior_j);
}

inline void apply_y_face_rigid(int i, int k, int face_j, int interior_j)
{
    state_.fields.u[i][face_j][k] = static_cast<float>(
        (*state_.domain_frame).velocity_ms.u_ms);
    state_.fields.v[i][face_j][k] = static_cast<float>(
        (*state_.domain_frame).velocity_ms.v_ms);
    state_.fields.w[i][face_j][k] = 0.0f;
    copy_y_face_scalars(i, k, face_j, interior_j);
}

inline void apply_y_face_rigid_slip(int i, int k, int face_j, int interior_j)
{
    state_.fields.u[i][face_j][k] = state_.fields.u[i][interior_j][k];
    state_.fields.v[i][face_j][k] = static_cast<float>(
        (*state_.domain_frame).velocity_ms.v_ms);
    state_.fields.w[i][face_j][k] = state_.fields.w[i][interior_j][k];
    copy_y_face_scalars(i, k, face_j, interior_j);
}

inline void apply_y_periodic_pair(int i, int k)
{
    state_.fields.u[i][0][k]     = state_.fields.u[i][state_.grid.nth - 2][k];     state_.fields.u[i][state_.grid.nth - 1][k]     = state_.fields.u[i][1][k];
    state_.fields.v[i][0][k]     = state_.fields.v[i][state_.grid.nth - 2][k];     state_.fields.v[i][state_.grid.nth - 1][k]     = state_.fields.v[i][1][k];
    state_.fields.w[i][0][k]     = state_.fields.w[i][state_.grid.nth - 2][k];     state_.fields.w[i][state_.grid.nth - 1][k]     = state_.fields.w[i][1][k];
    state_.fields.rho[i][0][k]   = state_.fields.rho[i][state_.grid.nth - 2][k];   state_.fields.rho[i][state_.grid.nth - 1][k]   = state_.fields.rho[i][1][k];
    state_.fields.p[i][0][k]     = state_.fields.p[i][state_.grid.nth - 2][k];     state_.fields.p[i][state_.grid.nth - 1][k]     = state_.fields.p[i][1][k];
    state_.fields.theta[i][0][k] = state_.fields.theta[i][state_.grid.nth - 2][k]; state_.fields.theta[i][state_.grid.nth - 1][k] = state_.fields.theta[i][1][k];
    state_.fields.qv[i][0][k]    = state_.fields.qv[i][state_.grid.nth - 2][k];    state_.fields.qv[i][state_.grid.nth - 1][k]    = state_.fields.qv[i][1][k];
    state_.fields.qc[i][0][k]    = state_.fields.qc[i][state_.grid.nth - 2][k];    state_.fields.qc[i][state_.grid.nth - 1][k]    = state_.fields.qc[i][1][k];
    state_.fields.qr[i][0][k]    = state_.fields.qr[i][state_.grid.nth - 2][k];    state_.fields.qr[i][state_.grid.nth - 1][k]    = state_.fields.qr[i][1][k];
    state_.fields.qi[i][0][k]    = state_.fields.qi[i][state_.grid.nth - 2][k];    state_.fields.qi[i][state_.grid.nth - 1][k]    = state_.fields.qi[i][1][k];
    state_.fields.qs[i][0][k]    = state_.fields.qs[i][state_.grid.nth - 2][k];    state_.fields.qs[i][state_.grid.nth - 1][k]    = state_.fields.qs[i][1][k];
    state_.fields.qg[i][0][k]    = state_.fields.qg[i][state_.grid.nth - 2][k];    state_.fields.qg[i][state_.grid.nth - 1][k]    = state_.fields.qg[i][1][k];
    state_.fields.qh[i][0][k]    = state_.fields.qh[i][state_.grid.nth - 2][k];    state_.fields.qh[i][state_.grid.nth - 1][k]    = state_.fields.qh[i][1][k];
    if (!state_.fields.tracer.empty()) { state_.fields.tracer[i][0][k] = state_.fields.tracer[i][state_.grid.nth - 2][k]; state_.fields.tracer[i][state_.grid.nth - 1][k] = state_.fields.tracer[i][1][k]; }
    if (!state_.fields.tke.empty()) { state_.fields.tke[i][0][k] = state_.fields.tke[i][state_.grid.nth - 2][k]; state_.fields.tke[i][state_.grid.nth - 1][k] = state_.fields.tke[i][1][k]; }
}

inline void dispatch_y_face(LateralBC bc, int i, int k, int face_j, int interior_j)
{
    switch (bc)
    {
        case LateralBC::ZeroGradient: apply_y_face_open(i, k, face_j, interior_j);       break;
        case LateralBC::Rigid:        apply_y_face_rigid(i, k, face_j, interior_j);      break;
        case LateralBC::RigidSlip:    apply_y_face_rigid_slip(i, k, face_j, interior_j); break;
        case LateralBC::Periodic:     /* handled by apply_y_periodic_pair */              break;
        case LateralBC::Radiative:
        case LateralBC::LegacyOpen:   /* rejected before dispatch */                      break;
    }
}

// ─── Acoustic-substep variants (momentum + rho + p only) ─────

inline void copy_x_face_acoustic_scalars(int j, int k, int dst_i, int src_i)
{
    state_.fields.rho[dst_i][j][k] = state_.fields.rho[src_i][j][k];
    state_.fields.p[dst_i][j][k]   = state_.fields.p[src_i][j][k];
}

inline void dispatch_x_face_acoustic(LateralBC bc, int j, int k, int face_i, int interior_i)
{
    switch (bc)
    {
        case LateralBC::ZeroGradient:
            state_.fields.u[face_i][j][k] = state_.fields.u[interior_i][j][k];
            state_.fields.v[face_i][j][k] = state_.fields.v[interior_i][j][k];
            state_.fields.w[face_i][j][k] = state_.fields.w[interior_i][j][k];
            copy_x_face_acoustic_scalars(j, k, face_i, interior_i);
            break;
        case LateralBC::Rigid:
            state_.fields.u[face_i][j][k] = static_cast<float>(
                (*state_.domain_frame).velocity_ms.u_ms);
            state_.fields.v[face_i][j][k] = static_cast<float>(
                (*state_.domain_frame).velocity_ms.v_ms);
            state_.fields.w[face_i][j][k] = 0.0f;
            copy_x_face_acoustic_scalars(j, k, face_i, interior_i);
            break;
        case LateralBC::RigidSlip:
            state_.fields.u[face_i][j][k] = static_cast<float>(
                (*state_.domain_frame).velocity_ms.u_ms);
            state_.fields.v[face_i][j][k] = state_.fields.v[interior_i][j][k];
            state_.fields.w[face_i][j][k] = state_.fields.w[interior_i][j][k];
            copy_x_face_acoustic_scalars(j, k, face_i, interior_i);
            break;
        case LateralBC::Periodic: /* handled separately */ break;
        case LateralBC::Radiative:
        case LateralBC::LegacyOpen: /* rejected before dispatch */ break;
    }
}

inline void apply_x_periodic_pair_acoustic(int j, int k)
{
    state_.fields.u[0][j][k]   = state_.fields.u[state_.grid.nr - 2][j][k];   state_.fields.u[state_.grid.nr - 1][j][k]   = state_.fields.u[1][j][k];
    state_.fields.v[0][j][k]   = state_.fields.v[state_.grid.nr - 2][j][k];   state_.fields.v[state_.grid.nr - 1][j][k]   = state_.fields.v[1][j][k];
    state_.fields.w[0][j][k]   = state_.fields.w[state_.grid.nr - 2][j][k];   state_.fields.w[state_.grid.nr - 1][j][k]   = state_.fields.w[1][j][k];
    state_.fields.rho[0][j][k] = state_.fields.rho[state_.grid.nr - 2][j][k]; state_.fields.rho[state_.grid.nr - 1][j][k] = state_.fields.rho[1][j][k];
    state_.fields.p[0][j][k]   = state_.fields.p[state_.grid.nr - 2][j][k];   state_.fields.p[state_.grid.nr - 1][j][k]   = state_.fields.p[1][j][k];
}

inline void dispatch_y_face_acoustic(LateralBC bc, int i, int k, int face_j, int interior_j)
{
    switch (bc)
    {
        case LateralBC::ZeroGradient:
            state_.fields.u[i][face_j][k]   = state_.fields.u[i][interior_j][k];
            state_.fields.v[i][face_j][k]   = state_.fields.v[i][interior_j][k];
            state_.fields.w[i][face_j][k]   = state_.fields.w[i][interior_j][k];
            state_.fields.rho[i][face_j][k] = state_.fields.rho[i][interior_j][k];
            state_.fields.p[i][face_j][k]   = state_.fields.p[i][interior_j][k];
            break;
        case LateralBC::Rigid:
            state_.fields.u[i][face_j][k]   = static_cast<float>(
                (*state_.domain_frame).velocity_ms.u_ms);
            state_.fields.v[i][face_j][k]   = static_cast<float>(
                (*state_.domain_frame).velocity_ms.v_ms);
            state_.fields.w[i][face_j][k]   = 0.0f;
            state_.fields.rho[i][face_j][k] = state_.fields.rho[i][interior_j][k];
            state_.fields.p[i][face_j][k]   = state_.fields.p[i][interior_j][k];
            break;
        case LateralBC::RigidSlip:
            state_.fields.u[i][face_j][k]   = state_.fields.u[i][interior_j][k];
            state_.fields.v[i][face_j][k]   = static_cast<float>(
                (*state_.domain_frame).velocity_ms.v_ms);
            state_.fields.w[i][face_j][k]   = state_.fields.w[i][interior_j][k];
            state_.fields.rho[i][face_j][k] = state_.fields.rho[i][interior_j][k];
            state_.fields.p[i][face_j][k]   = state_.fields.p[i][interior_j][k];
            break;
        case LateralBC::Periodic: /* handled separately */ break;
        case LateralBC::Radiative:
        case LateralBC::LegacyOpen: /* rejected before dispatch */ break;
    }
}

inline void apply_y_periodic_pair_acoustic(int i, int k)
{
    state_.fields.u[i][0][k]   = state_.fields.u[i][state_.grid.nth - 2][k];   state_.fields.u[i][state_.grid.nth - 1][k]   = state_.fields.u[i][1][k];
    state_.fields.v[i][0][k]   = state_.fields.v[i][state_.grid.nth - 2][k];   state_.fields.v[i][state_.grid.nth - 1][k]   = state_.fields.v[i][1][k];
    state_.fields.w[i][0][k]   = state_.fields.w[i][state_.grid.nth - 2][k];   state_.fields.w[i][state_.grid.nth - 1][k]   = state_.fields.w[i][1][k];
    state_.fields.rho[i][0][k] = state_.fields.rho[i][state_.grid.nth - 2][k]; state_.fields.rho[i][state_.grid.nth - 1][k] = state_.fields.rho[i][1][k];
    state_.fields.p[i][0][k]   = state_.fields.p[i][state_.grid.nth - 2][k];   state_.fields.p[i][state_.grid.nth - 1][k]   = state_.fields.p[i][1][k];
}

/// Vertical BCs shared by full and acoustic paths. Dispatches on the
/// global_bottom_bc / global_top_bc enums set from the v1 schema
/// (`boundary_conditions.{bottom,top}`).
///
/// Bottom variants:
///   RigidNoSlip — w=0, u=0, v=0 (full friction at the surface)
///   RigidSlip   — w=0, u=u[1], v=v[1] (free-slip; the legacy default)
///
/// Top variants:
///   Rigid           — w=0, u/v zero-gradient (legacy default)
///   RigidWithSponge — same BC; the sponge itself is C3's deliverable
///   Open            — no w constraint (radiative-style); u/v zero-gradient
///
/// Thermodynamic ghost values close perturbations relative to the exact stored
/// base profile. Reconstructing a new hydrostatic approximation here makes
/// p-p0 and rho-rho0 nonzero at rest, which injects a false fast-mode pressure
/// gradient into the first interior level.
void require_vertical_base_profiles(bool full_state)
{
    const auto has_nz = [this](const std::vector<double>& profile)
    {
        return profile.size() >= static_cast<std::size_t>(state_.grid.nz);
    };
    if (!has_nz(state_.base.p0) || !has_nz(state_.base.rho0)
        || !has_nz(state_.base.u0) || !has_nz(state_.base.v0)
        || (full_state && (!has_nz(state_.base.theta0) || !has_nz(state_.base.qv0))))
    {
        throw std::runtime_error(
            "collocated vertical boundary closure requires complete p0, rho0, "
            "u0, v0, theta0, and qv0 base profiles");
    }
}

inline float close_base_perturbation(float interior_value,
                                     const std::vector<double>& base,
                                     int ghost_k, int interior_k)
{
    const float ghost_base = static_cast<float>(
        base[static_cast<std::size_t>(ghost_k)]);
    const float interior_base = static_cast<float>(
        base[static_cast<std::size_t>(interior_k)]);
    return ghost_base + (interior_value - interior_base);
}

const TerrainMetrics3D* active_collocated_terrain_metrics()
{
    return tmv::physical_grid_metrics::terrain_is_active(state_)
        ? tmv::physical_grid_metrics::active_terrain_metrics(state_) : nullptr;
}

float close_mapped_profile_perturbation(
    float interior_value,
    const std::vector<double>& profile,
    const TerrainMetrics3D& metrics,
    int i, int j, int ghost_k, int interior_k)
{
    const double ghost_reference =
        tmv::terrain_transport::reference_profile_value_fp32(
            profile, metrics.z(i, j, ghost_k), metrics.dzeta);
    const double interior_reference =
        tmv::terrain_transport::reference_profile_value_fp32(
            profile, metrics.z(i, j, interior_k), metrics.dzeta);
    return static_cast<float>(ghost_reference
        + (static_cast<double>(interior_value) - interior_reference));
}

void apply_vertical_bcs_momentum()
{
    using tmv::init::config::BottomBC;
    using tmv::init::config::TopBC;

    const BottomBC bottom = state_.configuration->bottom_bc;
    const TopBC    top    = state_.configuration->top_bc;
    const TerrainMetrics3D* terrain = active_collocated_terrain_metrics();
    if (terrain != nullptr)
        tmv::terrain_transport::validate_geometry_shape(*terrain);

    for (int i = 0; i < state_.grid.nr; ++i)
        for (int j = 0; j < state_.grid.nth; ++j)
        {
            // ── Bottom (k=0) ──
            if (terrain != nullptr && bottom == BottomBC::RigidNoSlip)
            {
                const double frame_u = tmv::physical_grid_metrics::surface_velocity_ms(state_).u_ms;
                const double frame_v = tmv::physical_grid_metrics::surface_velocity_ms(state_).v_ms;
                state_.fields.u[i][j][0] = static_cast<float>(
                    2.0 * frame_u - static_cast<double>(state_.fields.u[i][j][1]));
                state_.fields.v[i][j][0] = static_cast<float>(
                    2.0 * frame_v - static_cast<double>(state_.fields.v[i][j][1]));
                state_.fields.w[i][j][0] = -state_.fields.w[i][j][1];
            }
            else if (terrain != nullptr)
            {
                state_.fields.u[i][j][0] = close_mapped_profile_perturbation(
                    state_.fields.u[i][j][1], state_.base.u0, *terrain, i, j, 0, 1);
                state_.fields.v[i][j][0] = close_mapped_profile_perturbation(
                    state_.fields.v[i][j][1], state_.base.v0, *terrain, i, j, 0, 1);
                const auto ground =
                    tmv::terrain_transport::ground_surface_frame(
                        *terrain, i, j);
                const double tangent_w = ground.slope_x * (
                        0.5 * (static_cast<double>(state_.fields.u[i][j][0])
                             + static_cast<double>(state_.fields.u[i][j][1]))
                        - tmv::physical_grid_metrics::surface_velocity_ms(state_).u_ms)
                    + ground.slope_y * (
                        0.5 * (static_cast<double>(state_.fields.v[i][j][0])
                             + static_cast<double>(state_.fields.v[i][j][1]))
                        - tmv::physical_grid_metrics::surface_velocity_ms(state_).v_ms);
                state_.fields.w[i][j][0] = static_cast<float>(
                    2.0 * tangent_w - static_cast<double>(state_.fields.w[i][j][1]));
            }
            else if (bottom == BottomBC::RigidNoSlip)
            {
                state_.fields.w[i][j][0] = 0.0f;
                state_.fields.u[i][j][0] = 0.0f;
                state_.fields.v[i][j][0] = 0.0f;
            }
            else  // RigidSlip
            {
                state_.fields.w[i][j][0] = 0.0f;
                state_.fields.u[i][j][0] = close_base_perturbation(
                    state_.fields.u[i][j][1], state_.base.u0, 0, 1);
                state_.fields.v[i][j][0] = close_base_perturbation(
                    state_.fields.v[i][j][1], state_.base.v0, 0, 1);
            }

            // ── Top (k=NZ-1) ──
            if (terrain != nullptr && top == TopBC::Open)
            {
                state_.fields.u[i][j][state_.grid.nz - 1] = close_mapped_profile_perturbation(
                    state_.fields.u[i][j][state_.grid.nz - 2], state_.base.u0, *terrain,
                    i, j, state_.grid.nz - 1, state_.grid.nz - 2);
                state_.fields.v[i][j][state_.grid.nz - 1] = close_mapped_profile_perturbation(
                    state_.fields.v[i][j][state_.grid.nz - 2], state_.base.v0, *terrain,
                    i, j, state_.grid.nz - 1, state_.grid.nz - 2);
            }
            else if (terrain != nullptr)
            {
                state_.fields.w[i][j][state_.grid.nz - 1] = -state_.fields.w[i][j][state_.grid.nz - 2];
                state_.fields.u[i][j][state_.grid.nz - 1] = close_mapped_profile_perturbation(
                    state_.fields.u[i][j][state_.grid.nz - 2], state_.base.u0, *terrain,
                    i, j, state_.grid.nz - 1, state_.grid.nz - 2);
                state_.fields.v[i][j][state_.grid.nz - 1] = close_mapped_profile_perturbation(
                    state_.fields.v[i][j][state_.grid.nz - 2], state_.base.v0, *terrain,
                    i, j, state_.grid.nz - 1, state_.grid.nz - 2);
            }
            else if (top == TopBC::Open)
            {
                // No w-constraint; tangential velocities still zero-gradient
                // so the stencil at k=NZ-2 has a defined neighbour.
                state_.fields.u[i][j][state_.grid.nz - 1] = close_base_perturbation(
                    state_.fields.u[i][j][state_.grid.nz - 2], state_.base.u0, state_.grid.nz - 1, state_.grid.nz - 2);
                state_.fields.v[i][j][state_.grid.nz - 1] = close_base_perturbation(
                    state_.fields.v[i][j][state_.grid.nz - 2], state_.base.v0, state_.grid.nz - 1, state_.grid.nz - 2);
            }
            else  // Rigid or RigidWithSponge (sponge applied separately, C3)
            {
                state_.fields.w[i][j][state_.grid.nz - 1] = 0.0f;
                state_.fields.u[i][j][state_.grid.nz - 1] = close_base_perturbation(
                    state_.fields.u[i][j][state_.grid.nz - 2], state_.base.u0, state_.grid.nz - 1, state_.grid.nz - 2);
                state_.fields.v[i][j][state_.grid.nz - 1] = close_base_perturbation(
                    state_.fields.v[i][j][state_.grid.nz - 2], state_.base.v0, state_.grid.nz - 1, state_.grid.nz - 2);
            }

            if (terrain != nullptr)
            {
                state_.fields.p[i][j][0] = close_mapped_profile_perturbation(
                    state_.fields.p[i][j][1], state_.base.p0, *terrain, i, j, 0, 1);
                state_.fields.p[i][j][state_.grid.nz - 1] = close_mapped_profile_perturbation(
                    state_.fields.p[i][j][state_.grid.nz - 2], state_.base.p0, *terrain,
                    i, j, state_.grid.nz - 1, state_.grid.nz - 2);
                const double rho_reference_bottom =
                    tmv::terrain_transport::reference_density_fp32(
                        state_.base.p0, state_.base.theta0, terrain->z(i, j, 0),
                        terrain->dzeta);
                const double rho_reference_first =
                    tmv::terrain_transport::reference_density_fp32(
                        state_.base.p0, state_.base.theta0, terrain->z(i, j, 1),
                        terrain->dzeta);
                const double rho_reference_top =
                    tmv::terrain_transport::reference_density_fp32(
                        state_.base.p0, state_.base.theta0,
                        terrain->z(i, j, state_.grid.nz - 1), terrain->dzeta);
                const double rho_reference_last =
                    tmv::terrain_transport::reference_density_fp32(
                        state_.base.p0, state_.base.theta0,
                        terrain->z(i, j, state_.grid.nz - 2), terrain->dzeta);
                state_.fields.rho[i][j][0] = static_cast<float>(rho_reference_bottom
                    + (static_cast<double>(state_.fields.rho[i][j][1])
                       - rho_reference_first));
                state_.fields.rho[i][j][state_.grid.nz - 1] = static_cast<float>(rho_reference_top
                    + (static_cast<double>(state_.fields.rho[i][j][state_.grid.nz - 2])
                       - rho_reference_last));
            }
            else
            {
                state_.fields.rho[i][j][0] = close_base_perturbation(
                    state_.fields.rho[i][j][1], state_.base.rho0, 0, 1);
                state_.fields.rho[i][j][state_.grid.nz - 1] = close_base_perturbation(
                    state_.fields.rho[i][j][state_.grid.nz - 2], state_.base.rho0, state_.grid.nz - 1, state_.grid.nz - 2);
                state_.fields.p[i][j][0] = close_base_perturbation(
                    state_.fields.p[i][j][1], state_.base.p0, 0, 1);
                state_.fields.p[i][j][state_.grid.nz - 1] = close_base_perturbation(
                    state_.fields.p[i][j][state_.grid.nz - 2], state_.base.p0, state_.grid.nz - 1, state_.grid.nz - 2);
            }
        }
}


public:
    explicit CartesianBCScheme(const SimulationStateView& state) : state_(state) {}

    void apply_full() override
    {
        require_explicit_collocated_lateral_bcs();
        require_vertical_base_profiles(true);
        const LateralBC west  = state_.configuration->lateral_bc_west;
        const LateralBC east  = state_.configuration->lateral_bc_east;
        const LateralBC south = state_.configuration->lateral_bc_south;
        const LateralBC north = state_.configuration->lateral_bc_north;
        const bool x_periodic = (west == LateralBC::Periodic) ||
                                (east == LateralBC::Periodic);
        const bool y_periodic = (south == LateralBC::Periodic) ||
                                (north == LateralBC::Periodic);

        // Lateral x faces (west=i=0, east=i=NR-1)
        for (int j = 0; j < state_.grid.nth; ++j)
            for (int k = 0; k < state_.grid.nz; ++k)
            {
                if (x_periodic)
                {
                    apply_x_periodic_pair(j, k);
                }
                else
                {
                    dispatch_x_face(west, j, k, /*face=*/0,     /*interior=*/1);
                    dispatch_x_face(east, j, k, /*face=*/state_.grid.nr - 1, /*interior=*/state_.grid.nr - 2);
                }
            }

        // Lateral y faces (south=j=0, north=j=NTH-1)
        for (int i = 0; i < state_.grid.nr; ++i)
            for (int k = 0; k < state_.grid.nz; ++k)
            {
                if (y_periodic)
                {
                    apply_y_periodic_pair(i, k);
                }
                else
                {
                    dispatch_y_face(south, i, k, /*face=*/0,      /*interior=*/1);
                    dispatch_y_face(north, i, k, /*face=*/state_.grid.nth - 1, /*interior=*/state_.grid.nth - 2);
                }
            }

        // Vertical faces: rigid lid/surface + hydrostatic p (C2 will expose these)
        apply_vertical_bcs_momentum();

        // Vertical scalars (theta + moisture): zero-gradient
        const TerrainMetrics3D* terrain = active_collocated_terrain_metrics();
        for (int i = 0; i < state_.grid.nr; ++i)
            for (int j = 0; j < state_.grid.nth; ++j)
            {
                if (terrain != nullptr)
                {
                    state_.fields.theta[i][j][0] = close_mapped_profile_perturbation(
                        state_.fields.theta[i][j][1], state_.base.theta0, *terrain,
                        i, j, 0, 1);
                    state_.fields.theta[i][j][state_.grid.nz - 1] = close_mapped_profile_perturbation(
                        state_.fields.theta[i][j][state_.grid.nz - 2], state_.base.theta0, *terrain,
                        i, j, state_.grid.nz - 1, state_.grid.nz - 2);
                    state_.fields.qv[i][j][0] = close_mapped_profile_perturbation(
                        state_.fields.qv[i][j][1], state_.base.qv0, *terrain,
                        i, j, 0, 1);
                    state_.fields.qv[i][j][state_.grid.nz - 1] = close_mapped_profile_perturbation(
                        state_.fields.qv[i][j][state_.grid.nz - 2], state_.base.qv0, *terrain,
                        i, j, state_.grid.nz - 1, state_.grid.nz - 2);
                }
                else
                {
                    state_.fields.theta[i][j][0] = close_base_perturbation(
                        state_.fields.theta[i][j][1], state_.base.theta0, 0, 1);
                    state_.fields.theta[i][j][state_.grid.nz - 1] = close_base_perturbation(
                        state_.fields.theta[i][j][state_.grid.nz - 2], state_.base.theta0, state_.grid.nz - 1, state_.grid.nz - 2);
                    state_.fields.qv[i][j][0] = close_base_perturbation(
                        state_.fields.qv[i][j][1], state_.base.qv0, 0, 1);
                    state_.fields.qv[i][j][state_.grid.nz - 1] = close_base_perturbation(
                        state_.fields.qv[i][j][state_.grid.nz - 2], state_.base.qv0, state_.grid.nz - 1, state_.grid.nz - 2);
                }
                state_.fields.qc[i][j][0] = state_.fields.qc[i][j][1]; state_.fields.qc[i][j][state_.grid.nz - 1] = state_.fields.qc[i][j][state_.grid.nz - 2];
                state_.fields.qr[i][j][0] = state_.fields.qr[i][j][1]; state_.fields.qr[i][j][state_.grid.nz - 1] = state_.fields.qr[i][j][state_.grid.nz - 2];
                state_.fields.qi[i][j][0] = state_.fields.qi[i][j][1]; state_.fields.qi[i][j][state_.grid.nz - 1] = state_.fields.qi[i][j][state_.grid.nz - 2];
                state_.fields.qs[i][j][0] = state_.fields.qs[i][j][1]; state_.fields.qs[i][j][state_.grid.nz - 1] = state_.fields.qs[i][j][state_.grid.nz - 2];
                state_.fields.qg[i][j][0] = state_.fields.qg[i][j][1]; state_.fields.qg[i][j][state_.grid.nz - 1] = state_.fields.qg[i][j][state_.grid.nz - 2];
                state_.fields.qh[i][j][0] = state_.fields.qh[i][j][1]; state_.fields.qh[i][j][state_.grid.nz - 1] = state_.fields.qh[i][j][state_.grid.nz - 2];
                if (!state_.fields.tracer.empty())
                {
                    state_.fields.tracer[i][j][0] = state_.fields.tracer[i][j][1];
                    state_.fields.tracer[i][j][state_.grid.nz - 1] = state_.fields.tracer[i][j][state_.grid.nz - 2];
                }
                if (!state_.fields.tke.empty())
                {
                    state_.fields.tke[i][j][0] = state_.fields.tke[i][j][1];
                    state_.fields.tke[i][j][state_.grid.nz - 1] = state_.fields.tke[i][j][state_.grid.nz - 2];
                }
            }
    }

    FinalBoundaryEnergyReportV1 apply_full_attributed() override
    {
        // The collocated physical domain excludes its one-cell halo on every
        // side. apply_full() mutates only those halo cells, including k=0 and
        // k=NZ-1, so every basis-v2 physical reservoir is unchanged.
        apply_full();
        FinalBoundaryEnergyReportV1 report;
        report.applied = true;
        report.complete = true;
        return report;
    }

    BoundaryFieldWriteSet full_attributed_write_set() const noexcept override
    {
        static const char* const field_ids[] = {
            "u", "v", "w", "rho", "p", "theta", "qv",
            "qc", "qr", "qi", "qs", "qg", "qh", "tracer", "tke"};
        return {field_ids, sizeof(field_ids) / sizeof(field_ids[0]), true};
    }

    void apply_acoustic(double /*dt_small*/) override
    {
        require_explicit_collocated_lateral_bcs();
        require_vertical_base_profiles(false);
        const LateralBC west  = state_.configuration->lateral_bc_west;
        const LateralBC east  = state_.configuration->lateral_bc_east;
        const LateralBC south = state_.configuration->lateral_bc_south;
        const LateralBC north = state_.configuration->lateral_bc_north;
        const bool x_periodic = (west == LateralBC::Periodic) ||
                                (east == LateralBC::Periodic);
        const bool y_periodic = (south == LateralBC::Periodic) ||
                                (north == LateralBC::Periodic);

        // Lateral x faces (momentum + rho + p only)
        for (int j = 0; j < state_.grid.nth; ++j)
            for (int k = 0; k < state_.grid.nz; ++k)
            {
                if (x_periodic)
                {
                    apply_x_periodic_pair_acoustic(j, k);
                }
                else
                {
                    dispatch_x_face_acoustic(west, j, k, 0,      1);
                    dispatch_x_face_acoustic(east, j, k, state_.grid.nr - 1, state_.grid.nr - 2);
                }
            }

        // Lateral y faces (momentum + rho + p only)
        for (int i = 0; i < state_.grid.nr; ++i)
            for (int k = 0; k < state_.grid.nz; ++k)
            {
                if (y_periodic)
                {
                    apply_y_periodic_pair_acoustic(i, k);
                }
                else
                {
                    dispatch_y_face_acoustic(south, i, k, 0,       1);
                    dispatch_y_face_acoustic(north, i, k, state_.grid.nth - 1, state_.grid.nth - 2);
                }
            }

        // Vertical: shared momentum BCs
        apply_vertical_bcs_momentum();
    }

    void apply_acoustic_attributed(
        double dt_small,
        CartesianCGridAcousticEnergyReportV1& report) override
    {
        if (!std::isfinite(dt_small) || dt_small < 0.0)
        {
            throw std::invalid_argument(
                "attributed collocated acoustic boundary closure requires finite dt >= 0");
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
        if (std::any_of(
                std::begin(prior_values), std::end(prior_values),
                [](double value) { return !std::isfinite(value); })
            || (report.boundary_application_count > 0
                && !report.boundary_complete))
        {
            throw std::invalid_argument(
                "attributed collocated acoustic boundary closure received an invalid report");
        }

        // The collocated physical domain excludes its one-cell halo on all
        // six sides. This closure mutates only those halo cells, so its exact
        // moist-mixture basis-v2 physical-inventory change is zero. The
        // topology-owned interior null-mode projection is attributed by the
        // dynamics orchestrator before this closure.
        apply_acoustic(dt_small);

        CartesianCGridAcousticEnergyReportV1 next = report;
        ++next.boundary_application_count;
        next.boundary_complete = true;
        report = next;
    }

    std::string get_scheme_name() const override { return "cartesian"; }
};

// Factory access
std::unique_ptr<BoundaryConditionScheme> create_cartesian_bc_scheme(const SimulationStateView& state)
{
    if (!state.configuration || !state.domain_frame || !state.boundary_condition_runtime || !state.dynamics_runtime)
        throw std::invalid_argument("boundary execution requires a complete level context");
    return std::make_unique<CartesianBCScheme>(state);
}

std::unique_ptr<BoundaryConditionScheme> create_cartesian_bc_scheme()
{
    return create_cartesian_bc_scheme(global_simulation_state());
}

} // namespace tmv
