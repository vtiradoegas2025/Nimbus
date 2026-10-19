#pragma once

#include <cstdint>
#include "compute/dispatch/conservative_remap.hpp"
#include "compute/dispatch/earth_geometry.hpp"
#include "compute/dispatch/moving_momentum.hpp"
#include "compute/dispatch/moving_energy.hpp"
#include "compute/dispatch/reference_atmosphere.hpp"
#include "compute/dispatch/column_remap.hpp"
#include "compute/dispatch/held_sources.hpp"

struct MappedTerrainMassContinuityReport
{
    std::uint32_t ground_tangency_dispatch_count = 0;
    std::uint32_t mass_flux_continuity_dispatch_count = 0;
    bool complete = false;
};

struct MappedTerrainMassContinuityDispatch
{
    using CompletedObserver = void (*)(void* context,
                                       const float* ground_physical_w,
                                       const float* drho_dt,
                                       const float* dp_dt,
                                       const float* mass_flux_x_positive,
                                       const float* mass_flux_y_positive,
                                       const float* mass_flux_zeta_positive);

    const float* rho = nullptr;
    const float* u = nullptr;
    const float* v = nullptr;
    const float* w = nullptr;
    const float* p = nullptr;
    const float* jacobian = nullptr;
    const float* jacobian_x_positive_face = nullptr;
    const float* jacobian_y_positive_face = nullptr;
    const float* metric_x_positive_zeta_face = nullptr;
    const float* metric_y_positive_zeta_face = nullptr;
    const float* metric_x_ground_face = nullptr;
    const float* metric_y_ground_face = nullptr;

    float* ground_physical_w = nullptr;
    float* drho_dt = nullptr;
    float* dp_dt = nullptr;
    float* mass_flux_x_positive = nullptr;
    float* mass_flux_y_positive = nullptr;
    float* mass_flux_zeta_positive = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float dzeta = 0.0f;
    float gamma = 0.0f;
    float frame_u_ms = 0.0f;
    float frame_v_ms = 0.0f;
    MappedTerrainMassContinuityReport* report = nullptr;
    CompletedObserver completed_observer = nullptr;
    void* completed_observer_context = nullptr;
};

struct MappedTerrainFastMomentumReport
{
    std::uint32_t pressure_force_dispatch_count = 0;
    bool complete = false;
};

struct MappedTerrainFastMomentumDispatch
{
    using CompletedObserver = void (*)(void* context,
                                       const float* du_dt,
                                       const float* dv_dt,
                                       const float* dw_dt);

    const float* rho = nullptr;
    const float* p = nullptr;
    const float* physical_height = nullptr;
    const float* reference_pressure = nullptr;
    float* du_dt = nullptr;
    float* dv_dt = nullptr;
    float* dw_dt = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float nominal_dz = 0.0f;
    MappedTerrainFastMomentumReport* report = nullptr;
    CompletedObserver completed_observer = nullptr;
    void* completed_observer_context = nullptr;
    // Appends per-cell canonical pressure/density/vapor triples after the nz-entry pressure profile.
    bool sampled_reference_state = false;
};

struct MappedTerrainSlowTendenciesReport
{
    std::uint32_t slow_tendency_dispatch_count = 0;
    bool advection_attribution_complete = false;
    bool complete = false;
};

struct MappedTerrainSlowTendenciesDispatch
{
    using CompletedObserver = void (*)(void* context,
                                       const float* du_dt,
                                       const float* dv_dt,
                                       const float* dw_dt,
                                       const float* drho_dt,
                                       const float* dp_dt,
                                       const float* du_advection_dt,
                                       const float* dv_advection_dt,
                                       const float* dw_advection_dt);

    const float* u = nullptr;
    const float* v = nullptr;
    const float* w = nullptr;
    const float* rho = nullptr;
    const float* p = nullptr;
    const float* physical_height = nullptr;
    const float* jacobian_x_positive_face = nullptr;
    const float* jacobian_y_positive_face = nullptr;
    const float* metric_x_positive_zeta_face = nullptr;
    const float* metric_y_positive_zeta_face = nullptr;
    const float* ground_physical_w = nullptr;
    const float* reference_pressure = nullptr;
    const float* reference_theta = nullptr;
    const float* qv = nullptr;
    const float* reference_qv = nullptr;
    const float* hydrometeor_loading = nullptr;
    float* du_dt = nullptr;
    float* dv_dt = nullptr;
    float* dw_dt = nullptr;
    float* drho_dt = nullptr;
    float* dp_dt = nullptr;
    float* du_advection_dt = nullptr;
    float* dv_advection_dt = nullptr;
    float* dw_advection_dt = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    bool has_moisture = false;
    bool has_loading = false;
    float dx = 0.0f;
    float dy = 0.0f;
    float dzeta = 0.0f;
    float nominal_dz = 0.0f;
    float gravity = 0.0f;
    MappedTerrainSlowTendenciesReport* report = nullptr;
    CompletedObserver completed_observer = nullptr;
    void* completed_observer_context = nullptr;
    // Appends per-cell canonical pressure/density/vapor triples after the nz-entry pressure profile.
    bool sampled_reference_state = false;
};

struct MappedTerrainScalarTransportReport
{
    std::uint32_t scalar_flux_dispatch_count = 0;
    std::uint32_t positive_definite_limiter_dispatch_count = 0;
    std::uint32_t scalar_update_dispatch_count = 0;
    bool complete = false;
};

struct MappedTerrainScalarTransportDispatch
{
    using CompletedObserver = void (*)(void* context,
                                       const float* scalar_end,
                                       const float* scalar_flux_x,
                                       const float* scalar_flux_y,
                                       const float* scalar_flux_z,
                                       const float* limiter_factors);

    // Separate gain/loss rates retain near-cancellation below the device's normal float range.
    using AppliedObserver = void (*)(void* context, const float* scalar_end,
                                    const float* flux_x, const float* flux_y, const float* flux_z,
                                    const float* gain_rate, const float* loss_rate);

    const float* scalar_begin = nullptr;
    const float* density_begin = nullptr;
    const float* density_end = nullptr;
    const float* mass_flux_x = nullptr;
    const float* mass_flux_y = nullptr;
    const float* mass_flux_zeta = nullptr;
    const float* jacobian = nullptr;
    float* scalar_end = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    bool positive_definite = false;
    bool                                weno                       = true;
    bool                                preserve_positive_support  = false;
    float                               dx                         = 0.0f;
    float dy = 0.0f;
    float dzeta = 0.0f;
    float dt = 0.0f;
    float density_floor = 0.0f;
    MappedTerrainScalarTransportReport* report = nullptr;
    CompletedObserver completed_observer = nullptr;
    void* completed_observer_context = nullptr;
    AppliedObserver applied_observer = nullptr;
    void* applied_observer_context = nullptr;
};

enum class MappedTerrainStateUpdateMode : std::uint32_t
{
    Slow = 0,
    ContinuityEos = 1,
    FastMomentum = 2,
};

struct MappedTerrainStateUpdateReport
{
    std::uint32_t dispatch_count = 0;
    MappedTerrainStateUpdateMode mode = MappedTerrainStateUpdateMode::Slow;
    bool complete = false;
};

struct MappedTerrainStateUpdateDispatch
{
    using CompletedObserver = void (*)(void* context,
                                       const float* u,
                                       const float* v,
                                       const float* w,
                                       const float* rho,
                                       const float* p);

    const float* u = nullptr;
    const float* v = nullptr;
    const float* w = nullptr;
    const float* rho = nullptr;
    const float* p = nullptr;
    const float* theta = nullptr;
    const float* du_dt = nullptr;
    const float* dv_dt = nullptr;
    const float* dw_dt = nullptr;
    const float* drho_dt = nullptr;
    const float* dp_dt = nullptr;
    float* u_candidate = nullptr;
    float* v_candidate = nullptr;
    float* w_candidate = nullptr;
    float* rho_candidate = nullptr;
    float* p_candidate = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    MappedTerrainStateUpdateMode mode = MappedTerrainStateUpdateMode::Slow;
    float dt = 0.0f;
    MappedTerrainStateUpdateReport* report = nullptr;
    CompletedObserver completed_observer = nullptr;
    void* completed_observer_context = nullptr;
};

struct MappedTerrainFluxAccumulationReport
{
    std::uint32_t dispatch_count = 0;
    bool complete = false;
};

struct MappedTerrainFluxAccumulationDispatch
{
    using CompletedObserver = void (*)(void* context,
                                       const float* mass_flux_x,
                                       const float* mass_flux_y,
                                       const float* mass_flux_zeta);

    const float* accumulated_x = nullptr;
    const float* accumulated_y = nullptr;
    const float* accumulated_zeta = nullptr;
    const float* step_x = nullptr;
    const float* step_y = nullptr;
    const float* step_zeta = nullptr;
    float* accumulated_x_candidate = nullptr;
    float* accumulated_y_candidate = nullptr;
    float* accumulated_zeta_candidate = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float time_weight = 0.0f;
    MappedTerrainFluxAccumulationReport* report = nullptr;
    CompletedObserver completed_observer = nullptr;
    void* completed_observer_context = nullptr;
};

struct MappedTerrainFinalStateReport
{
    std::uint32_t eos_dispatch_count = 0;
    std::uint32_t boundary_dispatch_count = 0;
    bool complete = false;
};

struct MappedTerrainFinalStateDispatch
{
    using CompletedObserver = void (*)(void* context,
                                       const float* u,
                                       const float* v,
                                       const float* w,
                                       const float* rho,
                                       const float* p);

    const float* u = nullptr;
    const float* v = nullptr;
    const float* w = nullptr;
    const float* rho = nullptr;
    const float* p = nullptr;
    const float* theta = nullptr;
    float* u_final = nullptr;
    float* v_final = nullptr;
    float* w_final = nullptr;
    float* rho_final = nullptr;
    float*                         p_final                    = nullptr;
    int                            nx                         = 0;
    int                            ny                         = 0;
    int                            nz                         = 0;
    MappedTerrainFinalStateReport* report                     = nullptr;
    CompletedObserver              completed_observer         = nullptr;
    void*                          completed_observer_context = nullptr;
    bool                           apply_eos                  = true;
    int west = 0, east = 0, south = 0, north = 0;
    int bottom = 1, top = 0;
    float frame_u = 0, frame_v = 0;
};

class TerrainComputeKernels
{
public:
    virtual bool supports_held_sources() const
    {
        return false;
    }
    virtual bool dispatch_held_sources(const HeldSourcesDispatch&)
    {
        return false;
    }
    virtual bool supports_column_remap() const
    {
        return false;
    }
    virtual bool dispatch_column_remap(const ColumnRemapDispatch&)
    {
        return false;
    }
    virtual bool supports_reference_atmosphere() const
    {
        return false;
    }
    virtual bool dispatch_reference_atmosphere(const ReferenceAtmosphereDispatch&)
    {
        return false;
    }

    virtual bool supports_moving_energy() const
    {
        return false;
    }
    virtual bool dispatch_moving_energy(const MovingEnergyDispatch&)
    {
        return false;
    }

    virtual bool supports_moving_momentum() const
    {
        return false;
    }
    virtual bool dispatch_moving_momentum(const MovingMomentumDispatch&)
    {
        return false;
    }

    virtual bool supports_earth_geometry() const
    {
        return false;
    }
    virtual bool dispatch_earth_geometry(const EarthGeometryDispatch&)
    {
        return false;
    }

    virtual bool supports_conservative_remap() const
    {
        return false;
    }
    virtual bool dispatch_conservative_remap(const ConservativeRemapDispatch&)
    {
        return false;
    }

    virtual ~TerrainComputeKernels() = default;
    virtual bool supports_mapped_mass_continuity() const
    {
        return false;
    }
    virtual bool dispatch_mapped_mass_continuity(const MappedTerrainMassContinuityDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    virtual bool supports_mapped_fast_momentum() const { return false; }
    virtual bool dispatch_mapped_fast_momentum(
        const MappedTerrainFastMomentumDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    virtual bool supports_mapped_slow_tendencies() const { return false; }
    virtual bool dispatch_mapped_slow_tendencies(
        const MappedTerrainSlowTendenciesDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    virtual bool supports_mapped_scalar_transport() const { return false; }
    virtual bool dispatch_mapped_scalar_transport(
        const MappedTerrainScalarTransportDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    virtual bool supports_mapped_final_state() const { return false; }
    virtual bool dispatch_mapped_final_state(
        const MappedTerrainFinalStateDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    virtual bool supports_mapped_state_update() const { return false; }
    virtual bool dispatch_mapped_state_update(
        const MappedTerrainStateUpdateDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    virtual bool supports_mapped_flux_accumulation() const { return false; }
    virtual bool dispatch_mapped_flux_accumulation(
        const MappedTerrainFluxAccumulationDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
};
