#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/field/field3d.hpp"
#include "dynamics/cartesian_cgrid_momentum_flux.hpp"
#include "dynamics/dynamics_base.hpp"
#include "numerics/advection/advection_cartesian.hpp"

namespace tmv
{

// Geometry-dependent work is rebuilt from a level's fields. Only the
// interface register and its capture policy participate in continuation.
struct DynamicsExecutionWorkspace
{
    static constexpr std::size_t acoustic_profile_row_count = 4;
    Field3D du_dt;
    Field3D dv_dt;
    Field3D dw_dt;
    Field3D drho_dt;
    Field3D dp_dt;
    Field3D du_advection_dt;
    Field3D dv_advection_dt;
    Field3D dw_advection_dt;

    std::vector<std::array<double, 3>> slow_pressure_energy_columns;
    std::vector<float> state_relaxation_profiles;
    std::vector<float> acoustic_profile_rows;
    CartesianCGridTransportContext transport_context;
 };

struct DynamicsContinuationState
{
    CartesianCGridMomentumFluxRegister momentum_interface_flux_register;
    bool momentum_interface_flux_capture_enabled = false;

 };

struct DynamicsRuntimeState : DynamicsContinuationState, DynamicsExecutionWorkspace
{
    std::unique_ptr<DynamicsScheme> scheme;
    DynamicsRuntimeState() = default;
    DynamicsRuntimeState(const DynamicsRuntimeState& other)
        : DynamicsContinuationState(other)
    {
        if (other.scheme)
        {
            scheme = other.scheme->clone();
            if (!scheme)
                throw std::runtime_error(
                    "dynamics scheme does not support deep runtime snapshots");
        }
    }
    DynamicsRuntimeState& operator=(const DynamicsRuntimeState& other)
    {
        if (this != &other)
        {
            DynamicsRuntimeState copy(other);
            *this = std::move(copy);
        }
        return *this;
    }
    DynamicsRuntimeState(DynamicsRuntimeState&&) noexcept = default;
    DynamicsRuntimeState& operator=(DynamicsRuntimeState&&) noexcept = default;
};

extern DynamicsRuntimeState& global_dynamics_runtime_state;

// Select and own the process-compatible dynamics scheme.
void initialize_dynamics_scheme(const std::string& scheme_name);

} // namespace tmv
