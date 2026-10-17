#include "cpu_backend.hpp"
#include "core/runtime/simulation_state.hpp"

#include "numerics/advection/advection_cartesian.hpp"

#include <array>
#include <cstddef>

namespace nimbus::compute::cpu_internal
{

bool CpuComputeBackend::dispatch_advect_scalars_conservative_cartesian_cgrid(
    const ConservativeCartesianCgridTransportDispatch& d)
{
    constexpr int kMaxHostScalars = 16;
    if (d.n_scalars < 0 || d.n_scalars > kMaxHostScalars ||
        (d.n_scalars > 0 && (!d.host_fields || !d.host_kappas || !d.host_context || !(d.dt_step_host > 0.0))))
    {
        return false;
    }
    if (d.n_scalars == 0)
    {
        return true;
    }

    std::array<tmv::CartesianScalarTransport, kMaxHostScalars> scalars{};
    for (int s = 0; s < d.n_scalars; ++s)
    {
        Field3D* field = d.host_fields[s];
        if (field == nullptr || d.q_in == nullptr || d.q_out == nullptr || d.positive_definite == nullptr ||
            field->data() != d.q_in[s] || field->data() != d.q_out[s])
        {
            return false;
        }
        tmv::ScalarMoistEnergyTransportReportV1* energy_report =
            d.host_energy_reports != nullptr ? d.host_energy_reports[s] : nullptr;
        const tmv::ScalarMoistEnergyKind energy_kind =
            d.host_energy_kinds != nullptr ? d.host_energy_kinds[s] : tmv::ScalarMoistEnergyKind::None;
        if ((energy_report == nullptr) != (energy_kind == tmv::ScalarMoistEnergyKind::None))
        {
            return false;
        }
        if (d.preserve_positive_support && d.positive_definite[s] == 0)
            return false;
        scalars[static_cast<std::size_t>(s)] = {
            field,
            d.host_kappas[s],
            d.positive_definite[s] != 0,
            d.host_boundary_reports != nullptr ? d.host_boundary_reports[s] : nullptr,
            energy_kind,
            energy_report,
            d.host_interface_registers != nullptr ? d.host_interface_registers[s] : nullptr,
        };
    }

    tmv::advect_scalars_conservative_3d_cartesian(
        d.execution_state ? *d.execution_state : global_simulation_state(),
        scalars.data(),
        static_cast<std::size_t>(d.n_scalars),
        d.dt_step_host,
        d.host_context,
        d.preserve_positive_support
            ? (d.weno ? tmv::CartesianFaceReconstruction::WENO5PositiveSupport
                      : tmv::CartesianFaceReconstruction::MonotonicCenteredPositiveSupport)
            : (d.weno ? tmv::CartesianFaceReconstruction::WENO5 : tmv::CartesianFaceReconstruction::MonotonicCentered));
    return true;
}

} // namespace nimbus::compute::cpu_internal
