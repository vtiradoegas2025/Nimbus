#include "microphysics/schemes/thompson/thompson.hpp"
#include "microphysics/schemes/thompson/completion_tables.hpp"
#include "compute/backend_runtime.hpp"
#include <stdexcept>

namespace tmv
{
bool ThompsonScheme::try_native_completion(const Field3D&                       p,
                                           const Field3D&                       rho,
                                           const std::array<const Field3D*, 7>& water,
                                           double                               dt,
                                           const NativeCloudAdjustment&         cloud,
                                           const std::array<Field3D, 10>&       warm,
                                           const std::array<Field3D, 13>&       deposition,
                                           const std::array<Field3D, 13>&       freezing,
                                           thompson::PrognosticState&           state,
                                           const std::array<Field3D*, 8>&       tendency,
                                           Field3D&                             snow_speed)
{
    auto* runtime = compute_runtime();
    auto* kernels = runtime && runtime->backend ? runtime->backend->microphysics_kernels() : nullptr;
    if (!kernels || !kernels->supports(MicrophysicsKernelOp::ThompsonCompletion))
        return false;
    const auto                          domain = physical_cell_domain(p.size_r(), p.size_th(), p.size_z());
    const std::array<const Field3D*, 5> numbers{&prognostic_state_.cloud_number(),
                                                &prognostic_state_.rain_number(),
                                                &prognostic_state_.ice_number(),
                                                &prognostic_state_.water_aerosol(),
                                                &prognostic_state_.ice_aerosol()};
    const std::array<Field3D*, 5>       number_rates{&state.cloud_number_tendency(),
                                                     &state.rain_number_tendency(),
                                                     &state.ice_number_tendency(),
                                                     &state.water_aerosol_tendency(),
                                                     &state.ice_aerosol_tendency()};
    std::vector<std::size_t>            indices;
    const auto capacity = std::min(nimbus::compute::thompson_completion_chunk_cells,
                                   static_cast<std::size_t>(domain.i_end - domain.i_begin) *
                                       (domain.j_end - domain.j_begin) * (domain.k_end - domain.k_begin));
    indices.reserve(capacity);
    std::vector<float> input(capacity * nimbus::compute::thompson_completion_input_stride), output(capacity * 18);
    ThompsonCompletionDispatch d;
    d.tables         = thompson::native_completion_tables();
    d.input          = input.data();
    d.output         = output.data();
    d.dt             = dt;
    const auto flush = [&]()
    {
        if (indices.empty())
            return;
        d.cells = indices.size();
        if (!kernels->dispatch_thompson_completion(d))
            throw std::runtime_error("native Thompson cold completion failed");
        for (std::size_t row = 0; row < indices.size(); ++row)
        {
            const auto n = indices[row];
            for (std::size_t f = 0; f < 8; ++f)
                tendency[f]->data()[n] = output[row * 18 + f];
            for (std::size_t f = 0; f < 5; ++f)
                number_rates[f]->data()[n] = output[row * 18 + 8 + f];
            snow_speed.data()[n] = output[row * 18 + 13];
        }
        indices.clear();
    };
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const auto n   = (static_cast<std::size_t>(i) * p.size_th() + j) * p.size_z() + k;
                float*     row = input.data() + indices.size() * nimbus::compute::thompson_completion_input_stride;
                row[0]         = cloud.temperature.data()[n];
                row[1]         = p.data()[n];
                row[2]         = rho.data()[n];
                for (std::size_t f = 0; f < 7; ++f)
                    row[3 + f] = water[f]->data()[n];
                for (std::size_t f = 0; f < 5; ++f)
                    row[10 + f] = numbers[f]->data()[n];
                row[15] = cloud.vapor_change.data()[n];
                row[16] = cloud.temperature_change.data()[n];
                row[17] = cloud.cloud_number_tendency.data()[n];
                row[18] = cloud.aerosol_tendency.data()[n];
                row[49] = cloud.vapor_change_low.data()[n];
                row[50] = cloud.temperature_change_low.data()[n];
                for (std::size_t f = 0; f < 9; ++f)
                    row[19 + f] = warm[f].data()[n];
                for (std::size_t f = 0; f < 11; ++f)
                    row[28 + f] = deposition[f].data()[n];
                for (std::size_t f = 0; f < 10; ++f)
                    row[39 + f] = freezing[f].data()[n];
                indices.push_back(n);
                if (indices.size() == nimbus::compute::thompson_completion_chunk_cells)
                    flush();
            }
    flush();
    return true;
}
} // namespace tmv
