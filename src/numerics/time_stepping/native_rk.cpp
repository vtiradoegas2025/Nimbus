#include "numerics/time_stepping/native_rk.hpp"
#include "compute/backend_interface.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace tmv
{
const NativeRkTendencyWorkspace::Fields&
NativeRkTendencyWorkspace::run(ComputeBackend& backend, RkTendencyDispatch dispatch, std::size_t bank)
{
    auto& outputs = banks_.at(bank);
    for (std::size_t f = 0; f < outputs.size(); ++f)
    {
        auto& field = outputs[f];
        if (field.size_r() != dispatch.nx || field.size_th() != dispatch.ny || field.size_z() != dispatch.nz)
            field.resize(dispatch.nx, dispatch.ny, dispatch.nz, 0);
        dispatch.output[f] = field.data();
    }
    for (auto& input : dispatch.input)
        if (!input)
            input = dispatch.input[0];
    const auto status = backend.dispatch_rk_tendency(dispatch);
    if (status != RkCombinationStatus::Completed)
        throw std::runtime_error("native RK tendency dispatch failed: operation=" +
                                 std::to_string(int(dispatch.operation)) + ", status=" + std::to_string(int(status)));
    return outputs;
}
void NativeRkTendencyWorkspace::copy(Field3D& destination, const Field3D& source)
{
    if (destination.size_r() != source.size_r() || destination.size_th() != source.size_th() ||
        destination.size_z() != source.size_z())
        throw std::invalid_argument("native RK tendency output shape mismatch");
    std::copy_n(source.data(), source.size(), destination.data());
}
} // namespace tmv
