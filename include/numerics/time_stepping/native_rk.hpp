#pragma once
#include "core/field/field3d.hpp"
#include <array>
#include <cstddef>

class ComputeBackend;
struct RkTendencyDispatch;

namespace tmv
{
class NativeRkTendencyWorkspace
{
public:
    using Fields = std::array<Field3D, 5>;
    const Fields& run(ComputeBackend& backend, RkTendencyDispatch dispatch, std::size_t bank = 0);
    static void   copy(Field3D& destination, const Field3D& source);

private:
    std::array<Fields, 3> banks_;
};
} // namespace tmv
