#pragma once
#include <array>
#include <vector>
class TerrainComputeKernels;
namespace tmv::held_remap_detail
{
enum class Storage
{
    Float,
    Double,
    Slab,
    Rain
};
struct Columns
{
    std::array<std::vector<double>, 3> values;
    std::vector<float>                 rounded, residual;
};
Columns remap_native(TerrainComputeKernels& kernels,
                     const Columns&         source,
                     int                    nx,
                     int                    ny,
                     int                    levels,
                     double                 sx,
                     double                 sy,
                     Storage                storage);
} // namespace tmv::held_remap_detail
