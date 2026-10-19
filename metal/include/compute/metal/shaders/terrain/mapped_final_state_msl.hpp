#pragma once

namespace nimbus::metal::shaders
{

inline constexpr const char* kMappedFinalStateMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct MappedFinalConstants { int nx; int ny; int nz; int pad0; };

inline uint mapped_final_at(int i, int j, int k, int ny, int nz)
{ return uint((i * ny + j) * nz + k); }

kernel void mapped_eos(
    device const float* density [[buffer(0)]],
    device const float* theta [[buffer(1)]],
    device float* pressure [[buffer(2)]],
    constant MappedFinalConstants& c [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint count=uint(c.nx)*uint(c.ny)*uint(c.nz);if(gid>=count)return;
    const float base=density[gid]*287.0f*theta[gid]/100000.0f;
    pressure[gid]=clamp(100000.0f*pow(base,1.40027894f),1000.0f,200000.0f);
}

)msl";

} // namespace nimbus::metal::shaders
