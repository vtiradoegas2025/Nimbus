#pragma once

namespace nimbus::metal::shaders
{

// Runtime-compiled MSL translation of
// vulkan/shaders/compute/domain/domain_recycling_homogeneous_field.comp.
// Field order matches the canonical 44-byte HomogeneousDomainFieldShiftPushConstants ABI.
inline constexpr char kDomainRecyclingHomogeneousFieldMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;

struct HomogeneousDomainFieldShiftPushConstants
{
    int nr;
    int nth;
    int nz;
    int shift_i_cells;
    int shift_j_cells;
    int destination_i_begin;
    int destination_i_end;
    int destination_j_begin;
    int destination_j_end;
    int leading_profile_offset;
    int leading_is_field;
};

kernel void domain_recycling_homogeneous_field_shift(
    device const float* source_values [[buffer(0)]],
    device const float* leading_profile [[buffer(1)]],
    device float* destination_values [[buffer(2)]],
    constant HomogeneousDomainFieldShiftPushConstants& pc [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = uint(pc.nr) * uint(pc.nth) * uint(pc.nz);
    if (gid >= total)
    {
        return;
    }

    const int k = int(gid % uint(pc.nz));
    const uint horizontal = gid / uint(pc.nz);
    const int j = int(horizontal % uint(pc.nth));
    const int i = int(horizontal / uint(pc.nth));

    const bool retained = i >= pc.destination_i_begin
        && i < pc.destination_i_end
        && j >= pc.destination_j_begin
        && j < pc.destination_j_end;
    if (retained)
    {
        const int source_i = i + pc.shift_i_cells;
        const int source_j = j + pc.shift_j_cells;
        const uint source_index =
            (uint(source_i) * uint(pc.nth) + uint(source_j)) * uint(pc.nz) + uint(k);
        destination_values[gid] = source_values[source_index];
    }
    else
    {
        destination_values[gid] =
            leading_profile[uint(pc.leading_profile_offset) + (pc.leading_is_field != 0 ? gid : uint(k))];
    }
}
)msl";

} // namespace nimbus::metal::shaders
