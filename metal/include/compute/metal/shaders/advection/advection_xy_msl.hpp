#pragma once

namespace nimbus::metal::shaders
{

inline constexpr const char* kAdvectionXMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct PushConstants
{
    int nr; int nth; int nz;
    float dx; float dt; float pad0; float pad1; float pad2;
};

kernel void advect_x(
    device const float* src       [[buffer(0)]],
    device const float* velocity  [[buffer(1)]],
    device float* dst             [[buffer(2)]],
    constant PushConstants& p     [[buffer(3)]],
    uint gid                      [[thread_position_in_grid]])
{
    int interior_nr = p.nr - 2;
    int interior_nz = p.nz - 2;
    uint total = uint(interior_nr) * uint(p.nth) * uint(interior_nz);
    if (gid >= total) return;
    int k = int(gid % uint(interior_nz)) + 1;
    uint ij = gid / uint(interior_nz);
    int j = int(ij % uint(p.nth));
    int i = int(ij / uint(p.nth)) + 1;
    uint c = uint((i * p.nth + j) * p.nz + k);
    uint previous = uint(((i - 1) * p.nth + j) * p.nz + k);
    uint next = uint(((i + 1) * p.nth + j) * p.nz + k);
    float speed = velocity[c];
    float center = src[c];
    float gradient = 0.0f;
    if (speed > 0.0f) gradient = (center - src[previous]) / p.dx;
    else if (speed < 0.0f) gradient = (src[next] - center) / p.dx;
    dst[c] = center - p.dt * speed * gradient;
}
)msl";

inline constexpr const char* kAdvectionYMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct PushConstants
{
    int nr; int nth; int nz;
    float dy; float dt; float pad0; float pad1; float pad2;
};

kernel void advect_y(
    device const float* src       [[buffer(0)]],
    device const float* velocity  [[buffer(1)]],
    device float* dst             [[buffer(2)]],
    constant PushConstants& p     [[buffer(3)]],
    uint gid                      [[thread_position_in_grid]])
{
    int interior_nth = p.nth - 2;
    int interior_nz = p.nz - 2;
    uint total = uint(p.nr) * uint(interior_nth) * uint(interior_nz);
    if (gid >= total) return;
    int k = int(gid % uint(interior_nz)) + 1;
    uint ij = gid / uint(interior_nz);
    int j = int(ij % uint(interior_nth)) + 1;
    int i = int(ij / uint(interior_nth));
    uint c = uint((i * p.nth + j) * p.nz + k);
    uint previous = uint((i * p.nth + (j - 1)) * p.nz + k);
    uint next = uint((i * p.nth + (j + 1)) * p.nz + k);
    float speed = velocity[c];
    float center = src[c];
    float gradient = 0.0f;
    if (speed > 0.0f) gradient = (center - src[previous]) / p.dy;
    else if (speed < 0.0f) gradient = (src[next] - center) / p.dy;
    dst[c] = center - p.dt * speed * gradient;
}
)msl";

} // namespace nimbus::metal::shaders
