#pragma once

#include "compute/metal/shaders/terrain/mapped_scalar_math_msl.hpp"

namespace nimbus::metal::shaders
{

inline constexpr const char* kMappedScalarTransportMsl = R"msl(


#include <metal_stdlib>
using namespace metal;

)msl"
NIMBUS_MAPPED_SCALAR_MATH_MSL
R"msl(

struct LimiterConstants
{
    int   nx;
    int   ny;
    int   nz;
    int   pad0;
    float inv_dx;
    float inv_dy;
    float inv_dzeta;
    float dt;
    float pad1;
    float pad2;
    float pad3;
    float pad4;
};
struct UpdateConstants
{
    int   nx;
    int   ny;
    int   nz;
    int   positive_definite;
    float inv_dx;
    float inv_dy;
    float inv_dzeta;
    float dt;
    float density_floor;
    float pad1;
    float pad2;
    float pad3;
};

inline uint mapped_scalar_at(int i, int j, int k, int ny, int nz)
{
    return uint((i * ny + j) * nz + k);
}
inline float mapped_bounded_factor(float value, bool support_limit)
{
    if (value > 0.0f && (value < 1.0f || support_limit))
    {
        float bounded = value * (1.0f - 32.0f * 1.1920928955078125e-7f);
        if (bounded > 0.0f)
            bounded = as_type<float>(as_type<uint>(bounded) - 1u);
        return bounded;
    }
    return value;
}
kernel void mapped_scalar_limiter(device const float*        q [[buffer(0)]],
                                  device const float*        rho_begin [[buffer(1)]],
                                  device const float*        sfx [[buffer(2)]],
                                  device const float*        sfy [[buffer(3)]],
                                  device const float*        sfz [[buffer(4)]],
                                  device const float*        jacobian [[buffer(5)]],
                                  device float*              limiter [[buffer(6)]],
                                  constant LimiterConstants& p [[buffer(7)]],
                                  uint                       gid [[thread_position_in_grid]])
{
    const uint total = uint(p.nx) * uint(p.ny) * uint(p.nz);
    if (gid >= total)
        return;
    const int   k     = int(gid % uint(p.nz));
    const uint  ij    = gid / uint(p.nz);
    const int   j     = int(ij % uint(p.ny));
    const int   i     = int(ij / uint(p.ny));
    const auto  at    = [&](int ii, int jj, int kk) { return mapped_scalar_at(ii, jj, kk, p.ny, p.nz); };
    const float inv_j = 1.0f / jacobian[gid];
    float       rate  = 0.0f, flux;
    if (i <= p.nx - 2 && sfx[gid] > 0.0f)
        rate += sfx[gid] * p.inv_dx * inv_j;
    if (i >= 1)
    {
        flux = sfx[at(i - 1, j, k)];
        if (flux < 0.0f)
            rate -= flux * p.inv_dx * inv_j;
    }
    if (j <= p.ny - 2 && sfy[gid] > 0.0f)
        rate += sfy[gid] * p.inv_dy * inv_j;
    if (j >= 1)
    {
        flux = sfy[at(i, j - 1, k)];
        if (flux < 0.0f)
            rate -= flux * p.inv_dy * inv_j;
    }
    if (sfz[gid] > 0.0f)
        rate += sfz[gid] * p.inv_dzeta * inv_j;
    if (k >= 1)
    {
        flux = sfz[at(i, j, k - 1)];
        if (flux < 0.0f)
            rate -= flux * p.inv_dzeta * inv_j;
    }
    float factor = 1.0f;
    if (rate > 0.0f)
    {
        const float available = rho_begin[gid] * max(q[gid], 0.0f) / p.dt;
        factor                = mapped_bounded_factor(min(1.0f, available / rate), p.pad0 != 0 && available <= rate);
    }
    limiter[gid] = factor;
}

kernel void mapped_scalar_update(device const float*       q [[buffer(0)]],
                                 device const float*       rho_begin [[buffer(1)]],
                                 device const float*       rho_end [[buffer(2)]],
                                 device const float*       sfx [[buffer(3)]],
                                 device const float*       sfy [[buffer(4)]],
                                 device const float*       sfz [[buffer(5)]],
                                 device const float*       jacobian [[buffer(6)]],
                                 device const float*       limiter [[buffer(7)]],
                                 device float*             out [[buffer(8)]],
                                 device float*             applied_x [[buffer(9)]],
                                 device float*             applied_y [[buffer(10)]],
                                 device float*             applied_z [[buffer(11)]],
                                 device float*             requested [[buffer(12)]],
                                 device float*             loss [[buffer(13)]],
                                 constant UpdateConstants& p [[buffer(14)]],
                                 uint                      gid [[thread_position_in_grid]])
{
    const uint total = uint(p.nx) * uint(p.ny) * uint(p.nz);
    if (gid >= total)
        return;
    const int  k       = int(gid % uint(p.nz));
    const uint ij      = gid / uint(p.nz);
    const int  j       = int(ij % uint(p.ny));
    const int  i       = int(ij / uint(p.ny));
    const auto at      = [&](int ii, int jj, int kk) { return mapped_scalar_at(ii, jj, kk, p.ny, p.nz); };
    const auto limited = [&](float flux, uint low, uint high)
    { return p.positive_definite == 0 || flux == 0.0f ? flux : flux * limiter[flux > 0.0f ? low : high]; };
    const uint upper = k + 1 < p.nz ? at(i, j, k + 1) : gid;
    applied_x[gid]   = i + 1 < p.nx ? limited(sfx[gid], gid, at(i + 1, j, k)) : 0.0f;
    applied_y[gid]   = j + 1 < p.ny ? limited(sfy[gid], gid, at(i, j + 1, k)) : 0.0f;
    applied_z[gid]   = limited(sfz[gid], gid, upper);
    requested[gid]   = 0.0f;
    loss[gid]        = 0.0f;
    if (i <= 0 || i >= p.nx - 1 || j <= 0 || j >= p.ny - 1)
    {
        out[gid] = q[gid];
        return;
    }
    const uint  west = at(i - 1, j, k), south = at(i, j - 1, k), east = at(i + 1, j, k), north = at(i, j + 1, k);
    const float xp = limited(sfx[gid], gid, east), xm = limited(sfx[west], west, gid);
    const float yp = limited(sfy[gid], gid, north), ym = limited(sfy[south], south, gid);
    const float zp    = limited(sfz[gid], gid, upper);
    const float zm    = k > 0 ? limited(sfz[at(i, j, k - 1)], at(i, j, k - 1), gid) : 0.0f;
    const float inv_j = 1.0f / jacobian[gid];
    if (p.positive_definite != 0)
    {
        float outgoing = 0.0f, incoming = 0.0f;
        if (xp >= 0.0f)
            outgoing += xp * p.inv_dx * inv_j;
        else
            incoming -= xp * p.inv_dx * inv_j;
        if (xm <= 0.0f)
            outgoing -= xm * p.inv_dx * inv_j;
        else
            incoming += xm * p.inv_dx * inv_j;
        if (yp >= 0.0f)
            outgoing += yp * p.inv_dy * inv_j;
        else
            incoming -= yp * p.inv_dy * inv_j;
        if (ym <= 0.0f)
            outgoing -= ym * p.inv_dy * inv_j;
        else
            incoming += ym * p.inv_dy * inv_j;
        if (zp >= 0.0f)
            outgoing += zp * p.inv_dzeta * inv_j;
        else
            incoming -= zp * p.inv_dzeta * inv_j;
        if (k > 0)
        {
            if (zm <= 0.0f)
                outgoing -= zm * p.inv_dzeta * inv_j;
            else
                incoming += zm * p.inv_dzeta * inv_j;
        }
        requested[gid] = incoming;
        loss[gid]      = outgoing;
        out[gid] = positive_mapped_scalar_update(
            q[gid], rho_begin[gid], max(rho_end[gid], p.density_floor), p.dt, incoming, outgoing);
    }
    else
    {
        const float divergence = ((xp - xm) * p.inv_dx + (yp - ym) * p.inv_dy + (zp - zm) * p.inv_dzeta) * inv_j;
        requested[gid]         = -divergence;
        out[gid] = signed_mapped_scalar_update(q[gid], rho_begin[gid], max(rho_end[gid], p.density_floor), p.dt, divergence);
        return;
    }
}
)msl";

} // namespace nimbus::metal::shaders
