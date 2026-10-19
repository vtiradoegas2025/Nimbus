#pragma once

#include "compute/metal/shaders/advection/weno_common_msl.hpp"
#include "compute/metal/shaders/advection/positive_scalar_budget_msl.hpp"

namespace nimbus::metal::shaders
{

// Native flat Cartesian C-grid transport. Mass fluxes are produced once and
// shared across every scalar. The scalar face-flux, donor limiter, and update
// phases are fused per output cell; this is algebraically identical to the
// canonical Vulkan stages but avoids materializing four transient fields per
// scalar during the Metal bootstrap.
inline constexpr const char* kConservativeCgridTransportMsl = R"msl(


#include <metal_stdlib>
using namespace metal;
)msl" NIMBUS_POSITIVE_SCALAR_BUDGET_MSL R"msl(

struct MassConstants
{
    int   nx;
    int   ny;
    int   nz;
    int   pad0;
    float frame_u;
    float frame_v;
};
struct UpdateConstants
{
    int   nx;
    int   ny;
    int   nz;
    int   use_pd;
    float inv_dx;
    float inv_dy;
    float inv_dz;
    float dt_step;
    float rho_floor;
    int   use_coupled_mass;
    int capture_offset;
    int capture_i_begin;
    int capture_i_end;
    int capture_j_begin;
    int capture_j_end;
    int capture_k_begin;
    int capture_k_end;
    int report_offset;
    int weno;
};

inline uint at3(int i, int j, int k, int ny, int nz)
{
    return uint((i * ny + j) * nz + k);
}

kernel void conservative_mass_flux(device const float*     u [[buffer(0)]],
                                   device const float*     v [[buffer(1)]],
                                   device const float*     w [[buffer(2)]],
                                   device const float*     rho [[buffer(3)]],
                                   device float*           mfx [[buffer(4)]],
                                   device float*           mfy [[buffer(5)]],
                                   device float*           mfz [[buffer(6)]],
                                   constant MassConstants& p [[buffer(7)]],
                                   uint                    gid [[thread_position_in_grid]])
{
    uint total = uint(p.nx) * uint(p.ny) * uint(p.nz);
    if (gid >= total)
        return;
    int  k   = int(gid % uint(p.nz));
    uint ij  = gid / uint(p.nz);
    int  j   = int(ij % uint(p.ny));
    int  i   = int(ij / uint(p.ny));
    mfx[gid] = i <= p.nx - 2 ? 0.5f * (rho[gid] + rho[at3(i + 1, j, k, p.ny, p.nz)]) * (u[gid] - p.frame_u) : 0.0f;
    mfy[gid] = j <= p.ny - 2 ? 0.5f * (rho[gid] + rho[at3(i, j + 1, k, p.ny, p.nz)]) * (v[gid] - p.frame_v) : 0.0f;
    mfz[gid] = k <= p.nz - 2 ? 0.5f * (rho[gid] + rho[at3(i, j, k + 1, p.ny, p.nz)]) * w[gid] : 0.0f;
}

inline float minmod3(float a, float b, float c)
{
    if (a > 0.0f && b > 0.0f && c > 0.0f)
        return min(a, min(b, c));
    if (a < 0.0f && b < 0.0f && c < 0.0f)
        return max(a, max(b, c));
    return 0.0f;
}
inline float mc_slope(float qm, float q0, float qp)
{
    float right = qp - q0;
    float left  = q0 - qm;
    return minmod3(0.5f * (right + left), 2.0f * right, 2.0f * left);
}
)msl" NIMBUS_WENO_COMMON_MSL R"msl(

inline bool pd_nonzero_underflow_product(float a, float b)
{
    return (as_type<uint>(a) & 0x7fffffffu) != 0u
        && (as_type<uint>(b) & 0x7fffffffu) != 0u && pd_product_protected(a, b);
}
inline float pd_raw_flux(float mass, float base, float offset, constant UpdateConstants& p)
{
    if (p.use_pd != 0 && (pd_nonzero_underflow_product(mass, base)
        || pd_nonzero_underflow_product(mass, offset)))
        return pd_add(pd_product(mass, base), pd_product(mass, offset));
    return fma(mass, base, mass * offset);
}
inline float flux_x(device const float* q, device const float* mfx, int i, int j, int k, constant UpdateConstants& p)
{
    if (i > p.nx - 2)
        return 0.0f;
    uint  c  = at3(i, j, k, p.ny, p.nz);
    float mf = mfx[c], face, base;
    if (mf >= 0.0f)
    {
        float q0 = q[c];
        base     = q0;
        face     = ((p.weno & 1) != 0 && i >= 2 && i + 2 <= p.nx - 1)
                       ? weno5_offset(q[at3(i - 2, j, k, p.ny, p.nz)],
                                      q[at3(i - 1, j, k, p.ny, p.nz)],
                                      q0,
                                      q[at3(i + 1, j, k, p.ny, p.nz)],
                                      q[at3(i + 2, j, k, p.ny, p.nz)])
                       : 0.5f * (i >= 1 ? mc_slope(q[at3(i - 1, j, k, p.ny, p.nz)], q0, q[at3(i + 1, j, k, p.ny, p.nz)])
                                        : 0.0f);
    }
    else
    {
        float q0 = q[at3(i + 1, j, k, p.ny, p.nz)];
        base     = q0;
        face     = ((p.weno & 1) != 0 && i - 1 >= 0 && i + 3 <= p.nx - 1)
                       ? weno5_offset(q[at3(i + 3, j, k, p.ny, p.nz)],
                                      q[at3(i + 2, j, k, p.ny, p.nz)],
                                      q0,
                                      q[c],
                                      q[at3(i - 1, j, k, p.ny, p.nz)])
                       : -0.5f * (i + 1 <= p.nx - 2 ? mc_slope(q[c], q0, q[at3(i + 2, j, k, p.ny, p.nz)]) : 0.0f);
    }
    if ((p.weno & 2) != 0 && (base <= 0.0f || base + face <= 0.0f))
        face = 0.0f;
    return pd_raw_flux(mf, base, face, p);
}
inline float flux_y(device const float* q, device const float* mfy, int i, int j, int k, constant UpdateConstants& p)
{
    if (j > p.ny - 2)
        return 0.0f;
    uint  c  = at3(i, j, k, p.ny, p.nz);
    float mf = mfy[c], face, base;
    if (mf >= 0.0f)
    {
        float q0 = q[c];
        base     = q0;
        face     = ((p.weno & 1) != 0 && j >= 2 && j + 2 <= p.ny - 1)
                       ? weno5_offset(q[at3(i, j - 2, k, p.ny, p.nz)],
                                      q[at3(i, j - 1, k, p.ny, p.nz)],
                                      q0,
                                      q[at3(i, j + 1, k, p.ny, p.nz)],
                                      q[at3(i, j + 2, k, p.ny, p.nz)])
                       : 0.5f * (j >= 1 ? mc_slope(q[at3(i, j - 1, k, p.ny, p.nz)], q0, q[at3(i, j + 1, k, p.ny, p.nz)])
                                        : 0.0f);
    }
    else
    {
        float q0 = q[at3(i, j + 1, k, p.ny, p.nz)];
        base     = q0;
        face     = ((p.weno & 1) != 0 && j - 1 >= 0 && j + 3 <= p.ny - 1)
                       ? weno5_offset(q[at3(i, j + 3, k, p.ny, p.nz)],
                                      q[at3(i, j + 2, k, p.ny, p.nz)],
                                      q0,
                                      q[c],
                                      q[at3(i, j - 1, k, p.ny, p.nz)])
                       : -0.5f * (j + 1 <= p.ny - 2 ? mc_slope(q[c], q0, q[at3(i, j + 2, k, p.ny, p.nz)]) : 0.0f);
    }
    if ((p.weno & 2) != 0 && (base <= 0.0f || base + face <= 0.0f))
        face = 0.0f;
    return pd_raw_flux(mf, base, face, p);
}
inline float flux_z(device const float* q, device const float* mfz, int i, int j, int k, constant UpdateConstants& p)
{
    uint  c  = at3(i, j, k, p.ny, p.nz);
    float mf = mfz[c];
    if (k == p.nz - 1)
        return mf * q[c];
    float face, base;
    if (mf >= 0.0f)
    {
        float q0 = q[c];
        base     = q0;
        face     = ((p.weno & 1) != 0 && k >= 2 && k + 2 <= p.nz - 1)
                       ? weno5_offset(q[at3(i, j, k - 2, p.ny, p.nz)],
                                      q[at3(i, j, k - 1, p.ny, p.nz)],
                                      q0,
                                      q[at3(i, j, k + 1, p.ny, p.nz)],
                                      q[at3(i, j, k + 2, p.ny, p.nz)])
                       : 0.5f * (k >= 1 ? mc_slope(q[at3(i, j, k - 1, p.ny, p.nz)], q0, q[at3(i, j, k + 1, p.ny, p.nz)])
                                        : 0.0f);
    }
    else
    {
        float q0 = q[at3(i, j, k + 1, p.ny, p.nz)];
        base     = q0;
        face     = ((p.weno & 1) != 0 && k - 1 >= 0 && k + 3 <= p.nz - 1)
                       ? weno5_offset(q[at3(i, j, k + 3, p.ny, p.nz)],
                                      q[at3(i, j, k + 2, p.ny, p.nz)],
                                      q0,
                                      q[c],
                                      q[at3(i, j, k - 1, p.ny, p.nz)])
                       : -0.5f * (k + 1 <= p.nz - 2 ? mc_slope(q[c], q0, q[at3(i, j, k + 2, p.ny, p.nz)]) : 0.0f);
    }
    if ((p.weno & 2) != 0 && (base <= 0.0f || base + face <= 0.0f))
        face = 0.0f;
    return pd_raw_flux(mf, base, face, p);
}

inline float donor_factor(device const float*       q,
                          device const float*       rho,
                          device const float*       rho_begin,
                          device const float*       mfx,
                          device const float*       mfy,
                          device const float*       mfz,
                          int                       i,
                          int                       j,
                          int                       k,
                          constant UpdateConstants& p)
{
    uint c = at3(i, j, k, p.ny, p.nz);
    float xp = i <= p.nx - 2 ? flux_x(q, mfx, i, j, k, p) : 0.0f;
    float xm = i >= 1 ? flux_x(q, mfx, i - 1, j, k, p) : 0.0f;
    float yp = j <= p.ny - 2 ? flux_y(q, mfy, i, j, k, p) : 0.0f;
    float ym = j >= 1 ? flux_y(q, mfy, i, j - 1, k, p) : 0.0f;
    float zp = flux_z(q, mfz, i, j, k, p);
    float zm = k >= 1 ? flux_z(q, mfz, i, j, k - 1, p) : 0.0f;
    float rate = 0.0f;
    if (xp > 0.0f) rate += xp * p.inv_dx;
    if (xm < 0.0f) rate -= xm * p.inv_dx;
    if (yp > 0.0f) rate += yp * p.inv_dy;
    if (ym < 0.0f) rate -= ym * p.inv_dy;
    if (zp > 0.0f) rate += zp * p.inv_dz;
    if (zm < 0.0f) rate -= zm * p.inv_dz;
    float r = 1.0f;
    bool support_limit = false;
    if (rate > 0.0f)
    {
        float rb = p.use_coupled_mass != 0 ? rho_begin[c] : rho[c];
        float available = pd_positive_divide(pd_inventory(rb, q[c]), p.dt_step);
        r = min(1.0f, pd_positive_divide(available, rate));
        support_limit = (p.weno & 2) != 0 && available <= rate;
    }
    if (r > 0.0f && (r < 1.0f || support_limit))
    {
        r *= 1.0f - 32.0f * 1.1920928955078125e-7f;
        if (r > 0.0f)
            r = as_type<float>(as_type<uint>(r) - 1u);
    }
    float rb = p.use_coupled_mass != 0 ? rho_begin[c] : rho[c];
    return pd_verified_factor(r, rb, q[c], p.dt_step,
        xp, xm, yp, ym, zp, zm, p.inv_dx, p.inv_dy, p.inv_dz);
}

kernel void conservative_scalar_update(device const float*       q [[buffer(0)]],
                                       device const float*       rho [[buffer(1)]],
                                       device const float*       mfx [[buffer(2)]],
                                       device const float*       mfy [[buffer(3)]],
                                       device const float*       mfz [[buffer(4)]],
                                       device const float*       rho_begin [[buffer(5)]],
                                       device float*             out [[buffer(6)]],
                                       constant UpdateConstants& p [[buffer(7)]],
                                       device float*             interface_flux [[buffer(8)]],
                                       device float*             requested_update [[buffer(9)]],
                                       uint                      gid [[thread_position_in_grid]])
{
    uint total = uint(p.nx) * uint(p.ny) * uint(p.nz);
    if (gid >= total)
        return;
    int  k  = int(gid % uint(p.nz));
    uint ij = gid / uint(p.nz);
    int  j  = int(ij % uint(p.ny));
    int  i  = int(ij / uint(p.ny));
    if (p.report_offset >= 0)
        requested_update[gid] = 0;
    if (i < 1 || i > p.nx - 2 || j < 1 || j > p.ny - 2)
    {
        out[gid] = q[gid];
        return;
    }
    uint  im = at3(i - 1, j, k, p.ny, p.nz), jm = at3(i, j - 1, k, p.ny, p.nz);
    float mass_div = (mfx[gid] - mfx[im]) * p.inv_dx + (mfy[gid] - mfy[jm]) * p.inv_dy +
                     (mfz[gid] - (k >= 1 ? mfz[gid - 1] : 0.0f)) * p.inv_dz;
    float xc = flux_x(q, mfx, i, j, k, p), xm = flux_x(q, mfx, i - 1, j, k, p);
    float yc = flux_y(q, mfy, i, j, k, p), ym = flux_y(q, mfy, i, j - 1, k, p);
    float zc = flux_z(q, mfz, i, j, k, p), zm = k >= 1 ? flux_z(q, mfz, i, j, k - 1, p) : 0.0f;
    if (p.use_pd != 0)
    {
        float rc = donor_factor(q, rho, rho_begin, mfx, mfy, mfz, i, j, k, p);
        xc = pd_scaled_face(xc, pd_positive(xc) ? rc : donor_factor(q, rho, rho_begin, mfx, mfy, mfz, i + 1, j, k, p));
        xm = pd_scaled_face(xm, pd_positive(xm) ? donor_factor(q, rho, rho_begin, mfx, mfy, mfz, i - 1, j, k, p) : rc);
        yc = pd_scaled_face(yc, pd_positive(yc) ? rc : donor_factor(q, rho, rho_begin, mfx, mfy, mfz, i, j + 1, k, p));
        ym = pd_scaled_face(ym, pd_positive(ym) ? donor_factor(q, rho, rho_begin, mfx, mfy, mfz, i, j - 1, k, p) : rc);
        zc = pd_scaled_face(zc, pd_positive(zc) ? rc : donor_factor(q, rho, rho_begin, mfx, mfy, mfz, i, j, min(k + 1, p.nz - 1), p));
        zm = pd_scaled_face(zm, pd_positive(zm) ? donor_factor(q, rho, rho_begin, mfx, mfy, mfz, i, j, k - 1, p) : rc);
    }
    if (p.capture_offset >= 0 && k >= p.capture_k_begin && k < p.capture_k_end)
    {
        int nk = p.capture_k_end - p.capture_k_begin;
        int x_size = (p.capture_j_end - p.capture_j_begin) * nk;
        int y_size = (p.capture_i_end - p.capture_i_begin) * nk;
        int kk = k - p.capture_k_begin;
        if (j >= p.capture_j_begin && j < p.capture_j_end)
        {
            int bin = p.capture_offset + (j - p.capture_j_begin) * nk + kk;
            if (i == p.capture_i_begin)
                interface_flux[bin] = xm;
            if (i == p.capture_i_end - 1)
                interface_flux[bin + x_size] = xc;
        }
        if (i >= p.capture_i_begin && i < p.capture_i_end)
        {
            int bin = p.capture_offset + 2 * x_size + (i - p.capture_i_begin) * nk + kk;
            if (j == p.capture_j_begin)
                interface_flux[bin] = ym;
            if (j == p.capture_j_end - 1)
                interface_flux[bin + y_size] = yc;
        }
    }
    if (p.report_offset >= 0)
    {
        int xs = (p.ny - 2) * p.nz;
        int ys = (p.nx - 2) * p.nz;
        int xb = p.report_offset + (j - 1) * p.nz + k;
        int yb = p.report_offset + 2 * xs + (i - 1) * p.nz + k;
        if (i == 1) interface_flux[xb] = xm;
        if (i == p.nx - 2) interface_flux[xb + xs] = xc;
        if (j == 1) interface_flux[yb] = ym;
        if (j == p.ny - 2) interface_flux[yb + ys] = yc;
        if (k == p.nz - 1)
            interface_flux[p.report_offset + 2 * (xs + ys) + (i - 1) * (p.ny - 2) + j - 1] = zc;
    }
    float rb = p.use_coupled_mass != 0 ? rho_begin[gid] : rho[gid];
    float rho_phi;
    if (p.use_pd != 0)
    {
        float outgoing = pd_rate(xc, xm, yc, ym, zc, zm, p.inv_dx, p.inv_dy, p.inv_dz);
        float incoming = pd_rate(pd_reverse(xc), pd_reverse(xm), pd_reverse(yc), pd_reverse(ym),
            pd_reverse(zc), pd_reverse(zm), p.inv_dx, p.inv_dy, p.inv_dz);
        if (p.report_offset >= 0)
            requested_update[gid] = pd_add(incoming, pd_reverse(outgoing));
        rho_phi = pd_next_inventory(pd_remaining(rb, q[gid], p.dt_step, outgoing), p.dt_step, incoming);
    }
    float rho_den = p.use_coupled_mass != 0 ? rho[gid] : rho[gid] - p.dt_step * mass_div;
    float denom   = max(rho_den, p.rho_floor);
    if (p.use_pd == 0)
    {
        float scalar_div = (xc - xm) * p.inv_dx + (yc - ym) * p.inv_dy + (zc - zm) * p.inv_dz;
        // Recover the increment without rounding a large background mass product.
        float increment = ((rb - denom) * q[gid] - p.dt_step * scalar_div) / denom;
        if (p.report_offset >= 0)
            requested_update[gid] = increment;
        out[gid] = q[gid] + increment;
    }
    else
        {
        bool zero_transfer = ((as_type<uint>(xc) | as_type<uint>(xm) | as_type<uint>(yc)
            | as_type<uint>(ym) | as_type<uint>(zc) | as_type<uint>(zm)) & 0x7fffffffu) == 0u;
        out[gid] = zero_transfer && as_type<uint>(rb) == as_type<uint>(denom)
            ? q[gid] : pd_positive_divide(rho_phi, denom);
    }
}
)msl";

} // namespace nimbus::metal::shaders
