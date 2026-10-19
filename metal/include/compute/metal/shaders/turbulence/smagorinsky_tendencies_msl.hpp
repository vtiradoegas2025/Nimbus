#pragma once

namespace nimbus::metal::shaders
{

// Runtime-compiled translation of
// vulkan/shaders/compute/turbulence/smagorinsky_tendencies.comp.
// Buffer indices deliberately match the canonical Vulkan binding order.
inline constexpr const char* kSmagorinskyTendenciesMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct PushConstants
{
    int nx;
    int ny;
    int nz;
    int use_horizontal;
    int use_vertical;
    int has_qv;
    int periodic_x;
    int periodic_y;
    float dx;
    float dy;
    int closed_lateral;
};

inline uint at(int i, int j, int k, constant PushConstants& p)
{
    return uint((i * p.ny + j) * p.nz + k);
}

inline int periodic_prev(int value, int count)
{
    return value <= 1 ? count - 2 : min(value - 1, count - 2);
}

inline int periodic_next(int value, int count)
{
    return value >= count - 2 ? 1 : max(value + 1, 1);
}

inline bool finite_f(float value)
{
    return !isnan(value) && !isinf(value);
}

inline float sanitize(float value)
{
    return finite_f(value) ? value : 0.0f;
}

inline float center_distance(device const float* dz, int lower_k)
{
    return 0.5f * (dz[lower_k] + dz[lower_k + 1]);
}

inline float rho_c(device const float* rho, int i, int j, int k,
                   constant PushConstants& p)
{
    return rho[at(i, j, k, p)];
}

inline float rho_xface(device const float* rho, int i, int j, int k,
                       constant PushConstants& p)
{
    int i1 = p.periodic_x != 0 ? periodic_next(i, p.nx) : min(i + 1, p.nx - 1);
    return 0.5f * (rho_c(rho, i, j, k, p) + rho_c(rho, i1, j, k, p));
}

inline float rho_yface(device const float* rho, int i, int j, int k,
                       constant PushConstants& p)
{
    int j1 = p.periodic_y != 0 ? periodic_next(j, p.ny) : min(j + 1, p.ny - 1);
    return 0.5f * (rho_c(rho, i, j, k, p) + rho_c(rho, i, j1, k, p));
}

inline float mu_c(device const float* rho, device const float* nu,
                  int i, int j, int k, constant PushConstants& p)
{
    return rho_c(rho, i, j, k, p) * max(nu[at(i, j, k, p)], 0.0f);
}

inline float mu_xyedge(device const float* rho, device const float* nu,
                       int i, int j, int k, constant PushConstants& p)
{
    int i1 = p.periodic_x != 0 ? periodic_next(i, p.nx) : min(i + 1, p.nx - 1);
    int j1 = p.periodic_y != 0 ? periodic_next(j, p.ny) : min(j + 1, p.ny - 1);
    return 0.25f * (mu_c(rho, nu, i, j, k, p) + mu_c(rho, nu, i1, j, k, p)
                  + mu_c(rho, nu, i, j1, k, p) + mu_c(rho, nu, i1, j1, k, p));
}

inline float mu_xzedge(device const float* rho, device const float* nu,
                       device const float* dz, int i, int j, int k,
                       constant PushConstants& p)
{
    int i1 = p.periodic_x != 0 ? periodic_next(i, p.nx) : min(i + 1, p.nx - 1);
    float lower = 0.5f * (mu_c(rho, nu, i, j, k, p) + mu_c(rho, nu, i1, j, k, p));
    float upper = 0.5f * (mu_c(rho, nu, i, j, k + 1, p) + mu_c(rho, nu, i1, j, k + 1, p));
    return (dz[k + 1] * lower + dz[k] * upper) / (dz[k] + dz[k + 1]);
}

inline float mu_yzedge(device const float* rho, device const float* nu,
                       device const float* dz, int i, int j, int k,
                       constant PushConstants& p)
{
    int j1 = p.periodic_y != 0 ? periodic_next(j, p.ny) : min(j + 1, p.ny - 1);
    float lower = 0.5f * (mu_c(rho, nu, i, j, k, p) + mu_c(rho, nu, i, j1, k, p));
    float upper = 0.5f * (mu_c(rho, nu, i, j, k + 1, p) + mu_c(rho, nu, i, j1, k + 1, p));
    return (dz[k + 1] * lower + dz[k] * upper) / (dz[k] + dz[k + 1]);
}

inline float wf(device const float* w, int i, int j, int k,
                constant PushConstants& p)
{
    return k < 0 ? 0.0f : w[at(i, j, k, p)];
}

kernel void smagorinsky_tendencies(
    device const float* u_in       [[buffer(0)]],
    device const float* v_in       [[buffer(1)]],
    device const float* w_in       [[buffer(2)]],
    device const float* theta_in   [[buffer(3)]],
    device const float* qv_in      [[buffer(4)]],
    device const float* nu_t       [[buffer(5)]],
    device const float* K_theta    [[buffer(6)]],
    device const float* K_q        [[buffer(7)]],
    device const float* dz_prof    [[buffer(8)]],
    device const float* rho_in     [[buffer(9)]],
    device float* dudt_sgs         [[buffer(10)]],
    device float* dvdt_sgs         [[buffer(11)]],
    device float* dwdt_sgs         [[buffer(12)]],
    device float* dthetadt_sgs     [[buffer(13)]],
    device float* dqvdt_sgs        [[buffer(14)]],
    constant PushConstants& p      [[buffer(15)]],
    uint gid                       [[thread_position_in_grid]])
{
    uint total = uint(p.nx) * uint(p.ny) * uint(p.nz);
    if (gid >= total)
        return;

    int k = int(gid % uint(p.nz));
    uint ij = gid / uint(p.nz);
    int j = int(ij % uint(p.ny));
    int i = int(ij / uint(p.ny));
    uint c = gid;

    bool horiz = p.use_horizontal != 0;
    bool vert = p.use_vertical != 0;
    bool wrap_x = p.periodic_x != 0;
    bool wrap_y = p.periodic_y != 0;
    float dz_k = dz_prof[k];
    int jmm = wrap_y ? periodic_prev(j, p.ny) : max(j - 1, 0);
    int jpp = wrap_y ? periodic_next(j, p.ny) : min(j + 1, p.ny - 1);
    int imm = wrap_x ? periodic_prev(i, p.nx) : max(i - 1, 0);
    int ipp = wrap_x ? periodic_next(i, p.nx) : min(i + 1, p.nx - 1);
    bool j_interior = j >= 1 && j <= p.ny - 2;

    float du_out = 0.0f;
    float dv_out = 0.0f;
    float dw_out = 0.0f;
    float dth_out = 0.0f;
    float dqv_out = 0.0f;

    if (i >= 1 && i <= (wrap_x ? p.nx - 2 : p.nx - 3) &&
        j_interior && k <= p.nz - 2)
    {
        float div = 0.0f;
        if (horiz)
        {
            float S11_r = (u_in[at(ipp, j, k, p)] - u_in[c]) / p.dx;
            float S11_l = (u_in[c] - u_in[at(imm, j, k, p)]) / p.dx;
            float T11_r = 2.0f * mu_c(rho_in, nu_t, ipp, j, k, p) * S11_r;
            float T11_l = 2.0f * mu_c(rho_in, nu_t, i, j, k, p) * S11_l;
            div += (T11_r - T11_l) / p.dx;

            float T12[2];
            int je[2] = {j, jmm};
            for (int a = 0; a < 2; ++a)
            {
                int je1 = wrap_y ? periodic_next(je[a], p.ny) : min(je[a] + 1, p.ny - 1);
                float S12 = 0.5f * ((u_in[at(i, je1, k, p)] - u_in[at(i, je[a], k, p)]) / p.dy
                                  + (v_in[at(ipp, je[a], k, p)] - v_in[at(i, je[a], k, p)]) / p.dx);
                T12[a] = 2.0f * mu_xyedge(rho_in, nu_t, i, je[a], k, p) * S12;
            }
            div += (T12[0] - T12[1]) / p.dy;
        }
        if (vert)
        {
            float T13[2];
            int ke[2] = {k, k - 1};
            for (int a = 0; a < 2; ++a)
            {
                if (ke[a] < 0 || ke[a] >= p.nz - 1)
                    T13[a] = 0.0f;
                else
                {
                    float dudz = (u_in[at(i, j, ke[a] + 1, p)] - u_in[at(i, j, ke[a], p)]) /
                                 center_distance(dz_prof, ke[a]);
                    float gradient = dudz;
                    if (horiz)
                        gradient += (wf(w_in, ipp, j, ke[a], p) - wf(w_in, i, j, ke[a], p)) / p.dx;
                    T13[a] = mu_xzedge(rho_in, nu_t, dz_prof, i, j, ke[a], p) * gradient;
                }
            }
            div += (T13[0] - T13[1]) / dz_k;
        }
        du_out = sanitize(div / rho_xface(rho_in, i, j, k, p));
    }

    bool vface_j_ok = wrap_y ? (j >= 1 && j <= p.ny - 2) : (j >= 1 && j <= p.ny - 3);
    if (i >= 1 && i <= p.nx - 2 && vface_j_ok && k <= p.nz - 2)
    {
        float div = 0.0f;
        if (horiz)
        {
            float S22_n = (v_in[at(i, jpp, k, p)] - v_in[c]) / p.dy;
            float S22_s = (v_in[c] - v_in[at(i, jmm, k, p)]) / p.dy;
            float T22_n = 2.0f * mu_c(rho_in, nu_t, i, jpp, k, p) * S22_n;
            float T22_s = 2.0f * mu_c(rho_in, nu_t, i, j, k, p) * S22_s;
            div += (T22_n - T22_s) / p.dy;
            float T21[2];
            int ie[2] = {i, imm};
            int j1 = wrap_y ? periodic_next(j, p.ny) : min(j + 1, p.ny - 1);
            for (int a = 0; a < 2; ++a)
            {
                int ie1 = wrap_x ? periodic_next(ie[a], p.nx) : min(ie[a] + 1, p.nx - 1);
                float S12 = 0.5f * ((u_in[at(ie[a], j1, k, p)] - u_in[at(ie[a], j, k, p)]) / p.dy
                                  + (v_in[at(ie1, j, k, p)] - v_in[at(ie[a], j, k, p)]) / p.dx);
                T21[a] = 2.0f * mu_xyedge(rho_in, nu_t, ie[a], j, k, p) * S12;
            }
            div += (T21[0] - T21[1]) / p.dx;
        }
        if (vert)
        {
            float T23[2];
            int ke[2] = {k, k - 1};
            int j1 = wrap_y ? periodic_next(j, p.ny) : min(j + 1, p.ny - 1);
            for (int a = 0; a < 2; ++a)
            {
                if (ke[a] < 0 || ke[a] >= p.nz - 1)
                    T23[a] = 0.0f;
                else
                {
                    float dvdz = (v_in[at(i, j, ke[a] + 1, p)] - v_in[at(i, j, ke[a], p)]) /
                                 center_distance(dz_prof, ke[a]);
                    float gradient = dvdz;
                    if (horiz)
                        gradient += (wf(w_in, i, j1, ke[a], p) - wf(w_in, i, j, ke[a], p)) / p.dy;
                    T23[a] = mu_yzedge(rho_in, nu_t, dz_prof, i, j, ke[a], p) * gradient;
                }
            }
            div += (T23[0] - T23[1]) / dz_k;
        }
        dv_out = sanitize(div / rho_yface(rho_in, i, j, k, p));
    }

    if (i >= 1 && i <= p.nx - 2 && j_interior && k <= p.nz - 2)
    {
        float dz_dual = center_distance(dz_prof, k);
        float mass = 0.5f * (rho_c(rho_in, i, j, k, p) * dz_prof[k] +
                             rho_c(rho_in, i, j, k + 1, p) * dz_prof[k + 1]);
        float force = 0.0f;
        if (vert)
        {
            float S33_t = (wf(w_in, i, j, k + 1, p) - wf(w_in, i, j, k, p)) / dz_prof[k + 1];
            float S33_b = (wf(w_in, i, j, k, p) - wf(w_in, i, j, k - 1, p)) / dz_k;
            force += 2.0f * mu_c(rho_in, nu_t, i, j, k + 1, p) * S33_t
                   - 2.0f * mu_c(rho_in, nu_t, i, j, k, p) * S33_b;
        }
        if (horiz)
        {
            float T31[2];
            int ie[2] = {i, imm};
            for (int a = 0; a < 2; ++a)
            {
                int ie1 = wrap_x ? periodic_next(ie[a], p.nx) : min(ie[a] + 1, p.nx - 1);
                float gradient = (wf(w_in, ie1, j, k, p) - wf(w_in, ie[a], j, k, p)) / p.dx;
                if (vert)
                    gradient += (u_in[at(ie[a], j, k + 1, p)] - u_in[at(ie[a], j, k, p)]) / dz_dual;
                T31[a] = mu_xzedge(rho_in, nu_t, dz_prof, ie[a], j, k, p) * gradient;
            }
            force += dz_dual * (T31[0] - T31[1]) / p.dx;
            float T32[2];
            int je[2] = {j, jmm};
            for (int a = 0; a < 2; ++a)
            {
                int je1 = wrap_y ? periodic_next(je[a], p.ny) : min(je[a] + 1, p.ny - 1);
                float gradient = (wf(w_in, i, je1, k, p) - wf(w_in, i, je[a], k, p)) / p.dy;
                if (vert)
                    gradient += (v_in[at(i, je[a], k + 1, p)] - v_in[at(i, je[a], k, p)]) / dz_dual;
                T32[a] = mu_yzedge(rho_in, nu_t, dz_prof, i, je[a], k, p) * gradient;
            }
            force += dz_dual * (T32[0] - T32[1]) / p.dy;
        }
        dw_out = sanitize(force / mass);
    }

    if (i >= 1 && i <= p.nx - 2 && j_interior)
    {
        if (!wrap_x && (p.closed_lateral & 1) != 0 && i == 1)
            imm = i;
        if (!wrap_x && (p.closed_lateral & 2) != 0 && i == p.nx - 2)
            ipp = i;
        if (!wrap_y && (p.closed_lateral & 4) != 0 && j == 1)
            jmm = j;
        if (!wrap_y && (p.closed_lateral & 8) != 0 && j == p.ny - 2)
            jpp = j;
        float div_theta = 0.0f;
        if (horiz)
        {
            float Fx_e = 0.5f * (rho_c(rho_in, i, j, k, p) * K_theta[c] + rho_c(rho_in, ipp, j, k, p) * K_theta[at(ipp, j, k, p)]) * (theta_in[at(ipp, j, k, p)] - theta_in[c]) / p.dx;
            float Fx_w = 0.5f * (rho_c(rho_in, imm, j, k, p) * K_theta[at(imm, j, k, p)] + rho_c(rho_in, i, j, k, p) * K_theta[c]) * (theta_in[c] - theta_in[at(imm, j, k, p)]) / p.dx;
            float Fy_n = 0.5f * (rho_c(rho_in, i, j, k, p) * K_theta[c] + rho_c(rho_in, i, jpp, k, p) * K_theta[at(i, jpp, k, p)]) * (theta_in[at(i, jpp, k, p)] - theta_in[c]) / p.dy;
            float Fy_s = 0.5f * (rho_c(rho_in, i, jmm, k, p) * K_theta[at(i, jmm, k, p)] + rho_c(rho_in, i, j, k, p) * K_theta[c]) * (theta_in[c] - theta_in[at(i, jmm, k, p)]) / p.dy;
            div_theta += (Fx_e - Fx_w) / p.dx + (Fy_n - Fy_s) / p.dy;
        }
        if (vert)
        {
            float Fz_t = k < p.nz - 1 ? 0.5f * (rho_c(rho_in, i, j, k, p) * K_theta[c] + rho_c(rho_in, i, j, k + 1, p) * K_theta[at(i, j, k + 1, p)]) * (theta_in[at(i, j, k + 1, p)] - theta_in[c]) / center_distance(dz_prof, k) : 0.0f;
            float Fz_b = k > 0 ? 0.5f * (rho_c(rho_in, i, j, k - 1, p) * K_theta[at(i, j, k - 1, p)] + rho_c(rho_in, i, j, k, p) * K_theta[c]) * (theta_in[c] - theta_in[at(i, j, k - 1, p)]) / center_distance(dz_prof, k - 1) : 0.0f;
            div_theta += (Fz_t - Fz_b) / dz_k;
        }
        dth_out = sanitize(div_theta / rho_c(rho_in, i, j, k, p));

        if (p.has_qv != 0)
        {
            float div_qv = 0.0f;
            if (horiz)
            {
                float Fx_e = 0.5f * (rho_c(rho_in, i, j, k, p) * K_q[c] + rho_c(rho_in, ipp, j, k, p) * K_q[at(ipp, j, k, p)]) * (qv_in[at(ipp, j, k, p)] - qv_in[c]) / p.dx;
                float Fx_w = 0.5f * (rho_c(rho_in, imm, j, k, p) * K_q[at(imm, j, k, p)] + rho_c(rho_in, i, j, k, p) * K_q[c]) * (qv_in[c] - qv_in[at(imm, j, k, p)]) / p.dx;
                float Fy_n = 0.5f * (rho_c(rho_in, i, j, k, p) * K_q[c] + rho_c(rho_in, i, jpp, k, p) * K_q[at(i, jpp, k, p)]) * (qv_in[at(i, jpp, k, p)] - qv_in[c]) / p.dy;
                float Fy_s = 0.5f * (rho_c(rho_in, i, jmm, k, p) * K_q[at(i, jmm, k, p)] + rho_c(rho_in, i, j, k, p) * K_q[c]) * (qv_in[c] - qv_in[at(i, jmm, k, p)]) / p.dy;
                div_qv += (Fx_e - Fx_w) / p.dx + (Fy_n - Fy_s) / p.dy;
            }
            if (vert)
            {
                float Fz_t = k < p.nz - 1 ? 0.5f * (rho_c(rho_in, i, j, k, p) * K_q[c] + rho_c(rho_in, i, j, k + 1, p) * K_q[at(i, j, k + 1, p)]) * (qv_in[at(i, j, k + 1, p)] - qv_in[c]) / center_distance(dz_prof, k) : 0.0f;
                float Fz_b = k > 0 ? 0.5f * (rho_c(rho_in, i, j, k - 1, p) * K_q[at(i, j, k - 1, p)] + rho_c(rho_in, i, j, k, p) * K_q[c]) * (qv_in[c] - qv_in[at(i, j, k - 1, p)]) / center_distance(dz_prof, k - 1) : 0.0f;
                div_qv += (Fz_t - Fz_b) / dz_k;
            }
            dqv_out = sanitize(div_qv / rho_c(rho_in, i, j, k, p));
        }
    }

    dudt_sgs[c] = du_out;
    dvdt_sgs[c] = dv_out;
    dwdt_sgs[c] = dw_out;
    dthetadt_sgs[c] = dth_out;
    dqvdt_sgs[c] = dqv_out;
}
)msl";

} // namespace nimbus::metal::shaders
