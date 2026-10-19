#pragma once

namespace nimbus::metal::shaders
{

inline constexpr const char* kAcousticCompleteAgridMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct PressureEosConstants {
    int nx; int ny; int nz; float dt_small;
    float reference_pressure; float gas_constant_dry_air; float gamma_value;
    float rho_min; float rho_max; float p_min; float p_max; float padding;
};

kernel void acoustic_pressure_eos_cartesian_agrid(
    device const float* rho_in [[buffer(0)]],
    device const float* theta [[buffer(1)]],
    device const float* drho_dt [[buffer(2)]],
    device float* rho_out [[buffer(3)]],
    device float* p_out [[buffer(4)]],
    constant PressureEosConstants& c [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = uint(c.nx) * uint(c.ny) * uint(c.nz);
    if (gid >= count) return;
    float next_rho = clamp(rho_in[gid] + c.dt_small * drho_dt[gid], c.rho_min, c.rho_max);
    float ratio = next_rho * c.gas_constant_dry_air * theta[gid] / c.reference_pressure;
    rho_out[gid] = next_rho;
    p_out[gid] = clamp(c.reference_pressure * pow(ratio, c.gamma_value), c.p_min, c.p_max);
}

struct MomentumConstants {
    int nx; int ny; int nz; float dx; float dy; float dz; float dt_small;
    float wind_clamp_h; float wind_clamp_v; float padding;
};

inline uint agrid_at(int i, int j, int k, constant MomentumConstants& c) {
    return uint((i * c.ny + j) * c.nz + k);
}

kernel void acoustic_momentum_cartesian_agrid(
    device const float* rho [[buffer(0)]],
    device const float* pressure [[buffer(1)]],
    device const float* p0 [[buffer(2)]],
    device const float* u_in [[buffer(3)]],
    device const float* v_in [[buffer(4)]],
    device const float* w_in [[buffer(5)]],
    device float* u_out [[buffer(6)]],
    device float* v_out [[buffer(7)]],
    device float* w_out [[buffer(8)]],
    constant MomentumConstants& c [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = uint(c.nx) * uint(c.ny) * uint(c.nz);
    if (gid >= count) return;
    int k = int(gid % uint(c.nz)); uint ij = gid / uint(c.nz);
    int j = int(ij % uint(c.ny)); int i = int(ij / uint(c.ny));
    if (i == 0 || i == c.nx - 1 || j == 0 || j == c.ny - 1 || k == 0 || k == c.nz - 1) {
        u_out[gid] = u_in[gid]; v_out[gid] = v_in[gid]; w_out[gid] = w_in[gid]; return;
    }
    float px = (pressure[agrid_at(i + 1, j, k, c)] - pressure[agrid_at(i - 1, j, k, c)]) / (2.0f * c.dx);
    float py = (pressure[agrid_at(i, j + 1, k, c)] - pressure[agrid_at(i, j - 1, k, c)]) / (2.0f * c.dy);
    float pz = (pressure[agrid_at(i, j, k + 1, c)] - pressure[agrid_at(i, j, k - 1, c)]) / (2.0f * c.dz);
    float p0z = (p0[k + 1] - p0[k - 1]) / (2.0f * c.dz);
    u_out[gid] = clamp(u_in[gid] - c.dt_small * px / rho[gid], -c.wind_clamp_h, c.wind_clamp_h);
    v_out[gid] = clamp(v_in[gid] - c.dt_small * py / rho[gid], -c.wind_clamp_h, c.wind_clamp_h);
    w_out[gid] = clamp(w_in[gid] - c.dt_small * (pz - p0z) / rho[gid], -c.wind_clamp_v, c.wind_clamp_v);
}
)msl";

inline constexpr const char* kAcousticAgridClosureMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct LinePassConstants { int nx; int ny; int nz; int pass; };
inline uint line_at(int i, int j, int k, constant LinePassConstants& c) {
    return uint((i * c.ny + j) * c.nz + k);
}

inline float mixture_mass(device const float* rho,
                          device const float* qv, device const float* qc,
                          device const float* qr, device const float* qi,
                          device const float* qs, device const float* qg,
                          device const float* qh,
                          int i, int j, int k, constant LinePassConstants& c) {
    uint n = line_at(i, j, k, c);
    return rho[n] * (1.0f + qv[n] + qc[n] + qr[n] + qi[n] + qs[n] + qg[n] + qh[n]);
}

kernel void acoustic_null_mode_constraint_cartesian_agrid(
    device const float* rho [[buffer(0)]],
    device const float* qv [[buffer(1)]], device const float* qc [[buffer(2)]],
    device const float* qr [[buffer(3)]], device const float* qi [[buffer(4)]],
    device const float* qs [[buffer(5)]], device const float* qg [[buffer(6)]],
    device const float* qh [[buffer(7)]],
    device float* u [[buffer(8)]], device float* v [[buffer(9)]], device float* w [[buffer(10)]],
    constant LinePassConstants& c [[buffer(11)]], uint gid [[thread_position_in_grid]])
{
    uint line_count = c.pass == 0 ? uint(c.ny - 2) * uint(c.nz - 2)
                    : c.pass == 1 ? uint(c.nx - 2) * uint(c.nz - 2)
                                  : uint(c.nx - 2) * uint(c.ny - 2);
    if (gid >= line_count) return;
    int fixed_a = c.pass == 0 ? int(gid / uint(c.nz - 2)) + 1
                : c.pass == 1 ? int(gid / uint(c.nz - 2)) + 1
                              : int(gid / uint(c.ny - 2)) + 1;
    int fixed_b = c.pass == 0 ? int(gid % uint(c.nz - 2)) + 1
                : c.pass == 1 ? int(gid % uint(c.nz - 2)) + 1
                              : int(gid % uint(c.ny - 2)) + 1;
    int count = c.pass == 0 ? c.nx - 2 : (c.pass == 1 ? c.ny - 2 : c.nz - 2);
    float midpoint = 0.5f * float(count - 1);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, t0 = 0.0f, t1 = 0.0f;
    for (int n = 0; n < count; ++n) {
        int i = c.pass == 0 ? n + 1 : fixed_a;
        int j = c.pass == 1 ? n + 1 : (c.pass == 0 ? fixed_a : fixed_b);
        int k = c.pass == 2 ? n + 1 : fixed_b;
        float m = mixture_mass(rho, qv, qc, qr, qi, qs, qg, qh, i, j, k, c);
        float x = float(n) - midpoint; float parity = (n & 1) == 0 ? 1.0f : -1.0f;
        s0 += m; s1 += m * x; s2 += m * x * x; t0 += m * parity; t1 += m * x * parity;
    }
    float determinant = s0 * s2 - s1 * s1;
    if (!(determinant > 0.0f) || !isfinite(determinant)) return;
    float intercept = (t0 * s2 - t1 * s1) / determinant;
    float slope = (t1 * s0 - t0 * s1) / determinant;
    float numerator = 0.0f, denominator = 0.0f;
    for (int n = 0; n < count; ++n) {
        int i = c.pass == 0 ? n + 1 : fixed_a;
        int j = c.pass == 1 ? n + 1 : (c.pass == 0 ? fixed_a : fixed_b);
        int k = c.pass == 2 ? n + 1 : fixed_b; uint index = line_at(i, j, k, c);
        float m = mixture_mass(rho, qv, qc, qr, qi, qs, qg, qh, i, j, k, c);
        float x = float(n) - midpoint;
        float residual = ((n & 1) == 0 ? 1.0f : -1.0f) - intercept - slope * x;
        float velocity = c.pass == 0 ? u[index] : (c.pass == 1 ? v[index] : w[index]);
        numerator += m * residual * velocity; denominator += m * residual * residual;
    }
    if (!(denominator > 0.0f) || !isfinite(denominator) || !isfinite(numerator)) return;
    float amplitude = numerator / denominator;
    for (int n = 0; n < count; ++n) {
        int i = c.pass == 0 ? n + 1 : fixed_a;
        int j = c.pass == 1 ? n + 1 : (c.pass == 0 ? fixed_a : fixed_b);
        int k = c.pass == 2 ? n + 1 : fixed_b; uint index = line_at(i, j, k, c);
        float x = float(n) - midpoint;
        float residual = ((n & 1) == 0 ? 1.0f : -1.0f) - intercept - slope * x;
        if (c.pass == 0) u[index] -= amplitude * residual;
        else if (c.pass == 1) v[index] -= amplitude * residual;
        else w[index] -= amplitude * residual;
    }
}

kernel void acoustic_boundary_cartesian_agrid(
    device const float* p0 [[buffer(0)]], device const float* rho0 [[buffer(1)]],
    device const float* u0 [[buffer(2)]], device const float* v0 [[buffer(3)]],
    device float* u [[buffer(4)]], device float* v [[buffer(5)]], device float* w [[buffer(6)]],
    device float* rho [[buffer(7)]], device float* pressure [[buffer(8)]],
    constant LinePassConstants& c [[buffer(9)]], uint gid [[thread_position_in_grid]])
{
    if (c.pass == 0) {
        uint count = uint(c.ny) * uint(c.nz); if (gid >= count) return;
        int k = int(gid % uint(c.nz)); int j = int(gid / uint(c.nz));
        for (int face = 0; face < 2; ++face) {
            int i = face == 0 ? 0 : c.nx - 1; int source = face == 0 ? 1 : c.nx - 2;
            uint n = line_at(i,j,k,c), s = line_at(source,j,k,c);
            u[n]=0.0f;v[n]=0.0f;w[n]=0.0f;rho[n]=rho[s];pressure[n]=pressure[s];
        }
    } else if (c.pass == 1) {
        uint count = uint(c.nx) * uint(c.nz); if (gid >= count) return;
        int k = int(gid % uint(c.nz)); int i = int(gid / uint(c.nz));
        for (int face = 0; face < 2; ++face) {
            int j = face == 0 ? 0 : c.ny - 1; int source = face == 0 ? 1 : c.ny - 2;
            uint n = line_at(i,j,k,c), s = line_at(i,source,k,c);
            u[n]=0.0f;v[n]=0.0f;w[n]=0.0f;rho[n]=rho[s];pressure[n]=pressure[s];
        }
    } else {
        uint count = uint(c.nx) * uint(c.ny); if (gid >= count) return;
        int j = int(gid % uint(c.ny)); int i = int(gid / uint(c.ny));
        uint b=line_at(i,j,0,c), bi=line_at(i,j,1,c), t=line_at(i,j,c.nz-1,c), ti=line_at(i,j,c.nz-2,c);
        w[b]=0.0f;w[t]=0.0f;u[b]=u0[0]+(u[bi]-u0[1]);v[b]=v0[0]+(v[bi]-v0[1]);
        u[t]=u0[c.nz-1]+(u[ti]-u0[c.nz-2]);v[t]=v0[c.nz-1]+(v[ti]-v0[c.nz-2]);
        rho[b]=rho0[0]+(rho[bi]-rho0[1]);rho[t]=rho0[c.nz-1]+(rho[ti]-rho0[c.nz-2]);
        pressure[b]=p0[0]+(pressure[bi]-p0[1]);pressure[t]=p0[c.nz-1]+(pressure[ti]-p0[c.nz-2]);
    }
}
)msl";

} // namespace nimbus::metal::shaders
