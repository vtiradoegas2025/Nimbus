#pragma once

namespace nimbus::metal::shaders
{

inline constexpr char kAcousticContinuityMassFluxCartesianAgridMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;

struct Constants {
    int nx; int ny; int nz;
    float dx; float dy; float dz; float dt_small;
    int west; int east; int south; int north; int top; int accumulate_mass_flux;
    float mass_flux_weight;
    float frame_u; float frame_v;
};

inline uint idx(constant Constants& c, int i, int j, int k) {
    return uint((i * c.ny + j) * c.nz + k);
}
inline bool closed_lateral(int bc) { return bc == 2 || bc == 3; }
inline int periodic_index(int value, int extent) {
    if (value < 1) return extent - 2;
    if (value > extent - 2) return 1;
    return value;
}

inline float corrected_x(device const float* u, device const float* rho,
                         device const float* p, constant Constants& c,
                         int face, int j, int k) {
    int lo = face, hi = face + 1;
    bool periodic = (face == 0 && c.west == 0) ||
                    (face == c.nx - 2 && c.east == 0);
    if (periodic) { lo = c.nx - 2; hi = 1; }
    float rlo = rho[idx(c, lo, j, k)], rhi = rho[idx(c, hi, j, k)];
    float rf = 0.5f * (rlo + rhi);
    float vf = 0.5f * (u[idx(c, lo, j, k)] + u[idx(c, hi, j, k)]);
    int lom = periodic ? periodic_index(lo - 1, c.nx) : lo - 1;
    int lop = periodic ? periodic_index(lo + 1, c.nx) : lo + 1;
    int him = periodic ? periodic_index(hi - 1, c.nx) : hi - 1;
    int hip = periodic ? periodic_index(hi + 1, c.nx) : hi + 1;
    float glo = (p[idx(c, lop, j, k)] - p[idx(c, lom, j, k)]) / (2.0f * c.dx);
    float ghi = (p[idx(c, hip, j, k)] - p[idx(c, him, j, k)]) / (2.0f * c.dx);
    float ac = -0.5f * (glo / rlo + ghi / rhi);
    float af = -(p[idx(c, hi, j, k)] - p[idx(c, lo, j, k)]) / (c.dx * rf);
    return vf + c.dt_small * (af - ac);
}

inline float corrected_y(device const float* v, device const float* rho,
                         device const float* p, constant Constants& c,
                         int i, int face, int k) {
    int lo = face, hi = face + 1;
    bool periodic = (face == 0 && c.south == 0) ||
                    (face == c.ny - 2 && c.north == 0);
    if (periodic) { lo = c.ny - 2; hi = 1; }
    float rlo = rho[idx(c, i, lo, k)], rhi = rho[idx(c, i, hi, k)];
    float rf = 0.5f * (rlo + rhi);
    float vf = 0.5f * (v[idx(c, i, lo, k)] + v[idx(c, i, hi, k)]);
    int lom = periodic ? periodic_index(lo - 1, c.ny) : lo - 1;
    int lop = periodic ? periodic_index(lo + 1, c.ny) : lo + 1;
    int him = periodic ? periodic_index(hi - 1, c.ny) : hi - 1;
    int hip = periodic ? periodic_index(hi + 1, c.ny) : hi + 1;
    float glo = (p[idx(c, i, lop, k)] - p[idx(c, i, lom, k)]) / (2.0f * c.dy);
    float ghi = (p[idx(c, i, hip, k)] - p[idx(c, i, him, k)]) / (2.0f * c.dy);
    float ac = -0.5f * (glo / rlo + ghi / rhi);
    float af = -(p[idx(c, i, hi, k)] - p[idx(c, i, lo, k)]) / (c.dy * rf);
    return vf + c.dt_small * (af - ac);
}

inline float pprime(device const float* p, device const float* p0,
                    constant Constants& c, int i, int j, int k) {
    return p[idx(c, i, j, k)] - p0[k];
}
inline float corrected_z(device const float* w, device const float* rho,
                         device const float* p, device const float* p0,
                         constant Constants& c, int i, int j, int face) {
    int lo = face, hi = face + 1;
    float rlo = rho[idx(c, i, j, lo)], rhi = rho[idx(c, i, j, hi)];
    float rf = 0.5f * (rlo + rhi);
    float vf = 0.5f * (w[idx(c, i, j, lo)] + w[idx(c, i, j, hi)]);
    float glo = (pprime(p, p0, c, i, j, lo + 1) -
                 pprime(p, p0, c, i, j, lo - 1)) / (2.0f * c.dz);
    float ghi = (pprime(p, p0, c, i, j, hi + 1) -
                 pprime(p, p0, c, i, j, hi - 1)) / (2.0f * c.dz);
    float ac = -0.5f * (glo / rlo + ghi / rhi);
    float af = -(pprime(p, p0, c, i, j, hi) -
                 pprime(p, p0, c, i, j, lo)) / (c.dz * rf);
    return vf + c.dt_small * (af - ac);
}

inline float xface(device const float* u, device const float* rho,
                   device const float* p, constant Constants& c,
                   int face, int j, int k) {
    if (face == 0) {
        if (closed_lateral(c.west)) return 0.0f;
        if (c.west == 0)
            return 0.5f * (rho[idx(c, c.nx - 2, j, k)] + rho[idx(c, 1, j, k)]) *
                   (corrected_x(u, rho, p, c, face, j, k) - c.frame_u);
        return rho[idx(c, 0, j, k)] * (u[idx(c, 0, j, k)] - c.frame_u);
    }
    if (face == c.nx - 2) {
        if (closed_lateral(c.east)) return 0.0f;
        if (c.east == 0)
            return 0.5f * (rho[idx(c, c.nx - 2, j, k)] + rho[idx(c, 1, j, k)]) *
                   (corrected_x(u, rho, p, c, face, j, k) - c.frame_u);
        return rho[idx(c, c.nx - 1, j, k)] *
               (u[idx(c, c.nx - 1, j, k)] - c.frame_u);
    }
    return 0.5f * (rho[idx(c, face, j, k)] + rho[idx(c, face + 1, j, k)]) *
           (corrected_x(u, rho, p, c, face, j, k) - c.frame_u);
}

inline float yface(device const float* v, device const float* rho,
                   device const float* p, constant Constants& c,
                   int i, int face, int k) {
    if (face == 0) {
        if (closed_lateral(c.south)) return 0.0f;
        if (c.south == 0)
            return 0.5f * (rho[idx(c, i, c.ny - 2, k)] + rho[idx(c, i, 1, k)]) *
                   (corrected_y(v, rho, p, c, i, face, k) - c.frame_v);
        return rho[idx(c, i, 0, k)] * (v[idx(c, i, 0, k)] - c.frame_v);
    }
    if (face == c.ny - 2) {
        if (closed_lateral(c.north)) return 0.0f;
        if (c.north == 0)
            return 0.5f * (rho[idx(c, i, c.ny - 2, k)] + rho[idx(c, i, 1, k)]) *
                   (corrected_y(v, rho, p, c, i, face, k) - c.frame_v);
        return rho[idx(c, i, c.ny - 1, k)] *
               (v[idx(c, i, c.ny - 1, k)] - c.frame_v);
    }
    return 0.5f * (rho[idx(c, i, face, k)] + rho[idx(c, i, face + 1, k)]) *
           (corrected_y(v, rho, p, c, i, face, k) - c.frame_v);
}

inline float zface(device const float* w, device const float* rho,
                   device const float* p, device const float* p0,
                   constant Constants& c, int i, int j, int face) {
    if (face == 0) return 0.0f;
    if (face == c.nz - 2) {
        if (c.top != 2) return 0.0f;
        return rho[idx(c, i, j, c.nz - 1)] * w[idx(c, i, j, c.nz - 1)];
    }
    return 0.5f * (rho[idx(c, i, j, face)] + rho[idx(c, i, j, face + 1)]) *
           corrected_z(w, rho, p, p0, c, i, j, face);
}

kernel void acoustic_continuity_massflux_cartesian_agrid(
    device const float* u [[buffer(0)]], device const float* v [[buffer(1)]],
    device const float* w [[buffer(2)]], device const float* rho [[buffer(3)]],
    device const float* p [[buffer(4)]], device const float* p0 [[buffer(5)]],
    device float* drho [[buffer(6)]], device float* mfx [[buffer(7)]],
    device float* mfy [[buffer(8)]], device float* mfz [[buffer(9)]],
    constant Constants& c [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = uint(c.nx) * uint(c.ny) * uint(c.nz);
    if (gid >= total) return;
    int k = int(gid % uint(c.nz));
    uint ij = gid / uint(c.nz);
    int j = int(ij % uint(c.ny));
    int i = int(ij / uint(c.ny));
    float fx = i < c.nx - 1 ? xface(u, rho, p, c, i, j, k) : 0.0f;
    float fy = j < c.ny - 1 ? yface(v, rho, p, c, i, j, k) : 0.0f;
    float fz = k < c.nz - 1 ? zface(w, rho, p, p0, c, i, j, k) : 0.0f;
    if (c.accumulate_mass_flux != 0) {
        mfx[gid] += c.mass_flux_weight * fx;
        mfy[gid] += c.mass_flux_weight * fy;
        mfz[gid] += c.mass_flux_weight * fz;
    } else { mfx[gid] = fx; mfy[gid] = fy; mfz[gid] = fz; }
    if (i == 0 || i == c.nx - 1 || j == 0 || j == c.ny - 1 ||
        k == 0 || k == c.nz - 1) { drho[gid] = 0.0f; return; }
    float divm = (xface(u, rho, p, c, i, j, k) - xface(u, rho, p, c, i - 1, j, k)) / c.dx +
                 (yface(v, rho, p, c, i, j, k) - yface(v, rho, p, c, i, j - 1, k)) / c.dy +
                 (zface(w, rho, p, p0, c, i, j, k) - zface(w, rho, p, p0, c, i, j, k - 1)) / c.dz;
    drho[gid] = -divm;
}
)msl";

} // namespace nimbus::metal::shaders
