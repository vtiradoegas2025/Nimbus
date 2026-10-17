#pragma once

namespace nimbus::metal::shaders
{

// Runtime-compiled MSL translation of the canonical Vulkan MYNN kernel.
inline constexpr char kMynnLevel25StabilityMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;

struct MynnLevel25StabilityPushConstants
{
    int count;
    int padding0;
    int padding1;
    int padding2;
};

inline bool finite_f(float value) { return !isnan(value) && !isinf(value); }

kernel void mynn_level25_stability(
    device const float* gm_s2 [[buffer(0)]],
    device const float* gh_s2 [[buffer(1)]],
    device const float* mixing_length_m [[buffer(2)]],
    device const float* q_squared_m2s2 [[buffer(3)]],
    device float* sm [[buffer(4)]],
    device float* sh [[buffer(5)]],
    device float* gradient_richardson [[buffer(6)]],
    device float* flux_richardson [[buffer(7)]],
    device float* equilibrium_limiter [[buffer(8)]],
    constant MynnLevel25StabilityPushConstants& pc [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(pc.count)) return;
    const float invalid = as_type<float>(uint(0x7fc00000u));
    const float gm = gm_s2[gid];
    const float gh = gh_s2[gid];
    const float length = mixing_length_m[gid];
    const float q2 = q_squared_m2s2[gid];
    if (!finite_f(gm) || gm < 0.0f || !finite_f(gh) || !finite_f(length) ||
        length <= 0.0f || !finite_f(q2) || q2 <= 0.0f)
    {
        sm[gid] = invalid; sh[gid] = invalid;
        gradient_richardson[gid] = invalid; flux_richardson[gid] = invalid;
        equilibrium_limiter[gid] = invalid;
        return;
    }
    const float g1 = 0.235f, b1 = 24.0f, b2 = 15.0f, c2 = 0.75f;
    const float c3 = 0.352f, c5 = 0.2f, pr = 0.74f;
    const float a1 = b1 * (1.0f - 3.0f * g1) / 6.0f;
    const float c1 = g1 - 1.0f / (3.0f * a1 * pow(b1, 1.0f / 3.0f));
    const float a2 = a1 * (g1 - c1) / (g1 * pr);
    const float g2 = b2 / b1 * (1.0f - c3) + 2.0f * a1 / b1 * (3.0f - 2.0f * c2);
    const float rfc = g1 / (g1 + g2);
    const float f1 = b1 * (g1 - c1) + 3.0f * a2 * (1.0f - c2) * (1.0f - c5)
                   + 2.0f * a1 * (3.0f - 2.0f * c2);
    const float f2 = b1 * (g1 + g2) - 3.0f * a1 * (1.0f - c2);
    const float rf1 = b1 * (g1 - c1) / f1;
    const float rf2 = b1 * g1 / f2;
    const float smc = a1 / a2 * f1 / f2;
    const float shc = 3.0f * a2 * (g1 + g2);
    const float ri1 = 0.5f / smc;
    const float ri2 = rf1 * smc;
    const float ri3 = 4.0f * rf2 * smc - 2.0f * ri2;
    const float ri4 = ri2 * ri2;
    const float ri = -gh / max(gm, 1.0e-10f);
    const float discriminant = max(0.0f, ri * ri - ri3 * ri + ri4);
    const float rf = min(ri1 * (ri + ri2 - sqrt(discriminant)), rfc);
    float sh_value = shc * (rfc - rf) / (1.0f - rf);
    float sm_value = smc * (rf1 - rf) / (rf2 - rf) * sh_value;
    const float length2 = length * length;
    const float equilibrium_q2 = b1 * length2 * (sm_value * gm + sh_value * gh);
    float qdiv = 1.0f;
    if (equilibrium_q2 > 0.0f && q2 < equilibrium_q2)
    {
        qdiv = sqrt(q2 / equilibrium_q2); sm_value *= qdiv; sh_value *= qdiv;
    }
    else
    {
        const float e5c = 6.0f * a1 * a1;
        const float e1c = 3.0f * a2 * b2 * (1.0f - c3);
        const float e2c = 9.0f * a1 * a2 * (1.0f - c2);
        const float e3c = 9.0f * a2 * a2 * (1.0f - c2) * (1.0f - c5);
        const float e4c = 12.0f * a1 * a2 * (1.0f - c2);
        const float gmel = gm * length2, ghel = gh * length2;
        const float e1 = q2 - e1c * ghel, e2 = q2 - e2c * ghel;
        const float e3 = e1 + e3c * ghel, e4 = e1 - e4c * ghel;
        const float denominator = e2 * e4 + e3 * e5c * gmel;
        if (!finite_f(denominator) || denominator <= 1.0e-20f)
        {
            sm[gid] = invalid; sh[gid] = invalid;
            gradient_richardson[gid] = invalid; flux_richardson[gid] = invalid;
            equilibrium_limiter[gid] = invalid;
            return;
        }
        sm_value = q2 * a1 * (e3 - 3.0f * c1 * e4) / denominator;
        sh_value = q2 * a2 * (e2 + 3.0f * c1 * e5c * gmel) / denominator;
    }
    if (!finite_f(sm_value) || !finite_f(sh_value) || sm_value < 0.0f || sh_value < 0.0f)
    {
        sm[gid] = invalid; sh[gid] = invalid;
        gradient_richardson[gid] = invalid; flux_richardson[gid] = invalid;
        equilibrium_limiter[gid] = invalid;
        return;
    }
    sm[gid] = sm_value; sh[gid] = sh_value;
    gradient_richardson[gid] = ri; flux_richardson[gid] = rf;
    equilibrium_limiter[gid] = qdiv;
}
)msl";

} // namespace nimbus::metal::shaders
