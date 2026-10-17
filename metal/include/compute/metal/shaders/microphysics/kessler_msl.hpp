#pragma once

namespace nimbus::metal::shaders
{

inline constexpr const char* kKesslerPointwiseMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct PushConstants
{
    int nr; int nth; int nz;
    int i_begin; int i_end; int j_begin; int j_end; int k_begin; int k_end;
    float qc0; float c_auto; float c_accr; float evaporation_efficiency;
    float Lv_cp; float T0; float dt_step;
};

inline float saturation_mixing_ratio_water(float T, float pressure,
                                            constant PushConstants& p)
{
    float T_c = T - p.T0;
    float e_sat = 611.21f * exp((18.678f - T_c / 234.5f) * T_c / (257.14f + T_c));
    return clamp(0.62198f * e_sat / max(pressure - e_sat, 1.0f), 0.0f, 0.2f);
}

kernel void kessler_pointwise(
    device const float* temperature [[buffer(0)]],
    device const float* pressure    [[buffer(1)]],
    device const float* qv          [[buffer(2)]],
    device const float* qc          [[buffer(3)]],
    device const float* qr          [[buffer(4)]],
    device const float* rho_d       [[buffer(5)]],
    device const float* qh          [[buffer(6)]],
    device float* dtheta_dt         [[buffer(7)]],
    device float* dqv_dt            [[buffer(8)]],
    device float* dqc_dt            [[buffer(9)]],
    device float* dqr_dt            [[buffer(10)]],
    device float* dqg_dt            [[buffer(11)]],
    device float* dqh_dt            [[buffer(12)]],
    constant PushConstants& p       [[buffer(13)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint total = uint(p.nr) * uint(p.nth) * uint(p.nz);
    if (gid >= total) return;
    int k = int(gid % uint(p.nz));
    uint column = gid / uint(p.nz);
    int j = int(column % uint(p.nth));
    int i = int(column / uint(p.nth));
    if (i < p.i_begin || i >= p.i_end || j < p.j_begin || j >= p.j_end ||
        k < p.k_begin || k >= p.k_end)
    {
        dtheta_dt[gid] = 0.0f; dqv_dt[gid] = 0.0f; dqc_dt[gid] = 0.0f;
        dqr_dt[gid] = 0.0f; dqg_dt[gid] = 0.0f; dqh_dt[gid] = 0.0f;
        return;
    }

    float T = temperature[gid];
    float pressure_value = pressure[gid];
    float qv_value = qv[gid];
    float qc_value = qc[gid];
    float qr_value = qr[gid];
    float rho_value = rho_d[gid];
    float d_theta = 0.0f;
    float d_qv = 0.0f;
    float d_qc = 0.0f;
    float d_qr = 0.0f;
    float production_rate = 0.0f;
    if (p.dt_step > 0.0f && qc_value > 0.0f)
    {
        float factor = 1.0f / (1.0f + p.c_accr * p.dt_step * pow(max(qr_value, 0.0f), 0.875f));
        float amount = qc_value * (1.0f - factor) +
            factor * p.c_auto * p.dt_step * max(qc_value - p.qc0, 0.0f);
        amount = clamp(amount, 0.0f, qc_value);
        production_rate = amount / p.dt_step;
        d_qc -= production_rate;
        d_qr += production_rate;
    }
    if (qr_value > 0.0f || production_rate > 0.0f)
    {
        float qvsat = saturation_mixing_ratio_water(T, pressure_value, p);
        if (qvsat > 0.0f && qv_value < qvsat && rho_value > 0.0f && pressure_value > 0.0f)
        {
            float qr_available = qr_value + production_rate * p.dt_step;
            float rcgs = 1.0e-3f * rho_value;
            float rain_content = rcgs * max(qr_available, 0.0f);
            float ventilation = (1.6f + 124.9f * pow(rain_content, 0.2046f)) *
                                pow(rain_content, 0.525f);
            float resistance = 2.55e8f / (pressure_value * qvsat) + 5.4e5f;
            float raw_rate = p.evaporation_efficiency * (ventilation / resistance) *
                             ((qvsat - qv_value) / (rcgs * qvsat));
            float gamma_sat = p.Lv_cp * (2.5e6f / 461.5f) * qvsat / max(T * T, 1.0f);
            float saturation_bound = (qvsat - qv_value) / (1.0f + gamma_sat);
            float evap_rate = clamp(raw_rate, 0.0f,
                min(saturation_bound / p.dt_step, qr_available / p.dt_step));
            d_qv += evap_rate;
            d_qr -= evap_rate;
            d_theta -= p.Lv_cp * evap_rate;
        }
    }
    dtheta_dt[gid] = d_theta;
    dqv_dt[gid] = d_qv;
    dqc_dt[gid] = d_qc;
    dqr_dt[gid] = d_qr;
    dqg_dt[gid] = 0.0f;
    dqh_dt[gid] = 0.0f;
    (void)qh;
}
)msl";

} // namespace nimbus::metal::shaders
