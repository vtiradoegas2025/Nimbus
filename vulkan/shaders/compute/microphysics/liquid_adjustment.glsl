#include "milbrandt_thermodynamics.glsl"
float value(vec2 x)
{
    return x.x + x.y;
}
vec2 bounded(vec2 x, vec2 lo, vec2 hi)
{
    if (ml_greater(lo, x))
        return lo;
    if (ml_greater(x, hi))
        return hi;
    return x;
}
vec2 slope(vec2 T, vec2 q, float pressure)
{
    if (!ml_greater(T, vec2(150, 0)) || !ml_greater(vec2(400, 0), T) || !ml_greater(q, vec2(0)) ||
        !ml_greater(vec2(0.20000000298023224, -2.980232227667301e-09), q))
        return vec2(0);
    vec2 p           = vec2(clamp(pressure, 100.0, 200000.0), 0);
    vec2 es          = df_div(df_mul(q, p), df_add(vec2(0.62198001146316528, -1.1463165305869438e-08), q));
    vec2 tc          = df_sub(T, vec2(273.14999389648438, 6.1035156022626325e-06));
    vec2 denominator = df_add(vec2(257.1400146484375, -1.4648437513642421e-05), tc);
    vec2 numerator   = df_mul(df_sub(vec2(18.677999496459961, 5.0354003988672957e-07), df_div(tc, vec2(234.5, 0))), tc);
    vec2 derivative =
        df_sub(vec2(18.677999496459961, 5.0354003988672957e-07), df_div(df_mul(vec2(2, 0), tc), vec2(234.5, 0)));
    vec2 log_slope = df_div(df_sub(df_mul(derivative, denominator), numerator), df_mul(denominator, denominator));
    vec2 pressure_difference = df_sub(p, es);
    if (pressure_difference.x < 1.0)
        pressure_difference = vec2(1, 0);
    return df_mul(df_div(df_mul(q, p), pressure_difference), log_slope);
}
vec2 liquid_adjustment(float temperature, float pressure, float vapor, float cloud)
{
    vec2 T = vec2(temperature, 0), qv = vec2(vapor, 0), qc = vec2(cloud, 0);
    vec2 latent  = vec2(2490.039794921875, 4.5715575197391445e-05);
    vec2 qs      = milbrandt_qsat(T, pressure);
    vec2 initial = df_sub(qv, qs);
    if (initial.x == 0.0 && initial.y == 0.0)
        return vec2(0);
    bool condensing = ml_greater(qv, qs);
    if (!condensing && cloud == 0.0)
        return vec2(0);
    vec2 sign       = vec2(condensing ? 1.0 : -1.0, 0);
    vec2 available  = condensing ? qv : qc;
    vec2 headroom   = df_div(condensing ? df_sub(vec2(400, 0), T) : df_sub(T, vec2(150, 0)), latent);
    vec2 maximum    = bounded(headroom, vec2(0), available);
    vec2 endpoint_T = df_add(T, df_mul(sign, df_mul(latent, maximum)));
    vec2 endpoint   = df_sub(df_sub(qv, df_mul(sign, maximum)), milbrandt_qsat(endpoint_T, pressure));
    if (ml_greater(df_mul(sign, endpoint), vec2(0)))
        return df_mul(sign, maximum);
    vec2 lower = vec2(0), upper = maximum;
    vec2 amount = bounded(
        df_div(df_mul(sign, initial), df_add(vec2(1, 0), df_mul(latent, slope(T, qs, pressure)))), lower, upper);
    for (int iteration = 0; iteration < 10; ++iteration)
    {
        vec2 adjusted_T  = df_add(T, df_mul(sign, df_mul(latent, amount)));
        vec2 adjusted_qs = milbrandt_qsat(adjusted_T, pressure);
        vec2 residual    = df_sub(df_sub(qv, df_mul(sign, amount)), adjusted_qs);
        if (abs(value(residual)) <= 1.7763568394002505e-15)
            break;
        if (ml_greater(df_mul(sign, residual), vec2(0)))
            lower = amount;
        else
            upper = amount;
        vec2 derivative = df_mul(-sign, df_add(vec2(1, 0), df_mul(latent, slope(adjusted_T, adjusted_qs, pressure))));
        vec2 candidate  = df_sub(amount, df_div(residual, derivative));
        amount          = ml_greater(candidate, lower) && ml_greater(upper, candidate)
                              ? candidate
                              : df_mul(vec2(.5, 0), df_add(lower, upper));
    }
    return df_mul(sign, amount);
}
