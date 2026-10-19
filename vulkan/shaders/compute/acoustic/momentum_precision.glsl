float acoustic_precise_momentum_rate(float pressure_low, float pressure_high,
                                    float density_low, float density_high,
                                    float spacing, vec2 base_gradient)
{
    vec2 density = df_mul(df_add(vec2(density_low, 0), vec2(density_high, 0)), vec2(0.5, 0));
    if (isnan(density.x) || isinf(density.x) || density.x <= 1e-6)
        density = vec2(1, 0);
    vec2 gradient = df_div(df_sub(vec2(pressure_high, 0), vec2(pressure_low, 0)), vec2(spacing, 0));
    vec2 rate = -df_div(df_sub(gradient, base_gradient), density);
    return rate.x + rate.y;
}

float acoustic_precise_update(float value, float rate, float dt_small)
{
    vec2 candidate = df_add(vec2(value, 0), df_mul(vec2(rate, 0), vec2(dt_small, 0)));
    return candidate.x + candidate.y;
}

vec2 acoustic_precise_mass_flux(float density_a, float density_b, float velocity, float frame)
{
    return df_mul(df_mul(df_add(vec2(density_a, 0), vec2(density_b, 0)), vec2(0.5, 0)),
                  df_sub(vec2(velocity, 0), vec2(frame, 0)));
}

float acoustic_precise_density_rate(vec2 xp, vec2 xm, vec2 yp, vec2 ym, vec2 zp, vec2 zm,
                                   float spacing_x, float spacing_y, float spacing_z)
{
    vec2 x = df_div(df_sub(xp, xm), vec2(spacing_x, 0));
    vec2 y = df_div(df_sub(yp, ym), vec2(spacing_y, 0));
    vec2 z = df_div(df_sub(zp, zm), vec2(spacing_z, 0));
    vec2 rate = -df_add(df_add(x, y), z);
    return rate.x + rate.y;
}
