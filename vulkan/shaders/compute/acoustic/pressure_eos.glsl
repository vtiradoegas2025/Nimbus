float acoustic_pressure_eos(float density, float theta, vec2 gamma_pair)
{
    vec2 ratio = df_div(df_mul(df_mul(vec2(density, 0), vec2(287, 0)), vec2(theta, 0)), vec2(100000, 0));
    vec2 power = df_exp(df_mul(gamma_pair, df_log(ratio)));
    vec2 pressure = df_mul(vec2(100000, 0), power);
    return pressure.x + pressure.y;
}
