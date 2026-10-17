bool ml_greater(vec2 a, vec2 b)
{
    vec2 difference = df_sub(a, b);
    return difference.x > 0.0 || (difference.x == 0.0 && difference.y > 0.0);
}
vec2 milbrandt_qsat(vec2 T, float pressure)
{
    if (T.x < 150.0)
        T = vec2(150, 0);
    if (T.x > 400.0)
        T = vec2(400, 0);

    vec2 tc = df_sub(T, vec2(273.14999389648438, 6.1035156022626325e-06));
    vec2 exponent =
        df_div(df_mul(df_sub(vec2(18.677999496459961, 5.0354003988672957e-07), df_div(tc, vec2(234.5, 0))), tc),
               df_add(vec2(257.1400146484375, -1.4648437513642421e-05), tc));
    vec2 es          = df_mul(vec2(611.21002197265625, -2.1972656213620212e-05), df_exp(exponent));
    vec2 denominator = df_sub(vec2(clamp(pressure, 100.0, 200000.0), 0), es);
    if (denominator.x < 1.0)
        denominator = vec2(1, 0);
    vec2 q = df_div(df_mul(vec2(0.62198001146316528, -1.1463165305869438e-08), es), denominator);
    return ml_greater(q, vec2(0.20000000298023224, -2.980232227667301e-09))
               ? vec2(0.20000000298023224, -2.980232227667301e-09)
               : q;
}
vec2 qsat_water(float T, float pressure)
{
    return milbrandt_qsat(vec2(T, 0), pressure);
}
