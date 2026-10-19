#include "thompson_number_math.glsl"

vec2 liquid_saturation(int n)
{
    vec2 x = df_sub(vec2(temperature[n], 0.0), vec2(273.16000366210938, -3.6621093499888957e-06));
    if (x.x < -80.0)
    {
        x = vec2(-80, 0);
    }
    vec2 e     = vec2(-3.2158239423154059e-14, 1.2315406151497315e-22);
    e          = df_add(vec2(3.795342946172342e-12, 1.5382765800993744e-19), df_mul(x, e));
    e          = df_add(vec2(7.0262071716342689e-09, -1.9163426923402981e-16), df_mul(x, e));
    e          = df_add(vec2(2.0315417259553215e-06, 9.4044678576094996e-14), df_mul(x, e));
    e          = df_add(vec2(0.00029929107404313982, 6.956860163897377e-12), df_mul(x, e));
    e          = df_add(vec2(0.026422431692481041, 4.0751895954982409e-10), df_mul(x, e));
    e          = df_add(vec2(1.4317715167999268, 5.3200073235615264e-08), df_mul(x, e));
    e          = df_add(vec2(44.460689544677734, 5.532226765581072e-08), df_mul(x, e));
    e          = df_add(vec2(611.58367919921875, 1.9800781274170731e-05), df_mul(x, e));
    vec2 limit = df_mul(vec2(pressure[n], 0.0), vec2(0.15000000596046448, -5.9604644830901776e-09));
    if (df_sub(e, limit).x > 0.0)
    {
        e = limit;
    }
    return df_div(df_mul(vec2(0.62199997901916504, 2.0980834958272965e-08), e), df_sub(vec2(pressure[n], 0.0), e));
}
