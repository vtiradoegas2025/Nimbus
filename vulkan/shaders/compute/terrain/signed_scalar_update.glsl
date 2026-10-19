float mapped_scalar_quotient(vec2 numerator, float denominator)
{
    float quotient = numerator.x / denominator;
    float remainder = fma(-quotient, denominator, numerator.x);
    float correction = fma(1.0, remainder, numerator.y) / denominator;
    return fma(1.0, quotient, correction);
}

float signed_mapped_scalar_update(float q, float density_begin, float denominator, float dt, float divergence)
{
    if (divergence == 0.0 && density_begin == denominator)
        return q;
    vec2 numerator = df_sub(df_mul(vec2(density_begin, 0.0), vec2(q, 0.0)),
                            df_mul(vec2(dt, 0.0), vec2(divergence, 0.0)));
    return mapped_scalar_quotient(numerator, denominator);
}

float positive_mapped_scalar_update(float q, float density_begin, float denominator,
                                    float dt, float incoming, float outgoing)
{
    if (incoming == outgoing && density_begin == denominator)
        return q;
    vec2 retained = df_sub(df_mul(vec2(density_begin, 0.0), vec2(q, 0.0)),
                           df_mul(vec2(dt, 0.0), vec2(outgoing, 0.0)));
    retained = df_add(retained, df_mul(vec2(dt, 0.0), vec2(incoming, 0.0)));
    return mapped_scalar_quotient(retained, denominator);
}
