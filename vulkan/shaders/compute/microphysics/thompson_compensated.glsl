// Retain low bits so saturation and table-bin decisions agree for nearly saturated FP32 states.
vec2 df_add(vec2 a, vec2 b)
{
    precise float sum  = a.x + b.x;
    precise float v    = sum - a.x;
    precise float tail = ((b.x - v) + (a.x - (sum - v))) + a.y + b.y;
    precise float high = sum + tail;
    return vec2(high, tail - (high - sum));
}
vec2 df_sub(vec2 a, vec2 b)
{
    return df_add(a, -b);
}
vec2 df_mul(vec2 a, vec2 b)
{
    precise float high = a.x * b.x;
    precise float tail = fma(a.x, b.x, -high) + a.x * b.y + a.y * b.x + a.y * b.y;
    return df_add(vec2(high, 0.0), vec2(tail, 0.0));
}
vec2 df_div(vec2 a, vec2 b)
{
    precise float quotient  = a.x / b.x;
    vec2          remainder = df_sub(a, df_mul(b, vec2(quotient, 0.0)));
    return df_add(vec2(quotient, 0.0), vec2((remainder.x + remainder.y) / b.x, 0.0));
}
vec2 df_exp(vec2 x)
{
    if (isnan(x.x) || x.x > 88.0)
    {
        return vec2(uintBitsToFloat(0x7fc00000u));
    }
    if (x.x < -103.0)
    {
        return vec2(0.0);
    }
    int  exponent = int(floor(x.x / 0.6931471805599453 + 0.5));
    vec2 reduced  = df_sub(x, df_mul(vec2(0.69314718246459961, -1.904654323148236e-09), vec2(float(exponent), 0.0)));
    vec2 term     = vec2(1.0, 0.0);
    vec2 sum      = term;
    for (int order = 1; order <= 13; ++order)
    {
        term = df_div(df_mul(term, reduced), vec2(float(order), 0.0));
        sum  = df_add(sum, term);
    }
    return df_mul(sum, vec2(exp2(float(exponent)), 0.0));
}
vec2 df_log(vec2 x)
{
    int  exponent = int(floor(log2(x.x)));
    vec2 scaled   = df_mul(x, vec2(exp2(-float(exponent)), 0.0));
    vec2 z        = df_div(df_sub(scaled, vec2(1, 0)), df_add(scaled, vec2(1, 0)));
    vec2 z2       = df_mul(z, z);
    vec2 term     = z;
    vec2 sum      = term;
    for (int order = 3; order <= 33; order += 2)
    {
        term = df_mul(term, z2);
        sum  = df_add(sum, df_div(term, vec2(float(order), 0.0)));
    }
    return df_add(df_mul(vec2(2, 0), sum),
                  df_mul(vec2(0.69314718246459961, -1.904654323148236e-09), vec2(float(exponent), 0.0)));
}
vec2 df_sqrt(vec2 x)
{
    float root      = sqrt(x.x);
    vec2  remainder = df_sub(x, df_mul(vec2(root, 0.0), vec2(root, 0.0)));
    return df_add(vec2(root, 0.0), df_div(remainder, vec2(2.0 * root, 0.0)));
}
