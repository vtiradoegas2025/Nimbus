vec2 df_cbrt(vec2 value)
{
    if (value.x == 0.0 && value.y == 0.0)
    {
        return vec2(0.0);
    }
    float root     = pow(value.x, 1.0 / 3.0);
    vec2  square   = df_mul(vec2(root, 0.0), vec2(root, 0.0));
    vec2  residual = df_sub(value, df_mul(square, vec2(root, 0.0)));
    return df_add(vec2(root, 0.0), df_div(residual, df_mul(vec2(3, 0), square)));
}

float scalar(vec2 value)
{
    return value.x + value.y;
}

int nonnegative_floor(vec2 value)
{
    int result = int(max(0.0, value.x));
    if (value.x == float(result) && value.y < 0.0)
    {
        --result;
    }
    return max(0, result);
}

float stored_tendency(float initial, vec2 candidate)
{
    float rate = scalar(df_div(df_sub(candidate, vec2(initial, 0.0)), vec2(dt_high, dt_low)));
    if (rate < 0.0)
    {
        // Scale tiny donors before subtraction so a negative residual cannot underflow to zero.
        int exponent;
        frexp(max(initial, -rate), exponent);
        int shift = clamp(-exponent, 0, 100);
        vec2 remaining = df_sub(vec2(ldexp(initial, shift), 0.0),
                                df_mul(vec2(ldexp(-rate, shift), 0.0), vec2(dt_high, dt_low)));
        if (remaining.x < 0.0)
        {
            rate = uintBitsToFloat(floatBitsToUint(rate) - 1u);
        }
    }
    return rate;
}

vec2 normalize_cloud(vec2 mass, vec2 supplied)
{
    vec2 number = supplied;
    if (df_sub(number, vec2(2, 0)).x < 0.0)
    {
        number = vec2(2, 0);
    }
    if (df_sub(number, vec2(1999000064, -64)).x > 0.0)
    {
        number = vec2(1999000064, -64);
    }
    for (int iteration = 0; iteration < 16; ++iteration)
    {
        vec2 coordinate = df_div(vec2(1000000000, 0), number);
        int shape = coordinate.x >= 13.0 ? 15 : min(15, nonnegative_floor(df_add(coordinate, vec2(.5, 0))) + 2);
        vec2 ratio = vec2(float((shape + 1) * (shape + 2) * (shape + 3)), 0);
        vec2 coefficient = df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08),
                                  vec2(166.66667175292969, -5.0862630303072365e-06));
        vec2 slope = df_cbrt(df_div(df_mul(df_mul(number, coefficient), ratio), mass));
        vec2 diameter = df_div(vec2(float(shape + 4), 0), slope);
        vec2 minimum = vec2(9.9999999747524271e-07, 2.5247572468697606e-15);
        vec2 maximum = vec2(9.9999997473787516e-05, 2.5262125290942405e-12);
        if (df_sub(diameter, minimum).x >= 0.0 && df_sub(diameter, maximum).x <= 0.0)
            return number;
        vec2 bounded_diameter = df_sub(diameter, minimum).x < 0.0 ? minimum : maximum;
        slope = df_div(vec2(float(shape + 4), 0), bounded_diameter);
        vec2 next = df_div(df_mul(mass, df_mul(df_mul(slope, slope), slope)), df_mul(coefficient, ratio));
        if (df_sub(next, vec2(1999000064, -64)).x > 0.0)
            next = vec2(1999000064, -64);
        vec2 change = df_sub(next, number);
        if (change.x == 0.0 && change.y == 0.0)
            return number;
        number = next;
    }
    return number;
}

vec2 normalize_rain(vec2 mass, vec2 supplied)
{
    vec2 coefficient =
        df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08), vec2(166.66667175292969, -5.0862630303072365e-06));
    vec2 number = supplied;
    if (df_sub(number, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x <= 0.0)
    {
        vec2 slope = df_div(vec2(3.6719999313354492, 6.8664550934016688e-08),
                            vec2(0.0010000000474974513, -4.7497451284572723e-11));
        number     = df_div(df_mul(mass, df_mul(df_mul(slope, slope), slope)), df_mul(coefficient, vec2(6, 0)));
    }
    vec2 slope    = df_cbrt(df_div(df_mul(df_mul(coefficient, vec2(6, 0)), number), mass));
    vec2 diameter = df_div(vec2(3.6719999313354492, 6.8664550934016688e-08), slope);
    if (df_sub(diameter, vec2(0.0024999999441206455, 5.5879353227927808e-11)).x > 0.0)
    {
        diameter = vec2(0.0024999999441206455, 5.5879354528970415e-11);
    }
    else if (df_sub(diameter, vec2(3.7500001781154424e-05, -1.7811544340134988e-12)).x < 0.0)
    {
        diameter = vec2(3.7500001781154424e-05, -1.7811544272372352e-12);
    }
    slope = df_div(vec2(3.6719999313354492, 6.8664550934016688e-08), diameter);
    return df_div(df_mul(mass, df_mul(df_mul(slope, slope), slope)), df_mul(coefficient, vec2(6, 0)));
}
